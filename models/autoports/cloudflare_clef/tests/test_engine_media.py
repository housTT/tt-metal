import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)
os.environ.setdefault("CLEF_MODEL", SNAPSHOT)
os.environ.setdefault("HF_MODEL", SNAPSHOT)

from models.autoports.cloudflare_clef.scripts.clef_paths import read_jsonl
from models.autoports.cloudflare_clef.tt.engine import ClefEngine, tp2_mesh

REFERENCE_DIR = Path("/home/hous/dev/clef/reports/reference")
IMAGE_RECORDS = REFERENCE_DIR / "records_image.jsonl"
VIDEO_RECORDS = REFERENCE_DIR / "records_video.jsonl"
TWO_IMAGE_RECORDS = REFERENCE_DIR / "records_two_image.jsonl"
TWO_IMAGE_RECORD = "two_image_2a7ddcfe_6900e8fa"
TEXT_RECORDS = REFERENCE_DIR / "records_text.jsonl"
N_LAYERS = int(os.environ.get("CLEF_MEDIA_TEST_LAYERS", "4"))
REPORT = Path(f"/home/hous/dev/clef/reports/stage2_engine_media_l{N_LAYERS}.json")
RESULTS = {"cases": {}}
IMAGE_A = "2a7ddcfe4724ee1403a6291d21347162"
IMAGE_B = "6900e8fa480aa890209946c139b55693"


def write_report():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(RESULTS, indent=2, default=str))


def max_dp(a, b):
    return max(abs(a[qid][o] - b[qid][o]) for qid in a for o in a[qid])


def by_id(path, record_id):
    return next(row for row in read_jsonl(path) if row["id"] == record_id)


@pytest.fixture(scope="module")
def submesh():
    with tp2_mesh(os.environ.get("CLEF_PARENT", "1x4")) as sub:
        yield sub


@pytest.fixture(scope="module")
def engine(submesh):
    t0 = time.perf_counter()
    engine = ClefEngine(submesh, n_layers=N_LAYERS, snapshot_slots=4)
    RESULTS["engine"] = dict(
        n_layers=engine.args.n_layers,
        load_seconds=round(time.perf_counter() - t0, 1),
        timings=engine.timings,
        vision=engine.vision.describe(),
        dram_free_after_text_weights_bytes=engine.dram_free_after_text_weights,
        dram_free_after_weights_bytes=engine.dram_free_after_weights,
        dram_free_after_slots_bytes=engine.dram_free_after_slots,
        snapshot_slots=engine.snapshot_slots,
    )
    write_report()
    return engine


def cached_vs_full(engine, record, other, name):
    full = engine.probs_for_request(record, mode="full", slot=0)
    miss = engine.probs_for_request(record, mode="cached", slot=3)
    hit = engine.probs_for_request(record, mode="cached", slot=3)
    assert miss["cache_hit"] is False and hit["cache_hit"] is True
    other_full = engine.probs_for_request(other, mode="full", slot=1)
    text = engine.probs_for_request(by_id(TEXT_RECORDS, "readme_checkout"), mode="cached", slot=2)
    after_other = engine.probs_for_request(record, mode="cached", slot=3)
    assert after_other["cache_hit"] is True
    rope_delta = int(engine.model.rope.rope_delta)
    placeholder = int(engine.model._vision_placeholder_token_id())
    state_ids = engine.handles[3].state_ids[0]
    positions = torch.nonzero(state_ids == placeholder).reshape(-1)
    row = dict(
        record=record["id"],
        input_tokens=full["input_tokens"],
        grid=full["vision"]["grid"],
        media_rows=full["vision"]["n_rows"],
        media_patches=full["vision"]["n_patches"],
        padded_rows=full["vision"]["tower"]["rows"],
        window=full["vision"]["tower"].get("window"),
        state_S=engine.handles[3].S,
        state_S0=engine.handles[3].S0,
        media_first_position=int(positions[0]),
        media_last_position=int(positions[-1]),
        media_rows_in_cached_prefix=int((positions < engine.handles[3].S0).sum()),
        rope_delta_after_cached_schema=rope_delta,
        full_probs=full["probs"],
        cached_miss_max_dp=max_dp(full["probs"], miss["probs"]),
        cached_hit_max_dp=max_dp(full["probs"], hit["probs"]),
        cached_after_other_media_max_dp=max_dp(full["probs"], after_other["probs"]),
        other_record=other["id"],
        other_full_rows=other_full["vision"]["n_rows"],
        text_cached_hit=text["cache_hit"],
        vision_seconds_full=full["timing"]["vision_s"],
        device_seconds_full=full["timing"]["device_s"],
        device_seconds_cached_hit=hit["timing"]["device_s"],
        seconds_full=full["seconds"],
        seconds_cached_hit=hit["seconds"],
    )
    RESULTS["cases"][name] = row
    write_report()
    logger.info(f"{name}: {row}")
    assert full["probs"] == miss["probs"], f"cached miss differs from full by {row['cached_miss_max_dp']}"
    assert full["probs"] == hit["probs"], f"cached hit differs from full by {row['cached_hit_max_dp']}"
    assert (
        full["probs"] == after_other["probs"]
    ), f"cached hit after another media request differs from full by {row['cached_after_other_media_max_dp']}"
    assert rope_delta != 0, "schema continuation of a media state must run with the request's M-RoPE table"
    assert int(positions.numel()) == row["media_rows"]


@pytest.mark.timeout(1500)
def test_device_image_cached_matches_full(engine):
    cached_vs_full(engine, by_id(IMAGE_RECORDS, IMAGE_A), by_id(IMAGE_RECORDS, IMAGE_B), "image")


@pytest.mark.timeout(1500)
def test_device_video_cached_matches_full(engine):
    if not VIDEO_RECORDS.exists():
        pytest.skip(f"missing {VIDEO_RECORDS}")
    cached_vs_full(engine, by_id(VIDEO_RECORDS, f"video_{IMAGE_A}"), by_id(IMAGE_RECORDS, IMAGE_B), "video")


@pytest.mark.timeout(1500)
def test_device_two_image_cached_matches_full(engine):
    if not TWO_IMAGE_RECORDS.exists():
        pytest.skip(f"missing {TWO_IMAGE_RECORDS}")
    record = by_id(TWO_IMAGE_RECORDS, TWO_IMAGE_RECORD)
    single_a = engine.probs_for_request(by_id(IMAGE_RECORDS, IMAGE_A), mode="full", slot=0)
    single_b = engine.probs_for_request(by_id(IMAGE_RECORDS, IMAGE_B), mode="full", slot=0)
    cached_vs_full(engine, record, by_id(IMAGE_RECORDS, IMAGE_B), "two_image")
    row = RESULTS["cases"]["two_image"]
    row["single_image_rows"] = [single_a["vision"]["n_rows"], single_b["vision"]["n_rows"]]
    row["single_image_probs"] = {IMAGE_A: single_a["probs"], IMAGE_B: single_b["probs"]}
    write_report()
    assert row["media_rows"] == single_a["vision"]["n_rows"] + single_b["vision"]["n_rows"]
    assert len(row["full_probs"]) == 2


@pytest.mark.timeout(1500)
def test_device_media_rejected_without_tower(engine, expect_error):
    record = by_id(IMAGE_RECORDS, IMAGE_A)
    tower = engine.vision
    engine.vision = None
    try:
        with expect_error(ValueError, "without the vision tower"):
            engine.probs_for_request(record, mode="full", slot=0)
    finally:
        engine.vision = tower

import json
import os
import statistics
import subprocess
import sys
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

from models.autoports.cloudflare_clef.scripts import reference_rows
from models.autoports.cloudflare_clef.scripts.clef_paths import read_jsonl, write_jsonl
from models.autoports.cloudflare_clef.tt import encode as clef_encode
from models.autoports.cloudflare_clef.tt.engine import BUCKETS, REFERENCE_GRIDS, ClefEngine, tp2_mesh

os.environ.setdefault("CLEF_VISION_WARM_GRID", REFERENCE_GRIDS)

REFERENCE_DIR = Path("/home/hous/dev/clef/reports/reference")
RECORDS = REFERENCE_DIR / "records_text.jsonl"
IMAGE_RECORDS = REFERENCE_DIR / "records_image.jsonl"
VIDEO_RECORDS = REFERENCE_DIR / "records_video.jsonl"
LONG_RECORDS = REFERENCE_DIR / "long_records.jsonl"
REFERENCE = REFERENCE_DIR / "ref_text_bf16.jsonl"
IMAGE_REFERENCE = REFERENCE_DIR / "ref_image_bf16.jsonl"
PARITY_COMPARE = Path(__file__).resolve().parent.parent / "scripts" / "parity_compare.py"
REPORT_DIR = Path("/home/hous/dev/clef/reports")
N_LAYERS = int(os.environ.get("CLEF_N_LAYERS", "4"))
TRACE_REGION = int(os.environ.get("CLEF_TRACE_REGION", str(1 << 30)))
MAX_STATE_LEN = 16384
TRACED_VS_EAGER_MAX_DP = 1e-3
TAIL_LENGTHS = (50, 200, 450, 1000)
COMPARE_EAGER = os.environ.get("CLEF_TRACED_TEST_EAGER", "1") == "1"
RESULTS = {"n_layers": N_LAYERS, "trace_region_size": TRACE_REGION}
_engine = {}


def tag():
    return os.environ.get("CLEF_REPORT_TAG", "")


def report_path():
    return REPORT_DIR / f"stage3_traced_l{N_LAYERS}{tag()}.json"


def write_report():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path().write_text(json.dumps(RESULTS, indent=2, default=str))


def pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def max_dp(rows_a, rows_b):
    worst = 0.0
    for ra, rb in zip(rows_a, rows_b):
        assert ra["id"] == rb["id"]
        assert "error" not in ra, ra
        assert "error" not in rb, rb
        for qid, dist in ra["probs"].items():
            for option, p in dist.items():
                worst = max(worst, abs(p - rb["probs"][qid][option]))
    return worst


@pytest.fixture(scope="module")
def submesh():
    with tp2_mesh(os.environ.get("CLEF_PARENT", "1x4"), trace_region_size=TRACE_REGION) as sub:
        yield sub


@pytest.fixture(scope="module")
def engine(submesh):
    if "engine" not in _engine:
        if "error" in _engine:
            pytest.fail(f"engine build failed earlier: {_engine['error']}")
        try:
            _engine["engine"] = ClefEngine(
                submesh, max_state_len=MAX_STATE_LEN, n_layers=N_LAYERS, snapshot_slots=4, traced=True
            )
        except Exception as error:
            _engine["error"] = repr(error)
            raise
        e = _engine["engine"]
        RESULTS["engine"] = dict(
            n_layers=e.args.n_layers,
            precision=e.precision,
            device_dtypes=e.device_dtypes,
            timings=e.timings,
            traces=len(e.traces),
            trace_keys=[str(k) for k in e.traces],
            trace_bytes=e.trace_bytes,
            trace_mib=round(e.trace_bytes / 2**20, 1),
            trace_guard=getattr(e, "trace_guard", None),
            snapshot_slots=e.snapshot_slots,
            spare_blocks=e.spare_blocks,
            dram_free_after_weights_gib=round(e.dram_free_after_weights / 2**30, 3),
            dram_free_after_slots_gib=round(e.dram_free_after_slots / 2**30, 3),
            dram_free_after_traces_gib=round(e.dram_free_after_traces / 2**30, 3),
        )
        write_report()
    return _engine["engine"]


@pytest.fixture(scope="module")
def tokenizer(engine):
    return engine.tokenizer


def run_rows(engine, rows, mode, slot, traced):
    engine.traced = traced
    try:
        if traced:
            return reference_rows.run_rows(engine, rows, mode=mode, slot=slot)
        with engine._misses_allowed():
            return reference_rows.run_rows(engine, rows, mode=mode, slot=slot)
    finally:
        engine.traced = True


def compare(candidate, reference, name):
    out_json = REPORT_DIR / f"{name}.json"
    out_md = REPORT_DIR / f"{name}.md"
    cmd = [
        sys.executable,
        str(PARITY_COMPARE),
        "--reference",
        str(reference),
        "--candidate",
        str(candidate),
        "--margin",
        "0.05",
        "--name",
        name,
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(f"parity_compare {name}: rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return json.loads(out_json.read_text()) if out_json.exists() else {"rc": proc.returncode}


def test_setup(engine):
    expected = len(BUCKETS) + 2 * engine.snapshot_slots + 1
    assert len(engine.traces) == expected, (len(engine.traces), expected)
    assert engine.trace_bytes > 0
    assert engine.trace_bytes <= TRACE_REGION
    guard = engine.trace_guard
    assert guard["probe_address_after_release"] < guard["address"] + guard["bytes"], guard
    logger.info(f"traces={len(engine.traces)} region_used={engine.trace_bytes / 2**20:.1f} MiB guard={guard}")


def test_text_records(engine):
    rows = read_jsonl(RECORDS)
    t = time.perf_counter()
    traced_full = run_rows(engine, rows, "full", 0, traced=True)
    traced_full_s = time.perf_counter() - t
    t = time.perf_counter()
    traced_cached = run_rows(engine, rows, "cached", 1, traced=True)
    traced_cached_s = time.perf_counter() - t
    eager_full, eager_full_s = None, None
    if COMPARE_EAGER:
        t = time.perf_counter()
        eager_full = run_rows(engine, rows, "full", 2, traced=False)
        eager_full_s = time.perf_counter() - t
    traced_cached_again = run_rows(engine, rows, "cached", 1, traced=True)
    traced_full_again = run_rows(engine, rows, "full", 3, traced=True)
    base = REPORT_DIR / f"stage3_tt_text_l{N_LAYERS}{tag()}"
    write_jsonl(f"{base}_traced_full.jsonl", traced_full)
    write_jsonl(f"{base}_traced_cached.jsonl", traced_cached)
    dp_cached_vs_full = max_dp(traced_cached, traced_full)
    dp_repeat = max_dp(traced_cached_again, traced_cached)
    dp_full_repeat = max_dp(traced_full_again, traced_full)
    RESULTS["text_records"] = dict(
        records=len(rows),
        compare_eager=COMPARE_EAGER,
        traced_cached_vs_traced_full_max_dp=dp_cached_vs_full,
        traced_cached_repeat_max_dp=dp_repeat,
        traced_full_repeat_max_dp=dp_full_repeat,
        seconds=dict(traced_full=traced_full_s, traced_cached=traced_cached_s, eager_full=eager_full_s),
        device_s=dict(
            traced_full=[r["timing"]["device_s"] for r in traced_full],
            traced_cached_hit=[r["timing"]["device_s"] for r in traced_cached_again],
        ),
        input_tokens=[r["input_tokens"] for r in traced_full],
        parity_traced_full_vs_bf16=compare(
            f"{base}_traced_full.jsonl", REFERENCE, f"stage3_parity_l{N_LAYERS}{tag()}_traced_full"
        ),
        counters=dict(engine.counters),
    )
    if COMPARE_EAGER:
        write_jsonl(f"{base}_eager_full.jsonl", eager_full)
        RESULTS["text_records"]["traced_full_vs_eager_full_max_dp"] = max_dp(traced_full, eager_full)
        RESULTS["text_records"]["device_s"]["eager_full"] = [r["timing"]["device_s"] for r in eager_full]
        RESULTS["text_records"]["parity_eager_full_vs_bf16"] = compare(
            f"{base}_eager_full.jsonl", REFERENCE, f"stage3_parity_l{N_LAYERS}{tag()}_eager_full"
        )
    write_report()
    logger.info(json.dumps({k: v for k, v in RESULTS["text_records"].items() if "parity" not in k}, default=str))
    assert dp_repeat == 0.0
    assert dp_full_repeat == 0.0
    if COMPARE_EAGER:
        dp_traced_vs_eager = RESULTS["text_records"]["traced_full_vs_eager_full_max_dp"]
        assert dp_traced_vs_eager <= TRACED_VS_EAGER_MAX_DP, dp_traced_vs_eager


def test_tail_buckets(engine, tokenizer):
    rows = read_jsonl(RECORDS)
    record = next(r for r in rows if r["id"] == "readme_invoice")
    encoded = clef_encode.encode(tokenizer, record)
    state_part, tail_part, _ = clef_encode.split_for_cache(encoded, tokenizer, record)
    state_ids = torch.tensor([state_part], dtype=torch.long)
    results = {}
    engine.traced = True
    handle_t = engine.prefill_state(state_ids, slot=0, key="tail-traced")
    handle_e = None
    if COMPARE_EAGER:
        with engine._misses_allowed():
            engine.traced = False
            handle_e = engine.prefill_state(state_ids, slot=1, key="tail-eager")
            engine.traced = True
    for L in TAIL_LENGTHS:
        tail = torch.tensor([(tail_part * ((L // len(tail_part)) + 1))[:L]], dtype=torch.long)
        engine.traced = True
        t = time.perf_counter()
        traced_rows = engine.schema_hidden(handle_t, tail)
        traced_s = time.perf_counter() - t
        traced_rows2 = engine.schema_hidden(handle_t, tail)
        total = len(handle_t.suffix_ids[0]) + L
        results[L] = dict(
            chunks=[engine.bucket_for(min(1024, total - cs)) for cs in range(0, total, 1024)],
            rows=int(traced_rows.shape[0]),
            repeat_bit_equal=bool(torch.equal(traced_rows, traced_rows2)),
            traced_s=round(traced_s, 4),
        )
        if COMPARE_EAGER:
            with engine._misses_allowed():
                engine.traced = False
                t = time.perf_counter()
                eager_rows = engine.schema_hidden(handle_e, tail)
                eager_s = time.perf_counter() - t
                engine.traced = True
            results[L].update(
                pcc_traced_vs_eager=pcc(traced_rows, eager_rows),
                max_abs_traced_vs_eager=float((traced_rows - eager_rows).abs().max()),
                eager_s=round(eager_s, 4),
            )
        logger.info(f"tail L={L}: {results[L]}")
    RESULTS["tail_buckets"] = dict(state_tokens=int(state_ids.shape[1]), S0=handle_t.S0, tails=results)
    write_report()
    for L, r in results.items():
        assert r["repeat_bit_equal"], (L, r)
        if COMPARE_EAGER:
            assert r["pcc_traced_vs_eager"] >= 0.999, (L, r)


def test_media_records(engine):
    image_rows = read_jsonl(IMAGE_RECORDS)
    video_rows = read_jsonl(VIDEO_RECORDS)
    traced_img = run_rows(engine, image_rows, "full", 0, traced=True)
    traced_img_cached = run_rows(engine, image_rows, "cached", 1, traced=True)
    eager_img = run_rows(engine, image_rows, "full", 2, traced=False) if COMPARE_EAGER else None
    traced_vid = run_rows(engine, video_rows, "cached", 3, traced=True)
    traced_vid_hit = run_rows(engine, video_rows, "cached", 3, traced=True)
    eager_vid = run_rows(engine, video_rows, "full", 2, traced=False) if COMPARE_EAGER else None
    traced_img_after = run_rows(engine, image_rows, "full", 0, traced=True)
    text_after = run_rows(engine, read_jsonl(RECORDS)[:3], "cached", 1, traced=True)
    base = REPORT_DIR / f"stage3_tt_image_l{N_LAYERS}{tag()}"
    write_jsonl(f"{base}_traced_full.jsonl", traced_img)
    write_jsonl(f"{base}_traced_cached.jsonl", traced_img_cached)
    write_jsonl(f"{REPORT_DIR}/stage3_tt_video_l{N_LAYERS}{tag()}_traced_cached.jsonl", traced_vid)
    RESULTS["media_records"] = dict(
        image_records=len(image_rows),
        compare_eager=COMPARE_EAGER,
        image_traced_cached_vs_traced_full_max_dp=max_dp(traced_img_cached, traced_img),
        image_traced_full_repeat_max_dp=max_dp(traced_img_after, traced_img),
        video_cached_hit_repeat_max_dp=max_dp(traced_vid_hit, traced_vid),
        video_hit=traced_vid_hit[0]["cache_hit"],
        text_after_media_vs_before_max_dp=max_dp(
            text_after, read_jsonl(f"{REPORT_DIR}/stage3_tt_text_l{N_LAYERS}{tag()}_traced_cached.jsonl")[:3]
        ),
        vision_seconds_traced=[r["timing"].get("vision_s") for r in traced_img],
        device_s_traced=[r["timing"]["device_s"] for r in traced_img],
        grids=[r.get("vision", {}).get("grid") for r in traced_img],
        parity_image_traced_full_vs_bf16=compare(
            f"{base}_traced_full.jsonl", IMAGE_REFERENCE, f"stage3_parity_image_l{N_LAYERS}{tag()}_traced_full"
        ),
        counters=dict(engine.counters),
        program_cache_entries=int(engine.mesh.num_program_cache_entries()),
    )
    if COMPARE_EAGER:
        write_jsonl(f"{base}_eager_full.jsonl", eager_img)
        RESULTS["media_records"]["image_traced_full_vs_eager_full_max_dp"] = max_dp(traced_img, eager_img)
        RESULTS["media_records"]["video_traced_cached_vs_eager_full_max_dp"] = max_dp(traced_vid, eager_vid)
        RESULTS["media_records"]["device_s_eager"] = [r["timing"]["device_s"] for r in eager_img]
    write_report()
    logger.info(json.dumps({k: v for k, v in RESULTS["media_records"].items() if "parity" not in k}, default=str))
    assert traced_vid_hit[0]["cache_hit"] is True
    assert RESULTS["media_records"]["video_cached_hit_repeat_max_dp"] == 0.0
    assert RESULTS["media_records"]["image_traced_full_repeat_max_dp"] == 0.0
    assert RESULTS["media_records"]["text_after_media_vs_before_max_dp"] == 0.0
    if COMPARE_EAGER:
        assert RESULTS["media_records"]["image_traced_full_vs_eager_full_max_dp"] <= TRACED_VS_EAGER_MAX_DP
        assert RESULTS["media_records"]["video_traced_cached_vs_eager_full_max_dp"] <= TRACED_VS_EAGER_MAX_DP


def test_unseen_grid_refused(engine, expect_error):
    if engine.vision is None:
        pytest.skip("engine built without the vision tower")
    warmed = {tuple(g) for g, _ in engine.vision_warmed_grids}
    grid = (1, 20, 24)
    assert grid not in warmed, warmed
    media = {
        "pixel_values": torch.zeros(int(torch.tensor(grid).prod()), 3 * 2 * 16 * 16),
        "image_grid_thw": torch.tensor([list(grid)]),
    }
    tower_runs = engine.counters["tower_runs"]
    entries = int(engine.mesh.num_program_cache_entries())
    engine.traced = True
    with expect_error(ValueError, "not in this traced server's warm list"):
        engine._vision_request(media)
    RESULTS["unseen_grid"] = dict(
        grid=list(grid),
        warmed=sorted(warmed),
        tower_runs_unchanged=engine.counters["tower_runs"] == tower_runs,
        program_cache_entries_unchanged=int(engine.mesh.num_program_cache_entries()) == entries,
    )
    write_report()
    assert RESULTS["unseen_grid"]["tower_runs_unchanged"]
    assert RESULTS["unseen_grid"]["program_cache_entries_unchanged"]


def test_two_grid_request_refused(engine, expect_error):
    if engine.vision is None:
        pytest.skip("engine built without the vision tower")
    warmed = sorted(tuple(g) for g, _ in engine.vision_warmed_grids)
    grids = [warmed[0], warmed[1 % len(warmed)]]
    n_patches = sum(int(torch.tensor(g).prod()) for g in grids)
    media = {
        "pixel_values": torch.zeros(n_patches, 3 * 2 * 16 * 16),
        "image_grid_thw": torch.tensor([list(g) for g in grids]),
    }
    tower_runs = engine.counters["tower_runs"]
    replays = engine.counters["replays"]
    entries = int(engine.mesh.num_program_cache_entries())
    engine.traced = True
    with expect_error(ValueError, "one grid per request"):
        engine._vision_request(media)
    entries_after_refusal = int(engine.mesh.num_program_cache_entries())
    tower_runs_after_refusal = engine.counters["tower_runs"]
    ids = torch.randint(1000, 100000, (1, 128), generator=torch.Generator().manual_seed(0), dtype=torch.long)
    engine.prefill_hidden(ids, slot=0)
    RESULTS["two_grid"] = dict(
        grids=[list(g) for g in grids],
        n_patches=n_patches,
        tower_runs_unchanged=tower_runs_after_refusal == tower_runs,
        program_cache_entries=entries,
        program_cache_entries_unchanged=entries_after_refusal == entries,
        replays_after_refusal=engine.counters["replays"] - replays,
        program_cache_entries_after_replay=int(engine.mesh.num_program_cache_entries()),
    )
    write_report()
    logger.info(json.dumps(RESULTS["two_grid"]))
    assert RESULTS["two_grid"]["tower_runs_unchanged"]
    assert RESULTS["two_grid"]["program_cache_entries_unchanged"]
    assert RESULTS["two_grid"]["replays_after_refusal"] >= 1
    assert RESULTS["two_grid"]["program_cache_entries_after_replay"] == entries


def test_long_state(engine):
    rows = [r for r in read_jsonl(LONG_RECORDS) if r["id"] in ("long/2000", "long/max")]
    traced = [run_rows(engine, [row], "cached", i, traced=True)[0] for i, row in enumerate(rows)]
    traced_hit = [run_rows(engine, [row], "cached", i, traced=True)[0] for i, row in enumerate(rows)]
    eager = run_rows(engine, rows, "full", len(rows), traced=False) if COMPARE_EAGER else None
    RESULTS["long_state"] = dict(
        records=[r["id"] for r in rows],
        compare_eager=COMPARE_EAGER,
        input_tokens=[r["input_tokens"] for r in traced],
        traced_hit_repeat_max_dp=max_dp(traced_hit, traced),
        device_s_traced_miss=[r["timing"]["device_s"] for r in traced],
        device_s_traced_hit=[r["timing"]["device_s"] for r in traced_hit],
        hits=[r["cache_hit"] for r in traced_hit],
    )
    if COMPARE_EAGER:
        RESULTS["long_state"]["traced_cached_vs_eager_full_max_dp"] = max_dp(traced, eager)
        RESULTS["long_state"]["device_s_eager"] = [r["timing"]["device_s"] for r in eager]
    write_report()
    logger.info(json.dumps(RESULTS["long_state"], default=str))
    assert all(RESULTS["long_state"]["hits"])
    assert RESULTS["long_state"]["traced_hit_repeat_max_dp"] == 0.0
    if COMPARE_EAGER:
        assert RESULTS["long_state"]["traced_cached_vs_eager_full_max_dp"] <= 1e-2


def test_timing(engine, tokenizer):
    g = torch.Generator().manual_seed(0)

    def ids(n):
        return torch.randint(1000, 100000, (1, n), generator=g, dtype=torch.long)

    def median(fn, reps=5):
        fn()
        samples = []
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t)
        return round(statistics.median(samples), 4)

    timing = {"buckets": {}, "state": {}}
    for b in BUCKETS:
        x = ids(b)
        engine.traced = True
        traced_s = median(lambda: engine.prefill_hidden(x, slot=0))
        with engine._misses_allowed():
            engine.traced = False
            eager_s = median(lambda: engine.prefill_hidden(x, slot=0))
            engine.traced = True
        timing["buckets"][b] = dict(traced_s=traced_s, eager_s=eager_s)
        logger.info(f"bucket {b}: traced {traced_s:.4f} s, eager {eager_s:.4f} s")
    for S in (2200, 8192):
        x = ids(S)
        schema = ids(198)
        engine.traced = True
        traced_prefill = median(lambda: engine.prefill_state(x, slot=0, key=f"t{S}"), reps=3)
        handle = engine.prefill_state(x, slot=0, key=f"t{S}")
        traced_schema = median(lambda: engine.schema_hidden(handle, schema))
        with engine._misses_allowed():
            engine.traced = False
            eager_prefill = median(lambda: engine.prefill_state(x, slot=1, key=f"e{S}"), reps=3)
            handle_e = engine.prefill_state(x, slot=1, key=f"e{S}")
            eager_schema = median(lambda: engine.schema_hidden(handle_e, schema))
            engine.traced = True
        timing["state"][S] = dict(
            traced_prefill_s=traced_prefill,
            traced_schema_198_s=traced_schema,
            eager_prefill_s=eager_prefill,
            eager_schema_198_s=eager_schema,
        )
        logger.info(f"state {S}: {timing['state'][S]}")
    timing["counters"] = dict(engine.counters)
    RESULTS["timing"] = timing
    write_report()


def test_exact_buckets(engine):
    if not COMPARE_EAGER:
        pytest.skip("eager comparison disabled (CLEF_TRACED_TEST_EAGER=0)")
    g = torch.Generator().manual_seed(1)
    cases = {}
    for b in BUCKETS:
        for length in (b, b - 28):
            ids = torch.randint(1000, 100000, (1, length), generator=g, dtype=torch.long)
            engine.traced = True
            traced = engine.prefill_hidden(ids, slot=0)
            traced2 = engine.prefill_hidden(ids, slot=0)
            with engine._misses_allowed():
                engine.traced = False
                eager = engine.prefill_hidden(ids, slot=1)
                engine.traced = True
            diff = (traced - eager).abs()
            rows_differ = int((diff.max(dim=1).values > 0).sum())
            cases[f"{b}:{length}"] = dict(
                bucket=b,
                length=length,
                padded=length != b,
                bit_equal=bool(torch.equal(traced, eager)),
                repeat_bit_equal=bool(torch.equal(traced, traced2)),
                max_abs=float(diff.max()),
                mean_abs=float(diff.mean()),
                rows_differ=rows_differ,
                first_row_differ=int((diff.max(dim=1).values > 0).nonzero().min()) if rows_differ else None,
                pcc=pcc(traced, eager),
                hidden_abs_max=float(eager.abs().max()),
            )
            logger.info(f"exact bucket {b} length {length}: {cases[f'{b}:{length}']}")
    RESULTS["exact_buckets"] = cases
    write_report()
    for key, c in cases.items():
        assert c["repeat_bit_equal"], (key, c)

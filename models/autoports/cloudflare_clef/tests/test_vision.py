import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.cloudflare_clef.tt import vision as clef_vision
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)
REF_DIR = Path("/home/hous/dev/clef/reports/reference/vision_ref")
REPORT = Path(os.environ.get("CLEF_VISION_REPORT", "/home/hous/dev/clef/reports/stage2_vision_tower.json"))
PARENTS = {"1x4": ("FABRIC_1D", (1, 4)), "2x2": ("FABRIC_2D", (2, 2))}
RECORDS = [
    "2a7ddcfe4724ee1403a6291d21347162",
    "6900e8fa480aa890209946c139b55693",
    "e000b5c0a8358750424503bd92a9a656",
    "848089ea8a254944feaa3c3456e331d5",
]
VIDEO_RECORD = "video_2a7ddcfe4724ee1403a6291d21347162"
SENSITIVE_RECORDS = ["7b7ef38338fffa28b9106027c882c1b4"]
ROWS_DIR = Path(os.environ.get("CLEF_VISION_ROWS_DIR", "/home/hous/dev/clef/reports/stage2r_tower_rows"))
UPSTREAM_ROWS = Path("/home/hous/dev/clef/reports/stage2_device_vision_rows") / f"{RECORDS[0]}.pt"
BASE_PCC_80 = 0.98901
PCC_END_TO_END = 0.97
BLOCK_BARS = {index: (0.99 if index < 24 else 0.85) for index in clef_vision.BLOCK_TAPS}
TP = 2
RESULTS = {"cases": {}}

os.environ.setdefault("CLEF_MODEL", SNAPSHOT)
os.environ.setdefault("HF_MODEL", SNAPSHOT)


def write_report():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(RESULTS, indent=2, default=str))


def pcc(golden, candidate):
    _, value = comp_pcc(golden.float(), candidate.float(), 0.0)
    return round(float(value), 6)


def vision_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(SNAPSHOT).vision_config


@pytest.mark.eager_host_side
def test_host_key_mapping_matches_upstream_modules():
    config = vision_config()
    state_dict = clef_vision.read_vision_state_dict(SNAPSHOT)
    assert len(state_dict) == 333
    head_dim = config.hidden_size // config.num_heads
    mapped = clef_vision.tt_vision_state_dict(state_dict, head_dim)
    expected = clef_vision.expected_tt_keys(config.depth)
    assert expected <= set(mapped)
    assert set(mapped) - expected == clef_vision.HOST_ONLY_KEYS
    assert not any(".q_norm." in key or ".k_norm." in key for key in mapped)
    assert tuple(mapped["visual.blocks.0.attention.wq.weight"].shape) == (config.hidden_size, config.hidden_size)
    assert tuple(mapped["visual.blocks.0.feed_forward.linear_fc1.weight"].shape) == (
        config.intermediate_size,
        config.hidden_size,
    )
    assert tuple(mapped["visual.merger.linear_fc2.weight"].shape) == (
        config.out_hidden_size,
        config.hidden_size * config.spatial_merge_size**2,
    )
    assert torch.equal(
        mapped["visual.blocks.0.attention.wv.weight"], state_dict["blocks.0.attn.qkv.weight"][2 * config.hidden_size :]
    )


@pytest.mark.eager_host_side
def test_host_clef_vision_config_fits_tp2():
    config = vision_config()
    report = clef_vision.vision_shape_report(config, TP)
    assert report["depth"] == 27 and report["dim"] == 1152 and report["n_heads"] == 16
    assert report["head_dim"] == 72 and report["padded_head_dim"] == 96
    assert report["hidden_dim"] == 4352 and report["out_hidden_size"] == 5120
    assert report["deepstack_visual_indexes"] == []
    assert all(report["divisible"].values()), report["divisible"]
    assert clef_vision.padded_rows(320, 2048) == 2048
    assert clef_vision.padded_rows(2000, 2048) == 2048
    assert clef_vision.padded_rows(2049, 2048) == 4096
    assert clef_vision.padded_rows(320, 128) == 384
    assert clef_vision.padded_rows(1064, 128) == 2048
    for rows in (clef_vision.padded_rows(n, 128) for n in range(1, 4200, 37)):
        assert rows % 128 == 0
        assert rows <= 1024 or rows % 1024 == 0
        assert rows <= 2048 or rows % 2048 == 0


@pytest.mark.eager_host_side
def test_host_hf_visual_loads_standalone():
    visual = clef_vision.load_hf_visual(SNAPSHOT)
    assert visual.dtype == torch.bfloat16
    assert visual.rotary_pos_emb.inv_freq.dtype == torch.float32
    assert visual.config._attn_implementation == "sdpa"
    assert len(visual.blocks) == 27
    assert sum(p.numel() for p in visual.parameters()) == 460730096


@pytest.mark.eager_host_side
def test_host_reference_index_lists_the_test_images():
    index = json.loads((REF_DIR / "index.json").read_text())
    by_id = {row["id"]: row for row in index["records"]}
    for record_id in RECORDS:
        assert record_id in by_id, record_id
        assert Path(by_id[record_id]["file"]).exists()
        assert by_id[record_id]["placeholders"] == by_id[record_id]["n_tokens"]
    assert index["taps"] == list(clef_vision.BLOCK_TAPS)


@pytest.fixture(scope="module")
def submesh():
    name = os.environ.get("CLEF_PARENT", "1x4")
    fabric_name, shape = PARENTS[name]
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fabric_name))
    parent = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*shape),
        l1_small_size=GDN_CONV1D_L1_SMALL_SIZE,
        num_command_queues=2,
        trace_region_size=int(os.environ.get("CLEF_TRACE_REGION", "0")),
    )
    sub = parent.create_submesh(ttnn.MeshShape(1, 2), offset=ttnn.MeshCoordinate(0, 0))
    if hasattr(sub, "enable_program_cache"):
        sub.enable_program_cache()
    RESULTS["mesh"] = dict(
        parent=name,
        fabric=fabric_name,
        parent_shape=list(parent.shape),
        parent_device_ids=list(parent.get_device_ids()),
        submesh_shape=list(sub.shape),
        submesh_device_ids=list(sub.get_device_ids()),
        cluster_type=str(ttnn.cluster.get_cluster_type()),
    )
    logger.info(f"mesh: {RESULTS['mesh']}")
    yield sub
    for child in parent.get_submeshes():
        ttnn.close_mesh_device(child)
    ttnn.close_mesh_device(parent)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def tower(submesh):
    t0 = time.perf_counter()
    hf_visual = clef_vision.load_hf_visual(SNAPSHOT)
    t_hf = time.perf_counter() - t0
    t0 = time.perf_counter()
    args = clef_vision.ClefVisionArgs(submesh, SNAPSHOT, max_batch_size=1, max_seq_len=4096)
    t_args = time.perf_counter() - t0
    vision = clef_vision.ClefVision(submesh, args, hf_visual)
    RESULTS["tower"] = dict(
        hf_visual_seconds=round(t_hf, 1),
        vision_args_seconds=round(t_args, 1),
        ccl_topology=str(args.ccl_topology()),
        ccl_num_links=vision.tt_model.tt_ccl.get_num_links(),
        device_name=args.device_name,
        **vision.describe(),
    )
    logger.info(f"tower: {RESULTS['tower']}")
    write_report()
    return vision


def load_reference(name):
    path = REF_DIR / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"missing reference {path}; run scripts/vision_reference.py")
    return torch.load(path)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("record_id", RECORDS)
def test_device_image_tower_matches_hf(submesh, tower, record_id):
    ref = load_reference(record_id)
    pixel_values = ref["pixel_values"]
    grid = ref["image_grid_thw"]
    n_tokens = int(ref["merged"].shape[0])

    t0 = time.perf_counter()
    merged, taps = tower.block_outputs_torch(pixel_values, grid, taps=clef_vision.BLOCK_TAPS)
    t_first = time.perf_counter() - t0
    run = dict(tower.last_run)

    t0 = time.perf_counter()
    merged_warm = tower.image_features_torch(pixel_values, grid)
    t_warm = time.perf_counter() - t0

    rows = tower.image_features(pixel_values, grid)
    rows_shape = list(rows.shape)
    rows_torch = tower.rows_to_torch(rows)
    ttnn.deallocate(rows)

    upstream = tower.upstream_features_torch(pixel_values, grid)
    t_upstream = tower.last_run["seconds"]

    row = dict(
        record=record_id,
        grid=grid.tolist(),
        n_patches=int(pixel_values.shape[0]),
        n_tokens=n_tokens,
        padded_rows=run.get("rows"),
        masked=run.get("masked"),
        window=run.get("window"),
        merged_shape=list(merged.shape),
        image_features_shape=rows_shape,
        pcc_merged=pcc(ref["merged"], merged),
        pcc_merged_warm=pcc(ref["merged"], merged_warm),
        pcc_blocks={str(i): pcc(ref["blocks"][i], taps[i]) for i in clef_vision.BLOCK_TAPS},
        pcc_upstream_unmasked=pcc(ref["merged"], upstream),
        image_features_roundtrip_max_abs=float((rows_torch - merged_warm).abs().max()),
        warm_vs_first_max_abs=float((merged - merged_warm).abs().max()),
        hf_cpu_seconds=ref["seconds"],
        tt_seconds_first_call=round(t_first, 3),
        tt_seconds_warm_call=round(t_warm, 3),
        tt_seconds_upstream_unmasked=round(t_upstream, 3),
    )
    row["passed"] = bool(
        row["pcc_merged"] >= PCC_END_TO_END
        and all(row["pcc_blocks"][str(i)] >= bar for i, bar in BLOCK_BARS.items())
        and rows_shape == [n_tokens, tower.out_hidden_size // tower.tp]
    )
    RESULTS["cases"][record_id] = row
    write_report()
    logger.info(f"vision {record_id}: {row}")

    assert torch.isfinite(merged).all(), "non-finite tower output"
    assert list(merged.shape) == [n_tokens, tower.out_hidden_size]
    assert rows_shape == [n_tokens, tower.out_hidden_size // tower.tp], rows_shape
    assert list(rows_torch.shape) == [n_tokens, tower.out_hidden_size]
    assert row["image_features_roundtrip_max_abs"] < 1e-2
    for index, bar in BLOCK_BARS.items():
        assert row["pcc_blocks"][str(index)] >= bar, f"block {index}: PCC {row['pcc_blocks'][str(index)]} < {bar}"
    assert row["pcc_merged"] >= PCC_END_TO_END, f"merged PCC {row['pcc_merged']} < {PCC_END_TO_END}"


@pytest.mark.timeout(1500)
def test_device_video_tower_matches_hf(submesh, tower):
    ref = load_reference(VIDEO_RECORD)
    pixel_values = ref["pixel_values_videos"]
    grid = ref["video_grid_thw"]
    t0 = time.perf_counter()
    merged = tower.video_features_torch(pixel_values, grid)
    seconds = time.perf_counter() - t0
    row = dict(
        record=VIDEO_RECORD,
        grid=grid.tolist(),
        frames=ref["frames"],
        n_patches=int(pixel_values.shape[0]),
        n_tokens=int(ref["merged"].shape[0]),
        padded_rows=tower.last_run.get("rows"),
        masked=tower.last_run.get("masked"),
        window=tower.last_run.get("window"),
        pcc_merged=pcc(ref["merged"], merged),
        hf_cpu_seconds=ref["seconds"],
        tt_seconds=round(seconds, 3),
    )
    row["passed"] = bool(row["pcc_merged"] >= PCC_END_TO_END)
    RESULTS["cases"][VIDEO_RECORD] = row
    write_report()
    logger.info(f"vision {VIDEO_RECORD}: {row}")
    assert list(merged.shape) == [row["n_tokens"], tower.out_hidden_size]
    assert row["pcc_merged"] >= PCC_END_TO_END, f"video merged PCC {row['pcc_merged']} < {PCC_END_TO_END}"


@pytest.fixture(scope="module")
def tower_upstream(submesh):
    args = clef_vision.ClefVisionArgs(
        submesh, SNAPSHOT, max_batch_size=1, max_seq_len=4096, precision="upstream", activation_bf16=False
    )
    vision = clef_vision.ClefVision(submesh, args, clef_vision.load_hf_visual(SNAPSHOT))
    RESULTS["tower_upstream"] = vision.describe()
    write_report()
    return vision


@pytest.mark.timeout(1500)
def test_device_upstream_defaults_reproduce_saved_rows(submesh, tower_upstream):
    if not UPSTREAM_ROWS.exists():
        pytest.skip(f"missing {UPSTREAM_ROWS}")
    ref = load_reference(RECORDS[0])
    saved = torch.load(UPSTREAM_ROWS).to(torch.bfloat16)
    merged = tower_upstream.image_features_torch(ref["pixel_values"], ref["image_grid_thw"]).to(torch.bfloat16)
    describe = tower_upstream.describe()
    row = dict(
        record=RECORDS[0],
        wqkv_dtype=describe["wqkv_dtype"],
        sdpa_dtype=describe["sdpa_dtype"],
        mlp_fc1_dtype=describe["mlp_fc1_dtype"],
        mlp_fidelity=describe["mlp_fidelity"],
        merger_fidelity=describe["merger_fidelity"],
        max_abs_vs_saved=float((merged.float() - saved.float()).abs().max()),
        bit_equal_vs_saved=bool(torch.equal(merged, saved)),
        pcc_vs_hf=pcc(ref["merged"], merged),
    )
    RESULTS["cases"]["upstream_defaults_" + RECORDS[0]] = row
    write_report()
    logger.info(f"upstream defaults: {row}")
    assert "BFLOAT8_B" in row["wqkv_dtype"] and "BFLOAT8_B" in row["sdpa_dtype"] and "BFLOAT8_B" in row["mlp_fc1_dtype"]
    assert "HiFi2" in row["mlp_fidelity"] and "HiFi2" in row["merger_fidelity"]
    assert row["bit_equal_vs_saved"], f"default-args tower differs from the saved rows by {row['max_abs_vs_saved']}"


@pytest.fixture(scope="module")
def tower_actbf16(submesh, tower_upstream):
    args = clef_vision.ClefVisionArgs(
        submesh, SNAPSHOT, max_batch_size=1, max_seq_len=4096, precision="upstream", activation_bf16=True
    )
    vision = clef_vision.ClefVision(submesh, args, tower_upstream.hf_visual)
    RESULTS["tower_actbf16"] = vision.describe()
    write_report()
    return vision


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("record_id", SENSITIVE_RECORDS)
def test_device_sensitive_record_all_precisions(submesh, tower, tower_upstream, tower_actbf16, record_id):
    ref = load_reference(record_id)
    pixel_values = ref["pixel_values"]
    grid = ref["image_grid_thw"]
    ROWS_DIR.mkdir(parents=True, exist_ok=True)
    row = dict(
        record=record_id,
        grid=grid.tolist(),
        n_patches=int(pixel_values.shape[0]),
        n_tokens=int(ref["merged"].shape[0]),
        towers={},
    )
    for name, vision in (("accuracy", tower), ("upstream", tower_upstream), ("act_bf16", tower_actbf16)):
        merged, taps = vision.block_outputs_torch(pixel_values, grid, taps=clef_vision.BLOCK_TAPS)
        rows_file = ROWS_DIR / f"{record_id}_{name}.pt"
        torch.save(merged, rows_file)
        torch.save({str(i): taps[i] for i in clef_vision.BLOCK_TAPS}, ROWS_DIR / f"{record_id}_{name}_taps.pt")
        describe = vision.describe()
        row["towers"][name] = dict(
            precision=describe["precision"],
            activation_bf16=describe["activation_bf16"],
            bfp8_weights=describe["bfp8_weights"],
            window=vision.last_run.get("window"),
            padded_rows=vision.last_run.get("rows"),
            pcc_merged=pcc(ref["merged"], merged),
            pcc_blocks={str(i): pcc(ref["blocks"][i], taps[i]) for i in clef_vision.BLOCK_TAPS},
            merged_max_abs=float(merged.abs().max()),
            merged_std=float(merged.std()),
            reference_std=float(ref["merged"].float().std()),
            tt_seconds=round(vision.last_run["seconds"], 3),
            rows_file=str(rows_file),
        )
    RESULTS["cases"]["sensitive_" + record_id] = row
    write_report()
    logger.info(f"sensitive {record_id}: {row}")
    accuracy = row["towers"]["accuracy"]
    for index, bar in BLOCK_BARS.items():
        assert (
            accuracy["pcc_blocks"][str(index)] >= bar
        ), f"block {index}: PCC {accuracy['pcc_blocks'][str(index)]} < {bar}"
    assert accuracy["pcc_merged"] >= PCC_END_TO_END, f"merged PCC {accuracy['pcc_merged']} < {PCC_END_TO_END}"


@pytest.mark.timeout(1500)
def test_device_precision_improves_80_token_image(submesh, tower, tower_upstream):
    ref = load_reference(RECORDS[0])
    accurate = tower.image_features_torch(ref["pixel_values"], ref["image_grid_thw"])
    upstream = tower_upstream.image_features_torch(ref["pixel_values"], ref["image_grid_thw"])
    row = dict(
        record=RECORDS[0],
        precision=tower.describe()["precision"],
        pcc_accuracy=pcc(ref["merged"], accurate),
        pcc_upstream_defaults=pcc(ref["merged"], upstream),
        base_pcc_bf16_run=BASE_PCC_80,
    )
    RESULTS["cases"]["precision_80_tokens"] = row
    write_report()
    logger.info(f"precision: {row}")
    assert row["pcc_accuracy"] > BASE_PCC_80
    assert row["pcc_accuracy"] > row["pcc_upstream_defaults"]

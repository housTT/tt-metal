# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest

from models.autoports.convaiinnovations_laya.tt import model_config as mc

DOC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "doc", "optimized_full_model")
CELLS = ("1", "5", "10", "50")


def _load(name):
    path = os.path.join(DOC_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{path} not written yet")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def replay():
    return _load("replay_trace_check_all_buckets.json")


@pytest.fixture(scope="module")
def latency():
    return _load("latency_table.json")


@pytest.fixture(scope="module")
def summary():
    return _load("perf_summary.json")


def test_tracked_replay_covers_every_deployment_bucket(replay):
    assert replay["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert replay["unsafe_allocation_error"] is None
    assert replay["pass"] is True
    got = sorted((b["batch"], b["seq"]) for b in replay["buckets"])
    assert got == sorted(mc.deployment_buckets())
    assert replay["rounds"] >= 3
    for b in replay["buckets"]:
        assert b["traced_vs_eager_bit_identical"], b
        assert b["repeated_replay_identical"], b
        assert b["updated_input_changes_output"], b
        assert not b["nan"], b
        assert b["replays"] >= 2 * replay["rounds"]


def test_trace_region_holds_the_deployment_set(replay, summary):
    assert replay["trace_bytes_total"] > 0
    assert replay["trace_bytes_total"] <= replay["trace_region_size"]
    assert summary["trace_region_fits"] is True
    assert summary["trace_region_bytes"] == 512 * 1024 * 1024
    assert len(summary["trace_bytes_per_bucket"]) == len(mc.deployment_buckets())
    assert all(v > 0 for v in summary["trace_bytes_per_bucket"].values())


def test_latency_table_has_both_process_kinds_for_every_published_cell(latency):
    for n in CELLS:
        c = latency["cells"][n]
        assert c["deployment_process"] is not None and c["fresh_process"] is not None, n
        for kind in ("deployment_process", "fresh_process"):
            assert c[kind]["end_to_end_ms_p50"] > 0
            assert c[kind]["device_ms_p50"] > 0
            assert c[kind]["host_tail_ms_p50"] >= 0
            assert c[kind]["loadavg"][0] < 8.0, (n, kind, c[kind]["loadavg"])
        assert c["t4_published_ms"] > 0 and c["build0_host_served_client_ms"] > 0
        assert c["bucket"] == list(mc.select_bucket(int(n), max(c["row_lengths"])))
        assert c["deployment_process"]["end_to_end_ms_p50"] < c["build0_host_served_device_ms"], n


def test_throughput_cells_present(latency):
    for key in ("64x128", "64x256", "64x512"):
        c = latency["throughput_b64"][key]
        assert c["rows"] == 64 and c["rows_per_s"] > 0 and c["tokens_per_s_real"] > 0
        assert c["replay_only_ms_p50"] > 0 and c["replay_only_ms_p50"] <= c["end_to_end_ms_p50"]


def test_perf_summary_is_complete(summary):
    for key in (
        "policy",
        "port",
        "row_buckets",
        "seq_buckets",
        "trace_bytes_per_bucket",
        "trace_bytes_total",
        "warmup_seconds",
        "published_cells",
        "per_bucket",
        "throughput_b64",
        "reconciliation",
        "trace_safety_all_buckets",
        "stage6_gates_on_final_configuration",
        "ab",
    ):
        assert key in summary, key
    assert summary["row_buckets"] == list(mc.ROW_BUCKETS) and summary["seq_buckets"] == list(mc.SEQ_BUCKETS)
    assert len(summary["per_bucket"]) == len(mc.deployment_buckets())
    assert summary["warmup_seconds"]["phase1"] > 0 and summary["warmup_seconds"]["phase2"] > 0
    assert summary["trace_safety_all_buckets"]["pass"] is True
    assert len(summary["reconciliation"]) >= 2
    for name, r in summary["reconciliation"].items():
        assert "lower_bound_ms" in r["segments"], name
        assert r["traced_ms"] is not None and r["gap_pct"] is not None, name
        assert 0 <= r["gap_pct"] < 25, (name, r["gap_pct"])
    g = summary["stage6_gates_on_final_configuration"]
    assert g["pass"] is True and g["agreement_pass"] is True
    for k, v in g["gates"].items():
        assert v["pass"], k


def test_bucket_selection_rule_matches_the_served_shapes():
    assert mc.select_bucket(1, 194) == (1, 256)
    assert mc.select_bucket(5, 205) == (5, 256)
    assert mc.select_bucket(10, 205) == (10, 256)
    assert mc.select_bucket(50, 205) == (50, 256)
    assert mc.select_bucket(1, 103) == (1, 128)
    assert mc.select_bucket(1, 51) == (1, 128)
    assert mc.select_bucket(5, 308) == (5, 512)
    assert mc.select_bucket(40, 205) == (50, 256)
    assert mc.select_bucket(64, 512) == (64, 512)
    with pytest.raises(ValueError):
        mc.select_bucket(65, 205)
    with pytest.raises(ValueError):
        mc.select_bucket(1, 513)


@pytest.mark.skipif(
    os.environ.get("LAYA_PERFORMANT_LIVE") != "1",
    reason="set LAYA_PERFORMANT_LIVE=1 to capture two buckets on the device",
)
def test_live_traced_equals_eager_on_two_buckets():
    import torch

    from models.autoports.convaiinnovations_laya.tests import laya_inputs as LI
    from models.autoports.convaiinnovations_laya.tt.engine import LayaEngine

    buckets = [(1, 256), (5, 256)]
    engine = LayaEngine(row_buckets=(1, 5), seq_buckets=(256,), warmup_shapes=buckets, threads=4)
    try:
        for b, s in buckets:
            x = LI.build_inputs(batch_size=b, seq_len=s, fill=b > 1)
            traced = engine.runner.run(x["input_ids"], x["attention_mask"], x["qtype"])
            eager = engine.model.forward(x["input_ids"], x["attention_mask"], x["qtype"], bucket=(b, s))
            assert traced["bucket"] == (b, s)
            assert torch.equal(traced["logits"], eager["logits"]) and torch.equal(traced["cls"], eager["cls"])
            times = []
            for _ in range(8):
                times.append(engine.runner.run(x["input_ids"], x["attention_mask"], x["qtype"])["device_ms"])
            assert sorted(times)[len(times) // 2] < 100.0
        assert engine.runner.trace_bytes_total() > 0
    finally:
        engine.close()

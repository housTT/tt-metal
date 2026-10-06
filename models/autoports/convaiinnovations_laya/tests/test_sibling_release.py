# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest

from models.autoports.convaiinnovations_laya.tests.decision_agreement import GATE_MAX_ABS_DP, N_QUESTIONS
from models.autoports.convaiinnovations_laya.tests.run_fidelity import GATES
from models.autoports.convaiinnovations_laya.tt import model_config as mc

DOC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "doc", "release_typed_decisions")
POLICY = os.environ.get("LAYA_POLICY") or "bf8w_hifi3_erf"
SIBLING_REVISION = "e929ae5cf69bc34259cd2f95c9e91145b818b1f0"
SEQ1024 = [(b, 1024) for b in mc.ROW_BUCKETS_AT_1024]


def _load(name):
    path = os.path.join(DOC_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{path} not written yet")
    with open(path) as f:
        return json.load(f)


def _fidelity_checks(fid):
    assert fid["gate_subset"]["n"] == 200 and fid["corpus"].endswith("parity_corpus_td.npz")
    assert fid["model_dir"].endswith("laya-typed-decisions")
    g = fid["gates"]
    assert g["confident_argmax_agreement"]["value"] >= GATES["confident_agreement"], g
    assert g["median_max_abs_dp"]["value"] <= GATES["median_max_abs_dp"], g
    assert g["scorer_logit_pcc"]["value"] >= GATES["scorer_pcc"], g
    assert g["no_nan"]["pass"], g
    h = g.get("hidden_state_pcc")
    assert h is not None and h["encoder_pooled"] >= GATES["hidden_pcc"] and h["head_pooled"] >= GATES["hidden_pcc"], h
    assert fid["pass"] is True
    assert fid["all_items"]["n"] == 488 and fid["all_items"]["nan_rows"] == 0


def test_fidelity_natural_buckets():
    fid = _load(f"fidelity_{POLICY}_td_natural.json")
    _fidelity_checks(fid)
    assert fid["seq_buckets"] == [128, 256, 512, 1024]
    assert all(k.endswith(("x128", "x256", "x512")) for k in fid["bucket_histogram"]), fid["bucket_histogram"]
    assert not any(k.endswith("x1024") for k in fid["bucket_histogram"]), fid["bucket_histogram"]


def test_fidelity_forced_1024_bucket():
    fid = _load(f"fidelity_{POLICY}_td_seq1024.json")
    _fidelity_checks(fid)
    assert fid["seq_buckets"] == [1024] and fid["row_buckets"] == list(mc.ROW_BUCKETS_AT_1024)
    assert all(k.endswith("x1024") for k in fid["bucket_histogram"]), fid["bucket_histogram"]
    assert sum(fid["bucket_histogram"].values()) == fid["calls"]


def _invariance_checks(inv, largest):
    assert inv["summary"]["questions"] == N_QUESTIONS
    assert inv["summary"]["placements"] == ["alone", "b2", "b4", "mixed_b8", largest]
    assert inv["gates"]["same_argmax"]["pass"], inv["gates"]
    assert inv["gates"]["max_abs_dp_alone_vs_in_batch"]["value"] <= GATE_MAX_ABS_DP, inv["gates"]
    for p, v in inv["summary"]["max_abs_dp_alone_vs"].items():
        assert v <= GATE_MAX_ABS_DP, (p, v)
    assert inv["pass"] is True and inv["corpus"].endswith("parity_corpus_td.npz")


def test_invariance_natural_buckets():
    inv = _load(f"decision_agreement_{POLICY}_td_natural.json")
    _invariance_checks(inv, "b64")
    assert inv["buckets"]["b64"] == [64, 512] and inv["buckets"]["mixed_b8"] == [8, 512]
    assert inv["engine"]["row_buckets_by_seq"]["1024"] == list(mc.ROW_BUCKETS_AT_1024)


def test_invariance_forced_1024_bucket():
    inv = _load(f"decision_agreement_{POLICY}_td_seq1024.json")
    _invariance_checks(inv, "b16")
    assert inv["largest_rows"] == 16 and inv["b16_real_rows"] == 16
    assert inv["buckets"] == {
        "alone": [1, 1024],
        "b2": [2, 1024],
        "b4": [4, 1024],
        "mixed_b8": [8, 1024],
        "b16": [16, 1024],
    }


def test_tracked_replay_covers_the_sibling_set():
    rep = _load("replay_trace_check_sibling.json")
    assert (
        rep["TT_METAL_TRACE_ALLOC_TRACKING"] == "1" and rep["unsafe_allocation_error"] is None and rep["pass"] is True
    )
    got = sorted((b["batch"], b["seq"]) for b in rep["buckets"])
    assert got == sorted(mc.sibling_deployment_buckets()) and len(got) == 37
    for b in rep["buckets"]:
        assert (
            b["traced_vs_eager_bit_identical"]
            and b["repeated_replay_identical"]
            and b["updated_input_changes_output"]
            and not b["nan"]
        ), b
    assert 0 < rep["trace_bytes_total"] <= rep["trace_region_size"] == 512 * 1024 * 1024
    assert all(rep["trace_bytes"][f"{b}x{s}"] > 0 for b, s in SEQ1024)


def test_sibling_bench_has_every_bucket_bit_identical_to_eager():
    bench = _load("bench_sibling_final.json")
    cells = bench["variants"]["default"]["buckets"]
    assert bench["variants"]["default"]["error"] is None
    assert sorted(cells) == sorted(f"{b}x{s}" for b, s in mc.sibling_deployment_buckets())
    for key, c in cells.items():
        assert c["traced_vs_eager_max_abs_logits"] == 0.0, key
        assert c["traced_ms_p50"] > 0 and c["loadavg"][0] < 8.0, (key, c["loadavg"])
    for b in mc.ROW_BUCKETS_AT_1024[1:]:
        assert cells[f"{b}x1024"]["traced_ms_p50"] > cells["1x1024"]["traced_ms_p50"]


def test_ab_scatter_pair_recorded():
    a = _load("ab/bench_default_a.json")["variants"]["default_a"]["buckets"]
    b = _load("ab/bench_default_b.json")["variants"]["default_b"]["buckets"]
    for key in (f"{r}x1024" for r in mc.ROW_BUCKETS_AT_1024):
        assert abs(a[key]["traced_ms_p50"] - b[key]["traced_ms_p50"]) / a[key]["traced_ms_p50"] < 0.03, key


def test_served_health_records_the_sibling_buckets():
    h = _load("served_health_td.json")
    assert h["revision"] == SIBLING_REVISION and h["model_dir"].endswith("laya-typed-decisions")
    assert h["max_len"] == 1024 and h["head_max_len"] == 256
    assert (
        h["shapes"]["row_buckets_by_seq"]["1024"] == list(mc.ROW_BUCKETS_AT_1024)
        and h["shapes"]["max_rows_by_seq"]["1024"] == 16
    )
    assert h["limits"]["max_batch_tokens"] == 16384
    assert h["seq_buckets"] == [128, 256, 512, 1024]
    shapes = h.get("shapes") or h.get("backend_shapes") or {}
    warm = [tuple(w) for w in (h.get("warm_shapes") or shapes.get("warm_shapes") or [])]
    assert sorted(warm) == sorted(mc.sibling_deployment_buckets())

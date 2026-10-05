# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")

CORPUS = "/home/hous/dev/laya/reference/parity_corpus.npz"
DOC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "doc", "multichip_decoder")
REF_JSON = os.path.join(DOC_DIR, "ref_1x1.json")
AGREEMENT_JSON = os.path.join(DOC_DIR, "dp_agreement_1x4.json")
BENCH_JSON = os.path.join(DOC_DIR, "bench_1x4.json")
PAD_ID = 50283
SEQ = 512
ROWS = 64
GATE_MAX_ABS_LOGIT = 1e-3
GATE_PCC = 0.999
GATE_CONFIDENT = 0.99
GATE_SPEEDUP = 3.0
CONFIDENT_MARGIN = 0.10


def loadavg():
    with open("/proc/loadavg") as f:
        return [float(x) for x in f.read().split()[:3]]


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64) * 1000.0, q))


def corpus_rows(n_rows=ROWS, corpus=CORPUS):
    z = np.load(corpus, allow_pickle=True)
    sel = np.where(z["source"] == 0)[0][:n_rows]
    ids = torch.as_tensor(z["input_ids"][sel]).long()
    att = torch.as_tensor(z["attention_mask"][sel]).long()
    ids = ids.masked_fill(ids < 0, PAD_ID)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": torch.as_tensor(z["marker_pos"][sel]).long(),
        "marker_mask": torch.as_tensor(z["marker_mask"][sel]).bool(),
        "qtype": torch.as_tensor(z["qtype"][sel]).long(),
        "k": z["k"][sel].astype(int).tolist(),
        "temperature": z["temperature"][sel].astype(np.float32),
        "logits_fp32": z["logits_fp32"][sel].astype(np.float32),
        "probs_fp32": z["probs"][sel].astype(np.float32),
        "rows": sel.tolist(),
    }


def gather_markers(logits_all, marker_pos, marker_mask):
    g = torch.gather(logits_all, 1, marker_pos.clamp(min=0))
    return g.masked_fill(~marker_mask, -1e4)


def probs_from_logits(marker_logits, k, temperature):
    out = np.full(marker_logits.shape, 0.0, dtype=np.float32)
    for r in range(marker_logits.shape[0]):
        z = np.asarray(marker_logits[r, : k[r]], dtype=np.float32) / float(temperature[r])
        e = np.exp(z - z.max())
        out[r, : k[r]] = e / e.sum()
    return out


def compare(name, a_logits, b_logits, k, temperature, confident_source=None):
    """a and b are (n, kmax) marker logits of the same rows; confident_source selects the margin reference (default b)."""
    a = np.asarray(a_logits, dtype=np.float32)
    b = np.asarray(b_logits, dtype=np.float32)
    n = a.shape[0]
    pa = probs_from_logits(a, k, temperature)
    pb = probs_from_logits(b, k, temperature)
    pc = (
        pb
        if confident_source is None
        else probs_from_logits(np.asarray(confident_source, dtype=np.float32), k, temperature)
    )
    valid = np.zeros(a.shape, dtype=bool)
    for r in range(n):
        valid[r, : k[r]] = True
    da = a[valid].astype(np.float64)
    db = b[valid].astype(np.float64)
    pcc = float(np.corrcoef(da, db)[0, 1]) if da.std() > 0 and db.std() > 0 else float("nan")
    agree = [int(np.argmax(pa[r, : k[r]]) == np.argmax(pb[r, : k[r]])) for r in range(n)]
    margins = []
    for r in range(n):
        s = np.sort(pc[r, : k[r]])[::-1]
        margins.append(float(s[0] - s[1]) if k[r] >= 2 else 1.0)
    conf = [i for i in range(n) if margins[i] >= CONFIDENT_MARGIN]
    dp = [float(np.abs(pa[r, : k[r]] - pb[r, : k[r]]).max()) for r in range(n)]
    return {
        "comparison": name,
        "rows": n,
        "max_abs_logit_delta": float(np.abs(da - db).max()),
        "mean_abs_logit_delta": float(np.abs(da - db).mean()),
        "pcc_marker_logits": pcc,
        "argmax_agree": int(sum(agree)),
        "argmax_agree_rate": float(sum(agree) / n),
        "confident_rows": len(conf),
        "confident_agree": int(sum(agree[i] for i in conf)),
        "confident_agree_rate": float(sum(agree[i] for i in conf) / len(conf)) if conf else float("nan"),
        "max_abs_dp": float(max(dp)),
        "median_abs_dp": float(np.median(dp)),
        "nan": bool(np.isnan(a).any()),
    }


def build_model(device, row_buckets, policy_name=None, port_overrides=None):
    from models.autoports.convaiinnovations_laya.tests import laya_inputs as LI
    from models.autoports.convaiinnovations_laya.tt import model_config as mc
    from models.autoports.convaiinnovations_laya.tt.laya_model import TtnnLayaModel
    from models.autoports.convaiinnovations_laya.tt.weights import load_state_dict

    policy = mc.policy_from_name(policy_name)
    port = mc.DEFAULT_PORT.with_(**(port_overrides or {}))
    config = LI.load_config()
    sd = load_state_dict()
    model = TtnnLayaModel(
        device, config, state_dict=sd, policy=policy, port=port, row_buckets=row_buckets, seq_buckets=(SEQ,)
    )
    return model


def run_rows(runner, data, bucket, rows_per_call, timed_reps=0):
    """Run all rows in calls of rows_per_call at the given per-device bucket; returns marker logits (n, kmax) and timing."""
    n = data["input_ids"].shape[0]
    outs = []
    for start in range(0, n, rows_per_call):
        sl = slice(start, min(start + rows_per_call, n))
        out = runner.run(data["input_ids"][sl], data["attention_mask"][sl], data["qtype"][sl], bucket=bucket)
        outs.append(gather_markers(out["logits"], data["marker_pos"][sl], data["marker_mask"][sl]))
    timing = None
    if timed_reps:
        sl = slice(0, rows_per_call)
        splits = []
        for _ in range(timed_reps):
            t = runner.run_timed(data["input_ids"][sl], data["attention_mask"][sl], data["qtype"][sl], bucket=bucket)
            splits.append((t["write_ms"], t["replay_ms"], t["readback_ms"], t["device_ms"]))
        arr = np.asarray(splits)
        timing = {
            "write_ms_p50": float(np.median(arr[:, 0])),
            "replay_ms_p50": float(np.median(arr[:, 1])),
            "readback_ms_p50": float(np.median(arr[:, 2])),
            "blocking_total_ms_p50": float(np.median(arr[:, 3])),
            "reps": timed_reps,
        }
    return torch.cat(outs, 0), timing


def bench_bucket(runner, data, bucket, rows_per_call, warm=3, reps=20):
    sl = slice(0, rows_per_call)
    times = []
    for _ in range(warm + reps):
        t0 = time.perf_counter()
        runner.run(data["input_ids"][sl], data["attention_mask"][sl], data["qtype"][sl], bucket=bucket)
        times.append(time.perf_counter() - t0)
    times = times[warm:]
    p50 = pct(times, 50)
    return {
        "rows_per_call": rows_per_call,
        "bucket": list(bucket),
        "traced_ms_p50": p50,
        "traced_ms_min": float(min(times)) * 1000.0,
        "traced_ms_p95": pct(times, 95),
        "rows_per_s": rows_per_call / (p50 / 1000.0),
        "warm": warm,
        "reps": reps,
        "loadavg": loadavg(),
    }


def run_1x1(args):
    from models.autoports.convaiinnovations_laya.tt.laya_model import close_device, open_device
    from models.autoports.convaiinnovations_laya.tt.runner import LayaTraceRunner

    data = corpus_rows()
    buckets = [8, 16, 64]
    device = open_device(device_id=0, l1_small_size=79104, trace_region_size=args.trace_region)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "mesh": "1x1",
        "device_ids": [0],
        "rows": ROWS,
        "corpus_rows": data["rows"],
        "loadavg_start": loadavg(),
        "per_bucket": {},
    }
    try:
        model = build_model(device, buckets, args.policy)
        report["model"] = model.describe()
        runner = LayaTraceRunner(model, [(b, SEQ) for b in buckets])
        runner.warmup()
        report["warmup_seconds"] = runner.warmup_seconds
        for b in buckets:
            logits, timing = run_rows(runner, data, (b, SEQ), b, timed_reps=args.timed_reps)
            cell = {"marker_logits": logits.tolist(), "timing_split": timing}
            cell["bench"] = bench_bucket(runner, data, (b, SEQ), b, warm=args.warm, reps=args.reps)
            cell["vs_cpu_fp32"] = compare(
                f"1x1 B{b} vs CPU fp32",
                logits.numpy(),
                data["logits_fp32"],
                data["k"],
                data["temperature"],
                confident_source=data["logits_fp32"],
            )
            report["per_bucket"][str(b)] = cell
            print("CELL 1x1", b, json.dumps({k: v for k, v in cell.items() if k != "marker_logits"}))
        runner.release()
        model.close()
    finally:
        close_device(device)
    report["loadavg_end"] = loadavg()
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("DP_1x1_DONE", args.out)


def run_1x4(args):
    from models.autoports.convaiinnovations_laya.tt.laya_model import close_device, open_mesh
    from models.autoports.convaiinnovations_laya.tt.runner import LayaTraceRunner

    with open(args.ref) as f:
        ref = json.load(f)
    data = corpus_rows()
    if ref["corpus_rows"] != data["rows"]:
        raise RuntimeError("1x1 reference was built on different corpus rows")
    per_device = [8, 16]
    t0 = time.perf_counter()
    device = open_mesh((1, 4), l1_small_size=79104, trace_region_size=args.trace_region)
    open_s = time.perf_counter() - t0
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "mesh": "1x4",
        "fabric_config": None,
        "device_ids": list(device.get_device_ids()),
        "mesh_open_seconds": round(open_s, 2),
        "rows": ROWS,
        "corpus_rows": data["rows"],
        "reference_1x1": args.ref,
        "loadavg_start": loadavg(),
        "gates": {},
        "comparisons": [],
        "per_cell": {},
    }
    bench = {
        "timestamp": report["timestamp"],
        "mesh": "1x4",
        "device_ids": report["device_ids"],
        "cells": {},
        "reference_1x1": {},
    }
    try:
        model = build_model(device, per_device, args.policy)
        report["model"] = model.describe()
        runner = LayaTraceRunner(model, [(b, SEQ) for b in per_device])
        runner.warmup()
        report["warmup_seconds"] = runner.warmup_seconds
        results = {}
        for b in per_device:
            rows_per_call = b * 4
            logits, timing = run_rows(runner, data, (b, SEQ), rows_per_call, timed_reps=args.timed_reps)
            results[b] = logits
            ref_cell = ref["per_bucket"][str(b)]
            n = min(rows_per_call, ROWS)
            same = compare(
                f"1x4 B{b}x4 vs 1x1 B{b} (same per-chip bucket), rows 0..{n - 1}",
                logits[:n].numpy(),
                np.asarray(ref_cell["marker_logits"])[:n],
                data["k"][:n],
                data["temperature"][:n],
            )
            cpu = compare(
                f"1x4 B{b}x4 vs CPU fp32",
                logits.numpy(),
                data["logits_fp32"],
                data["k"],
                data["temperature"],
                confident_source=data["logits_fp32"],
            )
            cell_bench = bench_bucket(runner, data, (b, SEQ), rows_per_call, warm=args.warm, reps=args.reps)
            ref_bench = ref_cell["bench"]
            ref_split = ref_cell["timing_split"]
            cell = {
                "per_device_bucket": b,
                "rows_per_call": rows_per_call,
                "same_bucket_vs_1x1": same,
                "vs_cpu_fp32": cpu,
                "bench": cell_bench,
                "timing_split_1x4": timing,
                "timing_split_1x1_same_bucket": ref_split,
                "replay_overlap": (ref_split["replay_ms_p50"] / timing["replay_ms_p50"])
                if timing and ref_split
                else None,
                "host_concat_cost_ms": (timing["readback_ms_p50"] - ref_split["readback_ms_p50"])
                if timing and ref_split
                else None,
                "readback_ms_1x4": timing["readback_ms_p50"] if timing else None,
                "readback_ms_1x1": ref_split["readback_ms_p50"] if ref_split else None,
                "speedup_vs_1x1_same_bucket_x4_calls": (4 * ref_bench["traced_ms_p50"]) / cell_bench["traced_ms_p50"],
                "rows_per_s_1x1_same_bucket": ref_bench["rows_per_s"],
            }
            report["per_cell"][f"{b}x4"] = cell
            report["comparisons"] += [same, cpu]
            bench["cells"][f"{b}x4"] = cell_bench
            bench["reference_1x1"][str(b)] = ref_bench
            print(
                "CELL 1x4",
                b,
                json.dumps({k: v for k, v in cell.items() if k not in ("same_bucket_vs_1x1", "vs_cpu_fp32")}),
                json.dumps(same),
                json.dumps(cpu),
            )
        ref64 = ref["per_bucket"]["64"]
        bench["reference_1x1"]["64"] = ref64["bench"]
        cross = compare(
            "1x4 B16x4 vs 1x1 B64 (64 rows)",
            results[16].numpy(),
            np.asarray(ref64["marker_logits"]),
            data["k"],
            data["temperature"],
        )
        cross_cpu_1x1 = ref64["vs_cpu_fp32"]
        report["comparisons"].append(cross)
        speedup = report["per_cell"]["16x4"]["bench"]["rows_per_s"] / ref64["bench"]["rows_per_s"]
        g1 = report["per_cell"]["8x4"]["same_bucket_vs_1x1"]
        g1b = report["per_cell"]["16x4"]["same_bucket_vs_1x1"]
        report["gates"] = {
            "same_inputs_equal_per_chip_bucket_max_abs_logit_delta": {
                "B8x4": g1["max_abs_logit_delta"],
                "B16x4": g1b["max_abs_logit_delta"],
                "threshold": GATE_MAX_ABS_LOGIT,
                "pass": bool(max(g1["max_abs_logit_delta"], g1b["max_abs_logit_delta"]) <= GATE_MAX_ABS_LOGIT),
            },
            "same_inputs_equal_per_chip_bucket_argmax_agreement": {
                "B8x4": g1["argmax_agree_rate"],
                "B16x4": g1b["argmax_agree_rate"],
                "threshold": 1.0,
                "pass": bool(g1["argmax_agree_rate"] == 1.0 and g1b["argmax_agree_rate"] == 1.0),
            },
            "B64_1x1_vs_B16x4_pcc": {
                "value": cross["pcc_marker_logits"],
                "threshold": GATE_PCC,
                "pass": bool(cross["pcc_marker_logits"] >= GATE_PCC),
            },
            "B64_1x1_vs_B16x4_confident_agreement": {
                "value": cross["confident_agree_rate"],
                "agree": cross["confident_agree"],
                "of": cross["confident_rows"],
                "threshold": GATE_CONFIDENT,
                "pass": bool(cross["confident_agree_rate"] >= GATE_CONFIDENT),
            },
            "throughput_B16x4_vs_1x1_B64": {
                "rows_per_s_1x4": report["per_cell"]["16x4"]["bench"]["rows_per_s"],
                "rows_per_s_1x1_B64": ref64["bench"]["rows_per_s"],
                "speedup": speedup,
                "threshold": GATE_SPEEDUP,
                "pass": bool(speedup >= GATE_SPEEDUP),
            },
        }
        report["cross_check_vs_cpu"] = {
            "1x1_B64": cross_cpu_1x1,
            "1x4_B16x4": report["per_cell"]["16x4"]["vs_cpu_fp32"],
        }
        report["pass"] = all(g["pass"] for g in report["gates"].values())
        bench["speedup_B16x4_vs_1x1_B64"] = speedup
        bench["gates"] = report["gates"]
        runner.release()
        model.close()
    finally:
        close_device(device)
    report["loadavg_end"] = loadavg()
    bench["loadavg_end"] = report["loadavg_end"]
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    with open(args.bench, "w") as f:
        json.dump(bench, f, indent=1)
    print("DP_GATES", json.dumps(report["gates"]))
    print("DP_1x4_DONE", args.out, "pass", report["pass"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", choices=["1x1", "1x4"], required=True)
    ap.add_argument("--ref", default=REF_JSON)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bench", default=BENCH_JSON)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--trace-region", type=int, default=512 * 1024 * 1024)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--timed-reps", type=int, default=10)
    a = ap.parse_args(argv)
    os.makedirs(DOC_DIR, exist_ok=True)
    if a.mesh == "1x1":
        a.out = a.out or REF_JSON
        run_1x1(a)
    else:
        a.out = a.out or AGREEMENT_JSON
        run_1x4(a)


@pytest.fixture(scope="module")
def dp_evidence():
    if not os.path.exists(AGREEMENT_JSON):
        pytest.skip(f"{AGREEMENT_JSON} missing; run test_dp_mesh.py --mesh 1x1 then --mesh 1x4")
    with open(AGREEMENT_JSON) as f:
        return json.load(f)


def test_mesh_opened_without_fabric(dp_evidence):
    assert dp_evidence["mesh"] == "1x4"
    assert dp_evidence["fabric_config"] is None
    assert len(dp_evidence["device_ids"]) == 4


def test_same_inputs_equal_per_chip_bucket(dp_evidence):
    g = dp_evidence["gates"]
    assert g["same_inputs_equal_per_chip_bucket_max_abs_logit_delta"]["pass"], g
    assert g["same_inputs_equal_per_chip_bucket_argmax_agreement"]["pass"], g


def test_b64_vs_b16x4(dp_evidence):
    g = dp_evidence["gates"]
    assert g["B64_1x1_vs_B16x4_pcc"]["pass"], g
    assert g["B64_1x1_vs_B16x4_confident_agreement"]["pass"], g


def test_throughput(dp_evidence):
    g = dp_evidence["gates"]["throughput_B16x4_vs_1x1_B64"]
    assert g["pass"], g


@pytest.mark.skipif(
    os.environ.get("LAYA_DP_LIVE") != "1", reason="set LAYA_DP_LIVE=1 to open the 1x4 mesh inside pytest"
)
def test_live_1x4_against_saved_1x1_reference(tmp_path):
    if not os.path.exists(REF_JSON):
        pytest.skip(f"{REF_JSON} missing")
    out = str(tmp_path / "dp_agreement_1x4.json")
    bench = str(tmp_path / "bench_1x4.json")
    main(["--mesh", "1x4", "--ref", REF_JSON, "--out", out, "--bench", bench, "--reps", "5", "--timed-reps", "3"])
    with open(out) as f:
        rep = json.load(f)
    assert rep["gates"]["same_inputs_equal_per_chip_bucket_argmax_agreement"]["pass"]
    assert rep["gates"]["B64_1x1_vs_B16x4_pcc"]["pass"]


if __name__ == "__main__":
    main()

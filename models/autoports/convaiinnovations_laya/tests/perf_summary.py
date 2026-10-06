# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import glob
import json
import os
import socket
import time
from collections import defaultdict
from datetime import datetime, timezone

os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
AUTOPORT = os.path.dirname(HERE)
DOC_DIR = os.path.join(AUTOPORT, "doc", "optimized_full_model")
T4_PUBLISHED_MS = {1: 39.5, 5: 84.5, 10: 158.6, 50: 771.0}
BUILD0_CLIENT_MS = {1: 14.5, 5: 68.9, 10: 142.9, 50: 545.1}
BUILD0_DEVICE_MS = {1: 12.5, 5: 64.7, 10: 136.0, 50: 522.8}
BUILD0_BUCKETS = {1: "1x512", 5: "8x512", 10: "16x512", 50: "64x512"}
N_LAYERS = 28
N_HEAD_LAYERS = 2


def loadavg():
    with open("/proc/loadavg") as f:
        return [float(x) for x in f.read().split()[:3]]


def cmd_profile(a):
    """Run under `python -m tracy -r -p -v -o DIR`: one warm and one measured eager forward of one bucket."""
    import torch

    import ttnn
    from models.autoports.convaiinnovations_laya.tests import laya_inputs as LI
    from models.autoports.convaiinnovations_laya.tests.bench_latency import request_rows, speed_table_inputs
    from models.autoports.convaiinnovations_laya.tt import model_config as mc
    from models.autoports.convaiinnovations_laya.tt.laya_model import TtnnLayaModel
    from models.autoports.convaiinnovations_laya.tt.weights import load_state_dict

    torch.set_num_threads(6)
    policy = mc.policy_from_name(a.policy)
    port = mc.DEFAULT_PORT.with_(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.loads(a.port).items()})
    config = LI.load_config()
    sd = load_state_dict()
    b, s = a.batch, a.seq
    if a.inputs == "speed_table":
        STATE_EN, _, _, qs = speed_table_inputs()
        x = request_rows(LI.load_tokenizer(), STATE_EN, qs(b))
        if x["input_ids"].shape[1] > s:
            raise ValueError("speed-table rows do not fit the seq bucket")
    else:
        x = LI.build_inputs(batch_size=b, seq_len=s, fill=b > 1)
    device = ttnn.open_device(device_id=0, l1_small_size=79104, trace_region_size=0)
    report = {
        "batch": b,
        "seq": s,
        "policy": policy.describe(),
        "port": port.describe(),
        "inputs": a.inputs,
        "loadavg": loadavg(),
    }
    try:
        model = TtnnLayaModel(
            device, config, state_dict=sd, policy=policy, port=port, row_buckets=(b,), seq_buckets=(s,)
        )
        bucket = model.build_bucket(b, s)
        report["plan"] = mc.describe_plan(bucket.encoder.plan)
        times = []
        for i in range(a.repeats):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = model.forward(x["input_ids"], x["attention_mask"], x["qtype"], bucket=(b, s))
            ttnn.synchronize_device(device)
            times.append((time.perf_counter() - t0) * 1000.0)
        report["eager_ms"] = times
        report["eager_ms_last"] = times[-1]
        report["logits_first_row"] = out["logits"][0, :8].tolist()
        model.close()
    finally:
        ttnn.close_device(device)
    print("PROFILE_FORWARD", json.dumps(report))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)


def _pick(row, *names):
    for n in names:
        if n in row and row[n] not in ("", None):
            return row[n]
    return None


def read_ops(perf_csv):
    with open(perf_csv) as f:
        rows = list(csv.DictReader(f))
    ops = []
    for r in rows:
        raw = _pick(r, "Device Time", "DEVICE KERNEL DURATION [ns]")
        if raw in (None, ""):
            continue
        us = float(raw) if "Device Time" in r else float(raw) / 1000.0
        ops.append(
            {
                "op": _pick(r, "OP Code", "OP CODE", "op_code") or "",
                "us": us,
                "cores": _pick(r, "Cores", "CORE COUNT"),
                "fidelity": _pick(r, "Math Fidelity", "MATH FIDELITY") or "",
                "in0": _pick(r, "Input 0 Datatype", "INPUT_0_DATATYPE") or "",
                "in1": _pick(r, "Input 1 Datatype", "INPUT_1_DATATYPE") or "",
                "flops_pct": _pick(r, "FLOPs %", "PM FPU UTIL (%)") or "",
                "dram_pct": _pick(r, "DRAM %") or "",
                "bound": _pick(r, "Bound") or "",
            }
        )
    return ops


def last_pass(ops, repeats):
    names = [o["op"].split(" ")[0] for o in ops]
    n = len(names)
    for k in sorted({n // repeats, n // 2}):
        if k > 0 and names[n - k :] == names[n - 2 * k : n - k]:
            return ops[n - k :]
    for k in range(max(1, n // (repeats + 1) - 40), n // 2 + 1):
        if names[n - k :] == names[n - 2 * k : n - k]:
            return ops[n - k :]
    return ops


def base_name(op):
    return op["op"].split(" ")[0]


def segment(ops):
    """Split one forward pass into masks, embeddings, the 28-layer stack, final norm, type add, head layers, scorer and the CLS slice."""
    names = [base_name(o) for o in ops]
    emb = [i for i, n in enumerate(names) if n.startswith("Embedding")]
    ln = [i for i, n in enumerate(names) if n.startswith("LayerNorm")]
    if len(emb) < 2:
        return {"error": f"expected two Embedding ops, found {len(emb)}"}
    e1, e2 = emb[0], emb[1]
    ln_after_e1 = [i for i in ln if i > e1]
    ln_before_e2 = [i for i in ln if i < e2]
    ln_after_e2 = [i for i in ln if i > e2]
    if not ln_after_e1 or not ln_before_e2 or len(ln_after_e2) < 5:
        return {"error": "LayerNorm pattern not found"}
    emb_norm = ln_after_e1[0]
    final_norm = ln_before_e2[-1]
    scorer_norm = ln_after_e2[4]
    slice_idx = [i for i, n in enumerate(names) if n.startswith("Slice")]
    cls_idx = slice_idx[-1] if slice_idx else len(ops)
    seg = {
        "masks": ops[:e1],
        "embeddings": ops[e1 : emb_norm + 1],
        "encoder_stack": ops[emb_norm + 1 : final_norm],
        "final_norm": ops[final_norm : final_norm + 1],
        "type_add": ops[final_norm + 1 : e2 + 2],
        "head_layers": ops[e2 + 2 : scorer_norm],
        "scorer": ops[scorer_norm:cls_idx],
        "cls_slice": ops[cls_idx:],
    }
    rot = sum(1 for o in seg["encoder_stack"] if base_name(o).startswith("RotaryEmbedding"))
    out = {"ops_in_pass": len(ops), "total_us": sum(o["us"] for o in ops), "segments": {}}
    for k, v in seg.items():
        out["segments"][k] = {"ops": len(v), "us": round(sum(o["us"] for o in v), 1)}
    out["rotary_ops_in_stack"] = rot
    out["layers_inferred_from_rotary"] = rot // 2
    out["layer_us_mean"] = out["segments"]["encoder_stack"]["us"] / N_LAYERS
    out["head_layer_us_mean"] = out["segments"]["head_layers"]["us"] / N_HEAD_LAYERS
    out["scorer_us"] = out["segments"]["scorer"]["us"]
    out["other_us"] = sum(
        out["segments"][k]["us"] for k in ("masks", "embeddings", "final_norm", "type_add", "cls_slice")
    )
    out["lower_bound_ms"] = out["total_us"] / 1000.0
    out["layers_plus_head_plus_scorer_ms"] = (
        N_LAYERS * out["layer_us_mean"] + N_HEAD_LAYERS * out["head_layer_us_mean"] + out["scorer_us"]
    ) / 1000.0
    return out


def group_ops(ops):
    groups = defaultdict(lambda: {"n": 0, "us": 0.0, "cores": set(), "attrs": set()})
    total = sum(o["us"] for o in ops)
    for o in ops:
        g = groups[o["op"]]
        g["n"] += 1
        g["us"] += o["us"]
        g["cores"].add(o["cores"])
        if o["fidelity"] or o["in0"]:
            g["attrs"].add(f"{o['fidelity']} {o['in0']}x{o['in1']}".strip())
    rows = []
    for k, g in sorted(groups.items(), key=lambda kv: -kv[1]["us"]):
        rows.append(
            {
                "op": k,
                "n": g["n"],
                "us": round(g["us"], 1),
                "share_pct": round(100.0 * g["us"] / total, 1) if total else None,
                "cores": sorted(c for c in g["cores"] if c),
                "attrs": sorted(g["attrs"]),
            }
        )
    return rows


def cmd_reconcile(a):
    ops_all = read_ops(a.perf_csv)
    ops = last_pass(ops_all, a.repeats)
    seg = segment(ops)
    out = {
        "perf_csv": a.perf_csv,
        "ops_total_in_csv": len(ops_all),
        "repeats": a.repeats,
        "segments": seg,
        "groups": group_ops(ops),
        "traced_ms": a.traced_ms,
    }
    if a.traced_ms and "lower_bound_ms" in seg:
        out["gap_ms"] = a.traced_ms - seg["lower_bound_ms"]
        out["gap_pct"] = 100.0 * (a.traced_ms - seg["lower_bound_ms"]) / a.traced_ms
        out["gap_vs_layers_head_scorer_ms"] = a.traced_ms - seg["layers_plus_head_plus_scorer_ms"]
    if a.profile_json and os.path.exists(a.profile_json):
        out["profile"] = json.load(open(a.profile_json))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    print("RECONCILE", json.dumps({k: v for k, v in out.items() if k not in ("groups", "profile")}))


def _load(path):
    with open(path) as f:
        return json.load(f)


def cmd_summary(a):
    """Assemble perf_summary.json and latency_table.json from the stage 7 evidence files."""
    deploy = _load(a.deployment)
    fresh = {}
    for p in sorted(glob.glob(a.fresh_glob)) if a.fresh_glob else []:
        d = _load(p)
        for n, c in d["cells"].items():
            fresh[n] = {
                "cell": c,
                "file": os.path.relpath(p, AUTOPORT),
                "engine_load_seconds": d.get("engine_load_seconds"),
                "warmup_seconds": d.get("warmup_seconds"),
            }
    tp = _load(a.throughput) if a.throughput and os.path.exists(a.throughput) else None
    replay = _load(a.replay) if a.replay and os.path.exists(a.replay) else None
    all_b = _load(a.all_buckets) if a.all_buckets and os.path.exists(a.all_buckets) else None
    recon = {}
    for p in sorted(glob.glob(a.reconcile_glob)) if a.reconcile_glob else []:
        recon[os.path.basename(p).replace(".json", "")] = _load(p)
    fid = _load(a.fidelity) if a.fidelity and os.path.exists(a.fidelity) else None
    agree = _load(a.agreement) if a.agreement and os.path.exists(a.agreement) else None
    ab = {}
    for p in sorted(glob.glob(a.ab_glob)) if a.ab_glob else []:
        d = _load(p)
        for name, v in d["variants"].items():
            ab[name] = {
                "port_overrides": {k: vv for k, vv in v["port"].items() if vv != deploy["port"].get(k)},
                "error": v.get("error"),
                "cells": {
                    k: {
                        "traced_ms_p50": c.get("traced_ms_p50"),
                        "eager_ms_p50": c.get("eager_ms_p50"),
                        "loadavg": c.get("loadavg"),
                    }
                    for k, c in v["buckets"].items()
                },
            }
    cells = {}
    for n in ("1", "5", "10", "50"):
        dc = deploy["cells"].get(n)
        fc = fresh.get(n, {}).get("cell")
        cells[n] = {
            "questions": int(n),
            "bucket": dc["bucket"] if dc else None,
            "row_lengths": dc["row_lengths"] if dc else None,
            "deployment_process": {
                "end_to_end_ms_p50": dc["end_to_end_ms"]["p50"],
                "end_to_end_ms_min": dc["end_to_end_ms"]["min"],
                "end_to_end_ms_p95": dc["end_to_end_ms"]["p95"],
                "device_ms_p50": dc["device_ms"]["p50"],
                "host_tail_ms_p50": dc["host_tail_ms"]["p50"],
                "decode_ms_p50": dc["decode_ms"]["p50"],
                "replay_only_ms_p50": dc["blocking_split"].get("replay_ms_p50"),
                "write_ms_p50": dc["blocking_split"].get("write_ms_p50"),
                "readback_ms_p50": dc["blocking_split"].get("readback_ms_p50"),
                "loadavg": dc["loadavg"],
            }
            if dc
            else None,
            "fresh_process": {
                "end_to_end_ms_p50": fc["end_to_end_ms"]["p50"],
                "end_to_end_ms_min": fc["end_to_end_ms"]["min"],
                "end_to_end_ms_p95": fc["end_to_end_ms"]["p95"],
                "device_ms_p50": fc["device_ms"]["p50"],
                "host_tail_ms_p50": fc["host_tail_ms"]["p50"],
                "replay_only_ms_p50": fc["blocking_split"].get("replay_ms_p50"),
                "loadavg": fc["loadavg"],
                "file": fresh[n]["file"],
            }
            if fc
            else None,
            "t4_published_ms": T4_PUBLISHED_MS[int(n)],
            "build0_host_served_client_ms": BUILD0_CLIENT_MS[int(n)],
            "build0_host_served_device_ms": BUILD0_DEVICE_MS[int(n)],
            "build0_bucket": BUILD0_BUCKETS[int(n)],
        }
        if dc and fc:
            cells[n]["multi_trace_penalty_ms"] = dc["end_to_end_ms"]["p50"] - fc["end_to_end_ms"]["p50"]
            cells[n]["multi_trace_penalty_pct"] = (
                100.0 * (dc["end_to_end_ms"]["p50"] - fc["end_to_end_ms"]["p50"]) / fc["end_to_end_ms"]["p50"]
            )
    per_bucket = {}
    if all_b:
        v = list(all_b["variants"].values())[0]
        r = v.get("runner", {})
        for k, c in v["buckets"].items():
            per_bucket[k] = {
                "traced_ms_p50": c.get("traced_ms_p50"),
                "traced_ms_min": c.get("traced_ms_min"),
                "eager_ms_p50": c.get("eager_ms_p50"),
                "rows_per_s": c.get("rows_per_s"),
                "traced_vs_eager_max_abs_logits": c.get("traced_vs_eager_max_abs_logits"),
                "trace_bytes": r.get("trace_bytes", {}).get(k),
                "plan": {
                    kk: c["plan"][kk]
                    for kk in ("rows", "attention_memory", "mlp_sharded", "mlp_width", "qkv_minimal", "wo_minimal")
                },
                "loadavg": c.get("loadavg"),
            }
    throughput = {}
    if tp:
        for k, c in tp["throughput"].items():
            throughput[k] = {
                "bucket": c["bucket"],
                "rows": c["rows"],
                "end_to_end_ms_p50": c["end_to_end_ms"]["p50"],
                "rows_per_s": c["rows_per_s_p50"],
                "tokens_per_s_real": c["tokens_per_s_p50"],
                "padded_tokens_per_s": c["padded_tokens_per_s_p50"],
                "replay_only_ms_p50": c["blocking_split"].get("replay_ms_p50"),
                "rows_per_s_device_only": 1000.0 * c["rows"] / c["blocking_split"]["replay_ms_p50"]
                if c["blocking_split"].get("replay_ms_p50")
                else None,
                "host_tail_ms_p50": c["host_tail_ms"]["p50"],
                "loadavg": c["loadavg"],
            }
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "policy": deploy["policy"],
        "port": deploy["port"],
        "row_buckets": deploy["row_buckets"],
        "seq_buckets": deploy["seq_buckets"],
        "buckets_captured": deploy["runner"]["captured"],
        "trace_bytes_per_bucket": deploy["runner"]["trace_bytes"],
        "trace_bytes_total": deploy["runner"]["trace_bytes_total"],
        "trace_region_bytes": deploy["runner"]["trace_region_bytes"],
        "trace_region_fits": deploy["runner"]["trace_bytes_total"] <= deploy["runner"]["trace_region_bytes"],
        "warmup_seconds": deploy["warmup_seconds"],
        "warmup_build_seconds_per_bucket": deploy["runner"].get("build_seconds"),
        "warmup_eager_seconds_per_bucket": deploy["runner"].get("eager_seconds"),
        "warmup_capture_seconds_per_bucket": deploy["runner"].get("capture_seconds"),
        "engine_load_seconds": deploy.get("engine_load_seconds"),
        "engine_shapes": {k: v for k, v in deploy["engine"].items() if k not in ("policy", "port", "trace_bytes")},
        "published_cells": cells,
        "per_bucket": per_bucket,
        "throughput_b64": throughput,
        "reconciliation": recon,
        "trace_safety_all_buckets": {
            "file": os.path.relpath(a.replay, AUTOPORT) if replay else None,
            "pass": replay.get("pass") if replay else None,
            "tracker": replay.get("TT_METAL_TRACE_ALLOC_TRACKING") if replay else None,
            "unsafe_allocation_error": replay.get("unsafe_allocation_error") if replay else None,
            "buckets": len(replay.get("buckets", [])) if replay else 0,
            "trace_bytes_total": replay.get("trace_bytes_total") if replay else None,
        },
        "stage6_gates_on_final_configuration": {
            "fidelity_file": os.path.relpath(a.fidelity, AUTOPORT) if fid else None,
            "gates": fid.get("gates") if fid else None,
            "reported_not_gated": fid.get("reported_not_gated") if fid else None,
            "bucket_histogram": fid.get("bucket_histogram") if fid else None,
            "pass": fid.get("pass") if fid else None,
            "agreement_file": os.path.relpath(a.agreement, AUTOPORT) if agree else None,
            "agreement_gates": agree.get("gates") if agree else None,
            "agreement_buckets": agree.get("buckets") if agree else None,
            "agreement_pass": agree.get("pass") if agree else None,
        },
        "ab": ab,
        "loadavg_deployment_run": {"start": deploy.get("loadavg_start"), "end": deploy.get("loadavg_end")},
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    table = {
        "timestamp": summary["timestamp"],
        "protocol": deploy["protocol"],
        "policy": deploy["policy"]["name"],
        "cells": cells,
        "throughput_b64": throughput,
        "notes": [
            "deployment process: all buckets of the deployment set captured before the first timed call",
            "fresh process: one process per cell with only that cell's bucket captured",
            "end_to_end_ms = input write + trace replay + two readbacks + host tail + temperature softmax, measured in process without HTTP",
            "T4 values are the model card's; build 0 values are the host-served client and device p50 of host_tt_p150_b0_20261005T225931Z at seq bucket 512",
        ],
    }
    with open(a.latency_out, "w") as f:
        json.dump(table, f, indent=1)
    print("PERF_SUMMARY_WRITTEN", a.out, a.latency_out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--policy", default=None)
    p.add_argument("--port", default="{}")
    p.add_argument("--inputs", choices=["speed_table", "typed_decisions"], default="speed_table")
    p.add_argument("--out", default=None)
    r = sub.add_parser("reconcile")
    r.add_argument("--perf-csv", required=True)
    r.add_argument("--repeats", type=int, default=2)
    r.add_argument("--traced-ms", type=float, default=None)
    r.add_argument("--profile-json", default=None)
    r.add_argument("--out", default=None)
    s = sub.add_parser("summary")
    s.add_argument("--deployment", required=True)
    s.add_argument("--fresh-glob", default=os.path.join(DOC_DIR, "latency_fresh_*.json"))
    s.add_argument("--throughput", default=os.path.join(DOC_DIR, "latency_throughput.json"))
    s.add_argument("--replay", default=os.path.join(DOC_DIR, "replay_trace_check_all_buckets.json"))
    s.add_argument("--all-buckets", default=os.path.join(DOC_DIR, "bench_all_buckets_final.json"))
    s.add_argument("--reconcile-glob", default=os.path.join(DOC_DIR, "reconcile_*.json"))
    s.add_argument("--fidelity", default=None)
    s.add_argument("--agreement", default=None)
    s.add_argument("--ab-glob", default=os.path.join(DOC_DIR, "ab", "bench_*.json"))
    s.add_argument("--out", default=os.path.join(DOC_DIR, "perf_summary.json"))
    s.add_argument("--latency-out", default=os.path.join(DOC_DIR, "latency_table.json"))
    a = ap.parse_args(argv)
    {"profile": cmd_profile, "reconcile": cmd_reconcile, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Datatype-sweep driver for Kokoro-82M plbert full model.

Runs the candidate matrix ONE device job at a time (subprocess per candidate;
see $tt-device-usage), then aggregates candidates/<id>.json into
sweep_results.json + sweep_results.csv.

The matrix covers the material matmul groups (attention QKV/dense, MLP FF1/FF2,
embed->hidden map) across weight dtype (bf16/BFP8/BFP4) x compute fidelity
(LoFi/HiFi2/HiFi4) and the CCL payload dtype (all_gather / reduce_scatter). KV
cache is N/A (non-autoregressive bidirectional encoder), so there is no KV-cache
dtype axis. Every material BFP4 group has a BFP4+LoFi candidate (skill rule).

  python sweep_driver.py            # run missing candidates, then aggregate
  python sweep_driver.py --aggregate-only
  python sweep_driver.py --only baseline,bfp8_lofi
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CAND = HERE / "candidates"
LOGS = HERE / "logs"
REPO = HERE.parents[4]  # tt-metal root

# id -> spec. policy/opt are kwargs to PrecisionPolicy / OptConfig.
CANDIDATES = [
    (
        "baseline",
        {
            "policy": {},
            "opt": {},
            "desc": "SELECTED baseline: attn/mlp/map=BFP8, HiFi2, fp32acc, CCL bf16 (current default)",
        },
    ),
    (
        "bf16_weights",
        {
            "policy": {"attn_weight": "bf16", "mlp_weight": "bf16", "map_weight": "bf16"},
            "opt": {},
            "desc": "All linear weights BF16 (higher-precision reference), HiFi2",
        },
    ),
    (
        "bfp8_lofi",
        {
            "policy": {"matmul_fidelity": "LoFi"},
            "opt": {},
            "desc": "BFP8 weights + LoFi matmul fidelity (fidelity sweep on dominant BFP8 projections)",
        },
    ),
    (
        "bfp8_hifi4",
        {
            "policy": {"matmul_fidelity": "HiFi4"},
            "opt": {},
            "desc": "BFP8 weights + HiFi4 matmul fidelity (higher-fidelity comparison)",
        },
    ),
    (
        "mlp_bfp4_lofi",
        {
            "policy": {"mlp_weight": "bfp4", "matmul_fidelity": "LoFi"},
            "opt": {},
            "desc": "MLP FF1/FF2 weights BFP4 + LoFi (required BFP4+LoFi for the MLP group)",
        },
    ),
    (
        "mlp_bfp4_hifi2",
        {
            "policy": {"mlp_weight": "bfp4", "matmul_fidelity": "HiFi2"},
            "opt": {},
            "desc": "MLP FF1/FF2 weights BFP4 + HiFi2 (BFP4 vs LoFi comparison on the MLP group)",
        },
    ),
    (
        "attn_bfp4_lofi",
        {
            "policy": {"attn_weight": "bfp4", "matmul_fidelity": "LoFi"},
            "opt": {},
            "desc": "Attention QKV/dense weights BFP4 + LoFi (required BFP4+LoFi for the attention group)",
        },
    ),
    (
        "all_bfp4_lofi",
        {
            "policy": {"attn_weight": "bfp4", "mlp_weight": "bfp4", "map_weight": "bfp4", "matmul_fidelity": "LoFi"},
            "opt": {},
            "desc": "All linear weights BFP4 + LoFi (aggressive lower-precision win attempt)",
        },
    ),
    (
        "ccl_bfp8",
        {
            "policy": {},
            "opt": {"ag_dtype": "bfp8", "rs_dtype": "bfp8"},
            "desc": "BFP8 CCL payload for both all_gather + reduce_scatter (halve ethernet payload)",
        },
    ),
    (
        "ccl_ag_bfp8",
        {
            "policy": {},
            "opt": {"ag_dtype": "bfp8"},
            "desc": "BFP8 all_gather payload only (RS bf16); AG-only per OptConfig note",
        },
    ),
]


def run_matrix(only=None):
    env = dict(os.environ)
    env["TT_METAL_HOME"] = str(REPO)
    env["PYTHONPATH"] = f"{REPO}/ttnn:{REPO}"
    LOGS.mkdir(exist_ok=True)
    for cid, spec in CANDIDATES:
        if only and cid not in only:
            continue
        cj = CAND / f"{cid}.json"
        if cj.exists() and not only:
            print(f"skip {cid} (exists)", flush=True)
            continue
        print(f"=== running candidate {cid} ===", flush=True)
        log = (LOGS / f"{cid}.log").open("w")
        r = subprocess.run(
            [sys.executable, str(HERE / "run_candidate.py"), "--id", cid, "--spec", json.dumps(spec)],
            cwd=str(REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=2400,
        )
        log.close()
        print(f"    {cid} rc={r.returncode}", flush=True)


def aggregate():
    rows = []
    for cid, _ in CANDIDATES:
        cj = CAND / f"{cid}.json"
        if not cj.exists():
            print(f"WARN missing {cid}.json", flush=True)
            continue
        rows.append(json.loads(cj.read_text()))

    (HERE / "sweep_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fields = [
        "id",
        "desc",
        "status",
        "policy_label",
        "opt_label",
        "attn_w",
        "mlp_w",
        "map_w",
        "matmul_fidelity",
        "sdpa_fidelity",
        "fp32acc",
        "ag_ccl",
        "rs_ccl",
        "kv_cache_dtype",
        "prefill_top1",
        "prefill_top5",
        "prefill_top100",
        "tf_top1",
        "tf_top5",
        "tf_top100",
        "pcc_min",
        "ttft_ms",
        "tf_decode_t_s_u",
        "tokenout_T128_t_s_u",
        "tokenout_T512_t_s_u",
        "measurement_regime",
        "hardware",
        "mesh",
        "command",
    ]
    with (HERE / "sweep_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            pol = r.get("policy", {})
            opt = r.get("opt", {})
            tf = r.get("teacher_forcing", {})
            pre = r.get("prefill_check", {})
            to = r.get("token_out", {})
            w.writerow(
                {
                    "id": r["id"],
                    "desc": r.get("desc", ""),
                    "status": r["status"],
                    "policy_label": r.get("policy_label", ""),
                    "opt_label": r.get("opt_label", ""),
                    "attn_w": pol.get("attn_weight"),
                    "mlp_w": pol.get("mlp_weight"),
                    "map_w": pol.get("map_weight"),
                    "matmul_fidelity": pol.get("matmul_fidelity"),
                    "sdpa_fidelity": pol.get("sdpa_fidelity"),
                    "fp32acc": pol.get("fp32_dest_acc"),
                    "ag_ccl": opt.get("ag_dtype"),
                    "rs_ccl": opt.get("rs_dtype"),
                    "kv_cache_dtype": "N/A (no KV cache)",
                    "prefill_top1": pre.get("top1"),
                    "prefill_top5": pre.get("top5"),
                    "prefill_top100": pre.get("top100"),
                    "tf_top1": tf.get("top1"),
                    "tf_top5": tf.get("top5"),
                    "tf_top100": tf.get("top100"),
                    "pcc_min": r.get("pcc_vs_hf", {}).get("min"),
                    "ttft_ms": tf.get("ttft_ms"),
                    "tf_decode_t_s_u": tf.get("decode_t_s_u"),
                    "tokenout_T128_t_s_u": to.get("T128_t_s_u"),
                    "tokenout_T512_t_s_u": to.get("T512_t_s_u"),
                    "measurement_regime": "traced teacher-forcing decode t/s/u (ranking); warmed min-of-3x30 traced token-out",
                    "hardware": r.get("hardware", ""),
                    "mesh": r.get("mesh", ""),
                    "command": r.get("command", ""),
                }
            )
    print(f"aggregated {len(rows)} candidates -> sweep_results.json + .csv", flush=True)
    # quick ranked table
    passing = [r for r in rows if r["status"] == "pass"]
    passing.sort(key=lambda r: -(r.get("teacher_forcing", {}).get("decode_t_s_u") or 0))
    print("\nPASSING (ranked by traced TF decode t/s/u):", flush=True)
    for r in passing:
        tf = r["teacher_forcing"]
        print(
            f"  {r['id']:<16} tf_decode={tf['decode_t_s_u']:.1f} t/s/u  top1={tf['top1']:.4f} top5={tf['top5']:.4f}",
            flush=True,
        )
    print("\nFAIL/ERROR:", flush=True)
    for r in rows:
        if r["status"] != "pass":
            tf = r.get("teacher_forcing", {})
            print(
                f"  {r['id']:<16} status={r['status']} top1={tf.get('top1')} top5={tf.get('top5')} pcc={r.get('pcc_vs_hf',{}).get('min')} err={r.get('error')}",
                flush=True,
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated candidate ids")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    if not args.aggregate_only:
        run_matrix(only=only)
    aggregate()

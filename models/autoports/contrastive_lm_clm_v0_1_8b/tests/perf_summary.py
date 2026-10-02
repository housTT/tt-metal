# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
AUTOPORT = os.path.dirname(HERE)
N_LAYERS = 36


def layer_pass_times(perf_csv):
    rows = [r for r in csv.DictReader(open(perf_csv)) if r["Device Time"]]
    passes = []
    current = []
    for r in rows:
        current.append(r)
        if r["OP Code"].startswith("BinaryNg") and len(current) >= 20:
            passes.append(current)
            current = []
    out = {}
    for p in passes:
        dev = sum(float(r["Device Time"]) for r in p) / 1000.0
        has_1024 = any("1024" in (r.get("OP Code") or "") for r in p)
        out.setdefault(1024 if has_1024 else 128, []).append(dev)
    return {k: sum(v) / len(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument(
        "--perf-csv",
        default=os.path.join(AUTOPORT, "doc", "functional_decoder", "tracy", "layer0", "prefill_perf_report.csv"),
    )
    ap.add_argument("--out", default=os.path.join(AUTOPORT, "doc", "optimized_full_model", "perf_summary.json"))
    ap.add_argument(
        "--layer-ms",
        default="128=1.557,1024=4.519",
        help="per-layer device ms by padded length from the Tracy op-count split of doc/functional_decoder/tracy/layer0 (accuracy policy)",
    )
    a = ap.parse_args()
    bench = json.load(open(a.bench))
    layer_ms = (
        {int(k): float(v) for k, v in (kv.split("=") for kv in a.layer_ms.split(","))}
        if a.layer_ms
        else (layer_pass_times(a.perf_csv) if os.path.exists(a.perf_csv) else {})
    )
    rows = []
    for r in bench["rows"]:
        lb = layer_ms.get(r["padded"]) if r["batch"] == 1 else None
        row = dict(r)
        if lb:
            row["layer_stack_lower_bound_ms"] = round(N_LAYERS * lb, 2)
            row["gap_to_lower_bound_pct"] = round(100.0 * (r["p50_ms"] - N_LAYERS * lb) / r["p50_ms"], 1)
        rows.append(row)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_bench": os.path.relpath(a.bench, AUTOPORT),
        "precision": bench.get("precision"),
        "device": bench.get("device_name"),
        "mesh": bench.get("mesh"),
        "model_load_seconds": bench.get("load_seconds"),
        "per_layer_device_ms_from_tracy": {str(k): round(v, 3) for k, v in layer_ms.items()},
        "rows": rows,
        "notes": [
            "p50 over repeats of enc.embed_ids on warm traces, batch 1 and 4 and 8, host readback and host RMSNorm included",
            "lower bound = 36 x per-layer device time measured by the Tracy profiler on layer 0 under the accuracy policy (1.557 ms at 128 tokens, 4.519 ms at 1024 (layer ops only; the harness tilize and typecast rows are excluded)); the terminal work (embedding lookup, output readback, host norm) is the remainder; for the bfp8_attn rows the bound is an upper estimate because that policy reads fewer weight bytes",
        ],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(
        "PERF_SUMMARY",
        json.dumps(
            {
                "precision": summary["precision"],
                "rows": [
                    {
                        k: r.get(k)
                        for k in ("tokens", "batch", "p50_ms", "layer_stack_lower_bound_ms", "gap_to_lower_bound_pct")
                    }
                    for r in rows
                    if r["batch"] == 1
                ],
            }
        ),
    )


if __name__ == "__main__":
    main()

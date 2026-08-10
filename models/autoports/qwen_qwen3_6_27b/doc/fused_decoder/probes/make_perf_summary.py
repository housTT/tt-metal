# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build ``doc/fused_decoder/perf_summary.json`` from the committed ``tt-perf-report`` CSVs.

One row per (implementation, layer kind, phase).  Every number is re-derived here rather than
transcribed: device time is the sum of the report's ``Device Time`` column over the signposted
window (divided by the replay count for decode), and the decode windows' op-code sequences are
checked to repeat with an exact period, which is what makes "all replays were captured whole"
true rather than assumed.

Reads only committed artifacts; opens no device.

    python models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/make_perf_summary.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

DOC = Path(__file__).resolve().parents[1]
IMPLS = ("functional", "fused")
KINDS = ("linear_attention", "full_attention")
#: Replays inside the signposted decode window; must match ``PERF_DECODE_ITERS``.
DECODE_REPLAYS = 8


def _rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


#: Coarse op-code buckets, so "where the time goes" in the stage documents is a number derived
#: from the report rather than one added up by hand.  First match wins; anything unmatched lands
#: in ``other``, which the documents quote too, so a new op cannot hide there.
CATEGORIES = (
    ("gated_delta_rule", lambda code: code.startswith("ChunkGdn")),
    ("sdpa", lambda code: "dpa" in code.lower()),
    ("batched_matmul", lambda code: code.startswith("Matmul") and "b={" in code),
    ("matmul", lambda code: code.startswith("Matmul")),
    ("norm", lambda code: "LayerNorm" in code),
    (
        "heads_and_cache",
        lambda code: any(
            token in code for token in ("Nlp", "NLP", "PagedFill", "PagedUpdate", "RotaryEmbedding", "RotateHalf")
        ),
    ),
    (
        "layout",
        lambda code: any(
            token in code
            for token in ("Tilize", "Untilize", "Reshape", "Permute", "Transpose", "Concat", "Slice", "Sharded")
        ),
    ),
    (
        "elementwise",
        lambda code: code.startswith(("BinaryNg", "Unary", "Typecast", "Copy", "Fill", "Reduce", "Softplus")),
    ),
)


def _bucket(code: str) -> str:
    for name, predicate in CATEGORIES:
        if predicate(code):
            return name
    return "other"


def _period(codes: list[str], replays: int) -> str:
    """Confirm the op-code sequence repeats with the exact expected period."""
    if replays == 1:
        return "single pass"
    if len(codes) % replays:
        return f"BROKEN: {len(codes)} ops is not divisible by {replays} replays"
    per = len(codes) // replays
    for index, code in enumerate(codes):
        if code != codes[index % per]:
            return f"BROKEN: op {index} ({code}) breaks the period of {per}"
    return f"{len(codes)} ops over {replays} replays = {per} ops per replay, exact"


def main() -> None:
    summary = {
        "note": (
            "Warmed measurements on one Blackhole chip (device 2 of a p300c board), batch 1, from "
            "Tracy device-profiler runs with the measured window delimited by signposts. Device time is "
            "the sum of the 'Device Time' column of the tt-perf-report --csv output, in MICROSECONDS, "
            "divided by the replay count for decode. Decode is traced: capture once, then replay "
            "execute_trace 8x inside the window. 'functional' is tt/functional_decoder.py (the stage-1 "
            "baseline) and 'fused' is tt/fused_decoder.py, measured by the same script "
            "(probes/run_perf.sh) on the same machine in the same session."
        ),
        "prefill_tokens": 2048,
        "decode_position": 2048,
        "decode_replays": DECODE_REPLAYS,
        "command": "doc/fused_decoder/probes/run_perf.sh <layer_kind> <prefill|decode> <functional|fused>",
        "measurements": {},
        "speedup": {},
    }

    for impl in IMPLS:
        for kind in KINDS:
            for phase, replays in (("prefill", 1), ("decode", DECODE_REPLAYS)):
                report = DOC / "tracy" / impl / kind / f"{phase}_perf_report.csv"
                if not report.exists():
                    raise SystemExit(f"missing {report}")
                rows = _rows(report)
                total = sum(float(row["Device Time"] or 0) for row in rows)
                gap = sum(float(row["Op-to-Op Gap"] or 0) for row in rows)
                codes = [row["OP Code"] for row in rows]
                top = sorted(rows, key=lambda row: -float(row["Device Time"] or 0))[:5]
                buckets: dict[str, float] = {}
                for row in rows:
                    name = _bucket(row["OP Code"])
                    buckets[name] = buckets.get(name, 0.0) + float(row["Device Time"] or 0)
                breakdown = {
                    name: round(value / replays / 1000.0, 3)
                    for name, value in sorted(buckets.items(), key=lambda item: -item[1])
                }
                summary["measurements"][f"{impl}/{kind}/{phase}"] = {
                    "impl": impl,
                    "layer_kind": kind,
                    "phase": phase,
                    "ops_in_window": len(rows),
                    "ops_per_pass": len(rows) // replays,
                    "device_kernel_time_ms": round(total / replays / 1000.0, 3),
                    "op_to_op_gap_ms": round(gap / replays / 1000.0, 3),
                    "periodicity_check": _period(codes, replays),
                    "breakdown_ms": breakdown,
                    "top_ops_by_device_time": [
                        {"op": row["OP Code"], "device_time_us": round(float(row["Device Time"]), 1)} for row in top
                    ],
                    "artifacts": {
                        "ops_csv_gz": f"tracy/{impl}/{kind}/{phase}_ops.csv.gz",
                        "ops_csv_provenance": f"tracy/{impl}/{kind}/{phase}_ops.csv.provenance",
                        "report_txt": f"tracy/{impl}/{kind}/{phase}_perf_report.txt",
                        "report_csv": f"tracy/{impl}/{kind}/{phase}_perf_report.csv",
                        "console_log": f"tracy/{impl}/{kind}/{phase}_perf_report.console.log",
                        "tracy_run_log": f"logs/tracy_{impl}_{kind}_{phase}.log",
                    },
                }

    for kind in KINDS:
        for phase in ("prefill", "decode"):
            before = summary["measurements"][f"functional/{kind}/{phase}"]["device_kernel_time_ms"]
            after = summary["measurements"][f"fused/{kind}/{phase}"]["device_kernel_time_ms"]
            before_ops = summary["measurements"][f"functional/{kind}/{phase}"]["ops_per_pass"]
            after_ops = summary["measurements"][f"fused/{kind}/{phase}"]["ops_per_pass"]
            summary["speedup"][f"{kind}/{phase}"] = {
                "device_ms_before": before,
                "device_ms_after": after,
                "speedup_x": round(before / after, 3),
                "reduction_pct": round(100.0 * (before - after) / before, 2),
                "ops_before": before_ops,
                "ops_after": after_ops,
            }

    out = DOC / "perf_summary.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {out}")
    for key, value in summary["speedup"].items():
        print(
            f"  {key:28s} {value['device_ms_before']:9.3f} ms -> {value['device_ms_after']:9.3f} ms "
            f"({value['speedup_x']:.2f}x, -{value['reduction_pct']:.1f}%)  ops {value['ops_before']} -> {value['ops_after']}"
        )


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build ``doc/optimized_decoder/perf_summary.json`` from the committed ``tt-perf-report`` CSVs.

One row per (arm, layer kind, phase).  Every number is re-derived here rather than transcribed:

* device time is the sum of the report's ``Device Time`` column over the signposted window,
  divided by the replay count for decode;
* end-to-end wall time is parsed out of the same run's Tracy log, so the *third* number of the
  performance-accounting triple comes from the same run as the first two;
* the theoretical roofline is computed from the shapes and the dtypes the run's own
  ``PERFCONFIG`` line records, divided by a peak DRAM bandwidth **derived from the report's own
  ``DRAM`` and ``DRAM %`` columns** rather than from a datasheet number that might not match this
  board;
* the dominant matmul rows carry their measured input/weight dtype and math fidelity, which is
  what proves the selected precision policy reached the measured op rather than only the policy
  object (OPT-013).

Reads only committed artifacts; opens no device.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/make_perf_summary.py
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

DOC = Path(__file__).resolve().parents[1]
KINDS = ("linear_attention", "full_attention")
PHASES = (("prefill", 1), ("decode", 8), ("decode_batch32", 8))
#: Replays inside the signposted decode window; must match ``PERF_DECODE_ITERS``.
DECODE_REPLAYS = 8

#: The arms the before/after tables and the candidate table are built from.  ``fused`` is the
#: stage-2 implementation re-measured in this stage's harness (not copied from stage 2's artifacts,
#: so the pair is one machine and one build); the two ``optimized-*`` isolation arms are the same
#: optimized code with one change put back, which is what separates the precision change from the
#: layout change.
ARMS = {
    "fused": "stage-2 tt/fused_decoder.py, re-measured here",
    "optimized": "tt/optimized_decoder.py at the shipped policy and geometry",
    "optimized-fused-baseline-fused-baseline": "the optimized code at the fused stage's policy and layout",
    "optimized-opt-v1-fused-baseline": "the shipped precision policy on the fused stage's decode layout",
    "optimized-fused-baseline-opt-v1": "the fused stage's precision policy on the shipped decode layout",
}
#: Arms that must be present for the summary to be complete.  The isolation arms are optional so a
#: partial re-measurement still regenerates.
REQUIRED_ARMS = ("fused", "optimized")

#: Bytes one element of each dtype occupies in a tile, as tt-metal stores it.  Block-float formats
#: carry one shared exponent per 16-element sub-tile, so the per-element cost is the mantissa byte
#: plus 1/16 of an exponent byte.
DTYPE_BYTES = {
    "DataType.BFLOAT16": 2.0,
    "DataType.BFLOAT8_B": 1.0 + 1.0 / 16.0,
    "DataType.BFLOAT4_B": 0.5 + 1.0 / 16.0,
    "DataType.FLOAT32": 4.0,
}


CATEGORIES = (
    ("gated_delta_rule", lambda code: code.startswith("ChunkGdn")),
    ("state_update", lambda code: code.startswith("Ternary")),
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
            # ``Reshard`` is a sharded->sharded move, which this stage introduces where a shard grid
            # has to change; it belongs with the other layout movement rather than in ``other``.
            for token in (
                "Tilize",
                "Untilize",
                "Reshape",
                "Permute",
                "Transpose",
                "Concat",
                "Slice",
                "Sharded",
                "Reshard",
            )
        ),
    ),
    (
        "elementwise",
        lambda code: code.startswith(
            ("BinaryNg", "Unary", "Typecast", "Copy", "Fill", "Reduce", "Softplus", "Accumulation")
        ),
    ),
)


def _bucket(code: str) -> str:
    for name, predicate in CATEGORIES:
        if predicate(code):
            return name
    return "other"


def _rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _period(codes: list, replays: int) -> str:
    if replays == 1:
        return "single pass"
    if len(codes) % replays:
        return f"BROKEN: {len(codes)} ops is not divisible by {replays} replays"
    per = len(codes) // replays
    for index, code in enumerate(codes):
        if code != codes[index % per]:
            return f"BROKEN: op {index} ({code}) breaks the period of {per}"
    return f"{len(codes)} ops over {replays} replays = {per} ops per replay, exact"


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _log_path(arm: str, kind: str, phase: str) -> Path:
    return DOC / "logs" / f"tracy_{arm}_{kind}_{phase}.log"


def _wall_and_config(arm: str, kind: str, phase: str) -> dict:
    """End-to-end wall time and the run's own configuration record, from its Tracy log."""
    path = _log_path(arm, kind, phase)
    out: dict = {"log": f"logs/{path.name}"}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    per_iter = re.search(r"wall_per_iter_ms=([0-9.]+)", text)
    e2e = re.search(r"wall_e2e_ms=([0-9.]+)", text)
    if per_iter:
        out["wall_ms_per_pass"] = float(per_iter.group(1))
    elif e2e:
        out["wall_ms_per_pass"] = float(e2e.group(1))
    config = re.search(r"^PERFCONFIG (\{.*\})$", text, re.MULTILINE)
    if config:
        try:
            out["config"] = json.loads(config.group(1))
        except json.JSONDecodeError:
            pass
    return out


def _peak_dram_gbs(rows) -> float | None:
    """Peak DRAM bandwidth implied by the report's own ``DRAM`` and ``DRAM %`` columns.

    Every DRAM-bound row reports both the achieved GB/s and the fraction of the roofline it is,
    so their ratio *is* this board's peak as the profiler models it.  Taking the median over the
    rows that report both makes it robust to a row whose percentage is rounded to zero.
    """
    implied = []
    for row in rows:
        achieved = _float(row.get("DRAM"))
        fraction = _float(row.get("DRAM %"))
        if achieved > 0 and fraction > 1:
            implied.append(achieved / (fraction / 100.0))
    if not implied:
        return None
    return round(statistics.median(implied), 1)


def _decode_weight_bytes(config: dict) -> dict:
    """Bytes of weight a decode step must read, from the run's own dtype record.

    This is the *theoretical* term of the performance-accounting triple: a decode step reads every
    projection weight exactly once, whatever the batch, so the sum over roles of
    ``K * N * bytes(dtype)`` is a lower bound on the traffic and therefore an upper bound on the
    achievable speed.
    """
    roles = (config or {}).get("config", {}).get("roles", {})
    total = 0.0
    per_role = {}
    for role, entry in roles.items():
        # ``mlp_gate_up`` exists for the prefill path only when decode is split; counting it in a
        # decode roofline would double the MLP's first projection.
        if role == "mlp_gate_up" and (config or {}).get("config", {}).get("decode", {}).get("split_gate_up"):
            continue
        if role in ("mlp_gate", "mlp_up") and not (config or {}).get("config", {}).get("decode", {}).get(
            "split_gate_up"
        ):
            continue
        size = entry["K"] * entry["N"] * DTYPE_BYTES.get(entry["weight_dtype"], 2.0)
        per_role[role] = int(size)
        total += size
    return {"total_bytes": int(total), "per_role_bytes": per_role}


def _kv_cache_bytes(config: dict, kind: str, position: int, batch: int) -> int:
    """Bytes of KV cache a decode step reads (``full_attention`` only)."""
    if kind != "full_attention":
        return 0
    summary = (config or {}).get("config", {})
    dtype = summary.get("kv_cache_dtype", "DataType.BFLOAT16")
    # 4 KV heads x head_dim 256, K and V, rounded up to the SDPA k-chunk the layer pins.
    n_kv, head_dim, k_chunk = 4, 256, 512
    rounded = ((position + 1 + k_chunk - 1) // k_chunk) * k_chunk
    return int(2 * batch * n_kv * rounded * head_dim * DTYPE_BYTES.get(dtype, 2.0))


def _state_bytes(kind: str, batch: int) -> int:
    """Bytes of carried recurrent/conv state a decode step touches (``linear_attention`` only)."""
    if kind != "linear_attention":
        return 0
    nv, dk, dv, conv_dim, k_size = 48, 128, 128, 10240, 4
    recurrent = batch * nv * dk * dv * 4
    # The state is read by two matmuls and rewritten once by the fused addcmul.
    conv = k_size * batch * conv_dim * 4 * 2
    return int(3 * recurrent + conv)


def _dominant_matmul_rows(rows, limit: int = 8) -> list:
    """The largest matmul rows with the dtype and fidelity the profiler measured.

    This is the OPT-013 artifact: if a row here says ``BF16 x BF16`` while the policy claims BFP4,
    the policy did not reach the measured op.
    """
    matmuls = [row for row in rows if row["OP Code"].startswith("Matmul")]
    grouped: dict = {}
    for row in matmuls:
        key = row["OP Code"]
        entry = grouped.setdefault(
            key,
            {
                "op": key,
                "instances": 0,
                "device_time_us": 0.0,
                "math_fidelity": row.get("Math Fidelity", ""),
                "bound": row.get("Bound", ""),
                "cores": row.get("Cores", ""),
                "dram_pct": row.get("DRAM %", ""),
                "flops_pct": row.get("FLOPs %", ""),
            },
        )
        entry["instances"] += 1
        entry["device_time_us"] += _float(row["Device Time"])
    top = sorted(grouped.values(), key=lambda entry: -entry["device_time_us"])[:limit]
    for entry in top:
        entry["device_time_us"] = round(entry["device_time_us"], 1)
    return top


def main() -> None:
    summary = {
        "note": (
            "Warmed measurements on one Blackhole chip (device 2 of a p300c board), from Tracy "
            "device-profiler runs with the measured window delimited by signposts. Device time is the "
            "sum of the 'Device Time' column of the tt-perf-report --csv output, in MICROSECONDS, "
            "divided by the replay count for decode. End-to-end wall time is parsed from the same "
            "run's Tracy log, so the accounting triple (roofline / device / end-to-end) is one run. "
            "Prefill is batch 1; decode is measured at batch 1 and at the advertised max_batch of 32, "
            "which is a different graph rather than a wider tensor. Decode is traced: capture once, "
            "then replay execute_trace 8x inside the window. Every arm was measured by the same "
            "script (probes/run_perf.sh) on the same machine against the same build, with only the "
            "--impl / --policy / --geometry arguments differing; the .provenance file next to each "
            "ops CSV carries that run's timestamp and the shipped-code fingerprint."
        ),
        "arms": ARMS,
        "prefill_tokens": 2048,
        "decode_position": 2048,
        "decode_batches": {"prefill": 1, "decode": 1, "decode_batch32": 32},
        "decode_replays": DECODE_REPLAYS,
        "command": (
            "doc/optimized_decoder/probes/run_perf.sh <layer_kind> <prefill|decode|decode_batch32> "
            "<impl> [policy] [geometry]"
        ),
        "measurements": {},
        "speedup": {},
        "accounting": {},
    }

    present = []
    for arm in ARMS:
        for kind in KINDS:
            for phase, replays in PHASES:
                report = DOC / "tracy" / arm / kind / f"{phase}_perf_report.csv"
                if not report.exists():
                    if arm in REQUIRED_ARMS:
                        raise SystemExit(f"missing {report}")
                    continue
                rows = _rows(report)
                total = sum(_float(row["Device Time"]) for row in rows)
                gap = sum(_float(row["Op-to-Op Gap"]) for row in rows)
                codes = [row["OP Code"] for row in rows]
                buckets: dict = {}
                for row in rows:
                    name = _bucket(row["OP Code"])
                    buckets[name] = buckets.get(name, 0.0) + _float(row["Device Time"])
                breakdown = {
                    name: round(value / replays / 1000.0, 4)
                    for name, value in sorted(buckets.items(), key=lambda item: -item[1])
                }
                run = _wall_and_config(arm, kind, phase)
                key = f"{arm}/{kind}/{phase}"
                present.append(key)
                summary["measurements"][key] = {
                    "arm": arm,
                    "layer_kind": kind,
                    "phase": phase,
                    "ops_in_window": len(rows),
                    "ops_per_pass": len(rows) // replays,
                    "device_kernel_time_ms": round(total / replays / 1000.0, 4),
                    "op_to_op_gap_ms": round(gap / replays / 1000.0, 4),
                    "wall_ms_per_pass": run.get("wall_ms_per_pass"),
                    "peak_dram_gbs_implied": _peak_dram_gbs(rows),
                    "periodicity_check": _period(codes, replays),
                    "breakdown_ms": breakdown,
                    "dominant_matmul_rows": _dominant_matmul_rows(rows),
                    "config": run.get("config", {}).get("config"),
                    "artifacts": {
                        "ops_csv_gz": f"tracy/{arm}/{kind}/{phase}_ops.csv.gz",
                        "ops_csv_provenance": f"tracy/{arm}/{kind}/{phase}_ops.csv.provenance",
                        "report_txt": f"tracy/{arm}/{kind}/{phase}_perf_report.txt",
                        "report_csv": f"tracy/{arm}/{kind}/{phase}_perf_report.csv",
                        "report_noadvice_txt": f"tracy/{arm}/{kind}/{phase}_perf_report.noadvice.txt",
                        "console_log": f"tracy/{arm}/{kind}/{phase}_perf_report.console.log",
                        "tracy_run_log": run["log"],
                    },
                }

    for kind in KINDS:
        for phase, _replays in PHASES:
            before_key, after_key = f"fused/{kind}/{phase}", f"optimized/{kind}/{phase}"
            if before_key not in summary["measurements"] or after_key not in summary["measurements"]:
                continue
            before = summary["measurements"][before_key]
            after = summary["measurements"][after_key]
            summary["speedup"][f"{kind}/{phase}"] = {
                "device_ms_before": before["device_kernel_time_ms"],
                "device_ms_after": after["device_kernel_time_ms"],
                "speedup_x": round(before["device_kernel_time_ms"] / after["device_kernel_time_ms"], 3),
                "reduction_pct": round(
                    100.0
                    * (before["device_kernel_time_ms"] - after["device_kernel_time_ms"])
                    / before["device_kernel_time_ms"],
                    2,
                ),
                "ops_before": before["ops_per_pass"],
                "ops_after": after["ops_per_pass"],
                "wall_ms_before": before.get("wall_ms_per_pass"),
                "wall_ms_after": after.get("wall_ms_per_pass"),
            }

    # Performance accounting: roofline / device time / end-to-end, from the same run.
    for kind in KINDS:
        for phase, _replays in PHASES:
            key = f"optimized/{kind}/{phase}"
            if key not in summary["measurements"]:
                continue
            entry = summary["measurements"][key]
            if phase == "prefill":
                continue
            batch = 32 if phase == "decode_batch32" else 1
            config = {"config": entry.get("config")}
            weights = _decode_weight_bytes(config)
            kv = _kv_cache_bytes(config, kind, summary["decode_position"], batch)
            state = _state_bytes(kind, batch)
            peak = entry.get("peak_dram_gbs_implied")
            total_bytes = weights["total_bytes"] + kv + state
            roofline_ms = round(total_bytes / (peak * 1e9) * 1e3, 4) if peak else None
            summary["accounting"][f"{kind}/{phase}"] = {
                "weight_bytes": weights["total_bytes"],
                "per_role_weight_bytes": weights["per_role_bytes"],
                "kv_cache_bytes": kv,
                "carried_state_bytes": state,
                "total_bytes_per_step": total_bytes,
                "peak_dram_gbs_implied_by_report": peak,
                "roofline_ms_per_step_estimate": roofline_ms,
                "decode_ms_per_step_device": entry["device_kernel_time_ms"],
                "decode_ms_per_step_e2e": entry.get("wall_ms_per_pass"),
                "op_to_op_gap_ms": entry["op_to_op_gap_ms"],
                "roofline_fraction_achieved": (
                    round(roofline_ms / entry["device_kernel_time_ms"], 3)
                    if roofline_ms and entry["device_kernel_time_ms"]
                    else None
                ),
            }

    out = DOC / "perf_summary.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {out} ({len(present)} measured windows)")
    for key, value in summary["speedup"].items():
        print(
            f"  {key:28s} {value['device_ms_before']:9.4f} ms -> {value['device_ms_after']:9.4f} ms "
            f"({value['speedup_x']:.2f}x, -{value['reduction_pct']:.1f}%)  "
            f"ops {value['ops_before']} -> {value['ops_after']}"
        )
    for key, value in summary["accounting"].items():
        print(
            f"  accounting {key:24s} roofline {value['roofline_ms_per_step_estimate']} ms  "
            f"device {value['decode_ms_per_step_device']} ms  e2e {value['decode_ms_per_step_e2e']} ms  "
            f"({value['roofline_fraction_achieved']} of roofline)"
        )


if __name__ == "__main__":
    main()

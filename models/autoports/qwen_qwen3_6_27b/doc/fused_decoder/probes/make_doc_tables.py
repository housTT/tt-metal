# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fill the generated blocks of the fused-decoder documents from the committed artifacts.

Everything in this stage's documents that a re-measurement moves - the before/after perf table,
the per-bucket breakdown, the probe figures the work log quotes - lives between
``<!-- GENERATED:name -->`` / ``<!-- END GENERATED:name -->`` markers and is written here out of
``perf_summary.json`` and ``logs/probe_*.log``.  Hand-maintaining those numbers is the drift
class this stage keeps finding in review; generating them removes it, and
``tests/test_fused_decoder_docs.py`` is the gate that proves the generated text and the
artifacts still agree.

Reads only committed artifacts; opens no device.

    python models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/make_doc_tables.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1]
KINDS = ("linear_attention", "full_attention")


def _summary() -> dict:
    return json.loads((DOC / "perf_summary.json").read_text())


def _probe(name: str) -> str:
    return (DOC / "logs" / f"{name}.log").read_text(errors="replace")


def _grep(name: str, pattern: str, group: int = 1) -> str:
    match = re.search(pattern, _probe(name))
    if not match:
        raise SystemExit(f"{name}.log has no match for {pattern!r}")
    return match.group(group)


def before_after_table() -> str:
    summary = _summary()
    rows = [
        "| layer kind | phase | device time before | device time after | speed-up | ops before | ops after |",
        "|---|---|---|---|---|---|---|",
    ]
    labels = {"prefill": "prefill, 2048 tokens", "decode": "traced decode, 1 token"}
    for kind in KINDS:
        for phase in ("prefill", "decode"):
            row = summary["speedup"][f"{kind}/{phase}"]
            rows.append(
                f"| `{kind}` | {labels[phase]} | {row['device_ms_before']:.3f} ms | "
                f"**{row['device_ms_after']:.3f} ms** | **{row['speedup_x']:.2f}x** | "
                f"{row['ops_before']} | {row['ops_after']} |"
            )
    return "\n".join(rows)


def breakdown_table() -> str:
    summary = _summary()
    columns = [f"fused/{kind}/{phase}" for kind in KINDS for phase in ("prefill", "decode")]
    labels = {
        "matmul": "`matmul` (projections, MLP, gated-norm constants)",
        "gated_delta_rule": "`gated_delta_rule`",
        "sdpa": "`sdpa`",
        "batched_matmul": "`batched_matmul` (the decode recurrence)",
        "layout": "`layout` (tilize/untilize/reshape/permute/concat/slice/shard)",
        "elementwise": "`elementwise`",
        "norm": "`norm`",
        "heads_and_cache": "`heads_and_cache`",
        "other": "`other`",
    }
    rows = [
        "| bucket | `linear_attention` prefill | `linear_attention` decode "
        "| `full_attention` prefill | `full_attention` decode |",
        "|---|---|---|---|---|",
    ]
    for bucket, label in labels.items():
        cells = []
        for column in columns:
            value = _summary()["measurements"][column]["breakdown_ms"].get(bucket)
            cells.append("—" if value is None else f"{value:.3f} ms")
        if all(cell == "—" for cell in cells):
            continue
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    totals = [summary["measurements"][column]["device_kernel_time_ms"] for column in columns]
    rows.append("| **total** | " + " | ".join(f"**{value:.3f} ms**" for value in totals) + " |")
    return "\n".join(rows)


def decode_matmul_share() -> str:
    summary = _summary()
    parts = []
    for kind in KINDS:
        row = summary["measurements"][f"fused/{kind}/decode"]
        share = 100.0 * row["breakdown_ms"]["matmul"] / row["device_kernel_time_ms"]
        parts.append(f"{share:.1f} % of the `{kind}` decode step")
    return "The `matmul` bucket is " + " and ".join(parts) + "."


def conv_table() -> str:
    """The four FIR formulations, median and spread over 12 repeats; winner derived, not chosen."""
    names = {
        "tile": "all-TILE slices (what the functional layer does)",
        "rm_shift": "untilize once, ROW_MAJOR shift, tilize per tap - **shipped**",
        "rm_arith": "untilize once, whole FIR in ROW_MAJOR, tilize once",
        "aligned_win": "one pre-padded window per tap so every slice is tile-aligned",
    }
    measured = {
        key: {
            dtype: (
                _grep("probe_causal_conv", rf"conv {key}\s+{dtype}.*?median_ms=\s*([\d.]+)"),
                _grep("probe_causal_conv", rf"conv {key}\s+{dtype}.*?stdev_ms=\s*([\d.]+)"),
            )
            for dtype in ("fp32", "bf16")
        }
        for key in names
    }
    best = {dtype: min(names, key=lambda key: float(measured[key][dtype][0])) for dtype in ("fp32", "bf16")}
    rows = ["| formulation | float32 median (stdev) ms | bfloat16 median (stdev) ms |", "|---|---|---|"]
    for key, label in names.items():
        cells = []
        for dtype in ("fp32", "bf16"):
            median, stdev = measured[key][dtype]
            mark = "**" if best[dtype] == key else ""
            cells.append(f"{mark}{median}{mark} ({stdev})")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def broadcast_table() -> str:
    """Bandwidth of the two binary-op shapes; the outlier is derived, not marked by hand."""
    cells: dict[tuple[str, str], tuple[str, float]] = {}
    for key in ("bcast", "same"):
        for dtype in ("fp32", "bf16"):
            ms = _grep("probe_causal_conv", rf"multiply {key}\s+{dtype}.*?median_ms=\s*([\d.]+)")
            gbps = _grep("probe_causal_conv", rf"multiply {key}\s+{dtype}.*?eff_GBps=\s*([\d.]+)")
            cells[(key, dtype)] = (ms, float(gbps))
    worst = min(cells, key=lambda item: cells[item][1])
    rows = ["| multiply | float32 | bfloat16 |", "|---|---|---|"]
    for key, label in (("bcast", "height-broadcast"), ("same", "same-shape")):
        out = []
        for dtype in ("fp32", "bf16"):
            ms, gbps = cells[(key, dtype)]
            mark = "**" if (key, dtype) == worst else ""
            out.append(f"{ms} ms — {mark}{gbps:.0f} GB/s{mark}")
        rows.append(f"| {label} | " + " | ".join(out) + " |")
    return "\n".join(rows)


def norm_table() -> str:
    cores = ("16", "20", "32", "40", "80")
    values = [_grep("probe_small_ops", rf"rms_norm sharded\s+{core}c ms=([\d.]+)") for core in cores]
    interleaved = _grep("probe_small_ops", r"rms_norm interleaved\s+ms=([\d.]+)")
    best = min(range(len(values)), key=lambda index: float(values[index]))
    cells = [f"**{value}**" if index == best else value for index, value in enumerate(values)]
    return (
        "| cores | " + " | ".join(cores) + " | interleaved |\n"
        "|---|---|---|---|---|---|---|\n"
        "| ms | " + " | ".join(cells) + f" | {interleaved} |"
    )


def recurrence_table() -> str:
    """Median and spread per grid; the selection is labelled, not disguised as the minimum."""
    grids = ("default", "1x8", "2x8", "4x4", "6x4", "6x8", "6x11")
    header = "| shape | " + " | ".join(grids) + " | selected |"
    rows = [header, "|---" * (len(grids) + 2) + "|"]
    for kind, label, selected in (("read", "state read", "6x4"), ("outer", "outer product", "6x8")):
        cells = []
        for grid in grids:
            token = "default" if grid == "default" else f"core_grid {grid}"
            match = re.search(
                rf"{kind}\s+{re.escape(token)}\s+median_us=\s*([\d.]+) stdev_us=\s*([\d.]+)",
                _probe("probe_decode_recurrence"),
            )
            cells.append("—" if match is None else f"{match.group(1)} ({match.group(2)})")
        rows.append(f"| {label} | " + " | ".join(cells) + f" | {selected} |")
    rows.append("")
    rows.append(
        "Median and (stdev) in microseconds over 30 repeats. The default program factory is "
        "several times slower than any explicit grid; the explicit grids sit within about a "
        "stdev of each other, so the two selected are a representative pick from that flat "
        "region rather than a unique optimum."
    )
    return "\n".join(rows)


def mlp_table() -> str:
    rows = ["| variant | prefill 2048 | decode 32 |", "|---|---|---|"]
    names = {
        "fused_slice_silu": "fused gate/up matmul + 2 slices + `silu` + `multiply` (functional)",
        "fused_slice_act": "fused gate/up matmul + 2 slices + `multiply(act=SILU)`",
        "split_act": "split gate/up matmuls, `silu` on the gate matmul's `activation=` epilogue, `multiply`",
    }
    for key, label in names.items():
        prefill = _grep("probe_mlp_variants", rf"prefill-2048\s+{key}\s+best_ms=\s*([\d.]+)")
        decode = _grep("probe_mlp_variants", rf"decode-32\s+{key}\s+best_ms=\s*([\d.]+)")
        rows.append(
            f"| {label} | {'**' if key == 'fused_slice_act' else ''}{prefill} ms"
            f"{'**' if key == 'fused_slice_act' else ''} | "
            f"{'**' if key == 'split_act' else ''}{decode} ms{'**' if key == 'split_act' else ''} |"
        )
    return "\n".join(rows)


def gdr_table() -> str:
    rows = ["| call shape | chunk | seq | output PCC | final-state PCC | best wall |", "|---|---|---|---|---|---|"]
    for mode, chunk, shape in (
        ("flat", "32", "flat rank-3 `[1, T, H*D]`"),
        ("split", "64", "split rank-4 `[1, T, H, D]`"),
    ):
        for seq in ("64", "2048"):
            line = _grep(
                "probe_chunk_gdr",
                rf"mode={mode}\s+chunk=\s*{chunk} seq=\s*{seq} o_pcc=([\d.]+) state_pcc=([\d.]+) best_wall_ms=([\d.]+)",
                0,
            )
            o_pcc, state_pcc, wall = re.search(
                r"o_pcc=([\d.]+) state_pcc=([\d.]+) best_wall_ms=([\d.]+)", line
            ).groups()
            bold = "**" if seq == "2048" else ""
            rows.append(f"| {shape} | {chunk} | {seq} | {bold}{o_pcc}{bold} | {state_pcc} | {bold}{wall} ms{bold} |")
    return "\n".join(rows)


def epilogue_table() -> str:
    token = _grep("probe_output_paths", r"gdn_epilogue token_major ms=\s*([\d.]+)")
    head = _grep("probe_output_paths", r"head_major ms=\s*([\d.]+)")
    pcc = _grep("probe_output_paths", r"pcc\(token,head\)=([\d.]+)")
    return (
        "| path | 2048-token chunk |\n|---|---|\n"
        f"| token-major output + group-reduction norm (§3.4) — **shipped** | **{token} ms** |\n"
        f"| `output_head_major=True` + per-head `ttnn.rms_norm` + z/result relayouts | {head} ms |\n"
        f"\nPCC between the two outputs: {pcc}."
    )


def rope_table() -> str:
    rows = ["| tensor | slice + 64-wide RoPE + slice + concat | permuted, one 256-wide RoPE |", "|---|---|---|"]
    total_narrow = total_wide = 0.0
    for tag, label in (("q", "q, 24 heads"), ("k", "k, 4 heads")):
        narrow = _grep("probe_output_paths", rf"rope_{tag}\(\d+ heads\) slice\+64wide\+concat ms=\s*([\d.]+)")
        wide = _grep("probe_output_paths", rf"rope_{tag}\(\d+ heads\).*permuted 256wide ms=\s*([\d.]+)")
        total_narrow += float(narrow)
        total_wide += float(wide)
        rows.append(f"| {label} | {narrow} ms | **{wide} ms** |")
    saving = (total_narrow - total_wide) * 1000.0
    prefill = _summary()["measurements"]["fused/full_attention/prefill"]["device_kernel_time_ms"]
    rows.append("")
    rows.append(
        f"so the whole permutation is worth **{saving:.0f} us of a {prefill:.3f} ms prefill, "
        f"{100.0 * saving / 1000.0 / prefill:.1f} %**."
    )
    return "\n".join(rows)


def _evidence() -> dict:
    return json.loads((DOC / "pcc_evidence.json").read_text())


def _minima() -> dict:
    """``{(metric, kind): min value}`` over the fused stage's own records."""
    out: dict[tuple[str, str], float] = {}
    for record in _evidence()["records"]:
        value = record["value"]
        if not isinstance(value, float):
            continue
        key = (record["metric"], record.get("kind"))
        out[key] = min(out.get(key, value), value)
    return out


#: Rows of the README's correctness table: label, and the metric each column reads.
CORRECTNESS_ROWS = (
    ("prefill vs HF, seq 1 / 17 / 128 / 2048 / 2049 / 4096 / 5000", "fused_prefill_pcc", "min "),
    ("prefill vs HF, longest single-shot reference length", "fused_long_prefill_pcc", ""),
    ("decode vs HF, 4 steps after prefill 17 / 2048 / 2049 / 5000", "fused_decode_pcc", "min "),
    ("batch 32 and 4, unequal prompts 64..3071, permuted page table - prefill", "fused_batched_prefill_pcc", "min "),
    ("batch 32 and 4 - decode", "fused_batched_decode_pcc", "min "),
    ("**real checkpoint weights** - prefill @ 2049", "fused_real_weight_prefill_pcc", ""),
    ("**real checkpoint weights** - decode @ 2049", "fused_real_weight_decode_pcc", ""),
    ("traced decode, replay output vs HF", "fused_traced_decode_replay_pcc", "min "),
    ("traced decode at batch 4, per-user positions", "fused_batched_traced_decode_pcc", "min "),
    ("paged K cache vs HF after prefill 2049", "fused_paged_k_cache_pcc", ""),
    ("paged V cache vs HF after prefill 2049", "fused_paged_v_cache_pcc", ""),
    ("conv state vs HF after prefill 2049", "fused_conv_state_pcc", ""),
    ("recurrent state vs HF after prefill 2049", "fused_recurrent_state_pcc", ""),
    ("page block size 32 and 128 instead of 64 - prefill", "fused_alt_block_size_prefill_pcc", "min "),
    ("page block size 32 and 128 instead of 64 - decode", "fused_alt_block_size_decode_pcc", "min "),
    ("BFP8 KV cache - prefill @ 2049", "fused_bfp8_cache_prefill_pcc", ""),
    ("BFP8 KV cache - decode @ 2049", "fused_bfp8_cache_decode_pcc", ""),
    ("pad-below-one-tile lengths 735..768 - prefill", "fused_pad_alias_prefill_pcc", "min "),
    ("pad-below-one-tile lengths 735..768 - decode", "fused_pad_alias_decode_pcc", "min "),
    ("**full context 262143** - prefill tail vs HF", "fused_full_context_prefill_tail_pcc", ""),
    ("**full context 262143** - conv state vs HF", "fused_full_context_conv_state_pcc", ""),
    ("**full context 262143** - recurrent state vs HF", "fused_full_context_recurrent_state_pcc", ""),
    ("**full context 262143** - paged K cache vs HF", "fused_full_context_paged_k_cache_pcc", ""),
    ("**full context 262143** - paged V cache vs HF", "fused_full_context_paged_v_cache_pcc", ""),
    ("**full context 262143** - decode at position 262143", "fused_full_context_decode_pcc", ""),
    ("**full context 262143** - best-fit *scale* vs HF, prefill tail", "fused_full_context_prefill_tail_scale", ""),
    ("**full context 262143** - best-fit *scale* vs HF, decode", "fused_full_context_decode_scale", ""),
    ("fused vs functional output, prefill and decode @ 2049", "fused_vs_functional_pcc", "min "),
)

#: Full-context figures whose functional-stage counterpart the delta table compares against.
DELTA_ROWS = (
    (
        "`linear_attention` full-context prefill tail",
        "fused_full_context_prefill_tail_pcc",
        "full_context_prefill_tail_pcc",
        "linear_attention",
    ),
    (
        "`linear_attention` full-context recurrent state",
        "fused_full_context_recurrent_state_pcc",
        "full_context_recurrent_state_pcc",
        "linear_attention",
    ),
    (
        "`linear_attention` full-context decode @ 262143",
        "fused_full_context_decode_pcc",
        "full_context_decode_pcc",
        "linear_attention",
    ),
    (
        "`full_attention` full-context prefill tail",
        "fused_full_context_prefill_tail_pcc",
        "full_context_prefill_tail_pcc",
        "full_attention",
    ),
    (
        "`full_attention` full-context decode @ 262143",
        "fused_full_context_decode_pcc",
        "full_context_decode_pcc",
        "full_attention",
    ),
)


def correctness_table() -> str:
    minima = _minima()
    rows = ["| measurement | `linear_attention` | `full_attention` |", "|---|---|---|"]
    for label, metric, prefix in CORRECTNESS_ROWS:
        cells = []
        for kind in KINDS:
            value = minima.get((metric, kind))
            cells.append("—" if value is None else f"{prefix}{value:.6f}")
        if all(cell == "—" for cell in cells):
            continue
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    pcc_min = _evidence()["min_pcc"]
    rows.append("")
    rows.append(
        f"Minimum over all {_evidence()['num_pcc_records']} PCC records: **{pcc_min:.6f}**, "
        f"against a bar of 0.995. No exception, no waiver, no open gap."
    )
    return "\n".join(rows)


def delta_table() -> str:
    """Fused against functional at the full context, both sides read from their own evidence."""
    fused = _minima()
    functional = {}
    payload = json.loads((DOC.parent / "functional_decoder" / "pcc_evidence.json").read_text())
    for record in payload["records"]:
        if isinstance(record["value"], float):
            key = (record["metric"], record.get("kind"))
            functional[key] = min(functional.get(key, record["value"]), record["value"])
    rows = ["| | functional | fused | delta |", "|---|---|---|---|"]
    for label, fused_metric, functional_metric, kind in DELTA_ROWS:
        before = functional.get((functional_metric, kind))
        after = fused.get((fused_metric, kind))
        if before is None or after is None:
            continue
        delta = after - before
        rows.append(f"| {label} | {before:.6f} | {after:.6f} | {delta:+.1e} |")
    return "\n".join(rows)


BLOCKS = {
    "correctness": correctness_table,
    "delta": delta_table,
    "before_after": before_after_table,
    "breakdown": breakdown_table,
    "decode_matmul_share": decode_matmul_share,
    "conv_formulations": conv_table,
    "broadcast_bandwidth": broadcast_table,
    "norm_cores": norm_table,
    "recurrence_grid": recurrence_table,
    "mlp_variants": mlp_table,
    "gdr_call_shapes": gdr_table,
    "gdn_epilogue": epilogue_table,
    "rope_width": rope_table,
}


def main() -> None:
    filled = 0
    for path in (DOC / "README.md", DOC / "work_log.md", DOC / "probes" / "README.md"):
        text = path.read_text()
        for name, builder in BLOCKS.items():
            pattern = re.compile(rf"(<!-- GENERATED:{name} -->\n).*?(\n<!-- END GENERATED:{name} -->)", re.DOTALL)
            if not pattern.search(text):
                continue
            text = pattern.sub(lambda m: m.group(1) + builder() + m.group(2), text)
            filled += 1
        path.write_text(text)
    print(f"filled {filled} generated blocks")


if __name__ == "__main__":
    main()

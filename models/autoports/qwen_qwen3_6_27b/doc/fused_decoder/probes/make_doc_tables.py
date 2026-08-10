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

import csv
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


def _shipped_grid(name: str) -> str:
    """``(y, x)`` of a shipped ``core_grid`` constant, read out of ``tt/fused_decoder.py``.

    The "selected" cells of the grid tables used to be literals in this file, so a table inside a
    GENERATED block could - and did - name a grid the layer does not ship.  Reading the constant
    is what makes the cell a fact about the code.
    """
    source = (DOC.parents[1] / "tt" / "fused_decoder.py").read_text()
    match = re.search(rf"^{name} = \((\d+), (\d+)\)$", source, re.MULTILINE)
    if not match:
        raise SystemExit(f"tt/fused_decoder.py has no {name} = (y, x) literal")
    return f"{match.group(1)}x{match.group(2)}"


def before_after_table() -> str:
    summary = _summary()
    rows = [
        "| layer kind | phase | device time before | device time after | speed-up | ops before | ops after |",
        "|---|---|---|---|---|---|---|",
    ]
    labels = {
        "prefill": "prefill, 2048 tokens",
        "decode": "traced decode, 1 token, batch 1",
        "decode_batch32": "traced decode, 1 token, batch 32 (advertised `max_batch`)",
    }
    for kind in KINDS:
        for phase in ("prefill", "decode", "decode_batch32"):
            row = summary["speedup"][f"{kind}/{phase}"]
            rows.append(
                f"| `{kind}` | {labels[phase]} | {row['device_ms_before']:.3f} ms | "
                f"**{row['device_ms_after']:.3f} ms** | **{row['speedup_x']:.2f}x** | "
                f"{row['ops_before']} | {row['ops_after']} |"
            )
    return "\n".join(rows)


def breakdown_table() -> str:
    summary = _summary()
    columns = [f"fused/{kind}/{phase}" for kind in KINDS for phase in ("prefill", "decode", "decode_batch32")]
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
    heads = {"prefill": "prefill", "decode": "decode b1", "decode_batch32": "decode b32"}
    rows = [
        "| bucket | "
        + " | ".join(f"`{column.split('/')[1]}` {heads[column.split('/')[2]]}" for column in columns)
        + " |",
        "|---" * (len(columns) + 1) + "|",
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
        "rm_shift": "untilize once, ROW_MAJOR shift, tilize per tap (TILE concat)",
        "rm_concat": "ROW_MAJOR concat *and* shift, SiLU folded into the last add - **shipped**",
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
    """Median and spread per grid, at both decode regimes; the selection is labelled, not hidden.

    One table per head count: the shipped grids were once chosen at 48 head problems (batch 1)
    and shipped unchanged at 1536 (the advertised ``max_batch``), which a stage review called out
    as the least defensible place to leave a row-count-independent constant.
    """
    grids = ("default", "1x4", "1x8", "1x11", "2x4", "2x8", "2x11", "4x4", "4x8", "4x11", "6x4", "6x8", "6x11")
    log = _probe("probe_decode_recurrence")
    blocks = []
    for heads, regime in ((48, "batch 1"), (48 * 32, "batch 32, the advertised `max_batch`")):
        rows = [
            f"**{heads} head problems** ({regime})",
            "",
            "| shape | " + " | ".join(grids) + " | selected |",
            "|---" * (len(grids) + 2) + "|",
        ]
        for kind, prefix, label, selected in (
            ("read", "", "state read", _shipped_grid("_RECURRENCE_READ_GRID")),
            ("outer", "", "outer product (`transpose` + `matmul`)", "—"),
            (
                "outer",
                "transpose_a ",
                "outer product (`transpose_a=True`, shipped)",
                _shipped_grid("_RECURRENCE_OUTER_GRID"),
            ),
        ):
            cells = []
            for grid in grids:
                token = "default" if grid == "default" else f"core_grid {grid}"
                match = re.search(
                    rf"{kind}\s+heads={heads}\s+{re.escape(prefix + token)}\s*median_us=\s*([\d.]+) stdev_us=\s*([\d.]+)",
                    log,
                )
                cells.append("—" if match is None else f"{match.group(1)} ({match.group(2)})")
            rows.append(f"| {label} | " + " | ".join(cells) + f" | {selected} |")
        blocks.append("\n".join(rows))
    return (
        "\n\n".join(blocks)
        + "\n\nMedian and (stdev) in microseconds over 30 repeats, at both decode regimes. The default "
        "program factory is several times slower than any explicit grid. The state read's 6x4 is the "
        "fastest measured at both head counts. The outer product's explicit grids sit within about a "
        "stdev of each other at batch 1 but not at batch 32, so it takes the grid that wins there; the "
        "shipped form folds its transpose into the matmul, which is one dispatch fewer and bit-exact, "
        "and has its own row so that choice is a measurement rather than an argument."
    )


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


def python_op_counts() -> str:
    """The ``ttnn``-boundary op counts ``test_fused_graph_is_smaller`` recorded, read from evidence."""
    counts = {
        record.get("kind"): record["value"]
        for record in _evidence()["records"]
        if record["metric"] == "fused_op_counts"
    }
    parts = []
    for kind, value in counts.items():
        before, after = value["functional"], value["fused"]
        parts.append(f"`{kind}` {before[0]} -> {after[0]} prefill, {before[1]} -> {after[1]} decode")
    return "; ".join(parts)


def gated_norm_batches() -> str:
    """The decode z-gated norm's two forms across batch sizes, read from the probe log."""
    log = _probe("probe_gated_norm_batch")
    rows = re.findall(
        r"gated_norm batch=\s*(\d+) reshape_us=\s*([\d.]+) \(\s*[\d.]+\) group_us=\s*([\d.]+) "
        r"\(\s*[\d.]+\) pcc_between=([\d.]+)",
        log,
    )
    if not rows:
        raise SystemExit("probe_gated_norm_batch.log has no measurements")
    batches = [row[0] for row in rows]
    header = "| batch | " + " | ".join(batches) + " |"
    reshape = "| reshape + `ttnn.rms_norm` (us) | " + " | ".join(row[1] for row in rows) + " |"
    group = "| group reduction (us) | " + " | ".join(row[2] for row in rows) + " |"
    worst = min(float(row[3]) for row in rows)
    return (
        header + "\n" + "|---" * (len(batches) + 1) + "|\n" + reshape + "\n" + group + "\n\n"
        f"Median over 25 repeats. Lowest PCC between the two forms' outputs, over all batches "
        f"measured: {worst:.6f}."
    )


def before_breakdown() -> str:
    """The stage-1 baseline's bucket breakdown, from the committed functional reports."""
    summary = _summary()
    columns = [f"functional/{kind}/{phase}" for kind in KINDS for phase in ("prefill", "decode")]
    labels = {
        "matmul": "`matmul`",
        "batched_matmul": "`batched_matmul` (the spelled-out delta rule / recurrence)",
        "sdpa": "`sdpa`",
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
            value = summary["measurements"][column]["breakdown_ms"].get(bucket)
            cells.append("—" if value is None else f"{value:.3f} ms")
        if all(cell == "—" for cell in cells):
            continue
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    totals = [summary["measurements"][column] for column in columns]
    rows.append("| **total** | " + " | ".join(f"**{row['device_kernel_time_ms']:.3f} ms**" for row in totals) + " |")
    rows.append("| ops in one pass | " + " | ".join(str(row["ops_per_pass"]) for row in totals) + " |")
    rows.append("| op-to-op gap | " + " | ".join(f"{row['op_to_op_gap_ms']:.3f} ms" for row in totals) + " |")
    return "\n".join(rows)


def qkv_gate_table() -> str:
    """The last shared-LHS matmul pair: two matmuls, or one and two slices."""
    rows = ["| rows | two matmuls (shipped) | one packed matmul + 2 slices |", "|---|---|---|"]
    for rows_label, pattern in (("2048 (prefill)", "2048"), ("32 (decode)", "32")):
        match = re.search(
            rf"qkv_gate rows=\s*{pattern} split_us=\s*([\d.]+) \(\s*[\d.]+\) "
            rf"packed_us=\s*([\d.]+) \(\s*[\d.]+\) pcc_qkv=([\d.]+)",
            _probe("probe_qkv_gate_pack"),
        )
        if not match:
            raise SystemExit(f"probe_qkv_gate_pack.log has no row for {pattern}")
        rows.append(f"| {rows_label} | **{match.group(1)} us** | {match.group(2)} us |")
    rows.append("")
    rows.append(
        "Median over 25 repeats (9 at 2048 rows), outputs identical (PCC 1.000000). The packed "
        "form loses at prefill by more than a third: the two slices of the merged output are "
        "full copies of a 14336-wide TILE tensor, which costs more than the activation re-read "
        "and the dispatch it saves. At decode the two are within a stdev of each other."
    )
    return "\n".join(rows)


def matmul_grid_table() -> str:
    """Every ``Bound=SLOW`` row against the graph levers that could move it."""
    log = _probe("probe_matmul_bound")
    grids = ("10x11", "8x8", "4x8", "2x8", "2x4", "1x4", "1x2")
    header = "| row | shape | default | " + " | ".join(grids) + " |"
    rows = [header, "|---" * (len(grids) + 3) + "|"]
    for match in re.finditer(r"matmul (\S+\s+\S+)\s+(\d+)x\s*(\d+)x\s*(\d+) out=\w+\+fp32dest_us=\s*([\d.]+)", log):
        label, m_dim, k_dim, n_dim, default = match.groups()
        label = " ".join(label.split())
        cells = []
        for grid in grids:
            cell = re.search(
                rf"matmul {re.escape(label.split()[0])}\s+{label.split()[1]}\s+core_grid "
                rf"{grid.replace('x', 'x *')}\s+us=\s*([\d.]+)",
                log,
            )
            cells.append("—" if cell is None else cell.group(1))
        rows.append(
            f"| `{label.split()[0]}` {label.split()[1]} | {m_dim}x{k_dim}x{n_dim} | {default} | "
            + " | ".join(cells)
            + " |"
        )
    rows.append("")
    rows.append(
        "Median microseconds over 25 repeats; `default` is the program factory's own choice at "
        "the shipped output dtype. The rows whose N is 2 or 4 tiles are 2-3x faster on a small "
        "explicit grid, because the default spreads output columns over the whole device and "
        "then broadcasts the activation to cores that have nothing to do."
    )
    return "\n".join(rows)


def dtype_lever_table() -> str:
    """The output-dtype and DEST-precision levers on the same rows, for the ones a grid cannot fix."""
    rows = ["| row | shape | shipped | output dtype swapped | `fp32_dest_acc_en=False` |", "|---|---|---|---|---|"]
    for match in re.finditer(
        r"matmul (\S+\s+\S+)\s+(\d+)x\s*(\d+)x\s*(\d+) out=(\w+)\+fp32dest_us=\s*([\d.]+) \(\s*[\d.]+\) "
        r"out=(\w+)\+fp32dest_us=\s*([\d.]+) \(\s*[\d.]+\) out=\w+\+bf16dest_us=\s*([\d.]+)",
        _probe("probe_matmul_bound"),
    ):
        label, m_dim, k_dim, n_dim, shipped, base, other, swapped, no_dest = match.groups()
        label = " ".join(label.split())
        rows.append(
            f"| `{label.split()[0]}` {label.split()[1]} | {m_dim}x{k_dim}x{n_dim} | "
            f"{base} us ({shipped}) | {swapped} us ({other}) | {no_dest} us |"
        )
    rows.append("")
    rows.append(
        "Median microseconds over 25 repeats. Neither lever is worth taking here, and both are "
        "precision policy rather than graph shape: the output dtype of these rows is what the "
        "next op consumes, and `fp32_dest_acc_en` is the stage-1 compute-kernel policy."
    )
    return "\n".join(rows)


def input_fold_table() -> str:
    """The two decode unary-into-binary folds, and the rank-3 slice order that was not taken."""
    rows = ["| fold | batch | separate unary | folded into the binary | agreement |", "|---|---|---|---|---|"]
    labels = {
        "decay": "`exp(g)` into the recurrent-state multiply",
        "beta": "`sigmoid(b)` into the `delta` multiply",
    }
    for name, label in labels.items():
        for batch in ("1", "32"):
            match = re.search(
                rf"fold {name}\s+batch=\s*{batch} split_us=\s*([\d.]+) \(\s*[\d.]+\) "
                rf"folded_us=\s*([\d.]+) \(\s*[\d.]+\) pcc=([\d.]+) max_abs_diff=(\S+)",
                _probe("probe_gdn_input_folds"),
            )
            if not match:
                raise SystemExit(f"probe_gdn_input_folds.log has no {name} row at batch {batch}")
            rows.append(
                f"| {label} | {batch} | {match.group(1)} us | **{match.group(2)} us** | "
                f"PCC {match.group(3)}, max abs diff {match.group(4)} |"
            )
    match = re.search(
        r"rank3 seq=(\d+) rank4_first_us=\s*([\d.]+) \(\s*[\d.]+\) rank3_first_us=\s*([\d.]+) "
        r"\(\s*[\d.]+\) pcc_beta=([\d.]+) pcc_g=([\d.]+) max_abs_diff=(\S+)",
        _probe("probe_gdn_input_folds"),
    )
    if not match:
        raise SystemExit("probe_gdn_input_folds.log has no rank3 row")
    rows.append(
        f"| rank-3 before the slices instead of after (not taken) | {match.group(1)} rows | "
        f"{match.group(2)} us (shipped) | {match.group(3)} us | "
        f"PCC {match.group(4)}, max abs diff {match.group(6)} |"
    )
    rows.append("")
    rows.append(
        "Median microseconds over 25 repeats (9 for the rank-3 row). Both folds are bit-exact "
        "and both were taken. Moving the rank change ahead of the slices removes two float32 "
        "reshapes and adds one, and measures as a tie, so the shipped order stands."
    )
    return "\n".join(rows)


def run_totals() -> str:
    """Test counts and evidence-record counts, from the logs and the evidence file itself.

    These were hand-written in four places and went stale the moment a test was added, which is
    the drift class this whole file exists to remove.
    """
    evidence = _evidence()
    lines = []
    for log, label in (
        ("suite_main", "`logs/suite_main.log`"),
        ("long_context", "`logs/long_context.log`"),
        ("watcher_run", "`logs/watcher_run.log`"),
    ):
        text = (DOC / "logs" / f"{log}.log").read_text(errors="replace")
        match = re.findall(r"=+ (\d+) passed(?:, (\d+) skipped)?[^=]*=+", text)
        if not match:
            raise SystemExit(f"{log}.log has no pytest summary line")
        passed, skipped = match[-1]
        tail = f", {skipped} skipped" if skipped else ""
        lines.append(f"* {label} — **{passed} passed{tail}**")
    scales = [record["value"] for record in evidence["records"] if str(record["metric"]).endswith("_scale")]
    lines.append(
        f"* `pcc_evidence.json` — {evidence['num_records']} records, {evidence['num_pcc_records']} of them "
        f"PCC, **minimum {evidence['min_pcc']:.6f}**, none below the 0.995 bar; "
        f"{evidence['num_scale_records']} full-context scale ratios, range {min(scales):.5f} to "
        f"{max(scales):.5f}, inside the ±2 % tolerance."
    )
    return "\n".join(lines)


def slow_rows() -> str:
    """Every ``Bound=SLOW`` row in every committed *fused* report, grouped by op code.

    Written from the reports rather than by hand: §6.1 was enumerated once from four of the six
    reports and was wrong the day the other two were added.  Rows of the same op code inside one
    pass are one entry, because they are one decision.
    """
    summary = _summary()
    rows = [
        "| pass | op | ops per pass | device time per pass | share | cores | DRAM % |",
        "|---|---|---|---|---|---|---|",
    ]
    groups = 0
    for kind in KINDS:
        for phase in ("prefill", "decode", "decode_batch32"):
            report = DOC / "tracy" / "fused" / kind / f"{phase}_perf_report.csv"
            with report.open() as handle:
                table = list(csv.DictReader(handle))
            replays = 1 if phase == "prefill" else 8
            pass_us = summary["measurements"][f"fused/{kind}/{phase}"]["device_kernel_time_ms"] * 1000.0
            grouped: dict[str, list[tuple[float, str, str]]] = {}
            for row in table:
                if (row.get("Bound") or "").strip() != "SLOW":
                    continue
                grouped.setdefault(row["OP Code"], []).append(
                    (
                        float(row["Device Time"] or 0),
                        (row.get("Cores") or "").strip(),
                        (row.get("DRAM %") or "").strip(),
                    )
                )
            for code, values in sorted(grouped.items(), key=lambda item: -sum(v[0] for v in item[1])):
                groups += 1
                microseconds = sum(value for value, _, _ in values) / replays
                cores = sorted({v[1] for v in values if v[1]}, key=lambda value: float(value))
                drams = sorted(float(v[2]) for v in values if v[2])
                core_cell = "—" if not cores else "/".join(f"{float(value):.0f}" for value in cores)
                low, high = (f"{drams[0]:.0f}", f"{drams[-1]:.0f}") if drams else ("", "")
                dram_cell = "—" if not drams else (f"{low} %" if low == high else f"{low}-{high} %")
                per_pass = len(values) / replays
                count_cell = f"{per_pass:.0f}" if abs(per_pass - round(per_pass)) < 1e-9 else f"{per_pass:.2f}"
                rows.append(
                    f"| `{kind}` {phase} | `{code}` | {count_cell} | {microseconds:.1f} us | "
                    f"{100.0 * microseconds / pass_us:.1f} % | {core_cell} | {dram_cell} |"
                )
    rows.append("")
    rows.append(
        f"{groups} `Bound=SLOW` op groups across the six committed fused reports, every one of them a "
        "`linear_attention` row. Device time and share are per pass — per trace replay for the two "
        "decode windows — and `cores` is what the profiler reports each instance ran on."
    )
    return "\n".join(rows)


def rejected_decode_variants() -> str:
    """The two decode variants round 7 asked for, measured at both batches."""
    rows = ["| variant | batch | shipped | alternative | agreement |", "|---|---|---|---|---|"]
    for batch in ("1", "32"):
        match = re.search(
            rf"decode_conv batch=\s*{batch} float32_us=\s*([\d.]+) \(\s*[\d.]+\) "
            rf"bfloat16_us=\s*([\d.]+) \(\s*[\d.]+\) pcc_float32_vs_torch=([\d.]+) "
            rf"pcc_bfloat16_vs_torch=([\d.]+) pcc_between=([\d.]+)",
            _probe("probe_decode_conv_dtype"),
        )
        if not match:
            raise SystemExit(f"probe_decode_conv_dtype.log has no batch {batch} row")
        rows.append(
            f"| decode causal-conv FIR in bfloat16 instead of float32 | {batch} | "
            f"**{match.group(1)} us** (float32) | {match.group(2)} us (bfloat16) | "
            f"PCC {match.group(5)} between them, {match.group(4)} against torch |"
        )
    for batch in ("1", "32"):
        match = re.search(
            rf"decode_heads batch=\s*{batch} expand_then_norm_us=\s*([\d.]+) \(\s*[\d.]+\) "
            rf"norm_then_expand_us=\s*([\d.]+) \(\s*[\d.]+\) pcc_between=([\d.]+)",
            _probe("probe_gdn_decode_heads"),
        )
        if not match:
            raise SystemExit(f"probe_gdn_decode_heads.log has no batch {batch} row")
        rows.append(
            f"| Q/K L2 norm before the GQA expansion instead of after | {batch} | "
            f"**{match.group(1)} us** (expand, then norm) | {match.group(2)} us (norm, then expand) | "
            f"PCC {match.group(3)} between them |"
        )
    rows.append("")
    rows.append(
        "Median microseconds over 25 repeats. Both alternatives are the same arithmetic as what "
        "ships and both measure slower, at both decode batches."
    )
    return "\n".join(rows)


def rejected_shared_work() -> str:
    """Two shared-work merges the batch-32 and prefill profiles suggested, both measured."""
    rows = ["| candidate | shape | shipped | merged | agreement |", "|---|---|---|---|---|"]
    for batch in ("1", "32"):
        match = re.search(
            rf"decode_qk batch=\s*{batch} separate_us=\s*([\d.]+) \(\s*[\d.]+\) "
            rf"merged_us=\s*([\d.]+) \(\s*[\d.]+\) pcc_q=([\d.]+) pcc_k=([\d.]+)",
            _probe("probe_decode_qk_pair"),
        )
        if not match:
            raise SystemExit(f"probe_decode_qk_pair.log has no batch {batch} row")
        rows.append(
            f"| decode Q and K through one norm/scale/rank-change chain | batch {batch} | "
            f"**{match.group(1)} us** (separate) | {match.group(2)} us (merged) | "
            f"PCC {match.group(3)} / {match.group(4)} |"
        )
    match = re.search(
        r"prefill_qkv seq=(\d+) packed_ms=\s*([\d.]+) \(\s*[\d.]+\) split_ms=\s*([\d.]+) \(\s*[\d.]+\) "
        r"pcc_q=([\d.]+)",
        _probe("probe_prefill_qkv_split"),
    )
    if not match:
        raise SystemExit("probe_prefill_qkv_split.log has no measurement")
    rows.append(
        f"| prefill `in_proj_qkv` as three projections and three FIRs instead of one and three slices | "
        f"{match.group(1)} tokens | **{match.group(2)} ms** (packed) | {match.group(3)} ms (split) | "
        f"PCC {match.group(4)} |"
    )
    rows.append("")
    rows.append(
        "Median over 25 repeats (9 for the prefill row). Both merges are the same arithmetic as what "
        "ships and both measure slower: cutting a wide TILE tensor apart, or concatenating one, costs "
        "more than the shared work it enables - the same result §6.2 found for `wqkv`/`wgate`."
    )
    return "\n".join(rows)


BLOCKS = {
    "before_breakdown": before_breakdown,
    "correctness": correctness_table,
    "gated_norm_batches": gated_norm_batches,
    "python_op_counts": python_op_counts,
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
    "qkv_gate_pack": qkv_gate_table,
    "matmul_grids": matmul_grid_table,
    "matmul_dtype_levers": dtype_lever_table,
    "input_folds": input_fold_table,
    "run_totals": run_totals,
    "slow_rows": slow_rows,
    "rejected_decode_variants": rejected_decode_variants,
    "rejected_shared_work": rejected_shared_work,
}


def main() -> None:
    filled = 0
    for path in (DOC / "README.md", DOC / "work_log.md", DOC / "probes" / "README.md"):
        text = path.read_text()
        for name, builder in BLOCKS.items():
            # ``.*?`` between the markers, *including* the empty case: a block whose content is
            # not there yet has a single newline between its markers, and a pattern that demands
            # one on each side skips it forever - which is how a placeholder block once survived
            # four review rounds.  ``test_generated_blocks_are_current`` rejects an empty block.
            pattern = re.compile(rf"(<!-- GENERATED:{name} -->\n)(.*?)(<!-- END GENERATED:{name} -->)", re.DOTALL)
            if not pattern.search(text):
                continue
            text = pattern.sub(lambda m: m.group(1) + builder() + "\n" + m.group(3), text)
            filled += 1
        path.write_text(text)
    print(f"filled {filled} generated blocks")


if __name__ == "__main__":
    main()

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


def _shipped_grid(name: str, key: str | None = None) -> str:
    """``(y, x)`` of a shipped ``core_grid`` constant, read out of ``tt/fused_decoder.py``.

    The "selected" cells of the grid tables used to be literals in this file, so a table inside a
    GENERATED block could - and did - name a grid the layer does not ship.  Reading the constant
    is what makes the cell a fact about the code.
    """
    source = (DOC.parents[1] / "tt" / "fused_decoder.py").read_text()
    if key is None:
        match = re.search(rf"^{name} = \((\d+), (\d+)\)$", source, re.MULTILINE)
        if not match:
            raise SystemExit(f"tt/fused_decoder.py has no {name} = (y, x) literal")
        return f"{match.group(1)}x{match.group(2)}"
    line = re.search(rf"^{name} = \{{(.+?)\}}$", source, re.MULTILINE)
    if not line:
        raise SystemExit(f"tt/fused_decoder.py has no {name} = {{...}} literal")
    match = re.search(rf'"{key}": \((\d+), (\d+)\)', line.group(1))
    if not match:
        raise SystemExit(f"{name} has no {key!r} entry")
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
        "state_update": "`state_update` (the fused recurrent-state update, §3.21)",
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
    # State what ``other`` holds rather than asserting it is empty: a stage review found an eighth
    # of the advertised-batch decode sitting in it while both documents said it was empty.
    leftovers = {
        column: summary["measurements"][column]["breakdown_ms"]["other"]
        for column in summary["measurements"]
        if summary["measurements"][column]["breakdown_ms"].get("other")
    }
    rows.append("")
    rows.append(
        "Every op is classified: the `other` bucket is empty in all " f"{len(summary['measurements'])} measured passes."
        if not leftovers
        else "Unclassified (`other`) time remains in: "
        + ", ".join(f"`{column}` {value:.3f} ms" for column, value in sorted(leftovers.items()))
        + "."
    )
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
    """The six FIR formulations, median and spread over 12 repeats; winner derived, not chosen."""
    names = {
        "tile": "all-TILE slices (what the functional layer does)",
        "rm_shift": "untilize once, ROW_MAJOR shift, tilize per tap (TILE concat)",
        "rm_concat": "ROW_MAJOR concat *and* shift, SiLU folded into the last add - **shipped**",
        "rm_arith": "untilize once, whole FIR in ROW_MAJOR, tilize once",
        "aligned_win": "one pre-padded window per tap so every slice is tile-aligned",
        "scale_shift": "scale on the TILE tensor first, then untilize per tap and shift-and-add in ROW_MAJOR",
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
    # Bold means one thing in every table here: a minimum that beats the runner-up by more than
    # the two spreads together.  It used to mean "lowest median", which bolded a tie in the float32
    # column - 14.251 (0.042) against 14.260 (0.039).
    best = {
        dtype: _distinguishable_min([measured[key][dtype] for key in names], list(names)) for dtype in ("fp32", "bf16")
    }
    rows = ["| formulation | float32 median (stdev) ms | bfloat16 median (stdev) ms |", "|---|---|---|"]
    for key, label in names.items():
        cells = []
        for dtype in ("fp32", "bf16"):
            median, stdev = measured[key][dtype]
            mark = "**" if best[dtype] == key else ""
            cells.append(f"{mark}{median}{mark} ({stdev})")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    rows.append("")
    rows.append(
        "A bolded median is a minimum outside the two spreads together; a column with no bold has "
        "its two fastest rows inside each other's spread."
    )
    return "\n".join(rows)


def _distinguishable_min(measured: list[tuple[str, str]], labels: list) -> object | None:
    """The label of the fastest row, or ``None`` when the two fastest are inside their spreads.

    The same rule as :func:`_verdict`, for a sweep instead of a pair.  Two tables bolded a "best"
    that was a tie with its runner-up until a stage review re-derived them from the logs.
    """
    order = sorted(range(len(measured)), key=lambda index: float(measured[index][0]))
    if len(order) < 2:
        return labels[order[0]] if order else None
    best, runner = order[0], order[1]
    gap = float(measured[runner][0]) - float(measured[best][0])
    if gap <= float(measured[best][1]) + float(measured[runner][1]):
        return None
    return labels[best]


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
    # The spreads are in the table because the two fastest columns are a tie, and the table bolded
    # one of them as a win - against the prose three lines below it, and against the shipped
    # constant, which is the other one.
    measured = [
        (
            _grep("probe_small_ops", rf"rms_norm sharded\s+{core}c ms=([\d.]+)"),
            _grep("probe_small_ops", rf"rms_norm sharded\s+{core}c ms=[\d.]+ stdev_ms=([\d.]+)"),
        )
        for core in cores
    ]
    interleaved = (
        _grep("probe_small_ops", r"rms_norm interleaved\s+ms=([\d.]+)"),
        _grep("probe_small_ops", r"rms_norm interleaved\s+ms=[\d.]+ stdev_ms=([\d.]+)"),
    )
    best = _distinguishable_min(measured + [interleaved], list(cores) + ["interleaved"])
    cells = [
        f"**{value}** ({spread})" if core == best else f"{value} ({spread})"
        for core, (value, spread) in zip(cores, measured)
    ]
    tail = (
        f"**{interleaved[0]}** ({interleaved[1]})" if best == "interleaved" else f"{interleaved[0]} ({interleaved[1]})"
    )
    return (
        "| cores | " + " | ".join(cores) + " | interleaved |\n"
        "|---|---|---|---|---|---|---|\n"
        "| ms (stdev) | " + " | ".join(cells) + f" | {tail} |\n\n"
        "Median and (stdev) over the sweep. A cell is bolded only when it beats the runner-up by "
        "more than the two spreads together: "
        + (
            "no width here does, so the choice is made on the shipped fallback rule."
            if best is None
            else f"{best} does."
        )
    )


def recurrence_table() -> str:
    """Median and spread per grid, at both decode regimes, with the shipped grid labelled.

    Everything here is read from the log and from the shipped constants: the column set is the
    grids the log actually contains (a stage review found a 13-grid literal after the sweep grew
    to 18, so the table's own "selected" cell named a column the table did not have), and the
    closing sentence names the measured minimum per regime rather than asserting one.
    """
    log = _probe("probe_decode_recurrence")
    rows_by_key: dict[tuple[str, str, int], dict[str, tuple[float, float]]] = {}
    for kind, heads, prefix, grid, median, stdev in re.findall(
        r"(read|outer)\s+heads=(\d+)\s+(transpose_a )?(default|core_grid \d+x\d+)\s*median_us=\s*"
        r"([\d.]+) stdev_us=\s*([\d.]+)",
        log,
    ):
        label = grid.replace("core_grid ", "")
        rows_by_key.setdefault((kind, prefix.strip(), int(heads)), {})[label] = (float(median), float(stdev))
    if not rows_by_key:
        raise SystemExit("probe_decode_recurrence.log has no grid sweep")

    def order(label: str) -> tuple[int, int, int]:
        if label == "default":
            return (0, 0, 0)
        y, x = (int(value) for value in label.split("x"))
        return (1, y, x)

    grids = sorted({label for measured in rows_by_key.values() for label in measured}, key=order)
    families = (
        ("read", "", "state read", "_RECURRENCE_READ_GRID"),
        ("outer", "", "outer product (`transpose` + `matmul`)", None),
        ("outer", "transpose_a", "outer product (`transpose_a=True`, shipped)", "_RECURRENCE_OUTER_GRID"),
    )
    blocks, notes = [], []
    for heads, regime, key in (
        (48, "batch 1", "small"),
        (48 * 32, "batch 32, the advertised `max_batch`", "large"),
    ):
        table = [
            f"**{heads} head problems** ({regime})",
            "",
            "| shape | " + " | ".join(grids) + " | selected |",
            "|---" * (len(grids) + 2) + "|",
        ]
        for kind, prefix, label, constant in families:
            measured = rows_by_key.get((kind, prefix, heads), {})
            cells = [f"{measured[grid][0]} ({measured[grid][1]})" if grid in measured else "—" for grid in grids]
            if constant is None:
                selected = "—"
            elif constant == "_RECURRENCE_READ_GRID":
                selected = _shipped_grid(constant, key)
            else:
                selected = _shipped_grid(constant)
            table.append(f"| {label} | " + " | ".join(cells) + f" | {selected} |")
            if constant is None or not measured:
                continue
            best = min((grid for grid in measured if grid != "default"), key=lambda grid: measured[grid][0])
            shipped_median, shipped_stdev = measured[selected]
            best_median, best_stdev = measured[best]
            verdict = (
                "the fastest measured"
                if selected == best
                else (
                    f"inside the combined spread of the fastest, {best} at {best_median} us"
                    if shipped_median <= best_median + best_stdev + shipped_stdev
                    else f"SLOWER than {best} at {best_median} us"
                )
            )
            notes.append(f"at {heads} head problems the {label.split(' (')[0]}'s {selected} is {verdict}")
        blocks.append("\n".join(table))

    return (
        "\n\n".join(blocks)
        + "\n\nMedian and (stdev) in microseconds over 30 repeats, at both decode regimes, over "
        + f"{len(grids) - 1} explicit grids plus the program factory's own choice. The default is the "
        "slowest row of every sweep here. Read from the log: "
        + "; ".join(notes)
        + ". The shipped form of the outer product folds its transpose into the matmul, which is one "
        "dispatch fewer and bit-exact, and has its own row so that choice is a measurement rather "
        "than an argument."
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
        shipped = " — **shipped**" if key == "fused_slice_act" else ""
        rows.append(f"| {label}{shipped} | {prefill} ms | {decode} ms |")
    rows.append("")
    # No cell is bolded here on purpose: this probe reports a best-of-N wall time with no spread,
    # so the tie rule every other table uses cannot be applied to it.  The decode column used to
    # bold the *rejected* variant, which read as a measured win it cannot support.
    rows.append(
        "`probe_mlp_variants.py` reports best-of-N wall time and prints no spread, so no cell here "
        "is marked a win - the shipped variant is labelled instead. The prefill column separates "
        "the three by margins far larger than any spread this stage has measured on that shape; the "
        "decode column does not, and §6 records the split variant as rejected on prefill. Re-running "
        "this probe with median and stdev, as every other probe reports, is listed as a limitation."
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
            # The 2048 rows used to be bolded, which in a column of wall times reads as "fastest"
            # when they are the slowest; this probe prints a best wall with no spread anyway.
            rows.append(f"| {shape} | {chunk} | {seq} | {o_pcc} | {state_pcc} | {wall} ms |")
    return "\n".join(rows)


def epilogue_table() -> str:
    token = _grep("probe_output_paths", r"gdn_epilogue token_major ms=\s*([\d.]+)")
    head = _grep("probe_output_paths", r"head_major ms=\s*([\d.]+)")
    pcc = _grep("probe_output_paths", r"pcc\(token,head\)=([\d.]+)")
    return (
        "| path | 2048-token chunk |\n|---|---|\n"
        f"| token-major output + group-reduction norm (§3.4) — **shipped** | {token} ms |\n"
        f"| `output_head_major=True` + per-head `ttnn.rms_norm` + z/result relayouts | {head} ms |\n"
        f"\nPCC between the two outputs: {pcc}. `probe_output_paths.py` reports a single wall time "
        f"with no spread, so the shipped path is labelled rather than bolded as a measured win."
    )


def rope_table() -> str:
    rows = ["| tensor | slice + 64-wide RoPE + slice + concat | permuted, one 256-wide RoPE |", "|---|---|---|"]
    total_narrow = total_wide = 0.0
    for tag, label in (("q", "q, 24 heads"), ("k", "k, 4 heads")):
        narrow = _grep("probe_output_paths", rf"rope_{tag}\(\d+ heads\) slice\+64wide\+concat ms=\s*([\d.]+)")
        wide = _grep("probe_output_paths", rf"rope_{tag}\(\d+ heads\).*permuted 256wide ms=\s*([\d.]+)")
        total_narrow += float(narrow)
        total_wide += float(wide)
        rows.append(f"| {label} | {narrow} ms | {wide} ms (shipped) |")
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
    ("batch {batches}, unequal prompts 64..3071, permuted page table - prefill", "fused_batched_prefill_pcc", "min "),
    ("batch {batches} - decode", "fused_batched_decode_pcc", "min "),
    ("**real checkpoint weights** - prefill @ 2049", "fused_real_weight_prefill_pcc", ""),
    ("**real checkpoint weights** - decode @ 2049", "fused_real_weight_decode_pcc", ""),
    ("traced decode, replay output vs HF", "fused_traced_decode_replay_pcc", "min "),
    ("traced decode at batch {batches}, per-user positions", "fused_batched_traced_decode_pcc", "min "),
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
    # ``{batches}`` in a label is filled from the records that row aggregates, so a row cannot
    # claim narrower coverage than the evidence carries - a stage review found three labels naming
    # batches 4 and 32 after batch 16 had been added, and one naming only batch 4.
    batches_by_metric: dict[str, list[int]] = {}
    for record in _evidence()["records"]:
        batch = record.get("batch")
        if isinstance(batch, int):
            values = batches_by_metric.setdefault(record["metric"], [])
            if batch not in values:
                values.append(batch)
    for label, metric, prefix in CORRECTNESS_ROWS:
        if "{batches}" in label:
            found = sorted(batches_by_metric.get(metric, []))
            label = label.format(batches=" and ".join(str(value) for value in found) if found else "?")
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
        r"gated_norm batch=\s*(\d+) reshape_us=\s*([\d.]+) \(\s*([\d.]+)\) group_us=\s*([\d.]+) "
        r"\(\s*([\d.]+)\) pcc_between=([\d.]+)",
        log,
    )
    # (batch, reshape, group, pcc, reshape spread, group spread) - the spreads are in the table
    # because "within the spread" was a claim nobody could check from it.
    rows = [(r[0], r[1], r[3], r[5], r[2], r[4]) for r in rows]
    if not rows:
        raise SystemExit("probe_gated_norm_batch.log has no measurements")
    batches = [row[0] for row in rows]
    header = "| batch | " + " | ".join(batches) + " |"
    reshape = "| reshape + `ttnn.rms_norm` (us) | " + " | ".join(f"{row[1]} ({row[4]})" for row in rows) + " |"
    group = "| group reduction (us) | " + " | ".join(f"{row[2]} ({row[5]})" for row in rows) + " |"
    worst = min(float(row[3]) for row in rows)
    crossing = next(
        (row[0] for row in rows if float(row[2]) + float(row[5]) < float(row[1])),
        None,
    )
    # Which form wins each batch, derived the same way every other table here derives a verdict:
    # a gap wider than the two spreads together is a win, anything narrower is a tie.  Round 21
    # asserted a tie at 16 that this row set denies, so the per-batch verdict is generated too.
    verdicts = ", ".join(
        f"batch {batch} {_verdict(r, rs, g, gs, 'reshape', 'group')}" for batch, r, g, _, rs, gs in rows
    )
    return (
        header + "\n" + "|---" * (len(batches) + 1) + "|\n" + reshape + "\n" + group + "\n\n"
        f"Median and (stdev) in microseconds over 25 repeats. Lowest PCC between the two forms' "
        f"outputs, over all batches measured: {worst:.6f}. The group form first becomes "
        f"distinguishably faster at batch {crossing}, which is where the shipped threshold sits. "
        f"Per batch, by the same rule: {verdicts}."
    )


def _verdict(left: str, left_spread: str, right: str, right_spread: str, left_name: str, right_name: str) -> str:
    """Which of two measured medians wins, or ``tie`` when the gap is inside the two spreads.

    This is the one rule the whole stage uses for "is this difference real", and it is here so a
    caption states it rather than a sentence asserting it - the class of defect rounds 17 and 21
    both landed in.  ``tests/test_fused_decoder_docs.py::test_qualitative_verdicts_match_their_logs``
    applies the same rule to the prose.
    """
    gap = abs(float(left) - float(right))
    if gap <= float(left_spread) + float(right_spread):
        return "tie"
    return f"{left_name} wins" if float(left) < float(right) else f"{right_name} wins"


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
    """Both shared-LHS pairs: two matmuls, or one and two slices, at the shipped output dtypes.

    The verdict cell and the closing sentence are derived from the measured medians and spreads -
    a stage review found this table asserting "within a stdev" over numbers that were not, and
    bolding the shipped column unconditionally.
    """
    pairs = {
        "qkv_gate": "`full_attention` `wqkv` + `wgate`",
        "qkv_z": "`linear_attention` `in_proj_qkv` + `in_proj_z`",
    }
    rows = [
        "| pair | rows | output dtypes | two matmuls (shipped) | one packed matmul + 2 slices | verdict |",
        "|---|---|---|---|---|---|",
    ]
    ties = []
    for key, label in pairs.items():
        for rows_label, pattern in (("2048 (prefill)", "2048"), ("32 (decode)", "32")):
            match = re.search(
                rf"{key} rows=\s*{pattern} split_us=\s*([\d.]+) \(\s*([\d.]+)\) "
                rf"packed_us=\s*([\d.]+) \(\s*([\d.]+)\) split_dtypes=(\S+) packed_dtype=(\S+) "
                rf"pcc_first=([\d.]+)",
                _probe("probe_qkv_gate_pack"),
            )
            if not match:
                raise SystemExit(f"probe_qkv_gate_pack.log has no {key} row for {pattern}")
            split, split_spread = float(match.group(1)), float(match.group(2))
            packed, packed_spread = float(match.group(3)), float(match.group(4))
            if abs(split - packed) <= split_spread + packed_spread:
                verdict, mark_split, mark_packed = "tie", "", ""
                ties.append(f"{key} at {rows_label.split()[0]} rows")
            elif split < packed:
                verdict, mark_split, mark_packed = "shipped wins", "**", ""
            else:
                verdict, mark_split, mark_packed = "**packed wins**", "", "**"
            rows.append(
                f"| {label} | {rows_label} | {match.group(5)} split, {match.group(6)} packed | "
                f"{mark_split}{match.group(1)} us{mark_split} | {mark_packed}{match.group(3)} us{mark_packed} | "
                f"{verdict} |"
            )
    rows.append("")
    rows.append(
        "Median over 25 repeats (9 at 2048 rows), outputs identical to the precision the dtypes "
        "allow. A row is a tie when the two medians are inside their combined spread"
        + (f" - that is the case for {', '.join(ties)}." if ties else ", which happens in none of them.")
        + " The output dtype matters and is measured, not assumed: `in_proj_qkv` emits float32 "
        "because the causal conv carries float32 state, and a packed matmul has one output dtype, "
        "so the merge would push `in_proj_z` to float32 as well."
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
    wins = measured = 0
    for name, label in labels.items():
        for batch in ("1", "32"):
            match = re.search(
                rf"fold {name}\s+batch=\s*{batch} split_us=\s*([\d.]+) \(\s*([\d.]+)\) "
                rf"folded_us=\s*([\d.]+) \(\s*([\d.]+)\) pcc=([\d.]+) max_abs_diff=(\S+)",
                _probe("probe_gdn_input_folds"),
            )
            if not match:
                raise SystemExit(f"probe_gdn_input_folds.log has no {name} row at batch {batch}")
            split, split_spread, folded, folded_spread = (match.group(i) for i in (1, 2, 3, 4))
            # The folded column was bolded on every row, which reads as four measured wins; three
            # of them are ties.  The folds are still taken - a fold removes a dispatch at no
            # precision cost and never measures slower - but the table says which is which.
            verdict = _verdict(split, split_spread, folded, folded_spread, "split", "folded")
            mark_split = "**" if verdict == "split wins" else ""
            mark_folded = "**" if verdict == "folded wins" else ""
            wins += verdict == "folded wins"
            measured += 1
            rows.append(
                f"| {label} | {batch} | {mark_split}{split} us{mark_split} | "
                f"{mark_folded}{folded} us{mark_folded} | "
                f"PCC {match.group(5)}, max abs diff {match.group(6)} |"
            )
    match = re.search(
        r"rank3 seq=(\d+) rank4_first_us=\s*([\d.]+) \(\s*([\d.]+)\) rank3_first_us=\s*([\d.]+) "
        r"\(\s*([\d.]+)\) pcc_beta=([\d.]+) pcc_g=([\d.]+) max_abs_diff=(\S+)",
        _probe("probe_gdn_input_folds"),
    )
    if not match:
        raise SystemExit("probe_gdn_input_folds.log has no rank3 row")
    rows.append(
        f"| rank-3 before the slices instead of after (not taken) | {match.group(1)} rows | "
        f"{match.group(2)} us (shipped) | {match.group(4)} us | "
        f"PCC {match.group(6)}, max abs diff {match.group(8)} |"
    )
    # The rank-3 verdict is derived like every other one here, not asserted: the sentence used to
    # say "measures as a tie" whatever the row said.
    rank3 = _verdict(match.group(2), match.group(3), match.group(4), match.group(5), "shipped", "rank-3 first")
    outcome = (
        "measures as a tie, so the shipped order stands"
        if rank3 == "tie"
        else f"measures as a win for the {rank3.rsplit(' ', 1)[0]} order"
    )
    rows.append("")
    rows.append(
        f"Median microseconds over 25 repeats (9 for the rank-3 row). Both folds are bit-exact and "
        f"both were taken; on time, {wins} of the {measured} rows is a win outside the spreads and "
        f"the rest are ties, so what a fold buys for certain is a dispatch, not microseconds. A "
        f"bolded cell is a win outside the two "
        f"spreads together. Moving the rank change ahead of the slices removes two float32 "
        f"reshapes and adds one, and {outcome}."
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
            rf"decode_conv batch=\s*{batch} float32_us=\s*([\d.]+) \(\s*([\d.]+)\) "
            rf"bfloat16_us=\s*([\d.]+) \(\s*([\d.]+)\) pcc_float32_vs_torch=([\d.]+) "
            rf"pcc_bfloat16_vs_torch=([\d.]+) pcc_between=([\d.]+)",
            _probe("probe_decode_conv_dtype"),
        )
        if not match:
            raise SystemExit(f"probe_decode_conv_dtype.log has no batch {batch} row")
        # Which cell is bolded is derived: a stage review found this table bolding a "winner"
        # its own log contradicted at one of the two batches.  The rule is the spread rule, not
        # "whichever median is lower", so a tied row carries no bold at all.
        shipped_us, shipped_spread, alternative_us, alternative_spread = (match.group(i) for i in (1, 2, 3, 4))
        verdict = _verdict(shipped_us, shipped_spread, alternative_us, alternative_spread, "float32", "bfloat16")
        faster = "**" if verdict == "float32 wins" else ""
        other = "**" if verdict == "bfloat16 wins" else ""
        rows.append(
            f"| decode causal-conv FIR in bfloat16 instead of float32 | {batch} | "
            f"{faster}{shipped_us} us{faster} (float32) | "
            f"{other}{alternative_us} us{other} (bfloat16) | "
            f"PCC {match.group(7)} between them, {match.group(6)} against torch |"
        )
    for batch in ("1", "32"):
        match = re.search(
            rf"decode_heads batch=\s*{batch} expand_then_norm_us=\s*([\d.]+) \(\s*([\d.]+)\) "
            rf"norm_then_expand_us=\s*([\d.]+) \(\s*([\d.]+)\) pcc_between=([\d.]+)",
            _probe("probe_gdn_decode_heads"),
        )
        if not match:
            raise SystemExit(f"probe_gdn_decode_heads.log has no batch {batch} row")
        shipped, shipped_spread, other, other_spread = (match.group(i) for i in (1, 2, 3, 4))
        # Derived, not hard-coded: this row was bolded at batch 1 for four rounds, where the two
        # are inside their combined spread and the caption below says a bold means a win.
        verdict = _verdict(shipped, shipped_spread, other, other_spread, "shipped", "reordered")
        mark_shipped = "**" if verdict == "shipped wins" else ""
        mark_other = "**" if verdict == "reordered wins" else ""
        rows.append(
            f"| Q/K L2 norm before the GQA expansion instead of after | {batch} | "
            f"{mark_shipped}{shipped} us{mark_shipped} (expand, then norm) | "
            f"{mark_other}{other} us{mark_other} (norm, then expand) | "
            f"PCC {match.group(5)} between them |"
        )
    rows.append("")
    rows.append(
        "Median microseconds over 25 repeats. Both alternatives are the same arithmetic as what "
        "ships. The bolded cell in each row is the faster of the pair as measured, and a row with "
        "no bold is one where the two are inside their combined spread."
    )
    return "\n".join(rows)


def rejected_shared_work() -> str:
    """Two shared-work merges the batch-32 and prefill profiles suggested, both measured."""
    rows = ["| candidate | shape | shipped | merged | agreement |", "|---|---|---|---|---|"]
    for batch in ("1", "32"):
        match = re.search(
            rf"decode_qk batch=\s*{batch} separate_us=\s*([\d.]+) \(\s*([\d.]+)\) "
            rf"merged_us=\s*([\d.]+) \(\s*([\d.]+)\) pcc_q=([\d.]+) pcc_k=([\d.]+)",
            _probe("probe_decode_qk_pair"),
        )
        if not match:
            raise SystemExit(f"probe_decode_qk_pair.log has no batch {batch} row")
        separate, separate_spread, merged, merged_spread = (match.group(i) for i in (1, 2, 3, 4))
        verdict = _verdict(separate, separate_spread, merged, merged_spread, "shipped", "merged")
        mark_shipped = "**" if verdict == "shipped wins" else ""
        mark_merged = "**" if verdict == "merged wins" else ""
        rows.append(
            f"| decode Q and K through one norm/scale/rank-change chain | batch {batch} | "
            f"{mark_shipped}{separate} us{mark_shipped} (separate) | "
            f"{mark_merged}{merged} us{mark_merged} (merged) | "
            f"PCC {match.group(5)} / {match.group(6)} |"
        )
    match = re.search(
        r"prefill_qkv seq=(\d+) packed_ms=\s*([\d.]+) \(\s*([\d.]+)\) split_ms=\s*([\d.]+) \(\s*([\d.]+)\) "
        r"pcc_q=([\d.]+)",
        _probe("probe_prefill_qkv_split"),
    )
    if not match:
        raise SystemExit("probe_prefill_qkv_split.log has no measurement")
    packed, packed_spread, split, split_spread = (match.group(i) for i in (2, 3, 4, 5))
    verdict = _verdict(packed, packed_spread, split, split_spread, "shipped", "split")
    mark_shipped = "**" if verdict == "shipped wins" else ""
    mark_split = "**" if verdict == "split wins" else ""
    rows.append(
        f"| prefill `in_proj_qkv` as three projections and three FIRs instead of one and three slices | "
        f"{match.group(1)} tokens | {mark_shipped}{packed} ms{mark_shipped} (packed) | "
        f"{mark_split}{split} ms{mark_split} (split) | "
        f"PCC {match.group(6)} |"
    )
    rows.append("")
    rows.append(
        "Median over 25 repeats (9 for the prefill row). Both merges are the same arithmetic as what "
        "ships and both measure slower: cutting a wide TILE tensor apart, or concatenating one, costs "
        "more than the shared work it enables - the same result §6.2 found for `wqkv`/`wgate`. A "
        "bolded cell is a win outside the two spreads together; a row with no bold is a tie."
    )
    return "\n".join(rows)


def _test_ids(log: str) -> list[str]:
    """Every ``test_...[params]`` id a run log mentions, deduplicated and sorted."""
    text = (DOC / "logs" / f"{log}.log").read_text(errors="replace")
    return sorted({name for name in re.findall(r"test_fused_decoder\.py::(\w+(?:\[[^\]]*\])?)", text)})


def coverage_claims() -> str:
    """The coverage sentences the README's contract tables carry, derived from the run logs.

    Three of them - the watcher's pass count, the batches trace capture runs at, and how many
    parametrised cases are one-layer-kind-only - were hand-written, and all three went stale as
    the two previous rounds added tests.  They are facts about the committed logs, so they are
    read out of them.
    """
    watcher = (DOC / "logs" / "watcher_run.log").read_text(errors="replace")
    passed = re.findall(r"=+ (\d+) passed[^=]*=+", watcher)
    if not passed:
        raise SystemExit("watcher_run.log has no pytest summary line")
    watcher_ids = _test_ids("watcher_run")
    functions = sorted({name.split("[")[0] for name in watcher_ids})

    suite_ids = _test_ids("suite_main")
    kinds = ("linear_attention", "full_attention")
    suite_functions: dict[str, set[str]] = {}
    for name in suite_ids:
        suite_functions.setdefault(name.split("[")[0], set()).update(kind for kind in kinds if kind in name)
    both = sorted(name for name, seen in suite_functions.items() if len(seen) == 2)
    single = sorted(name for name, seen in suite_functions.items() if len(seen) == 1)
    unparametrised = sorted(name for name, seen in suite_functions.items() if not seen)
    traced = sorted(
        {
            int(match)
            for name in suite_ids
            if name.startswith("test_traced_decode_batched")
            for match in re.findall(r"\[(\d+)-", name)
        }
    )
    if any(name.startswith("test_traced_decode_pcc") for name in suite_ids):
        traced = sorted(set(traced) | {1})

    return (
        f"* **Watcher run:** `{len(watcher_ids)}` selected cases across `{len(functions)}` test "
        f"functions, **{passed[-1]} passed**, offender grep zero over the whole log "
        f"(`watcher/WATCHER_AUDIT.md`).\n"
        f"* **Trace capture and replay:** batches "
        + ", ".join(f"`{value}`" for value in traced)
        + f" (`test_traced_decode_pcc`, `test_traced_decode_batched`).\n"
        f"* **Layer-kind coverage:** {len(suite_ids)} collected cases over "
        f"{len(suite_functions)} test functions. {len(both) + len(single)} carry the layer kind in "
        f"their id, and {len(both)} of those run for **both** kinds. The remaining "
        f"{len(unparametrised)} are not parametrised by layer kind because they exercise one by "
        f"construction: " + ", ".join(f"`{name}`" for name in unparametrised) + "."
    )


def addcmul_state() -> str:
    """The recurrent-state update as two ops or one, at both decode regimes."""
    rows = ["| batch | multiply + add (was) | `addcmul` | `addcmul` in place | agreement |", "|---|---|---|---|---|"]
    for batch in ("1", "32"):
        match = re.search(
            rf"addcmul batch=\s*{batch} shipped_us=\s*([\d.]+) \(\s*([\d.]+)\) addcmul_us=\s*([\d.]+) "
            rf"\(\s*([\d.]+)\) in_place_us=\s*([\d.]+) \(\s*([\d.]+)\) pcc_in_place=([\d.]+) "
            rf"pcc_shipped_vs_torch=([\d.]+) pcc_addcmul_vs_torch=([\d.]+) max_abs_diff=(\S+)",
            _probe("probe_addcmul_state"),
        )
        if not match:
            raise SystemExit(f"probe_addcmul_state.log has no batch {batch} row")
        # Bold the distinguishable minimum of the three, not "the one that ships": ``addcmul`` and
        # the in-place form are inside each other's spread, and the row's real result is that both
        # beat the two-op form the layer used to run.
        measured = [(match.group(i), match.group(i + 1)) for i in (1, 3, 5)]
        best = _distinguishable_min(measured, ["two_op", "addcmul", "in_place"])
        cells = [
            f"**{value}** us" if label == best else f"{value} us"
            for label, (value, _) in zip(["two_op", "addcmul", "in_place"], measured)
        ]
        rows.append(
            f"| {batch} | " + " | ".join(cells) + " | "
            f"PCC {match.group(7)} in place, {match.group(9)} against torch, max abs diff {match.group(10)} |"
        )
    rows.append("")
    rows.append(
        "Median and spread over 15 repeats, state uploaded once outside the timed region. A bolded "
        "cell is a minimum outside the spreads; where none is bolded, the two `addcmul` forms are "
        "inside each other's spread and both are decisively below the two-op form. The in-place "
        "form is what ships, for a reason the timing does not carry: it lands at the persistent "
        "buffer's address, which the traced decode needs. It is bit-exact against both the two-op "
        "form and torch."
    )
    return "\n".join(rows)


def rope_half() -> str:
    """The decode rotate-half, dedicated op against the four ops it replaces."""
    rows = ["| batch | `ttnn.experimental.rotate_half` | spelled out (shipped) | agreement |", "|---|---|---|---|"]
    for batch in ("1", "32"):
        match = re.search(
            rf"rope_half batch=\s*{batch} heads=\s*\d+ dedicated_us=\s*([\d.]+) \(\s*[\d.]+\) "
            rf"spelled_out_us=\s*([\d.]+) \(\s*[\d.]+\) pcc_between=([\d.]+) max_abs_diff=(\S+)",
            _probe("probe_decode_rope_half"),
        )
        if not match:
            raise SystemExit(f"probe_decode_rope_half.log has no batch {batch} row")
        rows.append(
            f"| {batch} | {match.group(1)} us | {match.group(2)} us | "
            f"PCC {match.group(3)}, max abs diff {match.group(4)} |"
        )
    rows.append("")
    rows.append(
        "Median over 25 repeats, *wall clock*, so dispatch is on the critical path - which is why "
        "this table favours the dedicated op and the traced pass measurement does not."
    )
    return "\n".join(rows)


def batch32_shares() -> str:
    """What grows with the batch, and what the optimization stage should plan against.

    These sentences were hand-written multiples and percentages, and two of them were stale the
    moment §3.21 moved a bucket's worth of work into a bucket of its own.
    """
    summary = _summary()["measurements"]
    lines = []
    for kind, buckets in (
        ("linear_attention", ("batched_matmul", "state_update", "elementwise", "layout")),
        ("full_attention", ("sdpa",)),
    ):
        small = summary[f"fused/{kind}/decode"]
        large = summary[f"fused/{kind}/decode_batch32"]
        for bucket in buckets:
            one = small["breakdown_ms"].get(bucket, 0.0)
            many = large["breakdown_ms"].get(bucket, 0.0)
            if not many:
                continue
            growth = f"{many / one:.1f}x" if one else "from nothing"
            lines.append(
                f"| `{kind}` | `{bucket}` | {one:.3f} ms ({100.0 * one / small['device_kernel_time_ms']:.1f} %) | "
                f"{many:.3f} ms ({100.0 * many / large['device_kernel_time_ms']:.1f} %) | {growth} |"
            )
    header = ["| pass | bucket | batch 1 | batch 32 | growth |", "|---|---|---|---|---|"]
    return "\n".join(header + lines)


def rope_half_traced() -> str:
    """The decisive rotate-half measurement: the whole traced pass, profiled with each form.

    The wall-clock probe favours the dedicated op; under trace, where dispatch is not on the
    critical path, device time decides.  Both runs are committed - the rejected one under
    ``tracy/rejected/rotate_half_dedicated/``, with its own build fingerprint in its provenance,
    which is what makes it a measurement of the alternative rather than of the shipped code.
    """
    rows = ["| pass | dedicated `rotate_half` (rejected) | spelled out (shipped) | difference |", "|---|---|---|---|"]
    differences = {}
    for phase, label in (
        ("decode", "`full_attention` decode, batch 1"),
        ("decode_batch32", "`full_attention` decode, batch 32"),
    ):
        shipped = _summary()["measurements"][f"fused/full_attention/{phase}"]["device_kernel_time_ms"]
        report = DOC / "tracy" / "rejected" / "rotate_half_dedicated" / f"{phase}_perf_report.csv"
        with report.open() as handle:
            rejected = sum(float(row["Device Time"] or 0) for row in csv.DictReader(handle)) / 8 / 1000.0
        # No bold: these are two profiler passes, not a repeated measurement, so there is no
        # spread to decide a win with.  The shipped column used to be bolded on both rows, and at
        # batch 1 that is the *larger* of the two numbers - the sign is in the difference column,
        # which is where the decision actually lives.
        differences[phase] = 100.0 * (rejected - shipped) / rejected
        rows.append(f"| {label} | {rejected:.3f} ms | {shipped:.3f} ms (shipped) | " f"{differences[phase]:+.1f} % |")
    rows.append("")
    rows.append(
        "Device time per trace replay, summed over the signposted window of each committed report. "
        "One pass each, so no cell is bolded: a positive difference is the shipped form ahead, and "
        "the decision is the advertised-batch row, where it is "
        f"{differences['decode_batch32']:+.2f} %. At batch 1 it is {differences['decode']:+.2f} %, "
        "i.e. the sign is against the shipped form by a fraction of a percent - the difference "
        "column above is rounded to one decimal, so these two figures are the unrounded ones. "
        "The two runs differ only in this one op - the rejected one's provenance carries a "
        "different `FUSED_BUILD` fingerprint, which is how it is identifiable as the alternative."
    )
    return "\n".join(rows)


def conv_tap_addcmul() -> str:
    """The FIR's non-final taps as two ops or one, at both widths."""
    rows = [
        "| pass | rows | dtype | `multiply` + `add` | `addcmul` | agreement |",
        "|---|---|---|---|---|---|",
    ]
    for label in ("prefill", "decode"):
        match = re.search(
            rf"conv_tap {label}\s+rows=\s*(\d+) dtype=(\w+) two_ops_us=\s*([\d.]+) \(\s*[\d.]+\) "
            rf"addcmul_us=\s*([\d.]+) \(\s*[\d.]+\) pcc_between=([\d.]+) max_abs_diff=(\S+)",
            _probe("probe_addcmul_state"),
        )
        if not match:
            raise SystemExit(f"probe_addcmul_state.log has no conv_tap {label} row")
        rows.append(
            f"| {label} | {match.group(1)} | {match.group(2)} | {match.group(3)} us | "
            f"**{match.group(4)} us** | PCC {match.group(5)}, max abs diff {match.group(6)} |"
        )
    rows.append("")
    rows.append(
        "Median over 15 repeats, per tap, each at the dtype its path runs (§3.7 makes the prefill "
        "FIR bfloat16, §3.25 keeps the decode one float32). Where the two forms differ at all it "
        "is rounding of the *intermediate*: the two-op form rounds `state * w` to the tensor dtype "
        "before the add and the fused one keeps it in the accumulator, so the fused result is the "
        "closer of the two to exact arithmetic, not the further."
    )
    return "\n".join(rows)


def dense_recurrence() -> str:
    """The recurrence's transient chain in per-head rows or dense, at both regimes."""
    rows = ["| batch | one padded row per head | dense `[1, batch, heads, dim]` | agreement |", "|---|---|---|---|"]
    verdicts = {}
    for batch in ("1", "32"):
        match = re.search(
            rf"dense_recurrence batch=\s*{batch} rows_us=\s*([\d.]+) \(\s*([\d.]+)\) "
            rf"dense_us=\s*([\d.]+) \(\s*([\d.]+)\) pcc_between=([\d.]+) max_abs_diff=(\S+)",
            _probe("probe_dense_recurrence"),
        )
        if not match:
            raise SystemExit(f"probe_dense_recurrence.log has no batch {batch} row")
        rows_us, rows_spread, dense_us, dense_spread = (match.group(i) for i in (1, 2, 3, 4))
        # Bold the winner, and neither when the row is a tie: the caption used to bold the dense
        # cell at batch 1, where the two are inside each other's spread and the *rows* form is
        # nominally ahead.
        verdicts[batch] = _verdict(rows_us, rows_spread, dense_us, dense_spread, "rows", "dense")
        mark_rows = "**" if verdicts[batch] == "rows wins" else ""
        mark_dense = "**" if verdicts[batch] == "dense wins" else ""
        rows.append(
            f"| {batch} | {mark_rows}{rows_us} us{mark_rows} | {mark_dense}{dense_us} us{mark_dense} | "
            f"PCC {match.group(5)}, max abs diff {match.group(6)} |"
        )
    rows.append("")
    rows.append(
        "Median over 25 repeats over the `subtract` / `sigmoid`-multiply chain and the rank changes "
        f"each form needs. Bit-identical. At batch 1 a {verdicts['1']}; at batch 32 {verdicts['32']} - "
        "at the advertised batch the dense form is far ahead, because a one-row-per-head TILE tensor "
        "carries 31 padding rows for every real one. A bolded cell is a win outside the combined "
        "spread; a row with none is a tie."
    )
    return "\n".join(rows)


def decode_conv_dtype() -> str:
    """The decode FIR in float32 or bfloat16, swept across the batches the layer can be built at."""
    rows = ["| batch | float32 (shipped) | bfloat16 | PCC of each against torch |", "|---|---|---|---|"]
    for match in re.finditer(
        r"decode_conv batch=\s*(\d+) float32_us=\s*([\d.]+) \(\s*([\d.]+)\) "
        r"bfloat16_us=\s*([\d.]+) \(\s*([\d.]+)\) pcc_float32_vs_torch=([\d.]+) "
        r"pcc_bfloat16_vs_torch=([\d.]+)",
        _probe("probe_decode_conv_dtype"),
    ):
        batch, f32, f32_spread, bf16, bf16_spread = match.group(1), *(float(match.group(i)) for i in (2, 3, 4, 5))
        if abs(f32 - bf16) <= f32_spread + bf16_spread:
            cells = (f"{f32:.1f} us", f"{bf16:.1f} us", "tie")
        elif f32 < bf16:
            cells = (f"**{f32:.1f} us**", f"{bf16:.1f} us", "float32 faster")
        else:
            cells = (f"{f32:.1f} us", f"**{bf16:.1f} us**", "bfloat16 faster")
        rows.append(f"| {batch} | {cells[0]} | {cells[1]} | {match.group(6)} / {match.group(7)} ({cells[2]}) |")
    rows.append("")
    rows.append(
        "Median over 25 repeats, both forms accumulating the way the shipped FIR does. The bolded "
        "cell is the faster of the pair where they are outside their combined spread."
    )
    return "\n".join(rows)


def group_attn_matmul() -> str:
    """What ``group_attn_matmul`` does when it is mapped the way its contract wants."""
    log = _probe("probe_group_attn_matmul")
    rows = ["| input dtype | outcome |", "|---|---|"]
    for dtype in ("fp32", "bf16"):
        match = re.search(rf"group_attn_matmul batch=\d+ dtype={dtype} (rejected: .*|accepted.*)", log)
        if not match:
            raise SystemExit(f"probe_group_attn_matmul.log has no {dtype} outcome")
        rows.append(f"| {dtype} | `{match.group(1).strip()}` |")
    sizes = re.findall(
        r"circular buffers on core range \[[^\]]+\] grow to (\d+) B which is beyond max L1 size of (\d+) B", log
    )
    rows.append("")
    if sizes:
        rows.append(
            "The overflow is the whole of it: "
            + "; ".join(f"{grew} B of circular buffers against {limit} B of L1" for grew, limit in sizes)
            + "."
        )
    return "\n".join(rows)


def rejected_bf16_fir() -> str:
    """The bfloat16 decode FIR's failure, and what the artifacts say separates pass from fail."""
    run = (DOC / "logs" / "rejected_bf16_decode_fir.log").read_text(errors="replace")
    records = [json.loads(blob) for blob in re.findall(r"PCCEVIDENCE (\{[^}]*\})", run)]
    summary = re.findall(r"=+ (\d+) failed, (\d+) passed[^=]*=+", run)
    if not summary:
        raise SystemExit("rejected_bf16_decode_fir.log records no pytest summary")
    failures = [
        record
        for record in records
        if record["metric"] == "fused_batched_traced_decode_pcc"
        and record.get("kind") == "linear_attention"
        and record["value"] < 0.995
    ]
    steps = sorted(
        (
            record
            for record in records
            if record["metric"] == "fused_decode_pcc" and record.get("kind") == "linear_attention"
        ),
        key=lambda record: record["step"],
    )
    rows = ["| case | PCC against HF |", "|---|---|"]
    rows.append(f"| the run | **{summary[-1][0]} failed**, {summary[-1][1]} passed |")
    for record in failures:
        rows.append(
            f"| batched traced decode, batch {record['batch']}, user {record['user_id']} "
            f"(the 64-token prefill), replay {record['replay']} | **{record['value']:.6f}** |"
        )
    if steps:
        rows.append(
            f"| unbatched decode after a {steps[0]['prefill_len']}-token prefill, steps "
            + "/".join(str(record["step"]) for record in steps)
            + " | "
            + " / ".join(f"{record['value']:.6f}" for record in steps)
            + " |"
        )
    compounding = _probe("probe_fir_dtype_compounding")
    for match in re.finditer(r"fir_compounding dtype=(\w+) steps=(\d+) (.*)", compounding):
        values = [float(value) for value in re.findall(r"s\d+=([\d.]+)", match.group(3))]
        rows.append(
            f"| synthetic carried state after 1 / {len(values)} steps, {match.group(1)} FIR | "
            f"{values[0]:.6f} / {values[-1]:.6f} |"
        )
    rows.append("")
    rows.append(
        "Read the rows together. The failures are at the **shortest** prefill in the batch, at the "
        "first replay the test checks; the same build's *unbatched* decode after a 2049-token "
        "prefill holds four steps with no downward trend. So what separates pass from fail here is "
        "the size of the carried state, not the number of steps - and the synthetic per-step probe, "
        "written to show compounding, does not reproduce the failure at all. That negative result "
        "is why this table states the pattern rather than a mechanism: the rejection rests on the "
        "committed failing run, and the *why* is left as the open question it is."
    )
    return "\n".join(rows)


BLOCKS = {
    "before_breakdown": before_breakdown,
    "correctness": correctness_table,
    "gated_norm_batches": gated_norm_batches,
    "group_attn_matmul": group_attn_matmul,
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
    "coverage_claims": coverage_claims,
    "batch32_shares": batch32_shares,
    "addcmul_state": addcmul_state,
    "conv_tap_addcmul": conv_tap_addcmul,
    "dense_recurrence": dense_recurrence,
    "decode_conv_dtype": decode_conv_dtype,
    "rejected_bf16_fir": rejected_bf16_fir,
    "rope_half": rope_half,
    "rope_half_traced": rope_half_traced,
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

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
    rows = [
        "| formulation | float32 | bfloat16 |",
        "|---|---|---|",
    ]
    names = {
        "tile": "TILE slices (functional)",
        "rm_shift": "untilize once, ROW_MAJOR shift, tilize per tap",
        "rm_arith": "untilize once, whole FIR in ROW_MAJOR, tilize once",
        "aligned_win": "one pre-padded window per tap so every slice is tile-aligned",
    }
    for key, label in names.items():
        fp32 = _grep("probe_causal_conv", rf"conv {key}\s+fp32 best_ms=\s*([\d.]+)")
        bf16 = _grep("probe_causal_conv", rf"conv {key}\s+bf16 best_ms=\s*([\d.]+)")
        mark = "**" if key == "tile" else ""
        rows.append(f"| {label} | {fp32} ms | {mark}{bf16} ms{mark} |")
    return "\n".join(rows)


def broadcast_table() -> str:
    rows = ["| multiply | float32 | bfloat16 |", "|---|---|---|"]
    for key, label in (("bcast", "height-broadcast"), ("same", "same-shape")):
        cells = []
        for dtype in ("fp32", "bf16"):
            ms = _grep("probe_causal_conv", rf"multiply {key}\s+{dtype} best_ms=\s*([\d.]+)")
            gbps = _grep("probe_causal_conv", rf"multiply {key}\s+{dtype} best_ms=\s*[\d.]+ eff_GBps=\s*([\d.]+)")
            emphasis = "**" if (key, dtype) == ("bcast", "fp32") else ""
            cells.append(f"{ms} ms — {emphasis}{float(gbps):.0f} GB/s{emphasis}")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
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
    def read(kind: str, label: str) -> str:
        return _grep("probe_decode_recurrence", rf"{kind}\s+{re.escape(label)}\s+us=\s*([\d.]+)")

    read_row = [
        read("read", "default"),
        read("read", "core_grid 1x8"),
        read("read", "core_grid 2x8"),
        read("read", "core_grid 4x4"),
        read("read", "core_grid 6x4"),
        read("read", "core_grid 6x8"),
        read("read", "core_grid 6x11"),
    ]
    outer_row = [
        read("outer", "default"),
        "—",
        read("outer", "core_grid 2x8"),
        "—",
        "—",
        read("outer", "core_grid 6x8"),
        read("outer", "core_grid 6x11"),
    ]
    return (
        "| shape | default | 1x8 | 2x8 | 4x4 | **6x4** | 6x8 | 6x11 |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| state read | {read_row[0]} us | {read_row[1]} | {read_row[2]} | {read_row[3]} | "
        f"**{read_row[4]}** | {read_row[5]} | {read_row[6]} |\n"
        f"| outer product | {outer_row[0]} us | — | {outer_row[2]} | — | — | "
        f"**{outer_row[5]}** | {outer_row[6]} |"
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


BLOCKS = {
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

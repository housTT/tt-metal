# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regenerate every ``<!-- GENERATED:... -->`` block in ``doc/optimized_decoder/``.

Each block is written from a committed artifact - ``perf_summary.json``, ``pcc_evidence.json`` or a
probe log's ``PROBEROW`` lines - so no figure in the stage documents is transcribed by hand and a
re-measurement cannot leave a table stale.  ``tests/test_optimized_decoder_docs.py`` re-runs this
generator into a temporary copy and fails if the committed documents differ from what it produces.

Reads only committed artifacts; opens no device.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/make_doc_tables.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1]
KINDS = ("linear_attention", "full_attention")
PHASES = ("prefill", "decode", "decode_batch32")
PHASE_LABEL = {
    "prefill": "prefill, 2048 tokens",
    "decode": "traced decode, 1 token, batch 1",
    "decode_batch32": "traced decode, 1 token, batch 32 (advertised `max_batch`)",
}


def _summary() -> dict:
    return json.loads((DOC / "perf_summary.json").read_text())


def _evidence() -> dict:
    path = DOC / "pcc_evidence.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _probe_rows(name: str) -> list:
    path = DOC / "logs" / name
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("PROBEROW "):
            try:
                rows.append(json.loads(line[len("PROBEROW ") :]))
            except json.JSONDecodeError:
                continue
    return rows


def _fmt(value, digits=4, suffix=" ms"):
    if value is None:
        return "—"
    return f"{value:.{digits}f}{suffix}"


# ------------------------------------------------------------------- generators


def before_after(summary: dict) -> list:
    lines = [
        "| layer kind | phase | device time before | device time after | speed-up | "
        "end-to-end before | end-to-end after | ops before | ops after |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for kind in KINDS:
        for phase in PHASES:
            key = f"{kind}/{phase}"
            row = summary["speedup"].get(key)
            if not row:
                continue
            lines.append(
                f"| `{kind}` | {PHASE_LABEL[phase]} | {_fmt(row['device_ms_before'], 3)} | "
                f"**{_fmt(row['device_ms_after'], 3)}** | **{row['speedup_x']:.2f}x** | "
                f"{_fmt(row.get('wall_ms_before'), 3)} | {_fmt(row.get('wall_ms_after'), 3)} | "
                f"{row['ops_before']} | {row['ops_after']} |"
            )
    return lines


def breakdown(summary: dict) -> list:
    columns = [(kind, phase) for kind in KINDS for phase in PHASES]
    columns = [c for c in columns if f"optimized/{c[0]}/{c[1]}" in summary["measurements"]]
    header = "| bucket | " + " | ".join(f"`{kind}` {phase}" for kind, phase in columns) + " |"
    lines = [header, "|" + "---|" * (len(columns) + 1)]
    buckets: list = []
    for kind, phase in columns:
        for name in summary["measurements"][f"optimized/{kind}/{phase}"]["breakdown_ms"]:
            if name not in buckets:
                buckets.append(name)
    for name in buckets:
        cells = []
        for kind, phase in columns:
            value = summary["measurements"][f"optimized/{kind}/{phase}"]["breakdown_ms"].get(name)
            cells.append(_fmt(value, 4) if value is not None else "—")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    totals = [
        _fmt(summary["measurements"][f"optimized/{kind}/{phase}"]["device_kernel_time_ms"], 4)
        for kind, phase in columns
    ]
    lines.append("| **total** | " + " | ".join(f"**{value}**" for value in totals) + " |")
    empty_other = all(
        summary["measurements"][f"optimized/{kind}/{phase}"]["breakdown_ms"].get("other", 0.0) == 0.0
        for kind, phase in columns
    )
    lines.append("")
    lines.append(
        "Every op is classified: the `other` bucket is empty in all measured passes."
        if empty_other
        else "The `other` bucket is **not** empty; an op code fell through the classifier."
    )
    return lines


def isolation(rows: list) -> list:
    """The precision/layout 2x2, from ``probe_optimized.py isolation``'s own log.

    Wall-clock from the in-model harness rather than profiler device time, because these four arms
    exist to attribute the win between two levers and that only needs one consistent clock; the
    headline before/after numbers above are the profiler's.
    """
    selected = [row for row in rows if row.get("sweep") == "isolation"]
    if not selected:
        return ["_no rows: probe_optimized_isolation.log is not committed_"]
    kinds = []
    for row in selected:
        if row["kind"] not in kinds:
            kinds.append(row["kind"])
    arms = []
    for row in selected:
        if row["candidate"] not in arms:
            arms.append(row["candidate"])
    header = "| arm | " + " | ".join(f"`{k}` traced decode | `{k}` prefill" for k in kinds) + " |"
    lines = [header, "|" + "---|" * (1 + 2 * len(kinds))]
    blockers: dict = {}
    for arm in arms:
        cells = []
        for kind in kinds:
            match = next((r for r in selected if r["kind"] == kind and r["candidate"] == arm), None)
            # An arm that does not allocate is a *result*, and "—" does not say so.  The cell names the
            # blocker and the note below the table carries the op's own message, because an empty cell
            # in an isolation table reads as "not tried".
            if match and match.get("error"):
                blockers[(arm, kind)] = str(match["error"])
                cells.extend(["**does not allocate**", "**does not allocate**"])
                continue
            cells.append(_MS(match["decode_ms"]) if match and match.get("decode_ms") else "—")
            cells.append(_MS3(match["prefill_ms"]) if match and match.get("prefill_ms") else "—")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    if blockers:
        lines.append("")
        for (arm, kind), error in sorted(blockers.items()):
            lines.append(f"*{arm}* on `{kind}` does not allocate: `{error[:200]}`")
    return lines


def dominant_matmuls(summary: dict) -> list:
    """Per pass, the largest matmul rows with the dtype and fidelity the profiler measured.

    This is the OPT-013 artifact: the shipped policy is only implemented if these rows say so.
    """
    lines = [
        "| pass | op | instances per pass | device time per pass | math fidelity (measured) | "
        "bound | cores | DRAM % | FLOPs % |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for kind in KINDS:
        for phase in PHASES:
            entry = summary["measurements"].get(f"optimized/{kind}/{phase}")
            if not entry:
                continue
            replays = 8 if phase.startswith("decode") else 1
            for row in entry["dominant_matmul_rows"][:6]:
                lines.append(
                    f"| `{kind}` {phase} | `{row['op']}` | {max(1, row['instances'] // replays)} | "
                    f"{row['device_time_us'] / replays:.1f} us | `{row['math_fidelity']}` | "
                    f"{row['bound'] or '—'} | {_span([row['cores']], 0)} | "
                    f"{_span([row['dram_pct']], 1)} | {_span([row['flops_pct']], 1)} |"
                )
    return lines


def accounting(summary: dict) -> list:
    lines = [
        "| pass | bytes per step | implied peak DRAM | roofline | device time | end-to-end | "
        "op-to-op gap | fraction of roofline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, value in summary["accounting"].items():
        lines.append(
            f"| `{key}` | {value['total_bytes_per_step'] / 1e6:.1f} MB | "
            f"{value['peak_dram_gbs_implied_by_report']} GB/s | "
            f"{_fmt(value['roofline_ms_per_step_estimate'], 4)} | "
            f"{_fmt(value['decode_ms_per_step_device'], 4)} | "
            f"{_fmt(value['decode_ms_per_step_e2e'], 4)} | "
            f"{_fmt(value['op_to_op_gap_ms'], 4)} | "
            f"{value['roofline_fraction_achieved']} |"
        )
    return lines


def _sweep_table(rows: list, sweep: str, columns: tuple) -> list:
    """Generic ``PROBEROW`` table: one row per candidate, per layer kind."""
    selected = [row for row in rows if row.get("sweep") == sweep]
    if not selected:
        return ["_no rows: the probe log for this sweep is not committed_"]
    header = "| layer kind | candidate | " + " | ".join(label for label, _key, _fmt_ in columns) + " |"
    lines = [header, "|" + "---|" * (len(columns) + 2)]
    for row in selected:
        cells = []
        for _label, key, formatter in columns:
            cells.append(
                formatter(row.get(key)) if row.get(key) is not None else ("ERROR" if row.get("error") else "—")
            )
        lines.append(f"| `{row.get('kind', '?')}` | {row['candidate']} | " + " | ".join(cells) + " |")
        if row.get("error"):
            lines.append(f"| | ↳ blocker | {row['error']} |" + " |" * (len(columns) - 1))
    return lines


_MS = lambda value: f"{value:.4f} ms"  # noqa: E731
_MS3 = lambda value: f"{value:.3f} ms"  # noqa: E731
_PCC = lambda value: f"{value:.6f}"  # noqa: E731
_US = lambda value: f"{value:.1f} us"  # noqa: E731


def policy_sweep(rows: list) -> list:
    return _sweep_table(
        rows,
        "policy",
        (
            ("prefill", "prefill_ms", _MS3),
            ("traced decode b1", "decode_ms", _MS),
            ("prefill PCC", "prefill_pcc", _PCC),
            ("decode PCC", "decode_pcc", _PCC),
        ),
    )


def geometry_sweep(rows: list) -> list:
    return _sweep_table(
        rows,
        "geometry",
        (
            ("traced decode b1", "decode_ms", _MS),
            ("prefill PCC", "prefill_pcc", _PCC),
            ("decode PCC", "decode_pcc", _PCC),
        ),
    )


def in0_block_w_sweep(rows: list) -> list:
    return _sweep_table(rows, "in0_block_w", (("traced decode b1", "decode_ms", _MS),))


def prefill_sweep(rows: list) -> list:
    return _sweep_table(
        rows,
        "prefill",
        (("prefill", "prefill_ms", _MS3), ("prefill PCC", "prefill_pcc", _PCC), ("decode PCC", "decode_pcc", _PCC)),
    )


def real_weight_policy(rows: list) -> list:
    """Real-checkpoint PCC per precision candidate - the table that decides the policy."""
    return _sweep_table(
        rows,
        "real_weight_policy",
        (
            ("prefill PCC @2049", "prefill_pcc", _PCC),
            ("decode PCC, 4 steps", "decode_pcc", _PCC),
            ("traced decode PCC, 5 replays", "traced_decode_pcc", _PCC),
            # ``in_proj_qkv`` candidates change the tensor the recurrence *carries*, so the output PCC
            # alone cannot clear them - these two columns are the state itself.
            ("conv state PCC", "conv_state_pcc", _PCC),
            ("recurrent state PCC", "recurrent_state_pcc", _PCC),
        ),
    )


def long_context_fp32acc(rows: list) -> list:
    """The full-context verification of the float32-destination-accumulation fix."""
    selected = [r for r in rows if r.get("sweep") == "long_context_precision"]
    if not selected:
        return ["_no rows: probe_long_context_fp32acc.log is not committed_"]
    lines = [
        "| arm | tail PCC | tail scale | decode PCC | decode scale | paged K PCC | paged V PCC | blocker |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in selected:
        lines.append(
            f"| {row['candidate']} | "
            f"{_PCC(row['prefill_tail_pcc']) if row.get('prefill_tail_pcc') is not None else '—'} | "
            f"{_PCC(row['prefill_tail_scale']) if row.get('prefill_tail_scale') is not None else '—'} | "
            f"{_PCC(row['decode_pcc']) if row.get('decode_pcc') is not None else '—'} | "
            f"{_PCC(row['decode_scale']) if row.get('decode_scale') is not None else '—'} | "
            f"{_PCC(row['paged_k_cache_pcc']) if row.get('paged_k_cache_pcc') is not None else '—'} | "
            f"{_PCC(row['paged_v_cache_pcc']) if row.get('paged_v_cache_pcc') is not None else '—'} | "
            f"{(row.get('error') or '—')[:90]} |"
        )
    return lines


def _long_context_table(rows: list, sweep: str, real: bool, columns: tuple) -> list:
    """One arm per row for a full-context sweep, filtered by weight source."""
    selected = [r for r in rows if r.get("sweep") == sweep and bool(r.get("real_weights")) == real]
    if not selected:
        return [f"_no rows: the {'real' if real else 'synthetic'}-weight log for this sweep is not committed_"]
    lines = [
        "| arm | " + " | ".join(label for label, _k in columns) + " | blocker |",
        "|" + "---|" * (len(columns) + 2),
    ]
    for row in selected:
        cells = [_PCC(row[key]) if row.get(key) is not None else "—" for _label, key in columns]
        lines.append(f"| {row['candidate']} | " + " | ".join(cells) + f" | {(row.get('error') or '—')[:90]} |")
    return lines


#: The columns of the two full-context tables.  The *scale* columns are the ones that matter here: at
#: this context PCC stays above the acceptance bar in every arm, and the gate that moves is the scale.
_FULL_COLUMNS = (
    ("tail PCC", "prefill_tail_pcc"),
    ("tail scale", "prefill_tail_scale"),
    ("decode PCC", "decode_pcc"),
    ("decode scale", "decode_scale"),
    ("paged K PCC", "paged_k_cache_pcc"),
    ("paged V PCC", "paged_v_cache_pcc"),
    ("K scale", "paged_k_cache_scale"),
    ("V scale", "paged_v_cache_scale"),
)
_LINEAR_COLUMNS = (
    ("tail PCC", "prefill_tail_pcc"),
    ("tail scale", "prefill_tail_scale"),
    ("decode PCC", "decode_pcc"),
    ("decode scale", "decode_scale"),
    ("conv PCC", "conv_state_pcc"),
    ("recurrent PCC", "recurrent_state_pcc"),
    ("recurrent scale", "recurrent_state_scale"),
)


def fidelity_gain(rows: list) -> list:
    """The model-free mechanism: a matmul's systematic gain by fidelity and operand dtype."""
    selected = [r for r in rows if r.get("sweep") == "fidelity_gain"]
    if not selected:
        return ["_no rows: probe_fidelity_gain.log is not committed_"]
    depths = sorted({r["K"] for r in selected})
    lines = [
        "| weight dtype | fidelity | fp32 dest acc | "
        + " | ".join(f"gain at K={d}" for d in depths)
        + " | PCC at K=5120 |",
        "|" + "---|" * (len(depths) + 4),
    ]
    for dtype in ("bfp4", "bfp8", "bf16"):
        for fid in ("LoFi", "HiFi2", "HiFi4"):
            for acc in (False, True):
                cells = []
                pcc_cell = "—"
                for depth in depths:
                    match = next(
                        (
                            r
                            for r in selected
                            if r["K"] == depth
                            and r["weight_dtype"] == dtype
                            and r["fidelity"] == fid
                            and bool(r["fp32_dest_acc"]) == acc
                        ),
                        None,
                    )
                    cells.append(f"{match['gain']:.6f}" if match and match.get("gain") is not None else "—")
                    if match and depth == 5120 and match.get("pcc") is not None:
                        pcc_cell = f"{match['pcc']:.6f}"
                if any(c != "—" for c in cells):
                    lines.append(f"| {dtype} | {fid} | {str(acc).lower()} | " + " | ".join(cells) + f" | {pcc_cell} |")
    return lines


def scale_vs_length(rows: list) -> list:
    """Is the scale error length-dependent, or has it been there at every length?"""
    selected = [r for r in rows if r.get("sweep") == "scale_vs_length"]
    if not selected:
        return ["_no rows: probe_scale_vs_length.log is not committed_"]
    lengths = sorted({r["seq_len"] for r in selected})
    arms = []
    for row in selected:
        if row["candidate"] not in arms:
            arms.append(row["candidate"])
    lines = [
        "| arm | metric | " + " | ".join(str(n) for n in lengths) + " |",
        "|" + "---|" * (len(lengths) + 2),
    ]
    for arm in arms:
        for label, key in (("prefill scale", "prefill_scale"), ("V cache scale", "paged_v_cache_scale")):
            cells = []
            for n in lengths:
                match = next((r for r in selected if r["candidate"] == arm and r["seq_len"] == n), None)
                cells.append(_PCC(match[key]) if match and match.get(key) is not None else "—")
            lines.append(f"| {arm} | {label} | " + " | ".join(cells) + " |")
    return lines


def prefill_fidelity_roles(rows: list) -> list:
    """The smallest prefill-only fidelity change that brings the full context inside the gate."""
    selected = [r for r in rows if r.get("sweep") == "prefill_fidelity_roles"]
    if not selected:
        return ["_no rows: probe_prefill_fidelity_roles.log is not committed_"]
    lines = [
        "| arm | tail PCC | tail scale | decode PCC | decode scale | K scale | V scale | blocker |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in selected:
        cells = [
            _PCC(row[key]) if row.get(key) is not None else "—"
            for key in (
                "prefill_tail_pcc",
                "prefill_tail_scale",
                "decode_pcc",
                "decode_scale",
                "paged_k_cache_scale",
                "paged_v_cache_scale",
            )
        ]
        lines.append(f"| {row['candidate']} | " + " | ".join(cells) + f" | {(row.get('error') or '—')[:80]} |")
    return lines


def prefill_fidelity_cost(rows: list) -> list:
    """What a prefill-only fidelity raise costs, per role set and per layer kind."""
    selected = [r for r in rows if r.get("sweep") == "prefill_fidelity_cost"]
    if not selected:
        return ["_no rows: probe_prefill_fidelity_cost.log is not committed_"]
    arms = []
    for row in selected:
        if row["candidate"] not in arms:
            arms.append(row["candidate"])
    lines = [
        "| arm | `linear_attention` prefill | `full_attention` prefill |",
        "|---|---|---|",
    ]
    for arm in arms:
        cells = []
        for kind in KINDS:
            match = next((r for r in selected if r["kind"] == kind and r["candidate"] == arm), None)
            cells.append(_MS3(match["prefill_ms"]) if match and match.get("prefill_ms") else "—")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return lines


def prefill_fidelity_mixed(rows: list) -> list:
    """Accuracy per millisecond: mixed per-role prefill fidelities, scale and cost together."""
    selected = [r for r in rows if r.get("sweep") == "prefill_fidelity_mixed"]
    if not selected:
        return ["_no rows: probe_prefill_fidelity_mixed.log is not committed_"]
    lines = [
        "| arm | tail scale | decode scale | V cache scale | tail PCC | "
        "`linear_attention` prefill | `full_attention` prefill |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in selected:
        lines.append(
            f"| {row['candidate']} | "
            f"{_PCC(row['prefill_tail_scale']) if row.get('prefill_tail_scale') is not None else '—'} | "
            f"{_PCC(row['decode_scale']) if row.get('decode_scale') is not None else '—'} | "
            f"{_PCC(row['paged_v_cache_scale']) if row.get('paged_v_cache_scale') is not None else '—'} | "
            f"{_PCC(row['prefill_tail_pcc']) if row.get('prefill_tail_pcc') is not None else '—'} | "
            f"{_MS3(row['prefill_ms_linear_attention']) if row.get('prefill_ms_linear_attention') else '—'} | "
            f"{_MS3(row['prefill_ms_full_attention']) if row.get('prefill_ms_full_attention') else '—'} |"
        )
    return lines


def long_context_real_full(rows: list) -> list:
    return _long_context_table(rows, "long_context_precision", True, _FULL_COLUMNS)


def long_context_real_linear(rows: list) -> list:
    return _long_context_table(rows, "long_context_linear", True, _LINEAR_COLUMNS)


def long_context_linear(rows: list) -> list:
    """The ``linear_attention`` half of the full-context attribution."""
    selected = [r for r in rows if r.get("sweep") == "long_context_linear"]
    if not selected:
        return ["_no rows: probe_long_context_linear.log is not committed_"]
    lines = [
        "| arm | tail PCC | tail scale | decode PCC | decode scale | conv PCC | recurrent PCC | recurrent scale | blocker |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in selected:
        cells = [
            row["candidate"],
            *(
                _PCC(row[key]) if row.get(key) is not None else "—"
                for key in (
                    "prefill_tail_pcc",
                    "prefill_tail_scale",
                    "decode_pcc",
                    "decode_scale",
                    "conv_state_pcc",
                    "recurrent_state_pcc",
                    "recurrent_state_scale",
                )
            ),
            (row.get("error") or "—")[:90],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def stream_grid(rows: list) -> list:
    """Rectangular against row-wise, on all four axes the choice trades between."""
    selected = [r for r in rows if r.get("sweep") == "stream_grid"]
    if not selected:
        return ["_no rows: probe_stream_grid.log is not committed_"]
    lines = [
        "| layer kind | batch | stream core grid | traced decode | reshards | INTERLEAVED reshape "
        "fallbacks | computed-vs-provided mismatches | decode PCC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in selected:
        lines.append(
            f"| `{row['kind']}` | {row['batch']} | {row['candidate']} | "
            f"{_MS(row['decode_ms']) if row.get('decode_ms') else 'ERROR'} | "
            f"{row.get('reshards', '—')} | {row.get('interleaved_reshape_fallbacks', '—')} | "
            f"{row.get('computed_vs_provided_mismatches', '—')} | "
            f"{_PCC(row['decode_pcc']) if row.get('decode_pcc') is not None else '—'} |"
        )
        if row.get("error"):
            lines.append(f"| | ↳ blocker | {row['error']} | | | | | |")
    return lines


def norm_repeat_order(rows: list) -> list:
    """The two layout ops the q/k head expansion used to cost, at both decode regimes."""
    return _sweep_table(
        rows,
        "norm_repeat_order",
        (
            ("batch", "batch", lambda value: str(value)),
            ("traced decode", "decode_ms", _MS),
            ("prefill PCC", "prefill_pcc", _PCC),
            ("decode PCC", "decode_pcc", _PCC),
        ),
    )


def prefill_grid_alignment(rows: list) -> list:
    """Why a prefill 2D matmul over a DRAM width-sharded weight needs one column per DRAM bank."""
    selected = [r for r in rows if r.get("sweep") == "prefill_grid_alignment"]
    if not selected:
        return ["_no rows: probe_prefill_grid_alignment.log is not committed_"]
    lines = [
        "| role | K | N | compute columns | DRAM banks | per_core_N | finite | PCC vs the heuristic | blocker |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in selected:
        finite = {True: "yes", False: "**no**", None: "—"}[row.get("finite")]
        lines.append(
            f"| `{row['role']}` | {row['K']} | {row['N']} | "
            f"{row['columns']}{' **(= banks)**' if row['columns'] == row['dram_banks'] else ''} | "
            f"{row['dram_banks']} | {row['per_core_N']} | {finite} | "
            f"{_PCC(row['pcc_vs_heuristic']) if row.get('pcc_vs_heuristic') is not None else '—'} | "
            f"{(row.get('error') or '—')[:80]} |"
        )
    return lines


def bfp4_gateup(rows: list) -> list:
    """Every legal program config for the one ``Bound=SLOW`` decode row that is material."""
    selected = [r for r in rows if r.get("sweep") == "bfp4_gateup"]
    if not selected:
        return ["_no rows: probe_bfp4_gateup.log is not committed_"]
    lines = [
        "| shape | dtype | candidate | cores | in0_block_w | per_core_N | median | PCC vs float32 | blocker |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in selected:
        lines.append(
            f"| {row['shape']} `{row['M']}x{row['K']}x{row['N']}` | {row['dtype']} | {row['candidate']} | "
            f"{row.get('cores', '—')} | {row.get('in0_block_w', '—')} | {row.get('per_core_N', '—')} | "
            f"{_US(row['median_us']) if row.get('median_us') else '—'} | "
            f"{_PCC(row['pcc']) if row.get('pcc') is not None else '—'} | {row.get('error') or '—'} |"
        )
    return lines


def recurrence_advice(rows: list) -> list:
    """Every remaining piece of report advice on the recurrence rows, tried and measured."""
    selected = [r for r in rows if r.get("sweep") == "recurrence_advice"]
    if not selected:
        return ["_no rows: probe_recurrence_advice.log is not committed_"]
    lines = [
        "| head problems | candidate | median | PCC vs float32 | blocker |",
        "|---|---|---|---|---|",
    ]
    for row in selected:
        lines.append(
            f"| {row['head_problems']} (batch {row['batch']}) | {row['candidate']} | "
            f"{_US(row['median_us']) if row.get('median_us') else '—'} | "
            f"{_PCC(row['pcc']) if row.get('pcc') is not None else '—'} | {row.get('error') or '—'} |"
        )
    return lines


def _span(values, digits: int) -> str:
    """``min-max`` over a set of numeric strings, rounded - one cell per op group, not per instance.

    An op group can have many instances in a window (one per trace replay, and more than one per
    replay for a repeated role), and printing every instance's percentage made this table unreadable.
    A range says the same thing and says it in one cell.
    """
    numbers = []
    for value in values:
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numbers:
        return "—"
    low, high = min(numbers), max(numbers)
    if digits == 0:
        return f"{int(low)}" if low == high else f"{int(low)}-{int(high)}"
    return f"{low:.{digits}f}" if abs(high - low) < 10**-digits else f"{low:.{digits}f}-{high:.{digits}f}"


def slow_rows(summary: dict) -> list:
    """Every ``Bound=SLOW`` op group in the committed optimized reports.

    ``tt-perf-report`` labels a row SLOW when it reaches neither the DRAM nor the FLOP roofline.  The
    label is not a diagnosis, so the work log accounts for each one; this block is what keeps that
    accounting from going stale as the reports are re-measured.
    """
    import csv as _csv

    lines = [
        "| pass | op | instances per pass | device time per pass | share | cores | DRAM % | FLOPs % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    found = 0
    for kind in KINDS:
        for phase in PHASES:
            entry = summary["measurements"].get(f"optimized/{kind}/{phase}")
            if not entry:
                continue
            replays = 8 if phase.startswith("decode") else 1
            with (DOC / entry["artifacts"]["report_csv"]).open() as handle:
                rows = list(_csv.DictReader(handle))
            total = sum(float(r["Device Time"] or 0) for r in rows) / replays
            grouped: dict = {}
            for row in rows:
                if (row.get("Bound") or "").strip().upper() != "SLOW":
                    continue
                key = row["OP Code"]
                item = grouped.setdefault(key, {"n": 0, "us": 0.0, "cores": set(), "dram": set(), "flops": set()})
                item["n"] += 1
                item["us"] += float(row["Device Time"] or 0)
                item["cores"].add(row.get("Cores", ""))
                item["dram"].add(row.get("DRAM %", ""))
                item["flops"].add(row.get("FLOPs %", ""))
            for key, item in sorted(grouped.items(), key=lambda kv: -kv[1]["us"]):
                found += 1
                per_pass = item["us"] / replays
                lines.append(
                    f"| `{kind}` {phase} | `{key}` | {max(1, item['n'] // replays)} | {per_pass:.1f} us | "
                    f"{100.0 * per_pass / total:.2f} % | {_span(item['cores'], 0)} | "
                    f"{_span(item['dram'], 1)} | {_span(item['flops'], 1)} |"
                )
    lines.append("")
    lines.append(
        f"{found} `Bound=SLOW` op groups across the six committed optimized reports."
        if found
        else "No `Bound=SLOW` op group appears in any of the six committed optimized reports."
    )
    return lines


def projection_packing(rows: list) -> list:
    """Packed versus separate shared-input projections at this stage's dtypes and layout."""
    selected = [r for r in rows if r.get("sweep") == "projection_packing"]
    if not selected:
        return ["_no rows: probe_projection_packing.log is not committed_"]
    lines = ["| pair | phase | rows | candidate | median | in0_block_w | blocker |", "|---|---|---|---|---|---|---|"]
    for row in selected:
        lines.append(
            f"| {row['pair']} | {row['phase']} | {row['rows']} | {row['candidate']} | "
            f"{_US(row['median_us']) if row.get('median_us') else '—'} | "
            f"{row.get('in0_block_w', '—')} | {row.get('error') or '—'} |"
        )
    return lines


def matmul_envelope(log: str = "probe_matmul_policy.log") -> list:
    """The model-free legal envelope: the best candidate per role and phase, plus the blockers."""
    path = DOC / "logs" / log
    if not path.exists():
        return ["_no rows: probe_matmul_policy.log is not committed_"]
    payload = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("PROBE_JSON "):
            payload = json.loads(line[len("PROBE_JSON ") :])
    if payload is None:
        return ["_no rows: probe_matmul_policy.log has no PROBE_JSON line_"]
    best: dict = {}
    baseline: dict = {}
    blockers: dict = {}
    for row in payload:
        key = (row["phase"], row["role"])
        if row.get("error"):
            blockers.setdefault(key, []).append(row)
            continue
        if row["median_us"] is None:
            continue
        if row["candidate"] == "interleaved bf16/HiFi4":
            baseline[key] = row
        if key not in best or row["median_us"] < best[key]["median_us"]:
            best[key] = row
    lines = [
        "| phase | role | K x N | fused-stage form | best measured candidate | cores | in0_block_w | "
        "median | PCC vs float32 | L1 blockers hit |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(best):
        phase, role = key
        row = best[key]
        base = baseline.get(key)
        lines.append(
            f"| {phase} | `{role}` | {row['k']} x {row['n']} | "
            f"{_US(base['median_us']) if base else '—'} | {row['candidate']} | "
            f"{row['cores'] or '—'} | {row['in0_block_w'] or '—'} | {_US(row['median_us'])} | "
            f"{_PCC(row['pcc'])} | {len(blockers.get(key, []))} |"
        )
    return lines


def correctness(evidence: dict) -> list:
    """Minimum PCC per recorded metric, per layer kind, out of ``pcc_evidence.json``."""
    records = evidence.get("records") or []
    if not records:
        return ["_no rows: pcc_evidence.json is not committed_"]
    grouped: dict = {}
    for record in records:
        metric = record.get("metric", "")
        kind = record.get("kind", "—")
        value = record.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        grouped.setdefault(metric, {}).setdefault(kind, []).append(float(value))
    lines = ["| measurement | `linear_attention` | `full_attention` |", "|---|---|---|"]
    for metric in sorted(grouped):
        row = grouped[metric]
        cells = []
        for kind in KINDS:
            values = row.get(kind)
            cells.append(f"min {min(values):.6f} ({len(values)})" if values else "—")
        if all(cell == "—" for cell in cells):
            merged = [v for values in row.values() for v in values]
            cells = [f"min {min(merged):.6f} ({len(merged)})", "—"]
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    scored = [
        float(r["value"])
        for r in records
        if isinstance(r.get("value"), (int, float))
        and not isinstance(r.get("value"), bool)
        and "_scale" not in r.get("metric", "")
    ]
    if scored:
        lines.append("")
        lines.append(f"Minimum over all {len(scored)} PCC records: **{min(scored):.6f}**, against a bar of 0.995.")
    return lines


# ------------------------------------------------------------------- rendering

BLOCKS = {}


def _register(name, builder):
    BLOCKS[name] = builder


def build_blocks() -> dict:
    summary = _summary()
    evidence = _evidence()
    optimized_rows = _probe_rows("probe_optimized_policy.log")
    optimized_rows += _probe_rows("probe_optimized_geometry.log")
    optimized_rows += _probe_rows("probe_optimized_prefill.log")
    optimized_rows += _probe_rows("probe_optimized_final.log")
    optimized_rows += _probe_rows("probe_optimized_isolation.log")
    optimized_rows += _probe_rows("probe_real_weight_policy.log")
    optimized_rows += _probe_rows("probe_projection_packing.log")
    optimized_rows += _probe_rows("probe_bfp4_gateup.log")
    optimized_rows += _probe_rows("probe_recurrence_advice.log")
    optimized_rows += _probe_rows("probe_norm_repeat_order.log")
    optimized_rows += _probe_rows("probe_stream_grid.log")
    return {
        "before_after": before_after(summary),
        "breakdown": breakdown(summary),
        "isolation": isolation(optimized_rows),
        "dominant_matmuls": dominant_matmuls(summary),
        "accounting": accounting(summary),
        "policy_sweep": policy_sweep(optimized_rows),
        "geometry_sweep": geometry_sweep(optimized_rows),
        "in0_block_w_sweep": in0_block_w_sweep(optimized_rows),
        "prefill_sweep": prefill_sweep(optimized_rows),
        "matmul_envelope": matmul_envelope(),
        "real_weight_policy": real_weight_policy(optimized_rows),
        "projection_packing": projection_packing(optimized_rows),
        "slow_rows": slow_rows(summary),
        "bfp4_gateup": bfp4_gateup(optimized_rows),
        "long_context_fp32acc": long_context_fp32acc(_probe_rows("probe_long_context_fp32acc.log")),
        "long_context_linear": long_context_linear(_probe_rows("probe_long_context_linear.log")),
        "long_context_real_full": long_context_real_full(_probe_rows("probe_long_context_precision_real.log")),
        "fidelity_gain": fidelity_gain(_probe_rows("probe_fidelity_gain.log")),
        "scale_vs_length": scale_vs_length(_probe_rows("probe_scale_vs_length.log")),
        "prefill_fidelity_roles": prefill_fidelity_roles(_probe_rows("probe_prefill_fidelity_roles.log")),
        "prefill_fidelity_cost": prefill_fidelity_cost(_probe_rows("probe_prefill_fidelity_cost.log")),
        "prefill_fidelity_mixed": prefill_fidelity_mixed(_probe_rows("probe_prefill_fidelity_mixed.log")),
        "long_context_real_linear": long_context_real_linear(_probe_rows("probe_long_context_linear_real.log")),
        "prefill_grid_alignment": prefill_grid_alignment(_probe_rows("probe_prefill_grid_alignment.log")),
        "norm_repeat_order": norm_repeat_order(optimized_rows),
        "stream_grid": stream_grid(optimized_rows),
        "recurrence_advice": recurrence_advice(optimized_rows),
        "correctness": correctness(evidence),
    }


def render(text: str, blocks: dict) -> str:
    """Replace every ``<!-- GENERATED:name -->`` ... ``<!-- END GENERATED:name -->`` span."""
    for name, lines in blocks.items():
        pattern = re.compile(
            rf"(<!-- GENERATED:{re.escape(name)} -->\n).*?(<!-- END GENERATED:{re.escape(name)} -->)",
            re.DOTALL,
        )
        replacement = "\\1" + "\n".join(lines) + "\n\\2"
        text = pattern.sub(lambda match, r=replacement: match.expand(r), text)
    return text


def main() -> None:
    blocks = build_blocks()
    for name in ("README.md", "work_log.md"):
        path = DOC / name
        if not path.exists():
            continue
        original = path.read_text()
        updated = render(original, blocks)
        path.write_text(updated)
        print(f"{'updated' if updated != original else 'unchanged'} {path}")
    missing = [name for name in blocks if not blocks[name]]
    if missing:
        print(f"WARNING: empty blocks: {missing}")


if __name__ == "__main__":
    main()

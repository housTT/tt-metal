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
    for arm in arms:
        cells = []
        for kind in kinds:
            match = next((r for r in selected if r["kind"] == kind and r["candidate"] == arm), None)
            cells.append(_MS(match["decode_ms"]) if match and match.get("decode_ms") else "—")
            cells.append(_MS3(match["prefill_ms"]) if match and match.get("prefill_ms") else "—")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return lines


def dominant_matmuls(summary: dict) -> list:
    """Per pass, the largest matmul rows with the dtype and fidelity the profiler measured.

    This is the OPT-013 artifact: the shipped policy is only implemented if these rows say so.
    """
    lines = [
        "| pass | op | instances | device time | math fidelity (measured) | bound | cores | DRAM % | FLOPs % |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for kind in KINDS:
        for phase in PHASES:
            entry = summary["measurements"].get(f"optimized/{kind}/{phase}")
            if not entry:
                continue
            for row in entry["dominant_matmul_rows"][:6]:
                lines.append(
                    f"| `{kind}` {phase} | `{row['op']}` | {row['instances']} | "
                    f"{row['device_time_us']:.1f} us | `{row['math_fidelity']}` | {row['bound'] or '—'} | "
                    f"{row['cores']} | {row['dram_pct'] or '—'} | {row['flops_pct'] or '—'} |"
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
        ),
    )


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
                    f"| `{kind}` {phase} | `{key}` | {item['n'] // replays} | {per_pass:.1f} us | "
                    f"{100.0 * per_pass / total:.2f} % | {'/'.join(sorted(item['cores']))} | "
                    f"{'/'.join(sorted(item['dram']))} | {'/'.join(sorted(item['flops']))} |"
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

"""Summarize one or two layer_pcc_probe.py reports (host only).

Prints, per report: the teacher-forced delta PCC of the GDN and attention layers (mean of the
per-layer means, worst layer, worst row), the five worst layers, the first-block and last-block
means, and the accumulated residual PCC (min / mean / rows below 0.99) at a fixed set of layers
plus the final norm. With --before, the same numbers of a pre-fix report are printed next to them.

The last layer (63 of 64) is excluded from the teacher-forced rows and the exclusion is printed:
HF hidden_states[-1] is the post-final-norm tensor (it equals last_hidden_state), so the probe's
layer 63 row compares a pre-norm TT residual with a normed reference and its delta PCC (about 0.44
to 0.51) is a bookkeeping artifact, not a device fault. The final norm row is the valid layer 63
comparison.

Usage:
  python layer_probe_summary.py --after REPORT.json [--before REPORT.json] [--out OUT.md]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CURVE_LAYERS = (0, 3, 7, 15, 23, 31, 39, 47, 55, 62)


def excluded_layer_note(report):
    last = report["n_layers"] - 1
    row = next((r for r in report["layers"] if r["layer"] == last), None)
    if row is None or "teacher_delta_mean_pcc" not in row:
        return f"layer {last} excluded from the teacher-forced rows (no teacher-forced row in the report)"
    return (
        f"layer {last} excluded from the teacher-forced rows: HF hidden_states[-1] is the post-final-norm tensor, so the probe "
        f"compares a pre-norm TT residual with a normed reference there (delta PCC {row['teacher_delta_mean_pcc']}, tt norm "
        f"{row['tt_norm_mean']} vs ref norm {row['ref_norm_mean']}); the final norm row is the valid layer {last} comparison"
    )


def teacher_rows(report):
    rows = [r for r in report["layers"] if "teacher_delta_mean_pcc" in r and r["layer"] < report["n_layers"] - 1]
    out = {}
    for kind in ("gdn", "attn"):
        k = [r for r in rows if r["kind"] == kind]
        if not k:
            continue
        worst = min(k, key=lambda r: r["teacher_delta_mean_pcc"])
        blocks = list(k[0].get("teacher_delta_block_mean", {}).keys())
        out[kind] = dict(
            layers=len(k),
            mean_of_means=round(statistics.mean(r["teacher_delta_mean_pcc"] for r in k), 6),
            worst_layer=worst["layer"],
            worst_layer_mean=worst["teacher_delta_mean_pcc"],
            worst_row=round(min(r["teacher_delta_min_pcc"] for r in k), 6),
            worst_row_layer=min(k, key=lambda r: r["teacher_delta_min_pcc"])["layer"],
            five_worst=[
                (r["layer"], r["teacher_delta_mean_pcc"])
                for r in sorted(k, key=lambda r: r["teacher_delta_mean_pcc"])[:5]
            ],
            block_means={b: round(statistics.mean(r["teacher_delta_block_mean"][b] for r in k), 6) for b in blocks}
            or "n/a",
        )
    return out


def curve(report):
    by = {r["layer"]: r for r in report["layers"]}
    rows = {}
    for L in CURVE_LAYERS:
        if L in by:
            r = by[L]
            rows[L] = (r["kind"], r["acc_min_pcc"], r["acc_mean_pcc"], r["acc_rows_below_0_99"])
    f = report["final_norm"]
    rows["final"] = ("norm", round(f["min_pcc"], 6), round(f["mean_pcc"], 6), None)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--after", required=True)
    parser.add_argument("--before", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    after = json.loads(Path(args.after).read_text())
    before = json.loads(Path(args.before).read_text()) if args.before else None
    lines = [
        f"T={after['T']}, {after['n_layers']} layers. after: {args.after}"
        + (f"; before: {args.before}" if before else "")
    ]
    ta = teacher_rows(after)
    tb = teacher_rows(before) if before else {}
    lines.append(excluded_layer_note(after))
    lines.append("")
    lines.append("| teacher-forced delta PCC | before (bf16 gate) | after (fp32 gate) |")
    lines.append("|---|---|---|")
    for kind in ("gdn", "attn"):
        if kind not in ta:
            continue
        a = ta[kind]
        b = tb.get(kind, {})
        for key in ("mean_of_means", "worst_layer_mean", "worst_row", "five_worst", "block_means"):
            lines.append(f"| {kind} {key} | {b.get(key, 'n/a')} | {a[key]} |")
        lines.append(
            f"| {kind} worst layer / worst-row layer | {b.get('worst_layer', 'n/a')} / {b.get('worst_row_layer', 'n/a')} | {a['worst_layer']} / {a['worst_row_layer']} |"
        )
    ca = curve(after)
    cb = curve(before) if before else {}
    lines.append("")
    lines.append("| layer | kind | accumulated min / mean / rows<0.99, before | after |")
    lines.append("|---|---|---|---|")
    for L, (kind, mn, mean, below) in ca.items():
        b = cb.get(L)
        bs = f"{b[1]} / {b[2]} / {b[3]}" if b else "n/a"
        lines.append(f"| {L} | {kind} | {bs} | {mn} / {mean} / {below} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()

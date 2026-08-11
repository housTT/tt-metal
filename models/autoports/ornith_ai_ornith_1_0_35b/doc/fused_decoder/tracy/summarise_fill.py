# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Attribute the elementwise op-code aggregates in each window to what they actually are.

`tt-perf-report` groups by op code, so the third- and fourth-largest items in the traced decode
window arrive as undifferentiated `UnaryDeviceOperation` and `BinaryNgDeviceOperation` totals.
Treating those as unattributable understates what is known: the raw capture carries an `ATTRIBUTES`
column and per-tensor shape columns, and most of both totals is one identifiable MoE cost.

**Window identification.** The raw capture contains the signpost rows themselves — a row whose op
code is `PERF_PREFILL` / `PERF_DECODE` and a matching `..._END` — so the window is exactly the rows
strictly between them. An earlier version of this script took the trailing `N` rows instead and
checked the total against the profiler's windowed report with a per-row tolerance; review round 9
showed that was off by one row in all four windows and that the tolerance was far too loose to
notice (it would also have accepted a shift of a whole decode replay). The signpost method is exact,
and the result is still cross-checked against the report's row count and total device time with a
tolerance of half a microsecond per row, which is what the report's own rounding can produce.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/tracy/summarise_fill.py

Run by ``logs/run_evidence.sh`` into ``tracy/fill_summary.txt``. Both this stage's captures and the
functional stage's are summarised, because README §5.4 claims the cost is not a regression and that
claim needs the baseline in the same artifact.
"""

import csv
import gzip
import re
import sys
from pathlib import Path

T = Path(__file__).resolve().parent
FUNCTIONAL = T.parent.parent / "functional_decoder/tracy"

#: (kind, phase, repeats inside the signposted window).
WINDOWS = [
    ("linear_attention", "prefill", 1),
    ("linear_attention", "decode", 32),
    ("full_attention", "prefill", 1),
    ("full_attention", "decode", 32),
]

#: The aggregates being split, and the attribute substring that names each one's dominant member.
AGGREGATES = {
    "UnaryDeviceOperation": ("FILL", "UnaryOpType::FILL"),
    "BinaryNgDeviceOperation": ("MUL", "BinaryOpType::MUL"),
}

SIGNPOST = {"prefill": ("PERF_PREFILL", "PERF_PREFILL_END"), "decode": ("PERF_DECODE", "PERF_DECODE_END")}

SHAPE_COLS = ("OUTPUT_0_Z_PAD[LOGICAL]", "OUTPUT_0_Y_PAD[LOGICAL]", "OUTPUT_0_X_PAD[LOGICAL]")

#: Unary activations folded into a binary op's operands. Two ops with the same reported output shape
#: can be entirely different work — review round 10 found the routed and shared SwiGLUs sharing a
#: reported `1x128x512` while differing 17x in cost — so members are reported **per launch within one
#: iteration**, with their folded activation, rather than bucketed by a shape column that is not the
#: real output shape for every op.
ACTIVATIONS = ("SILU", "SIGMOID", "SOFTPLUS")


def window_rows(root: Path, kind: str, phase: str):
    """The raw rows strictly between the two signposts, cross-checked against the windowed report."""
    with gzip.open(root / f"{kind}/{phase}_ops.csv.gz", "rt", newline="") as handle:
        raw = list(csv.DictReader(handle))
    start_name, end_name = SIGNPOST[phase]
    starts = [i for i, r in enumerate(raw) if r["OP CODE"].strip() == start_name]
    ends = [i for i, r in enumerate(raw) if r["OP CODE"].strip() == end_name]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise SystemExit(f"{root.name}/{kind}/{phase}: expected exactly one {start_name}/{end_name} pair")
    rows = raw[starts[0] + 1 : ends[0]]

    report = list(csv.DictReader((root / f"{kind}/{phase}_perf_report.csv").open(newline="")))
    if len(rows) != len(report):
        raise SystemExit(
            f"{root.name}/{kind}/{phase}: {len(rows)} rows between the signposts but the windowed "
            f"report has {len(report)} — the window is not what tt-perf-report summarised"
        )
    report_us = sum(
        float(re.match(r"^([\d.]+)", (r.get("Device Time") or "0").replace(",", "")).group(1)) for r in report
    )
    raw_us = sum(float(r["DEVICE KERNEL DURATION [ns]"] or 0) for r in rows) / 1000
    # The report prints each row to whole microseconds, so N rows can differ by up to N/2 us.
    if abs(report_us - raw_us) > 0.5 * len(report):
        raise SystemExit(
            f"{root.name}/{kind}/{phase}: signposted rows total {raw_us:.1f} us, report {report_us:.1f} us"
        )
    return rows, raw_us


def shape_of(row) -> str:
    dims = [row.get(c, "").strip() for c in SHAPE_COLS]
    return "x".join(d for d in dims if d) or "?"


def width_of(row) -> str:
    """The innermost reported dimension, i.e. the row width."""
    return (row.get("OUTPUT_0_X_PAD[LOGICAL]") or "?").strip()


def consumer_of(rows, index):
    """``(op code, width)`` of the next op after ``index``, or ``None``.

    This is what identifies a member, and the reason the reported shape alone cannot: review round 11
    found two `FILL` launches reporting an identical `…x1024` while costing 93 and 174 us, because the
    second zero-initialises the *down* projection's 2048-wide output and the shape column had not
    caught up. The op a fill immediately precedes is the op whose output it is clearing, so the
    consumer's width is the real one, and it is read here rather than trusted from the fill's own row.
    """
    for row in rows[index + 1 :]:
        code = row["OP CODE"].strip()
        if code and not code.startswith("PERF_"):
            return code, width_of(row)
    return None


def split(rows, op_code, marker, repeats):
    """``(total_us, marked_us, [(us, shape, activation), ...] for one iteration)``.

    The per-iteration member list is the launches of the *first* iteration, which for a traced decode
    replay is one full step. Reporting launches rather than shape buckets is deliberate: see
    ``ACTIVATIONS``.
    """
    total = marked_us = 0.0
    for row in rows:
        if row["OP CODE"].strip() != op_code:
            continue
        us = float(row["DEVICE KERNEL DURATION [ns]"] or 0) / 1000
        total += us
        if marker in (row.get("ATTRIBUTES") or ""):
            marked_us += us
    one = rows[: max(1, len(rows) // repeats)]
    members = []
    marked_count = 0
    for index, row in enumerate(one):
        if row["OP CODE"].strip() != op_code:
            continue
        attrs = row.get("ATTRIBUTES") or ""
        if marker in attrs:
            marked_count += 1
        folded = next((a for a in ACTIVATIONS if a in attrs), "-")
        consumer = consumer_of(one, index)
        members.append(
            (
                float(row["DEVICE KERNEL DURATION [ns]"] or 0) / 1000,
                shape_of(row),
                folded,
                f"{consumer[0]}/w={consumer[1]}" if consumer else "-",
            )
        )
    members.sort(reverse=True)
    return total, marked_us, members, marked_count


def main() -> int:
    out = [
        "Attribution of the elementwise op-code aggregates inside each signposted window, for this",
        "stage's captures and the functional stage's. Recovered from the raw *_ops.csv.gz ATTRIBUTES",
        "and OUTPUT_0 columns; the window is the rows strictly between the signpost rows, and",
        "'next' is the op each member immediately precedes, with that op's row width - which is what",
        "identifies the member, because the member's own reported shape can be stale (see the script).",
        "is cross-checked against the profiler's own windowed report (see this script's docstring).",
        "",
        "Per iteration of the window: prefill = one 2048-token pass, decode = one traced replay.",
    ]
    for label, root in (("fused", T), ("functional", FUNCTIONAL)):
        out.append("")
        out.append(f"=== {label} ===")
        for kind, phase, repeats in WINDOWS:
            rows, window_us = window_rows(root, kind, phase)
            out.append(f"{kind}/{phase}: window {window_us / repeats:.1f} us/iter")
            for op_code, (short, marker) in AGGREGATES.items():
                total, marked_us, members, marked_count = split(rows, op_code, marker, repeats)
                if not total:
                    continue
                out.append(
                    f"  {op_code:26s} total {total / repeats:9.1f} us/iter"
                    f"   {short} {marked_us / repeats:9.1f} us/iter ({100 * marked_us / total:5.1f} %)"
                    f"   {100 * marked_us / window_us:5.1f} % of window"
                    f"   {len(members)} launches/iter"
                    f"   of which {marked_count} are {short}"
                )
                for us, shape, folded, consumer in members[:4]:
                    out.append(f"      {us:9.3f} us  folded={folded:8s} next={consumer:40s} reported_out={shape}")
    body = "\n".join(out) + "\n"
    (T / "fill_summary.txt").write_text(body)
    print(body, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

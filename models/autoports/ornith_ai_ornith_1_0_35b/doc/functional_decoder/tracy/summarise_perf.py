# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduce the filtered ``tt-perf-report --csv`` outputs to one headline latency per window.

``tt-perf-report --csv`` writes one row per op inside the signposted window. Summing its device
time column gives the on-device kernel time for that window. Which column and unit was used is
printed here and recorded in ``PROVENANCE.md`` — some versions expose ``Device Time`` in
microseconds while raw Tracy ops CSVs expose ``DEVICE KERNEL DURATION [ns]``.

Usage::

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/functional_decoder/tracy/summarise_perf.py
"""

import csv
from pathlib import Path

ART = Path(__file__).resolve().parent

#: (kind, phase, repeats inside the signposted window)
WINDOWS = [
    ("linear_attention", "prefill", 1),
    ("linear_attention", "decode", 32),
    ("full_attention", "prefill", 1),
    ("full_attention", "decode", 32),
]

TIME_COLUMNS = [
    ("Device Time", 1e-6),  # microseconds -> seconds
    ("DEVICE KERNEL DURATION [ns]", 1e-9),
    ("Device Kernel Duration [ns]", 1e-9),
]


def summarise(path: Path, repeats: int):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    column = unit = None
    for name, scale in TIME_COLUMNS:
        if name in rows[0]:
            column, unit = name, scale
            break
    if column is None:
        raise KeyError(f"no known time column in {path}; have {sorted(rows[0])[:12]}")
    total = 0.0
    for row in rows:
        raw = (row.get(column) or "").replace(",", "").strip()
        if raw:
            try:
                total += float(raw)
            except ValueError:
                continue
    return {
        "ops": len(rows),
        "column": column,
        "total_s": total * unit,
        "per_iter_ms": total * unit / repeats * 1e3,
    }


def breakdown(path: Path, repeats: int, top: int = 4):
    """Per-op-code device time inside the window: ``[(op, share, ms_per_iter, launches_per_iter)]``."""
    agg: dict[str, float] = {}
    launches: dict[str, int] = {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    column = next((name for name, _ in TIME_COLUMNS if rows and name in rows[0]), None)
    scale = dict(TIME_COLUMNS)[column]
    for row in rows:
        raw = (row.get(column) or "").replace(",", "").strip()
        if not raw:
            continue
        code = (row.get("OP Code") or "?").strip()
        agg[code] = agg.get(code, 0.0) + float(raw)
        launches[code] = launches.get(code, 0) + 1
    total = sum(agg.values())
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])[:top]
    return [
        (code, 100 * value / total, value * scale / repeats * 1e3, launches[code] // repeats) for code, value in ranked
    ]


def elementwise(path: Path, repeats: int):
    """Combined ``Unary`` + ``BinaryNg`` device time — the many-small-launches group."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    column = next((name for name, _ in TIME_COLUMNS if rows and name in rows[0]), None)
    scale = dict(TIME_COLUMNS)[column]
    total = group = 0.0
    launches = 0
    for row in rows:
        raw = (row.get(column) or "").replace(",", "").strip()
        if not raw:
            continue
        value = float(raw)
        total += value
        code = (row.get("OP Code") or "").strip()
        if code.startswith(("UnaryDeviceOperation", "BinaryNgDeviceOperation")):
            group += value
            launches += 1
    return 100 * group / total, group * scale / repeats * 1e3, launches // repeats


def main():
    print(f"{'kind':<18}{'phase':<9}{'ops':>6}{'iters':>7}{'device ms/iter':>16}  time column")
    for kind, phase, repeats in WINDOWS:
        path = ART / kind / f"{phase}_perf_report.csv"
        if not path.is_file():
            print(f"{kind:<18}{phase:<9}  MISSING {path}")
            continue
        info = summarise(path, repeats)
        if info is None:
            print(f"{kind:<18}{phase:<9}  EMPTY (no rows in the signposted window)")
            continue
        print(f"{kind:<18}{phase:<9}{info['ops']:>6}{repeats:>7}{info['per_iter_ms']:>16.3f}  {info['column']}")

    print("\nWhere the device time goes (top op codes per window):")
    for kind, phase, repeats in WINDOWS:
        path = ART / kind / f"{phase}_perf_report.csv"
        if not path.is_file():
            continue
        print(f"  {kind}/{phase}:")
        for code, share, ms, launches in breakdown(path, repeats):
            print(f"    {share:5.1f}%  {ms:9.3f} ms/iter  {launches:4d} launches/iter  {code}")
        # The elementwise ops are many small launches; report them as one group too, which is how
        # the README quotes them.
        share, ms, launches = elementwise(path, repeats)
        print(
            f"    {share:5.1f}%  {ms:9.3f} ms/iter  {launches:4d} launches/iter  "
            f"[group] Unary + BinaryNg elementwise"
        )


if __name__ == "__main__":
    main()

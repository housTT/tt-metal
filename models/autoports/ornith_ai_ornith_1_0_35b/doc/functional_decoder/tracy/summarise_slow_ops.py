# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Extract the ``SLOW``-flagged rows from the ``tt-perf-report`` tables into one summary.

``tt-perf-report`` flags an op ``SLOW`` when neither its DRAM bandwidth nor its FLOP rate comes
close to the device roofline. Those rows are the concrete hand-off to the optimization stage: the
functional decoder deliberately does not sweep matmul program configs or math fidelity, so this
script records *which* geometries are leaving headroom on the table rather than leaving a reader to
re-derive it from four reports of 369 / 3712 / 335 / 3424 op rows.

Counts are reported per signposted window and, for decode, also per traced step (the window
replays the trace 32 times, so the raw count is 32x the per-step count).

Usage::

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/functional_decoder/tracy/summarise_slow_ops.py

Writes ``slow_ops_summary.txt`` next to this script and prints the same text.
"""

import re
from pathlib import Path

ART = Path(__file__).resolve().parent

#: (kind, phase, trace replays inside the signposted window)
WINDOWS = [
    ("linear_attention", "prefill", 1),
    ("linear_attention", "decode", 32),
    ("full_attention", "prefill", 1),
    ("full_attention", "decode", 32),
]

#: ``  850    0.3 %   SLOW  MatmulDeviceOperation 2048 x 2048 x 4096   3   947 us   0 us   110   44 GB/s   8.7 %  36.3 TFLOPs   23.9 %  HiFi4 ...``
#: The geometry field also takes the batched form ``b={32} x 32 x 128 x 128``.
ROW = re.compile(
    r"^\s*\d+\s+(?P<share>[\d.]+)\s*%\s+SLOW\s+(?P<op>\S+)\s+(?P<geom>(?:b=\{\d+\}\s+x\s+)?[\d x]+?)\s+\d+\s+"
    r"(?P<total>[\d,]+)\s*\S?s\s+[\d,]+\s*\S?s\s+(?P<cores>\d+)\s+"
    r"(?P<bw>[\d,]+)\s*GB/s\s+(?P<bw_pct>[\d.]+)\s*%\s+(?P<flops>[\d.]+)\s*TFLOPs\s+(?P<flop_pct>[\d.]+)\s*%\s+(?P<fid>.*\S)"
)


def rows_for(path: Path):
    out, unparsed = [], []
    for line in path.read_text(errors="replace").splitlines():
        if " SLOW " not in line:
            continue
        m = ROW.match(line)
        if m:
            d = m.groupdict()
            d["total_us"] = int(d["total"].replace(",", ""))
            out.append(d)
        else:
            unparsed.append(line)
    return out, unparsed


def aggregate(rows):
    """One entry per (op, geometry, fidelity, **core count**), summed over its launches.

    The core count is part of the key on purpose. The same geometry can be launched with two very
    different program configs in one window — the DeltaNet decode ``b={32} x 32 x 128 x 128`` matmul
    runs once on 110 cores at 12 us and once on 4 cores at 60 us — and merging them would report one
    config's utilization for both, which points the optimization stage at the wrong problem. The
    utilization is still a mean over a group, so the range is reported whenever the group is not
    uniform.
    """
    agg = {}
    for r in rows:
        key = (r["op"], " ".join(r["geom"].split()), r["fid"], r["cores"])
        slot = agg.setdefault(key, {"us": 0, "n": 0, "bw": [], "flop": []})
        slot["us"] += r["total_us"]
        slot["n"] += 1
        slot["bw"].append(float(r["bw_pct"]))
        slot["flop"].append(float(r["flop_pct"]))
    return sorted(agg.items(), key=lambda kv: -kv[1]["us"])


def span(values):
    """``"42.4"`` for a uniform group, ``"10.2-10.4"`` when the group varies."""
    lo, hi = min(values), max(values)
    return f"{lo:.1f}" if round(lo, 1) == round(hi, 1) else f"{lo:.1f}-{hi:.1f}"


def window_total_us(kind: str, phase: str) -> float:
    """Device time of the whole signposted window, in microseconds.

    Read from the same filtered CSV ``summarise_perf.py`` reduces, so a group's share of its window
    is a generated figure rather than something a reader has to divide by hand — and so a prefill
    row's sub-1 % share is never generalised to a decode window, where the same rows are several
    percent each.
    """
    import csv

    path = ART / kind / f"{phase}_perf_report.csv"
    if not path.is_file():
        return 0.0
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0.0
    column = next((c for c in ("Device Time", "DEVICE KERNEL DURATION [ns]") if c in rows[0]), None)
    scale = 1.0 if column == "Device Time" else 1e-3  # report microseconds either way
    total = 0.0
    for row in rows:
        raw = (row.get(column) or "").replace(",", "").strip()
        if raw:
            total += float(raw) * scale
    return total


def main():
    lines = []
    for kind, phase, repeats in WINDOWS:
        path = ART / kind / f"{phase}_perf_report.txt"
        if not path.is_file():
            lines.append(f"{kind}/{phase}: MISSING {path.name}")
            continue
        raw = sum(1 for ln in path.read_text(errors="replace").splitlines() if " SLOW " in ln)
        parsed, unparsed = rows_for(path)
        window = window_total_us(kind, phase)
        slow_us = sum(r["total_us"] for r in parsed)
        per_step = f", {raw // repeats} per traced step" if repeats > 1 else ""
        share = f", {100 * slow_us / window:.1f} % of the window's {window:.0f} us" if window else ""
        lines.append(f"{kind}/{phase}: {raw} SLOW-flagged rows in the window{per_step}{share}")
        # Every group, not a top-N: a truncated list would make the un-listed groups look absent,
        # and README §7 item 5 quotes the 110-core sibling of a listed row to make its point.
        for (op, geom, fid, cores), v in aggregate(parsed):
            pct = f"{100 * v['us'] / window:5.2f} %" if window else "     ? %"
            lines.append(
                f"    {v['us']:>7} us ({pct} of window) over {v['n']:>3} launches  {cores:>3} cores  "
                f"{span(v['bw']):>9} % DRAM  {span(v['flop']):>9} % FLOPs  {op} {geom}  {fid}"
            )
        if unparsed:
            lines.append(f"    ({len(unparsed)} row(s) not parsed for detail; the count above is the grep count)")
    text = "\n".join(lines) + "\n"
    (ART / "slow_ops_summary.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

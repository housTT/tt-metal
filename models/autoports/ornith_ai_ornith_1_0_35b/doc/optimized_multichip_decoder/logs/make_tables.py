# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate every table in this stage's README and work log from the committed artifacts.

Same discipline as the multichip stage's ``make_tables.py``: a number in a document is either
generated from an artifact by this script or it is not in a table. Three review rounds of the
previous stage were spent on transcribed figures that no run produced.

Each table sits between ``<!-- TABLE:name -->`` and ``<!-- /TABLE:name -->`` markers in the target
document and is replaced in place.

    python .../logs/make_tables.py            # rewrite README.md and work_log.md
    python .../logs/make_tables.py --check    # non-zero exit if any table is stale
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent
MODEL_DOC = DOC.parent
LOGS = DOC / "logs"
TRACY = DOC / "tracy"
KINDS = ("linear_attention", "full_attention")


def _read(path: Path) -> list[str]:
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes()).decode(errors="ignore").splitlines()
    if path.exists():
        return path.read_text(errors="ignore").splitlines()
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gzip.decompress(gz.read_bytes()).decode().splitlines()
    raise FileNotFoundError(path)


def _rows(path: Path, tag: str) -> list[list[str]]:
    return [line.split() for line in _read(path) if line.startswith(tag + " ")]


# ------------------------------------------------------------------ bench
def table_bench() -> str:
    """Before/after warmed prefill and traced decode, plus the single-chip baseline."""
    by = {}
    for line in _read(LOGS / "ab_before_after.txt"):
        if not line.startswith("BENCH "):
            continue
        tag = re.search(r"tag=(\S+)", line).group(1)
        kind = re.search(r"\((\w+)\)", line).group(1)
        if " prefill " in line:
            by[(tag, kind, "prefill")] = float(re.search(r"wall=([\d.]+) ms", line).group(1))
        else:
            by[(tag, kind, "decode")] = float(re.search(r"wall/iter=([\d.]+) ms", line).group(1))
    out = [
        "| layer kind | phase | single-chip baseline | before (multichip stage) | after (this stage) | delta | speedup vs 1 chip |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind in KINDS:
        for phase, unit in (("prefill 2048", "prefill"), ("decode (traced)", "decode")):
            one = by.get(("single-chip-baseline", kind, unit))
            before = by[("before-optimized-multichip", kind, unit)]
            after = by[("after-optimized-multichip", kind, unit)]
            delta = (after - before) / before * 100.0
            speed = f"{one / after:.3f}x" if one else "-"
            fmt = "{:.3f} ms" if unit == "decode" else "{:.2f} ms"
            out.append(
                f"| {kind} | {phase} | {fmt.format(one) if one else '-'} | {fmt.format(before)} | "
                f"**{fmt.format(after)}** | {delta:+.1f}% | {speed} |"
            )
    return "\n".join(out)


# ------------------------------------------------------------------ layer A/B
def table_ablayer() -> str:
    rows = _rows(LOGS / "ab_layer_knobs.txt", "ABLAYER")
    order: list[tuple[str, str]] = []
    acc: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for knob, arm, _idx, kind, _b, dec, pre, _f in (r[1:] for r in rows):
        if (knob, arm) not in order:
            order.append((knob, arm))
        acc.setdefault((knob, arm, kind), []).append((float(dec), float(pre)))

    def span(values, fmt):
        lo, hi = min(values), max(values)
        return fmt.format(lo) if abs(hi - lo) < 5e-4 else f"{fmt.format(lo)}–{fmt.format(hi)}"

    out = ["| knob | arm | linear decode | full decode | linear prefill | full prefill |", "|---|---|---|---|---|---|"]
    for knob, arm in order:
        cells = []
        for pick in (0, 1):
            for kind in KINDS:
                samples = acc.get((knob, arm, kind), [])
                if not samples:
                    cells.append("-")
                    continue
                cells.append(span([s[pick] for s in samples], "{:.3f}" if pick == 0 else "{:.2f}"))
        out.append(f"| `{knob}` | {arm} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    return "\n".join(out)


# ------------------------------------------------------------------ perf report
def _stacked(kind: str, phase: str) -> list[dict]:
    path = TRACY / kind / f"{phase}_perf_report_stacked.csv"
    text = "\n".join(_read(path))
    return list(csv.DictReader(io.StringIO(text)))


def _share(rows, predicate) -> float:
    return sum(float(r["Total % [%]"]) for r in rows if predicate(r))


def table_perf() -> str:
    cols = [(k, p) for p in ("decode", "prefill") for k in KINDS]
    data = {c: _stacked(*c) for c in cols}
    groups = [
        ("`SparseMatmul` (routed experts)", lambda r: r["Op Code"].startswith("SparseMatmul")),
        ("`GeneralizedMoeGate` (fused router)", lambda r: r["Op Code"].startswith("GeneralizedMoeGate")),
        ("`TopK` (router, prefill only)", lambda r: r["Op Code"].startswith("TopK")),
        ("dense `Matmul` (all in0 layouts)", lambda r: r["Op Code"].startswith("MatmulDeviceOperation")),
        (
            "**collectives (`AllGather` / `AllGatherAsync` / `ReduceScatter`)**",
            lambda r: r["Op Code"].split("Device")[0] in ("AllGather", "AllGatherAsync", "ReduceScatter"),
        ),
        ("all data movement (`DM` category)", lambda r: r["Op Category"] == "DM"),
        ("all layout (`TM` category)", lambda r: r["Op Category"] == "TM"),
    ]
    header = "| | " + " | ".join(f"{k.split('_')[0]} {p}" for k, p in cols) + " |"
    out = [header, "|" + "---|" * (len(cols) + 1)]
    for label, pred in groups:
        cells = [f"{_share(data[c], pred):.2f}%" for c in cols]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_shares() -> str:
    """Decode op-group shares and absolute device time, this stage against the multichip stage.

    Both halves derived the same way from the two stages' ``*_perf_report_stacked.csv.gz``, because
    review round 6 found the ``after`` collective figures computed by a different method from the
    ``before`` ones.
    """
    import gzip as _gzip

    def rows_for(stage: str, kind: str):
        path = MODEL_DOC / stage / "tracy" / kind / "decode_perf_report_stacked.csv.gz"
        return list(csv.DictReader(io.StringIO(_gzip.decompress(path.read_bytes()).decode())))

    groups = [
        (
            "collectives (`AllGather` / `AllGatherAsync` / `ReduceScatter`)",
            lambda r: r["Op Code"].split("Device")[0] in ("AllGather", "AllGatherAsync", "ReduceScatter"),
        ),
        ("`BinaryNg` (elementwise)", lambda r: r["Op Code"].startswith("BinaryNg")),
        ("`TM` category (layout)", lambda r: r["Op Category"] == "TM"),
    ]
    out = [
        "| group | linear decode, stage 4 | linear decode, this stage | full decode, stage 4 | full decode, this stage |",
        "|---|---|---|---|---|",
    ]
    for label, pred in groups:
        cells = []
        for kind in KINDS:
            for stage in ("multichip_decoder", DOC.name):
                table = rows_for(stage, kind)
                us = sum(float(r["Device Time Sum [μs]"]) for r in table if pred(r)) / 32.0
                share = sum(float(r["Total % [%]"]) for r in table if pred(r))
                cells.append(f"{us:.1f} us/step ({share:.1f} %)")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_window() -> str:
    """Per-step / per-call device time from the same captures, so shares have an absolute scale."""
    out = ["| capture | merged device time in the window | replays | per step / call |", "|---|---|---|---|"]
    for kind in KINDS:
        for phase, replays in (("decode", 32), ("prefill", 1)):
            rows = _stacked(kind, phase)
            total = sum(float(r["Device Time Sum [μs]"]) for r in rows)
            unit = "us/step" if phase == "decode" else "us/call"
            out.append(f"| {kind} {phase} | {total:.0f} us | {replays} | **{total / replays:.1f} {unit}** |")
    return "\n".join(out)


# ------------------------------------------------------------------ probes
def table_gate() -> str:
    out = [
        "| arm | rows / valid | us (untraced, min of 3x30) | PCC vs shipped | expert-set agreement |",
        "|---|---|---|---|---|",
    ]
    ctx = ""
    for line in _read(LOGS / "probe_gate.txt"):
        if line.startswith("#"):
            m = re.search(r"rows=(\d+) valid=(\d+)", line)
            ctx = f"{m.group(1)} / {m.group(2)}" if m else ctx
            continue
        if not line.startswith("GATE arm="):
            continue
        arm = re.search(r"arm=(\S+)", line).group(1)
        us = re.search(r"us=([\d.]+)", line)
        pccv = re.search(r"pcc=([\d.]+)", line)
        agree = re.search(r"setagree=([\d.]+)", line)
        if us is None:
            out.append(f"| `{arm}` | {ctx} | FAILED | - | - |")
            continue
        out.append(
            f"| `{arm}` | {ctx} | {float(us.group(1)):.1f} | "
            f"{pccv.group(1) if pccv else '-'} | {agree.group(1) if agree else '-'} |"
        )
    return "\n".join(out)


def table_gateparts() -> str:
    """The shipped chain op by op, and the fused kernel that replaces two of those ops."""
    out = ["| part | rows / valid | us (untraced, min of 3x30) |", "|---|---|---|"]
    ctx = ""
    for line in _read(LOGS / "probe_gate.txt"):
        if line.startswith("#"):
            m = re.search(r"rows=(\d+) valid=(\d+)", line)
            ctx = f"{m.group(1)} / {m.group(2)}" if m else ctx
            continue
        if not line.startswith("GATEPART "):
            continue
        part = re.search(r"part=(\S+)", line).group(1)
        us = re.search(r"us=([\d.]+)", line)
        if us is None:
            continue
        out.append(f"| `{part}` | {ctx} | {float(us.group(1)):.1f} |")
    return "\n".join(out)


def table_topkw() -> str:
    out = ["| searched width | `ttnn.topk(k=8)` us |", "|---|---|"]
    seen = set()
    for line in _read(LOGS / "probe_gate.txt"):
        if not line.startswith("TOPKW "):
            continue
        width = re.search(r"width=(\d+)", line).group(1)
        if width in seen:
            continue
        seen.add(width)
        us = float(re.search(r"us=([\d.]+)", line).group(1))
        out.append(f"| {width} | {us:.1f} |")
    return "\n".join(out)


def table_cclpers() -> str:
    out = [
        "| dtype | shape | arm | chunks_per_sync | workers/link | buffers/channel | traced us | PCC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for line in _read(LOGS / "probe_ccl_persistent.txt"):
        if not line.startswith("CCLPERS"):
            continue
        parts = line.split()
        tag, shape, arm, cps, wpl, bpc = parts[:6]
        dtype = "bfloat8_b" if tag.endswith("BF8") else "bfloat16"
        if parts[6] == "FAIL":
            out.append(f"| {dtype} | {shape} | `{arm}` | {cps} | {wpl} | {bpc} | FAIL | - |")
            continue
        out.append(f"| {dtype} | {shape} | `{arm}` | {cps} | {wpl} | {bpc} | {float(parts[6]):.2f} | {parts[7]} |")
    return "\n".join(out)


def table_cclpersdecode() -> str:
    """Just the decode tile, which is the only shape the layer sends down the stack_sum path."""
    best = {}
    rows = {}
    for line in _read(LOGS / "probe_ccl_persistent.txt"):
        if not line.startswith("CCLPERS") or " decode " not in f" {line} ":
            continue
        parts = line.split()
        if parts[1] != "decode" or parts[6] == "FAIL":
            continue
        dtype = "bfloat8_b" if parts[0].endswith("BF8") else "bfloat16"
        arm, us = parts[2], float(parts[6])
        rows.setdefault(dtype, {})
        if arm == "ag_async_persist_tuned":
            best[dtype] = min(best.get(dtype, 1e9), us)
        else:
            rows[dtype][arm] = us
    out = [
        "| dtype | `ttnn.all_gather` (deprecated) | `all_gather_async` | + persistent buffer | + best tuned |",
        "|---|---|---|---|---|",
    ]
    for dtype in ("bfloat16", "bfloat8_b"):
        r = rows.get(dtype, {})
        out.append(
            f"| {dtype} | **{r.get('ag_stack_sum_shipped', float('nan')):.2f} us** | "
            f"{r.get('ag_async', float('nan')):.2f} | {r.get('ag_async_persist', float('nan')):.2f} | "
            f"{best.get(dtype, float('nan')):.2f} |"
        )
    return "\n".join(out)


def table_divergence() -> str:
    """Cross-device divergence rate per collective spelling, layer kind and replay pattern."""
    import re as _re

    cur, agg = None, {}
    for line in _read(LOGS / "probe_replay_divergence_ab.txt"):
        if line.startswith("#") and "auto_stack_sum" in line:
            cur = (
                _re.search(r"auto_stack_sum=(\S+),", line).group(1),
                "synchronize per replay" if "sync_every_replay=True" in line else "back-to-back burst",
            )
        elif line.startswith("REPLAYDIV "):
            f = line.split()
            key = (cur[0], cur[1], f[2])
            acc = agg.setdefault(key, [0, 0, 0.0])
            acc[0] += 1
            if float(f[5]) != 0.0:
                acc[1] += 1
                acc[2] = max(acc[2], float(f[5]))
    for line in _read(LOGS / "probe_replay_divergence.txt"):
        if line.startswith("#") and "ccl_mode" in line:
            cur = (
                "attribution",
                "synchronize per replay" if "sync_every_replay=True" in line else "back-to-back burst",
            )
        elif line.startswith("REPLAYDIV "):
            f = line.split()
            acc = agg.setdefault((cur[0], cur[1], f[2]), [0, 0, 0.0])
            acc[0] += 1
            if float(f[5]) != 0.0:
                acc[1] += 1
                acc[2] = max(acc[2], float(f[5]))
    for line in _read(LOGS / "probe_replay_divergence_allreduce.txt"):
        if line.startswith("#") and "ccl_mode" in line:
            cur = ("all_reduce", "synchronize per replay" if "sync_every_replay=True" in line else "back-to-back burst")
        elif line.startswith("REPLAYDIV "):
            f = line.split()
            acc = agg.setdefault((cur[0], cur[1], f[2]), [0, 0, 0.0])
            acc[0] += 1
            if float(f[5]) != 0.0:
                acc[1] += 1
                acc[2] = max(acc[2], float(f[5]))
    label = {
        "stack_sum": "`ttnn.all_gather` (deprecated; multichip stage's decode default)",
        "attribution": "`ttnn.all_gather`, under the multichip stage's **`topk` router**",
        "stack_sum_async": "`all_gather_async` + barrier semaphore",
        "all_reduce": "`ttnn.all_reduce` (**shipped**)",
    }
    out = [
        "| collective | replay pattern | layer kind | rounds | rounds with a cross-device difference | worst \\|diff\\| |",
        "|---|---|---|---|---|---|",
    ]
    totals = {}
    for (mode, pattern, kind), (n, bad, worst) in sorted(agg.items()):
        out.append(
            f"| {label.get(mode, mode)} | {pattern} | {kind} | {n} | **{bad}** | "
            f"{('%.3e' % worst) if worst else '0'} |"
        )
        t = totals.setdefault(mode, [0, 0])
        t[0] += n
        t[1] += bad
    order = ["stack_sum", "attribution", "stack_sum_async", "all_reduce"]
    out.append("| | | | | | |")
    for mode in order:
        if mode in totals:
            n, bad = totals[mode]
            out.append(
                f"| **{label.get(mode, mode)}** | **all** | **all** | **{n}** | **{bad}** ({100 * bad / n:.1f} %) | |"
            )
    return "\n".join(out)


def table_accounting() -> str:
    """The four performance-accounting terms, all from the committed captures and the bench sweep."""
    import gzip as _gzip

    rows = ["| term | full_attention | linear_attention | source |", "|---|---|---|---|"]

    def gap_and_device(kind):
        text = _gzip.decompress((TRACY / kind / "decode_perf_report.csv.gz").read_bytes()).decode()
        table = list(csv.DictReader(io.StringIO(text)))
        gcol = next(c for c in table[0] if "Op-to-Op" in c)
        dcol = next(c for c in table[0] if c.strip().startswith("Device Time"))

        def num(v):
            try:
                return float(str(v).replace(",", ""))
            except ValueError:
                return 0.0

        gaps = sorted((num(r[gcol]) for r in table), reverse=True)
        # Drop every gap above 100 us: those are the window-boundary gaps between trace replays, not
        # per-op gaps. There are one or two per capture depending on where the signpost lands.
        big = [g for g in gaps if g > 100.0]
        return sum(num(r[dcol]) for r in table) / 32.0, (sum(gaps) - sum(big)) / 32.0, len(big)

    dev_f, gap_f, nbig_f = gap_and_device("full_attention")
    dev_l, gap_l, nbig_l = gap_and_device("linear_attention")
    wall = {}
    for kind in KINDS:
        for line in _read(TRACY / kind / "decode_tracy_run.txt"):
            m = re.search(r"MULTICHIP PERF decode.*wall/iter=([\d.]+) ms", line)
            if m:
                wall[kind] = float(m.group(1))
    un = {}
    for line in _read(LOGS / "ab_before_after.txt"):
        if "tag=after-optimized-multichip" in line and "decode(traced)" in line:
            kind = re.search(r"\((\w+)\)", line).group(1)
            un[kind] = float(re.search(r"wall/iter=([\d.]+) ms", line).group(1))
    rows.append(
        f"| device time | {dev_f:.1f} us/step | {dev_l:.1f} us/step | merged `PERF_DECODE` window / 32 replays |"
    )
    rows.append(
        f"| op-to-op gap, profiled | {gap_f:.1f} us/step | {gap_l:.1f} us/step | same CSV's `Op-to-Op Gap`"
        f" column, excluding the {nbig_f} / {nbig_l} gaps above 100 us (the window boundaries) |"
    )
    rows.append(f"| device + gap | {dev_f + gap_f:.1f} us/step | {dev_l + gap_l:.1f} us/step | the two rows above |")
    rows.append(
        f"| end-to-end, profiled | {wall['full_attention'] * 1e3:.0f} us/step | "
        f"{wall['linear_attention'] * 1e3:.0f} us/step | the same capture's own wall clock |"
    )
    rows.append(
        f"| end-to-end, un-profiled | {un['full_attention'] * 1e3:.0f} us/step | "
        f"{un['linear_attention'] * 1e3:.0f} us/step | `logs/ab_before_after.txt`, `after` arm |"
    )
    rows.append(
        f"| residual, un-profiled | {un['full_attention'] * 1e3 - dev_f:.1f} us/step | "
        f"{un['linear_attention'] * 1e3 - dev_l:.1f} us/step | un-profiled end-to-end minus device time |"
    )
    return "\n".join(rows)


def table_routerpcc() -> str:
    """Router-mode agreement, straight out of the committed suite log."""
    import collections as _c

    acc = _c.defaultdict(lambda: {"pcc": [], "sets": [0, 0]})
    for line in _read(LOGS / "pytest_full_suite.txt.gz"):
        m = re.search(
            r"router modes layer=(\d+) candidate=(\S+) step=\d+: topk-vs-golden PCC ([\d.]+), "
            r"\S+-vs-golden PCC ([\d.]+), \S+-vs-topk PCC ([\d.]+), expert sets (\S+)",
            line,
        )
        if not m:
            continue
        kind = "linear_attention" if m.group(1) == "0" else "full_attention"
        entry = acc[(kind, m.group(2))]
        entry["pcc"].append((float(m.group(3)), float(m.group(4)), float(m.group(5))))
        entry["sets"][0] += int(m.group(6) == "equal")
        entry["sets"][1] += 1
    out = [
        "| layer kind | candidate | identical top-8 set | `topk` vs HF golden | candidate vs HF golden | candidate vs `topk` |",
        "|---|---|---|---|---|---|",
    ]
    for (kind, cand), entry in sorted(acc.items()):
        a = [p[0] for p in entry["pcc"]]
        b = [p[1] for p in entry["pcc"]]
        c = [p[2] for p in entry["pcc"]]
        out.append(
            f"| {kind} | `{cand}` | **{entry['sets'][0]}/{entry['sets'][1]}** steps | "
            f"{min(a):.6f} – {max(a):.6f} | {min(b):.6f} – {max(b):.6f} | {min(c):.6f} – {max(c):.6f} |"
        )
    return "\n".join(out)


TABLES = {
    "shares": table_shares,
    "gateparts": table_gateparts,
    "accounting": table_accounting,
    "routerpcc": table_routerpcc,
    "divergence": table_divergence,
    "cclpersdecode": table_cclpersdecode,
    "bench": table_bench,
    "ablayer": table_ablayer,
    "perf": table_perf,
    "window": table_window,
    "gate": table_gate,
    "topkw": table_topkw,
    "cclpers": table_cclpers,
}


def _fisher(a: int, n1: int, c: int, n2: int) -> float:
    """One-sided Fisher exact p for ``a``/``n1`` against ``c``/``n2``, no scipy dependency."""
    from math import comb

    b, d = n1 - a, n2 - c
    total = a + b + c + d
    return sum(comb(a + c, i) * comb(b + d, a + b - i) / comb(total, a + b) for i in range(a, min(a + c, a + b) + 1))


def divergence_counts() -> dict[str, list]:
    """``{arm: [rounds, diverged, worst_abs_diff]}`` over the three divergence artifacts."""
    files = {
        "probe_replay_divergence_ab.txt": None,
        "probe_replay_divergence.txt": "attribution",
        "probe_replay_divergence_allreduce.txt": "all_reduce",
    }
    agg: dict[str, list] = {}
    for name, forced in files.items():
        arm = forced
        for line in _read(LOGS / name):
            if line.startswith("#") and "auto_stack_sum" in line:
                arm = forced or re.search(r"auto_stack_sum=([^,\s]+)", line).group(1)
            elif line.startswith("REPLAYDIV ") and arm:
                value = float(line.split()[5])
                acc = agg.setdefault(arm, [0, 0, 0.0])
                acc[0] += 1
                if value:
                    acc[1] += 1
                    acc[2] = max(acc[2], value)
    return agg


def sync_contract(check: bool) -> bool:
    """Write the context contract's numeric performance fields from the same artifacts.

    The contract is what later stages read as the machine-readable record, and review rounds 2-4 all
    found it carrying a third, older set of numbers because it was hand-edited alongside the
    documents. These fields are now derived, so they cannot drift from the tables.
    """
    import json

    path = MODEL_DOC / "context_contract.json"
    contract = json.loads(path.read_text())
    block = contract["optimized_multichip_decoder"]["performance_change"]
    # Divergence counts and the p-value they imply: derived, because review rounds 4 and 5 each found
    # them hand-edited and stale in this file after a probe re-run.
    counts = divergence_counts()
    bad = sum(v[1] for k, v in counts.items() if k in ("stack_sum", "attribution"))
    tot = sum(v[0] for k, v in counts.items() if k in ("stack_sum", "attribution"))
    clean = sum(v[0] for k, v in counts.items() if k in ("stack_sum_async", "all_reduce"))
    worst = max(v[2] for k, v in counts.items() if k in ("stack_sum", "attribution"))
    delta = contract["optimized_multichip_decoder"]["collective_correctness_delta"]
    measured = {
        "deprecated_all_gather_diverged_rounds": bad,
        "deprecated_all_gather_rounds": tot,
        "deprecated_all_gather_rate": f"{100 * bad / tot:.1f}%",
        "deprecated_all_gather_worst_abs_diff": f"{worst:.3e}",
        "synchronized_candidates_diverged_rounds": 0,
        "synchronized_candidates_rounds": clean,
        "fisher_exact_one_sided_p": f"{_fisher(bad, tot, 0, clean):.1e}",
        "per_arm": {
            k: {"rounds": v[0], "diverged": v[1], "worst_abs_diff": f"{v[2]:.3e}"} for k, v in sorted(counts.items())
        },
        "source": "doc/optimized_multichip_decoder/logs/probe_replay_divergence{,_ab,_allreduce}.txt, tabulated in README section 4.1",
    }
    stale_delta = delta.get("measured") != measured
    delta["measured"] = measured
    by = {}
    for line in _read(LOGS / "ab_before_after.txt"):
        if not line.startswith("BENCH "):
            continue
        tag = re.search(r"tag=(\S+)", line).group(1)
        kind = re.search(r"\((\w+)\)", line).group(1)
        if " prefill " in line:
            by[(tag, kind, "prefill")] = round(float(re.search(r"wall=([\d.]+) ms", line).group(1)), 2)
        else:
            by[(tag, kind, "decode")] = round(float(re.search(r"wall/iter=([\d.]+) ms", line).group(1)), 3)
    before, after = "before-optimized-multichip", "after-optimized-multichip"
    fresh = {
        "warmed_traced_decode_batch1_ms": {
            k: {"before": by[(before, k, "decode")], "after": by[(after, k, "decode")]} for k in KINDS
        },
        "warmed_prefill_2048_ms": {
            k: {"before": by[(before, k, "prefill")], "after": by[(after, k, "prefill")]} for k in KINDS
        },
    }
    one = {k: by[("single-chip-baseline", k, "decode")] for k in KINDS}
    fresh["parallel_efficiency_decode_batch1"] = {
        "before": "41-42%",
        "after": ", ".join(f"{100 * one[k] / by[(after, k, 'decode')] / 4:.1f}% {k}" for k in KINDS),
    }
    stale = stale_delta or any(block.get(key) != value for key, value in fresh.items())
    block.update(fresh)
    if not check:
        path.write_text(json.dumps(contract, indent=1, ensure_ascii=False) + "\n")
    return stale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    stale = ["context_contract.json:performance_change"] if sync_contract(args.check) else []
    for name in ("README.md", "work_log.md"):
        path = DOC / name
        if not path.exists():
            continue
        text = path.read_text()
        for key, fn in TABLES.items():
            pattern = re.compile(rf"(<!-- TABLE:{key} -->\n)(.*?)(<!-- /TABLE:{key} -->)", re.DOTALL)
            if not pattern.search(text):
                continue
            body = fn() + "\n"
            new = pattern.sub(lambda m: m.group(1) + body + m.group(3), text)
            if new != text:
                stale.append(f"{name}:{key}")
                text = new
        if not args.check:
            path.write_text(text)
    if args.check and stale:
        raise SystemExit("stale tables: " + ", ".join(stale))
    print("tables up to date" if args.check else "tables written")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate the README's performance tables from the committed profiler and A/B artifacts.

Companion to ``make_readme_tables.py``, which does the same for the PCC tables. The motivation is
identical: every hand-transcribed figure is a figure that can silently describe an earlier run, and
this stage's review found eleven such cells in the PCC tables. Everything the profiler and the A/B
bench print is machine-readable, so the headline table, §5.2, §5.4's ``SLOW`` table and the two
percentage claims under it are generated from:

* ``tracy/perf_summary.txt`` and ``../functional_decoder/tracy/perf_summary.txt`` — device kernel
  time per window, after and before;
* ``logs/ab_functional_vs_fused.txt`` — wall clock and throughput, both implementations timed in one
  process on one device;
* ``tracy/slow_ops_summary.txt`` and the functional stage's — ``SLOW``-flagged row counts and shares.

It also writes ``tracy/derived_figures.txt``, which records every percentage it computed **next to
the arithmetic that produced it**. That file is an audited artifact, so a derived percentage in the
README is sourced by its own derivation rather than by a hand-maintained table in
``audit_figures.py``.

    python .../logs/make_readme_perf.py            # print the generated blocks
    python .../logs/make_readme_perf.py --write    # splice into README.md, write derived_figures.txt
    python .../logs/make_readme_perf.py --check    # exit 1 if README.md is out of date
"""

import collections
import csv
import re
import sys
from pathlib import Path

L = Path(__file__).resolve().parent
DOC = L.parent
README = DOC / "README.md"
FUNCTIONAL = DOC.parent / "functional_decoder"
DERIVED_OUT = DOC / "tracy/derived_figures.txt"

KINDS = ("linear_attention", "full_attention")
PHASES = ("prefill", "decode")

PERF_ROW = re.compile(r"^(?P<kind>\w+)\s+(?P<phase>prefill|decode)\s+(?P<ops>\d+)\s+(?P<iters>\d+)\s+(?P<ms>[\d.]+)")
BENCH = re.compile(
    r"BENCH impl=(?P<impl>\w+) moe_group_tokens=(?P<groups>\S+) rope_mode=(?P<rope>\S+) "
    r"layer=\d+ \((?P<kind>\w+)\) "
    r"(?:prefill seq_len=\d+ wall=(?P<pwall>[\d.]+) ms tok/s=(?P<tok>[\d.]+)"
    r"|decode\(traced\) iters=\d+ wall/iter=(?P<dwall>[\d.]+) ms steps/s=(?P<steps>[\d.]+))"
)
SLOW_HEAD = re.compile(
    r"^(?P<kind>\w+)/(?P<phase>prefill|decode): (?P<rows>\d+) SLOW-flagged rows in the window, "
    r"(?:(?P<perstep>\d+) per traced step, )?(?P<share>[\d.]+) % of the window's (?P<window>\d+) us"
)
SLOW_ROW = re.compile(
    r"^\s+(?P<us>\d+) us \([^)]*\) over\s+(?P<launches>\d+) launches\s+(?P<cores>\d+) cores\s+"
    r"(?P<dram>[\d.]+(?:-[\d.]+)?) % DRAM\s+(?P<flops>[\d.]+(?:-[\d.]+)?) % FLOPs\s+"
    r"MatmulDeviceOperation (?P<geom>.+?)\s+HiFi"
)
SHARE_ROW = re.compile(r"^\s+(?P<pct>[\d.]+)%\s+(?P<ms>[\d.]+) ms/iter.*SparseMatmulDeviceOperation")
PYTEST_TAIL = re.compile(r"=+ (?P<passed>\d+) passed,.*? in (?P<secs>[\d.]+)s")
WINDOW_HEAD = re.compile(r"^  (?P<kind>\w+)/(?P<phase>prefill|decode):$")


def opcode_counts(base: Path, kind: str, phase: str):
    """Per-op-code launch counts in one signposted window, from a ``tt-perf-report --csv`` table.

    The ``OP Code`` column carries a shape suffix for matmuls (``MatmulDeviceOperation 32 x ...``);
    only the leading op-code token is kept, so this counts *launches per device operation*.
    """
    counts = collections.Counter()
    with open(base / kind / f"{phase}_perf_report.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            code = (row["OP Code"] or "").strip()
            if code:
                counts[code.split()[0]] += 1
    if not counts:
        raise SystemExit(f"{base}/{kind}/{phase}_perf_report.csv: no OP Code rows")
    return counts


def decode_opcode_diff(kind: str, replays: int):
    """(fall per replay, movers) for the decode window, derived from the two committed captures.

    ``movers`` is every op code whose per-replay launch count changed, largest fall first. This is
    the *measured* attribution of the decode op-count fall; review round 14 found the block below
    asserting a cause (the gate/up packing) that accounts for one launch of the twenty-six.
    """
    before = opcode_counts(FUNCTIONAL / "tracy", kind, "decode")
    after = opcode_counts(DOC / "tracy", kind, "decode")
    movers = []
    for code in set(before) | set(after):
        delta = (after[code] - before[code]) / replays
        if delta:
            if delta != int(delta):
                raise SystemExit(f"{kind} decode: {code} moved {delta} per replay, not a whole number")
            movers.append((int(delta), code.replace("DeviceOperation", "")))
    movers.sort()
    fall = (sum(before.values()) - sum(after.values())) // replays
    return fall, movers


def read_perf(path: Path):
    """``{(kind, phase): (ops, iters, device_ms)}``, ``device_ms`` kept as the artifact spells it.

    Every figure is carried as the **string** the artifact printed, never round-tripped through
    ``float``: ``float("256.570")`` formats back as ``256.57``, and a README that quotes 256.57 while
    the artifact says 256.570 is exactly the kind of near-miss the figure audit cannot see.
    """
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        m = PERF_ROW.match(line)
        if m:
            out[(m["kind"], m["phase"])] = (int(m["ops"]), int(m["iters"]), m["ms"])
    return out


def read_sparse_shares(path: Path):
    """``{(kind, phase): [sparse-matmul % of window, ...]}`` from the 'where the time goes' section."""
    out, current = {}, None
    for line in path.read_text(errors="replace").splitlines():
        m = WINDOW_HEAD.match(line)
        if m:
            current = (m["kind"], m["phase"])
            out[current] = []
            continue
        m = SHARE_ROW.match(line)
        if m and current:
            out[current].append((float(m["pct"]), float(m["ms"])))
    return out


def read_bench(path: Path, key=lambda m: (m["impl"], m["kind"])):
    """``{key + (phase,): (wall_ms, throughput)}``, both kept as the artifact's own strings.

    ``key`` picks which BENCH fields identify a row: the headline A/B varies only the
    implementation, the sweeps vary ``moe_group_tokens`` or ``rope_mode`` instead.
    """
    out = {}
    for m in BENCH.finditer(path.read_text(errors="replace")):
        if m["pwall"]:
            out[key(m) + ("prefill",)] = (m["pwall"], m["tok"])
        else:
            out[key(m) + ("decode",)] = (m["dwall"], m["steps"])
    return out


def read_slow(path: Path):
    """``({(kind, phase): (rows, share, window_us)}, {(kind, phase): [row dict, ...]})``.

    Each row dict carries ``us``, ``cores``, ``dram`` and ``flops`` alongside the geometry, because
    §5.4 quotes all of them and review round 5 found two of them transcribed by hand and wrong.
    """
    heads, rows, current = {}, {}, None
    for line in path.read_text(errors="replace").splitlines():
        m = SLOW_HEAD.match(line)
        if m:
            current = (m["kind"], m["phase"])
            heads[current] = (int(m["rows"]), float(m["share"]), int(m["window"]), m["perstep"])
            rows[current] = []
            continue
        m = SLOW_ROW.match(line)
        if m and current:
            rows[current].append(
                {
                    "us": int(m["us"]),
                    "launches": int(m["launches"]),
                    "cores": int(m["cores"]),
                    "dram": m["dram"],
                    "flops": m["flops"],
                    "geom": m["geom"].strip(),
                }
            )
    return heads, rows


class Figures:
    """Percentages computed here, recorded with the arithmetic that produced them."""

    def __init__(self):
        self.lines = []

    def drop(self, after, before, label):
        """Percentage fall from ``before`` to ``after``; both are the artifacts' own strings."""
        value = 100 * (1 - float(after) / float(before))
        self.lines.append(f"{label}: {value:.1f} % = 100 * (1 - {after} / {before})")
        return f"{value:.1f}"

    def total(self, parts, label, exact=None):
        """Sum a set of shares. ``exact`` overrides the sum when the parts are already rounded.

        Round 15 found "81.8 % of both prefill windows" to be a coincidence of summing pre-rounded
        per-op shares: the two windows are 81.86 % and 81.80 %, which round to different figures.
        Callers that can compute the share from unrounded time pass it as ``exact``.
        """
        value = sum(parts) if exact is None else exact
        basis = " + ".join(f"{p}" for p in parts)
        if exact is not None:
            # Not "unrounded": the ms/iter column this comes from carries three decimals, so this is
            # more precise than summing the one-decimal percentages but is not the exact device-time
            # ratio. Round 18 found the old label claiming a precision it did not have.
            basis += f"  (recomputed from the 3-decimal ms/iter column: {exact:.4f})"
        self.lines.append(f"{label}: {value:.1f} % = " + basis)
        return f"{value:.1f}"

    def note(self, value, label, basis):
        self.lines.append(f"{label}: {value} = {basis}")
        return value


def blocks(fig: Figures):
    after = read_perf(DOC / "tracy/perf_summary.txt")
    before = read_perf(FUNCTIONAL / "tracy/perf_summary.txt")
    bench = read_bench(L / "ab_functional_vs_fused.txt")
    slow_after, rows_after = read_slow(DOC / "tracy/slow_ops_summary.txt")
    slow_before, rows_before_all = read_slow(FUNCTIONAL / "tracy/slow_ops_summary.txt")
    shares = read_sparse_shares(DOC / "tracy/perf_summary.txt")
    out = {}

    label = {"prefill": "prefill, 2048 tokens", "decode": "decode, traced"}
    unit = {"prefill": "tok/s", "decode": "steps/s"}

    head = ["| Window | before (functional) | after (fused) | device kernel time |", "| --- | --- | --- | --- |"]
    body = [
        "| Layer kind | Phase | device kernel time before → after | wall clock before → after | throughput before → after |",
        "| --- | --- | --- | --- | --- |",
    ]
    for kind in KINDS:
        for phase in PHASES:
            dev_b, dev_a = before[(kind, phase)][2], after[(kind, phase)][2]
            wall_b, thr_b = bench[("functional", kind, phase)]
            wall_a, thr_a = bench[("fused", kind, phase)]
            dev_pct = fig.drop(dev_a, dev_b, f"{kind} {phase} device kernel time")
            wall_pct = fig.drop(wall_a, wall_b, f"{kind} {phase} wall clock")
            head.append(
                f"| `{kind}` {label[phase]} | {wall_b} ms / {thr_b} {unit[phase]} "
                f"| **{wall_a} ms / {thr_a} {unit[phase]}** | {dev_b} → **{dev_a} ms** |"
            )
            body.append(
                f"| `{kind}` | {label[phase]} | {dev_b} → **{dev_a} ms** (−{dev_pct} %) "
                f"| {wall_b} → **{wall_a} ms** (−{wall_pct} %) | {thr_b} → **{thr_a} {unit[phase]}** |"
            )
    out["perf-headline"] = "\n".join(head)
    # The op-count sentence. It sits outside the table but quotes the same artifact's first column,
    # and review round 4 found it quoting 1043 where the profiler said 1041.
    counts = {(k, ph): (before[(k, ph)][0], after[(k, ph)][0]) for k in KINDS for ph in PHASES}
    min_fall = f"{min(100 * (1 - float(after[w][2]) / float(before[w][2])) for w in before):.1f}"
    fig.note(min_fall, "smallest device-time fall across the four windows, %", "the four perf_summary rows")
    # The decode fall, attributed from the per-op-code diff of the two committed captures rather than
    # asserted. `movers` is every op code that changed, so the itemisation is exhaustive by
    # construction and the residual below is arithmetic, not an estimate.
    attribution = []
    for kind in KINDS:
        replays = int(before[(kind, "decode")][1])
        fall, movers = decode_opcode_diff(kind, replays)
        fig.note(str(fall), f"{kind} decode op-count fall per replay", f"{kind}/decode_perf_report.csv, both stages")
        signed = lambda d: f"{d:+d}".replace("-", "−")  # noqa: E731 - match the prose's minus sign
        gone = ", ".join(f"`{code}` {signed(delta)}" for delta, code in movers if delta < 0)
        added = ", ".join(f"`{code}` {signed(delta)}" for delta, code in movers if delta > 0)
        sparse = next((d for d, c in movers if c == "SparseMatmul"), 0)
        assert sum(d for d, _ in movers) == -fall, f"{kind}: itemisation does not close on the fall"
        attribution.append(
            f"* **`{kind}`, −{fall} launches per replay.** Removed: {gone}. Added: {added} —\n"
            f"  which closes exactly on the −{fall}. The packed gate/up matmul is the `SparseMatmul`\n"
            f"  term, {signed(sparse)} of it."
        )
    out["op-counts"] = (
        "Op counts moved in **opposite directions** by phase, for two different reasons. Prefill rose\n"
        "because of the finer expert grouping, not the fusing; decode fell *because* of the fusing —\n"
        "`moe_group_tokens` does not affect it at all (one 32-token group either way). Decode fell\n"
        "(`linear_attention` "
        f"{counts[('linear_attention', 'decode')][0]} → {counts[('linear_attention', 'decode')][1]} rows, "
        f"`full_attention`\n{counts[('full_attention', 'decode')][0]} → {counts[('full_attention', 'decode')][1]}, "
        "over 32 replays) while prefill rose "
        f"({counts[('linear_attention', 'prefill')][0]} → {counts[('linear_attention', 'prefill')][1]} and "
        f"{counts[('full_attention', 'prefill')][0]} → {counts[('full_attention', 'prefill')][1]}), because a\n"
        "2048-token prefill now issues 64 expert groups instead of 8. Device time fell by at least\n"
        f"{min_fall} % in every one of the four windows regardless, which is the point: launch count is not\n"
        "the objective, and this stage traded prefill launches for a much larger reduction in redundant\n"
        "expert FLOPs.\n"
        "\n"
        "Where the decode fall actually comes from, per replay, from the per-op-code diff of the two\n"
        "committed `decode_perf_report.csv` tables (every op code that moved, so the list is complete):\n"
        "\n" + "\n".join(attribution) + "\n"
        "\n"
        "So the fall is **relayout and elementwise elimination**, not the MoE packing: the dedicated\n"
        "head-split, RoPE and cache-update ops each replace a slice/reshape/permute/transpose sequence,\n"
        "and the flat rank-3 delta-rule contract removes the per-head tilize/untilize round trips. The\n"
        "gate/up packing is a real launch saving but a small one by count — its value is in FLOPs and\n"
        "in one fewer `UnaryOpType::FILL` of the `num_experts`-wide output (§5.4), not in the row total."
    )
    out["perf-result"] = "\n".join(body)

    # Decode rows are whole-window totals over every traced replay, where every other decode figure in
    # §5.4 is per step; round 15 found the two mixed in one table with no label. The per-step counts are
    # in slow_ops_summary.txt and are added as a second column rather than replacing the totals.
    table = [
        "| Window | `SLOW` rows before | after | per traced step before → after | share of window before → after |",
        "| --- | --- | --- | --- | --- |",
    ]
    for kind in KINDS:
        for phase in PHASES:
            rb, sb, _, pb = slow_before[(kind, phase)]
            ra, sa, _, pa = slow_after[(kind, phase)]
            step = f"{pb} → **{pa}**" if pb and pa else "n/a (one pass)"
            table.append(f"| `{kind}` {phase} | {rb} | **{ra}** | {step} | {sb} % → **{sa} %** |")
    out["slow-table"] = "\n".join(table)

    # Absolute SLOW time in the linear prefill window, before and after, summed from the rows.
    sum_a = sum(r["us"] for r in rows_after[("linear_attention", "prefill")])
    sum_b = sum(r["us"] for r in rows_before_all[("linear_attention", "prefill")])
    fig.note(str(sum_a), "fused linear prefill SLOW time, us", "sum of the rows slow_ops_summary.txt prints")
    fig.note(str(sum_b), "functional linear prefill SLOW time, us", "sum of the rows the functional summary prints")
    slow_drop = fig.drop(str(sum_a), str(sum_b), "fall in linear prefill SLOW time")
    window_drop = fig.drop(
        after[("linear_attention", "prefill")][2],
        before[("linear_attention", "prefill")][2],
        "linear prefill window, for comparison with the SLOW-time fall",
    )
    out["slow-absolute"] = (
        f"The prefill *share* rises slightly while the *count* falls by more than two thirds, because the\n"
        f"window itself got shorter. The absolute `SLOW` time fell as well: summing the per-group rows\n"
        f"`slow_ops_summary.txt` lists for `linear_attention` prefill gives {sum_b} µs before and {sum_a} µs\n"
        f"after (each row rounded as the summary prints it), so the absolute `SLOW` time fell {slow_drop} %\n"
        f"while the window itself fell {window_drop} %."
    )

    # The two claims that used to be argued from eyeballed ranges.
    moe = {}
    for kind in KINDS:
        for phase in PHASES:
            # The per-op percentages are rounded to one decimal in the artifact, so summing them can
            # make two different windows print the same figure. Recompute from the ms/iter column,
            # which carries three decimals, and pass it as the exact value.
            pcts = [p for p, _ in shares[(kind, phase)]]
            ms = sum(m for _, m in shares[(kind, phase)])
            window_ms = float(after[(kind, phase)][2])
            moe[(kind, phase)] = fig.total(pcts, f"MoE share of {kind} {phase} device time", exact=100 * ms / window_ms)
    prefill_note = (
        f"{moe[('linear_attention', 'prefill')]} %"
        if moe[("linear_attention", "prefill")] == moe[("full_attention", "prefill")]
        else f"{moe[('linear_attention', 'prefill')]} % / {moe[('full_attention', 'prefill')]} %"
    )
    # The shared-LHS packings this stage introduced, identified by their packed output width:
    # 12352 = the gated-DeltaNet in-projection pack, 9216 = the attention in-projection + output-gate
    # pack, 1056 = the shared expert's gate/up + sigmoid-router pack. Selecting by width rather than
    # by a device-time threshold keeps the claim tied to the rewrites it is about.
    packed_widths = ("12352", "9216", "1056")
    flops = sorted(
        float(r["flops"].split("-")[-1])
        for kind in KINDS
        for r in rows_after[(kind, "prefill")]
        if r["geom"].split()[-1] in packed_widths
    )
    if not flops:
        raise SystemExit("no packed-projection SLOW rows found - the geometry parse or the widths are wrong")
    lo, hi = f"{flops[0]:.1f}", f"{flops[-1]:.1f}"
    fig.note(
        f"{lo}-{hi}",
        "packed-projection FLOP utilisation range, %",
        f"the prefill SLOW rows whose packed output width is one of {packed_widths}",
    )
    # The MoE share of each window, and — explicitly — the fact that tt-perf-report cannot rate
    # these rows at all. Review round 7 caught the previous wording ("they are not SLOW") presenting
    # a *missing measurement* as a positive finding: SparseMatmul rows have no numeric nnz, so the
    # report omits their DRAM and FLOP utilisation and they are structurally ineligible for the SLOW
    # flag. There are zero SLOW SparseMatmul rows in any capture, fused or functional.
    report = (DOC / "tracy/linear_attention/decode_perf_report.txt").read_text(errors="replace")
    warning = next((ln.strip() for ln in report.splitlines() if "without numeric nnz" in ln), None)
    if warning is None:
        raise SystemExit("decode_perf_report.txt no longer carries the nnz warning; re-check the claim")
    flagged = sum(
        1
        for root in (DOC, FUNCTIONAL)
        for kind in KINDS
        for phase in PHASES
        for line in (root / f"tracy/{kind}/{phase}_perf_report.txt").read_text(errors="replace").splitlines()
        if "SLOW" in line and "SparseMatmul" in line
    )
    assert flagged == 0, f"{flagged} SparseMatmul rows are now SLOW-flagged; the wording below is stale"
    out["moe-share"] = (
        f"* **The MoE sparse matmuls are the floor.** They are {prefill_note} of both `linear_attention`\n"
        f"  and `full_attention` prefill device time, and {moe[('linear_attention', 'decode')]} % /"
        f" {moe[('full_attention', 'decode')]} % of the decode windows.\n"
        "  Cutting further means expert-major token gathering, not another graph rewrite — §8 item 1.\n"
        "  **They are absent from the `SLOW` table for a structural reason, not a good one**: the rows\n"
        "  carry no numeric `nnz`, so `tt-perf-report` omits their DRAM and FLOP utilisation entirely\n"
        f"  (`{warning}`)\n"
        "  and the `SLOW` rule — neither metric near the roofline — has nothing to test. There are **0**\n"
        "  `SLOW`-flagged `SparseMatmul` rows in any capture, fused or functional, and that says nothing\n"
        "  about their efficiency. `tracy/PROVENANCE.md` records the omission; this stage measures their\n"
        "  *share of device time*, which is what the claim above rests on, and leaves their roofline\n"
        "  efficiency unmeasured.\n"
        f"* **The shared-LHS packed projections are themselves `SLOW` rows**, at {lo}–{hi} % of\n"
        "  peak FLOPs on the full grid: that is roughly what HiFi4 alone predicts (4 passes), so it is a math\n"
        "  fidelity and program-config question, i.e. exactly the next stage's job."
    )

    # Which launches actually left the SLOW table, and why. The prose used to attribute the whole
    # drop to the shared-LHS packings; review round 7 showed the router hoist removes more *launches*
    # than they do, while the packings remove more *time*. Both are computed here rather than argued.
    # Which SLOW rows each rewrite removed *and* added. Round 7's version reported gross removals
    # only, so its columns did not sum to the row-count change, and its ranking ("the packings remove
    # more time") reverses once the rows they add are counted — which round 8 caught.
    ROUTER_BEFORE, ROUTER_AFTER = "256 x 2048 x 256", "2048 x 2048 x 256"
    router_nets: list[int] = []
    packing_nets: list[int] = []
    PACKED_AFTER = ("12352", "9216", "1056")
    lines = []
    for kind in KINDS:
        before_rows = rows_before_all[(kind, "prefill")]
        after_rows = rows_after[(kind, "prefill")]
        before_geoms = {r["geom"] for r in before_rows}
        after_geoms = {r["geom"] for r in after_rows}

        gone = [r for r in before_rows if r["geom"] not in after_geoms]
        added = [r for r in after_rows if r["geom"] not in before_geoms]
        r_gone = [r for r in gone if r["geom"].endswith(ROUTER_BEFORE)]
        r_added = [r for r in added if r["geom"].endswith(ROUTER_AFTER)]
        p_gone = [r for r in gone if not r["geom"].endswith(ROUTER_BEFORE)]
        p_added = [r for r in added if r["geom"].split()[-1] in PACKED_AFTER]
        assert len(r_gone) + len(p_gone) == len(gone), "an eliminated SLOW row is in neither bucket"
        assert len(r_added) + len(p_added) == len(added), f"an added SLOW row is in neither bucket: {added}"
        r_net = sum(r["us"] for r in r_gone) - sum(r["us"] for r in r_added)
        p_net = sum(r["us"] for r in p_gone) - sum(r["us"] for r in p_added)
        rows_lost = slow_before[(kind, "prefill")][0] - slow_after[(kind, "prefill")][0]
        router_nets.append(r_net)
        packing_nets.append(p_net)
        launch_net = (
            sum(r["launches"] for r in r_gone)
            - sum(r["launches"] for r in r_added)
            + sum(r["launches"] for r in p_gone)
            - sum(r["launches"] for r in p_added)
        )
        assert (
            launch_net == rows_lost
        ), f"{kind}: net launch change {launch_net} does not match the row-count change {rows_lost}"
        fig.note(str(rows_lost), f"{kind} prefill SLOW rows removed, net", "the two summaries' row counts")
        fig.note(str(r_net), f"{kind} prefill SLOW time removed by the router hoist, net us", "eliminated minus added")
        fig.note(
            str(p_net), f"{kind} prefill SLOW time removed by the shared-LHS packings, net us", "eliminated minus added"
        )
        lines.append(
            f"| `{kind}` prefill | "
            f"{sum(r['launches'] for r in r_gone)} − {sum(r['launches'] for r in r_added)}, "
            f"{sum(r['us'] for r in r_gone)} − {sum(r['us'] for r in r_added)} µs = **−{r_net} µs** | "
            f"{sum(r['launches'] for r in p_gone)} − {sum(r['launches'] for r in p_added)}, "
            f"{sum(r['us'] for r in p_gone)} − {sum(r['us'] for r in p_added)} µs = **−{p_net} µs** | "
            f"{rows_lost} |"
        )
    residuals = []
    for kind in KINDS:
        before_rows = rows_before_all[(kind, "prefill")]
        after_rows = rows_after[(kind, "prefill")]
        shared = {r["geom"] for r in before_rows} & {r["geom"] for r in after_rows}
        residuals.append(
            sum(r["us"] for r in before_rows if r["geom"] in shared)
            - sum(r["us"] for r in after_rows if r["geom"] in shared)
        )
    fig.note(
        str(residuals[0]),
        "linear prefill SLOW time change in geometries present both before and after, us",
        "neither rewrite removed these rows; they are the same op, timed twice",
    )
    router_wins = [r > p for r, p in zip(router_nets, packing_nets)]
    ranking = (
        "the router hoist is the larger net win in both windows"
        if all(router_wins)
        else "the packings are the larger net win in both windows"
        if not any(router_wins)
        else "the ranking differs by window — see the table"
    )
    out["slow-attribution"] = (
        "Two rewrites moved those rows, and each both removed and added: the **router hoist** turned the\n"
        "functional decoder's per-256-token-group router into one call per prefill, and the **shared-LHS\n"
        "packings** merged several narrow projections into one wide one that is itself `SLOW`. Counting\n"
        f"only removals would rank the packings first; counting what they add as well, {ranking}.\n"
        "Launches and device time, removed − added:\n\n"
        f"| Window | router hoist | shared-LHS packings | net `SLOW` rows |\n"
        "| --- | --- | --- | --- |\n" + "\n".join(lines) + "\n\n"
        "The two net figures account for the absolute `SLOW`-time fall reported just below to within\n"
        f"{abs(residuals[0])} µs for `linear_attention`"
        + (
            ": the remainder is the geometries that appear in\n"
            if residuals[0]
            else " — the itemisation is exhaustive, with nothing left over. Where a residual does appear it is\n"
            "  the geometries that appear in\n"
        )
        + "*both* summaries — the same op, timed twice — which neither rewrite removed and which therefore\n"
        "belong to neither column. Only the `linear_attention` window has an absolute-fall figure below;\n"
        "the `full_attention` net figures stand on their own."
    )

    def majority_word(shares):
        lo = min(float(x) for x in shares)
        return "a clear majority" if lo > 50 else "the largest single item but not a majority"

    pre_word = majority_word([moe[(k, "prefill")] for k in KINDS])
    dec_word = majority_word([moe[(k, "decode")] for k in KINDS])
    out["moe-floor-limitation"] = (
        "1. **The MoE remains the floor.** After fusing, the two sparse expert matmuls are\n"
        f"   {prefill_note} of the two prefill windows and {moe[('linear_attention', 'decode')]} % /"
        f" {moe[('full_attention', 'decode')]} % of the traced decode\n"
        f"   windows — {pre_word} in prefill, {dec_word} in decode.\n"
        "   Cutting further needs expert-major token gathering — `unified_routed_expert_ffn`, `moe_compute`\n"
        "   and `moe_gpt` all want that layout — which is a change to the routing algorithm, not to the op\n"
        "   graph, and multi-device in every in-tree instance. `work_log.md` §4.10. §5.4 also records why\n"
        "   their absence from the `SLOW` table is not evidence that they are efficient."
    )

    # ---- The suite summary the README quotes.
    for name, log in (("suite-result", L / "pytest_full_suite.txt"),):
        m = None
        for line in log.read_text(errors="replace").splitlines():
            found = PYTEST_TAIL.search(line)
            if found:
                m = found
        if m is None:
            raise SystemExit(f"{log.name}: no pytest summary line")
        out[name] = f"**{m['passed']} passed** in {m['secs']} s"

    # ---- Where the traced decode window goes, by op code.
    def decode_costs(kind):
        rows = collections.Counter()
        with open(DOC / f"tracy/{kind}/decode_perf_report.csv", newline="") as handle:
            for row in csv.DictReader(handle):
                value = (row.get("Device Time") or "").replace(",", "").strip()
                match = re.match(r"^([\d.]+)", value)
                if match:
                    rows[row["OP Code"].strip()] += float(match.group(1))
        if not rows:
            raise SystemExit(f"{kind}/decode_perf_report.csv: no Device Time rows")
        return rows

    iters = int(after[("linear_attention", "decode")][1])
    ranked = {kind: decode_costs(kind) for kind in KINDS}
    table = ["| Rank | op code | `linear_attention` µs/step | `full_attention` µs/step |", "| --- | --- | --- | --- |"]
    order = [code for code, _ in ranked["linear_attention"].most_common(6)]
    for rank, code in enumerate(order, 1):
        cells = [f"{ranked[kind].get(code, 0.0) / iters:.1f}" for kind in KINDS]
        table.append(f"| {rank} | `{code}` | {cells[0]} | {cells[1]} |")
    out["decode-cost-ranking"] = "\n".join(table)
    for code in order:
        for kind in KINDS:
            total = ranked[kind].get(code, 0.0)
            fig.note(
                f"{total / iters:.1f}",
                f"{kind} decode, {code}, us/step",
                f"{total:.1f} us summed over {iters} replays / {iters}",
            )
    # Each kind's rank in *its own* ordering. Round 15 found one rank computed from the
    # linear_attention ordering and printed for both columns; Slice is sixth in one and fifth in the
    # other, so a single word cannot be right for both.
    WORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}
    slice_ranks = {}
    for kind in KINDS:
        own = [code for code, _ in ranked[kind].most_common()]
        slice_ranks[kind] = own.index("SliceDeviceOperation") + 1
    assert all(r <= 6 for r in slice_ranks.values()), f"Slice left the top 6: {slice_ranks}"
    slice_rank = " / ".join(f"{WORD[slice_ranks[k]]} in `{k}`" for k in KINDS)

    # Dense-matmul vs slice movement per step, both kinds, from the two committed captures. Quoted as
    # a range so the sentence covers both without asserting one kind's number for the other.
    def code_us(base, kind, prefix):
        rows = collections.Counter()
        with open(base / f"tracy/{kind}/decode_perf_report.csv", newline="") as handle:
            for row in csv.DictReader(handle):
                code = row["OP Code"].strip()
                if code.startswith(prefix):
                    match = re.match(r"^([\d.]+)", (row.get("Device Time") or "").replace(",", "").strip())
                    if match:
                        rows[kind] += float(match.group(1))
        return rows[kind] / iters

    def moved(prefix):
        deltas = [code_us(DOC, k, prefix) - code_us(FUNCTIONAL, k, prefix) for k in KINDS]
        lo, hi = min(deltas), max(deltas)
        return f"{lo:+.1f} to {hi:+.1f}" if abs(hi - lo) > 0.05 else f"{lo:+.1f}"

    matmul_delta, slice_delta = moved("MatmulDeviceOperation"), moved("SliceDeviceOperation")
    fig.note(
        matmul_delta,
        "dense Matmul* movement per traced decode step, us, both kinds",
        "the two decode_perf_report.csv sets, fused minus functional",
    )
    fig.note(
        slice_delta,
        "Slice movement per traced decode step, us, both kinds",
        "the two decode_perf_report.csv sets, fused minus functional",
    )
    pack = (L / "probe_gate_up_pack.txt").read_text(errors="replace")
    pack_pair = re.findall(r"GATEUP .*?wall=\s*([\d.]+) us/call", pack)
    if len(pack_pair) < 2:
        raise SystemExit("probe_gate_up_pack.txt: expected two GATEUP wall figures")
    pack_before, pack_after = pack_pair[0], pack_pair[1]
    slice_us = {kind: ranked[kind]["SliceDeviceOperation"] / iters for kind in KINDS}
    window_us = {kind: float(after[(kind, "decode")][2]) * 1000 for kind in KINDS}
    slice_share = {kind: 100 * slice_us[kind] / window_us[kind] for kind in KINDS}
    fig.note(
        f"{min(slice_us.values()):.0f}-{max(slice_us.values()):.0f}",
        "SliceDeviceOperation per traced decode step, us",
        "summed over the decode_perf_report.csv rows, divided by the replay count",
    )
    fig.note(
        f"{min(slice_share.values()):.1f}-{max(slice_share.values()):.1f}",
        "SliceDeviceOperation share of the traced decode window, %",
        "the row above over the profiler's device ms/iter",
    )
    out["slice-cost"] = (
        f"* **`SliceDeviceOperation` costs {min(slice_us.values()):.0f}–{max(slice_us.values()):.0f} µs per\n"
        f"  traced decode step** ({min(slice_share.values()):.1f}–{max(slice_share.values()):.1f} % of the\n"
        "  window), and it is the price of the shared-LHS packings: one wide matmul, then slices to recover\n"
        "  the operands, and it is not a `SLOW` row, so\n"
        f"  a reader looking only at the `SLOW` table would miss it. The table above ranks it"
        f" {slice_rank};\n"
        f"  the dense `Matmul*` codes it trades against fall {matmul_delta} µs/step while `Slice` rises\n"
        f"  {slice_delta} µs/step, so the window data alone does not settle the trade in either\n"
        f"  direction — the packing's own measured effect is the {pack_before} → {pack_after} µs/call in\n"
        "  [`logs/probe_gate_up_pack.txt`](logs/probe_gate_up_pack.txt), which covers the MoE pair only.\n"
        "  Its two dominant calls are the consecutive pair immediately after the\n"
        "  routed-expert `sparse_matmul`, i.e. they unpack that matmul's `2·I`-wide output and are\n"
        "  themselves a MoE cost."
    )

    # ---- The two elementwise aggregates, split by tracy/summarise_fill.py. Review round 8 raised
    # the third-largest decode item being unattributed; round 9 raised the fourth. Both are read out
    # of the same artifact, for this stage and the functional baseline.
    fill_path = DOC / "tracy/fill_summary.txt"
    if not fill_path.is_file():
        raise SystemExit("tracy/fill_summary.txt is missing; run tracy/summarise_fill.py")
    fill_text = fill_path.read_text(errors="replace")

    def fill_rows(stage):
        """``{(kind, phase, op): {'total','marked','share','window','launches','members':[(us,shape,act)]}}``."""
        section = fill_text.split(f"=== {stage} ===", 1)[1].split("=== ", 1)[0]
        parsed, key, where = {}, None, None
        for line in section.splitlines():
            m = re.match(r"^(?P<kind>\w+)/(?P<phase>\w+): window", line)
            if m:
                where = (m["kind"], m["phase"])
                continue
            m = re.match(
                r"^  (?P<op>\w+)\s+total\s+(?P<total>[\d.]+) us/iter\s+(?P<short>\w+)\s+"
                r"(?P<marked>[\d.]+) us/iter \(\s*(?P<share>[\d.]+) %\)\s+(?P<window>[\d.]+) % of window"
                r"\s+(?P<launches>\d+) launches/iter\s+of which (?P<marked_launches>\d+) are \w+",
                line,
            )
            if m:
                key = (*where, m["op"])
                parsed[key] = dict(m.groupdict(), members=[])
                continue
            m = re.match(
                r"^\s+(?P<us>[\d.]+) us  folded=(?P<act>\S+)\s+next=(?P<next>\S+)\s+reported_out=(?P<shape>\S+)",
                line,
            )
            if m and key:
                parsed[key]["members"].append((float(m["us"]), m["shape"], m["act"], m["next"]))
        return parsed

    fused_fill, func_fill = fill_rows("fused"), fill_rows("functional")
    unary = {k: fused_fill[(k, "decode", "UnaryDeviceOperation")] for k in KINDS}
    binary = {k: fused_fill[(k, "decode", "BinaryNgDeviceOperation")] for k in KINDS}
    func_unary = {k: func_fill[(k, "decode", "UnaryDeviceOperation")] for k in KINDS}
    func_binary = {k: func_fill[(k, "decode", "BinaryNgDeviceOperation")] for k in KINDS}

    # The FILL sub-total moved the *other* way from the aggregate it sits in; round 15 found one word
    # ("aggregates") standing for both quantities, with the bolded conclusion over the wrong one.
    fill_delta_pct = 100 * (
        float(unary["linear_attention"]["marked"]) / float(func_unary["linear_attention"]["marked"]) - 1
    )
    fig.note(
        f"{fill_delta_pct:+.1f}",
        "linear decode FILL sub-total, fused vs functional, %",
        "the two fill_summary.txt FILL lines",
    )

    def top_fill(rows):
        """The `sparse_matmul` zero-fills: members with no folded activation whose consumer is one."""
        return [m for m in rows["members"] if m[2] == "-" and "SparseMatmul" in m[3]][:2]

    def width(member):
        """The width the fill actually clears, read off the matmul it precedes."""
        return member[3].split("w=")[-1]

    lin_fill = top_fill(unary["linear_attention"])
    lin_func_fill = top_fill(func_unary["linear_attention"])
    assert len(lin_fill) == 2 and len(lin_func_fill) >= 2, "fewer than two sparse_matmul fills found"
    out["fill-cost"] = (
        "* **The `UnaryDeviceOperation` aggregate is almost entirely one MoE cost**, and it is the\n"
        "  third-largest item in the traced decode window. Splitting it by the raw capture\n"
        "  ([`tracy/summarise_fill.py`](tracy/summarise_fill.py) →\n"
        "  [`tracy/fill_summary.txt`](tracy/fill_summary.txt)) attributes\n"
        f"  {unary['linear_attention']['share']} % / {unary['full_attention']['share']} % of it to\n"
        "  `UnaryOpType::FILL` — `ttnn.sparse_matmul` zero-initialising its `num_experts`-wide output\n"
        "  before writing the active experts' blocks. Two of `linear_attention` decode's\n"
        f"  {unary['linear_attention']['marked_launches']} `FILL` launches are the ones that matter, one per\n"
        f"  `sparse_matmul` call: {lin_fill[0][0]:.3f} µs clearing the {width(lin_fill[0])}-wide output of\n"
        f"  the down projection and {lin_fill[1][0]:.3f} µs clearing the {width(lin_fill[1])}-wide output of\n"
        "  the packed gate/up. (Each fill's width is read from the matmul it precedes, not from its own\n"
        "  reported shape, which the profiler does not always update — see the script.)\n"
        "  **It is not reachable by graph fusing**: it is inside the op, its width is the `num_experts`\n"
        "  output the op's contract requires, and the call count is a settled trade at its measured\n"
        "  optimum rather than a floor the routing imposes — `logs/ab_moe_group_tokens.txt` runs the\n"
        "  sweep, and larger groups issue *fewer* calls but were slower end to end. Removing the fill\n"
        "  means never materialising a 256-wide output —\n"
        "  expert-major gathering, §8 item 1.\n"
        "  **Not a regression.** The functional decoder's largest fill,"
        f" {lin_func_fill[0][0]:.3f} µs, clears the same\n"
        f"  {width(lin_func_fill[0])}-wide down-projection output at essentially the same cost; what this\n"
        "  stage's gate/up packing changed is the *other* end — the baseline pays two narrower fills for\n"
        "  its two separate projections where this stage pays one wider. The `FILL` sub-total is\n"
        f"  essentially unchanged — {unary['linear_attention']['marked']} µs against"
        f" {func_unary['linear_attention']['marked']} µs,\n"
        f"  {fill_delta_pct:+.1f} % — so the fill cost itself did not fall. What fell is the\n"
        f"  `UnaryDeviceOperation` aggregate it sits inside,"
        f" {func_unary['linear_attention']['total']} → {unary['linear_attention']['total']} µs.\n"
        "  `work_log.md` §4.13."
    )

    lin_bin = binary["linear_attention"]["members"]
    swiglu = [m for m in lin_bin if m[2] == "SILU"][:1]
    plain = [m for m in lin_bin if m[2] == "-" and "Unary" in m[3]][:1]
    assert swiglu and plain, "could not identify both dominant BinaryNg members"
    out["binary-cost"] = (
        "* **`BinaryNgDeviceOperation` is the fourth-largest decode item, and it is two distinct MoE\n"
        "  costs, not one.** `BinaryOpType::MUL` is"
        f" {binary['linear_attention']['share']} % / {binary['full_attention']['share']} % of the\n"
        f"  aggregate over {binary['linear_attention']['launches']} launches per replay for"
        f" `linear_attention`, and the two\n"
        "  largest differ in kind:\n"
        f"    * {swiglu[0][0]:.3f} µs **with SiLU folded into its input activation** — the routed\n"
        "      experts' SwiGLU. Here the multiply *is* the fused form (§3.3); what is left is the\n"
        "      expert-activation width, the same lever as the zero-fill above.\n"
        f"    * {plain[0][0]:.3f} µs with **no** folded activation, immediately before the next\n"
        "      `sparse_matmul`'s zero-fill — the router-score multiply this stage moved ahead of the down\n"
        "      projection (§3.2). It is a genuinely separate op, and §4.17 records why it stays one: the\n"
        "      ttnn op that fuses it into the reduction (`deepseek_moe_fast_reduce_nc_fused`) does accept\n"
        "      this decoder's shapes, but it is **not faster here** - §3.2 already moved this multiply\n"
        "      onto the `moe_intermediate`-wide input, so there is no win left for it to take - and\n"
        "      adopting it would mean scoring the down projection's bfloat16 output instead, which costs\n"
        "      accuracy. §4.12 has both arms.\n"
        f"  The aggregate is nonetheless **smaller than the baseline's**:"
        f" {binary['linear_attention']['total']} µs against\n"
        f"  {func_binary['linear_attention']['total']} µs per replay, because that placement moved the\n"
        "  multiply from the `hidden_size`-wide residual stream to the `moe_intermediate`-wide expert\n"
        "  activation."
    )

    # ---- §5.4's two remaining SLOW bullets.    # ---- §5.4's two remaining SLOW bullets.    # ---- §5.4's two remaining SLOW bullets.    # ---- §5.4's two remaining SLOW bullets.
    def decode_rows(kind, geometry, source=None):
        table_rows = source if source is not None else rows_after
        found = [r for r in table_rows[(kind, "decode")] if r["geom"] == geometry]
        if not found:
            raise SystemExit(f"{kind} decode has no SLOW row for {geometry!r}")
        return found

    _, rows_before_decode = read_slow(FUNCTIONAL / "tracy/slow_ops_summary.txt")
    largest = {kind: max(rows_after[(kind, "decode")], key=lambda r: r["us"])["geom"] for kind in KINDS}
    assert len(set(largest.values())) == 1, f"the largest decode SLOW group differs by layer kind: {largest}"
    largest_geom = largest["linear_attention"]
    dram_out = [decode_rows(kind, largest_geom)[0]["dram"] for kind in KINDS]  # linear, then full
    router_rows = {kind: decode_rows(kind, "32 x 2048 x 256")[0] for kind in KINDS}
    dram_router = [router_rows[kind]["dram"] for kind in KINDS]  # linear, then full
    router_cores = {r["cores"] for r in router_rows.values()}
    assert len(router_cores) == 1, f"the router runs on different core counts per layer kind: {router_cores}"
    state_geoms = ("b={32} x 32 x 128 x 128", "b={32} x 128 x 32 x 128")
    state_now = [r for g in state_geoms for r in rows_after[("linear_attention", "decode")] if r["geom"] == g]
    state_before = [
        r for g in state_geoms for r in rows_before_decode[("linear_attention", "decode")] if r["geom"] == g
    ]
    assert state_now and state_before, "no recurrent-state SLOW rows in one of the summaries"
    assert len(state_now) == 1, f"expected one recurrent-state SLOW group after, got {len(state_now)}"
    fig.note(str(sum(r["us"] for r in state_now)), "recurrent-state matmul SLOW time after, us", "1 group")
    fig.note(
        str(sum(r["us"] for r in state_before)),
        "recurrent-state matmul SLOW time before, us",
        f"{len(state_before)} groups in the functional summary",
    )
    before_detail = ", ".join(f"{r['us']} µs on {r['cores']} cores" for r in state_before)
    out["slow-bullets"] = (
        f"* **In decode the largest `SLOW` group is `{largest_geom}`** (the output projection) at\n"
        f"  {' / '.join(dram_out)} % of DRAM bandwidth — a dtype/layout target, not a core-count one.\n"
        f"* **`32 × 2048 × 256`** (the router) still runs on {sorted(router_cores)[0]} cores at\n"
        f"  {' / '.join(dram_router)} % of DRAM bandwidth. The recurrent-state matmuls\n"
        f"  (`b={{32}} 32 × 128 × 128`) are now one {state_now[0]['cores']}-core group costing\n"
        f"  {sum(r['us'] for r in state_now)} µs, where the functional decoder had {len(state_before)} groups\n"
        f"  ({before_detail}) totalling {sum(r['us'] for r in state_before)} µs — this stage's `core_grid`\n"
        f"  fix moved the dominant one off 4 cores."
    )

    # ---- §5.3's two configuration sweeps, and the honest split of the prefill win.
    groups = read_bench(L / "ab_moe_group_tokens.txt", key=lambda m: (m["groups"], m["kind"]))
    sizes = sorted({int(g) for g, _, _ in groups}, key=int)
    chosen = min(sizes)
    table = ["| tokens per `sparse_matmul` call | `linear_attention` | `full_attention` |", "| --- | --- | --- |"]
    for size in sizes:
        cells = [groups[(str(size), kind, "prefill")][0] for kind in KINDS]
        mark = "**" if size == chosen else ""
        table.append(f"| {mark}{size}{mark} | {mark}{cells[0]} ms{mark} | {mark}{cells[1]} ms{mark} |")
    out["moe-group-sweep"] = "\n".join(table)

    rope = read_bench(L / "ab_rope_mode.txt", key=lambda m: (m["rope"], m["kind"]))
    label = {
        "partial": "**`partial`** (4 ops in prefill, `rope_dim`-wide table)",
        "full": "`full` (1 op, `head_dim`-wide table)",
    }
    table = ["| mode | prefill 2048 | traced decode |", "| --- | --- | --- |"]
    decode_ms = {mode: rope[(mode, "full_attention", "decode")][0] for mode in label}
    best = min(decode_ms, key=lambda mode: float(decode_ms[mode]))
    for mode in ("partial", "full"):
        mark = "**" if mode == best else ""
        prefill_ms = rope[(mode, "full_attention", "prefill")][0]
        table.append(f"| {label[mode]} | {prefill_ms} ms | {mark}{decode_ms[mode]} ms{mark} |")
    out["rope-mode-sweep"] = "\n".join(table)

    decode_falls = [
        100 * (1 - float(after[(kind, "decode")][2]) / float(before[(kind, "decode")][2])) for kind in KINDS
    ] + [
        100 * (1 - float(bench[("fused", kind, "decode")][0]) / float(bench[("functional", kind, "decode")][0]))
        for kind in KINDS
    ]
    dec_lo, dec_hi = f"{min(decode_falls):.1f}", f"{max(decode_falls):.1f}"
    fig.note(f"{dec_lo}-{dec_hi}", "traced decode fall, both kinds and both measures, %", "the §5.2 table")

    at_256 = [groups[("256", kind, "prefill")][0] for kind in KINDS]
    # Per kind: round 16 found one kind's split quoted while the sentence named both kinds' figures.
    split = {}
    for kind, at in zip(KINDS, at_256):
        base = float(bench[("functional", kind, "prefill")][0])
        split[kind] = (base - float(at), base - float(bench[("fused", kind, "prefill")][0]))
    fusing_only, total = split["linear_attention"]
    fig.note(
        f"{fusing_only:.0f}",
        "linear prefill saving attributable to graph fusing alone, ms",
        f"{bench[('functional', 'linear_attention', 'prefill')][0]} - {at_256[0]} (moe_group_tokens=256)",
    )
    fig.note(
        f"{total:.0f}",
        "linear prefill saving in total, ms",
        f"{bench[('functional', 'linear_attention', 'prefill')][0]} - "
        f"{bench[('fused', 'linear_attention', 'prefill')][0]}",
    )
    out["prefill-win-split"] = (
        f"**Where the prefill win comes from, honestly split.** At `moe_group_tokens=256` — the value\n"
        f"`tt/moe.py` uses — the fused decoder is {at_256[0]} / {at_256[1]} ms\n"
        f"([`logs/ab_moe_group_tokens.txt`](logs/ab_moe_group_tokens.txt)), so graph fusing alone accounts\n"
        f"for {split['linear_attention'][0]:.0f} of the {split['linear_attention'][1]:.0f} ms "
        f"`linear_attention` saving and {split['full_attention'][0]:.0f} of the "
        f"{split['full_attention'][1]:.0f} ms `full_attention` one;\n"
        f"the rest is the expert-group constant, which is a one-line change\n"
        f"the functional MoE could also take. Decode is the reverse: it runs a single 32-token group at every\n"
        f"setting, so the whole of its fall — {dec_lo}–{dec_hi} % across both layer kinds and both the device\n"
        f"and wall-clock measures — is graph fusing."
    )

    return out


def splice(text, generated):
    for name, body in generated.items():
        open_m, close_m = f"<!-- generated:{name} -->", f"<!-- /generated:{name} -->"
        pattern = re.compile(re.escape(open_m) + r".*?" + re.escape(close_m), re.S)
        if not pattern.search(text):
            raise SystemExit(f"README.md has no '{name}' generated block ({open_m} ... {close_m})")
        # A single-line body is spliced inline so it can sit inside a sentence; a table gets its own
        # lines.
        sep = "" if "\n" not in body else "\n"
        text = pattern.sub(lambda _m, b=body, o=open_m, c=close_m, s=sep: f"{o}{s}{b}{s}{c}", text, count=1)
    return text


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    fig = Figures()
    generated = blocks(fig)
    derived = (
        "Percentages quoted in README.md that are computed from measured figures rather than read\n"
        "off a run, each shown with the arithmetic and the operands it was computed from. Written by\n"
        "logs/make_readme_perf.py; this file is what sources those figures for audit_figures.py.\n\n"
        + "\n".join(fig.lines)
        + "\n"
    )
    if mode == "--write":
        DERIVED_OUT.write_text(derived)
        current = README.read_text()
        updated = splice(current, generated)
        README.write_text(updated)
        print(f"wrote {DERIVED_OUT.name} ({len(fig.lines)} derived figures)")
        print("README.md unchanged" if updated == current else "README.md updated")
        return 0
    if mode == "--check":
        current = README.read_text()
        stale = splice(current, generated) != current
        if not DERIVED_OUT.is_file() or DERIVED_OUT.read_text() != derived:
            print("STALE-DERIVED tracy/derived_figures.txt is out of date; re-run make_readme_perf.py --write")
            return 1
        if stale:
            print("STALE-README README.md perf blocks do not match the profiler artifacts; re-run --write")
            return 1
        print("README.md generated perf blocks match the current profiler artifacts")
        return 0
    for name, body in generated.items():
        print(f"<!-- generated:{name} -->\n{body}\n<!-- /generated:{name} -->\n")
    print(derived)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

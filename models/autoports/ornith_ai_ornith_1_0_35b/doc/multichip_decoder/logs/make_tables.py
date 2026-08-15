# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the measured tables in README.md and work_log.md from the committed artifacts.

Every review round of this stage found transcribed figures that had gone stale, and every re-run of
``run_evidence.sh`` moves a few microseconds and invalidates a few more. ``audit_figures.py`` catches
that; this removes the cause. Each table below is rebuilt from its artifact and spliced back between
markers, so a re-measurement is a one-command update rather than a transcription exercise.

    python .../doc/multichip_decoder/logs/make_tables.py            # rewrite
    python .../doc/multichip_decoder/logs/make_tables.py --check    # fail if a table is stale

The markers are HTML comments, invisible in rendered markdown:

    <!-- TABLE:name -->
    ...generated...
    <!-- /TABLE:name -->

Prose figures are still hand-written and still checked by ``audit_figures.py``; only the tables are
generated. That split is deliberate — a table is a transcription of a log and should not be typed,
while a sentence that says *why* a number matters cannot be generated.
"""

from __future__ import annotations

import collections
import csv
import gzip
import io
import math
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent
LOGS = DOC / "logs"
TRACY = DOC / "tracy"
KINDS = ("linear_attention", "full_attention")


def read(path: Path) -> str:
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes()).decode(errors="replace")
    if path.is_file():
        return path.read_text(errors="replace")
    packed = path.with_suffix(path.suffix + ".gz")
    return gzip.decompress(packed.read_bytes()).decode(errors="replace")


def bench() -> dict:
    rows = {}
    for line in read(LOGS / "ab_single_vs_multichip.txt").splitlines():
        if not line.startswith("BENCH"):
            continue
        parts = line.split()
        tag = next(x for x in parts if x.startswith("tag="))[4:]
        layer = next(x for x in parts if x.startswith("layer="))[6:]
        phase = "prefill" if "prefill" in line else "decode"
        value = (
            float(line.split("wall=")[1].split()[0])
            if phase == "prefill"
            else float(line.split("wall/iter=")[1].split()[0])
        )
        rows[(tag, layer, phase)] = value
    return rows


def table_bench() -> str:
    rows = bench()
    out = [
        "| layer kind | phase | single-chip | 1x4 replication control | multichip | speedup | efficiency |",
        "|---|---|---|---|---|---|---|",
    ]
    for layer, kind in (("0", "linear_attention"), ("3", "full_attention")):
        for phase, label in (("prefill", "prefill 2048"), ("decode", "decode (traced)")):
            b = rows[("single-chip-baseline", layer, phase)]
            c = rows[("replication-control", layer, phase)]
            m = rows[("multichip", layer, phase)]
            # Decode is quoted to 3 decimals and prefill to 2, matching the precision the
            # harness prints: rounding further would mint a figure no artifact contains.
            digits = 2 if phase == "prefill" else 3
            out.append(
                f"| {kind} | {label} | {b:.{digits}f} ms | {c:.{digits}f} ms | **{m:.{digits}f} ms** | "
                f"**{b / m:.3f}x** | {100 * b / m / 4:.1f}% |"
            )
    return "\n".join(out)


def table_ccl() -> str:
    text = read(LOGS / "probe_ccl.txt")
    shapes = [
        ("decode", "decode (batch 1, 32 rows)"),
        ("rows64", "64 rows"),
        ("rows96", "96 rows"),
        ("rows128", "128 rows"),
        ("rows256", "256 rows"),
        ("rows512", "512 rows"),
        ("decode_b32", "decode batch 32 (1024 rows)"),
        ("prefill_2048", "prefill 2048"),
    ]
    arms = ["all_reduce_ring", "rs_ag_ring", "rs_only_ring", "all_reduce_linear", "ag_stack_sum", "all_reduce_async"]
    head = ["`all_reduce` Ring", "`rs_ag` Ring", "`rs_only` Ring", "`all_reduce` Linear", "`stack_sum`", "`async`"]
    out = ["| shape | " + " | ".join(head) + " |", "|---" * (len(head) + 1) + "|"]
    for key, label in shapes:
        cells = []
        for arm in arms:
            m = re.search(rf"^CCL {key} {arm} trace ([0-9.]+)", text, re.M)
            cells.append(m.group(1) if m else "—")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_packet() -> str:
    """The fabric packet size the runtime asks for, against the build default it warns about.

    Traced microseconds per collective at the shapes the layer produces. `CCL` rows are the shipped
    configuration (8192 B); `CCLPKT` rows are the same sweep at the build default (4352 B), which is
    what every CCL dispatch warned about until review round 5 measured it.
    """
    text = read(LOGS / "probe_ccl.txt")
    shapes = [
        ("decode", "decode (batch 1, 32 rows)"),
        ("rows128", "128 rows"),
        ("rows512", "512 rows"),
        ("decode_b32", "decode batch 32 (1024 rows)"),
        ("prefill_2048", "prefill 2048"),
    ]
    arms = [("all_reduce_ring", "`all_reduce`"), ("ag_stack_sum", "`stack_sum`")]
    head = [f"{label} {size}" for _, label in arms for size in ("8192 B (shipped)", "4352 B (default)")]
    out = ["| shape | " + " | ".join(head) + " |", "|---" * (len(head) + 1) + "|"]
    for key, label in shapes:
        cells = []
        for arm, _ in arms:
            for tag, prefix in (("CCL", ""), ("CCLPKT", r"\d+ ")):
                m = re.search(rf"^{tag} {prefix}{key} {arm} trace ([0-9.]+)", text, re.M)
                cells.append(m.group(1) if m else "—")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_fabric() -> str:
    """Ring fabric against line fabric, on the two arms the layer actually ships.

    A separate question from the ``topology`` argument the ``ccl`` table sweeps: the fabric config is
    set before ``open_mesh_device``, so it needs its own process, and rounds 0-3 of this stage quoted
    the argument comparison as if it were this one.
    """
    text = read(LOGS / "probe_ccl.txt")
    shapes = [
        ("decode", "decode (batch 1, 32 rows)"),
        ("rows64", "64 rows"),
        ("rows128", "128 rows"),
        ("rows512", "512 rows"),
        ("decode_b32", "decode batch 32 (1024 rows)"),
        ("prefill_2048", "prefill 2048"),
    ]
    head = [
        "`all_reduce` ring fabric",
        "`all_reduce` line fabric",
        "`stack_sum` ring fabric",
        "`stack_sum` line fabric",
    ]
    out = ["| shape | " + " | ".join(head) + " |", "|---" * (len(head) + 1) + "|"]
    for key, label in shapes:
        cells = []
        for arm in ("all_reduce_ring", "ag_stack_sum"):
            for tag, prefix in (("CCL", ""), ("CCLFAB", "line ")):
                m = re.search(rf"^{tag} {prefix}{key} {arm} trace ([0-9.]+)", text, re.M)
                cells.append(m.group(1) if m else "—")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_moepar() -> str:
    """EP against intermediate-TP against unsharded, from ``probe_expert_parallel.txt``.

    Each arm at **its own** representative active-expert count, because that is what the layer runs:
    EP is precisely what changes the count, so holding it fixed across arms would price a workload
    neither arm has.
    """
    rows = []
    for line in read(LOGS / "probe_expert_parallel.txt").splitlines():
        p = line.split()
        if p and p[0] == "MOEPAR":
            rows.append((p[1], p[2], int(p[5]), float(p[7])))
    out = ["| phase | arm | experts/device | active/device | us |", "|---|---|---|---|---|"]
    labels = {
        "single": "unsharded, still gate-selected",
        "ep": "**expert parallelism** (shipped)",
        "tp": "intermediate sharded 4 ways",
    }
    for phase in ("decode", "prefill"):
        for arm in ("single", "ep", "tp"):
            picked = [r for r in rows if r[0] == phase and r[1] == arm]
            if not picked:
                continue
            # The EP arm is swept over the active counts it can produce; the representative point is
            # the one the phase actually runs -- 2 at batch-1 decode (mean local of 8 global) and the
            # expected local union at prefill, which is the largest swept count below the block size.
            if arm == "ep":
                # Two points, not one. The mean local active count is what an average step runs; the
                # *expected maximum* over the four devices is what the collective barrier waits for,
                # and section 3 quotes its ratios at that conservative point. Review round 4 asked
                # for the arms to be priced on one basis and round 5 found only the mean shipped.
                wanted = (2, 4) if phase == "decode" else (41,)
                picked = [min(picked, key=lambda r: abs(r[2] - w)) for w in wanted]
            for index, (_, _, active, us) in enumerate(picked):
                experts = 256 if arm != "ep" else 64
                note = ""
                if arm == "ep" and phase == "decode":
                    note = " (mean local)" if index == 0 else " (>= the expected maximum, 3.512)"
                out.append(f"| {phase} | {labels[arm]}{note} | {experts} | {active} | {us:.2f} |")
    return "\n".join(out)


def table_ablayer() -> str:
    arms = collections.defaultdict(list)
    for line in read(LOGS / "ab_layer_knobs.txt").splitlines():
        p = line.split()
        if p and p[0] == "ABLAYER":
            arms[(p[1], p[2], p[4])].append((float(p[6]), float(p[7])))
    order = [
        ("ccl", "auto", "**`auto`** (shipped)"),
        ("ccl", "stack_sum", "`stack_sum`"),
        ("ccl", "all_reduce", "`all_reduce`"),
        ("ccl", "rs_ag", "`rs_ag`"),
        ("geometry", "multichip-retuned", "**multichip-retuned** (shipped)"),
        ("geometry", "single-chip-inherited", "single-chip-inherited"),
        ("routing", "select_matmul", "**`select_matmul`** (shipped)"),
        ("routing", "gather", "`gather`"),
        ("sparse", "tp-rescaled", "**tp-rescaled** (shipped)"),
        ("sparse", "single-chip-inherited", "single-chip-inherited"),
        ("cast", "block-float", "**block-float** (shipped)"),
        ("cast", "bf16", "`bf16`"),
    ]
    out = [
        "| knob | arm | linear decode | full decode | linear prefill | full prefill |",
        "|---|---|---|---|---|---|",
    ]

    def rng(values, digits):
        lo, hi = min(values), max(values)
        return f"{lo:.{digits}f}" if f"{lo:.{digits}f}" == f"{hi:.{digits}f}" else f"{lo:.{digits}f}–{hi:.{digits}f}"

    for knob, arm, label in order:
        cells = []
        for kind, index, digits in (
            ("linear_attention", 0, 3),
            ("full_attention", 0, 3),
            ("linear_attention", 1, 2),
            ("full_attention", 1, 2),
        ):
            values = [v[index] for v in arms.get((knob, arm, kind), [])]
            cells.append(rng(values, digits) if values else "—")
        out.append(f"| `{knob}` | {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_perf() -> str:
    cols, roof = [], []
    for kind in KINDS:
        for phase in ("decode", "prefill"):
            rows = list(csv.DictReader(io.StringIO(read(TRACY / kind / f"{phase}_perf_report_stacked.csv.gz"))))
            pick = {
                "sparse": lambda r: r["Op Code"].startswith("SparseMatmul"),
                "topk": lambda r: r["Op Code"].startswith("TopK"),
                "matmul": lambda r: r["Op Code"].startswith("MatmulDeviceOperation"),
                "ccl": lambda r: "AllGather" in r["Op Code"] or "ReduceScatter" in r["Op Code"],
                "dm": lambda r: r["Op Category"] == "DM",
            }
            cols.append({k: sum(float(r["Total % [%]"]) for r in rows if f(r)) for k, f in pick.items()})
            console = read(TRACY / kind / f"{phase}_perf_report.console.txt")
            roof.append(console.split("roofline (modeled ops):")[1].split("\n")[0].strip())
    order = [0, 2, 1, 3]  # linear decode, full decode, linear prefill, full prefill
    labels = [
        ("`SparseMatmul` (routed experts)", "sparse"),
        ("`TopK` (router)", "topk"),
        ("dense `Matmul` (all in0 layouts)", "matmul"),
        ("**collectives (`AllGather` / `ReduceScatter`)**", "ccl"),
        ("all data movement (`DM` category)", "dm"),
    ]
    out = [
        "| | linear_attention decode | full_attention decode | linear_attention prefill | full_attention prefill |",
        "|---|---|---|---|---|",
    ]
    for label, key in labels:
        cells = [f"{cols[i][key]:.2f}%" for i in order]
        if key == "ccl":
            cells = [f"**{c}**" for c in cells]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("| DRAM roofline (modeled ops) | " + " | ".join(roof[i] for i in order) + " |")
    return "\n".join(out)


def _dense():
    rows, repeats = collections.defaultdict(list), collections.defaultdict(list)
    for line in read(LOGS / "probe_dense_matmul.txt").splitlines():
        p = line.split()
        if len(p) >= 11 and p[0] == "DENSE" and p[5] == "mcast1d" and p[9][0].isdigit():
            rows[p[1]].append((float(p[9]), float(p[10]), int(p[6]), int(p[7])))
            repeats[(p[1], int(p[6]), int(p[7]))].append(float(p[9]))
    widest = {}
    for (role, _, _), values in repeats.items():
        if len(values) > 1:
            widest[role] = max(widest.get(role, 0.0), max(values) - min(values))
    return rows, widest


def table_category() -> str:
    """Every ``Op Category`` of the merged 4-device window, against the single-chip stage's own.

    Added after review round 4, which pointed out that section 5.4 reported six *op-code* rows summing
    to about 55% of the decode window and called it "top of the stack", while the ``TM`` (layout)
    category alone was a fifth of that window and appeared nowhere. The single-chip column is the
    control that says whether a category is this stage's doing: it is the same profile, same phases,
    from ``doc/optimized_decoder/tracy/``.
    """
    import csv
    import io

    def shares(root, kind, phase):
        body = read(root / kind / f"{phase}_perf_report_stacked.csv.gz")
        totals: dict = {}
        for row in csv.DictReader(io.StringIO(body)):
            totals[row["Op Category"]] = totals.get(row["Op Category"], 0.0) + float(row["Total % [%]"])
        return totals

    single = TRACY.parent.parent / "optimized_decoder" / "tracy"
    cols = [
        ("linear_attention", "decode"),
        ("full_attention", "decode"),
        ("linear_attention", "prefill"),
        ("full_attention", "prefill"),
    ]
    mine = [shares(TRACY, k, p) for k, p in cols]
    theirs = [shares(single, k, p) for k, p in cols]
    out = [
        "| `Op Category` | linear decode | full decode | linear prefill | full prefill |",
        "|---|---|---|---|---|",
    ]
    for cat, label in (
        ("Compute", "`Compute`"),
        ("TM", "`TM` (layout)"),
        ("DM", "`DM` (data movement)"),
        ("Other", "`Other`"),
    ):
        cells = [f"{m.get(cat, 0.0):.1f}% ({t.get(cat, 0.0):.1f}%)" for m, t in zip(mine, theirs)]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_tm() -> str:
    """The layout rows that make up the decode ``TM`` share, largest first."""
    import csv
    import io

    out = ["| op | linear decode | full decode |", "|---|---|---|"]
    rows = {}
    for i, kind in enumerate(("linear_attention", "full_attention")):
        body = read(TRACY / kind / "decode_perf_report_stacked.csv.gz")
        for row in csv.DictReader(io.StringIO(body)):
            if row["Op Category"] != "TM":
                continue
            name = row["Op Code"].split("DeviceOperation")[0]
            cell = rows.setdefault(name, [0.0, 0.0])
            cell[i] += float(row["Total % [%]"])
    for name, (a, b) in sorted(rows.items(), key=lambda kv: -sum(kv[1]))[:8]:
        out.append(f"| `{name}` | {a:.2f}% | {b:.2f}% |")
    return "\n".join(out)


#: Which logged PCC belongs to which inventory, by the exact phrasing each test prints. Explicit
#: rather than inferred: review round 4 found section 4.2's hand-written breakdown 4 values short and
#: misattributed after round 3 added two assertions to an existing test, and a "count every PCC in
#: the log" rule would sweep in the agreement, weight-partition and self-consistency comparisons,
#: which are different claims with different bars.
PCC_INVENTORIES = {
    "golden": [
        ("test_prefill_pcc", r"PCC=([01]\.\d+)"),
        ("test_decode_pcc", r"PCC=([01]\.\d+)"),
        ("test_batched_prefill_decode_pcc", r"PCC=([01]\.\d+)"),
        ("test_batched_decode_ragged_positions", r"PCC=([01]\.\d+)"),
        ("test_long_context_pcc", r"(?:prefill|decode) ([01]\.\d+)"),
        ("test_unaligned_max_context", r"PCC=([01]\.\d+)"),
        ("test_traced_decode_pcc", r"PCC=([01]\.\d+)"),
        ("test_permuted_page_table", r"PCC=([01]\.\d+)"),
        ("test_prefill_continuation", r"PCC vs HF golden=([01]\.\d+)"),
    ],
    "baseline": [
        ("test_prefill_matches_single_chip", r"PCC=([01]\.\d+)"),
        ("test_decode_matches_single_chip", r"PCC=([01]\.\d+)"),
    ],
}


def _pcc_rows(which):
    body = read(LOGS / "pytest_full_suite.txt.gz")
    lines = body.splitlines()
    counts, values = {}, {}
    for test, pattern in PCC_INVENTORIES[which]:
        hits = []
        for line in lines:
            if f"test_multichip_decoder:{test}:" not in line:
                continue
            hits += [float(v) for v in re.findall(pattern, line)]
        if hits:
            counts[test], values[test] = len(hits), min(hits)
    return counts, values


def table_pcc_inventory() -> str:
    """Every HF-golden PCC the committed suite log prints, counted by the test that printed it."""
    counts, values = _pcc_rows("golden")
    out = ["| test | values | minimum |", "|---|---|---|"]
    for name in sorted(counts, key=lambda n: (-counts[n], n)):
        out.append(f"| `{name}` | {counts[name]} | {values[name]:.6f} |")
    out.append(f"| **total** | **{sum(counts.values())}** | **{min(values.values()):.6f}** |")
    return "\n".join(out)


def table_pcc_baseline() -> str:
    """The same, for the primary bar: this stage against the single-chip TTNN decoder in-process."""
    counts, values = _pcc_rows("baseline")
    out = ["| test | values | minimum |", "|---|---|---|"]
    for name in sorted(counts, key=lambda n: (-counts[n], n)):
        out.append(f"| `{name}` | {counts[name]} | {values[name]:.6f} |")
    out.append(f"| **total** | **{sum(counts.values())}** | **{min(values.values()):.6f}** |")
    return "\n".join(out)


def table_dense() -> str:
    #: Inherited entry per role, from `optimized_decoder.DECODE_MATMUL_GEOMETRY`, and this stage's.
    inherited = {
        "attn_in": (32, 2),
        "o_proj": (16, 16),
        "gdn_in": (110, 2),
        "gdn_out": (24, 8),
        "shared_in": (32, 32),
        "shared_down": (48, 16),
        "router": (32, 32),
    }
    shipped = {"attn_in": (110, 8), "gdn_in": (110, 8), "shared_down": (8, 4), "expert_select": (8, 8)}
    k_dim = {
        "attn_in": 2048,
        "o_proj": 1024,
        "gdn_in": 2048,
        "gdn_out": 1024,
        "shared_in": 2048,
        "shared_down": 128,
        "router": 2048,
        "expert_select": 256,
    }
    rows, widest = _dense()

    def realise(cores, grid_x=11):
        cols = min(grid_x, cores)
        return cols * math.ceil(cores / cols)

    out = [
        "| role | inherited (realised) | us | local winner | us | delta | winner spread | repeatability | shipped |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for role in ("attn_in", "gdn_in", "shared_down", "expert_select", "o_proj", "gdn_out", "shared_in", "router"):
        ladder = sorted(rows[role])
        best = ladder[0]
        inh_us, inh_label = None, "new role"
        if role in inherited:
            cores, cap = inherited[role]
            rc, rw = realise(cores), min(cap, k_dim[role] // 32)
            hits = [x[0] for x in ladder if x[2] == rc and x[3] == rw]
            inh_us = min(hits) if hits else None
            inh_label = f"({cores},{cap}) → {rc}/{rw}"
        ship = "inherited"
        if role in shipped:
            cores, cap = shipped[role]
            hits = [x[0] for x in ladder if x[2] == realise(cores) and x[3] == min(cap, k_dim[role] // 32)]
            ship = f"**{'retuned ' if role in inherited else ''}{shipped[role]}**"
            if hits:
                ship += f", {min(hits):.2f}"
        delta = f"{inh_us - best[0]:.2f}" if inh_us is not None else "—"
        out.append(
            f"| `{role}` | {inh_label} | {f'{inh_us:.2f}' if inh_us is not None else '—'} | "
            f"{best[2]}/{best[3]} | {best[0]:.2f} | {delta} | {best[1]:.2f} | {widest.get(role, 0):.2f} | {ship} |"
        )
    return "\n".join(out)


def table_sparse_ladder() -> str:
    by = collections.defaultdict(dict)
    for line in read(LOGS / "probe_sparse_matmul_local.txt").splitlines():
        if not line.startswith("SPARSEL "):
            continue
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if "us" not in fields:
            continue
        cores = int(fields["cores"].split("(")[0])
        key = (int(fields["active"]), fields["role"])
        by[key][cores] = min(by[key].get(cores, 1e9), float(fields["us"]))
    counts = (4, 8, 16, 32, 64)
    out = ["| active | " + " | ".join(f"{c} cores" for c in counts) + " | winner |", "|---" * 7 + "|"]
    for key in sorted(by):
        win = min(by[key], key=lambda c: by[key][c])
        cells = []
        for c in counts:
            if c not in by[key]:
                cells.append("—")
            else:
                cells.append(f"**{by[key][c]:.1f}**" if c == win else f"{by[key][c]:.1f}")
        out.append(f"| {key[0]}, `{key[1]}` | " + " | ".join(cells) + f" | {win} |")
    return "\n".join(out)


def table_decode_batch() -> str:
    arms, shapes = collections.defaultdict(dict), {}
    for line in read(LOGS / "probe_decode_batch.txt").splitlines():
        p = line.split()
        if p and p[0] == "DECODEB":
            arms[(p[2], int(p[3]))].setdefault(p[4], []).append(float(p[6]))
        if p and p[0] == "SHAPE" and "site=mixer" in line:
            shapes[(p[1], int(p[2].split("=")[1]))] = int(
                next(x for x in p if x.startswith("physical_rows")).split("=")[1]
            )
    batches = sorted({b for _, b in arms})
    out = ["| batch | mixer physical rows | off | on | delta |", "|---|---|---|---|---|"]
    for batch in batches:
        cells = {}
        for kind in ("full_attention", "linear_attention"):
            off, on = min(arms[(kind, batch)]["off"]), min(arms[(kind, batch)]["on"])
            cells[kind] = (off, on, 1000 * (off - on))
        f, l = cells["full_attention"], cells["linear_attention"]
        delta = f"{f[2]:+.0f} / {l[2]:+.0f} us"
        if f[2] < 0 or abs(f[2]) > 50:
            delta = f"**{delta}**"
        out.append(
            f"| {batch} | {shapes[('full_attention', batch)]} | {f[0]:.3f} / {l[0]:.3f} | "
            f"{f[1]:.3f} / {l[1]:.3f} | {delta} |"
        )
    return "\n".join(out)


def table_sparse_batch() -> str:
    arms = collections.defaultdict(dict)
    for line in read(LOGS / "probe_decode_batch.txt").splitlines():
        p = line.split()
        if p and p[0] == "SPARSEB":
            arms[(p[2], int(p[3]))].setdefault(p[4], []).append(float(p[6]))
    batches = sorted({b for _, b in arms})
    out = ["| batch | inherited rule | shipped rule | delta |", "|---|---|---|---|"]
    for batch in batches:
        cells = {}
        for kind in ("full_attention", "linear_attention"):
            off, on = min(arms[(kind, batch)]["off"]), min(arms[(kind, batch)]["on"])
            cells[kind] = (off, on, 1000 * (off - on))
        f, l = cells["full_attention"], cells["linear_attention"]
        out.append(f"| {batch} | {f[0]:.3f} / {l[0]:.3f} | {f[1]:.3f} / {l[1]:.3f} | {f[2]:+.0f} / {l[2]:+.0f} us |")
    return "\n".join(out)


def table_fused() -> str:
    rows = collections.defaultdict(dict)
    for line in read(LOGS / "probe_fused_ccl.txt").splitlines():
        p = line.split()
        if len(p) >= 6 and p[0] == "FUSED" and p[4] == "trace":
            rows[(p[1], p[2])][p[3]] = float(p[5])
    out = ["| boundary | arm | decode | decode batch 32 | prefill 2048 |", "|---|---|---|---|---|"]
    plan = [
        ("o_proj", "mm_then_all_reduce", "matmul + `all_reduce`"),
        ("o_proj", "mm_then_stack_sum", "matmul + `stack_sum` (shipped at the decode tile)"),
        ("o_proj", "fused_mm_rs_then_ag", "**fused** matmul+reduce-scatter, then all-gather"),
        ("o_proj", "fused_mm_rs_only", "**fused**, no gather — the sharded-residual bound"),
        ("attn_in", "mm_replicated", "matmul on a replicated input (shipped: no collective)"),
        ("attn_in", "ag_then_mm", "all-gather + matmul — what a sharded residual would pay"),
    ]
    for boundary, arm, label in plan:
        cells = [
            f"{rows[(boundary, shape)].get(arm):.2f}" if rows[(boundary, shape)].get(arm) else "—"
            for shape in ("decode", "decode_b32", "prefill_2048")
        ]
        out.append(f"| `{boundary}` | {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


TABLES = {
    "bench": table_bench,
    "ccl": table_ccl,
    "moepar": table_moepar,
    "fabric": table_fabric,
    "packet": table_packet,
    "ablayer": table_ablayer,
    "perf": table_perf,
    "category": table_category,
    "tm": table_tm,
    "pcc_inventory": table_pcc_inventory,
    "pcc_baseline": table_pcc_baseline,
    "dense": table_dense,
    "sparse_ladder": table_sparse_ladder,
    "decode_batch": table_decode_batch,
    "sparse_batch": table_sparse_batch,
    "fused": table_fused,
}


def splice(text: str, seen: dict) -> tuple[str, list]:
    """Fill every table marker in ``text``; ``seen`` accumulates which markers were actually found.

    A table whose markers were deleted -- replaced by a hand-written copy, which is exactly the drift
    these markers exist to prevent -- used to pass silently, because `found` was computed and never
    read (review round 5).
    """
    stale = []
    for name, build in TABLES.items():
        pattern = re.compile(rf"(<!-- TABLE:{name} -->\n)(.*?)(\n<!-- /TABLE:{name} -->)", re.S)
        body = build()
        found = False

        def swap(match):
            nonlocal found
            found = True
            if match.group(2) != body:
                stale.append(name)
            return match.group(1) + body + match.group(3)

        text = pattern.sub(swap, text)
        seen[name] = seen.get(name, False) or found
    return text, stale


def main() -> int:
    check = "--check" in sys.argv[1:]
    problems = []
    seen: dict = {}
    for doc in (DOC / "README.md", DOC / "work_log.md"):
        text = doc.read_text()
        new, stale = splice(text, seen)
        if stale:
            problems += [f"STALE-TABLE  {doc.name}: {name}" for name in stale]
        if not check and new != text:
            doc.write_text(new)
    # A generator with no marker anywhere means someone replaced a generated table with a hand-written
    # copy, which is the drift these markers exist to prevent.
    problems += [f"UNPLACED-TABLE  {name} has no <!-- TABLE:{name} --> marker" for name in TABLES if not seen.get(name)]
    for problem in problems:
        print(problem)
    if check:
        print(f"{len(problems)} stale table(s)")
        return 1 if problems else 0
    print(f"regenerated {len(TABLES)} table kinds in README.md and work_log.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

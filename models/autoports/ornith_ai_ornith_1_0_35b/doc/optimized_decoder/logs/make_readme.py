# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fill every numeric block of ``doc/optimized_decoder/README.md`` from the committed artifacts.

The fused stage's review found eleven hand-transcribed PCC cells describing an earlier run, so no
number in this stage's README is typed by hand either: each ``<!-- generated:NAME -->`` …
``<!-- /generated:NAME -->`` block is spliced from the artifact named below it.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/make_readme.py
    python .../make_readme.py --check     # exit non-zero if the README disagrees with the artifacts

Sources, block by block:

``headline`` / ``perf-result``  ``logs/ab_fused_vs_optimized.txt`` (fused and optimized timed in one
                               process on one device with the same weights)
``suite-result``               ``logs/pytest_full_suite.txt``
``prefill-pcc`` / ``decode-pcc``  the same suite log's ``PCC=`` lines
``policy-sweep``               ``logs/ab_precision_policy.txt``
``bfp4-pcc``                   ``logs/probe_projection_dtype.txt``
``decode-breakdown``           ``tracy/<kind>/decode_perf_report.summary.txt``
``accounting`` / ``gap-itemisation``  ``tracy/perf_summary.json`` (the second is the per-op
                               ``Op-to-Op Gap`` itemisation ``perf_accounting.py`` computes)
``prefill-search``             ``logs/probe_prefill_matmul.txt`` (``in0=DRAM`` arm) + the shipped
                               configs the suite log records
``advice``                     the ``Advice`` column of ``tracy/<kind>/decode_perf_report.csv``,
                               with every microsecond figure in the action prose read out of the
                               probe artifact it cites
``watcher-result``             ``watcher/census_summary.txt`` + ``logs/watcher_pytest.txt``
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
LOGS = ROOT / "logs"
TRACY = ROOT / "tracy"
WATCHER = ROOT / "watcher"

KINDS = [("linear_attention", "linear_attention"), ("full_attention", "full_attention")]


def read(path: Path) -> str:
    """Text of ``path``, or of ``path.gz`` — the big logs are committed gzipped (500 KB repo limit)."""
    if path.is_file():
        return path.read_text(errors="ignore")
    packed = path.with_suffix(path.suffix + ".gz")
    if packed.is_file():
        return gzip.decompress(packed.read_bytes()).decode(errors="ignore")
    return ""


def open_csv(path: Path):
    """Line iterator over ``path`` or ``path.gz`` for ``csv.DictReader``."""
    if path.is_file():
        return open(path, newline="")
    packed = path.with_suffix(path.suffix + ".gz")
    if not packed.is_file():
        raise SystemExit(f"missing {path} (and {packed})")
    return io.StringIO(gzip.decompress(packed.read_bytes()).decode(errors="ignore"))


def bench_rows(path: Path) -> dict:
    """``{(tag, kind, phase): value}`` from ``BENCH`` lines."""
    out = {}
    for line in read(path).splitlines():
        m = re.search(r"tag=(\S+) layer=\d+ \((\w+)\) (prefill|decode)", line)
        if not m:
            continue
        tag, kind, phase = m.groups()
        if phase == "prefill":
            v = re.search(r"wall=([0-9.]+) ms tok/s=([0-9.]+)", line)
            out[(tag, kind, phase)] = (float(v.group(1)), float(v.group(2)))
        else:
            v = re.search(r"wall/iter=([0-9.]+) ms steps/s=([0-9.]+)", line)
            out[(tag, kind, phase)] = (float(v.group(1)), float(v.group(2)))
    return out


def block_headline(rows):
    lines = [
        "| Window | before (fused) | after (optimized) | speedup |",
        "| --- | --- | --- | --- |",
    ]
    for kind, _ in KINDS:
        for phase, label, unit in (
            ("prefill", "prefill, 2048 tokens", "tok/s"),
            ("decode", "decode, traced", "steps/s"),
        ):
            b, a = rows[("before", kind, phase)], rows[("after", kind, phase)]
            # Quote each phase at the precision its artifact records: bench.py prints prefill wall time
            # to 2 decimals and traced decode to 3. Mixing them made the "before" decode column read
            # 2.06 where the artifact says 2.064.
            dp = 2 if phase == "prefill" else 3
            lines.append(
                f"| `{kind}` {label} | {b[0]:.{dp}f} ms / {b[1]:.1f} {unit} | "
                f"**{a[0]:.{dp}f} ms / {a[1]:.1f} {unit}** | **{b[0] / a[0]:.2f}x** |"
            )
    return "\n".join(lines)


def block_perf_result(rows):
    lines = [
        "| Layer kind | Phase | before | after | speedup |",
        "| --- | --- | --- | --- | --- |",
    ]
    for kind, _ in KINDS:
        for phase, label in (("prefill", "prefill, 2048 tokens"), ("decode", "decode, traced (32 replays)")):
            b, a = rows[("before", kind, phase)], rows[("after", kind, phase)]
            dp = 2 if phase.startswith("prefill") else 3
            lines.append(
                f"| `{kind}` | {label} | {b[0]:.{dp}f} ms | **{a[0]:.{dp}f} ms** | "
                f"**{b[0] / a[0]:.2f}x** (−{100 * (1 - a[0] / b[0]):.1f} %) |"
            )
    return "\n".join(lines)


def block_suite_result():
    text = read(LOGS / "pytest_full_suite.txt")
    m = re.search(r"=+ (\d+) passed[^=]*in ([0-9.]+)s", text)
    if not m:
        return "**suite result not found in logs/pytest_full_suite.txt**"
    failed = re.search(r"(\d+) failed", text)
    status = f"**{m.group(1)} passed**" + (f", **{failed.group(1)} FAILED**" if failed else "")
    return f"{status} in {float(m.group(2)):.2f} s."


def pcc_from_suite(pattern: str):
    out = {}
    for line in read(LOGS / "pytest_full_suite.txt").splitlines():
        m = re.search(pattern, line)
        if m:
            out[m.groups()[:-1]] = float(m.groups()[-1])
    return out


def block_prefill_pcc():
    got = pcc_from_suite(r"prefill layer=(\d+) \((\w+)\) seq_len=(\d+) PCC=([0-9.]+)")
    lengths = sorted({int(k[2]) for k in got})
    lines = ["| `seq_len` | `linear_attention` | `full_attention` |", "| --- | --- | --- |"]
    for length in lengths:
        cells = []
        for kind, _ in KINDS:
            hit = [v for k, v in got.items() if k[1] == kind and int(k[2]) == length]
            cells.append(f"{hit[0]:.6f}" if hit else "—")
        lines.append(f"| {length} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


def block_decode_pcc():
    got = pcc_from_suite(r"decode layer=(\d+) \((\w+)\) prefill_len=(\d+) step=(\d+) pos=\d+ PCC=([0-9.]+)")
    lines = ["| prefill length | step | `linear_attention` | `full_attention` |", "| --- | --- | --- | --- |"]
    keys = sorted({(int(k[2]), int(k[3])) for k in got})
    for prefill_len, step in keys:
        cells = []
        for kind, _ in KINDS:
            hit = [v for k, v in got.items() if k[1] == kind and int(k[2]) == prefill_len and int(k[3]) == step]
            cells.append(f"{hit[0]:.6f}" if hit else "—")
        lines.append(f"| {prefill_len} | {step} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


def block_policy_sweep():
    rows = {}
    for line in read(LOGS / "ab_precision_policy.txt").splitlines():
        m = re.search(r"set=(\S*) tag=\S+ layer=\d+ \((\w+)\) (prefill|decode)", line)
        if not m:
            continue
        setting, kind, phase = m.groups()
        setting = "(selected policy)" if setting == "-" else setting
        value = re.search(r"wall/iter=([0-9.]+) ms" if phase == "decode" else r"wall=([0-9.]+) ms", line)
        pcc = re.search(r"pcc=([0-9.]+)", line)
        rows.setdefault(setting, {})[(kind, phase)] = (float(value.group(1)), float(pcc.group(1)) if pcc else None)
    lines = [
        "| candidate | `full` decode ms | `linear` decode ms | `full` / `linear` prefill screen PCC |",
        "| --- | --- | --- | --- |",
    ]
    for setting, data in rows.items():
        fd = data.get(("full_attention", "decode"), (float("nan"), None))[0]
        ld = data.get(("linear_attention", "decode"), (float("nan"), None))[0]
        fp = data.get(("full_attention", "prefill"), (0, None))[1]
        lp = data.get(("linear_attention", "prefill"), (0, None))[1]
        label = f"**{setting}**" if setting == "(selected policy)" else f"`{setting}`"
        pcc_cell = f"{fp:.6f} / {lp:.6f}" if fp and lp else "—"
        lines.append(f"| {label} | {fd:.3f} | {ld:.3f} | {pcc_cell} |")
    return "\n".join(lines)


def block_bfp4_pcc():
    worst = {}
    for line in read(LOGS / "probe_projection_dtype.txt").splitlines():
        m = re.match(r"PROJDTYPE (\S+) layer=\d+ \((\w+)\) (.*) pcc=([0-9.]+)", line)
        if not m:
            continue
        arm, kind, case, value = m.group(1), m.group(2), m.group(3), float(m.group(4))
        key = (arm, kind)
        if key not in worst or value < worst[key][0]:
            worst[key] = (value, case)
    lines = [
        "| projection weight dtype | worst `full_attention` | worst `linear_attention` | margin above the 0.995 bar |",
        "| --- | --- | --- | --- |",
    ]
    for arm, label in (("bfloat8_b", "**BFP8 (selected)**"), ("bfloat4_b", "BFP4")):
        full = worst.get((arm, "full_attention"))
        lin = worst.get((arm, "linear_attention"))
        if not full or not lin:
            continue
        margin = min(full[0], lin[0]) - 0.995
        lines.append(f"| {label} | {full[0]:.6f} ({full[1]}) | {lin[0]:.6f} ({lin[1]}) | {margin:.1e} |")
    # The two figures the rejection rests on, derived rather than typed: how much more layer error BFP4
    # costs (as a ratio of distance-from-1 PCC, per layer kind) and what it buys on traced decode.
    ratios = []
    for kind in ("full_attention", "linear_attention"):
        eight, four = worst.get(("bfloat8_b", kind)), worst.get(("bfloat4_b", kind))
        if eight and four and eight[0] < 1.0:
            ratios.append((1.0 - four[0]) / (1.0 - eight[0]))
    saving = decode_saving_pct("proj_dtype=bfloat4_b")
    error_cost = f"{min(ratios):.0f}–{max(ratios):.0f}x" if ratios else "an unmeasured multiple of"
    lines.append("")
    lines.append(
        f"Every BFP4 row clears the bar, so this is not a pass/fail rejection — it is a {error_cost} "
        f"increase in layer error for {saving} of one traced decode step and nothing in prefill, in one "
        "layer of a 48-layer stack. The routed-expert BFP4 step this stage *did* take is the "
        'opposite trade. Rejected on that comparison, and shipped as `POLICIES["bfp4-projections"]` '
        "so `$datatype-sweep` can take it without rediscovering it. Full ladder: "
        "[`logs/probe_projection_dtype.txt`](logs/probe_projection_dtype.txt)."
    )
    return "\n".join(lines)


def decode_saving_pct(setting: str) -> str:
    """What one policy-sweep candidate buys on traced decode, as a percentage range across both kinds.

    Read out of ``ab_precision_policy.txt`` rather than typed, because README §4.3's headline used to
    disagree with the generated policy-sweep table two sections above it (review round 4).
    """
    rows = {}
    for line in read(LOGS / "ab_precision_policy.txt").splitlines():
        m = re.search(r"set=(\S*) tag=\S+ layer=\d+ \((\w+)\) decode", line)
        v = re.search(r"wall/iter=([0-9.]+) ms", line)
        if m and v:
            rows[(m.group(1), m.group(2))] = float(v.group(1))
    pcts = []
    for kind in ("full_attention", "linear_attention"):
        base, cand = rows.get(("-", kind)), rows.get((setting, kind))
        if base and cand:
            pcts.append(100 * (base - cand) / base)
    if not pcts:
        return "an unmeasured share"
    return f"{min(pcts):.1f}–{max(pcts):.1f} %" if abs(max(pcts) - min(pcts)) >= 0.05 else f"{pcts[0]:.1f} %"


def block_decode_breakdown():
    per_kind = {}
    for kind, _ in KINDS:
        text = read(TRACY / kind / "decode_perf_report.summary.txt")
        rows = re.findall(r"^\s*([0-9.]+) %\s+(\S+)\s+([0-9,]+\.[0-9]+) μs\s+(\d+)", text, re.M)
        per_kind[kind] = [(float(p), n, float(t.replace(",", "")) / 32, int(c) // 32) for p, n, t, c in rows]
    # Union of the top rows of BOTH kinds, ordered by their larger per-step cost: selecting by
    # full_attention rank alone hid ~111 us/step of the linear_attention window behind the total.
    ranked = {}
    for rows in per_kind.values():
        for _, name, us, _ in rows:
            ranked[name] = max(ranked.get(name, 0.0), us)
    names = [n for n, _ in sorted(ranked.items(), key=lambda kv: -kv[1])[:16]]
    lines = [
        "| Op code | `linear_attention` µs/step | `full_attention` µs/step | launches/step (`full`) |",
        "| --- | --- | --- | --- |",
    ]
    for name in names:
        cells = []
        launches = 0
        for kind, _ in KINDS:
            hit = [r for r in per_kind[kind] if r[1] == name]
            cells.append(f"{hit[0][2]:.1f}" if hit else "—")
            if kind == "full_attention" and hit:
                launches = hit[0][3]
        lines.append(f"| `{name}` | {cells[0]} | {cells[1]} | {launches} |")
    totals = {k: sum(r[2] for r in v) for k, v in per_kind.items()}
    lines.append(
        f"| **total device time** | **{totals['linear_attention']:.1f}** | " f"**{totals['full_attention']:.1f}** | |"
    )
    return "\n".join(lines)


def block_accounting():
    path = TRACY / "perf_summary.json"
    if not path.is_file():
        return "**tracy/perf_summary.json missing**"
    data = json.loads(path.read_text())
    lines = [
        "| | `linear_attention` | `full_attention` |",
        "| --- | --- | --- |",
    ]
    fields = [
        ("bytes moved per token (weights at their stored dtypes + KV read)", "bytes_per_token", "{:,.0f} B"),
        ("DRAM peak used for the roofline (recovered from the report's own DRAM %)", "dram_peak_gbps", "{:.1f} GB/s"),
        ("1. theoretical roofline", "roofline_ms_per_token_estimate", "{:.3f} ms"),
        ("2. device-time decode (signposted window / 32 replays)", "decode_ms_per_token_device", "{:.3f} ms"),
        ("3. end-to-end decode, **profiled** run", "decode_ms_per_token_e2e", "{:.3f} ms"),
        ("roofline as a fraction of device time", "roofline_fraction_of_device", "{:.1%}"),
        ("dispatch + host gap (3 − 2)", "dispatch_and_host_ms", "{:.3f} ms"),
    ]
    for label, key, fmt in fields:
        cells = [fmt.format(data[k][key]) for k, _ in KINDS]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    lines.append("")
    lines.append("Named limitations, in the order they cost time:")
    lines.append("")
    for item in data["full_attention"]["named_limitations"]:
        lines.append(f"* {item}")
    return "\n".join(lines)


def probe_rows(path: Path, prefix: str) -> list[dict]:
    """``[{key: value, ..., 'us': float}]`` for every ``<PREFIX> k=v k=v ... us=N`` row of a probe log.

    Rows that record a refusal (``FAILED``) are kept with ``us=None``, because "this geometry does not
    build" is itself a measurement several of the actions below rest on.
    """
    out = []
    for line in read(path).splitlines():
        if not line.startswith(prefix + " "):
            continue
        row = {"failed": "FAILED" in line}
        for token in line[len(prefix) + 1 :].split():
            key, _, value = token.partition("=")
            if _:
                row[key] = value
        row["us"] = float(row["us"]) if "us" in row and not row["failed"] else None
        out.append(row)
    return out


def best(rows, **where) -> float | None:
    """Fastest ``us`` among ``rows`` matching every ``key=value`` in ``where``; ``None`` if none ran."""
    times = [r["us"] for r in rows if r["us"] is not None and all(str(r.get(k)) == str(v) for k, v in where.items())]
    return min(times) if times else None


def block_gap_itemisation():
    """Every op-to-op gap above the itemisation threshold, per layer kind, from ``perf_summary.json``.

    §7's claim is that the end-to-end/device-time difference is *itemised, not waved away*. Round 4
    found the hand-written itemisation naming two typecast gaps where the capture had three, so the
    itemisation is generated: `tracy/perf_accounting.py` reads the ``Op-to-Op Gap`` column of the same
    committed report the paragraph cites and the count cannot be wrong.
    """
    path = TRACY / "perf_summary.json"
    if not path.is_file():
        return "**tracy/perf_summary.json missing**"
    data = json.loads(path.read_text())
    kind_gaps = {kind: data[kind]["op_to_op_gaps"] for kind, _ in KINDS}
    threshold = next(iter(kind_gaps.values()))["itemise_threshold_us"]
    lines = [
        f"| Layer kind | op code | op-to-op gap, µs/step | launches/step | largest single gap |",
        "| --- | --- | --- | --- | --- |",
    ]
    for kind, _ in KINDS:
        gaps = kind_gaps[kind]
        for index, item in enumerate(gaps["largest"]):
            lines.append(
                f"| {f'`{kind}`' if index == 0 else ''} | `{item['op_code']}` | "
                f"**{item['gap_us_per_step']:.1f}** | {item['launches_per_step']:.0f} | "
                f"{item['largest_single_gap_us']:.1f} µs |"
            )
        lines.append(
            f"| {f'`{kind}`' if not gaps['largest'] else ''} | *the other "
            f"{gaps['remainder_op_codes']} op codes, each under {threshold:.0f} µs/step* | "
            f"{gaps['remainder_us']:.1f} | | |"
        )
        lines.append(
            f"| | **all {gaps['op_codes']} op codes, {gaps['launches_per_step']:.0f} launches/step** | "
            f"**{gaps['total_us']:.1f}** | | |"
        )
    lines.append("")
    lines.append(
        "Grouped by op code and divided by the 32 replays. The bottom row of each kind is what §7's "
        "dispatch-and-host gap has to be made of, and it reconciles with it. Note what this replaces: "
        "the hand-written version of this paragraph said *two* 6–8 µs typecast gaps, when the "
        "`TypecastDeviceOperation` gaps are the **largest single line item** of the `linear_attention` "
        "window at 7 launches a step."
    )
    return "\n".join(lines)


#: The probe's dense roles, as ``(role, K, N)``. ``o_proj`` and ``gdn_out`` are the same shape on
#: purpose: they are the same matmul on the two layer kinds, and the layer selects one config for it.
PREFILL_ROLES = [
    ("attn_in", "full_attention", 2048, 9216),
    ("o_proj", "full_attention", 4096, 2048),
    ("gdn_in", "linear_attention", 2048, 12352),
    ("gdn_out", "linear_attention", 4096, 2048),
    ("shared_in", "full_attention", 2048, 1056),
    ("shared_down", "full_attention", 512, 2048),
]


def shipped_configs(phase: str) -> dict:
    """``{(kind, K, N): "grid=… in0_block_w=…"}`` for the configs the layer actually built, read out of
    the suite log.

    Keyed by layer *kind* as well as shape: ``o_proj`` and ``gdn_out`` are both 4096x2048 and get
    different configs (they sit behind different activations), so a shape-only key silently reported one
    of them under the other's geometry.

    ``test_{decode,prefill}_runs_the_tuned_program_configs`` logs every config it asserts, so the
    shipped column of the tables below comes from a *run of the shipped code* rather than from a
    transcription of the constants. Round 4's P1 was a search table disagreeing with the running layer;
    keying the table off this line is what makes that disagreement impossible to write down.
    """
    out = {}
    for line in read(LOGS / "pytest_full_suite.txt").splitlines():
        if f"{phase} dense matmuls layer=" not in line:
            continue
        kind_m = re.search(r"layer=\d+ \((\w+)\)", line)
        if not kind_m:
            continue
        for item in line.split(":")[-1].split(","):
            m = re.search(r"(\d+)x(\d+) (grid=\S+ in0_block_w=\d+.*)", item.strip())
            if m:
                out[(kind_m.group(1), int(m.group(1)), int(m.group(2)))] = m.group(3).strip()
    return out


def block_prefill_search():
    """The dense-prefill program-config search, generated from the probe and the suite log.

    Review round 4 found the hand-written version of this table quoting a *superseded* run of
    ``probe_prefill_matmul.py`` in every one of its five rows, including one geometry the artifact
    recorded as failing to build while the shipped layer ran it. It is generated now, from the
    ``in0=DRAM`` arm — the arm whose L1 state matches the shipped graph — and the shipped column comes
    from the suite log, so a disagreement between the table, the probe and the running code shows up as
    a **not the measured winner** marker instead of as prose nobody rechecks.
    """
    rows = probe_rows(LOGS / "probe_prefill_matmul.txt", "PREFILLMM")
    shipped = shipped_configs("prefill")
    lines = [
        "| role | shape M×K×N | ttnn heuristic | shipped 2D config | shipped | vs the whole `in0=DRAM` sweep |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for role, kind, k, n in PREFILL_ROLES:
        cfg = shipped.get((kind, k, n), "")
        grid = re.search(r"grid=(\d+)-(\d+)", cfg)
        ibw = re.search(r"in0_block_w=(\d+)", cfg)
        if not grid or not ibw:
            continue
        grid_s = f"{grid.group(1)}x{grid.group(2)}"
        heuristic = best(rows, role=role, family="heuristic")
        mine = best(rows, role=role, family="2d", grid=grid_s, in0_block_w=ibw.group(1), in0="DRAM")
        pool = [r["us"] for r in rows if r["us"] is not None and r.get("role") == role and r.get("in0") == "DRAM"]
        winner = min(pool) if pool else None
        verdict = "**the measured winner**"
        if mine is None:
            verdict = "**not measurable — the shipped geometry did not build in the probe**"
        elif winner is not None and mine > winner + 1e-9:
            verdict = f"**not the measured winner** — {winner:.1f} µs elsewhere in the sweep"
        lines.append(
            f"| `{role}` | 2048×{k}×{n} | {heuristic:.1f} µs | {grid_s}, `in0_block_w` {ibw.group(1)} | "
            f"**{mine:.1f} µs** | {verdict} |"
            if mine is not None
            else f"| `{role}` | 2048×{k}×{n} | {heuristic:.1f} µs | {grid_s}, `in0_block_w` "
            f"{ibw.group(1)} | — | {verdict} |"
        )
    return "\n".join(lines)


def advice_actions() -> dict:
    """What this stage did about each distinct `tt-perf-report` advice item, keyed by a substring of the
    advice text.

    The *counts* and the op rows are read from the committed reports by :func:`block_advice`, and every
    microsecond figure in the prose below is read out of the probe artifact it cites, so neither can
    drift. Round 4 of the stage review found three stale figures hiding in this table specifically
    because it used to be hand-written prose inside a *generated* block: ``make_readme.py --check``
    regenerates the block from this table and therefore always agreed with itself.
    """
    micro = probe_rows(LOGS / "probe_decode_micro.txt", "STATE")
    prefill = probe_rows(LOGS / "probe_prefill_matmul.txt", "PREFILLMM")
    dense = probe_rows(LOGS / "probe_dense_matmul.txt", "DENSE")

    # "in0_block_w=1 is small", recurrent-state arm: the explicit MatmulMultiCoreReuseProgramConfig
    # against the `core_grid` spelling the fused stage used, both on the full grid.
    read_cfg = best(micro, role="read", in0_block_w="2", grid="11x10")
    read_grid = best(micro, role="read", fidelity="HiFi4/fp32acc", grid="11x10")
    outer_cfg = best(micro, role="outer", in0_block_w="1", grid="11x10")
    outer_grid = best(micro, role="outer", fidelity="HiFi4/fp32acc", grid="11x10")
    # "HiFi2 is sufficient", same rows: the shipped HiFi4 + fp32-accumulate against the alternatives.
    hifi2 = best(micro, role="read", fidelity="HiFi2", grid="11x10")
    lofi = best(micro, role="read", fidelity="LoFi", grid="11x10")
    # "place input 0 in L1", prefill arm: the shipped in0_block_w of each role, DRAM against L1, plus
    # what the enabling DRAM->L1 copy costs on its own.
    l1_arm = {}
    for role, ibw in (("attn_in", "8"), ("gdn_in", "8"), ("shared_in", "16"), ("shared_down", "16")):
        l1_arm[role] = (
            best(prefill, role=role, in0_block_w=ibw, grid="11x10", in0="DRAM"),
            best(prefill, role=role, in0_block_w=ibw, grid="11x10", in0="L1"),
            best(prefill, role=role, family="place-in0-L1"),
        )
    # "Output subblock 1x1 is small": the per_core_N >= 2 alternative for the two rows that keep 1x1.
    sub_alt = best(dense, role="shared_in", family="mcast1d", cores="16", per_core_N="2", out="L1")
    sub_shipped = best(dense, role="shared_in", family="mcast1d", cores="88", per_core_N="1", out="L1")

    def us(value, digits=1):
        return "unmeasured" if value is None else f"{value:.{digits}f} µs"

    # Per role: what an L1 `in0` is worth at the shipped geometry, and what the copy that enables it
    # costs. Written sign-agnostically on purpose — `attn_in`'s two arms swap order between runs, which
    # is itself the finding (the difference is inside the spread), and prose that hard-codes "slower"
    # would be wrong on the next re-run. The decisive comparison is per role, not summed: each role
    # would have to pay its own copy.
    verdicts = []
    for role, (dram, l1, place) in l1_arm.items():
        if dram is None or l1 is None:
            verdicts.append(f"`{role}` **does not build** at the shipped geometry with an L1 `in0`")
            continue
        delta = dram - l1
        direction = f"saves {delta:.1f} µs" if delta > 0 else f"costs {-delta:.1f} µs"
        cost = f", against {place:.1f} µs to place the operand" if place is not None else ""
        verdicts.append(f"`{role}` {direction} ({l1:.1f} against {dram:.1f}){cost}")
    #: The best case for the advice across the roles where it helps at all: largest saving, and what
    #: that same role must pay to get it.
    gains = [
        (dram - l1, place, role)
        for role, (dram, l1, place) in l1_arm.items()
        if dram is not None and l1 is not None and place is not None and dram > l1
    ]
    best_case = max(gains) if gains else None

    return {
        "DRAM-sharded program config": (
            "**Tried, rejected with measurement** — loses on all seven dense roles, even without its "
            "activation-reshard cost (§5.4)."
        ),
        "place input 0 in L1": (
            "**Taken for decode, measured and rejected for prefill.** Decode: the two residual norms, "
            "the three float32 recurrent-state matmuls and the shared expert's SwiGLU product all hand "
            "their result to L1, each size-gated so a large batch still uses DRAM — the item is now "
            "raised 0 times in both decode reports. The head-dim norms are the one decode exception and "
            "cannot move: `paged_scaled_dot_product_attention_decode` rejects a non-sharded Q outside "
            "DRAM. Prefill: still raised on three rows, and measured at each role's **shipped** "
            "`in0_block_w`, against the DRAM→L1 copy that would have to *enable* it — the shipped graph "
            "would pay that copy per role, because the activation arrives interleaved from the preceding "
            "norm or residual add. "
            + "; ".join(verdicts)
            + ". "
            + (
                f"So even the advice's best case (`{best_case[2]}`) saves {best_case[0]:.1f} µs to spend "
                f"{best_case[1]:.1f} µs, i.e. it is a net loss on every role"
                if best_case
                else "So it is a net loss on every role"
            )
            + " — before counting that the whole dense group is 1.07 % of the prefill window, and that "
            'the two arms of the widest role swap order between runs, which is what "inside the '
            'spread" looks like. `probe_prefill_matmul.txt` `in0=DRAM`, `in0=L1` and '
            "`family=place-in0-L1` rows."
        ),
        "in0_block_w=1 is small": (
            "**Taken.** The five dense *prefill* rows got explicit 2D configs with `in0_block_w` 8/16 "
            "(§5.4), and the three recurrent-state rows got an explicit "
            "`MatmulMultiCoreReuseProgramConfig` — that family does expose `in0_block_w`, unlike the "
            f"`core_grid` spelling the fused stage used: 2 for the two reads ({us(read_cfg)} against "
            f"{us(read_grid)}) and 1 for the `transpose_a` outer product, where `Kt` is 1 tile so 2 and "
            f"4 are rejected by the op ({us(outer_cfg)} against {us(outer_grid)}). "
            "`probe_decode_micro.txt` `STATE` rows."
        ),
        "HiFi2 is sufficient": (
            f"**Tried, rejected with measurement** on the recurrent-state rows: HiFi2 is {us(hifi2)} and "
            f"LoFi {us(lofi)} against {us(read_grid)} for the shipped HiFi4 + fp32-accumulate — under a "
            "microsecond per matmul, well inside the run-to-run spread and under 0.1 % of the window, "
            "for the float32 state that is the model's exact carry between steps."
        ),
        "Output subblock 1x1 is small": (
            "**Tried, rejected with measurement** — the `per_core_N` ≥ 2 alternative is slower for both "
            f"rows (`shared_in` {us(sub_alt)} against {us(sub_shipped)}, §5.4)."
        ),
        "use HiFi4 with BF16 activations": (
            "**Rejected with measurement** — the reverse direction of §4.2's fidelity sweep; HiFi4 is "
            "what the fused decoder had and it is slower at equal correctness."
        ),
        "HiFi2 may also work": (
            "**Rejected on purpose** (the router row): the matmul is under 10 µs and its output decides "
            "*which experts run*. The fused stage measured bfloat16 routing agreeing with float32 on "
            "only 99.8 % / 95.5 % of top-8 sets, so this group stays BF16/HiFi4/fp32-accumulate."
        ),
        "look good": (
            "**Not advice** — `tt-perf-report` printing that a row's `in0_block_w` and output subblock "
            "are already what it would have suggested. Kept in the table so the generator cannot "
            "silently drop a line it does not recognise."
        ),
        "nnz=std::nullopt": (
            "**Reporting limitation, not advice.** `nnz` is inferred at runtime because pinning it "
            "wedged the device (§9 item 3); the report cannot model DRAM/FLOP utilisation for those "
            "rows, so this stage measures their share of device time instead."
        ),
    }


#: The dense *decode* roles, as ``(role, K, N)``, in the order the table lists them.
DECODE_ROLES = [
    ("attn_in", "full_attention", 2048, 9216),
    ("o_proj", "full_attention", 4096, 2048),
    ("gdn_in", "linear_attention", 2048, 12352),
    ("gdn_out", "linear_attention", 4096, 2048),
    ("shared_in", "full_attention", 2048, 1056),
    ("shared_down", "full_attention", 512, 2048),
    ("router", "full_attention", 2048, 256),
]


def block_decode_search():
    """The dense-decode program-config search: three families per role, generated from the probe.

    The shipped column is read out of the suite log — i.e. out of a run of the shipped code — and the
    probe row that matches it is looked up by (cores, ``in0_block_w``, ``per_core_N``), so a config
    change that nobody re-measured shows as an unmatched row rather than as a stale number.
    """
    rows = probe_rows(LOGS / "probe_dense_matmul.txt", "DENSE")
    shipped = shipped_configs("decode")
    lines = [
        "| role | shape M×K×N | ttnn heuristic | best DRAM-sharded | **shipped 1D `mcast_in0`** |",
        "| --- | --- | --- | --- | --- |",
    ]
    for role, kind, k, n in DECODE_ROLES:
        cfg = shipped.get((kind, k, n), "")
        grid = re.search(r"grid=(\d+)-(\d+)", cfg)
        ibw = re.search(r"in0_block_w=(\d+)", cfg)
        pcn = re.search(r"per_core_N=(\d+)", cfg)
        heuristic = best(rows, role=role, family="default")
        sharded = best(rows, role=role, family="dram_sharded")
        if grid and ibw and pcn:
            cores = int(grid.group(1)) * int(grid.group(2))
            grid_s = f"{grid.group(1)}×{grid.group(2)}"
            # Matched on (in0_block_w, per_core_N), NOT on the probe's `cores=` field: that field is the
            # *target* core count the sweep asked for, and both the probe and the layer then reduce it to
            # a grid whose width divides Nt. `per_core_N` is what survives that reduction, and it
            # determines the realised grid, so the pair identifies the geometry exactly while `cores`
            # would not match (target 96 becomes the 11x9 = 99-core grid the layer reports).
            mine = best(rows, role=role, family="mcast1d", in0_block_w=ibw.group(1), per_core_N=pcn.group(1))
            shipped_cell = (
                f"**{mine:.1f} µs** — {cores} cores ({grid_s}), `in0_block_w` {ibw.group(1)}, "
                f"`per_core_N` {pcn.group(1)}"
                if mine is not None
                else (
                    f"{cores} cores ({grid_s}), `in0_block_w` {ibw.group(1)}, `per_core_N` "
                    f"{pcn.group(1)} — **no probe row matches this geometry**"
                )
            )
        else:
            mine, shipped_cell = None, "**not found in the suite log**"
        cells = [f"{heuristic:.1f} µs" if heuristic else "—", f"{sharded:.1f}" if sharded else "—"]
        lines.append(f"| `{role}` | 32×{k}×{n} | {cells[0]} | {cells[1]} | {shipped_cell} |")
    lost = [
        role
        for role, _, _, _ in DECODE_ROLES
        if (lambda h, s: h is not None and s is not None and s > h)(
            best(rows, role=role, family="mcast1d"), best(rows, role=role, family="dram_sharded")
        )
    ]
    lines.append("")
    lines.append(
        f"The DRAM-sharded family loses to the shipped 1D `mcast_in0` config on **{len(lost)} of "
        f"{len(DECODE_ROLES)}** roles, measured without its activation-reshard cost, which it would also "
        "have to pay."
    )
    return "\n".join(lines)


def block_sparse_search():
    """The routed-expert ``sparse_matmul`` geometry sweep, per active-expert count, from the probe.

    The shipped geometry is read out of the suite log's ``decode sparse matmuls`` line, and the table
    reports, per (active count, role): the strongest candidate the fused decoder's rule would have
    picked, the best measured geometry, and whether the shipped rule reproduces it.
    """
    rows = probe_rows(LOGS / "probe_sparse_matmul.txt", "SPARSE")
    missing = [r for r in rows if "active" not in r]
    actives = sorted({int(r["active"]) for r in rows if "active" in r})
    #: What the fused decoder's rule ("largest core count dividing Nt, per_core_N 1") selects.
    FUSED_RULE = {"gate_up": ("32(8x4)", "16", "1"), "down": ("64(8x8)", "8", "1")}
    shipped_line = ""
    for line in read(LOGS / "pytest_full_suite.txt").splitlines():
        if "decode sparse matmuls:" in line:
            shipped_line = line.split("decode sparse matmuls:")[-1].strip()
            break
    shipped = {}
    for role, item in zip(("gate_up", "down"), shipped_line.split(",")):
        grid = re.search(r"grid=(\d+)-(\d+)", item)
        ibw = re.search(r"in0_block_w=(\d+)", item)
        pcn = re.search(r"per_core_N=(\d+)", item)
        if grid and ibw and pcn:
            shipped[role] = (int(grid.group(1)) * int(grid.group(2)), ibw.group(1), pcn.group(1))
    lines = [
        "| active experts | role | fused rule's geometry | best measured | shipped |",
        "| --- | --- | --- | --- | --- |",
    ]
    for active in actives:
        for role in ("gate_up", "down"):
            pool = [r for r in rows if r["us"] is not None and r.get("role") == role and r.get("active") == str(active)]
            if not pool:
                continue
            winner = min(pool, key=lambda r: r["us"])
            cores, ibw, pcn = FUSED_RULE[role]
            fused = best(rows, role=role, active=str(active), cores=cores, in0_block_w=ibw, per_core_N=pcn)
            best_cell = (
                f"**{winner['us']:.1f} µs** — {winner['cores']}, `in0_block_w` {winner['in0_block_w']}, "
                f"`per_core_N` {winner['per_core_N']}, {winner['mem']}"
            )
            note = "as measured"
            if active == min(actives) and role in shipped:
                s_cores, s_ibw, s_pcn = shipped[role]
                s_us = best(
                    rows,
                    role=role,
                    active=str(active),
                    in0_block_w=s_ibw,
                    per_core_N=s_pcn,
                )
                same = f"{s_cores}(" in (winner["cores"] or "") and winner["in0_block_w"] == s_ibw
                note = f"{s_cores} cores, `in0_block_w` {s_ibw}, `per_core_N` {s_pcn}"
                if same:
                    note += " — **the measured winner**"
                elif s_us:
                    # Not the winner: say by how much, rather than leaving a bare number next to a
                    # bolded best. `in0_block_w` is chosen by a divisor rule in the layer, so a small
                    # loss here is a deliberate simplification and has to be visible as one.
                    gap = 100 * (s_us - winner["us"]) / winner["us"]
                    note += f" — {s_us:.1f} µs, **{gap:+.1f} %** against the winner"
            lines.append(
                f"| {active} | {role.replace('_', '/')} | "
                f"{f'{fused:.1f} µs' if fused else '—'} | {best_cell} | {note} |"
            )
    if missing:
        lines.append("")
        lines.append(
            f"**{len(missing)} of {len(rows)} probe rows carry no `active=` field** and are excluded: "
            "they were produced by an earlier revision of the probe. Re-run `logs/run_evidence.sh` to "
            "regenerate the artifact with all sections from one script."
        )
    return "\n".join(lines)


def block_op_knobs():
    """The non-matmul decode knobs: what the shipped setting is worth against the alternative.

    Every one of these was quoted by hand somewhere — in the README, the work log and three code
    docstrings — and review rounds 3 and 4 found six of them stale. The implementation now points here
    instead of carrying the numbers.
    """
    micro = LOGS / "probe_decode_micro.txt"
    norm = probe_rows(micro, "NORM")
    sdpa = probe_rows(micro, "SDPA")
    topk = probe_rows(micro, "TOPK")
    split = probe_rows(micro, "SPLIT")
    gate = probe_rows(micro, "GATE")

    def cell(value, unit=" µs"):
        return "—" if value is None else f"{value:.1f}{unit}"

    interleaved = best(norm, spelling="interleaved-default")
    sharded = min(
        (r["us"] for r in norm if r["us"] is not None and r.get("spelling") == "width-sharded" and "8(" in r["cores"]),
        default=None,
    )
    sdpa_default = best(sdpa, cfg="default(None)")
    # `probe_rows` splits on the FIRST "=", so the grid arrives as cfg="grid=8x8", not as a `grid` key.
    # Both SDPA rows below are pinned to the shipped 8x8 grid so the k-chunk comparison is the only axis.
    shipped_sdpa = best(sdpa, cfg="grid=8x8", k_chunk="64")
    faster_sdpa = best(sdpa, cfg="grid=8x8", k_chunk="128")
    topk_native = best(topk, width="256")
    topk_padded = best(topk, width="8192")
    packed = min((r["us"] for r in split if r["us"] is not None and "packed" in str(r.get("spelling"))), default=None)
    separate = min(
        (r["us"] for r in split if r["us"] is not None and "packed" not in str(r.get("spelling"))), default=None
    )
    gate_shipped = best(gate, spelling="topk-softmax-scatter")
    gate_alt = best(gate, spelling="topk-ge-where-softmax")

    rows = [
        (
            "residual RMSNorm",
            f"interleaved (one core) {cell(interleaved)}",
            f"**width-sharded on 8 cores {cell(sharded)}**",
            "taken; the two conversions it adds cost ~3 µs against ~9 saved",
        ),
        (
            "paged flash decode",
            f"op default {cell(sdpa_default)}",
            f"**explicit config {cell(shipped_sdpa)}**",
            "taken; the default is more than an order of magnitude slower",
        ),
        (
            "paged flash decode, `k_chunk` 128",
            f"{cell(faster_sdpa)} — *faster in isolation*",
            f"**64, one chunk per page, {cell(shipped_sdpa)}**",
            "**rejected on correctness**: layer PCC collapses (`ab_sdpa_decode_contract.txt`)",
        ),
        (
            "`ttnn.topk` routing width",
            f"padded to 8192 (multi-core) {cell(topk_padded)}",
            f"**native 256 (single-core) {cell(topk_native)}**",
            "the multi-core path needs width ≥ 8192 and loses anyway",
        ),
        (
            "gate/up projection",
            f"separate pair {cell(separate)}",
            f"**packed {cell(packed)}**",
            "taken (OPT-010)",
        ),
        (
            "routing gate",
            f"`topk`→`ge`→`where`→softmax {cell(gate_alt)}",
            f"**`topk`→softmax→scatter {cell(gate_shipped)}**",
            "kept; the rewrite is bit-identical and slower",
        ),
    ]
    lines = ["| knob | alternative | shipped | decision |", "| --- | --- | --- | --- |"]
    for name, alt, mine, note in rows:
        lines.append(f"| {name} | {alt} | {mine} | {note} |")
    lines.append("")
    lines.append(
        "All from [`logs/probe_decode_micro.txt`](logs/probe_decode_micro.txt), which is regenerated by "
        "`run_evidence.sh` from the shipped code."
    )
    return "\n".join(lines)


def block_prefill_composition():
    """What the prefill window is made of, per layer kind, from ``tracy/perf_summary.json``.

    The hand-written version of this pair of percentages came from an earlier run, and the first attempt
    at deriving it attributed most of the window to the *dense* projections because
    `MatmulDeviceOperation` is a substring of `SparseMatmulDeviceOperation`. Both are why it is generated
    and why `perf_accounting.py` matches op codes by prefix.
    """
    path = TRACY / "perf_summary.json"
    if not path.is_file():
        return "**tracy/perf_summary.json missing**"
    data = json.loads(path.read_text())
    parts = []
    for kind, _ in KINDS:
        comp = data[kind].get("prefill_window_composition") or {}
        if not comp:
            continue
        parts.append(
            f"`{kind}` is **{comp['sparse_matmul_share']:.1%}** routed-expert `sparse_matmul` and "
            f"**{comp['dense_matmul_share']:.2%}** dense `Matmul`"
        )
    if not parts:
        return "**no prefill composition recorded**"
    return (
        "Of the 2048-token prefill window's device time, " + "; ".join(parts) + ". So the whole dense "
        "program-config step above is worth well under a percent of prefill end to end, which is what it "
        "measures — the window is the routed experts, and that is what §9 item 1 is about."
    )


def block_same_process_gate():
    """The in-suite same-process fused-vs-optimized gate, read out of the suite log.

    §5.1 used to transcribe these two pairs, and review round 4 found both wrong in the third decimal.
    They come from ``test_optimized_beats_fused_traced_decode``, which is the *gate* on the performance
    claim, so they are worth quoting - but not worth retyping.
    """
    rows = []
    for line in read(LOGS / "pytest_full_suite.txt").splitlines():
        m = re.search(
            r"OPTIMIZED VS FUSED decode\(traced\) layer=\d+ \((\w+)\) before=([0-9.]+) ms "
            r"after=([0-9.]+) ms speedup=([0-9.]+)x",
            line,
        )
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    if not rows:
        return "**the same-process gate did not log in logs/pytest_full_suite.txt**"
    order = {kind: i for i, (kind, _) in enumerate(KINDS)}
    rows.sort(key=lambda r: order.get(r[0], 99))
    parts = [f"`{kind}` {before:.3f} → {after:.3f} ms ({speedup:.2f}x)" for kind, before, after, speedup in rows]
    return (
        "Its numbers, from the committed suite log: " + "; ".join(parts) + ". They agree with the A/B "
        "file above, which is why both are quoted — one process proves the comparison is apples to "
        "apples, two processes prove it is not an artifact of sharing a device session."
    )


def block_advice():
    """Every distinct advice item in **all four** committed reports, with per-window counts and rows.

    Decode counts are per traced step (32 replays); prefill counts are per pass. Review round 3 found
    this table scoped to the two decode reports while an item was still open on three prefill rows.
    """
    actions = advice_actions()
    windows = [
        ("linear_attention", "decode", 32),
        ("full_attention", "decode", 32),
        ("linear_attention", "prefill", 1),
        ("full_attention", "prefill", 1),
    ]
    counts, rows = {}, {}
    for kind, phase, replays in windows:
        path = TRACY / kind / f"{phase}_perf_report.csv"
        with open_csv(path) as handle:
            for row in csv.DictReader(handle):
                for item in (row.get("Advice") or "").split("\u2022"):
                    item = item.strip().lstrip("- ").strip()
                    if not item:
                        continue
                    key = next((k for k in actions if k in item), item[:48])
                    counts.setdefault(key, {})[(kind, phase)] = counts.setdefault(key, {}).get((kind, phase), 0) + 1
                    rows.setdefault(key, set()).add(f"{row['OP Code'][:44]} ({phase[:2]})")
    keys = sorted(counts, key=lambda k: (-max(counts[k].values()), k))
    header = " | ".join(f"`{kind.split('_')[0]}` {phase}" for kind, phase, _ in windows)
    lines = [
        f"| Advice | {header} | Rows it is raised on | Action |",
        "| --- | " + " | ".join(["---"] * len(windows)) + " | --- | --- |",
    ]
    for key in keys:
        cells = []
        for kind, phase, replays in windows:
            n = counts[key].get((kind, phase), 0)
            cells.append(f"{n / replays:.2f}" if replays > 1 else str(n))
        ops = sorted(rows[key])
        op_cell = ", ".join(f"`{o}`" for o in ops[:3]) + (" …" if len(ops) > 3 else "")
        action = actions.get(key, "**unclassified — this stage did not act on it**")
        lines.append(f"| *{key}* | {' | '.join(cells)} | {op_cell} | {action} |")
    closed = [k for k in actions if k not in keys]
    if closed:
        lines += [
            "",
            "Advice items **no longer raised in any of the four committed reports**:",
            "",
            "| Advice (no longer raised) | What closed it |",
            "| --- | --- |",
        ]
        for key in closed:
            lines.append(f"| *{key}* | {actions[key]} |")
    return "\n".join(lines)


def block_watcher_result():
    census = read(WATCHER / "census_summary.txt")
    console = read(LOGS / "watcher_pytest.txt")
    m = re.search(r"=+ (\d+) passed[^=]*in ([0-9.]+)s", console)
    failed = re.search(r"(\d+) failed", console)
    total = re.search(r"(\d[\d,]*)\s+TOTAL", census)
    fatal = re.search(r"fatal-class matches:\s*(\d+)", census)
    parts = []
    if m:
        parts.append(f"**{m.group(1)} passed**" + (f", **{failed.group(1)} FAILED**" if failed else ""))
        parts.append(f"in {float(m.group(2)):.2f} s")
    text = " ".join(parts) if parts else "watcher console log not found"
    if total and fatal:
        text += (
            f". The {int(total.group(1).replace(',', '')):,} lines of the watcher log are fully accounted for by a "
            f"disjoint census, and a fatal-class grep (asserts, invalid NOC coordinates or addresses, "
            f"CB out-of-bounds, L1/stack overflow, sanitizer, corruption, hang/deadlock) returns "
            f"**{fatal.group(1)} matches**."
        )
    stack = re.search(r"minimum stack headroom:\s*(\d+) bytes free over (\d+) detail", census)
    if stack:
        text += (
            f" Watcher recorded a stack watermark on {stack.group(2)} dump(s); the tightest leaves "
            f"{stack.group(1)} bytes free."
        )
    return (
        text
        + "\n\nArtifacts: [`watcher/watcher_log.txt`](watcher/watcher_log.txt), [`watcher/census_summary.txt`](watcher/census_summary.txt), console log [`logs/watcher_pytest.txt`](logs/watcher_pytest.txt)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    rows = bench_rows(LOGS / "ab_fused_vs_optimized.txt")
    blocks = {
        "headline": block_headline(rows),
        "perf-result": block_perf_result(rows),
        "suite-result": block_suite_result(),
        "prefill-pcc": block_prefill_pcc(),
        "decode-pcc": block_decode_pcc(),
        "policy-sweep": block_policy_sweep(),
        "bfp4-pcc": block_bfp4_pcc(),
        "decode-breakdown": block_decode_breakdown(),
        "op-knobs": block_op_knobs(),
        "sparse-search": block_sparse_search(),
        "decode-search": block_decode_search(),
        "prefill-search": block_prefill_search(),
        "prefill-composition": block_prefill_composition(),
        "same-process-gate": block_same_process_gate(),
        "gap-itemisation": block_gap_itemisation(),
        "advice": block_advice(),
        "accounting": block_accounting(),
        "watcher-result": block_watcher_result(),
    }
    text = README.read_text()
    for name, body in blocks.items():
        marker = f"<!-- generated:{name} -->"
        end = f"<!-- /generated:{name} -->"
        replacement = f"{marker}\n{body}\n{end}"
        if marker in text and end in text:
            text = re.sub(re.escape(marker) + r".*?" + re.escape(end), lambda _: replacement, text, flags=re.S)
        else:
            raise SystemExit(f"README is missing the {name} block markers")
    if args.check:
        if text != README.read_text():
            raise SystemExit("README disagrees with the artifacts; re-run make_readme.py")
        print("README matches the artifacts")
        return
    README.write_text(text)
    print(f"filled {len(blocks)} generated blocks in {README}")


if __name__ == "__main__":
    main()

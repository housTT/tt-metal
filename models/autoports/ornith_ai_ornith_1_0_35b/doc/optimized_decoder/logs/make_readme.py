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
WORK_LOG = ROOT / "work_log.md"
LOGS = ROOT / "logs"
TRACY = ROOT / "tracy"
WATCHER = ROOT / "watcher"

KINDS = [("linear_attention", "linear_attention"), ("full_attention", "full_attention")]

#: Mirrors ``tt/optimized_decoder.SPARSE_MIN_CORES``: the routed core count at or below which a call is the
#: narrow (batch-1 decode) geometry. ``audit_figures.check_mirrored_constants`` compares the two, so this
#: cannot drift silently — which is the class of bug that put a decode `in0_block_w` on a prefill row.
SPARSE_MIN_CORES_MIRROR = 8

#: Model shapes the routed matmuls produce, mirroring ``tt/optimized_decoder._sparse_n_tiles``: gate/up is
#: ``2 * moe_intermediate`` wide and down is ``dim`` wide, in tiles. ``audit_figures`` compares these against
#: the implementation.
SPARSE_N_TILES_MIRROR = {"gate_up": 2 * 512 // 32, "down": 2048 // 32}

#: Traced decode replays inside one signposted profiler window. The captures are per-window, so every µs/step
#: figure derived from them divides by this.
TRACED_REPLAYS = 32


def model_fact(name: str) -> int:
    """One shape constant, read out of `logs/model_facts.txt` rather than typed.

    Round 9 found this generator's prose stating the wrong layer count — 48, where the checkpoint has 40 — in a
    *generated* block — so the block was reproducible and wrong together. Reading the artifact means the
    generator cannot hold a stale opinion about the checkpoint's shape.
    """
    text = (LOGS / "model_facts.txt").read_text()
    match = re.search(rf"^FACT {name}=(\d+)$", text, re.M)
    if not match:
        raise SystemExit(f"logs/model_facts.txt has no FACT {name}= row; run logs/model_facts.py")
    return int(match.group(1))


def _sparse_n_tiles_mirror(role: str) -> int:
    return SPARSE_N_TILES_MIRROR[role]


def _largest_divisor_at_most(value: int, cap: int) -> int:
    """Largest divisor of ``value`` no greater than ``cap`` — the layer's own reduction, mirrored."""
    for candidate in range(min(cap, value), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


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
            # to 2 decimals and traced decode to 3. Mixing them truncated the "before" decode column by a
            # digit, so it disagreed with the artifact it is generated from.
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
        f"layer of a {model_fact('num_hidden_layers')}-layer stack. The routed-expert BFP4 step this stage "
        "*did* take is the "
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
    # The listed rows are the top 16 by max-across-kinds, so each column hides a different remainder - and on
    # `full_attention` the hidden set contained its second and third largest attention-side ops while smaller
    # rows were kept, because the ranking is shared between the two columns. Review round 11 found the table
    # printing an all-codes total over a truncated body with no threshold disclosed; §7's gap table already
    # did this correctly. The remainder is a row of its own now, so the column sums reconcile.
    listed = {kind: sum(r[2] for r in rows if r[1] in set(names)) for kind, rows in per_kind.items()}
    hidden_counts = {kind: sum(1 for r in rows if r[1] not in set(names)) for kind, rows in per_kind.items()}
    lines.append(
        f"| *the other {hidden_counts['linear_attention']} / {hidden_counts['full_attention']} op codes "
        f"(`linear` / `full`), each outside the top 16 by the larger of its two per-step costs* | "
        f"{totals['linear_attention'] - listed['linear_attention']:.1f} | "
        f"{totals['full_attention'] - listed['full_attention']:.1f} | |"
    )
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
    lines.append(
        "Named limitations, in the order they cost time. Stated **per layer kind**, because the two windows "
        "differ — review rounds 2 and 6 both found this list rendering only the `full_attention` figures, "
        "unlabelled, under a table that is per-kind, and under-reporting the kind with the *larger* "
        "dispatch gap."
    )
    lines.append("")
    for kind, _ in KINDS:
        lines.append("")
        lines.append(f"`{kind}`:")
        lines.append("")
        for item in data[kind]["named_limitations"]:
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


def best(rows, *, absent=(), **where) -> float | None:
    """Fastest ``us`` among ``rows`` matching every ``key=value`` in ``where``; ``None`` if none ran.

    ``where`` matches a *subset* of each row's fields, so a row carrying extra labelled fields still matches -
    which is how a probe arm added later can silently become the answer to an earlier question. ``absent``
    names fields the row must NOT carry, so "the shipped arm" means the one with no extra knob on it. The
    under-constrained-lookup defect rounds 5-10 kept finding in `block_sparse_search` was this shape; the
    round-11 `max_cores_per_head_batch` arms are faster than the shipped row, so without ``absent`` they
    would have become it.
    """
    times = [
        r["us"]
        for r in rows
        if r["us"] is not None
        and all(str(r.get(k)) == str(v) for k, v in where.items())
        and not any(r.get(k) is not None for k in absent)
    ]
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


def prefill_dense_share() -> str:
    """The dense-`Matmul` share of the prefill window, as a range across the layer kinds.

    Read from ``tracy/perf_summary.json``, which ``perf_accounting.py`` computes by matching op codes by
    prefix. Round 5 found this quoted as a hand-typed value here - from an earlier run, wrong by a third,
    inside the action prose of a *generated* block, which is exactly the place round 4 was supposed to
    have emptied of hand-typed figures.
    """
    path = TRACY / "perf_summary.json"
    if not path.is_file():
        return "an unmeasured share"
    data = json.loads(path.read_text())
    shares = [
        data[kind]["prefill_window_composition"]["dense_matmul_share"]
        for kind, _ in KINDS
        if (data.get(kind) or {}).get("prefill_window_composition")
    ]
    if not shares:
        return "an unmeasured share"
    return f"{min(shares):.2%}" if max(shares) - min(shares) < 5e-5 else f"{min(shares):.2%}-{max(shares):.2%}"


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
    # The two rows that keep a 1x1 output subblock, and the per_core_N >= 2 alternative for each. Looked
    # up by the *realised* grid, and reported as "not expressible" rather than "unmeasured" where the
    # sweep has no such candidate: `router` has Nt = 8, so a per_core_N >= 2 config would need <= 4 cores
    # and the ladder starts at 8. Round 5 found the previous lookup keyed on a core count the probe never
    # emits, which rendered the shipped side of the comparison as the word "unmeasured".
    def subblock_pair(role, shipped_cores):
        shipped_us = min(
            (
                r["us"]
                for r in dense
                if r["us"] is not None
                and r.get("role") == role
                and r.get("family") == "mcast1d"
                and r.get("out") == "L1"
                and r.get("per_core_N") == "1"
                and r.get("cores")
                and 11 * -(-int(r["cores"]) // 11) == shipped_cores
            ),
            default=None,
        )
        alt = min(
            (
                r["us"]
                for r in dense
                if r["us"] is not None
                and r.get("role") == role
                and r.get("family") == "mcast1d"
                and r.get("out") == "L1"
                and r.get("per_core_N") not in (None, "1")
            ),
            default=None,
        )
        return shipped_us, alt

    sub_shipped, sub_alt = subblock_pair("shared_in", 88)
    router_shipped, router_alt = subblock_pair("router", 33)

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
    dense_share = prefill_dense_share()
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
            + f" — before counting that the whole dense group is {dense_share} of the prefill window, and that "
            'the two arms of the widest role swap order between runs, which is what "inside the '
            'spread" looks like. `probe_prefill_matmul.txt` `in0=DRAM`, `in0=L1` and '
            "`family=place-in0-L1` rows."
        ),
        "in0_block_w=1 is small": (
            "**Taken.** Every dense *prefill* projection got an explicit 2D config with `in0_block_w` 8/16 "
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
            "**Tried on the row where it is expressible, rejected with measurement.** `shared_in`: the "
            f"`per_core_N` ≥ 2 alternative measures {us(sub_alt)} against {us(sub_shipped)} for the "
            "shipped `per_core_N` 1 at the same output placement — slower, because at 32 rows of M these "
            "are bandwidth-bound rather than block-bound. `router`: "
            + (
                f"{us(router_alt)} against {us(router_shipped)}."
                if router_alt is not None
                else f"**not expressible** — `Nt` is 8, so `per_core_N` ≥ 2 needs ≤ 4 cores and the "
                f"sweep's core ladder starts at 8; the shipped row is {us(router_shipped)}."
            )
        ),
        "use HiFi4 with BF16 activations": (
            "**Rejected with measurement** — the reverse direction of §4.2's fidelity sweep; HiFi4 is "
            "what the fused decoder had and it is slower at equal correctness."
        ),
        "HiFi2 may also work": (
            "**Rejected on purpose** (the router row): the matmul is under 10 µs and its output decides "
            "*which experts run*. The preceding stages measured bfloat16 routing agreeing with float32 on "
            "only 99.8 % / 95.5 % of top-8 sets "
            "(`doc/functional_decoder/logs/router_precision_ab.txt`), so this group stays "
            "BF16/HiFi4/fp32-accumulate."
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
    #: Where the shipped decode projections put their output. `_decode_1d_matmul_config`'s callers hand
    #: the dense decode results to L1, so the L1 arm of the sweep is the comparable one.
    shipped_out = "L1"
    lines = [
        "| role | shape M×K×N | ttnn heuristic | best DRAM-sharded | **shipped 1D `mcast_in0`** | "
        "vs the whole 1D sweep |",
        "| --- | --- | --- | --- | --- | --- |",
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

            # Match the geometry the layer actually runs. The probe's `cores=` field is the *target* it
            # asked for; both the probe and the layer then round it up to a full 11-wide grid, so the
            # realised core count is `11 * ceil(target / 11)` — target 80 and 96 both become the shipped
            # 11x8 = 88 and 11x9 = 99. Review round 5 found the first version matching on
            # (in0_block_w, per_core_N) alone on the theory that `per_core_N` pins the grid: it does not,
            # because for a narrow output every target from 24 to 110 yields `per_core_N` 1, so three rows
            # reported a time measured on a different grid, and two on a DRAM output while the layer
            # ships L1. Both filters are explicit now.
            def realised(row):
                """The core count the probe's *target* actually becomes, mirroring `mcast1d_config`.

                A target of 11 or fewer builds a single row of that width; above 11 it fills whole 11-wide
                rows. `11 * ceil(target/11)` alone reports 11 for a target of 8, which is wrong — no shipped
                role uses a sub-11-core grid today, so this was latent, but the two rules must agree.
                """
                try:
                    target = int(row.get("cores"))
                except (TypeError, ValueError):
                    return None
                return target if target <= 11 else 11 * -(-target // 11)

            candidates = [
                r
                for r in rows
                if r["us"] is not None
                and r.get("role") == role
                and r.get("family") == "mcast1d"
                and r.get("in0_block_w") == ibw.group(1)
                and r.get("per_core_N") == pcn.group(1)
                and r.get("out") == shipped_out
                and realised(r) == cores
            ]
            mine = min((r["us"] for r in candidates), default=None)
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
        # Re-check the shipped row against every 1D candidate for this role at the shipped output
        # placement, so "the sweep chose this grid" is asserted on regeneration. Round 5 found README
        # prose claiming "every smaller grid was measured too and lost" while a smaller grid was faster.
        # The pool keeps whole rows, not just times: where the shipped row is behind, the reader needs the
        # alternative's *geometry* to judge whether it is shippable, and round 9 found this naming only its
        # time. `cores`/`in0_block_w`/`per_core_N` come straight off the winning probe row.
        pool = [
            r
            for r in rows
            if r["us"] is not None
            and r.get("role") == role
            and r.get("family") == "mcast1d"
            and r.get("out") == shipped_out
        ]
        winner_row = min(pool, key=lambda r: r["us"]) if pool else None
        winner = winner_row["us"] if winner_row is not None else None
        # Per-row spread, matching block_sparse_search: the maximum over the whole role sweep was up to
        # 6 us on some roles, which let a genuine sub-microsecond gap read as "inside the spread".
        winner_spread = float(winner_row.get("spread") or 0) if winner_row is not None else 0.0
        winner_geom = (
            f"{winner_row.get('cores')} cores, `in0_block_w` {winner_row.get('in0_block_w')}, "
            f"`per_core_N` {winner_row.get('per_core_N')}"
            if winner_row is not None
            else "—"
        )
        mine_spread = (
            max(
                (float(r.get("spread") or 0) for r in candidates if r["us"] == mine),
                default=0.0,
            )
            if mine is not None
            else 0.0
        )
        spread = max(winner_spread, mine_spread)
        if mine is None or winner is None:
            verdict = "—"
        elif mine <= winner + 1e-9:
            verdict = "**the measured winner**"
        elif round(mine - winner, 1) <= round(spread, 1):
            verdict = (
                f"+{mine - winner:.1f} µs against {winner:.1f} ({winner_geom}), inside the ±{spread:.1f} µs "
                f"row spread"
            )
        else:
            # Name the share of a decode step too: at these shapes a "beyond the spread" gap can still be a
            # few hundredths of a percent, and the reader should not have to divide to find that out.
            share = f", {100 * (mine - winner) / max(mine, 1e-9):.2f} % of this op"
            verdict = (
                f"**+{mine - winner:.1f} µs** against {winner:.1f} ({winner_geom}), beyond the "
                f"±{spread:.1f} µs row spread{share}"
            )
        cells = [f"{heuristic:.1f} µs" if heuristic else "—", f"{sharded:.1f}" if sharded else "—"]
        lines.append(f"| `{role}` | 32×{k}×{n} | {cells[0]} | {cells[1]} | {shipped_cell} | {verdict} |")
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
        "have to pay. The last column compares each shipped row against every 1D candidate for that role at "
        "the shipped output placement, using that row's own measured spread rather than the sweep's widest — "
        "review round 8 pointed out the looser rule could hide a real sub-microsecond gap. Where a row is "
        "behind, the gap and the alternative are named: none of them is worth a core-count special case at "
        "these shapes, and the ladder is flat enough there that the neighbouring targets sit between the two."
    )
    return "\n".join(lines)


def block_sparse_search():
    """The routed-expert ``sparse_matmul`` geometry sweep, per active-expert count, from the probe.

    **Every** row is checked against the artifact — the shipped core count, ``in0_block_w``,
    ``per_core_N``, output placement *and grid orientation* — and the gap is printed wherever the shipped
    choice is not the measured winner. Review round 5 found the previous version checking one of eight
    rows and printing the literal string "as measured" for the other seven, four of which were 1-3 %
    behind the other rectangle of the same core count.

    Round 6 then found the lookup *still* under-constrained: it filtered on the core count and the output
    placement only, took the minimum over everything else, and so printed a time measured at an
    ``in0_block_w`` the layer does not use, which made a shipped row that was a couple of percent behind
    read as "the measured winner". Round 7 found the repair half-done: the cap was read from the *decode*
    log line and applied to every row, on the reasoning that ``in0_block_w`` is a function of ``K`` alone —
    true until round 6 made the gate/up cap depend on the realised core count as well, after which the
    prefill rows were reported at the decode cap and understated by roughly ten percent.

    So: the cap comes from a run of the shipped code, per phase.
    ``test_{decode,prefill}_runs_the_tuned_program_configs`` log both routed configs for both phases, and
    which phase applies is decided here from the realised core count, mirroring ``_sparse_cfg``. Everything
    else — ``per_core_N``, ``out_block_w``, ``out_subblock_w`` — follows from the core count and ``Nt``,
    which the matched probe row already carries.
    """
    rows = probe_rows(LOGS / "probe_sparse_matmul.txt", "SPARSE")
    rows = [r for r in rows if r["us"] is not None and "active" in r]
    missing = [r for r in probe_rows(LOGS / "probe_sparse_matmul.txt", "SPARSE") if "active" not in r]
    actives = sorted({int(r["active"]) for r in rows})
    #: The layer's rules, mirrored: cores = clamp(active / k, 8, 32) with k = 2 for gate/up and 4 for
    #: down, then the grid fills one axis first (cy = largest divisor of cores that is <= 10).
    K_PER_ROLE = {"gate_up": 2, "down": 4}

    def shipped_grid(cores):
        cy = min(cores, 10)
        while cores % cy:
            cy -= 1
        return cores // cy, cy

    #: What the fused decoder's rule picked: the largest core count dividing Nt, per_core_N 1.
    FUSED_RULE = {"gate_up": ("32(8x4)", "16", "1"), "down": ("64(8x8)", "8", "1")}

    #: The `in0_block_w` each routed role actually runs, per phase, read out of the suite log's own record
    #: of the configs the layer built. BOTH lines are read: `decode sparse matmuls` for the narrow-grid
    #: phase and `prefill sparse matmuls` for the wide one.
    #:
    #: Reading only the decode line was this generator's bug in round 7. Its docstring justified that with
    #: "`in0_block_w` is a function of `K` alone, so the logged value applies at every active count" — true
    #: before round 6 and false after it, because round 6's whole point was to make the gate/up cap depend
    #: on the realised core count too. The table then reported the *decode* cap on the prefill row and
    #: understated shipped prefill sparse performance by ~10 % on the largest op in that window.
    shipped_ibw = _shipped_sparse_ibw()
    lines = [
        "| active experts | role | fused rule | best measured | shipped | shipped vs winner |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for active in actives:
        for role in ("gate_up", "down"):
            pool = [r for r in rows if r.get("role") == role and r.get("active") == str(active)]
            if not pool:
                continue
            winner = min(pool, key=lambda r: r["us"])
            cores = max(8, min(32, max(1, active // K_PER_ROLE[role])))
            gx, gy = shipped_grid(cores)
            # Which phase's cap applies is decided by the **realised** core count, mirroring
            # `OptimizedMoE._sparse_cfg`: the target reduces to a divisor of `Nt` first, and only then does
            # the count select the cap. Round 8 found this reading the *target* — the same defect round 7
            # had just fixed in the layer — which happens to agree at the probe's four active points and
            # diverges at 24, 40, 48, 72 and 96, i.e. it would silently mis-key any point added later.
            realised = _largest_divisor_at_most(_sparse_n_tiles_mirror(role), max(1, cores))
            phase = "narrow" if realised <= SPARSE_MIN_CORES_MIRROR else "wide"
            # Every axis the probe sweeps must be pinned here, or `min()` silently reports a *faster*
            # geometry the layer does not build. The probe sweeps `out_block_w` and `sub_w` independently,
            # while the layer derives both from `per_core_N` (`_sparse_matmul_config`: `out_block_w =
            # per_core_N`, `out_subblock_w = _largest_divisor_at_most(per_core_N, 8)`), so at (32, `down`)
            # three rows share the (cores, in0_block_w, per_core_N) key and `min()` took the `sub_w=2` one -
            # round 10's finding, and the fifth round in a row that this one lookup was under-constrained by
            # exactly one axis. The rule below is mirrored from the layer and enforced by
            # `audit_figures.check_mirrored_constants`.
            per_core_n = _sparse_n_tiles_mirror(role) // max(1, realised)
            shipped_sub_w = _largest_divisor_at_most(per_core_n, 8)
            mine = min(
                (
                    r
                    for r in pool
                    if r.get("cores") == f"{cores}({gx}x{gy})"
                    and r.get("mem") == "L1"
                    and r.get("in0_block_w") == shipped_ibw.get((phase, role))
                    and r.get("per_core_N") == str(per_core_n)
                    and r.get("out_block_w") == str(per_core_n)
                    and r.get("sub_w") == str(shipped_sub_w)
                ),
                key=lambda r: r["us"],
                default=None,
            )
            cfg, ibw, pcn = FUSED_RULE[role]
            fused = best(rows, role=role, active=str(active), cores=cfg, in0_block_w=ibw, per_core_N=pcn)
            spread = max(float(winner.get("spread") or 0), float((mine or {}).get("spread") or 0))
            if mine is None:
                verdict = "**no probe row at the shipped geometry**"
                mine_cell = "—"
            else:
                gap = mine["us"] - winner["us"]
                mine_cell = (
                    f"**{mine['us']:.1f} µs** — {cores} cores ({gx}x{gy}), `in0_block_w` "
                    f"{mine['in0_block_w']}, `per_core_N` {mine['per_core_N']}"
                )
                if gap <= 1e-9:
                    verdict = "**the measured winner**"
                elif round(gap, 1) <= round(spread, 1):
                    verdict = (
                        f"+{gap:.1f} µs (+{100 * gap / winner['us']:.1f} %), **inside the ±{spread:.1f} µs spread**"
                    )
                else:
                    verdict = (
                        f"**+{gap:.1f} µs (+{100 * gap / winner['us']:.1f} %)**, beyond the ±{spread:.1f} µs spread"
                    )
            lines.append(
                f"| {active} | {role.replace('_', '/')} | {f'{fused:.1f} µs' if fused else '—'} | "
                f"**{winner['us']:.1f} µs** — {winner['cores']}, `in0_block_w` {winner['in0_block_w']}, "
                f"`per_core_N` {winner['per_core_N']}, {winner['mem']} | {mine_cell} | {verdict} |"
            )
    lines.append("")
    lines.append(
        "`spread` is the max-minus-min of three repeats of the same measurement, so a gap smaller than it "
        "is not a result. The 8-active rows are the tuned batch-1 decode target; ~162 is a 32-token "
        "prefill group; 32 and 64 are decode batch 4 and 8, supported for correctness and explicitly not "
        "tuned (§9 item 5)."
    )
    if missing:
        lines.append("")
        lines.append(
            f"**{len(missing)} probe rows carry no `active=` field** and are excluded: they were produced "
            "by an earlier revision of the probe. Re-run `logs/run_evidence.sh`."
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
    #: `absent`: the sweep also contains arms carrying `ckc=` and `max_cores_per_head_batch=`, which satisfy
    #: the same subset of keys and are faster, so the plain arms have to say what they do not carry.
    PLAIN = ("ckc", "max_cores_per_head_batch")
    sdpa_default = best(sdpa, cfg="default(None)", absent=PLAIN)
    # `probe_rows` splits on the FIRST "=", so the grid arrives as cfg="grid=8x8", not as a `grid` key.
    # The k-chunk rows are pinned to the shipped grid so the chunk is the only axis, and the grid rows are
    # pinned to the shipped chunk pair for the same reason. Both must track the layer: round 9 took 8x4 on the
    # sweep's advice and these pins had to move, then the suite sent the grid back to 8x8 on a capability bound
    # and they had to move back. A generated table can go stale against the code exactly like prose can.
    shipped_sdpa = best(sdpa, cfg="grid=8x8", k_chunk="64", absent=PLAIN)
    faster_sdpa = best(sdpa, cfg="grid=8x8", k_chunk="128", absent=PLAIN)
    faster_grid_sdpa = best(sdpa, cfg="grid=8x4", k_chunk="64", absent=PLAIN)
    # `max_cores_per_head_batch` decides how many cores flash-decode actually activates (default 16, so
    # 16 * B * num_kv_heads = 32 of the 64-core grid at batch 1). Round 11 pointed out the stage called this
    # config swept with that field defaulted; these rows are the sweep.
    mcphb_default = shipped_sdpa
    mcphb_low = best(sdpa, max_cores_per_head_batch="8")
    mcphb_high = best(sdpa, max_cores_per_head_batch="64")
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
            "taken; it saves "
            + (f"{interleaved - sharded:.1f} µs" if None not in (interleaved, sharded) else "more")
            + " per norm on the isolated op and adds two layout conversions. At the whole layer the two "
            "roughly cancel: `ab_norm_shard_cores.txt` shows the core count barely moving the layer at all, "
            "and the only layer-level sharded-vs-interleaved comparison is step 8 of work_log §3's "
            "development ladder. `ab_norm_shard_width.txt` answers a different question — whether the narrow "
            "head-dim norms should shard too — and both of its arms are sharded",
        ),
        (
            "paged flash decode",
            f"op default {cell(sdpa_default)}",
            f"**explicit config {cell(shipped_sdpa)}**",
            "taken; the default is more than an order of magnitude slower",
        ),
        (
            "paged flash decode, `max_cores_per_head_batch`",
            f"8 → {cell(mcphb_low)}; 64 → {cell(mcphb_high)}",
            f"**default 16, {cell(mcphb_default)}**",
            "kept; the field caps flash-decode's active cores at `16 * batch * kv_heads` = 32 of the grid's 64 "
            "at batch 1, so it — not the grid — is what sets SDPA's parallelism here. Raising it frees no time "
            "(the op is not core-bound at this geometry), halving it costs "
            + (
                f"{mcphb_low - mcphb_default:.1f} µs"
                if None not in (mcphb_low, mcphb_default)
                else "tens of microseconds"
            )
            + ". Swept in round 11 because the stage had called this config swept with this field defaulted",
        ),
        (
            "paged flash decode, grid",
            f"8x4 {cell(faster_grid_sdpa)} — *faster in isolation*",
            f"**8x8, 64 cores, {cell(shipped_sdpa)}**",
            "**rejected on capability**: faster by "
            + (
                f"{shipped_sdpa - faster_grid_sdpa:.1f} µs"
                if None not in (faster_grid_sdpa, shipped_sdpa)
                else "a fraction of a microsecond"
            )
            + " in isolation and a dead heat at the layer (`ab_sdpa_decode_grid.txt`), but flash-decode needs "
            "one core per batch row, so 32 cores cap decode at batch 32 and the supported batch-40/56 cases "
            "die in the op — the grid sets the servable batch, not the latency",
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
            + (f", **{comp['sdpa_share']:.2%}** `SDPA`" if comp.get("sdpa_share") else "")
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
                    code = (row["OP Code"] or "").strip()
                    # Ellipsis on truncation: round 8 found the bare 44-character cut printing shapes that
                    # do not exist (`… x 128 x 12` for `… x 128 x 128`), which a reader cannot match back to
                    # the committed report.
                    shown = code if len(code) <= 44 else code[:43] + "…"
                    rows.setdefault(key, set()).add(f"{shown} ({phase[:2]})")
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
    summaries = re.search(r"^\s*(\d+)\s+stack usage summary\s*$", census, re.M)
    dumps = re.search(r"^dumps: (\d+)$", census, re.M)
    if stack:
        # The census counts *detail lines* - one per RISC processor of the core that recorded a watermark -
        # not dumps, and the generator used to re-render that count as "dump(s)". Review round 11 caught it:
        # the evidence is one summary inside one dump, five processors, so the tightest figure is a single
        # sample rather than the floor of five. Firmware only emits a summary where it recorded a watermark,
        # so the breadth is not this stage's choice, but the sentence has to say which is which.
        text += (
            f" Watcher recorded a stack watermark in {summaries.group(1) if summaries else '?'} of its "
            f"{dumps.group(1) if dumps else '?'} dumps, across {stack.group(2)} RISC processors of the one "
            f"core that reported; the tightest leaves {stack.group(1)} bytes free, which is that single "
            f"sample rather than a measured floor across the run."
        )
    return (
        text + "\n\nArtifacts: [`watcher/watcher_log.txt.gz`](watcher/watcher_log.txt.gz), "
        "[`watcher/census_summary.txt`](watcher/census_summary.txt), console log "
        "[`logs/watcher_pytest.txt.gz`](logs/watcher_pytest.txt.gz) — the two large ones are committed gzipped, "
        "which round 11 found these links ignoring."
    )


#: The op codes the operation-topology audit (`work_log.md` §2) tabulates, per layer kind, with what each is
#: and what happened to it. Only the **times** are generated; the prose columns are the audit's own judgement.
#: Round 10 found five of these times wrong - a baseline table carrying post-optimization values, and one row
#: taken from the other layer kind - because they are two-digit integers, which `ALLOWED_INT` exempts from
#: sourcing wholesale. Generating them from the fused stage's committed capture is the only way this table
#: cannot drift again, and its wrongness mattered: it is the table the whole stage was planned from.
TOPOLOGY_AUDIT = {
    "full_attention": [
        (
            "SparseMatmulDeviceOperation active=?/256 x 32 x 2048 x 1024",
            "`SparseMatmul active=?/256 x 32 x 2048 x 1024`",
            (
                "packed routed-expert gate/up",
                "BFP4 weights + LoFi; core/block geometry; L1 output; fewer active experts",
                "**all four taken** (§3.1, §3.2, §3.3, §3.4)",
            ),
        ),
        (
            "SparseMatmulDeviceOperation active=?/256 x 32 x 512 x 2048",
            "`SparseMatmul active=?/256 x 32 x 512 x 2048`",
            (
                "routed-expert down",
                "same",
                "**all four taken**",
            ),
        ),
        (
            "UnaryDeviceOperation",
            "`UnaryDeviceOperation`",
            (
                "~99 % `UnaryOpType::FILL` — `sparse_matmul` zeroing its 256-expert-wide output",
                "move the output to L1; halve its dtype",
                "**taken** (§3.2, §3.3)",
            ),
        ),
        (
            "BinaryNgDeviceOperation",
            "`BinaryNgDeviceOperation`",
            (
                "SwiGLU multiply + router-score multiply, both over the 256-expert axis",
                "L1 + BFP8 intermediates",
                "**taken** (§3.2, §3.3)",
            ),
        ),
        (
            "MatmulDeviceOperation 32 x 4096 x 2048",
            "`MatmulDeviceOperation 32 x 4096 x 2048`",
            (
                "`o_proj` — flagged `SLOW`, 23.2 % of DRAM bandwidth",
                "explicit decode program config; DRAM-sharded",
                "**explicit 1D config taken**, DRAM-sharded measured and rejected (§4.1)",
            ),
        ),
        (
            "SliceDeviceOperation",
            "`SliceDeviceOperation`",
            (
                "mostly the two slices that unpack the packed gate/up output",
                "split the pair instead; L1 + BFP8",
                "**packed kept** (§4.2), L1+BFP8 taken",
            ),
        ),
        (
            "MatmulDeviceOperation 32 x 2048 x 9216",
            "`MatmulDeviceOperation 32 x 2048 x 9216`",
            (
                "packed attention in-projection",
                "explicit config; BFP8/BFP4 weights",
                "**BFP8 + explicit config taken**, BFP4 measured (§4.6)",
            ),
        ),
        (
            "TopKDeviceOperation",
            "`TopKDeviceOperation`",
            (
                "router top-8 over 256 experts, single core",
                "pad to the multi-core width; replace the gate op",
                "**both measured and rejected** (§4.3, §4.4)",
            ),
        ),
        (
            "LayerNormDeviceOperation",
            "`LayerNormDeviceOperation`",
            (
                "four RMSNorms; the two residual ones run on **one core**",
                "width-sharded L1 + `LayerNormShardedMultiCoreProgramConfig`",
                "**taken** (§3.6)",
            ),
        ),
        (
            "MatmulDeviceOperation 32 x 2048 x 256",
            "`MatmulDeviceOperation 32 x 2048 x 256`",
            (
                "router projection, 8 cores, 9.4 % of DRAM bandwidth",
                "explicit config",
                "**taken** (§3.5)",
            ),
        ),
        (
            "DeepseekMoEFastReduceNCDeviceOperation",
            "`DeepseekMoEFastReduceNC`",
            (
                "expert-axis reduction over the 256-wide down output",
                "L1 + BFP8 input",
                "**taken**",
            ),
        ),
        (
            "SdpaDecodeDeviceOperation",
            "`SdpaDecodeDeviceOperation`",
            (
                "paged flash-decode",
                "reduced cache dtype; program config sweep",
                "**BFP8 cache taken** (§3.7), config swept (§4.5)",
            ),
        ),
    ],
    "linear_attention": [
        (
            "MatmulDeviceOperation 32 x 2048 x 12352",
            "`MatmulDeviceOperation 32 x 2048 x 12352`",
            (
                "packed DeltaNet in-projection",
                "BFP8 + explicit decode config (§3.5)",
            ),
        ),
        (
            "MatmulDeviceOperation 32 x 4096 x 2048",
            "`MatmulDeviceOperation 32 x 4096 x 2048`",
            (
                "`out_proj`",
                "explicit decode config (§3.5) + a bfloat16 activation (§4.11)",
            ),
        ),
        (
            "MatmulDeviceOperation b={32} x 32 x 128 x 128",
            "3 x `MatmulDeviceOperation b={32} 32 x 128 x 128`",
            (
                "the float32 recurrent-state matmuls: decay read, delta outer product, output read",
                "operands moved to L1 — 19 µs (§3.9 item 2); fidelity swept and rejected",
            ),
        ),
        (
            "ReshapeViewDeviceOperation+PermuteDeviceOperation",
            "`ReshapeViewDeviceOperation` + `PermuteDeviceOperation`",
            (
                "the one-shot head-major relayout of the conv output",
                "inherited from the fused stage, which already reduced it from three round trips to one",
            ),
        ),
        (
            "TilizeWithValPaddingDeviceOperation+ConcatDeviceOperation+UntilizeWithUnpaddingDeviceOperation",
            "`TilizeWithValPadding` + `Concat` + `UntilizeWithUnpadding`",
            (
                "`repeat_interleave`'s GQA head expansion",
                "output moved to L1 (§3.9 item 5); the once-per-step op-to-op stall in front of it is the largest single gap in README §7's generated itemisation",
            ),
        ),
        (
            "TernaryDeviceOperation",
            "`TernaryDeviceOperation`",
            (
                "the `addcmul` conv-tap accumulation",
                "inherited; the fused stage measured `addcmul` against `mac` and kept it",
            ),
        ),
    ],
}


def fused_op_totals(kind: str) -> dict:
    """``{op code: us/step}`` from the **fused** stage's committed decode capture — the audit's baseline.

    The fused stage's reports are read rather than this stage's on purpose: §2 is the pre-optimization read of
    the measured path, so its numbers must be the ones that were there before any change. Round 10 found three
    rows carrying this stage's *post*-change values, which understated the stage's own largest dense win.
    """
    path = ROOT.parent / "fused_decoder/tracy" / kind / "decode_perf_report.csv"
    raw = read(path)
    totals: dict = {}
    for row in csv.DictReader(io.StringIO(raw)):
        code, time = row.get("OP Code"), row.get("Device Time")
        if not code:
            continue
        try:
            totals[code] = totals.get(code, 0.0) + float((time or "0").replace(",", ""))
        except ValueError:
            pass
    return {k: v / TRACED_REPLAYS for k, v in totals.items()}


def topology_time(kind: str, code: str) -> str:
    """One cell of the audit table: a single op code, or a ``+``-joined chain summed as one."""
    totals = fused_op_totals(kind)
    parts = code.split("+")
    if any(p not in totals for p in parts):
        missing = [p for p in parts if p not in totals]
        raise SystemExit(f"work_log §2: no {missing} row in the fused {kind} capture")
    return f"{sum(totals[p] for p in parts):.1f}"


#: The composite chains README §6 itemises: a chain is a *consecutive* run of op codes inside one traced step,
#: so it is summed positionally rather than by op code. Round 10 found the `repeat_interleave` figure roughly
#: doubled, because summing by op code sweeps in the router's and topk's untilizes as well - they share codes
#: with this chain and are separate calls. Generated for the same reason as everything else in §5: these are
#: two-digit integers when written by hand, and `ALLOWED_INT` exempts those from sourcing.
COMPOSITE_CHAINS = {
    "ttnn.repeat_interleave (GQA head expansion)": (
        "linear_attention",
        ("UntilizeWithUnpaddingDeviceOperation", "ConcatDeviceOperation", "TilizeWithValPaddingDeviceOperation"),
    ),
    "ttnn.scatter (router)": (
        "linear_attention",
        (
            "UntilizeDeviceOperation",
            "UntilizeWithUnpaddingDeviceOperation",
            "UntilizeWithUnpaddingDeviceOperation",
            "ScatterDeviceOperation",
            "TilizeDeviceOperation",
        ),
    ),
}


def optimized_step_ops(kind: str) -> list:
    """``[(op code, us)]`` for one traced decode step of the shipped decoder, in dispatch order."""
    raw = read(TRACY / kind / "decode_perf_report.csv")
    rows = [r for r in csv.DictReader(io.StringIO(raw)) if r.get("OP Code")]
    per_step = len(rows) // TRACED_REPLAYS
    return [(r["OP Code"], float((r["Device Time"] or "0").replace(",", ""))) for r in rows[:per_step]]


def composite_chain_us(kind: str, codes: tuple) -> float:
    """Sum of the first consecutive run matching ``codes`` in one step. Positional, not by op code."""
    ops = optimized_step_ops(kind)
    for start in range(len(ops) - len(codes) + 1):
        window = ops[start : start + len(codes)]
        if all(have == want for (have, _), want in zip(window, codes)):
            return sum(us for _, us in window)
    raise SystemExit(f"README §6: no consecutive {codes} chain in the {kind} decode capture")


def block_composite_chains() -> str:
    rows = ["| Composite op | Chain, in dispatch order | µs/step |", "| --- | --- | --- |"]
    for label, (kind, codes) in COMPOSITE_CHAINS.items():
        pretty = " → ".join(c.replace("DeviceOperation", "") for c in codes)
        rows.append(f"| `{label}` | {pretty} | **{composite_chain_us(kind, codes):.1f}** |")
    rows.append("")
    rows.append(
        "Both are measured on `linear_attention`, summed over the *consecutive* ops of each call rather than by "
        "op code — the router's and `topk`'s untilizes share op codes with the head-expansion chain and are "
        "separate calls, which is how review round 10 found the `repeat_interleave` figure roughly doubled."
    )
    return "\n".join(rows)


def _harness_runs(path, pattern: str) -> dict:
    """``{key: [ms, ...]}`` from a layer-level A/B artifact, in file order."""
    runs: dict = {}
    for match in re.finditer(pattern, read(LOGS / path)):
        *key, ms = match.groups()
        runs.setdefault(tuple(key), []).append(float(ms))
    return runs


#: The active-expert counts work_log §4.14's orientation ladder tabulates, with what each one is. Every other
#: field of the comparison — core count, `in0_block_w`, `per_core_N`, output block and subblock — is derived
#: from the shipped rule below rather than typed, because hardcoding them is how the first draft of this table
#: compared rows the layer does not build (three of six were wrong by hundreds of microseconds).
ORIENTATION_POINTS = [
    (8, "the tuned batch-1 decode target"),
    (162, "a 32-token prefill group"),
    (64, "decode batch 8, **not tuned**"),
]


def _shipped_sparse_ibw() -> dict:
    """``{(phase, role): in0_block_w}``, read out of the suite log's record of the configs the layer built.

    Factored out of `block_sparse_search` in round 11 so the orientation ladder uses the same map rather than a
    second copy of the derivation — two copies of a mirrored rule is how the mirror drifts.
    """
    shipped_ibw: dict = {}
    # Matched without the trailing colon: the prefill line carries `layer=N (kind):` between the marker and
    # the configs, so a marker ending in ":" silently matched nothing and every wide-geometry row rendered
    # "no probe row at the shipped geometry".
    for phase, marker in (("narrow", "decode sparse matmuls"), ("wide", "prefill sparse matmuls")):
        for line in read(LOGS / "pytest_full_suite.txt").splitlines():
            if marker not in line:
                continue
            found = re.findall(r"in0_block_w=(\d+)", line[line.index(marker) :])
            if len(found) >= 2:
                shipped_ibw[(phase, "gate_up")] = found[0]
                shipped_ibw[(phase, "down")] = found[1]
                break

    return shipped_ibw


def block_orientation_ladder() -> str:
    """work_log §4.14's op-level orientation table, generated from `probe_sparse_matmul.txt`.

    Transcribing this has been wrong with the sign inverted (rounds 8 and 9) and stale against a fresh sweep
    three times. The comparison is mechanical — same active count, role, `in0_block_w` and output block, the two
    grid rectangles of one core count — so it is generated, and the verdict is computed from the two times and
    their spreads. Every field is derived from the layer's rule, including the ones that are easy to get wrong.
    """
    rows = [r for r in probe_rows(LOGS / "probe_sparse_matmul.txt", "SPARSE") if r["us"] is not None]
    shipped_ibw = _shipped_sparse_ibw()
    lines = ["| point | role | shipped (column) | other (row) | verdict |", "| --- | --- | --- | --- | --- |"]
    for active, note in ORIENTATION_POINTS:
        for role in ("gate_up", "down"):
            n_tiles = SPARSE_N_TILES_MIRROR[role]
            target = min(32, max(SPARSE_MIN_CORES_MIRROR, active // {"gate_up": 2, "down": 4}[role]))
            cores = _largest_divisor_at_most(n_tiles, max(1, target))
            per_core_n = n_tiles // cores
            sub_w = _largest_divisor_at_most(per_core_n, 8)
            phase = "narrow" if cores <= SPARSE_MIN_CORES_MIRROR else "wide"
            ibw = shipped_ibw.get((phase, role))
            found = {}
            for r in rows:
                if (
                    r.get("active") == str(active)
                    and r.get("role") == role
                    and r.get("in0_block_w") == str(ibw)
                    and r.get("per_core_N") == str(per_core_n)
                    and r.get("out_block_w") == str(per_core_n)
                    and r.get("sub_w") == str(sub_w)
                    and r.get("mem") == "L1"
                ):
                    shape = str(r.get("cores")).split("(")[-1].rstrip(")")
                    cx, _, cy = shape.partition("x")
                    if cx.isdigit() and cy.isdigit() and int(cx) * int(cy) == cores:
                        found["column" if int(cy) >= int(cx) else "row"] = (r["us"], float(r.get("spread") or 0))
            label = f"{active} active" + (f" — {note}" if note and role == "gate_up" else "")
            if "column" not in found or "row" not in found:
                lines.append(f"| {label} | {role.replace('_', '/')} | — | — | no matched probe row |")
                continue
            (col_us, col_sp), (row_us, row_sp) = found["column"], found["row"]
            span, gap = max(col_sp, row_sp), abs(col_us - row_us)
            winner = "column" if col_us < row_us else "row"
            verdict = (
                f"{winner} nominally ahead, inside the ±{span:.1f} µs spread"
                if round(gap, 1) <= round(span, 1)
                else f"**{winner}** wins by {gap:.1f} µs, beyond the ±{span:.1f} µs spread"
            )
            cells = (
                f"**{col_us:.1f} µs**" if winner == "column" else f"{col_us:.1f} µs",
                f"**{row_us:.1f} µs**" if winner == "row" else f"{row_us:.1f} µs",
            )
            lines.append(f"| {label} | {role.replace('_', '/')} | {cells[0]} | {cells[1]} | {verdict} |")
    return "\n".join(lines)


def block_layer_ab() -> str:
    """The layer-level A/B figures README §5.1, §5.4 and §9 quote, generated rather than transcribed.

    These are the numbers that change on every sweep, and hand-typing them has been wrong twice: round 10 found
    §5.1 claiming the harness's repeats "agree to the last digit" and §9 crediting the `q0/k0` candidate with a
    layer win, both contradicted by the artifact of the day; round 11's own text then went stale against the
    next run before it was committed. Generated, they cannot.
    """
    harness = _harness_runs(
        "ab_decode_harness.txt",
        r"arm=(\S+) trace_region=(\d+) run=\d+ layer=\d+ \((\w+)\).*?wall/iter=([\d.]+)",
    )
    grid = _harness_runs(
        "ab_sdpa_decode_grid.txt",
        r"knob=(\S+) arm=(\S+) run=\d+ layer=\d+ \((\w+)\) (\S+) wall/iter=([\d.]+)",
    )
    lines = [
        "| A/B | arm | builds, ms | span |",
        "| --- | --- | --- | --- |",
    ]
    for (arm, region, kind), values in sorted(harness.items()):
        if region != "0":
            continue
        lines.append(
            f"| SDPA `q0/k0` candidate, {kind} | `{arm}` | {' / '.join(f'{v:.3f}' for v in values)} | "
            f"{max(values) - min(values):.3f} |"
        )
    for (knob, arm, kind, phase), values in sorted(grid.items()):
        if not phase.startswith("prefill"):
            continue
        lines.append(
            f"| routed `down` orientation, {kind} | `{arm}` | {' / '.join(f'{v:.3f}' for v in values)} | "
            f"{max(values) - min(values):.3f} |"
        )
    # The conclusions are computed, not asserted: which arm leads, and by how much against the spans, changes
    # between sweeps. Round 10 found prose claiming one direction and round 11's replacement went stale against
    # the next run in the same direction, so the sentence below states this run and the reasoning that survives
    # either outcome.
    q0 = {kind: values for (arm, region, kind), values in harness.items() if region == "0" and "candidate" in arm}
    shipped = {kind: values for (arm, region, kind), values in harness.items() if region == "0" and "shipped" in arm}
    lead = {
        kind: (min(shipped[kind]) - min(q0[kind])) * 1000 for kind in q0 if kind in shipped
    }  # µs, positive = candidate ahead
    spans = [max(v) - min(v) for v in harness.values()]
    lines.append("")
    lines.append(
        f"Read the **span** column first. In this run three fresh builds of one decode arm agree to within "
        f"{max(spans) * 1000:.0f} µs, and the `q0/k0` candidate leads the shipped arm by "
        + ", ".join(f"{v:.1f} µs on `{k}`" for k, v in sorted(lead.items()))
        + " — but the previous sweep of the same file had the same arms in the *opposite* order by a similar "
        "margin, and its spans were an order of magnitude wider. A layer difference of a microsecond or two is "
        "therefore not a property of the configuration, which is why §9 item 8 rejects that candidate on its "
        "invariant rather than on timing, and why no rejection in this stage rests on a sub-span layer gap. The "
        "orientation arms are the contrast: every build of the shipped column beats every build of the row "
        "candidate, on both layer kinds, by far more than any span here — which is what settles §5.4."
    )
    return "\n".join(lines)


def block_topology_audit(kind: str) -> str:
    """work_log §2's operation-topology audit table for one layer kind, times generated from the fused capture."""
    ranked = kind == "full_attention"
    head = (
        "| Rank | Op code | µs/step | What it is | Candidate | Action |"
        if ranked
        else ("| Op code | µs/step | What it is | Action |")
    )
    rule = "| --- | --- | --- | --- | --- | --- |" if ranked else "| --- | --- | --- | --- |"
    lines = [head, rule]
    for index, (code, label, rest) in enumerate(TOPOLOGY_AUDIT[kind], start=1):
        cells = ([str(index)] if ranked else []) + [label, topology_time(kind, code), *rest]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def splice(path: Path, blocks: dict, check: bool) -> int:
    """Replace every ``<!-- generated:NAME -->`` block in ``path``; returns how many were filled."""
    text = original = path.read_text()
    for name, body in blocks.items():
        marker, end = f"<!-- generated:{name} -->", f"<!-- /generated:{name} -->"
        if marker not in text or end not in text:
            raise SystemExit(f"{path.name} is missing the {name} block markers")
        replacement = f"{marker}\n{body}\n{end}"
        text = re.sub(re.escape(marker) + r".*?" + re.escape(end), lambda _: replacement, text, flags=re.S)
    if check:
        if text != original:
            raise SystemExit(f"{path.name} disagrees with the artifacts; re-run make_readme.py")
        return 0
    path.write_text(text)
    return len(blocks)


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
        "composite-chains": block_composite_chains(),
        "layer-ab": block_layer_ab(),
    }
    # work_log §2's operation-topology audit is generated too, from the *fused* stage's capture: round 10
    # found five of its times wrong, and they are two-digit integers that the figure audit exempts wholesale,
    # so generation is the only thing that keeps them honest.
    work_log_blocks = {
        "orientation-ladder": block_orientation_ladder(),
        "topology-audit-full": block_topology_audit("full_attention"),
        "topology-audit-linear": block_topology_audit("linear_attention"),
    }
    if args.check:
        splice(README, blocks, check=True)
        splice(WORK_LOG, work_log_blocks, check=True)
        print("README matches the artifacts")
        return
    filled = splice(README, blocks, check=False) + splice(WORK_LOG, work_log_blocks, check=False)
    print(f"filled {filled} generated blocks in {README.name} and {WORK_LOG.name}")


if __name__ == "__main__":
    main()

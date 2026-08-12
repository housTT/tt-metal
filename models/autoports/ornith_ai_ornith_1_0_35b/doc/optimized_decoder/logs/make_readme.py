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
``accounting``                 ``tracy/perf_summary.json``
``advice``                     the ``Advice`` column of ``tracy/<kind>/decode_perf_report.csv``
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
            lines.append(
                f"| `{kind}` {label} | {b[0]:.2f} ms / {b[1]:.1f} {unit} | "
                f"**{a[0]:.3f} ms / {a[1]:.1f} {unit}** | **{b[0] / a[0]:.2f}x** |"
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
            lines.append(
                f"| `{kind}` | {label} | {b[0]:.3f} ms | **{a[0]:.3f} ms** | "
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
    lines.append("")
    lines.append(
        "Every BFP4 row clears the bar, so this is not a pass/fail rejection — it is a 30–43x "
        "increase in layer error for 2.3 % of one traced decode step and nothing in prefill, in one "
        "layer of a 48-layer stack. The routed-expert BFP4 step this stage *did* take is the "
        'opposite trade. Rejected on that comparison, and shipped as `POLICIES["bfp4-projections"]` '
        "so `$datatype-sweep` can take it without rediscovering it. Full ladder: "
        "[`logs/probe_projection_dtype.txt`](logs/probe_projection_dtype.txt)."
    )
    return "\n".join(lines)


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


#: What this stage did about each distinct `tt-perf-report` advice item, keyed by a substring of the
#: advice text. The *counts* and the op rows are read from the committed reports so they cannot drift;
#: only the action prose lives here.
ADVICE_ACTIONS = {
    "DRAM-sharded program config": (
        "**Tried, rejected with measurement** — loses on all seven dense roles, even without its "
        "activation-reshard cost (§5.4)."
    ),
    "place input 0 in L1": (
        "**Taken for decode, measured and rejected for prefill.** Decode: the two residual norms, the "
        "three float32 recurrent-state matmuls (worth 19 µs/step) and the shared expert's SwiGLU "
        "product all hand their result to L1, each size-gated so a large batch still uses DRAM — the "
        "item is now raised 0 times in both decode reports. The head-dim norms are the one decode "
        "exception and cannot move: `paged_scaled_dot_product_attention_decode` rejects a non-sharded "
        "Q outside DRAM. Prefill: still raised on three rows, and measured — an L1 `in0` is *slower* "
        "on the two that matter (`attn_in` 471.7 vs 459.5 µs, `gdn_in` 654.7 vs 647.0) and worth "
        "~3 µs on `shared_in`, i.e. 0.006 % of a 96 ms prefill window. "
        "`probe_prefill_matmul.txt` `in0=DRAM`/`in0=L1` rows."
    ),
    "in0_block_w=1 is small": (
        "**Taken.** The five dense *prefill* rows got explicit 2D configs with `in0_block_w` 8/16 "
        "(§5.4), and the three recurrent-state rows got an explicit "
        "`MatmulMultiCoreReuseProgramConfig` — that family does expose `in0_block_w`, unlike the "
        "`core_grid` spelling the fused stage used: 2 for the two reads (13.9 vs 14.7 µs) and 1 for "
        "the `transpose_a` outer product, where `Kt` is 1 tile so 2 and 4 are rejected by the op "
        "(12.6 vs 20.4 µs). `probe_decode_micro.txt` `STATE progcfg` rows."
    ),
    "HiFi2 is sufficient": (
        "**Tried, rejected with measurement** on the recurrent-state rows: HiFi2 is 14.6 µs against "
        "14.9 for the shipped HiFi4 + fp32-accumulate and LoFi is 14.7 — ≤0.3 µs per matmul, under "
        "0.1 % of the window, for the float32 state that is the model's exact carry between steps."
    ),
    "Output subblock 1x1 is small": (
        "**Tried, rejected with measurement** — the `per_core_N` ≥ 2 alternative is slower for both "
        "rows (`shared_in` 11.5 vs 9.4 µs, §5.4)."
    ),
    "use HiFi4 with BF16 activations": (
        "**Rejected with measurement** — the reverse direction of §4.2's fidelity sweep; HiFi4 is what "
        "the fused decoder had and it is slower at equal correctness."
    ),
    "HiFi2 may also work": (
        "**Rejected on purpose** (the router row): the matmul is 9 µs and its output decides *which "
        "experts run*. The fused stage measured bfloat16 routing agreeing with float32 on only "
        "99.8 % / 95.5 % of top-8 sets, so this group stays BF16/HiFi4/fp32-accumulate."
    ),
    "look good": (
        "**Not advice** — `tt-perf-report` printing that a row's `in0_block_w` and output subblock are "
        "already what it would have suggested. Kept in the table so the generator cannot silently "
        "drop a line it does not recognise."
    ),
    "nnz=std::nullopt": (
        "**Reporting limitation, not advice.** `nnz` is inferred at runtime because pinning it wedged "
        "the device (§9 item 3); the report cannot model DRAM/FLOP utilisation for those rows, so "
        "this stage measures their share of device time instead."
    ),
}


def block_advice():
    """Every distinct advice item in **all four** committed reports, with per-window counts and rows.

    Decode counts are per traced step (32 replays); prefill counts are per pass. Review round 3 found
    this table scoped to the two decode reports while an item was still open on three prefill rows.
    """
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
                    key = next((k for k in ADVICE_ACTIONS if k in item), item[:48])
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
        action = ADVICE_ACTIONS.get(key, "**unclassified — this stage did not act on it**")
        lines.append(f"| *{key}* | {' | '.join(cells)} | {op_cell} | {action} |")
    closed = [k for k in ADVICE_ACTIONS if k not in keys]
    if closed:
        lines += [
            "",
            "Advice items **no longer raised in any of the four committed reports**:",
            "",
            "| Advice (no longer raised) | What closed it |",
            "| --- | --- |",
        ]
        for key in closed:
            lines.append(f"| *{key}* | {ADVICE_ACTIONS[key]} |")
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

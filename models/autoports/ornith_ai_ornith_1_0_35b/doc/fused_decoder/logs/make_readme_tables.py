# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate the README's PCC tables from the two stages' metric summaries, and splice them in.

The functional and fused suites log the same metrics under the same test names, so §2.1, §2.2 and
§2.3 of the README can be *generated* rather than transcribed. Round 3 of this stage's review found
why that matters: the tables had been written by hand from an earlier run and eleven cells disagreed
with `logs/pcc_summary.txt`. Hand-transcription is the defect, so this script owns those blocks
outright — including the "material delta" sentence under §2.2, whose numbers are computed from the
same rows rather than eyeballed off the table.

Each owned block sits between ``<!-- generated:NAME -->`` and ``<!-- /generated:NAME -->`` markers in
README.md. Three modes:

    python .../logs/make_readme_tables.py           # print the generated blocks to stdout
    python .../logs/make_readme_tables.py --write   # splice them into README.md
    python .../logs/make_readme_tables.py --check   # exit 1 if README.md is out of date

``--check`` runs from ``audit_figures.py``, so a stale table fails the figure audit rather than
surviving to review.
"""

import math
import re
import sys
from pathlib import Path

L = Path(__file__).resolve().parent
FUSED = L / "pcc_summary.txt"
FUNCTIONAL = L.parent.parent / "functional_decoder/logs/pcc_summary.txt"
README = L.parent / "README.md"

KIND = {0: "linear_attention", 3: "full_attention"}
BAR = 0.995
#: tests/test_fused_decoder.py::EQUIV_BAR — the fused-vs-functional acceptance bar.
EQUIV_BAR = 0.9999

PREFILL = re.compile(r"prefill layer=(?P<layer>\d+) \([a-z_]+\) seq_len=(?P<seq>\d+) PCC=(?P<pcc>[\d.]+)")
LONG_PREFILL = re.compile(
    r"long-context PCC layer=(?P<layer>\d+) seq_len=(?P<seq>\d+) prefill=(?P<pcc>[\d.]+) decode=(?P<dpcc>[\d.]+)"
)
DECODE = re.compile(
    r"decode layer=(?P<layer>\d+) \([a-z_]+\) prefill_len=(?P<plen>\d+) step=(?P<step>\d+) "
    r"pos=\d+ PCC=(?P<pcc>[\d.]+)"
)
#: §2.4's rows. Each entry is (property, test label, extractor). The extractor is handed the
#: pcc_summary lines for its test and returns the "Result" cell, so the *numbers* are read from the
#: artifact and only the prose is written here. Review round 4 found three stale cells in this table
#: — it was the one part of §2 that had never been converted, and it drifted exactly like the rest.
CAPABILITY = re.compile(r"^(?P<test>[a-z_]+): (?P<msg>.*)$", re.M)

EQUIV = re.compile(
    r"fused-vs-functional layer=(?P<layer>\d+) \([a-z_]+\) seq_len=(?P<seq>\d+) "
    r"prefill PCC=(?P<pre>[\d.]+) decode PCC=(?P<dec>[\d.]+)"
)


def scrape(path: Path):
    text = path.read_text(errors="replace")
    prefill, decode = {}, {}
    for m in PREFILL.finditer(text):
        prefill[(int(m["layer"]), int(m["seq"]))] = m["pcc"]
    for m in LONG_PREFILL.finditer(text):
        prefill[(int(m["layer"]), int(m["seq"]))] = m["pcc"]
        decode[(int(m["layer"]), int(m["seq"]), 0)] = m["dpcc"]
    for m in DECODE.finditer(text):
        decode[(int(m["layer"]), int(m["plen"]), int(m["step"]))] = m["pcc"]
    return prefill, decode


def scrape_equivalence(path: Path):
    equiv = {}
    for m in EQUIV.finditer(path.read_text(errors="replace")):
        equiv[(int(m["layer"]), int(m["seq"]))] = (m["pre"], m["dec"])
    return equiv


ROLE = {
    1: "single token",
    7: "sub-tile, sub-page",
    32: "exactly one tile",
    64: "exactly one page block",
    128: "exactly the physical alignment",
    129: "one past the alignment",
    250: "tile-padded height already equals the padded length",
    2048: "exactly one internal chunk",
    2049: "one past a chunk boundary",
    3000: "multi-chunk, non-divisible",
    8000: "long, non-divisible (`test_long_context_pcc`)",
}


def capability_lines(path: Path):
    """``{test_name: [logged message, ...]}`` from a metric summary."""
    out = {}
    for m in CAPABILITY.finditer(path.read_text(errors="replace")):
        out.setdefault(m["test"], []).append(m["msg"])
    return out


def _pccs(lines, *, contains=None):
    """Every ``PCC=...`` value in the matching lines, in order."""
    picked = [ln for ln in lines if contains is None or contains in ln]
    return [v for ln in picked for v in re.findall(r"PCC=([\d.]+)", ln)]


def _span(values):
    """``lo-hi`` over a set of PCC strings, or the single value if they are all equal."""
    lo, hi = min(values), max(values)
    return lo if lo == hi else f"{lo}–{hi}"


def _by_layer(lines, fmt):
    """``linear …; full …`` built from the layer-0 and layer-3 lines."""
    return "; ".join(
        f"{name} {fmt([ln for ln in lines if f'layer={layer}' in ln])}" for layer, name in ((0, "linear"), (3, "full"))
    )


def capability_rows(cap):
    """The (property, test, result) triples of §2.4, with every number read from the summary."""

    def pair(lines):
        pccs = _pccs(lines)
        return " / ".join(pccs)

    def rope(_lines):
        vals = set(_pccs(cap["test_rope_matches_hf"]))
        assert len(vals) == 1, f"rope PCCs are no longer uniform: {sorted(vals)}"
        return f"every prefill and decode-gather cos/sin PCC {vals.pop()}"

    def batched(batch):
        def build(_lines):
            rows = [ln for ln in cap["test_batched_prefill_decode_pcc"] if f"batch={batch}" in ln]
            return _by_layer(rows, lambda ls: " / ".join(_pccs(ls)))

        return build

    def ragged(_lines):
        parts = []
        for batch in (4, 13):
            vals = _pccs([ln for ln in cap["test_batched_decode_ragged_positions"] if f"batch={batch} " in ln])
            parts.append(f"batch {batch}: {_span(vals)}")
        return "; ".join(parts)

    def above_limit(_lines):
        lines = cap["test_decode_batch_above_head_split_limit"]
        out = []
        for batch, what in (
            (40, "head-split fallback, fused cache update"),
            (56, "head-split **and** two-launch cache update"),
        ):
            (pcc,) = _pccs(lines, contains=f"batch={batch} ")
            out.append(f"{batch} ({what}): {pcc}")
        return "; ".join(out)

    def smaller(_lines):
        (pre, dec) = _pccs(cap["test_batch_smaller_than_allocated_state"])
        return (
            f"`full_attention` batch 1 on a batch-8 allocation: prefill {pre}, decode {dec}; "
            "`linear_attention` raises, as the functional decoder does, because its DeltaNet state is per-row"
        )

    def permuted(_lines):
        lines = cap["test_permuted_page_table"]
        (pre,) = _pccs(lines, contains="prefill")
        dec = _pccs(lines, contains="decode")
        slot = re.search(r"first_slot=(\d+)", " ".join(lines))[1]
        return f"prefill {pre} (first physical slot {slot}); decode {' / '.join(dec)}"

    def poisoned(_lines):
        lines = cap["test_forward_with_poisoned_free_pool"]
        return _by_layer(
            lines,
            lambda ls: (
                f"prefill {_span(_pccs([re.sub(r' decode PCC=[\d.]+', '', ln) for ln in ls]))}, "
                f"decode {_span([re.search(r'decode PCC=([\d.]+)', ln)[1] for ln in ls])}"
            ),
        )

    def determinism(lines):
        runs = {m for ln in lines for m in re.findall(r"(\d+/\d+) runs bit-identical", ln)}
        kinds = {ln.split("layer=")[1][0] for ln in lines if "layer=" in ln}
        assert runs and len(kinds) == 2, f"determinism evidence is incomplete: {lines}"
        return f"{'/'.join(sorted(runs))} runs **bit-identical**, both kinds"

    def stress(lines):
        cycles = {m for ln in lines for m in re.findall(r"(\d+) prefill\+(?:\d+)-step-decode cycles", ln)}
        steps = {m for ln in lines for m in re.findall(r"prefill\+(\d+)-step-decode", ln)}
        lengths = {m for ln in lines for m in re.findall(r"over lengths \[([\d, ]+)\]", ln)}
        growth = {m for ln in lines for m in re.findall(r"\(growth (\d+)\)", ln)}
        assert len(cycles) == len(steps) == len(lengths) == 1 and lines, f"stress evidence is incomplete: {lines}"
        assert all("repeats bit-identical" in ln for ln in lines), "a stress run was not bit-identical"
        pretty = " / ".join(x.strip() for x in lengths.pop().split(","))
        return (
            f"{cycles.pop()} prefill + {steps.pop()}-step-decode cycles over prompt lengths {pretty}: "
            f"repeats **bit-identical**, DRAM allocation growth **{'/'.join(sorted(growth))}** bytes"
        )

    def host_fallback(lines):
        assert len(lines) == 2 and all(
            "both guards verified to fire" in ln for ln in lines
        ), f"host-fallback evidence is incomplete: {lines}"
        assert all("clean for ttnn" in ln for ln in lines), "a host-fallback audit was not clean"
        return "clean for both kinds, with positive controls proving both guards fire — §6"

    def std_tail(test):
        """The `tail std` values a completion-only test logs, per layer kind.

        These are **not** PCCs and must never be printed as if they were: at the advertised context
        there is no tractable HF golden, so the assertion is completion + finiteness +
        non-degeneracy, and the standard deviation of the output tail is the non-degeneracy witness.
        A bare `0.5372` in a column of `0.9999…` reads as a catastrophic failure, which is exactly
        how review round 5 found it.
        """

        def build(_lines):
            body = "; ".join(
                f"{name} tail std {'/'.join(re.findall(r'tail std=([\d.]+)', ' '.join(ls)))}"
                for ls, name in (
                    ([ln for ln in cap[test] if "layer=0" in ln], "linear"),
                    ([ln for ln in cap[test] if "layer=3" in ln], "full"),
                )
            )
            return f"completes, finite, non-degenerate — {body} (no PCC: no tractable golden at this length)"

        return build

    rows = [
        (
            "M-RoPE reduces to 1-D partial RoPE, **in both lowerings**",
            "`test_rope_matches_hf` (`seq_len` 1 / 64 at start offset 12345, 4096 at 0; `rope_mode` `partial` and `full`)",
            rope,
        ),
        (
            "the two RoPE lowerings are interchangeable",
            "`test_rope_mode_equivalence`",
            lambda ls: "prefill {}, decode {}".format(*_pccs(ls)),
        ),
        ("batch 4 prefill + batched decode", "`test_batched_prefill_decode_pcc[4]`", batched(4)),
        ("batch 32 prefill + batched decode", "`test_batched_prefill_decode_pcc[32]`", batched(32)),
        (
            "batched decode with **distinct per-user positions** over a shuffled disjoint page table",
            "`test_batched_decode_ragged_positions[4]` / `[13]`",
            ragged,
        ),
        (
            "decode batch past the dedicated ops' limits, so both fallbacks run",
            "`test_decode_batch_above_head_split_limit[40]` / `[56]`",
            above_limit,
        ),
        (
            "a batch **smaller** than the allocated one — the per-user-prefill serving pattern",
            "`test_batch_smaller_than_allocated_state`",
            smaller,
        ),
        ("shuffled, offset page table", "`test_permuted_page_table`", permuted),
        (
            "chunked-prefill continuation (2 calls, `start_pos > 0`) equals one call",
            "`test_prefill_continuation`",
            lambda ls: _by_layer(ls, lambda x: _pccs(x)[0]),
        ),
        (
            "decode under captured/replayed trace, PCC measured **from the replay**",
            "`test_traced_decode_pcc`",
            lambda ls: _by_layer(ls, lambda x: " / ".join(_pccs(x))),
        ),
        ("real checkpoint weights", "`test_real_weights_pcc`", lambda ls: _by_layer(ls, pair)),
        (
            "synthetic weights from recorded real statistics (the CI path)",
            "`test_synthetic_weights_pcc`",
            lambda ls: _by_layer(ls, pair),
        ),
        (
            "correctness independent of freed-DRAM contents",
            "`test_forward_with_poisoned_free_pool` (`seq_len` 1 / 250 / 300)",
            poisoned,
        ),
        (
            "determinism on repeated identical inputs",
            "`test_determinism_repeated_inputs`",
            determinism,
        ),
        (
            "repeated-run stress",
            "`test_repeated_run_stress`",
            stress,
        ),
        (
            "`max_context` not a multiple of the alignment",
            "`test_unaligned_max_context` (5000)",
            std_tail("test_unaligned_max_context"),
        ),
        (
            "advertised context — prefill + decode at the last legal slot (262143), prefill at the full 262144",
            "`test_full_context_prefill_and_decode[262143]` / `[262144]`",
            std_tail("test_full_context_prefill_and_decode"),
        ),
        (
            "full-context result independent of the internal chunking",
            "`test_full_context_chunk_size_invariance`",
            lambda ls: "262144-token prefill under chunk 2048 vs 1024: tail PCC **{}**, both kinds".format(
                _span(re.findall(r"=([\d.]+)$", "\n".join(ls)))
            ),
        ),
        (
            "no host fallback in a measured pass",
            "`test_no_host_fallback_in_forward`",
            host_fallback,
        ),
    ]
    body = ["| Property | Test | Result |", "| --- | --- | --- |"]
    for prop, test, extract in rows:
        name = re.search(r"`(test_[a-z_]+)", test)[1]
        body.append(f"| {prop} | {test} | {extract(cap.get(name, []))} |")
    return "\n".join(body)


def cell(before, after):
    if before is None or after is None:
        return "—"
    return f"{before} → **{after}**"


def _moves(before, after, keys):
    """``[(key, before, after)]`` for every key present in both summaries."""
    return [(k, float(before[k]), float(after[k])) for k in keys if k in before and k in after]


def _ceil_1sig(value):
    """Round a small positive delta *up* to one significant figure, as `Ne-M`.

    The narrative claims "at most X"; rounding up keeps that claim true.
    """
    if value <= 0:
        return "0"
    exp = 0
    scaled = value
    while scaled < 1:
        scaled *= 10
        exp += 1
    digit = int(scaled) + (1 if scaled > int(scaled) else 0)
    if digit == 10:
        digit, exp = 1, exp - 1
    return f"{digit}e-{exp}"


def reduce_pccs():
    """The two expert-axis reduction PCCs the §2.2 narrative attributes the largest move to."""
    text = (L / "probe_router_and_reduce.txt").read_text(errors="replace")
    got = dict(re.findall(r"REDUCE (\S+)\s+pcc=([\d.]+)", text))
    try:
        return got["deepseek_moe_fast_reduce_nc"], got["fast_reduce_nc"]
    except KeyError as exc:  # noqa: TRY003 - the probe format changed; fail loudly rather than quote a stale pair
        raise SystemExit(f"probe_router_and_reduce.txt: missing REDUCE row {exc}") from exc


def blocks():
    reduce_new, reduce_old = reduce_pccs()
    fp, fd = scrape(FUSED)
    bp, bd = scrape(FUNCTIONAL)
    equiv = scrape_equivalence(FUSED)
    out = {}

    rows = [
        "| `seq_len` | boundary role | `linear_attention` before → after | `full_attention` before → after |",
        "| --- | --- | --- | --- |",
    ]
    for seq in sorted({s for (_, s) in list(fp) + list(bp)}):
        rows.append(
            f"| {seq} | {ROLE.get(seq, '')} | {cell(bp.get((0, seq)), fp.get((0, seq)))} "
            f"| {cell(bp.get((3, seq)), fp.get((3, seq)))} |"
        )
    out["prefill-pcc"] = "\n".join(rows)

    rows = [
        "| prefill_len | step | `linear_attention` before → after | `full_attention` before → after |",
        "| --- | --- | --- | --- |",
    ]
    for plen, step in sorted({(p, s) for (_, p, s) in list(fd) + list(bd)}):
        rows.append(
            f"| {plen} | {step} | {cell(bd.get((0, plen, step)), fd.get((0, plen, step)))} "
            f"| {cell(bd.get((3, plen, step)), fd.get((3, plen, step)))} |"
        )
    out["decode-pcc"] = "\n".join(rows)

    # The delta narrative, computed from exactly the rows above.
    pre_moves = _moves(bp, fp, sorted(set(bp) & set(fp)))
    dec_moves = _moves(bd, fd, sorted(set(bd) & set(fd)))
    max_pre = max(abs(a - b) for _, b, a in pre_moves)
    max_dec = max(abs(a - b) for _, b, a in dec_moves)
    worst = min(min(a for _, _, a in pre_moves), min(a for _, _, a in dec_moves))
    (bl, bp_len, bstep), before_v, after_v = max(dec_moves, key=lambda m: m[2] - m[1])
    # Decades between the largest row *move* and the worst row's distance to the bar, derived rather
    # than asserted in prose. Note this is a move-vs-margin ratio, not the PCC error itself: the worst
    # row sits ~1.7 decades inside the bar, which is a different and smaller number.
    gap = worst - BAR
    dec_lo, dec_hi = sorted((math.log10(gap / max_pre), math.log10(gap / max_dec)))
    out["pcc-delta"] = (
        f"**Material delta: none.** Every prefill row moves by at most `{_ceil_1sig(max_pre)}` and every\n"
        f"decode row by at most `{_ceil_1sig(max_dec)}`, against a distance from the worst row "
        f"({worst:.6f}) to the\n"
        f"{BAR} bar of `{_ceil_1sig(gap)}` — the *largest* prefill and decode moves are respectively\n"
        f"{dec_hi:.1f} and {dec_lo:.1f} orders of magnitude smaller than that margin, and every other move is\n"
        f"smaller still — and the moves go\n"
        f"both ways. The largest single change is an *improvement* (`{KIND[bl]}`, prefill {bp_len},\n"
        f"decode step {bstep}: {before_v:.6f} → {after_v:.6f}). The stage's one accuracy-relevant change is\n"
        f"the expert-axis reduction switching to `deepseek_moe_fast_reduce_nc`, whose accumulation is\n"
        f"measurably more accurate than `fast_reduce_nc`'s (PCC {reduce_new} vs {reduce_old} against a\n"
        f"float32 sum of 256 bfloat16 expert blocks —\n"
        f"[`logs/probe_router_and_reduce.txt`](logs/probe_router_and_reduce.txt)); that is the plausible\n"
        f"source, but no artifact here isolates this row's move to it."
    )

    rows = [
        "| `seq_len` | `linear_attention` prefill / decode | `full_attention` prefill / decode |",
        "| --- | --- | --- |",
    ]
    for seq in sorted({s for (_, s) in equiv}):
        cells = []
        for layer in (0, 3):
            pair = equiv.get((layer, seq))
            cells.append(f"{pair[0]} / {pair[1]}" if pair else "—")
        rows.append(f"| {seq} | {cells[0]} | {cells[1]} |")
    out["equivalence-pcc"] = "\n".join(rows)
    cap = capability_lines(FUSED)
    out["capability-table"] = capability_rows(cap)
    out["equivalence-bar"] = equivalence_bar(cap, equiv)
    out["layout-budget"] = layout_budget(cap)
    out["conv1d-coverage"] = conv1d_coverage()
    out["conv1d-coverage-cause"] = conv1d_coverage_cause()
    out["watcher-result"] = watcher_result()
    out["watcher-subset"] = watcher_subset()
    out["conv1d-forward-risk"] = conv1d_forward_risk()
    return out


def equivalence_bar(cap, equiv):
    """§2.3's characterisation of the 0.9999 equivalence bar, computed rather than asserted.

    Round 11 rewrote this sentence to fix a false comparison and introduced two new errors in doing
    so — a count and an order of magnitude, both hand-written. Both are now derived: the count of
    HF-golden agreements below the bar, and the margin between the bar and the worst measured
    equivalence.
    """
    hf = [
        float(v)
        for test, lines in cap.items()
        if test != "test_fused_matches_functional"
        for line in lines
        for v in re.findall(r"PCC[= ]([\d.]+)", line)
    ]
    assert hf, "no HF-golden PCCs found in the summary"
    below = sorted(v for v in hf if v < EQUIV_BAR)
    worst_equiv = min((v for pair in equiv.values() for v in pair), key=float)
    bar_margin = 1 - EQUIV_BAR
    equiv_margin = 1 - float(worst_equiv)
    decades = math.log10(bar_margin / equiv_margin) if equiv_margin else float("inf")
    plural = "" if len(below) == 1 else "s"
    return (
        f"state dict and drives them with the same inputs and page table. The bar is {EQUIV_BAR:g} — an\n"
        f"order of magnitude tighter than the {BAR} acceptance bar, though not tighter than every measured\n"
        f"HF-golden agreement: {len(below)} of them sit below it, the lowest at {below[0]}. What makes this\n"
        "the stricter check is not the number but the comparison — both sides share dtypes, device and\n"
        f"page table, so the only difference left is the graph. The measured equivalences are {worst_equiv}\n"
        f"–1.000000, i.e. the worst is {decades:.1f} decades inside the bar."
    )


def layout_budget(cap):
    """README §6's layout-op budget, from what `test_no_layout_churn_in_measured_forward` logged."""
    rows = {}
    for line in cap.get("test_no_layout_churn_in_measured_forward", []):
        m = re.search(
            r"layer=(?P<layer>\d+) \((?P<kind>\w+)\) seq_len=(?P<seq>\d+) "
            r"prefill=(?P<pre>\d+) (?P<pred>\{[^}]*\}) decode=(?P<dec>\d+) (?P<decd>\{[^}]*\})",
            line,
        )
        if m:
            rows[(m["kind"], int(m["seq"]))] = (m["pre"], m["dec"], m["pred"], m["decd"])
    assert rows, "no layout-op lines in the summary"
    what = {
        "linear_attention": (
            "prefill: 4 conv ROW_MAJOR conversions (3 state buffers + the QKV stream) + 2 "
            "`sharded_to_interleaved` for the two `ttnn.conv1d` halves + 2 `to_layout` calls on their output "
            "that dispatch nothing (conv1d already returns TILE) + 3 conv-history row "
            "writebacks + **one MoE group mask per MoE call** (not per expert group — work_log §4.16). "
            "decode: the MoE group mask only"
        ),
        "full_attention": (
            "prefill: 2 RoPE-table `to_layout` tilizes + one MoE group mask per MoE call. decode: 3 "
            "`sharded_to_interleaved` off `nlp_create_qkv_heads_decode` + 2 height-shards for the fused "
            "cache update + 1 MoE group mask"
        ),
    }
    body = ["| Layer kind | prefill @256 | prefill @2048 | decode | what they are |", "| --- | --- | --- | --- | --- |"]
    for kind in ("linear_attention", "full_attention"):
        at256, at2048 = rows[(kind, 256)], rows[(kind, 2048)]
        assert at256[1] == at2048[1], f"{kind} decode budget differs by sequence length, which it must not"
        # Round 22: the itemisation above said the MoE term scaled with the sequence while the
        # measured counts did not, and nothing compared the two. Since §4.16 hoisted that mask to one
        # per call, NO term scales - so assert that directly, and the prose cannot drift from it
        # again without failing here.
        assert at256[0] == at2048[0], (
            f"{kind} prefill layout ops differ between seq 256 and 2048 ({at256[0]} vs {at2048[0]}): "
            "the itemisation in this generator says no term scales with the sequence"
        )
        body.append(f"| `{kind}` | {at256[0]} | {at2048[0]} | {at256[1]} | {what[kind]} |")
    return "\n".join(body)


def conv1d_coverage():
    """README §2's per-batch `ttnn.conv1d` coverage, from the criticals classifier's report."""
    path = L / "suite_criticals.txt"
    if not path.is_file():
        raise SystemExit("logs/suite_criticals.txt is missing; run logs/classify_suite_criticals.py")
    rows = re.findall(r"batch\s+(\d+):\s+(\d+)/(\d+) block lengths accepted", path.read_text())
    assert rows, "suite_criticals.txt has no coverage rows"
    # The suite also builds decoders with a non-default prefill_chunk (test_prefill_continuation uses
    # 128, so one candidate length instead of 16). Those are a different denominator and would make
    # the table's batch-1 row depend on log order, so only the default-chunk rows are tabulated.
    widest = max(int(total) for _, _, total in rows)
    rows = [r for r in rows if int(r[2]) == widest]
    note = {
        True: "every block uses `ttnn.conv1d`; this is the measured configuration in §5",
        False: "the rest use the FIR fallback",
    }
    body = ["| allocated batch | prefill block lengths `ttnn.conv1d` accepts | what runs |", "| --- | --- | --- |"]
    for batch, got, total in rows:
        if got == "0":
            text = "entirely FIR — correct, just not accelerated"
        else:
            text = note[got == total]
        body.append(f"| {batch} | {got} / {total} | {text} |")
    return "\n".join(body)


def conv1d_coverage_cause():
    """Why coverage falls with the batch — read from the classifier's per-batch refusal split.

    Review round 13 found five places attributing the whole fall to the conv's circular buffers.
    That is the *minority* class at every batch above 4 (1 of 16 refusals at batch 32); the dominant
    one is a sharded-tensor bank allocation refusal. The split is measured, so it is quoted.
    """
    text = (L / "suite_criticals.txt").read_text(errors="replace")
    rows = re.findall(
        r"batch\s+(\d+):\s+(\d+) refusals of (\d+)\s+conv-probe-l1-buffer=(\d+)\s+conv-probe-cb-overflow=(\d+)",
        text,
    )
    assert rows, "suite_criticals.txt has no per-batch refusal split"
    parts = ", ".join(f"batch {b}: {l1} / {cb}" for b, _, _, l1, cb in rows)
    # Which class dominates is batch-dependent, so say so per batch rather than generalising from one
    # row: round 14 found this sentence asserting the bank class dominates, which is false at batch 4.
    bank = [r for r in rows if int(r[3]) > int(r[4])]
    cbuf = [r for r in rows if int(r[4]) >= int(r[3])]
    assert bank and cbuf, f"one class dominates at every batch; the split sentence needs rewriting: {rows}"
    span = lambda rs: "/".join(r[0] for r in rs)  # noqa: E731
    top = max(bank, key=lambda r: int(r[3]))
    return (
        "Coverage falls as the batch rises, and the refusals split between **two different L1 limits**,\n"
        "which the classifier counts per batch (bank-allocation / circular-buffer): "
        f"{parts}.\n"
        f"Which limit dominates depends on the batch: the program-build circular-buffer set is the\n"
        f"majority at batch {span(cbuf)}, but from batch {min(int(r[0]) for r in bank)} up it is the sharded input's\n"
        f"**per-bank allocation** — {top[3]} of {top[1]} refusals at batch {top[0]} — which grows\n"
        "with the batch because the conv shards its activation over the same cores. That matters for what\n"
        "a future stage would change: at the batches where coverage actually collapses, a program-config\n"
        "or CB-sizing fix addresses only the smaller half.\n"
        "This is a *performance* fallback — every batch produces a result that clears the PCC bar, and\n"
        "§2.4's batched rows are measured on whichever path that batch selected — but the two paths are\n"
        "not bit-identical to each other, which §8 item 7 records. Only the `linear_attention` batches\n"
        "have a conv path at all; the `full_attention`-only rows (13, 40, 56) are unaffected by this."
    )


def conv1d_forward_risk():
    """README §8 item 3's bound on the in-forward `ttnn.conv1d` risk, from the measured split.

    Round 16 found this stated as "at the batches where coverage collapses most of them are the second
    kind", which is false at batch 8 (9 of 10 are the pressure-dependent kind). Since the sentence
    exists to *bound* the risk, understating it is the wrong direction to be wrong in, so it is
    generated from the per-batch split the classifier now prints.
    """
    text = (L / "suite_criticals.txt").read_text(errors="replace")
    rows = re.findall(r"batch\s+(\d+): pressure-dependent=\s*(\d+)\s+hard-overflow=\s*(\d+)", text)
    assert rows, "suite_criticals.txt has no per-batch hard/soft split"
    parts = ", ".join(f"batch {b}: {soft} / {hard}" for b, soft, hard in rows)
    worst = max(rows, key=lambda r: int(r[1]))
    soft_total = sum(int(r[1]) for r in rows)
    hard_total = sum(int(r[2]) for r in rows)
    return (
        f"Of the {soft_total + hard_total} bank-allocation refusals, {soft_total} are pressure-dependent —\n"
        f"the per-bank share would fit an empty bank, so whether they refuse depends on what else is\n"
        f"resident — and {hard_total} are hard overflows no amount of free space would satisfy. Only the\n"
        f"first kind can behave differently between the setup probe and a forward pass, and the split is\n"
        f"batch-dependent (pressure-dependent / hard): {parts}. The risk is therefore **largest at batch\n"
        f"{worst[0]}**, where {worst[1]} of {int(worst[1]) + int(worst[2])} are the kind that can differ, "
        f"not at the batch where\ncoverage is worst."
    )


def watcher_result():
    """README §7's watcher figures, from the census the watcher run produced."""
    census = (L.parent / "watcher/census_summary.txt").read_text(errors="replace")
    total = re.search(r"^\s*(\d+)\s+TOTAL", census, re.M)
    fatal = re.search(r"fatal-class matches: (\d+)", census)
    log = (L / "watcher_pytest.txt").read_text(errors="replace")
    tail = None
    for line in log.splitlines():
        m = re.search(r"=+ (\d+) passed,.*? in ([\d.]+)s", line)
        if m:
            tail = m
    assert total and fatal and tail, "watcher evidence is incomplete"
    stack = "carries no stack-headroom evidence" if "not reported in this log" in census else "reports stack headroom"
    # Thin space between thousands, matching how the rest of the document writes large counts.
    lines = f"{int(total[1]):,}".replace(",", " ")
    return (
        f"**{tail[1]} passed** in {tail[2]} s. The {lines} lines of the watcher log are fully "
        f"accounted for by a disjoint census summing exactly to {lines}, and a fatal-class grep "
        f"(asserts, invalid NOC coordinates or addresses, CB out-of-bounds, L1/stack overflow, sanitizer, "
        f"corruption, hang/deadlock) returns **{fatal[1]} fatal-class matches**. The log {stack}."
    )


def watcher_subset():
    """README §7's description of *which* tests ran under the watcher, read off the console log.

    Round 15 found this stated as a rule — "every path that writes a cache or state buffer in place,
    replays a trace, or is specific to this stage" — while the hand-written list behind it omitted
    several tests that satisfy the rule, including that round's own use-after-free regression test.
    The filter was widened to match the rule; this generates the list from what actually ran, so the
    two cannot drift apart again.
    """
    log = (L / "watcher_pytest.txt").read_text(errors="replace")
    names = []
    for m in re.finditer(r"::(test_\w+)", log):
        if m[1] not in names:
            names.append(m[1])
    assert names, "watcher_pytest.txt lists no test node ids"
    counts = re.search(r"(\d+) passed", log)
    deselected = re.search(r"(\d+) deselected", log)
    assert counts, "watcher_pytest.txt has no pass count"
    # Round 16: generating the list from the run makes it accurate about what ran but enforces nothing
    # about the *rule*, and it found qualifying tests still outside the subset. So the rule is stated as
    # what it is - a chosen subset - and the tests that must not silently leave it are asserted by name.
    REQUIRED = (
        "test_repeated_prefill_at_a_masked_chunk_length",  # the use-after-free regression
        "test_batched_paged_fill_is_one_launch_per_cache",  # the fused batched cache fill
        "test_decode_batch_above_head_split_limit",  # both cache-update fallbacks
        "test_traced_decode_pcc",  # trace capture and replay
        "test_determinism_repeated_inputs",
        "test_repeated_run_stress",
        "test_permuted_page_table",
        "test_forward_with_poisoned_free_pool",
    )
    missing = [n for n in REQUIRED if n not in names]
    assert not missing, f"the watcher subset no longer covers {missing}; widen the -k filter"
    listed = "".join(f"\n* `{n}`" for n in sorted(names))
    extra = f", with {deselected[1]} cases deselected" if deselected else ""
    return (
        f"The subset is a chosen one, not an exhaustive sweep: it covers every in-place cache/state\n"
        f"writer, trace replay and fused-only rewrite that this stage introduced or changed, and it is\n"
        f"listed from the run's own console log rather than described. Tests that drive prefill or decode\n"
        f"purely to check numerics (the PCC ladders, the weight-source cases, the full-context cases) are\n"
        f"deliberately out — they exercise no memory pattern the tests below do not. The generator\n"
        f"asserts by name that the load-bearing ones are present, so widening the filter cannot silently\n"
        f"drop them. {len(names)} test functions, {counts[1]} cases{extra}:\n{listed}"
    )


def splice(text, generated):
    for name, body in generated.items():
        open_m, close_m = f"<!-- generated:{name} -->", f"<!-- /generated:{name} -->"
        pattern = re.compile(re.escape(open_m) + r".*?" + re.escape(close_m), re.S)
        if not pattern.search(text):
            raise SystemExit(f"README.md has no '{name}' generated block ({open_m} ... {close_m})")
        text = pattern.sub(lambda _m, b=body, o=open_m, c=close_m: f"{o}\n{b}\n{c}", text, count=1)
    return text


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    generated = blocks()
    if mode == "--write":
        current = README.read_text()
        updated = splice(current, generated)
        README.write_text(updated)
        print("README.md unchanged" if updated == current else "README.md updated")
        return 0
    if mode == "--check":
        current = README.read_text()
        if splice(current, generated) != current:
            print(
                "STALE-README README.md generated blocks do not match logs/pcc_summary.txt; "
                "re-run make_readme_tables.py --write"
            )
            return 1
        print("README.md generated PCC blocks match the current summaries")
        return 0
    for name, body in generated.items():
        print(f"<!-- generated:{name} -->\n{body}\n<!-- /generated:{name} -->\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

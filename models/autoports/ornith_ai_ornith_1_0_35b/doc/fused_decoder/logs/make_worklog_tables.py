# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate the work log's probe-measurement table from the committed probe logs.

Third and last of the table generators, and the one that closes the loop. `make_readme_tables.py`
and `make_readme_perf.py` took the README's numbers out of human hands after review round 3 found
eleven stale PCC cells — but the *work log* still quoted per-probe figures inline, and the very next
evidence re-run made twenty of them stale. Every one of those figures moves a little on every run, so
none of them belongs in prose.

They live in one generated table instead (work log §4.12), and §3/§4 reference it. Prose that says
"slower, and rejected on latency" stays true across runs; prose that says "120.1 µs vs 103.8 µs" does
not.

    python .../logs/make_worklog_tables.py            # print the generated block
    python .../logs/make_worklog_tables.py --write     # splice into work_log.md
    python .../logs/make_worklog_tables.py --check     # exit 1 if work_log.md is out of date
"""

import re
import sys
from pathlib import Path

L = Path(__file__).resolve().parent
WORKLOG = L.parent / "work_log.md"
MARKER = "worklog-probe-figures"


def grab(path: Path, pattern: str, *groups):
    """First match of ``pattern`` in ``path``, as a tuple of the named groups."""
    m = re.search(pattern, path.read_text(errors="replace"))
    if not m:
        raise SystemExit(f"{path.name}: no match for {pattern!r} - the probe output format changed")
    return tuple(m[g] for g in groups)


def rows():
    fused = L / "probe_fused_ops.txt"
    micro = L / "probe_decode_micro.txt"
    router = L / "probe_router_and_reduce.txt"
    conv = L / "probe_conv1d_and_norm.txt"
    tail = L / "probe_conv_tail.txt"
    rope = L / "ab_rope_mode.txt"

    (silu_pcc,) = grab(fused, r"sparse_matmul fused_activation=SILU: pcc\(silu\(plain\) vs fused\)=(?P<p>[\d.]+)", "p")
    concat, permute = grab(
        micro,
        r"CONCATHEADS seq=\s*2048\s+nlp_concat_heads=\s*(?P<a>[\d.]+) us\s+permute\+reshape=\s*(?P<b>[\d.]+) us",
        "a",
        "b",
    )
    (scatter,) = grab(router, r"ROUTER scatter .*wall=(?P<w>[\d.]+) us/call", "w")
    ch1, pr1, ur1 = grab(
        micro,
        r"CONCATHEADS seq=\s*1\s+nlp_concat_heads=\s*(?P<a>[\d.]+) us\s+permute\+reshape=\s*(?P<b>[\d.]+) us"
        r".*?untilize/reshape/tilize=\s*(?P<c>[\d.]+) us",
        "a",
        "b",
        "c",
    )
    # The exact-equality verdicts the probes now print, rather than "bit-identical" inferred from a
    # PCC rounded to six decimals — which is what review round 12 found these cells doing.
    micro_text = micro.read_text(errors="replace")
    seq1 = next((ln for ln in micro_text.splitlines() if "CONCATHEADS seq=    1" in ln), "")
    verdicts = set(re.findall(r"pcc=[\d.]+ (\S+?)[);]", seq1))
    exact1 = (
        ("all three bitwise-equal" if verdicts == {"bitwise-equal"} else f"exact-equality verdicts: {sorted(verdicts)}")
        if verdicts
        else "no exact-equality verdict in the probe log"
    )
    tail_verdicts = re.findall(r"CONVTAIL tile tail\s+.*?pcc_vs_shipped=[\d.]+ (\S+)", tail.read_text(errors="replace"))
    tail_exact = f"tile-tail variant vs shipped: {tail_verdicts[-1]}" if tail_verdicts else "no verdict"
    flat_pccs = re.findall(r"untilize/reshape/tilize=\s*[\d.]+ us \(pcc=([\d.]+) \S+; only equivalent", micro_text)
    if len(flat_pccs) < 3:
        raise SystemExit("probe_decode_micro.txt: expected three CONCATHEADS rows")
    state_default, state_grid = grab(
        micro, r"STATEREAD\s+default=\s*(?P<a>[\d.]+) us\s+core_grid=\s*(?P<b>[\d.]+) us", "a", "b"
    )
    outer_t, outer_a = grab(
        micro, r"OUTER\s+transpose\+matmul=\s*(?P<a>[\d.]+) us\s+matmul\(transpose_a\)=\s*(?P<b>[\d.]+) us", "a", "b"
    )
    (where,) = grab(router, r"ROUTER where\s+.*wall=(?P<w>[\d.]+) us/call", "w")
    (conv1d_ms,) = grab(conv, r"CONV1DTIME conv1d x2 @4096ch\s+(?P<t>[\d.]+) ms", "t")
    # "shipped-fallback", not "4-tap": round 14 found the probe timing a generic FIR (ttnn.mac, all
    # four taps sliced) rather than the fallback that actually ships, which overstated the arm.
    (fir_ms,) = grab(conv, r"CONV1DTIME shipped-fallback FIR @8192ch\s+(?P<t>[\d.]+) ms", "t")
    act_plain_pcc, act_plain_ms = grab(
        conv,
        r"CONV1DACT conv1d \+ separate silu \(shipped\)\s+pcc=(?P<p>[\d.]+)\s+(?P<t>[\d.]+) ms",
        "p",
        "t",
    )
    act_fold_pcc, act_fold_ms = grab(
        conv,
        r"CONV1DACT conv1d\(activation=silu\) folded\s+pcc=(?P<p>[\d.]+)\s+(?P<t>[\d.]+) ms",
        "p",
        "t",
    )
    gate_sep, gate_fold = grab(
        fused,
        r"GATEFOLD prefill f32xbf16\(real\).*?separate\(shipped\) pcc=(?P<a>[\d.]+) nonfinite=0 \| "
        r"folded pcc=\S+ nonfinite=(?P<b>\d+)",
        "a",
        "b",
    )
    gate_bf16, gate_bf16_fold = grab(
        fused,
        r"GATEFOLD prefill bf16xbf16.*?separate\(shipped\) pcc=(?P<a>[\d.]+) nonfinite=0 \| "
        r"folded pcc=(?P<b>[\d.]+) nonfinite=0",
        "a",
        "b",
    )
    hoist_per, hoist_all = grab(
        router,
        r"MASKHOIST per-group \(superseded\)\s+(?P<a>[\d.]+) ms[\s\S]*?"
        r"MASKHOIST hoisted whole-call \(shipped\)\s+(?P<b>[\d.]+) ms",
        "a",
        "b",
    )
    (rm_tail,) = grab(tail, r"CONVTAIL row-major tail \(shipped\)\s+best=\s*(?P<t>[\d.]+) ms", "t")
    (tile_tail,) = grab(tail, r"CONVTAIL tile tail\s+best=\s*(?P<t>[\d.]+) ms", "t")
    norm_text = conv.read_text(errors="replace")
    norm = re.findall(r"RMSNORM (\S+(?: x\d+)?)\s+pcc=[\d.]+ wall=\s*([\d.]+) us/call", norm_text)
    if len(norm) < 2:
        raise SystemExit("probe_conv1d_and_norm.txt: no RMSNORM rows")
    norm_cell = ", ".join(f"{what} {us} µs" for what, us in norm)
    bench = rope.read_text(errors="replace")

    def rope_fig(mode, phase):
        pattern = (
            rf"rope_mode={mode} .*?prefill seq_len=\d+ wall=(?P<t>[\d.]+) ms"
            if phase == "prefill"
            else rf"rope_mode={mode} .*?decode\(traced\) iters=\d+ wall/iter=(?P<t>[\d.]+) ms"
        )
        m = re.search(pattern, bench)
        if not m:
            raise SystemExit(f"ab_rope_mode.txt: no {mode}/{phase} row")
        return m["t"]

    return [
        (
            '§3.1, §4.1 — `sparse_matmul` with `fused_activation=SILU`: PCC(`silu(plain)`, "fused")',
            f"**{silu_pcc}** — the activation is silently ignored",
            "`probe_fused_ops.txt`",
        ),
        (
            "§3.1 — `nlp_concat_heads` vs `permute + reshape`, prefill at seq 2048",
            f"{concat} µs vs {permute} µs",
            "`probe_decode_micro.txt`",
        ),
        (
            "§4.3 — router `scatter` (shipped) vs the threshold rewrite "
            "`topk -> ge(kth) -> where -> softmax(256)`, per decode call. "
            "(§4.2's `generalized_moe_gate` is a different candidate, rejected on bfloat16 accuracy "
            "and never timed.)",
            f"{scatter} µs vs {where} µs",
            "`probe_router_and_reduce.txt`",
        ),
        (
            "§3.1, §4.4 — `ttnn.conv1d` (2 × 4096 ch) vs the 4-tap FIR (8192 ch), 2048 tokens",
            f"{conv1d_ms} ms vs {fir_ms} ms",
            "`probe_conv1d_and_norm.txt`",
        ),
        (
            "§4.8 — DeltaNet output gate, SiLU separate (shipped) vs folded into the multiply, at the "
            "**real** `float32 x bfloat16` operand pairing, prefill shape",
            f"separate PCC {gate_sep}, 0 non-finite; folded **{gate_fold} non-finite values** at the "
            f"smallest magnitude tested",
            "`probe_fused_ops.txt`",
        ),
        (
            "§4.8 — the same fold with **matched** `bfloat16 x bfloat16` operands, i.e. what a naive "
            "op-level probe writes, and why it passes",
            f"separate PCC {gate_bf16} vs folded PCC {gate_bf16_fold}, both 0 non-finite",
            "`probe_fused_ops.txt`",
        ),
        (
            "§4.16 — MoE per-group mask + score-operand rebuild (superseded) vs one whole-call pair "
            "with per-group slices (shipped), 2048-token prefill",
            f"{hoist_per} ms vs {hoist_all} ms per MoE call",
            "`probe_router_and_reduce.txt`",
        ),
        (
            "§4.15 — SiLU applied separately (shipped) vs folded into `Conv1dConfig(activation=…)`, "
            "one 4096-channel depthwise call over 2048 tokens",
            f"PCC {act_plain_pcc} at {act_plain_ms} ms vs PCC {act_fold_pcc} at {act_fold_ms} ms — "
            f"the folded form is faster and **fails the 0.995 bar**",
            "`probe_conv1d_and_norm.txt`",
        ),
        (
            "§5 — conv history tail kept ROW_MAJOR (shipped) vs tilized, warmed 2048-token prefill",
            f"{rm_tail} ms vs {tile_tail} ms; {tail_exact}",
            "`probe_conv_tail.txt`",
        ),
        (
            "§4.5 — RMSNorm interleaved (shipped) vs width-sharded over 8/16/32/64 cores",
            norm_cell,
            "`probe_conv1d_and_norm.txt`",
        ),
        (
            "§3.3, §4.7 — decode head merge at `seq_len 1`: `permute + reshape` (shipped) vs "
            "`nlp_concat_heads` vs the flat untilize/reshape/tilize",
            f"{pr1} µs vs {ch1} µs vs {ur1} µs; {exact1}",
            "`probe_decode_micro.txt`",
        ),
        (
            "§4.7 — the flat untilize/reshape/tilize spelling above `seq_len 1` (it does not transpose "
            "head↔token, so it is not an alternative there at all)",
            f"PCC {flat_pccs[1]} at 128 and {flat_pccs[2]} at 2048 against the other two",
            "`probe_decode_micro.txt`",
        ),
        (
            "§3.2 — explicit `core_grid` on the recurrent-state read",
            f"{state_default} µs → {state_grid} µs",
            "`probe_decode_micro.txt`",
        ),
        (
            "§3.3 — delta-rule outer product: `transpose + matmul` vs `matmul(transpose_a=True)`",
            f"{outer_t} µs → {outer_a} µs",
            "`probe_decode_micro.txt`",
        ),
        (
            "§4.6 — `rope_mode` `partial` (shipped) vs `full`, traced decode",
            f"{rope_fig('partial', 'decode')} ms vs {rope_fig('full', 'decode')} ms",
            "`ab_rope_mode.txt`",
        ),
        (
            "§4.6 — `rope_mode` `partial` (shipped) vs `full`, 2048-token prefill",
            f"{rope_fig('partial', 'prefill')} ms vs {rope_fig('full', 'prefill')} ms",
            "`ab_rope_mode.txt`",
        ),
    ]


def block():
    out = ["| Comparison | Measured | Artifact |", "| --- | --- | --- |"]
    out += [f"| {what} | {value} | {where} |" for what, value, where in rows()]
    return "\n".join(out)


def splice(text, body):
    open_m, close_m = f"<!-- generated:{MARKER} -->", f"<!-- /generated:{MARKER} -->"
    pattern = re.compile(re.escape(open_m) + r".*?" + re.escape(close_m), re.S)
    if not pattern.search(text):
        raise SystemExit(f"work_log.md has no '{MARKER}' generated block ({open_m} ... {close_m})")
    return pattern.sub(lambda _m: f"{open_m}\n{body}\n{close_m}", text, count=1)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    body = block()
    if mode == "--write":
        current = WORKLOG.read_text()
        updated = splice(current, body)
        WORKLOG.write_text(updated)
        print("work_log.md unchanged" if updated == current else "work_log.md updated")
        return 0
    if mode == "--check":
        current = WORKLOG.read_text()
        if splice(current, body) != current:
            print("STALE-WORKLOG work_log.md probe table is out of date; re-run make_worklog_tables.py --write")
            return 1
        print("work_log.md generated probe table matches the current probe logs")
        return 0
    print(f"<!-- generated:{MARKER} -->\n{body}\n<!-- /generated:{MARKER} -->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

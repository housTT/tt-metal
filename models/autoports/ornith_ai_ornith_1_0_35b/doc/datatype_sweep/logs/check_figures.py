# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Re-derive every number this stage's README states, from the artifacts, and fail if one drifted.

The previous stage learned this the hard way (`doc/optimized_full_model/logs/check_prose_figures.py`):
a README figure that is typed rather than generated goes stale the moment a run is repeated. Every
assertion below reads a JSON artifact and compares it against the value the prose claims.

    python .../doc/datatype_sweep/logs/check_figures.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

D = Path("models/autoports/ornith_ai_ornith_1_0_35b/doc/datatype_sweep")
GIB = 2**30
failures: list[str] = []


def chk(name, got, want, tol=0.0):
    ok = abs(got - want) <= tol if isinstance(want, (int, float)) and not isinstance(want, bool) else got == want
    print(f"{'OK ' if ok else 'BAD'}  {name:<38} got {got!r} want {want!r}")
    if not ok:
        failures.append(f"{name}: got {got!r}, README says {want!r}")


def main():
    sr = json.loads((D / "sweep_results.json").read_text())
    rows = {r["config_id"]: r for r in sr["results"]}
    c6, s0 = rows["C06-proj-bfp4-lofi"], rows["S00-baseline-optimized"]

    # ---- section 1: the selected config and its numbers ----
    chk("selected config id", sr["selected"], "C06-proj-bfp4-lofi")
    chk("configurations evaluated", len(sr["results"]), 24)
    chk("selected prefill top-1", c6["prefill_top1"], 0.920)
    chk("selected teacher top-1", c6["teacher_top1"], 0.920)
    chk("baseline prefill top-1", s0["prefill_top1"], 0.940)
    chk("baseline teacher top-1", s0["teacher_top1"], 0.970)
    chk("every config top-5 == 1.000", all(r["top5"] == 1.0 for r in sr["results"]), True)
    chk("every config top-100 == 1.000", all(r["top100"] == 1.0 for r in sr["results"]), True)
    chk("selected teacher decode t/s/u", round(c6["teacher_decode_t_s_u"], 3), 42.296, 5e-4)
    chk("selected teacher decode ms", round(c6["teacher_decode_ms_per_token"], 3), 23.643, 5e-4)
    chk("selected teacher TTFT ms", round(c6["ttft_ms"], 1), 178.9, 0.05)
    chk("baseline teacher decode t/s/u", round(s0["teacher_decode_t_s_u"], 3), 41.962, 5e-4)
    chk("baseline teacher TTFT ms", round(s0["ttft_ms"], 1), 181.2, 0.05)
    chk(
        "teacher-forcing gain %",
        round((c6["teacher_decode_t_s_u"] / s0["teacher_decode_t_s_u"] - 1) * 100, 2),
        0.80,
        5e-3,
    )
    chk("failing configurations", sum(1 for r in sr["results"] if r["status"] == "fail"), 1)
    chk(
        "matrix range %",
        round(
            (
                max(r["teacher_decode_t_s_u"] for r in sr["results"])
                / min(r["teacher_decode_t_s_u"] for r in sr["results"])
                - 1
            )
            * 100,
            1,
        ),
        3.1,
        0.05,
    )
    chk("warm repeats per config", sorted({r["warm_repeats"] for r in sr["results"]}), [9])
    passing = [r for r in sr["results"] if r["status"] == "pass"]
    chk("min warm spread %, passing", round(min(r["teacher_decode_spread_pct"] for r in passing), 2), 0.18, 5e-3)
    chk("max warm spread %, passing", round(max(r["teacher_decode_spread_pct"] for r in passing), 2), 0.48, 5e-3)
    chk(
        "warm spread %, failing config",
        round(rows["C18-combo-plus-kv-bfp4"]["teacher_decode_spread_pct"], 2),
        0.70,
        5e-3,
    )
    chk("C06 within-build warm spread %", round(c6["teacher_decode_spread_pct"], 2), 0.26, 5e-3)
    chk(
        "C16 within-build warm spread %",
        round(rows["C16-combo-proj-lmhead-bfp4-lofi"]["teacher_decode_spread_pct"], 2),
        0.26,
        5e-3,
    )
    chk(
        "C06 and C16 differ by %",
        round(
            abs(c6["teacher_decode_t_s_u"] / rows["C16-combo-proj-lmhead-bfp4-lofi"]["teacher_decode_t_s_u"] - 1) * 100,
            3,
        ),
        0.023,
        5e-4,
    )

    # ---- section 1: post-selection token-out ----
    sel = json.loads((D / "post_selection_token_out.json").read_text())
    base = json.loads((D / "post_selection_token_out_baseline.json").read_text())
    chk("token-out policy is the selected one", sel["capability"]["policy"], "C06-proj-bfp4-lofi")
    chk("token-out baseline policy", base["capability"]["policy"], "optimized")
    chk("token-out selected t/s/u", round(sel["token_out_decode"]["t/s/u"], 3), 43.169, 5e-4)
    chk("token-out selected ms", round(sel["token_out_decode"]["ms_per_token"], 3), 23.165, 5e-4)
    chk("token-out baseline t/s/u", round(base["token_out_decode"]["t/s/u"], 3), 42.916, 5e-4)
    chk("token-out baseline ms", round(base["token_out_decode"]["ms_per_token"], 3), 23.301, 5e-4)
    chk(
        "token-out gain %",
        round((sel["token_out_decode"]["t/s/u"] / base["token_out_decode"]["t/s/u"] - 1) * 100, 2),
        0.59,
        5e-3,
    )
    chk("token-out TTFT median ms", round(sel["ttft_ms"]["median"], 1), 139.5, 0.05)
    chk("token-out TTFT min ms", round(sel["ttft_ms"]["min"], 1), 133.9, 0.05)
    chk("token-out decode_syncs", sel["steady_state_counters"]["decode_syncs"], 0)
    chk("token-out decode_calls", sel["steady_state_counters"]["decode_calls"], 127)
    chk(
        "baseline reproduces 23.300 to %",
        round(abs(base["token_out_decode"]["ms_per_token"] / 23.300 - 1) * 100, 3),
        0.005,
        5e-4,
    )

    # ---- section 2: the launch-bound argument ----
    bs = sel["performance_accounting"]["roofline_bytes_per_token_per_device"]
    bb = base["performance_accounting"]["roofline_bytes_per_token_per_device"]
    chk("selected bytes/token/device", bs, 569421184.0)
    chk("baseline bytes/token/device", bb, 749907328.0)
    chk("byte reduction %", round((1 - bs / bb) * 100, 0), 24.0, 0.5)
    chk(
        "selected roofline fraction %",
        round(sel["performance_accounting"]["roofline_fraction_achieved"] * 100, 2),
        4.80,
        5e-3,
    )
    chk(
        "baseline roofline fraction %",
        round(base["performance_accounting"]["roofline_fraction_achieved"] * 100, 2),
        6.28,
        5e-3,
    )
    for cid, want in (("C13-ccl-bfp8", -1.33), ("C14-residual-bfp8", -1.87), ("C10-experts-bfp4-hifi2", -2.20)):
        chk(f"{cid} delta %", round(rows[cid]["decode_speedup_vs_baseline_pct"], 2), want, 5e-3)

    # ---- section 1 / 4: the frontier ----
    chk("C23 top-1", rows["C23-proj-bfp4-lofi-lmhead-bfp8-lofi"]["top1"], 0.930)
    chk("C23 t/s/u", round(rows["C23-proj-bfp4-lofi-lmhead-bfp8-lofi"]["teacher_decode_t_s_u"], 3), 42.141, 5e-4)
    chk("C04 top-1", rows["C04-proj-bfp8-lofi"]["top1"], 0.960)
    chk("C04 t/s/u", round(rows["C04-proj-bfp8-lofi"]["teacher_decode_t_s_u"], 3), 42.081, 5e-4)
    chk(
        "C23 vs C06 %",
        round(
            (rows["C23-proj-bfp4-lofi-lmhead-bfp8-lofi"]["teacher_decode_t_s_u"] / c6["teacher_decode_t_s_u"] - 1)
            * 100,
            2,
        ),
        -0.37,
        5e-3,
    )
    chk(
        "C04 vs C06 %",
        round((rows["C04-proj-bfp8-lofi"]["teacher_decode_t_s_u"] / c6["teacher_decode_t_s_u"] - 1) * 100, 2),
        -0.51,
        5e-3,
    )
    chk(
        "C17 vs C06 %",
        round(
            (rows["C17-combo-bfp4-lofi-first-last-bfp8"]["teacher_decode_t_s_u"] / c6["teacher_decode_t_s_u"] - 1)
            * 100,
            2,
        ),
        -0.32,
        5e-3,
    )
    chk("C18 prefill top-1", rows["C18-combo-plus-kv-bfp4"]["prefill_top1"], 0.870)
    chk("C18 teacher top-1", rows["C18-combo-plus-kv-bfp4"]["teacher_top1"], 0.860)
    chk("C11 prefill top-1", rows["C11-kv-bfp4"]["prefill_top1"], 0.940)
    chk("C11 teacher top-1", rows["C11-kv-bfp4"]["teacher_top1"], 0.950)

    # ---- section 6: the built policy is the artifact's ----
    chk("C06 built LM head dtype", rows["C06-proj-bfp4-lofi"]["built_lm_head_weight_dtype"], "DataType.BFLOAT4_B")
    chk("C06 built LM head fidelity", rows["C06-proj-bfp4-lofi"]["built_lm_head_math_fidelity"], "MathFidelity.LoFi")
    chk("C06 resolved prefill SDPA chunk", rows["C06-proj-bfp4-lofi"]["prefill_sdpa_chunk"], 256)
    chk("baseline resolved prefill SDPA chunk", s0["prefill_sdpa_chunk"], 256)
    chk("every config trace-verified", all(r["trace_verified"] for r in sr["results"]), True)

    # ---- section 7: capacity ----
    cap = {Path(p).stem: json.loads(Path(p).read_text()) for p in glob.glob(str(D / "capacity" / "*.json"))}
    chk("capacity probes", sorted(cap), ["C11-kv-bfp4", "C12-kv-bf16", "optimized", "selected"])
    chk("every cache dtype fits 262144", all(v["capacity"]["advertised_context_fits"] for v in cap.values()), True)
    chk("selected free GiB", round(cap["selected"]["per_device_bytes"]["free_for_activations"] / GIB, 3), 24.377, 5e-4)
    chk(
        "optimized free GiB", round(cap["optimized"]["per_device_bytes"]["free_for_activations"] / GIB, 3), 24.165, 5e-4
    )
    chk(
        "selected weights GiB",
        round(cap["selected"]["per_device_bytes"]["weights_embedding_lm_head"] / GIB, 3),
        5.581,
        5e-4,
    )
    chk(
        "optimized weights GiB",
        round(cap["optimized"]["per_device_bytes"]["weights_embedding_lm_head"] / GIB, 3),
        5.794,
        5e-4,
    )
    chk(
        "free DRAM gain GiB",
        round(
            (
                cap["selected"]["per_device_bytes"]["free_for_activations"]
                - cap["optimized"]["per_device_bytes"]["free_for_activations"]
            )
            / GIB,
            2,
        ),
        0.21,
        5e-3,
    )
    chk(
        "bf16 cache costs GiB vs optimized",
        round(
            (
                cap["optimized"]["per_device_bytes"]["free_for_activations"]
                - cap["C12-kv-bf16"]["per_device_bytes"]["free_for_activations"]
            )
            / GIB,
            2,
        ),
        1.17,
        5e-3,
    )
    chk(
        "bf16 cache costs GiB vs selected",
        round(
            (
                cap["selected"]["per_device_bytes"]["free_for_activations"]
                - cap["C12-kv-bf16"]["per_device_bytes"]["free_for_activations"]
            )
            / GIB,
            2,
        ),
        1.38,
        5e-3,
    )

    # ---- section 7: non-aligned prompts ----
    lp = json.loads((D / "long_prompt.json").read_text())
    chk("longest non-aligned prompt", lp["largest_completed_non_aligned_prompt"], 262143)
    chk("every non-aligned length ok", all(r["status"] == "ok" for r in lp["results"]), True)
    chk("every non-aligned length finite", all(r["finite_logits"] and r["token_in_vocab"] for r in lp["results"]), True)

    # ---- section 8: qualitative ----
    q = json.loads((D / "qualitative_comparison.json").read_text())
    chk("qualitative prompts", q["n_prompts"], 6)
    chk("qualitative degenerate", q["any_degenerate"], False)
    chk("qualitative chat template", q["prompt_format"]["prompt_mode"], "chat")

    # ---- section 9.1: the near-tie ----
    tie = {
        t: json.loads((D / f"batch_slot_tie_{t}.json").read_text()) for t in ("selected", "baseline", "fused_parity")
    }
    chk("fused-parity slots agree", tie["fused_parity"]["all_repeats_agree_across_slots"], True)
    chk("baseline slots agree", tie["baseline"]["all_repeats_agree_across_slots"], True)
    chk("selected slots disagree", tie["selected"]["all_repeats_agree_across_slots"], False)
    r0 = tie["selected"]["rounds"][0]
    chk("selected slot-0 top1/top2 margin", r0["slots"][0]["top1_minus_top2"], 0.0)
    chk("selected max cross-slot diff", round(max(r0["max_abs_logit_diff_against_slot0"]), 3), 0.500, 5e-4)
    chk(
        "fused-parity max cross-slot diff",
        round(max(tie["fused_parity"]["rounds"][0]["max_abs_logit_diff_against_slot0"]), 3),
        0.109,
        1e-3,
    )
    chk("selected min PCC", round(min(r0["logit_pcc_against_slot0"]), 4), 0.9993, 5e-5)

    # ---- section 3: the official-runner control and the estimator ----
    import statistics

    base_run = json.loads((D / "runs" / "S00-baseline-optimized.json").read_text())
    ctl = base_run["official_runner_control"]
    warm_base = base_run["result"]["teacher_decode_t/s/u_warm_repeats"]
    chk("official runner prefill top-1", ctl["prefill"][0]["top1"], 0.940)
    chk("official runner teacher top-1", ctl["teacher"][0]["top1"], 0.970)
    chk("official runner teacher top-5", ctl["teacher"][0]["top5"], 1.0)
    chk("official runner decode t/s/u", round(ctl["teacher"][0]["decode_t/s/u"], 3), 41.737, 5e-4)
    chk("driver warm range low", round(min(warm_base), 3), 41.819, 5e-4)
    chk("driver warm range high", round(max(warm_base), 3), 41.962, 5e-4)
    chk(
        "official vs best-warm %",
        round((ctl["teacher"][0]["decode_t/s/u"] / max(warm_base) - 1) * 100, 3),
        -0.535,
        5e-4,
    )
    chk("driver warm spread %, baseline", round((max(warm_base) / min(warm_base) - 1) * 100, 2), 0.34, 5e-3)

    warm = {r["config_id"]: r["teacher_decode_warm_repeats"] for r in sr["results"]}
    warm = {k: json.loads(v) for k, v in warm.items()}
    order = {
        "best": lambda v: max(v),
        "median": lambda v: statistics.median(v),
        "mean": lambda v: statistics.mean(v),
        "worst": lambda v: min(v),
    }
    for est, fn in order.items():
        top4 = sorted(
            ((fn(warm[r["config_id"]]), r["config_id"]) for r in sr["results"] if r["status"] == "pass"),
            reverse=True,
        )[:4]
        chk(f"{est}-warm top four", [c.split("-")[0] for _, c in top4], ["C06", "C16", "C05", "C17"])
    chk("median-warm C06", round(statistics.median(warm["C06-proj-bfp4-lofi"]), 3), 42.265, 5e-4)
    chk("worst-warm C06", round(min(warm["C06-proj-bfp4-lofi"]), 3), 42.186, 5e-4)

    # ---- section 4 / artifacts: the 4-repeat pass as a second measurement ----
    first = {}
    for path in glob.glob(str(D / "runs_4repeat_firstpass" / "*.json")):
        rec = json.loads(Path(path).read_text())
        first[rec["config_id"]] = rec["result"]["teacher_decode_t/s/u"]
    deltas = sorted((abs(rows[k]["teacher_decode_t_s_u"] / first[k] - 1) * 100, k) for k in first if k in rows)
    chk("4-vs-10 repeat configs compared", len(deltas), 21)
    chk("4-vs-10 repeat max |delta| %", round(deltas[-1][0], 3), 0.172, 5e-4)
    chk("4-vs-10 repeat max is C18", deltas[-1][1], "C18-combo-plus-kv-bfp4")
    chk("4-vs-10 repeat median |delta| %", round(deltas[len(deltas) // 2][0], 3), 0.048, 5e-4)
    chk("4-vs-10 repeat inside 0.06 %", sum(1 for d, _ in deltas if d <= 0.06), 11)

    # ---- section 4.0: the step-6 extension ----
    c24 = rows["C24-union-of-every-non-negative-arm"]
    chk("C24 status", c24["status"], "pass")
    chk("C24 top-1", c24["top1"], 0.930)
    chk("C24 t/s/u", round(c24["teacher_decode_t_s_u"], 3), 42.126, 5e-4)
    chk("C24 vs C06 %", round((c24["teacher_decode_t_s_u"] / c6["teacher_decode_t_s_u"] - 1) * 100, 2), -0.40, 5e-3)

    # ---- section 6.1: the perf-report rows ----
    import re as _re

    def fidelity_rows(path):
        counts = {}
        for line in open(path, errors="ignore"):
            line = _re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = _re.search(r"(LoFi|HiFi2|HiFi4)\s+(\S+ x \S+ => \S+)", line)
            if m and "MatmulDeviceOperation" in line:
                counts[m.group(1) + " " + m.group(2)] = counts.get(m.group(1) + " " + m.group(2), 0) + 1
        return counts

    sel_rows = fidelity_rows(D / "tracy" / "decode_perf_report.txt")
    base_rows = fidelity_rows(D.parent / "optimized_full_model" / "tracy" / "decode_perf_report.txt")
    chk("baseline capture BFP8/HiFi2 dense rows", base_rows.get("HiFi2 BF16 x BFP8 => BF16"), 72)
    chk("baseline capture BFP4/LoFi dense rows", base_rows.get("LoFi BF16 x BFP4 => BF16", 0), 0)
    chk("selected capture BFP4/LoFi dense rows", sel_rows.get("LoFi BF16 x BFP4 => BF16"), 40)
    chk("selected capture BFP8/HiFi2 dense rows", sel_rows.get("HiFi2 BF16 x BFP8 => BF16"), 32)
    chk(
        "40 + 32 == the baseline's 72",
        sel_rows["LoFi BF16 x BFP4 => BF16"] + sel_rows["HiFi2 BF16 x BFP8 => BF16"],
        base_rows["HiFi2 BF16 x BFP8 => BF16"],
    )
    chk("router rows unchanged", sel_rows.get("HiFi4 BF16 x BF16 => BF16"), base_rows.get("HiFi4 BF16 x BF16 => BF16"))
    chk("state rows unchanged", sel_rows.get("HiFi4 FP32 x FP32 => FP32"), base_rows.get("HiFi4 FP32 x FP32 => FP32"))

    # ---- section 9.1: the negative control ----
    nm = json.loads((D / "batch_slot_tie_no_merge.json").read_text())
    nr = nm["rounds"][0]
    chk("no-merge control ran with the merge off", nm["negative_control_merge_disabled"], True)
    chk("no-merge batch-1 oracle unaffected", nm["batch1_expected_tokens"], [58573, 45568])
    chk("no-merge prefill still correct", set(nr["prefill_argmax_per_slot"]), {58573})
    chk("no-merge decode collapses", set(nr["decode_argmax_per_slot"]), {267})
    chk("no-merge min PCC", round(min(nr["logit_pcc_against_slot0"][1:]), 5), 0.99931, 5e-5)
    chk("no-merge max PCC", round(max(nr["logit_pcc_against_slot0"][1:]), 5), 0.99954, 5e-5)
    chk("no-merge min margin", round(min(s2["top1_minus_top2"] for s2 in nr["slots"]), 4), 1.6875, 5e-4)

    def contender_spread(record):
        return max(max(v) - min(v) for v in record["rounds"][0]["per_slot_contender_logits"].values())

    for tag, spread, margin, fires in (
        ("fused_parity", 0.0, 0.3125, True),
        ("baseline", 0.1875, 0.3125, True),
        ("no_merge", 0.125, 1.6875, True),
        ("selected", 0.1875, 0.0, False),
    ):
        rec = json.loads((D / f"batch_slot_tie_{tag}.json").read_text())
        got_spread = contender_spread(rec)
        got_margin = min(s2["top1_minus_top2"] for s2 in rec["rounds"][0]["slots"])
        chk(f"{tag} contender spread", round(got_spread, 4), spread, 5e-4)
        chk(f"{tag} min margin", round(got_margin, 4), margin, 5e-4)
        chk(f"{tag} strict branch fires", got_margin > got_spread, fires)

    # ---- section 9.1.1: batch 4 on the full stack ----
    for tag, want_top1, want_agree in (
        ("selected", [0.93, 0.92, 0.92, 0.94], [1.0, 0.99, 0.99, 0.99]),
        ("baseline", [0.94, 0.96, 0.95, 0.96], [1.0, 0.96, 0.97, 0.98]),
    ):
        b4 = json.loads((D / f"batch4_accuracy_{tag}.json").read_text())
        chk(f"batch4 {tag} is the full stack", (b4["layers"], b4["reduced"], b4["batch"]), (40, False, 4))
        chk(f"batch4 {tag} per-slot top-1", [p2["top1"] for p2 in b4["per_slot"]], want_top1)
        chk(f"batch4 {tag} per-slot top-5", [p2["top5"] for p2 in b4["per_slot"]], [1.0] * 4)
        chk(f"batch4 {tag} per-slot top-100", [p2["top100"] for p2 in b4["per_slot"]], [1.0] * 4)
        chk(f"batch4 {tag} agreement with slot 0", b4["slot_token_agreement_with_slot0"], want_agree)
    chk(
        "batch4 selected clears the top-1 gate on every slot",
        min(p2["top1"] for p2 in json.loads((D / "batch4_accuracy_selected.json").read_text())["per_slot"]) >= 0.90,
        True,
    )

    # ---- section 5.2: the blocked arm ----
    blocked = json.loads((D / "blocked" / "C19-expert-act-bfp4.json").read_text())
    chk("C19 blockers recorded", len(blocked["blockers"]), 2)
    chk("C19 status", blocked["status"], "blocked")

    print(f"\n{len(failures)} problem(s)")
    for f in failures:
        print("  " + f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

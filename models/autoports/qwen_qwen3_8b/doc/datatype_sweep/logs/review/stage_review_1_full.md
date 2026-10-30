# Stage Review (review 1, verbatim from the fresh read-only subagent, 12:39 to 12:52 ET)

Verdict: more-work-needed

The selected policy (`c9_lm_head_bfp4`, BFP4 LoFi LM head on the stage-7 policy) is consumed by the default construction path, proven by readback and perf-report rows, and the post-selection default-path chain is complete and coherent. The verdict is `more-work-needed` because the committed stage (`cd64cfe5aa`, `2a6cc0d845`, `ee31ae063d`) does not prove the headline contract item ("the fastest evaluated config that satisfies the gate"): two faster candidates are recorded as `pass`, the committed README says the opposite, and the evidence that resolves it was produced after the commits during this review (12:38 to 12:44 ET) and is uncommitted and undocumented. Two smaller contract and evidence gaps follow.

## Required Work

- P1: The committed stage records two candidates faster than the selection as `pass`, and the README contradicts its own results table and chart
  Evidence: At `ee31ae063d`, `sweep_results.json` and `sweep_results.csv` list `c8_attn_bfp4_inner` (48.82 t/s/u) and `c10_mlp_bfp4_inner_hifi2` (50.28 t/s/u) with `status: pass`, `main_reference_pass: true` and `robustness_references: {}`; the selected `c9_lm_head_bfp4` is 47.24 t/s/u. The acceptance rule in the same file passes any candidate that was not run on the robustness references. The committed `top1_perf_pareto.png` draws c8 and c10 green above the red c9 point. README section 2 says "Every point faster than the baseline family except c9 is grey" (false against the committed chart), section 3.1 rejects c8 with an unmeasured inference, section 7.2 says "Passing set: c0, c1, c3, c9", and section 7 lists c8 and c10 as having no natural-text row. During this review the stage owner ran the missing rows on commit `ee31ae063d` (12:38 to 12:44 ET): c8 natural text top-1 0.8965 (1316 of 1468 positions; the 0.90 bar needs 1322, so 6 short, inside the binomial noise of about 11 positions), top-5 0.9898; c8 aime_p1 PC/TF 0.97/0.97; c8 book32 PC/TF 0.95/0.95, top-5 0.99; c10 natural text top-1 0.8297, bit-identical to c5. `sweep_results.json`, `.csv` and both charts were regenerated at 12:44 ET and now mark c8 and c10 `fail`. All of these are untracked or modified and appear nowhere in `README.md` or `work_log.md`.
  Why this matters: The goal contract makes "fastest evaluated config that satisfies the gate" the central deliverable, and the stage-review standard names a "fastest evaluated config" claim that holds only because an evaluated candidate was not put through the gate. The committed artifacts support selecting c10 or c8, not c9. The new rows close the question, but a supervisor pushing `ee31ae063d` ships the contradictory state.
  Required next step: No device work. Record the c8 and c10 rows in `work_log.md` and in `README.md` sections 2, 3.1, 3.2, 3.3, 3.4 and 7 (c8: fails natural text by 6 positions at the bar, inside noise; c10: identical to c5 on natural text). Record why a c8+HiFi2 and a c8+c9 run are not needed: the failing reference is the prefill path, whose BFP4 fidelity is fixed at LoFi, and c9 lowers natural text by a further point. Commit the new sidecars, natural-text JSONs, `logs/natural_queue2.{sh,status}`, the two modified chain status files, the regenerated `sweep_results.json`, both charts, and `sweep_results.csv`. Then rereview.

- P2: Resumed-prefill gate: a new failing hit window under the selected policy is dismissed as a "near-tie flip" with no probe
  Evidence: `readiness/prefill_resume_check_stage8_default_thresholds.json` (default path) and `prefill_resume_check_stage8_c9_lm_head_bfp4_default_thresholds.json` (chip 1) both fail two hit windows: `L=1500 s=1472` top-1 0.9643 (27 of 28) and `L=16384 s=16352` 0.9688 (31 of 32). Every stage-7 variant has 1472 at 1.000; stage 7 failed only 16352 (or 2528 in round 1) and probed each. The 257-position windows also moved from 0.9961 (stage 7) to 0.9922 at every length. README sections 5 and 7 call these "one near-tie flip each, the stage-7 class" and "the stage-7 values: the decoder is unchanged"; the PCC minima are unchanged, the top-1 rows are not. The only stage-8 margin probes were run on the withdrawn c14, not on c9, and none targets 1472.
  Why this matters: The stage-review standard treats a prose dismissal of a visible anomaly as unresolved, and stage 9 declares prefix caching from this gate.
  Required next step: Run `logs/resume_margin_probe.py 1500:1472` through the default path on one free chip and record the HF margin at the flipped position, as stage 7 did for 16352; correct the README wording so "unchanged" applies to the PCC minima only and the top-1 rows are quoted against stage 7.

- P2: `doc/context_contract.json` claims its token budget equals the runtime's no-mesh formula; the runtime constant gives a different number, and the goal-named CSV is not in the commits
  Evidence: `datatype_sweep.max_tokens_all_users` is 114432 with the note "matching Qwen3Generator.max_tokens_all_users without a mesh". `Qwen3Generator.max_tokens_all_users(model=None, mesh_device=None)` uses the fixed `WEIGHTS_RESIDENT_BYTES = 14_928_363_520` (stage-6 BFP8 LM head): 112320 for every config. `logs/update_context_contract.py` computes resident bytes from the config instead. Separately, `sweep_results.csv` and every `logs/*.log` the README cites are excluded by the repo `.gitignore` (`*.csv`, `*.log`); stages 1 to 3 force-added CSV files under `doc/`.
  Why this matters: The contract note is false as written; the mesh path reads real free DRAM so serving is unaffected, but the no-mesh fallback under-reports by 2112 tokens and the contract says otherwise. The CSV is a goal deliverable that will not reach the pushed branch.
  Required next step: Host only. Either derive `WEIGHTS_RESIDENT_BYTES` from the loaded config or change the contract note; `git add -f` `sweep_results.csv` (and `profile/*.csv` or drop them from the index); logs stay uncommitted as in stage 7, so say so once in README section 10.

## Other Concerns

- Natural-text reference covers the prefill LM head only (`gen.prefill_logits`, LM head LoFi fp32 accumulation); the decode LM head kernel (LoFi, bf16 accumulation, DRAM-sharded) is covered on natural text only by the 100-token book32 TF (0.97) and the batch-32 case. State this in README section 1.
- The per-group `fidelity` field governs decode only; `OptimizedPolicy.prefill_fidelity` hard-codes BF16 HiFi4, BFP8 `prefill_bfp8_fidelity`, BFP4 LoFi. That is why `c10` reproduces `c5` to the last digit on natural text. Document in the config schema. Also a useful determinism control: c5 ran on chip 0, c10 on chip 2.
- `c3_kv_bfp8` passes the bar and the operator's two cache cases. It is rejected on batch-1 noise plus the stage-3 per-user PCC. Its capacity advantage (211456 tokens, 5 full-context users against 2) and the batch-32 decode step are not weighed or timed. Carry the batch-32 timing and the capacity trade to the vLLM stage.
- `bringup/references/qwen3_8b_book32.refpt` (1.5 MB) is untracked; HF greedy determinism across hosts is unverified.
- README section 4 says the 36-layer propagation check ran "through the default path (no environment variable)"; `logs/propagation_check.py` sets `QWEN3_PRECISION_CONFIG` to the selected file itself. Same file, different resolution; fix the sentence.
- `readiness_selected_default_chip2.json` entry `selected_prefill_resume_check` records `"top1": 1.0` because `write_sidecar.py` takes the last `top1=` line of any log, while the check failed on top-1 agreement.
- Wording: README 3.1's c8 sentence is garbled; `selected_precision_config.json` and `sweep_results.json` label the natural-text reference "teacher-forced against HF fp32 logits", it is a prefill-logits comparison.
- Shell helpers under `logs/` carry `#` comment lines; `tt/` and `tests/` changes carry none.
- The concurrent c8 runs are accuracy-only and marked `concurrent`; the work log must say the runs were review-triggered and dated.

## Hard-Check Gaps

- `top5_perf_pareto.png` draws no frontier (every point at top-5 1.000); mark the single frontier point or annotate the degenerate case.
- `natural_text_c1_all_bfp8_lofi_chip2.json`, `natural_text_c11_down_bfp4_inner_chip1.json` and `natural_text_c9_lm_head_bfp4_chip0.json` still carry the first-revision fields.
- `tt/precision.py` `check_propagation` does not compare `fp32_acc`, `packer_l1_acc`, `in0_block_w` or per-layer prefill fidelity.
- README section 8 and work log section 9 hold placeholders for this review.

## Anomaly Ledger

- Two green points (c8, c10) above the red selected point on the committed top-1 chart: more-work-needed (evidence exists in the worktree; not recorded, not committed).
- New resume-check failure at `L=1500 s=1472` (27 of 28) under c9; 257-position windows 0.9961 to 0.9922: more-work-needed (run the probe).
- c10 natural-text numbers bit-identical to c5: controlled; document (decode-only fidelity field).
- c13 TTFT 58.93 ms in round 2: controlled (concurrent row, ranks nothing).
- "Allocating device buffers is potentially unsafe due to the existence of an active trace" in `logs/selected_*.log`: controlled (stage-7 acknowledged allocations; tracked workload passed with program cache 666 flat).
- Two-layer propagation builds generate gibberish: controlled (expected for 2 of 36 layers; the 36-layer build generates " Paris. The capital of the United States").

## Scope Inspected

Goal/skill paths, artifact paths, code paths and commands as listed by the reviewer: the review prompt, the four skill files read from disk, every README/work-log/results/chart/config/sidecar/natural-text/propagation/batch-32/qualitative/profile artifact of the stage, the readiness resume JSON files (all 15), `doc/context_contract.json`, `doc/full_model/perf_batch1_128_128{.json,.stage8.json}`, the stage-7 README, `.gitignore`, `.pre-commit-config.yaml`; `tt/precision.py` in full and the `git show cd64cfe5aa` diffs; read-only git, shell and `python3 -I` commands and image reads of both charts. No device, server or implementation file was touched.

## Residual Risk

- The c9 natural-text margin is 14 positions above the bar (0.9094 against 0.90 on 1468 positions, binomial standard deviation about 11); c8 is 6 below. The benchmark stage's lm-eval subsets are the first sample large enough to confirm the ordering.
- The stage-7 baseline scores 0.919 on natural text against 0.968 for all-BFP8 (c1); the BFP4 up projection chosen in stage 3 costs 5 points on this reference. Not a stage-8 regression, but later stages inherit it.
- The decode LM-head kernel change (bf16 accumulation) has 100 tokens of natural-text decode evidence.
- The 2112-token gap between the contract and the no-mesh fallback is conservative; the mesh path is authoritative.
- Every `logs/*.log` cited by the README lives only in this worktree.

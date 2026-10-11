# Stage Review

Stage 03, optimized decoder (single-device decoder-layer optimization), re-review 6.
Previous reviews:
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review.md`,
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review_2.md`
through
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review_5.md`
(review 5: more-work-needed, three P2).
Target `Qwen/Qwen3.5-35B-A3B` revision `59d61f3ce65a6d9863b86d2e96597125219dc754`. One Blackhole
ASIC (application-specific integrated circuit), 1x1 mesh. Reviewer: fresh independent subagent,
2026 Oct 11. I ran no device, server or hardware job. I ran the CPU-only pytest of
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/test_prefill_plan.py`
and small artifact scripts.

Abbreviations: PCC (Pearson correlation coefficient), GDN (Gated DeltaNet, the linear-attention
layer), MoE (mixture of experts), KV (key/value cache), HF (Hugging Face reference), BF16/FP32
(16/32-bit float), BFP8/BFP4 (8/4-bit block floating point), L1 (per-core SRAM, static
random-access memory), DRAM (device main memory), CPU (host processor), ULP (unit in the last
place), us/ms (micro/milliseconds), KiB/MiB/GiB (2^10/2^20/2^30 bytes), SHA-256 (secure hash
algorithm, 256 bit).

Source identity recheck (reviewer): all 37 `tt/*.py` and `tests/*.py` entries of
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/final_evidence15/source_manifest_start.json`
match the live files by SHA-256. No live file is missing or extra. The start and end manifests
are equal. `git rev-parse HEAD` is `1d857aa3`. The SHA-256 of `git diff HEAD` outside the
autoport is the empty-input hash. `optimized_decoder.py` is `419ae4e6`, the same hash as in
`final_evidence15/summary.json` and in every device-pytest `result.json`. Against the
`final_evidence14` source snapshot, the only implementation change is the planner cost
(`_PLAN_COST` gets a fifth value and `_chunk_plan` uses it when `p == offset == 0`).
`run_layer.py`, `run_reuse.py`, `run_positions.py` and `test_optimized_decoder_device.py` are
unchanged.

Verdict: clean-pass

## Review-5 findings: status

| Review-5 finding | Status | Reviewer evidence |
|---|---|---|
| P2 short whole prompts slower after recalibration | Resolved | Whole-prompt step cost is now 1.05 ms (full) and 1.6 ms (linear). `test_short_plan_choice` pins full 2-3 to decode steps, full 4-5 and linear 3-5 to masked 32, linear 2 to decode steps, and the 4100 tails. Final chain: full 4 tokens 5623 to 4419 us, linear 3 tokens 6005 to 4952 us (`final_evidence14` versus `final_evidence15/prefill_{full,linear}_{3,4}/default.json`). Both gains exceed the single-sample noise (see Other Concerns). README table updated with the new values. |
| P2 `hybrid_routed_expert_moe` rejection on a configurable limit | Resolved | `review5/hybrid_worker_l1/hybrid_l3_{1024,4096}_wl1.json`: mesh opened with `worker_l1_size` 1444736. The default run had 1452928 B largest free plus 8192 B allocated per bank, so the default is 1461120 B and the reduction is 16 KiB as claimed. Router L1 tensors freed (`l1_allocated_after_free` 0). All four thresholds run. Median of 7, same captured real dispatch inputs: 1024 tokens 8142-8714 us versus 5905 us fused; 4096 tokens 9780-10867 us versus 8135 us fused. The op has no grid parameter (`hybrid_routed_expert_ffn.hpp`), so the threshold is its only tuning knob. Rejection is earned on measurement. |
| P2 full-layer pytest tokens 684 and 592 | Resolved | `review5/token684_tt/l3_1000_random.json` runs on source `419ae4e6` and reproduces 0.99306 at 684 and 0.99499 at 592. `review5/token684_ref/token684.json`: BF16 router logits for experts 70, 10 and 92 are all 1.04688 (8th to 10th place). `token684_variants.json`: forcing 70 or 92 into the top 8 gives TT versus reference 0.99751 and 0.99771; forcing 10 gives 0.99298. Token 592 has no tie, BF16 versus FP32 reference 0.999975; classified TT-side, same class as linear token 1720. Classification is earned (wording note below). |
| Other: raw headline values | Resolved | README 32491.0 us and 25904.8 us match `final_evidence15/perf_{linear,full}/default.json` (32490.986, 25904.768) to rounding. |

## Required Work

None.

## Other Concerns

- Stale planner text in the contract. `prefill_chunk_buckets.planner` in
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`
  says "1-2 token remainders use decode-path steps". The code (`rem <= 8` in `_chunk_plan`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py`
  line 282) and the README allow up to 8, and full 4100 uses 4 decode-path steps. The warm-up
  shape list is not affected (decode-path steps are the listed `[1, "exact"]` shape). Fix the
  sentence before the checkpoint commit, because Stage 6 and later read this contract.
- Token-684 label. The README and work log call it "reference tie noise". The FP32 reference
  also picks expert 10, by 2.2e-4 over expert 70 (`token684.json`: 1.0468 versus 1.04658). The
  gap is far below one BF16 ULP at 1.0 (0.0078), so in the BF16 router (TT and HF BF16) the
  three experts are an exact tie and TT follows a valid tie-break. A more precise label is
  "BF16 router-logit tie; FP32 favors expert 10 by 2.2e-4". The datatype sweep can test FP32
  router logits if it wants to remove such ties. This does not change the classification.
- Short-length timings are single samples (`run_layer.py` lines 206 to 217 time one warmed
  prefill). Same-plan, same-path rows vary between chains: full 2 tokens 3197, 3468, 3501 and
  4120 us (`final_evidence11` to `15`, the planner diff does not touch this plan), linear masked
  32/4 4889 versus 5263 us, linear masked 32/5 5038 to 5890 us. Within `final_evidence15`, linear
  masked 32/3, /4, /5 are 4952, 5263, 5890 us. That spread has the same size as the cross-chain
  noise, so the README claim "short whole prompts rise with length" is not shown above noise
  (unverified). Full 3 tokens (3 decode steps, 4451 us) versus masked 32/4 (4419 us) is a tie
  within noise. Later stages that tune short-prompt behaviour should use a median of several
  warmed repeats.
- Carried from earlier reviews, unchanged and accepted by disclosure: GDN prefill head split
  (about 6.5 ms per 4096 tokens); 16 to 23 us slower trace replay on physical device 0; no
  chained two-layer unaligned prefill check; unresolved root cause of prefill repeat drift.
  My scan of 183 `prefill_repeat_difference` records in `final_evidence15` finds four
  non-identical runs (`control_grid_linear_2048`, `prefill_full_2176_s128`,
  `prefill_linear_1100`, `prefill_linear_2048`), max abs 0.0029, repeat PCC at least
  0.99999994, as the README states.
- Memory for later stages: both routed-expert representations stay resident (852.7 MiB
  prefill dispatch plus 560.0 MiB decode per layer). About 60 GiB for 40 layers against 31.8 GiB
  per chip. The p150 (one chip) profile is the largest open risk for Stages 4 and later. The
  stage states this in the README and the contract.

## Hard-Check Gaps

- No full-attention unaligned profile. Chain log
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/final_evidence15_chain.log`
  shows the four non-zero exits (`prof_unaligned_3_{4095,1000}` rc=1 and the `report_*`
  steps rc=2) with "Source location hashes are colliding" at lines 2511 and 2787. This matches
  the recorded blocker.
- `prefill_repeat_identical` is recorded, not asserted (accepted by disclosure).
- `tests/test_prefill_plan.py` checks plan invariants and nine pinned short or tail plans, not
  cost or output order.
- Short-length prefill timings are single samples (see Other Concerns).
- No peak-DRAM measurement for the 262143-token prefill.

## Anomaly Ledger

- Observed anomaly: Short whole prompts were slower than one token longer (review 5).
  Evidence: `final_evidence14` versus `final_evidence15/prefill_{full,linear}_{2,3,4,5}/default.json`.
  Affected path: `_chunk_plan` decode-step option for whole short prompts.
  Control or comparison: masked 32 plans; unchanged-plan rows across chains.
  Likely subsystem: planner step cost.
  Investigation performed: separate whole-prompt step cost; pinned plan test; reruns.
  Resolution: fixed.

- Observed anomaly: Same-plan short-prompt timing spread up to 0.94 ms (linear masked 32/3 to /5; full 2 tokens 4120 us versus 3197 to 3501 us in earlier chains).
  Evidence: `final_evidence1{1,2,3,4,5}/prefill_{full,linear}_{2,4,5}/default.json`.
  Affected path: timing harness, single warmed sample.
  Control or comparison: full 2 tokens has the same plan and unchanged code path in all chains.
  Likely subsystem: host and dispatch timing noise at about 4 to 6 ms.
  Investigation performed: reviewer cross-chain comparison and source diff.
  Resolution: controlled (noise level; no default-path decision depends on a margin below it).

- Observed anomaly: `hybrid_routed_expert_moe` program exceeded the default kernel config buffer.
  Evidence: `review4/hybrid/`, `review5/hybrid_worker_l1/`.
  Affected path: prefill routed experts (candidate only).
  Control or comparison: `moe_fused_swiglu` and `unified_routed_expert_moe` in the same run.
  Likely subsystem: device `worker_l1_size` configuration.
  Investigation performed: rerun with 16 KiB smaller worker L1; four thresholds; two lengths.
  Resolution: controlled (runs; slower by 1.6 to 2.8 ms; rejected on measurement).

- Observed anomaly: Full-layer pytest token 684 at 0.99306.
  Evidence: `review5/token684_ref/`, `review5/token684_tt/`.
  Affected path: router top-k at an exact BF16 tie.
  Control or comparison: forced tie-break variants 0.99751 and 0.99771.
  Likely subsystem: BF16 router-logit tie.
  Investigation performed: CPU reference in BF16 and FP32, forced variants, TT tensor on final source.
  Resolution: controlled.

- Observed anomaly: Full-layer pytest token 592 at 0.99499.
  Evidence: `review5/token684_ref/token684.json`, `token684_variants.json`.
  Affected path: full-attention prefill, 1000 tokens.
  Control or comparison: BF16 versus FP32 reference 0.999975, no top-10 change.
  Likely subsystem: TT precision.
  Investigation performed: reference-precision control.
  Resolution: controlled (TT-side, same rule as linear token 1720, recorded for the datatype sweep).

- Observed anomaly: Linear token 950, linear token 1720, full token 911, linear B32 row 17 token 4, prefill repeat drift, linear profile replay 100.
  Evidence: as in review 5; final-chain values unchanged in `final_evidence15`.
  Affected path: as in review 5.
  Control or comparison: as in review 5.
  Likely subsystem: as in review 5.
  Investigation performed: as in review 5; reviewer drift scan of `final_evidence15`.
  Resolution: controlled (accepted in review 5).

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/03-03-optimized-decoder.prompt.txt`
  (contract as cited in review 5), `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`.
- Artifact paths (under `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/`):
  `optimized_decoder/stage_review_5.md`, `optimized_decoder/README.md`,
  `optimized_decoder/work_log.md` ("Stage review 5 response"), `optimized_decoder/checklist.md`,
  `context_contract.json` (`optimized_decoder_validation`, `serving_contract`),
  `optimized_decoder/final_evidence15/` (`summary.json`, `source_manifest_{start,end}.json`,
  `perf_*`, `prefill_*`, `pytest/` (all `result.json`, `commands.json`, `device.log`),
  `plan_pytest.xml`, all 11 `watcher.log` files, `lane*.log`),
  `optimized_decoder/final_evidence15_chain.log`, `optimized_decoder/final_evidence14/source_snapshot.tar.gz`,
  `optimized_decoder/review5/` (all files), earlier `final_evidence11` to `14` short-prompt rows,
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/evidence_chain15.sh`,
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/telemetry/packets/9ba51029-cb68-4989-ae12-8f9ed3691bdd.json`.
- Code paths: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py`
  (`_chunk_plan`, `_PLAN_COST`),
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/test_prefill_plan.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_layer.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/review5/hybrid_worker_l1/hybrid_isolated.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/review5/token684_ref/token684.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/review5/token684_ref/token684_variants.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/hybrid_routed_expert_ffn/hybrid_routed_expert_ffn.hpp`.
- Commands run: SHA-256 manifest comparison; `git rev-parse`, `git status`, `git diff HEAD`
  hash outside the autoport; chain-log exit-code scan; JSON scans of short-prompt rows across
  `final_evidence11` to `15`, repeat drift, pytest results and watcher logs; diff of the
  `final_evidence14` source snapshot against the live source;
  `python -m pytest models/autoports/qwen_qwen3_5_35b_a3b/tests/test_prefill_plan.py`
  (1125 passed in 155 s, my rerun).

## Residual Risk

- The planner cost model is a few constants calibrated on single-sample timings. Unmeasured
  lengths can pick a plan that is up to about 0.5 ms slower than the best (unverified).
- Prefill repeat drift has no root cause. It stays at rounding level in every measured run.
- A serving page table that breaks the `serving_contract` ownership rule gets padded K/V
  writes into another request's pages. Stage 6 and later must enforce the rule.
- The selected path keeps two routed-expert representations. The one-chip (p150) profile cannot
  hold 40 layers without a later change.

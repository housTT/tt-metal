# Stage Review (review 2, verbatim from the fresh read-only subagent, 13:05 to 13:22 ET)

Verdict: clean-pass

Review 2 of stage 8 (datatype sweep) for `Qwen/Qwen3-8B`, at HEAD `867deef747` on `hous/qwen3-8b-autoport` in the live worktree `/home/hous/dev/tt-metal-wt-qwen3`. The worktree is clean except the known untracked `bringup/references/qwen3_8b_book32.refpt`. The three review-1 Required Work items are closed by committed evidence, not by prose: the c8 and c10 robustness rows exist as sidecars and natural-text JSON files and flip both candidates to `fail` in `sweep_results.{json,csv}` and on both charts; the two c9 hit-window flips have default-path margin probes on chip 0; and `Qwen3Generator.max_tokens_all_users(model)` now computes the budget the contract states (114432, measured in `token_budget_check.json`). The selected policy `c9_lm_head_bfp4` is the fastest candidate that passes every reference it ran on, every faster candidate has a recorded failing reference, the policy is consumed by the default construction path (36-layer readback with `problems=[]`, `tt-perf-report` rows `LoFi BF16 x BFP4` on all ten LM-head splits), and the post-selection token-out benchmark ran through that path (48.13 t/s/u, TTFT 41.71 ms, `sampler_traced` true). Items below are notes and gaps, none of which contradicts an artifact or hides a correctness risk.

## Required Work

None.

## Other Concerns

- Acceptance-rule wording overstates coverage. `sweep_results.json` and `selected_precision_config.json` `acceptance.rule` say "every candidate faster than the selected one was run on every robustness reference". c12, c13, c16 and c17 were not (each had already failed the main or book32 reference). README section 7 names c5, c6, c7, c8, c10, c11, c14, c15 as "every candidate faster than the selected one that passed its main reference", but c13, c16 and c17 also passed the main reference (0.90, 0.91, 0.91) and are faster; they are covered only by the preceding sentence. The evidence resolves every faster candidate; the sentences do not match it. Wording in documents changed for review 1: note only.
- README sections 3.1 and 3.3 infer from the c10 = c5 natural-text identity that "the BFP4 weight quantization, not the fidelity, sets the loss". On the natural-text reference the fidelity was not varied: it runs the prefill path, where `OptimizedPolicy.prefill_fidelity` hard-codes BFP4 to LoFi, so the two runs are the same computation (bit-identical across chips 0 and 2). The HiFi2-versus-LoFi comparisons that did change the measured kernel are the decode teacher-forcing rows (c10 0.97 against c5 0.97; c4 0.98 against c0 0.98). Section 1 states the mechanism correctly. The inference is still supported by c1 against c0 (5 top-1 points from BFP4 on one projection at equal LoFi fidelity), so no run is required.
- The top-1 Pareto frontier passes through `c13` at its `concurrent` number (52.66 t/s/u). README section 1 and work log 4.4 say concurrent rows "rank nothing"; `logs/collect_results.py` does not mark them on the chart. c13 fails book32, so the selection is unaffected.
- `Qwen3Generator.max_tokens_all_users` is a classmethod whose `kv_cache_dtype` defaults to BF16 whatever cache the generator was built with. `logs/token_budget_check.py` passes `gen.kv_cache_dtype`; the stage-9 plugin hook must do the same, or a BFP8-cache config would get a conservative (too small) budget.
- The two natural-text JSON files written by the fixed script (c8, c10) carry `hf_top1_equals_next_book_token` and `tt_top1_equals_next_book_token` computed with the corrected index; the nine first-revision files had the fields removed and carry a `note` instead. The values are correct; the eleven files do not share one schema, and work log 7.1 reads as if the fields no longer exist.
- `sweep_results.json` stores the chain-1 row under the key `round2_concurrent_row` for c5, c6 and c9, whose chain-1 rows were serialized. Cosmetic.

## Hard-Check Gaps

- No `TT_METAL_TRACE_ALLOC_TRACKING=1` run on the selected policy in stage 8: `doc/full_model/tracked_replay_workload.stage8.json` records `tracking_env` all `null`, as does the stage-7 untracked copy; the tracked evidence is stage 7's `tracked_replay_workload.tracked.json`. Stage 8 changes no capture or replay code and only swaps the dtype of weight tensors allocated before capture; `program_cache_entries_after_capture` stays 666 in every stage-8 run. Not required work under the stage-review standard; stage 9 should include the selected policy in its tracked run.
- `sweep_results.csv` has no `command`, `hardware`, `mesh` or `timing_regime` column, so the goal's "measurement regime, command, hardware, mesh" fields exist only in `sweep_results.json`.
- `check_propagation` does not assert `embedding_dtype`, final-norm dtype and fidelity, activation dtype or the decode-logits/sampling dtype; `describe_runtime` reports them and the 36-layer readback matches the config by inspection.
- README section 8 and work log section 9 hold the placeholder for this review (expected).

## Anomaly Ledger

- Two candidates faster than the selection (c8, c10) recorded as `pass` at `ee31ae063d` (review-1 P1): fixed (robustness rows committed in `26856835e8`; every faster candidate fails at least one reference; c9 is the fastest `pass`).
- Resumed-prefill check under c9 fails top-1 on two hit windows at default thresholds: controlled (probes show near ties: fresh-path margin 0.078 / continuation 0.031 / HF 0.641 at 1487; 0.125 / 0.016 / 0.393 at 16369; both TT paths agree with HF equally; lowered-threshold run passes; PCC minima equal stage 7); stage 9 reruns the gate on this policy.
- c10 natural-text numbers bit-identical to c5 on different chips: controlled and documented (per-group fidelity governs decode only); also a cross-chip determinism control.
- "Allocating device buffers is potentially unsafe due to the existence of an active trace" in the selected-config logs: controlled (stage-7 tracked run on unchanged capture code; program cache 666 flat).
- c13 TTFT 58.93 ms: controlled (concurrent row; fails book32).

## Scope Inspected

Goal/skill paths (the review prompt, the four skill files read from disk, the review-1 record); every stage-8 artifact under `doc/datatype_sweep/` (README, work log, results JSON and CSV, selected config and candidate configs, both charts viewed, all eleven natural-text files, the c9 and c8 sidecars, the default-path sidecar, the 36-layer propagation check, `token_budget_check.json`, both c9 probes, the batch-32 case, the qualitative suite with TT and HF texts read, the profile tables); `doc/context_contract.json`, `doc/full_model/*.stage8*.json` and the stage-7 tracked twin, the five resume JSON files, the stage-7 README; code: `tt/precision.py` in full and the diffs `2e52cebf61..HEAD` and `ee31ae063d..26856835e8`; read-only git and shell commands, `python3 -I` scripts re-deriving pass/fail for all 20 candidates, `py_compile` on the six committed Python files. No device, server or implementation file was touched. Push state: five local commits, none pushed by the stage agent; the supervisor pushes.

## Residual Risk

- Natural-text margin of the selection is 13 positions above the bar (1335 of 1468 against 1322 needed; binomial standard deviation about 11); c8 is 6 below. The benchmark stage's lm-eval subsets are the first larger sample.
- Fresh-versus-continuation top-1 agreement moved from 1 to 2 or 3 disagreements per 257-position window and from one to two flipped hit windows under the BFP4 LM head; each is a probed near-tie, and stage 9 declares prefix caching from this gate.
- `OptimizedPolicy.prefill_kernel` maps a BF16 gate or up projection to the BFP8-probed table row; BF16 tiles are twice BFP8 tiles, so a BF16 MLP config could hit the L1 overflow c1 hit at 512 rows. No candidate uses BF16 projections.
- The committed `tt/precision.py` at `26856835e8` is the black-reformatted file; the 36-layer propagation check ran on the pre-format tree. Parse check passes; formatting only.
- `bringup/references/qwen3_8b_book32.refpt` (1.5 MB) is uncommitted by design; HF greedy determinism across hosts is unverified.
- Every `logs/*.log` (148 files) lives only in this worktree, as in stage 7.
- The decode LM-head kernel (bf16 accumulation) has 100 tokens of natural-text decode evidence plus the batch-32 case; the 1468-position reference exercises the prefill LM head.

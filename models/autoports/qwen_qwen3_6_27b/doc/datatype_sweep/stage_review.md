# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Complete the required same-dtype BFP4 fidelity comparison before selecting `full_down_bfp4_lofi`.
  Evidence: The selected policy uses BFP4+LoFi for linear- and full-attention MLP gate, up, and down groups. The completed matrix contains `full_down_bfp4_lofi`, but no BFP4+HiFi2 candidate for the newly selected full-attention down group and no clean BFP4+LoFi versus BFP4+HiFi2 comparison for the selected gate/up groups. `full_down_bfp8_hifi2` changes full-attention down fidelity while retaining BFP8 weights, and `canonical_runnable_bfp8_hifi2_kv_bf16` changes multiple weight, CCL, KV, and fidelity fields; neither is a same-dtype comparison. The AutoFix evidence establishes legality rather than a blocker: its repaired traced run executes linear-down BFP4+HiFi2, while full-down BFP8+HiFi2 and gate/up BFP8+HiFi2 fit and run, so smaller BFP4 weights have no recorded exact op-contract blocker.
  Why this matters: `$datatype-sweep` explicitly requires BFP4+LoFi versus BFP4+HiFi2 evidence when both are legal and the group is a meaningful decode cost. Without it, the candidate matrix is incomplete and the claim that the fastest required precision/fidelity policy was selected is unsupported.
  Required next step: Run traced full-model AIME24 candidates that provide an interpretable BFP4+LoFi versus BFP4+HiFi2 comparison for every material selected BFP4 group (at minimum the newly changed full-attention down group, plus coherent evidence covering gate/up and linear down), record top-1/top-5/top-100 and traced decode performance in both ledgers and plots, and reselect from the completed passing matrix. If any comparison cannot run, record the exact blocker and complete the required AutoFix loop.

- P2: Add branch/commit and dirty-runtime provenance to every sweep row.
  Evidence: All eight `sweep_results.json` rows and the corresponding CSV rows omit branch, commit, and environment/dirty-state provenance. The work log names the runtime checkout but does not identify the measured source revision; the live review started at `b7b52f83305e1e7c350bde15c7b648d43652e4e2` on `agentic-research/hous/qwen3.8-27b` with stage-owned dirty changes.
  Why this matters: `$datatype-sweep` requires exact branch/commit and environment notes for each evaluated configuration. Commands and timestamps alone do not make results reproducible when measurements were made from a dirty worktree.
  Required next step: Add explicit branch, starting commit, and a concise stage-owned dirty-source identifier or equivalent reproducible diff provenance to the JSON/CSV rows and document it in the work log. Regenerate the ledgers and verify the recorded provenance matches the measured implementation state.

## Other Concerns

- The selected advantage is only 0.174% over the baseline. Three selected-path observations are consistent (22.614, 22.608, and the final default-path 22.607 t/s/u), but the baseline has only one observation at 22.572 t/s/u. This is not independently blocking given the tight selected repeat spread, but the final decision should reassess the margin after adding the missing fidelity candidates and prefer the safer policy if results are within cross-policy measurement noise.
- Every 64-token qualitative output ends mid-`<think>` reasoning before producing the requested answer. The TT behavior closely matches the same prompt-correct HF control and shows no TT-specific degeneration, so this is controlled rather than stage-blocking; however, these artifacts establish prefix coherence, not completed-task quality.

## Hard-Check Gaps

- The baseline prefill and teacher-forcing metric JSON files predate live precision-summary capture and therefore do not themselves record the resolved baseline policy. The explicit candidate command/config and deterministic loader code tie the policy to the run sufficiently for this review, but future baseline refreshes should retain the live `precision_summary` like the completed candidate and final metrics do.
- `non_aligned.junit.xml` proves the named four-layer selected-default test passed but does not retain its console details. Inspection of `test_reduced_full_model_prefill_decode_and_split_trace` confirms that the named test exercises logical lengths 65/67, split trace replay, and mixed slots under the default policy.
- The live strict-validation/KV propagation fix was made after the hardware artifacts. It does not alter selected runtime semantics because the measured and selected page block size is still 64; the work log records eight passing host checks after the fix. No new hardware result is required solely for that equivalence.

## Anomaly Ledger

- Observed anomaly: The original canonical BFP8+HiFi2/BF16-cache candidate failed traced warmup with an L1/static-CB collision.
  Evidence: `evidence/candidates/canonical_bfp8_hifi2_kv_bf16/failure.json` records L1 floor 928000 and static-CB end 1333760; `autofix/canonical_l1/AUTOFIX.md` derives the collision and the BFP4-linear-down repair.
  Affected path: Linear-attention MLP down during traced decode warmup.
  Control or comparison: `canonical_runnable_bfp8_hifi2_kv_bf16` reduces the affected static-CB end to 811520 and completes traced full-model teacher forcing at 0.98/1.00/1.00 and 18.257 t/s/u.
  Likely subsystem: DRAM-sharded MLP weight/static circular-buffer L1 allocation.
  Investigation performed: Source arithmetic, recursive-policy resolution, AutoFix report, and exact repaired full-model traced rerun.
  Resolution: fixed.

- Observed anomaly: Initial review found selected KV layout/block-size and logits/sampling metadata accepted without complete runtime consumption or strict rejection.
  Evidence: The live worktree was updated during review. `tt/precision.py` now exact-validates section keys, fixed KV layout/block size, logits assumptions, and layer overrides; `tt/model.py` now uses policy `page_block_size` for decoder construction, state allocation, and mixed-prompt page arithmetic.
  Affected path: Selected precision-policy loading and full-model paged-cache geometry.
  Control or comparison: Selected and measured values remain `tile_dram_paged` with block size 64 and the existing BF16 logits/sampling contract, so the updated path is behavior-equivalent for existing hardware evidence.
  Likely subsystem: Precision-policy schema and propagation.
  Investigation performed: Before/after source inspection, selected artifact comparison, and review of added host-policy assertions.
  Resolution: fixed.

- Observed anomaly: The winning traced teacher-forcing edge over baseline is small.
  Evidence: Selected average 22.611371 t/s/u versus baseline 22.572120 t/s/u; selected repeats differ by 0.026%, and final default construction gives 22.606825 t/s/u.
  Affected path: Precision selection ranking.
  Control or comparison: Two candidate runs plus the separate final default run consistently exceed the single baseline observation.
  Likely subsystem: Normal device timing variability versus the full-attention down BFP4 change.
  Investigation performed: Re-derived ranking and repeat arithmetic from raw metric JSON, not README values.
  Resolution: controlled, with residual cross-policy timing risk.

- Observed anomaly: All qualitative completions expose reasoning text and truncate before a final response.
  Evidence: Direct inspection of all six HF and six TT outputs; all are exactly 64 completion tokens. Matching prefixes range from 11 to 63 tokens, and the degeneracy report passes all prompts.
  Affected path: Selected-config qualitative generation.
  Control or comparison: The exact-checkpoint HF chat-template controls show the same behavior on every prompt.
  Likely subsystem: Checkpoint/chat-template reasoning style plus the 64-token generation cap, not TT token feedback or precision.
  Investigation performed: Compared rendered prompts, token IDs, decoded outputs, automated degeneration results, and manual verdict.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: original datatype-sweep contract; `.agents/skills/datatype-sweep/SKILL.md`, `stage-review/SKILL.md`, `tt-device-usage/SKILL.md`, `qualitative-check/SKILL.md`, and `autofix/SKILL.md`.
- Artifact paths: datatype-sweep README/work log; selected and all candidate configs; JSON/CSV ledgers; both Pareto PNGs; baseline, all candidate, final teacher-forcing/token-out/non-aligned/qualitative evidence; canonical AutoFix reports; `doc/context_contract.json`; and the 100-token AIME24 reference metadata/content.
- Code paths: `tt/precision.py`, `tt/model.py`, `tt/generator.py`, `tt/multichip_decoder.py`, `tt/optimized_decoder.py`, `tests/test_precision_policy.py`, `tests/test_full_model.py`, `tests/run_qualitative_suite.py`, readiness-runner diffs, and `build_artifacts.py`.
- Commands run: read-only `git status/diff/log`, `find`, `rg`, `sed`, `jq`, CSV/JSON consistency and ranking scripts, reference inspection, file existence/timestamp checks, `git diff --check`, and visual inspection of both PNGs. No TT device, server, vLLM, reset, reservation, or hardware experiment was run.
- Scope isolation: No vLLM implementation was started. Stage-owned changes are confined to the disclosed autoport/readiness-runner paths; unrelated dirty paths remain `tt_metal/third_party/tracy`, `tt_metal/third_party/umd`, and untracked `tt_metal/third_party/tt-cluster-descriptors/`.

## Residual Risk

- The missing same-dtype fidelity rows may not beat LoFi, but that conclusion currently rests on confounded candidates rather than the required full-model comparison.
- Runtime policy summaries prove resolved configuration and source paths, while actual dtype/fidelity application is established primarily by construction-code inspection rather than profiler rows. This is acceptable under the stage contract because the propagation evidence is direct and non-contradictory, but later changes to TTNN lowering could still warrant profiler confirmation.
- Capacity arithmetic for BFP8/BF16 KV caches and selected BFP4 weight savings is internally consistent and preserves the 262,144-token shared pool, but no new maximum-context hardware run was performed specifically for the behavior-equivalent strict-validation patch.

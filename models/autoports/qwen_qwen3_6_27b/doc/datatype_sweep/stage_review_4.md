# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Make dtype validation and runtime consumption agree for accepted case variants.
  Evidence: `PrecisionPolicy._dtype()` lowercases values only for validation and returns the normalized string, but `_validate()` discards that return value for `ccl.*` and `kv_cache.dtype`. A host-only probe against the selected config showed both `kv_cache.dtype="BFP8"` and `ccl.linear_attention_mlp="BFP8"` are accepted while the stored runtime values remain uppercase. `OptimizedDecoder.allocate_paged_kv_cache()` then indexes `{"bfp8": ..., "bf16": ...}` with the raw cache value, causing a `KeyError`; the CCL path compares the raw value to lowercase `"bfp8"`, so its pre/post typecasts are skipped even though buffer/all-reduce dtype selection takes the non-BF16 branch. The retained `autofix/canonical_l1/AUTODEBUG.md` already identified this validation/consumption mismatch, but the final schema tests cover unknown keys and unsupported lowercase values only.
  Why this matters: The stage contract requires every accepted precision-policy field to be consumed as recorded. A config that passes validation but crashes or silently executes a different CCL conversion policy violates that contract and makes future candidate overrides unsafe.
  Required next step: Either exact-reject noncanonical case or normalize and store canonical dtype values before runtime consumption. Add host tests for uppercase/mixed-case KV and CCL values that prove the chosen contract, regenerate the retained JUnit, update the final runtime source hash/provenance artifacts, and rerun a fresh stage review. Because all measured configs are already canonical lowercase, this is a host/schema repair and does not by itself require a hardware rerun.

## Other Concerns

- `work_log.md` ends by saying the second fresh rereview is pending and does not record `stage_review_3.md` or its baseline remediation. The baseline sections themselves are current, but the review/commit chronology should be updated before checkpoint handoff.
- The selected mean remains only 0.194% above the refreshed baseline. Two selected candidate runs and the separate default-path result are tightly consistent, so the ordering is supported, but cross-policy timing noise remains less characterized than within-selected repeatability.
- All six qualitative runs stop inside visible reasoning at the 64-token cap. Exact-checkpoint HF controls show the same prompt-correct behavior, so this is controlled prefix-quality evidence rather than a TT regression.

## Hard-Check Gaps

- The raw baseline metrics embed the resolved policy and absolute config path but not the shell command. The exact explicit-override command is retained in the generated JSON/CSV ledger and work log, and there is no remaining contradictory provenance claim.
- `evidence/final/token_out_metrics.json` does not embed a precision summary. The benchmark test constructs the default `Generator`, and the separate final teacher-forcing artifact proves that default resolves to the selected config.
- `non_aligned.junit.xml` retains the passing test identity but not console detail. Source inspection confirms that the named test exercises mixed logical lengths 65/67, split trace capture/replay, changed-only page tables, and sampling modes.
- Runtime dtype/fidelity application is proved by raw resolved summaries plus direct construction/op code rather than profiler rows. No retained artifact contradicts the lowercase policies actually measured.

## Anomaly Ledger

- Observed anomaly: The prior review's baseline command and policy provenance were contradictory.
  Evidence: Fresh raw teacher-forcing and prefill artifacts were generated at `2026-08-21T00:40:05Z` and `2026-08-21T00:48:29Z`; both embed `config_id=baseline_optimized_mixed` and the exact absolute baseline config path. Teacher forcing records 0.97/1.00/1.00, 975.833597 ms TTFT, `decode_trace_enabled=true`, and 22.567674884 t/s/u. Prefill records 0.97/1.00/1.00. The JSON/CSV baseline command explicitly sets the same override, and the README/work log no longer claim it was omitted.
  Affected path: Baseline accuracy/performance control and Pareto ranking.
  Control or comparison: The baseline resolved config equals its raw precision summary; all referenced paths exist; JSON/CSV semantic parity is exact.
  Likely subsystem: Readiness metric provenance and ledger generation.
  Investigation performed: Compared both raw baseline artifacts, recursive config resolution, ledgers, artifact builder, documentation, timestamps, and source cohort.
  Resolution: fixed.

- Observed anomaly: The final baseline cohort claims exact final-runtime source identity.
  Evidence: `source_provenance.json` maps only `baseline_optimized_mixed` to `final_runtime_v3`, marks exact source preserved, and records `sha256:6ab2b2093de4652e7e358cc2afda422f327b4cccf8a0778e85d52671bd46f2a7`. Recomputing the builder's hash over all seven listed runtime files yields the identical digest; every source-file mtime precedes both refreshed baseline runs.
  Affected path: Baseline measured-source reproducibility.
  Control or comparison: Older candidate cohorts remain explicitly marked as unpreserved dirty source with row-specific behavior-equivalence statements.
  Likely subsystem: Source provenance manifest.
  Investigation performed: Recomputed the hash, checked file lists/timestamps, cohort coverage, row identifiers, and all referenced paths.
  Resolution: fixed.

- Observed anomaly: The first review found no clean same-dtype BFP4 LoFi/HiFi2 comparison.
  Evidence: `all_mlp_bfp4_hifi2` matches selected BFP4 gate/up/down weights and all non-MLP fields, changing only the six material MLP fidelities to HiFi2. Raw traced results are 0.95/1.00/1.00 and 19.670285 t/s/u versus the selected LoFi mean 0.95/1.00/1.00 and 22.611371 t/s/u.
  Affected path: Linear- and full-attention MLP fidelity selection.
  Control or comparison: Recursive config resolution and raw precision summaries match; both ledgers and both Pareto plots include the completed point.
  Likely subsystem: MLP compute fidelity.
  Investigation performed: Compared configs, raw summaries, accuracy/performance, ranking, and plots.
  Resolution: fixed.

- Observed anomaly: Unknown root and nested precision keys were previously accepted.
  Evidence: Current exact-key checks reject extra root, weight-group, fidelity, activation, CCL, KV, logits, and runtime-loading keys; retained JUnit reports 10/10 precision-policy cases and 7/7 readiness-runner cases, while the static full-model JUnit passes 1/1. However, accepted case variants remain raw and inconsistent at runtime as described in Required Work.
  Affected path: Precision-policy schema and runtime overrides.
  Control or comparison: Canonical lowercase selected/candidate configs resolve correctly and match every raw metric summary.
  Likely subsystem: `tt/precision.py` normalization and raw CCL/KV consumers.
  Investigation performed: Inspected validator/tests/JUnits and ran host-only uppercase KV/CCL construction probes with bytecode writes disabled.
  Resolution: more-work-needed.

- Observed anomaly: The canonical all-BFP8/HiFi2/BF16-cache policy failed traced warmup at a static-CB/L1 collision.
  Evidence: The failure artifact records L1 floor 928000 and CB end 1333760. AutoDebug reproduces the endpoint exactly; the one-field BFP4 linear-down repair computes 811520 and passes full traced teacher forcing at 0.98/1.00/1.00 and 18.257202 t/s/u.
  Affected path: Linear-attention MLP down during traced decode warmup.
  Control or comparison: `canonical_runnable_bfp8_hifi2_kv_bf16` preserves the other canonical fields and completes the original workload.
  Likely subsystem: DRAM-sharded BFP8 weight/static-CB L1 footprint.
  Investigation performed: Reviewed failure JSON, AutoDebug/AutoFix reports, repaired config, resolved summary, and raw metrics.
  Resolution: fixed.

- Observed anomaly: The selected performance advantage is small.
  Evidence: Selected repeats are 22.614326 and 22.608416 t/s/u (0.026% spread), with a separate default-path result of 22.606825, versus the refreshed baseline 22.567675.
  Affected path: Candidate ranking.
  Control or comparison: Three selected observations are mutually consistent; all completed points pass the accuracy gate; the selected row remains the fastest.
  Likely subsystem: Device timing variability versus the full-attention-down BFP4 change.
  Investigation performed: Re-derived repeat means, thresholds, ranking, and Pareto dominance from raw artifacts.
  Resolution: controlled.

- Observed anomaly: Qualitative outputs end during visible reasoning.
  Evidence: All HF and TT completions are capped at 64 tokens. Prompt metadata records the exact checkpoint/tokenizer/chat template, rendered prompts, token ids, and greedy settings; HF/TT matching prefixes range from 11 to 63 tokens and all degeneracy checks pass.
  Affected path: Selected-config qualitative evidence.
  Control or comparison: Exact-checkpoint HF outputs show the same reasoning style and truncation on every prompt.
  Likely subsystem: Checkpoint behavior plus the short generation cap, not TT precision or token feedback.
  Investigation performed: Read all prompt metadata, raw HF/TT outputs, automated checks, and manual verdict.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: supplied datatype-sweep contract; `.agents/skills/datatype-sweep/SKILL.md`, `.agents/skills/stage-review/SKILL.md`, `.agents/skills/tt-device-usage/SKILL.md`, `.agents/skills/qualitative-check/SKILL.md`, and `.agents/skills/autofix/SKILL.md`.
- Artifact paths: datatype-sweep README/work log and all three prior reviews; all candidate/selected configs; source provenance; JSON/CSV ledgers; every baseline/candidate/final raw metric or failure artifact; all four retained JUnits; qualitative metadata/HF/TT outputs/degeneracy/verdict; both Pareto PNGs; canonical AutoDebug/AutoFix reports; `doc/context_contract.json`; and the 100-token AIME24 reference.
- Code paths: `tt/precision.py`, `tt/model.py`, `tt/generator.py`, `tt/multichip_decoder.py`, `tt/optimized_decoder.py`, `tests/test_precision_policy.py`, relevant full-model/non-aligned/token-out/qualitative code, readiness-runner diffs, and `build_artifacts.py`.
- Commands run: read-only `git status/branch/rev-parse/diff/diff --check`, `find`, `rg`, `sed`, `jq`, `stat`, `sha256sum`, JSON/CSV/path/config/cohort/hash/ranking analysis scripts, host-only in-memory policy probes, and visual inspection of both PNGs. No TT device, server, vLLM, reset, reservation, or hardware work was run.
- Scope isolation: No vLLM implementation was added. Stage-owned changes are confined to the disclosed autoport/readiness paths; unrelated Tracy/UMD/cluster-descriptor dirtiness was excluded.

## Residual Risk

- The selected/default policy and every measured candidate use canonical lowercase values, so the normalization defect does not invalidate the recorded accuracy or ranking. It does prevent the schema/runtime contract from being called complete until accepted values are either canonicalized or rejected consistently.
- After that host-only repair, artifact/hash/JUnit regeneration, and review-chronology update, the retained evidence supports the selected policy without another TT run unless the fix changes canonical lowercase behavior.
- The local checkpoint and no-push handoff remain post-clean-review actions; they cannot occur before this reviewer returns a clean pass.

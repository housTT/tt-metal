# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Finish per-row dirty-source provenance; the added fields do not identify the measured source and contain a false forward reference.
  Evidence: All nine JSON rows and all nine CSV rows now contain `runtime_git_branch`, `runtime_git_base_commit`, `runtime_source_state`, and `runtime_environment_notes`, so the field-presence part of the first review is closed. However, every row has the same prose-only source state: the worktree was dirty and stage-owned changes were uncommitted, followed by “the post-review local checkpoint SHA is recorded in work_log.md.” `work_log.md` instead says the checkpoint SHAs are pending, `git rev-parse HEAD` is still the base `b7b52f83305e1e7c350bde15c7b648d43652e4e2`, and `git status` shows the stage implementation/artifacts remain dirty or untracked. The raw metrics do not carry a patch hash, tree hash, checkpoint, or other source-state identifier. Their timestamps also span source edits: the earlier candidate/final measurements precede the 20:01–20:03 strict-policy edits, while `all_mlp_bfp4_hifi2` was generated at 20:14. The same generic source-state string therefore cannot identify each measured dirty revision.
  Why this matters: The first review required a concise stage-owned dirty-source identifier or equivalent reproducible diff provenance, not only four populated columns. A future checkpoint of the final tree neither exists yet nor, without row-specific equivalence notes, identifies the earlier dirty revisions from which most measurements were produced.
  Required next step: Replace the false checkpoint statement and record a durable source identifier for each measurement cohort (for example, a saved stage-owned patch/tree hash plus the base commit, with explicit row mapping). Where later source changes were behavior-equivalent for a row, record the exact change and equivalence basis. Regenerate JSON/CSV and verify parity. The normal post-clean-review checkpoint can remain a main-agent follow-up, but it must not be claimed as already recorded.

- P2: Complete exact precision-schema validation so policy fields cannot be accepted and ignored.
  Evidence: `PrecisionPolicy._validate()` exact-checks the immediate keys of `weight_groups`, `compute_fidelities`, `activation_residual`, `ccl`, `kv_cache`, and `logits_sampling`, but it does not exact-check the root object or the nested per-kind objects. An in-memory probe using the current class showed both `weight_groups.mlp_down.full_attentoin="bf16"` and `compute_fidelities.mlp_down.full_attentoin="hifi4"` are accepted while silently ignored. `tests/test_precision_policy.py` has no unknown-root or unknown-nested-key case. This contradicts the first review’s recorded remediation that the loader now “exact-validates section keys” and repeats the class of accepted-but-unconsumed policy metadata that prompted the propagation fix.
  Why this matters: The datatype-sweep contract requires the runtime to consume every selected dtype/fidelity field. A typo or extra per-kind policy entry currently looks valid yet has no runtime effect, making candidate summaries and future sweep rows capable of claiming an unmeasured policy.
  Required next step: Define and enforce exact allowed root and nested keys (including exactly `linear_attention` and `full_attention` for per-kind weight/fidelity maps), add regression tests for unknown root/nested fields, and retain the existing strict layout, block-size, logits, and layer-override checks. This is host-only remediation; no TT rerun is required for the unchanged selected values.

## Other Concerns

- The selected advantage remains small: the selected mean is 22.611371 t/s/u versus one baseline observation at 22.572120 t/s/u (+0.174%). Two selected runs and the separate default-path result at 22.606825 are tightly consistent, so this is not independently blocking, but cross-policy timing noise remains less well measured than within-selected repeatability.
- `doc/context_contract.json` has correct BFP8/BF16 capacity arithmetic and preserves 262,144 tokens, but `datatype_sweep.kv_cache_candidates.bf16.config_ids` omits the completed `canonical_runnable_bfp8_hifi2_kv_bf16` row while listing the failed canonical row. This does not change the capacity conclusion because the isolated BF16 KV control passed, but the candidate inventory should be made complete when regenerating documentation.
- `work_log.md` says eight host policy/static checks passed, while the current `tests/test_precision_policy.py` collects seven cases (two standalone tests plus five parameterizations), and no retained host-test log identifies an eighth check. Clarify the command/count or retain the log.

## Hard-Check Gaps

- The baseline metric artifacts predate `precision_summary`; their resolved policy is reconstructed from the candidate config and deterministic loader rather than embedded in the raw run. This remains acceptable for the current comparison but is weaker than the other completed rows.
- No retained log proves the claimed host-policy and 12-test readiness regression results. Source inspection supports the selected default propagation, and the hardware artifacts are not contradicted, but the exact check set cannot be independently replayed from a saved report in this read-only review.
- Actual dtype/fidelity application is proved primarily by resolved `precision_summary` plus construction/operation code inspection, not profiler rows. The code directly passes the selected weight dtypes and per-role compute configs into the material matmuls, so this is not a required new profiler run.

## Anomaly Ledger

- Observed anomaly: The first review’s missing BFP4 LoFi-versus-HiFi2 comparison.
  Evidence: `all_mlp_bfp4_hifi2.json` inherits the optimized baseline, changes full-attention down from BFP8 to BFP4, and changes all six linear/full gate/up/down fidelities from LoFi to HiFi2. Its raw `precision_summary` confirms all selected BFP4 weights and all non-MLP fields match the selected policy. Raw traced metrics are 0.95/1.00/1.00 and 19.670285 t/s/u versus the selected LoFi 0.95/1.00/1.00 and 22.611371 t/s/u mean.
  Affected path: Material linear- and full-attention MLP gate/up/down matmuls.
  Control or comparison: `full_down_bfp4_lofi` plus its repeat; identical accuracy and 13.0% higher LoFi throughput.
  Likely subsystem: Per-role MLP compute fidelity.
  Investigation performed: Resolved both recursive configs, compared raw precision summaries and metrics, verified ledger inclusion/ranking, and visually inspected both regenerated plots.
  Resolution: fixed.

- Observed anomaly: Per-row provenance columns exist but do not identify the dirty measured source and claim a nonexistent recorded checkpoint.
  Evidence: All rows share the same generic source-state prose; `work_log.md` says checkpoint SHAs are pending; HEAD remains the base commit; raw timestamps straddle later implementation edits.
  Affected path: JSON/CSV reproducibility and measurement provenance.
  Control or comparison: Branch, base commit, environment, hardware, mesh, command, and evidence paths are present and JSON/CSV semantic parity is otherwise exact.
  Likely subsystem: Artifact generation/provenance capture.
  Investigation performed: Compared every JSON/CSV row, raw timestamps, `git status`, HEAD, work log, and `build_artifacts.py` constants.
  Resolution: more-work-needed.

- Observed anomaly: Unknown root/nested policy keys can be accepted without runtime consumption.
  Evidence: Direct in-memory validation probes accepted misspelled extra keys in nested MLP dtype and fidelity maps; source inspection shows no exact root/nested key checks.
  Affected path: Precision-policy schema and future candidate/default construction.
  Control or comparison: Immediate section keys, KV layout/block size, logits assumptions, supported activation values, and layer-override names are rejected strictly; selected values themselves are consumed.
  Likely subsystem: `tt/precision.py` validation.
  Investigation performed: Source/test inspection plus host-only in-memory `PrecisionPolicy` probes; no file or device mutation.
  Resolution: more-work-needed.

- Observed anomaly: The canonical all-BFP8/HiFi2/BF16-cache policy hit the linear-down static-CB L1 boundary.
  Evidence: The failure artifact reports floor 928000 and CB end 1333760; AutoDebug reproduces that arithmetic exactly.
  Affected path: Linear-attention MLP down during traced warmup.
  Control or comparison: The one-field BFP4-linear-down repair reduces the endpoint to 811520 and its raw full-model run passes at 0.98/1.00/1.00 and 18.257202 t/s/u.
  Likely subsystem: DRAM-sharded BFP8 weight/static-CB L1 footprint.
  Investigation performed: Reviewed failure JSON, AutoDebug/AutoFix reports, repaired config, resolved runtime summary, and raw traced metrics.
  Resolution: fixed.

- Observed anomaly: All six qualitative generations end during visible reasoning without a final task answer.
  Evidence: Direct inspection of HF and TT outputs shows every completion is capped at 64 tokens; prompt metadata uses the exact checkpoint chat template and both sides show matching behavior.
  Affected path: Qualitative selected-config evidence.
  Control or comparison: Exact-checkpoint HF controls match TT prefixes for 11–63 tokens; degeneration checks pass with no wrong-language drift, control leakage, or mechanical repetition.
  Likely subsystem: Checkpoint reasoning style plus short generation cap, not TT precision/token feedback.
  Investigation performed: Compared prompt metadata, rendered prompts/token IDs, all raw outputs, and degeneration report.
  Resolution: controlled.

- Observed anomaly: The selected performance edge over baseline is small.
  Evidence: 22.611371 versus 22.572120 t/s/u; selected repeats differ by 0.026%, and the default-path result is 22.606825.
  Affected path: Candidate ranking.
  Control or comparison: Three selected-path observations are mutually consistent; all completed rows pass the accuracy gate and the selected row remains the fastest after adding HiFi2.
  Likely subsystem: Device timing variation versus the full-attention-down BFP4 change.
  Investigation performed: Re-derived repeat means, ranking, thresholds, and Pareto dominance from raw artifacts.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: supplied datatype-sweep contract; `.agents/skills/datatype-sweep/SKILL.md`, `.agents/skills/stage-review/SKILL.md`, `.agents/skills/tt-device-usage/SKILL.md`, `.agents/skills/qualitative-check/SKILL.md`, and `.agents/skills/autofix/SKILL.md`.
- Artifact paths: datatype-sweep `README.md`, `work_log.md`, first `stage_review.md`, selected/all candidate configs, JSON/CSV ledgers, every baseline/candidate/final raw metric or failure artifact, both Pareto PNGs, both JUnit files, qualitative metadata/HF/TT outputs/verdict/degeneracy report, canonical AutoDebug/AutoFix reports, `doc/context_contract.json`, and the AIME24 reference metadata/content.
- Code paths: `tt/precision.py`, `tt/model.py`, `tt/generator.py`, `tt/multichip_decoder.py`, `tt/optimized_decoder.py`, `tests/test_precision_policy.py`, relevant full-model benchmark tests, readiness-runner diffs, and `build_artifacts.py`.
- Commands run: read-only `git status/branch/rev-parse/diff`, `find`, `rg`, `sed`, `jq`, `stat`, JSON/CSV/path/ranking/Pareto analysis scripts, reference inspection, in-memory schema-validation probes, `git diff --check`, and visual inspection of both PNGs. No TT device, server, vLLM, reset, reservation, or hardware work was run.

## Residual Risk

- Once provenance and exact schema validation are repaired, no hardware rerun appears necessary: the new direct BFP4 fidelity comparison is valid, selected policy values are unchanged and consumed, ranking is correct, non-aligned/context gates are preserved, and qualitative behavior is controlled by the matching HF run.
- A future checkpoint of the final tree cannot retroactively prove an earlier dirty measurement state by itself. The remediation should preserve the distinction between exact source identity and an evidence-backed behavior-equivalence statement for later host-only validation edits.

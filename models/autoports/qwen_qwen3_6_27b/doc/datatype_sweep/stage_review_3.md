# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Reconcile and preserve the actual baseline command and precision-policy provenance.
  Evidence: `work_log.md` explicitly says, “The baseline omitted the override,” but the `baseline_optimized_mixed` row in both `sweep_results.json` and `sweep_results.csv` records an expanded command containing `QWEN36_PRECISION_CONFIG=.../candidates/baseline_optimized_mixed.json`. `build_artifacts.py` deterministically synthesizes that row with `result_from_metrics(baseline_metrics, baseline_policy)`, so the ledger command is reconstructed rather than copied from the run. The raw baseline teacher-forcing artifact was generated at 18:00 and contains no `precision_summary`; the current baseline candidate and selected-config files were subsequently modified at 19:06/19:07. The new cohort manifest honestly says the exact historical dirty patch was not preserved, but its equivalence statement does not address the changed default-config artifact or the baseline-only readiness reporting change that added `precision_summary` after this run.
  Why this matters: The datatype-sweep contract requires the exact command and active policy for every evaluated row. The baseline is the accuracy/performance control and is plotted only 0.174% below the selected result, so a contradictory reconstructed command plus a raw artifact without the resolved policy leaves a material reproducibility gap in the selection evidence.
  Required next step: Determine which baseline command actually ran and make the README, work log, JSON, and CSV agree. Preserve the strongest available evidence that the no-override default at that time resolved to `baseline_optimized_mixed` (for example a contemporaneous log/artifact, saved config identity, or other durable run evidence). If exact identity was not preserved, state that limitation row-specifically and extend the cohort equivalence record to cover the historical default-config state and the reporting-only readiness-runner change; do not label a synthesized override command as the exact command.

## Other Concerns

- The selected mean, 22.611371 t/s/u, is only 0.174% above the single baseline observation at 22.572120 t/s/u. Two selected candidate runs and the separate default-path result at 22.606825 are tightly consistent, so this is not a separate blocker, but cross-policy timing noise remains less characterized than within-selected repeatability.
- All six 64-token qualitative completions stop inside visible reasoning without reaching task answers. Exact-checkpoint HF controls do the same and TT remains coherent and nondegenerate, so this is controlled prefix-quality evidence rather than a TT regression.
- The representative token-out artifact retains only the first and last generated token, both token 198, rather than the full 128-token sequence. It is used as performance evidence, while the separate prompt-correct qualitative suite covers output quality, so this is not independently blocking.

## Hard-Check Gaps

- `evidence/baseline/teacher_forcing_metrics.json` and `prefill_metrics.json` predate live `precision_summary` capture. This becomes material only in combination with the contradictory baseline command above.
- `evidence/final/token_out_metrics.json` records all 64 layer indices, trace timing, sampler mode, workload shape, and caller-visible timing, but not the resolved precision summary. The opt-in test constructs `Generator` without an override and the final teacher-forcing artifact separately proves the default selected policy.
- `non_aligned.junit.xml` retains only the passing test identity, not console details. Source inspection confirms that exact test exercises selected-default four-layer prefill/decode with mixed logical lengths 65/67, split trace capture/replay, changed-only page tables, and seeded/unseeded sampling.
- Runtime dtype/fidelity consumption is established by resolved metric summaries plus direct construction/operation code, not profiler rows. The code passes selected weight dtypes and per-role compute configs into the material projection, MLP, LM-head, CCL, and KV-cache paths without a contradictory artifact.

## Anomaly Ledger

- Observed anomaly: The baseline command differs between the work log and both ledgers.
  Evidence: `work_log.md` says no override; the JSON/CSV row contains `QWEN36_PRECISION_CONFIG=...baseline_optimized_mixed.json`; `build_artifacts.py` synthesizes the latter; raw baseline metrics have no `precision_summary`.
  Affected path: Baseline accuracy/performance control, Pareto ranking, and per-row provenance.
  Control or comparison: Later candidate metrics embed resolved summaries, and the final default-path metrics embed the selected summary; the baseline does not.
  Likely subsystem: Artifact generation and historical command/config provenance.
  Investigation performed: Compared raw timestamps, work-log prose, JSON/CSV rows, candidate/default config timestamps, and ledger-generation code.
  Resolution: more-work-needed.

- Observed anomaly: The prior review found unknown root and nested precision keys were accepted and ignored.
  Evidence: Current `tt/precision.py` rejects extra root keys, requires exactly `linear_attention` and `full_attention` in nested weight/fidelity maps, and exact-validates runtime-loading keys. `host_policy_readiness.junit.xml` proves 10/10 policy cases plus 7/7 readiness cases; `full_model_static.junit.xml` proves 1/1.
  Affected path: Precision-policy schema and future candidate construction.
  Control or comparison: The retained JUnit includes all three typo probes named in `tests/test_precision_policy.py`.
  Likely subsystem: Host precision-policy validation.
  Investigation performed: Inspected validator/test source and every retained JUnit testcase.
  Resolution: fixed.

- Observed anomaly: Exact historical uncommitted runtime patches were not preserved.
  Evidence: `source_provenance.json` explicitly sets `exact_historical_dirty_patch_preserved=false`, maps all nine rows to `pre_strict_policy_v1` or `strict_policy_v2`, and does not claim a final commit was measured. JSON/CSV parity is exact, cohort coverage has no omissions or duplicates, every referenced evidence/config/manifest path exists, and the final runtime source hash recomputes to `6ab2b2093de4652e7e358cc2afda422f327b4cccf8a0778e85d52671bd46f2a7`.
  Affected path: Dirty-source reproducibility.
  Control or comparison: The manifest gives behavior-equivalence arguments for unchanged valid page-block-64 policies, unreachable projection HiFi4, and validation-only root/nested rejection.
  Likely subsystem: Measurement provenance capture.
  Investigation performed: Recomputed the source hash and checked manifest/row/CSV parity, timestamps, branch, base commit, paths, and unrelated exclusions.
  Resolution: controlled for the explicitly described implementation changes, but the baseline command/default-config discrepancy above remains more-work-needed.

- Observed anomaly: The first review found no same-dtype BFP4 LoFi/HiFi2 comparison.
  Evidence: `all_mlp_bfp4_hifi2` matches selected BFP4 gate/up/down weights and every non-MLP policy field, changing only the six material MLP fidelities to HiFi2. Raw traced results are 0.95/1.00/1.00 and 19.670285 t/s/u versus selected LoFi 0.95/1.00/1.00 and 22.611371 t/s/u.
  Affected path: Material linear- and full-attention MLP precision selection.
  Control or comparison: Both raw precision summaries match their recursively resolved configs; JSON/CSV rows and both plots include the comparison.
  Likely subsystem: MLP compute fidelity.
  Investigation performed: Resolved configs, compared raw summaries/metrics, re-derived ranking, and visually inspected both Pareto plots.
  Resolution: fixed.

- Observed anomaly: The original canonical all-BFP8/HiFi2/BF16-cache policy failed traced warmup at an L1/static-CB collision.
  Evidence: The failure artifact reports L1 floor 928000 and CB end 1333760; AutoDebug derives the same endpoint. The one-field BFP4 linear-down repair reduces the endpoint to 811520 and passes traced full-model teacher forcing at 0.98/1.00/1.00 and 18.257202 t/s/u.
  Affected path: Linear-attention MLP down during traced decode warmup.
  Control or comparison: `canonical_runnable_bfp8_hifi2_kv_bf16` preserves the other canonical fields and completes the original workload.
  Likely subsystem: DRAM-sharded BFP8 weight/static-CB L1 footprint.
  Investigation performed: Reviewed failure JSON, AutoDebug/AutoFix reports, repaired config, resolved runtime summary, and raw metrics.
  Resolution: fixed.

- Observed anomaly: The selected performance advantage is small.
  Evidence: Selected repeats are 22.614326 and 22.608416 t/s/u (0.026% spread), with a separate default-path result of 22.606825, versus baseline 22.572120.
  Affected path: Candidate ranking.
  Control or comparison: Three selected-path measurements are mutually consistent and all completed points pass the stated accuracy gate.
  Likely subsystem: Device timing variability versus the full-attention-down BFP4 change.
  Investigation performed: Re-derived repeat means, thresholds, ordering, and Pareto dominance from raw metrics.
  Resolution: controlled, with residual cross-policy timing risk.

- Observed anomaly: Qualitative outputs end during visible reasoning.
  Evidence: Direct inspection shows all six HF and TT completions are capped at 64 tokens. Prompt metadata records the exact checkpoint, tokenizer, chat template, rendered prompts, and greedy settings; TT/HF matching prefixes range from 11 to 63 tokens and all degeneration checks pass.
  Affected path: Selected-config qualitative evidence.
  Control or comparison: Exact-checkpoint HF outputs show the same reasoning style and truncation for every prompt.
  Likely subsystem: Checkpoint behavior plus the short generation cap, not TT token feedback or precision.
  Investigation performed: Read all rendered prompts, token IDs, HF/TT outputs, degeneration checks, and manual verdict.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: supplied datatype-sweep contract; `.agents/skills/datatype-sweep/SKILL.md`, `.agents/skills/stage-review/SKILL.md`, `.agents/skills/tt-device-usage/SKILL.md`, `.agents/skills/qualitative-check/SKILL.md`, and `.agents/skills/autofix/SKILL.md`.
- Artifact paths: datatype-sweep README/work log and both prior reviews; selected/all candidate configs; source-provenance manifest; JSON/CSV ledgers; all baseline/candidate/final raw metrics and failure evidence; all four retained JUnits; qualitative metadata, prompts, HF/TT outputs, degeneracy report, and verdict; both Pareto PNGs; canonical AutoDebug/AutoFix reports; `doc/context_contract.json`; and the 100-token AIME24 reference.
- Code paths: `tt/precision.py`, `tt/model.py`, `tt/generator.py`, `tt/multichip_decoder.py`, `tt/optimized_decoder.py`, `tests/test_precision_policy.py`, relevant full-model/nonaligned/token-out tests, qualitative runner, readiness-runner diffs, and `build_artifacts.py`.
- Commands run: read-only `git status/branch/rev-parse/diff`, `find`, `rg`, `sed`, `jq`, `stat`, `wc`, JSON/CSV/path/cohort/hash/config/ranking analysis scripts, reference inspection, `git diff --check`, and visual inspection of both PNGs. No TT device, server, vLLM, reset, reservation, or hardware command was run.
- Scope isolation: No vLLM implementation was added. Stage-owned implementation changes are confined to the disclosed autoport and readiness-runner paths; unrelated dirty Tracy/UMD submodules and `tt-cluster-descriptors` were excluded.

## Residual Risk

- After the baseline command/default-policy evidence is reconciled, no additional hardware rerun appears necessary if retained evidence can establish the measured baseline policy. The strict-schema change is validation-only for all recorded valid configs, the selected values are unchanged and consumed, the direct BFP4 fidelity comparison is complete, BF16/BFP8 cache candidates preserve the 262,144-token shared pool, and qualitative behavior is controlled by matching HF output.
- A later local checkpoint can make the final implementation durable but cannot retroactively identify an earlier dirty measurement. The stage must continue to distinguish the final source hash from the measured cohort and must not replace the baseline row's actual command with a reconstructed one.

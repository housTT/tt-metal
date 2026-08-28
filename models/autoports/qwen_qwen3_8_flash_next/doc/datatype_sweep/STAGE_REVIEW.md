# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- `sweep_results.csv` is still matched by the repository-wide `*.csv` ignore rule, so it had to be force-tracked. I do not classify this as required work because `git ls-files --error-unmatch` resolves it, the committed CSV is LF-only, it parses as 16 result rows with the same 30 fields and config order as `sweep_results.json`, and temp-copy `finalize_sweep.py` regeneration reproduced it byte-for-byte.
- The retained result corpus contains two source digests because earlier primary sweep rows are preserved alongside current-source anomaly-control/post-selection rows. I do not classify this as required work: each selection cohort has one internally consistent digest, the close promoted selected policy and displaced shared-HiFi2 finalist were both rerun under current digest `8e1a9cf08fa18531087c5d4dd120b3a74458784925bc02de899b9975abe75b72`, raw measurements retain their digests, and the final selected normal-construction teacher-forcing/token-out artifacts use the current default source.
- The advertised-context construction JSONs intentionally retain the displaced shared-HiFi2 construction IDs. This is acceptable for this capacity-only evidence because the context-candidate JSONs explicitly name the final selected config, set `construction_capacity_equivalent_to_selected=true`, and explain that the reused construction differs only in shared-projection compute fidelity, not allocated tensor dtype, geometry, KV cache, endpoint footprint, or capacity bytes.
- Host-backed exact expert H2D timing remains variable enough to affect wall-clock throughput samples. The stage now treats this as controlled evidence rather than dismissing it: the challenged rows have current-source controls, the selected policy uses a three-sample current-source median, the displaced finalist uses a two-sample current-source median, and the transient slow displaced-policy sample is retained only under an explicitly named anomaly-control path.
- The qualitative shared suite is prompt-format correct and mechanically non-degenerate, but it is not a semantic quality proof. `qualitative_review.md` accurately labels the fixed 128-token explanation/coding truncation, and comparison against prior full-model/optimized-full-model qualitative artifacts shows the same cap-shaped behavior rather than a new datatype-stage regression.
- The post-selection logs contain nanobind reference-leak warnings during teardown. I do not classify these as required datatype-sweep work because the corresponding JUnit XMLs pass with zero failures/errors, the devices close cleanly in the logs, and the warnings do not mention precision-policy, cache, trace replay, host-weight semantics, or output corruption.
- No vLLM adapter was reviewed. The stage explicitly records that vLLM integration was not started; the later vLLM stage must still prove that its adapter consumes `selected_precision_config.json`.

## Hard-Check Gaps

- I did not run TT hardware, reserve/reset devices, start servers, run vLLM, or rerun full-model hardware tests. This review is static/read-only except for overwriting this report, per the review instruction.
- I did not require a fresh advertised-context construction under the exact final selected shared-LoFi config because the retained BFP8/BF16 context-candidate artifacts record and justify capacity equivalence for the only fields relevant to context/capacity.
- I did not require semantic acceptance of the fixed-cap qualitative suite. The stage uses AIME24 top-1/top-5/top-100 and traced teacher-forcing throughput for datatype selection, while qualitative evidence is prompt-format/prohibited-fallback/non-degeneration coverage.

## Anomaly Ledger

- Observed anomaly: The required CSV artifact is ignored by the repository-wide `*.csv` rule.
  Evidence: `git check-ignore -v --no-index models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/sweep_results.csv` reports `.gitignore:8:*.csv`; `git ls-files --error-unmatch` resolves the CSV as tracked. The file is 99,596 bytes, SHA256 `953c17aeadf2c694c662ae041985bae30d98baf22f4a7ee8d72e9027e635387b`, contains no CR bytes, ends in LF, parses as 16 rows, and its field values match `sweep_results.json`.
  Affected path: Required sweep ledger artifact.
  Control or comparison: A temp-copy run of `finalize_sweep.py` reproduced `sweep_results.json`, `sweep_results.csv`, and `selected_precision_config.json` byte-for-byte and printed `{"evaluated": 16, "passing": 14, "selected": "qsa_bfp8_hifi2_lm_head_bf16_hifi2"}`.
  Likely subsystem: Artifact tracking/generation for ignored CSV files.
  Investigation performed: Checked git tracking/ignore status, parsed CSV and JSON, compared JSON-serialized CSV fields against ledger rows, inspected the `csv.DictWriter(..., lineterminator="\n")` writer, and regenerated in a temporary copy outside the worktree.
  Resolution: fixed.

- Observed anomaly: Candidate-generation and result-policy drift had previously been possible.
  Evidence: Current `make_candidates.py` uses `BASELINE_SEED = ROOT / "baseline_precision_config.json"`, and `python models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/make_candidates.py --check` returned `{"artifacts": 17, "reproducible": true}`. `selected_precision_config.json` is byte-identical to `candidates/qsa_bfp8_hifi2_lm_head_bf16_hifi2.json`. A direct audit of 30 primary, replicate, anomaly-control, and selected teacher-forcing result JSONs found zero config-ID mismatches and zero 61-leaf propagation mismatches against their candidate JSONs.
  Affected path: Reproducibility of `candidate_matrix.json`, `candidates/*.json`, `selected_precision_config.json`, and sweep result rows.
  Control or comparison: `static_contracts_final.xml` includes passing `test_datatype_sweep_candidate_matrix_is_reproducible_and_matches_results`; `finalize_sweep.py` independently checks result config IDs and expected precision-propagation leaves.
  Likely subsystem: Candidate generation base policy and result-finalization guardrails.
  Investigation performed: Inspected `make_candidates.py` and `finalize_sweep.py`, ran the candidate check, compared selected/candidate bytes, and independently validated all retained result propagation checks.
  Resolution: fixed.

- Observed anomaly: Host H2D service-time variance materially changes some throughput samples.
  Evidence: `sweep_results.json` records the selected `qsa_bfp8_hifi2_lm_head_bf16_hifi2` row as a three-sample current-source anomaly-control median at 4.384021 traced teacher-forcing tokens/s/user, while the displaced shared-HiFi2 finalist uses a two-sample current-source median at 4.366890. The preserved displaced transient has the same 50,059 misses and 138,403,123,200 H2D bytes as its normal control but much larger H2D time.
  Affected path: Pareto ranking and fastest-passing policy selection.
  Control or comparison: Raw measurements and host counters remain in `sweep_results.json`; the selected final normal-construction teacher-forcing artifact measures 4.386243 tokens/s/user with 50,069 misses, 138,430,771,200 H2D bytes, 16.805 s H2D time, 102 PLE lookups, and 3,480 real PLE rows.
  Likely subsystem: Host-backed exact expert DMA submission/runtime scheduling variance.
  Investigation performed: Parsed selection cohorts, raw measurements, source digests, and host-service counters from candidate-result artifacts.
  Resolution: controlled.

- Observed anomaly: Context construction evidence uses displaced shared-HiFi2 config IDs rather than exact final selected config IDs.
  Evidence: `context_contract_candidates/kv_bfp8.json` and `kv_bf16.json` record `selected_config_id=qsa_bfp8_hifi2_lm_head_bf16_hifi2`, `construction_capacity_equivalent_to_selected=true`, and exact equivalence rationale. BFP8 capacity remains 262,144 tokens with 2,340,421,632 cache bytes/device and 24,220,969,896 bytes/device headroom; BF16 capacity remains 262,144 tokens with 4,227,858,432 cache bytes/device and 22,333,533,096 bytes/device headroom.
  Affected path: Context/capability evidence labeling.
  Control or comparison: `doc/context_contract.json` mirrors both datatype-sweep context candidates, keeps `capability_reduction=null`, and preserves `current_supported_context=262144`. Referenced BFP8 and BF16 construction JSONs exist and report 48-layer construction with 36 QSA cache tensors at `max_seq_len=262144`.
  Likely subsystem: Context-capacity evidence reuse after final winner promotion.
  Investigation performed: Parsed `doc/context_contract.json`, context candidate JSONs, construction JSONs, and referenced non-aligned evidence paths.
  Resolution: controlled.

- Observed anomaly: Qualitative explanation/coding outputs are incomplete at the fixed 128-token cap.
  Evidence: `post_selection/qualitative/qualitative_shared_suite_final.json` records chat prompt mode, tokenizer `apply_chat_template(add_generation_prompt=True)`, prompt IDs `explanation`, `coding`, and `summarization`, generation length 128, final selected config ID, 61/61 consumed precision leaves, and non-degenerate HF/TT mechanical reviews. Manual review records explanation as coherent but incomplete, coding as still in reasoning, and summarization as complete/correct; HF controls are also capped/truncated for explanation/coding.
  Affected path: Prompt-based qualitative evidence.
  Control or comparison: `qualitative_review.md` records the limitation; prior `doc/full_model/qualitative_shared_suite_final.json` and `doc/optimized_full_model/qualitative_shared_suite_final.json` show the same fixed-cap qualitative shape, so this is not a new datatype-stage regression.
  Likely subsystem: Fixed qualitative generation cap and reasoning-model output format.
  Investigation performed: Read the qualitative-check skill, qualitative review, selected qualitative JSON, and prior full-model qualitative artifacts.
  Resolution: controlled.

- Observed anomaly: An earlier wrong-mesh static run failed.
  Evidence: `static_contracts_wrong_mesh.xml` records failures from a 1x1 fixture. The authoritative `static_contracts_final.xml` records 37 collected tests: 24 passed, 13 explicitly gated skipped, 0 failures, and 0 errors.
  Affected path: Static validation provenance.
  Control or comparison: `static_contracts_final.xml` on the correct 1x2 fixture.
  Likely subsystem: Test environment setup.
  Investigation performed: Parsed JUnit summaries and inspected passing/skipped test names.
  Resolution: controlled.

- Observed anomaly: Post-selection logs print nanobind reference-leak warnings at teardown.
  Evidence: `post_selection/teacher_forcing_selected.log`, `post_selection/token_out.log`, and `post_selection/qualitative.log` end with nanobind ref-leak warnings followed by JIT telemetry and clean device close messages; their JUnit XML aliases each report zero failures and zero errors.
  Affected path: Test teardown diagnostics.
  Control or comparison: Passing `post_selection/*.xml` and clean UMD cluster close messages in the same logs.
  Likely subsystem: TTNN/nanobind binding teardown, not datatype policy/runtime output.
  Investigation performed: Inspected post-selection log tails and parsed corresponding JUnit XML summaries.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths:
  - `.agents/skills/stage-review/SKILL.md`
  - `.agents/skills/datatype-sweep/SKILL.md`
  - `.agents/skills/qualitative-check/SKILL.md`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/README.md`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/work_log.md`
  - previous `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/STAGE_REVIEW.md`

- Artifact paths:
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/sweep_results.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/sweep_results.csv`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/baseline_precision_config.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/candidate_matrix.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/candidates/*.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/full_runs/**`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/replicates/**`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/anomaly_controls/**`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/post_selection/**`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/context_contract_candidates/*.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/static_contracts*.xml`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/qualitative_review.md`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/top1_perf_pareto.png`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/top5_perf_pareto.png`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/readiness_aime24_chat.refpt`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/qualitative_shared_suite_final.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/qualitative_shared_suite_final.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/context_contract.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/host_weight_contract.json`

- Code paths:
  - `models/autoports/qwen_qwen3_8_flash_next/tt/precision_config.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tt/model.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tt/generator.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tt/functional_decoder.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tt/optimized_decoder.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tt/host_weight_cache.py`
  - `models/autoports/qwen_qwen3_8_flash_next/demo/full_model.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tests/test_host_weight_cache.py`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/make_candidates.py`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/finalize_sweep.py`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/recompute_context_contract.py`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/update_host_contract.py`

- Commands run:
  - `sed` over `stage-review`, `datatype-sweep`, `qualitative-check`, stage README/work log, precision plumbing, and selected tests.
  - `git status --short`, `git log --oneline`, `git show --stat/--name-status`, `git ls-files`, `git check-ignore -v --no-index`, and scoped `git diff --check`.
  - `find`, `rg`, `file`, and log `tail` for artifact/source discovery and stale-path checks.
  - `python models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/make_candidates.py --check`.
  - Temp-copy `finalize_sweep.py` regeneration outside the worktree, comparing regenerated `sweep_results.json`, `sweep_results.csv`, and `selected_precision_config.json` byte-for-byte against the committed artifacts.
  - Python read-only audits over JSON/CSV/XML artifacts for ledger parity, 30 result/candidate propagation matches, selected post-selection metrics, host fallback flags, context capacity, qualitative controls, AIME24 reference shape, AIME100 evidence, JUnit summaries, image metadata, source digests, and AST parseability.

## Residual Risk

- Hardware evidence was not rerun during this review. The clean-pass verdict depends on the recorded candidate, post-selection, context, qualitative, and static artifacts being the accepted evidence for this stage.
- Some non-finalist sweep rows remain historical-source measurements. The stage records their source digests and raw measurements; if a later consumer wants a fully same-source performance table, it can rerun those rows, but no required datatype-sweep gate remains open.
- Later vLLM/serving integration still needs its own default-policy propagation check once a vLLM adapter exists.

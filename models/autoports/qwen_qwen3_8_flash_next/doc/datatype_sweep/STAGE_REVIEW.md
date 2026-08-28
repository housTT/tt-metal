# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The advertised-context construction JSON/XML artifacts intentionally retain the displaced shared-HiFi2 construction IDs, while the context-candidate JSONs now name the final selected config separately and state `construction_capacity_equivalent_to_selected=true`. I do not classify this as required work because the recorded rationale is exact for this capacity question: the displaced construction and final selected-derived capacity policy have identical weight dtypes, KV geometry, KV dtype, and endpoint footprint, and differ only in shared-projection compute fidelity, which does not change allocated tensor bytes.
- The host-backed measurement regime still has material host H2D timing variance. This is now controlled for selection rather than dismissed: the challenged rows are reranked from current-source anomaly-control cohorts, raw measurements remain in `sweep_results.json`, and the displaced shared-HiFi2 transient is kept only under the explicitly named anomaly-control directory.
- The qualitative shared suite is prompt-format correct and mechanically non-degenerate, but it is not a semantic-quality proof. `qualitative_review.md` accurately classifies the 128-token explanation/coding truncation and does not present that diagnostic as replacing the AIME24 full-model top-1/top-5/top-100 gate.
- No vLLM adapter was reviewed. The stage explicitly records that vLLM integration was not started; the later vLLM stage must prove it consumes `selected_precision_config.json`.
- The worktree is live and dirty/untracked across stage-owned files plus broader model-bringup artifacts. That is acceptable for this review, but the main agent's post-review checkpoint must isolate only stage-owned changes.

## Hard-Check Gaps

- I did not run TT hardware, reserve/reset devices, start servers, run vLLM, or rerun full-model tests. This review was static/read-only except for overwriting this report, per the review instruction.
- I did not regenerate the context construction artifacts under the exact final selected LoFi shared-projection config. Existing context artifacts are accepted only because the context candidates and `context_contract.json` explicitly document the displaced construction IDs and prove capacity equivalence for the fields that affect cache/context memory.
- I did not require a new empirical run for the displaced-policy artifact relocation. Static checks show the top-level post-selection XML aliases are byte-identical to the nested final-selected XMLs, and a direct search found no displaced-policy ID under `post_selection/`.

## Anomaly Ledger

- Observed anomaly: Prior post-selection XML/log aliases described the displaced shared-HiFi2 policy.
  Evidence: Current `post_selection/teacher_forcing_selected.xml`, `post_selection/token_out.xml`, `post_selection/qualitative.xml`, and `post_selection/precision_smoke.xml` are byte-identical to their nested `result_final_selected.xml` counterparts. Parsed XML/JSON show final `config_id=qsa_bfp8_hifi2_lm_head_bf16_hifi2`, source digest `8e1a9cf08fa18531087c5d4dd120b3a74458784925bc02de899b9975abe75b72`, selected teacher-forcing 4.386243 tokens/s/user, selected token-out 4.317901 tokens/s/user, and 61/61 consumed policy leaves. `rg` found no `qsa_bfp8_hifi2_shared_bfp8_hifi2_lm_head_bf16_hifi2`, `result_current_exact`, or `88b0cdba` match under `post_selection/`.
  Affected path: Final selected-policy post-selection provenance.
  Control or comparison: Displaced policy artifacts now live under `anomaly_controls/qsa_bfp8_hifi2_shared_bfp8_hifi2_lm_head_bf16_hifi2/`, including `historical_post_selection_88b/`, `transient_h2d/`, and explicitly named control subdirectories.
  Likely subsystem: Artifact retention/naming after winner promotion.
  Investigation performed: Compared top-level and nested XML hashes, parsed JUnit properties, searched post-selection logs/XML/JSON for old policy IDs, and listed the displaced-policy anomaly-control directory.
  Resolution: fixed.

- Observed anomaly: Candidate generation previously used mutable `selected_precision_config.json` as its base and could drift after winner promotion.
  Evidence: `make_candidates.py` now uses `BASELINE_SEED = ROOT / "baseline_precision_config.json"`. `python models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/make_candidates.py --check` returned `{"artifacts": 17, "reproducible": true}`. `baseline_precision_config.json` is byte-identical to `candidates/baseline_optimized_bfp4lofi_bfp8hifi2.json`, and `selected_precision_config.json` is byte-identical to `candidates/qsa_bfp8_hifi2_lm_head_bf16_hifi2.json`.
  Affected path: Reproducibility of `candidate_matrix.json`, `candidates/*.json`, and regenerated sweep ledgers.
  Control or comparison: A static audit over 31 primary, replicate, anomaly-control, and selected result artifacts found zero config-ID or 61-leaf policy mismatches against the retained candidate JSONs. `static_contracts_final.xml` includes passing `test_datatype_sweep_candidate_matrix_is_reproducible_and_matches_results`.
  Likely subsystem: Artifact-generation script base policy and candidate/result guardrails.
  Investigation performed: Ran `make_candidates.py --check`, inspected `make_candidates.py`, inspected `finalize_sweep.py`, compared selected/baseline hashes, and checked candidate/result leaves.
  Resolution: fixed.

- Observed anomaly: Host H2D service time variance materially changes throughput samples.
  Evidence: The selected `qsa_bfp8_hifi2_lm_head_bf16_hifi2` row uses three current-source anomaly-control samples with median 4.384021 tokens/s/user and roughly 16.6-16.8 s expert H2D, while its older primary row remains retained but unused at 4.050711 tokens/s/user and 32.267 s H2D. The displaced shared-HiFi2 finalist uses two current-source controls with median 4.366890 tokens/s/user, and the preserved transient has the same miss/byte class but much larger H2D time.
  Affected path: Sweep ranking and fastest-passing selected policy.
  Control or comparison: `sweep_results.json` records selection cohorts and raw retained measurements for selected, displaced finalist, GDN LoFi, BF16 KV, BF16 LM head, QSA LoFi, and residual BFP8 rows.
  Likely subsystem: Host-backed exact expert DMA submission/runtime scheduling variance.
  Investigation performed: Parsed `sweep_results.json` raw measurements and checked the host counters recorded in candidate-result artifacts.
  Resolution: controlled.

- Observed anomaly: Context construction evidence uses displaced shared-HiFi2 config IDs rather than exact final selected config IDs.
  Evidence: `context_contract_candidates/kv_bfp8.json` and `kv_bf16.json` both record `selected_config_id=qsa_bfp8_hifi2_lm_head_bf16_hifi2`, the displaced `construction_evidence_config_id`, `construction_capacity_equivalent_to_selected=true`, and an exact equivalence reason. BFP8 context records 262,144 tokens, 2,340,421,632 cache bytes/device, and 24,220,969,896 bytes/device headroom; BF16 records 262,144 tokens, 4,227,858,432 cache bytes/device, and 22,333,533,096 bytes/device headroom. The referenced construction and non-aligned evidence paths all exist.
  Affected path: Context-capability evidence labeling.
  Control or comparison: `doc/context_contract.json` mirrors both context candidates, keeps `capability_reduction=null`, and preserves 262,144 supported context tokens. Selected post-selection precision propagation separately proves the final selected runtime policy.
  Likely subsystem: Context-capacity evidence reuse after final winner promotion.
  Investigation performed: Parsed context candidate JSONs, `context_contract.json`, selected config, and referenced path existence.
  Resolution: controlled.

- Observed anomaly: Qualitative explanation/coding outputs are incomplete under the fixed 128-token cap.
  Evidence: `post_selection/qualitative/qualitative_shared_suite_final.json` records chat prompt mode, tokenizer chat-template rendering with `add_generation_prompt=True`, three prompt IDs, generation length 128, final selected config, and all 61 precision leaves consumed. Manual review classifies explanation as coherent but incomplete, coding as still in reasoning, and summarization as complete/correct; HF controls are also capped/truncated for explanation/coding.
  Affected path: Prompt-based qualitative evidence.
  Control or comparison: `qualitative_review.md` and the qualitative JSON record HF controls, TT completions, prompt-format metadata, runtime fallback audits, and non-degenerate mechanical checks.
  Likely subsystem: Fixed qualitative generation cap and reasoning-model output format, not datatype policy selection.
  Investigation performed: Read `qualitative_review.md`, inspected qualitative metadata, and sampled the HF/TT completions.
  Resolution: controlled.

- Observed anomaly: An earlier wrong-mesh static run failed.
  Evidence: `static_contracts_wrong_mesh.xml` records 1x1 fixture failures. The authoritative `static_contracts_final.xml` records 37 collected tests: 24 passed, 13 explicitly gated skipped, 0 failures, and 0 errors.
  Affected path: Static validation provenance.
  Control or comparison: `static_contracts_final.xml`.
  Likely subsystem: Test fixture/environment setup.
  Investigation performed: Parsed JUnit summaries and skipped-test messages.
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
  - `sed`, `nl`, `rg --files`, `rg`, `find`, `file`, `sha256sum`, `git rev-parse`, and `git status --short` for source/artifact inspection.
  - `python models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/make_candidates.py --check`.
  - Python read-only JSON/XML audits over `sweep_results.json`, `sweep_results.csv`, candidate files, primary/replicate/anomaly-control results, post-selection XML/JSON/logs, context contracts, host-weight contract, qualitative evidence, and JUnit XML summaries.
  - Python `ast.parse` checks over the datatype-sweep scripts, `test_host_weight_cache.py`, and `tt/precision_config.py`.
  - `git diff --check --` scoped to datatype-sweep/default-policy source and contract files.

## Residual Risk

- Hardware was intentionally not rerun. The clean-pass verdict depends on the recorded full-model, post-selection, context, and static artifacts being the accepted evidence for this stage.
- Later vLLM/serving integration still needs an explicit propagation check through its adapter once that adapter exists.
- The main agent still needs to append this clean review result and create isolated local checkpoint commit(s) for stage-owned changes before marking the broader stage handoff complete.

# Stage 7 Optimized Full Model Rereview

Verdict: clean-pass

Target model: `Qwen/Qwen3.8-Flash-Next`
Autoport root: `models/autoports/qwen_qwen3_8_flash_next`
Evidence root: `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model`
Reviewed branch/base: `hous/qwen3.8-flash-next`, base/head `a38187012fbc55714487c3d53a78a848809c8cf1`
Claimed frozen source digest: `e85ca93a2fc788bcd70095d93284f538ab2f2ae30754b0ed7523c1bbe6a631d6`

## Required Work

- None.

## Clean-Pass Basis

- The final docs now point unambiguously at the `final_frozen_*` artifacts, and I treated `README.md`, `work_log.md`, and `AUTOFIX.md` as claims rather than proof.
- Selected frozen source provenance is current. The embedded/source-provenance checks for `final_frozen_performance_no_timeline/full_model_performance.json`, `final_frozen_performance/full_model_performance.json`, `final_frozen_accuracy/aime24_autoregressive_100_report_final.json`, `final_frozen_qualitative/qualitative_shared_suite_final.json`, and `final_frozen_profiler/source_provenance.json` all report digest `e85ca93a2fc788bcd70095d93284f538ab2f2ae30754b0ed7523c1bbe6a631d6`; all recorded per-file hashes match the current source.
- Selected final gates pass:
  - `final_frozen_aime_prefill_teacher_autoreg.xml`: 3 tests, 0 failures/errors/skips.
  - `final_frozen_cold_warm.xml`: 1 test, 0 failures/errors/skips.
  - `final_frozen_split_sampling_mixed_contracts.xml`: 5 tests, 0 failures/errors, 1 skip for the env-gated sampler A/B node; the separate `final_frozen_sampler_strategy_ab.xml` reruns that node and passes.
  - `final_frozen_sampler_strategy_ab.xml`: 1 test, 0 failures/errors/skips.
  - `final_frozen_batch32_and_context_capacity.xml`: 2 tests, 0 failures/errors/skips.
  - `final_frozen_full48_watcher_alloc_tracker.xml`: 1 test, 0 failures/errors/skips.
  - `final_frozen_static_contracts.xml`: 36 tests, 0 failures/errors/skips.
  - `final_frozen_lm_head_bfp8_hifi2.xml`: 1 test, 0 failures/errors/skips.
- Final performance numbers rederive from the JSON/CSV evidence:
  - warmed, no-timeline token-out: TTFT `7.856140708 s`, decode `0.231489929 s/token`, `4.319842358 tokens/s/user`, 126 measured traced decode tokens, device sampling, not teacher forced.
  - instrumented token-out: `0.261457291 s/token`; timeline mean equals the reported value.
  - teacher-forcing evidence is correctly labeled separately as full-logit teacher forcing, not token-out sampling.
- The lower-bound arithmetic is now honest and reproducible:
  - decoder stack `111.817691 ms/token`;
  - selected LM-head plus traced greedy sampling floor `1.387908 ms/token`;
  - raw dual-owner H2D physical floor `57.952608 ms/token`;
  - PLE lookup/staging `5.655693 ms/token`;
  - optimistic physical lower bound `176.813899 ms/token`;
  - observed token-out `231.489929 ms/token`;
  - remaining gap `54.676029 ms/token`, or `30.923%` of the optimistic lower bound.
- The remaining >15% gap is not misrepresented as closed. The evidence separates the raw physical H2D floor from measured exact Python/TTNN implementation diagnostics: pure dual-owner staging is `140.511592 ms/token` for the observed bytes, the additive implementation diagnostic is `259.372884 ms/token`, and the observed run is below that because exact transfers overlap stack work. The docs and `AUTOFIX.md` leave the residual as a measured native-API limitation after exact Python-level variants failed.
- The LM-head/profiler prior findings are addressed:
  - selected BFP8/HiFi2 LM head passes with PCC `0.999922454`, top-5 overlap `5`, top-100 overlap `98`, row mean `0.000972307 s`;
  - BFP4/LoFi has the sampler boundary repaired by BF16 logits but is rejected on accuracy (`0.976244986` PCC, top-100 `67`);
  - DRAM-sharded attempts are covered: `s1` and `s4` fail with compile/static-CB errors, while `s5` is correct but slower at `0.001330563 s/row`;
  - frozen profiler report hashes in `perf_summary.json` match the generated `reports/.../ops_perf_results_*.csv` files, and the summary shows no generic full-vocab TopK dominating the selected traced greedy sampler.
- The host-weight-cache and PLE contract evidence is coherent:
  - final instrumented decode totals rederive to 41,607 expert misses, 18,873 expert hits, `115,035,033,600` owner H2D bytes, equal peer-zero D2D bytes, 680 PLE rows / 217,600 PLE bytes, 0 source-pack seconds, and 0 expert/PLE completion syncs during measured prepacked decode;
  - cold/warm evidence records exact cold source-pack work and warm reuse with zero source-pack seconds;
  - current `tt/host_weight_cache.py` preserves fixed slots, exact owner H2D, peer-zero writes, PLE row lookup/staging, stale-generation invalidation, and no host expert/PLE projection fallback in the optimized path.
- The selected source mechanisms are present in code:
  - `tt/model.py` has BFP8/HiFi2 selected LM-head policy plus DRAM/BFP4 frontiers, persistent token/position/page-table state, changed-only page-table updates, `tt_out_tok` sampling feedback, traced position advance, split decode/terminal/sampling traces, and explicit host-sampling compatibility separation.
  - `tt/generator.py` uses the optimized token-out traced path with device greedy/top-k/top-p sampling, compact token readback, explicit teacher-forcing/full-logit compatibility paths, and timeline instrumentation.
  - `tt/multichip_decoder.py` and tests retain the inherited dtype/fidelity/KV/activation/CCL/residual policy rather than silently reopening datatype frontier work.
- The 262,144-token context contract is preserved. `doc/context_contract.json` records `current_supported_context` and `full_model_supported_context_tokens` as 262,144 with no capability reduction, and the batch32/context artifact passes with planned bytes `23,393,673,304` and headroom `10,831,847,336`.
- Generated qualitative HF/TT outputs were inspected directly. The shared-suite JSON and `QUALITATIVE_REVIEW.md` correctly frame the 128-token fixed-generation limitation: explanation/coding TT samples remain in reasoning by token 128, but HF controls are also truncated under the same budget; summarization is coherent and reaches a final summary. This is documented as a demo-budget limitation rather than hidden correctness evidence.

## Prior Required Findings Rechecked

- Physical-bound mistake: fixed. The old cache-service-rate lower bound is no longer used as a physical floor; `lower_bound.csv` now separates raw x4+x4 H2D ceiling, measured exact Python/TTNN H2D staging, and observed overlap.
- LM-head DRAM/BFP4 coverage: fixed. The selected policy and rejection ledger are backed by `final_frozen_lm_head_bfp8_hifi2.xml`, `autofix_lm_head_bfp4_lofi_bf16_boundary.xml`, `autofix_lm_head_dram_s1_c40.xml`, `autofix_lm_head_dram_s4_c40.xml`, and `autofix_lm_head_dram_s5_c40.xml`.
- Stale source/profiler evidence: fixed for the selected final artifacts. Per-file source hashes match current files, and profiler summary hashes match the generated profiler report CSVs.
- Teacher-forcing label: fixed. The final docs distinguish warmed token-out, instrumented token-out, and full-logit teacher forcing.
- GDN preservation: acceptable. The final docs no longer present an unsupported GDN/LoFi rejection; they preserve the inherited decoder policy and explicitly leave datatype-frontier work out of this stage.
- Host AutoFix failure evidence: acceptable. Failed Python-level variants are retained as evidence, and the remaining native-API boundary is stated without claiming closure.
- CSV/doc coherence: acceptable. `before_after.csv`, `optimization_matrix.csv`, `lower_bound.csv`, and `host_decode_timeline_summary.csv` parse with consistent row widths. `lm_head_sweep.csv` contains a trailing blank line, but the substantive rows parse and this does not affect claims.

## Other Concerns

- `artifact_manifest.sha256` is stale and fails for `../../tt/host_weight_cache.py`, `../../tt/model.py`, `../context_contract.json`, and `../host_weight_contract.json`. I do not treat this as blocking because the final README/contracts do not point to this manifest for the frozen package, and the selected frozen artifacts have their own current source provenance.
- `final_frozen_profiler/source_provenance.json` records digest `e85ca93...` while listing 8 files, including `tests/test_full_model_perf.py`. The per-file hash for the profiler harness matches current source, and `profiler_provenance.txt` records the harness SHA separately, so the profiler evidence is source-bound enough for this stage; still, future stages should avoid ambiguous aggregate digest semantics.
- Several smaller XML gates (`final_frozen_split_sampling_mixed_contracts.xml`, `final_frozen_sampler_strategy_ab.xml`, `final_frozen_full48_watcher_alloc_tracker.xml`, and `final_frozen_static_contracts.xml`) do not embed a `source_digest` property. This is non-blocking here because adjacent selected JSON/XML artifacts are source-bound to `e85ca93...`, the current file hashes match, and the tests are contract/provenance companions rather than the sole performance/accuracy source of truth.

## Hard-Check Gaps

- None blocking. Per instruction, I did not open TT devices, run hardware tests, start servers, reset devices, run vLLM, or regenerate artifacts. This rereview is based on frozen artifacts, current source hash checks, direct source inspection, and read-only parsing/recomputation.

## Anomaly Ledger

- Anomaly: prior lower-bound evidence used an implementation/cache-service rate as if it were a physical floor.
  - Recheck: `lower_bound.csv`, `host_decode_timeline_summary.csv`, `final_frozen_performance/full_model_performance.json`, and `autofix_pure_completed_dual_owner_h2d_reused_executor.xml`.
  - Resolution: controlled; final docs now separate raw physical H2D floor, measured exact H2D implementation diagnostic, observed overlap, and residual native-API limitation.
- Anomaly: prior LM-head advice/BFP4 boundary was incomplete.
  - Recheck: `LM_HEAD_POLICIES` in `tt/model.py`, LM-head XML frontier artifacts, and `lm_head_sweep.csv`.
  - Resolution: controlled; selected BFP8/HiFi2 is correct and fastest among passing candidates, BFP4 boundary is fixed but rejected on accuracy, and DRAM-sharded `s5` is correct but slower.
- Anomaly: final profiler/source evidence previously looked stale.
  - Recheck: embedded source provenance, profiler `source_provenance.json`, `profiler_provenance.txt`, and report CSV SHA-256 matches.
  - Resolution: controlled for selected artifacts; minor digest-semantics ambiguity documented above.
- Anomaly: qualitative outputs could be mistaken for final-answer quality evidence.
  - Recheck: generated HF/TT output text and `QUALITATIVE_REVIEW.md`.
  - Resolution: controlled; the limitation is documented as fixed 128-token budget behavior, with HF controls showing similar truncation.

## Scope Inspected

- Skills/instructions read and applied: `stage-review`, `multichip`, `optimize`, `host-weight-cache`, `tt-device-usage`, `full-model`, `tt-enable-tracing`, and `qualitative-check`, plus the relevant multi-device and optimization sections of `tech_reports/LLMs/llms.md`.
- Docs/contracts inspected: `README.md`, `work_log.md`, `AUTOFIX.md`, `STAGE_REVIEW_FINAL.md`, `doc/context_contract.json`, and `doc/host_weight_contract.json`.
- Final artifacts inspected: final frozen performance JSONs, accuracy JSON/XML/text outputs, qualitative JSON/text review, cold/warm XML, split sampling/mixed contracts XML, sampler strategy A/B XML, batch32/context XML, full48 watcher/allocator XML, static contracts XML, LM-head frontier XMLs, cache-service AutoFix XMLs, profiler provenance/summary/table/report CSVs, and all requested CSV matrices.
- Source/tests inspected: `tt/generator.py`, `tt/host_weight_cache.py`, `tt/model.py`, `tt/multichip_decoder.py`, `tests/test_full_model.py`, `tests/test_host_weight_cache.py`, and `tests/test_multichip_decoder.py`.
- Read-only checks run: source provenance/per-file hash validation, JUnit pass/fail/skip parsing, performance/timeline metric recomputation, lower-bound arithmetic recomputation, profiler report SHA-256 matching, CSV schema checks, `git status --short`, `git diff --check` for touched Python files, and AST parsing of touched Python files without compiling or writing bytecode.

## Residual Risk

- The main residual performance risk is the measured 30.923% gap from the optimistic physical floor. This is acceptable for this stage because exact Python-level cache/H2D variants were tried and retained, the selected observed path is already faster than the additive exact-H2D diagnostic due to overlap, and the next plausible lever is a native batched TTNN H2D/cache-service primitive outside the current stage.
- Qualitative generation remains fixed at 128 generated tokens; two prompts do not reach a final answer in that budget. This is a documented demo-budget risk, not a hidden failure of the optimized stage.

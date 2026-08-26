# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- QSA long-context traced decode is shape/trace-geometry evidence, not full-cache semantic evidence.
  Evidence:
  `tests/test_functional_decoder.py::test_qsa_traced_decode_at_advertised_context` constructs a layer with `max_seq_len=262144`, a shuffled full-context page table, full RoPE tables, and `current_pos=262143`, then captures and replays `decode_forward`; it does not prefill a nonzero full cache first. The separate `test_full_advertised_context[3]` does run public QSA prefill at `262144` and then eager decode at `262143`.
  Why this is not required work:
  The original prior-review gap was the absence of any advertised-context QSA trace/page/RoPE geometry, and that is now closed by `qsa_max_context_trace_decode.log`. Real-weight traced decode PCC is covered for layer 3 at the smaller correctness context, and full public prefill/cache progression is covered by `long_context_qsa_public_full.log`.

- Performance artifacts are valid latency evidence but not real-weight/router-distribution benchmarks.
  Evidence:
  `tests/test_functional_decoder_perf.py` builds layers with `FunctionalDecoder.from_state_dict(None, ...)` and zero hidden/PLE inputs. The `tt-perf-report` text files warn that `SparseMatmulDeviceOperation` rows lack numeric `nnz`, so DRAM/FLOP utilization for sparse expert rows is omitted unless `--active-experts` is supplied.
  Why this is not required work:
  The stage contract asks for warmed prefill/traced-decode performance artifacts, not optimized or real-weight throughput characterization. The report CSVs are signpost-filtered, contain real device rows, and their `Device Time` sums match the README: layer 0 `23790/5396` us, layer 1 `27269/5922` us, and layer 3 `408094/11081` us for prefill/decode.

- `QSA_AUTODEBUG.md` remains a point-in-time diagnostic and its conclusion still uses residual-risk wording for fixed-512 top-k.
  Evidence:
  The file now opens with a final-status note saying the underfilled fixed-`k` risk was later repaired and closed by `topk_multiset.log`, post-fix PCC, traced replay, and final watcher evidence. Its final paragraph still says that behavior "remains a separate semantic risk that warrants an exact multiplicity test."
  Why this is not required work:
  The authoritative README/work log/AUTOFIX summary and the actual implementation/tests reflect the repair: `_selected_virtual_tokens` masks complete-block lanes with `selected < complete_tokens` and tail lanes with `tail <= current_pos`, while `test_qsa_underfilled_selected_token_multiset` requires exact `{0, ..., p}` multiplicity for positions around block/page/tile/budget boundaries.

## Hard-Check Gaps

- None blocking for this functional-decoder stage.

## Anomaly Ledger

- Observed anomaly:
  The first stage review found that QSA advertised-context public prefill was not evidenced.
  Evidence:
  The preserved `stage_review.md` reported the earlier QSA long-context test only exercised a reduced terminal chunk and left `largest_public_prefill_tokens_tested: 2049`. Current code in `tests/test_functional_decoder.py::test_full_advertised_context` now calls public `prefill_forward` on `[1, 1, 262144, 10240]` for layer 3, and `long_context_qsa_public_full.log` records `test_full_advertised_context[3]` passing in `1072.59s`.
  Affected path:
  QSA public prefill, advertised context support, cache/page-table progression.
  Control or comparison:
  `long_context_linear0_full.log` and `long_context_linear1_full.log` cover the same public full-context test for GDN and GDN+PLE; `context_contract.json` now records layer-3 `largest_public_prefill_tokens_tested: 262144`.
  Likely subsystem:
  QSA long-context coverage / capability contract.
  Investigation performed:
  Compared prior review, current test code, current context contract, README/work-log claims, and final long-context logs.
  Resolution:
  fixed / controlled

- Observed anomaly:
  The first stage review found an underfilled QSA fixed-512 top-k semantic discrepancy.
  Evidence:
  Current `_selected_virtual_tokens` retains fixed-shape `ttnn.topk(k=512)` but separately invalidates filler complete-block lanes with `selected < complete_tokens` and validates the incomplete tail with `tail <= pos`. `topk_multiset.log` records `test_qsa_underfilled_selected_token_multiset` passing, and `topk_fix_real_qsa.log` records post-fix real layer-3 PCC `prefill=0.99679226`, traced decode `0.99988198`.
  Affected path:
  QSA selected-token construction for underfilled contexts.
  Control or comparison:
  The regression checks positions `0, 2, 3, 4, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2046, 2047, 2048, 2049` and requires each valid token from `0` through the query position exactly once.
  Likely subsystem:
  QSA indexer/top-k/tail mask construction.
  Investigation performed:
  Inspected current code, `QSA_AUTODEBUG.md`, `AUTOFIX.md`, `topk_multiset.log`, and post-fix PCC logs.
  Resolution:
  fixed / controlled

- Observed anomaly:
  The first stage review found that the context contract overstated near-max non-aligned prefill execution.
  Evidence:
  Current `tests/test_functional_decoder.py::test_near_max_non_aligned_context` executes public `prefill_forward` at `262143` for all three representative layer kinds. `near_max_gdn_public_final.log`, `near_max_ple_public_final.log`, and `near_max_qsa_public.log` each pass; `context_contract.json` now records `largest_non_aligned_public_prefill_tokens_tested: 262143` for layers 0, 1, and 3.
  Affected path:
  Public padding/chunking and final short-chunk output slicing near supported context.
  Control or comparison:
  The earlier failed PLE attempts in `near_max_linear_public.log` and `near_max_ple_public_retry.log` were caused by a fixed trace reservation/DRAM allocation issue; current `tests/conftest.py` opens the mesh with `trace_region_size=0`, and the final PLE near-max log passes.
  Likely subsystem:
  Public prefill chunking / trace-region reservation.
  Investigation performed:
  Compared old failed logs, current conftest, current tests, final near-max logs, README/work-log claims, and context JSON.
  Resolution:
  fixed / controlled

- Observed anomaly:
  An earlier watcher run found a tiled-index embedding NOC/CB overflow.
  Evidence:
  `watcher_embedding_failure_20260826_1244/pytest.log` reports a 4096-byte NOC read into a 512-byte circular buffer in `embedding_ind_tilized.cpp`; `AUTOTRIAGE.md` identifies the model-scope workaround. Current `_embedding_tiled_output` converts indices to row-major before `ttnn.embedding`, and `static_checks.log` shows only one guarded embedding call site.
  Affected path:
  QSA RoPE/cache/indexer embedding lookups.
  Control or comparison:
  `watcher_qsa_fix_20260826_1251/pytest.log` passes the focused real QSA test, and `watcher_final_20260826_1419/pytest.log` records `34 passed, 7 skipped`. Grepping the final watcher log/kernel names found no fatal watcher exception, NOC/CB/L1 sanitizer error, or `embedding_ind_tilized` entry.
  Likely subsystem:
  TTNN embedding tilized-index kernel selection.
  Investigation performed:
  Read `AUTOTRIAGE.md`, inspected the helper/code path, and searched final watcher artifacts.
  Resolution:
  fixed / controlled

- Observed anomaly:
  Long/aggregate logs contain normal platform warnings and nanobind reference-leak messages.
  Evidence:
  Logs repeatedly warn that a P300 board has one visible chip under the custom p150 mesh descriptor, that the motherboard is unknown, and sometimes that AICLK settled within 5% of target. Several pytest logs also end with nanobind leaked-instance/type/function messages after passing and closing devices.
  Affected path:
  Test/profiler process teardown and platform discovery logging.
  Control or comparison:
  The functional stage is explicitly single-chip 1x1; `ttenv.sh` pins `TT_VISIBLE_DEVICES=0` and the p150 mesh descriptor. The final correctness, long-context, watcher, and profiler logs all pass and close devices; the watcher log does not report core, stack, L1, CB, NOC, stale-input, corruption, or sanitizer faults.
  Likely subsystem:
  Environment/platform logging and Python binding teardown, not decoder semantics.
  Investigation performed:
  Searched final logs and watcher artifacts for failure and hardware-corruption signatures.
  Resolution:
  controlled

## Scope Inspected

- Goal/skill paths:
  `/home/ttuser/dev/qwen3.8-flash-next/tt-metal/.agents/skills/stage-review/SKILL.md`; `/home/ttuser/dev/qwen3.8-flash-next/tt-metal/.agents/skills/functional-decoder/SKILL.md`; `/home/ttuser/dev/qwen3.8-flash-next/tt-metal/.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths:
  `models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/README.md`; `work_log.md`; `../context_contract.json`; `correctness.log`; `watcher_final_20260826_1419/pytest.log`; `watcher_final_20260826_1419/generated/watcher/watcher.log`; `watcher_final_20260826_1419/generated/watcher/kernel_names.txt`; `long_context_linear0_full.log`; `long_context_linear1_full.log`; `long_context_qsa_public_full.log`; `near_max_gdn_public_final.log`; `near_max_ple_public_final.log`; `near_max_qsa_public.log`; `qsa_max_context_trace_decode.log`; `trace_region_zero_validation.log`; `topk_multiset.log`; `topk_fix_real_qsa.log`; `batch32_decode.log`; `two_user_prefill_decode.log`; `perf_host_timing.log`; `static_checks.log`; `AUTODEBUG.md`; `AUTOFIX.md`; `QSA_AUTODEBUG.md`; `AUTOTRIAGE.md`; preserved prior `stage_review.md`; older failed/stale logs `long_context.log`, `long_context_qsa_final.log`, `near_max_linear_public.log`, `near_max_ple_public_retry.log`, `watcher_final/pytest.log`, `watcher_embedding_failure_20260826_1244/pytest.log`, and `watcher_qsa_fix_20260826_1251/pytest.log`; all `tracy/*/*_perf_report.{txt,csv}`, `*_ops.csv`, `*_provenance.log`, and QSA console/stacked report sidecars.
- Code paths:
  `models/autoports/qwen_qwen3_8_flash_next/tt/functional_decoder.py`; `tt/model_config.py`; `tests/test_functional_decoder.py`; `tests/test_functional_decoder_perf.py`; `tests/harness.py`; `tests/conftest.py`; local checkpoint `config.json` and installed Transformers Qwen4Exp config normalization.
- Commands run:
  `sed`/`nl`/`tail` to read skills, docs, code, and logs; `git status --short --branch`; `find`/`wc -l` to enumerate artifacts; `rg` to search logs/code/watcher artifacts for PCC, pass/fail, warning, fallback, trace, NOC/CB/L1/stack/sanitizer, and stale-artifact signatures; small host-only Python scripts to parse local checkpoint config, query installed Transformers config normalization without TT device access, enumerate Tracy signpost rows, and sum `tt-perf-report` CSV `Device Time`. I did not run pytest, Tracy/profiler, watcher, `tt-smi`, hardware reset/list, TT device open/import probes for hardware, servers, or vLLM.

## Residual Risk

- This was an artifact/code review only by instruction; no hardware test, watcher run, profiler run, or server was rerun.
- Full-context correctness uses zero/synthetic structural tensors for the expensive capacity tests, while real checkpoint HF-vs-TTNN PCC is demonstrated at shorter target shapes for each layer kind.
- The PLE n-gram embedding table remains outside the decoder as a caller-prepared tensor because the full BF16 table is recorded as 102400491520 bytes. That boundary is reasonable for the functional decoder layer stage, but downstream full-model work must revalidate the preprocessing and device tensor contract.
- Sparse expert perf reports omit active-expert utilization modeling (`active=?/...`), so the CSVs should be used as latency/provenance evidence, not as final sparse-MoE FLOP/DRAM efficiency claims.

# Stage Review

Verdict: clean-pass

## Findings

- Clean-pass: I found no Required Work for the optimized-vLLM serving stage. The stage evidence supports real vLLM TT-plugin serving with `sample_on_device_mode=all`, `trace_mode=decode_only`, `max_model_len=262144`, non-aligned prompts, model-owned virtual decode slots over physical B1, and vLLM-owned attention KV.
- The headline metrics are correctly single-user 128/128/1 TTFT plus TPOT-derived decode tokens/s/user. The CI 100/100/32 burst is secondary capacity evidence and is not used as the headline decode result.

| Check | Before | After / selected | Review result |
| --- | ---: | ---: | --- |
| Primary 128/128/1 TTFT | 4007.855929 ms | 4484.083915 ms | Same workload/regime; TTFT variability documented. |
| Primary 128/128/1 TPOT | 268.980406 ms | 269.661299 ms | Neutral within repeat variation; same-source repeat A was 267.016417 ms. |
| Primary decode t/s/u | 3.717743 | 3.708356 | Same regime; README uses this as headline, not burst throughput. |
| CI 100/100/32 TPOT | 608.992298 ms | 594.469610 ms | Secondary burst improved by 2.38%. |
| CI aggregate output throughput | 3.059929 t/s | 3.123112 t/s | Secondary burst improved by 2.06%. |
| Optimized full-model token-out | n/a | 231.594000 ms / 4.317901 t/s/u | vLLM after is 269.661299 ms / 3.708356 t/s/u, 16.44% latency overhead; same operating regime. |

- Raw vLLM result JSONs match the normalized benchmark JSONs and README/work_log claims: `before/vllm_result.json`, `after/vllm_result.json`, `before/vllm_ci_serving_result.json`, `after/vllm_ci_serving_result.json`, and the corresponding `*_benchmark.json` files agree on completed requests, token counts, TTFT, TPOT, ITL, and throughput.
- Async decode is real in both plugin wiring and model implementation. The plugin async path calls `decode_forward(..., read_from_device=False)` and then `read_decode_output(..., async_read=True)`. The adapter declares `supports_async_decode=True`; server logs show async scheduling enabled; `after/serving_host_metrics.json` shows primary `trace_replays=126`, `primary_initial_trace_captures=1`, `async_feedback_host_reuses=126`, `async_feedback_device_fallbacks=1`, `model_only_trace_replays=0`, and CI `trace_replays=3167`, `ci_prefill_trace_invalidations=1`, `async_feedback_host_reuses=3136`, `async_feedback_device_fallbacks=32`.
- Trace/persistent-input behavior is supported by source and evidence. `Qwen38FullModel._replay_decode_traces()` executes ingress, terminal, sampling, and position traces with `blocking=False`; token, current-position, page-table, sampling, cache, and PLE inputs are persistent buffers. The tests cover changed/unchanged page-table behavior, token feedback reuse/fallback, stale virtual generation rejection before device execution, non-aligned prompt/page handling, trace invalidation on unsafe prefill program-cache changes, and virtual B2 state isolation.
- On-device sampling is preserved for the measured path. `sample_on_device_mode=all` is in the live server config, plugin sampling tests passed `72 passed, 1 expected skip`, the adapter code contains no sampling/full-logits policy, and measured runtime counters show `host_sampling_compatibility_calls_delta=0`, `model_only_trace_replays=0`, `sampling_seed_host_copies=0`, and no prohibited host sampling/argmax work. The implementation uses the selected full-model device argmax/top-k sampler path and only uses host compatibility for unsupported sampling modes outside the measured windows.
- Host-backed expert/PLE evidence is consistent with the contract. The selected after primary window reports `expert_h2d_bytes == expert_direct_slot_h2d_bytes == 123,572,736,000`, `expert_owner_d2d_bytes=0`, `expert_dma_completion_syncs=0`, `ple_lookup_calls=129`, `ple_selected_rows=4096`, `ple_unique_rows=4096`, `ple_device_h2d_bytes=2,621,440`, and `ple_device_completion_syncs=0`. CI reports the same contract at larger scale: `expert_h2d_bytes == expert_direct_slot_h2d_bytes == 4,162,420,224,000`, `expert_owner_d2d_bytes=0`, `ple_lookup_calls=3201`, `ple_selected_rows=101,904`, and no PLE/expert completion syncs.
- Exactness/lifecycle gates are credible: `after/final_static.xml` totals 64 tests with 0 failures, 0 errors, and 2 explicit hardware-gated skips; `async_feedback_virtual_b2_tt.xml` and `direct_target_completed_cache_tt.xml` pass; `after/host_serving_lifecycle.json` passes active/cancel/follow-up isolation; `after/process_cleanup_audit.json` passes with no matching processes and no holders on `/dev/tenstorrent/0` or `/dev/tenstorrent/1`.
- Qualitative/prompt-format checks are valid for the model type. The raw completions artifact is non-degenerate but not the gating prompt-format artifact. The gated `after/qualitative_tt_chat.json` uses `tokenizer.apply_chat_template(add_generation_prompt=True)`, includes chat template/checkpoint/suite/control hashes, compares against the shared HF control source, and passes. The non-aligned prompt check passes exact logical prompt usage for lengths 1, 63, 64, 65, 67, 127, and 129.
- No prohibited profiler collection was found in the optimized-vLLM evidence. Searches found only documentation stating that Tracy, `tt-perf-report`, live-server profiler, adapter profiler, and `ReadDeviceProfiler` were not collected.

## Required Work

- None.

## Other Concerns

- The selected sampling policy is named `full_vocabulary_argmax` in `doc/datatype_sweep/selected_precision_config.json`, while the stage goal text says “greedy split device sampling.” I did not make this Required Work because the code and runtime evidence show the selected full-model device sampling path, no host greedy/top-1 argmax, no full-logits readback in measured windows, no generic/eager sampler fallback, and zero model-only trace replays. The terminology should be tightened before external-facing release notes.
- The optimized-vLLM evidence tree still contains stale/candidate artifacts that are not selected as final evidence. `after/stage_gate.log` flags reduced 4096-context readiness logs as advisory only, and the context contract plus final server log prove `262144`. Separately, a stale top-level `doc/optimized_full_model/full_model_performance.json` reports older slower numbers, while the README correctly uses `doc/datatype_sweep/post_selection/token_out/full_model_performance.json` / `doc/optimized_full_model/final_frozen_performance_no_timeline/full_model_performance.json` for the 231.594 ms comparison.
- The external plugin root `/home/ttuser/dev/vllm-tt-plugin` is dirty in `src/vllm_tt_plugin/platform.py`. I treated it as read-only context as requested. The dirty file does register Qwen3.8/Qwen4Exp to the autoport adapter and gates async/sample-on-device support correctly, but the stage evidence depends on that local plugin state.
- The server log has a generic custom-scheduler warning that says async scheduling may be disabled for non-AsyncScheduler subclasses. It is contradicted by explicit async-enabled log lines and the model-level async counters, so it is not pass-blocking.
- Server logs contain many TTNN warning lines for deprecated `ttnn.all_gather` arguments with stated removal in September 2026. The current stage passed, but this is a near-term maintenance risk.
- Shutdown logs include nanobind leak diagnostics and `after/process_cleanup_audit.json` has `shutdown_log_has_device_close=false`; the cleanup audit still passes with no leftover server process or device holder and healthy devices.

## Hard-Check Gaps

- None pass-blocking. I did not rerun vLLM, open TT devices, reserve/reset hardware, start servers, or run hardware tests, per the review instructions. The review is based on existing artifacts, raw logs/JSON/XML, and source inspection.
- No live profiler evidence exists by policy; this matches the vLLM profiler exception and the stage goal’s prohibition.

## Anomaly Ledger

- Observed anomaly: Focused layer-0 host-backed/reference PCC failure remains at `0.8637402653694153 >= 0.995`.
  Evidence: `focused_owner_zero_tt.xml`, `autofix_clean_head_control.xml`, `autofix_force_peer_zero_control.xml`, and `autofix_sync_after_indexed_tt.xml` all show the same PCC failure; `AUTODEBUG.md` and `AUTOFIX.md` conclude it exists at starting HEAD `adcfaa2191584bdb6e56c1d2e8b49ecc278a9a51`.
  Affected path: `tests/test_multichip_decoder.py::test_host_backed_layer0_decode_matches_optimized_reference`.
  Control or comparison: Clean detached starting HEAD, forced peer-zero copies, sync-after-indexed experiment, and current optimized path all produce the same PCC; `direct_target_completed_cache_tt.xml` proves all-slot exactness for the optimized direct-slot cache.
  Likely subsystem: Pre-existing reduced focused multichip decode/reference mismatch outside the optimized-vLLM serving delta.
  Investigation performed: Reviewed AutoDebug/AutoFix reports, failing XMLs, passing direct-target/async TT gates, and host cache code.
  Resolution: Accepted as a documented unresolved limitation justified by failed AutoFix; not Required Work for this stage.

- Observed anomaly: Stale reduced-context readiness artifacts mention `max_model_len=4096`.
  Evidence: `after/stage_gate.log` lines 57-60 list advisory context caps in `readiness_vllm/autofix_virtual_b2_reduced/server.log`, `readiness_vllm/autofix_virtual_capacity32_reduced/server.log`, and nested `readiness_vllm/stage_gate.log`.
  Affected path: Stale readiness/AutoFix candidate evidence under `readiness_vllm/`, not the selected optimized-vLLM after artifacts.
  Control or comparison: `after/server.log` non-default args show `max_model_len=262144`; `doc/context_contract.json` reports `current_supported_context=262144`; `after/stage_gate.log` line 61 says context contract OK for target/support `262144`.
  Likely subsystem: Evidence hygiene / stale artifact retention.
  Investigation performed: Compared stage gate advisories to context contract, server config, non-aligned prompt artifact, and README selected commands.
  Resolution: Non-blocking Other Concern; final stage evidence uses the full-context path.

- Observed anomaly: The live server log emits a generic custom-scheduler async warning.
  Evidence: `after/server.log` line 102 warns that degraded performance can occur if a Scheduler subclass is not AsyncScheduler.
  Affected path: vLLM scheduler/plugin log messaging.
  Control or comparison: `after/server.log` lines 19 and 103 show asynchronous scheduling enabled; plugin `async_decode.py` submits `decode_forward(..., read_from_device=False)` and defers readback; derived primary/CI async counters prove host reuse/fallback counts and nonblocking replays.
  Likely subsystem: vLLM generic warning text versus TT plugin scheduler implementation.
  Investigation performed: Inspected plugin platform/async code and `after/serving_host_metrics.json`.
  Resolution: Non-blocking; async decode is exercised and evidenced.

- Observed anomaly: Server shutdown includes nanobind leak diagnostics and no explicit device-close marker.
  Evidence: `after/server.log` contains nanobind leak messages; `after/process_cleanup_audit.json` reports `shutdown_log_has_device_close=false`.
  Affected path: Shutdown diagnostics / process cleanup.
  Control or comparison: `after/process_cleanup_audit.json` verdict is pass, with `matching_processes=[]`, no holders for `/dev/tenstorrent/0` or `/dev/tenstorrent/1`, and healthy P300 device observations.
  Likely subsystem: Python/native object finalization diagnostics during server teardown.
  Investigation performed: Reviewed cleanup audit and server warning classes.
  Resolution: Non-blocking residual risk; no leftover serving process or device holder found.

- Observed anomaly: TTNN all-gather deprecation warnings are numerous and imminent.
  Evidence: `after/server.log` contains repeated warnings that several `ttnn.all_gather` arguments will be removed in September 2026.
  Affected path: On-device sampling/all-gather and related TTNN CCL call sites.
  Control or comparison: The current stage completed and sampling/static/serving gates passed; no runtime failure or profiler use is attached to the warning.
  Likely subsystem: TTNN API migration debt.
  Investigation performed: Classified warning classes in the server log and inspected the on-device sampling path.
  Resolution: Not pass-blocking for this stage, but should be addressed before the TTNN API removal lands.

## Scope Inspected

- Goal/skill paths:
  - `.agents/skills/stage-review/SKILL.md`
  - `.agents/skills/vllm-integration/SKILL.md`
  - `.agents/skills/optimize/SKILL.md`
  - `.agents/skills/host-weight-cache/SKILL.md`
  - `.agents/skills/tt-device-usage/SKILL.md`
  - `.agents/skills/tt-enable-tracing/SKILL.md`
  - `.agents/skills/qualitative-check/SKILL.md`
  - `.agents/skills/autofix/SKILL.md`
- Artifact paths:
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/README.md`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/work_log.md`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/perf_summary.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/before_after.csv`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/before/*.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/after/*.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/after/*.log`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/after/final_static.xml`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/*_tt.xml`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/autofix_*.xml`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/AUTODEBUG.md`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/AUTOFIX.md`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/candidates/direct_slot_async/*`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/candidates/ple_decode_dedup/*`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/context_contract.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/host_weight_contract.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/selected_precision_config.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/post_selection/token_out/full_model_performance.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/final_frozen_performance_no_timeline/full_model_performance.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/qualitative_shared_suite_final.json`
- Code paths:
  - `models/autoports/qwen_qwen3_8_flash_next/tt/generator.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tt/generator_vllm.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tt/host_weight_cache.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tt/model.py`
  - `models/common/modules/sampling/sampling_1d.py`
  - `models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/derive_serving_host_metrics.py`
  - `models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/run_host_serving_lifecycle.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tests/test_generator_vllm.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tests/test_host_weight_cache.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tests/test_readiness_vllm_scripts.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tests/test_vllm_virtual_slots_tt.py`
  - `/home/ttuser/dev/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`
  - `/home/ttuser/dev/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py`
- Commands run:
  - Read-only repository checks: `git rev-parse HEAD`, `git status --short`, `git branch --show-current`, `find .. -name AGENTS.md`.
  - Full skill/doc/code reads with `sed`, `nl`, and `rg`.
  - One-off read-only Python parsers over existing JSON/XML files to re-derive benchmark numbers, metric-window deltas, qualitative metadata, process cleanup, XML pass/fail totals, and AST syntax validity for touched Python files.
  - `find`/`rg` searches for prohibited profiler artifacts/strings.
  - Read-only external plugin checks: `git -C /home/ttuser/dev/vllm-tt-plugin status --short` and `rg`/`nl` over plugin registration/async-dispatch code.
  - No vLLM server, hardware test, TT device reservation, reset, profiler, Tracy, or `tt-perf-report` command was run.

## Residual Risk

- The stage has no pass-blocking gaps, but productionizing this exact local state should preserve the local vLLM TT plugin registration changes or move them into a versioned/reproducible plugin release.
- Primary single-user decode is neutral rather than materially faster; the stage appropriately documents direct-slot cache/PLE candidates and does not overclaim a serving speedup.
- Host expert prepack remains large, about 236-244 seconds in the documented runs, and physical decode remains B1 with virtual B2 multiplexing.
- The TTNN all-gather deprecation warning should be handled soon because the log explicitly names September 2026 removal.
- The AutoFix-justified focused layer-0 PCC limitation is pre-existing but still unresolved; future stages should not silently use that focused failing test as a serving correctness proxy without carrying the same limitation.

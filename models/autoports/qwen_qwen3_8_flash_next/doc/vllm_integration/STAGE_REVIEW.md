# Stage Review
Verdict: clean-pass

## Required Work

- None.

The prior blocking P1 is resolved. The public vLLM capacity contract is now narrowed to the actually proven active B2 surface: `tt/generator_vllm.py` advertises `MAX_NUM_SEQS = 2`, `get_max_tokens_all_users` rejects `max_num_seqs > 2`, and `initialize_vllm_model` rejects `max_batch_size > 2`. The adapter tests now assert the B2 cap, accept 2, reject 3, and initialize the canonical generator with physical traced `max_batch=1` plus `virtual_slot_capacity=2`. The readiness capacity artifact records `advertised_max_num_seqs=2` and `largest_tested_active_max_num_seqs=2`, with larger active widths explicitly unclaimed pending equivalent proof. README/work log/server command claims align to `--max-num-seqs 2`.

## Other Concerns

- Wider active serving capacity remains future work only. This is not a stage blocker because source, tests, readiness artifacts, and docs now reject or withhold public claims above B2.
- The sampling attach command in the work log uses a larger pytest fanout argument while documenting that the live server stayed capped at two active sequences. I did not treat this as a capacity claim because the launch contract, readiness capacity JSON, README, and source cap are all B2.
- Raw qualitative stress artifacts include expected base/chat-model behaviors such as visible `<think>` text, truncation at fixed output budgets, prompt echo, and one learned-Q&A-style drift. The prompt-correct chat-template suite is the gating qualitative evidence and passes 3/3 with HF controls and exact prompt-token accounting.
- The main `readiness_vllm/server.log.gz` contains one conservative active-trace allocation warning. This is documented and controlled by the exact B2 unfiltered tracker evidence: `trace_allocation_tracker_b2_audit.json` and `final_b2_trace_tracker_fixed/server.log.gz` show zero unsafe live-allocation errors, zero active-trace warnings, zero corruption warning blocks, zero runtime errors, and clean lifecycle/cleanup.
- Post-serving cleanup artifacts still show non-semantic shutdown rough edges in surrounding logs, including EngineCore force termination/nanobind leak text and `shutdown_log_has_device_close=false`. The cleanup audit is clean: no matching vLLM/plugin processes, no `/dev/tenstorrent/*` holders, and healthy device status at audit time.
- Cumulative fallback counters in host metrics include earlier compatibility-test history outside the measured benchmark windows. The primary and CI performance windows both show `host_sampling_compatibility_calls_delta=0`, `model_only_trace_replays_delta=0`, and `sampling_seed_host_copies_delta=0`.

## Hard-Check Gaps

- None requiring stage work.
- I did not request a hardware/server rerun for the source cap reduction. The change narrows the public contract to already-proven active B2, as requested.
- I performed local read-only inspection only and did not run vLLM, TT hardware tests, server experiments, reset/recovery flows, or profiling.

## Anomaly Ledger

- Observed anomaly: Previous stage review found source/tests advertised B8 while final evidence validated only active B2.
  Evidence: Current `tt/generator_vllm.py` sets `MAX_NUM_SEQS = 2` and rejects `max_num_seqs`/`max_batch_size` above two; `tests/test_generator_vllm.py` asserts cap 2, accepts 2, rejects 3, and initializes `virtual_slot_capacity=2`; `readiness_vllm/max_num_seqs_limit.json` records advertised 2 and validated active 2; README/work log/server command align to B2.
  Affected path: vLLM public capacity contract and adapter admission control.
  Control or comparison: Final full-model B2 readiness artifacts remain the proof boundary; larger reduced-layer/client-fanout evidence is explicitly non-public.
  Likely subsystem: vLLM adapter capacity declaration, virtual decode-state admission, documentation/evidence alignment.
  Investigation performed: Compared source, tests, readiness JSON, README, work log, and refreshed host/stage-gate logs against the prior P1.
  Resolution: Fixed and non-blocking.

- Observed anomaly: Current primary `server.log.gz` has one conservative active-trace allocation warning.
  Evidence: Gzip log scan found a single warning at line 539 about device-buffer allocation while an active trace may exist.
  Affected path: Trace lifetime safety and decode trace validity.
  Control or comparison: `trace_allocation_tracker_b2_audit.json` reports generic active-trace allocation warnings 0, unsafe live allocation errors 0, corruption warning blocks 0, runtime errors 0, tracebacks 0, stale rejections 0, active/valid slots ending at 0; `final_b2_trace_tracker_fixed/server.log.gz` also has zero active-trace warnings/errors.
  Likely subsystem: TTNN trace capture/replay allocation tracking.
  Investigation performed: Inspected both main and tracker-fixed compressed server logs plus tracker audit JSON and lifecycle counters.
  Resolution: Controlled and non-blocking.

- Observed anomaly: Historical trace-allocation failure mentioned 75 buffers allocated during an active trace.
  Evidence: Historical/autofix notes and tracker audit record this as superseded by the fixed run.
  Affected path: Trace replay stale-state/corruption risk.
  Control or comparison: Exact B2 unfiltered tracker-fixed run shows no unsafe warnings/errors and clean survivor/cancel/followup behavior.
  Likely subsystem: Trace invalidation/recapture around prefill and decode.
  Investigation performed: Compared historical failure notes with `trace_allocation_tracker_b2_audit.json`, fixed server log, and lifecycle/cleanup evidence.
  Resolution: Fixed/controlled.

- Observed anomaly: Raw qualitative stress outputs are sometimes awkward or truncated.
  Evidence: `vllm_qualitative_outputs.json` includes raw-format `<think>` continuations, fixed-budget truncation, prompt echo, and a non-gating learned-Q&A-style response.
  Affected path: Qualitative readiness interpretation.
  Control or comparison: `qualitative_prompt_format.json`, `qualitative_tt_chat.json`, and `QUALITATIVE_REVIEW.md` use tokenizer chat-template prompts through the vLLM completions endpoint, verify exact prompt-token usage, compare with HF controls, and pass 3/3 primary qualitative judgments.
  Likely subsystem: Prompt formatting and base/chat checkpoint behavior rather than vLLM/TT corruption.
  Investigation performed: Reviewed actual generated text, prompt-token accounting, review labels, and HF control outputs.
  Resolution: Controlled and non-blocking.

- Observed anomaly: Non-aligned prompt smoke outputs are short and not semantically meaningful.
  Evidence: `non_aligned_prompt_check.json` covers prompt lengths 1, 63, 64, 65, 67, 127, and 129 with exact prompt usage and deterministic repeated outputs, but some two-token completions are odd strings such as `wen3`/`.Q.Q`.
  Affected path: Non-aligned prefill/decode boundary validation.
  Control or comparison: The artifact is a deterministic prompt-length smoke, not the qualitative gate; quality is covered by the chat-template qualitative suite.
  Likely subsystem: Tiny deterministic continuation budget and model tokenization, not alignment failure.
  Investigation performed: Checked case lengths, usage equality, HTTP status, repeat equality, and separation from qualitative-review evidence.
  Resolution: Controlled and non-blocking.

- Observed anomaly: Server logs include generic vLLM/custom scheduler warnings that can read as though async scheduling is disabled.
  Evidence: Main and fixed server logs contain generic compatibility warnings while README/work log/source/tests declare the async decode path and measured metrics show traced decode replay on the optimized path.
  Affected path: Async decode claim.
  Control or comparison: Plugin `async_decode.py`, `model_runner.py`, adapter capabilities, virtual-slot stale-generation tests, lifecycle audit, and metrics show async trace replay/late-result rejection behavior.
  Likely subsystem: vLLM warning text around custom scheduler class compatibility.
  Investigation performed: Inspected plugin async decode/model runner code, adapter capability declarations, tests, lifecycle audit, and benchmark host metrics.
  Resolution: Controlled and non-blocking.

- Observed anomaly: Cleanup/shutdown logs contain force-termination/leak-style text.
  Evidence: `process_cleanup_audit.json` records `shutdown_log_has_device_close=false`; prior logs contain EngineCore force termination and nanobind leak messages.
  Affected path: Post-serving resource cleanup.
  Control or comparison: Cleanup audit verdict is pass, with no matching processes, no Tenstorrent device holders, and healthy device status.
  Likely subsystem: vLLM process shutdown logging after benchmark/cancel tests.
  Investigation performed: Checked cleanup audit, lifecycle audit, and prior anomaly notes.
  Resolution: Controlled and non-blocking.

- Observed anomaly: Host metrics cumulative counters include nonzero host-compatibility history.
  Evidence: `serving_host_metrics.json` cumulative counters retain earlier compatibility-test activity, while primary and CI measured windows both report zero host sampling compatibility deltas and zero model-only trace replay deltas.
  Affected path: Performance-path purity claim.
  Control or comparison: Primary/CI benchmark-window deltas show optimized serving did not use host sampling/logit fallback; adapter `compute_logits` is not implemented and optimized decode uses token-out sampling feedback.
  Likely subsystem: Metrics windowing/cumulative counter interpretation.
  Investigation performed: Compared cumulative counters with primary/CI start/end deltas and source fallback guards.
  Resolution: Controlled and non-blocking.

## Scope Inspected

- Goal/skill paths: `.agents/skills/stage-review/SKILL.md`, `.agents/skills/vllm-integration/SKILL.md`, `.agents/skills/host-weight-cache/SKILL.md`, `.agents/skills/tt-device-usage/SKILL.md`; original vLLM integration contract and prior P1 remediation instructions.
- Artifact paths: `models/autoports/qwen_qwen3_8_flash_next/doc/vllm_integration/README.md`, `work_log.md`, `STAGE_REVIEW.md` prior contents, `readiness_vllm/max_num_seqs_limit.json`, `capacity_alignment_host.xml`, `host_weight_cache_tests.log`, `host_weight_cache_tests.xml`, `stage_gate.log`, `server.log.gz`, `final_b2_trace_tracker_fixed/server.log.gz`, `trace_allocation_tracker_b2_audit.json`, `vllm_benchmark.json`, `vllm_result.json`, `vllm_ci_serving_benchmark.json`, `vllm_ci_serving_result.json`, `sampling_tests.log`, `plugin_non_tt_tests.xml`, focused TT XMLs, `non_aligned_prompt_check.json`, `qualitative_prompt_format.json`, `qualitative_tt_chat.json`, `vllm_qualitative_outputs.json`, `QUALITATIVE_REVIEW.md`, `serving_host_metrics.json`, `host_serving_lifecycle.json`, `process_cleanup_audit.json`, `doc/context_contract.json`, `doc/host_weight_contract.json`, and `doc/datatype_sweep/selected_precision_config.json`.
- Code paths: autoport `tt/generator_vllm.py`, `tt/generator.py`, `tt/model.py`, `tt/host_weight_cache.py`, `tt/precision_config.py`, `tests/test_generator_vllm.py`, `tests/test_virtual_decode_state_bank.py`; plugin `vllm_tt_plugin/model_executor/platform.py`, `model_runner.py`, `model_input.py`, `async_decode.py`, and relevant plugin sampling/lane tests.
- Commands run: read-only `sed`, `rg`/`rg --files`, `git status --short --branch`, `git -C ... rev-parse HEAD`, JSON/XML/log inspection snippets, and gzip log scans. No server, TT device, reset, hardware test, vLLM run, profiler, or network experiment was run.

## Residual Risk

- The accepted public serving capacity is active B2 only. Higher active widths must remain rejected until they receive the same full-model, final-source evidence ladder.
- Performance numbers are accepted as artifact-backed measurements from prior runs, not reproduced in this rereview.
- Qualitative quality is judged on the prompt-correct chat-template suite plus HF controls. Raw completions remain useful as stress evidence but are not a stronger semantic-quality proof.
- Trace-allocation safety is accepted on the tracker-fixed B2 run. Future changes to trace capture, prefill invalidation, virtual state banking, or allocation behavior should rerun the tracker audit before broadening claims.

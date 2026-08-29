# AutoFix Report: multi-active host sampling compatibility

## Starting Evidence

- Original server evidence: `autofix_multi_active_host_sampling_server_before.log`, lines 4263 and 4309-4318. The EngineCore had 2 running and 8 waiting requests, then raised `ValueError: multi-active virtual decode requires canonical on-device sampling` from `generator.py::_decode_virtual_slots`.
- The failure occurred only after the successful all-device greedy/non-aligned checks, when the full shared sampling profile introduced host-only parameters or logprobs.

## Hypothesis Experiments

- Hypothesis: the plugin produces a per-row mixed token/logits result.
  Experiment: inspect `TTModelRunner.check_perform_device_sampling`, `TTModelInput.perform_device_sampling`, async submission/finalization, and lane extraction; add a two-row mixed-eligibility policy test.
  Result: refuted. Sampling mode is intentionally cohort-global. If any active row needs a host-only feature, the whole cohort receives logits and uses the host sampler.
  Evidence: `tests/test_device_sampling_capabilities.py::test_mixed_eligibility_cohort_uses_one_host_compatibility_mode`.

- Hypothesis: the Qwen physical-B1 virtual adapter implemented host compatibility only for one logical row.
  Experiment: exercise two valid virtual leases with `sampling_params=None` and assert execution/output ABI.
  Result: verified. The prior `rows != 1` guard caused the exact fatal error.
  Fix: preflight every row, serialize each stable slot through the existing model-only trace, synchronously materialize each row's logits before the reused trace output can be overwritten, commit row-local recurrence/position state, and return stacked `[rows, 1, vocab]` logits through the existing async wrapper. Partial execution fail-stops the cohort. The `sampling_params != None` branch is unchanged and remains the canonical traced token-out/device-feedback path.
  Verification: `test_generator_virtual_decode_supports_multi_active_host_compatibility`, `test_generator_multi_host_decode_preflights_later_stale_row`, and the existing device microbatch test.

- Hypothesis: alternating saved sampler modes unnecessarily invalidates a model-only trace.
  Experiment: restore a force-argmax slot while a model-only trace is live, then repeat with a token-out trace.
  Result: verified. Sampler shape matters only to token-out replay; model-only replay never executes the sampling trace.
  Fix: restrict saved-sampler mismatch invalidation to `trace_execution_mode == "token_out"`. Returning to token-out already recaptures a model-only trace through `decode_token_out_traced`.
  Verification: `test_mixed_sampler_slot_activation_keeps_model_only_trace` and `test_mixed_sampler_slot_activation_invalidates_incompatible_trace`.

## Verification

- Main host suites: 34 passed.
  Artifact: `autofix_multi_active_host_sampling_host.xml`.
- Plugin non-TT suite: 167 passed, 8 skipped.
  Artifact: `autofix_multi_active_host_sampling_plugin.xml`.
- No TT hardware was used for this isolated repair, per coordinator instruction.

## Final Status

- Source/host fix: verified.
- Original live full sampling profile: pending coordinator TT rerun. That rerun is still required to prove real host sampler/logprobs responses and absence of the original EngineCore failure.
- Performance-path risk: none found in source or focused tests. Full-logits readback exists only in the explicit cohort-wide host compatibility branch; all-device benchmark cohorts retain the canonical on-device traced sampling path with direct device token feedback.

# AutoFix Report: multi-active host-sampling prefill

## Starting Evidence

- The second live full sampling-profile run failed in its first server-backed
  test, `TestHostOnlyParameters::test_min_p`, after the 11 host-only collection
  and configuration tests passed.  Later failures were connection errors after
  EngineCore exited.
- `autofix_sampling_crash2_server_before.log` records two newly admitted
  prompt-length-3 requests in one cohort, with `min_p` values `0.7` and `0.2`
  and `top_k=20`.  The plugin intentionally selected cohort-wide host sampling,
  so `sampling_params=None` reached the model.  EngineCore then failed at
  `generator.py::_prefill_forward_virtual` with `ValueError: multi-active
  virtual prefill requires canonical on-device sampling` before either row ran.
- Preserved failure artifacts:
  - `autofix_sampling_crash2_server_before.log` (SHA-256
    `343422a143bdee618de79169108323d2a80369e9e8ad2abc9d8dc6ab3b50228d`)
  - `autofix_sampling_crash2_tests_before.log` (SHA-256
    `7fe95fbccddce5be908fa5266d98fc618717b53522d41ff61dc005dfcbe25fbf`)

## Hypothesis Experiments

- Hypothesis: the prior multi-active host-decode repair covered the complete
  host-sampling lifecycle.
  Experiment: inspect the live stack, scheduler dump, prefill/decode branches,
  and the exact first failing sampling test.
  Result: refuted.  Decode accepted multiple host-logits rows, but virtual
  prefill retained both an early and late `rows > 1` host-mode rejection.
  Verdict: verified as the immediate crash cause.

- Hypothesis: the plugin should mix per-row token and logits output modes.
  Experiment: inspect `check_perform_device_sampling`, the prefill submission
  ABI, and the shared host sampler; exercise two rows with an active host-only
  logits processor.
  Result: refuted.  Output mode is intentionally cohort-global.  If one row
  needs `min_p`, logprobs, penalties, or another host-only processor, every row
  in that cohort must return logits to the shared vLLM host sampler.
  Evidence: plugin regression
  `test_two_host_only_prefill_rows_consume_stacked_torch_logits`.

- Hypothesis: retaining both physical-B1 TT logits and reading them after both
  row submissions is safe.
  Experiment: make both fake prefills return the same reusable output object,
  change its logical contents per row, and require the final stacked logits to
  preserve both rows.
  Result: refuted.  Each row must be materialized before the next physical-B1
  prefill can overwrite the shared output storage.
  Evidence: main regression
  `test_generator_virtual_prefill_supports_multi_active_host_compatibility`.

## Fix

- `tt/generator.py::_prefill_forward_virtual` now handles an explicit
  multi-row host-compatibility cohort by serializing every row through the
  canonical full-model physical-B1 prefill, finishing that row's virtual lease,
  immediately materializing and deallocating its TT logits, and returning
  stacked Torch logits shaped `[rows, 1, vocab]` to vLLM's existing host
  sampler.
- All claims, continuation owners, page-table width, and device-sampling
  parameters are checked before row 0 can mutate physical state.  A runtime
  failure after execution begins fail-stops virtual execution until its request
  lifetimes are released.
- The `sampling_params != None` device path still delegates to the canonical
  full-model sampler and direct token-feedback path.  Its two-row call sequence
  and compact-token output are covered unchanged by
  `test_generator_virtual_prefill_reuses_one_physical_state_for_two_slots`.
- `generator_vllm.py` remains a thin delegate.  No sampling policy, argmax,
  top-k fallback, or token-feedback reconstruction was added to the adapter.

## Verification

- Focused pre-fix experiment: new host-prefill regression failed at the exact
  production guard while the existing device-prefill regression passed.
- Main host protocol/state-bank suites:

  ```text
  python_env/bin/python -m pytest -q \
    models/autoports/qwen_qwen3_8_flash_next/tests/test_generator_vllm.py \
    models/autoports/qwen_qwen3_8_flash_next/tests/test_virtual_decode_state_bank.py \
    --junitxml=models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/autofix_sampling_crash2_host.xml
  # 38 passed
  ```

- Plugin non-TT suite:

  ```text
  python_env/bin/python -m pytest -q tests --ignore=tests/tt \
    --junitxml=.../readiness_vllm/autofix_sampling_crash2_plugin.xml
  # 168 passed, 8 skipped
  ```

- Passing artifacts:
  - `autofix_sampling_crash2_host.xml` (SHA-256
    `a4b04396b30f3456c9d23d68587ea45dba2e2e4c82dd54ccd3efcd3c531c219a`)
  - `autofix_sampling_crash2_plugin.xml` (SHA-256
    `27af59b70f94094605457324dc813d74db718f1827ec7455bd3082b11a841672`)
- Scoped `git diff --check` passed in both repositories.  No TT hardware,
  reset, or server process was used for this repair.

## Final Status

- Source and host-side ABI repair: fixed with focused and full non-TT evidence.
- The live full sampling-profile rerun remains a coordinator gate because this
  isolated repair was explicitly prohibited from starting or touching the TT
  server.

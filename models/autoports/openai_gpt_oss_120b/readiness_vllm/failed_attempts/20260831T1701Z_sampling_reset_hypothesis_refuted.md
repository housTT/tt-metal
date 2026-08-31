# Sampling-reset hypothesis refuted

Environment: workspace-local official vLLM, P150x4, full 36-layer model,
`max_model_len=131072`, `max_num_seqs=32`, async scheduling, and
`sample_on_device_mode=all`.

Hypothesis under test: preserve the on-device sampler state across a
removal-only decode-layout reset, while rebuilding it after admission/prefill.

Targeted command: `pytest --count=10` for
`TestBatchIsolation::test_mixed_params_batch` against the live server.

Observed before the terminal session output was lost: repeats 1 through 5
failed, repeat 6 passed, and repeats 7 through 9 failed. Thus the candidate
failed at least 8 of the first 9 completed repetitions and did not repair the
peer-lifetime/presence-penalty instability. The candidate implementation and
its unit test were reverted. This artifact intentionally reports the lower
bound rather than inventing the unavailable final repeat result.

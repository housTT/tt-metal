# Serving plugin changes

Pinned serving pair for the p150 stage 9 run: `tenstorrent/vllm-tt-plugin` checkout at
`/home/hous/dev/ornith-1.5-9b/vllm-tt-plugin`, local HEAD `9bab7d527d4adbece91de7338302c99e652fd642`
(upstream base `f1593cbbf577330dfa839918d9d413c39050e667`, `origin/main`), with upstream vLLM
`0.26.0` installed from the pinned `docs/install-vllm-tt.sh` recipe. The upstream vLLM wheel is
unchanged. One local plugin commit exists on top of the upstream base; its record follows.

## Entry 1: zero stale block-table columns and gate host sampling at admission

1. **Requirement and failure.** (a) A request's page table must describe only that request's
   physical blocks. `InputBatch.block_tables_for_rows` returned the persistent per-slot
   `BlockTable` buffer up to the traced width, so columns past `num_blocks_per_row` still held
   the slot's previous occupant's block ids. The model received another request's physical
   blocks, or a duplicate of its own, and identical temperature-0 requests stopped being
   reproducible (reproducer: `tests/test_determinism_assertions.py` in the plugin checkout; the
   served symptom is recorded in `tt-model.yaml`, "FIXED 2026-09-16: temperature-0
   nondeterminism"). (b) A model that samples only on device cannot serve a request whose
   parameters select the host sampler; the generator raised inside `EngineCore` and killed the
   engine for every other request. The request must be rejected at admission with HTTP 400.
2. **Owner.** (a) Case 1, plugin defect: the plugin's own page-table contract
   (`docs/DECODE_RELOAD_CONTRACT.md`, "Page-table-only refresh ... page tables come from the
   current scheduler allocation") and upstream vLLM block-table semantics give the model only the
   request's allocated blocks; the stale tail violated that. (b) Case 2, missing shared behavior:
   only the plugin's `validate_request` runs before `EngineCore` admission, so a model cannot
   reject the request itself without crashing the engine. Both fixes are model-independent.
3. **Model-side alternative.** (a) The adapter could zero columns beyond `cdiv(prompt_len,
   block)` itself, but on a steady async decode the plugin's `start_pos` is deliberately one step
   stale and no per-row block count is passed, so the adapter cannot know the allocated width;
   masking by stale lengths would hide a real new block. (b) The adapter already raises for host
   sampling without `ORNITH_VLLM_ALLOW_HOST_SAMPLING=1`; raising inside the engine is the
   failure, not a fix.
4. **Existing mechanisms.** `supports_sample_on_device`, `supports_async_decode`,
   `supports_prefix_caching`, `supports_chunked_prefill`, `output_tokens_per_step`, the
   `--additional-config tt.sample_on_device_mode` option and `EXTRA_MODELS_DIR` registration
   were checked; none zeroes stale block-table columns, routes a host-only request to a 400, or
   lets a model declare that its sampled-token logprob or masked chunked-prefill sampling runs on
   device.
5. **Effect on other models.** (a) Every model now receives zeros past each row's allocated
   block count; a model that never read those columns sees no change. (b) Admission rejection
   applies only to a model that declares `supports_sample_on_device=True` and
   `supports_host_sampling=False`; the default `True` keeps every existing model on its previous
   path. `sampled_raw_logprobs_devices` defaults to `()` and
   `supports_masked_prefill_sampling` to `False`, so no existing model takes the new branches.
6. **Authorization.** The change adds shared interface: three `model_capabilities` keys
   (`supports_host_sampling`, `sampled_raw_logprobs_devices`, `supports_masked_prefill_sampling`)
   and a plugin default-behavior change (zeroed stale columns). The commit is authored by the
   operator (Adam Housman, 2026 Oct 23 19:34 UTC) and the operator's stage 9 (p150) task
   statement names "the standalone plugin `tenstorrent/vllm-tt-plugin` checkout at
   `/home/hous/dev/ornith-1.5-9b/vllm-tt-plugin` (local HEAD 9bab7d5, editable install)" as the
   serving pair for this run. This stage did not change the plugin; it records the operator's
   pinned commit. No further plugin change is proposed.
7. **Validation.** Regression tests in the commit: `tests/test_determinism_assertions.py`
   (stale-column leak fails before, passes after), `tests/test_model_runner.py`,
   `tests/test_lane_model_runner.py`. Plugin host suite on 2026 Oct 27:
   `python -m pytest tests -q -x --ignore=tests/tt` in the plugin checkout, 474 passed
   (`/tmp/ornith_p150_plugin_host_tests.log`; copied to
   `doc/vllm_integration/p150/plugin_host_tests.log`). Affected serving checks on p150:
   `readiness_vllm/sampling_tests.log` (host-only parameters, seeds, determinism) and
   `doc/vllm_integration/p150/nonaligned_requests.json`.
8. **Status.** `implemented`, local plugin commit `9bab7d527d4adbece91de7338302c99e652fd642`
   on branch `hous/ornith-1.5-9b`; not pushed by this stage.

## Entry 2 (proposed): `test_frequency_penalty_changes_output` premise

1. **Requirement and failure.** `tests/tt/test_tt_penalties.py::TestFrequencyPenalty::test_frequency_penalty_changes_output`
   asserts that `frequency_penalty=2.0` changes the greedy output of all eight
   fixed prompts ("count*F is unbounded, so every prompt should move"). On the
   p150 profile seven prompts change and `After a while, she` does not.
   Reproducer: `doc/vllm_integration/p150/penalty_margin_probe.py` against any
   p150 server with the host-sampling compatibility mode on; result in
   `doc/vllm_integration/p150/b32_compat_prefinal/penalty_margin_probe.json`.
2. **Owner.** Case 1, plugin test defect: the device output is the penalized
   argmax of the raw logits at every step (the repeated tokens keep 3.25 and
   7.6 nats over the runner-up after the 2.0 and 4.0 nat penalties), the
   sibling tests `test_different_frequency_penalties` and
   `test_frequency_penalty_mixed_batch` pass, and the API rejects
   `frequency_penalty` above 2.0, so the test premise cannot hold for every
   prompt on every precision policy.
3. **Model-side alternative.** None: the penalty is applied exactly; changing
   the model to move this prompt would mean changing its logits.
4. **Existing mechanisms.** No capability key, config option or environment
   variable selects test prompts; the host-sampling compatibility mode does
   not change the penalty math.
5. **Effect on other models.** A margin-aware assertion (every repeated token
   either moves or keeps a raw margin above `count * F`) keeps the test's
   intent for every model; a prompt-set refresh affects only this test.
6. **Authorization.** `pending`: a plugin test change needs the plugin
   maintainers. Not implemented in this stage; the plugin test is unmodified.
7. **Validation.** Planned regression: the margin probe as a pytest case that
   computes the penalized argmax from `logprobs=20` prefill responses for each
   prompt that did not move; the existing penalty tests as the check after
   implementation.
8. **Status.** `proposed`.

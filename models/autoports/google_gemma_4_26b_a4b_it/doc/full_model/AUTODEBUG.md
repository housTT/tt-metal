# AUTODEBUG: Gemma 4 Full-Model Generator Stage-Review Blockers

> Historical diagnosis snapshot. This report records the pre-repair source and
> evidence that drove the AutoFix loop; its line numbers, performance values,
> and unresolved wording are intentionally not current. The repair outcomes and
> final reruns are recorded in `work_log.md` and `README.md`.

Focus path: `models/autoports/google_gemma_4_26b_a4b_it`

This was an inspection-only investigation. I did not edit implementation code and did not try to reproduce hardware-dependent failures. Evidence below comes from the current diff, source inspection, retained artifacts, and cheap host-side/static checks.

## Executive Summary

The stage-review blockers are real. The most direct functional bug is that the public full-model prefill pads prompts before calling the decoder stack, so the layer decoders mistake the physical padded length for the logical prompt length. This bypasses the decoder-level nonaligned-tail cache protection and can overwrite live sliding-window cache rows for prompts such as 1025 tokens.

The trace setup has a separate correctness/lifecycle problem: `_get_or_capture_decode_trace()` warms decode at the requested position, advances the persistent device position tensors, then captures another decode at the next position, restoring only token/current/position inputs afterward. The caller's live KV cache is mutated at both positions during setup. Near the full-attention capacity boundary, the capture-side lookahead can touch position `max_seq_len`, even though `generate()` only admits `logical_len + max_new_tokens - 1 <= slot_capacity`.

Several other blockers are contract or evidence issues rather than proven value corruption: request/activity state changes are not part of the trace key or refresh policy; stochastic sampling reseeds the device RNG with the same seeds on every step; raw caller-owned KV uses an internally allocated wrapper state and therefore cannot preserve trace identity; batched `return_all_logits=True` returns a different type and physical-length local logits; B32 evidence is a shape/capability smoke, not full-model numerical correctness; and the delivered 21.49 ms/token number is a low-level no-readback `decode_forward()` boundary, not high-level `generate()` throughput.

## Verified Headline Findings

### 1. Full-model prefill can corrupt sliding KV for nonaligned logical prompts

`Gemma4Generator.prefill_forward()` rounds each prompt to `_padded_prefill_len()` and sends that physical token tensor and physical `position_ids` to `Gemma4FullModel.prefill_forward()` (`tt/generator.py:75-78`, `tt/generator.py:409-425`, `tt/generator.py:435-449`). The full model embeds the already-padded tensor, calls every layer, and does not forward the original logical length into the layer call (`tt/model.py:534-563`).

The lower-level decoders are written to receive the true logical length. `FunctionalDecoder` and `OptimizedDecoder` derive `logical_seq_len = hidden_states.shape[-2]`, pad internally, and use `_bounded_cache_fill_plan()` for nonaligned tails so physical padding is not written as live cache (`tt/functional_decoder.py:434-447`, `tt/functional_decoder.py:637-656`; `tt/optimized_decoder.py:2158-2182`, `tt/optimized_decoder.py:2870-2911`). `MultichipDecoder` preserves that same contract and bulk-fills only when the perceived `logical_seq_len % 32 == 0` (`tt/multichip_decoder.py:1224-1235`, `tt/multichip_decoder.py:1271-1292`).

Concrete case: logical prompt `S=1025` is padded by the generator to `1056`. Sliding layers use `cache_position_modulo=1024` (`tt/model.py:562`). Because the layer sees `1056`, it takes the tile-aligned bulk-fill path. Padding positions `1025..1055` alias to sliding-cache slots `1..31`; the first decode at position `1025` overwrites only slot `1`, leaving slots `2..31` corrupted inside the live sliding window. This explains why a decoder-only nonaligned test can pass while the full-model public prefill is still unsafe.

Current evidence does not refute this. `long_context_probe_tp*.json` uses `prompt_len=max_seq_len-1`, which leaves only one padded row before a decode that overwrites the corresponding next slot, and the test checks shape/final position rather than cache contents or logits parity.

Repair recommendation: keep logical tensors logical at the model/layer boundary, or pass an explicit `logical_seq_len` through `Gemma4FullModel.prefill_forward()` into every decoder prefill and use that value for cache fill/slicing. Add a full-model nonaligned sliding-cache regression at lengths with dangerous tails, at minimum `1025` and `1055`, where logits after prefill+decode are compared against the decoder/HF reference or against a known-good unpadded path.

### 2. Decode trace capture mutates future live KV and violates capacity bounds

`_get_or_capture_decode_trace()` does an eager warm decode at the caller's `start_pos`, optionally warms the sampler with `tt_out_tok=token_input`, then runs `ttnn.plus_one(current_pos)` and `ttnn.plus_one(position_ids)` before synchronization (`tt/generator.py:591-607`). It then begins trace capture and calls `model.decode_forward()` again using the already-advanced device positions, followed by another `plus_one()` inside the captured trace (`tt/generator.py:609-615`). The restore block only copies host values back into `token_input`, `current_pos`, and `position_ids` (`tt/generator.py:626-631`); it does not restore `state.kv_cache`.

Decode writes KV through `paged_update_cache(... update_idxs_tensor=current_pos ...)` before attention (`tt/functional_decoder.py:910-925`; the optimized/multichip paths use the same current-position update contract). Therefore the trace setup writes the caller's real cache at `p` during warmup and at `p+1` during capture before replay zero ever starts.

`generate()` admits requests with `required_context = logical_len + max_new_tokens - 1` up to slot capacity (`tt/generator.py:676-682`). That check ignores the trace setup's extra decode at `p+1`. For example, a valid public request that needs a final decode at `max_seq_len - 1` can make trace capture touch `max_seq_len`. RoPE caches are built for `torch.arange(max_seq_len)` and `_rope_rows()` embeds position IDs directly (`tt/model.py:323-344`, `tt/model.py:483-506`), so position `max_seq_len` is out of the constructed range; full-attention page tables also only cover valid positions `0..max_seq_len-1`.

The retained long-context artifacts actually show the boundary mismatch: TP1 starts at `50623` with `max_seq_len=50624` and ends at `50625`; TP2/TP4 start at `262143` and end at `262145`. Those artifacts demonstrate that the test advanced beyond the advertised context boundary; they do not prove the lookahead write is safe.

Repair recommendation: capture replay zero at the same logical position the caller will execute. Either restore `current_pos`/`position_ids` to `p` before `begin_trace_capture()` or avoid the pre-capture `plus_one()` entirely. Avoid warming/capturing against production KV if the setup decode can write positions the real request has not reached; if unavoidable, widen admission by the setup lookahead and document it, but that would reduce usable context and still leaves future-KV side effects. Add a near-capacity trace test that exercises the exact first traced decode position and asserts no decode/capture path touches `max_seq_len`.

### 3. Request/activity changes can reuse stale traced row state

The trace key is `(tokens.shape[0], sampling_mode, sampling_spec.key, id(state), page_table_ids)` (`tt/generator.py:552-555`). It omits active mask and absolute positions. On a cache hit, `_get_or_capture_decode_trace()` returns immediately; `decode_forward()` refreshes token/current/position tensors only in host/teacher mode or when `_request_boundary` is true (`tt/generator.py:500-524`).

The vLLM adapter releases traces on `reset_batch`, non-identity `slot_remap`, or first decode readiness (`tt/generator_vllm.py:389-404`). It computes the active rows from `start_pos`, then executes either batch one or the fixed `max_batch_size` lane space (`tt/generator_vllm.py:405-438`). If vLLM changes the active-row set or absolute positions without `reset_batch`/remap, the generator can reuse an old trace whose persistent row-local current positions are still advancing. Newly active rows can start from stale or `-1` state; newly inactive rows can keep participating if their traced state was active.

I could not inspect the external vLLM plugin handoff because `../vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py` is absent locally. The local contract test that tries to read it fails for that reason. The source-level generator/adapter risk is still present.

Repair recommendation: make activity and position refresh an explicit scheduler boundary. Track the previous active mask/logical batch/start positions in the adapter, force `_request_boundary=True` or recapture when they change, and consider including an activity generation in the trace key. Add a host-side/static contract test for active-mask transitions and a hardware test with active rows changing from N to M without page-table/remap changes.

### 4. Stochastic sampling reseeds the RNG with the same per-row seeds every step

`Sampling1D._sample_topk()` calls `ttnn.manual_seed()` on every sampling invocation when a seed tensor is supplied (`models/common/modules/sampling/sampling_1d.py:436-442`). `Gemma4Generator.generate()` passes the same `seeds` argument into every decode step (`tt/generator.py:725-736`), and `_sampling_spec()` folds those seed values into the trace key rather than advancing them per token (`tt/generator.py:250-260`, `tt/generator.py:617-624`).

The vLLM adapter has the same issue. `_sampling_values()` derives seeds from row plus `_unseeded_epoch` (`tt/generator_vllm.py:212-227`), but `_unseeded_epoch` is incremented only at prefill (`tt/generator_vllm.py:310-312`), and decode reuses those values (`tt/generator_vllm.py:429-438`). The common runtime has explicit seed-state machinery that refreshes Sampling1D seed buffers per decode position (`models/common/llm_runtime/decode.py:575-588`, `models/common/modules/sampling/seed_manager_1d.py:308-372`).

This does not affect greedy evidence because `temperature <= 0` forces a greedy `SamplingSpec`. It does affect stochastic generation (`temperature > 0`): repeated manual seeding can replay the same RNG stream for each step/lane rather than producing a deterministic per-token stream.

Repair recommendation: use a caller-owned seed state equivalent to the common `SeedManager` path, hashing explicit request seeds with absolute decode position and advancing unseeded streams once per generated token. Add a stochastic trace test that keeps logits fixed and verifies consecutive decode steps do not reuse the same seed stream unless the absolute position is deliberately repeated.

### 5. Public raw-KV ownership path allocates unused state and breaks trace identity

`Gemma4Generator._state_from_args()` wraps a raw `kv_cache` by first calling `model.allocate_state(max_batch_size=batch_size)` and only then replacing `state.kv_cache = list(kv_cache)` (`tt/generator.py:136-143`). This allocates an unused full KV/page-table state before adopting caller-owned raw K/V. The wrapper state object is fresh for each call, so the trace key's `id(state)` changes and forces recapture even if the caller passes the same raw K/V buffers (`tt/generator.py:552-568`).

That allocation happens before `_get_or_capture_decode_trace()` can release old traces. For raw-KV callers this can allocate new device buffers while an active trace still pins allocator addresses, the exact lifecycle the trace-cache release block is trying to avoid.

The vLLM adapter mostly avoids this specific public raw-KV path: it builds the generator with `create_kv_cache=False`, creates a `FullModelState` around vLLM-owned cache tensors, and rejects prefill/decode calls unless `kv_cache is state.kv_cache` (`tt/generator_vllm.py:139-151`, `tt/generator_vllm.py:159-207`, `tt/generator_vllm.py:299-302`, `tt/generator_vllm.py:379-381`). The public generator API remains unsafe for raw KV.

Repair recommendation: require `FullModelState` for traceable decode, or add a no-allocation `FullModelState` wrapper constructor for raw K/V plus explicit page tables/cache specs. The trace key should include stable K/V tensor identities if mutable `FullModelState.kv_cache` lists can be replaced in place.

### 6. Batched `return_all_logits=True` violates the public contract

The public signature advertises `torch.Tensor | ttnn.Tensor` (`tt/generator.py:359-370`). For a single prompt, `return_all_logits=True` gathers vocab shards to host and slices to the logical length (`tt/generator.py:453-454`). For multiple prompts, the method appends each row's model output to `outputs` and returns the raw Python list when `return_all_logits` is true (`tt/generator.py:403-432`).

Each list item is produced by `Gemma4FullModel.prefill_forward(... return_all_logits=True)`. In that mode the model skips last-token slicing and returns terminal logits over the physical padded sequence length (`tt/model.py:564-569`). So batched all-logits returns per-row TT tensors with local vocab shards and physical lengths, not a consistent logical host tensor/list contract. For mixed lengths such as `[33, 65]`, the caller would see sequence lengths `[64, 96]`.

Repair recommendation: either explicitly reject `return_all_logits=True` for batch > 1 until ragged output is specified, or return a documented structure whose entries are gathered and sliced to each row's logical length. Add a contract test for mixed-length batched all-logits.

## Evidence and Documentation Gaps

### B32 evidence is not full-model correctness evidence

`doc/full_model/artifacts/batch32_probe_tp4.json` records a full 30-layer TP4 B32 run with `verdict: pass`, prompt length 32, output shape, trace counters, and final position. The producing test pre-fills one row and then decodes `probe_batch=32`; it asserts sampled tensor shape, trace replay count, and row-0 current position (`tests/test_full_model_contract.py:172-230`, `tests/test_full_model_contract.py:453-463`). It does not compare B32 logits/tokens against HF or another numerical oracle, and rows 1-31 are not prefill-correctness checked.

The retained `batch32_trace_tracking.log` is not the all-layer B32 run claimed nearby in `work_log.md`. It loads `/tmp/gemma4_full_model_probe_cache`, which corresponds to the reduced `[0, 5]` path, and reports a 5.67 s test call. `work_log.md` claims the all-layer B32 rerun passed in 71.79 s (`doc/full_model/work_log.md:49-68`). Treat the log as reduced split-trace safety evidence only.

Recommendation: add a B32 numerical correctness artifact for the full model, preferably with varied rows and nonaligned prompt lengths. At minimum include row-wise token/logit comparisons for rows beyond row 0 and a retained command/log that matches the claimed all-layer B32 run.

### Delivered performance number is a low-level boundary

`doc/full_model/artifacts/token_out_trace_tp4.json` reports 21.4879 ms/token, zero token readbacks, and zero synchronizations. The test branch that produces it repeatedly calls `generator.decode_forward()` and synchronizes only around the measurement window (`tests/test_full_model_contract.py:465-510`). That is a valid steady-state split traced model+sampling+device-feedback boundary.

It is not the high-level `Gemma4Generator.generate()` boundary. `generate()` synchronizes/reads the first sampled token, then reads a token on the host each subsequent step to append to the returned Python list and check EOS/teacher forcing (`tt/generator.py:693-715`, `tt/generator.py:725-742`). The README partially labels this as `token-out`, but the statement that the `primary full-stack workload is B1, prompt 128, generate 128` overstates what the 21.49 ms/token artifact measures (`doc/full_model/README.md:107-115`).

Recommendation: document the 21.49 ms/token result as the low-level steady-state `decode_forward(read_from_device=False)`/device-feedback path. Keep separate public `generate()` throughput numbers when host-visible tokens are part of the delivered API.

### Provenance is incomplete in this checkout

`doc/full_model/artifacts/provenance.json` hashes for `tt/model.py`, `tt/generator.py`, `tt/optimized_decoder.py`, and `tt/multichip_decoder.py` match the current files I inspected. The recorded `precision_policy_sha256` cannot be verified because the default policy path named by `tt/precision_policy.py` and `tt/generator_vllm.py`, `doc/datatype_sweep/selected_precision_config.json`, is absent in this checkout. `load_precision_policy()` silently returns an empty policy when that file is missing (`tt/precision_policy.py:31-42`).

Recommendation: either commit/retain the selected precision policy file or update provenance to point to the actual policy source used for the full-model artifacts. If the empty-policy fallback is intentional for this stage, state that explicitly and remove the unverifiable policy hash.

## Refuted or Demoted Hypotheses

- Decoder-only nonaligned tail handling is not the direct bug. The decoder implementations have an explicit logical-length tail path; the full-model wrapper bypasses it by pre-padding before the decoder sees the tensor.
- Page-table refresh itself is mostly sound. The standalone generator copies caller-supplied tables into stable tensors, and the vLLM adapter compares host clones and refreshes stable device tensors only when contents change (`tt/generator.py:211-230`; `tt/generator_vllm.py:236-258`, `tt/generator_vllm.py:382-389`). Missing activity/position refresh is the sharper issue.
- vLLM raw-KV ownership does not appear to use the public raw-KV wrapper path. The adapter constructs a `FullModelState` around vLLM-owned tensors and checks object identity. The remaining public raw-KV problem is still real for non-vLLM callers.
- Current-position advancement during normal device replay is intentionally inside the captured trace, not host-stepped after replay. The issue is the trace setup's pre-capture advancement and KV side effects, plus missing refresh on request/activity changes.

## Other Potential Issues

- `Gemma4Generator.reset()` no longer releases cached traces or clears `_trace_cache` (`tt/generator.py:763-773`), whereas the previous diff released traces before replacing/resetting state. `_request_boundary=True` refreshes persistent inputs before next replay, so I do not claim this alone corrupts values, but it leaves active traces pinned across cache fills and request reset. Restoring trace release/clear in `reset()` is the safer lifecycle contract.
- The vLLM adapter's static contract test `test_padded_decode_uses_only_the_contiguous_logical_batch` still expects `execution_batch = logical_batch`, but source now uses `execution_batch = 1 if logical_batch == 1 else self.max_batch_size` (`tests/test_vllm_adapter_contract.py:133-138`, `tt/generator_vllm.py:413-426`). The source comment gives a performance/correctness rationale for the fixed-width path, so this is a stale test or unresolved contract decision rather than a proven runtime bug.
- `Gemma4ForCausalLM.get_kv_cache_spec()` advertises global KV-head counts (`8` sliding, `2` full), while `Gemma4FullModel._make_cache_specs()` expects TP-local heads (`8 // tp`, `max(1, 2 // tp)`) and `allocate_kv_cache_per_layer()` validates against the TP-local shape (`tt/generator_vllm.py:98-114`, `tt/model.py:346-359`, `tt/generator_vllm.py:166-170`). This looks wrong for TP2/TP4 unless the absent external vLLM plugin compensates before calling the adapter.

## Recommended Verification Matrix

1. Host-only/static: add contract tests that prove full-model prefill passes the true logical length into decoder cache fill, raw-KV wrapping does not allocate, trace keys or request-boundary refresh include active-mask transitions, batched `return_all_logits` is either rejected or normalized, and `reset()` releases trace IDs.
2. Hardware, small model/reduced layers: run full-model public prefill+decode with prompt lengths `33`, `1025`, and `1055`, comparing logits or cache-derived output to an unpadded logical reference. Include sliding and full attention layers.
3. Hardware, trace bounds: run first traced decode at `max_seq_len-1` with watcher/allocation checks, and rerun the long-context artifact as either `max-2` plus two replays or `max-1` plus one replay so final position remains in bounds.
4. Hardware, serving state: run vLLM-style decode with active rows changing across steps without page-table/remap changes, plus stochastic sampling with fixed request seeds over multiple positions.
5. Evidence: regenerate B32 full-model correctness artifacts with a numerical oracle and retain command/log/provenance that match the all-layer run.

## Local Checks Performed

- `python_env/bin/pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py -k 'not reduced_real_weight_full_model_probe and not reduced_mixed_prompt_and_inactive_slot_probe'` passed: 9 passed, 4 deselected.
- `python_env/bin/pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_adapter_contract.py` failed in host-only static tests: missing external plugin file at `../vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`, and stale expectation that decode uses `execution_batch = logical_batch`.
- `git diff --check` passed.
- Read-only/static commands included `rg`, `find`, `git diff`, `git status`, `nl`, `sed`, `jq`, and `sha256sum`.

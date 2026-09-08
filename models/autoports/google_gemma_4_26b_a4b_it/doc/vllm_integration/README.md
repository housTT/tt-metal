# Gemma 4 26B-A4B vLLM integration

Status: **ready for text-only vLLM serving on the P150, P150x2, and P150x4
proxy profiles.** Upstream image and video inputs are outside this release.

## Primary single-user results

These are the headline serving numbers. Every row is a warmed, greedy,
single-user `128 input / 128 output / 1 request / concurrency 1` run. Decode
`t/s/u` is `1000 / mean TPOT`; TTFT is reported separately.

| Profile | TTFT P50 / P99 | TPOT mean / P99 | ITL P50 / P99 | Output throughput | Decode t/s/u | Full-model teacher-forcing lower bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P150 (1 chip) | 235.8 / 235.8 ms | 27.7 / 27.7 ms | 26.8 / 27.1 ms | 34.1 tok/s | **36.2** | 34.5 t/s/u |
| P150x2 (2 chips) | 200.3 / 200.3 ms | 22.7 / 22.7 ms | 21.5 / 25.3 ms | 41.5 tok/s | **44.0** | 42.1 t/s/u |
| P150x4 (4 chips) | 194.7 / 194.7 ms | 20.6 / 20.6 ms | 19.1 / 21.6 ms | 45.6 tok/s | **48.6** | 45.0 t/s/u |

The teacher-forcing values are lower bounds from the completed datatype sweep,
not apples-to-apples serving benchmarks. The primary vLLM results leave no
measured avoidable vLLM-specific decode overhead relative to those bounds.

Raw and normalized results are in each profile directory as
`readiness_vllm/<profile>/vllm_result.json` and `vllm_benchmark.json`.

## Secondary CI serving-burst results

The CI-shaped workload is greedy `100 input / 100 output / 32 requests` with
unbounded client concurrency, matching the nightly serving-burst shape.

| Profile | Completed | TTFT P50 / P99 | TPOT mean / P99 | ITL P50 / P99 | Output throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| P150 | 32/32 | 4331.5 / 6494.3 ms | 271.7 / 306.5 ms | 245.3 / 607.1 ms | 100.5 tok/s |
| P150x2 | 32/32 | 3446.4 / 5143.3 ms | 224.8 / 252.3 ms | 199.5 / 628.1 ms | 122.2 tok/s |
| P150x4 | 32/32 | 2989.5 / 4487.9 ms | 215.2 / 239.1 ms | 186.5 / 769.2 ms | 129.5 tok/s |

The corresponding files are `vllm_ci_serving_result.json` and
`vllm_ci_serving_benchmark.json`. The burst profile is secondary capacity/CI
evidence. Its TPOT-derived values are deliberately not presented as headline
decode `t/s/u`, because burst admission and chunked prefill affect TPOT.

## Serving matrix

All measurements used `max-num-seqs=32`, block size 64, async scheduling,
decode tracing, `sample_on_device_mode=all`, and the `gemma4` reasoning and
tool-call parsers.

| Profile | Proxy mesh | Advertised `max_model_len` | TT configuration |
| --- | --- | ---: | --- |
| P150 | 1x1 P300C (`N150`) | 50,624 | `trace_region_size=220000000` |
| P150x2 | 1x2 P300C submesh (`N300`) | 262,144 | `FABRIC_2D`, physical parent 2x2, submesh offset 0x0, trace 220 MB |
| P150x4 | 1x4 P300C (`P300x2`) | 262,144 | `FABRIC_1D_RING`, trace 220 MB |

P150 is intentionally capped at 50,624 by the hard physical DRAM evidence in
`doc/context_contract.json`; P150x2 and P150x4 advertise the full checkpoint
context of 262,144. Serving checks accepted a valid 29-token prompt, proving
that public prompt lengths do not need to align to a page, tile, chunk, or trace
size. The completed context stage separately proves non-aligned boundary
prefills of 50,623 and 262,143 tokens.

Launches used this common command shape:

```bash
TT_GEMMA4_TEXT_VER=google_gemma_4_26b_a4b_it_autoport \
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --hf-model google/gemma-4-26B-A4B-it \
  --mesh-device <N150|N300|P300x2> \
  --max-num-seqs 32 --max-model-len <50624|262144> \
  --output-subdir <P150|P150x2|P150x4> \
  --tt-config '<profile JSON>' \
  --additional-server-args \
    '--async-scheduling --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4'
```

The profile JSON values are:

```text
P150:   {"trace_region_size":220000000}
P150x2: {"trace_region_size":220000000,"fabric_config":"FABRIC_2D","parent_mesh_shape":[2,2],"submesh_offset":[0,0]}
P150x4: {"trace_region_size":220000000,"fabric_config":"FABRIC_1D_RING"}
```

## Adapter, cache, precision, and sampling contracts

`tt/generator_vllm.py` is a thin adapter over `tt/generator.py`. It delegates
prefill, traced decode, terminal logits, and sampling to the full-model
generator. The performance path uses the canonical split traced local-top-32
plus semantic-k1 token output, with the sampled device token fed directly into
the next traced decode. It has no separate sampler, host argmax, generic greedy
fallback, full-logits readback, or Python token readback/writeback feedback loop.

vLLM owns the serving KV cache and page tables. The adapter validates and
passes that cache through without creating a hidden standalone cache. Hybrid
memory sharing keys each allocation by vLLM tensor index and physical block
volume, which preserves legal sharing while separating TP4's physically wider
full-attention blocks.

The adapter consumes
`doc/datatype_sweep/selected_precision_config.json` unchanged:

- BF16 norms, attention QKV/O, activations, residuals, CCL, and KV cache;
- BFP8_B dense and expert weights, FP32 router/routing, BFP8_B LM head;
- HIFI4 sliding attention, HIFI2 full attention, and LOFI MLP/experts;
- layers 5/11/17/23/29 use BFP8_B attention and BF16 dense weights;
- P150 additionally uses BFP4_B expert gate/up on those layers;
- P150x4 uses BFP4_B expert gate/up and down, with BFP8_B packed decode gate/up.

`supports_async_decode=True`; async scheduling and traced decode were enabled in
every served profile. The TP4 mutable-buffer artifact proves stale host tokens
cannot overwrite device feedback, current position increments on device,
changed page tables refresh once into stable addresses, unchanged tables do not
refresh, and inactive rows stay inactive. Live overlap checks additionally
crossed a 64-token page boundary while another request completed and vacated a
slot, with no request contamination or degeneracy.

The final sampling profile is the full shared plugin suite on every profile:
each reported `72 passed, 1 skipped`. The single expected skip is the
all-vocabulary chat-logprobs case. The suite covers seeded and unseeded
sampling, mixed-request parameters, `top_k` through 32, penalties, structured
output, request isolation, and logprobs.
The optional host sampling compatibility path is explicit and is used only for
features the device sampler does not implement: penalties, explicit seeds,
stochastic `top_k > 32`, structured output, or requested logprob modes. It does
not replace the traced on-device path used for qualitative runs and performance.

## OpenAI API and qualitative evidence

Every profile passed OpenAI-compatible chat requests for:

- a 29-token non-aligned prompt;
- automatic `gemma4` tool parsing into `get_weather(city="Paris")`;
- `gemma4` reasoning parsing with special-token preservation;
- reported served `max_model_len` matching the profile contract.

All six prompts and both greedy/sampled completions per profile were read in
full. The outputs are coherent, on-topic, non-repetitive, non-gibberish, free of
wrong-language drift, and free of cross-request contamination. The per-profile
`qualitative_verdict.json` files record this judgment; the raw text is in
`vllm_qualitative_outputs.json`. Long explanatory/story answers may end at the
intentional 256-token evidence cap but remain coherent through truncation.
Greedy requests explicitly use `top_k=1`; stochastic requests explicitly use
`top_k=32`, keeping both on the supported device-sampling path. Automated
degeneracy results are retained in `qualitative_degeneracy_check.json`.

Each profile also passed a fresh `logit_determinism.json`. Its two-token direct
requests follow the exact standalone B1 trajectory: original 32-token prompt,
prefill argmax as decode input, and one decode token. vLLM B1 matches the
profile-local real-weight oracle exactly at both positions. Repeated runs and
duplicate prompt groups in different positions of one concurrent batch are
bit-identical (zero selected-logprob/common-top-10 delta and 10/10 overlap).

The B1 and padded-B32 kernels are intentionally different optimized graphs, so
their distribution comparison is reported separately from same-graph
determinism. Every live profile preserves the exact selected token and text. A
fresh TP4 full-vocabulary control shows identical per-group top-10/logprobs
across all B32 row positions and across homogeneous versus mixed batches, with
all rows selecting the B1 global maximum. Its worst B1/B32 raw/centered logits
cosine is 0.8968/0.9587 and probability total variation is 0.2706. This is
retained as deterministic input-dependent graph-shape sensitivity, not hidden
behind a fitted tolerance; concurrent qualitative, sampling, and overlap gates
remain the model-visible correctness controls.

## Limitations and cleanup

- This release is text-only. Upstream image and video request inputs are
  excluded and were not tested.
- Prefix caching is disabled for this model because the adapter does not claim
  the required cache semantics.
- P150's standalone test harness cannot physically allocate a complete B32
  state with `max_seq_len=2048`. The live paged serving path supports 32
  concurrent requests and passes exact selected-token and duplicate-position
  gates; the standalone full-vocabulary B32 localization is retained on TP4.
- Split model/sampling trace capture uses TT-Metal's
  `corruptible_allocation_scope`. Safety is established by explicit trace
  release/recapture lifetimes, stable-address checks, page-table refresh
  counters, and stale-feedback tests; allocator warning counts are diagnostic
  only and are not used as correctness evidence.
- The Python runtime reports nanobind object/type/function diagnostics at
  interpreter shutdown. Device close still completes and the final audit finds
  no API server or EngineCore process holding hardware.

`readiness_vllm/runtime_cleanup_audit.json` records the final process/device
audit. All four P300C devices were visible after cleanup on firmware 19.13.1.

## Artifact index

Each `readiness_vllm/P150*` directory contains the losslessly compressed
`server.log.gz`,
`openai_feature_checks.json`, `sampling_tests.log`,
`vllm_qualitative_outputs.json`, `qualitative_verdict.json`,
`qualitative_degeneracy_check.json`, `async_overlap_state_test.json`,
`logit_determinism.json`, the primary raw/summary/log triplet, and the CI burst
raw/summary/log triplet. P150x2 additionally retains
`server_full_sampling_20260908.log.xz` because its final sampling suite and
remaining evidence used two clean server lifetimes. P150x4 additionally contains
`mutable_decode_state.json` and the classified pre-model heartbeat failure log.
Each profile retains its fresh determinism server lifetime as
`server_logit_determinism_final_20260908.log.gz`. The profile-local standalone
B1 oracles/JUnit files and the TP4 full-vocabulary B1/B32 localization are under
`readiness_vllm/standalone_baselines/`.

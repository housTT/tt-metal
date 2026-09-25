# Batch-one serving optimization

Date: 2026-09-03

This is the selected four-device TP4+EP4 serving path. It deliberately admits
one request at a time and keeps the physical decode batch at one. All 512
routed experts remain resident in device DRAM; only sparse PLE n-gram row
lookup and upload remain host-backed.

## Changes

- The model card and vLLM adapter now advertise `max_num_seqs=1`. The adapter
  takes the direct physical-B1 path and does not restore or commit virtual-slot
  state between tokens.
- vLLM chunked prefill is enabled for `qwen4_exp` with a 1,024-token scheduler
  budget. `Qwen38BatchState.computed_lens` makes prefill resumable: continuation
  chunks preserve GDN, convolution, PLE history, QSA KV, and compressed-index
  state; only the final chunk computes logits and prepares decode state.
- Internal prefill is now stack-major in fixed 128-token microchunks. Each
  microchunk stays physically padded while it crosses all 48 layers, and is
  trimmed only at the stack boundary. This removes the old per-layer
  slice/pad/concatenate cycle while preserving a `layer_major` A/B control.
- The stable logical-to-physical compressed QSA page map is generated once
  from the host block table and shared by all 12 QSA layers. Decode and B1
  prefill chunks consume that tensor directly instead of rebuilding it with a
  gather and affine subgraph in every layer.
- The vLLM adapter reuses the asynchronous compact token read for both response
  delivery and the next PLE n-gram lookup. It no longer issues a second token
  D2H read per decode step.
- Existing selected fusions remain active: the native single-program recurrent
  GDN update/read, fused prefill causal-convolution plus SiLU, packed GDN
  projections, fused routed-expert dispatch/compute/combine, fused router
  top-k/counting, and device-side sampling/feedback.
- GDN and QSA output gates now apply sigmoid inside the consuming multiply.
  Resident B1 token-out decode is captured as two complete stack graphs around
  the only host boundary (the layer-1 PLE lookup), reducing replay submission
  from roughly 52 trace calls to two without moving any expert work to the CPU.
- The server enables a revision-scoped persistent TTNN cache for the 48 layers
  of resident BFP4 expert tensors. A complete layer loads without reading any
  checkpoint expert tensors; an interrupted/partial layer safely falls back to
  conversion and fills the missing cache files.
- The latency launcher now defaults to a 4,096-token cache/selector capacity,
  matching the direct performance gate. QSA scores every configured compressed
  block before masking future blocks, so constructing the advertised 262,144-
  token capacity for a 128-token workload adds device time to every decode.
  `QWEN38_MAX_MODEL_LEN` retains an explicit long-context override.

## Performance

The direct workload is B1, ISL 128, OSL 128, device-greedy sampling, with 126
warmed trace replays.

| Metric | Previous resident B1 | Optimized B1 | Change |
| --- | ---: | ---: | ---: |
| Prefill | 1.264 s | 1.397 s | +10.5% |
| TTFT | 1.274 s | 1.412 s | +10.8% |
| Decode latency | 99.940 ms/token | 87.536 ms/token | **-12.4%** |
| Decode throughput | 10.006 tok/s/user | 11.424 tok/s/user | **+14.2%** |
| Runtime expert H2D / route D2H | 0 / 0 | 0 / 0 | unchanged |

The next implementation pass retained decode correctness and reduced the same
direct workload's prefill to **1.164 s** and TTFT to **1.174 s**, improvements
of 16.7% and 16.9% respectively over the optimized-B1 values above. Decode was
statistically flat at **87.479 ms/token** and **11.431 tok/s/user**; the
two-segment trace therefore remains a host-submission simplification, not a
claimed device-time speedup.

The prefill/TTFT values are single-run host timings and were not improved by
the decode optimization. At the targeted QSA layer level, extending the cached
page map to prefill reduced its warmed prefill window from 16.029 ms to 14.730
ms (8.1%).

The OpenAI-compatible endpoint used vLLM 0.24.0, max concurrency one, exact
random token lengths, EOS ignored, and a warm request before the measured
128/128 sweep.

| Configured context | ISL | OSL | Requests | Median / p99 TTFT | Mean / p99 TPOT | Decode tok/s/user | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 128 | 128 | 3/3 | **1,265 / 1,271 ms** | **86.84 / 86.87 ms** | **11.516** | **10.412** |
| 262,144 | 128 | 128 | 3/3 | 1,387 / 1,398 ms | 125.69 / 125.73 ms | **7.956** | **7.376** |
| 262,144 | 1,025 | 8 | 1/1 | 4,626 ms | 108.82 ms | diagnostic only | 1.485 |

At the selected 4K capacity, the 128/128 endpoint improves from the 262K
control's 125.69 ms TPOT to 86.84 ms (-30.9%) and from 7.956 to 11.516
tok/s/user (+44.7%). This also closes the apparent endpoint/direct decode gap:
the direct 4K result is 87.48 ms/token. The older 262K result improves over the
prior 181.61 ms TPOT / 5.506 tok/s/user endpoint, but it is no longer the B1
latency default. The 1,025-token request is a correctness/TTFT probe, not a
decode-throughput sample: it forces a 1,024-token first chunk and a one-token
continuation, and completed successfully.

The reduced layer-3 Tracy comparison is summarized in
`qsa_decode_before_after.csv`: device-op time fell from 4.763 ms to 3.738 ms,
and aggregate gather time fell from 1.068 ms to 0.082 ms. The endpoint JSON and
full-model report are retained in `endpoint/` and `full_model/`.

## Run the server

From the `tt-metal` root:

```bash
models/autoports/qwen_qwen3_8_flash_next/serve_batch1.sh
```

The script intentionally uses the vLLM dependency environment only for the
vLLM executable, then prepends this checkout's tt-metal, TTNN, model, and
`vllm-tt-plugin-a48857a` sources to `PYTHONPATH`. This avoids silently importing
the stale TTNN checkout installed in the vLLM environment. Override
`VLLM_BIN`, `VLLM_TT_PLUGIN_SRC`, or `PORT` if needed.

The selected runtime controls are `QWEN38_PREFILL_SCHEDULE=stack_major` and
`QWEN38_DECODE_TRACE_SCHEDULE=resident_stack`. The revision-scoped expert cache
defaults under `~/.cache/ttnn/models/`; set
`QWEN38_EXPERT_WEIGHT_CACHE=off` for a one-off uncached control run. The first
cached startup performs the normal conversion and writes the cache; subsequent
startups load the cached BFP4 tensors directly.

`QWEN38_MAX_MODEL_LEN` defaults to `4096` for this latency-oriented launcher.
Set `QWEN38_MAX_MODEL_LEN=262144` only when the advertised long-context capacity
is required; until the QSA selector is prefix-bounded or tiered, that setting
has a measurable per-token cost even for short requests.

Expected startup policy lines include:

```text
Resolved architecture: Qwen4ExpForConditionalGeneration
Chunked prefill is enabled with max_num_batched_tokens=1024.
Asynchronous scheduling is enabled.
```

## Validation

- Full 48-layer B1 128/128 gate: passed, 126 traced decode replays.
- Reduced real-weight layers 0/1/3: layer-major 1,025-token, stack-major
  1,025-token, and stack-major 1,024+1 continuation all agree with PCC at least
  0.995 and top-100 overlap 98/100.
- Reduced real-weight GDN+PLE+QSA stochastic token-out gate: the two-segment
  stack trace, PLE history, sampling feedback, and trace release all passed.
- Real endpoint 1,025/8 chunk-boundary request: HTTP success.
- Real 4K-capacity endpoint 128/128 sweep: 3/3 success, 86.84 ms mean
  TPOT, 11.516 tok/s/user, and 1,265 ms median TTFT.
- Adapter/model protocol tests: 37 passed.
- Plugin chunked-prefill policy tests: 7 passed.
- LM-head TP4 geometry and fused recurrent lifetime contracts: passed.
- Runtime audit: all 48 expert layers `resident_ep4`; runtime expert H2D,
  expert route D2H, host expert projection, activation round trip, and host
  sampling all zero/false.
- Speculation audit: the checkpoint declares one MTP layer, but speculative
  decode stays disabled until the target path can verify multiple candidates
  and atomically commit or roll back GDN, convolution, QSA, and PLE state. The
  host PLE n-gram table supplies model features and is not itself a safe draft
  token proposer. These facts are emitted in serving runtime telemetry.

## Remaining work

- Prefix-bound or tier the QSA compressed-key selector at long configured
  context, then fuse its remaining query selection, validity mask, and gather.
- Add a target multi-token verifier and transactional recurrent/cache state
  before enabling the checkpoint's MTP head or any separate n-gram proposer.
- Measure warm startup time and exact disk use of the persistent expert cache.

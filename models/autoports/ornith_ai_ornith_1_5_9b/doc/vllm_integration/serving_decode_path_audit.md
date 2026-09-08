# Serving decode path audit

Source and existing-artifact audit, 2026-09-08. No device was opened for this
audit. Supervising serving benchmarks and execution-counter evidence are
recorded below; they use no live-serving profiler.

## Construction, precision, and state ownership

`TTOrnithForCausalLM.initialize_vllm_model` constructs the existing
`OrnithModel` with the native 262144-token context contract. It accepts one
TP4 model, 1–32 fixed slots, and no alternative `optimizations` object.
Unless the explicitly reduced `ORNITH_VLLM_LAYER_INDICES` override is set,
it loads all 32 layers: 24 linear-attention and eight full-attention layers.
The selected precision policy is loaded by `tt/precision.py` from
[`selected_precision_config.json`](../datatype_sweep/selected_precision_config.json),
config ID **`head4_lofi_last8_c32_k4_r2`**. Serving passes no precision or
head overrides.

| Boundary | Current selected policy |
| --- | --- |
| Attention projection, MLP gate/up, MLP down weights | BFP4_B by default |
| Layer 31 projection exception | Attention and both MLP groups BFP8_B |
| Packed full-attention decode QKVG | BFP8_B, including layer 31 |
| LM head | BFP4_B weights, LoFi; 32 cores, 32768 columns per local head slice, K block width 4, two readers |
| Embeddings and norm weights | BF16 |
| Projection compute fidelity | LoFi for attention projections, MLP gate/up/down and LM head; no layer fidelity exception |
| Projection matmul flags | Approximation off, FP32 destination accumulation off, packer L1 accumulation on |
| LM-head matmul flags | Approximation off, FP32 destination accumulation on, packer L1 accumulation on |
| Public activation, residual, logits, sampling values | BF16 |
| Full-attention paged KV cache | BFP8_B |
| CCL payload | Native producer dtype; no BFP8 communication cast selected |
| Recurrent state and precision-sensitive GDN arithmetic | FP32; convolution state BF16 |
| Token feedback / current position / RoPE index | UINT32 / INT32 / UINT32 |

The projection flags are not a blanket description of all math. Recurrent
state matmuls retain HiFi4, FP32 destination accumulation, approximation off,
and packer L1 accumulation off. Optimized **decode SDPA** uses HiFi2,
approximation on, FP32 destination accumulation off, and packer L1
accumulation off; the inherited prefill SDPA config uses HiFi2,
approximation off and FP32 destination accumulation on. GDN coefficients
and gated intermediates include FP32 operations. Mixed BF16/FP32 sharded
residual sums promote before adding and round back to the residual dtype.
These existing exceptions are visible in `functional_decoder.py`,
`optimized_decoder.py`, and `multichip_decoder.py`; the audit changes none.

The adapter validates the scheduler's KV shape, then calls
`model.allocate_cache(B, logical_context, num_blocks=blocks)` exactly once.
The supplied physical block count is shared across fixed slots and is not
multiplied by B or logical context. The returned `ModelCache` contains paged
K/V plus model-owned recurrent/conv buffers. Both the adapter and generator
reject a different cache object. The generator's `owns_cache` is false, so
standalone cache allocation and page-table construction are bypassed.
The scheduler remains authoritative for physical page mapping. Block size
is 64; native page-table width is 4096. The paged-update primitive requires
physical blocks at least this width; the contract probe used 4192 shared
blocks for B3, rather than 3×4096 blocks. This is a physical-pool contract,
not evidence that three native-length requests fit simultaneously.

Warmup primes the canonical sampler with the pinned plugin's neutral parameters
(disabled logprobs use `num_logprobs=-2`), then calls
`ensure_traces(preserve_cache=False)` only for the newly allocated empty serving
cache, avoiding a temporary full-pool snapshot. It exercises an actual
128-token adapter prefill, decode admission and deferred read, then resets
cache, seed/input/history and request-row state while retaining warmed keys
and traces. This fixes the measured cold first-decode compilation/recapture
pause; [AutoFix](AUTOFIX_b1_latency.md) contains before/after controls and the
native startup allocation guard. The final change is confined to startup.
Serving prefill remains eager: the generator's optional prefill trace is
limited to an owned standalone B1 cache. A new prefill program can cause safe
decode recapture before the next replay.

## Steady device decode

1. The plugin `async_decode.submit_decode` passes fixed-slot tokens,
   positions, scheduler page table and the exact cache to the adapter. With
   device sampling enabled it also supplies formatted parameters; histories,
   `reset_batch`, and slot permutations describe scheduler transitions.
2. In an unchanged active batch, the adapter skips sampling reconfiguration
   and input refresh. It calls `generator.decode_forward(None, None, ...,
   read_from_device=False, sample_on_device=True)`. Stale host token and
   position values therefore cannot overwrite ongoing device state.
3. The generator checks cache identity, trace existence, program-cache
   stability and page-table equality. An unchanged table is not uploaded.
   `_replay` queues the model trace followed by the separate sampler trace,
   both on CQ0 with **`blocking=False`**.
4. The model trace consumes persistent tokens, embeds and gathers hidden
   channels, runs the existing TP4 decoder stack, final norm and common
   `LMHead1D`. The selected 32-core head input matches final norm geometry,
   avoiding the alternate head-input reshard. Projection output reductions
   use native two-link ring all-reduce. Embedding and sampler gathers use
   `SamplingCCL` with one-link ring async gather. The model increments current
   positions on device, skipping negative inactive entries, and increments
   RoPE indices on device.
5. The sampler trace calls `SamplingGenerator.sample(enable_trace=False,
   tt_out_tok=persistent_tokens)` inside the outer trace; it does not use a
   second internally managed sampling trace. The common implementation masks
   invalid vocabulary, selects local top-k candidates, gathers candidate
   values/indices, forms global indices, applies its greedy tie handling,
   runs `manual_seed` immediately before `ttnn.sampling`, and writes sampled
   tokens directly into the next model input. The adapter does not select
   `force_argmax`; even greedy serving uses this canonical candidate path.
   The fixed sampler has 32 lanes, max top-k 32, vocabulary 248320 padded to
   262144, and four vocabulary shards of width 65536.
   This is the faster validated canonical path: the inherited
   [32-row greedy comparison](../optimized_full_model/sampler_greedy_comparison.json)
   measured split sampling at **0.572463 ms** versus force-argmax at
   **2.744207 ms**, with both eager and traced choices exactly matching CPU.
   Those are prior isolated sampler measurements, not new vLLM timings.
   No alternate sampling selection was introduced by this integration.
6. When active, the same sampler trace applies penalty state and counts the
   newly sampled tokens afterward. The outer trace advances the persistent
   seed tensor. There is no host RNG write or history reconstruction per
   steady step. Ordinary serving uses the token trace, not the standalone
   output-history collection trace.

The audit exposed a shared combined-penalty order mismatch. After isolated
CPU proof, `tt_penalties.apply_penalties` was minimally reordered to match
the pinned host: **repetition, frequency, presence**. Separate presence,
frequency and repetition controls remain passing. See
[`AUTODEBUG_combined_penalty_order.md`](AUTODEBUG_combined_penalty_order.md)
and [`AUTOFIX_combined_penalty_order.md`](AUTOFIX_combined_penalty_order.md)
for proof and the remaining hardware/full-run status. This was an operation
order correction within the canonical sampler, with unchanged dtypes.
The corrected graph now passes the exact TP4 B32 score/token probe and the
restarted all-layer B32 live host/device combined-penalty control; final
full-suite evidence is tracked in that report.

## Scheduler boundaries and explicit host compatibility

On remap, the generator moves recurrent/conv state, active masks, persistent
tokens/current/RoPE inputs, sampler seeds, and penalty histories on device.
It snapshots source rows to preserve cycles and NaNs, copying results back
to stable addresses. Full-width 2D histories use broadcast `where`, avoiding
the proven oversized-repeat circular buffer. Small masks/index vectors are
uploaded; live feedback is not read back. Host prompt/seed metadata is
reindexed separately. Physical KV pages stay put and the scheduler supplies
the new table.

After remap or admission, the adapter refreshes only new/prefilled/host-owned
lanes. Device `where` merges token and both position inputs, preserving
continuing lanes; inactive current positions become -1 and active-state masks
are refreshed. Changed page tables incur one table upload. Changed sampling
keys can upload parameters/history, warm the sampler while preserving live
state, and recapture traces. Fresh prefill rows may activate penalties without
old histories only at start position zero; ongoing rows require real history.
These are scheduler-boundary costs and are not zero-cost operations.

Host-owned penalized rows also restore caller histories when parameters are
unchanged. Fresh host-prefilled rows carry a pending device-seed flag through
remaps until their first device decode or continuation prefill; the generator
then initializes only those rows using their actual configured request seeds.
Fresh device prefill retains its normal seed/history admission, and continuing
device streams are preserved. These checks occur only inside the existing
boundary branch. Source/CPU and reduced real-adapter evidence are tracked in
[`AUTOFIX_host_mode_transitions.md`](AUTOFIX_host_mode_transitions.md).

Host sampling requires explicit **`ORNITH_VLLM_ALLOW_HOST_SAMPLING=1`**.
The plugin routes unsupported options such as min-p, bad words, logit bias,
allowed-token restrictions, minimum-token processors and structured outputs
to host sampling. TP4 logprobs also require host sampling. Absent that explicit
environment setting, the adapter rejects this route. Compatibility prefill
returns logits without sampling; compatibility decode still replays the
device model but passes `sample_on_device=False`, preserving the canonical
token trace for later device use. All vocabulary shards are then read and
formatted as FP32 host logits `[B, 1, 248320]`. The host sampler owns filtering,
history and the chosen next token, which must be uploaded on its next step.
This mode has different transfer and synchronization costs and must not be
included silently in device-sampling benchmarks.

## Deferred output and measured contract counters

Immediately after submitting a step, the plugin calls
`adapter.read_decode_output(async_read=True)`. The generator queues
`.cpu(blocking=False)` and a CQ0 event **before the next model/sampler replay
can overwrite the persistent output**. Device sampling reads the first
replicated token shard only: a fixed 32-lane UINT32 buffer. Final formatting
returns the first B tokens as host INT64. Explicit host compatibility instead
copies all logits shards. The plugin waits on recorded events in
`finalize_decode`; this wait is outside the generator counters. Therefore a
zero generator `synchronizations` counter does not mean the entire request
pipeline performs no completion waits.

[`adapter_device_tracker_startup_final.json`](adapter_device_tracker_startup_final.json) is a
**reduced real-weight** proof (layers 0 and 3, B3, native logical context),
with native trace-allocation tracking enabled and program-cache allocations
included. It is not a full-model performance result. Its source hashes match
the final adapter, generator and model, including the documented host-mode
history and seed repairs. The final serving manifests independently record
the same runtime source provenance.
The tracker conservatively accounts for live allocations; it does not measure
physical address overlap.

| Two-step case | Model / sampler replays | Token / current / RoPE uploads | Page-table uploads | Deferred reads | Sampling updates |
| --- | --- | --- | --- | --- | --- |
| Deferred with deliberately stale host feedback | 2 / 2 | 0 / 0 / 0 | 0 | 2 | 0 |
| Scheduler changes one mapped page | 2 / 2 | 0 / 0 / 0 | 1 | 2 | 0 |
| Slot permutation `[2,0,1]` | 2 / 2 | 0 / 0 / 0 | 1 | 2 | 1 |

The deferred pair was submitted before either wait, matched synchronous
tokens, and advanced current/RoPE positions exactly twice. Successive token
vectors differed, excluding a constant-output comparison. The changed-page
case left the old page unchanged and wrote the new page on every KV shard.
The remap case preserved the permuted continuation. All cases retained
persistent addresses and reported zero generator global synchronizations.

## Remaining overhead visible in source

No repeated steady token/position/RoPE/parameter upload, full-logit host read,
or blocking trace replay was found in the default device path. The following
are candidates for a future measured optimization pass, not speedup claims:

**Costs specific to vLLM integration or scheduler transitions:**

- The host still compares the whole B×4096 page table each step, and the
  plugin converts sampling tensors to Python lists before the adapter can
  skip unchanged configuration. Scheduler versioning could reduce this CPU
  work if the plugin exposes a reliable change contract.
- Sampling reconfiguration preserves full logits/history and recaptures at
  scheduler boundaries. Serving prefill is eager while eligible standalone
  prefill can trace. These may affect admission and end-to-end latency; they
  are not measured steady-loop regressions.

**Inherited generator/kernel costs, also present outside vLLM:**

- At B>1, every linear-attention layer snapshots recurrent/conv buffers and
  selects inactive rows back after computation, even when all rows are
  active. A separately validated all-active trace or masked state kernel
  could avoid those device copies; the current work protects inactive slots.
  This branch is absent at B1 and cannot explain a B1 serving gap.
- The canonical sampler evaluates 32 lanes even at B1. Its shared gather
  wrapper also clones persistent collective output so consumers can safely
  deallocate their result. Removing either cost would change shape/ownership
  contracts and needs focused correctness and latency evidence.
- Any nonneutral penalty activates the shared full penalty graph, including
  neutral companion penalties. Specialization could reduce elementwise work,
  but would add trace variants and requires combined-penalty parity coverage.

The startup omission was measured and fixed, without changing the steady path.
On the final native-context B1 server, the immediate first128/128/1/concurrency1
greedy request records62.613ms TTFT,11.402ms mean TPOT and87.706tokens/s/user;
the same-workload repeat records50.127ms TTFT and87.898tokens/s/user. Both raw
interval vectors are in [the comparison](b1_startup_benchmark_comparison.json).
The original cold first request was47.970tokens/s/user because of a roughly
1.2s isolated pause; that artifact remains archived. The final first request's
longest interval is19.106ms, and its steady spacing is about11.36ms.

[`full_b1_startup_final_counters.json`](full_b1_startup_final_counters.json)
records255 model/sampler/device/deferred replays: one startup decode plus254
request decodes across the two128-output requests. Host decodes are0. There
are6 token/current/RoPE/page-table refreshes across startup and both admissions,
not per token; no generator global synchronization or history readback occurs.
Event completion waits still exist as described above. Host compatibility is
disabled in this manifest, so the measured path cannot silently return full
logits or fall back to host sampling.

The selected predecessor's87.115tokens/s/user teacher-forcing result (B1,
AIME24 chat100 reference positions/99 decode replays, physical cache context2048)
is only a decoder-cost lower-bound reference. It includes explicit logits
readback and uses a different workload; no serving speedup is claimed from
their small rate difference. These measurements and path controls leave no
observed material avoidable vLLM-specific decode overhead in the measured path.

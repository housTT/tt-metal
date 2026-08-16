# Ornith-1.0-35B — full model (TTNN, 4-chip Blackhole ring)

The whole 40-layer text model, end to end on the mesh: token embeddings, the optimized multichip
decoder stack, the final zero-centered RMSNorm, a column-parallel LM head, and canonical split
sampling that turns vocab-sharded logits into the next token **on device**, inside a captured trace,
with no host argmax anywhere in the measured path.

Two files: [`tt/model.py`](../../tt/model.py) and [`tt/generator.py`](../../tt/generator.py). The
decoder layer is `tt/multichip_decoder.py` exactly as `doc/optimized_multichip_decoder/` shipped it —
not a line changed, and the suite re-asserts its dtype, collective, router and residual contract from
the full model's side.

---

## 1. Result

Warmed, batch 1, the vLLM primary single-user profile (prompt 128 / generate 128), real checkpoint
weights, all 40 layers, `logs/bench_full_model.py` → [`perf_summary.json`](perf_summary.json). The
KV cache for this run is allocated at `cache_context=8192`, which is a *cache allocation* and not the
model's context: allocating the full 262144 changes no decode-step work (the paged kernel reads the
blocks the positions name, not the whole table) and §3 measures that build separately.

| figure | value | what is in it |
|---|---|---|
| **TTFT** | **140 ms** median (min 140, max 167 over 3 runs) | embedding + 40 layers + final norm + LM head + **on-device** sampling of the first token, through the public generator. Host-sensitive: it carries the prompt upload and the first-token readback, and successive full evidence sweeps on this host recorded medians of 130, 132, 134 and 140 ms with a worst run of 167 ms. The decode figures below move by <0.1 % across the same sweeps |
| **token-out decode** | **41.9 t/s/u** — 23.88 ms/token | model trace replay + sampling trace replay + synchronize + the caller's token readback |
| traced logits-only decode | 44.9 t/s/u — 22.26 ms/token | model trace replay alone; the PERF-style figure that is comparable with the decoder stage's per-layer numbers |
| teacher-forcing decode | **38.19 t/s/u** — `models.common.readiness_check.run_teacher_forcing` | the same token-out path **plus** the harness's per-token `next_input` callback and a host token write on the ~6 % of steps where the forced token differs from the sampled one; see §7 for why this is *not* a logits-only number |
| layer-stack lower bound | 46.6 t/s/u — 21.45 ms/token | 30 × 0.564 ms + 10 × 0.453 ms, the decoder stage's own warmed traced-decode latencies |

Every replay is `ttnn.execute_trace(..., blocking=False)`; the synchronize the token-out figure
includes is the generator's own, before the caller's readback.

The full model is **11 % above its own layer-stack lower bound**, and every part of that 2.43 ms is
named: 0.81 ms embedding + final norm + LM head + device-side position advance, 1.17 ms sampling,
0.45 ms synchronize and token readback (§8).

Accuracy against a freshly generated AIME24 chat-template reference (161-token prompt, 100
HF-generated continuation tokens, top-100), both bars cleared:

| gate | top-1 | top-5 | top-100 | bar |
|---|---|---|---|---|
| `run_prefill_check` | 0.950 | **1.000** | **1.000** | top-5 ≥ 0.98, top-100 = 1.00 |
| `run_teacher_forcing` (traced decode) | 0.940 | **1.000** | **1.000** | top-5 ≥ 0.98, top-100 = 1.00 |

Free-running generation is coherent and tracks the HF control closely — §6 quotes both sides. The
advertised 262144-token context is measured, not projected: a **262143-token** non-aligned prompt
prefills through the whole stack and decodes, with 24.15 GiB of DRAM still free (§3).

Three real defects were found and fixed on the way, and two of them destroyed output silently
rather than failing: a **post-trace-capture compilation hazard** that corrupted every request after
the first (§5.1), **40 layers' worth of persistent L1 router buffers** that pushed prefill's SDPA
circular buffers out of L1 (§5.2), and **lazy trace capture wiping the prompt** it was supposed to
decode from (§5.3). Each has a minimal repro or a regression test, and §5.3 is the one the suite
did not catch on its own.

---

## 2. What the model is

```
tokens [b, s]
  -> ttnn.embedding, replicated, bfloat16 TILE DRAM-interleaved            (tt/model.py)
  -> 40 x MultichipDecoder                                    (tt/multichip_decoder.py, unchanged)
  -> ttnn.rms_norm, zero-centered (+1 folded at load), local and exact
  -> LM head: column-parallel [2048, 248320] sharded 4 ways, bfloat8_b/HiFi2
  -> logits [1, 1, 32, 62080] per device  ==  sampler-ready, no gather
  -> models.common.sampling.SamplingGenerator -> next token, written into the decode token buffer
```

`OrnithModel.capability()` is the machine-readable version of that and is asserted by
`tests/test_full_model.py::test_context_contract_is_the_advertised_one`.

### 2.1 What is carried forward from the decoder stage, unchanged

`test_the_decoder_policy_is_carried_through_unchanged` pins this from the full model's side so it
cannot drift silently: every dtype, `CCL_MODE`, `ROUTER_MODE`, `tp`, the page block size, the expert
count and EP factor, the mesh shape, the fabric config and packet size, and the CCL topology and
link count. The one row it does **not** re-assert is "exactly two collectives per layer", which is
the decoder stage's own `test_collectives_per_forward`; the residual layout is asserted separately
by `test_the_inter_layer_residual_is_replicated_on_every_device`.

| item | value |
|---|---|
| mesh | `1x4` Blackhole `p300c` ring, `FABRIC_1D_RING`, `fabric_router_config(8192)` |
| parallelism | TP=4 dense, EP=4 over the 256 routed experts |
| routed experts | bfloat4_b weights, LoFi, bfloat8_b activations |
| dense projections + shared expert | bfloat8_b, HiFi2, packer-L1 accumulate |
| router | bfloat16 weight, HiFi4, float32 accumulate; `ROUTER_MODE="fused_gate"` at decode, `topk` chain at prefill |
| DeltaNet state | float32 recurrent, bfloat16 conv history |
| **paged KV cache** | **bfloat8_b**, 64-token blocks, bfloat16 `paged_update_cache` inputs — the dtype split the decoder stage measured |
| collectives | `CCL_MODE="all_reduce"`, exactly two per layer, both **inside** the layer |
| inter-layer residual | `[b, s, 2048]` bfloat16 TILE **DRAM-interleaved**, **replicated**, bitwise identical on all four devices, **no** collective at the boundary |

The LM head is the only new dense projection, so it takes the dense projection group's dtype
(bfloat8_b / HiFi2) rather than inventing one. The embedding is **replicated** for the same reason
the residual is: a vocabulary- or hidden-sharded embedding would need a collective to produce the
replicated residual the decoder stack contracts for.

`test_the_inter_layer_residual_is_replicated_on_every_device` walks a real prefill layer by layer and
asserts shape, dtype, layout, **memory config** and bitwise cross-device equality at every boundary.

### 2.2 Rejection ledger, inherited and extended

Inherited unchanged from `doc/optimized_multichip_decoder/`: the fused matmul+reduce-scatter
residual family (net worse once the consuming column-parallel projection has to all-gather),
`all_gather_async` as the decode collective (correct but 14 µs/step behind `all_reduce`), persistent
CCL output buffers for the decode collective (the shipped op takes none), BFP4 dense projections
(a tie at decode after TP=4 quartered them), HiFi2 on the router and DeltaNet state matmuls (ties),
an L1 decode residual (tie), and `generalized_moe_gate` at prefill (one token per core).

Added by this stage, each with its measurement in §4 and §5:

* `models/common/modules/sampling/sampling_1d.py` as the sampler — rejected, §4.1;
* a persistent output buffer for the sampler's candidate gather — rejected on an op-contract
  conflict, `tt/model.py::OrnithSamplingCCL`;
* `pad_logits_to_power_of_2` before the local top-k — rejected on measurement, §4.3;
* an ungrouped local top-k — rejected on measurement, §4.3;
* force-argmax greedy (`allow_force_argmax`) — rejected by construction, §4.2.

---

## 3. Context and batch contract

`doc/context_contract.json` is recomputed for the full stack by this stage; the measurement behind
it is `logs/probe_footprint.py` → [`footprint.json`](footprint.json), which reads the allocator's own
DRAM view after each construction stage rather than modelling it.

**No capability is reduced.** The advertised 262144-token context is what the model builds and what
the tests assert.

**Measured** by `logs/probe_footprint.py`, which reads the allocator's own DRAM view after each
construction stage, at the full advertised context and batch 1
([`footprint.json`](footprint.json)):

| stage | bytes per device |
|---|---|
| weights + token embedding + LM head | 6,220,578,816 |
| paged KV cache (4096 blocks = 262144 tokens) + per-batch state | 1,802,170,880 |
| decode + sampling traces and the sampler's tables | 7,687,168 |
| **total resident** | **8,030,436,864 (7.48 GiB)** |
| allocatable DRAM | 33,978,715,136 (31.64 GiB) |
| **free for activations** | **25,948,278,272 (24.17 GiB)** — 3.2x the *measured* resident set (`doc/context_contract.json` reports 4.29, which is allocatable / the *arithmetic* resident set on the row below) |

The arithmetic that predicts it, for cross-checking (it counts long-lived model tensors only, so it
comes out 1.4 % below the measurement — plan against the measurement):

| item | bytes | note |
|---|---|---|
| 40 decoder layers incl. paged KV at 262144 | 6,767,378,472 | the projection in `doc/context_contract.json`'s `multichip_decoder.full_model_projection`, minus the nine duplicate RoPE tables this stage shares |
| token embedding (replicated, bfloat16) | 1,017,118,720 | |
| LM head (column-parallel, bfloat8_b) | 135,086,080 | |
| final norm | 4,096 | |
| total | 7,919,587,368 (7.38 GiB) | vs 8,030,436,864 measured |

Sharing one RoPE table pair across the ten `full_attention` layers saves 608,698,368 B per device;
`doc/context_contract.json`'s `multichip_decoder.full_model_projection` explicitly flagged the
per-layer copies as its own conservatism, and the full model does not have to inherit it.

**Batch.** Batch 1 is the optimized target. The model, generator, cache, page table, positions,
sampling and output formatting are all batch-parameterised: `max_batch_size` up to 32 (the bound
`ttnn.sampling` imposes — it runs one core per user and asserts `1 <= num_users <= 32` — which is
also the decoder stage's advertised decode batch). Tested at 1, 4 and **32**:

* batch 4 — mixed prompt lengths, fixed slots and an **inactive row** (position `-1`, which
  `ttnn.plus_one(..., skip_negative_entries=True)` keeps inactive across replays), plus the
  assertion that slot 0 of a batch-4 model predicts what the batch-1 model predicts;
* batch 32 — `test_batch_32_prefill_and_decode` prefills 32 prompts of 32 different lengths and
  takes one traced decode step, checking every row's position advanced;
  [`footprint_batch32.json`](footprint_batch32.json) is the measured 40-layer build at batch 32:
  8.31 GiB resident, 23.33 GiB free. No batch was found to be infeasible.

At batch 32 `ttnn.conv1d`'s weight preparation refuses every prefill block length and logs caught
`Out of Memory` L1 messages while probing. That is the decoder stage's documented behaviour — its
conv coverage shrinks as the batch grows and `_prepare_conv1d_weights_local` degrades to the FIR
path rather than raising — not a failure of this stage. It is also why prefill runs at batch 1.

What *is* bounded is the product of batch and context, because paged KV is 5,440 B per token per
device across the ten `full_attention` layers. That is an allocation choice the caller makes through
`cache_context`, exactly as `--num-gpu-blocks` is in vLLM, not a reduction of the model's advertised
context: at batch 1 the cache for the full 262144 tokens fits with 24.17 GiB to spare, and at batch
32 the ceiling is **152,113 tokens per user** — `(33,978,715,136 allocatable − 6,220,578,816 weights
− 8,072,192 traces/sampler − 1,270,209,024 per-batch state) / 5,440 / 32`, with *nothing* left for
activations, so a real deployment sits below it. Both numbers come from measured builds
(`footprint.json`, `footprint_batch32.json`) and the arithmetic is written out in
`doc/context_contract.json`'s `batch32_kv_ceiling_arithmetic`. `logs/probe_footprint.py` prints the
measured numbers for any `(batch, cache_context)` pair.

**Prompt length is a logical input.** `prefill_forward` accepts any length in
`[1, supported_context]`. The generator owns the chunking (2048-token internal blocks), the physical
padding (128-token alignment), the tail masking, the paged cache fill, the position bookkeeping and
the output slicing. `test_prefill_accepts_any_logical_prompt_length` runs 1, 7, 31, 33, 63, 129, 250,
1000, 2049 and 3000 — straddling the tile, the 64-token page, the 128-token alignment and the
2048-token chunk — and `test_full_stack_non_aligned_long_prompt` runs 5003 through the complete
stack.

**Up to and including one token short of the advertised context.** `logs/probe_long_prompt.py` walks
non-aligned lengths through the same public path on the full stack with the full 262144-token cache
allocated, prefilling each and then taking one traced token-out step
([`long_prompt.json`](long_prompt.json)):

| prompt | 5003 | 8191 | 16381 | 32749 | 65521 | 131071 | **262143** |
|---|---|---|---|---|---|---|---|
| prefill | 2.98 s | 3.44 s | 6.97 s | 14.30 s | 30.24 s | 67.83 s | **163.82 s** |
| tokens/s | 1680.5 | 2378.7 | 2350.0 | 2290.6 | 2166.8 | 1932.4 | **1600.2** |
| DRAM free after | 24.16 GiB | 24.16 | 24.16 | 24.16 | 24.15 | 24.15 | **24.15 GiB** |

Every row returns finite logits and a valid sampled token id. Nothing was refused and nothing ran
out of memory: the advertised context is not a projection here, it is a length this path has
actually run.

---

## 4. Sampling

### 4.1 Which common sampler, and why

Both were read against this model before anything was written, as `$full-model` requires.

| contract item | `models/common/sampling` (`SamplingGenerator` + `TTSampling`) | `models/common/modules/sampling/sampling_1d.py` (`Sampling1D`) |
|---|---|---|
| trace ownership | **owns capture/replay**, keyed by (penalties, logprobs, force-argmax) | none — the caller must write the trace wrapper *and* key it |
| `tt_out_tok` | threaded through `capture_trace`/`sample`, and `_validate_trace_inputs` refuses a replay whose output tensor is not the captured one | accepted as a forward arg only |
| greedy tie-break | `_adjust_values_for_tiebreak` boosts the lowest **global index** among tied maxima, so `k=1` means what `torch.argmax` means | none; `ttnn.topk(stable=...)` is best-effort (tenstorrent/tt-metal#33492) |
| penalties | `TTPenalties`, presence/frequency/repetition | not present |
| seeds | `SeedManager`, per-request seeds, trace-aware | a per-call tensor |
| logprobs | `LogProbsCalculator`, top-k and sampled-token forms | sampled-token only, and gated to 8/32-device meshes |
| mesh/topology | 1D and 2D, `cluster_shape`-driven | 1D only (fits, but no headroom) |
| padding/layout | `vocab_padding` masks, power-of-2 padding knob | same helpers |

**Selected: `models/common/sampling`.** Two of those rows decide it. The greedy tie-break is a
correctness property for this stage — the readiness gates compare against `torch.topk` of an HF
reference, and a tie broken by array position instead of index is a wrong token, not a slower one.
And trace ownership is exactly the thing `$tt-enable-tracing` warns not to hand-roll: `SamplingGenerator`
already keys traces by sampling mode, which is the documented cause of "greedy output nondeterministic
after a sampled request".

**Rejected: `Sampling1D`.** It is the cleaner module — stateless, declarative, no mutable sampling
state — and if this model only ever ran greedy on a mesh where ties cannot happen it would be the
better choice. It is rejected because it has no tie-break, no trace ownership and no penalties, so
the generator would have to re-implement all three, and because the vLLM stage will need seeds,
penalties and logprobs that `SamplingGenerator` already has. Both samplers hit the same `ttnn.topk`
cost in §4.3 and both need the CCL shim in §4.4, so neither of those is a discriminator.

No custom sampler code was written. `TTSampling` gained one **opt-in, default-off** capability
(§4.3); everything else is the shared implementation as shipped.

### 4.2 Greedy is semantically greedy split sampling

Greedy is requested as `SamplingParams(temperature=0.0, top_k=1, top_p=1.0)`, which
`format_sampling_params` turns into the device's compact greedy representation (`temp=1, k=1, p=0`)
for every row. It then runs **the same graph every other sampling mode runs**: local top-32 per
vocabulary shard, gather the 4 × 32 candidates, `ttnn.sampling` with `k=1`. There is no separate
greedy-only path, and `allow_force_argmax` is deliberately **not** enabled — force-argmax would
all-gather the full 248320-wide logits and run a global `ttnn.argmax`, which is precisely the
"sampler op dominates token-out decode" shape the goal forbids. Top-k/top-p sampling is the same
captured graph with different `k`/`p`/`temp` tensors.

### 4.3 `ttnn.topk` is linear in the reduced width, and the vocabulary shard is 62080 wide

This is where the stage's one shared-code change comes from. `logs/probe_topk.txt` measures
`ttnn.topk(k=32)` on the real per-device logits shape:

| reduced width | 1940 | 3104 | 3880 | 7760 | 15520 | 31040 | 62080 |
|---|---|---|---|---|---|---|---|
| device time (`stable=False`) | 0.33 ms | 0.50 ms | 0.64 ms | 1.25 ms | 2.47 ms | 4.94 ms | **9.86 ms** |

(The group-count table below is `stable=True`, which is what ships — it costs 18 % more at width
62080, 11.62 against 9.86 ms, and both figures are in `logs/probe_topk.txt`. The ladder above is
`stable=False` so the width relationship is read without that constant factor on top.)

Exactly linear in the width, and **independent of every other dimension**: 32, 64 and 128 rows of
width 7760 all cost 1.25 ms, and `[1, 32, 32, 1940]` costs the same 0.33 ms as `[1, 4, 32, 1940]`
despite having 8x the elements. Core count does not move it either — 32, 64 and 110 cores are all
9.86 ms at width 62080, and at 4 and 8 cores the op refuses the split outright
(`topk_utils.cpp:93: split_size != 0`).

At 248320 tokens over four devices that is a 62080-wide shard per device, and the sampler cost
**11.8 ms of a 13.7 ms token-out step** — 86 % of a decode step on the reduced probe, and it would
have been ~33 % of the 40-layer step. Two knobs were tried and rejected before the fix:

* `pad_logits_to_power_of_2` (62080 → 65536): **worse**, 10.87 ms vs 9.86 ms. The pad itself is free
  (0.03 ms); the wider reduction is not;
* `stable=False`: 9.86 ms against 11.62 ms, but `stable=True` is what upstream asks for and the
  greedy tie-break depends on the ordering, so it is not taken;
* a *shallow* split — 2 or 4 groups — barely helps (5.96 ms and 3.08 ms), because the stage-1 width
  still dominates. The win needs the balance point.

The fix follows directly from "independent of every other dimension": present the shard as `groups`
rows of `width/groups` columns. `TTSampling` gained `args.topk_num_groups` (default **1** — every
existing caller is byte-for-byte unchanged), which replaces one width-62080 reduction with a
width-3104 one plus a width-640 one over the group winners, and recovers the shard-local vocabulary
index with one `ttnn.gather`. It is **exact**, not an approximation: each group contributes its own
top-`max_top_k`, so no member of the shard's true top-32 can be dropped, and
`test_grouped_local_topk_matches_a_single_reduction` asserts identical indices and values against
`torch.topk` on well-separated maxima.

`groups=20` is the measured optimum among the divisors of 1940 (= 62080/32) that keep every group
width tile-aligned — a non-aligned group width would let the reduction read tile padding:

| groups | group width | stage-2 width | measured | exact vs `torch.topk` |
|---|---|---|---|---|
| 1 (upstream) | 62080 | — | 11.62 ms | yes |
| 2 | 31040 | 64 | 5.96 ms | yes |
| 4 | 15520 | 128 | 3.08 ms | yes |
| 5 | 12416 | 160 | 2.50 ms | yes |
| 10 | 6208 | 320 | 1.38 ms | yes |
| **20** | **3104** | **640** | **0.96 ms** | **yes** |
| 97 | 640 | 3104 | 2.29 ms | yes |
| 194 | 320 | 6208 | 4.32 ms | yes |

Every row is checked for exactness in the same run, against `torch.topk` on well-separated maxima —
`logs/probe_topk.txt` is the committed output.

**What "exact" means here.** For distinct values the grouped reduction returns the same k values and
the same indices as the single reduction; that is what the probe and
`test_grouped_local_topk_matches_a_single_reduction` check, on deliberately well-separated maxima
(on random bfloat16 the comparison is dominated by tie-breaking and tests nothing). *Ties* are the
one place the two spellings can differ: `TTSampling._adjust_values_for_tiebreak` only sees the
gathered stage-2 winners, and its own docstring records that a shard holding more than `max_top_k`
maxima tied at one value may not surface the lowest global id. Grouping applies that same bound per
3104-wide group instead of once per 62080-wide shard. It costs nothing in the greedy evidence here —
both probe arms emit byte-identical tokens on a real prompt, and the readiness gates agree with HF
at top-5 1.000 — but a caller that needs a guaranteed lowest-id tie-break across a fully tied
vocabulary should know the bound is per group.

End to end, on the reduced probe (`logs/probe_terminal_single.txt` vs
`logs/probe_terminal_grouped.txt`, line for line):

| row | single reduction | grouped, 20 |
|---|---|---|
| `sampling trace replay` | 11.820 ms | **1.181 ms** |
| `sampler (eager)` | 11.815 ms | 1.166 ms |
| `token-out step (replay + sample + sync + readback)` | 13.702 ms | **2.846 ms** |

On the delivered 40-layer model the sampling stage costs **1.17 ms** (`perf_summary.json`'s
`full_model_only_cost.sampling_ms`) — a different measurement from the probe's 1.181, and 4.9 % of
the step rather than a third of it. The greedy tokens are unchanged: both probe arms print the
16 tokens they generate from the same fixed prompt and the two lists are identical, and
`test_grouped_local_topk_matches_a_single_reduction` makes the same comparison at the op level
against `torch.topk`.

### 4.4 The sampler's gather does not use the deprecated `ttnn.all_gather`

`TTSampling._perform_all_gather` prefers `tt_ccl.line_all_gather` and otherwise falls back to
`ttnn.all_gather` — the deprecated op that `doc/optimized_multichip_decoder/` §4.1 measured
producing a different result on one device from the others in of order 1 % of sustained traced-replay
rounds, and removed from the decoder for exactly that reason. Letting the sampler reintroduce it, on
the tensor that decides the emitted token, would have undone that finding silently.

`OrnithSamplingCCL` (in `tt/model.py`) subclasses the shared `TT_CCL` and provides
`line_all_gather` built on `ttnn.experimental.all_gather_async` with the two gather semaphores the op
asserts plus a barrier semaphore — the other 0-of-600 arm in that same table.
`test_the_deprecated_all_gather_is_not_reachable_from_the_sampler` asserts `TTSampling` actually
picked the shim up and that the shim's counter advances during a real generation.

Semaphore **cycling** comes from the base class and is not optional: one sampling call issues two
gathers back to back, and an earlier version of the shim that returned one fixed handle set for both
deadlocked the mesh under a 40-layer decode capture.

A persistent output buffer for the gather (OPT-009 measured persistence as worth ~11 % inside the
async family) was tried and **rejected on an op-contract conflict, not on latency**: `TTSampling`
ends every call with `ttnn.deallocate(topk_values_gathered_bf16_interleaved)`, and that tensor *is*
the gather's output buffer, so a caller-supplied persistent buffer is freed by the first sampling
call and the second dies on `input_tensor.is_allocated()`.

---

## 5. Three defects, two of them silent

### 5.1 Programs compiled after trace capture are overwritten by the first replay

**Symptom.** The first request in a process was perfect; the second emitted token 0 (`!`) followed by
gibberish, wrong-language runs and single-token collapse — and from then on *every* request was
broken, including a repeat of the first. A bare prefill of the first prompt, with no decode at all,
returned token 0. Deterministic: two runs produced byte-identical garbage.

**Mechanism.** tt-metal hands the trace's intermediates back to the allocator at
`end_trace_capture` while the captured commands still write to those addresses, and warns about it:
*"Allocating device buffers is unsafe due to the existence of an active trace. These buffers may be
corrupted once a trace is executed"* (`tt_metal/impl/allocator/allocator.cpp`). A **program's kernel
binaries are such a buffer** — and unlike an activation they live in the program cache for the rest
of the process. So: request 1 captures the decode traces; its prefill compiles the programs for its
prompt length (the `ttnn.slice` offsets, the MoE valid-token count and the conv1d length are all
keyed by logical prompt length); the first trace replay overwrites their binaries; every later
prefill that reuses them executes corrupt code.

**Repro.** `logs/probe_bisect.py` isolates it in one binary, printing
`mesh.num_program_cache_entries()` at every step:

```
--order after                                   --order before
  capture traces                                  prefill A          top1=90700  programs 91->218
  prefill A  top1=90700  programs 268->375        prefill B          top1= 8160  programs 218->225
  prefill B  top1= 8160  programs 375->382        capture traces                 programs   ->382
  replay                                          replay
  prefill A  top1=    0  <-- corrupted            prefill A          top1=90700
  prefill B  top1=    0  <-- corrupted            prefill B          top1= 8160
  RESULT A stable=False B stable=False            RESULT A stable=True B stable=True
```

Same programs, same count (382), same replays; only the order differs.

**Fix.** `OrnithGenerator._ensure_traces_replay_safe` records the program-cache size at capture and
re-captures the traces if anything has been compiled since — before the first replay can run. Capture
records without executing, so a re-capture between a prefill and its decode loop preserves the KV
cache, the DeltaNet state and the positions. The check is one integer comparison; a re-capture costs
a few hundred milliseconds and happens once per newly seen prompt length and never again
(`test_traces_are_recaptured_when_a_new_program_is_compiled`). The same guard covers a caller-owned
KV cache attached after capture, which invalidates the traces for the same reason.

`test_requests_of_different_prompt_lengths_do_not_corrupt_each_other` is the regression test:
short → long → short → long, all four reproducing, plus a bare prefill that must still agree.

### 5.2 Forty layers of persistent L1 router buffers vs prefill's SDPA circular buffers

**Symptom.** A 384-token prefill on the 40-layer stack died at
`chunked_scaled_dot_product_attention` with *"Statically allocated circular buffers in program 502
clash with L1 buffers on core range [0-0 - 10-9]. L1 buffer allocated at 1196032 and static circular
buffer region ends at 1430912"*. The two-layer probe never saw it.

**Mechanism.** `MultichipMoE.prepare_decode_gate` allocates five **persistent L1** tensors per layer
for the fused router gate — a zero bias, the expert-id table, two preallocated gate outputs (all
HEIGHT_SHARDED, one 32×32 shard per core) and a ROW_MAJOR scatter base. Their contents depend only on
the expert count, the top-k and the mesh, so forty layers hold forty identical copies: ~8 KiB of L1
per core per layer, ~320 KiB per core in total, which is exactly the headroom the shipped 256-token
prefill SDPA chunk needs.

**Fix.** `OrnithModel._share_fused_gate_buffers` keeps one set for the whole stack. Safe because the
layers run sequentially in the traced graph as well as eagerly: a layer's gate output is consumed by
its own scatter before the next layer's gate call writes the buffer again. This preserves the decoder
stage's measured prefill SDPA chunk, which is the alternative that would otherwise have had to be
given up.

**A hypothesis that was wrong, kept because it is now a contract.** The first suspect was the
embedding: both `ttnn_prefill_forward` and `ttnn_decode_forward` now name `ttnn.DRAM_MEMORY_CONFIG`
explicitly, and the residual test asserts the memory config as well as the values. That change did
**not** fix the clash — same addresses, same error — and the reason it is right anyway is not the
one first written down here. `ttnn.embedding` does not default to L1; it defaults to the *indices*
tensor's memory config (`output_mem_config.value_or(input_tensor_arg.memory_config())`,
`embedding_device_operation.cpp`). So the residual's placement would silently follow wherever a
caller happened to build its token tensor, and the decoder stage's inter-layer contract says DRAM
interleaved. Naming it pins the contract; it never was the L1 pressure.

### 5.3 Lazy trace capture threw the prompt away

**Symptom.** None visible — which is the point. The low-level pair (`prefill_forward` then
`decode_forward(enable_trace=True)`) on a generator that had not captured yet returned a fluent,
finite, in-vocabulary token that had **not read the prompt**.

**Mechanism.** Trace capture is not free of side effects: it warm-compiles a real decode step, which
advances every DeltaNet recurrent row and writes one paged KV entry, and that contamination has to
be wiped before the capture is usable. The wipe is `model.reset_state()` — it zeroes the recurrent
and conv state *and* the whole paged KV cache. Capture was lazy, first triggered inside
`decode_forward`, so on a fresh generator the order was: prefill writes the prompt → decode triggers
capture → capture wipes the prompt → the traced step decodes from an empty cache. `generate()` was
never affected (it captures before it prefills), and neither was any probe or test that called
`_ensure_decode_trace()` up front — which, it turned out, all of them did. Every assertion around the
low-level path checked `isfinite` and position advance, both of which an empty cache satisfies.

**Fix.** `prefill_forward` captures **before** it writes anything, so by the time any decode runs the
traces exist and no wipe is pending. The remaining orderings are refused rather than tolerated: the
generator tracks whether prompt state is live and `_ensure_decode_trace` raises instead of silently
wiping. `test_low_level_prefill_then_decode_sees_the_prompt` is the regression test — it drives the
low-level pair on a generator whose traces have been released and asserts the token matches what the
high-level path produces for the same prompt, which is the comparison that would have caught it.

**The A/B**, committed as [`logs/probe_lazy_capture_ab.txt`](logs/probe_lazy_capture_ab.txt). With
the fix reverted in place (`prefill_forward` not capturing, the live-state guard disabled) and
nothing else changed, the new test fails on the token itself:

```
assert int(step[0]) == int(expected[1])
E   assert 102909 == 10980
```

10980 is what the same prompt produces through `generate()`; 102909 is what a decode step reads out
of a cache that was zeroed a moment earlier. Not a tolerance, not a near miss — a different token.
Unlike §5.1's, this A/B is **not** reproduced by `run_evidence.sh`: reproducing it means shipping the
defect, since its trigger is the call order the fix removes rather than an order a probe can choose.
The committed file is the console output of that one-off run, labelled as such.

Found by stage review, not by the suite: the reviewer read the call graph rather than the assertions.

---

## 6. Accuracy and generated text

Reference: `readiness_aime24_chat.refpt`, generated **fresh by this stage** with
`readiness_aime24_chat.meta.json` recording the HF model id, the snapshot revision
(`5df2ed3f675c7beaa490328cc70bb573b65fb660`), the tokenizer class, the chat-template flag, the prompt
source and index, the generation length, the top-k and the exact command. Nothing was carried
forward.

```
python -m models.common.readiness_check.generate --hf-model ornith-ai/Ornith-1.0-35B \
  --prompt-source aime24 --chat-template --gen-len 100 --top-k 100 \
  --output models/autoports/ornith_ai_ornith_1_0_35b/readiness_aime24_chat.refpt
```

That run predates the log discipline the rest of this directory follows, so there is no committed
`logs/*.txt` for it — the metadata (including the artifact's sha256) is its provenance. What the
missing log would have shown is in the committed ones anyway: every readiness run that loads the HF
side prints

```
Loading ornith-ai/Ornith-1.0-35B as Qwen3_5MoeForConditionalGeneration
  (checkpoint declares architectures=['Qwen3_5MoeForConditionalGeneration'])
```

with **no** missing-or-newly-initialised-weight warning anywhere in the log — which is the exact
failure `models/common/readiness_check/hf_model.py` exists to prevent (§8 of the work log).

`run_prefill_check` 0.950 / 1.000 / 1.000 and `run_teacher_forcing` 0.940 / 1.000 / 1.000
(top-1 / top-5 / top-100) — [`readiness_prefill.json`](readiness_prefill.json),
[`readiness_teacher.json`](readiness_teacher.json).

### 6.1 Free-running generation

`run_autoregressive`, 128 tokens, greedy, both sides. Read, not just scored.

**Chat-template prompt** ("explain how a rainbow forms, for a curious ten-year-old",
`readiness_autoregressive_chat/`) — the TT model reproduces the HF control's structure and content
almost exactly, including the model's `<think>` scaffold. 42 of 128 tokens are identical to the
control and the first divergence is at token 15, after which the two texts keep converging and
re-diverging on wording rather than on meaning:

> HF: `1. **Analyze User Request:** - **Topic:** How a rainbow forms - **Audience:** Curious
> ten-year-old ... - Raindrops act like tiny prisms - Light refracts (`
>
> TT: `1. **Analyze User Input:** - **Topic:** How a rainbow forms - **Audience:** Curious
> ten-year-old ... - Water droplets in the air act like tiny prisms - Light refracts (bends) when
> entering the droplet - Light reflects off the inside back of the droplet ...`

**Raw continuation prompt** (the shared runner's default, `readiness_autoregressive/`) — labelled
continuation stress coverage for an instruct model, per `$qualitative-check`, not the quality
verdict. Fully coherent English, no repetition, no language drift; it diverges from HF at token 11
on a creative continuation and then follows its own consistent story:

> HF: `something strange: the shadows of the trees were getting shorter as the sun rose higher.`
>
> TT: `something strange: the shadows of the trees were getting shorter each morning. Elena was
> fascinated. She decided to keep a journal and record the length of the shadows every day ...`

`check_degenerate_output.py --missing-artifacts critical --scope autoregressive` is clean over both —
[`degenerate_report.json`](degenerate_report.json).

### 6.2 The shared qualitative suite

`models/common/readiness_check/vllm_prompts.txt`, all six prompts, rendered with
`tokenizer.apply_chat_template(add_generation_prompt=True)` because the checkpoint has a chat
template — the prompt-format decision, the rendered prompts and the token ids are all recorded in
[`readiness_qualitative.json`](readiness_qualitative.json) alongside an HF control generated the same
way. Every TT completion is coherent, on-topic and structurally matched to its HF control.

The one oddity worth naming: prompt 3's TT completion opens `Here's a thinking thinking sequence`.
The **HF control for the same prompt in the same run opens with the identical phrase**, so the
doubled word is the checkpoint's, not the port's — the kind of thing that reads as a repetition bug
until the control is checked.

This suite is also what found §5.1: it was the first thing to drive **six requests through one
generator**, and it is the reason the corruption was caught in this stage rather than in vLLM.

---

## 7. Split-sampling and trace contract

Two captures, and the evidence that they are wired together rather than merely present
(`test_split_sampling_feeds_the_token_back_on_device`):

1. the **model** trace: token buffer → embedding → 40 layers → final norm → LM head → sampler-ready
   vocab-sharded logits, then `ttnn.plus_one(current_pos, skip_negative_entries=True)` and
   `ttnn.plus_one(rot_idxs)` **inside the captured graph**;
2. the **sampling** trace, owned by `SamplingGenerator`, captured over the model trace's own output
   tensor with `tt_out_tok` bound to the persistent `[1, 1, 1, 32]` decode token buffer.

The test asserts, over four consecutive replays:

* `slot["output"][0] is tok_buf` and `slot["input"] is self._trace_logits` — the sampler consumes the
  model trace's output and writes into the decode token input, by tensor identity;
* the token consumed by replay *N+1* is the token the sampler wrote in replay *N*;
* `current_pos` and `rot_idxs` both advance by exactly one per replay and stay equal to each other;
* `page_table_refreshes`, `token_refreshes` and `position_refreshes` are **unchanged** across the
  four replays.

`test_a_changed_page_table_is_copied_exactly_once` covers the other half: an unchanged page table is
never copied, a changed one is copied exactly once, the copy reaches the trace input tensor, and a
repeat of the same table copies nothing.

`test_greedy_decode_has_no_host_fallback` monkeypatches `ttnn.from_torch`, `ttnn.to_torch`,
`ttnn.copy_host_to_device_tensor` and `ttnn.argmax` to raise, then runs three traced token-out steps.

Measured steady-state counters for a 128-token generation ([`perf_summary.json`](perf_summary.json)):

```
decode_calls 127   token_refreshes 0   position_refreshes 1   rope_refreshes 1   page_table_refreshes 0
```

One position/RoPE write at the request boundary, and nothing per token. Teacher forcing is the one
mode that does write tokens from host — by construction, since the harness overrides the prediction —
and it writes only when the forced token differs from the sampled one.

**On the teacher-forcing number.** `run_teacher_forcing` drives
`generator.generate(..., next_input=..., enable_trace=True)`; the predicted token still comes out of
the token-out path, sampler included, so its **38.19 t/s/u**
([`readiness_teacher.json`](readiness_teacher.json)) is a token-out figure and not a logits-only
one — the comparable logits-only measurement is the separate 44.9 t/s/u row in §1.

The same run's *generator* loop logs **41.80 t/s/u** for its own decode window
(`logs/readiness_teacher.txt`). The 3.61 t/s/u between them is not the model: the runner times
between its own per-token `next_input` callbacks, so its window additionally contains that callback
(which does torch work to score the token) and the host token write on the ~6 % of steps where the
forced token differs from the sampled one. 38.19 is the honest number for "what the readiness
harness measures"; 41.80 is the honest number for "what the generator's decode loop costs", and it
agrees with the free-running 41.87 t/s/u in §1.

---

## 8. Performance accounting

`logs/bench_full_model.py`, warmed, three runs, best decode:

```
layer-stack lower bound        21.45 ms/token   30 x 0.564 (linear) + 10 x 0.453 (full)
+ embedding, final norm,
  LM head, device plus_one      0.81 ms         -> traced logits-only   22.26 ms/token
+ sampling trace                1.17 ms         -> replay + sample      23.43 ms/token
+ synchronize and readback      0.45 ms         -> token-out            23.88 ms/token
```

The full-model-only cost is 2.43 ms/token, 11 % over the lower bound, and none of it is avoidable
sampler work: the sampler is 4.9 % of the step after §4.3, against 33 % before it.

`tt-perf-report` for the **reduced profiling variant** (one real `linear_attention` layer, one real
`full_attention` layer, real weights, real cache/page-table shapes, real terminal path) is in
[`tracy/`](tracy/), regenerated by `tracy/run_profiling.sh`. Op shares in the signposted decode
window:

| op group | share of the *reduced* decode window |
|---|---|
| `TopKDeviceOperation` (sampler stage 1 + 2) | 30.0 % |
| `MatmulDeviceOperation 32 x 2048 x 62080` (LM head) | 15.7 % |
| `BinaryNg` (elementwise) | 7.3 % |
| routed `SparseMatmul` (gate/up + down) | 9.1 % |
| `ReduceScatter` + `AllGather` (`ttnn.all_reduce`'s two per-layer collectives) | 5.1 % |
| `AllGatherAsync` (the sampler's candidate gather, through the shim) | 1.0 % |

**Read that table with the denominator in mind.** The reduced variant has 2 layers, not 40, so its
terminal path is ~20x over-represented: the same `TopK` rows are 3.0 % and the same LM head 1.6 % of
the delivered 40-layer step, which is what §1's wall-clock accounting measures. The reduced capture
is what `$full-model` asks for and what the profiler can actually reassemble — see limitation 2.

The decoder-layer rows in the capture reproduce stage 5's picture exactly: `GeneralizedMoeGate` at
2 µs on 32 cores, no `TopK` in the router chain at decode, `HiFi2 BF16 x BFP8 => BF16` on the dense
projections, `LoFi BF16 x BFP4 => BFP8` on the routed gate/up. Claimed policy equals measured policy,
in the full model as in the layer.

---

## 9. Runtime fallback audit

| path | host work in the steady state | evidence |
|---|---|---|
| model decode (`ttnn_decode_forward`) | none. Device tensors in, device tensors out; positions advanced on device | `test_greedy_decode_has_no_host_fallback` |
| sampling | none. Traced, `tt_out_tok` into the decode token buffer | `test_split_sampling_feeds_the_token_back_on_device` |
| token feedback | none. No readback-and-rewrite | same test |
| page table | copied only when it changes | `test_a_changed_page_table_is_copied_exactly_once` |
| positions / RoPE | one host write per request, at the boundary | counters in `perf_summary.json` |
| caller readback | one `to_torch` of the 32-entry token buffer per step — caller-visible by contract, 0.066 ms (`token readback only`) | `probe_terminal_grouped.txt` |
| prefill | host token upload per 2048-token chunk and a host logits compose; **not** in the decode loop, and inside TTFT | `logs/bench_full_model.py` |
| model construction | `ttnn.as_tensor` per weight, `allocate_state`'s conv1d preparation | setup only; `allocate_state` is called explicitly before any capture or measurement |
| cache ownership | explicit, and **sticky**. `kv_cache=None` uses the generator's own cache and page table *until* a call passes one; `OrnithModel.attach_kv_cache` has no inverse, so after `prefill_forward(kv_cache=X)` a later `kv_cache=None` keeps driving X and the caller has to re-attach the original. A caller-owned cache is re-validated against the traces (identity is part of what makes a capture valid), and no evidence run here ever attaches one | `test_a_caller_can_own_the_cache`, `test_a_caller_owned_cache_attached_late_forces_a_recapture` (both restore the generator's cache explicitly, which is the same observation) |
| host-logit boundary | only `decode_logits_to_host` and `_logits_to_host`, used by `return_all_logits`, by the host-sampling compatibility mode and by tests — never by the measured token-out path | `sampling_mode="host"` is a constructor argument, not a fallback |
| reset | zeroes the DeltaNet recurrent/conv state and the paged KV cache, re-stages the trace inputs, keeps traces and weights | `test_reset_wipes_state_but_keeps_traces_and_weights` |
| eager decode | `generate(enable_trace=False)` exists as a model-local debug path and is used for no evidence here | `_generate_eager` |
| sampling params | persist across `generate()` calls when the caller omits `sampling_params`. The constructor sets greedy, so every evidence run here is greedy; a serving caller that omits them inherits the previous request's `k`/`p`/`temp` | `_apply_sampling_params` is keyed on the formatted tuple and is a no-op when unchanged |
| trace capture | happens inside `prefill_forward` (and `generate`, after its reset), before any prompt state exists, because capture warm-compiles a real decode step and then wipes the state — §5.3. Liveness is tracked on the **model** (`OrnithModel.state_is_live`), so it covers every write path including `prefill_forward_single`, which `generate` uses; a capture attempted with live state raises | `test_low_level_prefill_then_decode_sees_the_prompt`, `test_the_eager_debug_path_does_not_poison_the_traced_one` |
| eager decode via `generate(enable_trace=False)` | writes state without capturing, and does not poison the traced path: the next `generate()` resets before it captures, and the two agree token for token | `test_the_eager_debug_path_does_not_poison_the_traced_one` |
| `sample_on_device` in host mode | refused with a `ValueError`; the host compatibility mode builds no sampler graph, so there is nothing to replay | `test_host_sampling_mode_refuses_device_sampling` |
| chunked prefill continuation | `prefill_forward(..., continue_from_state=True)` carries the DeltaNet state across calls at batch 1 and **raises** above it rather than resuming from another slot's state. Each continuation's `start_pos` must be a multiple of the 2048-token prefill chunk — the decoder layer's own `start_pos % chunk_size == 0` requirement, raised as a `ValueError` rather than silently mis-positioned | `test_chunked_prefill_continuation_matches_a_single_call` |
| page-table validation | rows are checked against the decode batch **and** blocks-per-user against the highest position the call addresses; a short row would make the paged SDPA kernel read past the row's end | `test_a_page_table_too_narrow_for_the_position_is_rejected` |
| high-level `generate()` at `max_batch_size > 1` | prefills **slot 0 only** — one page-table row, one merged state — then broadcasts the position (and, in teacher forcing, the token) to every row and returns slot 0. Slots 1..B-1 decode from a zeroed state and their tokens are discarded; `batch_slots.json` measures that they do not affect slot 0. It **merges the prefill state into slot 0 first** (`prefill_request_into_slot`) — prefill runs on the batch-1 state pack and the decode trace is bound to the batch-B pack, so without the merge the recurrent layers would decode from a zeroed state. It is a single-request convenience; the **batched** surface is the low-level `prefill_forward`/`decode_forward` pair, which takes per-row prompts, positions and inactive rows | `test_batch_four_generate_agrees_with_batch_one`, `test_the_batched_prefill_state_reaches_every_decode_slot`, `test_batched_prefill_and_decode_with_mixed_prompt_lengths` |

`sampling_mode="host"` is the explicit host-sampling compatibility mode the goal asks for: it reads
the full logits and argmaxes on host. `test_host_sampling_compatibility_mode_agrees_with_device_sampling`
asserts it emits the same greedy tokens as the on-device path, and every performance number in this
document is from `sampling_mode="device"`.

---

## 10. Watcher

`logs/run_watcher.sh` runs the suite under `TT_METAL_WATCHER=10` in a separate run from the profiler,
with `TT_METAL_WATCHER_DISABLE_ETH=1` inherited from the decoder stage's limitation 2 (watcher does
not cover ACTIVE_ETH cores on this configuration; every worker-core assert stays armed). All **42**
fast cases pass under watcher with **0** error/assert/hang lines in either the pytest log or
`generated/watcher/watcher.log` — [`watcher/watcher_error_count.txt`](watcher/watcher_error_count.txt),
[`watcher/watcher_pytest.txt.gz`](watcher/watcher_pytest.txt.gz),
[`watcher/watcher.log.gz`](watcher/watcher.log.gz).

One caveat on the artifacts: tt-metal truncates `generated/watcher/watcher.log` every time the mesh
is opened, and the suite opens it once per test, so the committed copy is the **last** session's log
only. The pytest log is the artifact that covers all 42 cases — watcher writes its errors and asserts
to stderr as well as to its own file, and there are none in either.

---

## 11. Known limitations

1. **Batch × context is bounded by DRAM, and the advertised context is not.** At batch 1 the full
   262144-token cache fits with 24.17 GiB spare and a 262143-token prompt runs through it; at batch
   32 the KV ceiling is 152,113 tokens per user with zero activation headroom. The caller chooses
   through `cache_context`; §3 has the arithmetic and `logs/probe_footprint.py` measures it.
2. **The `tt-perf-report` capture is a 4-replay window on the reduced variant.** `process_ops_logs`
   fails with *"Device data missing: Op N not present in cpp_device_perf_report.csv"* on longer
   windows and on the 2048-token prefill chunk's setup probes, and raising `--op-support-count` past
   the op ids involved makes the post-processing consume 14 GB and over an hour without finishing.
   Four replays is enough for op shares and is what the committed report covers; the *latency*
   numbers in §1 and §8 come from un-profiled wall clock, which is the right basis for them anyway.
3. **A newly seen prompt length costs one trace re-capture, and that cost is unmeasured.** It is the
   price of §5.1's correctness guard, and it happens once per length. It lands in *neither* reported
   metric: `generate` stops the TTFT clock when the first token is sampled and starts the decode
   clock after `_ensure_traces_replay_safe`, so a cold-length request is slower than any number here
   says. `logs/bench_full_model.py` warms the length first and asserts `trace_recaptures` does not
   move during the measured runs, so the published figures are honest for a warmed request and
   silent about a cold one. Pre-compiling a bucket of prompt lengths at construction would remove the
   cost entirely and is left to the optimized-full-model stage; `generator.trace_recaptures` makes
   the event visible rather than hidden.
4. **`models/common/tests/test_sampling.py::test_log_probs_calculation` fails on this mesh**, before
   and after this stage's change (verified by stashing it): `LogProbsCalculator` supports 8- and
   32-device meshes and this is a 1x4. Pre-existing, unrelated, and logprobs are not used here.
5. **Long-context *accuracy* is inherited, not re-measured at the full-model level.** The decoder
   stage validated PCC to 8000 tokens against an eager HF reference that is not tractable on host
   beyond that, and this stage's accuracy gates run at 261 tokens (the AIME24 chat reference). The
   *shape and capacity* side is no longer bounded by that: `logs/probe_long_prompt.py` runs 262143
   tokens through the full stack (§3). What is missing above 8000 is a reference to compare
   against, not a path that works.
6. **One mesh shape.** `DEFAULT_MESH_SHAPE = (1, 4)`, as the goal directs.
7. **An inactive row's RoPE index advances while its position does not.** `current_pos` carries a
   `-1` sentinel that `ttnn.plus_one(..., skip_negative_entries=True)` preserves; `rot_idxs` is
   `max(current_pos, 0)` because the RoPE gather reads its table unconditionally, so it has no
   sentinel to skip and advances on every replay. Harmless and bounded — the row's RoPE output feeds
   a Q/K it never writes to cache, the index can only reach the number of replays in one request,
   and the context bound caps that below the table's `align_up(max_context, chunk) + chunk` rows;
   every request boundary rewrites both tensors. Deriving `rot_idxs` from `current_pos` inside the
   graph would remove the asymmetry for two ops per step and is the right fix for the serving stage.
8. **The trace-allocation guard is a generator-level mitigation, not a tt-metal fix.**
   `_ensure_traces_replay_safe` keys on program-cache growth and on the attached KV cache changing,
   which covers the demonstrated trigger (§5.1) and the caller-owned-cache case. The underlying
   hazard is broader: *any* device buffer allocated after capture and still alive when a trace
   replays can be corrupted. Every such buffer in the delivered path is now either short-lived or
   built at setup — the per-request page-table row is explicitly deallocated after prefill, and the
   per-slot merge masks, which used to be built lazily on the first batched prefill, are
   preallocated by `_prebuild_slot_masks` in `allocate_state` for exactly this reason (they were
   caught only incidentally, because `_merge_rows` also compiles programs). A *future* path that
   allocates a long-lived buffer without compiling a program would still be unprotected, and a
   program compiled *inside* the decode loop is only warned about after the fact.
   `logs/probe_bisect.py` is a small self-contained reproducer and is worth reporting upstream; no
   issue has been filed yet.
9. **Batched prefill fills slots 0..B-1 in order; it cannot target one slot.**
   `prefill_forward` maps row `u` of `tokens` to page-table row `u` and merges that user's DeltaNet
   state into decode slot `u`, so a scheduler that wants to prefill *only* slot 2 has to pass three
   rows. Fixed slots, mixed lengths and inactive rows all work (§3), and the paged KV half is
   already per-row through the page table; what is missing is a `slot`/`request_id` argument and the
   slot → prefill-pack restore that `continue_from_state` would need above batch 1. It belongs with
   the serving adapter, and the vLLM stage will want it on day one.
10. **Greedy trajectories at batch 4 and batch 1 separate after the first decoded token.**
   The prompt token and the first decoded token are identical; from the third token the two runs
   pick different near-ties. `logs/probe_batch_slots.py` →
   [`batch_slots.json`](batch_slots.json) pins what this is and is not: slot 0's tokens are
   **byte-identical whether its three neighbours hold the same prompt or three different ones**, so
   it is not cross-request leakage — it is the batch changing the matmul and MoE-grouping geometry
   and therefore the last bits of the logits. The same probe shows what a *real* state defect looks
   like for comparison: with the prefill→slot merge removed, the **first** decoded token already
   changes (240560 → 169222).
11. **Non-greedy sampling is wired and smoke-tested, not measured.**
   `test_alternating_greedy_and_sampled_requests_stay_correct` runs a real top-k/top-p request
   between two greedy ones and asserts greedy still reproduces and that no trace is re-captured, but
   there is no accuracy or latency evidence for sampled decoding, and `LogProbsCalculator` does not
   support a 1x4 mesh at all (it requires 8 or 32 devices), so vLLM logprobs will need work.

---

## 12. Exact artifacts

```
models/autoports/ornith_ai_ornith_1_0_35b/
├── tt/model.py                                the model: embeddings, stack, norm, LM head, sampler wiring
├── tt/generator.py                            the readiness/serving generator: traces, split sampling
├── tests/test_full_model.py                   42 fast cases + 5 long ones
│                                              (logs/pytest_full_model.txt.gz and logs/pytest_long.txt)
├── readiness_aime24_chat.refpt                the fresh AIME24 chat-template reference
├── readiness_aime24_chat.meta.json            its provenance, for the "exact match" test a later stage makes
├── readiness_autoregressive/                  HF vs TT, raw continuation prompt
├── readiness_autoregressive_chat/             HF vs TT, chat-template prompt
└── doc/full_model/
    ├── README.md                              this file
    ├── work_log.md                            what was done, in order, with the measurements
    ├── perf_summary.json                      §1 and §8, written by logs/bench_full_model.py
    ├── footprint.json                         §3, written by logs/probe_footprint.py
    ├── footprint_batch32.json                   §3, the 40-layer build at the advertised batch bound
    ├── long_prompt.json                         §3, the non-aligned prompt ladder up to 262143
    ├── batch_slots.json                         §11.10, what batch 4 changes about slot 0
    ├── defect_qualitative_before_fix.json       §5.1, the recorded symptom, from the pre-fix code
    ├── readiness_{prefill,teacher}.json         §6 accuracy
    ├── readiness_autoregressive{,_chat}.json   §6.1
    ├── readiness_qualitative.json              §6.2, prompt-format metadata + HF control + TT
    ├── degenerate_report.json                  the runner-side degeneracy gate's own output
    ├── autoregressive_chat_prompt.txt          the chat-rendered free-running prompt
    ├── logs/
    │   ├── run_evidence.sh                    regenerates everything below, in order
    │   ├── run_readiness.py                   drives the shared readiness runners on a 1x4 mesh
    │   ├── bench_full_model.py                §1 / §8
    │   ├── probe_footprint.py                 §3
    │   ├── probe_long_prompt.py               §3, the near-context non-aligned prompt ladder
    │   ├── probe_batch_slots.py               §11.10, slot merge and batch-geometry divergence
    │   ├── probe_lazy_capture_ab.txt          §5.3, the pre-fix arm's console output
    │   ├── probe_terminal.py                  §4.3, the terminal-cost breakdown
    │   ├── probe_topk.txt                     §4.3, the ttnn.topk width/rows/cores ladder
    │   ├── probe_bisect.py                    §5.1, the minimal repro
    │   ├── update_context_contract.py         §3, writes doc/context_contract.json's full_model block
    │   ├── probe_multi_prompt.py              §5.1, the multi-request regression probe
    │   ├── profile_reduced.py                 the signposted window tracy/ captures
    │   ├── smoke.py                           the two-minute reduced-model probe used while debugging
    │   ├── run_watcher.sh                     §10
    │   └── *.txt                              each script's committed output
    ├── tracy/                                 §8, run_profiling.sh + the two perf reports
    ├── watcher/                               §10
    └── triage/                                tt-triage from the hang in work log §4
```

Reproduce, in this order. `run_evidence.sh` covers the readiness gates, the qualitative suite, the
benchmark, the probes, the degeneracy gate and the fast suite; the other six are separate commands
because the profiler and the watcher must not share a run with each other or with anything else, and
because the long suite, the prompt-length ladder, the two footprint probes and the contract update
are capacity and shape evidence rather than behaviour:

```bash
R=models/autoports/ornith_ai_ornith_1_0_35b/doc/full_model
bash $R/logs/run_evidence.sh                       # readiness, qualitative, bench, probes, fast suite
python $R/logs/probe_topk.py > $R/logs/probe_topk.txt
pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_full_model.py -m long -q
python $R/logs/probe_long_prompt.py --budget-s 2400
python $R/logs/probe_footprint.py --cache-context 262144 --batch 1  --output $R/footprint.json
python $R/logs/probe_footprint.py --cache-context 8192   --batch 32 --output $R/footprint_batch32.json
python $R/logs/update_context_contract.py
bash $R/tracy/run_profiling.sh                     # separate run: profiler
bash $R/logs/run_watcher.sh                        # separate run: watcher
```

Commit SHAs for this stage are at the end of [`work_log.md`](work_log.md).

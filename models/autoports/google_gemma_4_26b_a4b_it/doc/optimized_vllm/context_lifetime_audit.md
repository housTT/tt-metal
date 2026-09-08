# Retained-trace prefill capacity audit

Independent stage reviewer; read-only source and existing artifact analysis.
No TTNN import, hardware operation, server, pytest, reset, or profiler was run.

## Finding

Retaining decode traces across prefill extends tensor lifetimes, so unchanged
tensor shapes do **not** establish an identical peak allocation. The earlier
maximum-context probes prefill before creating their first decode trace; they
do not directly exercise this new overlap.

The actual serving configuration uses a 2,048-token scheduler prefill budget on
each profile. Reconstructing the conservative source-live activation envelope
for that budget, using each profile's own observed construction data and actual
serving cache allocation, covers the retained-trace overlap with positive
accounting headroom on all three profiles. This is a source-derived serving
capacity argument, **not a measured high-water or fragmentation result**.

No additional physical-capacity failure is demonstrated by this audit. The
50,624 / 262,144 / 262,144 logical context limits remain applicable to the
recorded serving configuration. Larger caller-selected prefill budgets or
different cache/trace configurations require new accounting; this report does
not establish their peak memory.

## Recorded serving configuration

The following are independent observations from each profile's
`before_warmed/run_manifest.json` and `before_warmed/server.log`, under
`../../readiness_vllm/<profile>/optimized_vllm/`. All use 32 scheduler slots and
a 220,000,000-byte trace reservation. These are P300C Blackhole 1/2/4-chip
proxies for the named profiles.

| Profile | Served logical context | Scheduler prefill budget | KV blocks | Relevant server-log lines |
| --- | ---: | ---: | ---: | --- |
| P150 | 50,624 | 2,048 | 823 | 40: chunk budget; 736: block override; 738: max-context concurrency 1.28x |
| P150x2 | 262,144 | 2,048 | 4,128 | 40: chunk budget; 743: block override; 745: max-context concurrency 1.80x |
| P150x4 | 262,144 | 2,048 | 4,128 | 40: chunk budget; 769: block override; 771: max-context concurrency 1.80x |

The generic logged `GPU KV cache size` of 8,768 / 44,032 / 44,032 tokens is
not a replacement for the heterogeneous model's logical maximum. The log's
maximum-context concurrency calculation and the worker's
`_validate_tt_kv_cache_capacity` account for the hybrid groups.

The adapter slices `tokens[row, start:end]` and sends `end-start` tokens to the
generator (`tt/generator_vllm.py:382-410`). Thus the scheduler's prefill budget
limits the transient query/prefill sequence; it does not truncate the logical
request context or reduce the allocated KV/RoPE capacity.

## Allocation ownership

- `DecodeTrace` in `tt/generator.py:62` retains one terminal logits tensor,
  token input/output, current positions, RoPE positions, sampler parameters,
  seed sentinel, and references to the existing state. The sampled token aliases
  the persistent token input. KV cache and page tables are already owned by the
  serving state and are not duplicated by retaining its reference.
- `_get_or_capture_decode_trace` clears the previous trace set on a cache miss.
  The adapter releases on physical batch-shape changes. This path keeps one
  model/sampling trace pair, not a growing collection of request traces.
- `_terminal(..., sampler_ready=True)` retains BF16 logits with 32 physical
  rows and vocabulary-sharded width `262144 / TP`. That is 16,777,216 /
  8,388,608 / 4,194,304 bytes per device. The eight token/position/parameter
  vectors have only 32 elements each. The table below charges **32 MiB on
  every profile** for these retained tensors, comfortably exceeding their
  direct payload plus alignment; this is an accounting allowance, not a
  measured allocation value.
- The sampler's lazy persistent index/seed/user buffers are preloaded in the
  generator constructor. The capacity artifacts explicitly record
  `sampler_buffers_preloaded: true`; these are already in the construction
  baseline. Sampling temporaries are local to `Sampling1D` calls and do not
  become additional per-request retained state.
- `MeshTraceDescriptor` (`tt_metal/distributed/mesh_trace.hpp`) stores host
  command vectors and descriptors. `MeshTrace::populate_mesh_buffer`
  (`mesh_trace.cpp:74`) allocates command storage inside the configured trace
  region when that region is nonzero. Registering an active trace updates
  allocation-safety tracking (`impl/allocator/trace_allocation_tracker.cpp:76`);
  it does not reserve every intermediate tensor in the capture's address
  footprint. The full 220 MB trace region is charged independently below.
- `Gemma4Generator.prefill_forward` runs users serially and retains their
  last-token logits until concatenation. Every one-row output has a physical
  32-row tile. Conservatively charge 32 such outputs plus a 32-row concat
  output: `33 * 32 * (262144 / TP) * 2` bytes. This also covers two simultaneous
  terminal softcap outputs while earlier users' outputs remain live.
- `Gemma4FullModel` does not pass `max_batch_size` into decoder-weight or RoPE
  construction. Apart from cache/state allocation, these constructor tensors
  are unchanged between the B1 capacity control and serving. Page-table shapes
  in the capacity controls already have 32 rows and match serving's layer
  geometries.

Cold program-cache growth can add runtime buffers during prefill. The adapter
retires retained traces after detecting this growth and before the next replay;
that guard establishes replay safety, not equal peak memory. Runtime/alignment
and fragmentation allowance remains charged separately in this calculation.

## Per-profile KV and construction accounting

The observed baselines are
`../optimized_full_model/final/capacity/capacity_tp{1,2,4}.json`. They include the
BF16 row-major embedding, complete 30-layer model, RoPE caches, final norm,
preloaded sampler and persistent CCL resources. No old BFP8 embedding projection
is used as the construction baseline. TP4's later selected BFP4 expert policy
reduces storage; this calculation takes **no credit** for that reduction and
keeps the older, larger observed allocation as a conservative baseline.

The plugin derives local cache shapes in `model_runner.py:_kv_cache_shape` and
shared tensor IDs in `_build_per_layer_specs`. Six hybrid groups of five layers
produce five shared K/V pairs when full/sliding physical block volumes agree.
The adapter keys allocations by `(tensor_idx, physical_block_elements)`:

- P150: both block kinds contain 131,072 BF16 elements. KV bytes are
  `5 * 2 * 823 * 131072 * 2 = 2,157,445,120`.
- P150x2: both contain 65,536 elements. KV bytes are
  `5 * 2 * 4128 * 65536 * 2 = 5,410,652,160`.
- P150x4: sliding blocks contain 32,768 elements and full blocks contain
  65,536 elements; their storage cannot alias. KV bytes are
  `5 * 2 * 4128 * (32768 + 65536) * 2 = 8,115,978,240`.

These are source-derived physical tile bytes from the logged block counts, not
a live allocator snapshot. These cache shapes have exact tile volumes.

`AllocatorImpl::init_one_bank_per_channel` rounds each bank's trace region to
`kMaxTraceBufPageSize=8192`. With eight banks, the requested 220,000,000 bytes
therefore reserves `8 * ceil(220000000 / 8 / 8192) * 8192 = 220,004,352` bytes.
The table includes this 4,352-byte rounding charge; the earlier 64/128 MiB
control reservations are already exact multiples of this allocation unit.

| Bytes per device | P150 | P150x2 | P150x4 |
| --- | ---: | ---: | ---: |
| Observed usable DRAM in capacity control | 34,111,622,144 | 34,111,622,144 | 34,044,513,280 |
| Control's trace reservation | 67,108,864 | 67,108,864 | 134,217,728 |
| Additional rounded trace reservation | 152,895,488 | 152,895,488 | 85,786,624 |
| Adjusted usable DRAM | 33,958,726,656 | 33,958,726,656 | 33,958,726,656 |
| Observed constructed allocations | 29,391,144,448 | 18,321,312,768 | 12,225,484,288 |
| Standalone KV removed from that observation | 1,247,805,440 | 2,789,212,160 | 2,736,783,360 |
| Serving KV substituted | 2,157,445,120 | 5,410,652,160 | 8,115,978,240 |
| Adjusted constructed allocations | 30,300,784,128 | 20,942,752,768 | 17,604,679,168 |
| Available after adjusted construction | 3,657,942,528 | 13,015,973,888 | 16,354,047,488 |

## Conservative prefill envelope and reconciliation

Use the source-live formulas in `../multichip_decoder/capacity_projection.json`,
independently reconstructed in `tests/test_multichip_decoder.py:375-433`, with
physical `S=2048` and logical `L=2047`. Charging the nonaligned padding copies
at `L=2047` also bounds the aligned 2,048-token case. Keep the older, conservative
1,024-token MoE chunk accounting; no credit is taken for smaller expert tiles
or the later explicit Q lifetime improvement.

Define BF16 byte terms:

```text
H = S * 2816 * 2
Q = S * (16 / TP) * 512 * 2
QKV = S * ((16 / TP) + 2 * max(1, 2 / TP)) * 512 * 2
R = 2 * S * 512 * 2
P = L * (2816 + 2 * 512) * 2
Hchunk = 1024 * 2816 * 2
G = 128 * 1024 * local_expert_width * 2
D = 128 * 1024 * 2816 * 2
router = S * 128 * 2; router_chunk = 1024 * 128 * 2

attention_concat = 2*H + 2*QKV + 2*Q + R + P
attention_reduce = (4 if TP=1 else 5)*H + 2*QKV + Q + R + P
dense = (6 if TP=1 else 7)*H + 5*S*local_dense_width*2 + R + P
moe = (9 if TP<4 else 10)*H + Hchunk + 3*G + 2*D
      + router + R + 2*router_chunk + P
```

Local dense widths are 2,112 / 1,056 / 544 and local expert widths are
704 / 352 / 192, including profile-specific tile padding.

| Bytes per device | P150 | P150x2 | P150x4 |
| --- | ---: | ---: | ---: |
| Attention concat envelope | 193,978,880 | 118,481,408 | 84,926,976 |
| Attention reduce envelope | 183,493,120 | 136,307,200 | 111,141,376 |
| Dense MLP envelope | 132,375,040 | 122,282,496 | 111,796,736 |
| MoE envelope, largest of these terms | 2,160,583,168 | 1,883,759,104 | 1,769,464,320 |
| 32 retained prefill logits plus concat output | 553,648,128 | 276,824,064 | 138,412,032 |
| Retained decode trace tensor allowance | 33,554,432 | 33,554,432 | 33,554,432 |
| Inherited allocator/runtime reserve | 330,301,440 | 581,959,680 | 707,788,800 |
| **Remaining accounting headroom** | **579,855,360** | **10,239,876,608** | **13,704,827,904** |

For example, P150 is exactly
`3,657,942,528 - 2,160,583,168 - 553,648,128 - 33,554,432 - 330,301,440
= 579,855,360` bytes. The other two columns use their own inputs and the same
reconciliation; their conclusion is not inferred from P150's result.

The transient envelope and all-user retained-logits charge are deliberately
added even though the scheduler cannot give every user a 2,048-token chunk in
one submission. This preserves slack rather than taking credit for that
mutual exclusion. Source-local token/position staging and alignment are small
relative to the separately charged runtime reserve.

## Relationship to the earlier maximum-context proof

The original P150 full-stack 50,623-token public prefill and final legal traced
decode remain important logical-context and layer-stack evidence:
`../optimized_full_model/final/full_stack_context_tp1/lifetime_fix_50624/`.
That run exposed and fixed a full-Q/concat lifetime fragmentation problem. It
does not record retained-trace prefill or a same-run peak-memory measurement.
TP2/TP4 retain their respective representative-layer boundary and full-stack
allocation evidence; this audit does not turn it into full-stack long-prefill
runtime proof.

The older P150 conservative headroom of 268,670,464 bytes is **not** used to
justify the new overlap. That projection predates the 692,060,160-byte BF16
embedding increment and serving's larger trace reservation, and explicitly
excluded serving-only allocations. The calculations above instead start from
the later observed BF16 construction and replace the cache/trace geometry.

Residual limitation: the numerical result is a conservative byte envelope with
an inherited allocator reserve, not a direct guarantee for every fragmentation
history or future program-cache growth pattern. Current cold/warmed request
regressions validate allocation tracking and replay safety; current serving
tests validate the recorded configuration. No maximum-context retained-trace
serving rerun or current allocator high-water measurement is claimed here.

Large server/watcher logs are stored losslessly as `.log.xz`; use `xz -dc` to
read the original text and line numbers. [Archive index](artifact_compression.json)
records original paths, byte counts and uncompressed SHA-256. Embedded runner
paths and historical line references name those original decompressed logs.

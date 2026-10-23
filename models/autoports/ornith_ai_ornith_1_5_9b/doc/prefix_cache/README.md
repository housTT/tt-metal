# Hybrid prefix-cache implementation candidate

This is an opt-in standalone bringup candidate. It is not enabled in the live
server, and the adapter still declares `supports_prefix_caching: False`.
Device copying, trace lifetime, capacity, accuracy, and served API validation
remain required before enabling it in a released profile.

Construct the full model with `prefix_cache_bytes=<per-device budget>` and
`prefix_cache_max_tokens=<largest stored prefix, default 4096>`. Both keywords
also pass through `build_generator`. Zero bytes retains the original path.

Each stored entry owns the matching packed attention KV, FP32 recurrent state,
and convolution history at one page boundary. The model restores these into a
new request's caller-owned physical KV pages and its prefill state, then computes
the remaining suffix. It does not reuse a live request slot or keep scheduler
page ownership after the request finishes. Packed KV remains in its cache dtype;
the TTNN paged-fill operation copies raw tiles rather than converting precision.

The cache saves one checkpoint per fresh prompt, at the largest complete page
before its final token, capped by `prefix_cache_max_tokens`. It chooses the
longest matching stored prefix and always computes at least one suffix token
for logits. A divergent prompt without a matching checkpoint computes normally.
This policy does not claim that every possible page boundary is cached.

An LRU bounds persistent snapshot payload per device. It evicts before allocating
a replacement and skips entries larger than the budget. Snapshot construction
also needs temporary buffers; physical peak memory still needs measurement.
Native-context allocation must be repeated with the selected cache budget.
Counters expose hits, misses, stored entries, evictions, oversize skips, and
the number of reused tokens. `model.clear_prefix_cache()` releases snapshots.
The owner must call it before closing the mesh; decode-cache resets intentionally
preserve independent prefix entries so repeated requests can reuse them.

Cache lookup and snapshot copying execute outside trace capture. The existing
whole-prompt B1 prefill trace is bypassed while this mode is enabled, because it
would otherwise bypass prefix lookup and the checkpoint boundary. New persistent snapshots invalidate the existing traces even on a program-cache
hit. Recapture records them as preceding allocations; snapshots are never marked
corruptible. A cache hit does not allocate another persistent snapshot. Canonical
traced serving first-token sampling and decode remain in the generator. Measure the
overall tradeoff before choosing profile settings; no speedup is claimed yet.

Nine host tests execute the actual model prefill orchestration around CPU KV
and recurrent-state boundaries. They cover restoration into different physical
pages, slot isolation, non-aligned suffixes, longer and divergent prefixes,
eviction/recompute, budget overflow, repeated use, and bypass of the prefill
trace. Their CPU storage oracle is not a device-codec validation.

Next gates:

1. Run actual TTNN snapshot/copy tests for packed KV and FP32/BF16 state on each
   mesh. Include non-contiguous page assignments and exact per-rank equality.
2. With one real layer of each kind, compare cold-cache and warm-cache logits,
   sampled outputs, and continued traced decode; exercise slot reuse and eviction.
   Enable native trace allocation tracking and Watcher. Then repeat full-model
   accuracy and memory/performance gates with the selected precision.
3. Wire the explicit serving prefix-cache option to this model-owned mode and
   expose its counters. The vLLM scheduler must continue allocating complete
   active KV tables: enabling its ordinary attention-only APC would skip tokens
   without restoring hybrid state and is incorrect for this design.
4. Validate real repeated/concurrent/cancelled requests, streaming/tool follow-up,
   prefix-cache hits and misses, and native context from the packaged image.

The alternative scheduler-managed hybrid/Mamba cache path requires broader TT
plugin allocation support: the current plugin accepts attention cache specs
only. It must not be enabled merely by setting an attention-only capability flag.

## Experimental serving probe

The serving adapter accepts `ORNITH_PREFIX_CACHE_BYTES` (nonnegative payload
budget per device, default0) and `ORNITH_PREFIX_CACHE_MAX_TOKENS` (default4096,
rounded down to complete64-token pages). Set a positive budget only for validation
until a profile has measured cache capacity and passed the device/API gates.
These allocations are additional to the active KV pool and transient copies;
the budget does not describe peak DRAM use.

Keep `--no-enable-prefix-caching` in vLLM: the scheduler must allocate the full
active prompt and submit it from position0. The adapter's model restores matching
KV and all recurrent state internally. Its `supports_prefix_caching=False` refers
to scheduler APC, not this opt-in model-owned mode. This initial integration does
not implement vLLM's scheduler cache-reset endpoint or cache-salt isolation; do
not advertise those options. Prefix counters are available through
`adapter.prefix_cache_stats()` and logged on teardown, including startup warmup.
Teardown releases generator traces before owned snapshots and is idempotent.

Host configuration/lifetime regressions pass; real server prefix hits,
non-aligned/divergent prompts, cancellation, concurrent slot reuse, eviction,
cache-on/off accuracy and peak memory remain required. No released profile enables
this experimental mode yet.

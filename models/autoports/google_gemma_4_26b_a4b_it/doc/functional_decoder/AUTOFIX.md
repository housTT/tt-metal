# AutoFix summary

All observed functional-decoder gates are resolved; AutoFix did not exhaust
its repair loop.

## Stage-review runtime fallback finding

The first independent review found one remaining host-backed `ttnn.full` in
the non-aligned prefill tail. It created each INT32 cache-update position inside
the runtime pass, while the lexical audit did not ban that known host-backed
TTNN API. The replacement uses `ttnn.moreh_full((1, 1), ..., int32,
ROW_MAJOR)`; the rank-2 shape satisfies the primitive and is one interleaved
update-index stick. The focused cache-integrity test passes. The audit now bans
literal `ttnn.full(` across the functional runtime call graph.

The same review challenged the bounded-decode trace claim. The wrap regression
now captures the complete `decode_forward` once, overwrites stable hidden,
RoPE, and current-position tensors, replays all 1104 positions through the
bounded 1024-token cache, and compares positions 1023/1024/1025/1103 with an
eager unbounded control at PCC 1.0. The existing batch-32 A/B/A mutable-buffer
suite was also run for both layer kinds and passed with independently permuted
page tables and PCC 1.0.

## Paged SDPA API mismatch

The full-attention real-weight prefill/decode tests initially raised a
`TypeError`: the current paged SDPA bindings no longer accept loose
`block_size` and `num_kv_heads` keywords. Source/API inspection showed that
paged fill/update still use loose view fields, while paged SDPA requires a
`PagedCacheGeometryOverride`. The smallest fix introduced
`_sdpa_cache_view_kwargs`, leaving write calls unchanged and giving every full
SDPA read the atomic `(128, 2)` geometry. Natural and shared-cache real-weight
tests then passed with prefill PCC 0.998457 and decode PCC 0.999860. A host
regression test locks the keyword and geometry contract.

## Advertised-context initialization hang

The 262144-context sliding decode test hung while allocating two 1 GiB caches.
Fresh-context AutoTriage proved it had not reached decoder compute: a host-
backed `ttnn.full` upload was stopped in fast dispatch with five missing NoC
read responses. `ttnn.moreh_full` is the true device initializer in this
checkout. Replacing only the two test initializers eliminated the host vector
and transfer. Both layer kinds then passed traced decode at position 262143,
and all four 262143/262144 real-weight prefill probes passed. Evidence and the
bounded device reset/recovery sequence are in `AUTOTRIAGE.md` and `triage/`.

## Tracy post-processing failure

The first sliding profiler workload passed, but report merging found a host op
without a device row. A separate xhigh AutoFix diagnosis compared raw and
summarized logs: finite per-RISC buffers had dropped 51 device IDs, so relaxing
the merger would have undercounted the layer. Draining the supported device
profiler between complete passes produced zero missing IDs for both layer
kinds and valid reports. See `AUTOFIX_TRACY_POSTPROCESS.md`.

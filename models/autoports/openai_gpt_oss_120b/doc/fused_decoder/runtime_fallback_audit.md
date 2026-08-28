# Fused runtime and data-movement audit

Scope: the final real-checkpoint `PERF_PREFILL` and `PERF_DECODE` signpost
windows for sliding layer 0 and full-attention layer 1, plus the reachable
`FusedDecoder` call tree. FullLocal and indexed decode were captured in
separate, exact-revision, path-asserted processes. Setup records were drained
before each warmed window.

## Result

| Layer kind | Path | Prefill ops / device sum | Traced decode ops / device sum | Host ops |
| --- | --- | ---: | ---: | ---: |
| sliding | FullLocal | 66 / 36.220 ms | 34 / 0.678 ms | 0 / 0 |
| sliding | indexed A/B | 66 / 36.396 ms | 67 / 1.312 ms | 0 / 0 |
| full | FullLocal | 66 / 35.558 ms | 34 / 0.689 ms | 0 / 0 |
| full | indexed A/B | 66 / 35.596 ms | 67 / 1.320 ms | 0 / 0 |

- No Torch, `ttnn.from_torch`, `ttnn.to_torch`, Python fallback, reshard, or
  fabric operation occurs in any warmed signpost window.
- The FullLocal runtime path is genuinely constructed: its trace contains
  `MoEComputeDeviceOperation` and
  `DeepseekMoEFastReduceNCFusedDeviceOperation`, and contains no indexed expert
  sparse matmul.
- The indexed comparison is like-for-like at the same checkpoint revision and
  input; its trace contains the two top-4 `SparseMatmulDeviceOperation` rows.
- FullLocal reduces traced decode from 67 to 34 device operations and, more
  importantly, lowers the measured sum by 48.32% sliding and 47.80% full.
  The separate 500-replay wall test confirms the win at 47.46% and 46.68%.

FullLocal's dominant decode row is `MoEComputeDeviceOperation` at 226.572 us
sliding and 238.477 us full. The fused score reducer is 6.347/6.435 us. The
indexed baseline instead spends 555.879/561.898 us in packed gate/up sparse
matmul and 214.524/216.377 us in down sparse matmul.

## Setup-only host work

The following work is intentionally outside warmed runtime:

- checkpoint tensor loading and setup-only gate/up/down packing;
- public `quantize_weights_via_host` conversion of calibrated FullLocal weights;
- TTNN weight, page-table, cache, rotary, and expert-mapping construction;
- first-seen token-count regroup/reduction buffers;
- test-only profiler construction drains.

The profiler helper processes eight experts at a time to prevent the setup
buffer overflowing. Every chunk now has a unique cache prefix, all model
dimensions are forwarded, and a host test proves 128 distinct expert cache
inputs in original order. This fixed a provenance bug in an earlier profiler
run that reused the first eight cached experts. All final profiler artifacts
were regenerated after the fix.

## Remaining layout operations

| Phase | Operation | Count | Sliding time | Full time | Required contract crossing |
| --- | --- | ---: | ---: | ---: | --- |
| prefill | `TilizeWithValPaddingDeviceOperation` | 2 | 31.031 us | 31.018 us | Integer routing cumsum/sort endpoints cross TILE and row-major contracts. |
| prefill | `UntilizeWithUnpaddingDeviceOperation` | 6 | 36.094 us | 36.050 us | Counts, offsets, indices, and scores feed row-major gather/scatter/embedding. |
| prefill | `UntilizeCodegenDeviceOperation` | 2 | 138.977 us | 140.276 us | H-wide embeddings and the dedicated routed-expert op require different fixed layouts. |
| decode | `InterleavedToShardedDeviceOperation` | 1 | 0.622 us | 0.597 us | Shared attention's dedicated decode concat-head contract requires height sharding. |
| decode | `UntilizeWithUnpaddingDeviceOperation` | 3 | 4.042 us | 4.091 us | Router indices/scores and FullLocal one-core sharded inputs require row-major form. |
| decode | `TilizeWithValPaddingDeviceOperation` | 1 | 8.113 us | 8.084 us | FullLocal emits expert-major row-major H=2880 slots; the fused reducer requires padded TILE input. |

There is no generic tilize/untilize or reshard row. Decode adapters total only
12.777 us sliding and 12.772 us full (under 1.9% of total device time). Direct
experiments showed that H=2880 cannot use the DeepSeek post-combine tilizer;
the generic tilize is therefore the available TTNN contract adapter. Attempts
to fold the routed index conversion into a one-core row-major sharded buffer
were rejected by the hardware tile-shape contract.

## Reachable call tree

`FusedDecoder._forward` reaches only:

1. dedicated RMSNorm;
2. shared GPT-OSS paged attention, already using packed QKV, dedicated head,
   RoPE, SDPA, and cache-update operations;
3. `_FusedMLP._run_chunk` in prefill, `_decode_full_local` for qualified decode,
   or `_decode_single_user` for indexed fallback;
4. device residual adds.

The source contains setup-time Torch construction but no `to_torch` call and
no import or call to `FunctionalDecoder`. Test host reads exist only outside
signposts to assert finite output, PCC, path identity, and determinism. The raw
Tracy CSVs, filtered reports, stacked summaries, and hashes under
`tracy/final_full_local/` provide the independent zero-host-op record.

# Gathered routed-expert prefill capture (ORNITH_MOE_GATHER=1)

Companion to `../linear_attention/prefill_perf_report.*`, which is the same window with the shipped
`ttnn.sparse_matmul` MoE. Same test, same seed, same signposts, same `tt-perf-report` invocation and
the same `--active-experts 41`, so the two per-op tables are directly comparable.

Captured 2026-08-21 on 4x Blackhole p300c, mesh (1, 4), after `tt-smi -r`:

    ORNITH_MOE_GATHER=1 python -m tracy -r -p -v --op-support-count 50000 -o <out> -m pytest \
      models/autoports/ornith_ai_ornith_1_0_35b/tests/test_multichip_decoder.py::"test_perf_prefill[blackhole-2048-linear_attention-mesh_device0-device_params0]"

    tt-perf-report <ops.csv> --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
      --active-experts 41 { --no-summary | --csv ... | --group-by op }

`--active-experts` is inert here: the gathered path emits no `sparse_matmul` rows. It is passed only so
the two invocations are identical.

Headline (single linear_attention layer, 2048-token chunk, four devices merged):

| | sparse (reference) | gathered | |
|---|---|---|---|
| total device time | 30,204 us | 33,174 us | +9.8% |
| expert compute | 22,640 us (SparseMatmul, 128 ops) | 10,815 us (UnifiedRoutedExpertFfn, 128 ops) | 2.09x faster |
| everything else | 7,564 us | 22,359 us | +14,795 us |
| wall clock, signpost to signpost | ~31 ms | 34.39 ms | |

Attribution, MoE-attributable device time (sparse 27,688 us of 30,204; gathered 31,227 us of 33,174):

| item | us | note |
|---|---|---|
| `UnifiedRoutedExpertFfn` | 10,815 | 128 launches, 84.5 us +/- 2.1 us each |
| `Embeddings` (row gather) | 10,456 | forward gather 9,935 of it: 65,536 rows x 2048 x bf16 per sub-chunk |
| `Typecast` + `Untilize` (y_buf bf8 TILE -> bf16 ROW_MAJOR) | 5,618 | format round trip over the same buffer |
| `Sort` (2 per sub-chunk) | 1,418 | forward sort 1,386 of it, on **2 cores** |
| `Accumulation` (cumsum, rank) | 1,056 | on **2 cores** |
| `Gather` (rank / score select) | 234 | |
| score multiply + reduce + slices | 1,629 | |

Two findings decide this:

1. **The FFN's time is ~85% fixed cost per launch.** 84.5 us +/- 2.1 us over 128 launches, a 14% spread,
   while expert token counts vary enough that a 2-tile expert should take ~2x a 1-tile one. At 84.5 us a
   launch achieves 2.4 TFLOP/s (1.3% of the ~177 TFLOP/s chip peak) and 21 GB/s of weight reads (~4% of
   DRAM). The op is neither compute- nor bandwidth-bound: it is grid-fill and launch bound, so the
   on-device `counts[e]` chunk skipping does not convert the 20.5x arithmetic saving into time at ~32
   rows per expert. `unified_routed_expert_moe` issuing one program per local expert is what costs this;
   ~9.2 ms of the 10.8 ms is not arithmetic and not weight traffic.

2. **The dispatch costs more than the FFN saves.** 18,783 us of new glue against 11,825 us saved.

Ceilings for the layer, against the sparse 30,204 us:

| scenario | layer | speedup |
|---|---|---|
| dispatch entirely free | 14,391 us | 2.10x |
| compacted regions (65,536 -> ~4,600 rows) | 18,248 us | 1.66x |
| ...and `sort`/`cumsum` off their 2 cores | 15,878 us | 1.90x |
| as measured today | 33,174 us | 0.91x |

As with the reference capture, the raw `ops_perf_results_*.csv` is NOT committed (6.8 MB, against the
repo's 500 KB pre-commit limit); the machine-readable per-op table for the exact window
(`prefill_perf_report.csv.gz`) and the two human tables are.

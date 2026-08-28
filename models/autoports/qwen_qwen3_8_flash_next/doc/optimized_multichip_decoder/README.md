# Qwen3.8-Flash-Next optimized multichip decoder

This stage optimizes the completed `Qwen/Qwen3.8-Flash-Next` decoder in
place. The measured and selected path is the exact host-backed decoder on P300
Blackhole dies 0 and 1 as a fixed `1x2` `FABRIC_1D` mesh. It is not a
single-chip or replicated fallback, and it does not include full-model or vLLM
work.

## Final default

- Routed MoE remains gate-selected top-10 execution. Experts use EP2 owner
  `expert_id % 2`, an exact full-K BFP4 owner shard, and one shared exact-zero
  peer shard. Dense all-512-expert execution is never used.
- Each layer has ten persistent device expert slots plus one fixed upload
  staging pair per rank. Decode uses a generation-checked LRU and a persistent
  replicated `uint16` slot-index row, so router-order changes do not reload
  expert weights that are already resident.
- The packed host cache holds all 512 experts per layer when populated. One
  packed zero shard is shared. This removes checkpoint read/packing from warm
  prefill and most decode hits without changing exact miss semantics.
- PLE uses safetensors mmap handles, exact EOS-aware n-gram hashes,
  `torch.unique` request deduplication, per-shard batched `index_select`, an
  8,192-row LRU, persistent TT prefill/decode staging, and 128-token internal
  waves. Public non-aligned lengths are padded, masked, and sliced internally.
- Row-parallel QSA output and shared-expert down partials are BF16. QSA decode
  uses selected 1D width sharding `qsa_input:110,attn_out:20`. The existing
  BFP8 paged KV/index cache contract and 262,144-token context remain intact.
- The selected P300 fabric contract is `FABRIC_1D`, two collective links, and
  an 8,192-byte maximum packet payload. `final_fabric_contract.xml` reads the
  live payload value and asserts the decoder default.
- Prefill uses the direct path. Decode uses two fixed-address trace segments
  around the declared route-id D2H and exact expert/PLE host service.

The full-model stack ABI must preserve residual layout `S`:

```text
S = mesh-sharded BF16 [1, 1, 4*M, 1280]
    shard dimension = 3 across the 1x2 mesh
    local storage = DRAM interleaved
```

Every decoder consumes and returns `S`. Stack ingress performs one
`mesh_partition` from public replicated `[1,1,M,10240]`; stack exit performs
one `all_gather` and reshape. There is no gather, reshard, or all-reduce
between decoder layers. A replicated consumer was 5.37% faster in isolation
(0.221804 versus 0.233722 ms), but it was rejected because restoring a
collective at every layer boundary is slower as a stack and violates this
contract. This ABI is also recorded in `doc/context_contract.json`.

## Correctness and latency

All values below are medians of seven function-scoped two-device runs, 100
decode replays per run, real checkpoint weights and activations, logical
prefill length 33, and exact host service. PCC is identical before and after.

| Layer kind | Prefill PCC | Decode PCC | Warm prefill before / after | Speedup | Segmented decode before / after | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GDN layer 0 | 0.99942303 | 0.99996978 | 2065.630386 / 158.877355 ms | 13.001x | 3.427797 / 2.082220 ms | 1.646x |
| PLE+GDN layer 1 | 0.99949104 | 0.99984211 | 1526.015021 / 120.853601 ms | 12.627x | 4.118693 / 2.613159 ms | 1.576x |
| QSA layer 3 | 0.99972457 | 0.99988294 | 1969.037359 / 162.370719 ms | 12.127x | 3.059096 / 3.027236 ms | 1.011x |

The exact source data is in `baseline_perf_medians.csv`,
`final_default_perf_medians.csv`, and `before_after.csv`. Final claims come
from `final_default_perf_count7.xml`, which exercises the final defaults, not
an earlier candidate configuration.

The exact host-backed path still includes service that a resident local layer
does not: its corresponding optimized resident TT baselines are 17.57–43.73
ms prefill and 1.11–2.86 ms traced decode. Those numbers are controls, not the
optimization-stage before values.

## Host service accounting

Every measured window reports expert requests/waves/hits/misses/evictions,
packed host hits/misses, checkpoint reads/bytes/time/bandwidth, source packing,
expert and index H2D bytes/time/bandwidth, PLE selected/unique/table rows,
PLE bytes/lookup/H2D, overlap, unattributed stall, and end-to-end time.

A representative final PLE+GDN warm prefill window is 120.853601 ms end to
end: 12 expert waves, 114 device misses served entirely from 114 packed-host
hits, 630,374,400 physical expert H2D bytes in 98.814318 ms (6.379 GB/s), 528
selected/528 unique exact PLE rows, zero table reads, 337,920 logical PLE bytes
in a 1,310,720-byte padded TT upload, 0.371790 ms lookup, and 0.227715 ms PLE
H2D. The boundary is serialized, so declared overlap is 0. Prefill `stall_ms`
is the non-negative accounting remainder and is zero by construction when the
recorded suboperations fill the service window; it is not an independently
timed stall measurement.

For the associated 100-token decode window, the harness reports 976 expert
hits/24 misses, 23 packed hits/one exact checkpoint miss, 9,830,400 checkpoint
bytes, 132,710,400 expert H2D bytes at 6.393 GB/s, 1,440 index bytes, 1,600
selected/1,600 unique PLE rows, zero table reads, 1,024,000 PLE H2D bytes,
3.362758 ms lookup, 8.634629 ms PLE H2D, and 27.768330 ms unattributed host
stall across the whole window. Median checkpoint read/packing is
1.723209/8.035640 ms, expert H2D is 20.758524 ms at 6.393 GB/s, and the
100-token end-to-end window is 261.315892 ms. Segmented decode is
2.613159 ms/token.

The 129-token PLE locality sweep is in `ple_row_cache_sweep.csv`. With 2,064
distinct exact rows, capacity 256 rereads 1,808 rows on the second lookup;
capacity 8,192 rereads zero and reduces warm lookup from 2.197265 to 0.541153
ms. Repeated outputs are bit-exact. Torch pinned allocation is unavailable in
the installed CPU-only build; `host_pinning_probe.txt` records the exact
runtime error. TT runtime staging remains the truthful selected boundary.

## Optimization evidence

`optimization_matrix.csv` is the compact candidate ledger. Material results:

- Stable indexed LRU, packed capacity 512, and one shared zero shard are
  selected. Capacities 12 and 20 were adapted and rerun; larger device banks
  reduce some misses but slow the traced back segment, and capacity 12 also
  misses the layer-1 accepted PCC.
- Threaded rank H2D was numerically correct but no faster. The final path uses
  explicit serial rank uploads and a device fence; measured overlap remains
  zero rather than claiming unsafe concurrency.
- One versus two CCL links, 4,352 versus 8,192-byte fabric packets, and their
  combination were measured for all layer kinds. The initially mixed one-shot
  result was promoted to a seven-repeat complete-decoder comparison; two links
  plus 8,192-byte payload won every representative median and is the final
  default. A second no-override count-seven run reproduced accepted PCC and
  the selected result in `final_default_perf_count7.xml`.
- Async collectives were ported to model shapes with preallocated semaphores,
  then retuned. Layer-0 prefill/decode were slower than the synchronous
  control, so sync CCL stays selected. The exact recovered stdout and
  transcript-record hashes are in
  `candidate_async_ccl_recovered_provenance.txt`.
- Fused matmul-reduce-scatter required AutoFix. It was retried with Linear
  topology dispatch, matmul-N-derived output shape, the doubled Linear
  persistent intermediate, one worker, adapted 8x1 placement, and the missing
  post-signaler atomic flush. It then produced PCC 1.00000012/1.00000119, but
  was 1.736x slower at decode and 1.643x slower at M=32 prefill, and fabric
  ERISC teardown remained unhealthy. The temporary core changes were removed.
- QSA decode 1D activation sharding is selected. Prefill 2D sharding is
  slower. DRAM-sharded QSA input/output was retried at HiFi2 and HiFi4; both
  miss decode PCC. This is direct evidence against the profiler suggestion for
  these exact shapes, not an API-only dismissal.
- BF16 row-parallel partials are selected. FP32 CCL payload is slower and
  causes a QSA decode PCC miss. GDN BFP8, QSA BFP8, QSA input/output BFP4, and
  shared-projection BFP4 were all tested on real activations; each either
  changes the accepted baseline PCC or fails the 0.995 gate.
- Packed projections remain in the selected graph: GDN packed QKV, QSA packed
  Q/K/V/index projection, hyper down+injection, shared gate+up, router+shared
  input, and expert gate+up. Sparse expert matmuls retain active top-10
  execution. No split alternative removed enough movement to justify
  unpacking these established weights.
- Persistent resources include expert slots, upload staging, route-index row,
  PLE staging, KV/index/recurrent state, shared GDN L1 workspace, and the two
  trace segments. The fused persistent CCL candidate is rejected by measured
  latency and teardown health, not deferred.

No applicable optimization is deferred. The full candidate values and exact
artifact mapping are in `optimization_matrix.csv`; the chronological commands
and adapted retries are in `work_log.md`.

## Profiler, fallback, stress, and health gates

Fresh final-default Tracy captures cover prefill and decode for all three
layer kinds. `tt-perf-report` accepted all six signpost ranges with no profiler
buffer drops. Detailed CSVs, summary CSVs/PNGs, complete human tables, advice,
raw reports, and hashes are under `tracy_final/`; `profiler_provenance.txt` records
the exact commands. Raw inputs are retained losslessly as `.csv.xz` with both
content and packed hashes. `profiler_operation_audit.csv` records repeated
matmul, active-sparse matmul, collective, copy, concat, reshape, slice,
permute, tilize/untilize, fill-pad, and mesh-partition counts from every phase.
For example, final layer-1 decode contains 14 dense and two active-sparse
matmuls, nine all-gathers, one reduce-scatter, and 51 copies; these are
intra-layer operations, while the layer boundary stays fractured.
`profiler_family_summary.csv` shows that layout/TM is
28.88–60.78% of merged decode device time, dense matmuls 12.37–25.61%, and
collectives 4.80–8.07%. In prefill, layout/TM is 27.70–47.09% and active
sparse matmuls are 11.57–20.46%.

The final static gate is 32/32 passes and includes the host boundary whitelist,
no-host-fallback collective audit, capacity arithmetic, exact expert packing,
PLE semantics, and logical lengths through 262,144. The final watcher gate is
4/4 passes: 100 changing tokens for GDN, PLE+GDN, and paged QSA, plus two live
segmented traces sharing the GDN workspace. The 2,027-line retained log has no
watcher/NoC/assert/panic/hang signature. A post-watcher source-backed `1x2`
mesh open/close passes.

Watcher uses `TT_METAL_WATCHER_DISABLE_ETH=1`, the inherited P300 control for
the known instrumentation-only fabric teardown issue; Tensix/NoC/CB/stack
watching remains active. Profiler and watcher are never enabled together.

## Context and limitations

There is no context reduction. `doc/context_contract.json` still declares
262,144 tokens, BFP8 paged QSA KV/index cache, and the same per-device state
budget. QSA activation sharding is temporary intra-layer state and the packed
host cache consumes host RAM, so neither reduces maximum context. Exact
capacity remains physically backed with 24,122,217,472 bytes planned TT DRAM
headroom per device.

Known limitations are explicit:

- exact host-backed decode is batch one; the resident decoder retains its
  previously validated batch-32 contract;
- host expert/PLE service is serialized and overlap is zero after the parallel
  candidate failed to win;
- Torch pinned host memory is unavailable on this machine;
- standalone stack ingress/exit and necessary intra-layer collectives still
  emit deprecated CCL-argument/L1-small advice, but there is no inter-layer
  collective;
- full-model assembly and vLLM are deliberately outside this stage.

The independent `$stage-review` verdict is recorded in `STAGE_REVIEW.md`.

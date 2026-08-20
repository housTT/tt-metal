# Qwen3.6-27B multichip decoder

## Target hardware and selected topology

The target is the four-device Blackhole P300c ring visible on this host.  The
selected model topology is a `1 x 4` mesh with `FABRIC_1D_RING`,
`ttnn.Topology.Ring`, and tensor parallel degree four.  A mesh open/close and a
BF16 ring all-gather both passed before implementation work began.

Qwen3.6-27B is dense; there is no MoE router or expert path.  The model has 64
layers: 48 recurrent linear-attention layers and 16 full-attention layers.  TP=4
is head aligned for both layer kinds.  Linear-attention MLP shards divide
directly.  Full-attention MLP shards are padded independently from 4,352 to
4,608 so the decode program can use 16 cores without overlapping live L1 and
static circular buffers.  Padding is algebraic zero and remains internal.

| Quantity | Global | Per device at TP=4 | Padding / ownership |
|---|---:|---:|---|
| Residual hidden | 5,120 | 1,280 when fractured | no padding |
| Linear-layer MLP intermediate | 17,408 | 4,352 | no padding; 8 decode cores |
| Full-layer MLP intermediate | 17,408 | 4,352 logical / 4,608 physical | 256 zero rows/columns per device; 16 decode cores |
| Full Q heads | 24 x 256 | 6 x 256 = 1,536 | contiguous head groups |
| Full KV heads | 4 x 256 | 1 x 256 | one KV head and cache shard |
| Linear key heads | 16 x 128 | 4 x 128 = 512 | each local head repeats 3x |
| Linear value heads | 48 x 128 | 12 x 128 = 1,536 | local recurrent-state owner |
| Linear convolution width | 10,240 | 2,560 | Q/K/V head aligned |
| Linear beta / decay rows | 48 | 12 logical, 32 tile-local | 20 inert rows/device |

### Tensor and state strategy

| Tensor / operation | Mesh placement | Local shape or contract | Collective |
|---|---|---|---|
| Input/post-attention RMSNorm weights | replicated BF16 | `[1,1,1,5120]` | local norm for replicated stream; distributed stats when fractured |
| Full packed Q/K/V and gate projections | column parallel | K=5120, local Q/K/V/gate widths 1536/256/256/1536 | none after projection |
| Full Q/K norm and RoPE | local heads | Q `[B,6,S,256]`, K `[B,1,S,256]` | none |
| Full paged K/V cache | KV-head sharded | each tensor `[blocks,1,64,256]`, BFP8 | page table and positions replicated; no cache collective |
| Full SDPA | local Q and KV heads | 6 Q heads consume the device's one KV head | none |
| Full O projection | row parallel | local K=1536, output partial `[B,1,S,5120]` | reduce family at residual boundary |
| Linear input projections | column/head parallel | local Q/K/V/Z widths 512/512/1536/1536; beta/a 12 logical | none |
| Linear convolution and GDN recurrence | local head ownership | conv state `[B,1,4,2560]`; recurrent `[B,12,128,128]` FP32 | none |
| Linear output projection | row parallel | local K=1536, output partial `[B,1,S,5120]` | reduce family at residual boundary |
| Linear MLP gate/up, down | column / row parallel | `[5120,4352]`, `[4352,5120]` local | one reduce at residual boundary |
| Full MLP gate/up, down | column / row parallel | `[5120,4608]`, `[4608,5120]`; 4,352 logical | one reduce at residual boundary |

The public prefill/decode compatibility methods retain logical lengths, page tables, current
positions, and caller-owned state.  Tile, page, chunk, and trace alignment stay
internal.  Prefill keeps the replicated contract because it is dominated by
compute and already supports chunked native context. Real-layer A/B selects
the replicated `[B,1,S,5120]` TILE/DRAM input/output for decode stacks as well.
`stacked_decode_forward` retains a validated TP-fractured `[B,1,S,1280]`
alternative, but it is not the production contract on this mesh.

### Collective/residual candidates

BF16 payload sizes below are for one logical decode row.  Prefill scales the
same widths by the physical chunk length while preserving the logical slice.

| Family | Residual before -> after | Next consumer | Nominal payload per device/boundary | Persistent plan | Disposition |
|---|---|---|---:|---|---|
| Local row matmul + all-reduce | replicated -> replicated | local RMSNorm | 10,240 B partial | TTNN lowers to ring reduce-scatter + all-gather | **selected decode stack** |
| Row matmul + reduce-scatter, delayed gather | fractured -> 1,280-wide fractured | distributed RMSNorm / gathered normalized next projection | 2,560 B result shard | DRAM result shards | correctness-valid, real-layer slower |
| Reduce-scatter + immediate all-gather | replicated -> replicated | local RMSNorm | RS 2,560 B + AG 10,240 B | ping-pong buffers | compatibility control only |
| Fused all-gather + local-output matmul | fractured -> fractured/local output | residual add/norm | gathered input 10,240 B | fused op output buffer | adapt and measure where weight packing permits |
| Fused matmul + reduce-scatter | replicated -> fractured | distributed residual/norm | 2,560 B result shard | persistent output if API supports | test installed Blackhole op contract |

The compatibility all-reduce is visible in the original profile as one
`ReduceScatterDeviceOperation` plus one `AllGatherDeviceOperation` at each of
the two algebraic boundaries.  The later shape-faithful, traced topology probe
adapted the next consumer: reduce-scatter, fractured residual add, distributed
RMSNorm, all-gather of normalized activations, and the next column projection.
At K=1,536 it measured 257.913 us versus 282.176 us replicated (8.6% faster);
at padded K=4,608 it measured 337.104 us versus 361.541 us (6.8% faster), with
four-device PCC 0.999815. The family was implemented and its two real
layer-kind trace tests pass with repeat PCC 1.0. The decisive real-weight
whole-layer A/B reversed the micrograph result: linear measured 772.178 us
replicated versus 871.299 us fractured, and full measured 551.352 us versus
652.115 us. Distributed norms outweigh the saved all-gathers, so replicated is
selected.
The installed minimal fused matmul/reduce-scatter and all-gather/matmul APIs
were also executed on this exact 1x4 Blackhole ring.  The fused RS probe passed
all devices at PCC 0.999987; the shape-relevant fused AG-matmul probe
`M=3072,K=5120,N=3456` passed all devices at PCC 0.9999915 / relative RMSE
0.0050 using two links and four workers/link.  These APIs require persistent
buffers, global semaphores, and MinimalMatmul layouts not produced by the
optimized DRAM-sharded decoder programs. The exact program, semaphore,
persistent-buffer, and precision blockers plus decoder-shape candidate configs
are recorded in `autofix/fused_decoder/AUTOFIX.md`.

### Actual Blackhole shard and program contracts

The P300c exposes eight DRAM banks and an 11x10 worker grid.  All decode weights
are DRAM width-sharded over `(0,0)..(7,0)`.  Runtime outputs use row-major Tensix
grids: 8 cores `(0,0)..(7,0)`, 16 cores `(0,0)..(10,0)+(0,1)..(4,1)`, or the
rectangular RMSNorm grids stated below.

| Program | DRAM weight shard `[K,N/8]` | Tensix output shard | Program `(cores,in0_block_w,per_core_M,per_core_N)` |
|---|---|---|---|
| Linear input projection | `[5120,544]` (4,352 physical; 4,160 logical) | `[32,544]`, 8 cores | `(8,20,1,17)` |
| Linear/full output projection | `[1536,640]` | `[32,320]`, 16 cores | `(16,3,1,10)` |
| Full packed QKV/gate | `[5120,448]` | `[32,224]`, 16 cores | `(16,10,1,7)` |
| Linear MLP gate/up | `[5120,544]` | `[32,544]`, 8 cores | `(8,10,1,17)` |
| Linear MLP down | `[4352,640]` | `[32,640]`, 8 cores | `(8,17,1,20)` |
| Full MLP gate/up | `[5120,576]` | `[32,288]`, 16 cores | `(16,10,1,9)` |
| Full MLP down | `[4608,640]` | `[32,320]`, 16 cores | `(16,9,1,10)` |

Decode input RMSNorm uses 32 cores in an 8x4 rectangle with shard `[32,160]`.
Post-attention RMSNorm uses 8x1 / `[32,640]` for linear layers and 8x2 /
`[32,320]` for full layers.  Fractured norm weights and residuals are 1,280
wide/device; the gathered normalized tensor is converted back to the relevant
8- or 16-core L1 width-sharded contract before the MLP.

### Rejected model-level alternatives

| Alternative | Reason rejected before final implementation |
|---|---|
| Data parallel x4 | replicates roughly 27B parameters and does not reduce single-user decoder latency or per-device weight bandwidth. |
| TP=2 plus DP=2 | leaves half the available aggregate DRAM bandwidth unused for one user and duplicates weights/caches. |
| Flattened 2D TP | the physical four-chip system is a 1D ring and all attention/GDN head counts divide by four; a 2D decomposition adds a second collective axis without a weight- or head-divisibility benefit. |
| Expert parallelism | Qwen3.6-27B is dense and has no router or experts. |
| Sequence parallel attention/cache | adds token exchange while KV-head TP already gives one full-attention KV head per device and linear recurrence is independent by value head. |

## Context and memory plan

The HF-advertised 262,144-token context is preserved.  For the 16
full-attention layers, BFP8 paged K/V storage at TP=4 is 136 MiB/device/layer,
including the 1,088-byte BFP8 tile overhead, or 2.125 GiB/device for the stack.
Batch-32 linear state is 29 MiB/device/layer: 24 MiB recurrent FP32 plus 5 MiB
for the BF16 conv state whose tiled height pads from 4 to 32.  That is 1.359 GiB
across 48 layers; batch-1 is 43.5 MiB across those layers.  The calculated
64-layer decoder-stack weight
storage is 9.347 GiB/device: 48 linear layers at 133.656 MiB plus 16 full layers
at 197.219 MiB.  This includes interleaved and DRAM-sharded duplicate layouts
and tile-format overhead (BFP4/BFP8/BF16 tiles are 576/1,088/2,048 bytes).
Adding max-context KV cache, batch-32 state, and a conservative 4 GiB
trace/activation/CCL/fragmentation reserve totals 16.831 GiB/device.  This
leaves 15.2 GiB for full-model components outside this decoder stage on each
32 GiB device.  No capability reduction is required.

## Acceptance results

All tests use real checkpoint weights.  A separate two-phase gate writes
single-chip `OptimizedDecoder` tensors and compares TP=4 directly using the
same seeds, weights, page tables, positions, cache/state transitions, and trace
calls.  Every replicated compatibility output is also inspected separately on
all four devices for shape, TILE/BF16/DRAM placement, HF PCC, and replica PCC.
Page tables and current positions are replicated, while cache
and recurrent state remain local by head.

| Gate | Result |
|---|---|
| Linear layer, logical length 65 | prefill PCC 0.998115; decode 0.997494; traced decode 0.998360; repeat 1.0 |
| Full layer, logical length 33 | prefill PCC 0.998384; decode 0.999116 |
| Full paged warmed trace | PCC 0.998096; repeated replay 1.0 |
| Direct optimized baseline, linear | prefill 0.999504; decode 0.999411; traced decode 0.999820 |
| Direct optimized baseline, full | prefill 0.997760; decode 0.999443; traced decode 0.999568 |
| All-device replicated boundary | every device has TILE/BF16/DRAM `[B,1,S,5120]`; replica PCC 1.0 in prefill, decode, and trace |
| Fractured decode stack | both layer kinds pass traced HF comparison; linear 0.998354, full 0.998317, repeat 1.0 |
| Forced chunked full prefill | PCC 0.997940 |
| Batch 32 | both layer kinds passed; per-user decode PCC about 0.9988--0.9993 |
| Local contracts | full Q/KV heads 6/1, cache `[blocks,1,64,256]`; linear state `[B,12,128,128]`, conv `[B,1,4,2560]` |
| Advertised context | 32,769 and 262,144 full prefill passed; full position 262,143 trace PCC 0.999836/repeat 1.0; linear 262,144 prefill plus final traced decode passed |
| Runtime audit | no host conversion, layout fallback, reshard, or Torch call in multichip runtime methods |
| Watcher | 4 compatibility tests plus both fractured stack traces passed with `TT_METAL_WATCHER_DISABLE_ETH=1`; the final 10,304-line fractured log is clean; the unscoped attempt hit the documented ACTIVE_ETH instrumentation-size limit |

### Warmed decode performance

The same real-weight harness, runtime, batch-32 physical decode row, and 20
trace replays were used.  Efficiency is `speedup / 4`.

| Layer kind | Single chip | 1x4 TP | Speedup | Efficiency |
|---|---:|---:|---:|---:|
| Linear attention | 1,456.214 us | 772.178 us | 1.886x | 47.1% |
| Full attention | 1,133.241 us | 551.352 us | 2.055x | 51.4% |

Advice-enabled `tt-perf-report` artifacts are under `profiler/`.  The corrected
`decode_ops_blackhole.csv` files explicitly stamp `DEVICE ARCH=blackhole` and
`AVAILABLE WORKER CORE COUNT=110`; the full-layer `report_blackhole.*` and
linear-layer `selected_bw10_report_blackhole.*` are the authoritative
human-readable/CSV reports. Full decode's modeled overall DRAM
roofline is 39.6% (203 GB/s); linear decode is 22.1% (113 GB/s).  The full
decode report contains 54 device ops / 479 us of kernels and 88 us of gaps.
Its five dominant BFP8 matmuls are marked optimized; paged BFP8 cache updates
are 4 us each.  The two collectives are 27+19 us and 27+17 us.  The linear
report contains 82 device ops / 588 us of kernels. Its BFP4 MLP rows are about
43 us each and remain labeled `SLOW`. `autofix/bfp4_geometry/AUTOFIX.md` closes
that label with all eight legal exact-shape block candidates plus the only
useful 16-worker padded candidate; 10/10/17 is the measured winner. The two
collective pairs are 29+18 us and 28+17 us.
The original raw CSV omitted architecture metadata; those original reports are
retained only as provenance.  The corrected reports use the tool's Blackhole
constants: 512 GB/s DRAM and 110 workers, with LoFi peak computed as
`4096 * 1.35 / 1000 = 5.5296 TFLOP/s/core`.  Multi-device rows are merged by
logical operation for wall-time reporting; raw per-device rows remain in the
corrected input CSVs.

## Limitations

Only the exact 1x4 P300c mesh is supported. Prefill, standalone decode, and
production decode stacks use replicated outputs because the real-layer trace
A/B rejects the lower-payload fractured alternative on this runtime/hardware.
The linear
native-context gate is intentionally expensive (709.8 s) because it enqueues
4,096 chunks twice; capability remains 262,144.  No MoE strategy applies: this
checkpoint is dense.

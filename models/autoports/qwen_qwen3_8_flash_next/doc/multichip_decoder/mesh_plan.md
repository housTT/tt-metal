# Qwen3.8-Flash-Next multichip decoder plan

The target was locked before implementation on 2026-08-27 and was not
changed: one P300 board, its Blackhole dies 0 and 1 as a fixed `1x2` mesh,
and `ttnn.FabricConfig.FABRIC_1D`. Each die has an `11x10`
compute-with-storage grid, eight DRAM channels, and 34,225,520,640 bytes of
nominal DRAM. The correctness/performance baseline is the unchanged
`OptimizedDecoder` graph replicated on the same two-die mesh without CCL.

## Delivered ABI and topology

`MultichipDecoder` subclasses `OptimizedDecoder`. The public input/output ABI
remains replicated BF16 `[1,1,logical_sequence_or_batch,10240]`. At stack
ingress it reshapes the four hyperconnection streams and partitions hidden
width into the persistent internal S layout `[1,1,4*logical_rows,1280]`.
Layers exchange S directly; only an explicit stack/test exit all-gathers back
to the public ABI.

Logical lengths are never replaced by aligned lengths. The 128-token chunk
planner pads only internal work and trims every result. Structural tests cover
`1,31,32,33,63,64,65,127,128,129,2047,2048,2049,262143,262144`; real
prefill/PCC uses non-aligned length 33. Decode covers batch 1 and 32 with
device INT32 positions, and QSA batch 32 uses distinct page-table rows.

The delivered split is:

- persistent fractured residual and distributed hyperconnection;
- replicated GDN recurrence/state with column-parallel output projection;
- TP2 QSA query/KV heads and row-parallel output reduced into S;
- TP2 shared expert at intermediate width 320/die;
- deterministic EP2 routed experts, full intermediate width 640 on the owner;
- exact-zero full-shaped routed slot on the non-owner;
- replicated router input/logits and indexer;
- exact bounded route/expert/PLE host boundary outside trace capture.

## Tensor and activation placement

Checkpoint shapes below use `[output,input]`. `R` is replicated, `S` is the
within-stream residual shard, `TP` is a distinct tensor-parallel shard, and
`EP-owner` means full content on `expert_id % 2` with exact zeros on the peer.

| Role | Global logical shape | Placement | Per-die logical shape/result |
| --- | --- | --- | --- |
| stack residual | `[...,4,2560]` | S hidden-width | `[...,4,1280]` |
| HC norm | four local 1280 shards | distributed | RMS stats all-gathered |
| HC down+inject | `[10240,324]` | column-sharded | local partial into S |
| HC up | `[320,10240]` | row-sharded | local S output |
| router | `[2560,512]` | R input/logits | identical top-k |
| shared gate/up | `[640,2560]` each | TP output | `[320,2560]` |
| shared down | `[2560,640]` | TP input | `[2560,320]` partial |
| routed packed gate+up | `[512,2560,1280]` | EP-owner | selected `[2560,1280]` or zero |
| routed down | `[512,640,2560]` runtime orientation | EP-owner | selected `[640,2560]` or zero |
| PLE key/value | `[160,12800]` | R | unchanged |
| GDN qkv | `[10240,2560]` | R | 16 key/48 value heads |
| GDN z | `[6144,2560]` | R | 48 value heads |
| GDN b/a | `[48,2560]` each | R | 48 value heads |
| GDN conv | `[10240,1,4]` | R | replicated state |
| GDN out | `[2560,6144]` | column-parallel | `[1280,6144]` into S |
| QSA q+gate | `[12288,2560]` | TP head | `[6144,2560]` |
| QSA k/v | `[512,2560]` each | TP head | `[256,2560]` |
| QSA out | `[2560,6144]` | TP input | `[2560,3072]` partial |
| index q/k | `[512,2560]` / `[128,2560]` | R | identical virtual-token selection |

Every die can address all 512 routed experts through ten fixed slots. Only
gate-selected experts execute (`top_k=10` per logical row). Prefill unions
larger than ten are partitioned into deterministic waves. A sparse kernel tile
may contain more expert IDs because 32 logical rows share a routing tile; the
logical top-10 contract is checked separately.

## Why routed experts are EP2, not split-K TP2

The routed expert is numerically sensitive at its down projection. The
original TP2 form computed K=640 as two K=320 BFP4 sparse matmuls and reduced
them. An isolated same-input diagnostic showed:

- HC, GDN, router/top-k, checkpoint slices, and gate/up reconstruction were
  effectively exact;
- the K=320+320 down partial sum was about 0.926 PCC;
- routed output was about 0.938 PCC and the final layer-0 result fell below
  the 0.995 gate;
- HiFi2, FP32 down output, fused-precision changes, and baseline routes did
  not fix the boundary.

The accepted host-backed form keeps the full K=640 on one deterministic owner
and exact zeros on the peer. The existing MoE collective then sums one routed
owner plus both shared-expert TP partials. Full K also requires the g40 sparse
program; full K with the old g20 program still failed at 0.99064916, while
g40 passed at 0.99942303. Owner parity naturally balances the 512-expert
universe; a particular top-10 token may be imbalanced, which is accepted for
exactness.

## State, paging, and cache placement

- GDN recurrent state is replicated FP32 `[batch,48,128,128]`; convolution
  state covers the complete q/k/value packing.
- QSA K and V caches are local BFP8 `[max_blocks,1,64,256]`; global main KV
  head 0 is on rank 0 and head 1 on rank 1.
- The raw index cache is replicated BFP8 `[max_blocks,1,64,128]`.
- The compressed index-key cache is replicated BF16
  `[max_blocks,1,16,128]`.
- Page tables, rotary tables, and INT32 positions are replicated. Addresses
  use logical positions, never activation padding.

At batch 1 and context 262,144, `max_blocks=4096`. Across the twelve QSA
layers, tile-padded maximum-context cache storage is exactly 2,340,421,632
bytes/die. The maximum-position trace addresses position 262,143, its shuffled
last page, and the production cache update/gather graph.

Batch-one GDN/PLE persistent state is canonical in DRAM. Exact physical TILE
storage is:

- per GDN layer: recurrent `48*128*128*4 = 3,145,728 B` plus three FP32 taps
  `3*32*10240*4 = 3,932,160 B`;
- 36 GDN layers plus nine BF16 PLE taps: 260,702,208 B/die;
- additional live prefill/user state: 208,928,768 B/die.

All GDN layers share one stable L1 workspace in model order. Its GDN portion
is 65,536 bytes/worker and its PLE portion 55,296 bytes/worker, for a peak of
120,832 bytes/worker. Actual/bank-reserved mesh footprints are
12,976,128/13,291,520 bytes/die.

## Collective plan

The persistent residual remains local. QSA and MoE row-parallel results use
`reduce_scatter` to sum partials and retain the destination residual shard.
Distributed norm statistics and explicit stack/test exit use `all_gather`.
The fixed collective settings are mesh axis 1, linear topology, one link, and
DRAM results. GDN recurrence has no collective; its output projection writes
the local residual width directly.

All acceptance runs verify rank/local layout, direct stack flow, and output
PCC. Profiler evidence keeps the communication rows visible rather than
subtracting them from the measured windows.

## Full-stack residency boundary

Ordinary resident TP2 routed/shared experts have 58,982,400 logical tiles per
die over 48 layers. BFP4 storage alone is 33,973,862,400 bytes/die, so it
cannot coexist with the caches, non-expert graph, endpoints, state, and
runtime reserve. Reducing context does not solve the primary expert-capacity
problem.

The exact host-backed EP2 resolution is:

| Resource | Per die/rank | Full-stack bytes/die |
| --- | ---: | ---: |
| fixed routed slots | 10 full experts/layer | 1,327,104,000 |
| fixed upload staging | 1 full expert/layer | 132,710,400 |
| PLE staging | prefill + decode | 819,200 |
| canonical decode state | 36 GDN + PLE taps | 260,702,208 |
| additional prefill/user state | live combined state | 208,928,768 |
| max-context QSA caches | local/replicated as above | 2,340,421,632 |
| decoder non-expert weights | final fractured/replicated placement | 3,479,858,176 |
| full-text endpoints | natural TP2 placement | 1,279,016,960 |
| runtime/trace reserve | fixed allowance | 1,073,741,824 |
| planned total |  | **10,103,303,168** |
| planned headroom |  | **24,122,217,472** |

Each rank-local expert slot is 2,764,800 bytes in BFP4. Each miss reads one
9,830,400-byte BF16 checkpoint expert, packs full-shaped owner/zero pairs, and
transfers 5,529,600 bytes across both ranks. A cold top-10 decode therefore
transfers 55,296,000 bytes; a hit transfers zero. A generation is published
only after both projections on both ranks upload successfully.

The PLE table has 320,001,446 logical rows padded to 320,001,536, row width
160 BF16, and total size 102,400,491,520 bytes. Sixteen exact EOS-aware n-gram
rows assemble each 2560-wide embedding; only selected rows enter stable TT
staging.

## Trace allocation strategy

Each host-backed token uses a front TT trace through PLE/attention/router and
state, exact host expert service, and a back TT trace through selected
experts/MoE/collectives. GDN state hydrates and commits through the shared L1
workspace. The persistent newest tap uses a direct FP32 preallocated slice as
its final writer.

Before registering any layer trace, the stack warms every distinct capture
signature and then freezes program-cache misses. Only crossings regenerated
before read are marked corruptible. Two simultaneous GDN/PLE traces were
allocation-tracked with stable workspace addresses and exact independent
state; 100 changing-token replays passed for GDN, PLE+GDN, and paged QSA.

## Rejected alternatives

- **GDN head TP2:** the 24-value-head kernel lost too much PCC; FP32 output and
  reduction did not repair it.
- **Routed split-K TP2:** K=320+320 sparse down partials caused the final
  correctness miss. Full-K EP2 is the proven repair.
- **Full-K EP2 with the old g20 program:** failed; the full-width g40 program
  is required.
- **Replicated/dense all-expert MoE:** exceeds memory and defeats sparse
  gate-selected execution.
- **Pipeline placement as the capacity fix:** moving full layers does not
  make a balanced 48-layer stack fit on this two-die target.
- **Unbounded/ad-hoc streaming:** superseded by fixed slots, bounded waves,
  exact mmap sources, stable staging, and explicit failure semantics.
- **Smaller context as the primary fix:** even zero KV cache does not make
  ordinary resident BFP4 experts fit with the rest of the stack.
- **Aligned-only public lengths:** incompatible with the decoder contract and
  unnecessary because internal padding/trimming is exact.
- **Private L1 state workspace per GDN layer:** exceeds L1 at full trace-stack
  scale; canonical DRAM plus one serialized workspace fits.
- **Capture while lazily growing the program cache:** can overlap younger
  persistent buffers with older trace temporaries; warm-all/freeze/capture-all
  eliminates that allocation hazard.

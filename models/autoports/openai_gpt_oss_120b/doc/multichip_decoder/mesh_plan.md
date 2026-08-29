# GPT-OSS 120B multichip decoder mesh plan

Status: final selection after correctness, topology, trace, profiler, and
watcher qualification. The initial plan was recorded before implementation;
the measured decisions and rejected candidates are incorporated below.

## Target and baseline

- Model: `openai/gpt-oss-120b`, revision
  `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
- Hardware: Blackhole P150-class devices in `(1, 1)`, `(1, 2)`, and `(1, 4)`
  meshes. Multi-device cases use `FABRIC_1D_RING` and ring CCL on mesh axis 1.
- Baseline: `tt/optimized_decoder.py::OptimizedDecoder`. The `(1, 1)` case is
  that implementation, not a second approximation of it.
- Meaningful layer kinds: even sliding-attention layers with window 128 and odd
  full-attention layers.
- Public limits retained: paged cache, page size 64, configured logical batch
  1--32, logical context 1--131072, and non-aligned logical prefill lengths.
  Hardware correctness is qualified through batch 2; batch 3--32 is
  capacity-only evidence for this decoder stage. Internal tile, page, CCL, and
  expert-routing padding is sliced before return.

## Selected parallelism

Use one-dimensional tensor parallelism across columns (`tp=N` for a `(1, N)`
mesh). The final stack boundary is a replicated hidden/residual tensor. That
keeps local RMSNorm mathematically exact and lets the next packed-QKV and
router projections consume the complete hidden vector. A coherent
sharded-residual family—row-parallel output `reduce-scatter`, distributed
RMSNorm, then fused `all-gather` plus packed-QKV—was measured rather than
rejected on API shape alone. It was slower than the selected TP4 physical-width
replicated path and would broaden every layer-stack boundary.

Attention uses packed column-parallel QKV, local heads, local SDPA, row-parallel
O projection, and one hidden reduction. Experts keep the router replicated and
execute only its four selected experts with ordinary TP sparse matmuls:
gate/up are column parallel, down is row parallel, scores are applied before
the expert reduction, then one hidden reduction restores the residual
contract. Decode must never run all 128 experts densely.

## Per-device tensor plan

`H=2880`, `I=2880`, `Q=4096`, `K=V=512`, 64 Q heads, 8 KV heads,
head dimension 64, 128 experts, and top-k 4. TP4's logical local hidden and
intermediate width is 720 and its TT physical tile width is 736. The packed
gate/up local output is `2 * 720 = 1440`, already tile aligned. All padding is
internal and sliced before a public output.

| Tensor / boundary | Mesh mapping | TP=1 local shape | TP=2 local shape | TP=4 local shape | Dtype / memory and padding |
| --- | --- | ---: | ---: | ---: | --- |
| Residual, logical decode B=1 | replicated | `[1,1,1,2880]` | same | same | BF16/BFP8 device tensor; physical row is 32; initial final contract replicated |
| Input/post-attention norm weight | replicated | `[2880]` | same | same | BF16; local RMSNorm while residual is replicated |
| Packed QKV weight | column parallel on output | `[2880,5120]` | `[2880,2560]` | `[2880,1280]` | BFP8 DRAM; no padding |
| Packed QKV bias | column parallel | `[5120]` | `[2560]` | `[1280]` | BF16; applied once to the local projection |
| Local Q heads | head ownership | 64 | 32 | 16 | head dim 64; height-sharded L1 around RoPE/SDPA |
| Local KV heads / cache | head ownership | 8 | 4 | 2 | paged BFP8 DRAM cache, `[blocks,local_kv,64,64]` per K/V |
| O-projection weight | row parallel on input K | `[4096,2880]` | `[2048,2880]` | `[1024,2944]` | BFP8 DRAM; TP4 keeps 2944 through CCL, then slices to 2880 |
| O-projection bias | device 0 owns value; others zero | `[2880]` | `[2880]` | physical `[2944]` | BF16; prevents bias multiplication by the TP reduction |
| Router weight / bias | replicated | `[2880,128] / [128]` | same | same | BF16, intended L1 resident; router output is top-k 4 |
| Packed gate/up expert weight | column parallel on I | `[128,2880,5760]` | `[128,2880,2880]` | `[128,2880,1440]` | BFP4 DRAM; every rank receives paired gate/up halves for its local SwiGLU |
| Packed gate/up expert bias | column parallel on I | `[128,5760]` | `[128,2880]` | `[128,1440]` | BF16; packed TP4 width is tile aligned |
| Down expert weight | row parallel on I | `[128,2880,2880]` | `[128,1440,2880]` | logical `[128,720,2880]`, physical K 736 | BFP4 DRAM; TT load padding is zero/inert |
| Down bias | value on TP rank 0; zero elsewhere | `[128,2880]` | same | same | BF16; reduced exactly once |
| Layer output | stack boundary | replicated `[... ,2880]` | same | same | same logical/padded shape as input; alternative sharded boundary is measured separately |

For a batch-1, 131072-token cache, there are 2048 pages per user. Combined K
and V cache per layer/device is 136 MiB at TP=1, 68 MiB at TP=2, and 34 MiB at
TP=4, using the 1088-byte physical BFP8 tile size.

## Row-parallel topology selection

Decode B=1 carries 90 hidden tiles. A BFP8 physical hidden payload is 90 x
1088 = 97,920 bytes per device before topology factors; BF16 is 184,320 bytes.
Each decoder has two material row-parallel boundaries: attention O and expert
down.

| Candidate | Residual before -> after | Next consumer | Collective / expected payload | Dtype and buffers | Final decision |
| --- | --- | --- | --- | --- | --- |
| Canonical logical-width all-reduce | replicated -> replicated | local RMSNorm, then packed QKV/router | TP4 slice/copy then 2880-wide `AllBroadcast`; 97,920 B BFP8 logical payload | BFP8 L1 with pre-CCL DRAM/L1 normalization | rejected for TP4 attention: 0.193193 ms boundary |
| Physical-width all-reduce then slice | replicated -> replicated | local RMSNorm, then packed QKV | native RS+AG over 2944; 736 columns/23 tiles per rank | BFP8 L1; short-lived 2944 output then 2880 slice | selected for TP4 attention: 0.110467 ms, PCC 0.999937 |
| Reduce-scatter + distributed norm + fused AG/QKV | hidden-sharded through norm, replicated QKV input gathered inside projection | next packed QKV | RS2944, stats AG, fused hidden AG/QKV | BFP8 residual, BF16 stats; CCL manager semaphores | rejected: 0.129149 ms and a broader stack contract |
| TP2 fused all-gather + matmul | hidden-sharded -> QKV | packed QKV | fused AG/MM requires Ring topology | persistent CCL buffers | rejected: two-device fabric maps Linear and the fused op refuses it |
| Expert logical-width all-reduce | replicated -> replicated | residual add | 2880-wide reduction | BFP8 L1 | selected: 0.070270 ms |
| Expert runtime pad/reduce/slice | replicated -> replicated | residual add | pad 2880->2944, native RS+AG, slice | BFP8 L1 | rejected: 0.115074 ms, 1.638x current |

The final implementation therefore specializes only TP4 attention decode.
Prefill already carries attention's physical width through RS+AG. TP2's 90
hidden tiles divide evenly over two ranks and remains canonical. Experts remain
logical-width on both targets. Exact warmed probe evidence is under
`artifacts/20260828_final_v2/topology_probe/`.

## Memory-capacity calculation

Estimates below include 36 decoder layers, physical TT BFP4 tiles (576 bytes),
BFP8 tiles (1088 bytes), BF16 biases/norms/router, TP=4 tile padding, and the
full-context paged K/V caches. They exclude embeddings, final norm, LM head,
trace/activation buffers, allocator fragmentation, and runtime CCL buffers, so
they are lower bounds for the next full-model stage.

| Mesh | Decoder weights/device | B=1 full-context KV/device | Lower-bound total/device | Full-stack interpretation |
| --- | ---: | ---: | ---: | --- |
| P150 `(1,1)` | 61.113 GiB | 4.781 GiB | 65.895 GiB | exceeds 32 GiB before non-decoder/runtime allocations; decoder-layer stage remains testable but a resident 120B stack is a hard physical impossibility |
| P150x2 `(1,2)` | 30.477 GiB | 2.391 GiB | 32.867 GiB | exceeds 32 GiB before non-decoder/runtime allocations; not a resident full-context stack target without a different weight/storage contract |
| P150x4 `(1,4)` | 15.571 GiB | 1.195 GiB | 16.767 GiB | leaves about 15.2 GiB/device before non-decoder/runtime allocations; full advertised context at batch 1 is physically feasible |

The decoder API therefore keeps 131072 on every tested mesh; this stage does
not invent a smaller public context. It records that only P150x4 can be the
resident full-stack baseline under the selected TT weight formats. A future
full-model stage must measure its actual non-decoder and trace allocations
before advertising a larger full-stack batch. The decoder-layer batch-32 cache
contract remains valid as construction/capacity evidence, but correctness is
qualified only through batch 2 and it is not a claim that 36 batch-32 caches
plus all weights fit concurrently.

## Rejected alternatives

- Data parallelism: duplicates the >61 GiB decoder stack and gives no
  single-user latency or capacity benefit.
- Expert replication: duplicates the dominant 61 GiB decoder-weight footprint
  and does not fit the target full stack.
- Dense all-expert decode: violates gate-selected execution and reads/executes
  32x more expert work than top-k 4.
- Galaxy throughput-expert/fused all-to-all path: designed for a 2D Galaxy
  dispatch/replication topology; it is not the starting contract for a 1D
  non-Galaxy P150 mesh.
- Two-dimensional TP: no second non-trivial mesh axis exists in `(1,N)`.
- Expert parallelism across the sole axis: competes with attention TP and adds
  token dispatch/combine; ordinary active-expert TP divides every expert
  projection while preserving local routing.
- Aligned-only prefill: rejected as a public contract. Tile/routing padding is
  internal and output/cache positions use logical lengths.

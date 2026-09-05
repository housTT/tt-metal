# Four-chip plan before implementation

Target: four Blackhole chips on two P300c boards, logical mesh 1x4, FABRIC_1D_RING.
Mesh smoke: logs/mesh_smoke.log; device order [1,0,3,2], degree two per chip,
11x10 compute grid and 8x1 DRAM grid. TP follows logical mesh order, never PCI IDs.
Baseline: optimized_decoder.py at d085eb6d1abcc8b25f213fede6b68aa873d6cd6b.
This checkpoint is dense (no router, experts or expert replication).

## Tensor plan

All matrix shapes below are TTNN [K,N], BF16 activations and BFP4/LoFi weights.
Initial plan: separate prefill interleaved and decode DRAM-width-sharded weight copies.
Measured refinement below shares the interleaved copy for both modes.
No head replication is needed. Partition contiguous groups of heads, retaining
DeltaNet's two value heads per key head and attention's four Q heads per KV head.

| Tensor | Global | Each chip / ownership |
|---|---|---|
| Residual | B,T,4096 | compare replicated4096 and hidden-sharded1024 |
| Q/K/V/gate attention | 4096,10240 | 4096,2560, packed local Q1024/K256/V256/gate1024 |
| Attention output | 4096,4096 | 1024,4096, row parallel |
| Paged K and V separately | pages,4,64,256 | pages,1,64,256; local head rank |
| DeltaNet Q/K/V | 4096,8192 | 4096,2048; Q512/K512/V1024 |
| DeltaNet A/B | 4096,32 each | 4096,8 each; pad each field to32 in packed projection |
| DeltaNet packed QKV/A/B | 4096,8256 | 4096,2112 including zero A/B padding |
| DeltaNet Z | 4096,4096 | 4096,1024 |
| DeltaNet output | 4096,4096 | 1024,4096 |
| DeltaNet recurrent | B,32,128,128 FP32 | B,8,128,128 FP32 |
| DeltaNet conv taps/history | 8192 channels | local Q512/K512/V1024,2048 channels |
| MLP gate/up separately | 4096,12288 | 4096,3072 column parallel |
| MLP down | 12288,4096 | 3072,4096 row parallel |
| Residual norms | 4096 | replicate or shard1024 with global RMS statistics |
| Q/K norms, DeltaNet output norm | 256/128 | replicated, whole heads local |
| RoPE / positions / page tables | shared positional metadata | replicated; unchanged absolute positions and page IDs |
| Planned embedding reservation | 248320,4096 BF16 | vocabulary shard62080,4096; no padding needed |
| Planned untied head reservation | 4096,248320 BF16 | output vocabulary shard4096,62080; no padding needed |
| Planned final norm reservation | 4096 BF16 | replicated4096 |

Embedding/head/final-norm rows are capacity/interface planning only; this stage
implements and executes decoders. A future vocabulary-sharded embedding can
mask local ownership then reduce its4096-wide lookup; a column-parallel head
can retain local vocabulary logits. Neither path is implemented here.

Decode DRAM shard spec is [local K, ceil(local N/(32*8*readers))*32*readers]
on all eight banks. Initial role configs use8 activation cores, K-block4,
reader1, then sweep legal role-specific geometries. Local K values1024,3072,
4096 admit tiled K32,96,128. A/B logical8 must not be concatenated without
separate tile padding: packed slices start at2048 and2080. Prefill uses the
optimized geometry family with local K/N and legal divisor blocks.

## Collective and residual families to measure

Let S=B*Tphysical*4096*2 bytes (decode B1 uses a physical32-row tile).
A ring reduce-scatter sends approximately3S/4 per chip; all-gather of its
shards adds3S/4. These are algorithm payload estimates, excluding protocol.

| Family | Residual before/after | Next consumer | Per boundary payload | Buffer/precision plan | Assessment before measurements |
|---|---|---|---|---|---|
| local MM + all-reduce | replicated/replicated | local RMSNorm | 1.5S | BF16 CCL; reusable async semaphores | correctness/control, two reductions per layer |
| RS + delayed AG | hidden shard/shard | residual add + distributed norm then AG to next input MM | 1.5S + small stats | BF16 residual, test BFP8 CCL; persistent buffers where supported | less local norm/residual movement |
| fused AG-MM | hidden shard/shard | next projection consumes gathered normalized input | AG3S/4 plus preceding RS3S/4 | packed projections and persistent gathered buffer | test through norm/next projection |
| fused MM-RS | hidden shard/shard | residual add + distributed norm | RS3S/4 plus delayed AG3S/4 | row-packed weight and reusable output | test actual supported Blackhole op contract |
| column WO/down with gathered input | hidden shard/shard | distributed norm | mixer localwidth AG, MLP3072 AG | BF16/BFP8, fused AG-MM candidate | MLP gather wider than residual; may lose |

Common Attention1D lacks this checkpoint's output sigmoid gate, partial64-of256
RoPE and DeltaNet. MLP1D expresses the dense TP algebra but its shared defaults
are not the baseline's Blackhole BFP4 multi-reader programs. Preserve these
optimization families in model-local overrides; reuse optimized math/cache/
nonaligned orchestration and generic CCL helpers rather than replacing with
unvalidated conventional attention. Old35B code is a head/cache reference only;
its MoE,2048 hidden width, old RoPE and historical results are not reused.

TP2+DP2 halves single-user bandwidth and replicates model weights; TP4 is the
target. 2D TP2x2 adds partial reductions before head-local mixers; no Galaxy
axis or expert replication benefit exists. Reconsider only with measurements.

## Context plan

Retain native262144. Each full-attention layer needs2*8192*1*8*1088 =142606336
bytes per device for BFP8 K/V, eight layers total1140850688 bytes. Recurrent
state is524288 bytes per user per linear layer (24 layers); conv history12288
logical bytes per user per layer. Full-model accounting must include both
projection weight copies, all norms/constants, eight RoPE copies unless shared,
embedding and untied LM head, and8GiB reserved trace/activation storage. The native replicated residual is
2GiB: input, accumulated chunk outputs and final concatenation can overlap
(6GiB), leaving2GiB for trace, chunk working sets and allocator overhead.
This reserve was refined after inspecting inherited public orchestration.
These are estimates until allocation/runtime validation. No capability reduction
or completion claim follows from this plan.


## Measured selection refinement

The role-local DRAM-sharded sweep covered220 candidates at the selected
BFP4/LoFi policy. A larger interleaved-weight family then measured32,64 and110
workers.32 workers improves whole-layer decode; the provisional final path
uses an8x4 compute grid, interleaved BFP4 weights, BF16 L1 outputs, M=1tile/core,
N=ceil(localN/(32*32)) tiles/core and the largest divisor of N at most8 for
output subblock width. Per-role K blocks are QKVG2, GDN-packed16, Z32,
GDN-output8, O8, gate4, up16, down12. These give explicit output N/subblocks:
QKVG3/3, GDN-packed3/3, Z1/1, GDN-output4/4, O4/4, gate3/3, up3/3,
down4/4. N padding belongs to the kernel; logical local head slices are unchanged.

Replicated BF16 residuals and two native two-link all-reduces remain fastest
among measured stack-compatible collective families. A32×32 BF16 tile is2048
bytes:8192-byte router packets carry four tiles, versus two within the default
4352-byte payload. The
fabric policy must be installed before mesh open; `fabric_router_config()`
provides it for callers. Async candidates use one link due to the controlled
two-link corruption documented by AutoFix. The selected native all-reduce
uses a separately validated decomposition and semaphore policy, sharing the
reduce-scatter kernel factory with the experimental path.

All shape-faithful family controls and numerical failures remain in
`candidate_measurements.csv`, `geometry_search.md`, and AutoFix reports.
The pipeline does not claim global mathematical optimality or a silicon speedup
beyond measured paired timings.32-chip/Galaxy-only specializations and MoE
execution are inapplicable to this dense four-chip model.


The selected32-core path shares its interleaved projection weights between
prefill and decode and frees the provisional optimized decode copies during
setup. Persistent projection bytes are30818304 per linear layer and29491200
per full-attention layer,975568896 across24+8 layers per device. The prior
DRAM-sharded option retains its separate copies only when explicitly selected.
The refined total capacity estimate is12826869760 bytes/device, including the
same8GiB reserve; no context or batch limit was lowered to obtain this saving.


## Final mixed projection refinement (2026-09-05)

The topology remains TP4 replicated residuals. Precision-locked real-weight
QKV experiments select an extra decode-only BFP8 LoFi copy, retaining BFP4
prefill and all other projections. Local QKVG is[4096,2560], sharded across
8 DRAM banks with shard[4096,320], no extra N padding. Input[32,4096] is
width-sharded across32 worker cores with shard[32,128]; dedicated DRAM
matmul uses in0_block_w4 and reader1. The32 logical batch rows are tile
padding for smaller public batches, which remain1..32. Interleaved QKV
block8/16 is slower in the complete layer (~.292ms vs.283ms). Cores4/block4
DRAM is also slower (~.288ms); all-rank output is identical to cores32/block4.
BF4 block2 passes but is slower (~.309ms full layer); larger BF4 blocks fail
real per-user HF comparison. BF4 HiFi2 and FP32 accumulation do not repair
that gate. This selection is based on real checkpoint/recorded-input evidence.

Other decode projections retain8x4 standard multicast matmul: GDNpacked32,
Z32, GDNout8, O8, gate8, up8, down6 K blocks. per_core_M1 and
per_core_N=ceil(N/(32*32)); output subblock width is the largest divisor of
per_core_N no larger than8, height1. Thus local N2112 ->perN3/subW3,
N1024 ->1/1, N4096 ->4/4, N3072 ->3/3. Prefill program contracts above
remain unchanged. Input/output tensor formats and collective payloads are
unchanged; no host conversion enters runtime.

Extra QKV storage is4096/32*2560/32*1088=11141120 bytes/device/layer.
Across8 full-attention layers this adds89128960 bytes. All32-layer projection
storage is1064697856 bytes/device; conservative full-model capacity estimate
is12915998720 bytes/device. Native262144 is validated: release_mixed_watcher_contracts holds the updated
reservation during native full-attention execution and passes. The full99-case
contract suite also passes. Same-policy prefill8×8/11×10 block8/16 controls
retain11×10/block16 for both kinds; see PERF_ANALYSIS.md for the exact
FP32 gdn_out L1 limit and the15-sample warmed comparison.

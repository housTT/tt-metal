# Gemma 4 26B A4B multichip mesh plan

This plan was selected before changing the seeded TP=4 candidate into the final
profiled path. The hardware is one P300C QB2 exposing four Blackhole devices;
the accepted P150 proxies are a 1x1 device, a 1x2 compute submesh of a 2x2
parent, and the full 1x4 mesh for P150, P150x2, and P150x4 respectively. TP2
uses a `FABRIC_2D` parent because the physical QB2 links form a 2x2 fabric,
then uses a one-link Linear collective on its 1x2 submesh. TP4 uses
`FABRIC_1D_RING` and a two-link Ring collective on mesh axis 1. TP1 performs no
collective.

## Selected strategy

Use one-dimensional tensor parallelism with a replicated residual stream.
QKV, dense gate/up, and each gate-selected expert gate/up are column parallel.
O, dense down, and active-expert down are row parallel and are followed by a
hidden-width all-reduce.  Attention runs only local Q/K/V heads.  The router,
norm weights, layer scalar, page table, and current positions are replicated.
KV cache is paged and device-local by KV-head ownership.  Full-attention has
only two global KV heads, so TP=4 duplicates each head on a pair of ranks; the
pairing follows the two local Q-head groups that consume that KV head.

The public input/output contract remains `[1, 1, logical_M, 2816]` replicated
on the mesh.  Internal tile padding is not a logical batch or sequence
restriction.  Dense intermediate 2112 and expert intermediate 704 are padded
only when required by `TP * 32`; padded channels have zero weights and are
contracted away before the residual add.

The residual choice is deliberate.  A decode tile contains 32 x 2816 BF16 =
180,224 bytes.  The selected layer communicates three such row-parallel
partials (attention O, dense down, expert down).  A fractured residual would
replace the all-gather halves with distributed norm statistics, but it also
requires gathered-K or K-sharded consumers for QKV, dense gate/up, router, and
routed experts. The original distributed RMSNorm plus fused all-gather/matmul probe
proved the dense, router, and fixed-expert consumers, but its attention widths
were stale: sliding is 8192, while full is logically 10240 and physically
12288 on TP4 because KV heads are duplicated across rank pairs. Those corrected
attention cases pass in `artifacts/fused_agmm_qkv_corrected.xml`. The matching
fused matmul/reduce-scatter
family is subject to the Blackhole M=32 race noted in
`models/demos/gpt_oss/tt/attention/operations.py` (#46181). The exact Gemma
repro exposed invalid helper geometry and output-spec derivation before a
corrected numerical result was reached. `AUTODEBUG_FUSED_RS.md` records that
fused-family result; the later unfused family below supplied the decisive
complete-path comparison.

The current unfused APIs have no equivalent shape blocker: local row matmul
partials `[1,1,32,2816]` reduce-scatter to `[1,1,32,704]` with `Shard(dim=3)`
topology, which is directly consumable by distributed RMSNorm and fused AGMM.
The new opt-in exact-shape micro-chain covers all four row K widths and the
packed dense N=4352 consumer. All four cases pass both RS and final AGMM PCC
0.99 in `artifacts/unfused_fractured_chain.xml`. The follow-up AutoFix then
re-placed real expert gate/up weights over K, used indexed top-8 sparse
execution, persistent buffers, distributed norms, and K-sharded next-QKV. The
micro-chain passed changed-route 100-replay stress, watcher, and profiler gates
at 1.741x over its incumbent comparison. A complete real-weight paged sliding
layer remained correct but lost 29.47% by the warning-clean 30-sample median
(0.844279 versus 0.652086 ms) because it needed six RS, six norm-stat AG, and
one router AR. The selected replicated
residual is therefore a measured decision, not an untested convenience; see
`AUTOFIX_FRACTURED_RESIDUAL.md`.

Within each device, the selected optimized layout normally holds the residual
in the inherited 22-core height-sharded L1 form between contractions. TP1,
TP2, and TP4 sliding layers, plus TP1/TP2 full layers, select R22, folded router
projection, shared FFN norm, folded expert scale, fused final scalar, and
row-major routing. TP4 full layers use a correctness-gated exception: R0,
unfolded router projection, separate FFN norms, folded expert scale, fused
final scalar, packed expert decode, and non-row-major routing. Adding the
router fold to that exception lowers B32 PCC to 0.994890; adding shared FFN
norm lowers it to 0.994940. The maximal passing subset is 0.995025 and remains
the selected policy.

TP4 decode uses one reader per DRAM bank for O, packed dense gate/up, and dense
down. The inherited multi-reader helper calls a primitive whose worker-hop
geometry requires a unit `MeshDevice`; both two- and three-reader TP4 probes
hit that hard contract before timing. Multichip construction therefore rejects
workers greater than one for TP2/TP4 instead of exposing the fatal path.

## Per-profile shapes

| Tensor or operation | Global logical shape | P150 / TP1 local | P150x2 / TP2 local | P150x4 / TP4 local | Mapping / padding |
| --- | --- | --- | --- | --- | --- |
| residual, norm input/output | `[M, 2816]` | `[M, 2816]` | `[M, 2816]` | `[M, 2816]` | replicated; decode is tile-padded to M=32 internally |
| sliding Q | `[2816, 4096]` | `[2816, 4096]` | `[2816, 2048]` | `[2816, 1024]` | column parallel, 16 / 8 / 4 local Q heads |
| sliding K and V | each `[2816, 2048]` | each `[2816, 2048]` | each `[2816, 1024]` | each `[2816, 512]` | column parallel, 8 / 4 / 2 local KV heads |
| full Q | `[2816, 8192]` | `[2816, 8192]` | `[2816, 4096]` | `[2816, 2048]` | column parallel, 16 / 8 / 4 local Q heads |
| full K-as-V packed projection | K and V each `[2816, 1024]` | each `[2816, 1024]` | each `[2816, 512]` | each `[2816, 512]` | TP4 duplicates global KV head 0 on ranks 0/1 and head 1 on ranks 2/3 |
| sliding O | `[4096, 2816]` | `[4096, 2816]` | `[2048, 2816]` | `[1024, 2816]` | row parallel then hidden all-reduce |
| full O | `[8192, 2816]` | `[8192, 2816]` | `[4096, 2816]` | `[2048, 2816]` | row parallel then hidden all-reduce |
| dense gate and up | each `[2816, I]` | `I=2112` | `I_local=1056` | `I_local=544` | column parallel; TP4 pads global I 2112 -> 2176 |
| dense down | `[I, 2816]` | `K=2112` | `K_local=1056` | `K_local=544` | row parallel; padded rows are zero |
| 128 expert gate and up | each `[2816, E]` | `E=704` | `E_local=352` | `E_local=192` | active-expert TP; TP4 pads global E 704 -> 768 |
| 128 expert down | each `[E, 2816]` | `K=704` | `K_local=352` | `K_local=192` | active-expert row parallel then expert-score reduce and hidden all-reduce |
| router | `[2816, 128]` | full | full | full | replicated FP32; identical top-8 routes on all ranks |
| sliding KV cache per layer | K/V, 8 heads, 1024 x 256 | 8 heads, 8,388,608 B | 4 heads, 4,194,304 B | 2 heads, 2,097,152 B | paged BF16, head-sharded |
| full KV cache per layer | K/V, 2 heads, profile context x 512 | 2 heads, 50,688 physical tokens, 207,618,048 B | 1 head, 262,144 tokens, 536,870,912 B | 1 head, 262,144 tokens, 536,870,912 B | TP4 head-pair duplication is the physical limit |

`M` is the valid logical sequence length for prefill and the valid logical
batch for decode.  `I` is dense intermediate size and `E` is expert
intermediate size.

## Capacity and precision policy

The stack projection uses 25 sliding and 5 full layers with BF16 KV cache.
Sliding-attention projections are BF16. Full-attention, dense, and expert
projections are BFP8 except that TP1 full-attention expert gate/up weights are
BFP4. Broader TP1 expert-down BFP4 candidates failed PCC and are not selected.
TP4 retains decode-only DRAM copies of O, packed dense gate/up, dense down, and
packed expert gate/up. Capacity disables the packed expert copy on TP1 and
TP2. The final no-packed TP2 policy passed direct and independent-HF PCC,
mixed-layer stacked decode, and warmed trace/performance gates in
`final_profiles_after_capacity.xml` and `warmed_required_profiles_final.xml`.

The full-stack contract mandates downstream BFP8_B embedding and LM-head
storage using the physical 1,088-byte tile size. It also includes replicated
BF16 final norm, a 64 MiB trace region, prior profile allocator slack, and the
full conservative prefill lifetime. Source inspection shows that QKV is
materialized before chunked SDPA, output chunks remain live through concat,
and inherited residual/branch tensors remain live through attention and FFN.
At MoE concat both the accumulated output chunks and the new concat output
coexist. No implicit deallocation credit is taken.

| Profile/context | packed expert retained copy | decoder + KV + CCL | BFP8 terminals | final norm | trace + live tensors + allocator | full-stack total | 32 GiB headroom |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P150 / 50,624 contiguous maximum | disabled, 0 B | 27,284,654,080 B | 1,568,669,696 B | 180,224 B | 5,505,196,544 B | 34,358,700,544 B | 1,037,824 B |
| P150x2 / 262,144 | disabled, 0 B | 16,486,360,064 B | 784,334,848 B | 180,224 B | 16,300,113,920 B | 33,570,989,056 B | 788,749,312 B |
| P150x4 / 262,144 | 4,411,883,520 B | 15,251,053,568 B | 392,167,424 B | 180,224 B | 17,776,508,928 B | 33,419,910,144 B | 939,828,224 B |

TP1 uses the 50,623-token nonaligned peak in the table because it is tighter
than aligned 50,624 after accounting for caller-owned unpadded hidden/cos/sin
tensors. Its cache is block-rounded to 50,688. The next candidate, 50,625,
exceeds capacity by 673,280 B and is rejected before device allocation. At
262,144, TP1 would need 56,398,546,944 B, so the 50,624 limit is a hard
per-profile exception. The aligned limit itself leaves 389,822,464 B.
Real-weight sliding/full representative layers pass at both 50,623 and 50,624.
The earlier passing 53,343/53,344 probes are superseded/non-boundary evidence.

Each packed-expert tensor is `[1,128,2816,2*E_local]`. Physical BFP8 tile
accounting gives 8,088,453,120 B across 30 TP2 layers and 4,411,883,520 B on
TP4. TP1's all-BFP8 candidate is 16,176,906,240 B; even an unselected all-BFP4
physical-tile lower bound is 8,564,244,480 B, above its 7,075,084,288 B
decoder-stage headroom at the supported context. The machine-readable formula
and every intermediate are in `capacity_projection.json`.

This is an analytical placement contract, not a measured full-model
allocation. The multichip decoder does not load terminal tensors; the
downstream full-model stage must honor the BFP8_B terminal dtype and separately
prove terminal/logit correctness.

## Rejected and conditional alternatives

| Alternative | Residual before/after | Collective payload and buffers | Next consumer | Status |
| --- | --- | --- | --- | --- |
| local row matmul + synchronous all-reduce | replicated -> replicated | BF16 hidden partial; runtime CCL buffers | local norm/residual | rejected for decode: the second back-to-back TP2 reduction can wait |
| local row matmul + persistent async all-reduce | replicated -> replicated | BF16 hidden partial; three rotating buffers/semaphores | local norm/residual | selected for TP2/TP4 decode; verified by all profile PCC gates and trace replay |
| reduce-scatter then immediate all-gather | replicated -> fractured -> replicated | same hidden payload plus two launches | local norm/residual | rejected structurally unless fusion overlaps enough to win |
| unfused local row matmul + delayed-gather reduce-scatter | fractured -> fractured | six BF16 RS, six norm-stat AG, router AR in a complete sliding layer | distributed RMSNorm, indexed top-8 experts, then K-sharded next projection | rejected after full integration: PCC 0.999805 and 20 bit-exact replays, but warning-clean 30-sample medians were 0.844279 ms versus 0.652086 ms incumbent (29.47% slower); the isolated expert/next-QKV chain was 1.741x faster, proving the complete-path comparison was necessary |
| fused matmul + reduce-scatter, delayed gather | replicated -> fractured | persistent RS buffers/semaphores | distributed norm then fused AG+matmul | rejected for this stage: helper geometry/spec is invalid and no corrected numerical stress exists; see `AUTODEBUG_FUSED_RS.md` |
| fused all-gather + matmul | fractured -> column output | persistent AG buffers/semaphores | QKV, dense gate/up, router, expert gate/up | dense/router/fixed-expert and corrected 8192/12288 attention shape probes pass |
| expert parallel by expert ownership | dispatch/return routed tokens | token dispatch plus score-weighted return | active experts | rejected for 1-4 chips: top-8 single-user traffic and imbalance dominate, while TP gives every selected expert all DRAM bandwidth |
| dense all-expert MoE | replicated | no route sparsity | dense 128-expert compute | rejected; violates the gate-selected active-expert contract and multiplies decode work |
| QKV in decode DRAM-sharded roles | unchanged | unchanged | local attention | rejected: 10.6055 ms vs 10.6147 ms selected TP4 B32 is within noise and retains an extra copy |
| wider QKV decode block (`block_w=4`) | unchanged | unchanged | local attention | rejected: 10.6153 ms TP4 B32, no improvement |
| router input staged to L1 | unchanged | unchanged | replicated router | rejected: 10.6345 ms TP4 B32, slower than selected |
| two or three DRAM readers per bank | unchanged | candidate duplicates/pads decode weights | O / dense contractions | unsupported: inherited worker-hop primitive requires a unit `MeshDevice`; selected value is one |
| one TP4 sliding O block width for all decode batches | unchanged | one shared BF16 O weight | row-parallel O then hidden all-reduce | rejected: block 4 misses HF B1 at 0.994744; block 2 misses B32 optimized PCC at 0.994908; selected dual configs reuse the weight |
| BFP4 decode QKV | unchanged | extra retained decode-only QKV tensor | local attention and BF16 paged cache | rejected on real weights with the role observed at runtime: PCC 0.984498 sliding and 0.978292 full |
| BFP4 decode O | unchanged | extra retained decode-only O tensor | hidden all-reduce | rejected: sliding PCC 0.991448; full PCC 0.998744 passes but is only 0.39% nominally faster and does not justify another retained layer exception |

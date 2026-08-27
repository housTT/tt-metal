# Qwen3.8-Flash-Next multichip decoder plan

The target was locked before implementation on 2026-08-27 and was not changed:
one P300 board, its two Blackhole dies as a fixed `1x2` mesh, and
`ttnn.FabricConfig.FABRIC_1D`.  Each die has an `11x10` compute-with-storage
grid, eight DRAM channels, and 34,225,520,640 bytes (31.875 GiB) of nominal
DRAM.  The single-chip baseline is `OptimizedDecoder` on P300c chip 0.

## Delivered strategy and public boundary

`MultichipDecoder` subclasses `OptimizedDecoder`.  Setup builds the exact
optimized graph for rank 0 and rank 1, patches rank 1's setup-only tensor
shards into the shared mesh allocation, and releases the temporary graph.
The original resident runtime performs no Torch or host conversion.  The
resumed exact host-backed runtime adds only declared compact route-id D2H,
selected expert/PLE mmap lookup, and bounded H2D outside capture; all decoder
math and collectives remain on TT.

The public input and output stay replicated BF16
`[1, 1, logical_sequence_or_batch, 10240]`.  The public sequence/batch length
is never replaced by an aligned length.  The inherited 128-token chunk planner
pads only internal work and trims each result; structural tests cover lengths
`1,31,32,33,63,64,65,127,128,129,2047,2048,2049,262143,262144`, and real
prefill/PCC covers non-aligned length 33.  Decode covers batch 1 and 32 with
per-user INT32 positions; QSA batch 32 uses distinct, shuffled page tables.

The final split is:

- hyperconnection, PLE, router, indexer, and GDN: replicated;
- QSA query/KV heads: TP2;
- shared and routed MoE intermediate width: TP2;
- QSA and MoE row-parallel partials: BF16 linear `all_reduce` along mesh axis 1;
- boundary activations: replicated after every reduction.

GDN replication is a measured correction to the pre-code plan.  Splitting 48
value heads into two 24-head recurrence kernels produced about 0.992 final
decode PCC; FP32 output projection and FP32 reduction did not repair it.  The
replicated graph restores exact optimized program geometry and removes the GDN
collective while retaining the much larger QSA/MoE split.

## Tensor and activation placement

Shapes use checkpoint convention `[output,input]`.  `R` is replicated and
`TP` is a distinct per-die shard.  There is no logical padding in these weight
shards; tile padding is internal to TTNN.

| Role | Global logical shape | Placement | Per-die logical shape | Runtime result |
| --- | --- | --- | --- | --- |
| hyper residual | `[...,10240]` | R | `[...,10240]` | public boundary |
| HC norm | `[10240]` | R | `[10240]` | no CCL |
| HC down+inject | `[10240,324]` | R | `[10240,324]` | no CCL |
| HC up | `[320,10240]` | R | `[320,10240]` | no CCL |
| router | `[2560,512]` | R | `[2560,512]` | identical logits/top-k |
| shared gate/up, each | `[640,2560]` | TP output | `[320,2560]` | local activation |
| shared down | `[2560,640]` | TP input | `[2560,320]` | MoE partial |
| routed packed gate+up | `[512,1280,2560]` | TP intermediate | `[512,640,2560]` | selected experts only |
| routed down | `[512,2560,640]` | TP intermediate | `[512,2560,320]` | MoE partial |
| PLE key/value | `[160,12800]` | R | unchanged | no CCL |
| GDN qkv | `[10240,2560]` | R | unchanged | 16 key/48 value heads |
| GDN z | `[6144,2560]` | R | unchanged | 48 value heads |
| GDN b/a | `[48,2560]` each | R | unchanged | 48 value heads |
| GDN `dt_bias`/`A_log` | `[48]` each | R | unchanged | 48 value heads |
| GDN conv | `[10240,1,4]` | R | unchanged | replicated state |
| GDN out | `[2560,6144]` | R | unchanged | no CCL |
| QSA q+gate | `[12288,2560]` | TP head | `[6144,2560]` | 12 query heads |
| QSA k/v, each | `[512,2560]` | TP head | `[256,2560]` | one KV head |
| QSA out | `[2560,6144]` | TP input | `[2560,3072]` | QSA partial |
| index q/k | `[512,2560]` / `[128,2560]` | R | unchanged | identical selected tokens |

Local QSA activations are 12 query heads and one K/V head of width 256;
the attention epilogue has local width 3072 before a `[... ,2560]` partial.
Local routed/shared expert activations have width 320.  Every die can address
all 512 checkpoint experts through ten fixed slots and executes only
gate-selected experts (`top_k=10` per logical token).  Prefill unions larger
than ten are partitioned into bounded waves.  The sparse kernel's tile union
can exceed ten because 32 logical rows share a routing tile; tests separately
assert exactly ten nonzero weights for the first logical row.

## State, paging, and cache placement

- GDN recurrent state is replicated FP32 `[batch,48,128,128]`; its convolution
  state is replicated for the complete q/k/value packing.
- QSA K and V caches are local BFP8
  `[max_blocks,1,64,256]`; global KV head 0 is on rank 0 and head 1 on rank 1.
- The raw index cache is replicated BFP8 `[max_blocks,1,64,128]` so both ranks
  select the same virtual tokens.
- The compressed index-key cache is replicated BF16
  `[max_blocks,1,16,128]`.
- Page tables, rotary tables, and INT32 current positions are replicated.
  Addresses are calculated from logical positions, never activation padding.

At batch 1 and context 262,144, `max_blocks=4096`.  Per QSA layer and die the
tile-padded allocations are 71,303,168 bytes for each local BFP8 K/V cache,
35,651,584 bytes for the BFP8 raw index cache, and 16,777,216 bytes for the
BF16 compressed index cache.  Across 12 QSA layers this is exactly
2,340,421,632 bytes (2.1796875 GiB) per die.  The maximum-position trace uses
position 262,143, the last shuffled page, these exact cache shapes, and the
same page/update/gather graph captured for replay.

## Collective plan and measured rows

The reduction tensor is BF16 `[1,1,logical_rows,2560]`.  Its tile-padded
payload is 163,840 bytes at decode (`32x2560`) and 655,360 bytes at 128-token
prefill.  GDN layers perform one reduction after MoE; QSA layers perform one
after attention and one after MoE.  The fixed settings are mesh axis 1, linear
topology, one link, and DRAM output.  Tracy labels the two-die reduction's
fabric row `AllGatherDeviceOperation`.

| Layer kind | Prefill fabric rows | Prefill device time | Decode fabric rows | Decode device time |
| --- | ---: | ---: | ---: | ---: |
| GDN layer 0 | 1 | 21.917 us | 1 | 8.5865 us |
| PLE+GDN layer 1 | 1 | 22.4475 us | 1 | 9.156 us |
| QSA layer 3 | 2 | 44.8895 us total | 2 | 18.4225 us total |

An L1-output reduction candidate was faster in isolation but made layer-0
rank outputs diverge in the combined correctness/trace test.  The accepted
DRAM result is rank-identical.  FP32 row partials doubled payload without a
PCC need and were rejected in favor of BF16.

## Full-stack residency boundary

The hard unresolved capacity gate is independent of the working per-layer
graph.  TP2 routed/shared experts have 58,982,400 tiles per die over 48 layers.
Ordinary BFP4 therefore needs 33,973,862,400 bytes (31.640625 GiB) per die;
uniform BFP2 needs 18,874,368,000 bytes (17.578125 GiB).  With the measured
maximum-context cache, a conservative 3,921,895,424-byte non-expert allowance
(including replicated GDN), and a 1 GiB activation/allocator/trace reserve,
only 26,889,461,760 bytes (25.04276 GiB) remain for experts.

Ignoring compressed-kernel bank padding, a BFP2/BFP4 mix must satisfy:

```text
expert_bytes = 58,982,400 * (320 + 256*f_bfp4)
f_bfp4 <= 0.5308186849
```

Thus at least 46.918% of logical tiles must be BFP2/zero even in the optimistic
packing model.  The DeepSeek compressed expert kernel pads this model's local
640 gate/up width to 768 across eight banks.  Its physical routed-expert count
is `(512*80*24 + 512*10*80)*48 = 66,846,720` tiles/die; uniform BFP2 then
needs 21,390,950,400 bytes and the practical BFP4 fraction is only
32.13106043198529%, requiring 67.86893956801471% BFP2/zero.  Tested fit
candidates did not clear correctness: uniform/heavy BFP2 was about 0.73 PCC;
real layer-0 selected-top10 down BFP2 reached
0.982582; a low-error 60% BFP4 mix reached 0.983209; BFP2 plus a rank-96 BFP4
residual reached 0.968320.  `AUTOFIX.md` records the candidate audit.
The extracted raw command outputs are in `capacity_candidate_probes.log`.

### Resume-1 exact host-backed resolution

The resident arithmetic above remains the reason ordinary all-expert weights
cannot be used.  The resumed plan resolves it without reducing the advertised
context:

| Resource | Per die/rank | Full-stack calculation |
| --- | ---: | --- |
| fixed expert slots | 10 experts/layer | `48 * 10 * 1,382,400 = 663,552,000 B` |
| fixed expert upload staging | 1 expert/layer | `48 * 1 * 1,382,400 = 66,355,200 B` |
| total expert device storage | 11 experts/layer | `729,907,200 B` |
| exact PLE prefill staging | `[1,1,128,2560]` BF16 | `655,360 B` |
| exact PLE decode staging | `[1,1,1,2560]`, physical 32 rows | `163,840 B` |
| max-context cache | TP-local/replicated as above | `2,340,421,632 B` |
| non-expert weights | TP2 plus replicated GDN | `3,921,895,424 B` |
| runtime/trace reserve | fixed allowance | `1,073,741,824 B` |
| planned total |  | `8,066,785,280 B/die` |
| planned headroom |  | `26,158,735,360 B/die` |

Each expert miss reads one BF16 checkpoint expert (9,830,400 bytes), packs
rank-local gate/up `[1,1,2560,640]` and down `[1,1,320,2560]`, and transfers
1,382,400 B/rank in BFP4.  A cold top-10 decode transfers 27,648,000 bytes
across both ranks; a hit transfers zero.  Ordered decode slots preserve fixed
back-trace addresses.  Generation-checked LRU prefill waves publish a slot only
after both projections on both ranks have uploaded successfully.

The PLE table is 128 mmap shards, 320,001,446 logical rows padded to
320,001,536, row width 160 BF16, total 102,400,491,520 bytes.  Sixteen exact
EOS-aware n-gram rows assemble each 2560-wide token embedding; only selected
rows enter TT staging.  The complete boundary and failure semantics are in
`../host_weight_contract.json`.

## Rejected alternatives

- **GDN TP2:** decode PCC failure described above.
- **Expert parallelism or 24-layer pipeline placement:** the same global
  expert bytes are split across two dies, so neither creates capacity; EP also
  load-imbalances ten selected experts.
- **Replicated/dense all-expert MoE:** exceeds memory and defeats gate-selected
  execution.
- **Immediate sharded residual:** requires distributed four-stream norms plus
  gathers before replicated HC/router/GDN, adding more communication than it
  removes on two dies.  The simpler L1 reduction experiment also failed rank
  correctness.
- **Unbounded/ad-hoc host weight streaming:** rejected.  It is superseded by
  the exact bounded fixed-slot contract above, whose D2H/H2D boundaries are
  explicit and whose expert math remains on TT.
- **Smaller context as the primary fix:** even zero KV cache cannot make
  standard BFP4 experts plus non-expert weights and runtime reserve fit.
- **Aligned-only public lengths:** incompatible with the existing decoder
  contract and unnecessary; internal chunk padding already works.

# Graph inventory and fusion ledger

This inventory records the source scan and adapted experiments for this stage.
Final numerical, trace, capacity, watcher and device-time gates are recorded separately.
Inputs are actual layer-0/3 pinned weights, hidden width 2048, GDN K16/V32/D128,
full attention Q16/KV2/D256, and 256 routed experts with top-8 and width512.
The baseline CSVs are `../functional_decoder/tracy/{linear_attention,full_attention}/ops.csv.gz`.
Their first decode replay has 166 and 101 device programs, respectively.

## Complete logical sequence

The prefill planner repeats a full layer for 128-token GDN chunks or 512/128-token
full-attention chunks, using single-token execution for unaligned tails. Logical
length, absolute page indices and request slots remain the functional contract.

| Order / branch | Operations and inputs | Data movement / state |
|---|---|---|
| Prefill entry | Flatten BF16 input; resident indices + scalar offset; gather chunk | Input untilize; embedding tilizes each chunk; output row-major buffer |
| Full decode metadata | Cast absolute positions; embed cos and sin | Resident row-major RoPE tables → tiled gathered rows |
| Both | Cast FP32 → RMSNorm → FP32 affine → cast BF16 | Interleaved DRAM; two outer norms per layer |
| GDN projection | Packed QKV+A+B+Z linear, split columns | BF16 weights, HiFi4, FP32 accumulate; DRAM |
| GDN convolution | State concat; four slices/casts/multiplies and three adds; BF16 round; FP32 SiLU; BF16 round | Three BF16 history tokens; repeated untilize/tilize; copy new history |
| GDN gates | sigmoid(B); softplus(A+dt_bias) × −exp(A_log) | FP32 gate arithmetic; beta returns BF16 |
| GDN Q/K/V | Split, head reshape, Q/K square→sum→epsilon→rsqrt→multiply | BF16 normalization; row-major bridge for head axes |
| GDN prefill | Dedicated exact chunk_gated_delta_rule | 32-token native chunks; resident masks; FP32 persistent state |
| GDN token | Repeat Q/K to V heads; cast FP32; scale Q; exp decay; reshape scalars; decay state; K/state product + reduce; delta; outer product + add; Q/state product + reduce | Explicit FP32 SFPU maintains precision; state copy; axis permutations |
| GDN output | BF16 round; FP32 RMSNorm; BF16 round/weight; FP32 SiLU(Z)×normalized; BF16 cast; linear | Head flatten; BF16/HiFi4 output projection |
| Full projection | Q+gate linear/split; K linear; V linear; Q/K RMSNorm; head transposes | BFP8/LoFi; decode L1, prefill DRAM |
| Full RoPE | First64 split; half rotate with neg/concat; two multiplies + add for Q and K; concat untouched192 | Prefill `[1,H,T,D]`, decode later transposed to `[1,1,H,D]` |
| Full cache | Prefill two paged fills; decode two paged updates | BF16 caller-owned DRAM cache; decode K/V height-sharded L1 |
| Full attention | Dedicated chunked causal SDPA or paged decode SDPA; concatenate heads | Paged indices/absolute positions on device; 32-token decode chunks for odd page capacities |
| Full output | sigmoid(gate) × attention; output linear | BFP8/LoFi output projection |
| Both | Residual add; second outer RMSNorm | BF16 residual boundary retained |
| Router | Linear→FP32 full softmax→top8→sum/div→BF16→scatter dense | Scatter introduces row-major conversions; normalized scores |
| Experts | Routing mask; packed sparse gate/up; SwiGLU; sparse down; routing multiply; expert sum | Decode L1; prefill DRAM; decode expert↔token permutations; prefill tile-union mask |
| Shared expert | Gate linear/sigmoid; gate linear with SiLU; up linear; multiply; down linear; output×gate | Mixed BF16/BFP4/BFP8, existing SiLU epilogue |
| Both exit | Routed+shared; final residual add | Prefill output untilize/slice_write and final tilize; decode no host conversion |

## Skill pattern closure

Every skill pattern was checked against the complete graph above. Applicable
rewrites were tried on device, including layout/packing repairs after first API
failures. Candidate flags are research controls; only `DEFAULT_FUSIONS` is the
delivered path. `candidate_results.csv` indexes screening runs; exact commands,
source hashes, HF and functional equivalence PCC live in the referenced JSON.
All screening latency columns are explicitly host wall time. Final device tables
come from whole-layer Tracy spans, never a sum of matmul durations.

| Skill pattern | Resolution and evidence |
|---|---|
| Dedicated SiLU/sigmoid/softplus | Existing dedicated ops retained; binary input/post activation fusion selected (`linear_trials129`, `full_trials129_adapted`). |
| Softmax recognition | Already native. FP32 selected-logit softmax cancels full-router denominator and passes; specialized MoE gate is faster (`router`, `moe_gate` trials). |
| RMSNorm recognition / affine | Existing RMSNorm; affine-fold trial passes but offers no measurable decode win. GDN L2-as-RMS with scaled epsilon passes and wins. |
| Distributed RMSNorm | No distributed tensors/collectives on the authorized single-device stage. |
| Dedicated SDPA | Already uses native paged decode/chunked prefill, including odd-capacity k_chunk_size32. No primitive score-softmax-value chain remains. |
| Packed QKV + head split | Selected actual Q/K/V/gate weight repacking and native split (`qkv`, `heads`). |
| Decode create QKV heads | Adapted sharded outputs back through required normalization/RoPE layouts; correct but slower than selected head split (`extra_trials129_fixed`, `dedicated_trials129`). |
| Decode concatenate heads | Adapted SDPA DRAM output through required sharding; correct but slower (`dedicated_trials129`). Direct sharded GQA SDPA output was invalid, so first error was not used as rejection. |
| Prefill concatenate heads | Already native `transformer.concatenate_heads`; retained. |
| Dedicated RoPE / decode layout fold | Initial partial64 rotary_embedding adapter passed; final selected indexed partial64 copies untouched192 in-kernel. Coherent16-channel-group Q/K/gamma/cos packing preserves BFP8 exponent groups. Initial natural-head round trip was removed before selection:920.399us vs924.030us decode, real HF min.99575436 (`indexed_rope4096`, `indexed_rope_native4096`). Both prefill/decode use the same K basis; no mixed populated functional/fused cache. |
| TopK / specialized routing | Generalized MoE gate selected. Decode directly passes its top8 values/IDs to compact experts, eliminating dense scatter and repeated topk (`compact_trials4096`). |
| Parallel conv-sum / RepVGG | No parallel spatial-convolution branches. |
| Shared-LHS matmul packing | GDN mega already packed. Full QKV selected; shared gate/up packed minimal-matmul SwiGLU beats separate and ordinary packed matmul (`dedicated_trials129`). |
| Spatial mean | No spatial tensors or pooling. |
| Permute/reshape cancellation | Selected direct tiled GDN heads, flat chunk V, head-major chunk output, grouped Q/K state broadcasting, KDA tuple return and direct prefill history tail. Expert-major decode adaptation passes but is slower (`grouped_trials129`, `finalist_trials129`, `kda_final4096`, `residual_linear4096`). |
| Conv bias/scale/BN folding | Checkpoint FIR has no bias, channel scale or batch norm to fold. |
| Conv activation | Selected dedicated KDA causal convolution+SiLU, internal32-row decode padding, exact logical history update. |
| Linear activation | Existing SiLU epilogue assessed; shared sigmoid linear epilogue passes but loses to binary sigmoid-input multiply. Shared and routed SwiGLU epilogues selected. |
| Binary input activation | Selected attention/shared sigmoid and GDN decode FP32 SiLU input activation. |
| Linear bias | No projection bias. GDN dt_bias is already absorbed with softplus into one binary post-activation; moving it across the BF16 mega-projection boundary would not remove another dispatch and changes rounding. |
| Matmul transpose argument | Projection weights transpose only during construction; SDPA already owns runtime QK transposition. |
| Slice into operands | All projection columns are consumed. KDA's concat followed by exact Q/K/V slices was removed directly. |
| Padding into consumers | Existing SDPA causal/page contract retained. KDA token padding is internal; arbitrary public logical tails remain supported. No pooling padding chain. |
| Stable softmax / max subtraction | Already native softmax/generalized gate; no remaining explicit stability chain. |
| Reduction+reshape | Compact expert output avoids256-region reduction. Dedicated weighted reduce selected; GDN scalar/group axes and native keepdim reductions retained. |
| Sum-to-mean | No spatial mean. The L2 sum is expressed by RMSNorm with correctly scaled epsilon, measured as its own candidate. |
| Fused paired KV update | Selected disjoint K/V shard grids and native paired paged update; cache dtype/page ownership unchanged. |
| Native recurrent GDN | Reassessed headline, source, and final surrounding graph with real B32 requests. `native_positions_final_candidate.json` fails at minimum decode PCC0.9884342; `positions_final/linear.json` passes0.9977745. TF32 source-register recurrence is unsafe on long unaligned tails; FP32 grouped/ternary recurrence selected. |
| FP32 ternary/state in-place | Selected grouped addcmul into persistent FP32 state; removes outer-product intermediate and state copy. FIR addcmul and consolidated RM/F32 tilize also pass but lose to KDA convolution. |
| Fused exact chunk GDN | Selected `ChunkGdnFusedProgramConfig`, flat V and head-major output; real equivalence passes (`finalist_trials129`). |
| Fused routed FFN | Fixed native region-capacity adapter; compact8 output-slot mapping plus direct top8 routing and weighted reduction wins. All256 weight descriptors remain resident, only8 experts execute (`ffn_repaired129`, `compact_trials129`, `compact_trials4096`). |
| Routed prefill SwiGLU | Selected tile-preserving expert weight packing and8 groups of32 `minimal_matmul_split` outputs. Unsplit and single-call32-split alternatives pass but are slower (`split_full4096`, `split_linear4096`, `split32_*`). |
| GDN norm + gate | Selected prefill-only KDA sigmoid-gated RMS followed by FP32 multiply by raw Z, giving SiLU gating. Token padding adaptation passes but has no decode gain and lowers PCC, so decode retains the faster precise existing decomposition (`kda_final4096`). |
| Projection + residual epilogue | Adapted native dit_minimal_matmul_addcmul_fused with retained first residual and BF16-upcast full-attention weight. Repaired missing token residual. GDN passes but is slower; full attention is slower and fails .995 on headline (`residual_*4096`). |
| Shared-down + sigmoid gate + routed add | Reviewer-required adapted `dit_minimal_matmul_addcmul_fused`, BF16-upcast down weight and expanded sigmoid gate: linear passes but is slower1279.977 vs1259.137us; full slower944.960 vs924.030us and fails PCC0.9945282 (`shared_epilogue4096`). |
| Residual add + RMSNorm | Residual is consumed again by final layer add. Available norm APIs return only normalized output, so they cannot eliminate the residual materialization without recomputing it or changing its BF16 boundary. No dual-output matching op found. |

Search roots include operation validators/factories/kernels and tests under
`ttnn/cpp/ttnn/operations`, `tests/ttnn`, common MoE modules, Blackhole Qwen36,
Qwen38, DeepSeek prefill/gating, and the experimental gated attention/GDN model.
`AUTODEBUG.md`, `ffn_hypothesis.md`, and
`triage/prefill_split/AUTOTRIAGE.md` record independent source findings and exact
contracts. The 256-output split overflow was repaired inside the permitted
Python adapter; no shared C++ kernel was changed. General matmul geometry,
precision sweeps, multi-device and model assembly are separate user-excluded
stages, not silently claimed here.

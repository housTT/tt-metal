# Graph-fusion exhaustion assessment

This is the final pattern inventory required by the `graph-fusing` skill. Each
applicable rewrite was checked on real GPT-OSS 120B tensor shapes. A candidate
was retained only when it preserved the functional-stage contract and improved
measured latency; op count or topology alone was not an acceptance criterion.

## Delivered graph

Prefill uses a device-local count/sort/regroup around Blackhole's dedicated
`unified_routed_expert_moe`, followed by `post_combine_reduce`. Decode selects
between two fused graphs:

- Layers 0-13, 21, 23-27, 29, 31, and 33-35 use public FullLocal
  `moe_compute` plus `deepseek_moe_fast_reduce_nc_fused` when the caller
  supplies checkpoint revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`,
  the hardware exposes matmul ring size 8, and configured maximum batch is 1
  or 2. Setup uses public host BF4 quantization, which was materially more
  accurate than device BF16-to-BF4 typecast.
- All other layers, revisions, ring sizes, and batch configurations use the
  compact top-4 indexed graph: setup-packed gate/up weights, indexed sparse
  gate/up and down matmuls, indexed biases, exact OAI-SwiGLU, and weighted
  reduction.

The 25-layer FullLocal set is the intersection of real-checkpoint batch-1 and
batch-2 direct-fusion sweeps at PCC >= 0.995. Marginal layers were tested with
an independent hidden-state seed; layer 28 was removed after the second seed
reached only 0.994515765. The complete numeric ledger is
`candidates/full_local_moe_compute/host_quant_layer_qualification.csv`.

At sequence 128, the 500-replay whole-decoder A/B measured FullLocal versus the
best indexed baseline at 0.719116 versus 1.368594 ms for sliding attention and
0.728915 versus 1.367084 ms for full attention: 47.46% and 46.68% lower traced
decode wall latency. The final Tracy/`tt-perf-report` operation totals are under
`tracy/final_full_local/`.

## Dedicated fused operations

| Skill pattern / discovered op | Assessment | Decision |
| --- | --- | --- |
| Elementwise activation recognition | Prefill folds expert gate/up biases, clamp, OAI-SwiGLU, down projection, and down bias into `UnifiedRoutedExpertFfnDeviceOperation`. FullLocal decode uses the public `SWIGLU` mode implementing the GPT-OSS OAI activation. | Applied in both final fused paths. |
| Softmax | The router already calls dedicated `ttnn.softmax`; no exp/sum spelling exists. | Already fused. |
| RMSNorm | Both layer norms already use the GPT-OSS dedicated RMSNorm module/`LayerNormDeviceOperation`. | Already fused. |
| Distributed RMSNorm | This stage intentionally retains the functional `1x1` device mesh. | Not applicable; multichip stage only. |
| SDPA | Shared attention already uses dedicated prefill and decode SDPA operations with paged cache updates. | Already fused. |
| Split QKV / head transforms | QKV is one packed projection and the graph already uses dedicated create/concat-head operations in both modes. | Already fused. |
| RoPE | Q and K use dedicated Llama rotary operations. | Already fused. |
| TopK | The router uses `TopKDeviceOperation`. `topk_router_gpt` requires 12 DRAM-aligned cores, but the P150-class Blackhole exposes 8 for this contract. | Current dedicated TopK retained; alternate unavailable. |
| Unified routed-expert MoE | Direct prefill A/B at M=192 produced functional-to-fused PCC 0.999770207 and 205.288-to-32.829 ms warmed wall time (6.253x). Its Tracy sums were 205.203-to-32.623 ms with zero host ops. | Applied in prefill. |
| FullLocal `moe_compute` | The first device-typecast candidate was rejected at PCC 0.980417. Investigation found and fixed a score-axis harness bug, integrated exact score reduction, and then swept public host quantization across every real layer at batch 1 and 2. Twenty-five layers passed >=0.995, including second-seed qualification of marginal layers. | Applied only behind exact revision/layer/ring/batch gates; indexed remains the correctness fallback. |
| Fused FullLocal score reduction | `deepseek_moe_fast_reduce_nc_fused` matched primitive transpose/multiply/sum at PCC >0.999996 and was faster at batch 1, 2, and 32. | Applied. |
| Generalized MoE gate | Trace capture fails because the op requires unsupported output-buffer writes. | Rejected: trace-contract violation. |
| DeepSeek dispatch/combine | Source requires fabric neighbors even on `1x1`; a focused `FABRIC_1D` attempt timed out on a remote handshake. | Rejected; device-local regroup applied. |
| Unified routed-expert kernel for decode | E=128/M=64 decode measured 36.565 ms plus mapping overhead, far slower than sparse decode. | Rejected on latency. |
| GPT MoE fused op | `moe_gpt` does not support the three GPT-OSS per-expert bias tensors. | Rejected on semantics. |

## Structural and algebraic rewrites

| Skill pattern | Assessment | Decision |
| --- | --- | --- |
| RepVGG conv-sum | Decoder has no convolution. | Not applicable. |
| Shared-LHS matmul | Packing gate/up changed the correct decode MLP from 1.755 ms/46 ops to 1.537 ms/42 ops. Compact indexed top-4 projection then changed 1.531 to 0.993 ms at PCC 1.0 to packed and 0.999708177 to Torch. | Applied in indexed fallback. |
| Spatial mean | No spatial reduction exists. | Not applicable. |
| Permute-reshape-permute identity | Attention uses dedicated head transforms. Remaining reshape operations are views or fixed sparse/embedding contracts. | Exhausted. |
| Local routing rewrite | Masked bincount, aligned offsets, sort, gather/scatter, and inverse mapping replace two invalid fabric collectives while preserving exact top-4 slot order. | Applied in prefill. |
| FullLocal score-axis rewrite | Public FullLocal output is expert-major `[K,T,H]`, while router scores are `[1,T,K]`. The direct oracle explicitly transposes scores; the delivered fused reducer consumes the explicit token-major score tensor and performs that axis mapping internally. The earlier reshape-only oracle was invalid and its measurements were discarded. | Applied and regression-tested. |
| Compact weighted reduction | For indexed decode, primitive mul/sum was 0.9975 ms/replay versus 1.0525 for padded `post_combine_reduce` and 1.0075 for ordinary `fast_reduce_nc`. For FullLocal expert-major output, the dedicated fused score reducer is faster and eliminates that primitive chain. | Path-specific fastest choices retained. |
| FullLocal output placement | A reducer microbenchmark slightly favored L1, but the two-layer end-to-end A/B measured 1.443122 ms for L1 versus 1.440433 ms for DRAM, with bitwise-identical output. | DRAM output retained on end-to-end latency. |
| Dummy index elimination | Zero and routed indices produced identical reduction output, but the required TILE-to-one-core row-major sharded fold is invalid for this tensor shape. The existing DRAM indices are already required upstream. | Rejected; no runtime operation removed. |
| Batch widening | Direct FullLocal was exact through batch 8 for sampled layers, but whole-decoder batch 16/32 replay was nondeterministic. Batch 9 also exceeded a shared-attention contract. | FullLocal capped at configured max batch 2; indexed preserves advertised batch 32. |

## Adjacent-operation merging

| Skill pattern | Assessment | Decision |
| --- | --- | --- |
| Conv/bias/activation, BN+conv, pad+pool | No convolution, batch norm, or pooling exists. | Not applicable. |
| Matmul/linear + activation | Prefill and qualified decode use dedicated MoE kernels. Indexed sparse matmul exposes no OAI-SwiGLU epilogue, and the packed halves must be split for exact math. | Applied where the API permits; indexed split retained. |
| Input activation into binary | OAI-SwiGLU is clamp, scaled sigmoid, shifted gate, and multiply across two semantic halves, not an available unary input activation. | Not applicable. |
| Matmul + bias | Both dedicated MoE paths accept expert biases. Indexed sparse matmul has no bias parameter, so top-4 indexed embeddings are the minimal device operation. Shared-attention bias folds were profiled but require edits outside the authorized stage; decode o-proj was also slower. | Applied where available; indexed embeddings retained. |
| Transpose/permute + matmul | Weight transposition and packing happen only at construction. | Already folded. |
| Slice after matmul | Narrowing packed gate/up would restore two projections and undo the measured shared-LHS win. | Rejected by A/B. |
| Pad into consumer | Private 64-row prefill padding is mandated by the routed-expert kernel, which has no logical-length argument. FullLocal H=2880 output requires generic tilize padding because the DeepSeek post-combine tilizer does not accept this width. | Required adapters retained; public lengths remain arbitrary. |
| Numeric-stable softmax | Router already uses the dedicated numeric softmax path. | Already fused. |
| Reduction + reshape | Applicable indexed and FullLocal reducers were measured directly, including exact trace replay. | Fastest correct choice applied per path. |
| Scaled sum to mean | Top-4 scores are nonuniform probabilities. | Not applicable. |
| Decode RoPE reshape | Shared attention already uses dedicated decode rotary/head layouts. | Already fused. |

## Final decision

Every skill pattern and every repository MoE/router/reducer candidate found by
source search was applied, already present, measured and rejected, structurally
inapplicable, physically unavailable, or outside the authorized file boundary.
The final allowlisted FullLocal graph beats the best correct indexed candidate;
the indexed graph remains the fastest correct path where BF4 precision or
hardware/batch qualification is insufficient. No remaining correct decoder
fusion is available within current TTNN single-device capabilities.

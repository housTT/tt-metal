# Fused-decoder graph inventory

This is the final repeated inventory required by the `graph-fusing` skill.
Rows describe logical runtime regions; the six signpost-filtered
`tracy/*/*_perf_report.csv` files are the authoritative device-op tables.

## Primitive graph to retained graph

| Region | Functional graph | Retained fused graph | Decision |
| --- | --- | --- | --- |
| Hyper mix | two same-LHS projections, scales, four-stream sum/scale | packed down/inject projection, setup-folded scales, `mean` | Removes peer matmul and scaled-sum dispatch. |
| Hyper inject | sigmoid, multiply, scalar multiply by 2, residual add | sigmoid input activation in binary multiply, then scalar `mac` | Exact one-dispatch multiply+add. Folding 2 into output weights materially reduced layer-0 decode PCC to 0.99700934 and was not uniformly faster. |
| MoE input | router/shared gate/up/scalar peer matmuls | one packed projection and required slices | One LHS read/matmul. |
| Router | full-width softmax, top-k, sum, divide | top-k logits, softmax selected 10, scatter | Global softmax normalizer cancels exactly for unbiased normalized top-k; removes full-width softmax and sum/divide. |
| Routed gate/up | two sparse matmuls | packed sparse gate/up, binary SwiGLU | Shared LHS plus activation fold. |
| Routed down | four group calls, scans, slices, post-down routing multiply, concat | routing weights applied at intermediate width, one group-major A-sparse/B-dense call, fast expert reduction | Removes three sparse dispatches, output-width multiplies, slices/scans, and concat. |
| Shared expert | SwiGLU, down, output-width sigmoid multiply | sigmoid scalar applied at intermediate width, then down | Exact linearity rewrite reduces eltwise width. |
| GDN projection | qkv/beta/decay/z peers, add decay bias | packed FP32 qkv/beta/decay `linear` with zero/zero/bias packed bias; separate BF16 z | Bias add folded. Packing z plus output typecast was correct but slower. |
| GDN prefill FIR | four multiply/add taps, SiLU, split QKV | `qkv_causal_conv1d_silu`, chunk 640 | Promoted: about 1 ms faster; BF16 boundary remains above PCC bar. |
| GDN decode FIR | concat/slice rolling history, multiply/add taps | persistent split taps and ternary MAC chain | Eliminates per-token history concat/slices. |
| GDN recurrence | spelled-out norms/scales/exp/update | fused chunk kernel for prefill; RMSNorm weights fold q/k scales, ternary arithmetic and persistent state for decode | Setup-folded exact scales and fewer dispatches. |
| GDN epilogue | RMSNorm, sigmoid, multiply | retained primitive epilogue | KDA fused epilogue is exact and faster for T=128, but makes later decode address/lifetime-sensitive: the valid caller-retained-input path fails layer-1 PCC at about 0.86 despite bit-identical saved/prepared state. AutoFix exhausted in-scope lifetime/layout/cache variants. |
| PLE projection/state | key/value peers; concat/slice dilation history | packed key/value; six split decode taps | Shared LHS and no per-token history concat/slices. |
| PLE score | elementwise product + hidden reduction | retained reduction | Exact batched-matmul adaptation passed PCC but regressed prefill to 32.126572 ms. |
| PLE cleanup | dead decode reshape and explicit SiLU | dead reshape removed; consumer activation fold | Removes unused/adjacent dispatches. |
| QSA projection | six same-LHS projections | packed q/k/v/gate/index-q/index-k | One LHS read/matmul. |
| QSA heads/RoPE | generic split, legacy rotary, inverse V/Q/head TMs | dedicated prefill/decode split, HF partial RoPE, direct token-major SDPA output reshape | Removes inverse V permutes, decode Q permute/reshape, and concat-heads entirely. |
| QSA K/V update | separate K and V paged updates | `paged_fused_update_cache`; index update remains separate | One paired cache operation. Batch-one V stays sharded to the update; batch>1 converts before pad due TTNN pad contract. |
| Compressed index keys | every query gathers full raw index cache, mean/norm/RoPE | persistent normalized/RoPE-applied paged compressed-key cache updated per four-token group | Removes repeated full-context preprocessing. Static block cos/sin is cached once. |
| Compressed addresses | per-call shifts/masks on static ids, repeats | setup-precomputed virtual page/in-page ids and broadcast fast paths | Removes two static runtime bitwise ops and avoidable decode repeats; batch fallbacks preserve legality. |
| Index score | transpose, matmul, scale, ReLU, head sum | transpose/scale/activation folded into norm/matmul; head sum retained | Dedicated `indexer_score_dsa` exact adaptation was slower. |
| Selected tokens | top-k, offset arithmetic, redundant valid mask multiply | top-k, integer `addcmul`, padding already supplies zero validity | Removes redundant multiply. |
| QSA gather | packed two-head gather, reshape padded head axis, repeat to 24 | independent per-KV-head gather, native 24:2 GQA | Removes 24-head expansion and tiled dimension-2 padding. |
| QSA attention | K transpose, qk, scale, mask, softmax, pv | prefill GQA SDPA; dedicated decode SDPA | Streaming fused attention. Decode expands the required head mask only. |
| QSA output | permute to head-major, concatenate heads, sigmoid/multiply, projection | reshape token-major SDPA result directly; sigmoid input activation, projection | Cancels the inverse movement and concat-heads dispatch. |

## Dedicated-operation search

| Operation/family | Source-contract assessment and measured disposition |
| --- | --- |
| `qkv_causal_conv1d_silu` | Legal for prefill T=128 and 10240 channels after a BF16 row-major boundary. Chunks 320/640/1280 were measured; 640 was fastest/stable and promoted. Decode T=1 is binding-illegal. |
| `sigmoid_gated_rms_norm` | Legal prefill-only adaptation exactly replaces RMSNorm+sigmoid+multiply and saves about 0.60/0.65 ms on L0/L1 prefill. The standard retained-caller-input gate fails L1 eager/traced decode at 0.86030912/0.86216938 even though saved and prepared recurrent/conv/PLE states are bit-identical and the kernel address audit is bounded. Throwaway decode inputs pass at 0.99988902, proving allocator/address sensitivity. L1 output, deferred KDA tensor lifetimes and program-cache clearing did not repair it; rejected after AutoFix as a current TTNN composition/runtime limitation. |
| `indexer_score_dsa` | Exact decode adaptation used 31 zero rows; exact prefill required four residue-class calls. PCC passed, but 43.660651/5.588385 ms lost to 43.666625/5.540206. Rejected. |
| `scaled_dot_product_attention_decode` | Explicit noncausal mask expanded to 24 heads; K chunk 32. Promoted after PCC and timing. |
| `sparse_sdpa` | Math mapping is exact with two padded 12-to-32-head calls and combined `[V,K]` rows. Hard scoped blocker: it requires a persistent row-major combined KV cache, while arbitrary paged fill/update validates tiled caches; no row-major paged writer or tiled-cache sparse reader exists. A duplicate cache cannot preserve arbitrary page-table writes from this file. |
| `rotary_embedding_hf` | Exact Qwen-family partial-RoPE op; promoted for prefill/decode. Fused QK rotary is incompatible with this 64/256 partial convention. |
| `paged_fused_update_cache` | Legal K/V pair update; promoted. The compressed index cache remains a distinct update. |
| `nlp_create_qkv_heads_decode` | Promoted. Maximal sharded Q/K/V propagation was tested: sharded RMSNorm rejected the splitter layout; the retained batch-one path keeps V sharded but legally bridges Q/K. Batch>1 uses the tested V fallback. |
| `nlp_concat_heads_decode` | Tested, then superseded by the stronger exact cancellation: SDPA is already token-major at the epilogue boundary, so direct reshape eliminates concat-heads and its preceding inverse permute. |
| `generalized_moe_gate` | Binding/kernel supports at most 512 experts but k only in `{4,6,8}`; Qwen requires k=10. Hard mismatch. |
| `topk_router_gpt` | Hard-coded for 128 experts and decode batch 32; cannot represent 512 experts/all stage modes. |
| DeepSeek fused MoE reduce/unified routed expert | Requires pre-dispatched compact/all-to-all expert metadata absent from this single-chip top-10 graph. |
| `minimal_matmul(fuse_swiglu=True)` | Structurally dominated for shared expert: it splits the retained packed router/gate/up/scalar shared-LHS projection into two matmuls/input reads. `minimal_matmul_split` requires equal output chunks, which these widths are not. |
| KDA `reduce_affine_transforms` | Lower-level affine-summary primitive, not a replacement for the already dedicated full chunk gated-delta-rule recurrence. |

## Shared-LHS and sparse alternatives

- Every same-input peer projection in this decoder was considered. Packing is
  retained for hyperconnection, MoE, routed gate/up, GDN qkv/beta/decay, PLE,
  and QSA. GDN z-in-FP32 was measured with a typecast and lost; it remains a
  separate BF16 projection.
- Indexed sparse MoE was implemented with a fixed 320-id group list derived
  on device (`32 * top10`), compact-A/B-sparse down, and zero-weight fillers.
  It preserved PCC but measured 78.541340/14.560467 ms on layer 3, so the one
  group-major A-sparse/B-dense down is retained.
- Group routing/shared gates were moved before down projections. This exact
  linearity rewrite is retained and its full three-layer PCC/timing transcript
  is `candidates/moe_pre_down_gate_placement.log`.

## Structural and folding checklist

| Skill pattern | Final disposition |
| --- | --- |
| Elementwise activation fusion | Applied for sigmoid, SiLU, and ReLU consumers. Unsafe GDN output fold rejected on PCC. |
| Softmax/TopK | QSA softmax is inside SDPA; MoE uses top-k then selected softmax. The fused generalized gate has a hard k mismatch. |
| RMSNorm | Dedicated TTNN RMSNorm retained; q/k recurrence scales folded into setup weights. Distributed RMSNorm is not applicable to single chip. |
| Split QKV/heads, RoPE, SDPA | Applied with dedicated prefill/decode ops and exact HF partial RoPE. |
| Shared-LHS matmul | Exhausted as described above, including empirical z-packing rejection. |
| RepVGG conv-sum, Conv2d/BN, pad-pool | Not applicable: no Conv2d, BatchNorm, or pool graph exists. GDN FIR dedicated op was assessed separately. |
| Permute-reshape-permute cancellation | Applied to QSA V, decode Q, and attention epilogue. Remaining TMs change logical order or satisfy a target binding. |
| Persistent state | Applied to GDN FIR/recurrent state, PLE taps, compressed index keys, and static block RoPE/address data. |
| Matmul bias/linear | Applied to packed GDN decay bias. Other packed outputs have no common bias. |
| Matmul activation / binary input activation | Applied wherever all packed outputs share semantics; mixed-output packed projections cannot accept one common activation. |
| Transpose into matmul | Applied to index score. |
| Constant scale/mean | Hyper scales and q/k norm scales folded; four-stream scaled sum replaced with mean. Output-weight factor-2 fold was numerically/materially inferior, so scalar MAC retained. |
| Slice after matmul | Packed slices feed distinct consumers. Removing them recreates peer matmuls; no profitable push-through remains. |
| Reduction+reshape | `keepdim`/mean used where contracts allow. PLE batched dot was slower. |
| Numeric-stable softmax rewrite | No spelled-out max/subtract graph remains. Router normalization uses selected softmax; attention uses SDPA. |

## Final movement audit

- The delivered class contains no `torch`, `from_torch`, `to_torch`,
  `as_tensor`, or host fallback. Tests make those APIs raise inside measured
  calls and inspect the fused source.
- Setup packing and placement occur in `from_state_dict` or
  `prepare_decode_state`, outside timed windows.
- No explicit host transfer appears in any filtered Tracy window. Layer-0/1
  layout conversions are binding boundaries for KDA, sparse-matmul metadata,
  and output formats. Layer-3 prefill has no reshard row.
- Layer-3 decode retains one 0.34 us `ReshardDeviceOperation`, three 2.04 us
  total interleaved-to-sharded rows, and two 1.38 us total
  sharded-to-interleaved rows. These are the minimum legal Q/K
  norm/RoPE/decode-SDPA bridges after the maximal-sharded candidate failed the
  RMSNorm memory-layout validator; V remains sharded on the batch-one cache
  path. Removing the bridge is not a legal graph optimization.
- Remaining tilize/untilize rows are owned by required TTNN op contracts
  (KDA row-major input, sparse metadata, gather, top-k/scatter, cache, or SDPA)
  and do not round-trip through the host.

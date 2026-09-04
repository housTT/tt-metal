# Final fused-decoder graph audit

The four final integrated captures account for every measured device operation.
The rescan found two additional applicable rewrites: GDN outer-product transpose
fusion is now selected after a supported adaptation, a profile proving dispatch
removal and matched timing; GDN decode head concatenation is correct but slower
and rejected. **No material graph-fusion candidate remains unresolved in this
rescan.** Refreshed short, native-context, Watcher and four-mode profiler gates
pass on the final source. Final independent review and commit remain with the
coordinator; this report does not declare the stage complete.

## Evidence and reading conventions

The complete sequences and four measurements below belong to the final
[fused_decoder.py](../../tt/fused_decoder.py), SHA256
`18d59502e7e168e584e58762396b9cc61eae6de304045542d15bbcc070f9b11d`, with inherited
orchestration from [functional_decoder.py](../../tt/functional_decoder.py).
[performance.json](performance.json) and adjacent `*_provenance.json` record
report/raw hashes, coverage and measurement commands. The four
`profile_final_*_v2.provenance.json` files under [logs](logs/) pin this same
runtime hash. All four log hashes were checked against provenance.

For historical comparison, [performance_v1.json](performance_v1.json) preserves
the initial `3cf319b63a169574a20d318b7cdc78ecff03aba5498e07de862ec1de5fb84083`
source's measurements, and each Tracy directory retains `v1_*` reports/raw CSVs,
including [the old linear-decode sequence](tracy/linear_attention/v1_decode_ops.csv.gz).
Comparing final raw operation sequences against those archives confirms that
the only operation-count change is removal of the old linear-decode ordinal-35
outer-product transpose: 50→49. The tables below use **current** ordinals.

| Forward | Raw operations / derived report | Operations per forward | Kernel sum per forward | Measured HF PCC |
|---|---|---:|---:|---:|
| Linear prefill | [raw](tracy/linear_attention/prefill_ops.csv.gz), [report](tracy/linear_attention/prefill_perf_report.csv) | 120 | 25.647288 ms | 0.99951420 |
| Linear decode | [raw](tracy/linear_attention/decode_ops.csv.gz), [report](tracy/linear_attention/decode_perf_report.csv) | 49 | 1.4412645 ms | 0.99976799 |
| Full prefill | [raw](tracy/full_attention/prefill_ops.csv.gz), [report](tracy/full_attention/prefill_perf_report.csv) | 33 | 23.002021 ms | 0.99949588 |
| Full decode | [raw](tracy/full_attention/decode_ops.csv.gz), [report](tracy/full_attention/decode_perf_report.csv) | 36 | 1.20938125 ms | 0.99889471 |

All captures use one Blackhole chip on physical P300c hardware, topology name
`p150`. Prefill measures one B1/T2048 forward between `PERF_PREFILL` markers.
Decode prefills 128 tokens, uses absolute position 128 (129 attended tokens),
and measures four trace replays, sessions 5–8, between `PERF_DECODE` markers.
The replay counts are 196 and 144 raw rows, respectively. The reporter confirms
that each replay has the warm template's operation IDs and no dropped markers.
These numbers sum device kernel durations; they exclude dispatch gaps and are
not the paired wall medians at decode position 2048 in [README](README.md).

The following ordinals are positions inside the measured forward, rather than
global call IDs. Decode ordering is taken from session 5 and checked against
sessions 6–8. Ordered lists in one row expand to consecutive operations; the
eight FIR groups and three history writes have explicit ordinal formulas, so
every operation is represented once. `MM`, `RMS`, `binary`, `unary`, `view`, and
`transpose` abbreviate the corresponding Matmul, LayerNorm, BinaryNg, Unary,
ReshapeView, and Transpose device operations. A recorded `ReshapeView` can run a
movement kernel; its name is not evidence of a free metadata operation.

Unless a row says otherwise, tensors are tiled, interleaved DRAM, BF16. `RM`
means row-major; `HS` means height-sharded L1. Shapes state the runtime's logical
contract: some cached raw operation metadata repeats an earlier shape for a
reused program, so those stale shape strings are not used to infer a new tensor
layout. Operation order, attributes, dtypes and memory transitions are checked
against source. The table covers the captured B1 forwards; batch subgroups,
partial-tile masks, continuation alignment and public long-context chunk loops
add conditionally required operations outside this aligned capture.

## Complete linear-prefill sequence

Here `X=[B,T,4096]`, Q and K have 16 heads of width 128, V and the recurrent
state have 32 heads, and the temporal FIR has four taps. `P` is packed QKVAB
width 8256; the separate Z width is 4096. The prefill state is FP32
`[B,32,128,128]` and the three convolution-history rows are BF16 width 8192.

| Ordinal | Operation(s), in execution order | Inputs, output and movement |
|---|---|---|
| 1 | RMS | X and entry norm weight → normalized X. |
| 2 | MM | Normalized X × packed QKVAB weight → P. |
| 3–5 | QKV slice; A slice; B slice | P → widths 8192, 32, 32. |
| 6 | MM | Normalized X × Z weight → raw BF16 Z; no prefill SiLU epilogue. |
| 7–9 | UntilizeWithUnpadding ×3 | Three persistent history tiles → three real RM rows. |
| 10 | Concat | RM history rows → `[B,3,8192]`. |
| 11 | Untilize | QKV → RM `[B,T,8192]`. |
| 12 | Concat | History and QKV → RM `[B,T+3,8192]`, once for all channel groups. |
| 13+7g, g=0…7 | RM slice | Select group g's 1024 contiguous channels from the shared input. |
| 14+7g | InterleavedToSharded | RM DRAM group → RM HS. |
| 15+7g | Halo | HS group → HS temporal halo buffer. |
| 16+7g | Move | Halo allocation relocated in L1 by the ordinary-conv wrapper. |
| 17+7g | Conv2d | Four-tap depthwise Conv1d implemented as Conv2d; RM HS input → tiled HS BF16 output, real length T. |
| 18+7g | ShardedToInterleaved | Tiled group output → DRAM. |
| 19+7g | Unary SiLU | BF16 convolution result → BF16 activated group. The last group ends at ordinal 68. |
| 69–71 | Q concat; K concat; V concat | Join 2/2/4 activated channel groups → flat widths 2048/2048/4096. |
| 72 | RM slice | Shared pre-convolution input → last three real logical rows, excluding prompt padding. |
| 73–74 | Sigmoid; typecast | B projection → BF16 sigmoid → FP32 beta. |
| 75–78 | Typecast; bias add; softplus; A-negative multiply | A → FP32 before dt_bias; standalone FP32 softplus; multiply by setup A-negative → g. |
| 79–83 | UntilizeCodegen; RM view; TilizeWithValPadding; RMS; scale | Flat Q → `[B,T,16,128]`; pad physical head rows to 32; functional BF16 L2 norm, including BF16 `1/sqrt(128)` product. |
| 84–88 | Same five operations for K | Flat K → independently normalized rank-four K with the same rounding. |
| 89–90 | Q transpose; K transpose | Token/head axes → head-major input for chunk preparation. |
| 91–92 | g transpose; beta transpose | FP32 gate axes moved to the chunk interface. |
| 93–94 | RepeatInterleave ×2 | Normalized Q/K heads 16 → 32 value-head groups. |
| 95 | Q scale | Additional query scale used by the functional chunk recurrence. |
| 96–97 | g view; beta view | FP32 gates repacked into the chunk interface's head-major shapes. |
| 98 | ChunkGdnPrepOperation | Q/K/V, g/beta, initial FP32 state and setup triangular constants → FP32 prepared chunk data. |
| 99 | ChunkGdnScanOperation | Prepared chunks and initial state → FP32 head-major core and final state. |
| 100 | Copy | Final FP32 state → fixed persistent recurrent-state allocation. |
| 101+3r, r=0…2 | RM slice | Real three-row tail → history row r. |
| 102+3r | TilizeWithValPadding | RM row → tiled BF16 row. |
| 103+3r | Copy | Tiled row → fixed history slot r. The last write is ordinal 109. |
| 110 | SigmoidGatedRmsNormOperation | FP32 head-major core, raw BF16 Z and norm vector → flat FP32 norm × sigmoid(Z). |
| 111 | Binary multiply | Multiply by raw Z to complete SiLU gating, FP32 result. |
| 112 | MM | FP32 gated core × BF16 output weight → FP32 mixer output. |
| 113 | Residual add | X and mixer output → BF16 residual stream. |
| 114 | RMS | Residual stream and MLP norm weight → normalized MLP input. |
| 115 | MM | MLP input × packed gate/up weight → BF16 width 24576. |
| 116–117 | Gate slice; up slice | Packed output → two width-12288 tensors. |
| 118 | Binary SiLU-input multiply | SiLU(gate) × up → BF16. |
| 119 | MM | Activated MLP × down weight → BF16 width 4096. |
| 120 | Residual add | MLP output plus ordinal-113 residual → BF16 block output. |

At this T2048 capture, normalized Q/K stay in DRAM. The integrated short-chunk
path retains L1 normalization results when `batch * physical_seq <= 512` and
feeds them directly to the chunk reader; it does not introduce an L1→DRAM copy.

## Complete linear-decode sequence

| Ordinal | Operation(s), in execution order | Inputs, output and movement |
|---|---|---|
| 1–2 | RMS; packed MM | X `[B,1,4096]` → normalized X → QKVAB width 8256. |
| 3–5 | QKV slice; A slice; B slice | Packed projection → BF16 widths 8192, 32, 32. |
| 6 | MM with SiLU epilogue | Normalized X × separate Z weight → already activated BF16 Z. |
| 7 | Binary multiply | Current QKV × FIR tap 3. |
| 8–10 | Addcmul ×3 | Accumulator plus history row × tap for taps 0, 1, 2; BF16 boundaries retained. |
| 11 | Unary SiLU | Final BF16 FIR sum → activated QKV. |
| 12–14 | Copy ×3 | Shift persistent history slots 1→0, 2→1, and current unactivated QKV→2. |
| 15 | View | Activated flat QKV → `[B,1,64,128]`. |
| 16 | Transpose | Head axis → `[B,64,1,128]`. |
| 17–18 | V slice; joint QK slice | Last 32 heads form V; first 32 form Q/K. |
| 19 | RMS | One BF16 QK normalization, epsilon `1e-6/128`. |
| 20 | RepeatInterleave | Duplicate the normalized 16 Q and 16 K heads into 32 Q and 32 K value-head groups. |
| 21–22 | Q slice; K slice | Joint normalized result → separate Q and K. |
| 23 | Unary chain | B → sigmoid, explicit BF16 result rounding, FP32 beta output. |
| 24–26 | Bias add; softplus; A-negative multiply | BF16 A plus FP32 bias → FP32; standalone softplus and multiply → g. |
| 27 | Binary Q scale | Q × exact `1/128` → FP32 directly in DRAM. |
| 28 | Unary K chain | Multiply by BF16 coefficient `181/2048`, round product to BF16, emit FP32 K in interleaved L1. |
| 29–30 | Beta view; g view | FP32 `[B,1,32]` → broadcast head scalars `[B,32,1,1]`. |
| 31 | Binary EXP-input multiply | Persistent FP32 state × exp(g), in place. |
| 32 | MM | L1 K row × FP32 state → DRAM state read. |
| 33 | Mixed-dtype subtract | BF16 V minus FP32 state read → FP32 difference, without a separate V cast. |
| 34 | Binary multiply | Difference × tensor beta → FP32 delta. |
| 35 | MM with native A transpose | FP32 L1 K row and DRAM delta row → FP32 outer product in DRAM; whole-matrix reuse program folds the K transpose into matmul. |
| 36 | Binary add | Add outer product into the persistent FP32 state in place. |
| 37 | MM | Scaled FP32 Q × updated state → FP32 head-major core. |
| 38 | RMS | FP32 core and GDN norm weight → FP32 normalized heads. |
| 39–40 | Transpose; view | `[B,32,1,128]` → token-major → flat `[B,1,4096]`. |
| 41 | Binary multiply | Normalized flat output × activated BF16 Z → FP32 gated output. |
| 42 | MM | FP32 gated output × BF16 output weight → FP32 mixer output. |
| 43–44 | Residual add; RMS | BF16 residual stream, then normalized MLP input. |
| 45–46 | Gate MM; up MM | Two separate BF16 width-12288 projections sharing the MLP input. |
| 47 | Binary SiLU-input multiply | SiLU(gate) × up. |
| 48–49 | Down MM; residual add | MLP output → BF16 block output. |

## Complete full-attention prefill sequence

Here Q has 16 heads, K/V have four, each width 256. Only the first 64 channels
receive HF half-split RoPE. Packed QKVG width is 10240. Cache pages contain 64
tokens and both KV caches stay BF16.

| Ordinal | Operation(s), in execution order | Inputs, output and movement |
|---|---|---|
| 1 | RMS | X `[B,T,4096]` → normalized X. |
| 2–3 | Cos slice; sin slice | Setup tiled native-context tables → shared `[1,1,T,64]` rotary rows. |
| 4 | Packed MM | Normalized X × QKVG weight → width 10240. |
| 5–6 | QKV slice; gate slice | Packed output → QKV width 6144 and gate width 4096. |
| 7 | NlpCreateHeads | QKV → Q `[B,16,T,256]`, K/V `[B,4,T,256]`, no key transpose. |
| 8–9 | Q RMS; K RMS | Per-head BF16 normalization. |
| 10–13 | Q rotary slice; RotaryEmbedding; tail slice; concat | Q channels 0:64 → dedicated half-rotation; concatenate untouched 64:256. |
| 14–17 | Same four operations for K | Partial64 RoPE and unchanged 192-channel tail. |
| 18–19 | Page-table slice; batch-index slice | RM integer setup tables → active request/page range. |
| 20–21 | PagedFillCache K; V | Batched BF16 head tensors → paged K/V caches, no Python per-user fill loop. |
| 22 | SDPA | Q, paged caches and page table → causal BF16 attention result. |
| 23 | NLPConcatHeads | Head-major attention → tiled token-major flat output. |
| 24 | Binary SIGMOID-input multiply | Attention output × sigmoid(gate), BF16. |
| 25 | Output MM | Gated attention × output weight. |
| 26–27 | Residual add; RMS | BF16 residual stream, then normalized MLP input. |
| 28 | Packed gate/up MM | MLP input → width 24576. |
| 29–30 | Gate slice; up slice | Two width-12288 fields. |
| 31 | Binary SiLU-input multiply | SiLU(gate) × up. |
| 32–33 | Down MM; residual add | MLP output → BF16 block output. |

## Complete full-attention decode sequence

| Ordinal | Operation(s), in execution order | Inputs, output and movement |
|---|---|---|
| 1–2 | RMS; packed QKVG MM | X `[B,1,4096]` → normalized X → BF16 width 10240. |
| 3 | QKV slice | Packed projection → QKV directly in interleaved L1. |
| 4 | Gate slice | Gate stays in DRAM. |
| 5 | NLPCreateQKVHeadsDecode | L1 QKV → native Q/K/V HS layouts; V shard retained for cache update. |
| 6–7 | ShardedToInterleaved Q; K | Q/K → DRAM for supported per-head RMSNorm. |
| 8–9 | Q RMS; K RMS | BF16 normalized native heads. |
| 10–11 | Cos embedding; TilizeWithValPadding | RM position indices and RM cos table → RM rows → tiled BF16 rows. |
| 12–13 | Sin embedding; TilizeWithValPadding | Corresponding sin lookup and tiled rows. |
| 14–15 | Cos transpose; sin transpose | Shared trig rows reshaped to native batch/head axes and emitted directly on rectangular HS grids. |
| 16–19 | Q rotary slice; RotaryEmbeddingHf; tail slice; concat | DRAM Q → partial64 HS; HF RoPE → HS; direct HS tail slice and join. |
| 20–23 | K rotary slice; RotaryEmbeddingHf; tail slice; concat | Same partial64 graph for K on the supported rotary grid. |
| 24 | Reshard K | Move K to the cache-update grid disjoint from the retained V grid. |
| 25 | PagedFusedUpdateCache | HS K/V plus integer current positions/page table → persistent BF16 paged caches in one operation. |
| 26 | SdpaDecode | Native Q shard and paged caches → interleaved DRAM output. |
| 27 | View | Native SDPA output → flat `[B,1,4096]`. |
| 28 | Binary SIGMOID-input multiply | Flat attention × sigmoid(gate). |
| 29 | Output MM | Gated attention × BF16 output weight. |
| 30–31 | Residual add; RMS | BF16 residual stream, then normalized MLP input. |
| 32–33 | Gate MM; up MM | Separate BF16 MLP projections. |
| 34 | Binary SiLU-input multiply | SiLU(gate) × up. |
| 35–36 | Down MM; residual add | MLP output → BF16 block output. |

## Actual numerical policies and largest operations

Generic projection/recurrence matmuls, ordinary depthwise convolution, native
HF decode RoPE and KDA gated norm use HiFi4, `math_approx_mode=False`, and FP32
destination accumulation. Chunk preparation/scan also records HiFi4/FP32
compute. Model weights, ordinary activations, RoPE and page-64 KV are BF16;
linear gates, recurrent state, chunk core and linear mixer output are FP32.
The residual adds explicitly restore the BF16 residual stream. The key unary
chain preserves BF16 scalar/product rounding before emitting FP32; it is not a
change to the cache or model precision policy.

This does **not** mean every specialized kernel has the generic matmul config.
Default RMSNorm and older prefill RotaryEmbedding use their own HiFi4 defaults,
with approximation enabled and FP32 destination disabled. Prefill SDPA uses
HiFi2, approximation disabled, FP32 destination enabled. Decode SDPA's actual
raw attributes are HiFi2, approximation enabled, FP32 destination disabled;
its separate `exp_approx_mode=False` program attribute is also retained.
PagedFusedUpdateCache records LoFi/approximate/non-FP32 compute defaults for a
copy operation; this does not change BF16 cache storage. The audit makes these
distinctions from source and raw attributes rather than assigning every op the
decoder's generic compute-config label.

| Forward | Largest kernels (mean microseconds per forward; share of total) |
|---|---|
| Linear prefill | Packed MLP gate/up 6757.687 (26.35%); MLP down 5584.944 (21.78%); QKVAB 3749.237 (14.62%); GDN output 1988.825 (7.75%); Z 890.215 (3.47%); chunk scan 795.202 (3.10%); chunk prep 780.302 (3.04%). |
| Linear decode | MLP up 253.751 (17.61%); gate 253.379 (17.58%); down 235.802 (16.36%); QKVAB 167.998 (11.66%); GDN output 84.688 (5.88%); Z 83.055 (5.76%); Q×state 60.794 (4.22%). |
| Full prefill | Packed MLP gate/up 6752.809 (29.36%); down 5597.318 (24.33%); QKVG 4873.729 (21.19%); output 1890.161 (8.22%); SDPA 1442.751 (6.27%). |
| Full decode | MLP gate 253.918 (21.00%); up 253.567 (20.97%); down 236.026 (19.52%); QKVG 206.725 (17.09%); output 81.456 (6.74%). |

These costs point mainly to dense weight matmuls after the graph rewrites.
Reporter utilization or `SLOW` labels do not by themselves identify another
fusion: changing general grid geometry, memory policies or dtypes is outside
the requested graph-fusing scope. Supported program changes were investigated
when necessary to express a real activation or transpose fusion.

## Remaining-boundary reconciliation

[patterns.md](patterns.md) records every skill family, original controls,
failed-first-attempt adaptations and links to measured logs. The following
rescan maps the surviving boundaries above to those concrete outcomes. Historic
candidate times establish tested alternatives, not current integrated timing.

| Surviving boundary | Fusion/rewrite assessment and evidence |
|---|---|
| Packed projection slices in all modes | Setup packing already merges shared-input matmuls. The consumers require separate QKV/gate or QKV/A/B fields; they do not accept a packed trailing payload. Slicing a shared-LHS output cannot simply be pushed into its LHS because these are output columns. Separate/full GDN packing variants and mode-specific MLP packing were measured in the ledger. |
| Linear prefill untilizes, history concat and eight channel slices | Ordinary Conv1d consumes RM activations; history and current input are joined once and reused. [Shared FIR](logs/shared_fir_rows_v1.log.gz) and ordinary [512](logs/conv1d_narrow_v1.log.gz)/[1024](logs/conv1d_1024_v1.log.gz) adaptations are tested. The 2048-channel attempt exhausted L1 selection; 1024 is the selected supported geometry. Removing necessary RM history alignment would alter the four-tap temporal graph. |
| Conv I2S→halo→move→Conv→S2I | These are the selected ordinary convolution's layout contract. The DRAM-slicing wrapper forcibly sets `deallocate_activation=True` and `reallocate_halo_output=True`, then the halo path calls move: [conv2d.cpp](../../../../../ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp), lines 297 and 681–682. A model-level flag does not remove this move. Replacing the internal allocation/halo implementation is a kernel change, not a hidden free neighbor fold. |
| Conv SiLU and Q/K/V concatenations | Ordinary depthwise compute does not execute the generic activation macro. Dedicated KDA conv+SiLU was repaired and localized, but [B32 isolation](logs/hybrid_conv_only_batch32_trace.log.gz) fails HF PCC 0.99257131; changing FP32 destination does not restore functional rounding. Moving SiLU after the concatenations is a valid tested graph rewrite, [post_concat_silu](logs/post_concat_silu_v1.log.gz): 26.2548 ms versus coherent 26.1943 ms prefill, slower. |
| Linear prefill Q/K untilize→view→tilize→norm/scale | Raw flat internal normalization fails [B32 HF](logs/flat_gdn_batch32_trace.log.gz), PCC 0.99420974. Functional rank-four BF16 Q/K normalization passes the same case. [Joint rank-four prefill norm](logs/joint_prefill_norm_v1.log.gz) is correct but slower after concat/split work; [arithmetic combination](logs/joint_prefill_arithmetic_v1.log.gz) also loses. Their untile/retile removal cannot be claimed independently of those numeric and timing results. |
| Linear prefill Q/K head transposes, repetitions and gate repacks | These are the selected chunk operator's input adapters. Shared decode QK normalization already moves before repetition. Prefill's raw-flat/joint-normalization alternatives above were measured; flat V and head-major output are retained where legal. Dedicated QKV-head split assumes equal K and V head counts, whereas this graph has 16 K and 32 V heads, so it cannot consume the current QKV packing as a drop-in. |
| Short-prefill normalized Q/K handoff | [Matched short-input experiment](logs/short_norm_handoff_v2.log.gz) gives bit-identical 2.465545 ms direct L1 input versus 2.493768 ms L1→DRAM. The selected `batch*physical_seq<=512` rule removes that copy while bounding L1 capacity; T2048 retains DRAM. |
| Prefill typecasts around beta/A and standalone softplus | Decode-only exact-boundary gate folds are selected. Applying them in all modes is tested but [slower in prefill](logs/all_mode_gate_gdn_v1.log.gz). [Softplus-in-bias-add localization](logs/softplus_localization_v1.log.gz) shows lost positive tails below −5; standalone FP32 softplus remains. This is an observed changed numeric contract, not a generic higher-precision recommendation. |
| Persistent recurrent/history copies | These writes preserve fixed addresses used by trace replay and logical continuation. [Chunk GDN API](../../../../../ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.hpp) returns a new final-state tensor; it has no caller-provided persistent-state output. TilizeWithValPadding also lacks an `output_tensor` argument. Direct persistent output would need a new operator output contract; swapping Python references cannot replace these writes inside a captured graph. |
| Decode FIR addcmul→SiLU | `addcmul` exposes no activation epilogue. The legal padded-T32 KDA decode adaptation passes but is [slower](logs/kda_conv_decode_v1.log.gz). The BF16 four-tap accumulation boundaries must remain. |
| Decode QKV view/transpose and Q/K/V slices | One shared head transformation replaces separate field transformations. The unequal K/V-head contract prevents native QKV-head split. Further packing must preserve this checkpoint's 16/16/32 head map; no equivalent lower-dispatch existing head operator was found. |
| Decode query/key/value FP32 transitions | Query scale emits FP32, K's unary chain explicitly reproduces BF16 coefficient/product rounding then emits FP32 L1, and V is accepted by a mixed subtract. [All-cast experiment](logs/all_cast_v1.log.gz) validates these three merges together. The final profiles contain no separate recurrence Q/K/V typecasts. Folding scale into norm weights was independently [slower](logs/norm_weight_gdn_v1.log.gz). |
| Decode beta/g scalar views; subtract→beta multiply | Gates have a different broadcast/padded geometry from projection fields; views perform that required adapter. Beta is a tensor, so a scalar post-unary multiplier cannot express the second operation. Rewriting to addcmul requires computing another scaled operand and retains the same number of binary operations. |
| State decay and rank-one update | exp(g) is already folded into the in-place state multiply. Dedicated KDA recurrence uses a different per-key-gate/intermediate-precision contract. Supported FP32 broadcast rank-one [addcmul adaptation](logs/rank_one_gdn_v1.log.gz) passes but is slower. The K transpose is now genuinely fused by the selected whole-matrix reuse program, detailed below. |
| GDN output normalization/gating | Dedicated KDA gated RMSNorm is selected in prefill with raw Z; the final Z multiply completes SiLU. Decode uses a genuine Z matmul SiLU epilogue and ordinary FP32 RMSNorm. [ModeNorm](logs/mode_norm_gdn_v1.log.gz) tests this coherent combination. Mixed-dtype SiLU-binary folding fails; both supported same-FP32 operand-order adaptations pass but are slower, as linked in the ledger. |
| GDN decode output transpose→view | **New rescan experiment completed:** [ConcatGDN](../../tests/gdn_concat_candidates.py) substitutes ordinary FP32 `nlp_concat_heads` and keeps all other arithmetic. [Real-weight pair](logs/gdn_concat_v1.log.gz) passes with the same recorded PCC values as the initial integrated graph, but decode is 1.5096605 ms versus its 1.4618729 ms, about 48 µs slower. Source factory uses `B*padded_T/32` work blocks, only one at B1, and double-buffers 128 FP32 tiles (1 MiB). This 9B result earns rejection; pinned 35B timings are not used as evidence. |
| Full prefill partial64 slice/rotate/tail/concat | A native partial-width output still needs the untouched tail. Full-width identity-tail RoPE, dedicated HF prefill, joint QK and fused-Llama interleaved-basis adaptations were all measured in [attention diagnosis](AUTODEBUG_attention.md) and the ledger. They are slower; the selected partial64 operator preserves HF half-split semantics and BF16 cache channels. |
| Full decode Q/K S2I before RMS | [LayerNorm validator](../../../../../ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp) rejects height-sharded input. Native head creation and rotary/cache operators require those height shards; a different sharding/RMS geometry is not a free removal of the intervening moves. QKV's former DRAM→L1 copy is already removed by direct slice output. |
| Decode embeddings→tilize and trig transpose | The caller already requests tiled embedding output. [embedding.cpp](../../../../../ttnn/cpp/ttnn/operations/embedding/embedding.cpp) fuses tilization only when the indices' padded last dimension is divisible by 32. At B1–31, RM position inputs require the fallback; B32 can use the fused path. Position padding would add work to the caller's trace-input contract. Packing cos/sin saves lookup/tilize calls but adds two field slices and retains batch-axis transposes, so does not provide a lower-dispatch drop-in. |
| Decode partial64 slices and K reshard | Slices already emit target L1 shards. HF rotary's bounding-rectangle factory and fused-cache disjoint-grid requirement explain the K move. [Direct-K adaptation](logs/direct_k_rope_v1.log.gz) passes and removes that reshard but duplicates trig movement; 1.266676 ms versus native 1.266287 ms is a tie with no measured win. The existing reshard is about 0.34 µs in the final capture. Native partial64 remains selected. |
| Full-attention concat/reshape | Prefill already uses dedicated concat. GQA decode SDPA produces interleaved output; the native reshape is retained. Both supported [ordinary handoff](logs/concat_decode_v1.log.gz) and [sharded handoff](logs/concat_decode_sharded_v1.log.gz) to `nlp_concat_heads_decode` pass but are slower. These full-attention results are distinct from the GDN concat experiment above. |
| Residual add→RMS | The residual sum is also the input to the block's final residual add. Residual-input RMSNorm returns only the normalized tensor, not both norm and sum; it cannot eliminate the add without recomputation or another output contract. |
| MLP shared-input projections and SiLU | Packed gate/up is selected only in prefill. Packing decode passes but is slower. Default matmul `activation='silu'` merely dispatches a unary under the default program; supported explicit-program epilogues and geometry-matched controls were actually tested and the [epilogue](logs/mlp_epilogue_v1.log.gz) loses to its [control](logs/mlp_epilogue_control_v1.log.gz). Current SiLU-input binary already fuses the activation with the required multiply. A packed epilogue cannot activate only the gate half. |

No collectives, exposed softmax primitive chain, sorting, TopK, spatial mean,
batch normalization, convolution bias, RepVGG branches, or unfused reduction
followed by keepdim repair remains in this single-chip decoder graph. Those
skill patterns do not match this workload. The table above covers every
surviving untilize, tilize, copy, move, reshard, transpose and cast in the four
captured forwards.

## Selected native outer-product transpose and AutoFix trace control

[matmul.cpp](../../../../../ttnn/cpp/ttnn/operations/matmul/matmul.cpp) materializes
the transpose unless the chosen program supports native transpose. The initial
graph's default `MatmulMultiCoreProgramConfig` therefore produced ordinal 35
despite `transpose_a=True`. The earlier source spelling and paired result do
not establish a removed dispatch.

The isolated [reuse candidates](../../tests/transpose_fusion_candidates.py)
preserve current HiFi4, exact FP32 destination, FP32 state, rounding and all
other graph operations. Both native and explicit-transpose controls initially
used `per_core_M=1, per_core_N=4`, and both failed with PCC 0.23859875. The
failure is shared by the reuse geometry, not evidence that native transpose or
FP32 is inherently unsupported. Source inspection of the optimized reuse
factory/readers shows whole-matrix batch strides inside a core's loop, while
that strip geometry can assign more than one M strip per core.

The supported adaptation assigns each full 128×128 head matrix to a work block:
`per_core_M=4`, `per_core_N=4`, `in0_block_w=1`, subblock 1×4, on the device's
existing grid. [Native whole-matrix](logs/reuse_outer_whole_v1.log.gz) passes
seven tests: one paired test and six core tests. The
[explicit-transpose whole-matrix control](logs/reuse_outer_whole_control_v1.log.gz)
passes **one paired test only**, not the six core probes. Recorded traced medians are
1.461164877 and 1.463037530 ms, respectively. The native result is only about
0.71 µs below the earlier integrated 1.461872907 ms; a different-run difference
of that size alone is insufficient for selection.

The [candidate profile](reuse_outer_profile.json), backed by its
[raw capture](tracy/linear_attention/reuse_outer_whole_decode_ops.csv.gz), proves
49 operations per replay versus the initial integrated runtime's 50: the outer-product
transpose is physically gone. Candidate kernel sum 1.43966375 ms and the
initial integrated 1.439590 ms differ by just 0.074 µs. This proves a real graph
fusion but, by itself, does not establish a wall-time win.

The first two-live-trace matched test failed its final stress comparison at
PCC 0.25064918 after initially passing eager comparison:
[log](logs/reuse_outer_matched_v1.log.gz),
[provenance](logs/reuse_outer_matched_v1.provenance.json). That test allocated
decoder B's persistent buffers after capturing A. It did not check each replay
and therefore did not show that corruption began specifically at replay 64.
The fresh [AutoDebug report](AUTODEBUG_transpose_trace.md) identified the
allocator/trace lifetime hazard before any kernel change was attempted.

The [positive control](logs/trace_allocation_repro_v1.log.gz) uses two identical
`FusedDecoder` classes with the bad allocation order. Replaying only A once
changes 12 persistent B buffers, including its recurrent state, convolution
history, norm weights and FIR taps; B has not run. The control records addresses,
changed-element counts and nonfinite values, directly proving cross-trace
corruption without using the native-reuse candidate. Restoring B's state alone
could not repair its changed weights.

The harness fix constructs both decoders, uploads all persistent inputs, runs
both prefills/eager warmups and saves states **before either trace capture**.
No candidate arithmetic or correctness bar changed. With that sole ordering
repair, [matched v2](logs/reuse_outer_matched_v2.log.gz) passes the original
15-window/64-replay comparison, one-replay eager equality, state checks and
stress PCC 0.9999999999999881. Its small −0.162 µs median paired difference
motivated a longer matched measurement rather than immediate selection.

[Matched v3](logs/reuse_outer_matched_v3.log.gz) alternates the initial integrated
runtime and native candidate for **31 measured windows × 256 replays**, after
two warm windows. Each window restores identical state before timing. Native
reuse is faster in 25/31 pairs: the mean native-minus-default difference is
−0.248252 µs, sample standard error 0.040963 µs, and median paired difference
−0.250328 µs. The separate medians are 1.461293437 → 1.461055383 ms. Eager PCC is
0.9999999999999812 and final stress PCC is 0.999999999999993; eager/replay and
state assertions also pass. These are measured small savings, not a claimed
large speedup from dispatch count alone. [Provenance](logs/reuse_outer_matched_v3.provenance.json)
and [source snapshot](logs/reuse_outer_matched_v3.sources.json.gz) pin the compared
graphs and fixed harness.

The coordinator selected the whole-matrix reuse config in production using
only setup-time config creation and the outer matmul's `program_config`
argument. HiFi4, FP32 destination/state, BF16 surrounding graph and cache
policies are unchanged. Current source SHA256 is
`18d59502e7e168e584e58762396b9cc61eae6de304045542d15bbcc070f9b11d`.
All applicable material fusion families identified in this rescan now have a
measured implementation/adaptation or a precise non-matching existing-op
contract.

## Final integrated validation

The final source hash above is identical in all seven refreshed test/profile
provenance files, each returns zero, and all corresponding log hashes match.
This is new-source evidence, separate from the initial 84/9-test runs and v1
profiles retained in the historical archives.

| Gate | Final-source result | Evidence |
|---|---|---|
| Short integrated suite | 85 passed, 9 deselected, 150.23 s | [log](logs/final_short_v2.log.gz), [provenance](logs/final_short_v2.provenance.json) |
| Native/long-context suite | 9 passed, 73 deselected, 169.40 s | [log](logs/final_long_v2.log.gz), [provenance](logs/final_long_v2.provenance.json) |
| Watcher | 23 passed, 59 deselected, 44.57 s | [log](logs/watcher_final_v2.log.gz), [provenance](logs/watcher_final_v2.provenance.json), [audit](watcher_audit.json) |
| Four final profiles | All four HF measured-output checks pass; complete coverage, no dropped markers; 120/49/33/36 ops per forward | [performance.json](performance.json), current raw/report links above |

The short suite's paired wall medians are functional→final linear prefill
40.662267→26.180386 ms and decode 1.623890→1.462064 ms; full prefill
35.449361→23.498724 ms and decode 1.512215→1.263523 ms. Minimum paired output
PCC remains 0.999897954 for linear and 0.999967148 for full attention. These
independent functional-versus-final windows validate the integrated graph;
the controlled default-versus-native v3 experiment above establishes the small
outer-transpose selection benefit. Independent [stage review](STAGE_REVIEW.md) returns `clean-pass`; no runtime
validation or material-fusion decision is pending. Local checkpoint details are
recorded in [work_log.md](work_log.md).

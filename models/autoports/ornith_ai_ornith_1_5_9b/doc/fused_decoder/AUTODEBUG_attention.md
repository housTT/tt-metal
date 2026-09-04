# AutoDebug: remaining full-attention fusions

2026-09-04. Inspection-only diagnosis for the current fused-decoder stage; no
device access, measurements, or runtime changes were made by this investigator.
Read AutoFix, graph-fusing, the model contract, current fused source, candidate
source and work log, and the pinned Ornith-1.0-35B fused decoder/work log. Old
35B measurements are not evidence for this checkpoint.

## Starting evidence

The local work log records the accepted packed Q/K/V/gate projection, native
decode head layout, partial HF RoPE, batched prefill cache fills, dedicated
prefill head concatenation, and fused decode K/V cache update. Its current
2048-position paired trace result is 1.279 ms versus functional 1.510 ms.
`full_width_attention_v2` passed but was slower than native partial RoPE. These
are existing main-agent results, not new measurements in this report.

Current dimensions are B=1–32, Q heads=16, KV heads=4, head width=256, rotated
width=64. Weights, activations and paged KV cache remain BF16. The generic
compute config is HiFi4, exact math, FP32 destination accumulation enabled.
The existing prefill SDPA-specific HiFi2 config is a separate inherited policy.

## Current attention graph and movement

| Subgraph | Operations and material handoffs |
| --- | --- |
| Projection | One packed BF16 linear; QKV/gate slices. Decode QKV moves DRAM→L1 interleaved and `nlp_create_qkv_heads_decode` creates height shards. |
| Q/K norm | Q and K move height-sharded L1→DRAM, then separate learned RMSNorms. V retains its original height shard. |
| Decode RoPE | Two table embeddings. Per Q/K: slice64, transpose batch/head, interleaved `rotary_embedding_hf`, inverse transpose, slice192, width concat256. |
| Decode cache/SDPA | K moves to B L1 cores disjoint from V; `paged_fused_update_cache`; GQA paged SDPA consumes Q and BF16 caches and emits DRAM. |
| Decode output | Reshape `[1,B,16,256]` to `[B,1,4096]`; fused sigmoid/multiply gate; output linear. |
| Prefill RoPE | Setup-tiled table slices; per Q/K slice64, dedicated rotate-half, trim, slice192, concat. |
| Prefill cache/output | Two batched paged fills; chunked paged SDPA; `nlp_concat_heads`; fused sigmoid/multiply; output linear. |

## Exact source constraints

Source paths below are relative to the repository root.

1. **HF partial64 slicing is no longer blocked by the 35B explanation.**
   `ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp:268`
   derives reduced shard specs, and line 344 dispatches TILE slices through
   TensorAccessor with native support for sharded buffers. A 64-wide slice is
   still data movement, but it can write a `[32,64]` height shard directly.
   The prior source argument that partial slicing must leave sharding cannot
   close the current candidate.
2. **Native HF decode RoPE supports width64 with FP32 accumulation.**
   `.../experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_device_operation.cpp:35`
   permits widths32 or multiples64; native decode needs height-sharded Q/K
   and sharded cos/sin with matching batch. Its factory binds resident L1
   buffers, so the trig tensors must cover the input cores in the same order.
   `rotary_embedding_hf_sharded_program_factory.cpp:250` uses the shard-grid
   bounding rectangle, unlike the fused llama factory. Avoid holes: choose a
   filled rectangular grid whose core count divides B, and allow multiple
   users per core where required. A sparse B13 core set is not a safe probe.
3. **Q/K norms cannot directly consume their existing height shards.**
   `.../normalization/layernorm/device/layernorm_device_operation.cpp:163`
   expressly rejects HEIGHT_SHARDED inputs. Keeping the existing DRAM norm
   boundary makes the RoPE experiment independent of norm/config tuning.
4. **Fused llama QK64 is feasible after an exact basis permutation.**
   `.../experimental/transformer/rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_device_operation.cpp:24`
   requires BF16 height-sharded Q/K/cos/sin/transformation matrix; Q and K
   have equal B≤32, equal tile-multiple widths, disjoint grids, and at most64
   total cores. Cos/sin have logical batch2B; the transformation matrix has
   one32×32 tile per participating core. The factory runs on the union of
   Q/K core sets, including irregular batches without bounding-box holes.
   The stock interleaved rotation has different math from HF rotate-half.
   Its tile-local transformation matrix cannot reach the HF64 midpoint32,
   so a custom32×32 transformation alone cannot make unchanged HF64 inputs
   equivalent. Permute each Q/K head to
   `[0,32,1,33,...,31,63,64,65,...,255]` at weight setup and permute each
   learned Q/K norm vector identically. Use trig rows
   `[c0,c0,c1,c1,...,c31,c31]` and corresponding sin rows. This preserves
   Q·K and leaves V/gates/output-projection basis unchanged. Prefill must
   use the same basis and llama-style partial rotation, and test-side K
   cache comparison must inverse-permute the final dimension.
5. **Fused llama full256 is blocked by the preserved accumulator policy.**
   The same validator at line59 requires `fp32_dest_acc_en=False` when
   width>128. Partial64 needs no precision change. The measured HF full256
   rejection does not prove anything about fused llama partial64: the
   permutation, table width, dispatch count and resident buffers differ.
6. **Decode concat-heads cannot consume SDPA output directly here.**
   `.../transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:412`
   treats four KV heads as GQA and rejects sharded output. Therefore this
   model needs a DRAM→L1 height-shard handoff before concat-decode.
   `.../experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp:43`
   requires one `[32,256]` shard per user. Its output spec at line79 is
   WIDTH_SHARDED L1 `[1,1,32,4096]`, including **logical** batch32 for B<32;
   the exposed memory_config argument is ignored. The comparison must count
   trimming/reshaping and matching the gate/output linear. For irregular
   input grids pass the complete available `sub_core_grids` explicitly.

## Experiments still required, in priority order

| Hypothesis | Smallest useful experiment and prediction | Current verdict |
| --- | --- | --- |
| Fused llama partial Q/K64 saves two independent rotations and their decode transposes | Setup-only Q/K/norm/trig permutation; direct DRAM slices into disjoint64-wide shards; one fused rotation; width concat tails on those grids; direct K-cache shard. Validate prefill, continuation, trace, changed positions and inverse-permuted cache. | Applicable dedicated fusion; not tested. |
| Native HF partial64 sharded decode removes the batch/head transpose pair for Q and K | Preserve DRAM norms; slice directly into L1 shards, use one shared sharded trig pair for Q/K, rotate separately, append sharded tails. Use filled rectangles/multiple users per core for all B. Count the necessary K shard handoff and any Q interleave for grouped-user shards. | Applicable dedicated-layout fusion; not tested. |
| Decode concat-heads beats the current reshape after all handoffs | DRAM SDPA→height shard→concat-decode→matching width-sharded gate and output linear, trimming padded batch at the public output. Also measure the simple restore-to-DRAM gate boundary if direct sharded output projection needs a different matmul program. | Applicable dedicated fusion; not tested. |
| One HF RoPE call on concatenated Q/K beats two calls without changing basis | Concatenate the post-norm interleaved Q/K head axis, apply one partial HF rotation, split Q/K. Include concat/split costs. Height-sharded head-axis concat is rejected by the concat validator, so this requires an interleaved boundary or a different valid layout. | Applicable structural peer merge; not tested. |

The last candidate has extra movement and is lower priority than the dedicated
Q/K fused op, but source alone does not establish a latency loss. Native
sharded and llama probes should not be rejected on their first invalid-layout
error; narrow the exact failing op and retry a supported handoff.

## Remaining pattern coverage and acceptance

SDPA already absorbs scale/softmax/matmul and preserves paged causality; fused
cache update and batched fills are already present. Projection peer packing,
native head splits and prefill head concatenation are already present. No
projection bias, standalone transpose-before-attention-matmul, unused packed
projection columns, standalone softmax, distributed RMSNorm, convolution or
top-k exists in this full-attention subgraph. Output sigmoid/multiply is fused;
folding the sigmoid into a different projection must still pay for its distinct
packed output partition. Residual/MLP patterns are tracked by the main/linear
investigation, outside this bounded attention report.

Use the existing paired equivalence harness at position2048 for comparable
trace timing; additionally validate B1/4/13/32, distinct per-user positions,
non-aligned prefill, prefill→decode cache continuity, and eager/replay equality.
Keep BF16 cache and original compute policy. A passing probe is not an accepted
rewrite until its whole block meets the prior-stage PCC gate, passes relevant
cache/trace contracts, and beats the best accepted candidate with all necessary
movement included. No candidate in this report is currently a proven fix or a
new performance claim.

## Isolated probe handoff

At the coordinator's follow-up request, authored only
`tests/attention_fusion_candidates.py`, importing the frozen
`tests/fusion_baseline.py` parent. Classes: `NativeShardedHFRope`,
`FusedLlamaQKRope`, `ConcatDecode`, `ConcatDecodeSharded`, and `JointHFRope`.
The last class is a decode peer-merge probe; the llama candidate also adapts
prefill so its cache representation stays consistent. No runtime module,
shared harness, or device state was changed by this investigator.

Host-only validation completed:

- `python_env/bin/python -m black --target-version py310 --check models/autoports/ornith_ai_ornith_1_5_9b/tests/attention_fusion_candidates.py`
- `python_env/bin/python -m py_compile models/autoports/ornith_ai_ornith_1_5_9b/tests/attention_fusion_candidates.py`
- A CPU float64 algebra check of permuted learned RMSNorm + partial RoPE,
  inverse permutation, unchanged tail, and Q·K invariance passed at
  `atol=rtol=1e-12`. This proves the basis identity, not device numerics.

No C++/CMake changed, so no build is required. Hardware correctness and
performance are intentionally pending the coordinator's serialized runs.

## Follow-up: MLP epilogues and direct K destination

The coordinator requested additional source-only probes after the packed-MLP
prefill win and slight decode loss. Added `tests/mlp_fusion_candidates.py` with
four classes: `MatmulSiluEpilogue`, `MatmulSiluEpilogueControl`,
`PackedPrefillSeparateDecode`, and `PackedPrefillEpilogueDecode`.

- **Real matmul SiLU fusion:** `ttnn/cpp/ttnn/operations/matmul/matmul.cpp:355`
  dispatches `unary_chain` for the activation keyword without `core_grid`.
  Supplying only `program_config.fused_activation=UnaryWithParam(SILU)` reaches
  the multicast matmul's PACK-side SiLU implementation
  (`device/kernels/compute/bmm_fused_activation.hpp`). Do not supply the
  activation keyword as well: that would apply SiLU twice with an explicit
  config and no core_grid. The alternative `core_grid` selector carries the
  keyword into its generated multicast config and suppresses unary fallback.
- **Matched program evidence:** existing functional profiler CSVs on the
  11×10 P300c compute grid record the gate `[M,4096]@[4096,12288]` prefill2048
  program as 2D, in0 block1, per-core M7/N35, output block7×7, subblock1×1;
  decode is 1D multicast-in0, in0 block2, per-core M1/N4, output/subblock1×4.
  The candidate and control explicitly use those same programs, differing
  only in the SiLU location. Other shapes use the same full-grid selector for
  both, rather than inventing a geometry sweep. BF16 output and the existing
  HiFi4/FP32 accumulation config are preserved. Moving SiLU before the output
  BF16 pack can change rounding, so passing source algebra is not enough.
- **Mode-specific packing:** only prefill uses packed gate/up; decode retains
  separate linears, optionally with the proven-on-device epilogue if that
  candidate passes. A packed gate/up matmul epilogue applies activation to
  every output column; it cannot selectively activate the gate half. Applying
  SiLU to the up half changes SwiGLU and is an exact math mismatch.
- **Residual RMSNorm remains blocked as a useful fusion here:**
  `normalization/rmsnorm/rmsnorm.cpp` returns one Tensor from `prim::layer_norm`;
  `layernorm/device/layernorm_device_operation.cpp:482` either allocates that
  normalized output or aliases the input for sharded in-place normalization.
  It does not return/write a separate `x + mixed` residual sum. This block
  needs that sum again after the MLP; producing it still costs an add, and
  normalizing in-place destroys it. No residual-norm probe was added under the
  requirement to keep that residual available without an extra add.

Appended `NativeShardedDirectK` to the attention probe file, preserving the
earlier classes. Its K64 slice, trig buffers and passthrough192 slice are placed
on a filled B-core rectangle disjoint from V's first B cores; width concat
produces the final256-wide cache input on that same rectangle. Fused cache
update accepts arbitrary disjoint equal-size core sets, so the old canonical
K destination is not mandatory. For the actual11×10 device, B1/4/32 each have
a valid disjoint rectangle; B13 cannot factor into a rectangle fitting that
grid and retains grouped-user HF RoPE followed by cache reshaping. HF RoPE's
output spec copies its input shard spec, so changing only its output memory
config cannot redirect K to another grid. The candidate explicitly includes
two small cos/sin shard copies; eliminating the256-wide K reshard alone does
not establish a performance win.

The same epilogue recipe applies to a **separate GDN Z projection**: put SiLU
in the multicast program config, remove the downstream SiLU application, and
retain plain multiplication with the separately normalized recurrent output.
Z head splitting is an ordering transform, so elementwise SiLU commutes with
it algebraically. Compare against the current12352-wide packed QKV/Z/A/B
projection with an8256-wide packed QKV/A/B projection plus separate Z; this
trades a projection dispatch for the activation placement and requires its
own whole-block evidence. No GDN projection change is included in these files.

## Follow-up: joint rank-four GDN norm and HF prefill RoPE

At the next coordinator request, added only
`tests/gdn_joint_prefill_candidates.py`. `JointPrefillNormGDN` derives from
the passing `HybridNormGDN` and concatenates flat Q/K along width before
reshaping `[B,T,4096]` to `[B,T,32,128]`. It calls the same `l2_norm_ttnn`
exactly once, then slices the16-head Q and K halves into DRAM. The helper
still computes RMSNorm with epsilon`1e-6/128`, BF16 output, then a separate
BF16 multiply by`128**-0.5`; it does not fold that scale into a norm weight.
The joined head dimension is tile-aligned and has no16→32 head padding.
The reshape changes physical ordering and may dispatch a native tiled
reshape; alignment is not a claim that it is a metadata-only view.

The chunk adapter's source (`transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp:155`)
sets `flat_qk` from rank, and line207 derives internal `qk_norm` from that
rank even when the exposed `use_qk_l2norm` argument is false. Consequently
the new probe deliberately passes **rank-four normalized Q/K**, preventing
implicit second normalization; V remains rank-three, and the chunk op still
returns head-major output. Its launch flags, constants, state tensors and
sub-batch loop copy the passing HybridNorm launch structure without invoking
the inherited normalization method. On11×10 with32 value heads, each launch
still covers at most3 users. Clean aliases compose the same normalization
with `HybridArithmetic`, `HybridCombinedGDN`, `HybridConvOnly`, and
`HybridKdaNorm`; no existing linear candidate was edited.

The same file now also contains `SeparateZSiluGDN` and its matched
`SeparateZSiluControlGDN`: packed QKV/A/B plus separate Z, full-device-grid
matmul selection for Z, and only the fused variant supplies its SiLU
epilogue. The output method applies no second SiLU to the fused Z.
`JointPrefillSeparateZSiluGDN` composes that projection probe with the joint
prefill norm. These are isolated experiments, not promoted runtime changes.

Host-only AST/stub tests passed for joint-Q/K head order and the unchanged
BF16 norm/scale rounding sequence at B1/4/13/32. They counted exactly one
normalization and checked rank-four Q/K plus unchanged flat V at the launch
boundary. All three new/extended candidate modules pass Black with
`--target-version py310` and `py_compile`. Device PCC remains the coordinator's
responsibility.

Finally appended `HFPrefillRope(NativeShardedHFRope)` to the attention file.
Only `_apply_partial_rope` changes: slice64 from `[B,H,T,256]`, call
`rotary_embedding_hf` with shared`[1,1,T,64]` tables and the current explicit
HiFi4/FP32 accumulation config, then concat the unchanged192 tail. This
compares the HF prefill kernel with the older rotary kernel's default config.

Coordinator update: `direct_k_rope_v1` passed correctness and measured
1.266676 ms against native sharded1.266287 ms, an effective tie; native
shared-trig RoPE remains selected. The two extra trig shard copies offset
the saved K reshard. These measurements were supplied by the coordinator;
this investigator made no device calls.

## Follow-up: coherent mode-specific output normalization

Added only `tests/mode_fusion_candidates.py` for the coordinator's last output
norm combination. `PlainNormFIRGDN` combines `JointGateGDN` with shared FIR
row conversion/state writes. `ModeNormGDN` retains those methods and the
decode-only gate fusion, adding a setup BF16 KDA norm vector. QKV/A/B remain
packed into8256 columns, and Z remains a separate projection using the same
full-device `core_grid` in **both** modes, as in the measured separate-Z
control. Only decode supplies the SiLU epilogue. Prefill therefore feeds raw
Z to KDA sigmoid-gated RMSNorm and multiplies by raw Z once; decode feeds
already-activated Z to the plain norm/multiply output method. Direct calls
select only KdaNormGDN's positive-prefill branch, avoiding its unrelated
decode `super()` chain. No AllModeGate prefill fusion was inherited.

Black, `py_compile`, and host AST/stub checks passed. The checks covered
QKV/A/B field boundaries, identical Z core-grid selection at T1/128/2048,
decode-only SiLU, preserved BF16/compute policy, inherited gate/FIR methods,
and the separate prefill/decode output dispatch. No device calls were made.

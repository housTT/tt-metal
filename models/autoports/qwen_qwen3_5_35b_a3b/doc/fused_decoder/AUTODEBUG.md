# AutoDebug: fused-decoder contract audit

Inspection date: 2026-10-10. This is a source investigation, not a device run or a fusion-selection verdict.

The strongest findings are:

1. **`native_recurrence` reinstates the precision path that the functional stage repaired.** Its FP32 buffers do not prevent repeated TF32 operand conversion. A Python compute-config change cannot turn this kernel's FPU/matmul implementation into the functional FP32 SFPU recurrence.
2. **KDA convolution is not an equivalent implementation of the functional FP32 FIR.** It packs BF16 partial sums after the first three taps and applies SiLU before a separate final BF16 convolution-output boundary. Both differ from the functional graph. Its shape/history adapter is substantially sound; precision is the obstacle.
3. **FP32 `addcmul` is a real remaining fusion opportunity, not another TF32 path.** All three FP32 inputs receive full-FP32 unpacking. Convolution is especially promising because its multiplicands originate in BF16. Recurrent state `addcmul` changes the separate multiply/add rounding boundary and requires a separate decision and test.
4. **Some apparently structural fusions change arithmetic too.** In particular, outer `rms_norm(weight=)` moves the affine multiply from FP32 SFPU to an FPU gamma path; GDN L2-as-RMSNorm changes BF16 intermediate rounding. Their passing short runs do not establish preservation of the existing numerical contract.
5. **Useful local structural work remains:** direct tiled reshapes, grouped GQA broadcasting, flat-V chunk input, consolidation of convolution layout conversions, and writing the final FP32 state add directly to the persistent buffer. None requires editing shared C++.

## Scope and evidence identity

Only this report was written by this investigator. No implementation edits, device discovery, model execution, resets, or child agents were used. All proposed implementation interventions are confined to `tt/fused_decoder.py`; tests and documentation may also change. Shared C++/LLK source is evidence only, not an available repair surface.

HEAD was `53274a46856a80e3872967fd0e79d2d9758f48a7`. The working tree already contained the untracked candidate and stage artifacts, plus changes to `tests/run_layer.py`. The candidate continued changing during inspection: it initially had 418 lines; a later inspected snapshot had 532 lines and SHA256 `f9a2d2e2a0a528cce0f65fa74b4390d7bf62d4c9cac7b75faf335ae098f6db08`. References to candidate methods below identify the audited behavior even if subsequent edits move line numbers. Newly added grouped/state-ternary and attention/shared/router candidates are observations, not accepted replacements.

That 532-line snapshot was observed at 01:45 UTC. Further changes, including additional expert/FFN branches and test-runner extensions, appeared during report review. The GDN/native/KDA/normalization source findings remain applicable to the inspected methods; this report does not certify the subsequently expanded candidate as a whole. Rebind final test evidence to the exact implementation that is selected.

Stable functional file hashes inspected:

| File | SHA256 |
|---|---|
| `tt/functional_gdn.py` | `fe01dfacefd617bd1892eb9de2f5993d341eef9b6dd768cdb77aec446736c286` |
| `tt/functional_decoder.py` | `2e25b47139c58f5eb043f94e0fa0e8c25acda2b5875e064a47a62ce15623dc48` |

The target remains the pinned `Qwen/Qwen3.5-35B-A3B` checkpoint, revision `59d61f3ce65a6d9863b86d2e96597125219dc754`. A header-only read of the local safetensors shard verified layer-0 `linear_attn.conv1d.weight` is **BF16 `[8192,1,4]`**; no weight payload or device was loaded for that check.

### Direct observations versus interpretations

The `linear_trials129` artifacts record B1, 129-token prefill, eight decode steps, real weights, trace replay, and runtime audit. Selected numbers are:

| Recorded candidate | HF prefill PCC | HF minimum decode PCC | Functional minimum decode PCC |
|---|---:|---:|---:|
| functional | .999614835 | .998985827 | — |
| norm | .999585688 | .998990119 | .999943316 |
| norm/router/native_recurrence | .999585092 | .998969674 | .999918938 |
| norm/router/kda_conv | .999580979 | .998764873 | .999926686 |
| norm/router/gdn_ternary | .999585152 | .998989820 | .999943316 |
| norm/router/gdn_heads | .999585152 | .998989820 | .999943316 |
| norm/router/gdn_norm | .999590874 | .998989999 | .999950886 |

These are **observed short-run passes**, not invented failures. However:

- Prefix 129 exercises only one prefill token through the recurrent tail. Prefixes 255 and 465 exercise 127 and 81 tail updates, respectively, before decode. The short native pass does not contradict the known long-tail failure.
- These variants combine candidates with `norm,router`; most are not one-variable contrasts against a matching `norm,router` control. Equal whole-layer PCC values do not prove equal intermediates or that a rewrite saved a device program.
- These JSONs identify fused source hash `50de1cd7…`; adapted full-attention JSONs identify `4fb2ef5a…`; initial full-attention JSONs identify `15f49087…`. None identifies the later 532-line snapshot. Do not attribute a later adapter's behavior to an earlier artifact.
- `run_layer.py:272` labels all layer-0 runs `native_exact_chunk_and_sfpu_fp32_token`, including `native_recurrence`. This metadata is demonstrably misleading; consult `fusions` and source identity. The comparison called `unfused_equivalence` gates **PCC >= .995**, not elementwise or state equality.
- `traced_replay_host_us` times blocking host replay. It includes a completion boundary, but it is not the warmed device-program timing used in the functional profiler reports. It cannot establish which individual fusion won.

## The functional GDN contract being preserved

[Functional GDN](../../tt/functional_gdn.py) is more specific than “same real-number equation, FP32 output.” Relevant boundaries are at lines 112–147, 149–200, and 223–281:

| Stage | Required behavior |
|---|---|
| Projection | BF16 checkpoint mega weight; Q/K/V widths 2048/2048/4096; BF16 activations; HiFi4 and FP32 destination accumulation |
| FIR | Three **raw projected BF16** history tokens; four FP32 products in tap order and three FP32 adds; BF16 cast of complete sum; FP32 SiLU; BF16 cast |
| Q/K norm | Split K16/D128; BF16 square, sum, epsilon addition, rsqrt and multiply boundaries, epsilon `1e-6` |
| Gates | `beta = BF16(sigmoid(FP32(b)))`; `g = -exp(A_log) * softplus(FP32(a)+dt_bias)` in FP32 |
| Token recurrence | Adjacent key-head sharing: value head `hv` uses key head `hv//2`; scale normalized Q **after FP32 conversion**; FP32 state `[1,32,128,128]` |
| Output | Read updated state, cast recurrence output BF16; FP32 RMSNorm; BF16 cast then BF16 norm weight; FP32 SiLU(Z) and multiply; BF16 output-projection input |
| Ownership | Outer decoder selects stable per-request convolution/recurrent buffers; copy returned state into that selected buffer; padding never advances a request |

For each token the FP32 recurrence is:

```text
D = S * exp(g)
m = sum_K(D * k)
delta = (v - m) * beta
S_next = D + k * delta
o = sum_K(S_next * (q / sqrt(128)))
```

The functional implementation deliberately keeps multiply, subtract, add, and accurate reduction boundaries separate. Both sums use `fast_and_approximate_mode=False` and FP32 destination accumulation. `operations/reduction/generic/device/reduce_op.cpp:197–216` selects the accurate FP32 SFPU path on Blackhole. Matching FP32 binary operands select SFPU, with explicit `UnpackToDestFp32` in `operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp:1256–1265`.

Exact architecture/ownership, preservation of this dtype policy, and a measured PCC acceptance gate are distinct requirements. A candidate can pass .995 on one input while changing a protected precision boundary. No bitwise HF-equivalence claim is made for the functional implementation itself.

## 1. Native recurrent GDN: confirmed contract obstacle

The candidate `_FusedGDN._recurrent_step_fp32` calls `ttnn.transformer.fused_recurrent_gated_delta_rule` with the same HiFi4/FP32 compute config and persistent state. This is the same relevant lowered path described in [functional AutoFix](../functional_decoder/AUTOFIX.md).

The complete source chain is under `ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/`:

- Public wrapper `fused_recurrent_gated_delta_rule.cpp:84–113`: Q is cast FP32 and scaled, Q/K are head-split and repeated adjacently, V is head-split, and `exp(g)`/beta become per-head scalar tiles. For the target, `B=1,T=1,H=16,HV=32,K=V=128,G=2,BH=32`.
- Factory `device/fused_recurrent_gated_delta_rule_program_factory.cpp:49–98,128–157`: `Kt=Vt=4`, 16 state tiles/head, 32 head cores, compute arguments `{4,4,0}` and runtime `T=1`. All buffers are Float32. The compute descriptor sets FP32 destination accumulation but **no unpack-to-destination mode**.
- `tt_metal/jit_build/genfiles.cpp:974–980` selects TF32 conditional unpack format for this configuration. `tt_metal/jit_build/data_format.cpp:215–221` gives a Float32 exception only to buffers with explicit nondefault unpack mode.
- Compute `device/kernels/compute/fused_recurrent_gated_delta_rule.cpp`: state decay uses FPU broadcast multiply (138/189); memory and output use matmul (196/222); residual uses FPU subtraction (113); state update seeds destination from decayed state (77) then adds a rank-one matmul (81). The value lost on a source-register crossing cannot be recovered by FP32 accumulation or FP32 packing.

The recurrence order and head mapping agree with the functional mathematics. The source evidence supports **repeated state/operand quantization**, not an omitted beta, wrong GQA grouping, or pre-update output read. For one invocation, state/decay produce 16 decayed tiles; K and V contribute four tiles each; memory produces four tiles; delta processes `3+1` destination tiles; the update produces 16 next-state tiles and output reads them. The wait/pop flow is consistent with this target. There is no source-supported buffer-count/hang diagnosis in this investigation.

The recorded controlled comparison is stronger than the new 129-token smoke. `components_positions18_fp32sfpu.json` and `components_positions18_sfpu_recurrence.json` both identify input SHA256 `b0b96bb8c5369dc096167abe309e43e9f1adaed486850956ecd435b07d150a6b`, prefix 465, source row18, and eight decode steps. At step3, GDN relative L2 changes from **.05019733 to .00846517**, and final-layer minimum PCC improves from **.98972017 to .99806225**. The same step records HF/TT top-8 overlap improving from 7 to 8. The earlier native control's router-on-identical-input checks agree on the selected set; this supports a causal chain from changed GDN output, through residual/norm, to a routing cutoff, rather than an independent router-ordering bug.

These historical artifacts localize the numerical problem to the recurrent implementation and support the TF32 explanation; they do not isolate the contribution of each native instruction. They are not fresh hardware results from this investigation.

**Smallest permitted intervention:** retain the functional FP32 token recurrence in the local subclass and optimize its layout/dispatch around it. Do not select native recurrence as an exact replacement on the strength of the short smoke. `UnpackToDestFp32` is not a Python `DeviceComputeKernelConfig` switch exposed by this factory; moreover `tt_metal/api/tt-metalium/base_types.hpp:33–40` explicitly makes that mode incompatible with source-register unpack. Merely adding that mode everywhere would not make existing matmuls full-FP32. Shared kernel changes are outside scope.

## 2. KDA convolution: BF16 partials and a different activation boundary

The current adapter supplies RM BF16 input/history, tiled BF16 taps, constant device `actual_start=0`, and history as predecessor carry. For a single token it pads the current tensor to 32 rows, uses only output row0, and updates history from the **unpadded** real prefix. For one device with start zero, the KDA reader takes history rows 0/1/2 followed by current rows. Thus token padding need not corrupt history: `[h0,h1,h2,x0]` produces the next history `[h1,h2,x0]`. For an aligned chunk T, the last three projected rows start at offset T in `[history,current]`, as the adapter uses. Do not blame padding or tap order without a contrary observation.

The obstacle lies below that correct adapter. In `ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/`:

- `device/qkv_causal_conv1d_silu_device_operation.cpp:34–55` requires BF16 input, history, and all taps; output specs at 128–132 are BF16.
- Factory lines 71–98 derive **every** dataflow-buffer format, including `partial`, from BF16 input. FP32 destination accumulation does not change that buffer format.
- Compute lines 36–80 multiply a tap, add the previous partial, and pack into BF16 `partial` for taps 0,1,2. Tap3 applies `silu_tile` before packing the final output. The functional graph instead rounds the complete FP32 convolution to BF16 *before* converting back to FP32 for SiLU.

For target widths `(2048,2048,4096)` and channel chunk512: `Ct=256`, `block_ct=16`, 16 channel blocks. T128 has `Mt=4`, 64 work items; padded T1 has `Mt=1`, 16 work items. Each item consumes 16 activation tiles per tap and 64 weight tiles per channel block. Partial capacity is two blocks, 32 tiles. The compute loop reserves groups of four while retaining the previous 16-tile block, then pops that block: the two-block capacity covers this use. This audit found a precision mismatch, not a demonstrated queue-capacity mismatch.

A cheap host arithmetic illustration, **not a simulated hardware result**, makes the discrepancy concrete: exact BF16 products `[1, 2^-8, -1, 0]` sum to `2^-8` in FP32. BF16 partial rounding at the second tap rounds `1+2^-8` to 1 (nearest-even); the next tap cancels it to zero. SiLU cannot recover the difference. The repository KDA golden (`tests/ttnn/unit_tests/operations/experimental/kda/kda_test_utils.py:265`) uses its own tap-sum reference, not this autoport's explicit FP32 FIR contract.

**Smallest permitted intervention:** keep the local FP32 FIR or replace its add/multiply pairs with qualified FP32 ternary operations. No configuration in this KDA API exposes FP32 partial buffers or inserts the missing pre-SiLU BF16 boundary. Upcasting the output, using HiFi4, or changing channel chunk size cannot restore discarded partial bits. A separate precision-policy experiment may evaluate KDA, but it is not an exact fusion repair under the stated contract.

## 3. FP32 ternary arithmetic: where it helps and where it changes rounding

`ttnn.addcmul(a,b,c,value=1)` is not merely a Python spelling for three unfused operations in this checkout. The relevant source is `operations/eltwise/ternary/`:

- `ternary.cpp:314–365` requires matching input dtypes, classifies broadcast, and dispatches supported FP32 cases to `prim::ternary`; unsupported cases fall back to a composite.
- `device/ternary_program_factory.cpp:1307–1379` excludes FLOAT32 from the FPU choice, enables FP32 destination output, and marks input CB0/1/2 `UnpackToDestFp32`.
- `device/ternary_op_utils.cpp:230–243,694–803` supports both convolution row broadcast and recurrent mixed row/column broadcast.
- The SFPU kernels copy all three inputs to destination before arithmetic. Blackhole helper `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h:49–59` computes `value*b`, then the MAD expression `product*c+a`, retaining FP32 output.

Concrete dispatch cases, all tiled interleaved FP32 with FP32 output:

| Use | A / B / C logical shapes | Selected broadcast |
|---|---|---|
| FIR T128 | `[1,128,8192]` / same / `[1,1,8192]` | ROW_BCAST; reader replicates each weight row into its tile |
| FIR T1 | three `[1,1,8192]` | NONE; only logical row0 matters |
| State update, flat heads | `[1,32,128,128]` / `[1,32,128,1]` / `[1,32,1,128]` | ROW_COL_BCAST |
| State update, grouped heads | `[16,2,128,128]` / `[16,1,128,1]` / `[16,2,1,128]` | ROW_COL_BCAST plus outer group broadcast |

In the recurrent case, B is column-broadcast across V and C is row-broadcast across K. The mixed reader explicitly fills B's first column and C's first row (`ternary_reader_row_col_bcast_ttt.cpp:162–177,228–237`), and the SFPU consumes expanded tiles. This is supported lowering, not an assumption that ternary broadcasting resembles binary broadcasting.

**FIR is the priority.** Its actual input and checkpoint taps are BF16 values widened to FP32. For finite products within normal FP32 range, their at-most-16-bit significand product is exactly representable in FP32. With `value=1`, removing a product pack should therefore not remove a meaningful FP32 product rounding step. Preserve tap order, final BF16 cast, FP32 SiLU, and state-copy semantics. Current `gdn_ternary` already implements this pattern, but needs a direct FIR comparison isolated from outer norm/router changes.

**State `addcmul` is a different experiment.** K is widened BF16, but delta is a general FP32 result. Fused MAD need not match `add(D,multiply(k,delta))`. A host IEEE-rounding example with `D=-129/256`, `k=129/256`, `delta=1-2^-23` yields `-5.96046448e-8` with separate product/add rounding and `-6.00703061e-8` with one final rounding. This is not TF32 degradation, nor proof of an accuracy failure; it refutes a universal bitwise-equivalence assertion. The later `gdn_state_ternary` candidate needs exactly this independent adjudication. If explicit arithmetic boundaries are nonnegotiable, keep the separate FP32 multiply and add.

Do not specify a BF16 preallocated output to an FP32 ternary in order to “fold” the final cast: this factory derives destination width from the output format while full-FP32 unpack follows input formats. The current all-FP32 candidate does not trigger that mixed-output concern. Test coverage of mixed-broadcast addc operations in `test_ternary_bcast.py:108–142` uses BF16, so it is not proof of this FP32 GDN matrix.

## 4. Normalization and gate fusions require numerical adjudication

### Outer norm affine folding changes the selected primitive

`_FunctionalRMSNorm` (`functional_decoder.py:496–523`) constructs FP32 `1+weight`, calls FP32 RMSNorm without weight, then performs a matching-FP32 SFPU multiply. `_FusedRMSNorm` passes the same weight into RMSNorm instead. It preserves the zero-centered weight mathematically, but the RMSNorm gamma branch uses `ckl::BinaryFpu<Mul>` on the normalized intermediate and gamma (`normalization/layernorm/device/kernels/compute/layernorm.cpp:320–350`; large-tensor sibling at 368–375). The newer full-FP32 finalizer condition explicitly excludes RMSNorm (`layernorm_op_multi_core.cpp:570–573`).

Thus this fold reintroduces reduced source precision for the extra affine multiply. The norm smoke's non-unit functional PCC is consistent with a changed calculation; its exact error attribution is unmeasured. This is a concrete precision-policy discrepancy, **not** evidence that all norm candidates fail .995. To preserve the existing affine contract, retain the separate FP32 multiply locally. If approximate-equivalent fusion is permitted, qualify this candidate independently on both layer kinds and the known route-sensitive rows before default selection.

### GDN L2-as-RMSNorm is algebraically correct, not boundary-identical

`RMSNorm(x,eps/128)/sqrt(128)` equals `x/sqrt(sum(x*x)+eps)` in real arithmetic. `_FusedGDN._normalize_key` uses the correct scaled epsilon. However the functional path rounds BF16 square/sum/epsilon/rsqrt/product intermediates; the fused RMSNorm uses its own internal accumulation and an additional BF16 output/scale sequence. Retain it as a numerical candidate, not a proven exact substitute. Feeding **flat Q/K** to the chunk op also silently opts into its in-kernel Q/K normalization (`chunk_gated_delta_rule.cpp:203–209`); that is not merely removal of a head reshape.

### Gate operations with concrete local adapters

- `ttnn.add(a_fp32,dt_bias_fp32,activations=[SOFTPLUS(1,20)])` folds the post-add softplus; preserve the subsequent FP32 `a_neg` multiply. It keeps the softplus parameters and matching-FP32 binary route. Compare decay-log and exponentiated decay directly, not just final output.
- `ttnn.multiply(z_fp32,normalized_fp32,input_tensor_a_activations=[SILU])` folds the GDN output gate. Keep the earlier BF16 RMSNorm-output/weight boundaries and the final BF16 cast. Both operands are full-shape FP32, so this is the no-subtile-broadcast SFPU binary route.
- Full-attention/shared `multiply(gate,output,input_tensor_a_activations=[SIGMOID])` is another candidate; its BF16 activation rounding differs from the FP32 case and must be checked separately. Attach the activation to the actual gate operand. Check emitted device programs: an input-activation API may still stage an intermediate internally.
- Packing `dt_bias` into the mega linear bias is **not** automatically equivalent: the functional `a` slice is already BF16-rounded before FP32 bias addition. Likewise folding residual add into RMSNorm must retain the BF16 residual tensor needed by both norm and later residual consumers.
- `experimental.kda.sigmoid_gated_rms_norm` is not the GDN output gate: its compute applies sigmoid (`.../sigmoid_gated_rms_norm/device/kernels/compute/...cpp:81–94`), while this model needs SiLU plus the explicit norm-weight rounding sequence. Do not substitute it based on the name.

## 5. Remaining structural opportunities within `fused_decoder.py`

| Rewrite | Concrete adapter and invariant | Limitation / discriminating check |
|---|---|---|
| GDN heads | Direct `ttnn.reshape(x,(1,T,H,128))` for Q/K/V/Z instead of explicit RM→reshape→TILE | This is a **data-moving tiled reshape**, not generally a metadata view. Compare logical elements and downstream reductions for H16 and H32, T1 and T128. |
| Recurrent scalars | Direct tiled FP32 reshape `[1,1,32] → [1,32,1,1]`, or grouped `[16,2,1,1]` | Removes explicit layout conversions while preserving full FP32 bytes; requires warming the reshape map and preserving each head's scalar. |
| GQA without Q/K repetition | View state as `[16,2,128,128]`; Q/K columns `[16,1,128,1]`; V `[16,2,1,128]`; gates `[16,2,1,1]`. Broadcast Q/K across group2, reduce K, restore state/output shapes. | Value-head order is `hv=2*hk+group`. State reshape is a view; scalar/V/output rearrangements may move padded tiles. The later `gdn_grouped` candidate implements this idea. Keep it separate from state ternary. |
| Flat V into aligned chunk GDN | Keep normalized Q/K rank4; pass V directly as BF16 `[1,T,4096]` to `chunk_gated_delta_rule`, with unchanged constants/state/config | `flat_v` and `flat_qk` are independent flags. For C32,T128, HV32,Vt4, `reader_chunk_gdn_prep.cpp:93–113` reads page `c*128 + hv*4 + ct`, the same four V tiles/head/chunk. Skip model V head-split and native head permutation without opting into a new Q/K norm. Test all chunks/heads. |
| Consolidate FIR layout work | Create one RM `[history,current]` window, take the four tap views there, tilize each directly to FP32 if using `to_layout(...,dtype=float32)`, accumulate with FP32 ops; keep one BF16 history tail | Nonaligned tiled slice can itself untilize (`slice.cpp:342–347`). The direct dtype option works on an actual RM→TILE conversion; `to_layout` ignores dtype if layout already matches (`core/to_layout/to_layout_op.cpp:62–80`). Check both aligned slices and T1 padding. |
| Remove recurrent state copy | Compute the final **separate FP32 add** with `output_tensor` bound to the selected persistent FP32 state, then read that buffer for output; avoid the subsequent self-copy in the local forward | Old state must already have been consumed into D before overwrite; retain fixed addresses and CQ ordering. Test inactive slots, warm/eager/replay restoration and repeated reuse. This does not require changing multiply/add rounding. |
| Keep chunk output tiled with an explicit adapter | Use `chunk_gated_delta_rule(...,output_head_major=True)` for aligned T: the wrapper returns tiled `[32,T,128]` via a metadata reshape. Restore token-major axes with a device permutation, or retain head-major layout through norm/gate and restore it before the output projection. | The **default** chunk wrapper returns ROW_MAJOR `[1,T,32,128]`; its current `to_layout(TILE)` is necessary. Test the head-major adapter's output rounding and head order. It can replace the native untilize/RM-permute/retile path, not simply delete the final conversion. |

The reshape path deserves a precise host/trace qualification. `data_movement/reshape_view/reshape.cpp:709–722` only takes a metadata view when width and tiled-dimension conditions allow it. The Q/K width changes 2048→128 and the V/Z width 4096→128 do not satisfy that shortcut. Its tiled factory (`device/reshape_tiled_program_factory.cpp:300–322`) **builds and uploads a host-computed page map on a cache miss**, then retains that map as an op-owned tensor on cache hits. Activation values still move entirely on device. Warm each fixed shape before capture; do not describe the first call as host-free.

The Python runtime audit patches `from_torch`, `to_torch`, `as_tensor` and Torch dispatch; it cannot see this C++ metadata upload. Program-cache-miss prohibition and allocation tracking, including program-cache allocations, are therefore essential. The current chunk caller correctly supplies all four resident constants and initial state; dropping one constant causes the wrapper's eager host-upload fallback (`chunk_gated_delta_rule.cpp:241–266`). Keep setup uploads, diagnostic reads, cold program metadata, and warmed replay separate in the evidence.

### Full-attention and MoE candidates remain available

These operations are mathematically suitable local adapters; the short full-attention runs do not finalize them:

- **Packed Q/K/V/gate linear:** reconstruct actual Q-projection rows as `[head,2*256,2048]`, take each head's first256 Q rows and last256 gate rows, and pack `[Q4096,K512,V512,gate4096]`. The candidate follows this order. Preserve existing BFP8 weights and projection config; inspect packed quantization and per-component results. `split_query_key_value_and_split_heads(...,num_heads=16,num_kv_heads=2,transpose_key=False)` then replaces the separate head reshapes/transposes.
- **Decode head adapters:** `experimental.nlp_create_qkv_heads_decode` accepts tiled BF16/FP32 `[1,1,B,5120]`, with height-sharded output; an interleaved input requires overlapping Q/K core grids. `experimental.nlp_concat_heads_decode` consumes height-sharded `[1,B,padded_heads,256]`. For the outer decoder's B1 row, request sharded decode-SDPA output where supported, concatenate 16 heads, select the one logical batch row, and return `[1,1,4096]`. Retain the 32-token SDPA chunk config for odd logical page capacity. Reshard costs can outweigh eliminated transposes.
- **Partial RoPE:** `experimental.rotary_embedding` on only the first64 coordinates, concatenate the untouched192. In decode the cache is already the position-gathered row: use `token_index=0` on `[1,1,H,64]`, rather than indexing that one-row cache by the absolute position. Test at nonzero/large positions and inspect the untouched tail exactly.
- **Paired cache update:** `experimental.paged_fused_update_cache` can replace two decode writes if K and V shards occupy equal-sized, disjoint core grids. Validation explicitly rejects overlap (`experimental/paged_cache/device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:226–234`). The candidate's K core(0,0)/V core(1,0), shard `[32,256]`, addresses this prerequisite. Keep original page tables for writes, including small physical pools.
- **Expert-major decode:** keep sparse gate/up output `[1,256,1,1024]` through `swiglu` and sparse down, multiply matching routing weights, then reduce the expert axis. The candidate's final single-row slice is necessary because `fast_reduce_nc` can expose 32 physical rows. The public B32 path remains row-serialized, so the local B1 invariant is legitimate. Compare reduction numerics and all expert/weight pairings; an unsorted top-k alone is harmless when pairs remain associated.
- **Normalized top-k logits:** `topk(FP32(logits),8)` followed by FP32 softmax over the selected logits and final BF16 cast cancels the full-softmax denominator mathematically. Check FP32 tie/underflow cases and expert identities, especially 8th/9th ties; equal ranking in exact arithmetic does not promise identical tie choices after floating-point softmax.
- **Shared expert:** pack `[up|gate]` weights, use `swiglu`, then existing down projection; sigmoid-input fusion is separate. Existing shared MLP uses low-precision weights and a SiLU linear epilogue. Packing may move the SiLU rounding boundary, so preserve actual weight values and compare shared output independently. `linear(...,activation='sigmoid')` must be checked for actual epilogue dispatch instead of counted as fused from its API spelling.

The later `moe_gate` experiment uses `TTMoEGate`/`generalized_moe_gate`. That op's validation requires BF16 input/bias/output (`.../generalized_moe_gate/device/generalized_moe_gate_device_operation.cpp:61–65`), whereas functional routing retains FP32 probability arithmetic. HiFi4 gate matmul does not by itself prove this gate satisfies that contract. Treat it as a precision-sensitive alternative; retain the FP32 top-k/softmax adapter as the cleaner exact-policy candidate. Its local dense scatter output and trace-time buffer creation also need auditing.

## Discriminating real-weight tests for the owning agent

No tests below were run on a device by this investigator. Ensure each test actually selects the fused decoder; extend any remaining functional-only entrypoints and verify newly added selectors. A functional-only result is not fused evidence.

1. **Establish a matched control.** Use pinned layer0/layer3 weights and identical BF16 hidden streams. Run `fusions=()` against functional, then each candidate alone; separately compare candidate+`norm,router` to `norm,router`. Record effective fusion flags, actual recurrence implementation, start/end source hashes including lowered dependencies, input hash, dtype/layout/memory configs, and command exit status. Use a fresh artifact directory. `run_fusion_trials.py` currently allows existing directories and reads an existing JSON after a failed process; stale JSON must not be interpreted as the failed attempt's result.
2. **Capture the earliest divergent GDN boundary.** In a diagnostic, non-timed run, save real projected QKV, three-token history, four FP32 products/running sums, rounded convolution, normalized Q/K, decay, beta, D, memory, delta, state and recurrent output. Reuse identical pre-step snapshots for each candidate. Compare finiteness, exact equality where expected, max absolute/ULP error, relative L2 and PCC, independently for state and output. A post-norm whole-layer PCC can hide accumulated state drift.
3. **Restore the known failure streams exactly.** Use `run_components.positions_row_input(10,2048)` and `(18,2048)`; the helper consumes all preceding random draws from B32/seed1234. Merely setting length255 or465 with seed1234 is not the same input. Compare native versus SFPU with all other toggles fixed, at prefill completion and every one of eight decode steps, especially row18/step3. Include the 4096/128 run, which tests a longer decode chain after fully aligned prefill. Measure routes on both identical and each implementation's own FFN input.
4. **FIR adapter matrix.** Real checkpoint taps, both zero and nonzero history, T1 repeated and T128; compare each complete output and next raw history. Add T31/32/33,127/128/129,255/256/257 public prefixes to cover planner transitions. KDA versus functional should be evaluated as a changed partial-precision path; FP32 `addcmul` should first be tested without any KDA/norm/gate rewrite. The synthetic cancellation example is a small op discriminator, not a replacement for these real-weight checks.
5. **Grouped/reshape/flat-V controls.** Tag heads/channels with distinguishable finite patterns for mapping checks, then use real projected data. Verify `hv=2*hk+group`, no padded lane becomes a logical head, and flat-V reads all four internal chunks of T128. Compare updated state as well as output. Test grouped arithmetic without state ternary, then add state ternary as its own contrast. Preserve the precise BF16 normalization and FP32 sum settings.
6. **Ownership and trace matrix.** Run the existing heterogeneous B32 prefixes with nonzero/reordered request slots, inactive-slot sentinels, reset and continuation. Warm T1/T128 shapes, capture once with program misses forbidden and allocation tracking enabled, refresh stable inputs/state, and replay with changed requests. Verify eager/replay and repeated-identical output **and state** after blocking completion. Reuse physical buckets with withheld logical lengths/start positions (the previous functional failure included logical36 after warming35). Do not replace device offsets with host-created selectors to remove a reshape.
7. **Full-attention matrix.** Test RoPE first64/tail192 at nonzero and near-context-limit positions; Q/K/V head identities; permuted pages; page sizes32/128; small and odd page pools; continuation crossing page boundaries; B1 and row-serialized B32. Include the recorded small-pool shape `layer3,length65,steps31,batch2,minimum_cache_pages0,cache_guard_pages0` and unseen-length reuse. Compare the cache bytes written and unwritten sentinel pages, not just attention output.
8. **Acceptance and performance after component checks.** Keep HF PCC >=.995 for prefill and every decode token/request, functional comparisons, all256 experts/top8, native context and ownership gates. For accepted combinations, obtain warmed whole-layer device profiler evidence at 4096/128 and count changed programs/data movement. Rank combined end-to-end cost including reshards and map warmup; do not extrapolate from the eight host replay timings. A final combination needs its own acceptance results even when its members passed separately.

`run_layer.py` does perform useful completion checks: prefill/eager output is read back, capture follows an explicit synchronization, and replay uses `blocking=True` before reading results (149–151,196–220,229–240). The current artifacts are therefore more than “Python reached the end.” The gaps are breadth, precision/state observability, isolated controls and revision identity, rather than absence of all synchronization.

## Review of initial suspicions / other potential issues

The draft findings were revisited against caller, wrapper, factory and kernel code:

| Suspicion | Adjudication |
|---|---|
| Native recurrence changes GQA or recurrence order | Not supported: adjacent repetition, decay-before-read, delta, update and post-update output agree. Precision remains the leading obstacle. |
| KDA single-token padding necessarily advances 32 state tokens | Refuted for the current adapter: only one output row is used and history is rebuilt from the real prefix. Precision mismatch remains. |
| FP32 ternary necessarily suffers the native recurrence's TF32 loss | Refuted for matching FP32 inputs/output: explicit full-FP32 unpack and SFPU route. MAD rounding is a separate issue. |
| Direct tiled reshape is free and performs no host work | Refuted: changing inner dimensions can invoke a mapped device copy and a cold host map upload. Warm replay can reuse the map. |
| Default chunk output is already tiled, so its caller tilize can be deleted | Refuted during report review: the default wrapper explicitly returns ROW_MAJOR. Only `output_head_major=True` provides the tiled alternative, with an accompanying axis adapter. |
| GDN RMSNorm formula uses the wrong epsilon | Refuted: epsilon is scaled correctly; intermediate rounding still differs. |
| Unsorted expert IDs alone imply wrong MoE math | Refuted when IDs and scores stay paired. Test cutoff selection and contribution values instead. |
| KDA sigmoid-gated norm can replace GDN SiLU-gated norm directly | Refuted by the actual gate primitive and dtype boundaries. |
| A passing 129-token native run disproves the earlier failure | Refuted by tail counts and mismatched input/revision scope. |
| Generic native chunk or `gated_delta_attn_seq` is automatically an exact single-token substitute | Unestablished. Chunk changes normalization/rounding and padding semantics; `gated_delta_attn_seq` requires prepared chunk operands and C=K=V=128 (`device/gated_delta_attn_device_operation.cpp:43–50`). Neither supplies a demonstrated drop-in FP32 token adapter here. |

Static checks performed: Python AST parsing of functional GDN/decoder, current candidate and trial harness; JSON/source-hash inspection; safetensors-header dtype inspection; the two explicitly labeled host arithmetic illustrations; and direct rechecking of each headline source chain. No fresh hardware accuracy, trace-allocation or latency claim is made. The unexplained quantity is the exact per-instruction contribution to historical native error, not the existence of its precision-contract mismatch.

## Owner closure after independent review

This report is an intermediate source investigation, not the final rejection
ledger. Adapted native shared-down/gate/routed-add was measured at4096/128 and
rejected for latency (and full-attention PCC). Indexed partial-width RoPE was
adapted with16-channel-group Q/K/gamma/cos packing, measured faster after removing
a caller layout round trip, and selected. `graph_inventory.md`, final README,
review report and indexed_final suites supersede any earlier pending/dismissed
candidate statements in this investigation.

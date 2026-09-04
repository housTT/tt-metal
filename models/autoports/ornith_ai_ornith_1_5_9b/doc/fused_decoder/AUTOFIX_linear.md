# AutoFix linear candidate experiments

Starting diagnosis: [AUTODEBUG_linear.md](AUTODEBUG_linear.md). Main agent owns all hardware, verification and final integration. This agent authored only `tests/linear_fusion_candidates.py` and this report; no runtime implementation or shared test harness edits. Candidate scaffolding is not a verified fix.

## Candidate mapping

| Candidate | Control / isolated hypothesis | Status |
|---|---|---|
| `FlatGDN` | Current `FusedDecoder`; H1 raw flat prefill plus tightly coupled H2 head-major output merge. Retains four independent projections. Decode remains inherited. | Main reports paired real-weight layer prefill/decode/32-step PCC pass and state PCC pass; prefill 40.624→36.534 ms. Raw core amplitude probe still needed. |
| `PackedGDN` | `FlatGDN`; H4 one 12352-wide QKV/Z/A/B projection. Decode retains functional layout and recurrence. | Main reports same PCC as FlatGDN; decode 1.620→1.581 ms, prefill regresses 36.534→36.805 ms. Mixed result, not an unconditional promotion. |
| `SplitPackedGDN` | `PackedGDN` methods with two projections: QKV+Z width12288, A+B width64. Constructor preserves the exact supplied projection dtype. | Ready for H4 verify/refute alternative; untested here. |
| `DecodeLayoutGDN` | `PackedGDN`; H5 one QKV reshape/permute, adjacent QK repeat, head-major recurrence and output. Original L2/cast/scale ordering and matmul configs retained. | Main testing. |
| `ExpGDN` | `DecodeLayoutGDN`; H6a EXP folded into FP32 state multiply only. | Main testing. |
| `TransposeGDN` | `DecodeLayoutGDN`; H6b outer matmul `transpose_a=True` only. | Main testing. |
| `QueryScaleGDN` | `DecodeLayoutGDN`; H6c query L2 multiply incorporates query scale only. This changes the location of a BF16 rounding point. | Main testing; raw core amplitude and recurrence checks required. |
| `ArithmeticGDN` | `DecodeLayoutGDN`; all three H6 flags. Run single-merge controls before interpreting combined result. | Ready; untested here. |
| `KDAConvGDN` | `PackedGDN`; H3 dedicated four-tap prefill conv+SiLU+QKV split. One B=1 launch per user; existing decode retained. | Main testing. |

The measurements in this table are coordinator-reported results, not runs made by the candidate author. Exact commands/logs and promotion decisions belong in the main stage work log. Every candidate keeps 4096 hidden width, config-derived head widths, FP32 state/gates, default HiFi4 matmuls and original context orchestration. No 35B MoE machinery or 35B performance claims were copied.

`KDAConvGDN` creates its required 256-channel program object at weight construction. QKV/history inputs are ROW_MAJOR BF16; output is TILE BF16. It selects the tail from the final three **real** QKV inputs; for logical lengths1/2 it combines the remaining original history with real new rows. Batch users are sliced separately, preserving recurrence independence. History writes still use existing in-place persistent buffers. The KDA kernel is prefill-only; T=1 decode is not padded through it.

## Focused raw-core and recurrence probe

`probe_linear_core_equivalence(functional, candidate, prompt, tokens, pcc_bar=0.995, amplitude_rtol=0.05)` accepts already-built real-weight linear decoders and host activations. Supply at least32 distinct decode tokens. It resets both decoders, temporarily wraps pre-output-norm boundary methods, reads each raw core as canonical `[B,T,H,V]`, then compares every prefill chunk and every decode step. It also compares recurrent and conv state after each step and whole-layer output.

Returned metrics include PCC, relative L2 error, norm ratio and maximum absolute error for every boundary. Finite checks and PCC≥0.995 are mandatory; the explicit 5% norm-ratio diagnostic catches a dropped query scale that output RMSNorm or PCC alone could hide. This is an untimed eager diagnostic with intentional device-to-host reads, **never** a traced or timed path. It restores temporary method interception in `finally` and leaves device state advanced through all supplied inputs.

Example inside the main agent's hardware-owned pytest probe, using the existing harness:

```python
from . import test_functional_decoder as H
from .linear_fusion_candidates import FlatGDN, probe_linear_core_equivalence

reference, _, _ = H.build_decoder(mesh_device, 0, "real")
with monkeypatch.context() as patch:
    patch.setattr(H, "FunctionalDecoder", FlatGDN)
    candidate, _, _ = H.build_decoder(mesh_device, 0, "real")
report = probe_linear_core_equivalence(
    reference,
    candidate,
    H.make_activations(1, 2048, seed=71),
    [H.make_activations(1, 1, seed=100 + step) for step in range(32)],
)
```

For H6 use `DecodeLayoutGDN` as the immediate control and test each of `ExpGDN`, `TransposeGDN`, `QueryScaleGDN`; retain original FunctionalDecoder control separately. Existing paired harness adds real-weight eager/replay equality, final32 output/state, and warmed traced timing. For KDA conv add logical lengths1/2/127/129/2047 and B1/B4/B32. Run native-context and complete prior-stage gates only after selecting a measured final combination.

## Static validation by candidate author

Executed:

```bash
python_env/bin/black --target-version py310 models/autoports/ornith_ai_ornith_1_5_9b/tests/linear_fusion_candidates.py
python_env/bin/python -m py_compile models/autoports/ornith_ai_ornith_1_5_9b/tests/linear_fusion_candidates.py
```

Both pass. The initial default Black invocation warned about the repo's Python3.14 target while running Python3.10; rerunning with explicit py310 resolved it. No build required for Python/test documentation. No devices accessed by this author. Source inspection removed unsupported `ttnn.Shape` slicing before the main hardware runs; all shape indexing is explicit. Whole-tensor slice ownership and reshape aliases are handled deliberately; poisoned-pool and traced state checks remain required device evidence.

## Additional arithmetic and normalization probes

The coordinator froze the original runtime class as `tests/fusion_baseline.py: FusionBaseline` and changed the candidate module's base import to that frozen class. This prevents later production integration from silently altering experiment parents or removing projection keys they need. The candidate author preserved that import and all existing method bodies while the main agent was running them; the following candidates were appended only.

| Candidate | Immediate control / exact change | Numerical or operator constraint |
|---|---|---|
| `SoftplusGDN` | `ArithmeticGDN`; replace FP32 add then softplus with `add(..., activations=[UnaryWithParam(SOFTPLUS,1.0,20.0)])`. | These are the standalone operator's beta/threshold defaults and the pinned 35B spelling. Keep FP32 before bias; retain padding mask afterward. |
| `RankOneGDN` | `ArithmeticGDN`; replace `matmul(k,delta,transpose_a=True)` plus state add with `transpose(k)` and `addcmul(state,k_col,delta,output_tensor=state)`. | Logical operands `[B,H,128,1]` and `[B,H,1,128]`, all FP32. Current ternary source explicitly recognizes `ROW_COL_BCAST`, which takes the dedicated branch. Unsupported ternary fallback ignores `output_tensor`, so this exact contract matters. Compare state after every update; verify trace addresses unchanged. |
| `KdaNormGDN` | `ArithmeticGDN`; prefill-only KDA sigmoid-gated RMSNorm with FP32 output, then multiply by flat BF16 z, then output projection. | Computes SiLU algebraically as z×sigmoid(z); rank1 BF16 norm weight is uploaded at setup. The dedicated kernel requires T%32 and FP32 destination accumulation; decode retains ArithmeticGDN. Rounding differs from standalone BF16 SiLU and requires measured equivalence. |
| `MixedSiluGDN` | `ArithmeticGDN`; flat `multiply(FP32_core,BF16_z,input_tensor_b_activations=[SILU],dtype=FP32)`. | Tests exact mixed dtype boundary previously reported non-finite in 35B. Reject non-finite outputs. |
| `MixedSiluAGDN` | Same control; place BF16 z in operand A with input-A SiLU activation. | Explicit FP32 output prevents operand-A dtype inference from silently downcasting the result. This is a separate operand-order probe. |
| `NormWeightGDN` | `ArithmeticGDN`; replace Q/K scalar normalization multiplies with RMSNorm weights created at setup. | Q weight1/128 is exactly BF16; K weight1/sqrt(128) is rounded to BF16. Norm activations remain BF16 and recurrence FP32; use raw core/state probe to assess that coefficient/rounding change. |

`RankOneGDN` and `NormWeightGDN` share an appended arithmetic helper to avoid altering the tested `ArithmeticGDN` method. With each respective flag off, the helper follows ArithmeticGDN's Q normalization/scale, K normalization, FP32 casts, fused state decay, delta computation, transpose-a matmul update and Q read. Neither changes matmul program configuration.

Coordinator update before these appended experiments: ArithmeticGDN passed paired tests and raw-core probes at lengths127/2047/2049 with32 distinct decode tokens. Reported core minimum PCC>0.9999758, norm ratio0.99752–1.00381, recurrent minimum PCC>0.999988, trace median1.50335 ms. These strengthen H1/H6 evidence but do not prove any appended candidate.

The coordinator also corrected KDA conv program-config construction to the exported `ttnn.QkvCausalConv1dSiluProgramConfig`; the original nested namespace produced `AttributeError` before a kernel launch. That API correction is distinct from a numerical or performance rejection of KDA conv.

Appended classes were formatted with Black's Python API using line length120 and target py310, then the complete module was compiled with Python `compile(..., 'exec')`. A prefix comparison confirmed the existing classes/helpers were unchanged by formatting. No hardware or new runtime edits were performed by this author.

## Softplus accuracy localization and mixed-dtype SiLU follow-up

Coordinator-reported results: SoftplusGDN passes the broad PCC/core bars but worsens minimum layer PCC to0.999285 from ArithmeticGDN0.999876 for a trace reduction1.50335→1.49887 ms. RankOneGDN is correct but slower1.54249 ms, so reject its performance hypothesis. KdaNormGDN is correct and prefill36.874→36.622 ms in that pair. MixedSiluGDN operand B produces catastrophic PCC -0.00139; operand A produces NaN. These errors must be localized, not described as normal approximation drift.

### Softplus: identical scalar parameters, different compiled approximation

Source inspection identifies a concrete discrepancy:

1. `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp:683–692` defaults standalone softplus to beta1 and threshold20. `unary.cpp:247–265` builds a two-parameter `UnaryWithParam(SOFTPLUS,{beta,threshold})`, exactly matching SoftplusGDN.
2. `unary/common/unary_op_utils.cpp:542–552` requires **exactly two** parameters and emits `softplus_tile(i,beta,beta_reciprocal,threshold)`. There is no per-call FP32 approximation flag in this interface; adding a third value is not a supported fix. A string-conversion helper elsewhere constructs three parameters, but neither tested path invokes it.
3. The standalone factory `unary/device/unary_program_factory.cpp:26–28,420–422` sets `INP_FLOAT32` for FP32 input. The binary factory's `binary_ng/device/binary_ng_utils.cpp:581–598` emits activation calls and activation include macros but does **not** set `INP_FLOAT32`; neither does `binary_ng_program_factory.cpp`.
4. Blackhole `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softplus.h:105–145` branches on `INP_FLOAT32`: the FP32 path uses a degree8 polynomial and an exponential tail beyond |x|5; the other path uses the BF16-oriented degree6 polynomial and sets that residual to zero beyond |x|5. Thus an FP32 binary softplus activation silently selects the BF16-oriented approximation even though state/gate tensors and destination remain FP32.

This predicts a distinct signature: for biased A less than -5, the fused result becomes exactly0 while standalone softplus remains positive; around [-5,5] their polynomial errors also differ. This is source-proven algorithm selection; its contribution on the actual Ornith gate distribution still needs the focused operator experiment. Do not attribute the observed change to different beta/threshold defaults or a deliberate gate dtype change.

Appended `probe_softplus_bias_fusion(decoder,a_raw)` compares standalone and fused results from identical actual BF16 projected A→FP32 bias inputs, alongside a Torch FP32 softplus oracle. It reports biased input range, oracle PCC/relative-L2/max error, negative-tail zero counts, and the eight largest elementwise differences. Run it on real projected A for prefill and decode, plus a boundary tensor adjusted by the actual dt_bias so the biased inputs cover [-12,12] near ±5. Then substitute standalone softplus gates into the candidate recurrence to confirm that the pre-norm core drift disappears. Existing ArithmeticGDN is the final-path control; a model-level flags change cannot repair the missing compile definition. Any library repair would require a separately scoped C++ change/build and is not authored here.

### Mixed SiLU: same-dtype adaptation and remaining uncertainty

Appended `MixedSiluFP32GDN` casts the existing BF16 z projection to FP32 immediately before input-B SiLU in multiply; `MixedSiluFP32AGDN` makes the same change with z in operand A. Both multiply and output projection retain FP32 activation/output. The extra cast removes the mixed-format boundary but also moves SiLU's intermediate rounding from BF16 to FP32; it is a diagnostic graph adaptation requiring real-weight PCC and latency, not a new precision policy recommendation. It may have the same operation count and worse latency than standalone BF16 SiLU.

Relevant source boundary: `binary_ng_program_factory.cpp:1062–1104` creates operand-specific activation intermediate buffers preserving each input format. `kernels/compute/eltwise_utils_sfpu.hpp:17–51` preprocesses an activation through copy/SFPU/pack and switches pack formats, but does not explicitly switch the unpack source format; the downstream two-operand SFPU body **does** explicitly switch source formats (`eltwise_binary_sfpu_no_bcast.cpp:59–75`). The factory also configures FP32 destination accumulation when **any** operand/output is FP32 (`binary_ng_program_factory.cpp:1205–1208`). This combination makes the per-operand format transition a strong candidate for the observed corruption. It is not yet a proven kernel root cause; compiled-kernel selection and a focused operator probe are needed before assigning fault to a specific helper.

Compare separate `silu(z16)`×core32 against all four activated-multiply variants using the same real core/Z tensors, assert finite results, and record actual dtypes/layout/memory/output dtype and lowered kernel. Include small-magnitude Z as well as actual projected magnitudes to separate format handling from overflow. A passing FP32/FP32 control would verify sensitivity to the mixed input boundary; it would not automatically prove which missing reconfiguration causes it.

If both same-FP32 controls still corrupt outputs, stop reasoning from the old 35B mixed-format explanation. Request a fresh isolated AutoDebug pass with the exact lowered BinaryNG kernel, both operand-order inputs, intermediate/pre-norm core/Z snapshots or reproducible real-weight generator, poisoned-pool result and failing finite/PCC signature. Preserve the current failures and do not promote any SiLU fold until a corrected path passes the original real-weight and traced checks.

Only the new FP32 controls and softplus probe were formatted and syntax-compiled in this follow-up; prefix comparison confirmed all previously tested classes were unchanged. No hardware access or final runtime edits by this author.

## Combined paths and remaining decode folds

These follow-up candidates are appended test scaffolding only. The frozen baseline import and every previously tested method were preserved. Coordinator-run evidence remains the promotion authority; a single B1 paired pass does not establish batch1–32 equivalence.

| Candidate | Exact control |
|---|---|
| `CombinedGDN` | Cooperative MRO combines KDA prefill conv, KDA prefill output norm, and Arithmetic decode. |
| `SeparatePrefillCombinedGDN` | Same combined path, but setup retains separate original QKV/Z/A/B weights for prefill and selects packed projection only for physical T1 decode. |
| `CombinedRMTailGDN` / `SeparatePrefillCombinedRMTailGDN` | Return the three real history rows in ROW_MAJOR and tilize each individual row immediately before its in-place write. Removes the three-row tilize→untilize round trip. |
| `KDAConvDecodeGDN` | Adapt the minimum legal dedicated conv T32 to decode T1. A setup-allocated31-row zero tensor pads the input; only output row0 reaches QKV and only logical token1 reaches the three persistent history buffers. One B1 conv call per user remains necessary. Extra conversions/slices make this an applicability experiment, not an assumed win. |
| `GateAddGDN` | Arithmetic decode only: add BF16 projected A directly to the FP32 dt_bias with explicit FP32 output. This folds the lossless A typecast into the binary add while keeping standalone FP32 softplus. Coordinator reports a passing pair at1.5027 ms. |
| `BetaChainGDN` | Arithmetic decode only: one unary chain for sigmoid, explicit FP32→BF16 round-to-nearest-even, then BF16→FP32 output. The explicit rounding preserves the original BF16 sigmoid materialization boundary. |
| `UnroundedBetaChainGDN` | Diagnostic control that removes that explicit BF16 rounding; a result from it cannot be attributed solely to dispatch removal. |
| `GateChainGDN` | Combine the independently measured GateAdd and rounded BetaChain folds. |
| `JointQKNormGDN` | Arithmetic decode: one unweighted RMSNorm on the existing adjacent expanded QK tensor, then slice Q/K and retain their original BF16 scalar/cast boundaries. No extra concatenation. |
| `JointQKNormBeforeRepeat` | Same joint normalization, moved before repeat_interleave. Per-head width128 normalization commutes with head repetition and processes half as many norm rows. |
| `BiasProjectionGDN` | Same packed projection with an FP32 bias tensor that is zero except in the A columns. Output stays BF16; standalone gate bias add is removed. This moves the A rounding boundary and requires gate/core localization. |
| `BiasProjectionFP32GDN` | FP32 packed-output adaptation of that same bias fold. Keep A+bias FP32; restore QKV/Z/B fields to BF16 immediately after slicing. Increased output traffic/casts may erase the fusion benefit. |

Unary source support for the beta chain: `eltwise/unary/unary.cpp:29–56` chooses output dtype from the final TYPECAST and enables FP32 destination accumulation when output is FP32. `unary/common/unary_op_utils.cpp:568–580` accepts two explicit dtype parameters. Blackhole `ckernel_sfpu_typecast.h:247–264` implements FP32→BF16 RNE by rounding and masking low16 bits while storing FP32, so an internal BF16 rounding boundary is expressible without a separate tensor write. The actual sigmoid approximation under the changed destination mode still needs the paired/raw-gate check.

Bias folding source support: `matmul.cpp:255–260` accounts for the supplied bias's own data format; device bias validation requires TILE layout, batch1 and compatible padded width/height (`matmul_device_operation.cpp:366–417`). The model bias tensors satisfy those shape rules. Algebraic equality does not preserve the old BF16 A projection before the FP32 add: BF16-output fusion rounds after bias instead, and FP32-output fusion removes the original pre-bias rounding. These are unavoidable API ordering differences to measure, not a license to change the final stage's precision policy.

The proposed final-conv `addcmul(..., activation=SILU)` is unavailable: its nanobind signature exposes only value, memory_config and output_tensor (`eltwise/ternary/ternary_nanobind.cpp:214–228`). Likewise beta is a tensor, so delta=(V−read)×beta cannot be represented by a scalar post-unary multiplier on subtract. No unsupported keyword was added.

## Batch32 accuracy localization changes H1/H3 verdicts

New coordinator evidence supersedes the earlier B1-only optimism:

- Original functional complete matrix passed85 tests.
- KDAConvGDN batch32 traced decode fails per-user HF PCC at0.993153, below0.995 (`logs/kda_conv_matrix.log:226`). The exact fixture prefills63 tokens from seed31 and replays three distinct decode inputs from seeds3100–3102.
- FlatGDN alone also fails that exact fixture at0.99420974 (`logs/flat_gdn_batch32_trace.log`). Thus KDA conv is not the only source of drift.
- `HybridNormGDN` passes the exact batch32 trace fixture. It restores functional Q/K head split→BF16 L2 normalization→DRAM rank4 inputs, while retaining flat V and head-major core output. This verifies that raw flat Q/K normalization must not be promoted unconditionally from B1 paired/core results.
- `HybridCombinedGDN` still fails at0.992974, so restoring Q/K normalization alone does not establish the combined path.
- `HybridConvOnly` fails at0.992571 with original separate projections and functional decode. This isolates an additional KDA prefill convolution accuracy difference. No KDA path should be promoted on the earlier B1 passes alone.

The hybrid API is explicitly supported: `chunk_gated_delta_rule.cpp:159–188` determines `flat_qk` and `flat_v` independently, and rank4 Q/K take the original normalization contract. Flat Q/K instead trigger in-kernel q²→row-sum→rsqrt(+epsilon)→scaled Q/K (`device/kernels/compute/chunk_gdn_prep.cpp:406–425`). That differs from the functional BF16 RMSNorm, BF16 scalar product, and subsequent scale ordering. The passing hybrid exact fixture is stronger evidence than a broad claim that the fused norm is mathematically equivalent.

The appended isolation classes are:

| Candidate | Prefill conv / QK / output | Decode |
|---|---|---|
| `HybridNormGDN` | Original FIR; functional rank4 Q/K; flat V/head-major RMSNorm output; separate projections. | Original functional. |
| `HybridCombinedGDN` | KDA conv; functional rank4 Q/K; KDA output norm; packed projection. | Arithmetic. |
| `HybridConvOnly` | KDA conv only; otherwise HybridNorm. Setup retains original weight keys. | Original functional. |
| `HybridArithmetic` | Original FIR; functional rank4 Q/K; head-major RMSNorm output; packed projection. | Arithmetic. |
| `HybridKdaNorm` | Original FIR; functional rank4 Q/K; KDA output norm; separate projections. | Original functional. |

The two remaining isolated controls were still running when these notes were written. Full per-user HF trace checks, raw recurrence and conv-state comparisons remain necessary after choosing a final combination. Do not use an implicit batch-size fallback to hide either failure.

## Focused convolution boundary probe and ordinary Conv1d alternative

`probe_linear_conv_equivalence(functional,kda_candidate,prompt,compute_kernel_config=None)` is an untimed diagnostic. Recommended prompt is `H.make_activations(32,63,seed=31)` before prefill, using the two already-built real-weight linear decoders. It reads the functional history without changing either decoder's state. It runs the functional input RMSNorm and QKV projection once, and feeds those exact BF16 activations/taps/history to both convolution paths. This excludes projection packing, Q/K normalization and recurrence as confounders.

The helper returns every user's and Q/K/V field's PCC, maximum absolute error, relative L2 and finite status; it identifies the worst user. It also captures the functional pre-SiLU accumulator, compares both outputs against a Torch FP32 depthwise-convolution+SiLU oracle, and compares standalone TTNN SiLU against Torch SiLU on the same BF16 accumulator. The oracle begins at identical quantized QKV/taps/history; it is a convolution-boundary oracle, not a replacement whole-HF-layer reference. Correlate the reported user with the failing final trace user to test amplification through recurrence.

Source predicts a real rounding difference: KDA's final tap executes SiLU before the final BF16 pack (`qkv_causal_conv1d_silu/device/kernels/compute/qkv_causal_conv1d_silu.cpp:45–84`), whereas functional `addcmul` materializes BF16 before standalone SiLU. Both KDA intermediate partial buffers are BF16, but its FPU multiply/add path and final unrounded accumulator differ from the standalone ternary/SFPU path. The optional helper compute config is applied only to KDA, allowing an accumulator-mode localization control without changing matmuls/state/norm. `kda/factory/kda_factory_utils.cpp:103–108` forbids packer_l1_acc but permits fp32_dest_acc_en=False; the conv validator separately requires math_approx_mode=False. That legal control is not a precision-policy fix, and no existing candidate's compute config was mutated.

`Conv1dGDN` is a separate ordinary depthwise-convolution experiment deriving HybridNormGDN. It preserves separate projections, standalone BF16 SiLU, functional Q/K normalization and functional decode. Conv1d receives explicit three-row history followed by the physical prompt, padding0, groups=channels, kernel4 and stride1; only the real logical tail is written to the original persistent buffers.

The dedicated ordinary depthwise path supports height sharding only (`conv2d_utils.cpp:1244–1246`). The candidate uses four independent2048-channel groups to bound the width footprint, with32-row activation blocks. Each group's weights are prepared at state setup for every supported physical prefill length128..pos_ramp length, using `prepare_conv_weights` with the same BF16, HiFi4 and output policy as forward. At the default2048 chunk this creates64 prepared weight objects; report setup/memory costs if retaining the experiment. Q/K groups return directly, and the two V groups concatenate once. No host upload occurs in its forward or inherited decode trace path.

Although `Conv1dConfig` has an activation field and the generic factory emits activation defines (`conv2d_op_sharded_program_factory.cpp:919–923`), the specialized `compute_depthwise_conv1d.cpp` has no activation-macro execution. The candidate therefore keeps standalone SiLU. Source coverage includes grouped Conv1d tests at wide channels in `tests/ttnn/unit_tests/operations/conv/test_conv1d.py`; the exact Ornith BF162048-group setup still needs the coordinator's device check and measured PCC/latency. Its changed FIR accumulation may also fail, so it is not an assumed replacement for the passing functional FIR.

All follow-up candidates/helpers were formatted only in their appended region using Black's Python API (line_length120, target py310) and syntax-compiled. A static class-only reconstruction validated all36 C3 method resolution orders without importing TTNN or opening devices. Prefix comparisons verified that previously tested code stayed unchanged. No production runtime, shared harness, C++ source, device state or library precision policy was changed by this author.

## Ordinary Conv1d L1 retry

Coordinator's first `Conv1dGDN` paired run fails in auto-slice selection before convolution correctness can be measured (`logs/conv1d_gdn_v1.log`). The input is B1 with2048 physical prompt tokens plus three history rows, and each independent depthwise group has2048 channels. The reported free L1 is1,436,672 bytes.

The log explicitly says width slicing failed before the final height-slicing failure. `sliding_window/op_slicing/op_slicing.cpp:225–263` automatically retries the other axis; the final message mentioning height1 is therefore the fallback failure, not evidence that width slicing was omitted. The L1 footprint still exceeds capacity at the smallest supported slice. The32-row activation block is already the minimum tile-aligned override, so narrower independent channel groups are the bounded shape adaptation.

Two appended classes preserve the failed candidate unchanged:

- `Conv1dNarrowGDN`:512 channels per depthwise call,16 channel groups. Q/K/V assemble from4/4/8 groups.
- `Conv1dNarrow256GDN`:256 channels per call,32 groups. Q/K/V assemble from8/8/16 groups.

Both retain BF16 inputs/weights/output, the existing HiFi4 compute config,32-row activation blocks, automatic DRAM slicing, functional SiLU, functional Q/K normalization and functional decode. Setup remains inherited and prepares all physical prefill lengths before forward. Field concatenation now follows each configured Q/K/V width; it cannot accidentally apply the original four-group indexing to a narrower adapter. With the default2048 prefill chunk these classes prepare256 and512 weight objects respectively. Extra launches/setup are explicit performance costs to measure after the smallest valid retry.

Prediction: narrower groups reduce channel-dependent activation/weight/partial buffers enough for auto-slice selection to succeed. First rerun the exact real-weight paired check with `conv1d_narrow` selecting `Conv1dNarrowGDN`; if legal, collect output/core/state PCC and warmed prefill latency. Use256 only as an isolated additional footprint control if512 still cannot fit. A successful allocation alone is not an accuracy or speed pass.

Coordinator also completed the optional KDA accumulator localization. FP32-destination-disabled KDA was worse against functional convolution: relative L2 approximately0.00510 versus0.00481 for the existing FP32-destination control; maximum absolute error remains0.0625 (`conv_localization_v2` diagnostic). Both diagnostics completed, but this refutes using the smaller destination mode to restore the missing functional rounding boundary. No precision change was retained.

Both narrower classes were appended, Black-formatted and syntax-compiled; all class MROs validate. A byte-for-byte prefix comparison confirms the original failed Conv1d class and every previously tested method stayed unchanged. No hardware was accessed by the candidate author.

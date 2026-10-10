# GDN precision hypothesis and local candidate

Inspection and implementation date: 2026-10-10 UTC. This investigation did not
import the target or TTNN, open a device, run a hardware test, or modify shared
Qwen36/TTNN code. The parent explicitly expanded the initial read-only task to
allow the one implementation file `tt/functional_gdn.py`. Hardware results below
are read from the parent's artifacts, not produced by this investigation.

## Failing contract and controls

Every real-weight decode token must meet PCC >= 0.995 after a 4096-token prefill.

| Existing artifact | Prefill PCC | Aggregate decode PCC | Minimum token PCC | Tokens below 0.995 |
|---|---:|---:|---:|---:|
| `linear_4096_128_per_token.json` | 0.9976155162 | 0.9950028658 | 0.9701313972 | 38/128 |
| `linear_alpha0.json` | 0.9989612103 | 0.9975501299 | 0.9866248369 | 16/128 |
| `linear_alpha0_bf16experts.json` | 0.9989665151 | 0.9975613356 | 0.9866843224 | 16/128 |

The alpha control establishes that diagonal damping contributes to the error,
but removing it does not restore the contract. Changing expert precision barely
changes that residual failure. Neither observation alone proves which GDN
precision boundary causes the remaining error. The original per-token series
has lower average PCC in its second half (0.9942979384) than its first
(0.9967554212), but that trend is not a causal state-error measurement.

## Exact reached precision path

1. `models/demos/blackhole/qwen36/tt/gdn/weights.py` loads QKV, a, b, z and output
   projection weights as BFP8. It constructs its mega weight by downloading those
   already-quantized tensors and repacking them as BFP8. Upcasting this mega
   weight would not reconstruct a BF16 checkpoint baseline.
2. `gdn/gated_deltanet.py` configures LoFi projections with FP32 destination
   accumulation. The default matmul output dtype is still its first input's
   dtype (`operations/matmul/device/matmul_device_operation.cpp:2980`), so
   BF16 activation inputs produce BF16 stored projection results.
3. `ttnn_gated_deltanet.py` computes `a + dt_bias`, softplus, `A_neg`, and beta in
   those inherited dtypes. HF explicitly computes `A_log.exp()` and
   `softplus(a + dt_bias)` in FP32 (`modeling_qwen3_5_moe.py:514–516`).
4. Chunk preprocessing and `ttnn.transformer.gated_delta_attn_seq` use FP32
   tensors, but `ttnn_delta_rule_seq.py:199–203` defaults the returned final state
   to BF16. `gdn/decode.py:82–86` copies it into the class's persistent BF16
   buffer. Merely setting `QWEN_GDN_FP32_STATE=1` leaves that BF16 destination.
5. Decode's reached implementation is the Python TTNN composite
   `recurrent_gated_delta_rule_decode_inplace_ttnn`, not the newer fused native
   op. `ttnn_gated_deltanet.py:718–727` omits its `high_precision` argument, so
   it remains False. State decay, delta and state-write intermediates therefore
   retain BF16 storage; the k/state and q/state matmuls use HiFi2/FP32
   accumulation but still write the input dtype. The final copy again targets
   persistent BF16 (`ttnn_delta_rule_ops.py:383–510`). HF keeps this recurrence
   and its returned state FP32 (`modeling_qwen3_5_moe.py:325–366`).

This proves an independent repeated quantization mechanism remains after
alpha=0. For example, a BF16 state value 1 multiplied by exp(-0.001) rounds back
to 1; repeated FP32 decay reaches about 0.87985 after 128 steps. That scalar
witness shows the mechanism can matter, not that these exact checkpoint heads
have that decay. Projection errors also alter q/k/beta/g at every step and can
change the downstream MoE routes. Target intermediate/state measurements are
still needed to allocate the observed error between those mechanisms.

## Existing API for exact alpha

The reached seq adapter has **no alpha argument**: it reads
`QWEN_GDN_DIAG_ALPHA` and `QWEN_GDN_INV_DOUBLING` inside preprocessing. The
lowered `gated_delta_attn_seq` op only consumes the preprocessed system. Neither
provides a caller-local alpha hook.

`ttnn.transformer.chunk_gated_delta_rule` directly implements the required
unit-diagonal recurrence. `chunk_gdn_math.hpp:307–337` forms
`-strictly_lower(k_beta @ k.T * decay_mask)` by multiplying its diagonal by zero;
lines 798 onward solve `I + strictly_lower`. Thus exact alpha=0 requires no
alpha parameter or copied preprocessing module. Its explicit API accepts
caller-owned `eye`, `tril`, `ones`, `masks`, `initial_state`,
`compute_kernel_config`, `program_config`, and `wy_inverse`. The latter chooses
inverse arithmetic, not diagonal damping. AUTO uses the SFPU forward solve on
Blackhole at chunk size 32 and Horner on other supported paths. FP32 final state
is retained. Q/K/V are explicitly BF16 inputs; this is not an all-FP32-input op.

The corresponding `ttnn.transformer.fused_recurrent_gated_delta_rule` accepts
device-normalized Q/K, FP32 persistent initial state, `output_final_state=True`,
and an explicit HiFi4/FP32 compute config. Its device validation requires FP32
inputs and state; its public wrapper promotes activations on device, and all
kernel circular buffers and outputs are FP32. The per-token recurrence follows
decay, read, delta, rank-one write, then output read. Supplying the initial state
avoids the eager zero-state fallback. The source's phrase "host-normalized"
means normalization outside the native kernel; on-device TTNN RMSNorm suffices.

## Implemented local AutoFix candidate

`tt/functional_gdn.py` defines
`FunctionalGatedDeltaNet.from_existing(gdn, *, state_dict)`. This owns local
composition rather than modifying process globals or shared module functions.

- It loads BF16 mega/output projection weights directly from the remapped
  checkpoint substate, uses HiFi4/FP32 projection accumulation, and builds
  FP32 decay constants at setup.
- Four FIR taps accumulate in FP32, with the BF16 convolution/SiLU boundary
  and BF16 history retained. No optional helper can upload a mask or tap at
  runtime.
- Prefill calls the exact native chunk op with internal chunk size 32, phased
  execution, and all four setup-created constant tensors. The outer decoder
  already supplies 128-token chunks or single-token tails.
- Decode calls the FP32 native recurrent op. Q/K normalization is a local
  device RMSNorm calculation. Both modes copy final state into the same
  preallocated FP32 allocation, preserving trace address identity.
- The output path explicitly follows HF's BF16 recurrence-output boundary,
  FP32 RMSNorm, BF16 normalized-weight product, FP32 SiLU gate product, then
  BF16 projection input. It has no inherited clipping or alpha environment
  dependency.
- `.recurrent_state`, `.fused_conv_state`, `._zero_recurrent`, `._zero_conv`,
  `.forward`, and `.reset_state_inplace` retain the interfaces consumed by the
  existing per-request slot manager. Request-slot state allocation must happen
  after the replacement so every slot receives FP32 recurrent storage.

Integration immediately after Qwen36DecoderLayer construction, for the GDN
branch only, and before `_prepare_request_buffers`:

```python
from .functional_gdn import FunctionalGatedDeltaNet

prefix = f"layers.{layer_idx}.linear_attn."
layer.attention = FunctionalGatedDeltaNet.from_existing(
    layer.attention,
    state_dict={key[len(prefix):]: value for key, value in remapped.items()
                if key.startswith(prefix)},
)
```

## Validation and remaining uncertainty

Performed: `ast.parse` of the new module; AST inspection of every class method
except the setup factory found no Torch call, `ttnn.from_torch`,
`ttnn.to_torch`, or `ttnn.as_tensor`. Inspected the exact C++/nanobind signatures,
native state/output dtypes, supplied-constant handling and grouped-head paths.
No target import or hardware run was performed by this subagent.

The parent subsequently integrated the candidate and produced
`linear_native_smoke.json` / `linear_native_smoke.log`: real-weight 129-token
prefill PCC **0.9995185733**, two traced decode tokens with minimum PCC
**0.9997554421**, `runtime_audit=true`, and `repeated_identical=true`. The parent
also reports allocation tracking passed for that command. These are parent-run
smoke results, not this subagent's measurements or a full 4096+128/context gate.
The JSON's `gdn_diag_alpha="0.25"` field records an inherited environment value;
the new native recurrence does not read that environment setting and its
effective diagonal is exactly unit diagonal. Runner provenance should identify
the new local GDN file and the effective native recurrence explicitly.

The final source recheck found no output-layout or gated-norm cast-order defect:
`chunk_gated_delta_rule.cpp:415–434` returns `[B,T,HV,V]` ROW_MAJOR, which the
candidate tilizes before norm. With this checkpoint HV=32, flattening its last
two dimensions yields the intended `[1,T,4096]` projection input. The BF16
output/norm-weight boundaries and FP32 SiLU-gate order match the installed HF
module's explicit casts. Native FP32 Q/K normalization can differ in rounding
from HF's pure-Torch BF16 helper; the real-weight PCC gate remains the numerical
acceptance test, and this is not evidence of a shape or recurrence error.

Parent validation next: 4096+128 per-token PCC and state comparison. Run repeated replay and
allocation-tracking checks after preparing every supported path. A failure
should be narrowed at convolution output, q/k, g/beta, recurrent state and
GDN output before testing another unrelated decoder component. The candidate
deliberately changes several independently incorrect precision boundaries;
passing it establishes the explicit functional policy, not a measured
attribution of all old error to one boundary. Long-context stability and
complete-layer performance remain unmeasured for this candidate.

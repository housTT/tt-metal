# Component diagnosis: norms and router sensitivity

Date: 2026-10-10 UTC. This subagent inspected source and existing artifacts and
wrote `tests/run_components.py`. It did not import the target/TTNN, open a
device, or produce hardware measurements. The commands below are for the parent
to run serially when this diagnostic is needed.

## Contract and existing observations

The real-weight layer-0 4096-token prefill followed by 128 decode steps must pass
PCC >= 0.995 for every decode token. The exact native GDN candidate improved
`linear_native_4096_128.json` to prefill PCC 0.9995970726 and aggregate decode
PCC 0.9991455674, but zero-based decode steps 71 and 109 were respectively
0.9942064285 and 0.9943133593. Changing routed expert weights to BF16 did not
resolve these two tokens, according to the parent control.

During this investigation, the parent repaired both zero-centered layer norms
locally. Its `linear_native_fp32norm.json` records prefill PCC **0.9996982813**,
aggregate decode PCC **0.9992589355**, and minimum token PCC **0.9952653050**.
All 128 tokens pass, with inherited expert precision, traced decode,
`runtime_audit=true`, and repeated identical replay. These are parent-run
artifacts; they close this particular numerical gate, not all layer capability
or stage gates.

## Source hypotheses

`models/common/rmsnorm.py:85–90` takes the checkpoint BF16 weight, adds 1 before
upload, and uploads BF16. The addition has already rounded its result to BF16.
The reached common norm uses HiFi2 and BF16 input/output storage. In contrast,
installed HF `Qwen3_5MoeRMSNorm.forward` computes normalization in FP32,
`1.0 + weight.float()` in FP32, multiplies those FP32 values, then casts the
final result to the activation dtype. The norm repair changes both normalization
and affine precision, so the passing whole-layer control does not separately
measure the contribution of the weight addition alone.

This difference affects the input to recurrent GDN and the post-attention norm
input to the MoE router. Small continuous errors can change top-8 expert
membership when the eighth/ninth scores are close; a discrete route change can
amplify otherwise small GDN/norm errors into a larger decoder-output delta.

`models/demos/blackhole/qwen36/tt/moe/router.py:46–60` computes matmul logits,
softmax, top-k and selected-weight normalization without an FP32 output dtype.
BF16 input therefore leaves stored logits/probabilities BF16. HF's
`Qwen3_5MoeTopKRouter.forward` deliberately performs softmax and selected-weight
normalization in FP32, then casts selected weights to the logits dtype. This
is another plausible tie/selection mechanism, but **no router repair is justified
solely by this source observation**. The current norm repair passes the failing
workload. Future anomalies should use the controls below to locate the first
divergence before changing router behavior.

## Diagnostic runner

The runner loads the exact checkpoint layer/config and uses the same seed/input
distribution as `run_layer.py`. It consumes all 4096 prompt tokens and all 128
decode tokens on both implementations. It calls the actual reference decoder
and actual TT decoder, recording selected boundaries through reference hooks and
TT per-instance probes. No process-global function is patched.

Default zero-based diagnostic steps are 70, 71, 72, 108, 109, 110, covering each
previously failing token and its neighbors. The JSON contains:

- Prefill metrics and final-output PCC for every decode token.
- Selected-step PCC, maximum absolute difference and relative L2 error for
  attention input norm, GDN output, attention residual, FF norm, MoE output,
  router logits, and final layer output. Component minimums explicitly cover
  selected steps only.
- Actual HF and TT top-8 IDs, overlap counts, and eighth/ninth logit/probability
  margins. TT selected IDs come from the production router's actual dense
  routing tensor, preserving its real top-k decisions.
- Host FP32 softmax applied to TT logits, and its overlap with actual TT routes,
  separating a BF16 softmax tie from differences already present in logits.
- HF router on the TT FF-normalized input and actual TT router on the HF
  FF-normalized input, separating upstream input errors from router arithmetic.
- HF FF norm and HF MoE evaluated on the corresponding TT inputs, separating
  local component error from error inherited from the preceding component.
- Source hashes captured at startup and a check for source changes during the
  command. No tensors, raw activation dumps or timing claims are persisted.

Router logits are observed through a duplicate stateless matmul with the actual
router's weights and compute config. The runner explicitly requires the current
`Qwen36Router` class; if its implementation changes, update this probe before
using its logit comparisons. Actual routes and actual MoE output always come
from the unmodified production router/experts calls.

## Commands and interpretation

Run current norms:

```bash
python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.run_components \
  --output models/autoports/qwen_qwen3_5_35b_a3b/doc/functional_decoder/components_current.json
```

For a direct old/new norm diagnosis, the optional control reconstructs the exact
original common RMSNorm in this test instance before prefill. GDN, router,
experts, target shapes and input sequence remain the same:

```bash
python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.run_components \
  --norm-policy inherited \
  --output models/autoports/qwen_qwen3_5_35b_a3b/doc/functional_decoder/components_inherited_norms.json
```

A cheap integration smoke can use `--length 129 --steps 2 --selected 0,1`, but
only the default workload preserves the sequence history of steps 71/109.

If actual TT routes disagree with host FP32 softmax on the *same TT logits*,
inspect softmax/top-k storage/ties. If HF-on-TT-input and actual TT routes differ,
the difference lies in router arithmetic or tie handling. If those same-input
routes agree but original HF/TT routes differ, upstream activation error is the
route-change trigger. If routes match and `moe_same_tt_input` still diverges,
investigate expert/shared-expert arithmetic separately. An ordering difference
inside the same selected expert set is not itself a wrong routing decision.

## Validation boundaries

Performed here: Python AST parsing and `python3 tests/run_components.py --help`.
The module's top-level imports are standard-library-only, so this validation did
not import Torch, TTNN or the target implementation. Source inspection confirms
that the probes read outputs before the parent forward deallocates them and
that stateless router controls cannot advance GDN state.

Intermediate host reads deliberately synchronize work and are incompatible with
a measured device-only forward or traced performance test. The result sets
`runtime_audit=false`, `traced=false`, `performance_measurement=false`, and
`host_reads_between_components=true`. It supplements causal diagnosis and must
never substitute for the regular runtime audit, traced PCC or profiler evidence.
At report creation no component diagnostic hardware command had been run by
this subagent, and no per-component measurement is claimed.

# AutoDebug: mixed-format fused Z gating

Inspection date: 2026-09-05. Fresh, isolated source investigation requested by
AutoFix. No device commands, TTNN imports, builds, or kernel/model implementation
edits were run for this report. The parent owns the serialized hardware lane.

## Starting evidence

Reproduction supplied by the parent, from the repository root:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/record_run.py packed_gdn_fused_z_layer0 timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 0 --length 2048 --variant packed_gdn_fused_z
```

Read the archived logs and provenance, current work log, model AGENTS.md,
AutoFix/AutoDebug instructions, and current source. Both source archives identify
HEAD `65abe7f69dbf128012108ef99262bf337f7cdc70`; the candidate/probe work is
uncommitted. Preserve that work. Hardware provenance in the logs is four Blackhole
chips on P300c boards, logical 1x4 ring.

| Recorded candidate | Prefill PCC, each rank | Decode PCC, each rank | Eager/trace max difference | Exit |
| --- | --- | --- | --- | --- |
| `packed_gdn_shared_mlp_layer0` | 0.9999630806578761 | 0.9999879890284807 | 0 | 0 |
| `packed_gdn_fused_z_layer0` | 0.9999630806578761 | -0.08527425494792604 | 0 | 1 |

Evidence: `logs/packed_gdn_{shared_mlp,fused_z}_layer0.log.gz` and their
`.provenance.json` files. The failing assertion is the existing decode PCC >=0.995
gate. A deterministic, catastrophic correctness regression is present. Trace
exactness does not establish correctness, and this is not ordinary precision loss.

## Headline finding: RHS activation reads BF16 with the initial FP32 unpack format

**Verdict: concrete source contract violation; hardware causality pending the
focused experiment below.**

The failing candidate moves Z's SiLU from a separate BF16 unary op into
`ttnn.multiply(merged, z, input_tensor_b_activations=[SILU])` in
`tests/optimized_multichip_candidates.py`. The inherited decoder has the same
RMSNorm, head permutation, reshape, and output projection. `merged` is FP32 from
the recurrent result and RMSNorm; packed projection Z is BF16. No additional
SiLU is needed in the failing candidate: the candidate deliberately returns raw
Z and applies it once at the output boundary.

The lowered path is:

1. `ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp`,
   `invoke_binary_ng_impl`, retains operand order and floating input dtypes here.
   Its integer promotion branch does not apply. Output dtype defaults to the
   FP32 LHS; layout remains tiled. No associative swap moves SiLU onto the LHS.
2. `binary_ng/device/binary_ng_device_operation.cpp`, `is_binary_sfpu_op`, chooses
   SFPU MUL when fast/approximate mode is false/default. Equal logical shapes
   select no broadcast.
3. `binary_ng/device/binary_ng_program_factory.cpp` creates FP32 CB0 for the LHS,
   BF16 CB1 for RHS input, BF16 CB4 for activated RHS, and FP32 CB2 for output.
   The RHS intermediate preserves its input dtype, so this fusion still has a
   BF16 activation materialization boundary. FP32 destination accumulation is
   correctly enabled when either input or output is FP32; its absence is not
   the present source defect.
4. `binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp:103`
   starts hardware with `compute_kernel_hw_startup(cb_post_lhs_id, cb_out_id)`.
   With no LHS activation, `cb_post_lhs_id` is FP32 CB0. LHS preprocessing is
   compiled out. RHS preprocessing runs before the binary input copies.
5. `binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp:19`,
   `preprocess_sfpu_impl`, reconfigures only the **packer**, then calls
   `copy_init(cb_pre)` and `copy_tile(cb_pre, ...)` for BF16 CB1. It never
   reconfigures SrcA to BF16 before this read.
6. `tt_metal/hw/inc/api/compute/tile_move_copy.h:25` explicitly documents that
   `copy_init` does **not** reconfigure unpacker data types. Its implementation
   follows that contract. The deprecated `copy_tile_to_dst_init_short_with_dt`
   wrapper performs an explicit `reconfig_data_format_srca` before `copy_init`.
7. The binary kernel later calls explicit SrcA format reconfiguration before
   copying each operand for multiplication. That occurs after SiLU already
   consumed the incorrectly configured RHS input and stored it in CB4.

The preprocessing helper's comment that downstream binary copies switch formats
does not address the helper's own earlier input read. This is a caller contract
violation at the helper/`copy_init` boundary, not evidence that the public copy
API should implicitly change its documented behavior.

### Concrete first-work-unit ledger

The checkpoint has 32 value heads of width128. TP4 partitions eight heads per
rank: both multiply inputs have logical `[1,1,1024]`, tiled physical
`[1,32,1024]`. There are 32 output tiles/rank. With interleaved inputs/output and
the default full worker grid, `num_tiles_per_cycle=1`; the work splitter assigns
one tile to each of 32 active cores. The diagnostic must print the actual
metadata to confirm these source-derived values.

| Phase on each active core, first and only tile | SrcA format state | Buffer action |
| --- | --- | --- |
| Hardware startup on CB0 | FP32 | No input consumed |
| LHS preprocessing absent | FP32 | CB0 left for binary multiply |
| RHS preprocessing | **Still FP32; CB1 needs BF16** | Wait CB1(1), reserve CB4(1), copy/SiLU/pack, pop CB1(1), push CB4(1) |
| Binary input copies | Explicitly changes to FP32 then BF16 | Wait CB0/CB4(1), copy both, multiply |
| Output | BF16 SrcA; FP32 packer restored | Push CB2(1), pop CB0/CB4(1) |

The buffer counts balance. The format state at the RHS preprocessing read does
not. Because every active core starts with that state and processes one tile,
the fault can affect the entire logical row rather than a small tail. It is
deterministic in eager and trace execution.

## Smallest verify/refute experiment

Capture device clones of the real recurrent `core` and raw BF16 Z at layer0's
decode output boundary after the same 2048-token prefill. Preserve their local
TP4 metadata and reconstruct the unchanged norm/permute/reshape once. On exactly
those same tensors compare:

| Case | Operation | Prediction if the format hypothesis causes the failure |
| --- | --- | --- |
| Control | BF16 standalone SiLU(Z), then multiply FP32 merged | Matches CPU oracle and passing model |
| Original fused RHS | `multiply(merged_fp32, z_bf16, rhs=[SILU])` | Large local gating error |
| Same-format control | `multiply(merged_fp32, typecast(z_bf16, fp32), rhs=[SILU])` | Error disappears; small SiLU rounding differences allowed |
| Operand-order control | `multiply(z_bf16, merged_fp32, lhs=[SILU], dtype=fp32)` | At this one-tile/core shape, first activation input uses startup BF16 format and should pass |

For the swapped control, force output FP32 and the original output memory config
so consumer precision/layout do not become additional variables. This order
control is not a proposed general fix: multiple iterations per core can leave
SrcA in the other operand's format before the next activation pass.

Log all-rank PCC, max absolute error, finiteness, logical/padded shapes, dtypes,
layout and memory config for the captured values and each result. Compare the
gate before `gdn_out`, then pass each gate through the unchanged `gdn_out` path
to establish the downstream consequence. The CPU oracle uses the captured
FP32 normalized values multiplied by BF16-rounded SiLU of captured BF16 Z.

If original fused RHS is locally correct, this root-cause attribution is
refuted for the supplied failure even though the source contract issue remains;
next inspect original-vs-captured operand ownership and the output projection.
If only FP32 Z passes, that proves sensitivity to this boundary, not inherent
BF16 numerical instability. The operand-order control and explicit kernel
format repair give stronger causal separation.

## Intervention and verification boundaries

The smallest general kernel intervention is to make the activation helper
configure SrcA to the actual `cb_pre` format before its own copy. A one-argument
format reconfiguration can avoid assuming which operand format is live. Audit
the following binary copy's assumed prior format and all shared scalar/broadcast
callers before integrating a general fix. Do not change `copy_init` globally or
blindly copy the FPU helper's pre/post arguments: RHS post and pre are both BF16
here, so that comparison would not describe the live FP32 state.

A model-local FP32 Z cast or order change is only an experiment/workaround until
the hardware comparison passes and whole-layer correctness and latency are
remeasured. The general kernel fix would require the repository's prescribed
build plus focused mixed-format activation tests. No kernel edit is made here.

Keep the existing precision policy: BFP4/LoFi projection groups, BF16 residual,
activation and CCL payload, FP32 GDN state/normalization, local paged KV64 policy
unchanged (KV is unused in this linear-attention layer). Packing, collective,
state allocation, input sequence, and weights are identical between the named
failing/passing candidates. Run the original PCC gate and trace checks after
any proven model intervention, then nearby batch coverage if that intervention
changes operand-order or work-unit assumptions. No performance claim is made.

## Diagnostic handoff

At the parent's subsequent request, added only
`tests/multichip_z_diagnostic.py` for the experiment above. It executes the
unchanged failing candidate once, retains device clones of its boundary inputs,
and emits comparison metadata/metrics without writing raw tensors. This is a
diagnostic, not a model fix or a passing correctness gate. Suggested parent run:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/record_run.py z_fusion_boundary timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --layer 0 --length 2048
```

Host verification completed: Python AST parse; `python_env/bin/python -m black
--check --target-version py310 models/autoports/ornith_ai_ornith_1_5_9b/tests/multichip_z_diagnostic.py`;
`git diff --check`. No build is needed for this Python/docs-only addition. The
diagnostic has not been executed by this investigator; the parent must record
its device result in the stage work log and adjudicate this hypothesis.

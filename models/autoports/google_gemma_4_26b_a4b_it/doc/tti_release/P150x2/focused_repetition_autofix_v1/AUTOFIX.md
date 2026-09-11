# AutoFix disposition: focused Meta GPQA repetition investigation

**Final status: the original repetition/answer-completion failure remains.**
The combined prefill-and-decode expert-gate intervention failed the unchanged
case. No model precision policy is retained. Three independently verified
sparse-matmul contract repairs remain committed, but they are not a repetition
repair. This bounded investigation has reached an attribution limitation; it
does not establish an inherent checkpoint defect or pass quality/Stage11.

## Starting evidence and scope

The latest diagnosis was
`W/meta_gpqa_after_expert_quality_diagnosis_v1/AUTODEBUG.md`: exact prompt/template
audit passed, decode-only gate correction failed quality, and the omitted
prefill gate phase supplied one distinct remaining component hypothesis.
`W=/home/hous/dev/tti-release-gemma4`.

The preserved failing case is Meta GPQA doc58, 326 native prompt IDs,
max2048, T0, seed42, stop[], context262144, 32 slots, HMA4128 and active B1.
The user prohibits CPU model inference/evaluation, changed evaluation recipes,
lowered thresholds and broad replacement evaluations. I reread that constraint,
read the complete latest continuation, inspected saved source/ownership bindings
and performed host tokenization only. Root owns all hardware measurements below.

## Hypothesis experiments and dispositions

| Hypothesis | Discriminating evidence | Disposition |
| --- | --- | --- |
| Wrong/missing native prompt or instruct/thinking format causes this request's failure | Exact 326 IDs encode/decode to the saved fixture transport; checkpoint Jinja default and explicit `enable_thinking=False` render identically with one BOS. The fixture explicitly requests `## Step 1` and the final answer format. | No format defect found. Preserve the recipe. |
| Correcting the measured packed expert decode gate error sufficiently repairs repetition | Valid TT SFPU reference: mean gate error 2.4804%→0.1675%; post-MoE norm 1.3212%→0.3097%. All local guards passed. The actual decode-only full-case candidate still generated 2048 tokens without a final answer. | Local arithmetic improvement verified; sufficient quality repair refuted. |
| The omitted prefill gate phase prevents that correction from repairing the original case | Exact original-prefill capture and native replay passed. Mean prefill gate error 2.8341%→0.1675%; norm 0.9726%→0.3142%, both chips. The combined candidate then covered all prefill and B1 decode gate calls. It still generated 2048 tokens without a final answer. | Local prefill improvement verified; combined correction as a sufficient repair refuted. |

The component results are
`meta_gpqa_moe_gate_fp32_fidelity_run_v1/result.json` and
`meta_gpqa_prefill_gate_control_run_v1/result.json`.
The failed decode-only quality result is
`meta_gpqa_expert_precision_quality_control_run_v1/result.json`.
The earlier decode-SDPA quality control also failed. Prior cache, mask,
expert-index, serialization and exact native replay controls remain preserved;
none supplies a newly identified remaining implementation cause.

## Latest actual same-case result

`meta_gpqa_expert_both_phase_quality_control_run_v1/result.json` SHA256:
`c78f72a0ed0e46d7598d8537bd8c7b70a934a2d644c182c6972e243b12658abf`.

Complete candidate text SHA256:
`d52602d3ffd78c01c1f2c724bde96880aeeb4f7876f8c2a3cb21f338e9834326`.

Independent retained-artifact inspection confirms:

- Exactly 2048 saved output IDs, decoding exactly to the complete saved text;
  `finish_reason=length`; no native stop ID emitted and no required final answer.
- All 330 prefill gates and 61,410 decode gates executed under the candidate;
  all 61,410 decode SDPA calls retained their native configuration. Execution
  did not silently omit prefill or a decode layer.
- Native and candidate configuration objects, operand/program scopes and
  restoration were checked by the reviewed probe. The runtime and environment
  match the preserved full-case native/decode-only control.
- Actual duration 199.716 seconds; owned worker756543 returned0, cleanup recorded
  no survivors or signal errors, and the mesh closed cleanly. Root separately
  reported session45093 reaped and the worker absent. This reviewer performed
  no process-control or device action.

The continuation identifies cyclohexane but again treats Y as consisting only
of saturated compounds. It repeatedly considers incompatible C6/C3/C9
possibilities, returns to a hydrogen count of20 that is absent from the choices,
and ends during another calculation. Those observations come from the complete
text, not merely its missing answer regex. Some early wording changed, but the
requested outcome did not improve to a completed answer.

This result refutes **sufficiency of the tested intervention**. It does not prove
that expert numerical error never contributes, that every operation is correct,
or that the quantized TT graph equals a canonical checkpoint implementation.
The reference validates selected gate products using the actual resident
quantized operands; it is not an end-to-end canonical-model oracle.

## Verified repairs retained separately

Local commit `bc4752de65dce32b8e68eacd7ebcb40dac276cbd` contains only the sparse
factory changes and the durable sparse cancellation regression:

1. Size CB5 allocation/pages from the selected intermediate format, preventing
   a 2048-byte BF16-sized page for a 4096-byte Float32 intermediate.
2. Skip input-A multicast for a singleton receiver rectangle, preventing the
   proven zero-recipient NoC hang while retaining local publication/handshake.
3. Reload Float32 CB5 directly to destination under the matching FP32 condition,
   preserving the measured cross-block cancellation residue.

The focused before/after controls verified those defects and corrections;
native BF16 outputs stayed exact. The durable regression passed all4 cases.
The local sparse Watcher check passed with fabric disabled; the earlier
FABRIC_2D initialization-capacity failure remains disclosed. Required Docker
build wrapper execution was unavailable, and root's disclosed native
`ninja -C build -j2 ttnncpp` / runtime-install fallback succeeded. Full evidence
and limits are in `meta_gpqa_sparse_kernel_retention_review_v1/REVIEW.md`.
Unrelated working-tree changes are outside this retention claim.

## Remaining uncertainty and required direction

**No genuinely distinct implementation cause supported by the inspected source
and retained measurements was found after this failure.** The remaining unknown
is whether the exact request would succeed in an independently faithful model
execution, or whether another unlocalized TT computation difference changes its
trajectory. Neither explanation is established.

The successful R1 question remains an unsuitable causal control: it changes
instructions, answer ordering, sampling and budget, while using the same native
closed-thought mode. Its answer occurs beyond2048 re-tokenized tokens. That does
not show that this greedy Meta request would recover with more tokens, and it
does not authorize changing Meta's budget or thinking mode.

The current evidence does not justify another precision sweep, another copy of
the same full-generation experiment, or a new speculative production edit.
Continuing this particular repair chain now needs human/engineering direction
on a new source of discriminating evidence: for example, an independently
validated **same-checkpoint, same-request** execution available within the TT-only
constraint, or a new concrete operation-contract discrepancy. Such evidence
could establish a new localization target. None is claimed available here, and
no prohibited CPU reference or recipe change is requested or launched.

Preserve the failing quality finding and original acceptance gates. Keep the
verified kernel repairs, leave model precision unchanged, and do not label this
bounded diagnostic limitation as release readiness or a completed repetition fix.

## Commands and evidence verification

The recorded root-owned worker command for the final actual control was:

```sh
/home/hous/dev/tt-metal/python_env/bin/python -I -B /home/hous/dev/tti-release-gemma4/meta_gpqa_expert_both_phase_quality_control_v1/probe_tt.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_expert_both_phase_quality_control_v1/plan.json --plan-sha256 698b1524a915194faa2cff5850c90a4330db399e4f0724b86e17a03a1c8e1648
```

This reviewer did not rerun it. The independent retained-artifact check exited0:

```sh
/home/hous/dev/tt-metal/python_env/bin/python -I -B /home/hous/dev/tti-release-gemma4/meta_gpqa_focused_repetition_autofix_v1/inspect_retained.py
```

[EVIDENCE.json](EVIDENCE.json) binds the actual result, complete text, executed
plan/probe, owner record, local admissions and prior reports. The inspection
script blocks model/framework imports and performs no model arithmetic or new
inference. No build was required for these report-only artifacts.

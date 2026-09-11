# Combined prefill/decode gate quality-control source review

**No blocking findings in the frozen package.** It prepares one candidate on the
unchanged failing case and reuses the preserved native and decode-only
trajectories. This review does not establish an actual quality improvement,
authorize a production precision policy, or pass Stage11.

Directory:
`/home/hous/dev/tti-release-gemma4/meta_gpqa_expert_both_phase_quality_control_v1`.

| Reviewed artifact | SHA256 |
| --- | --- |
| `probe_tt.py` | `4784ff0acd9e7ac1a3c4ed35946473662cfc39bc12948e82b1fdd7799fb90120` |
| `plan.template.json` | `6a6eebd1a3b430c95571c4afb6e0d0311da7cfd9607774a51d49ead69306f59d` |
| `materialize_plan.py` | `21ff5cb285293ab213c6fb657afc38c4dd9b3a8c09ee1023b3ddc1535602ba71` |

## Scope and causal admission

The actual prefill component result is
`meta_gpqa_prefill_gate_control_run_v1/result.json`, SHA256
`34b23070daa5adeb3f6751b5cd76ff51f33805af884f04322f330593a47d9d38`.
I verified its retained native-logit, gate and downstream exact guards and all
fixed admission checks. Mean gate relative L2 improved
0.0283409069→0.00167508293, and normalized error improved on both chips.
That supports testing the previously omitted prefill phase together with the
already admitted decode configuration. It does not undo the failed decode-only
quality result.

The new validator pins that actual prefill result to its executed plan and
inherits the plan's source hashes. It also pins the previous full-case result
to its plan, current checkout commit and TT library; verifies saved baseline
IDs against the earlier baseline and checks both retained text artifacts. The
prefill component and saved full-case result report the same runtime library
and Python extension. Execution rechecks loaded runtime paths before opening
the mesh. Root owns final plan materialization and quiescence admission.

## Exact request and isolated implementation change

The original payload remains 326 native prompt IDs, max2048, T0, seed42, stop[],
non-streaming, context262144, 32 slots, HMA4128 and active width1. Exactly one
candidate arm runs. Fresh-page and HMA allocation guards remain; the 2048-token
limit and native EOS IDs `{1,50,106}` are unchanged.

Compared with the previously executed exact-baseline probe, host greedy
selection, TT logit readback, incremental detokenization and termination logic
are unchanged. The model remains on TT; host argmax is the preserved sampling
path. No CPU model inference, repetition penalty, answer recovery, native
thinking switch or budget extension is introduced. Since prefill now changes,
its first logits are recorded against history without incorrectly requiring
native equality.

Both `_moe_prefill` and B1 `_moe_decode_single_user` are wrapped on all 30 layers.
Each original method runs with only the admitted gate configuration:
HiFi4 and FP32 destination enabled. Packer accumulation, approximate mode,
destination sync, throttling and all other fields retain native values. The
same resident packed weight object serves both phases.

The sparse observer checks actual gate dispatches: K-block44, grid11×2,
per-core M/N1, subblock1×1, and `fuse_batch=False`. Prefill retains full 32-row
groups, non-indexed routing and DRAM output; decode retains its indexed B1 path
and L1 output. The observer delegates every operation unchanged. Calls using
the expert down tensor must retain its original compute-config object. Router,
attention, collective, activation and normalization methods are not replaced;
decode SDPA only has a pass-through observer requiring its native `None`
configuration.

Counters require one prefill method per layer and eleven sparse prefill gates
per layer, totaling330. Decode method and sparse-gate counters must each equal
`generated_tokens - 1` per layer. Thus a successful completed run cannot silently
omit either phase. This is one combined intervention, not a new precision sweep.

## Restoration and limits

Each wrapper verifies the actual phase, delegates the original method and
restores the native gate configuration and active observer context in `finally`.
It does not alter the model's phase transitions. Down configuration identity is
checked after each invocation. Arm cleanup restores both original methods on
every layer and both global operation observers, releases traces, frees the
request and clears serving-state references. Outer cleanup closes the parent
mesh and records clean closure after successful return.

Complete candidate token IDs, text, stop reason, execution counters and source
bindings are retained. Quality and release booleans remain false pending root's
review of the complete output. Reusing the prior native trajectory is explicit;
this is not a newly rerun baseline. An unsuccessful candidate must remain a
quality failure and must not lead to automatic policy retention or additional
broad generation.

## Verification performed

I reread the no-CPU continuation, inspected the source delta against the executed
full-case probe and checked the underlying model phase/sparse paths. I checked
the three frozen artifact hashes and seven direct probe/result/plan bindings,
and parsed the new probe with standard-library AST without importing it. The
AST contains one prefill and one decode call site, with a single candidate arm
and the fixed2048 iteration bound. The preparer separately reported all508
static pins passing; I did not rerun its validator. No TT import, probe launch,
hardware action or production edit was performed by this reviewer.

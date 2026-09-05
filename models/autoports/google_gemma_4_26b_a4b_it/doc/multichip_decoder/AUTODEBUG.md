# AutoDebug: multichip decoder constructor rejects its inherited routing default

## Headline finding

`MultichipDecoder.from_state_dict` is internally inconsistent with the current
`OptimizedDecoder.__init__` contract.  The multichip factory explicitly passes
`residual_shard_cores=0` (`tt/multichip_decoder.py:320-332`), while the inherited
constructor defaults `GEMMA4_OPT_ROUTING_ROW_MAJOR` to true and rejects row-major
routing whenever `residual_shard_cores == 0`
(`tt/optimized_decoder.py:951-953`).  With the failure command's environment,
the effective values are therefore:

```text
routing_row_major = True       # inherited env default
folded_expert_scale = True     # inherited constructor default
residual_shard_cores = 0       # explicit multichip argument
```

The guard must raise before `_configure_residual_chain`, weight post-processing,
or either attention-kind runtime path.  This accounts for both parametrized
failures (layer 0 sliding attention and layer 5 full attention) at the same
constructor location; the failure is not layer-specific and no device kernel is
involved.

The `residual_shard_cores=0` / row-major-default-true pairing is intentionally
incompatible in the optimized baseline, not an accidental boolean expression.
The baseline has an explicit guard with a purpose-specific error, its selected
default is tested by `test_selected_row_major_routing_is_trace_stable_and_default`,
and the row-major sparse metadata path is only consumed by the optimized
single-user sharded-residual MoE path (`optimized_decoder.py:3101-3143`).  The
multichip implementation deliberately retains a replicated/interleaved residual
and independently all-reduces TP contraction outputs, so it should not inherit
that single-chip candidate by default.

## Causal history and integration boundary

The multichip file was introduced in commit `fb1941075d17`.  At that revision,
`OptimizedDecoder.__init__` defaulted `residual_shard_cores` to 0 and had neither
the graph-fold switches nor `routing_row_major`.  Commit `c46a6ce40781` later
changed the optimized baseline: its constructor defaults became R22 plus four
enabled graph folds, and it added the row-major routing default and guard.  The
multichip factory retained its manual R0 construction.  Thus the earliest
divergence is the inherited constructor-policy default, not the guard that
correctly reports the incompatible effective policy.

## Important next failure if only the guard is bypassed

Setting `GEMMA4_OPT_ROUTING_ROW_MAJOR=0` is a decisive constructor experiment,
but it is not a complete repair.  The current multichip loader uploads the raw
state dict directly, while `OptimizedDecoder.__init__` now also defaults these
flags to true:

- `folded_router_projection`
- `shared_ffn_norm`
- `folded_expert_scale`
- `fused_final_scalar`

The canonical optimized factory first calls `_prepare_folded_state_dict`
(`optimized_decoder.py:1389-1401`) and later sets `layer_scalar_value`
(`optimized_decoder.py:1737-1741`).  The multichip factory does neither.  If the
constructor guard alone is disabled, inherited runtime code will skip the
router scale, FFN norm weights, and per-expert routing scale even though those
values were not folded into the multichip weights.  It will also enter
`_final_residual` with `fused_final_scalar=True` and
`layer_scalar_value=None`.  The likely result is wrong output or a later
argument/type failure, so an environment-only workaround must not be accepted
as proof of correctness.

This second discrepancy is part of the same constructor-policy drift and is
directly on the exact test's subsequent prefill/decode path.

## Focused experiments and predictions

No hardware run was performed, per the inspection-only constraint.  The
smallest follow-ups are:

1. Run only default construction with
   `GEMMA4_OPT_ROUTING_ROW_MAJOR=0`.  Prediction: the reported `ValueError`
   disappears for both layer cases, proving the immediate guard inputs.  Do not
   interpret later correctness as fixed yet.
2. In addition, force all four graph-fold switches off for the multichip
   construction.  Prediction: the factory retains raw weights and the inherited
   runtime applies the corresponding operations explicitly; this restores the
   policy under which the multichip loader was originally authored.  The exact
   PCC test should then reach real multichip execution and reveal any independent
   device-path issue.
3. After a code fix, run the exact reported command unchanged.  Prediction: both
   constructors complete without requiring callers to know optimized-baseline
   environment knobs, and the test proceeds to its prefill/decode PCC checks.
4. Add/run a narrow regression which asserts the multichip factory's effective
   policy after construction: R0 residual, row-major routing false, all four
   graph-fold flags false (unless the loader is deliberately taught all fold
   preparation), and non-`None` operands for every unfused runtime operation.

## Repair options, ranked

### 1. Preferred behavior-preserving repair: make the multichip policy explicit

Keep `residual_shard_cores=0`, explicitly select the unfused/raw-weight policy
for all four graph-fold flags, and ensure the subclass constructs with
`routing_row_major=False`.  This matches the existing multichip weight
preparation and replicated residual design and is the smallest semantic change.

There is currently no constructor argument or overridable hook for
`routing_row_major`; the optimized constructor reads the process environment
directly.  Because `optimized_decoder.py` is frozen for this stage, implementing
this wholly in `multichip_decoder.py` needs a carefully scoped subclass shim
around the inherited constructor (with restoration and an explicit policy for a
caller that sets `GEMMA4_OPT_ROUTING_ROW_MAJOR=1`).  A process-environment shim
is globally observable and therefore should be documented and covered; a test-
only `monkeypatch` is not a product fix.  If the intervention boundary were ever
relaxed, an explicit `routing_row_major` constructor parameter/hook in the base
would be cleaner, but it is out of scope here.

### 2. Prepare the multichip weights for inherited graph folds, but keep R0

Mirror the optimized factory's host-side router/FFN/expert transforms and scalar
setup in the multichip loader, while still disabling row-major routing because
R0 remains intentionally incompatible.  This can preserve the newer graph
fusion behavior, but touches more weight/cache/lifetime logic and is not needed
to fix the regression.  It requires new PCC evidence because folding before TP
sharding and quantization changes numerical behavior.

### 3. Move multichip to an R22 local residual and retain row-major routing

Changing the explicit `residual_shard_cores=0` to 22 makes the immediate guard
inputs compatible, but is not a focused constructor fix.  It selects different
decode orchestration, layouts, program configs, full-attention fidelity, and
sparse MoE code.  The multichip attention/all-reduce overrides were developed
for the current interleaved residual handoffs, and the loader still lacks the
required graph-fold preparation and row-major zero-base setup performed by the
canonical optimized factory.  This option needs a separate multichip R22 bringup
and should not be used as a one-line bypass.

### 4. Test/configuration workaround only

The test can set `GEMMA4_OPT_ROUTING_ROW_MAJOR=0` plus all four graph-fold env
flags to false.  This is useful as experiment 2 and may unblock diagnosis, but
leaves `MultichipDecoder.from_state_dict` broken under its public defaults and
leaks optimized-baseline implementation details to every caller.  It should not
be the final repair.

## Verification scope

Inspected the failing factory and test, the inherited constructor and router/MoE
paths, the canonical optimized folded-state preparation, relevant optimized
unit assertions, and the introducing commits.  No implementation or test was
edited and no device command was run.  The only investigation write is this
report.

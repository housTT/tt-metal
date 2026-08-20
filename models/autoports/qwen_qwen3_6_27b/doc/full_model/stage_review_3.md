# Stage review 3

Verdict: `more-work-needed`

## Required work

1. Device stochastic sampling reset only the parameter tensors. It did not
   initialize or advance the common sampler's request seed, prompt-penalty, or
   output-penalty state. The existing CPU compatibility checks therefore did
   not prove the optimized device path's explicit-seed or penalty contract.
2. Standard top-k/top-p still used `_perform_all_gather()`'s hard-coded Linear
   fallback. The Qwen physical-small-Ring opt-in covered only force argmax, even
   though safe Watcher had already rejected the standard Linear route.
3. `artifact_manifest.sha256` became stale after the selected Ring profiler
   wording in `evidence/final_validation.md` was corrected.

Required remediation is to wire the full common-sampler request lifecycle,
advance explicit seeds per generated token, include prompt/token-zero/output
history for penalties, extend the model-scoped physical Ring route to standard
top-k/top-p, pass a traced stochastic safe-Watcher test, and regenerate the
manifest. The stage checkpoint remains correctly deferred until a clean pass.

## Evidence and anomaly ledger

- `tt/generator.py::_configure_sampling` called only
  `reset_sampling_params`; no `apply_prefill_state`, penalty-state reset, or
  seed-manager advancement occurred.
- The public CPU tests covered token zero and host compatibility; the reduced
  device stochastic check used `seed=None`, default penalties, and only a token
  range assertion.
- `models/common/sampling/tt_sampling.py::_perform_all_gather` hard-coded
  `Topology.Linear`; `allow_small_ring` affected force argmax alone.
- Earlier safe-Watcher evidence stopped in the standard top-k Linear gather.
  Ring force argmax was the passing control.
- The corrected selected Ring profiler sums to 3,962.61 us: argmax 1,417.43 us,
  Ring all-gather 883.30 us. The stale manifest entry was the only artifact
  authentication failure observed.

## Residual assessment

The rest of the evidence was internally strong: 97% top-1 and 100% top-5 and
top-100 accuracy, 44.556 ms/token selected Ring token-out, reproducible
19.534645 GiB/device capacity arithmetic, coherent qualitative outputs, and
passing trace/reset/page-table/max-context gates. No vLLM work is present and
the unrelated submodule dirt remains outside stage scope.

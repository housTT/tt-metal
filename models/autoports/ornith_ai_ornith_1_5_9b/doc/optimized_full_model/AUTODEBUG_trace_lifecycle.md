# AutoDebug: shared-generator trace lifecycle

Investigation on 2026-09-05 began source-only. After the host reproduction and
fix, the parent authorized one serialized reduced hardware verification.

## Symptom and source finding

`logs/qualitative_norm_v1.log` reports trace allocation 108429312 bytes exceeding
the 100000000-byte trace reservation on the seventh request, the AIME prompt.
Six preceding requests generated output. This failure does not establish a
context-capacity or model-accuracy limit.

Both qualitative runners intentionally save the public cleanup method and set
`gen.teardown = lambda: None` while reusing one generator through readiness
runners. The saved method is called once at final cleanup. The shared readiness
runner calls `teardown` after each request, so this override postpones external
cleanup until all requests complete.

The recent generator refactor made `_capture` call `self.teardown()` to release
the prior model, plain-sampler, and history-sampler traces. The public override
therefore suppresses internal release too. `_capture` replaces the three trace
ID attributes with new IDs, losing the references needed to release the older
live traces. The original `_capture` directly released its old traces and was
not affected by the public override.

Additional affected internal call sites are `_configure_sampling` on a changed
sampling key and `configure_sampling` before warming a live sampler. Suppressing
those releases can also retain stale traces or warm a new mode behind old live
traces. All internal releases should use one private lifecycle method; the
public teardown hook should delegate to that method.

## Hypothesis and focused experiment

Hypothesis: with a no-op public `teardown`, repeated `_capture` or
`_ensure_replay_safe` accumulates live trace IDs instead of keeping exactly one
model/plain-sampler/history-sampler set. A sampling-key change likewise fails to
clear prior trace IDs. This predicts the observed trace-reservation growth.

Prepare `test_trace_lifecycle.py` using the actual generator AST and mocked
TTNN begin/end/release calls. Track the live trace-ID set, without importing
TTNN. Reproduce the public override used by the qualitative runner. Assert that
recapture keeps three live IDs, releases every superseded ID, sampling-mode
changes clear the old set before warming, and the saved final public teardown
releases the last set exactly once.

First run against unchanged source must fail on retained IDs. Only after that
focused reproduction, introduce `_release_traces` for the three internal call
sites and make public `teardown` delegate to it. Rerun the same test. Do not
increase trace reservation or remove the shared-generator reuse policy.

## Host reproduction, fix, and validation

Command against the unchanged generator:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  --confcutdir=models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/test_trace_lifecycle.py -q
```

All three tests failed for the predicted reasons: the first recapture retained
IDs 1..6 rather than three live IDs, a sampling-key change retained IDs 1..3, and
live sampler warmup observed unreleased traces. **Hypothesis verified.**

Smallest fix: move the existing release body to `_release_traces`; replace all
three internal `self.teardown()` calls with `self._release_traces()`; make public
`teardown` delegate to the private method. The qualitative runner's public
override still defers external cleanup, while internal recapture and sampling
reconfiguration always release their own traces. No trace size or hardware
policy was changed.

The same three tests then passed, including seven consecutive recaptures,
idempotent final cleanup, unchanged-key reuse, and preserved live sampler state
across warmup. With parent authorization, the existing host contract test's
single internal-release mock was updated from `gen.teardown` to
`gen._release_traces`. The combined existing/new suite passed all **19 tests**:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  --confcutdir=models/autoports/ornith_ai_ornith_1_5_9b/doc \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/test_generator_host_contract.py \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/test_trace_lifecycle.py -q
```

All applicable pre-commit checks passed. The change is Python only; no build or
dependency installation was needed.

## Reduced hardware verification

`probe_trace_lifecycle.py` reuses a real native TP4 generator with layers 0 and 3,
BF16 HiFi4 head, the candidate sharded final norm, 512-token cache context, and
the same no-op public teardown override. It performs eight explicit recaptures
and repeated two-token requests. It asserts exact repeated tokens, constant
TRACE allocation after every iteration, and zero TRACE allocation after calling
the saved public teardown.

`trace_lifecycle_v1` failed during Torch/Transformers import before device open
because the command omitted the runner's persistent `TORCHINDUCTOR_CACHE_DIR`.
The known environment prerequisite was restored for `trace_lifecycle_v2`; no
reset was necessary for the import-only failure. The v2 command uses watcher 10,
ETH watcher disabled, allocation tracking enabled, profiler disabled, and a
150-second process bound. `logs/trace_lifecycle_v2.provenance.json` records the
exact command/environment/source snapshot and `trace_lifecycle_v2.json` records
the measurements.

V2 passed with exit 0. All eight recaptures returned exact repeated tokens;
TRACE allocation stayed at **1,638,400 bytes per device** for every iteration
and returned to **0 bytes** after saved final teardown. The watcher stopped and
all devices closed normally. Hardware ownership was returned to the parent;
the investigator will not start another hardware workload.

Status: **trace-lifecycle cause verified, fixed, and passing reduced hardware
verification; full qualitative rerun remains a parent stage gate**.


## Final stage closure

The earlier investigation status above is preserved as historical evidence.
The selected implementation and completed final gates are recorded in the
[stage report](README.md), [runtime audit](runtime_audit.md),
[final full32 watcher control](prefill_integration_full32_v2/summary.json),
[long exact replay controls](prefill_integration_long_v2.json), and
[final profiling report](tracy/README.md). Earlier pending experiments are not
claims that these final gates remain unrun; rejected hypotheses and failed
receipts remain preserved. Independent stage review owns the final verdict.

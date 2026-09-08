# Worker cleanup experiment

Starting evidence: the repeated-shutdown appendix in [AUTOTRIAGE.md](AUTOTRIAGE.md).
This is one isolated AutoFix hypothesis experiment; no hardware operations or
TTNN imports were performed by this investigator.

## Verdict

**Verified host lifecycle bug and real serving shutdown; the original Ethernet
timeout is not attributed to the worker change alone.**
`EngineCore.shutdown()` → `UniProcExecutor.shutdown()` →
`WorkerWrapperBase.shutdown()` reached the inherited, empty
`WorkerBase.shutdown()`. `TTWorker` previously closed its mesh only in `__del__`.
One shared `suppress(AttributeError)` also skipped mesh close when
`model_runner` was absent.

A pre-edit AST/stdlib probe executed the actual destructor with a dummy base
class and mocked close helper. It verified both predictions: explicit shutdown
performed zero closes, and destructor cleanup with a missing runner performed
zero closes despite an owned mesh. No package imports were needed.

## Change and ownership checks

This cleanup experiment changes these two files in `/home/hous/dev/vllm`:

- `plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py`: add explicit
  `shutdown()`, remove the runner independently of mesh presence, call the
  existing close helper, and clear mesh ownership after successful close.
  INFO markers are `TTWorker mesh shutdown starting` and
  `TTWorker mesh shutdown complete`. Explicit errors propagate. `__del__`
  delegates to shutdown as a best-effort fallback.
- `plugins/vllm-tt-plugin/tests/test_worker_parent_mesh.py`: test explicit
  shutdown with/without a runner, repeated shutdown plus destructor, missing
  mesh/non-device ranks, destructor fallback, and visible explicit close
  failure. The real helper closes nested submesh → model mesh → physical
  parent, then resets the physical fabric, exactly once after successful close.

The existing `close_mesh_device()` helper is unchanged. The experiment adds no
firmware, timeout, adapter, trace, or model changes.

## Verification

Executed:

```bash
python3 /tmp/tt-worker-cleanup-host-probe.py
git -C /home/hous/dev/vllm diff --check
git -C /home/hous/dev/vllm diff --stat
```

The local probe extracts `TTWorker.shutdown`, `TTWorker.__del__`, and the actual
open/close helpers with AST, then executes all nine cases from the test file
using stdlib mocks and a small monkeypatch/raises shim. **Nine passed**, including
all three pre-existing parent/direct-mesh cases. It asserts that TTNN is absent
from `sys.modules`. This is a host ownership proof, not a normal pytest run.
`git diff --check` passed. The first probe attempt exposed a bug in the temporary
shim (`__getattr__` raised `KeyError`); fixing it to raise `AttributeError` allowed
mock teardown and all cases to complete. No implementation change was required.

Review the implementation/test diff with:

```bash
git -C /home/hous/dev/vllm diff -- plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py plugins/vllm-tt-plugin/tests/test_worker_parent_mesh.py
```

The coordinating agent subsequently ran the normal host pytest suite against
`/home/hous/dev/vllm/plugins/vllm-tt-plugin/tests/test_worker_parent_mesh.py`.
[worker_cleanup_pytest.log](worker_cleanup_pytest.log) records **9 passed, 16
warnings in 4.69 s**. The warnings are dependency deprecations. The investigator
read that log and reviewed the final two-file sibling diff; no additional
implementation edits were needed. The retained log does not include the full
launch command, so no exact environment/argv claim is inferred from it.

The sibling's final pre-commit run also passes all applicable checks, including
Ruff lint/format, mypy, and SPDX validation; see
[worker_lint_final.log](worker_lint_final.log). The coordinator reports only
formatter changes with unchanged tested semantics. No implementation edit was
made during this report update.

Python-only changes require no C++ build. No performance claim is made.

## Real serving verification

The candidate P150 full serving run completed all API, sampling and qualitative
gates. Its [server log](../../readiness_vllm/P150/optimized_vllm/after/server.log)
contains both worker shutdown markers and completed UMD/device closure. No
EngineCore, API server or readiness-runner process remained. The immediate
separate 2x2 parent / 1x2 submesh reopen completed without a reset; see
[cleanup_evidence.json](cleanup_evidence.json) and
[after_P150_reopen.log](after_P150_reopen.log).

The warmed baseline subsequently completed P150, P150x2 and P150x4 sequentially,
with server-runner exit 0 and no intervening resets. The subsequent candidate
P150 server also opened, completed both benchmarks, and closed normally. These
runs verify actual serving lifecycle with the combined worker and router fixes.
They do not isolate the worker as the sole cause of the original heartbeat
timeout. The separately reproduced router failure and repair are documented in
[kernel_cleanup_experiment.md](kernel_cleanup_experiment.md).

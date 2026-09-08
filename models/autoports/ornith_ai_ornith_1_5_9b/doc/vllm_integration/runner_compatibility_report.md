# Runner compatibility and process cleanup verification

Date: 2026-09-08 UTC. Host-only AutoFix verification of the parent's runner and
plugin changes. The only implementation-tree edit in this pass is four added
tests in `models/common/readiness_check/test_run_vllm_server.py`; the runner,
plugin, adapter, and generator implementations were not edited.

## Results

**All eight runner tests pass**, comprising four restored tests and four new
regressions. The actual pinned EngineArgs selects `--additional-config`, P150x4
is accepted by the CLI, launches request a private session, and shutdown kills
and reaps a TERM-resistant orphan after its launcher has already exited.
The registry resolves both Ornith aliases to the actual autoport adapter using
the checkpoint's model configuration. No real server or TT device was opened.

```bash
USER=hous ../state/serving-env/bin/python -m pytest -q \
    models/common/readiness_check/test_run_vllm_server.py
USER=hous python_env/bin/pre-commit run --files \
    models/common/readiness_check/test_run_vllm_server.py
```

Results: **8 passed in 5.03s**, and every applicable pre-commit hook passed.
Logs: `runner_compatibility_tests.log` and `runner_compatibility_format.log`.
This is a Python test change, so no TT build was needed.

## Hypothesis experiments

### TT configuration and mesh compatibility

Hypothesis: the runner uses a configuration option supported by installed vLLM
and accepts the four-chip Blackhole mesh profile.

The tests import actual `vllm.engine.arg_utils.EngineArgs`, inspect its dataclass
fields, and check `_tt_config_flag()`. A separate direct probe asserts the exact
result `--additional-config` under vLLM source
`bf98d556bb46a5cda25fac540629251e7f474200`.
The mocked launch checks that the resulting command contains a JSON object
`{"tt": <configuration>}`, passes `MESH_DEVICE=P150x4`, and requests
`start_new_session=True`. The CLI test invokes `_main()` with `--stages serve`
and `--mesh-device P150x4`, mocking network checks and all launch/lifecycle
operations. It verifies that P150x4 reaches the launch boundary.

Verdict: **verified** for argument construction and parsing. These are host
contract checks, not four-chip serving or device-shape validation.

### Orphan process-group cleanup

Hypothesis: `_shutdown` cleans a launcher-owned private process group even when
the launcher exits before cleanup and a child ignores SIGTERM.

The regression runs an isolated helper interpreter with Linux child-subreaper
status, creates a dummy launcher with `start_new_session=True`, and has that
launcher create a child in the same private session. The child installs
SIGTERM-ignore handling before publishing a readiness record. The launcher
then exits normally, and the helper waits for that exit before invoking the
actual `_shutdown` function.

Before signaling, the test verifies the child's session and process-group IDs
equal the newly created launcher's PID and differ from the helper's group. It
then confirms the child exits specifically from SIGKILL and reaps it with
`waitpid`. The helper's `finally` block signals only that newly created private
group and reaps its dummy child if an assertion fails. No existing process is
selected by name, scanned for signaling, or signaled.

Verdict: **verified**. The positive regression passed, and the child was reaped;
the test leaves neither a running dummy child nor its zombie behind.

The negative control ran the same test through `runpy.run_path`, intercepting
its helper `subprocess.run` invocation and replacing only the helper's
`_shutdown(launcher, ...)` call with `pass`. The helper's final cleanup remained
active. That controlled bypass failed with
`AssertionError: Orphan child survived runner shutdown`, establishing that the
regression detects the original failure mechanism. The helper subsequently
killed and reaped its dummy process in `finally`. Evidence:
`runner_shutdown_negative_control.log`.

The initial test run had a nested-string `IndentationError` in the newly written
helper, before it could create dummy processes. The harness was corrected by
passing separately dedented child/launcher scripts as arguments. This was a
test-authoring failure, not evidence of a runner defect. The original log is
preserved as `runner_compatibility_tests_initial_harness_error.log`.

### Actual model registry resolution

The direct host probe loads the pinned local checkpoint configuration with the
Ornith architecture override, then calls both `inspect_model_cls` and
`resolve_model_cls`. It does not construct the adapter or load model weights:

```python
from pathlib import Path
from vllm.config import ModelConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm_tt_plugin.platform import register_tt_models
from models.common.readiness_check.run_vllm_server import _tt_config_flag
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm import TTOrnithForCausalLM

assert _tt_config_flag() == "--additional-config"
register_tt_models()
snapshot = str(Path("../upstream").resolve())
config = ModelConfig(
    model=snapshot, tokenizer=snapshot,
    hf_overrides={"architectures": ["OrnithForCausalLM"]},
    max_model_len=2048, dtype="bfloat16", skip_tokenizer_init=True,
)
for alias in ["OrnithForCausalLM", "TTOrnithForCausalLM"]:
    info, inspected = ModelRegistry.inspect_model_cls([alias], config)
    cls, resolved = ModelRegistry.resolve_model_cls([alias], config)
    assert cls is TTOrnithForCausalLM
    assert inspected == resolved == alias
    assert info.is_text_generation_model
```

Verdict: **verified**. Both aliases resolve to
`models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm.TTOrnithForCausalLM`,
recognized as a text-generation model and exposing `initialize_vllm_model`.
Evidence: `runner_registry_probe.json` and `runner_registry_probe.log`.

## Remaining scope

No runner implementation failure was observed by these focused experiments.
The parent's reduced-server L1 failure, real serving initialization, sampling,
parser API behavior, all mesh profiles, and performance/accuracy gates remain
independent work. These host checks do not waive or establish those gates.

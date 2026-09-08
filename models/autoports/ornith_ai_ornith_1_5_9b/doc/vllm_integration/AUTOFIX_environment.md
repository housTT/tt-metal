# AutoFix Report: serving environment

**2026-09-08 follow-through:** the updated root/model AGENTS authorization
resolved the historical no-install interpretation below. Provisioning is now
complete in `../state/serving-env`: the TT platform imports, both original help
probes pass, all four restored runner host tests pass, and full `pip check`
passes. The original 373-package TTNN environment is unchanged. See
`provision_work_log.md`, `provision_clean_host_probes.json`, and
`serving_source_pins.json` for the isolated runtime, commands, logs and pins.
This repairs the environment prerequisite; it does not mark the serving stage
complete. The original report is preserved below as starting evidence.

Date: 2026-09-06 UTC. Stage 9, `ornith-ai/Ornith-1.5-9B`. Focused host-only verification following `AUTODEBUG_environment.md`. No installation, model implementation edit, hardware access, or server launch occurred.

## Starting Evidence

- Diagnosis: `AUTODEBUG_environment.md` in this directory.
- Original failing entrypoint: `python_env/bin/python -m vllm.entrypoints.openai.api_server --help`, exit 1, `ModuleNotFoundError: No module named 'vllm'`.
- The owner restored `models/common/readiness_check/run_vllm_server.py` and `test_run_vllm_server.py` from workflow commit `70a596f92229ada922fba743cd0cd9d2658a5c1c`.
- Existing `runner_host_tests.log` shows missing passwd lookup for UID 1002 during PyTorch import. `runner_host_tests_user.log` shows that `USER=hous` proceeds past that error and then fails collecting the restored runner tests because `openai` is absent.

## Hypothesis Experiments

### Hypothesis: restoring the pinned runner plus USER=hous resolves serving prerequisites

Prediction: the restored runner and vLLM entrypoint should reach `--help`, and `openai`, `vllm`, and `vllm_tt_plugin` should be discoverable in at least one already provisioned interpreter.

Experiment: run the following three commands for each interpreter listed below, with a Python `subprocess.run(..., timeout=30)` bound on every command, `USER=hous`, and repository-root working directory:

```bash
USER=hous <python> -c 'import importlib.util,json; print(json.dumps({name: importlib.util.find_spec(name) is not None for name in ["openai","vllm","vllm_tt_plugin"]}))'
USER=hous <python> -m models.common.readiness_check.run_vllm_server --help
USER=hous <python> -m vllm.entrypoints.openai.api_server --help
```

| Interpreter | Package probes (exit 0) | Runner --help | vLLM --help |
| --- | --- | --- | --- |
| `/home/hous/dev/ornith-1.5-9b/tt-metal/python_env/bin/python` | all three false | exit 1: missing `openai` | exit 1: missing `vllm` |
| `/home/hous/dev/ornith-1.5-9b/state/device-env/bin/python` | all three false | exit 1: missing `torch` | exit 1: missing `vllm` |
| `/opt/venv/bin/python` | all three false | exit 1: missing `torch` | exit 1: missing `vllm` |

No command timed out. The active environment's runner now reaches the restored source and its `import openai` line. The other environments fail earlier because they also lack PyTorch. No adapter, EngineCore, server, or device code was reached.

Verdict: **refuted**. Runner restoration and `USER=hous` repair two prerequisite layers but do not provide missing Python packages. Switching to either discovered alternate interpreter also fails.

Exact commands, environment override, exit statuses, and complete subprocess output: `autofix_environment_probes.json`.

### Verify the source-restoration fix independently

Experiment:

```bash
python_env/bin/python - <<'PY'
import ast, hashlib, subprocess
from pathlib import Path
ref = '70a596f92229ada922fba743cd0cd9d2658a5c1c'
for name in ['run_vllm_server.py', 'test_run_vllm_server.py']:
    path = Path('models/common/readiness_check') / name
    original = subprocess.check_output(['git', 'show', f'{ref}:{path}'])
    actual = path.read_bytes()
    ast.parse(actual, filename=str(path))
    print(path, hashlib.sha256(actual).hexdigest(),
          hashlib.sha256(original).hexdigest(), actual == original, 'AST=PASS')
PY
```

Result: both files exactly match the pinned source bytes and parse successfully.

| File | Working-tree and pinned-source SHA256 |
| --- | --- |
| `run_vllm_server.py` | `051d45493d4a822da38395f506188f08d46cba981ef1e1e9d7a0a74efd7c291c` |
| `test_run_vllm_server.py` | `3f5758b449b5d74990da2193e064c067698b7186d5c53705d270f45adae6e590` |

Verdict: **verified** as an exact source restoration. AST success is syntax evidence only; runner tests still fail at collection, so this is not runtime validation or a serving pass. No speculative implementation patch was applied by this experiment.

## Process cleanup audit

After every help probe exited, a host-only `/proc` audit checked process names for `EngineCore`/`VLLM::EngineCor` and Python module arguments exactly matching `models.common.readiness_check.run_vllm_server`, `vllm.entrypoints.openai.api_server`, or `vllm.entrypoints.cli.main`. The audit inspected process UID and whether the working directory belonged to this task, but did not print full command lines, environment variables, or credentials.

At `2026-09-06T00:01:09.673351+00:00`, no matching serving or EngineCore processes existed; zero processes were unreadable during the audit. No process kill or device reset was needed. Evidence: `autofix_environment_process_audit.json`. This confirms cleanup for the probes; it is not a hardware health check.

## Final Status

**Unresolved external dependency under the current no-install instruction.** The user's root AGENTS instructions state: “Do not try to install compilers or dependencies; do not run `install_dependencies.sh`.” No usable existing environment was found in the diagnosis, and the focused experiment confirms all discovered runnable interpreters lack vLLM and the TT plugin. The active interpreter also lacks the runner's `openai` dependency.

AutoFix cannot close serving readiness through source restoration or environment selection alone. Required next input is an existing compatible serving environment or explicit authorization to provision dependencies. Do not mark the vLLM integration stage complete: model registration, reduced/full serving checks, sampling, qualitative evidence, benchmarks, and independent stage review remain unverified. Other authorized source/interface work can continue independently.

After provisioning, repeat the exact package/help probes, run the restored host tests, record runtime pins, and only then proceed to the serialized reduced-layer serving loop. No measurements or quality claims from the historical 35B reference establish readiness for this model.

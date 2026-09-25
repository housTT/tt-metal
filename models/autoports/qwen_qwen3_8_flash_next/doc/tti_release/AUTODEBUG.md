# AUTODEBUG: TTI external runtime spec argparse failure

## Scope

Source-only investigation of the TTI checkout:

- TTI path: `/home/ttuser/dev/qwen3.8-flash-next/tti-release/qwen3_8_flash_next/tt-inference-server`
- TTI revision: `v0.20.0`, HEAD `6ab1de736f303b899f84ae07d184d33f9889946e`
- Runtime spec: `models/autoports/qwen_qwen3_8_flash_next/doc/tti_release/specs/smoke_runtime_model_spec.json`
- Target external model: `Qwen/Qwen3.8-Flash-Next`
- Target external code path: `models/autoports/qwen_qwen3_8_flash_next`

No implementation code was edited. No server or TT hardware path was used.

## Headline findings

### 1. Immediate root cause: `run.py` rejects the external model name before the runtime JSON path is reachable

The observed command fails in argparse, before `resolve_runtime()` can load the supplied spec.

Evidence:

- `run.py:92-114` builds `valid_models` from import-time `MODEL_SPECS` and registers `--model` with `choices=valid_models`.
- `run.py:642` calls `parser.parse_args()`.
- `run.py:927-943` only reaches the external-spec branch after parsing, where it would call `ModelSpec.from_json(args.runtime_model_spec_json)`.
- Running the exact failing command exits 2 with `run.py: error: argument --model: invalid choice: 'Qwen3.8-Flash-Next'`.
- Direct `ModelSpec.from_json(smoke_runtime_model_spec.json)` succeeds and returns:
  - `model_name = Qwen3.8-Flash-Next`
  - `device_type = P300`
  - `device_model_spec.max_context = 262144`
  - `impl.code_path = models/autoports/qwen_qwen3_8_flash_next`

This fully explains the reported exit 2: the custom spec is valid enough to deserialize, but the parser never lets control reach that loader.

### 2. Omitting `--model` is not a valid workaround in this checkout

Although `--model` is registered with `required=False`, `parse_arguments()` has a post-parse required-model check before `resolve_runtime()`.

Evidence:

- `run.py:669-674` errors when `args.model is None` for every workflow except `prefill_decode`.
- Running the same command without `--model` exits 2 with:
  `the following arguments are required: --model (optional only for --workflow prefill_decode).`
- Bypassing only that parser check proves `resolve_runtime()` can load the spec with `args.model = None`, but `RuntimeConfig.model` remains `None`. A clean omit-`--model` design should therefore fill the runtime identity from the loaded spec/config, not merely skip the error.

So a parser/code change is warranted. The correct invocation is not simply “omit `--model`” with the current TTI revision.

### 3. `run_workflows.py` repeats the same catalog-only `--model` parser boundary

Fixing only `run.py` is not enough for the reported `benchmarks` path.

Evidence:

- For LLM benchmarks, `workflows/workflow_dispatch.py:605-613` builds a `launchers/run_llm_bench.py` command.
- `launchers/_launcher_common.py:26-55` re-execs `run_workflows.py` and forwards CLI arguments verbatim.
- `run_workflows.py:69-87` also builds `valid_models` from import-time `MODEL_SPECS` and registers `--model` with `choices=valid_models`.
- A direct parser probe of `run_workflows.py --model Qwen3.8-Flash-Next --runtime-model-spec-json ...` also exits 2 with an invalid-choice error.

The smallest complete parser fix for this benchmark route must cover both `run.py` and `run_workflows.py`.

### 4. Runtime spec source-of-truth is incomplete: `resolve_runtime()` ignores embedded `runtime_config` / legacy `cli_args`

The supplied smoke spec is a flat `ModelSpec` JSON with legacy `cli_args`. `RuntimeConfig.from_json()` already supports both current combined JSON (`runtime_config`) and legacy flat specs (`cli_args`), but top-level `run.py` does not use it in the external-spec branch.

Evidence:

- `workflows/runtime_config.py:253-274` implements `RuntimeConfig.from_json()`:
  - `runtime_config` block wins for current combined JSON.
  - top-level `cli_args` is supported for legacy flat specs.
- The smoke spec contains top-level `cli_args` with:
  - `model = Qwen3.8-Flash-Next`
  - `workflow = benchmarks`
  - `device/tt_device = p300`
  - `impl = autoport-qwen-qwen3-8-flash-next`
  - `engine = vllm`
  - `service_port = 8021`
  - `server_url = http://127.0.0.1`
  - `no_auth = true`
  - `skip_system_sw_validation = true`
  - `disable_trace_capture = true`
  - `limit_samples_mode = smoke-test`
- `run.py:937-943` loads only `ModelSpec.from_json(...)` and then builds `runtime_config = RuntimeConfig.from_args(args)`.
- `run.py:911-924` then calls `populate_model_spec_cli_args()`, which clears `model_spec.cli_args` and replaces it with the CLI-derived `RuntimeConfig`.
- A synthetic parse with a catalog-valid placeholder CLI model and conflicting CLI port showed:
  - `RuntimeConfig.from_json(spec)` returns `Qwen3.8-Flash-Next`, port `8021`, impl `autoport-qwen-qwen3-8-flash-next`.
  - `resolve_runtime(args)` returns the CLI placeholder model, CLI port, CLI server URL, and `impl=None`.
  - `model_spec.cli_args` is overwritten with those CLI-derived values.
- `run.py:1021-1024` writes a new combined runtime JSON from this CLI-derived `RuntimeConfig`, so the mismatch is propagated to subprocesses.

This does not cause the first observed argparse exit, but it does violate the likely #4345/external-runtime-spec source-of-truth contract if the spec is supposed to control workflow/server mode/port. It should be fixed at the same boundary if external specs are intended to be authoritative for runtime configuration, not just for model structure.

### 5. Downstream release/eval registry issue remains after parser fixes

This is not causal for the reported `benchmarks` exit 2, but it matters for release readiness.

Evidence:

- `workflows/validate_setup.py:100-102` already skips catalog membership when `runtime_config.runtime_model_spec_json` is set.
- `workflows/validate_setup.py:119-122` still requires `model_spec.model_name in EVAL_CONFIGS` for `evals`.
- `workflows/validate_setup.py:203-209` still requires `model_spec.model_name in EVAL_CONFIGS` for `release`.
- `reference_config/evals/eval_config.py:5273-5277` builds `EVAL_CONFIGS` by intersecting `_eval_config_map` with import-time `MODEL_SPECS`.
- There is no `Qwen/Qwen3.8-Flash-Next` / `Qwen3.8-Flash-Next` entry in the current eval registry search results.

Adding an `EvalConfig(hf_model_repo="Qwen/Qwen3.8-Flash-Next", ...)` alone is not sufficient for a runtime-only model while `EVAL_CONFIGS` is derived only from `MODEL_SPECS`. Release/eval support needs either a catalog entry or a runtime-spec-aware eval lookup helper keyed by the resolved `model_spec.hf_model_repo` / `model_spec.model_name`.

## Smallest correct fix recommendation

### Parser boundary

In both `run.py` and `run_workflows.py`:

1. Remove `choices=valid_models` from `--model`.
2. After parsing, enforce catalog membership only when `--runtime-model-spec-json` is absent.
3. Preserve the current typo protection for normal catalog runs by emitting `parser.error(...)` for unknown `--model` without a runtime spec.
4. Make the post-parse required-model check JSON-aware:
   - Without runtime JSON, keep requiring `--model` except for existing `prefill_decode` behavior.
   - With runtime JSON, either allow `--model` to be omitted and fill it from the JSON, or require it to match the JSON model name.

If the external spec is intended to supply identity, the cleanest behavior is:

- `--runtime-model-spec-json` can be the source of `model`, `workflow`, `device`, server mode, `service_port`, and `server_url`.
- Any CLI identity/runtime fields supplied alongside it are either ignored with a clear warning or rejected on mismatch. Do not silently let them override the spec.

### RuntimeConfig boundary

In `run.py.resolve_runtime(args)`, the external-spec branch should load both halves from the JSON:

```python
if args.runtime_model_spec_json:
    model_spec = ModelSpec.from_json(args.runtime_model_spec_json)
    runtime_config = RuntimeConfig.from_json(args.runtime_model_spec_json)
    runtime_config.runtime_model_spec_json = args.runtime_model_spec_json
else:
    ...
```

If support for a bare ModelSpec without `runtime_config` or `cli_args` is required, fall back deliberately and visibly to `RuntimeConfig.from_args(args)` only for missing runtime config. For the current smoke spec, `RuntimeConfig.from_json()` should succeed via top-level `cli_args`.

Then `populate_model_spec_cli_args(model_spec, runtime_config)` will normalize the spec from the JSON-derived runtime config instead of overwriting it with CLI-derived placeholders.

### Optional `--impl` parity

The reported command does not pass `--impl`, so `--impl choices=valid_impls` is not the immediate failure. But the smoke spec’s legacy `cli_args.impl` is `autoport-qwen-qwen3-8-flash-next`, which is not in the catalog impl choices. If external-spec CLI invocations are expected to pass custom `--impl`, apply the same conditional-validation pattern to `--impl`: validate against catalog impls only when runtime JSON is absent.

## Focused tests to add

### `run.py` parser tests

- Unknown model without runtime JSON still exits 2:
  `run.py --model NotInCatalog --workflow benchmarks --tt-device p300`
- External model with runtime JSON parses:
  `run.py --model Qwen3.8-Flash-Next --runtime-model-spec-json <tmp/spec.json> --workflow benchmarks --tt-device p300`
- Omitted model with runtime JSON parses or resolves according to the chosen contract:
  `run.py --runtime-model-spec-json <tmp/spec.json>` plus any still-required minimal flags.
- Both spellings are covered:
  - `--runtime-model-spec-json <path>`
  - `--runtime-model-spec-json=<path>`

### `resolve_runtime()` source-of-truth tests

- Build a temp flat ModelSpec JSON with legacy `cli_args` containing `model`, `workflow`, `device`, `service_port`, `server_url`, server mode, `impl`, and `engine`.
- Call `resolve_runtime()` with deliberately conflicting CLI placeholder values.
- Assert the returned `RuntimeConfig` uses JSON values, not CLI placeholder values.
- Assert `model_spec.cli_args` is repopulated from the JSON-derived runtime config.
- Assert `get_runtime_model_spec()` is not called in the external-spec branch.
- Assert catalog runs without runtime JSON still apply normal CLI overrides through `RuntimeConfig.from_args()` and `model_spec.apply_overrides()`.

### `run_workflows.py` parser / command-factory tests

- Unknown model without runtime JSON still exits 2.
- `run_workflows.py --model Qwen3.8-Flash-Next --runtime-model-spec-json <tmp/spec.json> ...` parses successfully.
- `CommandFactory._build_context()` loads `ModelSpec.from_json()` and `RuntimeConfig.from_json()` for a runtime-only model instead of resolving from `MODEL_SPECS`.
- A generated LLM benchmark command for the external spec is parseable by `run_workflows.py`.

### Benchmark path tests

- Loading the smoke spec and constructing a `RuntimeConfig(workflow="benchmarks", device="p300", runtime_model_spec_json=<path>, limit_samples_mode="smoke-test")` dispatches through the LLM benchmark path.
- `llm_module.benchmark_configs.get_llm_configs(...)` returns the expected smoke point from the spec’s perf reference:
  `isl=8`, `osl=8`, `max_concurrency=1`, `num_prompts=1`.

### Release/eval registry tests

- `benchmarks` does not require an `EVAL_CONFIGS` entry.
- `evals` / `release` fail clearly for `Qwen3.8-Flash-Next` until an eval policy is added.
- If runtime-only evals are supported, add a helper test proving lookup by resolved `model_spec.hf_model_repo` works even when the model is absent from `MODEL_SPECS`.

## Review against observations

The parser bug accounts for:

- exit code 2,
- argparse “invalid choice” text,
- failure before server or hardware setup,
- direct `ModelSpec.from_json()` success,
- presence of the #4345 later-stage custom-spec fixes without this command working.

The omit-`--model` path is not a workaround because the post-parse required-model check runs before runtime resolution.

The runtime-config source-of-truth bug is separate from the first exit 2, but it is real and will matter once parser validation lets external specs through.

The eval registry issue is downstream release/eval readiness, not the current benchmark argparse failure.

# AutoFix Report: vLLM Parameter Request Timeout

## Starting Evidence

- Source state: TTI `f07a31d2a2f908aa04098685034e7a5bde7554ea`.
  The existing root `AUTODEBUG.md` / `AUTOFIX.md` cover the earlier suite-selection
  bug, so this focused experiment starts from the saved Stage 11 report and the
  supplied timeout hypothesis instead.
- Original run: `evidence/tti_release.log:4948-4954` launched
  `VLLMParamConformanceTest`; the saved report records `test_n[2]`, `test_n[3]`,
  and `test_non_uniform_seeding` as the only three `ReadTimeout` failures, each
  with `read timeout=30`.
- At `f07a31d2`, `test_fixtures/conftest.py:135-148` defaulted every
  `api_client` call to scalar `timeout=30` and forwarded it directly to
  `requests`. Both target test functions call the fixture without a timeout
  override. The outer spec-test budget is 3600 seconds, so the 30-second request
  deadline was a separate inner harness boundary.

## Hypothesis Experiment

- **Hypothesis:** the target cases reach the client read deadline before this
  backend finishes multi-choice / 32-way generation.
- **Prediction:** the no-override path forwards 30 seconds; extending only the
  read deadline beyond the archived completion window removes this artificial
  boundary while keeping fast connect failure.
- **Pre-fix experiment:** invoke `api_client.__wrapped__` with a fake request
  method (no socket/server/hardware access) and capture the keyword arguments.
  It printed `pre-fix default timeout forwarded to requests: 30 seconds`.
- **Archived timing result:** `test_n[2]` and `[3]` fail in consecutive
  approximately 30-second windows rather than on an API status/assertion. The
  server log also shows B32 work continuing more than 30 seconds after
  `test_non_uniform_seeding` starts, followed by HTTP 200 completion records.
  The nearest archived B32 benchmark completed 256/256 requests with zero
  failures, mean E2E latency 190.713 seconds and p99 206.808 seconds. All 68
  requests with at most 50 output tokens took 120.785-202.025 seconds, so a
  120-second read deadline would still lack empirical coverage. Evidence:
  `tti_cache/workflow_logs/reports_output/release/gpt-oss-120b_p150x4_release/llm/benchmark_openai__gpt-oss-120b_2026-09-02_08-48-05_isl-128_osl-128_maxcon-32_n-256.json`.
- **Verdict:** verified at the harness boundary. The saved artifacts prove the
  client gave up at 30 seconds while backend work continued past it; they do not
  preserve the late response bodies, so final content assertions remain an
  end-to-end verification item.

## Fix

- `test_fixtures/conftest.py`: replace the scalar 30-second default with
  `DEFAULT_API_TIMEOUT = (30, 300)`, using requests' `(connect, read)` form.
  This retains the existing connection deadline and gives the read side a
  bounded deadline above the archived 206.808-second B32 p99.
- `tests/test_api_client_fixture.py`: add mocked, host-only regression checks
  that the new default is forwarded and that explicit overrides such as
  `timeout=None` remain unchanged.

## Verification

- Pre-fix mocked probe: passed, observed scalar `30`.
- Synthetic no-network A/B: a valid response modeled at 150 seconds (inside the
  archived short-output B32 range) made all three target cases fail with the
  120-second read deadline and pass with `(30, 300)`.
- `.workflow_venvs/.venv_workflow_run_script/bin/python -m pytest -q tests/test_api_client_fixture.py tests/test_module/llm_tests/test_vllm_param_conformance_tests.py tests/test_module/test_dispatch_spec_tests.py tests/test_module/test_matrix_expansion.py`
  -> `34 passed, 1 warning` (the pre-existing `TestConfig` collection warning).
- `.workflow_venvs/.venv_workflow_run_script/bin/python -m pytest --collect-only -q llm_module/test_vllm_chat_completions.py`
  -> `22 tests collected`.
- `git diff --check -- test_fixtures/conftest.py`, plus
  `git diff --no-index --check /dev/null` for each new test/report artifact ->
  passed.
- `pre-commit run --files test_fixtures/conftest.py tests/test_api_client_fixture.py`
  could not run its ruff/pytest hooks because the checkout lacks
  `.pre-commit/bin/activate`; its copyright hook passed. Direct pytest coverage
  above passed.

## Final Status

The timeout defect is fixed with host-only evidence. No request was sent to the
live server and no hardware/server action was taken. The original API suite was
intentionally not rerun under this workflow's isolation constraint, so an
authorized serialized rerun must still confirm the three late response bodies
pass their semantic assertions and that 300 seconds has sufficient hardware
variance headroom.

# AutoFix Report

## Starting Evidence

- Fresh source-only diagnosis: `AUTODEBUG.md`.
- Failed run: `workflow_logs/run_logs/run_2026-08-31_10-04-55_id_autoport_Qwen3.8-Flash-Next_p300_release_release_A8_tO1JV.log`.
- The release command passed `timeout=7200,num_concurrent=2`, but the ten
  requests were created together and the remaining calls raised
  `asyncio.TimeoutError` after three approximately 300-second attempts.

## Hypothesis Experiments

- Hypothesis: lm-eval 0.4.4 accepts `timeout=7200` through `**kwargs` but drops
  it, leaving aiohttp's 300-second cumulative default active.
  Experiment: instantiate the installed `LocalChatCompletion` with
  `timeout=7200` and inspect the model and default session.
  Result: the model had no timeout field; `ClientSession().timeout.total` was
  300 seconds.
  Verdict: verified.
  Evidence: installed `lm_eval/models/api_models.py`; focused probe output.

- Hypothesis: connector concurrency waiting consumes aiohttp's cumulative
  timeout.
  Experiment: send two local aiohttp requests through `TCPConnector(limit=1)`;
  make the first take 0.20 seconds and set total timeout to 0.30 seconds.
  Result: the first completed at 0.202 seconds; the queued second timed out at
  0.301 seconds instead of receiving 0.30 seconds after admission.
  Verdict: verified.
  Evidence: aiohttp starts the total timer before `connector.connect()`; local
  queue probe.

- Hypothesis: a retained `ClientTimeout` plus semaphore admission before each
  retry attempt fixes both defects.
  Experiment: patch a minimal pinned-source fixture and the actual installed
  EVALS_META source; use a fake model call that records session timeout and
  start times without network or inference.
  Result: `timeout=7200` was retained and installed on the session; at
  concurrency one, the second call began only after the first released the
  logical slot. Reapplying the patch was idempotent.
  Verdict: verified.
  Fix: `workflows/workflow_venvs.py` now applies a version- and source-guarded
  lm-eval 0.4.4 compatibility patch after every EVALS_META dependency setup.
  It preserves the connector limit, applies `ClientTimeout(total=timeout)`, and
  acquires an `asyncio.Semaphore` before entering `session.post` on each retry
  attempt. Non-positive timeouts fail closed.
  Verification:
  - `pytest -q tests/test_meta_eval_dataset_mapping.py tests/test_qwen38_external_eval_config.py`
    -> 11 passed.
  - Installed patched source compiled with `python -m py_compile`.
  - Installed-source semantic probe -> stored/session timeouts matched and
    queue admission ordering passed.
  - Focused Ruff lint and `git diff --check` passed.

## Final Status

- Fixed locally with focused evidence. Full hardware-backed release rerun is
  owned by the parent release workflow and was intentionally not launched or
  modified during this investigation.
- No context length, generation budget, request count, or sampling policy was
  changed.

# AutoDebug: lm-eval IFEval timeout

## Headline finding

The configured `timeout=7200` never reaches aiohttp in the pinned lm-eval 0.4.4 adapter. `TemplateAPI.__init__` accepts it only through `**kwargs` and then discards it (`.workflow_venvs/.venv_evals_meta/lib/python3.10/site-packages/lm_eval/models/api_models.py:54-80,92-119`). `get_batched_requests` constructs a default `ClientSession` (`api_models.py:425-461`), so installed aiohttp 3.14.3 uses `ClientTimeout(total=300, sock_connect=30)` (`aiohttp/client.py:231-253,366`). The traceback is therefore an aiohttp 300-second total timeout, not the requested 7200-second timeout.

This also explains the approximately 15 minutes from actual generation launch to failure. The log records `Running generate_until requests` at 10:05:02, two successes, and final exceptions at 10:20:06 (`workflow_logs/run_logs/run_2026-08-31_10-04-55_id_autoport_Qwen3.8-Flash-Next_p300_release_release_A8_tO1JV.log:102-106,131-175`). lm-eval wraps `amodel_call` in tenacity with `stop_after_attempt(3)` (`api_models.py:437-441`), giving three approximately 300-second attempts. The apparent 10:08 start is tqdm/subprocess carriage-return buffering: the first visible progress update at 10:08:22 already reports 178.46 seconds elapsed (`release log:104-105`).

## Queue semantics

Yes: connector queue waiting consumes each aiohttp total timeout.

- The release command passes `num_concurrent=2` and `timeout=7200`, and lm-eval confirms both parsed values (`release log:66,75`).
- lm-eval creates `TCPConnector(limit=2)`, then starts all ten asyncio tasks immediately (`api_models.py:425-461`). Each task enters `session.post` directly (`api_models.py:349-375`).
- aiohttp starts its total timer before the request body that acquires a connector (`aiohttp/client.py:646-701,812`), while `TCPConnector.connect` waits for an available pool slot (`aiohttp/connector.py:623-637,682-709`). Thus time queued behind the two active requests is charged to the same 300-second attempt budget.
- The release traceback terminates inside aiohttp's request/response timer and raises `asyncio.TimeoutError` (`release log:135-175,181-222`), consistent with this client path.

The server's long request latency is the condition that filled the two-slot client pool, but it is not the source of the 300-second exception. No server/hardware interaction is needed to establish the client-side cause.

## Focused patch assessment

For the pinned lm-eval 0.4.4 source patch, the narrow fix is:

1. Accept and retain a positive numeric `timeout` in `TemplateAPI`.
2. Construct the aiohttp session with `ClientTimeout(total=self.timeout)`.
3. Gate each logical attempt with `asyncio.Semaphore(max(1, self._concurrent))` acquired before `session.post`, so waiting for a logical slot occurs before aiohttp starts the per-attempt total timer.
4. Keep `TCPConnector(limit=self._concurrent)` as defense in depth.

This gives the supplied 7200 seconds per active HTTP attempt and excludes lm-eval's own concurrency queue from that budget. The patch should explicitly define or reject non-positive/`None` values; the current release path supplies the positive integer 7200, so that edge policy does not block this fix.

## Focused verification experiments

1. Static/unit test: instantiate a minimal `TemplateAPI` subclass with `timeout=7200` and assert the value is retained and used to create `ClientTimeout(total=7200)`.
2. Queue test without inference: use a local aiohttp handler and concurrency one. Hold the first request longer than a deliberately short timeout, start a second request simultaneously, and prove the semaphore-gated second request receives its full timeout only after it acquires the logical slot. The current connector-only implementation should fail the control by charging queue time.
3. Retry test: force an aiohttp timeout and assert exactly `max_retries` attempts, with a fresh HTTP timeout per acquired attempt.
4. Release smoke after patch: retain `num_concurrent=2,timeout=7200`; verify more than 300 seconds can elapse for an active request without a client timeout and that queued requests do not share the earlier request's deadline.

## Scope

This was a source/log-only investigation. No inference, server, process, or hardware actions were performed, and no implementation files were changed.

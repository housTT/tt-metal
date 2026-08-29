# AutoFix Report: Final Serving Metrics Windows

## Starting evidence

- Failing command: `derive_serving_host_metrics.py --server-log readiness_vllm/server.log`.
- Failure: no markers matched the hard-coded primary `+127` and CI `+3168`
  raw `trace_replays` signatures.
- Final-source benchmark artifacts independently prove one 128/128 request at
  concurrency one and 32 100/100 requests at unbounded client concurrency:
  `vllm_benchmark.json` and `vllm_ci_serving_benchmark.json`.
- An AutoFix diagnosis fork was requested, but the shared agent-thread limit was
  full. The same hypothesis loop was run serially without touching hardware,
  the live server, or production model/plugin code.

## Hypothesis experiment

- Hypothesis: prefill program-cache growth invalidated the live decode trace, so
  the first decode step recaptured it and did not increment `trace_replays`.
- Experiment: enumerate every chained marker triple with exact logical request
  deltas `+1` and `+32`, then match canonical decode steps as
  `trace_replays delta + prefill_trace_invalidations delta`.
- Result: exactly one triple matched: marker indices `631 -> 632 -> 649`, with
  cumulative request counts `806 -> 807 -> 839`.
  - Primary: `126` replays + `1` invalidation/recapture = `127` decode steps.
  - CI: `3167` replays + `1` invalidation/recapture = `3168` decode steps.
- Verdict: verified. Requiring only the off-by-one replay counts without an
  invalidation remains rejected by a regression test.
- Fix: select on exact request counts plus replay-or-recapture steps. Counter
  resets/decreases remain invalid, and raw replay/invalidation components are
  recorded in the output artifact.

## Window evidence

Primary 128/128/1:

- assignments/releases/stale: `1/1/0`; bank commits/restores/resets: `0/0/0`.
- expert requests/hits/misses/evictions: `6784/24868/42796/42796`.
- expert H2D: `118322380800` bytes in `9.5012118568` s.
- PLE lookups/selected rows/H2D: `129/4096/2621440` bytes.
- route-read-and-TT stall: `24.0206442053` s.

CI 100/100/32:

- assignments/releases/stale: `32/32/0`; bank
  commits/restores/resets: `3200/3167/32`.
- expert requests/hits/misses/evictions: `169512/181758/1506457/1506457`.
- expert H2D: `4165052313600` bytes in `329.6592053064` s.
- PLE lookups/selected rows/H2D: `3201/101904/74393600` bytes.
- route-read-and-TT stall: `621.8338766684` s.

Both windows have zero model-only trace replays, seed copies, host-sampling
compatibility calls, sampling-mode switches, and stale rejections.

## Verification and final status

- `python_env/bin/python -m pytest -q models/autoports/qwen_qwen3_8_flash_next/tests/test_readiness_vllm_scripts.py`
  passed `8/8`.
- The original derivation command now succeeds and writes
  `readiness_vllm/serving_host_metrics.json`.
- Source artifacts:
  - `server.log`: SHA-256 `487182b4fde44ddc68e3926e2f8cb3313493c6475dbe95c8957f1039f45585ac`
  - `vllm_benchmark.json`: SHA-256 `03dd117d3ae06d9d8abd35744512edfa3b9ddf387e09d2b855e5935320747085`
  - `vllm_ci_serving_benchmark.json`: SHA-256 `b039c35873c97a07d42b8a61663f6b274ec145e72a66564e0404d5e74c723307`
- Final status: fixed. No hardware or production serving code changed.

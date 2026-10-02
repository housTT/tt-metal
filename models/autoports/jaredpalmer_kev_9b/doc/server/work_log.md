# Stage 2 (serving): work log

Date: 2026 Oct 01, 20:10 to 20:25 UTC (every time in this log is the host clock, UTC). One chip (chip 0 through `ttnn.open_device(device_id=0)`), eager, no traces. All paths are absolute. Reports under `/home/hous/dev/kev/reports`, logs under `/home/hous/dev/kev/logs`.

## Environment

- Interpreter: `/home/hous/dev/kev/tt-metal/python_env/bin/python` through `/home/hous/dev/kev/bin/devrun` (device) and `/home/hous/dev/kev/bin/hostrun` (host). fastapi 0.142.2, uvicorn 0.54.0, httpx 0.28.1.
- kev client: `cd /home/hous/dev/kev/kev && uv run python -m kev.benchmark` (HEAD 952ce9d).
- Weights: `HF_MODEL=/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/68c46c4b3498877f3ef123c856ecfde50c39f404`, `KEV_RUN=/home/hous/.cache/huggingface/hub/models--jaredpalmer--kev-9b/snapshots/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0`, `TT_CACHE_PATH=/home/hous/dev/kev/tt_cache` (warm `P150/tensor_cache_bfp8_kev_2b2a70cf`), `HF_HUB_OFFLINE=1`, `KEV_MESH_SHAPE=1x1`, `KEV_DEVICE_ID=0`. Defaults otherwise: `KEV_PREFIX_CACHE=8`, `KEV_MAX_STATE=8192`, `KEV_MAX_QUESTION=2048`, `KEV_TRUNCATE_STATES=0`, trace region 1 GiB (unused, eager).

## Fix made before the device run

`tt/server.py`, `Worker`: the worker called `engine.prefill_state(ids)` without a slot, so every cached `StateHandle` pointed at GDN snapshot slot 0 and a cache hit on any state but the last one prefilled would have read another state's recurrent and conv state. Now the cache maps state key to `(handle, slot)`, the worker keeps `free_slots = range(cache_size)` after the engine is built (`cache_size` clamped to `engine.snapshot_slots`, 8 in `KevEngine`), a miss takes a free slot or evicts the least recently used entry and reuses its slot, and `prefill_state(ids, slot=slot)` is called with it. `FakeEngine.prefill_state` accepts the `slot` keyword. `tests/test_server_api.py`: 12 passed (fake engine, 3.0 s) after the change. No other change to the device path was needed: engine construction on the worker thread, warmup, device close and the 422 paths worked on the first device run.

## Server run

```bash
cd /home/hous/dev/kev/tt-metal && export HF_MODEL=... KEV_RUN=... TT_CACHE_PATH=/home/hous/dev/kev/tt_cache HF_HUB_OFFLINE=1 KEV_MESH_SHAPE=1x1 KEV_DEVICE_ID=0
/home/hous/dev/kev/bin/devrun timeout 3600 python -m uvicorn models.autoports.jaredpalmer_kev_9b.tt.server:app --host 127.0.0.1 --port 8008 --lifespan on > /home/hous/dev/kev/logs/stage2_server.log 2>&1
```

Log `/home/hous/dev/kev/logs/stage2_server.log`: `starting:` at 20:11:00, `worker 0 ready` at 20:11:29 (HF load plus host LoRA merge plus weight cache load, 29 s), warmup 3 questions 588.9 ms, `Application startup complete` at 20:11:29. The worker label in the log is `1`: `str(d.id())` on the device returned by `ttnn.open_device(device_id=0)` is the MeshDevice id (`Enabling program cache on MeshDevice 1`), not the chip id. `devrun` held `/home/hous/dev/kev/.device.lock` for the life of the server.

## Smoke (20:12 UTC)

- `GET /health`: `{"status":"ok","workers":1,"queued":0}`, 8 ms.
- `GET /v1/models`: two cards (`kev-latest`, `jev-latest`), `backend ttnn`, `device ttnn 1x1 x1 worker(s)`, `dtype bf16`, `temperature 2.193649959389252`, `max_state_tokens 8192`, `max_question_tokens 2048`, `prefix_cache {size 8, hits 0, misses 1, cached_states 1}` after warmup.
- Quickstart request (state "I was charged twice for order 1182. Please refund one of the charges.", `team` choice billing / shipping / returns, `urgent` noul):

```json
{"model":"kev-latest","answers":{"team":{"type":"choice","choice":"billing","confidence":0.9248,"probabilities":{"billing":0.9499,"shipping":0.0246,"returns":0.0256}},"urgent":{"type":"noul","noul":0.3713}},"usage":{"input_tokens":62,"output_tokens":85},"latency_ms":274.4}
```

Wall 281 ms, `server-timing: app;dur=276.4`. Expected from the fp32 reference: team = billing (matches), p(urgent) = 0.43 (served 0.3713, |dp| 0.0587, the same gap the stage 1 full-row engine test measured for this row, record 12 `urgent` in `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/functional/work_log.md`).

## Parity through HTTP (20:12 UTC)

Script: `/tmp/claude-1002/-home-hous-dev-kev/0a6793e1-6f42-41ef-926e-8b91dbe0b95b/scratchpad/parity.py` (scratch; its logic is: POST `api_request(record)` for the 16 records of `/home/hous/dev/kev/reports/reference/records.jsonl`, rebuild one probability vector per question in the reference key order, noul as `[1 - noul, noul]`, then kev's `agreement()` from `scripts/serving_bench.py:104`). Output `/home/hous/dev/kev/reports/stage2_parity.json`.

| vs | questions | max_dp | mean_dp | argmax_flips |
|---|---|---|---|---|
| probs_fp32.json | 29 | 0.0947 | 0.0359 | 0 |
| probs_bf16.json | 29 | 0.0992 | 0.0366 | 0 |
| (bf16 reference vs fp32 reference, for scale) | 29 | 0.0189 | 0.0034 | 0 |

Served probabilities are rounded to 4 decimals by the API (kev `round_prob`), so each max_dp includes up to 5e-5 of rounding. Per record (questions, input tokens, server `latency_ms`, max |dp| vs fp32, vs bf16):

| record | q | tokens | latency_ms | fp32 | bf16 |
|---|---|---|---|---|---|
| 0 | 2 | 354 | 620.3 | 0.0825 | 0.0837 |
| 1 | 1 | 173 | 242.2 | 0.0559 | 0.0568 |
| 2 | 2 | 336 | 428.5 | 0.0035 | 0.0041 |
| 3 | 1 | 181 | 243.0 | 0.0095 | 0.0098 |
| 4 | 1 | 1585 | 2761.1 | 0.0333 | 0.0329 |
| 5 | 1 | 389 | 365.0 | 0.0125 | 0.0144 |
| 6 | 1 | 77 | 125.8 | 0.0450 | 0.0439 |
| 7 | 2 | 268 | 367.1 | 0.0385 | 0.0409 |
| 8 | 1 | 501 | 871.6 | 0.0027 | 0.0029 |
| 9 | 1 | 208 | 187.5 | 0.0461 | 0.0458 |
| 10 | 1 | 248 | 242.4 | 0.0231 | 0.0230 |
| 11 | 1 | 614 | 662.4 | 0.0010 | 0.0012 |
| 12 | 2 | 62 | 250.4 | 0.0587 | 0.0565 |
| 13 | 6 | 253 | 747.3 | 0.0947 | 0.0992 |
| 14 | 5 | 2392 | 2271.6 | 0.0866 | 0.0921 |
| 15 | 1 | 63 | 125.6 | 0.0512 | 0.0557 |

Stage 1 measured the same rows through `prefill_hidden` (one full row per question): max |dp| 0.1024, 29/29 argmax. The served path (state prefix pass, GDN snapshot, tail per question) gives 0.0947, so the prefix cache path is at least as close to the reference as the full-row path. Median server latency 366 ms, sum 10.5 s for the 16 records.

## kev client eval, smoke-v1 (20:12 to 20:13 UTC)

```bash
cd /home/hous/dev/kev/kev && uv run python -m kev.benchmark --remote http://127.0.0.1:8008 --suite /home/hous/dev/kev/kev/evals/smoke-v1 --out /home/hous/dev/kev/reports/eval/smoke-v1/development --remote-concurrency 1
```

Exit 0, log `/home/hous/dev/kev/logs/stage2_smoke_v1_eval.log`. `report.json`: coverage 30/30 records, 40/40 questions, 0 rejected, 0 truncated; `clean`: n 18, acc 0.8889, brier 0.1606, ece 0.0854, nll 0.2637; `latency_ms` (client round trip) median 217.8, p95 739.3; `remote.served_model kev-latest`. `summarize_eval.py` prints the same row: `smoke-v1/development n=18 acc 0.889 brier 0.161 ece 0.085 latency p50 218 ms p95 739 ms`. No fp32 smoke-v1 reference exists on this box, so these numbers are recorded, not compared.

## Prefix cache (20:13 UTC)

Script (scratch) `cache_check.py`, output `/home/hous/dev/kev/reports/stage2_cache_check.json`.

- Same 27-token state, `team` then `urgent`: 149.8 ms then 125.9 ms, log `cache_hit=False` then `cache_hit=True`. The drop is small because a state under 128 tokens has no 128-aligned prefix to cache (`S0 = 0`); the hit saves only the GDN reset and RoPE staging, and the state tokens still run in every question row.
- 2,200-token state (bench `request(LONG, 7)`, 2,317 input tokens, 5 questions): fresh 2307.8 ms, hit 654.2 ms, answers equal (dict equality on the 4-decimal probabilities). Short 6-question state: fresh 840.1 ms, hit 788.4 ms, answers equal. Recorded in `/home/hous/dev/kev/reports/stage2_robustness.json`.
- Eviction with 8 slots: 9 distinct states (28 tokens each) then state 0 again: state 0 is a miss (evicted, `misses` 45 to 46, `cached_states` stays 8), recomputed answers equal the fresh ones; state 1 again is then also a miss (it became the least recently used) with equal answers; state 8 again is a hit (`hits` 14 to 15) with equal answers. No crash, no error in the server log.

## Robustness (20:14 UTC)

- 8 clients x 5 requests from `/home/hous/dev/kev/kev/evals/v7/decision-v7/development.jsonl` (first 40 records, sha256 checked): 40/40 HTTP 200, 0 non-200, 27.0 s wall, 1.48 requests/s; model `latency_ms` p50 666.3, p99 796.9, max 830.0; client wall p50 5331.8 ms, p99 5463.4 ms (one eager worker, so a request waits for the 7 ahead of it). `/home/hous/dev/kev/reports/stage2_concurrent.json`. Server log: zero 5xx, zero tracebacks.
- Oversize state, 70,081 tokens: 422 in 69 ms with kev's wording (`state is 70,082 tokens, over the 65,536-token limit ...`). A 10,952-token state: 422 against the 8,192 `KEV_MAX_STATE` limit with the `KEV_TRUNCATE_STATES=1` hint. Truncation mode itself was not exercised on the device (covered by the fake-engine test `test_truncate_mode_marks_responses`).

## Serving bench (20:15 UTC)

```bash
cd /home/hous/dev/kev/tt-metal && /home/hous/dev/kev/bin/hostrun timeout 2400 python models/autoports/jaredpalmer_kev_9b/scripts/serving_bench_remote.py --base-url http://127.0.0.1:8008 --label "P150 (1 chip, eager, stage 2)" --reps 20 --quick --concurrency 1,8,32,64 --out /home/hous/dev/kev/reports/bench/p150_stage2
```

Exit 0, 20:15 to 20:22 UTC, log `/home/hous/dev/kev/logs/stage2_bench.log`, report `/home/hous/dev/kev/reports/bench/p150_stage2/report.json` (`quick: true`, `reps: 20`, `records: 32`). Reps: the latency section used the full 20 (median of 20 server `latency_ms` after 2 warm requests per case and mode); the throughput section used `--quick` (32 / 32 / 8 requests per level and pass instead of 256 / 256 / 64) to fit inside the server's 1 h `timeout 3600`.

| case | tokens | first_ms | new_ms | cached_ms |
|---|---|---|---|---|
| 2 questions, short state | 89 | 505.8 | 490.1 | 505.7 |
| 6 questions, short state | 253 | 754.7 | 753.2 | 750.8 |
| 5 questions, 370-token state | 567 | 1058.5 | 1119.6 | 879.5 |
| 5 questions, 2,200-token state | 2392 | 2271.3 | 2271.4 | 647.5 |

Card row (written to `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/server/README.md`, section "Stage 2 baseline (eager, 1 chip)"):

| Device | 6 questions, short state | 5 questions, 2,200-token state | Requests/s, 64 clients |
|---|---|---|---|
| P150 (1 chip, eager, stage 2) | 753.2 / 750.8 ms | 2271.4 / 647.5 ms | 1.3 |

Throughput, second pass (first pass in parentheses where it differs), `p50_ms / p99_ms / requests_per_s`:

| sample | 1 client | 8 clients | 32 clients | 64 clients |
|---|---|---|---|---|
| 6 questions, new short state (32) | 758 / 764 / 1.3 | 6025 / 6039 / 1.3 | 12502 / 23416 / 1.3 | 12435 / 23339 / 1.3 |
| decision-v7 development (32, repeats) | 129 / 306 / 5.9 | 1261 / 1793 / 6.0 | 2470 / 5214 / 6.0 | 2469 / 5150 / 6.0 |
| 5 questions, 2,200-token state (8) | 654 / 659 / 1.5 (first 2280 / 2281 / 0.4) | 2918 / 4539 / 1.5 | 2926 / 4546 / 1.5 | 2927 / 4549 / 1.5 |

One eager worker serializes everything, so `requests_per_s` is flat across client counts and wall latency grows with the queue. The 8-request long sample fits the 8 cache slots, so every pass after the first is all hits. Server totals at the end: 863 POSTs, 0 5xx, 0 tracebacks, prefix cache hits 17 / misses 89 at the last `/v1/models` read by the bench.


## Shutdown

20:22:30 UTC: `kill -TERM <uvicorn pid>`. Log: `Shutting down`, `Waiting for application shutdown.`, `Application shutdown complete.`, `Finished server process [2423029]`. The `devrun` wrapper exited with 143: uvicorn 0.54 re-raises the captured SIGTERM after a graceful shutdown (`python_env/lib/python3.12/site-packages/uvicorn/server.py:348`), so the process ends by signal and tt-metal's static destructor line `Closing user mode device drivers` does not appear in the server log. Checks after exit: `pgrep -af "uvicorn models.autoports"` empty, no `flock` process, `flock -n /home/hous/dev/kev/.device.lock true` succeeds (lock free). Device release proven by a fresh process: `devrun timeout 180 python -c "import ttnn; d = ttnn.open_device(device_id=0); ttnn.close_device(d)"` opened chip 0 in 2.8 s (`d.id()` 1, the MeshDevice id) and closed with `Closing user mode device drivers`, log `/home/hous/dev/kev/logs/stage2_device_release_check.log`.

Files touched in this stage: `tt/server.py` (slot allocation), `doc/server/README.md` (prefix cache paragraph, stage 2 baseline section), `doc/server/work_log.md` (this file). Reports: `/home/hous/dev/kev/reports/stage2_parity.json`, `stage2_cache_check.json`, `stage2_concurrent.json`, `stage2_robustness.json`, `/home/hous/dev/kev/reports/eval/smoke-v1/development/`, `/home/hous/dev/kev/reports/bench/p150_stage2/`. Nothing committed.


## Open issues

- Eager only: each question row is a separate chunked prefill with host-side dispatch; a 6-question short request costs 0.8 s and a single short question 0.13 s. Stage 3 (traces, batching of question rows into one pass) owns this.
- Record 4 (1,585-token state, one question) takes 2.76 s, longer than record 14 (2,392 tokens, five questions, 2.27 s). Stage 1 saw the same row at 2.56 s through the full-row path. Cause not investigated.
- `card.device` and the worker label show the MeshDevice id, not the chip id.
- The prefix cache aligns to 128 tokens, so short states never benefit from a hit beyond the GDN reset. kev's reference server caches whole states.
- `/v1/models` `max_state_tokens` is 8,192 (the engine limit) while kev's own admission limit (65,536) still applies first; the two 422 messages name different limits depending on which one trips.
- breadth-v1 is still NOT-AVAILABLE (private mirror). The other suites (hard-v1, devtools-v1, documents-v1) were not run in this stage; stage 6 owns them.
- Multi-chip (`KEV_MESH_SHAPE=2x2`, four workers) is untested on the device; stage 5 owns it.

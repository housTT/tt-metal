# kev-9b serving on Tenstorrent: design, environment, bench and eval

Code: `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/tt/server.py` (ASGI `app`, FastAPI).
Reference it mirrors: `/home/hous/dev/kev/kev/kev/serve.py` and `/home/hous/dev/kev/kev/kev/api.py` at kev commit 952ce9d.

## Routes

| Route | Body |
|---|---|
| `POST /v1/systemone` | `{"model", "answers", "usage": {"input_tokens", "output_tokens"}, "latency_ms"}`. With `KEV_TRUNCATE_STATES=1` every response also carries `truncated` and `usage.state_tokens` / `usage.state_tokens_used`. Same answer shapes as kev: `choice` (`choice`, `confidence`, `probabilities`), `noul` (`noul`), `score` (`score`, `legend`, `probabilities`, `confidence`). |
| `GET /v1/models` | One card per name in `kev-latest`, `jev-latest`: `description`, `release_date`, `run`, `base`, `device`, `backend` (`ttnn` or `fake`), `dtype`, `temperature`, `max_state_tokens`, `max_question_tokens`, `truncate_states`, `prefix_cache` (pooled hits, misses, cached states), `workers` (per worker queue depth, cache, hits, misses, requests), `batches`. |
| `GET /health`, `GET /v1/health` | `{"status": "ok", "workers", "queued"}`. Never behind the API key. |

Not ported from kev's `serve.py`: `KEV_DATE_FACTS` preprocessing (`prepare()`), `/v1/systemone/permute` and `/v1/systemone/separate` (404 here); the `/v1/models` card lacks kev's `cuda_graphs` and the `prefix_cache.min_state_tokens` / `max_tokens` / `oom_retries` fields and adds `max_question_tokens` and `workers`. Byte compatibility covers `POST /v1/systemone`, the health routes and the error texts.

Every response carries `x-typesafe-request-id` (echoed from the request when given) and `server-timing`. `KEV_API_KEY` set means `/v1/*` (except the health routes) needs `Authorization: Bearer <key>`, else 401 with the same detail text as kev. Errors: 422 with kev's wording for a state over the limit (and the hint to set `KEV_TRUNCATE_STATES=1`), for a question row over the limit, and for request validation. `latency_ms` is the model time of the request on its worker (state prefill or cache hit, then the question rows and the head), not its wait in the queue.

## Design

```
request thread            worker thread k (one per chip)              device k
  rows_for_record  ---->  queue  ->  prefill_state(S) or LRU hit  ---> ttnn ops
  (422 here)              |          per question: question_hidden  ---> ttnn ops
  pick worker             |          PointerHead.probs (CPU fp32)
  await future    <----   done
```

- Startup runs inside the ASGI lifespan: resolve `KEV_RUN`, load the tokenizer (`HF_MODEL`, falling back to the adapter directory) and `PointerHead`, open the devices, build one `KevEngine` per worker on that worker's own thread, then one warmup request per worker. uvicorn prints `Application startup complete` only after that, so the readiness line is the real readiness.
- One process owns the devices. `ttnn.open_mesh_device(MeshShape(r, c), l1_small_size=24576, num_command_queues=2, trace_region_size=KEV_TRACE_REGION)` then `create_submeshes(MeshShape(1, 1))`; one worker thread per submesh. For a single chip (`1x1`) it is `ttnn.open_device(device_id=KEV_DEVICE_ID, ...)`. Every ttnn call for a device happens on its worker thread only; the main thread opens and closes the devices.
- A request goes whole to one worker: the worker whose prefix cache holds the state (key = sha1 of the state token ids), else the least loaded (queue depth plus in flight, ties by id).
- Prefix cache: a per-worker least-recently-used map from state key to `(StateHandle, slot)`, at most `KEV_PREFIX_CACHE` entries (default 8), clamped to `engine.snapshot_slots` (the engine fits as many slots as the requested count and the free device DRAM allow, 8 at `KEV_MAX_STATE` 65536 on a P150; unlimited in the fake engine). Each slot owns its own GDN snapshot and its own disjoint range of paged KV blocks with a per-slot page table (stage 3 fix of the stage 1 and 2 review P1: before it, every slot shared one KV cache and a hit after another state's prefill read that state's keys and values). A miss takes a free slot, or evicts the least recently used entry and reuses its slot, then calls `prefill_state(ids, slot=slot)`; if `prefill_state` raises, the slot goes back to the free list. Only the 128-aligned prefix of a state is cached; the remainder (under 128 tokens) is recomputed with every question row, so a hit on a short state saves little and a hit on a 2,200-token state saves the whole state pass (2.3 s to 0.65 s eager).
- Admission: the engine is built with `max_state_len=KEV_MAX_STATE` (default 65536, kev's own serving limit) and `MAX_QUESTION_LEN` 2048, so a state over `KEV_MAX_STATE` tokens (the `<state>` token included) gets a 422 unless `KEV_TRUNCATE_STATES=1`, which keeps its first `KEV_MAX_STATE` tokens and marks the response. kev's own limit (65,536) still applies first through the vendored `admit`. A question row over `KEV_MAX_QUESTION` tokens, or whose tail (state remainder past the 128-aligned prefix plus the question) is over it, gets a 422 either way.
- Sync entry points for tests and tools: `Server.predict(record) -> (probs, stats)`, `Server.answer(req) -> body`. The HTTP handler awaits a `Future` per worker queue (`answer_async`), so a waiting request holds no thread.
- One log line per request: `worker=<id> S=<state tokens used> questions=<n> latency_ms=<ms> cache_hit=<bool>`.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `HF_MODEL` | `Qwen/Qwen3.5-9B-Base` | Base checkpoint (Hub id or local dir). The server sets it when unset, because `Qwen36ModelArgs` reads it too. |
| `KEV_RUN` | `jaredpalmer/kev-9b` | Adapter + `head.pt` (Hub id with optional `@revision`, or a local dir). `KevModelArgs` reads the same variable. |
| `KEV_MESH_SHAPE` | unset | Parent mesh shape `RxC`. tt-model sets it from the profile's mesh (`P150x4` gives `1x4`). On the p300c x2 box use `2x2` (STATUS.md rule). |
| `MESH_DEVICE` | unset | When `KEV_MESH_SHAPE` is unset: `P150` (or unset) means one chip; `P150x4`, `P300x2`, `QB2` mean `2x2`. |
| `KEV_DEVICES` | `all` | Which submeshes get a worker, by index, e.g. `0,2`. |
| `KEV_DEVICE_ID` | `0` | Chip id for the single-chip path. |
| `KEV_TRACE_REGION` | `1073741824` | `trace_region_size` for the device open. |
| `KEV_PREFIX_CACHE` | `8` | States kept per worker (clamped by the engine's snapshot slots). |
| `KEV_MAX_STATE` | `65536` | Engine `max_state_len` and the 422 limit for states (kev serves 65,536; stage 3 moved the default from 8192 after the per-slot KV change, 2.06 GiB of KV per slot at this length). |
| `KEV_TRACED` | `1` | `0` builds the eager engine (stage 2 path, no traces); `1` captures the stage 3 traces at startup. |
| `KEV_MATMUL_POLICY` | `1` | `0` disables the stage 3 matmul program-config policy (`MATMUL_POLICY` in `tt/engine.py`). |
| `KEV_MAX_QUESTION` | `2048` | 422 limit for a question row. |
| `KEV_TRUNCATE_STATES` | `0` | `1` reads the first `KEV_MAX_STATE` state tokens instead of refusing. |
| `KEV_API_KEY` | unset | Bearer key for `/v1/*`. |
| `KEV_FAKE_ENGINE` | `0` | `1` opens no device: `KEV_FAKE_WORKERS` (default 2) workers return deterministic pseudo-random hidden rows, the real head and tokenizer run. For API tests on a CPU. |
| `HF_HUB_OFFLINE` | unset | `1` resolves Hub ids from the local cache only. |

## Run

One chip (chip 0, the dev chip in STATUS.md), from the tt-metal env:

```bash
export HF_MODEL=/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/68c46c4b3498877f3ef123c856ecfde50c39f404
export KEV_RUN=/home/hous/.cache/huggingface/hub/models--jaredpalmer--kev-9b/snapshots/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0
export HF_HUB_OFFLINE=1 KEV_MESH_SHAPE=1x1 KEV_DEVICE_ID=0 TT_CACHE_PATH=/home/hous/dev/kev/tt_cache
/home/hous/dev/kev/bin/devrun python -m uvicorn models.autoports.jaredpalmer_kev_9b.tt.server:app --host 127.0.0.1 --port 8008 --lifespan on
```

Four chips (one process, 2x2 parent mesh, four workers):

```bash
export KEV_MESH_SHAPE=2x2
/home/hous/dev/kev/bin/devrun python -m uvicorn models.autoports.jaredpalmer_kev_9b.tt.server:app --host 0.0.0.0 --port 8008 --lifespan on
```

Two of the four chips: add `KEV_DEVICES=2,3`. `devrun` takes the box-wide device lock; only one device-owning process runs at a time.

No device (API tests and plumbing checks):

```bash
KEV_FAKE_ENGINE=1 KEV_FAKE_WORKERS=4 python -m uvicorn models.autoports.jaredpalmer_kev_9b.tt.server:app --port 8018 --lifespan on
```

Quickstart request:

```bash
curl -s localhost:8008/v1/systemone -H 'content-type: application/json' -d '{"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.", "model": "kev-latest", "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}}, "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"}, "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]}}}'
```

## Tests

```bash
/home/hous/dev/kev/bin/hostrun python -m pytest models/autoports/jaredpalmer_kev_9b/tests/test_server_api.py -q --noconftest
```

The file sets `KEV_FAKE_ENGINE=1` itself and uses the cached adapter and base snapshots. It covers the choice / noul / score round trip and field sets, request id echo, determinism and cache hits, `/v1/models`, both health routes, validation 422s, the oversize-state 422 text, truncation marks (`Server.start` with `truncate=True`), bearer auth (health routes stay open), the sync `predict`, and concurrent requests over two workers. Until `python_env` exists it also runs from the kev venv: `cd /home/hous/dev/kev/kev && PYTHONPATH=/home/hous/dev/kev/tt-metal HF_HUB_OFFLINE=1 .venv/bin/python -m pytest <same path> -q --noconftest -p no:cacheprovider`.

## Serving bench (port of kev `scripts/serving_bench.py`, over HTTP)

```bash
python models/autoports/jaredpalmer_kev_9b/scripts/serving_bench_remote.py --base-url http://127.0.0.1:8008 --label "P150 (1 chip)" --reps 20 --concurrency 1,8,32,64 --out /home/hous/dev/kev/reports/bench/p150
```

Same workloads as the original (TICKET, QUESTIONS, PARAGRAPH x 30, `request(case, i)`), same measurement: `first_ms`, then `new_ms` and `cached_ms` as the median of the server's `latency_ms` over `reps + 3` requests with the first 2 dropped; throughput on 256 requests of 6 questions over a new short state, 256 decision-v7 development records (`--suite`, sha256 checked against the manifest) and 64 requests of 5 questions over a 2,200-token state, at each concurrency level, two passes, the second reported with the first under `first`; `p50_ms`, `p99_ms`, `requests_per_s` computed as in the original. `report.json` holds `latency`, `throughput`, the served model card and `card_row`. The script prints the model-card row: `| <label> | <new> / <cached> ms | <new long> / <cached long> ms | <requests_per_s at 64 clients, 6 questions new short state> |`. `--quick` (32 / 8 throughput requests) is for plumbing checks only and is recorded in the report.

## Stage 2 baseline (eager, 1 chip)

Measured 2026 Oct 01, 20:15 to 20:22 ET, on chip 0 (`KEV_MESH_SHAPE=1x1`, `KEV_DEVICE_ID=0`), eager, no traces, one worker, prefix cache 8. Command: `serving_bench_remote.py --reps 20 --quick --concurrency 1,8,32,64 --label "P150 (1 chip, eager, stage 2)" --out /home/hous/dev/kev/reports/bench/p150_stage2`. The latency columns use the full `reps 20` (median of 20 server `latency_ms` values after 2 warm requests, per the original script); the throughput section ran in `--quick` mode (32 short requests, 32 decision-v7 records, 8 long requests per level and pass instead of 256 / 256 / 64) so that the whole run fit in the server's 1 h `timeout`. Report: `/home/hous/dev/kev/reports/bench/p150_stage2/report.json`, log `/home/hous/dev/kev/logs/stage2_bench.log`.

| Device | 6 questions, short state | 5 questions, 2,200-token state | Requests/s, 64 clients |
|---|---|---|---|
| P150 (1 chip, eager, stage 2) | 753.2 / 750.8 ms | 2271.4 / 647.5 ms | 1.3 (`--quick`: 32 requests per level, not 256) |

Columns are new / cached state. Other latency rows: 2 questions, short state 490.1 / 505.7 ms; 5 questions, 370-token state 1119.6 / 879.5 ms. Throughput (second pass, `requests_per_s`): 6 questions new short state 1.3 at every level (one eager worker; p50 wall 758 ms at 1 client, 12.4 s at 32 and 64 clients); decision-v7 development 5.9 to 6.0 (p50 129 ms at 1 client; the 32-record `--quick` sample repeats records, so it includes cache hits); 2,200-token state 1.5 at every level (the 8-request `--quick` sample fits the 8 cache slots, so the second pass is all hits: p50 654 ms at 1 client against 2280 ms on the first pass). A short state (under 128 tokens) gains nothing from a cache hit because only the 128-aligned prefix is cached.

## Stage 3 (traced, 1 chip)

Measured 2026 Oct 01, 21:33 to 21:39 ET, on chip 0 (`KEV_MESH_SHAPE=1x1`, `KEV_DEVICE_ID=0`), traced engine with the matmul policy (defaults), one worker, prefix cache 8 slots with per-slot KV (`KEV_MAX_STATE` 65536). Startup to `Application startup complete`: 44 s (weights 24 s, 8 KV slots and 27 traces 20 s). Same bench command as stage 2 with `--label "P150 (1 chip, traced, stage 3)" --out /home/hous/dev/kev/reports/bench/p150_stage3`; latency columns full `--reps 20`, throughput `--quick` (32 / 32 / 8 requests per level). Report `/home/hous/dev/kev/reports/bench/p150_stage3/report.json`, logs `/home/hous/dev/kev/logs/stage3_server.log`, `stage3_bench.log`.

| Device | 6 questions, short state | 5 questions, 2,200-token state | Requests/s, 64 clients |
|---|---|---|---|
| P150 (1 chip, eager, stage 2) | 753.2 / 750.8 ms | 2271.4 / 647.5 ms | 1.3 (`--quick`, 32 requests per level) |
| P150 (1 chip, traced, stage 3) | 604.7 / 605.2 ms | 1528.5 / 525.5 ms | 1.6 (`--quick`, 32 requests per level) |

Columns are new / cached state. Other latency rows: 2 questions, short state 202.1 / 201.6 ms (stage 2: 490.1 / 505.7); 5 questions, 370-token state 862.1 / 677.0 ms (stage 2: 1119.6 / 879.5). Throughput (second pass): 6 questions new short state 1.6 requests/s at every client count; decision-v7 development 7.6 (stage 2: 6.0); 2,200-token state 1.9 (stage 2: 1.5). Server totals: 790 requests, 0 5xx, 0 tracebacks, prefix cache 151 hits / 639 misses.

Parity through HTTP with the prefix cache exercised (`scripts/parity_remote.py`, two passes over the 16 reference records, the second in reverse order so every state is revisited after the others ran; `/home/hous/dev/kev/reports/stage3_parity.json`): 29 questions, max |dp| 0.0947, mean |dp| 0.0358, 0 argmax flips against fp32 (stage 2: 0.0947 / 0.0359); 0.0992 / 0.0364 against the bf16 reference; 16 of 16 revisited records returned answers equal to their first pass (24 misses, 10 hits over 8 slots). Median `latency_ms` 295 (stage 2: 366), record 14 (2,392 tokens, 5 questions) 1528.5 ms new and 524.2 ms on the revisit.

Shutdown: `kill -TERM` to the uvicorn process; `Shutting down`, `Application shutdown complete`, `Finished server process`; the lock was free afterwards and a fresh process opened chip 0 (`/home/hous/dev/kev/logs/stage3_device_release_check.log`). The driver script's first `kill` went to the `timeout` wrapper (`pgrep` matched it first) and did not stop the server; the second, to the python process, did.

## Evaluation

```bash
models/autoports/jaredpalmer_kev_9b/scripts/run_eval.sh --base-url http://127.0.0.1:8008 --concurrency 4            # development split
models/autoports/jaredpalmer_kev_9b/scripts/run_eval.sh --base-url http://127.0.0.1:8008 --concurrency 4 --test     # adds the locked test split (--allow-test)
models/autoports/jaredpalmer_kev_9b/scripts/run_eval.sh --base-url http://127.0.0.1:8018 --suites smoke-v1 --concurrency 2
```

Runs `kev.benchmark --remote <url> --suite evals/<suite> --out /home/hous/dev/kev/reports/eval/<suite>/<split> --remote-concurrency N [--allow-test]` in the kev venv (`uv run`) for hard-v1, devtools-v1, documents-v1, breadth-v1 and smoke-v1. Before each run it asks `kev.suite.load_split` for the partition and prints `NOT-AVAILABLE <suite>/<split>: <reason>` instead of failing when the data is not readable. breadth-v1 keeps only `manifest.json` in git and names a private mirror (`jaredpalmer/kev-private-evals`, revision `aeaaed0f`); `load_split` fetches from it only for an account with access, else raises `PermissionError`. A suite whose `report.json` exists is skipped; an incomplete output dir is moved aside.

```bash
python models/autoports/jaredpalmer_kev_9b/scripts/summarize_eval.py
```

Prints, per suite and split, `clean.n / acc / brier / ece` from each `report.json` (`clean` is kev's own headline subset: the records whose source is not `unknowable` and that are not control variants; for smoke-v1 that is 18 of the 40 questions, the other 22 being the `permuted`, `none_present` and `none_absent` controls, so the smoke-v1 `clean` row is a plumbing check, not a quality claim) (kev's `metrics()` already computes ECE with 10 equal-width bins over the top probability) plus the request latency, and a pooled `hard-v1 + devtools-v1 audited` row recomputed from `rows.json` with the same formulas and the model card's exclusions (source `flakeflagger`, task `commitpackft_type`). The model-card references are printed below the table: hard-v1 + devtools-v1 audited 0.821 development / 0.822 test, documents-v1 0.902 / 0.900, breadth-v1 0.700 / 0.698 with test ECE 0.034, transfer-v4 locked test ECE 0.034.

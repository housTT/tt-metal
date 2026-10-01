# kev-9b stage 5: data parallel over 4 chips (one process, four submeshes, question-level fan-out)

Date: 2026 Oct 01. Box: p300c, 4 Blackhole P150 chips. Host-only preparation; nothing in this directory has run on a device yet. Device numbers below are stage 3 measurements on chip 0 (`/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/optimized/perf_summary.json`) or cost-model predictions, marked as such.

Acronyms: DP (data parallel), KV (key/value), GDN (Gated DeltaNet), GIL (global interpreter lock), LPT (longest processing time first), req/s (requests per second).

## Files

| Path | Role |
|---|---|
| `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/tt/dispatch.py` | New. Pure Python, engine-agnostic: `CostModel`, `Policy`, `WorkerView`, `plan`, `share_cost_ms`, `merge_results`, `collect`. |
| `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/tests/test_dispatch.py` | New. 16 pure tests plus 2 server-level tests that skip until the patch below is applied. |
| `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/multichip/server_patch.diff` | Unified diff for `tt/server.py`. Not applied. Apply with `cd /home/hous/dev/kev/tt-metal && patch -p1 < models/autoports/jaredpalmer_kev_9b/doc/multichip/server_patch.diff`. |
| `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/scripts/parity_compare.py` | New. Compares the served answers of several `parity_remote.py` reports; exit 1 when any answer differs. |

`tt/server.py` and `tt/engine.py` are unchanged by this stage so far (stage 4 owns `tt/loader.py` and may add an import line to `tt/server.py`; the patch touches the import block one line below the `api` import, so apply it after stage 4 lands and resolve a one-line offset if needed).

## What already exists (stage 2 and 3)

One process opens `ttnn.open_mesh_device(MeshShape(2, 2))` and splits it into four `(1, 1)` submeshes (`Server.open_devices`); submesh `i` maps to physical chip `[1], [0], [2], [3]` (STATUS.md). One worker thread per submesh builds its own `KevEngine` and owns a per-worker prefix cache of 8 KV slots. A request goes whole to one worker: the worker whose cache holds the state, else the least loaded by queue depth. Measured on one chip (traced, stage 3): question tail at bucket 128 is 104.9 ms and op-count bound; 6 short questions 605 ms; a 2,392-token state costs 1,049 ms once, then 5 questions 526 ms; 1.6 req/s at 64 clients, flat across client counts because one worker serializes.

## Design

Two levels of parallelism, both in `dispatch.plan`:

1. Whole-request DP (throughput). Each request runs on one worker; four workers run four requests at once. This is the stage 2 and 3 policy with the load measured in estimated milliseconds instead of queue depth.
2. Question-level fan-out (latency). The rows of one request are independent causal rows over the same state (one row per question, no cross-question attention), so idle workers can each take a share of the questions. A worker that takes a share must hold the state in its prefix cache or prefill it first, so fan-out replicates the state prefill once per extra worker. The planner spends that replication only when it buys latency.

```
request thread                         worker 0..3 (one per submesh)
  encode -> rows, key
  views = [WorkerView(id, backlog_ms, cache)]
  plan(rows, views, key)  ---->  [(wid, [row idx]), ...]
  for each share: worker.submit(rows[idx], key, cost_ms)   -> queue -> prefill or hit, tails, head
  collect(parts) -> merge_results  <----  (probs, hit, ms) per share
  stats: latency_ms = max share ms, latency_ms_sum = sum
```

### Cost model (`CostModel`)

Filled by the server from `perf_summary.json` (`CostModel.load`, default path `doc/optimized/perf_summary.json`, override with `KEV_PERF_SUMMARY=<file>`; built-in constants when the file is missing). Per `engine_ms` entry it takes the `traced_policy` number, else `traced`, else `eager`.

| Constant | Source | Value |
|---|---|---|
| `tail_ms[bucket]` | `tail_bucket_<b>` | 128: 104.9, 256: 151.0, 512: 266.4, 1024: 476.4, 2048: 917.3 ms |
| `state_ms_per_block` | mean of `state_<n>` / (n // 128) over `state_2048` (56.18) and `state_2392` (58.26) | 57.22 ms per 128-token block |

Definitions: `S0 = (S // 128) * 128` is the cached prefix of a state of `S` tokens; the state cost of a miss is `state_ms_per_block * S0 / 128` (zero for a state under 128 tokens, which the cache does not hold); a question of `Q` tokens runs as a tail of `(S - S0) + Q` tokens in the smallest bucket that fits (largest bucket when none fits), at `tail_ms[bucket]`. These are the same shapes the traced engine runs (`doc/optimized/README.md`). The model is linear in blocks and ignores the per-question host work (under 1 ms per question measured in stage 3).

### Planner (`plan`)

Inputs: the request rows (`state_ids`, `question_ids` lengths), one `WorkerView(id, backlog_ms, cached_state_keys)` per worker, the state key. Output: `[(worker_id, [row indices])]`, indices ascending inside a share, shares ordered by worker id. Deterministic; about 15 us per call on this host for 6 questions over 4 workers (test asserts under 1 ms).

1. `offset[w] = backlog_ms[w] + (0 if w holds the key else state_cost)`.
2. Degrade rule (`Policy`): workers with `backlog_ms <= fanout_backlog_ms` (default 200 ms, `KEV_FANOUT_BACKLOG_MS`) are idle. When `KEV_FANOUT=0`, or the request has one question, or fewer than 2 workers are idle, the whole request goes to `argmin (offset[w] + sum of tails, id)`: least loaded in estimated ms with cache affinity (a cached worker wins until its backlog exceeds another worker's backlog plus the state cost).
3. Otherwise LPT greedy over the eligible workers (idle ones plus every cache holder): questions in decreasing tail cost, each to the worker with the smallest resulting finish time `offset + assigned tails + tail`, ties to the lowest id. The objective is the makespan, `max over sharing workers of (backlog + state cost if miss + sum of tails)`, which is the request's model latency.
4. Replication rule, applied when the state cost is positive: try to drop a miss worker from the plan (smallest share first, then highest id) and re-run the greedy without it; keep the drop when the makespan grows by at most the threshold. Threshold 0 for a short state (`S0 < short_state_tokens`, default 256: at most one block, 57 ms, cheap to replicate), the full state cost for a long state (a worker takes a long state only when its share saves more than the prefill it causes). Repeat until no miss worker can be dropped. A long state no worker holds is therefore prefilled by exactly one worker; a long state held by one worker stays there unless that worker's backlog exceeds the replication cost by enough to pay for a prefill elsewhere.

Predictions of the planner on the card cases, 4 idle workers (cost model, unverified on device; stage 3 single-chip measurements in brackets):

| Case | Plan | Predicted model latency |
|---|---|---|
| 2 questions, short state | 1 / 1 | 105 ms [202] |
| 6 questions, short state | 2 / 2 / 1 / 1 | 210 ms [605] |
| 5 questions, 370-token state, new | 2 / 2 / 1, state prefilled on 3 workers (114 ms each) | 416 ms [862] |
| 5 questions, 370-token state, cached on one worker | 3 on the holder / 2 on a fresh worker | 453 ms [677] |
| 5 questions, 2,392-token state, new | all on one worker | 1,554 ms [1,528] |
| 5 questions, 2,392-token state, cached | all on the holder | 525 ms [526] |

Under load (all backlogs above 200 ms) the planner is the stage 3 policy with millisecond loads: the 64-client test in `test_dispatch.py` assigns 64 short requests 16 / 16 / 16 / 16 with no fan-out.

### Result assembly (`collect`, `merge_results`)

Each share returns `(probs, hit, ms)` from `Worker._serve` as before. `collect` waits for every share and `merge_results` puts the probabilities back in request row order (which is question order). Stats:

- `latency_ms`: the maximum over the sharing workers of their model section for this request (`_serve` wall time: state prefill or cache hit, then the question tails and the head). It is the critical-path model time of the request and the number the card and the bench report ("model time per request" in the kev reference tables, where one GPU runs the whole request). It excludes queue wait.
- `latency_ms_sum`: the sum over shares, the device time the request consumed. Recorded in the server log line and in the `stats` returned by `Server.submit` / `predict`; the HTTP body keeps kev's field set (`model`, `answers`, `usage`, `latency_ms`).
- `prefix_cache_hit`: true only when every share hit. `worker` is the worker of the first row (kept for the existing test), `workers` lists all, `shares` gives rows, latency and hit per share.

Error path: the first share exception fails the request future; the other shares still run to completion on their workers (cancelling a queued share would make the worker thread's `set_result` raise) and their results are discarded. KV slots are released by the existing `_serve` code (a failed prefill returns its slot); `backlog_ms` is decremented in the worker's `finally`. `test_server_share_error_fails_request_and_releases_slots` checks that after a failing share every worker has `backlog_ms == 0`, no load, and `cached + free == cache_size`, and that the same request then succeeds with cache hits.

### Thread safety

`Server.dispatch` runs `plan` and the share submits under `Server.lock`, so two request threads cannot plan on the same snapshot of backlogs. `WorkerView.cached_state_keys` is the worker's live `OrderedDict`, used for membership only (no iteration, no copy); `backlog_ms` is a float updated under `Worker.counter`. Worker threads and queues are unchanged.

## Server patch summary (`server_patch.diff`, 155 lines)

- Import of `dispatch`.
- `Settings`: `fanout` (`KEV_FANOUT`, default 1), `fanout_backlog_ms` (`KEV_FANOUT_BACKLOG_MS`, default 200), `perf_summary` (`KEV_PERF_SUMMARY`, default empty = the stage 3 file).
- `Worker`: `backlog_ms` counter, `submit(rows, key, cost_ms=0.0)` carrying the estimate through the queue tuple and the `finally`, `view()`.
- `Server`: `cost_model`, `policy`, `fanouts`; `pick` replaced by `dispatch`; `submit` uses `collect` and the merged stats; the log line prints `workers`, `latency_ms`, `latency_ms_sum`.
- `/v1/models` card: per worker `backlog_ms`; new `dispatch` block (`fanout`, `fanout_backlog_ms`, `short_state_tokens`, `fanout_requests`, `cost_model`).

Verified on the host against a patched copy: `tests/test_server_api.py` 14 passed, `tests/test_dispatch.py` 18 passed (the two server-level tests run once the patch is in place).

## How to validate on device

Weight cache first. All four workers build their engines at the same time on four threads and read the same `TT_CACHE_PATH` tensor cache. The stage 3 cache (`/home/hous/dev/kev/tt_cache`, 17 GB) is warm for the current precision config. If stage 4 selects a different config (new cache root or new dtype file names), warm it once with a single worker before any 4-worker launch, so no two threads write the same cache file.

Common environment (every command below is run from `/home/hous/dev/kev/tt-metal`; the server through `devrun`, the clients through `hostrun`):

```bash
export HF_MODEL=/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/68c46c4b3498877f3ef123c856ecfde50c39f404
export KEV_RUN=/home/hous/.cache/huggingface/hub/models--jaredpalmer--kev-9b/snapshots/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0
export HF_HUB_OFFLINE=1 TT_CACHE_PATH=/home/hous/dev/kev/tt_cache KEV_MESH_SHAPE=2x2
SERVER="python -m uvicorn models.autoports.jaredpalmer_kev_9b.tt.server:app --host 127.0.0.1 --port 8008 --lifespan on"
BENCH="python models/autoports/jaredpalmer_kev_9b/scripts/serving_bench_remote.py --base-url http://127.0.0.1:8008 --reps 20 --concurrency 1,8,32,64"
```

Step 0, warm the weight cache (only when the cache root is new): `KEV_DEVICES=0 /home/hous/dev/kev/bin/devrun timeout 1800 $SERVER > /home/hous/dev/kev/logs/stage5_warm.log 2>&1`, wait for `Application startup complete`, then `kill -TERM` the python process (not the `timeout` wrapper; see the stage 3 shutdown note).

Step A, whole-request DP, throughput scaling and the GIL check. Run the server three times with `KEV_FANOUT=0` and `KEV_DEVICES=0`, `KEV_DEVICES=0,1`, unset (all four), and bench each:

```bash
KEV_FANOUT=0 KEV_DEVICES=0   /home/hous/dev/kev/bin/devrun timeout 7200 $SERVER > /home/hous/dev/kev/logs/stage5_server_A1.log 2>&1 &
/home/hous/dev/kev/bin/hostrun $BENCH --quick --label "P150 x1 (whole-request)" --out /home/hous/dev/kev/reports/bench/p150x1_whole
KEV_FANOUT=0 KEV_DEVICES=0,1 /home/hous/dev/kev/bin/devrun timeout 7200 $SERVER > /home/hous/dev/kev/logs/stage5_server_A2.log 2>&1 &
/home/hous/dev/kev/bin/hostrun $BENCH --quick --label "P150 x2 (whole-request)" --out /home/hous/dev/kev/reports/bench/p150x2_whole
KEV_FANOUT=0                 /home/hous/dev/kev/bin/devrun timeout 7200 $SERVER > /home/hous/dev/kev/logs/stage5_server_A4.log 2>&1 &
/home/hous/dev/kev/bin/hostrun $BENCH --label "P150 x4 (whole-request)" --out /home/hous/dev/kev/reports/bench/p150x4_whole
```

Stop each server with `kill -TERM` before the next launch (one device-owning process at a time). Expected if the host does not serialize: 1.6, 3.2, 6.4 req/s at 64 clients on the short case, 1.9 / 3.8 / 7.6 on the long case, 7.6 / 15 / 30 on decision-v7, with the single-client latency columns unchanged (605 / 605 and 1,528 / 526). A shortfall from linear is the GIL and host-side cost: each worker's Python thread does the per-segment `from_torch` writes, the trace replays and the `to_torch` readback, and a binding that holds the GIL while it waits on the device stalls the other three workers. If x2 scales and x4 does not, the readback wait is the first suspect; a remedy is outside dispatch (release the GIL around the blocking read or move the per-worker loop to a subprocess, which the cluster lock currently forbids). The 4-worker run uses the full bench (256 / 256 / 64 requests per level) because the server is fast enough now; the 1- and 2-worker runs keep `--quick` to fit the time budget.

Step B, fan-out latency. Same four-chip server with the default `KEV_FANOUT=1`:

```bash
/home/hous/dev/kev/bin/devrun timeout 7200 $SERVER > /home/hous/dev/kev/logs/stage5_server_B.log 2>&1 &
/home/hous/dev/kev/bin/hostrun $BENCH --label "P150 x4 (fan-out)" --out /home/hous/dev/kev/reports/bench/p150x4_fanout
curl -s localhost:8008/v1/models | python -c 'import json,sys; c=json.load(sys.stdin)["models"][0]; print(c["dispatch"], [ (w["id"], w["requests"], w["hits"], w["misses"]) for w in c["workers"]])'
```

Checks: the latency columns against the prediction table above (6 short questions near 210 ms, 2 questions near 105 ms, the long state unchanged), `dispatch.fanout_requests` greater than zero after the latency section, and the 64-client throughput within noise of step A (the degrade rule must keep whole-request assignment under load; if throughput drops, lower `KEV_FANOUT_BACKLOG_MS`). The server log carries `latency_ms_sum` per request for the device-time cost of fan-out.

Step C, identical answers per chip on the 16 reference records. One single-worker server per chip, one parity pass each, then the four-worker fan-out server with two passes, then compare every report to the stage 3 parity file:

```bash
for k in 0 1 2 3; do
  KEV_DEVICES=$k /home/hous/dev/kev/bin/devrun timeout 1800 $SERVER > /home/hous/dev/kev/logs/stage5_server_chip$k.log 2>&1 &
  # wait for "Application startup complete"
  /home/hous/dev/kev/bin/hostrun python models/autoports/jaredpalmer_kev_9b/scripts/parity_remote.py --passes 1 --out /home/hous/dev/kev/reports/stage5_parity_chip$k.json
  # kill -TERM the python server process
done
/home/hous/dev/kev/bin/devrun timeout 1800 $SERVER > /home/hous/dev/kev/logs/stage5_server_parity4.log 2>&1 &
/home/hous/dev/kev/bin/hostrun python models/autoports/jaredpalmer_kev_9b/scripts/parity_remote.py --passes 2 --out /home/hous/dev/kev/reports/stage5_parity_x4.json
/home/hous/dev/kev/bin/hostrun python models/autoports/jaredpalmer_kev_9b/scripts/parity_compare.py /home/hous/dev/kev/reports/stage3_parity.json /home/hous/dev/kev/reports/stage5_parity_chip{0,1,2,3}.json /home/hous/dev/kev/reports/stage5_parity_x4.json --out /home/hous/dev/kev/reports/stage5_parity_compare.json
```

Pass criterion: `all_identical true` (29 served answers equal across the four chips, the fan-out server and stage 3) and `revisits.answers_equal_to_first_pass 16/16` in the x4 report. `KEV_DEVICES=k` selects submesh `k`, which is physical chip `[1], [0], [2], [3]` for `k = 0..3`.

## Stage 6 inputs this stage produces

The card table rows `P150 x4 (data parallel)` come from `/home/hous/dev/kev/reports/bench/p150x4_fanout/report.json` (`card_row`), with the footnote that `latency_ms` is the model time of the request's critical path over its workers (max over shares), new / cached state, and the throughput from the same report at 64 clients. Keep `/home/hous/dev/kev/reports/bench/p150x4_whole/report.json` next to it to show the fan-out cost in device time (`latency_ms_sum` in the server log) and that the 64-client throughput is the same in both modes.

## Open questions

- GIL scaling is unmeasured (step A). The design assumes device time dominates and the four Python threads overlap; stage 3 measured host work under 1 % per request on one worker, which says nothing about a GIL-held device wait.
- `backlog_ms` is an estimate that does not decay while a job runs; a long in-flight job counts at its full cost until it finishes. Under steady load this biases towards whole-request assignment, which is the safe side.
- The cost model comes from chip 0 at the stage 3 precision config. If stage 4 changes the matmul dtypes or fidelity, the tail and state numbers shift together; the planner's decisions depend on ratios (tail versus state block), so re-fitting is a correctness nicety, not a blocker. Point `KEV_PERF_SUMMARY` at a refreshed file when one exists.
- Four engines build concurrently at startup: four `load_state_dict` calls and four tensor-cache reads at once. Host RAM is 249 GB (fine); startup time with four concurrent builds is unmeasured (44 s for one).
- `latency_ms_sum` is not in the HTTP body to keep kev's field set; adding it behind a flag is a one-line change in `Server.body` if stage 6 wants it client-side.
- `KEV_DEVICES` subsetting, `trace_region_size` per submesh (1 GiB each) and 8 KV slots per chip were set up in stage 2 and 3 but have not run with four live workers yet.

# Stage 5 (data parallel over 4 chips): work log

Date: 2026 Oct 01 23:11 to Oct 02 (times below are 24-hour UTC, the host clock, as the logs print them). Box p300c, four Blackhole P150 chips in one process: `ttnn.open_mesh_device(MeshShape(2, 2))`, `create_submeshes(MeshShape(1, 1))`, one `KevEngine` and one worker thread per submesh (`KEV_MESH_SHAPE=2x2`). Submesh `i` is physical chip `[1], [0], [2], [3]`; the server names a worker by its mesh id (1 to 4). Design and validation plan: `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/multichip/README.md`. Reports under `/home/hous/dev/kev/reports`, logs under `/home/hous/dev/kev/logs` (`stage5_server_<tag>.log`, `stage5_bench_<tag>.log`, `stage5_driver.log`).

Acronyms: DP (data parallel), GIL (global interpreter lock), KV (key/value), req/s (requests per second), p50 / p90 / p99 (percentiles), LRU (least recently used).

## Environment

- Code: tt-metal worktree `/home/hous/dev/kev/tt-metal`, branch `hous/kev-9b-bringup`, HEAD `ed917633df8` at the end of the stage (base `7eac776e926`, origin/main), clean tree; the measurements ran at `9a06a5bed76` plus the one line in `tt/engine.py` described under "Fix" below, which `ed917633df8` (02:41 UTC) committed unchanged. HEAD moved from `0591d956196` to `9a06a5bed76` at 00:54 while this stage ran (the stage 4 agent's commit; it also swept in this stage's new scripts `scripts/parity_sample.py` and `scripts/final_numbers.py`). The 1-worker and 2-worker servers below started before that commit; the only code difference is `engine.py` setting `args.prefill_progcfg = None` under the matmul policy in place of the `QWEN9B_MLP_DOWN_AUTO` environment knob, which is the same program choice.
- Precision: `KEV_PRECISION=selected` (`mlp_bfp8`: bfp8 gate / up / down and projections, LoFi with fp32 accumulation, bf16 KV cache, `KEV_MATMUL_POLICY=1`), weight cache `/home/hous/dev/kev/tt_cache/P150/tensor_cache_bfp8_kev_2b2a70cf_gu-bfp8_dn-bfp8_pj-bfp8`, warm. Traced engine (`KEV_TRACED=1`), 8 KV slots per worker, `KEV_MAX_STATE=65536`, trace region 1 GiB per submesh.
- Server: `python -m uvicorn models.autoports.jaredpalmer_kev_9b.tt.server:app --host 127.0.0.1 --port 8008 --lifespan on` through `/home/hous/dev/kev/bin/devrun timeout 21600` with `HF_MODEL`, `KEV_RUN`, `HF_HUB_OFFLINE=1`, `TT_CACHE_PATH=/home/hous/dev/kev/tt_cache`, `KEV_MESH_SHAPE=2x2`, and per run `KEV_DEVICES` and `KEV_FANOUT`. Clients through `/home/hous/dev/kev/bin/hostrun`. Stopping: `kill -TERM` to the uvicorn python process (not to `flock` or `timeout`: a TERM to `flock` leaves the children running with the lock), then wait for the lock.
- Startup, from the server's `starting:` line to `ready:`: 1 worker 59 s (39 s once the cache files were in the page cache), 2 workers 59 s, 4 workers 96 s (four concurrent engine builds from the shared tensor cache; host memory in use about 34 GB).
- A CPU job for stage 6 (kev's fp32 `LocalPredictor` on 64 records, `nice -n 10`, 8 torch threads) ran on the host from 23:14 to 23:47. It slowed the server's host-side work by 10 to 15 % (see A1 below); every number in the tables was measured after it finished, or re-measured.

## A. Throughput scaling, whole-request DP (`KEV_FANOUT=0`)

`scripts/serving_bench_remote.py --reps 20 --concurrency 1,8,32,64`; `--quick` is 32 / 32 / 8 requests per level (short / decision-v7 / long), full is 256 / 256 / 64. Requests/s at 64 clients, p50 / p99 wall time per request in ms.

| Workers | Method | 6 questions, new short state | decision-v7 development | 5 questions, new 2,200-token state | Report |
|---|---|---|---|---|---|
| 1 (`KEV_DEVICES=0`) | full | 1.6 req/s, 39,116 / 39,504 | 5.7 req/s, 10,820 / 14,335 | 0.7 req/s, 50,093 / 96,897 | `reports/bench/p150x1_whole` |
| 2 (`KEV_DEVICES=0,1`) | quick | 2.0 req/s, 8,517 / 15,583 | 1.8 req/s at 32 clients (see note) | 2.2 req/s, 2,124 / 3,206 | `reports/bench/p150x2_whole` |
| 4 | full | 2.5 req/s, 24,872 / 26,292 | 7.2 req/s, 8,568 / 12,084 | 0.9 req/s, 41,815 / 68,732 | `reports/bench/p150x4_whole` |
| 4, readback fix | quick | 6.6 req/s, 2,735 / 4,858 | 20.9 req/s, 549 / 1,374 | 5.0 req/s, 799 / 1,063 | `reports/bench/p150x4_whole_fix_quick` |
| 4, readback fix | full | 6.5 req/s, 9,716 / 9,940 | 15.6 req/s, 3,580 / 5,893 | 2.6 req/s, 13,049 / 24,549 | `reports/bench/p150x4_whole_fix` |

Notes. The 2-worker decision-v7 column is the quick run's 64-client level (1.8 req/s; the quick decision-v7 sample has 32 requests). The long case differs by method: the full bench sends 64 distinct 2,200-token states through 8 KV slots per worker, so almost every request misses (one worker: 0.7 req/s at a 1.5 s prefill each; stage 4's `--quick` value of 1.9 req/s was 8 states that all hit after the first pass). Single-client latency columns were unchanged in every run (607 / 607 and 1,533 / 527 ms, see B).

Scaling before the fix was 1.25x for 2 workers and 1.56x for 4 (expected 2x and 4x). Where the time went:

1. Per-request model time under load, from the server log (`latency_ms` is `Worker._serve` wall time, which excludes queue wait). One worker alone: 607 ms per 6-question request at every client count. Two workers with the 32-request level split 16 / 16: median 1,058 ms per request on each worker (1.74x). Four workers split 64 / 64 / 64 / 64: median 1,522 ms (2.5x); the first level of the 4-worker run that stayed on one worker was 607 ms again. The device time per request is unchanged (same traces), so each worker thread waited on the others.
2. `py-spy` (0.4.2, `sudo py-spy record --gil --threads --nonblocking --rate 200 --duration 30`, during the 64-client short-state run of the 4-worker server, `reports/bench/gil_A4/gil_012352.raw`): 5,969 of a possible 6,000 samples found a thread holding the GIL (99.5 % of wall time), split evenly over the four worker threads (23 to 27 % each); 98.6 % of the GIL-held samples were in `KevEngine._gather` at `ttnn.to_torch` (`ttnn/decorators.py:1195`, the native call). The on-CPU profile taken right after (`all_012421.raw`, no `--idle`) has 182 samples in 30 s: the process is off-CPU 97 % of the time. The 32 periodic `py-spy dump`s over the run (`dumps.txt`) agree: of 79 busy worker-thread samples, 30 sat in the `_gather` readback.
3. Cause in the bindings: `ttnn.to_torch` calls `ttnn.from_device` for a device tensor; the module function `from_device` (`ttnn/cpp/ttnn-nanobind/operations/core.cpp:165`) is bound without `nb::gil_scoped_release` and its body is `tensor.cpu(blocking, queue_id)` (`ttnn/cpp/ttnn/operations/core/core.cpp:55`), which blocks until the forward and gather traces finish. The method `Tensor.cpu` (`ttnn/cpp/ttnn-nanobind/pytensor.cpp`, `.def("cpu", ...)`) is bound with `nb::call_guard<nb::gil_scoped_release>()`. `execute_trace` also releases the GIL, and the engine replays with `blocking=False`, so the whole device wait of a question happened inside `from_device` with the GIL held, and the other workers could not even enqueue their next segment.

Fix (one line, `tt/engine.py` `_gather`): `ttnn.to_torch(B["rows"].cpu())` in place of `ttnn.to_torch(B["rows"])`. Same copy (`from_device` is `tensor.cpu`), GIL released during the wait. Result (4 workers, `--quick`, `reports/bench/p150x4_whole_fix_quick`): 6.4 / 6.6 / 6.6 req/s at 8 / 32 / 64 clients on the short case (4.1x of one worker), 22.4 / 22.5 / 20.9 req/s on decision-v7 (3.7 to 3.9x of 5.7), 5.0 req/s on the long case (cache hits in the quick sample); per-request model time under 4-way load back to 607 ms median (p90 610 to 619) with the split 8 / 8 / 8 / 8 at every level. The GIL profile during that run's 64-client section (`reports/bench/gil_A4fix/gil_014507.raw`): 58 GIL-held samples in 20 s at 200 Hz (1.5 % of wall time), half of them the uvicorn main thread. Every later run in this log uses the fix. Answers are unaffected (C and the x4 parity below ran with it).

## B. Fan-out latency (4 workers, `KEV_FANOUT=1`, readback fix)

`reports/bench/p150x4_fanout` (full bench, latency section at 01:48 to 01:52). Model latency per request in ms (`latency_ms` = max over the sharing workers), new / cached state; predictions from the README cost model; `latency_ms_sum` is the device time the request consumed, from the server log.

| Case | Plan seen in the log | Measured new / cached | Predicted | Device time (sum) | Whole-request (A, 4 workers) |
|---|---|---|---|---|---|
| 2 questions, short state | 1 / 1 on workers 0, 1 | 101.9 / 101.4 | 105 | 204 | 202 / 202 |
| 6 questions, short state | 2 / 2 / 1 / 1 | 203.9 / 203.9 | 210 | 613 | 607.6 / 607.4 |
| 5 questions, 370-token state | new: 2 / 2 / 1 on 3 workers; cached: 3 on the holder, 2 on a second | 439.7 / 395.2 | 416 / 453 | 1,172 / 689 | 878 / 687 |
| 5 questions, 2,200-token state | all on one worker | 1,533.4 / 527.0 | 1,554 / 525 | 1,533 / 527 | 1,533.6 / 526.7 |

`dispatch.fanout_requests` was above zero after the latency section (every short request fanned out). The 370-token cached case beats its prediction because the two-worker split (3 on the holder, 2 on a fresh worker that prefills three blocks) finishes in 395 ms, under the planner's conservative 453 ms estimate.

## C. Per-chip identity, 16 reference records

One single-worker server per submesh (`KEV_DEVICES=k`, `KEV_FANOUT=0`), one pass of `scripts/parity_remote.py` each (reports `stage5_parity_chip0..3.json`, 29 served answers each), compared with `scripts/parity_compare.py` against the stage 4 single-chip report `/home/hous/dev/kev/reports/stage4r_parity.json`:

| Server | Mesh id (physical chip) | vs fp32: max dp / mean dp / flips | identical to stage 4 |
|---|---|---|---|
| chip0 | 1 ([1]) | 0.0878 / 0.0272 / 1 | yes |
| chip1 | 2 ([0]) | 0.0878 / 0.0272 / 1 | yes |
| chip2 | 3 ([2]) | 0.0878 / 0.0272 / 1 | yes |
| chip3 | 4 ([3]) | 0.0878 / 0.0272 / 1 | yes |

`all_identical true` (`reports/stage5_parity_compare_chips.json`). The one flip is the known near-tie (record 0 `choice`, fp32 top-2 margin 0.0315). All four passes ran before the readback fix; the x4 fan-out parity below ran with it and is compared to the same base.

## D. Full bench of the final configuration (4 workers, fan-out on, readback fix)

`reports/bench/p150x4_fanout/report.json`, 01:48 to 02:04, full method (256 / 256 / 64 requests per level, two passes, the second timed), `--reps 20`. The one-chip row is `reports/bench/p150x1_whole/report.json` (1 worker, `KEV_FANOUT=0`, same method, 00:04 to 00:48, idle host).

| Row | 6 questions, short state new / cached | 5 questions, 2,200-token state new / cached | req/s at 64 clients (short case) |
|---|---|---|---|
| P150 (1 chip) | 607.8 / 607.7 ms | 1,533.5 / 527.0 ms | 1.6 |
| P150 x4 (data parallel, fan-out) | 203.9 / 203.9 ms | 1,533.4 / 527.0 ms | 6.5 |

Throughput table of the x4 fan-out server, requests/s and p50 / p99 wall ms per request:

| Sample | 1 client | 8 clients | 32 clients | 64 clients |
|---|---|---|---|---|
| 6 questions, new short state | 4.8, 207 / 212 | 6.5, 1,216 / 1,319 | 6.5, 4,859 / 4,970 | 6.5, 9,723 / 9,944 |
| decision-v7 development | 6.2, 104 / 477 | 16.6, 406 / 1,143 | 15.5, 1,939 / 3,307 | 15.6, 3,620 / 6,157 |
| 5 questions, new 2,200-token state | 0.9, 1,538 / 1,544 | 2.6, 3,068 / 3,082 | 2.6, 12,269 / 12,294 | 2.5, 13,054 / 24,666 |

Same table for one chip (`p150x1_whole`): short 1.6 req/s at every level (p50 611 / 4,884 / 19,573 / 39,116 ms); decision-v7 5.6 / 5.7 / 5.7 / 5.7 req/s (p50 105 / 1,368 / 5,077 / 10,820); long 0.6 / 0.7 / 0.7 / 0.7 req/s (p50 1,539 / 12,283 / 44,683 / 50,093).

Fan-out under load. The server log for the decision-v7 section shows the degrade rule working: in the loaded blocks (8 to 64 clients) 0 to 11 of 256 requests fanned out (mean 1.0 to 1.1 shares per request), against 107 of 256 at 1 client. The short-case throughput at 64 clients is the same with fan-out on (6.5) and off (6.6, quick, `p150x4_whole_fix_quick`), and the fan-out server's counters after the bench read `fanout_requests` 719 against a per-worker total of about 7,100, where the per-worker `requests` counters count worker shares (one per worker that served part of a request, warm-ups included), not requests; counted from the request lines of `stage5_server_B.log` up to the 02:04 parity run: 4,806 requests, 722 fanned out, 6,704 worker shares. The whole-request full run with the fix (`reports/bench/p150x4_whole_fix`, 02:20 to 02:40, fresh server) gives the same 64-client throughput as the fan-out run on every sample: short 6.5 against 6.5 req/s, decision-v7 15.6 against 15.6, long 2.6 against 2.5, with the same latency columns (606.9 / 607.0 and 1,533.7 / 526.5 ms). Fan-out therefore costs no throughput; the higher quick-method decision-v7 figure (20.9 req/s) was the quick sample's 32 states hitting the 32 KV slots, where the full sample's 256 states mostly miss.

Worker split of the fan-out bench: worker shares 1,619 / 1,542 / 1,486 / 2,038, hits 100 / 74 / 32 / 28 (`/v1/models` after the bench). Zero tracebacks and zero HTTP 500s in the server log at 64 clients in every run of this stage.

## Parity through the fan-out server

16 reference records, `scripts/parity_remote.py --passes 2` on the x4 fan-out server (`reports/stage5_parity_x4.json`, 02:04): vs fp32 max dp 0.0878, mean dp 0.0272, 1 flip (the near-tie record 0 `choice`, margin 0.0315), vs bf16 max dp 0.0875; revisits 16 / 16 equal to the first pass with the states spread over the four workers' caches (32 cached states at the end); latency median 203 ms, max 1,533 ms. `scripts/parity_compare.py` over stage 4, the four single-chip reports and the x4 report: `all_identical true` (`reports/stage5_parity_compare.json`, 29 served answers each).

64 development records (stage 6 F): `scripts/parity_sample.py` (`reports/parity64/`: `records.jsonl`, 16 records at indices 4 to 19 of each of hard-v1, devtools-v1, documents-v1 development, plus the 16 reference records; `cpu_fp32.json` from kev's `LocalPredictor(device="cpu")` in fp32 with `SERVING_CONTEXT`, 33 min of scoring on 8 threads, bit-identical to `reports/reference/probs_fp32.json` on the 16 shared records; `server_x4.json` from the fan-out server; `compare.json`). 104 questions: max dp 0.416, mean dp 0.0368, 4 argmax flips, of which 2 at an fp32 top-2 margin >= 0.05 and 2 near ties.

| Source | questions | max dp | mean dp | flips |
|---|---|---|---|---|
| hard-v1 development 4 to 19 | 27 | 0.4160 | 0.0663 | 1 (record 0 `date`, margin 0.194) |
| devtools-v1 development 4 to 19 | 19 | 0.0692 | 0.0321 | 1 (record 19 `action`, margin 0.088) |
| documents-v1 development 4 to 19 | 29 | 0.1322 | 0.0220 | 1 near tie (margin 0.015) |
| reference 16 | 29 | 0.0878 | 0.0272 | 1 near tie (margin 0.0315) |

The large deviations are all hard-v1 reasoning rows near the decision boundary: record 0 (a date-arithmetic memo, 132 tokens, correct option `a`) has fp32 at 0.45 on `d` and the served path at 0.66 on `c` (whether either path answers the label says nothing about their agreement; the control is the paired comparison below); records 3 and 13 (spec sheets labelled `none_qualifies`, dp 0.18 and 0.16) move probability toward the label. Median per-record max dp over the 64 is 0.031. For scale, kev's own bf16 serving path on an H100 is max dp 0.0217 and mean 0.0014 against fp32 on 200 decision-v7 records (`runs/serve-9b-h100/report.json` block `eager_vs_fp32` in the kev repository, `/home/hous/dev/kev/reports/kev_internals.md` section 7; the model card text gives the same comparison as "within 0.017 of the fp32 evaluation path on 280 questions", a different run); the bfp8 / LoFi device path is an order of magnitude further from fp32 in mean dp, and the stage 6 evals measure what that costs in accuracy.

Control (stage review, `/home/hous/dev/kev/reports/review_stage5_6.md`, "Required Work", fourth item, CPU only): the TT eval rows paired on `(id, question)` with kev's own fp32 rows for this checkpoint (`runs/r18-9b-joint-lr2e5-{hard,devtools,docs}/rows.json` for development, `runs/r27c-9b-cand-{hardtest,devtest,docs1test}/rows.json` for test, in the kev repository) on 6,172 clean knowable questions: mean |dp| 0.025 to 0.048 per read, flips two-sided (TT right / fp32 wrong against fp32 right / TT wrong: 20 / 25, 16 / 30, 6 / 12 on the development reads, 16 / 24, 14 / 18, 2 / 5 on the test reads), net -0.3 to -1.3 pp per read. The 0.416 row (`hard-v1/temporal_numeric/development/00048`) reproduces at 0.415 against kev's GPU fp32 rows; the largest paired row (`hard-v1/temporal_numeric/development/00044`, |dp| 0.68) favours the TT path. The full table is in `doc/benchmark/README.md`, "Parity against fp32".

## E. Stage 6 evaluation on the fan-out server

`scripts/run_eval.sh --base-url http://127.0.0.1:8008 --concurrency 4` (development splits, 02:04 to 02:11), then with `--test` (test splits once, 02:11 to 02:18), outputs under `/home/hous/dev/kev/reports/eval/<suite>/<split>/` (`report.json`, `rows.json`, `predictions.jsonl`), logs `/home/hous/dev/kev/logs/stage6_eval_dev.log` and `stage6_eval_test.log`. breadth-v1 printed `NOT-AVAILABLE` (private mirror) on both splits. The stage 2 one-chip smoke-v1 report was moved to `reports/eval/smoke-v1/development.stage2_1chip` so the suite re-ran on the four chips. `scripts/summarize_eval.py` table (kev `clean` metrics; latency is the client wall time at concurrency 4):

| Suite / split | n | acc | Brier | ECE | p50 / p95 ms | model card |
|---|---|---|---|---|---|---|
| hard-v1 / development | 1,083 | 0.808 | 0.272 | 0.057 | 352 / 2,270 | |
| devtools-v1 / development | 1,072 | 0.759 | 0.342 | 0.083 | 302 / 862 | |
| documents-v1 / development | 920 | 0.896 | 0.150 | 0.018 | 774 / 1,836 | 0.902 |
| smoke-v1 / development | 18 | 0.889 | 0.164 | 0.123 | 207 / 540 | |
| hard-v1 + devtools-v1 audited / development | 1,855 | 0.811 | 0.275 | 0.071 | | 0.821 |
| hard-v1 / test | 1,088 | 0.826 | 0.244 | 0.053 | 393 / 2,261 | 0.834 / ECE 0.054 |
| devtools-v1 / test | 1,071 | 0.787 | 0.319 | 0.101 | 296 / 884 | 0.791 / 0.098 |
| documents-v1 / test | 936 | 0.896 | 0.157 | 0.015 | 764 / 1,484 | 0.900 / 0.017 |
| smoke-v1 / test | 18 | 0.944 | 0.127 | 0.187 | 208 / 561 | |
| hard-v1 + devtools-v1 audited / test | 1,859 | 0.814 | 0.262 | 0.059 | | 0.822 |

Table regenerated after the stage review with the card's `drop_ids` (`experiments/rounds/r27.json`: `codereviewer/cls-test/13657`, `codereviewer/cls-test/19245`) applied in `scripts/summarize_eval.py`; the first version had n 1,074 / 1,073 for devtools-v1 and 1,857 / 1,861 pooled, accuracies at most 0.05 pp higher.

No record was rejected or truncated. The fan-out server's counters after the bench, both parity passes and all evals: 9,286 requests (9,290 request lines with the four warm-ups), 1,373 fanned out, 11,847 worker shares (11,851 with the warm-ups; the per-worker counters 2,824 / 2,770 / 2,786 / 3,471 count shares, not requests), zero tracebacks and zero HTTP 500s in `stage5_server_B.log`. The final table in the model-card format, with footnotes, is `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/benchmark/README.md`; the machine-readable copy is `/home/hous/dev/kev/reports/final_numbers.json`.

## G. Shutdown

Every server was stopped with `kill -TERM` to the uvicorn python process and shut down cleanly (`Application shutdown complete`, exit 0 of the python process; the `timeout` and `flock` wrappers report 143). After the last server (whole-request full run, 02:39:43): no `uvicorn models.autoports` process, `flock -n /home/hous/dev/kev/.device.lock true` succeeds (lock free). Device-side runs of this stage: 1 + 1 + 3 + 1 + 1 + 1 + 1 = 9 server launches, all through `devrun`; the stage 6 CPU job and all clients through `hostrun` or the kev venv.

## Open items

- The `engine.py` readback line is committed as `ed917633df8` (it was uncommitted while this stage ran); it is the difference between 2.5 and 6.5 req/s on four chips. The underlying issue is the `ttnn.from_device` binding (no `gil_scoped_release`), which a one-line tt-metal change would fix for every multi-threaded user.
- Cache-affinity skew under whole-request dispatch (decision-v7 and long samples land 2 to 3 times more often on worker 0) does not cost throughput while the GIL is free, but the backlog estimate is a static cost model and the tie rule prefers worker 0; a live per-worker busy measurement would balance it.
- Long states: 8 KV slots per chip bound the long-state throughput (2.6 req/s on four chips for 64 distinct 2,200-token states). More slots cost 2.06 GiB of DRAM each at `KEV_MAX_STATE=65536`; a smaller `KEV_MAX_STATE` buys more slots.
- The 64-record parity shows hard-v1 reasoning rows moving by up to 0.42 in probability under bfp8 / LoFi; the evals put the cost at 0.3 to 0.8 accuracy points on the test splits. A precision step up (stage 4 sweep rows) is the lever.

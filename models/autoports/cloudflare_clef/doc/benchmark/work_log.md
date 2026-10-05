# Stage 5 work log (evaluation)

Times are ET (UTC-4); the host clock is UTC. Host side only until the device agent appends its entries.

## 2026 Oct 05, 13:58 ET: host preparation (no device)

Written (all under `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/`): `scripts/eval_remote.py` (SystemOne JSONL to a running server: concurrency, retries, resume, per-request `latency_ms`, CPU-reference row format), `scripts/run_eval.sh` (the ordered driver: ARC-Challenge test, BANKING77 test, New Yorker matching test, the three 100-item samples, optional `parity` step, the Kev suites through `kev.benchmark --remote`, then the summary), `scripts/summarize_eval.py` (`eval_metrics.score` per file, `parity_compare.summarize` of the TT samples against the CPU controls, the Kev `clean` metrics, `/home/hous/dev/clef/reports/final_numbers.json`, `doc/benchmark/EVAL.md`), `doc/benchmark/README.md` (links `EVAL.md`; the stage 3 agent adds the performance table), `doc/benchmark/EVAL.md` (generated with no results: method, the model-card table with `not run`, the wall-time estimate).

Kev client check (`/home/hous/dev/kev/kev`, `kev/benchmark.py` lines 171 to 210, `kev/predictors.py` `RemotePredictor`): `--remote-model` defaults to `kev-latest` and is only echoed into the request `model` field; the Clef server accepts any `model` string (`tt/api.py` `SystemOneRequest.model: str`, no check against `CLEF_MODEL_NAMES`), so `--remote-model clef` is used. `--suite evals/<suite> --allow-test` loads the test split with the manifest sha256 check; `--data FILE` loads a plain labelled JSONL (`kev.data.load_records`, needs `label` on every question, which the suite files carry), used for the 20-record slice. `uv run` in that directory prints a `VIRTUAL_ENV` mismatch warning when run under `hostrun`; `run_eval.sh` runs it with `env -u VIRTUAL_ENV`. 422 from the server is a refused record (`REFUSAL_STATUSES`), which `kev.benchmark` counts in `coverage.rejected_records` and stops on.

## 14:00 ET: fake-server test (host, port 8009)

Server: `cd /home/hous/dev/clef/tt-metal && nohup env CLEF_FAKE_ENGINE=1 CLEF_TRACED=0 OMP_NUM_THREADS=4 /home/hous/dev/clef/bin/hostrun python -m uvicorn models.autoports.cloudflare_clef.tt.server:app --port 8009 > <scratchpad>/fake_server_8009.log 2>&1 &` (port 8009; the stage 3 real server on 8008 was not touched). Startup to `Application startup complete` under 20 s.

Runner:

```
S=/tmp/claude-1002/-home-hous-dev-clef/2a64a29f-04fe-4cd3-b935-9017907901d0/scratchpad/eval_fake
bash models/autoports/cloudflare_clef/scripts/run_eval.sh --base-url http://127.0.0.1:8009 --out-root $S --log-dir $S/logs --log-prefix fake5 --concurrency 4 --steps "samples kev summarize" --kev-suites hard-v1 --kev-records 20 --final-json $S/final_numbers.json --doc $S/EVAL.md
```

Output (abridged):

```
server http://127.0.0.1:8009: backend=fake device=fake x1 precision={... all None: the fake server does not apply precision_defaults}
SUMMARY /home/hous/dev/clef/evals/arc_challenge_test_sample100.jsonl: {"records": 100, "ok": 100, "failed": 0, "sent_this_run": 100, "run_wall_s": 4.9, "records_per_s_this_run": 20.438, "latency_ms_median": 44.5, "latency_ms_p95": 70.1, "latency_ms_max": 75.0, "wall_ms_median": 191.8, "tokens_median": 200.0, "tokens_max": 274, "errors": []}
SUMMARY /home/hous/dev/clef/evals/banking77_test_sample100.jsonl: {"records": 100, "ok": 100, "failed": 0, "run_wall_s": 29.3, "records_per_s_this_run": 3.416, "latency_ms_median": 297.7, "latency_ms_p95": 317.2, "tokens_median": 1831.0, "tokens_max": 1871, "errors": []}
SUMMARY /home/hous/dev/clef/evals/newyorker_matching_test_sample100.jsonl: {"records": 100, "ok": 100, "failed": 0, "run_wall_s": 10.3, "records_per_s_this_run": 9.679, "latency_ms_median": 90.85, "latency_ms_p95": 135.1, "tokens_median": 499.5, "tokens_max": 791, "errors": []}
OK arc_challenge_test_sample100 / OK banking77_test_sample100 / OK newyorker_matching_test_sample100
RUN kev hard-v1: first 20 test records ($S/kev/hard-v1/test_head20.jsonl) -> $S/kev/hard-v1/test
  "acc": 0.3870967741935484 ... "coverage": {"requested_records": 20, "requested_questions": 31, "evaluated_records": 20, "evaluated_questions": 31, "rejected_records": 0, "truncated_records": 0}
OK kev hard-v1
RUN summarize_eval.py
ARC-Challenge test            accuracy  full not run  sample TT 19.0  CPU 95.0  gap -76.00 pp  card 97.7  parity max dp 0.9555 mean 0.7132 flips@margin 79 near-tie 0  latency p50 44 ms p95 70 ms wall 5 s
BANKING77 test                macro_f1  full not run  sample TT  0.0  CPU 91.1  gap -91.13 pp  card 94.2  parity max dp 0.9895 mean 0.9087 flips@margin 97 near-tie 3  latency p50 298 ms p95 317 ms wall 29 s
New Yorker caption matching   accuracy  full not run  sample TT 19.0  CPU 60.0  gap -41.00 pp  card 69.5  parity max dp 0.9455 mean 0.5309 flips@margin 77 near-tie 3  latency p50 91 ms p95 135 ms wall 10 s
kev hard-v1/test: n=31 acc 0.387 brier 0.751 ece 0.209 coverage 20/20 rejected 0 latency p50 322 ms p95 816 ms
coverage ok: False ['kev hard-v1: 20 of 700 test records evaluated, 0 rejected, 0 truncated']; findings: [three sample gaps above 1.0 pp]
wall-time estimate: 60 min to 93 min for 7254 requests
RUN_EVAL_DONE failures=0
```

The fake engine returns seeded random hidden rows, so its accuracies (19 percent on ARC, 0 macro-F1) and the parity flips are meaningless and the "findings" are the summary doing its job on random data; the coverage issue is the 20-record slice against the 700-record suite. The run showed: 300 sample requests with images and 1.8k-token schemas served and written in the reference row format with `latency_ms`, `wall_ms`, `correct`; resume files written; the kev client scoring the Clef server with `--remote-model clef` and `--data`; the summary scoring, comparing, reading the Kev report and writing `final_numbers.json`, `EVAL.md` and `metrics/`. Files: `$S/*.jsonl`, `$S/*.summary.json`, `$S/kev/hard-v1/test/{report.json,rows.json,predictions.jsonl}`, `$S/metrics/parity_*`, `$S/final_numbers.json`, `$S/EVAL.md`, logs `$S/logs/fake5_*.log`.

Fake server stopped at 14:03 ET (`Application shutdown complete`, port 8009 free). The real server on 8008 (stage 3 device agent) kept running.

## 14:04 ET: initial EVAL.md

`hostrun python scripts/summarize_eval.py --out-root /home/hous/dev/clef/reports/eval --final-json <scratchpad>/final_numbers_skeleton.json --doc doc/benchmark/EVAL.md --write`: with no results under the real out root, `EVAL.md` carries the method, the `not run` table with the CPU controls and card numbers, the Kev-9B column, and the wall-time estimate (60 to 93 minutes for 7,254 requests at one TP=2 worker). `/home/hous/dev/clef/reports/final_numbers.json` is not written until the real run (the skeleton went to the scratchpad).

## Open for the device agent

- Start the real server (`CLEF_TRACED=1`, port 8008, through `devrun`), confirm `/v1/models` shows the selected precision, then `bash scripts/run_eval.sh --steps "arc banking77 newyorker samples parity kev summarize"` (host side, no `devrun` needed for the client). Expected wall 1 to 1.5 h; `eval_remote.py` resumes a file if interrupted.
- The gate: coverage complete, every parity set within the stage 1 bars, benchmark numbers next to the CPU controls; a TT-versus-CPU sample gap above 1.0 pp is a finding (the summary prints it).

## 2026 Oct 05, 17:35 ET: shipped server started for stage 5 (device)

Device agent. Lock free and no device process before the start (`flock -n` on `/home/hous/dev/clef/.device.lock` succeeded; `ps` showed no uvicorn, pytest or sweep process). No `QWEN36_*`, `QWEN_GDN_*`, `CLEF_PRECISION`, `CLEF_TRACED` or `CLEF_PLANNER` variable in the shell (`env | grep -c` printed 0); `precision_defaults.profile_name()` on the host returned `selected`.

Command (`/tmp/claude-1002/-home-hous-dev-clef/2a64a29f-04fe-4cd3-b935-9017907901d0/scratchpad/stage5_start_server.sh`; the server README "Run" command with the defaults written out and a 5 h bound):

```
cd /home/hous/dev/clef/tt-metal
SNAP=/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c
nohup /home/hous/dev/clef/bin/devrun timeout 18000 env OMP_NUM_THREADS=8 CLEF_MODEL=$SNAP HF_MODEL=$SNAP HF_HUB_OFFLINE=1 CLEF_MESH_SHAPE=1x2 MESH_DEVICE=P150x2 CLEF_TRACED=0 CLEF_PLANNER=1 python -m uvicorn models.autoports.cloudflare_clef.tt.server:app --host 127.0.0.1 --port 8008 --lifespan on > /home/hous/dev/clef/logs/stage5_server.log 2>&1 &
```

Log `/home/hous/dev/clef/logs/stage5_server.log`: `starting:` 21:35:52 UTC (`traced=False planner=True trace_region=0`), `mesh: 4 chip(s) visible, FABRIC_1D parent [1, 4] chips [1, 0, 3, 2], 1 TP group(s) [[1, 0]]` 21:35:56, engine `precision={... 'QWEN36_GDN_GATE_FP32': '1', 'CLEF_VISION_PRECISION': 'accuracy', 'CLEF_VISION_ACT_BF16': '1'}` (the seven `runtime_flags` of `doc/datatype_sweep/selected_precision_config.json`), `ClefEngine ready: load 100.2 s (state dict 1.8 s, build 89.4 s), DRAM free after weights 14.77 GiB, after slots 11.72 GiB, slots 4, ... traced=False traces=0 ... gdn_conv=fir, planner=True` 21:37:37, `worker 0 ready on [1, 0], prefix cache 4 state(s), media=True, mode=eager, gdn_conv=fir, planner=True, warm_grids=None`, warmup 375.1 ms, `ready: 1 worker(s) on ttnn 1x2: 1 TP=2 worker(s), mode=eager, gdn_conv=fir, planner=True, warm_grids=None`, `Application startup complete.` at 21:37:37 UTC (105 s from `starting`). `/v1/health` `{"status":"ok","workers":1,"queued":0}`; `/v1/models` (saved as `/home/hous/dev/clef/reports/stage5_models_start.json`): `backend ttnn`, `device "ttnn 1x2: 1 TP=2 worker(s)"`, `mode eager`, `traced false`, `gdn_conv fir`, `prefix_planner true`, `traced_media null`, `max_state_tokens 16384`, `precision` equal to the engine log line above.

Host-side fix before the run: `scripts/summarize_eval.py` rendered the EVAL.md start command with `CLEF_TRACED=1` (written before the stage 3 review made eager the default); changed to the command above (`CLEF_TRACED=0 CLEF_PLANNER=1`, `timeout 18000`). black unchanged.

## 17:37 ET: run_eval.sh started; step arc (ARC-Challenge test, 1,172 records)

Command (`/tmp/claude-1002/-home-hous-dev-clef/2a64a29f-04fe-4cd3-b935-9017907901d0/scratchpad/stage5_run_eval.sh`, host side, the client holds no device): `cd /home/hous/dev/clef/tt-metal && nohup bash models/autoports/cloudflare_clef/scripts/run_eval.sh --base-url http://127.0.0.1:8008 --concurrency 4 --steps "arc banking77 newyorker samples parity kev summarize" > /home/hous/dev/clef/logs/stage5_run_eval.log 2>&1 &` at 21:37:51 UTC. Step log `/home/hous/dev/clef/logs/stage5_eval_arc_challenge_test.log`, rows `/home/hous/dev/clef/reports/eval/arc_challenge_test.jsonl`, summary `arc_challenge_test.summary.json`.

Outcome: `EVAL_REMOTE_DONE failed=0`, 1,172 of 1,172 answered (0 errors), wall 274.7 s (21:37:51 to 21:42:26 UTC), 4.267 records/s at concurrency 4; server `latency_ms` median 231.9, p95 239.7, max 471.8 (sum 274.4 s, so the worker was busy for the whole wall); client `wall_ms` median 929.8 (queue wait of 4 in flight); input tokens median 198, max 333. `eval_metrics.py` on the host: 1,146 of 1,172 correct, accuracy 97.78 percent (card 97.7); missing 0, errors 0, unanswered 0. Estimate was 4 to 7 min; measured 4.6 min.

## 17:42 ET: step banking77 (BANKING77 test, 3,080 records)

Log `/home/hous/dev/clef/logs/stage5_eval_banking77_test.log`, rows `/home/hous/dev/clef/reports/eval/banking77_test.jsonl`. `EVAL_REMOTE_DONE failed=0`, 3,080 of 3,080 answered, wall 2,606.3 s (21:42:26 to 22:25:52 UTC, 43.4 min; estimate 36 to 51 min), 1.182 records/s; server `latency_ms` median 839.1, p95 869.9, max 989.7 (sum 2,594 s: the worker was busy for 99.5 percent of the wall); input tokens median 1,831, max 1,898 (every request carries the 77-option schema). Score (`final_numbers.json` `benchmarks.banking77.full`): 2,908 of 3,080 correct, accuracy 94.42 percent, macro-F1 over the 77 intents 94.40 (card 94.2); missing 0, errors 0, unanswered 0.

## 18:25 ET: step newyorker (New Yorker caption matching test, 528 records with one image each)

Log `/home/hous/dev/clef/logs/stage5_eval_newyorker_matching_test.log`, rows `/home/hous/dev/clef/reports/eval/newyorker_matching_test.jsonl`. `EVAL_REMOTE_DONE failed=0`, 528 of 528 answered, wall 318.6 s (22:25:52 to 22:31:11 UTC, 5.3 min; estimate 4 to 6 min), 1.657 records/s; `latency_ms` median 575.0, p95 961.5, max 2,886.6. The tail is the first request of each new input signature: grouping the 528 image requests of the server log by their state token count S (image tokens included; 55 distinct values), the first request of each S has p50 956 ms, p95 2,608 ms, max 2,887 ms, every later request of the same S has p50 570 ms, p95 759 ms, max 846 ms, and all 21 requests above 1,200 ms are such first requests. This is consistent with the eager tower compiling its programs per image grid (`doc/optimized/README.md`, "The compiling allocations"); S is a proxy for the grid, the grid itself is not in the server log; input tokens median 523, max 849; 14 prefix-cache hits (the dataset repeats a cartoon with different caption sets). Score: 308 of 528, accuracy 58.33 percent (card 69.5; the CPU control on the 100-item sample is 60.0 under this rendering, see the samples step). 0 missing, 0 errors, 0 unanswered, 0 refused (no 422 in the server log).

## 18:31 ET: step samples (three 100-item stratified samples, the CPU bf16 controls)

Logs `/home/hous/dev/clef/logs/stage5_eval_{arc_challenge,banking77,newyorker_matching}_test_sample100.log`, rows and summaries under `/home/hous/dev/clef/reports/eval/`. All three `failed=0`, 100 of 100. Walls 23.4 s, 83.4 s, 54.0 s (22:31:11 to 22:33:52 UTC). The TT sample rows are bit-identical to the same ids in the full-set runs (ARC checked: max dp 0 over the 100 records; the server is deterministic across the run). Against the CPU bf16 controls (`parity_compare.summarize`, margin 0.05, `reports/eval/metrics/parity_*_sample100.{json,md}`):

| Sample | TT | CPU bf16 | gap pp | max dp | mean dp | argmax flips | flips at margin | near-tie |
|---|---|---|---|---|---|---|---|---|
| ARC-Challenge, accuracy | 96.0 | 95.0 | +1.00 | 0.3112 | 0.0064 | 1 | 1 | 0 |
| BANKING77, macro-F1 (accuracy 93.0 both) | 91.1 | 91.1 | 0.00 | 0.1101 | 0.0055 | 2 | 0 | 2 |
| New Yorker, accuracy | 60.0 | 60.0 | 0.00 | 0.2124 | 0.0312 | 4 | 3 | 1 |

The ARC gap is exactly 1.0 pp (one question of 100: `Mercury_7177398`, TT right, CPU wrong), which is not above the 1.0 pp threshold of the rule; the first summary run printed it as a finding because `100 * (0.96 - 0.95)` is `1.0000000000000009` in floating point, so `summarize_eval.py` now compares the rounded gap and reports an exact-threshold gap as a note. The record is investigated in `EVAL_findings.md` with a CPU fp32 control (below).

## 18:33 ET: step parity (the stage 1 and 2 sets over HTTP)

Logs `/home/hous/dev/clef/logs/stage5_eval_{reference_text,reference_image,dev64_text,dev16_image}.log`, rows under `/home/hous/dev/clef/reports/eval/`, comparisons in `reports/eval/metrics/parity_<set>_vs_{bf16,fp32}.{json,md}`. All four `failed=0`; walls 5.9, 4.1, 27.6, 9.3 s (22:33:52 to 22:34:39 UTC). reference_text 27 questions: max dp 0.0722, mean 0.0109, 0 flips against bf16 (0.0759 / 0.0104 / 0 against fp32): the stage 1, 3 and 4 numbers reproduced by the shipped server. reference_image 8: 0.0463 / 0.0200 / 0 (fp32 0.0383 / 0.0189 / 0): the stage 2 number. dev64_text 95 questions: max dp 0.1923, mean 0.0124, 1 argmax flip, 1 at margin; its first 16 records equal reference_text (max dp 0.0722, 0 flips) and the 48 new records (68 questions) carry the two questions above 0.10. dev16_image 16: max dp 0.1754, mean 0.0318, 0 flips; the rows equal the stage 4 sweep's dev16 rows (`reports/sweep/baseline_stage1/dev16_image.jsonl`) within the 4-decimal API rounding (max dp 5e-5), so the server reproduces the engine of stage 4. Both are discussed in `EVAL_findings.md`.

## 18:34 ET: step kev (Kev suites, test splits, `kev.benchmark --remote`)

Logs `/home/hous/dev/clef/logs/stage5_kev_{hard-v1,devtools-v1,documents-v1}.log`, reports `/home/hous/dev/clef/reports/eval/kev/<suite>/test/{report.json,rows.json,predictions.jsonl}`. Command per suite (from `run_eval.sh`): `cd /home/hous/dev/kev/kev && env -u VIRTUAL_ENV uv run python -m kev.benchmark --remote http://127.0.0.1:8008 --out /home/hous/dev/clef/reports/eval/kev/<suite>/test --remote-concurrency 4 --remote-model clef --suite evals/<suite> --allow-test`. Walls: hard-v1 6 min 19 s (22:34:39 to 22:40:58 UTC), devtools-v1 5 min 41 s, documents-v1 5 min 43 s. `clean` block: hard-v1 n=1,088 acc 0.774 Brier 0.321 ECE 0.044; devtools-v1 n=1,073 acc 0.741 Brier 0.399 ECE 0.143; documents-v1 n=936 acc 0.893 Brier 0.169 ECE 0.044. Coverage 700/700, 900/900, 574/574 records, 0 rejected, 0 truncated, no `long_rows`. Client latency (queue wait at 4 in flight included) p50 / p95 ms: 1,608 / 4,243, 1,464 / 2,126, 2,309 / 3,164; server model time per request in the same windows (from `stage5_server.log`): p50 409 / p95 1,840, 395 / 643, 493 / 1,072. Kev-9B on one P150 for comparison (`/home/hous/dev/kev/reports/final_numbers.json`): 0.826 / 0.053, 0.787 / 0.101, 0.896 / 0.015 (acc / ECE).

## 18:52 ET: step summarize, then the server stop (device)

`summarize_eval.py --write` at 22:52:22 UTC (log `/home/hous/dev/clef/logs/stage5_summarize.log`): `RUN_EVAL_DONE failures=0`. Whole run 21:37:51 to 22:52:22 UTC: 74.5 min for 7,358 client requests (the estimate table counted 7,254 because it omitted the 104 parity records; estimate 60 to 93 min); with the 105 s server start, 76.3 min.

Server stop at 23:08:42 UTC: `kill -TERM 1521160` (the python process; `pgrep -f "^python -m uvicorn models.autoports.cloudflare_clef"`), `Application shutdown complete` at 23:09:14 UTC, `Finished server process [1521160]`, no uvicorn, timeout or flock process left, `flock -n /home/hous/dev/clef/.device.lock true` succeeds (lock free). The server log has 7,359 `latency_ms=` lines: the 7,358 eval requests (one `worker=0 ... latency_ms=` line each, parsed for the distribution below) plus the startup warmup line; 0 tracebacks, 0 non-200 responses (`/v1/models` and `/v1/health` calls do not log a latency line).

Server latency distribution over the whole run (model time `latency_ms` from `/home/hous/dev/clef/logs/stage5_server.log`, 7,358 requests): p50 626.5 ms, p90 853.1, p95 872.2, p99 1,504.1, max 2,886.6, min 224.7, sum 4,447 s (the worker was busy 99.5 percent of the 4,471 s run). Histogram: 0 to 300 ms 2,093; 300 to 500 ms 1,091; 500 to 800 ms 710; 800 to 1,000 ms 3,306 (BANKING77); 1,000 to 1,500 ms 84; 1,500 to 2,500 ms 70; 2,500 to 5,000 ms 4; above 5 s 0. Per step p50 / p95: ARC 232 / 240, BANKING77 839 / 870, New Yorker 575 / 962, dev64 434 / 880, dev16 569 / 772, Kev hard-v1 409 / 1,840, devtools-v1 395 / 643, documents-v1 493 / 1,072. 17 prefix-cache hits in the whole run (14 New Yorker, 2 New Yorker sample, 1 reference image: repeated cartoons); everything else was a miss, as expected for distinct states.

## 19:11 ET: CPU fp32 control on the three finding records (host, no device)

`findings_records.jsonl` (3 records: `hard-v1/temporal_numeric/development/00008`, `hard-v1/tradeoff/development/00013`, `Mercury_7177398`) through `cpu_reference.py --dtype float32 --threads 8` (`/home/hous/dev/clef/logs/stage5_findings_fp32_control.log`, output `/home/hous/dev/clef/reports/eval/findings_records.ref_fp32.jsonl`). Result in `EVAL_findings.md`.

## 19:15 ET: findings written, summary regenerated, stage 5 device work closed

fp32 control result: on all three records the CPU fp32 and CPU bf16 references agree (dp 0.0079, 0.0339, 0.0299, same argmax) and the device differs from both (dp 0.09 to 0.34), so the three disagreements are device-side; details, the family table and the classification against the stage 4 ledger are in `doc/benchmark/EVAL_findings.md`, which `summarize_eval.py --write` appends to `EVAL.md` (regenerated 23:15:26 UTC; `final_numbers.json` `findings` 2, `notes` 1, `coverage_ok` true). `summarize_eval.py` changes of this stage: the rendered start command (eager, planner, `timeout 18000`), the rounded gap comparison with an exact-threshold note, flip record names and the largest-dp record in each finding, the `notes` list in `final_numbers.json`, the `EVAL_findings.md` inclusion. black (target py312) unchanged on `summarize_eval.py` and `eval_remote.py`; no code comments. No `tt/*.py`, `qwen36` or `run_eval.sh` change. Device lock free since 23:09:14 UTC; no commit (the orchestrator commits after review).

## Stage 5 remediation (2026 Oct 05, 19:25 ET onward; review `/home/hous/dev/clef/reports/review_stage5.md`)

Scope: P2-1 (per-layer probe of the three stage 5 disagreements), P2-2 (evaluation rows in `doc/benchmark/README.md`), the nits (Kev devtools-v1 n, the long-record sentence, `7e59b11c` as an open item, the documents-v1 count). Files owned: `doc/benchmark/*`, `doc/datatype_sweep/README.md` (ledger section only), `scripts/summarize_eval.py`. No `tt/*.py`, `qwen36` or `scripts/sweep_anomaly_probe.py` change (the probe's `--records`, `--ref-bf16`, `--ref-fp32` and `--sweep-rows` flags were enough). Logs `/home/hous/dev/clef/logs/stage5r_*.log`. A container build was running on the host; every command used `OMP_NUM_THREADS=8` and a `timeout`.

## 19:29 ET: HF bf16 hidden states of the three records (host, 21 s)

Records `/home/hous/dev/clef/reports/eval/findings_records.jsonl` (the 3 records of the fp32 control: `hard-v1/tradeoff/development/00013`, `hard-v1/temporal_numeric/development/00008`, `Mercury_7177398`). Command: `cd /home/hous/dev/clef/tt-metal && env OMP_NUM_THREADS=8 CLEF_MODEL=<snapshot> HF_MODEL=<snapshot> timeout 1200 /home/hous/dev/clef/bin/hostrun python models/autoports/cloudflare_clef/scripts/sweep_anomaly_probe.py --hf-only --hf-cache <scratchpad>/hf_cache --records /home/hous/dev/clef/reports/eval/findings_records.jsonl`; log `/home/hous/dev/clef/logs/stage5r_anomaly_probe_hf.log`. `00013`: T=530, tail at row 314, HF forward 11.4 s; `00008`: T=314, tail at 141, 5.6 s; `Mercury_7177398`: T=208, tail at 61, 3.7 s. `HF_ONLY_DONE` at 23:29:47 UTC. The New Yorker record `7e59b11c` was not given to the probe: `tt/encode.py` `encode` without a processor raises on a record with images, the HF step runs the text model alone and the device step opens the engine with `vision=False`.

Inputs built for the device step (scratchpad, no code change): `stage5r_findings.ref_bf16.jsonl` (the ARC sample bf16 reference and `dev64_text.ref_bf16.jsonl` concatenated, 164 rows) for `--ref-bf16`, and `stage5r_findings.served_rows.jsonl` (`reports/eval/arc_challenge_test_sample100.jsonl` and `reports/eval/dev64_text.jsonl` concatenated) for `--sweep-rows`, so the probe anchors the TT head against the rows the server served; the probe looks rows up by id.

## 19:30 to 19:35 ET: per-layer probe of the three records, two precisions (device, one lock hold, 4 min 49 s)

Batch `<scratchpad>/stage5r_anomaly_batch.sh` under `env DEVRUN_WAIT=3600 devrun timeout 3600` (log `/home/hous/dev/clef/logs/stage5r_anomaly_batch.log`); the lock was free (`flock -n` succeeded, no uvicorn or sweep process). Two processes, each `timeout 1500 env OMP_NUM_THREADS=8 CLEF_MODEL=<snapshot> HF_MODEL=<snapshot> CLEF_TRACED=0 [knob] python models/autoports/cloudflare_clef/scripts/sweep_anomaly_probe.py --variant <v> --hf-cache <scratchpad>/hf_cache --dump-dir /home/hous/dev/clef/reports/stage5r_anomaly_dumps --records .../findings_records.jsonl --ref-bf16 <scratchpad>/stage5r_findings.ref_bf16.jsonl --ref-fp32 /home/hous/dev/clef/reports/eval/findings_records.ref_fp32.jsonl --sweep-rows <scratchpad>/stage5r_findings.served_rows.jsonl --out /home/hous/dev/clef/reports/stage5r_anomaly_probe_<v>.json`: `selected` (no knob) 23:30:09 to 23:32:31 UTC (engine load 85.8 s, `vision=False`, 15.27 GiB free after the weights) and `hifi4` (`QWEN36_MATMUL_FIDELITY=HiFi4`; readout `mlp_fidelity HiFi4`, GDN and attention HiFi2) 23:32:31 to 23:34:58 UTC (load 86.4 s). Both `exit 0`, `ANOMALY_PROBE_DONE`. Logs `/home/hous/dev/clef/logs/stage5r_anomaly_probe_{selected,hifi4}.log`, reports `/home/hous/dev/clef/reports/stage5r_anomaly_probe_{selected,hifi4}.json`, dumps under `/home/hous/dev/clef/reports/stage5r_anomaly_dumps/` (file prefix `stage4r_hidden_`, the probe's fixed name). Per record the device work was 1.1 to 3.9 s accumulated pass, 8 to 18 s splice; the two engine loads were most of the time.

Anchors (selected): TT head equals the served row on every question (dp 0.0: `Mercury_7177398` D 0.5957 / A 0.3393, `00008` c 0.3988 / b 0.3770, `00013` wildfern_agency 0.7302 / paper_kite 0.2633); HF bf16 head equals the CPU bf16 reference within 0.0004, 0.0018, 0.0002. Per record and precision (teacher-forced delta PCC mean of means GDN / attention, worst layer mean, layers below 0.999; stage 1 band 0.999785 / 0.999761, worst layer 0.999542, floor 0.999):

| Record, precision | T, bucket | GDN / attention | worst layer (mean) | below 0.999 | final norm min / mean | TT top-2 | splice |
|---|---|---|---|---|---|---|---|
| `Mercury_7177398`, selected | 208, 256 | 0.999787 / 0.999767 | 12 (0.999507) | none | 0.851 / 0.9920 | D 0.5957, A 0.3393 | non-monotonic; D at the prefixes 0, 2, 5, 7 to 10, 17, 19; converges from 47 |
| `Mercury_7177398`, HiFi4 | 208, 256 | 0.999789 / 0.999775 | 12 (0.999502) | none | 0.956 / 0.9967 | A 0.5082, D 0.4254 | non-monotonic; D at 4, 6 to 9 |
| `00008`, selected | 314, 512 | 0.999781 / 0.999766 | 16 (0.999520) | none | 0.949 / 0.9969 | c 0.3988, b 0.3770 | HF layer 0 alone restores b (0.4015 / 0.3613); converges from 38 |
| `00008`, HiFi4 | 314, 512 | 0.999785 / 0.999774 | 16 (0.999531) | none | 0.842 / 0.9963 | b 0.4318, c 0.3197 | b at every prefix |
| `00013`, selected | 530, 1024 | 0.999777 / 0.999754 | 16 (0.999524) | none | 0.670 / 0.9946 | wildfern_agency 0.7302, paper_kite 0.2633 | wanders 0.49 to 0.75; paper_kite at 24, 26; converges from 46 |
| `00013`, HiFi4 | 530, 1024 | 0.999780 / 0.999763 | 16 (0.999550) | none | 0.655 / 0.9962 | wildfern_agency 0.6074, paper_kite 0.3865 | paper_kite at 27 only |

Outcome: every layer in band on all three records at both precisions, no stop condition; classification as the stage 4 ledger entry 1 class confirmed and written as "Anomaly ledger entry 2" in `doc/datatype_sweep/README.md`, referenced from `EVAL_findings.md` sections 1 and 2 and the new "Stated limitation" paragraph. Observation recorded, not acted on: under MLP HiFi4 all three records land on the CPU argmax while the per-layer means move by at most 1e-5 (`all_hifi4` was not eligible in stage 4: one ref16 flip at margin, 17 percent slower). Extraction script for the tables: `<scratchpad>/stage5r_probe_table.py` over the two reports.

## 19:36 ET: documentation (host, no device)

- `doc/benchmark/README.md`: line 3 now points at the new section "Evaluation rows (stage 5)" (under the model-card footnotes, above "Latency detail"): the model-card table and the Kev suites table copied from `EVAL.md`, the parity line, links to `EVAL.md`, `EVAL_findings.md` and `final_numbers.json`, and the statement that `EVAL.md` is the generated source and `summarize_eval.py` does not rewrite the README. Stage 3 tables untouched. Acronyms CPU, pp, ECE, macro-F1 added.
- `scripts/summarize_eval.py`: the Kev-9B cell carries `(n=<Kev n>)` when Kev's clean question count differs from Clef's, and the footnote states Kev's devtools-v1 n=1,071 after two audited drops (`drop_ids` `codereviewer/cls-test/13657`, `19245` in `/home/hous/dev/kev/reports/final_numbers.json`) against Clef's 1,073, at most 0.2 pp. `black --check -l 120`: unchanged; 0 comment lines.
- `doc/benchmark/EVAL_findings.md`: the "all gate-named sets pass over HTTP" sentence now names the three sets that went over HTTP in this stage and states that the 8 long records did not (stage 1 engine-level logs `stage1r_parity_long_records_*.log`, stage 3 remediation `long/max` over HTTP); sections 1 and 2 carry the probe result; section 4 states why `7e59b11c` has no control or probe path and that "device-side" is not established for it; the gate statement and the open items rewritten (item 1 closed, `7e59b11c` open as item 2, the long records as item 6); the "Stated limitation (for stage 6)" paragraph added.
- `EVAL.md` regenerated: `/home/hous/dev/clef/bin/hostrun python models/autoports/cloudflare_clef/scripts/summarize_eval.py --write` (log `/home/hous/dev/clef/logs/stage5r_summarize.log`, 23:38:43 UTC); `final_numbers.json` `findings` 2, `notes` 1, `coverage_ok` true, numbers unchanged.
- documents-v1 count: `grep -rn 568` over the autoport docs finds the number only in `doc/functional/work_log.md` (the development split, 568 states of `development.jsonl`, which is correct) and in unrelated tracy and JSON values; `EVAL.md`, `README.md` and `work_log.md` carry 574 test records; the plan's Stage 5 text reads `700/900/574`. Nothing to change in the autoport docs.
- No commit (the orchestrator commits after review). Device lock free since 23:34:58 UTC.

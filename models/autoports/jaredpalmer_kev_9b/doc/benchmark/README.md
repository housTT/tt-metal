# kev-9b on Blackhole P150: benchmark and evaluation (stage 6)

Date: 2026 Oct 02, 00:04 to 02:40 UTC (the host clock; every time in this document is UTC; the stage 5 device runs started Oct 01 23:11 UTC). Box p300c, four Blackhole P150 chips. Server `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/tt/server.py`, engine `tt/engine.py`, dispatcher `tt/dispatch.py`. Machine-readable copy of every number here: `/home/hous/dev/kev/reports/final_numbers.json` (written by `scripts/final_numbers.py`). Stage 5 chronology and the GIL analysis: `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/multichip/work_log.md`.

Acronyms: ECE (expected calibration error), GIL (global interpreter lock), KV (key/value), req/s (requests per second), p50 / p95 / p99 (percentiles), DP (data parallel), LoFi (low-fidelity math mode), bfp8 (block floating point, 8 bits per element).

## Model-card table

Format of the kev model card (`docs/model-cards/kev-9b.md` in the kev repository): model time per request, new / repeated state, and requests per second at 64 concurrent clients.

| Device | 6 questions, short state | 5 questions, 2,200-token state | Requests/s, 64 clients |
|---|---|---|---|
| P150 (1 chip) [1] | 607.8 / 607.7 ms | 1,533.5 / 527.0 ms | 1.6 |
| P150 x4 (data parallel) [2] | 203.9 / 203.9 ms | 1,533.4 / 527.0 ms | 6.5 |

Footnotes.

- New / repeated state: the first column of each pair is a request whose state the server has not seen (prefix-cache miss: the state is prefilled, then the questions run), the second a request whose state is in the prefix cache (hit: the KV and GDN state are restored, then the questions run). Each worker holds 8 states (8 KV slots of 65,536 tokens).
- Model time per request is the server's `latency_ms` field: the wall time of the model section of the request (state prefill or cache restore, question tails, pointer head), measured in the worker thread. It excludes queue wait and HTTP. With fan-out it is the maximum over the workers that shared the request (the request's critical path); the device time consumed is the sum over shares (`latency_ms_sum` in the server log: 613 ms for the 6-question case against 607 ms on one chip). The x4 short-state value is therefore one request's critical path over four chips, where the card's GPU rows run the whole request on one device. A state under 128 tokens is not held by the prefix cache (`tt/engine.py`: `S0 = 0`, the question tail carries the whole state), so new and cached are equal by construction in the short-state column.
- Precision: `mlp_bfp8` (`doc/datatype_sweep/selected_precision_config.json`): bfp8 weights for every projection and the MLP, LoFi matmuls with fp32 accumulation, bf16 KV cache, bf16 GDN state, SDPA HiFi2, GDN chunk kernel fp32, host pointer head in fp32. Weight cache `/home/hous/dev/kev/tt_cache/P150/tensor_cache_bfp8_kev_2b2a70cf_gu-bfp8_dn-bfp8_pj-bfp8`.
- Prefill is traced (`KEV_TRACED=1`): one trace per bucket (128 to 2,048 tokens) for the forward and the readout, plus state save / restore traces; the matmul program-config policy is on (`KEV_MATMUL_POLICY=1`).
- Code: tt-metal worktree `/home/hous/dev/kev/tt-metal`, branch `hous/kev-9b-bringup`, HEAD `ed917633df8` (base `7eac776e926`, origin/main), clean tree. The rows were measured at `9a06a5bed76` plus the one-line `tt/engine.py` change that `ed917633df8` then committed unchanged (`_gather` reads with `B["rows"].cpu()`, which releases the GIL during the device wait; without it four workers reach only 2.5 req/s). kev client at `952ce9d`.
- Method: `scripts/serving_bench_remote.py --reps 20 --concurrency 1,8,32,64` (port of kev's `scripts/serving_bench.py` over HTTP, `httpx` clients on the same host). Latency columns are the median of 20 requests after 2 warm-up requests; throughput is the full method, 256 requests of the 6-question short-state case (distinct states) per client level, two passes, the second timed. The one-chip row is the first full-method one-chip run; stages 2 to 4 reported `--quick` (32 requests per level), which gives the same 1.6 req/s for one chip.
- [1] One worker on one submesh of the 2x2 mesh (`KEV_DEVICES=0`, physical chip 1), whole-request dispatch (`KEV_FANOUT=0`); `/home/hous/dev/kev/reports/bench/p150x1_whole/report.json`. The same configuration measured with `ttnn.open_device(0)` in stage 4 gave 606.4 / 606.4 and 1,531.6 / 525.5 ms.
- [2] Four workers, one per submesh, question-level fan-out on (`KEV_FANOUT=1`, degrade threshold 200 ms of backlog); `/home/hous/dev/kev/reports/bench/p150x4_fanout/report.json`. The short-state request is split 2 / 2 / 1 / 1 questions over the four chips; the 2,200-token state stays on one chip because replicating a 1.0 s prefill buys nothing. Under load the planner falls back to whole requests (0 to 4 % of requests fanned out at 8 to 64 clients), so the 64-client throughput equals the whole-request figure on every sample (`/home/hous/dev/kev/reports/bench/p150x4_whole_fix/report.json`, same method: 6.5 / 15.6 / 2.6 req/s against 6.5 / 15.6 / 2.5 with fan-out).

Throughput detail, requests/s and p50 / p99 wall time per request in ms (client side, queue wait included):

| Sample, device | 1 client | 8 clients | 32 clients | 64 clients |
|---|---|---|---|---|
| 6 questions, new short state, 1 chip | 1.6, 611 / 650 | 1.6, 4,884 / 5,055 | 1.6, 19,573 / 19,778 | 1.6, 39,116 / 39,504 |
| 6 questions, new short state, x4 fan-out | 4.8, 207 / 212 | 6.5, 1,216 / 1,319 | 6.5, 4,859 / 4,970 | 6.5, 9,723 / 9,944 |
| decision-v7 development, 1 chip | 5.6, 105 / 477 | 5.7, 1,368 / 2,280 | 5.7, 5,077 / 7,599 | 5.7, 10,820 / 14,335 |
| decision-v7 development, x4 fan-out | 6.2, 104 / 477 | 16.6, 406 / 1,143 | 15.5, 1,939 / 3,307 | 15.6, 3,620 / 6,157 |
| 5 questions, new 2,200-token state, 1 chip | 0.6, 1,539 / 1,570 | 0.7, 12,283 / 13,395 | 0.7, 44,683 / 50,219 | 0.7, 50,093 / 96,897 |
| 5 questions, new 2,200-token state, x4 fan-out | 0.9, 1,538 / 1,544 | 2.6, 3,068 / 3,082 | 2.6, 12,269 / 12,294 | 2.5, 13,054 / 24,666 |

The long-state sample sends 64 distinct 2,200-token states through 8 KV slots per chip, so nearly every request prefills (1.5 s); stage 4's `--quick` value of 1.9 req/s for this sample was 8 states that all hit after the first pass.

Scaling of whole-request DP, 64 clients, short-state sample: 1 worker 1.6 req/s (full method), 2 workers 2.0 (`--quick`: 32 requests at the 64-client level, `/home/hous/dev/kev/reports/bench/p150x2_whole/report.json`), 4 workers 2.5 (full method) as patched; 4 workers 6.5 (full method) and 6.6 (`--quick`) with the GIL-releasing readback (4.1x). The 2-worker point is not the same method as the 1- and 4-worker points, so the three-point curve is directional. Cause and evidence (`py-spy` profiles): `doc/multichip/work_log.md`, section A.

## Evaluation against the model card

`scripts/run_eval.sh --base-url http://127.0.0.1:8008 --concurrency 4` (kev's `kev.benchmark --remote`, suites from `/home/hous/dev/kev/kev/evals`, checksums verified) on the x4 fan-out server, 02:04 to 02:11 (development) and 02:11 to 02:18 (test; the test splits were run once, after the development splits, and are labelled as such). Metrics are kev's `clean` block (clean variant, knowable rows): accuracy, Brier, ECE (10 bins on the top probability). Latency is the client wall time per request at concurrency 4, p50 / p95. Model-card numbers from `docs/model-cards/kev-9b.md` (jaredpalmer/kev-9b v2, Qwen3.5-9B-Base, fp32 evaluation on an H100).

| Suite / split | n (questions) | acc | Brier | ECE | latency p50 / p95 ms | model card acc / ECE |
|---|---|---|---|---|---|---|
| hard-v1 / development | 1,083 | 0.808 | 0.272 | 0.057 | 352 / 2,270 | (pooled below) |
| devtools-v1 / development | 1,072 | 0.759 | 0.342 | 0.083 | 302 / 862 | (pooled below) |
| documents-v1 / development | 920 | 0.896 | 0.150 | 0.018 | 774 / 1,836 | 0.902 |
| smoke-v1 / development | 18 | 0.889 | 0.164 | 0.123 | 207 / 540 | not on the card |
| hard-v1 + devtools-v1 audited / development | 1,855 | 0.811 | 0.275 | 0.071 | | 0.821 |
| hard-v1 / test | 1,088 | 0.826 | 0.244 | 0.053 | 393 / 2,261 | 0.834 / 0.054 |
| devtools-v1 / test | 1,071 | 0.787 | 0.319 | 0.101 | 296 / 884 | 0.791 / 0.098 |
| documents-v1 / test | 936 | 0.896 | 0.157 | 0.015 | 764 / 1,484 | 0.900 / 0.017 |
| smoke-v1 / test | 18 | 0.944 | 0.127 | 0.187 | 208 / 561 | not on the card |
| hard-v1 + devtools-v1 audited / test | 1,859 | 0.814 | 0.262 | 0.059 | | 0.822 |
| breadth-v1 | not available | | | | | 0.698 / 0.034 |

"Audited" pools the clean knowable rows of hard-v1 and devtools-v1 and drops devtools-v1 rows with source `flakeflagger` and task `commitpackft_type`; every suite row also drops the card's `drop_ids` (`codereviewer/cls-test/13657`, two clean rows in the devtools-v1 development split, and `codereviewer/cls-test/19245`, two in the test split). That is the card's definition (`experiments/rounds/r27.json`, applied by kev's `scripts/release_numbers.py`) and gives the card's n (devtools-v1 1,072 / 1,071, pooled 1,855 / 1,859). The first version of this table omitted the `drop_ids` (n 1,074 / 1,073 and 1,857 / 1,861, accuracies at most 0.05 pp higher); found by `/home/hous/dev/kev/reports/review_stage5_6.md`, fixed in `scripts/summarize_eval.py`. breadth-v1's data lives in a private Hub mirror this account cannot read (`NOT-AVAILABLE LocalEntryNotFoundError`). Coverage was complete: no record rejected or truncated in any suite (700 + 900 + 568 + 30 records on the development splits, 700 + 900 + 574 + 30 on the test splits). The whole evaluation took 14 min on the four chips (about 5 req/s at concurrency 4).

## Parity against fp32

- 16 reference records (`/home/hous/dev/kev/reports/reference`, 29 questions) through the x4 fan-out server, two passes (`reports/stage5_parity_x4.json`): vs kev's CPU fp32 `LocalPredictor` max |dp| 0.0878, mean |dp| 0.0272, 1 argmax flip on a near tie (fp32 top-2 margin 0.0315), 0 flips at margin >= 0.05; 16 / 16 revisited states answered identically. The served answers are byte-identical on each of the four chips alone and to the stage 4 single-chip server (`reports/stage5_parity_compare.json`).
- 64 development records (16 each from hard-v1, devtools-v1, documents-v1 at indices 4 to 19, plus the 16 reference records; 104 questions; `reports/parity64/compare.json`), server vs CPU fp32 `LocalPredictor` (`SERVING_CONTEXT`, 33 min on 8 host threads): max |dp| 0.416, mean |dp| 0.0368, 4 argmax flips: 2 at margin >= 0.05 (hard-v1 record 0 `date`, margin 0.194; devtools-v1 record 19 `action`, margin 0.088) and 2 near ties. By suite: hard-v1 max 0.416 / mean 0.066, devtools-v1 0.069 / 0.032, documents-v1 0.132 / 0.022, reference 0.088 / 0.027. Median per-record max |dp| 0.031. The reference for the shared 16 records reproduces `reports/reference/probs_fp32.json` bit for bit.
- For scale: kev's own bf16 H100 serving path against its fp32 path is max |dp| 0.0217, mean 0.0014, 0 flips on 200 decision-v7 records (`runs/serve-9b-h100/report.json` in the kev repository, block `eager_vs_fp32`; `/home/hous/dev/kev/reports/kev_internals.md` section 7). The model card text states the same comparison as "within 0.017 of the fp32 evaluation path on 280 questions" (`docs/model-cards/kev-9b.md` line 248), a different run.
- Control against kev's own fp32 rows (computed by the stage review, `/home/hous/dev/kev/reports/review_stage5_6.md`, "Required Work", fourth item; CPU only): the TT eval rows paired on `(id, question)` with kev's fp32 rows for this checkpoint (`runs/r18-9b-joint-lr2e5-{hard,devtools,docs}/rows.json` for development, `runs/r27c-9b-cand-{hardtest,devtest,docs1test}/rows.json` for test, re-served at temperature 2.193649959389252 with `kev.metrics.served_at`; they reproduce the card to four decimals), 6,172 clean knowable questions, labels identical on every pair:

| Read | matched | mean abs dp | p95 | max abs dp | argmax flips | flips at fp32 margin >= 0.05 | TT right / fp32 wrong | fp32 right / TT wrong | net |
|---|---|---|---|---|---|---|---|---|---|
| hard-v1 development | 1,083 | 0.048 | 0.152 | 0.681 | 54 (5.0 %) | 34 | 20 | 25 | -0.46 pp |
| devtools-v1 development | 1,073 | 0.032 | 0.091 | 0.287 | 56 (5.2 %) | 18 | 16 | 30 | -1.30 pp |
| documents-v1 development | 920 | 0.026 | 0.112 | 0.296 | 21 (2.3 %) | 8 | 6 | 12 | -0.65 pp |
| hard-v1 test | 1,088 | 0.042 | 0.155 | 0.430 | 46 (4.2 %) | 35 | 16 | 24 | -0.74 pp |
| devtools-v1 test | 1,072 | 0.032 | 0.095 | 0.206 | 40 (3.7 %) | 14 | 14 | 18 | -0.37 pp |
| documents-v1 test | 936 | 0.025 | 0.107 | 0.332 | 7 (0.7 %) | 6 | 2 | 5 | -0.32 pp |

  Deviations are two-sided and the net accuracy cost is 0.3 to 1.3 pp per read. The 0.416 row of the 64-record sample is `hard-v1/temporal_numeric/development/00048`; it deviates by 0.415 against kev's GPU fp32 rows, so the CPU `LocalPredictor` reference and kev's own fp32 reads agree. The single largest paired row, `hard-v1/temporal_numeric/development/00044` (|dp| 0.68), favours the TT path (TT 0.798 on the labelled option, fp32 0.873 on a wrong one). The 64-record sample (mean 0.037, hard-v1 0.066, documents 0.022) sits inside these whole-suite figures. No task family shows a loss beyond small-n noise; the deviations concentrate on hard-v1 reasoning families (mean |dp| 0.05 to 0.07 against 0.02 to 0.03 elsewhere).

## Known gaps

- Accuracy below the card on every split: test hard-v1 0.826 against 0.834, devtools-v1 0.787 against 0.791, documents-v1 0.896 against 0.900, pooled audited 0.814 against 0.822 (0.4 to 0.8 points); development pooled audited 0.811 against 0.821, with hard-v1 0.808 and devtools-v1 0.759 the furthest. ECE matches the card within 0.003 on the test splits. This is the bfp8 / LoFi precision of the device path (stage 4 chose it inside a 1-point window on a 291-row subset); the paired comparison above puts the net cost at 0.3 to 1.3 pp per read with two-sided flips, concentrated on the hard-v1 reasoning families. A bf16-weight or HiFi2 configuration costs throughput (stage 4 sweep).
- breadth-v1 is not evaluated (private data).
- The one-line `engine.py` readback change is committed as `ed917633df8` (it was uncommitted when the rows were measured); it is required for the x4 throughput row. `ttnn.from_device` holding the GIL remains a tt-metal binding issue (`ttnn/cpp/ttnn-nanobind/operations/core.cpp`, no `gil_scoped_release`).
- Long-state throughput is prefill-bound with 8 KV slots per chip: 64 distinct 2,200-token states give 2.5 req/s on four chips; the planner keeps a long state on one chip, so fan-out cannot help a new long state.
- decision-v7 throughput depends on the sample: 15.6 req/s at 64 clients with 256 distinct states (full method, both dispatch modes), 20.9 to 22.5 req/s with the `--quick` sample of 32 states that fit the 32 KV slots.
- Eval latency includes the client's queue wait at concurrency 4 and is not comparable to the model-time columns of the card table.

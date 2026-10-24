# Optimized vLLM serving: Gemma 4 26B A4B IT (P150, 2026 Oct 24 resume)

Primary single-user **128 input / 128 output / 1 request / concurrency 1**, greedy, through the real
vLLM TT plugin (`tt/generator_vllm.py`), one Blackhole P300 chip as the P150 profile, 32 scheduler
slots, block size 64, `max_model_len` 131072, prefix caching on, trace mode `all`,
`sample_on_device_mode=all`, async scheduling. Before is the per-row prefill code (`4f3c2b13fa`);
after is this stage's final code (the commit recorded as "final code" in the work log, section "Final server and serving evidence (code after the second review)"; the final server ran that code).

| Workload | TTFT median ms, before -> after | TPOT mean ms, before -> after | ITL median / p99 ms, after | Output tok/s, after | Decode t/s/u (1000 / TPOT mean), before -> after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/128/1, C1 | 173.23 -> 173.20 | 27.945 -> 27.897 | 27.033 / 28.266 | 34.440 | 35.78 -> 35.85 |

Decode t/s/u is 1000 / mean TPOT, not aggregate throughput or a burst value. The before run was taken
on the live per-row server that the run environment provided (code `4f3c2b13fa`, two commits before the
stage base `067b6fc879`; those two commits touch the resumed-prefill driver and terminal, not the
128/128/1 path) and the same server supplied the burst serial reference, as the task defined it; the
prescribed A/B through the environment override on the final code was not repeated, and the direction
of that bias is conservative (FP32 accumulation makes the final per-row path slightly slower). The optimized-vLLM
decode path is unchanged in this resume; the work is prefill: prefix caching (resumed prefill),
batched prefill, and an expert-sorted MoE prefill. Device-time, roofline and profiler fields stay
null: no Tracy, tt-perf-report or device profiler ran against serving (see `perf_summary.json`).

The previous stage result on the 50,624-token P150 context is archived as
[README_20260908.md](README_20260908.md) and [STAGE_REVIEW_20260908.md](STAGE_REVIEW_20260908.md).

## Batched prefill burst (stage 10 gate)

`prefill_burst`: 32 prompts x 128 tokens, 32 output tokens, concurrency 32, greedy, same server
configuration. Serial reference measured on the per-row server before any change.

| Path | TTFT p50 ms | TTFT p99 ms | TTFT mean ms | p99 ratio | mean ratio | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| serial (per-row prefill, `vllm_prefill_burst_serial_benchmark.json`) | 3645.5 | 6712.7 | 4981.8 | 1 | 1 | reference |
| batched dense pass only (first batched run) | 2463.1 | 4685.6 | 3434.1 | 0.698 | 0.689 | fail (0.6 / 0.5) |
| batched + expert-sorted MoE on `45376c17e2` (run 1) | 1061.6 | 1907.9 | 1431.4 | 0.284 | 0.287 | pass |
| batched + expert-sorted MoE, shape-stable groups on `27c84d5e4d` | 1121.5 | 2027.3 | 1517.4 | 0.302 | 0.305 | pass |
| final code (slot-row assembly, sampler ceiling 32; `vllm_prefill_burst_benchmark.json`) | 1123.0 | 2032.6 | 1520.3 | 0.303 | 0.305 | pass |

Secondary CI serving burst **100/100/32**, unbounded client concurrency (vLLM-nightly parity and
capacity context, not headline decode):

| Phase | TTFT mean / p99 ms | TPOT mean / p99 ms | ITL median / p99 ms | Output tok/s |
| --- | ---: | ---: | ---: | ---: |
| before (per-row) | 10277.4 / 14089.3 | 114.288 / 174.934 | 74.357 / 121.513 | 148.19 |
| after (final code) | 2304.3 / 3093.3 | 87.674 / 103.371 | 78.467 / 125.010 | 291.27 |

## What changed and why

1. **Resumed prefill was numerically wrong, and the cause was the matmuls, not the cache.**
   The generator-level resume check (a cache hit, chunked continuation and preemption resume all
   enter `prefill_forward` with `start_pos > 0`) failed with top-1 agreement 0.28 to 0.66 on the
   30-layer model while every layer kind passed in isolation. The automatic `ttnn.linear` program
   config changes its K-blocking with the physical row count M, and with BF16 packer accumulation
   each block boundary rounds, so a continuation computed at M = suffix length produced QKV outputs
   that differed from the fresh forward in 83 percent of elements; the MoE stack amplified that.
   Prefill attention linears and the dense MLP now accumulate in FP32 (`GEMMA4_PREFILL_FP32_ACC`),
   the chunked and sliding continuation attention align to the configured 128-token SDPA chunk grid,
   and one SDPA compute config serves every prefill attention call. Result: K/V bit-identical
   between arms, resume check 14 of 14 cases at L = 2560 and 16384 with PCC min 0.9999, top-1 1.0,
   both negative controls failing (`readiness/prefill_resume_check.json`). FP32 accumulation costs
   about 0.9 ms per layer per matmul at 2048 rows and nothing measurable at 128 rows.
2. **Batched prefill.** Rows that start at position 0 and fit a bucket (128, 256, 512, 1024) run the
   dense work once as `[1, 1, B x S, H]`, with per-row cache fill, one batched SDPA and an index-gather
   terminal; the row count pads to a power of two and all 14 (B, S) variants are prepared before the
   first decode trace (`Gemma4Generator.prepare_batched_prefill`, called from the adapter's
   `warmup_model_prefill`). A prepared group adds no programs afterwards: the rows are split by
   shape, each row's cache fill is cut to its block-aligned length along the sequence only, and
   `paged_fill_cache` takes the row index as a tensor (acceptance: zero growth in every group of 2 to
   16 rows; served 31-row group: 5129 entries before and after). When a prefill step mixes a batched
   group with single rows, the terminal input is assembled from hidden rows instead of slicing the
   group's 262,144-wide logits per row: every source gathers its last hidden rows into their request
   slots of one fixed `[1, 1, 32, 2816]` tile with the runtime-index gather, a slot mask zeroes the
   other slots, the sources are summed, and one terminal runs over the sum. The per-row logits slice
   it replaces compiled one program per row index after warmup and untilized the full logits per row
   (found by the allocation tracker, second review). Continuations and rows above 1024 tokens keep the single-row path. Batched
   equals solo on the full model for 2 to 16 rows: top-1 equal on every row, PCC min 0.99998, K/V
   pages bit-identical, 32-token greedy continuations identical through 8 tokens
   (`readiness_vllm/vllm_batched_equals_solo.json`). 32 rows cannot be allocated in the
   generator-owned cache layout on one chip (6.7 GB of sliding windows); serving covers 32 slots.
3. **Expert-sorted MoE prefill.** The dense pass alone gained little (calibration ratio 0.87 at
   16 x 128) because this prefill is dominated by the MoE expert path, which computed the per-group
   expert union (nearly all 128 experts for every 128-token group). For prefills of 512 rows or more
   the tokens are sorted by expert on the host, gathered on device, and `sparse_matmul` runs with a
   one-hot expert per 128-row group and a compact output, so only routed experts are computed; the
   combine reduces the eight routed outputs in ascending expert order with the union path's ops. The
   output is bit-identical to the union path at every length measured, and the MoE time per layer
   drops from 99.8 to 20.2 ms at 2048 rows (4.95x) and from 25.3 to 17.3 ms at 512 rows
   (`doc/vllm_integration/continuation_bisect/sorted_moe_layer5.json`). Below 512 rows the union
   path is kept, so the 128-token single-user prefill is unchanged.

4. **Device sampler ceiling corrected to 32.** The adapter declared `max_device_top_k` 64 and the
   sampler was built with `max_top_k=64`, but `ttnn.sampling` draws from one 32-candidate tile per
   user: a device-level probe (`doc/optimized_vllm/sampler_probe/sampling1d_topk_probe.json`) shows
   identical draws for k = 64 and k = 32 and no sampled rank above 31, and the op's writer kernel
   reads past the user's row for k > 32. On the served path that overrun returned out-of-vocabulary
   token ids for slot 31 under the sampling suite's temperature-5 batch tests (second review P1;
   reproduced with the adapter's new range check: five ids, all in slot 31, all with k = 64). The
   ceiling is now 32 (`GEMMA4_DEVICE_MAX_TOP_K`); requests without `top_k` keep running on device at
   the 32 candidates the device computed before, and stochastic requests with `top_k` 33..64 go to
   the plugin's host sampler. Every sampled vector is range checked on the host and a violation is
   logged as `GEMMA4_TRACE_EVENT invalid_sampled_token`.

Environment knobs added: `GEMMA4_BATCHED_PREFILL` (default 1), `GEMMA4_BATCHED_PREFILL_BUCKETS`
(128,256,512,1024), `GEMMA4_BATCHED_PREFILL_MAX_TOKENS` (4096), `GEMMA4_BATCHED_PREFILL_MIN_ROWS` (2),
`GEMMA4_PREFILL_FP32_ACC` (1), `GEMMA4_OPT_PREFILL_SDPA_FP32_ACC` (1), `GEMMA4_SORTED_MOE_PREFILL` (1),
`GEMMA4_SORTED_MOE_MIN_ROWS` (512), `GEMMA4_SORTED_MOE_CHUNK_ROWS` (2048), `GEMMA4_DEVICE_MAX_TOP_K`
(32), and the diagnostic `GEMMA4_SORTED_MOE_PROFILE` (default off). The batched prefill path uses no
`corruptible_allocation_scope`: a tracked replay run with the tracker on and program-cache
allocations included finds nothing alive at replay once the programs are warm (see the evidence
table), so nothing needed an exemption. None is read by the plugin; the plugin and vLLM checkouts
are unchanged at `f96662a7d`. The adapter no longer declares the two capability keys the pinned plugin
does not read (`supports_async_decode_overlap`, `state_slots_are_stateless`).

## Serving correctness evidence (final server)

Final server: relaunched 17:45 UTC on the final code (runner pid 1034593), healthy at 17:48:30,
warmed with two single and one paired chat request, then every stage in sequence on the same server
(`readiness_vllm/P150_prefill_features/*_runner_final3.log`, server log
`readiness_vllm/P150_prefill_features/server.log`; the `27c84d5e4d` server's log, including the
review-era probes, is `server_27c84d5e4d_final_and_probes.log`, and the top-k 64 experiment server's
log is `server_topk64_probe.log`):

| Check | Result | Artifact |
| --- | --- | --- |
| Full plugin sampling suite (repo runner, `--sampling-profile full`, 32 slots) | 72 passed, 1 skipped (configured all-vocabulary logprobs skip), 5 min 29 s | `readiness_vllm/P150_prefill_features/sampling_tests.log` |
| Qualitative greedy and sampled outputs, degeneracy check | 6 chat prompts coherent; "No degenerate output detected"; greedy haiku identical to the earlier P150x2 output | `readiness_vllm/vllm_qualitative_outputs.json` |
| Served prefix cache (2305-token prompt cold/warm/reset, long 257-token resume, shared-prefix bench) | pass: cold TTFT 1474 ms, warm 136 to 149 ms (ratio 0.094, hit 0.9996), long resume first token and 16 greedy tokens identical (hit 0.899), shared-prefix hit 0.824 | `readiness_vllm/prefix_cache_check.json`, `vllm_run_summary.json` |
| Generator-level resume check (current runner, natural text, L = 2560 and 16384) | pass 14/14, PCC min 0.9999, top-1 1.0, both controls fail at every start | `readiness/prefill_resume_check.json` |
| Batched equals solo (full model, 2 to 16 rows) | pass | `readiness_vllm/vllm_batched_equals_solo.json` |
| Batched group program-cache stability (full model, groups of 2 to 16 rows, all buckets, mixed fill lengths) | pass: zero program-cache growth in every batched group after warmup | `readiness_vllm/batched_prefill_acceptance_final.log` |
| Served 32-slot group equals solo (32 chat prompts of 20 to 30 tokens sent at once, then each alone, greedy, 8 tokens) | pass: one batched group of 31 rows padded to 32 (the engine admits the first arrival in its own step), first greedy token equal on all 32 rows, 28 of 32 continuations identical through 8 tokens (the other 4 split at token 3 to 8 on the batch-1 versus 32-slot decode lane). On the final server the two releases in the burst were `prefill_program_cache_growth` from the first-seen single-row prompt lengths that preceded the group; on the `27c84d5e4d` server, where those lengths were already warm, the group ran with 5129 program-cache entries before and after it | `readiness_vllm/trace_reuse/batched_group_equivalence.json` |
| Non-aligned prompt lengths through serving | 129, 130, 131, 257, 300, 2200, 3100, 4500 tokens all completed (mixed-shape workload); 2305 (prefix-cache stage) | `readiness_vllm/trace_reuse/mixed_shape_check.json` |
| Mixed-shape trace reuse | fail (open item below) with four recorded reasons: 16 decode trace captures and 15 releases during 32 requests (`prefill_program_cache_growth` 10, `slot_remap` 4, `batch_shape_change` 1); the greedy 129-token raw prompt gave one text in the batch-1 decode lane and another in the 32-slot lane (first token identical, prefill path identical; decode-width program configs, pre-existing); one raw completion tripped the check's word-repetition heuristic (raw-prompt control in the work log). The run now also forms batched groups next to live decode rows: eight short prompts fired while a long request decoded formed one 8-row group (TTFT 816 to 819 ms) and the paired admission a 2-row group. TTFT p50 456 ms, p95 4.2 s, p99 8.6 s (first-seen chunk lengths compile) | same, `readiness_vllm/trace_reuse/raw_prompt_control.json` |
| Sampled token ids in range (adapter range check on every sampled vector; detokenizer scan of the server log) | pass: 0 detokenizer errors and 0 adapter events over the whole final server log (3,212 lines: the five temperature-5 batch tests run first, the full sampling suite, and every other stage); on the `27c84d5e4d` server the same five tests gave 2 invalid ids and on the top-k 64 experiment server 5, all in slot 31 (`invalid_token_scan_27c84d5e4d.json`, `invalid_token_scan_topk64.json`, `invalid_token_scan_final3.json`) | `readiness_vllm/P150_prefill_features/sampling_subset_final3_window.txt`, `server.log` |
| Tracked replay of the batched path (`TT_METAL_TRACE_ALLOC_TRACKING=1`, program-cache allocations included, acceptance groups of 2 to 16 rows with mixed singles, decode traces live) | pass twice: with the (since removed) scopes still wrapping the batched forwards, and with no scope at all once the single-row programs are warm; status pass, zero program-cache growth in every group and every step, no buffer alive at any replay (`acceptance_tracked_scopes_on.log`, `acceptance_tracked_no_scopes.log`). The only allocation the tracker ever flagged in the batched path after the row-assembly fix was a first-seen per-length program compile for a single row (`acceptance_tracked_scopes_off_gather_unwarmed.log`), the class the adapter retires traces for | `doc/optimized_vllm/trace_reuse_probe/batched_prefill_tracked/`, `..._scopes_off/` |
| Plugin and vLLM checkout | unchanged at `f96662a7d`, working tree clean; the inherited capability key `supports_prefix_caching_sliding_window` (plugin commit `cd25e6602`) is `proposed` for operator authorization | `doc/vllm_integration/plugin_changes.md` |


## Trace lifetime (open item, unchanged in this resume)

The adapter still releases and recaptures the decode traces on a B1/B32 execution-width change, on
host-mode decode, on a slot remap, after any prefill that compiled new programs, and on every
sampling-spec key miss (a greedy-to-sampled or sampled-to-greedy change, or a change of request
seeds, recaptures all traces: `Gemma4Generator._get_or_capture_decode_trace`, pre-existing at the
stage base). On the final server log the release reasons were `prefill_program_cache_growth` 19, `slot_remap` 16, `host_sampling_prefill` 10, `batch_shape_change` 3 (49 captures over 486 events; no sampling-spec key miss occurred in that run, while the `27c84d5e4d` server log had 27 of them). The task for
this resume excluded a trace-lifetime redesign. What was observed on the final server is recorded in
`readiness_vllm/trace_reuse/mixed_shape_check.json` and the work log; the warmed batched buckets add
no programs after preparation, and the per-length programs of the single-row path remain the source of
`prefill_program_cache_growth` releases.

## Reproduction

Server (from the worktree, environment as in the work log):

```bash
python -m models.common.readiness_check.run_vllm_server --stages serve \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it --hf-model google/gemma-4-26B-A4B-it \
  --mesh-device N150 --max-num-seqs 32 --block-size 64 --max-model-len 131072 --port 8000 \
  --output-subdir P150_prefill_features --tt-config '{"trace_region_size":220000000}' \
  --additional-server-args '--async-scheduling --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --enable-prefix-caching'
```

with `VLLM_SERVER_DEV_MODE=1` and a non-shared `GEMMA4_TENSOR_CACHE_PATH`. Stages attach with
`--server-url http://127.0.0.1:8000 --server-log <server.log>`; `prefill_burst` takes
`--prefill-burst-serial-result readiness_vllm/vllm_prefill_burst_serial_benchmark.json`; the sampling
suite runs through the repo copy `models.common.readiness_check.run_vllm_server --stages sampling
--sampling-profile full` because the installed runner rejects this nested plugin layout. Generator-level
checks: `readiness_check.run_prefill_resume_check --lengths 2560,16384 --block-sizes 64,128
--sliding-window 1024` and `tests/run_batched_prefill_acceptance.py --phase both --batches 2,4,8,16
--max-batch-size 16 --max-seq-len 32768`, one chip each. The tracked replay run is the acceptance
with `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
TT_METAL_TRACE_ALLOC_TRACEBACKS=1` and `--phase acceptance --warm-lengths`. Sampler checks:
`tests/probe_sampling1d_topk.py` (device, no model), `tests/run_vllm_invalid_token_probe.py`
(served) and `tests/scan_server_log_tokens.py` (server-log scan, also run at the end of the serving
chain).

| Evidence | Location |
| --- | --- |
| Resume check, superseded first run, bisection probes | `../../readiness/prefill_resume_check.json`, `../vllm_integration/continuation_bisect/` |
| Batched equals solo, calibration, acceptance logs | `../../readiness_vllm/vllm_batched_equals_solo.json`, `batched_prefill_calibration.json`, `batched_prefill_acceptance_*.log` |
| Burst serial reference and batched run | `../../readiness_vllm/vllm_prefill_burst_serial_benchmark.json`, `vllm_prefill_burst_benchmark.json` |
| Before/after single-user and CI burst | `../../readiness_vllm/optimized_vllm_batched_prefill/{before,after}/` |
| Sampling, qualitative, prefix cache, run summary | `../../readiness_vllm/P150_prefill_features/sampling_tests.log`, `../../readiness_vllm/vllm_qualitative_outputs.json`, `prefix_cache_check.json`, `vllm_run_summary.json` |
| Mixed-shape trace reuse observation, served 32-slot group equivalence | `../../readiness_vllm/trace_reuse/mixed_shape_check.json`, `batched_group_equivalence.json` |
| Sampler ceiling evidence (device probe, served probes, subset reproductions, log scans) | `sampler_probe/`, `../../readiness_vllm/P150_prefill_features/invalid_token_probe*.json`, `sampling_subset_temperature*.log`, `invalid_token_scan_*.json` |
| Tracked replay runs of the batched path | `trace_reuse_probe/batched_prefill_tracked/`, `trace_reuse_probe/batched_prefill_tracked_scopes_off/` |
| Decisions, failed attempts, commands, commit SHAs | [work_log.md](work_log.md) |
| Independent review of this attempt | [STAGE_REVIEW.md](STAGE_REVIEW.md) (the 2026 Sep 08 review is archived as STAGE_REVIEW_20260908.md) |

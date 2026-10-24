# Optimized vLLM serving: Gemma 4 26B A4B IT (P150, 2026 Oct 24 resume)

Primary single-user **128 input / 128 output / 1 request / concurrency 1**, greedy, through the real
vLLM TT plugin (`tt/generator_vllm.py`), one Blackhole P300 chip as the P150 profile, 32 scheduler
slots, block size 64, `max_model_len` 131072, prefix caching on, trace mode `all`,
`sample_on_device_mode=all`, async scheduling. Before is the per-row prefill code (`4f3c2b13fa`);
after is this stage's final code (the commit recorded as "final code" in the work log, section "Final server and serving evidence (code after the second review)"; the final server ran that code).

| Workload | TTFT median ms, before -> after | TPOT mean ms, before -> after | ITL median / p99 ms, after | Output tok/s, after | Decode t/s/u (1000 / TPOT mean), before -> after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/128/1, C1 | 173.23 -> 174.63 | 27.945 -> 27.933 | 27.021 / 28.073 | 34.385 | 35.78 -> 35.80 |

The after row is the third of three repeats taken right after the stage chain on the final server;
all four single-user measurements on that server (same command, `readiness_vllm/optimized_vllm_batched_prefill/after*/`):

| Run | TTFT ms | TPOT mean ms | ITL p50 / p99 ms | Output tok/s | Decode t/s/u |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/128/1, C1 (chain run) | 179.61 | 31.899 | 27.059 / 110.986 | 30.252 | 31.35 |
| 128/128/1, C1 (rep2) | 173.89 | 27.957 | 27.034 / 28.146 | 34.365 | 35.77 |
| 128/128/1, C1 (rep3) | 174.05 | 27.917 | 27.035 / 27.876 | 34.410 | 35.82 |
| 128/128/1, C1 (rep4) | 174.63 | 27.933 | 27.021 / 28.073 | 34.385 | 35.80 |

The chain run's TPOT (mean 31.90 ms over a 27.06 ms median stream, ITL standard deviation 27.98 ms
against about 10 ms in the before run and the repeats) carries about 0.6 s of stalled intervals,
at least one of them in the 300 ms class; its cause is not established. The three repeats have ITL
p99 of 27.9 to 28.1 ms and TPOT within 0.1 percent of the before run. Decode t/s/u is 1000 / mean TPOT, not aggregate throughput or a burst value. The before run was taken
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
| final code, no served top_k override (17:45 server; run superseded) | 1123.0 | 2032.6 | 1520.3 | 0.303 | 0.305 | pass |
| final code with the served `top_k` 32 default (final, `vllm_prefill_burst_benchmark.json`) | 1566.2 | 2492.0 | 1961.6 | 0.371 | 0.394 | pass |

Secondary CI serving burst **100/100/32**, unbounded client concurrency (vLLM-nightly parity and
capacity context, not headline decode):

| Phase | TTFT mean / p99 ms | TPOT mean / p99 ms | ITL median / p99 ms | Output tok/s |
| --- | ---: | ---: | ---: | ---: |
| before (per-row) | 10277.4 / 14089.3 | 114.288 / 174.934 | 74.357 / 121.513 | 148.19 |
| after (final code with the served `top_k` 32 default, chain run) | 2334.3 / 3123.7 | 87.746 / 103.587 | 78.391 / 127.176 | 290.29 |
| after (same server, repeat 4; repeats 2 and 3 within 0.5 percent on TTFT, TPOT and throughput and within 0.9 percent on ITL p99; each repeat had its first 21-row prefill step host-sampled by the plugin, see the serving evidence section) | 2258.6 / 3005.7 | 87.227 / 102.616 | 78.203 / 127.330 | 293.68 |

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
   16 rows; served 31-row group on the `27c84d5e4d` server: 5129 entries before and after; on the final server 4973). When a prefill step mixes a batched
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
   ceiling is now 32 (`GEMMA4_DEVICE_MAX_TOP_K`), and every sampled vector is range checked on the
   host (a violation is logged as `GEMMA4_TRACE_EVENT invalid_sampled_token`). Consequence found by
   the third review: the model's `generation_config.json` sets `top_k: 64`, and vLLM applies it to
   every sampled request that does not name `top_k`, so under the 32 ceiling the pinned plugin routes
   the default recipe to its host sampler (full-vocabulary logits readback, eager decode, every
   decode trace released at the prefill). Measured on the server without a served override
   (`readiness_vllm/P150_prefill_features/default_recipe_latency_ceiling32_no_override.json`,
   128/128/1): a default-recipe sampled request decodes at 97.0 ms per token (greedy on device:
   27.1 ms) and drags a concurrent greedy user from 27 to 98 ms per token. The served default is
   therefore `--override-generation-config '{"top_k":32}'` in the reproduction command below, which
   keeps the default recipe on the device sampler at the 32 candidates the device sampler has
   always drawn from (identical draws for k = 64 and k = 32 in the device probe). This changes the
   served recipe's `top_k` from 64 to 32 and is recorded as a proposal pending operator acceptance;
   with the override in place the final evidence below comes from the device sampler
   (trace events attributed to each stage by request structure in the server log, since the generator's event lines carry no timestamp: the temperature-5 reproduction tests, 105 requests, 13 sampled 32-slot and 1 sampled batch-1 device captures, no host-sampling release; the full suite 23 sampled 32-slot and 12 sampled batch-1 device captures and 4 `host_sampling_prefill` releases from its logprobs and penalty tests). Stochastic requests that name `top_k` above 32 still use the plugin's host
   sampler.

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

Final server: relaunched 18:37 UTC on the final code with the served `top_k` 32 default (runner pid
1205092), healthy at 18:41:27, warmed with two single and one paired chat request, then every stage
in sequence on the same server (`readiness_vllm/P150_prefill_features/*_runner_final4.log`, server
log `readiness_vllm/P150_prefill_features/server.log`). Earlier servers' logs: `27c84d5e4d` with the
review-era probes `server_27c84d5e4d_final_and_probes.log`; the top-k 64 experiment
`server_topk64_probe.log`; the final code without the served override (17:45 UTC, `*_runner_final3.log`)
`server_final3_ceiling32_default_recipe64.log`. Sampling modes over the whole final log
up to the end of the chain (`invalid_token_scan_final4.json`, 4,086 lines): 116 device sampled captures, 67
device greedy captures, 6 `host_sampling_prefill` releases (4 in the suite: its host-only-parameter
and logprobs tests, the 32-request structured-output test and the penalty tests; 1 from the
prefix-cache stage's cold request; 1 from the `top_k` 50 phase of the k32 probe), and 0
`host_sampling_decode` release events. Those are release events, which the adapter emits only while
a decode trace is alive, so they undercount host-sampled steps: the plugin host-samples every
request that asks for logprobs on a single device, so the seven prefix-cache requests (`logprobs 1`)
decoded on the host at 52 to 53 ms per token, as did the k32 probe's `top_k` 50 phase; the
device-sampled cache-hit path is covered by the mixed-shape `prefix_resume_2200` request (TTFT 210 ms
against 3461 ms cold, identical 16-token greedy text) and the greedy shared-prefix bench. The plugin
also host-samples any prefill step that carries an intermediate chunk (prefill-features.md), so the
CI burst's first 21-row step (twenty 100-token rows plus one 48-token partial chunk at the 2048-token
budget) was host-sampled in the chain run and in all three repeats; the release line is visible
only in the repeats because in the chain run a first-seen 100-token single row had already released
the traces one step earlier. The single-user runs are greedy batch-1 device captures with no host
release:

| Check | Result | Artifact |
| --- | --- | --- |
| Full plugin sampling suite (repo runner, `--sampling-profile full`, 32 slots) | 72 passed, 1 skipped (configured all-vocabulary logprobs skip), 5 min 29 s | `readiness_vllm/P150_prefill_features/sampling_tests.log` |
| Qualitative greedy and sampled outputs, degeneracy check | 6 chat prompts coherent; "No degenerate output detected"; greedy haiku identical to the earlier P150x2 output | `readiness_vllm/vllm_qualitative_outputs.json` |
| Served prefix cache (2305-token prompt cold/warm/reset, long 257-token resume, shared-prefix bench) | pass: cold TTFT 1473 ms, warm 136 to 148 ms (ratio 0.094, hit 0.9996), long resume first token and 16 greedy tokens identical (hit 0.899), shared-prefix hit 0.824 | `readiness_vllm/prefix_cache_check.json`, `vllm_shared_prefix_benchmark.json` |
| Generator-level resume check (current runner, natural text, L = 2560 and 16384) | pass 14/14, PCC min 0.9999, top-1 1.0, both controls fail at every start | `readiness/prefill_resume_check.json` |
| Batched equals solo (full model, 2 to 16 rows) | pass on the final code (8 cases, top-1 equal and K/V equal on every row, zero program-cache growth in every group and step) | `doc/optimized_vllm/trace_reuse_probe/batched_prefill_tracked_scopes_off/vllm_batched_equals_solo.json` (final code, tracker on); `readiness_vllm/vllm_batched_equals_solo.json` is the `27c84d5e4d` run before the slot-row assembly |
| Batched group program-cache stability (full model, groups of 2 to 16 rows, all buckets, mixed fill lengths) | pass: zero program-cache growth in every batched group after warmup, on `27c84d5e4d` and on the final code | `readiness_vllm/batched_prefill_acceptance_final.log`, `doc/optimized_vllm/trace_reuse_probe/batched_prefill_tracked*/` |
| Served 32-slot group equals solo (32 chat prompts of 20 to 30 tokens sent at once, then each alone, greedy, 8 tokens) | pass: one batched group of 31 rows padded to 32 (the engine admits the first arrival in its own step), first greedy token equal on all 32 rows, 28 of 32 continuations identical through 8 tokens (the other 4 split at token 3 to 8 on the batch-1 versus 32-slot decode lane; same rows in every run). Program-cache entries 4973 before the group and at the capture after it (no growth); the burst's releases were one `prefill_program_cache_growth` from a first-seen single-row prompt length before the group and one `slot_remap` | `readiness_vllm/trace_reuse/batched_group_equivalence.json` |
| Non-aligned prompt lengths through serving | 129, 130, 131, 257, 300, 2200, 3100, 4500 tokens all completed (mixed-shape workload); 2305 (prefix-cache stage) | `readiness_vllm/trace_reuse/mixed_shape_check.json` |
| Mixed-shape trace reuse | fail (open item below) with four recorded reasons: 15 decode trace captures and 14 releases during 32 requests (`prefill_program_cache_growth` 9, `slot_remap` 3, `batch_shape_change` 2); the greedy 129-token raw prompt gave one text in the batch-1 decode lane and another in the 32-slot lane (first token identical; decode-width program configs, pre-existing); raw mid-paragraph completions loop or drift (`batched_group_while_decoding` rows). Control for the batched-group rows on the standalone generator (`readiness_vllm/trace_reuse/raw_prompt_control_batched_group.json`, same token ids, greedy): the standalone solo prefill, the standalone batched 9-row group and the served group agree on the first token for 9 of the 10 prompts; the tenth (`short_3`) is an exact tie between two tokens (top-1 minus top-2 logit 0.0) that the served step broke the other way; the standalone continuations loop the same way (`-로-로-로`, `and the and the`, `the emphasis on the emphasis on`), so the degeneracy is this model's greedy behaviour on these raw prompts; the continuations then split on the batch-1 versus 32-slot lane (3 of 8 identical through 16 tokens). The 8-row group formed while a 64-token request decoded (TTFT 800 to 803 ms for the shorts). TTFT p50 286 ms, p95 3.5 s, p99 7.9 s (first-seen chunk lengths compile) | `readiness_vllm/trace_reuse/mixed_shape_check.json`, `raw_prompt_control.json`, `raw_prompt_control_batched_group.json` |
| Sampled token ids in range (adapter range check on every sampled vector; detokenizer scan of the server log) | pass: 0 detokenizer errors and 0 adapter events over the whole final server log, which includes the five temperature-5 batch tests run first (device sampler, 13 sampled 32-slot and 1 sampled batch-1 captures over their 105 requests, no host-sampling release), the full sampling suite, and a 4,271-token temperature-5 `top_k` 32 probe at 32 and 37 concurrent requests with slot 31 live (55 sampled B32 captures, `invalid_token_probe_endings_k32.json`); on the `27c84d5e4d` server the same five tests gave 2 invalid ids and on the top-k 64 experiment server 5, all in slot 31 (`invalid_token_scan_27c84d5e4d.json`, `invalid_token_scan_topk64.json`, `invalid_token_scan_final4.json`) | `readiness_vllm/P150_prefill_features/sampling_subset_final4_window.txt`, `server.log` |
| Tracked replay of the batched path (`TT_METAL_TRACE_ALLOC_TRACKING=1`, program-cache allocations included, acceptance groups of 2 to 16 rows with mixed singles, decode traces live) | pass twice: with the (since removed) scopes still wrapping the batched forwards, and with no scope at all once the single-row programs are warm; status pass, zero program-cache growth in every group and every step, no buffer alive at any replay (`acceptance_tracked_scopes_on.log`, `acceptance_tracked_no_scopes.log`). The only allocation the tracker ever flagged in the batched path after the row-assembly fix was a first-seen per-length program compile for a single row (`acceptance_tracked_scopes_off_gather_unwarmed.log`), the class the adapter retires traces for | `doc/optimized_vllm/trace_reuse_probe/batched_prefill_tracked/`, `..._scopes_off/` |
| Plugin and vLLM checkout | unchanged at `f96662a7d`, working tree clean; the inherited capability key `supports_prefix_caching_sliding_window` (plugin commit `cd25e6602`) is `proposed` for operator authorization | `doc/vllm_integration/plugin_changes.md` |


## Trace lifetime (open item, unchanged in this resume)

The adapter still releases and recaptures the decode traces on a B1/B32 execution-width change, on
host-mode decode, on a slot remap, after any prefill that compiled new programs, and on every
sampling-spec key miss (a greedy-to-sampled or sampled-to-greedy change, or a change of request
seeds, recaptures all traces: `Gemma4Generator._get_or_capture_decode_trace`, pre-existing at the
stage base). On the final server log the release reasons were ``slot_remap` 116, `trace_key_miss` 25, `prefill_program_cache_growth` 23, `batch_shape_change` 12, `host_sampling_prefill` 6 (183 captures over 837 events; the sampling-spec key misses are the greedy-to-sampled switches of the suite and the qualitative stage, now on the device sampler). The task for
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
  --additional-server-args "--async-scheduling --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --enable-prefix-caching --override-generation-config '{\"top_k\":32}'"
```

The runner splits `--additional-server-args` with `shlex`, so the JSON of the override keeps its own
quotes inside the argument string (the first launch attempt without them failed at vLLM argument
parsing, `readiness_vllm/P150_prefill_features/server_final4_failed_launch.log`).

The `--override-generation-config` entry is the proposed served default (see item 4 above); the
`27c84d5e4d`, `af32f9ea86`-without-override and earlier servers ran without it.

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

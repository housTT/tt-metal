# Qwen3-8B optimized vLLM

Status: `Qwen/Qwen3-8B` serves through the TT vLLM plugin on one Blackhole
P300 chip (1x1 mesh, p150 mesh graph descriptor; single chip is the target
topology, stages 4 and 5 were skipped by the operator) from the autoport
adapter `tt/generator_vllm.py` with batched prefill, device penalties, the
stage-7 1D multicast O and down prefill kernels, prefix caching, async
decode and on-device split sampling. The 32 x 128 prefill burst reaches a
TTFT ratio of 0.39 (p99 and mean) against the stage-9 serial reference
(gate 0.6 and 0.5), the primary single-user TTFT is 10 percent lower than
stage 9 at unchanged TPOT, batched prefill equals solo prefill bit for bit,
and the served workload adds no program-cache entry. Stage gate and stage
review: sections 8 and 9.

Abbreviations: APC (automatic prefix caching), HF (Hugging Face), KV
(key/value), LM (language model), PCC (Pearson correlation coefficient),
DRAM (device dynamic RAM), L1 (per-core local memory), TTFT (time to first
token), TPOT (time per output token), ITL (inter-token latency), e2e (end
to end), t/s/u (tokens per second per user), BF16 (bfloat16), BFP4/BFP8
(block floating point), LoFi (low fidelity), SDPA (scaled dot-product
attention), RoPE (rotary position embedding), CI (continuous integration),
ET (Eastern Time).

## 1. Headline serving numbers (chip 1, final code, 17:36 to 17:38 ET Oct 30)

Primary single-user profile, `vllm bench serve` 128-token random prompt,
128 output tokens, 1 request, `--max-concurrency 1`, greedy
(`--temperature 0.0`, `generation_config_mode explicit`), `ignore_eos`;
server `--max-num-seqs 32 --max-model-len 40960 --block-size 32`,
`--enable-prefix-caching --async-scheduling`, TT config
`{"sample_on_device_mode": "all", "trace_region_size": 536870912}`
(`readiness_vllm/vllm_benchmark.json`, raw `vllm_result.json`; before =
stage 9's final server with the same launch line,
`served_stage9_final/vllm_benchmark.json`):

| Metric (128/128/1) | Stage 9 (before) | Stage 10 (after) |
| --- | ---: | ---: |
| TTFT p50 / p99 | 49.3 / 49.3 ms | 44.6 / 44.6 ms (-9.5 percent) |
| TPOT mean / p99 | 20.71 / 20.71 ms | 20.71 / 20.71 ms (-0.04 percent) |
| ITL p50 / p99 | 20.67 / 21.47 ms | 20.68 / 21.53 ms |
| Decode t/s/u (1000 / mean TPOT) | 48.28 | 48.29 |
| Aggregate output throughput | 47.75 tok/s | 47.86 tok/s |

Single-user TTFT and TPOT are within the 5 percent envelope (TTFT is 4.7
ms lower from the carried 1D O and down prefill kernels, section 3.2;
TPOT equal). Regime: concurrent, measured while chip 0 ran the tracked
full-model tests (17:29 to 17:53 ET) and chip 2 the resume check on the
same host, the same regime as stage 9's headline (its benchmark also ran
beside a chip-0 tracked run); TPOT equals the stage-8 and stage-9
single-host numbers, so the host-wall caveat of `tt-device-usage` is
recorded and the before/after comparison is like for like.

Generator-level lower bound (stage 8, teacher forcing, traced decode and
sampler): 47.24 t/s/u, TTFT 48.05 ms (158-token prompt). Serving decode is
2 percent faster than teacher forcing, as in stage 9; no serving overhead
is left in the measured decode path (every step replays the decode and
sampling traces with no token, position or page-table refresh;
`doc/full_model/host_work_counters.stage10.json`).

Secondary CI serving-burst profile (vLLM-nightly shape: 100-token prompts,
100 output tokens, 32 requests, no concurrency cap, greedy, `ignore_eos`;
`readiness_vllm/vllm_ci_serving_benchmark.json`):

| Metric (100/100/32) | Stage 9 | Stage 10 |
| --- | ---: | ---: |
| TTFT p50 / p99 | 1217.9 / 1218.7 ms | 477.5 / 478.3 ms |
| TPOT mean / p99 | 25.06 / 32.87 ms | 24.82 / 27.68 ms |
| ITL p50 / p99 | 25.08 / 25.90 ms | 25.14 / 25.91 ms |
| Aggregate output throughput | 873.2 tok/s | 1094.4 tok/s |
| t/s/u from mean TPOT | 39.9 | 40.3 |
| Completed | 32 / 32 | 32 / 32 |

Not a headline decode number: burst admission prefills the 32 prompts in
one batched pass (one 1-row call plus one 31-row call, section 2) and the
decode runs at batch 32.

Batched-prefill gate (`readiness_vllm/vllm_prefill_burst_benchmark.json`,
32 x 128-token prompts at concurrency 32, 32 output tokens, greedy):

| | Serial reference (stage 9, one row per pass) | Stage 10 batched | Ratio |
| --- | ---: | ---: | ---: |
| TTFT p50 | 1231.1 ms | 480.3 ms | 0.390 |
| TTFT p99 | 1231.9 ms | 481.1 ms | 0.391 (gate <= 0.6) |
| TTFT mean | 1194.5 ms | 467.0 ms | 0.391 (gate <= 0.5) |

`serial_reference.config_matches` true (same prompt and output lengths,
request count and concurrency; the reference is
`doc/optimized_vllm/vllm_prefill_burst_serial_benchmark.stage9.json`, a
copy of stage 9's `readiness_vllm/vllm_prefill_burst_serial_benchmark.json`,
passed as `--prefill-burst-serial-result`, as the operator brief
directs). That reference was measured with the stage-7 solo kernel table.
The served like-for-like comparison from the two server logs: stage 9's
128-token single-row calls took 41.1 to 41.2 ms (`doc/vllm_integration/served_final/server.log.gz`
calls 219, 237 and 239) and its burst's 31-row per-row-loop calls 1166.5
ms (calls 238 and 240); stage 10's single rows take 36.9 to 37.4 ms
(calls 205, 223 and 225; the 1D O and down kernels save 4.0 ms per
128-token prefill, which is the single-user TTFT change) and its 31-row
batched calls 422.0 ms (calls 224 and 226). An in-stage serial reference
would therefore sit near 31 x 37.1 ms plus the lone row and the plugin's
admission (about 1110 ms), and the ratios near 0.43, inside both gates;
the generator-level pair on the final code (`logs/batched_prefill_probe_L36_chip2_v2.json`:
32 x 128 serial call 1094.9 ms, batched 426.6 ms, ratio 0.39) is the
like-for-like control the stage measured. The generator-level pair on the final code
(`logs/batched_prefill_probe_L36_chip2_v2.json`: 32 x 128 serial call
1094.9 ms, batched 426.6 ms, ratio 0.39) shows the gate holds against a
like-for-like serial reference as well. The stage ran twice on the
same server (first run p99 ratio 0.393, mean 0.394, `logs/stage_prefill_burst.log`;
the JSON holds the repeat), 32 of 32 completed each time. The 31-row
batched call of the burst takes 422 ms in the server log (`prefill
call=224 rows=31 ... ends=[128, ...] ... ms=422.0 ...
batched_prefill_rows=31`, and call 226 at 421.8 ms in the repeat; call
207 at 421.7 ms is the CI burst's 31-row call of 100-token prompts padded
to the same 128 bucket), against 1095 ms for the same call with batched
prefill off at the generator level (section 3).

## 2. What was built

- Batched prefill (`tt/optimized_decoder.py` `prefill_forward_batched`,
  `tt/model.py` `prefill_hidden_batched`, `tt/generator.py`
  `_batched_prefill_groups`): the equal-padded-length design of
  `prefill-features.md`. Fresh rows (`start_pos` 0) of one `prefill_forward`
  call are grouped by their prefill bucket (32, 64, 128, 256, 512 tokens),
  padded to a row bucket (2, 4, 8, 16, 32 rows, dummy rows repeating the
  last real prompt with an all-scratch page-table row) and run as one
  `[1, 1, rows * bucket, H]` dense pass; `rows * bucket <= 4096` is always
  a prepared single-row bucket, so the dense programs exist before the
  first trace capture. Attention and the cache fill stay per row inside one
  op each: `nlp_create_qkv_heads` on the `[rows, 1, bucket, W]` view of the
  QKV output, one `[1, 1, bucket, D]` RoPE gather for every row, batched
  causal SDPA over `[rows, heads, bucket, D]`, `paged_fill_cache` with a
  `batch_idx_tensor` over the `[rows, blocks]` page table, `nlp_concat_heads`
  viewed back as `[1, 1, rows * bucket, heads * D]`. Padded tails sit after
  each row's real tokens (never read by causal attention; written to the
  row's own last block or the scratch block). Every other row of the call
  (nonzero start, bucket above 512, a singleton group) runs on the stage-9
  single-row path in the same call, and the output keeps the call's row
  order (one fixed 32-wide runtime gather when one group covers the call;
  otherwise per-row gathers and the stage-6 `rows_to_tile`). Program keys
  depend only on `(rows, bucket)` and `rows * bucket`; 22 shapes are warmed
  in `prepare()` (`warmup batched prefill R x S` lines in the server log).
  Override: `QWEN3_BATCHED_PREFILL=0` (environment, read by
  `build_generator`) runs every row on the single-row path.
- A bit-equal batched kernel table (`BATCHED_PREFILL_KERNEL_TABLE`,
  `_linear_batched`): the batched dense projections run only 2D and 1D
  multicast kernels, because `minimal_matmul` orders its K accumulation
  differently and flips rare bf16 roundings against the single-row kernels
  (section 3.1). Wide outputs that do not fit one 2D block (gate and up,
  N = 12288; QKV and O/down above 2048 rows) run as row pieces through the
  batch dimension of one `ttnn.linear` (`fuse_batch` off): QKV 2D/16 in
  1024-row pieces, O and down 2D/16 in 2048-row pieces, gate 1D (110
  cores, `in0_block_w` 8, 1x4) and up 2D/16 in 512-row pieces.
- Solo prefill kernel table `stage10` (`PREFILL_KERNEL_TABLES`, the
  default; `QWEN3_PREFILL_KERNEL_TABLE=stage7` restores stage 7): gate at
  512 rows moves from `minimal_matmul` to 1D (110, 8, 1x4) (0.345 against
  0.337 ms per layer, so 512-token groups stay bit equal), and the carried
  stage-7 review item lands: O and down at 64/128/256 rows on the 1D
  multicast kernels (O: (110, 4), (96, 8), (64, 16); down: (96, 4), (64, 8),
  (110, 16), all 2x2 subblocks), which take the 128-token token-out TTFT
  from 42.1 to 37.9 ms at the generator level.
- Device penalties (`supports_device_penalties: True`): the common
  `SamplingGenerator`'s presence, frequency and repetition penalties run on
  device; `prepare()` precompiles the penalties-on sampler variant per
  decode bucket (with one counted pass, because the sampler's `precompile`
  skips the output-count update), `capture_decode_traces` captures and
  executes a penalties-on sampling trace per bucket beside the greedy one
  (server log `'1/device+penalties': 'sampling'`), the prompt mask is built
  at prefill from the call's tokens and reloaded with the output counts
  from the plugin's `prompt_tokens` / `output_tokens` on reload steps, and
  steady steps count the sampled token inside the sampling trace. Penalised
  batches therefore stay on the device sampler (189 penalised device steps
  in the final sampling suite, `penalized_sampling_steps=1` lines).
- Adapter log lines carry `batched_prefill_groups`,
  `batched_prefill_rows`, `single_row_prefill_rows` and
  `penalized_sampling_steps` per call; the stage-9 mixed-shape checker
  reads them unchanged.

## 3. Batched prefill evidence

### 3.1 Batched equals solo (bit for bit)

- Generator level, 36 layers, chip 0 (`doc/full_model/batched_prefill_equals_solo_*.stage10.json`
  and `*.stage10_tracked.json`, `logs/pytest_full_model_36_stage10_chip0.log`):
  mixed lengths 5, 31, 32, 33, 127, 128, 129, 257, 640, 1000 and 1024 in
  both orders (groups 32 x {5, 31, 32} and 128 x {127, 128} batched beside
  six single-row rows in the same call), uniform calls of 2, 4, 8 and 32
  rows of 65 to 128 random tokens, and eight chat prompts: every row's
  last-token logits and its K/V pages of layers 0 and 35 at the real
  positions are bit equal to the single-row result (PCC 0.99999 to 1.0,
  top-1 equal), the rerun and the reversed order are bit equal, and the
  program cache stays at 645 on the first call and the rerun.
- The 22 prepared shapes, 36 layers, chip 2
  (`logs/batched_prefill_probe_L36_chip2_v2.json`): bit equal in logits and
  K/V at every shape, program cache 645 before and after every call.
- Greedy continuation with the same decode batch
  (`doc/full_model/batch_decode_{8,16,24,32}.stage10.json`): the multi-row
  prefill (batched path for its groups) followed by 32 greedy decode steps
  equals the solo-prefill arm for every user over all 32 tokens.
- How the equality was reached: the first policy (`minimal_matmul` at 512
  dense rows and above, `logs/batched_prefill_probe_L36_chip2.json`) was
  fast (32x128 in 403 ms) but not bit equal above 256 dense rows (PCC
  minimum 0.9996, top-1 flips on random tokens). A per-op bisect of one
  layer (`logs/op_bisect_probe2_chip0.json`) showed every op bit equal
  except the down projection under `minimal_matmul` (one element in 2 of 32
  rows), and a retry on the recorded real input showed every 2D kernel at
  4096 rows reproducing the solo 2D kernel exactly. The kernel sweeps
  (`logs/dense_kernel_probe_chip0.json`, `logs/wide_kernel_probe_chip0.json`)
  then selected the fastest 2D/1D configuration per role and row count
  (work log section 4). Cost: 7.75 against 6.92 ms per layer at 4096 rows
  (32x128 batched call 427 against 403 ms at the generator level).

### 3.2 Calibration (36 layers, chip 2, median of 3; `logs/batched_prefill_probe_L36_chip2_v2.json`)

| Shape (rows x bucket) | Serial call (per-row loop) | Batched call | Gain |
| --- | ---: | ---: | ---: |
| 2 x 32 | 71.5 ms | 33.5 ms | 2.1x |
| 32 x 32 | 1060.9 ms | 114.2 ms | 9.3x |
| 32 x 64 | 976.8 ms | 217.7 ms | 4.5x |
| 2 x 128 | 72.9 ms | 48.6 ms | 1.5x |
| 8 x 128 | 277.8 ms | 112.1 ms | 2.5x |
| 32 x 128 | 1094.9 ms | 426.6 ms | 2.6x |
| 16 x 256 | 736.3 ms | 425.7 ms | 1.7x |
| 2 x 512 | 134.8 ms | 113.3 ms | 1.2x |
| 8 x 512 | 524.0 ms | 431.0 ms | 1.2x |

Every one of the 22 shapes wins, so all stay enabled (the full table is in
the work log and the JSON). Single-row prefill (the serial arm's rows)
after the stage-10 kernel table, served: 128-token rows 36.9, 37.4 and
37.1 ms (`prefill call=205`, `223`, `225`, `rows=1 ends=[128]`); the
generator-level solo arm of the 2x128 probe row is 39.0 ms per row
(`solo_sum_ms` 78.0 for two rows).

### 3.3 Served workload, program cache and tracker

- Mixed-shape trace reuse and request lifecycle on the final server
  (`readiness_vllm/mixed_shape_trace_reuse.json`, the stage-9 workload:
  unseen lengths 129, 130, 131, 5, 33, 257, 1025 solo then repeated, a
  concurrent burst of 31 to 4097 tokens, prefix-cache resumes at start 992
  and a 5000-token prompt during a 400-token decode, greedy -> sampled ->
  greedy -> seeded -> seeded -> greedy, the burst reordered, logprob and
  penalty requests beside a device-sampled one): 47 of 47 completed, 32
  prefill calls (30 rows with nonzero starts), 928 decode steps (905
  device-sampled, 23 host-sampled), 0 decode or sampling trace captures, 0
  trace releases, program cache 651 before, after and at maximum, DRAM
  31752 MB flat, every greedy repeat identical, same seed identical, 0
  out-of-vocabulary ids.
- Every prefill call of the final server (sampling suite, qualitative,
  benchmarks, prefix cache, bursts, mixed shape) reports
  `program_cache=651` (the count after warmup and capture); the 31-row
  burst calls are batched (`batched_prefill_rows=31`).
- Tracked serving (`TT_METAL_TRACE_ALLOC_TRACKING=1
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` in the engine environment,
  same launch line, final code; `logs/tracked_served/`): section 8 records
  the result of the smoke sampling subset, the prefix-cache stage, the
  32 x 128 burst (batched prefill followed by decode replays) and the
  mixed-shape workload on that server.
- Tracked generator-level gate (`logs/pytest_full_model_36_stage10_tracked_chip0.log`,
  `doc/full_model/*.stage10_tracked.json`): section 8.

## 4. Other served evidence (final server, chip 1, `logs/evidence_chain_chip1.status`)

| Check | Result | Artifact |
| --- | --- | --- |
| Sampling suite, full profile, `--tt-max-num-seqs 32` (73 tests) | 71 passed, 1 failed, 1 skipped in 126 s. Failed: `test_request_isolation.py::test_mixed_params_batch` (section 6); skipped: `test_chat_logprobs_all_vocab` (single chip). All six `test_tt_penalties.py` tests pass on the device sampler; the stage-9 failure `test_presence_penalty_mixed_batch` is closed by device penalties | `readiness_vllm/sampling_tests.log` |
| Qualitative (6 prompts, chat template, greedy and sampled, 256 tokens) | served greedy text token-identical to the stage-9 served greedy text for all 256 tokens on every prompt (`first_divergence_vs_stage9_served_greedy` 256); identical to the stage-9 generator control for its 128 tokens; first divergence from the HF greedy control at tokens 86, 37, 32, 39, 33, 22 (unchanged since stage 7); every output coherent, on topic, English, HF-like `<think>` structure, adjacent duplication 0.0, no control-token leakage; degenerate-output check: none | `readiness_vllm/vllm_qualitative_outputs.json`, `qualitative_verdict.json`, `logs/stage_degenerate.log` |
| Prefix cache (2305-token prompt, APC counters from `/metrics`) | pass; warm/cold TTFT ratio 0.180; first token and logprob identical, greedy identical through 16 tokens; long resume and shared-prefix bench pass | `readiness_vllm/prefix_cache_check.json`, `vllm_run_summary.json` (`apc_log_enabled` true) |
| Non-aligned prompt lengths and larger batches | the mixed-shape lengths above (5 to 5000 tokens, nonzero starts), the sampling suite at `max_num_seqs` 32, the CI burst at 32 concurrent requests | `readiness_vllm/mixed_shape_trace_reuse.json`, `vllm_ci_serving_benchmark.json` |
| Served max context | 40960 (`doc/context_contract.json` unchanged) | `readiness_vllm/vllm_run_summary.json` |
| Serving contract | 6 rows updated (batched prefill, multi-row prefill, device penalties state, `supports_device_penalties`, penalties, `prefill_forward`), checker OK, 37 rows | `doc/full_model/serving_contract.json` |

The served chain ran 17:32 to 17:38 ET; the final `readiness_vllm/` is
archived in `served_final/` (server log gzipped).

## 5. Carried items

| Item | Outcome |
| --- | --- |
| (a) 1D multicast O and down prefill kernels (stage-7 review-2 P2) | integrated in the `stage10` table (section 2); generator TTFT 42.1 -> 37.9 ms, served TTFT p50 49.3 -> 44.6 ms; continuation numerics unchanged (resume check identical to stage 9 at every start) |
| (b) device penalties | implemented and served (section 2); `test_tt_penalties.py` 6 of 6 on the device sampler, `test_presence_penalty_mixed_batch` passes |
| (c) device versus host sampler at temperature 0.01 | characterized (`logs/low_temperature_probe_chip2.json`, 36 layers): for the stage-9 prompts `Letter: `, `Random: ` and a third, the device sampler's first token equals the host argmax in 48 of 48 seeded draws at temperature 0.01 (top-2 logit gaps 0.14 to 0.60; bf16 scaling merges no leader), and at 0.7 the device draws follow the host distribution's rank order. The stage-9 `'198'` against `'100'` pair (the stage-9 instance of `test_mixed_params_batch`) differs in the second token: a seeded request whose prefill step was device-sampled and whose decode steps were host-sampled takes its first draw from the device stream and the rest from its host generator, one draw offset from a request host-sampled throughout; the two streams cannot agree by construction. Not a distribution difference. The stage-10 instances of the same test (different rows, temperatures 0.5 and 1.0) are in section 6 and work log 8.1 |
| (d) resume check at default thresholds on the final code | identical to stage 9 at every start: all PCC gates pass (min 0.9886), both controls fail at every window, top-1 fails only the 32-position hit window at 16352 (31 of 32, the stage-7 near tie); lowered run passes (`readiness/prefill_resume_check.json`); default run kept beside it (`readiness/prefill_resume_check_stage10_default_thresholds.failed.json`) |

## 6. Limitations and open items

- `test_mixed_params_batch` still fails (one row per run; full suite:
  request 0, `temp=0.5 seed=42`, a frequency-penalty prompt; smoke:
  request 2, `temp=0.5 seed=42`; tracked smoke: request 1, `temp=1.0
  seed=7`, the top-p prompt). Its batch holds a `top_k 100` request, above
  the device sampler's 32-candidate ceiling, so the plugin host-samples
  every step that contains it; the server log shows the 9-row group
  admitted device-sampled in one run (the `top_k 100` request was admitted
  alone before it) and host-sampled in the other, so every seeded request
  in the group takes its first draw on a different sampler and its stream
  is offset by one draw (work log 8.1, prefill calls 70 to 73 and 11 to
  14 with the decode-mode windows after them). Reproducibility-only:
  correctness, logprobs, crash-free serving and the qualitative suite pass
  on the same server. A fix needs a device top-k above 32 (the common
  sampler draws from one 32-candidate tile) or a plugin routing rule; the
  former is model work beyond this stage, the latter a plugin change.
  Carried to stage 11 (benchmarks) by name: `test_mixed_params_batch`,
  reproducibility-only, mechanism in work log 8.1; the next stage records
  it in its work log and its review checks the carry.
- Batched prefill applies to fresh rows (`start_pos` 0) of up to 512
  tokens; prefix-cache hits, chunked continuations and longer prompts run
  one row at a time (the stage-9 path). A burst of long prompts therefore
  still serializes.
- The batched dense pass costs about 0.8 ms per layer more than
  `minimal_matmul` would at 4096 rows, the price of bit equality with the
  single-row kernels.
- Prefill is eager (no prefill trace); the first sampler step after prefill
  runs eagerly.
- The chunked-prefill scheduler path is disabled by the pinned plugin for
  `model_type qwen3`; nonzero starts come from prefix-cache hits and the
  model's own 4096-row chunking.
- No profiler was collected in this stage (vLLM stage rule); device-time
  fields in `perf_summary.json` are null with the stage-7 reduced-variant
  profile cited.

## 7. Reproduction

From the worktree root (`S` = `doc/full_model/logs/stage_env.sh`, `PY` =
`/home/hous/dev/tt-metal/python_env/bin/python`, `M` =
`models/autoports/qwen_qwen3_8b`, `D` = `$M/doc/optimized_vllm/logs`,
`V` = `$M/doc/vllm_integration/logs`):

```bash
$S 2 $PY $D/batched_prefill_probe.py $D/batched_prefill_probe_L36_chip2_v2.json --layers 36 --reps 3
$S 0 env QWEN3_FULL_MODEL_LAYERS=36 QWEN3_EVIDENCE_SUFFIX=stage10 $PY -m pytest -q -s $M/tests/test_full_model.py -k 'test_batched_prefill_equals_solo or test_batch_decode or test_split_sampling_trace or test_host_work_counters or test_perf_batch1 or test_multi_row_prefill_equals_solo' --timeout 3600 -p no:cacheprovider
$S 0 env QWEN3_FULL_MODEL_LAYERS=36 QWEN3_EVIDENCE_SUFFIX=stage10_tracked TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 $PY -m pytest -q -s $M/tests/test_full_model.py -k 'test_split_sampling_trace or test_multi_row_prefill_equals_solo or test_batched_prefill_equals_solo or test_batch_decode or test_host_work_counters or test_tracked_replay_workload' --timeout 3600 -p no:cacheprovider
$S 1 $PY $D/penalty_probe.py $D/penalty_probe_L2_chip1.json --layers 2
$S 2 $PY $D/low_temperature_probe.py $D/low_temperature_probe_chip2.json --layers 36 --draws 48
$S 2 $PY -m readiness_check.run_prefill_resume_check --model-dir $M --mesh-device N150 --lengths 2560,16384 --block-sizes 32 --output $M/readiness/prefill_resume_check_stage10_default_thresholds.json --output-on-pass-only
$S 2 $PY -m readiness_check.run_prefill_resume_check --model-dir $M --mesh-device N150 --lengths 2560,16384 --block-sizes 32 --pcc-min-min 0.90 --top1-min 0.93 --output $M/readiness/prefill_resume_check.json
$V/serve.sh 1 8000 21600                       # full server on chip 1 (background); batched prefill on by default
$D/evidence_chain_stage10.sh 8000 $D/evidence_chain_chip1.status   # sampling, qualitative, benchmark, prefix_cache, prefill_burst x2 (stage-9 serial reference), mixed-shape, degenerate
$V/serve.sh 1 8000 7200 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0; $D/tracked_chain_stage10.sh 8000 $D/tracked_chain_chip1.status
$PY $D/write_perf_summary.py; $PY $D/qualitative_verdict_stage10.py; $PY $D/update_serving_contract_stage10.py
MODEL_DIR=$M bash /home/hous/dev/skills/plugins/tt-model-bringup/prompts/model_bringup_multigoal/10-optimized-vllm.check.sh
```

Server launch line (unchanged from stage 9; `serve.sh` wraps the in-tree
runner with the stage environment, `VLLM_SERVER_DEV_MODE=1`,
`EXTRA_MODELS_DIR=.../models/autoports/qwen_qwen3_8b/vllm`, `MESH_DEVICE=N150`,
`TT_VISIBLE_DEVICES=1`):

```text
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B --block_size 32 --max_num_seqs 32 --port 8000 --max_model_len 40960 --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 536870912}}' --enable-prefix-caching --async-scheduling
```

## 8. Stage gate, tracked runs and artifact index

- Gate `10-optimized-vllm.check.sh` (`logs/gate_10_optimized_vllm.log`):
  exit 1, advisory only: no degenerate output; context contract OK
  (40960); serving contract OK (37 rows); prefill features: the burst
  comparison passes (ratios 0.39 / 0.39, `config_matches` true), one
  advisory for the resumed-prefill check passing at lowered thresholds
  with the default-threshold run and the stage-7 margin probes beside it
  (the stage-9 classification: one BF16 near-tie flip in the 32-position
  hit window at 16352).
- Tracked serving (`logs/tracked_served/`, work log 10): the same launch
  line with `TT_METAL_TRACE_ALLOC_TRACKING=1
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` in the engine; the smoke
  sampling subset, the prefix-cache stage (pass), the 32 x 128 burst
  (batched prefill, ratios 0.403 / 0.403 under the tracker, decode replays
  after it) and the mixed-shape workload (47 of 47, 0 captures, program
  cache 651 flat) ran with no tracker error; positive control raised in the
  same environment (`logs/positive_control_chip1.log`).
- Watcher (`TT_METAL_WATCHER=10`, reduced two-layer model, no server,
  `watcher/summary.json`, work log 12.1 and 13), three runs with 0
  fatal-pattern matches in every watcher log: (1) the 22-shape batched
  prefill probe on chip 1 (`logs/batched_prefill_probe_L2_watcher_chip1.json`,
  full warmup buckets 32 to 4096 on two layers): every prepared batched
  shape 2x32 to 32x128, 16x256 and 8x512, so the batched kernel table at
  1024, 2048 and 4096 dense rows (2D/16 row pieces for QKV, O, up and
  down; the 1D (110, 8, 1x4) gate in 512-row pieces), the batched
  attention, fill and concat ops at 2 to 32 rows and buckets 32 to 512,
  plus the serial per-row and solo arms of every shape; bit equal in
  logits and K/V at every shape, program cache 650 flat, 60 dumps; (2) the
  pytest battery on chip 1 (reduced fixture, warmup buckets 32 to 256):
  the batched shapes 2x32, 4x32, 8x32, 2x64, 4x64 and 2x128 (at most 256
  dense rows, which run the `stage10` solo dense kernels; the batched
  table starts at 1024 rows), the mixed and reversed cases with batched
  groups beside single-row rows, `uniform_4/8/32` on the single-row path
  (no prepared shape for them in that fixture), the `stage10` 1D O and
  down kernels at 32 to 256 rows, batch decode after batched prefill and
  split sampling (10 passed, 2 skipped buckets, 28 dumps); (3) the penalty
  probe on chip 2 (penalties-on sampler precompile, capture and replay,
  12 dumps).
- Tracked generator-level gate (`logs/pytest_full_model_36_stage10_tracked_chip0.log`,
  `doc/full_model/*.stage10_tracked.json`, work log 10.1): split sampling,
  multi-row prefill, the seven batched-equals-solo cases, the four batch
  decode buckets, host-work counters and the representative replay
  workload with program-cache allocations included.

- `readiness_vllm/vllm_prefill_burst_benchmark.log` and
  `vllm_shared_prefix_benchmark.log` were overwritten by the tracked runs'
  stages (the runner writes to the fixed root and `served_final/` holds
  JSON only); the final run's runner output is `logs/stage_prefill_burst_repeat.log`
  (first run `logs/stage_prefill_burst.log`), and the JSON beside the log
  is the final run's. The other `readiness_vllm/` files are the final
  run's (restored from `served_final/`, md5 identical).
- `readiness_vllm/`: the final served evidence (`server.log`,
  `sampling_tests.log`, `vllm_qualitative_outputs.json`,
  `vllm_benchmark.json` + `vllm_result.json`, `vllm_ci_serving_*`,
  `prefix_cache_check.json`, `vllm_shared_prefix_*`,
  `vllm_prefill_burst_*` (the repeat), `mixed_shape_trace_reuse.json`,
  `vllm_run_summary.json`); copy in `served_final/`.
- `served_stage9_final/`: stage 9's final served JSON (the before numbers).
- `vllm_prefill_burst_serial_benchmark.stage9.json`: the serial reference.
- `perf_summary.json`, `qualitative_verdict.json`.
- `qualitative_verdict.json` cites its controls; the prompt-format record
  it relies on is `doc/optimized_full_model/qualitative/qualitative_prompt_format.json`
  (chat template, stage 7, unchanged tokenizer and prompts).
- `logs/`: probes and their JSON (`batched_prefill_probe*`,
  `kernel_equality_probe`, `op_bisect_probe*`, `down_kernel_probe`,
  `dense_kernel_probe`, `wide_kernel_probe`, `penalty_probe`,
  `low_temperature_probe`), the pytest logs, the stage logs of the chain,
  `tracked_served/`, the chain scripts, `write_perf_summary.py`,
  `qualitative_verdict_stage10.py`, `update_serving_contract_stage10.py`.
- `readiness/readiness_stage10_chip{0,1,2}.json`: provenance sidecars
  (commit, chip, command, artifacts) for every run.
- `doc/full_model/*.stage10.json`, `*.stage10_tracked.json`: the
  full-model test artifacts on this stage's code.

## 9. Stage review

Filled at closure.

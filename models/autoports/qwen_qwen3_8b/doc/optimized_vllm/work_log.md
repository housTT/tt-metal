# Qwen3-8B optimized vLLM: work log

Stage 10 of the model bringup for `Qwen/Qwen3-8B` on one Blackhole P300 chip
(1x1 mesh, p150 mesh graph descriptor; single chip is the target topology,
stages 4 and 5 were skipped by the operator). Times are Eastern Time (ET)
unless marked UTC; the stage started at 16:40 ET on Oct 30 (cap 06:40 ET
Oct 31). Worktree `/home/hous/dev/tt-metal-wt-qwen3`, branch
`hous/qwen3-8b-autoport`, start commit `534fb6fda1`. Every device command
runs through `doc/full_model/logs/stage_env.sh <chip> <command>` (the
operator's environment) under `timeout`; `logs/` below is
`doc/optimized_vllm/logs/`.

Abbreviations: APC (automatic prefix caching), HF (Hugging Face), KV
(key/value), LM (language model), PCC (Pearson correlation coefficient),
DRAM (device dynamic RAM), L1 (per-core local memory), TTFT (time to first
token), TPOT (time per output token), ITL (inter-token latency), t/s/u
(tokens per second per user), BF16 (bfloat16), BFP4/BFP8 (block floating
point), LoFi (low fidelity), SDPA (scaled dot-product attention), RoPE
(rotary position embedding), CQ (command queue).

## 1. Startup (16:40 to 16:50 ET)

- Skills read from disk (`tt-model-bringup` 0.1.21): `optimize`,
  `vllm-integration` with `prefill-features.md`, `serving-stack.md`,
  `plugin-changes.md`, `tt-device-usage`, `tt-enable-tracing`,
  `qualitative-check`, `stage-review`, `model-bringup` (startup).
- `TT_AUTODEBUG_ROOT=/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.6`
  (enabled; `claude plugin list`); `scripts/environment.py` exits 0.
- Stage 7, 8 and 9 documents read. Carried items: (a) the 1D multicast O
  and down prefill kernels at 64 to 256 rows (stage-7 review-2 P2,
  `doc/optimized_full_model/probe_candidates.review2_prefill_mm2.json`),
  (b) device penalties, (c) the device-versus-host sampler at temperature
  0.01, (d) the resume check at default thresholds on the final code.
- Chips 0, 1 and 2 free at start (`fuser /dev/tenstorrent/{0,1,2}` empty);
  chip 3 is the operator's and is never touched.
- Serial reference for the burst gate: stage 9's
  `readiness_vllm/vllm_prefill_burst_serial_benchmark.json` (32 x 128 at
  concurrency 32, 32 output tokens, one row per decoder pass): TTFT p50
  1231.1 ms, p99 1231.9 ms, mean 1194.5 ms. Copied to
  `doc/optimized_vllm/vllm_prefill_burst_serial_benchmark.stage9.json`
  (raw result beside it). Gate: p99 ratio <= 0.6 (739 ms) and mean ratio
  <= 0.5 (597 ms).

## 2. Batched prefill design (16:50 to 17:05 ET)

Design 1 of `prefill-features.md` (equal padded length), chosen because
every row of a burst pads to the same bucket, the dense row count then
equals an already prepared single-row bucket, and the ttnn attention ops
take a batch dimension natively (`nlp_create_qkv_heads` on `[B, 1, S, W]`,
causal `scaled_dot_product_attention` over `[B, heads, S, D]`,
`paged_fill_cache` with `batch_idx_tensor`, `nlp_concat_heads` on
`[B, heads, S, D]`); the packed-row design would need block-diagonal causal
attention, which the SDPA op offers only non-causally (`cu_window_seqlens`).

- `OptimizedDecoder.prefill_forward_batched(x [1, 1, B*S, H], rows, seq_len,
  page_table [B, blocks], batch_idx [B], kv_cache)`: both norms, every
  projection and the MLP run once over `B*S` rows; attention and the cache
  fill run per row inside one op each. `xqkv` is viewed as `[B, 1, S, W]`
  (`ttnn.reshape` is a zero-cost view when the last dim is unchanged and the
  second-last is tile aligned), RoPE uses one `[1, 1, S, D]` cos/sin gather
  (every row starts at position 0; the hf RoPE factory indexes cos/sin by
  tile row modulo `S/32`, so a `[B, heads, S, D]` input needs no reshape),
  the fill writes K/V of all rows through the `[B, blocks]` page table with
  `batch_idx = arange(B)`, SDPA runs causal with chunk `min(S, 256)`, the
  concat output is viewed back as `[1, 1, B*S, heads*D]`.
- `Qwen3Model.prefill_hidden_batched(token_rows, rows, seq_len, page_table,
  kv_cache)`: one `[1, B*S]` id tensor (row `i` at `[i*S, (i+1)*S)`, zero
  ids past the real tokens, dummy rows past the real count repeating the
  last real prompt) through the shared `_embed`, then every layer.
- `Qwen3Generator.prefill_forward`: `_batched_prefill_groups` keeps rows
  with `start_pos == 0` whose bucket `prefill_compute_len(len)` is one of
  the batched sequence buckets (32, 64, 128, 256, 512), grouped by bucket,
  when at least two rows share it and `(batched_row_bucket(n), S)` is a
  prepared shape (row buckets 2, 4, 8, 16, 32; `B*S <= 4096`, so every
  dense row count is a prepared single-row bucket). Every other row
  (nonzero start, longer bucket, singleton group) runs on the single-row
  path of the same call. Output rows keep the call's order: with one group
  covering the call, the last-token rows are gathered with one fixed
  32-wide runtime index (`gather_from_rows`, dummy indices repeating the
  last) into the `[1, 1, 32, H]` LM-head tile; otherwise every row's last
  hidden row is gathered one at a time (the same programs as the single-row
  path) and `rows_to_tile` places them. Program keys depend only on
  `(B, S)` and `B*S`; token ids, lengths, page tables and last-row indices
  are runtime data. Dummy rows get an all-scratch page-table row.
- `prepare()` warms every prepared `(B, S)` shape with `B-1` real rows of
  alternating lengths `S` and `S-1` (both output modes share the programs
  past the LM head); the counters `batched_prefill_groups`,
  `batched_prefill_rows` and `single_row_prefill_rows` are logged per call
  by the adapter. Environment override `QWEN3_BATCHED_PREFILL=0` disables
  the path for A/B measurement (`build_generator`).

## 3. First device pass and the bit-equality finding (16:47 to 17:15 ET)

`logs/batched_prefill_probe.py` (solo arm, serial arm = the same multi-row
call with batched prefill off, batched arm; per-row logits and K/V pages
of the first and last layer at the real positions, zero tolerance; wall
time per arm, median of `--reps`; program-cache count around the batched
call).

- 2 layers, chip 0, five shapes (`logs/batched_prefill_probe_L2_chip0.json`,
  16:47 ET): program cache 626 before and after every batched call,
  batched 1.3x to 2.7x faster than the serial call; 2x32 and 4x128 bit
  equal in logits and K/V, 32x128 / 16x256 / 8x512 not (logits max
  difference 0.09 to 0.22, K/V of layer 0 always equal, layer 1 differing
  for 2 of 32 rows at 32x128).
- 36 layers, chip 2, all 22 shapes with the first kernel policy
  (`logs/batched_prefill_probe_L36_chip2.json`, 16:52 to 16:57 ET): every
  batched call faster than the serial call (32x128: 402.6 ms against
  1217.4 ms, 32x32: 10.1x), program cache flat at 626; bit equal only up
  to 256 dense rows; above, PCC minimum 0.99963 to 0.99983 and top-1 flips
  on random tokens at 16x32, 32x32, 32x64, 16x128, 32x128, 16x256, 8x512.

Bisect (`logs/kernel_equality_probe.py`, `logs/op_bisect_probe.py` v1 and
v2, `logs/down_kernel_probe.py`, chip 0, 1 and 2 layers):

- the sharded input norm and the QKV projection are bit equal row for row
  across every kernel (1D multicast at 128 rows, 2D at 512, `minimal_matmul`
  at 4096) on real activations;
- recording every op of one layer (`op_bisect_probe2_chip0.json`, 32x128
  and 16x256 with real tokens): norms, QKV, head split, q/k norms, RoPE,
  SDPA (batched causal against the chunked paged solo op), concat heads, O
  projection, gate, up, SiLU-mul all bit equal; the down projection
  (`linear_down_proj`, K = 12288) differs in one element for 2 of 32 rows
  (differences 3e-5 and 3.9e-3), and the retry on the recorded real input
  shows `minimal_matmul` reproducing its own batched output while every 2D
  kernel at 4096 rows (`in0_block_w` 4, 2, auto) reproduces the solo 2D
  result bit for bit;
- on Gaussian random inputs every kernel agrees (`down_kernel_probe`), so
  the `minimal_matmul` difference is a data-dependent accumulation-order
  effect, not a bug in either kernel.

Classification: the stage-7 solo table selects `minimal_matmul` for the
gate projection at 512 rows and the stage-3 rule selects it for every role
from 1024 rows; the 2D and 1D multicast kernels share one compute kernel
and one sequential fp32 accumulation over K, so they agree with each other
at any row count and `in0_block_w`, while `minimal_matmul` orders the K
accumulation differently and flips rare bf16 roundings. Bit equality
between the batched pass and the solo rows therefore needs the batched
dense ops on the 2D/1D family only, and the solo gate at 512 rows off
`minimal_matmul`.

## 4. Kernel selection for the batched dense pass (17:15 to 17:35 ET)

`logs/dense_kernel_probe.py` (2D over `[pieces, 1, rows/pieces, K]` row
pieces through the batch dimension, `fuse_batch` off; `in0_block_w` 16, 8,
4) and `logs/wide_kernel_probe.py` (1D `mcast_in0` over pieces for the
N = 12288 roles), chip 0, every candidate bit equal to the solo kernel;
times per layer in ms (`minimal_matmul` at the same rows in brackets):

| role | 1024 rows | 2048 rows | 4096 rows | kept |
| --- | ---: | ---: | ---: | --- |
| qkv (K 4096, N 6144, BFP8) | 2D/16 0.271 [0.288] | 2D/16 2x1024 0.534 [0.581] | 2D/16 4x1024 0.997 [0.991] | 2D/16, pieces of 1024 |
| o (4096 x 4096) | 2D/16 0.188 [0.205] | 2D/16 0.308 [0.402] | 2D/16 2x2048 0.586 [0.697] | 2D/16, pieces of 2048 |
| gate (4096 x 12288, BFP8) | 1D 110c/8 2x512 0.661 [0.494] | 4x512 1.269 [1.010] | 8x512 2.476 [1.821] | 1D (110, 8, 1x4), pieces of 512 |
| up (4096 x 12288, BFP4) | 2D/16 2x512 0.565 [0.482] | 4x512 1.101 [0.967] | 8x512 2.151 [1.797] | 2D/16, pieces of 512 |
| down (12288 x 4096) | 2D/16 0.449 [0.458] | 2D/16 0.776 [0.861] | 2D/16 2x2048 1.541 [1.570] | 2D/16, pieces of 2048 |

One 2D block of a 12288-wide output does not fit L1 above 512 rows
(`program.cpp:1875` at 2D/16 and 2D/8; 2D/4 fits at 4096 rows but at 2.8 ms
for down), so gate and up run as 512-row pieces. Per layer at 4096 rows the
2D/1D table costs 7.75 ms against 6.92 ms with `minimal_matmul` (+0.83 ms,
about 30 ms on the 32x128 burst), which the burst gate absorbs.

`BATCHED_PREFILL_KERNEL_TABLE` (`tt/optimized_decoder.py`) holds the kept
choices; `_linear_batched` applies them (view to `[pieces, 1, rows/pieces,
K]`, one `ttnn.linear`, view back). Solo table `stage10` = `stage7` with
gate at 512 rows on 1D (110, 8, 1x4) (0.345 ms against 0.337 ms with
`minimal_matmul`, `wide_kernel_probe`) plus the carried item (a) below;
`QWEN3_PREFILL_KERNEL_TABLE=stage7` restores the stage-7 table.

### 4.1 Carried item (a): 1D multicast O and down kernels at 64 to 256 rows

The stage-7 review-2 recommendation applied in the `stage10` table:
`o_proj` 64/128/256 rows -> 1D (110, 4, 2x2), (96, 8, 2x2), (64, 16, 2x2);
`down_proj` -> 1D (96, 4, 2x2), (64, 8, 2x2), (110, 16, 2x2). The 1D and 2D
kernels are one family, so the change keeps the batched pass bit equal to
the solo rows. Effect: token-out TTFT of the 128-token prompt 42.1 ms
(stage-9 code, `doc/full_model/perf_batch1_128_128.json`) -> 37.9 ms
(`perf_batch1_128_128.stage10.json`, chip 0, 17:12 ET); decode 21.0 ->
20.67 ms/token (unchanged path, run-to-run range).

## 5. Final generator-level evidence (17:04 to 17:13 ET)

- 2 layers, chip 0, all 22 shapes (`logs/batched_prefill_probe_L2_chip0_v2.json`):
  bit equal in logits and K/V at every shape, program cache 638 flat,
  batched 1.14x to 5.6x faster than the serial call.
- 36 layers, chip 2, all 22 shapes (`logs/batched_prefill_probe_L36_chip2_v2.json`,
  17:07 to 17:13 ET, prepare 59.0 s): bit equal in logits and in the K/V
  pages of layers 0 and 35 at every shape (PCC 0.99996 to 1.0), top-1
  equal, program cache 645 before and after every batched call, reordered
  rows bit equal. Calibration (median of 3, ms):

| shape | solo rows summed | serial call | batched call | gain |
| --- | ---: | ---: | ---: | ---: |
| 2x32 | 76.0 | 71.5 | 33.5 | 2.13x |
| 8x32 | 316.5 | 270.3 | 49.7 | 5.43x |
| 32x32 | 1261.4 | 1060.9 | 114.2 | 9.29x |
| 2x64 | 71.3 | 67.0 | 38.3 | 1.75x |
| 32x64 | 1180.9 | 976.8 | 217.7 | 4.49x |
| 2x128 | 78.0 | 72.9 | 48.6 | 1.50x |
| 4x128 | 156.4 | 140.8 | 65.1 | 2.16x |
| 8x128 | 311.3 | 277.8 | 112.1 | 2.48x |
| 16x128 | 619.8 | 549.7 | 218.9 | 2.51x |
| 32x128 | 1247.0 | 1094.9 | 426.6 | 2.57x |
| 2x256 | 102.1 | 96.0 | 66.0 | 1.45x |
| 16x256 | 807.4 | 736.3 | 425.7 | 1.73x |
| 2x512 | 139.3 | 134.8 | 113.3 | 1.19x |
| 8x512 | 556.6 | 524.0 | 431.0 | 1.22x |

  Every prepared shape wins against the serial call, so all 22 stay
  enabled (the full table is in the JSON). The serial call of 32x128
  (1094.9 ms) matches the stage-9 served burst TTFT (1194 ms mean, which
  adds the server's admission and sampling).
- Full-model pytest, chip 0, untracked (`logs/pytest_full_model_36_stage10_chip0.log`,
  17:07 to 17:12 ET, 15 passed in 308 s, artifacts
  `doc/full_model/*.stage10.json`): `test_batched_prefill_equals_solo`
  in seven cases (mixed lengths 5, 31, 32, 33, 127, 128, 129, 257, 640,
  1000, 1024 in both orders: groups 32 x {5, 31, 32} and 128 x {127, 128}
  batched, six rows single-row in the same call; uniform 2, 4, 8 and 32
  rows of 65 to 128 random tokens; eight chat prompts): every row bit equal
  to its solo logits and K/V pages (layers 0 and 35), rerun and reordered
  bit equal, program cache 645 flat on the first call and the rerun;
  `test_batch_decode` 8/16/24/32 (multi-row prefill through the batched
  path, then 32 greedy decode steps in the same batch as the solo arm):
  every user identical for all 32 tokens; `test_multi_row_prefill_equals_solo`
  (stage-6 contract, now through the batched path for its groups) PCC
  >= 0.99999; split sampling, host-work counters and `test_perf_batch1`
  unchanged in behaviour.

## 6. Carried item (b): device penalties (17:05 to 17:30 ET)

The common `SamplingGenerator` already applies presence, frequency and
repetition penalties on device (`TTPenalties`: a prompt mask, output
counts updated inside the sampling trace by the sampled token, a separate
sampling trace per bucket keyed on penalties on or off). The generator now
owns them:

- `prepare()` precompiles the penalties-on sampler variant for every decode
  bucket (`penalty_sampling_params`), including the prompt-mask and
  output-count resets and one counted sampling pass (the sampler's
  `precompile` skips the count update, which left the scatter path
  uncompiled and made the first capture attempt fail with `TT_FATAL
  !is_capturing_trace`, `logs/penalty_probe_L2_chip1.log` first run);
- `capture_decode_traces` captures and executes the penalties-on sampling
  trace of every bucket beside the greedy one (5 more sampling traces;
  the decode trace is shared);
- `_apply_sampling_params(prompt_tokens, output_tokens)`: at prefill the
  prompt mask is rebuilt for the admitted rows from the call's own tokens;
  on a reload decode step (`reset_batch`, first step after prefill, mode
  change) the mask is reloaded from the plugin's `prompt_tokens` and the
  output counts from its `output_tokens`; steady steps count the sampled
  token in the trace. Both are no-ops while no penalty is active;
- the adapter declares `supports_device_penalties: True` and passes
  `prompt_tokens` / `output_tokens` through; the plugin then keeps
  penalised batches on the device sampler (and disables its async overlap
  for them, `async_decode.py` `no_penalties`).

Probe (`logs/penalty_probe.py`, 2 layers, chip 1, 17:28 ET,
`logs/penalty_probe_L2_chip1.json`): greedy repeats identical before and
after penalised cases; repetition 2.0 and presence 2.0 change the greedy
continuation; a mixed batch of 8 rows (penalty on odd rows) reproduces the
solo greedy rows and the solo penalised rows exactly; a seeded (42) mixed
batch is deterministic within its two groups; program cache 313 flat
across every case after the captures; `penalized_sampling_steps` counted
on every penalised step, 0 on the others.

## 7. Resume check at default thresholds (carried item d) and serving runs

Launched 17:28 ET on chip 2 with the final prefill numerics (`stage10`
kernel table): default thresholds to
`readiness/prefill_resume_check_stage10_default_thresholds.json`
(`--output-on-pass-only`), then the lowered run (`--pcc-min-min 0.90
--top1-min 0.93`, as stages 6 to 9) to the canonical
`readiness/prefill_resume_check.json`.

Results (chip 2, 17:28 to 17:40 ET, `logs/resume_check_{default,lowered}_chip2.log`):
identical to stage 9 at every start, to the digit. Default thresholds: all
16 PCC gates pass (PCC mean >= 0.9996, min 0.9886 at the tile-unaligned
start 68 of both lengths), both controls fail at every window of at least
64 positions, top-1 0.988 to 1.000 everywhere except the 32-position
prefix-cache-hit window at 16352 of the 16384-token prompt (31 of 32 =
0.969, the stage-7 near tie with its margin probe
`doc/optimized_full_model/resume_margin_probe_L16384_s16352_fp32.json`),
which makes `status=fail` (exit 1,
`readiness/prefill_resume_check_stage10_default_thresholds.failed.json`).
Lowered (`--pcc-min-min 0.90 --top1-min 0.93`): `status=pass` at all 8
starts (exit 0, the canonical `readiness/prefill_resume_check.json`). The
`stage10` kernel table (1D O and down rows) therefore left the continuation
numerics unchanged; the gate reports the lowered run as advisory as in
stage 9.

Tracked full-model run launched 17:29 ET on chip 0
(`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`,
`-k 'split_sampling_trace or multi_row_prefill_equals_solo or
batched_prefill_equals_solo or batch_decode or host_work_counters or
tracked_replay_workload'`, suffix `stage10_tracked`). Results in section 8.

Full server launched 17:29 ET on chip 1 with the stage-9 launch line
(`doc/vllm_integration/logs/serve.sh 1 8000 21600`; batched prefill on by
default) and the stage-10 chain `logs/evidence_chain_stage10.sh` (sampling
smoke and full, qualitative, benchmark, prefix cache, prefill burst twice
against the stage-9 serial reference, mixed-shape trace reuse, degenerate
check). Results in section 8.

## 8. Served chain on the final code (chip 1, 17:29 to 17:38 ET)

Server `doc/vllm_integration/logs/serve.sh 1 8000 21600` (healthy after
140 s: weights on device, `prepare()` 59.5 s including the 22 batched
shapes and the penalties-on sampler variants, five decode traces plus ten
sampling traces captured, `max_tokens_all_users` and the 3584-block vLLM
cache as in stage 9; program cache 651 after capture). Chain
`logs/evidence_chain_stage10.sh` (`logs/evidence_chain_chip1.status`,
17:32:06 to 17:38:46 ET; the first attempt aborted on the smoke result
because the script treated it as fatal, `logs/evidence_chain_chip1.attempt1.status`):

| Stage | Result |
| --- | --- |
| sampling smoke (4 targets) | 2 passed, 1 skipped, `test_mixed_params_batch` failed (section 8.1) |
| sampling full (`--tt-max-num-seqs 32`, 73 tests) | 71 passed, 1 failed (`test_mixed_params_batch`), 1 skipped in 126 s; all six penalty tests pass on the device sampler (189 penalised device-sampled decode steps in the log, 0 out-of-vocabulary ids) |
| qualitative | `qualitative_verdict.json`: served greedy identical to the stage-9 served greedy text for all 256 tokens on all six prompts, identical to the stage-9 generator control (128 tokens), HF first divergence 86, 37, 32, 39, 33, 22; degenerate check none |
| benchmark 128/128/1 | TTFT p50 44.59 ms, TPOT mean 20.71 ms, ITL p50 20.68 ms, 48.29 t/s/u, 47.86 tok/s (stage 9: 49.3, 20.71, 20.67, 48.28, 47.75) |
| benchmark CI 100/100/32 | TTFT p50 477.5 ms, TPOT mean 24.82 ms, 1094.4 tok/s, 40.3 t/s/u, 32 of 32 (stage 9: 1217.9, 25.06, 873.2, 39.9) |
| prefix_cache | pass, warm/cold TTFT ratio 0.180, long resume and shared-prefix bench pass |
| prefill_burst (stage-9 serial reference) | 32 of 32; TTFT p50 480.3, p99 481.1, mean 467.0 ms; ratios p99 0.3905, mean 0.3910, `config_matches` true (first run 0.393 / 0.394, `logs/stage_prefill_burst.log`; the JSON holds the repeat) |
| mixed-shape trace reuse | 47 of 47, 32 prefill calls, 928 decode steps (905 device, 23 host), 0 captures, 0 releases, program cache 651 flat, DRAM 31752 MB flat, greedy repeats identical, same seed identical; TTFT p50 83 ms, p99 1331 ms (the concurrent burst's 4097-token row and the long single-row prefills) |
| degenerate | none |

Burst mechanics in the server log: the plugin admits the first request
alone (`call=223 rows=1 ends=[128]`, 37.4 ms) and the other 31 in one call
(`prefill call=224 rows=31 ends=[128, ...] mode=device-sampled ms=422.0
... batched_prefill_groups=1 batched_prefill_rows=31`; the repeat: calls
225 and 226, 37.1 and 421.8 ms), so TTFT p50 and p99 sit at 480 ms (422
ms batched call plus the lone row and the plugin's sampling and
admission), against 1194 ms mean with the per-row loop. (Call 207, 421.7
ms, is the CI burst's 31-row call of 100-token prompts in the same 128
bucket.)

`perf_summary.json` written by `logs/write_perf_summary.py` (before and
after rows, burst ratios, device-time fields null with the no-profiler
reason). Serving contract rows updated by
`logs/update_serving_contract_stage10.py` (6 rows; checker OK, 37 rows).
`readiness_vllm/` archived in `served_final/` (server log gzipped);
stage 9's final JSON kept in `served_stage9_final/` as the before numbers.

### 8.1 `test_mixed_params_batch` (rewritten after review 1, from the saved logs)

The test sends ten requests twice (the second time in shuffled order) and
asserts that greedy and seeded rows reproduce. The failing row differs per
run: full suite (`readiness_vllm/sampling_tests.log`): request 0
(`temp=0.5, seed=42`, the frequency-penalty prompt), `'10, 20, 30'`
against `'1, 2, 3, 4'`; smoke (`logs/sampling_smoke_full_model.log`):
request 2 (`temp=0.5, seed=42`), `'10, 2'` against `'3, 6,'`; tracked smoke
(`logs/tracked_served/sampling_smoke.log`): request 1 (`temp=1.0, seed=7`,
the top-p prompt), `'5\n ...'` against `'10000000'`.

Server log of the full run (`readiness_vllm/server.log`): run 1 admits
one row alone, `prefill call=70 rows=1 ends=[3] mode=logits` (the `top_k
100` request, above `max_device_top_k` 32, so the step is host-sampled),
then the other nine, `call=71 rows=9 ends=[3, 6, 3, 3, 3, 3, 7, 3, 4]
mode=device-sampled` (no host-only request in that step: the seeded
penalised prompts take their first draw on the device sampler); the decode
steps 540 to 544 are host-sampled (the `top_k 100` request is in the
batch) and 545 to 549 device-sampled after it finishes. Run 2 admits a
different lone row device-sampled (`call=72 rows=1 ends=[4]`) and the
nine others with the `top_k 100` request host-sampled (`call=73 rows=9
mode=logits`); steps 550 to 553 host, 554 to 558 device. The smoke run
shows the same flip at calls 11 to 14. Every seeded request of the 9-row
group therefore takes its first draw on the device in one run and on the
host in the other, and its later host draws are offset by one; at
temperature 0.5 and 1.0 that flips tokens without any near tie. The
`top_k 100` request sets the step mode through the plugin's per-step
routing (`check_perform_device_sampling`).

Device penalties removed the penalty-routing cause of stage 9
(`test_presence_penalty_mixed_batch` and the other five penalty tests
pass; penalised rows are device-sampled whenever the step has no host-only
request). The `top_k 100` cause needs a device top-k above 32 (the common
sampler draws from one 32-candidate tile) or a plugin routing rule, and
stays reproducibility-only: correctness, logprobs, crash-free serving, the
greedy repeats of the mixed-shape check and the qualitative suite pass on
the same server. The stage-9 instance of this test (`'198'` against
`'100'` at temperature 0.01) is the subject of carried item (c), section
9; the low-temperature probe characterises that instance, not the
stage-10 rows above.

## 9. Carried item (c): the device sampler at temperature 0.01 (chip 2, 17:32 to 17:36 ET)

`logs/low_temperature_probe.py` (36 layers,
`logs/low_temperature_probe_chip2.json`): for `Letter: `, `Random: ` and
`The three laws of thermodynamics are`, the fp32 first-token logits, the
host distribution at temperature 0.01 and 0.7 (top-k 20, top-p 0.95) and
48 device draws with explicit seeds 0 to 47 at the same parameters. At
0.01 the host distribution is one-hot (top-2 logit gaps 0.14, 0.59 and
0.22; the bf16 view of the scaled logits keeps the same leader) and the
device sampler returns that token in 48 of 48 draws on every prompt. At
0.7 the device draws follow the host distribution's rank order (`Letter:`
host 1 0.31, 2 0.25, 3 0.12, 5 0.10; device counts 1: 14, 5: 10, 2: 8, 3:
3 of 48). So the device and host samplers agree on the token at 0.01; the
stage-9 `'198'` against `'100'` pair differs in the second token through
the seeded-stream offset of section 8.1.

## 10. Tracked runs, gate and commits (17:40 to 17:50 ET)

- Tracked serving (chip 1, `logs/tracked_chain_chip1.status`,
  `logs/tracked_served/`): the final server stopped at idle with SIGINT
  (chip released in 2 s), relaunched with
  `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`
  in the engine environment (healthy after 140 s), then the smoke sampling
  subset (2 passed, 1 skipped, `test_mixed_params_batch` failed as in 8.1),
  the prefix-cache stage without the shared bench (pass, ratio 0.16), the
  32 x 128 burst against the serial reference (32 of 32, TTFT p50 495 ms,
  ratios 0.403 / 0.403 under the tracker; the 31-row call batched) with
  32 decode replays after it, and the full mixed-shape workload (47 of 47,
  0 captures, 0 releases, program cache 651 flat, greedy repeats
  identical). No `still alive` line and no tracker error in
  `logs/tracked_served/server.log`; `/health` 200 at the end; stopped at
  idle (2 s). Positive control in the same environment on chip 1
  (`logs/positive_control_chip1.log`, the stage-3 control): raised
  "Found 1 device buffer(s) still alive before trace replay".
- Tracked generator-level gate (chip 0, `logs/pytest_full_model_36_stage10_tracked_chip0.log`,
  `doc/full_model/*.stage10_tracked.json`): recorded in 10.1 when the run
  ends.
- `readiness_vllm/` restored from `served_final/` after the tracked run
  (md5 identical for the burst and prefix-cache JSON the tracked stages
  had overwritten).
- Gate (`logs/gate_10_optimized_vllm.log`): `MODEL_DIR=models/autoports/qwen_qwen3_8b
  bash .../10-optimized-vllm.check.sh` exit 1: no degenerate output;
  context contract OK (40960); serving contract OK (37 rows); prefill
  features: one advisory (the resumed-prefill check passed with lowered
  thresholds; the default-threshold run and the stage-7 margin probes are
  beside it), the burst comparison required by `--require-burst` passes
  (ratios 0.39 / 0.39 against the serial reference, `config_matches`
  true), no critical.
- Repository hooks: the first commit (`2f371d340a`) was made with the
  hooks bypassed; the second (`24a48e130b`) ran them, and black and
  autoflake reformatted `tt/generator.py`, `tt/model.py`,
  `tt/optimized_decoder.py` and the probe scripts (AST identical for every
  source and test file, checked with `ast.dump`; two unused imports removed
  from probe scripts). Every device run of this stage executed the
  `2f371d340a` code, which the formatted files reproduce statement for
  statement.

| Commit | Content |
| --- | --- |
| `2f371d340a` | batched prefill, the batched 2D/1D kernel table, the `stage10` solo table (1D O and down, gate 512 on 1D), device penalties, the acceptance test, probes, generator-level evidence, sidecars, work log |
| `24a48e130b` | served evidence (`readiness_vllm/` copy in `served_final/`), resume checks, low-temperature probe, serving contract rows, `perf_summary.json`, `qualitative_verdict.json`, README, work log, hook formatting |
| `4f72f52bb2` | tracked generator-level gate, tracked serving, positive control, gate log, `readiness_vllm/` restored, sidecars, README and work log |

### 10.1 Tracked generator-level gate (chip 0, 17:29 to 17:53 ET)

`logs/pytest_full_model_36_stage10_tracked_chip0.log`: 15 passed, 2
deselected in 1418.8 s (23 min 39 s) with `TT_METAL_TRACE_ALLOC_TRACKING=1
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` (program-cache allocations
included), no tracker error. `doc/full_model/*.stage10_tracked.json`:
the seven batched-equals-solo cases bit equal in logits and K/V with
program cache 650 before and after the first call and the rerun; the four
batch-decode buckets identical to the solo arm for all 32 tokens after
the batched multi-row prefill; `tracked_replay_workload`: 241 prefill
calls (74 batched groups, 597 batched rows, 241 single-row rows), 2786
decode and 2785 sampling replays, 6 decode and 5 sampling captures (the
module's deferred first captures), 0 eager decode steps, program cache
650 at the end. Sidecar `readiness/readiness_stage10_chip0.json`.

## 11. Stage review 1 and the fixes (17:55 to 18:40 ET)

Review 1 (`logs/review/stage_review_1.md`, fresh read-only subagent at
HEAD `4f72f52bb2`): `more-work-needed` with one P2: work log 8.1 and
README section 6 attributed the `test_mixed_params_batch` failure to the
stage-9 instance (`'100'` against `'198'`, temperature 0.01) and described
the admission pattern backwards; the saved logs show other rows failing
(temperature 0.5 and 1.0 seeded prompts) with the `top_k 100` request
setting the step mode. Fixed from the saved logs without hardware:
section 8.1 and README sections 5 (c) and 6 rewritten with the failing
request per run, the prefill calls 70 to 73 and 11 to 14 with their modes
and the decode-mode windows, and the stage-9 pair moved to carried item
(c). Other concerns addressed: README section 1 regime sentence corrected
(chip 0 ran the tracked tests during the headline benchmark, the stage-9
regime) and the serial-reference age recorded beside the ratio with the
like-for-like generator-level pair; `4f72f52bb2` added to the commit
table; a `regime` field added to every run in the three sidecars; the
prompt-format control path added to `qualitative_verdict.json`; the
device-penalty dependencies (plugin `prompt_tokens` / `output_tokens` on
every penalised step; the `_run_sampling(count_tokens=True)` warmup
coupling to the common sampler) and the undocumented capability key noted
in the serving contract. Hard-check gap closed: the adapter now logs
`ttnn.trace_allocation_config.TRACE_ALLOC_TRACKING` with the two
environment variables at model init (`qwen3-vllm trace allocation
tracking in this engine process: True (...)`), and a second tracked
server run on chip 1 (`logs/tracked_served2/`, `logs/tracked_chain2_chip1.status`)
shows the line in the engine process beside the same served workload
(smoke subset, prefix-cache stage, 32 x 128 burst, mixed-shape); results
in 11.1. The log line runs once at init and changes no numerics; the
headline and burst evidence stand on the code of `2f371d340a`.

### 11.1 Second tracked server run (chip 1, 18:20 to 18:24 ET)

`doc/vllm_integration/logs/serve.sh 1 8000 3600 TT_METAL_TRACE_ALLOC_TRACKING=1
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` with the adapter log line
(`logs/tracked_served2/server.log.gz`: `qwen3-vllm trace allocation
tracking in this engine process: True (TT_METAL_TRACE_ALLOC_TRACKING='1',
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE='0')` in `EngineCore_DP0`), healthy
after 130 s; `logs/tracked_chain2_stage10.sh`: smoke subset (2 passed, 1
skipped, `test_mixed_params_batch` failed as in 8.1), prefix-cache stage
pass (ratio 0.156), the 32 x 128 burst 32 of 32 (TTFT p50 487.4 ms,
ratios 0.396 / 0.397 against the serial reference, 32 decode replays
after it), mixed-shape 47 of 47 (0 captures, 0 releases, program cache
651 flat, greedy repeats identical); no `still alive` line, `/health` 200
at the end, stopped at idle (2 s), chip released. `readiness_vllm/`
restored from `served_final/` afterwards (md5 identical).

## 12. Stage review 2 and the fixes (18:45 to 19:00 ET)

Review 2 (`logs/review/stage_review_2.md`, fresh read-only subagent at
HEAD `1ae8602de6`): every review-1 fix verified; `more-work-needed` with
one P2: no watcher run covered the device paths this stage added (the
`optimize` skill's watcher item in its vLLM-stage form, a reduced-layer
generator probe). Other concerns: two server-log citations named the
wrong calls (call 207 is the CI burst's 31-row call, the prefill burst's
are 224 and 226 at 422.0 and 421.8 ms; call 5 is a 3-token prompt, the
served 128-token single rows are calls 205, 223 and 225 at 36.9 to 37.4
ms, about the stage-7 value), `readiness_vllm/vllm_prefill_burst_benchmark.log`
is the tracked2 run's log beside the final run's JSON, one status file
untracked. All fixed: citations corrected in README sections 1 and 3.2
and work log 8 (the served single-row claim withdrawn; the generator-level
pair and TTFT remain the serial-reference-age evidence), the log
mismatch recorded in the README artifact index with the final run's
runner output (`logs/stage_prefill_burst_repeat.log`), the status file
committed.

### 12.1 Watcher runs (chips 1 and 2, 18:45 to 18:48 ET)

`TT_METAL_WATCHER=10` on the reduced two-layer model, separate
`TT_METAL_LOGS_PATH` per run (`doc/optimized_vllm/watcher/{pytest,penalty}`),
no server, each under `timeout` (`logs/watcher_summary_stage10.py` ->
`doc/optimized_vllm/watcher/summary.json`):

- chip 1: `QWEN3_FULL_MODEL_LAYERS=2 QWEN3_EVIDENCE_SUFFIX=watcher pytest -k
  'test_batched_prefill_equals_solo or test_batch_decode or
  test_split_sampling_trace'` (`logs/pytest_watcher_reduced_chip1.log`):
  10 passed, 2 skipped (decode buckets 16 and 24 are not allocated in the
  reduced fixture), 5 deselected in 132.9 s; the seven batched-equals-solo
  cases (the reduced fixture's prepared shapes 2x32, 4x32, 8x32, 2x64,
  4x64, 2x128 plus single-row rows on the `stage10` 1D O and down kernels),
  batch decode 8 and 32 after batched prefill, split sampling; watcher log
  311 KB, 28 dumps, 0 matches of fatal, assert, bad noc, hung, stack
  overflow or corrupt (and 0 of exception, overflow, error, invalid);
  artifacts `doc/full_model/*.watcher.json` (the batched and batch-decode
  files are this run's; `host_work_counters`, `multi_row_prefill` and
  `tracked_replay_workload` `.watcher.json` are stage 7's and are listed by
  the summary because they share the suffix).
- chip 2: `penalty_probe.py --layers 2` under the watcher
  (`logs/penalty_probe_watcher_chip2.log`, `logs/penalty_probe_L2_watcher_chip2.json`):
  every summary check true (greedy repeats equal, penalties change the
  output, mixed rows equal their solo arms, seeded groups deterministic,
  program cache flat); watcher log 132 KB, 12 dumps, 0 fatal-pattern
  matches.

Watcher and tracker ran in separate processes; no profiler was collected.

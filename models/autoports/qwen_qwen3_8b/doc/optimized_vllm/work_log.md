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
`readiness/prefill_resume_check.json`. Results in section 8.

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

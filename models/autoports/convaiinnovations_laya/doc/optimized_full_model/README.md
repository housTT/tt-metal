# Stage 7: optimized full model (bucket set, warmup, trace memory, latency ladder, reconciliation)

Plugin stage "optimize (full model)" mapped to Laya (PLAN.md section 5 row 7, Appendix A.5, amendment A10). This stage
adds the seq buckets 128 and 256 and the exact row buckets 5, 10 and 50 that the served token-length histogram of stage
6 asked for, re-derives and A/Bs the per-bucket program plans at the new row counts, measures the trace memory and the
two-phase warmup of the whole deployment set, closes the review R1 items deferred to stage 7 (tracked replay over all
buckets in one process, the clean Wo threshold pair, the GeGLU plan under the erf policy), and publishes the latency
ladder, the B 64 throughput, the lower-bound reconciliation and the host tail timing. PCC = Pearson correlation
coefficient; CLS = the first token position; SDPA = scaled dot-product attention; GeGLU = the gated GELU MLP of
ModernBERT. Every timing carries the host 1-minute load average at which it was taken (amendment A5: under 8).

## Files

| file | role |
|---|---|
| `tt/model_config.py` | `ROW_BUCKETS = (1, 2, 4, 5, 8, 10, 16, 32, 50, 64)`, `SEQ_BUCKETS = (128, 256, 512)`, `select_bucket`, `deployment_buckets`, `parse_buckets`; 128 in `SDPA_MEASURED_SEQ_LENS`; `PortConfig.mlp_grid_y_choices` and `mlp_grid_rows` (GeGLU grid rule); `PortConfig.shard_yields_to_grid_rows`; the sharded plan declines for bf16 weights; `STAGE3_PORT` keeps the stage 3 port |
| `tt/runner.py` | trace bytes per capture (`ttnn.get_memory_view(device, ttnn.BufferType.TRACE)` before and after `end_trace_capture`), build, eager and capture seconds per bucket, `trace_bytes_total`, `describe()` |
| `tt/engine.py` | host tail timed per call (`last_host_tail_ms`), trace bytes in `shapes()`; the default bucket set is the deployment set |
| `tests/bench_latency.py` | device-side latency ladder: the STATE_EN speed-table rows through the server's request-to-rows path, modes `deployment` (all buckets captured), `fresh` (one process per cell), `throughput` (B 64 at S 128, 256, 512); end to end, device, host tail, decode and the blocking write / replay / readback split |
| `tests/perf_summary.py` | `profile` (one eager forward under Tracy), `reconcile` (segments the per-op CSV into masks, embeddings, the 28-layer stack, final norm, type add, head layers, scorer, CLS slice and compares with the traced p50), `summary` (writes `perf_summary.json` and `latency_table.json`) |
| `tests/replay_trace_check.py` | `--buckets all`, inputs that fit each seq bucket, trace bytes and loads recorded |
| `tests/run_fidelity.py`, `tests/decision_agreement.py` | `--seq-buckets`, `--row-buckets` (defaults unchanged: the stage 6 protocol), bucket histogram per run, `--hidden-cache` for the CPU reference hidden states |
| `tests/bench_buckets.py` | `--buckets all`, runner description (trace bytes, warmup split) in the JSON |
| `tests/laya_inputs.py` | `build_inputs(..., max_len=None)` so typed-decisions rows fit short buckets |
| `tests/test_performant.py` | pytest gates over the evidence files of this directory (`LAYA_PERFORMANT_LIVE=1` captures two buckets on the device) |
| `bench_all_buckets_default.json` | the first run of the 30 buckets on the stage 3 plans (smoke, 10 replays) |
| `ab/bench_*.json` | the A/B matrix (one process per variant, 20 replays) |
| `bench_all_buckets_final.json` | the final configuration, all 30 buckets, 20 replays |
| `replay_trace_check_all_buckets.json` | the tracked replay over all 30 buckets in one process |
| `latency_deployment.json`, `latency_fresh_{1,5,10,50}.json`, `latency_throughput.json` | the latency ladder |
| `tracy/forward_b{1,5,50,64}s256`, `tracy/forward_b64s512` | Tracy op reports of one eager forward (`perf_report.csv` from `tt-perf-report`) |
| `reconcile_b*.json` | the reconciliation per profiled bucket |
| `fidelity_bf8w_hifi3_erf_final_buckets.json`, `decision_agreement_final_buckets.json`, `decision_agreement_final_port_seq512.json` | the stage 6 gates on the final configuration |
| `perf_summary.json`, `latency_table.json` | the assembled evidence |
| `work_log.md` | timeline and decisions |

## Bucket decision

Input (stage 6, `../full_model/served_token_lengths.json`): speed-table rows are 194 to 205 tokens; E3 rows p50 51
(Emotion) and 103 (AG News, 91 percent at or under 128); E2 rows p50 308 with 32 percent at or under 256 and 91 percent
at or under 384; the published cells are 1, 5, 10 and 50 questions, and the batch cells are 40, 80, 160, 320 and 640
rows. Decision (amendment A10, confirmed here): seq buckets 128, 256 and 512; row buckets 1, 2, 4, 5, 8, 10, 16, 32,
50, 64; 30 traces. Selection rule (`model_config.select_bucket`, the same rule as `server/engine.py: Buckets`): the
smallest seq bucket that fits the longest row of the call, then the smallest row bucket that fits the rows. A 384
bucket was not added: it would serve 58 percent of the E2 rows at 10 more traces, and E2 is one call per 5-question
case whose longest row decides (the gate run below shows 25 percent of the cases at 5x256, 75 percent at 5x512).

What the served workloads map to: E5 cells 1x256, 5x256, 10x256, 50x256 (build 0 ran them at 1x512, 8x512, 16x512,
64x512); E5 batch 8 states x 5 questions = 40 rows at 50x256 (build 0: 64x512), 64 x 5 at 64x256 x 5 calls, 8 x 10 =
80 rows at 64x256 plus 16x256; E3 at 1x128 (Emotion all, AG News 91 percent) else 1x256; E2 at 5x256 or 5x512.

## Per-bucket plans of the final configuration (`DEFAULT_PORT`, policy `bf8w_hifi3_erf`)

The stage 3 levers are keyed by rows = B x S, so the new buckets inherit them: L1 attention chain to 4096 rows (SDPA
chunk 128 there), DRAM above; `minimal_matmul` on 11x10 for Wqkv everywhere and for Wo from 4096 rows; SDPA on 8x8 with
chunk 128 below 768 rows and 256 above; GeGLU padded to 2816. New in stage 7: the interleaved GeGLU program config takes
the first grid-row count in (10, 8, 5, 4, 2, 1) that divides the tile rows (`mlp_grid_y_choices`), so 640, 1280, 2560,
5120, 6400, 12800 and 25600 rows run on 11x10 (110 cores) and 128 rows on 11x4; the block-sharded plan (1024 to 2048
rows) yields to that config where the 11x10 grid applies (1280 rows). The row counts 128 and 640 had no explicit
config before (ttnn's automatic matmul); 128-token sequences had no SDPA program config (128 was missing from the
measured-length table).

| rows x seq | rows | attention chain | GeGLU plan (grid) | Wqkv | Wo | SDPA chunk (8x8) |
|---|---|---|---|---|---|---|
| 1x128 | 128 | L1 | interleaved 2816 (11x4) | minimal 11x10 | core grid 8x8 | 128 |
| 2x128 | 256 | L1 | interleaved 2816 (11x8) | minimal 11x10 | core grid 8x8 | 128 |
| 4x128 | 512 | L1 | interleaved 2816 (11x8) | minimal 11x10 | core grid 8x8 | 128 |
| 5x128 | 640 | L1 | interleaved 2816 (11x10) | minimal 11x10 | core grid 8x8 | 128 |
| 8x128 | 1024 | L1 | block-sharded 2816 (8x8) | minimal 11x10 | core grid 8x8 | 128 |
| 10x128 | 1280 | L1 | interleaved 2816 (11x10) | minimal 11x10 | core grid 8x8 | 128 |
| 16x128 | 2048 | L1 | block-sharded 2816 (8x8) | minimal 11x10 | core grid 8x8 | 128 |
| 32x128 | 4096 | L1 | interleaved 2816 (11x8) | minimal 11x10 | minimal 11x10 | 128 |
| 50x128 | 6400 | DRAM | interleaved 2816 (11x10) | minimal 11x10 | minimal 11x10 | 128 |
| 64x128 | 8192 | DRAM | interleaved 2816 (11x8) | minimal 11x10 | minimal 11x10 | 128 |
| 1x256 | 256 | L1 | interleaved 2816 (11x8) | minimal 11x10 | core grid 8x8 | 128 |
| 2x256 | 512 | L1 | interleaved 2816 (11x8) | minimal 11x10 | core grid 8x8 | 128 |
| 4x256 | 1024 | L1 | block-sharded 2816 (8x8) | minimal 11x10 | core grid 8x8 | 256 |
| 5x256 | 1280 | L1 | interleaved 2816 (11x10) | minimal 11x10 | core grid 8x8 | 256 |
| 8x256 | 2048 | L1 | block-sharded 2816 (8x8) | minimal 11x10 | core grid 8x8 | 256 |
| 10x256 | 2560 | L1 | interleaved 2816 (11x10) | minimal 11x10 | core grid 8x8 | 256 |
| 16x256 | 4096 | L1 | interleaved 2816 (11x8) | minimal 11x10 | minimal 11x10 | 128 |
| 32x256 | 8192 | DRAM | interleaved 2816 (11x8) | minimal 11x10 | minimal 11x10 | 256 |
| 50x256 | 12800 | DRAM | interleaved 2816 (11x10) | minimal 11x10 | minimal 11x10 | 256 |
| 64x256 | 16384 | DRAM | interleaved 2816 (11x8) | minimal 11x10 | minimal 11x10 | 256 |
| 1x512 | 512 | L1 | interleaved 2816 (11x8) | minimal 11x10 | core grid 8x8 | 128 |
| 2x512 | 1024 | L1 | block-sharded 2816 (8x8) | minimal 11x10 | core grid 8x8 | 256 |
| 4x512 | 2048 | L1 | block-sharded 2816 (8x8) | minimal 11x10 | core grid 8x8 | 256 |
| 5x512 | 2560 | L1 | interleaved 2816 (11x10) | minimal 11x10 | core grid 8x8 | 256 |
| 8x512 | 4096 | L1 | interleaved 2816 (11x8) | minimal 11x10 | minimal 11x10 | 128 |
| 10x512 | 5120 | DRAM | interleaved 2816 (11x10) | minimal 11x10 | minimal 11x10 | 256 |
| 16x512 | 8192 | DRAM | interleaved 2816 (11x8) | minimal 11x10 | minimal 11x10 | 256 |
| 32x512 | 16384 | DRAM | interleaved 2816 (11x8) | minimal 11x10 | minimal 11x10 | 256 |
| 50x512 | 25600 | DRAM | interleaved 2816 (11x10) | minimal 11x10 | minimal 11x10 | 256 |
| 64x512 | 32768 | DRAM | interleaved 2816 (11x8) | minimal 11x10 | minimal 11x10 | 256 |

Rotary is never sharded; the head layers reuse the attention plan; the scorer's last linear is fp32. Policies with
bf16 matmul weights (`bf16_hifi4`, `bf16w_hifi3`) run the interleaved config at every bucket: with bf16 weights the
block-sharded down projection overflows L1 at 1024 rows (found by the stage 8 sweep, which runs every policy over the
whole deployment set; stage 3 had measured those policies only at 1x512, 8x512 and 64x512).

## Method

`tests/bench_buckets.py --buckets all` (eager p50 of 3, then one trace per bucket, p50 of 20 replays after 3 warm, real
typed-decisions rows built at the bucket's sequence budget, bit identity of traced against eager per cell, load per
cell). A/B pairs run one process per variant with the same protocol, against a `default_pairs` arm run in the same
queue; run-to-run scatter of the same configuration between the smoke run and `default_pairs` is within 0.9 percent at
all 24 shared buckets (load 0.4 to 3.3), so differences above about 2 percent are readable. The latency ladder uses the
exact server path for the STATE_EN rows (`server/engine.py: encode_state` on the vendored builder, 194 to 205 tokens)
through `LayaEngine.forward_detailed` plus the temperature softmax of the decode step, 3 warm calls and p50 of 20, then
a blocking split of 10 (`run_timed`: input write, `execute_trace(blocking=True)`, the two readbacks). Reconciliation:
one eager forward per bucket under `python -m tracy -r -p -v`, `tt-perf-report` CSV, the second of two passes.

## All 30 buckets on the final configuration (`bench_all_buckets_final.json`, load 2.2 to 3.7)

| rows x seq | traced p50 ms | min | p95 | eager p50 | rows per s | tokens per s (padded) | trace MiB | traced == eager |
|---|---|---|---|---|---|---|---|---|
| 1x128 | 7.92 | 7.90 | 7.97 | 8.33 | 126.3 | 16.2 k | 3.6 | yes |
| 2x128 | 8.89 | 8.85 | 8.94 | 9.62 | 224.9 | 28.8 k | 4.1 | yes |
| 4x128 | 11.75 | 11.67 | 11.81 | 12.10 | 340.4 | 43.6 k | 4.1 | yes |
| 5x128 | 13.37 | 13.33 | 13.49 | 13.65 | 373.9 | 47.9 k | 4.2 | yes |
| 8x128 | 18.91 | 18.81 | 18.99 | 19.39 | 423.0 | 54.1 k | 4.6 | yes |
| 10x128 | 21.66 | 21.52 | 21.75 | 22.13 | 461.8 | 59.1 k | 4.3 | yes |
| 16x128 | 33.69 | 33.64 | 33.88 | 34.15 | 474.9 | 60.8 k | 4.8 | yes |
| 32x128 | 59.44 | 59.09 | 59.60 | 59.87 | 538.4 | 68.9 k | 5.1 | yes |
| 50x128 | 98.25 | 97.94 | 98.71 | 98.43 | 508.9 | 65.1 k | 5.2 | yes |
| 64x128 | 127.92 | 127.49 | 128.33 | 128.13 | 500.3 | 64.0 k | 5.1 | yes |
| 1x256 | 9.25 | 9.20 | 9.34 | 9.52 | 108.2 | 27.7 k | 4.1 | yes |
| 2x256 | 12.14 | 12.05 | 12.17 | 12.41 | 164.7 | 42.2 k | 4.1 | yes |
| 4x256 | 19.52 | 19.39 | 19.63 | 19.73 | 205.0 | 52.5 k | 4.6 | yes |
| 5x256 | 22.78 | 22.72 | 22.88 | 23.03 | 219.5 | 56.2 k | 4.4 | yes |
| 8x256 | 34.56 | 34.43 | 34.64 | 34.98 | 231.5 | 59.3 k | 4.8 | yes |
| 10x256 | 40.35 | 40.27 | 40.47 | 40.84 | 247.8 | 63.4 k | 4.6 | yes |
| 16x256 | 62.05 | 61.99 | 62.62 | 62.43 | 257.9 | 66.0 k | 5.1 | yes |
| 32x256 | 129.52 | 129.24 | 129.87 | 129.75 | 247.1 | 63.2 k | 5.1 | yes |
| 50x256 | 192.45 | 192.11 | 192.97 | 192.24 | 259.8 | 66.5 k | 5.2 | yes |
| 64x256 | 253.23 | 252.71 | 253.47 | 253.23 | 252.7 | 64.7 k | 5.1 | yes |
| 1x512 | 12.57 | 12.49 | 12.67 | 12.81 | 79.6 | 40.7 k | 4.1 | yes |
| 2x512 | 20.35 | 20.30 | 20.37 | 20.64 | 98.3 | 50.3 k | 4.6 | yes |
| 4x512 | 36.08 | 35.99 | 36.21 | 36.15 | 110.9 | 56.8 k | 4.8 | yes |
| 5x512 | 45.06 | 44.98 | 45.22 | 45.32 | 111.0 | 56.8 k | 4.6 | yes |
| 8x512 | 65.09 | 64.93 | 65.19 | 65.34 | 122.9 | 62.9 k | 5.1 | yes |
| 10x512 | 88.57 | 88.35 | 88.92 | 88.74 | 112.9 | 57.8 k | 5.2 | yes |
| 16x512 | 136.66 | 136.33 | 137.00 | 137.12 | 117.1 | 59.9 k | 5.1 | yes |
| 32x512 | 267.35 | 267.16 | 267.55 | 268.40 | 119.7 | 61.3 k | 5.1 | yes |
| 50x512 | 427.51 | 426.76 | 427.96 | 427.93 | 117.0 | 59.9 k | 5.2 | yes |
| 64x512 | 524.90 | 523.99 | 525.37 | 525.09 | 121.9 | 62.4 k | 5.1 | yes |

Against stage 3 (same policy, stage 3 port): 1x512 12.96 -> 12.57, 2x512 21.01 -> 20.35, 4x512 36.71 -> 36.08, 8x512
65.81 -> 65.09, 16x512 137.57 -> 136.66, 32x512 268.83 -> 267.35, 64x512 528.23 -> 524.90 ms (the 512 buckets kept their
plans; the differences are scatter plus the 1 to 3 percent lower load). The fixed cost floor is about 7.5 ms (1x128:
450 device ops at roughly 15 us each); the device reaches 60 to 69 k padded tokens per second from 2048 rows.

## Trace memory and warmup

- Trace bytes per captured bucket: 3.6 MiB (1x128) to 5.2 MiB (50x128, 50x256, 10x512, 50x512); the size follows the
  op count (456 ops per forward, two extra reshards per layer on the sharded buckets), not the batch. Total for the 30
  buckets: 141.5 MiB (148,373,504 bytes) of the 512 MiB trace region (27.6 percent); `trace_region_size` stays at
  512 MiB (`LAYA_TRACE_REGION_SIZE` default). Measured with `ttnn.get_memory_view(device, ttnn.BufferType.TRACE)`
  before and after every capture; the region's allocated bytes after the 30 captures equal the sum.
- Two-phase warmup over the deployment set in `LayaEngine` (the server's path, `latency_deployment.json`): phase 1
  (build the 30 buckets 0.35 s, run each once eagerly 3.13 s) 3.48 s; phase 2 (30 captures) 0.53 s; engine load
  (device open, config, safetensors, weight upload 1.69 s, warmup) 7.0 s in all. The replay check and the bench runs
  measured the same warmup (2.79 s plus 0.53 to 0.58 s with the buckets built beforehand). A fresh process with one
  bucket loads in 3.1 to 3.3 s (warmup 0.06 to 0.23 s plus 0.015 s capture). Every trace is captured in the lifespan
  before the server logs ready (`server/app.py`), as in stage 6.
- Device memory beyond the traces: static masks `band` and `zeros` per bucket (`2 x B x S x S x 2` bytes: 192 MiB at
  S 512, 48 MiB at 256, 12 MiB at 128 over the ten row buckets), the per-bucket trace outputs (at most 4 MiB each),
  the weights (unchanged). No bucket is dropped.

## A/B results (traced p50 ms, 20 replays after 3 warm; delta against `default_pairs` from the same queue)

| variant (`PortConfig` overrides) | cells: default -> variant (delta percent) | outcome |
|---|---|---|
| `mlp_grid_11x10` (GeGLU config on 11x10) | 5x128 19.79 -> 13.38 (-32.4); 10x256 42.64 -> 40.59 (-4.8); 5x512 47.17 -> 45.22 (-4.1); 50x128 103.43 -> 98.52 (-4.7); 50x256 204.63 -> 193.68 (-5.4); 50x512 450.98 -> 428.96 (-4.9) | adopted as the grid rule |
| `mlp_grid_11x5`, `mlp_grid_11x4` at 5x128 | 19.79 -> 15.18 (-23.3), 16.30 (-17.6); 11x4 at 10x256 +25.5 | 11x10 wins where legal |
| grid rule confirmation (`gridrule_default` against `gridrule_old_8_only`) | 1x128 8.35 -> 7.92 (-5.2, the 11x4 config replaces ttnn's automatic one); 5x128 19.86 -> 13.65 (-31.3); 10x256 -5.3; 5x512 -4.4; 50x128 -5.0; 50x256 -5.6; 50x512 -5.1; untouched buckets -0.2 to +1.2 | confirmed |
| `geglu_interleaved` under erf (R1 deferred item) | 2x512 20.37 -> 20.24 (-0.6); 4x512 36.11 -> 35.60 (-1.4); 4x256 -0.1; 8x256 -1.5; 8x128 -0.7; 16x128 -1.3 (all 11x8); with the new grid at 1280 rows: 5x256 23.66 -> 22.86 (-3.3), 10x128 22.32 -> 21.65 (-3.0) | dead heat at 1024 and 2048 rows, kept sharded there (the stage 6 validated numerics); interleaved 11x10 at 1280 rows |
| `wo_minimal_all` (`wo_minimal_min_rows` 0, R1 deferred clean pair) | 2x512 20.37 -> 20.55 (+0.9); 4x512 36.11 -> 36.08 (-0.1); 1x128 +2.4; 5x128 -2.3; 1x256 +0.3; 5x256 +1.3; 8x256 +0.3; 10x256 -1.4; 5x512 -1.4 | within scatter; threshold 4096 unchanged |
| `sdpa_128` (chunks 128 above 768 rows) | 4x256 -0.1; 5x256 +1.4; 8x256 +0.6; 10x256 +4.3; 32x256 +2.0; 50x256 +5.7; 64x256 +2.2; 2x512 0.0; 4x512 +1.5; 5x512 +4.3 | loses; 256 kept |
| `sdpa_64` | 1x128 -0.3; 2x128 +0.5; 4x128 +1.4; 5x128 +2.4; 1x256 -1.0; 2x256 +2.1; 5x256 +9.6 | loses or scatter; kept |
| `sdpa_full_grid` (11x10) | 1x128 +0.9; 5x128 -0.3; 1x256 +2.6; 5x256 -3.1; 10x256 -0.5; 50x256 -0.8 | within scatter; 8x8 kept |
| `chain_dram` (`l1_attention_max_rows` 0) | 1x128 +6.3; 5x128 +6.9; 1x256 +8.5; 5x256 +12.2; 10x256 +14.4 | L1 chain kept |
| `chain_l1_8192` | fails at 32x256 / 64x128 / 16x512: static circular buffers clash with L1 buffers (program 8) | threshold stays 4096 |
| `qkv_auto` (Wqkv on ttnn's matmul) | 1x128 -2.8; 2x128 +7.1; 5x128 +13.1; 1x256 +0.5 | `minimal_matmul` kept |

Loads during the A/B queue: 0.4 to 3.4 (all cells under the limit of 8). JSON: `ab/bench_<variant>.json`; the two R1
deferred pairs are `ab/bench_geglu_interleaved.json`, `ab/bench_gridrule_geglu_interleaved.json` and
`ab/bench_wo_minimal_all.json` against `ab/bench_default_pairs.json` and `ab/bench_gridrule_default.json`.

## Published cells (`latency_table.json`; policy `bf8w_hifi3_erf`; STATE_EN rows 194 to 205 tokens)

End to end = input write + trace replay + two readbacks + host tail + temperature softmax, in process without HTTP.
"Deployment" = all 30 buckets captured before the first timed call; "fresh" = one process per cell with only that
bucket captured. Build 0 = the host-served client and device p50 of stage 6 at seq bucket 512.

| questions | bucket | deployment p50 (min, p95) | fresh p50 (min) | device (replay only) | host tail | T4 published | build 0 client (device) at bucket | load deploy / fresh |
|---|---|---|---|---|---|---|---|---|
| 1 | 1x256 | 9.43 ms (9.37, 9.54) | 9.40 ms (9.36) | 9.18 (9.06) | 0.12 ms | 39.5 ms | 14.5 (12.5) at 1x512 | 1.3 / 1.6 |
| 5 | 5x256 | 22.92 ms (22.86, 23.33) | 22.92 ms (22.86) | 22.63 (22.41) | 0.13 ms | 84.5 ms | 68.9 (64.7) at 8x512 | 1.6 / 1.5 |
| 10 | 10x256 | 40.68 ms (40.51, 41.02) | 40.71 ms (40.49) | 40.24 (39.84) | 0.18 ms | 158.6 ms | 142.9 (136.0) at 16x512 | 1.6 / 2.7 |
| 50 | 50x256 | 192.95 ms (192.32, 193.42) | 193.20 ms (192.67) | 191.59 (189.85) | 0.46 ms | 771 ms | 545.1 (522.8) at 64x512 | 1.7 / 2.3 |

- Multi-trace penalty (deployment against fresh): +0.03, 0.00, -0.02 and -0.25 ms (+0.3 to -0.1 percent); 30 captured
  traces cost nothing measurable per replay on this box.
- Against build 0's device time: -27 percent (1 question), -65 percent (5), -70 percent (10), -63 percent (50); the
  gains come from the 256 bucket (every speed-table row was padded to 512) and the exact row buckets (5 and 10 rows
  were padded to 8 and 16, 50 to 64).
- Host tail per call (gather at the markers, -1e4 fill, softmax, four features, act head, fp32): 0.12 ms at 1 row,
  0.13 at 5, 0.18 at 10, 0.46 at 50; the temperature softmax of the decode step 0.006 to 0.02 ms; the blocking split
  puts the input write at 0.02 to 0.16 ms and the two readbacks at 0.09 to 1.57 ms. The device replay is 97 to 99
  percent of the end-to-end time.

## B 64 throughput (`latency_throughput.json`; 3 warm, p50 of 20; device-only = `execute_trace(blocking=True)`)

| cell | rows | row content | end to end p50 | replay only | rows per s (end to end / device only) | real tokens per s | padded tokens per s | load |
|---|---|---|---|---|---|---|---|---|
| 64x128 | 64 | short state, about 40 tokens | 128.96 ms | 126.05 ms | 496.3 / 507.7 | 20.6 k | 63.5 k | 2.4 |
| 64x256 | 64 | STATE_EN rows, 194 to 205 tokens | 255.05 ms | 251.18 ms | 250.9 / 254.8 | 50.1 k | 64.2 k | 2.3 |
| 64x512 | 64 | typed-decisions rows, one filled to 512 | 528.48 ms | 523.33 ms | 121.1 / 122.3 | 21.0 k | 62.0 k | 2.0 |

The all-bucket bench (typed-decisions rows, 20 replays) gives the same picture: 64x128 127.92 ms, 64x256 253.23 ms,
64x512 524.90 ms. The 64x128 cell carries 4 times the rows per second of 64x512 at the same padded token rate (62 to
64 k per second), so the device is bandwidth bound at B 64 and the seq bucket decides the useful work per call.

Stage 4's 1x4 mesh (16 rows per chip, 64 per call) measured 3.74x the single-chip B 64 rate at seq 512; it is served
in the final check as the `p150x4` column.

## Lower-bound reconciliation (`reconcile_b*.json`; device kernel time of one eager forward against the traced p50)

| bucket | ops per pass | device kernel sum (ms) | 28 x layer + 2 x head layer + scorer (ms) | layer (us) | head layer (us) | scorer (us) | other (masks, embeddings, final norm, type add, CLS) (us) | traced p50 (ms) | gap (ms, percent) | load |
|---|---|---|---|---|---|---|---|---|---|---|
| 1x256 | 456 | 8.90 | 8.80 | 292.0 | 280.0 | 66.4 | 94.8 | 9.25 | 0.35 (3.8) | 1.7 |
| 5x256 | 456 | 22.12 | 22.00 | 723.5 | 769.0 | 200.3 | 126.8 | 22.78 | 0.66 (2.9) | 2.9 |
| 50x256 | 456 | 186.19 | 185.51 | 6032.9 | 7286.8 | 2016.1 | 682.6 | 192.45 | 6.26 (3.3) | 2.4 |
| 64x256 | 456 | 245.36 | 244.47 | 7989.8 | 9108.1 | 2542.8 | 886.9 | 253.23 | 7.87 (3.1) | 1.4 |
| 64x512 | 456 | 511.54 | 509.31 | 16800.1 | 17177.2 | 4553.8 | 2225.2 | 524.90 | 13.36 (2.5) | 3.3 |

The gap (2.5 to 3.8 percent) is the op-to-op dispatch inside the trace plus the input write and the readbacks; the
whole forward is 456 device ops (28 layers of 15 ops, layer 0 without the attention norm, 25 head-layer ops, 4 scorer
ops, masks, embeddings, final norm, type add, CLS slice). The 64x512 layer time (16.80 ms) equals stage 3's layer
profile (16.80 ms, `../optimized_decoder/README.md`). Where the time goes at 1x256 (fixed-cost regime): the two Wi
matmuls 20 percent, LayerNorm 13 percent (62 calls on 8 cores at 256 rows), Wqkv 13 percent, Wo 10 percent, head
split 10 percent, SDPA 9 percent, the three elementwise ops 7 percent, rotary 6 percent. At 50x256 (bandwidth regime):
Wi 25 percent, the 91 elementwise ops (two residual adds and the gate multiply per layer, the mask adds, the type add)
13 percent, Wqkv 10 percent, rotary 9 percent, SDPA 9 percent, Wo 8 percent, head split 8 percent, LayerNorm 5 percent.
Two observations for later work: the head layers' ReLU runs as a separate `UnaryDeviceOperation` (two calls, 690 us
each at 64x256; the stage 2 README said it was fused, a dated note is appended there), and the residual adds are DRAM
bound at large batches (a fused add in the Wo matmul would remove 60 of the 91 elementwise ops).

## Trace-safety gate over all 30 buckets in one process (closes the R1 deferred item)

`TT_METAL_TRACE_ALLOC_TRACKING=1 tests/replay_trace_check.py --buckets all --rounds 3 --repeats 5 --eager-repeats 2`
(`replay_trace_check_all_buckets.json`, job `s7_measure2`, log
`/home/hous/dev/laya/logs/p3_s7_measure2_20261005T234359Z.log`): the 30 buckets built, run eagerly and captured in one
process, then three rounds in forward and reverse bucket order with two inputs per bucket (the first input replayed 5
extra times in round 0), 10 replays per bucket, 300 replays in all. Result: `pass` true, tracker error none, traced
equals eager bit for bit at every bucket (logits and CLS, max abs delta 0.0), repeated replays identical, a changed input
moves the logits by 1.79 to 5.27, no NaN; trace bytes 141.5 MiB; warmup 2.80 s plus 0.58 s; load 1.5 to 2.4. The
timings in that file are not usable (the tracker runs `gc.collect()` before every replay, as stages 2 and 4 recorded).
The same run on the first candidate configuration (interleaved GeGLU everywhere, `s7_measure`) also passed.

## Stage 6 gates on the final configuration

`tests/run_fidelity.py --policy bf8w_hifi3_erf --items gate --hidden-cases 40 --seq-buckets 128,256,512 --row-buckets
1,2,4,5,8,10,16,32,50,64` (`fidelity_bf8w_hifi3_erf_final_buckets.json`; the 40 five-question calls ran at 5x256 (10
calls) and 5x512 (30 calls); load 3.2 to 3.7):

| gate | measured | threshold | result |
|---|---|---|---|
| confident argmax agreement | 149 of 149 | >= 98 percent | pass |
| median over decisions of max abs delta p | 0.01104 | <= 0.02 | pass |
| scorer-logit PCC | 0.9965 | >= 0.99 | pass |
| hidden-state PCC encoder / head (pooled, 40 cases) | 0.9957 (worst call 0.9918) / 0.9996 (worst call 0.9975) | >= 0.99 | pass |
| NaN | 0 | 0 | pass |

Reported, not gated: plain argmax agreement 195 of 200 (97.5 percent), act argmax agreement 200 of 200 (act-logit PCC
0.99994), p95 of max abs delta p 0.0300, max 0.0760, max abs logit delta 1.52. Stage 6 at 8x512: 194 of 200, median
0.0108, PCC 0.9962, hidden 0.9955 / 0.9996 (the same decisions within placement noise).

Alone versus in batch (`tests/decision_agreement.py` with the final buckets, `decision_agreement_final_buckets.json`,
load 0.6 to 2.1): 16 of 16 the same argmax in all five placements; max abs delta p alone against B 2 / B 4 / mixed B 8 /
B 64 = 0.00587 / 0.00587 / 0.00890 / 0.00897 (gate 0.01): pass. The alone calls ran at 1x256 (5 rows) and 1x512 (11),
B 2 at 2x256 (1) and 2x512 (7), B 4 at 4x512, mixed B 8 at 8x512, B 64 at 64x512. On the stage 6 protocol (512 only,
`decision_agreement_final_port_seq512.json`) the final port reproduces stage 6's value to the last digit (0.009276; 16
of 16), so the final configuration equals the stage 6 numerics at the 512 buckets.

The first candidate (interleaved GeGLU at every bucket) missed this gate by 0.0005 (0.0105 alone against B 2 and B 4
at 2x512 and 4x512, the buckets whose plan had changed; `work_log.md` 23:41) while passing every fidelity gate; the
block-sharded plan was kept at 1024 and 2048 rows for that reason (it is also a dead heat in time).

## Served check with the final buckets (host server, build 1)

`/home/hous/dev/laya/bin/serve-tt.sh` under devlock with `LAYA_SEQ_BUCKETS=128,256,512`,
`LAYA_ROW_BUCKETS=1,2,4,5,8,10,16,32,50,64`, `LAYA_PRECISION=bf8w_hifi3_erf`, `LAYA_RAW_FORWARD=1`: the server loaded in
7.0 s (warmup 3.5 s plus 0.54 s, 30 traces, 148,373,504 trace bytes) and logged ready with every trace captured; then
`TARGET=host_tt PROFILE=p150 BUILD=1 RAW_FORWARD=1 bash /home/hous/dev/laya/bin/run-evals.sh` wrote
`/home/hous/dev/laya/evals/results/host_tt_p150_b1_20261006T000400Z/SUMMARY.md` (E1 to E5, demo feed, health before and
after; the `p150x4` column merged from `.../host_tt_p150x4_b1_20261006T000708Z`, served once with `LAYA_MESH_SHAPE=1x4`
on chips 0 to 3, `LAYA_MAX_ROWS=256`; mesh load 11.6 s, warmup 8.1 s plus 0.7 s). Load 1.2 to 2.7 throughout.

| questions per call | T4 (published) | p150 client p50, build 1 | p150 server | p150 device forward | p150 bucket | build 0 client (device) | p150x4 client p50 | p150x4 device |
|---|---|---|---|---|---|---|---|---|
| 1 | 39.5 ms | 11.0 ms | 10.2 ms | 9.2 ms | 1x256 | 14.5 (12.5) | 13.3 ms | 10.7 ms |
| 5 | 84.5 ms | 26.5 ms (5.3 ms per question) | 25.7 ms | 22.7 ms | 5x256 | 68.9 (64.7) | 16.8 ms | 12.3 ms |
| 10 | 158.6 ms | 46.0 ms (4.6) | 45.0 ms | 40.2 ms | 10x256 | 142.9 (136.0) | 26.5 ms | 19.8 ms |
| 50 | 771 ms | 212.9 ms (4.3) | 211.1 ms | 191.7 ms | 50x256 | 545.1 (522.8) | 85.4 ms | 63.0 ms |

Batched throughput (`/v1/systemone/batch`, warm 3, reps 10): 187 to 231 questions per second on one p150 (build 0: 70
to 117; T4 published 103 to 332): 1x5 187.4 (5x256), 1x10 215.7 (10x256), 8x5 190.6 (50x256), 8x10 230.5 (16x256 plus
64x256), 32x5 229.0, 32x10 230.8, 64x5 230.2, 64x10 231.2 (64x256 calls). On the 1x4 mesh 299 to 712 questions per
second (8x5 651.8 at one 40x256 call, 64x10 711.0 at 128x256 plus 256x256 calls). `X-Laya-Batch` histogram over the
whole E5 run: 64x256 230, 10x256 25, 50x256 25, 5x256 25, 1x256 15, 16x256 10, 32x256 10 (build 0: 64x512 255, 16x512
35, 8x512 25, 1x512 15, 32x512 10). E2 ran at 5x256 (110 cases) and 5x512 (290); E3 at 1x128 (765 of 800 cases) and
1x256 (35). The device time of the served cells equals the in-process ladder within 0.3 ms; the served client time
carries 1.8 to 21 ms more (request tokenization and decode in the server, 22 ms at 50 questions in build 0 too), now
10 percent of the 50-question cell; recorded as an open item for the serving track. Accuracy tables of the build 1 run
and the stage 8 policy decision: `../datatype_sweep/README.md`.

## What proved wrong or incomplete in the plan, and open items

- Appendix A.5 expected "about 20 traces" and the trace region to be the constraint; 30 traces take 141.5 MiB of 512.
- The stage 3 plan tables were keyed by rows and silently left two row counts without an explicit GeGLU config (128
  and 640 rows) and one sequence length without an SDPA config (128); the grid rule and the table entry fix both.
- The sharded GeGLU plan's stage 3 win (under tanh, 5 percent) does not hold under erf: dead heat at 1024 and 2048
  rows, a 3 percent loss at 1280 rows against the 11x10 interleaved config.
- `bf16_hifi4` and `bf16w_hifi3` cannot run the block-sharded plan (L1 clash); the plan declines for bf16 weights.
- The head layers' ReLU is a separate op (0.5 percent at 64x256); the 91 elementwise ops are 13 percent at large
  batches; LayerNorm runs on 8 cores at 256 rows. None is pursued in this stage.
- The all-bucket tracked replay shows the once-per-process allocator warning of stage 2 for every capture after the
  first; the tracker finds no corrupted buffer over 300 replays.
- The Appendix A.7 sweep gate list has no alone-versus-in-batch gate; stage 8 found a policy (`bf8_act`) that passes
  every A.7 gate and the A11 served confirmation but moves probabilities by 0.038 between buckets (gate 0.01). The
  invariance test must be part of any policy selection; see `../datatype_sweep/README.md`.

## How to run

```
source /home/hous/dev/laya/bin/ttenv.sh; cd $TT_METAL_HOME; A=models/autoports/convaiinnovations_laya; DL=/home/hous/dev/laya/bin/devlock
TT_METAL_VISIBLE_DEVICES=0 $DL python $A/tests/bench_buckets.py --buckets all --variants '{"default": {}}' --out $A/doc/optimized_full_model/bench_all_buckets_final.json
TT_METAL_VISIBLE_DEVICES=0 TT_METAL_TRACE_ALLOC_TRACKING=1 $DL python $A/tests/replay_trace_check.py --buckets all --rounds 3 --repeats 5 --eager-repeats 2 --out $A/doc/optimized_full_model/replay_trace_check_all_buckets.json
TT_METAL_VISIBLE_DEVICES=0 $DL python $A/tests/bench_latency.py --mode deployment --cells 1,5,10,50 --out $A/doc/optimized_full_model/latency_deployment.json
TT_METAL_VISIBLE_DEVICES=0 $DL python $A/tests/bench_latency.py --mode fresh --cells 5 --out $A/doc/optimized_full_model/latency_fresh_5.json
TT_METAL_VISIBLE_DEVICES=0 $DL python $A/tests/bench_latency.py --mode throughput --cells 1 --throughput-cells 64x128,64x256,64x512 --out $A/doc/optimized_full_model/latency_throughput.json
TT_METAL_VISIBLE_DEVICES=0 $DL python -m tracy -r -p -v -o $A/doc/optimized_full_model/tracy/forward_b1s256 $A/tests/perf_summary.py profile --batch 1 --seq 256
tt-perf-report <ops csv> --csv $A/doc/optimized_full_model/tracy/forward_b1s256/perf_report.csv
python $A/tests/perf_summary.py reconcile --perf-csv $A/doc/optimized_full_model/tracy/forward_b1s256/perf_report.csv --traced-ms 9.25 --out $A/doc/optimized_full_model/reconcile_b1s256.json
python $A/tests/perf_summary.py summary --deployment $A/doc/optimized_full_model/latency_deployment.json --fidelity $A/doc/optimized_full_model/fidelity_bf8w_hifi3_erf_final_buckets.json --agreement $A/doc/optimized_full_model/decision_agreement_final_buckets.json
TT_METAL_VISIBLE_DEVICES=0 $DL python $A/tests/run_fidelity.py --policy bf8w_hifi3_erf --items gate --hidden-cases 40 --seq-buckets 128,256,512 --row-buckets 1,2,4,5,8,10,16,32,50,64 --hidden-cache /home/hous/dev/laya/state/tt_cache/fidelity_hidden_ref_gate40.pt --out $A/doc/optimized_full_model/fidelity_bf8w_hifi3_erf_final_buckets.json
TT_METAL_VISIBLE_DEVICES=0 $DL python $A/tests/decision_agreement.py --policy bf8w_hifi3_erf --seq-buckets 128,256,512 --row-buckets 1,2,4,5,8,10,16,32,50,64 --out $A/doc/optimized_full_model/decision_agreement_final_buckets.json
python -m pytest $A/tests/test_model_config.py $A/tests/test_performant.py -q -p no:cacheprovider -o addopts=""
```

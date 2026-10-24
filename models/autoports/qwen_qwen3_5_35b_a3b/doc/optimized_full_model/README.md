# Qwen3.5-35B-A3B optimized full model: stage 07

Abbreviations: TTFT (time to first token), t/s/u (tokens per second per user), t/s (tokens per second), ISL
(input sequence length, tokens), B (decode batch: active rows in one decode step), HF (Hugging Face CPU
reference), TT (Tenstorrent TTNN implementation), TTNN (Tenstorrent neural network library), KV (key/value cache),
GDN (Gated DeltaNet linear-attention layer), MoE (mixture of experts), EP (expert parallel), TP (tensor parallel),
LM head (final vocabulary projection), BF16 (16-bit brain float), BFP4/BFP8 (4/8-bit block float), FP32 (32-bit
float), DRAM (device memory), L1 (core SRAM), SRAM (static RAM), PCC (Pearson correlation coefficient), KL
(Kullback-Leibler divergence, nats), TV (total variation distance), PRNG (pseudo-random number generator), CCL
(collective communication library), EOS (end of sequence token), MiB (2^20 bytes), GB and MB (10^9 and 10^6 bytes).

Profiles: p150 (1 chip), p150x2 (2 chips, TP 2), p150x4 (4 chips, TP 4, the optimization target).

- Base: stage 06 commit `dfe0c6bd` (checkpoint `fb8ce438`).
- Work log: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/work_log.md`.
- Final evidence, batched decode (B > 1): `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence5/` (called `evidence5/` below).
- Final evidence, B1 decode and prefill: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence3/` (called `evidence3/` below).
- Why two roots: every change after `evidence3/` applies only to calls with more than 1 row inside batched decode.
  B1 and prefill programs are unchanged. `evidence5/` B1 repeats match `evidence3/`: token-out 9.11 / 11.30 /
  15.80 ms (`evidence5/m*/full_evidence.json`, `perf_128_128`).
- Raw runs and scripts: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/`.

## Headline

### Batch 1 (ISL 128, greedy, 127 steps, median of 3 runs, `run_batch_perf`)

| Profile | TTFT before | TTFT after | Token-out before | Token-out after | Traced teacher-forcing decode after |
|---|---:|---:|---:|---:|---:|
| p150x4 | 97.8 ms | 97.4 ms | 9.18 ms (108.9 t/s/u) | 9.11 ms (109.8 t/s/u) | 99.83 t/s/u |
| p150x2 | 114.0 ms | 114.3 ms | 11.35 ms (88.1 t/s/u) | 11.30 ms (88.5 t/s/u) | 82.88 t/s/u |
| p150 | 183.4 ms | 183.5 ms | 15.87 ms (63.0 t/s/u) | 15.80 ms (63.3 t/s/u) | 60.85 t/s/u |

- Before: stage 06 `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence5/m*/batch_perf.json`. After: `evidence3/m*/batch_perf_sampled0.json`.
- `evidence5/m*/batch_perf_sampled0.json` B1 repeats: 9.11 / 11.30 / 15.80 ms.
- Teacher-forcing decode: readiness `run_teacher_forcing`, AIME24 chat, traced, logits read every step
  (`evidence3/m*/teacher_forcing_aime24.log`). Stage 06: 99.83 / 83.07 / 60.92 t/s/u.
- B1 gain (0.05 to 0.07 ms) comes from the pipelined token readback (below). B1 layers are unchanged.

### Batched decode (same harness, greedy, ms per step; aggregate t/s in brackets)

| Profile | B4 before | B4 after | B8 after | B16 after | B32 before | B32 after |
|---|---:|---:|---:|---:|---:|---:|
| p150x4 | 35.87 (112) | 13.82 (289) | 18.89 (423) | 28.54 (561) | 282.31 (113) | 47.97 (667) |
| p150x2 | 43.95 (91) | 19.73 (203) | 29.26 (273) | 48.62 (329) | 343.90 (93) | 87.22 (367) |
| p150 | 59.64 (67) | 35.70 (112) | 59.66 (134) | 109.16 (147) | 466.10 (69) | 215.00 (149) |

- Before: stage 06, per-row decode; it measured only B1, B4 and B32.
- After: `evidence5/m*/batch_perf_sampled0.json`.
- Sampled rows (T 1.0, top-k 32, host seeds every step; `evidence5/m*/batch_perf_sampled1.json`) are within
  0.01 ms of greedy, except p150 B32: 222.17 ms (+3.3%; EP dispatch path, data-dependent routing).
- Run spread: at most 0.01 ms within a file, except the first B1 run (includes trace capture, excluded by the
  median) and p150 sampled B32 (1.8 ms).

Definitions:
- Prompt: 128 book tokens per row, a distinct slice per row, one request slot per row.
- TTFT: wall time from `generate()` to the first token on the host (prefill, LM head, device sample, readback).
- Token-out: B1 is the `generate()` time after the first token divided by 127. B4 to B32 time 126 calls of
  `decode_steps` (traced decode replay, traced device sample, token readback overlapped with the next replay).
- t/s/u = 1000 / token-out ms. Aggregate t/s = B x t/s/u.
- B is the number of active decode rows, not HTTP client concurrency. Serving concurrency is stages 10 and 13.
- Tracker off (`QWEN35_STRICT_CACHE=1`). Tracker-on runs (`evidence5/m*/full_evidence.json`, `perf_128_128`):
  TTFT 99.4 / 114.0 / 183.4 ms, token-out 9.11 / 11.30 / 15.80 ms.

## What changed

| Change | Files | Effect (p150x4 unless stated) |
|---|---|---|
| Batched decode: one layer call for B rows | `tt/functional_decoder.py`, `tt/optimized_decoder.py`, `tt/multichip_decoder.py`, `tt/generator.py` | B32 282.31 -> 63.89 ms (row loop / dispatch MoE) |
| LM head for B > 1: one `[1,B,2048]` matmul | `tt/model.py` | 32 one-row matmuls (12.0 ms per step) became one |
| Batched GDN decode kernels (conv, recurrence, gate), per-slot state addresses | `tt/gdn_batch.py`, `tt/kernels/gdn_batch/` | 2-layer B32 6.39 -> 4.27 ms |
| Slot MoE decode for 2 to 32 rows: router linear over B rows, `expert_swiglu_decode` gemv per row slot, indexed `sparse_matmul` down, `fast_reduce_nc` | `tt/optimized_decoder.py` (`moe_slot_rows` 32; 16 on p150) | B4 21.99 -> 17.98, B16 54.61 -> 35.81, B32 63.89 -> 60.23 ms |
| Shared expert uses the decode program config at decode row counts | `tt/optimized_decoder.py` | fixes L1 overflow at B > 1 |
| Pipelined token readback (`decode_steps`): replay i+1 is enqueued before the wait on token i | `tt/generator.py` | B1 9.18 -> 9.11, B32 60.23 -> 60.12 ms |
| Projection fix: batched `_Projection` keeps the decode program config | `tt/optimized_decoder.py` | B4 17.91 -> 16.70, B32 60.12 -> 58.85 ms |
| Decode programs at batched rows (fix2): router linear passes the B1 program config; router top-k and `_Projection` take the prefill branch only outside decode rows (p150 `attention_in` included); expert index sticks built once per layer | `tt/optimized_decoder.py` | B4 16.70 -> 14.58, B32 58.85 -> 52.84 ms; p150 B32 220.85 -> 214.83 ms |
| Slot glue (fix3): routing weights scale the activations before the down `sparse_matmul`; hidden stays `[1,B,2048]` across all layers (`packed`) | `tt/optimized_decoder.py`, `tt/functional_decoder.py`, `tt/model.py` | B4 14.58 -> 13.82, B32 52.84 -> 47.97 ms |
| Slot selectors shared across layers | `tt/model.py` (`_share_slot_selectors`), `tt/optimized_decoder.py` | layer-private DRAM +37.5 -> +1.0 MiB vs stage 06 |
| p150 EP dispatch branch keeps prefill programs (fix5, `PREFILL_PROGRAMS`) | `tt/optimized_decoder.py` | p150 B32 rows with the same prompt identical again; 212.97 -> 215.00 ms |
| Attention: B-core KV shard configs | `tt/optimized_decoder.py` | batched SDPA decode |
| Sampler seed mode `host` default; `skip` raises for TP > 1 | `tt/generator.py` | device token copies stay equal |
| Optional DRAM-sharded LM head (`lm_head_dram`, default off) | `tt/model.py` | measured slower (rejected) |

- `build_generator(batched_decode=True)` is the default for every profile. B1 keeps the per-row path.
- Fix2, fix3 and fix5 trials: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/fix2/`, `fix3/`, `fix5/`.
- The scale-before-down change is exact in real arithmetic: the selector masks and scales the same slot, and down
  is linear. It changes BF16 rounding order (accuracy section).

## Performance accounting

`evidence5/perf_accounting.json` (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/perf_accounting5.py`; B1 rows
from `evidence3/` profiles, B32 rows from `evidence5/` profiles) and
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/perf_summary.json`.

Method: three reduced traced profiles per case, layers (0,3), (0,1,3) and (0,3,7), real weights, 4 warmed
replays, device 0 (device 3 on p150), C++ device report. GDN layer (GDN attention plus its MoE) = (0,1,3) - (0,3).
Full layer (full attention plus its MoE) = (0,3,7) - (0,3). Terminal (embedding, final norm, LM head) = (0,3) -
GDN - full. 40-layer device time = terminal + sampler trace + 30 GDN + 10 full. Roofline:
`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/roofline.py` (stored weight dtypes, expected routed-expert union,
FP32 GDN state read and write, KV at 192 tokens, 512 GB/s per chip).

| Case | End to end | Device (40 layers) | Host gap | Roofline | Roofline / e2e | GDN layer | Full layer | Terminal | Sampler |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p150x4 B1 | 9.11 ms | 9.05 ms | 0.06 ms | 1.50 ms | 16% | 195 us | 224 us | 464 us | 499 us |
| p150x4 B32 | 47.97 ms | 48.51 ms | -0.54 ms | 10.43 ms | 22% | 1257 us | 982 us | 501 us | 497 us |
| p150x2 B1 | 11.30 ms | 11.20 ms | 0.10 ms | 2.91 ms | 26% | 234 us | 251 us | 825 us | 828 us |
| p150x2 B32 | 87.22 ms | 86.89 ms | 0.34 ms | 20.70 ms | 24% | 2336 us | 1512 us | 878 us | 813 us |
| p150 B1 | 15.80 ms | 15.90 ms | -0.10 ms | 5.75 ms | 36% | 324 us | 327 us | 1623 us | 1295 us |
| p150 B32 | 215.00 ms | 220.29 ms | -5.29 ms | 41.32 ms | 19% | 6612 us | 1970 us | 937 us | 1274 us |

- Before (`evidence4/perf_accounting.json`), B32 GDN / full layer: p150x4 1514 / 1247 us, p150x2 2718 / 1874 us,
  p150 6752 / 2073 us.
- Decode is device-bound on every profile. Host gap is at most 0.34 ms.
- Negative host gaps: the profiler adds time. p150 B32 profiled replays rise from 9017 to 9776 us over 4
  replays (`m1_b32`, `decode_per_replay_us`); the end-to-end runs are flat (215.00 ms, 3 runs within 0.002 ms).
  The method sums kernel durations and excludes device op-to-op gaps.
- B1: 16 to 36% of the DRAM roofline. Per-op latency bound: 93 ops per 2-layer replay on p150x4.
- B32: the 40 layers are 98 to 99% of device time; terminal and sampler are 1 to 2%.
- B32 per layer: a GDN-type layer costs 1.28x (p150x4), 1.54x (p150x2) and 3.36x (p150) a full-type layer.
- Sampler trace: 5.5% (p150x4), 7.3% (p150x2), 8.2% (p150) of B1 token-out. Not dominant.

### Decoder-layer lower bound (batch 1)

| Profile | Stage 05 layer sum (traced) | Layer sum + terminal + sampler | Token-out |
|---|---:|---:|---:|
| p150x4 | 9.39 ms | 10.35 ms | 9.11 ms |
| p150x2 | 10.92 ms | 12.57 ms | 11.30 ms |
| p150 | 14.42 ms | 17.34 ms | 15.80 ms |

- Stage 05 sums: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/README.md`
  (single-layer traced harness, per-layer host sync included).
- With stage 05 kernel sums instead (197.9 and 248.2 us per layer), the p150x4 bound is 9.38 ms. Token-out is below
  it on every profile. No gap above 10 to 15% exists to close.

## Profiles and tt-perf-report tables

Reduced traced model: layers (0,3) (one GDN-type, one full-type), embedding, final norm, LM head, sampler, real
weights. Signposted windows: one prefill (ISL 128) and 4 warmed decode replays with their sampler replays.
Collected with `python -m tracy -r` (named ops).

| Case | Tables | Prefill ops / kernel time | Decode window ops / kernel time | Top decode ops (window, us) |
|---|---|---|---|---|
| p150x4 B1 | `evidence3/m4/profile_b1/` | 229 / 5068 us | 484 / 5538 us | Matmul 1864, TopkLargeIndices 1140, BinaryNg 347 |
| p150x4 B32 | `evidence5/m4/profile_b32/` | 229 / 5057 us | 805 / 13048 us | GenericOp 4224, SparseMatmul 2531, Matmul 1887 |
| p150x2 B1 | `evidence3/m2/profile_b1/` | 229 / 7633 us | 484 / 8548 us | Matmul 3337, TopkLargeIndices 2263, GenericOp 410 |
| p150x2 B32 | `evidence5/m2/profile_b32/` | 229 / 7631 us | 905 / 22256 us | GenericOp 9129, SparseMatmul 3948, Matmul 3370 |
| p150 B1 | `evidence3/m1/profile_b1/` | 216 / 12528 us | 440 / 14262 us | Matmul 6671, TopkLargeIndices 4281, GenericOp 622 |
| p150 B32 | `evidence5/m1/profile_b32/` | 216 / 12538 us | 2455 / 43195 us | MoeFusedSwiGlu 10472, GenericOp 7607, Matmul 7555 |

- Each directory: `{prefill,decode}_perf_report.{txt,csv}`, console log, `summary.json`, raw CSV `sha256`.
- `tt-perf-report` drops trace-replay rows with invalid device durations from its per-op table. Per-replay
  named totals come from `evidence5/named_breakdown.json`
  (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/named_breakdown5.py`).
- `GenericOp` is the batched GDN kernels (conv, recurrence, gate) and the slot expert gemv.
- `TopkLargeIndices` is the sampler local top-k over the vocabulary shard.
- Prefill windows are unchanged from `evidence3/` (prefill programs unchanged).

## Batched decode (B > 1): operation topology

One B32 decode replay, layers (0,3), device 0, by op type (`evidence4/named_breakdown.json` before,
`evidence5/named_breakdown.json` after).

| Op type | Role at B > 1 | p150x4 before / after (us) | p150x2 before / after (us) | Action |
|---|---|---:|---:|---|
| GenericOp | batched GDN conv, recurrence, gate; slot expert gemv | 1063 / 1056 | 2285 / 2282 | kept |
| SparseMatmul | indexed expert down per slot | 628 / 633 | 1051 / 987 | kept |
| Matmul | attention in/out, router linear (B1 program config), shared down, LM head | 507 / 472 | 873 / 842 | decode programs |
| TopK + Softmax | router prefill top-k (1 core) | 77 / 0 | 77 / 0 | replaced by `GeneralizedMoeGate` (5 us) |
| BinaryNg | score multiply over 32 rows per slot, residual adds | 229 / 66 | 226 / 71 | scale before down |
| UntilizeWithUnpadding | index sticks per chunk | 122 / 0 | 221 / 0 | sticks once per layer |
| ReshapeView | `[B,1,H]` to `[1,B,H]` and back per layer | 175 / 66 | 180 / 68 | hidden packed across layers |
| TilizeWithValPadding | index sticks, embedding entry | 86 / 66 | 116 / 66 | reduced |
| Typecast | index sticks | 39 / 11 | 42 / 11 | reduced |
| FastReduceNC | sum over expert slots | 73 / 74 | 72 / 72 | kept |
| AllReduceAsync | attention out and MoE out, 2 per layer | 32 / 54 | 29 / 68 | kept; see note |
| ShardedToInterleaved | all-reduce output and slot path | 6 / 18 | 6 / 18 | see note |
| MinimalMatmul | shared expert fused SwiGLU (decode config) | 24 / 24 | 27 / 27 | kept |
| LayerNorm, SdpaDecode, Rotary, PagedFusedUpdateCache | norms, full attention | 136 / 133 | 131 / 131 | kept |
| Remaining layout ops (Slice, Transpose, InterleavedToSharded, Concat) | attention head layout, slot chunks | 72 / 70 | 90 / 80 | kept |
| Replay total | | 3282 / 2763 (187 / 133 ops) | 5443 / 4747 (257 / 153 ops) | |

- p150 (B32 uses EP dispatch, `moe_slot_rows` 16): 9748 -> 9526 us, 472 -> 468 ops. `MinimalMatmul` 301 -> 49 us
  and `Matmul` 1762 -> 1889 us (`attention_in` moved to the decode program: net about -125 us per replay);
  `ReshapeView` 145 -> 44 us (packed hidden). The dispatch MoE keeps its prefill programs (fix5).
- AllReduceAsync note: same input shape and memory config before and after (`[1,1,32,2048]` BF16 tile, L1 width
  sharded). Mean per call on p150x4: 7.9 -> 13.4 us. Most calls take about 8 us; the last call of each replay
  takes 20 to 55 us. CCL kernel time includes the wait for peer devices. Unverified
  cause: less skew absorbed by the ops before the all-reduce. Not isolated. The replay is 519 us shorter in total.
- ShardedToInterleaved note: 4 new height-sharded calls per replay at 2.9 us each.
- Layout and glue ops on p150x4 (untilize, tilize, reshape, typecast, slice, transpose, sharded/interleaved
  conversions, concat, copy): 502 -> 234 us per 2-layer replay (8% of 2763 us).

## Accuracy (final source; B1 and prefill `evidence3/`, batched `evidence5/`)

HF gates (B1 path, `evidence3/`; readiness runners, tracker on; equal to stage 06 on every profile):

| Profile | Prefill AIME24 top-1 / top-5 / top-100 | Teacher forcing AIME24 | Teacher forcing book 8193 |
|---|---|---|---|
| p150x4 | 99 / 100 / 100% | 99 / 100 / 100% | 92.2 / 100 / 100% |
| p150x2 | 97 / 100 / 100% | 96 / 100 / 100% | 90.6 / 98.4 / 100% |
| p150 | 99 / 100 / 100% | 98 / 100 / 100% | 89.1 / 98.4 / 100% |

Batched decode against HF (AIME24 chat in row 0, 100 teacher-forced positions: 1 prefill plus 99 decode; B1 from
`evidence3/calibration_compare.json`, B16 and B32 from `evidence5/calibration_compare.json`
(`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/calibration3/compare5.py`)).
B16 and B32 fill rows 1 to B-1 with distinct book slices, so routing differs per row.

| Profile | B1 KL mean / max | B1 top-1 | B16 KL mean / max | B16 top-1 | B32 KL mean / max | B32 top-1 |
|---|---|---:|---|---:|---|---:|
| p150x4 | 0.0070 / 0.155 | 99 | 0.0083 / 0.181 | 97 | 0.0083 / 0.181 | 97 |
| p150x2 | 0.0074 / 0.186 | 96 | 0.0050 / 0.049 | 99 | 0.0050 / 0.049 | 99 |
| p150 | 0.0055 / 0.051 | 98 | 0.0070 / 0.219 | 99 | 0.0071 / 0.144 | 99 |

- HF top-1 is in TT top-5 at all 100 positions in every case.
- B16 and B32 row 0 are equal on p150x4 and p150x2: both use the slot path, which is row-independent.
  p150 B32 uses EP dispatch, so it differs.
- KL history (B16 mean, p150x4 / p150x2 / p150): `evidence4/` 0.0068 / 0.0067 / 0.0056; fix2 0.0058 / 0.0062 /
  0.0055; final 0.0083 / 0.0050 / 0.0070. The changes go both ways across profiles.
- TT-to-TT KL between consecutive sources (same 100 positions): `evidence4/` to fix2 0.0033 / 0.0026 / 0.0025;
  fix2 to fix3 0.0039 / 0.0021 / 0.0026. Both changes are the same size, and smaller than TT-to-HF KL.
- Classification: BF16 rounding-order change, not an accuracy loss. Evidence: the size above; top-1 97 to 99 at
  B1 level; HF top-1 in TT top-5 at every position; layer delta PCC equal or better (below).
- p150x4 B16 top-1 97 vs B1 99: 2 of 100 positions. Both are in TT top-5.

Batched against B1 (`run_full_evidence`, rows of length 5 to 12000, B1 tokens teacher-forced, 8 steps, `audit=True`):

| Profile | B4 KL to B1 mean / max | B32 KL to B1 mean / max | Argmax agree (B32) | Worst B1-token rank |
|---|---|---|---:|---:|
| p150x4 | 0.0024 / 0.041 | 0.0012 / 0.041 | 252 / 256 | 1 |
| p150x2 | 0.0020 / 0.019 | 0.0010 / 0.019 | 248 / 256 | 1 |
| p150 | 0.0019 / 0.013 | 0.0010 / 0.020 | 252 / 256 | 1 |

- Every disagreement is the 5-token row, where the B1 token is rank 1 (second) at one or two steps. That row
  holds every per-row maximum above 0.003.
- `evidence4/` maxima: 0.020 / 0.022 / 0.010 (B32). p150x4 rose to 0.041 (the 5-token row); p150x2 fell.
- B8 and B16 give the same result (`evidence5/m*/full_evidence.json`, `batch_8`, `batch_16`).
- Per-check comparison of `evidence4/` and `evidence5/`: `evidence5/check_compare_ev4_ev5.json`
  (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/compare_ev45.py`).

Other checks (all profiles pass):

| Check | Artifact | Result |
|---|---|---|
| Layer delta PCC, batched vs per-row, B4/8/16/32 | `evidence5/m*/layer_check_{gdn,full}.json` | min 0.99992 (GDN), 0.99957 (full; p150); p150x4 and p150x2 full 0.9996 -> 0.99993 |
| Inactive rows, buckets 1/4/8/16/32 | `evidence5/m*/inactive_rows.json` | active rows equal the all-active bucket; hidden live slot rejected; B1 after inactive identical |
| Sampling audit, greedy and sampled B1/4/8/16/32 | `evidence5/m*/sampling_audit.json` | passed; device token copies agree in all 12 runs; TV 0.017 to 0.024 |
| Host seed advance, B32, 128 steps | `evidence5/m*/seed_host.json` | 0 mismatched device copies |
| Split-sampling trace loop (B1 path) | `evidence3/m*/split_sampling_trace.json` | all checks true; no host refresh in loop; 6 seed writes in 6 sampled steps |
| Watcher, batched 2-layer B1 to B32 | `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence5_work/watcher_m*/`, `evidence5/m*/batched_l2_watcher.json` | no watcher errors |
| Determinism and non-aligned prompts (5 to 12000) | `evidence5/m*/full_evidence.json` | greedy repeat identical; prefill logits bitwise |
| Runtime fallback audit (B1 to B32 capture) | `evidence5/m*/full_evidence.json` | `pad_fallbacks` 0 |
| Degenerate output (B1 autoregressive) | `evidence3/degenerate_check.json` | no findings |

- Layer checks compare by PCC and maximum absolute difference. They do not show bitwise equality.
- Correctness runs use `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`.

Qualitative suite (chat template, greedy, 128 tokens; `evidence5/qualitative_compare.json` from
`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/qual_compare5.py`):
- B1 on every profile (`evidence3/qualitative/`, copied to `evidence5/qualitative/`): coherent; token-identical
  to stage 06.
- B8 and B32 (`evidence5/qualitative/tt_m*_b{8,32}_outputs.json`): the six shared prompts repeat across rows.
  Every row is coherent (read; no 8-token sequence repeats in any row).
- Rows with the same prompt are identical on every profile and both B.
- B8 rows equal B32 rows on p150x4 and p150x2. On p150 they differ: B32 uses EP dispatch.
- HF control: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/qualitative/hf_outputs.json`. Common prefix with HF,
  B32 vs B1, per prompt p0 to p5:

| Profile | B1 | B32 |
|---|---|---|
| p150x4 | 66, 48, 119, 19, 44, 29 | 66, 90, 119, 19, 44, 29 |
| p150x2 | 74, 90, 69, 0, 44, 12 | 66, 50, 62, 0, 44, 29 |
| p150 | 39, 90, 62, 19, 44, 12 | 39, 90, 62, 19, 96, 29 |

- p150x2 p3 differs from HF at token 0 in B1 too ("Here's a thinking process" vs "Thinking Process"). Same in
  stage 06.

Long context (`evidence5/m*/full_evidence.json`): 262100-token prefill plus 16 tokens: 62.1 s, 90.8 s, 139.2 s.
The 262143-token boundary returns 2 tokens. Native 262144 context kept.

## Sampler

| Choice | Measured | Decision |
|---|---|---|
| Greedy: split top-k (local top-32, gather 32 candidates) vs force-argmax (gather full vocabulary) | p150x4 B1 9.18 vs 11.15 ms; B32 63.86 vs 65.99 ms; tokens equal | split top-k |
| Seeds: in-trace Weyl add | chi-square p 0.0005 | rejected (biased) |
| Seeds: SKIP sentinel | device token copies diverge at B32 on p150x4 (128 of 128 steps) | rejected on meshes |
| Seeds: host write every sampled step | copies equal; sampled = greedy within 0.01 ms on the slot path | default |

- Force-argmax was measured only on p150x4. It gathers the full vocabulary instead of 32 candidates per device.
- SKIP failure (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/seed_copies/`): with MoE EP dispatch between
  samples, copies on devices 1 to 3 differ from device 0. A negative control
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence2/m4/sampling_audit_skip_control.json`) proves the audit detects it.

## Batched MoE decode: candidates

Raw: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/union/`. Full-model p150x4, greedy, ms per step.

| Candidate | B4 | B8 | B16 | B32 | Decision |
|---|---:|---:|---:|---:|---|
| Row loop (B <= 8) / EP dispatch with tile combine | 21.99 | 36.21 | 54.61 | 63.89 | replaced |
| Union mask, dense-scan `sparse_matmul` over 256 groups | 47.20 | 52.64 | 60.66 | 72.09 | rejected: about 10 us per active group |
| Slot path (before fix2 and fix3) | 17.98 | 24.18 | 35.81 | 60.23 | selected |
| Slot path (final) | 13.82 | 18.89 | 28.54 | 47.97 | default for 2 to 32 rows |

- Micro (one device, real shapes, traced): indexed `sparse_matmul` gate_up about 4 us per slot; custom
  `expert_swiglu_decode` gemv 1.4 us per slot (210 GB/s).
- p150x2 B32: slot 105.05 = dispatch 105.05 ms (before fix2). p150 B32: slot 235.97 vs dispatch 222.11 ms, so p150
  uses `moe_slot_rows` 16.
- The p150 B > 16 dispatch path runs the prefill MoE (BFP4 gate_up, LoFi, no FP32 accumulation;
  `tt/multichip_decoder.py`). Its router and projections inside `moe.forward(mode="prefill")` use prefill programs
  (`PREFILL_PROGRAMS`, fix5).

## LM head

| Candidate | p150x4 B1 | p150x4 B32 | p150 B1 | Decision |
|---|---:|---:|---:|---|
| Interleaved (default) | 9.117 ms | 60.113 ms | 15.809 ms | kept |
| DRAM-sharded: padded vocabulary 248832, 62208-column chunks, 3 readers, 64 cores, `in0_block_w` 1 | 9.222 ms | 60.245 ms | 15.922 ms | rejected: slower end to end |

- Standalone traced matmul: 336.9 vs 390 us (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/lmhead/trace.json`).
- In the model, reshard in, unshard out and the sampler tail mask cost more than the 53 us gain.
- B1 tokens identical. `lm_head_dram` stays available, default off. Not measured in model on p150x2.

## Token readback

- The host needs each token, so one readback per step remains.
- `decode_steps`: `tt_tokens.cpu(blocking=False)` plus `record_event`; replay i+1 is enqueued before
  `event_synchronize` on step i. The read overlaps the next replay.
- Effect (p150x4): B1 9.18 -> 9.11 ms, B4 17.98 -> 17.91 ms, B32 60.23 -> 60.12 ms.
- p150x2: B1 11.36 -> 11.30 ms. p150: B1 15.88 -> 15.80 ms (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/union/slot_perf_m*.json`).
- Early stop semantics (`tt/generator.py` `decode_steps`): when `stop` fires on token i, replay i+1 has already run
  with token i as input. KV cache, GDN state and device position are one step past token i. The read of token i+1
  is dropped. The runtime pins the host buffer of a non-blocking read, so no use-after-free. `generate()` resets
  the slot at its next call. A caller that continues a slot after a stop must reset it first. No run exercised
  this branch (all outputs reach 128 tokens).

## Operation topology (full-model path)

| Segment | Ops per step | Layout | Collective | Decision |
|---|---|---|---|---|
| Embedding | 1 gather; B > 1: tilize and one reshape to `[1,B,2048]` | replicated table, BF16 | none | kept |
| 40 layers | stage 05 contract; B > 1 packed `[1,B,2048]` | replicated residual | 2 all-reduces per layer (stage 05 family audit) | kept |
| MoE, 2 to 32 rows | router linear (B1 program config), `GeneralizedMoeGate`, gemv per slot, scale, indexed `sparse_matmul`, reduce | TP decode expert copy | none inside MoE | slot path |
| MoE, p150 17 to 32 rows | padded prefill MoE with EP dispatch | stacked expert copy | none | dispatch path |
| Final norm | 1 fused RMS norm | replicated | none | kept |
| LM head | 1 matmul `[1,B,2048] x [2048, V/tp]` | vocabulary sharded over devices | none | interleaved kept |
| Sampler | local top-k, gather 32 x tp candidates, sample | vocabulary sharded | all-gather of candidates | split top-k |
| Token feedback | `tt_out_tok` on device | replicated | none | kept |

B > 1 per-op detail: "Batched decode (B > 1): operation topology" above.

## Dispositions of items assigned to stage 07

| Item | Source | Disposition | Evidence |
|---|---|---|---|
| Routed experts stored twice on p150x2 and p150x4 (EP prefill copy, TP decode copy) | stage 06 README; stage 05 README | measured and kept | `dispatch_weights=stacked` frees about 5.9 GB (p150x4) and 11.7 GB (p150x2), estimate from buffer sizes; prefill +3.8 to 6.7% slower (p150x4), +3.8 to 5.2% (p150x2) (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/work_log.md`). Capacity contract met with both copies. `stacked` stays selectable. |
| DRAM remainder not itemized | stage 06 README | measured as a total; not itemized by buffer | unnamed DRAM after build: 272 / 312 / 183 MiB (1.4%, 1.0%, 0.6% of allocated). The census walks Python-reachable tensors; it has no allocator-side buffer list. Open item, carried to stage 08 (datatype sweep rebuilds the weight set). |
| Construction peaks not measured | stage 06 README | unchanged | construction path unchanged since stage 06; stage 05 per-layer setup peaks apply (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/memory_handoff_recheck.json`). |
| Long prefill (262100 tokens in 62 to 139 s) | stage 07 work log | measured, not changed | prefill programs unchanged in stage 07. Deferred to stage 10 (serving TTFT) and stage 13 (131072-token sweep). |
| Memory handoff recheck | `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md` | done | `memory_handoff_recheck.json`: final source hashes, census per profile, DRAM deltas vs stage 06 (layer-private +1.0 / +0.9 / +0.4 MiB), L1 deltas vs stage 06 (+3604480 / +3604480 / +2703360 bytes: 32 / 32 / 24 KiB per bank, router cached slices for batched widths), stage 07 allocations, duplicate disposition. |

## Checklist ($optimize and $multichip, full-model path)

| Item | Evidence |
|---|---|
| Decode traced, no host boundary in the loop | `split_sampling_trace.json` checks; counters in `full_evidence.json` |
| Device-bound, accounting reconciled | accounting table (host gap at most 0.34 ms) |
| Profiles and `tt-perf-report` tables | profile section |
| Operation-topology audit at B > 1 | B > 1 topology table |
| Decode programs at decode rows | fix2; `TopK` and prefill `MinimalMatmul` gone from the slot path |
| Sampler not dominant; split greedy kept | sampler table |
| LM head DRAM-sharded tried | LM head table |
| MoE active-expert path at decode | slot path; union-mask rejected with measurements |
| Precision policy unchanged from stage 05 | no dtype change in this stage |
| Context contract kept | context section |
| Non-aligned prompt lengths | `nonaligned_single` 5 to 12000 |
| Watcher clean | watcher row |

## Context contract

`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`,
section `optimized_full_model_validation`, from `evidence5/`
(`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/update_contract7_ev5.py`).

- Native 262144 context kept. KV pool, block size and 32 request slots unchanged.
- Trace captures (buckets 1, 4, 8, 16, 32 plus sampler traces): 72.5, 97.3 and 229.4 MB of DRAM per device.
- Free DRAM after all captures: 13.00, 2.89 and 1.39 GB (p150x4, p150x2, p150).
- Persistent L1 after build: 9405440, 7603200 and 4505600 bytes over 110 banks (stage 06: 5800960, 3998720,
  1802240). The growth is the router cached slices for widths 4 to 32 (widths 4 to 16 on p150), shared by all
  layers and created in prepare before any capture. Attribution is from an exact byte match, not an allocator dump.
- Full-context sequences in the KV pool: 4.0, 2.0 and 1.0.

## Limitations

- p150 GDN runs per row in batched decode: the batched GDN kernel needs num_v_heads x 4 = 128 cores. p150 B32
  GDN-type layer: 6612 us against 1970 us for a full-type layer (30 GDN layers about 198 ms of 215 ms). Two
  16-head calls (64 cores each) were not tried.
- Batched decode is not bitwise equal to B1: different programs (KL to B1 at most 0.041).
- Slot MoE reads each selected expert once per row slot; no device-side de-duplication of the expert union.
- `generic_op` program-cache hash covers the core count but not per-core runtime-argument length. The batched GDN
  kernels put the user count and span in compile arguments. Upstream issue candidate; not filed.
- The op that advances the core PRNG between SKIP samples is not identified (SKIP is rejected on meshes).
- Final norm is a 1-core FP32 `LayerNorm` of 24.6 us plus two typecasts per step (about 0.3% of B1).

## Provenance

- Stage commit: `e9e7ebd0058d9c0e3e15c1eab4524f961eb7dfa9` (branch `feat/qwen3-5-35b-a3b`; pushed to fork `housTT/tt-metal` branch
  `hous/qwen3.5-35b-a3b`). Review 3 (`stage_review3.md`): rereview clean-pass.
- Batched claims use `evidence5/`; B1 and prefill claims use `evidence3/`.
- `evidence5/` p150x4 and p150x2: chain `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain5.sh`
  (Oct 24 02:11 to 07:49), 48 of 48 steps rc 0, full `source_manifest_end.txt` equals `source_manifest_start.txt`
  (`SOURCE_STABLE`). Snapshot `evidence5/source_snapshot.tgz`; diff against `25bbbe3c`:
  `evidence5/source_diff_vs_head.patch`.
- `evidence5/` p150: chain `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain5b.sh` (Oct 24 08:17 to
  09:46), all 16 p150 steps rc 0, `source_manifest_end_m1b.txt` equals `source_manifest_start_m1b.txt`.
  Snapshot `evidence5/source_snapshot_m1b.tgz`; diff `evidence5/source_diff_vs_head_m1b.patch`.
- Chain 5 to chain 5b source delta: only fix5 in `tt/optimized_decoder.py` (`PREFILL_PROGRAMS` around the padded
  `moe.forward(mode="prefill")` call). That branch runs only when rows exceed `moe_slot_rows`. p150x4 and p150x2
  use `moe_slot_rows` 32 and the largest bucket is 32, so they never reach it. Their chain 5 evidence holds.
- Chain 5 p150 evidence (pre-fix5) is kept in `evidence5/m1_pre_fix5/` and `evidence5/qualitative/pre_fix5/`.
- The chain scripts compare full manifests (the `head -n -1` defect named in review 2 is fixed).
- `evidence4/` and `evidence3/` batched results are history after the projection fix and fix2 to fix5.
- `evidence1/` and `evidence2/` ran an earlier source. They are kept as history and are not used for any final
  claim. The chain 1 `tt/generator.py` bytes were not archived.

## Incidents

| Time | Event | Action |
|---|---|---|
| Oct 15 04:48 | Two device jobs overlapped; the new one stalled at fabric init | stopped by PID, `tt-smi -r`, results discarded |
| Oct 15 09:20 | About 210 GB host memory used outside any process; `calibration_b32_m4` killed | stale locks saved and removed, reset, host reboot |
| Oct 15 16:15 | Driver re-init during `calibration_b16/b32_m2` | rerun in chain 3c (pass) |
| Oct 15 17:05 to Oct 23 | Another user's vLLM job held device 0; its process wedged a driver mutex from Oct 19 | chain stopped; host reboot Oct 23 10:10; chain 3c rerun |

## Anomaly ledger

- Observed anomaly: SKIP seed mode lets device token copies diverge on meshes.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/seed_copies/m4_skip_b1.json`.
  Affected path: sampled decode with EP dispatch between draws.
  Control or comparison: host seeds 0 mismatches (`evidence5/m*/seed_host.json`).
  Likely subsystem: core PRNG state shared with dispatch kernels.
  Investigation performed: decode-path matrix; audit check plus negative control.
  Resolution: fixed (host default; SKIP raises for TP > 1).
- Observed anomaly: B32 candidate 60.93 ms not reproduced (63.89 ms) in the first review.
  Evidence: `stage_review1.md`.
  Affected path: B16 and B32 decode default.
  Control or comparison: slot path replaced the dispatch default; final 47.97 ms reproduces in 3 runs within 0.003 ms.
  Likely subsystem: superseded MoE dispatch path.
  Investigation performed: path replaced; final default measured in `evidence5/`.
  Resolution: fixed.
- Observed anomaly: first B32 profile used one prompt for all rows.
  Evidence: `evidence1/m4/profile_b32_run.json`.
  Affected path: profile conclusions.
  Control or comparison: `run_model_profile --distinct 1` (default); `evidence5/m*/profile_b32_*_run.json` rows differ.
  Likely subsystem: profile harness input.
  Investigation performed: harness fixed, profiles rerun.
  Resolution: fixed.
- Observed anomaly: p150 B32 profiled replays rise 8% over 4 replays.
  Evidence: `evidence5/perf_accounting.json` (`m1_b32`, `decode_per_replay_us`, 9017 to 9776 us).
  Affected path: p150 B32 device-time estimate only.
  Control or comparison: end-to-end runs flat at 215.00 ms (3 runs within 0.002 ms).
  Likely subsystem: profiler overhead.
  Investigation performed: compared to end-to-end runs.
  Resolution: controlled.
- Observed anomaly: `generic_op` reused a cached program with shorter per-core runtime arguments (NaN at 32 users after 4).
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/batched/layer_check1/m4.json`.
  Affected path: batched GDN kernels.
  Control or comparison: `evidence5/m*/layer_check_gdn.json` at 4/8/16/32 users in one process.
  Likely subsystem: `generic_op` program hash.
  Investigation performed: source reading; compile-argument workaround.
  Resolution: fixed by workaround; upstream issue not filed.
- Observed anomaly: batched decode selected prefill programs (projections; then router linear without program
  config, router prefill top-k on 1 core, p150 `attention_in` prefill `minimal_matmul`).
  Evidence: `evidence3/named_breakdown.json`, `stage_review2.md` required work item 1.
  Affected path: every bucket B4 to B32.
  Control or comparison: B1 rows with the same padded shape; `evidence5/named_breakdown.json` has no `TopK` and no
  prefill `MinimalMatmul` on the slot path.
  Likely subsystem: program selection by `length > 1` inside the decode batch context.
  Investigation performed: code fix (projection fix, fix2); all batched checks rerun in `evidence5/`.
  Resolution: fixed.
- Observed anomaly: after fix2, p150 B32 rows with the same prompt diverged (chain 5, pre-fix5); `evidence4/` had
  them identical.
  Evidence: `evidence5/qualitative/pre_fix5/`, `evidence5/m1_pre_fix5/`.
  Affected path: p150 B17 to B32, padded `moe.forward(mode="prefill")` (EP dispatch).
  Control or comparison: p150x4 and p150x2 rows identical (slot path); `evidence4/` p150 rows identical.
  Likely subsystem: fix2 made the router top-k and `_Projection` inside the padded prefill MoE pick decode programs
  at decode row counts, mixing prefill dispatch with decode programs.
  Investigation performed: code reading of the branch; fix5 restores prefill programs in that branch only;
  trial `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/fix5/`; p150 chain 5b rerun.
  Resolution: fixed by restoring the `evidence4/` programs in that branch. Rows identical in
  `evidence5/qualitative_compare.json` (`m1_b32`); B32 KL to HF 0.0071. The exact op that made identical rows
  differ is not identified.
- Observed anomaly: persistent L1 grew by 32 KiB per bank (meshes) and 24 KiB per bank (p150) vs stage 06.
  Evidence: `memory_after_build.l1` in `evidence5/m*/full_evidence.json` vs stage 06 and `evidence4/`.
  Affected path: L1 headroom for all profiles.
  Control or comparison: byte arithmetic for router cached slices (16 or 12 pages x 2048 bytes x 110 banks) matches
  exactly; fix4 (widths 4, 32) shows 8 pages; fix5 shows no further change.
  Likely subsystem: router cached slices for batched widths (fix2 routes batched rows through that branch).
  Investigation performed: per-run L1 comparison; code reading of `_OptimizedGateModule.route` and the sharing in
  `tt/model.py`.
  Resolution: controlled; recorded in `memory_handoff_recheck.json` (`stage7_allocations`, per-profile L1 deltas).
- Observed anomaly: batched KL to HF and to B1 shifts with each program change (for example p150x4 B16 0.0058 ->
  0.0083; B32 KL to B1 maximum 0.020 -> 0.041), in both directions across profiles.
  Evidence: `evidence5/calibration_compare.json`, `evidence5/check_compare_ev4_ev5.json`, fix2 and fix3 trials.
  Affected path: batched decode numerics.
  Control or comparison: TT-to-TT KL between consecutive sources 0.002 to 0.004, same size for every change;
  p150x2 KL fell (0.0062 -> 0.0050); layer delta PCC equal or better; HF top-1 in TT top-5 at every position.
  Likely subsystem: BF16 accumulation order (scale before down, decode programs).
  Investigation performed: per-source KL to HF and TT-to-TT KL on the same 100 positions.
  Resolution: controlled. The exact rounding sites are not identified.
- Observed anomaly: the projection fix left single-layer PCC checks unchanged but changed full-model batched logits
  on the meshes.
  Evidence: `evidence4/m*/layer_check_*.json`; review 2 `torch.equal` on saved logits: B16 logits first differ at
  decode position 3 (p150x4) and 2 (p150x2).
  Affected path: batched decode numerics on meshes.
  Control or comparison: PCC over 4 x 2048 values cannot resolve 1-ULP (unit in the last place) BF16 changes, so
  equal PCC is not bitwise evidence. HF KL stayed at the B1 level.
  Likely subsystem: different accumulation in the mesh decode program.
  Investigation performed: call-site instrumentation, layer probes, per-position logit comparison.
  Resolution: controlled; superseded by fix2 to fix5 numerics above.
- Observed anomaly: AllReduceAsync time per call rose 7.9 -> 13.4 us (p150x4) with the same shape.
  Evidence: `evidence4/named_breakdown.json`, `evidence5/named_breakdown.json` raw CSV rows.
  Affected path: B32 decode all-reduce.
  Control or comparison: replay total 3282 -> 2763 us; end-to-end 58.85 -> 47.97 ms.
  Likely subsystem: CCL kernel time includes peer-device wait (unverified).
  Investigation performed: shape and memory config compared per call.
  Resolution: controlled for end-to-end time; cause not isolated.

## Evidence

| Artifact | Contents |
|---|---|
| `evidence5/m{4,2,1}/` | final batched evidence: watcher run, layer checks, seed test, inactive rows, batch perf (greedy, sampled), qualitative logs, sampling audit, full evidence, calibration B16 and B32, B32 profiles |
| `evidence5/perf_accounting.json`, `evidence5/calibration_compare.json`, `evidence5/named_breakdown.json` | accounting (B1 from `evidence3/`), KL to HF for B16 and B32, B32 named per-op totals |
| `evidence5/check_compare_ev4_ev5.json`, `evidence5/qualitative_compare.json` | per-check comparison, batched qualitative comparison with HF control |
| `evidence5/m1_pre_fix5/`, `evidence5/qualitative/pre_fix5/` | p150 chain 5 evidence before fix5 (history) |
| `evidence4/` | batched evidence after the projection fix (history) |
| `evidence3/m{4,2,1}/` | B1 and prefill evidence: HF gates, autoregressive, micro, split trace, calibration B1, B1 profiles |
| `evidence3/qualitative/` | B1 outputs (final), rendered prompts |
| `evidence3/degenerate_check.json` | degenerate-output check |
| `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/perf_summary.json` | B1 and B32 summary per profile |
| `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/memory_handoff_recheck.json` | stage 07 memory handoff recheck |
| `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/batched_decode_audit.md` | batched-path audit |
| `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/stage_review1.md`, `stage_review2.md` | first and second stage reviews |

Commands: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain5.sh`, `evidence_chain5b.sh`,
`perf_accounting5.py`, `named_breakdown5.py`, `process_profiles.sh`, `compare_ev45.py`, `qual_compare5.py`,
`write_perf_summary5.py`, `update_contract7_ev5.py`, `memory_handoff_recheck7.py`, `calibration3/compare5.py` (all in
`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/`).

# Qwen3.5-35B-A3B optimized full model: stage 07

Abbreviations: TTFT (time to first token), t/s/u (tokens per second per user), t/s (tokens per second), ISL/OSL
(input/output sequence length, in tokens), B (decode batch: active rows in one decode step), HF (Hugging Face CPU
reference), TT (Tenstorrent TTNN implementation), KV (key/value cache), GDN (Gated DeltaNet linear-attention
layer), MoE (mixture of experts), EP (expert parallel), LM head (final vocabulary projection), BF16 (16-bit brain
float), BFP8 (8-bit block float), DRAM (device memory), L1 (core SRAM), PCC (Pearson correlation coefficient), KL
(Kullback-Leibler divergence), TV (total variation distance), PRNG (pseudo-random number generator), CPU (host
processor), CCL (collective communication library).

Base: stage 06 commit `dfe0c6bd` (checkpoint `fb8ce438`). Work log: `work_log.md`. Raw runs:
`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/`.

## Headline

### Batch 1 (tracker off, ISL 128, greedy, 127 steps, median of 3; `run_batch_perf`)

| Profile | TTFT before | TTFT after | Token-out before | Token-out after |
|---|---:|---:|---:|---:|
| p150x4 | 97.8 ms | 97.0 ms | 9.18 ms (108.9 t/s/u) | 9.19 ms (108.8 t/s/u) |
| p150x2 | 114.0 ms | 113.8 ms | 11.35 ms (88.1 t/s/u) | 11.36 ms (88.0 t/s/u) |
| p150 | 183.4 ms | 183.7 ms | 15.87 ms (63.0 t/s/u) | 15.87 ms (63.0 t/s/u) |

Before: `../full_model/evidence5/m*/batch_perf.json`. After: `evidence2/m*/batch_perf_sampled0.json`.
Tracker-on headline (`evidence1/m*/full_evidence.json`, `perf_128_128`): TTFT 99.2 / 113.8 / 183.5 ms; token-out
9.20 / 11.40 / 15.85 ms (108.7 / 87.7 / 63.1 t/s/u); logits-only decode 8.59 / 10.48 / 14.51 ms (equal to stage 06).

Batch 1 is unchanged on purpose. It was already below the decoder-layer lower bound (next section). This stage
changes the batch path.

### Batched decode (same harness; greedy rows, ms per step, aggregate t/s in brackets)

| Profile | B4 before | B4 after | B32 before | B32 after |
|---|---:|---:|---:|---:|
| p150x4 | 35.87 (112) | 21.99 (182) | 282.31 (113) | 63.89 (501) |
| p150x2 | 43.95 (91) | 28.93 (138) | 343.90 (93) | 105.05 (305) |
| p150 | 59.64 (67) | 45.19 (89) | 466.10 (69) | 222.11 (144) |

Full matrix, B1 to B32, greedy and sampled (`evidence2/m*/batch_perf_sampled{0,1}.json`):

| Profile | Mode | B1 | B4 | B8 | B16 | B32 |
|---|---|---:|---:|---:|---:|---:|
| p150x4 | greedy | 9.19 | 21.99 | 36.21 | 54.61 | 63.89 |
| p150x4 | sampled | 9.18 | 22.00 | 36.22 | 55.51 | 65.22 |
| p150x2 | greedy | 11.36 | 28.93 | 48.42 | 82.40 | 105.05 |
| p150x2 | sampled | 11.40 | 28.94 | 48.41 | 84.57 | 107.58 |
| p150 | greedy | 15.87 | 45.19 | 78.13 | 158.89 | 222.11 |
| p150 | sampled | 15.85 | 45.21 | 78.17 | 160.05 | 226.71 |

Definitions:
- Prompt: 128 book tokens per row (distinct slices), one request slot per row.
- TTFT: wall time from `generate()` to the first token on the host (prefill, LM head, device sample, readback).
- Token-out: B1 is the `generate()` wall time after the first token divided by 127. B4 to B32 time 126 loop
  iterations of traced decode replay, traced device sample and token readback, after one `decode_forward`.
- t/s/u = 1000 / token-out ms. Aggregate t/s = B x t/s/u.
- Sampled rows: T 1.0, top-k 32, host seed write every step. B1 rows are greedy in both files.
- B is the number of active decode rows, not HTTP client concurrency. Serving concurrency is measured in stages
  10 and 13.
- Tracker off (`QWEN35_STRICT_CACHE=1` only). The tracker-on B1 runs (`evidence1`) agree within 2.3% (TTFT) and 0.4% (token-out).

## What changed

| Change | Files | Effect |
|---|---|---|
| Batched decode: one layer call for B rows, `[1,B,2048]` | `tt/functional_decoder.py`, `tt/optimized_decoder.py`, `tt/multichip_decoder.py`, `tt/generator.py` | B32 p150x4 282.31 -> 63.89 ms |
| LM head for B > 1: unshard and reshape to `[1,B,2048]` before the matmul | `tt/model.py` | 32 one-row matmuls (12.0 ms per step) became one |
| Batched GDN decode kernels (conv, recurrence, gate) with per-slot state addresses | `tt/gdn_batch.py`, `tt/kernels/gdn_batch/` | 2-layer B32 6.39 -> 4.27 ms |
| MoE: per-row loop for B <= 8; EP dispatch with tile combine above | `tt/optimized_decoder.py` (`moe_row_loop_max` 8, `dispatch_tiles_upto` 32) | B4 46.85 -> 21.99 ms; B32 66.77 -> 60.93 ms |
| Attention: B-core KV shard configs | `tt/optimized_decoder.py` | needed for B > 1 batched SDPA decode |
| Sampler seed mode: `host` default; `skip` rejected on meshes | `tt/generator.py` | device token copies stay equal |

`build_generator(batched_decode=True)` is the default for every profile. B1 does not use the batched path.

## Decoder-layer lower bound (batch 1)

| Profile | Stage 05 layer sum (10 full + 30 linear, traced) | Token-out (after) | Logits-only decode |
|---|---:|---:|---:|
| p150x4 | 9.39 ms | 9.19 ms | 8.59 ms |
| p150x2 | 10.92 ms | 11.36 ms | 10.48 ms |
| p150 | 14.42 ms | 15.87 ms | 14.51 ms |

- Terminal work per step (2-layer profile, p150x4 device 0): LM head 390 us, top-k 286 us.
- p150x4 token-out is 2% below the layer sum alone. One trace for 40 layers removes per-layer host time.
- p150x2: 4% above layer sum; layer sum plus LM head (738 us) and top-k (567 us) is 12.2 ms.
- p150: 10% above layer sum; layer sum plus LM head (1480 us) and top-k (1070 us) is 17.0 ms.
- No profile is more than 10% above the layer sum plus terminal work. No avoidable gap to close.

## Accuracy

HF gates (readiness runners, tracker on, `evidence1/m*/`): equal to stage 06 on every profile (B1 path unchanged).

| Profile | Prefill AIME24 top-1 / top-5 / top-100 | Teacher forcing AIME24 | Teacher forcing book 8193 |
|---|---|---|---|
| p150x4 | 99 / 100 / 100% | 99 / 100 / 100% | 92.2 / 100 / 100% |
| p150x2 | 97 / 100 / 100% | 96 / 100 / 100% | 90.6 / 98.4 / 100% |
| p150 | 99 / 100 / 100% | 98 / 100 / 100% | 89.1 / 98.4 / 100% |

Batched decode against HF (AIME24 chat, 100 teacher-forced positions, every row the same prompt;
`evidence1/calibration_compare.json`):

| Profile | B1 KL mean / max | B1 top-1 agree | B32 batched KL mean / max | B32 top-1 agree | B32 row spread |
|---|---|---:|---|---:|---:|
| p150x4 | 0.0070 / 0.155 | 99 | 0.0050 / 0.047 | 98 | 0.0 |
| p150x2 | 0.0074 / 0.186 | 96 | 0.0059 / 0.112 | 99 | 0.0 |
| p150 | 0.0055 / 0.051 | 98 | 0.0055 / 0.059 | 97 | 0.0 |

- Batched decode is deterministic and row-independent. It is not bitwise equal to B1: MoE dispatch and batched GDN
  run different programs. KL to HF is equal or lower than B1 on every profile.
- `run_full_evidence` batch checks (mixed prompt lengths 5 to 12000, B1 tokens teacher-forced, 8 steps): the B1
  token is in the batched top-5 at every step on every row; worst B1-token rank 1.
- Inactive rows (`run_inactive_rows`): active rows equal the all-active bucket bitwise; a hidden live slot is
  rejected; B1 after inactive buckets is identical.
- Sampling audit (`evidence2/m*/sampling_audit.json`): passed on all profiles, including sampled B32 decode with
  device token copies equal on every device.
- Qualitative suite (`evidence1/qualitative/`): token-identical to stage 06 on all profiles. Autoregressive
  degenerate-output check: none detected (`evidence1/degenerate_check.json`).
- Long context: 262100-token prefill plus 16 tokens: 62.2 s, 91.0 s, 140.4 s (stage 06: 62.2, 91.0, 140.4). The
  262143-token boundary returns 2 tokens. Context contract unchanged at 262144.

## Prefill and TTFT

No prefill change in this stage. Full-model TTFT tracks the stage 05 layer sum:

| Profile | TTFT ISL 128 | Layer sum at 100 tokens | TTFT book ISL 8193 | Layer sum, 2 x 4096 |
|---|---:|---:|---:|---:|
| p150x4 | 97.0 ms | 136 ms | 909 ms | 856 ms |
| p150x2 | 113.8 ms | 163 ms | 1214 ms | 1144 ms |
| p150 | 183.7 ms | 225 ms | 1708 ms | 1628 ms |

- Layer sums use stage 05 warm single-layer prefill (30 linear + 10 full). The 2 x 4096 sum ignores the longer
  attention prefix of the second chunk, so it is a low estimate. TTFT is within 7% of it.
- The ISL 128 TTFT is below the 100-token layer sum: the single-layer harness includes per-layer host sync.
- 4096-token prefill profile (2 layers, `evidence1/m4/profile_p4096/`): MoE dispatch data movement
  (embedding gather 725 us, tilize 650 us, reduce 434 us per layer) and expert SwiGLU (1830 us per layer) lead.
  These are stage 05 layer costs, not full-model overhead.

## Sampler

| Choice | Measured | Decision |
|---|---|---|
| Greedy: split top-k (local top-32, gather 32 candidates) vs force-argmax (gather full vocabulary) | B1 9.18 vs 11.15 ms; B32 63.86 vs 65.99 ms; tokens equal | split top-k |
| Seeds: in-trace Weyl add | chi-square p 0.0005 | rejected (biased) |
| Seeds: SKIP sentinel (host writes only after reset) | 1 core-PRNG stream per device; device token copies diverge at B32 on p150x4 (128 of 128 steps) | rejected on meshes; allowed on p150 only |
| Seeds: host write every sampled step | copies equal; +1.24 ms at B32 p150x4 (2%), +0.02 ms at B4 | default |

SKIP failure (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/seed_copies/`): SKIP keeps each core's PRNG state
between steps. With MoE EP dispatch between samples (B > 8 decode, and every prefill), the copies on devices 1 to
3 differ from device 0 on 10 to 12 of 32 rows from the first step. Per-row MoE or host seeds: 0 mismatches. A
negative control (sampling audit forced to SKIP, `evidence2/m4/sampling_audit_skip_control.json`) fails the new
sampled B32 device-copy check, so the audit now detects this defect. The op that advances the PRNG is not
identified.

## Profiles (`tt-perf-report`, reduced model: layers 0 and 3, real weights, 4 traced decode steps)

Tables: `evidence1/m{4,2,1}/profile_b{1,32}/{prefill,decode}_perf_report.{txt,csv}`, plus
`evidence1/m4/profile_p4096/`. Raw CSV hashes: `ops_perf_results.sha256`. Summary per window: `summary.json`.

| Window, device 0 | Ops | Kernel time | Leading ops |
|---|---:|---:|---|
| p150x4 B1 decode, 4 steps | 484 | 5541 us | matmul 1867 (LM head 4 x 390), top-k 1142, binary 347, generic 297 |
| p150x4 B32 decode, 4 steps | 1095 | 13249 us | matmul 2431, generic (GDN batch) 1539, top-k 1135, MoE SwiGLU 1034, binary 953, tilize 810 |
| p150x2 B32 decode, 4 steps | 1095 | 20062 us | generic 3834, matmul 3828, top-k 2268, MoE SwiGLU 1992 |
| p150 B32 decode, 4 steps | 2475 | 37930 us | generic 7603, matmul 6948, top-k 4275, MoE SwiGLU 4129, conv 2569 |
| p150x4 prefill 4096 | 255 | 24662 us | MoE SwiGLU 3653, embedding gather 3053, binary 2656, reduce-scatter 2072 |

Conclusions and actions:

| Finding | Action |
|---|---|
| B > 1 LM head ran as 32 one-row matmuls (12.0 ms per step) | fixed (`[1,B,2048]` reshape) |
| B32 per-row GDN loop: about 2700 us per 2-layer step | batched GDN kernels: generic ops 1539 us for 4 steps (all generic ops, not only GDN) |
| MoE dispatch costs about 46 ms per full-model step at any B; per-row MoE costs 3.55 ms per row | row loop for B <= 8; dispatch above (crossover near 13 rows) |
| `post_combine_reduce` runs on 1 core at 32 tokens | tile combine (`fast_reduce_nc`) for dispatch lengths <= 32: B32 66.77 -> 60.93 ms |
| LM head DRAM-interleaved, 66 to 69% of DRAM bandwidth | DRAM-sharded candidate measured, rejected (below) |
| Top-k on the local vocabulary: 286 us (p150x4) to 1070 us (p150) per step | kept: needed for top-k sampling; force-argmax is slower |
| p150 B32: GDN still per-row (batched kernel needs 128 cores) | limitation |

LM head DRAM-sharded candidate (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/lmhead/`, one device, BFP8
weight, M 32, K 2048, N 62080, traced): best legal config 3 readers per bank, 64 input cores, `in0_block_w` 1:
336.9 us including reshard and unshard (PCC 0.99998). The model's interleaved LM head is 390 us. Gain about 53 us
per step (0.6% of p150x4 token-out). 1 reader per bank and 2 readers with fewer than 32 input cores exceed L1; 4
readers is not legal. It needs global vocabulary padding plus the sampler tail mask. Not adopted: the gain is
inside run-to-run spread and needs full requalification.

## Operation topology (full-model terminal path)

| Segment | Ops per step | Layout | Collective | Decision |
|---|---|---|---|---|
| Embedding | 1 gather | replicated table, BF16 | none | kept |
| 40 layers | stage 05 contract | replicated residual | per-layer all-reduce (stage 05 family audit) | kept |
| Final norm | 1 fused RMS norm | replicated | none | kept |
| LM head | 1 matmul `[1,B,2048] x [2048, V/tp]` | vocabulary sharded over devices | none | B>1 reshape fixed |
| Sampler | local top-k, gather 32 x tp candidates, sample | vocabulary sharded | all-gather of candidates | split top-k kept |
| Token feedback | `tt_out_tok` on device | replicated | none | kept |

## Context contract

`../context_contract.json` adds `optimized_full_model_validation`. Native 262144 context kept; KV pool and slots
unchanged. Trace captures (buckets 1, 4, 8, 16, 32 plus sampler traces) use 115, 137 and 253 MB of DRAM per device;
free DRAM after all captures: 12.96, 2.85 and 1.37 GB (p150x4, p150x2, p150).

## Rejected or deferred

| Option | Evidence | Reason |
|---|---|---|
| MoE dispatch for every B > 1 | B4 46.85 vs 21.99 ms | row loop faster below about 13 rows |
| `post_combine_reduce` at decode lengths | B32 66.77 vs 60.93 ms | tile combine faster |
| Force-argmax greedy | B1 11.15 vs 9.18 ms | slower |
| In-trace Weyl seeds | chi-square p 0.0005 | biased |
| SKIP seeds on meshes | device copies diverge | incorrect |
| DRAM-sharded LM head | 336.9 vs 390 us | 0.6% gain, needs vocabulary padding |
| Remove the second routed-expert copy (p150x2, p150x4) | TP copy used for B <= 8 decode; EP copy for B >= 16 and prefill | both consumed |

## Limitations

- p150 GDN layers run per-row in batched decode. The batched GDN kernel needs num_v_heads x 4 = 128 cores.
- p150 B16 (158.89 ms) is slower per row than B8 (78.13 ms): EP dispatch on one chip.
- Batched decode logits are not bitwise equal to B1 (KL to HF equal or better).
- `generic_op` program-cache hash covers the core count but not per-core runtime-argument length. The batched GDN
  kernels put the user count and span in compile arguments to avoid a stale program. Upstream issue candidate.
- The op that advances the core PRNG between SKIP samples is not identified.

## Evidence

| Artifact | Contents |
|---|---|
| `evidence1/` | chain 1 (54 steps, rc 0, `SOURCE_STABLE`): watcher, seed, inactive rows, HF gates, autoregressive, qualitative, sampling, full evidence, calibration, profiles |
| `evidence1/batch_perf.json` (per profile) | per-row baseline: the chain omitted `--batched 1` (work log) |
| `evidence2/` | chain 2 (14 steps, rc 0, `SOURCE_STABLE`): SKIP control, sampling audit, host seed test, batched greedy and sampled batch perf |
| `evidence1/calibration_compare.json` | HF KL for B1 and B32 |
| `evidence1/m*/watcher.log.gz` | watcher logs (no errors) |
| `batched_decode_audit.md` | batched-path audit |

Commands: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain1.sh`, `evidence_chain2.sh`,
`process_profiles.sh`, `update_contract7.py`.

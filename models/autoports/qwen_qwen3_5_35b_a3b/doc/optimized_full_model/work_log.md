# Stage 07 work log: optimized full model

## Start (Oct 14)

- Base: stage 06 commit `dfe0c6bd`, checkpoint `fb8ce438`.
- Before numbers: stage 06 `full_model/evidence5/` (tracker on) and `batch_perf.json` (tracker off).

| Profile | TTFT ISL 128 | Token-out ms (t/s/u) | Logits-only ms | B32 ms per step |
|---|---:|---:|---:|---:|
| p150x4 | 98.9 | 9.19 (108.9) | 8.59 | 282.31 |
| p150x2 | 114.1 | 11.36 (88.0) | 10.48 | 343.90 |
| p150 | 183.4 | 15.85 (63.1) | 14.51 | 466.10 |

- Batch-1 decoder stack bound (stage 05 single-layer traced sums): 9.39, 10.92, 14.42 ms.
- Batch-1 logits-only already below that sum; one trace removes per-layer host time.

Candidate work, ordered by measured cost:
- B4/B32 decode loops the whole layer per row. Throughput flat.
- Sampled mode writes 32 seeds from host per step.
- Top-k on full local vocabulary: 285 to 1070 us per step.
- LM head DRAM-interleaved at 66 to 70% DRAM use.
- Routed experts stored twice on p150x2 and p150x4.
- Long prefill: 262100 tokens take 62 to 140 s.

## Sampled-mode seeds without per-step host writes (Oct 14, 12:50 to 13:45)

Raw: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/seed_advance/`. Test: `tests/run_seed_advance.py`.
2-layer model (layers 0, 3), p150x4, B32, 128 steps. Each step: real decode trace replay, then fixed
synthetic logits copied in, then the traced sampler (T 1.0, top-k 32). Tracker on; strict cache off (test
helper ops are not model path).

| Seed mode | Host seed writes | Pooled TV | Chi-square p (4096 draws) | Result |
|---|---:|---:|---:|---|
| Device Weyl add in decode trace | 1 | 0.044 | 0.0005 | rejected: biased |
| SKIP (manual protocol) | 2 | 0.031 | 0.27 | pass |
| Host fresh seeds (stage 06 default) | 129 | 0.026 | 0.44 | pass |
| SKIP (generator default) | 2 | 0.038 | 0.27 | pass |
| Host fresh seeds (generator) | 129 | 0.030 | 0.24 | pass |

- Lag 1 to 32 match rates equal the collision rate 0.144 within 0.013.
- Probe: in-trace `uint32` add wraps mod 2^32 (`seed_probe/result.json`).
- Selected: `seed_mode="skip"` default. Host writes only after reset.
- Failed attempts kept: `failed_strict_cache/`, `failed_live_buffers/` (test buffer after capture).
- Still to rerun on the final source: full sampling audit on all profiles.

## Batched decode, milestone 1 (Oct 14, 13:45 to 14:00)

Raw: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/batched/`. Test: `tests/run_batched_decode.py`
(2 layers 0 and 3, p150x4, tracker and strict cache on; mixed prompt lengths 5 to 1152; B4, B32, B4 with
rows 1 and 3 inactive; traced step timing, 32 steps, median of 3).

- `build_generator(batched_decode=True)`: one layer call for B rows, `[1,B,2048]`.
- Batched: projections, norms, decode all-reduce, attention with B-core KV shards.
- MoE B>1: prefill dispatch path, rows zero-padded to 32.
- GDN: batched projections; per-row conv and decode kernel loop.
- Failed attempts kept: `failed_l2_try1/` (test slice, dispatch `KeyError: 4`), `failed_l2_try2_test_slice/`.

LM head defect found by profile (`profile_b32/`): B>1 logits ran as 32 separate one-row
matmuls (input `[32,1,2048]`), 12.0 ms per step. Fix: `Qwen35Model.decode` reshapes to `[1,B,2048]` first.
Per-row logits after the fix are bitwise equal to before (`l2_lmhead/m4_b0` against `l2_try3/m4_b0`).

| 2-layer step ms | B1 | B4 | B32 |
|---|---:|---:|---:|
| Per-row, before LM head fix | 1.46 | 3.98 | 27.24 |
| Per-row, after | 1.45 | 2.80 | 15.03 |
| Batched, before LM head fix | 1.46 | 4.77 | 18.58 |
| Batched, after | 1.45 | 3.57 | 6.39 |

- Batched against per-row logits: PCC 0.991 to 0.998, not bitwise.
- Cause candidate: MoE prefill expert path numerics. Accuracy check pending.

### Batched decode accuracy (full model, p150x4, Oct 14 14:50)

`tests/run_calibration_probe.py --batch 4 --batched {0,1}`: AIME24 chat reference, 4 identical rows,
100 teacher-forced positions; raw `runs/main/stage07/batched/calib/` (`compare.json`).

| Path | KL mean | KL max | Top-1 agree with HF |
|---|---:|---:|---:|
| Per-row B4 decode | 0.0070 | 0.155 | 99 |
| Batched B4 decode | 0.0058 | 0.081 | 99 |

- Per-row B4 logits bitwise equal stage 06 B1 (`evidence5`). The LM head change is exact.
- Rows within a batch are identical in both modes.

### Batched profile, 2 layers (`profile2_b32/`, `profile2_b4/`)

| Per step, device kernel time | B32 | B4 |
|---|---:|---:|
| Total | 5582 us | 3501 us |
| GDN per-row loop (generic ops, conv, tilize/untilize/slice) | about 2700 us | about 500 us |
| MoE prefill dispatch (post-combine, swiglu, scatter, gather, top-k) | about 1000 us | about 1000 us |

- Next: batched GDN decode kernels (user loop, per-slot state addresses).

### Batched GDN decode kernels (Oct 14, 15:10 to 16:40)

- New `tt/gdn_batch.py`, `tt/kernels/gdn_batch/`. B=1 kernels unchanged.
- `conv_batch`: one channel tile per worker; per-slot conv rows to history tiles; FIR, silu; slot shift.
- `gdn_decode_batch`: (head, block) workers times user groups; per-slot state address args.
- Gate program: own compute kernel; per-row norm (COL broadcast).
- Component check `tests/run_gdn_batch_check.py` against a torch GDN step: PCC >= 0.999996 at 4 and 32 rows.
- Layer check `tests/run_layer_batch_check.py`: batched equals per-row, PCC 1.0, 4 and 32 rows.

Defects found and fixed on the way (raw `runs/main/stage07/batched/gdn_check*/`, `layer_check*/`):

| Defect | Evidence | Fix |
|---|---|---|
| B=1 `norm_gate` uses SCALAR broadcast: row 0 norm for all rows | rows 1 to 3 PCC 0.98 against torch | batched gate kernel uses COL broadcast |
| `generic_op` cache key ignores per-core runtime-arg count | 32-user call after 4-user call: NaN state | user count and span in compile args |
| Test reference through `_AllReduceOutput` wrapper | forward gated output summed over devices | torch reference instead |

- The `generic_op` hash (`generic_op_device_operation.cpp`) hashes `runtime_args.size()` (core count), not
  per-core lengths. A later call with longer per-core args can reuse the cached program. Candidate upstream
  issue.

| 2-layer step ms, p150x4 | B1 | B4 | B32 |
|---|---:|---:|---:|
| Per-row | 1.45 | 2.80 | 15.03 |
| Batched, per-row GDN loop | 1.45 | 3.57 | 6.39 |
| Batched, batched GDN kernels | 1.45 | 3.31 | 4.27 |

### Full-model batched decode timing, p150x4 (Oct 14, 16:40 to 17:50)

`tests/run_batch_perf.py --batched 1`, tracker off, strict cache on, ISL 128, 127 steps, median of 3.
Raw: `runs/main/stage07/batched/full_perf{1,2}/`.

| Step ms (aggregate t/s) | B1 | B4 | B8 | B16 | B32 |
|---|---:|---:|---:|---:|---:|
| Per-row (stage 06 `evidence5`) | 9.18 (109) | 35.87 (112) | | | 282.31 (113) |
| Batched, MoE dispatch padded to 32 | 9.16 (109) | 46.85 (85) | 51.13 (156) | 57.67 (277) | 66.77 (479) |
| Batched, MoE per-row loop for B <= 8 | 9.18 (109) | 21.99 (182) | 36.19 (221) | | 66.77 (479) |

- Per-row MoE loop: about 3.55 ms per extra row. Dispatch: about 46 ms plus 0.65 ms per row.
- Crossover near 13 rows. Dispatch post-combine runs on 1 core for 32 tokens.
- Next: tile combine (`fast_reduce_nc`) for decode dispatch lengths.

| Step ms (aggregate t/s), dispatch combine | B8 | B16 | B32 |
|---|---:|---:|---:|
| `post_combine_reduce` (1 core at 32 tokens) | 51.13 (156) | 57.67 (277) | 66.77 (479) |
| Tile combine, `dispatch_tiles_upto=32` (`full_perf3/`) | 46.52 (172) | 52.89 (302) | 60.93 (525) |

- Defaults set: `dispatch_tiles_upto=32`, `moe_row_loop_max=8`, `build_generator(batched_decode=True)`.

### Batched accuracy with final defaults (p150x4, `runs/main/stage07/batched/calib2/`)

| Path | KL mean | KL max | Top-1 agree with HF |
|---|---:|---:|---:|
| Per-row (stage 06) | 0.0070 | 0.155 | 99 |
| Batched B4 (MoE per-row loop) | 0.0059 | 0.090 | 98 |
| Batched B32 (dispatch, tile combine) | 0.0050 | 0.047 | 98 |

- Rows within each batch identical (`max_row_spread` 0.0).

### Batched decode on p150x2 and p150 (2 layers, `runs/main/stage07/batched/l2_profiles/`)

| Step ms | B1 | B4 per-row | B4 batched | B32 per-row | B32 batched |
|---|---:|---:|---:|---:|---:|
| p150x2 | 2.20 | 3.75 | 3.04 | 18.01 | 6.53 |
| p150 | 3.61 | 5.59 | 4.95 | 23.86 | 12.25 |

- p150x2 batched against per-row logits: PCC min 0.999 (B4), 0.992 (B32).
- p150 B4 bitwise equal: GDN batch kernel falls back (128 cores needed), MoE row loop.
- p150 B32 PCC min 0.988: MoE dispatch path. Full-model accuracy to rerun per profile.

### Greedy sampler choice (p150x4, `runs/main/stage07/sampler/`)

`run_batch_perf --greedy-sampler {topk,argmax}`; argmax uses the common sampler force-argmax path
(`SAMPLING_AG_CONFIG`, full local vocabulary all-gather, Ring).

| Greedy path | B1 step ms | B1 TTFT ms | B32 step ms |
|---|---:|---:|---:|
| Split top-k (local top-32, gather 32 candidates) | 9.18 | 99.7 | 63.86 |
| Force-argmax (gather full vocabulary) | 11.15 | 99.2 | 65.99 |

- Tokens equal (B1, 128 steps). Top-k kept as default (`greedy_sampler="topk"`).
- B32 top-k 63.86 here, 60.93 in `full_perf3`: run-to-run spread about 5%.

### Evidence checks changed for batched decode (Oct 14, 16:20)

Batched decode is deterministic and row-independent but not bitwise equal to B1 (different MoE and GDN
programs). Checks that required bitwise B1 equality now test the right property:

| Test | Old check | New check |
|---|---|---|
| `run_inactive_rows` | active rows equal B1 `generate` | active rows bitwise equal the same bucket with every row active (filler prompts); B1 match kept as `matches_b1` |
| `run_full_evidence` batch_4, batch_32 | 8 greedy tokens equal B1 | B1 tokens teacher-forced; per-step argmax agreement and B1 token rank (top-5 bar) |
| `run_sampling_audit` | B32 rows equal B4 | B32 rows with the same prompt identical; B4 equality count kept as data |

### Evidence chain 1 launched (16:32, PID 2685719)

`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain1.sh` into `evidence1/`: stage 06 chain 5
gates plus 2-layer batched watcher run, SKIP seed check, batch perf B4/B8/B16/B32, B32 batched calibration,
2-layer profiles at B1 and B32 per profile, and a 4096-token prefill profile on p150x4.

## Evidence chain 1 defect: batch_perf ran per-row (18:20)

- `run_batch_perf.py` `--batched` defaults to 0 and overrides the generator default.
- The chain did not pass `--batched 1`, so `evidence1/m*/batch_perf.json` measures the per-row path.
- p150x4 result matches per-row: B4 34.69 ms, B8 68.69 ms per step.
- Other runners use the generator default (batched on). Only batch_perf is affected.
- Action: after the chain, rerun batch_perf with `--batched 1` on all profiles. Keep the per-row file as the baseline.

## Evidence chain 1: p150x4 complete (19:11)

- All 17 p150x4 steps rc=0 (16:35 to 19:11).
- sampling_audit passed=true; `greedy_b32_same_prompt_rows_identical` true.
- full_evidence: batch 4 `all_within_top5` true, batch1 token rank max 1; 262143-token boundary returns 2 tokens.
- calibration B1 and B32 batched: `max_row_spread` 0.0.
- Profiles processed by /home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/process_profiles.sh into evidence1/m4/profile_b1 and profile_b32 (2-layer model, 4 decode steps).
- B32 decode window: 1095 ops, 13249 us kernel on device 0. Top: Matmul 2431, GenericOp 1539, TopkLargeIndices 1135, MoeFusedSwiGlu 1034, BinaryNg 953, Tilize 810 us.
- Host memory dropped to 8 GB available during the B32 Tracy run (Tracy RSS 54 GB). The run completed. Claude Code reaped the log watcher shell.

## Evidence chain 1: p150x2 complete (21:48)

- All 17 p150x2 steps rc=0 (19:11 to 21:46).
- inactive_rows, sampling_audit passed; full_evidence batch_4 and batch_32 `all_within_top5` true.
- batch_32: argmax agrees with B1 at all 8 steps on all 32 rows.
- calibration B1 and B32 batched: `max_row_spread` 0.0. Logits-only decode 10.48 ms.
- batch_perf (per-row, see defect above): B1 11.37 ms, TTFT 114.1 ms.
- Profiles: B1 decode window 8545 us, B32 20062 us (2 layers, 4 steps, device 0).

### Anomaly: seed_skip `device_copies_agree` false on p150x4 and p150x2

- Observed: `evidence1/m4/seed_skip.json` and `m2/seed_skip.json` show `device_copies_agree` false.
- Control: the same test at 17:19 (`runs/main/stage07/seed_advance/m4_skip.json`) shows true.
- Changed since: generator `batched_decode` default true; the test now runs B32 batched decode.
- Production sampled path: `sampling_audit` `device_token_copies_agree` true on p150x4 and p150x2.
- Draw statistics pass (pooled TV 0.027 and 0.030; no outside-support draws).
- The test records only one boolean. It has no step or row count.
- Next (after the chain): add per-step mismatch counts; rerun seed_skip m4 with batched 0 and 1.

## Evidence chain 1: p150 complete, chain done (23:42)

- All 54 chain steps rc=0. `SOURCE_STABLE`. Degenerate check: no degenerate output.
- HF teacher forcing and prefill top-1/top-5 equal stage 06 `evidence5` on all profiles (B1 path unchanged).
- p150 full_evidence batch_4 and batch_32 `all_within_top5` true; 262100-token prefill 140.4 s.
- Profiles processed: m1 B1 and B32, m4 4096-token prefill (prefill window 24662 us, 2 layers).

## Seed copy anomaly resolved: SKIP mode unsafe on multichip (23:45 to 00:05)

Raw: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/seed_copies/`. `run_seed_advance` now records
per-step mismatching rows (`mismatch_steps`, `mismatch_rows`, `mismatches`) and takes `--batched`.
p150x4, B32, 2 layers, 128 steps.

| Seed mode | Decode path | Steps with copy mismatch | Mismatched row-copies |
|---|---|---:|---:|
| skip | batched (MoE dispatch at B32) | 128 | 3967 |
| skip | per-row | 0 | 0 |
| skip | batched, GDN batch kernel off | 128 | 3949 |
| skip | batched, MoE row loop (`moe_row_loop_max` 32) | 0 | 0 |
| skip | batched, both off | 0 | 0 |
| host | batched | 0 | 0 |

- Rows 0 to 3, 11 to 14, 22 to 25 differ from step 0 on devices 1 to 3.
- Trigger: MoE expert-parallel dispatch between SKIP sampling steps.
- SKIP mode keeps each core's PRNG state across steps. Dispatch work differs per device.
- Exact consuming op not identified. Prefill uses the same dispatch path, so serving would also hit it.
- The earlier SKIP pass (17:19) ran per-row decode only, no dispatch between samples.
- `sampling_audit` did not catch it: its distribution loop samples without decode between draws.
- Fix: `seed_mode` default `host`; `skip` raises ValueError when tp > 1. Host mode keeps copies equal.
- SKIP stays available on p150 (one device). CPU suite: 1219 passed.

### Host seed cost, p150x4 (`runs/main/stage07/batch_perf2/`, batched, ISL 128, median of 3)

| Step ms | B1 | B4 | B8 | B16 | B32 |
|---|---:|---:|---:|---:|---:|
| Greedy | 9.18 | 21.99 | 36.21 | 54.62 | 63.89 |
| Sampled (T 1.0, top-k 32), host seeds every step | 9.18 | 22.01 | 36.21 | 55.53 | 65.13 |

- B1 row is greedy in both runs (`generate`). Host seed cost at B32: 1.24 ms (2%).
- `run_batch_perf --batched` default now 1; `--sampled` added.
- `run_sampling_audit` adds a sampled B32 decode run (`sampled_b32_within_top_k`) inside the device-copy check.

## Evidence chain 2 launched (00:20, PID 4005126)

`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain2.sh` into `evidence2/`: CPU tests, a
negative control (sampling audit forced to SKIP on p150x4, expected device-copy failure), then per profile:
sampling audit, host seed test, batched batch_perf greedy and sampled.

## Evidence chain 2 results (02:27, all 14 steps rc=0, `SOURCE_STABLE`)

- Negative control (p150x4, sampling audit forced to SKIP): `s32` device copies disagree, `passed` false. The new check detects the defect.
- Sampling audit, host seeds: passed on all profiles; all 8 device-copy runs agree, including sampled B32.
- Seed test, host seeds, batched B32: 0 mismatched copies; pooled TV 0.039, 0.034, 0.027; no outside-support draws.

Batched token-out, tracker off, ISL 128, 127 steps, median of 3 (`evidence2/m*/batch_perf_sampled{0,1}.json`).
B1 is greedy in both files; B4 to B32 are greedy (s0) or sampled T 1.0 top-k 32 with host seeds (s1).

| Profile | Mode | B1 | B4 | B8 | B16 | B32 | B32 aggregate t/s |
|---|---|---:|---:|---:|---:|---:|---:|
| p150x4 | greedy | 9.19 | 21.99 | 36.21 | 54.61 | 63.89 | 501 |
| p150x4 | sampled | 9.18 | 22.00 | 36.22 | 55.51 | 65.22 | 491 |
| p150x2 | greedy | 11.36 | 28.93 | 48.42 | 82.40 | 105.05 | 305 |
| p150x2 | sampled | 11.40 | 28.94 | 48.41 | 84.57 | 107.58 | 297 |
| p150 | greedy | 15.87 | 45.19 | 78.13 | 158.89 | 222.11 | 144 |
| p150 | sampled | 15.85 | 45.21 | 78.17 | 160.05 | 226.71 | 141 |

## LM head DRAM-sharded candidate (02:40, `runs/main/stage07/lmhead/`)

One device, BFP8 weight, M 32, K 2048. `lmhead_micro.py` (eager) and `lmhead_trace.py` (traced, 200 replays).

| Config (N 62080) | Traced us per call | PCC |
|---|---:|---:|
| Interleaved, default matmul (standalone) | 914.3 | 0.99962 |
| DRAM-sharded, 2 readers, 32 cores, k_block 1 | 395.6 | 0.99998 |
| DRAM-sharded, 3 readers, 32 cores, k_block 2 | 344.8 | 0.99998 |
| DRAM-sharded, 3 readers, 64 cores, k_block 1 | 336.9 | 0.99998 |

- In-model interleaved LM head (profile): 390 us on 108 cores, 66% DRAM.
- Standalone default matmul picks a slower program (914 us); not the model's program.
- 1 reader, or 2 readers with 8 or 16 cores: L1 overflow. 4 readers: not legal (`workers_per_bank` 1 to 3).
- Gain about 53 us per step (0.6% of p150x4 token-out). Needs global vocabulary padding and sampler tail mask.
- Not adopted. Recorded in README.

## Stage 7 README written; tensor dumps moved

- `evidence1/m*/batched_l2_watcher.pt` (38 MB each) moved to `runs/main/stage07/evidence1_work/watcher_pt/`.
- Watcher logs copied as `evidence1/m*/watcher.log.gz`.
- `context_contract.json`: `optimized_full_model_validation` section (`runs/main/stage07/update_contract7.py`).

## Stage review 1: more-work-needed (03:10)

Report: `stage_review1.md`. P1: no B8/B16 correctness or tracker evidence. P2: qualitative only at B1;
per-token readback; perf accounting missing; B32 profile same-prompt and no decode-shaped MoE tried; default
does not reproduce 60.93 ms; LM head rejection reason invalid; chain 1 source differs from final.

## Decode-shaped batched MoE (03:15 to 04:10, `runs/main/stage07/union/`)

Candidate 1: union mask, dense-scan `sparse_matmul` over 256 groups (qwen36 pattern).
- Layer check (delta to per-row) passed; full-model p150x4 slower (`perf_m4.json`):
  B4 47.20, B8 52.64, B16 60.66, B32 72.09 ms. Rejected.
- Micro (`micro_i128.json`, one device, real shapes, random weights, traced), us per layer:

| Rows (union experts) | gate_up | + swiglu | + down | + weight | + reduce |
|---|---:|---:|---:|---:|---:|
| 4 (31) | 307 | 284 | 646 | 738 | 774 |
| 32 (165) | 773 | 801 | 1323 | 1415 | 1451 |

- Cause: dense-scan `sparse_matmul` costs about 10 us per active group, not weight bandwidth.
- Indexed `sparse_matmul` (8 ids per row, `indexed_i128.json`): B4 202, B32 1502 us. gate_up about 4 us per slot (8 cores).
- Custom `expert_swiglu_decode` kernel (`gemv_i128.json`): 1.4 us per slot (54 slots 76 us, 210 GB/s).

Candidate 2 (selected for test): slot path. Per row 8 slots; gate_up with `expert_swiglu_decode` in row chunks
that fit the grid; down with indexed `sparse_matmul`; selector times scores; `fast_reduce_nc` over slots.
- Layer check `slot_layer{0,1}_m4.json`: layer delta (output minus input) PCC to per-row >= 0.9996 at B4 and B32.
- Shared expert at decode rows now uses `shared_decode_minimal_config` (the prefill config overflowed L1).

## Infrastructure incident: two device jobs overlapped (04:48 to 05:00)

- Cause: I started an LM head rerun while an earlier loop's second run still held the devices.
- The new job stalled after fabric init for 8 minutes. Both runs were stopped by PID.
- `tt-smi -r` reset all 4 devices; `tt-smi -ls --local` lists 4 boards.
- Logs kept: `runs/main/stage07/lmhead/failed_concurrent/`, `failed_rank3/` (3-D input slice bug, fixed).
- The overlapped p150x4 interleaved-head result (B1 9.116, B4 17.881, B32 60.111 ms) is not used.

## Slot MoE: full-model timing (`runs/main/stage07/union/slot_perf_m*.json`, greedy, ISL 128, median of 3)

| Profile | Path | B4 | B8 | B16 | B32 |
|---|---|---:|---:|---:|---:|
| p150x4 | row loop / dispatch (evidence2) | 21.99 | 36.21 | 54.61 | 63.89 |
| p150x4 | slot | 17.98 | 24.18 | 35.81 | 60.23 |
| p150x2 | row loop / dispatch (evidence2) | 28.93 | 48.42 | 82.40 | 105.05 |
| p150x2 | slot | 25.48 | 36.40 | 59.11 | 105.05 |
| p150 | row loop / dispatch (evidence2) | 45.19 | 78.13 | 158.89 | 222.11 |
| p150 | slot | 43.05 | 69.61 | 124.76 | 235.97 |

- Default: slot path for 2 to 32 rows (`moe_slot_rows` 32); p150 uses 16, so p150 B32 keeps dispatch.
- p150x2 B32 runs: slot 105.01 / 105.08 / 105.05, dispatch 105.05 / 105.02 / 105.06 ms (equal).
- Batch rows use distinct book slices, so routing differs per row.

## Token readback pipelined (generator `decode_steps`)

- `generate()` and `run_batch_perf` now enqueue replay i+1 before waiting on the token read of step i.
- Read: `tt_tokens.cpu(blocking=False)` plus `record_event`; wait: `event_synchronize`.
- Still one token readback per step (the host needs each token); it overlaps the next replay.

## DRAM-sharded LM head, full model (`runs/main/stage07/lmhead/full_m4_head{0,1}.json`)

`_ShardedHead`: vocabulary padded globally to 248832 (tail on the last device, sampler tail mask), chunks of
62208 columns, 3 readers per bank, 64 input cores, `in0_block_w` 1, HiFi2, FP32 accumulation.

| p150x4, ms per step | B1 (runs) | B1 TTFT | B4 | B32 |
|---|---|---:|---:|---:|
| Interleaved LM head | 9.117 (9.733, 9.111, 9.117) | 98.9 | 17.884 | 60.113 |
| DRAM-sharded LM head | 9.222 (9.847, 9.222, 9.221) | 97.0 | 17.998 | 60.245 |

- B1 greedy tokens identical (first 10 compared, full list in JSON).
- DRAM-sharded is 0.105 ms slower end to end: reshard in, unshard out and the sampler tail mask cost more than the 53 us matmul gain.

## p150 DRAM-sharded LM head (`runs/main/stage07/lmhead/full_m1_head{0,1}.json`)

| p150, ms per step | B1 (runs) | B1 TTFT | B4 |
|---|---|---:|---:|
| Interleaved | 15.809 (16.248, 15.808, 15.809) | 183.6 | 43.036 |
| DRAM-sharded, 4 chunks | 15.922 (16.382, 15.922, 15.921) | 183.9 | 43.090 |

- Slower again (+0.11 ms). `lm_head_dram` stays available, default off. B1 tokens identical.

## Profiler memory and log size

- B32 2-layer profile with `-r` (ops report) timed out at 1500 s: pyarrow CSV parse error, then the slow pandas fallback on a 13 GB `tracy_ops_times.csv` (`runs/main/stage07/profile3/`).
- The C++ device report (`cpp_device_perf_report.csv`, 20 MB) has per-op device time per trace replay session. `runs/main/stage07/trace_sessions.py` summarizes the last 4 replays of each trace.
- p150x4 B32, 2 layers, distinct rows, slot MoE: decode trace 187 ops, 3344 us per replay; sampler trace 28 ops, 499 us.

## Evidence chain 3 launched (06:12, PID 755337, final source; first launch failed in setup, log in failed_chain3_start/)

`runs/main/stage07/evidence_chain3.sh` into `evidence3/`. Adds: source snapshot and diff, layer checks at
B4/8/16/32, buckets 8 and 16 in inactive rows, sampling audit and full evidence (with KL to B1), qualitative
at B8 and B32, distinct-row calibration at B16 and B32, profiles at layers (0,3) and (0,1,3) for B1 and B32.

## Chain 3 interrupted: host memory exhaustion, stale lock, reboot (09:20 to 12:20)

- `calibration_b16_m4` failed: filler book text too short (`* 40`); fixed to `* 400` (`run_calibration_probe.py`).
- `calibration_b32_m4` was killed (log ends `Killed`). The host had about 16 GB available and about 210 GB
  used outside any process, cgroup or `/proc/meminfo` category; tenstorrent debugfs showed no leaked pins.
  Cause not identified.
- The killed process left `CHIP_IN_USE_*` UMD locks; `profile_b1_l03_m4` waited 1 hour on them (rc 124).
- Recovery: stopped the chain by PID, kept copies of the stale locks (`runs/main/stage07/stale_lock_1015/`),
  removed them, `tt-smi -r`, mesh smoke OK. The host then rebooted (uptime 3 min at 12:18); memory available 229 GB.
- Resume chain `evidence_chain3b.sh`: skips the 18 completed p150x4 steps; new source manifest and snapshot
  (`*_b`); profiles without the `-r` ops report (C++ device report only) to bound host memory.
- First resume (12:19): `calibration_b32_m4` failed again (filler too short at stride 997; the book is 59 tokens
  repeated, 23600 tokens at `* 400`); all profiles failed: the reboot removed the `/tmp/q35-functional-profiler-root`
  link. Fixed: stride 97, link recreated. Stopped by PID at m2 watcher, locks removed, `tt-smi -r`, relaunched
  (12:44). Failed outputs in `runs/main/stage07/failed_chain3_start/`.

## p150x4 chain 3 results (13:15)

- All p150x4 steps rc 0 after the resume. Calibration vs HF (decode, 100 positions): B1 KL mean 0.0070 (top-1 99),
  B16 distinct rows 0.0049 (99), B32 distinct rows 0.0049 (99).
- Profiles without `-r` have empty `OP NAME` in the C++ report; per-trace totals are still valid.
- Traced replay device time, device 0 (us): B1 layers (0,3) 883, (0,1,3) 1078; B32 (0,3) 3352, (0,1,3) 4883.
  Sampler trace: B1 499, B32 497.
- `calibration_b32_m2` failed at mesh open (`Query mappings failed on device 2: No such device`, 16:15).
  The kernel log shows `FW log forwarding enabled` on all 4 devices at the same time (driver re-init); another
  user (`stisi`) logged in at 16:15. No other user TT process seen afterwards. The next step (profile) waited
  about 40 s on ETH training, then ran. Rerun after the chain.

## Chain 3b paused: another user holds the devices (17:33)

- `tf_book_m1` waited 25 minutes on `CHIP_IN_USE_0_PCIe`.
- tenstorrent debugfs: user `stisi` runs `VLLM::EngineCore` (Docker, started 17:05) mapped to device 0.
- I stopped my chain and its waiting child by PID. I did not touch the other user's process.
- Completed: all p150x4 steps; p150x2 except `calibration_b16_m2` (rc 135) and `calibration_b32_m2` (rc 1)
  at the 16:15 device event; p150 up to `tf_aime_m1`.

## Devices held by another user until host reboot (Oct 15 17:33 to Oct 23 10:10)

- Another user's vLLM process held device 0 from Oct 15 17:05.
- Kernel log, Oct 19 02:34: telemetry tasks blocked on a driver mutex owned by that process.
- My launcher was stopped at the user's request (Oct 15 21:10). No device job of mine ran.
- Host rebooted Oct 23 10:10. All 4 devices free. Profiler `/tmp` link recreated; 4-chip mesh smoke OK.

## Chain 3c (Oct 23 14:08 to 16:12, all 21 steps rc 0, `SOURCE_STABLE`)

- p150x2 calibration B16/B32 rerun; all p150 steps; p4096 profile; degenerate check.
- `tt/` identical to chains 3 and 3b (manifests).

## Accounting profiles (16:17 to 16:46)

- `runs/main/stage07/profile_l037.sh`: layers (0,3,7) for B1/B32 on every profile (`SOURCE_STABLE`).
  First launch failed in setup (missing shell functions), log in `runs/main/stage07/failed_profile_l037_setup/`.
- `runs/main/stage07/profile_named.sh`: `-r` profiles with op names, layers (0,3), B1/B32, every profile.
  All 6 rc 0; p150x4 B32 took 6 minutes (the earlier 1500 s timeout no longer occurs).
- `perf_accounting3.py`: GDN-type layer = (0,1,3) - (0,3), full-type = (0,3,7) - (0,3). Each includes its MoE.
- Device time within 1% of end to end, except p150 B32 (+1.9%, profiler replay drift).
- tt-perf-report per-op tables omit trace-replay rows ("495 rows had invalid device durations"); the
  per-replay breakdown uses the raw ops CSV (`runs/main/stage07/named_breakdown.py`).

## Projection wiring defect at batched decode (16:47)

- p150x4 B32 profile: GDN `attention_in` ran `minimal_matmul` 32 x 2048 x 3136 (49.9 us, 25% DRAM).
  B1 runs the decode 1D program (23.6 us, 53% DRAM).
- Cause: `_Projection.__call__` applied `prefill_minimal` / `prefill_grid` for any length > 1,
  overriding the `decode_grid` program config at batched decode rows.
- Fix: prefill branches apply only when no decode program config is set.
- B1 (length 1) and prefill (no decode batch context) take the same branches as before.

Full model, greedy, ms per step (`runs/main/stage07/projfix/`, `SOURCE_STABLE`):

| Profile | B4 | B8 | B16 | B32 |
|---|---|---|---|---|
| p150x4 | 17.91 -> 16.70 | 24.10 -> 22.87 | 35.71 -> 34.52 | 60.12 -> 58.85 |
| p150x2 | 25.41 -> 23.04 | 36.31 -> 33.99 | 58.98 -> 56.65 | 104.89 -> 102.61 |
| p150 | 43.04 -> 42.32 | 69.51 -> 68.74 | 124.60 -> 124.20 | 221.75 -> 220.85 |

- Kept. Chain 4 (`runs/main/stage07/evidence_chain4.sh`) reruns the batched checks into `evidence4/`.

## Chain 4: batched requalification after the projection fix (Oct 23 17:26 to 22:58)

- `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain4.sh`: 48 of 48 steps rc 0, `SOURCE_STABLE`.
- Every batched check passes on every profile (`evidence4/check_compare_ev3_ev4.json`).
- Accounting rerun for B32 (`perf_accounting4.py`); B1 rows from `evidence3/`.
- KL to HF, B16 and B32: p150x4 0.0068 / 0.149, p150x2 0.0067 / 0.255, p150 unchanged.
- Batched qualitative rows coherent; same-prompt rows identical.
- Context contract regenerated from `evidence4/` (`update_contract7_ev4.py`).

## Layer-check anomaly probe (Oct 23 22:59 to 23:06)

- Layer checks in `evidence4/` are bitwise equal to `evidence3/`.
- Instrumented probe (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/projfix/layer_probe/`):
  - post-fix batched path calls the 1D decode config;
  - pre-fix path (patched in the probe only) calls `minimal_matmul`;
  - both give bitwise-equal B4 outputs for GDN and full layers.
- Full-model B16 logits equal before and after for the first 2 to 3 positions, then differ.
- B1 logits are bitwise equal across 4 runs on 3 sources.
- Classified as controlled in the README anomaly ledger.

## Review 2 remediation: decode programs at batched rows (Oct 23 23:36 to Oct 24 00:46)

- Review 2 P2 #1: batched decode still ran prefill programs and slot glue.
- Changes (`tt/optimized_decoder.py`):
  - router linear passes the batch-1 program config;
  - router top-k and `_Projection` prefill branches test `not decode_rows(length)`;
  - slot MoE lays out expert indices once and slices per chunk.
- Trial `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/fix2/`: all rc 0, `SOURCE_STABLE`.

| Profile | B4 | B8 | B16 | B32 |
|---|---|---|---|---|
| p150x4 | 16.70 -> 14.58 | 22.87 -> 20.21 | 34.52 -> 31.09 | 58.85 -> 52.84 |
| p150x2 | 23.04 -> 20.47 | 33.99 -> 30.51 | 56.65 -> 51.08 | 102.62 -> 91.88 |
| p150 | 42.32 -> 36.50 | 68.74 -> 61.11 | 124.20 -> 111.44 | 220.85 -> 214.83 |

- Layer delta PCC equal or better; B32 full-attention layer 0.9996 -> 0.99993.
- B16 KL to HF: p150x4 0.0058, p150x2 0.0062, p150 0.0055.

## Review 2 remediation: slot glue (Oct 24 00:48 to 02:11)

- Routing weights now scale the expert activations before the down `sparse_matmul`.
  - Removes the multiply over the 2048-wide padded down output.
  - Mask and scale use the same selector; down is linear, so the sum is the same.
- Batched hidden stays `[1,B,H]` across all layers (`packed` in `decode_forward`).
  - Removes two reshapes per layer; one reshape at the embedding.
- Trial `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/fix3/`: all rc 0, `SOURCE_STABLE`.

| Profile | B4 | B8 | B16 | B32 |
|---|---|---|---|---|
| p150x4 | 14.58 -> 13.82 | 20.21 -> 18.90 | 31.09 -> 28.54 | 52.84 -> 47.97 |
| p150x2 | 20.47 -> 19.73 | 30.51 -> 29.26 | 51.08 -> 48.62 | 91.88 -> 87.23 |
| p150 | 36.50 -> 35.70 | 61.11 -> 59.66 | 111.44 -> 109.17 | 214.83 -> 212.51 |

- Layer delta PCC unchanged from fix2.
- B16 KL to HF: p150x4 0.0083, p150x2 0.0050, p150 0.0070.
- TT-to-TT KL fix2 vs fix3: 0.0039 / 0.0021 / 0.0026 (x4 / x2 / p150).
  - Same size as evidence4 vs fix2: 0.0033 / 0.0026 / 0.0025.
  - Rounding-order change, not an accuracy loss.

## Slot selectors shared across layers (Oct 24 02:00 to 02:11)

- Census growth since Stage 6: +37.5 MiB (x4, x2), +17.5 MiB (p150).
- Cause: per-layer slot selector constants, 0.9375 MiB per layer at rows 4,8,16,32.
- Fix: `tt/model.py` `_share_slot_selectors` gives all layers one cache.
- Smoke `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/fix4/`: p150x4 B4 and B32 step unchanged.
- Chain 5 (`runs/main/stage07/evidence_chain5.sh`) requalifies into `evidence5/`.

## Chain 5: requalification into evidence5 (Oct 24 02:11 to 07:49)

- `runs/main/stage07/evidence_chain5.sh`: 48 of 48 steps rc 0, `SOURCE_STABLE` (full manifest compare).
- p150x4 and p150x2 pass every check. Final step ms in README.
- p150 B32 qualitative: rows with the same prompt diverged. `evidence4/` had them identical.

## Fix5: prefill programs inside the p150 dispatch branch (Oct 24 07:50 to 08:17)

- Cause: fix2 made router top-k and `_Projection` pick decode programs at decode row counts.
  - The p150 B32 branch pads to 32 rows and runs `moe.forward(mode="prefill")`.
  - Inside it, those calls now used decode programs with prefill dispatch.
- Fix: `PREFILL_PROGRAMS` context variable set around that call; `prefill_rows(length)` tests it.
- Reach: only rows above `moe_slot_rows`; p150x4 and p150x2 never reach it (slot rows 32).
- Trial `runs/main/stage07/fix5/`: p150 B32 same-prompt rows identical; KL to HF 0.0071; 215.12 ms.

## Chain 5b: p150 rerun on the fix5 source (Oct 24 08:17 to 09:46)

- `runs/main/stage07/evidence_chain5b.sh`: all 16 p150 steps rc 0, `SOURCE_STABLE` (`_m1b` manifests).
- Pre-fix5 p150 evidence moved to `evidence5/m1_pre_fix5/` and `evidence5/qualitative/pre_fix5/`.
- p150 step ms: B4 35.70, B8 59.66, B16 109.16, B32 215.00 (sampled 222.17).
- p150 B32 KL to HF 0.0071 (top-1 99); KL to B1 maximum 0.020.
- Post-processing rerun: profiles, named breakdown, accounting, calibration, checks, qualitative, contract,
  perf summary, memory handoff recheck.
- `qual_compare5.py` now adds HF common prefix and a B8-equals-B32 row check.
- Memory recheck: layer-private DRAM vs stage 06 +1.0 / +0.9 / +0.4 MiB (selectors shared).

## Review 3 remediation (Oct 24 10:17)

- Review 3: `stage_review3.md`, more-work-needed, documentation only.
- L1 growth vs stage 06: +32 KiB per bank (meshes), +24 KiB (p150).
  - Matches router cached slices for widths 4 to 32 exactly.
  - Added to `memory_handoff_recheck.json` and README context section.
- `perf_summary.json` `named_limitations` now computed from evidence5 accounting.
- README wording: p150 matmul net change, all-reduce spread, calibration positions.

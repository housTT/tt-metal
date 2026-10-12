# Qwen3.5-35B-A3B optimized multichip decoder: stage 05

- Checkpoint: `Qwen/Qwen3.5-35B-A3B`, revision `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- Hardware: four Blackhole ASICs on two P300c cards; Ethernet ring 0-1-2-3-0.
- Implementation: `tt/multichip_decoder.py` (`MultichipDecoder`) on `tt/optimized_decoder.py`, plus two
  model-local device kernels: `tt/gdn_decode.py` (`tt/kernels/gdn_decode/`) and `tt/expert_decode.py`
  (`tt/kernels/expert_gemv/`).
- Start point: Stage 4 commit `dca97aa5` (`../multichip_decoder/evidence7/`).
- Profiles: p150 = 1x1 (physical device 3), p150x2 = 1x2 (`TT_VISIBLE_DEVICES=2,3`, `FABRIC_1D`),
  p150x4 = 1x4 ring (`FABRIC_1D_RING`). Optimization target: p150x4.
- Final evidence: `evidence12/`, the full chain on the final source (`runs/main/stage05/evidence_chain12.sh`, log
  `evidence_chain12.log`, 2026 Oct 12 15:07 to 17:03 Eastern). Decode, aligned SDPA, 126 length rows, 128-step
  rows, B32, acceptance (reuse, positions, context, watcher, real inputs, device and CPU pytest), long watcher
  runs, nondeterminism, probes, repeated prefill table, placement, inventory, long-chunk accuracy, long context
  (8192-token external chunks) and profiles. SOURCE_STABLE, CHILD_FAILURES 0, 255 steps, all with exit code 0.
  Derived: `capacity_model.json`, `memory_handoff_recheck.json`, `roofline.json`, `profiles/`
  (`runs/main/stage05/process_profiles12.sh`).
- Stage commit: `1f9fb518c5ed9428ad54be1be700714447991ee2` (local branch `feat/qwen3-5-35b-a3b`, not pushed).
- Source delta from chain 10 (`evidence12/source_delta.diff`, `evidence12/ttnn_patch.diff`): indexed
  `sparse_matmul` without zero-fill (ttnn), conv-state shift in the GDN gate writer, 8192-row full-attention
  prefill step from position 8192, the conv row reference fix, the profiler zone label, the `run_layer --profile`
  guard. Details: "What changed" and `work_log.md` (review 4 follow-up).
- Superseded, kept: `evidence1_aborted/`, `evidence2/` to `evidence10/`, `evidence8_aborted*/` (reasons in
  `work_log.md`). `evidence9/` is the control for the addcmul change; `evidence10/` is the same chain on the source
  before the review 4 changes and is the control for them.
- Failed attempt, kept: `evidence11/` (chain 11, same script and source as chain 12 without the conv row reference
  fix). 6 steps exited with code 1, CHILD_FAILURES 3: `accept_reuse_m{4,2,1}` linear and `real_reuse_0_m{4,2,1}`
  failed with the trace allocation tracker reporting 1 buffer alive before replay (`evidence_chain11.log`).
  Cause and fix: "Trace, determinism, audit, watcher".
- History, every screen and every rejected candidate: `work_log.md`. Reviews: `stage_review1.md` to
  `stage_review4.md`.

Abbreviations: TP (tensor parallel), EP (expert parallel), MoE (mixture of experts), GDN (Gated DeltaNet, the
linear-attention layer), KV (key/value cache), CCL (collective communication), PCC (Pearson correlation
coefficient), HF (Hugging Face reference), BF16/FP32 (16/32-bit float), BFP8/BFP4 (8/4-bit block float), DRAM
(device memory), L1 (core SRAM), CB (circular buffer), SDPA (scaled dot-product attention), TTFT (time to first
token), ULP (unit in the last place), I2S/S2I (interleaved-to-sharded / sharded-to-interleaved), RoPE (rotary
position embedding), RS/AG (reduce-scatter / all-gather), AR (all-reduce).

## Results

### Traced decode (`evidence12/headline_*`, `evidence12/aligned_sdpa/`; 4096 prefill tokens, 128 traced steps)

Host time per traced decode step: median of the 128 `traced_replay_host_us` values per run, then the median of 3
runs. Recorded HF inputs. Runtime audit on every run.

| Profile | Full decode, aligned table | Full decode, 132-block table | Linear decode | Chain 10 (aligned / 132-block / linear) | Stage 4 full / linear |
|---|---:|---:|---:|---:|---:|
| p150x4 | 254.4 us | 257.0 us | 228.1 us | 257.4 / 260.6 / 236.4 us | 346.5 / 496.9 us |
| p150x2 | 287.7 us | 290.5 us | 268.1 us | 289.8 / 293.3 / 277.5 us | 377.1 / 575.0 us |
| p150 | 365.9 us | 367.9 us | 358.7 us | 368.9 / 370.6 / 369.8 us | 434.7 / 709.1 us |

- "Aligned table": page-table coverage a multiple of 512 tokens (serving tables are 8192 blocks); 3 runs each
  (`aligned_sdpa/m{1,2,4}_r{1,2,3}.json`, x4 254.2 / 254.4 / 254.5 us). The headline rows use the harness
  132-block table; SDPA then runs with k_chunk 128.
- Chain 12 vs chain 10: linear decode -8.3 / -9.4 / -11.1 us (x4 / x2 / p150, -3.0% to -3.5%); full decode -2.1 to
  -3.6 us (-0.7% to -1.4%). This matches the review 4 screens: zero-fill removal -2.6 to -3.0 us on all 6 rows,
  gate-writer conv shift -5.5 / -6.9 / -8.5 us on linear rows (`work_log.md`, `trials/indexed_nofill/`).
- Chain 9 (source before the addcmul change) gave 260.1 / 263.3 / 239.3 us on x4, 291.0 / 294.4 / 278.5 us on
  x2 and 371.3 / 373.3 / 372.2 us on p150. The addcmul screen predicted -2.3 / -2.7 us on x4 (`work_log.md`).
  The p150x2 gain from the addcmul change is small: -0.7 / -0.1 us (linear / full) in the screen, against -2.3 /
  -2.7 us on x4 and -2.4 / -2.3 us on p150 (`work_log.md`, "Shared-expert tail: fused addcmul with a BFP8
  weight").
- p150x2 on physical devices 0 and 1 (`placement_p150x2_01_*`, one run): 295.9 us full, 274.3 us linear (devices 2
  and 3: 290.5 / 268.1 us). Chain 10 (299.3 / 284.4 us) and chain 7 (300.2 / 285.0 us) show the same gap; the cause
  is not profiled.
- Stage 4 numbers: `../multichip_decoder/evidence7/`, same harness.

### Warm prefill, repeated (`evidence12/prefill_timing/`; Stage 4: `../multichip_decoder/evidence7/micro/prefill_timing7/`)

Same script for both stages (`timing.py`): synchronized `prefill_forward` host time, 1 compile run, 6 warm runs.
The table gives the median of the 6 warm runs (mean of the 3rd and 4th sorted values, `statistics.median` over
`warm_runs_us`). The `warm_median_us` field that `timing.py` writes is the 4th sorted value; the table does not
use it. Final source. ms.

| Profile / layer | 2 | 3 | 100 | 257 | 1025 | 1300 | 1536 | 2049 | 2176 | 2600 | 3300 | 4095 | 4096 | 4100 | Stage 4 at 4096 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p150x4 linear | 2.69 | 2.66 | 3.43 | 3.88 | 4.92 | 6.51 | 6.60 | 6.71 | 9.44 | 10.85 | 10.99 | 11.10 | 10.56 | 11.73 | 20.67 |
| p150x4 full | 2.67 | 2.56 | 3.32 | 4.14 | 5.23 | 6.86 | 6.80 | 7.16 | 9.93 | 11.12 | 11.52 | 11.45 | 11.14 | 12.51 | 18.81 |
| p150x2 linear | 2.75 | 2.67 | 4.03 | 4.85 | 6.49 | 8.83 | 9.16 | 9.22 | 12.94 | 14.33 | 14.59 | 14.76 | 14.19 | 15.55 | 24.83 |
| p150x2 full | 2.58 | 2.56 | 4.19 | 4.71 | 6.78 | 8.65 | 8.76 | 9.26 | 12.75 | 14.45 | 14.87 | 14.97 | 14.63 | 16.17 | 21.59 |
| p150 linear | 2.23 | 3.20 | 5.66 | 6.71 | 9.86 | 12.67 | 13.25 | 13.70 | 17.24 | 19.07 | 20.14 | 21.38 | 20.44 | 22.21 | 32.01 |
| p150 full | 3.02 | 2.93 | 5.53 | 6.58 | 9.45 | 12.21 | 12.41 | 13.08 | 17.07 | 18.53 | 19.79 | 20.35 | 20.10 | 21.76 | 25.21 |

- All 21 lengths per row are in the JSON (2, 3, 100, 257, 1000, 1025, 1100, 1300, 1536, 2049, 2176, 2400, 2600,
  3000, 3300, 3968, 4095, 4096, 4097, 4100, 4103). Every length that Stage 4 also measured (13 lengths) is faster
  than Stage 4 on every row.
- Short lengths are noisy: 15 of 126 cells (all at 2-1025 tokens) have a max/min warm-run ratio above 1.2, up to
  1.46; the slow run is the first warm run in every one of them (x4 linear 2 tokens, chain 12: runs 3.70, 2.83,
  2.74, 2.64, 2.64, 2.58 ms; chain 10: 3.68, 2.65, 2.68, 2.61, 2.53, 2.57 ms; chain 9 gave a 2.92 ms median with
  the same plan).
- Chain 10 vs chain 12 per cell: at most 0.35 ms apart (p150 full 3 tokens, 3.29 -> 2.93 ms). 22 of 126 cells move
  by more than 2%; 17 of them are at 2-1025 tokens (the noisy range). The others are x4 full 1300 / 2400 / 3300
  (+2.6% / +2.5% / +2.2%), x2 full 1300 (+2.3%) and p150 full 1536 (-2.0%); they are not profiled. The review 4
  source changes target decode and full-attention steps at or after position 8192 (`evidence12/source_delta.diff`).
- Mesh mid-length plans (Prefill planner, below): x4 linear 1300 7.95 -> 6.51 ms, 2600 11.84 -> 10.85 ms, 3300
  11.76 -> 10.99 ms (chain 8 -> chain 12). x2 is slower at 2600 and 3000 (linear +0.30 / +0.52 ms, full +0.79 /
  +0.86 ms vs chain 8); disclosed below.
- Mesh prefill is still not monotonic between 2176 and 4095 tokens (x4 linear 2600 = 10.85 ms vs 4096 = 10.56
  ms): 2176 runs a 2048 chunk plus a 128 chunk, and 2600-4095 run one masked 4096 chunk. The masked-chunk cost is
  attributed (Anomaly ledger, `micro/masked_chunk/`).
- Headline warm prefill at 4096 (harness, chain 12, median of 3 runs of `warmed_prefill_host_us`): p150x4 10.74 /
  11.55 ms (linear / full), p150x2 14.37 / 15.05, p150 20.51 / 20.52. Chain 10: 10.83 / 11.56, 14.41 / 15.16,
  20.56 / 20.71.

### 40-layer layer-sum estimate (10 full + 30 linear; not a full-model measurement)

| Profile | Decode per token, Stage 4 | Stage 5 | Speedup | Prefill 4096, Stage 4 | Stage 5 | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| p150x4 | 18.37 ms | 9.39 ms | 1.96x | 808 ms | 428 ms | 1.89x |
| p150x2 | 21.02 ms | 10.92 ms | 1.92x | 961 ms | 572 ms | 1.68x |
| p150 | 25.62 ms | 14.42 ms | 1.78x | 1212 ms | 814 ms | 1.49x |

- Decode uses the aligned full-layer value; prefill uses the repeated table at 4096 (true median of 6 warm runs
  for both stages). Chain 10 Stage 5 values: decode 9.67 / 11.22 / 14.78 ms, prefill 427 / 570 / 814 ms.

### Accuracy (headline rows, 3 runs each, `evidence12/headline_*`)

- HF prefill PCC 0.99950-0.99951 (full) / 0.99968-0.99969 (linear) on every profile.
- HF decode min (tie-aware, below) 0.99858-0.99859 (full), 0.99971-0.99972 (linear); strict 0.99837 (full),
  0.99040-0.99078 (linear, router near tie at one token).
- Control (Stage 4 single-chip baseline tensors): prefill 0.99987-0.99989, decode min 0.99960-0.99967.
- Against chain 10: prefill PCC, per-token prefill lists, HF decode min, strict decode min and control decode min
  are identical on all 18 headline runs. The zero-fill removal and the gate-writer conv shift do not change these
  metrics.
- Against chain 9 (source before the addcmul change): prefill values and per-token prefill lists are identical;
  HF decode within 0.00002; control decode min higher on all 6 rows; strict decode min within 0.00015.
- Random-activation rows (B32, positions, context, watcher) have lower decode minimums than chain 9 by 0.0004 to
  0.0010; all pass 0.995 (Non-aligned lengths, batch, acceptance). Chain 12 gives the same values as chain 10.

## What changed (each measured as a screen, `work_log.md`)

| Change | Policy / code | p150x4 effect (per layer unless noted) |
|---|---|---|
| Fused FP32 GDN decode kernel | `gdn_decode_kernel`, `tt/gdn_decode.py` | linear decode 457 -> 356 us |
| GDN in-projection BFP8 HiFi2 + geometry | `projections.attention_in` (linear) | 356 -> 339 us |
| 16-core sharded decode norms (linear layers) | `decode_norm_cores` | about -30 us |
| Direct decode all-reduce, persistent width-sharded L1 buffer | `_decode_allreduce` | 2 x (16.8 -> 12.0 us) |
| Routed expert gate/up + SwiGLU kernel | `expert_decode_kernel`, `tt/expert_decode.py` | -23 / -25 us |
| Shared expert as an extra slot of that kernel | `expert_decode_shared` | -9 / -10 us |
| Decode fill-pad ops removed | `decode_input_fill`, `decode_norm_fill` false | -15.7 us |
| Width-sharded L1 residual stream | `residual_sharded_decode` (tp2, tp4) | -4 to -7 us |
| Norm output in L1; router constant slices cached | `decode_norm_output`, `router_cached_slices` | -10 to -12 us |
| SDPA decode grid 11x10, k_chunk 512 | `sdpa` | -5 us at 4096; 571 -> 228 us at 131072 (op) |
| Chunked SDPA prefill q128/k256 | `sdpa.prefill` | 62.1 -> 57.2 ms per chunk at 126976 (op) |
| Multi-row dispatch-map scatter; EP dispatch spread | `dispatch_scatter_rows`, spread default | prefill -0.8 to -1.2 ms |
| Conv decode fill removed | `conv_decode_fill` false | linear decode -6.8 us |
| GDN decay and beta folded into the decode kernel | `gdn_decode_gates` | linear decode -21 us |
| GDN kernel reads conv outputs and z in place (4 slices removed) | `gdn_decode_direct` | linear decode 245.3 -> 240.1 us (x2 -7.4, p150 -7.4) |
| Router + shared gate at HiFi2 (decode and prefill) | `gate_pack_fidelity`, `projections.router` | -0.5 to -2.7 us on all 6 rows |
| Prefill norm: one `rms_norm`, BF16 gamma | `prefill_norm` bf16 | prefill -1.1 ms (linear) |
| Prefill router: one `topk` + FP32 softmax | `router_topk_prefill` | prefill -2.7 / -2.8 ms |
| Prefill GDN q/k normalized in the chunk kernel | `gdn_flat_qk` | prefill -1.4 ms (linear) |
| GDN output gate in mixed precision | `gdn_gate_mixed` | prefill -0.2 ms (linear) |
| One MoE piece when the second piece is >= 1024 rows (meshes) | `prefill_moe_merge_min` | 2600-3000 rows: up to -1.5 ms (x2), -0.5 ms (x4) |
| Planner costs: masked chunk for 2-3 token prompts (meshes); linear one-step tails | `_PLAN_COST`, `plan_start_step_cost` | 3 tokens full 4.64 -> 2.58 ms; 2049 tokens linear 8.68 -> 6.84 ms |
| Mesh planner chunk and row costs (one masked chunk instead of a ladder) | `plan_chunk_cost` 2.5, `plan_row_cost` 0.002 (tp2, tp4) | 1100-3300 tokens: up to -2.9 ms (x4), see "Prefill planner" |
| Duplicate BF16 GDN in-projection released | `_release_gdn_input_weight` | -12.8 MB per linear layer (x4) |
| Shared down, sigmoid gate and add in one `dit_minimal_matmul_addcmul_fused`; routed down output BFP8 | `shared_addcmul_config` (K block 16 / 8 / 4 for tp 1 / 2 / 4) | -2.3 / -2.7 us (x4 linear / full), x2 -0.7 / -0.1 (small), p150 -2.4 / -2.3 |
| Indexed expert-down `sparse_matmul` without zero-fill (ttnn): the output is no longer zero-filled because the in1 writer writes every active slot | `use_indices` counts as a compact output (`evidence12/ttnn_patch.diff`; `runs/main/stage05/trials/indexed_nofill/`) | removes one FILL op (3.1-3.4 us); -2.6 to -3.0 us on all 6 rows (sequential builds) |
| Conv-state shift fused into the GDN gate writer kernel | `conv_shift="gate"` (linear default), `tt/kernels/gdn_decode/gate_writer_shift.cpp` | removes 4 ops (5.6 us); linear decode -5.5 / -6.9 / -8.5 us (x4 / x2 / p150, new build) |
| Conv row reference released after the GDN kernel | `_decode_kernel` takes and clears `_conv_current`; `_decode_heads` raises if a gate-shift row is pending | trace safety; fixes the chain 11 failure (`runs/main/stage05/micro/conv_ref_fix/`) |
| 8192-row full-attention prefill step from position 8192; long SDPA configs | `prefill_long_chunk` 8192, `prefill_long_from` 8192, `sdpa_prefill_long` q64/k256 (x4), q128/k512 (x2, p150) | 131072-token full layer, ms per 4096 tokens after the first chunk: x4 40.5 -> 39.4, x2 73.7 -> 58.5, p150 108.4 -> 92.4 (`micro/long_chunk/`) |
| Profiler zone label in the SDPA compute kernel | "Reduce max" -> "Reduce_max" in `compute_streaming.hpp` (`evidence12/ttnn_patch.diff`) | none; behavior unchanged; removes a 16-bit zone hash collision that stopped full-layer profiles |

## Operation-topology audit

### Decode, linear layer (p150x4; Stage 4 `../multichip_decoder/evidence7/profiles/`, final `evidence12/profiles/`)

| Segment | Stage 4 ops / us | Action | Final (ops `decode_step_ops.txt`) |
|---|---|---|---|
| Input norm (fill, typecast, 1-core FP32 norm) | 5 / 43.1 | 16-core sharded norm, sharded residual input | I2S 1.2 + norm 6.0 + S2I 1.0 us |
| GDN in-projection | 1 / 41.7 | BFP8 HiFi2, [10,2] grid k8; LoFi screened (no gain, lower PCC) | 23.7 us |
| Conv (`qkv_causal_conv1d_silu`), state shift | | conv kernel; state shift moved into the GDN gate writer (below) | slice + untilize 5.0, conv 14.7 us; shift: no separate op (chain 10: 4 ops, 5.6 us) |
| q/k L2 norm, head layout, recurrence, output norm, gate, decay/beta, input slices | 40 / 158 + 8 ops | fused FP32 kernel reads conv outputs and z in place; the gate program writer also shifts the conv state | 26.7 + 13.7 us (2 ops; chain 10: 26.7 + 14.1) |
| Attention-out + AR (RS + AG) | 4 / 32.4 | direct all_reduce_async, persistent buffer | 9.6 + I2S 1.7 + AR 7.6 us |
| Residual add + post norm | 5 / 41.1 | sharded residual, sharded norm | add 1.6 + norm 5.8 + S2I 1.0 us |
| Router + shared gate, MoE gate | 7 / 11.6 | HiFi2 (HiFi4 before); top-k decode +40 us, rejected | matmul 6.9 + 7 small ops 11.6 us |
| Routed + shared expert gate/up + SwiGLU | 7 / 60.7 | expert kernel, 36 cores | 16.6 us |
| Expert down + weighted reduce | | indexed `sparse_matmul` without zero-fill (ttnn) + fused reduce, BFP8 output | 12.0 + 7.1 us (chain 10: fill 3.3 + 13.0 + 6.8; `evidence7/profiles/`, BF16 output: 13.4 + 7.1) |
| Shared down, sigmoid gate, add, AR | | shared down, gate and add in one `dit_minimal_matmul_addcmul_fused`: adopted (below); shared down folded into the routed `sparse_matmul` as expert 256 (2 variants): +5 to +7.5 us, rejected | slice 0.6 + gate multiply 3.9 + addcmul 6.9 + I2S 1.6 + AR 9.5 + add 1.4 us (`evidence7/profiles/`: slice 0.6 + down 4.6 + multiply 4.8 + add 2.5, then the same I2S, AR, add) |

- Linear decode step: 33 ops, 197.9 us kernel (chain 10: 38 ops, 208.7 us). The 5 removed ops are the 4 shift
  ops and the expert-down fill.

- Repeated same-input matmuls: none (GDN in-projection packed; router + shared gate packed; shared gate/up in
  the expert kernel).
- Collectives: two all-reduces per layer, nothing between layers.
- Residual-boundary conversions (S2I after norms, I2S before the ARs, 5.5 us): the consumers that would remove
  them are slower (DRAM-sharded in-projection 28.7 vs 22.3 us; attention-out on the AR grid 10.55 vs 8.58 us).
  Kept.
- Conv-state shift (`trials/indexed_nofill/`, review 4): `qkv_causal_conv1d_silu` has no state output, so
  chain 10 shifted the state with 4 ops (slice, slice, concat, copy, 5.6 us). The shift now runs in the GDN gate
  program writer (`tt/kernels/gdn_decode/gate_writer_shift.cpp`, `conv_shift="gate"`, linear default). Each gate
  core copies a 512-byte column span of state rows 1-2 and the new qkv row into L1, then writes state rows 0-2.
  The gate program runs after the conv op, so the conv reads the old state. Screen (median of 3 interleaved runs,
  recorded inputs, HF and control decode min identical across arms): linear -5.5 / -6.9 / -8.5 us (x4 / x2 /
  p150). The alternative `conv_shift="concat35"` (3 ops) was not faster (p150 +4.4 us); rejected.
- Expert-down zero-fill (`trials/indexed_nofill/`, review 4): reviews 2 and 3 named the 3.1-3.4 us op before the
  indexed expert-down `sparse_matmul` a sigmoid; it is the `ttnn::zeros_like` FILL in `create_output_tensors`. In
  indexed mode the in1 writer writes every active slot, so the fill is redundant. The ttnn change
  (`evidence12/ttnn_patch.diff`) treats `use_indices` as a compact output and skips the fill. Screen: -2.6 to
  -3.0 us on all 6 rows (sequential builds, not interleaved). ttnn unit tests on the rebuilt library
  (`_ttnncpp.so` sha256 prefix b38c43f3): `test_sparse_matmul_indexed.py` 59 passed, including the case that
  prefills the output with 99.0; `test_sparse_matmul.py` 29 passed, 2 skipped
  (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/micro/review5/test_sparse_matmul*.log`).
- Shared-expert fold (`runs/main/stage05/trials/shared_fold/`, meshes, where decode experts are a separate TP
  copy): shared down appended as expert 256 of a 257-expert BFP8 down weight, the gate/up kernel writes the
  shared activation as a 9th row, ids gain 256, scores gain the sigmoid gate, one fused reduce over 9 rows. It
  removes the shared down matmul, gate multiply and add (12.5 us). Variant 1 (runtime id/score concat), x4,
  3 interleaved pairs: linear 239.3 -> 246.8 us, full 263.0 -> 270.0 us, PCC identical; each row-major concat
  lowers to 4 ops (8.4 us for two). Variant 2 (preallocated 9-entry buffers, per-step `slice_write`): kernel sum
  210.8 vs 210.1 us; the 257-expert `sparse_matmul` takes 17.1 us vs 13.4 us and cancels the removed ops; host
  +5 to +7 us. Rejected; source removed. API failures on the way are kept (`failed_*`, including
  `TT_FATAL: sparsity logical_volume (256) must equal batch_length (257)`, fixed with a 257-entry placeholder).
- Shared tail as a fused addcmul (`micro/shared_addcmul/`, `trials/shared_addcmul_bfp8/`,
  `trials/shared_addcmul_bf16w/`): `scaled = sigmoid(gate) x activated` (one op), then `routed + scaled @ W_down`
  in one `dit_minimal_matmul_addcmul_fused` with a ones row as the multiplier. It replaces the shared down
  matmul, the gate multiply and the add.
  - The op requires the addend in the weight's data format (`TT_FATAL ... minimal_matmul_program_descriptor.cpp:457:
    ternary_a_tile_size == in1_tile_size`). A first try with a BF16 copy of the weight and no program config
    was +23 us (review 2).
  - Adopted variant: BFP8 weight; the routed down `sparse_matmul` and its fused reduce write BFP8, so the addend
    needs no typecast (a typecast in the chain is slower than the control at op level, 14.95 vs 14.38 us).
    Config sweep at op level (grid, K block, N block, subblock): grid 8x2, N block 8, subblock 1x2, K block 4 /
    8 / 16 for tp 4 / 2 / 1.
  - BF16 weight, BF16 addend, HiFi2 (op level as fast as BFP8 on x4, PCC 0.999995): layer +2.3 to +5.4 us on all
    6 rows, control decode lower on all 6 rows. Rejected.

### Decode, full-attention layer (p150x4, `evidence12/profiles/m4_full_attention/decode_step_ops.txt`)

| Ops | Segment | us | Candidate | Result |
|---|---|---:|---|---|
| 0-4 | RoPE cos/sin gather + tilize | 9.4 | combined cos/sin table | -0.5 us, rejected; gather once per step for all 10 layers: Stage 6 |
| 5-7 | input norm (sharded) | 9.5 | | same as linear layer |
| 8 | qkvg 32 x 2048 x 2560 BFP8 LoFi, 40 cores | 21.1 | full geometry sweep; BFP4 | best geometry; BFP4 fails (0.98455) |
| 9-11 | q/gate and k/v slices, `nlp_create_heads` | 5.4 | `nlp_create_qkv_heads_decode` (no transposes) | +11.5 us, rejected (below) |
| 12, 14 | q-norm (4 cores), k-norm (1 core) | 10.4 | | per-head RMSNorm with separate weights; heads fill 4 and 1 cores |
| 13, 16 | indexed RoPE q, k (110 cores each) | 21.8 | `rotary_embedding_llama` prefill mode; decode mode with height-sharded heads; one joint q/k call | +13 us; +11.5 us; +11 to +16 us; all rejected |
| 15, 17-20 | transposes + I2S | 6.5 | removed by the decode-heads candidate | that candidate is +11.5 us |
| 21 | fused KV update | 3.9 | | |
| 22 | query copy to DRAM | 0.9 | query in L1 | `TT_FATAL: Q tensor buffer type must be DRAM when not sharded`; sharded query is part of the decode-heads candidate |
| 23 | SDPA decode (4096 context) | 40.0 | grid 11x10, k_chunk 512 | adopted (-5 us; 131072: 571 -> 228 us) |
| 24-25 | output transpose + concat heads | 5.3 | L1 output (and SDPA output in L1) | no gain (263.1 vs 262.9 us), rejected |
| 26 | sigmoid-gate multiply (110 cores) | 7.3 | L1 (same screen); fast sigmoid | no gain; fast sigmoid changes PCC, rejected |
| 27-30 | attention-out, I2S, AR, add | 20.5 | | same family as the linear layer |
| 31-50 | post norm, router, experts, shared expert, AR | 86.2 | indexed `sparse_matmul` without zero-fill | same as the linear layer (chain 10: 31-51, 89.5 us, including the 3.2 us fill; chain 7: 31-52, 91.7 us) |

- Full decode step: 51 ops, 248.2 us kernel (chain 10: 52 ops, 251.3 us).
- Op-level profiles of the two RoPE layout variants failed: `TT_THROW: Source location hashes are colliding`
  (`profiler.cpp:376`). Two zones in `ttnn/.../sdpa/device/kernels/compute/compute_streaming.hpp`, "Reduce
  max" (line 1691) and "Softmax(Q@KT)@V" (line 1770), share the 16-bit hash 0x992. Failed runs:
  `runs/main/stage05/trials/review2b/profiles/`. The layer-level measurements stand.
- The same collision stopped every full-layer profile after the ttnn rebuild (`micro/masked_chunk/failed_zone_hash/`).
  Fix: the zone label "Reduce max" is now "Reduce_max" (hash 0x6b1a, no collision among the 31 zones of that run;
  `work_log.md`). It is a profiler label only; kernel behavior is unchanged. The change is in
  `evidence12/ttnn_patch.diff`. The chain 12 full-layer profiles complete with it. The two RoPE variant op splits
  were not rerun.

### Prefill (p150x4, 4096 tokens, first device, `evidence12/profiles/m4_*/prefill_raw_summary.json`)

| Segment | Chain 5 (before the prefill knobs) | Candidate | Action | Final |
|---|---|---|---|---|
| Input and post norms | typecast FP32, rms_norm, multiply FP32 gamma, typecast BF16 (x2): 1.30 ms | one `rms_norm`, BF16 gamma | adopted (`prefill_norm`) | 0.25 ms |
| Router | 38 `generalized_moe_gate` calls, untilize + reshape + 2 concats: 2.37 ms | `topk` k 8 + FP32 softmax | adopted | 0.12 ms matmul + 0.07 ms top-k + small ops |
| GDN q/k head split and L2 norm | 2.08 ms | normalize token-major q/k in the chunk kernel | adopted (`gdn_flat_qk`) | in `ChunkGdn` |
| GDN output gate | 0.19 ms | BF16 gate x FP32 norm output | adopted (`gdn_gate_mixed`) | 0.08 ms |
| MoE combine (untilize, restore gather, tilize, multiply, `fast_reduce_nc`) | 3.24 ms | (a) `post_combine_reduce`; (b) fused reduce with scores | (a) Stage 3; (b) slower, L1 overflow at 4096 rows | 3.2 ms |
| Dispatch gather | 0.72 ms | local-only capacity | rejected (host read, `trials/localcap/`) | 0.72 ms |
| Routed experts `moe_fused_swiglu` | 1.73 ms | grid | Stage 3/4 sweeps | 1.72-1.83 ms |
| Attention AR | RS + AG 0.35 ms x 2 | links | 2 links is the maximum (`micro/ccl_links/`) | 0.34-0.41 ms x 2 |
| SDPA prefill (full) | 1.00 ms | q128/k256 HiFi2; LoFi; 8192-row step with long configs at or after position 8192 | LoFi rejected on real weights (below); 8192-row step adopted for long context (Long context) | 1.00 ms |
| MoE pieces for 2049-4095 rows (meshes) | 2 pieces (2048 + 1024) | one 4096 piece when the second piece >= 1024 rows | adopted | see "What changed" |
| Short prompts and tails | Stage 3 single-chip planner costs | measured mesh costs | adopted | see "Prefill planner" |

- p150x4 linear prefill: 424 ops / 16.39 ms kernel (chain 5) -> 106 ops / 10.49 ms (chain 10: 10.50 ms); full
  416 / 14.16 -> 111 / 10.71 ms (chain 10: 10.73 ms).

### CCL and residual families (decode)

| Family | Residual before / after | Collective | Measured | Result |
|---|---|---|---|---|
| Local matmul + RS + AG (Stage 4) | replicated interleaved | 2 ops | 16.8 us per all-reduce (x4) | replaced |
| Local matmul + direct all-reduce, persistent L1 buffer | replicated interleaved | 1 op | 12.0 us (x4), 9.5 us (x2) micro | adopted |
| Same + width-sharded L1 residual (sharded adds and norms) | replicated width-sharded L1 | 1 op | -4 to -7 us per layer | adopted |
| RS + fractured residual + distributed norm, no output gather (lower bound) | fractured | RS + AG(stats) | x4 47.9 us, x2 61.9 us vs adopted 26.6 / 27.1 us per boundary (`micro/residual_family/`) | rejected by bound |
| Same + AG of the normed slice | fractured | RS + 2 AG | x4 60.9 us, x2 70.8 us | rejected |
| Fused matmul + RS (`matmul_reduce_scatter_async`) + distributed norm | fractured | fused | x4 61.4 us; x2 hangs (`micro/residual_family/triage_x2_mmrs/`) | rejected |
| Fused AG + next in-projection (`all_gather_matmul_async`) | fractured | fused | "bad optional access" without a program config; family bounded by the no-gather row | not pursued |

- Async CCL: decode `ttnn.experimental.all_reduce_async` with ping-pong global semaphores and a persistent
  buffer; prefill `all_reduce_async` with construction-time semaphores.
- Decode all-reduce flags (`runs/main/stage05/micro/ccl_flags/`, traced op micro): the direct op exposes
  `use_noc1_only` and `use_optimal_ccl_for_llama` and no chunks-per-sync, workers-per-link or
  buffers-per-channel argument. On the 8x2 shard grid: default 9.94 us (x4) / 6.78 us (x2); `use_noc1_only`
  11.74 / 7.54; `use_optimal_ccl_for_llama` (worker placement only) 9.89 / 6.78. Neither adopted. A row-wise
  16-core grid is 0.25-0.43 us faster per call on x4 but changes the residual and norm shard layout (not
  measured in the layer).

### Inter-layer residual contract (decode)

- tp2 and tp4: BF16, TILE, width-sharded L1, shard [32, 128], cores (0,0)-(7,1), row-major. tp1: interleaved.
- The two all-reduces produce it; both residual adds and both decode norms consume it without reshard.
- Row 0 holds the token; rows 1-31 are tile padding. `evidence12/probes/padding_state_x*`: zero, 1e4, Inf, NaN
  and random padding leave layer output row 0 and both GDN states bitwise unchanged (36 of 36 cases). The probe
  JSON files are byte-identical to chain 10.
- Coverage gap: the padding probe checks the linear-layer GDN states and the layer output. It does not read the
  full-layer KV cache after padding injection. The KV update writes one row per step (`paged_fused_update_cache`
  with the position tensor), and the full-layer output row 0 is bitwise unchanged in the same probe.
- `decode_output_residual_contract.matches` true on the p150x2 and p150x4 headline rows.

## Prefill planner (`runs/main/stage05/micro/plan_cost/`)

| Case | x4 old / new | x2 old / new | p150 old / new |
|---|---|---|---|
| full, 3 tokens | 4.64 / 2.58 ms | 4.32 / 2.61 | 3.73 / 2.94 |
| full, 2 tokens | 3.29 / 2.67 | 3.02 / 2.55 | 2.56 (unchanged, steps win) |
| linear, 257 tokens | 5.64 / 3.99 | 6.92 / 4.81 | 7.54 / 6.88 |
| linear, 1025 tokens | 6.78 / 5.03 | 8.55 / 6.64 | 12.30 / 9.92 |
| linear, 2049 tokens | 8.68 / 6.84 | 11.26 / 9.27 | 17.41 / 13.65 |
| linear, 4100 tokens | 12.22 / 11.82 | 16.09 / 15.63 | 23.08 / 22.13 |
| linear, 4353 tokens | 13.36 / 13.40 | 18.42 / 18.33 | 27.28 / 27.29 |

- The Stage 3 cost table chose eager decode steps for 3-token full-layer prompts (1.45 ms per step on meshes)
  and a larger masked chunk for linear lengths of a bucket sum plus 1.

Mid lengths on meshes (`micro/plan_cost/mid_*`, `cap_*`, 1 compile + 6 warm runs, median, ms). Each extra chunk
costs about 2.5 ms on a mesh; the Stage 3 table assumes 0.5-0.8 ms, so it chose a ladder of chunks (for example
1300 = 256 + 1024 + 32 masked). Mesh costs `plan_chunk_cost` 2.5 and `plan_row_cost` 0.002 choose one masked
chunk (1300 -> one 2048 chunk, 2600 -> one 4096 chunk).

| Length | x4 linear old / new | x4 full | x2 linear | x2 full | p150 linear (not adopted) |
|---|---|---|---|---|---|
| 1100 | 7.72 / 6.68 | 7.37 / 6.60 | 9.93 / 8.82 | 9.59 / 8.33 | 12.17 / 12.87 |
| 1300 | 8.25 / 6.47 | 6.64 / 6.64 | 9.20 / 8.97 | 8.40 / 8.40 | 12.83 / 13.01 |
| 1536 | 9.46 / 6.60 | 6.77 / 6.77 | 11.84 / 9.03 | 8.54 / 8.56 | 13.22 / 13.24 |
| 2400 | 10.87 / 10.96 | 10.78 / 10.49 | 14.50 / 14.27 | 13.54 / 13.53 | 18.29 / 20.25 |
| 2600 | 11.77 / 10.86 | 11.72 / 11.02 | 14.33 / 14.49 | 13.73 / 14.41 | 18.98 / 20.41 |
| 3000 | 10.98 / 10.95 | 11.56 / 11.09 | 14.00 / 14.50 | 13.83 / 14.73 | 19.28 / 20.72 |
| 3300 | 11.79 / 11.00 | 11.16 / 11.26 | 14.65 / 14.72 | 14.87 / 14.78 | 20.23 / 20.88 |

- Full table (1000-4096, both layer kinds, all profiles): `work_log.md`.
- p150x4: gains up to 2.9 ms, losses at most 0.1 ms. Adopted.
- p150x2: gains up to 2.8 ms at 1100-1536; losses of 0.2 to 0.9 ms at 2600-3000 (one masked 4096 chunk is slower
  than the old ladder, for example 2600 = 256 + 2048 + 256 + 64 masked). Net -3.3 ms over the 20 measured rows. A cap at 2048-row masked chunks fixed 2400-3000
  but made 3300-4095 run as 2 x 2048 (+0.35 ms at 3300); rejected. Adopted with the 2600-3000 loss disclosed.
  Chain 10 repeats the loss (2600 / 3000: linear +0.34 / +0.48 ms, full +0.78 / +0.77 ms vs chain 8); chain 12
  also (linear +0.30 / +0.52 ms, full +0.79 / +0.86 ms vs chain 8).
- p150: slower with mesh costs (up to +2.0 ms). p150 keeps the Stage 3 table.
- CPU grid search over chunk and row costs (`plan_sweep.py` data): no pair that selects only measured plans beats
  the adopted pair on x4 or x2.

Scope of the cost change:
- CPU sweep (`plan_sweep.py`): changed classes are full 2-3 tokens from the start, linear 2 tokens (meshes),
  linear 4-token tails, linear bucket-sum-plus-1 lengths and (meshes) mid lengths that ran 2 or more chunks.
- Positions and reuse acceptance rows also change plan (257 and 513-token slots; reuse segments 353 and 640
  tokens after a cached prefix). Chain 12 reruns them on the final source (below).
- Chain 12 runs the profiles, long context, inventory and placement on the final source, so no result depends on
  an older planner. The only planner change after chain 10 is the 8192-row full-attention step at or after
  position 8192 (`prefill_long_chunk`); CPU plan invariants allow it only unmasked (`test_long_chunk_merge`,
  `test_long_chunk_moe_pieces`, `work_log.md`). (`micro/plan_cost/evidence9_scope.py` also shows the planner change leaves the plans of those
  runs unchanged: 97 spans per profile and layer kind, 0 changed.)

## Precision (real weights, recorded activations)

| Group | Selected | Tried | Result |
|---|---|---|---|
| GDN in-projection | BFP8 HiFi2 | BF16 HiFi4/HiFi2/LoFi, BFP8 LoFi, BFP4 LoFi/HiFi2 | BFP8 LoFi in the layer: no gain (245.7 vs 246.4 us), HF decode 0.99971 -> 0.99944; BFP4 fails (0.886) |
| qkvg (full) | BFP8 LoFi | BFP4 LoFi | fails (0.98455) |
| attention-out (full) | BFP8 LoFi [8,4] k8 | BFP4 LoFi k16 | failed at 2600 tokens step 53 (0.99342); reverted |
| attention-out (linear) | BFP8 HiFi2 | BFP4 LoFi | fails (0.99014) |
| shared down | BFP8 (in the fused addcmul) | BFP4; BF16 copy in the fused addcmul | BFP4: no gain, HF decode 0.99861 -> 0.99807; BF16 fused: +2.3 to +5.4 us |
| expert down | BFP8 | BFP4 | fails (full layer) |
| expert down output and fused reduce (decode) | BFP8 (the addcmul addend must match the BFP8 weight) | BF16 | BFP8 with the fused addcmul: -0.1 to -2.7 us; recorded inputs: HF decode within 0.00002, control decode min higher on all 6 rows; random-activation rows: decode min 0.0004-0.0010 lower, all pass (Non-aligned lengths, batch, acceptance) |
| expert gate/up | BFP4, FP32 accumulation | | kernel PCC 0.99999 vs 0.99958 (`sparse_matmul` BF16 accumulation) |
| router + shared gate | BF16 HiFi2 | HiFi4, LoFi, BFP8 weights HiFi2 | HiFi2 adopted (-0.5 to -2.7 us, HF decode within 0.00003); LoFi -1.5 to -2.3 us, passes every layer gate (control decode 0.99957 -> 0.99918 full, 0.99961 -> 0.99944 linear) but changes expert choices; not adopted here, handed to the datatype sweep with full-model accuracy; BFP8 no gain |
| GDN recurrence | FP32 | | state error 5e-8 to 9e-8 |
| SDPA decode | HiFi | | |
| KV cache (full layers) | BFP8 (`bfloat8_b`) paged, per-device local KV head `[pages, 1, 32, 256]` (`kv_dtype` "bfp8" in the headline `optimization` dict; `../context_contract.json`) | BF16 in the fused decoder, an earlier stage (`fused_decoder_validation.kv_dtype`) | 1088 bytes per token per full layer, BF16 was 2048 (`../context_contract.json`, `optimized_decoder_validation`); this stage changes neither dtype nor layout |
| SDPA prefill, steps of 8192 rows at or after position 8192 | HiFi2, x4 q64/k256, x2 and p150 q128/k512 (`sdpa_prefill_long`) | q128/k256, q64/k256, q128/k512 per profile (`micro/long_chunk/`); q256/q512 arms exceed L1 (`micro/sdpa_prefill/`) | `evidence12/long_chunk/` (random input, 16384 and 20000 tokens): full prefill PCC 0.99842-0.99846, decode min 0.99670-0.99762; the same policy without the long chunk gave equal values within 1e-5 (`micro/long_chunk/accuracy/`) |
| SDPA prefill | HiFi2 q128/k256 | LoFi (k256, k512) | real weights at 262143 tokens (`trials/review2c/context_lofi/`): prefill PCC 0.99853 -> 0.99534, 0.0003 above the 0.995 gate. Not adopted here: the stacked-layer effect is not measured in a single layer. Handed to the datatype sweep with full-model long-context accuracy (about -13% full-layer long-context prefill) |
| Decode norms (linear) | 16-core sharded, BF16 input, FP32 accumulation | 1-core FP32 | -30 us per layer |
| Prefill norms | BF16 input and gamma, FP32 accumulation | FP32 input; FP32 gamma | same speed; HF PCC unchanged |
| Prefill router softmax | FP32 over the top 8 | | HF PCC unchanged |

- tt-perf-report rows confirm the dtypes (x4 decode): GDN in-projection `HiFi2 BF16 x BFP8`, qkvg and
  attention-out (full) `LoFi BF16 x BFP8`, attention-out (linear) `HiFi2 BF16 x BFP8`, router `HiFi2 BF16 x
  BF16`, expert down `sparse_matmul` `LoFi BF16 x BFP8 => BFP8`, fused reduce `BFP8, BF16 => BFP8`, fused
  addcmul (`MinimalMatmulDeviceOperation 32 x 128 x 2048`, 16 cores) `LoFi BF16 x BFP8 => BF16`; prefill experts
  `BFLOAT4_B` (`evidence12/profiles/`).

### Router ties in the accuracy gate

- The HF reference marks every expert whose BF16 router logit is within one BF16 ULP of the 8th-ranked logit.
  Candidates are all top-8 sets that keep the experts above the window. The TT output is compared to the nearest
  candidate; strict (HF top-8 only) PCC is always reported.
- TT router capture (`evidence12/probes/router_l0_x4.json`, `router_l3_2600_x4.json`, HiFi2 router; byte-identical
  to chain 10): 17 of 56 and 23 of 60 steps select a different set. Every flip but one is within one BF16 ULP of the HF boundary.
- One flip is 2 ULP (layer 3, 2600 tokens, step 6: HF expert 120 at -4.5 vs expert 163 at -4.5625; TT logits
  -4.5 for both). That step's strict output PCC is 0.99898 (chain 7: 0.99895); it passes without the tie gate. The same flip is
  present in chain 6 (HiFi4 router).
- Limit: a router bias smaller than one BF16 ULP is not detectable by this gate.

### Prefill per-token PCC tail (`runs/main/stage05/micro/prefill_token_tail/`)

- Linear headline rows have 4-6 tokens below 0.995 (min 0.9909-0.9911, token 3515); full rows 1 token (0.9914,
  token 911). Aggregate PCC is unchanged.
- Every such token has an exact BF16 tie between the 8th and 9th HF router logit (`router_gap.json`).
- `tie_swap.json`: HF output with the tied alternative expert set vs HF output gives 0.98705-0.99704 (token
  3515), 0.99375-0.99395 (token 1290), 0.99011-0.99869 (token 911). Every TT value sits within or 0.0002-0.0007
  below its token's swap range. The tail is router tie choice, not a new error source.

## Non-aligned lengths, batch, acceptance (`evidence12/`, final source)

- Lengths (`len_*`): 3, 100, 1000, 1300, 2176, 2176 (128 + 2048 split), 2400, 2600, 3000, 3300, 3968, 4095, 4097,
  2, 257, 1025, 1100, 1536, 2049, 4100, 4103 on all profiles and both layer kinds, 4 traced decode steps: 126
  rows pass, all audited, prefill repeat bitwise on 126 of 126. Min HF prefill 0.99931, min HF decode 0.99818
  (full 1100 tokens, all three profiles within 1e-5), min control prefill 0.99983, min control decode 0.99975
  (control tensors exist for the Stage 4 lengths). The 1100 full rows have no control tensor; HF decode there is
  above the gate with the same value on all profiles.
- The lengths cover each changed planner class: 2-3 tokens, bucket sum plus 1 (257, 1025, 2049), one-step tails
  (4100), masked tails (4097, 4103), mid lengths that now run one masked chunk (1100-3300), and their neighbors.
- 128-step rows at 1300, 2600, 4095 (`long128_*`): 18 rows; min HF prefill 0.9995, min HF decode 0.99854.
- B32 (129 tokens, 8 traced steps, random activations, `b32_*`): 6 rows pass; HF decode min 0.99536 (full; chain
  9: 0.99626), control prefill min 0.99978, control decode min 0.99979. B > 1 decode is a per-row loop; batched
  decode is full-model work (below).
- Acceptance per profile (`accept_*`): reuse (full, full with 128-token pages, linear; 18 cases each),
  positions, context, watcher (4 cases); device pytest 12 passed (`accept_pytest/device.log`); CPU tests 1213
  passed (`cpu_pytest.xml`; chain 10: 1207).
- Positions and reuse exercise the changed linear plans on the final source. Plans (CPU, final planner):
  257-token slots 256 + 1 step (was 128 + masked 256); 513-token slots one masked 1024 chunk on meshes, 256 + 256
  + 1 step on p150; reuse segment 160 + 353 ends 256 + 1 step; segment 513 + 640 ends with a masked 1024 chunk on
  meshes.
- Positions (`accept_positions_m*`, B32, slots 0-31, mixed lengths, random activations): warm and replay
  identical; decode min 0.99556-0.99593 (full), 0.99745-0.99762 (linear).
- Real-input positions (`real_inputs_m*`): tie-aware min 0.99949-0.99956 (L0), 0.99663-0.99664 (L3); strict
  0.99031-0.99042 (L0, router near tie), 0.99660-0.99661 (L3). Real-input reuse: min prefill 0.99793, min decode
  0.99674. Synthetic reuse: min prefill 0.99832, min decode 0.99624. All 270 reuse cases pass the 0.995 gate.
- Native context (`accept_context_m*`): 262143 tokens prefill + decode and 262144 prefill on every profile, both
  layer kinds; prefill PCC 0.99816-0.99949 (chain 10: 0.99822-0.99949), decode 0.99785-0.99983
  (`../context_contract.json`). The full-layer rows now run 8192-row attention steps from position 8192.
- Effect of the BFP8 routed sum (chain 9 is the control: same chain, source before the addcmul change). Prefill
  values are unchanged. Decode minimums on random-activation probes are lower; recorded-input probes change by
  -0.00006 to +0.00006. Chain 12 gives the chain 10 value in every cell of this table:

| Probe (decode min) | Input | Chain 9 | Chain 10 and chain 12 |
|---|---|---|---|
| positions full, x4 / x2 / p150 | random | 0.99658 / 0.99641 / 0.99640 | 0.99593 / 0.99563 / 0.99556 |
| positions linear, x4 / x2 / p150 | random | 0.99796 / 0.99797 / 0.99798 | 0.99756 / 0.99762 / 0.99745 |
| B32 full, x4 / x2 / p150 | random | 0.99630 / 0.99633 / 0.99626 | 0.99541 / 0.99550 / 0.99536 |
| watcher small_pool_65_b1, x4 / x2 / p150 | random | 0.99657 / 0.99680 / 0.99670 | 0.99553 / 0.99591 / 0.99559 |
| context full 262143, x4 / x2 / p150 | random | 0.99829 / 0.99829 / 0.99830 | 0.99786 / 0.99790 / 0.99785 |
| synthetic reuse, all | random | 0.99695 | 0.99624 |
| real-input positions L0 / L3, x4 | recorded | 0.99947 / 0.99659 | 0.99949 / 0.99663 |
| real-input reuse, all | recorded | 0.99677 | 0.99674 |

- Op-level attribution (`micro/shared_addcmul/`, random routed input): the BFP8 addend gives op PCC 0.99984 vs
  0.99996 for the separate-op control; a BF16 weight and addend give 0.999995 but cost +2.3 to +5.4 us in the
  layer. The lowest value, 0.99536, is 0.0044 above the 0.995 gate. Full-model accuracy for this choice is
  checked by the datatype sweep (Work owned by later stages).

## Continuation prefill with the 8192-row step (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/micro/review5/`)

`run_layer --trace --audit --sparse-reference`, full layer 3, 26000 tokens, 4 decode steps, two prefill calls.
Control: the same run with `prefill_long_chunk` 0 (`control_policy.json`, current full-layer default otherwise).
CPU plans for each call: `plans.txt`. Split 8320 gives 8192-row steps at 8320 and 16512. Neither start is a
multiple of `k_chunk_size` 512 (p150, p150x2) or 256 (p150x4). Split 8192 is the aligned control.

| Profile | Split 8320, long chunk: prefill / decode PCC | Split 8320, control | Split 8192, long chunk |
|---|---|---|---|
| p150x4 | 0.998593 / 0.998254 | 0.998593 / 0.998254 | 0.998594 / 0.998254 |
| p150x2 | 0.998586 / 0.998267 | 0.998598 / 0.998267 | 0.998586 / 0.998267 |
| p150 | 0.998589 / 0.998191 | 0.998592 / 0.998191 | 0.998589 / 0.998191 |

- PCC (Pearson correlation coefficient) is against the HF reference with the sparse expert reference.
- All 9 runs rc=0 with the runtime audit; source and `_ttnncpp.so` hashes match before and after.
- Largest prefill PCC difference to the control: 0.000012 (p150x2). Decode PCC is identical.

## Long context (`evidence12/long_context/`, timing only)

Real layer weights, random hidden input, 8192-token external chunks (`--chunk 8192`), page table aligned to 16
blocks. Full layers run each 8192-token chunk at or after position 8192 as one 8192-row attention step
(`prefill_long_chunk`); the MoE still runs in 4096-row pieces. Linear layers run two 4096-row groups per chunk.
Decode traced at the final position. All outputs finite.

| Profile | Full: chunks 2-16 (tokens 8192-131071) / chunk 2 / chunk 3 / last chunk | Full decode at 131072 | Linear: chunks 2-16 / last chunk | Linear decode |
|---|---|---:|---|---:|
| p150x4 | 1224 ms / 73.8 / 37.8 / 126.4 ms | 458.1 us | 326 ms / 21.7 ms | 214.3 us |
| p150x2 | 1784 ms / 74.6 / 52.7 / 193.7 ms | 480.0 us | 435 ms / 29.0 ms | 252.5 us |
| p150 | 2796 ms / 87.2 / 79.2 / 308.1 ms | 719.6 us | 623 ms / 41.5 ms | 340.4 us |

- Chunk 1 (tokens 0-8191) includes the program build. Chunk 2 is the first 8192-row attention step and is slower
  than chunk 3 although its KV prefix is shorter (x4 73.8 vs 37.8 ms). A two-pass control measures this as
  first-use program build (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/micro/long_cold/`, x4 full layer,
  32768 tokens, 8192 chunks): warm pass chunks take 24.1 / 30.8 / 37.8 / 44.5 ms, so the chunk at 8192 is 30.8 ms
  when warm. The serving warmup owns this build cost. Chunks 2-16 here are one cold pass.
- Same token range in both chains (chunks that start at or after 16384, 114688 tokens), chain 10 (4096-token
  chunks) -> chain 12: full x4 1207 -> 1150 ms (-4.7%), x2 2219 -> 1709 ms (-23.0%), p150 3266 -> 2708 ms
  (-17.1%); linear within +1.0% on every profile. Last 8192 tokens (chain 10: sum of the last two 4096 chunks):
  full x4 133.3 -> 126.4 ms, x2 253.4 -> 193.7 ms, p150 376.0 -> 308.1 ms.
- Chain 10 table (4096-token chunks, chunks 2-32, tokens 4096-131071): full 1252 / 2286 / 3361 ms, last chunk 67.6
  / 128.5 / 190.9 ms, decode 460.8 / 483.2 / 721.7 us; linear 334 / 446 / 641 ms, decode 222.9 / 262.7 / 351.9 us
  (x4 / x2 / p150).
- Linear decode at 131072 is 3.3% to 3.9% faster than chain 10 (gate-writer conv shift and no zero-fill); full
  decode is within 0.7%.
- p150x4 full layer to 262112 tokens (32 chunks, last chunk 8160 tokens): 4188 ms (chunks 2-32), last chunk 256.1
  ms, decode 664.5 us. Chain 10 (64 chunks of 4096): 4386 ms (chunks 2-64), last chunk 137.6 ms (4064 tokens),
  decode 667.1 us.
- Layer-sum estimate at 131072 tokens (not a model measurement), p150x4: TTFT about 22.0 s (10 x 1.224 s + 30 x
  0.326 s, plus chunk 1); decode about 11.0 ms per token (10 x 458.1 + 30 x 214.3 us). Chain 10: 22.5 s, 11.3 ms.
- Full-layer chunk time grows with the KV length (chunked SDPA). This dominates long-context TTFT. SDPA prefill
  LoFi would cut it by about 13%; it is handed to the datatype sweep (Work owned by later stages).

8192-row attention step (`runs/main/stage05/micro/long_chunk/`, `micro/sdpa_prefill/`, review 4, `work_log.md`).
Layer timing at 131072 tokens, 8192-token external chunks, chunks after the first, ms per 4096 tokens:

| Profile | 4096 control | 8192 q128 k256 | 8192 q64 k256 | 8192 q128 k512 |
|---|---:|---:|---:|---:|
| p150x4 | 40.5 | 41.5 | 39.4 | 40.5 |
| p150x2 | 73.7 | 59.8 | 64.9 | 58.5 |
| p150 | 108.4 | 95.1 | 115.6 | 92.4 |

- Adopted: full layers `prefill_long_chunk` 8192, `prefill_long_from` 8192; `sdpa_prefill_long` q64/k256 on
  p150x4 (mesh policy) and q128/k512 on p150x2 and p150. Linear layers unchanged.
- The planner merges two consecutive unmasked 4096-row full-attention chunks into one 8192-row attention step when
  the step starts at or after `prefill_long_from`. `_moe_pieces` splits rows above the largest bucket into 4096-row
  pieces, so the UINT16 route maps stay 4096-row. Plans with an 8192 step use the planned (device-index) path.
- SDPA op micro, one device, 131072-token paged BFP8 KV: q256 k512, q512 k256 and q512 k512 exceed L1 (`TT_THROW:
  Statically allocated circular buffers ... beyond max L1 size of 1572864 B`); q256 k256 is about 2x slower than
  q128 k256 (113.2 vs 57.2 ms at 126976 on x4).
- Accuracy: `evidence12/long_chunk/` (`run_layer --sparse-reference`, random input, 16384 and 20000 tokens, 4 traced
  decode steps, audit on, all profiles): full prefill PCC 0.99842-0.99846, decode min 0.99670-0.99762; linear
  prefill 0.99938-0.99947, decode min 0.99834-0.99971. `micro/long_chunk/accuracy/`: the same policy with
  `prefill_long_chunk` 0 gives equal values within 1e-5.
- Failed attempt, kept: `micro/long_chunk/failed_no_merge_tiled_path/` (aligned inputs took the tiled path and
  re-split the 8192 step; its q64/k512 arms failed with a `TypeError` in the decode `SDPAProgramConfig`).

Prefill chunk size at 131072 tokens before the 8192-row step (`runs/main/stage05/micro/chunk_size/`, p150x4, same
harness, chain 10 source):

| External chunk | Full: chunks after the first | Full decode | Linear: chunks after the first |
|---|---|---:|---|
| 2048 | 2270 ms (72.1 ms per 4096 tokens) | 465.1 us | 448 ms (14.2 per 4096) |
| 4096 (chain 10 default) | 1253 ms (40.4 per 4096) | 465.7 us | 334 ms (10.8 per 4096) |
| 8192 (two 4096 groups) | 1234 ms over 122880 tokens (41.1 per 4096) | 464.3 us | 326 ms (10.9 per 4096) |

- An 8192-row planner bucket fails at construction: `ValueError: Internal local-dispatch chunk must fit UINT16
  route maps` (8192 rows x 8 routes = 65536 > 65535). This is an exact op-contract limit of the dispatch route
  map. A runtime override fails earlier (`KeyError: 8192`, offsets are built per bucket, kept in
  `failed_runtime_bucket/`). The adopted 8192-row step does not use an 8192 MoE bucket (above).
- Correction (review 4, `work_log.md`): an earlier version of this section bounded the 126976-token chunk by a
  71 MB KV read (0.14 ms) and 4.4e12 SDPA FLOPs (14.5 ms at HiFi2 peak). That bound was wrong. Chunked SDPA streams
  the KV prefix once per query chunk and head, not once per call, and the FLOP figure counted QK and PV twice. The
  per-call cost is not FLOP-bound at HiFi2 peak. A corrected KV-traffic and FLOP bound for the adopted chunking
  (8192-row steps; q64/k256 on p150x4, q128/k512 on p150x2 and p150) is not derived here. Measured SDPA op time:
  57.2 ms per 4096-row call at 126976 on x4 (`micro/sdpa_prefill/`).

## Trace, determinism, audit, watcher

- Traced decode on every row: one capture, warm and replay outputs identical, program-cache misses forbidden
  during capture.
- Trace allocation tracking (`TT_METAL_TRACE_ALLOC_TRACKING=1`, program-cache allocations included) on the
  positions, reuse, watcher and device pytest runs; only the final trace output is acknowledged as corruptible.
- Failed attempt, kept (`evidence11/`, chain 11): with `conv_shift="gate"` the layer kept the decode conv row in
  `self._conv_current` after capture, so the tensor stayed referenced across replays. The tracker reported 1
  buffer alive before replay on the linear reuse runs (`accept_reuse_m{4,2,1}` linear, `real_reuse_0_m{4,2,1}`).
  Fix (`tt/optimized_decoder.py`): `_decode_kernel` takes and clears the reference; `_decode_heads` raises if a
  gate-shift row is pending. Fix check (`runs/main/stage05/micro/conv_ref_fix/`, `run_reuse` layer 0, tracking
  on): exit code 0 on p150x4, p150x2 and p150, 18 cases each, replicas identical; `failed_wrong_p150_env/` keeps a run
  with the wrong device variable. Chain 12 passes every reuse run.
- Runtime fallback audit (`--audit`, device-only guard): clean on all 203 chain 12 harness rows (headline 18,
  aligned SDPA 9, lengths 126, 128-step 18, B32 6, long watcher 12, placement 2, long-chunk 12) and on every
  acceptance run.
- Watcher (`TT_METAL_WATCHER=10`), chain 12, final source: 24 logs, no error, assert, hang, overflow or tripped
  line in any of them.
  - 12 acceptance logs (`accept_watcher_m*`, 4 cases per profile): linear and full layers at 129 tokens with 8
    traced steps, and full layers at 257 tokens batch 2 and 65 tokens batch 1 with 31 traced steps (small page
    pools). The chain sets `TT_METAL_WATCHER_NOINLINE=1` for the p150x4 acceptance runs.
  - 12 long runs (`watcher_long/`, `TT_METAL_WATCHER_NOINLINE=1`): every profile, both layer kinds, 4096 and 1300
    tokens, 8 traced steps; on meshes 1300 runs one masked 2048 chunk.
  - Paths exercised: the linear runs use `conv_shift="gate"`, and `gate_writer_shift.cpp` is in the watcher
    kernel list of all 12 linear logs (`kernel_names.txt`). Every run uses the indexed `sparse_matmul`
    (`sparse_indexed` true); the no-fill change is host-side output allocation, so the watcher does not observe
    it directly.
  - 8192-row attention step and long SDPA configs: 3 watcher runs, one per profile
    (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/micro/review5/m{4,2,1}_split8320_watcher2.*`,
    `watcher_logs_m*/`, `TT_METAL_WATCHER_NOINLINE=1`), full layer 3, 26000 tokens in two prefill calls split at
    8320. The second call runs 8192-row steps at 8320 and 16512 and a masked 2048 tail. Watcher logs have no
    error, assert or hang lines. These are post-chain runs on the chain 12 source (`source_*.sha256` match).
  - Failed attempt kept: `review5/failed_watcher_no_noinline/` (p150x4 without `TT_METAL_WATCHER_NOINLINE=1`:
    Ethernet kernel config buffer overflow at program build, the same setup limit the chain works around).
- Prefill repeat (first vs warm prefill, bitwise): 191 of 191 chain 12 rows that run a warm prefill (the 12
  long-chunk rows do not).
- Decode replays repeat bitwise on every row.

### MoE prefill nondeterminism (`evidence7/nondeterminism/`, `evidence10/nondeterminism/`, `evidence12/nondeterminism/`)

| Probe | Chain 5 (gate-loop router) | Chain 7 | Chain 10 | Chain 12 (final) |
|---|---|---|---|---|
| Standard probe, B32 full, 3 runs, p150x4 | 39-61 differing captures per run (`moe_fused_swiglu`, combine) | 0 in 3 of 3 | 0 in 3 of 3 | 0 in 3 of 3 |
| Same, p150x2 / p150 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| E3/E1: same inputs again, cloned inputs (x2, x4) | identical | identical | identical | identical (1536 and 768 per-device comparisons, x4 and x2) |
| Standalone op repeat loops (`moe_repeat`) | 0 of 20 differ | 0 of 20 differ | 0 of 20 differ | 0 of 20 differ (9 configurations) |

- Chain 5 AutoDebug (`evidence3/nondeterminism/followup/AUTODEBUG.md`): each p150x4 MoE event perturbed one
  Tensix core; refuted CB-region L1, semaphores, NoC timing, inputs and addresses; open: unprogrammed per-core
  state left by an earlier kernel.
- Chains 6 and 7 remove the 38 `generalized_moe_gate` calls per prefill (top-k router). The drift is not
  observed after that change (chains 6, 7, 9 and 10: 620 harness rows, 12 probe runs; chain 12: 191 prefill-repeat
  rows and the probes above). The mechanism is not isolated; this is a measured change,
  not a root cause. `generalized_moe_gate` still runs in decode; a decode-then-prefill bitwise repeat on p150x4
  is covered only by the 4097-token rows (one gate call before a repeated prefill).
- Serving stages must still run a repeatability check on identical requests.

## Profile (`evidence12/profiles/m{4,2,1}_*`)

- Method: `tests/run_profiles.py --tracy-arg=--disable-device-data-push-to-tracy`, post-processing
  `runs/main/stage05/process_profiles12.sh`. Per profile: `decode_perf_report.{txt.gz,csv}`,
  `prefill_perf_report.{txt.gz,csv}`, stacked CSV, `decode_replay_summary.json`, `decode_step_ops.txt`,
  `prefill_raw_summary.json`.

Decode kernel time per step, first device, one replay (`evidence12/profiles/decode_family_table.md`, `family_table.py`):

| Op family | x4 linear | x4 full | x2 linear | x2 full | p150 linear | p150 full |
|---|---:|---:|---:|---:|---:|---:|
| All ops (ops per step) | 197.9 (33) | 248.2 (51) | 237.2 (33) | 274.7 (51) | 328.7 (30) | 352.6 (48) |
| Matmul | 47.1 | 44.8 | 66.1 | 60.2 | 113.6 | 97.7 |
| Generic ops (GDN, expert kernels) | 57.0 | 16.6 | 73.1 | 28.0 | 106.9 | 48.1 |
| Layout (reshape, typecast, slice, concat, copy, transpose, tilize, untilize, permute) | 12.6 | 29.7 | 15.9 | 31.8 | 14.1 | 35.9 |
| Eltwise binary / unary | 6.8 | 14.1 | 3.9 | 12.1 | 16.6 | 23.4 |
| SDPA decode |  | 40.0 |  | 39.2 |  | 47.0 |
| Norms | 11.8 | 22.9 | 12.4 | 23.7 | 12.3 | 24.2 |
| CCL | 17.1 | 17.4 | 13.5 | 13.0 |  |  |
| Expert down sparse_matmul | 12.0 | 12.0 | 19.8 | 19.9 | 28.0 | 27.7 |
| I2S / S2I | 10.1 | 12.0 | 9.4 | 11.4 | 8.2 | 10.6 |
| RoPE |  | 24.0 |  | 21.8 |  | 22.8 |
| GDN conv | 14.7 |  | 16.6 |  | 22.6 |  |
| KV update, head split and concat |  | 6.0 |  | 7.1 |  | 8.9 |
| MoE gate, top-k, fused reduce | 8.6 | 8.6 | 6.5 | 6.5 | 6.3 | 6.3 |

- Against `evidence10/profiles/` (chain 10): x4 linear 208.7 -> 197.9 us (38 -> 33 ops), x4 full 251.3 -> 248.2
  (52 -> 51), x2 linear 247.3 -> 237.2 (38 -> 33), x2 full 279.0 -> 274.7 (52 -> 51), p150 linear 341.6 -> 328.7
  (35 -> 30), p150 full 358.3 -> 352.6 (49 -> 48). Linear layers lose the 4 conv-shift ops and the expert-down
  fill; full layers lose the fill. Expert-down `sparse_matmul` 13.0 -> 12.0 us on x4, 21.0 -> 19.8 on x2, 30.0 ->
  28.0 on p150.
- Against `evidence7/profiles/` (before the addcmul change): x4 linear 210.1 -> 197.9 us (39 -> 33 ops), x4 full
  253.6 -> 248.2 (53 -> 51), p150 linear 350.4 -> 328.7 (38 -> 30), p150 full 366.3 -> 352.6 (52 -> 48).

SLOW rows (p150x4 decode, chain 12 tt-perf-report, median of 128 replays):

| Op | Shape | Time | DRAM % | Status |
|---|---|---:|---:|---|
| GDN in-projection | 32 x 2048 x 3136 BFP8 HiFi2, 20 cores | 23.7 us | 53 | full grid sweep: best op geometry 20.4 us (11x3) gives no layer gain; LoFi no gain |
| qkvg | 32 x 2048 x 2560 BFP8 LoFi, 40 cores | 21.1 us | 49 | best in the full sweep |
| attention-out | 32 x 1024 x 2048 BFP8, 32 cores | 9.5-9.6 us | 43 | best measured (Stage 4 sweep, BFP8 sweep) |
| router + shared gate | 32 x 2048 x 288 BF16 HiFi2, 9 cores | 6.9 us | 33 | HiFi2 adopted; LoFi and BFP8 screened |
| fused addcmul (shared down, gate scale, add) | 32 x 128 x 2048 BFP8 LoFi, 16 cores (8x2) | 7.0-7.4 us | 7 | op-level sweep of grid, K block, N block and subblock; replaces 3 ops of 11.9 us; M is 1 row padded to 32 |

- Advice "use HiFi2/HiFi4 with BF16 activations" on LoFi BFP8 rows: those rows pass the gates at LoFi; higher
  fidelity is slower. Not adopted. "Output subblock 1x1": 9-tile and 98-tile outputs on 9 and 20 cores give one
  tile per core per block; covered by the geometry sweeps.
- Prefill p150x4 (raw rows, first device): linear 10.49 ms kernel / 10.73 ms span, full 10.71 / 11.41 ms (chain
  10: 10.50 / 10.55, 10.73 / 11.48). Largest items: MoE combine about 3.2 ms (embeddings, untilize, tilize,
  binary, `fast_reduce_nc`), `moe_fused_swiglu` 1.73-1.82 ms, SDPA 1.00 ms (full).

### Roofline (`evidence12/roofline.json`, `runs/main/stage05/roofline.py`)

Compulsory bytes and useful FLOPs per device. Peaks: tt-perf-report `blackhole` (512 GB/s DRAM; 4096
FLOP/cycle/core at LoFi, 1.35 GHz, 110 cores; HiFi2 /2, HiFi4 /4). Decode at batch 1, context 4096. Prefill 4096
tokens. Not a hardware counter: tile padding, multicast and temporaries are excluded. Router at HiFi2 in decode and
prefill. Kernel columns come from `evidence12/profiles/`; host columns are the chain 12 headline medians.

| Profile / layer | Decode bytes | Decode roofline | Kernel sum | Traced host | Roofline / host | Prefill roofline (compute / memory) | Prefill kernel | Prefill host | Roofline / host |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| p150x4 linear | 16.3 MB | 31.9 us | 197.9 us | 228.1 us | 14.0% | 0.36 / 0.37 ms | 10.49 ms | 10.74 ms | 3.4% |
| p150x4 full | 16.4 MB | 32.0 us | 248.2 us | 257.0 us | 12.4% | 0.32 / 0.37 ms | 10.71 ms | 11.55 ms | 3.2% |
| p150x2 linear | 31.5 MB | 61.5 us | 237.2 us | 268.1 us | 22.9% | 0.71 / 0.67 ms | 14.08 ms | 14.37 ms | 4.9% |
| p150x2 full | 28.2 MB | 55.1 us | 274.7 us | 290.5 us | 19.0% | 0.61 / 0.67 ms | 14.40 ms | 15.05 ms | 4.5% |
| p150 linear | 61.8 MB | 120.7 us | 328.7 us | 358.7 us | 33.7% | 1.40 / 1.28 ms | 20.31 ms | 20.51 ms | 6.8% |
| p150 full | 55.2 MB | 107.9 us | 352.6 us | 367.9 us | 29.3% | 1.21 / 1.28 ms | 19.79 ms | 20.52 ms | 6.2% |

- Gap 1 (host vs kernel): dispatch and op-to-op gaps, 8.9-30.9 us per step (3.5-13.2% of host time). Linear rows
  have the larger gap (30.0-30.9 us); removing 5 ops cut kernel time by 10.1-12.9 us but host time by 8.3-11.1 us.
- Gap 2 (kernel vs roofline), decode: 30-51 ops per step on one token. Each op has a fixed cost of 1-3 us, and the
  weight matmuls reach 33-53% of DRAM bandwidth at M = 32 (one useful row). The largest single items are the
  GDN kernel (40.4 us, FP32 state read/write plus serial recurrence, now also the conv-state shift), SDPA (40.0 us
  at 4096 context) and RoPE (24.0 us).
- Gap 2, prefill: the useful work is small (0.4 ms on x4). Embeddings (dispatch gathers), binary, untilize,
  tilize, scatter and reduce ops sum to 5.1-5.2 ms on 4096 rows, mostly the MoE dispatch/combine path;
  `moe_fused_swiglu` adds 1.7-1.8 ms.
- Decode at 131072 context, x4 full: roofline 166.9 us (`roofline_us_131072`; the KV read alone is 71.3 MB, 139.3
  us); measured layer 458.1 us (chain 10: 460.8 us), SDPA decode op 228 us (op micro, "What changed").

## Memory and capacity (`evidence12/inventory/`, `capacity_model.json`, `memory_handoff_recheck.json`)

| Profile | Layer weights (linear / full) | 40 layers | Stage 4 40 layers | 262144-token sequences, 1 / 8 / 32 GDN slots | Stage 4 (8 / 32) |
|---|---:|---:|---:|---|---|
| p150 | 608.6 / 596.1 MiB | 23.65 GiB | 24.3 GiB | 1.71 / 1.55 / 1.01 | 1.32 / 0.77 |
| p150x2 | 585.9 / 579.6 MiB | 22.82 GiB | 23.1 GiB | 4.26 / 4.10 / 3.56 | 3.89 |
| p150x4 | 295.0 / 292.4 MiB | 11.50 GiB | 11.6 GiB | 12.89 / 12.81 / 12.54 | 12.7 |

- New persistent state: decode all-reduce buffer [1,1,32,2048*tp] BF16 L1 per mesh; dispatch scatter buffers per
  (capacity, length); router cached slices per layer; BF16 prefill norm gammas (2 x 128 KiB per layer); shared
  addcmul ones row (64 BFP8 tiles, 69632 bytes per layer, the whole layer-weight change from chain 7). Constants
  counted once. The planner cost change adds no state. The 8192-row step adds one index-offset table per full
  layer (`decoder._index_offsets[8192]`, FP32 [1, 8192] DRAM, 1048576 bytes, `evidence12/inventory/m*_layer3.json`);
  it is counted once as a shareable constant (+1048576 bytes vs chain 10 on every profile). Layer weights and
  full-context sequences are unchanged from chain 10 at the precision shown. `../context_contract.json` lists
  this table under `new_persistent_buffers`.
- Allocator after construction per layer (MiB, linear / full): p150 773.3 / 756.6; p150x2 708.9 / 732.8; p150x4
  397.2 / 442.3 (chain 10 full: 755.5 / 731.8 / 441.3). Replica merge peak 1391.0 / 1389.2 (x2), 1417.3 / 1546.6
  (x4); replicas released. Host peak RSS 6.5 GiB (x2), 4.5 GiB (x4).
- Handoff recheck: 7 of 9 items addressed; open for the full model: share RoPE tables once per mesh; measure
  capacity on the full model.
- No capability reduction: 262144 tokens on every profile (`../context_contract.json`, section
  `optimized_multichip_decoder_validation`, pointing at `evidence12/`).

Context contract (`../context_contract.json`, `optimized_multichip_decoder_validation`, written by
`update_contract.py evidence12`). Supported context 262144 tokens, capability reduction none. KV cache: BFP8
(`bfloat8_b`) paged, per-device local KV head `[pages, 1, 32, 256]` (`kv_dtype` "bfp8" in every headline
`optimization` dict). GDN recurrent state: FP32, per-device local value heads `[1, 32/tp, 128, 128]` per configured
slot. PCC columns: 262143-token prefill / decode at position 262143 / 262144-token prefill (`accept_context_m*`,
random input). Sequences: full 262144-token sequences that fit with 1 / 8 / 32 configured GDN slots (capacity
model, 2 GiB reserve estimate).

| Profile | Mesh | Linear PCC | Full PCC | KV bytes per token, all 10 full layers, per device | GDN state bytes per slot per layer | 262144-token sequences, 1 / 8 / 32 slots |
|---|---|---|---|---:|---:|---|
| p150x4 | 1x4 ring, `FABRIC_1D_RING` | 0.99929 / 0.99983 / 0.99949 | 0.99852 / 0.99786 / 0.99822 | 5440 | 536576 | 12.89 / 12.81 / 12.54 |
| p150x2 | 1x2, `TT_VISIBLE_DEVICES=2,3`, `FABRIC_1D` | 0.99929 / 0.99981 / 0.99948 | 0.99854 / 0.99790 / 0.99817 | 5440 | 1073152 | 4.26 / 4.10 / 3.56 |
| p150 | 1x1, physical device 3 | 0.99929 / 0.99982 / 0.99948 | 0.99852 / 0.99785 / 0.99816 | 10880 | 2146304 | 1.71 / 1.55 / 1.01 |

- Configured GDN slots are separate from active batch and from HTTP (HyperText Transfer Protocol) client
  concurrency; the native-context rows run batch 1.

## Work owned by later stages

- Datatype sweep (Stage 8), with full-model accuracy: router LoFi (-1.5 to -2.3 us per layer), SDPA prefill LoFi
  (about -13% full-layer prefill at long context), and a check of the adopted BFP8 routed sum (random-activation
  decode minimums 0.0004-0.0010 lower; BF16 alternative measured at +2.3 to +5.4 us per layer).
- Batched decode (B rows per layer pass): Stage 7 (full model). B > 1 is correct today (per-row loop, 7.5 ms per
  layer step at B32 on x4). It dominates the concurrency-32 sweep.
- Expert weight duplication on meshes (EP prefill copy + TP decode copy; p150x2 layer weights close to p150):
  Stage 7 memory. The stacked option is measured (+4-7% prefill) and selectable.
- RoPE gather once per step for all 10 full layers: Stage 6 (depends only on the position).
- Prefill tail shapes build programs on first use: Stage 6/9 warmup of tail-length shapes.

## Anomaly ledger

| Anomaly | Evidence | Control | Resolution |
|---|---|---|---|
| Positions row 24 token 6 strict PCC 0.990 (all profiles) | `evidence12/real_inputs_m*/positions_0.json` (strict 0.99031-0.99042, same as chain 10) | HF with the rank-10 expert gives 0.98954 | controlled: router near tie; tie-aware 0.99949-0.99956 |
| Linear headline strict decode min 0.99040-0.99078 | `evidence12/headline_m*_linear` (same as chain 10) | tie records in the row | controlled: router near ties; tie-aware 0.99971-0.99972 |
| Prefill tokens below 0.995 (4-6 linear, 1 full) | `evidence12/headline_*` (lists identical to chains 7 and 10) | HF tie-swap PCC per token | controlled: exact BF16 router ties (`micro/prefill_token_tail/`) |
| Router flip at 2 BF16 ULP (layer 3, 2600, step 6) | `evidence12/probes/router_l3_2600_x4.json` (byte-identical to chain 10) | strict PCC 0.99898 at that step | controlled: passes without the tie gate; same flip with the HiFi4 router |
| Random-activation decode minimums 0.0004-0.0010 lower than chain 9 on every profile (B32 full 0.99536) | `evidence12/b32_*`, `accept_positions_m*`, `accept_context_m*`, `accept_watcher_m*` (same decode minimums as chain 10) | chain 9 (same chain, source before the addcmul change); op micro (`micro/shared_addcmul/`); BF16-weight layer screen (`trials/shared_addcmul_bf16w/`) | controlled: caused by the BFP8 routed sum the fused op requires; recorded-input rows change by at most 0.00006; all rows pass 0.995; the BF16 alternative costs +2.3 to +5.4 us; full-model check in the datatype sweep |
| p150x2 on devices 0 and 1 is 5-7 us slower than on devices 2 and 3 | `evidence12/placement_p150x2_01_*` (295.9 / 274.3 vs 290.5 / 268.1 us, one run vs median of 3) | same gap in chain 10 (299.3 / 284.4 vs 293.3 / 277.5 us) and chain 7 (300.2 / 285.0 us); PCC identical | open, not profiled; the p150x2 profile uses devices 2 and 3 |
| Full-layer 3-token prefill slower than 100 tokens | `evidence7/prefill_timing/` | forced plans (`micro/plan_cost/`) | fixed: planner costs |
| Mesh prefill at 1100-3300 tokens slower than at 2049 or 4096 (review 3) | `evidence8/prefill_timing/` | forced single masked chunk (`micro/plan_cost/mid_*`) | fixed on x4 (1300: 7.95 -> 6.51 ms in chain 12); x2 2600-3000 now 0.3-0.9 ms slower than chain 8, disclosed (Prefill planner) |
| 2600-4095 tokens still above 4096 on meshes (x4 linear 2600 10.85 vs 4096 10.56 ms; up to +0.8 ms at 3968 on x2 linear) | `evidence12/prefill_timing/`; Tracy profiles `runs/main/stage05/micro/masked_chunk/` (p150x4, recorded inputs, 4096 / 4095 / 3000 tokens) | 4096 runs the same 4096 chunk unmasked | controlled, attributed: at 4095 the linear layer has +18 ops and +0.48 ms kernel (124 vs 106 ops, 11.05 vs 10.57 ms), the full layer +12 ops and +0.26 ms (123 vs 111, 10.96 vs 10.70 ms). The extra ops are planned-path row-major crossings (input row gather, untilize, untilize-with-unpadding, fill-pad, small binaries, typecasts, permutes), kept for trace safety: a new `valid_len` in a warmed bucket compiles no offset-specific program while another request's decode trace is live. On linear layers the GDN padding mask and conv-tail gather keep padding rows out of the recurrent and conv state (correctness). SDPA, MoE and GDN kernel time do not grow; the rest of the span is op-to-op gaps of the extra ops. Failed attempts kept: `failed_profile_guard/`, `failed_zone_hash/` |
| x4 linear 2-token prefill 2.92 ms in chain 9 (chains 8 and 10: 2.63 ms; chain 12: 2.69 ms) | `evidence9/prefill_timing/p150x4_layer0.json` runs 2.85-4.09 ms; `evidence10/` runs 2.53-3.68 ms; `evidence12/` runs 2.58-3.70 ms | same plan in all four chains | controlled: warm-run noise at short lengths (the first warm run is the slow one in all 15 noisy cells) |
| Full 1100-token decode min 0.99818 (lowest length row) | `evidence12/len_m*_full_1100` (same as chain 10) | same value on p150, x2, x4 within 1e-5 | above the 0.995 gate; mesh-independent; no Stage 4 control tensor at this length |
| Watcher 3-token full rows decode min 0.99662 (review 3) | `evidence8/watcher/m{1,4}_layer3_3.json` | `decode_router_ties`: step 4 has a 4-expert window, strict 0.99660, chosen 0.99662 | controlled: router near tie at that step; same on p150 and x4 |
| 4353-token rows fail with the recorded input | `evidence8/failed_input_too_short/` | random-input rows pass | controlled: recorded input holds 4352 tokens (harness assert before device use) |
| MoE prefill not bitwise repeatable on p150x4 (chains 2-5) | "MoE prefill nondeterminism" | p150 bitwise; identical inputs | not observed after the top-k router (chains 6, 7, 9, 10, 12); mechanism not isolated |
| Profiler hash collision on SDPA compute zones ("Reduce max" and "Softmax(Q@KT)@V", 0x992) | `trials/review2b/profiles/`; `micro/masked_chunk/failed_zone_hash/` (every full-layer profile failed after the ttnn rebuild) | layer-level timing | fixed: zone label "Reduce_max" (`evidence12/ttnn_patch.diff`), profiler label only; chain 12 full-layer profiles complete; the RoPE variant op splits were not rerun |
| BFP4 attention-out failed at 2600 tokens step 53 | `runs/main/stage05/micro/router_capture/` | BFP8: 0.99824 | fixed: reverted to BFP8 |
| Top-k decode router +40 us | `trials/topk_decode/` | 3 alternating pairs | rejected, source removed; cause not profiled |
| Fused prefill combine slower; 4096-row call overflows L1 | `trials/fused_combine/` | control rows | rejected, source removed |
| Screens with harness errors | `trials/prefill/smoke16_*_failed_harness`, `topk_l1_oom.log`, `trials/fused_combine/failed_missing_mapping/` | reruns pass | fixed and kept |
| Chain 11 linear reuse runs fail the trace allocation check (1 buffer alive before replay) | `evidence11/accept_reuse_m{4,2,1}/linear.log`, steps `real_reuse_0_m{4,2,1}` in `evidence_chain11.log` | fix check `runs/main/stage05/micro/conv_ref_fix/` (exit code 0 on all profiles) | fixed: `conv_shift="gate"` held `_conv_current` across replays; `_decode_kernel` now takes and clears it; chain 12 passes; chain 11 kept as a failed attempt |
| Earlier items (chains 1-6) | `work_log.md` | | fixed or controlled, unchanged |

## Code changes

| File | Change |
|---|---|
| `tt/gdn_decode.py`, `tt/kernels/gdn_decode/` (new) | fused FP32 GDN decode: L2 norms, decay/beta from `a`/`b`, delta-rule state update, output, RMSNorm and SiLU gate; z read at a page offset |
| `tt/expert_decode.py`, `tt/kernels/expert_gemv/` (new) | decode expert gate/up + SwiGLU kernel, FP32 accumulation, optional shared-expert slot |
| `tt/optimized_decoder.py` | decode policies above; prefill policies `prefill_norm`, `router_topk_prefill`, `gdn_flat_qk`, `gdn_gate_mixed`, `prefill_moe_merge_min`; router fidelity; planner costs; multi-row dispatch scatter; shared-expert fused addcmul (`shared_addcmul_config`, BFP8 routed down output) |
| `tt/multichip_decoder.py` | direct decode all-reduce, persistent buffer, sharded output; `_ShardedResidualLayer`; `_release_gdn_input_weight`; EP dispatch spread and scratch rows; mesh policies |
| `tt/functional_decoder.py` | B > 1 decode converts sharded rows before concat |
| `tests/hf_reference.py`, `tests/run_layer.py`, `tests/run_positions.py` | tie-aware decode gate with strict metrics kept; steps-0 guard |
| `tests/test_multichip_plan.py`, `tests/test_prefill_plan.py` | Stage 5 policy, MoE piece and planner tests; `conv_shift` default; long-chunk plan invariants (`test_long_chunk_merge`, `test_long_chunk_moe_pieces`) |
| `tt/kernels/gdn_decode/gate_writer_shift.cpp` (new), `tt/gdn_decode.py` | gate program writer that also shifts the conv state (`conv=` argument) |
| `tt/optimized_decoder.py` (review 4) | `conv_shift` ("ops", "gate", "concat35"; "gate" default on linear layers); conv row reference taken and cleared by `_decode_kernel`, guard in `_decode_heads`; `prefill_long_chunk`, `prefill_long_from`, `sdpa_prefill_long`; `_moe_pieces` splits rows above the largest bucket; index offsets for the 8192 step |
| `tt/multichip_decoder.py` (review 4) | p150x4 mesh policy `sdpa_prefill_long` q64/k256 |
| `tests/run_layer.py` (review 4) | `--profile` guard: steps 128, batch 1, trace, no audit, any length |
| `ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp` | indexed (`use_indices`) output counts as compact; no zero-fill |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp` | profiler zone label "Reduce max" -> "Reduce_max"; behavior unchanged |

- Source identity: `evidence12/source_manifest_start.json` equals `source_manifest_end.json` (55 files: the chain
  10 set plus `tt/kernels/gdn_decode/gate_writer_shift.cpp`); the snapshot is `evidence12/source_snapshot.tar.gz`.
  A read-only hash check on 2026 Oct 12 after chain 12 found every manifest file unchanged in the working tree.
- `evidence12/ttnn_patch.diff` (`git diff HEAD -- ttnn`, sha256 066fd7da...) holds the two Stage 5 ttnn changes:
  the indexed `sparse_matmul` no-fill and the SDPA zone label rename. The manifest key `git_diff_outside_autoport`
  has the same hash, and the current `git diff HEAD -- ttnn` matches it. `evidence10/ttnn_patch.diff` and earlier
  chains have an empty ttnn patch. The rebuilt `_ttnncpp.so` hash is in `work_log.md` (review 4 follow-up).

## Reproduction

```bash
source /home/hous/dev/qwen3.5-35b-a3b/scripts/environment.sh
cd /home/hous/dev/qwen3.5-35b-a3b/tt-metal
bash /home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/evidence_chain12.sh
bash /home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/process_profiles12.sh
E=/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence12
python3 /home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/capacity_model.py $E/inventory $E $E/capacity_model.json
python3 /home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/memory_handoff_recheck.py $E
python3 /home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/roofline.py $E $E/roofline.json
python3 /home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/family_table.py $E/profiles > $E/profiles/decode_family_table.md
python3 /home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/update_contract.py evidence12
```

- The chain needs a ttnn build that contains `evidence12/ttnn_patch.diff`. On this host the change was built with
  `build_metal.sh` (`work_log.md`, review 4 follow-up); the checkout still carries it.
- Micro benchmarks and screens: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/micro/`, `trials/`.

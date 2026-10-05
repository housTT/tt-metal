# Stage 3: optimized decoder (per-op profile, A/B matrix, reconciliation)

Plugin stage "optimize" mapped to Laya (PLAN.md section 5, row 3): a per-op device report for one layer at B 1, 8
and 64 (S 512), an A/B for every placement and program-config choice the port ships, and a reconciliation of per-op
device time against the traced end-to-end time at B 1 and 64. Every timing in this document carries the host
1-minute load average at which it was taken; the orchestrator's rule for this run is that no performance number is
taken above a load of 8 (the host had 16 cores and three CPU tracks computing fp32 references).

## Method

- Per-op profile: `tests/profile_layer.py --batch B --seq 512 --layer 1 --repeats 5` under
  `python -m tracy -r -p -v -o <dir>` then `tt-perf-report <ops csv> --csv <dir>/perf_report.csv`. Layer 1 is a sliding
  layer with the attention norm (layer 0 lacks it); the input is a fixed random `(B,512,1024)` bf16 tensor (the per-op
  cost does not depend on the content). The last of the five repeats is the reported pass
  (`/home/hous/dev/laya/scratch/op_report.py` groups the CSV by the repeated op sequence).
- End-to-end: `tests/bench_buckets.py --buckets 1x512,8x512,64x512 --variants '{name: PortConfig overrides}'`, one
  process per variant (so an out-of-memory variant cannot poison the next), eager p50 of 5 and traced p50 of 20 after 3
  warm replays through `LayaTraceRunner`, real typed-decisions inputs, bit-identity of traced versus eager recorded per
  cell, `loadavg` recorded per cell. JSON per variant in this directory (`bench_<variant>.json`).
- A/B matrix (`/home/hous/dev/laya/scratch/t1_quiet_queue.sh`, step 4):

| variant | `PortConfig` overrides | question |
|---|---|---|
| default | none | shipped baseline of stage 1: Wqkv 8x8 mcast config, Wo on an 8x8 core grid, GeGLU block-sharded 2816 for 12 tiles per core up to 2048 rows, L1 attention chain up to 2048 rows, SDPA on the full grid with chunks 128 (rows < 768) and 256, rotary sharded when rows >= 24576 and at most 512 KiB per core |
| qkv_minimal_11x10 | `qkv_mode=minimal_11x10` | Wqkv through `experimental.minimal_matmul` on 11x10 (CLM measured 2 to 4 percent per layer) |
| down_minimal_11x10 | `down_grid=minimal_11x10` | attn Wo and mlp Wo (interleaved path) through `minimal_matmul` on 11x10 |
| both_minimal_11x10 | both | combination |
| geglu_interleaved | `geglu_plan=interleaved` | no sharded GeGLU, unpadded 2624 weights, ttnn's own matmul choice |
| geglu_3072 | `intermediate_pad=3072` | the 1x4-subblock padded width against 2816's 1x1 |
| geglu_shard_to_4096 | `shard_max_rows=4096` | does the sharded plan fit and pay at B 8 |
| sdpa_q64 | `sdpa_q_chunk=64, sdpa_k_chunk=64` | tt_transformers' Blackhole prefill chunking |
| sdpa_128 | `sdpa_q_chunk=128, sdpa_k_chunk=128` | smaller chunks at B 8 and 64 |
| sdpa_grid_8x8 | `sdpa_grid=8x8` | 64 cores against 110 |
| chain_dram_always | `l1_attention_max_rows=0` | DRAM attention chain at every bucket |
| chain_l1_to_4096_outblock4 | `l1_attention_max_rows=4096, qkv_out_block_h_max=4` | L1 chain at B 8 with halved matmul circular buffers |
| rotary_never_sharded | `rotary_shard_max_bytes_per_core=0` | interleaved rotary at every bucket |
| rotary_always_sharded | `rotary_shard_min_rows=0, rotary_shard_max_bytes_per_core=4 MiB` | sharded rotary at every bucket including B 64 (1 MiB per core) |
| down_grid_auto | `down_grid=auto` | ttnn's own grid for Wo |

## Per-op device profiles of the stage 1 baseline (`PortConfig` defaults of stage 1)

Source: `../functional_decoder/tracy/layer1_b{1,8,64}s512/perf_report.csv`, last of five passes, grouped by op
(`/home/hous/dev/laya/scratch/op_report.py`). Device kernel durations come from device timestamps and do not depend
on host load; the loads are recorded anyway.

### B 1 (L1 attention chain, interleaved GeGLU, ttnn's automatic matmul for Wi and Wo)

Layer 1 at B 1, S 512: 16 device ops, 429.4 us device kernel time per pass (load at start 8.06).

| op | calls | device us | share | cores | FLOPs % / DRAM % (tt-perf-report) | advice |
|---|---|---|---|---|---|---|
| MatmulDeviceOperation 512 x 1024 x 2624 | 2 | 87.5 | 20.4 % | 88 | 39 / 29 | If possible place input 0 in L1 (currently in DEV_1_DRAM_INTERLEAVED) • in0_block_w=1 is small, try in0_block_ |
| SDPAOperation | 1 | 75.0 | 17.5 % | 110 |  |  |
| MatmulDeviceOperation 512 x 2624 x 1024 | 1 | 45.0 | 10.5 % | 64 | 52 / 28 | If possible place input 0 in L1 (currently in DEV_1_DRAM_INTERLEAVED) • in0_block_w=2 and output subblock 1x2  |
| BinaryNgDeviceOperation | 3 | 41.3 | 9.6 % | 110 |  |  |
| MatmulDeviceOperation 512 x 1024 x 3072 | 1 | 39.1 | 9.1 % | 64 | 70 / 21 | Increase grid size (currently using 64) |
| LayerNormDeviceOperation | 2 | 36.3 | 8.5 % | 16 |  |  |
| NlpCreateHeadsDeviceOperation | 1 | 31.0 | 7.2 % | 16 |  |  |
| RotaryEmbeddingHfDeviceOperation | 2 | 30.3 | 7.1 % | 110 |  |  |
| UnaryDeviceOperation | 1 | 17.1 | 4.0 % | 110 |  |  |
| MatmulDeviceOperation 512 x 1024 x 1024 | 1 | 15.9 | 3.7 % | 64 | 57 / 13 | in0_block_w=4 and output subblock 1x2 look good • If your matmuls are not FLOP-bound use HiFi4 with BF16 act |
| NLPConcatHeadsDeviceOperation | 1 | 10.8 | 2.5 % | 16 |  |  |

### B 8 (DRAM chain above 2048 rows, interleaved GeGLU, rotary sharded with two reshards)

Layer 1 at B 8, S 512: 20 device ops, 2751.5 us device kernel time per pass (load at start 7.09).

| op | calls | device us | share | cores | FLOPs % / DRAM % (tt-perf-report) | advice |
|---|---|---|---|---|---|---|
| MatmulDeviceOperation b={8} x 512 x 1024 x 2624 | 2 | 699.5 | 25.4 % | 88 | 39 / 18 | If possible place input 0 in L1 (currently in DEV_1_DRAM_INTERLEAVED) • in0_block_w=1 is small, try in0_block_ |
| SDPAOperation | 1 | 406.5 | 14.8 % | 110 |  |  |
| MatmulDeviceOperation b={8} x 512 x 1024 x 3072 | 1 | 301.1 | 10.9 % | 64 | 73 / 24 | Increase grid size (currently using 64) |
| BinaryNgDeviceOperation | 3 | 265.6 | 9.7 % | 110 |  |  |
| MatmulDeviceOperation b={8} x 512 x 2624 x 1024 | 1 | 261.4 | 9.5 % | 64 | 71 / 24 | Increase grid size (currently using 64) |
| RotaryEmbeddingHfDeviceOperation | 2 | 159.2 | 5.8 % | 64 |  |  |
| LayerNormDeviceOperation | 2 | 132.8 | 4.8 % | 110 |  |  |
| NlpCreateHeadsDeviceOperation | 1 | 130.7 | 4.8 % | 110 |  |  |
| MatmulDeviceOperation b={8} x 512 x 1024 x 1024 | 1 | 119.6 | 4.3 % | 64 | 61 / 29 | If possible place input 0 in L1 (currently in DEV_1_DRAM_INTERLEAVED) • in0_block_w=4 and output subblock 1x2  |
| UnaryDeviceOperation | 1 | 115.5 | 4.2 % | 110 |  |  |
| ShardedToInterleavedDeviceOperation | 2 | 56.8 | 2.1 % | 64 |  |  |
| NLPConcatHeadsDeviceOperation | 1 | 54.4 | 2.0 % | 110 |  |  |
| InterleavedToShardedDeviceOperation | 2 | 48.4 | 1.8 % | 64 |  |  |

### B 64 (DRAM chain, interleaved GeGLU, rotary interleaved)

Layer 1 at B 64, S 512: 16 device ops, 20839.2 us device kernel time per pass (load at start 8.49).

| op | calls | device us | share | cores | FLOPs % / DRAM % (tt-perf-report) | advice |
|---|---|---|---|---|---|---|
| MatmulDeviceOperation b={64} x 512 x 1024 x 2624 | 2 | 5589.2 | 26.8 % | 88 | 39 / 17 | If possible place input 0 in L1 (currently in DEV_1_DRAM_INTERLEAVED) • in0_block_w=1 is small, try in0_block_ |
| SDPAOperation | 1 | 2649.8 | 12.7 % | 110 |  |  |
| MatmulDeviceOperation b={64} x 512 x 1024 x 3072 | 1 | 2393.7 | 11.5 % | 64 | 73 / 22 | Increase grid size (currently using 64) |
| MatmulDeviceOperation b={64} x 512 x 2624 x 1024 | 1 | 2084.5 | 10.0 % | 64 | 72 / 23 | Increase grid size (currently using 64) |
| BinaryNgDeviceOperation | 3 | 1986.0 | 9.5 % | 110 |  |  |
| MatmulDeviceOperation b={64} x 512 x 1024 x 1024 | 1 | 1657.6 | 8.0 % | 64 | 35 / 16 | If possible place input 0 in L1 (currently in DEV_1_DRAM_INTERLEAVED) • in0_block_w=1 is small, try in0_block_ |
| RotaryEmbeddingHfDeviceOperation | 2 | 1520.1 | 7.3 % | 110 |  |  |
| NlpCreateHeadsDeviceOperation | 1 | 1004.9 | 4.8 % | 110 |  |  |
| UnaryDeviceOperation | 1 | 881.4 | 4.2 % | 110 |  |  |
| LayerNormDeviceOperation | 2 | 711.6 | 3.4 % | 110 |  |  |
| NLPConcatHeadsDeviceOperation | 1 | 360.4 | 1.7 % | 110 |  |  |

What the profiles say: the four matmuls are 45 to 56 percent of a layer; the two Wi matmuls and the attention Wo on
ttnn's automatic configuration run at 35 to 39 percent of the FLOP roofline with `in0_block_w` 1 ("SLOW"), while the
explicit 8x8 Wqkv configuration reaches 73 percent on 64 cores; SDPA is 13 to 18 percent; the three elementwise ops
(two residual adds and the gate multiply) are 9.5 percent and DRAM-bandwidth bound at B 64; the separate
`UnaryDeviceOperation` (4 percent) is the GELU that the automatic matmul path does not fuse; at B 8 the sharded rotary
pays 105 us of reshards on top of 159 us of rotary; LayerNorm and the head reshapes sit on 16 cores at 512 rows (one
core per tile row) and on 110 cores from 4096 rows.

## Reconciliation: device kernel time versus traced end to end (baseline)

| rows | ops per layer | layer device us | 28 layers + 2 head layers (device kernel time, head layer estimated as a layer without rotary) | traced end to end p50 | eager p50 | gap | load |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 429.4 | 12.82 ms | 13.92 ms | 14.73 ms | 1.09 ms (7.9 %) | 8.7 |
| 8 | 20 | 2751.5 | 82.23 ms | 85.52 ms | 85.93 ms | 3.29 ms (3.8 %) | 8.5 |
| 64 | 16 | 20839.2 | 622.14 ms | 646.04 ms | 645.99 ms | 23.91 ms (3.7 %) | 8.7 |

The gap holds embeddings, the two mask adds, the final norm, the scorer (one 1024x1024 matmul, one typecast, one fp32
1024x1 matmul), the CLS slice, the two readbacks and the dispatch of about 450 ops; at B 1 it is 1.1 ms, which is also
the whole difference between eager and traced (trace removes 0.8 ms of host dispatch there and nothing above B 8).
Throughput of the baseline: 71.9, 93.5 and 99.1 rows per second at B 1, 8 and 64 (36.8 k, 47.9 k and 50.7 k tokens per
second), so batching beyond 8 buys little: the device is already busy, at low matmul efficiency.

## A/B results (traced p50, 20 replays after 3 warm, real inputs; delta against `default` = the stage 1 baseline)

Every row is one process (`bench_<variant>.json`), host 1-minute load per cell in the load column. The orchestrator's
rule for this run was a load under 8; the queues waited for it before each process, but the load rose during several
processes: of the 132 A/B cells, 46 record a load at or above 8.0 and 12 sit at 9.3 to 10.5
(`b2b4_sharded_2816` 9.7, `b2b4_interleaved_2816_11x8` 9.6, `b2b4_interleaved_2816_8x8_wo` 9.3 to 9.4,
`b2b4_interleaved_auto` 10.0, `geglu_interleaved` 10.0 to 10.5, `il2816_8x8` at 64x512 10.1). The shipped cells
(`bench_shipped_final*.json`) sit at 7.45 to 8.18. Run-to-run scatter: `default` against `default_rerun` 0.9 / 0.2 /
0.0 percent at B 1 / 8 / 64; at B 1 two pairs of identical configurations differ by about 5 percent (`sdpa_128` and
`rotary_never_sharded` at 1x512 equal `default` by construction and measured 13.28 against 13.92 ms; `cand_C3...`
against `qkv_minimal_only` 11.86 against 12.28 ms), so B 1 differences under about 5 percent are not readable; B 8 and
B 64 are device bound and repeat within 1 percent. `qkv_minimal_only` started after the shipped defaults were set and
therefore measures the shipped port (its JSON records the port).

| variant | policy | 1x512 ms (delta) | 2x512 | 4x512 | 8x512 ms (delta) | 64x512 ms (delta) | load | what it ran, outcome |
|---|---|---|---|---|---|---|---|---|
| b2b4_interleaved_2816_11x8 | bf8w_hifi3 |  | 19.53 | 34.06 |  |  | 9.6, 9.6 |  |
| b2b4_interleaved_2816_8x8_wo | bf8w_hifi3 |  | 19.89 | 35.02 |  |  | 9.4, 9.3 |  |
| b2b4_interleaved_auto | bf8w_hifi3 |  | 22.14 | 39.15 |  |  | 10.0, 10.0 |  |
| b2b4_sharded_2816 | bf8w_hifi3 |  | 18.55 | 32.15 |  |  | 9.7, 9.7 |  |
| b2b4_sharded_3072 | bf8w_hifi3 |  |  |  |  |  |  |  FAILED: Statically allocated circular buffers in program 104 clash with L1 buffers on core range [0-0 - 7-7]. L1 buffer allocated at 707328 and static circula |
| both_minimal_11x10 | bf8w_hifi3 | 14.04 (+0.9 %) |  |  | 81.51 (-4.7 %) | 566.17 (-12.4 %) | 6.5, 6.5, 6.8 | same ops as qkv_minimal_11x10 (duplicate measurement) |
| cand_C1_b2b4 | bf8w_hifi3 |  | 18.94 | 32.12 |  |  | 7.8, 7.8 |  |
| cand_C1_minimal_qkv_wo_il11x8_sdpa8x8_rotnever | bf8w_hifi3 | 12.66 (-9.0 %) |  |  | 67.60 (-21.0 %) | 482.30 (-25.3 %) | 7.0, 7.0, 7.1 |  |
| cand_C2_minimal_qkv_womcast_il11x8_sdpa8x8_rotnever | bf8w_hifi3 | 12.31 (-11.5 %) |  |  | 69.22 (-19.1 %) | 507.58 (-21.4 %) | 8.2, 8.2, 8.1 |  |
| cand_C3_minimal_qkv_only_il11x8_sdpa8x8_rotnever | bf8w_hifi3 | 11.86 (-14.7 %) |  |  | 69.77 (-18.4 %) | 556.73 (-13.8 %) | 7.4, 7.4, 7.5 |  |
| cand_C4_C1_geglu_interleaved_b2b4 | bf8w_hifi3 |  | 19.54 | 32.55 |  |  | 8.0, 8.0 |  |
| chain_dram_always | bf8w_hifi3 | 14.79 (+6.3 %) |  |  | 84.81 (-0.8 %) | 644.25 (-0.3 %) | 4.2, 4.2, 3.5 | DRAM attention chain at every bucket |
| chain_l1_4096_sdpa128 | bf8w_hifi3_erf |  |  |  | 65.80 (-23.1 %) |  | 8.0 | L1 chain at B 8 with 128-token SDPA chunks |
| chain_l1_to_4096_outblock4 | bf8w_hifi3 |  |  |  |  |  |  | L1 chain at B 8 with a halved Wqkv out block FAILED: Statically allocated circular buffers in program 72 clash with L1 buffers on core range [0-0 - 10-9]. L1 buffer allocated at 1030912 and static circul |
| default | bf8w_hifi3 | 13.92 (+0.0 %) |  |  | 85.52 (+0.0 %) | 646.04 (+0.0 %) | 8.7, 8.5, 8.7 | stage 1 baseline (Wqkv 8x8 mcast, Wo on an 8x8 core grid, GeGLU auto with a separate GELU, L1 chain to 2048 rows, SDPA full grid 128/256, rotary sharded from 24576 rows) |
| default_rerun | bf8w_hifi3 | 13.79 (-0.9 %) |  |  | 85.69 (+0.2 %) | 646.00 (-0.0 %) | 7.0, 7.0, 7.0 | baseline again at low load (noise estimate) |
| down_grid_auto | bf8w_hifi3 | 16.41 (+17.9 %) |  |  | 96.66 (+13.0 %) | 716.10 (+10.8 %) | 1.9, 1.9, 2.4 | ttnn's own grid for Wo |
| down_minimal_11x10 | bf8w_hifi3 | 13.71 (-1.5 %) |  |  | 83.46 (-2.4 %) | 593.37 (-8.2 %) | 5.5, 5.5, 5.9 | attn Wo and mlp Wo through minimal_matmul 11x10 |
| fp32res_on_C1 | bf8w_hifi3_fp32res | 13.94 (+0.2 %) |  |  | 80.38 (-6.0 %) | 581.77 (-9.9 %) | 8.2, 8.2, 8.1 |  |
| geglu_3072 | bf8w_hifi3 | 13.89 (-0.2 %) |  |  | 85.76 (+0.3 %) | 646.16 (+0.0 %) | 8.6, 8.6, 8.2 | padded width 3072 for the sharded plan; identical to the baseline at these buckets |
| geglu_interleaved | bf8w_hifi3 | 13.85 (-0.4 %) |  |  | 85.46 (-0.1 %) | 646.77 (+0.1 %) | 10.5, 10.5, 10.0 | no sharded plan; identical to the baseline at these buckets (the sharded plan only engages at B 2 and 4) |
| geglu_shard_to_4096 | bf8w_hifi3 |  |  |  |  |  |  | sharded GeGLU at B 8 FAILED: Statically allocated circular buffers in program 86 clash with L1 buffers on core range [0-0 - 7-7]. L1 buffer allocated at 740096 and static circular |
| gelu_separate_erf | bf8w_hifi3_erf | 12.92 (-7.2 %) | 21.33 | 37.14 | 66.23 (-22.6 %) | 529.66 (-18.0 %) | 6.7, 6.7, 6.8, 6.8, 6.9 |  |
| il2816_11x8 | bf8w_hifi3 | 12.55 (-9.8 %) |  |  | 75.45 (-11.8 %) | 587.64 (-9.0 %) | 8.6, 8.6, 8.4 | same on an 11x8 grid (88 cores, per_core_N 8, out_subblock 1x4) |
| il2816_11x8_wo_mcast | bf8w_hifi3 | 12.87 (-7.5 %) |  |  | 74.76 (-12.6 %) | 536.61 (-16.9 %) | 8.8, 8.8, 8.7 |  |
| il2816_11x8_wo_mcast_sdpa128 | bf8w_hifi3 | 12.79 (-8.1 %) |  |  | 76.54 (-10.5 %) | 545.75 (-15.5 %) | 8.5, 8.4, 8.3 |  |
| il2816_8x8 | bf8w_hifi3 | 13.30 (-4.4 %) |  |  | 85.32 (-0.2 %) | 627.49 (-2.9 %) | 7.7, 8.3, 10.1 | interleaved GeGLU padded to 2816 with an explicit 8x8 mcast config and the GELU fused |
| il3072_8x8 | bf8w_hifi3 | 13.52 (-2.9 %) |  |  | 81.08 (-5.2 %) | 633.81 (-1.9 %) | 7.5, 7.5, 7.4 | interleaved GeGLU padded to 3072, 8x8 mcast config, GELU fused |
| policy_bf16_hifi4 | bf16_hifi4 | 14.56 (+4.6 %) |  |  | 81.47 (-4.7 %) | 582.07 (-9.9 %) | 7.2, 7.3, 7.5 |  |
| policy_bf16w_hifi3 | bf16w_hifi3 | 12.97 (-6.8 %) |  |  | 70.31 (-17.8 %) | 493.39 (-23.6 %) | 7.6, 7.6, 7.5 |  |
| policy_bf8w_hifi2 | bf8w_hifi2 | 11.83 (-15.0 %) |  |  | 62.15 (-27.3 %) | 442.76 (-31.5 %) | 4.8, 5.6, 5.9 |  |
| policy_bf8w_hifi3_erf | bf8w_hifi3_erf | 12.97 (-6.8 %) |  |  | 73.04 (-14.6 %) | 527.28 (-18.4 %) | 6.7, 6.7, 6.8 |  |
| policy_bf8w_hifi3_head_bf16 | bf8w_hifi3_head_bf16 | 11.81 (-15.1 %) |  |  | 67.29 (-21.3 %) | 482.64 (-25.3 %) | 6.7, 6.7, 5.8 |  |
| policy_bf8w_hifi4 | bf8w_hifi4 | 13.24 (-4.9 %) |  |  | 73.29 (-14.3 %) | 528.76 (-18.2 %) | 7.5, 7.5, 7.4 |  |
| policy_bf8w_lofi_mlp | bf8w_lofi_mlp | 11.79 (-15.3 %) |  |  | 61.23 (-28.4 %) | 435.50 (-32.6 %) | 6.5, 6.5, 6.6 |  |
| qkv_minimal_11x10 | bf8w_hifi3 | 14.23 (+2.3 %) |  |  | 81.32 (-4.9 %) | 565.81 (-12.4 %) | 4.0, 4.0, 4.9 | as wired at the time: Wqkv, attn Wo and mlp Wo through minimal_matmul 11x10 |
| qkv_minimal_only | bf8w_hifi3 | 12.28 (-11.8 %) |  |  | 67.53 (-21.0 %) | 481.49 (-25.5 %) | 7.3, 7.3, 7.9 | Wqkv through minimal_matmul 11x10, Wo on the 8x8 core grid (after the wiring fix) |
| rotary_always_sharded | bf8w_hifi3 |  |  |  |  |  |  | rotary sharded everywhere FAILED: Out of Memory: Not enough space to allocate 67108864 B L1 buffer across 64 banks, where each bank needs to store 1048576 B, but bank size is 1382144 B |
| rotary_never_sharded | bf8w_hifi3 | 13.28 (-4.5 %) |  |  | 83.30 (-2.6 %) | 644.61 (-0.2 %) | 2.8, 2.7, 2.3 | rotary interleaved everywhere (the baseline shards at B 8) |
| sdpa_128 | bf8w_hifi3 | 13.28 (-4.5 %) |  |  | 86.37 (+1.0 %) | 653.17 (+1.1 %) | 7.1, 7.1, 5.9 | SDPA chunks 128/128 at every bucket |
| sdpa_grid_8x8 | bf8w_hifi3 | 12.70 (-8.7 %) |  |  | 82.99 (-3.0 %) | 642.19 (-0.6 %) | 5.2, 5.2, 4.3 | SDPA on 64 cores |
| sdpa_q64 | bf8w_hifi3 | 14.10 (+1.3 %) |  |  | 93.93 (+9.8 %) | 724.63 (+12.2 %) | 7.6, 7.6, 7.4 | SDPA chunks 64/64 |
| shipped | bf8w_hifi3 | 12.42 (-10.8 %) |  |  | 68.12 (-20.3 %) | 481.39 (-25.5 %) | 8.4, 8.4, 8.2 |  |
| shipped_b2_b4_b16_b32 | bf8w_hifi3_erf |  | 21.01 | 36.87 |  |  | 8.4, 8.4 |  |
| shipped_erf | bf8w_hifi3_erf | 12.93 (-7.1 %) |  |  | 73.24 (-14.4 %) | 527.51 (-18.3 %) | 5.6, 5.6, 5.9 |  |
| shipped_erf_b2_b4_b16_b32 | bf8w_hifi3_erf |  | 21.32 | 36.75 |  |  | 6.1, 6.1 |  |
| shipped_final | bf8w_hifi3_erf | 12.96 (-6.8 %) |  |  | 65.81 (-23.0 %) | 528.23 (-18.2 %) | 7.5, 7.5, 7.5 |  |
| shipped_final_b2_b4_b16_b32 | bf8w_hifi3_erf |  | 21.01 | 36.71 |  |  | 8.2, 8.2 |  |
| wo_mcast | bf8w_hifi3 | 13.81 (-0.8 %) |  |  | 85.25 (-0.3 %) | 622.42 (-3.7 %) | 7.8, 7.8, 7.6 | explicit 8x8 mcast configs (in0_block_w 8) for attn Wo and mlp Wo |

Readings per lever:

- Wqkv: `minimal_matmul` on 11x10 beats the 8x8 mcast config at every bucket (C3 against `il2816_11x8`: 11.86 against
  12.55 ms at B 1, 69.8 against 75.5 at B 8, 556.7 against 587.6 at B 64).
- Wo (attn and mlp): `minimal_matmul` wins from 4096 rows (C1 against C3: 67.6 against 69.8 at B 8, 482.3 against
  556.7 at B 64) and loses below (12.66 against 11.86 at B 1; 18.94 / 32.12 against 18.55 / 32.15 at B 2 / 4), the
  explicit 8x8 mcast Wo config (C2) sits between; shipped: `minimal_matmul` from 4096 rows, the 8x8 core grid below.
  ttnn's own grid for Wo (`down_grid_auto`) costs 11 to 18 percent.
- GeGLU: above 2048 rows the padded 2816 interleaved plan with the fused-GELU mcast config on 11x8 (88 cores) is the
  best (`il2816_11x8`: -9.8 / -11.8 / -9.0 percent); the same on 8x8 -4.4 / -0.2 / -2.9; padded 3072 on 8x8 -2.9 /
  -5.2 / -1.9. At 1024 to 2048 rows the block-sharded 2816 plan wins (18.55 / 32.15 ms against 19.53 / 34.06 for the
  interleaved 11x8 config and 22.14 / 39.15 for ttnn's automatic path); the sharded plan at 4096 rows and the sharded
  3072 plan at 1024 rows fail with the circular-buffer clash. The padding is exact (stage 1).
- SDPA: 8x8 beats the full 11x10 grid (-8.7 / -3.0 / -0.6 percent alone); chunks 64/64 lose 10 to 12 percent above
  B 1; 128/128 loses 1 percent to 256/256 above 768 rows (also inside the combination: 76.5 against 74.8 ms at B 8).
- Attention chain: L1 interleaved is worth 6.3 percent at B 1 (`chain_dram_always` 14.79 ms); at 4096 rows it does
  not fit next to SDPA's circular buffers (same clash address with a halved Wqkv block; the 128-chunk follow-up is in
  `bench_chain_l1_4096_sdpa128.json`).
- Rotary: the sharded round trip loses on p150 (`rotary_never_sharded` -2.6 percent at B 8, where the baseline shards;
  sharding everywhere runs out of L1 at B 64); shipped: never sharded.
- Combination C1 (shipped levers, Wo minimal everywhere): 12.66 / 67.60 / 482.3 ms (-9.0 / -21.0 / -25.3 percent).

## Precision policies on the C1 port: accuracy (correctness runs) and cost (traced p50, ms)

| policy | encoder B 1 | fill row B 1 | B 2 | B 4 | B 8 | S 1024 | markers B 1 (PCC, max abs) | markers B 8 | cost at B 1 / 8 / 64 (traced ms) |
|---|---|---|---|---|---|---|---|---|---|
| bf8w_hifi3 | 0.99737 | 0.99106 | 0.99497 | 0.99514 | 0.99297 | 0.99740 | 0.99294, 0.400 | 0.99209, 0.351 | 12.66 / 67.60 / 482.30 |
| bf8w_hifi3_erf | 0.99936 | 0.99894 | 0.99550 | 0.99848 | 0.99785 | 0.99939 | 0.99986, 0.090 | 0.99934, 0.153 | 12.97 / 73.04 / 527.28 |
| bf16_hifi4 | 0.99958 | 0.99797 | 0.99652 |  |  | 0.99957 | 0.99974, 0.119 | 0.99946, 0.115 | 14.56 / 81.47 / 582.07 |
| bf8w_hifi4 | 0.99738 | 0.99214 | 0.99509 |  |  |  | 0.99433, 0.329 | 0.99062, 0.400 | 13.24 / 73.29 / 528.76 |
| bf16w_hifi3 | 0.99737 | 0.99237 | 0.99449 |  |  |  | 0.99424, 0.372 | 0.99154, 0.384 | 12.97 / 70.31 / 493.39 |
| bf8w_hifi2 | 0.99746 | 0.99093 | 0.99504 |  |  | 0.99738 | 0.99598, 0.306 | 0.99271, 0.353 | 11.83 / 62.15 / 442.76 |
| bf8w_hifi3_head_bf16 | 0.99737 | 0.99106 | 0.99497 |  |  |  | 0.99311, 0.396 | 0.99212, 0.351 | 11.81 / 67.29 / 482.64 |
| bf8w_hifi3_fp32res | 0.99739 | 0.99293 | 0.99503 |  |  | 0.99722 | 0.99332, 0.352 | 0.99226, 0.349 | 13.94 / 80.38 / 581.77 |

- The tanh GELU approximation is the dominant error of the upstream default on -large: `bf8w_hifi3_erf` (exact erf
  fused in the Wi_act matmul) lifts the encoder from 0.9974 to 0.9994 at B 1, the "fill row" from 0.9911 to 0.9989,
  and the end-to-end marker logits from PCC 0.9929 / max abs 0.343 to 0.99986 / 0.090 at B 1 and 0.99934 / 0.153 at
  B 8, for +2.4 / +8.0 / +9.3 percent of time. Neither bf16 weights alone (`bf16w_hifi3`) nor HiFi4 alone
  (`bf8w_hifi4`) nor the fp32 residual stream (`bf8w_hifi3_fp32res`, +10 to +21 percent) moves the accuracy; HiFi2
  equals HiFi3 and is 8 percent faster; LoFi for Wi and Wo did not hang on p150 (the plan's N300 note) and is
  10 percent faster; its accuracy is for the stage 8 sweep.
- Erf cost at B 2 and B 4 (sharded GeGLU plan): the shipped erf cells are 21.01 / 36.71 ms against the tanh C1 cells
  18.94 / 32.12 ms (`bench_cand_C1_b2b4.json`), +10.9 / +14.3 percent; that pair is confounded by the Wo lever (C1 ran
  Wo through `minimal_matmul` at every row count, the shipped port uses the 8x8 core grid below 4096 rows, worth about
  2 percent in the other direction at these buckets), and the erf cost on the sharded GeGLU plan alone was not
  separately measured.
- Shipped default policy: `bf8w_hifi3_erf` (`DEFAULT_POLICY_NAME`); `bf8w_hifi3` remains in the table for the sweep
  and as the plan's starting point.

## Shipped configuration and its numbers

`PortConfig` defaults (`tt/model_config.py: DEFAULT_PORT`) and `DEFAULT_POLICY_NAME = "bf8w_hifi3_erf"`:

| rows x seq | attention chain | GeGLU | Wqkv | Wo | SDPA chunk (grid 8x8) |
|---|---|---|---|---|---|
| 1x512 | L1 | interleaved 2816 | minimal | core grid 8x8 | 128 |
| 2x512 | L1 | sharded 2816 | minimal | core grid 8x8 | 256 |
| 4x512 | L1 | sharded 2816 | minimal | core grid 8x8 | 256 |
| 8x512 | L1 | interleaved 2816 | minimal | minimal | 128 |
| 16x512 | DRAM | interleaved 2816 | minimal | minimal | 256 |
| 32x512 | DRAM | interleaved 2816 | minimal | minimal | 256 |
| 64x512 | DRAM | interleaved 2816 | minimal | minimal | 256 |

Rotary is never sharded; the head layers reuse the attention module with the same plan; the scorer's last linear
runs in fp32. Every value below comes from `bench_shipped_final.json` and `bench_shipped_final_b2b4b16b32.json`
(20 traced replays after 3 warm, real typed-decisions inputs, one process per file).

| rows x seq | traced p50 | traced min | p95 | rows per second | tokens per second | eager p50 | stage 1 baseline (tanh, stage 1 port) | change | load |
|---|---|---|---|---|---|---|---|---|---|
| 1x512 | 12.96 ms | 12.85 | 13.17 | 77.1 | 39496 | 13.52 | 13.92 | -6.8 % | 7.5 |
| 2x512 | 21.01 ms | 20.85 | 21.30 | 95.2 | 48728 | 22.16 |  |  | 8.2 |
| 4x512 | 36.71 ms | 36.46 | 37.03 | 109.0 | 55796 | 37.65 |  |  | 8.2 |
| 8x512 | 65.81 ms | 65.55 | 66.11 | 121.6 | 62240 | 66.07 | 85.52 | -23.0 % | 7.5 |
| 16x512 | 137.57 ms | 137.14 | 138.16 | 116.3 | 59547 | 137.63 |  |  | 8.1 |
| 32x512 | 268.83 ms | 268.32 | 269.53 | 119.0 | 60945 | 268.90 |  |  | 8.0 |
| 64x512 | 528.23 ms | 526.27 | 529.09 | 121.2 | 62033 | 528.09 | 646.04 | -18.2 % | 7.5 |

Layer 1 at B 8 under the shipped configuration: 15 device ops, 2081.2 us per pass (load at start 6.73; `tracy/shipped_final_layer1_b8s512/perf_report.csv`).

| op | calls | device us | share | cores |
|---|---|---|---|---|
| MatmulDeviceOperation b={8} x 512 x 1024 x 2816 | 2 | 655.0 | 31.5 % | 88 |
| SDPAOperation | 1 | 323.9 | 15.6 % | 64 |
| BinaryNgDeviceOperation | 3 | 257.5 | 12.4 % | 110 |
| MinimalMatmulDeviceOperation b={8} x 512 x 2816 x 1024 | 1 | 200.7 | 9.6 % | 110 |
| MinimalMatmulDeviceOperation b={8} x 512 x 1024 x 3072 | 1 | 196.8 | 9.5 % | 110 |
| RotaryEmbeddingHfDeviceOperation | 2 | 134.5 | 6.5 % | 110 |
| LayerNormDeviceOperation | 2 | 134.2 | 6.4 % | 110 |
| MinimalMatmulDeviceOperation b={8} x 512 x 1024 x 1024 | 1 | 94.2 | 4.5 % | 110 |
| NlpCreateHeadsDeviceOperation | 1 | 62.7 | 3.0 % | 110 |
| NLPConcatHeadsDeviceOperation | 1 | 21.7 | 1.0 % | 110 |

Reconciliation at B 8: 28 layers plus two head layers of device kernel time 62.17 ms against the traced end to end 65.81 ms (gap 3.64 ms, 5.5 percent: embeddings, mask adds, final norm, scorer, CLS slice, readbacks).
At B 64 (profile `tracy/shipped_erf_layer1_b64s512`, taken with the L1 chain at 2048 rows, identical placement at this bucket): layer 16800.0 us, bound 501.01 ms against traced 528.23 ms (gap 5.2 percent).
At B 1 (profile `tracy/shipped_erf_layer1_b1s512`, taken with the L1 chain at 2048 rows, identical placement at this bucket): layer 400.2 us, bound 11.95 ms against traced 12.96 ms (gap 7.8 percent).

Trace safety on the shipped configuration (`replay_trace_check_shipped_tracked.json`, `TT_METAL_TRACE_ALLOC_TRACKING=1`,
tracker error: None):

| bucket | traced == eager | repeated replay identical | input change moves output | NaN |
|---|---|---|---|---|
| 1x512 | True | True | True (3.28) | False |
| 8x512 | True | True | True (3.32) | False |
| 64x512 | True | True | True (3.69) | False |

Correctness on the shipped configuration: `pcc_rows_shipped.json` (first pass, L1 chain 2048), the final-queue-4
log (L1 chain 4096) and the per-layer trace `layer_pcc_shipped.json` (196 rows, policy `bf8w_hifi3_erf`, shipped
port with the L1 chain at 4096 rows, written by the 22:31 UTC encoder run of `s3_final_tests2`; same layout as the
stage 1 file): encoder 0.99940 (1x512), 0.99805 (fill row), 0.99547 (2x512 sharded), 0.99704 (2x512
interleaved), 0.99845 (4x512), 0.99784 (8x512), 0.99939 (1x1024); end to end marker logits 0.99992 with max abs 0.055
(B 1) and 0.99926 with 0.123 (B 8); MLP 0.99999, attention 0.99986 / 0.99984, layers 0.99997 to 0.9999987, head layers
0.999996; all encoder-side negative controls unchanged (0.38 to 0.98); head controls 0.906 and 0.756.

No host fallback runs in a measured pass: every op in the profiles is a device op; the only host work per call is the
three input copies and the two readbacks.


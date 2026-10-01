# Stage 3: optimized decoder layer (encoder mapping)

The plugin's optimize checklist (op-topology audit, a search table for every dominant matmul, dtype and fidelity
choices with real-weight evidence, roofline vs device vs end-to-end reconciliation) is mapped to the prefill path,
the only path this encoder runs.

## Op-topology audit (layer 0, p150, from `../functional_decoder/tracy/layer0/prefill_perf_report.csv`)

One 128-token layer pass: 24 device ops, 1.609 ms; of that, 52 us (tilize + typecast of the fp32 test input) are
test-harness ops, so the layer itself is 1.557 ms. Exact rows:

| op | time | share | cores | notes from tt-perf-report |
|---|---|---|---|---|
| QKV matmul 128x4096x6144 (bf16 x bf16, HiFi4) | 263 us | 16.4 % | 32 | in0_block_w = 1, "place input 0 in L1"; largest single op |
| MLP w1 matmul 128x4096x12288 (bfp8, HiFi2) | 249 us | 15.5 % | 32 | DRAM 43 %, FLOPs 58 % |
| MLP w3 matmul 128x4096x12288 | 249 us | 15.5 % | 32 | same |
| MLP w2 matmul 128x12288x4096 | 228 us | 14.1 % | 32 | DRAM 45 %, FLOPs 64 % |
| wo matmul 128x4096x4096 (HiFi4) | 142 us | 8.8 % | 32 | "Increase grid size (currently using 32); HiFi2 is sufficient for BFP8" |
| NlpCreateHeads | 81 us | 5.0 % | 4 | |
| RMSNorm (attention input) | 65 us | 4.1 % | 4 | |
| RMSNorm (MLP input) | 65 us | 4.0 % | 4 | |
| NLPConcatHeads | 52 us | 3.2 % | 4 | |
| RoPE q, k | 47 + 17 us | 3.9 % | 110 | |
| SDPA (128 tokens) | 22 us | 1.3 % | 64 | |
| residual adds, q/k norms, typecasts, paged fill cache | 72 us | 4.5 % | 32 to 110 | two dead `paged_fill_cache` writes per layer (KV is never read back) |

Matmuls are 70 percent of the layer and run on 32 of 110 cores: `model_config.find_prefill_grid` is capped at
8x8 and at M = 128 tokens (4 tile rows) it yields a 4x8 grid. The two RMSNorms and the head reshapes run on 4
cores (17 percent of the layer). At 1024 tokens the matmuls use 64 cores at 79 to 90 percent FLOPs utilization.

## Dtype and fidelity search (real weights, real texts)

Policy definitions and the full result table, including the decision-agreement gate, are in
`../datatype_sweep/README.md`. The bfp4 MLP policy fails both the cosine gate and the decision-agreement gate on
real-weight evidence (79 percent agreement). The accuracy policy (bf16 attention weights, HiFi4 attention math) is
the only candidate above 98 percent agreement on confident decisions; the stock-default policy (bfp8 attention,
HiFi2) is 6 percent faster at 97.3 percent.

## Matmul geometry and small-grid ops

`tests/grid_experiment.py` patches `ModelArgs.find_prefill_grid` (no framework edit; the prefill matmul program
configs call it lazily, `model_config.py` `mlp1_3_grid`, `mlp2_grid` and the QKV / wo config lambdas) and times
layer 0 at 128 tokens, eager, 20 repeats, PCC against the HF fp32 layer reference. Results:
`geometry_experiment.json` (cap mode) and `geometry_experiment_forced.json` (force mode).

Cap mode (keep the divisibility rule, raise the 8x8 cap to 8x10, 8x11 and 10x11): every cap still yields 4x8.
M = 128 tokens is 4 tile rows, and the column tile counts (4096 / 32 = 128, 6144 / 32 = 192, 12288 / 32 = 384)
have no divisor between 9 and 11. Layer time 1.608 to 1.616 ms, PCC 0.99979 for all four caps.

Force mode (return the grid unconditionally, uneven per-core N):

| grid | layer ms p50 | layer ms min | PCC vs HF fp32 |
|---|---|---|---|
| 4x8 (stock) | 1.656 | 1.629 | 0.99979 |
| 4x9 | 1.631 | 1.607 | 0.99979 |
| 4x10 | 1.650 | 1.609 | 0.99979 |
| 4x11 | 1.653 | 1.623 | 0.99979 |

The spread is under 2 percent, the same as the run-to-run spread of the stock grid between the two runs (1.616 vs
1.656 ms). The profile showed the 128-token matmuls at 43 to 45 percent DRAM and 58 to 64 percent FLOPs utilization
on 32 cores; adding 4 to 12 cores on the N axis does not move the time, so core count is not the limit at this M.
More rows on M are impossible at 4 tile rows and more than 11 columns do not exist on the 11x10 compute grid, so the
grid lever is exhausted without kernel-level changes (a different in0 block or K split), which is outside this port.
The two RMSNorms and the head reshapes (17 percent of the layer on 4 cores) use fixed framework configs and were not
changed. The optimization that did land in this stage is the precision policy choice above.

## Reconciliation (roofline vs device vs end to end)

| padded length | per-layer device time (layer ops only) | 36-layer bound | measured end to end (batch 1, accuracy) | gap |
|---|---|---|---|---|
| 128 | 1.557 ms | 56.1 ms | 57.6 ms | +2.8 % |
| 1024 | 4.519 ms | 162.7 ms | 170.5 ms | +4.6 % |

(The earlier text quoted 1.609 and 4.692 ms per pass, which included the test harness's input tilize and typecast.)
The remaining 1.5 to 8 ms are the token embedding, the readback of the pre-norm residual (1 to 8 MB) and the host
norm; dispatch is hidden (eager and traced execution measure the same, `../fused_decoder/README.md`). DRAM roofline
for the modeled ops is 23.6 percent, so the headroom is in kernel efficiency and core count, not in the host.

## Watcher

Stage 1's watcher run (`TT_METAL_WATCHER=10`) covers these ops; no watcher errors.

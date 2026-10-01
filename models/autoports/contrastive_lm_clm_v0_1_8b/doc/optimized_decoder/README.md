# Stage 3: optimized decoder layer (encoder mapping)

The plugin's optimize checklist (op-topology audit, a search table for every dominant matmul, dtype and fidelity
choices with real-weight evidence, roofline vs device vs end-to-end reconciliation) is mapped to the prefill path,
the only path this encoder runs.

## Op-topology audit (layer 0, p150, from `../functional_decoder/tracy/layer0/prefill_perf_report.csv`)

24 device ops per layer pass. Device time per layer pass: 1.61 ms at 128 tokens, 4.69 ms at 1024 tokens.

| op (128-token pass) | time share | reported utilization |
|---|---|---|
| MLP w1 and w3 matmuls, 128x4096x12288, bfp8 x bf16, HiFi2 | 2 x 0.25 ms | DRAM 43 %, FLOPs 58 %, 64 cores |
| MLP w2 matmul, 128x12288x4096 | 0.23 ms | DRAM 45 %, FLOPs 64 % |
| QKV matmul 128x4096x6144 (bf16, HiFi4 in the accuracy policy) | 0.26 ms | DRAM 39 %, FLOPs 55 % |
| wo matmul 128x4096x4096 | 0.14 ms | DRAM 47 %, FLOPs 68 % |
| SDPA, RMSNorm x2, RoPE x2, create heads, residual adds, tilize | remainder (about 0.45 ms) | |

Matmuls are 70 % of the layer time. At 128 tokens they are neither DRAM- nor compute-saturated: `tt-perf-report`
advises "Increase grid size (currently using 64)" on every large matmul (the chip has 110 worker cores;
`model_config.find_prefill_grid` caps prefill grids at 8x8 and carries a TODO for Blackhole) and "HiFi2 is
sufficient for BFP8 multiplication" on the HiFi4 attention matmuls.

## Dtype and fidelity search (real weights, real texts)

Candidates were run as full-encoder fidelity and latency sweeps (`tests/run_fidelity.py`, `tests/bench_encoder.py`,
aggregated by `tests/datatype_sweep.py` into `../datatype_sweep/`):

| policy | attention weights | MLP weights / math | cosine vs HF fp32 mean / min | 128-token latency |
|---|---|---|---|---|
| accuracy (stock) | bf16, HiFi4 | bfp8, HiFi2 | 0.99909 / 0.99588 | 58.0 ms |
| bfp8_attn | bfp8, HiFi4 | bfp8, HiFi2 fp16 acc | 0.99901 / 0.99379 | 54.2 ms |
| bfp8_attn_hifi2 | bfp8, HiFi2 | bfp8, HiFi2 fp16 acc | 0.99891 / 0.99310 | 53.9 ms |
| performance (stock) | bf16, HiFi4 | bfp4, LoFi | 0.98742 / 0.92386 | 48.9 ms |

The bfp4 MLP policy is 16 % faster but fails the fidelity gate (minimum cosine 0.924, head projection minimum
0.913): real-weight evidence vetoes it for an embedding model, where the pooled vector is the output. The bfp8
attention policies pass; see `../datatype_sweep/README.md` for the selection.

## Matmul geometry

Not swept. The MLP and attention prefill matmuls run on 64 of 110 cores because `find_prefill_grid` is capped at
8x8 for all architectures. Changing it is a `models/tt_transformers` framework change affecting every model on
Blackhole, which this port did not take on; the profile evidence above is the pointer for that work. Expected
upside from the report's utilization numbers: up to about 1.4x on the matmul share (about 25 % end to end) if the
grid scales.

## Reconciliation (roofline vs device vs end to end)

| padded length | per-layer device time (Tracy) | 36-layer bound | measured end to end (batch 1) | gap |
|---|---|---|---|---|
| 128 | 1.61 ms | 58.0 ms | 57.7 ms | 0 % |
| 1024 | 4.69 ms | 169 ms | 168.5 ms | 0 % |

The end-to-end latency equals the sum of device kernel time; dispatch, host work and the readback are hidden or
negligible. Eager and traced execution measure the same (`../fused_decoder/README.md`). DRAM roofline for the
modeled ops is 23.6 % (121 GB/s of 512 GB/s), so the remaining headroom is in kernel efficiency, not in the host.

## Watcher

Stage 1's watcher run (`TT_METAL_WATCHER=10`) covers these ops; no watcher errors.

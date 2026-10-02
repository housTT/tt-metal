# Stage 3: optimized decoder layer (encoder mapping)

The plugin's optimize checklist (op-topology audit, a search table for every dominant matmul, dtype and fidelity
choices with real-weight evidence, roofline vs device vs end-to-end reconciliation) is mapped to the prefill path,
the only path this encoder runs.

## Op-topology audit (layer 0, p150, from `../functional_decoder/tracy/layer0/prefill_perf_report.csv`)

One 128-token layer pass: 24 device ops, 1.609 ms; of that, 52 us (tilize + typecast of the fp32 test input) are
test-harness ops, so the layer itself is 1.557 ms (pass 2 of the four 128-token passes, rows 36 to 59; the four passes
agree within 0.3 percent). Shares below are of the 1.557 ms layer:

| op | time | share | cores | notes from tt-perf-report |
|---|---|---|---|---|
| QKV matmul 128x4096x6144 (bf16 x bf16, HiFi4) | 263 us | 16.9 % | 32 | hard-coded config `in0_block_w = 1`, out subblock 1x1 ("FIXME: optimize this config for prefill" upstream); "in0_block_w=1 is small" |
| MLP w1 matmul 128x4096x12288 (bfp8, HiFi2) | 249 us | 16.0 % | 32 | DRAM 43 %, FLOPs 58 % |
| MLP w3 matmul 128x4096x12288 | 249 us | 16.0 % | 32 | same |
| MLP w2 matmul 128x12288x4096 | 228 us | 14.6 % | 32 | DRAM 45 %, FLOPs 64 % |
| wo matmul 128x4096x4096 (HiFi4) | 142 us | 9.1 % | 32 | "Increase grid size (currently using 32); HiFi2 is sufficient for BFP8" |
| NlpCreateHeads | 81 us | 5.2 % | 4 | |
| RMSNorm (attention input) | 65 us | 4.2 % | 4 | |
| RMSNorm (MLP input) | 65 us | 4.2 % | 4 | |
| NLPConcatHeads | 52 us | 3.3 % | 4 | |
| RoPE q, k | 47 + 17 us | 4.1 % | 110 | |
| SDPA (128 tokens) | 22 us | 1.4 % | 64 | |
| residual adds, q/k norms, three typecasts (incl. 5.9 us bf16 to bfp8 before SDPA), two paged fill cache | 79 us | 5.0 % | 32 to 110 | the two `BF16 => BF16` casts and the two `paged_fill_cache` writes are dead for the encoder (`../fused_decoder/README.md`) |

Matmuls are 72.6 percent of the layer and run on 32 of 110 cores. The reason is not `find_prefill_grid` (see the
geometry section): the QKV config is hard-coded in `model_config.get_attn_qkv_program_config` (grid 8x10,
`per_core_M = 1`, `per_core_N = 24`), and the wo, w1/w3 and w2 configs come from `matmul_config` with
`find_prefill_grid(8, k_tiles) = (8, 8)` and `per_core_M = ceil(4 / 8) = 1`; with M = 4 tile rows and
`per_core_M = 1`, four of the eight grid rows are used, so every matmul lands on 4 x 8 = 32 cores. The two RMSNorms
and the head reshapes run on 4 cores (16.9 percent of the layer) because the interleaved kernels parallelize over
tile rows. At 1024 tokens the QKV and w2 matmuls switch to `MinimalMatmul` (80 and 64 cores, 71 and 85 percent
FLOPs) and wo, w1 and w3 use 64 cores at 79 to 90 percent FLOPs (rows 111 to 130).

## Dtype and fidelity search (real weights, real texts)

Policy definitions and the full result table, including the decision-agreement gate, are in
`../datatype_sweep/README.md`. The bfp4 MLP policy fails both the cosine gate and the decision-agreement gate on
real-weight evidence (79 percent agreement). The accuracy policy (bf16 attention weights, HiFi4 attention math) is
the only candidate above 98 percent agreement on confident decisions; the stock-default policy (bfp8 attention,
HiFi2) is 6 percent faster at 97.3 percent.

## Matmul geometry and small-grid ops

Three experiments were run; the first two are retracted and kept for the record.

1. `geometry_experiment.json` and `geometry_experiment_forced.json` (2026 Oct 1 22:57 and 23:11 UTC) patched
   `find_prefill_grid` on the `ModelArgs` instance. Every row has the stock PCC to 16 digits because the four program
   config getters are `lru_cache` methods keyed on the shared instance: after the first (stock) iteration they returned
   the cached stock configs, so nothing was measured. Both files carry a `retracted` note.
2. `geometry_experiment_class_patch.json` (23:45 UTC) patched the class before construction. Its force mode asked for
   `grid_size = (4, 11)`, which the framework reads as 4 columns and 11 rows (`compute_with_storage_grid_size` is
   (x, y)), so it tripped the row-count assertion in `matmul_config` (`k % (32 * grid_size[1]) == 0`) and never ran.
   The earlier conclusion drawn from it ("divisibility contract, admissible column counts 1, 2, 4, 8") was wrong on
   two counts: the assertion constrains the M split, not the N split, and the largest 128-token op (QKV) does not pass
   through `find_prefill_grid` at all. Review A2 caught both.
3. `tests/program_config_experiment.py` (2026 Oct 2 00:52 to 00:55 UTC, `program_config_experiment_<len>.json`)
   overrides the program-config getters on the `ModelArgs` instance (instance attributes shadow the cached class
   methods), records the configuration the layer consumed, and measures layer 0 eagerly (20 or 10 repeats with a
   device sync per call, so the times include dispatch and are comparable with each other, not with the Tracy device
   time) against the HF bf16 layer reference. Accuracy policy, batch 1.

| candidate | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| stock | 1.609 ms, PCC 0.99979 | 1.984 ms, PCC 0.99979 | 2.600 ms, PCC 0.99977 | 4.703 ms, PCC 0.99977 | 8.822 ms, PCC 0.99976 |
| QKV `in0_block_w` 2, out subblock 1x2 (stock 32-core footprint) | 1.577 ms (-2.0 %), PCC 0.99978 | | | | |
| QKV `in0_block_w` 4, out subblock 1x4 | 1.559 ms (-3.1 %), PCC 0.99979 | | | | |
| QKV `in0_block_w` 8, out subblock 1x4 | 1.576 ms (-2.1 %), PCC 0.99979 | | | | |
| QKV `per_core_N` 18 on 11 columns (uneven N split), `in0_block_w` 4 / 8 | 1.543 / 1.570 ms, PCC 0.030 / 0.031: wrong output | | | | |
| QKV `MinimalMatmul` 8x10 grid, M block 8 / 4 | 1.628 / 1.633 ms (+1.2 / +1.5 %), PCC 0.99978 | | | | |
| QKV `MinimalMatmul` 11x10 grid (stock is 8x10) | | 1.956 ms (-1.4 %) | 2.517 ms (-3.2 %) | 4.577 ms (-2.7 %) | 8.641 ms (-2.1 %); PCC unchanged |
| wo `per_core_N` 12 on 11 columns | 1.607 ms, PCC -0.002: wrong output | | | 4.516 ms, PCC 0.00002: wrong output | |
| wo `in0_block_w` 16 | 1.617 ms (+0.5 %), PCC 0.99979 | | | L1 overflow (`program.cpp:1875`) | |
| w1/w3 `per_core_N` 36 on 11 columns | 1.614 ms, PCC 0.000: wrong output | | | 4.355 ms, PCC -0.002: wrong output | |
| w1/w3 `in0_block_w` 16 | L1 overflow | | | L1 overflow | |
| w2 `per_core_N` 12 on 11 columns (`in0_block_w` 8 / 16) | 1.595 / 1.626 ms, PCC 0.000: wrong output | | | | |
| w2 `in0_block_w` 16 | 1.618 ms (+0.6 %), PCC 0.99974 | | | | |
| w2 `MinimalMatmul` 11x10 grid (stock is 8x8) | | 1.894 ms (-4.5 %) | 2.499 ms (-3.9 %) | 4.564 ms (-3.0 %) | 8.459 ms (-4.1 %); PCC unchanged |
| RMSNorm block-sharded on 8x4 cores (`to_memory_config`, sharded `rms_norm`, `sharded_to_interleaved`) | 1.572 ms (-2.3 %), PCC 0.99978 | 1.965 ms (-1.0 %) | 2.553 ms (-1.8 %) | `dataflow_buffer.cpp` error | error |
| RMSNorm block-sharded on 8x8 cores | | 1.969 ms (-0.8 %) | 2.603 ms (+0.1 %) | 4.682 ms (-0.4 %) | error |
| QKV 1D matmul with width-sharded L1 output (for a sharded create-heads) | circular-buffer config error (`circular_buffer_config.cpp:222`) | | | | |
| combination of the kept rows | 1.522 ms (-5.4 %), PCC 0.99978 | 1.900 ms (-4.2 %) | 2.427 ms (-6.7 %) | 4.415 ms (-6.1 %) | 8.300 ms (-5.9 %) |

Findings:

- A wider N split with `per_core_N` that does not divide the N tile count runs without any error and returns wrong
  numbers (PCC 0.00 to 0.03) on every shape tried. The multicast 2D matmul path in this tt-metal build does not
  handle the partial last block for these configurations, so "uneven per-core N" is not a usable lever and must be
  guarded against, not just avoided. Recorded with the consumed configs in the JSON files.
- `in0_block_w` 16 overflows L1 for the bf16-weight (wo) and the 12288-wide (w1/w3) matmuls; it fits for w2 and gains
  nothing.
- The kept levers are small and additive: a better QKV block shape at 128 tokens (3.1 percent of the layer), the
  11x10 `MinimalMatmul` grid for QKV and w2 above 128 tokens (2 to 4.5 percent each), and the block-sharded RMSNorm
  for 128 to 512 rows (1 to 2.3 percent). Together they take 5.4 to 6.7 percent off the layer at every bucket.
- Not reachable without a framework edit: `MinimalMatmul` for w2 at 128 tokens (`mlp.py` branches on `seq_len > 128`),
  a sharded `nlp_create_qkv_heads` (the matmul cannot produce the needed sharded output), and the dead typecast and
  fill-cache ops (`../fused_decoder/README.md`).

The kept rows are installed by `tt/encoder.py` (`_install_program_configs`, `_install_sharded_norms`; environment
toggles `CLM_PROGRAM_CONFIGS=0` and `CLM_SHARDED_NORM=0` restore the stock path; single-chip meshes only) and
validated end to end in stage 7 (`../optimized_full_model/README.md`, "Program-config overrides").

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

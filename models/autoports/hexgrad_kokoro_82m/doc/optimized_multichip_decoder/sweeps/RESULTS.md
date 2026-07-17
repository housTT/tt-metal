# Optimized-multichip-decoder — operation-topology audit + sweep results

All numbers are warmed **traced** decode (production path), 50 replays, real
Kokoro weights, (1,4) ring mesh on 4× Blackhole p300c. PCC is measured vs the
single-chip TTNN `OptimizedDecoder` (`vs_sc`, isolation bar 0.997) and vs HF
(`vs_hf`, bar 0.995). Reproduce: `sweeps/sweep_opt.py <variant...>`.

## 1. Operation-topology audit of the measured (stage-03 multichip) path

Per-layer op sequence (12 weight-tied AlbertLayers). Only the attention block
has collectives; embedding / norms / FFN are per-token on the 1/TP sequence shard.

| # | Op (per layer) | Notes / cost (stage-03 perf report, merged 4-dev) | Audit action |
|---|---|---|---|
| 1 | all_gather(residual, dim=2) | ~21 µs; feeds QKV; bf16 payload | try bfp8 payload; try fused AG+matmul; try persistent buffer |
| 2 | QKV matmul [512×768×576] | ~12 µs, 29% DRAM, DRAM-bound | packed already (stage-02); geometry set; fidelity sweep |
| 3 | nlp_create_qkv_heads | ~9 µs TM | fused op already |
| 4 | SDPA [1,3,512,64] | ~28 µs; full-seq attention | try exp_approx / program config |
| 5 | nlp_concat_heads | ~4 µs TM | fused op already |
| 6 | WO matmul [512×192×768] | ~7 µs, 30% DRAM, DRAM-bound | try fused matmul+RS |
| 7 | reduce_scatter(WO partial, dim=2) | ~30 µs; bf16 payload | try bfp8 payload; try fused matmul+RS; persistent buffer |
| 8 | +dense_b, +residual, attn LayerNorm | **norm ~20 µs on only 4 cores** (interleaved, M=4 tiles) | **shard the norm** (biggest lever); try fused residual |
| 9 | FF1 [128×768×2048] + gelu | ~33 µs, SLOW, 13.7% DRAM, DRAM-bound | geometry swept stage-03 (in0_block_w 8); BFP4/LoFi trial |
| 10 | FF2 [128×2048×768] | ~17 µs, 26% DRAM, DRAM-bound | BFP4/LoFi trial |
| 11 | +residual, full LayerNorm | **norm ~20 µs on only 4 cores** | **shard the norm**; try fused residual |

Repeated same-input matmuls: none beyond the already-packed QKV (Q/K/V share the
post-norm activation → one packed matmul, stage-02). Gate/up are a 2-matmul (FF1
+FF2) SwiGLU-less MLP, not a packable gate/up pair. Material collectives: 1 AG +
1 RS/layer (already the attention-only floor). Reshard/layout conversions in the
baseline: none in the layer (all DRAM-interleaved). Dominant finding: **LayerNorm
runs on 4 cores** because the interleaved multi-core norm only parallelises over
the M row-tiles (S/TP/32 = 4 @T=512), wasting the 110-core grid — this is the
single clearest defect and the primary optimization.

## 2. Family sweep (traced decode ms, PCC vs_sc / vs_hf)

Same-session comparisons (thermal-comparable). Baseline = stage-03 multichip
(all opts off).

| Variant | T=512 ms | T=128 ms | vs_sc@512 | vs_hf@512 | Verdict |
|---|---|---|---|---|---|
| **orig (stage-03 baseline)** | 2.565 | 1.957 | 0.99927 | 0.99834 | baseline |
| residual-fusion (interleaved) | 2.675 | 2.052 | 0.99938 | 0.99837 | ❌ ~4% slower (fused-norm kernel) |
| **sharded norm** | 2.398 | 1.757 | 0.99930 | 0.99824 | ✅ 6.5%/10.1%, PCC kept |
| fused-residual + sharded | 2.389 | 1.742 | 0.99903 | 0.99729 | ❌ negligible gain, lower PCC |
| ag_dtype=bfp8 | 2.527/2.62¹ | 1.970 | 0.99879 | 0.99891 | ❌ helps @512 only, regresses @128, erodes margin |
| rs_dtype=bfp8 | 2.667 | 2.036 | 0.99787 | — | ❌ slower (reduce-in-bfp8 + typecast op) AND lower PCC |
| sdpa exp_approx | 2.594 | 1.951 | 0.99927 | 0.99834 | ❌ latency-neutral |
| **sharded + persistent CCL (SELECTED)** | **2.318** | **1.679** | **0.99930** | 0.99824 | ✅ +2.5%/5% on top of sharded |
| sharded + LoFi | 2.375 | 1.745 | **0.99580** | 0.99209 | ❌ vs_sc < 0.998 bar; matmuls memory-bound (no gain) |
| sharded + BFP4 mlp | 2.370 | 1.757 | **0.96119** | 0.96228 | ❌ PCC collapse; memory-bound (no gain) |
| sharded + BFP4 attn | 2.317 | 1.675 | **0.98188** | 0.98093 | ❌ vs_sc < 0.998 bar; no latency gain |

¹ ag_bfp8 measured 2.527 in the CCL/SDPA session and 2.62 in the fused session
(thermal variance); either way it regresses T=128 (1.97 vs 1.76 selected) and
trims the PCC margin, so it is not selected.

## 3. Fused CCL+matmul (adapted attempt — see `fused_ccl_matmul.log`)

`all_gather_matmul_async` and `matmul_reduce_scatter_async` were attempted for the
attention AG+QKV and WO+RS fusions with the **real head-parallel layout** (each
device holds its own 3 heads: QKV input gathered over seq; WO input a full local
`[b,full_seq,192]` — NOT a dim-3 shard of a shared 192 — with `dense_w` `[192,768]`).
Adapted past every API/config error — rank-4 weight requirement, required
persistent intermediate/output buffers, and the matmul `out_block_w %
out_subblock_w` divisibility — until each op hit the same hard **op-contract
blocker**:

- `all_gather_matmul_async`: `TT_FATAL ... all_gather_async_attributes.dim == 3`
  — the fused op only gathers over the **feature/K dim (dim=3)**.
- `matmul_reduce_scatter_async`: `TT_FATAL ... reduce_scatter_params.dim == 3`
  — the fused op only reduce-scatters over the **feature dim (dim=3)**.

Kokoro's sequence-parallel design gathers the residual and scatters the WO partial
over the **sequence dim (dim=2)** (reconstruct/redistribute the full sequence for
attention while keeping the FFN + both LayerNorms collective-free). Neither fused
op can express a dim-2 (sequence) collective; adopting them would force feature-dim
tensor parallelism, which stage-03 already measured slower (the "Intermediate-TP
FFN" and "Replicated residual + all_reduce" rejections). The same-harness
**separate** WO matmul + `reduce_scatter(dim=2)` measured **38.7 µs/rep** (matches
perf-report WO ~7 µs + RS ~30 µs). **Rejected: hard `dim==3` op-contract limit
reached after a fully adapted, correctly-laid-out attempt — not a tunable failure.**

## 4. Selected policy

`OptConfig` defaults: `norm_sharded=True`, `persistent_ccl=True`,
`fuse_residual=False`, `ag_dtype=rs_dtype=bf16`, `sdpa_exp_approx=False`; precision
policy inherited (bf16 act / BFP8 weights / HiFi2 / fp32-dest-acc). Headline (same
session): **orig 2.565/1.957 ms → selected 2.318/1.679 ms = 1.11×/1.17×
(9.6%/14.2% faster)**, PCC preserved (worst vs single-chip 0.99802, vs HF 0.99694
over the full T=8..512 / batch 2..32 / masked / non-aligned sweep).

## 5. Artifacts

- `sweep_opt.py` (+ `sweep_results.json`), logs: `sweep_ccl_sdpa.log`,
  `sweep_norm.log`, `sweep_persist_prec.log`, `sweep_final_confirm.log`,
  `sweep_headline.log`.
- `probe_fused_ccl_matmul.py` + `fused_ccl_matmul.log` (adapted-attempt evidence).

# Kokoro-82M — Optimized Multichip Decoder work log (stage 04)

Branch `agentic-research/hous/kokoro-82m-p150`. Runtime env: `TT_METAL_HOME` +
`PYTHONPATH` to the dev checkout (project memory `tt-metal-dev-env`). Hardware:
4× Blackhole p300c, `(1,4)` ring mesh (`FABRIC_1D_RING`/`Topology.Ring`).

## 0. Baseline + hardware

- Start from completed stage-03 `tt/multichip_decoder.py` (2.565/1.957 ms traced
  decode @T=512/128, PCC vs single-chip 0.99803, vs HF 0.99686). `tt-smi -ls` →
  4× p300c healthy.
- New module `tt/optimized_multichip_decoder.py::OptimizedMultichipDecoder` = a
  copy of the stage-03 module with an `OptConfig` of tunable knobs (defaults were
  set to the selected optimized policy after the sweeps).

## 1. Operation-topology audit (sweeps/RESULTS.md §1)

Per-layer op sequence read from the code + stage-03 `tt-perf-report`. Findings:
only the attention block has collectives (1 AG + 1 RS/layer = the attention-only
floor); QKV already packed (stage-02); no other repeated same-input matmuls; no
reshards in the baseline layer. **Dominant defect: LayerNorm runs on only 4 cores**
(~20 µs) because the interleaved multi-core norm parallelises over the M row-tiles
(S/TP/32 = 4 @T=512). That is the primary optimization target.

## 2. Family sweep (traced decode; sweeps/sweep_opt.py; RESULTS.md §2)

Measured coherent families in same-session A/Bs vs the stage-03 baseline:

- **Sharded LayerNorm** (block-shard [M,H] to a (8, m_tiles) L1 grid, 32 cores
  @T=512): 2.565→2.398 @512, 1.957→1.757 @128, PCC preserved (vs_sc 0.9993).
  ✅ SELECTED.
- **Persistent CCL buffers** (preallocated AG out + RS interm/out, cached by
  shape): on top of sharded, 2.398→2.318 @512, 1.757→1.679 @128, identical PCC.
  ✅ SELECTED (OPT-009).
- Residual-add fusion into the norm (`residual_input_tensor`): ~4 % SLOWER
  interleaved, negligible+lower-PCC when sharded. ❌.
- CCL payload bf16→bfp8: AG marginal @512 / regresses @128 + trims PCC; RS slower
  + lower PCC. ❌ (launch-bound: the typecast op ≈ the payload saved).
- SDPA `exp_approx_mode`: latency-neutral. ❌.
- LoFi / BFP4-mlp / BFP4-attn (real weights): no latency gain (matmuls
  memory-bound) and PCC fails the 0.998 vs-single-chip bar (LoFi 0.9958, BFP4-attn
  0.9819) or collapses (BFP4-mlp 0.9612). ❌ — confirms the stage-02 real-weight
  finding on the multichip shapes.

## 3. Fused CCL+matmul (adapted attempt; probe_fused_ccl_matmul.py)

Attempted `all_gather_matmul_async` (AG+QKV) and `matmul_reduce_scatter_async`
(WO+RS) with the **real head-parallel layout** (each device its own 3 heads: WO
input a full local `[b,full_seq,192]`, `dense_w` `[192,768]`). Adapted past every
API/config error — rank-4 weight requirement, required persistent
intermediate/output buffers, and the matmul `out_block_w % out_subblock_w`
divisibility — until BOTH ops hit the same hard **op-contract blocker**:
`AllGatherMatmulAsync requires dim==3` and `MatmulReduceScatterAsync requires
dim==3`. Both fused ops only operate on the **feature dim (dim=3)**; Kokoro
gathers/scatters over the **sequence dim (dim=2)** to keep the FFN + both norms
collective-free, so adopting them would force feature-dim TP (stage-03 measured
slower). Same-harness separate WO+RS = 38.7 µs/rep. ❌ rejected on a hard
`dim==3` op-contract limit after a fully adapted, correctly-laid-out attempt.
(An earlier probe iteration mis-sharded the WO input over dim=3, producing a
`48 vs 192` shape error; that was a probe bug, corrected here to reach the real
`dim==3` contract blocker.)

## 4. Selected policy + code

`OptConfig` defaults: `norm_sharded=True`, `persistent_ccl=True`,
`fuse_residual=False`, `ag_dtype=rs_dtype="bf16"`, `sdpa_exp_approx=False`.
`_layer_norm` dispatches interleaved vs block-sharded; `_norm_sharded_config`
computes the (gx=8, gy=m_tiles) block grid (subblock_w | block_w). `_all_gather_seq`
/ `_reduce_scatter_seq` lazily allocate + cache persistent buffers keyed by shape;
`deallocate(hidden_full)` is skipped when persistent (the buffer is reused).

## 5. Evidence (all on the target (1,4) ring mesh)

- **Tests** (`tests/test_optimized_multichip_decoder.py`, `logs/test_suite.log`):
  **33 passed** — PCC vs single-chip (≥0.997) + vs HF (≥0.995) over
  T∈{8,16,31,32,33,64,96,128,256,500,511,512}, decode T∈{16,32,64,128,500,511,512},
  batch {2,4,8,32}, IPA, masked non-aligned, component (embedding + AlbertLayer),
  determinism, stateless, stress; plus `test_selected_optimizations_active`.
- **gen_evidence.py** (`pcc_results.json`, `perf_summary.json`,
  `logs/gen_evidence.log`): worst PCC vs single-chip **0.99802**, vs HF **0.99694**.
  Before/after (same session): decode 2.565→2.318 @512 (1.11×), 1.957→1.679 @128
  (1.17×). Eager prefill regresses (host-dispatch-bound, non-production — see README).
- **tt-perf-report** (`tracy/{decode,prefill}_perf_report.txt`, advice enabled):
  LayerNorm now **32 cores @ ~10 µs** (was 4 @ ~20 µs); norm family 19 %→~13 %.
  Matmul 36 %, SDPA 15 %, RS 14 %, AG 9.5 %. Modeled roofline ~6.9 %
  (launch/movement-bound — precision doesn't help; advice not adopted, carried
  from stage-02/03).
- **Watcher** (`watcher/watcher.log`, ETH-disabled): 0 fault/assert/tripped/overflow,
  min stack 1348 B free, 8 tests passed.
- **Fallback audit** (`fallback_audit.txt`): traced replay pure TTNN; collectives'
  `from_torch` is one-time cached persistent-buffer setup; sharded-norm reshards
  are on-device (measured net win).

## 6. Context contract

`../context_contract.json → optimized_multichip_decoder`: mesh (1,4), TP=4, context
512 (advertised = supported, `capability_reduction=false`). Added persistent CCL
buffers ~1.7 MB/device/shape (DRAM, rounding error). **Inter-layer residual
contract recorded**: sequence-sharded `[b,1,S/TP,H]` DRAM-interleaved bf16, no
layer-to-layer collective; sharded-norm reshards are layer-internal.

## 7. Stage review + commit

- `$stage-review` (fresh xhigh subagent, read-only): initial verdict
  **more-work-needed** — one P2: the fused `matmul_reduce_scatter_async` (WO+RS)
  rejection was not earned because the probe **mis-sharded** the WO input over dim=3
  (splitting the local 192 into 48/device), producing a self-inflicted
  `width=48 height=192` shape error rather than a real op-contract limit.
- Remediation: corrected the probe to the real head-parallel layout (per-device
  full local `[1,1,512,192]` via ReplicateTensorToMesh, `dense_w [192,768]`, dim=2
  scatter, persistent buffers) + added a same-harness separate WO+RS baseline
  (38.7 µs/rep). Re-ran on hardware: adapted past the shape error AND a matmul
  `out_block_w % out_subblock_w` config error, then reached the genuine hard
  blocker `MatmulReduceScatterAsync requires dim==3` — symmetric with the AG+QKV
  `AllGatherMatmulAsync requires dim==3`. Updated RESULTS.md §3 / README / work_log.
- `$stage-review` re-review: **clean-pass**, no required work. (Other Concerns /
  Hard-Check Gaps — PCC_BAR_SC=0.997 test gate vs 0.998 contract [all values clear
  0.998], ag_bfp8-in-persistent-family gap, all-off==stage-03 A/B assumption — are
  non-blocking and acknowledged.)
- tt-metal branch `agentic-research/hous/kokoro-82m-p150`. Stage commit SHA
  `617fe1dd00fd1a3c6568e7c04e6bc6fde8f6b319` (stage-owned files only:
  tt/optimized_multichip_decoder.py, tests/test_optimized_multichip_decoder.py,
  tests/test_perf_optimized_multichip.py, doc/optimized_multichip_decoder/,
  doc/context_contract.json; pre-existing unrelated .agents/* excluded). Not pushed.
  (Pre-commit isort/black reformatted the sweep/test/module files; cosmetic only.)

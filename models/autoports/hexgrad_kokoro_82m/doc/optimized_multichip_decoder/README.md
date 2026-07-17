# Kokoro-82M — Optimized Multichip Decoder (TTNN, TP=4)

Stage 04 (optimized-multichip-decoder) for `hexgrad/Kokoro-82M` on **4× Blackhole
p300c** (`ClusterType.P300_X2`, a physical 4-ring exposed as a `(1, 4)` mesh).
Implements `models/autoports/hexgrad_kokoro_82m/tt/optimized_multichip_decoder.py`
(`OptimizedMultichipDecoder`), which starts from the stage-03
`tt/multichip_decoder.py` and optimizes it **in place** on the target mesh.

Target component (unchanged from stages 01–03): Kokoro's only attention-transformer,
`plbert` = HF `AlbertModel` — 12 weight-tied `AlbertLayer`s, hidden 768, 12 heads,
head_dim 64, intermediate 2048, vocab 178, max ctx 512. **Bidirectional,
non-autoregressive, no KV cache, no MoE.** "Decode" = the full-sequence encode
captured into a TTNN trace and replayed for a fixed `(batch, padded_seq_len)`;
prefill and decode are the same computation.

## What changed vs stage-03 (and what did not)

The stage-03 parallelization is preserved **exactly**: TP=4 head-parallel
attention (fractured QKV/WO, 3 heads/device) + sequence-parallel FFN (replicated
FFN weights) + **sequence-sharded residual `[b,1,S/4,H]`**, with **1 all_gather +
1 reduce_scatter per layer** and the inherited precision policy (bf16 act / BFP8
weights / HiFi2 / fp32-dest-acc). `PrecisionPolicy` and the op-builder helpers are
imported from the optimized decoder so the stages cannot numerically drift.

Two optimizations were **selected** (both are measured wins that preserve PCC):

1. **Block-sharded L1 LayerNorm.** The stage-03 interleaved `layer_norm`
   parallelised only over the M row-tiles (`S/TP/32` = 4 tiles @T=512), so it ran
   on **~4 cores at ~20 µs** and was ~19 % of traced decode. Each in-layer norm is
   now block-sharded to an L1 `(8, m_tiles)` grid (32 cores @T=512) via
   `LayerNormShardedMultiCoreProgramConfig`, with an `InterleavedToSharded` in and
   `ShardedToInterleaved` out. LayerNorm dropped to **~10 µs on 32 cores**; the
   whole norm family (norm + both reshards) is ~13.2 % of decode.
2. **Persistent CCL buffers.** The per-layer `all_gather` and
   `reduce_scatter_minimal_async` now use preallocated (cached-by-shape) output /
   intermediate buffers, removing per-call collective-output allocation from the
   repeated attention collectives.

Everything else in the layer is unchanged.

## Result summary (warmed, real weights, same session)

Baseline = **stage-03 multichip** (`OptimizedMultichipDecoder` with all opts off,
== `tt/multichip_decoder.py`). Traced decode is the production path.

| Metric | stage-03 multichip | **optimized** | Speedup |
|---|---|---|---|
| Traced warmed decode T=512 | 2.565 ms | **2.318 ms** | **1.11× (9.6 %)** |
| Traced warmed decode T=128 | 1.957 ms | **1.679 ms** | **1.17× (14.2 %)** |
| Decode T=512 vs single-chip | 2.949 ms | — | 1.27× (31.8 % eff) |
| Decode T=128 vs single-chip | 2.461 ms | — | 1.47× (36.7 % eff) |
| Worst PCC vs single-chip TTNN | 0.99803 | **0.99802** | preserved |
| Worst PCC vs HF | 0.99686 | **0.99694** | preserved |

PCC is preserved over the full sweep: prefill T∈{8,16,31,32,33,64,96,128,256,500,
511,512}, traced decode T∈{16,32,64,128,500,511,512} (incl. non-tile-aligned),
batch {2,4,8,32}, real IPA sentences, masked variable-length batches. All 33
correctness tests pass (`tests/test_optimized_multichip_decoder.py`).

### Warmed prefill note (host-dispatch-bound, non-production)

Eager (untraced) `prefill_forward` on the mesh is **host-dispatch-bound**: its
wall-clock (stage-03 ~5.6 ms → optimized ~7.3–8.4 ms) is dominated by Python
op-dispatch, not device time — the tracy device-op window for prefill is identical
to decode (same `_encode`). The optimizations trade a few extra host-dispatched
ops (the sharded-norm reshards) for device efficiency, which **only** shows up in
the traced path. The production path is traced decode; the traced prefill equals
the traced decode number (2.318 ms @T=512), i.e. also improved vs stage-03's
traced 2.565 ms. The eager-prefill wall time is reported honestly but is not the
optimization target (same framing as stage-03).

## Performance accounting (traced decode, T=512)

`tt-perf-report` (merged 4-device, `tracy/decode_perf_report.txt`): Matmul 36.4 %,
SDPA 14.9 %, ReduceScatter 13.6 %, **LayerNorm (block_sharded) 10.2 %**, AllGather
9.5 %, NlpCreateHeads 4.7 %, BinaryNg 4.1 %, NLPConcatHeads 2.0 %,
InterleavedToSharded 1.6 % + ShardedToInterleaved 1.4 % (the sharded-norm
reshards). LayerNorm rows now show **32 cores @ ~10 µs** (was 4 cores @ ~20 µs).

- **Roofline:** unchanged from stage-03 — weights (BFP8, ~3.9 MB/device) + tiny
  activations over 4× p300c aggregate DRAM BW → modeled ~6.9 %. The model is
  **launch/movement-bound**, not DRAM-bandwidth-bound (matmul DRAM util 13–31 %),
  which is *why* op-count/collective/norm-core optimizations help and precision
  (memory-bound matmuls) does not.
- **Device vs end-to-end:** traced decode end-to-end 2.318 ms @T=512; the device
  op sum in the signposted window matches within trace-replay overhead (no host
  gap in the replay — persistent buffers + captured trace).
- The perf-report advice "place input 0 in L1" and "use HiFi4" is **not adopted**
  (carried from stage-02/03): launch/movement-bound not DRAM-bound, and HiFi4 adds
  latency for negligible PCC. See §Rejected.

## Rejected / compared alternatives (all measured — `sweeps/RESULTS.md`)

| Candidate | Result | Verdict |
|---|---|---|
| Residual-add fused into interleaved norm (`residual_input_tensor`) | 2.675 vs 2.565 ms @512 | ❌ fused-norm kernel ~4 % slower |
| Residual-fusion + sharded norm | 2.389 ms, vs_hf 0.99729 | ❌ negligible gain, lower PCC |
| all_gather payload bf16→bfp8 | helps @512, regresses @128 (1.97 vs 1.68), trims PCC margin | ❌ launch-bound: typecast op ≈ payload saved |
| reduce_scatter payload bf16→bfp8 | 2.667 ms, vs_sc 0.99787 | ❌ slower (reduce-in-bfp8 + typecast) AND lower PCC |
| SDPA `exp_approx_mode=True` | 2.594 ms | ❌ latency-neutral |
| **Fused all_gather_matmul (AG+QKV)** | `TT_FATAL: AllGather dim==3` | ❌ fused op gathers feature-dim only; we gather sequence (dim=2) |
| **Fused matmul_reduce_scatter (WO+RS)** | `TT_FATAL: ReduceScatter dim==3` | ❌ fused op scatters feature-dim only; we scatter sequence (dim=2) |
| LoFi (matmul fidelity) | vs_sc **0.9958 < 0.998**, no latency gain | ❌ memory-bound matmuls; PCC fail (real weights) |
| BFP4 MLP weights | vs_sc **0.9612**, no latency gain | ❌ PCC collapse; memory-bound |
| BFP4 attention weights | vs_sc **0.9819 < 0.998**, no latency gain | ❌ PCC fail; memory-bound |

The two fused CCL+matmul rejections are **fundamental**: with the real
head-parallel layout (each device its own 3 heads), and after adapting past the
rank-4-weight, persistent-buffer, and matmul out_subblock divisibility errors,
both ops hit a hard `dim==3` assert — `all_gather_matmul_async` gathers only the
feature dim and `matmul_reduce_scatter_async` scatters only the feature dim, while
Kokoro deliberately gathers/scatters over the **sequence dim (dim=2)** to keep the
FFN and both norms collective-free (stage-03 already measured feature-dim families
slower). The same-harness separate WO+RS measured 38.7 µs/rep. See
`sweeps/probe_fused_ccl_matmul.py` / `fused_ccl_matmul.log`.

## Inter-layer residual layout contract (for full-model bringup)

**UNCHANGED and must be preserved:** the decoder layer input *and* output is the
sequence-sharded residual `[b, 1, S/TP, H]`, DRAM-interleaved bf16. There is **no
gather / reshard / all-reduce between decoder layers**; each layer internally does
exactly 1 `all_gather` (residual→full seq, feeds QKV) + 1 `reduce_scatter` (WO
partial→seq shard). The sharded-norm `InterleavedToSharded`/`ShardedToInterleaved`
conversions are **internal to a layer** and restore the interleaved sequence-shard
contract before the layer boundary. Full-model bringup must stack layers on this
`[b,1,S/TP,H]` contract and must **not** insert a layer-to-layer all_gather.
Recorded in `../context_contract.json → optimized_multichip_decoder`.

## Correctness / capability (preserved)

- **PCC vs single-chip ≥ 0.998** and **vs HF ≥ 0.995** at every tested length
  (worst 0.99802 / 0.99694); per-layer-kind component PCC (embedding + one
  `AlbertLayer`) validated separately.
- **Non-aligned logical lengths supported** — public API takes any length 1..512;
  internal 128-multiple padding + masking; sliced at the boundary. Sharded-norm
  grid is legal for every tile-padded local-seq (m_tiles 1..4) and batch (m up to
  1024).
- **Stateless** (`decode == prefill`), **determinism**, **repeated traced replay**
  across revisited shapes (bit-identical) — all validated on the mesh.
- **Watcher clean** (`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`): 0
  fault/assert/tripped/overflow markers, min stack 1348 B free
  (`watcher/watcher.log`). ETH excluded (fabric ETH kernels overflow the
  ACTIVE_ETH config buffer — infra limit, not model code; same as stage-03).
- **Fallback audit clean** (`fallback_audit.txt`): traced replay is pure TTNN; the
  only `from_torch` in the collectives is one-time cached persistent-buffer setup
  during warm-up (not per-replay), and the sharded-norm reshards are on-device.
- **Context contract**: advertised = supported = 512, **no reduction**
  (`capability_reduction=false`); optimization is latency-only.
- KV cache / paged cache / current-position / MoE: **N/A** (non-autoregressive).

## Reproduce

```bash
ENV="TT_METAL_HOME=/home/ttuser/dev/tt-metal PYTHONPATH=/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal"

# Correctness suite (33 tests):
env $ENV python -m pytest models/autoports/hexgrad_kokoro_82m/tests/test_optimized_multichip_decoder.py -v

# PCC + before/after (stage-03 baseline vs optimized) + single-chip context:
env $ENV python models/autoports/hexgrad_kokoro_82m/doc/optimized_multichip_decoder/gen_evidence.py

# Family/knob sweep (opens the ring mesh once):
env $ENV python models/autoports/hexgrad_kokoro_82m/doc/optimized_multichip_decoder/sweeps/sweep_opt.py orig selected

# Profiling (one window/run, KOKORO_PERF_ITERS=2):
env $ENV KOKORO_PERF_ITERS=2 python -m tracy -r -p -v -m pytest \
  models/autoports/hexgrad_kokoro_82m/tests/test_perf_optimized_multichip.py -k decode   # or prefill

# Watcher (separate from profiler; ETH excluded):
env $ENV TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 python -m pytest \
  models/autoports/hexgrad_kokoro_82m/tests/test_optimized_multichip_decoder.py \
  -k "512 or determinism or stress or masked or component or stateless or optimizations"
```

## Artifacts

- `pcc_results.json`, `perf_summary.json`, `logs/{gen_evidence,test_suite,tracy_*}.log`
- `sweeps/RESULTS.md` (operation-topology audit + full family sweep) + `sweep_opt.py`,
  `probe_fused_ccl_matmul.py`, sweep logs, `fused_ccl_matmul.log`, `sweep_results.json`
- `tracy/{decode,prefill}_perf_report.txt` (advice-enabled tables), `*_perf_report.csv`
- `watcher/watcher.log` (clean), `fallback_audit.txt`, `work_log.md`

## Limitations / deviations

- Single-utterance decode speedup over stage-03 is modest (~10–14 %) because the
  encoder is tiny and launch/movement-bound; the remaining device time is Matmul
  (memory-bound) + SDPA + the two attention collectives, all near their floor.
- Eager (untraced) prefill wall-clock regresses (host-dispatch-bound, non-production;
  the extra sharded-norm reshard dispatches inflate the Python loop). The traced
  path — the production path — improves.
- Precision reductions (LoFi/BFP4) do not help (matmuls are memory-bound) and fail
  the PCC bars on real weights; the inherited BFP8/HiFi2 policy is kept.

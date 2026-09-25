# DeepSeek-V4-Flash — Multi-chip expert-parallel MoE (4-chip QuietBox)

**Goal:** get to a multi-chip expert-parallel MoE and a tt-inference-server command to test it.
**Result:** built, wired end-to-end, and validated correct on the real 4-chip (1,4) mesh.

> **UPDATE (2026-07-15) — COALESCED MoE is now the default: 1.69 tok/s single-user (best measured).**
> The first per-expert mesh MoE measured 1.25 tok/s (slower than single-chip 1.39) because it streamed
> each expert as many tiny sharded `to_device` calls. Root-causing showed the bottleneck is **host→device
> DMA**, which parallelizes ~4× on the mesh (measured 6.35 → 24.6 GB/s aggregate) **but only with coalesced
> transfers**. Rewriting the MoE to stream 2 fused sharded transfers/expert, **device-concat** the routed set,
> and run **one batched matmul pair** cut ~209 ms/token → **591 ms/token = 1.69 tok/s** warm, correct
> (" Paris."). That is now the fastest single-user config: **1.35× over per-expert mesh, 1.22× over single-chip.**
> Levers ruled out along the way (all measured): Metal-Trace (MoE is DMA-bound, not dispatch-bound: 0.5 ms/layer;
> mesh attn 1.01×), expert-grouping-for-compute, single-chip transfer coalescing (size-independent at 6.35 GB/s).
> A 111-min "hang" during this work was **transient device state** (wedged chip, 101% one core) — recovered by
> `tt-smi -r`; the identical code then ran clean. Ladder (measured): recompute 0.15 → single-chip fp4 1.39 →
> mesh per-expert 1.25 → **mesh coalesced 1.69**. Micros: `demo/micro_{fp4_trace,dma_bw,trace_mesh,fp4_coalesce}.py`.
>
> Below is the original per-expert write-up (superseded on the perf numbers; the analysis still holds).

## What was built

Multi-chip expert-parallel (tensor-parallel-sharded) MoE, fully integrated into the fast decode path:

- **`tt/mla_v4_device.py`**
  - `_dev` now REPLICATES every resident attention/mHC/compressor/norm weight across the mesh
    (`ReplicateTensorToMesh`); `_host` reads a replicated result back from chip 0. This one change
    makes the whole non-MoE decode SPMD-correct on the mesh.
  - `Fp4ExpertCacheMesh`: host multi-device, TP-SHARDED bfloat4_b experts (gate/up column-sharded on
    the intermediate dim → `[H, I/C]` per chip; down row-sharded → `[I/C, H]` per chip). `ttnn.to_device`
    then streams each chip's 1/C slice over its own PCIe in parallel.
  - `moe_device_fp4_mesh`: per-chip I-slice routed experts + TP-sharded shared expert, accumulated
    LOCALLY per chip (the cross-chip reduction is linear), then a SINGLE `all_gather`+`sum` per layer.
- **`tt/fast_decode.py` / `LayerDeviceWeights`**: auto-detect a >1 mesh, shard the shared expert,
  mesh-aware head read.
- **`tt/generator_vllm.py`**: removed the `(1,1)` submesh drop — the fast path now keeps the FULL mesh
  vLLM hands it (e.g. `(1,4)`). `DEEPSEEK_V4_FORCE_SUBMESH=1` restores the old single-chip behaviour.

The tt-inference-server plugin already sets `FABRIC_1D` by default for the 4-device P300X2 config and
passes the full `(1,4)` mesh to the model, so no extra fabric flags are needed.

## Validation (real 4-chip mesh, standalone)

- Correctness gate: `"The capital of France is"` → `" Paris. The capital of France"` ✅
- `micro_fp4_tp.py` / `micro_fp4_tp2.py`: sharded MoE + `all_gather` correct; 4-chip MoE ≈ 381–403 ms/token.

## Measured warm decode, and why multi-chip doesn't help single-stream

| section (per token) | single-chip | 4-chip mesh |
|---|---|---|
| MoE | 561 ms | ~450 ms  (sharding helped modestly) |
| attn + mHC + compressor (replicated) | ~160 ms | ~296 ms  (got WORSE under mesh SPMD) |
| 43× `all_gather` collectives | 0 | fabric latency |
| **total** | **~720 ms (1.39 tok/s)** | **~800 ms (1.25 tok/s)** |

**Root cause:** single-stream batch=1 decode is **dispatch/collective-bound, not weight-bandwidth
bound.** Sharding the MoE weights 4 ways only trims MoE modestly (the bandwidth was never the limit);
the per-layer `all_gather` adds latency the single-chip path never paid; and the majority of the layer
(attention/mHC, which cannot usefully shard at batch=1) runs replicated and is actually **slower** under
4-device SPMD coordination (~160 → ~296 ms). Net: wash-to-slightly-worse. This corrects the REPORT_10
projection of ~3 tok/s, which wrongly assumed the decode was bandwidth-bound.

## tt-inference-server command (multi-chip path is now the default)

```bash
cd /home/ttuser/code/tt-inference-server && python3 run.py \
  --model DeepSeek-V4-Flash --workflow server --local-server \
  --tt-device p300x2 --no-auth --disable-trace-capture --disable-metal-timeout
```
Benchmark via `/v1/completions` (warm request after a cold prompt tilizes+shards the fp4 cache).

## Where the real multi-chip win is (not pursued here)

- **Batching / continuous batching (B>1):** the genuine multi-chip lever. The ~800 ms forward produces
  B tokens, so aggregate throughput scales with batch (projected ~10–20+ tok/s aggregate). Requires
  batched decode + `max_concurrency` > 1.
- **Mesh Metal-Trace of the replicated attention/mHC ops:** would remove the ~296 ms of 4-device SPMD
  dispatch overhead (the part that regressed on the mesh); MoE stays eager (data-dependent routing).

# Batched decode and indexed prefill for GPT-OSS 120B on P150x4

Date: 2026-09-10. Scope: `tt/multichip_decoder.py`, `tt/model.py`,
`tt/generator_vllm.py`. Hardware: 4x Blackhole (2x p300c), mesh (1, 4), TP=4.

## Problem

The 2026-09-10 latency sweep showed decode TPOT of 15.4 ms at one user and
506 ms at two users (2 tok/s/u, 4 tok/s aggregate). Two causes:

1. The MoE block ran a static per-user loop of the batch-1 expert graph for
   any decode batch above one, and vLLM padded every batch above one to 32
   because the model advertised only decode buckets `(1, 32)`. That is
   32 x 36 = 1152 sequential MoE graphs per token.
2. Prefill computed every 32-token group against the union of its experts
   with an expanded dense output over all 128 experts (zero-filled), so the
   expert matmuls showed `active=512/512` and the elementwise tail ran over
   `tokens x 128 experts`. Prefill ran at about 460 tok/s at every length.

## Changes

### Decode (`_ActiveExpertTPMLP`)

- `_run_indexed_slots_decode` for `1 < B <= 8`: gathers the `top_k * B`
  routed expert slots by index (duplicates allowed, static trace shape),
  applies each slot's routing score to its user's row through a one-hot
  selector before the linear down projection, reduces over slots once.
- `_run_grouped_decode` for `B > 8`: the padded 32-row decode tile is one
  token group; a union expert mask (`ttnn.sum` of dense routing weights over
  rows) drives both sparse matmuls, routing weights are applied before the
  down projection, one reduction over experts.
- Shared: two-op fused SwiGLU (`ttnn.mul` with input activation chains for
  clamp, alpha scale, sigmoid, and `+1`), down bias folded through the dense
  routing weights with one `[rows, 128] x [128, 2880]` matmul, dense routing
  built from the same fused router as batch 1 plus `ttnn.scatter`, per-width
  fused-router bias tensors.
- Weight layout: each rank's expert intermediate slice is padded 720 -> 768
  (cache suffix `_pad768`) so the packed gate/up split is a tile-aligned slice
  and the down contraction is 24 tiles. 46 tiles (736 padding) is rejected by
  the sparse matmul: every core of the rectangular grid must have work, so
  the gate/up grid is (6, 8) = 48 cores at one tile each.
- Decode trace buckets `(1, 4, 8, 32)` (`tt/model.py::decode_trace_buckets`,
  advertised per instance by the vLLM adapter and warmed/captured in order).

### Prefill

Shipping path (`prefill_indexed_experts=True`, prompts of 512+ tokens):
`_run_indexed_prefill` gathers each expert's routed tokens into per-expert
slabs and runs compact indexed sparse matmuls, replacing the packed
32-token-group path with its dense 128-expert expanded outputs. Design points
that were needed to make it work in serving, not just in a layer test:

- An expert's tokens fill `count // 512` full slabs of 512 rows plus one
  remainder slab whose height is a power-of-two tile multiple (32..256), so
  padding is at most one short slab per expert. Taller slabs overflow L1 in
  the sparse matmul (per_core_M x in0_block_w input tiles); 4096-row slabs
  crashed 16k-token prompts.
- Slabs of equal height form a group; a group is split into power-of-two
  chunks (largest first, at most 128 ids, the indexed mode's limit) with no
  padding slabs, and each chunk is one indexed sparse matmul pair. With
  routing-dependent shapes every prompt recompiled programs (32 s TTFT);
  with these shape classes the set is 5 heights x 8 sizes and stays cached.
  The slab buffer is rounded to 1/8 of its power-of-two octave (<= 12.5%
  waste; the earlier full power-of-two rounding plus padding slabs made the
  16k-token dispatch gather 2.5x larger than the slots).
- `warmup_indexed_prefill_shapes` compiles that set once on layer 0 during
  the adapter's prefill warmup (program caches key on shapes, not weights),
  including the per-group token gather.
- Prefill is untraced and host-synchronised, so the routing runs in torch:
  the router linear runs on device and its bf16 logits are read back (4 MB
  at 16k tokens, the same bytes as the padded index/score tiles); top-4,
  softmax, per-expert counts, slab layout, the stable argsort of slots and
  the gather-back columns are torch and are uploaded as four small tensors.
  A device `ttnn.sort` of the slots alone cost 5-18 ms per layer, and the
  device `ttnn.topk` path was no faster than the readback.
- Token rows are gathered straight into each group's slab stack with one
  tile-layout `ttnn.embedding` per group (no capacity-sized copy). Expert
  rows come back with one `[top_k, rows]` embedding gather, are multiplied
  by the `[1, top_k, rows, 1]` routing weights and reduced over `top_k`
  (`fast_reduce_nc`); the down bias is folded in through the dense routing
  weights. Below 512 tokens the packed path is used, since the per-layer
  sync costs more than it saves there.
- Matmul blocking is specific to this path (`_indexed_prefill_matmul_config`):
  gate/up `(in0_block_w, out_block_h, out_subblock_h, out_subblock_w)` =
  (30, 4, 2, 1), down (24, 8, 4, 2). With `out_block_h = 1` (the decode
  configs) the kernel re-reads every weight block once per slab tile row:
  16k-token gate/up went from 27.7 ms to 15.5 ms per layer and down from
  20.8 ms to 6.8 ms. Both matmuls are bound by the single in0 multicast
  sender of the 1D kernel (bf16 slabs: 425 MB per chip at 16k tokens); a
  bfp8 slab saved 3 ms on the matmul but cost 2 ms in the cast and lowered
  the PCC, so it is not used (`indexed_prefill_slab_dtype`).

Packed path improvements that remain in use below 512 tokens: 768 layout
(tile-aligned gate/up slices), pre-transposed gate/up bias, down bias folded
through the dense routing weights.

## Measurements

Layer 0 (sliding attention), TP4, traced decode, median of 5 x 20 replays
(`tests/test_multichip_batched_decode_perf.py`):

| batch | before (per-user loop) | after | per user |
|---:|---:|---:|---:|
| 1 | 0.384 ms | 0.383 ms | 0.383 ms |
| 2 | 0.961 ms | 0.411 ms | 0.205 ms |
| 8 | 3.551 ms | 0.863 ms | 0.108 ms |
| 32 | 14.030 ms | 1.326 ms | 0.041 ms |

Every batched row matches the batch-1 indexed path at PCC >= 0.99993.
Batch-1 acceptance against the single-chip baseline artifacts
(`test_real_weight_multichip_against_baseline_artifact`, TP4): prefill PCC
0.993 / 0.990, decode PCC 0.9992 / 0.9978 (thresholds 0.95).

Full layer-0 prefill on real token embeddings, one call with a host sync
(packed group-sparse vs experimental indexed), `test_indexed_prefill_moe`:

| tokens | packed | indexed | speedup | PCC |
|---:|---:|---:|---:|---:|
| 128 | 7.9 ms | 4.5 ms | 1.8x | 0.99984 |
| 1,024 | 53.8 ms | 8.0 ms | 6.7x | 0.99997 |
| 4,096 | 228.8 ms | 19.6 ms | 11.7x | 0.99995 |
| 16,384 | 912.5 ms | 61.7 ms | 14.8x | 0.99995 |

Per-stage times at 16,384 tokens (`GPT_OSS_120B_PREFILL_STAGES=1`, device
sync after every op, so the sum is above the pipelined 61.7 ms): gate/up
matmuls 15.5 ms, down matmuls 6.8 ms, gather-back + weights + reduce 7.3 ms,
token gathers 3.8 ms, all-reduce 3.8 ms, SwiGLU 3.2 ms, logits readback
3.0 ms, uploads 2.8 ms, down untilize 2.8 ms, gate/up slices 2.6 ms, gate/up
bias 4.3 ms, host layout 1.5 ms. The earlier version of this path measured
14.9 / 33 / 174 ms at 1k / 4k / 16k.

The packed numbers already include the 768 layout and bias fold (the 127-token
layer prefill in the acceptance test went from 14.4 ms to 9.2 ms).

Serving, `python -m vllm.entrypoints.openai.api_server` from this tree,
ISL 128 / OSL 128, `bench_client.py`, 4 x users prompts (max 64), zero
failures. "Before" is the 2026-09-10 sweep of the shipped container. v8 =
768 layout, batched decode, buckets (1, 4, 8, 32), packed prefill:

| users | TPOT before | TPOT v8 | decode tok/s/u v8 | decode-only aggregate v8 | TTFT before | TTFT v8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.4 ms | 15.3 ms | 65 | 65 | 0.28 s | 0.25 s |
| 2 | 506 ms | 21.9 ms | 46 | 91 | 0.60 s | 0.48 s |
| 4 | n/a | 21.9 ms | 46 | 183 | n/a | 0.95 s |
| 8 | ~590 ms | 32.6 ms | 31 | 245 | n/a | 1.9 s |
| 16 | ~590 ms | 61.7 ms | 16 | 259 | n/a | 3.7 s |
| 32 | ~590 ms | 53.2 ms | 18.8 | 602 | n/a | 7.5 s |

TTFT at high concurrency is the serialized queue of 128-token prefills (about
235 ms each); the plugin does not mix prefill and decode steps.

Long prompts, one user, OSL 64-128 ("before" is the 2026-09-10 sweep of the
shipped container; v10 = packed path with the 768 layout; v13 = indexed
prefill, warm shapes; cold = first prompt of that length after server start):

| ISL | TTFT before | TTFT v10 | TTFT v13 warm | TTFT v13 cold | TPOT v13 |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 2.15 s | 1.73 s | 0.79 s | 4.5 s | 15.4 ms |
| 4,096 | 8.86 s | 7.19 s | 2.47 s | 5.5 s | 15.6 ms |
| 16,384 | 35.6 s | 28.9 s | 10.8 s | 16.5 s | 15.9 ms |
| 32,768 | 71.7 s | 58.3 s | 23.0 s | 31 s | 16.4 ms |

1024-token prompts at concurrency (v13): 8 users TTFT 6.4 s (v8: 13.7 s),
16 users 12.4 s (27.4 s), 32 users 24.7 s (not measured before); decode TPOT
33 / 51 / 53 ms. TTFT at concurrency is still the serialized prefill queue.

## Remaining per-layer overheads at batch 32 (device profile, v5)

sparse matmuls 643 us (near DRAM roofline for ~81 experts), two unsharded
layernorms 82 us (sharding at a full tile was disabled during bring-up after a
nondeterministic trace), zero-fill of the expanded sparse outputs 56 us,
gate_up bias add 41 us, QKV matmul 44 us (19 us at batch 1, auto program
config), fused router 40 us.

## Reproduce

```bash
cd /home/ttuser/dev/gpt-oss-120b/tt-metal
SNAP=/home/ttuser/dev/gpt-oss-120b/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
env GPT_OSS_120B_BATCHED_DECODE_PERF=1 GPT_OSS_120B_BATCHED_DECODE_CHECK=1 GPT_OSS_120B_SNAPSHOT=$SNAP \
  scripts/run_safe_pytest.sh --run-all \
  models/autoports/openai_gpt_oss_120b/tests/test_multichip_batched_decode_perf.py -q -s -k sliding
bash /home/ttuser/dev/gpt-oss-120b/tti-release/openai_gpt_oss_120b/launch_autoport_server.sh
/home/ttuser/dev/gpt-oss-120b/benchmarks/perf_plan/run_conc_sweep.sh <out_dir> http://127.0.0.1:8000 128 128 1 2 4 8 16 32
```

Raw logs: `/home/ttuser/dev/gpt-oss-120b/benchmarks/perf_plan/`.

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

Shipping path: the packed 32-token-group union-mask path, now with the 768
layout (tile-aligned gate/up slices), pre-transposed gate/up bias, no per-call
bias transposes, and the down bias folded through the dense routing weights.

Experimental (`prefill_indexed_experts`, default off): `_run_indexed_prefill`
groups experts by their routed-token count rounded to a power-of-two tile
multiple, gives every expert a contiguous slab, and runs one compact indexed
sparse matmul pair per group (no zero-filled expanded output). It is exact
(full-layer PCC 0.99999 on real token embeddings, where one expert took 314 of
1024 tokens) and 2.5x / 3.3x faster per layer at 1024 / 4096 tokens in the
synced layer test. In the serving path it is not usable yet: its tensor
shapes depend on the routing of each layer and prompt, so the runtime created
over 13,000 programs and TTFT rose to about 32 s at 1024 tokens, and a
16k-token prompt hit an L1 clash. The path also syncs the host once per layer
to read expert counts. Two findings along the way: `ttnn.gather` returns
wrong values for a long single-stick index (worked around with a
`[rows, top_k]` index against a row-repeated table), and a uniform per-expert
slab sized by the largest count blows up memory under real routing imbalance.
The follow-up is to make the shapes static (fixed slab count per height
bucket with dummy slabs) or to use a device-side routed kernel that reads the
counts on device.

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
| 128 | 7.7 ms | 7.8 ms | 1.0x | 0.99999 |
| 1,024 | 54.3 ms | 22.1 ms | 2.5x | 0.99999 |
| 4,096 | 228.9 ms | 69.8 ms | 3.3x | 0.99999 |

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

Long prompts, one user, OSL 128, v10 (= v8 configuration; "before" is the
2026-09-10 sweep of the shipped container):

| ISL | TTFT before | TTFT v10 | prefill tok/s before | prefill tok/s v10 | TPOT v10 |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 2.15 s | 1.73 s | 475 | 593 | 15.5 ms |
| 4,096 | 8.86 s | 7.19 s | 462 | 570 | 15.6 ms |
| 16,384 | 35.6 s | 28.9 s | 461 | 567 | 16.1 ms |
| 32,768 | 71.7 s | 58.3 s | 457 | 562 | 16.6 ms |

Prefill is about 23% faster at every length from the 768 layout and the bias
fold alone; the remaining gap to the compute roofline is the dense
128-expert expanded output of the packed path (see the experimental indexed
prefill above).

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

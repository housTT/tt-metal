<!--
SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# OPTIMIZATION_LOG — DeepSeek-V4 TT-NN (Checkpoint 3)

Each entry: change → PCC re-validation (on Blackhole, vs HF reference) → expected perf effect + prior-art
reference. **Correctness (PCC) is proven here; actual speed is a Checkpoint-4 measurement only** (GOAL §3:
"Do not read tok/s off the simulator" / do not read perf from correctness runs).

## 1. Weight precision sweep — clamped SwiGLU MLP (shared expert)

Test: `tests/test_optimizations.py` (512 tokens, hidden 256). PCC re-checked after each dtype change.

| weight dtype | PCC vs HF ref | PCC ≥ 0.99 | decision |
|---|---|---|---|
| `bfloat16` (baseline) | 0.99993 | yes | correctness baseline |
| `bfloat8_b` | 0.99986 | **yes** | **safe** for this MLP; adopt |
| `bfloat4_b` | 0.98087 | **NO** | **unsafe** here — reject (or needs HiFi + per-tensor care) |

**Finding (the point of the re-validation gate):** dropping the shared-expert MLP weights to `bfloat4_b` breaks
the PCC bar (0.981 < 0.99). `bfloat8_b` holds. This matches the GOAL warning that precision drops are the usual
PCC killers, and the tt_transformers guidance to keep sensitive matmuls at higher precision while pushing
FF/MLP to `bfp8`/`bfp4` **only where PCC holds**.

**Expected perf effect + prior art:** `bfloat8_b` weights ≈ half the DRAM traffic of `bf16` and let the matmul
run at `MathFidelity.LoFi/HiFi2`, which is DRAM-bound-favorable for decode FF matmuls — see `llms.md` §2.5
(MLP: "use `bfloat8_b`, `MathFidelity.HiFi2`") and §4 (op configs); `tt_transformers` `precision_cfg`
(`ff1_3: bfp4, wqkv: bfp8, kv_cache: bfp8`) and `MODEL_UPDATES.md`. The DeepSeek experts ship `expert_dtype=fp4`
natively (`deepseek_v3_b1` fp4 experts), consistent with pushing expert FF to 4-bit — but the PCC check above
shows 4-bit must be applied with block-scale/HiFi care, not blindly.

> **Latency caveat (honesty):** the raw ms/iter printed by the test is **host-transfer + dispatch dominated**
> (tiny tensors; weights re-created and re-copied to device every call in the correctness-style functional
> harness). It is **not** a valid performance signal and is not reported as one. Real per-op/model latency comes
> from Checkpoint 4 with Metal Trace + persistent on-device weights + 2-CQ (see `TT_REUSE_MAP.md` / AdvancedPerf).

## Remaining ladder (not yet applied — require the full model from Checkpoint 2)
2. **Sharding** (height/width/block; keep consistent across consecutive ops) — `ttcnn.md`, `llms.md` §2.3/§4.3.
3. **Matmul program configs / DRAM-sharded matmul** for big FF + LM head — `llms.md` §2.7.
4. **Attention**: optimized SDPA / flash-decode, paged attention, `bfloat8_b` KV cache — `llms.md` §2.4.
5. **Metal Trace** (prefill + decode traces separately, persistent inputs) — AdvancedPerf §1.
6. **2 command queues** (overlap input writes with compute; event sync) — AdvancedPerf §2-3.
7. **Multi-device TP** across the 4 Blackhole chips (col-parallel QKV/gate/up, row-parallel o/down; CCLs) —
   Bring-Up Guide §6, `llms.md` §3.3. Respect head/chip divisibility (64 heads ÷ {1,2,4} fine).

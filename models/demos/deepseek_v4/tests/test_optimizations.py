# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint 3 — optimization ladder step 1 (precision) with PCC re-validation.

Demonstrates the Checkpoint-3 discipline on a validated module (clamped SwiGLU MLP):
sweep on-device weight precision bf16 -> bfloat8_b -> bfloat4_b, re-check PCC vs the
HF reference after EACH change (GOAL: "re-prove PCC on sim/hw after each optimization"),
and record real on-device latency `[hw: measured]` so the expected perf effect is grounded.

This is a module-level demonstration of the methodology, not a full-model result.
Runs on real Blackhole.
"""
import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v4.reference.reduced_config import build_reduced_model
from models.demos.deepseek_v4.tt import modules as M

torch.manual_seed(0)

DTYPES = [("bfloat16", ttnn.bfloat16), ("bfloat8_b", ttnn.bfloat8_b), ("bfloat4_b", ttnn.bfloat4_b)]


def bench(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter (host wall, incl. transfers)


def main():
    model, cfg = build_reduced_model(seed=0)
    se = model.model.layers[0].mlp.shared_experts
    H = cfg.hidden_size
    # larger token count so the matmul is non-trivial for a meaningful (relative) latency read
    x = torch.randn(1, 512, H)
    with torch.no_grad():
        ref = se(x)

    dev = ttnn.CreateDevice(device_id=0)
    rows = []
    try:
        for name, dt in DTYPES:
            got = M.clamped_swiglu_mlp(
                x,
                se.gate_proj.weight.data,
                se.up_proj.weight.data,
                se.down_proj.weight.data,
                dev,
                limit=cfg.swiglu_limit,
                weight_dtype=dt,
            )
            _, pcc = comp_pcc(ref, got, 0.99)
            ms = bench(
                lambda dt=dt: M.clamped_swiglu_mlp(
                    x,
                    se.gate_proj.weight.data,
                    se.up_proj.weight.data,
                    se.down_proj.weight.data,
                    dev,
                    limit=cfg.swiglu_limit,
                    weight_dtype=dt,
                )
            )
            rows.append((name, float(pcc), ms))
            print(f"[opt] SwiGLU weights={name:10s} PCC={float(pcc):.5f}  {ms:.3f} ms/iter [hw: measured]")
    finally:
        ttnn.CloseDevice(dev)

    print("\n=== Checkpoint-3 precision sweep (clamped SwiGLU MLP, 512 tokens) ===")
    print("weight_dtype  PCC       latency_ms[hw]  PCC>=0.99")
    for name, pcc, ms in rows:
        print(f"  {name:10s}  {pcc:.5f}   {ms:8.3f}      {'yes' if pcc >= 0.99 else 'NO'}")
    return rows


if __name__ == "__main__":
    main()

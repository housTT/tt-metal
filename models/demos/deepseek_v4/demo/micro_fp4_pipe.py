# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Measure the REAL fp4 MoE floor: pre-tilized bf4 experts, on-device fp32 accumulate, NO
per-expert read-back, async dispatch (one sync). This is the corrected bottleneck (MoE expert
DMA) measured properly — vs micro_fp4.py which synced+read every expert. Projects decode tok/s."""
import argparse
import time

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=6)  # top-6/layer
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    from transformers import AutoConfig

    snap = RW.find_snapshot()
    cfg = AutoConfig.from_pretrained(snap)
    I, H, limit = cfg.moe_intermediate_size, cfg.hidden_size, cfg.swiglu_limit
    store = RW.RealWeightStore(snap)
    E = args.experts

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        # pre-tilize E experts to host bf4 (one-time)
        hgu, hdn = [], []
        for e in range(E):
            gu_T, dn_T = RW.expert_fused(store, 3, e)
            hgu.append(ttnn.from_torch(gu_T, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT))
            hdn.append(ttnn.from_torch(dn_T, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT))
        x = torch.randn(1, H) * 0.1
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        w = [0.16] * E

        def moe_one_layer():
            """One layer's MoE: 6 experts, DMA-only fp4 upload + matmuls + on-device fp32 accum, NO reads."""
            acc = None
            for k in range(E):
                tgu = ttnn.to_device(hgu[k], dev)
                tdn = ttnn.to_device(hdn[k], dev)
                gu = ttnn.matmul(tx, tgu)
                gate = ttnn.clamp(gu[..., :I], max=limit)
                up = ttnn.clamp(gu[..., I:], min=-limit, max=limit)
                act = ttnn.multiply(ttnn.silu(gate), up)
                y = ttnn.matmul(act, tdn)
                yf = ttnn.multiply(ttnn.typecast(y, ttnn.float32), w[k])
                acc = yf if acc is None else ttnn.add(acc, yf)
                for t in (tgu, tdn, gu, gate, up, act, y):
                    ttnn.deallocate(t)
            return acc

        # warm
        a = moe_one_layer()
        ttnn.deallocate(a)
        ttnn.synchronize_device(dev)

        # time: iters * layers of MoE, one sync at end (async dispatch overlaps host)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            for _l in range(args.layers):
                a = moe_one_layer()
                ttnn.deallocate(a)
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0
        per_layer = dt / (args.iters * args.layers)
        moe_tok = per_layer * args.layers
        print(f"[fp4 MoE pipelined] {per_layer*1000:.2f} ms/layer (6 experts) -> MoE/token({args.layers} layers) = {moe_tok*1000:.0f} ms", flush=True)
        attn = 0.086  # measured ~2ms/layer x43
        for lbl, extra in [("MoE-only", 0.0), ("+attn+misc", attn + 0.05)]:
            tok = moe_tok + extra
            print(f"[proj decode {lbl}] ~{tok*1000:.0f} ms/tok -> {1.0/tok:.2f} tok/s", flush=True)
        print("MICRO_FP4_PIPE_OK", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()

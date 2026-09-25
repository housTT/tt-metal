# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Does BATCHING experts into one grouped fp4 matmul beat serial fp4? (decides the traced
build's expert path). Memory says grouping HURT in bf16 (per-token host stack+tilize). Here
weights are PRE-TILIZED bf4 host tensors (stack built once), so per-token is DMA + one batched
matmul. Compares, for E experts (one layer's routed set), serial-fp4 vs grouped-fp4 per-token
wall time. All on (1,1) mesh."""
import argparse
import time

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=6)  # top-6 per layer
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    from transformers import AutoConfig

    snap = RW.find_snapshot()
    cfg = AutoConfig.from_pretrained(snap)
    I = cfg.moe_intermediate_size
    H = cfg.hidden_size
    limit = cfg.swiglu_limit
    store = RW.RealWeightStore(snap)
    E = args.experts

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        gus, dns = [], []
        for e in range(E):
            gu_T, dn_T = RW.expert_fused(store, args.layer, e)  # host bf16 [H,2I],[I,H]
            gus.append(gu_T)
            dns.append(dn_T)

        # --- serial fp4: E separate pre-tilized bf4 weights, E separate matmuls ---
        s_gu = [ttnn.from_torch(g, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT) for g in gus]
        s_dn = [ttnn.from_torch(d, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT) for d in dns]
        x = torch.randn(1, H) * 0.1

        def serial():
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            for hg, hd in zip(s_gu, s_dn):
                tgu = ttnn.to_device(hg, dev)
                tdn = ttnn.to_device(hd, dev)
                gu = ttnn.matmul(tx, tgu)
                gate = ttnn.clamp(gu[..., :I], max=limit)
                up = ttnn.clamp(gu[..., I:], min=-limit, max=limit)
                act = ttnn.multiply(ttnn.silu(gate), up)
                y = ttnn.matmul(act, tdn)
                ttnn.to_torch(y)
                for t in (tgu, tdn, gu, gate, up, act, y):
                    ttnn.deallocate(t)
            ttnn.deallocate(tx)

        # --- grouped fp4: stacked bf4 [E,H,2I]/[E,I,H], one batched matmul each stage ---
        GU = torch.stack(gus, 0)  # [E,H,2I]
        DN = torch.stack(dns, 0)  # [E,I,H]
        g_gu = ttnn.from_torch(GU, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
        g_dn = ttnn.from_torch(DN, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
        xE = x.reshape(1, -1).expand(E, -1).reshape(E, 1, -1).contiguous()  # [E,1,H]

        def grouped():
            tx = ttnn.from_torch(xE, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            tgu = ttnn.to_device(g_gu, dev)
            tdn = ttnn.to_device(g_dn, dev)
            gu = ttnn.matmul(tx, tgu)  # [E,1,2I]
            gate = ttnn.clamp(gu[..., :I], max=limit)
            up = ttnn.clamp(gu[..., I:], min=-limit, max=limit)
            act = ttnn.multiply(ttnn.silu(gate), up)
            y = ttnn.matmul(act, tdn)  # [E,1,H]
            ttnn.to_torch(y)
            for t in (tx, tgu, tdn, gu, gate, up, act, y):
                ttnn.deallocate(t)

        for fn in (serial, grouped):  # warm
            fn()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            serial()
        t_serial = (time.perf_counter() - t0) / args.iters
        t0 = time.perf_counter()
        for _ in range(args.iters):
            grouped()
        t_grouped = (time.perf_counter() - t0) / args.iters

        print(f"[E={E} layer={args.layer}] serial-fp4 {t_serial*1000:.1f}ms  grouped-fp4 {t_grouped*1000:.1f}ms  -> {t_serial/t_grouped:.2f}x", flush=True)
        print(f"[MoE/token x43 layers] serial {t_serial*43:.2f}s  grouped {t_grouped*43:.2f}s", flush=True)
        for lbl, tmoe in [("serial", t_serial * 43), ("grouped", t_grouped * 43)]:
            dt = tmoe + 1.0
            print(f"[proj decode {lbl}] ~{dt:.2f}s/tok -> {1.0/dt:.2f} tok/s", flush=True)
        print("GROUPED_FP4_OK", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()

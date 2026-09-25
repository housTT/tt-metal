# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Last decode lever: ON-DEVICE fp32 accumulate vs per-expert read-back+host-accumulate.

Current sparse_moe_streaming reads each expert's [n,H] output back to host and accumulates in
fp32 (258 device->host reads/token). This tests accumulating on device (fused_expert_ondevice
returns a device tensor; we do w_e*y + acc on device in fp32) and reading ONCE per layer.
Measures per-layer wall time (E experts) both ways + PCC of the accumulated result, on (1,1)."""
import argparse
import time

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import modules as M


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    from transformers import AutoConfig

    snap = RW.find_snapshot()
    cfg = AutoConfig.from_pretrained(snap)
    I, H, limit = cfg.moe_intermediate_size, cfg.hidden_size, cfg.swiglu_limit
    store = RW.RealWeightStore(snap)
    E = args.experts
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        experts = [RW.expert_fused(store, args.layer, e) for e in range(E)]
        x = torch.randn(1, H) * 0.1
        w_e = [float(v) for v in torch.softmax(torch.randn(E), 0)]  # fake routing weights

        # (A) current: per-expert readback + host fp32 accumulate
        def host_accum():
            out = torch.zeros(1, H, dtype=torch.float32)
            for (gu_T, dn_T), w in zip(experts, w_e):
                y = M.fused_experts(x, gu_T, dn_T, I, dev, limit=limit)  # host [1,H]
                out += w * y.float()
            return out

        # (B) on-device fp32 accumulate, one read/layer
        def dev_accum():
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            acc = None
            for (gu_T, dn_T), w in zip(experts, w_e):
                y = M.fused_expert_ondevice(tx, gu_T, dn_T, I, dev, limit=limit)  # device [1,H]
                yf = ttnn.multiply(ttnn.typecast(y, ttnn.float32), w)
                acc = yf if acc is None else ttnn.add(acc, yf)
                ttnn.deallocate(y)
            out = ttnn.to_torch(acc)
            ttnn.deallocate(acc)
            ttnn.deallocate(tx)
            return out

        oa, ob = host_accum(), dev_accum()  # warm + correctness
        t0 = time.perf_counter()
        for _ in range(args.iters):
            host_accum()
        t_host = (time.perf_counter() - t0) / args.iters
        t0 = time.perf_counter()
        for _ in range(args.iters):
            dev_accum()
        t_dev = (time.perf_counter() - t0) / args.iters

        print(f"[E={E} layer={args.layer}] host-accum {t_host*1000:.1f}ms/layer  dev-accum {t_dev*1000:.1f}ms/layer -> {t_host/t_dev:.2f}x", flush=True)
        print(f"[accum PCC dev-vs-host] {pcc(oa, ob):.5f}", flush=True)
        print(f"[MoE/token x43] host {t_host*43:.2f}s  dev {t_dev*43:.2f}s", flush=True)
        print("MICRO_ACCUM_OK", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()

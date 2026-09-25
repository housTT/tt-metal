# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Is the routed-MoE cost per-to_device-CALL overhead or raw DMA? Compare 6 separate
to_device(bf4 expert) vs 1 to_device of a stacked [6,...] bf4 tensor, per 'layer' x43."""
import time

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW


def main():
    from transformers import AutoConfig

    snap = RW.find_snapshot()
    cfg = AutoConfig.from_pretrained(snap)
    I, H = cfg.moe_intermediate_size, cfg.hidden_size
    store = RW.RealWeightStore(snap)
    E = 6
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        gus = [RW.expert_fused(store, 3, e)[0] for e in range(E)]  # [H,2I] bf16
        sep = [ttnn.from_torch(g, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT) for g in gus]
        stacked = ttnn.from_torch(torch.stack(gus, 0), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)  # [6,H,2I]

        def sep_upload():
            ts = [ttnn.to_device(s, dev) for s in sep]
            for t in ts:
                ttnn.deallocate(t)

        def stacked_upload():
            t = ttnn.to_device(stacked, dev)
            ttnn.deallocate(t)

        for fn in (sep_upload, stacked_upload):
            fn()
        ttnn.synchronize_device(dev)
        N = 43 * 5
        t0 = time.perf_counter()
        for _ in range(N):
            sep_upload()
        ttnn.synchronize_device(dev)
        t_sep = (time.perf_counter() - t0) / N
        t0 = time.perf_counter()
        for _ in range(N):
            stacked_upload()
        ttnn.synchronize_device(dev)
        t_st = (time.perf_counter() - t0) / N
        print(f"[6 separate to_device] {t_sep*1000:.2f} ms  -> x43 = {t_sep*43*1000:.0f} ms", flush=True)
        print(f"[1 stacked to_device ] {t_st*1000:.2f} ms  -> x43 = {t_st*43*1000:.0f} ms  ({t_sep/t_st:.1f}x)", flush=True)
        print("MICRO_UPLOAD_OK", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()

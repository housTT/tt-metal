# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Does COALESCING the per-layer expert upload realize the measured 24.6 GB/s 4-chip aggregate?
The integrated mesh MoE streams 6 experts as ~18 tiny sharded to_device calls/layer; the DMA micro
showed one BIG sharded transfer hits ~24.6 GB/s while tiny ones don't. This compares, on the (1,4)
mesh, MoE/layer with:
  (A) separate : 6 experts x (gate,up,down) = 18 small sharded to_device  (current integrated path)
  (B) coalesced: all 6 experts' gate_up STACKED + down STACKED = 2 big sharded to_device, then 6
      separate matmuls (slice per expert — avoids the grouped-matmul penalty) + 1 all_gather.
Same math both ways; only the upload batching differs. Decides whether coalescing is the lever."""
import argparse
import time

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    from transformers import AutoConfig

    snap = RW.find_snapshot()
    cfg = AutoConfig.from_pretrained(snap)
    I, H, limit = cfg.moe_intermediate_size, cfg.hidden_size, cfg.swiglu_limit
    store = RW.RealWeightStore(snap)
    E = args.experts

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        C = mesh.get_num_devices()
        shard_col = ttnn.ShardTensorToMesh(mesh, dim=1)  # split I of [.,I]
        shard_row = ttnn.ShardTensorToMesh(mesh, dim=0)  # split I of [I,.]
        rep = ttnn.ReplicateTensorToMesh(mesh)

        w = [1.0 / E] * E
        gu_list, dn_list = [], []
        for e in range(E):
            gu_T, dn_T = RW.expert_fused(store, 3, e)  # [H,2I],[I,H]
            gu_list.append(gu_T)
            dn_list.append(dn_T)

        # (A) per-expert sharded host tensors
        sepA = []
        for e in range(E):
            g = ttnn.from_torch(gu_list[e][:, :I].contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard_col)
            u = ttnn.from_torch(gu_list[e][:, I:].contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard_col)
            d = ttnn.from_torch(dn_list[e].contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard_row)
            sepA.append((g, u, d))

        # (B) coalesced: stack the E experts on a BATCH dim so ONE sharded to_device uploads all of
        # them, then a BATCHED matmul runs all E at once (bf4 can't be sliced on device, so we batch
        # rather than slice). GU [E,H,2I] shard 2I (dim=2); DN [E,I,H] shard I (dim=1).
        GU = torch.stack([gu_list[e] for e in range(E)], dim=0).contiguous()  # [E, H, 2I]
        DN = torch.stack([dn_list[e] for e in range(E)], dim=0).contiguous()  # [E, I, H]
        gu_co = ttnn.from_torch(GU, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2))  # split 2I
        dn_co = ttnn.from_torch(DN, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))  # split I
        txE = ttnn.from_torch((torch.randn(1, H) * 0.1).reshape(1, H).repeat(E, 1).reshape(E, 1, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)  # [E,1,H]
        wE = ttnn.from_torch(torch.tensor(w).reshape(E, 1, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

        tx = ttnn.from_torch(torch.randn(1, H) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        w = [1.0 / E] * E

        def reduce(local):
            lr = ttnn.reshape(local, [1, 1, 1, H])
            yg = ttnn.all_gather(lr, dim=0)
            y = ttnn.sum(yg, dim=0)
            ttnn.deallocate(lr); ttnn.deallocate(yg)
            return y

        def layer_separate():
            local = None
            for k in range(E):
                g, u, d = sepA[k]
                tg = ttnn.to_device(g, mesh); tu = ttnn.to_device(u, mesh); td = ttnn.to_device(d, mesh)
                gt = ttnn.clamp(ttnn.matmul(tx, tg), max=limit)
                up = ttnn.clamp(ttnn.matmul(tx, tu), min=-limit, max=limit)
                act = ttnn.multiply(ttnn.silu(gt), up)
                yp = ttnn.matmul(act, td)
                yf = ttnn.multiply(ttnn.typecast(yp, ttnn.float32), float(w[k]))
                local = yf if local is None else ttnn.add(local, yf)
                for t in (tg, tu, td, gt, up, act, yp):
                    ttnn.deallocate(t)
            y = reduce(local); ttnn.deallocate(local); return y

        def layer_coalesced():
            tgu = ttnn.to_device(gu_co, mesh)  # [E, H, 2I/C] per chip — ONE big transfer
            tdn = ttnn.to_device(dn_co, mesh)  # [E, I/C, H] per chip — ONE big transfer
            half = tgu.shape[2] // 2
            gu = ttnn.matmul(txE, tgu)  # [E,1,2I/C] batched
            gt = ttnn.clamp(ttnn.slice(gu, [0, 0, 0], [E, 1, half]), max=limit)  # bf16 activation is sliceable
            up = ttnn.clamp(ttnn.slice(gu, [0, 0, half], [E, 1, tgu.shape[2]]), min=-limit, max=limit)
            act = ttnn.multiply(ttnn.silu(gt), up)  # [E,1,I/C]
            yp = ttnn.matmul(act, tdn)  # [E,1,H] batched partial per chip
            yf = ttnn.multiply(ttnn.typecast(yp, ttnn.float32), wE)  # weight per expert
            local = ttnn.sum(yf, dim=0)  # [1,H] partial (sum over experts)
            y = reduce(local)
            for t in (tgu, tdn, gu, gt, up, act, yp, yf, local):
                ttnn.deallocate(t)
            return y

        def timeit(fn):
            fn(); ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(args.iters * args.layers):
                o = fn(); ttnn.deallocate(o)
            ttnn.synchronize_device(mesh)
            return (time.perf_counter() - t0) / (args.iters * args.layers)

        tA = timeit(layer_separate)
        tB = timeit(layer_coalesced)
        L = args.layers
        print(f"[A separate 18 uploads/layer ] {tA*1000:.2f} ms/layer -> MoE/token {tA*L*1000:.0f} ms", flush=True)
        print(f"[B coalesced 2 uploads/layer ] {tB*1000:.2f} ms/layer -> MoE/token {tB*L*1000:.0f} ms  ({tA/tB:.2f}x)", flush=True)
        for lbl, moe in [("A", tA * L), ("B", tB * L)]:
            tokt = moe + 0.18  # + measured attn/mHC/comp ~180ms
            print(f"  [{lbl}] +180ms attn -> {tokt*1000:.0f} ms/tok -> {1.0/tokt:.2f} tok/s", flush=True)
        print("MICRO_FP4_COALESCE_OK", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        try: ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception: pass


if __name__ == "__main__":
    main()

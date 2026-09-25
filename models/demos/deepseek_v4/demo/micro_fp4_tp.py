# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DECISIVE 4-chip test: tensor-parallel sharded fp4 MoE. Each expert's weight is sharded across
the 4 chips (gate/up split on the intermediate dim, down likewise), so each chip uploads 1/4 of
the weight over its OWN PCIe in parallel -> ~4x host->DRAM bandwidth vs single chip. Measures
MoE/token on the full QB2 mesh -> projects real single-stream decode tok/s."""
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
        nchips = mesh.get_num_devices()
        shard1 = ttnn.ShardTensorToMesh(mesh, dim=1)  # split cols (I) across chips
        shard0 = ttnn.ShardTensorToMesh(mesh, dim=0)  # split rows (I) across chips
        rep = ttnn.ReplicateTensorToMesh(mesh)

        experts = []
        for e in range(E):
            gu_T, dn_T = RW.expert_fused(store, 3, e)  # gu_T [H,2I], dn_T [I,H]
            gate = ttnn.from_torch(gu_T[:, :I].contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard1)  # [H,I]->[H,I/n]
            up = ttnn.from_torch(gu_T[:, I:].contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard1)
            down = ttnn.from_torch(dn_T.contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard0)  # [I,H]->[I/n,H]
            experts.append((gate, up, down))
        x = torch.randn(1, H) * 0.1
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

        def moe_one_layer():
            acc = None
            for gate, up, down in experts:
                # host->device DMA of the (already-tilized) sharded bf4 weights, 1/n per chip in parallel
                tg = ttnn.to_device(gate, mesh)
                tu = ttnn.to_device(up, mesh)
                td = ttnn.to_device(down, mesh)
                gp = ttnn.matmul(tx, tg)  # [1,I/n]
                upp = ttnn.matmul(tx, tu)
                actp = ttnn.multiply(ttnn.silu(ttnn.clamp(gp, max=limit)), ttnn.clamp(upp, min=-limit, max=limit))
                yp = ttnn.matmul(actp, td)  # [1,H] partial (per-chip partial sum over its I-slice)
                yr = ttnn.reshape(yp, [1, 1, 1, H])
                yg = ttnn.all_gather(yr, dim=0)  # [n,1,1,H] on every chip
                y = ttnn.sum(yg, dim=0)  # [1,1,H] reduced
                acc = y if acc is None else ttnn.add(acc, y)
                for t in (tg, tu, td, gp, upp, actp, yp, yr, yg):
                    ttnn.deallocate(t)
            return acc

        a = moe_one_layer()
        ttnn.deallocate(a)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            for _l in range(args.layers):
                a = moe_one_layer()
                ttnn.deallocate(a)
        ttnn.synchronize_device(mesh)
        dt = time.perf_counter() - t0
        per_layer = dt / (args.iters * args.layers)
        moe_tok = per_layer * args.layers
        print(f"[fp4 MoE TP x{nchips} chips] {per_layer*1000:.2f} ms/layer -> MoE/token = {moe_tok*1000:.0f} ms", flush=True)
        for lbl, extra in [("MoE-only", 0.0), ("+attn+misc", 0.13)]:
            tok = moe_tok + extra
            print(f"[proj decode {lbl}] ~{tok*1000:.0f} ms/tok -> {1.0/tok:.2f} tok/s", flush=True)
        print("MICRO_FP4_TP_OK", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


if __name__ == "__main__":
    main()

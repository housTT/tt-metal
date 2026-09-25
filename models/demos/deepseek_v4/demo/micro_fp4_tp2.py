# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Deferred-reduction tensor-parallel fp4 MoE across 4 chips. Same sharding as micro_fp4_tp
(gate/up split on I, down split on I so each chip yields a [1,H] PARTIAL sum over its I-slice),
but because the cross-chip reduction is linear we accumulate every routed expert's partial
LOCALLY on each chip and do a SINGLE all_gather+sum per layer (was 1 per expert => 6x fewer
collectives). Measures MoE/token on the full QB2 mesh."""
import argparse
import time

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--stack", action="store_true", help="stack gate|up|down uploads per expert")
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
            gate = ttnn.from_torch(gu_T[:, :I].contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard1)
            up = ttnn.from_torch(gu_T[:, I:].contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard1)
            down = ttnn.from_torch(dn_T.contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard0)
            experts.append((gate, up, down))
        x = torch.randn(1, H) * 0.1
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        w = [0.3] * E  # dummy routing weights

        def moe_one_layer():
            local = None  # [1,H] per-chip partial over its I-slice, accumulated across experts
            for k, (gate, up, down) in enumerate(experts):
                tg = ttnn.to_device(gate, mesh)
                tu = ttnn.to_device(up, mesh)
                td = ttnn.to_device(down, mesh)
                gp = ttnn.matmul(tx, tg)  # [1,I/n]
                upp = ttnn.matmul(tx, tu)
                actp = ttnn.multiply(ttnn.silu(ttnn.clamp(gp, max=limit)), ttnn.clamp(upp, min=-limit, max=limit))
                yp = ttnn.matmul(actp, td)  # [1,H] partial
                yf = ttnn.multiply(ttnn.typecast(yp, ttnn.float32), float(w[k]))
                local = yf if local is None else ttnn.add(local, yf)
                for t in (tg, tu, td, gp, upp, actp, yp):
                    ttnn.deallocate(t)
            # single collective for the whole layer
            lr = ttnn.reshape(local, [1, 1, 1, H])
            yg = ttnn.all_gather(lr, dim=0)  # [n,1,1,H]
            y = ttnn.sum(yg, dim=0)  # [1,1,H] reduced (replicated)
            ttnn.deallocate(local)
            ttnn.deallocate(lr)
            ttnn.deallocate(yg)
            return y

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
        print(f"[fp4 MoE TP2 x{nchips} deferred-reduce] {per_layer*1000:.2f} ms/layer -> MoE/token = {moe_tok*1000:.0f} ms", flush=True)
        for lbl, extra in [("MoE-only", 0.0), ("+attn+misc", 0.16)]:
            tok = moe_tok + extra
            print(f"[proj decode {lbl}] ~{tok*1000:.0f} ms/tok -> {1.0/tok:.2f} tok/s", flush=True)
        print("MICRO_FP4_TP2_OK", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Resident MoE with experts SHARDED (tensor-parallel) across the 4 chips — the fix for the
single-device OOM. Each expert's gate/up is column-sharded and down is row-sharded across the
mesh (TP-exact for SwiGLU), so the working set is ~1/4 per chip and FITS. Weights are resident
(uploaded once, reused every token) — warm decode does no per-token weight transfer.

Measures: (1) it fits (no OOM) with a full layer's worth of routed experts resident-sharded,
(2) warm decode-step MoE latency, (3) PCC vs a torch reference. This is the memory-dominant
piece of the model; the projection follows in the report."""
import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc

torch.manual_seed(0)


def main():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=90_000_000)
    try:
        C = mesh.get_num_devices()
        H, I = 4096, 2048  # DeepSeek-V4 dims
        limit = 10.0
        E = 32  # routed experts touched this decode step (working-set size for a layer)
        N = 1  # decode: 1 token
        x = torch.randn(N, H, dtype=torch.bfloat16)
        gate = torch.randn(E, H, I, dtype=torch.bfloat16) * 0.02
        up = torch.randn(E, H, I, dtype=torch.bfloat16) * 0.02
        down = torch.randn(E, I, H, dtype=torch.bfloat16) * 0.02
        # torch reference: run token through all E experts, sum (uniform weights=1 for the test)
        ref = torch.zeros(N, H)
        for e in range(E):
            g = (x.float() @ gate[e].float()).clamp(max=limit)
            u = (x.float() @ up[e].float()).clamp(min=-limit, max=limit)
            ref += (torch.nn.functional.silu(g) * u) @ down[e].float()

        # RESIDENT sharded expert weights on the mesh: gate/up col-sharded [E,H,I/C];
        # down row-sharded [E,I/C,H]. Uploaded ONCE (bf16, fast). Working set /C per chip.
        col = ttnn.ShardTensorToMesh(mesh, dim=-1)
        row = ttnn.ShardTensorToMesh(mesh, dim=1)  # shard the I dim of down [E,I,H]
        mk = lambda t, m: ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=m,
        )
        t0 = time.perf_counter()
        tgate = mk(gate, col)  # [E, H, I/C] per chip
        tup = mk(up, col)
        tdown = mk(down, row)  # [E, I/C, H] per chip
        upload_s = time.perf_counter() - t0
        print(f"[res-moe] uploaded {E} experts SHARDED across {C} chips (resident): {upload_s:.2f}s")

        tx = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        def moe_step():
            # each chip: token through its I-shard of every expert, sum over experts, then
            # sum the per-chip partials (TP reduce) on read.
            acc = None
            for e in range(E):
                g = ttnn.clamp(ttnn.matmul(tx, tgate[e]), max=limit)  # [N, I/C]
                u = ttnn.clamp(ttnn.matmul(tx, tup[e]), min=-limit, max=limit)
                y = ttnn.matmul(ttnn.multiply(ttnn.silu(g), u), tdown[e])  # [N, H] partial per chip
                acc = y if acc is None else ttnn.add(acc, y)
            return acc

        out = moe_step()
        parts = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))  # [C*N, H]
        got = parts.reshape(C, N, H).sum(0)  # TP reduce across chips
        _, pcc = comp_pcc(ref, got, 0.99)
        print(f"[res-moe] decode MoE ({E} experts, sharded-resident): PCC={float(pcc):.5f}")

        # warm latency (weights resident — no re-upload)
        for _ in range(2):
            moe_step()
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        R = 10
        for _ in range(R):
            moe_step()
        ttnn.synchronize_device(mesh)
        warm_ms = (time.perf_counter() - t0) / R * 1e3
        print(
            f"[res-moe] warm decode-MoE latency (resident, no transfer): {warm_ms:.2f} ms/layer  "
            f"-> {warm_ms * 43 / 1000:.2f}s for 43 layers"
        )
        print("RES_MOE_RESULT", "PASS" if pcc >= 0.99 else "FAIL")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

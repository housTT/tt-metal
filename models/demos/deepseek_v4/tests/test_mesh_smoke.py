# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stage 1: 4-chip MeshDevice with a RESIDENT column-sharded weight + tensor-parallel matmul.

Proves weights can be sharded across the 4 Blackhole chips, kept resident (loaded once,
reused), and computed in parallel with a CCL gather — the foundation for keeping weights
resident instead of streaming them per token."""
import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc

torch.manual_seed(0)


def main():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    try:
        n = mesh.get_num_devices()
        print(f"[mesh] opened {n} devices: shape {tuple(mesh.shape)}")
        M, K, N = 128, 4096, 4096
        x = torch.randn(M, K, dtype=torch.bfloat16)
        w = torch.randn(K, N, dtype=torch.bfloat16)
        ref = (x.float() @ w.float())

        # weight sharded column-wise across the 4 chips, RESIDENT (built once)
        tw = ttnn.from_torch(
            w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),  # each chip holds [K, N/4]
        )
        # input replicated to all chips
        tx = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        def tp_matmul():
            y = ttnn.matmul(tx, tw)  # per-chip [M, N/4]
            return ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))  # -> [M, N]

        out = tp_matmul()  # first (compiles)
        _, pcc = comp_pcc(ref, out, 0.99)
        print(f"[mesh] TP matmul {M}x{K}x{N} sharded over {n} chips: PCC={float(pcc):.5f} out={tuple(out.shape)}")

        # reuse RESIDENT weight (no re-transfer) — time steady-state
        for _ in range(3):
            tp_matmul()
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(20):
            tp_matmul()
        ttnn.synchronize_device(mesh)
        ms = (time.perf_counter() - t0) / 20 * 1e3
        print(f"[mesh] resident-weight TP matmul steady-state: {ms:.3f} ms/iter (weight NOT re-transferred)")
        print("MESH_SMOKE_RESULT", "PASS" if pcc >= 0.99 else "FAIL")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

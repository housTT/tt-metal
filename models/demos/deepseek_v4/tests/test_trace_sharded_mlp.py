# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stage 2: Metal Trace + resident sharded weights on a SwiGLU MLP (4-chip mesh).

The MLP intermediate dim I is sharded across the 4 chips (TP-exact: SwiGLU's nonlinearity
is per-I-channel, so gate/up column-sharded + down row-sharded + sum = the exact full MLP).
Weights are resident (loaded once). We capture the per-chip on-device compute into a Metal
Trace and replay it, measuring traced-replay vs eager to show the per-op host-dispatch
overhead (the profiled bottleneck) collapse."""
import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc

torch.manual_seed(0)


def main():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    try:
        C = mesh.get_num_devices()
        M, H, I = 32, 4096, 8192  # decode-ish token count; MLP dims
        limit = 10.0
        x = torch.randn(M, H, dtype=torch.bfloat16)
        gate = torch.randn(H, I, dtype=torch.bfloat16) * 0.02
        up = torch.randn(H, I, dtype=torch.bfloat16) * 0.02
        down = torch.randn(I, H, dtype=torch.bfloat16) * 0.02
        # torch reference (full SwiGLU)
        g = (x.float() @ gate.float()).clamp(max=limit)
        u = (x.float() @ up.float()).clamp(min=-limit, max=limit)
        ref = (torch.nn.functional.silu(g) * u) @ down.float()

        # RESIDENT sharded weights (intermediate dim I sharded, TP-exact):
        # gate/up col-sharded [H, I/C] (SAME I-channels per chip); down row-sharded [I/C, H]
        col = ttnn.ShardTensorToMesh(mesh, dim=-1)
        tgate = ttnn.from_torch(
            gate,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=col,
        )
        tup = ttnn.from_torch(
            up,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=col,
        )
        tdn = ttnn.from_torch(
            down,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        tx = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        def block(h):
            gate_o = ttnn.clamp(ttnn.matmul(h, tgate), max=limit)  # [M, I/C]
            up_o = ttnn.clamp(ttnn.matmul(h, tup), min=-limit, max=limit)
            act = ttnn.multiply(ttnn.silu(gate_o), up_o)  # [M, I/C]
            return ttnn.matmul(act, tdn)  # [M, H] partial per chip

        def reduce(y):  # sum the 4 chips' partials -> full [M,H]
            parts = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))  # [C*M, H]
            return parts.reshape(C, M, H).sum(0)

        # correctness (single block)
        _, pcc = comp_pcc(ref, reduce(block(tx)), 0.99)
        print(f"[trace] sharded SwiGLU MLP over {C} chips: PCC={float(pcc):.5f}")

        # chain R blocks to mimic a decode step's on-device op volume (~4 ops/block)
        R = 50

        def fwd():
            h = tx
            for _ in range(R):
                h = block(h)
            return h

        # eager timing (steady-state)
        for _ in range(2):
            fwd()
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(10):
            fwd()
        ttnn.synchronize_device(mesh)
        eager_ms = (time.perf_counter() - t0) / 10 * 1e3

        # capture a Metal Trace of the whole chain, then replay
        out = fwd()  # ensure program cache populated
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = fwd()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        for _ in range(2):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(10):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        traced_ms = (time.perf_counter() - t0) / 10 * 1e3

        print(
            f"[trace] {R} blocks (~{4 * R} ops): eager={eager_ms:.2f} ms  traced={traced_ms:.2f} ms  "
            f"speedup={eager_ms / traced_ms:.2f}x (host-dispatch overhead removed)"
        )
        print("TRACE_RESULT", "PASS" if pcc >= 0.99 and traced_ms < eager_ms else "CHECK")
        ttnn.release_trace(mesh, tid)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

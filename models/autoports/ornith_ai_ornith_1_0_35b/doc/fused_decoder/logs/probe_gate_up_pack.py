# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared-LHS packing candidate for the two routed-expert sparse matmuls.

``gate`` and ``up`` share the same activation, so they can run as one ``sparse_matmul`` over
``concat([W_gate, W_up], dim=-1)``. The motivation is core count: the sparse factory gives each
core one N-tile block and cannot split M (the token axis is a single 32-row tile), so an N=512
matmul is pinned to ``512/32 = 16`` cores while an N=1024 one can use 32.

The cost is that the halves then have to be sliced apart. This probe measures whether the wider
matmul pays for those slices, at the exact decode shapes.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs/probe_gate_up_pack.py
"""

import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.fused_decoder import _sparse_matmul_config

TILE = 32
E = 256
HIDDEN = 2048
INTER = 512
ACTIVE = 16  # experts a batch-1 decode step actually selects


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    try:
        gen = torch.Generator().manual_seed(11)
        a = dev(mesh, torch.randn(1, 1, TILE, HIDDEN, generator=gen).to(torch.bfloat16))
        w_gate = torch.randn(1, E, HIDDEN, INTER, generator=gen).to(torch.bfloat16)
        w_up = torch.randn(1, E, HIDDEN, INTER, generator=gen).to(torch.bfloat16)
        wg, wu = dev(mesh, w_gate), dev(mesh, w_up)
        wgu = dev(mesh, torch.cat([w_gate, w_up], dim=-1))
        mask = torch.zeros(1, 1, 1, E)
        mask[0, 0, 0, :ACTIVE] = 1.0
        sparsity = dev(mesh, mask, layout=ttnn.ROW_MAJOR_LAYOUT)

        cfg512 = _sparse_matmul_config(TILE, INTER, 16)
        cfg1024 = _sparse_matmul_config(TILE, 2 * INTER, 16)
        print(
            f"cfg N=512 grid={cfg512.compute_with_storage_grid_size} per_core_N={cfg512.per_core_N}  "
            f"cfg N=1024 grid={cfg1024.compute_with_storage_grid_size} per_core_N={cfg1024.per_core_N}"
        )
        common = dict(
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([TILE, TILE]),
            compute_kernel_config=ckc,
            dtype=ttnn.bfloat16,
        )

        def split_path():
            g = ttnn.sparse_matmul(a, wg, program_config=cfg512, **common)
            u = ttnn.sparse_matmul(a, wu, program_config=cfg512, **common)
            h = ttnn.multiply(
                g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            return h

        def packed_path():
            y = ttnn.sparse_matmul(a, wgu, program_config=cfg1024, **common)
            shape = [int(d) for d in y.shape]
            lo = list(shape)
            lo[-1] = INTER
            g = ttnn.slice(y, [0] * len(shape), lo)
            begins = [0] * len(shape)
            begins[-1] = INTER
            u = ttnn.slice(y, begins, shape)
            ttnn.deallocate(y)
            h = ttnn.multiply(
                g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            return h

        ref = ttnn.to_torch(split_path()).float()
        got = ttnn.to_torch(packed_path()).float()
        d = (ref.double().flatten() - ref.double().mean(), got.double().flatten() - got.double().mean())
        pcc = float((d[0] * d[1]).sum() / (d[0].norm() * d[1].norm() + 1e-12))
        for name, fn in (("two N=512 matmuls", split_path), ("one N=1024 matmul + 2 slices", packed_path)):
            fn()
            ttnn.synchronize_device(mesh)
            iters = 30
            start = time.time()
            for _ in range(iters):
                out = fn()
                ttnn.deallocate(out)
            ttnn.synchronize_device(mesh)
            per = (time.time() - start) / iters
            print(f"GATEUP {name:32s} wall={per * 1e6:8.1f} us/call", flush=True)
        exact = "bitwise-equal" if torch.equal(ref, got) else "differs"
        # PCC alone cannot support the "bit-exact" wording the documents use for this rewrite, so the
        # exact comparison is performed and printed rather than inferred from a rounded correlation.
        print(f"GATEUP equivalence pcc={pcc:.6f} {exact}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

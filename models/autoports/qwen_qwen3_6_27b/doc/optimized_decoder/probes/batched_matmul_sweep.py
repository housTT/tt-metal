# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Every batched delta-rule matmul shape the prefill actually dispatches, swept properly.

``O10``'s first implementation measured two shapes and then applied one fixed program config -
an 8x6 grid with ``in0_block_w = 1`` - to *every* tile-aligned batched matmul.  That was too
broad: the triangular inverse's ``b={768} x 32 x 32 x 32`` base case went 271.5 -> 535.9 us and
``b={384} x 32 x 32 x 32`` went 41.5 -> 69.4 us, because a 48-core grid is narrower than what the
default program picks for a 768-deep batch of single-tile matmuls.

This sweeps the real shapes, taken from the shipped profile
(``tracy/linear_attention/prefill_perf_report.csv``), over grid and ``in0_block_w`` and
``out_subblock_h``, against the default program, PCC-gated.

Emits ``BMSWEEP `` JSON lines.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")
sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes")

import torch  # noqa: E402
import ttnn  # noqa: E402
from matmul_sweep import time_op  # noqa: E402

#: ``(batch, M, K, N)`` for every batched matmul in the profile, with its share of the window.
SHAPES = [
    (48, 64, 128, 128),   # 1976.2 us over 128 dispatches - the largest
    (48, 64, 64, 128),    #  576.0 us over 96
    (768, 32, 32, 32),    #  535.9 us over 32 - triangular inverse base case
    (384, 64, 128, 64),   #  217.5 us over 4
    (384, 32, 32, 32),    #   69.4 us over 8
]
GRIDS = [(8, 8), (8, 6), (11, 10), (8, 10), (10, 10), (4, 8)]


def emit(**p):
    print("BMSWEEP " + json.dumps(p, sort_keys=True, default=str), flush=True)


def pcc(a, b):
    a = a.to(torch.float64).flatten(); a = a - a.mean()
    b = b.to(torch.float64).flatten(); b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    hifi4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                             math_approx_mode=False, fp32_dest_acc_en=True,
                                             packer_l1_acc=True)
    try:
        for batch, m, k, n in SHAPES:
            torch.manual_seed(0)
            mem = ttnn.L1_MEMORY_CONFIG if batch * m * max(k, n) * 4 < 60e6 else ttnn.DRAM_MEMORY_CONFIG
            a = ttnn.from_torch(torch.randn(1, batch, m, k) * 0.1, dtype=ttnn.float32,
                                layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
            b = ttnn.from_torch(torch.randn(1, batch, k, n) * 0.1, dtype=ttnn.float32,
                                layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
            shape = f"b={batch} x {m} x {k} x {n}"
            golden = ttnn.to_torch(ttnn.matmul(a, b, dtype=ttnn.float32,
                                               compute_kernel_config=hifi4, memory_config=mem)).float()
            row = {"shape": shape, "config": "default"}
            row["us"] = time_op(lambda: ttnn.matmul(a, b, dtype=ttnn.float32,
                                                    compute_kernel_config=hifi4,
                                                    memory_config=mem), device)
            emit(**row)
            k_tiles, m_tiles, n_tiles = k // 32, m // 32, n // 32
            for grid in GRIDS:
                for in0_block_w in sorted({d for d in (1, 2, 4, 8, k_tiles) if d <= k_tiles and k_tiles % d == 0}):
                    for out_subblock_h in sorted({h for h in (1, 2, 4) if m_tiles % h == 0}, reverse=True):
                        out_subblock_w = next((w for w in (4, 2, 1)
                                               if n_tiles % w == 0 and w * out_subblock_h <= 4), 1)
                        row = {"shape": shape, "config": f"{grid[0]}x{grid[1]}_ibw{in0_block_w}_osh{out_subblock_h}"}
                        try:
                            pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                                compute_with_storage_grid_size=grid, in0_block_w=in0_block_w,
                                out_subblock_h=out_subblock_h, out_subblock_w=out_subblock_w,
                                per_core_M=m_tiles, per_core_N=n_tiles)
                            got = ttnn.matmul(a, b, program_config=pc, dtype=ttnn.float32,
                                              compute_kernel_config=hifi4, memory_config=mem)
                            row["pcc"] = pcc(golden, ttnn.to_torch(got).float())
                            ttnn.deallocate(got)
                            row["us"] = time_op(lambda: ttnn.matmul(a, b, program_config=pc,
                                                                    dtype=ttnn.float32,
                                                                    compute_kernel_config=hifi4,
                                                                    memory_config=mem), device)
                        except Exception as exc:
                            lines = str(exc).splitlines()
                            row["error"] = lines[2] if len(lines) > 2 else str(exc)[:150]
                        emit(**row)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

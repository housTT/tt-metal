# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Explicit 2D program configs for the dense *prefill* projections, against ttnn's heuristic.

The optimized prefill window is 81.65 % routed-expert `sparse_matmul` and only **1.07 %** dense
matmul, but `tt-perf-report` flags every one of those dense rows with `in0_block_w=1 is small`, so
this measures whether an explicit `MatmulMultiCoreReuseMultiCastProgramConfig` with a larger inner
block beats the heuristic at the real 2048-token prefill shapes. The answer decides whether the
layer carries a prefill program config at all.

    python .../logs/probe_prefill_matmul.py
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import ttnn

TILE = 32
ROLES = [
    ("attn_in", 2048, 2048, 9216),
    ("o_proj", 2048, 4096, 2048),
    ("gdn_in", 2048, 2048, 12352),
    ("gdn_out", 2048, 4096, 2048),
    ("shared_in", 2048, 2048, 1056),
    ("shared_down", 2048, 512, 2048),
]


def largest_divisor_at_most(value, cap):
    for candidate in range(min(cap, value), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def timeit(mesh, fn, iters=6, warmup=2):
    for _ in range(warmup):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    start = time.time()
    for _ in range(iters):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    return (time.time() - start) / iters * 1e6


def cfg_2d(grid, m, k, n, in0_cap, fp32_acc=False):
    gx, gy = grid
    m_t, k_t, n_t = m // TILE, k // TILE, math.ceil(n / TILE)
    per_core_m = math.ceil(m_t / gy)
    per_core_n = math.ceil(n_t / gx)
    cap = 4 if fp32_acc else 8
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if per_core_m % i == 0 and i * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=largest_divisor_at_most(k_t, in0_cap),
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6)
    args = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    grid = mesh.compute_with_storage_grid_size()
    print(f"# grid {grid.x}x{grid.y} weight=bfloat8_b fidelity=HiFi2 (the shipped projection policy)")
    try:
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        gen = torch.Generator().manual_seed(23)
        for name, m, k, n in ROLES:
            x = ttnn.from_torch(
                torch.randn(1, 1, m, k, generator=gen) * 0.1,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            w = ttnn.from_torch(
                torch.randn(1, 1, k, n, generator=gen) * 0.02,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            print(
                f"PREFILLMM role={name} m={m} k={k} n={n} family=heuristic "
                f"us={timeit(mesh, lambda: ttnn.linear(x, w, compute_kernel_config=ckc), args.iters):.1f}",
                flush=True,
            )
            for gx, gy in ((grid.x, grid.y), (8, 8), (8, 10)):
                if gx > grid.x or gy > grid.y:
                    continue
                for in0_cap in (2, 4, 8, 16):
                    cfg = cfg_2d((gx, gy), m, k, n, in0_cap)
                    tag = (
                        f"PREFILLMM role={name} m={m} k={k} n={n} family=2d grid={gx}x{gy} "
                        f"in0_block_w={cfg.in0_block_w} per_core_M={cfg.per_core_M} per_core_N={cfg.per_core_N} "
                        f"sub={cfg.out_subblock_h}x{cfg.out_subblock_w}"
                    )
                    try:
                        us = timeit(
                            mesh,
                            lambda cfg=cfg: ttnn.linear(x, w, compute_kernel_config=ckc, program_config=cfg),
                            args.iters,
                        )
                        print(f"{tag} us={us:.1f}", flush=True)
                    except Exception as exc:  # noqa: BLE001 - illegal geometries are data
                        print(f"{tag} FAILED {str(exc).splitlines()[0][:100]}", flush=True)
            ttnn.deallocate(x)
            ttnn.deallocate(w)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone program-config search for the decoder's dominant matmuls.

Model-free and therefore cheap: it allocates one weight and one activation per role and times
the matmul alone, so a whole geometry sweep costs seconds instead of a layer rebuild per
candidate.  Two roles per shape:

``decode``
    ``M = 32`` (one tile-padded token row), DRAM-width-sharded weight, width-sharded L1
    activation, ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``.  Sweeps the
    activation core grid and therefore ``in0_block_w`` (``$optimize`` OPT-004).
``prefill``
    ``M = 2048``, the same DRAM-width-sharded weight, DRAM-interleaved activation,
    ``MatmulMultiCoreReuseMultiCastProgramConfig``.  Sweeps ``in0_block_w`` and the output
    block, which is what the per-core circular-buffer budget actually binds.

Output: one ``MMSWEEP `` JSON line per candidate, including failures - an illegal or
L1-overflowing configuration is evidence about the contract, so it is recorded rather than
skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")

import torch  # noqa: E402
import ttnn  # noqa: E402

TILE = 32
DRAM_BANKS = 8
DRAM_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_BANKS - 1, 0))})

B8, B4, B16 = ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat16
LOFI = ttnn.MathFidelity.LoFi

#: ``(role, k, n, weight dtype)`` for every matmul that is a material share of a decode step.
#: Shapes are the real Qwen3.6-27B ones; the MLP appears in both the packed and split forms.
ROLES = [
    ("mlp_gate_split", 5120, 17408, B4),
    ("mlp_gate_up_packed", 5120, 34816, B4),
    ("mlp_down", 17408, 5120, B8),
    ("wqkv", 5120, 8192, B8),
    ("wgate", 5120, 6144, B8),
    ("o_proj", 6144, 5120, B8),
    ("gdn_in_proj_qkv", 5120, 10240, B8),
    ("gdn_in_proj_z", 5120, 6144, B8),
    ("gdn_out_proj", 6144, 5120, B8),
]

ITERS = 20


def emit(**payload):
    print("MMSWEEP " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def dram_sharded_weight_cfg(k, n):
    stride = TILE * DRAM_BANKS
    padded = math.ceil(n / stride) * stride
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(DRAM_GRID, (k, padded // DRAM_BANKS), ttnn.ShardOrientation.ROW_MAJOR),
    )


def width_sharded_l1(rows, width, grid):
    return ttnn.create_sharded_memory_config(
        shape=(rows, width // grid.num_cores),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def grids_for(k_tiles, max_cores=64):
    """Every (rows, cols) whose core count divides ``k_tiles``, largest cores first."""
    out = []
    for cores in sorted({c for c in range(1, max_cores + 1) if k_tiles % c == 0}, reverse=True):
        for rows in range(1, 9):
            if cores % rows == 0 and cores // rows <= 8:
                out.append((rows, cores // rows, cores))
                break
    return out


def time_op(fn, device):
    fn()
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(ITERS):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - start) * 1e6 / ITERS


def sweep_decode(device, role, k, n, dtype, rows=32):
    weight = ttnn.from_torch(
        torch.randn(1, 1, k, n) * 0.02,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_sharded_weight_cfg(k, n),
    )
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=LOFI, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    act_torch = torch.randn(1, 1, rows, k) * 0.02
    for grid_rows, grid_cols, cores in grids_for(k // TILE):
        if cores < 8:
            continue
        grid = ttnn.CoreGrid(x=grid_cols, y=grid_rows)
        k_tiles_per_core = k // TILE // cores
        for in0_block_w in sorted({d for d in range(1, k_tiles_per_core + 1) if k_tiles_per_core % d == 0}):
            row = {
                "phase": "decode",
                "role": role,
                "k": k,
                "n": n,
                "dtype": str(dtype),
                "cores": cores,
                "grid": f"{grid_cols}x{grid_rows}",
                "in0_block_w": in0_block_w,
                "per_core_N": math.ceil(n / (TILE * cores)),
            }
            try:
                act = ttnn.from_torch(
                    act_torch,
                    dtype=B16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=width_sharded_l1(rows, k, grid),
                )
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=in0_block_w,
                    per_core_M=rows // TILE,
                    per_core_N=math.ceil(n / (TILE * cores)),
                    fused_activation=None,
                )
                row["us"] = time_op(
                    lambda: ttnn.linear(
                        act,
                        weight,
                        program_config=pc,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                        dtype=B16,
                        compute_kernel_config=ckc,
                    ),
                    device,
                )
                ttnn.deallocate(act)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[2] if len(str(exc).splitlines()) > 2 else exc}"
            emit(**row)
    ttnn.deallocate(weight)


def sweep_prefill(device, role, k, n, dtype, m=2048):
    weight = ttnn.from_torch(
        torch.randn(1, 1, k, n) * 0.02,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_sharded_weight_cfg(k, n),
    )
    interleaved = ttnn.from_torch(
        torch.randn(1, 1, k, n) * 0.02,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    act = ttnn.from_torch(
        torch.randn(1, 1, m, k) * 0.02,
        dtype=B16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=LOFI, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    # Reference point: the ttnn-chosen program on an interleaved weight, which is what the
    # fused stage measured and what the optimized decoder gives up by DRAM-sharding.
    row = {"phase": "prefill", "role": role, "k": k, "n": n, "dtype": str(dtype), "config": "auto_interleaved"}
    try:
        row["us"] = time_op(
            lambda: ttnn.linear(act, interleaved, dtype=B16, compute_kernel_config=ckc), device
        )
    except Exception as exc:
        row["error"] = str(exc).splitlines()[2] if len(str(exc).splitlines()) > 2 else str(exc)
    emit(**row)

    m_tiles = m // TILE
    per_core_N = (dram_sharded_weight_cfg(k, n).shard_spec.shape[1]) // TILE
    for grid_y in (10, 8, 5, 4):
        if m_tiles % grid_y:
            continue
        per_core_M = m_tiles // grid_y
        for in0_block_w in (4, 2, 1):
            if (k // TILE) % in0_block_w:
                continue
            for out_block_h in sorted({d for d in range(1, per_core_M + 1) if per_core_M % d == 0}, reverse=True):
                for out_block_w in sorted(
                    {d for d in range(1, per_core_N + 1) if per_core_N % d == 0}, reverse=True
                ):
                    row = {
                        "phase": "prefill",
                        "role": role,
                        "k": k,
                        "n": n,
                        "dtype": str(dtype),
                        "config": "dram_sharded_weight_2d",
                        "grid": f"{DRAM_BANKS}x{grid_y}",
                        "per_core_M": per_core_M,
                        "per_core_N": per_core_N,
                        "in0_block_w": in0_block_w,
                        "out_block_h": out_block_h,
                        "out_block_w": out_block_w,
                    }
                    try:
                        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(DRAM_BANKS, grid_y),
                            in0_block_w=in0_block_w,
                            out_subblock_h=1,
                            out_subblock_w=next((w for w in (4, 3, 2, 1) if out_block_w % w == 0), 1),
                            out_block_h=out_block_h,
                            out_block_w=out_block_w,
                            per_core_M=per_core_M,
                            per_core_N=per_core_N,
                            transpose_mcast=False,
                            fused_activation=None,
                            fuse_batch=False,
                        )
                        row["us"] = time_op(
                            lambda: ttnn.linear(
                                act, weight, program_config=pc, dtype=B16, compute_kernel_config=ckc
                            ),
                            device,
                        )
                    except Exception as exc:
                        lines = str(exc).splitlines()
                        row["error"] = lines[2] if len(lines) > 2 else str(exc)
                    emit(**row)
                    if "us" in row:
                        break  # largest legal output block for this in0_block_w is enough
                if "us" in row:
                    break
    for tensor in (weight, interleaved, act):
        ttnn.deallocate(tensor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="decode,prefill")
    parser.add_argument("--roles", default="")
    args = parser.parse_args()
    roles = [r for r in ROLES if not args.roles or r[0] in args.roles.split(",")]
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        for phase in args.phase.split(","):
            for role, k, n, dtype in roles:
                (sweep_decode if phase == "decode" else sweep_prefill)(device, role, k, n, dtype)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

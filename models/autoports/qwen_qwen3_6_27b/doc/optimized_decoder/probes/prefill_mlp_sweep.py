# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Focused prefill geometry search for the MLP gate/up projection.

``matmul_sweep.py`` showed the MLP gate/up is the one prefill role where the DRAM-sharded
weight contract costs time rather than saving it: pinning ``per_core_N`` to the weight's DRAM
shard width also pins the grid's x extent to 8 banks, which caps the grid at 80 of Blackhole's
110 workers and leaves each core 68 output tiles.  This sweeps the remaining freedom - grid
rows (with ceil, so 10 is allowed even though 64 M tiles do not divide it), ``in0_block_w`` and
the output block - to find the best legal configuration, and re-measures the packed form under
the same rules.

Emits ``MLPSWEEP `` JSON lines.
"""
from __future__ import annotations

import json
import math
import sys

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")
sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes")

import torch  # noqa: E402
import ttnn  # noqa: E402
from matmul_sweep import B16, B4, B8, DRAM_BANKS, LOFI, TILE, dram_sharded_weight_cfg, time_op  # noqa: E402

M = 2048
SHAPES = [("mlp_gate_split", 5120, 17408, B4), ("mlp_gate_up_packed", 5120, 34816, B4),
          ("mlp_down", 17408, 5120, B8)]


def emit(**payload):
    print("MLPSWEEP " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=LOFI, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        for role, k, n, dtype in SHAPES:
            weight = ttnn.from_torch(
                torch.randn(1, 1, k, n) * 0.02, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=dram_sharded_weight_cfg(k, n),
            )
            act = ttnn.from_torch(
                torch.randn(1, 1, M, k) * 0.02, dtype=B16, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            per_core_N = dram_sharded_weight_cfg(k, n).shard_spec.shape[1] // TILE
            m_tiles = M // TILE
            for grid_y in (10, 8, 6, 5, 4):
                per_core_M = math.ceil(m_tiles / grid_y)
                for in0_block_w in (8, 4, 2):
                    if (k // TILE) % in0_block_w:
                        continue
                    for out_block_h in sorted({d for d in range(1, per_core_M + 1) if per_core_M % d == 0}, reverse=True):
                        for out_block_w in sorted({d for d in range(1, per_core_N + 1) if per_core_N % d == 0}, reverse=True):
                            row = {"role": role, "k": k, "n": n, "grid": f"{DRAM_BANKS}x{grid_y}",
                                   "per_core_M": per_core_M, "per_core_N": per_core_N,
                                   "in0_block_w": in0_block_w, "out_block_h": out_block_h,
                                   "out_block_w": out_block_w}
                            try:
                                pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                                    compute_with_storage_grid_size=(DRAM_BANKS, grid_y),
                                    in0_block_w=in0_block_w, out_subblock_h=1,
                                    out_subblock_w=next((w for w in (4, 3, 2, 1) if out_block_w % w == 0), 1),
                                    out_block_h=out_block_h, out_block_w=out_block_w,
                                    per_core_M=per_core_M, per_core_N=per_core_N,
                                    transpose_mcast=False, fused_activation=None, fuse_batch=False,
                                )
                                row["us"] = time_op(
                                    lambda: ttnn.linear(act, weight, program_config=pc, dtype=B16,
                                                        compute_kernel_config=ckc), device)
                            except Exception as exc:
                                lines = str(exc).splitlines()
                                row["error"] = lines[2] if len(lines) > 2 else str(exc)
                            emit(**row)
                            if "us" in row:
                                break
                        if "us" in row:
                            break
            ttnn.deallocate(weight)
            ttnn.deallocate(act)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Which decode matmul *family* wins for this decoder's shapes?

``$optimize`` treats "DRAM-sharded decode matmul" as mandatory to try, but mandatory to try is
not the same as mandatory to keep.  This compares the three legal families on the real decode
shapes and dtypes, model-free:

``interleaved_auto``
    DRAM-interleaved weight, DRAM-interleaved activation, ttnn-chosen program.  This is what
    the fused stage ran and what the precision sweep measured.
``mcast1d``
    DRAM-interleaved weight, width-sharded L1 activation,
    ``MatmulMultiCoreReuseMultiCast1DProgramConfig(mcast_in0=True)``.
``dram_sharded``
    DRAM-width-sharded weight, width-sharded L1 activation,
    ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``.

Emits ``MMFAM `` JSON lines.
"""
from __future__ import annotations

import json
import math
import sys
import time

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")

import torch  # noqa: E402
import ttnn  # noqa: E402

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes")
from matmul_sweep import (  # noqa: E402
    B16,
    DRAM_BANKS,
    LOFI,
    ROLES,
    TILE,
    dram_sharded_weight_cfg,
    grids_for,
    time_op,
    width_sharded_l1,
)

ROWS = 32


def emit(**payload):
    print("MMFAM " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=LOFI, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        for role, k, n, dtype in ROLES:
            host_w = torch.randn(1, 1, k, n) * 0.02
            act_host = torch.randn(1, 1, ROWS, k) * 0.02
            interleaved_w = ttnn.from_torch(
                host_w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            act_dram = ttnn.from_torch(
                act_host, dtype=B16, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            base = {"role": role, "k": k, "n": n, "dtype": str(dtype),
                    "weight_mb": k * n * {"DataType.BFLOAT4_B": 0.5625, "DataType.BFLOAT8_B": 1.0625,
                                          "DataType.BFLOAT16": 2.0}[str(dtype)] / 1e6}
            row = dict(base, family="interleaved_auto")
            try:
                row["us"] = time_op(
                    lambda: ttnn.linear(act_dram, interleaved_w, dtype=B16, compute_kernel_config=ckc), device
                )
            except Exception as exc:
                row["error"] = str(exc).splitlines()[2] if len(str(exc).splitlines()) > 2 else str(exc)
            emit(**row)

            for grid_rows, grid_cols, cores in grids_for(k // TILE):
                if cores < 8:
                    continue
                grid = ttnn.CoreGrid(x=grid_cols, y=grid_rows)
                per_core_k = k // TILE // cores
                in0_block_w = max(d for d in range(1, per_core_k + 1) if per_core_k % d == 0 and d <= 8)
                act = ttnn.from_torch(
                    act_host, dtype=B16, layout=ttnn.TILE_LAYOUT, device=device,
                    memory_config=width_sharded_l1(ROWS, k, grid),
                )
                row = dict(base, family="mcast1d", cores=cores, grid=f"{grid_cols}x{grid_rows}",
                           in0_block_w=in0_block_w, per_core_N=math.ceil(n / (TILE * cores)))
                try:
                    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(grid_cols, grid_rows),
                        in0_block_w=in0_block_w,
                        out_subblock_h=1,
                        out_subblock_w=1,
                        per_core_M=ROWS // TILE,
                        per_core_N=math.ceil(n / (TILE * cores)),
                        fuse_batch=True,
                        mcast_in0=True,
                    )
                    row["us"] = time_op(
                        lambda: ttnn.linear(
                            act, interleaved_w, program_config=pc, dtype=B16,
                            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=ckc,
                        ),
                        device,
                    )
                except Exception as exc:
                    row["error"] = str(exc).splitlines()[2] if len(str(exc).splitlines()) > 2 else str(exc)
                emit(**row)
                ttnn.deallocate(act)
            ttnn.deallocate(interleaved_w)
            ttnn.deallocate(act_dram)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

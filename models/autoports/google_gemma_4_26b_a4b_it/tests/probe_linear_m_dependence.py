# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Check whether ttnn.linear output rows depend on the physical M under several compute configs."""

from __future__ import annotations

import json
import sys

import torch

import ttnn


def main() -> int:
    device = ttnn.open_device(device_id=0)
    try:
        arch = device.arch()
        torch.manual_seed(0)
        k_dim, n_dim = 2816, 8192
        x = torch.randn(1, 1, 2560, k_dim, dtype=torch.bfloat16) * 4
        w = torch.randn(1, 1, k_dim, n_dim, dtype=torch.bfloat16) * 0.05
        tt_w = ttnn.from_torch(
            w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        config = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        grid = device.compute_with_storage_grid_size()

        def run(m, block_w, sub_h, sub_w):
            tt_x = ttnn.from_torch(
                x[:, :, :m, :],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            m_tiles = m // 32
            per_core_m = -(-m_tiles // grid.y)
            per_core_n = -(-(n_dim // 32) // grid.x)
            while per_core_m % sub_h:
                sub_h -= 1
            while per_core_n % sub_w:
                sub_w -= 1
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                in0_block_w=block_w,
                per_core_M=per_core_m,
                per_core_N=per_core_n,
                out_subblock_h=sub_h,
                out_subblock_w=sub_w,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
            out = ttnn.linear(
                tt_x,
                tt_w,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=config,
                program_config=pc,
            )
            host = ttnn.to_torch(out).float()
            ttnn.deallocate(out)
            ttnn.deallocate(tt_x)
            return host, (sub_h, sub_w)

        base, _ = run(2560, 8, 1, 1)
        for m in (2560, 1536, 1024, 128, 64):
            for block_w, sub_h, sub_w in ((8, 1, 1), (8, 2, 4), (8, 1, 8), (8, 4, 2), (4, 1, 1), (11, 1, 1), (2, 1, 1)):
                try:
                    out, used = run(m, block_w, sub_h, sub_w)
                except Exception as error:
                    print(
                        json.dumps(
                            {
                                "m": m,
                                "block_w": block_w,
                                "subblock": [sub_h, sub_w],
                                "error": str(error).splitlines()[0][:120],
                            }
                        ),
                        flush=True,
                    )
                    continue
                diff = (base[..., :m, :] - out).abs()
                print(
                    json.dumps(
                        {
                            "m": m,
                            "block_w": block_w,
                            "subblock": list(used),
                            "nonzero_fraction_vs_m2560_bw8_s11": round(float((diff > 0).float().mean()), 6),
                            "max_abs": float(diff.max()),
                        }
                    ),
                    flush=True,
                )
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())

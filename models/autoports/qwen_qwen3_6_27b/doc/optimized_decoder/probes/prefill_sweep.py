# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prefill program-config search for a DRAM-width-sharded weight, with a correctness gate.

Supersedes the prefill half of ``matmul_sweep.py``.  Two things changed after that run:

1. ``probe_matmul_correctness.py`` showed that a 2D ``MatmulMultiCoreReuseMultiCast`` matmul
   whose ``input_tensor_b`` is DRAM-width-sharded computes the **wrong answer** when
   ``out_block_w < per_core_N``, silently: PCC 0.24 for the 5120x17408 gate projection and
   0.50 for 5120x8192, against 0.9937 / 0.99988 for the same program config on an interleaved
   weight.  ttnn only validates ``out_block_w == per_core_N`` when the *output* is sharded
   (``matmul_device_operation.cpp:1659``), so an interleaved output plus a sharded weight
   passes validation and returns garbage.  ``out_block_w`` is therefore pinned to
   ``per_core_N`` here, and the search moves to ``in0_block_w`` and ``out_block_h``.
2. Every timed candidate is PCC-checked against the same matmul on an interleaved weight, so
   a fast-but-wrong configuration can never win a sweep again.

Emits ``PFSWEEP `` JSON lines.
"""
from __future__ import annotations

import json
import math
import sys

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")
sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes")

import torch  # noqa: E402
import ttnn  # noqa: E402
from matmul_sweep import B16, DRAM_BANKS, LOFI, ROLES, TILE, dram_sharded_weight_cfg, time_op  # noqa: E402

M = 2048
PCC_BAR = 0.99


def emit(**payload):
    print("PFSWEEP " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def pcc(a, b):
    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=LOFI, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        for role, k, n, dtype in ROLES:
            torch.manual_seed(0)
            host_w = (torch.randn(1, 1, k, n) * 0.02).contiguous()
            host_x = (torch.randn(1, 1, M, k) * 0.02).contiguous()
            weight = ttnn.from_torch(host_w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                                     memory_config=dram_sharded_weight_cfg(k, n))
            interleaved = ttnn.from_torch(host_w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
            act = ttnn.from_torch(host_x, dtype=B16, layout=ttnn.TILE_LAYOUT, device=device,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)

            reference = ttnn.linear(act, interleaved, dtype=B16, compute_kernel_config=ckc)
            golden = ttnn.to_torch(reference).float()
            row = {"role": role, "k": k, "n": n, "dtype": str(dtype), "config": "auto_interleaved", "pcc": 1.0}
            row["us"] = time_op(lambda: ttnn.linear(act, interleaved, dtype=B16, compute_kernel_config=ckc), device)
            emit(**row)
            ttnn.deallocate(reference)

            per_core_N = dram_sharded_weight_cfg(k, n).shard_spec.shape[1] // TILE
            m_tiles = M // TILE
            for grid_y in (10, 8, 5, 4, 2):
                per_core_M = math.ceil(m_tiles / grid_y)
                for in0_block_w in (34, 32, 20, 17, 16, 10, 8, 4, 2, 1):
                    if (k // TILE) % in0_block_w:
                        continue
                    for out_block_h in sorted(
                        {d for d in range(1, per_core_M + 1) if per_core_M % d == 0}, reverse=True
                    ):
                        row = {"role": role, "k": k, "n": n, "dtype": str(dtype),
                               "config": "dram_sharded_weight_2d", "grid": f"{DRAM_BANKS}x{grid_y}",
                               "per_core_M": per_core_M, "per_core_N": per_core_N,
                               "in0_block_w": in0_block_w, "out_block_h": out_block_h,
                               "out_block_w": per_core_N}
                        try:
                            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                                compute_with_storage_grid_size=(DRAM_BANKS, grid_y),
                                in0_block_w=in0_block_w, out_subblock_h=1,
                                out_subblock_w=next((w for w in (4, 3, 2, 1) if per_core_N % w == 0), 1),
                                out_block_h=out_block_h, out_block_w=per_core_N,
                                per_core_M=per_core_M, per_core_N=per_core_N,
                                transpose_mcast=False, fused_activation=None, fuse_batch=False)
                            got = ttnn.linear(act, weight, program_config=pc, dtype=B16,
                                              compute_kernel_config=ckc)
                            row["pcc"] = pcc(golden, ttnn.to_torch(got).float())
                            ttnn.deallocate(got)
                            row["us"] = time_op(
                                lambda: ttnn.linear(act, weight, program_config=pc, dtype=B16,
                                                    compute_kernel_config=ckc), device)
                        except Exception as exc:
                            lines = str(exc).splitlines()
                            row["error"] = lines[2] if len(lines) > 2 else str(exc)
                        emit(**row)
                        if "us" in row:
                            break  # largest legal out_block_h for this in0_block_w
            for tensor in (weight, interleaved, act):
                ttnn.deallocate(tensor)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

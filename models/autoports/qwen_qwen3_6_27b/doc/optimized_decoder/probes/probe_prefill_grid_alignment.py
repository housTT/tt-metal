# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Why a prefill 2D matmul over a DRAM width-sharded weight needs one compute column per DRAM bank.

``_prefill_program_cfg`` pins the compute grid's column count to this chip's DRAM bank count and cites
this probe for it.  The rule is not a validation error the op reports - it *passes* validation and then
returns non-finite values - so it has to be measured, and this is the measurement.

For every prefill role whose weight lives in a DRAM width-sharded memory config, it runs the same 2D
program config at every legal column count (each divisor of the output tile count that fits the compute
grid) and reports whether the result is finite and how well it correlates with the *same* matmul run
through ``ttnn.linear``'s own heuristic on an interleaved copy of the same weight.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_prefill_grid_alignment.py
"""

from __future__ import annotations

import json
import math
import sys

import torch

import ttnn

TILE = 32
ROWS = 2048
#: ``(role, K, N, weight dtype, out dtype)`` at the shipped policy's dtypes.
ROLES = [
    ("wqkv", 5120, 8192, ttnn.bfloat8_b, ttnn.bfloat16),
    ("wgate", 5120, 6144, ttnn.bfloat8_b, ttnn.bfloat16),
    ("o_proj", 6144, 5120, ttnn.bfloat8_b, ttnn.bfloat16),
    ("mlp_gate", 5120, 17408, ttnn.bfloat4_b, ttnn.bfloat16),
    ("mlp_down", 17408, 5120, ttnn.bfloat8_b, ttnn.bfloat16),
    ("in_proj_qkv", 5120, 10240, ttnn.bfloat8_b, ttnn.float32),
    ("in_proj_z", 5120, 6144, ttnn.bfloat8_b, ttnn.bfloat16),
    ("out_proj", 6144, 5120, ttnn.bfloat8_b, ttnn.bfloat16),
]
_DEVICE = None


def _largest_at_most(value, limit):
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def dram_sharded_cfg(k, n):
    dram = _DEVICE.dram_grid_size()
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
    padded = math.ceil(n / (TILE * dram.x)) * TILE * dram.x
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, padded // dram.x), ttnn.ShardOrientation.ROW_MAJOR),
    )


def pcc(golden, actual):
    a = golden.to(torch.float64).flatten()
    b = actual.to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> int:
    global _DEVICE
    _DEVICE = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    grid = _DEVICE.compute_with_storage_grid_size()
    banks = _DEVICE.dram_grid_size().x
    results: list = []
    try:
        print(f"compute grid {grid.x}x{grid.y}; DRAM banks {banks}")
        for role, k, n, wdt, odt in ROLES:
            torch.manual_seed(0)
            act_t = torch.randn(1, 1, ROWS, k) * 0.05
            w_t = torch.randn(1, 1, k, n) * 0.02
            fp32 = odt == ttnn.float32
            ck = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2 if fp32 else ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32,
                packer_l1_acc=True,
            )
            act = ttnn.from_torch(
                act_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=_DEVICE,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            w_inter = ttnn.from_torch(
                w_t, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=_DEVICE, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            w_shard = ttnn.from_torch(
                w_t, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=_DEVICE, memory_config=dram_sharded_cfg(k, n)
            )
            # Reference: the heuristic on the interleaved copy of the same weight.
            reference = ttnn.to_torch(ttnn.linear(act, w_inter, dtype=odt, compute_kernel_config=ck)).float()
            k_tiles, n_tiles, m_tiles = k // TILE, n // TILE, ROWS // TILE
            y = _largest_at_most(m_tiles, grid.y)
            for x in [c for c in range(1, grid.x + 1) if n_tiles % c == 0]:
                per_core_m, per_core_n = math.ceil(m_tiles / y), n_tiles // x
                out_block_w, out_block_h = per_core_n, 1
                for candidate in [d for d in range(1, per_core_m + 1) if per_core_m % d == 0]:
                    if candidate * out_block_w <= 160:
                        out_block_h = candidate
                budget = 2 if fp32 else 4
                sw = _largest_at_most(out_block_w, budget)
                sh = _largest_at_most(out_block_h, max(1, budget // sw))
                row = {
                    "sweep": "prefill_grid_alignment",
                    "role": role,
                    "K": k,
                    "N": n,
                    "columns": x,
                    "rows_of_cores": y,
                    "dram_banks": banks,
                    "per_core_N": per_core_n,
                }
                try:
                    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(x, y),
                        in0_block_w=_largest_at_most(k_tiles, 4),
                        out_subblock_h=sh,
                        out_subblock_w=sw,
                        out_block_h=out_block_h,
                        out_block_w=out_block_w,
                        per_core_M=per_core_m,
                        per_core_N=per_core_n,
                        transpose_mcast=False,
                        fused_activation=None,
                    )
                    out = ttnn.linear(
                        act,
                        w_shard,
                        program_config=pc,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=odt,
                        compute_kernel_config=ck,
                    )
                    host = ttnn.to_torch(out).float()
                    row["finite"] = bool(torch.isfinite(host).all())
                    row["pcc_vs_heuristic"] = pcc(reference, host) if row["finite"] else None
                    ttnn.deallocate(out)
                except Exception as exc:  # noqa: BLE001 - a blocker is a result
                    row["finite"] = None
                    row["error"] = str(exc)[:160]
                results.append(row)
                print(
                    f"  {role:12s} N={n_tiles:5d} tiles  columns={x:3d} (banks={banks})  "
                    f"finite={row['finite']}  pcc_vs_heuristic="
                    f"{row['pcc_vs_heuristic'] if row.get('pcc_vs_heuristic') is not None else float('nan'):.6f}"
                    + (f"  ERROR {row['error']}" if row.get("error") else ""),
                    flush=True,
                )
            for tensor in (act, w_inter, w_shard):
                if tensor.is_allocated():
                    ttnn.deallocate(tensor)
    finally:
        for row in results:
            print("PROBEROW " + json.dumps(row, sort_keys=True, default=str), flush=True)
        ttnn.close_mesh_device(_DEVICE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

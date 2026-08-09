# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Is a DRAM-width-sharded weight numerically the same weight?

Isolates the O2 contract from the model: same torch weight, three matmul families, compared
against a torch reference.  Written because the first end-to-end O2 run produced PCC ~0 in
both layer kinds, which is an implementation bug rather than a precision effect.
"""
from __future__ import annotations

import json
import math
import sys

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")
sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes")

import torch  # noqa: E402
import ttnn  # noqa: E402
from matmul_sweep import B16, DRAM_BANKS, LOFI, TILE, dram_sharded_weight_cfg, width_sharded_l1  # noqa: E402

from models.autoports.qwen_qwen3_6_27b.tt import optimized_decoder as O  # noqa: E402


def pcc(a, b):
    a = a.to(torch.float64).flatten() - a.to(torch.float64).mean()
    b = b.to(torch.float64).flatten() - b.to(torch.float64).mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=LOFI, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        from matmul_sweep import ROLES
        for _role, k, n, dtype in [r for r in ROLES if r[0] != "mlp_gate_up_packed"]:
            torch.manual_seed(0)
            w = torch.randn(k, n) * 0.02
            for m in (64, 256, 768, 2048):
                x = torch.randn(1, 1, m, k) * 0.02
                golden = (x.reshape(m, k) @ w).reshape(1, 1, m, n)
                w_tt = w.reshape(1, 1, k, n)
                inter = ttnn.from_torch(w_tt, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
                shard = ttnn.from_torch(w_tt, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                                        memory_config=dram_sharded_weight_cfg(k, n))
                x_dram = ttnn.from_torch(x, dtype=B16, layout=ttnn.TILE_LAYOUT, device=device,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
                out = {}
                out["interleaved_auto"] = ttnn.to_torch(
                    ttnn.linear(x_dram, inter, dtype=B16, compute_kernel_config=ckc)).float()
                if m == 0:
                    cores, grid = 32, ttnn.CoreGrid(x=8, y=4)
                    x_sh = ttnn.from_torch(x, dtype=B16, layout=ttnn.TILE_LAYOUT, device=device,
                                           memory_config=width_sharded_l1(m, k, grid))
                    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=(k // TILE) // cores, per_core_M=1,
                        per_core_N=math.ceil(n / (TILE * cores)), fused_activation=None)
                    out["dram_sharded"] = ttnn.to_torch(
                        ttnn.linear(x_sh, shard, program_config=pc, dtype=B16,
                                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                    compute_kernel_config=ckc)).float()
                else:
                    # Every candidate the layer would consider, in its own order, so the sweep
                    # answers "is the config the layer picks correct at this M?".
                    dec = O.OptimizedDecoder.__new__(O.OptimizedDecoder)
                    dec.topology = O.DEFAULT_TOPOLOGY
                    for rank, pc in enumerate(dec._prefill_program_configs(m, k, shard)[:6]):
                        label = (f"r{rank}_g{pc.compute_with_storage_grid_size}"
                                 f"_pcM{pc.per_core_M}_obh{pc.out_block_h}_ibw{pc.in0_block_w}")
                        try:
                            out[label] = ttnn.to_torch(
                                ttnn.linear(x_dram, shard, program_config=pc, dtype=B16,
                                            compute_kernel_config=ckc)).float()
                        except Exception as exc:
                            lines = str(exc).splitlines()
                            out[label] = lines[2] if len(lines) > 2 else str(exc)
                for name, got in out.items():
                    value = got if isinstance(got, str) else pcc(golden, got[..., :n])
                    print(json.dumps({"k": k, "n": n, "m": m, "dtype": str(dtype),
                                      "family": name, "pcc": value}), flush=True)
                for t in (inter, shard, x_dram):
                    ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolation sweep for the terminal reconstruction-readout matmul (LM-head analog).

Shape per sequence shard: [1,1,local,768] @ [768,192] -> [1,1,local,192], bf16.
The stage-05 path used core_grid=CoreGrid(8,10) auto (perf report: SLOW,
in0_block_w=1, 6 cores). Compare against explicit MatmulMultiCoreReuseMultiCast
program configs with a K-dividing in0_block_w.
"""
import time

import torch

import ttnn

H, VOCAB_PAD = 768, 192


def _t(fn, mesh, iters=100):
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / iters * 1e6  # us


def main():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=90000000)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        w = ttnn.from_torch(
            torch.randn(H, VOCAB_PAD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )
        b = ttnn.from_torch(
            torch.zeros(1, VOCAB_PAD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )
        for full_seq in (128, 512):
            local = full_seq // 4
            x = ttnn.from_torch(
                torch.randn(1, 1, local, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
            )
            m_tiles = local // 32

            def auto():
                o = ttnn.linear(
                    x, w, bias=b, compute_kernel_config=ck, core_grid=ttnn.CoreGrid(y=8, x=10), dtype=ttnn.bfloat16
                )
                ttnn.deallocate(o)

            t_auto = _t(auto, mesh)

            # explicit 2D: grid gx covers N (6 tiles -> gx=6), gy covers M
            results = {"auto": round(t_auto, 1)}
            for gx in (6,):
                for gy in [g for g in (1, 2, 4) if m_tiles % g == 0] or [1]:
                    for in0bw in (8, 12, 24):
                        try:
                            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                                compute_with_storage_grid_size=(gx, gy),
                                in0_block_w=in0bw,
                                out_subblock_h=1,
                                out_subblock_w=1,
                                per_core_M=max(1, m_tiles // gy),
                                per_core_N=VOCAB_PAD // 32 // gx,
                                transpose_mcast=False,
                                fused_activation=None,
                            )

                            def ex():
                                o = ttnn.linear(
                                    x, w, bias=b, compute_kernel_config=ck, program_config=pc, dtype=ttnn.bfloat16
                                )
                                ttnn.deallocate(o)

                            results[f"gx{gx}_gy{gy}_bw{in0bw}"] = round(_t(ex, mesh), 1)
                        except Exception as e:
                            results[f"gx{gx}_gy{gy}_bw{in0bw}"] = f"ERR:{str(e)[:40]}"
            print(f"full_seq={full_seq} local={local} m_tiles={m_tiles}:", results, flush=True)
            ttnn.deallocate(x)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Adapted-attempt probe for fused CCL+matmul on the Kokoro multichip shapes.

The attention block is the only place with collectives:
  * all_gather (residual->full seq) + QKV matmul  -> all_gather_matmul_async
  * WO matmul + reduce_scatter (partial->seq shard) -> matmul_reduce_scatter_async

Separate-op device times are already known from the stage-03 tt-perf-report
(per layer, per device): AllGather ~21us, QKV matmul ~12us, WO matmul ~7us,
ReduceScatter ~30us -> AG+QKV ~33us, WO+RS ~37us. This probe attempts the FUSED
ops with correct sub-device / semaphore / core-grid-offset / matmul-program-config
setup and captures whether they run for our exact shapes and their traced time, or
the precise op-contract blocker.
"""
import os
import time
import traceback

import torch

import ttnn
from models.common.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    get_num_links,
    get_tt_ccl,
)

MESH_SHAPE = (1, 4)
H = 768
GW = 192
LOCAL_SEQ = 128
FULL_SEQ = 512
REPS = 12
ITERS = 50
LOG = []


def rec(m):
    print(m, flush=True)
    LOG.append(m)


def time_trace(mesh, fn):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(REPS):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / ITERS * 1e3
    ttnn.release_trace(mesh, tid)
    return dt


def main():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(MESH_SHAPE), trace_region_size=90000000)
    nl = get_num_links(mesh)
    cg = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cg.x - 1, cg.y - 1))})
    sdm = mesh.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh.load_sub_device_manager(sdm)
    mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    sub_id = ttnn.SubDeviceId(0)
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    # ---- AG + QKV ----
    try:
        res = ttnn.from_torch(
            torch.randn(1, 1, LOCAL_SEQ, H).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 2), mesh_shape=MESH_SHAPE),
        )
        qkv_w = ttnn.from_torch(
            torch.randn(1, 1, H, 3 * GW).bfloat16(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,  # rank-4 required
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ag_sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
        barrier = ttnn.create_global_semaphore(mesh, crs, 0)
        # matmul on the lower 6 rows (AG runs on rows 6+ via offset (0,6))
        mm_pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(cg.x, 6),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=(FULL_SEQ // 32 + 5) // 6,
            per_core_N=(3 * GW // 32 + cg.x - 1) // cg.x,
            transpose_mcast=False,
            fused_activation=None,
        )

        def fused_ag_mm():
            _, mm = ttnn.experimental.all_gather_matmul_async(
                res,
                qkv_w,
                dim=2,
                all_gather_core_grid_offset=(0, 6),
                multi_device_global_semaphore=ag_sems,
                barrier_semaphore=barrier,
                num_links=nl,
                topology=ttnn.Topology.Ring,
                subdevice_id=sub_id,
                program_config=mm_pc,
                compute_kernel_config=ck,
            )
            return mm

        t = time_trace(mesh, fused_ag_mm)
        rec(f"[AG+QKV] FUSED runs: {t*1000/REPS:.2f} us/rep vs separate ~33 us/rep")
    except Exception as e:
        rec(f"[AG+QKV] FUSED blocked: {repr(e)[:500]}")
        rec("  " + traceback.format_exc().splitlines()[-1][:200])

    # ---- WO + RS ----
    # REAL head-parallel layout: post-nlp_concat_heads each device holds its OWN 3
    # heads -> a FULL local [b, full_seq, 192] (NOT a dim-3 shard of a shared 192).
    # dense_w is [192, 768] per device (row/contraction-dim fractured). The local
    # matmul yields a [full_seq, 768] partial; reduce_scatter(dim=2) sums the
    # partials across devices and scatters over sequence. Replicate the probe
    # input/weight (timing/feasibility is value-independent) so each device gets a
    # full 192-wide input -> matmul in width 192 == weight height 192.
    try:
        attn = ttnn.from_torch(
            torch.randn(1, 1, FULL_SEQ, GW).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        dense_w = ttnn.from_torch(
            torch.randn(1, 1, GW, H).bfloat16(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,  # rank-4
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # ---- separate WO matmul + reduce_scatter baseline (same harness) ----
        tt_ccl = get_tt_ccl(mesh)
        core_grid_sep = ttnn.CoreGrid(y=min(8, cg.y), x=min(10, cg.x))

        def sep_wo_rs():
            mm = ttnn.linear(attn, dense_w, compute_kernel_config=ck, core_grid=core_grid_sep, dtype=ttnn.bfloat16)
            rs = ttnn.experimental.reduce_scatter_minimal_async(
                mm,
                persistent_output_buffers=None,
                dim=2,
                multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
                barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=nl,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                chunks_per_sync=CCL_CHUNKS_PER_SYNC,
                num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
                num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
            )
            ttnn.deallocate(mm)
            return rs

        t_sep = time_trace(mesh, sep_wo_rs)
        rec(f"[WO+RS] separate (same harness): {t_sep*1000/REPS:.2f} us/rep  ({t_sep:.4f} ms x{REPS})")

        rs_sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(3)]
        rs_interm = ttnn.from_torch(
            torch.zeros(1, 1, FULL_SEQ, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        rs_out = ttnn.from_torch(
            torch.zeros(1, 1, FULL_SEQ // 4, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        mm_pc2 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(cg.x, 6),
            in0_block_w=GW // 32,
            out_subblock_h=1,
            out_subblock_w=1,  # out_block_w(=per_core_N=3) % 1 == 0
            per_core_M=(FULL_SEQ // 32 + 5) // 6,
            per_core_N=(H // 32 + cg.x - 1) // cg.x,
            transpose_mcast=False,
            fused_activation=None,
        )

        def fused_mm_rs():
            _, rs = ttnn.experimental.matmul_reduce_scatter_async(
                attn,
                dense_w,
                rs_interm,
                rs_out,
                dim=2,
                reduce_scatter_core_grid_offset=(0, 6),
                multi_device_global_semaphore=rs_sems,
                num_links=nl,
                topology=ttnn.Topology.Ring,
                subdevice_id=sub_id,
                program_config=mm_pc2,
                compute_kernel_config=ck,
                memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            )
            return rs

        t2 = time_trace(mesh, fused_mm_rs)
        rec(f"[WO+RS] FUSED runs: {t2*1000/REPS:.2f} us/rep vs separate ~37 us/rep")
    except Exception as e:
        rec(f"[WO+RS] FUSED blocked: {repr(e)[:500]}")
        rec("  " + traceback.format_exc().splitlines()[-1][:200])

    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    with open(os.path.join(os.path.dirname(__file__), "fused_ccl_matmul.log"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

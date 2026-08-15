# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The fused matmul+collective family, at this layer's real row- and column-parallel boundaries.

``$multichip`` requires the pre-coding topology table to contain, for every "local result then
collective" boundary: local matmul plus all-reduce, reduce-scatter plus delayed gather, **fused
all-gather-matmul where the next matmul consumes gathered input**, **fused matmul plus
reduce-scatter/all-reduce where supported**, and a residual-sharded variant. Rounds 0-2 of this stage
measured the first, second and last (``probe_ccl.txt``) and skipped the two fused rows while a module
docstring claimed they had been measured. Review round 3 found that. This probe is the missing rows.

The two boundaries, at their **per-device** shapes:

``o_proj`` (row-parallel, the shipped shape)
    each device holds ``[M, 1024] x [1024, 2048]`` and produces a partial sum that must be reduced
    across the mesh. Arms:

    * ``mm_then_all_reduce`` — what ships: ``ttnn.linear`` then ``ttnn.all_reduce``.
    * ``mm_then_stack_sum`` — what ships at the decode tile: ``ttnn.linear`` then all-gather-on-a-new
      -axis plus a local sum.
    * ``fused_mm_rs_then_ag`` — ``ttnn.experimental.matmul_reduce_scatter_async`` (fused) followed by
      the ``all_gather`` needed to restore the replicated residual. Like-for-like with the two above:
      same input, same replicated output.
    * ``fused_mm_rs_only`` — the same fused op **without** the gather. This is not an equal-output
      arm; it is the sharded-residual variant, and it is the lower bound that design could reach at
      this boundary.

``attn_in`` (column-parallel, the *consumer* a sharded residual would need)
    under a sharded residual each device would hold a quarter of the hidden dim and the
    in-projection needs all of it. Arms:

    * ``ag_then_mm`` — ``ttnn.all_gather`` then ``ttnn.linear`` at ``[M, 2048] x [2048, 2560]``.
    * ``fused_ag_mm`` — ``ttnn.experimental.all_gather_matmul_async``. **DANGEROUS, off by default.**
      It hangs the mesh at the decode shape: `AllGatherMatmulAsyncDeviceOperation` on
      ``[1,1,32,512] x [1,1,2048,2560]`` with a ``[1,1,32,2048]`` persistent output sat on all four
      devices and 40 cores for ten minutes with no progress and had to be killed
      (`doc/multichip_decoder/triage/`). Devices recovered without a reset. `--fused-ag` re-enables
      it; do not pass it without being ready to reset.
    * ``mm_replicated`` — the **shipped** contract's cost at this boundary: no collective at all,
      because the residual is already replicated. This is the row the other two have to beat for a
      sharded residual to be worth adopting.

Everything is measured inside a captured trace as well as eagerly, for the reason ``probe_ccl.py``
gives: a decode step replays from a trace and the eager launch floor is not paid there.

    python .../doc/multichip_decoder/logs/probe_fused_ccl.py

Rows are ``FUSED <boundary> <shape> <arm> <mode> <us> <pcc>``; ``pcc`` is against a torch reference
of the same reduction, so an arm that is fast because it computes something else shows up as a
correctness failure rather than as a win. A refused op prints ``FAIL`` with the exact blocker.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC

#: ``M`` values: the batch-1 decode tile, the batch-32 decode tile, and the shipped prefill chunk.
ROWS = [("decode", 32), ("decode_b32", 32 * 32), ("prefill_2048", 2048)]

#: Copies captured inside one trace, so the fixed per-replay dispatch is amortised. Same reason and
#: same value as ``probe_ccl.py``.
REPS = 8

DIM = 2048
O_PROJ_K = 1024  # 4096 local heads / tp
ATTN_IN_N = 2560  # 9216 packed / tp, rounded to the local packing


def timed_eager(mesh, build, iters):
    for _ in range(3):
        ttnn.deallocate(build())
    ttnn.synchronize_device(mesh)
    start = time.time()
    for _ in range(iters):
        ttnn.deallocate(build())
    ttnn.synchronize_device(mesh)
    return (time.time() - start) / iters * 1e6


def timed_trace(mesh, build, iters):
    ttnn.deallocate(build())
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = [build() for _ in range(REPS)]
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(3):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    start = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    elapsed = (time.time() - start) / iters / REPS * 1e6
    for o in outs:
        ttnn.deallocate(o)
    ttnn.release_trace(mesh, tid)
    return elapsed


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


#: ``boundary arm residual_in= residual_out= collective= ccl_dtype= persistent_buffers=`` for every
#: arm below. ``residual_*`` is the layout of the 2048-wide hidden stream entering and leaving the
#: boundary, which is the thing a sharded-residual contract would change; ``collective`` is where the
#: fabric traffic sits relative to the matmul; ``persistent_buffers`` is what the async ops require to
#: be preallocated (the unfused arms require none, which is part of their cost story).
ARM_CONTRACTS = [
    "o_proj mm_then_all_reduce residual_in=replicated residual_out=replicated collective=after_matmul "
    "ccl_dtype=bfloat16 persistent_buffers=none",
    "o_proj mm_then_stack_sum residual_in=replicated residual_out=replicated collective=after_matmul "
    "ccl_dtype=bfloat16 persistent_buffers=none",
    "o_proj fused_mm_rs_then_ag residual_in=replicated residual_out=replicated collective=fused_into_matmul+gather "
    "ccl_dtype=bfloat16 persistent_buffers=intermediate+output+3_semaphores",
    "o_proj fused_mm_rs_only residual_in=replicated residual_out=sharded_dim3_4way collective=fused_into_matmul "
    "ccl_dtype=bfloat16 persistent_buffers=intermediate+output+3_semaphores",
    "attn_in mm_replicated residual_in=replicated residual_out=sharded_heads collective=none "
    "ccl_dtype=n/a persistent_buffers=none",
    "attn_in ag_then_mm residual_in=sharded_dim3_4way residual_out=sharded_heads collective=before_matmul "
    "ccl_dtype=bfloat16 persistent_buffers=none",
    "attn_in fused_ag_mm residual_in=sharded_dim3_4way residual_out=sharded_heads collective=fused_into_matmul "
    "ccl_dtype=bfloat16 persistent_buffers=output+2_semaphores (HUNG - see triage/)",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--links", type=int, default=2)
    ap.add_argument(
        "--shapes",
        default=",".join(name for name, _ in ROWS),
        help="comma-separated subset of the M points. run_evidence.sh runs one process per shape "
        "under `timeout`, so a fused-CCL hang cannot take the rest of the sweep with it.",
    )
    ap.add_argument(
        "--fused-ag",
        action="store_true",
        help="DANGEROUS. Also time all_gather_matmul_async, which HUNG the mesh at the decode shape "
        "(see the module docstring and doc/multichip_decoder/triage/). Off by default so this probe "
        "is safe to run from run_evidence.sh.",
    )
    args = ap.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=MC.fabric_router_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=0)
    n = mesh.get_num_devices()
    grid = mesh.compute_with_storage_grid_size()
    offset = ttnn.CoreCoord(0, 0)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    print(f"# fused matmul+CCL sweep, {n} devices, FABRIC_1D_RING, num_links={args.links}, grid {grid.x}x{grid.y}")
    # The per-arm contract `$multichip` asks to be recorded for every candidate at a row- or
    # column-parallel boundary. In the artifact rather than only in prose, because these are the
    # attributes that decide whether two arms are comparable at all.
    print("# columns: boundary shape arm mode us pcc")
    for line in ARM_CONTRACTS:
        print(f"# ARM {line}")
    gen = torch.Generator().manual_seed(5)

    def upload(t, shard_dim=None, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=(
                ttnn.shard_tensor_to_mesh_mapper(mesh, dim=shard_dim)
                if shard_dim is not None
                else ttnn.replicate_tensor_to_mesh_mapper(mesh)
            ),
        )

    def run(tag, arms, reference, per_device):
        """``per_device``: every device holds the whole answer (take device 0); else compose dim 3."""
        for arm, build in arms.items():
            try:
                probe = build()
                got = ttnn.to_torch(
                    probe, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, dim=0 if per_device else 3)
                )
                value = pcc(reference, got[0, 0] if per_device else got[0, 0])
                ttnn.deallocate(probe)
            except Exception as exc:  # noqa: BLE001 - a refused op is the result
                print(f"FUSED {tag} {arm} - FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:120]}", flush=True)
                continue
            for mode, fn in (("eager", timed_eager), ("trace", timed_trace)):
                try:
                    us = fn(mesh, build, iters=args.iters)
                    print(f"FUSED {tag} {arm} {mode} {us:.2f} {value:.6f}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"FUSED {tag} {arm} {mode} FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:120]}",
                        flush=True,
                    )

    def sems(k):
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        return [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(k)]

    def empty(shape, shard_dim=None):
        return upload(torch.zeros(*shape), shard_dim=shard_dim)

    def fused_program_config(cg, m, k_local, n):
        """Matmul program config for a CCL-fused matmul, restricted to ``cg`` so the collective's
        workers get rows the matmul does not use.

        A full-grid matmul plus a fused CCL **deadlocks** — the two want the same cores — so the
        matmul is confined to the first ``cg.y`` rows and the collective's ``core_grid_offset`` puts
        its workers below them. That constraint is why these arms cannot simply reuse the shipped
        `_decode_1d_matmul_config`, and it is itself part of the cost: the fused arm's matmul runs on
        a smaller grid than the unfused one's. Shape follows the reference implementation in
        `models/demos/blackhole/qwen36/tt/tp_common.py`, which is the only in-tree user of this op.
        """
        per_core_n = max(1, math.ceil(n / 32 / cg[0]))
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=cg,
            in0_block_w=min(4, max(1, k_local // 32 // cg[0])),
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=max(1, math.ceil(m / 32 / cg[1])),
            per_core_N=per_core_n,
            out_block_w=max(1, per_core_n // 2),
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
            allowed_worker_cores=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cg[0] - 1, cg[1] - 1))}
            ),
        )

    wanted = [w.strip() for w in args.shapes.split(",")]
    try:
        for shape_name, m in [r for r in ROWS if r[0] in wanted]:
            # ---------------- row-parallel boundary: o_proj -> reduce
            act_host = torch.randn(n, 1, m, O_PROJ_K, generator=gen) * 0.1
            w_host = torch.randn(n, 1, O_PROJ_K, DIM, generator=gen) * 0.05
            act = upload(act_host, shard_dim=0)
            w = upload(w_host, shard_dim=0)
            ref = sum(act_host[d, 0].double() @ w_host[d, 0].double() for d in range(n)).float()

            # The fused op needs a persistent intermediate (the un-reduced matmul output) and a
            # persistent output (the scattered result), plus its own semaphores. Allocating them here
            # rather than per call is the persistent-buffer plan the skill asks the table to record:
            # a per-call allocation would not be trace-safe and would not be comparable to the
            # unfused arms, which allocate nothing beyond their outputs.
            # Replicated, not sharded: the op writes each device's own slice into its own copy.
            rs_inter = empty((1, 1, m, DIM))
            rs_out = empty((1, 1, m, DIM // n))
            rs_sems = sems(3)
            rs_barrier = sems(1)[0]
            # Matmul on rows 0..cg_y-1, reduce-scatter workers below them.
            rs_cg = (grid.x, grid.y - 2)
            rs_pc = fused_program_config(rs_cg, m, O_PROJ_K, DIM)
            rs_offset = ttnn.CoreCoord(0, rs_cg[1])

            def mm():
                return ttnn.linear(act, w, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            def mm_then_all_reduce():
                part = mm()
                out = ttnn.all_reduce(
                    part, topology=ttnn.Topology.Ring, num_links=args.links, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ttnn.deallocate(part)
                return out

            def mm_then_stack_sum():
                part = mm()
                gathered = ttnn.all_gather(part, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(part)
                out = ttnn.sum(gathered, dim=0, keepdim=True)
                ttnn.deallocate(gathered)
                return out

            def fused_mm_rs():
                return ttnn.experimental.matmul_reduce_scatter_async(
                    act,
                    w,
                    persistent_intermediate_buffer=rs_inter,
                    persistent_output_buffer=rs_out,
                    dim=3,
                    multi_device_global_semaphore=rs_sems,
                    reduce_scatter_core_grid_offset=rs_offset,
                    barrier_semaphore=rs_barrier,
                    num_links=args.links,
                    memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Ring,
                    subdevice_id=None,
                    memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=rs_pc,
                    compute_kernel_config=ckc,
                )

            def fused_mm_rs_then_ag():
                scattered = fused_mm_rs()[-1]
                # Cloned because `rs_out` is the persistent buffer and the caller must not free it.
                return ttnn.all_gather(scattered, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            def fused_mm_rs_only():
                return ttnn.clone(fused_mm_rs()[-1], memory_config=ttnn.DRAM_MEMORY_CONFIG)

            run(
                f"o_proj {shape_name}",
                {
                    "mm_then_all_reduce": mm_then_all_reduce,
                    "mm_then_stack_sum": mm_then_stack_sum,
                    "fused_mm_rs_then_ag": fused_mm_rs_then_ag,
                },
                ref,
                per_device=True,
            )
            # The sharded-residual variant leaves the result fractured on the last dim, so its
            # correctness check composes that axis instead of taking one device's copy.
            run(f"o_proj {shape_name}", {"fused_mm_rs_only": fused_mm_rs_only}, ref, per_device=False)
            for t in (act, w, rs_inter, rs_out):
                ttnn.deallocate(t)

            # ---------------- column-parallel boundary: gather -> attn_in
            shard_host = torch.randn(n, 1, m, DIM // n, generator=gen) * 0.1
            full_host = torch.cat([shard_host[d, 0] for d in range(n)], dim=-1).unsqueeze(0).unsqueeze(0)
            w2_host = torch.randn(1, 1, DIM, ATTN_IN_N, generator=gen) * 0.05
            shard = upload(shard_host, shard_dim=0)
            full = upload(full_host)
            w2 = upload(w2_host)
            ref2 = (full_host[0, 0].double() @ w2_host[0, 0].double()).float()
            ag_out = empty((1, 1, m, DIM))
            ag_sems = sems(2)
            ag_cg = (grid.x, grid.y - 2)
            ag_pc = fused_program_config(ag_cg, m, DIM, ATTN_IN_N)
            ag_offset = ttnn.CoreCoord(0, ag_cg[1])

            def ag_then_mm():
                gathered = ttnn.all_gather(shard, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                out = ttnn.linear(gathered, w2, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(gathered)
                return out

            def fused_ag_mm():
                res = ttnn.experimental.all_gather_matmul_async(
                    shard,
                    w2,
                    persistent_output_buffer=ag_out,
                    dim=3,
                    multi_device_global_semaphore=ag_sems,
                    all_gather_core_grid_offset=ag_offset,
                    num_links=args.links,
                    memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Ring,
                    subdevice_id=None,
                    memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=ag_pc,
                    compute_kernel_config=ckc,
                )
                return ttnn.clone(res[-1], memory_config=ttnn.DRAM_MEMORY_CONFIG)

            def mm_replicated():
                return ttnn.linear(full, w2, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            attn_arms = {"mm_replicated": mm_replicated, "ag_then_mm": ag_then_mm}
            if args.fused_ag:
                attn_arms["fused_ag_mm"] = fused_ag_mm
            else:
                print(
                    f"FUSED attn_in {shape_name} fused_ag_mm - SKIPPED hung the mesh at this shape; "
                    f"see triage/, rerun with --fused-ag",
                    flush=True,
                )
            run(f"attn_in {shape_name}", attn_arms, ref2, per_device=True)
            for t in (shard, full, w2, ag_out):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Persistent / preallocated buffers for the repeated decode collective (OPT-009).

The multichip stage swept the collective *spelling* (``ttnn.all_reduce`` vs an explicit
reduce-scatter + all-gather vs the ``all_gather``-onto-a-new-axis + local sum it ships at the decode
tile) and the fabric and topology arguments. What it never measured is the thing OPT-009 asks for
directly: the shipped decode collective is ``ttnn.all_gather``, the deprecated non-persistent
spelling, which allocates its output every step. ``ttnn.experimental.all_gather_async`` takes a
``persistent_output_buffer`` plus explicit ``chunks_per_sync`` / ``num_workers_per_link`` /
``num_buffers_per_channel``, and none of those were tried.

Both operand dtypes are measured, because the layer runs both: the token mixer's collective carries
bfloat16 and the MoE's carries the routed-expert activation dtype, ``bfloat8_b``.

Every row is a **traced** measurement with ``REPS`` independent copies of the collective in one
trace, so the fixed per-replay dispatch is amortised the same way ``probe_ccl.py`` amortises it and
the two artifacts' numbers are comparable.

    python .../logs/probe_ccl_persistent.py --dtype bfloat16
    python .../logs/probe_ccl_persistent.py --dtype bfloat8_b
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC

#: The shapes the two per-layer collectives are actually handed. ``decode`` is the batch-1 tile that
#: ``CCL_MODE="auto"`` sends down the ``stack_sum`` path; ``decode_b32`` is the advertised batch
#: bound after ``CCL_COMPACT_ROWS`` folds it; ``prefill_2048`` is one prefill chunk.
SHAPES = [
    ("decode", (1, 1, 32, 2048)),
    ("rows64", (1, 1, 64, 2048)),
    ("decode_b32", (1, 1, 1024, 2048)),
    ("prefill_2048", (1, 1, 2048, 2048)),
]

REPS = 8


def timed_trace(mesh, build, iters=32, warmup=4, reps=REPS, free=True):
    out = build()
    ttnn.synchronize_device(mesh)
    if free:
        ttnn.deallocate(out)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    traced = [build() for _ in range(reps)]
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(warmup):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    start = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    elapsed = (time.time() - start) / iters / reps * 1e6
    ttnn.release_trace(mesh, trace_id)
    if free:
        for t in traced:
            ttnn.deallocate(t)
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--links", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bfloat8_b"])
    args = ap.parse_args()

    ttnn.set_fabric_config(MC.DEFAULT_FABRIC_CONFIG, router_config=MC.fabric_router_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=0)
    n = mesh.get_num_devices()
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    tag = "CCLPERS" if args.dtype == "bfloat16" else "CCLPERSBF8"
    print(f"# persistent-buffer CCL sweep, {n} devices, {MC.DEFAULT_FABRIC_CONFIG.name}, dtype={args.dtype}")
    print("# columns: shape arm chunks_per_sync workers_per_link buffers_per_channel us pcc")
    dtype = ttnn.bfloat16 if args.dtype == "bfloat16" else ttnn.bfloat8_b
    try:
        for name, shape in SHAPES:
            host = torch.randn(*shape, dtype=torch.float32) * 0.1
            per_device = torch.stack([host * (d + 1) for d in range(n)], dim=0)
            tt = ttnn.from_torch(
                per_device.reshape(n * shape[0], *shape[1:]).to(torch.bfloat16),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh, dim=0),
            )
            expect = per_device.sum(dim=0)

            def check(out):
                got = ttnn.to_torch(out, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, dim=0))
                a = expect.double().flatten()
                b = got[: shape[0]].double().flatten()
                a, b = a - a.mean(), b - b.mean()
                return float((a @ b) / (a.norm() * b.norm() + 1e-12))

            # The persistent gather target: [n, 1, rows, dim], allocated ONCE. This is the buffer the
            # shipped `ttnn.all_gather` allocates on every step.
            persistent = ttnn.from_torch(
                torch.zeros(n * shape[0], *shape[1:], dtype=torch.bfloat16),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
            )
            # TWO semaphores: `all_gather_async_device_operation.cpp:57` asserts
            # `semaphore.size() == 2` ("Default implementation requires 2 semaphores"). The first
            # spelling of this arm passed one and was refused; that is a call-shape error, not a
            # property of the op, and it is the same mistake rounds 0-3 of the multichip stage made
            # with `all_reduce_async`.
            sem = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
            barrier = ttnn.create_global_semaphore(mesh, crs, 0)

            def shipped():
                gathered = ttnn.all_gather(tt, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                out = ttnn.sum(gathered, dim=0, keepdim=True)
                ttnn.deallocate(gathered)
                return out

            def make_async(persist, cps, wpl, bpc, use_barrier=True):
                def build():
                    gathered = ttnn.experimental.all_gather_async(
                        tt,
                        persistent_output_buffer=persistent if persist else None,
                        dim=0,
                        multi_device_global_semaphore=sem,
                        num_links=args.links,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        topology=ttnn.Topology.Ring,
                        barrier_semaphore=barrier if use_barrier else None,
                        chunks_per_sync=cps,
                        num_workers_per_link=wpl,
                        num_buffers_per_channel=bpc,
                    )
                    out = ttnn.sum(gathered, dim=0, keepdim=True)
                    if not persist:
                        ttnn.deallocate(gathered)
                    return out

                return build

            arms = [("ag_stack_sum_shipped", shipped, None, None, None)]
            for persist in (False, True):
                label = "ag_async_persist" if persist else "ag_async"
                arms.append((label, make_async(persist, None, None, None), None, None, None))
            # Tuning knobs, on the persistent arm only: OPT-009 asks for the buffer plan and the
            # sync/worker/channel counts to be named, not just the op.
            for cps, wpl, bpc in ((2, None, None), (8, None, None), (None, 2, None), (None, None, 2), (8, 2, 2)):
                arms.append((f"ag_async_persist_tuned", make_async(True, cps, wpl, bpc), cps, wpl, bpc))

            for arm, build, cps, wpl, bpc in arms:
                persist = "persist" in arm
                try:
                    probe = build()
                    value = check(probe)
                    if not persist:
                        ttnn.deallocate(probe)
                except Exception as exc:  # noqa: BLE001 - a refused arm is a result
                    print(
                        f"{tag} {name} {arm} {cps} {wpl} {bpc} FAIL {" | ".join(str(exc).splitlines()[:3])[:400]}",
                        flush=True,
                    )
                    continue
                try:
                    us = timed_trace(mesh, build, iters=args.iters, free=True)
                    print(f"{tag} {name} {arm} {cps} {wpl} {bpc} {us:.2f} {value:.6f}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"{tag} {name} {arm} {cps} {wpl} {bpc} FAIL {" | ".join(str(exc).splitlines()[:3])[:400]}",
                        flush=True,
                    )
            ttnn.deallocate(tt)
            ttnn.deallocate(persistent)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

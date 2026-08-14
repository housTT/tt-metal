# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Collective-topology sweep for the two per-layer collectives of the multichip decoder.

The decoder's residual contract is *replicated*, so each collective is a full all-reduce of a
``[b, t, 2048]`` tensor. This probe measures every spelling that can produce that result on the
4-chip Blackhole ring, at the decode shape (one 32-row tile) and at the shipped 2048-token prefill
chunk, **inside a captured trace** as well as eagerly — a decode step replays from a trace, where the
eager launch floor (~8 us/op here) is not paid, so an eager-only ranking would be measuring the
wrong thing.

Arms:

``all_reduce_ring`` / ``all_reduce_linear``
    ``ttnn.all_reduce`` on the whole mesh. This is the shipped arm.
``rs_ag_ring`` / ``rs_ag_linear``
    the composite the decoder's ``CCL_MODE="rs_ag"`` builds: ``ttnn.reduce_scatter`` on the last dim
    then ``ttnn.all_gather`` back. Also the shape a **sharded-residual** contract would leave
    half-finished, so its reduce-scatter row alone is the lower bound on that alternative.
``rs_only``
    reduce-scatter with no gather: what a residual sharded on the hidden dim would pay per
    collective. The decoder cannot use it as-is (both RMSNorms and the expert-parallel MoE need the
    full 2048-wide activation) but it bounds that design.
``ag_stack_sum``
    ``ttnn.all_gather`` on a new leading axis followed by a local ``ttnn.sum`` — the spelling that
    moves 4x the bytes and reduces on-device.
``all_reduce_async``
    ``ttnn.experimental.all_reduce_async``. Recorded because it is the tuned experimental-tier op;
    on Blackhole it is expected to refuse a DRAM input outright
    (``all_reduce_async_device_operation.cpp``: "does not support blackhole dram").

    python .../doc/multichip_decoder/logs/probe_ccl.py
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn

SHAPES = [
    ("decode", (1, 1, 32, 2048)),
    ("rows64", (1, 1, 64, 2048)),
    ("rows96", (1, 1, 96, 2048)),
    ("rows128", (1, 1, 128, 2048)),
    ("rows256", (1, 1, 256, 2048)),
    ("rows512", (1, 1, 512, 2048)),
    ("decode_b32", (1, 1, 32 * 32, 2048)),
    ("prefill_2048", (1, 1, 2048, 2048)),
]


#: Copies of the collective captured inside one trace. A trace replay costs a fixed dispatch
#: regardless of what it holds (~17 us on this mesh), so a one-op trace measures that overhead as
#: much as it measures the op. Capturing REPS independent copies and dividing amortises it, which is
#: what makes these rows comparable to the per-op device times a whole-layer profile reports.
REPS = 8


def timed_trace(mesh, build, iters=32, warmup=4, reps=REPS):
    """Mean us per collective, from a trace holding ``reps`` independent copies of ``build``."""
    out = build()
    ttnn.synchronize_device(mesh)
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
    for t in traced:
        ttnn.deallocate(t)
    return elapsed


def timed_eager(mesh, build, iters=20, warmup=3):
    for _ in range(warmup):
        ttnn.deallocate(build())
    ttnn.synchronize_device(mesh)
    start = time.time()
    for _ in range(iters):
        ttnn.deallocate(build())
    ttnn.synchronize_device(mesh)
    return (time.time() - start) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--links", type=int, default=2)
    args = ap.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=0)
    n = mesh.get_num_devices()
    print(f"# CCL sweep, {n} devices, FABRIC_1D_RING, num_links={args.links}")
    print("# columns: shape arm mode us correct_pcc")
    try:
        for name, shape in SHAPES:
            host = torch.randn(*shape, dtype=torch.float32) * 0.1
            per_device = torch.stack([host * (d + 1) for d in range(n)], dim=0)
            tt = ttnn.from_torch(
                per_device.reshape(n * shape[0], *shape[1:]).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh, dim=0),
            )
            expect = per_device.sum(dim=0)

            def check(out, scattered=False):
                """PCC of the reduced result. ``scattered`` composes the last-dim shards first."""
                dim = 3 if scattered else 0
                got = ttnn.to_torch(out, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, dim=dim))
                a = expect.double().flatten()
                b = (got if scattered else got[: shape[0]]).double().flatten()
                a, b = a - a.mean(), b - b.mean()
                return float((a @ b) / (a.norm() * b.norm() + 1e-12))

            arms = {}
            for topo_name, topo in (("ring", ttnn.Topology.Ring), ("linear", ttnn.Topology.Linear)):
                arms[f"all_reduce_{topo_name}"] = lambda t=topo: ttnn.all_reduce(
                    tt, topology=t, num_links=args.links, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

                def rs_ag(t=topo):
                    part = ttnn.reduce_scatter(
                        tt, dim=3, topology=t, num_links=args.links, memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    out = ttnn.all_gather(part, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(part)
                    return out

                arms[f"rs_ag_{topo_name}"] = rs_ag
                arms[f"rs_only_{topo_name}"] = lambda t=topo: ttnn.reduce_scatter(
                    tt, dim=3, topology=t, num_links=args.links, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

            def ag_stack_sum():
                gathered = ttnn.all_gather(tt, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                out = ttnn.sum(gathered, dim=0, keepdim=True)
                ttnn.deallocate(gathered)
                return out

            arms["ag_stack_sum"] = ag_stack_sum

            def all_reduce_async():
                grid = mesh.compute_with_storage_grid_size()
                crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
                sems = {
                    "barrier": [ttnn.create_global_semaphore(mesh, crs, 0)],
                    "rs": [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(3)],
                    "ag": [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)],
                }
                return ttnn.experimental.all_reduce_async(
                    tt,
                    cluster_axis=None,
                    mesh_device=mesh,
                    barrier_semaphores=sems["barrier"],
                    rs_global_semaphores=sems["rs"],
                    ag_global_semaphores=sems["ag"],
                    math_op=ttnn.ReduceType.Sum,
                    topology=ttnn.Topology.Linear,
                    num_links=args.links,
                )

            arms["all_reduce_async"] = all_reduce_async

            for arm, build in arms.items():
                try:
                    probe = build()
                    value = check(probe, scattered=arm.startswith("rs_only"))
                    ttnn.deallocate(probe)
                except Exception as exc:  # noqa: BLE001 - a refused arm is a result
                    print(f"CCL {name} {arm} - FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
                    continue
                for mode, fn in (("eager", timed_eager), ("trace", timed_trace)):
                    try:
                        us = fn(mesh, build, iters=args.iters)
                        print(f"CCL {name} {arm} {mode} {us:.2f} {value:.6f}", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        print(f"CCL {name} {arm} {mode} FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
            ttnn.deallocate(tt)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

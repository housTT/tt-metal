# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer A/B over ``OptimizedDecoder.NORM_SHARD_CORES``.

The isolated ``NORM`` rows of ``probe_decode_micro.txt`` make **4 cores** the fastest, several microseconds
ahead of the shipped 8, and review round 6 found the shipped choice defended by a monotonicity claim the
artifact contradicted and by an A/B that varies a different knob. An isolated norm is not the decision
either way: each sharded norm also pays a ``to_memory_config`` in and a ``sharded_to_interleaved`` out.
This measures the layer, which is what the ship decision rests on.

What it finds is that the layer barely moves at all across 4/8/16/32, while the isolated rows predict a much
larger swing. Every arm lands inside the run-to-run band the layer harness itself shows (README §5.1), so this
artifact does **not** rank them — review round 11 caught this docstring, the layer's own comment and work_log
§4.13 all reading a ranking out of it and putting 8 first, which on ``linear_attention`` is the reverse of the
recorded numbers. The conversions are the obvious suspect for why the op ladder does not transfer, and they are
not measured here. What the file supports is one statement: the knob does not matter at the layer, so 8 ships
because it is the shard count the rest of the stage's norm evidence was taken at, not because it is fastest.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/ab_norm_shard_cores.py

Every row is ``NORMCORES cores=<n> run=<i> layer=<idx> (<kind>) decode(traced) ... wall/iter=<ms>``, two
runs per arm per layer kind so the spread is visible, all in one process on one device.
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import (
    POLICIES,
    OptimizedDecoder,
    num_blocks_for_context,
)

CTX = 8192
PREFILL_LEN = 130
LAYERS = {3: "full_attention", 0: "linear_attention"}


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def time_traced_decode(mesh, decoder, page_table, iters):
    """Warmed traced decode, ms per step — the same shape as bench.py's decode arm."""
    import time

    gen = torch.Generator().manual_seed(7)
    x = (torch.randn(1, 1, decoder.cfg.dim, generator=gen) * 0.5).to(torch.bfloat16)
    pos = torch.tensor([PREFILL_LEN], dtype=torch.int32)
    args = dict(
        current_pos=dev(mesh, pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        rot_idxs=dev(mesh, pos.reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        page_table=page_table,
    )
    xd = dev(mesh, x)
    for _ in range(2):
        ttnn.deallocate(decoder.decode_forward(xd, **args))
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = decoder.decode_forward(xd, **args)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    start = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    elapsed = (time.time() - start) / iters * 1e3
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(out)
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--cores", default="4,8,16,32")
    args = ap.parse_args()
    arms = [int(c) for c in args.cores.split(",")]

    cfg = R.load_text_config()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=90112 * 1024)
    baseline = OptimizedDecoder.NORM_SHARD_CORES
    print(f"# Whole-layer traced decode against NORM_SHARD_CORES. Shipped default: {baseline}.")
    print("# Two runs per arm per layer kind, one process, one device, real weights.")
    try:
        for layer_idx, kind in LAYERS.items():
            sd = R.load_layer_state_dict(layer_idx)
            # One throwaway arm first: the very first build+trace of a process pays first-touch costs
            # (weight upload, kernel compile) that land on whichever arm happens to run first and would
            # otherwise read as that arm being slow.
            for cores, runs in [(arms[0], 1)] + [(c, args.runs) for c in arms]:
                OptimizedDecoder.NORM_SHARD_CORES = cores
                warmup_arm = runs == 1 and cores == arms[0]
                for run in range(1, runs + 1):
                    decoder = OptimizedDecoder.from_state_dict(
                        sd,
                        hf_config=cfg,
                        layer_idx=layer_idx,
                        mesh_device=mesh,
                        max_context=CTX,
                        policy=POLICIES["optimized"],
                    )
                    blocks = num_blocks_for_context(CTX)
                    decoder.allocate_kv_cache(blocks)
                    decoder.allocate_state(1)
                    page_table = None
                    if decoder.is_full_attention:
                        page_table = dev(
                            mesh,
                            torch.arange(blocks, dtype=torch.int32).reshape(1, blocks),
                            dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                        )
                    gen = torch.Generator().manual_seed(11)
                    x = (torch.randn(1, PREFILL_LEN, cfg.hidden_size, generator=gen) * 0.5).to(torch.bfloat16)
                    ttnn.deallocate(decoder.prefill_forward(dev(mesh, x), page_table=page_table))
                    ms = time_traced_decode(mesh, decoder, page_table, args.iters)
                    if not warmup_arm:
                        print(
                            f"NORMCORES cores={cores} run={run} layer={layer_idx} ({kind}) "
                            f"decode(traced) iters={args.iters} wall/iter={ms:.3f} ms",
                            flush=True,
                        )
                    del decoder, page_table
            del sd
    finally:
        OptimizedDecoder.NORM_SHARD_CORES = baseline
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

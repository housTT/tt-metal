# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer A/B for the routed gate/up ``in0_block_w`` cap (work_log §4.15).

The sparse sweep says the two tuned points want opposite values: a 32-tile inner block wins a batch-1
decode step, the whole tiled ``K`` wins a 32-token prefill group, and both margins are several times the
measured spread. The shipped rule therefore keys the cap off the same active-expert bound that already
chooses the core count (``SPARSE_GATE_UP_IN0_BLOCK_W``). This measures that decision on the *layer*, both
arms back to back in one process on one device with the same weights:

* ``before`` — one cap for every call, the whole tiled ``K``, which is what shipped before review round 6.
* ``after``  — the phase-aware cap.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/ab_gate_up_in0_block_w.py

Every row is ``GATEUPIBW arm=<before|after> run=<i> layer=<idx> (<kind>) <phase> ...``. Prefill is measured
too, because the whole point of the change is that the two phases disagree — a decode win that cost prefill
would not be a win.
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as impl

CTX = 8192
PREFILL_LEN = 130
PREFILL_SEQ = 2048
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


def build(mesh, cfg, sd, layer_idx):
    decoder = impl.OptimizedDecoder.from_state_dict(
        sd,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh,
        max_context=CTX,
        policy=impl.POLICIES["optimized"],
    )
    blocks = impl.num_blocks_for_context(CTX)
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
    return decoder, page_table


def time_prefill(mesh, decoder, page_table, cfg, warmups=2):
    gen = torch.Generator().manual_seed(11)
    x = (torch.randn(1, PREFILL_SEQ, cfg.hidden_size, generator=gen) * 0.5).to(torch.bfloat16)
    xd = dev(mesh, x)
    for _ in range(warmups):
        ttnn.deallocate(decoder.prefill_forward(xd, page_table=page_table))
    ttnn.synchronize_device(mesh)
    start = time.time()
    ttnn.deallocate(decoder.prefill_forward(xd, page_table=page_table))
    ttnn.synchronize_device(mesh)
    return (time.time() - start) * 1e3


def time_traced_decode(mesh, decoder, page_table, iters):
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
    args = ap.parse_args()

    shipped = dict(impl.SPARSE_GATE_UP_IN0_BLOCK_W)
    #: ``before`` is the pre-round-6 rule: one cap for every call, the largest divisor of the tiled K up to
    #: 64. Written as "both bounds get the same cap", so the only thing changing between arms is the phase
    #: awareness itself.
    arms = {"before": {False: 64, True: 64}, "after": shipped}

    cfg = R.load_text_config()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=90112 * 1024)
    print("# Whole-layer A/B for the routed gate/up in0_block_w cap. One process, one device, real weights.")
    print(f"# before = one cap for every call (64); after = phase-aware {shipped}.")
    print("# The first arm of each layer kind also pays this process's first-touch cost, so it runs twice.")
    try:
        for layer_idx, kind in LAYERS.items():
            sd = R.load_layer_state_dict(layer_idx)
            first = True
            for arm, table in arms.items():
                for run in range(1, args.runs + 1):
                    impl.SPARSE_GATE_UP_IN0_BLOCK_W.clear()
                    impl.SPARSE_GATE_UP_IN0_BLOCK_W.update(table)
                    decoder, page_table = build(mesh, cfg, sd, layer_idx)
                    prefill_ms = time_prefill(mesh, decoder, page_table, cfg)
                    del decoder, page_table
                    decoder, page_table = build(mesh, cfg, sd, layer_idx)
                    gen = torch.Generator().manual_seed(11)
                    x = (torch.randn(1, PREFILL_LEN, cfg.hidden_size, generator=gen) * 0.5).to(torch.bfloat16)
                    ttnn.deallocate(decoder.prefill_forward(dev(mesh, x), page_table=page_table))
                    decode_ms = time_traced_decode(mesh, decoder, page_table, args.iters)
                    del decoder, page_table
                    if first:
                        # Discard: first build+trace of the process pays weight upload and kernel compile.
                        first = False
                        continue
                    print(
                        f"GATEUPIBW arm={arm} run={run} layer={layer_idx} ({kind}) "
                        f"decode(traced) iters={args.iters} wall/iter={decode_ms:.3f} ms "
                        f"prefill seq_len={PREFILL_SEQ} wall={prefill_ms:.2f} ms",
                        flush=True,
                    )
            del sd
    finally:
        impl.SPARSE_GATE_UP_IN0_BLOCK_W.clear()
        impl.SPARSE_GATE_UP_IN0_BLOCK_W.update(shipped)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer A/B for dropping the K cache-write input's kv-head tile pad.

Review round 27 removed the interleave-pad-reshard round trip for V but left K paying its pad. Review
round 28 pointed out that the shipped tree already contained the counter-example: since round 27, V
reaches the *same* `paged_fused_update_cache` call with its logical kv-head dimension unpadded, and the
whole suite passes. Both inputs cannot both require the pad.

The op takes its head count from the **cache**, not the input
(`paged_tiled_fused_update_cache_program_factory.cpp`), and its writer kernel advances one row per head
for exactly `num_heads` heads - so rows past the real kv heads are never read and do not need zeroing.
No validation in the device op constrains the input's head dimension; the shard rules are all on the
padded shape, which a tile-layout tensor already satisfies.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/ab_kv_pad_free_write.py

Rows are ``KVPADFREE arm=<name> run=<i> layer=<idx> (<kind>) decode(traced) wall/iter=<ms> ms``. Arms
alternate build-by-build and each arm's first build is discarded.
"""


from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as impl

CTX = 8192
LAYERS = {3: "full_attention"}
DECODE_ITERS = 32
PREFILL_LEN = 128


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


def decode_ms(mesh, decoder, page_table):
    """Traced warmed decode, the same protocol `bench.py` uses for the headline number."""
    gen = torch.Generator().manual_seed(43)
    # Prefill first, so decode runs against a populated cache and state exactly as `bench.py` does.
    prefill = (torch.randn(1, PREFILL_LEN, decoder.cfg.dim, generator=gen) * 0.5).to(torch.bfloat16)
    ttnn.deallocate(decoder.prefill_forward(dev(mesh, prefill), page_table=page_table))
    x = (torch.randn(1, 1, decoder.cfg.dim, generator=gen) * 0.5).to(torch.bfloat16)
    xd = dev(mesh, x)
    step_pos = torch.tensor([PREFILL_LEN], dtype=torch.int32)
    pos = dev(mesh, step_pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    # uint32, which `ttnn.embedding` requires for the rotary index lookup.
    rot = dev(mesh, step_pos.reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def step():
        return decoder.decode_forward(xd, current_pos=pos, rot_idxs=rot, page_table=page_table)

    ttnn.deallocate(step())
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = step()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    start = time.time()
    for _ in range(DECODE_ITERS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    elapsed = (time.time() - start) / DECODE_ITERS * 1e3
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(out)
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="builds per arm; the first is discarded")
    args = ap.parse_args()

    cfg = R.load_text_config()
    print("# Whole-layer A/B: the K cache-write input with and without its kv-head tile pad.")
    print(f"# context={CTX} decode_iters={DECODE_ITERS}; arms alternate build-by-build, first build per arm discarded.")
    print("# pre-round-28-k-pad pays one FillPad per decode step for rows the writer kernel never reads.")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    shipped = impl.DECODE_KV_PAD_FREE_WRITE
    try:
        for layer_idx, kind in LAYERS.items():
            sd = R.load_layer_state_dict(layer_idx)
            arms = {"shipped-no-k-pad": True, "pre-round-28-k-pad": False}
            assert shipped is True, "this A/B calls the pad-free write the shipped arm"
            seen = {arm: 0 for arm in arms}
            for run in range(1, args.runs + 2):
                for arm, enabled in arms.items():
                    impl.DECODE_KV_PAD_FREE_WRITE = enabled
                    decoder, page_table = build(mesh, cfg, sd, layer_idx)
                    ms = decode_ms(mesh, decoder, page_table)
                    del decoder, page_table
                    seen[arm] += 1
                    if seen[arm] == 1:
                        continue  # discard: the first build of an arm pays weight upload and compile
                    print(
                        f"KVPADFREE arm={arm} run={run} layer={layer_idx} ({kind}) decode(traced) "
                        f"wall/iter={ms:.3f} ms",
                        flush=True,
                    )
            impl.DECODE_KV_PAD_FREE_WRITE = shipped
    finally:
        impl.DECODE_KV_PAD_FREE_WRITE = shipped
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

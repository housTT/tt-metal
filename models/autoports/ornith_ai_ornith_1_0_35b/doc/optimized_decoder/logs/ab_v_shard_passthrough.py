# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer A/B for writing the decode V head-split output to the paged cache as produced.

`nlp_create_qkv_heads_decode` emits V as HEIGHT_SHARDED L1 with shard `[32, head_dim]` on the first
`batch` cores. `_kv_update_memory_configs` used to rebuild exactly that config - so the layer converted
V to DRAM interleaved, zero-padded its kv-head dim, and converted it back, once per decode step, to
arrive at the layout it already had. The work log called all three conversions "required by an op
contract" and the topology audit called the pad "irreducible at this layer". Review round 27 found that
`paged_fused_update_cache` accepts the head split's output directly: it reads the head count from the
*cache*, not the input, and requires only that its two inputs be sharded, ROW_MAJOR, non-width-sharded,
with matching shard width and a height its shard height divides.

Two things had to move for it. `nlp_create_qkv_heads_decode` is called with
`overlap_qk_coregrid=False`, which puts K on a range disjoint from Q and V; and the cache write's grids
are swapped so **V** takes the first `batch` cores - the range the head split emits it on - and K takes
the second. The fused update only requires the two inputs to be disjoint, so moving K is free while
moving V would cost the reshard this exists to remove.

Only `full_attention` runs this path.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/ab_v_shard_passthrough.py

Rows are ``VPASSTHRU arm=<name> run=<i> layer=<idx> (<kind>) decode(traced) wall/iter=<ms> ms``. Arms
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
    print("# Whole-layer A/B: decode V written to the paged cache as produced vs interleaved and rebuilt.")
    print(f"# context={CTX} decode_iters={DECODE_ITERS}; arms alternate build-by-build, first build per arm discarded.")
    print("# pre-round-27-rebuild pays sharded_to_interleaved + FillPad + InterleavedToSharded per step.")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    shipped = impl.DECODE_V_SHARD_PASSTHROUGH
    try:
        for layer_idx, kind in LAYERS.items():
            sd = R.load_layer_state_dict(layer_idx)
            arms = {"shipped-v-passthrough": True, "pre-round-27-rebuild": False}
            assert shipped is True, "this A/B calls passthrough the shipped arm"
            seen = {arm: 0 for arm in arms}
            for run in range(1, args.runs + 2):
                for arm, enabled in arms.items():
                    impl.DECODE_V_SHARD_PASSTHROUGH = enabled
                    decoder, page_table = build(mesh, cfg, sd, layer_idx)
                    ms = decode_ms(mesh, decoder, page_table)
                    del decoder, page_table
                    seen[arm] += 1
                    if seen[arm] == 1:
                        continue  # discard: the first build of an arm pays weight upload and compile
                    print(
                        f"VPASSTHRU arm={arm} run={run} layer={layer_idx} ({kind}) decode(traced) "
                        f"wall/iter={ms:.3f} ms",
                        flush=True,
                    )
            impl.DECODE_V_SHARD_PASSTHROUGH = shipped
    finally:
        impl.DECODE_V_SHARD_PASSTHROUGH = shipped
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

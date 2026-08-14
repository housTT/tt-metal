# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer A/B for the two sharded-`in0` in-projections' `in0_block_w`, under the family that ships.

Review round 25 made `attn_in` and `gdn_in` consume a width-sharded `in0`. Review round 26 then found
that README section 5.4 was still selecting and ranking those two rows in the DRAM-interleaved probe
family - so their `in0_block_w` had been tuned under a placement they no longer run, and the table
printed a time the shipped geometry does not have. Corrected, the table shows both rows behind a
candidate at `in0_block_w` 2, and `gdn_in`'s candidate is at the same core count it already uses.

An op-level gap is a hypothesis at this stage, not a result - section 4.14 is the row that taught that
- so both candidates are measured at the layer here. Arms flip `DECODE_MATMUL_GEOMETRY`, nothing else.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/ab_dense_in0_block_w.py

Rows are ``DENSEIBW arm=<name> run=<i> layer=<idx> (<kind>) decode(traced) wall/iter=<ms> ms``. Arms
alternate build-by-build and each arm's first build is discarded, the same protocol as
`ab_sharded_norm_in0.py`.
"""


from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as impl

CTX = 8192
LAYERS = {3: "full_attention", 0: "linear_attention"}
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
    print("# Whole-layer A/B: the sharded-in0 in-projections' in0_block_w, shipped vs the corrected table's best.")
    print(f"# context={CTX} decode_iters={DECODE_ITERS}; arms alternate build-by-build, first build per arm discarded.")
    print("# shipped-* is the cap tuned under the interleaved family; candidate-* is the corrected winner.")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    shipped = dict(impl.DECODE_MATMUL_GEOMETRY)
    # (role, before, after) per layer kind. `before` is the geometry this stage shipped up to review round
    # 25 - selected under the DRAM-interleaved probe family - and is written literally rather than read from
    # `DECODE_MATMUL_GEOMETRY`, because once the candidate is adopted the shipped dict *is* the candidate and
    # an arm derived from it would compare a geometry against itself. That is exactly what happened on this
    # A/B's first committed run.
    arms_by_layer = {
        3: ("attn_in", (96, 8), (32, 2)),
        0: ("gdn_in", (110, 8), (110, 2)),
    }
    try:
        for layer_idx, kind in LAYERS.items():
            sd = R.load_layer_state_dict(layer_idx)
            role, before, after = arms_by_layer[layer_idx]
            arms = {
                f"pre-round-26-{role}-{before[0]}c-ibw{before[1]}": before,
                f"shipped-{role}-{after[0]}c-ibw{after[1]}": after,
            }
            assert after == shipped[role], f"{role} ships {shipped[role]}, but this A/B calls {after} the shipped arm"
            seen = {arm: 0 for arm in arms}
            for run in range(1, args.runs + 2):
                for arm, geometry in arms.items():
                    impl.DECODE_MATMUL_GEOMETRY[role] = geometry
                    decoder, page_table = build(mesh, cfg, sd, layer_idx)
                    ms = decode_ms(mesh, decoder, page_table)
                    del decoder, page_table
                    seen[arm] += 1
                    if seen[arm] == 1:
                        continue  # discard: the first build of an arm pays weight upload and compile
                    print(
                        f"DENSEIBW arm={arm} run={run} layer={layer_idx} ({kind}) decode(traced) "
                        f"wall/iter={ms:.3f} ms",
                        flush=True,
                    )
            impl.DECODE_MATMUL_GEOMETRY.update(shipped)
    finally:
        impl.DECODE_MATMUL_GEOMETRY.update(shipped)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

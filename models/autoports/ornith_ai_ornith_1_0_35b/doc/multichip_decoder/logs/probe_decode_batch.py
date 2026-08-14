# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What the decode collective is actually handed at batch > 1, and what compacting it costs.

Review round 1 of this stage found that every performance number here was measured at batch 1, where
the token mixer's decode output ``[b, 1, dim]`` occupies exactly one tile row. At batch ``b`` it
occupies ``b`` tile rows — ``b * 32`` physical rows — of which only ``b`` carry data, because the
tile's row axis is the *sequence* axis and decode has one token per user. The layer's **second**
collective never had this problem: the MoE path already reshapes to ``[1, 1, tokens, dim]`` before
it. So in one batch-32 decode forward the same logical reduction runs once on 1024 physical rows and
once on 32.

``probe_ccl.txt`` had already priced both shapes (``decode`` = 32 rows, ``decode_b32`` = 1024 rows)
without the connection being made. This probe closes it at the layer:

    python .../doc/multichip_decoder/logs/probe_decode_batch.py

Two things are recorded per batch:

``SHAPE``
    the logical shape, padded shape and physical row count of the tensor handed to
    :meth:`MultichipDecoder._all_reduce` at each of its two call sites, captured by wrapping the
    method. This is the measurement, not an inference from the source.
``DECODEB``
    warmed traced decode, ``CCL_COMPACT_ROWS`` off and on, three builds per arm, both layer kinds.
    ``off`` is the pre-review behaviour.

Batch 32 is the advertised bound, 13 is the awkward non-power-of-two the suite already uses, and the
rest of the ladder exists to locate the crossover: the fold is not free, so at small batches the
reshape costs more than the padding it removes.
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import _physical_rows

CTX = 1024
PREFILL = 128
LAYERS = {0: "linear_attention", 3: "full_attention"}


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def build(mesh, cfg, sd, layer_idx, batch):
    decoder = MC.MultichipDecoder.from_state_dict(
        sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh, max_context=CTX
    )
    blocks = MC.num_blocks_for_context(CTX)
    decoder.allocate_kv_cache(blocks * batch)
    decoder.allocate_state(batch)
    page_table = None
    if decoder.is_full_attention:
        table = torch.arange(blocks * batch, dtype=torch.int32).reshape(batch, blocks)
        page_table = dev(mesh, table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return decoder, page_table


def decode_inputs(mesh, cfg, batch):
    x = dev(
        mesh,
        (torch.randn(batch, 1, cfg.hidden_size, generator=torch.Generator().manual_seed(82)) * 0.5).to(torch.bfloat16),
    )
    pos = torch.full((batch,), PREFILL, dtype=torch.int32)
    return (
        x,
        dev(mesh, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        dev(mesh, pos.reshape(1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
    )


def warm(mesh, decoder, page_table, cfg, batch):
    x = dev(
        mesh,
        (torch.randn(batch, PREFILL, cfg.hidden_size, generator=torch.Generator().manual_seed(81)) * 0.5).to(
            torch.bfloat16
        ),
    )
    ttnn.deallocate(decoder.prefill_forward(x, page_table=page_table))
    ttnn.deallocate(x)


def traced_decode_ms(mesh, decoder, page_table, cfg, batch, iters=32):
    x, pos_buf, rot_buf = decode_inputs(mesh, cfg, batch)

    def forward():
        return decoder.decode_forward(x, current_pos=pos_buf, rot_idxs=rot_buf, page_table=page_table)

    ttnn.deallocate(forward())
    ttnn.synchronize_device(mesh)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = forward()
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(4):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    start = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    elapsed = (time.time() - start) / iters * 1e3
    finite = bool(
        torch.isfinite(ttnn.to_torch(out, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, dim=0)).float()).all()
    )
    ttnn.release_trace(mesh, trace_id)
    return elapsed, finite


def record_shapes(mesh, decoder, page_table, cfg, batch, kind):
    """Wrap ``_all_reduce`` for one untraced decode and print what each call site hands it."""
    seen = []
    original = decoder._all_reduce

    def spy(tensor):
        seen.append((list(tensor.shape), list(tensor.padded_shape), _physical_rows(tensor.shape)))
        return original(tensor)

    decoder._all_reduce = spy
    x, pos_buf, rot_buf = decode_inputs(mesh, cfg, batch)
    ttnn.deallocate(decoder.decode_forward(x, current_pos=pos_buf, rot_idxs=rot_buf, page_table=page_table))
    decoder._all_reduce = original
    for site, (shape, padded, rows) in zip(("mixer", "moe"), seen):
        print(
            f"SHAPE {kind} batch={batch} site={site} shape={shape} padded={padded} "
            f"physical_rows={rows} useful_rows={batch} waste={rows / max(batch, 1):.1f}x",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--batches", default="1,2,4,8,13,16,32")
    ap.add_argument("--builds", type=int, default=3)
    args = ap.parse_args()

    cfg = R.load_text_config()
    ttnn.set_fabric_config(MC.DEFAULT_FABRIC_CONFIG)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MC.DEFAULT_MESH_SHAPE), l1_small_size=24576, trace_region_size=0)
    shipped = MC.CCL_COMPACT_ROWS
    print("# what the two per-layer collectives are handed at decode, and the cost of compacting it")
    print("# SHAPE rows are measured by wrapping MultichipDecoder._all_reduce, not read off the source")
    print("# columns: DECODEB layer kind batch arm build decode_ms finite")
    try:
        for layer_idx in [int(v) for v in args.layers.split(",")]:
            sd = R.load_layer_state_dict(layer_idx)
            kind = LAYERS[layer_idx]
            for batch in [int(v) for v in args.batches.split(",")]:
                MC.CCL_COMPACT_ROWS = False
                decoder, page_table = build(mesh, cfg, sd, layer_idx, batch)
                warm(mesh, decoder, page_table, cfg, batch)
                record_shapes(mesh, decoder, page_table, cfg, batch, kind)
                del decoder, page_table
                for arm in (False, True):
                    MC.CCL_COMPACT_ROWS = arm
                    for build_idx in range(args.builds):
                        decoder, page_table = build(mesh, cfg, sd, layer_idx, batch)
                        warm(mesh, decoder, page_table, cfg, batch)
                        ms, finite = traced_decode_ms(mesh, decoder, page_table, cfg, batch)
                        print(
                            f"DECODEB {layer_idx} {kind} {batch} {'on' if arm else 'off'} {build_idx} "
                            f"{ms:.3f} {finite}",
                            flush=True,
                        )
                        del decoder, page_table
            del sd
    finally:
        MC.CCL_COMPACT_ROWS = shipped
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

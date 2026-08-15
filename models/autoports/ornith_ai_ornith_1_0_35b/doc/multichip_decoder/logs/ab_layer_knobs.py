# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer A/B for the five knobs this stage introduced, on the target 4-chip mesh.

The isolated probes (``probe_ccl.txt``, ``probe_dense_matmul.txt``) rank candidates op by op. This
script measures what the *layer* does with each choice — warmed traced decode and warmed 2048-token
prefill — because an op-level win can be absorbed by the rest of the graph and an op-level tie can
still move the layer.

Arms:

``ccl``
    :data:`~...multichip_decoder.CCL_MODE` over ``auto`` (shipped), ``all_reduce``, ``rs_ag`` and
    ``stack_sum``. ``auto`` is ``stack_sum`` at the batch-1 decode tile and ``all_reduce`` for
    prefill, so the decode column should match ``stack_sum`` and the prefill column ``all_reduce``.
``geometry``
    the retuned ``MULTICHIP_DECODE_MATMUL_GEOMETRY`` against the single-chip
    ``DECODE_MATMUL_GEOMETRY`` it replaces, i.e. what the inherited program configs cost when
    applied unchanged to the 4x narrower per-device projections.
``sparse``
    :data:`~...multichip_decoder.SPARSE_SCALE_CORES_BY_TP` over ``on`` (shipped) and ``off`` (the
    inherited single-chip core rule). Only the prefill column can move: the rescaling changes the
    routed matmuls' realised core count from 16/8 to 32/32 at a 32-token prefill group's ~63 local
    active experts, and leaves batch-1 decode at 8 cores for both roles.
``cast``
    :data:`~...multichip_decoder.CCL_CAST_BLOCKFLOAT` over ``on`` (shipped) and ``off``. The MoE half
    produces ``bfloat8_b``, so without the cast the second per-layer collective runs on a block-float
    operand — 1492 us against the first collective's 100 us at the same logical shape in the prefill
    profile.
``routing``
    :data:`~...multichip_decoder.ROUTING_SELECT_MODE` over ``select_matmul`` (shipped) and
    ``gather``: the two spellings of "narrow the replicated 256-wide dense routing vector to this
    device's 64-wide block". Both are exact — ``test_routing_select_modes_agree`` asserts they
    produce identical layer output — so this arm is purely the cost of the narrowing op inside the
    layer.

Every arm is built fresh in the same process on the same device with the same weights, and each is
measured three times so the spread is visible next to the difference.

    python .../doc/multichip_decoder/logs/ab_layer_knobs.py
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import DECODE_MATMUL_GEOMETRY

CTX = 8192
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


def build(mesh, cfg, sd, layer_idx):
    decoder = MC.MultichipDecoder.from_state_dict(
        sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh, max_context=CTX
    )
    blocks = MC.num_blocks_for_context(CTX)
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


def traced_decode_ms(mesh, decoder, page_table, cfg, iters=32):
    ttnn.deallocate(
        decoder.prefill_forward(
            dev(
                mesh,
                (torch.randn(1, 128, cfg.hidden_size, generator=torch.Generator().manual_seed(81)) * 0.5).to(
                    torch.bfloat16
                ),
            ),
            page_table=page_table,
        )
    )
    x = dev(
        mesh, (torch.randn(1, 1, cfg.hidden_size, generator=torch.Generator().manual_seed(82)) * 0.5).to(torch.bfloat16)
    )
    pos = torch.tensor([128], dtype=torch.int32)
    pos_buf = dev(mesh, pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    rot_buf = dev(mesh, pos.reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

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


def prefill_ms(mesh, decoder, page_table, cfg, seq_len=2048):
    x = dev(
        mesh,
        (torch.randn(1, seq_len, cfg.hidden_size, generator=torch.Generator().manual_seed(71)) * 0.5).to(
            torch.bfloat16
        ),
    )
    for _ in range(2):
        ttnn.deallocate(decoder.prefill_forward(x, page_table=page_table))
    ttnn.synchronize_device(mesh)
    start = time.time()
    out = decoder.prefill_forward(x, page_table=page_table)
    ttnn.synchronize_device(mesh)
    elapsed = (time.time() - start) * 1e3
    ttnn.deallocate(out)
    ttnn.deallocate(x)
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--builds", type=int, default=3)
    ap.add_argument("--knobs", default="ccl,geometry,routing,sparse,cast")
    args = ap.parse_args()

    cfg = R.load_text_config()
    ttnn.set_fabric_config(MC.DEFAULT_FABRIC_CONFIG, router_config=MC.fabric_router_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MC.DEFAULT_MESH_SHAPE), l1_small_size=24576, trace_region_size=0)
    shipped_geometry = dict(MC._MultichipProjectionConfigs.GEOMETRY)
    shipped_routing = MC.ROUTING_SELECT_MODE
    shipped_sparse = MC.SPARSE_SCALE_CORES_BY_TP
    shipped_cast = MC.CCL_CAST_BLOCKFLOAT
    print("# whole-layer A/B on the 4-chip mesh; each arm built fresh, N builds per arm")
    print("# columns: knob arm layer kind build decode_ms prefill_ms finite")
    try:
        for layer_idx in [int(v) for v in args.layers.split(",")]:
            sd = R.load_layer_state_dict(layer_idx)
            arms = []
            if "ccl" in args.knobs:
                arms += [("ccl", mode) for mode in ("auto", "all_reduce", "rs_ag", "stack_sum")]
            if "geometry" in args.knobs:
                arms += [("geometry", "multichip-retuned"), ("geometry", "single-chip-inherited")]
            if "routing" in args.knobs:
                arms += [("routing", mode) for mode in ("select_matmul", "gather")]
            if "sparse" in args.knobs:
                arms += [("sparse", "tp-rescaled"), ("sparse", "single-chip-inherited")]
            if "cast" in args.knobs:
                arms += [("cast", "bf16"), ("cast", "block-float")]
            for knob, arm in arms:
                MC.CCL_MODE = "auto"
                MC.ROUTING_SELECT_MODE = shipped_routing
                MC.SPARSE_SCALE_CORES_BY_TP = shipped_sparse
                MC.CCL_CAST_BLOCKFLOAT = shipped_cast
                MC._MultichipProjectionConfigs.GEOMETRY = shipped_geometry
                if knob == "ccl":
                    MC.CCL_MODE = arm
                elif knob == "routing":
                    MC.ROUTING_SELECT_MODE = arm
                elif knob == "sparse":
                    MC.SPARSE_SCALE_CORES_BY_TP = arm == "tp-rescaled"
                elif knob == "cast":
                    MC.CCL_CAST_BLOCKFLOAT = arm == "bf16"
                elif arm == "single-chip-inherited":
                    MC._MultichipProjectionConfigs.GEOMETRY = dict(DECODE_MATMUL_GEOMETRY)
                for build_idx in range(args.builds):
                    decoder, page_table = build(mesh, cfg, sd, layer_idx)
                    decode_ms, finite = traced_decode_ms(mesh, decoder, page_table, cfg)
                    pf = prefill_ms(mesh, decoder, page_table, cfg)
                    print(
                        f"ABLAYER {knob} {arm} {layer_idx} {LAYERS[layer_idx]} {build_idx} "
                        f"{decode_ms:.3f} {pf:.2f} {finite}",
                        flush=True,
                    )
                    del decoder, page_table
            del sd
    finally:
        MC.CCL_MODE = "auto"
        MC.ROUTING_SELECT_MODE = shipped_routing
        MC.SPARSE_SCALE_CORES_BY_TP = shipped_sparse
        MC.CCL_CAST_BLOCKFLOAT = shipped_cast
        MC._MultichipProjectionConfigs.GEOMETRY = shipped_geometry
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

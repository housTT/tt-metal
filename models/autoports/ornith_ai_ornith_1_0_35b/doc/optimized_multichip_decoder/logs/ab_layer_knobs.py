# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer A/B for this stage's knobs, on the target 4-chip mesh.

Two jobs. The first is the new knob: :data:`~...multichip_decoder.ROUTER_MODE`, the fused
``generalized_moe_gate`` against the inherited ``topk`` + ``softmax`` + ``scatter`` chain. The second
is **re-verification**: every knob the multichip stage settled was settled against a different
default, and a router change that removes ~9 % of the decode window can move a crossover. An
optimized stage that only measured its own new knob would be reporting a stale table, so ``ccl``,
``geometry``, ``sparse`` and ``cast`` are all re-run here under the new default.

``policy`` is the OPT-007 item: the BFP4 dense-projection policy the single-chip stage measured,
found faster, and rejected on a compounding judgement, re-measured in the multichip topology — which
is what OPT-007 asks for, since the per-device projections are four times narrower here and the
op-count balance is different.

Arms:

``router``      ``fused_gate`` (shipped) vs ``fused_gate_local`` vs ``topk`` (the chain every earlier
                stage shipped). ``fused_gate_local`` feeds the gate op a DEVICE-LOCAL index tensor so
                the scatter lands straight in this device's 64-wide expert block, which removes the
                ``expert_select`` one-hot narrowing matmul and shrinks the scatter base from 256 to 65.
``policy``      the shipped ``optimized`` precision policy vs ``bfp4-projections``.
``collective``  the three decode collective spellings: ``ttnn.all_reduce`` (**shipped**), the
                barrier-semaphore ``all_gather_async``, and the deprecated ``ttnn.all_gather`` the
                multichip stage shipped. This one is a **correctness** decision priced here rather
                than a latency decision: the deprecated op diverges across devices under sustained
                traced replay and the other two do not
                (`logs/probe_replay_divergence*.txt`, tabulated in README section 4.1), so the choice
                is between the two clean arms and this table is what picks between them.
``geometry``    the re-swept ``MULTICHIP_DECODE_MATMUL_GEOMETRY`` vs the single-chip table.
``sparse``      the tp-rescaled routed-sparse core rule vs the inherited one.
``cast``        ``CCL_CAST_BLOCKFLOAT`` off (shipped) vs on.
``state_fidelity``
                ``PrecisionPolicy.state_fidelity`` over HiFi4 (inherited) and HiFi2, for the three
                float32 DeltaNet recurrent-state matmuls of the ``linear_attention`` decode step.
                ``tt-perf-report`` advises HiFi2 on those rows 64x per capture; the advisory's own
                sentence names BFP8 on an ``FP32 x FP32`` row, so it mis-fires, but the underlying
                candidate is real and ~5 us x 2 per step, which is why it is measured rather than
                handed off.
``residual``    ``DECODE_RESIDUAL_MEMORY``: the two decode residual adds in L1 vs the inherited
                DRAM-interleaved default (**shipped**, because the two are a tie). The largest untried lever review round 1 named:
                the decode window is 20-23 % layout and 11-13 % `BinaryNg`, on tensors that are one
                32-row tile.

Every arm is built fresh in the same process on the same device with the same real weights, and each
is measured ``--builds`` times so the spread is visible next to the difference.

    python .../doc/optimized_multichip_decoder/logs/ab_layer_knobs.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import DECODE_MATMUL_GEOMETRY, POLICIES

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


def build(mesh, cfg, sd, layer_idx, policy="optimized"):
    """``policy`` may be a name or a ``PrecisionPolicy``; the ``state_fidelity`` arm passes an object."""
    decoder = MC.MultichipDecoder.from_state_dict(
        sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh, max_context=CTX, policy=policy
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
    ap.add_argument(
        "--knobs",
        default="router,collective,router_fidelity,state_fidelity,policy,geometry,sparse,cast,residual",
    )
    args = ap.parse_args()

    cfg = R.load_text_config()
    ttnn.set_fabric_config(MC.DEFAULT_FABRIC_CONFIG, router_config=MC.fabric_router_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MC.DEFAULT_MESH_SHAPE), l1_small_size=24576, trace_region_size=0)
    shipped = {
        "geometry": dict(MC._MultichipProjectionConfigs.GEOMETRY),
        "routing": MC.ROUTING_SELECT_MODE,
        "sparse": MC.SPARSE_SCALE_CORES_BY_TP,
        "cast": MC.CCL_CAST_BLOCKFLOAT,
        "router": MC.ROUTER_MODE,
        "residual": MC.DECODE_RESIDUAL_MEMORY,
        "collective": MC.AUTO_STACK_SUM_MODE,
        "ccl_mode": MC.CCL_MODE,
        "router_fidelity": MC.ROUTER_DECODE_FIDELITY,
    }
    print("# whole-layer A/B on the 4-chip mesh; each arm built fresh, N builds per arm")
    print("# columns: knob arm layer kind build decode_ms prefill_ms finite")
    try:
        for layer_idx in [int(v) for v in args.layers.split(",")]:
            sd = R.load_layer_state_dict(layer_idx)
            arms = []
            if "router" in args.knobs:
                arms += [("router", "fused_gate"), ("router", "fused_gate_local"), ("router", "topk")]
            if "collective" in args.knobs:
                arms += [
                    ("collective", "all_reduce"),
                    ("collective", "stack_sum_async"),
                    ("collective", "stack_sum-deprecated"),
                ]
            if "policy" in args.knobs:
                arms += [("policy", "optimized"), ("policy", "bfp4-projections")]
            if "ccl" in args.knobs:
                arms += [("ccl", "auto"), ("ccl", "all_reduce")]
            if "geometry" in args.knobs:
                arms += [("geometry", "multichip-retuned"), ("geometry", "single-chip-inherited")]
            if "sparse" in args.knobs:
                arms += [("sparse", "tp-rescaled"), ("sparse", "single-chip-inherited")]
            if "cast" in args.knobs:
                arms += [("cast", "block-float"), ("cast", "bf16")]
            if "router_fidelity" in args.knobs:
                arms += [("router_fidelity", "hifi4-inherited"), ("router_fidelity", "hifi2")]
            if "state_fidelity" in args.knobs and layer_idx != 3:
                # linear_attention only: the DeltaNet recurrent-state matmuls do not exist on a
                # full_attention layer.
                arms += [("state_fidelity", "hifi4-inherited"), ("state_fidelity", "hifi2")]
            if "residual" in args.knobs:
                arms += [("residual", "l1"), ("residual", "dram-interleaved-inherited")]
            for knob, arm in arms:
                MC.CCL_MODE = shipped["ccl_mode"]
                MC.ROUTER_DECODE_FIDELITY = shipped["router_fidelity"]
                MC.ROUTING_SELECT_MODE = shipped["routing"]
                MC.SPARSE_SCALE_CORES_BY_TP = shipped["sparse"]
                MC.CCL_CAST_BLOCKFLOAT = shipped["cast"]
                MC.ROUTER_MODE = shipped["router"]
                MC.DECODE_RESIDUAL_MEMORY = shipped["residual"]
                MC.AUTO_STACK_SUM_MODE = shipped["collective"]
                MC._MultichipProjectionConfigs.GEOMETRY = shipped["geometry"]
                policy = "optimized"
                if knob == "ccl":
                    MC.CCL_MODE = arm
                elif knob == "router":
                    MC.ROUTER_MODE = arm
                elif knob == "policy":
                    policy = arm
                elif knob == "state_fidelity":
                    policy = (
                        replace(POLICIES["optimized"], state_fidelity=ttnn.MathFidelity.HiFi2)
                        if arm == "hifi2"
                        else "optimized"
                    )
                elif knob == "sparse":
                    MC.SPARSE_SCALE_CORES_BY_TP = arm == "tp-rescaled"
                elif knob == "cast":
                    MC.CCL_CAST_BLOCKFLOAT = arm == "bf16"
                elif knob == "collective":
                    # `all_reduce` is CCL_MODE; the two stack-sum arms are reached by putting CCL_MODE
                    # back to `auto` (which is stack_sum below the crossover) and varying which
                    # spelling `auto` resolves to.
                    if arm == "all_reduce":
                        MC.CCL_MODE = "all_reduce"
                    else:
                        MC.CCL_MODE = "auto"
                        MC.AUTO_STACK_SUM_MODE = "stack_sum" if arm == "stack_sum-deprecated" else arm
                elif knob == "router_fidelity":
                    MC.ROUTER_DECODE_FIDELITY = ttnn.MathFidelity.HiFi2 if arm == "hifi2" else None
                elif knob == "residual":
                    MC.DECODE_RESIDUAL_MEMORY = ttnn.L1_MEMORY_CONFIG if arm == "l1" else None
                elif arm == "single-chip-inherited":
                    MC._MultichipProjectionConfigs.GEOMETRY = dict(DECODE_MATMUL_GEOMETRY)
                for build_idx in range(args.builds):
                    decoder, page_table = build(mesh, cfg, sd, layer_idx, policy=policy)
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
        MC.CCL_MODE = shipped["ccl_mode"]
        MC.ROUTER_DECODE_FIDELITY = shipped["router_fidelity"]
        MC.ROUTING_SELECT_MODE = shipped["routing"]
        MC.SPARSE_SCALE_CORES_BY_TP = shipped["sparse"]
        MC.CCL_CAST_BLOCKFLOAT = shipped["cast"]
        MC.ROUTER_MODE = shipped["router"]
        MC.DECODE_RESIDUAL_MEMORY = shipped["residual"]
        MC.AUTO_STACK_SUM_MODE = shipped["collective"]
        MC._MultichipProjectionConfigs.GEOMETRY = shipped["geometry"]
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

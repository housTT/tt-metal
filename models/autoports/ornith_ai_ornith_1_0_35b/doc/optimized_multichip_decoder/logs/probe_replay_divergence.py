# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Cross-device bitwise equality of a traced decode output, under sustained replay.

Written to chase one observed failure. In the first full-suite run of the optimized default,
``test_traced_replay_does_not_leak[full_attention]`` failed its cross-device equality assertion after
128 traced replays — device 2's output differed from device 0's in the tail of the hidden dimension —
and then passed twice in isolation. An intermittent cross-device divergence in a traced decode is a
correctness question, not a flake to re-run until green, so this probe reproduces the exact shape of
that check at higher pressure: many rounds, many replays per round, both layer kinds, and both router
modes, reporting the first differing element rather than only a boolean.

    python .../logs/probe_replay_divergence.py --rounds 20 --replays 128
    python .../logs/probe_replay_divergence.py --rounds 20 --replays 128 --router-mode topk
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC

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


def shards(mesh, tensor):
    whole = ttnn.to_torch(tensor, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, dim=0))
    rows = int(tensor.shape[0])
    return [whole[i * rows : (i + 1) * rows] for i in range(mesh.get_num_devices())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--replays", type=int, default=128)
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--router-mode", default="")
    ap.add_argument(
        "--ccl-mode",
        default="",
        help="override multichip_decoder.CCL_MODE ('auto' | 'all_reduce' | 'rs_ag' | 'stack_sum' | "
        "'stack_sum_async'). 'all_reduce' takes the stable op at every shape, which is the arm the "
        "collective decision has to be re-taken against.",
    )
    ap.add_argument(
        "--auto-stack-sum",
        default="",
        help="override multichip_decoder.AUTO_STACK_SUM_MODE: 'stack_sum' is the deprecated "
        "ttnn.all_gather the multichip stage shipped, 'stack_sum_async' is the barrier-semaphore "
        "all_gather_async this stage ships. Empty means the shipped default.",
    )
    ap.add_argument("--blocking", action="store_true", help="replay with blocking=True instead of False")
    ap.add_argument(
        "--sync-every-replay",
        action="store_true",
        help="synchronize after EVERY replay, which is what test_traced_replay_does_not_leak does. "
        "A back-to-back non-blocking burst and a synchronize-per-replay loop are different "
        "synchronization pressure on the fabric, and the observed failure came from the second.",
    )
    args = ap.parse_args()

    if args.router_mode:
        MC.ROUTER_MODE = args.router_mode
    if args.auto_stack_sum:
        MC.AUTO_STACK_SUM_MODE = args.auto_stack_sum
    if args.ccl_mode:
        MC.CCL_MODE = args.ccl_mode
    cfg = R.load_text_config()
    ttnn.set_fabric_config(MC.DEFAULT_FABRIC_CONFIG, router_config=MC.fabric_router_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MC.DEFAULT_MESH_SHAPE), l1_small_size=24576, trace_region_size=0)
    print(
        f"# traced-replay cross-device divergence probe, router_mode={MC.ROUTER_MODE}, "
        f"ccl_mode={MC.CCL_MODE}, auto_stack_sum={MC.AUTO_STACK_SUM_MODE}, "
        f"rounds={args.rounds} replays={args.replays} blocking={args.blocking} "
        f"sync_every_replay={args.sync_every_replay}"
    )
    print("# columns: layer kind replay_pattern round max_abs_diff_vs_device0 first_bad_device n_differing_devices")
    try:
        for layer_idx in [int(v) for v in args.layers.split(",")]:
            sd = R.load_layer_state_dict(layer_idx)
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
            ttnn.deallocate(
                decoder.prefill_forward(
                    dev(
                        mesh,
                        (torch.randn(1, 128, cfg.hidden_size, generator=torch.Generator().manual_seed(44)) * 0.5).to(
                            torch.bfloat16
                        ),
                    ),
                    page_table=page_table,
                )
            )
            x = dev(
                mesh,
                (torch.randn(1, 1, cfg.hidden_size, generator=torch.Generator().manual_seed(45)) * 0.5).to(
                    torch.bfloat16
                ),
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
            worst = 0.0
            for rnd in range(args.rounds):
                for _ in range(args.replays):
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=args.blocking)
                    if args.sync_every_replay:
                        ttnn.synchronize_device(mesh)
                ttnn.synchronize_device(mesh)
                parts = [p.float() for p in shards(mesh, out)]
                diffs = [float((parts[0] - parts[d]).abs().max()) for d in range(1, len(parts))]
                bad = [d + 1 for d, v in enumerate(diffs) if v != 0.0]
                worst = max(worst, max(diffs) if diffs else 0.0)
                print(
                    f"REPLAYDIV {layer_idx} {LAYERS[layer_idx]} {'sync' if args.sync_every_replay else 'burst'} "
                    f"{rnd} {max(diffs) if diffs else 0.0:.6e} {bad[0] if bad else '-'} {len(bad)}",
                    flush=True,
                )
            print(
                f"REPLAYDIVMAX {layer_idx} {LAYERS[layer_idx]} "
                f"{'sync' if args.sync_every_replay else 'burst'} {worst:.6e}",
                flush=True,
            )
            ttnn.release_trace(mesh, trace_id)
            del decoder, page_table, sd
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

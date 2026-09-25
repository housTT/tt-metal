# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where does the drafter's 1.09 s per call go?

The end-to-end run showed a 2.56 B drafter taking ~12x longer than the 30 B
target it exists to accelerate, so the speculation win (4.0x fewer target
forwards) turned into a 12x wall-clock loss.  Before optimising anything, find
out what is actually slow.

Deliberately standalone: no target model, so an iteration costs seconds instead
of the ~3 min a full generator build takes.  A weight-bandwidth floor is computed
alongside the measurement, because the interesting question is not "how long does
it take" but "how far off the achievable floor is it".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference_dflash as R
from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_drafter import (
    DFlashDrafter,
    DFlashDrafterCache,
    bidirectional_sliding_mask,
    config_from_hf,
    rope_tables,
)

#: Blackhole DRAM read bandwidth per die, order of magnitude, for the floor estimate.
DRAM_GBPS_PER_DIE = 200.0


def _sync(mesh) -> None:
    for device in mesh.get_devices():
        ttnn.synchronize_device(device)


def time_stage(fn, mesh, *, warmup: int = 1, iters: int = 3) -> float:
    for _ in range(warmup):
        out = fn()
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
    _sync(mesh)
    started = time.perf_counter()
    for _ in range(iters):
        out = fn()
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
    _sync(mesh)
    return (time.perf_counter() - started) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-len", type=int, default=96)
    parser.add_argument("--mesh", default="1x4", choices=["1x1", "1x4"])
    parser.add_argument("--weight-dtype", default="bfloat8_b", choices=["bfloat8_b", "bfloat16"])
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    rows, cols = (1, 1) if args.mesh == "1x1" else (1, 4)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols), trace_region_size=0)
    try:
        config = config_from_hf(R.draft_config())
        block = config.block_size
        print(f"mesh={args.mesh} weights={args.weight_dtype} context_len={args.context_len} block={block}")

        built = time.perf_counter()
        drafter = DFlashDrafter.from_state_dict(
            R.draft_state_dict(),
            hf_config=R.draft_config(),
            mesh_device=mesh,
            weight_dtype=getattr(ttnn, args.weight_dtype),
            activation_dtype=ttnn.bfloat16,
        )
        print(f"drafter built in {time.perf_counter() - built:.1f}s")

        def to_dev(tensor: torch.Tensor) -> ttnn.Tensor:
            return ttnn.from_torch(
                tensor.to(torch.bfloat16),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        ctx_len = args.context_len
        noise_host = torch.randn(1, 1, block, config.hidden_size) * 0.02
        ctx_host = torch.randn(1, 1, ctx_len, config.context_fan_in)
        ctx_positions = torch.arange(ctx_len)
        noise_positions = torch.arange(ctx_len, ctx_len + block)

        results: dict[str, float] = {}

        # --- whole uncached forward (what PCC validates) --------------------------
        def full_forward():
            noise = to_dev(noise_host)
            ctx = to_dev(ctx_host)
            out = drafter(noise, ctx, position_ids=torch.arange(ctx_len + block))
            ttnn.deallocate(ctx)
            return out

        results["uncached_forward_s"] = time_stage(full_forward, mesh, iters=args.iters)

        # --- cached forward (what the runner uses) --------------------------------
        def cached_forward():
            cache = DFlashDrafterCache(config.num_hidden_layers)
            noise = to_dev(noise_host)
            ctx = to_dev(ctx_host)
            out = drafter.forward_cached(
                noise,
                ctx,
                context_positions=ctx_positions,
                noise_positions=noise_positions,
                cache=cache,
            )
            ttnn.deallocate(ctx)
            host = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            del host
            cache.release()
            return out

        results["cached_forward_s"] = time_stage(cached_forward, mesh, iters=args.iters)

        # --- component costs ------------------------------------------------------
        ctx_dev = to_dev(ctx_host)
        results["project_context_s"] = time_stage(lambda: drafter.project_context(ctx_dev), mesh, iters=args.iters)

        results["upload_context_s"] = time_stage(lambda: to_dev(ctx_host), mesh, iters=args.iters)

        rope_host = rope_tables(noise_positions, config.head_dim, config.rope_theta)[0]
        results["upload_rope_s"] = time_stage(
            lambda: to_dev(rope_host.reshape(1, 1, block, config.head_dim)), mesh, iters=args.iters
        )

        mask_host = bidirectional_sliding_mask(
            noise_positions, torch.arange(ctx_len + block), config.sliding_window, torch.bfloat16
        )
        results["build_and_upload_mask_s"] = time_stage(lambda: to_dev(mask_host), mesh, iters=args.iters)

        # A single layer, to see how the 5-layer total decomposes.
        layer = drafter.layers[0]
        ctx_proj = drafter.project_context(ctx_dev)
        cos_c, sin_c = rope_tables(ctx_positions, config.head_dim, config.rope_theta)
        cos_w, sin_w = rope_tables(noise_positions, config.head_dim, config.rope_theta)
        rope_ctx = (
            to_dev(cos_c.reshape(1, 1, ctx_len, config.head_dim)),
            to_dev(sin_c.reshape(1, 1, ctx_len, config.head_dim)),
        )
        rope_win = (
            to_dev(cos_w.reshape(1, 1, block, config.head_dim)),
            to_dev(sin_w.reshape(1, 1, block, config.head_dim)),
        )
        mask_dev = to_dev(mask_host)

        def one_layer():
            cache = DFlashDrafterCache(config.num_hidden_layers)
            hidden = to_dev(noise_host)
            out = layer.forward_cached(
                hidden,
                context=ctx_proj,
                context_len=ctx_len,
                rope_ctx=rope_ctx,
                rope_win=rope_win,
                mask=mask_dev,
                cache=cache,
                layer_idx=0,
            )
            cache.release()
            return out

        results["one_layer_s"] = time_stage(one_layer, mesh, iters=args.iters)

        # --- weight-bandwidth floor ----------------------------------------------
        bytes_per_param = 1.0 if args.weight_dtype == "bfloat8_b" else 2.0
        params = 2_555_988_304  # 5,111,976,608 bytes at bf16
        replicated_bytes = params * bytes_per_param  # every die reads the whole thing
        sharded_bytes = replicated_bytes / (rows * cols)
        floor_replicated = replicated_bytes / (DRAM_GBPS_PER_DIE * 1e9)
        floor_sharded = sharded_bytes / (DRAM_GBPS_PER_DIE * 1e9)

        print("\n" + "=" * 70)
        for name, seconds in results.items():
            print(f"{name:28s} {seconds * 1000:9.2f} ms")
        print("-" * 70)
        print(f"{'weight floor, replicated':28s} {floor_replicated * 1000:9.2f} ms   (what we do now)")
        print(f"{'weight floor, sharded 1x4':28s} {floor_sharded * 1000:9.2f} ms   (what sharding would give)")
        measured = results["cached_forward_s"]
        print(f"{'measured / replicated floor':28s} {measured / floor_replicated:9.1f}x")
        print(f"{'5 x one_layer':28s} {results['one_layer_s'] * 5 * 1000:9.2f} ms")
        print(
            f"{'unexplained by layers':28s} "
            f"{(measured - results['one_layer_s'] * 5) * 1000:9.2f} ms"
        )
        print("=" * 70)

        payload = {
            "mesh": args.mesh,
            "weight_dtype": args.weight_dtype,
            "context_len": ctx_len,
            "measurements_s": results,
            "weight_floor_replicated_s": floor_replicated,
            "weight_floor_sharded_s": floor_sharded,
            "ratio_to_replicated_floor": measured / floor_replicated,
        }
        out_path = Path(__file__).with_name("dflash_drafter_bench.json")
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out_path}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

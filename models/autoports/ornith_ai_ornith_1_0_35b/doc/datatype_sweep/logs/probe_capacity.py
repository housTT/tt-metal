# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Measured DRAM capacity at the **advertised** context, per KV-cache dtype.

`$datatype-sweep` requires the context contract to be recomputed for every KV-cache dtype candidate
that could change memory capacity, and forbids completing on a reduced advertised capability unless a
hard physical limit forces it. This probe is the measurement behind that: for each candidate it
builds the whole 40-layer model with that policy, allocates the paged KV cache at the full advertised
262144-token context for batch 1, captures the decode traces, and reads the allocator's own DRAM view
after each step - the same marks ``doc/optimized_full_model/logs/probe_footprint.py`` takes, so the
rows are directly comparable with the contract's existing blocks.

It then reports the **largest feasible context** for that dtype from what is left, which is the
figure the skill asks for when a candidate cannot reach the advertised one.

    python .../doc/datatype_sweep/logs/probe_capacity.py \
        --config .../candidates/C11-kv-bfp4.json --output .../capacity/C11-kv-bfp4.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import OrnithGenerator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import (
    OrnithModel,
    close_ornith_mesh,
    load_text_config,
    open_ornith_mesh,
    resolve_model_path,
)
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import DTYPE_BYTES
from models.autoports.ornith_ai_ornith_1_0_35b.tt.precision_config import policy_to_dict, resolve_policy

ADVERTISED_CONTEXT = 262144


def _view(mesh):
    device = mesh.get_devices()[0] if hasattr(mesh, "get_devices") else mesh
    return ttnn.get_memory_view(device, ttnn.BufferType.DRAM)


def dram_allocated(mesh) -> int:
    view = _view(mesh)
    return int(view.total_bytes_allocated_per_bank) * int(view.num_banks)


def dram_total(mesh) -> int:
    view = _view(mesh)
    return int(view.total_bytes_per_bank) * int(view.num_banks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--cache-context", type=int, default=ADVERTISED_CONTEXT)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    policy = resolve_policy(args.config or args.policy)
    mesh = open_ornith_mesh()
    marks = {}
    try:
        marks["empty"] = dram_allocated(mesh)
        marks["allocatable_total"] = dram_total(mesh)
        path = resolve_model_path()
        hf_config = load_text_config(path)
        model = OrnithModel.from_pretrained(path, mesh_device=mesh, hf_config=hf_config, policy=policy)
        marks["after_weights"] = dram_allocated(mesh)
        generator = OrnithGenerator(model, tokenizer=None, max_batch_size=args.batch, cache_context=args.cache_context)
        marks["after_state_and_kv_cache"] = dram_allocated(mesh)
        generator._ensure_decode_trace()
        marks["after_trace_capture"] = dram_allocated(mesh)

        # Per-token paged-KV bytes per device, from the live cache tensors rather than from a model.
        kv_bytes = 0.0
        for layer in model.layers:
            for cache in (getattr(layer, "k_cache", None), getattr(layer, "v_cache", None)):
                if cache is None:
                    continue
                elems = 1
                for d in cache.shape:
                    elems *= int(d)
                kv_bytes += elems * DTYPE_BYTES[cache.dtype]
        blocks = generator.total_blocks
        tokens_held = blocks * model.page_block_size
        kv_bytes_per_token = kv_bytes / tokens_held

        free = marks["allocatable_total"] - marks["after_trace_capture"]
        extra_tokens = int(free / kv_bytes_per_token / args.batch)
        report = {
            "config_id": policy.name,
            "precision_config_path": args.config,
            "kv_cache_dtype": str(policy.kv_cache_dtype),
            "dtype_policy": policy_to_dict(policy),
            "cache_context": args.cache_context,
            "batch": args.batch,
            "advertised_context": ADVERTISED_CONTEXT,
            "blocks_per_user": generator.blocks_per_user,
            "total_blocks": blocks,
            "page_block_size": model.page_block_size,
            "per_device_bytes": {
                "allocatable_total": marks["allocatable_total"],
                "at_start": marks["empty"],
                "weights_embedding_lm_head": marks["after_weights"] - marks["empty"],
                "kv_cache_and_per_batch_state": marks["after_state_and_kv_cache"] - marks["after_weights"],
                "trace_and_sampler": marks["after_trace_capture"] - marks["after_state_and_kv_cache"],
                "total_resident": marks["after_trace_capture"],
                "free_for_activations": free,
            },
            "kv_cache": {
                "per_device_bytes_all_layers": kv_bytes,
                "tokens_held_by_the_allocated_blocks": tokens_held,
                "per_device_bytes_per_token": kv_bytes_per_token,
            },
            "capacity": {
                "advertised_context_fits": args.cache_context >= ADVERTISED_CONTEXT,
                "largest_feasible_context_at_this_batch": ADVERTISED_CONTEXT + extra_tokens
                if args.cache_context >= ADVERTISED_CONTEXT
                else None,
                "largest_feasible_note": "the advertised context plus what the remaining free DRAM would "
                "hold at this dtype's measured bytes/token; it is a ceiling on paged KV alone and "
                "leaves nothing for activations, so it is reported as headroom evidence rather than as "
                "a context this model would advertise",
                "headroom_ratio": free / marks["after_trace_capture"],
            },
            "capability": model.capability(),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in ("dtype_policy", "capability")}, indent=2))
        generator.teardown()
        print("CAPACITY_OK")
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()

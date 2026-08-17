# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Measured per-device DRAM footprint of the whole 40-layer model, for the context contract.

Builds the real model, reads the allocator's own DRAM statistics after each stage of construction,
and prints the bytes each stage added. The context contract's capacity argument is then a
measurement rather than an arithmetic model — the arithmetic is still recorded next to it, and this
is what checks it.

    python .../doc/optimized_full_model/logs/probe_footprint.py --cache-context 8192
    python .../doc/optimized_full_model/logs/probe_footprint.py --cache-context 262144   # the advertised one
"""

from __future__ import annotations

import argparse
import json

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import OrnithGenerator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import (
    OrnithModel,
    close_ornith_mesh,
    load_text_config,
    open_ornith_mesh,
    resolve_model_path,
)


def _view(mesh):
    """The allocator's DRAM view for one device of the mesh.

    ``ttnn.get_memory_view`` reports per bank; the device figures below are per-bank x num_banks,
    which is what the context contract's "per device" numbers mean.
    """
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
    ap.add_argument("--cache-context", type=int, default=262144)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    mesh = open_ornith_mesh()
    marks = {}
    try:
        marks["empty"] = dram_allocated(mesh)
        marks["allocatable_total"] = dram_total(mesh)
        path = resolve_model_path()
        hf_config = load_text_config(path)
        model = OrnithModel.from_pretrained(path, mesh_device=mesh, hf_config=hf_config)
        marks["after_weights"] = dram_allocated(mesh)
        generator = OrnithGenerator(
            model,
            tokenizer=None,
            max_batch_size=args.batch,
            cache_context=args.cache_context,
        )
        marks["after_state_and_kv_cache"] = dram_allocated(mesh)
        generator._ensure_decode_trace()
        marks["after_trace_capture"] = dram_allocated(mesh)

        report = {
            "cache_context": args.cache_context,
            "batch": args.batch,
            "blocks_per_user": generator.blocks_per_user,
            "total_blocks": generator.total_blocks,
            "per_device_bytes": {
                "allocatable_total": marks["allocatable_total"],
                "at_start": marks["empty"],
                "weights_embedding_lm_head": marks["after_weights"] - marks["empty"],
                "kv_cache_and_per_batch_state": marks["after_state_and_kv_cache"] - marks["after_weights"],
                "trace_and_sampler": marks["after_trace_capture"] - marks["after_state_and_kv_cache"],
                "total_resident": marks["after_trace_capture"],
                "free_for_activations": marks["allocatable_total"] - marks["after_trace_capture"],
            },
            "capability": model.capability(),
        }
        print(json.dumps(report, indent=2, default=str))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json.dumps(report, indent=2, default=str) + "\n")
        generator.teardown()
        print("FOOTPRINT_OK")
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()

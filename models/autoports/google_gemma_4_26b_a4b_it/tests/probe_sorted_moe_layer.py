# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare the expert-sorted MoE prefill with the expert-union path on one real layer.

Usage (one chip, from the tt-metal worktree):
    TT_VISIBLE_DEVICES=2 ... python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_sorted_moe_layer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.test_full_model_contract import _config, _load_probe_state
from models.autoports.google_gemma_4_26b_a4b_it.tt import optimized_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import PRECISION_CONFIG
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import Gemma4FullModel


def _host(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--lengths", type=str, default="512,1024,2048")
    parser.add_argument("--output", type=Path, default=Path("/tmp/gemma4_sorted_moe_layer.json"))
    args = parser.parse_args()
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=0)
    results = []
    try:
        model = Gemma4FullModel(
            mesh_device=mesh,
            hf_config=_config(),
            state_dict=_load_probe_state(full_stack=False),
            max_seq_len=4096,
            max_batch_size=1,
            layer_indices=[args.layer],
            tensor_cache_path=os.environ.get("GEMMA4_TENSOR_CACHE_PATH", "/tmp/gemma4_sorted_moe_cache"),
            create_kv_cache=True,
            precision_config_path=PRECISION_CONFIG,
        )
        layer = model.layers[0]
        layer._in_decode_forward = False
        for m in [int(v) for v in args.lengths.split(",")]:
            tokens = torch.randint(1, 262_000, (1, m), generator=torch.Generator().manual_seed(m))
            tt_tokens = ttnn.from_torch(
                tokens.to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=model._replicate,
            )
            hidden = model.embed_tokens(tt_tokens)
            moe_in = layer._rms_norm(hidden, layer.weights.pre_ff_ln_2)
            routing = layer._router_weights(hidden)
            outputs = {}
            timings = {}
            for mode in ("union", "sorted", "union", "sorted"):
                optimized_decoder.SORTED_MOE_PREFILL = mode == "sorted"
                ttnn.synchronize_device(mesh)
                began = time.perf_counter()
                out = layer._moe_prefill(moe_in, routing)
                ttnn.synchronize_device(mesh)
                timings[mode] = min(timings.get(mode, 1e9), time.perf_counter() - began)
                outputs[mode] = _host(out).reshape(m, -1)
                ttnn.deallocate(out)
            diff = (outputs["union"] - outputs["sorted"]).abs()
            a = outputs["union"].flatten() - outputs["union"].mean()
            b = outputs["sorted"].flatten() - outputs["sorted"].mean()
            entry = {
                "m": m,
                "union_ms": round(timings["union"] * 1000, 2),
                "sorted_ms": round(timings["sorted"] * 1000, 2),
                "speedup": round(timings["union"] / timings["sorted"], 2),
                "pcc": round(float(a @ b / (a.norm() * b.norm())), 6),
                "max_abs": float(diff.max()),
                "nonzero_fraction": round(float((diff > 0).float().mean()), 6),
                "ref_abs_mean": round(float(outputs["union"].abs().mean()), 4),
            }
            results.append(entry)
            print(json.dumps(entry), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

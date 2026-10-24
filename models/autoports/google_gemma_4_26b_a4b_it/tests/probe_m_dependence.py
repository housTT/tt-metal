# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Find which prefill op changes its numerics with the physical sequence length M.

Usage (one chip, from the tt-metal worktree):
    TT_VISIBLE_DEVICES=1 ... python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_m_dependence \
        --layer 5 --lengths 64,128,1024,1536,2432 --reference 2560
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.test_full_model_contract import _config, _load_probe_state
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import PRECISION_CONFIG
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import Gemma4FullModel


def _host(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _compare(name: str, reference: torch.Tensor, candidate: torch.Tensor, m: int) -> dict:
    rows = candidate.shape[-2]
    ref = reference[..., :rows, :]
    diff = (ref - candidate).abs()
    return {
        "op": name,
        "m": m,
        "nonzero_fraction": round(float((diff > 0).float().mean()), 6),
        "max_abs": float(diff.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--lengths", type=str, default="64,128,1024,1536,2432")
    parser.add_argument("--reference", type=int, default=2560)
    parser.add_argument("--output", type=Path, default=Path("/tmp/gemma4_m_dependence.json"))
    args = parser.parse_args()
    lengths = [int(v) for v in args.lengths.split(",")]
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
            tensor_cache_path=os.environ.get("GEMMA4_TENSOR_CACHE_PATH", "/tmp/gemma4_m_dependence_cache"),
            create_kv_cache=True,
            precision_config_path=PRECISION_CONFIG,
        )
        layer = model.layers[0]
        layer_type = model.hf_config.layer_types[args.layer]
        kind = layer.layer_kind
        tokens = torch.randint(1, 262_000, (1, args.reference), generator=torch.Generator().manual_seed(0))
        positions = torch.arange(args.reference, dtype=torch.int32).reshape(1, -1)

        def stage_outputs(m: int) -> dict[str, torch.Tensor]:
            tt_tokens = ttnn.from_torch(
                tokens[:, :m].to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=model._replicate,
            )
            tt_pos = ttnn.from_torch(
                positions[:, :m],
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=model._replicate,
            )
            cos, sin = model._rope_rows(layer_type, tt_pos)
            layer._in_decode_forward = False
            hidden = model.embed_tokens(tt_tokens)
            out = {"embed": _host(hidden)}
            normed = layer._rms_norm(hidden, layer.weights.input_ln)
            out["input_rms_norm"] = _host(normed)
            xqkv = layer._linear(normed, "qkv", compute_kernel_config=layer.prefill_attention_compute_config)
            out["qkv_linear"] = _host(xqkv)
            local_kv = layer.local_full_kv_heads if kind.name == "full_attention" else layer.local_sliding_kv_heads
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                xqkv,
                num_heads=layer.local_q_heads,
                num_kv_heads=local_kv,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out["k_heads"] = _host(k)
            k_normed = layer._rms_norm(k, layer.weights.k_norm)
            out["k_norm"] = _host(k_normed)
            k_rope = ttnn.experimental.rotary_embedding_hf(k_normed, cos, sin, is_decode_mode=False)
            out["k_rope"] = _host(k_rope)
            q_normed = layer._rms_norm(q, layer.weights.q_norm)
            out["q_norm"] = _host(q_normed)
            mlp_in = layer._rms_norm(hidden, layer.weights.pre_ff_ln)
            out["dense_mlp"] = _host(layer._dense_mlp(mlp_in))
            router = layer._router_weights(hidden)
            out["router"] = _host(router)
            return out

        reference = stage_outputs(args.reference)
        for m in lengths:
            candidate = stage_outputs(m)
            for name in reference:
                entry = _compare(name, reference[name], candidate[name], m)
                results.append(entry)
                print(json.dumps(entry), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

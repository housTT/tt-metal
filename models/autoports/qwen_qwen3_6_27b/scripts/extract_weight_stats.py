# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Record per-tensor statistics of real Qwen3.6-27B decoder layers.

Usage::

    python -m models.autoports.qwen_qwen3_6_27b.scripts.extract_weight_stats

Writes ``doc/functional_decoder/weight_stats.json`` describing layer 0 (``linear_attention``)
and layer 3 (``full_attention``), plus the distribution of the hidden states that enter
layer 0 (sampled token embedding rows). Only the safetensors shards holding those tensors are
touched, never the whole 52 GB checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from models.autoports.qwen_qwen3_6_27b.reference.hf_reference import (
    FULL_VALUE_MAX_NUMEL,
    MODEL_ID,
    SNAPSHOT_PATH,
    WEIGHT_STATS_PATH,
    load_embedding_rows,
    load_real_layer_state_dict,
    load_text_config,
)


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.to(torch.float32)
    entry: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "abs_max": float(values.abs().max()),
    }
    if tensor.numel() <= FULL_VALUE_MAX_NUMEL:
        entry["values"] = [float(v) for v in values.flatten()]
    return entry


def collect_layer(layer_idx: int, layer_type: str) -> dict[str, Any]:
    state_dict = load_real_layer_state_dict(layer_idx, dtype=None)  # keep stored bfloat16
    return {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "tensors": {name: tensor_stats(tensor) for name, tensor in sorted(state_dict.items())},
    }


def collect_hidden_states_in(num_tokens: int, vocab_size: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    token_ids = torch.randint(0, vocab_size, (num_tokens,), generator=generator).tolist()
    embeddings = load_embedding_rows(token_ids, dtype=None)
    entry = tensor_stats(embeddings)
    entry.pop("values", None)
    entry["num_tokens"] = num_tokens
    entry["seed"] = seed
    entry["source"] = "model.language_model.embed_tokens.weight"
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3], help="decoder layer indices to profile")
    parser.add_argument("--num-tokens", type=int, default=256, help="token embedding rows to sample")
    parser.add_argument("--seed", type=int, default=0, help="seed for the sampled token ids")
    parser.add_argument("--out", type=Path, default=WEIGHT_STATS_PATH, help="output json path")
    args = parser.parse_args()

    config = load_text_config()

    layers: dict[str, Any] = {}
    for layer_idx in args.layers:
        layer_type = config.layer_types[layer_idx]
        print(f"[extract] layer {layer_idx} ({layer_type}) ...", flush=True)
        layers[str(layer_idx)] = collect_layer(layer_idx, layer_type)

    print(f"[extract] hidden_states_in from {args.num_tokens} embedding rows ...", flush=True)
    hidden_states_in = collect_hidden_states_in(args.num_tokens, config.vocab_size, args.seed)

    payload = {
        "model_id": MODEL_ID,
        "snapshot": str(SNAPSHOT_PATH),
        "layers": layers,
        "hidden_states_in": hidden_states_in,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    for layer_key, layer in layers.items():
        print(f"\nlayer {layer_key} ({layer['layer_type']}) - {len(layer['tensors'])} tensors")
        for name, entry in layer["tensors"].items():
            full = " [full values stored]" if "values" in entry else ""
            print(
                f"  {name:<40} {str(entry['shape']):<20} {entry['dtype']:<16}"
                f" mean={entry['mean']:+.5f} std={entry['std']:.5f} absmax={entry['abs_max']:.5f}{full}"
            )
    hs = hidden_states_in
    print(
        f"\nhidden_states_in: shape={hs['shape']} dtype={hs['dtype']}"
        f" mean={hs['mean']:+.6f} std={hs['std']:.6f} absmax={hs['abs_max']:.6f}"
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

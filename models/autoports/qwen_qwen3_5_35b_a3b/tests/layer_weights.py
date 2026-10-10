# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Selective checkpoint loading for one Qwen3.5 text-decoder layer."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping


def read_weight_map(index_path: Path) -> dict[str, str]:
    with Path(index_path).open() as stream:
        index = json.load(stream)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} has no weight_map object")
    return weight_map


def layer_weight_map(index_or_map: Path | Mapping[str, str], layer_idx: int) -> dict[str, str]:
    weight_map = read_weight_map(index_or_map) if isinstance(index_or_map, (str, Path)) else dict(index_or_map)
    prefixes = (f"model.language_model.layers.{layer_idx}.", f"model.layers.{layer_idx}.")
    selected = {key: filename for key, filename in weight_map.items() if key.startswith(prefixes)}
    if not selected:
        raise KeyError(f"no checkpoint weights found for text-decoder layer {layer_idx}")
    return selected


def layer_shards(index_or_map: Path | Mapping[str, str], layer_idx: int) -> tuple[str, ...]:
    return tuple(sorted(set(layer_weight_map(index_or_map, layer_idx).values())))


def load_layer_state_dict(checkpoint_dir: Path, layer_idx: int):
    """Load only safetensor shards/keys that contain ``layer_idx``."""
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_dir)
    selected = layer_weight_map(checkpoint_dir / "model.safetensors.index.json", layer_idx)
    by_file = defaultdict(list)
    for key, filename in selected.items():
        by_file[filename].append(key)

    tensors = {}
    for filename, keys in by_file.items():
        with safe_open(str(checkpoint_dir / filename), framework="pt", device="cpu") as shard:
            present = set(shard.keys())
            missing = set(keys) - present
            if missing:
                raise KeyError(f"{filename} is missing indexed tensors: {sorted(missing)}")
            tensors.update({key: shard.get_tensor(key) for key in keys})
    return tensors


def synthetic_layer_state_dict(stats_path: Path, seed: int = 1234):
    """Real target shapes and measured per-tensor statistics; no checkpoint needed."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    stats = json.loads(Path(stats_path).read_text())
    return {
        t["name"]: (torch.randn(t["shape"], generator=generator, dtype=torch.bfloat16) * t["std"] + t["mean"])
        for t in stats["tensors"]
    }

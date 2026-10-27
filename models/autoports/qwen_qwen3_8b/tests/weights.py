# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real and synthetic weight sources for the Qwen3-8B autoport tests.

Real weights come from the local Hugging Face snapshot (only the shards that
hold the requested layer are opened). Synthetic weights are generated
deterministically from ``doc/functional_decoder/real_weight_stats.json`` so
the default suite runs without the checkpoint while keeping real shapes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_DIR = Path("models/autoports/qwen_qwen3_8b")
STATS_PATH = MODEL_DIR / "doc" / "functional_decoder" / "real_weight_stats.json"
LAYER_KEYS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
CHUNK_ELEMENTS = 1 << 20


def model_snapshot() -> Path | None:
    explicit = os.getenv("QWEN3_MODEL_SNAPSHOT")
    if explicit:
        candidate = Path(explicit)
        return candidate if candidate.is_dir() else None
    roots = [Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")), Path.home() / ".cache/huggingface"]
    for root in roots:
        snapshots = root / "hub/models--Qwen--Qwen3-8B/snapshots"
        if snapshots.is_dir():
            for candidate in sorted(snapshots.iterdir()):
                if (candidate / "model.safetensors.index.json").is_file():
                    return candidate
    return None


def real_weights_available() -> bool:
    return model_snapshot() is not None


def _weight_map(snapshot: Path) -> dict[str, str]:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    return index["weight_map"]


def load_tensors(names: list[str], dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    snapshot = model_snapshot()
    if snapshot is None:
        raise FileNotFoundError("Qwen3-8B snapshot not found in the local Hugging Face cache")
    weight_map = _weight_map(snapshot)
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)
    out: dict[str, torch.Tensor] = {}
    for shard, shard_names in by_shard.items():
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as handle:
            for name in shard_names:
                out[name] = handle.get_tensor(name).to(dtype)
    return out


def load_layer_state(layer_idx: int, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Return the layer's canonical HF state dict with layer-local keys."""
    prefix = f"model.layers.{layer_idx}."
    tensors = load_tensors([prefix + key for key in LAYER_KEYS], dtype=dtype)
    return {key: tensors[prefix + key] for key in LAYER_KEYS}


def load_embedding(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return load_tensors(["model.embed_tokens.weight"], dtype=dtype)["model.embed_tokens.weight"]


def tensor_stats(name: str, tensor: torch.Tensor) -> dict:
    values = tensor.to(torch.float32)
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "seed": int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little"),
    }


def _seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def synthetic_weight(spec: dict) -> torch.Tensor:
    """Deterministic normal samples with the recorded mean/std, generated in fixed chunks."""
    numel = int(spec["numel"])
    generator = _seeded_generator(int(spec["seed"]))
    flat = torch.empty(numel, dtype=torch.float32)
    for start in range(0, numel, CHUNK_ELEMENTS):
        stop = min(start + CHUNK_ELEMENTS, numel)
        flat[start:stop] = torch.normal(
            float(spec["mean"]), float(spec["std"]), size=(stop - start,), generator=generator
        )
    dtype = getattr(torch, spec["dtype"])
    return flat.reshape(spec["shape"]).to(dtype)


def load_real_weight_stats() -> dict:
    return json.loads(STATS_PATH.read_text())


def synthetic_layer_state_dict(layer_idx: int) -> dict[str, torch.Tensor]:
    stats = load_real_weight_stats()
    layer = stats["layers"][str(layer_idx)]["weights"]
    return {key: synthetic_weight(layer[key]) for key in LAYER_KEYS}


def layer_state_dict(layer_idx: int) -> tuple[dict[str, torch.Tensor], str]:
    """Return ``(state_dict, source)`` where source is ``real`` or ``synthetic``."""
    if os.getenv("QWEN3_FORCE_SYNTHETIC_WEIGHTS") != "1" and real_weights_available():
        return load_layer_state(layer_idx), "real"
    return synthetic_layer_state_dict(layer_idx), "synthetic"

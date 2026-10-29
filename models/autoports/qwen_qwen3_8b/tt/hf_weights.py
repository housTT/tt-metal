# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hugging Face checkpoint access for the Qwen3-8B autoport.

Only the safetensors shards that hold the requested tensors are opened, so the
36 decoder layers, the embedding table and the LM head are loaded one group at
a time instead of materializing the 16 GB checkpoint on the host.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-8B"
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
EMBEDDING_KEY = "model.embed_tokens.weight"
FINAL_NORM_KEY = "model.norm.weight"
LM_HEAD_KEY = "lm_head.weight"


def model_snapshot(model_id: str = MODEL_ID) -> Path:
    explicit = os.getenv("QWEN3_MODEL_SNAPSHOT")
    if explicit and Path(explicit).is_dir():
        return Path(explicit)
    repo = "models--" + model_id.replace("/", "--")
    roots = [Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")), Path.home() / ".cache/huggingface"]
    for root in roots:
        snapshots = root / "hub" / repo / "snapshots"
        if snapshots.is_dir():
            for candidate in sorted(snapshots.iterdir()):
                if (candidate / "model.safetensors.index.json").is_file():
                    return candidate
    raise FileNotFoundError(f"{model_id} snapshot not found under HF_HOME (set QWEN3_MODEL_SNAPSHOT)")


class CheckpointReader:
    """Lazy reader over the safetensors shards of one snapshot."""

    def __init__(self, snapshot: Path | None = None):
        self.snapshot = Path(snapshot) if snapshot is not None else model_snapshot()
        index = json.loads((self.snapshot / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.config = json.loads((self.snapshot / "config.json").read_text())

    @property
    def revision(self) -> str:
        return self.snapshot.name

    def load(self, names: list[str], dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
        from safetensors import safe_open

        by_shard: dict[str, list[str]] = {}
        for name in names:
            by_shard.setdefault(self.weight_map[name], []).append(name)
        out: dict[str, torch.Tensor] = {}
        for shard, shard_names in by_shard.items():
            with safe_open(str(self.snapshot / shard), framework="pt", device="cpu") as handle:
                for name in shard_names:
                    out[name] = handle.get_tensor(name).to(dtype)
        return out

    def layer_state(self, layer_idx: int) -> dict[str, torch.Tensor]:
        prefix = f"model.layers.{layer_idx}."
        tensors = self.load([prefix + key for key in LAYER_KEYS])
        return {key: tensors[prefix + key] for key in LAYER_KEYS}

    def embedding(self) -> torch.Tensor:
        return self.load([EMBEDDING_KEY])[EMBEDDING_KEY]

    def final_norm(self) -> torch.Tensor:
        return self.load([FINAL_NORM_KEY])[FINAL_NORM_KEY]

    def lm_head(self) -> torch.Tensor:
        key = LM_HEAD_KEY if LM_HEAD_KEY in self.weight_map else EMBEDDING_KEY
        return self.load([key])[key]

    @property
    def tied_embeddings(self) -> bool:
        return LM_HEAD_KEY not in self.weight_map or bool(self.config.get("tie_word_embeddings", False))

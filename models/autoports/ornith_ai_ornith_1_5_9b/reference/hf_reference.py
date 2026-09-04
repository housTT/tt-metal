# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Layer-only HuggingFace reference for ornith-ai/Ornith-1.5-9B decoder layers.

The checkpoint is an 18 GB multimodal ``Qwen3_5ForConditionalGeneration``; loading the
whole causal LM to validate one decoder layer is neither necessary nor affordable. This
module instead:

* resolves the local snapshot and parses ``config.json`` into the real ``Qwen3_5TextConfig``;
* reads **only** the safetensors entries belonging to one ``layers.{i}`` subtree, using the exact checkpoint keys;
* instantiates a single ``Qwen3_5DecoderLayer`` and drives it through the real prefill and
  decode cache paths so the TTNN port is compared against HF semantics, not a re-derivation.

Weight-stat collection and deterministic synthetic-weight generation live here too, so CI-style
tests can run without the 18 GB download while still using the real shapes and distributions.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

HF_MODEL_ID = "ornith-ai/Ornith-1.5-9B"
HF_REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"

# Prefix of the text decoder inside the multimodal checkpoint.
CHECKPOINT_TEXT_PREFIX = "model.language_model."


# --------------------------------------------------------------------------------------
# Checkpoint discovery / config
# --------------------------------------------------------------------------------------
def resolve_model_path() -> Path:
    """Local snapshot directory for :data:`HF_MODEL_ID`.

    Uses the pinned local snapshot only. No unpinned network resolution is permitted.
    ORNITH_MODEL_PATH may name another copy of revision HF_REVISION.
    """
    import os

    return Path(os.environ.get("ORNITH_MODEL_PATH", "/home/hous/dev/ornith-1.5-9b/upstream"))


def load_text_config(model_path: Path | None = None) -> Qwen3_5TextConfig:
    """The real ``text_config`` from ``config.json`` — never a shrunken stand-in."""
    config_path = (
        Path(model_path) / "config.json"
        if model_path is not None
        else (Path(__file__).resolve().parents[1] / "doc/functional_decoder/hf_config.json")
    )
    with open(config_path) as f:
        raw = json.load(f)
    cfg = Qwen3_5TextConfig(**raw["text_config"])
    # `attn_implementation` decides which SDPA kernel the reference layer uses; eager keeps the
    # reference readable and deterministic and is what the PCC comparison is defined against.
    cfg._attn_implementation = "eager"
    return cfg


def layer_kind(text_config, layer_idx: int) -> str:
    return text_config.layer_types[layer_idx]


# --------------------------------------------------------------------------------------
# Per-layer state dict (real weights)
# --------------------------------------------------------------------------------------
def load_layer_state_dict(layer_idx: int, model_path: Path | None = None) -> dict:
    """Module-relative state dict for ``layers.{layer_idx}`` of the text decoder.

    Keys are relative to the ``Qwen3_5DecoderLayer`` module (``self_attn.q_proj.weight``,
    ``linear_attn.in_proj_qkv.weight``, ``mlp.gate_proj.weight``, ...).

    Dense MLP projections are loaded directly without expert conversion.
    """
    from safetensors import safe_open

    model_path = Path(model_path) if model_path is not None else resolve_model_path()
    with open(model_path / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    prefix = f"{CHECKPOINT_TEXT_PREFIX}layers.{layer_idx}."
    wanted = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    if not wanted:
        raise KeyError(f"no checkpoint entries under {prefix!r}")

    per_file: dict[str, list[str]] = {}
    for key, fname in wanted.items():
        per_file.setdefault(fname, []).append(key)

    raw: dict[str, torch.Tensor] = {}
    for fname, keys in per_file.items():
        with safe_open(str(model_path / fname), framework="pt", device="cpu") as f:
            for key in keys:
                raw[key] = f.get_tensor(key)

    return {key[len(prefix) :]: tensor for key, tensor in raw.items()}


# --------------------------------------------------------------------------------------
# Reference layer
# --------------------------------------------------------------------------------------
def build_reference_layer(text_config, layer_idx: int, state_dict: dict, dtype=torch.float32):
    """A single ``Qwen3_5DecoderLayer`` loaded with ``state_dict`` and put in eval mode.

    ``dtype`` defaults to float32: the golden reference is evaluated in float32 so the PCC bar
    measures the TTNN port, not torch's bf16 rounding.
    """
    with torch.device("meta"):
        layer = Qwen3_5DecoderLayer(text_config, layer_idx)
    layer.to_empty(device="cpu")
    missing, unexpected = layer.load_state_dict(
        {k: v.to(dtype) for k, v in state_dict.items()}, strict=False, assign=True
    )
    if unexpected:
        raise KeyError(f"unexpected keys for layer {layer_idx}: {sorted(unexpected)[:8]}")
    if missing:
        raise KeyError(f"missing keys for layer {layer_idx}: {sorted(missing)[:8]}")
    layer.eval()
    for p in layer.parameters():
        p.requires_grad_(False)
    return layer


def reference_position_embeddings(text_config, positions: torch.Tensor, dtype=torch.float32):
    """HF ``(cos, sin)`` for absolute text positions ``positions`` of shape ``[B, T]``.

    Ornith advertises interleaved M-RoPE; for a text-only request all three (t, h, w) grids carry
    the same position id, so the interleave selects identical rows and the result reduces to plain
    1-D partial RoPE over ``[B, T, rope_dim]``. Building it through the real
    ``Qwen3_5TextRotaryEmbedding`` keeps that claim honest rather than assumed.
    """
    rotary = Qwen3_5TextRotaryEmbedding(config=text_config)
    probe = torch.zeros(positions.shape[0], positions.shape[1], 1, dtype=dtype)
    pos3 = positions[None, ...].expand(3, positions.shape[0], positions.shape[1])
    return rotary(probe, pos3)


def _new_cache(text_config):
    return DynamicCache(config=text_config)


def reference_prefill(layer, text_config, hidden_states: torch.Tensor, start_pos: int = 0, cache=None):
    """Run reference prefill.

    Args:
        hidden_states: ``[B, T, hidden]`` float tensor entering the decoder layer.
        start_pos: absolute position of ``hidden_states[:, 0]``.
        cache: existing ``DynamicCache`` to continue from, or None to start fresh.

    Returns:
        ``(output, cache)`` where ``output`` is ``[B, T, hidden]``.
    """
    if cache is None:
        cache = _new_cache(text_config)
    B, T, _ = hidden_states.shape
    positions = torch.arange(start_pos, start_pos + T).unsqueeze(0).expand(B, T)
    cos, sin = reference_position_embeddings(text_config, positions, dtype=hidden_states.dtype)

    attention_mask = None
    if layer.layer_type == "full_attention":
        total = start_pos + T
        mask = torch.full((T, total), torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)
        rows = torch.arange(T).unsqueeze(1)
        cols = torch.arange(total).unsqueeze(0)
        mask = torch.where(cols <= start_pos + rows, torch.zeros((), dtype=hidden_states.dtype), mask)
        attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, T, total)

    out = layer(
        hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=attention_mask,
        position_ids=positions,
        past_key_values=cache,
    )
    return out, cache


def reference_decode(layer, text_config, hidden_states: torch.Tensor, positions: torch.Tensor, cache):
    """Run one reference decode step.

    Args:
        hidden_states: ``[B, 1, hidden]``.
        positions: ``[B]`` absolute positions of the token being decoded.
        cache: cache produced by :func:`reference_prefill` (mutated in place).
    """
    B = hidden_states.shape[0]
    pos2d = positions.reshape(B, 1)
    cos, sin = reference_position_embeddings(text_config, pos2d, dtype=hidden_states.dtype)

    attention_mask = None
    if layer.layer_type == "full_attention":
        # DynamicCache has already been extended by prefill; decode attends to everything cached
        # plus the new token, so a fully-unmasked row is correct.
        total = int(positions.max().item()) + 1
        attention_mask = torch.zeros(B, 1, 1, total, dtype=hidden_states.dtype)

    return layer(
        hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=attention_mask,
        position_ids=pos2d,
        past_key_values=cache,
    )


# --------------------------------------------------------------------------------------
# Weight statistics / synthetic weights
# --------------------------------------------------------------------------------------
def weight_stats(state_dict: dict) -> dict:
    """Per-tensor ``{name: {shape, dtype, mean, std, min, max}}`` for a layer state dict."""
    stats = {}
    for name, tensor in sorted(state_dict.items()):
        t = tensor.float()
        stats[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "mean": float(t.mean()),
            "std": float(t.std()) if t.numel() > 1 else 0.0,
            "min": float(t.min()),
            "max": float(t.max()),
        }
    return stats


def synthetic_state_dict(stats: dict, seed: int = 0, dtype=torch.float32) -> dict:
    """Deterministic synthetic weights matching the recorded real shapes/dtypes/moments.

    Uses the real shapes always. ``A_log`` and ``dt_bias`` are reproduced from their recorded
    ranges rather than as gaussians, because the gated-delta decay ``-exp(A_log) * softplus(a +
    dt_bias)`` is extremely sensitive to their sign and magnitude.
    """
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name, s in sorted(stats.items()):
        shape = tuple(s["shape"])
        if name.endswith("A_log"):
            # HF init: A ~ U(0, 16); A_log = log(A). Reproduce in the recorded range.
            lo, hi = s["min"], s["max"]
            t = torch.rand(shape, generator=gen) * (hi - lo) + lo
        elif name.endswith("dt_bias"):
            lo, hi = s["min"], s["max"]
            t = torch.rand(shape, generator=gen) * (hi - lo) + lo
        else:
            t = torch.randn(shape, generator=gen) * s["std"] + s["mean"]
        out[name] = t.to(dtype)
    return out

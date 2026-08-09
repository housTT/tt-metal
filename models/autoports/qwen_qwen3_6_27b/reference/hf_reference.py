# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace reference harness for Qwen/Qwen3.6-27B decoder layers.

This module gives the TT bring-up code a CPU/torch-only handle on a *single*
``Qwen3_5DecoderLayer`` without ever materialising the full 52 GB checkpoint:

* :func:`load_text_config`             - the ``Qwen3_5TextConfig`` (``text_config`` sub-config).
* :func:`load_real_layer_state_dict`   - one layer's weights, read only from the safetensors
                                         shards that actually hold that layer.
* :func:`build_reference_layer`        - an instantiated ``Qwen3_5DecoderLayer`` on CPU.
* :func:`make_rotary` / :func:`text_position_embeddings`
                                       - text-only mRoPE cos/sin, which provably collapse to
                                         plain RoPE with ``partial_rotary_factor=0.25``.
* :func:`synthetic_state_dict_from_stats` / :func:`synthetic_hidden_states`
                                       - deterministic stand-ins driven by
                                         ``doc/functional_decoder/weight_stats.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

MODEL_ID = "Qwen/Qwen3.6-27B"

# Used only when the hub cache cannot be probed (e.g. HF_HUB offline metadata missing).
_FALLBACK_SNAPSHOT_PATH = (
    "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/"
    "snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)


def _resolve_snapshot() -> str:
    try:
        return snapshot_download(MODEL_ID, local_files_only=True)
    except Exception:
        return _FALLBACK_SNAPSHOT_PATH


SNAPSHOT_PATH = _resolve_snapshot()

#: Prefix of a decoder layer inside the ``Qwen3_5ForConditionalGeneration`` checkpoint.
CHECKPOINT_LAYER_PREFIX = "model.language_model.layers."
#: Token embedding table of the language model.
EMBED_TOKENS_KEY = "model.language_model.embed_tokens.weight"

#: Where :mod:`..scripts.extract_weight_stats` writes its output.
WEIGHT_STATS_PATH = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder" / "weight_stats.json"

#: Tensors whose distribution carries semantics (SSM gating), never resampled from a normal.
VERBATIM_TENSOR_SUFFIXES = ("linear_attn.A_log", "linear_attn.dt_bias")

#: Tensors with at most this many elements are stored in full in ``weight_stats.json``.
FULL_VALUE_MAX_NUMEL = 64


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
def load_text_config() -> Qwen3_5TextConfig:
    """Return the ``qwen3_5_text`` sub-config of the Qwen3.6-27B checkpoint."""
    config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True).text_config
    assert isinstance(config, Qwen3_5TextConfig), f"unexpected text config type {type(config)}"
    config._attn_implementation = "eager"
    return config


# --------------------------------------------------------------------------------------
# real weights (partial safetensors load)
# --------------------------------------------------------------------------------------
def _weight_map() -> dict[str, str]:
    index_path = Path(SNAPSHOT_PATH) / "model.safetensors.index.json"
    with index_path.open() as fh:
        return json.load(fh)["weight_map"]


def _layer_checkpoint_keys(layer_idx: int) -> dict[str, str]:
    """Map ``<submodule-relative key> -> checkpoint key`` for one decoder layer."""
    prefix = f"{CHECKPOINT_LAYER_PREFIX}{layer_idx}."
    keys = {ckpt_key[len(prefix) :]: ckpt_key for ckpt_key in _weight_map() if ckpt_key.startswith(prefix)}
    assert keys, f"no checkpoint keys found for layer {layer_idx} (prefix {prefix!r})"
    return keys


def load_real_layer_state_dict(layer_idx: int, dtype: torch.dtype | None = torch.float32) -> dict[str, torch.Tensor]:
    """Load the real weights of decoder layer ``layer_idx``, keyed submodule-relative.

    Only the safetensors shards that actually contain this layer are opened, and only the
    tensors belonging to this layer are read out of them.

    Args:
        layer_idx: decoder layer index.
        dtype: dtype of the returned copies. ``None`` keeps the stored dtype (bfloat16).
    """
    weight_map = _weight_map()
    key_map = _layer_checkpoint_keys(layer_idx)

    shard_to_keys: dict[str, list[tuple[str, str]]] = {}
    for local_key, ckpt_key in key_map.items():
        shard_to_keys.setdefault(weight_map[ckpt_key], []).append((local_key, ckpt_key))

    state_dict: dict[str, torch.Tensor] = {}
    for shard, entries in sorted(shard_to_keys.items()):
        with safe_open(str(Path(SNAPSHOT_PATH) / shard), framework="pt", device="cpu") as fh:
            for local_key, ckpt_key in entries:
                tensor = fh.get_tensor(ckpt_key)
                state_dict[local_key] = tensor if dtype is None else tensor.to(dtype)
    return state_dict


def load_embedding_rows(token_ids: list[int], dtype: torch.dtype | None = torch.float32) -> torch.Tensor:
    """Read only the requested rows of ``model.language_model.embed_tokens.weight``."""
    shard = _weight_map()[EMBED_TOKENS_KEY]
    with safe_open(str(Path(SNAPSHOT_PATH) / shard), framework="pt", device="cpu") as fh:
        table = fh.get_slice(EMBED_TOKENS_KEY)
        rows = [table[int(i) : int(i) + 1, :] for i in token_ids]
    out = torch.cat(rows, dim=0)
    return out if dtype is None else out.to(dtype)


# --------------------------------------------------------------------------------------
# reference module
# --------------------------------------------------------------------------------------
def build_reference_layer(
    layer_idx: int,
    state_dict: dict[str, torch.Tensor] | None = None,
    dtype: torch.dtype = torch.float32,
) -> Qwen3_5DecoderLayer:
    """Instantiate a CPU ``Qwen3_5DecoderLayer`` and strict-load ``state_dict`` into it.

    ``state_dict`` defaults to the real checkpoint weights for that layer.
    """
    config = load_text_config()
    if state_dict is None:
        state_dict = load_real_layer_state_dict(layer_idx, dtype=dtype)

    layer = Qwen3_5DecoderLayer(config, layer_idx).to(device="cpu", dtype=dtype)
    layer.eval()

    missing, unexpected = layer.load_state_dict(state_dict, strict=True)
    assert not missing, f"layer {layer_idx}: missing keys {sorted(missing)}"
    assert not unexpected, f"layer {layer_idx}: unexpected keys {sorted(unexpected)}"
    return layer


# --------------------------------------------------------------------------------------
# rotary embeddings (text-only mRoPE)
# --------------------------------------------------------------------------------------
def make_rotary(config: Qwen3_5TextConfig) -> Qwen3_5TextRotaryEmbedding:
    return Qwen3_5TextRotaryEmbedding(config)


def build_text_position_ids(position_ids_1d: torch.Tensor, batch: int) -> torch.Tensor:
    """Replicate ``Qwen3_5TextModel.forward``'s text-only position ids: ``(4, batch, seq)``.

    Row 0 is the *text* position id; rows 1..3 are the temporal / height / width mRoPE rows,
    which for pure text carry the very same values.
    """
    assert position_ids_1d.ndim == 1, f"expected 1-D position ids, got {tuple(position_ids_1d.shape)}"
    return position_ids_1d.view(1, 1, -1).expand(4, batch, -1)


def _plain_rope_cos_sin(rotary: Qwen3_5TextRotaryEmbedding, position_ids_1d: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Plain (non-mRoPE) partial RoPE cos/sin for one position row: ``(seq, 2 * len(inv_freq))``."""
    freqs = position_ids_1d.float()[:, None] * rotary.inv_freq.float()[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * rotary.attention_scaling, emb.sin() * rotary.attention_scaling


def text_position_embeddings(
    rotary: Qwen3_5TextRotaryEmbedding,
    position_ids_1d: torch.Tensor,
    batch: int,
    verify_mrope_collapse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Text-only ``(cos, sin)`` of shape ``(batch, seq, 2 * len(inv_freq))`` == ``(batch, seq, 64)``.

    Built exactly the way ``Qwen3_5TextModel.forward`` does it: expand to 4 rows, hand rows
    ``[1:]`` to the rotary module. With ``verify_mrope_collapse`` the result is asserted to be
    bit-identical to plain RoPE with ``partial_rotary_factor=0.25``.
    """
    position_ids = build_text_position_ids(position_ids_1d, batch)
    rope_position_ids = position_ids[1:]
    probe = torch.zeros(batch, position_ids_1d.numel(), 1, dtype=torch.float32)
    cos, sin = rotary(probe, rope_position_ids)

    if verify_mrope_collapse:
        ref_cos, ref_sin = _plain_rope_cos_sin(rotary, position_ids_1d)
        assert torch.equal(cos, ref_cos.expand_as(cos)), "mRoPE sections did not collapse to plain RoPE (cos)"
        assert torch.equal(sin, ref_sin.expand_as(sin)), "mRoPE sections did not collapse to plain RoPE (sin)"
    return cos, sin


def build_causal_mask(
    config: Qwen3_5TextConfig,
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
    text_position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """The additive causal mask a ``full_attention`` layer expects, as built by the real model."""
    return create_causal_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )


# --------------------------------------------------------------------------------------
# synthetic weights / activations driven by weight_stats.json
# --------------------------------------------------------------------------------------
def load_weight_stats(path: Path | str = WEIGHT_STATS_PATH) -> dict[str, Any]:
    with Path(path).open() as fh:
        return json.load(fh)


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def synthetic_state_dict_from_stats(
    stats: dict[str, Any],
    layer_idx: int,
    config: Qwen3_5TextConfig,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Deterministic float32 stand-in weights matching the recorded name/shape/mean/std.

    Tensors recorded with full ``values`` (everything with <= ``FULL_VALUE_MAX_NUMEL``
    elements, which covers ``linear_attn.A_log`` and ``linear_attn.dt_bias``) are reproduced
    verbatim so that the SSM gating behaves realistically; everything else is a seeded normal
    draw with the recorded per-tensor mean and std.
    """
    layer_stats = stats["layers"][str(layer_idx)]
    assert layer_stats["layer_type"] == config.layer_types[layer_idx], (
        f"weight_stats layer {layer_idx} is {layer_stats['layer_type']!r} "
        f"but config says {config.layer_types[layer_idx]!r}"
    )

    state_dict: dict[str, torch.Tensor] = {}
    for name, entry in layer_stats["tensors"].items():
        shape = tuple(entry["shape"])
        if "values" in entry:
            tensor = torch.tensor(entry["values"], dtype=torch.float32).reshape(shape)
        else:
            assert not name.endswith(VERBATIM_TENSOR_SUFFIXES), f"{name} must be recorded verbatim, not sampled"
            generator = torch.Generator().manual_seed(_stable_seed(seed, name))
            tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
            tensor = tensor * float(entry["std"]) + float(entry["mean"])
        state_dict[name] = tensor

    _assert_shapes_consistent(state_dict, layer_idx, config)
    return state_dict


def _assert_shapes_consistent(state_dict: dict[str, torch.Tensor], layer_idx: int, config: Qwen3_5TextConfig) -> None:
    hidden = config.hidden_size
    assert state_dict["input_layernorm.weight"].shape == (hidden,)
    assert state_dict["post_attention_layernorm.weight"].shape == (hidden,)
    assert state_dict["mlp.gate_proj.weight"].shape == (config.intermediate_size, hidden)
    assert state_dict["mlp.down_proj.weight"].shape == (hidden, config.intermediate_size)
    if config.layer_types[layer_idx] == "linear_attention":
        key_dim = config.linear_key_head_dim * config.linear_num_key_heads
        value_dim = config.linear_value_head_dim * config.linear_num_value_heads
        assert state_dict["linear_attn.in_proj_qkv.weight"].shape == (2 * key_dim + value_dim, hidden)
        assert state_dict["linear_attn.conv1d.weight"].shape == (
            2 * key_dim + value_dim,
            1,
            config.linear_conv_kernel_dim,
        )
        assert state_dict["linear_attn.A_log"].shape == (config.linear_num_value_heads,)
        assert state_dict["linear_attn.dt_bias"].shape == (config.linear_num_value_heads,)
    else:
        assert state_dict["self_attn.q_proj.weight"].shape == (
            config.num_attention_heads * config.head_dim * 2,
            hidden,
        )
        assert state_dict["self_attn.k_proj.weight"].shape == (config.num_key_value_heads * config.head_dim, hidden)


def synthetic_hidden_states(
    config: Qwen3_5TextConfig,
    batch: int,
    seq: int,
    stats: dict[str, Any],
    seed: int = 0,
) -> torch.Tensor:
    """Activations approximating what really enters a decoder layer, from ``hidden_states_in``."""
    entry = stats["hidden_states_in"]
    generator = torch.Generator().manual_seed(_stable_seed(seed, "hidden_states_in"))
    noise = torch.randn(batch, seq, config.hidden_size, generator=generator, dtype=torch.float32)
    return noise * float(entry["std"]) + float(entry["mean"])

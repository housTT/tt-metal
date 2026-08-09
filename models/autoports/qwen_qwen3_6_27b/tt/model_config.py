# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Static shape/config derivation for the Qwen3.6-27B (``qwen3_5``) decoder layers.

Everything here is derived from the HuggingFace text config; nothing is inferred or
patched.  Invalid configurations fail immediately rather than being silently repaired.
"""

from __future__ import annotations

from dataclasses import dataclass

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"

#: Sub-chunk length of the gated-delta-rule recurrence.  This is the ``chunk_size``
#: argument of ``transformers``' ``torch_chunk_gated_delta_rule`` and is a property of
#: the algorithm, not of the model, so it is fixed here.
DELTA_CHUNK = 64


@dataclass(frozen=True)
class DecoderShapes:
    """Per-layer shape contract for one Qwen3.5/3.6 decoder layer."""

    layer_idx: int
    layer_type: str

    hidden_size: int
    intermediate_size: int
    rms_norm_eps: float
    max_position_embeddings: int

    # full_attention
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    attn_scaling: float

    # linear_attention
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    key_dim: int
    value_dim: int
    conv_dim: int
    conv_kernel_size: int

    @property
    def is_linear(self) -> bool:
        return self.layer_type == LINEAR_ATTENTION

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def v_per_k(self) -> int:
        return self.num_v_heads // self.num_k_heads


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def decoder_shapes(hf_config, layer_idx: int) -> DecoderShapes:
    """Derive the shape contract of decoder layer ``layer_idx`` from the HF text config."""
    layer_types = list(hf_config.layer_types)
    _require(
        0 <= layer_idx < len(layer_types),
        f"layer_idx {layer_idx} outside layer_types of length {len(layer_types)}",
    )
    layer_type = layer_types[layer_idx]
    _require(
        layer_type in (LINEAR_ATTENTION, FULL_ATTENTION),
        f"unsupported layer_type {layer_type!r} for layer {layer_idx}",
    )

    head_dim = hf_config.head_dim
    rope_parameters = dict(hf_config.rope_parameters)
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
    rotary_dim = int(head_dim * partial_rotary_factor)
    _require(rotary_dim % 32 == 0, f"rotary_dim {rotary_dim} must be a multiple of the tile width")
    _require(head_dim % 32 == 0, f"head_dim {head_dim} must be a multiple of the tile width")

    key_dim = hf_config.linear_key_head_dim * hf_config.linear_num_key_heads
    value_dim = hf_config.linear_value_head_dim * hf_config.linear_num_value_heads
    _require(
        hf_config.linear_num_value_heads % hf_config.linear_num_key_heads == 0,
        "linear_num_value_heads must be a multiple of linear_num_key_heads",
    )
    _require(
        hf_config.linear_key_head_dim == hf_config.linear_value_head_dim,
        "this implementation assumes linear_key_head_dim == linear_value_head_dim",
    )

    return DecoderShapes(
        layer_idx=layer_idx,
        layer_type=layer_type,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        max_position_embeddings=hf_config.max_position_embeddings,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        attn_scaling=head_dim**-0.5,
        num_k_heads=hf_config.linear_num_key_heads,
        num_v_heads=hf_config.linear_num_value_heads,
        head_k_dim=hf_config.linear_key_head_dim,
        head_v_dim=hf_config.linear_value_head_dim,
        key_dim=key_dim,
        value_dim=value_dim,
        conv_dim=2 * key_dim + value_dim,
        conv_kernel_size=hf_config.linear_conv_kernel_dim,
    )

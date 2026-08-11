# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Config parsing for ornith-ai/Ornith-1.0-35B (``Qwen3_5MoeForConditionalGeneration``).

The checkpoint is a hybrid Qwen3.5-MoE text decoder (plus a vision tower that this
stage does not touch). Two decoder-layer kinds alternate, selected per layer index by
``text_config.layer_types``:

* ``linear_attention`` — Gated DeltaNet token mixer (causal depthwise conv + gated
  delta rule); recurrent + conv state instead of a KV cache.
* ``full_attention``   — gated GQA softmax attention with partial RoPE and a paged
  KV cache.

Both kinds share the same MoE feed-forward block (256 routed experts, top-8, plus a
sigmoid-gated shared expert) and the same zero-centered RMSNorm/residual order.

Everything here is derived from the HF config; no field is invented or defaulted
behind the model's back.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OrnithDecoderConfig:
    """Static, config-derived shapes for one Ornith-1.0-35B decoder layer."""

    # --- shared ---
    dim: int  # hidden_size
    norm_eps: float  # rms_norm_eps
    max_position_embeddings: int  # HF-advertised context length
    num_hidden_layers: int
    layer_types: tuple  # per-layer kind

    # --- full attention ---
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: tuple
    mrope_interleaved: bool

    # --- gated deltanet (linear attention) ---
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int

    # --- MoE feed forward ---
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    hidden_act: str

    # ---- derived ----
    @property
    def rope_dim(self) -> int:
        """Rotated width of each attention head (partial RoPE)."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_q_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_k_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_v_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_dim(self) -> int:
        return self.linear_q_dim + self.linear_k_dim + self.linear_v_dim

    def layer_kind(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"

    @classmethod
    def from_hf_config(cls, hf_config) -> "OrnithDecoderConfig":
        """Build from either the top-level ``Qwen3_5MoeConfig`` or its ``text_config``."""
        text_config = getattr(hf_config, "text_config", None) or hf_config

        rope = text_config.rope_parameters
        if "rope_theta" not in rope:
            raise ValueError("text_config.rope_parameters must carry rope_theta")
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"unsupported rope_type {rope.get('rope_type')!r}; only 'default' is implemented")
        partial = rope.get("partial_rotary_factor", getattr(text_config, "partial_rotary_factor", 1.0))

        layer_types = tuple(text_config.layer_types)
        unknown = set(layer_types) - {"linear_attention", "full_attention"}
        if unknown:
            raise ValueError(f"unsupported layer_types {sorted(unknown)}")
        if text_config.hidden_act != "silu":
            raise ValueError(f"unsupported hidden_act {text_config.hidden_act!r}")

        return cls(
            dim=text_config.hidden_size,
            norm_eps=text_config.rms_norm_eps,
            max_position_embeddings=text_config.max_position_embeddings,
            num_hidden_layers=text_config.num_hidden_layers,
            layer_types=layer_types,
            n_heads=text_config.num_attention_heads,
            n_kv_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            rope_theta=float(rope["rope_theta"]),
            partial_rotary_factor=float(partial),
            mrope_section=tuple(rope.get("mrope_section", ())),
            mrope_interleaved=bool(rope.get("mrope_interleaved", False)),
            linear_num_key_heads=text_config.linear_num_key_heads,
            linear_num_value_heads=text_config.linear_num_value_heads,
            linear_key_head_dim=text_config.linear_key_head_dim,
            linear_value_head_dim=text_config.linear_value_head_dim,
            linear_conv_kernel_dim=text_config.linear_conv_kernel_dim,
            num_experts=text_config.num_experts,
            num_experts_per_tok=text_config.num_experts_per_tok,
            moe_intermediate_size=text_config.moe_intermediate_size,
            shared_expert_intermediate_size=text_config.shared_expert_intermediate_size,
            hidden_act=text_config.hidden_act,
        )


HF_MODEL_ID = "ornith-ai/Ornith-1.0-35B"

# Checkpoint prefix for the text decoder inside the multimodal checkpoint.
CHECKPOINT_TEXT_PREFIX = "model.language_model."

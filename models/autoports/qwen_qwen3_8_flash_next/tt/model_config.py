# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact Qwen/Qwen3.8-Flash-Next decoder-layer shape contract.

The Hugging Face checkpoint currently resolves to ``Qwen4ExpTextConfig``.  This
module deliberately validates the advertised target instead of accepting smaller
debug configurations: tests may use synthetic *values*, but never synthetic
shapes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

LINEAR_ATTENTION = "linear_attention"
QWEN_SPARSE_ATTENTION = "qwen_sparse_attention"
SUPPORTED_LAYER_TYPES = (LINEAR_ATTENTION, QWEN_SPARSE_ATTENTION)

HF_MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
HF_ADVERTISED_CONTEXT = 262_144

# Public prefill is logically unaligned.  Physical work is split into this
# target-independent recurrence/page-friendly quantum and the final chunk is
# masked and sliced back to its logical length.  128 is the datatype-ranked
# baseline; ``QWEN38_PREFILL_CHUNK`` may raise it in multiples of 128 so long
# prompts amortize the per-op fixed cost of the eager prefill graph.
PREFILL_CHUNK_BASE = 128
PREFILL_CHUNK = int(os.environ.get("QWEN38_PREFILL_CHUNK", PREFILL_CHUNK_BASE))
if PREFILL_CHUNK < PREFILL_CHUNK_BASE or PREFILL_CHUNK % PREFILL_CHUNK_BASE:
    raise ValueError(f"QWEN38_PREFILL_CHUNK must be a positive multiple of {PREFILL_CHUNK_BASE}, got {PREFILL_CHUNK}")
PAGE_BLOCK_SIZE = 64


@dataclass(frozen=True)
class DecoderShapes:
    layer_idx: int
    layer_type: str
    has_ple: bool
    hidden_size: int
    hc_count: int
    hc_hidden_size: int
    hc_lowrank: int
    rms_norm_eps: float
    max_position_embeddings: int

    # QSA
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int

    # Gated DeltaNet
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int

    # MoE
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int

    # PLE
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ple_conv_dilation: int
    ple_conv_state_len: int
    ngram_size: int
    heads_per_ngram: int

    @property
    def q_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def linear_qk_width(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_width(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_qkv_width(self) -> int:
        return 2 * self.linear_qk_width + self.linear_value_width


def decoder_shapes(hf_config, layer_idx: int) -> DecoderShapes:
    """Validate and capture the real text-decoder configuration."""

    cfg = getattr(hf_config, "text_config", hf_config)
    layer_types = tuple(cfg.layer_types)
    if not 0 <= layer_idx < len(layer_types):
        raise ValueError(f"layer_idx {layer_idx} is outside [0, {len(layer_types)})")
    layer_type = layer_types[layer_idx]
    if layer_type not in SUPPORTED_LAYER_TYPES:
        raise ValueError(f"unsupported Qwen4Exp decoder layer type: {layer_type!r}")

    expected = {
        "hidden_size": 2560,
        "hc_count": 4,
        "hc_lowrank": 320,
        "max_position_embeddings": HF_ADVERTISED_CONTEXT,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "ple_embed_dim": 2560,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
    }
    for name, value in expected.items():
        actual = getattr(cfg, name)
        if int(actual) != value:
            raise ValueError(f"{name}={actual!r}; this autoport requires target value {value}")

    if str(cfg.hidden_act) != "silu":
        raise ValueError(f"hidden_act={cfg.hidden_act!r}; expected 'silu'")
    if str(cfg.output_gate_type) != "sigmoid":
        raise ValueError(f"output_gate_type={cfg.output_gate_type!r}; expected 'sigmoid'")
    if not bool(cfg.norm_topk_prob):
        raise ValueError("Qwen3.8-Flash-Next requires normalized top-k router probabilities")
    if bool(cfg.attention_bias):
        raise ValueError("Qwen3.8-Flash-Next attention projections are bias-free")

    partial_rotary_factor = float(getattr(cfg, "partial_rotary_factor", cfg.rope_parameters["partial_rotary_factor"]))
    rotary_dim = int(cfg.head_dim * partial_rotary_factor)
    if rotary_dim != 64:
        raise ValueError(f"rotary_dim={rotary_dim}; expected 64")

    # HF stores PLE layer ids one-based and checks ``layer_idx + 1``.
    has_ple = layer_idx + 1 in tuple(int(i) for i in cfg.ple_layer_ids)
    ple_dilation = int(cfg.ngram_size)
    ple_state_len = (int(cfg.ple_conv_kernel_size) - 1) * ple_dilation

    return DecoderShapes(
        layer_idx=layer_idx,
        layer_type=layer_type,
        has_ple=has_ple,
        hidden_size=int(cfg.hidden_size),
        hc_count=int(cfg.hc_count),
        hc_hidden_size=int(cfg.hc_count) * int(cfg.hidden_size),
        hc_lowrank=int(cfg.hc_lowrank),
        rms_norm_eps=float(cfg.rms_norm_eps),
        max_position_embeddings=int(cfg.max_position_embeddings),
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=int(cfg.head_dim),
        rotary_dim=rotary_dim,
        indexer_n_heads=int(cfg.indexer_n_heads),
        indexer_kv_heads=int(cfg.indexer_kv_heads),
        indexer_head_dim=int(cfg.indexer_head_dim),
        indexer_budget=int(cfg.indexer_budget),
        indexer_compress_ratio=int(cfg.indexer_compress_ratio),
        linear_num_key_heads=int(cfg.linear_num_key_heads),
        linear_num_value_heads=int(cfg.linear_num_value_heads),
        linear_key_head_dim=int(cfg.linear_key_head_dim),
        linear_value_head_dim=int(cfg.linear_value_head_dim),
        linear_conv_kernel_dim=int(cfg.linear_conv_kernel_dim),
        num_experts=int(cfg.num_experts),
        num_experts_per_tok=int(cfg.num_experts_per_tok),
        moe_intermediate_size=int(cfg.moe_intermediate_size),
        shared_expert_intermediate_size=int(cfg.shared_expert_intermediate_size),
        ple_embed_dim=int(cfg.ple_embed_dim),
        ple_conv_kernel_size=int(cfg.ple_conv_kernel_size),
        ple_conv_dilation=ple_dilation,
        ple_conv_state_len=ple_state_len,
        ngram_size=int(cfg.ngram_size),
        heads_per_ngram=int(cfg.heads_per_ngram),
    )

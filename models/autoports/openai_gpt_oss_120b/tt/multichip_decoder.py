# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P150-family GPT-OSS 120B decoder layer with 1D tensor parallelism.

The single-device case is exactly :class:`OptimizedDecoder`.  P150x2 and
P150x4 use the repository GPT-OSS packed-QKV/local-head attention and routed
``sparse_matmul`` experts with TP-fractured weights and ring reductions.  The
public paged-cache and non-aligned logical-length contract is inherited from
the completed fused/optimized decoder boundary; padding stays internal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch

import ttnn
from models.autoports.openai_gpt_oss_120b.tt.fused_decoder import FusedDecoder, _local_layer_state_dict
from models.autoports.openai_gpt_oss_120b.tt.optimized_decoder import (
    DEFAULT_OPTIMIZED_POLICY,
    OptimizedDecoder,
    OptimizedDecoderPolicy,
    _DecodeShardedRMSNorm,
)
from models.common.lightweightmodule import LightweightModule
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.attention import Attention, AttentionConfig
from models.demos.gpt_oss.tt.attention.operations import apply_rope
from models.demos.gpt_oss.tt.attention_configs import GPTOSSAttentionProgramConfig
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig
from models.demos.gpt_oss.tt.experts.operations import (
    apply_routing_weights,
    apply_swiglu,
    apply_tensor_parallel_allreduce,
    reduce_experts,
)
from models.demos.gpt_oss.tt.mlp import MLP
from models.demos.gpt_oss.tt.topk import TopKRouter, topk_router
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name, get_default_num_links
from models.demos.gpt_oss.utils.substate import substate
from models.tt_transformers.tt.common import PagedAttentionConfig, rope_scaling_model_factory
from models.tt_transformers.tt.rope import RotarySetup

SUPPORTED_MESH_SHAPES = ((1, 1), (1, 2), (1, 4))
_SUPPORTED_LAYER_TYPES = {"sliding_attention", "full_attention"}
_DOWN_SUBBLOCK_WIDTH_BY_TP = {2: 1, 4: 3}


@dataclass(frozen=True)
class MultichipTensorPlan:
    """Calculated logical and padded per-rank dimensions for one mesh."""

    mesh_shape: tuple[int, int]
    tp: int
    hidden_size: int
    padded_local_hidden: int
    padded_hidden_size: int
    local_intermediate_size: int
    padded_local_intermediate_size: int
    local_q_heads: int
    local_kv_heads: int
    local_qkv_width: int


@dataclass(frozen=True)
class MultichipDecoderPolicy:
    """Static multichip dtype/topology policy selected at construction."""

    name: str = (
        "p150_1d_tp_replicated_residual_mixed_ccl_lofi_sparse_decode45x15_prefill45x45_tp2_subblock2_dram_output"
    )
    attention_weight_dtype: object = ttnn.bfloat8_b
    expert_weight_dtype: object = ttnn.bfloat4_b
    kv_cache_dtype: object = ttnn.bfloat8_b
    residual_layout: str = "replicated"
    topology: object = ttnn.Topology.Ring
    decode_dram_sharded_qkv: bool = False
    decode_dram_sharded_output: bool = True
    decode_dram_sharded_output_tp4: bool = False
    decode_dram_sharded_output_input_cores: int = 16
    decode_separate_qkv: bool = False
    decode_explicit_output_projection: bool = False
    decode_fused_output_projection_ccl: bool = False
    decode_separate_gate_up: bool = False
    router_weight_dtype: object = ttnn.bfloat16
    router_prefill_input_l1: bool = False
    router_prefill_explicit_program_config: bool = False
    activation_ccl_dtype: object = ttnn.bfloat16
    attention_activation_ccl_dtype: object | None = ttnn.bfloat8_b
    expert_activation_ccl_dtype: object | None = None
    projection_math_fidelity: object = ttnn.MathFidelity.LoFi
    expert_gate_up_cores: tuple[int, int] = (5, 9)
    expert_gate_up_in0_block_w: int = 30
    expert_gate_up_subblock_w: int = 1
    expert_gate_up_subblock_w_tp2: int | None = 2
    expert_down_cores: tuple[int, int] = (5, 3)
    expert_down_in0_block_w: int = 12
    expert_down_subblock_w: int | None = 6
    expert_prefill_down_cores: tuple[int, int] = (5, 9)
    expert_prefill_down_in0_block_w: int = 12
    expert_prefill_down_subblock_w: int | None = 2
    expert_prefill_down_cores_tp2: tuple[int, int] | None = None
    expert_prefill_down_subblock_w_tp2: int | None = None


DEFAULT_MULTICHIP_POLICY = MultichipDecoderPolicy()
PREOPTIMIZED_EXPERT_GEOMETRY_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_preoptimized_expert_geometry",
    decode_dram_sharded_output=False,
    expert_gate_up_cores=(3, 4),
    expert_gate_up_subblock_w=1,
    expert_gate_up_subblock_w_tp2=None,
    expert_down_cores=(5, 6),
    expert_down_subblock_w=None,
)
BFP8_ACTIVATION_CCL_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_bfp8_activation_ccl",
    activation_ccl_dtype=ttnn.bfloat8_b,
    attention_activation_ccl_dtype=ttnn.bfloat8_b,
    expert_activation_ccl_dtype=ttnn.bfloat8_b,
)
DRAM_SHARDED_QKV_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_dram_sharded_qkv",
    decode_dram_sharded_qkv=True,
)
DRAM_SHARDED_OUTPUT_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_dram_sharded_output",
    decode_dram_sharded_output=True,
    decode_dram_sharded_output_tp4=True,
    decode_dram_sharded_output_input_cores=8,
)
DRAM_SHARDED_OUTPUT_16_CORE_MULTICHIP_POLICY = replace(
    DRAM_SHARDED_OUTPUT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_dram_sharded_output_16_core",
    decode_dram_sharded_output_input_cores=16,
)
DRAM_SHARDED_OUTPUT_4_CORE_MULTICHIP_POLICY = replace(
    DRAM_SHARDED_OUTPUT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_dram_sharded_output_4_core",
    decode_dram_sharded_output_input_cores=4,
)
DRAM_SHARDED_OUTPUT_2_CORE_MULTICHIP_POLICY = replace(
    DRAM_SHARDED_OUTPUT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_dram_sharded_output_2_core",
    decode_dram_sharded_output_input_cores=2,
)
EXPLICIT_OUTPUT_PROJECTION_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_explicit_output_projection",
    decode_explicit_output_projection=True,
)
FUSED_OUTPUT_CCL_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_fused_output_ccl",
    decode_fused_output_projection_ccl=True,
)
SEPARATE_QKV_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_separate_qkv",
    decode_separate_qkv=True,
)
SEPARATE_GATE_UP_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_separate_gate_up",
    decode_separate_gate_up=True,
)
ATTENTION_BFP4_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_bfp4_attention",
    attention_weight_dtype=ttnn.bfloat4_b,
)
ATTENTION_LOFI_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_lofi_attention",
    projection_math_fidelity=ttnn.MathFidelity.LoFi,
)
ATTENTION_HIFI2_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_hifi2_attention",
    projection_math_fidelity=ttnn.MathFidelity.HiFi2,
)
ATTENTION_HIFI4_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_hifi4_attention",
    projection_math_fidelity=ttnn.MathFidelity.HiFi4,
)
EXPERT_BFP8_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_bfp8_experts",
    expert_weight_dtype=ttnn.bfloat8_b,
)
EXPERT_BF16_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_bf16_experts",
    expert_weight_dtype=ttnn.bfloat16,
)
EXPERT_GATE_UP_WIDE_SUBBLOCK_MULTICHIP_POLICY = replace(
    PREOPTIMIZED_EXPERT_GEOMETRY_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_gate_up_subblock2",
    expert_gate_up_subblock_w=2,
)
EXPERT_GATE_UP_NARROW_SUBBLOCK_MULTICHIP_POLICY = replace(
    PREOPTIMIZED_EXPERT_GEOMETRY_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_gate_up_subblock1",
    expert_gate_up_subblock_w=1,
)
EXPERT_GATE_UP_30_CORE_MULTICHIP_POLICY = replace(
    PREOPTIMIZED_EXPERT_GEOMETRY_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_gate_up_30_core",
    expert_gate_up_cores=(5, 6),
    expert_gate_up_subblock_w=2,
)
EXPERT_DOWN_48_CORE_MULTICHIP_POLICY = replace(
    PREOPTIMIZED_EXPERT_GEOMETRY_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_down_48_core",
    expert_down_cores=(6, 8),
    expert_down_subblock_w=2,
)
EXPERT_GATE_UP_15_CORE_MULTICHIP_POLICY = replace(
    PREOPTIMIZED_EXPERT_GEOMETRY_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_gate_up_15_core",
    expert_gate_up_cores=(5, 3),
    expert_gate_up_subblock_w=3,
)
EXPERT_DOWN_45_CORE_MULTICHIP_POLICY = replace(
    PREOPTIMIZED_EXPERT_GEOMETRY_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_down_45_core",
    expert_down_cores=(5, 9),
    expert_down_subblock_w=2,
)
EXPERT_GATE_UP_9_CORE_MULTICHIP_POLICY = replace(
    PREOPTIMIZED_EXPERT_GEOMETRY_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_gate_up_9_core",
    expert_gate_up_cores=(3, 3),
    expert_gate_up_subblock_w=5,
)
EXPERT_GATE_UP_45_CORE_MULTICHIP_POLICY = replace(
    PREOPTIMIZED_EXPERT_GEOMETRY_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_gate_up_45_core",
    expert_gate_up_cores=(5, 9),
    expert_gate_up_subblock_w=1,
)
EXPERT_GATE_UP_45_CORE_TP2_SUBBLOCK2_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_gate_up_45_core_tp2_subblock2",
    expert_gate_up_subblock_w_tp2=2,
)
EXPERT_DOWN_15_CORE_MULTICHIP_POLICY = replace(
    PREOPTIMIZED_EXPERT_GEOMETRY_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_down_15_core",
    expert_down_cores=(5, 3),
    expert_down_subblock_w=6,
)
EXPERT_DOWN_18_CORE_MULTICHIP_POLICY = replace(
    PREOPTIMIZED_EXPERT_GEOMETRY_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_down_18_core",
    expert_down_cores=(6, 3),
    expert_down_subblock_w=5,
)
PREFILL_DOWN_45_CORE_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_prefill_down_45_core",
    expert_prefill_down_cores=(5, 9),
    expert_prefill_down_subblock_w=2,
    expert_prefill_down_cores_tp2=(5, 9),
    expert_prefill_down_subblock_w_tp2=2,
)
ROUTER_PREFILL_EXPLICIT_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_router_prefill_explicit",
    router_prefill_explicit_program_config=True,
)
ROUTER_PREFILL_L1_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_router_prefill_l1",
    router_prefill_input_l1=True,
)
ROUTER_PREFILL_L1_EXPLICIT_MULTICHIP_POLICY = replace(
    ROUTER_PREFILL_EXPLICIT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_router_prefill_l1_explicit",
    router_prefill_input_l1=True,
)
SELECTED_EXPERT_GEOMETRY_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_selected_expert_geometry",
)
ROUTER_BFP8_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_bfp8_router",
    router_weight_dtype=ttnn.bfloat8_b,
)
ROUTER_BFP4_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_bfp4_router",
    router_weight_dtype=ttnn.bfloat4_b,
)
BF16_ACTIVATION_CCL_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_bf16_activation_ccl",
    activation_ccl_dtype=ttnn.bfloat16,
    attention_activation_ccl_dtype=ttnn.bfloat16,
    expert_activation_ccl_dtype=ttnn.bfloat16,
)
BFP4_ACTIVATION_CCL_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_bfp4_activation_ccl",
    activation_ccl_dtype=ttnn.bfloat4_b,
    attention_activation_ccl_dtype=ttnn.bfloat4_b,
    expert_activation_ccl_dtype=ttnn.bfloat4_b,
)
ATTENTION_BFP8_ACTIVATION_CCL_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_attention_bfp8_activation_ccl",
    attention_activation_ccl_dtype=ttnn.bfloat8_b,
    expert_activation_ccl_dtype=ttnn.bfloat16,
)
EXPERT_BFP8_ACTIVATION_CCL_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_expert_bfp8_activation_ccl",
    attention_activation_ccl_dtype=ttnn.bfloat16,
    expert_activation_ccl_dtype=ttnn.bfloat8_b,
)
ATTENTION_BFP4_ACTIVATION_CCL_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_attention_bfp4_activation_ccl",
    attention_activation_ccl_dtype=ttnn.bfloat4_b,
    expert_activation_ccl_dtype=ttnn.bfloat16,
)
EXPERT_BFP4_ACTIVATION_CCL_MULTICHIP_POLICY = replace(
    DEFAULT_MULTICHIP_POLICY,
    name="p150_1d_tp_replicated_residual_expert_bfp4_activation_ccl",
    attention_activation_ccl_dtype=ttnn.bfloat16,
    expert_activation_ccl_dtype=ttnn.bfloat4_b,
)
_SUPPORTED_MULTICHIP_POLICIES = (
    DEFAULT_MULTICHIP_POLICY,
    BFP8_ACTIVATION_CCL_MULTICHIP_POLICY,
    DRAM_SHARDED_QKV_MULTICHIP_POLICY,
    DRAM_SHARDED_OUTPUT_MULTICHIP_POLICY,
    DRAM_SHARDED_OUTPUT_16_CORE_MULTICHIP_POLICY,
    DRAM_SHARDED_OUTPUT_4_CORE_MULTICHIP_POLICY,
    DRAM_SHARDED_OUTPUT_2_CORE_MULTICHIP_POLICY,
    EXPLICIT_OUTPUT_PROJECTION_MULTICHIP_POLICY,
    FUSED_OUTPUT_CCL_MULTICHIP_POLICY,
    SEPARATE_QKV_MULTICHIP_POLICY,
    SEPARATE_GATE_UP_MULTICHIP_POLICY,
    ATTENTION_BFP4_MULTICHIP_POLICY,
    ATTENTION_LOFI_MULTICHIP_POLICY,
    ATTENTION_HIFI2_MULTICHIP_POLICY,
    ATTENTION_HIFI4_MULTICHIP_POLICY,
    EXPERT_BFP8_MULTICHIP_POLICY,
    EXPERT_BF16_MULTICHIP_POLICY,
    EXPERT_GATE_UP_WIDE_SUBBLOCK_MULTICHIP_POLICY,
    EXPERT_GATE_UP_NARROW_SUBBLOCK_MULTICHIP_POLICY,
    EXPERT_GATE_UP_30_CORE_MULTICHIP_POLICY,
    EXPERT_DOWN_48_CORE_MULTICHIP_POLICY,
    EXPERT_GATE_UP_15_CORE_MULTICHIP_POLICY,
    EXPERT_DOWN_45_CORE_MULTICHIP_POLICY,
    EXPERT_GATE_UP_9_CORE_MULTICHIP_POLICY,
    EXPERT_GATE_UP_45_CORE_MULTICHIP_POLICY,
    EXPERT_GATE_UP_45_CORE_TP2_SUBBLOCK2_MULTICHIP_POLICY,
    EXPERT_DOWN_15_CORE_MULTICHIP_POLICY,
    EXPERT_DOWN_18_CORE_MULTICHIP_POLICY,
    PREFILL_DOWN_45_CORE_MULTICHIP_POLICY,
    ROUTER_PREFILL_EXPLICIT_MULTICHIP_POLICY,
    ROUTER_PREFILL_L1_MULTICHIP_POLICY,
    ROUTER_PREFILL_L1_EXPLICIT_MULTICHIP_POLICY,
    SELECTED_EXPERT_GEOMETRY_MULTICHIP_POLICY,
    ROUTER_BFP8_MULTICHIP_POLICY,
    ROUTER_BFP4_MULTICHIP_POLICY,
    BF16_ACTIVATION_CCL_MULTICHIP_POLICY,
    BFP4_ACTIVATION_CCL_MULTICHIP_POLICY,
    ATTENTION_BFP8_ACTIVATION_CCL_MULTICHIP_POLICY,
    EXPERT_BFP8_ACTIVATION_CCL_MULTICHIP_POLICY,
    ATTENTION_BFP4_ACTIVATION_CCL_MULTICHIP_POLICY,
    EXPERT_BFP4_ACTIVATION_CCL_MULTICHIP_POLICY,
)


class _OptimizedMultichipBackend(FusedDecoder):
    """Decode backend that borrows its input and preserves the L1 stack boundary.

    The generic decoder clones every decode input to DRAM because its shared
    forward helper consumes the first residual.  A captured stack input is
    immutable, however: both residual adds write into their branch output.
    Keeping that first tensor borrowed removes the per-layer DRAM clone and
    allows one layer's L1 output to feed the next layer's sharded norm directly.
    """

    def _forward(self, *args, is_decode, **kwargs):
        self.input_layernorm.decode_mode = is_decode
        self.post_attention_layernorm.decode_mode = is_decode
        if not is_decode:
            return super()._forward(*args, is_decode=False, **kwargs)

        hidden_states = args[0]
        position_embeddings = kwargs["position_embeddings"]
        current_position = kwargs["current_position"]
        page_table = kwargs["page_table"]
        kv_cache = kwargs["kv_cache"]
        batch_size = kwargs["batch_size"]

        borrowed_input = hidden_states
        normed = self.input_layernorm(hidden_states)
        attention_out = self.self_attn(
            normed,
            rope_mats=position_embeddings,
            position_idx=current_position,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
            user_id=0,
            batch_size=batch_size,
        )
        normed.deallocate(True)
        hidden_states = ttnn.add(borrowed_input, attention_out, output_tensor=attention_out)

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed, is_decode=True)
        normed.deallocate(True)
        hidden_states = ttnn.add(residual, mlp_out, output_tensor=mlp_out)
        residual.deallocate(True)
        return hidden_states

    def decode_forward(
        self,
        hidden_states,
        *,
        position_embeddings,
        current_position,
        page_table,
        kv_cache=None,
        batch_size=1,
    ):
        if page_table is None:
            raise ValueError("FusedDecoder is paged-only and requires page_table")
        if len(hidden_states.shape) != 4 or hidden_states.shape[0] != 1 or hidden_states.shape[1] != 1:
            raise ValueError(f"decode requires [1, 1, batch, hidden] input, got {tuple(hidden_states.shape)}")
        if hidden_states.shape[-2] != batch_size or hidden_states.shape[-1] != self.hf_config.hidden_size:
            raise ValueError(
                "decode input must match batch_size and hidden size, "
                f"got {tuple(hidden_states.shape)} and batch_size={batch_size}"
            )
        if batch_size < 1 or batch_size > self.max_batch_size:
            raise ValueError(f"batch_size {batch_size} is outside configured maximum {self.max_batch_size}")
        if current_position is None or current_position.shape[-1] < batch_size:
            raise ValueError("decode requires a device-resident current_position covering the decode batch")
        if len(position_embeddings) != 2 or any(rope.shape[1] < batch_size for rope in position_embeddings):
            raise ValueError("decode position_embeddings must be a cosine/sine pair covering the decode batch")
        if page_table.shape[-2] < batch_size:
            raise ValueError(f"page_table has {page_table.shape[-2]} rows for batch_size={batch_size}")
        return self._forward(
            hidden_states,
            position_embeddings=position_embeddings,
            current_position=current_position,
            page_table=page_table,
            kv_cache=self.kv_cache if kv_cache is None else kv_cache,
            is_decode=True,
            user_id=0,
            batch_size=batch_size,
        )


def tensor_plan(mesh_shape, hf_config) -> MultichipTensorPlan:
    """Return the setup-time TP shape/padding plan and reject invalid meshes."""
    shape = tuple(int(value) for value in mesh_shape)
    if shape not in SUPPORTED_MESH_SHAPES:
        raise ValueError(f"multichip decoder supports mesh shapes {SUPPORTED_MESH_SHAPES}, got {shape}")
    tp = shape[1]
    if hf_config.hidden_size != 2880 or hf_config.intermediate_size != 2880 or hf_config.head_dim != 64:
        raise ValueError("Expected openai/gpt-oss-120b hidden/intermediate/head dimensions (2880, 2880, 64)")
    if hf_config.num_attention_heads % tp or hf_config.num_key_value_heads % tp:
        raise ValueError(
            f"TP={tp} must divide Q/KV heads ({hf_config.num_attention_heads}, {hf_config.num_key_value_heads})"
        )

    hidden_size = int(hf_config.hidden_size)
    local_hidden = hidden_size // tp
    padded_local_hidden = math.ceil(local_hidden / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    local_intermediate = int(hf_config.intermediate_size) // tp
    padded_local_intermediate = math.ceil(local_intermediate / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    qkv_width = int(hf_config.num_attention_heads) * int(hf_config.head_dim) + 2 * int(
        hf_config.num_key_value_heads
    ) * int(hf_config.head_dim)
    return MultichipTensorPlan(
        mesh_shape=shape,
        tp=tp,
        hidden_size=hidden_size,
        padded_local_hidden=padded_local_hidden,
        padded_hidden_size=padded_local_hidden * tp,
        local_intermediate_size=local_intermediate,
        padded_local_intermediate_size=padded_local_intermediate,
        local_q_heads=int(hf_config.num_attention_heads) // tp,
        local_kv_heads=int(hf_config.num_key_value_heads) // tp,
        local_qkv_width=qkv_width // tp,
    )


def _allreduce_physical_hidden(
    tensor,
    *,
    hidden_size,
    padded_hidden_size,
    mesh_config,
    ccl_manager,
    memory_config=None,
):
    """Reduce a tile-divisible physical width, then restore logical width.

    TP4's logical 2880-wide residual has 90 tiles, which cannot be evenly
    reduce-scattered over four ranks.  Its natural per-rank tile padding gives
    2944 = 4 * 736 columns (92 tiles), selecting the native ring RS+AG path.
    Padding is produced by the row-parallel projection weights rather than by
    the public activation contract.
    """
    if tensor.shape[-1] != padded_hidden_size:
        raise ValueError(f"physical-hidden collective expected width {padded_hidden_size}, got {tensor.shape[-1]}")
    kwargs = {}
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    reduced = ttnn.all_reduce(
        tensor,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        cluster_axis=mesh_config.tp_axis,
        **kwargs,
    )
    tensor.deallocate(True)
    if padded_hidden_size == hidden_size:
        return reduced
    logical = ttnn.slice(
        reduced,
        starts=[0] * len(reduced.shape),
        ends=[*[int(reduced.shape[index]) for index in range(len(reduced.shape) - 1)], hidden_size],
        steps=[1] * len(reduced.shape),
    )
    reduced.deallocate(True)
    return logical


class _PhysicalHiddenCollectiveAttention(Attention):
    """Attention decode variant that reduces TP4's 2944 physical columns."""

    def __call__(
        self,
        hidden_states,
        rope_mats,
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=True,
        user_id=0,
        batch_size=1,
    ):
        if not is_decode:
            return super().__call__(
                hidden_states,
                rope_mats,
                position_idx=position_idx,
                page_table=page_table,
                kv_cache=kv_cache,
                is_decode=False,
                user_id=user_id,
                batch_size=batch_size,
            )
        cache = kv_cache if kv_cache is not None else self.kv_cache
        transformation_mat = self.transformation_mats["decode"] if self.transformation_mats else None
        return self._decode_forward(
            hidden_states,
            rope_mats,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=cache,
            transformation_mat=transformation_mat,
        )

    def _decode_forward(
        self,
        hidden_states,
        rope_mats,
        *,
        position_idx,
        page_table,
        kv_cache,
        transformation_mat,
    ):
        """Canonical GPT-OSS decode with only the output-collective tail changed."""
        _, seq_len, batch_size, hidden_size = hidden_states.shape
        if seq_len != 1:
            raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")

        qkv_input = (
            ttnn.to_memory_config(hidden_states, self.decode_qkv_input_memory_config)
            if self.decode_qkv_input_memory_config is not None
            else hidden_states
        )
        if self.decode_separate_qkv:
            projected = []
            for weight, bias in zip(self.decode_separate_qkv_weights, self.decode_separate_qkv_biases):
                output = ttnn.linear(
                    qkv_input,
                    weight,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=self.decode_projection_compute_kernel_config,
                )
                output = ttnn.add(output, bias, output_tensor=output)
                projected.append(output)
            xqkv_fused = ttnn.concat(projected, dim=-1)
            for output in projected:
                output.deallocate(True)
        else:
            xqkv_fused = ttnn.linear(
                qkv_input,
                self.decode_wqkv,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.decode_projection_compute_kernel_config,
                program_config=self.decode_qkv_program_config,
            )
        if qkv_input is not hidden_states:
            qkv_input.deallocate(True)
        if self.decode_qkv_to_interleaved:
            sharded_qkv = xqkv_fused
            xqkv_fused = ttnn.sharded_to_interleaved(sharded_qkv, ttnn.DRAM_MEMORY_CONFIG)
            sharded_qkv.deallocate(True)
        if not self.decode_separate_qkv:
            ttnn.add(xqkv_fused, self.weights.wqkv_bias, output_tensor=xqkv_fused)

        num_local_heads = self.mesh_config.shard_size(self.config.num_heads)
        num_local_kv_heads = self.mesh_config.shard_size(self.config.num_kv_heads)
        tt_q, tt_k, tt_v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=num_local_heads,
            num_kv_heads=num_local_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        xqkv_fused.deallocate(True)

        tt_q_orig = tt_q
        tt_k_orig = tt_k
        tt_q = apply_rope(tt_q, rope_mats, transformation_mat, is_decode_mode=True)
        tt_k = apply_rope(tt_k, rope_mats, transformation_mat, is_decode_mode=True)
        tt_q_orig.deallocate(True)
        tt_k_orig.deallocate(True)

        k_cache, v_cache = kv_cache
        tt_k = ttnn.to_memory_config(tt_k, self.kv_mem_cfg)
        tt_v = ttnn.to_memory_config(tt_v, self.kv_mem_cfg)
        ttnn.experimental.paged_update_cache(
            k_cache,
            tt_k,
            update_idxs_tensor=position_idx,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            v_cache,
            tt_v,
            update_idxs_tensor=position_idx,
            page_table=page_table,
        )
        tt_k.deallocate(True)
        tt_v.deallocate(True)

        grid_size = ttnn.CoreCoord(8, 8)
        batch_grid = ttnn.num_cores_to_corerangeset(batch_size, grid_size, row_wise=True)
        padded_heads = math.ceil(num_local_heads / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        height_sharded_mem_config = ttnn.create_sharded_memory_config(
            shape=(padded_heads, self.config.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        if page_table is not None:
            tt_sdpa_tensor = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                tt_q,
                k_cache,
                v_cache,
                cur_pos_tensor=position_idx,
                sliding_window_size=self.config.sliding_window,
                attention_sink=self.weights.decode_sinks,
                page_table_tensor=page_table,
                scale=self.config.scaling,
                program_config=self.program_config.get_decode_sdpa_config(self.mesh_device),
                compute_kernel_config=self.program_config.get_compute_kernel_config(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            tt_sdpa_tensor = ttnn.transformer.scaled_dot_product_attention_decode(
                tt_q,
                k_cache,
                v_cache,
                cur_pos_tensor=position_idx,
                sliding_window_size=self.config.sliding_window,
                attention_sink=self.weights.decode_sinks,
                scale=self.config.scaling,
                program_config=self.program_config.get_decode_sdpa_config(self.mesh_device),
                compute_kernel_config=self.program_config.get_compute_kernel_config(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        tt_sdpa_tensor = ttnn.to_memory_config(tt_sdpa_tensor, height_sharded_mem_config)
        tt_q.deallocate(True)

        tt_sdpa_out = ttnn.experimental.nlp_concat_heads_decode(tt_sdpa_tensor, num_heads=num_local_heads)
        tt_sdpa_tensor.deallocate(True)
        output_input = (
            ttnn.to_memory_config(tt_sdpa_out, self.decode_output_input_memory_config)
            if self.decode_output_input_memory_config is not None
            and tt_sdpa_out.memory_config() != self.decode_output_input_memory_config
            else tt_sdpa_out
        )
        padded_local_hidden = math.ceil((hidden_size // self.mesh_config.tp) / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        padded_hidden = padded_local_hidden * self.mesh_config.tp
        if self.decode_fused_output_projection_ccl:
            fused_input = ttnn.to_memory_config(output_input, ttnn.DRAM_MEMORY_CONFIG)
            if fused_input.dtype != self.activation_ccl_dtype:
                cast_input = ttnn.typecast(fused_input, self.activation_ccl_dtype)
                if fused_input is not output_input:
                    fused_input.deallocate(True)
                fused_input = cast_input
            matmul_output, scattered = ttnn.experimental.matmul_reduce_scatter_async(
                fused_input,
                self.decode_o_proj,
                persistent_intermediate_buffer=self.decode_output_mmrs_intermediate,
                persistent_output_buffer=self.decode_output_mmrs_output,
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_rs_ping_pong_semaphore(),
                reduce_scatter_core_grid_offset=(0, 6),
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(),
                bias=self.decode_o_proj_bias,
                num_links=self.ccl_manager.num_links,
                memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_manager.topology,
                subdevice_id=None,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.decode_output_mmrs_program_config,
                compute_kernel_config=self.decode_projection_compute_kernel_config,
            )
            gathered = ttnn.experimental.all_gather_async(
                scattered,
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            tt_out = ttnn.slice(
                gathered,
                starts=[0, 0, 0, 0],
                ends=[gathered.shape[0], gathered.shape[1], gathered.shape[2], hidden_size],
                steps=[1, 1, 1, 1],
            )
            fused_input.deallocate(True)
            matmul_output.deallocate(True)
            gathered.deallocate(True)
            if output_input is not tt_sdpa_out:
                output_input.deallocate(True)
            tt_sdpa_out.deallocate(True)
            return ttnn.reshape(
                tt_out,
                (1, 1, batch_size, hidden_size),
                (1, 1, ttnn.TILE_SIZE, hidden_size),
            )

        tt_out = ttnn.linear(
            output_input,
            self.decode_o_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.decode_projection_compute_kernel_config,
            program_config=self.decode_output_program_config,
        )
        output_input.deallocate(True)
        if output_input is not tt_sdpa_out:
            tt_sdpa_out.deallocate(True)
        if self.decode_output_to_interleaved:
            sharded_output = tt_out
            tt_out = ttnn.sharded_to_interleaved(sharded_output, ttnn.DRAM_MEMORY_CONFIG)
            sharded_output.deallocate(True)
        tt_out = ttnn.add(tt_out, self.decode_o_proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        if tt_out.dtype != self.activation_ccl_dtype:
            projection_output = tt_out
            tt_out = ttnn.typecast(projection_output, self.activation_ccl_dtype)
            projection_output.deallocate(True)

        tt_out = ttnn.reshape(
            tt_out,
            (1, 1, batch_size, self.decode_output_physical_hidden),
            (1, 1, ttnn.TILE_SIZE, self.decode_output_physical_hidden),
        )
        output = _allreduce_physical_hidden(
            tt_out,
            hidden_size=hidden_size,
            padded_hidden_size=self.decode_output_physical_hidden,
            mesh_config=self.mesh_config,
            ccl_manager=self.ccl_manager,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # BFP4 is legal for the attention projection and ring payload but is
        # not a legal input dtype for the following sharded RMSNorm.  Restore
        # only at that consumer boundary so the lower-movement CCL candidate
        # is measured with its intended payload instead of rejected at the
        # first API validation error.
        if output.dtype == ttnn.bfloat4_b:
            converted = ttnn.typecast(output, ttnn.bfloat16)
            output.deallocate(True)
            output = converted
        return output


class _ReplicatedL1Router(TopKRouter):
    """Replicated BF16 router with setup-time L1-resident weights."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        tensor_cache_path=None,
        weight_dtype=ttnn.bfloat16,
        *,
        prefill_input_l1=False,
        prefill_explicit_program_config=False,
    ):
        self.top_k = hf_config.num_experts_per_tok
        self.num_experts = hf_config.num_local_experts
        self.hidden_dim = hf_config.hidden_size
        self.tensor_cache_path = tensor_cache_path
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        self.weight = ttnn.as_tensor(
            state_dict["weight"].transpose(0, 1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight_l1_replicated"),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.bias = ttnn.as_tensor(
            state_dict["bias"].unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "bias_l1_replicated"),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.compute_config = None
        self.prefill_input_l1 = prefill_input_l1
        self.prefill_program_config = (
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(4, 4),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=1,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
            if prefill_explicit_program_config
            else None
        )
        self.softmax_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Decode needs the ordinary sparse score tensor.  The fused router's
        # sparse [token,k] contract is reserved for throughput experts.
        self.use_fused_op = False
        self._fused_bias = None
        self._bias_torch = None

    def __call__(self, hidden_states, use_throughput_experts):
        """Apply the opt-in prefill placement/config while preserving decode."""
        actual_tokens = hidden_states.volume() // self.hidden_dim
        if actual_tokens <= ttnn.TILE_SIZE:
            return super().__call__(hidden_states, use_throughput_experts)

        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))
        router_input = hidden_states
        if self.prefill_input_l1:
            router_input = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        router_logits = ttnn.linear(
            router_input,
            self.weight,
            bias=self.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=self.prefill_program_config,
            compute_kernel_config=self.compute_config,
        )
        if router_input is not hidden_states:
            router_input.deallocate(True)
        expert_indices, expert_weights = topk_router(
            router_logits,
            self.top_k,
            use_throughput_experts,
            self.softmax_compute_config,
        )
        router_logits.deallocate(True)
        return expert_indices, expert_weights


class _ActiveExpertTPMLP(MLP):
    """Packed TP sparse experts with batch-safe decode at the autoport boundary."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        *,
        tensor_cache_path,
        mesh_config,
        expert_weight_dtype,
        router_weight_dtype=ttnn.bfloat16,
        router_prefill_input_l1=False,
        router_prefill_explicit_program_config=False,
        activation_ccl_dtype=ttnn.bfloat8_b,
        separate_gate_up=False,
        gate_up_cores=(3, 4),
        gate_up_in0_block_w=30,
        gate_up_subblock_w=1,
        down_cores=(5, 6),
        down_in0_block_w=12,
        down_subblock_w=None,
        prefill_down_cores=(5, 9),
        prefill_down_in0_block_w=12,
        prefill_down_subblock_w=2,
    ):
        super().__init__(
            mesh_device,
            hf_config,
            state_dict,
            ccl_manager,
            dtype=ttnn.bfloat16,
            tensor_cache_path=tensor_cache_path,
            mesh_config=mesh_config,
            use_throughput_experts=False,
        )
        old_router = self.router
        old_router.weight.deallocate(True)
        old_router.bias.deallocate(True)
        self.router = _ReplicatedL1Router(
            mesh_device,
            hf_config,
            substate(state_dict, "router"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "router"),
            weight_dtype=router_weight_dtype,
            prefill_input_l1=router_prefill_input_l1,
            prefill_explicit_program_config=router_prefill_explicit_program_config,
        )
        # Geometry is a policy because the legal subblock width depends on the
        # chosen core count.  Decode uses 45-core gate/up and 15-core down on
        # both meshes.  Prefill uses 45-core down on both meshes; measuring
        # 15-, 30-, and 45-core candidates through the full layer selected it.
        down_subblock_w = (
            _DOWN_SUBBLOCK_WIDTH_BY_TP[mesh_config.decode.tp] if down_subblock_w is None else down_subblock_w
        )
        prefill_down_subblock_w = (
            _DOWN_SUBBLOCK_WIDTH_BY_TP[mesh_config.prefill.tp]
            if prefill_down_subblock_w is None
            else prefill_down_subblock_w
        )
        self.experts.program_config = GPTOSSProgramConfig(
            decode_gate_up_cores=gate_up_cores,
            decode_gate_up_in0_block_w=gate_up_in0_block_w,
            decode_gate_up_subblock_w=gate_up_subblock_w,
            decode_down_cores=down_cores,
            decode_down_in0_block_w=down_in0_block_w,
            decode_down_subblock_w=down_subblock_w,
            prefill_gate_up_cores=gate_up_cores,
            prefill_gate_up_in0_block_w=gate_up_in0_block_w,
            prefill_gate_up_subblock_w=gate_up_subblock_w,
            prefill_down_cores=prefill_down_cores,
            prefill_down_in0_block_w=prefill_down_in0_block_w,
            prefill_down_subblock_w=prefill_down_subblock_w,
        )
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = int(hf_config.hidden_size)
        self.intermediate_size = int(hf_config.intermediate_size)
        self.local_intermediate_size = self.intermediate_size // mesh_config.decode.tp
        self.num_experts = int(hf_config.num_local_experts)
        self.top_k = int(hf_config.num_experts_per_tok)
        self.expert_weight_dtype = expert_weight_dtype
        self.activation_ccl_dtype = activation_ccl_dtype
        self.separate_gate_up = separate_gate_up
        self._load_indexed_decode_weights(
            substate(state_dict, "experts"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "indexed_decode"),
        )
        for tensor in (
            self.experts.weights.gate_proj,
            self.experts.weights.up_proj,
            self.experts.weights.down_proj,
            self.experts.weights.gate_proj_bias,
            self.experts.weights.up_proj_bias,
            self.experts.weights.down_proj_bias,
        ):
            tensor.deallocate(True)
        self.experts.weights = None
        self.decode_uses_gate_selected_sparse_experts = True

    def _load_indexed_decode_weights(self, expert_state, *, tensor_cache_path):
        """Load a compact top-k decode representation with TP-sharded weights."""
        tp = self.mesh_config.decode.tp
        local = self.local_intermediate_size
        gate = expert_state["gate_up_proj"][..., ::2].reshape(
            1, self.num_experts, self.hidden_size, self.intermediate_size
        )
        up = expert_state["gate_up_proj"][..., 1::2].reshape(
            1, self.num_experts, self.hidden_size, self.intermediate_size
        )
        gate_bias = expert_state["gate_up_proj_bias"][..., ::2].reshape(self.num_experts, self.intermediate_size)
        up_bias = expert_state["gate_up_proj_bias"][..., 1::2].reshape(self.num_experts, self.intermediate_size)

        # Arrange [gate_rank, up_rank] chunks consecutively.  Sharding the
        # resulting last dimension then gives every rank both operands for its
        # local SwiGLU instead of assigning whole gate/up halves to ranks.
        packed_gate_up = torch.cat(
            [
                torch.cat(
                    (
                        gate[..., rank * local : (rank + 1) * local],
                        up[..., rank * local : (rank + 1) * local],
                    ),
                    dim=-1,
                )
                for rank in range(tp)
            ],
            dim=-1,
        )
        packed_gate_up_bias = torch.cat(
            [
                torch.cat(
                    (
                        gate_bias[..., rank * local : (rank + 1) * local],
                        up_bias[..., rank * local : (rank + 1) * local],
                    ),
                    dim=-1,
                )
                for rank in range(tp)
            ],
            dim=-1,
        )
        column_mapper = self.mesh_config.column_parallel(self.mesh_device)
        row_mapper = self.mesh_config.row_parallel(self.mesh_device)
        if self.separate_gate_up:
            self.indexed_gate = ttnn.as_tensor(
                gate,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.expert_weight_dtype,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "separate_gate"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.indexed_up = ttnn.as_tensor(
                up,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.expert_weight_dtype,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "separate_up"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.indexed_gate_bias = ttnn.as_tensor(
                gate_bias,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "separate_gate_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.indexed_up_bias = ttnn.as_tensor(
                up_bias,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "separate_up_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.prefill_gate_bias = ttnn.as_tensor(
                gate_bias.unsqueeze(0),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "prefill_separate_gate_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.prefill_up_bias = ttnn.as_tensor(
                up_bias.unsqueeze(0),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "prefill_separate_up_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.indexed_gate_up = None
            self.indexed_gate_up_bias = None
            self.prefill_gate_up_bias = None
        else:
            self.indexed_gate_up = ttnn.as_tensor(
                packed_gate_up,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.expert_weight_dtype,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "packed_gate_up"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.indexed_gate_up_bias = ttnn.as_tensor(
                packed_gate_up_bias,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "packed_gate_up_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.prefill_gate_up_bias = ttnn.as_tensor(
                packed_gate_up_bias.unsqueeze(0),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "prefill_packed_gate_up_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        self.indexed_down = ttnn.as_tensor(
            expert_state["down_proj"].reshape(
                1,
                self.num_experts,
                self.intermediate_size,
                self.hidden_size,
            ),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.expert_weight_dtype,
            mesh_mapper=row_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "down"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        down_bias = expert_state["down_proj_bias"].reshape(self.num_experts, self.hidden_size)
        down_bias = torch.cat([down_bias] + [torch.zeros_like(down_bias)] * (tp - 1), dim=-1)
        self.indexed_down_bias = ttnn.as_tensor(
            down_bias,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=column_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "down_bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.prefill_down_bias = ttnn.as_tensor(
            down_bias.unsqueeze(0),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=column_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "prefill_down_bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.indexed_unused_sparsity = ttnn.as_tensor(
            torch.zeros((1, 1, 1, self.num_experts), dtype=torch.bfloat16),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=get_cache_file_name(tensor_cache_path, "unused_sparsity"),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _run_indexed_decode(self, hidden_states):
        """Run gate-selected top-4 TP experts without materializing 128 outputs."""
        expert_indices, routing_scores = self.router(hidden_states, True)
        expert_indices_rm = ttnn.to_layout(expert_indices, ttnn.ROW_MAJOR_LAYOUT)
        expert_indices.deallocate(True)
        expert_indices_rm = ttnn.reshape(expert_indices_rm, (1, 1, 1, self.top_k))
        embedding_indices = ttnn.typecast(expert_indices_rm, ttnn.uint32)
        output_tile = ttnn.Tile([32, 32])

        if self.separate_gate_up:
            projections = []
            for weight, bias in (
                (self.indexed_gate, self.indexed_gate_bias),
                (self.indexed_up, self.indexed_up_bias),
            ):
                projected = ttnn.sparse_matmul(
                    hidden_states,
                    weight,
                    sparsity=self.indexed_unused_sparsity,
                    indices=expert_indices_rm,
                    is_input_b_sparse=True,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    output_tile=output_tile,
                    program_config=self.experts.program_config.get_decode_gate_up_config(
                        hidden_states.shape[2],
                        weight.shape[3],
                        k=hidden_states.shape[-1],
                    ),
                    dtype=self.activation_ccl_dtype,
                )
                projected = ttnn.reshape(projected, (1, self.top_k, self.local_intermediate_size))
                projected_bias = ttnn.embedding(
                    embedding_indices,
                    bias,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                projected = ttnn.add(projected, projected_bias, output_tensor=projected)
                projected_bias.deallocate(True)
                projections.append(projected)
            gate, up = projections
        else:
            gate_up = ttnn.sparse_matmul(
                hidden_states,
                self.indexed_gate_up,
                sparsity=self.indexed_unused_sparsity,
                indices=expert_indices_rm,
                is_input_b_sparse=True,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=self.experts.program_config.get_decode_gate_up_config(
                    hidden_states.shape[2],
                    self.indexed_gate_up.shape[3],
                    k=hidden_states.shape[-1],
                ),
                dtype=self.activation_ccl_dtype,
            )
            gate_up = ttnn.reshape(gate_up, (1, self.top_k, 2 * self.local_intermediate_size))
            gate_up_bias = ttnn.embedding(
                embedding_indices,
                self.indexed_gate_up_bias,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            gate_up = ttnn.add(gate_up, gate_up_bias, output_tensor=gate_up)
            gate_up_bias.deallocate(True)
            if gate_up.dtype == ttnn.bfloat4_b:
                converted = ttnn.typecast(gate_up, ttnn.bfloat16)
                gate_up.deallocate(True)
                gate_up = converted
            gate = ttnn.slice(
                gate_up,
                [0, 0, 0],
                [1, self.top_k, self.local_intermediate_size],
                [1, 1, 1],
            )
            up = ttnn.slice(
                gate_up,
                [0, 0, self.local_intermediate_size],
                [1, self.top_k, 2 * self.local_intermediate_size],
                [1, 1, 1],
            )
            gate_up.deallocate(True)
        down_input = apply_swiglu(gate, up, self.experts.config)
        down_input = ttnn.reshape(down_input, (1, self.top_k, 1, self.local_intermediate_size))
        down = ttnn.sparse_matmul(
            down_input,
            self.indexed_down,
            sparsity=self.indexed_unused_sparsity,
            indices=expert_indices_rm,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=self.experts.program_config.get_decode_down_config(
                down_input.shape[2],
                self.indexed_down.shape[-1],
                k=down_input.shape[-1],
            ),
            dtype=self.activation_ccl_dtype,
        )
        down_input.deallocate(True)
        expert_indices_rm.deallocate(True)
        output = ttnn.reshape(down, (1, self.top_k, self.hidden_size))
        down_bias = ttnn.embedding(
            embedding_indices,
            self.indexed_down_bias,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        embedding_indices.deallocate(True)
        output = ttnn.add(output, down_bias, output_tensor=output)
        down_bias.deallocate(True)
        routing_scores_rm = ttnn.to_layout(routing_scores, ttnn.ROW_MAJOR_LAYOUT)
        routing_scores.deallocate(True)
        routing_scores_rm = ttnn.reshape(routing_scores_rm, (1, self.top_k, 1))
        output = ttnn.mul(output, routing_scores_rm, output_tensor=output)
        routing_scores_rm.deallocate(True)
        if output.dtype == ttnn.bfloat4_b:
            converted = ttnn.typecast(output, ttnn.bfloat16)
            output.deallocate(True)
            output = converted
        output = ttnn.sum(output, dim=1)
        output = ttnn.unsqueeze_to_4D(output)
        output = ttnn.unsqueeze_to_4D(output)
        output = apply_tensor_parallel_allreduce(
            output,
            self.mesh_config,
            self.mesh_device,
            1,
            self.ccl_manager,
        )
        return ttnn.reshape(
            output,
            (1, 1, 1, self.hidden_size),
            (1, 1, ttnn.TILE_SIZE, self.hidden_size),
        )

    def _process_packed_prefill_chunk(self, hidden_states, routing_weights):
        """Run one tile-aligned prefill chunk through the shared packed weights."""
        _, batch_size, sequence_length, _ = hidden_states.shape
        if batch_size != 1 or sequence_length % ttnn.TILE_SIZE:
            raise ValueError("packed TP expert prefill requires batch 1 and tile-aligned internal chunks")
        groups = sequence_length // ttnn.TILE_SIZE
        hidden_4d = ttnn.reshape(hidden_states, (1, groups, ttnn.TILE_SIZE, self.hidden_size))
        sparsity = ttnn.repeat(self.experts.prefill_sparsity, (1, 1, groups, 1))
        output_tile = ttnn.Tile([32, 32])
        if self.separate_gate_up:
            projections = []
            for weight, bias in (
                (self.indexed_gate, self.prefill_gate_bias),
                (self.indexed_up, self.prefill_up_bias),
            ):
                projected = ttnn.sparse_matmul(
                    hidden_4d,
                    weight,
                    sparsity=sparsity,
                    nnz=self.num_experts * groups,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    output_tile=output_tile,
                    program_config=self.experts.program_config.get_prefill_gate_up_config(
                        hidden_4d.shape[2],
                        weight.shape[3],
                        k=hidden_4d.shape[-1],
                    ),
                    dtype=self.activation_ccl_dtype,
                )
                projected = ttnn.transpose(projected, 1, 3)
                projected = ttnn.reshape(
                    projected,
                    (batch_size, self.num_experts, sequence_length, self.local_intermediate_size),
                )
                projected_bias = ttnn.transpose(bias, 1, 0)
                projected = ttnn.add(projected, projected_bias, output_tensor=projected)
                projections.append(projected)
            gate, up = projections
        else:
            gate_up = ttnn.sparse_matmul(
                hidden_4d,
                self.indexed_gate_up,
                sparsity=sparsity,
                nnz=self.num_experts * groups,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=self.experts.program_config.get_prefill_gate_up_config(
                    hidden_4d.shape[2],
                    self.indexed_gate_up.shape[3],
                    k=hidden_4d.shape[-1],
                ),
                dtype=self.activation_ccl_dtype,
            )
            gate_up = ttnn.transpose(gate_up, 1, 3)
            gate_up = ttnn.reshape(
                gate_up,
                (batch_size, self.num_experts, sequence_length, 2 * self.local_intermediate_size),
            )
            gate_up_bias = ttnn.transpose(self.prefill_gate_up_bias, 1, 0)
            gate_up = ttnn.add(gate_up, gate_up_bias, output_tensor=gate_up)
            # TP4 owns 720 intermediate elements per rank.  That logical split is
            # intentionally not tile aligned, and slice's internal untilize cannot
            # produce a row-major BFP4 tensor.  Keep the public/logical shape and
            # pay the explicit conversion in the BFP4 experiment instead of
            # rejecting the lower-precision family at its first API boundary.
            if gate_up.dtype == ttnn.bfloat4_b:
                converted = ttnn.typecast(gate_up, ttnn.bfloat16)
                gate_up.deallocate(True)
                gate_up = converted
            gate = ttnn.slice(
                gate_up,
                [0, 0, 0, 0],
                [batch_size, self.num_experts, sequence_length, self.local_intermediate_size],
                [1, 1, 1, 1],
            )
            up = ttnn.slice(
                gate_up,
                [0, 0, 0, self.local_intermediate_size],
                [batch_size, self.num_experts, sequence_length, 2 * self.local_intermediate_size],
                [1, 1, 1, 1],
            )
            gate_up.deallocate(True)
        down_input = apply_swiglu(gate, up, self.experts.config)
        down_input = ttnn.reshape(
            down_input,
            (1, self.num_experts, sequence_length, self.local_intermediate_size),
        )

        prefill_sparsity_2d = ttnn.reshape(self.experts.prefill_sparsity, (1, self.num_experts))
        routing_weights = ttnn.mul(routing_weights, prefill_sparsity_2d, output_tensor=routing_weights)
        routing_weights = ttnn.permute(routing_weights, (1, 0))
        routing_weights = ttnn.reshape(routing_weights, (batch_size, self.num_experts, sequence_length, 1))
        split_size = self.experts.program_config.get_down_split_size(sequence_length)
        if sequence_length > split_size:
            down_inputs = ttnn.split(down_input, split_size, dim=2)
            down_input.deallocate(True)
            routing_splits = ttnn.split(routing_weights, split_size, dim=2)
            routing_weights.deallocate(True)
        else:
            down_inputs = [down_input]
            routing_splits = [routing_weights]

        reduced_accumulator = None
        for down_input_split, routing_split in zip(down_inputs, routing_splits):
            down = ttnn.sparse_matmul(
                down_input_split,
                self.indexed_down,
                sparsity=self.experts.prefill_sparsity,
                nnz=self.num_experts,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                is_input_a_sparse=True,
                program_config=self.experts.program_config.get_prefill_down_config(
                    down_input_split.shape[2],
                    self.indexed_down.shape[-1],
                    k=down_input_split.shape[-1],
                ),
                dtype=self.activation_ccl_dtype,
            )
            split_sequence = down_input_split.shape[2]
            down_input_split.deallocate(True)
            next_states = ttnn.reshape(
                down,
                (batch_size, self.num_experts, split_sequence, self.hidden_size),
            )
            down_bias = ttnn.transpose(self.prefill_down_bias, 1, 0)
            next_states = ttnn.add(next_states, down_bias, output_tensor=next_states)
            next_states = apply_routing_weights(next_states, routing_split)
            routing_split.deallocate(True)
            if next_states.dtype == ttnn.bfloat4_b:
                converted = ttnn.typecast(next_states, ttnn.bfloat16)
                next_states.deallocate(True)
                next_states = converted
            reduced = reduce_experts(next_states)
            down.deallocate(True)
            if reduced_accumulator is None:
                reduced_accumulator = reduced
            else:
                concatenated = ttnn.concat((reduced_accumulator, reduced), dim=2)
                reduced_accumulator.deallocate(True)
                reduced.deallocate(True)
                reduced_accumulator = concatenated
        return reduced_accumulator

    def _run_packed_prefill(self, hidden_states, routing_weights):
        """Chunked full-context prefill using the decode-shared packed TP weights."""
        sequence_length = hidden_states.shape[2]
        chunk_size = self.experts.program_config.sequence_chunk_size
        if sequence_length > chunk_size:
            hidden_chunks = ttnn.split(hidden_states, chunk_size, dim=2)
            routing_chunks = ttnn.split(routing_weights, chunk_size, dim=0)
        else:
            hidden_chunks = [hidden_states]
            routing_chunks = [routing_weights]

        output_accumulator = None
        for hidden_chunk, routing_chunk in zip(hidden_chunks, routing_chunks):
            output = self._process_packed_prefill_chunk(hidden_chunk, routing_chunk)
            if output_accumulator is None:
                output_accumulator = output
            else:
                concatenated = ttnn.concat((output_accumulator, output), dim=2)
                output_accumulator.deallocate(True)
                output.deallocate(True)
                output_accumulator = concatenated
        output = apply_tensor_parallel_allreduce(
            output_accumulator,
            self.mesh_config,
            self.mesh_device,
            sequence_length,
            self.ccl_manager,
        )
        return ttnn.reshape(
            output,
            (1, 1, sequence_length, self.hidden_size),
            (1, 1, max(ttnn.TILE_SIZE, sequence_length), self.hidden_size),
        )

    def _run_one(self, hidden_states, *, is_decode):
        if is_decode:
            return self._run_indexed_decode(hidden_states)
        expert_indices, expert_weights = self.router(hidden_states, False)
        output = self._run_packed_prefill(hidden_states, expert_weights)
        expert_indices.deallocate(True)
        return output

    def __call__(self, hidden_states, *, is_decode):
        if not is_decode or hidden_states.shape[-2] == 1:
            return self._run_one(hidden_states, is_decode=is_decode)

        # The reusable sparse expert decode kernel represents users as its
        # batch dimension and currently accepts B=1.  Keep the autoport's
        # [1,1,B,H] public contract by executing the same active-expert graph
        # once per logical user, then restore the stack layout.  The loop is
        # static for the captured decode shape and remains device-only.
        user_inputs = ttnn.split(hidden_states, 1, dim=2)
        outputs = [self._run_one(user_input, is_decode=True) for user_input in user_inputs]
        output = ttnn.concat(outputs, dim=2)
        for user_output in outputs:
            user_output.deallocate(True)
        return output


class MultichipDecoder(LightweightModule):
    """Uniform P150/P150x2/P150x4 decoder wrapper."""

    optimization_manifest = OptimizedDecoder.optimization_manifest + (
        "p150_1d_tensor_parallel",
        "packed_qkv_column_parallel",
        "local_qkv_head_paged_cache",
        "row_parallel_output_ring_reduce",
        "tile_divisible_physical_hidden_attention_collective",
        "packed_rank_local_gate_up_and_row_parallel_down",
        "gate_selected_sparse_expert_tensor_parallel",
        "borrowed_l1_decode_residual",
        "decode_sharded_rmsnorm",
        "decode_attention_bfp8_prefill_attention_bf16_expert_bf16_collectives",
        "decode_lofi_prefill_qkv_hifi2_o_lofi_attention_projections",
        "sparse_expert_decode_45x15_prefill_45x45_geometry_with_tp2_gate_subblock2",
        "tp2_dram_sharded_output_projection",
        "replicated_decode_l1_prefill_dram_stack_residual_contract",
    )

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        max_batch_size=1,
        max_context_length=None,
        page_size=64,
        tensor_cache_path=None,
        calibrated_checkpoint_revision=None,
        policy: MultichipDecoderPolicy = DEFAULT_MULTICHIP_POLICY,
        optimized_policy: OptimizedDecoderPolicy | None = None,
    ):
        plan = tensor_plan(mesh_device.shape, hf_config)
        if policy not in _SUPPORTED_MULTICHIP_POLICIES:
            raise ValueError(f"unsupported multichip decoder policy: {policy!r}")
        if not 1 <= int(max_batch_size) <= 32:
            raise ValueError(f"max_batch_size must be within [1, 32], got {max_batch_size}")
        layer_type = hf_config.layer_types[layer_idx]
        if layer_type not in _SUPPORTED_LAYER_TYPES:
            raise ValueError(f"Unsupported GPT-OSS layer type {layer_type!r}")
        advertised_context = int(hf_config.max_position_embeddings)
        max_context_length = advertised_context if max_context_length is None else int(max_context_length)
        if not 1 <= max_context_length <= advertised_context:
            raise ValueError(f"max_context_length must be within [1, {advertised_context}], got {max_context_length}")
        if page_size <= 0 or page_size % ttnn.TILE_SIZE:
            raise ValueError(f"page_size must be a positive tile multiple, got {page_size}")

        if plan.tp == 1:
            backend = OptimizedDecoder.from_state_dict(
                state_dict,
                hf_config=hf_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                max_batch_size=max_batch_size,
                max_context_length=max_context_length,
                page_size=page_size,
                tensor_cache_path=tensor_cache_path,
                calibrated_checkpoint_revision=calibrated_checkpoint_revision,
                policy=optimized_policy,
            )
            return cls(backend=backend, tensor_plan=plan, policy=policy, single_chip_policy=backend.policy)

        if optimized_policy is not None:
            raise ValueError("optimized_policy configures only the exact TP=1 OptimizedDecoder baseline")

        local_state = _local_layer_state_dict(state_dict, layer_idx)
        cache_root = str(Path(tensor_cache_path) / f"tp{plan.tp}") if tensor_cache_path is not None else None
        mesh_config = MeshConfig(
            mesh_device.shape,
            decode=ModeConfig(tp=plan.tp, ep=1, sp=1),
            prefill=ModeConfig(tp=plan.tp, ep=1, sp=1),
        )
        ccl_manager = CCLManager(
            mesh_device,
            num_links=get_default_num_links(mesh_device),
            topology=policy.topology,
        )
        paged_attention_config = PagedAttentionConfig(
            block_size=page_size,
            max_num_blocks=max_batch_size * math.ceil(max_context_length / page_size),
        )
        attention_config = AttentionConfig(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            sliding_window=hf_config.sliding_window if layer_type == "sliding_attention" else None,
            max_seq_len=max_context_length,
            max_local_batch_size=max_batch_size,
            users_row_sharded=False,
        )
        rope_scaling_config = getattr(hf_config, "rope_scaling", None)
        rope_scaling = rope_scaling_model_factory(rope_scaling_config) if rope_scaling_config else None
        rope_theta = getattr(hf_config, "rope_theta", None) or getattr(hf_config, "default_theta", 150000.0)
        rope_setup = RotarySetup(
            device=mesh_device,
            batch_size=max_batch_size,
            head_dim=hf_config.head_dim,
            max_seq_len=max_context_length,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            datatype=ttnn.bfloat16,
        )
        attention_state = substate(local_state, "self_attn")
        attention = _PhysicalHiddenCollectiveAttention(
            mesh_device=mesh_device,
            config=attention_config,
            state_dict=attention_state,
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=GPTOSSAttentionProgramConfig(),
            layer_idx=layer_idx,
            paged_attention_config=paged_attention_config,
            transformation_mats=rope_setup.get_both_trans_mats(),
            weight_dtype=policy.attention_weight_dtype,
            tensor_cache_path=get_cache_file_name(cache_root, "self_attn"),
        )
        attention.decode_wqkv = attention.weights.wqkv
        attention.decode_separate_qkv = policy.decode_separate_qkv
        attention.decode_separate_qkv_weights = None
        attention.decode_separate_qkv_biases = None
        attention.decode_qkv_input_memory_config = None
        attention.decode_qkv_program_config = None
        attention.decode_qkv_to_interleaved = False
        attention.decode_output_input_memory_config = None
        attention.decode_output_program_config = None
        attention.decode_o_proj = attention.weights.o_proj
        attention.decode_o_proj_bias = attention.weights.o_proj_bias
        attention.decode_output_to_interleaved = False
        attention.decode_output_physical_hidden = plan.padded_hidden_size
        attention.activation_ccl_dtype = policy.attention_activation_ccl_dtype or policy.activation_ccl_dtype
        # The regular fused MM+RS kernel is correct and substantially faster
        # for TP4. Its TP2 topology hangs on Blackhole for both native and
        # 3072-column adapted shapes, so TP2 keeps the measured non-fused path.
        attention.decode_fused_output_projection_ccl = policy.decode_fused_output_projection_ccl and plan.tp == 4
        attention.decode_output_mmrs_intermediate = None
        attention.decode_output_mmrs_output = None
        attention.decode_output_mmrs_program_config = None
        attention.decode_projection_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.projection_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        if attention.decode_fused_output_projection_ccl:
            persistent_batch = ttnn.TILE_SIZE
            make_persistent = lambda width: ttnn.from_torch(
                torch.zeros((1, 1, persistent_batch, width)),
                device=mesh_device,
                dtype=attention.activation_ccl_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            attention.decode_output_mmrs_intermediate = make_persistent(plan.padded_hidden_size)
            attention.decode_output_mmrs_output = make_persistent(plan.padded_local_hidden)
            output_tiles = plan.padded_hidden_size // ttnn.TILE_SIZE
            per_core_n = math.ceil(output_tiles / 8)
            attention.decode_output_mmrs_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 6),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=per_core_n,
                out_block_w=max(1, per_core_n // 2),
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
        if policy.decode_separate_qkv:
            column_mapper = mesh_config.column_parallel(mesh_device)
            attention.decode_separate_qkv_weights = tuple(
                ttnn.as_tensor(
                    substate(attention_state, projection)["weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0),
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=policy.attention_weight_dtype,
                    mesh_mapper=column_mapper,
                    cache_file_name=get_cache_file_name(cache_root, f"self_attn/{projection}_separate_weight"),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for projection in ("q_proj", "k_proj", "v_proj")
            )
            attention.decode_separate_qkv_biases = tuple(
                ttnn.as_tensor(
                    substate(attention_state, projection)["bias"],
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=column_mapper,
                    cache_file_name=get_cache_file_name(cache_root, f"self_attn/{projection}_separate_bias"),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for projection in ("q_proj", "k_proj", "v_proj")
            )
        if policy.decode_dram_sharded_qkv:
            dram_grid_size = mesh_device.dram_grid_size()
            dram_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
                    )
                }
            )
            dram_banks = dram_grid.num_cores()
            if plan.local_qkv_width % (dram_banks * ttnn.TILE_SIZE):
                raise ValueError(
                    "DRAM-sharded QKV requires a tile-aligned per-bank width, "
                    f"got local width {plan.local_qkv_width} over {dram_banks} banks"
                )
            weight_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (hf_config.hidden_size, plan.local_qkv_width // dram_banks),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            attention.decode_wqkv = ttnn.to_memory_config(attention.weights.wqkv, weight_memory_config)
            input_cores = ttnn.num_cores_to_corerangeset(
                15,
                mesh_device.compute_with_storage_grid_size(),
                row_wise=True,
            )
            attention.decode_qkv_input_memory_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, hf_config.hidden_size // 15),
                core_grid=input_cores,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            attention.decode_qkv_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=6,
                per_core_M=1,
                per_core_N=5,
                fused_activation=None,
            )
            attention.decode_qkv_to_interleaved = True
        if policy.decode_dram_sharded_output and (plan.tp == 2 or policy.decode_dram_sharded_output_tp4):
            dram_grid_size = mesh_device.dram_grid_size()
            dram_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
                    )
                }
            )
            dram_banks = dram_grid.num_cores()
            output_alignment = dram_banks * ttnn.TILE_SIZE
            output_width = math.ceil(plan.padded_hidden_size / output_alignment) * output_alignment
            local_attention_width = hf_config.num_attention_heads * hf_config.head_dim // plan.tp
            output_weight = substate(attention_state, "o_proj")["weight"].transpose(-1, -2)
            output_bias = substate(attention_state, "o_proj")["bias"]
            output_weight = torch.nn.functional.pad(
                output_weight,
                (0, output_width - hf_config.hidden_size),
                "constant",
                value=0.0,
            )
            output_bias = torch.nn.functional.pad(
                output_bias,
                (0, output_width - hf_config.hidden_size),
                "constant",
                value=0.0,
            )
            output_bias = torch.cat([output_bias] + [torch.zeros_like(output_bias)] * (plan.tp - 1), dim=-1)
            output_weight_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (local_attention_width, output_width // dram_banks),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            attention.decode_o_proj = ttnn.as_tensor(
                output_weight,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=policy.attention_weight_dtype,
                mesh_mapper=mesh_config.row_parallel(mesh_device),
                cache_file_name=get_cache_file_name(
                    cache_root,
                    f"self_attn/decode_o_proj_dram_sharded_{output_width}",
                ),
                memory_config=output_weight_memory_config,
            )
            attention.decode_o_proj_bias = ttnn.as_tensor(
                output_bias,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=mesh_config.column_parallel(mesh_device),
                cache_file_name=get_cache_file_name(
                    cache_root,
                    f"self_attn/decode_o_proj_bias_dram_sharded_{output_width}",
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            input_cores = policy.decode_dram_sharded_output_input_cores
            output_input_grid = ttnn.num_cores_to_corerangeset(
                input_cores,
                mesh_device.compute_with_storage_grid_size(),
                row_wise=True,
            )
            input_shard_width = local_attention_width // input_cores
            attention.decode_output_input_memory_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, input_shard_width),
                core_grid=output_input_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            attention.decode_output_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=input_shard_width // ttnn.TILE_SIZE,
                per_core_M=1,
                per_core_N=output_width // dram_banks // ttnn.TILE_SIZE,
                fused_activation=None,
            )
            attention.decode_output_to_interleaved = True
            attention.decode_output_physical_hidden = output_width
        if policy.decode_explicit_output_projection:
            output_cores = 32 if plan.tp == 2 else 16
            output_grid = ttnn.CoreGrid(x=8, y=output_cores // 8)
            local_attention_width = hf_config.num_attention_heads * hf_config.head_dim // plan.tp
            attention.decode_output_input_memory_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, local_attention_width // output_cores),
                core_grid=output_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            per_core_n = 3 if plan.tp == 2 else 6
            attention.decode_output_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(output_grid.x, output_grid.y),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=per_core_n,
                out_block_h=1,
                out_block_w=per_core_n,
                per_core_M=1,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        backend = _OptimizedMultichipBackend(
            mesh_device=mesh_device,
            hf_config=hf_config,
            layer_idx=layer_idx,
            layer_type=layer_type,
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            page_size=page_size,
            input_layernorm=_DecodeShardedRMSNorm(
                mesh_device,
                hf_config,
                substate(local_state, "input_layernorm"),
                tensor_cache_path=get_cache_file_name(cache_root, "input_layernorm"),
                mesh_config=mesh_config,
                enable_decode_sharding=max_batch_size < ttnn.TILE_SIZE,
            ),
            post_attention_layernorm=_DecodeShardedRMSNorm(
                mesh_device,
                hf_config,
                substate(local_state, "post_attention_layernorm"),
                tensor_cache_path=get_cache_file_name(cache_root, "post_attention_layernorm"),
                mesh_config=mesh_config,
                enable_decode_sharding=max_batch_size < ttnn.TILE_SIZE,
            ),
            attention=attention,
            mlp=_ActiveExpertTPMLP(
                mesh_device,
                hf_config,
                substate(local_state, "mlp"),
                ccl_manager,
                tensor_cache_path=get_cache_file_name(cache_root, "mlp"),
                mesh_config=mesh_config,
                expert_weight_dtype=policy.expert_weight_dtype,
                router_weight_dtype=policy.router_weight_dtype,
                router_prefill_input_l1=policy.router_prefill_input_l1,
                router_prefill_explicit_program_config=policy.router_prefill_explicit_program_config,
                activation_ccl_dtype=policy.expert_activation_ccl_dtype or policy.activation_ccl_dtype,
                separate_gate_up=policy.decode_separate_gate_up,
                gate_up_cores=policy.expert_gate_up_cores,
                gate_up_in0_block_w=policy.expert_gate_up_in0_block_w,
                gate_up_subblock_w=(
                    policy.expert_gate_up_subblock_w_tp2
                    if plan.tp == 2 and policy.expert_gate_up_subblock_w_tp2 is not None
                    else policy.expert_gate_up_subblock_w
                ),
                down_cores=policy.expert_down_cores,
                down_in0_block_w=policy.expert_down_in0_block_w,
                down_subblock_w=policy.expert_down_subblock_w,
                prefill_down_cores=(
                    policy.expert_prefill_down_cores_tp2
                    if plan.tp == 2 and policy.expert_prefill_down_cores_tp2 is not None
                    else policy.expert_prefill_down_cores
                ),
                prefill_down_in0_block_w=policy.expert_prefill_down_in0_block_w,
                prefill_down_subblock_w=(
                    policy.expert_prefill_down_subblock_w_tp2
                    if plan.tp == 2 and policy.expert_prefill_down_subblock_w_tp2 is not None
                    else policy.expert_prefill_down_subblock_w
                ),
            ),
            calibrated_checkpoint_revision=calibrated_checkpoint_revision,
        )
        backend.mesh_config = mesh_config
        backend.ccl_manager = ccl_manager
        return cls(backend=backend, tensor_plan=plan, policy=policy, single_chip_policy=None)

    def __init__(self, *, backend, tensor_plan, policy, single_chip_policy):
        self.backend = backend
        self.tensor_plan = tensor_plan
        self.policy = policy
        self.single_chip_policy = single_chip_policy
        self.mesh_device = backend.mesh_device
        self.hf_config = backend.hf_config
        self.layer_idx = backend.layer_idx
        self.layer_type = backend.layer_type
        self.max_batch_size = backend.max_batch_size
        self.max_context_length = backend.max_context_length
        self.page_size = backend.page_size
        self.is_single_chip_baseline = isinstance(backend, OptimizedDecoder)

    @property
    def kv_cache(self):
        return self.backend.kv_cache

    @property
    def input_layernorm(self):
        return self.backend.input_layernorm

    @property
    def post_attention_layernorm(self):
        return self.backend.post_attention_layernorm

    @property
    def self_attn(self):
        return self.backend.self_attn

    @property
    def mlp(self):
        return self.backend.mlp

    def prefill_forward(self, hidden_states, **kwargs):
        return self.backend.prefill_forward(hidden_states, **kwargs)

    def decode_forward(self, hidden_states, **kwargs):
        return self.backend.decode_forward(hidden_states, **kwargs)

    def forward(self, hidden_states, *, mode, **kwargs):
        return self.backend.forward(hidden_states, mode=mode, **kwargs)


__all__ = [
    "DEFAULT_MULTICHIP_POLICY",
    "DEFAULT_OPTIMIZED_POLICY",
    "BF16_ACTIVATION_CCL_MULTICHIP_POLICY",
    "BFP4_ACTIVATION_CCL_MULTICHIP_POLICY",
    "DRAM_SHARDED_QKV_MULTICHIP_POLICY",
    "EXPLICIT_OUTPUT_PROJECTION_MULTICHIP_POLICY",
    "ROUTER_BFP4_MULTICHIP_POLICY",
    "ROUTER_BFP8_MULTICHIP_POLICY",
    "MultichipDecoder",
    "MultichipDecoderPolicy",
    "MultichipTensorPlan",
    "SUPPORTED_MESH_SHAPES",
    "tensor_plan",
]

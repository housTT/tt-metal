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
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.autoports.openai_gpt_oss_120b.tt.fused_decoder import FusedDecoder, _local_layer_state_dict
from models.autoports.openai_gpt_oss_120b.tt.optimized_decoder import (
    DEFAULT_OPTIMIZED_POLICY,
    OptimizedDecoder,
    OptimizedDecoderPolicy,
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
from models.demos.gpt_oss.tt.rms_norm import RMSNorm
from models.demos.gpt_oss.tt.topk import TopKRouter
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

    name: str = "p150_1d_tp_replicated_residual"
    attention_weight_dtype: object = ttnn.bfloat8_b
    expert_weight_dtype: object = ttnn.bfloat4_b
    kv_cache_dtype: object = ttnn.bfloat8_b
    residual_layout: str = "replicated"
    topology: object = ttnn.Topology.Ring


DEFAULT_MULTICHIP_POLICY = MultichipDecoderPolicy()


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

        xqkv_fused = ttnn.matmul(
            hidden_states,
            self.weights.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
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
        tt_out = ttnn.linear(
            tt_sdpa_out,
            self.weights.o_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        tt_sdpa_out.deallocate(True)
        tt_out = ttnn.add(tt_out, self.weights.o_proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        tt_out = ttnn.typecast(tt_out, ttnn.bfloat8_b)

        padded_local_hidden = math.ceil((hidden_size // self.mesh_config.tp) / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        padded_hidden = padded_local_hidden * self.mesh_config.tp
        tt_out = ttnn.reshape(
            tt_out,
            (1, 1, batch_size, padded_hidden),
            (1, 1, ttnn.TILE_SIZE, padded_hidden),
        )
        return _allreduce_physical_hidden(
            tt_out,
            hidden_size=hidden_size,
            padded_hidden_size=padded_hidden,
            mesh_config=self.mesh_config,
            ccl_manager=self.ccl_manager,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )


class _ReplicatedL1Router(TopKRouter):
    """Replicated BF16 router with setup-time L1-resident weights."""

    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None):
        self.top_k = hf_config.num_experts_per_tok
        self.num_experts = hf_config.num_local_experts
        self.hidden_dim = hf_config.hidden_size
        self.tensor_cache_path = tensor_cache_path
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        self.weight = ttnn.as_tensor(
            state_dict["weight"].transpose(0, 1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
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
        )
        # Wider gate/up subblocks failed the BFP4/LoFi PCC gate.  Keep their
        # accumulation order.  The 3-tile down subblock is validated on TP4;
        # TP2 keeps the one-tile accumulation order of the optimized baseline.
        down_subblock_w = _DOWN_SUBBLOCK_WIDTH_BY_TP[mesh_config.decode.tp]
        self.experts.program_config = GPTOSSProgramConfig(
            decode_gate_up_subblock_w=1,
            decode_down_subblock_w=down_subblock_w,
            prefill_gate_up_subblock_w=1,
            prefill_down_subblock_w=down_subblock_w,
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
            dtype=ttnn.bfloat8_b,
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
            dtype=ttnn.bfloat8_b,
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
            dtype=ttnn.bfloat8_b,
        )
        gate_up = ttnn.transpose(gate_up, 1, 3)
        gate_up = ttnn.reshape(
            gate_up,
            (batch_size, self.num_experts, sequence_length, 2 * self.local_intermediate_size),
        )
        gate_up_bias = ttnn.transpose(self.prefill_gate_up_bias, 1, 0)
        gate_up = ttnn.add(gate_up, gate_up_bias, output_tensor=gate_up)
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
                dtype=ttnn.bfloat8_b,
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
        "replicated_stack_residual_contract",
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
        if policy != DEFAULT_MULTICHIP_POLICY:
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
        attention_cls = _PhysicalHiddenCollectiveAttention if plan.padded_hidden_size != plan.hidden_size else Attention
        attention = attention_cls(
            mesh_device=mesh_device,
            config=attention_config,
            state_dict=substate(local_state, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=GPTOSSAttentionProgramConfig(),
            layer_idx=layer_idx,
            paged_attention_config=paged_attention_config,
            transformation_mats=rope_setup.get_both_trans_mats(),
            weight_dtype=policy.attention_weight_dtype,
            tensor_cache_path=get_cache_file_name(cache_root, "self_attn"),
        )
        backend = FusedDecoder(
            mesh_device=mesh_device,
            hf_config=hf_config,
            layer_idx=layer_idx,
            layer_type=layer_type,
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            page_size=page_size,
            input_layernorm=RMSNorm(
                mesh_device,
                hf_config,
                substate(local_state, "input_layernorm"),
                tensor_cache_path=get_cache_file_name(cache_root, "input_layernorm"),
                mesh_config=mesh_config,
            ),
            post_attention_layernorm=RMSNorm(
                mesh_device,
                hf_config,
                substate(local_state, "post_attention_layernorm"),
                tensor_cache_path=get_cache_file_name(cache_root, "post_attention_layernorm"),
                mesh_config=mesh_config,
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
    "MultichipDecoder",
    "MultichipDecoderPolicy",
    "MultichipTensorPlan",
    "SUPPORTED_MESH_SHAPES",
    "tensor_plan",
]

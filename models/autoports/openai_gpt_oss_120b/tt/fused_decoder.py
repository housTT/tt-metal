# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Graph-fused, single-device GPT-OSS 120B decoder layer.

This stage preserves :mod:`functional_decoder`'s paged prefill/decode contract
while replacing its spelled-out prefill expert graph with Blackhole's dedicated
``unified_routed_expert_moe`` kernel.  On the single-device functional-stage
topology, a device sort plus integer gather/scatter maps locally regroup routed
slots without invoking the fabric-only DeepSeek dispatch/combine collectives.
Decode uses a real-checkpoint-calibrated FullLocal ``moe_compute`` allowlist and
compact indexed Top-K sparse matmuls everywhere else. Both paths remain
device-resident and trace-capture safe, with no functional-decoder fallback.

Logical prefill lengths remain unrestricted.  The public boundary pads to a
tile exactly as the functional stage does; the fused MoE privately pads its
token stream to the 64-row routing-kernel granularity and slices that padding
away before returning.
"""

from __future__ import annotations

from pathlib import Path

import torch
from ttnn.experimental.moe_compute_utils import (
    auto_output_width_shard_dim,
    effective_matmul_ring_size,
    get_weight_core_shard_maps,
    get_weight_mem_configs,
    prepare_w0_w1_tensor_with_bias,
    prepare_w2_tensor_with_bias,
)
from ttnn.operations.ccl import MoEActivationFunction

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.attention import Attention, AttentionConfig
from models.demos.gpt_oss.tt.attention_configs import GPTOSSAttentionProgramConfig
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig
from models.demos.gpt_oss.tt.experts import ExpertConfig
from models.demos.gpt_oss.tt.experts.operations import apply_swiglu
from models.demos.gpt_oss.tt.rms_norm import RMSNorm
from models.demos.gpt_oss.tt.topk import TopKRouter
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name, get_default_num_links
from models.demos.gpt_oss.utils.substate import substate
from models.tt_transformers.tt.common import PagedAttentionConfig, rope_scaling_model_factory
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format
from models.tt_transformers.tt.rope import RotarySetup

_SUPPORTED_LAYER_TYPES = {"sliding_attention", "full_attention"}
_ROUTING_GRANULARITY = 64
_EXPERT_CHUNK_SIZE = 4 * 1024
# FullLocal's mandatory BF4 expert weights are checkpoint-sensitive.  Every
# layer was measured against openai/gpt-oss-120b revision
# b5c939de8f754692c1647ca79fbf85e8c1e70f8a; only these layers cleared the
# >=0.995 exact-route Torch bar at logical decode batches 1 and 2 with the
# production host-quantized weights.  Marginal candidates were also checked
# with an independent hidden-state seed.
_FULL_LOCAL_CHECKPOINT_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
_FULL_LOCAL_DECODE_LAYERS = frozenset(
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 23, 24, 25, 26, 27, 29, 31, 33, 34, 35}
)
_FULL_LOCAL_CALIBRATED_RING_SIZE = 8
_FULL_LOCAL_MAX_BATCH_SIZE = 2
_FULL_LOCAL_REDUCE_OUTPUT_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG


def _use_full_local_decode(
    mesh_device,
    layer_idx: int,
    calibrated_checkpoint_revision: str | None,
    max_batch_size: int,
) -> bool:
    """Enable only the exact checkpoint, ring, and trace-qualified batch range."""
    return (
        calibrated_checkpoint_revision == _FULL_LOCAL_CHECKPOINT_REVISION
        and layer_idx in _FULL_LOCAL_DECODE_LAYERS
        and effective_matmul_ring_size(mesh_device) == _FULL_LOCAL_CALIBRATED_RING_SIZE
        and 1 <= max_batch_size <= _FULL_LOCAL_MAX_BATCH_SIZE
    )


def _local_layer_state_dict(state_dict, layer_idx: int):
    """Return Meta-RoPE-formatted keys local to one HF decoder layer."""
    prefix = f"model.layers.{layer_idx}."
    if any(key.startswith(prefix) for key in state_dict):
        local = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    else:
        local = dict(state_dict)
    required = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "self_attn.o_proj.bias",
        "self_attn.sinks",
        "mlp.router.weight",
        "mlp.router.bias",
        "mlp.experts.gate_up_proj",
        "mlp.experts.gate_up_proj_bias",
        "mlp.experts.down_proj",
        "mlp.experts.down_proj_bias",
    }
    missing = sorted(required - local.keys())
    if missing:
        raise KeyError(f"Layer {layer_idx} state_dict is missing required GPT-OSS weights: {missing}")
    return convert_hf_qkv_to_meta_format(local, head_dim=64)


def _expert_torch_lists(state_dict, num_experts: int):
    """Adapt HF's packed expert tensors to ``TtRoutedExpert``'s lists.

    This is setup-only host work.  Each returned weight uses the conventional
    ``[out_features, in_features]`` orientation expected by the loader.
    """
    gate_up = state_dict["gate_up_proj"]
    gate_up_bias = state_dict["gate_up_proj_bias"]
    down = state_dict["down_proj"]
    down_bias = state_dict["down_proj_bias"]
    if gate_up.shape[0] != num_experts:
        raise ValueError(f"Expected {num_experts} experts, got packed shape {tuple(gate_up.shape)}")

    weights = []
    biases = []
    for expert in range(num_experts):
        weights.append(
            {
                "gate_proj": gate_up[expert, :, ::2].transpose(0, 1).contiguous(),
                "up_proj": gate_up[expert, :, 1::2].transpose(0, 1).contiguous(),
                "down_proj": down[expert].transpose(0, 1).contiguous(),
            }
        )
        biases.append(
            {
                "gate_proj_bias": gate_up_bias[expert, ::2].contiguous(),
                "up_proj_bias": gate_up_bias[expert, 1::2].contiguous(),
                "down_proj_bias": down_bias[expert].contiguous(),
            }
        )
    return weights, biases


def _full_local_decode_weights(mesh_device, expert_state, hidden_size: int, intermediate_size: int):
    """Pack one GPT-OSS layer for the public FullLocal ``moe_compute`` path."""
    num_layers = 1
    num_experts = expert_state["gate_up_proj"].shape[0]
    gate_up = expert_state["gate_up_proj"]
    gate_up_bias = expert_state["gate_up_proj_bias"]
    w0_w1_shard_map, w2_shard_map, dram_core_range_set = get_weight_core_shard_maps(
        mesh_device,
        hidden_size,
        intermediate_size,
    )
    w0_w1_mem_config, w2_mem_config, _, _ = get_weight_mem_configs(
        num_layers,
        num_experts,
        hidden_size,
        intermediate_size,
        w0_w1_shard_map,
        w2_shard_map,
        dram_core_range_set,
        has_bias=True,
    )
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def upload_bf16(host_tensor, memory_config):
        # Direct large biased BF4 packing is unsupported.  Upload the packed
        # BF16 tensor, then use the public higher-accuracy host quantizer.  The
        # two weights are prepared and quantized serially to bound setup peak.
        return ttnn.from_torch(
            host_tensor,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            mesh_mapper=replicate,
        )

    def quantize_bf4(bf16, memory_config):
        try:
            return ttnn.experimental.quantize_weights_via_host(
                bf16,
                dtype=ttnn.bfloat4_b,
                memory_config=memory_config,
            )
        finally:
            bf16.deallocate(True)

    # Serialize the two host pack/upload phases: retaining both reordered BF16
    # tensors at once adds several GiB of avoidable setup peak memory.
    w0 = gate_up[..., ::2].unsqueeze(0).contiguous()
    w1 = gate_up[..., 1::2].unsqueeze(0).contiguous()
    b0 = gate_up_bias[..., ::2].unsqueeze(0).contiguous()
    b1 = gate_up_bias[..., 1::2].unsqueeze(0).contiguous()
    packed_w0_w1 = prepare_w0_w1_tensor_with_bias(
        w0,
        w1,
        b0,
        b1,
        num_layers,
        num_experts,
        hidden_size,
        intermediate_size,
        w0_w1_shard_map,
    )
    del w0, w1, b0, b1
    bf16_w0_w1 = upload_bf16(packed_w0_w1, w0_w1_mem_config)
    del packed_w0_w1
    tt_w0_w1 = quantize_bf4(bf16_w0_w1, w0_w1_mem_config)

    try:
        w2 = expert_state["down_proj"].unsqueeze(0).contiguous()
        b2 = expert_state["down_proj_bias"].unsqueeze(0).contiguous()
        packed_w2 = prepare_w2_tensor_with_bias(
            w2,
            b2,
            num_layers,
            num_experts,
            intermediate_size,
            hidden_size,
            w2_shard_map,
            w0_w1_shard_map,
        )
        del w2, b2
        bf16_w2 = upload_bf16(packed_w2, w2_mem_config)
        del packed_w2
        tt_w2 = quantize_bf4(bf16_w2, w2_mem_config)
    except Exception:
        tt_w0_w1.deallocate(True)
        raise
    return tt_w0_w1, tt_w2


class _FusedMLP:
    """Device-only router plus dedicated fused routed-expert pipeline."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        tensor_cache_path,
        *,
        layer_idx=0,
        calibrated_checkpoint_revision=None,
        max_batch_size=1,
    ):
        if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
            raise NotImplementedError("Fused GPT-OSS experts require Blackhole/P150 hardware")
        self.mesh_device = mesh_device
        self.hidden_size = hf_config.hidden_size
        self.num_experts = hf_config.num_local_experts
        self.top_k = hf_config.num_experts_per_tok
        self.intermediate_size = hf_config.intermediate_size
        self.layer_idx = layer_idx
        self.decode_uses_full_local = _use_full_local_decode(
            mesh_device,
            layer_idx,
            calibrated_checkpoint_revision,
            max_batch_size,
        )
        self.decode_full_local_checkpoint_revision = (
            _FULL_LOCAL_CHECKPOINT_REVISION if self.decode_uses_full_local else None
        )
        self.router = TopKRouter(
            mesh_device,
            hf_config,
            substate(state_dict, "router"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "router"),
        )
        # Decode is split per logical user below, but TILE_LAYOUT pads each B=1
        # input to 32 rows.  The router's fused-op selector uses physical volume
        # and would therefore select its B=32-only kernel.  The standard device
        # linear/top-k/softmax route returns the sparse K entries required by the
        # indexed expert matmuls and is also valid for prefill.
        self.router.use_fused_op = False

        # Decode uses a graph rewrite rather than the collective routed-expert
        # pipeline below.  The exact-revision allowlist uses FullLocal
        # moe_compute; all other layers retain the higher-precision indexed
        # sparse path.  Do not allocate both decode representations.
        expert_state = substate(state_dict, "experts")
        if self.decode_uses_full_local:
            self.decode_full_local_w0_w1, self.decode_full_local_w2 = _full_local_decode_weights(
                mesh_device,
                expert_state,
                self.hidden_size,
                self.intermediate_size,
            )
            self.decode_full_local_output_height_shard_dim = 4
            self.decode_full_local_reduce_output_memory_config = _FULL_LOCAL_REDUCE_OUTPUT_MEMORY_CONFIG
            self.decode_full_local_output_width_shard_dim = auto_output_width_shard_dim(
                self.hidden_size,
                matmul_ring_size=effective_matmul_ring_size(mesh_device),
            )
            drain = ttnn.experimental.get_moe_tilize_drain_core(
                mesh_device,
                self.decode_full_local_output_height_shard_dim,
                self.decode_full_local_output_width_shard_dim,
                self.hidden_size,
            )
            self.decode_full_local_drain_core = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(drain.x, drain.y), ttnn.CoreCoord(drain.x, drain.y))}
            )
            decode_full_local_expert_mapping = torch.zeros((1, self.num_experts), dtype=torch.uint16)
            self.decode_full_local_expert_mapping = ttnn.from_torch(
                decode_full_local_expert_mapping,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            self.decode_full_local_reduce_expert_mapping = ttnn.from_torch(
                decode_full_local_expert_mapping,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            self._decode_full_local_buffers_by_tokens = {}
        else:
            gate_up = expert_state["gate_up_proj"]
            gate_up_bias = expert_state["gate_up_proj_bias"]
            packed_gate_up = torch.cat((gate_up[..., ::2], gate_up[..., 1::2]), dim=-1).reshape(
                1,
                self.num_experts,
                self.hidden_size,
                2 * self.intermediate_size,
            )
            packed_gate_up_bias = torch.cat((gate_up_bias[..., ::2], gate_up_bias[..., 1::2]), dim=-1).reshape(
                1,
                self.num_experts,
                2 * self.intermediate_size,
            )
            replicate = ttnn.ReplicateTensorToMesh(mesh_device)
            self.decode_packed_gate_up = ttnn.as_tensor(
                packed_gate_up,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
                cache_file_name=get_cache_file_name(tensor_cache_path, "decode_packed_gate_up"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.decode_packed_gate_up_bias = ttnn.as_tensor(
                packed_gate_up_bias,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
                cache_file_name=get_cache_file_name(tensor_cache_path, "decode_indexed_packed_gate_up_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.decode_down = ttnn.as_tensor(
                expert_state["down_proj"].reshape(
                    1,
                    self.num_experts,
                    self.intermediate_size,
                    self.hidden_size,
                ),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
                cache_file_name=get_cache_file_name(tensor_cache_path, "decode_down"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.decode_down_bias = ttnn.as_tensor(
                expert_state["down_proj_bias"].reshape(self.num_experts, self.hidden_size),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
                cache_file_name=get_cache_file_name(tensor_cache_path, "decode_indexed_down_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # Indexed sparse_matmul retains the legacy sparsity operand for API
            # compatibility but never reads it. Keep one setup-time constant.
            self.decode_unused_sparsity = ttnn.as_tensor(
                torch.zeros((1, 1, 1, self.num_experts), dtype=torch.bfloat16),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
                cache_file_name=get_cache_file_name(tensor_cache_path, "decode_unused_sparsity"),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self.decode_expert_config = ExpertConfig(
                intermediate_size=self.intermediate_size,
                num_experts=self.num_experts,
                hidden_size=self.hidden_size,
                num_experts_per_tok=self.top_k,
                swiglu_limit=hf_config.swiglu_limit,
            )
            self.decode_program_config = GPTOSSProgramConfig()

        expert_weights, expert_biases = _expert_torch_lists(
            expert_state,
            self.num_experts,
        )
        dispatch_group_size = 1
        num_dispatch_groups = 1
        expert_dispatch_table = ExpertMapping.create_dispatch_table(
            self.num_experts,
            dispatch_group_size,
            num_dispatch_groups,
        )
        self.experts_in_dispatch_group = ttnn.from_torch(
            expert_dispatch_table,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(None, 0)),
        )
        global_expert_idx = ttnn.from_torch(
            ExpertMapping.create_global_expert_idx_table(
                experts_per_chip=self.num_experts,
                dispatch_group_size=dispatch_group_size,
                num_dispatch_groups=num_dispatch_groups,
            ),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.global_expert_idx = ttnn.squeeze(ttnn.squeeze(global_expert_idx, 0), 0)
        self.experts = TtRoutedExpert(
            mesh_device=mesh_device,
            experts_per_chip=self.num_experts,
            global_expert_idx_table=self.global_expert_idx,
            emb_dim=hf_config.hidden_size,
            hidden_dim=hf_config.intermediate_size,
            max_tokens=_EXPERT_CHUNK_SIZE,
            torch_weights=expert_weights,
            torch_biases=expert_biases,
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat16,
            weight_cache_path=Path(tensor_cache_path) if tensor_cache_path is not None else None,
            cache_name_prefix="fused_experts",
            activation=ttnn.RoutedExpertActivation.SwiGluOai,
        )
        self._regroup_buffers_by_tokens = {}

    def _regroup_buffers(self, tokens: int):
        """Return setup-only constants for one static prefill token shape."""
        buffers = self._regroup_buffers_by_tokens.get(tokens)
        if buffers is not None:
            return buffers

        slots = tokens * self.top_k
        # Every expert region starts on a tile boundary.  Reserving 31 rows for
        # all but one region is the exact worst-case alignment overhead.
        capacity = slots + (ttnn.TILE_SIZE * (self.num_experts - 1))
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

        def as_uint32(host_tensor):
            return ttnn.from_torch(
                host_tensor,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        buffers = (
            as_uint32(torch.arange(slots, dtype=torch.int32).reshape(1, slots)),
            as_uint32(torch.zeros((1, slots), dtype=torch.int32)),
            as_uint32(torch.zeros((1, capacity), dtype=torch.int32)),
            capacity,
        )
        self._regroup_buffers_by_tokens[tokens] = buffers
        return buffers

    def _local_routing_setup(self, indices):
        """Build 1x1 dispatch metadata without initializing fabric.

        The shared DeepSeek helper all-gathers its histogram before computing
        offsets, even when the dispatch group has one device.  Here the local
        histogram is already the global count.  Integer TTNN ops round each
        expert's allocation to a tile and ``cumsum`` forms exact region starts;
        no metadata leaves the device or depends on host synchronization.
        """
        if len(indices.shape) == 3:
            indices = ttnn.squeeze(indices, 0)
        histograms = ttnn.experimental.deepseek_prefill.masked_bincount(
            indices,
            self.experts_in_dispatch_group,
            self.num_experts,
            self.top_k,
        )
        counts = ttnn.reshape(histograms, (1, self.num_experts))
        tiled_counts = ttnn.to_layout(counts, ttnn.TILE_LAYOUT)
        tiled_counts_i32 = ttnn.typecast(tiled_counts, ttnn.int32)
        rounded_counts = ttnn.bitwise_left_shift(
            ttnn.bitwise_right_shift(ttnn.add(tiled_counts_i32, ttnn.TILE_SIZE - 1), 5),
            5,
        )
        inclusive_offsets = ttnn.cumsum(
            rounded_counts,
            dim=-1,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inclusive_counts = ttnn.cumsum(
            tiled_counts_i32,
            dim=-1,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        region_offsets_i32 = ttnn.subtract(inclusive_offsets, rounded_counts)
        token_offsets_i32 = ttnn.subtract(inclusive_counts, tiled_counts_i32)
        padding_offsets_i32 = ttnn.subtract(region_offsets_i32, token_offsets_i32)
        region_offsets_rm = ttnn.to_layout(region_offsets_i32, ttnn.ROW_MAJOR_LAYOUT)
        padding_offsets_rm = ttnn.to_layout(padding_offsets_i32, ttnn.ROW_MAJOR_LAYOUT)
        region_offsets = ttnn.typecast(region_offsets_rm, ttnn.uint32)
        padding_offsets = ttnn.typecast(padding_offsets_rm, ttnn.uint32)

        tiled_counts.deallocate(True)
        tiled_counts_i32.deallocate(True)
        rounded_counts.deallocate(True)
        inclusive_offsets.deallocate(True)
        inclusive_counts.deallocate(True)
        region_offsets_i32.deallocate(True)
        token_offsets_i32.deallocate(True)
        padding_offsets_i32.deallocate(True)
        region_offsets_rm.deallocate(True)
        padding_offsets_rm.deallocate(True)
        return region_offsets, padding_offsets, counts, histograms

    def _run_chunk(self, hidden_states):
        logical_tokens = hidden_states.shape[-2]
        dispatch_tokens = ((logical_tokens + _ROUTING_GRANULARITY - 1) // _ROUTING_GRANULARITY) * _ROUTING_GRANULARITY
        if dispatch_tokens == logical_tokens:
            routed_input = hidden_states
        else:
            routed_input = ttnn.pad(
                hidden_states,
                [(0, 0), (0, 0), (0, dispatch_tokens - logical_tokens), (0, 0)],
                value=0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        positions, zero_slots, zero_capacity, capacity = self._regroup_buffers(dispatch_tokens)
        indices, scores = self.router(routed_input, use_throughput_experts=True)
        region_offsets, padding_offsets, counts, histograms = self._local_routing_setup(indices)

        # Sort the flattened routed slots by expert.  UINT16 reshape is not a
        # supported device operation, so use exact UINT32 while flattening and
        # convert back for the integer sort.
        flat_indices_u32 = ttnn.typecast(indices, ttnn.uint32)
        flat_indices_rm = ttnn.to_layout(flat_indices_u32, ttnn.ROW_MAJOR_LAYOUT)
        flat_indices = ttnn.reshape(flat_indices_rm, (1, dispatch_tokens * self.top_k))
        flat_indices_u16 = ttnn.typecast(flat_indices, ttnn.uint16)
        flat_indices_tiled = ttnn.to_layout(flat_indices_u16, ttnn.TILE_LAYOUT)
        sorted_experts, permutation = ttnn.sort(flat_indices_tiled, dim=-1, descending=False)
        sorted_experts_rm = ttnn.to_layout(sorted_experts, ttnn.ROW_MAJOR_LAYOUT)
        permutation_rm = ttnn.to_layout(permutation, ttnn.ROW_MAJOR_LAYOUT)
        sorted_experts_rm = ttnn.reshape(sorted_experts_rm, (1, dispatch_tokens * self.top_k))
        permutation_rm = ttnn.reshape(permutation_rm, (1, dispatch_tokens * self.top_k))

        padding_by_slot = ttnn.gather(padding_offsets, -1, index=sorted_experts_rm)
        destinations = ttnn.add(positions, padding_by_slot)
        source_rows = ttnn.logical_right_shift(permutation_rm, 2)
        dispatch_rows = ttnn.scatter(zero_capacity, -1, destinations, source_rows)
        slot_to_destination = ttnn.scatter(zero_slots, -1, permutation_rm, destinations)

        hidden_bf16 = routed_input
        if routed_input.dtype != ttnn.bfloat16:
            hidden_bf16 = ttnn.typecast(routed_input, ttnn.bfloat16)
        hidden_rm = ttnn.to_layout(hidden_bf16, ttnn.ROW_MAJOR_LAYOUT)
        hidden_2d = ttnn.reshape(hidden_rm, (dispatch_tokens, self.hidden_size))
        dispatched = ttnn.embedding(
            dispatch_rows,
            hidden_2d,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dispatched_2d = ttnn.reshape(dispatched, (capacity, self.hidden_size))
        expert_output = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            dispatched_2d,
            region_offsets,
            counts,
            self.global_expert_idx,
            self.experts.gate_projs,
            self.experts.up_projs,
            self.experts.down_projs,
            max_dispatched_tokens_per_expert=dispatch_tokens,
            compute_kernel_config=self.experts.compute_kernel_config,
            activation=ttnn.RoutedExpertActivation.SwiGluOai,
            gate_biases=self.experts.gate_biases,
            up_biases=self.experts.up_biases,
            down_biases=self.experts.down_biases,
        )
        dispatched.deallocate(True)
        expert_output_bf16 = ttnn.typecast(expert_output, ttnn.bfloat16)
        expert_output_rm = ttnn.to_layout(expert_output_bf16, ttnn.ROW_MAJOR_LAYOUT)
        slots = ttnn.embedding(
            slot_to_destination,
            expert_output_rm,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        combined = ttnn.reshape(slots, (1, 1, dispatch_tokens, self.top_k, self.hidden_size))
        scores_rm = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)
        scores_5d = ttnn.reshape(scores_rm, (1, 1, dispatch_tokens, self.top_k, 1))
        output = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            combined,
            scores_5d,
            None,
            None,
            expert_dim=3,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        flat_indices_u32.deallocate(True)
        flat_indices_rm.deallocate(True)
        flat_indices_u16.deallocate(True)
        flat_indices_tiled.deallocate(True)
        sorted_experts.deallocate(True)
        permutation.deallocate(True)
        sorted_experts_rm.deallocate(True)
        permutation_rm.deallocate(True)
        padding_by_slot.deallocate(True)
        destinations.deallocate(True)
        source_rows.deallocate(True)
        dispatch_rows.deallocate(True)
        slot_to_destination.deallocate(True)
        hidden_rm.deallocate(True)
        if hidden_bf16 is not routed_input:
            hidden_bf16.deallocate(True)
        expert_output.deallocate(True)
        expert_output_bf16.deallocate(True)
        expert_output_rm.deallocate(True)
        slots.deallocate(True)
        scores_rm.deallocate(True)
        region_offsets.deallocate(True)
        padding_offsets.deallocate(True)
        histograms.deallocate(True)
        scores.deallocate(True)
        indices.deallocate(True)
        if routed_input is not hidden_states:
            routed_input.deallocate(True)

        output = ttnn.reshape(output, (1, 1, dispatch_tokens, self.hidden_size))
        if dispatch_tokens != logical_tokens:
            padded_output = output
            output = ttnn.slice(
                padded_output,
                starts=[0, 0, 0, 0],
                ends=[1, 1, logical_tokens, self.hidden_size],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            padded_output.deallocate(True)
        return output

    def _decode_full_local_buffers(self, tokens: int):
        buffers = self._decode_full_local_buffers_by_tokens.get(tokens)
        if buffers is not None:
            return buffers

        shard_spec = ttnn.ShardSpec(
            self.decode_full_local_drain_core,
            [tokens, self.top_k],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        indices_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )
        scores_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )
        combine_output = ttnn.from_torch(
            torch.zeros((self.top_k, tokens, self.hidden_size), dtype=torch.bfloat16),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
        )
        buffers = (indices_memory_config, scores_memory_config, combine_output)
        self._decode_full_local_buffers_by_tokens[tokens] = buffers
        return buffers

    def _decode_full_local(self, hidden_states):
        """Run all logical decode users through one FullLocal fused kernel."""
        tokens = hidden_states.shape[-2]
        indices_memory_config, scores_memory_config, combine_output = self._decode_full_local_buffers(tokens)
        routed_input = ttnn.reshape(hidden_states, (1, tokens, 1, self.hidden_size))
        indices, scores = self.router(routed_input, use_throughput_experts=True)

        sparse_input_l1 = ttnn.reshape(routed_input, (1, tokens, self.hidden_size))
        sparse_input_l1 = ttnn.to_layout(
            sparse_input_l1,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        indices = ttnn.reshape(indices, (1, tokens, self.top_k))
        indices_rm = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        indices.deallocate(True)
        indices_l1 = ttnn.to_memory_config(indices_rm, indices_memory_config)

        scores = ttnn.reshape(scores, (1, tokens, self.top_k))
        scores_rm = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores.deallocate(True)
        scores_l1 = ttnn.to_memory_config(scores_rm, scores_memory_config)

        outputs = ttnn.experimental.moe_compute(
            sparse_input_l1,
            indices_l1,
            scores_l1,
            self.decode_full_local_expert_mapping,
            self.decode_full_local_w0_w1,
            self.decode_full_local_w2,
            layer_id=0,
            output_height_shard_dim=self.decode_full_local_output_height_shard_dim,
            intermediate_size=self.intermediate_size,
            has_bias=True,
            cluster_axis=None,
            topology=None,
            num_links=None,
            mux_core_range_set=None,
            optional_output_tensor=combine_output,
            optional_cross_device_semaphore=None,
            activation_type=MoEActivationFunction.SWIGLU,
            compute_only=False,
        )
        sparse_input_l1.deallocate(True)
        indices_l1.deallocate(True)
        scores_l1.deallocate(True)
        for tensor in (outputs[0], outputs[1], outputs[2], outputs[4]):
            tensor.deallocate(True)

        # FullLocal returns unweighted expert-major RM slots.  The dedicated
        # score/reduce kernel folds transpose + multiply + sum and restores a
        # TILE output for the residual/next-layer contract.
        slots4d = ttnn.reshape(outputs[5], (self.top_k, 1, tokens, self.hidden_size))
        padded_slots = ttnn.tilize_with_val_padding(
            slots4d,
            output_tensor_shape=(self.top_k, 1, 32, self.hidden_size),
            pad_value=0.0,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        indices4d = ttnn.reshape(indices_rm, (tokens, 1, 1, self.top_k))
        scores4d = ttnn.reshape(scores_rm, (tokens, 1, 1, self.top_k))
        reduced = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            padded_slots,
            indices4d,
            self.decode_full_local_reduce_expert_mapping,
            reduce_dim=0,
            split_size=self.hidden_size,
            cluster_axis=0,
            output_memory_config=self.decode_full_local_reduce_output_memory_config,
            scores_tensor=scores4d,
            num_shared_experts=0,
        )[0]
        padded_slots.deallocate(True)
        indices_rm.deallocate(True)
        scores_rm.deallocate(True)
        return reduced

    def _decode_single_user(self, hidden_states):
        """Run one token through compact Top-K indexed expert projections."""
        expert_indices, routing_scores = self.router(hidden_states, use_throughput_experts=True)
        expert_indices_rm = ttnn.to_layout(expert_indices, ttnn.ROW_MAJOR_LAYOUT)
        expert_indices.deallocate(True)
        expert_indices_rm = ttnn.reshape(expert_indices_rm, (1, 1, 1, self.top_k))
        embedding_indices = ttnn.typecast(expert_indices_rm, ttnn.uint32)
        output_tile = ttnn.Tile([32, 32])

        gate_up = ttnn.sparse_matmul(
            hidden_states,
            self.decode_packed_gate_up,
            sparsity=self.decode_unused_sparsity,
            indices=expert_indices_rm,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=self.decode_program_config.get_decode_gate_up_config(
                hidden_states.shape[2],
                self.decode_packed_gate_up.shape[3],
                k=hidden_states.shape[-1],
            ),
            dtype=ttnn.bfloat8_b,
        )
        gate_up = ttnn.reshape(gate_up, (1, self.top_k, 2 * self.intermediate_size))
        gate_up_bias = ttnn.embedding(
            embedding_indices,
            self.decode_packed_gate_up_bias,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        gate_up = ttnn.add(gate_up, gate_up_bias, output_tensor=gate_up)
        gate_up_bias.deallocate(True)
        gate = ttnn.slice(
            gate_up,
            [0, 0, 0],
            [1, self.top_k, self.intermediate_size],
            [1, 1, 1],
        )
        up = ttnn.slice(
            gate_up,
            [0, 0, self.intermediate_size],
            [1, self.top_k, 2 * self.intermediate_size],
            [1, 1, 1],
        )
        gate_up.deallocate(True)

        down_input = apply_swiglu(gate, up, self.decode_expert_config)
        down_input = ttnn.reshape(down_input, (1, self.top_k, 1, self.intermediate_size))
        down = ttnn.sparse_matmul(
            down_input,
            self.decode_down,
            sparsity=self.decode_unused_sparsity,
            indices=expert_indices_rm,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=self.decode_program_config.get_decode_down_config(
                down_input.shape[2],
                self.decode_down.shape[-1],
                k=down_input.shape[-1],
            ),
            dtype=ttnn.bfloat8_b,
        )
        down_input.deallocate(True)
        expert_indices_rm.deallocate(True)
        output = ttnn.reshape(down, (1, self.top_k, self.hidden_size))
        down_bias = ttnn.embedding(
            embedding_indices,
            self.decode_down_bias,
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
        return ttnn.reshape(
            output,
            (1, 1, 1, self.hidden_size),
            (1, 1, ttnn.TILE_SIZE, self.hidden_size),
        )

    def __call__(self, hidden_states, *, is_decode: bool):
        if is_decode:
            if self.decode_uses_full_local:
                return self._decode_full_local(hidden_states)
            if hidden_states.shape[-2] == 1:
                return self._decode_single_user(hidden_states)
            user_hidden_states = ttnn.split(hidden_states, 1, dim=2)
            outputs = []
            for user_hidden in user_hidden_states:
                outputs.append(self._decode_single_user(user_hidden))
                user_hidden.deallocate(True)
            output = ttnn.concat(outputs, dim=2)
            for user_output in outputs:
                user_output.deallocate(True)
            return output

        total_tokens = hidden_states.shape[-2]
        if total_tokens <= _EXPERT_CHUNK_SIZE:
            return self._run_chunk(hidden_states)

        chunks = ttnn.split(hidden_states, _EXPERT_CHUNK_SIZE, dim=2)
        outputs = [self._run_chunk(chunk) for chunk in chunks]
        output = ttnn.concat(outputs, dim=2)
        for input_chunk in chunks:
            input_chunk.deallocate(True)
        for chunk_output in outputs:
            chunk_output.deallocate(True)
        return output


class FusedDecoder(LightweightModule):
    """One GPT-OSS 120B decoder layer with a dedicated fused MoE graph."""

    fusion_manifest = (
        "checkpoint_allowlisted_full_local_moe_compute",
        "deepseek_moe_fast_reduce_nc_fused",
        "indexed_packed_gate_up_sparse_matmul",
        "indexed_sparse_down_matmul",
        "indexed_bias_embedding",
        "unified_routed_expert_moe",
        "local_sort_count_regroup",
        "post_combine_weighted_reduce",
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
    ):
        if tuple(mesh_device.shape) != (1, 1):
            raise ValueError(
                "FusedDecoder preserves the single-device functional-stage contract and requires mesh shape (1, 1); "
                "P150x2/P150x4 parallelization belongs to multichip-decoder"
            )
        if hf_config.hidden_size != 2880 or hf_config.head_dim != 64:
            raise ValueError("Expected the real openai/gpt-oss-120b hidden/head dimensions (2880, 64)")
        layer_type = hf_config.layer_types[layer_idx]
        if layer_type not in _SUPPORTED_LAYER_TYPES:
            raise ValueError(f"Unsupported GPT-OSS layer type {layer_type!r}")
        advertised_context = int(hf_config.max_position_embeddings)
        max_context_length = advertised_context if max_context_length is None else int(max_context_length)
        if not 1 <= max_context_length <= advertised_context:
            raise ValueError(f"max_context_length must be within [1, {advertised_context}], got {max_context_length}")
        if page_size <= 0 or page_size % ttnn.TILE_SIZE:
            raise ValueError(f"page_size must be a positive tile multiple, got {page_size}")

        local_state = _local_layer_state_dict(state_dict, layer_idx)
        cache_root = str(Path(tensor_cache_path)) if tensor_cache_path is not None else None
        mesh_config = MeshConfig(
            mesh_device.shape,
            decode=ModeConfig(tp=1, ep=1, sp=1),
            prefill=ModeConfig(tp=1, ep=1, sp=1),
        )
        ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))
        paged_attention_config = PagedAttentionConfig(
            block_size=page_size,
            max_num_blocks=max_batch_size * ((max_context_length + page_size - 1) // page_size),
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
        attention = Attention(
            mesh_device=mesh_device,
            config=attention_config,
            state_dict=substate(local_state, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=GPTOSSAttentionProgramConfig(),
            layer_idx=layer_idx,
            paged_attention_config=paged_attention_config,
            transformation_mats=rope_setup.get_both_trans_mats(),
            weight_dtype=ttnn.bfloat16,
            tensor_cache_path=get_cache_file_name(cache_root, "self_attn"),
        )
        return cls(
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
            mlp=_FusedMLP(
                mesh_device,
                hf_config,
                substate(local_state, "mlp"),
                get_cache_file_name(cache_root, "mlp"),
                layer_idx=layer_idx,
                calibrated_checkpoint_revision=calibrated_checkpoint_revision,
                max_batch_size=max_batch_size,
            ),
            calibrated_checkpoint_revision=calibrated_checkpoint_revision,
        )

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        layer_idx,
        layer_type,
        max_batch_size,
        max_context_length,
        page_size,
        input_layernorm,
        post_attention_layernorm,
        attention,
        mlp,
        calibrated_checkpoint_revision=None,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.max_batch_size = max_batch_size
        self.max_context_length = max_context_length
        self.page_size = page_size
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.self_attn = attention
        self.mlp = mlp
        self.calibrated_checkpoint_revision = calibrated_checkpoint_revision

    @property
    def kv_cache(self):
        return self.self_attn.kv_cache

    def _forward(
        self,
        hidden_states,
        *,
        position_embeddings,
        current_position,
        page_table,
        kv_cache,
        is_decode,
        user_id,
        batch_size,
    ):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attention_out = self.self_attn(
            normed,
            rope_mats=position_embeddings,
            position_idx=current_position,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            user_id=user_id,
            batch_size=batch_size,
        )
        # The attention prefill implementation consumes its input; decode
        # attention borrows it. Keep ownership explicit at this boundary.
        if is_decode:
            normed.deallocate(True)
        hidden_states = ttnn.add(residual, attention_out, output_tensor=attention_out)
        residual.deallocate(True)
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed, is_decode=is_decode)
        normed.deallocate(True)
        hidden_states = ttnn.add(residual, mlp_out, output_tensor=mlp_out)
        residual.deallocate(True)
        return hidden_states

    def prefill_forward(
        self,
        hidden_states,
        *,
        position_embeddings,
        page_table,
        kv_cache=None,
        user_id=0,
        batch_size=1,
    ):
        if page_table is None:
            raise ValueError("FusedDecoder is paged-only and requires page_table")
        if len(hidden_states.shape) != 4 or hidden_states.shape[0] != 1 or hidden_states.shape[1] != batch_size:
            raise ValueError(
                "prefill requires [1, batch, sequence, hidden] input matching batch_size, "
                f"got {tuple(hidden_states.shape)} and batch_size={batch_size}"
            )
        if batch_size < 1 or batch_size > self.max_batch_size:
            raise ValueError(f"batch_size {batch_size} is outside configured maximum {self.max_batch_size}")
        if hidden_states.shape[-1] != self.hf_config.hidden_size:
            raise ValueError(
                f"prefill hidden dimension must be {self.hf_config.hidden_size}, got {hidden_states.shape[-1]}"
            )
        if page_table.shape[-2] < batch_size:
            raise ValueError(f"page_table has {page_table.shape[-2]} rows for batch_size={batch_size}")
        logical_tokens = hidden_states.shape[-2]
        if logical_tokens < 1 or logical_tokens > self.max_context_length:
            raise ValueError(
                f"prefill logical sequence must be within [1, {self.max_context_length}], got {logical_tokens}"
            )
        padded_tokens = ((logical_tokens + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if len(position_embeddings) != 2 or any(rope.shape[-2] != logical_tokens for rope in position_embeddings):
            raise ValueError("prefill position_embeddings must be a cosine/sine pair of exact logical length")

        if padded_tokens == logical_tokens:
            working_hidden = ttnn.clone(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            working_rope = position_embeddings
        else:
            padding = [(0, 0), (0, 0), (0, padded_tokens - logical_tokens), (0, 0)]
            owned_hidden = ttnn.clone(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            working_hidden = ttnn.pad(owned_hidden, padding, value=0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            working_rope = [
                ttnn.pad(
                    ttnn.clone(rope, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                    [(0, 0), (0, 0), (0, padded_tokens - logical_tokens), (0, 0)],
                    value=0.0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for rope in position_embeddings
            ]

        working_hidden = ttnn.reshape(
            working_hidden,
            (1, 1, batch_size * padded_tokens, self.hf_config.hidden_size),
        )
        output = self._forward(
            working_hidden,
            position_embeddings=working_rope,
            current_position=None,
            page_table=page_table,
            kv_cache=self.kv_cache if kv_cache is None else kv_cache,
            is_decode=False,
            user_id=user_id,
            batch_size=batch_size,
        )
        if padded_tokens != logical_tokens:
            for rope in working_rope:
                rope.deallocate(True)
        output = ttnn.reshape(output, (1, batch_size, padded_tokens, self.hf_config.hidden_size))
        if padded_tokens != logical_tokens:
            output = ttnn.slice(
                output,
                starts=[0, 0, 0, 0],
                ends=[1, batch_size, logical_tokens, self.hf_config.hidden_size],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return output

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
        if current_position is None:
            raise ValueError("decode requires a device-resident current_position tensor")
        if current_position.shape[-1] < batch_size:
            raise ValueError("current_position is shorter than the decode batch")
        if len(position_embeddings) != 2 or any(rope.shape[1] < batch_size for rope in position_embeddings):
            raise ValueError("decode position_embeddings must be a cosine/sine pair covering the decode batch")
        if page_table.shape[-2] < batch_size:
            raise ValueError(f"page_table has {page_table.shape[-2]} rows for batch_size={batch_size}")
        return self._forward(
            ttnn.clone(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            position_embeddings=position_embeddings,
            current_position=current_position,
            page_table=page_table,
            kv_cache=self.kv_cache if kv_cache is None else kv_cache,
            is_decode=True,
            user_id=0,
            batch_size=batch_size,
        )

    def forward(self, hidden_states, *, mode, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")

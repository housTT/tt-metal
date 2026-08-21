# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Four-chip tensor-parallel decoder for Qwen/Qwen3.6-27B.

The implementation keeps :class:`OptimizedDecoder` as its local-kernel
baseline.  Projection and state tensors are head/feature sharded over the
machine's 1x4 Blackhole ring.  The measured layer-stack winner keeps the
residual stream replicated and reduces row-parallel outputs at the two
algebraically required boundaries.  A validated but slower fractured-boundary
alternative remains available for future fused-collective work.
"""

from __future__ import annotations

import math
import os

import torch
import ttnn

from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import (
    FULL_PREFILL_CHUNK_SIZE,
    FULL_PREFILL_Q_CHUNK_SIZE,
    FULL_PREFILL_SDPA_LIMIT,
    LINEAR_CHUNK_SIZE,
    _state_key,
)
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    OptimizedDecoder,
    _decode_l1_memory,
    _norm_l1_memory,
)
from models.autoports.qwen_qwen3_6_27b.tt.precision import PrecisionPolicy, load_precision_policy


TP_SIZE = 4
MESH_SHAPE = (1, 4)
FULL_LOCAL_INTERMEDIATE_SIZE = 4608


def _replicated_weight(source, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.as_tensor(
        source,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _fractured_norm_weight(source, mesh_device):
    return ttnn.as_tensor(
        source,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )


def _mesh_sharded_weight(
    source,
    mesh_device,
    *,
    shard_dim: int,
    dtype,
    dram_sharded: bool,
    local_k: int,
    local_n: int,
):
    """Materialize an already-transposed KxN tensor with one TP shard/device."""
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    if dram_sharded:
        dram_grid_size = mesh_device.dram_grid_size()
        dram_cores = dram_grid_size.x * dram_grid_size.y
        grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
                )
            }
        )
        padded_n = math.ceil(local_n / (32 * dram_cores)) * 32 * dram_cores
        if padded_n != local_n:
            raise ValueError(
                f"local N={local_n} must be padded before mesh sharding (DRAM N={padded_n})"
            )
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                grid,
                (local_k, local_n // dram_cores),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
    return ttnn.as_tensor(
        source,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim),
    )


def _column_weight(source, mesh_device, *, dtype, decode_copy=True):
    """Return interleaved and optional DRAM-sharded column-parallel copies."""
    transposed = source.transpose(-2, -1).contiguous()
    local_n = source.shape[0] // TP_SIZE
    common = dict(
        source=transposed,
        mesh_device=mesh_device,
        shard_dim=-1,
        dtype=dtype,
        local_k=source.shape[1],
        local_n=local_n,
    )
    normal = _mesh_sharded_weight(**common, dram_sharded=False)
    decode = _mesh_sharded_weight(**common, dram_sharded=True) if decode_copy else None
    return normal, decode


def _row_weight(source, mesh_device, *, dtype, decode_copy=True):
    """Return interleaved and optional DRAM-sharded row-parallel copies."""
    transposed = source.transpose(-2, -1).contiguous()
    local_k = source.shape[1] // TP_SIZE
    common = dict(
        source=transposed,
        mesh_device=mesh_device,
        shard_dim=-2,
        dtype=dtype,
        local_k=local_k,
        local_n=source.shape[0],
    )
    normal = _mesh_sharded_weight(**common, dram_sharded=False)
    decode = _mesh_sharded_weight(**common, dram_sharded=True) if decode_copy else None
    return normal, decode


def _per_device_cat(parts_by_device):
    """Pack each device's logical output group contiguously before mesh sharding."""
    return torch.cat([torch.cat(parts, dim=0) for parts in parts_by_device], dim=0)


class MultichipDecoder(OptimizedDecoder):
    """TP=4 Qwen decoder retaining the optimized single-chip local graph."""

    MULTICHIP_MANIFEST = (
        "tp4_column_projections",
        "tp4_row_projections",
        "local_attention_heads",
        "local_linear_state",
        "paged_local_kv_cache",
        "ring_all_reduce",
        "ring_reduce_scatter",
        "distributed_rms_norm",
    )
    TOTAL_MULTICHIP_COUNTS = {name: 0 for name in MULTICHIP_MANIFEST}
    # MeshDevice does not expose a documented weak-reference lifecycle. Keep a
    # validated identity entry for the process lifetime so decoder layers share
    # stable trace addresses instead of reserving CCL buffers per layer.
    _PERSISTENT_CCL_POOLS = {}
    LAST_PROVENANCE_INSTANCE = None

    @classmethod
    def reset_multichip_counts(cls):
        cls.TOTAL_MULTICHIP_COUNTS = {name: 0 for name in cls.MULTICHIP_MANIFEST}

    def _record_multichip(self, name: str):
        self.multichip_counters[name] += 1
        type(self).TOTAL_MULTICHIP_COUNTS[name] += 1

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        page_block_size: int = 64,
        precision_policy: PrecisionPolicy | None = None,
        **kwargs,
    ):
        del kwargs
        precision_policy = precision_policy or load_precision_policy()
        config = getattr(hf_config, "text_config", hf_config)
        if mesh_device.get_num_devices() != TP_SIZE or tuple(mesh_device.shape) != MESH_SHAPE:
            raise ValueError("MultichipDecoder requires this machine's 1x4 mesh")
        if (
            config.hidden_size != 5120
            or config.intermediate_size != 17408
            or config.num_attention_heads != 24
            or config.num_key_value_heads != 4
            or config.head_dim != 256
        ):
            raise ValueError("MultichipDecoder expects Qwen/Qwen3.6-27B")

        layer_kind = config.layer_types[layer_idx]
        hidden_size = config.hidden_size
        logical_local_intermediate = config.intermediate_size // TP_SIZE
        local_intermediate = (
            FULL_LOCAL_INTERMEDIATE_SIZE
            if layer_kind == "full_attention"
            else logical_local_intermediate
        )
        local_value_heads = config.linear_num_value_heads // TP_SIZE
        local_key_heads = config.linear_num_key_heads // TP_SIZE

        def norm_weight(suffix, *, offset):
            source = state_dict[_state_key(state_dict, layer_idx, suffix)].float()
            if offset:
                source = source + 1.0
            return _replicated_weight(source.reshape(1, 1, 1, -1), mesh_device)

        common = dict(
            mesh_device=mesh_device,
            config=config,
            layer_idx=layer_idx,
            layer_kind=layer_kind,
            hidden_size=hidden_size,
            intermediate_size=local_intermediate,
            rms_norm_eps=config.rms_norm_eps,
            max_context=config.max_position_embeddings,
            page_block_size=page_block_size,
            full_prefill_sdpa_limit=FULL_PREFILL_SDPA_LIMIT,
            full_prefill_chunk_size=FULL_PREFILL_CHUNK_SIZE,
            full_prefill_q_chunk_size=FULL_PREFILL_Q_CHUNK_SIZE,
            input_norm=norm_weight("input_layernorm.weight", offset=True),
            post_attention_norm=norm_weight("post_attention_layernorm.weight", offset=True),
        )
        for role, suffix in (
            ("input_norm_fractured", "input_layernorm.weight"),
            ("post_attention_norm_fractured", "post_attention_layernorm.weight"),
        ):
            source = state_dict[_state_key(state_dict, layer_idx, suffix)].float() + 1.0
            common[role] = _fractured_norm_weight(source.reshape(1, 1, 1, -1), mesh_device)

        dtype_by_name = {
            "bf16": ttnn.bfloat16,
            "bfp8": ttnn.bfloat8_b,
            "bfp4": ttnn.bfloat4_b,
        }
        mlp_policies = {
            role: precision_policy.weight_dtype(
                f"mlp_{role}", layer_kind=layer_kind, layer_idx=layer_idx
            )
            for role in ("gate", "up", "down")
        }
        if any(policy not in dtype_by_name for policy in mlp_policies.values()):
            raise ValueError("multichip MLP dtype must be bf16, bfp8, or bfp4")
        mlp_dtypes = {role: dtype_by_name[policy] for role, policy in mlp_policies.items()}
        mlp_sources = {
            role: state_dict[_state_key(state_dict, layer_idx, f"mlp.{role}_proj.weight")]
            for role in ("gate", "up", "down")
        }
        if local_intermediate != logical_local_intermediate:
            # Padding each TP-local intermediate shard independently preserves
            # device ownership while making 144 local tiles divisible over 16
            # Blackhole compute cores.  The added rows/columns are algebraic
            # zeros, so they remain internal to the MLP and do not alter the
            # replicated 5,120-wide residual contract.
            local_padding = local_intermediate - logical_local_intermediate
            for role in ("gate", "up"):
                mlp_sources[role] = torch.cat(
                    [
                        torch.nn.functional.pad(local_source, (0, 0, 0, local_padding))
                        for local_source in torch.chunk(mlp_sources[role], TP_SIZE, dim=0)
                    ],
                    dim=0,
                )
            mlp_sources["down"] = torch.cat(
                [
                    torch.nn.functional.pad(local_source, (0, local_padding, 0, 0))
                    for local_source in torch.chunk(mlp_sources["down"], TP_SIZE, dim=1)
                ],
                dim=1,
            )
        for role in ("gate", "up"):
            normal, decode = _column_weight(
                mlp_sources[role], mesh_device, dtype=mlp_dtypes[role]
            )
            common[f"mlp_{role}"] = normal
            common[f"mlp_{role}_decode"] = decode
            common[f"mlp_{role}_dram_prefill"] = decode
        normal, decode = _row_weight(
            mlp_sources["down"], mesh_device, dtype=mlp_dtypes["down"]
        )
        common["mlp_down"] = normal
        common["mlp_down_decode"] = decode
        common["mlp_down_dram_prefill"] = decode

        if layer_kind == "full_attention":
            input_projection_dtype = dtype_by_name[precision_policy.weight_dtype(
                "attention_input", layer_kind=layer_kind, layer_idx=layer_idx
            )]
            output_projection_dtype = dtype_by_name[precision_policy.weight_dtype(
                "attention_output", layer_kind=layer_kind, layer_idx=layer_idx
            )]
            local_q_heads = config.num_attention_heads // TP_SIZE
            local_kv_heads = config.num_key_value_heads // TP_SIZE
            q_and_gate = state_dict[_state_key(state_dict, layer_idx, "self_attn.q_proj.weight")]
            q_and_gate = q_and_gate.reshape(
                config.num_attention_heads, config.head_dim * 2, hidden_size
            )
            k_source = state_dict[_state_key(state_dict, layer_idx, "self_attn.k_proj.weight")]
            v_source = state_dict[_state_key(state_dict, layer_idx, "self_attn.v_proj.weight")]
            packed_by_device = []
            for device_index in range(TP_SIZE):
                q = q_and_gate[
                    device_index * local_q_heads : (device_index + 1) * local_q_heads,
                    : config.head_dim,
                ].reshape(-1, hidden_size)
                gate = q_and_gate[
                    device_index * local_q_heads : (device_index + 1) * local_q_heads,
                    config.head_dim :,
                ].reshape(-1, hidden_size)
                k = torch.chunk(k_source, TP_SIZE, dim=0)[device_index]
                v = torch.chunk(v_source, TP_SIZE, dim=0)[device_index]
                packed_by_device.append((q, k, v, gate))
            full_source = _per_device_cat(packed_by_device)
            full_qkv, full_qkv_decode = _column_weight(
                full_source, mesh_device, dtype=input_projection_dtype
            )
            o_source = state_dict[_state_key(state_dict, layer_idx, "self_attn.o_proj.weight")]
            o_proj, o_proj_decode = _row_weight(
                o_source, mesh_device, dtype=output_projection_dtype
            )
            common.update(
                num_heads=local_q_heads,
                num_kv_heads=local_kv_heads,
                head_dim=config.head_dim,
                rotary_dim=int(
                    config.head_dim * config.rope_parameters.get("partial_rotary_factor", 0.25)
                ),
                attention_scale=config.head_dim**-0.5,
                full_qkv=full_qkv,
                full_qkv_dram_decode=full_qkv_decode,
                o_proj=o_proj,
                o_proj_dram_decode=o_proj_decode,
                q_norm=ttnn.reshape(
                    norm_weight("self_attn.q_norm.weight", offset=True), [1, 1, 1, config.head_dim]
                ),
                k_norm=ttnn.reshape(
                    norm_weight("self_attn.k_norm.weight", offset=True), [1, 1, 1, config.head_dim]
                ),
            )
        elif layer_kind == "linear_attention":
            input_projection_dtype = dtype_by_name[precision_policy.weight_dtype(
                "attention_input", layer_kind=layer_kind, layer_idx=layer_idx
            )]
            output_projection_dtype = dtype_by_name[precision_policy.weight_dtype(
                "attention_output", layer_kind=layer_kind, layer_idx=layer_idx
            )]
            qkv_source = state_dict[_state_key(state_dict, layer_idx, "linear_attn.in_proj_qkv.weight")]
            z_source = state_dict[_state_key(state_dict, layer_idx, "linear_attn.in_proj_z.weight")]
            beta_source = state_dict[_state_key(state_dict, layer_idx, "linear_attn.in_proj_b.weight")]
            a_source = state_dict[_state_key(state_dict, layer_idx, "linear_attn.in_proj_a.weight")]
            q_width = config.linear_num_key_heads * config.linear_key_head_dim
            v_width = config.linear_num_value_heads * config.linear_value_head_dim
            packed_by_device = []
            conv_parts = []
            for device_index in range(TP_SIZE):
                q = qkv_source[
                    device_index * (q_width // TP_SIZE) : (device_index + 1) * (q_width // TP_SIZE)
                ]
                k = qkv_source[
                    q_width + device_index * (q_width // TP_SIZE) : q_width
                    + (device_index + 1) * (q_width // TP_SIZE)
                ]
                v = qkv_source[
                    2 * q_width + device_index * (v_width // TP_SIZE) : 2 * q_width
                    + (device_index + 1) * (v_width // TP_SIZE)
                ]
                z = torch.chunk(z_source, TP_SIZE, dim=0)[device_index]
                beta = torch.nn.functional.pad(
                    torch.chunk(beta_source, TP_SIZE, dim=0)[device_index], [0, 0, 0, 32 - local_value_heads]
                )
                decay = torch.nn.functional.pad(
                    torch.chunk(a_source, TP_SIZE, dim=0)[device_index], [0, 0, 0, 32 - local_value_heads]
                )
                # 4,160 logical rows become 4,352 so each local N shard spans
                # exactly 17 tiles per Blackhole DRAM bank.  The inherited
                # projection adapter slices the inert 192-row tail.
                tail = q.new_zeros((192, hidden_size))
                packed_by_device.append((q, k, v, z, beta, decay, tail))
                conv_parts.append(torch.cat((q, k, v), dim=0))
            linear_source = _per_device_cat(packed_by_device)
            linear_projections, linear_projections_decode = _column_weight(
                linear_source, mesh_device, dtype=input_projection_dtype
            )
            out_source = state_dict[_state_key(state_dict, layer_idx, "linear_attn.out_proj.weight")]
            linear_out_proj, linear_out_proj_decode = _row_weight(
                out_source, mesh_device, dtype=output_projection_dtype
            )

            conv_full = state_dict[_state_key(state_dict, layer_idx, "linear_attn.conv1d.weight")]
            conv_by_device = []
            for device_index, projection_rows in enumerate(conv_parts):
                del projection_rows
                indices = torch.cat(
                    (
                        torch.arange(device_index * (q_width // TP_SIZE), (device_index + 1) * (q_width // TP_SIZE)),
                        torch.arange(q_width + device_index * (q_width // TP_SIZE), q_width + (device_index + 1) * (q_width // TP_SIZE)),
                        torch.arange(2 * q_width + device_index * (v_width // TP_SIZE), 2 * q_width + (device_index + 1) * (v_width // TP_SIZE)),
                    )
                )
                conv_by_device.append(conv_full[indices, 0, :].transpose(0, 1).contiguous())
            conv_source = torch.cat(conv_by_device, dim=-1).reshape(
                1, 1, config.linear_conv_kernel_dim, -1
            )
            conv_weight = ttnn.as_tensor(
                conv_source,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            )
            a_log = state_dict[_state_key(state_dict, layer_idx, "linear_attn.A_log")]
            dt_bias = state_dict[_state_key(state_dict, layer_idx, "linear_attn.dt_bias")]
            a_decay = (-a_log.float().exp()).reshape(1, config.linear_num_value_heads, 1, 1)
            dt_bias = dt_bias.float().reshape(1, config.linear_num_value_heads, 1, 1)
            linear_norm_source = state_dict[_state_key(state_dict, layer_idx, "linear_attn.norm.weight")]
            chunk_ones = a_log.float().new_ones((1, 1, LINEAR_CHUNK_SIZE, LINEAR_CHUNK_SIZE))
            chunk_lower = chunk_ones.tril()
            chunk_strict_lower = chunk_ones.tril(diagonal=-1)
            common.update(
                linear_num_key_heads=local_key_heads,
                linear_num_value_heads=local_value_heads,
                linear_key_head_dim=config.linear_key_head_dim,
                linear_value_head_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                conv_dim=(2 * local_key_heads * config.linear_key_head_dim + local_value_heads * config.linear_value_head_dim),
                linear_chunk_size=LINEAR_CHUNK_SIZE,
                linear_projections=linear_projections,
                linear_projections_dram_decode=linear_projections_decode,
                linear_out_proj=linear_out_proj,
                linear_out_proj_dram_decode=linear_out_proj_decode,
                conv_weight=conv_weight,
                a_decay=ttnn.as_tensor(a_decay, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1)),
                dt_bias=ttnn.as_tensor(dt_bias, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1)),
                linear_chunk_lower=_replicated_weight(chunk_lower, mesh_device, dtype=ttnn.float32),
                linear_chunk_strict_lower=_replicated_weight(chunk_strict_lower, mesh_device, dtype=ttnn.float32),
                linear_chunk_identity=_replicated_weight(chunk_lower - chunk_strict_lower, mesh_device, dtype=ttnn.float32),
                # The 128-wide GDN norm is shared by all value heads.
                linear_norm=_replicated_weight(
                    linear_norm_source.reshape(1, 1, 1, -1), mesh_device
                ),
            )
        else:
            raise ValueError(f"unsupported layer kind {layer_kind}")

        decoder = cls(**common)
        decoder.fusion_counters = {name: 0 for name in cls.FUSION_MANIFEST}
        decoder.optimization_counters = {name: 0 for name in cls.OPTIMIZATION_MANIFEST}
        decoder.multichip_counters = {name: 0 for name in cls.MULTICHIP_MANIFEST}
        decoder._runtime_phase = None
        decoder.mlp_policies = mlp_policies
        decoder.mlp_weight_dtypes = mlp_dtypes
        decoder.mlp_compute_configs = {
            role: ttnn.WormholeComputeKernelConfig(
                math_fidelity=(
                    {
                        "lofi": ttnn.MathFidelity.LoFi,
                        "hifi2": ttnn.MathFidelity.HiFi2,
                        "hifi4": ttnn.MathFidelity.HiFi4,
                    }[precision_policy.fidelity(
                        f"mlp_{role}", layer_kind=layer_kind, layer_idx=layer_idx
                    )]
                ),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            for role in ("gate", "up", "down")
        }
        decoder.mlp_weight_dtype = mlp_dtypes["gate"]
        decoder.matmul_compute_config = decoder.mlp_compute_configs["gate"]
        decoder.dram_sharded_mlp = True
        mlp_cores = 16 if layer_kind == "full_attention" else 8
        decoder.dram_sharded_mlp_cores = {
            role: mlp_cores for role in ("gate", "up", "down")
        }
        default_mlp_blocks = (
            {"gate": 10, "up": 10, "down": 17}
            if layer_kind == "linear_attention"
            else {role: 0 for role in ("gate", "up", "down")}
        )
        decoder.dram_sharded_mlp_blocks = {
            role: int(
                os.environ.get(
                    f"QWEN36_MC_MLP_{role.upper()}_BLOCK",
                    str(default_mlp_blocks[role]),
                )
            )
            for role in ("gate", "up", "down")
        }
        decoder.mlp_padded_64 = False
        decoder.full_decode_mixed_mlp = False
        decoder.mlp_runtime_hidden_size = hidden_size
        decoder.mlp_runtime_intermediate_size = local_intermediate
        decoder.mlp_runtime_output_size = hidden_size
        decoder.dram_sharded_projections = "both"
        decoder.projection_fidelity = precision_policy.fidelity(
            "projection", layer_kind=layer_kind, layer_idx=layer_idx
        )
        decoder.input_projection_weight_dtype = input_projection_dtype
        decoder.output_projection_weight_dtype = output_projection_dtype
        linear_attention_ccl_dtype = precision_policy.ccl["linear_attention_attention"]
        full_attention_ccl_dtype = precision_policy.ccl["full_attention_attention"]
        linear_mlp_ccl_dtype = precision_policy.ccl["linear_attention_mlp"]
        full_mlp_ccl_dtype = precision_policy.ccl["full_attention_mlp"]
        decoder.ccl_payload_dtype = (
            linear_attention_ccl_dtype if layer_kind == "linear_attention" else full_attention_ccl_dtype
        )
        if layer_kind == "linear_attention":
            decoder.attention_ccl_payload_dtype = linear_attention_ccl_dtype
            decoder.mlp_ccl_payload_dtype = linear_mlp_ccl_dtype
        else:
            decoder.attention_ccl_payload_dtype = full_attention_ccl_dtype
            decoder.mlp_ccl_payload_dtype = full_mlp_ccl_dtype
        decoder.persistent_ccl_slot_dtypes = {
            "attention_16": tuple(dict.fromkeys((linear_attention_ccl_dtype, full_attention_ccl_dtype))),
            "linear_mlp_8": (linear_mlp_ccl_dtype,),
            "full_mlp_16": (full_mlp_ccl_dtype,),
        }
        if any(
            policy not in ("bf16", "bfp8")
            for policy in (
                decoder.attention_ccl_payload_dtype,
                decoder.mlp_ccl_payload_dtype,
            )
        ):
            raise ValueError("multichip attention/MLP CCL dtype must be bf16 or bfp8")
        ccl_links_override = os.environ.get("QWEN36_MC_CCL_NUM_LINKS")
        decoder.ccl_prefill_num_links = int(ccl_links_override or "1")
        decoder.ccl_decode_num_links = int(ccl_links_override or "2")
        if decoder.ccl_prefill_num_links not in (1, 2) or decoder.ccl_decode_num_links not in (1, 2):
            raise ValueError("QWEN36_MC_CCL_NUM_LINKS must be 1 or 2")
        decoder.persistent_ccl = os.environ.get("QWEN36_MC_PERSISTENT_CCL", "1") == "1"
        decoder.persistent_ccl_pool = None
        decoder.persistent_ccl_worker_id = None
        decoder.ccl_branch_provenance = {}
        decoder.precision_policy = precision_policy
        decoder.precision_summary = precision_policy.summary()
        decoder.gdn_decode_update_geometry = "reuse96m"
        decoder.gdn_prefill_geometry = "reuse"
        decoder.prefill_matmul_geometry = "auto"
        decoder.prefill_in0_block_w = 2
        decoder.sharded_input_norm = True
        decoder.sharded_qk_norm_rope = False
        if layer_kind == "full_attention":
            decoder.full_input_dram_cores = 16
            decoder.full_input_logical_n = 3584
            decoder.full_input_dram_block = 0
            decoder.full_output_dram_cores = 16
            decoder.full_output_logical_n = hidden_size
            decoder.full_output_dram_block = 0
        else:
            decoder.linear_input_dram_cores = 8
            decoder.linear_input_logical_n = 4160
            decoder.linear_input_dram_block = 0
            decoder.linear_output_dram_cores = 16
            decoder.linear_output_logical_n = hidden_size
            decoder.linear_output_dram_block = 0
        decoder._record_multichip("tp4_column_projections")
        decoder._record_multichip("tp4_row_projections")
        if os.environ.get("QWEN36_CCL_PROVENANCE") == "1":
            cls.LAST_PROVENANCE_INSTANCE = decoder
        return decoder

    def _ensure_persistent_ccl_pool(self):
        if not self.persistent_ccl or self.persistent_ccl_pool is not None:
            return
        mesh_device = self.mesh_device
        pool_key = id(mesh_device)
        pool = type(self)._PERSISTENT_CCL_POOLS.get(pool_key)
        if pool is not None and pool["mesh_device"] is not mesh_device:
            raise RuntimeError("stale persistent CCL mesh identity entry")
        if pool is None:
            compute_grid = mesh_device.compute_with_storage_grid_size()
            worker_cores = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1),
                    )
                }
            )
            worker_id = ttnn.SubDeviceId(0)
            manager = mesh_device.create_sub_device_manager(
                [ttnn.SubDevice([worker_cores])], 0
            )
            mesh_device.load_sub_device_manager(manager)
            mesh_device.set_sub_device_stall_group([worker_id])
            pool = {
                "mesh_device": mesh_device,
                "manager": manager,
                "worker_id": worker_id,
                "worker_cores": worker_cores,
                "buffers": {
                    slot: {}
                    for slot in (
                        "attention_16",
                        "full_mlp_16",
                        "linear_mlp_8",
                    )
                },
                "semaphores": {},
            }
            for slot, _ in (
                ("attention_16", 16),
                ("full_mlp_16", 16),
                ("linear_mlp_8", 8),
            ):
                pool["semaphores"][slot] = ttnn.create_global_semaphore(
                    mesh_device, worker_cores, 0
                )
            type(self)._PERSISTENT_CCL_POOLS[pool_key] = pool
        slot_cores = {"attention_16": 16, "linear_mlp_8": 8, "full_mlp_16": 16}
        for slot, payload_dtypes in self.persistent_ccl_slot_dtypes.items():
            for payload_dtype in payload_dtypes:
                if payload_dtype in pool["buffers"][slot]:
                    continue
                buffer_memory = _decode_l1_memory(
                    32, self.hidden_size * TP_SIZE, slot_cores[slot]
                )
                pool["buffers"][slot][payload_dtype] = ttnn.from_torch(
                    torch.zeros(
                        (1, 1, 32, self.hidden_size * TP_SIZE),
                        dtype=torch.bfloat16,
                    ),
                    device=mesh_device,
                    dtype=(
                        ttnn.bfloat16
                        if payload_dtype == "bf16"
                        else ttnn.bfloat8_b
                    ),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=buffer_memory,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
        self.persistent_ccl_pool = pool
        self.persistent_ccl_worker_id = pool["worker_id"]

    def decode_forward(self, *args, **kwargs):
        # Initialize before the first decode graph is enqueued, but never alter
        # the prefill worker/subdevice contract.
        self._ensure_persistent_ccl_pool()
        return super().decode_forward(*args, **kwargs)

    def _all_reduce_partial(self, tensor, *, role):
        self._record_multichip("ring_all_reduce")
        payload_dtype = (
            self.attention_ccl_payload_dtype
            if role == "attention"
            else self.mlp_ccl_payload_dtype
        )
        if payload_dtype == "bfp8":
            tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)
        if self.persistent_ccl and self._runtime_phase == "decode":
            slot = (
                "attention_16"
                if role == "attention"
                else "linear_mlp_8"
                if self.layer_kind == "linear_attention"
                else "full_mlp_16"
            )
            expected_cores = 8 if slot == "linear_mlp_8" else 16
            actual_cores = tensor.memory_config().shard_spec.num_cores()
            if actual_cores != expected_cores:
                raise ValueError(
                    f"persistent CCL slot {slot!r} expects {expected_cores} cores, got {actual_cores}"
                )
            output = ttnn.experimental.all_reduce_async(
                tensor,
                self.persistent_ccl_pool["buffers"][slot][payload_dtype],
                cluster_axis=1,
                mesh_device=self.mesh_device,
                multi_device_global_semaphore=self.persistent_ccl_pool["semaphores"][slot],
                dtype=(
                    ttnn.bfloat16
                    if payload_dtype == "bf16"
                    else ttnn.bfloat8_b
                ),
                memory_config=tensor.memory_config(),
                topology=ttnn.Topology.Ring,
                num_links=self._active_ccl_num_links(),
                subdevice_id=self.persistent_ccl_worker_id,
            )
            branch = "explicit_persistent_all_reduce_async"
            persistent_buffer = self.persistent_ccl_pool["buffers"][slot][
                payload_dtype
            ]
        else:
            output = ttnn.all_reduce(
                tensor,
                num_links=self._active_ccl_num_links(),
                topology=ttnn.Topology.Ring,
            )
            branch = "composite_all_reduce"
            persistent_buffer = None
        if os.environ.get("QWEN36_CCL_PROVENANCE") == "1":
            self.ccl_branch_provenance[role] = {
                "branch": branch,
                "input_dtype": str(tensor.dtype),
                "output_dtype": str(output.dtype),
                "persistent_buffer_dtype": (
                    str(persistent_buffer.dtype)
                    if persistent_buffer is not None
                    else "none"
                ),
            }
        return ttnn.typecast(output, ttnn.bfloat16) if payload_dtype == "bfp8" else output

    def print_ccl_provenance(self):
        pool = self.persistent_ccl_pool
        allocations = {
            slot: {
                payload: str(buffer.dtype)
                for payload, buffer in buffers.items()
            }
            for slot, buffers in pool["buffers"].items()
        }
        print(
            "CCL_RUNTIME_PROVENANCE "
            f"layer_kind={self.layer_kind} "
            f"attention_policy={self.attention_ccl_payload_dtype} "
            f"mlp_policy={self.mlp_ccl_payload_dtype} "
            f"persistent={self.persistent_ccl} "
            f"prefill_links={self.ccl_prefill_num_links} "
            f"decode_links={self.ccl_decode_num_links} "
            f"pool_id={id(pool)} allocations={allocations} "
            f"branches={self.ccl_branch_provenance}"
        )

    def _reduce_scatter_partial(self, tensor):
        self._record_multichip("ring_reduce_scatter")
        if self.ccl_payload_dtype == "bfp8":
            tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)
        output = ttnn.reduce_scatter(
            tensor,
            dim=3,
            num_links=self._active_ccl_num_links(),
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.typecast(output, ttnn.bfloat16) if self.ccl_payload_dtype == "bfp8" else output

    def _active_ccl_num_links(self):
        return (
            self.ccl_decode_num_links
            if self._runtime_phase == "decode"
            else self.ccl_prefill_num_links
        )

    def _distributed_norm_and_gather(self, fractured, weight):
        self._record_multichip("distributed_rms_norm")
        stats = ttnn.rms_norm_pre_all_gather(fractured, dtype=ttnn.bfloat16)
        stats = ttnn.all_gather(
            stats, dim=3, num_links=1, topology=ttnn.Topology.Ring
        )
        normalized = ttnn.rms_norm_post_all_gather(
            fractured,
            stats,
            epsilon=self.rms_norm_eps,
            weight=weight,
        )
        return ttnn.all_gather(
            normalized,
            dim=3,
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def fracture_replicated_residual(self, hidden_states):
        """One-time adapter for the embedding-to-first-layer stack boundary."""
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("fracture adapter expects a replicated 5,120-wide residual")
        return self._reduce_scatter_partial(ttnn.multiply(hidden_states, 1.0 / TP_SIZE))

    def stacked_decode_forward(
        self,
        hidden_states,
        *,
        current_positions,
        cos=None,
        sin=None,
        page_table=None,
        kv_cache=None,
        linear_state=None,
    ):
        """Decode one layer with an optional TP-fractured 1,280-wide boundary.

        Normalized activations are gathered only for the column-parallel
        projection.  Both row-parallel residual updates remain fractured, so
        a layer stack avoids the all-gather half of both per-layer all-reduces.
        Real-layer measurements on this mesh reject this graph because its
        distributed norms cost more than those saved all-gathers; replicated
        ``decode_forward`` is therefore the production layer-stack contract.
        """
        if hidden_states.shape[-1] != self.hidden_size // TP_SIZE:
            raise ValueError("stacked_decode_forward expects a 1,280-wide TP-fractured residual")
        previous_phase = self._runtime_phase
        self._runtime_phase = "decode"
        try:
            normalized = self._distributed_norm_and_gather(
                hidden_states, self.input_norm_fractured
            )
            if self.layer_kind == "full_attention":
                if any(
                    value is None
                    for value in (cos, sin, page_table, kv_cache, current_positions)
                ):
                    raise ValueError(
                        "full_attention stacked decode requires cos, sin, page_table, "
                        "current_positions, and kv_cache"
                    )
                mixed = self._full_decode(
                    normalized,
                    cos=cos,
                    sin=sin,
                    page_table=page_table,
                    current_positions=current_positions,
                    kv_cache=kv_cache,
                )
            else:
                if linear_state is None or current_positions is None:
                    raise ValueError(
                        "linear_attention stacked decode requires current_positions and linear_state"
                    )
                mixed = self._linear_decode(normalized, linear_state=linear_state)
            hidden_states = ttnn.add(
                hidden_states,
                self._reduce_scatter_partial(mixed),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            mlp_input = self._distributed_norm_and_gather(
                hidden_states, self.post_attention_norm_fractured
            )
            mlp_input = ttnn.to_memory_config(
                mlp_input,
                _norm_l1_memory(
                    32, self.hidden_size, self.dram_sharded_mlp_cores["gate"]
                ),
            )
            mlp_partial = super()._mlp(mlp_input)
            return ttnn.add(
                hidden_states,
                self._reduce_scatter_partial(mlp_partial),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        finally:
            self._runtime_phase = previous_phase

    def _mlp(self, hidden_states):
        return self._all_reduce_partial(super()._mlp(hidden_states), role="mlp")

    def _finish_layer(self, residual, mixed):
        return super()._finish_layer(
            residual,
            self._all_reduce_partial(mixed, role="attention"),
        )

    def allocate_paged_kv_cache(self, *, num_blocks: int, dtype=None):
        self._record_multichip("paged_local_kv_cache")
        return super().allocate_paged_kv_cache(num_blocks=num_blocks, dtype=dtype)

    def allocate_linear_state(self, *, batch_size: int):
        self._record_multichip("local_linear_state")
        return super().allocate_linear_state(batch_size=batch_size)

    def _full_qkv_prefill(self, *args, **kwargs):
        self._record_multichip("local_attention_heads")
        return super()._full_qkv_prefill(*args, **kwargs)

    def _full_qkv_decode(self, *args, **kwargs):
        self._record_multichip("local_attention_heads")
        return super()._full_qkv_decode(*args, **kwargs)


__all__ = ["MultichipDecoder", "MESH_SHAPE", "TP_SIZE"]

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device optimized GPT-OSS 120B decoder layer.

This stage preserves :class:`FusedDecoder`'s public prefill/decode and paged
cache contract while making the remaining material precision policy explicit.
Construction is local to this module: it does not call the fused or functional
decoder constructor, so tests can attest the optimized runtime path directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn
from models.autoports.openai_gpt_oss_120b.tt.fused_decoder import (
    _FULL_LOCAL_CHECKPOINT_REVISION,
    FusedDecoder,
    _FusedMLP,
    _local_layer_state_dict,
)
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.attention import Attention, AttentionConfig
from models.demos.gpt_oss.tt.attention.operations import apply_rope
from models.demos.gpt_oss.tt.attention_configs import GPTOSSAttentionProgramConfig
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.rms_norm import RMSNorm
from models.demos.gpt_oss.tt.topk import TopKRouter
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name, get_default_num_links
from models.demos.gpt_oss.utils.substate import substate
from models.tt_transformers.tt.common import PagedAttentionConfig, rope_scaling_model_factory
from models.tt_transformers.tt.rope import RotarySetup


@dataclass(frozen=True)
class _DRAMQKVGeometry:
    """Candidate DRAM-sharded packed-QKV block geometry."""

    name: str
    input_cores: int
    in0_block_w: int
    per_core_n: int


@dataclass(frozen=True)
class _OutputProjectionGeometry:
    """Candidate 1D output-projection working shard and block geometry."""

    name: str
    core_grid: tuple[int, int]
    in0_block_w: int
    per_core_n: int
    out_subblock_w: int


@dataclass(frozen=True)
class OptimizedDecoderPolicy:
    """Named precision policy for material decoder tensor groups."""

    name: str
    attention_weight_dtype: object
    decode_math_fidelity: object
    output_activation_dtype: object | None = None
    dram_sharded_qkv: bool = False
    separate_qkv: bool = False
    dram_qkv_geometry: _DRAMQKVGeometry | None = None
    dram_separate_qkv_per_core_n: tuple[int, int, int] | None = None
    output_projection_geometry: _OutputProjectionGeometry | None = None


_OUTPUT_PROJECTION_GEOMETRIES = (
    # N is 90 tiles.  Ceil coverage is mandatory: floor 90 // 64 == 1
    # left 26 output tiles uncovered in the rejected PCC=0.838 trial.
    _OutputProjectionGeometry("64c_ibw2_pcn2_sb2", (8, 8), 2, 2, 2),
    _OutputProjectionGeometry("32c_ibw4_pcn3_sb3", (8, 4), 4, 3, 3),
    _OutputProjectionGeometry("16c_ibw8_pcn6_sb6", (8, 2), 8, 6, 6),
    _OutputProjectionGeometry("8c_ibw16_pcn12_sb6", (8, 1), 16, 12, 6),
)
_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY = _OUTPUT_PROJECTION_GEOMETRIES[1]

ATTENTION_BFP8_POLICY = OptimizedDecoderPolicy(
    "attention_bfp8_hifi2",
    ttnn.bfloat8_b,
    ttnn.MathFidelity.HiFi2,
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)
ATTENTION_BFP8_LOFI_POLICY = OptimizedDecoderPolicy(
    "attention_bfp8_lofi",
    ttnn.bfloat8_b,
    ttnn.MathFidelity.LoFi,
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)
ATTENTION_BFP8_OUTACT_POLICY = OptimizedDecoderPolicy(
    "attention_bfp8_output_activation",
    ttnn.bfloat8_b,
    ttnn.MathFidelity.HiFi2,
    ttnn.bfloat8_b,
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)
ATTENTION_BFP8_LOFI_OUTACT_POLICY = OptimizedDecoderPolicy(
    "attention_bfp8_lofi_output_activation",
    ttnn.bfloat8_b,
    ttnn.MathFidelity.LoFi,
    ttnn.bfloat8_b,
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)
_DRAM_QKV_GEOMETRIES = (
    _DRAMQKVGeometry("90c_ibw1_pcn2", 90, 1, 2),
    _DRAMQKVGeometry("45c_ibw2_pcn4", 45, 2, 4),
    _DRAMQKVGeometry("30c_ibw3_pcn5", 30, 3, 5),
    _DRAMQKVGeometry("15c_ibw6_pcn10", 15, 6, 10),
    _DRAMQKVGeometry("10c_ibw9_pcn16", 10, 9, 16),
)
ATTENTION_BFP8_DRAM_SHARDED_POLICY = OptimizedDecoderPolicy(
    "attention_bfp8_dram_sharded_qkv",
    ttnn.bfloat8_b,
    ttnn.MathFidelity.HiFi2,
    dram_sharded_qkv=True,
    dram_qkv_geometry=_DRAM_QKV_GEOMETRIES[0],
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)
ATTENTION_BFP8_SEPARATE_QKV_POLICY = OptimizedDecoderPolicy(
    "attention_bfp8_separate_qkv",
    ttnn.bfloat8_b,
    ttnn.MathFidelity.HiFi2,
    separate_qkv=True,
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)
ATTENTION_BFP4_POLICY = OptimizedDecoderPolicy(
    "attention_bfp4_hifi2",
    ttnn.bfloat4_b,
    ttnn.MathFidelity.HiFi2,
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)
ATTENTION_BFP4_HIFI2_OUTACT_POLICY = OptimizedDecoderPolicy(
    "attention_bfp4_hifi2_output_activation",
    ttnn.bfloat4_b,
    ttnn.MathFidelity.HiFi2,
    ttnn.bfloat8_b,
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)
ATTENTION_BFP4_LOFI_POLICY = OptimizedDecoderPolicy(
    "attention_bfp4_lofi",
    ttnn.bfloat4_b,
    ttnn.MathFidelity.LoFi,
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)
ATTENTION_BFP4_LOFI_OUTACT_POLICY = OptimizedDecoderPolicy(
    "attention_bfp4_lofi_output_activation",
    ttnn.bfloat4_b,
    ttnn.MathFidelity.LoFi,
    ttnn.bfloat8_b,
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)
ATTENTION_BFP4_LOFI_SEPARATE_QKV_POLICY = OptimizedDecoderPolicy(
    "attention_bfp4_lofi_separate_qkv",
    ttnn.bfloat4_b,
    ttnn.MathFidelity.LoFi,
    separate_qkv=True,
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)
ATTENTION_BF16_CONTROL = OptimizedDecoderPolicy(
    "attention_bf16_control",
    ttnn.bfloat16,
    ttnn.MathFidelity.HiFi2,
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)


def _dram_qkv_candidate_policy(weight_dtype, fidelity, geometry):
    dtype_name = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}[weight_dtype]
    fidelity_name = {ttnn.MathFidelity.HiFi2: "hifi2", ttnn.MathFidelity.LoFi: "lofi"}[fidelity]
    return OptimizedDecoderPolicy(
        f"attention_{dtype_name}_{fidelity_name}_dram_qkv_{geometry.name}",
        weight_dtype,
        fidelity,
        dram_sharded_qkv=True,
        dram_qkv_geometry=geometry,
        output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
    )


ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY = _dram_qkv_candidate_policy(
    ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, _DRAM_QKV_GEOMETRIES[3]
)
_BFP8_DRAM_SEPARATE_QKV_CONTROL = OptimizedDecoderPolicy(
    "attention_bfp8_hifi2_dram_separate_qkv_15c",
    ttnn.bfloat8_b,
    ttnn.MathFidelity.HiFi2,
    dram_sharded_qkv=True,
    separate_qkv=True,
    dram_qkv_geometry=_DRAM_QKV_GEOMETRIES[3],
    dram_separate_qkv_per_core_n=(9, 2, 2),
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)
_BFP4_LOFI_DRAM_SEPARATE_QKV_CONTROL = OptimizedDecoderPolicy(
    "attention_bfp4_lofi_dram_separate_qkv_15c",
    ttnn.bfloat4_b,
    ttnn.MathFidelity.LoFi,
    dram_sharded_qkv=True,
    separate_qkv=True,
    dram_qkv_geometry=_DRAM_QKV_GEOMETRIES[3],
    dram_separate_qkv_per_core_n=(9, 2, 2),
    output_projection_geometry=_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
)
DEFAULT_OPTIMIZED_POLICY = ATTENTION_BFP8_POLICY
_MULTI_BATCH_OPTIMIZED_POLICY = ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY


# Private, opt-in precision x geometry matrix.  These are rejected/candidate
# policies retained to reproduce the measured sweep.  The promoted 15-core
# BFP4/LoFi entry is the production policy and shares the same object here.
_EXPERIMENTAL_DRAM_QKV_POLICIES = tuple(
    (
        ATTENTION_BFP8_DRAM_SHARDED_POLICY
        if dtype == ttnn.bfloat8_b and fidelity == ttnn.MathFidelity.HiFi2 and geometry == _DRAM_QKV_GEOMETRIES[0]
        else (
            ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY
            if dtype == ttnn.bfloat4_b and fidelity == ttnn.MathFidelity.LoFi and geometry == _DRAM_QKV_GEOMETRIES[3]
            else _dram_qkv_candidate_policy(dtype, fidelity, geometry)
        )
    )
    for dtype, fidelity in (
        (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
        (ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
        (ttnn.bfloat4_b, ttnn.MathFidelity.HiFi2),
        (ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
    )
    for geometry in _DRAM_QKV_GEOMETRIES
)
_BF16_DRAM_QKV_CONTROL = _dram_qkv_candidate_policy(ttnn.bfloat16, ttnn.MathFidelity.HiFi2, _DRAM_QKV_GEOMETRIES[0])

# Private, opt-in reproduction policies.  The 32-core geometry is repeated
# here under its evidence name; production policies above use the same object.
_EXPERIMENTAL_OUTPUT_PROJECTION_POLICIES = tuple(
    OptimizedDecoderPolicy(
        f"attention_bfp8_hifi2_oproj_{geometry.name}",
        ttnn.bfloat8_b,
        ttnn.MathFidelity.HiFi2,
        output_projection_geometry=geometry,
    )
    for geometry in _OUTPUT_PROJECTION_GEOMETRIES
)
_EXPERIMENTAL_DRAM10_OUTPUT_PROJECTION_POLICIES = tuple(
    OptimizedDecoderPolicy(
        f"attention_bfp4_lofi_dram_qkv_10c_{geometry.name}",
        ttnn.bfloat4_b,
        ttnn.MathFidelity.LoFi,
        dram_sharded_qkv=True,
        dram_qkv_geometry=_DRAM_QKV_GEOMETRIES[4],
        output_projection_geometry=geometry,
    )
    for geometry in _OUTPUT_PROJECTION_GEOMETRIES
)
_EXPERIMENTAL_DRAM15_OUTPUT_PROJECTION_POLICIES = tuple(
    OptimizedDecoderPolicy(
        f"attention_bfp4_lofi_dram_qkv_15c_{geometry.name}",
        ttnn.bfloat4_b,
        ttnn.MathFidelity.LoFi,
        dram_sharded_qkv=True,
        dram_qkv_geometry=_DRAM_QKV_GEOMETRIES[3],
        output_projection_geometry=geometry,
    )
    for geometry in _OUTPUT_PROJECTION_GEOMETRIES
)
_SUPPORTED_POLICIES = (
    (
        ATTENTION_BFP8_POLICY,
        ATTENTION_BFP8_LOFI_POLICY,
        ATTENTION_BFP8_OUTACT_POLICY,
        ATTENTION_BFP8_LOFI_OUTACT_POLICY,
        ATTENTION_BFP8_SEPARATE_QKV_POLICY,
        ATTENTION_BFP4_POLICY,
        ATTENTION_BFP4_HIFI2_OUTACT_POLICY,
        ATTENTION_BFP4_LOFI_POLICY,
        ATTENTION_BFP4_LOFI_SEPARATE_QKV_POLICY,
        ATTENTION_BFP4_LOFI_OUTACT_POLICY,
        ATTENTION_BF16_CONTROL,
        _BFP8_DRAM_SEPARATE_QKV_CONTROL,
        _BFP4_LOFI_DRAM_SEPARATE_QKV_CONTROL,
    )
    + _EXPERIMENTAL_DRAM_QKV_POLICIES
    + (_BF16_DRAM_QKV_CONTROL,)
    + _EXPERIMENTAL_OUTPUT_PROJECTION_POLICIES
    + _EXPERIMENTAL_DRAM10_OUTPUT_PROJECTION_POLICIES
    + _EXPERIMENTAL_DRAM15_OUTPUT_PROJECTION_POLICIES
)


def _validate_dram_qkv_geometry(geometry: _DRAMQKVGeometry, *, k: int, n: int) -> None:
    k_tiles, n_tiles = k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
    if k % ttnn.TILE_SIZE or n % ttnn.TILE_SIZE:
        raise ValueError(f"DRAM QKV requires tile-aligned K/N, got K={k}, N={n}")
    if geometry.input_cores <= 0 or k_tiles % geometry.input_cores:
        raise ValueError(f"DRAM QKV input cores must divide K tiles ({k_tiles}), got {geometry.input_cores}")
    shard_k_tiles = k_tiles // geometry.input_cores
    if geometry.in0_block_w <= 0 or shard_k_tiles % geometry.in0_block_w:
        raise ValueError(f"DRAM QKV in0_block_w={geometry.in0_block_w} must divide shard K={shard_k_tiles}")
    if geometry.per_core_n <= 0 or math.ceil(n_tiles / geometry.per_core_n) > 130:
        raise ValueError(f"invalid DRAM QKV per_core_n={geometry.per_core_n} for N tiles={n_tiles}")


def _validate_output_projection_geometry(geometry: _OutputProjectionGeometry, *, k: int, n: int) -> None:
    k_tiles, n_tiles = k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
    core_count = geometry.core_grid[0] * geometry.core_grid[1]
    if k % ttnn.TILE_SIZE or n % ttnn.TILE_SIZE:
        raise ValueError(f"output projection requires tile-aligned K/N, got K={k}, N={n}")
    if core_count <= 0 or k_tiles % core_count:
        raise ValueError(f"output projection cores must divide K tiles ({k_tiles}), got {core_count}")
    shard_k_tiles = k_tiles // core_count
    if geometry.in0_block_w <= 0 or shard_k_tiles % geometry.in0_block_w:
        raise ValueError(f"output projection in0_block_w={geometry.in0_block_w} must divide shard K={shard_k_tiles}")
    if geometry.per_core_n <= 0 or core_count * geometry.per_core_n < n_tiles:
        raise ValueError(
            f"output projection config covers {core_count * geometry.per_core_n} N tiles, requires {n_tiles}"
        )
    if geometry.out_subblock_w <= 0 or geometry.per_core_n % geometry.out_subblock_w:
        raise ValueError(f"output subblock {geometry.out_subblock_w} must divide per_core_n={geometry.per_core_n}")


def _make_output_projection_program_config(geometry: _OutputProjectionGeometry):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*geometry.core_grid),
        in0_block_w=geometry.in0_block_w,
        out_subblock_h=1,
        out_subblock_w=geometry.out_subblock_w,
        out_block_h=1,
        out_block_w=geometry.per_core_n,
        per_core_M=1,
        per_core_N=geometry.per_core_n,
        # The decode activation is explicitly width-sharded across K.  TTNN's
        # 1D multicast factory requires batch fusion for sharded input A.
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _resolve_policy(policy: OptimizedDecoderPolicy | None, max_batch_size: int) -> OptimizedDecoderPolicy:
    """Resolve omission by configured capacity, never logical runtime batch."""
    if policy is None:
        if max_batch_size > 1:
            return _MULTI_BATCH_OPTIMIZED_POLICY
        return DEFAULT_OPTIMIZED_POLICY
    if policy not in _SUPPORTED_POLICIES:
        raise ValueError(f"unsupported optimized decoder policy: {policy!r}")
    return policy


def _attention_weight_cache_namespace(policy: OptimizedDecoderPolicy) -> str:
    """Return the cache namespace for the effective attention weight dtype."""
    return {
        ttnn.bfloat16: "attention_bf16",
        ttnn.bfloat8_b: "attention_bfp8",
        ttnn.bfloat4_b: "attention_bfp4",
    }[policy.attention_weight_dtype]


def _optimized_attention_decode(
    hidden_states,
    rope_mats,
    weights,
    kv_cache,
    config,
    mesh_config,
    mesh_device,
    program_config,
    transformation_mat,
    position_idx,
    page_table,
    ccl_manager,
    projection_compute_kernel_config,
    qkv_program_config,
    output_program_config,
    output_activation_dtype,
    qkv_memory_config,
    qkv_to_interleaved,
    qkv_weight,
    qkv_input_memory_config,
    separate_qkv_weights,
    separate_qkv_program_configs,
    separate_qkv_to_interleaved,
    qkv_fuse_bias,
    output_input_memory_config,
):
    """Single-device decode with packed QKV and policy-specific device bias handling."""
    _, seq_len, batch_size, hidden_size = hidden_states.shape
    if seq_len != 1:
        raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")

    qkv_input = (
        ttnn.to_memory_config(hidden_states, qkv_input_memory_config)
        if qkv_input_memory_config is not None
        else hidden_states
    )
    if separate_qkv_weights is None:
        xqkv_fused = ttnn.linear(
            qkv_input,
            qkv_weight,
            bias=weights.wqkv_bias if qkv_fuse_bias else None,
            dtype=ttnn.bfloat16,
            memory_config=qkv_memory_config,
            compute_kernel_config=projection_compute_kernel_config,
            program_config=qkv_program_config,
        )
    else:
        # Legal separate-projection A/B for the packed same-input QKV gate.
        # Each projection uses the automatic matmul program for its own output
        # width; concat reconstructs the exact packed [Q,K,V] contract expected
        # by nlp_create_qkv_heads_decode.
        separate_outputs = []
        for (weight, bias), separate_program_config in zip(
            separate_qkv_weights, separate_qkv_program_configs, strict=True
        ):
            output = ttnn.linear(
                qkv_input,
                weight,
                bias=bias if separate_program_config is None else None,
                dtype=ttnn.bfloat16,
                memory_config=qkv_memory_config if separate_program_config is not None else ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=projection_compute_kernel_config,
                program_config=separate_program_config,
            )
            if separate_qkv_to_interleaved:
                sharded_output = output
                output = ttnn.sharded_to_interleaved(sharded_output, ttnn.DRAM_MEMORY_CONFIG)
                sharded_output.deallocate(True)
            separate_outputs.append(output)
        xqkv_fused = ttnn.concat(separate_outputs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for output in separate_outputs:
            output.deallocate(True)
    if qkv_input is not hidden_states:
        qkv_input.deallocate(True)
    if qkv_to_interleaved:
        qkv_sharded = xqkv_fused
        xqkv_fused = ttnn.sharded_to_interleaved(qkv_sharded, ttnn.DRAM_MEMORY_CONFIG)
        qkv_sharded.deallocate(True)
    if not qkv_fuse_bias:
        # Fused bias is broken for this DRAM-sharded matmul family; add it
        # after returning to interleaved DRAM, before the packed head split.
        ttnn.add(xqkv_fused, weights.wqkv_bias, output_tensor=xqkv_fused)

    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)
    head_dim = config.head_dim
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

    grid_size = ttnn.CoreCoord(8, 8)
    batch_grid = ttnn.num_cores_to_corerangeset(batch_size, grid_size, row_wise=True)
    # The inherited KV memory config is sized for configured capacity, while
    # paged_update_cache dispatches exactly one logical user per shard/core.
    # Rebuild only this small activation config from the call's logical batch
    # so a capacity-N decoder preserves valid smaller-batch decode semantics.
    runtime_kv_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    k_cache, v_cache = kv_cache
    tt_k = ttnn.to_memory_config(tt_k, runtime_kv_mem_cfg)
    tt_v = ttnn.to_memory_config(tt_v, runtime_kv_mem_cfg)
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

    padded_heads = ((num_local_heads + 31) // 32) * 32
    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(padded_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    sdpa_kwargs = dict(
        cur_pos_tensor=position_idx,
        sliding_window_size=config.sliding_window,
        attention_sink=weights.decode_sinks,
        scale=config.scaling,
        program_config=program_config.get_decode_sdpa_config(mesh_device),
        compute_kernel_config=program_config.get_compute_kernel_config(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if page_table is not None:
        tt_sdpa_tensor = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            page_table_tensor=page_table,
            **sdpa_kwargs,
        )
    else:
        tt_sdpa_tensor = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            **sdpa_kwargs,
        )
    tt_sdpa_tensor = ttnn.to_memory_config(tt_sdpa_tensor, height_sharded_mem_config)
    tt_q.deallocate(True)

    tt_sdpa_out = ttnn.experimental.nlp_concat_heads_decode(tt_sdpa_tensor, num_heads=num_local_heads)
    tt_sdpa_tensor.deallocate(True)
    if output_activation_dtype is not None and tt_sdpa_out.dtype != output_activation_dtype:
        tt_sdpa_out_to_deallocate = tt_sdpa_out
        tt_sdpa_out = ttnn.typecast(tt_sdpa_out, output_activation_dtype)
        tt_sdpa_out_to_deallocate.deallocate(True)
    output_input = (
        ttnn.to_memory_config(tt_sdpa_out, output_input_memory_config)
        if output_input_memory_config is not None and tt_sdpa_out.memory_config() != output_input_memory_config
        else tt_sdpa_out
    )
    tt_out = ttnn.linear(
        output_input,
        weights.o_proj,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=projection_compute_kernel_config,
        program_config=output_program_config,
    )
    output_input.deallocate(True)
    if output_input is not tt_sdpa_out:
        tt_sdpa_out.deallocate(True)
    tt_out = ttnn.add(tt_out, weights.o_proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_out = ttnn.typecast(tt_out, ttnn.bfloat8_b)

    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32
    padded_hidden = padded_local_hidden * mesh_config.tp if mesh_config.tp > 1 else hidden_size
    tt_out = ttnn.reshape(
        tt_out,
        (1, 1, batch_size, padded_hidden),
        (1, 1, ttnn.TILE_SIZE, padded_hidden),
    )
    if padded_hidden != hidden_size and mesh_config.tp > 1:
        tt_out = ttnn.slice(
            tt_out,
            starts=[0, 0, 0, 0],
            ends=[1, 1, batch_size, hidden_size],
            steps=[1, 1, 1, 1],
        )
        tt_out = ttnn.to_memory_config(tt_out, ttnn.DRAM_MEMORY_CONFIG)
        tt_out = ttnn.to_memory_config(tt_out, ttnn.L1_MEMORY_CONFIG)
    if mesh_config.tp > 1:
        tt_out = ttnn.all_reduce(
            tt_out,
            num_links=ccl_manager.num_links,
            topology=ttnn.Topology.Ring,
            cluster_axis=mesh_config.tp_axis,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    return tt_out


class _OptimizedAttention(Attention):
    """Attention wrapper that owns the optimized decode graph in this stage."""

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
        return _optimized_attention_decode(
            hidden_states,
            rope_mats,
            self.weights,
            cache,
            self.config,
            self.mesh_config,
            self.mesh_device,
            self.program_config,
            transformation_mat,
            position_idx,
            page_table,
            self.ccl_manager,
            self.decode_projection_compute_kernel_config,
            self.decode_qkv_program_config,
            self.decode_output_program_config,
            self.output_activation_dtype,
            self.decode_qkv_memory_config,
            self.decode_qkv_to_interleaved,
            self.decode_wqkv,
            self.decode_qkv_input_memory_config,
            self.decode_separate_qkv_weights,
            self.decode_separate_qkv_program_configs,
            self.decode_separate_qkv_to_interleaved,
            self.decode_qkv_fuse_bias,
            self.decode_output_input_memory_config,
        )


class _L1Router(TopKRouter):
    """Top-K router with its small persistent weight and bias resident in L1."""

    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None):
        self.top_k = hf_config.num_experts_per_tok
        self.num_experts = hf_config.num_local_experts
        self.hidden_dim = hf_config.hidden_size
        self.tensor_cache_path = tensor_cache_path
        self.weight = ttnn.as_tensor(
            state_dict["weight"].transpose(0, 1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight_l1"),
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
        self.bias = ttnn.as_tensor(
            state_dict["bias"].unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "bias_l1"),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # A logical decode token occupies one physical tile row. The fused
        # B=32 router would return 32 logical routes, so retain the sparse
        # linear/top-k path and optimize its data source instead.
        self.use_fused_op = False


class _OptimizedMLP(_FusedMLP):
    """Fused MoE with a persistent L1 router for both decode and prefill."""

    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path, **kwargs):
        super().__init__(mesh_device, hf_config, state_dict, tensor_cache_path, **kwargs)
        old_router = self.router
        old_router.weight.deallocate(True)
        old_router.bias.deallocate(True)
        self.router = _L1Router(
            mesh_device,
            hf_config,
            substate(state_dict, "router"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "router_l1"),
        )


class _DecodeShardedRMSNorm(RMSNorm):
    """Keep decode normalization on a legal ten-way L1 width shard."""

    def __init__(self, *args, enable_decode_sharding=True, **kwargs):
        super().__init__(*args, **kwargs)
        grid = ttnn.CoreGrid(x=10, y=1)
        self.decode_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, 288),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=3,
            block_h=1,
            block_w=9,
            inplace=False,
        )
        self.decode_mode = False
        self.enable_decode_sharding = enable_decode_sharding

    def forward(self, x):
        if not self.decode_mode or not self.enable_decode_sharding:
            return super().forward(x)
        owns_sharded = x.memory_config() != self.decode_memory_config
        sharded = ttnn.to_memory_config(x, self.decode_memory_config) if owns_sharded else x
        normed = ttnn.rms_norm(
            sharded,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=self.decode_program_config,
            memory_config=self.decode_memory_config,
        )
        if owns_sharded:
            sharded.deallocate(True)
        return normed


class OptimizedDecoder(FusedDecoder):
    """Fused decoder contract with an optimized, auditable precision policy."""

    optimization_manifest = FusedDecoder.fusion_manifest + (
        "decode_qkv_linear_bias_fusion",
        "persistent_l1_router_weights",
        "decode_l1_width_sharded_rmsnorm",
        "capacity_selected_packed_qkv_precision",
        "capacity_selected_output_projection_precision",
        "decode_32core_width_sharded_output_projection",
        "paged_bfp8_kv_cache",
        "full_local_bfp4_lofi_experts",
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
        policy: OptimizedDecoderPolicy | None = None,
        create_kv_cache: bool = True,
    ):
        if tuple(mesh_device.shape) != (1, 1):
            raise ValueError(
                "OptimizedDecoder is the single-device optimized stage and requires mesh shape (1, 1); "
                "cross-device parallelism belongs to multichip-decoder"
            )
        if hf_config.hidden_size != 2880 or hf_config.head_dim != 64:
            raise ValueError("Expected the real openai/gpt-oss-120b hidden/head dimensions (2880, 64)")
        layer_type = hf_config.layer_types[layer_idx]
        if layer_type not in {"sliding_attention", "full_attention"}:
            raise ValueError(f"Unsupported GPT-OSS layer type {layer_type!r}")
        advertised_context = int(hf_config.max_position_embeddings)
        max_context_length = advertised_context if max_context_length is None else int(max_context_length)
        if not 1 <= max_context_length <= advertised_context:
            raise ValueError(f"max_context_length must be within [1, {advertised_context}], got {max_context_length}")
        if page_size <= 0 or page_size % ttnn.TILE_SIZE:
            raise ValueError(f"page_size must be a positive tile multiple, got {page_size}")
        requested_policy = policy
        policy = _resolve_policy(policy, max_batch_size)

        local_state = _local_layer_state_dict(state_dict, layer_idx)
        # Only attention weights vary across precision candidates. Reuse the
        # fused-stage norm/MoE cache keys so a precision trial does not
        # needlessly rematerialize all 128 experts; isolate attention keys to
        # prevent stale BF16/BFP8/BFP4 tensor reuse.
        cache_root = str(Path(tensor_cache_path)) if tensor_cache_path is not None else None
        weight_cache_name = _attention_weight_cache_namespace(policy)
        attention_cache_root = (
            str(Path(tensor_cache_path) / weight_cache_name) if tensor_cache_path is not None else None
        )
        mesh_config = MeshConfig(
            mesh_device.shape,
            decode=ModeConfig(tp=1, ep=1, sp=1),
            prefill=ModeConfig(tp=1, ep=1, sp=1),
        )
        ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))
        program_config = GPTOSSAttentionProgramConfig()
        physical_context_length = (
            math.ceil(max_context_length / program_config.decode_k_chunk_size) * program_config.decode_k_chunk_size
        )
        paged_attention_config = PagedAttentionConfig(
            block_size=page_size,
            max_num_blocks=max_batch_size * math.ceil(physical_context_length / page_size),
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
        attention = _OptimizedAttention(
            mesh_device=mesh_device,
            config=attention_config,
            state_dict=substate(local_state, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=program_config,
            layer_idx=layer_idx,
            paged_attention_config=paged_attention_config,
            transformation_mats=rope_setup.get_both_trans_mats(),
            weight_dtype=policy.attention_weight_dtype,
            tensor_cache_path=get_cache_file_name(attention_cache_root, "self_attn"),
            create_kv_cache=create_kv_cache,
        )
        attention.decode_projection_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.decode_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        attention.decode_qkv_program_config = None
        attention.decode_output_program_config = None
        attention.output_activation_dtype = policy.output_activation_dtype
        attention.decode_qkv_memory_config = (
            ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mesh_config.tp > 1 else ttnn.DRAM_MEMORY_CONFIG
        )
        attention.decode_qkv_to_interleaved = False
        attention.decode_wqkv = attention.weights.wqkv
        attention.decode_qkv_input_memory_config = None
        attention.decode_separate_qkv_weights = None
        attention.decode_separate_qkv_program_configs = None
        attention.decode_separate_qkv_to_interleaved = False
        attention.decode_qkv_fuse_bias = True
        attention.decode_output_input_memory_config = None
        output_geometry = policy.output_projection_geometry
        if output_geometry is not None:
            output_k = attention_config.num_heads * attention_config.head_dim
            _validate_output_projection_geometry(output_geometry, k=output_k, n=hf_config.hidden_size)
            output_grid = ttnn.CoreGrid(x=output_geometry.core_grid[0], y=output_geometry.core_grid[1])
            attention.decode_output_input_memory_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, output_k // (output_grid.x * output_grid.y)),
                core_grid=output_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            attention.decode_output_program_config = _make_output_projection_program_config(output_geometry)
        if policy.separate_qkv:
            attention_state = substate(local_state, "self_attn")
            separate_qkv_weights = []
            for projection_name in ("q_proj", "k_proj", "v_proj"):
                projection_state = substate(attention_state, projection_name)
                weight = projection_state["weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0).contiguous()
                bias = projection_state["bias"]
                separate_qkv_weights.append(
                    (
                        ttnn.as_tensor(
                            weight,
                            device=mesh_device,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=policy.attention_weight_dtype,
                            cache_file_name=get_cache_file_name(
                                attention_cache_root, f"decode_separate_{projection_name}_weight"
                            ),
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        ),
                        (
                            None
                            if policy.dram_sharded_qkv
                            else ttnn.as_tensor(
                                bias,
                                device=mesh_device,
                                layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16,
                                cache_file_name=get_cache_file_name(
                                    attention_cache_root, f"decode_separate_{projection_name}_bias"
                                ),
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            )
                        ),
                    )
                )
            attention.decode_separate_qkv_weights = tuple(separate_qkv_weights)
            attention.decode_separate_qkv_program_configs = (None, None, None)
        if policy.dram_sharded_qkv:
            qkv_geometry = policy.dram_qkv_geometry
            if qkv_geometry is None:
                raise ValueError("DRAM-sharded QKV policy requires an explicit geometry")
            qkv_width = (
                attention_config.num_heads * attention_config.head_dim
                + 2 * attention_config.num_kv_heads * attention_config.head_dim
            )
            _validate_dram_qkv_geometry(qkv_geometry, k=hf_config.hidden_size, n=qkv_width)
            dram_grid_size = mesh_device.dram_grid_size()
            dram_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
                    )
                }
            )
            attention.decode_qkv_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            input_cores = ttnn.num_cores_to_corerangeset(
                qkv_geometry.input_cores,
                mesh_device.compute_with_storage_grid_size(),
                row_wise=True,
            )
            attention.decode_qkv_input_memory_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, hf_config.hidden_size // qkv_geometry.input_cores),
                core_grid=input_cores,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            if policy.separate_qkv:
                projection_widths = (
                    attention_config.num_heads * attention_config.head_dim,
                    attention_config.num_kv_heads * attention_config.head_dim,
                    attention_config.num_kv_heads * attention_config.head_dim,
                )
                per_core_ns = policy.dram_separate_qkv_per_core_n
                if per_core_ns is None:
                    raise ValueError("DRAM-sharded separate QKV requires three per-core-N values")
                converted_weights = []
                separate_program_configs = []
                for (weight, bias), projection_width, per_core_n in zip(
                    attention.decode_separate_qkv_weights,
                    projection_widths,
                    per_core_ns,
                    strict=True,
                ):
                    separate_geometry = _DRAMQKVGeometry(
                        "separate", qkv_geometry.input_cores, qkv_geometry.in0_block_w, per_core_n
                    )
                    _validate_dram_qkv_geometry(
                        separate_geometry,
                        k=hf_config.hidden_size,
                        n=projection_width,
                    )
                    if projection_width % (dram_grid.num_cores() * ttnn.TILE_SIZE):
                        raise ValueError(
                            "DRAM-sharded separate QKV width must divide evenly across tile-aligned DRAM banks, "
                            f"got width={projection_width}, banks={dram_grid.num_cores()}"
                        )
                    weight_memory_config = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.DRAM,
                        ttnn.ShardSpec(
                            dram_grid,
                            (hf_config.hidden_size, projection_width // dram_grid.num_cores()),
                            ttnn.ShardOrientation.ROW_MAJOR,
                        ),
                    )
                    converted_weight = ttnn.to_memory_config(weight, weight_memory_config)
                    weight.deallocate(True)
                    converted_weights.append((converted_weight, bias))
                    separate_program_configs.append(
                        ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                            in0_block_w=qkv_geometry.in0_block_w,
                            per_core_M=1,
                            per_core_N=per_core_n,
                            fused_activation=None,
                        )
                    )
                attention.decode_separate_qkv_weights = tuple(converted_weights)
                attention.decode_separate_qkv_program_configs = tuple(separate_program_configs)
                attention.decode_separate_qkv_to_interleaved = True
            else:
                weight_memory_config = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(dram_grid, (hf_config.hidden_size, 640), ttnn.ShardOrientation.ROW_MAJOR),
                )
                attention.decode_wqkv = ttnn.to_memory_config(attention.weights.wqkv, weight_memory_config)
                attention.decode_qkv_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=qkv_geometry.in0_block_w,
                    per_core_M=1,
                    per_core_N=qkv_geometry.per_core_n,
                    fused_activation=None,
                )
                attention.decode_qkv_to_interleaved = True
            attention.decode_qkv_fuse_bias = False
        decoder = cls(
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
                # A full tile of logical users makes the input norm's L1
                # lifetime trace-unstable on Blackhole. Keep the canonical
                # interleaved op only at that boundary; smaller batches retain
                # the measured ten-core decode optimization.
                enable_decode_sharding=max_batch_size < ttnn.TILE_SIZE,
            ),
            post_attention_layernorm=_DecodeShardedRMSNorm(
                mesh_device,
                hf_config,
                substate(local_state, "post_attention_layernorm"),
                tensor_cache_path=get_cache_file_name(cache_root, "post_attention_layernorm"),
                mesh_config=mesh_config,
                # The promoted 32-core output projection also leaves a
                # width-sharded L1 result.  At a full tile of users, following
                # it with another sharded L1 norm makes captured buffer
                # lifetimes overlap across replays on Blackhole.  Keep the
                # canonical interleaved norm only at that capacity boundary.
                enable_decode_sharding=max_batch_size < ttnn.TILE_SIZE,
            ),
            attention=attention,
            mlp=_OptimizedMLP(
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
        decoder.policy = policy
        decoder.requested_policy = requested_policy
        decoder.attention_weight_dtype = policy.attention_weight_dtype
        return decoder

    def _forward(self, *args, is_decode, **kwargs):
        self.input_layernorm.decode_mode = is_decode
        self.post_attention_layernorm.decode_mode = is_decode
        return super()._forward(*args, is_decode=is_decode, **kwargs)


__all__ = [
    "ATTENTION_BF16_CONTROL",
    "ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY",
    "ATTENTION_BFP4_LOFI_POLICY",
    "ATTENTION_BFP8_POLICY",
    "DEFAULT_OPTIMIZED_POLICY",
    "OptimizedDecoder",
    "OptimizedDecoderPolicy",
    "_FULL_LOCAL_CHECKPOINT_REVISION",
    "_attention_weight_cache_namespace",
    "_resolve_policy",
]

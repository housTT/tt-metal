# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device decoder for ``google/gemma-4-26B-A4B-it``.

The functional decoder owns the public tensor/cache semantics and setup-only
weight transformation.  This class deliberately overrides every material
measured projection path so optimized tests cannot silently exercise a
functional matmul fallback.  Candidate precision and sparse geometry are
constructor inputs; the defaults are the strongest cumulative policy selected
by this stage and are changed only after real-weight correctness and traced
batch-1/batch-32 measurements.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    HIDDEN_SIZE,
    MLP_INTERMEDIATE_SIZE,
    MOE_INTERMEDIATE_SIZE,
    NUM_EXPERTS,
    NUM_Q_HEADS,
    PREFILL_MOE_CHUNK_SIZE,
    TILE_SIZE,
    TOP_K_EXPERTS,
    FunctionalDecoder,
    _bounded_cache_fill_plan,
    _detect_layer_prefix,
    _make_decode_height_sharded_memory_config,
    _make_decode_rope_memory_config,
    _make_single_user_cache_update_memory_config,
    _replicate_mapper,
)
from models.common.modules.mlp.mlp_1d import (
    _create_dram_sharded_mem_config,
    _dram_matmul_config,
    _dram_shard_core_grid_k_n,
)
from models.demos.gemma4.tt.experts.operations import apply_geglu
from models.demos.gemma4.tt.experts.weights import ExpertWeights

_DTYPES = {
    "bf16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfp4": ttnn.bfloat4_b,
}
_FIDELITIES = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi4": ttnn.MathFidelity.HiFi4,
}
_RESIDUAL_SHARD_GEOMETRIES = {
    0: None,
    11: (11, 1, 256, 8, 4),
    22: (11, 2, 128, 4, 4),
}
_RESIDUAL_BOUNDARY_COUNTERS = (
    "residual_entry",
    "attention_qkv_input",
    "attention_sdpa_output",
    "attention_o_input",
    "attention_output",
    "router_input",
    "expert_input",
    "expert_output",
    "residual_exit",
)
_DRAM_SHARDED_ROLES = frozenset({"qkv", "o_proj", "mlp_gate", "mlp_up", "mlp_down", "packed_mlp_gate_up"})
_PACKED_EXPERT_WIDTH = 1536
_GRAPH_FUSION_ENV = {
    "folded_router_projection": "GEMMA4_OPT_FOLDED_ROUTER_PROJECTION",
    "shared_ffn_norm": "GEMMA4_OPT_SHARED_FFN_NORM",
    "folded_expert_scale": "GEMMA4_OPT_FOLDED_EXPERT_SCALE",
    "fused_final_scalar": "GEMMA4_OPT_FUSED_FINAL_SCALAR",
}
_ATTENTION_WORKING_GRIDS = {
    8: (8, 1),
    22: (11, 2),
    32: (8, 4),
}


def _candidate_from_env(name: str, default: Any, choices: dict[str, Any]) -> Any:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return choices[value.lower()]
    except KeyError as error:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}") from error


def _bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _optional_int_from_env(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def _prefill_attention_options_from_env() -> dict[str, dict[str, Any]]:
    """Keep the prefill matmul search independent of decode configuration."""
    if not _bool_from_env("GEMMA4_OPT_PREFILL_ATTENTION_2D"):
        return {}
    roles = {role.strip() for role in os.getenv("GEMMA4_OPT_PREFILL_ATTENTION_ROLES", "qkv,o_proj").split(",")}
    if not roles or roles - {"qkv", "o_proj"}:
        raise ValueError("GEMMA4_OPT_PREFILL_ATTENTION_ROLES must select qkv and/or o_proj")
    max_rows = _optional_int_from_env("GEMMA4_OPT_PREFILL_ATTENTION_MAX_ROWS", 1024)
    if max_rows < TILE_SIZE:
        raise ValueError("prefill attention candidate max rows must be at least 32")
    options = {}
    for role in sorted(roles):
        prefix = f"GEMMA4_OPT_PREFILL_{role.upper()}"
        options[role] = {
            "grid_x": _optional_int_from_env(f"{prefix}_GRID_X", None),
            "grid_y": _optional_int_from_env(f"{prefix}_GRID_Y", None),
            "in0_block_w": _optional_int_from_env(f"{prefix}_BLOCK_W", 8 if role == "qkv" else 4),
            "out_subblock_h": _optional_int_from_env(f"{prefix}_SUBBLOCK_H", None),
            "out_subblock_w": _optional_int_from_env(f"{prefix}_SUBBLOCK_W", None),
            "transpose_mcast": _bool_from_env(f"{prefix}_TRANSPOSE_MCAST"),
            "input_l1": _bool_from_env(f"{prefix}_INPUT_L1"),
            "max_rows": max_rows,
        }
    return options


def _prefill_attention_2d_geometry(
    *,
    m: int,
    k: int,
    n: int,
    grid_x: int,
    grid_y: int,
    in0_block_w: int,
    transpose_mcast: bool = False,
    out_subblock_h: int | None = None,
    out_subblock_w: int | None = None,
    destination_tiles: int = 8,
) -> dict[str, Any]:
    """Derive tile-safe 2D blocks, including transposed-multicast geometry."""
    if min(m, k, n, grid_x, grid_y, in0_block_w) < 1 or any(width % TILE_SIZE for width in (m, k, n)):
        raise ValueError("prefill 2D geometry requires positive grid/block values and tile-aligned M/K/N")
    if (k // TILE_SIZE) % in0_block_w:
        raise ValueError(f"prefill in0_block_w={in0_block_w} must divide K tiles={k // TILE_SIZE}")
    m_cores, n_cores = (grid_x, grid_y) if transpose_mcast else (grid_y, grid_x)
    per_core_m = math.ceil(m / (TILE_SIZE * m_cores))
    per_core_n = math.ceil(n / (TILE_SIZE * n_cores))
    subblocks = [
        (h, w)
        for h in range(1, min(per_core_m, destination_tiles) + 1)
        for w in range(1, min(per_core_n, destination_tiles) + 1)
        if per_core_m % h == 0
        and per_core_n % w == 0
        and h * w <= destination_tiles
        and (out_subblock_h is None or h == out_subblock_h)
        and (out_subblock_w is None or w == out_subblock_w)
    ]
    if not subblocks:
        raise ValueError(f"prefill subblock cannot divide per-core output {per_core_m}x{per_core_n} within DST")
    subblock_h, subblock_w = max(subblocks, key=lambda shape: (shape[0] * shape[1], shape[0]))
    return {
        "grid": [grid_x, grid_y],
        "physical_m": m,
        "logical_k": k,
        "logical_n": n,
        "in0_block_w": in0_block_w,
        "per_core_M": per_core_m,
        "per_core_N": per_core_n,
        "out_subblock_h": subblock_h,
        "out_subblock_w": subblock_w,
        "transpose_mcast": transpose_mcast,
        "active_cores": math.ceil(m / (TILE_SIZE * per_core_m)) * math.ceil(n / (TILE_SIZE * per_core_n)),
    }


def _attention_candidate_options_from_env(
    *,
    qkv_working_cores: int = 22,
    qkv_in0_block_w: int | None = 2,
    qkv_out_subblock_w: int | None = 3,
    full_o_working_cores: int = 22,
    full_o_in0_block_w: int | None = 6,
    full_o_out_subblock_w: int | None = 4,
    attention_allow_padding: bool = True,
) -> dict[str, int | bool | None]:
    """Resolve the isolated attention search surface without opening a device."""

    return {
        "qkv_working_cores": _optional_int_from_env("GEMMA4_OPT_QKV_WORKING_CORES", qkv_working_cores),
        "qkv_in0_block_w": _optional_int_from_env("GEMMA4_OPT_QKV_BLOCK_W", qkv_in0_block_w),
        "qkv_out_subblock_w": _optional_int_from_env("GEMMA4_OPT_QKV_OUT_SUBBLOCK_W", qkv_out_subblock_w),
        "full_o_working_cores": _optional_int_from_env("GEMMA4_OPT_FULL_O_WORKING_CORES", full_o_working_cores),
        "full_o_in0_block_w": _optional_int_from_env("GEMMA4_OPT_FULL_O_BLOCK_W", full_o_in0_block_w),
        "full_o_out_subblock_w": _optional_int_from_env("GEMMA4_OPT_FULL_O_OUT_SUBBLOCK_W", full_o_out_subblock_w),
        "attention_allow_padding": _bool_from_env("GEMMA4_OPT_ATTENTION_ALLOW_PADDING", attention_allow_padding),
    }


def _resolve_attention_candidate_geometry(
    *,
    layer_kind: str,
    q_width: int,
    qkv_width: int,
    qkv_working_cores: int = 22,
    qkv_in0_block_w: int | None = 2,
    qkv_out_subblock_w: int | None = 3,
    full_o_working_cores: int = 22,
    full_o_in0_block_w: int | None = 6,
    full_o_out_subblock_w: int | None = 4,
    attention_allow_padding: bool = True,
) -> dict[str, dict[str, Any]]:
    """Resolve legal G8/G22/G32 widths and blocks for attention A/B runs."""

    def resolve_role(
        role: str,
        *,
        logical_k: int,
        logical_n: int,
        cores: int,
        block_w: int | None,
        out_subblock_w: int | None,
    ) -> dict[str, Any]:
        try:
            grid_x, grid_y = _ATTENTION_WORKING_GRIDS[cores]
        except KeyError as error:
            raise ValueError(
                f"{role} working cores must be one of {sorted(_ATTENTION_WORKING_GRIDS)}, got {cores}"
            ) from error
        width_multiple = TILE_SIZE * cores
        padded_k = math.ceil(logical_k / width_multiple) * width_multiple
        padded_n = math.ceil(logical_n / width_multiple) * width_multiple
        if not attention_allow_padding and (padded_k != logical_k or padded_n != logical_n):
            raise ValueError(
                f"{role} G{cores} needs padding ({logical_k}x{logical_n} -> {padded_k}x{padded_n}); "
                "set GEMMA4_OPT_ATTENTION_ALLOW_PADDING=1"
            )
        local_k_tiles = padded_k // TILE_SIZE // cores
        resolved_block_w = (1 if role == "qkv" and cores == 8 else local_k_tiles) if block_w is None else block_w
        if resolved_block_w < 1 or local_k_tiles % resolved_block_w != 0:
            raise ValueError(f"{role} in0_block_w={resolved_block_w} must divide its {local_k_tiles}-tile input shard")
        per_core_n = padded_n // TILE_SIZE // cores
        resolved_subblock_w = (
            next(value for value in (4, 3, 2, 1) if per_core_n % value == 0)
            if out_subblock_w is None
            else out_subblock_w
        )
        if resolved_subblock_w < 1 or resolved_subblock_w > 8 or per_core_n % resolved_subblock_w != 0:
            raise ValueError(
                f"{role} out_subblock_w={resolved_subblock_w} must be at most 8 and divide per_core_N={per_core_n}"
            )
        return {
            "role": role,
            "grid": [grid_x, grid_y],
            "cores": cores,
            "logical_k": logical_k,
            "padded_k": padded_k,
            "logical_n": logical_n,
            "padded_n": padded_n,
            "input_shard_tiles": local_k_tiles,
            "in0_block_w": resolved_block_w,
            "per_core_M": 1,
            "per_core_N": per_core_n,
            "out_subblock_h": 1,
            "out_subblock_w": resolved_subblock_w,
            "input_padding": padded_k - logical_k,
            "output_padding": padded_n - logical_n,
        }

    o_cores = full_o_working_cores if layer_kind == "full_attention" else 8
    return {
        "qkv": resolve_role(
            "qkv",
            logical_k=HIDDEN_SIZE,
            logical_n=qkv_width,
            cores=qkv_working_cores,
            block_w=qkv_in0_block_w,
            out_subblock_w=qkv_out_subblock_w,
        ),
        "o_proj": resolve_role(
            "o_proj",
            logical_k=q_width,
            logical_n=HIDDEN_SIZE,
            cores=o_cores,
            block_w=full_o_in0_block_w if layer_kind == "full_attention" else None,
            out_subblock_w=full_o_out_subblock_w if layer_kind == "full_attention" else None,
        ),
    }


def _resolved_graph_fusion_policy(**defaults: bool) -> dict[str, bool]:
    """Resolve graph-fold candidates independently for focused A/B runs."""

    return {name: _bool_from_env(env_name, defaults.get(name, True)) for name, env_name in _GRAPH_FUSION_ENV.items()}


def _folded_tensor_cache_path(
    tensor_cache_path: str | Path | None,
    *,
    folded_router_projection: bool,
    shared_ffn_norm: bool,
    folded_expert_scale: bool,
) -> Path | None:
    """Keep transformed weights isolated from identically named base caches."""

    if tensor_cache_path is None:
        return None
    cache_root = Path(tensor_cache_path)
    if not (folded_router_projection or shared_ffn_norm or folded_expert_scale):
        return cache_root
    policy = (
        f"router{int(folded_router_projection)}_"
        f"sharedffn{int(shared_ffn_norm)}_"
        f"expertscale{int(folded_expert_scale)}"
    )
    return cache_root / f"optimized_graph_folds_{policy}"


def _prepare_folded_state_dict(
    state_dict: dict[str, Any],
    *,
    layer_idx: int,
    folded_router_projection: bool,
    shared_ffn_norm: bool,
    folded_expert_scale: bool,
) -> dict[str, Any]:
    """Apply algebraic constant folds before TTNN upload and quantization.

    The returned mapping owns only the transformed tensor entries; all other
    values remain shared with the caller.  The caller's mapping and tensors are
    never mutated.
    """

    if not (folded_router_projection or shared_ffn_norm or folded_expert_scale):
        return state_dict

    prefix = _detect_layer_prefix(state_dict, layer_idx)
    prepared = dict(state_dict)

    if folded_router_projection:
        router_scale = state_dict[f"{prefix}.router.scale"].float()
        router_projection = state_dict[f"{prefix}.router.proj.weight"].float()
        prepared[f"{prefix}.router.proj.weight"] = router_projection * router_scale.unsqueeze(0) * (HIDDEN_SIZE**-0.5)

    if shared_ffn_norm:
        dense_scale = state_dict[f"{prefix}.pre_feedforward_layernorm.weight"].float()
        for suffix in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            key = f"{prefix}.{suffix}"
            prepared[key] = state_dict[key].float() * dense_scale.unsqueeze(0)

        expert_scale = state_dict[f"{prefix}.pre_feedforward_layernorm_2.weight"].float()
        expert_gate_up_key = f"{prefix}.experts.gate_up_proj"
        prepared[expert_gate_up_key] = state_dict[expert_gate_up_key].float() * expert_scale.reshape(1, 1, -1)

    if folded_expert_scale:
        per_expert_scale = state_dict[f"{prefix}.router.per_expert_scale"].float()
        expert_down_key = f"{prefix}.experts.down_proj"
        prepared[expert_down_key] = state_dict[expert_down_key].float() * per_expert_scale.reshape(-1, 1, 1)

    return prepared


def _residual_shard_geometry(cores: int) -> tuple[int, int, int, int, int] | None:
    try:
        return _RESIDUAL_SHARD_GEOMETRIES[cores]
    except KeyError as error:
        raise ValueError(
            f"GEMMA4_OPT_RESIDUAL_SHARD_CORES must be one of {sorted(_RESIDUAL_SHARD_GEOMETRIES)}, got {cores}"
        ) from error


def _residual_shard_cores_from_env(default: int = 0) -> int:
    value = os.getenv("GEMMA4_OPT_RESIDUAL_SHARD_CORES", str(default))
    try:
        cores = int(value)
    except ValueError as error:
        raise ValueError(
            f"GEMMA4_OPT_RESIDUAL_SHARD_CORES must be one of {sorted(_RESIDUAL_SHARD_GEOMETRIES)}, got {value!r}"
        ) from error
    _residual_shard_geometry(cores)
    return cores


def _dram_grid(device: Any) -> ttnn.CoreRangeSet:
    size = device.dram_grid_size()
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(size.x - 1, size.y - 1))})


def _matrix_rows(tensor: ttnn.Tensor) -> int:
    rows = 1
    for index in range(len(tensor.shape) - 1):
        rows *= tensor.shape[index]
    return rows


def _load_prefill_expert_weights(
    state_dict: dict[str, Any],
    *,
    layer_idx: int,
    mesh_device: Any,
    dtype: ttnn.DataType,
    tensor_cache_path: str | Path | None,
) -> ExpertWeights:
    """Upload independent prefill experts directly from the HF source dtype."""
    prefix = _detect_layer_prefix(state_dict, layer_idx)
    gate_up = state_dict[f"{prefix}.experts.gate_up_proj"]
    expert_gate = gate_up[:, :MOE_INTERMEDIATE_SIZE, :].transpose(-2, -1).contiguous().unsqueeze(0)
    expert_up = gate_up[:, MOE_INTERMEDIATE_SIZE:, :].transpose(-2, -1).contiguous().unsqueeze(0)
    expert_down = state_dict[f"{prefix}.experts.down_proj"].transpose(-2, -1).contiguous().unsqueeze(0)
    cache_root = Path(tensor_cache_path) if tensor_cache_path is not None else None

    def upload(name: str, source: Any) -> ttnn.Tensor:
        kwargs = {
            "device": mesh_device,
            "layout": ttnn.TILE_LAYOUT,
            "dtype": dtype,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }
        mapper = _replicate_mapper(mesh_device)
        if mapper is not None:
            kwargs["mesh_mapper"] = mapper
        if cache_root is not None:
            kwargs["cache_file_name"] = str(cache_root / f"layer_{layer_idx}" / f"prefill_{name}")
        return ttnn.as_tensor(source, **kwargs)

    return ExpertWeights(
        gate_proj=upload("expert_gate", expert_gate),
        up_proj=upload("expert_up", expert_up),
        down_proj=upload("expert_down", expert_down),
        intermediate_size_per_device=MOE_INTERMEDIATE_SIZE,
    )


def _pack_expert_gate_up(gate: ttnn.Tensor, up: ttnn.Tensor) -> ttnn.Tensor:
    """Pack the same-input expert projections as ``[up, gate]`` in DRAM.

    The 704-wide halves are padded to 768 so the packed 1,536-wide projection
    uses the fused decoder's proven Blackhole sparse geometry. Padding is a
    setup-only weight transform and is sliced away before GeGLU.
    """
    output_dtype = gate.dtype
    if output_dtype == ttnn.bfloat4_b:
        # Device pad does not accept tiled BFP4.  This is a one-time setup
        # transformation; cast the finished packed weight back to BFP4 so the
        # measured sparse matmul still exercises the requested candidate.
        gate = ttnn.typecast(gate, ttnn.bfloat16, memory_config=gate.memory_config())
        up = ttnn.typecast(up, ttnn.bfloat16, memory_config=up.memory_config())
    padding = _PACKED_EXPERT_WIDTH // 2 - MOE_INTERMEDIATE_SIZE
    pad_spec = [(0, 0)] * (len(gate.shape) - 1) + [(0, padding)]
    padded_up = ttnn.pad(up, pad_spec, 0.0)
    padded_gate = ttnn.pad(gate, pad_spec, 0.0)
    packed = ttnn.concat(
        [padded_up, padded_gate],
        dim=-1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    padded_up.deallocate(True)
    padded_gate.deallocate(True)
    if packed.dtype != output_dtype:
        packed = ttnn.typecast(packed, output_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return packed


def _pad_attention_weight(
    weight: ttnn.Tensor,
    *,
    padded_k: int,
    padded_n: int,
) -> ttnn.Tensor:
    """Apply mathematically inert setup-only K/N padding to a linear weight."""

    logical_k, logical_n = weight.shape[-2], weight.shape[-1]
    if logical_k == padded_k and logical_n == padded_n:
        return weight
    if logical_k > padded_k or logical_n > padded_n:
        raise ValueError(f"attention padding cannot shrink a {logical_k}x{logical_n} weight to {padded_k}x{padded_n}")
    output_dtype = weight.dtype
    source = weight
    if output_dtype == ttnn.bfloat4_b:
        source = ttnn.typecast(weight, ttnn.bfloat16, memory_config=weight.memory_config())
    pad_spec = [(0, 0)] * (len(source.shape) - 2) + [(0, padded_k - logical_k), (0, padded_n - logical_n)]
    padded = ttnn.pad(source, pad_spec, 0.0)
    if source is not weight:
        source.deallocate(True)
    weight.deallocate(True)
    if padded.dtype != output_dtype:
        uncast = padded
        padded = ttnn.typecast(uncast, output_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        uncast.deallocate(True)
    return padded


def _dram_sharded_weight_and_config(
    weight: ttnn.Tensor,
    *,
    device: Any,
    block_w: int | None = None,
    workers_per_bank: int = 1,
) -> tuple[
    ttnn.Tensor,
    ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
    ttnn.MemoryConfig,
    ttnn.MemoryConfig,
]:
    """Create a decode-only DRAM-sharded weight and its sub-tile-M config."""
    k, n = weight.shape[-2], weight.shape[-1]
    dram_cores = device.dram_grid_size().x
    memory_config = _create_dram_sharded_mem_config(
        k=k,
        n=n,
        dram_grid=_dram_grid(device),
        dram_cores=dram_cores,
    )
    if workers_per_bank == 1:
        grid = _dram_shard_core_grid_k_n(k, n)
    else:
        grid_coord = _rectangular_grid(device, device.dram_grid_size().x * workers_per_bank)
        grid = ttnn.CoreGrid(x=grid_coord.x, y=grid_coord.y)
    num_cores = grid.x * grid.y
    input_memory_config = ttnn.create_sharded_memory_config(
        (TILE_SIZE, k // num_cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    output_memory_config = ttnn.create_sharded_memory_config(
        (TILE_SIZE, math.ceil(n / (TILE_SIZE * num_cores)) * TILE_SIZE),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    config = _dram_matmul_config(m=TILE_SIZE, k=k, n=n, num_cores=num_cores)
    if block_w is not None or workers_per_bank != 1:
        config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=config.in0_block_w if block_w is None else block_w,
            per_core_M=config.per_core_M,
            per_core_N=config.per_core_N,
            fused_activation=config.fused_activation,
            num_workers_per_dram_bank=workers_per_bank,
        )
    return ttnn.to_memory_config(weight, memory_config), config, input_memory_config, output_memory_config


def _pad_dram_weight_for_readers(
    weight: ttnn.Tensor,
    *,
    dram_cores: int,
    workers_per_bank: int,
    block_w: int | None,
) -> tuple[ttnn.Tensor, int | None, int | None]:
    """Pad output tiles so each DRAM bank can divide work over its readers.

    The DRAM-sharded kernel requires the per-bank output shard width to be a
    multiple of ``workers_per_bank``.  Padding is setup-only; ``_linear``
    slices the padded output on device before returning it to the decoder.
    """
    if workers_per_bank == 1:
        return weight, None, None
    logical_k, logical_n = weight.shape[-2], weight.shape[-1]
    worker_cores = dram_cores * workers_per_bank
    local_k_multiple = worker_cores * (block_w or 1) * TILE_SIZE
    output_multiple = worker_cores * TILE_SIZE
    padded_k = math.ceil(logical_k / local_k_multiple) * local_k_multiple
    padded_n = math.ceil(logical_n / output_multiple) * output_multiple
    if padded_k == logical_k and padded_n == logical_n:
        return weight, None, None
    pad_spec = [(0, 0)] * len(weight.shape)
    pad_spec[-2] = (0, padded_k - logical_k)
    pad_spec[-1] = (0, padded_n - logical_n)
    return (
        ttnn.pad(weight, pad_spec, 0.0),
        padded_k if padded_k != logical_k else None,
        logical_n if padded_n != logical_n else None,
    )


def _r22_dram_geometry(
    *, k: int, n: int, dram_banks: int, input_cores: int, readers: int, block_w: int | None
) -> dict[str, int]:
    """Resolve storage and reader tile geometry without changing logical K/N."""
    if readers not in (1, 2, 3) or dram_banks < 1:
        raise ValueError("R22 DRAM projections require positive bank count and one, two, or three readers")
    if input_cores not in (8, 11, 22, 32) or k % (TILE_SIZE * input_cores) or n % TILE_SIZE:
        raise ValueError(f"R22 DRAM projection needs clean tiled K storage, got {k=} {n=} {input_cores=}")
    local_k_tiles = k // (TILE_SIZE * input_cores)
    block_w = local_k_tiles if block_w is None else block_w
    if block_w < 1 or local_k_tiles % block_w:
        raise ValueError(f"{block_w=} must divide the {local_k_tiles}-tile input shard")
    n_tiles = n // TILE_SIZE
    bank_tiles = readers * math.ceil(n_tiles / (dram_banks * readers))
    # Match the factory's removal of banks containing padding only, then its
    # exact multi-reader partition check. Weight storage padding is not N padding.
    effective_banks = dram_banks - (bank_tiles * dram_banks - n_tiles) // bank_tiles
    reader_tiles = math.ceil(n_tiles / (effective_banks * readers))
    if readers > 1 and bank_tiles != readers * reader_tiles:
        raise ValueError(f"R22 DRAM reader partition is unsupported: {bank_tiles=} {readers=} {reader_tiles=}")
    output_tiles = math.ceil(n_tiles / 22)
    if math.ceil(n_tiles / output_tiles) != 22:
        raise ValueError(f"logical output {n} cannot use the required 22 storage cores")
    return {
        "logical_k": k,
        "logical_n": n,
        "input_cores": input_cores,
        "input_shard_tiles": local_k_tiles,
        "in0_block_w": block_w,
        "dram_banks": dram_banks,
        "effective_dram_banks": effective_banks,
        "readers_per_bank": readers,
        "reader_tiles": reader_tiles,
        "weight_bank_tiles": bank_tiles,
        "weight_storage_width": bank_tiles * dram_banks * TILE_SIZE,
        "per_core_M": 1,
        "per_core_N": output_tiles,
        "output_storage_cores": 22,
        "output_storage_width": output_tiles * 22 * TILE_SIZE,
    }


def _r22_dram_weight_and_config(
    weight: ttnn.Tensor, *, device: Any, input_cores: int, readers: int, block_w: int | None
) -> tuple[ttnn.Tensor, Any, ttnn.MemoryConfig, ttnn.MemoryConfig, dict[str, Any]]:
    """Build a decode candidate with independent activation and reader grids."""
    if readers > 1 and device.arch() != ttnn.device.Arch.BLACKHOLE:
        raise ValueError("multiple DRAM readers are supported only on Blackhole")
    geometry = _r22_dram_geometry(
        k=weight.shape[-2],
        n=weight.shape[-1],
        dram_banks=device.dram_grid_size().x,
        input_cores=input_cores,
        readers=readers,
        block_w=block_w,
    )
    grid_x, grid_y = {8: (8, 1), 11: (11, 1), 22: (11, 2), 32: (8, 4)}[input_cores]
    available_grid = device.compute_with_storage_grid_size()
    if grid_x > available_grid.x or grid_y > available_grid.y or available_grid.x != 11:
        raise ValueError("R22 DRAM output synthesis requires the validated 11-column worker grid")
    input_config = _width_sharded_memory_config(geometry["logical_k"], ttnn.CoreGrid(x=grid_x, y=grid_y))
    output_config = _width_sharded_memory_config(geometry["output_storage_width"], ttnn.CoreGrid(x=11, y=2))
    weight_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            _dram_grid(device),
            [geometry["logical_k"], geometry["weight_bank_tiles"] * TILE_SIZE],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=geometry["in0_block_w"],
        per_core_M=1,
        per_core_N=geometry["per_core_N"],
        num_workers_per_dram_bank=readers,
    )
    sharded_weight = ttnn.to_memory_config(weight, weight_config)
    if tuple(sharded_weight.shape) != tuple(weight.shape) or sharded_weight.dtype != weight.dtype:
        raise AssertionError("DRAM storage padding changed the logical weight shape or dtype")
    tile_bytes = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[weight.dtype]
    geometry.update(
        program_class=type(program_config).__name__,
        program_config=str(program_config),
        weight_dtype=str(sharded_weight.dtype),
        weight_memory_config=str(sharded_weight.memory_config()),
        input_memory_config=str(input_config),
        output_memory_config=str(output_config),
        reader_physical_row_bytes=geometry["reader_tiles"] * tile_bytes,
        supported_batch=1,
    )
    return sharded_weight, program_config, input_config, output_config, geometry


def _compute_config(
    device: Any,
    *,
    fidelity: ttnn.MathFidelity,
    fp32_dest_acc_en: bool = False,
) -> ttnn.DeviceComputeKernelConfig:
    arch = device.arch() if hasattr(device, "arch") else ttnn.device.GetDefaultDevice().arch()
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=not fp32_dest_acc_en,
    )


def _rectangular_grid(device: Any, num_cores: int) -> ttnn.CoreCoord:
    grid = device.compute_with_storage_grid_size()
    for height in range(min(grid.y, num_cores), 0, -1):
        if num_cores % height == 0 and num_cores // height <= grid.x:
            return ttnn.CoreCoord(num_cores // height, height)
    raise ValueError(f"cannot place {num_cores} cores on {grid.x}x{grid.y}")


def _width_sharded_memory_config(width: int, grid: ttnn.CoreGrid) -> ttnn.MemoryConfig:
    if width % (TILE_SIZE * grid.num_cores) != 0:
        raise ValueError(f"{width=} must divide exactly over {grid.num_cores} tile-aligned width shards")
    return ttnn.create_sharded_memory_config(
        (TILE_SIZE, width // grid.num_cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _width_sharded_linear_program_config(
    *,
    k: int,
    n: int,
    grid: ttnn.CoreGrid,
    in0_block_w: int | None = None,
    out_subblock_w: int | None = None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    if k % (TILE_SIZE * grid.num_cores) != 0 or n % (TILE_SIZE * grid.num_cores) != 0:
        raise ValueError(f"1D sharded linear requires exact tiled K/N division, got {k=} {n=} {grid.num_cores=}")
    local_k_tiles = k // TILE_SIZE // grid.num_cores
    in0_block_w = local_k_tiles if in0_block_w is None else in0_block_w
    if local_k_tiles % in0_block_w != 0:
        raise ValueError(f"{in0_block_w=} must divide the local input shard width {local_k_tiles}")
    per_core_n = n // TILE_SIZE // grid.num_cores
    if out_subblock_w is None:
        out_subblock_w = next(value for value in (4, 3, 2, 1) if per_core_n % value == 0)
    if out_subblock_w < 1 or out_subblock_w > 8 or per_core_n % out_subblock_w != 0:
        raise ValueError(f"{out_subblock_w=} must be at most 8 and divide {per_core_n=}")
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _optimized_sparse_prefill_config(
    device: Any,
    *,
    n: int,
    groups: int,
    requested_per_core_n: int,
    in0_block_w: int,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """Use the Blackhole grid while accounting for sparse group replication."""
    n_tiles = math.ceil(n / TILE_SIZE)
    grid = device.compute_with_storage_grid_size()
    max_cores = (grid.x * grid.y) // groups
    legal_per_core_n = [
        value
        for value in range(max(1, requested_per_core_n), n_tiles + 1)
        if n_tiles % value == 0 and n_tiles // value <= max_cores
    ]
    if not legal_per_core_n:
        raise ValueError(
            f"no sparse prefill config for n_tiles={n_tiles}, groups={groups}, "
            f"requested_per_core_n={requested_per_core_n}"
        )
    per_core_n = legal_per_core_n[0]
    projection_cores = n_tiles // per_core_n
    # sparse_matmul replicates the projection grid for each 32-token sparse
    # group. The program grid must therefore cover every replicated block.
    core_grid = _rectangular_grid(device, projection_cores * groups)
    out_subblock_w = 2 if per_core_n % 2 == 0 else 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=core_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _optimized_sparse_decode_config(
    device: Any,
    *,
    n: int,
    per_core_n: int,
    in0_block_w: int,
    out_subblock_w: int | None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """Place sparse decode across the full architecture grid."""
    n_tiles = math.ceil(n / TILE_SIZE)
    if n_tiles % per_core_n != 0:
        raise ValueError(f"{per_core_n=} must divide {n_tiles=} for sparse decode")
    num_cores = n_tiles // per_core_n
    core_grid = _rectangular_grid(device, num_cores)
    if out_subblock_w is None:
        out_subblock_w = next(value for value in (4, 2, 1) if per_core_n % value == 0)
    if out_subblock_w < 1 or per_core_n % out_subblock_w != 0:
        raise ValueError(f"{out_subblock_w=} must divide {per_core_n=} for sparse decode")
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=core_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


class OptimizedDecoder(FunctionalDecoder):
    """Gemma-4 decoder with explicit optimized precision/configuration paths."""

    def __init__(
        self,
        *,
        attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2,
        prefill_full_attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2,
        full_attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        residual_full_attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2,
        mlp_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        expert_gate_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        expert_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        expert_gate_in0_block_w: int = 22,
        expert_down_in0_block_w: int = 11,
        expert_gate_per_core_n: int = 1,
        expert_down_per_core_n: int = 1,
        expert_gate_out_subblock_w: int | None = None,
        expert_down_out_subblock_w: int | None = None,
        expert_decode_input_l1: bool = True,
        prefill_expert_input_l1: bool = False,
        prefill_expert_chunk_size: int = TILE_SIZE,
        prefill_expert_per_core_n: int = 2,
        prefill_expert_gate_in0_block_w: int = 44,
        prefill_expert_down_in0_block_w: int = 11,
        prefill_expert_tail_per_core_n: int = 11,
        prefill_expert_tail_in0_block_w: int = 1,
        prefill_routed_active: bool = True,
        dense_decode_dram_sharded: bool = False,
        packed_dense_gate_up: bool = True,
        r22_dram_sharded: bool = False,
        r22_packed_dense_gate_up: bool = False,
        packed_expert_decode_gate_up: bool = True,
        packed_expert_prefill_gate_up: bool = False,
        folded_router_projection: bool = True,
        shared_ffn_norm: bool = True,
        folded_expert_scale: bool = True,
        fused_final_scalar: bool = True,
        dram_in0_block_w: int | None = None,
        dram_workers_per_bank: int = 1,
        dram_sharded_roles: tuple[str, ...] = (),
        residual_shard_cores: int = 22,
        qkv_working_cores: int = 22,
        qkv_in0_block_w: int | None = 2,
        qkv_out_subblock_w: int | None = 3,
        full_o_working_cores: int = 22,
        full_o_in0_block_w: int | None = 6,
        full_o_out_subblock_w: int | None = 4,
        attention_allow_padding: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        resolved_prefill_attention_fidelity = (
            attention_math_fidelity
            if self.layer_kind.name == "sliding_attention"
            else prefill_full_attention_math_fidelity
        )
        resolved_decode_attention_fidelity = (
            attention_math_fidelity
            if self.layer_kind.name == "sliding_attention"
            else (residual_full_attention_math_fidelity if residual_shard_cores else full_attention_math_fidelity)
        )
        self.prefill_attention_compute_config = _compute_config(
            self.mesh_device, fidelity=resolved_prefill_attention_fidelity
        )
        self.decode_attention_compute_config = _compute_config(
            self.mesh_device, fidelity=resolved_decode_attention_fidelity
        )
        self.mlp_compute_config = _compute_config(self.mesh_device, fidelity=mlp_math_fidelity)
        self.expert_gate_compute_config = _compute_config(
            self.mesh_device,
            fidelity=expert_gate_math_fidelity,
            fp32_dest_acc_en=expert_gate_math_fidelity == ttnn.MathFidelity.HiFi4,
        )
        self.expert_compute_config = _compute_config(self.mesh_device, fidelity=expert_math_fidelity)
        self.expert_gate_in0_block_w = expert_gate_in0_block_w
        self.expert_down_in0_block_w = expert_down_in0_block_w
        self.expert_gate_per_core_n = expert_gate_per_core_n
        self.expert_down_per_core_n = expert_down_per_core_n
        self.expert_gate_out_subblock_w = expert_gate_out_subblock_w
        self.expert_down_out_subblock_w = expert_down_out_subblock_w
        self.expert_decode_input_l1 = expert_decode_input_l1
        self.prefill_expert_input_l1 = prefill_expert_input_l1
        self.prefill_expert_chunk_size = prefill_expert_chunk_size
        self.prefill_expert_per_core_n = prefill_expert_per_core_n
        self.prefill_expert_gate_in0_block_w = prefill_expert_gate_in0_block_w
        self.prefill_expert_down_in0_block_w = prefill_expert_down_in0_block_w
        self.prefill_expert_tail_per_core_n = prefill_expert_tail_per_core_n
        self.prefill_expert_tail_in0_block_w = prefill_expert_tail_in0_block_w
        self.prefill_routed_active = prefill_routed_active
        self.dense_decode_dram_sharded = dense_decode_dram_sharded
        self.packed_dense_gate_up = packed_dense_gate_up
        self.r22_dram_sharded = r22_dram_sharded
        self.r22_packed_dense_gate_up = r22_packed_dense_gate_up
        self.r22_projection_runtime = {}
        self.packed_expert_decode_gate_up = packed_expert_decode_gate_up
        self.packed_expert_prefill_gate_up = packed_expert_prefill_gate_up
        self.folded_router_projection = folded_router_projection
        self.shared_ffn_norm = shared_ffn_norm
        self.folded_expert_scale = folded_expert_scale
        self.fused_final_scalar = fused_final_scalar
        self.layer_scalar_value = None
        self.dram_in0_block_w = dram_in0_block_w
        self.dram_workers_per_bank = dram_workers_per_bank
        self.dram_sharded_roles = frozenset(dram_sharded_roles)
        self.qkv_working_cores = qkv_working_cores
        self.qkv_in0_block_w = qkv_in0_block_w
        self.qkv_out_subblock_w = qkv_out_subblock_w
        self.full_o_working_cores = full_o_working_cores
        self.full_o_in0_block_w = full_o_in0_block_w
        self.full_o_out_subblock_w = full_o_out_subblock_w
        self.attention_allow_padding = attention_allow_padding
        self.prefill_attention_options = _prefill_attention_options_from_env()
        self.prefill_attention_program_configs = {}
        self.prefill_attention_runtime = {}
        self.routing_row_major = _bool_from_env("GEMMA4_OPT_ROUTING_ROW_MAJOR", True)
        if self.routing_row_major and (not folded_expert_scale or residual_shard_cores == 0):
            raise ValueError("row-major routing candidate requires folded expert scales and the sharded residual path")
        self.routing_runtime = {}
        self.decode_attention_weights = {}
        self.optimized_path_counters = {
            "prefill_attention": 0,
            "decode_attention": 0,
            "dense_mlp": 0,
            "expert_decode": 0,
            "expert_prefill": 0,
            "packed_expert_decode": 0,
            "packed_expert_prefill": 0,
            "folded_router_projection": 0,
            "shared_ffn_norm": 0,
            "folded_expert_scale": 0,
            "fused_final_scalar": 0,
            "attention_qkv_input_padding": 0,
            "attention_qkv_output_slice": 0,
            "attention_o_input_padding": 0,
            "attention_o_output_slice": 0,
            "residual_chain_decode": 0,
            "r22_packed_dense": 0,
            "r22_dense_input_reshards": 0,
            "prefill_2d_qkv": 0,
            "prefill_2d_o_proj": 0,
            "routing_row_major_scatter": 0,
            "routing_row_major_metadata": 0,
            "routing_score_tilize": 0,
            **{f"r22_dram_{role}": 0 for role in _DRAM_SHARDED_ROLES},
            **{f"r22_dram_input_reshard_{role}": 0 for role in _DRAM_SHARDED_ROLES},
            **{name: 0 for name in _RESIDUAL_BOUNDARY_COUNTERS},
        }
        self._configure_residual_chain(residual_shard_cores)

    def _configure_residual_chain(self, residual_shard_cores: int) -> None:
        geometry = _residual_shard_geometry(residual_shard_cores)
        self.residual_shard_cores = residual_shard_cores
        self.residual_memory_config = None
        self.residual_norm_program_config = None
        self.residual_intermediate_memory_config = None
        self.residual_dense_program_configs = {}
        self.attention_sharded_memory_configs = {}
        self.attention_sharded_program_configs = {}
        self.attention_runtime_geometry = {}
        self.attention_batch32_sharded_memory_configs = {}
        self.attention_batch32_sharded_program_configs = {}
        self.attention_batch32_runtime_geometry = {}
        if geometry is None:
            return

        grid_x, grid_y, shard_width, norm_block_w, norm_subblock_w = geometry
        available_grid = self.mesh_device.compute_with_storage_grid_size()
        if grid_x > available_grid.x or grid_y > available_grid.y:
            raise ValueError(
                f"R{residual_shard_cores} needs a {grid_x}x{grid_y} worker grid, "
                f"but the device exposes {available_grid.x}x{available_grid.y}"
            )
        residual_grid = ttnn.CoreGrid(x=grid_x, y=grid_y)
        self.residual_memory_config = _width_sharded_memory_config(HIDDEN_SIZE, residual_grid)
        if HIDDEN_SIZE // residual_shard_cores != shard_width:
            raise AssertionError("residual shard geometry does not cover the hidden width exactly")
        self.residual_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid_x, grid_y],
            subblock_w=norm_subblock_w,
            block_h=1,
            block_w=norm_block_w,
            inplace=False,
        )

        self.residual_intermediate_memory_config = _width_sharded_memory_config(
            MLP_INTERMEDIATE_SIZE,
            residual_grid,
        )
        self.residual_dense_program_configs = {
            "mlp_gate": _width_sharded_linear_program_config(
                k=HIDDEN_SIZE,
                n=MLP_INTERMEDIATE_SIZE,
                grid=residual_grid,
            ),
            "mlp_up": _width_sharded_linear_program_config(
                k=HIDDEN_SIZE,
                n=MLP_INTERMEDIATE_SIZE,
                grid=residual_grid,
            ),
            "mlp_down": _width_sharded_linear_program_config(
                k=MLP_INTERMEDIATE_SIZE,
                n=HIDDEN_SIZE,
                grid=residual_grid,
            ),
        }
        if self.r22_packed_dense_gate_up:
            self.residual_dense_program_configs["packed_mlp_gate_up"] = _width_sharded_linear_program_config(
                k=HIDDEN_SIZE,
                n=2 * MLP_INTERMEDIATE_SIZE,
                grid=residual_grid,
                in0_block_w=_optional_int_from_env("GEMMA4_OPT_R22_PACKED_DENSE_BLOCK_W", 4),
                out_subblock_w=_optional_int_from_env("GEMMA4_OPT_R22_PACKED_DENSE_OUT_SUBBLOCK_W", 3),
            )
            self.r22_packed_dense_memory_config = _width_sharded_memory_config(2 * MLP_INTERMEDIATE_SIZE, residual_grid)

        q_width = self.layer_kind.q_width
        qkv_width = self.layer_kind.qkv_width
        self.attention_runtime_geometry = _resolve_attention_candidate_geometry(
            layer_kind=self.layer_kind.name,
            q_width=q_width,
            qkv_width=qkv_width,
            qkv_working_cores=self.qkv_working_cores,
            qkv_in0_block_w=self.qkv_in0_block_w,
            qkv_out_subblock_w=self.qkv_out_subblock_w,
            full_o_working_cores=self.full_o_working_cores,
            full_o_in0_block_w=self.full_o_in0_block_w,
            full_o_out_subblock_w=self.full_o_out_subblock_w,
            attention_allow_padding=self.attention_allow_padding,
        )

        def build_attention_configs(runtime_geometry):
            qkv_geometry = runtime_geometry["qkv"]
            o_geometry = runtime_geometry["o_proj"]
            qkv_grid = ttnn.CoreGrid(x=qkv_geometry["grid"][0], y=qkv_geometry["grid"][1])
            o_grid = ttnn.CoreGrid(x=o_geometry["grid"][0], y=o_geometry["grid"][1])
            memory_configs = {
                "qkv_input": _width_sharded_memory_config(qkv_geometry["padded_k"], qkv_grid),
                "qkv_output": _width_sharded_memory_config(qkv_geometry["padded_n"], qkv_grid),
                "o_input": _width_sharded_memory_config(o_geometry["padded_k"], o_grid),
                "o_output": _width_sharded_memory_config(o_geometry["padded_n"], o_grid),
            }
            program_configs = {
                "qkv": _width_sharded_linear_program_config(
                    k=qkv_geometry["padded_k"],
                    n=qkv_geometry["padded_n"],
                    grid=qkv_grid,
                    in0_block_w=qkv_geometry["in0_block_w"],
                    out_subblock_w=qkv_geometry["out_subblock_w"],
                ),
                "o_proj": _width_sharded_linear_program_config(
                    k=o_geometry["padded_k"],
                    n=o_geometry["padded_n"],
                    grid=o_grid,
                    in0_block_w=o_geometry["in0_block_w"],
                    out_subblock_w=o_geometry["out_subblock_w"],
                ),
            }
            for role in ("qkv", "o_proj"):
                runtime_geometry[role].update(
                    {
                        "program_class": "MatmulMultiCoreReuseMultiCast1DProgramConfig",
                        "input_memory_config": str(memory_configs["qkv_input" if role == "qkv" else "o_input"]),
                        "output_memory_config": str(memory_configs["qkv_output" if role == "qkv" else "o_output"]),
                    }
                )
            return memory_configs, program_configs

        self.attention_sharded_memory_configs, self.attention_sharded_program_configs = build_attention_configs(
            self.attention_runtime_geometry
        )
        self.attention_batch32_runtime_geometry = _resolve_attention_candidate_geometry(
            layer_kind=self.layer_kind.name,
            q_width=q_width,
            qkv_width=qkv_width,
            qkv_working_cores=8,
            qkv_in0_block_w=1,
            qkv_out_subblock_w=4,
            full_o_working_cores=8,
            full_o_in0_block_w=32,
            full_o_out_subblock_w=1,
            attention_allow_padding=False,
        )
        (
            self.attention_batch32_sharded_memory_configs,
            self.attention_batch32_sharded_program_configs,
        ) = build_attention_configs(self.attention_batch32_runtime_geometry)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: Any,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        attention_weight_dtype: ttnn.DataType | None = None,
        qkv_weight_dtype: ttnn.DataType | None = None,
        o_proj_weight_dtype: ttnn.DataType | None = None,
        mlp_weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        mlp_down_weight_dtype: ttnn.DataType | None = None,
        prefill_expert_weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        expert_weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        expert_gate_weight_dtype: ttnn.DataType | None = ttnn.bfloat4_b,
        expert_down_weight_dtype: ttnn.DataType | None = None,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        tensor_cache_path: str | Path | None = None,
        attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2,
        prefill_full_attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2,
        full_attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        residual_full_attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2,
        mlp_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        expert_gate_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        expert_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        expert_gate_in0_block_w: int = 22,
        expert_down_in0_block_w: int = 11,
        expert_gate_per_core_n: int = 1,
        expert_down_per_core_n: int = 1,
        expert_gate_out_subblock_w: int | None = None,
        expert_down_out_subblock_w: int | None = None,
        expert_decode_input_l1: bool = True,
        prefill_expert_input_l1: bool = False,
        prefill_expert_chunk_size: int = TILE_SIZE,
        prefill_expert_per_core_n: int = 2,
        prefill_expert_gate_in0_block_w: int = 44,
        prefill_expert_down_in0_block_w: int = 11,
        prefill_expert_tail_per_core_n: int = 11,
        prefill_expert_tail_in0_block_w: int = 1,
        prefill_routed_active: bool = True,
        dense_decode_dram_sharded: bool = False,
        packed_dense_gate_up: bool = True,
        r22_dram_sharded: bool | None = None,
        r22_packed_dense_gate_up: bool | None = None,
        packed_expert_decode_gate_up: bool = True,
        packed_expert_prefill_gate_up: bool = False,
        folded_router_projection: bool = True,
        shared_ffn_norm: bool = True,
        folded_expert_scale: bool = True,
        fused_final_scalar: bool = True,
        dram_in0_block_w: int | None = None,
        dram_workers_per_bank: int = 1,
        dram_sharded_roles: tuple[str, ...] = (),
        residual_shard_cores: int = 22,
        qkv_working_cores: int = 22,
        qkv_in0_block_w: int | None = 2,
        qkv_out_subblock_w: int | None = 3,
        full_o_working_cores: int = 22,
        full_o_in0_block_w: int | None = 6,
        full_o_out_subblock_w: int | None = 4,
        attention_allow_padding: bool = True,
        **kwargs: Any,
    ) -> "OptimizedDecoder":
        layer_type = hf_config.layer_types[layer_idx]
        weight_dtype = _candidate_from_env("GEMMA4_OPT_WEIGHT_DTYPE", weight_dtype, _DTYPES)
        attention_weight_dtype = _candidate_from_env(
            "GEMMA4_OPT_ATTENTION_WEIGHT_DTYPE",
            attention_weight_dtype,
            _DTYPES,
        )
        if attention_weight_dtype is None:
            attention_weight_dtype = ttnn.bfloat16 if layer_type == "sliding_attention" else ttnn.bfloat8_b
        qkv_weight_dtype = _candidate_from_env("GEMMA4_OPT_QKV_WEIGHT_DTYPE", qkv_weight_dtype, _DTYPES)
        o_proj_weight_dtype = _candidate_from_env("GEMMA4_OPT_O_PROJ_WEIGHT_DTYPE", o_proj_weight_dtype, _DTYPES)
        if qkv_weight_dtype is None:
            qkv_weight_dtype = attention_weight_dtype
        if o_proj_weight_dtype is None:
            o_proj_weight_dtype = attention_weight_dtype
        mlp_weight_dtype = _candidate_from_env(
            "GEMMA4_OPT_MLP_WEIGHT_DTYPE",
            mlp_weight_dtype,
            _DTYPES,
        )
        mlp_down_weight_dtype = _candidate_from_env(
            "GEMMA4_OPT_MLP_DOWN_WEIGHT_DTYPE",
            mlp_down_weight_dtype,
            _DTYPES,
        )
        if mlp_down_weight_dtype is None:
            mlp_down_weight_dtype = mlp_weight_dtype
        expert_weight_dtype = _candidate_from_env(
            "GEMMA4_OPT_EXPERT_WEIGHT_DTYPE",
            expert_weight_dtype,
            _DTYPES,
        )
        expert_gate_weight_dtype = _candidate_from_env(
            "GEMMA4_OPT_EXPERT_GATE_WEIGHT_DTYPE",
            expert_gate_weight_dtype,
            _DTYPES,
        )
        expert_down_weight_dtype = _candidate_from_env(
            "GEMMA4_OPT_EXPERT_DOWN_WEIGHT_DTYPE",
            expert_down_weight_dtype,
            _DTYPES,
        )
        if expert_gate_weight_dtype is None:
            expert_gate_weight_dtype = expert_weight_dtype
        if expert_down_weight_dtype is None:
            expert_down_weight_dtype = expert_weight_dtype
        prefill_expert_weight_dtype = _candidate_from_env(
            "GEMMA4_OPT_PREFILL_EXPERT_WEIGHT_DTYPE",
            prefill_expert_weight_dtype,
            _DTYPES,
        )
        attention_math_fidelity = _candidate_from_env(
            "GEMMA4_OPT_ATTENTION_FIDELITY",
            attention_math_fidelity,
            _FIDELITIES,
        )
        prefill_full_attention_math_fidelity = _candidate_from_env(
            "GEMMA4_OPT_PREFILL_FULL_ATTENTION_FIDELITY",
            prefill_full_attention_math_fidelity,
            _FIDELITIES,
        )
        full_attention_math_fidelity = _candidate_from_env(
            "GEMMA4_OPT_FULL_ATTENTION_FIDELITY",
            full_attention_math_fidelity,
            _FIDELITIES,
        )
        residual_full_attention_math_fidelity = _candidate_from_env(
            "GEMMA4_OPT_RESIDUAL_FULL_ATTENTION_FIDELITY",
            residual_full_attention_math_fidelity,
            _FIDELITIES,
        )
        mlp_math_fidelity = _candidate_from_env(
            "GEMMA4_OPT_MLP_FIDELITY",
            mlp_math_fidelity,
            _FIDELITIES,
        )
        expert_math_fidelity = _candidate_from_env(
            "GEMMA4_OPT_EXPERT_FIDELITY",
            expert_math_fidelity,
            _FIDELITIES,
        )
        expert_gate_math_fidelity = _candidate_from_env(
            "GEMMA4_OPT_EXPERT_GATE_FIDELITY",
            expert_gate_math_fidelity,
            _FIDELITIES,
        )
        expert_gate_in0_block_w = int(os.getenv("GEMMA4_OPT_EXPERT_GATE_BLOCK_W", expert_gate_in0_block_w))
        expert_down_in0_block_w = int(os.getenv("GEMMA4_OPT_EXPERT_DOWN_BLOCK_W", expert_down_in0_block_w))
        expert_gate_per_core_n = int(os.getenv("GEMMA4_OPT_EXPERT_GATE_PER_CORE_N", expert_gate_per_core_n))
        expert_down_per_core_n = int(os.getenv("GEMMA4_OPT_EXPERT_DOWN_PER_CORE_N", expert_down_per_core_n))
        env_gate_subblock = os.getenv("GEMMA4_OPT_EXPERT_GATE_OUT_SUBBLOCK_W")
        if env_gate_subblock is not None:
            expert_gate_out_subblock_w = int(env_gate_subblock)
        env_down_subblock = os.getenv("GEMMA4_OPT_EXPERT_DOWN_OUT_SUBBLOCK_W")
        if env_down_subblock is not None:
            expert_down_out_subblock_w = int(env_down_subblock)
        expert_decode_input_l1 = _bool_from_env("GEMMA4_OPT_EXPERT_DECODE_INPUT_L1", expert_decode_input_l1)
        prefill_expert_input_l1 = _bool_from_env("GEMMA4_OPT_PREFILL_EXPERT_INPUT_L1", prefill_expert_input_l1)
        prefill_expert_chunk_size = int(os.getenv("GEMMA4_OPT_PREFILL_EXPERT_CHUNK_SIZE", prefill_expert_chunk_size))
        prefill_expert_per_core_n = int(os.getenv("GEMMA4_OPT_PREFILL_EXPERT_PER_CORE_N", prefill_expert_per_core_n))
        prefill_expert_gate_in0_block_w = int(
            os.getenv("GEMMA4_OPT_PREFILL_EXPERT_GATE_BLOCK_W", prefill_expert_gate_in0_block_w)
        )
        prefill_expert_down_in0_block_w = int(
            os.getenv("GEMMA4_OPT_PREFILL_EXPERT_DOWN_BLOCK_W", prefill_expert_down_in0_block_w)
        )
        prefill_expert_tail_per_core_n = int(
            os.getenv("GEMMA4_OPT_PREFILL_EXPERT_TAIL_PER_CORE_N", prefill_expert_tail_per_core_n)
        )
        prefill_expert_tail_in0_block_w = int(
            os.getenv("GEMMA4_OPT_PREFILL_EXPERT_TAIL_BLOCK_W", prefill_expert_tail_in0_block_w)
        )
        prefill_routed_active = _bool_from_env("GEMMA4_OPT_PREFILL_ROUTED_ACTIVE", prefill_routed_active)
        dense_decode_dram_sharded = _bool_from_env("GEMMA4_OPT_DENSE_DECODE_DRAM_SHARDED", dense_decode_dram_sharded)
        residual_shard_cores = _residual_shard_cores_from_env(residual_shard_cores)
        env_roles = os.getenv("GEMMA4_OPT_DRAM_SHARDED_ROLES")
        auto_selected_mlp_dram = False
        if env_roles is not None:
            dram_sharded_roles = tuple(role.strip() for role in env_roles.split(",") if role.strip())
        packed_dense_gate_up = _bool_from_env("GEMMA4_OPT_PACKED_DENSE_GATE_UP", packed_dense_gate_up)
        env_r22_dram = os.getenv("GEMMA4_OPT_R22_DRAM_SHARDED")
        env_r22_packed = os.getenv("GEMMA4_OPT_R22_PACKED_DENSE_GATE_UP")
        if env_r22_dram is None:
            r22_dram_sharded = residual_shard_cores == 22 if r22_dram_sharded is None else r22_dram_sharded
        else:
            r22_dram_sharded = _bool_from_env("GEMMA4_OPT_R22_DRAM_SHARDED", bool(r22_dram_sharded))
        if env_r22_packed is None:
            r22_packed_dense_gate_up = (
                env_r22_dram is None and r22_dram_sharded
                if r22_packed_dense_gate_up is None
                else r22_packed_dense_gate_up
            )
        else:
            r22_packed_dense_gate_up = _bool_from_env(
                "GEMMA4_OPT_R22_PACKED_DENSE_GATE_UP", bool(r22_packed_dense_gate_up)
            )
        if not dram_sharded_roles:
            if r22_dram_sharded:
                attention_roles = ("o_proj",) if layer_type == "sliding_attention" else ()
                dense_roles = (
                    ("packed_mlp_gate_up", "mlp_down")
                    if r22_packed_dense_gate_up
                    else ("mlp_gate", "mlp_up", "mlp_down")
                )
                dram_sharded_roles = (*attention_roles, *dense_roles)
            elif residual_shard_cores == 0:
                # Retain the nonresidual advisor challenger for explicit R0
                # comparisons; the selected R22 path above is layer-aware.
                dram_sharded_roles = ("o_proj", "packed_mlp_gate_up", "mlp_down")
                auto_selected_mlp_dram = True
        invalid_roles = set(dram_sharded_roles) - _DRAM_SHARDED_ROLES
        if invalid_roles:
            raise ValueError(
                f"invalid DRAM-sharded roles {sorted(invalid_roles)}; choose from {sorted(_DRAM_SHARDED_ROLES)}"
            )
        if dense_decode_dram_sharded:
            dram_sharded_roles = tuple(sorted(_DRAM_SHARDED_ROLES))
        if (r22_dram_sharded or r22_packed_dense_gate_up) and residual_shard_cores != 22:
            raise ValueError("R22 projection candidates require GEMMA4_OPT_RESIDUAL_SHARD_CORES=22")
        if r22_dram_sharded:
            if not dram_sharded_roles:
                raise ValueError("GEMMA4_OPT_R22_DRAM_SHARDED requires explicit GEMMA4_OPT_DRAM_SHARDED_ROLES")
            inactive_roles = ({"mlp_gate", "mlp_up"} if r22_packed_dense_gate_up else {"packed_mlp_gate_up"}) & set(
                dram_sharded_roles
            )
            if inactive_roles:
                raise ValueError(
                    f"selected DRAM roles would be unused by the dense packing policy: {sorted(inactive_roles)}"
                )
        packed_expert_decode_gate_up = _bool_from_env(
            "GEMMA4_OPT_PACKED_EXPERT_DECODE_GATE_UP", packed_expert_decode_gate_up
        )
        packed_expert_prefill_gate_up = _bool_from_env(
            "GEMMA4_OPT_PACKED_EXPERT_PREFILL_GATE_UP", packed_expert_prefill_gate_up
        )
        graph_fusion_policy = _resolved_graph_fusion_policy(
            folded_router_projection=folded_router_projection,
            shared_ffn_norm=shared_ffn_norm,
            folded_expert_scale=folded_expert_scale,
            fused_final_scalar=fused_final_scalar,
        )
        folded_router_projection = graph_fusion_policy["folded_router_projection"]
        shared_ffn_norm = graph_fusion_policy["shared_ffn_norm"]
        folded_expert_scale = graph_fusion_policy["folded_expert_scale"]
        fused_final_scalar = graph_fusion_policy["fused_final_scalar"]
        env_dram_block_w = os.getenv("GEMMA4_OPT_DRAM_BLOCK_W")
        if env_dram_block_w is not None:
            dram_in0_block_w = int(env_dram_block_w)
        dram_workers_per_bank = int(os.getenv("GEMMA4_OPT_DRAM_WORKERS_PER_BANK", dram_workers_per_bank))
        if dram_workers_per_bank not in (1, 2, 3):
            raise ValueError("GEMMA4_OPT_DRAM_WORKERS_PER_BANK must be 1, 2, or 3")
        attention_candidate_options = _attention_candidate_options_from_env(
            qkv_working_cores=qkv_working_cores,
            qkv_in0_block_w=qkv_in0_block_w,
            qkv_out_subblock_w=qkv_out_subblock_w,
            full_o_working_cores=full_o_working_cores,
            full_o_in0_block_w=full_o_in0_block_w,
            full_o_out_subblock_w=full_o_out_subblock_w,
            attention_allow_padding=attention_allow_padding,
        )
        prepared_state_dict = _prepare_folded_state_dict(
            state_dict,
            layer_idx=layer_idx,
            folded_router_projection=folded_router_projection,
            shared_ffn_norm=shared_ffn_norm,
            folded_expert_scale=folded_expert_scale,
        )
        prepared_tensor_cache_path = _folded_tensor_cache_path(
            tensor_cache_path,
            folded_router_projection=folded_router_projection,
            shared_ffn_norm=shared_ffn_norm,
            folded_expert_scale=folded_expert_scale,
        )
        decoder = super().from_state_dict(
            prepared_state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            weight_dtype=weight_dtype,
            expert_weight_dtype=expert_weight_dtype,
            activation_dtype=activation_dtype,
            tensor_cache_path=prepared_tensor_cache_path,
            attention_math_fidelity=attention_math_fidelity,
            mlp_math_fidelity=mlp_math_fidelity,
            expert_gate_math_fidelity=expert_gate_math_fidelity,
            expert_math_fidelity=expert_math_fidelity,
            expert_gate_in0_block_w=expert_gate_in0_block_w,
            expert_down_in0_block_w=expert_down_in0_block_w,
            expert_gate_per_core_n=expert_gate_per_core_n,
            expert_down_per_core_n=expert_down_per_core_n,
            expert_gate_out_subblock_w=expert_gate_out_subblock_w,
            expert_down_out_subblock_w=expert_down_out_subblock_w,
            expert_decode_input_l1=expert_decode_input_l1,
            prefill_expert_input_l1=prefill_expert_input_l1,
            prefill_expert_chunk_size=prefill_expert_chunk_size,
            prefill_expert_per_core_n=prefill_expert_per_core_n,
            prefill_expert_gate_in0_block_w=prefill_expert_gate_in0_block_w,
            prefill_expert_down_in0_block_w=prefill_expert_down_in0_block_w,
            prefill_routed_active=prefill_routed_active,
            dense_decode_dram_sharded=dense_decode_dram_sharded,
            packed_dense_gate_up=packed_dense_gate_up,
            r22_dram_sharded=r22_dram_sharded,
            r22_packed_dense_gate_up=r22_packed_dense_gate_up,
            packed_expert_decode_gate_up=packed_expert_decode_gate_up,
            packed_expert_prefill_gate_up=packed_expert_prefill_gate_up,
            folded_router_projection=folded_router_projection,
            shared_ffn_norm=shared_ffn_norm,
            folded_expert_scale=folded_expert_scale,
            fused_final_scalar=fused_final_scalar,
            dram_in0_block_w=dram_in0_block_w,
            dram_workers_per_bank=dram_workers_per_bank,
            dram_sharded_roles=dram_sharded_roles,
            residual_shard_cores=residual_shard_cores,
            residual_full_attention_math_fidelity=residual_full_attention_math_fidelity,
            **attention_candidate_options,
            **kwargs,
        )
        resolved_prefill_attention_fidelity = (
            attention_math_fidelity
            if decoder.layer_kind.name == "sliding_attention"
            else prefill_full_attention_math_fidelity
        )
        resolved_decode_attention_fidelity = (
            attention_math_fidelity
            if decoder.layer_kind.name == "sliding_attention"
            else (residual_full_attention_math_fidelity if residual_shard_cores else full_attention_math_fidelity)
        )
        decoder.prefill_attention_compute_config = _compute_config(
            mesh_device, fidelity=resolved_prefill_attention_fidelity
        )
        decoder.decode_attention_compute_config = _compute_config(
            mesh_device, fidelity=resolved_decode_attention_fidelity
        )
        decoder.mlp_compute_config = _compute_config(mesh_device, fidelity=mlp_math_fidelity)
        decoder.expert_gate_compute_config = _compute_config(
            mesh_device,
            fidelity=expert_gate_math_fidelity,
            fp32_dest_acc_en=expert_gate_math_fidelity == ttnn.MathFidelity.HiFi4,
        )
        decoder.expert_compute_config = _compute_config(mesh_device, fidelity=expert_math_fidelity)
        decoder.expert_gate_in0_block_w = expert_gate_in0_block_w
        decoder.expert_down_in0_block_w = expert_down_in0_block_w
        decoder.expert_gate_per_core_n = expert_gate_per_core_n
        decoder.expert_down_per_core_n = expert_down_per_core_n
        decoder.expert_gate_out_subblock_w = expert_gate_out_subblock_w
        decoder.expert_down_out_subblock_w = expert_down_out_subblock_w
        decoder.expert_decode_input_l1 = expert_decode_input_l1
        decoder.prefill_expert_input_l1 = prefill_expert_input_l1
        decoder.prefill_expert_chunk_size = prefill_expert_chunk_size
        decoder.prefill_expert_per_core_n = prefill_expert_per_core_n
        decoder.prefill_expert_gate_in0_block_w = prefill_expert_gate_in0_block_w
        decoder.prefill_expert_down_in0_block_w = prefill_expert_down_in0_block_w
        decoder.prefill_expert_tail_per_core_n = prefill_expert_tail_per_core_n
        decoder.prefill_expert_tail_in0_block_w = prefill_expert_tail_in0_block_w
        decoder.prefill_routed_active = prefill_routed_active
        decoder.dense_decode_dram_sharded = dense_decode_dram_sharded
        decoder.packed_dense_gate_up = packed_dense_gate_up
        # FunctionalDecoder.from_state_dict accepts optimization kwargs for
        # compatibility but its constructor does not retain these R22-only
        # candidate switches. Restore them on the optimized instance just as
        # we do for the other scoped runtime policies below.
        decoder.r22_dram_sharded = r22_dram_sharded
        decoder.r22_packed_dense_gate_up = r22_packed_dense_gate_up
        decoder.packed_expert_decode_gate_up = packed_expert_decode_gate_up
        decoder.packed_expert_prefill_gate_up = packed_expert_prefill_gate_up
        decoder.folded_router_projection = folded_router_projection
        decoder.shared_ffn_norm = shared_ffn_norm
        decoder.folded_expert_scale = folded_expert_scale
        decoder.fused_final_scalar = fused_final_scalar
        decoder.layer_scalar_value = None
        decoder.dram_in0_block_w = dram_in0_block_w
        decoder.dram_workers_per_bank = dram_workers_per_bank
        decoder.dram_sharded_roles = frozenset(dram_sharded_roles)
        for name, value in attention_candidate_options.items():
            setattr(decoder, name, value)
        decoder._configure_residual_chain(residual_shard_cores)
        batch32_expert_gate = batch32_expert_up = None
        if expert_gate_weight_dtype != ttnn.bfloat8_b:
            batch32_expert_gate = ttnn.typecast(
                decoder.weights.expert_gate,
                ttnn.bfloat8_b,
                memory_config=decoder.weights.expert_gate.memory_config(),
            )
            batch32_expert_up = ttnn.typecast(
                decoder.weights.expert_up,
                ttnn.bfloat8_b,
                memory_config=decoder.weights.expert_up.memory_config(),
            )
        decoder.weights = replace(
            decoder.weights,
            qkv=ttnn.typecast(
                decoder.weights.qkv,
                qkv_weight_dtype,
                memory_config=decoder.weights.qkv.memory_config(),
            ),
            o_proj=ttnn.typecast(
                decoder.weights.o_proj,
                o_proj_weight_dtype,
                memory_config=decoder.weights.o_proj.memory_config(),
            ),
            mlp_gate=ttnn.typecast(
                decoder.weights.mlp_gate,
                mlp_weight_dtype,
                memory_config=decoder.weights.mlp_gate.memory_config(),
            ),
            mlp_up=ttnn.typecast(
                decoder.weights.mlp_up,
                mlp_weight_dtype,
                memory_config=decoder.weights.mlp_up.memory_config(),
            ),
            mlp_down=ttnn.typecast(
                decoder.weights.mlp_down,
                mlp_down_weight_dtype,
                memory_config=decoder.weights.mlp_down.memory_config(),
            ),
            expert_gate=ttnn.typecast(
                decoder.weights.expert_gate,
                expert_gate_weight_dtype,
                memory_config=decoder.weights.expert_gate.memory_config(),
            ),
            expert_up=ttnn.typecast(
                decoder.weights.expert_up,
                expert_gate_weight_dtype,
                memory_config=decoder.weights.expert_up.memory_config(),
            ),
            expert_down=ttnn.typecast(
                decoder.weights.expert_down,
                expert_down_weight_dtype,
                memory_config=decoder.weights.expert_down.memory_config(),
            ),
        )
        qkv_geometry = decoder.attention_runtime_geometry.get("qkv")
        o_geometry = decoder.attention_runtime_geometry.get("o_proj")
        decoder.decode_attention_weights = {
            "qkv": decoder.weights.qkv,
            "o_proj": decoder.weights.o_proj,
        }
        # Multi-user decode retains the unpadded G8 control geometry, which is
        # the real-weight/per-user-PCC-safe policy at batch 32.
        decoder.decode_attention_weights_batch32 = dict(decoder.decode_attention_weights)
        if qkv_geometry is not None and o_geometry is not None:
            for role, role_geometry, role_dtype in (
                ("qkv", qkv_geometry, qkv_weight_dtype),
                ("o_proj", o_geometry, o_proj_weight_dtype),
            ):
                if role_geometry["input_padding"] or role_geometry["output_padding"]:
                    logical_weight = getattr(decoder.weights, role)
                    candidate_weight = ttnn.clone(
                        logical_weight,
                        dtype=logical_weight.dtype,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    decoder.decode_attention_weights[role] = _pad_attention_weight(
                        candidate_weight,
                        padded_k=role_geometry["padded_k"],
                        padded_n=role_geometry["padded_n"],
                    )
                role_geometry["weight_dtype"] = str(role_dtype)
                role_geometry["activation_dtype"] = str(activation_dtype)
                role_geometry["math_fidelity"] = str(resolved_decode_attention_fidelity)
                role_geometry["decode_weight_padding"] = bool(
                    role_geometry["input_padding"] or role_geometry["output_padding"]
                )
        decoder.expert_weights = _load_prefill_expert_weights(
            prepared_state_dict,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            dtype=prefill_expert_weight_dtype,
            tensor_cache_path=prepared_tensor_cache_path,
        )
        decoder.decode_packed_expert_gate_up = None
        decoder.decode_packed_expert_gate_up_batch32 = None
        decoder.batch32_expert_gate = (
            decoder.weights.expert_gate if batch32_expert_gate is None else batch32_expert_gate
        )
        decoder.batch32_expert_up = decoder.weights.expert_up if batch32_expert_up is None else batch32_expert_up
        decoder.prefill_packed_expert_gate_up = None
        if packed_expert_decode_gate_up:
            decoder.decode_packed_expert_gate_up = _pack_expert_gate_up(
                decoder.weights.expert_gate,
                decoder.weights.expert_up,
            )
            if batch32_expert_gate is None:
                decoder.decode_packed_expert_gate_up_batch32 = decoder.decode_packed_expert_gate_up
            else:
                decoder.decode_packed_expert_gate_up_batch32 = _pack_expert_gate_up(
                    batch32_expert_gate,
                    batch32_expert_up,
                )
                batch32_expert_gate.deallocate(True)
                batch32_expert_up.deallocate(True)
                decoder.batch32_expert_gate = None
                decoder.batch32_expert_up = None
        if packed_expert_prefill_gate_up:
            decoder.prefill_packed_expert_gate_up = _pack_expert_gate_up(
                decoder.expert_weights.gate_proj,
                decoder.expert_weights.up_proj,
            )
        decoder.packed_mlp_gate_up = None
        if (packed_dense_gate_up and residual_shard_cores == 0) or r22_packed_dense_gate_up:
            decoder.packed_mlp_gate_up = ttnn.concat(
                [decoder.weights.mlp_gate, decoder.weights.mlp_up],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        decoder.decode_dram_weights = {}
        decoder.decode_dram_configs = {}
        decoder.decode_dram_input_configs = {}
        decoder.decode_dram_output_configs = {}
        decoder.decode_dram_padded_input_widths = {}
        decoder.decode_dram_logical_output_widths = {}
        decoder.decode_routing_zero_base = None
        if decoder.routing_row_major:
            # Trace capture cannot initialize a row-major tensor.  Allocate the
            # immutable scatter base during setup and reuse its stable address.
            decoder.decode_routing_zero_base = ttnn.zeros(
                (1, 1, 1, NUM_EXPERTS),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if dram_sharded_roles:
            candidates = {
                "qkv": decoder.weights.qkv,
                "o_proj": decoder.weights.o_proj,
                "mlp_gate": decoder.weights.mlp_gate,
                "mlp_up": decoder.weights.mlp_up,
                "mlp_down": decoder.weights.mlp_down,
            }
            if decoder.packed_mlp_gate_up is not None:
                candidates["packed_mlp_gate_up"] = decoder.packed_mlp_gate_up
            for name, weight in candidates.items():
                if name not in dram_sharded_roles:
                    continue
                role_block_w = os.getenv(f"GEMMA4_OPT_DRAM_BLOCK_W_{name.upper()}")
                role_workers = int(os.getenv(f"GEMMA4_OPT_DRAM_WORKERS_PER_BANK_{name.upper()}", dram_workers_per_bank))
                if role_workers not in (1, 2, 3):
                    raise ValueError(f"DRAM workers for {name} must be 1, 2, or 3")
                if (
                    role_block_w is None
                    and dram_in0_block_w is None
                    and auto_selected_mlp_dram
                    and name in {"packed_mlp_gate_up", "mlp_down"}
                ):
                    role_block_w = "11"
                elif role_block_w is None and dram_in0_block_w is None and auto_selected_mlp_dram and name == "o_proj":
                    # Block width eight was the best legal, correctness-clean
                    # O-projection geometry at the primary batch-1 decode point.
                    role_block_w = "8"
                resolved_block_w = int(role_block_w) if role_block_w is not None else dram_in0_block_w
                if r22_dram_sharded:
                    input_cores = _optional_int_from_env(
                        f"GEMMA4_OPT_DRAM_INPUT_CORES_{name.upper()}", 8 if name == "o_proj" else 22
                    )
                    sharded_weight, config, input_config, output_config, runtime = _r22_dram_weight_and_config(
                        weight,
                        device=mesh_device,
                        input_cores=input_cores,
                        readers=role_workers,
                        block_w=resolved_block_w,
                    )
                    decoder.decode_dram_weights[name] = sharded_weight
                    decoder.decode_dram_configs[name] = config
                    decoder.decode_dram_input_configs[name] = input_config
                    decoder.decode_dram_output_configs[name] = output_config
                    decoder.r22_projection_runtime[name] = runtime
                    continue
                weight, padded_input_width, logical_output_width = _pad_dram_weight_for_readers(
                    weight,
                    dram_cores=mesh_device.dram_grid_size().x,
                    workers_per_bank=role_workers,
                    block_w=resolved_block_w,
                )
                sharded_weight, config, input_config, output_config = _dram_sharded_weight_and_config(
                    weight,
                    device=mesh_device,
                    block_w=resolved_block_w,
                    workers_per_bank=role_workers,
                )
                decoder.decode_dram_weights[name] = sharded_weight
                decoder.decode_dram_configs[name] = config
                decoder.decode_dram_input_configs[name] = input_config
                decoder.decode_dram_output_configs[name] = output_config
                if padded_input_width is not None:
                    decoder.decode_dram_padded_input_widths[name] = padded_input_width
                if logical_output_width is not None:
                    decoder.decode_dram_logical_output_widths[name] = logical_output_width

        if r22_packed_dense_gate_up and "packed_mlp_gate_up" in decoder.decode_dram_weights:
            # The DRAM-sharded packed tensor is the sole B1 consumer. Prefill
            # and B32 retain the original separate gate/up weights, so release
            # this setup-only interleaved packing source after conversion.
            decoder.packed_mlp_gate_up.deallocate(True)
            decoder.packed_mlp_gate_up = None

        obsolete_weights = {}
        if folded_router_projection:
            decoder.weights.router_scale.deallocate(True)
            obsolete_weights["router_scale"] = None
        if shared_ffn_norm:
            decoder.weights.pre_ff_ln.deallocate(True)
            decoder.weights.pre_ff_ln_2.deallocate(True)
            obsolete_weights["pre_ff_ln"] = None
            obsolete_weights["pre_ff_ln_2"] = None
        if folded_expert_scale:
            decoder.weights.router_per_expert_scale.deallocate(True)
            obsolete_weights["router_per_expert_scale"] = None
        if fused_final_scalar:
            prefix = _detect_layer_prefix(state_dict, layer_idx)
            decoder.layer_scalar_value = float(state_dict[f"{prefix}.layer_scalar"].item())
            decoder.weights.layer_scalar.deallocate(True)
            obsolete_weights["layer_scalar"] = None
        cumulative_graph_folds = (
            folded_router_projection and shared_ffn_norm and folded_expert_scale and fused_final_scalar
        )
        if cumulative_graph_folds and packed_expert_decode_gate_up:
            # The packed decode tensor owns the only decode-time copy needed by
            # the cumulative candidate.  Prefill has its separately quantized
            # expert tensors, so retaining these unpacked decode operands would
            # consume two large persistent buffers without a reachable user.
            decoder.weights.expert_gate.deallocate(True)
            decoder.weights.expert_up.deallocate(True)
            obsolete_weights["expert_gate"] = None
            obsolete_weights["expert_up"] = None
        if obsolete_weights:
            decoder.weights = replace(decoder.weights, **obsolete_weights)
        return decoder

    def _activation_dtype_for(self, role: str) -> ttnn.DataType:
        """Return a role-specific activation dtype when a subclass supplies one."""

        return getattr(self, f"{role}_activation_dtype", self.activation_dtype)

    def _linear(
        self,
        x: ttnn.Tensor,
        weight_name: str,
        *,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ) -> ttnn.Tensor:
        if not getattr(self, "_in_decode_forward", False) and weight_name in self.prefill_attention_options:
            return self._prefill_attention_linear(x, weight_name, compute_kernel_config=compute_kernel_config)
        decode_candidate = self._use_decode_dram_weight(x, weight_name)
        weight = self.decode_dram_weights[weight_name] if decode_candidate else getattr(self.weights, weight_name)
        kwargs = {}
        if decode_candidate:
            kwargs["program_config"] = self.decode_dram_configs[weight_name]
            padded_input_width = self.decode_dram_padded_input_widths.get(weight_name)
            if padded_input_width is not None:
                pad_spec = [(0, 0)] * len(x.shape)
                pad_spec[-1] = (0, padded_input_width - x.shape[-1])
                x = ttnn.pad(x, pad_spec, 0.0)
            x = ttnn.to_memory_config(x, self.decode_dram_input_configs[weight_name], dtype=x.dtype)
            kwargs["memory_config"] = self.decode_dram_output_configs[weight_name]
        result = ttnn.linear(
            x,
            weight,
            dtype=self._activation_dtype_for("attention" if weight_name in {"qkv", "o_proj"} else "dense_mlp"),
            memory_config=kwargs.pop("memory_config", ttnn.DRAM_MEMORY_CONFIG),
            compute_kernel_config=compute_kernel_config,
            **kwargs,
        )
        if decode_candidate:
            result = ttnn.sharded_to_interleaved(result, ttnn.DRAM_MEMORY_CONFIG)
            logical_output_width = self.decode_dram_logical_output_widths.get(weight_name)
            if logical_output_width is not None:
                starts = [0] * len(result.shape)
                ends = list(result.shape)
                ends[-1] = logical_output_width
                result = ttnn.slice(
                    result,
                    starts,
                    ends,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
        return result

    def _prefill_attention_linear(
        self, x: ttnn.Tensor, role: str, *, compute_kernel_config: ttnn.DeviceComputeKernelConfig
    ) -> ttnn.Tensor:
        options = self.prefill_attention_options[role]
        weight = getattr(self.weights, role)
        rows = math.prod(list(x.padded_shape)[:-1])
        key = (role, rows, weight.shape[-2], weight.shape[-1])
        selected = rows <= options["max_rows"]
        kwargs = {}
        geometry = {"selected": selected, "physical_m": rows, "max_rows": options["max_rows"]}
        working = x
        if selected:
            if key not in self.prefill_attention_program_configs:
                available = self.mesh_device.compute_with_storage_grid_size()
                grid_x = options["grid_x"] if options["grid_x"] is not None else available.x
                grid_y = options["grid_y"] if options["grid_y"] is not None else available.y
                if grid_x > available.x or grid_y > available.y:
                    raise ValueError("prefill attention candidate grid exceeds available workers")
                resolved = _prefill_attention_2d_geometry(
                    m=rows,
                    k=weight.shape[-2],
                    n=weight.shape[-1],
                    grid_x=grid_x,
                    grid_y=grid_y,
                    in0_block_w=options["in0_block_w"],
                    transpose_mcast=options["transpose_mcast"],
                    out_subblock_h=options["out_subblock_h"],
                    out_subblock_w=options["out_subblock_w"],
                    destination_tiles=4 if compute_kernel_config.fp32_dest_acc_en else 8,
                )
                config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
                    in0_block_w=resolved["in0_block_w"],
                    per_core_M=resolved["per_core_M"],
                    per_core_N=resolved["per_core_N"],
                    out_subblock_h=resolved["out_subblock_h"],
                    out_subblock_w=resolved["out_subblock_w"],
                    transpose_mcast=resolved["transpose_mcast"],
                    fuse_batch=False,
                    fused_activation=None,
                )
                self.prefill_attention_program_configs[key] = (config, resolved)
            config, resolved = self.prefill_attention_program_configs[key]
            geometry.update(resolved)
            kwargs["program_config"] = config
            if options["input_l1"]:
                working = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=x.dtype)
            self.optimized_path_counters[f"prefill_2d_{role}"] += 1
        output = ttnn.linear(
            working,
            weight,
            dtype=self._activation_dtype_for("attention"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
            **kwargs,
        )
        if tuple(output.shape) != (*tuple(x.shape)[:-1], weight.shape[-1]):
            raise AssertionError("prefill attention candidate changed logical output shape")
        geometry.update(
            program_config=str(kwargs.get("program_config", "automatic_above_candidate_max_rows")),
            input_memory_config=str(working.memory_config()),
            input_dtype=str(working.dtype),
            input_l1=selected and options["input_l1"],
            weight_dtype=str(weight.dtype),
            weight_memory_config=str(weight.memory_config()),
            output_dtype=str(output.dtype),
            output_memory_config=str(output.memory_config()),
            compute_kernel_config=str(compute_kernel_config),
            logical_input_shape=list(x.shape),
        )
        self.prefill_attention_runtime[f"{role}_m{rows}"] = geometry
        if working is not x:
            working.deallocate(True)
        return output

    def _use_decode_dram_weight(self, x: ttnn.Tensor, weight_name: str) -> bool:
        """Select decode-tuned weights only inside the explicit decode phase."""
        return (
            getattr(self, "_in_decode_forward", False)
            and weight_name in self.decode_dram_weights
            and _matrix_rows(x) <= TILE_SIZE
        )

    def _use_r22_dram_weight(self, x: ttnn.Tensor, role: str) -> bool:
        """Keep the measured multi-user G8 policy independent of B1 candidates."""
        return (
            self.r22_dram_sharded
            and self.residual_shard_cores == 22
            # Some decode composites expose a tile-padded M=32 even for one
            # logical user. The call boundary owns the public batch contract.
            and getattr(self, "_decode_logical_batch", x.shape[-2]) == 1
            and self._use_decode_dram_weight(x, role)
        )

    def _r22_dram_linear(
        self, x: ttnn.Tensor, role: str, *, compute_kernel_config: ttnn.DeviceComputeKernelConfig
    ) -> ttnn.Tensor:
        if not self._use_r22_dram_weight(x, role):
            raise AssertionError(f"R22 DRAM role {role} was called outside its selected B1 decode contract")
        runtime = self.r22_projection_runtime[role]
        if x.shape[-1] != runtime["logical_k"]:
            raise AssertionError(f"R22 DRAM {role} input width differs from its logical weight K")
        input_config = self.decode_dram_input_configs[role]
        working = x
        if x.memory_config() != input_config:
            working = ttnn.to_memory_config(x, input_config, dtype=x.dtype)
            self.optimized_path_counters[f"r22_dram_input_reshard_{role}"] += 1
        weight = self.decode_dram_weights[role]
        config = self.decode_dram_configs[role]
        output = ttnn.linear(
            working,
            weight,
            dtype=self._activation_dtype_for("attention" if role in {"qkv", "o_proj"} else "dense_mlp"),
            program_config=config,
            memory_config=self.decode_dram_output_configs[role],
            compute_kernel_config=compute_kernel_config,
        )
        if output.memory_config() != self.decode_dram_output_configs[role]:
            raise AssertionError(f"R22 DRAM {role} did not produce the configured output storage grid")
        if tuple(output.shape) != (*tuple(x.shape)[:-1], runtime["logical_n"]):
            raise AssertionError(f"R22 DRAM {role} changed the logical batch or output width")
        self.optimized_path_counters[f"r22_dram_{role}"] += 1
        runtime.update(
            actual_input_dtype=str(working.dtype),
            actual_weight_dtype=str(weight.dtype),
            actual_output_dtype=str(output.dtype),
            actual_input_memory_config=str(working.memory_config()),
            actual_output_memory_config=str(output.memory_config()),
            actual_weight_memory_config=str(weight.memory_config()),
            actual_program_config=str(config),
            actual_compute_kernel_config=str(compute_kernel_config),
            actual_logical_output_shape=list(output.shape),
            host_dispatch_count=self.optimized_path_counters[f"r22_dram_{role}"],
        )
        if working is not x:
            working.deallocate(True)
        return output

    def _tracked_to_memory_config(
        self,
        x: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig,
        counter: str,
    ) -> ttnn.Tensor:
        self.optimized_path_counters[counter] += 1
        return ttnn.to_memory_config(x, memory_config, dtype=x.dtype)

    def _tracked_sharded_to_interleaved(
        self,
        x: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig,
        counter: str,
    ) -> ttnn.Tensor:
        self.optimized_path_counters[counter] += 1
        return ttnn.sharded_to_interleaved(x, memory_config)

    def _prepare_padded_attention_input(
        self,
        x: ttnn.Tensor,
        *,
        padded_width: int,
        memory_config: ttnn.MemoryConfig,
        boundary_counter: str,
        padding_counter: str,
    ) -> ttnn.Tensor:
        """Pad one candidate input with zeros, charging its boundary once."""

        if x.shape[-1] == padded_width:
            return self._tracked_to_memory_config(x, memory_config, boundary_counter)
        if x.shape[-1] > padded_width:
            raise ValueError(f"attention input width {x.shape[-1]} exceeds candidate width {padded_width}")
        self.optimized_path_counters[boundary_counter] += 1
        self.optimized_path_counters[padding_counter] += 1
        original = x
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            original.deallocate(True)
        pad_spec = [(0, 0)] * (len(x.shape) - 1) + [(0, padded_width - x.shape[-1])]
        unpadded = x
        x = ttnn.pad(unpadded, pad_spec, 0.0, memory_config=ttnn.L1_MEMORY_CONFIG)
        unpadded.deallocate(True)
        padded = x
        x = ttnn.to_memory_config(padded, memory_config, dtype=padded.dtype)
        padded.deallocate(True)
        return x

    def _residual_rms_norm(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor | None,
    ) -> ttnn.Tensor:
        if self.residual_memory_config is None or self.residual_norm_program_config is None:
            raise RuntimeError("the sharded residual RMS path requires R11 or R22 configuration")
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=weight,
            program_config=self.residual_norm_program_config,
            compute_kernel_config=self.correctness_compute_config,
            memory_config=self.residual_memory_config,
        )

    def _router_weights_from_normalized(self, router_in: ttnn.Tensor) -> ttnn.Tensor:
        """Route from one unweighted RMS-normalized residual tensor."""

        tokens = router_in.shape[-2]
        router_input_memory_config = self._router_input_memory_config(tokens)
        if router_in.is_sharded():
            router_in = self._tracked_sharded_to_interleaved(
                router_in,
                router_input_memory_config,
                "router_input",
            )
        if self.folded_router_projection:
            self.optimized_path_counters["folded_router_projection"] += 1
        else:
            router_in = ttnn.mul(router_in, self.weights.router_scale, memory_config=router_input_memory_config)
            router_in = ttnn.mul(router_in, self.router_hidden_scale, memory_config=router_input_memory_config)
        router_in = ttnn.reshape(router_in, [tokens, HIDDEN_SIZE])
        if router_input_memory_config == ttnn.L1_MEMORY_CONFIG:
            router_in = ttnn.typecast(router_in, ttnn.float32, memory_config=router_input_memory_config)
        else:
            # Preserve the established default call exactly; the opt-in hook
            # only constrains placement for the L1 candidate.
            router_in = ttnn.typecast(router_in, ttnn.float32)
        logits = ttnn.linear(
            router_in,
            self.weights.router_proj,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits = ttnn.typecast(logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(logits, k=TOP_K_EXPERTS, dim=-1, sorted=True)
        top_values = ttnn.softmax(
            top_values,
            dim=-1,
            numeric_stable=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        row_major_routing = self.routing_row_major and getattr(self, "_in_decode_forward", False) and tokens == 1
        routing_base = self.decode_routing_zero_base if row_major_routing else ttnn.zeros_like(logits)
        routing = ttnn.scatter(routing_base, dim=-1, index=top_indices, src=top_values)
        if row_major_routing:
            if routing.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise AssertionError("row-major scatter base did not preserve row-major routing")
            self.optimized_path_counters["routing_row_major_scatter"] += 1
            self.routing_runtime.update(
                scatter_base_layout=str(routing_base.layout),
                scatter_output_layout=str(routing.layout),
                scatter_output_memory_config=str(routing.memory_config()),
                logical_tokens=tokens,
                score_dtype=str(routing.dtype),
            )
        if self.folded_expert_scale:
            self.optimized_path_counters["folded_expert_scale"] += 1
        else:
            routing = ttnn.mul(routing, self.weights.router_per_expert_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            routing = ttnn.typecast(routing, ttnn.bfloat16)
        if (
            getattr(self, "indexed_expert_decode", False)
            and getattr(self, "_in_decode_forward", False)
            and self.folded_expert_scale
        ):
            # Indexed sparse matmul consumes an independently exact top-k
            # index tensor. BF16 routing scores cannot provide that invariant.
            self.decode_route_indices = ttnn.to_layout(
                ttnn.reshape(top_indices, [1, 1, tokens, TOP_K_EXPERTS]),
                ttnn.ROW_MAJOR_LAYOUT,
            )
            self.decode_compact_route_scores = ttnn.reshape(top_values, [1, 1, tokens, TOP_K_EXPERTS])
        return ttnn.reshape(routing, [1, 1, tokens, NUM_EXPERTS])

    def _router_input_memory_config(self, tokens: int) -> ttnn.MemoryConfig:
        """Return the router input placement; subclasses may opt in to L1."""

        del tokens
        return ttnn.DRAM_MEMORY_CONFIG

    def _router_weights(self, residual: ttnn.Tensor) -> ttnn.Tensor:
        """Use the optimized router for prefill and non-sharded candidates."""

        return self._router_weights_from_normalized(self._rms_norm(residual, None))

    def _router_weights_sharded(self, residual: ttnn.Tensor) -> ttnn.Tensor:
        """Normalize on the residual grid, then cross once into router FP32."""

        return self._router_weights_from_normalized(self._residual_rms_norm(residual, None))

    def _shared_ffn_input(self, residual: ttnn.Tensor, *, sharded: bool) -> ttnn.Tensor | None:
        if not self.shared_ffn_norm:
            return None
        self.optimized_path_counters["shared_ffn_norm"] += 1
        if sharded:
            return self._residual_rms_norm(residual, None)
        return self._rms_norm(residual, None)

    def _final_residual(
        self,
        residual: ttnn.Tensor,
        branch: ttnn.Tensor,
        *,
        memory_config: ttnn.MemoryConfig,
    ) -> ttnn.Tensor:
        if self.fused_final_scalar:
            self.optimized_path_counters["fused_final_scalar"] += 1
            return ttnn.add(
                residual,
                branch,
                activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.layer_scalar_value)],
                memory_config=memory_config,
            )
        merged = ttnn.add(residual, branch, memory_config=memory_config)
        return ttnn.mul(merged, self.weights.layer_scalar, memory_config=memory_config)

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int = 0,
        chunk_page_table: ttnn.Tensor | None = None,
        cache_position_modulo: int | None = None,
    ) -> ttnn.Tensor:
        """Run prefill without selecting the decode-only DRAM weight copies."""
        self._in_decode_forward = False
        return super().prefill_forward(
            hidden_states,
            position_cos=position_cos,
            position_sin=position_sin,
            page_table=page_table,
            kv_cache=kv_cache,
            user_id=user_id,
            chunk_page_table=chunk_page_table,
            cache_position_modulo=cache_position_modulo,
        )

    def _prefill_forward_single_user(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int,
        chunk_page_table: ttnn.Tensor | None,
        cache_position_modulo: int | None,
    ) -> ttnn.Tensor:
        """Run optimized prefill while preserving arbitrary logical lengths."""

        logical_seq_len = hidden_states.shape[-2]
        if logical_seq_len < 1:
            raise ValueError("prefill requires at least one logical token")
        padded_seq_len = math.ceil(logical_seq_len / TILE_SIZE) * TILE_SIZE
        if padded_seq_len != logical_seq_len:
            pad = [(0, 0), (0, 0), (0, padded_seq_len - logical_seq_len), (0, 0)]
            hidden_states = ttnn.pad(hidden_states, pad, 0.0)
            position_cos = ttnn.pad(position_cos, pad, 0.0)
            position_sin = ttnn.pad(position_sin, pad, 0.0)

        residual = hidden_states
        attn_in = self._rms_norm(hidden_states, self.weights.input_ln)
        attn_out = self._attention_prefill(
            attn_in,
            position_cos=position_cos,
            position_sin=position_sin,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            kv_cache=kv_cache,
            user_id=user_id,
            cache_position_modulo=cache_position_modulo,
            logical_seq_len=logical_seq_len,
        )
        attn_out = self._rms_norm(attn_out, self.weights.post_attn_ln)
        hidden_states = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        shared_ffn_in = self._shared_ffn_input(residual, sharded=False)
        mlp_in = shared_ffn_in if shared_ffn_in is not None else self._rms_norm(residual, self.weights.pre_ff_ln)
        mlp_out = self._dense_mlp(mlp_in)
        hidden_1 = self._rms_norm(mlp_out, self.weights.post_ff_ln_1)

        router_weights = (
            self._router_weights_from_normalized(shared_ffn_in)
            if shared_ffn_in is not None
            else self._router_weights(residual)
        )
        moe_in = shared_ffn_in if shared_ffn_in is not None else self._rms_norm(residual, self.weights.pre_ff_ln_2)
        hidden_2 = self._moe_prefill(moe_in, router_weights)
        hidden_2 = self._rms_norm(hidden_2, self.weights.post_ff_ln_2)

        hidden_states = ttnn.add(hidden_1, hidden_2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self._rms_norm(hidden_states, self.weights.post_ff_ln)
        hidden_states = self._final_residual(
            residual,
            hidden_states,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if padded_seq_len != logical_seq_len:
            hidden_states = ttnn.slice(
                hidden_states,
                starts=[0, 0, 0, 0],
                ends=[1, 1, logical_seq_len, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return hidden_states

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        cache_position_modulo: int | None = None,
    ) -> ttnn.Tensor:
        """Run the env-gated coherent R11/R22 hidden residual candidate."""
        self._in_decode_forward = True
        self._decode_logical_batch = hidden_states.shape[-2]
        if self.residual_shard_cores == 0:
            if any(
                (
                    self.folded_router_projection,
                    self.shared_ffn_norm,
                    self.folded_expert_scale,
                    self.fused_final_scalar,
                )
            ):
                return self._decode_forward_interleaved(
                    hidden_states,
                    position_cos=position_cos,
                    position_sin=position_sin,
                    current_pos=current_pos,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    cache_position_modulo=cache_position_modulo,
                )
            return super().decode_forward(
                hidden_states,
                position_cos=position_cos,
                position_sin=position_sin,
                current_pos=current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                cache_position_modulo=cache_position_modulo,
            )
        if hidden_states.shape[-2] < 1 or hidden_states.shape[-2] > TILE_SIZE:
            raise ValueError(f"sharded decode requires logical batch in [1, {TILE_SIZE}]")
        if hidden_states.shape[-1] != HIDDEN_SIZE:
            raise ValueError(f"sharded decode requires hidden width {HIDDEN_SIZE}")

        r22_counters_before = {}
        if hidden_states.shape[-2] == 1:
            if self.r22_dram_sharded:
                r22_counters_before = {
                    f"r22_dram_{role}": self.optimized_path_counters[f"r22_dram_{role}"]
                    for role in self.dram_sharded_roles
                }
            if self.r22_packed_dense_gate_up:
                r22_counters_before["r22_packed_dense"] = self.optimized_path_counters["r22_packed_dense"]
            if self.routing_row_major:
                routing_counters = ["routing_row_major_scatter", "routing_row_major_metadata"]
                if not getattr(self, "indexed_expert_decode", False):
                    routing_counters.append("routing_score_tilize")
                r22_counters_before.update(
                    (counter, self.optimized_path_counters[counter]) for counter in routing_counters
                )
        self.optimized_path_counters["residual_chain_decode"] += 1
        hidden_states = self._tracked_to_memory_config(
            hidden_states,
            self.residual_memory_config,
            "residual_entry",
        )

        residual = hidden_states
        attn_in = self._residual_rms_norm(hidden_states, self.weights.input_ln)
        attn_out = self._attention_decode(
            attn_in,
            position_cos=position_cos,
            position_sin=position_sin,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            cache_position_modulo=cache_position_modulo,
        )
        attn_out = self._residual_rms_norm(attn_out, self.weights.post_attn_ln)
        hidden_states = ttnn.add(residual, attn_out, memory_config=self.residual_memory_config)

        residual = hidden_states
        shared_ffn_in = self._shared_ffn_input(residual, sharded=True)
        mlp_in = (
            shared_ffn_in
            if shared_ffn_in is not None
            else self._residual_rms_norm(hidden_states, self.weights.pre_ff_ln)
        )
        mlp_out = self._dense_mlp(mlp_in)
        hidden_1 = self._residual_rms_norm(mlp_out, self.weights.post_ff_ln_1)

        router_weights = (
            self._router_weights_from_normalized(shared_ffn_in)
            if shared_ffn_in is not None
            else self._router_weights_sharded(residual)
        )
        moe_in = (
            shared_ffn_in if shared_ffn_in is not None else self._residual_rms_norm(residual, self.weights.pre_ff_ln_2)
        )
        moe_in = self._tracked_sharded_to_interleaved(
            moe_in,
            ttnn.L1_MEMORY_CONFIG,
            "expert_input",
        )
        hidden_2 = self._moe_decode(moe_in, router_weights)
        hidden_2 = self._tracked_to_memory_config(
            hidden_2,
            self.residual_memory_config,
            "expert_output",
        )
        hidden_2 = self._residual_rms_norm(hidden_2, self.weights.post_ff_ln_2)

        hidden_states = ttnn.add(hidden_1, hidden_2, memory_config=self.residual_memory_config)
        hidden_states = self._residual_rms_norm(hidden_states, self.weights.post_ff_ln)
        hidden_states = self._final_residual(
            residual,
            hidden_states,
            memory_config=self.residual_memory_config,
        )
        for counter, before in r22_counters_before.items():
            if self.optimized_path_counters[counter] != before + 1:
                raise AssertionError(f"selected R22 projection was not invoked exactly once: {counter}")
        return self._tracked_sharded_to_interleaved(
            hidden_states,
            ttnn.DRAM_MEMORY_CONFIG,
            "residual_exit",
        )

    def _decode_forward_interleaved(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        cache_position_modulo: int | None,
    ) -> ttnn.Tensor:
        """Preserve graph-fold semantics for the explicit non-R22 A/B path."""

        if hidden_states.shape[-2] < 1:
            raise ValueError("decode requires at least one batch row")
        residual = hidden_states
        attn_in = self._rms_norm(hidden_states, self.weights.input_ln)
        attn_out = self._attention_decode(
            attn_in,
            position_cos=position_cos,
            position_sin=position_sin,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            cache_position_modulo=cache_position_modulo,
        )
        attn_out = self._rms_norm(attn_out, self.weights.post_attn_ln)
        hidden_states = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        shared_ffn_in = self._shared_ffn_input(residual, sharded=False)
        mlp_in = shared_ffn_in if shared_ffn_in is not None else self._rms_norm(residual, self.weights.pre_ff_ln)
        hidden_1 = self._rms_norm(self._dense_mlp(mlp_in), self.weights.post_ff_ln_1)
        router_weights = (
            self._router_weights_from_normalized(shared_ffn_in)
            if shared_ffn_in is not None
            else self._router_weights(residual)
        )
        moe_in = shared_ffn_in if shared_ffn_in is not None else self._rms_norm(residual, self.weights.pre_ff_ln_2)
        hidden_2 = self._rms_norm(self._moe_decode(moe_in, router_weights), self.weights.post_ff_ln_2)

        hidden_states = ttnn.add(hidden_1, hidden_2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self._rms_norm(hidden_states, self.weights.post_ff_ln)
        return self._final_residual(
            residual,
            hidden_states,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _attention_prefill(
        self,
        x: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        page_table: ttnn.Tensor,
        chunk_page_table: ttnn.Tensor | None,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int,
        cache_position_modulo: int | None,
        logical_seq_len: int,
    ) -> ttnn.Tensor:
        self.optimized_path_counters["prefill_attention"] += 1
        kind = self.layer_kind
        seq_len = x.shape[-2]
        xqkv = self._linear(x, "qkv", compute_kernel_config=self.prefill_attention_compute_config)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=NUM_Q_HEADS,
            num_kv_heads=kind.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_heads = self._rms_norm(q_heads, self.weights.q_norm)
        k_heads = self._rms_norm(k_heads, self.weights.k_norm)
        v_heads = self._rms_norm(v_heads, None)
        q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=False)
        k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=False)

        key_cache, value_cache = kv_cache
        fill_table = chunk_page_table if chunk_page_table is not None else page_table
        self._fill_prefill_cache(
            key_cache,
            value_cache,
            k_heads,
            v_heads,
            fill_table,
            user_id=user_id,
            logical_seq_len=logical_seq_len,
            cache_position_modulo=cache_position_modulo,
        )

        from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import _prefill_attention_path

        attention_path = _prefill_attention_path(
            seq_len,
            is_sliding=kind.sliding_window is not None,
            has_paged_cache=fill_table is not None,
        )
        if attention_path == "sliding_chunked":
            attn_out = self._sliding_chunked_prefill_attention(q_heads, k_heads, v_heads)
        elif attention_path == "full_chunked":
            attn_out = self._full_chunked_prefill_attention(
                q_heads,
                key_cache,
                value_cache,
                fill_table,
                user_id=user_id,
            )
        else:
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q_heads,
                k_heads,
                v_heads,
                is_causal=True,
                sliding_window_size=kind.sliding_window,
                scale=1.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        attn_out = ttnn.reshape(attn_out, [1, NUM_Q_HEADS, seq_len, kind.head_dim])
        attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._linear(attn_out, "o_proj", compute_kernel_config=self.prefill_attention_compute_config)

    def _attention_decode(
        self,
        x: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        cache_position_modulo: int | None,
    ) -> ttnn.Tensor:
        self.optimized_path_counters["decode_attention"] += 1
        kind = self.layer_kind
        batch = x.shape[-2]
        sharded_residual = self.residual_shard_cores != 0
        if sharded_residual:
            if batch == 1:
                attention_geometry = self.attention_runtime_geometry
                attention_memory_configs = self.attention_sharded_memory_configs
                attention_program_configs = self.attention_sharded_program_configs
                attention_weights = self.decode_attention_weights
            else:
                attention_geometry = self.attention_batch32_runtime_geometry
                attention_memory_configs = self.attention_batch32_sharded_memory_configs
                attention_program_configs = self.attention_batch32_sharded_program_configs
                attention_weights = self.decode_attention_weights_batch32
            qkv_geometry = attention_geometry["qkv"]
            if self._use_r22_dram_weight(x, "qkv"):
                self.optimized_path_counters["attention_qkv_input"] += 1
                xqkv = self._r22_dram_linear(x, "qkv", compute_kernel_config=self.decode_attention_compute_config)
                qkv_geometry = {**qkv_geometry, "output_padding": 0}
            else:
                x = self._prepare_padded_attention_input(
                    x,
                    padded_width=qkv_geometry["padded_k"],
                    memory_config=attention_memory_configs["qkv_input"],
                    boundary_counter="attention_qkv_input",
                    padding_counter="attention_qkv_input_padding",
                )
                xqkv = ttnn.linear(
                    x,
                    attention_weights["qkv"],
                    dtype=self._activation_dtype_for("attention"),
                    program_config=attention_program_configs["qkv"],
                    memory_config=attention_memory_configs["qkv_output"],
                    # Keep the selected fidelity while changing projection storage.
                    compute_kernel_config=self.decode_attention_compute_config,
                )
            # The sharded decode splitter reads a next-core coordinate after
            # its final V tile and trips watcher's runtime-argument bounds
            # check in this checkout. Select the legal interleaved reader,
            # retaining the sharded matmul and height-sharded head outputs.
            sharded_xqkv = xqkv
            xqkv = ttnn.sharded_to_interleaved(sharded_xqkv, ttnn.L1_MEMORY_CONFIG)
            sharded_xqkv.deallocate(True)
            if qkv_geometry["output_padding"]:
                padded_xqkv = xqkv
                xqkv = ttnn.slice(
                    padded_xqkv,
                    starts=[0] * len(padded_xqkv.shape),
                    ends=[
                        *(padded_xqkv.shape[dim] for dim in range(len(padded_xqkv.shape) - 1)),
                        qkv_geometry["logical_n"],
                    ],
                    steps=[1] * len(padded_xqkv.shape),
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                padded_xqkv.deallocate(True)
                self.optimized_path_counters["attention_qkv_output_slice"] += 1
        else:
            xqkv = self._linear(x, "qkv", compute_kernel_config=self.decode_attention_compute_config)
        head_mem_config = _make_decode_height_sharded_memory_config(self.mesh_device, batch, kind.head_dim)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=NUM_Q_HEADS,
            num_kv_heads=kind.num_kv_heads,
            memory_config=head_mem_config,
        )
        if sharded_residual:
            # The split-head outputs own everything needed below. Releasing
            # the residual-grid norm result and fused projection immediately
            # prevents their batch-32 L1 allocations from overlapping the
            # paged-cache update program's static circular buffers.
            x.deallocate(True)
            xqkv.deallocate(True)
        # The head split explicitly produces this layout. During shard
        # analysis the optimizer owns tensor layout state, so use the
        # already-declared phase config instead of querying the traced tensor.
        q_mem_config = k_mem_config = v_mem_config = head_mem_config
        q_heads = ttnn.to_memory_config(q_heads, ttnn.L1_MEMORY_CONFIG, dtype=q_heads.dtype)
        k_heads = ttnn.to_memory_config(k_heads, ttnn.L1_MEMORY_CONFIG, dtype=k_heads.dtype)
        v_heads = ttnn.to_memory_config(v_heads, ttnn.L1_MEMORY_CONFIG, dtype=v_heads.dtype)
        q_heads = self._rms_norm(q_heads, self.weights.q_norm)
        k_heads = self._rms_norm(k_heads, self.weights.k_norm)
        v_heads = self._rms_norm(v_heads, None)
        if kind.name == "full_attention":
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=False)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=False)
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.to_memory_config(q_heads, q_mem_config, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(k_heads, k_mem_config, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, v_mem_config, dtype=v_heads.dtype)
        else:
            q_heads = ttnn.to_memory_config(q_heads, q_mem_config, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(k_heads, k_mem_config, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, v_mem_config, dtype=v_heads.dtype)
            rope_mem_config = _make_decode_rope_memory_config(self.mesh_device, batch, kind.head_dim)
            position_cos = ttnn.interleaved_to_sharded(position_cos, rope_mem_config)
            position_sin = ttnn.interleaved_to_sharded(position_sin, rope_mem_config)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=True)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=True)

        key_cache, value_cache = kv_cache
        cache_view = self._cache_view_kwargs(prefill=False, cache_position_modulo=cache_position_modulo)
        ttnn.experimental.paged_update_cache(
            key_cache,
            k_heads,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            **cache_view,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            v_heads,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            **cache_view,
        )
        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_heads,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=1.0,
            sliding_window_size=kind.sliding_window,
            program_config=self.sdpa_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **self._sdpa_cache_view_kwargs(cache_position_modulo=cache_position_modulo),
        )
        concat_mem_config = _make_decode_height_sharded_memory_config(self.mesh_device, batch, kind.head_dim)
        if sharded_residual:
            attn_out = self._tracked_to_memory_config(
                attn_out,
                concat_mem_config,
                "attention_sdpa_output",
            )
        else:
            attn_out = ttnn.to_memory_config(attn_out, concat_mem_config, dtype=attn_out.dtype)
        attn_out = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=NUM_Q_HEADS)
        if sharded_residual:
            o_geometry = attention_geometry["o_proj"]
            if self._use_r22_dram_weight(attn_out, "o_proj"):
                self.optimized_path_counters["attention_o_input"] += 1
                concat_output = attn_out
                attn_out = self._r22_dram_linear(
                    concat_output, "o_proj", compute_kernel_config=self.decode_attention_compute_config
                )
                concat_output.deallocate(True)
                o_geometry = {**o_geometry, "output_padding": 0}
            else:
                attn_out = self._prepare_padded_attention_input(
                    attn_out,
                    padded_width=o_geometry["padded_k"],
                    memory_config=attention_memory_configs["o_input"],
                    boundary_counter="attention_o_input",
                    padding_counter="attention_o_input_padding",
                )
                attn_out = ttnn.linear(
                    attn_out,
                    attention_weights["o_proj"],
                    dtype=self._activation_dtype_for("attention"),
                    program_config=attention_program_configs["o_proj"],
                    memory_config=attention_memory_configs["o_output"],
                    compute_kernel_config=self.decode_attention_compute_config,
                )
        else:
            attn_out = ttnn.sharded_to_interleaved(attn_out, ttnn.DRAM_MEMORY_CONFIG)
            attn_out = self._linear(attn_out, "o_proj", compute_kernel_config=self.decode_attention_compute_config)
        if sharded_residual and o_geometry["output_padding"]:
            padded_attn_out = ttnn.sharded_to_interleaved(attn_out, ttnn.L1_MEMORY_CONFIG)
            attn_out.deallocate(True)
            attn_out = ttnn.slice(
                padded_attn_out,
                starts=[0, 0, 0, 0],
                ends=[1, 1, batch, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            padded_attn_out.deallocate(True)
            self.optimized_path_counters["attention_o_output_slice"] += 1
        elif attn_out.shape[-2] != batch:
            attn_out = ttnn.slice(
                attn_out,
                starts=[0, 0, 0, 0],
                ends=[1, 1, batch, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=(attention_memory_configs["o_output"] if sharded_residual else ttnn.DRAM_MEMORY_CONFIG),
            )
        if sharded_residual:
            attn_out = self._tracked_to_memory_config(
                attn_out,
                self.residual_memory_config,
                "attention_output",
            )
        return attn_out

    def _dense_mlp(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.optimized_path_counters["dense_mlp"] += 1
        if self.residual_shard_cores != 0 and x.is_sharded():
            return self._dense_mlp_residual_sharded(x)
        if self.packed_dense_gate_up and self.packed_mlp_gate_up is not None and not self.r22_packed_dense_gate_up:
            decode_candidate = self._use_decode_dram_weight(x, "packed_mlp_gate_up")
            packed_weight = (
                self.decode_dram_weights["packed_mlp_gate_up"] if decode_candidate else self.packed_mlp_gate_up
            )
            kwargs = {}
            if decode_candidate:
                kwargs["program_config"] = self.decode_dram_configs["packed_mlp_gate_up"]
                padded_input_width = self.decode_dram_padded_input_widths.get("packed_mlp_gate_up")
                if padded_input_width is not None:
                    pad_spec = [(0, 0)] * len(x.shape)
                    pad_spec[-1] = (0, padded_input_width - x.shape[-1])
                    x = ttnn.pad(x, pad_spec, 0.0)
                x = ttnn.to_memory_config(
                    x,
                    self.decode_dram_input_configs["packed_mlp_gate_up"],
                    dtype=x.dtype,
                )
                kwargs["memory_config"] = self.decode_dram_output_configs["packed_mlp_gate_up"]
            gate_up = ttnn.linear(
                x,
                packed_weight,
                dtype=self._activation_dtype_for("dense_mlp"),
                memory_config=kwargs.pop("memory_config", ttnn.DRAM_MEMORY_CONFIG),
                compute_kernel_config=self.mlp_compute_config,
                **kwargs,
            )
            if decode_candidate:
                gate_up = ttnn.sharded_to_interleaved(gate_up, ttnn.DRAM_MEMORY_CONFIG)
                logical_output_width = self.decode_dram_logical_output_widths.get("packed_mlp_gate_up")
                if logical_output_width is not None:
                    starts = [0] * len(gate_up.shape)
                    ends = list(gate_up.shape)
                    ends[-1] = logical_output_width
                    gate_up = ttnn.slice(
                        gate_up,
                        starts,
                        ends,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
            width = self.weights.mlp_gate.shape[-1]
            ends = [gate_up.shape[index] for index in range(len(gate_up.shape))]
            gate_ends = list(ends)
            gate_ends[-1] = width
            up_starts = [0] * len(ends)
            up_starts[-1] = width
            gate = ttnn.slice(
                gate_up,
                starts=[0] * len(ends),
                ends=gate_ends,
                steps=[1] * len(ends),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            up = ttnn.slice(
                gate_up,
                starts=up_starts,
                ends=ends,
                steps=[1] * len(ends),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            gate = self._linear(x, "mlp_gate", compute_kernel_config=self.mlp_compute_config)
            up = self._linear(x, "mlp_up", compute_kernel_config=self.mlp_compute_config)
        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._linear(hidden, "mlp_down", compute_kernel_config=self.mlp_compute_config)

    def _dense_mlp_residual_sharded(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Keep the split gate/up/down family coherent with the residual grid."""
        if x.shape[-2] == 1 and (
            self.r22_packed_dense_gate_up
            or (self.r22_dram_sharded and self.dram_sharded_roles & {"mlp_gate", "mlp_up", "mlp_down"})
        ):
            return self._dense_mlp_r22_candidate(x)
        gate = ttnn.linear(
            x,
            self.weights.mlp_gate,
            dtype=self._activation_dtype_for("dense_mlp"),
            program_config=self.residual_dense_program_configs["mlp_gate"],
            memory_config=self.residual_intermediate_memory_config,
            compute_kernel_config=self.mlp_compute_config,
        )
        up = ttnn.linear(
            x,
            self.weights.mlp_up,
            dtype=self._activation_dtype_for("dense_mlp"),
            program_config=self.residual_dense_program_configs["mlp_up"],
            memory_config=self.residual_intermediate_memory_config,
            compute_kernel_config=self.mlp_compute_config,
        )
        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)],
            memory_config=self.residual_intermediate_memory_config,
        )
        return ttnn.linear(
            hidden,
            self.weights.mlp_down,
            dtype=self._activation_dtype_for("dense_mlp"),
            program_config=self.residual_dense_program_configs["mlp_down"],
            memory_config=self.residual_memory_config,
            compute_kernel_config=self.mlp_compute_config,
        )

    def _dense_mlp_r22_candidate(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Compare packed/separate projections with direct sharded consumers."""
        working = x
        gate_roles = ("mlp_gate", "mlp_up")
        if not self.r22_packed_dense_gate_up and all(self._use_r22_dram_weight(x, role) for role in gate_roles):
            gate_config, up_config = (self.decode_dram_input_configs[role] for role in gate_roles)
            if gate_config == up_config and x.memory_config() != gate_config:
                working = ttnn.to_memory_config(x, gate_config, dtype=x.dtype)
                self.optimized_path_counters["r22_dense_input_reshards"] += 1

        def project(activation: ttnn.Tensor, role: str) -> ttnn.Tensor:
            if self._use_r22_dram_weight(activation, role):
                return self._r22_dram_linear(activation, role, compute_kernel_config=self.mlp_compute_config)
            output_config = (
                self.residual_memory_config if role == "mlp_down" else self.residual_intermediate_memory_config
            )
            if role == "packed_mlp_gate_up":
                weight = self.packed_mlp_gate_up
                output_config = self.r22_packed_dense_memory_config
            else:
                weight = getattr(self.weights, role)
            return ttnn.linear(
                activation,
                weight,
                dtype=self._activation_dtype_for("dense_mlp"),
                program_config=self.residual_dense_program_configs[role],
                memory_config=output_config,
                compute_kernel_config=self.mlp_compute_config,
            )

        if self.r22_packed_dense_gate_up:
            packed_weight = self.decode_dram_weights.get("packed_mlp_gate_up", self.packed_mlp_gate_up)
            packed = project(x, "packed_mlp_gate_up")
            gate, up = ttnn.split(
                packed, MLP_INTERMEDIATE_SIZE, dim=-1, memory_config=self.residual_intermediate_memory_config
            )
            for half in (gate, up):
                if half.memory_config() != self.residual_intermediate_memory_config:
                    raise AssertionError("packed dense split must produce R22-sharded halves")
                if tuple(half.shape) != (*tuple(x.shape)[:-1], MLP_INTERMEDIATE_SIZE):
                    raise AssertionError("packed dense split changed logical batch or intermediate width")
            self.optimized_path_counters["r22_packed_dense"] += 1
            self.r22_projection_runtime["packed_dense_split"] = {
                "logical_packed_shape": list(packed.shape),
                "packed_memory_config": str(packed.memory_config()),
                "half_memory_config": str(gate.memory_config()),
                "split_implementation": "two tile-aligned sharded slices for decode M < 64",
                "gate_up_weight_dtype": str(packed_weight.dtype),
                "program_config": str(
                    self.decode_dram_configs["packed_mlp_gate_up"]
                    if self._use_r22_dram_weight(x, "packed_mlp_gate_up")
                    else self.residual_dense_program_configs["packed_mlp_gate_up"]
                ),
                "compute_kernel_config": str(self.mlp_compute_config),
                "host_dispatch_count": self.optimized_path_counters["r22_packed_dense"],
            }
            packed.deallocate(True)
        else:
            gate = project(working, "mlp_gate")
            up = project(working, "mlp_up")
        if working is not x:
            working.deallocate(True)
        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)],
            memory_config=self.residual_intermediate_memory_config,
        )
        gate.deallocate(True)
        up.deallocate(True)
        output = project(hidden, "mlp_down")
        hidden.deallocate(True)
        if output.memory_config() != self.residual_memory_config:
            raise AssertionError("dense candidate must return directly to the R22 norm/residual consumer")
        return output

    def _fill_prefill_cache(
        self,
        key_cache: ttnn.Tensor,
        value_cache: ttnn.Tensor,
        k_heads: ttnn.Tensor,
        v_heads: ttnn.Tensor,
        page_table: ttnn.Tensor,
        *,
        user_id: int,
        logical_seq_len: int,
        cache_position_modulo: int | None,
    ) -> None:
        """Allow low-precision cache trials without changing update semantics.

        ``paged_fill_cache`` requires source and cache dtypes to match, whereas
        ``paged_update_cache`` intentionally consumes BF16 and repacks into a
        low-precision cache. Cast every bulk-fill source to the destination
        dtype while retaining the functional BF16 update path for a bounded,
        non-aligned tail.
        """
        fill_kwargs = self._cache_view_kwargs(prefill=True, cache_position_modulo=cache_position_modulo)

        def bulk_fill(cache: ttnn.Tensor, source: ttnn.Tensor) -> None:
            fill_source = source
            if source.dtype != cache.dtype:
                fill_source = ttnn.typecast(source, cache.dtype)
            ttnn.experimental.paged_fill_cache(
                cache,
                fill_source,
                page_table,
                batch_idx=user_id,
                **fill_kwargs,
            )
            if fill_source is not source:
                fill_source.deallocate(True)

        if cache_position_modulo is None or logical_seq_len % TILE_SIZE == 0:
            bulk_fill(key_cache, k_heads)
            bulk_fill(value_cache, v_heads)
            return

        aligned_prefix, tail_positions = _bounded_cache_fill_plan(logical_seq_len)
        if aligned_prefix:
            k_prefix = ttnn.slice(
                k_heads,
                [0, 0, 0, 0],
                [k_heads.shape[0], k_heads.shape[1], aligned_prefix, k_heads.shape[3]],
            )
            v_prefix = ttnn.slice(
                v_heads,
                [0, 0, 0, 0],
                [v_heads.shape[0], v_heads.shape[1], aligned_prefix, v_heads.shape[3]],
            )
            bulk_fill(key_cache, k_prefix)
            bulk_fill(value_cache, v_prefix)
            k_prefix.deallocate(True)
            v_prefix.deallocate(True)

        page_table_row = page_table
        owns_page_table_row = False
        if page_table.shape[0] > 1:
            page_table_row = ttnn.slice(
                page_table,
                [user_id, 0],
                [user_id + 1, page_table.shape[1]],
            )
            owns_page_table_row = True
        update_mem_config = _make_single_user_cache_update_memory_config(self.mesh_device, self.layer_kind.head_dim)
        update_kwargs = self._cache_view_kwargs(prefill=False, cache_position_modulo=cache_position_modulo)
        for position in tail_positions:
            k_token = ttnn.slice(
                k_heads,
                [0, 0, position, 0],
                [1, self.layer_kind.num_kv_heads, position + 1, self.layer_kind.head_dim],
            )
            v_token = ttnn.slice(
                v_heads,
                [0, 0, position, 0],
                [1, self.layer_kind.num_kv_heads, position + 1, self.layer_kind.head_dim],
            )
            k_token = ttnn.transpose(k_token, 1, 2)
            v_token = ttnn.transpose(v_token, 1, 2)
            k_token = ttnn.to_memory_config(k_token, update_mem_config, dtype=k_token.dtype)
            v_token = ttnn.to_memory_config(v_token, update_mem_config, dtype=v_token.dtype)
            # moreh_full's row-major writer can issue a 64-byte burst. Allocate
            # a full tile-width INT32 stick so watcher sees an in-bounds write;
            # paged_update_cache still consumes its first value for this user.
            position_tensor = ttnn.moreh_full(
                (1, TILE_SIZE),
                position,
                self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.experimental.paged_update_cache(
                key_cache,
                k_token,
                update_idxs_tensor=position_tensor,
                page_table=page_table_row,
                **update_kwargs,
            )
            ttnn.experimental.paged_update_cache(
                value_cache,
                v_token,
                update_idxs_tensor=position_tensor,
                page_table=page_table_row,
                **update_kwargs,
            )
            k_token.deallocate(True)
            v_token.deallocate(True)
            position_tensor.deallocate(True)
        if owns_page_table_row:
            page_table_row.deallocate(True)

    def _moe_prefill(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        self.optimized_path_counters["expert_prefill"] += 1
        seq_len = hidden_states.shape[-2]
        if seq_len <= PREFILL_MOE_CHUNK_SIZE:
            return self._moe_prefill_chunk(hidden_states, routing_weights)

        chunks = []
        for start in range(0, seq_len, PREFILL_MOE_CHUNK_SIZE):
            end = min(start + PREFILL_MOE_CHUNK_SIZE, seq_len)
            hidden_chunk = ttnn.slice(
                hidden_states,
                starts=[0, 0, start, 0],
                ends=[1, 1, end, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            routing_chunk = ttnn.slice(
                routing_weights,
                starts=[0, 0, start, 0],
                ends=[1, 1, end, NUM_EXPERTS],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            chunks.append(self._moe_prefill_chunk(hidden_chunk, routing_chunk))
        return ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _moe_prefill_chunk(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        chunk_size = self.prefill_expert_chunk_size
        seq_len = hidden_states.shape[2]
        packed_expert_width = getattr(self, "packed_expert_width", _PACKED_EXPERT_WIDTH)
        if seq_len % TILE_SIZE != 0 or chunk_size % TILE_SIZE != 0:
            raise ValueError(f"physical prefill and expert chunk must be tile aligned, got {seq_len=} {chunk_size=}")
        if seq_len > chunk_size:
            hidden_chunks = ttnn.split(hidden_states, chunk_size, dim=2)
            routing_chunks = ttnn.split(routing_weights, chunk_size, dim=2)
        else:
            hidden_chunks = [hidden_states]
            routing_chunks = [routing_weights]

        results = []
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        for hidden_chunk, routing_chunk in zip(hidden_chunks, routing_chunks):
            physical_chunk = hidden_chunk.shape[2]
            groups = physical_chunk // TILE_SIZE
            tail_geometry = physical_chunk < self.prefill_expert_chunk_size
            per_core_n = self.prefill_expert_tail_per_core_n if tail_geometry else self.prefill_expert_per_core_n
            gate_block_w = (
                self.prefill_expert_tail_in0_block_w if tail_geometry else self.prefill_expert_gate_in0_block_w
            )
            down_block_w = (
                self.prefill_expert_tail_in0_block_w if tail_geometry else self.prefill_expert_down_in0_block_w
            )
            hidden_grouped = ttnn.reshape(hidden_chunk, (1, groups, TILE_SIZE, HIDDEN_SIZE))
            if self.prefill_expert_input_l1:
                hidden_grouped = ttnn.to_memory_config(
                    hidden_grouped,
                    ttnn.L1_MEMORY_CONFIG,
                    dtype=hidden_grouped.dtype,
                )
            if self.prefill_routed_active:
                # sparse_matmul selects one expert set per 32-token tile group.
                # Use the on-device union of token routes, then retain per-token
                # score weighting below. This preserves exact routing semantics
                # while avoiding the all-128-expert debug topology.
                sparsity = ttnn.max(ttnn.abs(routing_chunk), dim=2, keepdim=True)
                sparsity = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
                nnz = None
            else:
                sparsity = ttnn.repeat(self.expert_prefill_sparsity, (1, 1, groups, 1))
                nnz = NUM_EXPERTS * groups
            common = {
                "sparsity": sparsity,
                "nnz": nnz,
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "output_tile": output_tile,
                "dtype": self._activation_dtype_for("moe"),
                "compute_kernel_config": self.expert_compute_config,
            }
            gate_up_config = _optimized_sparse_prefill_config(
                self.mesh_device,
                n=(
                    packed_expert_width
                    if self.packed_expert_prefill_gate_up
                    else self.expert_weights.intermediate_size_per_device
                ),
                groups=groups,
                requested_per_core_n=per_core_n,
                in0_block_w=gate_block_w,
            )
            down_config = _optimized_sparse_prefill_config(
                self.mesh_device,
                n=HIDDEN_SIZE,
                groups=groups,
                requested_per_core_n=per_core_n,
                in0_block_w=down_block_w,
            )
            if self.packed_expert_prefill_gate_up:
                self.optimized_path_counters["packed_expert_prefill"] += 1
                gate_up = ttnn.sparse_matmul(
                    hidden_grouped,
                    self.prefill_packed_expert_gate_up,
                    program_config=gate_up_config,
                    **common,
                )
                gate_up = ttnn.transpose(gate_up, 1, 3)
                gate_up = ttnn.reshape(gate_up, (1, NUM_EXPERTS, physical_chunk, packed_expert_width))
                down_input = self._packed_expert_activation(gate_up)
            else:
                gate = ttnn.sparse_matmul(
                    hidden_grouped,
                    self.expert_weights.gate_proj,
                    program_config=gate_up_config,
                    **common,
                )
                sparse_intermediate = gate.shape[-1]
                gate = ttnn.transpose(gate, 1, 3)
                gate = ttnn.reshape(gate, (1, NUM_EXPERTS, physical_chunk, sparse_intermediate))
                up = ttnn.sparse_matmul(
                    hidden_grouped,
                    self.expert_weights.up_proj,
                    program_config=gate_up_config,
                    **common,
                )
                up = ttnn.transpose(up, 1, 3)
                up = ttnn.reshape(up, (1, NUM_EXPERTS, physical_chunk, sparse_intermediate))
                down_input = ttnn.reshape(
                    apply_geglu(gate, up),
                    (1, NUM_EXPERTS, physical_chunk, sparse_intermediate),
                )
            down = ttnn.sparse_matmul(
                down_input,
                self.expert_weights.down_proj,
                sparsity=sparsity,
                nnz=nnz,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=down_config,
                is_input_a_sparse=True,
                dtype=self._activation_dtype_for("moe"),
                compute_kernel_config=self.expert_compute_config,
            )
            next_states = ttnn.reshape(down, (1, NUM_EXPERTS, physical_chunk, HIDDEN_SIZE))
            routing_permuted = ttnn.permute(routing_chunk, (0, 3, 2, 1))
            next_states = ttnn.mul(next_states, routing_permuted)
            next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
            results.append(ttnn.reshape(next_states, (1, 1, physical_chunk, HIDDEN_SIZE)))
        return results[0] if len(results) == 1 else ttnn.concat(results, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _moe_decode(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        if self.residual_shard_cores == 0:
            return super()._moe_decode(hidden_states, routing_weights)
        batch = hidden_states.shape[2]
        if batch == 1:
            return self._moe_decode_single_user(hidden_states, routing_weights)

        outputs = []
        for batch_index in range(batch):
            hidden_row = ttnn.slice(
                hidden_states,
                [0, 0, batch_index, 0],
                [1, 1, batch_index + 1, HIDDEN_SIZE],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            routing_row = ttnn.slice(
                routing_weights,
                [0, 0, batch_index, 0],
                [1, 1, batch_index + 1, NUM_EXPERTS],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            outputs.append(self._moe_decode_single_user(hidden_row, routing_row, use_batch32_policy=True))
        return ttnn.concat(outputs, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)

    def _moe_decode_single_user(
        self,
        hidden_states: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
        *,
        use_batch32_policy: bool = False,
    ) -> ttnn.Tensor:
        self.optimized_path_counters["expert_decode"] += 1
        batch = hidden_states.shape[2]
        if self.expert_decode_input_l1:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=hidden_states.dtype)
        row_major_routing = self.routing_row_major and not use_batch32_policy
        if row_major_routing:
            if routing_weights.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise AssertionError("selected routing candidate did not supply row-major sparse metadata")
            sparsity = routing_weights
            self.optimized_path_counters["routing_row_major_metadata"] += 1
        else:
            sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        gate_up_config = _optimized_sparse_decode_config(
            self.mesh_device,
            n=(
                _PACKED_EXPERT_WIDTH
                if self.packed_expert_decode_gate_up
                else self.expert_weights.intermediate_size_per_device
            ),
            per_core_n=self.expert_gate_per_core_n,
            in0_block_w=11 if use_batch32_policy else self.expert_gate_in0_block_w,
            out_subblock_w=self.expert_gate_out_subblock_w,
        )
        down_config = _optimized_sparse_decode_config(
            self.mesh_device,
            n=HIDDEN_SIZE,
            per_core_n=self.expert_down_per_core_n,
            in0_block_w=self.expert_down_in0_block_w,
            out_subblock_w=self.expert_down_out_subblock_w,
        )
        common = {
            "sparsity": sparsity,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "output_tile": output_tile,
            "dtype": self._activation_dtype_for("moe"),
        }
        if self.packed_expert_decode_gate_up:
            self.optimized_path_counters["packed_expert_decode"] += 1
            gate_up_weight = (
                self.decode_packed_expert_gate_up_batch32 if use_batch32_policy else self.decode_packed_expert_gate_up
            )
            gate_up = ttnn.sparse_matmul(
                hidden_states,
                gate_up_weight,
                program_config=gate_up_config,
                compute_kernel_config=self.expert_compute_config,
                **common,
            )
            gate_up = ttnn.reshape(gate_up, (batch, NUM_EXPERTS, 1, _PACKED_EXPERT_WIDTH))
            gate_up = ttnn.transpose(gate_up, 1, 2)
            gate_up = ttnn.reshape(gate_up, (batch, NUM_EXPERTS, _PACKED_EXPERT_WIDTH))
            down_input = self._packed_expert_activation(gate_up)
        else:
            expert_gate = self.batch32_expert_gate if use_batch32_policy else self.weights.expert_gate
            expert_up = self.batch32_expert_up if use_batch32_policy else self.weights.expert_up
            gate = ttnn.sparse_matmul(
                hidden_states,
                expert_gate,
                program_config=gate_up_config,
                compute_kernel_config=self.expert_gate_compute_config,
                **common,
            )
            sparse_intermediate = gate.shape[-1]
            gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, 1, sparse_intermediate))
            gate = ttnn.transpose(gate, 1, 2)
            gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, sparse_intermediate))
            up = ttnn.sparse_matmul(
                hidden_states,
                expert_up,
                program_config=gate_up_config,
                compute_kernel_config=self.expert_compute_config,
                **common,
            )
            up = ttnn.reshape(up, (batch, NUM_EXPERTS, 1, sparse_intermediate))
            up = ttnn.transpose(up, 1, 2)
            up = ttnn.reshape(up, (batch, NUM_EXPERTS, sparse_intermediate))
            down_input = ttnn.mul(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        sparse_intermediate = down_input.shape[-1]
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, NUM_EXPERTS, batch, sparse_intermediate))
        down = ttnn.sparse_matmul(
            down_input,
            self.weights.expert_down,
            program_config=down_config,
            is_input_a_sparse=True,
            compute_kernel_config=self.expert_compute_config,
            **common,
        )
        next_states = ttnn.permute(down, (0, 2, 1, 3))
        next_states = ttnn.reshape(next_states, (batch, NUM_EXPERTS, HIDDEN_SIZE))
        routing_scores = routing_weights
        if row_major_routing:
            routing_scores = ttnn.to_layout(routing_weights, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.optimized_path_counters["routing_score_tilize"] += 1
            self.routing_runtime.update(
                sparsity_layout=str(sparsity.layout),
                score_consumer_layout=str(routing_scores.layout),
                nnz=None,
                nnz_policy="runtime_inferred",
                host_dispatch_count=self.optimized_path_counters["routing_row_major_metadata"],
            )
        routing_3d = ttnn.reshape(routing_scores, (batch, NUM_EXPERTS, 1))
        next_states = ttnn.mul(next_states, routing_3d)
        next_states = ttnn.sum(next_states, dim=1)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        return ttnn.reshape(
            next_states,
            (1, 1, batch, HIDDEN_SIZE),
            (1, 1, max(TILE_SIZE, batch), HIDDEN_SIZE),
        )

    def _packed_expert_activation(self, gate_up: ttnn.Tensor) -> ttnn.Tensor:
        half_width = _PACKED_EXPERT_WIDTH // 2
        up = ttnn.slice(
            gate_up,
            [0] * len(gate_up.shape),
            [*list(gate_up.shape)[:-1], MOE_INTERMEDIATE_SIZE],
            memory_config=gate_up.memory_config(),
        )
        gate = ttnn.slice(
            gate_up,
            [*([0] * (len(gate_up.shape) - 1)), half_width],
            [*list(gate_up.shape)[:-1], half_width + MOE_INTERMEDIATE_SIZE],
            memory_config=gate_up.memory_config(),
        )
        return ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)],
            memory_config=gate_up.memory_config(),
        )


__all__ = ["OptimizedDecoder"]

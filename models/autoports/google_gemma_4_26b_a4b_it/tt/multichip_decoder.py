# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profiled TP=1/2/4 Gemma-4 26B A4B decoder.

The implementation deliberately inherits the selected single-chip
``OptimizedDecoder`` orchestration and numerical policy.  Setup fractures all
material projection and active-expert tensors over a 1xN P300 proxy mesh.
Runtime keeps the public residual replicated across devices, uses the inherited
R22 local residual layout except for the TP4/full correctness exception, and
reduces only row-parallel contraction partials.  Expanded MLP/expert/head
activations therefore never cross the fabric.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    FULL_KIND,
    HIDDEN_SIZE,
    MLP_INTERMEDIATE_SIZE,
    MOE_INTERMEDIATE_SIZE,
    NUM_EXPERTS,
    NUM_Q_HEADS,
    PREFILL_FULL_CHUNK_SIZE,
    PREFILL_SLIDING_CHUNK_SIZE,
    TILE_SIZE,
    TOP_K_EXPERTS,
    _DecoderWeights,
    _detect_layer_prefix,
    _layer_kind,
    _make_decode_height_sharded_memory_config,
    _make_decode_rope_memory_config,
    _make_single_user_cache_update_memory_config,
    _prefill_attention_path,
    _text_config,
    _validate_text_config,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import (
    OptimizedDecoder,
    _dram_sharded_weight_and_config,
    _folded_tensor_cache_path,
    _matrix_rows,
    _optimized_sparse_decode_config,
    _pad_dram_weight_for_readers,
    _prepare_folded_state_dict,
    _r22_dram_weight_and_config,
    _resolved_graph_fusion_policy,
    _width_sharded_memory_config,
)
from models.demos.gemma4.tt.experts.weights import ExpertWeights

TP_SIZE = 4  # maximum/legacy exported profile size
SUPPORTED_TP_SIZES = (1, 2, 4)
ADVERTISED_CONTEXT_TOKENS = 262144
_PREFILL_CONTEXT_LIMIT_BY_TP = {1: 50624, 2: ADVERTISED_CONTEXT_TOKENS, 4: ADVERTISED_CONTEXT_TOKENS}
_ROUTING_ROW_MAJOR_ENV = "GEMMA4_OPT_ROUTING_ROW_MAJOR"
_PACKED_EXPERT_POLICY_FLAGS = (
    "packed_expert_decode_gate_up",
    "packed_expert_prefill_gate_up",
)
_GRAPH_FUSION_POLICY_FLAGS = (
    "folded_router_projection",
    "shared_ffn_norm",
    "folded_expert_scale",
    "fused_final_scalar",
)
_RAW_WEIGHT_POLICY_FLAGS = _PACKED_EXPERT_POLICY_FLAGS + _GRAPH_FUSION_POLICY_FLAGS


@contextmanager
def _multichip_routing_default(enabled: bool = False):
    """Select the prepared policy's routing default without leaking it to callers."""

    if _ROUTING_ROW_MAJOR_ENV in os.environ:
        yield
        return

    os.environ[_ROUTING_ROW_MAJOR_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        os.environ.pop(_ROUTING_ROW_MAJOR_ENV, None)


def _construct_raw_weight_decoder(
    decoder_cls: Any,
    /,
    *,
    _prepared_graph_folds: bool = False,
    _prepared_packed_expert_decode: bool = False,
    _prepared_packed_expert_prefill: bool = False,
    **kwargs: Any,
) -> Any:
    """Construct the inherited runtime without enabling unprepared policies."""

    unsupported = set(_RAW_WEIGHT_POLICY_FLAGS)
    if _prepared_graph_folds:
        unsupported.difference_update(_GRAPH_FUSION_POLICY_FLAGS)
    if _prepared_packed_expert_decode:
        unsupported.discard("packed_expert_decode_gate_up")
    if _prepared_packed_expert_prefill:
        unsupported.discard("packed_expert_prefill_gate_up")
    incompatible = {name: kwargs[name] for name in unsupported if name in kwargs and kwargs[name] is not False}
    if incompatible:
        raise ValueError(
            f"MultichipDecoder raw weights do not support enabled packing/folding policies: {incompatible}"
        )
    for name in _PACKED_EXPERT_POLICY_FLAGS:
        kwargs.setdefault(name, False)
    for name in _GRAPH_FUSION_POLICY_FLAGS:
        kwargs.setdefault(name, False)
    routing_default = bool(
        _prepared_graph_folds and kwargs.get("folded_expert_scale") and kwargs.get("residual_shard_cores") == 22
    )
    with _multichip_routing_default(routing_default):
        return decoder_cls(**kwargs)


def _multichip_optimized_defaults(tp_size: int, layer_kind: Any) -> tuple[dict[str, bool], int]:
    """Selected defaults, including the TP4/full B32 correctness exception."""

    graph = {name: True for name in _GRAPH_FUSION_POLICY_FLAGS}
    residual_shard_cores = 22
    if tp_size == 4 and layer_kind is FULL_KIND:
        graph["folded_router_projection"] = False
        graph["shared_ffn_norm"] = False
        residual_shard_cores = 0
    return graph, residual_shard_cores


def _multichip_graph_fusion_policy(
    kwargs: dict[str, Any],
    *,
    defaults: dict[str, bool] | None = None,
) -> dict[str, bool]:
    """Pop and resolve the selected setup-time folds, with env opt-outs."""

    defaults = defaults or {name: True for name in _GRAPH_FUSION_POLICY_FLAGS}
    requested = {name: bool(kwargs.pop(name, defaults[name])) for name in _GRAPH_FUSION_POLICY_FLAGS}
    return _resolved_graph_fusion_policy(**requested)


def _multichip_residual_shard_cores(default: int = 22) -> int:
    value = os.getenv("GEMMA4_MULTICHIP_RESIDUAL_SHARD_CORES", str(default))
    try:
        cores = int(value)
    except ValueError as error:
        raise ValueError("GEMMA4_MULTICHIP_RESIDUAL_SHARD_CORES must be 0 or 22") from error
    if cores not in (0, 22):
        raise ValueError("GEMMA4_MULTICHIP_RESIDUAL_SHARD_CORES must be 0 or 22")
    return cores


def _multichip_dram_workers_per_bank(role: str | None = None, *, tp_size: int | None = None) -> int:
    base_name = "GEMMA4_MULTICHIP_DRAM_WORKERS_PER_BANK"
    name = (
        f"{base_name}_{role.upper()}" if role is not None and f"{base_name}_{role.upper()}" in os.environ else base_name
    )
    value = os.getenv(name, "1")
    try:
        workers = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be 1, 2, or 3") from error
    if workers not in (1, 2, 3):
        raise ValueError(f"{name} must be 1, 2, or 3")
    if tp_size is not None and tp_size > 1 and workers > 1:
        raise ValueError(
            f"{name}={workers} is unsupported for TP{tp_size}: "
            "the multi-reader DRAM primitive currently requires a unit MeshDevice"
        )
    return workers


def _multichip_bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _router_output_subblock_contract(output_width: int = NUM_EXPERTS, output_cores: int = 4) -> dict[str, Any]:
    """Describe the exact subblock limit of the profiled four-core router."""

    if output_width % TILE_SIZE:
        raise ValueError(f"router output width {output_width} must be tile aligned")
    output_tiles = output_width // TILE_SIZE
    if output_cores < 1 or output_tiles % output_cores:
        raise ValueError(f"router output tiles {output_tiles} must divide over {output_cores} cores")
    per_core_n = output_tiles // output_cores
    return {
        "logical_output_width": output_width,
        "output_tiles": output_tiles,
        "profiled_output_cores": output_cores,
        "per_core_M": 1,
        "per_core_N": per_core_n,
        "max_legal_output_subblock_area": per_core_n,
        "larger_output_subblock_legal": per_core_n >= 2,
    }


@dataclass(frozen=True)
class TPProfile:
    tp_size: int
    padded_mlp_intermediate_size: int
    local_mlp_intermediate_size: int
    padded_moe_intermediate_size: int
    local_moe_intermediate_size: int
    local_q_heads: int
    local_sliding_kv_heads: int
    local_full_kv_heads: int


def _profile_for_tp(tp_size: int) -> TPProfile:
    if tp_size not in SUPPORTED_TP_SIZES:
        raise ValueError(f"Gemma-4 multichip decoder supports TP sizes {SUPPORTED_TP_SIZES}, got {tp_size}")

    alignment = tp_size * TILE_SIZE
    padded_mlp = ((MLP_INTERMEDIATE_SIZE + alignment - 1) // alignment) * alignment
    padded_moe = ((MOE_INTERMEDIATE_SIZE + alignment - 1) // alignment) * alignment
    return TPProfile(
        tp_size=tp_size,
        padded_mlp_intermediate_size=padded_mlp,
        local_mlp_intermediate_size=padded_mlp // tp_size,
        padded_moe_intermediate_size=padded_moe,
        local_moe_intermediate_size=padded_moe // tp_size,
        local_q_heads=NUM_Q_HEADS // tp_size,
        local_sliding_kv_heads=8 // tp_size,
        local_full_kv_heads=max(1, 2 // tp_size),
    )


def _prefill_context_limit(tp_size: int) -> int:
    """Return the per-profile full-stack prefill limit imposed by 32 GiB DRAM."""

    if tp_size not in SUPPORTED_TP_SIZES:
        raise ValueError(f"Gemma-4 multichip decoder supports TP sizes {SUPPORTED_TP_SIZES}, got {tp_size}")
    return _PREFILL_CONTEXT_LIMIT_BY_TP[tp_size]


# Preserve the original TP4 constants for focused shape-repro tests and
# downstream imports; runtime code uses the mesh-derived instance profile.
_TP4_PROFILE = _profile_for_tp(TP_SIZE)
LOCAL_Q_HEADS = _TP4_PROFILE.local_q_heads
PADDED_MLP_INTERMEDIATE_SIZE = _TP4_PROFILE.padded_mlp_intermediate_size
LOCAL_MLP_INTERMEDIATE_SIZE = _TP4_PROFILE.local_mlp_intermediate_size
PADDED_MOE_INTERMEDIATE_SIZE = _TP4_PROFILE.padded_moe_intermediate_size
LOCAL_MOE_INTERMEDIATE_SIZE = _TP4_PROFILE.local_moe_intermediate_size


def _packed_gate_up_mesh_source(gate: torch.Tensor, up: torch.Tensor, tp_size: int = TP_SIZE) -> torch.Tensor:
    """Pack each TP rank's gate/up slice before sharding across the mesh.

    Concatenating the complete gate and up matrices before ``ShardTensorToMesh``
    would put gate-only shards on the first ranks and up-only shards on the last
    ranks.  The production packed matmul instead needs ``[gate_i, up_i]`` on
    every rank.
    """

    import torch

    gate_shards = gate.chunk(tp_size, dim=-1)
    up_shards = up.chunk(tp_size, dim=-1)
    return (
        torch.cat(
            [torch.cat((gate_shards[rank], up_shards[rank]), dim=-1) for rank in range(tp_size)],
            dim=-1,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )


def _packed_expert_gate_up_mesh_source(
    gate: torch.Tensor,
    up: torch.Tensor,
    tp_size: int = TP_SIZE,
) -> torch.Tensor:
    """Pack each rank's expert ``[up, gate]`` pair before mesh sharding."""

    import torch

    gate_shards = gate.chunk(tp_size, dim=-1)
    up_shards = up.chunk(tp_size, dim=-1)
    return torch.cat(
        [torch.cat((up_shards[rank], gate_shards[rank]), dim=-1) for rank in range(tp_size)],
        dim=-1,
    ).unsqueeze(0)


def _upload_multichip_expert_gate_up(
    *,
    gate: Any,
    up: Any,
    tp_size: int,
    gate_dtype: ttnn.DataType,
    up_dtype: ttnn.DataType,
    mapper: Any,
    upload: Any,
    packed_decode: bool,
    packed_prefill: bool,
) -> tuple[Any | None, Any | None, Any | None]:
    """Upload only the expert gate/up forms reachable by the selected phases."""

    packed_both = packed_decode and packed_prefill
    gate_tt = up_tt = None
    if not packed_both:
        gate_tt = upload(f"expert_gate_tp{tp_size}", gate.unsqueeze(0), dtype=gate_dtype, mapper=mapper)
        up_tt = upload(f"expert_up_tp{tp_size}", up.unsqueeze(0), dtype=up_dtype, mapper=mapper)

    packed_tt = None
    if packed_decode or packed_prefill:
        packed_name = "packed_expert_gate_up" if packed_decode else "packed_expert_prefill_gate_up"
        packed_tt = upload(
            f"{packed_name}_tp{tp_size}",
            _packed_expert_gate_up_mesh_source(gate, up, tp_size),
            dtype=gate_dtype,
            mapper=mapper,
        )
    return gate_tt, up_tt, packed_tt


def _require_target_mesh(mesh_device: Any) -> TPProfile:
    if not isinstance(mesh_device, ttnn.MeshDevice):
        raise ValueError("MultichipDecoder requires a TTNN MeshDevice")
    shape = tuple(mesh_device.shape)
    if len(shape) != 2 or shape[0] != 1:
        raise ValueError(f"Gemma-4 multichip decoder requires a 1D 1xN mesh, got {shape}")
    return _profile_for_tp(shape[1])


def _capacity_expert_gate_up_dtype(tp_size: int, layer_kind: Any) -> ttnn.DataType:
    """Lowest validated gate/up dtype needed to leave usable TP1 DRAM headroom."""

    return ttnn.bfloat4_b if tp_size == 1 and layer_kind is FULL_KIND else ttnn.bfloat8_b


def _pad_last(torch_tensor: Any, padded: int) -> Any:
    import torch

    if torch_tensor.shape[-1] == padded:
        return torch_tensor
    if torch_tensor.shape[-1] > padded:
        raise ValueError(f"cannot pad width {torch_tensor.shape[-1]} to smaller width {padded}")
    return torch.nn.functional.pad(torch_tensor, (0, padded - torch_tensor.shape[-1]))


def _pad_penultimate(torch_tensor: Any, padded: int) -> Any:
    import torch

    if torch_tensor.shape[-2] == padded:
        return torch_tensor
    if torch_tensor.shape[-2] > padded:
        raise ValueError(f"cannot pad K {torch_tensor.shape[-2]} to smaller K {padded}")
    return torch.nn.functional.pad(torch_tensor, (0, 0, 0, padded - torch_tensor.shape[-2]))


class MultichipDecoder(OptimizedDecoder):
    """TP=1/2/4 optimized decoder with a replicated stack-compatible residual."""

    tp_size = TP_SIZE
    topology = ttnn.Topology.Ring
    cluster_axis = 1

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
        mlp_weight_dtype: ttnn.DataType | None = None,
        mlp_down_weight_dtype: ttnn.DataType | None = None,
        prefill_expert_weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        expert_weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        expert_gate_weight_dtype: ttnn.DataType | None = None,
        expert_up_weight_dtype: ttnn.DataType | None = None,
        expert_down_weight_dtype: ttnn.DataType | None = None,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        router_weight_dtype: ttnn.DataType = ttnn.float32,
        ccl_dtype: ttnn.DataType = ttnn.bfloat16,
        decode_weight_dtypes: dict[str, ttnn.DataType] | None = None,
        tensor_cache_path: str | Path | None = None,
        persistent_all_reduce_resources: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "MultichipDecoder":
        import torch

        profile = _require_target_mesh(mesh_device)
        tp_size = profile.tp_size
        del prefill_expert_weight_dtype
        dtype_names = {
            "bf16": ttnn.bfloat16,
            "bfp8": ttnn.bfloat8_b,
            "bfp4": ttnn.bfloat4_b,
        }
        fidelity_names = {
            "lofi": ttnn.MathFidelity.LoFi,
            "hifi2": ttnn.MathFidelity.HiFi2,
            "hifi4": ttnn.MathFidelity.HiFi4,
        }

        def dtype_from_env(name: str, current: ttnn.DataType | None) -> ttnn.DataType | None:
            value = os.getenv(name)
            if value is None:
                return current
            if value.lower() not in dtype_names:
                raise ValueError(f"{name}={value!r}; choose from {sorted(dtype_names)}")
            return dtype_names[value.lower()]

        def fidelity_from_env(name: str) -> ttnn.MathFidelity | None:
            value = os.getenv(name)
            if value is None:
                return None
            if value.lower() not in fidelity_names:
                raise ValueError(f"{name}={value!r}; choose from {sorted(fidelity_names)}")
            return fidelity_names[value.lower()]

        attention_weight_dtype = dtype_from_env("GEMMA4_MULTICHIP_ATTENTION_WEIGHT_DTYPE", attention_weight_dtype)
        mlp_weight_dtype = dtype_from_env("GEMMA4_MULTICHIP_MLP_WEIGHT_DTYPE", mlp_weight_dtype)
        mlp_down_weight_dtype = dtype_from_env("GEMMA4_MULTICHIP_MLP_DOWN_WEIGHT_DTYPE", mlp_down_weight_dtype)
        expert_weight_dtype = dtype_from_env("GEMMA4_MULTICHIP_EXPERT_WEIGHT_DTYPE", expert_weight_dtype)
        expert_gate_weight_dtype = dtype_from_env("GEMMA4_MULTICHIP_EXPERT_GATE_WEIGHT_DTYPE", expert_gate_weight_dtype)
        expert_up_weight_dtype = dtype_from_env("GEMMA4_MULTICHIP_EXPERT_UP_WEIGHT_DTYPE", expert_up_weight_dtype)
        expert_down_weight_dtype = dtype_from_env("GEMMA4_MULTICHIP_EXPERT_DOWN_WEIGHT_DTYPE", expert_down_weight_dtype)
        activation_dtype = dtype_from_env("GEMMA4_MULTICHIP_ACTIVATION_DTYPE", activation_dtype)
        attention_activation_dtype = dtype_from_env("GEMMA4_MULTICHIP_ATTENTION_ACTIVATION_DTYPE", activation_dtype)
        dense_mlp_activation_dtype = dtype_from_env("GEMMA4_MULTICHIP_DENSE_MLP_ACTIVATION_DTYPE", activation_dtype)
        moe_activation_dtype = dtype_from_env("GEMMA4_MULTICHIP_MOE_ACTIVATION_DTYPE", activation_dtype)
        residual_norm_dtype = dtype_from_env("GEMMA4_MULTICHIP_RESIDUAL_NORM_DTYPE", activation_dtype)
        ccl_dtype = dtype_from_env("GEMMA4_MULTICHIP_CCL_DTYPE", ccl_dtype)
        if ccl_dtype not in {ttnn.bfloat16, ttnn.bfloat8_b}:
            raise ValueError("GEMMA4_MULTICHIP_CCL_DTYPE must be bf16 or bfp8")
        for env_name, kwarg_name in (
            ("GEMMA4_MULTICHIP_ATTENTION_FIDELITY", "attention_math_fidelity"),
            (
                "GEMMA4_MULTICHIP_RESIDUAL_FULL_ATTENTION_FIDELITY",
                "residual_full_attention_math_fidelity",
            ),
            ("GEMMA4_MULTICHIP_MLP_FIDELITY", "mlp_math_fidelity"),
            ("GEMMA4_MULTICHIP_EXPERT_GATE_FIDELITY", "expert_gate_math_fidelity"),
            ("GEMMA4_MULTICHIP_EXPERT_FIDELITY", "expert_math_fidelity"),
        ):
            fidelity = fidelity_from_env(env_name)
            if fidelity is not None:
                kwargs[kwarg_name] = fidelity
        for env_name, kwarg_name in (
            ("GEMMA4_MULTICHIP_EXPERT_GATE_BLOCK_W", "expert_gate_in0_block_w"),
            ("GEMMA4_MULTICHIP_EXPERT_DOWN_BLOCK_W", "expert_down_in0_block_w"),
            ("GEMMA4_MULTICHIP_EXPERT_GATE_PER_CORE_N", "expert_gate_per_core_n"),
            ("GEMMA4_MULTICHIP_EXPERT_DOWN_PER_CORE_N", "expert_down_per_core_n"),
            ("GEMMA4_MULTICHIP_EXPERT_GATE_OUT_SUBBLOCK_W", "expert_gate_out_subblock_w"),
            ("GEMMA4_MULTICHIP_EXPERT_DOWN_OUT_SUBBLOCK_W", "expert_down_out_subblock_w"),
        ):
            value = os.getenv(env_name)
            if value is not None:
                kwargs[kwarg_name] = int(value)
        text_config = _text_config(hf_config)
        _validate_text_config(text_config)
        kind = _layer_kind(text_config.layer_types[layer_idx])
        kwargs.setdefault(
            "attention_math_fidelity",
            ttnn.MathFidelity.HiFi4 if kind.name == "sliding_attention" else ttnn.MathFidelity.HiFi2,
        )
        kwargs.setdefault("full_attention_math_fidelity", ttnn.MathFidelity.LoFi)
        kwargs.setdefault("prefill_full_attention_math_fidelity", ttnn.MathFidelity.HiFi2)
        kwargs.setdefault("residual_full_attention_math_fidelity", ttnn.MathFidelity.HiFi2)
        kwargs.setdefault("expert_gate_in0_block_w", 44)
        expert_gate_tiles = profile.local_moe_intermediate_size // TILE_SIZE
        kwargs.setdefault("expert_gate_per_core_n", 2 if expert_gate_tiles % 2 == 0 else 1)
        kwargs.setdefault("expert_down_per_core_n", 2)
        # TP2's 11-tile local expert width would otherwise select a single
        # projection core (per_core_n=11) in sparse prefill.  Keeping one tile
        # per core preserves the active-expert topology and avoids that
        # degenerate multi-device sparse program.
        kwargs.setdefault("prefill_expert_per_core_n", 1 if tp_size == 2 else 2)
        prefix = _detect_layer_prefix(state_dict, layer_idx)
        graph_defaults, residual_default = _multichip_optimized_defaults(tp_size, kind)
        graph_fusion_policy = _multichip_graph_fusion_policy(kwargs, defaults=graph_defaults)
        packed_expert_decode = _multichip_bool_from_env(
            "GEMMA4_MULTICHIP_PACKED_EXPERT_DECODE_GATE_UP",
            bool(kwargs.pop("packed_expert_decode_gate_up", True)),
        )
        packed_expert_prefill = _multichip_bool_from_env(
            "GEMMA4_MULTICHIP_PACKED_EXPERT_PREFILL_GATE_UP",
            bool(kwargs.pop("packed_expert_prefill_gate_up", True)),
        )
        original_state_dict = state_dict
        state_dict = _prepare_folded_state_dict(
            state_dict,
            layer_idx=layer_idx,
            folded_router_projection=graph_fusion_policy["folded_router_projection"],
            shared_ffn_norm=graph_fusion_policy["shared_ffn_norm"],
            folded_expert_scale=graph_fusion_policy["folded_expert_scale"],
        )
        cache_root = _folded_tensor_cache_path(
            tensor_cache_path,
            folded_router_projection=graph_fusion_policy["folded_router_projection"],
            shared_ffn_norm=graph_fusion_policy["shared_ffn_norm"],
            folded_expert_scale=graph_fusion_policy["folded_expert_scale"],
        )
        # Four-way sliding-attention contractions at batch 32 need BF16
        # projection weights to stay above the optimized-baseline PCC gate.
        # Full attention clears the gate with BFP8 and keeps that lower-memory
        # policy.  An explicit caller/environment choice still wins.
        attention_weight_dtype = attention_weight_dtype or (
            ttnn.bfloat16 if kind.name == "sliding_attention" else ttnn.bfloat8_b
        )
        # Prompt-derived layer-5 activations expose a full-attention prefill
        # error that random inputs hide: BFP8 dense gate/up reaches only
        # 0.9927 PCC while BF16 reaches 0.9998.  This is the fallback only:
        # explicit policy and environment values must take precedence so
        # datatype candidates reach the uploaded tensors they describe.
        if mlp_weight_dtype is None:
            mlp_weight_dtype = ttnn.bfloat16 if kind.name == "full_attention" else ttnn.bfloat8_b
        mlp_down_weight_dtype = mlp_down_weight_dtype or mlp_weight_dtype
        capacity_expert_dtype = _capacity_expert_gate_up_dtype(tp_size, kind)
        expert_gate_weight_dtype = expert_gate_weight_dtype or capacity_expert_dtype
        expert_up_weight_dtype = expert_up_weight_dtype or capacity_expert_dtype
        expert_down_weight_dtype = expert_down_weight_dtype or expert_weight_dtype
        if (packed_expert_decode or packed_expert_prefill) and expert_gate_weight_dtype != expert_up_weight_dtype:
            raise ValueError("packed expert gate/up requires matching gate and up weight dtypes")

        def get(name: str) -> Any:
            return state_dict[f"{prefix}.{name}"]

        def upload(
            name: str,
            source: Any,
            *,
            dtype: ttnn.DataType,
            mapper: Any,
            layout: ttnn.Layout = ttnn.TILE_LAYOUT,
        ) -> ttnn.Tensor:
            upload_kwargs = {
                "device": mesh_device,
                "layout": layout,
                "dtype": dtype,
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "mesh_mapper": mapper,
            }
            if cache_root is not None:
                upload_kwargs["cache_file_name"] = str(cache_root / "multichip" / f"layer_{layer_idx}" / name)
            return ttnn.as_tensor(source, **upload_kwargs)

        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        shard_n = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
        shard_k = ttnn.ShardTensorToMesh(mesh_device, dim=-2)

        q = get("self_attn.q_proj.weight").transpose(-2, -1).contiguous()
        k = get("self_attn.k_proj.weight").transpose(-2, -1).contiguous()
        v = k if kind.uses_k_as_v else get("self_attn.v_proj.weight").transpose(-2, -1).contiguous()
        q_shards = q.chunk(tp_size, dim=-1)
        if kind is FULL_KIND:
            # TP4 duplicates each of the two KV heads over the pair of ranks
            # whose Q heads consume it.  TP1/2 shard without duplication.
            k_heads = k.chunk(2, dim=-1)
            v_heads = v.chunk(2, dim=-1)
            if tp_size <= 2:
                k_shards = k.chunk(tp_size, dim=-1)
                v_shards = v.chunk(tp_size, dim=-1)
            else:
                k_shards = tuple(k_heads[rank // 2] for rank in range(tp_size))
                v_shards = tuple(v_heads[rank // 2] for rank in range(tp_size))
        else:
            k_shards = k.chunk(tp_size, dim=-1)
            v_shards = v.chunk(tp_size, dim=-1)
        qkv_mesh_source = (
            torch.cat([torch.cat((q_shards[i], k_shards[i], v_shards[i]), dim=-1) for i in range(tp_size)], dim=-1)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        mlp_gate = _pad_last(
            get("mlp.gate_proj.weight").transpose(-2, -1).contiguous(), profile.padded_mlp_intermediate_size
        )
        mlp_up = _pad_last(
            get("mlp.up_proj.weight").transpose(-2, -1).contiguous(), profile.padded_mlp_intermediate_size
        )
        mlp_down = _pad_penultimate(
            get("mlp.down_proj.weight").transpose(-2, -1).contiguous(), profile.padded_mlp_intermediate_size
        )
        gate_up = get("experts.gate_up_proj")
        expert_gate = _pad_last(
            gate_up[:, :MOE_INTERMEDIATE_SIZE, :].transpose(-2, -1).contiguous(),
            profile.padded_moe_intermediate_size,
        )
        expert_up = _pad_last(
            gate_up[:, MOE_INTERMEDIATE_SIZE:, :].transpose(-2, -1).contiguous(),
            profile.padded_moe_intermediate_size,
        )
        expert_down = _pad_penultimate(
            get("experts.down_proj").transpose(-2, -1).contiguous(), profile.padded_moe_intermediate_size
        )
        expert_gate_tt, expert_up_tt, packed_expert_gate_up = _upload_multichip_expert_gate_up(
            gate=expert_gate,
            up=expert_up,
            tp_size=tp_size,
            gate_dtype=expert_gate_weight_dtype,
            up_dtype=expert_up_weight_dtype,
            mapper=shard_n,
            upload=upload,
            packed_decode=packed_expert_decode,
            packed_prefill=packed_expert_prefill,
        )

        def replicated(name: str, source: Any, dtype: ttnn.DataType = weight_dtype, layout=ttnn.TILE_LAYOUT):
            return upload(name, source, dtype=dtype, mapper=replicate, layout=layout)

        weights = _DecoderWeights(
            layer_scalar=replicated("layer_scalar", get("layer_scalar").reshape(1, 1, 1, 1), ttnn.bfloat16),
            input_ln=replicated("input_ln", get("input_layernorm.weight").reshape(1, 1, 1, HIDDEN_SIZE)),
            post_attn_ln=replicated(
                "post_attn_ln", get("post_attention_layernorm.weight").reshape(1, 1, 1, HIDDEN_SIZE)
            ),
            pre_ff_ln=replicated("pre_ff_ln", get("pre_feedforward_layernorm.weight").reshape(1, 1, 1, HIDDEN_SIZE)),
            post_ff_ln=replicated("post_ff_ln", get("post_feedforward_layernorm.weight").reshape(1, 1, 1, HIDDEN_SIZE)),
            post_ff_ln_1=replicated(
                "post_ff_ln_1", get("post_feedforward_layernorm_1.weight").reshape(1, 1, 1, HIDDEN_SIZE)
            ),
            post_ff_ln_2=replicated(
                "post_ff_ln_2", get("post_feedforward_layernorm_2.weight").reshape(1, 1, 1, HIDDEN_SIZE)
            ),
            pre_ff_ln_2=replicated(
                "pre_ff_ln_2", get("pre_feedforward_layernorm_2.weight").reshape(1, 1, 1, HIDDEN_SIZE)
            ),
            q_norm=replicated("q_norm", get("self_attn.q_norm.weight").reshape(1, 1, 1, kind.head_dim)),
            k_norm=replicated("k_norm", get("self_attn.k_norm.weight").reshape(1, 1, 1, kind.head_dim)),
            qkv=upload(f"qkv_tp{tp_size}", qkv_mesh_source, dtype=attention_weight_dtype, mapper=shard_n),
            o_proj=upload(
                f"o_proj_tp{tp_size}",
                get("self_attn.o_proj.weight").transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0),
                dtype=attention_weight_dtype,
                mapper=shard_k,
            ),
            mlp_gate=upload(
                f"mlp_gate_tp{tp_size}", mlp_gate.unsqueeze(0).unsqueeze(0), dtype=mlp_weight_dtype, mapper=shard_n
            ),
            mlp_up=upload(
                f"mlp_up_tp{tp_size}", mlp_up.unsqueeze(0).unsqueeze(0), dtype=mlp_weight_dtype, mapper=shard_n
            ),
            mlp_down=upload(
                f"mlp_down_tp{tp_size}", mlp_down.unsqueeze(0).unsqueeze(0), dtype=mlp_down_weight_dtype, mapper=shard_k
            ),
            router_scale=replicated(
                "router_scale", get("router.scale").reshape(1, 1, 1, HIDDEN_SIZE), router_weight_dtype
            ),
            router_proj=replicated(
                "router_proj",
                get("router.proj.weight").transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0),
                router_weight_dtype,
            ),
            router_per_expert_scale=replicated(
                "router_per_expert_scale", get("router.per_expert_scale").reshape(1, NUM_EXPERTS), router_weight_dtype
            ),
            expert_gate=expert_gate_tt,
            expert_up=expert_up_tt,
            expert_down=upload(
                f"expert_down_tp{tp_size}",
                expert_down.unsqueeze(0),
                dtype=expert_down_weight_dtype,
                mapper=shard_k,
            ),
        )
        sparsity = replicated(
            "expert_prefill_sparsity",
            torch.ones(1, 1, 1, NUM_EXPERTS, dtype=torch.bfloat16),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        expert_down_block_w = 6 if tp_size == 4 else 11
        kwargs.setdefault("expert_down_in0_block_w", expert_down_block_w)
        kwargs.setdefault("prefill_expert_down_in0_block_w", expert_down_block_w)
        residual_shard_cores = _multichip_residual_shard_cores(
            int(kwargs.pop("residual_shard_cores", residual_default))
        )
        decoder = _construct_raw_weight_decoder(
            cls,
            _prepared_graph_folds=True,
            _prepared_packed_expert_decode=packed_expert_decode,
            _prepared_packed_expert_prefill=packed_expert_prefill,
            hf_config=text_config,
            layer_idx=layer_idx,
            layer_kind=kind,
            mesh_device=mesh_device,
            weights=weights,
            expert_prefill_sparsity=sparsity,
            activation_dtype=residual_norm_dtype,
            eps=text_config.rms_norm_eps,
            dense_decode_dram_sharded=False,
            dram_sharded_roles=(),
            residual_shard_cores=residual_shard_cores,
            packed_expert_decode_gate_up=packed_expert_decode,
            packed_expert_prefill_gate_up=packed_expert_prefill,
            **graph_fusion_policy,
            **kwargs,
        )
        decoder.ccl_dtype = ccl_dtype
        decoder.attention_activation_dtype = attention_activation_dtype
        decoder.dense_mlp_activation_dtype = dense_mlp_activation_dtype
        decoder.moe_activation_dtype = moe_activation_dtype
        decoder.residual_norm_activation_dtype = residual_norm_dtype
        decoder.tp_profile = profile
        decoder.tp_size = tp_size
        decoder.topology = ttnn.Topology.Ring if tp_size == 4 else ttnn.Topology.Linear
        decoder.local_q_heads = profile.local_q_heads
        decoder.local_sliding_kv_heads = profile.local_sliding_kv_heads
        decoder.local_full_kv_heads = profile.local_full_kv_heads
        decoder.expert_weights = ExpertWeights(
            gate_proj=weights.expert_gate,
            up_proj=weights.expert_up,
            down_proj=weights.expert_down,
            intermediate_size_per_device=profile.local_moe_intermediate_size,
        )
        decoder.decode_packed_expert_gate_up = None
        decoder.decode_packed_expert_gate_up_batch32 = None
        decoder.batch32_expert_gate = decoder.weights.expert_gate
        decoder.batch32_expert_up = decoder.weights.expert_up
        if packed_expert_decode:
            decoder.decode_packed_expert_gate_up = packed_expert_gate_up
            decoder.decode_packed_expert_gate_up_batch32 = decoder.decode_packed_expert_gate_up
        decoder.prefill_packed_expert_gate_up = None
        decoder.packed_expert_width = 2 * profile.local_moe_intermediate_size
        if packed_expert_prefill:
            decoder.prefill_packed_expert_gate_up = packed_expert_gate_up
        decoder.packed_expert_weights_capacity_neutral = packed_expert_prefill and packed_expert_decode
        if decoder.packed_expert_weights_capacity_neutral:
            if decoder.weights.expert_gate is not None or decoder.weights.expert_up is not None:
                raise AssertionError("packed-both expert setup allocated unreachable separate gate/up tensors")
        decoder.packed_mlp_gate_up = None
        if tp_size == 4 or residual_shard_cores == 0:
            decoder.packed_mlp_gate_up = ttnn.concat(
                [weights.mlp_gate, weights.mlp_up], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        decoder.decode_dram_weights = {}
        decoder.decode_dram_configs = {}
        decoder.decode_dram_batch32_configs = {}
        # OptimizedDecoder's packed dense path consults this even when the
        # rank-local TP weight needs no K padding.
        decoder.decode_dram_padded_input_widths = {}
        decoder.decode_dram_logical_output_widths = {}
        decoder.decode_dram_input_configs = {}
        decoder.decode_dram_output_configs = {}
        decoder.decode_routing_zero_base = None
        decoder.indexed_expert_decode = _multichip_bool_from_env(
            "GEMMA4_MULTICHIP_INDEXED_EXPERT_DECODE",
            packed_expert_decode and graph_fusion_policy["folded_expert_scale"],
        )
        decoder.decode_route_indices = None
        decoder.decode_compact_route_scores = None
        if decoder.routing_row_major:
            decoder.decode_routing_zero_base = ttnn.zeros(
                (1, 1, 1, NUM_EXPERTS),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        decoder.decode_weight_sources = {}
        decoder.multichip_execution_phase = "idle"
        r22_dram_sharded = _multichip_bool_from_env("GEMMA4_MULTICHIP_R22_DRAM_SHARDED", False)
        if r22_dram_sharded and residual_shard_cores != 22:
            raise ValueError("GEMMA4_MULTICHIP_R22_DRAM_SHARDED requires GEMMA4_MULTICHIP_RESIDUAL_SHARD_CORES=22")
        if r22_dram_sharded and kind is not FULL_KIND:
            raise ValueError("GEMMA4_MULTICHIP_R22_DRAM_SHARDED is validated only for full-attention layers")
        # Keep asynchronously produced dtype-conversion tensors alive until
        # their DRAM-sharded descendants and queued transfers are complete.
        decoder.decode_weight_intermediates = []
        if tp_size == 4:
            default_dram_roles = "o_proj,packed_mlp_gate_up,mlp_down"
            if kind is not FULL_KIND:
                default_dram_roles = f"qkv,{default_dram_roles}"
        elif tp_size == 2 and kind is FULL_KIND:
            default_dram_roles = "o_proj"
        else:
            default_dram_roles = ""
        roles_env = "GEMMA4_MULTICHIP_DRAM_SHARDED_ROLES"
        if r22_dram_sharded and roles_env not in os.environ:
            raise ValueError(
                "GEMMA4_MULTICHIP_R22_DRAM_SHARDED requires explicit " "GEMMA4_MULTICHIP_DRAM_SHARDED_ROLES"
            )
        candidate_roles = tuple(
            role.strip() for role in os.getenv(roles_env, default_dram_roles).split(",") if role.strip()
        )
        valid_roles = {"qkv", "o_proj", "mlp_gate", "mlp_up", "packed_mlp_gate_up", "mlp_down"}
        invalid_roles = set(candidate_roles) - valid_roles
        if invalid_roles:
            raise ValueError(
                f"invalid multichip DRAM-sharded roles {sorted(invalid_roles)}; " f"choose from {sorted(valid_roles)}"
            )
        if r22_dram_sharded:
            if not candidate_roles:
                raise ValueError(
                    "GEMMA4_MULTICHIP_R22_DRAM_SHARDED requires explicit " "GEMMA4_MULTICHIP_DRAM_SHARDED_ROLES"
                )
            unsupported_r22_roles = set(candidate_roles) - {"qkv", "mlp_gate", "mlp_up", "mlp_down"}
            if unsupported_r22_roles:
                raise ValueError(
                    "GEMMA4_MULTICHIP_R22_DRAM_SHARDED supports only qkv/mlp_gate/mlp_up/mlp_down; "
                    f"got {sorted(unsupported_r22_roles)}"
                )
            validated_roles = {1: {"qkv", "mlp_gate", "mlp_up", "mlp_down"}, 2: {"mlp_down"}}
            unvalidated_roles = set(candidate_roles) - validated_roles.get(tp_size, set())
            if unvalidated_roles:
                raise ValueError(
                    f"TP{tp_size} R22 DRAM roles {sorted(unvalidated_roles)} are outside the profiled advice matrix"
                )
        dram_candidates = {
            "qkv": weights.qkv,
            "o_proj": weights.o_proj,
            "mlp_gate": weights.mlp_gate,
            "mlp_up": weights.mlp_up,
            "packed_mlp_gate_up": decoder.packed_mlp_gate_up,
            "mlp_down": weights.mlp_down,
        }
        decode_dtype_env = {
            "qkv": "GEMMA4_MULTICHIP_DECODE_QKV_WEIGHT_DTYPE",
            "o_proj": "GEMMA4_MULTICHIP_DECODE_O_WEIGHT_DTYPE",
            "mlp_gate": "GEMMA4_MULTICHIP_DECODE_MLP_GATE_WEIGHT_DTYPE",
            "mlp_up": "GEMMA4_MULTICHIP_DECODE_MLP_UP_WEIGHT_DTYPE",
            "packed_mlp_gate_up": "GEMMA4_MULTICHIP_DECODE_MLP_GATE_UP_WEIGHT_DTYPE",
            "mlp_down": "GEMMA4_MULTICHIP_DECODE_MLP_DOWN_WEIGHT_DTYPE",
        }
        decode_weight_dtypes = decode_weight_dtypes or {}
        for role in candidate_roles:
            if r22_dram_sharded:
                default_block_w = {"qkv": "4", "mlp_gate": "4", "mlp_up": "4", "mlp_down": "3"}[role]
            else:
                default_block_w = {
                    "qkv": "11",
                    # Sliding B1 needs the two-tile accumulation group to avoid a
                    # real-weight HF oracle miss.  B32 retains the four-tile
                    # program below; both reuse this same DRAM-sharded weight.
                    "o_proj": "2" if tp_size == 4 and kind.name == "sliding_attention" else "4",
                    "packed_mlp_gate_up": "11",
                    "mlp_down": str({1: 33, 2: 11, 4: 17}[tp_size]),
                }.get(role)
            role_block_w_env = f"GEMMA4_MULTICHIP_DRAM_BLOCK_W_{role.upper()}"
            role_block_w = os.getenv(role_block_w_env, default_block_w)
            candidate_weight = dram_candidates[role]
            policy_decode_dtype = decode_weight_dtypes.get(role)
            # TP partial sums amplify the BFP4 packed-dense error.  BFP8 is
            # the lowest decode copy that clears PCC 0.995 for both kinds.
            decode_dtype_default = "bfp8" if role == "packed_mlp_gate_up" else None
            decode_dtype_name = os.getenv(decode_dtype_env[role], decode_dtype_default)
            if policy_decode_dtype is not None and os.getenv(decode_dtype_env[role]) is None:
                decode_dtype_name = next(name for name, dtype in dtype_names.items() if dtype == policy_decode_dtype)
            if decode_dtype_name is not None:
                if decode_dtype_name.lower() not in dtype_names:
                    raise ValueError(
                        f"{decode_dtype_env[role]}={decode_dtype_name!r}; choose from {sorted(dtype_names)}"
                    )
                decode_dtype = dtype_names[decode_dtype_name.lower()]
                if role == "packed_mlp_gate_up":
                    # A decode-only precision copy must not be derived from the
                    # prefill device tensor.  Upload an independently packed
                    # host source so its construction, allocation, and lifetime
                    # cannot mutate or alias ``decoder.packed_mlp_gate_up``.
                    candidate_weight = upload(
                        f"packed_mlp_gate_up_decode_{decode_dtype_name.lower()}_tp{tp_size}",
                        _packed_gate_up_mesh_source(mlp_gate, mlp_up, tp_size),
                        dtype=decode_dtype,
                        mapper=shard_n,
                    )
                    decoder.decode_weight_sources[role] = "independent_host_upload"
                else:
                    candidate_weight = ttnn.typecast(
                        candidate_weight,
                        decode_dtype,
                        memory_config=candidate_weight.memory_config(),
                    )
                    decoder.decode_weight_intermediates.append(candidate_weight)
                    decoder.decode_weight_sources[role] = "device_typecast_retained"
            else:
                decoder.decode_weight_sources[role] = "prefill_weight"
            workers_per_bank = _multichip_dram_workers_per_bank(role, tp_size=tp_size)
            if r22_dram_sharded:
                input_cores_default = 11 if role == "mlp_down" and tp_size == 2 else 22
                input_cores = int(
                    os.getenv(f"GEMMA4_MULTICHIP_DRAM_INPUT_CORES_{role.upper()}", str(input_cores_default))
                )
                sharded_weight, config, input_config, output_config, runtime = _r22_dram_weight_and_config(
                    candidate_weight,
                    device=mesh_device,
                    input_cores=input_cores,
                    readers=workers_per_bank,
                    block_w=int(role_block_w),
                )
                decoder.decode_dram_weights[role] = sharded_weight
                decoder.decode_dram_configs[role] = config
                decoder.decode_dram_input_configs[role] = input_config
                decoder.decode_dram_output_configs[role] = output_config
                decoder.r22_projection_runtime[role] = runtime
                continue
            if workers_per_bank > 1 and decoder.decode_weight_sources[role] == "prefill_weight":
                candidate_weight = ttnn.clone(
                    candidate_weight,
                    dtype=candidate_weight.dtype,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                decoder.decode_weight_intermediates.append(candidate_weight)
                decoder.decode_weight_sources[role] = "independent_device_clone"
            unpadded_candidate = candidate_weight
            candidate_weight, padded_input_width, logical_output_width = _pad_dram_weight_for_readers(
                candidate_weight,
                dram_cores=mesh_device.dram_grid_size().x,
                workers_per_bank=workers_per_bank,
                block_w=int(role_block_w) if role_block_w is not None else None,
            )
            if candidate_weight is not unpadded_candidate:
                decoder.decode_weight_intermediates.append(candidate_weight)
            sharded_weight, config, input_config, output_config = _dram_sharded_weight_and_config(
                candidate_weight,
                device=mesh_device,
                block_w=int(role_block_w) if role_block_w is not None else None,
                workers_per_bank=workers_per_bank,
            )
            decoder.decode_dram_weights[role] = sharded_weight
            decoder.decode_dram_configs[role] = config
            if (
                role == "o_proj"
                and tp_size == 4
                and kind.name == "sliding_attention"
                and role_block_w_env not in os.environ
            ):
                decoder.decode_dram_batch32_configs[role] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=4,
                    per_core_M=config.per_core_M,
                    per_core_N=config.per_core_N,
                    fused_activation=config.fused_activation,
                    num_workers_per_dram_bank=config.num_workers_per_dram_bank,
                )
            decoder.decode_dram_input_configs[role] = input_config
            decoder.decode_dram_output_configs[role] = output_config
            if padded_input_width is not None:
                decoder.decode_dram_padded_input_widths[role] = padded_input_width
            if logical_output_width is not None:
                decoder.decode_dram_logical_output_widths[role] = logical_output_width
        decoder.multichip_dram_sharded_roles = frozenset(candidate_roles)
        decoder.r22_dram_sharded = r22_dram_sharded
        decoder.r22_packed_dense_gate_up = False
        decoder.dram_sharded_roles = frozenset(candidate_roles) if r22_dram_sharded else frozenset()
        decoder.router_input_l1 = _multichip_bool_from_env("GEMMA4_MULTICHIP_ROUTER_INPUT_L1", False)
        decoder.optimized_path_counters["router_input_l1"] = 0
        decoder.coherent_r22_dense = _multichip_bool_from_env(
            "GEMMA4_MULTICHIP_COHERENT_R22_DENSE",
            tp_size <= 2,
        )
        if decoder.coherent_r22_dense and tp_size == 4 and residual_shard_cores:
            # TP4's rank-local dense intermediate is 17 tiles wide.  The
            # inherited whole-model R22 down-projection config uses a
            # three-tile accumulation block, which cannot divide that local K.
            # A one-tile block preserves the R22 output contract without
            # padding the public activation or restoring it through DRAM.
            decoder.residual_dense_program_configs["mlp_down"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 2),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=4,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        if os.getenv("GEMMA4_MULTICHIP_PACKED_DENSE_GATE_UP", "1") == "0":
            decoder.packed_dense_gate_up = False
        decoder.persistent_all_reduce_buffers = []
        decoder.persistent_all_reduce_semaphores = []
        decoder.persistent_all_reduce_index = 0
        decoder.persistent_all_reduce_resources = persistent_all_reduce_resources
        # Decode queues three reductions (attention, dense MLP, active experts).
        # Rotating explicit buffers/semaphores prevents the second collective
        # from reusing semaphore-free CCL state while the first is in flight,
        # and is also the trace-replay-safe path.
        persistent_default = "1"
        persistent_enabled = (
            tp_size > 1 and os.getenv("GEMMA4_MULTICHIP_PERSISTENT_ALL_REDUCE", persistent_default) == "1"
        )
        if persistent_enabled and persistent_all_reduce_resources is not None:
            resource_dtype = persistent_all_reduce_resources.get(
                "dtype", persistent_all_reduce_resources["buffers"][0].dtype
            )
            if resource_dtype != ccl_dtype:
                raise ValueError("persistent all-reduce resources and payload must use the same CCL dtype")
            decoder.persistent_all_reduce_buffers = persistent_all_reduce_resources["buffers"]
            decoder.persistent_all_reduce_semaphores = persistent_all_reduce_resources["semaphores"]
            decoder.persistent_all_reduce_memory_config = persistent_all_reduce_resources["memory_config"]
        elif persistent_enabled:
            ccl_grid = mesh_device.compute_with_storage_grid_size()
            ccl_cores = ttnn.num_cores_to_corerangeset(
                ccl_grid.x * ccl_grid.y,
                ccl_grid,
                row_wise=True,
            )
            decoder.persistent_all_reduce_memory_config = _width_sharded_memory_config(
                HIDDEN_SIZE,
                ttnn.CoreGrid(x=11, y=8),
            )
            persistent_buffer_memory_config = _width_sharded_memory_config(
                HIDDEN_SIZE * tp_size,
                ttnn.CoreGrid(x=11, y=8),
            )
            for _ in range(3):
                decoder.persistent_all_reduce_buffers.append(
                    ttnn.from_torch(
                        torch.zeros((1, 1, TILE_SIZE, HIDDEN_SIZE * tp_size), dtype=torch.bfloat16),
                        dtype=ccl_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh_device,
                        memory_config=persistent_buffer_memory_config,
                        mesh_mapper=replicate,
                    )
                )
                decoder.persistent_all_reduce_semaphores.append(ttnn.create_global_semaphore(mesh_device, ccl_cores, 0))
            ttnn.synchronize_device(mesh_device)
            decoder.persistent_all_reduce_resources = {
                "buffers": decoder.persistent_all_reduce_buffers,
                "semaphores": decoder.persistent_all_reduce_semaphores,
                "memory_config": decoder.persistent_all_reduce_memory_config,
                "index": 0,
                "dtype": ccl_dtype,
            }
        obsolete_weights = {}
        if graph_fusion_policy["folded_router_projection"]:
            decoder.weights.router_scale.deallocate(True)
            obsolete_weights["router_scale"] = None
        if graph_fusion_policy["shared_ffn_norm"]:
            decoder.weights.pre_ff_ln.deallocate(True)
            decoder.weights.pre_ff_ln_2.deallocate(True)
            obsolete_weights["pre_ff_ln"] = None
            obsolete_weights["pre_ff_ln_2"] = None
        if graph_fusion_policy["folded_expert_scale"]:
            decoder.weights.router_per_expert_scale.deallocate(True)
            obsolete_weights["router_per_expert_scale"] = None
        if graph_fusion_policy["fused_final_scalar"]:
            decoder.layer_scalar_value = float(original_state_dict[f"{prefix}.layer_scalar"].item())
            decoder.weights.layer_scalar.deallocate(True)
            obsolete_weights["layer_scalar"] = None
        if obsolete_weights:
            decoder.weights = replace(decoder.weights, **obsolete_weights)
        decoder.multichip_path_counters = {
            "all_reduce": 0,
            "attention_tp": 0,
            "dense_tp": 0,
            "expert_tp": 0,
            "collective_output_to_residual": 0,
            "coherent_r22_dense": 0,
        }
        return decoder

    def _decode_residual_output(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Return decode reductions directly in the next residual consumer's layout."""

        if self.multichip_execution_phase != "decode" or not self.residual_shard_cores:
            return tensor
        if tensor.memory_config() == self.residual_memory_config:
            return tensor
        self.multichip_path_counters["collective_output_to_residual"] += 1
        return ttnn.to_memory_config(tensor, self.residual_memory_config, dtype=tensor.dtype)

    def _all_reduce_hidden(self, partial: ttnn.Tensor) -> ttnn.Tensor:
        residual_dtype = self._activation_dtype_for("residual_norm")
        if self.tp_size == 1:
            if partial.dtype != residual_dtype:
                partial = ttnn.typecast(partial, residual_dtype, memory_config=partial.memory_config())
            return self._decode_residual_output(partial)
        self.multichip_path_counters["all_reduce"] += 1
        # A two-rank line has one usable neighbour link.  Requesting the TP4
        # ring's two-link policy on the QB2 1x2 submesh can leave the CCL
        # waiting for a second peer route that is not part of the submesh.
        default_num_links = 1 if self.tp_size == 2 else 2
        num_links = int(os.getenv("GEMMA4_MULTICHIP_ALL_REDUCE_NUM_LINKS", str(default_num_links)))
        ccl_dtype = self.ccl_dtype
        if ccl_dtype != partial.dtype:
            partial = ttnn.typecast(partial, ccl_dtype, memory_config=partial.memory_config())
        if self.persistent_all_reduce_buffers and _matrix_rows(partial) <= TILE_SIZE:
            if self.persistent_all_reduce_resources is not None:
                index = self.persistent_all_reduce_resources["index"]
                self.persistent_all_reduce_resources["index"] = (index + 1) % len(self.persistent_all_reduce_buffers)
            else:
                index = self.persistent_all_reduce_index
                self.persistent_all_reduce_index = (index + 1) % len(self.persistent_all_reduce_buffers)
            l1_partial = ttnn.to_memory_config(
                partial,
                self.persistent_all_reduce_memory_config,
                dtype=partial.dtype,
            )
            reduced = ttnn.experimental.all_reduce_async(
                l1_partial,
                self.persistent_all_reduce_buffers[index],
                cluster_axis=self.cluster_axis,
                mesh_device=self.mesh_device,
                multi_device_global_semaphore=self.persistent_all_reduce_semaphores[index],
                num_links=num_links,
                topology=self.topology,
                memory_config=self.persistent_all_reduce_memory_config,
            )
            if self.multichip_execution_phase == "decode" and self.residual_shard_cores:
                reduced = self._decode_residual_output(reduced)
            else:
                reduced = ttnn.to_memory_config(reduced, ttnn.DRAM_MEMORY_CONFIG, dtype=reduced.dtype)
            return (
                ttnn.typecast(reduced, residual_dtype, memory_config=reduced.memory_config())
                if reduced.dtype != residual_dtype
                else reduced
            )
        reduced = ttnn.all_reduce(
            partial,
            cluster_axis=self.cluster_axis,
            num_links=num_links,
            topology=self.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        reduced = self._decode_residual_output(reduced)
        return (
            ttnn.typecast(reduced, residual_dtype, memory_config=reduced.memory_config())
            if reduced.dtype != residual_dtype
            else reduced
        )

    def _linear(
        self,
        x: ttnn.Tensor,
        weight_name: str,
        *,
        compute_kernel_config: Any,
        logical_rows: int | None = None,
    ) -> ttnn.Tensor:
        decode_candidate = self._use_decode_dram_weight(x, weight_name)
        weight = self.decode_dram_weights[weight_name] if decode_candidate else getattr(self.weights, weight_name)
        kwargs = {}
        if decode_candidate:
            padded_input_width = self.decode_dram_padded_input_widths.get(weight_name)
            if padded_input_width is not None:
                pad_spec = [(0, 0)] * len(x.shape)
                pad_spec[-1] = (0, padded_input_width - x.shape[-1])
                x = ttnn.pad(x, pad_spec, 0.0)
            x = ttnn.to_memory_config(x, self.decode_dram_input_configs[weight_name], dtype=x.dtype)
            kwargs["program_config"] = (
                self.decode_dram_batch32_configs.get(weight_name, self.decode_dram_configs[weight_name])
                if logical_rows is not None and logical_rows > 1
                else self.decode_dram_configs[weight_name]
            )
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
                ends = list(result.shape)
                ends[-1] = logical_output_width
                result = ttnn.slice(
                    result,
                    starts=[0] * len(ends),
                    ends=ends,
                    steps=[1] * len(ends),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
        if weight_name in {"o_proj", "mlp_down"}:
            result = self._all_reduce_hidden(result)
        return result

    def _use_decode_dram_weight(self, x: ttnn.Tensor, weight_name: str) -> bool:
        # Shape alone is ambiguous: a valid prefill can contain exactly one
        # tile (S=32), and batch-32 decode has the same matrix row count.  The
        # public forward entrypoint is the authoritative phase boundary.
        if (
            weight_name == "qkv"
            and self.layer_kind.name == "full_attention"
            and not getattr(self, "r22_dram_sharded", False)
        ):
            return False
        if getattr(self, "r22_dram_sharded", False) and getattr(self, "_decode_logical_batch", x.shape[-2]) != 1:
            return False
        return (
            self.multichip_execution_phase == "decode"
            and weight_name in self.decode_dram_weights
            and _matrix_rows(x) <= TILE_SIZE
        )

    def _router_input_memory_config(self, tokens: int) -> ttnn.MemoryConfig:
        """Select the decode-only router-input L1 advice candidate."""

        if (
            getattr(self, "router_input_l1", False)
            and self.multichip_execution_phase == "decode"
            and tokens <= TILE_SIZE
        ):
            contract = _router_output_subblock_contract()
            if contract["max_legal_output_subblock_area"] != 1:
                raise AssertionError("the profiled four-core router no longer has the recorded 1x1 subblock limit")
            self.optimized_path_counters["router_input_l1"] += 1
            self.routing_runtime.update(
                router_input_candidate="L1",
                router_input_logical_tokens=tokens,
                router_output_subblock_contract=contract,
            )
            return ttnn.L1_MEMORY_CONFIG
        return super()._router_input_memory_config(tokens)

    def prefill_forward(
        self, hidden_states: ttnn.Tensor, *, prefill_start: int = 0, host_page_table=None, **kwargs: Any
    ) -> ttnn.Tensor:
        logical_seq_len = hidden_states.shape[-2]
        context_limit = _prefill_context_limit(self.tp_size)
        if logical_seq_len > context_limit:
            raise ValueError(
                f"TP{self.tp_size} prefill length {logical_seq_len} exceeds the {context_limit}-token "
                "32 GiB full-stack capacity contract"
            )
        if prefill_start < 0:
            raise ValueError("prefill_start must be nonnegative")
        self.multichip_prefill_start = prefill_start
        self.multichip_prefill_page_table = host_page_table
        self.multichip_execution_phase = "prefill"
        try:
            return super().prefill_forward(hidden_states, **kwargs)
        finally:
            self.multichip_prefill_start = 0
            self.multichip_prefill_page_table = None
            self.multichip_execution_phase = "idle"

    def decode_forward(self, *args, **kwargs) -> ttnn.Tensor:
        self.multichip_execution_phase = "decode"
        try:
            return super().decode_forward(*args, **kwargs)
        finally:
            self.multichip_execution_phase = "idle"

    def _cache_view_kwargs(self, *, prefill: bool, cache_position_modulo: int | None = None) -> dict[str, int]:
        """Use this layer's TP-local geometry even when its HMA alias differs."""
        kwargs: dict[str, int] = {"block_size": self.layer_kind.block_size}
        if not prefill:
            kwargs["num_kv_heads"] = (
                self.local_full_kv_heads if self.layer_kind.name == "full_attention" else self.local_sliding_kv_heads
            )
        if cache_position_modulo is not None:
            kwargs["cache_position_modulo"] = cache_position_modulo
        return kwargs

    def _sdpa_cache_view_kwargs(self, *, cache_position_modulo: int | None = None) -> dict[str, object]:
        """Return the SDPA form of the rank-local paged-cache geometry."""
        kwargs: dict[str, object] = {
            "paged_cache_geometry": ttnn.PagedCacheGeometryOverride(
                block_size=self.layer_kind.block_size,
                num_kv_heads=(
                    self.local_full_kv_heads
                    if self.layer_kind.name == "full_attention"
                    else self.local_sliding_kv_heads
                ),
            )
        }
        if cache_position_modulo is not None:
            kwargs["cache_position_modulo"] = cache_position_modulo
        return kwargs

    def _fill_prefill_cache(
        self,
        key_cache,
        value_cache,
        k_heads,
        v_heads,
        page_table,
        *,
        user_id,
        logical_seq_len,
        cache_position_modulo,
    ) -> None:
        """Modulo-safe cache fill using TP-local, rather than global, KV heads."""
        prefill_start = getattr(self, "multichip_prefill_start", 0)
        block_size = self.layer_kind.block_size
        # Fill kernels start at page column zero. Shift the table for complete
        # tiles beginning at a page boundary; update only real edge rows.
        leading = min(logical_seq_len, (-prefill_start) % block_size)
        aligned_prefix = ((logical_seq_len - leading) // TILE_SIZE) * TILE_SIZE
        tail_positions = tuple(range(leading)) + tuple(range(leading + aligned_prefix, logical_seq_len))
        if cache_position_modulo is not None:
            # Earlier edge rows may share ring addresses with newer bulk
            # rows. Only the final ring capacity survives this chunk.
            tail_positions = tuple(
                position for position in tail_positions if position >= logical_seq_len - cache_position_modulo
            )
        fill_kwargs = self._cache_view_kwargs(prefill=True, cache_position_modulo=cache_position_modulo)
        if k_heads.dtype != key_cache.dtype:
            k_heads = ttnn.typecast(k_heads, key_cache.dtype, memory_config=k_heads.memory_config())
        if v_heads.dtype != value_cache.dtype:
            v_heads = ttnn.typecast(v_heads, value_cache.dtype, memory_config=v_heads.memory_config())
        if aligned_prefix:
            first_page = (prefill_start + leading) // block_size
            fill_table = page_table
            if cache_position_modulo is not None:
                page_count = cache_position_modulo // block_size
                first_page %= page_count
                if first_page:
                    fill_table = ttnn.concat(
                        [
                            ttnn.slice(page_table, [0, first_page], [page_table.shape[0], page_count]),
                            ttnn.slice(page_table, [0, 0], [page_table.shape[0], first_page]),
                        ],
                        dim=1,
                    )
            elif first_page:
                fill_table = ttnn.slice(page_table, [0, first_page], [page_table.shape[0], page_table.shape[1]])
            for cache, heads in ((key_cache, k_heads), (value_cache, v_heads)):
                prefix = (
                    heads
                    if leading == 0 and aligned_prefix == heads.shape[2]
                    else ttnn.slice(
                        heads, [0, 0, leading, 0], [1, heads.shape[1], leading + aligned_prefix, heads.shape[3]]
                    )
                )
                ttnn.experimental.paged_fill_cache(cache, prefix, fill_table, batch_idx=user_id, **fill_kwargs)
                if prefix is not heads:
                    prefix.deallocate(True)
            if fill_table is not page_table:
                fill_table.deallocate(True)
        if not tail_positions:
            return
        page_table_row = page_table
        owns_page_table_row = False
        if page_table.shape[0] > 1:
            page_table_row = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
            owns_page_table_row = True
        update_mem = _make_single_user_cache_update_memory_config(self.mesh_device, self.layer_kind.head_dim)
        update_kwargs = self._cache_view_kwargs(prefill=False, cache_position_modulo=cache_position_modulo)
        local_kv_heads = k_heads.shape[1]
        for position in tail_positions:
            k_token = ttnn.slice(k_heads, [0, 0, position, 0], [1, local_kv_heads, position + 1, k_heads.shape[3]])
            v_token = ttnn.slice(v_heads, [0, 0, position, 0], [1, local_kv_heads, position + 1, v_heads.shape[3]])
            if k_token.dtype != ttnn.bfloat16:
                k_token = ttnn.typecast(k_token, ttnn.bfloat16, memory_config=k_token.memory_config())
            if v_token.dtype != ttnn.bfloat16:
                v_token = ttnn.typecast(v_token, ttnn.bfloat16, memory_config=v_token.memory_config())
            k_token = ttnn.to_memory_config(ttnn.transpose(k_token, 1, 2), update_mem, dtype=k_token.dtype)
            v_token = ttnn.to_memory_config(ttnn.transpose(v_token, 1, 2), update_mem, dtype=v_token.dtype)
            position_tensor = ttnn.full(
                (1,),
                prefill_start + position,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
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

    def _attention_prefill(self, x: ttnn.Tensor, **call: Any) -> ttnn.Tensor:
        self.optimized_path_counters["prefill_attention"] += 1
        self.multichip_path_counters["attention_tp"] += 1
        kind = self.layer_kind
        seq_len = x.shape[-2]
        local_kv_heads = self.local_full_kv_heads if kind.name == "full_attention" else self.local_sliding_kv_heads
        xqkv = self._linear(x, "qkv", compute_kernel_config=self.prefill_attention_compute_config)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.local_q_heads,
            num_kv_heads=local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_heads = self._rms_norm(q_heads, self.weights.q_norm)
        k_heads = self._rms_norm(k_heads, self.weights.k_norm)
        v_heads = self._rms_norm(v_heads, None)
        q_heads = ttnn.experimental.rotary_embedding_hf(
            q_heads, call["position_cos"], call["position_sin"], is_decode_mode=False
        )
        k_heads = ttnn.experimental.rotary_embedding_hf(
            k_heads, call["position_cos"], call["position_sin"], is_decode_mode=False
        )
        key_cache, value_cache = call["kv_cache"]
        fill_table = call.get("chunk_page_table")
        if fill_table is None:
            fill_table = call["page_table"]
        prefill_start = getattr(self, "multichip_prefill_start", 0)
        history = None
        if prefill_start and kind.sliding_window is not None:
            history = self._prefill_sliding_history(
                key_cache, value_cache, prefill_start, call.get("cache_position_modulo")
            )
        self._fill_prefill_cache(
            key_cache,
            value_cache,
            k_heads,
            v_heads,
            fill_table,
            user_id=call["user_id"],
            logical_seq_len=call["logical_seq_len"],
            cache_position_modulo=call.get("cache_position_modulo"),
        )
        attention_path = _prefill_attention_path(
            seq_len,
            is_sliding=kind.sliding_window is not None,
            has_paged_cache=fill_table is not None,
        )
        if history is not None:
            attn_out = self._sliding_continuation_attention(q_heads, k_heads, v_heads, history)
        elif prefill_start and kind.sliding_window is None:
            attn_out = self._full_chunked_prefill_attention(
                q_heads,
                key_cache,
                value_cache,
                call["page_table"],
                user_id=call["user_id"],
                prefill_start=prefill_start,
                logical_seq_len=call["logical_seq_len"],
            )
        elif attention_path == "sliding_chunked":
            attn_out = self._sliding_chunked_prefill_attention(q_heads, k_heads, v_heads)
        elif attention_path == "full_chunked":
            attn_out = self._full_chunked_prefill_attention(
                q_heads, key_cache, value_cache, fill_table, user_id=call["user_id"]
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
        attn_out = ttnn.reshape(attn_out, [1, self.local_q_heads, seq_len, kind.head_dim])
        attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._all_reduce_hidden(
            ttnn.linear(
                attn_out,
                self.weights.o_proj,
                dtype=self._activation_dtype_for("attention"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.prefill_attention_compute_config,
            )
        )

    def _prefill_sliding_history(self, key_cache, value_cache, prefill_start, cache_position_modulo):
        """Snapshot the external cache before the current chunk can overwrite it."""
        window = self.layer_kind.sliding_window
        table = self.multichip_prefill_page_table
        if table is None:
            raise ValueError("sliding continuation prefill requires the current host page-table metadata")
        first_position = max(0, prefill_start - window + 1)
        block_size = self.layer_kind.block_size
        first_page = first_position // block_size
        last_page = (prefill_start - 1) // block_size
        page_ids = []
        for virtual_page in range(first_page, last_page + 1):
            column = (
                virtual_page if cache_position_modulo is None else virtual_page % (cache_position_modulo // block_size)
            )
            if column >= len(table):
                raise ValueError("sliding continuation page table does not cover its history")
            page_ids.append(int(table[column]))
        result = []
        for cache in (key_cache, value_cache):
            shape = [cache.shape[0], self.local_sliding_kv_heads, block_size, self.layer_kind.head_dim]
            # This is a raw tile reinterpretation of an HMA alias, matching
            # the cache op geometry, rather than a value-preserving reshape.
            view = cache if list(cache.shape) == shape else ttnn.experimental.view(cache, shape)
            pages = [ttnn.slice(view, [page, 0, 0, 0], [page + 1, *shape[1:]]) for page in page_ids]
            joined = ttnn.concat(pages, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG) if len(pages) > 1 else pages[0]
            if len(pages) > 1:
                for page in pages:
                    page.deallocate(True)
            drop = first_position % block_size
            history = ttnn.slice(
                joined, [0, 0, drop, 0], [1, shape[1], drop + prefill_start - first_position, shape[3]]
            )
            # A complete slice can share its input buffer.
            if drop or history.shape[2] != joined.shape[2]:
                joined.deallocate(True)
            result.append(history)
        return result

    def _sliding_continuation_attention(self, q_heads, k_heads, v_heads, history):
        history_len = history[0].shape[2]
        seq_len = q_heads.shape[2]
        dummy_q = ttnn.zeros(
            [1, self.local_q_heads, history_len, self.layer_kind.head_dim],
            dtype=q_heads.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        joined = [
            ttnn.concat([past, current], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for past, current in zip([dummy_q, *history], [q_heads, k_heads, v_heads])
        ]
        dummy_q.deallocate(True)
        for past in history:
            past.deallocate(True)
        padded_len = ((history_len + seq_len + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        if padded_len != history_len + seq_len:
            joined = [
                ttnn.pad(item, [(0, 0), (0, 0), (0, padded_len - history_len - seq_len), (0, 0)], 0.0)
                for item in joined
            ]
        output = self._sliding_chunked_prefill_attention(*joined)
        result = ttnn.slice(
            output, [0, 0, history_len, 0], [1, self.local_q_heads, history_len + seq_len, self.layer_kind.head_dim]
        )
        output.deallocate(True)
        for item in joined:
            item.deallocate(True)
        return result

    def _full_chunked_prefill_attention(
        self,
        q_heads,
        key_cache,
        value_cache,
        page_table,
        *,
        user_id: int,
        prefill_start: int = 0,
        logical_seq_len: int | None = None,
    ) -> ttnn.Tensor:
        """Run the baseline paged long-prefill algorithm with four local Q heads."""
        original_len = q_heads.shape[2]
        logical_seq_len = original_len if logical_seq_len is None else logical_seq_len
        leading = prefill_start % TILE_SIZE
        aligned_start = prefill_start - leading
        # Only the query axis is aligned. Cache addresses and causal positions
        # retain their absolute values, including a nonaligned logical tail.
        required_len = ((leading + logical_seq_len + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        if leading:
            prefix = ttnn.zeros(
                [1, self.local_q_heads, leading, self.layer_kind.head_dim],
                dtype=q_heads.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            q_heads = ttnn.concat([prefix, q_heads], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            prefix.deallocate(True)
            if q_heads.shape[2] < required_len:
                q_heads = ttnn.pad(q_heads, [(0, 0), (0, 0), (0, required_len - q_heads.shape[2]), (0, 0)], 0.0)
        if q_heads.shape[2] != required_len:
            q_heads = ttnn.slice(q_heads, [0, 0, 0, 0], [1, self.local_q_heads, required_len, self.layer_kind.head_dim])
        num_pages = page_table.shape[-1]
        user_page_table = page_table
        owns_user_page_table = False
        if page_table.shape[0] > 1:
            user_page_table = ttnn.slice(page_table, [user_id, 0], [user_id + 1, num_pages])
            owns_user_page_table = True
        outputs = []
        seq_len = q_heads.shape[-2]
        for start in range(0, seq_len, PREFILL_FULL_CHUNK_SIZE):
            chunk_len = min(PREFILL_FULL_CHUNK_SIZE, seq_len - start)
            q_chunk = ttnn.slice(
                q_heads, [0, 0, start, 0], [1, self.local_q_heads, start + chunk_len, self.layer_kind.head_dim]
            )
            output = ttnn.transformer.chunked_scaled_dot_product_attention(
                q_chunk,
                key_cache,
                value_cache,
                user_page_table,
                chunk_start_idx=aligned_start + start,
                program_config=(
                    ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                        q_chunk_size=32,
                        k_chunk_size=32,
                    )
                    if (aligned_start + start) % 128
                    else None
                ),
                scale=1.0,
                **self._sdpa_cache_view_kwargs(),
                compute_kernel_config=self.correctness_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            q_chunk.deallocate(True)
            outputs.append(output)
        if owns_user_page_table:
            user_page_table.deallocate(True)
        if len(outputs) == 1:
            result = outputs[0]
            if leading or result.shape[2] != original_len:
                result = ttnn.slice(
                    result,
                    [0, 0, leading, 0],
                    [1, self.local_q_heads, leading + logical_seq_len, self.layer_kind.head_dim],
                )
                if logical_seq_len != original_len:
                    result = ttnn.pad(result, [(0, 0), (0, 0), (0, original_len - logical_seq_len), (0, 0)], 0.0)
            return result
        # The complete Q tensor is dead once every chunk has been dispatched,
        # and it has exactly the shape/dtype required by the concatenated
        # result. Release that contiguous allocation before concat so the
        # allocator can reuse it instead of requiring two full-sequence
        # attention buffers to coexist at the long-context boundary.
        q_heads.deallocate(True)
        result = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for output in outputs:
            output.deallocate(True)
        if leading or result.shape[2] != original_len:
            result = ttnn.slice(
                result, [0, 0, leading, 0], [1, self.local_q_heads, leading + logical_seq_len, self.layer_kind.head_dim]
            )
            if logical_seq_len != original_len:
                result = ttnn.pad(result, [(0, 0), (0, 0), (0, original_len - logical_seq_len), (0, 0)], 0.0)
        return result

    def _sliding_chunked_prefill_attention(self, q_heads, k_heads, v_heads) -> ttnn.Tensor:
        """Run the baseline windowed long-prefill algorithm with TP-local heads."""
        seq_len = q_heads.shape[-2]
        history = ((self.layer_kind.sliding_window + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        outputs = []
        for start in range(0, seq_len, PREFILL_SLIDING_CHUNK_SIZE):
            output_len = min(PREFILL_SLIDING_CHUNK_SIZE, seq_len - start)
            slice_start = max(0, start - history)
            slice_end = start + output_len
            q_slice = ttnn.slice(
                q_heads, [0, 0, slice_start, 0], [1, self.local_q_heads, slice_end, self.layer_kind.head_dim]
            )
            k_slice = ttnn.slice(
                k_heads, [0, 0, slice_start, 0], [1, k_heads.shape[1], slice_end, self.layer_kind.head_dim]
            )
            v_slice = ttnn.slice(
                v_heads, [0, 0, slice_start, 0], [1, v_heads.shape[1], slice_end, self.layer_kind.head_dim]
            )
            output = ttnn.transformer.scaled_dot_product_attention(
                q_slice,
                k_slice,
                v_slice,
                is_causal=True,
                sliding_window_size=self.layer_kind.sliding_window,
                scale=1.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            q_slice.deallocate(True)
            k_slice.deallocate(True)
            v_slice.deallocate(True)
            drop = start - slice_start
            if drop:
                full_output = output
                output = ttnn.slice(
                    full_output,
                    [0, 0, drop, 0],
                    [1, self.local_q_heads, slice_end - slice_start, self.layer_kind.head_dim],
                )
                full_output.deallocate(True)
            outputs.append(output)
        if len(outputs) == 1:
            return outputs[0]
        result = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for output in outputs:
            output.deallocate(True)
        return result

    def _attention_decode(self, x: ttnn.Tensor, **call: Any) -> ttnn.Tensor:
        self.optimized_path_counters["decode_attention"] += 1
        self.multichip_path_counters["attention_tp"] += 1
        kind = self.layer_kind
        batch = x.shape[-2]
        local_kv_heads = self.local_full_kv_heads if kind.name == "full_attention" else self.local_sliding_kv_heads
        r22_qkv = self._use_r22_dram_weight(x, "qkv")
        if x.is_sharded() and not r22_qkv:
            x = self._tracked_sharded_to_interleaved(
                x,
                ttnn.DRAM_MEMORY_CONFIG,
                "attention_qkv_input",
            )
        if r22_qkv:
            self.optimized_path_counters["attention_qkv_input"] += 1
            xqkv = self._r22_dram_linear(x, "qkv", compute_kernel_config=self.decode_attention_compute_config)
            sharded_xqkv = xqkv
            xqkv = ttnn.sharded_to_interleaved(sharded_xqkv, ttnn.L1_MEMORY_CONFIG)
            sharded_xqkv.deallocate(True)
        else:
            xqkv = self._linear(x, "qkv", compute_kernel_config=self.decode_attention_compute_config)
        if xqkv.dtype == ttnn.bfloat8_b:
            bf16_xqkv = ttnn.typecast(xqkv, ttnn.bfloat16)
            xqkv.deallocate(True)
            xqkv = bf16_xqkv
        head_mem = _make_decode_height_sharded_memory_config(self.mesh_device, batch, kind.head_dim)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv, num_heads=self.local_q_heads, num_kv_heads=local_kv_heads, memory_config=head_mem
        )
        q_heads = ttnn.to_memory_config(q_heads, ttnn.L1_MEMORY_CONFIG, dtype=q_heads.dtype)
        k_heads = ttnn.to_memory_config(k_heads, ttnn.L1_MEMORY_CONFIG, dtype=k_heads.dtype)
        v_heads = ttnn.to_memory_config(v_heads, ttnn.L1_MEMORY_CONFIG, dtype=v_heads.dtype)
        q_heads = self._rms_norm(q_heads, self.weights.q_norm)
        k_heads = self._rms_norm(k_heads, self.weights.k_norm)
        v_heads = self._rms_norm(v_heads, None)
        if kind.name == "full_attention":
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.experimental.rotary_embedding_hf(
                q_heads, call["position_cos"], call["position_sin"], is_decode_mode=False
            )
            k_heads = ttnn.experimental.rotary_embedding_hf(
                k_heads, call["position_cos"], call["position_sin"], is_decode_mode=False
            )
            q_heads = ttnn.to_memory_config(ttnn.transpose(q_heads, 1, 2), head_mem, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(ttnn.transpose(k_heads, 1, 2), head_mem, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, head_mem, dtype=v_heads.dtype)
        else:
            q_heads = ttnn.to_memory_config(q_heads, head_mem, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(k_heads, head_mem, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, head_mem, dtype=v_heads.dtype)
            rope_mem = _make_decode_rope_memory_config(self.mesh_device, batch, kind.head_dim)
            cos = ttnn.interleaved_to_sharded(call["position_cos"], rope_mem)
            sin = ttnn.interleaved_to_sharded(call["position_sin"], rope_mem)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, cos, sin, is_decode_mode=True)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, cos, sin, is_decode_mode=True)
        key_cache, value_cache = call["kv_cache"]
        # paged_update_cache accepts BF16/FP32 input and performs the cache
        # repack itself; passing an already-packed BFP8 token is illegal.
        if k_heads.dtype != ttnn.bfloat16:
            k_heads = ttnn.typecast(k_heads, ttnn.bfloat16, memory_config=k_heads.memory_config())
        if v_heads.dtype != ttnn.bfloat16:
            v_heads = ttnn.typecast(v_heads, ttnn.bfloat16, memory_config=v_heads.memory_config())
        cache_view = self._cache_view_kwargs(prefill=False, cache_position_modulo=call.get("cache_position_modulo"))
        for cache, value in ((key_cache, k_heads), (value_cache, v_heads)):
            ttnn.experimental.paged_update_cache(
                cache,
                value,
                update_idxs_tensor=call["current_pos"],
                page_table=call["page_table"],
                **cache_view,
            )
        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_heads,
            key_cache,
            value_cache,
            page_table_tensor=call["page_table"],
            cur_pos_tensor=call["current_pos"],
            scale=1.0,
            sliding_window_size=kind.sliding_window,
            program_config=self.sdpa_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **self._sdpa_cache_view_kwargs(cache_position_modulo=call.get("cache_position_modulo")),
        )
        attn_out = ttnn.to_memory_config(attn_out, head_mem, dtype=attn_out.dtype)
        attn_out = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=self.local_q_heads)
        attn_out = ttnn.sharded_to_interleaved(attn_out, ttnn.DRAM_MEMORY_CONFIG)
        result = self._linear(
            attn_out,
            "o_proj",
            compute_kernel_config=self.decode_attention_compute_config,
            logical_rows=batch,
        )
        if result.shape[-2] != batch:
            result = ttnn.slice(result, [0, 0, 0, 0], [1, 1, batch, HIDDEN_SIZE])
        if self.residual_shard_cores:
            result = self._tracked_to_memory_config(
                result,
                self.residual_memory_config,
                "attention_output",
            )
        return result

    def _dense_mlp(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.multichip_path_counters["dense_tp"] += 1
        sharded_input = x.is_sharded()
        if sharded_input and self.coherent_r22_dense:
            self.multichip_path_counters["coherent_r22_dense"] += 1
            local = self._dense_mlp_residual_sharded(x)
            return self._all_reduce_hidden(local)
        if sharded_input:
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        result = super()._dense_mlp(x)
        if sharded_input:
            result = ttnn.to_memory_config(result, self.residual_memory_config, dtype=result.dtype)
        return result

    def _moe_decode(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        self.multichip_path_counters["expert_tp"] += 1
        if self.packed_expert_decode_gate_up:
            batch = hidden_states.shape[2]
            if batch == 1:
                local = self._moe_decode_single_user(
                    hidden_states,
                    routing_weights,
                    route_indices=self.decode_route_indices,
                    compact_route_scores=self.decode_compact_route_scores,
                )
            else:
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
                    route_indices = None
                    compact_route_scores = None
                    if self.indexed_expert_decode:
                        route_indices = ttnn.slice(
                            self.decode_route_indices,
                            [0, 0, batch_index, 0],
                            [1, 1, batch_index + 1, TOP_K_EXPERTS],
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        compact_route_scores = ttnn.slice(
                            self.decode_compact_route_scores,
                            [0, 0, batch_index, 0],
                            [1, 1, batch_index + 1, TOP_K_EXPERTS],
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    outputs.append(
                        self._moe_decode_single_user(
                            hidden_row,
                            routing_row,
                            use_batch32_policy=True,
                            route_indices=route_indices,
                            compact_route_scores=compact_route_scores,
                        )
                    )
                local = ttnn.concat(outputs, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
            return self._all_reduce_hidden(local)
        return self._all_reduce_hidden(super()._moe_decode(hidden_states, routing_weights))

    def _moe_decode_single_user(
        self,
        hidden_states: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
        *,
        use_batch32_policy: bool = False,
        route_indices: ttnn.Tensor | None = None,
        compact_route_scores: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Run the packed optimized expert path with a TP-local packed width."""

        if not self.packed_expert_decode_gate_up:
            return super()._moe_decode_single_user(
                hidden_states,
                routing_weights,
                use_batch32_policy=use_batch32_policy,
            )

        self.optimized_path_counters["expert_decode"] += 1
        self.optimized_path_counters["packed_expert_decode"] += 1
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

        local_width = self.tp_profile.local_moe_intermediate_size
        packed_width = 2 * local_width
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        gate_up_config = _optimized_sparse_decode_config(
            self.mesh_device,
            n=packed_width,
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
        indexed = self.indexed_expert_decode and route_indices is not None and compact_route_scores is not None
        if indexed:
            common.update(indices=route_indices, is_input_b_sparse=True)
        packed_weight = (
            self.decode_packed_expert_gate_up_batch32 if use_batch32_policy else self.decode_packed_expert_gate_up
        )
        gate_up = ttnn.sparse_matmul(
            hidden_states,
            packed_weight,
            program_config=gate_up_config,
            compute_kernel_config=self.expert_gate_compute_config,
            **common,
        )
        if indexed:
            gate_up = ttnn.reshape(gate_up, (1, TOP_K_EXPERTS, 1, packed_width))
            down_input = self._packed_expert_activation(gate_up)
            down = ttnn.sparse_matmul(
                down_input,
                self.weights.expert_down,
                program_config=down_config,
                is_input_a_sparse=True,
                compute_kernel_config=self.expert_compute_config,
                **common,
            )
            down = ttnn.reshape(down, (1, TOP_K_EXPERTS, 1, HIDDEN_SIZE))
            compact_scores = ttnn.permute(compact_route_scores, (0, 3, 2, 1))
            next_states = ttnn.mul(down, compact_scores)
            next_states = ttnn.sum(next_states, dim=1)
            next_states = ttnn.unsqueeze_to_4D(next_states)
            return ttnn.reshape(next_states, (1, 1, 1, HIDDEN_SIZE), (1, 1, TILE_SIZE, HIDDEN_SIZE))
        gate_up = ttnn.reshape(gate_up, (batch, NUM_EXPERTS, 1, packed_width))
        gate_up = ttnn.transpose(gate_up, 1, 2)
        gate_up = ttnn.reshape(gate_up, (batch, NUM_EXPERTS, packed_width))
        down_input = self._packed_expert_activation(gate_up)
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, NUM_EXPERTS, batch, local_width))
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
            routing_scores = ttnn.to_layout(
                routing_weights,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.optimized_path_counters["routing_score_tilize"] += 1
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
        """Apply GeGLU to rank-local ``[up, gate]`` packed expert output."""

        local_width = self.tp_profile.local_moe_intermediate_size
        up = ttnn.slice(
            gate_up,
            [0] * len(gate_up.shape),
            [*list(gate_up.shape)[:-1], local_width],
            memory_config=gate_up.memory_config(),
        )
        gate = ttnn.slice(
            gate_up,
            [*([0] * (len(gate_up.shape) - 1)), local_width],
            [*list(gate_up.shape)[:-1], 2 * local_width],
            memory_config=gate_up.memory_config(),
        )
        return ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)],
            memory_config=gate_up.memory_config(),
        )

    def _moe_prefill(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        self.multichip_path_counters["expert_tp"] += 1
        return self._all_reduce_hidden(super()._moe_prefill(hidden_states, routing_weights))


__all__ = ["MultichipDecoder", "_packed_gate_up_mesh_source"]

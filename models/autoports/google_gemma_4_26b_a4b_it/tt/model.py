# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full TP=1/2/4 TTNN autoregressive model for google/gemma-4-26B-A4B-it.

This wrapper intentionally stacks :class:`MultichipDecoder` without changing
its replicated-BF16 inter-layer residual contract.  Embedding weights are
hidden-dimension sharded and gathered once at model entry.  The tied LM head
is vocabulary sharded and leaves its logits sharded for split sampling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    FULL_KIND,
    HIDDEN_SIZE,
    SLIDING_KIND,
    _layer_kind,
    _text_config,
    _validate_text_config,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import (
    SUPPORTED_TP_SIZES,
    TP_SIZE,
    MultichipDecoder,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.precision_policy import (
    decoder_kwargs,
    dtype_from_policy,
    dtype_name,
    fidelity_name,
    load_precision_policy,
    profile_policy,
    weight_dtype_from_policy,
)
from models.demos.gemma4.config import MeshConfig, ModeConfig

MODEL_ID = "google/gemma-4-26B-A4B-it"
DEFAULT_MAX_CONTEXT = 262_144
SLIDING_CACHE_TOKENS = 1_024
DECODE_SLOT_COUNT = 32
PROFILE_CONTEXT_LIMITS = {1: 50_624, 2: DEFAULT_MAX_CONTEXT, 4: DEFAULT_MAX_CONTEXT}
PROFILE_EMBEDDING_STORAGE = {tp_size: "bf16_row_major" for tp_size in SUPPORTED_TP_SIZES}


def _require_supported_mesh(mesh_device: Any) -> int:
    shape = tuple(mesh_device.shape) if hasattr(mesh_device, "shape") else None
    if not isinstance(mesh_device, ttnn.MeshDevice) or shape is None or len(shape) != 2 or shape[0] != 1:
        raise ValueError(f"Gemma4FullModel requires an optimized 1xN mesh, got {shape}")
    tp_size = int(shape[1])
    if tp_size not in SUPPORTED_TP_SIZES:
        raise ValueError(f"Gemma4FullModel supports TP sizes {SUPPORTED_TP_SIZES}, got {tp_size}")
    return tp_size


def _find_key(state_dict: dict[str, torch.Tensor], *keys: str) -> str:
    for key in keys:
        if key in state_dict:
            return key
    raise KeyError(f"none of the required checkpoint tensors exists: {keys}")


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class PagedCacheSpec:
    layer_idx: int
    layer_type: str
    block_size: int
    local_kv_heads: int
    head_dim: int
    capacity_tokens_per_slot: int
    cache_dtype: Any = ttnn.bfloat16

    @property
    def blocks_per_slot(self) -> int:
        return _round_up(self.capacity_tokens_per_slot, self.block_size) // self.block_size


@dataclass
class FullModelState:
    """Explicit cache/page-table/slot state shared by prefill and decode."""

    kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]]
    page_tables: list[ttnn.Tensor]
    cache_specs: list[PagedCacheSpec]
    max_batch_size: int
    slot_context_lengths: list[int]
    prompt_lens: list[int]
    positions: torch.Tensor
    active_mask: torch.Tensor


class Gemma4FullModel:
    """All-layer TTNN model preserving the optimized multichip decoder policy."""

    tp_size = TP_SIZE
    supports_on_device_sampling = True
    residual_layout = "replicated BF16 tile-layout DRAM [1,1,M,2816]"

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        hf_config: Any,
        state_dict: dict[str, torch.Tensor],
        max_seq_len: int | None = None,
        max_batch_size: int = 1,
        num_layers: int | None = None,
        layer_indices: Sequence[int] | None = None,
        tensor_cache_path: str | Path | None = None,
        create_kv_cache: bool = True,
        precision_config_path: str | Path | None = None,
        embedding_storage: str = "profile",
    ) -> None:
        tp_size = _require_supported_mesh(mesh_device)
        generation_eos = getattr(hf_config, "eos_token_id", None)
        self.eos_token_ids = tuple(
            int(token_id)
            for token_id in (generation_eos if isinstance(generation_eos, (list, tuple)) else [generation_eos])
            if token_id is not None
        )
        text_config = _text_config(hf_config)
        _validate_text_config(text_config)
        profile_context_limit = PROFILE_CONTEXT_LIMITS[tp_size]
        max_seq_len = profile_context_limit if max_seq_len is None else int(max_seq_len)
        if max_seq_len < 1 or max_seq_len > text_config.max_position_embeddings:
            raise ValueError(f"max_seq_len must be in [1, {text_config.max_position_embeddings}], got {max_seq_len}")
        if max_seq_len > profile_context_limit:
            raise ValueError(
                f"TP{tp_size} full-stack capacity supports at most {profile_context_limit} tokens, got {max_seq_len}"
            )
        if max_batch_size < 1 or max_batch_size > 32:
            raise ValueError(f"max_batch_size must be in [1, 32], got {max_batch_size}")

        self.mesh_device = mesh_device
        self.tp_size = tp_size
        self.profile_context_limit = profile_context_limit
        self.hf_config = text_config
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        if layer_indices is not None and num_layers is not None:
            raise ValueError("pass either num_layers or layer_indices, not both")
        self.layer_indices = (
            list(layer_indices)
            if layer_indices is not None
            else list(range(num_layers or text_config.num_hidden_layers))
        )
        self.num_layers = len(self.layer_indices)
        self.vocab_size = text_config.vocab_size
        self.embed_scale = text_config.hidden_size**0.5
        self.final_logit_softcapping = text_config.final_logit_softcapping
        self.tensor_cache_path = Path(tensor_cache_path) if tensor_cache_path is not None else None
        self.precision_policy, self.precision_config_path = load_precision_policy(precision_config_path)
        self.precision_policy = profile_policy(self.precision_policy, tp_size)
        self.activation_dtype = dtype_from_policy(
            self.precision_policy, "activation_residual", "activation_dtype", ttnn.bfloat16
        )
        self.residual_dtype = dtype_from_policy(
            self.precision_policy, "activation_residual", "residual_dtype", ttnn.bfloat16
        )
        self.kv_cache_dtype = dtype_from_policy(self.precision_policy, "kv_cache", "dtype", ttnn.bfloat16)
        self.logits_dtype = dtype_from_policy(self.precision_policy, "logits_sampling", "logits_dtype", ttnn.bfloat16)
        # The LM head remains in the completed full-model BFP8_B policy on all
        # profiles. A tiled BFP8 embedding forces ``ttnn.embedding`` to
        # untilize the complete vocabulary table on every decode replay. The
        # profile default therefore uses the row-major BF16 representation
        # that the op consumes directly. TP1 retains its 50,624-token limit;
        # the maximum-capacity allocation and non-aligned boundary probes are
        # part of the optimized-full-model evidence.
        self.terminal_weight_dtype = weight_dtype_from_policy(
            self.precision_policy,
            "lm_head",
            weight_dtype_from_policy(self.precision_policy, "embedding_lm_head", ttnn.bfloat8_b),
        )
        policy_embedding = self.precision_policy.get("embedding", {})
        if embedding_storage == "profile" and policy_embedding.get("storage") is not None:
            embedding_storage = policy_embedding["storage"]
        if embedding_storage not in ("profile", "bfp8_tile", "bf16_row_major"):
            raise ValueError("embedding_storage must be 'profile', 'bfp8_tile', or 'bf16_row_major'")
        self.embedding_storage = (
            PROFILE_EMBEDDING_STORAGE[tp_size] if embedding_storage == "profile" else embedding_storage
        )
        default_embedding_dtype = (
            ttnn.bfloat16 if self.embedding_storage == "bf16_row_major" else self.terminal_weight_dtype
        )
        self.embedding_weight_dtype = dtype_from_policy(
            self.precision_policy, "embedding", "weight_dtype", default_embedding_dtype
        )
        if self.embedding_storage == "bf16_row_major" and self.embedding_weight_dtype != ttnn.bfloat16:
            raise ValueError("Gemma-4 row-major embedding storage requires BF16 weights")
        self.embedding_layout = (
            ttnn.ROW_MAJOR_LAYOUT if self.embedding_storage == "bf16_row_major" else ttnn.TILE_LAYOUT
        )
        self.ccl_dtype = dtype_from_policy(self.precision_policy, "ccl", "dtype", ttnn.bfloat16)
        if self.activation_dtype != self.residual_dtype:
            raise ValueError("Gemma-4 currently requires activation_dtype == residual_dtype at layer boundaries")
        sampling_policy = self.precision_policy.get("logits_sampling", {})
        supported_sampling = {
            "sampling_parameter_dtype": "BF16",
            "token_dtype": "UINT32",
            "greedy": "chunked_local_topk32_then_semantic_k1_on_sharded_logits",
        }
        for field, expected in supported_sampling.items():
            if sampling_policy.get(field, expected) != expected:
                raise ValueError(f"unsupported Gemma-4 {field}: {sampling_policy[field]!r}")
        update_dtype = self.precision_policy.get("kv_cache", {}).get("decode_update_dtype", "BF16")
        if update_dtype != "BF16":
            raise ValueError("Gemma-4 paged cache decode updates currently require BF16")
        self.mesh_config = MeshConfig((1, tp_size), decode=ModeConfig(tp=tp_size))
        self._replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        embed_key = _find_key(
            state_dict,
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        )
        norm_key = _find_key(
            state_dict,
            "model.language_model.norm.weight",
            "language_model.norm.weight",
            "model.norm.weight",
        )
        embed_weight = state_dict[embed_key]
        if tuple(embed_weight.shape) != (self.vocab_size, HIDDEN_SIZE):
            raise ValueError(f"unexpected embedding shape {tuple(embed_weight.shape)}")

        cache_root = self.tensor_cache_path / "full_model" if self.tensor_cache_path is not None else None
        common = dict(device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        terminal_shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=tuple(mesh_device.shape))
        # Mesh mappers are not encoded in a tensorbin header.  Keep terminal
        # caches topology-specific so a replicated TP1 artifact can never be
        # reloaded as a hidden/vocab-sharded TP2 or TP4 tensor.
        embed_cache = (
            str(cache_root / f"embedding_tp{tp_size}_{self.embedding_storage}") if cache_root is not None else None
        )
        lm_cache = str(cache_root / f"lm_head_tp{tp_size}") if cache_root is not None else None
        norm_cache = str(cache_root / "final_norm") if cache_root is not None else None
        self.embedding_weight = ttnn.as_tensor(
            embed_weight.unsqueeze(0).unsqueeze(0),
            dtype=self.embedding_weight_dtype,
            layout=self.embedding_layout,
            mesh_mapper=terminal_shard,
            cache_file_name=embed_cache,
            **common,
        )
        self.lm_head_weight = ttnn.as_tensor(
            embed_weight.transpose(0, 1).contiguous().unsqueeze(0).unsqueeze(0),
            dtype=self.terminal_weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=terminal_shard,
            cache_file_name=lm_cache,
            **common,
        )
        self.final_norm_weight = ttnn.as_tensor(
            state_dict[norm_key].reshape(1, 1, 1, HIDDEN_SIZE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._replicate,
            cache_file_name=norm_cache,
            **common,
        )

        self.layers = []
        persistent_requested = tp_size > 1 and os.getenv("GEMMA4_MULTICHIP_PERSISTENT_ALL_REDUCE", "1") == "1"
        previous_persistent_setting = os.environ.get("GEMMA4_MULTICHIP_PERSISTENT_ALL_REDUCE")
        if persistent_requested:
            # Weight setup uses transient L1 for DRAM-to-sharded conversion.
            # Loading the complete stack after layer 0 has already pinned its
            # persistent collective buffers can collide with that conversion
            # region on TP4.  Defer only the allocation; the runtime resource
            # shape and ownership are restored below for every layer.
            os.environ["GEMMA4_MULTICHIP_PERSISTENT_ALL_REDUCE"] = "0"
        try:
            for layer_idx in self.layer_indices:
                self.layers.append(
                    MultichipDecoder.from_state_dict(
                        state_dict,
                        hf_config=text_config,
                        layer_idx=layer_idx,
                        mesh_device=mesh_device,
                        tensor_cache_path=self.tensor_cache_path,
                        ccl_dtype=self.ccl_dtype,
                        **decoder_kwargs(self.precision_policy, layer_idx),
                    )
                )
        finally:
            if persistent_requested:
                if previous_persistent_setting is None:
                    os.environ.pop("GEMMA4_MULTICHIP_PERSISTENT_ALL_REDUCE", None)
                else:
                    os.environ["GEMMA4_MULTICHIP_PERSISTENT_ALL_REDUCE"] = previous_persistent_setting
        if persistent_requested:
            persistent_memory_config = ttnn.create_sharded_memory_config(
                (32, HIDDEN_SIZE // (11 * 8)),
                ttnn.CoreGrid(x=11, y=8),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            persistent_buffer_memory_config = ttnn.create_sharded_memory_config(
                (32, HIDDEN_SIZE * tp_size // (11 * 8)),
                ttnn.CoreGrid(x=11, y=8),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            ccl_grid = mesh_device.compute_with_storage_grid_size()
            ccl_cores = ttnn.num_cores_to_corerangeset(ccl_grid.x * ccl_grid.y, ccl_grid, row_wise=True)
            buffers = [
                ttnn.from_torch(
                    torch.zeros((1, 1, 32, HIDDEN_SIZE * tp_size), dtype=torch.bfloat16),
                    dtype=self.ccl_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=persistent_buffer_memory_config,
                    mesh_mapper=self._replicate,
                )
                for _ in range(3)
            ]
            semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_cores, 0) for _ in range(3)]
            ttnn.synchronize_device(mesh_device)
            resources = {
                "buffers": buffers,
                "semaphores": semaphores,
                "memory_config": persistent_memory_config,
                "index": 0,
                "dtype": self.ccl_dtype,
            }
            for layer in self.layers:
                layer.persistent_all_reduce_buffers = buffers
                layer.persistent_all_reduce_semaphores = semaphores
                layer.persistent_all_reduce_memory_config = persistent_memory_config
                layer.persistent_all_reduce_resources = resources
        self.rope_caches = self._create_rope_caches(text_config, max_seq_len)
        self.cache_specs = self._make_cache_specs()
        self.state = self.allocate_state(max_batch_size=max_batch_size) if create_kv_cache else None

    def _create_rope_caches(self, config: Any, max_seq_len: int) -> dict[str, tuple[ttnn.Tensor, ttnn.Tensor]]:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        rope = Gemma4TextRotaryEmbedding(config)
        positions = torch.arange(max_seq_len).unsqueeze(0)
        # RotaryEmbedding only uses shape/device metadata from x.
        dummy = torch.empty(1, 1, config.hidden_size)
        caches = {}
        for layer_type in sorted({config.layer_types[i] for i in self.layer_indices}):
            cos, sin = rope(dummy, positions, layer_type=layer_type)
            caches[layer_type] = tuple(
                ttnn.as_tensor(
                    value.squeeze(0),
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self._replicate,
                )
                for value in (cos, sin)
            )
        return caches

    def _make_cache_specs(self) -> list[PagedCacheSpec]:
        specs = []
        for state_idx, layer_idx in enumerate(self.layer_indices):
            layer_type = self.hf_config.layer_types[layer_idx]
            kind = _layer_kind(layer_type)
            specs.append(
                PagedCacheSpec(
                    layer_idx=layer_idx,
                    layer_type=layer_type,
                    block_size=kind.block_size,
                    local_kv_heads=max(1, 2 // self.tp_size) if kind is FULL_KIND else 8 // self.tp_size,
                    head_dim=kind.head_dim,
                    capacity_tokens_per_slot=self.max_seq_len if kind is FULL_KIND else SLIDING_CACHE_TOKENS,
                    cache_dtype=getattr(self, "kv_cache_dtype", ttnn.bfloat16),
                )
            )
        return specs

    def allocate_state(
        self,
        *,
        max_batch_size: int | None = None,
        slot_context_lengths: Sequence[int] | None = None,
    ) -> FullModelState:
        batch = max_batch_size or self.max_batch_size
        if batch < 1 or batch > self.max_batch_size:
            raise ValueError(f"state batch must be in [1, {self.max_batch_size}], got {batch}")
        if slot_context_lengths is None:
            base, remainder = divmod(self.max_seq_len, batch)
            slot_context_lengths = [base + (row < remainder) for row in range(batch)]
        else:
            slot_context_lengths = [int(value) for value in slot_context_lengths]
            if len(slot_context_lengths) != batch:
                raise ValueError("slot_context_lengths must contain one value per active row")
            if any(value < 1 for value in slot_context_lengths):
                raise ValueError("slot context capacities must be positive")
            if sum(slot_context_lengths) > self.max_seq_len:
                raise ValueError("aggregate slot capacity exceeds the full-context KV budget")
        kv_cache = []
        page_tables = []
        for spec in self.cache_specs:
            cache_lengths = (
                [spec.capacity_tokens_per_slot] * batch
                if spec.layer_type == "sliding_attention"
                else [int(value) for value in slot_context_lengths]
            )
            row_blocks = [_round_up(value, spec.block_size) // spec.block_size for value in cache_lengths]
            blocks = sum(row_blocks)
            shape = (blocks, spec.local_kv_heads, spec.block_size, spec.head_dim)
            cache_pair = tuple(
                ttnn.zeros(
                    shape,
                    dtype=spec.cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            )
            table = torch.zeros((DECODE_SLOT_COUNT, spec.blocks_per_slot), dtype=torch.int32)
            next_block = 0
            for row, row_block_count in enumerate(row_blocks):
                table[row, :row_block_count] = torch.arange(next_block, next_block + row_block_count, dtype=torch.int32)
                next_block += row_block_count
            page_table = ttnn.from_torch(
                table,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._replicate,
            )
            kv_cache.append(cache_pair)
            page_tables.append(page_table)
        return FullModelState(
            kv_cache=kv_cache,
            page_tables=page_tables,
            cache_specs=self.cache_specs,
            max_batch_size=batch,
            slot_context_lengths=list(slot_context_lengths) + [0] * (DECODE_SLOT_COUNT - batch),
            prompt_lens=[0] * DECODE_SLOT_COUNT,
            positions=torch.full((DECODE_SLOT_COUNT,), -1, dtype=torch.int32),
            active_mask=torch.zeros(DECODE_SLOT_COUNT, dtype=torch.bool),
        )

    def embed_tokens(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        if len(tokens.shape) == 4:
            tokens = ttnn.reshape(tokens, (tokens.shape[-2], tokens.shape[-1]))
        hidden = ttnn.embedding(tokens, self.embedding_weight, dtype=self.residual_dtype, layout=ttnn.TILE_LAYOUT)
        hidden = ttnn.mul(hidden, self.embed_scale)
        hidden = ttnn.unsqueeze_to_4D(hidden) if len(hidden.shape) == 3 else hidden
        if self.tp_size > 1:
            # The synchronous gather derives Linear/Ring routing and link
            # count from the profile's fabric configuration.  Do not create a
            # second CCLManager here: its global semaphores duplicate the
            # optimized decoder's persistent all-reduce resources and exhaust
            # TP4 L1 before the full layer stack is loaded.
            hidden = ttnn.all_gather(
                hidden,
                dim=3,
                cluster_axis=self.mesh_config.tp_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return hidden if hidden.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(hidden, ttnn.TILE_LAYOUT)

    def precision_summary(self) -> dict[str, Any]:
        layers = []
        for layer_idx, layer in zip(self.layer_indices, self.layers):
            packed_expert_gate_up = layer.decode_packed_expert_gate_up
            if packed_expert_gate_up is None:
                packed_expert_gate_up = layer.prefill_packed_expert_gate_up
            expert_gate_up = packed_expert_gate_up if packed_expert_gate_up is not None else layer.weights.expert_gate
            layers.append(
                {
                    "layer": layer_idx,
                    "attention_weight": dtype_name(layer.weights.qkv.dtype),
                    "dense_gate_up_weight": dtype_name(layer.weights.mlp_gate.dtype),
                    "dense_down_weight": dtype_name(layer.weights.mlp_down.dtype),
                    "expert_gate_up_weight": dtype_name(expert_gate_up.dtype),
                    "expert_down_weight": dtype_name(layer.weights.expert_down.dtype),
                    "router_weight": dtype_name(layer.weights.router_proj.dtype),
                    "activation": dtype_name(layer.activation_dtype),
                    "prefill_attention_fidelity": fidelity_name(layer.prefill_attention_compute_config.math_fidelity),
                    "decode_attention_fidelity": fidelity_name(layer.decode_attention_compute_config.math_fidelity),
                    "dense_mlp_fidelity": fidelity_name(layer.mlp_compute_config.math_fidelity),
                    "expert_gate_fidelity": fidelity_name(layer.expert_gate_compute_config.math_fidelity),
                    "expert_down_fidelity": fidelity_name(layer.expert_compute_config.math_fidelity),
                    "decode_weight_dtypes": {
                        role: dtype_name(weight.dtype) for role, weight in layer.decode_dram_weights.items()
                    },
                    "decode_weight_sources": dict(layer.decode_weight_sources),
                }
            )
        return {
            "config_id": self.precision_policy.get("config_id", "compiled_default"),
            "profile": self.precision_policy["resolved_profile"],
            "precision_config_path": str(self.precision_config_path) if self.precision_config_path else None,
            "activation_dtype": dtype_name(self.activation_dtype),
            "residual_dtype": dtype_name(self.residual_dtype),
            "kv_cache_dtype": dtype_name(self.kv_cache_dtype),
            "ccl_dtype": dtype_name(self.ccl_dtype),
            "logits_dtype": dtype_name(self.logits_dtype),
            "embedding_storage": self.embedding_storage,
            "embedding_weight_dtype": dtype_name(self.embedding_weight.dtype),
            "lm_head_weight_dtype": dtype_name(self.lm_head_weight.dtype),
            "kv_decode_update_dtype": self.precision_policy.get("kv_cache", {}).get("decode_update_dtype", "BF16"),
            "sampling": self.precision_policy.get("logits_sampling", {}),
            "layers": layers,
        }

    def _rope_rows(
        self, layer_type: str, positions: ttnn.Tensor, *, decode: bool = False
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        cos_cache, sin_cache = self.rope_caches[layer_type]
        rows = []
        for table in (cos_cache, sin_cache):
            value = ttnn.embedding(positions, table, layout=ttnn.TILE_LAYOUT)
            if decode:
                batch = positions.shape[0]
                shape = (
                    (1, batch, 1, value.shape[-1])
                    if _layer_kind(layer_type) is SLIDING_KIND
                    else (
                        1,
                        1,
                        batch,
                        value.shape[-1],
                    )
                )
                value = ttnn.reshape(value, shape)
            elif len(value.shape) == 3:
                value = ttnn.unsqueeze_to_4D(value)
            rows.append(value)
        return tuple(rows)

    def _terminal(self, hidden: ttnn.Tensor, *, sampler_ready: bool = False) -> ttnn.Tensor:
        hidden = ttnn.rms_norm(
            hidden,
            weight=self.final_norm_weight,
            epsilon=self.hf_config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits = ttnn.linear(
            hidden, self.lm_head_weight, dtype=self.logits_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if self.final_logit_softcapping:
            logits = ttnn.mul(logits, 1.0 / self.final_logit_softcapping)
            logits = ttnn.tanh(logits)
            logits = ttnn.mul(logits, self.final_logit_softcapping)
        if sampler_ready and logits.shape[-2] != DECODE_SLOT_COUNT:
            # Sampling1D consumes a fixed 32-row terminal tile.  Put this pad
            # in the model trace so the sampling trace reads a stable model
            # output instead of allocating a transient tensor while the model
            # trace has allocator addresses pinned.
            logits = ttnn.pad(
                logits,
                padding=[(0, 0), (0, 0), (0, DECODE_SLOT_COUNT - logits.shape[-2]), (0, 0)],
                value=0.0,
            )
        return logits

    def prefill_forward(
        self,
        tokens: ttnn.Tensor,
        *,
        state: FullModelState,
        prompt_lens: Sequence[int],
        position_ids: ttnn.Tensor,
        user_id: int = 0,
        chunk_page_tables: Sequence[ttnn.Tensor | None] | None = None,
        return_all_logits: bool = False,
    ) -> ttnn.Tensor:
        """Device-only prefill over explicit externally owned state.

        ``tokens`` and ``position_ids`` are already padded physical tensors;
        ``prompt_lens`` remains the logical contract and selects final rows.
        """
        if len(prompt_lens) != 1:
            raise ValueError("Gemma4FullModel prefill consumes one logical user at a time")
        logical_seq_len = int(prompt_lens[0])
        if logical_seq_len < 1 or logical_seq_len > tokens.shape[-1]:
            raise ValueError(f"logical prompt length must be in [1, {tokens.shape[-1]}], got {logical_seq_len}")
        hidden = self.embed_tokens(tokens)
        if hidden.shape[-2] != logical_seq_len:
            # The optimized decoder owns physical tile padding because its
            # cache-fill path needs the true logical tail.  Passing the
            # generator's physical prompt shape here would make a 1025-token
            # prompt look aligned at 1056 and overwrite live sliding history.
            hidden = ttnn.slice(
                hidden,
                starts=[0, 0, 0, 0],
                ends=[hidden.shape[0], hidden.shape[1], logical_seq_len, hidden.shape[3]],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            position_ids = ttnn.slice(
                position_ids,
                [0, 0],
                [position_ids.shape[0], logical_seq_len],
                [1, 1],
            )
        for state_idx, (layer_idx, layer) in enumerate(zip(self.layer_indices, self.layers)):
            layer_type = self.hf_config.layer_types[layer_idx]
            cos, sin = self._rope_rows(layer_type, position_ids)
            hidden = layer.prefill_forward(
                hidden,
                position_cos=cos,
                position_sin=sin,
                page_table=state.page_tables[state_idx],
                kv_cache=state.kv_cache[state_idx],
                user_id=user_id,
                chunk_page_table=(None if chunk_page_tables is None else chunk_page_tables[state_idx]),
                cache_position_modulo=(SLIDING_CACHE_TOKENS if _layer_kind(layer_type) is SLIDING_KIND else None),
            )
        if not return_all_logits:
            if len(set(prompt_lens)) != 1:
                raise NotImplementedError("mixed-length terminal prefill slicing is owned by the generator")
            last = logical_seq_len - 1
            hidden = ttnn.slice(hidden, (0, 0, last, 0), (hidden.shape[0], 1, last + 1, HIDDEN_SIZE))
        return self._terminal(hidden)

    def decode_forward(
        self,
        tokens: ttnn.Tensor,
        *,
        state: FullModelState,
        current_pos: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        batch_size: int | None = None,
    ) -> ttnn.Tensor:
        """Trace-safe token-to-sharded-logits decode over stable device state."""
        hidden = self.embed_tokens(tokens)
        logical_batch = int(batch_size or hidden.shape[-2])
        if logical_batch < 1 or logical_batch > hidden.shape[-2]:
            raise ValueError(f"decode batch_size must be in [1, {hidden.shape[-2]}]")
        if logical_batch != hidden.shape[-2]:
            hidden = ttnn.slice(hidden, (0, 0, 0, 0), (1, 1, logical_batch, HIDDEN_SIZE))
        logical_current_pos = current_pos
        logical_position_ids = position_ids
        if current_pos.shape[0] != logical_batch:
            logical_current_pos = ttnn.slice(current_pos, [0], [logical_batch])
            logical_position_ids = ttnn.slice(position_ids, [0], [logical_batch])
        for state_idx, (layer_idx, layer) in enumerate(zip(self.layer_indices, self.layers)):
            layer_type = self.hf_config.layer_types[layer_idx]
            cos, sin = self._rope_rows(layer_type, logical_position_ids, decode=True)
            page_table = state.page_tables[state_idx]
            if page_table.shape[0] != logical_batch:
                page_table = ttnn.slice(page_table, [0, 0], [logical_batch, page_table.shape[1]])
            hidden = layer.decode_forward(
                hidden,
                position_cos=cos,
                position_sin=sin,
                current_pos=logical_current_pos,
                page_table=page_table,
                kv_cache=state.kv_cache[state_idx],
                cache_position_modulo=(SLIDING_CACHE_TOKENS if _layer_kind(layer_type) is SLIDING_KIND else None),
            )
        return self._terminal(hidden, sampler_ready=True)

    def sampling_args(self, *, max_batch_size: int) -> Any:
        """Minimal TTTv1 sampler args retained for the sampler comparison harness."""
        return SimpleNamespace(
            mesh_device=self.mesh_device,
            max_batch_size=max_batch_size,
            max_top_k=32,
            vocab_size=262_144,
            padded_vocab_size=262_144,
            num_devices=self.tp_size,
            sub_core_grids=None,
            sampling_core_grid=None,
            users_row_sharded=False,
            is_galaxy=False,
            is_llama_vision=lambda: False,
        )


__all__ = ["DECODE_SLOT_COUNT", "FullModelState", "Gemma4FullModel", "PagedCacheSpec"]

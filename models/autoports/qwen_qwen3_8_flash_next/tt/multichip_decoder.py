# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Two-die tensor-parallel Qwen3.8-Flash-Next decoder layer.

The fixed target is the 1x2 Blackhole P300 mesh present on the bring-up host.
Each rank owns half of the QSA head groups and half of the routed and shared
expert intermediate dimensions.  GDN, hyperconnection, PLE, indexer and
router tensors remain replicated.  Row-parallel QSA and MoE outputs are summed
before the next replicated hyperconnection boundary.  Replicated GDN is a
deliberate correctness result: the target recurrence loses too much numerical
agreement when its 48 value heads are split into two 24-head kernels.

``from_state_dict`` deliberately starts from :class:`OptimizedDecoder`: it
builds the exact optimized local graph twice on one shared mesh allocation,
patches logical rank 1 with its distinct setup-time shard, and releases the
temporary rank-1 allocation.  No torch or host conversion is used by prefill,
decode, or collective replay.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import functional_decoder as _functional_decoder
from models.autoports.qwen_qwen3_8_flash_next.tt.host_weight_cache import (
    EXPERT_PACKED_BYTES_PER_RANK,
    PLEDeviceStaging,
    Qwen38ExpertHostSource,
    Qwen38PLEHostStore,
    QwenDeviceExpertCache,
    SafetensorCheckpoint,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.model_config import LINEAR_ATTENTION
from models.autoports.qwen_qwen3_8_flash_next.tt.model_config import decoder_shapes as _target_decoder_shapes
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder

TP_SIZE = 2
TARGET_MESH = (1, 2)
DRAM_BYTES_PER_DEVICE = 34_225_520_640
RUNTIME_RESERVE_BYTES = 1 << 30
# BFP8 local main K/V and raw index caches plus BF16 compressed index caches
# across the twelve QSA layers at batch 1 and 262,144 tokens.
MAX_CONTEXT_CACHE_BYTES_PER_DEVICE = 2_340_421_632
# The pre-code TP2 estimate was 2,713,935,872 bytes.  The delivered graph keeps
# all 36 GDN layers replicated.  1.125 GiB is a conservative tile-padded BFP8
# allowance for the second half of qkv/b/a, z and output weights.
REPLICATED_GDN_OVERHEAD_BYTES_PER_DEVICE = 1_207_959_552
NON_EXPERT_WEIGHT_BYTES_PER_DEVICE = 3_921_895_424
EXPERT_TILES_PER_DEVICE = 58_982_400
# The available compressed expert kernel distributes the packed local gate/up
# width over eight banks and pads 640 to 768.  Down projection geometry is
# unchanged, so the resident physical count is
# (512 * 80 * 24 + 512 * 10 * 80) * 48.
COMPRESSED_EXPERT_TILES_PER_DEVICE = 66_846_720
BFP2_TILE_BYTES = 320
BFP4_TILE_BYTES = 576
# BF16 row-parallel partials retain the PCC gate while halving QSA collective
# payload versus FP32 partials.
ROW_PARALLEL_ROLES = frozenset()
HOST_EXPERT_SLOTS = 10
HOST_PACKED_EXPERTS = 16
FULL_STACK_EXPERT_CACHE_BYTES_PER_DEVICE = 48 * (HOST_EXPERT_SLOTS + 1) * EXPERT_PACKED_BYTES_PER_RANK
PLE_STAGING_BYTES_PER_DEVICE = (128 + 32) * 2560 * 2
_SHAPE_OVERRIDE_LOCK = threading.RLock()


@dataclasses.dataclass(frozen=True)
class MultichipMemoryPlan:
    """Calculated full-stack residency boundary for the fixed P300 mesh."""

    dram_bytes: int = DRAM_BYTES_PER_DEVICE
    runtime_reserve_bytes: int = RUNTIME_RESERVE_BYTES
    cache_bytes: int = MAX_CONTEXT_CACHE_BYTES_PER_DEVICE
    non_expert_weight_bytes: int = NON_EXPERT_WEIGHT_BYTES_PER_DEVICE
    expert_tiles: int = EXPERT_TILES_PER_DEVICE
    host_expert_slots: int = HOST_EXPERT_SLOTS
    ple_staging_bytes: int = PLE_STAGING_BYTES_PER_DEVICE

    @property
    def max_expert_bytes(self) -> int:
        return self.dram_bytes - self.runtime_reserve_bytes - self.cache_bytes - self.non_expert_weight_bytes

    @property
    def standard_bfp4_expert_bytes(self) -> int:
        return self.expert_tiles * BFP4_TILE_BYTES

    @property
    def uniform_bfp2_expert_bytes(self) -> int:
        return self.expert_tiles * BFP2_TILE_BYTES

    @property
    def max_bfp4_fraction(self) -> float:
        numerator = self.max_expert_bytes - self.uniform_bfp2_expert_bytes
        denominator = self.expert_tiles * (BFP4_TILE_BYTES - BFP2_TILE_BYTES)
        return numerator / denominator

    @property
    def max_compressed_bfp4_fraction(self) -> float:
        """Practical BFP4 limit after the available kernel's bank padding."""

        uniform_bfp2 = COMPRESSED_EXPERT_TILES_PER_DEVICE * BFP2_TILE_BYTES
        denominator = COMPRESSED_EXPERT_TILES_PER_DEVICE * (BFP4_TILE_BYTES - BFP2_TILE_BYTES)
        return (self.max_expert_bytes - uniform_bfp2) / denominator

    @property
    def standard_bfp4_fits(self) -> bool:
        return self.standard_bfp4_expert_bytes <= self.max_expert_bytes

    @property
    def host_expert_cache_bytes(self) -> int:
        return 48 * (self.host_expert_slots + 1) * EXPERT_PACKED_BYTES_PER_RANK

    @property
    def host_backed_stack_bytes(self) -> int:
        return (
            self.runtime_reserve_bytes
            + self.cache_bytes
            + self.non_expert_weight_bytes
            + self.host_expert_cache_bytes
            + self.ple_staging_bytes
        )

    @property
    def host_backed_stack_fits(self) -> bool:
        return self.host_backed_stack_bytes <= self.dram_bytes


@dataclasses.dataclass(frozen=True)
class HostDecodeAttention:
    """Stable outputs between the attention and router trace segments."""

    hyper: object
    block: object
    injection: object


@dataclasses.dataclass(frozen=True)
class HostDecodeFront:
    """Stable TT outputs crossing the declared compact host-service boundary."""

    work: object
    shared: object
    hyper: object
    injection: object
    route_ids: object
    route_weights: object


def _rank_local_config(hf_config, layer_idx: int | None = None):
    """Clone the HF config and express one of the two equal TP ranks."""

    local = copy.deepcopy(hf_config)
    cfg = local.text_config
    names = ["moe_intermediate_size", "shared_expert_intermediate_size"]
    if layer_idx is None or cfg.layer_types[layer_idx] != LINEAR_ATTENTION:
        names.extend(("num_attention_heads", "num_key_value_heads"))
    if layer_idx is None:
        names.extend(("linear_num_key_heads", "linear_num_value_heads"))
    for name in names:
        value = int(getattr(cfg, name))
        if value % TP_SIZE:
            raise ValueError(f"{name}={value} is not divisible by TP={TP_SIZE}")
        setattr(cfg, name, value // TP_SIZE)
    return local


@contextmanager
def _rank_local_shape_contract(global_config, layer_idx: int):
    """Let the exact-target loader materialize one rank-local TP graph.

    ``FunctionalDecoder`` intentionally validates only checkpoint-global
    shapes.  Multichip setup first performs that validation, then narrows the
    resulting immutable shape record while holding a process-wide lock.  The
    original resolver is restored before setup returns; no runtime method and
    no concurrent model construction can observe the local resolver.
    """

    global_shapes = _target_decoder_shapes(global_config, layer_idx)
    replacements = {
        "moe_intermediate_size": global_shapes.moe_intermediate_size // TP_SIZE,
        "shared_expert_intermediate_size": global_shapes.shared_expert_intermediate_size // TP_SIZE,
    }
    if global_shapes.layer_type != LINEAR_ATTENTION:
        replacements.update(
            num_attention_heads=global_shapes.num_attention_heads // TP_SIZE,
            num_key_value_heads=global_shapes.num_key_value_heads // TP_SIZE,
        )
    local_shapes = dataclasses.replace(global_shapes, **replacements)

    with _SHAPE_OVERRIDE_LOCK:
        original = _functional_decoder.decoder_shapes

        def resolve_local_shapes(_hf_config, requested_layer_idx: int):
            if requested_layer_idx != layer_idx:
                raise ValueError(f"rank-local shape resolver is scoped to layer {layer_idx}, got {requested_layer_idx}")
            return local_shapes

        _functional_decoder.decoder_shapes = resolve_local_shapes
        try:
            yield local_shapes
        finally:
            _functional_decoder.decoder_shapes = original


@contextmanager
def _bounded_host_expert_setup(local_shapes, enabled: bool):
    """Replace missing full-expert zeros with one-expert setup sentinels.

    The optimized decoder is still the graph-construction baseline, but absent
    host-backed expert keys must never cause a transient 512-expert allocation.
    These sentinels satisfy the fused setup graph until fixed cache slots are
    attached; no runtime operation observes them.
    """

    if not enabled:
        yield ()
        return
    expert_shapes = {
        (1, local_shapes.num_experts, local_shapes.hidden_size, local_shapes.moe_intermediate_size),
        (1, local_shapes.num_experts, local_shapes.moe_intermediate_size, local_shapes.hidden_size),
    }
    observed = []
    original_zeros = ttnn.zeros

    def bounded_zeros(shape, *args, **kwargs):
        logical = tuple(int(value) for value in shape)
        if logical in expert_shapes:
            observed.append(logical)
            shape = (logical[0], 1, logical[2], logical[3])
        return original_zeros(shape, *args, **kwargs)

    ttnn.zeros = bounded_zeros
    try:
        yield observed
    finally:
        ttnn.zeros = original_zeros


def _rank_local_state(state_dict: Mapping | None, rank: int, *, shard_gdn: bool = True):
    """Return setup-only checkpoint views/concats for one TP rank."""

    if state_dict is None:
        return None
    if rank not in (0, 1):
        raise ValueError(f"rank must be 0 or 1, got {rank}")

    state = dict(state_dict)

    # Shared expert: column-parallel gate/up and row-parallel down.
    for name in ("gate_proj", "up_proj"):
        key = f"mlp.shared_expert.{name}.weight"
        if key in state:
            state[key] = state[key][rank * 320 : (rank + 1) * 320]
    key = "mlp.shared_expert.down_proj.weight"
    if key in state:
        state[key] = state[key][:, rank * 320 : (rank + 1) * 320]

    # Routed expert: preserve packed [gate, up] ordering within each rank.
    key = "mlp.experts.gate_up_proj"
    if key in state:
        fused = state[key]
        state[key] = torch.cat(
            (
                fused[:, rank * 320 : (rank + 1) * 320],
                fused[:, 640 + rank * 320 : 640 + (rank + 1) * 320],
            ),
            dim=1,
        )
    key = "mlp.experts.down_proj"
    if key in state:
        state[key] = state[key][:, :, rank * 320 : (rank + 1) * 320]

    # The target fused recurrence changes numerical geometry at 24 local value
    # heads and misses decode PCC.  Keep GDN replicated; QSA and MoE remain TP.
    if shard_gdn:
        key = "linear_attn.in_proj_qkv.weight"
        if key in state:
            qkv = state[key]
            state[key] = torch.cat(
                (
                    qkv[rank * 1024 : (rank + 1) * 1024],
                    qkv[2048 + rank * 1024 : 2048 + (rank + 1) * 1024],
                    qkv[4096 + rank * 3072 : 4096 + (rank + 1) * 3072],
                ),
                dim=0,
            )
        key = "linear_attn.in_proj_z.weight"
        if key in state:
            state[key] = state[key][rank * 3072 : (rank + 1) * 3072]
        for suffix in ("in_proj_b.weight", "in_proj_a.weight", "dt_bias", "A_log"):
            key = f"linear_attn.{suffix}"
            if key in state:
                state[key] = state[key][rank * 24 : (rank + 1) * 24]
        key = "linear_attn.conv1d.weight"
        if key in state:
            conv = state[key]
            state[key] = torch.cat(
                (
                    conv[rank * 1024 : (rank + 1) * 1024],
                    conv[2048 + rank * 1024 : 2048 + (rank + 1) * 1024],
                    conv[4096 + rank * 3072 : 4096 + (rank + 1) * 3072],
                ),
                dim=0,
            )
        key = "linear_attn.out_proj.weight"
        if key in state:
            state[key] = state[key][:, rank * 3072 : (rank + 1) * 3072]

    # QSA: retain the checkpoint's per-head [query, gate] packing.
    key = "self_attn.q_proj.weight"
    if key in state:
        q_gate = state[key].reshape(24, 2, 256, 2560)
        state[key] = q_gate[rank * 12 : (rank + 1) * 12].reshape(6144, 2560)
    for suffix in ("k_proj.weight", "v_proj.weight"):
        key = f"self_attn.{suffix}"
        if key in state:
            state[key] = state[key][rank * 256 : (rank + 1) * 256]
    key = "self_attn.o_proj.weight"
    if key in state:
        state[key] = state[key][:, rank * 3072 : (rank + 1) * 3072]

    return state


def _patch_rank_one(target, source, seen: set[tuple[int, int]]) -> None:
    """Copy source logical rank 1 into target's shared 1x2 mesh buffer."""

    if isinstance(target, ttnn.Tensor) and isinstance(source, ttnn.Tensor):
        pair = (id(target), id(source))
        if pair in seen:
            return
        seen.add(pair)
        if target.is_allocated() != source.is_allocated():
            raise ValueError("rank-local optimized graphs disagree on live tensor ownership")
        if not target.is_allocated():
            return
        target_shards = ttnn.get_device_tensors(target)
        source_shards = ttnn.get_device_tensors(source)
        if len(target_shards) != TP_SIZE or len(source_shards) != TP_SIZE:
            raise ValueError("rank patch requires tensors distributed over exactly two devices")
        ttnn.copy(source_shards[1], target_shards[1])
        return
    if dataclasses.is_dataclass(target) and dataclasses.is_dataclass(source):
        for field in dataclasses.fields(target):
            _patch_rank_one(getattr(target, field.name), getattr(source, field.name), seen)
        return
    if isinstance(target, dict) and isinstance(source, dict):
        for key in target.keys() & source.keys():
            _patch_rank_one(target[key], source[key], seen)
        return
    if isinstance(target, (tuple, list)) and isinstance(source, (tuple, list)):
        if len(target) != len(source):
            raise ValueError("rank-local tensor containers disagree in length")
        for target_value, source_value in zip(target, source):
            _patch_rank_one(target_value, source_value, seen)


def _deallocate_tree(value, seen: set[int]) -> None:
    if isinstance(value, ttnn.Tensor):
        if id(value) not in seen:
            seen.add(id(value))
            if value.is_allocated():
                ttnn.deallocate(value)
        return
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            _deallocate_tree(getattr(value, field.name), seen)
        return
    if isinstance(value, dict):
        for item in value.values():
            _deallocate_tree(item, seen)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _deallocate_tree(item, seen)


class MultichipDecoder(OptimizedDecoder):
    """Optimized Qwen decoder layer tensor-parallelized over the fixed P300."""

    TP_SIZE = TP_SIZE
    TARGET_MESH = TARGET_MESH
    OPTIMIZATION_MANIFEST = OptimizedDecoder.OPTIMIZATION_MANIFEST + (
        "p300_1x2_tensor_parallel_heads_and_experts",
        "rank_local_paged_kv_cache",
        "replicated_indexer_selection",
        "attention_and_moe_output_all_reduce",
        "exact_checkpoint_host_expert_cache",
        "exact_mmap_ple_row_lookup",
        "fixed_generation_checked_expert_slots",
    )

    @classmethod
    def from_checkpoint_host_backed(
        cls,
        snapshot,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        expert_cache_slots: int = HOST_EXPERT_SLOTS,
        packed_host_experts: int = HOST_PACKED_EXPERTS,
        ple_store: Qwen38PLEHostStore | None = None,
        **kwargs,
    ) -> "MultichipDecoder":
        """Build one layer without ever constructing resident routed experts."""

        checkpoint = snapshot if isinstance(snapshot, SafetensorCheckpoint) else SafetensorCheckpoint(snapshot)
        state = checkpoint.layer_state(layer_idx, include_experts=False)
        expert_source = Qwen38ExpertHostSource(checkpoint, layer_idx)
        if layer_idx == 1 and ple_store is None:
            ple_store = Qwen38PLEHostStore(checkpoint)
        return cls.from_state_dict(
            state,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            host_expert_source=expert_source,
            expert_cache_slots=expert_cache_slots,
            packed_host_experts=packed_host_experts,
            ple_store=ple_store,
            **kwargs,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        **kwargs,
    ) -> "MultichipDecoder":
        mesh_shape = tuple(int(value) for value in mesh_device.shape)
        if mesh_shape != TARGET_MESH:
            raise ValueError(f"Qwen3.8 multichip target requires mesh {TARGET_MESH}, got {mesh_shape}")

        host_expert_source = kwargs.pop("host_expert_source", None)
        expert_cache_slots = int(kwargs.pop("expert_cache_slots", HOST_EXPERT_SLOTS))
        packed_host_experts = int(kwargs.pop("packed_host_experts", HOST_PACKED_EXPERTS))
        ple_store = kwargs.pop("ple_store", None)
        if host_expert_source is not None:
            if not isinstance(host_expert_source, Qwen38ExpertHostSource):
                raise TypeError("host_expert_source must be Qwen38ExpertHostSource")
            if host_expert_source.layer_idx != layer_idx:
                raise ValueError("host expert source layer does not match decoder layer")
            if state_dict is not None:
                state_dict = {
                    name: value
                    for name, value in state_dict.items()
                    if name not in {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
                }
        if ple_store is not None and layer_idx != 1:
            raise ValueError("PLE host store can only be attached to zero-based layer 1")

        local_config = _rank_local_config(hf_config, layer_idx)
        is_qsa = local_config.text_config.layer_types[layer_idx] != LINEAR_ATTENTION
        # GDN is intentionally replicated.  Only QSA head groups and MoE
        # intermediate dimensions are tensor parallel.
        shard_gdn = False
        local_kwargs = dict(kwargs)
        # Retain the exact single-chip program contracts for replicated GDN.
        # Disable incompatible global-width configs for local QSA projections;
        # sparse MoE gets the legal TP-local geometry below.
        if is_qsa:
            local_kwargs.setdefault("decode_1d_config", "")
            local_kwargs.setdefault("prefill_config", "")
            local_kwargs.setdefault("dram_sharded_role", "")
            # This exact optimized-baseline candidate already clears QSA PCC
            # and trace gates.  On TP2 it saves 1.7578125 GiB/device at maximum
            # context, which is required capacity rather than a cosmetic win.
            local_kwargs.setdefault("cache_policy", "bfp8")
        local_kwargs.setdefault("optimization_policy", "expert_bfp4_lofi_g20b16_d40b5")

        with _rank_local_shape_contract(hf_config, layer_idx) as local_shapes:
            setup_context = _bounded_host_expert_setup(local_shapes, host_expert_source is not None)
            with setup_context as bounded_expert_shapes:
                rank_zero_state = _rank_local_state(state_dict, 0, shard_gdn=shard_gdn)
                primary = OptimizedDecoder.from_state_dict(
                    rank_zero_state,
                    hf_config=local_config,
                    layer_idx=layer_idx,
                    mesh_device=mesh_device,
                    **local_kwargs,
                )
                del rank_zero_state
                gc.collect()

                rank_one_state = _rank_local_state(state_dict, 1, shard_gdn=shard_gdn)
                temporary = OptimizedDecoder.from_state_dict(
                    rank_one_state,
                    hf_config=local_config,
                    layer_idx=layer_idx,
                    mesh_device=mesh_device,
                    **local_kwargs,
                )
                del rank_one_state
                gc.collect()

        skip = {"weight_group_by_id", "weight_role_by_id", "mesh_device"}
        copied: set[tuple[int, int]] = set()
        for name, target in primary.__dict__.items():
            if name not in skip and name in temporary.__dict__:
                _patch_rank_one(target, temporary.__dict__[name], copied)
        ttnn.synchronize_device(mesh_device)

        released: set[int] = set()
        for name, value in temporary.__dict__.items():
            if name != "mesh_device":
                _deallocate_tree(value, released)
        del temporary
        gc.collect()

        primary.__class__ = cls
        primary.mesh_device = mesh_device
        primary.global_hf_config = hf_config
        primary.local_hf_config = local_config
        primary.tp_size = TP_SIZE
        primary.collective_topology = ttnn.Topology.Linear
        primary.collective_axis = 1
        primary.collective_num_links = 1
        primary.memory_plan = MultichipMemoryPlan()
        primary.host_expert_source = host_expert_source
        primary.host_expert_cache = None
        primary.host_ple_store = ple_store
        primary.ple_staging = None
        primary._host_route_ids = None
        primary._host_route_rows = None
        primary._host_logical_route_rows = None
        primary._host_boundary_active = False
        primary._host_segmented_trace_active = False
        primary.host_setup_expert_shapes = tuple(bounded_expert_shapes)
        if host_expert_source is not None:
            # Release one-expert setup sentinels and replace them with bounded,
            # fixed-address demand-loaded slots.
            if primary.expert_gate_up is not None and primary.expert_gate_up.is_allocated():
                ttnn.deallocate(primary.expert_gate_up)
            if primary.experts.down is not None and primary.experts.down.is_allocated():
                ttnn.deallocate(primary.experts.down)
            primary.expert_gate_up = None
            primary.experts = dataclasses.replace(primary.experts, gate=None, up=None, down=None)
            primary.host_expert_cache = QwenDeviceExpertCache(
                mesh_device,
                host_expert_source,
                capacity=expert_cache_slots,
                packed_host_capacity=packed_host_experts,
            )
        if ple_store is not None:
            primary.ple_staging = PLEDeviceStaging(mesh_device, max_batch=primary.max_batch)
        return primary

    def _routing_from_logits(self, logits):
        if self.host_expert_cache is None:
            return super()._routing_from_logits(logits)
        s = self.shapes
        zeros = ttnn.zeros_like(logits)
        values, indices = ttnn.topk(logits, k=s.num_experts_per_tok, dim=-1, sorted=True)
        ttnn.deallocate(logits)
        selected_values = values
        values = ttnn.softmax(selected_values, dim=-1)
        _functional_decoder._free(selected_values, values)
        indexed = self._decode_active and self.max_batch == 1
        if indexed:
            index_row = ttnn.slice(indices, [0, 0, 0, 0], [1, 1, 1, s.num_experts_per_tok])
            route_ids = ttnn.to_layout(index_row, ttnn.ROW_MAJOR_LAYOUT)
            _functional_decoder._free(index_row, indices, route_ids)
            self._decode_expert_weights = ttnn.slice(values, [0, 0, 0, 0], [1, 1, 1, s.num_experts_per_tok])
        else:
            route_ids = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
        self._host_route_ids = route_ids
        routing = ttnn.scatter(zeros, dim=-1, index=indices, src=values)
        ttnn.deallocate(zeros)
        ttnn.deallocate(values)
        ttnn.deallocate(indices)
        return routing

    def _read_compact_route_ids(self) -> tuple[int, ...]:
        """Declared D2H boundary: read top-k ids, never routing weights."""

        if self._host_route_ids is None:
            raise RuntimeError("host expert service has no compact route ids")
        route_shards = ttnn.get_device_tensors(self._host_route_ids)
        if len(route_shards) != TP_SIZE:
            raise RuntimeError("compact route tensor is not replicated over TP2")
        host = ttnn.to_torch(route_shards[0]).reshape(-1, self.shapes.num_experts_per_tok)
        rows = min(int(self._host_route_rows or host.shape[0]), int(host.shape[0]))
        return tuple(dict.fromkeys(int(value) for value in host[:rows].reshape(-1).tolist()))

    def _slot_route_weights(self, routing, expert_id: int):
        tokens = int(routing.shape[-2])
        index_host = torch.full((1, 1, tokens, 1), int(expert_id), dtype=torch.int32)
        index = ttnn.from_torch(
            index_host,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        weights = ttnn.gather(routing, dim=-1, index=index)
        ttnn.deallocate(index)
        return weights

    def _wave_route_weights(self, routing, expert_ids):
        tokens = int(routing.shape[-2])
        ids = tuple(int(expert_id) for expert_id in expert_ids)
        index_host = torch.tensor(ids, dtype=torch.int32).reshape(1, 1, 1, -1).expand(1, 1, tokens, -1)
        index = ttnn.from_torch(
            index_host,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        weights = ttnn.gather(routing, dim=-1, index=index)
        ttnn.deallocate(index)
        return weights

    def _routed_expert_slot(self, x, routing, slot_index: int, expert_id: int):
        """Run one selected expert on TT using a fixed rank-local slot."""

        s = self.shapes
        slot = self.host_expert_cache.slots[slot_index]
        tokens = int(x.shape[-2])
        groups = tokens // 32
        grouped_x = ttnn.reshape(x, (1, groups, 32, s.hidden_size))
        route_weights = self._slot_route_weights(routing, expert_id)
        routing_groups = ttnn.reshape(route_weights, (1, groups, 32, 1))
        sparsity = ttnn.max(routing_groups, dim=2, keepdim=True)
        sparsity = ttnn.reshape(sparsity, (1, 1, groups, 1))
        sparsity = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
        gate_up_sparse = ttnn.sparse_matmul(
            grouped_x,
            slot.gate_up,
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            program_config=self._sparse_matmul_config(32, 2 * s.moe_intermediate_size, s.hidden_size),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.expert_compute_cfg,
        )
        gate_up = ttnn.reshape(gate_up_sparse, (groups, 1, 32, 2 * s.moe_intermediate_size))
        _functional_decoder._free(gate_up_sparse, gate_up)
        gate = self._slice_last(gate_up, 0, s.moe_intermediate_size)
        up = self._slice_last(gate_up, s.moe_intermediate_size, 2 * s.moe_intermediate_size)
        ttnn.deallocate(gate_up)
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        token_weights = ttnn.reshape(routing_groups, (groups, 1, 32, 1))
        weighted = ttnn.multiply(hidden, token_weights)
        ttnn.deallocate(hidden)
        ttnn.deallocate(route_weights)
        _functional_decoder._free(routing_groups, token_weights, weighted)
        down = ttnn.sparse_matmul(
            weighted,
            slot.down,
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            program_config=self._sparse_matmul_config(32, s.hidden_size, s.moe_intermediate_size),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.expert_compute_cfg,
        )
        ttnn.deallocate(weighted)
        ttnn.deallocate(sparsity)
        return ttnn.reshape(down, (1, 1, tokens, s.hidden_size))

    def _routed_experts_indexed_host(self, x):
        """Run the exact ten decode experts as one compact indexed TT bank.

        The resident optimized baseline evaluates its selected experts in one
        indexed sparse kernel.  Ten independent matmuls changed accumulation
        geometry enough to miss the decoder PCC gate, so fixed slots are
        concatenated on device in router order and use the same indexed
        gate/up/down topology.  The bank contains only the selected experts.
        """

        s = self.shapes
        route_ids = self._read_compact_route_ids()
        if len(route_ids) != s.num_experts_per_tok:
            raise RuntimeError(f"decode selected {len(route_ids)} unique experts, expected {s.num_experts_per_tok}")
        plan = self.host_expert_cache.ensure_ordered(route_ids)
        self.host_expert_cache.validate(plan)
        if self._decode_expert_weights is None:
            raise RuntimeError("host indexed decode is missing selected route weights")
        output = self._routed_experts_indexed_ready(x, self._decode_expert_weights)
        for tensor in (self._host_route_ids, self._decode_expert_weights):
            if tensor is not None and tensor.is_allocated():
                ttnn.deallocate(tensor)
        self._host_route_ids = None
        self._decode_expert_weights = None
        return output

    def _routed_experts_indexed_ready(self, x, route_weights):
        """TT-only indexed expert graph for already serviced ordered slots."""

        s = self.shapes
        ordered_slots = self.host_expert_cache.slots[: s.num_experts_per_tok]
        gate_up_bank = ttnn.concat([slot.gate_up for slot in ordered_slots], dim=1)
        down_bank = ttnn.concat([slot.down for slot in ordered_slots], dim=1)
        local_indices = self.host_expert_cache.local_indices
        grouped_x = ttnn.reshape(x, (1, 1, 32, s.hidden_size))
        sparsity = ttnn.to_layout(route_weights, ttnn.ROW_MAJOR_LAYOUT)
        gate_up_sparse = ttnn.sparse_matmul(
            grouped_x,
            gate_up_bank,
            sparsity=sparsity,
            indices=local_indices,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            is_input_b_sparse=True,
            program_config=self._sparse_matmul_config(32, 2 * s.moe_intermediate_size, s.hidden_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _functional_decoder._free(grouped_x, x, gate_up_sparse)
        gate_up = ttnn.reshape(gate_up_sparse, (1, s.num_experts_per_tok, 32, 2 * s.moe_intermediate_size))
        _functional_decoder._free(gate_up_sparse, gate_up)
        gate = self._slice_last(gate_up, 0, s.moe_intermediate_size)
        up = self._slice_last(gate_up, s.moe_intermediate_size, 2 * s.moe_intermediate_size)
        ttnn.deallocate(gate_up)
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        selected_weights = ttnn.permute(route_weights, (0, 3, 2, 1))
        weighted_hidden = ttnn.multiply(hidden, selected_weights)
        ttnn.deallocate(hidden)
        _functional_decoder._free(selected_weights, route_weights, weighted_hidden)
        down = ttnn.sparse_matmul(
            weighted_hidden,
            down_bank,
            sparsity=sparsity,
            indices=local_indices,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=self._sparse_matmul_config(32, s.hidden_size, s.moe_intermediate_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _functional_decoder._free(weighted_hidden, down)
        for tensor in (sparsity, gate_up_bank, down_bank):
            if tensor is not None and tensor.is_allocated():
                ttnn.deallocate(tensor)
        out = ttnn.experimental.fast_reduce_nc(down, dims=[1])
        ttnn.deallocate(down)
        return ttnn.reshape(ttnn.unsqueeze_to_4D(out), (1, 1, 32, s.hidden_size))

    def _routed_expert_wave(self, x, routing, expert_ids, plan):
        """Evaluate one bounded prefill/batched wave as a compact sparse bank."""

        s = self.shapes
        tokens = int(x.shape[-2])
        groups = tokens // 32
        slot_by_expert = {
            expert_id: slot_index
            for slot_index, (expert_id, active) in enumerate(zip(plan.slot_expert_ids, plan.active_slots))
            if active
        }
        ordered_slots = [self.host_expert_cache.slots[slot_by_expert[expert_id]] for expert_id in expert_ids]
        gate_up_bank = ttnn.concat([slot.gate_up for slot in ordered_slots], dim=1)
        down_bank = ttnn.concat([slot.down for slot in ordered_slots], dim=1)
        route_weights = self._wave_route_weights(routing, expert_ids)
        grouped_x = ttnn.reshape(x, (1, groups, 32, s.hidden_size))
        routing_groups = ttnn.reshape(route_weights, (1, groups, 32, len(expert_ids)))
        sparsity = ttnn.max(routing_groups, dim=2, keepdim=True)
        sparsity = ttnn.reshape(sparsity, (1, 1, groups, len(expert_ids)))
        sparsity = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
        gate_up_sparse = ttnn.sparse_matmul(
            grouped_x,
            gate_up_bank,
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            program_config=self._sparse_matmul_config(32, 2 * s.moe_intermediate_size, s.hidden_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _functional_decoder._free(grouped_x, x, gate_up_sparse)
        gate_up = ttnn.reshape(
            gate_up_sparse,
            (groups, len(expert_ids), 32, 2 * s.moe_intermediate_size),
        )
        _functional_decoder._free(gate_up_sparse, gate_up)
        gate = self._slice_last(gate_up, 0, s.moe_intermediate_size)
        up = self._slice_last(gate_up, s.moe_intermediate_size, 2 * s.moe_intermediate_size)
        ttnn.deallocate(gate_up)
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        token_weights = ttnn.permute(routing_groups, (1, 3, 2, 0))
        weighted = ttnn.multiply(hidden, token_weights)
        ttnn.deallocate(hidden)
        _functional_decoder._free(token_weights, routing_groups, weighted)
        _functional_decoder._free(routing_groups, route_weights, weighted)
        _functional_decoder._free(route_weights, weighted)
        down = ttnn.sparse_matmul(
            weighted,
            down_bank,
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            program_config=self._sparse_matmul_config(32, s.hidden_size, s.moe_intermediate_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _functional_decoder._free(weighted, down)
        for tensor in (sparsity, gate_up_bank, down_bank):
            ttnn.deallocate(tensor)
        out = ttnn.experimental.fast_reduce_nc(down, dims=[1])
        ttnn.deallocate(down)
        return ttnn.reshape(ttnn.unsqueeze_to_4D(out), (1, 1, tokens, s.hidden_size))

    def _routed_experts(self, x, routing):
        if self.host_expert_cache is None:
            return super()._routed_experts(x, routing)
        if self._decode_active and self.max_batch == 1:
            ttnn.deallocate(routing)
            return self._routed_experts_indexed_host(x)
        route_ids = self._read_compact_route_ids()
        waves = self.host_expert_cache.waves(route_ids)
        if not waves:
            raise RuntimeError("router returned no active experts")
        accumulator = None
        for wave in waves:
            plan = self.host_expert_cache.ensure_wave(wave)
            self.host_expert_cache.validate(plan)
            value = self._routed_expert_wave(x, routing, wave, plan)
            if accumulator is None:
                accumulator = value
            else:
                updated = ttnn.add(accumulator, value)
                ttnn.deallocate(accumulator)
                ttnn.deallocate(value)
                accumulator = updated
        ttnn.deallocate(self._host_route_ids)
        self._host_route_ids = None
        ttnn.deallocate(routing)
        return accumulator

    def _all_reduce_block(self, partial):
        output = ttnn.all_reduce(
            partial,
            cluster_axis=self.collective_axis,
            num_links=self.collective_num_links,
            topology=self.collective_topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(partial)
        if output.dtype == ttnn.float32:
            reduced = ttnn.typecast(output, ttnn.bfloat16)
            ttnn.deallocate(output)
            output = reduced
        return output

    def _linear(self, x, weight, *, dtype=ttnn.bfloat16):
        role = self.weight_role_by_id.get(id(weight))
        if role in ROW_PARALLEL_ROLES and dtype == ttnn.bfloat16:
            dtype = ttnn.float32
        return self._linear_impl(x, weight, dtype=dtype)

    def _gdn_prefill(self, *args, **kwargs):
        return super()._gdn_prefill(*args, **kwargs)

    def _gdn_decode(self, *args, **kwargs):
        return super()._gdn_decode(*args, **kwargs)

    def _qsa_prefill(self, *args, **kwargs):
        return self._all_reduce_block(super()._qsa_prefill(*args, **kwargs))

    def _qsa_decode(self, *args, **kwargs):
        return self._all_reduce_block(super()._qsa_decode(*args, **kwargs))

    def _moe(self, *args, **kwargs):
        if self.host_expert_cache is None:
            return self._all_reduce_block(super()._moe(*args, **kwargs))
        if not args:
            raise TypeError("MoE input tensor is required")
        self._host_route_rows = int(self._host_logical_route_rows or args[0].shape[-2])
        self._host_boundary_active = True
        try:
            return self._all_reduce_block(super()._moe(*args, **kwargs))
        finally:
            self._host_boundary_active = False
            self._host_route_rows = None

    def _decode_attention_host(
        self, hidden_states, *, current_pos, page_table=None, rot_mats=None, ple_embeddings=None
    ):
        """First TT segment through PLE and GDN/QSA attention."""

        if self.host_expert_cache is None or self.max_batch != 1:
            raise RuntimeError("segmented host trace requires batch-one host expert slots")
        s = self.shapes
        if _functional_decoder._shape(hidden_states) != [1, 1, 1, s.hc_hidden_size]:
            raise ValueError("segmented decode hidden input has the wrong shape")
        if current_pos is None or _functional_decoder._shape(current_pos) != [1]:
            raise ValueError("segmented decode current_pos must be device int32 [1]")
        self._decode_active = True
        try:
            if s.has_ple:
                if ple_embeddings is None or _functional_decoder._shape(ple_embeddings) != [1, 1, 1, s.ple_embed_dim]:
                    raise ValueError("segmented PLE decode requires stable [1, 1, 1, 2560] embeddings")
                ple = self._ple_decode(hidden_states, ple_embeddings)
                front_hidden = ttnn.add(hidden_states, ple)
                ttnn.deallocate(ple)
            else:
                if ple_embeddings is not None:
                    raise ValueError("PLE embeddings passed to a layer without PLE")
                front_hidden = hidden_states

            mixed, hyper, injection = self._hyper_mix(front_hidden, "attn_hc")
            if s.layer_type == LINEAR_ATTENTION:
                block = self._gdn_decode(mixed)
            else:
                if page_table is None or rot_mats is None:
                    raise ValueError("segmented QSA decode requires page_table and RoPE tables")
                block = self._qsa_decode(mixed, current_pos=current_pos, page_table=page_table, rot_mats=rot_mats)
            ttnn.deallocate(mixed)
            return HostDecodeAttention(hyper, block, injection)
        finally:
            self._decode_active = False

    def _hyper_inject_crossing(self, hyper_input, block_output, injection):
        """Inject attention while retaining every captured crossing buffer."""

        s = self.shapes
        value = ttnn.reshape(block_output, (1, 1, s.hidden_size))
        gate = ttnn.reshape(injection, (1, s.hc_count, 1))
        projected = ttnn.multiply(value, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        _functional_decoder._free(value, block_output, projected)
        _functional_decoder._free(gate, injection, projected)
        projected = ttnn.reshape(projected, _functional_decoder._shape(hyper_input))
        out = ttnn.mac(projected, 2.0, hyper_input)
        ttnn.deallocate(projected)
        return out

    def _decode_router_host(self, attention: HostDecodeAttention):
        """Second TT segment through router and shared-expert projection."""

        s = self.shapes
        self._decode_active = True
        try:
            hidden = self._hyper_inject_crossing(attention.hyper, attention.block, attention.injection)
            mixed, hyper, injection = self._hyper_mix(hidden, "mlp_hc")
            work = _functional_decoder._pad_seq(mixed, 32, mixed)
            _functional_decoder._free(mixed, work)

            packed = self._linear(work, self.w["moe_input"])
            cursor = 0
            logits = self._slice_last(packed, cursor, cursor + s.num_experts)
            cursor += s.num_experts
            gate = self._slice_last(packed, cursor, cursor + s.shared_expert_intermediate_size)
            cursor += s.shared_expert_intermediate_size
            up = self._slice_last(packed, cursor, cursor + s.shared_expert_intermediate_size)
            cursor += s.shared_expert_intermediate_size
            scalar = self._slice_last(packed, cursor, cursor + 1)
            ttnn.deallocate(packed)

            values, indices = ttnn.topk(logits, k=s.num_experts_per_tok, dim=-1, sorted=True)
            ttnn.deallocate(logits)
            route_weights = ttnn.softmax(values, dim=-1)
            _functional_decoder._free(values, route_weights)
            index_row = ttnn.slice(indices, [0, 0, 0, 0], [1, 1, 1, s.num_experts_per_tok])
            route_ids = ttnn.to_layout(index_row, ttnn.ROW_MAJOR_LAYOUT)
            _functional_decoder._free(index_row, indices, route_ids)
            ttnn.deallocate(indices)
            all_route_weights = route_weights
            route_weights = ttnn.slice(
                all_route_weights,
                [0, 0, 0, 0],
                [1, 1, 1, s.num_experts_per_tok],
            )
            _functional_decoder._free(all_route_weights, route_weights)

            shared_hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            gated_shared = ttnn.multiply(
                shared_hidden,
                scalar,
                input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
            )
            ttnn.deallocate(shared_hidden)
            ttnn.deallocate(scalar)
            shared = self._linear(gated_shared, self.w["shared_down_proj"])
            ttnn.deallocate(gated_shared)
            return HostDecodeFront(work, shared, hyper, injection, route_ids, route_weights)
        finally:
            self._decode_active = False

    def _decode_front_host(self, hidden_states, *, current_pos, page_table=None, rot_mats=None, ple_embeddings=None):
        """Eager convenience wrapper matching the two captured front segments."""

        attention = self._decode_attention_host(
            hidden_states,
            current_pos=current_pos,
            page_table=page_table,
            rot_mats=rot_mats,
            ple_embeddings=ple_embeddings,
        )
        return self._decode_router_host(attention)

    def service_decode_front(self, front: HostDecodeFront):
        """Declared D2H ids plus exact ordered expert H2D between TT segments."""

        host = ttnn.to_torch(ttnn.get_device_tensors(front.route_ids)[0]).reshape(-1)
        route_ids = tuple(int(value) for value in host[: self.shapes.num_experts_per_tok].tolist())
        if len(route_ids) != len(set(route_ids)):
            raise RuntimeError(f"router returned duplicate top-k expert ids: {route_ids}")
        plan = self.host_expert_cache.ensure_ordered(route_ids)
        self.host_expert_cache.validate(plan)
        return route_ids, plan

    def _hyper_inject_preserve(self, hyper_input, block_output, injection):
        """Trace-back injection that preserves front-segment input buffers."""

        s = self.shapes
        value = ttnn.reshape(block_output, (1, 1, s.hidden_size))
        gate = ttnn.reshape(injection, (1, s.hc_count, 1))
        projected = ttnn.multiply(value, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        _functional_decoder._free(value, block_output, projected)
        _functional_decoder._free(gate, injection, projected)
        ttnn.deallocate(block_output)
        projected = ttnn.reshape(projected, _functional_decoder._shape(hyper_input))
        out = ttnn.mac(projected, 2.0, hyper_input)
        ttnn.deallocate(projected)
        return out

    def _decode_back_host(self, front: HostDecodeFront):
        """TT trace segment from fixed ordered expert slots to layer output."""

        routed = self._routed_experts_indexed_ready(front.work, front.route_weights)
        local = ttnn.add(routed, front.shared)
        ttnn.deallocate(routed)
        reduced = self._all_reduce_block(local)
        trimmed = ttnn.slice(reduced, [0, 0, 0, 0], [1, 1, 1, self.shapes.hidden_size])
        _functional_decoder._free(reduced, trimmed)
        reduced = trimmed
        return self._hyper_inject_preserve(front.hyper, reduced, front.injection)

    def _require_host_ple(self) -> None:
        if self.host_ple_store is None or self.ple_staging is None or not self.shapes.has_ple:
            raise RuntimeError("exact host PLE service is attached only to zero-based layer 1")

    def reset_host_request(self, request_id) -> None:
        self._require_host_ple()
        self.host_ple_store.reset_request(request_id)

    def cancel_host_request(self, request_id) -> None:
        self._require_host_ple()
        self.host_ple_store.cancel_request(request_id)

    def prefill_forward_host_backed(
        self,
        hidden_states,
        *,
        input_ids: torch.Tensor,
        request_id,
        user_id: int = 0,
        valid_mask: torch.Tensor | None = None,
    ):
        """Exact chunked PLE lookup/staging followed by the TP2 layer graph."""

        self._require_host_ple()
        s = self.shapes
        hidden_shape = _functional_decoder._shape(hidden_states)
        if len(hidden_shape) != 4 or hidden_shape[:2] != [1, 1]:
            raise ValueError("host-backed prefill expects [1, 1, seq, hidden]")
        seq_len = int(hidden_states.shape[-2])
        if int(hidden_states.shape[-1]) != s.hc_hidden_size:
            raise ValueError(f"prefill hidden width {int(hidden_states.shape[-1])} != {s.hc_hidden_size}")
        ids = torch.as_tensor(input_ids, dtype=torch.int64, device="cpu")
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if tuple(ids.shape) != (1, seq_len):
            raise ValueError(f"host PLE input ids must be [1, {seq_len}], got {tuple(ids.shape)}")
        mask = None if valid_mask is None else torch.as_tensor(valid_mask, dtype=torch.bool, device="cpu")
        if mask is not None and tuple(mask.shape) != tuple(ids.shape):
            raise ValueError("host PLE valid mask shape does not match input ids")
        if not 0 <= user_id < self.max_batch:
            raise ValueError(f"user_id {user_id} outside [0, {self.max_batch})")

        self._decode_active = False
        self._reset_user_state(user_id)
        pieces = []
        for start, logical, padded in self.prefill_chunk_plan(seq_len):
            x = _functional_decoder._slice_seq(hidden_states, start, logical, padded)
            embeddings = self.host_ple_store.prepare(
                [request_id],
                ids[:, start : start + logical],
                valid_mask=None if mask is None else mask[:, start : start + logical],
                reset=start == 0,
            )
            staged = self.ple_staging.upload_prefill(embeddings, logical=logical)
            ple = self._ple_prefill(x, staged, user_id=user_id, logical=logical)
            updated = ttnn.add(x, ple)
            _functional_decoder._free(x, hidden_states, updated)
            ttnn.deallocate(ple)
            mixed, hyper, injection = self._hyper_mix(updated, "attn_hc")
            block = self._gdn_prefill(mixed, user_id=user_id, logical=logical)
            ttnn.deallocate(mixed)
            hidden = self._hyper_inject(hyper, block, injection)
            mixed, hyper, injection = self._hyper_mix(hidden, "mlp_hc")
            self._host_logical_route_rows = logical
            try:
                block = self._moe(mixed)
            finally:
                self._host_logical_route_rows = None
            ttnn.deallocate(mixed)
            out = self._hyper_inject(hyper, block, injection)
            if logical != padded:
                trimmed = ttnn.slice(out, [0, 0, 0, 0], [1, 1, logical, s.hc_hidden_size])
                _functional_decoder._free(out, trimmed)
                out = trimmed
            pieces.append(out)
        if len(pieces) == 1:
            return pieces[0]
        output = ttnn.concat(pieces, dim=-2)
        for piece in pieces:
            ttnn.deallocate(piece)
        return output

    def decode_forward_host_backed(
        self,
        hidden_states,
        *,
        input_ids: torch.Tensor,
        request_ids,
        current_pos,
    ):
        """Service one real PLE row per request, then execute normal decode."""

        self._require_host_ple()
        request_ids = tuple(request_ids)
        if len(request_ids) != self.max_batch:
            raise ValueError(f"decode needs {self.max_batch} request ids")
        ids = torch.as_tensor(input_ids, dtype=torch.int64, device="cpu")
        if ids.ndim == 1:
            ids = ids.unsqueeze(1)
        if tuple(ids.shape) != (self.max_batch, 1):
            raise ValueError(f"decode input ids must be [{self.max_batch}, 1], got {tuple(ids.shape)}")
        embeddings = self.host_ple_store.prepare(request_ids, ids)
        staged = self.ple_staging.upload_decode(embeddings)
        return self.decode_forward(hidden_states, current_pos=current_pos, ple_embeddings=staged)

    def close_host_backing(self) -> None:
        """Release only the host-backed resources owned by this layer."""

        if self.host_expert_cache is not None:
            self.host_expert_cache.close()
            self.host_expert_cache = None
        if self.ple_staging is not None:
            self.ple_staging.close()
            self.ple_staging = None
        if self.host_ple_store is not None:
            self.host_ple_store.close()
            self.host_ple_store = None


class HostBackedSegmentedDecodeTrace:
    """Warmed QSA front/back TT traces around exact host expert service.

    Progressing optimized GDN state is deliberately rejected: repeated live
    trace replay corrupts its persistent FP32 L1 recurrence on the target
    runtime.  The stage work log records the isolated AutoFix evidence.
    """

    def __init__(self, layer, front, output, front_trace_id, back_trace_id):
        self.layer = layer
        self.front = front
        self.output = output
        self.front_trace_id = front_trace_id
        self.back_trace_id = back_trace_id
        self.last_timing = None
        self.last_route_ids = None
        self.released = False

    @staticmethod
    def _release_front(front: HostDecodeFront) -> None:
        for field in dataclasses.fields(front):
            tensor = getattr(front, field.name)
            if tensor.is_allocated():
                ttnn.deallocate(tensor)

    @staticmethod
    def _finish_failed_capture(mesh_device, trace_id, capture_open: bool) -> None:
        if trace_id is None:
            return
        if capture_open:
            try:
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            except Exception:
                pass
        try:
            ttnn.release_trace(mesh_device, trace_id)
        except Exception:
            pass

    @classmethod
    def capture(
        cls,
        layer: MultichipDecoder,
        hidden_states,
        *,
        current_pos,
        page_table=None,
        rot_mats=None,
        ple_input_ids: torch.Tensor | None = None,
        request_ids=None,
    ) -> "HostBackedSegmentedDecodeTrace":
        if layer.host_expert_cache is None:
            raise RuntimeError("segmented trace requires host-backed expert slots")
        if layer.shapes.layer_type == LINEAR_ATTENTION:
            raise RuntimeError(
                "progressing optimized GDN state is not trace-safe on this runtime; "
                "see doc/multichip_decoder/work_log.md"
            )
        if layer._host_segmented_trace_active:
            raise RuntimeError("segmented trace is already active for this layer")
        if ple_input_ids is not None or request_ids is not None:
            raise ValueError("QSA segmented trace does not accept PLE inputs")

        front_kwargs = {
            "current_pos": current_pos,
            "page_table": page_table,
            "rot_mats": rot_mats,
            "ple_embeddings": None,
        }
        layer._host_segmented_trace_active = True
        front = output = None
        front_trace_id = back_trace_id = None
        front_open = back_open = cache_locked = False
        try:
            warm_front = layer._decode_front_host(hidden_states, **front_kwargs)
            layer.service_decode_front(warm_front)
            warm_output = layer._decode_back_host(warm_front)
            ttnn.synchronize_device(layer.mesh_device)
            if warm_output.is_allocated():
                ttnn.deallocate(warm_output)
            cls._release_front(warm_front)

            layer.mesh_device.set_program_cache_misses_allowed(False)
            cache_locked = True
            front_trace_id = ttnn.begin_trace_capture(layer.mesh_device, cq_id=0)
            front_open = True
            front = layer._decode_front_host(hidden_states, **front_kwargs)
            ttnn.end_trace_capture(layer.mesh_device, front_trace_id, cq_id=0)
            front_open = False
            route_ids, _ = layer.service_decode_front(front)

            back_trace_id = ttnn.begin_trace_capture(layer.mesh_device, cq_id=0)
            back_open = True
            output = layer._decode_back_host(front)
            ttnn.end_trace_capture(layer.mesh_device, back_trace_id, cq_id=0)
            back_open = False
            ttnn.mark_corruptible(output)
            ttnn.execute_trace(layer.mesh_device, back_trace_id, cq_id=0, blocking=True)
        except Exception:
            cls._finish_failed_capture(layer.mesh_device, back_trace_id, back_open)
            cls._finish_failed_capture(layer.mesh_device, front_trace_id, front_open)
            if front is not None:
                cls._release_front(front)
            if output is not None and output.is_allocated():
                ttnn.deallocate(output)
            layer._host_segmented_trace_active = False
            raise
        finally:
            if cache_locked:
                layer.mesh_device.set_program_cache_misses_allowed(True)

        trace = cls(layer, front, output, front_trace_id, back_trace_id)
        trace.last_route_ids = route_ids
        return trace

    def replay(self, *, ple_input_ids: torch.Tensor | None = None, request_ids=None):
        if self.released:
            raise RuntimeError("segmented trace has been released")
        if ple_input_ids is not None or request_ids is not None:
            raise ValueError("QSA segmented trace does not accept PLE inputs")
        started = time.perf_counter()
        front_started = time.perf_counter()
        ttnn.execute_trace(self.layer.mesh_device, self.front_trace_id, cq_id=0, blocking=True)
        front_seconds = time.perf_counter() - front_started
        service_started = time.perf_counter()
        route_ids, plan = self.layer.service_decode_front(self.front)
        service_seconds = time.perf_counter() - service_started
        back_started = time.perf_counter()
        ttnn.execute_trace(self.layer.mesh_device, self.back_trace_id, cq_id=0, blocking=True)
        back_seconds = time.perf_counter() - back_started
        self.last_route_ids = route_ids
        self.last_timing = {
            "front_trace_seconds": front_seconds,
            "expert_service_seconds": service_seconds,
            "back_trace_seconds": back_seconds,
            "total_seconds": time.perf_counter() - started,
            "expert_hits": len(plan.hits),
            "expert_misses": len(plan.misses),
        }
        return self.output

    def release(self) -> None:
        if self.released:
            return
        ttnn.release_trace(self.layer.mesh_device, self.front_trace_id)
        ttnn.release_trace(self.layer.mesh_device, self.back_trace_id)
        self.layer._host_segmented_trace_active = False
        self._release_front(self.front)
        if self.output.is_allocated():
            ttnn.deallocate(self.output)
        self.released = True


__all__ = [
    "HostBackedSegmentedDecodeTrace",
    "HostDecodeAttention",
    "HostDecodeFront",
    "MultichipDecoder",
    "MultichipMemoryPlan",
]

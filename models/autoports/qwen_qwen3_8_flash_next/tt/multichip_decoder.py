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
from collections.abc import Mapping
from contextlib import contextmanager

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import functional_decoder as _functional_decoder
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
_SHAPE_OVERRIDE_LOCK = threading.RLock()


@dataclasses.dataclass(frozen=True)
class MultichipMemoryPlan:
    """Calculated full-stack residency boundary for the fixed P300 mesh."""

    dram_bytes: int = DRAM_BYTES_PER_DEVICE
    runtime_reserve_bytes: int = RUNTIME_RESERVE_BYTES
    cache_bytes: int = MAX_CONTEXT_CACHE_BYTES_PER_DEVICE
    non_expert_weight_bytes: int = NON_EXPERT_WEIGHT_BYTES_PER_DEVICE
    expert_tiles: int = EXPERT_TILES_PER_DEVICE

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

        with _rank_local_shape_contract(hf_config, layer_idx):
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
        return primary

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
        return self._all_reduce_block(super()._moe(*args, **kwargs))


__all__ = ["MultichipDecoder", "MultichipMemoryPlan"]

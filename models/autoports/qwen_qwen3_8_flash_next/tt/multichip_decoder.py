# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Two-die tensor-parallel Qwen3.8-Flash-Next decoder layer.

The fixed target is the 1x2 Blackhole P300 mesh present on the bring-up host.
Each rank owns half of the QSA head groups, shared-expert intermediate, and
every hyperconnection stream's hidden width.  Routed experts use deterministic
EP2 ownership with a full 640-wide projection on one rank and an exact-zero
slot on the other, avoiding a numerically divergent split-K down projection.
Stack-internal
residuals stay fractured as ``[1,1,4*M,1280]``: QSA and MoE use reduce-scatter,
while replicated-head GDN shards only its output projection.  PLE, indexer,
router inputs, and the 48-head GDN recurrence remain replicated.  Replicated
GDN recurrence is a deliberate correctness result: splitting it into two
24-head kernels loses too much numerical agreement.

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
from contextlib import contextmanager, nullcontext

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
# The mesh must be opened with this router payload before constructing a
# decoder; the hardware tests and context contract expose the same setting.
COLLECTIVE_NUM_LINKS = 2
FABRIC_PACKET_BYTES = 8192
RESIDUAL_SHARD_WIDTH = 1280
DRAM_BYTES_PER_DEVICE = 34_225_520_640
RUNTIME_RESERVE_BYTES = 1 << 30
# BFP8 local main K/V and raw index caches plus BF16 compressed index caches
# across the twelve QSA layers at batch 1 and 262,144 tokens.
MAX_CONTEXT_CACHE_BYTES_PER_DEVICE = 2_340_421_632
# Metadata-derived, tile-padded physical storage for the delivered decoder and
# the natural TP2 full-text endpoints.  NON_EXPERT_WEIGHT_INVENTORY.md and its
# CPU gate derive every row from checkpoint shapes, final packing, dtype, and
# mesh placement.  The replicated GDN delta is reported separately to retain
# the rejected TP2 comparison.
TP2_SHARDED_GDN_DECODER_WEIGHT_BYTES_PER_DEVICE = 2_914_749_440
REPLICATED_GDN_OVERHEAD_BYTES_PER_DEVICE = 1_271_914_496
FRACTURED_RESIDUAL_WEIGHT_SAVINGS_PER_DEVICE = 706_805_760
DECODER_NON_EXPERT_WEIGHT_BYTES_PER_DEVICE = (
    TP2_SHARDED_GDN_DECODER_WEIGHT_BYTES_PER_DEVICE
    + REPLICATED_GDN_OVERHEAD_BYTES_PER_DEVICE
    - FRACTURED_RESIDUAL_WEIGHT_SAVINGS_PER_DEVICE
)
FULL_TEXT_ENDPOINT_WEIGHT_BYTES_PER_DEVICE = 1_279_016_960
NON_EXPERT_WEIGHT_BYTES_PER_DEVICE = (
    DECODER_NON_EXPERT_WEIGHT_BYTES_PER_DEVICE + FULL_TEXT_ENDPOINT_WEIGHT_BYTES_PER_DEVICE
)
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
ROW_PARALLEL_ROLES = frozenset({"attn_out", "shared_down_proj"})
HOST_EXPERT_SLOTS = 10
HOST_PACKED_EXPERTS = 512
FULL_STACK_EXPERT_CACHE_BYTES_PER_DEVICE = 48 * (HOST_EXPERT_SLOTS + 2) * EXPERT_PACKED_BYTES_PER_RANK
PLE_STAGING_BYTES_PER_DEVICE = (128 + 32) * 2560 * 2
# Persistent batch-one decode state is canonical in DRAM.  The exact optimized
# L1 compute tensors are staged only while one layer executes, so the 36 GDN
# layers do not consume more worker L1 than the device physically provides.
GDN_RECURRENT_BYTES_PER_LAYER = 48 * 128 * 128 * 4
GDN_COMBINED_CONV_BYTES_PER_LAYER = 32 * 10240 * 4
GDN_CONV_BYTES_PER_LAYER = 3 * 32 * 10240 * 4
PLE_COMBINED_CONV_BYTES = 32 * 10240 * 2
PLE_CONV_BYTES = 9 * 32 * 10240 * 2
FULL_STACK_DECODE_STATE_BYTES_PER_DEVICE = (
    36 * (GDN_RECURRENT_BYTES_PER_LAYER + GDN_CONV_BYTES_PER_LAYER) + PLE_CONV_BYTES
)
FULL_STACK_PREFILL_STATE_BYTES_PER_DEVICE = (
    36 * (GDN_RECURRENT_BYTES_PER_LAYER + 2 * GDN_COMBINED_CONV_BYTES_PER_LAYER) + 2 * PLE_COMBINED_CONV_BYTES
)
GDN_TRANSIENT_L1_BYTES_PER_WORKER = 7 * 4096 + 3 * 3 * 4096
PLE_TRANSIENT_L1_BYTES_PER_WORKER = 9 * 3 * 2048
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
        return (
            self.dram_bytes
            - self.runtime_reserve_bytes
            - self.cache_bytes
            - self.non_expert_weight_bytes
            - self.ple_staging_bytes
            - self.all_runtime_state_bytes
        )

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
        return 48 * (self.host_expert_slots + 2) * EXPERT_PACKED_BYTES_PER_RANK

    @property
    def decode_state_bytes(self) -> int:
        return FULL_STACK_DECODE_STATE_BYTES_PER_DEVICE

    @property
    def prefill_state_bytes(self) -> int:
        return FULL_STACK_PREFILL_STATE_BYTES_PER_DEVICE

    @property
    def all_runtime_state_bytes(self) -> int:
        return self.decode_state_bytes + self.prefill_state_bytes

    @property
    def transient_l1_state_bytes_per_worker(self) -> int:
        return GDN_TRANSIENT_L1_BYTES_PER_WORKER + PLE_TRANSIENT_L1_BYTES_PER_WORKER

    @property
    def host_backed_stack_bytes(self) -> int:
        return (
            self.runtime_reserve_bytes
            + self.cache_bytes
            + self.non_expert_weight_bytes
            + self.host_expert_cache_bytes
            + self.ple_staging_bytes
            + self.all_runtime_state_bytes
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


def _rank_local_config(hf_config, layer_idx: int | None = None, *, expert_parallel: bool = False):
    """Clone the HF config and express one of the two equal TP ranks."""

    local = copy.deepcopy(hf_config)
    cfg = local.text_config
    names = ["shared_expert_intermediate_size"]
    if not expert_parallel:
        names.append("moe_intermediate_size")
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
def _rank_local_shape_contract(global_config, layer_idx: int, *, expert_parallel: bool = False):
    """Let the exact-target loader materialize one rank-local TP graph.

    ``FunctionalDecoder`` intentionally validates only checkpoint-global
    shapes.  Multichip setup first performs that validation, then narrows the
    resulting immutable shape record while holding a process-wide lock.  The
    original resolver is restored before setup returns; no runtime method and
    no concurrent model construction can observe the local resolver.
    """

    global_shapes = _target_decoder_shapes(global_config, layer_idx)
    replacements = {
        "shared_expert_intermediate_size": global_shapes.shared_expert_intermediate_size // TP_SIZE,
    }
    if not expert_parallel:
        replacements["moe_intermediate_size"] = global_shapes.moe_intermediate_size // TP_SIZE
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


def _install_fractured_residual_weights(layer) -> None:
    """Replace replicated HC/GDN outputs with exact within-stream TP2 shards.

    This is setup-only.  The original optimized weights are replicated on both
    ranks, so mesh-partitioning their four-stream axes preserves the represented
    BFP8 values exactly without a host round trip.  No conversion occurs in a
    forward, capture, or replay path.
    """

    replacements = {}
    created = []
    try:
        for prefix in ("attn_hc", "mlp_hc"):
            norm_name = f"{prefix}_norm"
            norm = layer.w[norm_name]
            norm_groups = ttnn.reshape(norm, (1, 1, layer.shapes.hc_count, layer.shapes.hidden_size))
            local_norm = ttnn.mesh_partition(
                norm_groups,
                dim=3,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            created.append(local_norm)
            replacements[norm_name] = local_norm

            down_name = f"{prefix}_down_inject"
            down = layer.w[down_name]
            down_groups = ttnn.reshape(
                down,
                (
                    1,
                    layer.shapes.hc_count,
                    layer.shapes.hidden_size,
                    layer.shapes.hc_lowrank + layer.shapes.hc_count,
                ),
            )
            local_down = ttnn.mesh_partition(
                down_groups,
                dim=2,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            created.append(local_down)
            replacements[down_name] = ttnn.reshape(
                local_down,
                (
                    1,
                    1,
                    layer.shapes.hc_hidden_size // TP_SIZE,
                    layer.shapes.hc_lowrank + layer.shapes.hc_count,
                ),
            )

            up_name = f"{prefix}_up"
            up = layer.w[up_name]
            up_groups = ttnn.reshape(
                up,
                (
                    1,
                    layer.shapes.hc_lowrank,
                    layer.shapes.hc_count,
                    layer.shapes.hidden_size,
                ),
            )
            local_up = ttnn.mesh_partition(
                up_groups,
                dim=3,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            created.append(local_up)
            replacements[up_name] = ttnn.reshape(
                local_up,
                (1, 1, layer.shapes.hc_lowrank, layer.shapes.hc_hidden_size // TP_SIZE),
            )

        if layer.shapes.layer_type == LINEAR_ATTENTION:
            output = layer.w["gdn_out"]
            local_output = ttnn.mesh_partition(
                output,
                dim=-1,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            created.append(local_output)
            replacements["gdn_out"] = local_output
        ttnn.synchronize_device(layer.mesh_device)
    except Exception:
        for tensor in created:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)
        raise

    for name, replacement in replacements.items():
        original = layer.w[name]
        group = layer.weight_group_by_id.pop(id(original), None)
        role = layer.weight_role_by_id.pop(id(original), None)
        layer.w[name] = replacement
        if group is not None:
            layer.weight_group_by_id[id(replacement)] = group
        if role is not None:
            layer.weight_role_by_id[id(replacement)] = role
        if original.is_allocated():
            ttnn.deallocate(original)


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


class MultichipDecodeStateWorkspace:
    """One fixed-address L1 state workspace shared by a batch-one layer stack.

    Layer-owned canonical state stays in DRAM.  Hydrate/compute/commit commands
    are captured against these stable L1 addresses, so all 36 GDN layer traces
    may reuse the workspace sequentially on CQ0 without retaining 36 copies in
    worker L1.
    """

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self._lock = threading.RLock()
        self._bound = False
        self._trace_users = 0
        self.closed = False
        allocated = []
        try:
            self.recurrent_state = ttnn.empty(
                (1, 48, 128, 128),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            allocated.append(self.recurrent_state)
            self.conv_state = tuple(
                ttnn.empty(
                    (1, 1, 1, 10240),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                for _ in range(3)
            )
            allocated.extend(self.conv_state)
            self.ple_conv_state = tuple(
                ttnn.empty(
                    (1, 1, 1, 10240),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                for _ in range(9)
            )
            allocated.extend(self.ple_conv_state)
        except Exception:
            for tensor in reversed(allocated):
                if tensor.is_allocated():
                    ttnn.deallocate(tensor)
            raise

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("decode-state workspace is closed")

    @staticmethod
    def _copy_state(sources, targets) -> None:
        if len(sources) != len(targets):
            raise RuntimeError(
                f"decode-state topology mismatch: {len(sources)} canonical tensors, {len(targets)} L1 tensors"
            )
        for source, target in zip(sources, targets):
            if (
                tuple(source.shape) != tuple(target.shape)
                or tuple(source.padded_shape) != tuple(target.padded_shape)
                or source.dtype != target.dtype
                or source.get_layout() != target.get_layout()
            ):
                raise RuntimeError("decode-state canonical and L1 tensor contracts differ")
            ttnn.copy(source, target)

    @contextmanager
    def bind_gdn(self, layer):
        """Temporarily bind stable L1 GDN tensors around one eager enqueue."""

        with self._lock:
            self._require_open()
            if self._bound:
                raise RuntimeError("decode-state workspace cannot execute concurrent layers")
            canonical_recurrent = layer.recurrent_state
            canonical_conv = layer.fused_conv_state
            if canonical_recurrent.memory_config() != ttnn.DRAM_MEMORY_CONFIG or any(
                tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG for tensor in canonical_conv
            ):
                raise RuntimeError("shared GDN workspace requires DRAM-canonical layer state")
            if (
                tuple(canonical_recurrent.shape) != tuple(self.recurrent_state.shape)
                or tuple(canonical_recurrent.padded_shape) != tuple(self.recurrent_state.padded_shape)
                or canonical_recurrent.dtype != self.recurrent_state.dtype
                or canonical_recurrent.get_layout() != self.recurrent_state.get_layout()
            ):
                raise RuntimeError("GDN recurrent-state canonical and L1 tensor contracts differ")
            self._bound = True
            try:
                ttnn.copy(canonical_recurrent, self.recurrent_state)
                self._copy_state(canonical_conv, self.conv_state)
                layer.recurrent_state = self.recurrent_state
                layer.fused_conv_state = self.conv_state
                yield
                ttnn.copy(self.recurrent_state, canonical_recurrent)
                self._copy_state(self.conv_state, canonical_conv)
            finally:
                layer.recurrent_state = canonical_recurrent
                layer.fused_conv_state = canonical_conv
                self._bound = False

    @contextmanager
    def bind_ple(self, layer):
        """Temporarily bind stable L1 PLE tensors around one eager enqueue."""

        with self._lock:
            self._require_open()
            if self._bound:
                raise RuntimeError("decode-state workspace cannot execute concurrent layers")
            canonical = layer.fused_ple_conv_state
            if any(tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG for tensor in canonical):
                raise RuntimeError("shared PLE workspace requires DRAM-canonical layer state")
            self._bound = True
            try:
                self._copy_state(canonical, self.ple_conv_state)
                layer.fused_ple_conv_state = self.ple_conv_state
                yield
                self._copy_state(self.ple_conv_state, canonical)
            finally:
                layer.fused_ple_conv_state = canonical
                self._bound = False

    @contextmanager
    def serialize_replay(self):
        """Prevent two shared-workspace traces from interleaving on the host."""

        with self._lock:
            self._require_open()
            yield

    def retain_trace(self) -> None:
        with self._lock:
            self._require_open()
            self._trace_users += 1

    def release_trace(self) -> None:
        with self._lock:
            if self._trace_users <= 0:
                raise RuntimeError("decode-state workspace trace ownership underflow")
            self._trace_users -= 1

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            if self._bound or self._trace_users:
                raise RuntimeError("decode-state workspace is still bound to active work")
            for tensor in (
                self.recurrent_state,
                *self.conv_state,
                *self.ple_conv_state,
            ):
                if tensor.is_allocated():
                    ttnn.deallocate(tensor)
            self.closed = True


class MultichipDecoder(OptimizedDecoder):
    """Optimized Qwen decoder layer tensor-parallelized over the fixed P300."""

    TP_SIZE = TP_SIZE
    TARGET_MESH = TARGET_MESH
    COLLECTIVE_NUM_LINKS = COLLECTIVE_NUM_LINKS
    FABRIC_PACKET_BYTES = FABRIC_PACKET_BYTES
    OPTIMIZATION_MANIFEST = OptimizedDecoder.OPTIMIZATION_MANIFEST + (
        "p300_1x2_tensor_parallel_heads_and_experts",
        "rank_local_paged_kv_cache",
        "replicated_indexer_selection",
        "persistent_within_stream_fractured_residual",
        "qsa_and_moe_output_reduce_scatter",
        "two_link_8192_byte_fabric_payload",
        "gdn_output_column_parallel",
        "distributed_hyperconnection_rmsnorm_and_projections",
        "exact_checkpoint_host_expert_cache",
        "exact_mmap_ple_row_lookup",
        "fixed_generation_checked_expert_slots",
        "dram_canonical_gdn_state_shared_l1_workspace",
        "direct_persistent_gdn_tap_trace_commit",
        "two_phase_stack_trace_program_warm",
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
        expert_host_packed_dtype = kwargs.pop("expert_host_packed_dtype", "bfp4")
        expert_host_packed_layout = kwargs.pop("expert_host_packed_layout", "tile")
        expert_device_staging_dtype = kwargs.pop("expert_device_staging_dtype", "bfp4")
        expert_device_staging_layout = kwargs.pop("expert_device_staging_layout", "tile")
        ple_staging_dtype = kwargs.pop("ple_staging_dtype", "bf16")
        ple_staging_layout = kwargs.pop("ple_staging_layout", "tile")
        ple_prefill_rows = int(kwargs.pop("ple_prefill_rows", 128))
        collective_num_links = int(kwargs.pop("collective_num_links", COLLECTIVE_NUM_LINKS))
        if collective_num_links not in (1, 2):
            raise ValueError("P300 TP2 collective_num_links must be 1 or 2")
        collective_payload_dtype = kwargs.pop("collective_payload_dtype", "bf16")
        if collective_payload_dtype not in {"bf16", "bfp8"}:
            raise ValueError("collective_payload_dtype must be 'bf16' or 'bfp8'")
        residual_dtype = kwargs.pop("residual_dtype", "bf16")
        if residual_dtype not in {"bf16", "bfp8"}:
            raise ValueError("residual_dtype must be 'bf16' or 'bfp8'")
        row_parallel_dtype = kwargs.pop("row_parallel_dtype", "bf16")
        if row_parallel_dtype not in {"bf16", "fp32"}:
            raise ValueError("row_parallel_dtype must be 'bf16' or 'fp32'")
        ple_store = kwargs.pop("ple_store", None)
        decode_state_workspace = kwargs.pop("decode_state_workspace", None)
        fractured_residual = bool(kwargs.pop("fractured_residual", True))
        if not fractured_residual and host_expert_source is not None:
            raise ValueError("replicated-residual A/B is supported only for resident decoder measurements")
        if decode_state_workspace is not None and not isinstance(decode_state_workspace, MultichipDecodeStateWorkspace):
            raise TypeError("decode_state_workspace must be a MultichipDecodeStateWorkspace")
        if decode_state_workspace is not None and decode_state_workspace.mesh_device is not mesh_device:
            raise ValueError("decode_state_workspace must belong to the decoder mesh")
        if decode_state_workspace is not None:
            decode_state_workspace._require_open()
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

        expert_parallel = host_expert_source is not None
        local_config = _rank_local_config(hf_config, layer_idx, expert_parallel=expert_parallel)
        is_qsa = local_config.text_config.layer_types[layer_idx] != LINEAR_ATTENTION
        if decode_state_workspace is not None and (
            host_expert_source is None or int(kwargs.get("max_batch", 1)) != 1 or is_qsa
        ):
            raise ValueError("shared decode-state workspace is valid only for batch-one host-backed GDN")
        # GDN is intentionally replicated.  Only QSA head groups and MoE
        # intermediate dimensions are tensor parallel.
        shard_gdn = False
        local_kwargs = dict(kwargs)
        # Retain the exact single-chip program contracts for replicated GDN.
        # Disable incompatible global-width configs for local QSA projections;
        # sparse MoE gets the legal TP-local geometry below.
        if is_qsa:
            # Decode is M=1, so keep both dominant QSA projections width
            # sharded on their legal TP-local grids.  This avoids an
            # interleaved activation round trip without changing the
            # fractured inter-layer residual ABI or public logical shape.
            local_kwargs.setdefault("decode_1d_config", "qsa_input:110,attn_out:20")
            local_kwargs.setdefault("prefill_config", "")
            local_kwargs.setdefault("dram_sharded_role", "")
            # This exact optimized-baseline candidate already clears QSA PCC
            # and trace gates.  On TP2 it saves 1.7578125 GiB/device at maximum
            # context, which is required capacity rather than a cosmetic win.
            local_kwargs.setdefault("cache_policy", "bfp8")
        if fractured_residual:
            # The optimized auxiliary DRAM-sharded weights preserve the
            # pre-fracture full width.  The fixed S topology instead owns
            # compact local weights and must never select those stale copies.
            requested_dram_roles = {
                role.strip() for role in local_kwargs.get("dram_sharded_role", "").split(",") if role.strip()
            }
            if requested_dram_roles - {"qsa_input", "attn_out"}:
                raise ValueError("fractured residual supports DRAM sharding only for qsa_input and attn_out")
            local_kwargs.setdefault("dram_sharded_role", "")
        local_kwargs.setdefault(
            "optimization_policy",
            "expert_bfp4_lofi_g40b16_d40b5" if expert_parallel else "expert_bfp4_lofi_g20b16_d40b5",
        )

        with _rank_local_shape_contract(hf_config, layer_idx, expert_parallel=expert_parallel) as local_shapes:
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
        primary.collective_num_links = collective_num_links
        primary.collective_payload_dtype = collective_payload_dtype
        primary.residual_dtype = residual_dtype
        primary.row_parallel_dtype = row_parallel_dtype
        primary.memory_plan = MultichipMemoryPlan()
        primary.host_expert_source = host_expert_source
        primary.host_expert_cache = None
        primary.host_ple_store = ple_store
        primary.ple_staging = None
        primary._host_route_ids = None
        primary._host_route_rows = None
        primary._host_logical_route_rows = None
        primary._last_host_service_timing = None
        primary._host_boundary_active = False
        primary._host_segmented_trace_active = False
        primary.fractured_residual = fractured_residual
        primary.decode_state_workspace = None
        primary._owns_decode_state_workspace = False
        primary.host_setup_expert_shapes = tuple(bounded_expert_shapes)
        if fractured_residual:
            _install_fractured_residual_weights(primary)
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
                packed_dtype=expert_host_packed_dtype,
                packed_layout=expert_host_packed_layout,
                staging_dtype=expert_device_staging_dtype,
                staging_layout=expert_device_staging_layout,
            )
        if ple_store is not None:
            primary.ple_staging = PLEDeviceStaging(
                mesh_device,
                max_batch=primary.max_batch,
                prefill_rows=ple_prefill_rows,
                dtype=ple_staging_dtype,
                layout=ple_staging_layout,
            )
        if host_expert_source is not None and primary.max_batch == 1 and not is_qsa:
            if decode_state_workspace is None:
                decode_state_workspace = MultichipDecodeStateWorkspace(mesh_device)
                primary._owns_decode_state_workspace = True
            try:
                primary._canonicalize_batch_one_decode_state()
            except Exception:
                if primary._owns_decode_state_workspace:
                    decode_state_workspace.close()
                raise
            primary.decode_state_workspace = decode_state_workspace
        return primary

    def _canonicalize_batch_one_decode_state(self) -> None:
        """Move split optimized decode state to DRAM without changing values."""

        if self.max_batch != 1 or self.shapes.layer_type != LINEAR_ATTENTION:
            raise RuntimeError("DRAM-canonical state is defined only for batch-one GDN")
        originals = (self.recurrent_state, *self.fused_conv_state, *getattr(self, "fused_ple_conv_state", ()))
        replacements = []
        try:
            for tensor in originals:
                replacements.append(ttnn.clone(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ttnn.synchronize_device(self.mesh_device)
        except Exception:
            for tensor in replacements:
                if tensor.is_allocated():
                    ttnn.deallocate(tensor)
            raise

        conv_end = 1 + len(self.fused_conv_state)
        self.recurrent_state = replacements[0]
        self.fused_conv_state = tuple(replacements[1:conv_end])
        if self.shapes.has_ple:
            self.fused_ple_conv_state = tuple(replacements[conv_end:])
        for tensor in originals:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)

    def _slice_fractured_seq(self, tensor, start: int, logical: int, padded: int):
        """Slice token-major four-stream rows without exposing padding."""

        width = RESIDUAL_SHARD_WIDTH
        first = self.shapes.hc_count * start
        count = self.shapes.hc_count * logical
        piece = ttnn.slice(tensor, [0, 0, first, 0], [1, 1, first + count, width])
        padded_rows = self.shapes.hc_count * padded
        if count != padded_rows:
            result = ttnn.pad(piece, [(0, 0), (0, 0), (0, padded_rows - count), (0, 0)], 0.0)
            _functional_decoder._free(piece, tensor, result)
            piece = result
        return piece

    def _ple_prefill_fractured(self, local, embeddings, *, user_id: int, logical: int):
        """Run the one replicated PLE layer and return its local residual update."""

        replicated = self.gather_residual(local)
        ple = self._ple_prefill(replicated, embeddings, user_id=user_id, logical=logical)
        local_ple = self.fracture_residual(ple)
        updated = ttnn.add(local, local_ple)
        ttnn.deallocate(replicated)
        ttnn.deallocate(ple)
        ttnn.deallocate(local_ple)
        return updated

    def _ple_decode_fractured(self, local, embeddings):
        """Decode counterpart of the layer-1 replicated PLE bridge."""

        replicated = self.gather_residual(local)
        ple = self._ple_decode(replicated, embeddings)
        local_ple = self.fracture_residual(ple)
        updated = ttnn.add(local, local_ple)
        ttnn.deallocate(replicated)
        ttnn.deallocate(ple)
        ttnn.deallocate(local_ple)
        return updated

    def prefill_forward_fractured(
        self,
        hidden_states,
        *,
        user_id: int = 0,
        page_table=None,
        page_tables_per_chunk=None,
        rot_mats=None,
        ple_embeddings=None,
    ):
        """Run prefill with the stack-internal ``[1,1,4*seq,1280]`` ABI."""

        original_hidden_states = hidden_states
        hidden_states = self._residual_compute_input(hidden_states)
        s = self.shapes
        shape = _functional_decoder._shape(hidden_states)
        if len(shape) != 4 or shape[:2] != [1, 1] or shape[-1] != RESIDUAL_SHARD_WIDTH:
            raise ValueError(f"fractured prefill expects [1, 1, 4*seq, {RESIDUAL_SHARD_WIDTH}], got {shape}")
        if shape[-2] % s.hc_count:
            raise ValueError("fractured prefill row count must be divisible by four streams")
        seq_len = shape[-2] // s.hc_count
        if not 1 <= seq_len <= self.max_seq_len:
            raise ValueError(f"prefill seq_len {seq_len} outside [1, {self.max_seq_len}]")
        if not 0 <= user_id < self.max_batch:
            raise ValueError(f"user_id {user_id} outside [0, {self.max_batch})")
        if s.has_ple:
            expected_ple = [1, 1, seq_len, s.ple_embed_dim]
            if ple_embeddings is None or _functional_decoder._shape(ple_embeddings) != expected_ple:
                raise ValueError(f"PLE layer needs embeddings {expected_ple}")
        elif ple_embeddings is not None:
            raise ValueError("ple_embeddings were passed to a layer without PLE")
        plan = self.prefill_chunk_plan(seq_len)
        if s.layer_type != LINEAR_ATTENTION:
            if page_table is None or page_tables_per_chunk is None or rot_mats is None:
                raise ValueError("QSA prefill requires page_table, page_tables_per_chunk and full RoPE tables")
            if len(page_tables_per_chunk) != len(plan):
                raise ValueError("page_tables_per_chunk does not match prefill_chunk_plan")

        self._decode_active = False
        self._reset_user_state(user_id)
        pieces = []
        for chunk_index, (start, logical, padded) in enumerate(plan):
            x = self._slice_fractured_seq(hidden_states, start, logical, padded)
            if s.has_ple:
                embedding_chunk = _functional_decoder._slice_seq(ple_embeddings, start, logical, padded)
                updated = self._ple_prefill_fractured(x, embedding_chunk, user_id=user_id, logical=logical)
                _functional_decoder._free(embedding_chunk, ple_embeddings)
                _functional_decoder._free(x, hidden_states, updated)
                x = updated

            mixed, hyper, injection = self._hyper_mix(x, "attn_hc")
            if s.layer_type == LINEAR_ATTENTION:
                block = self._gdn_prefill(mixed, user_id=user_id, logical=logical)
            else:
                block = self._qsa_prefill(
                    mixed,
                    page_table=page_table,
                    chunk_page_table=page_tables_per_chunk[chunk_index],
                    chunk_start=start,
                    rot_mats=rot_mats,
                )
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
                trimmed = ttnn.slice(
                    out,
                    [0, 0, 0, 0],
                    [1, 1, s.hc_count * logical, RESIDUAL_SHARD_WIDTH],
                )
                _functional_decoder._free(out, trimmed)
                out = trimmed
            pieces.append(out)

        if len(pieces) == 1:
            output = pieces[0]
        else:
            output = ttnn.concat(pieces, dim=-2)
            for piece in pieces:
                ttnn.deallocate(piece)
        return self._residual_boundary_output(output, original_hidden_states)

    def decode_forward_fractured(
        self,
        hidden_states,
        *,
        current_pos,
        page_table=None,
        rot_mats=None,
        ple_embeddings=None,
    ):
        """Run decode with the stack-internal ``[1,1,4*batch,1280]`` ABI."""

        original_hidden_states = hidden_states
        hidden_states = self._residual_compute_input(hidden_states)
        s = self.shapes
        expected = [1, 1, s.hc_count * self.max_batch, RESIDUAL_SHARD_WIDTH]
        if _functional_decoder._shape(hidden_states) != expected:
            raise ValueError(f"fractured decode expects {expected}, got {_functional_decoder._shape(hidden_states)}")
        if current_pos is None or _functional_decoder._shape(current_pos) != [self.max_batch]:
            raise ValueError(f"decode current_pos must be device int32 [{self.max_batch}]")
        if s.has_ple:
            expected_ple = [1, 1, self.max_batch, s.ple_embed_dim]
            if ple_embeddings is None or _functional_decoder._shape(ple_embeddings) != expected_ple:
                raise ValueError(f"PLE decode embeddings must be {expected_ple}")
        elif ple_embeddings is not None:
            raise ValueError("ple_embeddings were passed to a layer without PLE")

        self._decode_active = True
        try:
            if s.has_ple:
                hidden_states = self._ple_decode_fractured(hidden_states, ple_embeddings)
            mixed, hyper, injection = self._hyper_mix(hidden_states, "attn_hc")
            if s.layer_type == LINEAR_ATTENTION:
                block = self._gdn_decode(mixed)
            else:
                if page_table is None or rot_mats is None:
                    raise ValueError("QSA decode requires page_table and full RoPE tables")
                block = self._qsa_decode(mixed, current_pos=current_pos, page_table=page_table, rot_mats=rot_mats)
            ttnn.deallocate(mixed)
            hidden = self._hyper_inject(hyper, block, injection)
            mixed, hyper, injection = self._hyper_mix(hidden, "mlp_hc")
            block = self._moe(mixed)
            ttnn.deallocate(mixed)
            output = self._hyper_inject(hyper, block, injection)
            return self._residual_boundary_output(output, original_hidden_states)
        finally:
            self._decode_active = False

    def prefill_forward(self, hidden_states, **kwargs):
        """Standalone compatibility wrapper around the fractured stack ABI."""

        if not self.fractured_residual:
            return super().prefill_forward(hidden_states, **kwargs)
        local = self.fracture_residual(hidden_states)
        try:
            output = self.prefill_forward_fractured(local, **kwargs)
            gathered = self.gather_residual(output)
            _functional_decoder._free(output, gathered)
            return gathered
        finally:
            _functional_decoder._free(local)

    def decode_forward(self, hidden_states, **kwargs):
        """Standalone compatibility wrapper around the fractured stack ABI."""

        if not self.fractured_residual:
            return super().decode_forward(hidden_states, **kwargs)
        local = self.fracture_residual(hidden_states)
        try:
            output = self.decode_forward_fractured(local, **kwargs)
            gathered = self.gather_residual(output)
            _functional_decoder._free(output, gathered)
            return gathered
        finally:
            _functional_decoder._free(local)

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
        plan = self.host_expert_cache.ensure_indexed(route_ids)
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
        """TT-only indexed expert graph for already serviced stable slots."""

        s = self.shapes
        stable_slots = self.host_expert_cache.slots
        gate_up_bank = ttnn.concat([slot.gate_up for slot in stable_slots], dim=1)
        down_bank = ttnn.concat([slot.down for slot in stable_slots], dim=1)
        local_indices = self.host_expert_cache.local_indices
        cache_capacity = self.host_expert_cache.capacity
        if cache_capacity == s.num_experts_per_tok:
            padded_route_weights = route_weights
        else:
            padded_route_weights = ttnn.pad(
                route_weights,
                [(0, 0), (0, 0), (0, 0), (0, cache_capacity - s.num_experts_per_tok)],
                0.0,
            )
        grouped_x = ttnn.reshape(x, (1, 1, 32, s.hidden_size))
        sparsity = ttnn.to_layout(padded_route_weights, ttnn.ROW_MAJOR_LAYOUT)
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
        gate_up = ttnn.reshape(gate_up_sparse, (1, cache_capacity, 32, 2 * s.moe_intermediate_size))
        _functional_decoder._free(gate_up_sparse, gate_up)
        gate = self._slice_last(gate_up, 0, s.moe_intermediate_size)
        up = self._slice_last(gate_up, s.moe_intermediate_size, 2 * s.moe_intermediate_size)
        ttnn.deallocate(gate_up)
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        selected_weights = ttnn.permute(padded_route_weights, (0, 3, 2, 1))
        weighted_hidden = ttnn.multiply(hidden, selected_weights)
        ttnn.deallocate(hidden)
        _functional_decoder._free(selected_weights, padded_route_weights, weighted_hidden)
        if padded_route_weights is not route_weights:
            _functional_decoder._free(padded_route_weights, route_weights, weighted_hidden)
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
        if sparsity is not None and sparsity.is_allocated():
            ttnn.deallocate(sparsity)
        # ``ttnn.concat`` may return an alias when a prefill wave contains a
        # single expert.  Preserve the fixed-address cache slots in that case;
        # they must remain valid for later decode misses and trace capture.
        _functional_decoder._free(gate_up_bank, *(slot.gate_up for slot in stable_slots))
        _functional_decoder._free(down_bank, *(slot.down for slot in stable_slots))
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
        ttnn.deallocate(sparsity)
        # A last prefill wave can contain exactly one selected expert, for
        # which concat is permitted to alias the persistent slot tensor.
        _functional_decoder._free(gate_up_bank, *(slot.gate_up for slot in ordered_slots))
        _functional_decoder._free(down_bank, *(slot.down for slot in ordered_slots))
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
        payload_dtype = ttnn.bfloat8_b if self.collective_payload_dtype == "bfp8" else ttnn.bfloat16
        if partial.dtype != payload_dtype:
            payload = ttnn.typecast(partial, payload_dtype)
            ttnn.deallocate(partial)
            partial = payload
        output = ttnn.all_reduce(
            partial,
            cluster_axis=self.collective_axis,
            num_links=self.collective_num_links,
            topology=self.collective_topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(partial)
        if output.dtype != ttnn.bfloat16:
            reduced = ttnn.typecast(output, ttnn.bfloat16)
            ttnn.deallocate(output)
            output = reduced
        return output

    def _reduce_scatter_block(self, partial):
        """Sum a row-parallel block and retain its within-hidden TP2 shard."""

        payload_dtype = ttnn.bfloat8_b if self.collective_payload_dtype == "bfp8" else ttnn.bfloat16
        if partial.dtype != payload_dtype:
            payload = ttnn.typecast(partial, payload_dtype)
            ttnn.deallocate(partial)
            partial = payload
        output = ttnn.reduce_scatter(
            partial,
            dim=3,
            cluster_axis=self.collective_axis,
            num_links=self.collective_num_links,
            topology=self.collective_topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(partial)
        if output.dtype != ttnn.bfloat16:
            reduced = ttnn.typecast(output, ttnn.bfloat16)
            ttnn.deallocate(output)
            output = reduced
        return output

    def _residual_compute_input(self, value):
        """Expand a compressed inter-layer boundary for BF16 layer compute."""

        if self.residual_dtype == "bf16":
            if value.dtype != ttnn.bfloat16:
                raise RuntimeError(f"BF16 residual policy received {value.dtype}")
            return value
        if value.dtype == ttnn.bfloat8_b:
            return ttnn.typecast(value, ttnn.bfloat16)
        if value.dtype != ttnn.bfloat16:
            raise RuntimeError(f"BFP8 residual policy received unsupported {value.dtype}")
        return value

    def _residual_boundary_output(self, value, original_input):
        """Materialize the selected inter-layer residual representation."""

        target = ttnn.bfloat8_b if self.residual_dtype == "bfp8" else ttnn.bfloat16
        if value.dtype == target:
            return value
        converted = ttnn.typecast(value, target)
        _functional_decoder._free(value, original_input, converted)
        return converted

    def fracture_residual(self, replicated):
        """One-time stack ingress: R ``[1,1,M,10240]`` -> S ``[1,1,4M,1280]``."""

        shape = _functional_decoder._shape(replicated)
        s = self.shapes
        if len(shape) != 4 or shape[:2] != [1, 1] or shape[-1] != s.hc_hidden_size:
            raise ValueError(f"replicated residual must be [1, 1, M, {s.hc_hidden_size}], got {shape}")
        grouped = ttnn.reshape(replicated, (1, 1, shape[-2] * s.hc_count, s.hidden_size))
        local = ttnn.mesh_partition(
            grouped,
            dim=3,
            cluster_axis=self.collective_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _functional_decoder._free(grouped, replicated, local)
        return local

    def gather_residual(self, local):
        """One-time stack/test exit for a within-stream fractured residual."""

        shape = _functional_decoder._shape(local)
        s = self.shapes
        if len(shape) != 4 or shape[:2] != [1, 1] or shape[-1] != RESIDUAL_SHARD_WIDTH:
            raise ValueError(f"fractured residual must be [1, 1, 4*M, {RESIDUAL_SHARD_WIDTH}], got {shape}")
        if shape[-2] % s.hc_count:
            raise ValueError("fractured residual row count must be divisible by four streams")
        gathered = ttnn.all_gather(
            local,
            dim=3,
            cluster_axis=self.collective_axis,
            num_links=self.collective_num_links,
            topology=self.collective_topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output = ttnn.reshape(gathered, (1, 1, shape[-2] // s.hc_count, s.hc_hidden_size))
        _functional_decoder._free(gathered, output)
        return output

    def _hyper_mix(self, hyper_input, prefix: str):
        """Distributed four-stream hyper mixer over a persistent S residual."""

        if not self.fractured_residual:
            return super()._hyper_mix(hyper_input, prefix)
        s = self.shapes
        shape = _functional_decoder._shape(hyper_input)
        if shape[:2] != [1, 1] or shape[-1] != RESIDUAL_SHARD_WIDTH or shape[-2] % s.hc_count:
            raise ValueError(f"fractured hyper input has invalid shape {shape}")
        rows = shape[-2] // s.hc_count
        stats = ttnn.rms_norm_pre_all_gather(
            hyper_input,
            compute_kernel_config=_functional_decoder._hifi4(fp32=True),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gathered_stats = ttnn.all_gather(
            stats,
            dim=3,
            cluster_axis=self.collective_axis,
            num_links=self.collective_num_links,
            topology=self.collective_topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(stats)
        normed = ttnn.rms_norm_post_all_gather(
            hyper_input,
            gathered_stats,
            epsilon=s.rms_norm_eps,
            compute_kernel_config=_functional_decoder._hifi4(fp32=True),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gathered_stats)
        norm_weight = self.w[f"{prefix}_norm"]
        weight_rows = norm_weight if rows == 1 else ttnn.repeat(norm_weight, (1, 1, rows, 1))
        weighted = ttnn.multiply(normed, weight_rows)
        ttnn.deallocate(normed)
        _functional_decoder._free(weight_rows, norm_weight, weighted)

        flat = ttnn.reshape(weighted, (1, 1, rows, s.hc_hidden_size // TP_SIZE))
        packed_partial = self._linear_impl(flat, self.w[f"{prefix}_down_inject"], dtype=ttnn.bfloat16)
        _functional_decoder._free(flat, weighted, packed_partial)
        packed = self._all_reduce_block(packed_partial)
        low = self._slice_last(packed, 0, s.hc_lowrank)
        injection = self._slice_last(packed, s.hc_lowrank, s.hc_lowrank + s.hc_count)
        ttnn.deallocate(packed)
        low = ttnn.silu(low)
        local_mix = self._linear_impl(low, self.w[f"{prefix}_up"], dtype=ttnn.bfloat16)
        ttnn.deallocate(low)

        norm_groups = ttnn.reshape(weighted, (rows, s.hc_count, RESIDUAL_SHARD_WIDTH))
        mix_groups = ttnn.reshape(local_mix, (rows, s.hc_count, RESIDUAL_SHARD_WIDTH))
        local_mixed = ttnn.multiply(
            norm_groups,
            mix_groups,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        _functional_decoder._free(norm_groups, weighted, local_mixed)
        _functional_decoder._free(mix_groups, local_mix, local_mixed)
        ttnn.deallocate(weighted)
        ttnn.deallocate(local_mix)
        local_mixed = ttnn.mean(local_mixed, dim=1, keepdim=True)
        local_mixed = ttnn.reshape(local_mixed, (1, 1, rows, RESIDUAL_SHARD_WIDTH))
        mixed = ttnn.all_gather(
            local_mixed,
            dim=3,
            cluster_axis=self.collective_axis,
            num_links=self.collective_num_links,
            topology=self.collective_topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(local_mixed)
        return mixed, hyper_input, injection

    def _hyper_inject(self, hyper_input, block_output, injection):
        """Inject one local 1280 block shard into all four local streams."""

        if not self.fractured_residual:
            return super()._hyper_inject(hyper_input, block_output, injection)
        s = self.shapes
        rows = int(block_output.shape[-2])
        value = ttnn.reshape(block_output, (rows, 1, RESIDUAL_SHARD_WIDTH))
        gate = ttnn.reshape(injection, (rows, s.hc_count, 1))
        projected = ttnn.multiply(value, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        _functional_decoder._free(value, block_output, projected)
        _functional_decoder._free(gate, injection, projected)
        ttnn.deallocate(block_output)
        ttnn.deallocate(injection)
        projected = ttnn.reshape(projected, (1, 1, rows * s.hc_count, RESIDUAL_SHARD_WIDTH))
        output = ttnn.mac(projected, 2.0, hyper_input)
        ttnn.deallocate(projected)
        return output

    def _linear(self, x, weight, *, dtype=ttnn.bfloat16):
        role = self.weight_role_by_id.get(id(weight))
        if self.row_parallel_dtype == "fp32" and role in ROW_PARALLEL_ROLES and dtype == ttnn.bfloat16:
            dtype = ttnn.float32
        return self._linear_impl(x, weight, dtype=dtype)

    def _gdn_prefill(self, *args, **kwargs):
        return super()._gdn_prefill(*args, **kwargs)

    def _ple_decode(self, *args, **kwargs):
        workspace = self.decode_state_workspace
        if workspace is None or self.max_batch != 1:
            return super()._ple_decode(*args, **kwargs)
        with workspace.bind_ple(self):
            return super()._ple_decode(*args, **kwargs)

    def _commit_newest_gdn_state_direct(self, x) -> None:
        """Reproduce the mixed projection as the persistent tap's final writer."""

        s = self.shapes
        packed = self._linear_impl(
            x,
            self.w["gdn_qkv_b_a"],
            bias=self.w["gdn_qkv_b_a_bias"],
            dtype=ttnn.float32,
        )
        lead = _functional_decoder._shape(packed)[:-1]
        ttnn.slice(
            packed,
            [0] * len(lead) + [0],
            lead + [s.linear_qkv_width],
            output_tensor=self.fused_conv_state[-1],
        )
        ttnn.deallocate(packed)

    def _gdn_decode(self, *args, **kwargs):
        if self.max_batch != 1:
            return super()._gdn_decode(*args, **kwargs)
        if kwargs or len(args) != 1:
            raise TypeError("optimized GDN decode expects one input tensor")
        workspace = self.decode_state_workspace
        if workspace is None:
            output = super()._gdn_decode(args[0])
            self._commit_newest_gdn_state_direct(args[0])
            return output
        with workspace.bind_gdn(self):
            output = super()._gdn_decode(args[0])
            self._commit_newest_gdn_state_direct(args[0])
            return output

    def _qsa_prefill(self, *args, **kwargs):
        partial = super()._qsa_prefill(*args, **kwargs)
        return self._reduce_scatter_block(partial) if self.fractured_residual else self._all_reduce_block(partial)

    def _qsa_decode(self, *args, **kwargs):
        partial = super()._qsa_decode(*args, **kwargs)
        return self._reduce_scatter_block(partial) if self.fractured_residual else self._all_reduce_block(partial)

    def _moe(self, *args, **kwargs):
        if self.host_expert_cache is None:
            partial = super()._moe(*args, **kwargs)
            return self._reduce_scatter_block(partial) if self.fractured_residual else self._all_reduce_block(partial)
        if not args:
            raise TypeError("MoE input tensor is required")
        self._host_route_rows = int(self._host_logical_route_rows or args[0].shape[-2])
        self._host_boundary_active = True
        try:
            return self._reduce_scatter_block(super()._moe(*args, **kwargs))
        finally:
            self._host_boundary_active = False
            self._host_route_rows = None

    def _decode_attention_host(
        self, hidden_states, *, current_pos, page_table=None, rot_mats=None, ple_embeddings=None
    ):
        """First TT segment through PLE and GDN/QSA attention."""

        hidden_states = self._residual_compute_input(hidden_states)
        if self.host_expert_cache is None or self.max_batch != 1:
            raise RuntimeError("segmented host trace requires batch-one host expert slots")
        s = self.shapes
        if _functional_decoder._shape(hidden_states) != [1, 1, s.hc_count, RESIDUAL_SHARD_WIDTH]:
            raise ValueError("segmented decode hidden input has the wrong shape")
        if current_pos is None or _functional_decoder._shape(current_pos) != [1]:
            raise ValueError("segmented decode current_pos must be device int32 [1]")
        self._decode_active = True
        try:
            if s.has_ple:
                if ple_embeddings is None or _functional_decoder._shape(ple_embeddings) != [1, 1, 1, s.ple_embed_dim]:
                    raise ValueError("segmented PLE decode requires stable [1, 1, 1, 2560] embeddings")
                front_hidden = self._ple_decode_fractured(hidden_states, ple_embeddings)
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
        rows = int(block_output.shape[-2])
        value = ttnn.reshape(block_output, (rows, 1, RESIDUAL_SHARD_WIDTH))
        gate = ttnn.reshape(injection, (rows, s.hc_count, 1))
        projected = ttnn.multiply(value, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        _functional_decoder._free(value, block_output, projected)
        _functional_decoder._free(gate, injection, projected)
        projected = ttnn.reshape(projected, (1, 1, rows * s.hc_count, RESIDUAL_SHARD_WIDTH))
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
        """Declared D2H ids plus exact indexed expert service between TT segments."""

        route_started = time.perf_counter()
        host = ttnn.to_torch(ttnn.get_device_tensors(front.route_ids)[0]).reshape(-1)
        route_ids = tuple(int(value) for value in host[: self.shapes.num_experts_per_tok].tolist())
        route_seconds = time.perf_counter() - route_started
        if len(route_ids) != len(set(route_ids)):
            raise RuntimeError(f"router returned duplicate top-k expert ids: {route_ids}")
        cache_started = time.perf_counter()
        plan = self.host_expert_cache.ensure_indexed(route_ids)
        self.host_expert_cache.validate(plan)
        self._last_host_service_timing = {
            # With nonblocking layer traces, this compact read is the exact TT
            # completion boundary and therefore includes dependent back/front
            # device work queued since the preceding boundary.
            "route_read_and_tt_stall_seconds": route_seconds,
            "cache_control_dma_submit_seconds": time.perf_counter() - cache_started,
        }
        return route_ids, plan

    def _hyper_inject_preserve(self, hyper_input, block_output, injection):
        """Trace-back injection that preserves front-segment input buffers."""

        s = self.shapes
        rows = int(block_output.shape[-2])
        value = ttnn.reshape(block_output, (rows, 1, RESIDUAL_SHARD_WIDTH))
        gate = ttnn.reshape(injection, (rows, s.hc_count, 1))
        projected = ttnn.multiply(value, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        _functional_decoder._free(value, block_output, projected)
        _functional_decoder._free(gate, injection, projected)
        ttnn.deallocate(block_output)
        projected = ttnn.reshape(projected, (1, 1, rows * s.hc_count, RESIDUAL_SHARD_WIDTH))
        out = ttnn.mac(projected, 2.0, hyper_input)
        ttnn.deallocate(projected)
        return out

    def _decode_back_host(self, front: HostDecodeFront):
        """TT trace segment from fixed indexed expert slots to layer output."""

        routed = self._routed_experts_indexed_ready(front.work, front.route_weights)
        local = ttnn.add(routed, front.shared)
        ttnn.deallocate(routed)
        reduced = self._reduce_scatter_block(local)
        trimmed = ttnn.slice(reduced, [0, 0, 0, 0], [1, 1, 1, RESIDUAL_SHARD_WIDTH])
        _functional_decoder._free(reduced, trimmed)
        reduced = trimmed
        output = self._hyper_inject_preserve(front.hyper, reduced, front.injection)
        return self._residual_boundary_output(output, front.hyper)

    def _require_host_ple(self) -> None:
        if self.host_ple_store is None or self.ple_staging is None or not self.shapes.has_ple:
            raise RuntimeError("exact host PLE service is attached only to zero-based layer 1")

    def reset_host_request(self, request_id) -> None:
        self._require_host_ple()
        self.host_ple_store.reset_request(request_id)

    def cancel_host_request(self, request_id) -> None:
        self._require_host_ple()
        self.host_ple_store.cancel_request(request_id)

    def prefill_forward_host_backed_fractured(
        self,
        hidden_states,
        *,
        input_ids: torch.Tensor,
        request_id,
        user_id: int = 0,
        valid_mask: torch.Tensor | None = None,
    ):
        """Exact host PLE prefill over the stack-internal fractured ABI."""

        self._require_host_ple()
        original_hidden_states = hidden_states
        hidden_states = self._residual_compute_input(hidden_states)
        s = self.shapes
        hidden_shape = _functional_decoder._shape(hidden_states)
        if (
            len(hidden_shape) != 4
            or hidden_shape[:2] != [1, 1]
            or hidden_shape[-1] != RESIDUAL_SHARD_WIDTH
            or hidden_shape[-2] % s.hc_count
        ):
            raise ValueError("host-backed fractured prefill expects [1, 1, 4*seq, 1280]")
        seq_len = hidden_shape[-2] // s.hc_count
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
            x = self._slice_fractured_seq(hidden_states, start, logical, padded)
            embeddings = self.host_ple_store.prepare(
                [request_id],
                ids[:, start : start + logical],
                valid_mask=None if mask is None else mask[:, start : start + logical],
                reset=start == 0,
            )
            staged = self.ple_staging.upload_prefill(embeddings, logical=logical)
            updated = self._ple_prefill_fractured(x, staged, user_id=user_id, logical=logical)
            _functional_decoder._free(x, hidden_states, updated)
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
                trimmed = ttnn.slice(
                    out,
                    [0, 0, 0, 0],
                    [1, 1, s.hc_count * logical, RESIDUAL_SHARD_WIDTH],
                )
                _functional_decoder._free(out, trimmed)
                out = trimmed
            pieces.append(out)
        if len(pieces) == 1:
            output = pieces[0]
        else:
            output = ttnn.concat(pieces, dim=-2)
            for piece in pieces:
                ttnn.deallocate(piece)
        return self._residual_boundary_output(output, original_hidden_states)

    def prefill_forward_host_backed(self, hidden_states, **kwargs):
        """Standalone replicated wrapper for exact host-backed prefill."""

        local = self.fracture_residual(hidden_states)
        try:
            output = self.prefill_forward_host_backed_fractured(local, **kwargs)
            gathered = self.gather_residual(output)
            _functional_decoder._free(output, gathered)
            return gathered
        finally:
            _functional_decoder._free(local)

    def decode_forward_host_backed_fractured(
        self,
        hidden_states,
        *,
        input_ids: torch.Tensor,
        request_ids,
        current_pos,
    ):
        """Service one real PLE row, then execute fractured decode."""

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
        return self.decode_forward_fractured(hidden_states, current_pos=current_pos, ple_embeddings=staged)

    def decode_forward_host_backed(self, hidden_states, **kwargs):
        """Standalone replicated wrapper for exact host-backed decode."""

        local = self.fracture_residual(hidden_states)
        try:
            output = self.decode_forward_host_backed_fractured(local, **kwargs)
            gathered = self.gather_residual(output)
            _functional_decoder._free(output, gathered)
            return gathered
        finally:
            _functional_decoder._free(local)

    def close_host_backing(self) -> None:
        """Release only the host-backed resources owned by this layer."""

        if self._host_segmented_trace_active:
            raise RuntimeError("release the active segmented trace before closing host backing")
        # Close the owned state workspace first.  If it is unexpectedly bound,
        # fail before partially closing the expert/PLE resources.
        if self._owns_decode_state_workspace and self.decode_state_workspace is not None:
            self.decode_state_workspace.close()
            self.decode_state_workspace = None
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
    """Warmed front/back TT traces around exact host PLE/expert service.

    Batch-one GDN writes the mixed prefix of its packed FP32 projection directly
    into the fixed newest L1 workspace tap with ``slice(output_tensor=...)``;
    the workspace then commits it to layer-owned canonical DRAM state.  The
    captured producer therefore retains a fixed destination address without a
    transient slice followed by an address-sensitive copy.
    """

    def __init__(
        self,
        layer,
        front,
        output,
        front_trace_id,
        back_trace_id,
        *,
        captured_inputs=(),
    ):
        self.layer = layer
        self.front = front
        self.output = output
        self.front_trace_id = front_trace_id
        self.back_trace_id = back_trace_id
        self.captured_inputs = tuple(captured_inputs)
        self.last_timing = None
        self.last_route_ids = None
        self.released = False
        self._lock = threading.RLock()

    @staticmethod
    def _decode_state_tensors(layer) -> tuple:
        tensors = []
        if layer.shapes.layer_type == LINEAR_ATTENTION:
            tensors.extend((layer.recurrent_state, *layer.fused_conv_state))
        if layer.shapes.has_ple:
            tensors.extend(layer.fused_ple_conv_state)
        return tuple(tensors)

    @classmethod
    def _snapshot_decode_state(cls, layer) -> tuple:
        snapshots = []
        try:
            for tensor in cls._decode_state_tensors(layer):
                shards = ttnn.get_device_tensors(tensor)
                if len(shards) != TP_SIZE:
                    raise RuntimeError("segmented decode state is not replicated over TP2")
                snapshots.append(ttnn.clone(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ttnn.synchronize_device(layer.mesh_device)
            return tuple(snapshots)
        except Exception:
            cls._release_state_snapshots(snapshots)
            raise

    @classmethod
    def _restore_decode_state(cls, layer, snapshots: tuple) -> None:
        targets = cls._decode_state_tensors(layer)
        if len(targets) != len(snapshots):
            raise RuntimeError("segmented decode state topology changed during capture")
        for source, target in zip(snapshots, targets):
            ttnn.copy(source, target)
        ttnn.synchronize_device(layer.mesh_device)

    @staticmethod
    def _release_state_snapshots(snapshots) -> None:
        for tensor in snapshots:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)

    @staticmethod
    def _release_front(front: HostDecodeFront) -> None:
        for field in dataclasses.fields(front):
            tensor = getattr(front, field.name)
            if tensor.is_allocated():
                ttnn.deallocate(tensor)

    @staticmethod
    def _mark_front_corruptible(front: HostDecodeFront) -> None:
        """Declare retained front outputs safe to overwrite by older traces.

        Every field is produced by this trace before the host service reads it.
        This is what permits several layer traces to remain live: replaying an
        older layer may reuse addresses allocated while that trace was live,
        but the younger layer refreshes all six crossings before consuming
        them.
        """

        for field in dataclasses.fields(front):
            ttnn.mark_corruptible(getattr(front, field.name))

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

    @staticmethod
    def _capture_arguments(layer, *, current_pos, page_table, rot_mats, ple_input_ids, request_ids):
        linear_attention = layer.shapes.layer_type == LINEAR_ATTENTION
        ple_ids = ple_requests = None
        if layer.shapes.has_ple:
            layer._require_host_ple()
            if ple_input_ids is None or request_ids is None:
                raise ValueError("PLE segmented capture requires input ids and request ids")
            ple_requests = tuple(request_ids)
            if len(ple_requests) != layer.max_batch:
                raise ValueError(f"decode needs {layer.max_batch} request ids")
            ple_ids = torch.as_tensor(ple_input_ids, dtype=torch.int64, device="cpu")
            if ple_ids.ndim == 1:
                ple_ids = ple_ids.unsqueeze(1)
            if tuple(ple_ids.shape) != (layer.max_batch, 1):
                raise ValueError(f"decode input ids must be [{layer.max_batch}, 1], got {tuple(ple_ids.shape)}")
        elif ple_input_ids is not None or request_ids is not None:
            raise ValueError("PLE inputs were passed to a layer without PLE")

        if linear_attention and (page_table is not None or rot_mats is not None):
            raise ValueError("GDN segmented trace does not accept QSA page or RoPE inputs")
        return (
            linear_attention,
            ple_ids,
            ple_requests,
            {
                "current_pos": current_pos,
                "page_table": page_table,
                "rot_mats": rot_mats,
                "ple_embeddings": None,
            },
        )

    @staticmethod
    def _snapshot_ple_history(layer, ple_requests):
        if ple_requests is None:
            return None
        with layer.host_ple_store._lock:
            return {
                request_id: (
                    None
                    if request_id not in layer.host_ple_store._histories
                    else layer.host_ple_store._histories[request_id].clone()
                )
                for request_id in ple_requests
            }

    @staticmethod
    def _restore_ple_history(layer, snapshot) -> None:
        if snapshot is None:
            return
        with layer.host_ple_store._lock:
            for request_id, history in snapshot.items():
                if history is None:
                    layer.host_ple_store._histories.pop(request_id, None)
                else:
                    layer.host_ple_store._histories[request_id] = history

    @classmethod
    def warm_programs(
        cls,
        layer: MultichipDecoder,
        hidden_states,
        *,
        current_pos,
        page_table=None,
        rot_mats=None,
        ple_input_ids: torch.Tensor | None = None,
        request_ids=None,
    ) -> None:
        """Compile one capture signature before any stack trace is registered.

        Full stacks call this for every distinct layer kind, then pass
        ``programs_prepared=True`` to ``capture``.  This prevents persistent
        program-cache allocations from being created while an older layer
        trace is live.
        """

        if layer.host_expert_cache is None:
            raise RuntimeError("segmented trace requires host-backed expert slots")
        linear_attention, ple_ids, ple_requests, front_kwargs = cls._capture_arguments(
            layer,
            current_pos=current_pos,
            page_table=page_table,
            rot_mats=rot_mats,
            ple_input_ids=ple_input_ids,
            request_ids=request_ids,
        )
        workspace = layer.decode_state_workspace
        replay_scope = workspace.serialize_replay() if workspace is not None else nullcontext()
        snapshots = ()
        ple_history = None
        front = output = None
        with replay_scope:
            try:
                if linear_attention:
                    snapshots = cls._snapshot_decode_state(layer)
                ple_history = cls._snapshot_ple_history(layer, ple_requests)
                if ple_ids is not None:
                    embeddings = layer.host_ple_store.prepare(ple_requests, ple_ids)
                    front_kwargs["ple_embeddings"] = layer.ple_staging.upload_decode(embeddings)
                front = layer._decode_front_host(hidden_states, **front_kwargs)
                layer.service_decode_front(front)
                output = layer._decode_back_host(front)
                ttnn.synchronize_device(layer.mesh_device)
            finally:
                # Every cleanup step must run even if an earlier deallocation
                # or state restore fails.  In particular, a failed warm must
                # not leak DRAM snapshots or leave PLE host history advanced.
                try:
                    if output is not None and output.is_allocated():
                        ttnn.deallocate(output)
                finally:
                    try:
                        if front is not None:
                            cls._release_front(front)
                    finally:
                        try:
                            if snapshots:
                                cls._restore_decode_state(layer, snapshots)
                        finally:
                            try:
                                cls._release_state_snapshots(snapshots)
                            finally:
                                cls._restore_ple_history(layer, ple_history)

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
        programs_prepared: bool = False,
    ) -> "HostBackedSegmentedDecodeTrace":
        if layer.host_expert_cache is None:
            raise RuntimeError("segmented trace requires host-backed expert slots")
        if layer._host_segmented_trace_active:
            raise RuntimeError("segmented trace is already active for this layer")
        linear_attention, ple_ids, ple_requests, front_kwargs = cls._capture_arguments(
            layer,
            current_pos=current_pos,
            page_table=page_table,
            rot_mats=rot_mats,
            ple_input_ids=ple_input_ids,
            request_ids=request_ids,
        )
        state_workspace = layer.decode_state_workspace
        workspace_retained = False
        if state_workspace is not None:
            with state_workspace._lock:
                if state_workspace._trace_users and not programs_prepared:
                    raise RuntimeError(
                        "capture beside a live shared-workspace trace requires warm_programs() for every layer "
                        "and programs_prepared=True"
                    )
        if state_workspace is not None:
            state_workspace.retain_trace()
            workspace_retained = True
        workspace_lock = state_workspace._lock if state_workspace is not None else None
        if workspace_lock is not None:
            # Trace capture is process/CQ scoped.  Hold the shared-workspace
            # serialization lock across warm, capture, and the deployed first
            # replay so another layer cannot dispatch into this capture.
            workspace_lock.acquire()
        layer._host_segmented_trace_active = True
        front = output = warm_front = warm_output = None
        front_trace_id = back_trace_id = None
        front_open = back_open = cache_locked = False
        state_snapshot = ()
        ple_history_snapshot = None
        try:
            if programs_prepared:
                # Lock before even the snapshot clones: no persistent program
                # buffer may be allocated after an older stack trace exists.
                layer.mesh_device.set_program_cache_misses_allowed(False)
                cache_locked = True
            if linear_attention:
                state_snapshot = cls._snapshot_decode_state(layer)
            if ple_ids is not None:
                ple_history_snapshot = cls._snapshot_ple_history(layer, ple_requests)
                embeddings = layer.host_ple_store.prepare(ple_requests, ple_ids)
                front_kwargs["ple_embeddings"] = layer.ple_staging.upload_decode(embeddings)
                # A prior live layer trace may overwrite this later allocation.
                # Every PLE replay uploads the selected row before the front
                # trace consumes it, so the buffer is intentionally corruptible.
                ttnn.mark_corruptible(front_kwargs["ple_embeddings"])

            warm_front = layer._decode_front_host(hidden_states, **front_kwargs)
            layer.service_decode_front(warm_front)
            warm_output = layer._decode_back_host(warm_front)
            ttnn.synchronize_device(layer.mesh_device)
            if warm_output.is_allocated():
                ttnn.deallocate(warm_output)
            warm_output = None
            cls._release_front(warm_front)
            warm_front = None
            if linear_attention:
                # Warm compilation must not consume the caller's state.
                cls._restore_decode_state(layer, state_snapshot)

            if not cache_locked:
                layer.mesh_device.set_program_cache_misses_allowed(False)
                cache_locked = True
            front_trace_id = ttnn.begin_trace_capture(layer.mesh_device, cq_id=0)
            front_open = True
            front = layer._decode_front_host(hidden_states, **front_kwargs)
            ttnn.end_trace_capture(layer.mesh_device, front_trace_id, cq_id=0)
            front_open = False
            cls._mark_front_corruptible(front)
            if linear_attention:
                # Capturing the progressing GDN front mutates its persistent
                # recurrence.  Restore the user's pre-token state, then make
                # the first deployed invocation an ordinary trace replay.
                cls._restore_decode_state(layer, state_snapshot)
                ttnn.execute_trace(layer.mesh_device, front_trace_id, cq_id=0, blocking=True)
            route_ids, _ = layer.service_decode_front(front)

            back_trace_id = ttnn.begin_trace_capture(layer.mesh_device, cq_id=0)
            back_open = True
            output = layer._decode_back_host(front)
            ttnn.end_trace_capture(layer.mesh_device, back_trace_id, cq_id=0)
            back_open = False
            ttnn.mark_corruptible(output)
            ttnn.execute_trace(layer.mesh_device, back_trace_id, cq_id=0, blocking=True)
            cls._release_state_snapshots(state_snapshot)
            state_snapshot = ()
            layer.mesh_device.set_program_cache_misses_allowed(True)
            cache_locked = False
        except Exception:
            cls._finish_failed_capture(layer.mesh_device, back_trace_id, back_open)
            cls._finish_failed_capture(layer.mesh_device, front_trace_id, front_open)
            if warm_output is not None and warm_output.is_allocated():
                ttnn.deallocate(warm_output)
            if warm_front is not None:
                cls._release_front(warm_front)
            if front is not None:
                cls._release_front(front)
            if output is not None and output.is_allocated():
                ttnn.deallocate(output)
            if state_snapshot:
                try:
                    cls._restore_decode_state(layer, state_snapshot)
                except Exception:
                    pass
                cls._release_state_snapshots(state_snapshot)
            cls._restore_ple_history(layer, ple_history_snapshot)
            if workspace_retained:
                state_workspace.release_trace()
            layer._host_segmented_trace_active = False
            raise
        finally:
            try:
                if cache_locked:
                    layer.mesh_device.set_program_cache_misses_allowed(True)
            finally:
                if workspace_lock is not None:
                    workspace_lock.release()

        trace = cls(
            layer,
            front,
            output,
            front_trace_id,
            back_trace_id,
            captured_inputs=(
                hidden_states,
                current_pos,
                page_table,
                rot_mats,
                front_kwargs["ple_embeddings"],
            ),
        )
        trace.state_workspace = state_workspace
        trace.last_route_ids = route_ids
        return trace

    def replay(self, *, ple_input_ids: torch.Tensor | None = None, request_ids=None):
        with self._lock:
            if self.released:
                raise RuntimeError("segmented trace has been released")
            started = time.perf_counter()
            ple_seconds = 0.0
            workspace = getattr(self, "state_workspace", None)
            replay_scope = workspace.serialize_replay() if workspace is not None else nullcontext()
            with replay_scope:
                if self.layer.shapes.has_ple:
                    if ple_input_ids is None or request_ids is None:
                        raise ValueError("PLE segmented replay requires input ids and request ids")
                    ple_started = time.perf_counter()
                    request_ids = tuple(request_ids)
                    if len(request_ids) != self.layer.max_batch:
                        raise ValueError(f"decode needs {self.layer.max_batch} request ids")
                    ids = torch.as_tensor(ple_input_ids, dtype=torch.int64, device="cpu")
                    if ids.ndim == 1:
                        ids = ids.unsqueeze(1)
                    if tuple(ids.shape) != (self.layer.max_batch, 1):
                        raise ValueError(
                            f"decode input ids must be [{self.layer.max_batch}, 1], got {tuple(ids.shape)}"
                        )
                    embeddings = self.layer.host_ple_store.prepare(request_ids, ids)
                    self.layer.ple_staging.upload_decode(embeddings)
                    ple_seconds = time.perf_counter() - ple_started
                elif ple_input_ids is not None or request_ids is not None:
                    raise ValueError("PLE inputs were passed to a layer without PLE")

                front_started = time.perf_counter()
                # The compact route-id read in ``service_decode_front`` is the
                # required completion boundary for this segment.  Submitting
                # the trace nonblocking avoids an extra host wait before that
                # read while preserving CQ0 ordering.
                ttnn.execute_trace(self.layer.mesh_device, self.front_trace_id, cq_id=0, blocking=False)
                front_seconds = time.perf_counter() - front_started
                service_started = time.perf_counter()
                route_ids, plan = self.layer.service_decode_front(self.front)
                service_seconds = time.perf_counter() - service_started
                service_timing = self.layer._last_host_service_timing
                back_started = time.perf_counter()
                # The following layer's front trace is data-dependent and is
                # submitted on the same command queue.  Its route-id read is
                # the next completion boundary, so blocking here only adds an
                # avoidable host/device round trip.  For the final layer, the
                # terminal/sampling traces and compact token read preserve the
                # same ordering.
                ttnn.execute_trace(self.layer.mesh_device, self.back_trace_id, cq_id=0, blocking=False)
                back_seconds = time.perf_counter() - back_started
            self.last_route_ids = route_ids
            self.last_timing = {
                "ple_seconds": ple_seconds,
                "front_trace_seconds": front_seconds,
                "expert_service_seconds": service_seconds,
                "route_read_and_tt_stall_seconds": float(service_timing["route_read_and_tt_stall_seconds"]),
                "cache_control_dma_submit_seconds": float(service_timing["cache_control_dma_submit_seconds"]),
                "back_trace_seconds": back_seconds,
                "total_seconds": time.perf_counter() - started,
                "expert_hits": len(plan.hits),
                "expert_misses": len(plan.misses),
            }
            return self.output

    def release(self) -> None:
        with self._lock:
            if self.released:
                return
            workspace = getattr(self, "state_workspace", None)
            release_scope = workspace.serialize_replay() if workspace is not None else nullcontext()
            with release_scope:
                # Release consumers before producers.  Clear each handle only
                # after successful teardown so a partial failure is retryable.
                if self.back_trace_id is not None:
                    ttnn.release_trace(self.layer.mesh_device, self.back_trace_id)
                    self.back_trace_id = None
                if self.front_trace_id is not None:
                    ttnn.release_trace(self.layer.mesh_device, self.front_trace_id)
                    self.front_trace_id = None
                # Retain workspace ownership and the layer's active guard until
                # all trace-addressed tensors are gone.  If deallocation fails,
                # release remains retryable under the same serialization lock.
                self._release_front(self.front)
                if self.output.is_allocated():
                    ttnn.deallocate(self.output)
                self.captured_inputs = ()
                if workspace is not None:
                    workspace.release_trace()
                    self.state_workspace = None
                self.layer._host_segmented_trace_active = False
                self.released = True


__all__ = [
    "HostBackedSegmentedDecodeTrace",
    "HostDecodeAttention",
    "HostDecodeFront",
    "MultichipDecodeStateWorkspace",
    "MultichipDecoder",
    "MultichipMemoryPlan",
]

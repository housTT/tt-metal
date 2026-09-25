# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Four-die tensor/expert-parallel Qwen3.8-Flash-Next decoder layer.

The fixed target is a 4x1 Blackhole P300 mesh.  TP4 assigns each rank six QSA
query heads, one quarter of the shared-expert intermediate, and a 640-wide
slice of every hyperconnection stream.  The two attention KV heads are
replicated across rank pairs; the indexer KV head is replicated on all ranks.
Routed experts use contiguous EP4 ownership with 128 complete BFP4 experts per
device and device-only dispatch, compute, combine, and reduction.

Stack-internal residuals remain fractured as ``[1,1,4*M,640]``.  QSA and MoE
use reduce-scatter while the correctness-selected 48-head GDN recurrence stays
replicated and shards only its output projection.  ``from_state_dict`` builds
the optimized local graph for each logical rank on one shared mesh, patches
the primary allocation with ranks 1-3, and releases each temporary allocation.
No activation or expert-weight host round trip occurs in resident execution.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import math
import os
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import functional_decoder as _functional_decoder
from models.autoports.qwen_qwen3_8_flash_next.tt.host_weight_cache import (
    EXPERT_PACKED_BYTES_PER_RANK,
    GLOBAL_INTERMEDIATE,
    PLEDeviceStaging,
    Qwen38ExpertHostSource,
    Qwen38PLEHostStore,
    QwenDeviceExpertCache,
    SafetensorCheckpoint,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.model_config import LINEAR_ATTENTION, PREFILL_CHUNK, PREFILL_CHUNK_BASE
from models.autoports.qwen_qwen3_8_flash_next.tt.model_config import decoder_shapes as _target_decoder_shapes
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.parallel_config import P300_TP4_EP4, Qwen38ParallelConfig
from models.autoports.qwen_qwen3_8_flash_next.tt.resident_experts import (
    Qwen38ResidentExpertSource,
    Qwen38ResidentExperts,
    RESIDENT_EXPERT_BYTES_PER_DEVICE,
)

TP_SIZE = P300_TP4_EP4.dense_tp
TARGET_MESH = P300_TP4_EP4.mesh_shape
# The mesh must be opened with this router payload before constructing a
# decoder; the hardware tests and context contract expose the same setting.
COLLECTIVE_NUM_LINKS = 2
FABRIC_PACKET_BYTES = 8192
RESIDUAL_SHARD_WIDTH = 2560 // TP_SIZE
# Suffix under which ``_install_fractured_residual_weights`` retains the
# replicated full-width copies used by ``decode_forward_replicated``.
REPLICATED_WEIGHT_SUFFIX = "_replicated"
REPLICATED_DECODE_WEIGHT_NAMES = (
    "attn_hc_down_inject",
    "attn_hc_up",
    "mlp_hc_down_inject",
    "mlp_hc_up",
)
REPLICATED_DECODE_NORM_NAMES = ("attn_hc_norm", "mlp_hc_norm")
DECODE_RESIDUAL_LAYOUTS = ("fractured", "replicated")
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
EXPERT_TILES_PER_DEVICE = 29_491_200
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

    @property
    def resident_stack_bytes(self) -> int:
        """Conservative TP4 stack charge including every resident BFP4 expert."""

        return (
            self.runtime_reserve_bytes
            + self.cache_bytes
            + self.non_expert_weight_bytes
            + self.standard_bfp4_expert_bytes
            + self.ple_staging_bytes
            + self.all_runtime_state_bytes
        )

    @property
    def resident_stack_headroom_bytes(self) -> int:
        return self.dram_bytes - self.resident_stack_bytes

    @property
    def resident_stack_fits(self) -> bool:
        return self.resident_stack_headroom_bytes >= 0


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


def _rank_local_config(
    hf_config,
    layer_idx: int | None = None,
    *,
    expert_parallel: bool = False,
    parallel_config: Qwen38ParallelConfig = P300_TP4_EP4,
):
    """Clone the HF config and express one TP4 rank's logical geometry."""

    local = copy.deepcopy(hf_config)
    cfg = local.text_config
    names = ["shared_expert_intermediate_size"]
    if not expert_parallel:
        names.append("moe_intermediate_size")
    is_qsa = layer_idx is None or cfg.layer_types[layer_idx] != LINEAR_ATTENTION
    if is_qsa:
        names.extend(("num_attention_heads", "indexer_n_heads"))
    for name in names:
        value = int(getattr(cfg, name))
        if value % parallel_config.dense_tp:
            raise ValueError(f"{name}={value} is not divisible by TP={parallel_config.dense_tp}")
        setattr(cfg, name, value // parallel_config.dense_tp)
    if is_qsa:
        # Two attention KV heads are each replicated over two ranks.  The
        # indexer's single KV head is replicated over all four ranks.
        if int(cfg.num_key_value_heads) * parallel_config.kv_replication != parallel_config.dense_tp:
            raise ValueError("attention KV replication does not cover every TP rank")
        cfg.num_key_value_heads = 1
        if int(cfg.indexer_kv_heads) != 1 or parallel_config.indexer_kv_replication != parallel_config.dense_tp:
            raise ValueError("indexer KV head must be replicated over all TP ranks")
    return local


@contextmanager
def _rank_local_shape_contract(
    global_config,
    layer_idx: int,
    *,
    expert_parallel: bool = False,
    parallel_config: Qwen38ParallelConfig = P300_TP4_EP4,
):
    """Let the exact-target loader materialize one rank-local TP graph.

    ``FunctionalDecoder`` intentionally validates only checkpoint-global
    shapes.  Multichip setup first performs that validation, then narrows the
    resulting immutable shape record while holding a process-wide lock.  The
    original resolver is restored before setup returns; no runtime method and
    no concurrent model construction can observe the local resolver.
    """

    global_shapes = _target_decoder_shapes(global_config, layer_idx)
    replacements = {
        "shared_expert_intermediate_size": global_shapes.shared_expert_intermediate_size
        // parallel_config.dense_tp,
    }
    if not expert_parallel:
        replacements["moe_intermediate_size"] = global_shapes.moe_intermediate_size // parallel_config.dense_tp
    if global_shapes.layer_type != LINEAR_ATTENTION:
        replacements.update(
            num_attention_heads=global_shapes.num_attention_heads // parallel_config.dense_tp,
            num_key_value_heads=1,
            indexer_n_heads=global_shapes.indexer_n_heads // parallel_config.dense_tp,
            indexer_kv_heads=1,
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


def _rank_local_state(
    state_dict: Mapping | None,
    rank: int,
    *,
    shard_gdn: bool = True,
    expert_parallel: bool = False,
    parallel_config: Qwen38ParallelConfig = P300_TP4_EP4,
):
    """Return setup-only checkpoint views/concats for one TP rank."""

    if state_dict is None:
        return None
    parallel_config._validate_rank(rank)

    state = dict(state_dict)

    # Shared expert: column-parallel gate/up and row-parallel down.
    for name in ("gate_proj", "up_proj"):
        key = f"mlp.shared_expert.{name}.weight"
        if key in state:
            width = GLOBAL_INTERMEDIATE // parallel_config.dense_tp
            state[key] = state[key][rank * width : (rank + 1) * width]
    key = "mlp.shared_expert.down_proj.weight"
    if key in state:
        width = GLOBAL_INTERMEDIATE // parallel_config.dense_tp
        state[key] = state[key][:, rank * width : (rank + 1) * width]

    # Routed expert: preserve packed [gate, up] ordering within each rank.
    key = "mlp.experts.gate_up_proj"
    if key in state and not expert_parallel:
        fused = state[key]
        width = GLOBAL_INTERMEDIATE // parallel_config.dense_tp
        state[key] = torch.cat(
            (
                fused[:, rank * width : (rank + 1) * width],
                fused[:, GLOBAL_INTERMEDIATE + rank * width : GLOBAL_INTERMEDIATE + (rank + 1) * width],
            ),
            dim=1,
        )
    key = "mlp.experts.down_proj"
    if key in state and not expert_parallel:
        width = GLOBAL_INTERMEDIATE // parallel_config.dense_tp
        state[key] = state[key][:, :, rank * width : (rank + 1) * width]

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
        q_start, q_end = parallel_config.q_head_range(rank, 24)
        state[key] = q_gate[q_start:q_end].reshape((q_end - q_start) * 2 * 256, 2560)
    for suffix in ("k_proj.weight", "v_proj.weight"):
        key = f"self_attn.{suffix}"
        if key in state:
            kv_head = parallel_config.kv_head_for_rank(rank, 2)
            state[key] = state[key][kv_head * 256 : (kv_head + 1) * 256]
    key = "self_attn.o_proj.weight"
    if key in state:
        q_start, q_end = parallel_config.q_head_range(rank, 24)
        state[key] = state[key][:, q_start * 256 : q_end * 256]

    key = "self_attn.indexer.index_qk_proj.weight"
    if key in state:
        packed = state[key]
        query_width = 4 * 128
        query_start = rank * 128
        state[key] = torch.cat((packed[query_start : query_start + 128], packed[query_width:]), dim=0)

    return state


def _split_gdn_projection_weights(layer) -> None:
    """Split the packed GDN ``[K, 10240 | 96]`` projection into two weights.

    Decode then emits the 10240-wide fp32 qkv directly instead of slicing it
    out of a 10336-wide tensor (a 90 us op in the decode profile).  Setup-only;
    disabled with ``QWEN38_GDN_SPLIT_QKV=0``.
    """

    if os.environ.get("QWEN38_GDN_SPLIT_QKV", "1") != "1" or "gdn_qkv_b_a" not in layer.w:
        return
    s = layer.shapes
    packed = layer.w["gdn_qkv_b_a"]
    bias = layer.w["gdn_qkv_b_a_bias"]
    qkv_width = s.linear_qkv_width
    ba_width = 2 * s.linear_num_value_heads
    group = layer.weight_group_by_id.get(id(packed))
    role = layer.weight_role_by_id.get(id(packed))

    def _split_last(tensor):
        shape = [int(value) for value in tensor.shape]
        zeros = [0] * len(shape)
        parts = []
        for begins, ends in (
            (zeros, shape[:-1] + [qkv_width]),
            (zeros[:-1] + [qkv_width], shape[:-1] + [qkv_width + ba_width]),
        ):
            part = ttnn.slice(tensor, begins, ends)
            if part.dtype != tensor.dtype:
                # Slicing block-float tensors yields BF16; restore the policy dtype.
                cast = ttnn.typecast(part, tensor.dtype)
                ttnn.deallocate(part)
                part = cast
            parts.append(part)
        return parts[0], parts[1]

    qkv, b_a = _split_last(packed)
    qkv_bias, b_a_bias = _split_last(bias)
    ttnn.synchronize_device(layer.mesh_device)
    layer.w["gdn_qkv"] = qkv
    layer.w["gdn_b_a"] = b_a
    layer.w["gdn_qkv_bias"] = qkv_bias
    layer.w["gdn_b_a_bias"] = b_a_bias
    if group is not None:
        layer.weight_group_by_id[id(qkv)] = group
        layer.weight_group_by_id[id(b_a)] = group
    if role is not None:
        # The qkv half keeps the packed weight's role so its decode program
        # policy still applies; the tiny beta/decay matmul uses the heuristic.
        layer.weight_role_by_id[id(qkv)] = role
    # Drop the registry entries of the released tensors: Python ids are
    # recycled, and a stale id would attribute a later (BF16 bias) tensor to
    # the GDN weight group in the precision-propagation summary.
    for released in (packed, bias):
        layer.weight_group_by_id.pop(id(released), None)
        layer.weight_role_by_id.pop(id(released), None)
    ttnn.deallocate(packed)
    ttnn.deallocate(bias)
    del layer.w["gdn_qkv_b_a"]
    del layer.w["gdn_qkv_b_a_bias"]


def _install_fractured_residual_weights(layer) -> None:
    """Replace replicated HC/GDN outputs with exact within-stream TP4 shards.

    This is setup-only.  The original optimized weights are replicated on all
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
                cluster_axis=layer.parallel_config.collective_axis,
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
                cluster_axis=layer.parallel_config.collective_axis,
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
                cluster_axis=layer.parallel_config.collective_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            created.append(local_up)
            replacements[up_name] = ttnn.reshape(
                local_up,
                (1, 1, layer.shapes.hc_lowrank, layer.shapes.hc_hidden_size // TP_SIZE),
            )
            # Stream-blocked down+inject weight for the fused decode mixer
            # (QWEN38_MIXER_FUSED): [1, 1, 640, 4 * 352], block s = stream s.
            width = layer.shapes.hc_lowrank + layer.shapes.hc_count
            padded_width = 32 * math.ceil(width / 32)
            # pad the column axis first (a pad after the permute would tile-pad the 4-row axis)
            padded_groups = ttnn.pad(local_down, [(0, 0), (0, 0), (0, 0), (0, padded_width - width)], 0.0)
            blocks = ttnn.permute(padded_groups, (0, 2, 1, 3))  # [1, 640, 4, Lp]
            ttnn.deallocate(padded_groups)
            blocked = ttnn.reshape(blocks, (1, 1, RESIDUAL_SHARD_WIDTH, layer.shapes.hc_count * padded_width))
            _functional_decoder._free(blocks, blocked)
            created.append(blocked)
            replacements[f"{prefix}_down_inject_blocked"] = blocked

        if layer.shapes.layer_type == LINEAR_ATTENTION:
            output = layer.w["gdn_out"]
            local_output = ttnn.mesh_partition(
                output,
                dim=-1,
                cluster_axis=layer.parallel_config.collective_axis,
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

    layer.replicated_dram_sharded_weight_by_role = {}
    layer.replicated_dram_activation_config_by_role = {}
    layer.replicated_dram_sharded_cores_by_role = {}
    for name, replacement in replacements.items():
        original = layer.w.get(name)
        if original is None:
            # derived weight without an original (e.g. the stream-blocked down weight)
            layer.w[name] = replacement
            continue
        group = layer.weight_group_by_id.pop(id(original), None)
        role = layer.weight_role_by_id.pop(id(original), None)
        layer.w[name] = replacement
        if group is not None:
            layer.weight_group_by_id[id(replacement)] = group
        if role is not None:
            layer.weight_role_by_id[id(replacement)] = role
        if name in REPLICATED_DECODE_WEIGHT_NAMES and role is not None:
            # Retain the replicated full-width matmul weights for the
            # replicated decode residual path (``decode_forward_replicated``),
            # which runs every hyper mixer locally.  They are stored as
            # DRAM-sharded decode weights: at M=1 the 10240-wide down
            # projection is a pure weight-stream, and the 1D mcast config can
            # only spread its 11 output tiles over 11 cores.  N is padded to
            # the eight-bank shard multiple; consumers slice the logical width.
            if name.endswith("_down_inject"):
                n = int(original.shape[-1])
                padded_n = math.ceil(n / (32 * 8)) * 32 * 8
                if padded_n != n:
                    padding = [(0, 0)] * (len(original.shape) - 1) + [(0, padded_n - n)]
                    padded = ttnn.pad(original, padding, 0.0)
                    ttnn.deallocate(original)
                    original = padded
            dram_value = ttnn.to_memory_config(original, OptimizedDecoder._dram_weight_memory_config(original))
            activation_config, sharded_cores = OptimizedDecoder._dram_activation_memory_config(
                int(original.shape[-2])
            )
            ttnn.deallocate(original)
            layer.w[f"{name}{REPLICATED_WEIGHT_SUFFIX}"] = dram_value
            layer.replicated_dram_sharded_weight_by_role[name] = dram_value
            layer.replicated_dram_activation_config_by_role[name] = activation_config
            layer.replicated_dram_sharded_cores_by_role[name] = sharded_cores
            if group is not None:
                layer.weight_group_by_id[id(dram_value)] = group
            layer.weight_role_by_id[id(dram_value)] = role
        elif name in REPLICATED_DECODE_NORM_NAMES:
            layer.w[f"{name}{REPLICATED_WEIGHT_SUFFIX}"] = original
            if group is not None:
                layer.weight_group_by_id[id(original)] = group
            if role is not None:
                layer.weight_role_by_id[id(original)] = role
        elif original.is_allocated():
            ttnn.deallocate(original)
    ttnn.synchronize_device(layer.mesh_device)


def _patch_rank(target, source, rank: int, tp_size: int, seen: set[tuple[int, int, int]]) -> None:
    """Copy one source logical rank into the target's distributed buffer."""

    if isinstance(target, ttnn.Tensor) and isinstance(source, ttnn.Tensor):
        pair = (id(target), id(source), int(rank))
        if pair in seen:
            return
        seen.add(pair)
        if target.is_allocated() != source.is_allocated():
            raise ValueError("rank-local optimized graphs disagree on live tensor ownership")
        if not target.is_allocated():
            return
        target_shards = ttnn.get_device_tensors(target)
        source_shards = ttnn.get_device_tensors(source)
        if len(target_shards) != tp_size or len(source_shards) != tp_size:
            raise ValueError(f"rank patch requires tensors distributed over exactly {tp_size} devices")
        ttnn.copy(source_shards[rank], target_shards[rank])
        return
    if dataclasses.is_dataclass(target) and dataclasses.is_dataclass(source):
        for field in dataclasses.fields(target):
            _patch_rank(getattr(target, field.name), getattr(source, field.name), rank, tp_size, seen)
        return
    if isinstance(target, dict) and isinstance(source, dict):
        for key in target.keys() & source.keys():
            _patch_rank(target[key], source[key], rank, tp_size, seen)
        return
    if isinstance(target, (tuple, list)) and isinstance(source, (tuple, list)):
        if len(target) != len(source):
            raise ValueError("rank-local tensor containers disagree in length")
        for target_value, source_value in zip(target, source):
            _patch_rank(target_value, source_value, rank, tp_size, seen)


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


@dataclasses.dataclass(frozen=True)
class _VirtualDecodeSlotTensors:
    """Device-only snapshot of one request's model-owned decode state."""

    layers: tuple[tuple[object, ...], ...]
    io: dict[str, object]

    def all_tensors(self):
        for tensors in self.layers:
            yield from tensors
        yield from self.io.values()


class MultichipVirtualDecodeStateBank:
    """DRAM snapshots used to time-multiplex logical users over a B1 trace.

    QSA K/V and indexer tensors are deliberately absent: vLLM owns those
    caches and selects a request by copying its page-table row into the fixed
    physical input.  Expert slots are immutable, model-wide cache entries and
    are absent as well.  Only request-local GDN/PLE state and the trace-bound
    token/position/page/sampler inputs are copied.
    """

    def __init__(self, mesh_device, layers, *, capacity: int, io_tensors: Mapping[str, object]):
        if not 2 <= int(capacity) <= 32:
            raise ValueError("virtual decode-state capacity must be in [2, 32]")
        self.mesh_device = mesh_device
        self.capacity = int(capacity)
        self._layers = tuple(layers)
        self._io_tensors = dict(io_tensors)
        self._lock = threading.RLock()
        self.closed = False
        self.restore_count = 0
        self.commit_count = 0
        self.reset_count = 0
        self.restore_logical_bytes = 0
        self.commit_logical_bytes = 0
        self.restore_submit_seconds = 0.0
        self.commit_submit_seconds = 0.0
        self._layer_templates = tuple(self._decode_state_tensors(layer) for layer in self._layers)
        allocated = []
        try:
            zero_layers = tuple(
                tuple(ttnn.zeros_like(tensor) for tensor in templates) for templates in self._layer_templates
            )
            zero_io = {name: ttnn.zeros_like(tensor) for name, tensor in self._io_tensors.items()}
            self._zero = _VirtualDecodeSlotTensors(layers=zero_layers, io=zero_io)
            allocated.extend(self._zero.all_tensors())
            slots = []
            for _ in range(self.capacity):
                layer_tensors = tuple(
                    tuple(ttnn.zeros_like(tensor) for tensor in templates) for templates in self._layer_templates
                )
                io = {name: ttnn.zeros_like(tensor) for name, tensor in self._io_tensors.items()}
                slot = _VirtualDecodeSlotTensors(layers=layer_tensors, io=io)
                allocated.extend(slot.all_tensors())
                slots.append(slot)
            self._slots = tuple(slots)
        except BaseException:
            for tensor in reversed(allocated):
                if tensor.is_allocated():
                    ttnn.deallocate(tensor)
            raise
        self.logical_bytes_per_slot = sum(self._tensor_logical_bytes(tensor) for tensor in self._slots[0].all_tensors())

    @staticmethod
    def _decode_state_tensors(layer) -> tuple[object, ...]:
        tensors = []
        if layer.shapes.layer_type == LINEAR_ATTENTION:
            tensors.extend((layer.recurrent_state, *layer.fused_conv_state))
        if layer.shapes.has_ple:
            tensors.extend(layer.fused_ple_conv_state)
        return tuple(tensors)

    @staticmethod
    def _tensor_logical_bytes(tensor) -> int:
        elements = math.prod(int(value) for value in tensor.padded_shape)
        widths = {
            ttnn.float32: 4,
            ttnn.int32: 4,
            ttnn.uint32: 4,
            ttnn.bfloat16: 2,
            ttnn.bfloat8_b: 1,
            ttnn.bfloat4_b: 1,
        }
        return elements * widths.get(tensor.dtype, 4)

    def _require_slot(self, slot_id: int) -> int:
        self._require_open()
        slot = int(slot_id)
        if not 0 <= slot < self.capacity:
            raise ValueError(f"virtual slot {slot} outside [0, {self.capacity})")
        return slot

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("virtual decode-state bank is closed")

    @staticmethod
    def _copy_trees(sources, targets) -> None:
        if len(sources) != len(targets):
            raise RuntimeError("virtual decode-state topology changed")
        for source, target in zip(sources, targets):
            if (
                tuple(source.shape) != tuple(target.shape)
                or tuple(source.padded_shape) != tuple(target.padded_shape)
                or source.dtype != target.dtype
                or source.get_layout() != target.get_layout()
            ):
                raise RuntimeError("virtual decode-state tensor contract changed")
            ttnn.copy(source, target)

    def restore_slot(self, slot_id: int) -> None:
        """Restore one snapshot into the trace-bound physical-B1 tensors."""

        with self._lock:
            slot = self._slots[self._require_slot(slot_id)]
            started = time.perf_counter()
            for sources, targets in zip(slot.layers, self._layer_templates):
                self._copy_trees(sources, targets)
            self._copy_trees(tuple(slot.io.values()), tuple(self._io_tensors.values()))
            self.restore_count += 1
            self.restore_logical_bytes += self.logical_bytes_per_slot
            self.restore_submit_seconds += time.perf_counter() - started

    def commit_slot(self, slot_id: int) -> None:
        """Commit physical-B1 tensors to one request's device snapshot."""

        with self._lock:
            slot = self._slots[self._require_slot(slot_id)]
            started = time.perf_counter()
            for sources, targets in zip(self._layer_templates, slot.layers):
                self._copy_trees(sources, targets)
            self._copy_trees(tuple(self._io_tensors.values()), tuple(slot.io.values()))
            self.commit_count += 1
            self.commit_logical_bytes += self.logical_bytes_per_slot
            self.commit_submit_seconds += time.perf_counter() - started

    def reset_slot(self, slot_id: int) -> None:
        """Zero a released slot without allocating while traces are live."""

        with self._lock:
            slot = self._slots[self._require_slot(slot_id)]
            for sources, targets in zip(self._zero.layers, slot.layers):
                self._copy_trees(sources, targets)
            self._copy_trees(tuple(self._zero.io.values()), tuple(slot.io.values()))
            self.reset_count += 1

    def slot_tensor(self, slot_id: int, name: str):
        """Return a stable device tensor for deferred compact-output reads."""

        with self._lock:
            slot = self._slots[self._require_slot(slot_id)]
            try:
                return slot.io[name]
            except KeyError as error:
                raise KeyError(f"unknown virtual decode-state tensor {name!r}") from error

    def metrics(self) -> dict[str, int | float | bool]:
        return {
            "enabled": True,
            "capacity": self.capacity,
            "logical_bytes_per_slot": self.logical_bytes_per_slot,
            "zero_template_logical_bytes": self.logical_bytes_per_slot,
            "allocated_logical_bytes": self.logical_bytes_per_slot * (self.capacity + 1),
            "restores": self.restore_count,
            "commits": self.commit_count,
            "resets": self.reset_count,
            "restore_logical_bytes": self.restore_logical_bytes,
            "commit_logical_bytes": self.commit_logical_bytes,
            "restore_submit_seconds": self.restore_submit_seconds,
            "commit_submit_seconds": self.commit_submit_seconds,
            "closed": self.closed,
        }

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            for slot in (*self._slots, self._zero):
                for tensor in slot.all_tensors():
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
        "p300_4x1_tp4_ep4",
        "replicated_attention_kv_head_pairs",
        "replicated_single_indexer_kv_head",
        "rank_local_paged_kv_cache",
        "replicated_indexer_selection",
        "persistent_within_stream_fractured_residual",
        "shared_batch1_physical_compressed_page_map",
        "qsa_and_moe_output_reduce_scatter",
        "two_link_8192_byte_fabric_payload",
        "gdn_output_column_parallel",
        "distributed_hyperconnection_rmsnorm_and_projections",
        "resident_contiguous_ep4_bfp4_routed_experts",
        "device_only_expert_routing_dispatch_and_combine",
        "exact_mmap_ple_row_lookup",
        "parallel_exact_safetensors_ple_pread",
        "ple_lookup_overlapped_with_ingress_and_layer0",
        "zero_runtime_expert_weight_h2d_and_route_d2h",
        "dram_canonical_gdn_state_shared_l1_workspace",
        "direct_persistent_gdn_tap_trace_commit",
        "two_phase_stack_trace_program_warm",
        "stack_major_128_token_prefill_microchunks",
        "two_segment_resident_full_stack_decode_trace",
        "persistent_resident_expert_tensor_cache",
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
    def from_checkpoint_resident(
        cls,
        snapshot,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        ple_store: Qwen38PLEHostStore | None = None,
        parallel_config: Qwen38ParallelConfig = P300_TP4_EP4,
        key_prefix: str | None = None,
        cache_tag: str | None = None,
        **kwargs,
    ) -> "MultichipDecoder":
        """Build one TP4 layer with all routed experts resident in EP4 DRAM.

        ``key_prefix``/``cache_tag`` load a layer whose tensors mirror a decoder
        layer under another checkpoint prefix (the MTP layer ``mtp.layers.0.``);
        ``layer_idx`` then only selects the shape contract and precision policy.
        """

        checkpoint = snapshot if isinstance(snapshot, SafetensorCheckpoint) else SafetensorCheckpoint(snapshot)
        state = checkpoint.layer_state(layer_idx, include_experts=False, key_prefix=key_prefix)
        source = Qwen38ResidentExpertSource(
            checkpoint, layer_idx, parallel_config=parallel_config, key_prefix=key_prefix, cache_tag=cache_tag
        )
        if key_prefix is None and layer_idx == 1 and ple_store is None:
            ple_store = Qwen38PLEHostStore(checkpoint)
        return cls.from_state_dict(
            state,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            resident_expert_source=source,
            ple_store=ple_store,
            parallel_config=parallel_config,
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
        parallel_config = kwargs.pop("parallel_config", P300_TP4_EP4)
        if not isinstance(parallel_config, Qwen38ParallelConfig):
            raise TypeError("parallel_config must be Qwen38ParallelConfig")
        parallel_config.validate_mesh(mesh_device)

        host_expert_source = kwargs.pop("host_expert_source", None)
        resident_expert_source = kwargs.pop("resident_expert_source", None)
        if host_expert_source is not None and resident_expert_source is not None:
            raise ValueError("host-backed and resident experts are mutually exclusive")
        if host_expert_source is not None and parallel_config != P300_TP4_EP4:
            raise ValueError("the retained host-backed reference is scoped to the selected P300 mesh")
        if resident_expert_source is not None and not isinstance(
            resident_expert_source, Qwen38ResidentExpertSource
        ):
            raise TypeError("resident_expert_source must be Qwen38ResidentExpertSource")
        if resident_expert_source is not None and resident_expert_source.layer_idx != layer_idx:
            raise ValueError("resident expert source layer does not match decoder layer")
        expert_cache_slots = int(kwargs.pop("expert_cache_slots", HOST_EXPERT_SLOTS))
        packed_host_experts = int(kwargs.pop("packed_host_experts", HOST_PACKED_EXPERTS))
        expert_host_packed_dtype = kwargs.pop("expert_host_packed_dtype", "bfp4")
        expert_host_packed_layout = kwargs.pop("expert_host_packed_layout", "tile")
        expert_device_staging_dtype = kwargs.pop("expert_device_staging_dtype", "bfp4")
        expert_device_staging_layout = kwargs.pop("expert_device_staging_layout", "tile")
        ple_staging_dtype = kwargs.pop("ple_staging_dtype", "bf16")
        ple_staging_layout = kwargs.pop("ple_staging_layout", "tile")
        ple_prefill_rows = int(kwargs.pop("ple_prefill_rows", 128))
        resident_weight_cache_path = kwargs.pop("resident_weight_cache_path", None)
        if resident_weight_cache_path is not None and resident_expert_source is None:
            raise ValueError("resident_weight_cache_path requires resident EP4 experts")
        moe_kernel = str(kwargs.pop("moe_kernel", None) or os.environ.get("QWEN38_MOE_KERNEL", "sparse_bank"))
        collective_num_links = int(kwargs.pop("collective_num_links", COLLECTIVE_NUM_LINKS))
        if collective_num_links not in (1, 2):
            raise ValueError("P300 TP4 collective_num_links must be 1 or 2")
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

        expert_parallel = host_expert_source is not None or resident_expert_source is not None
        local_config = _rank_local_config(
            hf_config,
            layer_idx,
            expert_parallel=expert_parallel,
            parallel_config=parallel_config,
        )
        is_qsa = local_config.text_config.layer_types[layer_idx] != LINEAR_ATTENTION
        if decode_state_workspace is not None and (
            not expert_parallel or int(kwargs.get("max_batch", 1)) != 1 or is_qsa
        ):
            raise ValueError("shared decode-state workspace is valid only for batch-one expert-parallel GDN")
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
            # This optimized cache candidate clears QSA PCC and trace gates
            # while preserving device headroom at maximum context.
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
            os.environ.get("QWEN38_EXPERT_POLICY")
            or ("expert_bfp4_lofi_g40b16_d40b5" if expert_parallel else "expert_bfp4_lofi_g10b16_d40b5"),
        )

        with _rank_local_shape_contract(
            hf_config,
            layer_idx,
            expert_parallel=expert_parallel,
            parallel_config=parallel_config,
        ) as local_shapes:
            setup_context = _bounded_host_expert_setup(local_shapes, expert_parallel)
            with setup_context as bounded_expert_shapes:
                rank_zero_state = _rank_local_state(
                    state_dict,
                    0,
                    shard_gdn=shard_gdn,
                    expert_parallel=expert_parallel,
                    parallel_config=parallel_config,
                )
                primary = OptimizedDecoder.from_state_dict(
                    rank_zero_state,
                    hf_config=local_config,
                    layer_idx=layer_idx,
                    mesh_device=mesh_device,
                    **local_kwargs,
                )
                del rank_zero_state
                gc.collect()

                skip = {"weight_group_by_id", "weight_role_by_id", "mesh_device"}
                for rank in range(1, parallel_config.dense_tp):
                    rank_state = _rank_local_state(
                        state_dict,
                        rank,
                        shard_gdn=shard_gdn,
                        expert_parallel=expert_parallel,
                        parallel_config=parallel_config,
                    )
                    temporary = OptimizedDecoder.from_state_dict(
                        rank_state,
                        hf_config=local_config,
                        layer_idx=layer_idx,
                        mesh_device=mesh_device,
                        **local_kwargs,
                    )
                    del rank_state
                    copied: set[tuple[int, int, int]] = set()
                    for name, target in primary.__dict__.items():
                        if name not in skip and name in temporary.__dict__:
                            _patch_rank(target, temporary.__dict__[name], rank, parallel_config.dense_tp, copied)
                    ttnn.synchronize_device(mesh_device)
                    released: set[int] = set()
                    for name, value in temporary.__dict__.items():
                        if name != "mesh_device":
                            _deallocate_tree(value, released)
                    del temporary
                    gc.collect()

        _split_gdn_projection_weights(primary)
        primary.__class__ = cls
        primary.mesh_device = mesh_device
        primary.global_hf_config = hf_config
        primary.local_hf_config = local_config
        primary.parallel_config = parallel_config
        primary.tp_size = parallel_config.dense_tp
        primary.collective_topology = ttnn.Topology.Linear
        primary.collective_axis = parallel_config.collective_axis
        primary.collective_num_links = collective_num_links
        primary.collective_payload_dtype = collective_payload_dtype
        primary.residual_dtype = residual_dtype
        primary.row_parallel_dtype = row_parallel_dtype
        primary.memory_plan = MultichipMemoryPlan()
        primary.host_expert_source = host_expert_source
        primary.host_expert_cache = None
        primary.resident_experts = None
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
        if expert_parallel:
            # Release one-expert setup sentinels and replace them with bounded,
            # fixed-address demand-loaded slots.
            if primary.expert_gate_up is not None and primary.expert_gate_up.is_allocated():
                ttnn.deallocate(primary.expert_gate_up)
            if primary.experts.down is not None and primary.experts.down.is_allocated():
                ttnn.deallocate(primary.experts.down)
            primary.expert_gate_up = None
            primary.experts = dataclasses.replace(primary.experts, gate=None, up=None, down=None)
            if host_expert_source is not None:
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
            else:
                primary.resident_experts = Qwen38ResidentExperts(
                    mesh_device,
                    resident_expert_source,
                    max_batch=primary.max_batch,
                    prefill_rows=ple_prefill_rows,
                    num_links=collective_num_links,
                    topology=primary.collective_topology,
                    weight_cache_path=resident_weight_cache_path,
                    kernel=moe_kernel,
                )
        if ple_store is not None:
            primary.ple_staging = PLEDeviceStaging(
                mesh_device,
                max_batch=primary.max_batch,
                prefill_rows=ple_prefill_rows,
                dtype=ple_staging_dtype,
                layout=ple_staging_layout,
            )
        if expert_parallel and primary.max_batch == 1 and not is_qsa:
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
        chunk_start: int = 0,
        reset_state: bool = True,
        page_table=None,
        page_tables_per_chunk=None,
        rot_mats=None,
        ple_embeddings=None,
    ):
        """Layer-major compatibility wrapper over fixed-size microchunks.

        Full-model prefill uses :meth:`prefill_microchunk_forward_fractured`
        directly so one padded 128-token buffer crosses the complete decoder
        stack.  Keeping this wrapper provides a reference path for standalone
        layer tests and schedule A/B comparisons.
        """

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

        pieces = []
        for chunk_index, (start, logical, padded) in enumerate(plan):
            x = self._slice_fractured_seq(hidden_states, start, logical, padded)
            if s.has_ple:
                embedding_chunk = _functional_decoder._slice_seq(ple_embeddings, start, logical, padded)
            else:
                embedding_chunk = None
            out = self.prefill_microchunk_forward_fractured(
                x,
                logical=logical,
                user_id=user_id,
                chunk_start=chunk_start + start,
                reset_state=reset_state and chunk_index == 0,
                page_table=page_table,
                chunk_page_table=None
                if s.layer_type == LINEAR_ATTENTION
                else page_tables_per_chunk[chunk_index],
                rot_mats=rot_mats,
                ple_embeddings=embedding_chunk,
            )
            _functional_decoder._free(embedding_chunk, ple_embeddings)
            _functional_decoder._free(x, hidden_states, out)
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

    def prefill_microchunk_forward_fractured(
        self,
        hidden_states,
        *,
        logical: int,
        user_id: int = 0,
        chunk_start: int = 0,
        reset_state: bool = False,
        page_table=None,
        chunk_page_table=None,
        rot_mats=None,
        ple_embeddings=None,
    ):
        """Run one already-padded prefill microchunk without slicing or trim.

        The returned tensor retains its physical row count.  This is the key
        stack-major ABI: padding is materialized once before layer zero and is
        removed once after the final layer, while ``logical`` masks recurrent,
        paged-attention, and routed-expert state updates on the short tail.
        """

        original_hidden_states = hidden_states
        hidden_states = self._residual_compute_input(hidden_states)
        s = self.shapes
        shape = _functional_decoder._shape(hidden_states)
        if len(shape) != 4 or shape[:2] != [1, 1] or shape[-1] != RESIDUAL_SHARD_WIDTH:
            raise ValueError(
                f"fractured prefill microchunk expects [1, 1, 4*physical, {RESIDUAL_SHARD_WIDTH}], got {shape}"
            )
        if shape[-2] % s.hc_count:
            raise ValueError("fractured prefill microchunk rows must be divisible by four streams")
        physical = shape[-2] // s.hc_count
        if not 1 <= int(logical) <= physical:
            raise ValueError(f"logical prefill rows must be in [1, {physical}], got {logical}")
        if physical % PREFILL_CHUNK_BASE or not PREFILL_CHUNK_BASE <= physical <= PREFILL_CHUNK:
            # The adaptive plan mixes PREFILL_CHUNK-row and 128-row microchunks.
            raise ValueError(
                f"prefill microchunk physical length must be a multiple of {PREFILL_CHUNK_BASE} "
                f"in [{PREFILL_CHUNK_BASE}, {PREFILL_CHUNK}], got {physical}"
            )
        if not 0 <= user_id < self.max_batch:
            raise ValueError(f"user_id {user_id} outside [0, {self.max_batch})")
        if s.has_ple:
            expected_ple = [1, 1, physical, s.ple_embed_dim]
            if ple_embeddings is None or _functional_decoder._shape(ple_embeddings) != expected_ple:
                raise ValueError(f"PLE microchunk needs embeddings {expected_ple}")
        elif ple_embeddings is not None:
            raise ValueError("ple_embeddings were passed to a layer without PLE")
        if s.layer_type != LINEAR_ATTENTION and (
            page_table is None or chunk_page_table is None or rot_mats is None
        ):
            raise ValueError("QSA prefill microchunk requires page_table, chunk_page_table and full RoPE tables")

        self._decode_active = False
        if reset_state:
            self._reset_user_state(user_id)
        x = hidden_states
        if s.has_ple:
            updated = self._ple_prefill_fractured(x, ple_embeddings, user_id=user_id, logical=logical)
            _functional_decoder._free(x, original_hidden_states, updated)
            x = updated

        mixed, hyper, injection = self._hyper_mix(x, "attn_hc")
        if s.layer_type == LINEAR_ATTENTION:
            block = self._gdn_prefill(mixed, user_id=user_id, logical=logical)
        else:
            block = self._qsa_prefill(
                mixed,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start=chunk_start,
                rot_mats=rot_mats,
            )
        ttnn.deallocate(mixed)
        hidden = self._hyper_inject(hyper, block, injection)
        mixed, hyper, injection = self._hyper_mix(hidden, "mlp_hc")
        self._host_logical_route_rows = int(logical)
        try:
            block = self._moe(mixed)
        finally:
            self._host_logical_route_rows = None
        ttnn.deallocate(mixed)
        output = self._hyper_inject(hyper, block, injection)
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
        """Run decode with the stack-internal ``[1,1,4*batch,640]`` ABI."""

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

    # ------------------------------------------- replicated decode residual

    # Stack-selected inter-layer residual layout for decode.  ``fractured`` is
    # the ``[1,1,4*B,640]`` width-sharded ABI; ``replicated`` keeps the full
    # ``[1,1,B,10240]`` residual on every rank so both hyper-connection mixers
    # run without collectives and each block needs exactly one all-reduce.
    decode_residual_layout = "fractured"
    _replicated_decode_active = False

    def decode_step(self, hidden_states, **kwargs):
        """Run one decode token in the residual layout selected by the stack."""

        if self.decode_residual_layout == "replicated":
            return self.decode_forward_replicated(hidden_states, **kwargs)
        return self.decode_forward_fractured(hidden_states, **kwargs)

    @contextmanager
    def _replicated_decode_scope(self):
        """Expose the retained replicated weights under their canonical names."""

        if self._replicated_decode_active:
            yield
            return
        swapped = []
        saved_tables = (
            self.dram_sharded_roles,
            self.dram_sharded_weight_by_role,
            self.dram_activation_config_by_role,
            self.dram_sharded_cores_by_role,
        )
        try:
            for name in REPLICATED_DECODE_WEIGHT_NAMES + REPLICATED_DECODE_NORM_NAMES:
                retained = f"{name}{REPLICATED_WEIGHT_SUFFIX}"
                if retained in self.w:
                    self.w[name], self.w[retained] = self.w[retained], self.w[name]
                    swapped.append(name)
            # Only the retained full-width hyper weights are DRAM-sharded in
            # this scope; the fractured tables are restored on exit.
            self.dram_sharded_roles = frozenset(self.replicated_dram_sharded_weight_by_role)
            self.dram_sharded_weight_by_role = self.replicated_dram_sharded_weight_by_role
            self.dram_activation_config_by_role = self.replicated_dram_activation_config_by_role
            self.dram_sharded_cores_by_role = self.replicated_dram_sharded_cores_by_role
            self._replicated_decode_active = True
            yield
        finally:
            self._replicated_decode_active = False
            (
                self.dram_sharded_roles,
                self.dram_sharded_weight_by_role,
                self.dram_activation_config_by_role,
                self.dram_sharded_cores_by_role,
            ) = saved_tables
            for name in swapped:
                retained = f"{name}{REPLICATED_WEIGHT_SUFFIX}"
                self.w[name], self.w[retained] = self.w[retained], self.w[name]

    def decode_forward_replicated(self, hidden_states, **kwargs):
        """Run decode on a replicated ``[1,1,B,10240]`` residual.

        Both hyper-connection mixers execute locally on the full four-stream
        residual (no stats all-gather, no all-reduce, no output all-gather).
        GDN output projection is replicated; QSA and MoE blocks are summed by a
        single all-reduce each, so a GDN layer costs one collective and a QSA
        layer two, against eight and nine on the fractured ABI.
        """

        if self.residual_dtype != "bf16":
            raise ValueError("replicated decode residual requires the BF16 residual policy")
        if self.max_batch != 1:
            raise ValueError("replicated decode residual is validated for batch one only")
        if not self.fractured_residual:
            return super().decode_forward(hidden_states, **kwargs)
        with self._replicated_decode_scope():
            return super().decode_forward(hidden_states, **kwargs)

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
            raise RuntimeError(f"compact route tensor is not replicated over TP{TP_SIZE}")
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
        sparsity = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
        sparsity = ttnn.reshape(sparsity, (1, 1, groups, 1))
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
        sparsity = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
        # Untilize before folding the group axis.  The equivalent TILE reshape
        # maps four padded input pages into one output page and can deadlock on
        # a cached repeat.  In ROW_MAJOR both shapes have the same [groups, E]
        # physical footprint, so this reshape is a metadata-only view.
        sparsity = ttnn.reshape(sparsity, (1, 1, groups, len(expert_ids)))
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
        if getattr(self, "resident_experts", None) is not None:
            raise RuntimeError("resident EP4 is invoked by the fused MoE path, not dense routing")
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
        """Sum a row-parallel block and retain its within-hidden TP4 shard."""

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
        """One-time stack ingress: R ``[1,1,M,10240]`` -> S ``[1,1,4M,640]``."""

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

        if not self.fractured_residual or self._replicated_decode_active:
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
        # Op folding (QWEN38_MIXER_FOLD): produce the matmul inputs in L1 so the
        # decode 1D matmuls skip their input copies, and average the four
        # streams with one matmul instead of pad+reduce.
        fold = self._decode_active and os.environ.get("QWEN38_MIXER_FOLD", "1") == "1"
        weighted = ttnn.multiply(
            normed, weight_rows, memory_config=ttnn.L1_MEMORY_CONFIG if fold else ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(normed)
        _functional_decoder._free(weight_rows, norm_weight, weighted)

        if fold and rows == 1 and self.mixer_fused_kernels:
            # QWEN38_MIXER_FUSED: one matmul of the four stream rows against the
            # stream-blocked down weight (block s = stream s; no [4,640]->[1,2560]
            # relayout), one all-gather of the [4, 4*352] partial rows instead of
            # the 3-op all-reduce, and one kernel that picks row r*4+s of block s,
            # sums over ranks, and runs silu -> up matmul -> sigmoid x weighted ->
            # stream mean plus the injection gate row.
            gather_mode = os.environ.get("QWEN38_MIXER_FUSED_GATHER", "0")
            if gather_mode == "0":
                # v1: the composite all-reduce of the flat [1,1,1,L+S] partial row.
                flat = ttnn.reshape(weighted, (1, 1, rows, s.hc_hidden_size // TP_SIZE))
                packed_partial = self._linear_impl(flat, self.w[f"{prefix}_down_inject"], dtype=ttnn.bfloat16)
                _functional_decoder._free(flat, weighted, packed_partial)
                gathered = self._all_reduce_block(packed_partial)
            else:
                # v2: one matmul against the stream-blocked down weight and one
                # tile-aligned all-gather ("0": along the batch dim, "2": along rows);
                # the kernel picks row r*4+s of block s and sums the ranks.
                blocked = self._linear_impl(weighted, self._down_inject_blocked(prefix), dtype=ttnn.bfloat16)
                gathered = ttnn.all_gather(
                    blocked,
                    dim=0 if gather_mode == "1" else 2,
                    cluster_axis=self.collective_axis,
                    num_links=self.collective_num_links,
                    topology=self.collective_topology,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(blocked)
            local_mixed, injection = ttnn.experimental.kda.hc_mix_post(
                gathered,
                weighted,
                self.w[f"{prefix}_up"],
                s.hc_lowrank,
                s.hc_count,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(gathered)
            ttnn.deallocate(weighted)
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
        flat = ttnn.reshape(weighted, (1, 1, rows, s.hc_hidden_size // TP_SIZE))
        packed_partial = self._linear_impl(flat, self.w[f"{prefix}_down_inject"], dtype=ttnn.bfloat16)
        _functional_decoder._free(flat, weighted, packed_partial)
        packed = self._all_reduce_block(packed_partial)
        low = self._slice_last(packed, 0, s.hc_lowrank)
        injection = self._slice_last(packed, s.hc_lowrank, s.hc_lowrank + s.hc_count)
        ttnn.deallocate(packed)
        low = ttnn.silu(low, memory_config=ttnn.L1_MEMORY_CONFIG if fold else ttnn.DRAM_MEMORY_CONFIG)
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
        if fold and rows == 1:
            # mean over the four streams == [0.25 x4] @ [4, 640]: one matmul.
            averaged = ttnn.matmul(
                self._stream_average_row(),
                ttnn.reshape(local_mixed, (1, 1, s.hc_count, RESIDUAL_SHARD_WIDTH)),
                dtype=ttnn.bfloat16,
                compute_kernel_config=_functional_decoder._hifi4(fp32=True),
            )
            _functional_decoder._free(local_mixed, averaged)
            local_mixed = averaged
        else:
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

    def _stream_average_row(self):
        """Cached ``[1,1,1,4]`` bf16 row of 0.25 for the stream mean matmul."""

        cached = self.__dict__.get("_stream_average")
        if cached is None:
            cached = ttnn.from_torch(
                torch.full((1, 1, 1, self.shapes.hc_count), 1.0 / self.shapes.hc_count, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            self._stream_average = cached
        return cached

    def _down_inject_blocked(self, prefix: str):
        """``[1,1,640,4*352]`` copy of the rank's down+inject weight, one 352-wide block per stream.

        Built once from the ``[1,1,2560,324]`` partition (rows ``s*640..`` are
        stream ``s``): block ``s`` holds stream ``s``'s ``[640, 324]`` slice in
        columns ``s*352 .. s*352+323`` (zero padded to the tile), so one matmul
        of the ``[1,1,4,640]`` weighted residual yields every stream's partial
        row in row ``s`` of block ``s``.
        """

        prepared = self.w.get(f"{prefix}_down_inject_blocked")
        if prepared is not None and prepared.is_allocated():
            return prepared
        cache = self.__dict__.setdefault("_down_inject_blocked_cache", {})
        cached = cache.get(prefix)
        if cached is not None and cached.is_allocated():
            return cached
        s = self.shapes
        width = s.hc_lowrank + s.hc_count
        padded = 32 * math.ceil(width / 32)
        down = self.w[f"{prefix}_down_inject"]
        groups = ttnn.reshape(down, (1, s.hc_count, RESIDUAL_SHARD_WIDTH, width))
        padded_groups = ttnn.pad(groups, [(0, 0), (0, 0), (0, 0), (0, padded - width)], 0.0)
        _functional_decoder._free(groups, down, padded_groups)
        blocks = ttnn.permute(padded_groups, (0, 2, 1, 3))  # [1, 640, 4, Lp]
        ttnn.deallocate(padded_groups)
        blocked = ttnn.reshape(blocks, (1, 1, RESIDUAL_SHARD_WIDTH, s.hc_count * padded))
        _functional_decoder._free(blocks, blocked)
        cache[prefix] = blocked
        return blocked

    @property
    def mixer_fused_kernels(self) -> bool:
        """Use the fused ``hc_mix_post`` / ``hc_inject`` decode kernels."""

        return os.environ.get("QWEN38_MIXER_FUSED", "1") == "1"

    def _hyper_inject(self, hyper_input, block_output, injection):
        """Inject one local 640-wide block shard into all four local streams."""

        if not self.fractured_residual or self._replicated_decode_active:
            return super()._hyper_inject(hyper_input, block_output, injection)
        s = self.shapes
        rows = int(block_output.shape[-2])
        if (
            rows == 1
            and self._decode_active
            and self.mixer_fused_kernels
            and block_output.dtype == ttnn.bfloat16
            and hyper_input.dtype == ttnn.bfloat16
            and injection.dtype == ttnn.bfloat16
            and _functional_decoder._shape(injection) == [1, 1, 1, s.hc_count]
        ):
            output = ttnn.experimental.kda.hc_inject(
                hyper_input, block_output, injection, s.hc_count, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(block_output)
            ttnn.deallocate(injection)
            return output
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
        """Reproduce the mixed projection as the persistent tap's final writer.

        The fused decode body already copies the projection into the newest
        tap (and the shared workspace copies it back to the canonical DRAM
        state), so this second projection is redundant.  It is kept behind
        ``QWEN38_GDN_RECOMMIT_TAP=1`` for A/B evidence only.
        """

        if os.environ.get("QWEN38_GDN_RECOMMIT_TAP", "0") != "1":
            return
        s = self.shapes
        if "gdn_qkv" in self.w:
            packed = self._linear_impl(x, self.w["gdn_qkv"], bias=self.w["gdn_qkv_bias"], dtype=ttnn.float32)
        else:
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
        # The fused decode step streams each head's state from DRAM inside one
        # program, so the L1 workspace hydration (two 3 MB state copies and six
        # tap copies per layer) buys nothing; skip it (QWEN38_GDN_WORKSPACE=1
        # restores the binding).
        if workspace is not None and self.gdn_fused_step and os.environ.get("QWEN38_GDN_WORKSPACE", "0") != "1":
            workspace = None
        if workspace is None:
            output = super()._gdn_decode(args[0])
            self._commit_newest_gdn_state_direct(args[0])
        else:
            with workspace.bind_gdn(self):
                output = super()._gdn_decode(args[0])
                self._commit_newest_gdn_state_direct(args[0])
        if self._replicated_decode_active and int(output.shape[-1]) == RESIDUAL_SHARD_WIDTH:
            # ``gdn_out`` stays column-parallel (a quarter of the 15.7 MB
            # weight per rank); one all-gather replicates the block output.
            gathered = ttnn.all_gather(
                output,
                dim=3,
                cluster_axis=self.collective_axis,
                num_links=self.collective_num_links,
                topology=self.collective_topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(output)
            output = gathered
        return output

    def _qsa_prefill(self, *args, **kwargs):
        partial = super()._qsa_prefill(*args, **kwargs)
        return self._reduce_scatter_block(partial) if self.fractured_residual else self._all_reduce_block(partial)

    def _qsa_decode(self, *args, **kwargs):
        partial = super()._qsa_decode(*args, **kwargs)
        if self.fractured_residual and not self._replicated_decode_active:
            return self._reduce_scatter_block(partial)
        return self._all_reduce_block(partial)

    def _moe(self, *args, **kwargs):
        if self.resident_experts is not None:
            if kwargs or len(args) != 1:
                raise TypeError("resident EP4 MoE expects one input tensor")
            return self._moe_resident(args[0])
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

    def _moe_resident(self, x):
        """Run router, resident EP4 experts, and shared TP4 expert on device."""

        s = self.shapes
        input_rows = int(x.shape[-2])
        logical = int(self._host_logical_route_rows or input_rows)
        if not 1 <= logical <= input_rows:
            raise ValueError(f"logical resident MoE rows must be in [1, {input_rows}], got {logical}")
        padded = 32 * math.ceil(input_rows / 32)
        fused_rows = (
            input_rows == 1
            and self.max_batch == 1
            and self.moe_fused_kernels
            and self.resident_experts.kernel == "sparse_bank"
            and os.environ.get("QWEN38_MOE_FUSED_NOPAD", "1") == "1"
        )
        if fused_rows:
            # The fused batch-one path works on the single logical row (every
            # kernel/matmul reads whole tiles anyway), so skip the zero-fill
            # of the padding rows and the trailing trim.
            padded = input_rows
        work = _functional_decoder._pad_seq(x, padded, x) if padded != input_rows else x
        if work.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            # The fabric dispatch and routed expert FFN require a DRAM
            # interleaved activation.  The fractured ABI delivered one via its
            # all-gather; the replicated mixer output may still live in L1.
            staged = ttnn.to_memory_config(work, ttnn.DRAM_MEMORY_CONFIG)
            _functional_decoder._free(work, x, staged)
            work = staged

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

        replicated = self._replicated_decode_active
        sparse_bank = self.resident_experts.kernel == "sparse_bank"
        indices = scores = None
        if sparse_bank and padded in (1, 32) and self.max_batch == 1 and self.moe_fused_kernels:
            # Decode (QWEN38_MOE_FUSED): top-k, softmax and the rank-local slot
            # mapping in one kernel, the ten experts in indexed mode, and the
            # weighted expert sum in one kernel.
            routed = self._routed_experts_sparse_bank_fused(work, logits)
        else:
            selected, indices = ttnn.topk(logits, k=s.num_experts_per_tok, dim=-1, sorted=True)
            scores = ttnn.softmax(selected, dim=-1)
            _functional_decoder._free(selected, scores)
        if indices is None:
            pass
        elif sparse_bank and padded == 32 and self.max_batch == 1:
            # Decode: visit only the ten selected experts (indexed mode).
            ttnn.deallocate(logits)
            routed = self._routed_experts_sparse_bank_indexed(work, indices, scores)
        elif sparse_bank and padded == 32 and self.max_batch * s.num_experts_per_tok <= self.resident_experts.experts_per_device:
            # Small-batch decode (speculative pair): one bank slot per (row,
            # expert); the dense scan would also route the zero padding rows
            # and read almost every local expert.
            ttnn.deallocate(logits)
            routed = self._routed_experts_sparse_bank_indexed_rows(work, indices, scores, rows=self.max_batch)
        elif sparse_bank and self.moe_prefill_slabs and padded > PREFILL_CHUNK_BASE:
            # Prefill (QWEN38_MOE_PREFILL_SLABS): sort the rank's (row, expert)
            # hits into 32-row slabs so every active expert streams its weights
            # once against its own rows instead of once per 32-row group.
            # 128-row microchunks (short prompts and adaptive tails) keep the
            # dense scan: slabs gain nothing there and would reorder the sum.
            ttnn.deallocate(logits)
            routed = self._routed_experts_sparse_bank_slabs(work, indices, scores)
        elif sparse_bank:
            # Dense routing weights over all 512 experts; each rank keeps its
            # own 128-wide slice.  Rows past ``logical`` carry zero weights.
            zeros = ttnn.zeros_like(logits)
            routing = ttnn.scatter(zeros, dim=-1, index=indices, src=scores)
            ttnn.deallocate(zeros)
            ttnn.deallocate(logits)
            routed = self._routed_experts_sparse_bank(work, routing, logical_rows=logical)
            ttnn.deallocate(routing)
        else:
            ttnn.deallocate(logits)
            routed = self.resident_experts(
                work,
                indices,
                scores,
                logical_rows=logical,
                reduce_scatter=not replicated,
            )

        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        gated = ttnn.multiply(hidden, scalar, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(hidden)
        ttnn.deallocate(scalar)
        shared_partial = self._linear(gated, self.w["shared_down_proj"])
        ttnn.deallocate(gated)
        if replicated or sparse_bank:
            # Routed and shared outputs are both per-rank partial sums of the
            # full hidden width, so one collective finishes the block: an
            # all-reduce for the replicated residual, a reduce-scatter for the
            # fractured one.
            summed = ttnn.add(routed, shared_partial)
            ttnn.deallocate(routed)
            ttnn.deallocate(shared_partial)
            if replicated:
                out = self._all_reduce_block(summed)
                out_width = s.hidden_size
            else:
                out = self._reduce_scatter_block(summed)
                out_width = RESIDUAL_SHARD_WIDTH
        else:
            shared = self._reduce_scatter_block(shared_partial)
            out = ttnn.add(routed, shared)
            ttnn.deallocate(routed)
            ttnn.deallocate(shared)
            out_width = RESIDUAL_SHARD_WIDTH
        if indices is not None:
            ttnn.deallocate(indices)
            ttnn.deallocate(scores)
        if padded != input_rows:
            trimmed = ttnn.slice(out, [0, 0, 0, 0], [1, 1, input_rows, out_width])
            _functional_decoder._free(out, trimmed)
            out = trimmed
        return out

    @property
    def moe_prefill_slabs(self) -> bool:
        return os.environ.get("QWEN38_MOE_PREFILL_SLABS", "0") == "1"

    def _routed_experts_sparse_bank_slabs(self, x, indices, scores):
        """Prefill MoE over expert-sorted 32-row slabs (indexed sparse matmuls).

        ``moe_sort_slabs`` groups this rank's routing hits by expert into
        slabs; the slab rows are gathered from ``x`` with one embedding, both
        bank matmuls run in indexed mode with one bank slot per slab, and the
        weighted un-sort is a scatter of the routing weights into a
        ``[rows, P*32]`` one-hot followed by one matmul.
        """

        s = self.shapes
        bank = self.resident_experts
        k = s.num_experts_per_tok
        rows = int(x.shape[-2])
        local = bank.experts_per_device
        capacity = local + math.ceil(rows * k / 32) + 1
        slab_rows, slab_experts, slab_pos, local_scores = ttnn.experimental.kda.moe_sort_slabs(
            indices, scores, bank.rank_base, local, capacity, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        table = ttnn.reshape(x_rm, (rows, s.hidden_size))
        ids = ttnn.reshape(slab_rows, (1, capacity * 32))
        gathered = ttnn.embedding(ids, table, layout=ttnn.TILE_LAYOUT)
        _functional_decoder._free(table, x_rm, gathered)
        _functional_decoder._free(x_rm, x, gathered)
        _functional_decoder._free(ids, slab_rows, gathered)
        ttnn.deallocate(slab_rows)
        slabs = ttnn.reshape(gathered, (1, capacity, 32, s.hidden_size))
        _functional_decoder._free(gathered, slabs)
        output_tile = ttnn.Tile([32, 32])
        if os.environ.get("QWEN38_DEBUG_SLABS") == "1" and not self.__dict__.get("_slabs_reported"):
            self._slabs_reported = True
            print({"moe_prefill_slabs": {"rows": rows, "capacity": capacity, "local_experts": local}})
        # the indexed sparse matmul iterates at most `local` (= B's group count) ids per call
        down_chunks = []
        for start in range(0, capacity, local):
            count = min(local, capacity - start)
            if count == capacity:
                a_chunk, idx_chunk = slabs, slab_experts
            else:
                a_chunk = ttnn.slice(slabs, [0, start, 0, 0], [1, start + count, 32, s.hidden_size])
                idx_chunk = ttnn.slice(slab_experts, [0, 0, 0, start], [1, 1, 1, start + count])
            gate_up = ttnn.sparse_matmul(
                a_chunk,
                bank.gate_up_bank,
                sparsity=bank.indexed_sparsity,
                indices=idx_chunk,
                nnz=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                program_config=self._sparse_matmul_config(32, 2 * s.moe_intermediate_size, s.hidden_size),
                compute_kernel_config=self.expert_compute_cfg,
                dtype=ttnn.bfloat16,
            )
            if os.environ.get("QWEN38_MOE_FUSED_SWIGLU", "1") == "1":
                hidden = ttnn.experimental.kda.moe_swiglu(gate_up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(gate_up)
            else:
                gate = self._slice_last(gate_up, 0, s.moe_intermediate_size)
                up = self._slice_last(gate_up, s.moe_intermediate_size, 2 * s.moe_intermediate_size)
                ttnn.deallocate(gate_up)
                hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
                ttnn.deallocate(gate)
                ttnn.deallocate(up)
            down_chunk = ttnn.sparse_matmul(
                hidden,
                bank.down_bank,
                sparsity=bank.indexed_sparsity,
                indices=idx_chunk,
                nnz=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                program_config=self._sparse_matmul_config(32, s.hidden_size, s.moe_intermediate_size),
                compute_kernel_config=self.expert_compute_cfg,
                dtype=ttnn.bfloat16,
            )
            ttnn.deallocate(hidden)
            if a_chunk is not slabs:
                ttnn.deallocate(a_chunk)
                ttnn.deallocate(idx_chunk)
            down_chunks.append(down_chunk)
        ttnn.deallocate(slabs)
        ttnn.deallocate(slab_experts)
        if len(down_chunks) == 1:
            down = down_chunks[0]
        else:
            down = ttnn.concat(down_chunks, dim=1)
            for chunk in down_chunks:
                ttnn.deallocate(chunk)
        zeros = ttnn.zeros(
            (1, 1, rows, capacity * 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
        )
        unsort = ttnn.scatter(zeros, dim=-1, index=slab_pos, src=local_scores)
        ttnn.deallocate(zeros)
        ttnn.deallocate(slab_pos)
        ttnn.deallocate(local_scores)
        flat_down = ttnn.reshape(down, (1, 1, capacity * 32, s.hidden_size))
        out = ttnn.matmul(
            unsort,
            flat_down,
            dtype=ttnn.bfloat16,
            compute_kernel_config=_functional_decoder._hifi4(fp32=True),
        )
        _functional_decoder._free(flat_down, down, out)
        ttnn.deallocate(down)
        ttnn.deallocate(unsort)
        return out

    def _routed_experts_sparse_bank(self, x, routing, *, logical_rows: int):
        """Evaluate this rank's resident experts with sparse matmuls.

        ``x`` is the replicated, tile-padded ``[1,1,rows,2560]`` block input and
        ``routing`` the dense ``[1,1,rows,512]`` top-k weights.  The rank keeps
        its 128-wide routing slice; ``sparse_matmul`` skips (group, expert)
        pairs whose weight is zero and zero-fills their output, so the reduce
        over experts is the exact local partial of the routed MoE output.
        """

        s = self.shapes
        bank = self.resident_experts
        tokens = int(x.shape[-2])
        if tokens % 32:
            raise ValueError(f"sparse bank expert input must be tile padded, got {tokens}")
        groups = tokens // 32
        local_experts = bank.experts_per_device
        local_routing = ttnn.mesh_partition(
            routing,
            dim=3,
            cluster_axis=self.collective_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        grouped_x = ttnn.reshape(x, (1, groups, 32, s.hidden_size))
        routing_groups = ttnn.reshape(local_routing, (1, groups, 32, local_experts))
        sparsity = ttnn.max(routing_groups, dim=2, keepdim=True)
        sparsity = ttnn.reshape(sparsity, (1, 1, groups, local_experts))
        sparsity = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([32, 32])
        gate_up_sparse = ttnn.sparse_matmul(
            grouped_x,
            bank.gate_up_bank,
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=self._sparse_matmul_config(32, 2 * s.moe_intermediate_size, s.hidden_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _functional_decoder._free(grouped_x, x, gate_up_sparse)
        gate_up = ttnn.reshape(gate_up_sparse, (groups, local_experts, 32, 2 * s.moe_intermediate_size))
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
        _functional_decoder._free(routing_groups, local_routing, weighted)
        ttnn.deallocate(local_routing)
        down = ttnn.sparse_matmul(
            weighted,
            bank.down_bank,
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            program_config=self._sparse_matmul_config(32, s.hidden_size, s.moe_intermediate_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _functional_decoder._free(weighted, down)
        ttnn.deallocate(sparsity)
        if os.environ.get("QWEN38_MOE_SCAN_REDUCE", "fast") == "sum":
            out = ttnn.sum(down, dim=1, keepdim=True)
        else:
            out = ttnn.experimental.fast_reduce_nc(down, dims=[1])
        ttnn.deallocate(down)
        return ttnn.reshape(ttnn.unsqueeze_to_4D(out), (1, 1, tokens, s.hidden_size))

    def _row_onehot(self, row: int):
        """Cached ``[1,1,32,1]`` bf16 mask selecting one padded decode row."""

        cache = self.__dict__.setdefault("_row_onehots", {})
        if row not in cache:
            host = torch.zeros(1, 1, 32, 1, dtype=torch.bfloat16)
            host[0, 0, row, 0] = 1.0
            cache[row] = ttnn.from_torch(
                host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        return cache[row]

    def _routed_experts_sparse_bank_indexed_rows(self, x, indices, scores, *, rows: int):
        """Indexed-mode decode MoE for ``rows`` real tokens in one 32-row tile.

        Slot ``r*k + j`` evaluates row ``r``'s ``j``-th expert for all 32 rows;
        the per-slot weight column keeps only row ``r``'s score, so the reduce
        over slots is each row's exact routed output.  Costs ``rows * k``
        expert evaluations instead of the dense scan's near-full bank.
        """

        s = self.shapes
        bank = self.resident_experts
        k = s.num_experts_per_tok
        local_experts = bank.experts_per_device
        slots = rows * k
        ids = ttnn.typecast(indices, ttnn.int32)
        rel = ttnn.subtract(ids, bank.rank_base)
        ttnn.deallocate(ids)
        rel_f = ttnn.typecast(rel, ttnn.float32)
        ttnn.deallocate(rel)
        valid = ttnn.multiply(ttnn.ge(rel_f, 0.0), ttnn.lt(rel_f, float(local_experts)))
        local_f = ttnn.multiply(rel_f, valid)
        ttnn.deallocate(rel_f)
        local_i32 = ttnn.typecast(local_f, ttnn.int32)
        ttnn.deallocate(local_f)
        index_rows = ttnn.slice(local_i32, [0, 0, 0, 0], [1, 1, rows, k])
        ttnn.deallocate(local_i32)
        # Reshape while still int32 (reshape_view has no uint16 support), then
        # cast and untilize into the row-major index stick.
        index_flat = ttnn.reshape(index_rows, (1, 1, 1, slots))
        _functional_decoder._free(index_rows, index_flat)
        index_u16 = ttnn.typecast(index_flat, ttnn.uint16)
        ttnn.deallocate(index_flat)
        index_stick = ttnn.to_layout(index_u16, ttnn.ROW_MAJOR_LAYOUT)
        _functional_decoder._free(index_u16, index_stick)
        local_scores = ttnn.multiply(scores, ttnn.typecast(valid, ttnn.bfloat16))
        ttnn.deallocate(valid)
        # Weight column per slot: row r's j-th score placed on row r only.
        columns = []
        for row in range(rows):
            row_scores = ttnn.slice(local_scores, [0, 0, row, 0], [1, 1, row + 1, k])  # [1,1,1,k]
            per_slot = ttnn.permute(row_scores, (0, 3, 2, 1))  # [1,k,1,1]
            ttnn.deallocate(row_scores)
            column = ttnn.multiply(per_slot, self._row_onehot(row))  # [1,k,32,1]
            ttnn.deallocate(per_slot)
            columns.append(column)
        ttnn.deallocate(local_scores)
        slot_weights = ttnn.concat(columns, dim=1) if rows > 1 else columns[0]
        if rows > 1:
            for column in columns:
                ttnn.deallocate(column)

        grouped_x = ttnn.reshape(x, (1, 1, 32, s.hidden_size))
        output_tile = ttnn.Tile([32, 32])
        gate_up_sparse = ttnn.sparse_matmul(
            grouped_x,
            bank.gate_up_bank,
            sparsity=bank.indexed_sparsity,
            indices=index_stick,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_b_sparse=True,
            program_config=self._sparse_matmul_config(32, 2 * s.moe_intermediate_size, s.hidden_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _functional_decoder._free(grouped_x, x, gate_up_sparse)
        gate_up = ttnn.reshape(gate_up_sparse, (1, slots, 32, 2 * s.moe_intermediate_size))
        _functional_decoder._free(gate_up_sparse, gate_up)
        gate = self._slice_last(gate_up, 0, s.moe_intermediate_size)
        up = self._slice_last(gate_up, s.moe_intermediate_size, 2 * s.moe_intermediate_size)
        ttnn.deallocate(gate_up)
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        weighted = ttnn.multiply(hidden, slot_weights)
        ttnn.deallocate(hidden)
        ttnn.deallocate(slot_weights)
        down = ttnn.sparse_matmul(
            weighted,
            bank.down_bank,
            sparsity=bank.indexed_sparsity,
            indices=index_stick,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=self._sparse_matmul_config(32, s.hidden_size, s.moe_intermediate_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _functional_decoder._free(weighted, down)
        ttnn.deallocate(index_stick)
        out = ttnn.experimental.fast_reduce_nc(down, dims=[1])
        ttnn.deallocate(down)
        return ttnn.reshape(ttnn.unsqueeze_to_4D(out), (1, 1, 32, s.hidden_size))

    @property
    def moe_fused_kernels(self) -> bool:
        """Use ``moe_route_topk`` / ``moe_weighted_sum`` for batch-one decode."""

        return os.environ.get("QWEN38_MOE_FUSED", "1") == "1"

    def _routed_experts_sparse_bank_fused(self, x, logits):
        """Batch-one decode: fused routing kernel, indexed bank matmuls, fused weighted sum."""

        s = self.shapes
        bank = self.resident_experts
        k = s.num_experts_per_tok
        index_stick, local_scores = ttnn.experimental.kda.moe_route_topk(
            logits, bank.rank_base, k, bank.experts_per_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(logits)
        rows = int(x.shape[-2])
        grouped_x = ttnn.reshape(x, (1, 1, rows, s.hidden_size))
        output_tile = ttnn.Tile([32, 32])
        gate_up_sparse = ttnn.sparse_matmul(
            grouped_x,
            bank.gate_up_bank,
            sparsity=bank.indexed_sparsity,
            indices=index_stick,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_b_sparse=True,
            program_config=self._sparse_matmul_config(32, 2 * s.moe_intermediate_size, s.hidden_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _functional_decoder._free(grouped_x, x, gate_up_sparse)
        gate_up = ttnn.reshape(gate_up_sparse, (1, k, rows, 2 * s.moe_intermediate_size))
        _functional_decoder._free(gate_up_sparse, gate_up)
        if os.environ.get("QWEN38_MOE_FUSED_SWIGLU", "1") == "1":
            hidden = ttnn.experimental.kda.moe_swiglu(gate_up, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(gate_up)
        else:
            gate = self._slice_last(gate_up, 0, s.moe_intermediate_size)
            up = self._slice_last(gate_up, s.moe_intermediate_size, 2 * s.moe_intermediate_size)
            ttnn.deallocate(gate_up)
            hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
        down = ttnn.sparse_matmul(
            hidden,
            bank.down_bank,
            sparsity=bank.indexed_sparsity,
            indices=index_stick,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=self._sparse_matmul_config(32, s.hidden_size, s.moe_intermediate_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(hidden)
        ttnn.deallocate(index_stick)
        out = ttnn.experimental.kda.moe_weighted_sum(down, local_scores, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(down)
        ttnn.deallocate(local_scores)
        return out

    def _routed_experts_sparse_bank_indexed(self, x, indices, scores):
        """Batch-one decode over the local bank in sparse_matmul indexed mode.

        The ten global top-k ids are mapped to this rank's bank slots on
        device; ids owned by another rank become slot 0 with weight 0, so each
        rank evaluates exactly ten bank slots with compact ``[1,10,32,N]``
        outputs and the weighted reduce is this rank's exact partial sum.
        """

        s = self.shapes
        bank = self.resident_experts
        k = s.num_experts_per_tok
        local_experts = bank.experts_per_device
        ids = ttnn.typecast(indices, ttnn.int32)
        rel = ttnn.subtract(ids, bank.rank_base)
        ttnn.deallocate(ids)
        rel_f = ttnn.typecast(rel, ttnn.float32)
        ttnn.deallocate(rel)
        not_below = ttnn.ge(rel_f, 0.0)
        below_end = ttnn.lt(rel_f, float(local_experts))
        valid = ttnn.multiply(not_below, below_end)
        ttnn.deallocate(not_below)
        ttnn.deallocate(below_end)
        local_f = ttnn.multiply(rel_f, valid)
        ttnn.deallocate(rel_f)
        local_i32 = ttnn.typecast(local_f, ttnn.int32)
        ttnn.deallocate(local_f)
        local_u16 = ttnn.typecast(local_i32, ttnn.uint16)
        ttnn.deallocate(local_i32)
        index_row = ttnn.slice(local_u16, [0, 0, 0, 0], [1, 1, 1, k])
        ttnn.deallocate(local_u16)
        index_stick = ttnn.to_layout(index_row, ttnn.ROW_MAJOR_LAYOUT)
        _functional_decoder._free(index_row, index_stick)
        valid_bf16 = ttnn.typecast(valid, ttnn.bfloat16)
        ttnn.deallocate(valid)
        local_scores = ttnn.multiply(scores, valid_bf16)
        ttnn.deallocate(valid_bf16)

        grouped_x = ttnn.reshape(x, (1, 1, 32, s.hidden_size))
        output_tile = ttnn.Tile([32, 32])
        gate_up_sparse = ttnn.sparse_matmul(
            grouped_x,
            bank.gate_up_bank,
            sparsity=bank.indexed_sparsity,
            indices=index_stick,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_b_sparse=True,
            program_config=self._sparse_matmul_config(32, 2 * s.moe_intermediate_size, s.hidden_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _functional_decoder._free(grouped_x, x, gate_up_sparse)
        gate_up = ttnn.reshape(gate_up_sparse, (1, k, 32, 2 * s.moe_intermediate_size))
        _functional_decoder._free(gate_up_sparse, gate_up)
        gate = self._slice_last(gate_up, 0, s.moe_intermediate_size)
        up = self._slice_last(gate_up, s.moe_intermediate_size, 2 * s.moe_intermediate_size)
        ttnn.deallocate(gate_up)
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        selected_weights = ttnn.permute(local_scores, (0, 3, 2, 1))
        weighted = ttnn.multiply(hidden, selected_weights)
        ttnn.deallocate(hidden)
        _functional_decoder._free(selected_weights, local_scores, weighted)
        ttnn.deallocate(local_scores)
        down = ttnn.sparse_matmul(
            weighted,
            bank.down_bank,
            sparsity=bank.indexed_sparsity,
            indices=index_stick,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=self._sparse_matmul_config(32, s.hidden_size, s.moe_intermediate_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _functional_decoder._free(weighted, down)
        ttnn.deallocate(index_stick)
        if os.environ.get("QWEN38_MOE_IDX_REDUCE", "fast") == "sum":
            out = ttnn.sum(down, dim=1, keepdim=True)
        else:
            out = ttnn.experimental.fast_reduce_nc(down, dims=[1])
        ttnn.deallocate(down)
        return ttnn.reshape(ttnn.unsqueeze_to_4D(out), (1, 1, 32, s.hidden_size))

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
        chunk_start: int = 0,
        valid_mask: torch.Tensor | None = None,
    ):
        """Layer-major compatibility wrapper for exact host PLE prefill."""

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
            raise ValueError(f"host-backed fractured prefill expects [1, 1, 4*seq, {RESIDUAL_SHARD_WIDTH}]")
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

        pieces = []
        for chunk_index, (start, logical, padded) in enumerate(self.prefill_chunk_plan(seq_len)):
            x = self._slice_fractured_seq(hidden_states, start, logical, padded)
            out = self.prefill_microchunk_host_backed_fractured(
                x,
                input_ids=ids[:, start : start + logical],
                request_id=request_id,
                logical=logical,
                user_id=user_id,
                chunk_start=chunk_start + start,
                reset_state=chunk_start + start == 0 and chunk_index == 0,
                valid_mask=None if mask is None else mask[:, start : start + logical],
            )
            _functional_decoder._free(x, hidden_states, out)
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

    def prefill_microchunk_host_backed_fractured(
        self,
        hidden_states,
        *,
        input_ids: torch.Tensor,
        request_id,
        logical: int,
        user_id: int = 0,
        chunk_start: int = 0,
        reset_state: bool = False,
        valid_mask: torch.Tensor | None = None,
    ):
        """Prepare one host n-gram slice and run the stack-major PLE layer."""

        self._require_host_ple()
        ids = torch.as_tensor(input_ids, dtype=torch.int64, device="cpu")
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if tuple(ids.shape) != (1, int(logical)):
            raise ValueError(f"host PLE microchunk ids must be [1, {logical}], got {tuple(ids.shape)}")
        mask = None if valid_mask is None else torch.as_tensor(valid_mask, dtype=torch.bool, device="cpu")
        if mask is not None and tuple(mask.shape) != tuple(ids.shape):
            raise ValueError("host PLE microchunk valid mask shape does not match input ids")
        embeddings = self.host_ple_store.prepare(
            [request_id],
            ids,
            valid_mask=mask,
            reset=reset_state,
        )
        staged = self.ple_staging.upload_prefill(embeddings, logical=int(logical))
        # The staging buffer holds PREFILL_CHUNK rows; an adaptive 128-row tail
        # chunk needs embeddings of its own physical row count.
        physical = _functional_decoder._shape(hidden_states)[-2] // self.shapes.hc_count
        if int(staged.shape[-2]) != physical:
            staged = _functional_decoder._slice_seq(staged, 0, int(logical), physical)
        return self.prefill_microchunk_forward_fractured(
            hidden_states,
            logical=int(logical),
            user_id=user_id,
            chunk_start=chunk_start,
            reset_state=reset_state,
            ple_embeddings=staged,
        )

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
        if self.resident_experts is not None:
            self.resident_experts.close()
            self.resident_experts = None
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
                    raise RuntimeError(f"segmented decode state is not replicated over TP{TP_SIZE}")
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


class ResidentLayerDecodeTrace:
    """One full-layer trace for resident EP4 execution.

    There is no expert host boundary.  Layer 1 retains the single declared PLE
    boundary: selected n-gram rows are uploaded into its persistent staging
    tensor before the trace is submitted.
    """

    def __init__(self, layer, output, trace_id, *, state_workspace=None, captured_inputs=()):
        self.layer = layer
        self.output = output
        self.trace_id = trace_id
        self.state_workspace = state_workspace
        self.captured_inputs = tuple(captured_inputs)
        self.last_timing = None
        self.last_route_ids = None
        self.released = False
        self._lock = threading.RLock()

    @staticmethod
    def _arguments(layer, *, current_pos, page_table, rot_mats, ple_input_ids, request_ids):
        linear, ple_ids, ple_requests, kwargs = HostBackedSegmentedDecodeTrace._capture_arguments(
            layer,
            current_pos=current_pos,
            page_table=page_table,
            rot_mats=rot_mats,
            ple_input_ids=ple_input_ids,
            request_ids=request_ids,
        )
        del linear
        return ple_ids, ple_requests, kwargs

    @staticmethod
    def _stage_ple(layer, ids, requests, kwargs) -> None:
        if ids is None:
            return
        embeddings = layer.host_ple_store.prepare(requests, ids)
        kwargs["ple_embeddings"] = layer.ple_staging.upload_decode(embeddings)

    @classmethod
    def warm_programs(
        cls,
        layer,
        hidden_states,
        *,
        current_pos,
        page_table=None,
        rot_mats=None,
        ple_input_ids=None,
        request_ids=None,
    ) -> None:
        if layer.resident_experts is None:
            raise RuntimeError("resident trace requires resident EP4 experts")
        ple_ids, ple_requests, kwargs = cls._arguments(
            layer,
            current_pos=current_pos,
            page_table=page_table,
            rot_mats=rot_mats,
            ple_input_ids=ple_input_ids,
            request_ids=request_ids,
        )
        state = HostBackedSegmentedDecodeTrace._snapshot_decode_state(layer)
        history = HostBackedSegmentedDecodeTrace._snapshot_ple_history(layer, ple_requests)
        output = None
        workspace = layer.decode_state_workspace
        scope = workspace.serialize_replay() if workspace is not None else nullcontext()
        try:
            with scope:
                cls._stage_ple(layer, ple_ids, ple_requests, kwargs)
                output = layer.decode_step(hidden_states, **kwargs)
                ttnn.synchronize_device(layer.mesh_device)
        finally:
            try:
                if output is not None and output.is_allocated():
                    ttnn.deallocate(output)
            finally:
                try:
                    HostBackedSegmentedDecodeTrace._restore_decode_state(layer, state)
                finally:
                    HostBackedSegmentedDecodeTrace._release_state_snapshots(state)
                    HostBackedSegmentedDecodeTrace._restore_ple_history(layer, history)

    @classmethod
    def capture(
        cls,
        layer,
        hidden_states,
        *,
        current_pos,
        page_table=None,
        rot_mats=None,
        ple_input_ids=None,
        request_ids=None,
        programs_prepared=False,
    ) -> "ResidentLayerDecodeTrace":
        if layer.resident_experts is None:
            raise RuntimeError("resident trace requires resident EP4 experts")
        if not programs_prepared:
            cls.warm_programs(
                layer,
                hidden_states,
                current_pos=current_pos,
                page_table=page_table,
                rot_mats=rot_mats,
                ple_input_ids=ple_input_ids,
                request_ids=request_ids,
            )
        ple_ids, ple_requests, kwargs = cls._arguments(
            layer,
            current_pos=current_pos,
            page_table=page_table,
            rot_mats=rot_mats,
            ple_input_ids=ple_input_ids,
            request_ids=request_ids,
        )
        cls._stage_ple(layer, ple_ids, ple_requests, kwargs)
        workspace = layer.decode_state_workspace
        if workspace is not None:
            workspace.retain_trace()
        scope = workspace.serialize_replay() if workspace is not None else nullcontext()
        trace_id = output = None
        capture_open = False
        try:
            with scope:
                layer.mesh_device.set_program_cache_misses_allowed(False)
                trace_id = ttnn.begin_trace_capture(layer.mesh_device, cq_id=0)
                capture_open = True
                output = layer.decode_step(hidden_states, **kwargs)
                ttnn.end_trace_capture(layer.mesh_device, trace_id, cq_id=0)
                capture_open = False
                ttnn.mark_corruptible(output)
                ttnn.execute_trace(layer.mesh_device, trace_id, cq_id=0, blocking=True)
        except BaseException:
            HostBackedSegmentedDecodeTrace._finish_failed_capture(layer.mesh_device, trace_id, capture_open)
            if output is not None and output.is_allocated():
                ttnn.deallocate(output)
            if workspace is not None:
                workspace.release_trace()
            raise
        finally:
            layer.mesh_device.set_program_cache_misses_allowed(True)
        return cls(layer, output, trace_id, state_workspace=workspace, captured_inputs=(hidden_states,))

    def replay(self, *, ple_input_ids=None, request_ids=None, ple_embeddings=None):
        with self._lock:
            if self.released:
                raise RuntimeError("resident layer trace was released")
            started = time.perf_counter()
            ple_seconds = 0.0
            if self.layer.shapes.has_ple:
                ple_started = time.perf_counter()
                if ple_embeddings is None:
                    ids = torch.as_tensor(ple_input_ids, dtype=torch.int64, device="cpu")
                    if ids.ndim == 1:
                        ids = ids.unsqueeze(1)
                    requests = tuple(request_ids)
                    embeddings = self.layer.host_ple_store.prepare(requests, ids)
                else:
                    embeddings = ple_embeddings
                self.layer.ple_staging.upload_decode(embeddings)
                ple_seconds = time.perf_counter() - ple_started
            elif ple_input_ids is not None or request_ids is not None or ple_embeddings is not None:
                raise ValueError("PLE inputs were passed to a layer without PLE")
            scope = (
                self.state_workspace.serialize_replay()
                if self.state_workspace is not None
                else nullcontext()
            )
            submit_started = time.perf_counter()
            with scope:
                ttnn.execute_trace(self.layer.mesh_device, self.trace_id, cq_id=0, blocking=False)
            trace_seconds = time.perf_counter() - submit_started
            self.last_timing = {
                "ple_seconds": ple_seconds,
                "front_trace_seconds": trace_seconds,
                "expert_service_seconds": 0.0,
                "route_read_and_tt_stall_seconds": 0.0,
                "cache_control_dma_submit_seconds": 0.0,
                "back_trace_seconds": 0.0,
                "total_seconds": time.perf_counter() - started,
                "expert_hits": 0,
                "expert_misses": 0,
            }
            return self.output

    def release(self) -> None:
        with self._lock:
            if self.released:
                return
            if self.trace_id is not None:
                ttnn.release_trace(self.layer.mesh_device, self.trace_id)
                self.trace_id = None
            if self.output.is_allocated():
                ttnn.deallocate(self.output)
            self.captured_inputs = ()
            if self.state_workspace is not None:
                self.state_workspace.release_trace()
                self.state_workspace = None
            self.released = True


__all__ = [
    "HostBackedSegmentedDecodeTrace",
    "HostDecodeAttention",
    "HostDecodeFront",
    "MultichipDecodeStateWorkspace",
    "MultichipVirtualDecodeStateBank",
    "MultichipDecoder",
    "MultichipMemoryPlan",
    "ResidentLayerDecodeTrace",
]

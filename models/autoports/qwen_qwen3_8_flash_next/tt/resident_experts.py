# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-resident EP4 routed experts for Qwen3.8-Flash-Next.

All 512 experts in a layer are distributed in four contiguous 128-expert
blocks, matching the device offset-cumsum kernel's region contract.  Each
Blackhole owns complete gate/up/down BFP4 experts.  Runtime routing remains on
device: activations are dispatched over fabric, evaluated by the fused routed
expert kernel, combined at their source ranks, and reduce-scattered into the
TP4 residual representation.

The checkpoint adapter is deliberately lazy.  It presents the contiguous
ordering expected by the shared routed-expert loader while retaining only a
small setup-time LRU.  There is no 68-GB packed CPU expert store.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.host_weight_cache import (
    EXPERTS,
    GLOBAL_INTERMEDIATE,
    HIDDEN_SIZE,
    SafetensorCheckpoint,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.parallel_config import P300_TP4_EP4, Qwen38ParallelConfig
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

TOP_K = 10
EXPERTS_PER_DEVICE = EXPERTS // P300_TP4_EP4.expert_parallel
EXPERT_BFP4_BYTES = 2_764_800
RESIDENT_EXPERT_BYTES_PER_DEVICE_PER_LAYER = EXPERTS_PER_DEVICE * EXPERT_BFP4_BYTES
RESIDENT_EXPERT_BYTES_PER_DEVICE = 48 * RESIDENT_EXPERT_BYTES_PER_DEVICE_PER_LAYER


def _free(tensor, *live) -> None:
    """Deallocate a transient TT tensor unless it aliases a live view."""

    if tensor is None or not tensor.is_allocated():
        return
    address = tensor.buffer_address()
    if any(other is not None and other.is_allocated() and other.buffer_address() == address for other in live):
        return
    ttnn.deallocate(tensor)


def contiguous_dispatch_table(
    *,
    num_experts: int = EXPERTS,
    expert_parallel: int = P300_TP4_EP4.expert_parallel,
) -> torch.Tensor:
    """Map each global expert to its contiguous EP owner.

    The final sentinel entry is consumed by padding-aware dispatch kernels.
    """

    if num_experts % expert_parallel:
        raise ValueError("expert count must be divisible by expert parallel size")
    table = torch.full((1, num_experts + 1), -1, dtype=torch.int32)
    experts_per_device = num_experts // expert_parallel
    table[0, :num_experts] = torch.arange(num_experts, dtype=torch.int32) // experts_per_device
    return table


def contiguous_global_expert_table(
    *,
    num_experts: int = EXPERTS,
    expert_parallel: int = P300_TP4_EP4.expert_parallel,
) -> torch.Tensor:
    """Return ``[group, owner, local] -> global`` for contiguous ownership."""

    if num_experts % expert_parallel:
        raise ValueError("expert count must be divisible by expert parallel size")
    return torch.arange(num_experts, dtype=torch.int32).reshape(1, expert_parallel, -1)


def contiguous_loader_to_global_expert(
    index: int,
    *,
    num_experts: int = EXPERTS,
    expert_parallel: int = P300_TP4_EP4.expert_parallel,
) -> int:
    """Translate the shared loader's contiguous placement to global ownership."""

    if not 0 <= int(index) < num_experts:
        raise IndexError(index)
    return int(index)


class Qwen38ResidentExpertSource(Sequence):
    """Bounded lazy checkpoint view consumed by :class:`TtRoutedExpert`."""

    def __init__(
        self,
        checkpoint: SafetensorCheckpoint,
        layer_idx: int,
        *,
        parallel_config: Qwen38ParallelConfig = P300_TP4_EP4,
        cache_entries: int = 8,
    ) -> None:
        if cache_entries < parallel_config.expert_parallel:
            raise ValueError("resident expert source cache must cover one expert per EP rank")
        self.checkpoint = checkpoint
        self.layer_idx = int(layer_idx)
        self.parallel_config = parallel_config
        self.cache_entries = int(cache_entries)
        prefix = f"model.language_model.layers.{self.layer_idx}.mlp.experts"
        self.gate_up_key = f"{prefix}.gate_up_proj"
        self.down_key = f"{prefix}.down_proj"
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self.checkpoint_reads = 0
        self.checkpoint_bytes = 0
        self.checkpoint_read_seconds = 0.0

    def __len__(self) -> int:
        return EXPERTS

    def __getitem__(self, virtual_index: int) -> dict[str, torch.Tensor]:
        if isinstance(virtual_index, slice):
            return [self[index] for index in range(*virtual_index.indices(len(self)))]
        global_id = contiguous_loader_to_global_expert(
            int(virtual_index), expert_parallel=self.parallel_config.expert_parallel
        )
        cached = self._cache.pop(global_id, None)
        if cached is not None:
            self._cache[global_id] = cached
            return cached

        started = time.perf_counter()
        fused = self.checkpoint.indexed_tensor(self.gate_up_key, global_id)
        down = self.checkpoint.indexed_tensor(self.down_key, global_id)
        if tuple(fused.shape) != (2 * GLOBAL_INTERMEDIATE, HIDDEN_SIZE):
            raise ValueError(f"expert {global_id} gate/up shape is {tuple(fused.shape)}")
        if tuple(down.shape) != (HIDDEN_SIZE, GLOBAL_INTERMEDIATE):
            raise ValueError(f"expert {global_id} down shape is {tuple(down.shape)}")
        value = {
            "gate_proj": fused[:GLOBAL_INTERMEDIATE].contiguous(),
            "up_proj": fused[GLOBAL_INTERMEDIATE:].contiguous(),
            "down_proj": down.contiguous(),
        }
        self._cache[global_id] = value
        while len(self._cache) > self.cache_entries:
            self._cache.popitem(last=False)
        self.checkpoint_reads += 1
        self.checkpoint_bytes += int(fused.numel() + down.numel()) * fused.element_size()
        self.checkpoint_read_seconds += time.perf_counter() - started
        return value

    @property
    def manifest(self) -> tuple[dict[str, object], ...]:
        return self.checkpoint.manifest((self.gate_up_key, self.down_key))

    def release_host_tensors(self) -> None:
        self._cache.clear()

    def metrics(self) -> dict[str, object]:
        return {
            "mode": "resident_ep4",
            "layer": self.layer_idx,
            "checkpoint_expert_reads": self.checkpoint_reads,
            "checkpoint_bytes": self.checkpoint_bytes,
            "checkpoint_read_seconds": self.checkpoint_read_seconds,
            "host_cache_entries": len(self._cache),
            "expert_host_store_bytes": 0,
        }


@dataclasses.dataclass(frozen=True)
class _ResidentPipeline:
    seq_len: int
    routing: TtMoERoutingSetup
    dispatch: TtDispatchModule
    combine: TtCombineModule


class Qwen38ResidentExperts:
    """Trace-safe prefill/decode EP4 pipeline sharing one resident weight set."""

    def __init__(
        self,
        mesh_device,
        source: Qwen38ResidentExpertSource,
        *,
        max_batch: int,
        prefill_rows: int = 128,
        num_links: int = 2,
        topology=ttnn.Topology.Linear,
        weight_cache_path: Path | None = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.source = source
        self.parallel_config = source.parallel_config
        self.parallel_config.validate_mesh(mesh_device)
        self.max_batch = int(max_batch)
        self.prefill_rows = int(prefill_rows)
        self.num_links = int(num_links)
        self.topology = topology
        self.dispatch_group_size = self.parallel_config.expert_parallel
        self.num_dispatch_groups = 1
        self.experts_per_device = EXPERTS // self.dispatch_group_size
        self.dispatch_table_host = contiguous_dispatch_table(expert_parallel=self.dispatch_group_size)

        dispatch_table = TtDispatchModule.shard_expert_dispatch_table(
            mesh_device,
            self.dispatch_table_host,
            dispatch_axis=self.parallel_config.expert_dispatch_axis,
        )
        self.dispatch_table = dispatch_table
        global_table_host = contiguous_global_expert_table(expert_parallel=self.dispatch_group_size)
        global_table = ttnn.from_torch(
            global_table_host,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                mesh_shape=mesh_device.shape,
                dims=(1, None)
                if self.parallel_config.expert_dispatch_axis == 0
                else (None, 1),
            ),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )
        self.global_expert_table = ttnn.squeeze(ttnn.squeeze(global_table, 0), 0)

        # The shared loader expects contiguous per-device expert blocks.  The
        # lazy source reads only the four current per-rank entries.
        self.routed_expert = TtRoutedExpert(
            mesh_device=mesh_device,
            experts_per_chip=self.experts_per_device,
            global_expert_idx_table=self.global_expert_table,
            emb_dim=HIDDEN_SIZE,
            hidden_dim=GLOBAL_INTERMEDIATE,
            # A token has exactly one source rank.  The 32 minimum covers the
            # expert kernel's tile-aligned decode buffer; prefill can send all
            # rows in a chunk to the same expert.
            max_tokens=max(32, self.prefill_rows),
            torch_weights=source,
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat4_b,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"qwen38.layer_{source.layer_idx}.resident_ep4",
            activation=ttnn.RoutedExpertActivation.Silu,
        )
        source.release_host_tensors()

        decode_rows = max(32, 32 * math.ceil(self.max_batch / 32))
        self._pipelines = {
            decode_rows: self._make_pipeline(decode_rows, active_rows=self.max_batch),
            self.prefill_rows: self._make_pipeline(self.prefill_rows, active_rows=self.prefill_rows),
        }
        self._source_masks = {rows: self._make_source_mask(rows) for rows in self._pipelines}
        self.reduce = TtReduceModule(
            mesh_device=mesh_device,
            topk_dim=3,
            cluster_axis=self.parallel_config.collective_axis,
            num_links=self.num_links,
            topology=self.topology,
        )
        self.calls = 0
        self.decode_calls = 0
        self.prefill_calls = 0

    def _make_source_mask(self, seq_len: int):
        """Create a distributed mask assigning every logical row one source."""

        mesh_rows, mesh_cols = (int(value) for value in self.mesh_device.shape)
        host = torch.zeros((mesh_rows, mesh_cols, 1, 1, seq_len, 1), dtype=torch.int32)
        for token in range(seq_len):
            rank = token % self.dispatch_group_size
            row, col = divmod(rank, mesh_cols)
            host[row, col, 0, 0, token, 0] = 1
        mask = ttnn.from_torch(
            host,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                mesh_shape=self.mesh_device.shape,
                dims=(0, 1),
            ),
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            dtype=ttnn.uint32,
        )
        return ttnn.squeeze(ttnn.squeeze(mask, dim=0), dim=0)

    def _make_pipeline(self, seq_len: int, *, active_rows: int) -> _ResidentPipeline:
        routing = TtMoERoutingSetup(
            self.mesh_device,
            self.dispatch_table_host,
            num_links=self.num_links,
            experts_per_chip=self.experts_per_device,
            cluster_axis=self.parallel_config.expert_dispatch_axis,
        )
        # Every logical token is sourced by exactly one rank.  Reserve its ten
        # routes plus one alignment tile per potentially selected expert.
        max_routes = active_rows * TOP_K
        alignment_slack = min(EXPERTS, active_rows * TOP_K) * 31
        capacity = 32 * math.ceil((max_routes + alignment_slack) / 32)
        dispatch = TtDispatchModule(
            mesh_device=self.mesh_device,
            dispatch_group_size=self.dispatch_group_size,
            experts_per_chip=self.experts_per_device,
            num_routed_experts=EXPERTS,
            num_experts_per_tok=TOP_K,
            metadata_len=3,
            max_dispatch_buffer_token_size=capacity,
            seq_len_per_chip=seq_len,
            emb_dim=HIDDEN_SIZE,
            cluster_axis=self.parallel_config.expert_dispatch_axis,
            num_links=self.num_links,
            topology=self.topology,
        )
        combine = TtCombineModule(
            mesh_device=self.mesh_device,
            dispatch_group_size=self.dispatch_group_size,
            num_dispatch_groups=self.num_dispatch_groups,
            experts_per_chip=self.experts_per_device,
            num_experts_per_tok=TOP_K,
            seq_len_per_chip=seq_len,
            cluster_axis=self.parallel_config.expert_dispatch_axis,
            num_links=self.num_links,
            topology=self.topology,
            init_zeros=True,
        )
        return _ResidentPipeline(seq_len, routing, dispatch, combine)

    def __call__(self, x, indices, scores, *, logical_rows: int | None = None):
        rows = int(x.shape[-2])
        logical_rows = rows if logical_rows is None else int(logical_rows)
        if not 0 < logical_rows <= rows:
            raise ValueError(f"logical resident EP4 rows must be in [1, {rows}], got {logical_rows}")
        try:
            pipeline = self._pipelines[rows]
        except KeyError as exc:
            raise ValueError(f"resident EP4 supports padded row counts {tuple(self._pipelines)}, got {rows}") from exc

        # Each logical token is routed from one source rank only.  This avoids
        # four identical expert evaluations while reduce-scatter still returns
        # every output hidden shard.  Inactive padded rows use the trailing
        # dispatch-table sentinel and therefore perform no expert work.
        active_indices = indices
        if logical_rows != rows:
            real_indices = ttnn.slice(indices, [0, 0, 0, 0], [1, 1, logical_rows, TOP_K])
            active_indices = ttnn.pad(
                real_indices,
                [(0, 0), (0, 0), (0, rows - logical_rows), (0, 0)],
                value=EXPERTS,
            )
            _free(real_indices, indices, active_indices)
        active_indices_u32 = ttnn.typecast(active_indices, ttnn.uint32)
        owned_indices_u32 = ttnn.where(self._source_masks[rows], active_indices_u32, EXPERTS)
        owned_indices = ttnn.typecast(owned_indices_u32, ttnn.uint16)
        _free(active_indices_u32, active_indices, owned_indices_u32)
        _free(owned_indices_u32, owned_indices)

        # Dispatch operates on rank-three per-device tensors.  masked_bincount
        # uses a fixed 64-core grid, so append routing-only sentinel rows for a
        # one-tile decode buffer.
        routing_input = owned_indices
        if rows % 64:
            routing_input = ttnn.pad(
                routing_input,
                [(0, 0), (0, 0), (0, 64 - rows % 64), (0, 0)],
                value=EXPERTS,
            )
        routing_indices = ttnn.squeeze(ttnn.squeeze(routing_input, dim=0), dim=0)
        offsets, counts, region_offsets, histograms = pipeline.routing(
            routing_indices,
            num_routed_experts=EXPERTS,
            num_experts_per_tok=TOP_K,
        )
        indices_rm = ttnn.to_layout(owned_indices, ttnn.ROW_MAJOR_LAYOUT)
        scores_rm = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)
        x_dispatch = ttnn.squeeze(x, 0)
        indices_dispatch = ttnn.reshape(indices_rm, (1, rows, TOP_K))
        scores_dispatch = ttnn.reshape(scores_rm, (1, rows, TOP_K))
        raw_dispatched, metadata = pipeline.dispatch(
            x_dispatch,
            scores_dispatch,
            indices_dispatch,
            offsets,
            self.dispatch_table,
        )
        dispatched_view = ttnn.squeeze(ttnn.squeeze(raw_dispatched, dim=0), dim=0)
        dispatched = ttnn.to_layout(
            dispatched_view,
            ttnn.TILE_LAYOUT,
            dtype=self.routed_expert.activations_dtype,
        )
        _free(raw_dispatched, dispatched_view, dispatched)
        expert_output = self.routed_expert(dispatched, counts, region_offsets)
        _free(dispatched, expert_output)
        expert_output_view = ttnn.unsqueeze(ttnn.unsqueeze(expert_output, dim=0), dim=0)
        combined = pipeline.combine(expert_output_view, metadata, counts, region_offsets)
        _free(expert_output, expert_output_view, combined)
        local = self.reduce(
            combined,
            weights=scores_dispatch,
            indices=indices_dispatch,
            expert_dispatch_table=self.dispatch_table,
        )
        _free(combined, local)
        local = ttnn.reshape(ttnn.unsqueeze_to_4D(local), (1, 1, rows, HIDDEN_SIZE // self.dispatch_group_size))

        for transient, live in (
            (metadata, (local,)),
            (offsets, (local,)),
            (counts, (local,)),
            (region_offsets, (local,)),
            (histograms, (local,)),
            (indices_dispatch, (local, indices_rm)),
            (scores_dispatch, (local, scores_rm)),
            (indices_rm, (local, owned_indices)),
            (scores_rm, (local, scores)),
            (routing_indices, (local, routing_input)),
            (routing_input, (local, owned_indices)),
            (owned_indices, (local, active_indices, indices)),
            (active_indices, (local, indices)),
        ):
            _free(transient, *live)
        self.calls += 1
        if rows == 32:
            self.decode_calls += 1
        else:
            self.prefill_calls += 1
        return local

    def metrics(self) -> dict[str, object]:
        return {
            **self.source.metrics(),
            "experts_per_device": self.experts_per_device,
            "resident_expert_bytes_per_device": RESIDENT_EXPERT_BYTES_PER_DEVICE_PER_LAYER,
            "expert_weight_h2d_bytes_runtime": 0,
            "expert_route_d2h_bytes_runtime": 0,
            "calls": self.calls,
            "decode_calls": self.decode_calls,
            "prefill_calls": self.prefill_calls,
        }

    def close(self) -> None:
        tensors = [
            self.dispatch_table,
            self.global_expert_table,
            *self._source_masks.values(),
            self.routed_expert.global_expert_idx_table,
            *(pipeline.routing.experts_in_dispatch_group for pipeline in self._pipelines.values()),
            *self.routed_expert.gate_projs,
            *self.routed_expert.up_projs,
            *self.routed_expert.down_projs,
        ]
        seen = set()
        for tensor in tensors:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            if isinstance(tensor, ttnn.Tensor) and tensor.is_allocated():
                ttnn.deallocate(tensor)


__all__ = [
    "EXPERTS_PER_DEVICE",
    "Qwen38ResidentExpertSource",
    "Qwen38ResidentExperts",
    "RESIDENT_EXPERT_BYTES_PER_DEVICE",
    "RESIDENT_EXPERT_BYTES_PER_DEVICE_PER_LAYER",
    "contiguous_loader_to_global_expert",
    "contiguous_dispatch_table",
    "contiguous_global_expert_table",
]

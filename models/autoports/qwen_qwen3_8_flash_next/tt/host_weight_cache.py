# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact host-backed weights for the Qwen3.8 Flash Next multichip decoder.

This module contains the deliberately small host/device boundary used by the
P300 decoder.  Routed-expert projections and every operation after PLE row
assembly execute on TT.  The host is allowed to read compact route ids, mmap
checkpoint rows, maintain fixed-slot metadata, pack rank-local weights, and
copy those weights/PLE rows into stable device buffers.

The classes are usable without a TT device so hashing, mmap lookup, eviction,
generation, and checkpoint-layout semantics can be tested independently.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

PLE_LAYER = 1
PLE_SHARDS = 128
PLE_ROWS_PER_SHARD = 2_500_012
PLE_ROW_WIDTH = 160
PLE_HEADS = 16
PLE_EMBED_DIM = PLE_HEADS * PLE_ROW_WIDTH
PLE_LOGICAL_ROWS = 320_001_446
PLE_PADDED_ROWS = PLE_SHARDS * PLE_ROWS_PER_SHARD
PLE_TABLE_BYTES = PLE_PADDED_ROWS * PLE_ROW_WIDTH * 2
PLE_ROW_BYTES = PLE_ROW_WIDTH * 2

EXPERTS = 512
GLOBAL_INTERMEDIATE = 640
TP_SIZE = 2
HIDDEN_SIZE = 2560
BFP4_TILE_BYTES = 576
# Routed experts use deterministic EP2 ownership.  Each physical rank has a
# full-K slot so the selected owner executes the checkpoint-identical
# 640-wide projection; the non-owner slot contains exact zeros.  This avoids
# the numerically divergent 320+320 split-K down projection while preserving
# bounded, fixed-address host backing.
EXPERT_GATE_UP_TILES_PER_RANK = (HIDDEN_SIZE // 32) * (2 * GLOBAL_INTERMEDIATE // 32)
EXPERT_DOWN_TILES_PER_RANK = (GLOBAL_INTERMEDIATE // 32) * (HIDDEN_SIZE // 32)
EXPERT_PACKED_BYTES_PER_RANK = (EXPERT_GATE_UP_TILES_PER_RANK + EXPERT_DOWN_TILES_PER_RANK) * BFP4_TILE_BYTES


def _ordered_unique(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(value) for value in values))


class SafetensorCheckpoint:
    """Indexed, mmap-backed access to one local Hugging Face snapshot."""

    def __init__(self, snapshot: str | Path):
        self.snapshot = Path(snapshot).resolve()
        index_path = self.snapshot / "model.safetensors.index.json"
        with index_path.open() as handle:
            index = json.load(handle)
        self.weight_map: dict[str, str] = dict(index["weight_map"])
        self.total_size = int(index["metadata"]["total_size"])

    def path_for(self, key: str) -> Path:
        try:
            return self.snapshot / self.weight_map[key]
        except KeyError as exc:
            raise KeyError(f"checkpoint has no tensor {key!r}") from exc

    def tensor(self, key: str) -> torch.Tensor:
        with safe_open(self.path_for(key), framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def indexed_tensor(self, key: str, index: int) -> torch.Tensor:
        with safe_open(self.path_for(key), framework="pt", device="cpu") as handle:
            return handle.get_slice(key)[index]

    def layer_state(self, layer_idx: int, *, include_experts: bool = False) -> dict[str, torch.Tensor]:
        """Load one layer while never materializing the PLE table by accident."""

        prefix = f"model.language_model.layers.{layer_idx}."
        selected = {}
        for full_name in self.weight_map:
            if not full_name.startswith(prefix):
                continue
            local_name = full_name[len(prefix) :]
            if ".ple_embedding.ngram_embedding." in local_name:
                continue
            if not include_experts and local_name in {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}:
                continue
            selected[local_name] = self.tensor(full_name)
        return selected

    def manifest(self, keys: Iterable[str]) -> tuple[dict[str, object], ...]:
        """Return immutable HF blob ids and sizes without hashing 336 GiB again."""

        entries = []
        for filename in sorted({self.weight_map[key] for key in keys}):
            path = self.snapshot / filename
            entries.append(
                {
                    "filename": filename,
                    "blob_sha256": path.resolve().name,
                    "bytes": path.stat().st_size,
                }
            )
        return tuple(entries)


@dataclasses.dataclass(frozen=True, order=True)
class ExpertIdentity:
    layer_idx: int
    expert_id: int


@dataclasses.dataclass(frozen=True)
class PackedExpert:
    """BF16 EP2 matrices immediately before device-native BFP4 packing.

    Exactly one rank owns the full expert; the other rank holds exact zeros.
    """

    gate_up_by_rank: tuple[torch.Tensor, torch.Tensor]
    down_by_rank: tuple[torch.Tensor, torch.Tensor]

    def __post_init__(self):
        for value in self.gate_up_by_rank:
            if tuple(value.shape) != (1, 1, HIDDEN_SIZE, 2 * GLOBAL_INTERMEDIATE):
                raise ValueError(f"rank gate/up shape {tuple(value.shape)} is invalid")
        for value in self.down_by_rank:
            if tuple(value.shape) != (1, 1, GLOBAL_INTERMEDIATE, HIDDEN_SIZE):
                raise ValueError(f"rank down shape {tuple(value.shape)} is invalid")


class Qwen38ExpertHostSource:
    """Lazy exact checkpoint source for routed experts.

    The two large layer tensors remain mmap-backed.  A miss reads exactly one
    expert and creates a full-K matrix pair on its deterministic EP2 owner and
    an exact-zero pair on the other rank.
    """

    def __init__(self, checkpoint: SafetensorCheckpoint, layer_idx: int):
        self.checkpoint = checkpoint
        self.layer_idx = int(layer_idx)
        prefix = f"model.language_model.layers.{self.layer_idx}.mlp.experts"
        self.gate_up_key = f"{prefix}.gate_up_proj"
        self.down_key = f"{prefix}.down_proj"
        self.host_reads = 0
        self.host_bytes = 0
        self.read_seconds = 0.0
        self._zero_gate_up = torch.zeros((1, 1, HIDDEN_SIZE, 2 * GLOBAL_INTERMEDIATE), dtype=torch.bfloat16)
        self._zero_down = torch.zeros((1, 1, GLOBAL_INTERMEDIATE, HIDDEN_SIZE), dtype=torch.bfloat16)
        self._lock = threading.RLock()

    @property
    def checkpoint_bytes_per_expert(self) -> int:
        return (2 * GLOBAL_INTERMEDIATE * HIDDEN_SIZE + HIDDEN_SIZE * GLOBAL_INTERMEDIATE) * 2

    @property
    def manifest(self) -> tuple[dict[str, object], ...]:
        return self.checkpoint.manifest((self.gate_up_key, self.down_key))

    def load(self, expert_id: int) -> PackedExpert:
        expert_id = int(expert_id)
        if not 0 <= expert_id < EXPERTS:
            raise ValueError(f"expert id {expert_id} outside [0, {EXPERTS})")
        started = time.perf_counter()
        with self._lock:
            fused = self.checkpoint.indexed_tensor(self.gate_up_key, expert_id)
            down = self.checkpoint.indexed_tensor(self.down_key, expert_id)
        if tuple(fused.shape) != (2 * GLOBAL_INTERMEDIATE, HIDDEN_SIZE):
            raise ValueError(f"checkpoint gate/up shape {tuple(fused.shape)} is invalid")
        if tuple(down.shape) != (HIDDEN_SIZE, GLOBAL_INTERMEDIATE):
            raise ValueError(f"checkpoint down shape {tuple(down.shape)} is invalid")

        gate_up = (
            torch.cat(
                (fused[:GLOBAL_INTERMEDIATE].transpose(0, 1), fused[GLOBAL_INTERMEDIATE:].transpose(0, 1)),
                dim=-1,
            )
            .contiguous()
            .reshape(1, 1, HIDDEN_SIZE, 2 * GLOBAL_INTERMEDIATE)
        )
        down_full = down.transpose(0, 1).contiguous().reshape(1, 1, GLOBAL_INTERMEDIATE, HIDDEN_SIZE)
        owner = expert_id % TP_SIZE
        gate_up_by_rank = tuple(gate_up if rank == owner else self._zero_gate_up for rank in range(TP_SIZE))
        down_by_rank = tuple(down_full if rank == owner else self._zero_down for rank in range(TP_SIZE))

        elapsed = time.perf_counter() - started
        with self._lock:
            self.host_reads += 1
            self.host_bytes += self.checkpoint_bytes_per_expert
            self.read_seconds += elapsed
        return PackedExpert(gate_up_by_rank, down_by_rank)

    def metrics(self) -> dict[str, object]:
        with self._lock:
            return {
                "checkpoint_expert_reads": self.host_reads,
                "checkpoint_bytes": self.host_bytes,
                "checkpoint_read_seconds": self.read_seconds,
            }


@dataclasses.dataclass(frozen=True)
class SlotRecord:
    identity: ExpertIdentity | None
    generation: int
    last_used: int
    valid: bool


@dataclasses.dataclass(frozen=True)
class SlotPlan:
    layer_idx: int
    requested: tuple[int, ...]
    slot_expert_ids: tuple[int, ...]
    active_slots: tuple[bool, ...]
    generations: tuple[int, ...]
    hits: tuple[int, ...]
    misses: tuple[int, ...]
    evictions: tuple[ExpertIdentity, ...]


class ExpertSlotDirectory:
    """Deterministic LRU directory with generation/stale-slot protection."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("expert cache capacity must be positive")
        self.capacity = int(capacity)
        self._records = [SlotRecord(None, 0, 0, False) for _ in range(self.capacity)]
        self._by_identity: dict[ExpertIdentity, int] = {}
        self._clock = 0
        self._lock = threading.RLock()

    @property
    def records(self) -> tuple[SlotRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def waves(self, route_ids: Iterable[int]) -> tuple[tuple[int, ...], ...]:
        unique = _ordered_unique(route_ids)
        return tuple(unique[start : start + self.capacity] for start in range(0, len(unique), self.capacity))

    def ensure(self, layer_idx: int, expert_ids: Iterable[int], loader) -> SlotPlan:
        """Ensure one bounded wave, publishing metadata only after each load.

        ``loader(slot, identity, generation)`` must finish every rank/projection
        upload or raise.  A failed slot is invalidated, so stale bytes can never
        be consumed on a retry.
        """

        requested = _ordered_unique(expert_ids)
        if len(requested) > self.capacity:
            raise ValueError(f"wave has {len(requested)} experts but cache capacity is {self.capacity}")
        for expert_id in requested:
            if not 0 <= expert_id < EXPERTS:
                raise ValueError(f"expert id {expert_id} outside [0, {EXPERTS})")

        identities = tuple(ExpertIdentity(int(layer_idx), expert_id) for expert_id in requested)
        protected = set(identities)
        hits = []
        misses = []
        evictions = []
        with self._lock:
            for identity in identities:
                slot = self._by_identity.get(identity)
                if slot is not None and self._records[slot].valid:
                    self._clock += 1
                    record = self._records[slot]
                    self._records[slot] = dataclasses.replace(record, last_used=self._clock)
                    hits.append(identity.expert_id)
                    continue

                candidates = [
                    index
                    for index, record in enumerate(self._records)
                    if not record.valid or record.identity not in protected
                ]
                if not candidates:
                    raise RuntimeError("slot replacement would evict an expert protected by the current wave")
                slot = min(
                    candidates,
                    key=lambda index: (self._records[index].valid, self._records[index].last_used, index),
                )
                old = self._records[slot]
                if old.identity is not None:
                    self._by_identity.pop(old.identity, None)
                    if old.valid:
                        evictions.append(old.identity)
                generation = old.generation + 1
                self._records[slot] = SlotRecord(None, generation, self._clock, False)
                try:
                    loader(slot, identity, generation)
                except Exception:
                    self._records[slot] = SlotRecord(None, generation, self._clock, False)
                    raise
                self._clock += 1
                self._records[slot] = SlotRecord(identity, generation, self._clock, True)
                self._by_identity[identity] = slot
                misses.append(identity.expert_id)

            active = set(identities)
            plan = SlotPlan(
                layer_idx=int(layer_idx),
                requested=requested,
                slot_expert_ids=tuple(
                    record.identity.expert_id if record.valid and record.identity is not None else 0
                    for record in self._records
                ),
                active_slots=tuple(record.valid and record.identity in active for record in self._records),
                generations=tuple(record.generation for record in self._records),
                hits=tuple(hits),
                misses=tuple(misses),
                evictions=tuple(evictions),
            )
            self.validate(plan)
            return plan

    def ensure_batched(self, layer_idx: int, expert_ids: Iterable[int], batch_loader) -> SlotPlan:
        """Ensure one wave while deferring miss submission to ``batch_loader``.

        Slot selection, generations, LRU clocks, hits, and evictions are
        computed in the same router order as :meth:`ensure`.  The batch loader
        may change only the submission order of the reserved misses.  Miss
        metadata is published only after the whole batch has been submitted;
        on failure every reserved slot remains invalid.
        """

        requested = _ordered_unique(expert_ids)
        if len(requested) > self.capacity:
            raise ValueError(f"wave has {len(requested)} experts but cache capacity is {self.capacity}")
        for expert_id in requested:
            if not 0 <= expert_id < EXPERTS:
                raise ValueError(f"expert id {expert_id} outside [0, {EXPERTS})")

        identities = tuple(ExpertIdentity(int(layer_idx), expert_id) for expert_id in requested)
        protected = set(identities)
        hits = []
        misses = []
        evictions = []
        reserved: list[tuple[int, ExpertIdentity, int, int]] = []
        reserved_slots: set[int] = set()
        with self._lock:
            for identity in identities:
                slot = self._by_identity.get(identity)
                if slot is not None and self._records[slot].valid:
                    self._clock += 1
                    record = self._records[slot]
                    self._records[slot] = dataclasses.replace(record, last_used=self._clock)
                    hits.append(identity.expert_id)
                    continue

                candidates = [
                    index
                    for index, record in enumerate(self._records)
                    if index not in reserved_slots and (not record.valid or record.identity not in protected)
                ]
                if not candidates:
                    raise RuntimeError("slot replacement would evict an expert protected by the current wave")
                slot = min(
                    candidates,
                    key=lambda index: (self._records[index].valid, self._records[index].last_used, index),
                )
                old = self._records[slot]
                if old.identity is not None:
                    self._by_identity.pop(old.identity, None)
                    if old.valid:
                        evictions.append(old.identity)
                generation = old.generation + 1
                self._records[slot] = SlotRecord(None, generation, self._clock, False)
                self._clock += 1
                reserved.append((slot, identity, generation, self._clock))
                reserved_slots.add(slot)
                misses.append(identity.expert_id)

            try:
                if reserved:
                    batch_loader(tuple((slot, identity, generation) for slot, identity, generation, _ in reserved))
            except Exception:
                for slot, _identity, generation, _last_used in reserved:
                    self._records[slot] = SlotRecord(None, generation, 0, False)
                raise

            for slot, identity, generation, last_used in reserved:
                self._records[slot] = SlotRecord(identity, generation, last_used, True)
                self._by_identity[identity] = slot

            active = set(identities)
            plan = SlotPlan(
                layer_idx=int(layer_idx),
                requested=requested,
                slot_expert_ids=tuple(
                    record.identity.expert_id if record.valid and record.identity is not None else 0
                    for record in self._records
                ),
                active_slots=tuple(record.valid and record.identity in active for record in self._records),
                generations=tuple(record.generation for record in self._records),
                hits=tuple(hits),
                misses=tuple(misses),
                evictions=tuple(evictions),
            )
            self.validate(plan)
            return plan

    def ensure_ordered(self, layer_idx: int, expert_ids: Iterable[int], loader) -> SlotPlan:
        """Place requested experts in slots 0..N-1 in the given order.

        Segmented decode traces bind the back segment to fixed slot addresses.
        Router-order placement makes those addresses stable while still
        allowing every replay to select a completely different expert set.
        """

        requested = tuple(int(value) for value in expert_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("ordered slot request contains duplicate experts")
        if len(requested) > self.capacity:
            raise ValueError(f"ordered request has {len(requested)} experts but capacity is {self.capacity}")
        if any(not 0 <= expert_id < EXPERTS for expert_id in requested):
            raise ValueError("ordered slot request contains an invalid expert id")
        hits = []
        misses = []
        evictions = []
        with self._lock:
            for slot, expert_id in enumerate(requested):
                identity = ExpertIdentity(int(layer_idx), expert_id)
                old = self._records[slot]
                if old.valid and old.identity == identity:
                    self._clock += 1
                    self._records[slot] = dataclasses.replace(old, last_used=self._clock)
                    hits.append(expert_id)
                    continue

                existing = self._by_identity.pop(identity, None)
                if existing is not None and existing != slot:
                    displaced = self._records[existing]
                    self._records[existing] = SlotRecord(None, displaced.generation + 1, 0, False)
                if old.identity is not None:
                    self._by_identity.pop(old.identity, None)
                    if old.valid:
                        evictions.append(old.identity)
                generation = old.generation + 1
                self._records[slot] = SlotRecord(None, generation, self._clock, False)
                try:
                    loader(slot, identity, generation)
                except Exception:
                    self._records[slot] = SlotRecord(None, generation, self._clock, False)
                    raise
                self._clock += 1
                self._records[slot] = SlotRecord(identity, generation, self._clock, True)
                self._by_identity[identity] = slot
                misses.append(expert_id)

            active_slots = tuple(index < len(requested) for index in range(self.capacity))
            plan = SlotPlan(
                layer_idx=int(layer_idx),
                requested=requested,
                slot_expert_ids=tuple(
                    record.identity.expert_id if record.valid and record.identity is not None else 0
                    for record in self._records
                ),
                active_slots=active_slots,
                generations=tuple(record.generation for record in self._records),
                hits=tuple(hits),
                misses=tuple(misses),
                evictions=tuple(evictions),
            )
            self.validate(plan)
            return plan

    def validate(self, plan: SlotPlan) -> None:
        with self._lock:
            if len(plan.generations) != self.capacity:
                raise RuntimeError("slot plan has wrong capacity")
            present = set()
            for slot, active in enumerate(plan.active_slots):
                if not active:
                    continue
                record = self._records[slot]
                if not record.valid or record.generation != plan.generations[slot] or record.identity is None:
                    raise RuntimeError(f"stale expert slot {slot}")
                if record.identity.layer_idx != plan.layer_idx:
                    raise RuntimeError(f"slot {slot} contains the wrong layer")
                present.add(record.identity.expert_id)
            if present != set(plan.requested):
                raise RuntimeError(f"slot plan covers {sorted(present)}, expected {sorted(plan.requested)}")

    def reset(self) -> None:
        with self._lock:
            self._by_identity.clear()
            self._records = [SlotRecord(None, record.generation + 1, 0, False) for record in self._records]
            self._clock = 0


@dataclasses.dataclass
class ExpertCacheMetrics:
    requests: int = 0
    waves: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    packed_host_hits: int = 0
    packed_host_misses: int = 0
    h2d_bytes: int = 0
    source_pack_seconds: float = 0.0
    h2d_seconds: float = 0.0
    zero_d2d_bytes: int = 0
    deferred_dma_misses: int = 0
    dma_completion_syncs: int = 0
    index_h2d_bytes: int = 0
    index_upload_seconds: float = 0.0
    preload_entries: int = 0
    preload_seconds: float = 0.0
    policy_waves: int = 0
    policy_misses: int = 0
    policy_prepare_seconds: float = 0.0
    policy_submit_seconds: float = 0.0
    policy_h2d_submit_seconds: float = 0.0
    policy_d2d_submit_seconds: float = 0.0
    owner_0_misses: int = 0
    owner_1_misses: int = 0


@dataclasses.dataclass(frozen=True)
class DeviceExpertSlot:
    gate_up: object
    down: object


@dataclasses.dataclass(frozen=True)
class PreparedExpertSlotLoad:
    slot: int
    identity: ExpertIdentity
    generation: int
    packed: object
    gate_shards: tuple[object, ...]
    down_shards: tuple[object, ...]
    staging_index: int


def _replicated_device_zeros(mesh_device, shape: tuple[int, ...], *, dtype):
    """Allocate the full logical tensor on every TP rank.

    ``ttnn.zeros(..., device=mesh_device)`` applies the mesh's default tensor
    distribution.  A leading dimension of one is therefore split unevenly on
    TP2, which is never the contract for rank-local expert or PLE buffers.
    Creating from an explicitly replicated host tensor keeps both shard shapes
    and stable addresses exact.
    """

    import ttnn

    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class QwenDeviceExpertCache:
    """Fixed-address per-expert device slots for one decoder layer.

    Each slot is a separate one-expert tensor.  This is intentional: the
    available ``slice_write`` wrapper untilizes interleaved TILE inputs and
    therefore cannot update BFP4 slices.  Separate tensors retain stable trace
    addresses while copying only actual misses instead of a whole slot bank.
    """

    def __init__(
        self,
        mesh_device,
        source: Qwen38ExpertHostSource,
        *,
        capacity: int = 10,
        packed_host_capacity: int = 64,
        indexed_width: int = 10,
        miss_wave_policy: str | None = None,
        staging_depth: int | None = None,
        packed_dtype: str = "bfp4",
        packed_layout: str = "tile",
        staging_dtype: str = "bfp4",
        staging_layout: str = "tile",
    ):
        import ttnn

        self.mesh_device = mesh_device
        self.source = source
        self.layer_idx = source.layer_idx
        if (packed_dtype, packed_layout, staging_dtype, staging_layout) != ("bfp4", "tile", "bfp4", "tile"):
            raise ValueError(
                "the exact Qwen3.8 host expert ABI currently supports only BFP4 TILE host packing and staging"
            )
        self.packed_dtype = packed_dtype
        self.packed_layout = packed_layout
        self.staging_dtype = staging_dtype
        self.staging_layout = staging_layout
        self.directory = ExpertSlotDirectory(capacity)
        self.capacity = int(capacity)
        self.indexed_width = int(indexed_width)
        if not 1 <= self.indexed_width <= self.capacity:
            raise ValueError("indexed expert width must be positive and no larger than the device cache")
        self.packed_host_capacity = int(packed_host_capacity)
        if self.packed_host_capacity < 0:
            raise ValueError("packed host cache capacity cannot be negative")
        self.miss_wave_policy = miss_wave_policy or os.getenv("QWEN38_HOST_MISS_WAVE_POLICY", "serial")
        if self.miss_wave_policy not in {"serial", "owner_partitioned", "owner_coalesced", "owner_threaded"}:
            raise ValueError(
                "expert miss-wave policy must be serial, owner_partitioned, owner_coalesced, or owner_threaded, "
                f"got {self.miss_wave_policy!r}"
            )
        self.staging_depth = int(
            staging_depth if staging_depth is not None else os.getenv("QWEN38_HOST_STAGING_DEPTH", "1")
        )
        if not 1 <= self.staging_depth <= self.capacity:
            raise ValueError("expert upload staging depth must be between one and cache capacity")
        self.slots = tuple(
            DeviceExpertSlot(
                gate_up=_replicated_device_zeros(
                    mesh_device,
                    (1, 1, HIDDEN_SIZE, 2 * GLOBAL_INTERMEDIATE),
                    dtype=ttnn.bfloat4_b,
                ),
                down=_replicated_device_zeros(
                    mesh_device,
                    (1, 1, GLOBAL_INTERMEDIATE, HIDDEN_SIZE),
                    dtype=ttnn.bfloat4_b,
                ),
            )
            for _ in range(self.capacity)
        )
        slot_devices = tuple(shard.device() for shard in ttnn.get_device_tensors(self.slots[0].gate_up))
        if len(slot_devices) != TP_SIZE:
            raise RuntimeError("expert upload staging requires exactly two physical ranks")
        # Fixed rank-local upload pairs prevent transient device allocations
        # from colliding with live trace allocations. Depth one is the selected
        # baseline. Deeper env-gated A/Bs can isolate staging reuse dependency
        # from owner grouping/thread submission without changing slot storage.
        self.upload_by_rank = tuple(
            tuple(
                DeviceExpertSlot(
                    gate_up=ttnn.from_torch(
                        torch.zeros((1, 1, HIDDEN_SIZE, 2 * GLOBAL_INTERMEDIATE), dtype=torch.bfloat16),
                        dtype=ttnn.bfloat4_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                    down=ttnn.from_torch(
                        torch.zeros((1, 1, GLOBAL_INTERMEDIATE, HIDDEN_SIZE), dtype=torch.bfloat16),
                        dtype=ttnn.bfloat4_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                )
                for _ in range(self.staging_depth)
            )
            for device in slot_devices
        )
        # EP2 assigns every routed expert to exactly one rank.  Keep one
        # immutable zero expert on each rank so a miss uploads only the owner
        # weights; the non-owner half is reset with local D2D instead of
        # transferring a known-zero 2.64 MiB shard over PCIe.
        self.zero_by_rank = tuple(
            DeviceExpertSlot(
                gate_up=ttnn.from_torch(
                    torch.zeros((1, 1, HIDDEN_SIZE, 2 * GLOBAL_INTERMEDIATE), dtype=torch.bfloat16),
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                down=ttnn.from_torch(
                    torch.zeros((1, 1, GLOBAL_INTERMEDIATE, HIDDEN_SIZE), dtype=torch.bfloat16),
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            )
            for device in slot_devices
        )
        self.local_indices = ttnn.from_torch(
            torch.arange(self.capacity, dtype=torch.int16).reshape(1, 1, 1, -1),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._packed_zero = (
            ttnn.from_torch(
                source._zero_gate_up,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
            ),
            ttnn.from_torch(
                source._zero_down,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
            ),
        )
        self._packed: OrderedDict[ExpertIdentity, tuple[tuple[object, object], tuple[object, object]]] = OrderedDict()
        self._published_indices: tuple[int, ...] | None = tuple(range(self.capacity))
        self._metrics = ExpertCacheMetrics()
        self._next_staging_by_owner = [0] * TP_SIZE
        self._owner_h2d_executor: ThreadPoolExecutor | None = None
        self._lock = threading.RLock()

    @property
    def device_bytes_per_rank(self) -> int:
        return (self.capacity + 1 + self.staging_depth) * EXPERT_PACKED_BYTES_PER_RANK

    @property
    def packed_host_bytes(self) -> int:
        # Every expert stores one exact owner shard and reuses one common
        # packed-zero shard for its non-owner rank.
        return (len(self._packed) + 1) * EXPERT_PACKED_BYTES_PER_RANK

    def waves(self, route_ids: Iterable[int]) -> tuple[tuple[int, ...], ...]:
        return self.directory.waves(route_ids)

    def preload_packed_host(self) -> dict[str, int | float]:
        """Materialize every exact expert in the bounded packed-host cache.

        This is a model-load operation only: it moves checkpoint read and
        device-native packing out of token service without changing routes,
        slot residency, or any device bytes.  The full 512-entry capacity is
        required so the preload cannot silently evict earlier experts.
        """

        if self.packed_host_capacity < EXPERTS:
            raise ValueError(f"preloading needs packed_host_capacity >= {EXPERTS}")
        started = time.perf_counter()
        before = len(self._packed)
        with self._lock:
            for expert_id in range(EXPERTS):
                self._host_packed(ExpertIdentity(self.layer_idx, expert_id))
        elapsed = time.perf_counter() - started
        loaded = len(self._packed) - before
        self._metrics.preload_entries += loaded
        self._metrics.preload_seconds += elapsed
        return {"loaded_entries": loaded, "seconds": elapsed, "packed_host_entries": len(self._packed)}

    def _host_packed(self, identity: ExpertIdentity):
        import ttnn

        cached = self._packed.get(identity)
        if cached is not None:
            self._packed.move_to_end(identity)
            self._metrics.packed_host_hits += 1
            return cached
        started = time.perf_counter()
        source = self.source.load(identity.expert_id)
        owner = identity.expert_id % TP_SIZE
        ranks = []
        for rank in range(TP_SIZE):
            if rank == owner:
                ranks.append(
                    (
                        ttnn.from_torch(source.gate_up_by_rank[rank], dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT),
                        ttnn.from_torch(source.down_by_rank[rank], dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT),
                    )
                )
            else:
                ranks.append(self._packed_zero)
        packed = tuple(ranks)
        self._metrics.packed_host_misses += 1
        self._metrics.source_pack_seconds += time.perf_counter() - started
        if self.packed_host_capacity:
            self._packed[identity] = packed
            self._packed.move_to_end(identity)
            while len(self._packed) > self.packed_host_capacity:
                self._packed.popitem(last=False)
        return packed

    def _prepare_slot_load(
        self,
        slot: int,
        identity: ExpertIdentity,
        generation: int,
        staging_index: int = 0,
    ) -> PreparedExpertSlotLoad:
        import ttnn

        packed = self._host_packed(identity)
        target = self.slots[slot]
        gate_shards = ttnn.get_device_tensors(target.gate_up)
        down_shards = ttnn.get_device_tensors(target.down)
        if len(gate_shards) != TP_SIZE or len(down_shards) != TP_SIZE:
            raise RuntimeError("expert slots require exactly two device shards")
        return PreparedExpertSlotLoad(slot, identity, generation, packed, gate_shards, down_shards, staging_index)

    def _enqueue_prepared_h2d(self, prepared: PreparedExpertSlotLoad) -> float:
        import ttnn

        owner = prepared.identity.expert_id % TP_SIZE
        started = time.perf_counter()
        # A host copy to an extracted shard broadcasts through its parent
        # mesh.  Upload only the checkpoint-owning shard into its physical
        # staging tensor, then use rank-local D2D for both the owner copy and
        # the exact-zero non-owner reset.  All work stays ordered on CQ0 before
        # the following indexed expert trace.
        staging = self.upload_by_rank[owner][prepared.staging_index]
        ttnn.copy_host_to_device_tensor(prepared.packed[owner][0], staging.gate_up)
        ttnn.copy_host_to_device_tensor(prepared.packed[owner][1], staging.down)
        return time.perf_counter() - started

    def _enqueue_prepared_d2d(self, prepared: PreparedExpertSlotLoad) -> float:
        import ttnn

        owner = prepared.identity.expert_id % TP_SIZE
        non_owner = 1 - owner
        started = time.perf_counter()
        staging = self.upload_by_rank[owner][prepared.staging_index]
        ttnn.copy(staging.gate_up, prepared.gate_shards[owner])
        ttnn.copy(staging.down, prepared.down_shards[owner])
        zero = self.zero_by_rank[non_owner]
        ttnn.copy(zero.gate_up, prepared.gate_shards[non_owner])
        ttnn.copy(zero.down, prepared.down_shards[non_owner])
        # Do not fence each miss.  All uploads, rank-local copies, the compact
        # index update, and the consuming back trace use CQ0.  Completion is
        # therefore observed at the next required route-id read (or the final
        # compact token read), which preserves exactness while allowing a
        # whole layer's miss wave to be queued without per-expert stalls.
        return time.perf_counter() - started

    def _account_submitted(self, prepared: Sequence[PreparedExpertSlotLoad], enqueue_seconds: float) -> None:
        self._metrics.h2d_seconds += enqueue_seconds
        self._metrics.h2d_bytes += len(prepared) * EXPERT_PACKED_BYTES_PER_RANK
        self._metrics.zero_d2d_bytes += len(prepared) * EXPERT_PACKED_BYTES_PER_RANK
        self._metrics.deferred_dma_misses += len(prepared)

    def _owner_executor(self) -> ThreadPoolExecutor:
        if self._owner_h2d_executor is None:
            executor = ThreadPoolExecutor(max_workers=TP_SIZE, thread_name_prefix="qwen-owner-h2d")
            # ThreadPoolExecutor starts workers lazily. Force both workers to
            # exist before any timed wave so lifecycle cost is not mistaken
            # for H2D submission cost.
            barrier = threading.Barrier(TP_SIZE)
            tuple(executor.map(lambda _owner: barrier.wait(), range(TP_SIZE)))
            self._owner_h2d_executor = executor
        return self._owner_h2d_executor

    def _load_slot(self, slot: int, identity: ExpertIdentity, generation: int) -> None:
        owner = identity.expert_id % TP_SIZE
        staging_index = self._next_staging_by_owner[owner] % self.staging_depth
        self._next_staging_by_owner[owner] += 1
        prepared = self._prepare_slot_load(slot, identity, generation, staging_index)
        elapsed = self._enqueue_prepared_h2d(prepared) + self._enqueue_prepared_d2d(prepared)
        self._account_submitted((prepared,), elapsed)

    def _load_slots_owner_partitioned(self, loads: tuple[tuple[int, ExpertIdentity, int], ...]) -> None:
        prepare_started = time.perf_counter()
        owner_offsets = [0] * TP_SIZE
        prepared_rows = []
        for load in loads:
            owner = load[1].expert_id % TP_SIZE
            prepared_rows.append(self._prepare_slot_load(*load, owner_offsets[owner] % self.staging_depth))
            owner_offsets[owner] += 1
        prepared = tuple(prepared_rows)
        self._metrics.policy_prepare_seconds += time.perf_counter() - prepare_started

        by_owner = tuple(sorted(prepared, key=lambda value: value.identity.expert_id % TP_SIZE))
        submit_started = time.perf_counter()
        h2d_seconds = 0.0
        d2d_seconds = 0.0
        if self.miss_wave_policy == "owner_partitioned":
            for item in by_owner:
                h2d_seconds += self._enqueue_prepared_h2d(item)
                d2d_seconds += self._enqueue_prepared_d2d(item)
        else:
            # Coalescing must retain every owner payload until its later D2D.
            # Refuse to alias staging rather than silently corrupt a slot.
            required_depth = max(owner_offsets)
            if required_depth > self.staging_depth:
                raise ValueError(
                    f"{self.miss_wave_policy} needs staging depth >= {required_depth} for this miss wave"
                )
            if self.miss_wave_policy == "owner_threaded":
                owner_rows = tuple(
                    tuple(item for item in by_owner if item.identity.expert_id % TP_SIZE == owner)
                    for owner in range(TP_SIZE)
                )

                def enqueue_owner(rows):
                    return sum(self._enqueue_prepared_h2d(item) for item in rows)

                h2d_seconds = sum(self._owner_executor().map(enqueue_owner, owner_rows))
            else:
                h2d_seconds = sum(self._enqueue_prepared_h2d(item) for item in by_owner)
            # Threaded work above is physical-owner H2D only. Peer-zero D2D is
            # deliberately serialized here because each miss touches both
            # devices, so per-owner D2D workers would not be device-disjoint.
            d2d_seconds = sum(self._enqueue_prepared_d2d(item) for item in prepared)
        self._account_submitted(prepared, h2d_seconds + d2d_seconds)
        self._metrics.policy_submit_seconds += time.perf_counter() - submit_started
        self._metrics.policy_h2d_submit_seconds += h2d_seconds
        self._metrics.policy_d2d_submit_seconds += d2d_seconds
        self._metrics.policy_waves += 1
        self._metrics.policy_misses += len(prepared)
        self._metrics.owner_0_misses += sum(item.identity.expert_id % TP_SIZE == 0 for item in prepared)
        self._metrics.owner_1_misses += sum(item.identity.expert_id % TP_SIZE == 1 for item in prepared)

    def probe_completed_owner_h2d(self, expert_id: int) -> dict[str, int | float]:
        """Time only one packed owner shard's two H2D copies to completion.

        Host lookup/packing happens before the timer.  No slot D2D, peer-zero
        reset, directory/index update, mesh synchronization, or model work is
        included.  A physical-device CQ0 event provides the completion edge,
        and the byte denominator is the actual packed owner payload.
        """

        import ttnn

        identity = ExpertIdentity(self.layer_idx, int(expert_id))
        if not 0 <= identity.expert_id < EXPERTS:
            raise ValueError(f"expert id {identity.expert_id} outside [0, {EXPERTS})")
        with self._lock:
            packed = self._host_packed(identity)
            owner = identity.expert_id % TP_SIZE
            staging = self.upload_by_rank[owner][0]
            started = time.perf_counter()
            ttnn.copy_host_to_device_tensor(packed[owner][0], staging.gate_up)
            ttnn.copy_host_to_device_tensor(packed[owner][1], staging.down)
            enqueued = time.perf_counter()
            completion = ttnn.record_event(staging.gate_up.device(), 0)
            ttnn.event_synchronize(completion)
            completed = time.perf_counter()
        elapsed = completed - started
        return {
            "expert_id": identity.expert_id,
            "owner_rank": owner,
            "owner_h2d_bytes": EXPERT_PACKED_BYTES_PER_RANK,
            "enqueue_seconds": enqueued - started,
            "completion_wait_seconds": completed - enqueued,
            "completed_seconds": elapsed,
            "completed_gb_per_second": EXPERT_PACKED_BYTES_PER_RANK / elapsed / 1e9,
        }

    def probe_completed_dual_owner_h2d(self, expert_ids: tuple[int, int] = (0, 1)) -> dict[str, int | float]:
        """Time one independent packed H2D on each physical rank concurrently."""

        import ttnn

        identities = tuple(ExpertIdentity(self.layer_idx, int(expert_id)) for expert_id in expert_ids)
        if tuple(identity.expert_id % TP_SIZE for identity in identities) != tuple(range(TP_SIZE)):
            raise ValueError("dual-owner H2D probe needs one expert owned by rank 0 followed by one owned by rank 1")
        if any(not 0 <= identity.expert_id < EXPERTS for identity in identities):
            raise ValueError("dual-owner H2D probe expert id is out of range")
        with self._lock:
            packed = tuple(self._host_packed(identity) for identity in identities)

            def enqueue_owner(owner: int) -> None:
                staging = self.upload_by_rank[owner][0]
                ttnn.copy_host_to_device_tensor(packed[owner][owner][0], staging.gate_up)
                ttnn.copy_host_to_device_tensor(packed[owner][owner][1], staging.down)

            started = time.perf_counter()
            tuple(self._owner_executor().map(enqueue_owner, range(TP_SIZE)))
            enqueued = time.perf_counter()
            completions = tuple(
                ttnn.record_event(self.upload_by_rank[owner][0].gate_up.device(), 0) for owner in range(TP_SIZE)
            )
            for completion in completions:
                ttnn.event_synchronize(completion)
            completed = time.perf_counter()
        elapsed = completed - started
        transferred_bytes = TP_SIZE * EXPERT_PACKED_BYTES_PER_RANK
        return {
            "rank_0_expert_id": identities[0].expert_id,
            "rank_1_expert_id": identities[1].expert_id,
            "owner_h2d_bytes": transferred_bytes,
            "enqueue_seconds": enqueued - started,
            "completion_wait_seconds": completed - enqueued,
            "completed_seconds": elapsed,
            "completed_gb_per_second": transferred_bytes / elapsed / 1e9,
        }

    def ensure_wave(self, expert_ids: Iterable[int]) -> SlotPlan:
        with self._lock:
            plan = self.directory.ensure(self.layer_idx, expert_ids, self._load_slot)
            self._metrics.requests += 1
            self._metrics.waves += 1
            self._metrics.hits += len(plan.hits)
            self._metrics.misses += len(plan.misses)
            self._metrics.evictions += len(plan.evictions)
            return plan

    def ensure_ordered(self, expert_ids: Iterable[int]) -> SlotPlan:
        with self._lock:
            plan = self.directory.ensure_ordered(self.layer_idx, expert_ids, self._load_slot)
            self._metrics.requests += 1
            self._metrics.waves += 1
            self._metrics.hits += len(plan.hits)
            self._metrics.misses += len(plan.misses)
            self._metrics.evictions += len(plan.evictions)
            return plan

    def ensure_indexed(self, expert_ids: Iterable[int]) -> SlotPlan:
        """Keep experts in LRU slots and publish router-order slot indices.

        The indexed sparse kernels consume a stable bank of slot addresses and
        a small device-resident index row.  Updating that row lets a route hit
        an expert in any resident slot, avoiding a weight reload solely because
        router order changed between tokens.
        """

        import ttnn

        requested = tuple(int(value) for value in expert_ids)
        if len(requested) != self.indexed_width:
            raise ValueError(f"indexed request needs {self.indexed_width} experts, got {len(requested)}")
        with self._lock:
            if self.miss_wave_policy != "serial":
                plan = self.directory.ensure_batched(
                    self.layer_idx,
                    requested,
                    self._load_slots_owner_partitioned,
                )
            else:
                plan = self.directory.ensure(self.layer_idx, requested, self._load_slot)
            slot_by_expert = {
                expert_id: slot
                for slot, (expert_id, active) in enumerate(zip(plan.slot_expert_ids, plan.active_slots))
                if active
            }
            active_indices = tuple(slot_by_expert[expert_id] for expert_id in requested)
            indices = active_indices + tuple(slot for slot in range(self.capacity) if slot not in active_indices)
            if indices != self._published_indices:
                started = time.perf_counter()
                host_indices = ttnn.from_torch(
                    torch.tensor(indices, dtype=torch.int16).reshape(1, 1, 1, -1),
                    dtype=ttnn.uint16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                index_shards = ttnn.get_device_tensors(self.local_indices)
                if len(index_shards) != TP_SIZE:
                    raise RuntimeError("expert route indices require exactly two device shards")
                # Like PLE, the index row is replicated. A write through one
                # mesh shard broadcasts the same compact row to both ranks and
                # is ordered before the following back trace on CQ0.
                ttnn.copy_host_to_device_tensor(host_indices, index_shards[0])
                self._published_indices = indices
                self._metrics.index_upload_seconds += time.perf_counter() - started
                self._metrics.index_h2d_bytes += TP_SIZE * self.capacity * 2
            self._metrics.requests += 1
            self._metrics.waves += 1
            self._metrics.hits += len(plan.hits)
            self._metrics.misses += len(plan.misses)
            self._metrics.evictions += len(plan.evictions)
            return plan

    def validate(self, plan: SlotPlan) -> None:
        self.directory.validate(plan)

    def reset(self) -> None:
        self.directory.reset()
        self._published_indices = None
        self._next_staging_by_owner = [0] * TP_SIZE

    def metrics(self) -> dict[str, object]:
        return dataclasses.asdict(self._metrics) | {
            "capacity": self.capacity,
            "indexed_width": self.indexed_width,
            "device_bytes_per_rank": self.device_bytes_per_rank,
            "packed_host_entries": len(self._packed),
            "packed_host_bytes": self.packed_host_bytes,
            "miss_wave_policy": self.miss_wave_policy,
            "staging_depth": self.staging_depth,
            "h2d_timing_scope": "host DMA/D2D enqueue; completion is charged at the next route/token boundary",
        }

    def close(self) -> None:
        import ttnn

        if self._owner_h2d_executor is not None:
            self._owner_h2d_executor.shutdown()
            self._owner_h2d_executor = None
        if self.local_indices.is_allocated():
            ttnn.deallocate(self.local_indices)
        for rank_staging in self.upload_by_rank:
            for staging in rank_staging:
                for tensor in (staging.gate_up, staging.down):
                    if tensor.is_allocated():
                        ttnn.deallocate(tensor)
        for zero in self.zero_by_rank:
            for tensor in (zero.gate_up, zero.down):
                if tensor.is_allocated():
                    ttnn.deallocate(tensor)
        for slot in self.slots:
            for tensor in (slot.gate_up, slot.down):
                if tensor.is_allocated():
                    ttnn.deallocate(tensor)
        self._packed.clear()


@dataclasses.dataclass
class PLEMetrics:
    lookup_calls: int = 0
    selected_rows: int = 0
    unique_rows: int = 0
    table_rows_read: int = 0
    table_bytes_read: int = 0
    h2d_bytes: int = 0
    lookup_seconds: float = 0.0


class Qwen38PLEHostStore:
    """Exact mmap-backed Qwen4Exp hashed n-gram embedding lookup."""

    def __init__(
        self,
        checkpoint: SafetensorCheckpoint,
        *,
        layer_idx: int = PLE_LAYER,
        row_cache_capacity: int = 8192,
    ):
        if layer_idx != PLE_LAYER:
            raise ValueError(f"Qwen3.8 PLE exists only on zero-based layer {PLE_LAYER}")
        self.checkpoint = checkpoint
        self.layer_idx = int(layer_idx)
        self.row_cache_capacity = int(row_cache_capacity)
        if self.row_cache_capacity < 0:
            raise ValueError("row cache capacity cannot be negative")
        prefix = f"model.language_model.layers.{layer_idx}.ple.ple_embedding"
        self._prefix = prefix
        self._shard_keys = tuple(f"{prefix}.ngram_embedding.shard_{index}.weight" for index in range(PLE_SHARDS))
        self.layer_multipliers = checkpoint.tensor(f"{prefix}.layer_multipliers").to(torch.int64)
        self.head_vocab_sizes = checkpoint.tensor(f"{prefix}.ngram_heads_vocab_sizes").to(torch.int64)
        self.head_offsets = checkpoint.tensor(f"{prefix}.ngram_heads_offsets").to(torch.int64)
        if self.layer_multipliers.tolist() != [23703573157769, 20109073645365, 8052911324071]:
            raise ValueError("checkpoint PLE multipliers do not match the target revision")
        if int(self.head_offsets[-1] + self.head_vocab_sizes[-1]) != PLE_LOGICAL_ROWS:
            raise ValueError("checkpoint PLE logical row count is inconsistent")

        self.eos_token_id = 248044
        self.context_len = 2
        self._histories: dict[object, torch.Tensor] = {}
        self._row_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._metrics = PLEMetrics()
        self._lock = threading.RLock()
        self._stack = ExitStack()
        handles = {}
        self._tables = []
        for key in self._shard_keys:
            path = checkpoint.path_for(key)
            handle = handles.get(path)
            if handle is None:
                handle = self._stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                handles[path] = handle
            table = handle.get_tensor(key)
            if tuple(table.shape) != (PLE_ROWS_PER_SHARD, PLE_ROW_WIDTH) or table.dtype != torch.bfloat16:
                raise ValueError(f"PLE shard {key} has shape/dtype {tuple(table.shape)}/{table.dtype}")
            self._tables.append(table)

    @property
    def manifest(self) -> tuple[dict[str, object], ...]:
        return self.checkpoint.manifest(self._shard_keys)

    def close(self) -> None:
        self._stack.close()

    def reset_request(self, request_id: object) -> None:
        with self._lock:
            self._histories[request_id] = torch.full((self.context_len,), self.eos_token_id, dtype=torch.int64)

    def cancel_request(self, request_id: object) -> None:
        with self._lock:
            self._histories.pop(request_id, None)

    def _history(self, request_id: object) -> torch.Tensor:
        history = self._histories.get(request_id)
        if history is None:
            history = torch.full((self.context_len,), self.eos_token_id, dtype=torch.int64)
            self._histories[request_id] = history
        return history

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch, length = token_ids.shape
        positions = torch.arange(length, dtype=torch.int64)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat((torch.full((batch, 1), -1, dtype=torch.int64), previous_eos_inclusive[:, :-1]), dim=1)
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        shifted = token_ids.gather(1, source_positions.clamp_min(0).unsqueeze(0).expand(batch, -1))
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, torch.tensor(self.eos_token_id, dtype=torch.int64))

    def row_ids(
        self,
        request_ids: Sequence[object],
        input_ids: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        reset: bool = False,
    ) -> torch.Tensor:
        """Hash logical tokens and advance isolated two-token histories."""

        ids = torch.as_tensor(input_ids, dtype=torch.int64, device="cpu")
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[0] != len(request_ids):
            raise ValueError(f"input ids shape {tuple(ids.shape)} does not match {len(request_ids)} requests")
        if valid_mask is not None:
            mask = torch.as_tensor(valid_mask, dtype=torch.bool, device="cpu")
            if tuple(mask.shape) != tuple(ids.shape):
                raise ValueError("PLE valid mask shape does not match input ids")
            ids = torch.where(mask, ids, torch.tensor(self.eos_token_id, dtype=torch.int64))

        with self._lock:
            if reset:
                for request_id in request_ids:
                    self.reset_request(request_id)
            histories = torch.stack([self._history(request_id) for request_id in request_ids])
            token_history = torch.cat((histories, ids), dim=1)
            shifted = [self._shift_right_ignore_eos(token_history, shift) for shift in range(3)]
            blocks = []
            for ngram in (2, 3):
                start = (ngram - 2) * 8
                stop = start + 8
                mixed = shifted[0] * self.layer_multipliers[0]
                for position in range(1, ngram):
                    mixed = torch.bitwise_xor(mixed, shifted[position] * self.layer_multipliers[position])
                sizes = self.head_vocab_sizes[start:stop]
                offsets = self.head_offsets[start:stop]
                blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes.view(1, 1, -1)) + offsets.view(1, 1, -1))
            result = torch.cat(blocks, dim=-1)[:, -ids.shape[1] :]
            for row, request_id in enumerate(request_ids):
                self._histories[request_id] = token_history[row, -self.context_len :].clone()
            return result

    def lookup_rows(self, row_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(row_ids, dtype=torch.int64, device="cpu")
        if ids.ndim != 3 or ids.shape[-1] != PLE_HEADS:
            raise ValueError(f"PLE row ids must be [batch, seq, {PLE_HEADS}], got {tuple(ids.shape)}")
        if bool(torch.any(ids < 0)) or bool(torch.any(ids >= PLE_LOGICAL_ROWS)):
            raise ValueError("PLE row id outside logical table")
        started = time.perf_counter()
        flat = ids.reshape(-1)
        unique, inverse = torch.unique(flat, sorted=False, return_inverse=True)
        rows: dict[int, torch.Tensor] = {}
        misses = []
        with self._lock:
            for value in unique.tolist():
                cached = self._row_cache.get(value)
                if cached is None:
                    misses.append(value)
                else:
                    self._row_cache.move_to_end(value)
                    rows[value] = cached

            if misses:
                miss_tensor = torch.tensor(misses, dtype=torch.int64)
                shard_ids = torch.div(miss_tensor, PLE_ROWS_PER_SHARD, rounding_mode="floor")
                local_ids = torch.remainder(miss_tensor, PLE_ROWS_PER_SHARD)
                for shard_id in torch.unique(shard_ids).tolist():
                    positions = torch.nonzero(shard_ids == shard_id, as_tuple=False).flatten()
                    selected = self._tables[shard_id].index_select(0, local_ids[positions]).clone()
                    for position, row in zip(positions.tolist(), selected):
                        value = misses[position]
                        rows[value] = row
                        if self.row_cache_capacity:
                            self._row_cache[value] = row
                            self._row_cache.move_to_end(value)
                while len(self._row_cache) > self.row_cache_capacity:
                    self._row_cache.popitem(last=False)

            unique_rows = torch.stack([rows[value] for value in unique.tolist()])
            output = unique_rows[inverse].reshape(*ids.shape, PLE_ROW_WIDTH).flatten(-2).contiguous()
            self._metrics.lookup_calls += 1
            self._metrics.selected_rows += flat.numel()
            self._metrics.unique_rows += unique.numel()
            self._metrics.table_rows_read += len(misses)
            self._metrics.table_bytes_read += len(misses) * PLE_ROW_BYTES
            self._metrics.h2d_bytes += math.prod(ids.shape[:-1]) * PLE_EMBED_DIM * 2
            self._metrics.lookup_seconds += time.perf_counter() - started
            return output

    def prepare(
        self,
        request_ids: Sequence[object],
        input_ids: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        reset: bool = False,
    ) -> torch.Tensor:
        return self.lookup_rows(self.row_ids(request_ids, input_ids, valid_mask=valid_mask, reset=reset))

    def metrics(self) -> dict[str, int | float]:
        with self._lock:
            return dataclasses.asdict(self._metrics) | {
                "row_cache_entries": len(self._row_cache),
                "history_entries": len(self._histories),
            }


class PLEDeviceStaging:
    """Stable replicated PLE inputs updated only outside TT trace capture."""

    def __init__(
        self,
        mesh_device,
        *,
        max_batch: int,
        prefill_rows: int = 128,
        dtype: str = "bf16",
        layout: str = "tile",
    ):
        import ttnn

        if (dtype, layout) != ("bf16", "tile"):
            raise ValueError("the exact Qwen3.8 PLE staging ABI currently supports only BF16 TILE tensors")
        self.mesh_device = mesh_device
        self.max_batch = int(max_batch)
        self.prefill_rows = int(prefill_rows)
        self.dtype = dtype
        self.layout = layout
        self.prefill = _replicated_device_zeros(
            mesh_device,
            (1, 1, self.prefill_rows, PLE_EMBED_DIM),
            dtype=ttnn.bfloat16,
        )
        self.decode = _replicated_device_zeros(
            mesh_device,
            (1, 1, self.max_batch, PLE_EMBED_DIM),
            dtype=ttnn.bfloat16,
        )
        self.h2d_bytes = 0
        self.logical_h2d_bytes = 0
        self.h2d_seconds = 0.0
        self.deferred_uploads = 0
        self.completion_syncs = 0
        # ``copy_host_to_device_tensor`` may outlive the Python call.  Retain
        # the most recent packed host tensor until the next PLE service call;
        # every PLE layer reaches an exact route-id read after consuming it,
        # so that later call is a completion boundary without another fence.
        self._pending_source = None

    def _upload_replicated(self, host: torch.Tensor, target) -> None:
        import ttnn

        # The prior call's PLE projection and following MoE route read have
        # completed on CQ0 before another row can be staged.
        self._pending_source = None
        source = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        shards = ttnn.get_device_tensors(target)
        if len(shards) != TP_SIZE:
            raise RuntimeError("PLE staging requires exactly two device shards")
        # Writes through one shard of a replicated MeshTensor broadcast to the
        # parent mesh.  PLE is identical on both ranks, so one broadcast is the
        # intended TP2 transfer (expert shards deliberately use local D2D).
        ttnn.copy_host_to_device_tensor(source, shards[0])
        self._pending_source = source
        self.deferred_uploads += 1

    def upload_prefill(self, embeddings: torch.Tensor, *, logical: int):
        host = torch.as_tensor(embeddings, dtype=torch.bfloat16, device="cpu")
        if host.ndim == 3:
            host = host.unsqueeze(0)
        expected = (1, 1, int(logical), PLE_EMBED_DIM)
        if tuple(host.shape) != expected or not 1 <= logical <= self.prefill_rows:
            raise ValueError(f"PLE prefill staging expected {expected}, got {tuple(host.shape)}")
        if logical != self.prefill_rows:
            host = torch.nn.functional.pad(host, (0, 0, 0, self.prefill_rows - logical))
        started = time.perf_counter()
        self._upload_replicated(host, self.prefill)
        self.h2d_seconds += time.perf_counter() - started
        self.h2d_bytes += TP_SIZE * self.prefill_rows * PLE_EMBED_DIM * 2
        self.logical_h2d_bytes += TP_SIZE * logical * PLE_EMBED_DIM * 2
        return self.prefill

    def upload_decode(self, embeddings: torch.Tensor):
        host = torch.as_tensor(embeddings, dtype=torch.bfloat16, device="cpu")
        if host.ndim == 3:
            if host.shape[1] != 1:
                raise ValueError("PLE decode lookup must contain exactly one token per request")
            host = host[:, 0, :].reshape(1, 1, host.shape[0], host.shape[2])
        expected = (1, 1, self.max_batch, PLE_EMBED_DIM)
        if tuple(host.shape) != expected:
            raise ValueError(f"PLE decode staging expected {expected}, got {tuple(host.shape)}")
        started = time.perf_counter()
        self._upload_replicated(host, self.decode)
        self.h2d_seconds += time.perf_counter() - started
        self.h2d_bytes += TP_SIZE * self.max_batch * PLE_EMBED_DIM * 2
        self.logical_h2d_bytes += TP_SIZE * self.max_batch * PLE_EMBED_DIM * 2
        return self.decode

    def metrics(self) -> dict[str, object]:
        return {
            "h2d_bytes": self.h2d_bytes,
            "logical_h2d_bytes": self.logical_h2d_bytes,
            "h2d_seconds": self.h2d_seconds,
            "deferred_uploads": self.deferred_uploads,
            "completion_syncs": self.completion_syncs,
            "h2d_timing_scope": "host enqueue; completion is charged at the following exact route-id boundary",
        }

    def close(self) -> None:
        import ttnn

        if self._pending_source is not None:
            ttnn.synchronize_device(self.mesh_device)
            self.completion_syncs += 1
            self._pending_source = None
        for tensor in (self.prefill, self.decode):
            if tensor.is_allocated():
                ttnn.deallocate(tensor)


__all__ = [
    "DeviceExpertSlot",
    "EXPERT_PACKED_BYTES_PER_RANK",
    "ExpertIdentity",
    "ExpertSlotDirectory",
    "PackedExpert",
    "PLEDeviceStaging",
    "PLE_EMBED_DIM",
    "PLE_LOGICAL_ROWS",
    "PLE_PADDED_ROWS",
    "PLE_TABLE_BYTES",
    "QwenDeviceExpertCache",
    "Qwen38ExpertHostSource",
    "Qwen38PLEHostStore",
    "SafetensorCheckpoint",
    "SlotPlan",
]

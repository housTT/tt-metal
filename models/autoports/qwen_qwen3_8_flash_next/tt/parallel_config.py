# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parallelism contracts for Qwen3.8-Flash-Next.

The production target is a four-chip P300 mesh.  Dense projections use TP4,
routed experts use EP4, and the small number of KV heads is replicated where
an even TP4 split is impossible.  Keeping this information in one immutable
object prevents the old TP2 constants from silently leaking into loaders,
cache ownership, or serving validation.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Qwen38ParallelConfig:
    mesh_shape: tuple[int, int]
    dense_tp: int
    expert_parallel: int
    collective_axis: int
    expert_dispatch_axis: int
    kv_replication: int
    indexer_kv_replication: int
    name: str

    def __post_init__(self) -> None:
        devices = self.mesh_shape[0] * self.mesh_shape[1]
        if devices != self.dense_tp or devices != self.expert_parallel:
            raise ValueError("Qwen parallelism requires one dense and expert rank per device")
        if self.mesh_shape[self.collective_axis] != self.dense_tp:
            raise ValueError("dense collective axis does not span every TP rank")
        if self.mesh_shape[self.expert_dispatch_axis] != self.expert_parallel:
            raise ValueError("expert dispatch axis does not span every EP rank")
        if self.dense_tp % self.kv_replication or self.dense_tp % self.indexer_kv_replication:
            raise ValueError("KV replication groups must divide the TP size")

    @property
    def num_devices(self) -> int:
        return self.dense_tp

    def validate_mesh(self, mesh_device) -> None:
        actual = tuple(int(value) for value in mesh_device.shape)
        if actual != self.mesh_shape:
            raise ValueError(f"Qwen3.8 {self.name} requires mesh {self.mesh_shape}, got {actual}")
        actual_devices = int(mesh_device.get_num_devices())
        if actual_devices != self.num_devices:
            raise ValueError(f"Qwen3.8 {self.name} requires {self.num_devices} devices, got {actual_devices}")

    def q_head_range(self, rank: int, num_attention_heads: int) -> tuple[int, int]:
        self._validate_rank(rank)
        if num_attention_heads % self.dense_tp:
            raise ValueError(f"{num_attention_heads} query heads cannot be divided over TP{self.dense_tp}")
        per_rank = num_attention_heads // self.dense_tp
        return rank * per_rank, (rank + 1) * per_rank

    def kv_head_for_rank(self, rank: int, num_key_value_heads: int) -> int:
        """Return the replicated attention KV head owned by ``rank``.

        Qwen has two KV heads on TP4.  Ranks 0-1 share KV0 and ranks 2-3
        share KV1, matching the associated six-query-head slices.
        """

        self._validate_rank(rank)
        if num_key_value_heads * self.kv_replication != self.dense_tp:
            raise ValueError(
                f"{num_key_value_heads} KV heads with replication {self.kv_replication} "
                f"do not cover TP{self.dense_tp}"
            )
        return rank // self.kv_replication

    def expert_owner(self, expert_id: int, num_experts: int = 512) -> int:
        if not 0 <= int(expert_id) < int(num_experts):
            raise ValueError(f"expert id {expert_id} outside [0, {num_experts})")
        return int(expert_id) // (int(num_experts) // self.expert_parallel)

    def expert_local_index(self, expert_id: int, num_experts: int = 512) -> int:
        self.expert_owner(expert_id, num_experts)
        return int(expert_id) % (int(num_experts) // self.expert_parallel)

    def global_expert_id(self, rank: int, local_index: int, num_experts: int = 512) -> int:
        self._validate_rank(rank)
        experts_per_rank = int(num_experts) // self.expert_parallel
        if not 0 <= int(local_index) < experts_per_rank:
            raise ValueError(f"local expert {local_index} outside [0, {experts_per_rank})")
        return int(rank) * experts_per_rank + int(local_index)

    def _validate_rank(self, rank: int) -> None:
        if not 0 <= int(rank) < self.num_devices:
            raise ValueError(f"rank {rank} outside [0, {self.num_devices})")


P300_TP4_EP4 = Qwen38ParallelConfig(
    # DeepSeek's device dispatch/combine kernels currently support their
    # fabric group on mesh axis 0.  A 4x1 view is the same four P300 devices
    # as a 1x4 view, but makes that supported axis span all TP/EP ranks.
    mesh_shape=(4, 1),
    dense_tp=4,
    expert_parallel=4,
    collective_axis=0,
    expert_dispatch_axis=0,
    kv_replication=2,
    indexer_kv_replication=4,
    name="P300 TP4+EP4",
)


__all__ = ["P300_TP4_EP4", "Qwen38ParallelConfig"]

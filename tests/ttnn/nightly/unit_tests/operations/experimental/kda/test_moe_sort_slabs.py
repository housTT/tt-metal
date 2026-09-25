# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness of the prefill routing sort into expert slabs."""

from __future__ import annotations

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576, "trace_region_size": 16_000_000}),
]


def _upload(device, tensor, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("rows", [128, 512])
def test_moe_sort_slabs_groups_rows_by_expert(device: ttnn.Device, rows: int) -> None:
    experts, k, local, rank_base = 512, 10, 128, 128
    generator = torch.Generator().manual_seed(920 + rows)
    logits = torch.randn(rows, experts, generator=generator)
    values, indices = torch.topk(logits, k, dim=-1)
    scores = torch.softmax(values, dim=-1).bfloat16()
    capacity = local + math.ceil(rows * k / 32) + 1

    outputs = ttnn.experimental.kda.moe_sort_slabs(
        _upload(device, indices.to(torch.int32).reshape(1, 1, rows, k), ttnn.uint16),
        _upload(device, scores.reshape(1, 1, rows, k), ttnn.bfloat16),
        _upload(device, torch.full((1, 1, 1, 1), rank_base, dtype=torch.int32), ttnn.int32),
        local,
        capacity,
    )
    slab_rows = ttnn.to_torch(outputs[0]).reshape(-1).to(torch.int64)
    slab_experts = ttnn.to_torch(outputs[1]).reshape(-1).to(torch.int64)
    slab_pos = ttnn.to_torch(outputs[2]).reshape(rows, k).to(torch.int64)
    local_scores = ttnn.to_torch(outputs[3]).reshape(rows, k).float()

    rel = indices - rank_base
    is_local = (rel >= 0) & (rel < local)
    hits = int(is_local.sum())
    used = 0
    for r in range(rows):
        for j in range(k):
            if not is_local[r, j]:
                assert slab_pos[r, j] == capacity * 32 - 1
                assert local_scores[r, j] == 0.0
                continue
            position = int(slab_pos[r, j])
            assert position < capacity * 32 - 1
            assert slab_rows[position] == r, (r, j, position, slab_rows[position])
            assert slab_experts[position // 32] == int(rel[r, j]), (r, j, position)
            assert local_scores[r, j] == scores[r, j].float()
            used += 1
    assert used == hits
    # every used slab holds a single expert and positions are unique
    positions = slab_pos[is_local]
    assert positions.unique().numel() == positions.numel()
    slabs_used = int((slab_pos[is_local] // 32).max()) + 1
    print(f"MOE_SORT rows={rows} hits={hits} slabs_used={slabs_used} capacity={capacity}")

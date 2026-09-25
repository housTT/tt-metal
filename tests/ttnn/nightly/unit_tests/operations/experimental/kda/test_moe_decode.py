# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness of the batch-one MoE routing and weighted-sum kernels."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576, "trace_region_size": 16_000_000}),
]


def _upload(device, tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("rank_base", [0, 128, 384])
def test_moe_route_topk_matches_torch(device: ttnn.Device, rank_base: int) -> None:
    experts, k, local = 512, 10, 128
    generator = torch.Generator().manual_seed(900 + rank_base)
    logits = torch.randn(1, 1, 32, experts, generator=generator).bfloat16()
    row = logits[0, 0, 0].float()
    values, indices = torch.topk(row, k)
    scores = torch.softmax(values, dim=-1)
    rel = indices - rank_base
    valid = (rel >= 0) & (rel < local)
    expected_slots = torch.where(valid, rel, torch.zeros_like(rel)).to(torch.int64)
    expected_scores = torch.where(valid, scores, torch.zeros_like(scores))

    base = _upload(device, torch.full((1, 1, 1, 1), rank_base, dtype=torch.int32), ttnn.int32)
    slots, local_scores = ttnn.experimental.kda.moe_route_topk(_upload(device, logits), base, k, local)
    actual_slots = ttnn.to_torch(slots).reshape(-1).to(torch.int64)
    actual_scores = ttnn.to_torch(local_scores).float().reshape(-1)
    print(f"MOE_ROUTE base={rank_base} slots={actual_slots.tolist()} expected={expected_slots.tolist()}")
    print(f"MOE_ROUTE scores={actual_scores.tolist()} expected={expected_scores.tolist()}")
    assert slots.dtype == ttnn.uint16 and slots.layout == ttnn.ROW_MAJOR_LAYOUT
    assert list(slots.shape) == [1, 1, 1, k]
    assert torch.equal(actual_slots, expected_slots)
    assert torch.allclose(actual_scores, expected_scores, atol=2e-2), (actual_scores, expected_scores)


def test_moe_weighted_sum_matches_torch(device: ttnn.Device) -> None:
    k, width = 10, 2560
    generator = torch.Generator().manual_seed(901)
    groups = torch.randn(1, k, 32, width, generator=generator).bfloat16()
    scores = torch.softmax(torch.randn(1, 1, 1, k, generator=generator), dim=-1).bfloat16()
    expected = (groups.float()[0, :, 0, :] * scores.float().reshape(k, 1)).sum(0)

    out = ttnn.experimental.kda.moe_weighted_sum(_upload(device, groups), _upload(device, scores))
    actual = ttnn.to_torch(out).float()
    pcc = torch.corrcoef(torch.stack((expected.flatten(), actual[0, 0, 0].flatten())))[0, 1].item()
    max_abs = (expected - actual[0, 0, 0]).abs().max().item()
    print(f"MOE_WSUM_PCC out={pcc:.8f} max_abs={max_abs:.5f} other_rows_max={actual[0, 0, 1:].abs().max().item():.5f}")
    assert list(out.shape) == [1, 1, 32, width]
    assert pcc >= 0.999, pcc
    assert actual[0, 0, 1:].abs().max().item() == 0.0

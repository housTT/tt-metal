# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact FP32 residual rounding with compact physical rows for every batch."""

import json

import pytest
import torch

import ttnn

from ..tt.optimized_decoder import OptimizedDecoder
from . import test_functional_decoder as H

pytestmark = H.pytestmark


@pytest.mark.parametrize("batch", range(1, 33))
@pytest.mark.parametrize("public_memory", ["dram", "sharded"])
def test_compact_residual_oracle(mesh_device, batch, public_memory):
    decoder = OptimizedDecoder.__new__(OptimizedDecoder)
    decoder.device = mesh_device
    rng = torch.Generator().manual_seed(812 + batch)
    host_a = torch.randn((batch, 1, 4096), generator=rng).bfloat16()
    host_b = torch.randn((batch, 1, 4096), generator=rng) * 0.02
    a, b = [
        H.to_device(mesh_device, host, dtype=dtype) for host, dtype in ((host_a, ttnn.bfloat16), (host_b, ttnn.float32))
    ]
    if public_memory == "sharded" or batch == 1:
        mem = decoder._width_memory(4096, 32, batch * 32)
        a, b = [ttnn.to_memory_config(value, mem) for value in (a, b)]
    compact = decoder._width_memory(4096, 32)
    original_sum = decoder._residual_sum
    geometry = {}

    def checked_sum(residual, update, memory_config):
        geometry.update(
            logical=list(residual.shape),
            padded=list(residual.padded_shape),
            update_padded=list(update.padded_shape),
            memory=str(memory_config),
        )
        assert list(residual.padded_shape) == [1, 32, 4096]
        assert list(update.padded_shape) == [1, 32, 4096]
        assert residual.memory_config() == compact
        assert update.memory_config() == compact
        return original_sum(residual, update, memory_config)

    decoder._residual_sum = checked_sum

    def forward():
        return decoder._residual_add(a, b, compact)

    expected = (host_a.float() + host_b).bfloat16()
    for _ in range(3):
        out = forward()
        assert torch.equal(expected, ttnn.to_torch(out))
        ttnn.deallocate(out)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = forward()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for _ in range(4):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            assert torch.equal(expected, ttnn.to_torch(out))
        assert torch.equal(host_a, ttnn.to_torch(a))
        assert torch.equal(host_b, ttnn.to_torch(b))
        print(
            "COMPACT_RESIDUAL "
            + json.dumps(
                {
                    "batch": batch,
                    "public_memory": public_memory,
                    "geometry": geometry,
                    "oracle": "exact",
                    "inputs_unchanged": True,
                }
            ),
            flush=True,
        )
    finally:
        ttnn.release_trace(mesh_device, trace)


@pytest.mark.parametrize("batch", range(1, 33))
def test_sdpa_query_geometry(mesh_device, batch):
    from types import SimpleNamespace

    from ..tt.fused_decoder import _height_memory, _rectangular_rope_grid

    cfg = H.hf_config()
    decoder = OptimizedDecoder.__new__(OptimizedDecoder)
    decoder.device = mesh_device
    decoder.cfg = SimpleNamespace(head_dim=cfg.head_dim)
    rng = torch.Generator().manual_seed(1400 + batch)
    host = torch.randn((1, batch, cfg.num_attention_heads, cfg.head_dim), generator=rng).bfloat16()
    query = H.to_device(mesh_device, host)
    producer_grid, users = _rectangular_rope_grid(mesh_device, batch)
    if users == 1:
        query = ttnn.to_memory_config(query, _height_memory(producer_grid, cfg.head_dim))
    before = query.memory_config()
    program_grid = ttnn.CoreCoord(8, 8)
    out = decoder._sdpa_query(query, batch, program_grid)
    if users == 1:
        expected_grid = ttnn.num_cores_to_corerangeset(batch, program_grid, row_wise=True)
        assert out.memory_config() == _height_memory(expected_grid, cfg.head_dim)
    else:
        assert out.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert torch.equal(host, ttnn.to_torch(query))
    assert torch.equal(host, ttnn.to_torch(out))
    print(
        "SDPA_QUERY_GEOMETRY "
        + json.dumps(
            {
                "batch": batch,
                "before": str(before),
                "after": str(out.memory_config()),
                "padded": list(out.padded_shape),
                "program_grid": [8, 8],
                "exact": True,
            }
        ),
        flush=True,
    )

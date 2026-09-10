# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Probe: indexed sparse matmul with a sparse A operand and more than one tile row per group."""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

RUN = os.environ.get("GPT_OSS_120B_SPARSE_PROBE") == "1"


@pytest.mark.skipif(not RUN, reason="set GPT_OSS_120B_SPARSE_PROBE=1")
@pytest.mark.parametrize("rows", [32, 64, 128], ids=["m32", "m64", "m128"])
@pytest.mark.parametrize("b_sparse", [False, True], ids=["dense_a", "sparse_a"])
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_indexed_sparse_matmul_rows(mesh_device, device_params, rows, b_sparse, reset_seeds):
    del device_params, reset_seeds
    experts, k, n, active = 16, 2880, 384, 4
    generator = torch.Generator().manual_seed(1234 + rows)
    ids = [3, 7, 7, 11]
    weights = (torch.randn((1, experts, k, n), generator=generator) * 0.05).to(torch.bfloat16)
    if b_sparse:
        a_host = (torch.randn((1, active, rows, k), generator=generator) * 0.5).to(torch.bfloat16)
        reference = torch.stack([a_host[0, i].float() @ weights[0, e].float() for i, e in enumerate(ids)])
    else:
        a_host = (torch.randn((1, 1, rows, k), generator=generator) * 0.5).to(torch.bfloat16)
        reference = torch.stack([a_host[0, 0].float() @ weights[0, e].float() for e in ids])
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    a = ttnn.from_torch(a_host, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
    b = ttnn.from_torch(weights, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
    indices = ttnn.from_torch(
        torch.tensor(ids, dtype=torch.int32).reshape(1, 1, 1, active),
        device=mesh_device,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mapper,
    )
    sparsity = ttnn.from_torch(
        torch.zeros((1, 1, 1, experts), dtype=torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mapper,
    )
    config = GPTOSSProgramConfig(prefill_gate_up_cores=(3, 4), prefill_gate_up_in0_block_w=30)
    out = ttnn.sparse_matmul(
        a,
        b,
        sparsity=sparsity,
        indices=indices,
        is_input_a_sparse=b_sparse,
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=ttnn.Tile([32, 32]),
        program_config=config.get_prefill_gate_up_config(rows, n, k=k),
        dtype=ttnn.bfloat16,
    )
    ttnn.synchronize_device(mesh_device)
    host = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    host = host.reshape(active, rows, n)
    passing, detail = comp_pcc(reference, host, 0.99)
    print(f"SPARSE_PROBE rows={rows} sparse_a={b_sparse} out_shape={tuple(out.shape)} pcc={detail}")
    assert passing, detail

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Weight-free regression for packed GDN gating across worker tile cycles."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder

pytestmark = [
    pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 32768, "trace_region_size": 2 * 1024 * 1024, "fabric_config": ttnn.FabricConfig.DISABLED}],
        indirect=True,
    ),
]


@pytest.mark.parametrize(
    "geometry", ["tp4_local", "tp2_local", "tp1_local", "below_worker_boundary", "above_worker_boundary"]
)
def test_packed_gdn_epilogue_cycles(mesh_device, geometry):
    """Run the production epilogue, replacing only its unrelated projection."""
    grid = mesh_device.compute_with_storage_grid_size()
    workers = grid.x * grid.y
    heads = {
        "tp4_local": 8,
        "tp2_local": 16,
        "tp1_local": 32,
        "below_worker_boundary": max(1, workers // 4),
        "above_worker_boundary": workers // 4 + 1,
    }[geometry]
    width = heads * 128
    index = torch.arange(width, dtype=torch.float32)
    core_host = (0.01 * torch.sin(index * 0.11)).reshape(1, heads, 1, 128)
    core_host[..., 0] = 0.4
    core_host[0, 0, 0, 0] = 2.59188175201416
    gate_host = (2 * torch.sin(index * 0.17) + 0.7 * torch.cos(index * 0.031)).reshape(1, 1, width)
    gate_host[..., 0] = -12.8125
    gate_host[..., -1] = 10.0625
    gate_host = gate_host.bfloat16()
    weight_host = torch.linspace(0.53125, 0.9609375, 128).reshape(1, 1, 1, 128).bfloat16()

    def upload(value, dtype, memory=ttnn.L1_MEMORY_CONFIG):
        return ttnn.from_torch(
            value,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=memory,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def host(tensor):
        shards = ttnn.get_device_tensors(tensor)
        assert len(shards) == 1
        return ttnn.to_torch(shards[0]).float()

    def projection_identity(tensor, role):
        assert role == "gdn_out"
        # Production frees its gated temporary after projection; return owned storage.
        return ttnn.clone(tensor)

    core = upload(core_host, ttnn.float32)
    gate = upload(gate_host, ttnn.bfloat16)
    weight = upload(weight_host, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)
    decoder = SimpleNamespace(
        device=mesh_device,
        mesh_config=SimpleNamespace(pack_gdn=True),
        cfg=SimpleNamespace(linear_num_value_heads=heads, linear_value_head_dim=128, linear_v_dim=width, norm_eps=1e-6),
        w={"gdn_norm": weight},
        _linear=projection_identity,
    )
    trace, output, normed, baseline = None, None, None, None
    try:
        # Isolate gating from RMS approximation using its actual bounded device output.
        normed = ttnn.rms_norm(core, weight=weight, epsilon=1e-6)
        normalized = host(normed)
        assert torch.isfinite(normalized).all()
        merged_host = normalized.permute(0, 2, 1, 3).reshape(1, 1, width)
        expected = torch.nn.functional.silu(gate_host.float()).bfloat16().float() * merged_host
        ttnn.deallocate(normed)
        normed = None

        def forward():
            # The epilogue owns and consumes core; keep immutable input for replay.
            return MultichipDecoder._gdn_out_head_major(decoder, ttnn.clone(core), gate, 1, 1)

        output = forward()
        eager = host(output)
        assert torch.isfinite(eager).all()
        torch.testing.assert_close(eager, expected, atol=0.05, rtol=0.02)
        ttnn.deallocate(output)
        output = None

        if width // 32 <= workers:
            # Preserve the previously valid fused branch's exact numerical result.
            merged = upload(merged_host, ttnn.float32)
            try:
                baseline = ttnn.multiply(
                    gate, merged, dtype=ttnn.float32, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
                )
                assert torch.equal(eager, host(baseline))
                ttnn.deallocate(baseline)
                baseline = None
            finally:
                ttnn.deallocate(merged)

        # The original failure was identical in eager and traced decode. Check
        # that the guarded path and new temporary also replay without drift.
        output = forward()
        assert torch.equal(eager, host(output))
        ttnn.deallocate(output)
        output = None
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output = forward()
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        for _ in range(2):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            assert torch.equal(eager, host(output))
        assert torch.equal(host(core), core_host)
        assert torch.equal(host(gate), gate_host.float())
    finally:
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)
        for tensor in (output, normed, baseline, core, gate, weight):
            if tensor is not None and tensor.is_allocated():
                ttnn.deallocate(tensor)

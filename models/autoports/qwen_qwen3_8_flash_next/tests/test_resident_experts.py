# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only contracts for the TP4+EP4 resident expert path."""

from __future__ import annotations

import inspect
import os
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import pytest
import ttnn
from tests.ttnn.utils_for_testing import comp_pcc

from models.autoports.qwen_qwen3_8_flash_next.tt.parallel_config import P300_TP4_EP4
from models.autoports.qwen_qwen3_8_flash_next.tt.resident_experts import (
    EXPERTS_PER_DEVICE,
    Qwen38ResidentExpertSource,
    Qwen38ResidentExperts,
    RESIDENT_EXPERT_BYTES_PER_DEVICE,
    RESIDENT_EXPERT_BYTES_PER_DEVICE_PER_LAYER,
    contiguous_loader_to_global_expert,
    contiguous_dispatch_table,
    contiguous_global_expert_table,
)
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tt.host_weight_cache import SafetensorCheckpoint
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import FABRIC_PACKET_BYTES, MultichipDecoder


def _resident_device_params():
    router = ttnn._ttnn.fabric.FabricRouterConfig()
    router.max_packet_payload_size_bytes = FABRIC_PACKET_BYTES
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "fabric_router_config": router,
    }


def test_tp4_ep4_topology_and_replication_contract() -> None:
    config = P300_TP4_EP4
    config.validate_mesh(SimpleNamespace(shape=(4, 1), get_num_devices=lambda: 4))
    assert config.q_head_range(0, 24) == (0, 6)
    assert config.q_head_range(3, 24) == (18, 24)
    assert tuple(config.kv_head_for_rank(rank, 2) for rank in range(4)) == (0, 0, 1, 1)
    assert tuple(config.expert_owner(expert) for expert in (0, 127, 128, 255, 256, 511)) == (0, 0, 1, 1, 2, 3)
    assert tuple(config.global_expert_id(3, local) for local in range(3)) == (384, 385, 386)


def test_contiguous_dispatch_and_local_expert_tables_are_exact() -> None:
    dispatch = contiguous_dispatch_table()
    assert tuple(dispatch.shape) == (1, 513)
    assert torch.equal(dispatch[0, :512], torch.arange(512, dtype=torch.int32) // 128)
    assert int(dispatch[0, 512]) == -1

    global_ids = contiguous_global_expert_table()
    assert tuple(global_ids.shape) == (1, 4, 128)
    for owner in range(4):
        assert torch.equal(global_ids[0, owner], torch.arange(owner * 128, (owner + 1) * 128, dtype=torch.int32))
    assert tuple(contiguous_loader_to_global_expert(index) for index in (0, 127, 128, 255, 511)) == (
        0,
        127,
        128,
        255,
        511,
    )


def test_resident_memory_and_zero_runtime_host_traffic_contract() -> None:
    assert EXPERTS_PER_DEVICE == 128
    assert RESIDENT_EXPERT_BYTES_PER_DEVICE_PER_LAYER == 353_894_400
    assert RESIDENT_EXPERT_BYTES_PER_DEVICE == 16_986_931_200
    runtime = inspect.getsource(Qwen38ResidentExperts.__call__)
    for forbidden in ("to_torch", "copy_host_to_device", "host_expert", "ensure_indexed"):
        assert forbidden not in runtime


class _CheckpointSpy:
    def __init__(self) -> None:
        self.calls = []

    def indexed_tensor(self, key, index):
        self.calls.append((key, int(index)))
        if key.endswith("gate_up_proj"):
            return torch.empty((1280, 2560), dtype=torch.bfloat16, device="meta")
        return torch.empty((2560, 640), dtype=torch.bfloat16, device="meta")

    def manifest(self, keys):
        return tuple({"key": key} for key in keys)


def test_lazy_checkpoint_source_remaps_contiguous_loader_without_full_host_store() -> None:
    checkpoint = _CheckpointSpy()
    source = Qwen38ResidentExpertSource(checkpoint, 7, cache_entries=4)
    for virtual in (0, 128, 256, 384):
        weights = source[virtual]
        assert tuple(weights["gate_proj"].shape) == (640, 2560)
        assert tuple(weights["up_proj"].shape) == (640, 2560)
        assert tuple(weights["down_proj"].shape) == (2560, 640)
    assert tuple(index for _key, index in checkpoint.calls) == (0, 0, 128, 128, 256, 256, 384, 384)
    before = len(checkpoint.calls)
    for virtual in (0, 128, 256, 384):
        source[virtual]
    assert len(checkpoint.calls) == before
    metrics = source.metrics()
    assert metrics["checkpoint_expert_reads"] == 4
    assert metrics["host_cache_entries"] == 4
    assert metrics["expert_host_store_bytes"] == 0
    source.release_host_tensors()
    assert source.metrics()["host_cache_entries"] == 0


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_RESIDENT_EP4_TT") != "1",
    reason="explicit four-device resident EP4 hardware smoke",
)
@pytest.mark.parametrize(
    "device_params",
    [_resident_device_params()],
    indirect=True,
)
def test_resident_ep4_nonzero_decode_and_trace_hardware(bh_1d_mesh_device, device_params) -> None:
    del device_params
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*P300_TP4_EP4.mesh_shape))
    # The fixture applies the router config before opening the mesh; assert the
    # payload contract here so this test cannot silently measure another path.
    assert ttnn._ttnn.fabric.get_tt_fabric_max_payload_size_bytes() == FABRIC_PACKET_BYTES
    checkpoint = SafetensorCheckpoint(H.MODEL_SNAPSHOT)
    source = Qwen38ResidentExpertSource(checkpoint, 0)
    resident = Qwen38ResidentExperts(bh_1d_mesh_device, source, max_batch=1)
    local_global_tables = [
        ttnn.to_torch(shard).reshape(-1).to(torch.int32)
        for shard in ttnn.get_device_tensors(resident.global_expert_table)
    ]
    for rank, table in enumerate(local_global_tables):
        assert torch.equal(table, torch.arange(rank * 128, (rank + 1) * 128, dtype=torch.int32))
    mapper = ttnn.ReplicateTensorToMesh(bh_1d_mesh_device)
    generator = torch.Generator().manual_seed(38)
    x_host = torch.zeros((1, 1, 32, 2560), dtype=torch.bfloat16)
    x_host[0, 0, 0] = torch.randn((2560,), generator=generator, dtype=torch.bfloat16) * 0.05
    x = ttnn.from_torch(
        x_host,
        device=bh_1d_mesh_device,
        mesh_mapper=mapper,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    expert_ids = (0, 127, 128, 255, 256, 383, 384, 509, 510, 511)
    indices_host = torch.tensor(expert_ids, dtype=torch.int16).reshape(1, 1, 1, 10).expand(1, 1, 32, 10)
    indices = ttnn.from_torch(
        indices_host,
        device=bh_1d_mesh_device,
        mesh_mapper=mapper,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
    )
    scores = ttnn.from_torch(
        torch.full((1, 1, 32, 10), 0.1, dtype=torch.bfloat16),
        device=bh_1d_mesh_device,
        mesh_mapper=mapper,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    trace_id = None
    try:
        output = resident(x, indices, scores, logical_rows=1)
        ttnn.synchronize_device(bh_1d_mesh_device)
        assert tuple(output.shape) == (1, 1, 32, 640)
        shards = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(output)]
        actual = torch.cat(shards, dim=-1)[0, 0, 0]

        token = x_host[0, 0, 0].float()
        expected = torch.zeros((2560,), dtype=torch.float32)
        quantized_expected = torch.zeros((2560,), dtype=torch.float32)
        quantized_experts = []
        gate_up_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        down_key = "model.language_model.layers.0.mlp.experts.down_proj"
        for expert_id in expert_ids:
            gate_up = checkpoint.indexed_tensor(gate_up_key, expert_id).float()
            down = checkpoint.indexed_tensor(down_key, expert_id).float()
            gate = F.linear(token, gate_up[:640])
            up = F.linear(token, gate_up[640:])
            expected += 0.1 * F.linear(F.silu(gate) * up, down)
            owner = expert_id // 128
            local = expert_id % 128
            gate_q = ttnn.to_torch(
                ttnn.get_device_tensors(resident.routed_expert.gate_projs[local])[owner]
            ).float()
            up_q = ttnn.to_torch(
                ttnn.get_device_tensors(resident.routed_expert.up_projs[local])[owner]
            ).float()
            down_q = ttnn.to_torch(
                ttnn.get_device_tensors(resident.routed_expert.down_projs[local])[owner]
            ).float()
            hidden_q = F.silu(token @ gate_q) * (token @ up_q)
            expert_q = 0.1 * (hidden_q @ down_q)
            quantized_experts.append(expert_q)
            quantized_expected += expert_q
        passed, message = comp_pcc(quantized_expected, actual, 0.90)
        raw_pcc = comp_pcc(expected, actual, 0)[1]
        quantization_pcc = comp_pcc(expected, quantized_expected, 0)[1]
        cumulative = []
        partial = torch.zeros_like(quantized_expected)
        for expert_q in quantized_experts:
            partial += expert_q
            cumulative.append(comp_pcc(partial, actual, 0)[1])
        shard_pcc = [
            comp_pcc(expected[rank * 640 : (rank + 1) * 640], shard[0, 0, 0], 0)[1]
            for rank, shard in enumerate(shards)
        ]
        assert passed, (
            f"{message}; raw PCC={raw_pcc}; quantization PCC={quantization_pcc}; "
            f"per-shard raw PCC={shard_pcc}; cumulative expert PCC={cumulative}; "
            f"norms expected/quantized/actual="
            f"{expected.norm().item()}/{quantized_expected.norm().item()}/{actual.norm().item()}"
        )
        raw_passed, raw_message = comp_pcc(expected, actual, 0.90)
        assert raw_passed, raw_message
        assert resident.metrics()["expert_weight_h2d_bytes_runtime"] == 0
        assert resident.metrics()["expert_route_d2h_bytes_runtime"] == 0

        ttnn.deallocate(output)
        output = None
        trace_id = ttnn.begin_trace_capture(bh_1d_mesh_device, cq_id=0)
        output = resident(x, indices, scores, logical_rows=1)
        ttnn.end_trace_capture(bh_1d_mesh_device, trace_id, cq_id=0)
        ttnn.mark_corruptible(output)
        ttnn.execute_trace(bh_1d_mesh_device, trace_id, cq_id=0, blocking=True)
        replay_shards = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(output)]
        replay = torch.cat(replay_shards, dim=-1)[0, 0, 0]
        replay_passed, replay_message = comp_pcc(actual, replay, 0.999)
        assert replay_passed, replay_message
        replay_count = 20
        started = time.perf_counter()
        for _ in range(replay_count):
            ttnn.execute_trace(bh_1d_mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(bh_1d_mesh_device)
        replay_ms = (time.perf_counter() - started) * 1000.0 / replay_count
        print(f"PERFEVIDENCE resident_ep4_routed_block_b1_trace_ms={replay_ms:.6f}")
    finally:
        if trace_id is not None:
            ttnn.release_trace(bh_1d_mesh_device, trace_id)
        for tensor in (locals().get("output"), x, indices, scores):
            if isinstance(tensor, ttnn.Tensor) and tensor.is_allocated():
                ttnn.deallocate(tensor)
        resident.close()


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_RESIDENT_EP4_TT") != "1",
    reason="explicit four-device resident EP4 layer integration smoke",
)
@pytest.mark.parametrize(
    "device_params",
    [_resident_device_params()],
    indirect=True,
)
@pytest.mark.parametrize("layer_idx", [0, 3])
def test_resident_ep4_full_layer_decode_hardware(bh_1d_mesh_device, device_params, layer_idx) -> None:
    del device_params
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*P300_TP4_EP4.mesh_shape))
    checkpoint = SafetensorCheckpoint(H.MODEL_SNAPSHOT)
    max_seq_len = 4096 if layer_idx == 3 else 128
    layer = MultichipDecoder.from_checkpoint_resident(
        checkpoint,
        hf_config=H.target_config(),
        layer_idx=layer_idx,
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    generator = torch.Generator().manual_seed(380)
    host = torch.randn((1, 1, 1, 10240), generator=generator, dtype=torch.bfloat16) * 0.05
    grouped = host.reshape(1, 1, 4, 2560)
    hidden = ttnn.from_torch(
        grouped,
        device=bh_1d_mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(bh_1d_mesh_device, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    current_pos = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=bh_1d_mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_1d_mesh_device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    page_table = None
    rot_mats = None
    decode_kwargs = {"current_pos": current_pos}
    if layer_idx == 3:
        page_table = ttnn.from_torch(
            H.shuffled_page_table(max_seq_len),
            device=bh_1d_mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(bh_1d_mesh_device),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cos_host, sin_host = H.rope_tables(max_seq_len)
        rot_mats = tuple(
            ttnn.from_torch(
                table.reshape(1, 1, max_seq_len, -1),
                device=bh_1d_mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(bh_1d_mesh_device),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            for table in (cos_host, sin_host)
        )
        decode_kwargs.update(page_table=page_table, rot_mats=rot_mats)
    output = None
    trace_id = None
    try:
        layer.prepare_decode_state()
        output = layer.decode_forward_fractured(hidden, **decode_kwargs)
        ttnn.synchronize_device(bh_1d_mesh_device)
        assert tuple(output.shape) == (1, 1, 4, 640)
        for shard in ttnn.get_device_tensors(output):
            assert torch.isfinite(ttnn.to_torch(shard).float()).all()
        metrics = layer.resident_experts.metrics()
        assert metrics["expert_weight_h2d_bytes_runtime"] == 0
        assert metrics["expert_route_d2h_bytes_runtime"] == 0

        ttnn.deallocate(output)
        output = None
        trace_id = ttnn.begin_trace_capture(bh_1d_mesh_device, cq_id=0)
        output = layer.decode_forward_fractured(hidden, **decode_kwargs)
        ttnn.end_trace_capture(bh_1d_mesh_device, trace_id, cq_id=0)
        ttnn.mark_corruptible(output)
        ttnn.execute_trace(bh_1d_mesh_device, trace_id, cq_id=0, blocking=True)
        replay_count = 20
        started = time.perf_counter()
        for _ in range(replay_count):
            ttnn.execute_trace(bh_1d_mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(bh_1d_mesh_device)
        replay_ms = (time.perf_counter() - started) * 1000.0 / replay_count
        layer_kind = "gdn" if layer_idx == 0 else "qsa"
        print(f"PERFEVIDENCE resident_ep4_{layer_kind}_layer_b1_trace_ms={replay_ms:.6f}")
    finally:
        if trace_id is not None:
            ttnn.release_trace(bh_1d_mesh_device, trace_id)
        for tensor in (output, hidden, current_pos, page_table, *(rot_mats or ())):
            if isinstance(tensor, ttnn.Tensor) and tensor.is_allocated():
                ttnn.deallocate(tensor)
        layer.close_host_backing()

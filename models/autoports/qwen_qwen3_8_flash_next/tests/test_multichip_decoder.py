# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Structural and hardware gates for the fixed-P300 TP4+EP4 decoder."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tt.host_weight_cache import (
    EXPERT_PACKED_BYTES_PER_RANK,
    Qwen38ExpertHostSource,
    Qwen38PLEHostStore,
    QwenDeviceExpertCache,
    SafetensorCheckpoint,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.model_config import HF_ADVERTISED_CONTEXT
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    FABRIC_PACKET_BYTES,
    HostBackedSegmentedDecodeTrace,
    MultichipDecoder,
    MultichipDecodeStateWorkspace,
    MultichipMemoryPlan,
    _rank_local_config,
    _rank_local_state,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder


def _multichip_device_params(*, trace_region_size=None):
    fabric_router_config = ttnn._ttnn.fabric.FabricRouterConfig()
    fabric_router_config.max_packet_payload_size_bytes = FABRIC_PACKET_BYTES
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "fabric_router_config": fabric_router_config,
    }
    if trace_region_size is not None:
        params["trace_region_size"] = trace_region_size
    return params


LAYER_KINDS = (0, 1, 3)
_legacy_host_backed_tp2_only = pytest.mark.skipif(
    os.getenv("RUN_QWEN38_HOST_BACKED_TP4") != "1",
    reason="explicit TP4 host-backed expert diagnostic; selected production path uses resident EP4 experts",
)


def _replicated_upload(tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
    )


def _rank_zero_host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def _fractured_upload(tensor, mesh_device):
    """Upload R as the stack-internal four-stream/hidden TP4 residual S."""

    grouped = tensor.reshape(1, 1, tensor.shape[-2] * 4, 2560)
    return ttnn.from_torch(
        grouped,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _fractured_host(tensor):
    """Reconstruct the logical replicated residual from two local shards."""

    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]
    grouped = torch.cat(shards, dim=-1)
    return grouped.reshape(1, 1, grouped.shape[-2] // 4, 10240)


def _copy_fractured_input(host_tensor, target, mesh_device):
    grouped = host_tensor.reshape(1, 1, host_tensor.shape[-2] * 4, 2560)
    source = ttnn.from_torch(
        grouped,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.copy_host_to_device_tensor(source, target)


def _decode_state_host(layer):
    tensors = [layer.recurrent_state, *layer.fused_conv_state, *getattr(layer, "fused_ple_conv_state", ())]
    return tuple(_rank_zero_host(tensor).clone() for tensor in tensors)


def _paged_inputs(layer, mesh_device, page_table_host, cos_host, sin_host, seq_len):
    page_table = _replicated_upload(page_table_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    chunk_tables = []
    for start, _, padded in layer.prefill_chunk_plan(seq_len):
        first = start // layer.block_size
        last = (start + padded) // layer.block_size
        chunk_tables.append(
            _replicated_upload(
                page_table_host[:, first:last],
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        )
    cos = _replicated_upload(cos_host.reshape(1, 1, layer.max_seq_len, -1), mesh_device)
    sin = _replicated_upload(sin_host.reshape(1, 1, layer.max_seq_len, -1), mesh_device)
    return page_table, chunk_tables, (cos, sin)


def _capture_block(layer, method_name, captures):
    original = getattr(layer, method_name)

    def wrapped(*args, **kwargs):
        captures[f"{method_name}_input"] = ttnn.clone(args[0], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = original(*args, **kwargs)
        captures[f"{method_name}_output"] = ttnn.clone(output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return output

    setattr(layer, method_name, wrapped)


def _capture_gdn_epilogue(layer, captures):
    original = layer._gdn_epilogue

    def wrapped(core, z, *, public_shape):
        captures["gdn_core"] = ttnn.clone(core, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        captures["gdn_z"] = ttnn.clone(z, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return original(core, z, public_shape=public_shape)

    layer._gdn_epilogue = wrapped


def _capture_routing(layer, captures):
    original = layer._routed_experts

    def wrapped(x, routing):
        captures["routing"] = ttnn.clone(routing, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return original(x, routing)

    layer._routed_experts = wrapped


def test_multichip_class_and_memory_contract():
    assert issubclass(MultichipDecoder, OptimizedDecoder)
    assert MultichipDecoder.TARGET_MESH == (4, 1)
    assert MultichipDecoder.COLLECTIVE_NUM_LINKS == 2
    assert MultichipDecoder.FABRIC_PACKET_BYTES == 8192
    plan = MultichipMemoryPlan()
    assert plan.standard_bfp4_expert_bytes == 16_986_931_200
    assert plan.uniform_bfp2_expert_bytes == 9_437_184_000
    assert plan.standard_bfp4_fits is True
    assert plan.resident_stack_bytes == 25_630_419_968
    assert plan.resident_stack_headroom_bytes == 8_595_100_672
    assert plan.resident_stack_fits is True
    assert plan.max_compressed_bfp4_fraction == pytest.approx(0.2449097278071385)
    assert plan.host_expert_cache_bytes == 1_592_524_800
    assert plan.ple_staging_bytes == 819_200
    assert plan.decode_state_bytes == 260_702_208
    assert plan.prefill_state_bytes == 208_928_768
    assert plan.all_runtime_state_bytes == 469_630_976
    assert plan.transient_l1_state_bytes_per_worker == 120_832
    assert plan.host_backed_stack_bytes == 10_236_013_568
    assert plan.host_backed_stack_fits is True
    assert "resident_contiguous_ep4_bfp4_routed_experts" in MultichipDecoder.OPTIMIZATION_MANIFEST
    assert "parallel_exact_safetensors_ple_pread" in MultichipDecoder.OPTIMIZATION_MANIFEST
    assert "ple_lookup_overlapped_with_ingress_and_layer0" in MultichipDecoder.OPTIMIZATION_MANIFEST
    assert "stack_major_128_token_prefill_microchunks" in MultichipDecoder.OPTIMIZATION_MANIFEST
    assert "two_segment_resident_full_stack_decode_trace" in MultichipDecoder.OPTIMIZATION_MANIFEST
    assert "persistent_resident_expert_tensor_cache" in MultichipDecoder.OPTIMIZATION_MANIFEST
    assert "fused_sigmoid_gates_for_gdn_and_qsa" in MultichipDecoder.OPTIMIZATION_MANIFEST
    source = inspect.getsource(MultichipDecoder.from_state_dict)
    assert '"expert_bfp4_lofi_g10b16_d40b5"' in source


def test_rank_local_config_contract():
    config = H.target_config()
    gdn = _rank_local_config(config, 0).text_config
    qsa = _rank_local_config(config, 3).text_config
    ep = _rank_local_config(config, 0, expert_parallel=True).text_config
    assert gdn.linear_num_key_heads == 16
    assert gdn.linear_num_value_heads == 48
    assert qsa.num_attention_heads == 6
    assert qsa.num_key_value_heads == 1
    assert qsa.indexer_n_heads == 1
    assert qsa.indexer_kv_heads == 1
    for local in (gdn, qsa):
        assert local.moe_intermediate_size == 160
        assert local.shared_expert_intermediate_size == 160
        assert local.num_experts == 512 and local.num_experts_per_tok == 10
        assert local.hidden_size == 2560 and local.hc_count == 4
    assert ep.moe_intermediate_size == 640
    assert ep.shared_expert_intermediate_size == 160


def test_rank_local_checkpoint_shapes_without_allocating_weights():
    # Meta tensors exercise exact shape/slice/packing contracts without a 5 GiB
    # host allocation.  Runtime loading uses real checkpoint tensors.
    state = {
        "mlp.shared_expert.gate_proj.weight": torch.empty(640, 2560, device="meta"),
        "mlp.shared_expert.up_proj.weight": torch.empty(640, 2560, device="meta"),
        "mlp.shared_expert.down_proj.weight": torch.empty(2560, 640, device="meta"),
        "mlp.experts.gate_up_proj": torch.empty(512, 1280, 2560, device="meta"),
        "mlp.experts.down_proj": torch.empty(512, 2560, 640, device="meta"),
        "linear_attn.in_proj_qkv.weight": torch.empty(10240, 2560, device="meta"),
        "linear_attn.in_proj_z.weight": torch.empty(6144, 2560, device="meta"),
        "linear_attn.in_proj_b.weight": torch.empty(48, 2560, device="meta"),
        "linear_attn.in_proj_a.weight": torch.empty(48, 2560, device="meta"),
        "linear_attn.dt_bias": torch.empty(48, device="meta"),
        "linear_attn.A_log": torch.empty(48, device="meta"),
        "linear_attn.conv1d.weight": torch.empty(10240, 1, 4, device="meta"),
        "linear_attn.out_proj.weight": torch.empty(2560, 6144, device="meta"),
        "self_attn.q_proj.weight": torch.empty(12288, 2560, device="meta"),
        "self_attn.k_proj.weight": torch.empty(512, 2560, device="meta"),
        "self_attn.v_proj.weight": torch.empty(512, 2560, device="meta"),
        "self_attn.o_proj.weight": torch.empty(2560, 6144, device="meta"),
        "self_attn.indexer.index_qk_proj.weight": torch.empty(640, 2560, device="meta"),
    }
    for rank in range(4):
        local = _rank_local_state(state, rank, shard_gdn=False)
        assert local["mlp.experts.gate_up_proj"].shape == (512, 320, 2560)
        assert local["mlp.experts.down_proj"].shape == (512, 2560, 160)
        assert local["linear_attn.in_proj_qkv.weight"].shape == (10240, 2560)
        assert local["linear_attn.in_proj_z.weight"].shape == (6144, 2560)
        assert local["linear_attn.conv1d.weight"].shape == (10240, 1, 4)
        assert local["linear_attn.out_proj.weight"].shape == (2560, 6144)
        assert local["self_attn.q_proj.weight"].shape == (3072, 2560)
        assert local["self_attn.k_proj.weight"].shape == (256, 2560)
        assert local["self_attn.o_proj.weight"].shape == (2560, 1536)
        assert local["self_attn.indexer.index_qk_proj.weight"].shape == (256, 2560)


def test_runtime_collective_has_no_host_fallback():
    source = inspect.getsource(MultichipDecoder._all_reduce_block)
    assert "ttnn.all_reduce" in source
    for forbidden in ("torch", "to_torch", "from_torch", "as_tensor"):
        assert forbidden not in source


def test_host_backed_runtime_boundary_whitelist():
    """Tensor math and traced state movement stay TT-only between declared boundaries."""

    tt_only = (
        MultichipDecoder._decode_attention_host,
        MultichipDecoder._decode_back_host,
        MultichipDecoder._commit_newest_gdn_state_direct,
        MultichipDecoder._routed_experts_indexed_ready,
        MultichipDecoder._all_reduce_block,
        MultichipDecoder._reduce_scatter_block,
        MultichipDecoder._hyper_mix,
        MultichipDecoder._hyper_inject,
        MultichipDecoder.decode_forward_fractured,
        MultichipDecoder.prefill_forward_fractured,
        MultichipDecodeStateWorkspace.bind_gdn,
        MultichipDecodeStateWorkspace.bind_ple,
    )
    for method in tt_only:
        source = inspect.getsource(method)
        for forbidden in ("to_torch", "from_torch", "as_tensor", "copy_host_to_device_tensor"):
            assert forbidden not in source, (method.__qualname__, forbidden)
    route_source = inspect.getsource(MultichipDecoder._read_compact_route_ids)
    assert "to_torch" in route_source
    assert "from_torch" not in route_source and "copy_host_to_device_tensor" not in route_source


def test_host_expert_wave_service_precedes_compute_without_debug_fences(monkeypatch):
    """Production wave service stays ordered without host synchronization hooks."""

    events = []
    plans = {17: object(), 23: object()}

    class FakeCache:
        @staticmethod
        def waves(route_ids):
            events.append(("waves", route_ids))
            return ((17,), (23,))

        @staticmethod
        def ensure_wave(wave):
            events.append(("ensure", wave))
            return plans[wave[0]]

        @staticmethod
        def validate(actual):
            events.append(("validate", actual))

    layer = object.__new__(MultichipDecoder)
    layer.host_expert_cache = FakeCache()
    layer._decode_active = False
    layer._host_route_ids = object()
    layer._read_compact_route_ids = lambda: events.append(("read",)) or (17, 23)
    layer._routed_expert_wave = lambda _x, _routing, wave, actual: events.append(("compute", wave, actual)) or object()
    monkeypatch.setattr(ttnn, "add", lambda left, right: events.append(("add", left, right)) or object())
    monkeypatch.setattr(ttnn, "deallocate", lambda value: events.append(("deallocate", value)))
    monkeypatch.setattr(
        ttnn,
        "synchronize_device",
        lambda *_args, **_kwargs: pytest.fail("production wave path must not host-synchronize"),
    )

    layer._routed_experts(object(), object())

    names = [event[0] for event in events]
    assert names[:2] == ["read", "waves"]
    first_ensure = next(index for index, event in enumerate(events) if event[:2] == ("ensure", (17,)))
    first_validate = next(index for index, event in enumerate(events) if event == ("validate", plans[17]))
    first_compute = next(index for index, event in enumerate(events) if event[:2] == ("compute", (17,)))
    second_ensure = next(index for index, event in enumerate(events) if event[:2] == ("ensure", (23,)))
    second_validate = next(index for index, event in enumerate(events) if event == ("validate", plans[23]))
    second_compute = next(index for index, event in enumerate(events) if event[:2] == ("compute", (23,)))
    assert first_ensure < first_validate < first_compute < second_ensure < second_validate < second_compute


def test_route_sparsity_uses_row_major_metadata_reshape_without_debug_scaffolding():
    """Forbid the cached many-page-to-one-page TILE reshape that hung serving."""

    for method in (MultichipDecoder._routed_expert_slot, MultichipDecoder._routed_expert_wave):
        source = inspect.getsource(method)
        max_call = source.index("sparsity = ttnn.max(")
        to_layout_call = source.index("sparsity = ttnn.to_layout(")
        reshape_call = source.index("sparsity = ttnn.reshape(")
        sparse_matmul_call = source.index("gate_up_sparse = ttnn.sparse_matmul(")
        assert max_call < to_layout_call < reshape_call < sparse_matmul_call, method.__qualname__

    wave_source = inspect.getsource(MultichipDecoder._routed_expert_wave)
    assert "metadata-only view" in wave_source

    production_sources = "\n".join(
        inspect.getsource(method)
        for method in (
            MultichipDecoder._read_compact_route_ids,
            MultichipDecoder._wave_route_weights,
            MultichipDecoder._routed_expert_wave,
            MultichipDecoder._routed_experts,
            MultichipDecoder._moe,
            MultichipDecoder.prefill_forward_fractured,
            MultichipDecoder.prefill_forward_host_backed_fractured,
        )
    )
    for forbidden in (
        "QWEN38_HOST_EXPERT_WAVE_FENCE",
        "QWEN38_HOST_EXPERT_WAVE_REUSE_FENCE",
        "QWEN38_HOST_EXPERT_STAGE_FENCE",
        "QWEN38_VLLM_DEBUG_PREFILL_PROGRESS",
        "_debug_expert_",
        "synchronize_device",
    ):
        assert forbidden not in production_sources


def test_host_expert_slots_use_direct_coordinate_h2d(monkeypatch, expect_error):
    """Guard the allocator-safe direct persistent-slot upload path."""

    init_source = inspect.getsource(QwenDeviceExpertCache.__init__)
    assert "self._upload_storage" not in init_source
    assert "self._zero_storage" in init_source
    assert "self.slots[0].gate_up.device_coords()" in init_source
    assert "get_device_tensors(self._zero_storage.gate_up)" in init_source
    assert "device_coords()" in init_source
    assert "MeshCoordinateRange(mesh_device.shape)" not in init_source
    assert "shard.device()" not in init_source
    assert "experimental_to_single_device" not in init_source

    enqueue_source = inspect.getsource(QwenDeviceExpertCache._enqueue_prepared_h2d)
    assert "copy_host_to_device_tensor_at_coordinate" in enqueue_source
    assert "ttnn.copy_host_to_device_tensor(" not in enqueue_source
    for probe in (
        QwenDeviceExpertCache.probe_completed_owner_h2d,
        QwenDeviceExpertCache.probe_completed_dual_owner_h2d,
    ):
        probe_source = inspect.getsource(probe)
        assert "copy_host_to_device_tensor_at_coordinate" in probe_source
        assert "ttnn.copy_host_to_device_tensor(" not in probe_source
        assert "self.directory.reset()" in probe_source
        assert "self._published_indices = None" in probe_source
    close_source = inspect.getsource(QwenDeviceExpertCache.close)
    assert "self._upload_storage" not in close_source

    host_only = ttnn.from_torch(torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "expects a device tensor"):
        ttnn.copy_host_to_device_tensor_at_coordinate(host_only, host_only, ttnn.MeshCoordinate(0, 0))

    calls = []
    monkeypatch.setattr(
        ttnn,
        "copy_host_to_device_tensor_at_coordinate",
        lambda host, device, coordinate: calls.append((host, device, coordinate)),
    )
    cache = object.__new__(QwenDeviceExpertCache)
    cache._rank_coordinates = ("rank-0", "rank-1")
    cache.slots = (SimpleNamespace(gate_up="slot-gate", down="slot-down"),)
    prepared = SimpleNamespace(
        slot=0,
        identity=SimpleNamespace(expert_id=3),
        packed=((None, None), ("owner-gate", "owner-down")),
    )
    assert cache._enqueue_prepared_h2d(prepared) >= 0.0
    assert calls == [
        ("owner-gate", "slot-gate", "rank-1"),
        ("owner-down", "slot-down", "rank-1"),
    ]

    cpp_source = (Path(__file__).resolve().parents[4] / "ttnn/cpp/ttnn-nanobind/operations/core.cpp").read_text()
    primitive = cpp_source.split('"copy_host_to_device_tensor_at_coordinate"', 1)[1].split(
        '"copy_host_to_device_tensor_partial"', 1
    )[0]
    assert "enqueue_write_shards" in primitive
    assert "ShardDataTransfer{coordinate}" in primitive
    assert "get_mesh_tensor().impl().raw_mesh_buffer()" in primitive
    assert "device_storage() =" not in primitive
    assert "non_uniform_data_movement" not in primitive


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_RANK_LOCAL_STAGING_TT") != "1",
    reason="explicit rank-local topology-preserving H2D regression",
)
@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
def test_rank_local_staging_coordinate_write_preserves_parent_topology(bh_1d_mesh_device, device_params):
    """Prove concurrent owner writes touch one rank each without narrowing shared storage."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    staging = _replicated_upload(torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16), mesh_device)
    target = _replicated_upload(torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16), mesh_device)
    coordinates = tuple(staging.device_coords())
    expected_coordinates = [tuple(coordinate) for coordinate in coordinates]
    hosts = tuple(
        ttnn.from_torch(
            torch.full((1, 1, 32, 32), float(rank + 1), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        for rank in range(MultichipDecoder.TP_SIZE)
    )
    staging_shards = tuple(ttnn.get_device_tensors(staging))
    target_shards = tuple(ttnn.get_device_tensors(target))

    assert [tuple(coordinate) for coordinate in staging.device_coords()] == expected_coordinates
    assert [tuple(coordinate) for coordinate in staging.tensor_topology().mesh_coords()] == expected_coordinates
    assert all([tuple(coordinate) for coordinate in host.tensor_topology().mesh_coords()] == [(0, 0)] for host in hosts)
    assert [[tuple(coordinate) for coordinate in shard.device_coords()] for shard in staging_shards] == [
        [expected_coordinates[0]],
        [expected_coordinates[1]],
    ]

    with ThreadPoolExecutor(max_workers=MultichipDecoder.TP_SIZE) as executor:
        futures = tuple(
            executor.submit(ttnn.copy_host_to_device_tensor_at_coordinate, hosts[rank], staging, coordinates[rank])
            for rank in range(MultichipDecoder.TP_SIZE)
        )
        for future in futures:
            future.result()
    for rank in range(MultichipDecoder.TP_SIZE):
        ttnn.copy(staging_shards[rank], target_shards[rank])
    ttnn.synchronize_device(mesh_device)

    assert [tuple(coordinate) for coordinate in staging.device_coords()] == expected_coordinates
    assert [tuple(coordinate) for coordinate in staging.tensor_topology().mesh_coords()] == expected_coordinates
    for rank, (staging_shard, target_shard) in enumerate(zip(staging_shards, target_shards)):
        expected = torch.full((1, 1, 32, 32), float(rank + 1), dtype=torch.bfloat16)
        torch.testing.assert_close(ttnn.to_torch(staging_shard), expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(ttnn.to_torch(target_shard), expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_ROUTE_SPARSITY_TT") != "1",
    reason="explicit repeated route-sparsity ROW_MAJOR-first regression",
)
@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
def test_route_sparsity_row_major_first_reuses_programs_with_fresh_buffers(
    bh_1d_mesh_device,
    device_params,
    record_property,
):
    """Repeat the exact groups=4, E=7 path without the failing TILE reshape."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    cache_entries = []
    output_addresses = []
    for iteration in range(4):
        torch.manual_seed(20260828 + iteration)
        routing_groups_host = torch.randn((1, 4, 32, 7), dtype=torch.bfloat16)
        expected = torch.max(routing_groups_host, dim=2, keepdim=True).values.reshape(1, 1, 4, 7)
        routing_groups = _replicated_upload(routing_groups_host, bh_1d_mesh_device)

        sparsity = ttnn.max(routing_groups, dim=2, keepdim=True)
        sparsity_rm = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
        rm_addresses = tuple(tensor.buffer_address() for tensor in ttnn.get_device_tensors(sparsity_rm))
        sparsity_view = ttnn.reshape(sparsity_rm, (1, 1, 4, 7))
        view_addresses = tuple(tensor.buffer_address() for tensor in ttnn.get_device_tensors(sparsity_view))
        ttnn.synchronize_device(bh_1d_mesh_device)

        assert sparsity_view.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tuple(sparsity_view.shape) == (1, 1, 4, 7)
        assert view_addresses == rm_addresses
        torch.testing.assert_close(_rank_zero_host(sparsity_view), expected, rtol=0.0, atol=0.0)
        cache_entries.append(bh_1d_mesh_device.num_program_cache_entries())
        output_addresses.append(view_addresses)

    assert cache_entries[1:] == [cache_entries[0]] * 3
    record_property("program_cache_entries", str(cache_entries))
    record_property("output_addresses", str(output_addresses))


@pytest.mark.parametrize(
    "seq_len",
    [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049, HF_ADVERTISED_CONTEXT - 1, HF_ADVERTISED_CONTEXT],
)
def test_multichip_prefill_plan_preserves_non_aligned_contract(seq_len):
    layer = object.__new__(MultichipDecoder)
    layer.max_seq_len = 262144
    plan = layer.prefill_chunk_plan(seq_len)
    assert sum(logical for _, logical, _ in plan) == seq_len
    assert plan[-1][1] == ((seq_len - 1) % 128) + 1


@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
def test_multichip_fabric_packet_contract(bh_1d_mesh_device, device_params):
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    actual_packet_bytes = ttnn._ttnn.fabric.get_tt_fabric_max_payload_size_bytes()
    print(
        f"MC_FABRIC_CONTRACT mesh={MultichipDecoder.TARGET_MESH} packet_bytes={actual_packet_bytes} "
        f"collective_num_links={MultichipDecoder.COLLECTIVE_NUM_LINKS}"
    )
    assert actual_packet_bytes == MultichipDecoder.FABRIC_PACKET_BYTES


@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
def test_multichip_layer0_decode_structural_smoke(bh_1d_mesh_device, device_params):
    """Run local GDN/MoE graphs plus their collectives on the target P300."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    layer = MultichipDecoder.from_state_dict(
        None,
        hf_config=H.target_config(),
        layer_idx=0,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=128,
    )
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    hidden = ttnn.from_torch(
        torch.zeros(1, 1, 1, 10240, dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=mapper,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    current_pos = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=mesh_device,
        mesh_mapper=mapper,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    with H.ForbidHostFallback():
        output = layer.decode_forward(hidden, current_pos=current_pos)
    ttnn.synchronize_device(mesh_device)

    rank_outputs = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(output)]
    assert len(rank_outputs) == MultichipDecoder.TP_SIZE
    assert list(output.shape) == [1, 1, 1, 10240]
    assert torch.equal(rank_outputs[0], rank_outputs[1])


@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
@_legacy_host_backed_tp2_only
def test_host_backed_layer0_decode_matches_optimized_reference(bh_1d_mesh_device, device_params):
    """Real route-id D2H, EP4 expert H2D, and TT math match optimized TTNN."""

    torch.manual_seed(20260828)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    state = H.load_real_layer_state(0)
    reference = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=128,
    )
    host_backed = MultichipDecoder.from_checkpoint_host_backed(
        H.MODEL_SNAPSHOT,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=128,
    )
    prompt_hidden = (torch.randn(1, 1, 33, 10240, dtype=torch.bfloat16) * 0.02).contiguous()
    reference_prefill = reference.prefill_forward(_replicated_upload(prompt_hidden, mesh_device))
    host_prefill = host_backed.prefill_forward(_replicated_upload(prompt_hidden, mesh_device))
    ttnn.synchronize_device(mesh_device)
    assert H.pcc(_rank_zero_host(reference_prefill), _rank_zero_host(host_prefill)) >= 0.995
    assert list(host_prefill.shape) == [1, 1, 33, 10240]
    reference.prepare_decode_state()
    host_backed.prepare_decode_state()
    hidden_host = (torch.randn(1, 1, 1, 10240, dtype=torch.bfloat16) * 0.02).contiguous()
    current_pos_host = torch.tensor([33], dtype=torch.int32)
    reference_out = reference.decode_forward(
        _replicated_upload(hidden_host, mesh_device),
        current_pos=_replicated_upload(current_pos_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
    )
    host_out = host_backed.decode_forward(
        _replicated_upload(hidden_host, mesh_device),
        current_pos=_replicated_upload(current_pos_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
    )
    ttnn.synchronize_device(mesh_device)
    first_slot, first_record = next(
        (slot_index, record)
        for slot_index, record in enumerate(host_backed.host_expert_cache.directory.records)
        if record.valid
    )
    packed = host_backed.host_expert_source.load(first_record.identity.expert_id)
    for rank in range(MultichipDecoder.TP_SIZE):
        slot = host_backed.host_expert_cache.slots[first_slot]
        gate_host = ttnn.to_torch(ttnn.get_device_tensors(slot.gate_up)[rank])
        down_host = ttnn.to_torch(ttnn.get_device_tensors(slot.down)[rank])
        assert H.pcc(packed.gate_up_by_rank[rank], gate_host) >= 0.99
        assert H.pcc(packed.down_by_rank[rank], down_host) >= 0.99
    assert len(host_backed.host_setup_expert_shapes) == 6
    assert all(shape[1] == 512 for shape in host_backed.host_setup_expert_shapes)
    expected = _rank_zero_host(reference_out)
    actual = _rank_zero_host(host_out)
    assert H.pcc(expected, actual) >= 0.995
    assert torch.equal(_rank_zero_host(host_out), ttnn.to_torch(ttnn.get_device_tensors(host_out)[1]))
    metrics = host_backed.host_expert_cache.metrics()
    assert metrics["misses"] >= 10 and metrics["h2d_bytes"] == metrics["misses"] * 2_764_800
    # Construction-time slots are already exact zero on every rank, so the
    # first owner upload needs no redundant peer-zero D2D. Exact peer contents
    # were checked against the packed source immediately above.
    assert metrics["zero_d2d_bytes"] == 0
    assert metrics["zero_d2d_resets"] == 0
    assert metrics["zero_d2d_skips"] == metrics["misses"]
    assert metrics["direct_slot_h2d_bytes"] == metrics["h2d_bytes"]
    assert metrics["owner_d2d_bytes"] == 0
    assert metrics["device_bytes_per_rank"] == 30_412_800
    host_backed.close_host_backing()


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_HOST_DMA_BENCH") != "1",
    reason="explicit completed owner-H2D bandwidth benchmark",
)
@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
@_legacy_host_backed_tp2_only
def test_host_backed_completed_cache_service_bandwidth(bh_1d_mesh_device, device_params, record_property):
    """Measure completed, not enqueue-only, exact expert-cache service."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    layer = MultichipDecoder.from_checkpoint_host_backed(
        H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        layer_idx=0,
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=128,
        packed_host_experts=512,
    )
    cache = layer.host_expert_cache
    try:
        preload = cache.preload_packed_host()
        completed = []
        for wave_index in range(20):
            first = wave_index * 12
            before = cache.metrics()
            started = time.perf_counter()
            cache.ensure_indexed(range(first, first + 10))
            enqueued = time.perf_counter()
            ttnn.synchronize_device(bh_1d_mesh_device)
            finished = time.perf_counter()
            elapsed = finished - started
            after = cache.metrics()
            owner_bytes = int(after["h2d_bytes"] - before["h2d_bytes"])
            assert owner_bytes == 10 * 2_764_800
            # Each controlled wave advances expert ids by twelve, preserving the
            # per-slot EP4 owner. The peer shards remain exact zero and all ten
            # peer resets should therefore be skipped.
            assert int(after["zero_d2d_bytes"] - before["zero_d2d_bytes"]) == 0
            assert int(after["zero_d2d_skips"] - before["zero_d2d_skips"]) == 10
            completed.append(
                {
                    "wave": wave_index,
                    "owner_h2d_bytes": owner_bytes,
                    "enqueue_seconds": enqueued - started,
                    "completion_wait_seconds": finished - enqueued,
                    "completed_seconds": elapsed,
                    "owner_gb_per_second": owner_bytes / elapsed / 1e9,
                }
            )
        measured = completed[1:]
        completed_seconds = sorted(row["completed_seconds"] for row in measured)
        enqueue_seconds = sorted(row["enqueue_seconds"] for row in measured)
        completion_wait_seconds = sorted(row["completion_wait_seconds"] for row in measured)
        total_bytes = sum(row["owner_h2d_bytes"] for row in measured)
        aggregate_bandwidth = total_bytes / sum(completed_seconds) / 1e9
        p50_seconds = completed_seconds[(len(completed_seconds) - 1) // 2]
        p95_seconds = completed_seconds[max(0, (95 * len(completed_seconds) + 99) // 100 - 1)]
        enqueue_p50_seconds = enqueue_seconds[(len(enqueue_seconds) - 1) // 2]
        enqueue_p95_seconds = enqueue_seconds[max(0, (95 * len(enqueue_seconds) + 99) // 100 - 1)]
        wait_p50_seconds = completion_wait_seconds[(len(completion_wait_seconds) - 1) // 2]
        wait_p95_seconds = completion_wait_seconds[max(0, (95 * len(completion_wait_seconds) + 99) // 100 - 1)]
        # Untimed all-slot exactness guard after the same-owner waves. This
        # catches staging alias/reordering errors and proves skipped peer-zero
        # copies left every non-owner shard exact zero.
        for expert_id in range(228, 238):
            slot_index, record = next(
                (slot_index, record)
                for slot_index, record in enumerate(cache.directory.records)
                if record.valid and record.identity.expert_id == expert_id
            )
            exact = layer.host_expert_source.load(expert_id)
            slot = cache.slots[slot_index]
            for rank in range(MultichipDecoder.TP_SIZE):
                gate_host = ttnn.to_torch(ttnn.get_device_tensors(slot.gate_up)[rank])
                down_host = ttnn.to_torch(ttnn.get_device_tensors(slot.down)[rank])
                assert H.pcc(exact.gate_up_by_rank[rank], gate_host) >= 0.99
                assert H.pcc(exact.down_by_rank[rank], down_host) >= 0.99
        before_flip = cache.metrics()
        cache.ensure_indexed(range(229, 239))
        ttnn.synchronize_device(bh_1d_mesh_device)
        after_flip = cache.metrics()
        assert int(after_flip["zero_d2d_bytes"] - before_flip["zero_d2d_bytes"]) == 10 * EXPERT_PACKED_BYTES_PER_RANK
        assert int(after_flip["zero_d2d_resets"] - before_flip["zero_d2d_resets"]) == 10
        for expert_id in range(229, 239):
            slot_index, record = next(
                (slot_index, record)
                for slot_index, record in enumerate(cache.directory.records)
                if record.valid and record.identity.expert_id == expert_id
            )
            exact = layer.host_expert_source.load(expert_id)
            slot = cache.slots[slot_index]
            for rank in range(MultichipDecoder.TP_SIZE):
                gate_host = ttnn.to_torch(ttnn.get_device_tensors(slot.gate_up)[rank])
                down_host = ttnn.to_torch(ttnn.get_device_tensors(slot.down)[rank])
                assert H.pcc(exact.gate_up_by_rank[rank], gate_host) >= 0.99
                assert H.pcc(exact.down_by_rank[rank], down_host) >= 0.99
        mean_bandwidth = aggregate_bandwidth
        p50_bandwidth = 10 * EXPERT_PACKED_BYTES_PER_RANK / p50_seconds / 1e9
        print(
            {
                "preload": preload,
                "policy": cache.metrics()["miss_wave_policy"],
                "staging_depth": cache.metrics()["staging_depth"],
                "completed_cache_service": completed,
                "cache_metrics": cache.metrics(),
                "aggregate_owner_h2d_gb_per_second": aggregate_bandwidth,
                "enqueue_p50_seconds": enqueue_p50_seconds,
                "enqueue_p95_seconds": enqueue_p95_seconds,
                "completion_wait_p50_seconds": wait_p50_seconds,
                "completion_wait_p95_seconds": wait_p95_seconds,
                "completed_p50_seconds": p50_seconds,
                "completed_p95_seconds": p95_seconds,
            }
        )
        record_property("completed_owner_h2d_mean_gb_per_second", mean_bandwidth)
        record_property("completed_owner_h2d_p50_gb_per_second", p50_bandwidth)
        record_property("cache_service_policy", cache.metrics()["miss_wave_policy"])
        record_property("cache_service_staging_depth", cache.metrics()["staging_depth"])
        record_property("cache_service_enqueue_p50_seconds", enqueue_p50_seconds)
        record_property("cache_service_enqueue_p95_seconds", enqueue_p95_seconds)
        record_property("cache_service_completion_wait_p50_seconds", wait_p50_seconds)
        record_property("cache_service_completion_wait_p95_seconds", wait_p95_seconds)
        record_property("completed_cache_service_p50_seconds", p50_seconds)
        record_property("completed_cache_service_p95_seconds", p95_seconds)
        record_property("completed_owner_h2d_waves", len(completed))
        assert mean_bandwidth > 0
    finally:
        layer.close_host_backing()


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_HOST_DMA_BENCH") != "1",
    reason="explicit pure completed owner-H2D bandwidth benchmark",
)
@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
@_legacy_host_backed_tp2_only
def test_host_backed_pure_completed_owner_h2d_bandwidth(bh_1d_mesh_device, device_params, record_property):
    """Measure packed source-to-physical-staging H2D and its final event only."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    checkpoint = SafetensorCheckpoint(H.MODEL_SNAPSHOT)
    source = Qwen38ExpertHostSource(checkpoint, layer_idx=0)
    cache = QwenDeviceExpertCache(
        bh_1d_mesh_device,
        source,
        capacity=10,
        packed_host_capacity=32,
        indexed_width=10,
    )
    try:
        # One completed warmup per physical owner removes cache-construction
        # residue from the measured CQ0 event boundary.
        cache.probe_completed_owner_h2d(0)
        cache.probe_completed_owner_h2d(1)
        rows = [cache.probe_completed_owner_h2d(expert_id) for expert_id in range(2, 22)]
        assert all(row["owner_h2d_bytes"] == EXPERT_PACKED_BYTES_PER_RANK for row in rows)
        completed_seconds = sorted(float(row["completed_seconds"]) for row in rows)
        total_bytes = sum(int(row["owner_h2d_bytes"]) for row in rows)
        aggregate_bandwidth = total_bytes / sum(completed_seconds) / 1e9
        p50_seconds = completed_seconds[(len(completed_seconds) - 1) // 2]
        p95_seconds = completed_seconds[max(0, (95 * len(completed_seconds) + 99) // 100 - 1)]
        print(
            {
                "pure_completed_owner_h2d": rows,
                "aggregate_gb_per_second": aggregate_bandwidth,
                "completed_p50_seconds": p50_seconds,
                "completed_p95_seconds": p95_seconds,
                "bytes_per_sample": EXPERT_PACKED_BYTES_PER_RANK,
            }
        )
        record_property("pure_owner_h2d_aggregate_gb_per_second", aggregate_bandwidth)
        record_property("pure_owner_h2d_completed_p50_seconds", p50_seconds)
        record_property("pure_owner_h2d_completed_p95_seconds", p95_seconds)
        assert aggregate_bandwidth > 0
    finally:
        cache.close()


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_HOST_DMA_BENCH") != "1",
    reason="explicit pure completed all-owner H2D bandwidth benchmark",
)
@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
@_legacy_host_backed_tp2_only
def test_host_backed_pure_completed_dual_owner_h2d_bandwidth(
    bh_1d_mesh_device,
    device_params,
    record_property,
):
    """Measure one concurrent source-to-physical-staging H2D per TP4 rank."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    checkpoint = SafetensorCheckpoint(H.MODEL_SNAPSHOT)
    source = Qwen38ExpertHostSource(checkpoint, layer_idx=0)
    cache = QwenDeviceExpertCache(
        bh_1d_mesh_device,
        source,
        capacity=10,
        packed_host_capacity=32,
        indexed_width=10,
    )
    try:
        cache.probe_completed_all_owner_h2d((0, 1, 2, 3))
        rows = [cache.probe_completed_all_owner_h2d((4, 5, 6, 7)) for _ in range(20)]
        transferred_bytes = MultichipDecoder.TP_SIZE * EXPERT_PACKED_BYTES_PER_RANK
        assert all(row["owner_h2d_bytes"] == transferred_bytes for row in rows)
        completed_seconds = sorted(float(row["completed_seconds"]) for row in rows)
        total_bytes = sum(int(row["owner_h2d_bytes"]) for row in rows)
        aggregate_bandwidth = total_bytes / sum(completed_seconds) / 1e9
        p50_seconds = completed_seconds[(len(completed_seconds) - 1) // 2]
        p95_seconds = completed_seconds[max(0, (95 * len(completed_seconds) + 99) // 100 - 1)]
        print(
            {
                "pure_completed_dual_owner_h2d": rows,
                "aggregate_gb_per_second": aggregate_bandwidth,
                "completed_p50_seconds": p50_seconds,
                "completed_p95_seconds": p95_seconds,
                "bytes_per_sample": transferred_bytes,
            }
        )
        record_property("pure_dual_owner_h2d_aggregate_gb_per_second", aggregate_bandwidth)
        record_property("pure_dual_owner_h2d_completed_p50_seconds", p50_seconds)
        record_property("pure_dual_owner_h2d_completed_p95_seconds", p95_seconds)
        assert aggregate_bandwidth > 0
    finally:
        cache.close()


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_PROGRESSING_HF_DIAGNOSTIC") != "1",
    reason="explicit multichip HF trajectory diagnostic",
)
@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
@_legacy_host_backed_tp2_only
def test_host_backed_layer0_progressing_decode_against_hf(bh_1d_mesh_device, device_params):
    """Compare the real host-backed TP4 layer with HF across twelve transitions."""

    torch.manual_seed(20260828)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    state = H.load_real_layer_state(0)
    seq_len, decode_rows = 33, 12
    hidden = (torch.randn(1, seq_len + decode_rows, 10240, dtype=torch.bfloat16) * 0.02).contiguous()
    cos, sin = H.rope_tables(128)
    hf_layer = H.build_hf_layer(config, 0, state)
    expected_prefill = H.hf_forward(hf_layer, hidden[:, :seq_len], cos, sin)

    layer = MultichipDecoder.from_checkpoint_host_backed(
        H.MODEL_SNAPSHOT,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=128,
    )
    actual_prefill = layer.prefill_forward(_replicated_upload(hidden[:, :seq_len].unsqueeze(0), mesh_device))
    ttnn.synchronize_device(mesh_device)
    prefill_pcc = H.pcc(expected_prefill, _rank_zero_host(actual_prefill).squeeze(0))
    ttnn.deallocate(actual_prefill)
    layer.prepare_decode_state()

    pccs = []
    for step in range(decode_rows):
        stop = seq_len + step + 1
        expected = H.hf_forward(hf_layer, hidden[:, :stop], cos, sin)[:, -1:]
        current_pos = _replicated_upload(
            torch.tensor([stop - 1], dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        actual = layer.decode_forward(
            _replicated_upload(hidden[:, stop - 1 : stop].unsqueeze(0), mesh_device),
            current_pos=current_pos,
        )
        ttnn.synchronize_device(mesh_device)
        pccs.append(H.pcc(expected, _rank_zero_host(actual).squeeze(0)))
        ttnn.deallocate(actual)
        ttnn.deallocate(current_pos)
    print(f"PROGRESSING_MC_HF_PCC layer=0 prefill={prefill_pcc:.8f} decode={pccs}")
    assert prefill_pcc >= H.PCC_BAR
    assert min(pccs) >= H.PCC_BAR
    layer.close_host_backing()


@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
@_legacy_host_backed_tp2_only
def test_host_backed_real_ple_decode_matches_optimized_reference(bh_1d_mesh_device, device_params):
    """Real PLE prefill/history/decode plus EP4 experts match optimized TTNN."""

    torch.manual_seed(20260829)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    state = H.load_real_layer_state(1)
    reference = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=1,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=128,
    )
    host_backed = MultichipDecoder.from_checkpoint_host_backed(
        H.MODEL_SNAPSHOT,
        hf_config=config,
        layer_idx=1,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=128,
    )
    reference_store = Qwen38PLEHostStore(SafetensorCheckpoint(H.MODEL_SNAPSHOT), row_cache_capacity=32)
    prompt_ids = torch.tensor([[11, 248044, 17]], dtype=torch.int64)
    prompt_ple = reference_store.prepare(["reference"], prompt_ids, reset=True)
    prompt_hidden = (torch.randn(1, 1, 3, 10240, dtype=torch.bfloat16) * 0.02).contiguous()
    reference_prefill = reference.prefill_forward(
        _replicated_upload(prompt_hidden, mesh_device),
        ple_embeddings=_replicated_upload(prompt_ple.unsqueeze(0), mesh_device),
    )
    host_prefill = host_backed.prefill_forward_host_backed(
        _replicated_upload(prompt_hidden, mesh_device),
        input_ids=prompt_ids,
        request_id="host",
    )
    ttnn.synchronize_device(mesh_device)
    assert H.pcc(_rank_zero_host(reference_prefill), _rank_zero_host(host_prefill)) >= 0.995
    assert list(host_prefill.shape) == [1, 1, 3, 10240]
    reference.prepare_decode_state()
    host_backed.prepare_decode_state()

    token_ids = torch.tensor([[99]], dtype=torch.int64)
    ple_host = reference_store.prepare(["reference"], token_ids)
    hidden_host = (torch.randn(1, 1, 1, 10240, dtype=torch.bfloat16) * 0.02).contiguous()
    current_pos_host = torch.tensor([0], dtype=torch.int32)
    reference_out = reference.decode_forward(
        _replicated_upload(hidden_host, mesh_device),
        current_pos=_replicated_upload(current_pos_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        ple_embeddings=_replicated_upload(ple_host.unsqueeze(0), mesh_device),
    )
    host_out = host_backed.decode_forward_host_backed(
        _replicated_upload(hidden_host, mesh_device),
        input_ids=token_ids,
        request_ids=("host",),
        current_pos=_replicated_upload(current_pos_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
    )
    ttnn.synchronize_device(mesh_device)
    assert torch.equal(_rank_zero_host(host_backed.ple_staging.decode), ple_host.unsqueeze(0))
    assert H.pcc(_rank_zero_host(reference_out), _rank_zero_host(host_out)) >= 0.995
    assert torch.equal(_rank_zero_host(host_out), ttnn.to_torch(ttnn.get_device_tensors(host_out)[1]))
    assert host_backed.host_ple_store.metrics()["table_rows_read"] <= 64
    assert host_backed.ple_staging.metrics()["h2d_bytes"] == 1_320_960
    reference_store.close()
    host_backed.close_host_backing()


@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
@_legacy_host_backed_tp2_only
def test_host_backed_qsa_paged_prefill_decode_matches_optimized_reference(
    bh_1d_mesh_device,
    device_params,
    record_property,
):
    """TP4 host-backed and resident EP4 outputs match on identical QSA inputs."""

    source_root = Path(__file__).parents[1]
    source_files = (
        "tt/host_weight_cache.py",
        "tt/multichip_decoder.py",
        "tt/optimized_decoder.py",
        "tt/parallel_config.py",
        "tt/resident_experts.py",
        "tests/test_multichip_decoder.py",
    )
    source_digest = hashlib.sha256()
    source_hashes = {}
    for relative in source_files:
        payload = (source_root / relative).read_bytes()
        source_hashes[relative] = hashlib.sha256(payload).hexdigest()
        source_digest.update(relative.encode())
        source_digest.update(b"\0")
        source_digest.update(payload)
        source_digest.update(b"\0")
    record_property("source_digest", source_digest.hexdigest())
    record_property("source_digest_files", json.dumps(source_hashes, sort_keys=True))

    torch.manual_seed(20260830)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    state = H.load_real_layer_state(3)
    reference = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    resident = MultichipDecoder.from_checkpoint_resident(
        H.MODEL_SNAPSHOT,
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=4096,
        resident_weight_cache_path=os.environ.get("QWEN38_EXPERT_WEIGHT_CACHE"),
    )
    host_backed = MultichipDecoder.from_checkpoint_host_backed(
        H.MODEL_SNAPSHOT,
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    cos, sin = H.rope_tables(4096)
    page_host = H.shuffled_page_table(4096)
    page, chunk_pages, rot = _paged_inputs(reference, mesh_device, page_host, cos, sin, 3)
    prompt_hidden = (torch.randn(1, 1, 3, 10240, dtype=torch.bfloat16) * 0.02).contiguous()
    reference_prefill = reference.prefill_forward(
        _replicated_upload(prompt_hidden, mesh_device),
        page_table=page,
        page_tables_per_chunk=chunk_pages,
        rot_mats=rot,
    )
    resident_prefill = resident.prefill_forward(
        _replicated_upload(prompt_hidden, mesh_device),
        page_table=page,
        page_tables_per_chunk=chunk_pages,
        rot_mats=rot,
    )
    host_prefill = host_backed.prefill_forward(
        _replicated_upload(prompt_hidden, mesh_device),
        page_table=page,
        page_tables_per_chunk=chunk_pages,
        rot_mats=rot,
    )
    ttnn.synchronize_device(mesh_device)
    reference_prefill_host = _rank_zero_host(reference_prefill)
    resident_prefill_host = _rank_zero_host(resident_prefill)
    host_prefill_host = _rank_zero_host(host_prefill)
    reference_host_prefill_pcc = H.pcc(reference_prefill_host, host_prefill_host)
    resident_host_prefill_pcc = H.pcc(resident_prefill_host, host_prefill_host)
    print(
        "MC_EXPERT_PATH_AB phase=prefill layer=3 "
        f"reference_host_pcc={reference_host_prefill_pcc:.8f} "
        f"resident_host_pcc={resident_host_prefill_pcc:.8f}"
    )
    record_property("reference_host_prefill_pcc", reference_host_prefill_pcc)
    record_property("resident_host_prefill_pcc", resident_host_prefill_pcc)
    assert reference_host_prefill_pcc >= 0.995
    assert resident_host_prefill_pcc >= 0.995
    for resident_cache, host_cache in zip(resident.kv_cache, host_backed.kv_cache):
        for resident_rank, host_rank in zip(
            ttnn.get_device_tensors(resident_cache),
            ttnn.get_device_tensors(host_cache),
        ):
            assert H.pcc(ttnn.to_torch(resident_rank), ttnn.to_torch(host_rank)) >= 0.995
    assert list(host_backed.kv_cache[0].shape) == [64, 1, 64, 256]

    current_pos = _replicated_upload(
        torch.tensor([3], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    decode_hidden = (torch.randn(1, 1, 1, 10240, dtype=torch.bfloat16) * 0.02).contiguous()
    reference_out = reference.decode_forward(
        _replicated_upload(decode_hidden, mesh_device),
        current_pos=current_pos,
        page_table=page,
        rot_mats=rot,
    )
    resident_out = resident.decode_forward(
        _replicated_upload(decode_hidden, mesh_device),
        current_pos=current_pos,
        page_table=page,
        rot_mats=rot,
    )
    host_out = host_backed.decode_forward(
        _replicated_upload(decode_hidden, mesh_device),
        current_pos=current_pos,
        page_table=page,
        rot_mats=rot,
    )
    ttnn.synchronize_device(mesh_device)
    reference_decode_host = _rank_zero_host(reference_out)
    resident_decode_host = _rank_zero_host(resident_out)
    host_decode_host = _rank_zero_host(host_out)
    reference_host_decode_pcc = H.pcc(reference_decode_host, host_decode_host)
    resident_host_decode_pcc = H.pcc(resident_decode_host, host_decode_host)
    print(
        "MC_EXPERT_PATH_AB phase=decode layer=3 "
        f"reference_host_pcc={reference_host_decode_pcc:.8f} "
        f"resident_host_pcc={resident_host_decode_pcc:.8f}"
    )
    record_property("reference_host_decode_pcc", reference_host_decode_pcc)
    record_property("resident_host_decode_pcc", resident_host_decode_pcc)
    assert reference_host_decode_pcc >= 0.995
    assert resident_host_decode_pcc >= 0.995
    assert torch.equal(_rank_zero_host(host_out), ttnn.to_torch(ttnn.get_device_tensors(host_out)[1]))
    assert host_backed.host_expert_cache.metrics()["misses"] >= 10
    record_property("host_expert_metrics", json.dumps(host_backed.host_expert_cache.metrics(), sort_keys=True))
    record_property("resident_expert_metrics", json.dumps(resident.resident_experts.metrics(), sort_keys=True))
    resident.close_host_backing()
    host_backed.close_host_backing()


@pytest.mark.parametrize(
    "device_params",
    [_multichip_device_params(trace_region_size=100_000_000)],
    indirect=True,
)
@_legacy_host_backed_tp2_only
def test_host_backed_segmented_trace_replay_matches_direct_qsa_decode(bh_1d_mesh_device, device_params):
    """Stable QSA front/back traces bracket exact expert service."""

    torch.manual_seed(20260831)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    stress_steps = int(os.environ.get("QWEN38_MC_TRACE_STRESS_STEPS", "4"))
    if stress_steps < 4:
        raise ValueError("QWEN38_MC_TRACE_STRESS_STEPS must preserve the four-token QSA regression prefix")
    steps = []
    for position in range(stress_steps):
        scale = 0.02 if position == 0 else 0.02 + position * 0.01
        hidden = (torch.randn(1, 1, 1, 10240, dtype=torch.bfloat16) * scale).contiguous()
        steps.append((hidden, torch.tensor([position], dtype=torch.int32)))
    page_host = H.shuffled_page_table(4096)
    cos, sin = H.rope_tables(4096)

    direct = MultichipDecoder.from_checkpoint_host_backed(
        H.MODEL_SNAPSHOT,
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    direct.prepare_decode_state()
    expected = []
    expected_routes = []
    page, _, rot = _paged_inputs(direct, mesh_device, page_host, cos, sin, 1)
    direct_ensure_indexed = direct.host_expert_cache.ensure_indexed

    def record_direct_routes(route_ids):
        expected_routes.append(tuple(int(value) for value in route_ids))
        return direct_ensure_indexed(route_ids)

    direct.host_expert_cache.ensure_indexed = record_direct_routes
    for hidden, position in steps:
        output = direct.decode_forward(
            _replicated_upload(hidden, mesh_device),
            current_pos=_replicated_upload(
                position,
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            page_table=page,
            rot_mats=rot,
        )
        ttnn.synchronize_device(mesh_device)
        expected.append(_rank_zero_host(output).clone())
    expected_cache = tuple(_rank_zero_host(tensor).clone() for tensor in (*direct.kv_cache, direct.indexer_cache))
    direct.close_host_backing()

    layer = MultichipDecoder.from_checkpoint_host_backed(
        H.MODEL_SNAPSHOT,
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    layer.prepare_decode_state()
    stable_hidden = _fractured_upload(steps[0][0], mesh_device)
    stable_pos = _replicated_upload(
        steps[0][1],
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    segmented = HostBackedSegmentedDecodeTrace.capture(
        layer,
        stable_hidden,
        current_pos=stable_pos,
        page_table=page,
        rot_mats=rot,
    )
    ttnn.synchronize_device(mesh_device)
    assert H.pcc(expected[0], _fractured_host(segmented.output)) >= 0.995
    assert segmented.last_route_ids == expected_routes[0]

    initial_routes = segmented.last_route_ids
    for step, (next_hidden, next_pos_host) in enumerate(steps[1:], start=1):
        position_host = ttnn.from_torch(next_pos_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        _copy_fractured_input(next_hidden, stable_hidden, mesh_device)
        ttnn.copy_host_to_device_tensor(position_host, stable_pos)
        traced_out = segmented.replay()
        ttnn.synchronize_device(mesh_device)
        assert segmented.last_route_ids == expected_routes[step], step
        assert H.pcc(expected[step], _fractured_host(traced_out)) >= 0.995
        assert segmented.last_timing["total_seconds"] >= segmented.last_timing["expert_service_seconds"]
        assert segmented.last_timing["total_seconds"] >= (
            segmented.last_timing["front_trace_seconds"] + segmented.last_timing["back_trace_seconds"]
        )
    assert segmented.last_route_ids != initial_routes
    assert layer.host_expert_cache.metrics()["requests"] == stress_steps + 1
    segmented.release()
    actual_cache = tuple(_rank_zero_host(tensor) for tensor in (*layer.kv_cache, layer.indexer_cache))
    assert all(torch.equal(reference, actual) for reference, actual in zip(expected_cache, actual_cache))
    layer.close_host_backing()


@pytest.mark.parametrize(
    "device_params",
    [_multichip_device_params(trace_region_size=100_000_000)],
    indirect=True,
)
@pytest.mark.parametrize("layer_idx", (0, 1))
@_legacy_host_backed_tp2_only
def test_host_backed_gdn_segmented_trace_progression(bh_1d_mesh_device, device_params, layer_idx):
    """Progressing GDN/PLE state survives warm capture and changing replay."""

    torch.manual_seed(20260901 + layer_idx)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    base_token_ids = (23, 91, 248044, 7, 248044, 19, 31, 248044)
    stress_steps = int(os.environ.get("QWEN38_MC_TRACE_STRESS_STEPS", str(len(base_token_ids))))
    if stress_steps < len(base_token_ids):
        raise ValueError("QWEN38_MC_TRACE_STRESS_STEPS must preserve the eight-token regression prefix")
    token_ids = tuple(base_token_ids[index % len(base_token_ids)] for index in range(stress_steps))
    steps = []
    for position in range(len(token_ids)):
        scale = 0.02 + position * 0.01
        hidden = (torch.randn(1, 1, 1, 10240, dtype=torch.bfloat16) * scale).contiguous()
        steps.append((hidden, torch.tensor([position + 1], dtype=torch.int32)))
    prefix_hidden = (torch.randn(1, 1, 1, 10240, dtype=torch.bfloat16) * 0.015).contiguous()
    prefix_position = torch.tensor([0], dtype=torch.int32)
    prefix_id = torch.tensor([[17]], dtype=torch.int64)

    direct = MultichipDecoder.from_checkpoint_host_backed(
        H.MODEL_SNAPSHOT,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=128,
    )
    direct.prepare_decode_state()
    if layer_idx == 1:
        direct.decode_forward_host_backed(
            _replicated_upload(prefix_hidden, mesh_device),
            input_ids=prefix_id,
            request_ids=("direct",),
            current_pos=_replicated_upload(
                prefix_position,
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
        )
    else:
        direct.decode_forward(
            _replicated_upload(prefix_hidden, mesh_device),
            current_pos=_replicated_upload(
                prefix_position,
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
        )
    ttnn.synchronize_device(mesh_device)
    expected = []
    expected_routes = []
    expected_states = []
    direct_ensure_indexed = direct.host_expert_cache.ensure_indexed

    def record_direct_routes(route_ids):
        expected_routes.append(tuple(int(value) for value in route_ids))
        return direct_ensure_indexed(route_ids)

    direct.host_expert_cache.ensure_indexed = record_direct_routes
    for step, (hidden, position) in enumerate(steps):
        decode_kwargs = {
            "current_pos": _replicated_upload(position, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        }
        if layer_idx == 1:
            output = direct.decode_forward_host_backed(
                _replicated_upload(hidden, mesh_device),
                input_ids=torch.tensor([[token_ids[step]]], dtype=torch.int64),
                request_ids=("direct",),
                **decode_kwargs,
            )
        else:
            output = direct.decode_forward(_replicated_upload(hidden, mesh_device), **decode_kwargs)
        ttnn.synchronize_device(mesh_device)
        expected.append(_rank_zero_host(output).clone())
        expected_states.append(_decode_state_host(direct))
    direct.close_host_backing()

    layer = MultichipDecoder.from_checkpoint_host_backed(
        H.MODEL_SNAPSHOT,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=128,
    )
    layer.prepare_decode_state()
    if layer_idx == 1:
        layer.decode_forward_host_backed(
            _replicated_upload(prefix_hidden, mesh_device),
            input_ids=prefix_id,
            request_ids=("trace",),
            current_pos=_replicated_upload(
                prefix_position,
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
        )
    else:
        layer.decode_forward(
            _replicated_upload(prefix_hidden, mesh_device),
            current_pos=_replicated_upload(
                prefix_position,
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
        )
    ttnn.synchronize_device(mesh_device)
    original_newest = layer.fused_conv_state[-1]
    original_addresses = tuple(tensor.buffer_address() for tensor in ttnn.get_device_tensors(original_newest))
    original_ids = tuple(tensor.buffer_unique_id() for tensor in ttnn.get_device_tensors(original_newest))
    original_config = (
        original_newest.memory_config(),
        original_newest.dtype,
        original_newest.get_layout(),
        tuple(original_newest.shape),
        tuple(original_newest.padded_shape),
    )

    stable_hidden = _fractured_upload(steps[0][0], mesh_device)
    stable_pos = _replicated_upload(
        steps[0][1],
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    capture_kwargs = {}
    if layer_idx == 1:
        capture_kwargs.update(
            ple_input_ids=torch.tensor([[token_ids[0]]], dtype=torch.int64),
            request_ids=("trace",),
        )
    segmented = HostBackedSegmentedDecodeTrace.capture(layer, stable_hidden, current_pos=stable_pos, **capture_kwargs)
    ttnn.synchronize_device(mesh_device)
    assert layer.fused_conv_state[-1] is original_newest
    assert tuple(tensor.buffer_address() for tensor in ttnn.get_device_tensors(original_newest)) == original_addresses
    assert tuple(tensor.buffer_unique_id() for tensor in ttnn.get_device_tensors(original_newest)) == original_ids
    assert original_config[:3] == (ttnn.DRAM_MEMORY_CONFIG, ttnn.float32, ttnn.TILE_LAYOUT)
    assert isinstance(layer.decode_state_workspace, MultichipDecodeStateWorkspace)
    owned_workspace = layer.decode_state_workspace
    workspace_tensors = (
        owned_workspace.recurrent_state,
        *owned_workspace.conv_state,
        *owned_workspace.ple_conv_state,
    )
    assert owned_workspace._trace_users == 1
    assert (
        original_newest.memory_config(),
        original_newest.dtype,
        original_newest.get_layout(),
        tuple(original_newest.shape),
        tuple(original_newest.padded_shape),
    ) == original_config
    assert original_newest.is_allocated()
    assert segmented.last_route_ids == expected_routes[0]
    assert H.pcc(expected[0], _fractured_host(segmented.output)) >= 0.995
    assert all(
        torch.equal(reference, actual) for reference, actual in zip(expected_states[0], _decode_state_host(layer))
    )

    for step, (next_hidden, next_pos_host) in enumerate(steps[1:], start=1):
        position_host = ttnn.from_torch(next_pos_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        _copy_fractured_input(next_hidden, stable_hidden, mesh_device)
        ttnn.copy_host_to_device_tensor(position_host, stable_pos)
        replay_kwargs = {}
        if layer_idx == 1:
            replay_kwargs.update(
                ple_input_ids=torch.tensor([[token_ids[step]]], dtype=torch.int64),
                request_ids=("trace",),
            )
        traced_out = segmented.replay(**replay_kwargs)
        ttnn.synchronize_device(mesh_device)
        actual_states = _decode_state_host(layer)
        state_equal = tuple(
            torch.equal(reference, actual) for reference, actual in zip(expected_states[step], actual_states)
        )
        assert segmented.last_route_ids == expected_routes[step], step
        assert H.pcc(expected[step], _fractured_host(traced_out)) >= 0.995
        assert all(state_equal), (step, state_equal)
        assert (
            tuple(tensor.buffer_address() for tensor in ttnn.get_device_tensors(original_newest)) == original_addresses
        )
        assert tuple(tensor.buffer_unique_id() for tensor in ttnn.get_device_tensors(original_newest)) == original_ids
        assert layer.fused_conv_state[-1] is original_newest
        assert original_newest.is_allocated()

    segmented.release()
    assert owned_workspace._trace_users == 0
    assert layer.fused_conv_state[-1] is original_newest
    assert tuple(tensor.buffer_address() for tensor in ttnn.get_device_tensors(original_newest)) == original_addresses
    assert tuple(tensor.buffer_unique_id() for tensor in ttnn.get_device_tensors(original_newest)) == original_ids
    assert (
        original_newest.memory_config(),
        original_newest.dtype,
        original_newest.get_layout(),
        tuple(original_newest.shape),
        tuple(original_newest.padded_shape),
    ) == original_config
    assert original_newest.is_allocated()
    layer.close_host_backing()
    assert owned_workspace.closed
    assert all(not tensor.is_allocated() for tensor in workspace_tensors)


@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
@_legacy_host_backed_tp2_only
def test_host_backed_shared_decode_state_workspace_stack(bh_1d_mesh_device, device_params):
    """Two real GDN layers share one fixed L1 workspace and DRAM state."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    workspace = MultichipDecodeStateWorkspace(mesh_device)
    layers = [
        MultichipDecoder.from_checkpoint_host_backed(
            H.MODEL_SNAPSHOT,
            hf_config=H.target_config(),
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_batch=1,
            max_seq_len=128,
            decode_state_workspace=workspace,
        )
        for layer_idx in (0, 1)
    ]
    for layer in layers:
        layer.prepare_decode_state()
        assert layer.decode_state_workspace is workspace
        assert layer._owns_decode_state_workspace is False
        state = (layer.recurrent_state, *layer.fused_conv_state, *getattr(layer, "fused_ple_conv_state", ()))
        assert all(tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG for tensor in state)

    workspace_tensors = (workspace.recurrent_state, *workspace.conv_state, *workspace.ple_conv_state)
    addresses = tuple(
        tuple(shard.buffer_address() for shard in ttnn.get_device_tensors(tensor)) for tensor in workspace_tensors
    )
    unique_ids = tuple(
        tuple(shard.buffer_unique_id() for shard in ttnn.get_device_tensors(tensor)) for tensor in workspace_tensors
    )
    hidden = _replicated_upload(
        torch.randn(1, 1, 1, 10240, generator=torch.Generator().manual_seed(7301)).bfloat16() * 0.02,
        mesh_device,
    )
    current_pos = _replicated_upload(
        torch.tensor([0], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    output = layers[0].decode_forward_fractured(layers[0].fracture_residual(hidden), current_pos=current_pos)
    output = layers[1].decode_forward_host_backed_fractured(
        output,
        input_ids=torch.tensor([[91]], dtype=torch.int64),
        request_ids=("stack-1",),
        current_pos=current_pos,
    )
    ttnn.synchronize_device(mesh_device)
    assert list(output.shape) == [1, 1, 4, 1280]
    assert list(_fractured_host(output).shape) == [1, 1, 1, 10240]
    assert addresses == tuple(
        tuple(shard.buffer_address() for shard in ttnn.get_device_tensors(tensor)) for tensor in workspace_tensors
    )
    assert unique_ids == tuple(
        tuple(shard.buffer_unique_id() for shard in ttnn.get_device_tensors(tensor)) for tensor in workspace_tensors
    )

    for layer in layers:
        layer.close_host_backing()
    assert not workspace.closed
    workspace.close()
    assert all(not tensor.is_allocated() for tensor in workspace_tensors)


@pytest.mark.parametrize(
    "device_params",
    [_multichip_device_params(trace_region_size=200_000_000)],
    indirect=True,
)
@_legacy_host_backed_tp2_only
def test_host_backed_shared_workspace_segmented_trace_stack(bh_1d_mesh_device, device_params):
    """Two live layer traces safely reuse one fixed L1 state workspace."""

    torch.manual_seed(20260911)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    token_ids = (23, 91, 248044, 7)
    steps = tuple(
        (
            (torch.randn(1, 1, 1, 10240, dtype=torch.bfloat16) * (0.02 + 0.01 * step)).contiguous(),
            (torch.randn(1, 1, 1, 10240, dtype=torch.bfloat16) * (0.03 + 0.01 * step)).contiguous(),
            torch.tensor([step], dtype=torch.int32),
        )
        for step in range(len(token_ids))
    )

    expected_outputs = [[], []]
    expected_states = [[], []]
    expected_routes = [[], []]
    oracle_layers = [
        MultichipDecoder.from_checkpoint_host_backed(
            H.MODEL_SNAPSHOT,
            hf_config=H.target_config(),
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_batch=1,
            max_seq_len=128,
        )
        for layer_idx in (0, 1)
    ]
    for layer_idx, layer in enumerate(oracle_layers):
        layer.prepare_decode_state()
        ensure_indexed = layer.host_expert_cache.ensure_indexed

        def record_routes(route_ids, *, _layer_idx=layer_idx, _ensure=ensure_indexed):
            expected_routes[_layer_idx].append(tuple(int(value) for value in route_ids))
            return _ensure(route_ids)

        layer.host_expert_cache.ensure_indexed = record_routes
    for step, (hidden0, hidden1, position) in enumerate(steps):
        pos = _replicated_upload(position, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        outputs = (
            oracle_layers[0].decode_forward(_replicated_upload(hidden0, mesh_device), current_pos=pos),
            oracle_layers[1].decode_forward_host_backed(
                _replicated_upload(hidden1, mesh_device),
                input_ids=torch.tensor([[token_ids[step]]], dtype=torch.int64),
                request_ids=("oracle-1",),
                current_pos=pos,
            ),
        )
        ttnn.synchronize_device(mesh_device)
        for layer_idx, (layer, output) in enumerate(zip(oracle_layers, outputs)):
            expected_outputs[layer_idx].append(_rank_zero_host(output).clone())
            expected_states[layer_idx].append(_decode_state_host(layer))
    for layer in oracle_layers:
        layer.close_host_backing()

    workspace = MultichipDecodeStateWorkspace(mesh_device)
    layers = [
        MultichipDecoder.from_checkpoint_host_backed(
            H.MODEL_SNAPSHOT,
            hf_config=H.target_config(),
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_batch=1,
            max_seq_len=128,
            decode_state_workspace=workspace,
        )
        for layer_idx in (0, 1)
    ]
    for layer in layers:
        layer.prepare_decode_state()
    workspace_tensors = (workspace.recurrent_state, *workspace.conv_state, *workspace.ple_conv_state)
    addresses = tuple(
        tuple(shard.buffer_address() for shard in ttnn.get_device_tensors(tensor)) for tensor in workspace_tensors
    )
    unique_ids = tuple(
        tuple(shard.buffer_unique_id() for shard in ttnn.get_device_tensors(tensor)) for tensor in workspace_tensors
    )
    stable_hidden = [_fractured_upload(steps[0][index], mesh_device) for index in (0, 1)]
    stable_pos = _replicated_upload(steps[0][2], mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    # Warm every distinct capture signature before registering any trace.  In
    # particular, layer 1 compiles the BF16 PLE-state clone variant that must
    # never allocate a persistent program buffer beside layer 0's live trace.
    HostBackedSegmentedDecodeTrace.warm_programs(layers[0], stable_hidden[0], current_pos=stable_pos)
    HostBackedSegmentedDecodeTrace.warm_programs(
        layers[1],
        stable_hidden[1],
        current_pos=stable_pos,
        ple_input_ids=torch.tensor([[token_ids[0]]], dtype=torch.int64),
        request_ids=("trace-1",),
    )
    traces = [
        HostBackedSegmentedDecodeTrace.capture(
            layers[0], stable_hidden[0], current_pos=stable_pos, programs_prepared=True
        ),
        HostBackedSegmentedDecodeTrace.capture(
            layers[1],
            stable_hidden[1],
            current_pos=stable_pos,
            ple_input_ids=torch.tensor([[token_ids[0]]], dtype=torch.int64),
            request_ids=("trace-1",),
            programs_prepared=True,
        ),
    ]
    assert workspace._trace_users == 2
    for layer_idx, trace in enumerate(traces):
        assert trace.last_route_ids == expected_routes[layer_idx][0]
        assert H.pcc(expected_outputs[layer_idx][0], _fractured_host(trace.output)) >= 0.995
        assert all(
            torch.equal(reference, actual)
            for reference, actual in zip(expected_states[layer_idx][0], _decode_state_host(layers[layer_idx]))
        )

    for step, (hidden0, hidden1, position) in enumerate(steps[1:], start=1):
        for host_hidden, target in zip((hidden0, hidden1), stable_hidden):
            _copy_fractured_input(host_hidden, target, mesh_device)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(position, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), stable_pos
        )
        outputs = (
            traces[0].replay(),
            traces[1].replay(
                ple_input_ids=torch.tensor([[token_ids[step]]], dtype=torch.int64), request_ids=("trace-1",)
            ),
        )
        ttnn.synchronize_device(mesh_device)
        for layer_idx, output in enumerate(outputs):
            assert traces[layer_idx].last_route_ids == expected_routes[layer_idx][step]
            assert H.pcc(expected_outputs[layer_idx][step], _fractured_host(output)) >= 0.995
            assert all(
                torch.equal(reference, actual)
                for reference, actual in zip(expected_states[layer_idx][step], _decode_state_host(layers[layer_idx]))
            )
        assert addresses == tuple(
            tuple(shard.buffer_address() for shard in ttnn.get_device_tensors(tensor)) for tensor in workspace_tensors
        )
        assert unique_ids == tuple(
            tuple(shard.buffer_unique_id() for shard in ttnn.get_device_tensors(tensor)) for tensor in workspace_tensors
        )

    for trace in reversed(traces):
        trace.release()
    assert workspace._trace_users == 0
    for layer in layers:
        layer.close_host_backing()
    workspace.close()


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
def test_multichip_real_weights_match_optimized_baseline(bh_1d_mesh_device, device_params, layer_idx):
    """Direct TTNN optimized-baseline PCC for GDN, PLE+GDN and QSA."""

    torch.manual_seed(20260827 + layer_idx)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    state = H.load_real_layer_state(layer_idx)
    max_seq_len = 4096 if layer_idx == 3 else 256
    # 129 crosses the 128-token execution chunk and forces the final logical
    # token through fractured slicing, padding, trimming, and concatenation.
    seq_len = 129
    hidden = (torch.randn(1, 1, seq_len + 1, 10240, dtype=torch.bfloat16) * 0.02).contiguous()
    ple = None
    if layer_idx == 1:
        ple = (torch.randn(1, 1, seq_len + 1, 2560, dtype=torch.bfloat16) * 0.02).contiguous()

    baseline = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    multichip = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    prefill_hidden = _replicated_upload(hidden[:, :, :seq_len], mesh_device)
    baseline_kwargs = {}
    multichip_kwargs = {}
    if ple is not None:
        prefill_ple = _replicated_upload(ple[:, :, :seq_len], mesh_device)
        baseline_kwargs["ple_embeddings"] = prefill_ple
        multichip_kwargs["ple_embeddings"] = prefill_ple
    if layer_idx == 3:
        cos, sin = H.rope_tables(max_seq_len)
        page_host = H.shuffled_page_table(max_seq_len)
        page, chunk_pages, rot = _paged_inputs(baseline, mesh_device, page_host, cos, sin, seq_len)
        baseline_kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)
        multichip_kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)

    with H.ForbidHostFallback():
        baseline_prefill = baseline.prefill_forward(prefill_hidden, **baseline_kwargs)
        multichip_prefill = multichip.prefill_forward(prefill_hidden, **multichip_kwargs)
    ttnn.synchronize_device(mesh_device)
    multichip_prefill_ranks = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(multichip_prefill)]
    assert torch.equal(multichip_prefill_ranks[0], multichip_prefill_ranks[1])
    prefill_pcc = H.pcc(_rank_zero_host(baseline_prefill), _rank_zero_host(multichip_prefill))
    print(f"MULTICHIPPCC layer={layer_idx} prefill={prefill_pcc:.8f}")
    assert prefill_pcc >= H.PCC_BAR
    if layer_idx == 3:
        for cache_name, baseline_cache, multichip_cache in (
            ("key", baseline.kv_cache[0], multichip.kv_cache[0]),
            ("value", baseline.kv_cache[1], multichip.kv_cache[1]),
        ):
            assert list(multichip_cache.shape) == [64, 1, 64, 256]
            local_caches = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(multichip_cache)]
            reconstructed = torch.cat(local_caches, dim=1)
            cache_pcc = H.pcc(_rank_zero_host(baseline_cache), reconstructed)
            print(f"MULTICHIPCACHEPCC phase=prefill cache={cache_name} pcc={cache_pcc:.8f}")
            assert cache_pcc >= H.PCC_BAR
        assert list(multichip.indexer_cache.shape) == [64, 1, 64, 128]
        index_pcc = H.pcc(_rank_zero_host(baseline.indexer_cache), _rank_zero_host(multichip.indexer_cache))
        print(f"MULTICHIPCACHEPCC phase=prefill cache=indexer pcc={index_pcc:.8f}")
        assert index_pcc >= H.PCC_BAR

    baseline.prepare_decode_state()
    multichip.prepare_decode_state()
    if layer_idx != 3:
        user_recurrent_shards = ttnn.get_device_tensors(multichip.user_recurrent_state[0])
        decode_recurrent_shards = ttnn.get_device_tensors(multichip.recurrent_state)
        for rank, (expected, actual) in enumerate(zip(user_recurrent_shards, decode_recurrent_shards)):
            copy_pcc = H.pcc(ttnn.to_torch(expected), ttnn.to_torch(actual))
            print(f"MULTICHIPSTATECOPY layer={layer_idx} state=recurrent rank={rank} pcc={copy_pcc:.8f}")
        user_conv_shards = ttnn.get_device_tensors(multichip.user_conv_state[0])
        for tap, target in enumerate(multichip.fused_conv_state):
            target_shards = ttnn.get_device_tensors(target)
            for rank, (source, destination) in enumerate(zip(user_conv_shards, target_shards)):
                expected = ttnn.to_torch(source)[:, :, tap : tap + 1, :]
                copy_pcc = H.pcc(expected, ttnn.to_torch(destination))
                print(f"MULTICHIPSTATECOPY layer={layer_idx} state=conv{tap} rank={rank} pcc={copy_pcc:.8f}")
    baseline_captures = {}
    multichip_captures = {}
    attention_method = "_qsa_decode" if layer_idx == 3 else "_gdn_decode"
    for layer, captures in ((baseline, baseline_captures), (multichip, multichip_captures)):
        _capture_block(layer, attention_method, captures)
        _capture_block(layer, "_moe", captures)
        _capture_routing(layer, captures)
        if layer_idx != 3:
            _capture_gdn_epilogue(layer, captures)
    decode_hidden = _replicated_upload(hidden[:, :, -1:], mesh_device)
    current_pos = _replicated_upload(
        torch.tensor([seq_len], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    baseline_decode_kwargs = {"current_pos": current_pos}
    multichip_decode_kwargs = {"current_pos": current_pos}
    if ple is not None:
        decode_ple = _replicated_upload(ple[:, :, -1:], mesh_device)
        baseline_decode_kwargs["ple_embeddings"] = decode_ple
        multichip_decode_kwargs["ple_embeddings"] = decode_ple
    if layer_idx == 3:
        baseline_decode_kwargs.update(page_table=page, rot_mats=rot)
        multichip_decode_kwargs.update(page_table=page, rot_mats=rot)
    with H.ForbidHostFallback():
        baseline_decode = baseline.decode_forward(decode_hidden, **baseline_decode_kwargs)
        multichip_decode = multichip.decode_forward(decode_hidden, **multichip_decode_kwargs)
    ttnn.synchronize_device(mesh_device)
    multichip_decode_ranks = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(multichip_decode)]
    assert torch.equal(multichip_decode_ranks[0], multichip_decode_ranks[1])
    for name in (f"{attention_method}_input", f"{attention_method}_output", "_moe_input", "_moe_output"):
        baseline_value = _rank_zero_host(baseline_captures[name])
        local_values = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(multichip_captures[name])]
        multichip_value = (
            local_values[0]
            if local_values[0].numel() == baseline_value.numel()
            else torch.cat(local_values, dim=-1)
        )
        block_pcc = H.pcc(baseline_value, multichip_value)
        print(f"MULTICHIPBLOCKPCC layer={layer_idx} block={name} pcc={block_pcc:.8f}")
    if layer_idx != 3:
        for name in ("gdn_core", "gdn_z"):
            baseline_value = _rank_zero_host(baseline_captures[name])
            local_values = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(multichip_captures[name])]
            multichip_value = (
                local_values[0]
                if multichip.shapes.linear_num_value_heads == baseline.shapes.linear_num_value_heads
                else torch.cat(local_values, dim=-2)
            )
            value_pcc = H.pcc(baseline_value, multichip_value)
            print(f"MULTICHIPBLOCKPCC layer={layer_idx} block={name} pcc={value_pcc:.8f}")
    decode_pcc = H.pcc(_rank_zero_host(baseline_decode), _rank_zero_host(multichip_decode))
    print(f"MULTICHIPPCC layer={layer_idx} decode={decode_pcc:.8f}")
    assert decode_pcc >= H.PCC_BAR
    routing = _rank_zero_host(multichip_captures["routing"])
    assert int(routing[0, 0, 0].ne(0).sum()) == config.text_config.num_experts_per_tok
    if layer_idx == 3:
        for cache_name, baseline_cache, multichip_cache in (
            ("key", baseline.kv_cache[0], multichip.kv_cache[0]),
            ("value", baseline.kv_cache[1], multichip.kv_cache[1]),
        ):
            local_caches = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(multichip_cache)]
            cache_pcc = H.pcc(_rank_zero_host(baseline_cache), torch.cat(local_caches, dim=1))
            print(f"MULTICHIPCACHEPCC phase=decode cache={cache_name} pcc={cache_pcc:.8f}")
            assert cache_pcc >= H.PCC_BAR
        index_pcc = H.pcc(_rank_zero_host(baseline.indexer_cache), _rank_zero_host(multichip.indexer_cache))
        print(f"MULTICHIPCACHEPCC phase=decode cache=indexer pcc={index_pcc:.8f}")
        assert index_pcc >= H.PCC_BAR


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize(
    "device_params",
    [_multichip_device_params(trace_region_size=64_000_000)],
    indirect=True,
)
def test_multichip_decode_trace_replay_determinism(bh_1d_mesh_device, device_params, layer_idx):
    """Capture/replay the real local graph and CCL with nonzero setup weights."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    max_seq_len = 4096 if layer_idx == 3 else 128
    layer = MultichipDecoder.from_state_dict(
        H.make_partial_state(config, layer_idx),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    hidden = _replicated_upload(
        torch.randn(1, 1, 1, 10240, generator=torch.Generator().manual_seed(8800 + layer_idx)).bfloat16() * 0.02,
        mesh_device,
    )
    current_pos = _replicated_upload(
        torch.tensor([0], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kwargs = {"current_pos": current_pos}
    if layer_idx == 1:
        kwargs["ple_embeddings"] = _replicated_upload(torch.zeros(1, 1, 1, 2560, dtype=torch.bfloat16), mesh_device)
    if layer_idx == 3:
        cos, sin = H.rope_tables(max_seq_len)
        page_host = H.shuffled_page_table(max_seq_len)
        page, _, rot = _paged_inputs(layer, mesh_device, page_host, cos, sin, 1)
        kwargs.update(page_table=page, rot_mats=rot)

    layer.prepare_decode_state()
    with H.ForbidHostFallback():
        warm = layer.decode_forward(hidden, **kwargs)
    ttnn.synchronize_device(mesh_device)
    warm_host = _rank_zero_host(warm)
    layer.prepare_decode_state()
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    with H.ForbidHostFallback():
        traced = layer.decode_forward(hidden, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    replay_outputs = []
    for _ in range(5):
        layer.prepare_decode_state()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        replay_outputs.append(_rank_zero_host(traced))
    ttnn.release_trace(mesh_device, trace_id)
    replay_pcc = H.pcc(warm_host, replay_outputs[0])
    print(f"MULTICHIPTRACE layer={layer_idx} warm_replay_pcc={replay_pcc:.8f}")
    assert replay_pcc >= H.PCC_BAR
    assert all(torch.equal(replay_outputs[0], output) for output in replay_outputs[1:])


@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
def test_multichip_stacked_decoder_layout_contract(bh_1d_mesh_device, device_params):
    """One fractured residual flows across GDN, PLE+GDN, and QSA layers."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    layers = [
        MultichipDecoder.from_state_dict(
            H.make_partial_state(config, layer_idx),
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_batch=1,
            max_seq_len=4096 if layer_idx == 3 else 128,
        )
        for layer_idx in LAYER_KINDS
    ]
    for layer in layers:
        layer.prepare_decode_state()

    hidden = _replicated_upload(
        torch.randn(1, 1, 1, 10240, generator=torch.Generator().manual_seed(2401)).bfloat16() * 0.02,
        mesh_device,
    )
    current_pos = _replicated_upload(
        torch.tensor([0], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ple = _replicated_upload(torch.zeros(1, 1, 1, 2560, dtype=torch.bfloat16), mesh_device)
    cos, sin = H.rope_tables(4096)
    page_host = H.shuffled_page_table(4096)
    page, _, rot = _paged_inputs(layers[-1], mesh_device, page_host, cos, sin, 1)
    kwargs = (
        {"current_pos": current_pos},
        {"current_pos": current_pos, "ple_embeddings": ple},
        {"current_pos": current_pos, "page_table": page, "rot_mats": rot},
    )

    with H.ForbidHostFallback():
        output = layers[0].fracture_residual(hidden)
        for layer, layer_kwargs in zip(layers, kwargs):
            output = layer.decode_forward_fractured(output, **layer_kwargs)
    ttnn.synchronize_device(mesh_device)
    ranks = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(output)]
    assert list(output.shape) == [1, 1, 4, 1280]
    assert not torch.equal(ranks[0], ranks[1])
    gathered = layers[-1].gather_residual(output)
    gathered_ranks = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(gathered)]
    assert list(gathered.shape) == [1, 1, 1, 10240]
    assert torch.equal(gathered_ranks[0], gathered_ranks[1])


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("device_params", [_multichip_device_params()], indirect=True)
def test_multichip_batch32_decode_contract(bh_1d_mesh_device, device_params, layer_idx):
    """Match the optimized baseline for 32 distinct users and cache rows."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    # QSA's virtual-token selection has a fixed top-512 block geometry; the
    # GDN kinds need only their native 128-token construction.
    max_seq_len = 4096 if layer_idx == 3 else 128
    state = H.make_partial_state(config, layer_idx)
    baseline = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=32,
        max_seq_len=max_seq_len,
    )
    layer = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=32,
        max_seq_len=max_seq_len,
    )
    baseline.prepare_decode_state()
    layer.prepare_decode_state()
    hidden_host = (
        torch.randn(1, 1, 32, 10240, generator=torch.Generator().manual_seed(9300 + layer_idx)).bfloat16() * 0.02
    ).contiguous()
    baseline_hidden = _replicated_upload(hidden_host, mesh_device)
    hidden = _fractured_upload(hidden_host, mesh_device)
    positions_host = (torch.arange(32, dtype=torch.int32) + 33).contiguous()
    current_pos = _replicated_upload(
        positions_host,
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    baseline_kwargs = {"current_pos": current_pos}
    kwargs = {"current_pos": current_pos}
    if layer_idx == 1:
        ple = _replicated_upload(
            (torch.randn(1, 1, 32, 2560, generator=torch.Generator().manual_seed(9401)).bfloat16() * 0.02).contiguous(),
            mesh_device,
        )
        baseline_kwargs["ple_embeddings"] = ple
        kwargs["ple_embeddings"] = ple
    if layer_idx == 3:
        blocks_per_user = max_seq_len // layer.block_size
        page_host = torch.arange(32 * blocks_per_user, dtype=torch.int32).reshape(32, blocks_per_user)
        page_host[1::2] = page_host[1::2].flip(-1)
        page = _replicated_upload(
            page_host,
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cos, sin = H.rope_tables(max_seq_len)
        rot = (
            _replicated_upload(cos.reshape(1, 1, max_seq_len, -1), mesh_device),
            _replicated_upload(sin.reshape(1, 1, max_seq_len, -1), mesh_device),
        )
        baseline_kwargs.update(page_table=page, rot_mats=rot)
        kwargs.update(page_table=page, rot_mats=rot)

    with H.ForbidHostFallback():
        baseline_output = baseline.decode_forward(baseline_hidden, **baseline_kwargs)
        output = layer.decode_forward_fractured(hidden, **kwargs)
    ttnn.synchronize_device(mesh_device)
    assert list(output.shape) == [1, 1, 128, 1280]
    gathered = _fractured_host(output)
    assert list(gathered.shape) == [1, 1, 32, 10240]
    assert H.pcc(_rank_zero_host(baseline_output), gathered) >= H.PCC_BAR

    if layer_idx == 3:
        for baseline_cache, local_cache in zip(baseline.kv_cache, layer.kv_cache):
            local_shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(local_cache)]
            assert H.pcc(_rank_zero_host(baseline_cache), torch.cat(local_shards, dim=1)) >= H.PCC_BAR
        assert H.pcc(_rank_zero_host(baseline.indexer_cache), _rank_zero_host(layer.indexer_cache)) >= H.PCC_BAR
    else:
        assert H.pcc(_rank_zero_host(baseline.recurrent_state), _rank_zero_host(layer.recurrent_state)) >= H.PCC_BAR
        for baseline_tap, local_tap in zip(baseline.fused_conv_state, layer.fused_conv_state):
            assert H.pcc(_rank_zero_host(baseline_tap), _rank_zero_host(local_tap)) >= H.PCC_BAR
        if layer_idx == 1:
            for baseline_tap, local_tap in zip(baseline.fused_ple_conv_state, layer.fused_ple_conv_state):
                assert H.pcc(_rank_zero_host(baseline_tap), _rank_zero_host(local_tap)) >= H.PCC_BAR


@pytest.mark.long_context
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "device_params",
    [_multichip_device_params(trace_region_size=64_000_000)],
    indirect=True,
)
def test_multichip_qsa_trace_at_advertised_context(bh_1d_mesh_device, device_params):
    """Exercise local KV heads and maximum page/position geometry in trace."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    layer = MultichipDecoder.from_state_dict(
        None,
        hf_config=H.target_config(),
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=HF_ADVERTISED_CONTEXT,
    )
    assert layer.prefill_chunk_plan(HF_ADVERTISED_CONTEXT - 1)[-1] == (
        HF_ADVERTISED_CONTEXT - 128,
        127,
        128,
    )
    assert list(layer.kv_cache[0].shape) == [4096, 1, 64, 256]
    assert list(layer.indexer_cache.shape) == [4096, 1, 64, 128]
    hidden = _replicated_upload(torch.zeros(1, 1, 1, 10240, dtype=torch.bfloat16), mesh_device)
    current_pos = _replicated_upload(
        torch.tensor([HF_ADVERTISED_CONTEXT - 1], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    page_host = H.shuffled_page_table(HF_ADVERTISED_CONTEXT)
    page = _replicated_upload(
        page_host,
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cos, sin = H.rope_tables(HF_ADVERTISED_CONTEXT)
    rot = (
        _replicated_upload(cos.reshape(1, 1, HF_ADVERTISED_CONTEXT, -1), mesh_device),
        _replicated_upload(sin.reshape(1, 1, HF_ADVERTISED_CONTEXT, -1), mesh_device),
    )
    kwargs = {"current_pos": current_pos, "page_table": page, "rot_mats": rot}
    with H.ForbidHostFallback():
        layer.decode_forward(hidden, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    with H.ForbidHostFallback():
        traced = layer.decode_forward(hidden, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ranks = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(traced)]
    assert list(traced.shape) == [1, 1, 1, 10240]
    assert torch.equal(ranks[0], ranks[1])
    ttnn.release_trace(mesh_device, trace_id)

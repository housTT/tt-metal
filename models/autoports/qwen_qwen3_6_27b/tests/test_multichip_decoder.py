# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TP=4 acceptance coverage for the Qwen3.6-27B decoder layer."""

from __future__ import annotations

import inspect
import os
import time
from pathlib import Path

import pytest
import torch
import ttnn

from models.autoports.qwen_qwen3_6_27b.tests import test_functional_decoder as functional_tests
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    OptimizedDecoder,
    _decode_l1_memory,
    _dram_sharded_weight,
    _norm_l1_memory,
)


MESH_DEVICE = [4]
DIRECT_MESH_DEVICE = [int(os.environ.get("QWEN36_DIRECT_MESH_SIZE", "4"))]
DEVICE_PARAMS = [
    {
        "trace_region_size": 30_000_000,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    }
]
DIRECT_DEVICE_PARAMS = (
    DEVICE_PARAMS if DIRECT_MESH_DEVICE == [4] else [{"trace_region_size": 30_000_000}]
)


def _select_multichip_path(monkeypatch):
    MultichipDecoder.reset_fusion_counts()
    MultichipDecoder.reset_optimization_counts()
    MultichipDecoder.reset_multichip_counts()
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", MultichipDecoder)


def _assert_multichip_path():
    counts = MultichipDecoder.TOTAL_MULTICHIP_COUNTS
    assert counts["tp4_column_projections"] > 0
    assert counts["tp4_row_projections"] > 0
    assert counts["ring_all_reduce"] > 0


def _assert_all_replicas(reference, actual, threshold=0.995):
    """Validate every copy of the replicated stack boundary, not just device 0."""
    if actual.numel() == reference.numel():
        return functional_tests._pcc(reference, actual, threshold)
    assert actual.shape[0] == reference.shape[0] * 4, (reference.shape, actual.shape)
    replicas = torch.chunk(actual, 4, dim=0)
    messages = []
    for device_index, replica in enumerate(replicas):
        print(f"replicated_boundary_device={device_index} shape={tuple(replica.shape)}")
        messages.append(functional_tests._pcc(reference, replica, threshold))
        if device_index:
            functional_tests._pcc(replicas[0], replica, threshold=0.9999)
    return messages[0]


def _run_direct_optimized_comparison(monkeypatch, mesh_device, case):
    """Two-phase artifact test: write OptimizedDecoder, then compare TP=4 directly."""
    mode = os.environ.get("QWEN36_DIRECT_BASELINE_MODE")
    if mode not in ("write", "compare"):
        pytest.skip("set QWEN36_DIRECT_BASELINE_MODE=write or compare")
    artifact_root = Path(os.environ["QWEN36_DIRECT_BASELINE_DIR"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact = artifact_root / f"{case}.pt"
    actuals = []
    original_pcc = functional_tests._pcc
    original_to_torch = ttnn.to_torch

    def inspect_replicated_mesh_boundary(tensor, *args, **kwargs):
        device_tensors = ttnn.get_device_tensors(tensor)
        if mode == "compare" and len(device_tensors) == 4 and kwargs.get("mesh_composer") is not None:
            hosts = []
            for device_index, device_tensor in enumerate(device_tensors):
                assert device_tensor.layout == ttnn.TILE_LAYOUT
                assert device_tensor.dtype == ttnn.bfloat16
                assert device_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
                host = original_to_torch(device_tensor)
                assert host.shape[-1] == 5120
                print(
                    f"replicated_stack_boundary_device={device_index} "
                    f"shape={tuple(host.shape)} layout=TILE dtype=BF16 memory=DRAM"
                )
                hosts.append(host)
            for device_index in range(1, 4):
                original_pcc(hosts[0], hosts[device_index], threshold=0.9999)
        return original_to_torch(tensor, *args, **kwargs)

    def record_or_compare(reference, actual, threshold=0.995):
        if mode == "write":
            actuals.append(actual.detach().cpu().clone())
            return original_pcc(reference, actual, threshold)
        baseline = torch.load(artifact, weights_only=True)[len(actuals)]
        actuals.append(actual.detach().cpu().clone())
        # The inherited single-user composer indexes device zero before PCC;
        # the to_torch wrapper above independently validates all four physical
        # device tensors at every prefill/decode/trace boundary.
        print(f"direct_optimized_case={case} call={len(actuals)-1} device=0")
        original_pcc(baseline, actual, threshold=0.995)
        original_pcc(reference, actual, threshold=threshold)
        return "direct OptimizedDecoder and all-replica PCC passed"

    monkeypatch.setattr(functional_tests, "_pcc", record_or_compare)
    monkeypatch.setattr(ttnn, "to_torch", inspect_replicated_mesh_boundary)
    monkeypatch.setattr(
        functional_tests,
        "FunctionalDecoder",
        OptimizedDecoder if mode == "write" else MultichipDecoder,
    )
    if case == "linear_prefill_decode_trace":
        functional_tests.test_linear_attention_real_weight_prefill_and_decode(mesh_device, 65)
    elif case == "full_prefill_decode":
        functional_tests.test_full_attention_real_weight_paged_prefill(mesh_device, 33)
    elif case == "full_decode_trace":
        functional_tests.test_full_attention_real_weight_paged_decode_trace(mesh_device)
    else:
        raise AssertionError(case)
    if mode == "write":
        torch.save(actuals, artifact)
        print(f"wrote_direct_optimized_artifact={artifact} tensors={len(actuals)}")
    else:
        assert len(actuals) == len(torch.load(artifact, weights_only=True))
        _assert_multichip_path()


def test_multichip_runtime_has_no_host_or_layout_fallbacks():
    runtime_methods = (
        "_all_reduce_partial",
        "_mlp",
        "_finish_layer",
        "allocate_paged_kv_cache",
        "allocate_linear_state",
        "_full_qkv_prefill",
        "_full_qkv_decode",
        "_reduce_scatter_partial",
        "_distributed_norm_and_gather",
        "fracture_replicated_residual",
        "stacked_decode_forward",
    )
    source = "\n".join(inspect.getsource(getattr(MultichipDecoder, name)) for name in runtime_methods)
    for forbidden in ("torch", "from_torch", "to_torch", "tilize", "untilize", "reshard", "fallback"):
        assert forbidden not in source.lower()


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_multichip_local_tensor_and_state_contracts(mesh_device):
    config = functional_tests._config()

    full = MultichipDecoder.from_state_dict(
        functional_tests._load_layer_state(3),
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        page_block_size=64,
    )
    assert full.num_heads == 6
    assert full.num_kv_heads == 1
    assert full.intermediate_size == 4608
    assert [tuple(t.shape) for t in ttnn.get_device_tensors(full.full_qkv)] == [
        (5120, 3584)
    ] * 4
    key_cache, value_cache = full.allocate_paged_kv_cache(num_blocks=64, dtype=ttnn.bfloat8_b)
    for cache in (key_cache, value_cache):
        assert [tuple(t.shape) for t in ttnn.get_device_tensors(cache)] == [
            (64, 1, 64, 256)
        ] * 4

    linear = MultichipDecoder.from_state_dict(
        functional_tests._load_layer_state(0),
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        page_block_size=64,
    )
    assert linear.linear_num_key_heads == 4
    assert linear.linear_num_value_heads == 12
    assert linear.conv_dim == 2560
    assert linear.intermediate_size == 4352
    assert linear.dram_sharded_mlp_cores == {"gate": 8, "up": 8, "down": 8}
    assert linear.dram_sharded_mlp_blocks == {"gate": 10, "up": 10, "down": 17}
    conv_state, recurrent_state = linear.allocate_linear_state(batch_size=32)
    assert [tuple(t.shape) for t in ttnn.get_device_tensors(conv_state)] == [
        (32, 1, 4, 2560)
    ] * 4
    assert [tuple(t.shape) for t in ttnn.get_device_tensors(recurrent_state)] == [
        (32, 12, 128, 128)
    ] * 4


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_multichip_linear_non_aligned_prefill_decode(monkeypatch, mesh_device):
    _select_multichip_path(monkeypatch)
    functional_tests.test_linear_attention_real_weight_prefill_and_decode(mesh_device, 65)
    _assert_multichip_path()


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_multichip_full_non_aligned_paged_prefill_decode(monkeypatch, mesh_device):
    _select_multichip_path(monkeypatch)
    functional_tests.test_full_attention_real_weight_paged_prefill(mesh_device, 33)
    _assert_multichip_path()
    assert MultichipDecoder.TOTAL_MULTICHIP_COUNTS["paged_local_kv_cache"] > 0
    assert MultichipDecoder.TOTAL_MULTICHIP_COUNTS["local_attention_heads"] > 0


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_multichip_full_paged_decode_trace(monkeypatch, mesh_device):
    _select_multichip_path(monkeypatch)
    functional_tests.test_full_attention_real_weight_paged_decode_trace(mesh_device)
    _assert_multichip_path()


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_multichip_full_forced_chunked_prefill(monkeypatch, mesh_device):
    _select_multichip_path(monkeypatch)
    functional_tests.test_full_attention_real_weight_forced_chunked_prefill_pcc(mesh_device)
    _assert_multichip_path()


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["linear_attention", "full_attention"])
@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_multichip_real_weight_batch32(monkeypatch, mesh_device, layer_idx):
    _select_multichip_path(monkeypatch)
    original_pcc = functional_tests._pcc

    def compare_all_residual_replicas(reference, actual, threshold=0.995):
        if actual.shape[0] != reference.shape[0] * 4:
            return original_pcc(reference, actual, threshold)
        replicas = torch.chunk(actual, 4, dim=0)
        for device_index, replica in enumerate(replicas):
            print(f"batch32_replicated_boundary_device={device_index}")
            original_pcc(reference, replica, threshold)
            if device_index:
                original_pcc(replicas[0], replica, threshold=0.9999)
        return "all TP replicas passed"

    monkeypatch.setattr(functional_tests, "_pcc", compare_all_residual_replicas)
    functional_tests.test_real_weight_batch32_prefill_and_decode(mesh_device, layer_idx)
    _assert_multichip_path()


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["linear_attention", "full_attention"])
@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_multichip_decoder_perf(monkeypatch, mesh_device, layer_idx):
    _select_multichip_path(monkeypatch)
    signpost_times_ns = {}
    original_signpost = functional_tests.signpost

    def timed_signpost(tag):
        signpost_times_ns[tag] = time.perf_counter_ns()
        original_signpost(tag)

    monkeypatch.setattr(functional_tests, "signpost", timed_signpost)
    functional_tests.test_functional_decoder_perf(mesh_device, layer_idx)
    phase = os.environ.get("QWEN36_PERF_PHASE", "both")
    tag = "LINEAR_ATTENTION" if layer_idx == 0 else "FULL_ATTENTION"
    if phase != "decode":
        elapsed = signpost_times_ns[f"{tag}_PREFILL_END"] - signpost_times_ns[f"{tag}_PREFILL_START"]
        print(f"{tag}_MULTICHIP_PREFILL_E2E_US={elapsed / 1_000:.3f}")
    if phase != "prefill":
        replays = int(os.environ.get("QWEN36_DECODE_REPLAYS", "10"))
        elapsed = signpost_times_ns[f"{tag}_DECODE_TRACE_END"] - signpost_times_ns[f"{tag}_DECODE_TRACE_START"]
        print(f"{tag}_MULTICHIP_DECODE_TRACE_E2E_US_PER_REPLAY={elapsed / 1_000 / replays:.3f}")
    _assert_multichip_path()


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_multichip_full_native_context_decode(monkeypatch, mesh_device):
    _select_multichip_path(monkeypatch)
    functional_tests.test_full_attention_advertised_context_decode(mesh_device)
    _assert_multichip_path()


@pytest.mark.parametrize("logical_length", [32769, 262144], ids=["non_aligned_long", "native_max"])
@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_multichip_full_advertised_context_prefill(monkeypatch, mesh_device, logical_length):
    _select_multichip_path(monkeypatch)
    functional_tests.test_full_attention_advertised_context_prefill(mesh_device, logical_length)
    _assert_multichip_path()


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_multichip_linear_native_context(monkeypatch, mesh_device):
    _select_multichip_path(monkeypatch)
    functional_tests.test_linear_attention_advertised_context_prefill(mesh_device)
    functional_tests.test_linear_attention_advertised_context_decode(mesh_device)
    _assert_multichip_path()


@pytest.mark.parametrize(
    "case",
    ["linear_prefill_decode_trace", "full_prefill_decode", "full_decode_trace"],
)
@pytest.mark.parametrize("mesh_device", DIRECT_MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DIRECT_DEVICE_PARAMS, indirect=True)
def test_direct_optimized_baseline_and_all_replicas(monkeypatch, mesh_device, case):
    """Run once on one device in write mode, then on TP=4 in compare mode."""
    _run_direct_optimized_comparison(monkeypatch, mesh_device, case)


@pytest.mark.parametrize("local_k", [1536, 4608], ids=["attention_row", "full_mlp_down"])
@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_replicated_vs_fractured_stack_boundary(mesh_device, local_k):
    """Measure the shape-faithful lower-movement stacked-boundary family."""
    if os.environ.get("QWEN36_RUN_TOPOLOGY_PROBE") != "1":
        pytest.skip("set QWEN36_RUN_TOPOLOGY_PROBE=1")
    torch.manual_seed(2026 + local_k)
    hidden = 5120
    local_hidden = hidden // 4
    local_next_n = 3584
    activation_source = torch.randn(1, 1, 32, local_k * 4, dtype=torch.bfloat16) * 0.02
    row_weight_source = torch.randn(local_k * 4, hidden, dtype=torch.bfloat16) * 0.02
    residual_source = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16) * 0.02
    norm_source = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
    next_weight_source = torch.randn(hidden, local_next_n, dtype=torch.bfloat16) * 0.02

    def mesh_tensor(source, mapper):
        return ttnn.from_torch(
            source,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    activation = mesh_tensor(activation_source, ttnn.ShardTensorToMesh(mesh_device, dim=-1))
    row_weight = mesh_tensor(row_weight_source, ttnn.ShardTensorToMesh(mesh_device, dim=0))
    residual_replicated = mesh_tensor(residual_source, ttnn.ReplicateTensorToMesh(mesh_device))
    residual_fractured = mesh_tensor(residual_source, ttnn.ShardTensorToMesh(mesh_device, dim=-1))
    norm_replicated = mesh_tensor(norm_source, ttnn.ReplicateTensorToMesh(mesh_device))
    norm_fractured = mesh_tensor(norm_source, ttnn.ShardTensorToMesh(mesh_device, dim=-1))
    next_weight = mesh_tensor(next_weight_source, ttnn.ReplicateTensorToMesh(mesh_device))
    compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
    )

    def row_partial():
        return ttnn.matmul(activation, row_weight, dtype=ttnn.bfloat16, compute_kernel_config=compute)

    def replicated_path():
        reduced = ttnn.all_reduce(row_partial(), num_links=1, topology=ttnn.Topology.Ring)
        residual = ttnn.add(residual_replicated, reduced)
        normalized = ttnn.rms_norm(residual, epsilon=1e-6, weight=norm_replicated)
        return ttnn.matmul(normalized, next_weight, dtype=ttnn.bfloat16, compute_kernel_config=compute)

    def fractured_path():
        fractured = ttnn.reduce_scatter(
            row_partial(), dim=3, num_links=1, topology=ttnn.Topology.Ring
        )
        residual = ttnn.add(residual_fractured, fractured)
        stats = ttnn.rms_norm_pre_all_gather(residual, dtype=ttnn.bfloat16)
        stats = ttnn.all_gather(stats, dim=3, num_links=1, topology=ttnn.Topology.Ring)
        normalized = ttnn.rms_norm_post_all_gather(
            residual, stats, epsilon=1e-6, weight=norm_fractured
        )
        gathered = ttnn.all_gather(
            normalized, dim=3, num_links=1, topology=ttnn.Topology.Ring
        )
        return ttnn.matmul(gathered, next_weight, dtype=ttnn.bfloat16, compute_kernel_config=compute)

    replicated = replicated_path()
    fractured = fractured_path()
    ttnn.synchronize_device(mesh_device)
    replicated_hosts = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(replicated)]
    fractured_hosts = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(fractured)]
    for device_index in range(4):
        functional_tests._pcc(replicated_hosts[device_index], fractured_hosts[device_index], 0.995)

    def traced_us(path, replays=20):
        path()
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        path()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        start = time.perf_counter_ns()
        for _ in range(replays):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        elapsed_us = (time.perf_counter_ns() - start) / 1_000 / replays
        ttnn.release_trace(mesh_device, trace_id)
        return elapsed_us

    replicated_us = traced_us(replicated_path)
    fractured_us = traced_us(fractured_path)
    print(
        f"TOPOLOGY_PROBE local_k={local_k} replicated_us={replicated_us:.3f} "
        f"fractured_us={fractured_us:.3f} ratio={fractured_us / replicated_us:.3f}"
    )


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["linear_attention", "full_attention"])
@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_fractured_stacked_decode_trace(monkeypatch, mesh_device, layer_idx):
    """Validate the selected fractured decode graph through the HF harness."""
    _select_multichip_path(monkeypatch)

    def compatibility_decode(self, hidden_states, **kwargs):
        fractured = self.fracture_replicated_residual(hidden_states)
        output = self.stacked_decode_forward(fractured, **kwargs)
        assert output.shape[-1] == 1280
        return ttnn.all_gather(
            output,
            dim=3,
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    monkeypatch.setattr(MultichipDecoder, "decode_forward", compatibility_decode)
    if layer_idx == 0:
        functional_tests.test_linear_attention_real_weight_prefill_and_decode(mesh_device, 65)
    else:
        functional_tests.test_full_attention_real_weight_paged_decode_trace(mesh_device)
    assert MultichipDecoder.TOTAL_MULTICHIP_COUNTS["ring_reduce_scatter"] > 0
    assert MultichipDecoder.TOTAL_MULTICHIP_COUNTS["distributed_rms_norm"] > 0


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["linear_attention", "full_attention"])
@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_real_weight_replicated_vs_fractured_decode_perf(mesh_device, layer_idx):
    """A/B the warmed real-weight replicated and selected stack contracts.

    Cache/state setup, the one-time stack-entry fracture, and the test-boundary
    gather used for PCC are intentionally outside both measured traces.
    """
    if os.environ.get("QWEN36_RUN_STACKED_PERF") != "1":
        pytest.skip("set QWEN36_RUN_STACKED_PERF=1")
    torch.manual_seed(2036 + layer_idx)
    config = functional_tests._config()
    decoder = MultichipDecoder.from_state_dict(
        functional_tests._load_layer_state(layer_idx),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        page_block_size=64,
    )
    seq_len = 32
    prefill_host = torch.randn(
        1, 1, seq_len, config.hidden_size, dtype=torch.bfloat16
    ) * 0.1
    prefill = functional_tests._to_tt(prefill_host, mesh_device)

    if decoder.layer_kind == "full_attention":
        rotary = functional_tests.Qwen3_5TextRotaryEmbedding(config)
        cos_host, sin_host = rotary(
            prefill_host[:, 0], torch.arange(seq_len).unsqueeze(0)
        )
        cos = functional_tests._to_tt(cos_host.unsqueeze(1), mesh_device)
        sin = functional_tests._to_tt(sin_host.unsqueeze(1), mesh_device)
        page_table = functional_tests._to_tt(
            torch.arange(32, dtype=torch.int32).reshape(32, 1),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        replicated_cache = decoder.allocate_paged_kv_cache(num_blocks=32)
        fractured_cache = decoder.allocate_paged_kv_cache(num_blocks=32)
        decoder.prefill_forward(
            prefill,
            logical_seq_len=seq_len,
            cos=cos,
            sin=sin,
            page_table=page_table,
            kv_cache=replicated_cache,
        )
        decoder.prefill_forward(
            prefill,
            logical_seq_len=seq_len,
            cos=cos,
            sin=sin,
            page_table=page_table,
            kv_cache=fractured_cache,
        )
        decode_host = torch.zeros(
            1, 1, 32, config.hidden_size, dtype=torch.bfloat16
        )
        decode_host[:, :, 0] = (
            torch.randn(1, config.hidden_size, dtype=torch.bfloat16) * 0.1
        )
        decode_cos_host, decode_sin_host = rotary(
            decode_host[:, 0, :1], torch.tensor([[seq_len]])
        )
        decode_cos = functional_tests._to_tt(
            decode_cos_host.unsqueeze(0).expand(1, 32, 1, -1).contiguous(),
            mesh_device,
        )
        decode_sin = functional_tests._to_tt(
            decode_sin_host.unsqueeze(0).expand(1, 32, 1, -1).contiguous(),
            mesh_device,
        )
        positions = functional_tests._to_tt(
            torch.full((32,), seq_len, dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        replicated_kwargs = dict(
            current_positions=positions,
            cos=decode_cos,
            sin=decode_sin,
            page_table=page_table,
            kv_cache=replicated_cache,
        )
        fractured_kwargs = dict(replicated_kwargs, kv_cache=fractured_cache)
    else:
        replicated_state = decoder.allocate_linear_state(batch_size=1)
        fractured_state = decoder.allocate_linear_state(batch_size=1)
        decoder.prefill_forward(
            prefill, logical_seq_len=seq_len, linear_state=replicated_state
        )
        decoder.prefill_forward(
            prefill, logical_seq_len=seq_len, linear_state=fractured_state
        )
        decode_host = (
            torch.randn(1, 1, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1
        )
        positions = functional_tests._to_tt(
            torch.tensor([seq_len], dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        replicated_kwargs = dict(
            current_positions=positions, linear_state=replicated_state
        )
        fractured_kwargs = dict(
            current_positions=positions, linear_state=fractured_state
        )

    decode = functional_tests._to_tt(decode_host, mesh_device)
    fractured_decode = decoder.fracture_replicated_residual(decode)

    def replicated_path():
        return decoder.decode_forward(decode, **replicated_kwargs)

    def fractured_path():
        return decoder.stacked_decode_forward(fractured_decode, **fractured_kwargs)

    replicated_output = replicated_path()
    fractured_output = fractured_path()
    fractured_boundary = ttnn.all_gather(
        fractured_output,
        dim=3,
        num_links=1,
        topology=ttnn.Topology.Ring,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(mesh_device)
    replicated_hosts = [
        ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(replicated_output)
    ]
    fractured_hosts = [
        ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(fractured_boundary)
    ]
    for device_index in range(4):
        print(f"real_weight_stack_boundary_device={device_index}")
        functional_tests._pcc(
            replicated_hosts[device_index], fractured_hosts[device_index], 0.995
        )

    def traced_us(path):
        path()
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        path()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        replays = int(os.environ.get("QWEN36_STACKED_PERF_REPLAYS", "20"))
        assert replays >= 1
        start = time.perf_counter_ns()
        for _ in range(replays):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        elapsed_us = (time.perf_counter_ns() - start) / 1_000 / replays
        ttnn.release_trace(mesh_device, trace_id)
        return elapsed_us

    replicated_us = traced_us(replicated_path)
    fractured_us = traced_us(fractured_path)
    speedup = replicated_us / fractured_us
    print(
        f"REAL_WEIGHT_STACKED_DECODE layer={decoder.layer_kind} "
        f"replicated_us={replicated_us:.3f} fractured_us={fractured_us:.3f} "
        f"speedup={speedup:.3f}"
    )
    maximum_fractured_ratio = float(
        os.environ.get("QWEN36_MAX_FRACTURED_TO_REPLICATED_RATIO", "1.5")
    )
    assert fractured_us / replicated_us <= maximum_fractured_ratio
    assert replicated_us < fractured_us
    print("REAL_WEIGHT_STACKED_SELECTION selected=replicated")


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_installed_fused_ccl_contracts(mesh_device):
    """Exercise the installed fused RS-matmul and AG-matmul contracts."""
    if os.environ.get("QWEN36_RUN_FUSED_CCL_PROBE") != "1":
        pytest.skip("set QWEN36_RUN_FUSED_CCL_PROBE=1")
    from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import (
        run_test_linear,
    )
    from models.tt_dit.tests.models.wan2_2.test_strided_reduce_scatter_wan_t3000 import (
        _run_optional_feature_test,
    )

    _run_optional_feature_test(mesh_device, ttnn.Topology.Ring, 1, "bias")
    results = run_test_linear(
        mesh_device,
        3072,
        5120,
        3456,
        8,
        8,
        8,
        2,
        2,
        ttnn.Topology.Ring,
        core_grid=ttnn.CoreCoord(8, 8),
        num_workers_per_link=4,
        num_links=2,
        use_bias=False,
        use_non_fused=False,
        force_transpose=True,
        sp_axis=0,
        tp_axis=1,
        num_iters=1,
        enable_trace=False,
        cluster_axis=1,
    )
    assert all(item["pcc"] > 0.9995 for chunk in results[0] for item in chunk)


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_linear_attention_bfp4_mlp_geometry_sweep(mesh_device):
    """Sweep the exact TP-local BFP4 MLP shapes without changing production policy."""
    if os.environ.get("QWEN36_RUN_BFP4_GEOMETRY_SWEEP") != "1":
        pytest.skip("set QWEN36_RUN_BFP4_GEOMETRY_SWEEP=1")

    # Exact shapes have gcd(Kt, Nt) == 8, so an unpadded DRAM-sharded program
    # can use only the eight Blackhole DRAM-reader workers.  Sweep every useful
    # K-block divisor from largest/lowest-loop-count to the low-L1 fallback.
    cases = (
        ("gate_up", 5120, 4352, (20, 10, 5, 4, 2, 1)),
        ("down", 4352, 5120, (17, 1)),
    )
    torch.manual_seed(0)
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    for role, k, n, blocks in cases:
        host_input = torch.randn((1, 1, 32, k), dtype=torch.bfloat16)
        host_weight = torch.randn((n, k), dtype=torch.bfloat16)
        reference = torch.matmul(host_input.float(), host_weight.float().transpose(-2, -1))
        input_memory = _norm_l1_memory(32, k, 8)
        output_memory = _decode_l1_memory(32, n, 8)
        device_input = ttnn.as_tensor(
            host_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=input_memory,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        device_weight = _dram_sharded_weight(
            host_weight,
            mesh_device,
            dtype=ttnn.bfloat4_b,
        )
        for in0_block_w in blocks:
            program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=in0_block_w,
                per_core_M=1,
                per_core_N=n // 32 // 8,
                fused_activation=None,
            )
            output = ttnn.linear(
                device_input,
                device_weight,
                dtype=ttnn.bfloat16,
                program_config=program_config,
                memory_config=output_memory,
                compute_kernel_config=compute_config,
            )
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            for _ in range(5):
                output = ttnn.linear(
                    device_input,
                    device_weight,
                    dtype=ttnn.bfloat16,
                    program_config=program_config,
                    memory_config=output_memory,
                    compute_kernel_config=compute_config,
                )
            ttnn.synchronize_device(mesh_device)
            latency_us = (time.perf_counter() - start) * 1e6 / 5
            for device_index, tensor in enumerate(ttnn.get_device_tensors(output)):
                actual = ttnn.to_torch(tensor).float()
                functional_tests._pcc(reference, actual, threshold=0.95)
                print(
                    f"BFP4_GEOMETRY role={role} device={device_index} M=32 K={k} N={n} "
                    f"cores=8 in0_block_w={in0_block_w} latency_us={latency_us:.3f}"
                )

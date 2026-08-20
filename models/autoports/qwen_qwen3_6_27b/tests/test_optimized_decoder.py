# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized-path acceptance and profiler entry points for Qwen3.6-27B."""

import inspect
import os
import time

import pytest
import torch
import ttnn

from models.autoports.qwen_qwen3_6_27b.tests import test_functional_decoder as functional_tests
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder


def _select_optimized_path(monkeypatch):
    OptimizedDecoder.reset_fusion_counts()
    OptimizedDecoder.reset_optimization_counts()
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", OptimizedDecoder)


def _assert_optimized_path():
    assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["low_precision_mlp"] > 0
    assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["low_precision_projections"] > 0


def _select_fused_control(monkeypatch):
    FusedDecoder.reset_fusion_counts()
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", FusedDecoder)


def test_optimized_runtime_has_no_host_or_layout_fallbacks():
    runtime_methods = (
        "_mlp",
        "_dram_projection",
        "_linear_input_projections",
        "_full_qkv_prefill",
        "_full_qkv_decode",
        "_full_prefill",
        "_full_prefill_chunked_layer",
        "_full_decode",
        "_linear_token",
        "_linear_gated_delta_chunk",
        "_linear_chunk_inverse",
        "_linear_chunk",
        "_prefill_program_config",
        "_gdn_prefill_matmul",
        "_finish_layer",
        "_prefill_users_independently",
        "_linear_decode_users_independently",
        "prefill_forward",
        "decode_forward",
    )
    source = "\n".join(inspect.getsource(getattr(OptimizedDecoder, name)) for name in runtime_methods)
    for forbidden in ("torch", "from_torch", "to_torch", "tilize", "untilize", "reshard", "fallback"):
        assert forbidden not in source.lower()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_linear_attention_non_aligned_prefill_decode(monkeypatch, mesh_device):
    _select_optimized_path(monkeypatch)
    functional_tests.test_linear_attention_real_weight_prefill_and_decode(mesh_device, 65)
    _assert_optimized_path()
    assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["sharded_input_norm"] > 0


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_full_attention_non_aligned_prefill_decode(monkeypatch, mesh_device):
    _select_optimized_path(monkeypatch)
    functional_tests.test_full_attention_real_weight_paged_prefill(mesh_device, 33)
    _assert_optimized_path()
    assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["sharded_input_norm"] > 0


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_full_attention_paged_decode_trace(monkeypatch, mesh_device):
    _select_optimized_path(monkeypatch)
    functional_tests.test_full_attention_real_weight_paged_decode_trace(mesh_device)
    _assert_optimized_path()
    assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["sharded_input_norm"] > 0


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_full_attention_forced_chunked_prefill(monkeypatch, mesh_device):
    _select_optimized_path(monkeypatch)
    functional_tests.test_full_attention_real_weight_forced_chunked_prefill_pcc(mesh_device)
    _assert_optimized_path()


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["linear_attention", "full_attention"])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_real_weight_batch32_prefill_decode(monkeypatch, mesh_device, layer_idx):
    _select_optimized_path(monkeypatch)
    functional_tests.test_real_weight_batch32_prefill_and_decode(mesh_device, layer_idx)
    _assert_optimized_path()
    assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["dram_sharded_mlp"] > 0
    if layer_idx == 0:
        assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["gdn_read_geometry"] > 0


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_full_attention_native_context_decode(monkeypatch, mesh_device):
    _select_optimized_path(monkeypatch)
    functional_tests.test_full_attention_advertised_context_decode(mesh_device)
    _assert_optimized_path()


@pytest.mark.parametrize("logical_length", [32769, 262144], ids=["non_aligned_long", "native_max"])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_full_attention_advertised_context_prefill(
    monkeypatch, mesh_device, logical_length
):
    _select_optimized_path(monkeypatch)
    functional_tests.test_full_attention_advertised_context_prefill(mesh_device, logical_length)
    _assert_optimized_path()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_linear_attention_native_context(monkeypatch, mesh_device):
    _select_optimized_path(monkeypatch)
    functional_tests.test_linear_attention_advertised_context_prefill(mesh_device)
    functional_tests.test_linear_attention_advertised_context_decode(mesh_device)
    _assert_optimized_path()


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["linear_attention", "full_attention"])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_decoder_perf(monkeypatch, mesh_device, layer_idx):
    _select_optimized_path(monkeypatch)
    signpost_times_ns = {}
    original_signpost = functional_tests.signpost

    def timed_signpost(tag):
        signpost_times_ns[tag] = time.perf_counter_ns()
        original_signpost(tag)

    monkeypatch.setattr(functional_tests, "signpost", timed_signpost)
    functional_tests.test_functional_decoder_perf(mesh_device, layer_idx)
    phase = os.environ.get("QWEN36_PERF_PHASE", "both")
    assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["dram_sharded_mlp"] > 0
    if phase != "prefill":
        assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["sharded_input_norm"] > 0
        assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["dram_sharded_projections"] > 0
    if layer_idx == 0:
        if phase != "decode" and os.environ.get("QWEN36_OPT_GDN_PREFILL_GEOMETRY", "reuse") == "reuse":
            assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["gdn_prefill_geometry"] > 0
        if phase != "prefill":
            assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["gdn_read_geometry"] > 0
    tag = "LINEAR_ATTENTION" if layer_idx == 0 else "FULL_ATTENTION"
    if phase != "decode":
        prefill_ns = (
            signpost_times_ns[f"{tag}_PREFILL_END"]
            - signpost_times_ns[f"{tag}_PREFILL_START"]
        )
        print(f"{tag}_PREFILL_E2E_US={prefill_ns / 1_000:.3f}")
    if phase != "prefill":
        replays = int(os.environ.get("QWEN36_DECODE_REPLAYS", "10"))
        decode_ns = (
            signpost_times_ns[f"{tag}_DECODE_TRACE_END"]
            - signpost_times_ns[f"{tag}_DECODE_TRACE_START"]
        )
        print(f"{tag}_DECODE_TRACE_E2E_US_PER_REPLAY={decode_ns / 1_000 / replays:.3f}")
    assert OptimizedDecoder.TOTAL_FUSION_COUNTS["mlp_silu_multiply"] >= 1
    _assert_optimized_path()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_linear_attention_large_prefill_perf(mesh_device):
    """Profiler entry point for two warmed 64-token GDN chunks."""
    OptimizedDecoder.reset_optimization_counts()
    torch.manual_seed(31)
    config = functional_tests._config()
    decoder = OptimizedDecoder.from_state_dict(
        functional_tests._load_layer_state(0),
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        page_block_size=64,
    )
    seq_len = 128
    hidden_host = torch.randn(
        1, 1, seq_len, config.hidden_size, dtype=torch.bfloat16
    ) * 0.1
    hidden = functional_tests._to_tt(hidden_host, mesh_device)
    warm_state = decoder.allocate_linear_state(batch_size=1)
    decoder.prefill_forward(hidden, logical_seq_len=seq_len, linear_state=warm_state)
    ttnn.synchronize_device(mesh_device)
    measured_state = decoder.allocate_linear_state(batch_size=1)
    start_ns = time.perf_counter_ns()
    functional_tests.signpost("LINEAR_ATTENTION_LARGE_PREFILL_START")
    decoder.prefill_forward(hidden, logical_seq_len=seq_len, linear_state=measured_state)
    ttnn.synchronize_device(mesh_device)
    functional_tests.signpost("LINEAR_ATTENTION_LARGE_PREFILL_END")
    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000
    print(f"LINEAR_ATTENTION_LARGE_PREFILL_E2E_US={elapsed_us:.3f}")
    assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["gdn_prefill_geometry"] > 0
    assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["low_precision_mlp"] > 0


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_full_attention_large_prefill_perf(mesh_device):
    """Profiler entry point for warmed paged multi-chunk full attention."""
    OptimizedDecoder.reset_optimization_counts()
    torch.manual_seed(47)
    config = functional_tests._config()
    decoder = OptimizedDecoder.from_state_dict(
        functional_tests._load_layer_state(3),
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        page_block_size=64,
    )
    decoder.full_prefill_sdpa_limit = 128
    decoder.full_prefill_chunk_size = 128
    logical_length = 257
    physical_length = 384
    hidden_host = torch.randn(
        1, logical_length, config.hidden_size, dtype=torch.bfloat16
    ) * 0.1
    padded_hidden = torch.nn.functional.pad(
        hidden_host, (0, 0, 0, physical_length - logical_length)
    )
    positions = torch.arange(physical_length).unsqueeze(0)
    rotary = functional_tests.Qwen3_5TextRotaryEmbedding(config)
    cos, sin = rotary(padded_hidden, positions)
    hidden = functional_tests._to_tt(padded_hidden.unsqueeze(1), mesh_device)
    cos_tt = functional_tests._to_tt(cos.unsqueeze(1), mesh_device)
    sin_tt = functional_tests._to_tt(sin.unsqueeze(1), mesh_device)
    pages = physical_length // decoder.page_block_size
    page_table = functional_tests._to_tt(
        torch.arange(pages - 1, -1, -1, dtype=torch.int32).reshape(1, pages),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    def run_once():
        return decoder.prefill_forward(
            hidden,
            logical_seq_len=logical_length,
            cos=cos_tt,
            sin=sin_tt,
            page_table=page_table,
            kv_cache=decoder.allocate_paged_kv_cache(num_blocks=pages),
        )

    run_once()
    ttnn.synchronize_device(mesh_device)
    start_ns = time.perf_counter_ns()
    functional_tests.signpost("FULL_ATTENTION_LARGE_PREFILL_START")
    run_once()
    ttnn.synchronize_device(mesh_device)
    functional_tests.signpost("FULL_ATTENTION_LARGE_PREFILL_END")
    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000
    print(f"FULL_ATTENTION_LARGE_PREFILL_E2E_US={elapsed_us:.3f}")
    assert OptimizedDecoder.TOTAL_OPTIMIZATION_COUNTS["low_precision_mlp"] > 0


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["linear_attention", "full_attention"])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_fused_decoder_perf_e2e_control(monkeypatch, mesh_device, layer_idx):
    """Like-for-like warmed host timing control for the inherited fused path."""
    _select_fused_control(monkeypatch)
    signpost_times_ns = {}
    original_signpost = functional_tests.signpost

    def timed_signpost(tag):
        signpost_times_ns[tag] = time.perf_counter_ns()
        original_signpost(tag)

    monkeypatch.setattr(functional_tests, "signpost", timed_signpost)
    functional_tests.test_functional_decoder_perf(mesh_device, layer_idx)
    tag = "LINEAR_ATTENTION" if layer_idx == 0 else "FULL_ATTENTION"
    prefill_ns = (
        signpost_times_ns[f"{tag}_PREFILL_END"]
        - signpost_times_ns[f"{tag}_PREFILL_START"]
    )
    replays = int(os.environ.get("QWEN36_DECODE_REPLAYS", "10"))
    decode_ns = (
        signpost_times_ns[f"{tag}_DECODE_TRACE_END"]
        - signpost_times_ns[f"{tag}_DECODE_TRACE_START"]
    )
    print(f"FUSED_{tag}_PREFILL_E2E_US={prefill_ns / 1_000:.3f}")
    print(f"FUSED_{tag}_DECODE_TRACE_E2E_US_PER_REPLAY={decode_ns / 1_000 / replays:.3f}")
    assert FusedDecoder.TOTAL_FUSION_COUNTS["mlp_silu_multiply"] >= 1

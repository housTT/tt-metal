# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused-path acceptance and profiler entry points for Qwen3.6-27B."""

import inspect

import pytest

from models.autoports.qwen_qwen3_6_27b.tests import test_functional_decoder as functional_tests
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder


def _select_fused_path(monkeypatch):
    FusedDecoder.reset_fusion_counts()
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", FusedDecoder)


def test_fused_runtime_has_no_host_or_layout_fallbacks():
    """Keep host conversion and avoidable layout fallbacks out of measured methods."""
    runtime_methods = (
        "_mlp",
        "_partial_rope",
        "_full_qkv_prefill",
        "_full_qkv_decode",
        "_full_prefill",
        "_full_prefill_chunked_layer",
        "_full_decode",
        "_linear_input_projections",
        "_linear_chunk",
        "_linear_gated_delta_chunk",
        "_linear_token",
    )
    source = "\n".join(inspect.getsource(getattr(FusedDecoder, name)) for name in runtime_methods)
    for forbidden in (
        "from_torch",
        "to_torch",
        "torch.",
        "tilize",
        "untilize",
        "reshard",
        "fallback",
    ):
        assert forbidden not in source.lower()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_fused_full_attention_non_aligned_prefill_decode(monkeypatch, mesh_device):
    """Run the real-weight non-aligned paged path through fused overrides."""
    _select_fused_path(monkeypatch)
    functional_tests.test_full_attention_real_weight_paged_prefill(mesh_device, 33)
    counts = FusedDecoder.TOTAL_FUSION_COUNTS
    assert counts["partial_rope"] >= 2
    assert counts["attention_sigmoid_multiply"] >= 2
    assert counts["shared_lhs_full_qkv"] >= 2
    assert counts["dedicated_full_qkv_heads"] >= 2
    assert counts["mlp_silu_multiply"] >= 2


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_fused_full_attention_paged_decode_trace(monkeypatch, mesh_device):
    """Prove trace capture/replay uses the fused full-attention graph."""
    _select_fused_path(monkeypatch)
    functional_tests.test_full_attention_real_weight_paged_decode_trace(mesh_device)
    counts = FusedDecoder.TOTAL_FUSION_COUNTS
    assert counts["partial_rope"] >= 2
    assert counts["attention_sigmoid_multiply"] >= 2
    assert counts["shared_lhs_full_qkv"] >= 2
    assert counts["dedicated_full_qkv_heads"] >= 2
    assert counts["mlp_silu_multiply"] >= 2


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_fused_full_attention_forced_chunked_prefill(monkeypatch, mesh_device):
    """Exercise the distinct long-context graph, including a non-aligned tail."""
    _select_fused_path(monkeypatch)
    functional_tests.test_full_attention_real_weight_forced_chunked_prefill_pcc(mesh_device)
    counts = FusedDecoder.TOTAL_FUSION_COUNTS
    assert counts["shared_lhs_full_qkv"] >= 3
    assert counts["dedicated_full_qkv_heads"] >= 3
    assert counts["partial_rope"] >= 6
    assert counts["attention_sigmoid_multiply"] >= 3
    assert counts["mlp_silu_multiply"] >= 3


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_fused_full_attention_native_context_decode(monkeypatch, mesh_device):
    """Allocate the native 262,144-token cache and trace the last position."""
    _select_fused_path(monkeypatch)
    functional_tests.test_full_attention_advertised_context_decode(mesh_device)
    counts = FusedDecoder.TOTAL_FUSION_COUNTS
    assert counts["shared_lhs_full_qkv"] >= 2
    assert counts["dedicated_full_qkv_heads"] >= 2
    assert counts["attention_sigmoid_multiply"] >= 2
    assert counts["mlp_silu_multiply"] >= 2


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_fused_linear_attention_non_aligned_prefill_decode(monkeypatch, mesh_device):
    """Preserve DeltaNet state and non-aligned semantics on the fused layer tail."""
    _select_fused_path(monkeypatch)
    functional_tests.test_linear_attention_real_weight_prefill_and_decode(mesh_device, 65)
    counts = FusedDecoder.TOTAL_FUSION_COUNTS
    assert counts["shared_lhs_linear_projections"] >= 3
    assert counts["tile_aligned_linear_projection_tails"] >= 3
    assert counts["linear_silu_multiply"] >= 3
    assert counts["matmul_transpose_flags"] >= 3
    assert counts["l2_rms_norm"] >= 6
    assert counts["decode_beta_sigmoid_multiply"] >= 2
    assert counts["repeat_interleave_qk_heads"] >= 6
    assert counts["mlp_silu_multiply"] >= 3


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["linear_attention", "full_attention"])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_fused_real_weight_batch32_prefill_decode(monkeypatch, mesh_device, layer_idx):
    """Preserve 32-active-user state/cache behavior through the fused graph."""
    _select_fused_path(monkeypatch)
    functional_tests.test_real_weight_batch32_prefill_and_decode(mesh_device, layer_idx)
    counts = FusedDecoder.TOTAL_FUSION_COUNTS
    assert counts["mlp_silu_multiply"] >= 2
    if layer_idx == 0:
        assert counts["shared_lhs_linear_projections"] >= 2
        assert counts["tile_aligned_linear_projection_tails"] >= 2
        assert counts["repeat_interleave_qk_heads"] >= 4
    else:
        assert counts["shared_lhs_full_qkv"] >= 2
        assert counts["dedicated_full_qkv_heads"] >= 2
        assert counts["attention_sigmoid_multiply"] >= 2


@pytest.mark.parametrize("logical_length", [32769, 262144], ids=["non_aligned_long", "native_max"])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_fused_full_attention_advertised_context_prefill(monkeypatch, mesh_device, logical_length):
    """Run the fused paged/chunked graph at both advertised long boundaries."""
    _select_fused_path(monkeypatch)
    functional_tests.test_full_attention_advertised_context_prefill(mesh_device, logical_length)
    counts = FusedDecoder.TOTAL_FUSION_COUNTS
    assert counts["shared_lhs_full_qkv"] > 0
    assert counts["dedicated_full_qkv_heads"] > 0
    assert counts["partial_rope"] > 0
    assert counts["attention_sigmoid_multiply"] > 0


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_fused_linear_attention_advertised_context_prefill(monkeypatch, mesh_device):
    """Run all 262,144 tokens through the fused DeltaNet chunk graph."""
    _select_fused_path(monkeypatch)
    functional_tests.test_linear_attention_advertised_context_prefill(mesh_device)
    counts = FusedDecoder.TOTAL_FUSION_COUNTS
    assert counts["shared_lhs_linear_projections"] > 0
    assert counts["tile_aligned_linear_projection_tails"] > 0
    assert counts["l2_rms_norm"] > 0
    assert counts["repeat_interleave_qk_heads"] > 0


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_fused_linear_attention_advertised_context_decode(monkeypatch, mesh_device):
    """Build the native prefix and trace fused DeltaNet decode at its final position."""
    _select_fused_path(monkeypatch)
    functional_tests.test_linear_attention_advertised_context_decode(mesh_device)
    counts = FusedDecoder.TOTAL_FUSION_COUNTS
    assert counts["shared_lhs_linear_projections"] > 0
    assert counts["tile_aligned_linear_projection_tails"] > 0
    assert counts["decode_beta_sigmoid_multiply"] >= 2
    assert counts["matmul_transpose_flags"] > 0


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["linear_attention", "full_attention"])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_fused_decoder_perf(monkeypatch, mesh_device, layer_idx):
    """Profiler entry point for warmed fused prefill and traced fused decode."""
    _select_fused_path(monkeypatch)
    functional_tests.test_functional_decoder_perf(mesh_device, layer_idx)
    counts = FusedDecoder.TOTAL_FUSION_COUNTS
    assert counts["mlp_silu_multiply"] >= 1
    if layer_idx == 0:
        assert counts["shared_lhs_linear_projections"] >= 1
        assert counts["tile_aligned_linear_projection_tails"] >= 1
        assert counts["linear_silu_multiply"] >= 1
        assert counts["matmul_transpose_flags"] >= 1
        assert counts["l2_rms_norm"] >= 2
    else:
        assert counts["shared_lhs_full_qkv"] >= 1
        assert counts["dedicated_full_qkv_heads"] >= 1
        assert counts["attention_sigmoid_multiply"] >= 1

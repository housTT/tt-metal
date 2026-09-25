# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Correctness, paging, trace, capability and path gates for OptimizedDecoder."""

from __future__ import annotations

import inspect

import pytest

from models.autoports.qwen_qwen3_8_flash_next.tests import test_functional_decoder as functional_gates
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder, POLICIES

LAYER_KINDS = (0, 1, 3)


def _optimized(monkeypatch):
    monkeypatch.setattr(functional_gates, "FunctionalDecoder", OptimizedDecoder)
    assert functional_gates.FunctionalDecoder is OptimizedDecoder


def test_optimized_class_and_source_contract():
    assert issubclass(OptimizedDecoder, FusedDecoder)
    assert OptimizedDecoder is not FusedDecoder
    assert OptimizedDecoder.__name__ == "OptimizedDecoder"
    assert "bfp4_lofi_routed_experts" in OptimizedDecoder.OPTIMIZATION_MANIFEST
    assert "dram_sharded_decode_attention_projections" in OptimizedDecoder.OPTIMIZATION_MANIFEST
    assert OptimizedDecoder.DEFAULT_POLICY == "expert_bfp4_lofi_g40b16_d40b5"
    assert OptimizedDecoder.DEFAULT_DECODE_EXPERT_MODE == "indexed"
    assert OptimizedDecoder.DEFAULT_PREFILL_OUTPUT == "l1"
    assert OptimizedDecoder.DEFAULT_DRAM_SHARDED_ROLES == ("qsa_input", "attn_out")
    tp4_reference = POLICIES["expert_bfp4_lofi_g10b16_d40b5"]
    assert (tp4_reference.gate_up_cores, tp4_reference.gate_up_in0_block_w) == (10, 16)
    assert (tp4_reference.down_cores, tp4_reference.down_in0_block_w) == (40, 5)
    assert "gdn_qkv_b_a" not in OptimizedDecoder.DEFAULT_DECODE_1D_CONFIG
    # The former layer-0 55-core packed projection passed one-token PCC but
    # accumulated severe repeated-transition error.  Keep the parser's scoped-role
    # support without advertising that rejected geometry.
    assert OptimizedDecoder._parse_decode_1d_config("gdn_qkv_b_a@0:55", layer_idx=0) == {"gdn_qkv_b_a": 55}
    assert OptimizedDecoder._parse_decode_1d_config("gdn_qkv_b_a@0:55", layer_idx=1) == {}
    for role in ("in_proj_z", "gdn_out", "moe_input", "qsa_input", "ple_key_value"):
        assert f"{role}:" in OptimizedDecoder.DEFAULT_DECODE_1D_CONFIG
    source = inspect.getsource(OptimizedDecoder)
    for required in ("expert_weight_dtype", "expert_compute_cfg", "_sparse_matmul_config"):
        assert required in source
    for forbidden in ("ttnn.from_torch", "ttnn.to_torch", "ttnn.as_tensor", "torch."):
        assert forbidden not in source


def test_target_shape_and_layer_kind_contract(monkeypatch):
    _optimized(monkeypatch)
    functional_gates.test_target_shape_and_layer_kind_contract()


@pytest.mark.parametrize("seq_len", [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049])
def test_prefill_plan_accepts_non_aligned_boundaries(monkeypatch, seq_len):
    _optimized(monkeypatch)
    functional_gates.test_prefill_plan_accepts_non_aligned_boundaries(seq_len)


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_real_weights_hf_prefill_decode_pcc(monkeypatch, mesh_device, layer_idx):
    _optimized(monkeypatch)
    functional_gates.test_real_weights_hf_prefill_decode_pcc(mesh_device, layer_idx)


@pytest.mark.skipif(
    functional_gates.os.getenv("RUN_QWEN38_PROGRESSING_HF_DIAGNOSTIC") != "1",
    reason="explicit HF state diagnostic",
)
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_real_weights_progressing_decode_against_hf(monkeypatch, mesh_device, layer_idx):
    _optimized(monkeypatch)
    functional_gates.test_real_weights_progressing_decode_against_hf(mesh_device, layer_idx)


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_decode_trace_replay_and_determinism(monkeypatch, mesh_device, layer_idx):
    _optimized(monkeypatch)
    functional_gates.test_decode_trace_replay_and_determinism(mesh_device, layer_idx)


def test_runtime_source_has_no_host_fallback(monkeypatch):
    _optimized(monkeypatch)
    functional_gates.test_runtime_source_has_no_host_fallback()


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_exact_target_shapes_run_non_aligned_prefill_boundaries(monkeypatch, mesh_device, layer_idx):
    _optimized(monkeypatch)
    functional_gates.test_exact_target_shapes_run_non_aligned_prefill_boundaries(mesh_device, layer_idx)


def test_qsa_prefill_page_table_permutation_is_semantically_invariant(monkeypatch, mesh_device):
    _optimized(monkeypatch)
    functional_gates.test_qsa_prefill_page_table_permutation_is_semantically_invariant(mesh_device)


def test_qsa_batched_decode_uses_each_page_table_and_current_position(monkeypatch, mesh_device):
    _optimized(monkeypatch)
    functional_gates.test_qsa_batched_decode_uses_each_page_table_and_current_position(mesh_device)


def test_qsa_underfilled_selected_token_multiset(monkeypatch, mesh_device):
    _optimized(monkeypatch)
    functional_gates.test_qsa_underfilled_selected_token_multiset(mesh_device)


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_two_user_prefill_then_batched_decode(monkeypatch, mesh_device, layer_idx):
    _optimized(monkeypatch)
    functional_gates.test_two_user_prefill_then_batched_decode(mesh_device, layer_idx)


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_batch_32_decode_contract(monkeypatch, mesh_device, layer_idx):
    _optimized(monkeypatch)
    functional_gates.test_batch_32_decode_contract(mesh_device, layer_idx)


def test_qsa_public_prefill_long_non_aligned(monkeypatch, mesh_device):
    _optimized(monkeypatch)
    functional_gates.test_qsa_public_prefill_long_non_aligned(mesh_device)


@pytest.mark.long_context
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.timeout(3600)
def test_full_advertised_context(monkeypatch, mesh_device, layer_idx):
    _optimized(monkeypatch)
    functional_gates.test_full_advertised_context(mesh_device, layer_idx)


@pytest.mark.long_context
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.timeout(3600)
def test_near_max_non_aligned_context(monkeypatch, mesh_device, layer_idx):
    _optimized(monkeypatch)
    functional_gates.test_near_max_non_aligned_context(mesh_device, layer_idx)


@pytest.mark.long_context
@pytest.mark.timeout(600)
def test_qsa_traced_decode_at_advertised_context(monkeypatch, mesh_device):
    _optimized(monkeypatch)
    functional_gates.test_qsa_traced_decode_at_advertised_context(mesh_device)

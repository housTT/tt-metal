# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused-decoder correctness, paging, trace and capability gates.

The functional stage owns the exhaustive oracle helpers.  These wrappers
replace that module's decoder symbol with :class:`FusedDecoder` for the entire
call, so the same semantic gates execute the fused runtime without copying or
weakening their assertions.
"""

from __future__ import annotations

import inspect

import pytest

from models.autoports.qwen_qwen3_8_flash_next.tests import test_functional_decoder as functional_gates
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder

LAYER_KINDS = (0, 1, 3)


def _fused(monkeypatch):
    monkeypatch.setattr(functional_gates, "FunctionalDecoder", FusedDecoder)
    assert functional_gates.FunctionalDecoder is FusedDecoder


def test_fused_class_and_source_contract():
    assert FusedDecoder is not functional_gates.FunctionalDecoder
    assert FusedDecoder.__name__ == "FusedDecoder"
    assert len(FusedDecoder.FUSION_MANIFEST) >= 10
    source = inspect.getsource(FusedDecoder)
    for required in (
        "ttnn.mac",
        "rotary_embedding_hf",
        "qkv_causal_conv1d_silu",
        "paged_fused_update_cache",
        "split_query_key_value_and_split_heads",
        "input_tensor_b_activations",
        "_recurrent_gdn_decode",
    ):
        assert required in source
    for forbidden in (
        "ttnn.from_torch",
        "ttnn.to_torch",
        "ttnn.as_tensor",
        "torch.",
        "ttnn.transformer.concatenate_heads",
    ):
        assert forbidden not in source


def test_prefill_state_outputs_copy_into_fixed_buffers():
    functional_source = inspect.getsource(functional_gates.FunctionalDecoder)
    fused_source = inspect.getsource(FusedDecoder)

    assert functional_source.count("self._update_prefill_state(") == 3
    assert fused_source.count("self._update_prefill_state(") == 4
    for source in (functional_source, fused_source):
        for forbidden in (
            "ttnn.deallocate(self.user_recurrent_state",
            "ttnn.deallocate(self.user_conv_state",
            "ttnn.deallocate(self.user_ple_conv_state",
            "self.user_recurrent_state[user_id] =",
            "self.user_conv_state[user_id] =",
            "self.user_ple_conv_state[user_id] =",
        ):
            assert forbidden not in source


def test_target_shape_and_layer_kind_contract(monkeypatch):
    _fused(monkeypatch)
    functional_gates.test_target_shape_and_layer_kind_contract()


@pytest.mark.parametrize("seq_len", [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049])
def test_prefill_plan_accepts_non_aligned_boundaries(monkeypatch, seq_len):
    _fused(monkeypatch)
    functional_gates.test_prefill_plan_accepts_non_aligned_boundaries(seq_len)


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_real_weights_hf_prefill_decode_pcc(monkeypatch, mesh_device, layer_idx):
    _fused(monkeypatch)
    functional_gates.test_real_weights_hf_prefill_decode_pcc(mesh_device, layer_idx)


@pytest.mark.skipif(
    functional_gates.os.getenv("RUN_QWEN38_PROGRESSING_HF_DIAGNOSTIC") != "1",
    reason="explicit HF state diagnostic",
)
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_real_weights_progressing_decode_against_hf(monkeypatch, mesh_device, layer_idx):
    _fused(monkeypatch)
    functional_gates.test_real_weights_progressing_decode_against_hf(mesh_device, layer_idx)


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_decode_trace_replay_and_determinism(monkeypatch, mesh_device, layer_idx):
    _fused(monkeypatch)
    functional_gates.test_decode_trace_replay_and_determinism(mesh_device, layer_idx)


def test_runtime_source_has_no_host_fallback(monkeypatch):
    _fused(monkeypatch)
    functional_gates.test_runtime_source_has_no_host_fallback()


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_exact_target_shapes_run_non_aligned_prefill_boundaries(monkeypatch, mesh_device, layer_idx):
    _fused(monkeypatch)
    functional_gates.test_exact_target_shapes_run_non_aligned_prefill_boundaries(mesh_device, layer_idx)


def test_qsa_prefill_page_table_permutation_is_semantically_invariant(monkeypatch, mesh_device):
    _fused(monkeypatch)
    functional_gates.test_qsa_prefill_page_table_permutation_is_semantically_invariant(mesh_device)


def test_qsa_batched_decode_uses_each_page_table_and_current_position(monkeypatch, mesh_device):
    _fused(monkeypatch)
    functional_gates.test_qsa_batched_decode_uses_each_page_table_and_current_position(mesh_device)


def test_qsa_underfilled_selected_token_multiset(monkeypatch, mesh_device):
    _fused(monkeypatch)
    functional_gates.test_qsa_underfilled_selected_token_multiset(mesh_device)


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_two_user_prefill_then_batched_decode(monkeypatch, mesh_device, layer_idx):
    _fused(monkeypatch)
    functional_gates.test_two_user_prefill_then_batched_decode(mesh_device, layer_idx)


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_batch_32_decode_contract(monkeypatch, mesh_device, layer_idx):
    _fused(monkeypatch)
    functional_gates.test_batch_32_decode_contract(mesh_device, layer_idx)


def test_qsa_public_prefill_long_non_aligned(monkeypatch, mesh_device):
    _fused(monkeypatch)
    functional_gates.test_qsa_public_prefill_long_non_aligned(mesh_device)


@pytest.mark.long_context
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.timeout(3600)
def test_full_advertised_context(monkeypatch, mesh_device, layer_idx):
    _fused(monkeypatch)
    functional_gates.test_full_advertised_context(mesh_device, layer_idx)


@pytest.mark.long_context
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.timeout(3600)
def test_near_max_non_aligned_context(monkeypatch, mesh_device, layer_idx):
    _fused(monkeypatch)
    functional_gates.test_near_max_non_aligned_context(mesh_device, layer_idx)


@pytest.mark.long_context
@pytest.mark.timeout(600)
def test_qsa_traced_decode_at_advertised_context(monkeypatch, mesh_device):
    _fused(monkeypatch)
    functional_gates.test_qsa_traced_decode_at_advertised_context(mesh_device)

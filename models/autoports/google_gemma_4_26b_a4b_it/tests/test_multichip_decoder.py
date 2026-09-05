# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TP=1/2/4 profile, correctness, cache, trace, and performance gates."""

from __future__ import annotations

import inspect
import json
import math
import os
import time
from pathlib import Path

import pytest

import models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder as functional_tests
import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    FULL_KIND,
    HIDDEN_SIZE,
    PREFILL_SLIDING_CHUNK_SIZE,
    SLIDING_KIND,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import (
    ADVERTISED_CONTEXT_TOKENS,
    SUPPORTED_TP_SIZES,
    MultichipDecoder,
    _capacity_expert_gate_up_dtype,
    _construct_raw_weight_decoder,
    _multichip_bool_from_env,
    _multichip_dram_workers_per_bank,
    _multichip_graph_fusion_policy,
    _multichip_optimized_defaults,
    _multichip_residual_shard_cores,
    _packed_expert_gate_up_mesh_source,
    _packed_gate_up_mesh_source,
    _pad_last,
    _pad_penultimate,
    _prefill_context_limit,
    _profile_for_tp,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import (
    OptimizedDecoder,
    _optimized_sparse_decode_config,
    _optimized_sparse_prefill_config,
    _prepare_folded_state_dict,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

TP4_PROFILE = _profile_for_tp(4)
TP_SIZE = TP4_PROFILE.tp_size
PADDED_MLP_INTERMEDIATE_SIZE = TP4_PROFILE.padded_mlp_intermediate_size
PADDED_MOE_INTERMEDIATE_SIZE = TP4_PROFILE.padded_moe_intermediate_size


def _measurement_artifact_suffix() -> str:
    """Keep watcher and opt-in candidate measurements away from selected evidence."""

    suffix = os.getenv("GEMMA4_MULTICHIP_ARTIFACT_SUFFIX")
    if suffix is None:
        return "_watcher" if os.getenv("TT_METAL_WATCHER") else ""
    if not suffix.startswith("_") or not suffix[1:].replace("_", "").isalnum():
        raise ValueError("GEMMA4_MULTICHIP_ARTIFACT_SUFFIX must be an underscore-prefixed identifier")
    return suffix


@pytest.mark.parametrize(
    "tp_size, expected",
    [
        pytest.param(1, (2112, 2112, 704, 704, 16, 8, 2), id="tp1"),
        pytest.param(2, (2112, 1056, 704, 352, 8, 4, 1), id="tp2"),
        pytest.param(4, (2176, 544, 768, 192, 4, 2, 1), id="tp4"),
    ],
)
def test_multichip_shape_contract(tp_size, expected):
    assert SUPPORTED_TP_SIZES == (1, 2, 4)
    profile = _profile_for_tp(tp_size)
    assert profile.__dataclass_params__.frozen
    assert profile.tp_size == tp_size
    assert (
        profile.padded_mlp_intermediate_size,
        profile.local_mlp_intermediate_size,
        profile.padded_moe_intermediate_size,
        profile.local_moe_intermediate_size,
        profile.local_q_heads,
        profile.local_sliding_kv_heads,
        profile.local_full_kv_heads,
    ) == expected
    assert profile.padded_mlp_intermediate_size % (tp_size * 32) == 0
    assert profile.padded_moe_intermediate_size % (tp_size * 32) == 0


@pytest.mark.parametrize("tp_size", [0, 3, 8])
def test_multichip_profile_rejects_unsupported_tp_size(tp_size):
    try:
        _profile_for_tp(tp_size)
    except ValueError as error:
        assert "supports TP sizes (1, 2, 4)" in str(error)
        assert f"got {tp_size}" in str(error)
    else:
        pytest.fail(f"unsupported TP size {tp_size} was accepted")


def test_multichip_profile_is_wired_to_runtime_and_tp4_full_kv_pairs():
    source = inspect.getsource(MultichipDecoder.from_state_dict)
    assert "profile = _require_target_mesh(mesh_device)" in source
    for attribute in ("tp_profile", "tp_size", "local_q_heads", "local_sliding_kv_heads", "local_full_kv_heads"):
        assert f"decoder.{attribute} =" in source
    assert "if tp_size <= 2:" in source
    assert "k_shards = k.chunk(tp_size" in source
    assert "v_shards = v.chunk(tp_size" in source
    assert "k_heads[rank // 2]" in source
    assert "v_heads[rank // 2]" in source


def test_multichip_profile_expert_decode_geometry_is_legal():
    class FakeDevice:
        @staticmethod
        def compute_with_storage_grid_size():
            return ttnn.CoreCoord(11, 10)

    for tp_size, expected_per_core_n in ((1, 2), (2, 1), (4, 2)):
        profile = _profile_for_tp(tp_size)
        expert_gate_tiles = profile.local_moe_intermediate_size // 32
        per_core_n = 2 if expert_gate_tiles % 2 == 0 else 1
        assert per_core_n == expected_per_core_n
        config = _optimized_sparse_decode_config(
            FakeDevice(),
            n=profile.local_moe_intermediate_size,
            per_core_n=per_core_n,
            in0_block_w=44,
            out_subblock_w=None,
        )
        assert config.per_core_N == expected_per_core_n
        prefill_config = _optimized_sparse_prefill_config(
            FakeDevice(),
            n=profile.local_moe_intermediate_size,
            groups=1,
            requested_per_core_n=1 if tp_size == 2 else 2,
            in0_block_w=44,
        )
        assert prefill_config.per_core_N == expected_per_core_n

    source = inspect.getsource(MultichipDecoder.from_state_dict)
    assert "expert_gate_tiles = profile.local_moe_intermediate_size // TILE_SIZE" in source
    assert 'kwargs.setdefault("expert_gate_per_core_n", 2 if expert_gate_tiles % 2 == 0 else 1)' in source
    assert 'kwargs.setdefault("prefill_expert_per_core_n", 1 if tp_size == 2 else 2)' in source


def test_multichip_capacity_precision_policy_is_profile_and_layer_specific():
    assert _capacity_expert_gate_up_dtype(1, SLIDING_KIND) == ttnn.bfloat8_b
    assert _capacity_expert_gate_up_dtype(1, FULL_KIND) == ttnn.bfloat4_b
    for tp_size in (2, 4):
        assert _capacity_expert_gate_up_dtype(tp_size, SLIDING_KIND) == ttnn.bfloat8_b
        assert _capacity_expert_gate_up_dtype(tp_size, FULL_KIND) == ttnn.bfloat8_b

    source = inspect.getsource(MultichipDecoder.from_state_dict)
    assert "capacity_expert_dtype = _capacity_expert_gate_up_dtype(tp_size, kind)" in source
    assert "expert_gate_weight_dtype = expert_gate_weight_dtype or capacity_expert_dtype" in source
    assert "expert_up_weight_dtype = expert_up_weight_dtype or capacity_expert_dtype" in source


def test_multichip_prefill_context_limit_is_profile_specific(expect_error):
    assert tuple(_prefill_context_limit(tp_size) for tp_size in SUPPORTED_TP_SIZES) == (50624, 262144, 262144)
    with expect_error(ValueError, "supports TP sizes"):
        _prefill_context_limit(3)

    class TooLong:
        shape = (1, 1, 50625, HIDDEN_SIZE)

    decoder = object.__new__(MultichipDecoder)
    decoder.tp_size = 1
    with expect_error(ValueError, "TP1 prefill length 50625 exceeds the 50624-token"):
        decoder.prefill_forward(TooLong())


def test_multichip_capacity_projection_is_internally_consistent():
    projection_path = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/capacity_projection.json")
    projection = json.loads(projection_path.read_text())
    context_contract = json.loads((projection_path.parent.parent / "context_contract.json").read_text())
    context_profiles = context_contract["multichip_decoder"]["profiles"]
    assert projection["advertised_context_tokens"] == ADVERTISED_CONTEXT_TOKENS == 262144
    assert projection["kv_cache_dtype"] == "BF16"
    assert projection["capacity_basis_bytes_per_device"] == 32 * 1024**3
    terminal = projection["terminal_contract"]
    lifetime = projection["source_lifetime_contract"]
    packed_expert = projection["packed_expert_decode_contract"]
    assert terminal["status"].startswith("mandatory downstream full-model placement contract")
    assert terminal["embedding_and_lm_head_dtype"] == "BFP8_B"
    assert terminal["tiles_per_matrix"] == (terminal["vocab_size"] // 32) * (terminal["hidden_size"] // 32)
    assert terminal["unsharded_embedding_and_lm_head_bytes"] == (
        2 * terminal["tiles_per_matrix"] * terminal["bfp8_tile_bytes"]
    )
    assert terminal["bf16_rejected_control_bytes"] == 2 * terminal["vocab_size"] * terminal["hidden_size"] * 2
    assert terminal["final_norm_bytes_per_device"] == 32 * HIDDEN_SIZE * 2
    assert PREFILL_SLIDING_CHUNK_SIZE == 30720
    assert lifetime["sequence_tile_tokens"] == 32
    assert lifetime["full_attention_cache_block_tokens"] == 128
    assert lifetime["moe_chunk_tokens"] == 1024
    assert lifetime["trace_region_bytes_per_device"] == 64 * 1024**2
    assert packed_expert["num_layers"] == 30
    assert packed_expert["num_experts"] == 128
    assert packed_expert["hidden_size"] == HIDDEN_SIZE
    assert packed_expert["dtype"] == "BFP8_B"
    assert packed_expert["tile_bytes"] == terminal["bfp8_tile_bytes"] == 1088
    assert packed_expert["bfp4_lower_bound_tile_bytes"] == 576
    assert packed_expert["enabled_profiles"] == ["P150x4"]
    assert tuple(packed_expert["disabled_profiles"]) == ("P150", "P150x2")
    assert tuple(profile["tp_size"] for profile in projection["profiles"].values()) == SUPPORTED_TP_SIZES
    for profile_name, profile in projection["profiles"].items():
        tp_size = profile["tp_size"]
        tp_profile = _profile_for_tp(tp_size)
        logical_tokens = profile["worst_case_logical_context_tokens"]
        padded_tokens = math.ceil(logical_tokens / 32) * 32
        cache_tokens = math.ceil(profile["supported_context_tokens"] / 128) * 128
        expected_packed_tiles = (
            packed_expert["num_experts"]
            * (packed_expert["hidden_size"] // 32)
            * (2 * tp_profile.local_moe_intermediate_size // 32)
        )
        expected_packed_bytes_per_layer = expected_packed_tiles * packed_expert["tile_bytes"]
        packed_enabled = profile_name in packed_expert["enabled_profiles"]
        expected_packed_retained_bytes = (
            expected_packed_bytes_per_layer * packed_expert["num_layers"] if packed_enabled else 0
        )
        context_profile = context_profiles[profile_name]
        assert profile["supported_context_tokens"] == _prefill_context_limit(tp_size)
        assert profile["padded_activation_tokens"] == padded_tokens
        assert profile["kv_cache_capacity_tokens"] == cache_tokens
        assert profile["packed_expert_decode_enabled"] is packed_enabled
        assert profile["packed_expert_local_width"] == tp_profile.local_moe_intermediate_size
        assert profile["packed_expert_output_width"] == 2 * tp_profile.local_moe_intermediate_size
        assert profile["packed_expert_tiles_per_layer"] == expected_packed_tiles
        assert profile["packed_expert_bytes_per_layer"] == expected_packed_bytes_per_layer
        assert profile["packed_expert_retained_copy_bytes"] == expected_packed_retained_bytes
        assert profile["retained_decode_copy_bytes"] == (
            profile["prior_o_and_dense_retained_decode_copy_bytes"] + profile["packed_expert_retained_copy_bytes"]
        )
        assert context_profile["packed_expert_decode_enabled"] is packed_enabled
        assert context_profile["packed_expert_retained_copy_bytes_per_device"] == expected_packed_retained_bytes
        assert (
            context_profile["prior_o_and_dense_retained_copy_bytes_per_device"]
            == profile["prior_o_and_dense_retained_decode_copy_bytes"]
        )
        assert profile["decoder_weight_bytes"] == (
            profile["base_layer_stack_bytes"]
            + profile["packed_dense_extra_bytes"]
            + profile["retained_decode_copy_bytes"]
        )
        expected_sliding_kv = 25 * 1024 * tp_profile.local_sliding_kv_heads * 2 * 256 * 2
        expected_full_kv = 5 * cache_tokens * tp_profile.local_full_kv_heads * 2 * 512 * 2
        assert profile["sliding_kv_cache_bytes"] == expected_sliding_kv
        assert profile["full_kv_cache_bytes"] == expected_full_kv
        assert profile["kv_cache_bytes"] == expected_sliding_kv + expected_full_kv
        assert profile["projected_total_bytes"] == (
            profile["decoder_weight_bytes"] + profile["kv_cache_bytes"] + profile["persistent_ccl_bytes"]
        )
        assert profile["headroom_bytes"] == (
            projection["capacity_basis_bytes_per_device"] - profile["projected_total_bytes"]
        )
        assert profile["headroom_bytes"] > 0
        expected_terminal_bytes = terminal["unsharded_embedding_and_lm_head_bytes"] // tp_size
        full_hidden = padded_tokens * HIDDEN_SIZE * 2
        full_qkv_width = (tp_profile.local_q_heads + 2 * tp_profile.local_full_kv_heads) * FULL_KIND.head_dim
        full_qkv = padded_tokens * full_qkv_width * 2
        full_q = padded_tokens * tp_profile.local_q_heads * FULL_KIND.head_dim * 2
        full_rope = 2 * padded_tokens * FULL_KIND.head_dim * 2
        caller_padding = (
            logical_tokens * (HIDDEN_SIZE + 2 * FULL_KIND.head_dim) * 2 if logical_tokens != padded_tokens else 0
        )
        attention_concat = 2 * full_hidden + 2 * full_qkv + 2 * full_q + full_rope + caller_padding
        attention_reduce = (4 if tp_size == 1 else 5) * full_hidden + 2 * full_qkv + full_q + full_rope + caller_padding
        dense_local = padded_tokens * tp_profile.local_mlp_intermediate_size * 2
        dense_mlp = (6 if tp_size == 1 else 7) * full_hidden + 5 * dense_local + full_rope + caller_padding
        moe_chunk = lifetime["moe_chunk_tokens"]
        hidden_chunk = moe_chunk * HIDDEN_SIZE * 2
        grouped_local = 128 * moe_chunk * tp_profile.local_moe_intermediate_size * 2
        dispatch = 128 * moe_chunk * HIDDEN_SIZE * 2
        router = padded_tokens * 128 * 2
        router_chunk = moe_chunk * 128 * 2
        moe_outer_hidden_count = 8 if tp_size == 4 else 7
        moe = (
            (moe_outer_hidden_count + 2) * full_hidden
            + hidden_chunk
            + 3 * grouped_local
            + 2 * dispatch
            + router
            + full_rope
            + 2 * router_chunk
            + caller_padding
        )
        activation_peak = max(attention_concat, attention_reduce, dense_mlp, moe)
        assert profile["terminal_embedding_and_lm_head_bytes"] == expected_terminal_bytes
        assert profile["final_norm_bytes"] == terminal["final_norm_bytes_per_device"]
        assert profile["trace_region_bytes"] == lifetime["trace_region_bytes_per_device"]
        assert profile["full_hidden_bytes"] == full_hidden
        assert profile["full_local_qkv_bytes"] == full_qkv
        assert profile["full_local_q_bytes"] == full_q
        assert profile["full_rope_bytes"] == full_rope
        assert profile["nonaligned_caller_padding_bytes"] == caller_padding
        assert profile["attention_concat_peak_bytes"] == attention_concat
        assert profile["attention_output_reduce_peak_bytes"] == attention_reduce
        assert profile["dense_mlp_peak_bytes"] == dense_mlp
        assert profile["moe_peak_bytes"] == moe
        assert profile["activation_transient_reserve_bytes"] == activation_peak
        assert profile["operational_reserve_bytes"] == (
            profile["trace_region_bytes"]
            + profile["activation_transient_reserve_bytes"]
            + profile["allocator_reserve_bytes"]
        )
        assert profile["full_stack_projected_total_bytes"] == (
            profile["projected_total_bytes"]
            + profile["terminal_embedding_and_lm_head_bytes"]
            + profile["final_norm_bytes"]
            + profile["operational_reserve_bytes"]
        )
        assert profile["full_stack_headroom_bytes"] == (
            projection["capacity_basis_bytes_per_device"] - profile["full_stack_projected_total_bytes"]
        )
        assert profile["full_stack_fits"] is (profile["full_stack_headroom_bytes"] > 0)
        assert profile["full_stack_fits"]
        assert context_profile["capacity_projection_bytes_per_device"] == profile["projected_total_bytes"]
        assert context_profile["full_stack_projection_bytes_per_device"] == profile["full_stack_projected_total_bytes"]
        assert context_profile["full_stack_headroom_bytes_per_device"] == profile["full_stack_headroom_bytes"]
        assert context_profile["supported_context_tokens"] == profile["supported_context_tokens"]

    p150 = projection["profiles"]["P150"]
    assert "BFP4 expert gate/up" in p150["precision_exception"]
    assert p150["packed_expert_all_bfp8_candidate_bytes"] == (
        p150["packed_expert_bytes_per_layer"] * packed_expert["num_layers"]
    )
    assert p150["packed_expert_all_bfp4_lower_bound_bytes"] == (
        p150["packed_expert_tiles_per_layer"]
        * packed_expert["num_layers"]
        * packed_expert["bfp4_lower_bound_tile_bytes"]
    )
    assert p150["packed_expert_all_bfp4_lower_bound_bytes"] > p150["headroom_bytes"]
    assert p150["advertised_context_full_stack_projected_total_bytes"] > projection["capacity_basis_bytes_per_device"]
    assert p150["advertised_context_full_stack_headroom_bytes"] < 0
    assert p150["first_rejected_context_tokens"] == p150["supported_context_tokens"] + 1
    assert p150["first_rejected_headroom_bytes"] < 0
    assert p150["aligned_limit_full_stack_headroom_bytes"] > 0
    assert projection["profiles"]["P150x2"]["precision_exception"] is None
    assert projection["profiles"]["P150x4"]["precision_exception"] is None


def test_multichip_measurement_artifacts_isolate_watcher_and_candidates(monkeypatch):
    monkeypatch.delenv("GEMMA4_MULTICHIP_ARTIFACT_SUFFIX", raising=False)
    monkeypatch.delenv("TT_METAL_WATCHER", raising=False)
    assert _measurement_artifact_suffix() == ""
    monkeypatch.setenv("TT_METAL_WATCHER", "10")
    assert _measurement_artifact_suffix() == "_watcher"
    monkeypatch.setenv("GEMMA4_MULTICHIP_ARTIFACT_SUFFIX", "_qkv_dram_b1")
    assert _measurement_artifact_suffix() == "_qkv_dram_b1"
    monkeypatch.setenv("GEMMA4_MULTICHIP_ARTIFACT_SUFFIX", "../escape")
    try:
        _measurement_artifact_suffix()
    except ValueError as error:
        assert "underscore-prefixed identifier" in str(error)
    else:
        pytest.fail("unsafe artifact suffix was accepted")


def test_multichip_raw_weight_policy_restores_routing_environment(monkeypatch):
    class CapturePolicy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.routing_env = os.environ.get("GEMMA4_OPT_ROUTING_ROW_MAJOR")

    monkeypatch.delenv("GEMMA4_OPT_ROUTING_ROW_MAJOR", raising=False)
    decoder = _construct_raw_weight_decoder(CapturePolicy)
    assert decoder.routing_env == "0"
    assert "GEMMA4_OPT_ROUTING_ROW_MAJOR" not in os.environ
    for name in (
        "packed_expert_decode_gate_up",
        "packed_expert_prefill_gate_up",
        "folded_router_projection",
        "shared_ffn_norm",
        "folded_expert_scale",
        "fused_final_scalar",
    ):
        assert decoder.kwargs[name] is False

    monkeypatch.setenv("GEMMA4_OPT_ROUTING_ROW_MAJOR", "1")
    explicit = _construct_raw_weight_decoder(CapturePolicy)
    assert explicit.routing_env == "1"
    assert os.environ["GEMMA4_OPT_ROUTING_ROW_MAJOR"] == "1"

    source = inspect.getsource(MultichipDecoder.from_state_dict)
    assert "_construct_raw_weight_decoder(" in source

    monkeypatch.delenv("GEMMA4_OPT_ROUTING_ROW_MAJOR")
    prepared = _construct_raw_weight_decoder(
        CapturePolicy,
        _prepared_graph_folds=True,
        _prepared_packed_expert_decode=True,
        folded_router_projection=True,
        shared_ffn_norm=True,
        folded_expert_scale=True,
        fused_final_scalar=True,
        packed_expert_decode_gate_up=True,
        residual_shard_cores=22,
    )
    assert prepared.routing_env == "1"
    assert "GEMMA4_OPT_ROUTING_ROW_MAJOR" not in os.environ
    assert all(
        prepared.kwargs[name]
        for name in (
            "folded_router_projection",
            "shared_ffn_norm",
            "folded_expert_scale",
            "fused_final_scalar",
            "packed_expert_decode_gate_up",
        )
    )
    assert prepared.kwargs["packed_expert_prefill_gate_up"] is False

    monkeypatch.setenv("GEMMA4_OPT_ROUTING_ROW_MAJOR", "0")
    opted_out = _construct_raw_weight_decoder(
        CapturePolicy,
        _prepared_graph_folds=True,
        folded_expert_scale=True,
        residual_shard_cores=22,
    )
    assert opted_out.routing_env == "0"
    assert os.environ["GEMMA4_OPT_ROUTING_ROW_MAJOR"] == "0"

    optimized_constructor_source = inspect.getsource(OptimizedDecoder.__init__)
    assert "not folded_expert_scale or residual_shard_cores == 0" in optimized_constructor_source


def test_multichip_optimized_candidate_policy_defaults_and_validation(monkeypatch, expect_error):
    graph_env = {
        "folded_router_projection": "GEMMA4_OPT_FOLDED_ROUTER_PROJECTION",
        "shared_ffn_norm": "GEMMA4_OPT_SHARED_FFN_NORM",
        "folded_expert_scale": "GEMMA4_OPT_FOLDED_EXPERT_SCALE",
        "fused_final_scalar": "GEMMA4_OPT_FUSED_FINAL_SCALAR",
    }
    for env_name in graph_env.values():
        monkeypatch.delenv(env_name, raising=False)
    assert all(_multichip_graph_fusion_policy({}).values())
    for tp_size in (1, 2, 4):
        sliding_graph, sliding_residual = _multichip_optimized_defaults(tp_size, SLIDING_KIND)
        assert all(sliding_graph.values())
        assert sliding_residual == 22
    for tp_size in (1, 2):
        full_graph, full_residual = _multichip_optimized_defaults(tp_size, FULL_KIND)
        assert all(full_graph.values())
        assert full_residual == 22
    tp4_full_graph, tp4_full_residual = _multichip_optimized_defaults(4, FULL_KIND)
    assert tp4_full_graph == {
        "folded_router_projection": False,
        "shared_ffn_norm": False,
        "folded_expert_scale": True,
        "fused_final_scalar": True,
    }
    assert tp4_full_residual == 0
    for selected, env_name in graph_env.items():
        for candidate_env in graph_env.values():
            monkeypatch.delenv(candidate_env, raising=False)
        monkeypatch.setenv(env_name, "0")
        policy = _multichip_graph_fusion_policy({})
        assert not policy[selected]
        assert sum(policy.values()) == len(graph_env) - 1

    monkeypatch.delenv("GEMMA4_MULTICHIP_RESIDUAL_SHARD_CORES", raising=False)
    assert _multichip_residual_shard_cores() == 22
    for value in ("0", "22"):
        monkeypatch.setenv("GEMMA4_MULTICHIP_RESIDUAL_SHARD_CORES", value)
        assert _multichip_residual_shard_cores() == int(value)
    for value in ("11", "tp22"):
        monkeypatch.setenv("GEMMA4_MULTICHIP_RESIDUAL_SHARD_CORES", value)
        with expect_error(ValueError, "must be 0 or 22"):
            _multichip_residual_shard_cores()

    monkeypatch.delenv("GEMMA4_MULTICHIP_DRAM_WORKERS_PER_BANK", raising=False)
    assert _multichip_dram_workers_per_bank() == 1
    for workers in (1, 2, 3):
        monkeypatch.setenv("GEMMA4_MULTICHIP_DRAM_WORKERS_PER_BANK", str(workers))
        assert _multichip_dram_workers_per_bank() == workers
    monkeypatch.setenv("GEMMA4_MULTICHIP_DRAM_WORKERS_PER_BANK_O_PROJ", "3")
    assert _multichip_dram_workers_per_bank("o_proj") == 3
    monkeypatch.setenv("GEMMA4_MULTICHIP_DRAM_WORKERS_PER_BANK_O_PROJ", "4")
    with expect_error(ValueError, "must be 1, 2, or 3"):
        _multichip_dram_workers_per_bank("o_proj")
    monkeypatch.setenv("GEMMA4_MULTICHIP_DRAM_WORKERS_PER_BANK_O_PROJ", "2")
    with expect_error(ValueError, "multi-reader DRAM primitive currently requires a unit MeshDevice"):
        _multichip_dram_workers_per_bank("o_proj", tp_size=4)

    monkeypatch.delenv("GEMMA4_MULTICHIP_PACKED_EXPERT_DECODE_GATE_UP", raising=False)
    assert not _multichip_bool_from_env("GEMMA4_MULTICHIP_PACKED_EXPERT_DECODE_GATE_UP")
    source = inspect.getsource(MultichipDecoder.from_state_dict)
    assert 'kwargs.pop("packed_expert_decode_gate_up", tp_size == 4)' in source
    monkeypatch.setenv("GEMMA4_MULTICHIP_PACKED_EXPERT_DECODE_GATE_UP", "yes")
    assert _multichip_bool_from_env("GEMMA4_MULTICHIP_PACKED_EXPERT_DECODE_GATE_UP")
    monkeypatch.setenv("GEMMA4_MULTICHIP_PACKED_EXPERT_DECODE_GATE_UP", "sometimes")
    with expect_error(ValueError, "must be a boolean"):
        _multichip_bool_from_env("GEMMA4_MULTICHIP_PACKED_EXPERT_DECODE_GATE_UP")


def test_multichip_graph_folds_commute_with_tp_weight_fracture():
    import torch

    prefix = "layers.0"
    state = {
        f"{prefix}.self_attn.q_proj.weight": torch.zeros(1),
        f"{prefix}.router.scale": torch.tensor([2.0, 3.0, 4.0]),
        f"{prefix}.router.proj.weight": torch.arange(6.0).reshape(2, 3),
        f"{prefix}.pre_feedforward_layernorm.weight": torch.tensor([0.5, 1.0, 1.5]),
        f"{prefix}.mlp.gate_proj.weight": torch.arange(24.0).reshape(8, 3),
        f"{prefix}.mlp.up_proj.weight": torch.arange(24.0, 48.0).reshape(8, 3),
        f"{prefix}.pre_feedforward_layernorm_2.weight": torch.tensor([1.5, 1.0, 0.5]),
        f"{prefix}.experts.gate_up_proj": torch.arange(48.0).reshape(2, 8, 3),
        f"{prefix}.router.per_expert_scale": torch.tensor([0.5, 2.0]),
        f"{prefix}.experts.down_proj": torch.arange(48.0).reshape(2, 3, 8),
    }
    originals = {name: tensor.clone() for name, tensor in state.items()}
    prepared = _prepare_folded_state_dict(
        state,
        layer_idx=0,
        folded_router_projection=True,
        shared_ffn_norm=True,
        folded_expert_scale=True,
    )
    assert all(torch.equal(state[name], original) for name, original in originals.items())
    for rank, shard in enumerate(prepared[f"{prefix}.mlp.gate_proj.weight"].transpose(-2, -1).chunk(4, -1)):
        expected = originals[f"{prefix}.mlp.gate_proj.weight"].chunk(4, 0)[rank]
        expected = expected * originals[f"{prefix}.pre_feedforward_layernorm.weight"].unsqueeze(0)
        assert torch.equal(shard.transpose(-2, -1), expected)
    for rank, shard in enumerate(prepared[f"{prefix}.experts.down_proj"].transpose(-2, -1).chunk(4, -2)):
        expected = originals[f"{prefix}.experts.down_proj"].transpose(-2, -1).chunk(4, -2)[rank]
        expected = expected * originals[f"{prefix}.router.per_expert_scale"].reshape(-1, 1, 1)
        assert torch.equal(shard, expected)

    source = inspect.getsource(MultichipDecoder.from_state_dict)
    assert source.index("_prepare_folded_state_dict(") < source.index("expert_gate = _pad_last(")
    assert "_folded_tensor_cache_path(" in source
    assert "decoder.weights = replace(decoder.weights, **obsolete_weights)" in source


def test_multichip_r22_uses_projection_boundaries_not_global_dense_geometry():
    tile_aligned_r22_width = 22 * 32
    assert _profile_for_tp(1).local_mlp_intermediate_size % tile_aligned_r22_width == 0
    assert _profile_for_tp(2).local_mlp_intermediate_size % tile_aligned_r22_width != 0
    assert _profile_for_tp(4).local_mlp_intermediate_size % tile_aligned_r22_width != 0
    attention_source = inspect.getsource(MultichipDecoder._attention_decode)
    dense_source = inspect.getsource(MultichipDecoder._dense_mlp)
    assert "_tracked_sharded_to_interleaved(" in attention_source
    assert '"attention_qkv_input"' in attention_source
    assert '"attention_output"' in attention_source
    assert "ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)" in dense_source
    assert "ttnn.to_memory_config(result, self.residual_memory_config" in dense_source


@pytest.mark.parametrize("tp_size", SUPPORTED_TP_SIZES)
def test_rank_local_packed_expert_source_and_geometry(tp_size):
    import torch

    local_width = _profile_for_tp(tp_size).local_moe_intermediate_size
    padded_width = local_width * tp_size
    gate = torch.arange(2 * padded_width, dtype=torch.float32).reshape(2, 1, padded_width)
    up = gate + 10000
    packed = _packed_expert_gate_up_mesh_source(gate, up, tp_size).squeeze(0)
    for rank, local in enumerate(packed.chunk(tp_size, dim=-1)):
        expected = torch.cat((up.chunk(tp_size, -1)[rank], gate.chunk(tp_size, -1)[rank]), dim=-1)
        torch.testing.assert_close(local, expected, rtol=0, atol=0)

    expert_gate_tiles = local_width // 32
    per_core_n = 2 if expert_gate_tiles % 2 == 0 else 1
    config = _optimized_sparse_decode_config(
        type("FakeDevice", (), {"compute_with_storage_grid_size": lambda self: ttnn.CoreCoord(11, 10)})(),
        n=2 * local_width,
        per_core_n=per_core_n,
        in0_block_w=44,
        out_subblock_w=None,
    )
    assert config.per_core_N == per_core_n

    source = inspect.getsource(MultichipDecoder.from_state_dict)
    assert "decoder.batch32_expert_gate = decoder.weights.expert_gate" in source
    assert "decoder.batch32_expert_up = decoder.weights.expert_up" in source


def test_multichip_multi_reader_contract_is_end_to_end():
    setup_source = inspect.getsource(MultichipDecoder.from_state_dict)
    linear_source = inspect.getsource(MultichipDecoder._linear)
    assert "_pad_dram_weight_for_readers(" in setup_source
    assert 'decoder.decode_weight_sources[role] = "independent_device_clone"' in setup_source
    assert "workers_per_bank=workers_per_bank" in setup_source
    assert "decode_dram_padded_input_widths.get(weight_name)" in linear_source
    assert "decode_dram_logical_output_widths.get(weight_name)" in linear_source
    assert '"o_proj": "2" if tp_size == 4 and kind.name == "sliding_attention" else "4"' in setup_source
    assert "decoder.decode_dram_batch32_configs[role]" in setup_source
    assert "if logical_rows is not None and logical_rows > 1" in linear_source
    attention_source = inspect.getsource(MultichipDecoder._attention_decode)
    assert "logical_rows=batch" in attention_source


def test_multichip_tp_padding_preserves_logical_values_and_zero_fills_tail():
    import torch

    last = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    padded_last = _pad_last(last, 8)
    torch.testing.assert_close(padded_last[..., :5], last, rtol=0, atol=0)
    assert torch.count_nonzero(padded_last[..., 5:]) == 0

    penultimate = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    padded_penultimate = _pad_penultimate(penultimate, 8)
    torch.testing.assert_close(padded_penultimate[..., :5, :], penultimate, rtol=0, atol=0)
    assert torch.count_nonzero(padded_penultimate[..., 5:, :]) == 0


def test_multichip_initializes_reachable_inherited_factory_state():
    source = inspect.getsource(MultichipDecoder.from_state_dict)
    for attribute in (
        "expert_weights",
        "packed_mlp_gate_up",
        "decode_dram_weights",
        "decode_dram_configs",
        "decode_dram_batch32_configs",
        "decode_dram_padded_input_widths",
        "decode_dram_logical_output_widths",
        "decode_dram_input_configs",
        "decode_dram_output_configs",
        "decode_routing_zero_base",
    ):
        assert f"decoder.{attribute} =" in source


@pytest.mark.parametrize("tp_size", SUPPORTED_TP_SIZES)
def test_multichip_cache_view_separates_update_and_sdpa_geometry(tp_size):
    decoder = object.__new__(MultichipDecoder)
    profile = _profile_for_tp(tp_size)
    decoder.local_full_kv_heads = profile.local_full_kv_heads
    decoder.layer_kind = FULL_KIND

    update_kwargs = decoder._cache_view_kwargs(prefill=False, cache_position_modulo=257)
    assert update_kwargs == {
        "block_size": FULL_KIND.block_size,
        "num_kv_heads": profile.local_full_kv_heads,
        "cache_position_modulo": 257,
    }

    sdpa_kwargs = decoder._sdpa_cache_view_kwargs(cache_position_modulo=257)
    assert set(sdpa_kwargs) == {"paged_cache_geometry", "cache_position_modulo"}
    geometry = sdpa_kwargs["paged_cache_geometry"]
    assert geometry.block_size == FULL_KIND.block_size
    assert geometry.num_kv_heads == profile.local_full_kv_heads

    decoder.layer_kind = SLIDING_KIND
    assert decoder._cache_view_kwargs(prefill=False) == {}
    assert decoder._sdpa_cache_view_kwargs() == {}

    decode_source = inspect.getsource(MultichipDecoder._attention_decode)
    assert "**cache_view" in decode_source
    assert "**self._sdpa_cache_view_kwargs" in decode_source


@pytest.mark.parametrize("tp_size", SUPPORTED_TP_SIZES)
def test_decode_only_packed_source_preserves_per_rank_gate_up_pairing(tp_size):
    """The independent precision copy must reproduce device-local concat."""
    import torch

    gate = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    up = 1000 + torch.arange(32, dtype=torch.float32).reshape(2, 16)
    packed = _packed_gate_up_mesh_source(gate, up, tp_size).squeeze(0).squeeze(0)
    mesh_shards = packed.chunk(tp_size, dim=-1)
    for rank, actual in enumerate(mesh_shards):
        expected = torch.cat(
            (gate.chunk(tp_size, dim=-1)[rank], up.chunk(tp_size, dim=-1)[rank]),
            dim=-1,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_decode_only_packed_dtype_uses_independent_host_source():
    source = inspect.getsource(MultichipDecoder.from_state_dict)
    assert 'role == "packed_mlp_gate_up"' in source
    assert "_packed_gate_up_mesh_source(mlp_gate, mlp_up, tp_size)" in source
    assert '"independent_host_upload"' in source
    assert "decoder.decode_weight_intermediates.append(candidate_weight)" in source


def test_decode_weight_selection_uses_phase_not_ambiguous_tile_count():
    class FakeTensor:
        shape = (1, 1, 32, HIDDEN_SIZE)

    decoder = object.__new__(MultichipDecoder)
    decoder.decode_dram_weights = {"packed_mlp_gate_up": object()}
    decoder.multichip_execution_phase = "prefill"
    assert not decoder._use_decode_dram_weight(FakeTensor(), "packed_mlp_gate_up")
    decoder.multichip_execution_phase = "decode"
    assert decoder._use_decode_dram_weight(FakeTensor(), "packed_mlp_gate_up")


def test_multichip_inherits_optimized_baseline_and_has_no_host_hot_path():
    assert issubclass(
        MultichipDecoder,
        __import__(
            "models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder", fromlist=["OptimizedDecoder"]
        ).OptimizedDecoder,
    )
    hot_methods = (
        MultichipDecoder._attention_prefill,
        MultichipDecoder._attention_decode,
        MultichipDecoder._dense_mlp,
        MultichipDecoder._moe_prefill,
        MultichipDecoder._moe_decode,
    )
    forbidden = ("torch.", "ttnn.from_torch", "ttnn.to_torch", ".cpu(", ".numpy(")
    for method in hot_methods:
        source = inspect.getsource(method)
        assert not any(token in source for token in forbidden), (method.__name__, source)


def test_multichip_preserves_active_expert_execution():
    source = inspect.getsource(MultichipDecoder._moe_decode) + inspect.getsource(MultichipDecoder._moe_prefill)
    assert "super()._moe_decode" in source
    assert "super()._moe_prefill" in source
    inherited = inspect.getsource(MultichipDecoder.__mro__[1]._moe_decode_single_user)
    assert "ttnn.sparse_matmul" in inherited
    assert "TOP_K_EXPERTS" in inherited


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_tp4_ring_all_reduce_smoke(mesh_device):
    """Shape-faithful hidden payload smoke with the exact final ring topology."""
    import torch

    if tuple(mesh_device.shape) != (1, 4):
        pytest.skip(f"requires target 1x4 mesh, got {tuple(mesh_device.shape)}")
    shards = torch.cat(
        [torch.full((1, 1, 32, 2816), float(rank + 1), dtype=torch.bfloat16) for rank in range(4)], dim=0
    )
    value = ttnn.from_torch(
        shards,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    reduced = ttnn.all_reduce(value, cluster_axis=1, topology=ttnn.Topology.Ring)
    host_shards = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(reduced.cpu())]
    for shard in host_shards:
        assert torch.equal(shard, torch.full_like(shard, 10.0))


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "role,global_k,max_in0_block_w",
    [
        pytest.param("sliding_o", 4096, 4, id="sliding_o_k4096"),
        pytest.param("full_o", 8192, 8, id="full_o_k8192"),
        pytest.param("dense_down", PADDED_MLP_INTERMEDIATE_SIZE, 2, id="dense_down_k2176"),
        pytest.param("expert_down", PADDED_MOE_INTERMEDIATE_SIZE, 2, id="expert_down_k768"),
    ],
)
def test_tp4_fused_matmul_reduce_scatter_exact_shape_repro(mesh_device, role, global_k, max_in0_block_w):
    """Shape-faithful repro for the fractured-residual producer boundary.

    This is opt-in because it installs a sub-device manager and exercises an
    experimental async CCL.  It deliberately adapts every row-parallel Gemma
    contraction to the fused op's required 2D-multicast program contract.  A
    passing result is a 704-wide shard per rank; the four shards compose the
    2816-wide residual without restoring the replicated all-reduce contract.
    """
    if os.getenv("GEMMA4_MULTICHIP_FUSED_RS_REPRO") != "1":
        pytest.skip("set GEMMA4_MULTICHIP_FUSED_RS_REPRO=1 for the serialized fused-RS hardware repro")
    from tests.ttnn.unit_tests.operations.ccl.test_new_matmul_reduce_scatter import run_reduce_scatter_impl

    if tuple(mesh_device.shape) != (1, TP_SIZE):
        pytest.skip(f"requires target 1x{TP_SIZE} mesh, got {tuple(mesh_device.shape)}")
    assert HIDDEN_SIZE % TP_SIZE == 0
    assert global_k % (TP_SIZE * 32) == 0
    run_reduce_scatter_impl(
        mesh_device=mesh_device,
        num_devices=TP_SIZE,
        # The reduce-scatter consumes the row-parallel matmul's *output*, not
        # its K-fractured input.  Its global width is therefore hidden_size.
        rs_input_shape=[1, 1, 32, HIDDEN_SIZE],
        mm_shard_dim=2,
        rs_scatter_dim=3,
        num_links=2,
        mm_weights_shape=[1, 1, global_k, HIDDEN_SIZE],
        rs_input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        matmul_weights_dtype=ttnn.bfloat8_b,
        max_in0_block_w=max_in0_block_w,
        use_bias=False,
        mem_config_input=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        rs_topology=ttnn.Topology.Ring,
        use_non_fused=False,
        num_iters=1,
        enable_trace=False,
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "role,global_k,in0_block_w",
    [
        pytest.param("sliding_o", 4096, 2, id="sliding_o_k4096"),
        pytest.param("full_o", 8192, 4, id="full_o_k8192"),
        pytest.param("dense_down", PADDED_MLP_INTERMEDIATE_SIZE, 1, id="dense_down_k2176"),
        pytest.param("expert_down", PADDED_MOE_INTERMEDIATE_SIZE, 1, id="expert_down_k768"),
    ],
)
def test_tp4_unfused_row_reduce_scatter_distributed_consumer_exact_shape_repro(
    mesh_device, role, global_k, in0_block_w
):
    """Compose every unfused row producer with a fractured dense consumer.

    The test checks the reduce-scatter result before the residual add, then
    keeps the 704-wide shard through distributed RMSNorm and gathers only
    inside the next column-parallel packed dense matmul.
    """
    if os.getenv("GEMMA4_MULTICHIP_UNFUSED_FRACTURED_REPRO") != "1":
        pytest.skip("set GEMMA4_MULTICHIP_UNFUSED_FRACTURED_REPRO=1 for the serialized hardware repro")
    import torch

    from ttnn import ConcatMeshToTensor, ShardTensorToMesh

    if tuple(mesh_device.shape) != (1, TP_SIZE):
        pytest.skip(f"requires target 1x{TP_SIZE} mesh, got {tuple(mesh_device.shape)}")
    local_hidden = HIDDEN_SIZE // TP_SIZE
    packed_dense_width = 2 * PADDED_MLP_INTERMEDIATE_SIZE
    local_packed_dense_width = packed_dense_width // TP_SIZE
    local_k_tiles = global_k // TP_SIZE // 32
    assert HIDDEN_SIZE % TP_SIZE == 0
    assert global_k % (TP_SIZE * 32) == 0
    assert local_k_tiles % in0_block_w == 0
    assert packed_dense_width % (TP_SIZE * 32) == 0

    torch.manual_seed(2600 + global_k)
    row_input = torch.randn(1, 1, 32, global_k, dtype=torch.bfloat16)
    row_weight = torch.randn(1, 1, global_k, HIDDEN_SIZE, dtype=torch.bfloat16)
    residual_skip = torch.randn(1, 1, 32, HIDDEN_SIZE, dtype=torch.bfloat16)
    gamma = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16)
    consumer_weight = torch.randn(1, 1, HIDDEN_SIZE, packed_dense_width, dtype=torch.bfloat16)
    expected_row = torch.matmul(row_input, row_weight)
    expected_residual = (expected_row + residual_skip).bfloat16()
    expected_normalized = expected_residual.float() * torch.rsqrt(
        expected_residual.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6
    )
    expected_normalized = (expected_normalized * gamma.float()).bfloat16()
    expected_consumer = torch.matmul(expected_normalized, consumer_weight)

    row_input_tt = ttnn.from_torch(
        row_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
    )
    row_weight_tt = ttnn.from_torch(
        row_weight,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=2),
    )
    residual_skip_tt = ttnn.from_torch(
        residual_skip,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
    )
    gamma_tt = ttnn.from_torch(
        gamma.reshape(TP_SIZE, 1, local_hidden // 32, 32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),
    )
    consumer_weight_tt = ttnn.from_torch(
        consumer_weight,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
    )

    compute_grid = mesh_device.compute_with_storage_grid_size()
    worker_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([worker_cores])], 0)
    mesh_device.load_sub_device_manager(manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])
    rs_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(3)]
    norm_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(2)]
    agmm_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(3)]
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    row_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(11, 6),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=8,
        out_block_w=8,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    local_consumer_n_tiles = local_packed_dense_width // 32
    consumer_grid_x = min(11, local_consumer_n_tiles)
    consumer_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(consumer_grid_x, 4),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=(local_consumer_n_tiles + consumer_grid_x - 1) // consumer_grid_x,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    try:
        row_partial_tt = ttnn.linear(
            row_input_tt,
            row_weight_tt,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=row_program_config,
            compute_kernel_config=compute_kernel_config,
        )
        assert all(tuple(shard.shape) == (1, 1, 32, HIDDEN_SIZE) for shard in ttnn.get_device_tensors(row_partial_tt))
        fractured_tt = ttnn.experimental.reduce_scatter_minimal_async(
            row_partial_tt,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=rs_semaphores,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            subdevice_id=worker_sub_device_id,
            cluster_axis=1,
            compute_kernel_config=compute_kernel_config,
        )
        assert all(tuple(shard.shape) == (1, 1, 32, local_hidden) for shard in ttnn.get_device_tensors(fractured_tt))
        fractured_residual_tt = ttnn.add(
            fractured_tt,
            residual_skip_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        local_stats = ttnn.rms_norm_pre_all_gather(fractured_residual_tt, dtype=ttnn.bfloat16)
        global_stats = ttnn.experimental.all_gather_async(
            local_stats,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=norm_semaphores,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=worker_sub_device_id,
            cluster_axis=1,
        )
        normalized_tt = ttnn.rms_norm_post_all_gather(
            fractured_residual_tt,
            global_stats,
            epsilon=1e-6,
            weight=gamma_tt,
        )
        _, consumer_tt = ttnn.experimental.all_gather_matmul_async(
            normalized_tt,
            consumer_weight_tt,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=agmm_semaphores,
            all_gather_core_grid_offset=(0, 6),
            num_links=2,
            topology=ttnn.Topology.Ring,
            subdevice_id=worker_sub_device_id,
            memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            program_config=consumer_program_config,
            compute_kernel_config=compute_kernel_config,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
        actual_row = ttnn.to_torch(
            ttnn.from_device(fractured_tt), mesh_composer=ConcatMeshToTensor(mesh_device, dim=3)
        )[..., :HIDDEN_SIZE]
        actual_consumer = ttnn.to_torch(
            ttnn.from_device(consumer_tt), mesh_composer=ConcatMeshToTensor(mesh_device, dim=3)
        )[..., :packed_dense_width]
        assert_with_pcc(expected_row, actual_row, 0.99)
        assert_with_pcc(expected_consumer, actual_consumer, 0.99)
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_tp4_fractured_indexed_sparse_expert_contract(mesh_device):
    """Exercise the exact TP4 K-fractured dynamic top-8 expert shapes.

    The selected decoder keeps gate/up N-sharded, which requires a replicated
    H=2816 input.  This opt-in experiment instead K-shards the real gate/up
    weights, uses indexed sparse output, reduce-scatters the packed I=768
    dimension, and consumes the resulting local I=192 with the existing
    K-sharded down weights.  It is intentionally test-only until the complete
    traced chain has both correctness and latency evidence.
    """
    if os.getenv("GEMMA4_MULTICHIP_FRACTURED_SPARSE_REPRO") != "1":
        pytest.skip("set GEMMA4_MULTICHIP_FRACTURED_SPARSE_REPRO=1 for the serialized hardware repro")

    import hashlib

    import torch
    import torch.nn.functional as F

    from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
        MOE_INTERMEDIATE_SIZE,
        NUM_EXPERTS,
        TOP_K_EXPERTS,
    )
    from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh, ShardTensorToMesh

    if tuple(mesh_device.shape) != (1, TP_SIZE):
        pytest.skip(f"requires target 1x{TP_SIZE} mesh, got {tuple(mesh_device.shape)}")
    assert ttnn.CONFIG.throw_exception_on_fallback, "run this contract repro with fallback-throw enabled"

    cfg = functional_tests._load_text_config()
    state = functional_tests._load_layer_state(0)
    prefix = "model.language_model.layers.0."
    gate_up = state[prefix + "experts.gate_up_proj"]
    gate = _pad_last(
        gate_up[:, :MOE_INTERMEDIATE_SIZE, :].transpose(-2, -1).contiguous(),
        PADDED_MOE_INTERMEDIATE_SIZE,
    )
    up = _pad_last(
        gate_up[:, MOE_INTERMEDIATE_SIZE:, :].transpose(-2, -1).contiguous(),
        PADDED_MOE_INTERMEDIATE_SIZE,
    )
    down = _pad_penultimate(
        state[prefix + "experts.down_proj"].transpose(-2, -1).contiguous(),
        PADDED_MOE_INTERMEDIATE_SIZE,
    )
    # Arrange packed N as [up_rank, gate_rank] for every rank.  One RS over N
    # then gives each rank both local operands needed by GeGLU.
    packed_gate_up = _packed_expert_gate_up_mesh_source(gate, up, TP_SIZE)
    down = down.unsqueeze(0)
    q = state[prefix + "self_attn.q_proj.weight"].transpose(-2, -1).contiguous()
    k = state[prefix + "self_attn.k_proj.weight"].transpose(-2, -1).contiguous()
    v = state[prefix + "self_attn.v_proj.weight"].transpose(-2, -1).contiguous()
    q_shards = q.chunk(TP_SIZE, dim=-1)
    k_shards = k.chunk(TP_SIZE, dim=-1)
    v_shards = v.chunk(TP_SIZE, dim=-1)
    qkv = (
        torch.cat(
            [torch.cat((q_shards[rank], k_shards[rank], v_shards[rank]), dim=-1) for rank in range(TP_SIZE)],
            dim=-1,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    assert qkv.shape[-1] == SLIDING_KIND.qkv_width
    norm_gamma = state[prefix + "input_layernorm.weight"].to(torch.bfloat16)

    torch.manual_seed(2718)
    hidden = torch.zeros((1, 1, 32, HIDDEN_SIZE), dtype=torch.bfloat16)
    hidden[..., 0, :] = torch.randn((HIDDEN_SIZE,), dtype=torch.bfloat16)
    router_scale = state[prefix + "router.scale"].float()
    router_proj = state[prefix + "router.proj.weight"].float()
    router_per_expert_scale = state[prefix + "router.per_expert_scale"].float()
    router_in = hidden[..., 0, :].float()
    router_in = router_in * torch.rsqrt(router_in.pow(2).mean(dim=-1, keepdim=True) + cfg.rms_norm_eps)
    router_in = router_in * router_scale * (HIDDEN_SIZE**-0.5)
    router_logits = torch.matmul(router_in, router_proj.transpose(-2, -1))
    top_values, top_indices = torch.topk(router_logits, TOP_K_EXPERTS, dim=-1, sorted=True)
    top_scores = torch.softmax(top_values, dim=-1) * router_per_expert_scale[top_indices]
    active_ids = top_indices.reshape(-1).tolist()
    assert len(set(active_ids)) == TOP_K_EXPERTS
    assert active_ids != sorted(active_ids), "the indexed repro must exercise a non-monotonic dynamic route"

    routing = torch.zeros((1, 1, 1, NUM_EXPERTS), dtype=torch.bfloat16)
    routing[..., active_ids] = top_scores.to(torch.bfloat16)
    indices = top_indices.to(torch.int32).reshape(1, 1, 1, TOP_K_EXPERTS)
    compact_scores = torch.zeros((1, TOP_K_EXPERTS, 32, 1), dtype=torch.bfloat16)
    compact_scores[:, :, 0, 0] = top_scores.to(torch.bfloat16)

    torch.manual_seed(2719)
    hidden_b = torch.zeros_like(hidden)
    hidden_b[..., 0, :] = torch.randn((HIDDEN_SIZE,), dtype=torch.bfloat16)
    router_in_b = hidden_b[..., 0, :].float()
    router_in_b = router_in_b * torch.rsqrt(router_in_b.pow(2).mean(dim=-1, keepdim=True) + cfg.rms_norm_eps)
    router_in_b = router_in_b * router_scale * (HIDDEN_SIZE**-0.5)
    router_logits_b = torch.matmul(router_in_b, router_proj.transpose(-2, -1))
    top_values_b, top_indices_b = torch.topk(router_logits_b, TOP_K_EXPERTS, dim=-1, sorted=True)
    top_scores_b = torch.softmax(top_values_b, dim=-1) * router_per_expert_scale[top_indices_b]
    active_ids_b = top_indices_b.reshape(-1).tolist()
    assert active_ids_b != active_ids
    routing_b = torch.zeros_like(routing)
    routing_b[..., active_ids_b] = top_scores_b.to(torch.bfloat16)
    indices_b = top_indices_b.to(torch.int32).reshape(1, 1, 1, TOP_K_EXPERTS)
    compact_scores_b = torch.zeros_like(compact_scores)
    compact_scores_b[:, :, 0, 0] = top_scores_b.to(torch.bfloat16)

    hidden_tt = ttnn.from_torch(
        hidden,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
    )
    packed_weight_tt = ttnn.from_torch(
        packed_gate_up,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=2),
    )
    down_weight_tt = ttnn.from_torch(
        down,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=2),
    )
    routing_tt = ttnn.from_torch(
        routing,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    indices_tt = ttnn.from_torch(
        indices,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    compact_scores_tt = ttnn.from_torch(
        compact_scores,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    qkv_k_weight_tt = ttnn.from_torch(
        qkv,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=2),
    )
    qkv_n_weight_tt = ttnn.from_torch(
        qkv,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
    )
    norm_gamma_tt = ttnn.from_torch(
        norm_gamma.reshape(TP_SIZE, 1, HIDDEN_SIZE // TP_SIZE // 32, 32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),
    )

    compute_grid = mesh_device.compute_with_storage_grid_size()
    worker_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([worker_cores])], 0)
    mesh_device.load_sub_device_manager(manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])
    packed_rs_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(3)]
    down_rs_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(3)]
    norm_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(2)]
    qkv_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(3)]
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    packed_config = _optimized_sparse_decode_config(
        mesh_device,
        n=2 * PADDED_MOE_INTERMEDIATE_SIZE,
        per_core_n=2,
        in0_block_w=11,
        out_subblock_w=2,
    )
    down_config = _optimized_sparse_decode_config(
        mesh_device,
        n=HIDDEN_SIZE,
        per_core_n=2,
        in0_block_w=6,
        out_subblock_w=2,
    )
    local_qkv_tiles = SLIDING_KIND.qkv_width // TP_SIZE // 32
    qkv_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(11, 4),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=(local_qkv_tiles + 10) // 11,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    qkv_k_shard_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(11, 6),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=24,
        out_block_w=8,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    try:
        packed_partial_tt = ttnn.sparse_matmul(
            hidden_tt,
            packed_weight_tt,
            sparsity=routing_tt,
            indices=indices_tt,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            program_config=packed_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
        )
        packed_partial_shapes = [list(shard.shape) for shard in ttnn.get_device_tensors(packed_partial_tt)]
        assert packed_partial_shapes == [[1, 1, 1, TOP_K_EXPERTS, 32, 2 * PADDED_MOE_INTERMEDIATE_SIZE]] * TP_SIZE
        # Sparse matmul preserves both operands' broadcast axes.  Production
        # already canonicalizes this representation before consuming it; the
        # reshape is metadata-only and does not gather the mesh-fractured K.
        packed_partial_tt = ttnn.reshape(
            packed_partial_tt,
            (1, TOP_K_EXPERTS, 32, 2 * PADDED_MOE_INTERMEDIATE_SIZE),
        )
        packed_fractured_tt = ttnn.experimental.reduce_scatter_minimal_async(
            packed_partial_tt,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=packed_rs_semaphores,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            subdevice_id=worker_sub_device_id,
            cluster_axis=1,
            compute_kernel_config=compute_kernel_config,
        )
        local_packed = 2 * PADDED_MOE_INTERMEDIATE_SIZE // TP_SIZE
        local_intermediate = PADDED_MOE_INTERMEDIATE_SIZE // TP_SIZE
        packed_fractured_shapes = [list(shard.shape) for shard in ttnn.get_device_tensors(packed_fractured_tt)]
        assert packed_fractured_shapes == [[1, TOP_K_EXPERTS, 32, local_packed]] * TP_SIZE
        up_local_tt = ttnn.slice(
            packed_fractured_tt,
            [0, 0, 0, 0],
            [1, TOP_K_EXPERTS, 32, local_intermediate],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate_local_tt = ttnn.slice(
            packed_fractured_tt,
            [0, 0, 0, local_intermediate],
            [1, TOP_K_EXPERTS, 32, local_packed],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        down_input_tt = ttnn.mul(
            gate_local_tt,
            up_local_tt,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        down_partial_tt = ttnn.sparse_matmul(
            down_input_tt,
            down_weight_tt,
            sparsity=routing_tt,
            indices=indices_tt,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            program_config=down_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
        )
        down_partial_shapes = [list(shard.shape) for shard in ttnn.get_device_tensors(down_partial_tt)]
        assert down_partial_shapes == [[1, TOP_K_EXPERTS, 32, HIDDEN_SIZE]] * TP_SIZE
        weighted_tt = ttnn.mul(down_partial_tt, compact_scores_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        down_summed_tt = ttnn.sum(weighted_tt, dim=1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        fractured_output_tt = ttnn.experimental.reduce_scatter_minimal_async(
            down_summed_tt,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=down_rs_semaphores,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            subdevice_id=worker_sub_device_id,
            cluster_axis=1,
            compute_kernel_config=compute_kernel_config,
        )
        fractured_residual_tt = ttnn.add(
            fractured_output_tt,
            hidden_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        local_stats_tt = ttnn.rms_norm_pre_all_gather(fractured_residual_tt, dtype=ttnn.bfloat16)
        global_stats_tt = ttnn.experimental.all_gather_async(
            local_stats_tt,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=norm_semaphores,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=worker_sub_device_id,
            cluster_axis=1,
        )
        normalized_tt = ttnn.rms_norm_post_all_gather(
            fractured_residual_tt,
            global_stats_tt,
            epsilon=cfg.rms_norm_eps,
            weight=norm_gamma_tt,
        )
        qkv_partial_tt = ttnn.linear(
            normalized_tt,
            qkv_k_weight_tt,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=qkv_k_shard_config,
            compute_kernel_config=compute_kernel_config,
        )
        qkv_output_tt = ttnn.experimental.reduce_scatter_minimal_async(
            qkv_partial_tt,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=qkv_semaphores,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            subdevice_id=worker_sub_device_id,
            cluster_axis=1,
            compute_kernel_config=compute_kernel_config,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])

        packed_actual_shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(packed_fractured_tt.cpu())]
        down_input_actual_shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(down_input_tt.cpu())]
        packed_actual = ttnn.to_torch(
            ttnn.from_device(packed_fractured_tt), mesh_composer=ConcatMeshToTensor(mesh_device, dim=3)
        )
        output_actual = ttnn.to_torch(
            ttnn.from_device(fractured_output_tt), mesh_composer=ConcatMeshToTensor(mesh_device, dim=3)
        )[..., :HIDDEN_SIZE]
        qkv_actual = ttnn.to_torch(
            ttnn.from_device(qkv_output_tt), mesh_composer=ConcatMeshToTensor(mesh_device, dim=3)
        )[..., : SLIDING_KIND.qkv_width]

        selected_gate = gate[active_ids].float()
        selected_up = up[active_ids].float()
        hidden_matrix = hidden[0, 0].float()
        gate_ref = torch.einsum("mk,eki->emi", hidden_matrix, selected_gate)
        up_ref = torch.einsum("mk,eki->emi", hidden_matrix, selected_up)
        packed_ref = torch.cat(
            [
                torch.cat(
                    (
                        up_ref[..., rank * local_intermediate : (rank + 1) * local_intermediate],
                        gate_ref[..., rank * local_intermediate : (rank + 1) * local_intermediate],
                    ),
                    dim=-1,
                )
                for rank in range(TP_SIZE)
            ],
            dim=-1,
        ).unsqueeze(0)
        down_input_ref = F.gelu(gate_ref, approximate="none") * up_ref
        selected_down = down[0, active_ids].float()
        output_ref = torch.einsum("emi,eih->emh", down_input_ref, selected_down)
        output_ref = (output_ref * compact_scores[0].float()).sum(dim=0, keepdim=True).unsqueeze(0)
        residual_ref = output_ref + hidden.float()
        normalized_ref = residual_ref * torch.rsqrt(residual_ref.pow(2).mean(dim=-1, keepdim=True) + cfg.rms_norm_eps)
        normalized_ref = normalized_ref * norm_gamma.float()
        qkv_ref = torch.matmul(normalized_ref, qkv.float())
        selected_gate_b = gate[active_ids_b].float()
        selected_up_b = up[active_ids_b].float()
        hidden_matrix_b = hidden_b[0, 0].float()
        gate_ref_b = torch.einsum("mk,eki->emi", hidden_matrix_b, selected_gate_b)
        up_ref_b = torch.einsum("mk,eki->emi", hidden_matrix_b, selected_up_b)
        down_input_ref_b = F.gelu(gate_ref_b, approximate="none") * up_ref_b
        selected_down_b = down[0, active_ids_b].float()
        output_ref_b = torch.einsum("emi,eih->emh", down_input_ref_b, selected_down_b)
        output_ref_b = (output_ref_b * compact_scores_b[0].float()).sum(dim=0, keepdim=True).unsqueeze(0)
        residual_ref_b = output_ref_b + hidden_b.float()
        normalized_ref_b = residual_ref_b * torch.rsqrt(
            residual_ref_b.pow(2).mean(dim=-1, keepdim=True) + cfg.rms_norm_eps
        )
        normalized_ref_b = normalized_ref_b * norm_gamma.float()
        qkv_ref_b = torch.matmul(normalized_ref_b, qkv.float())

        packed_rank_pccs = []
        down_input_rank_pccs = []
        for rank in range(TP_SIZE):
            packed_rank_ref = packed_ref[..., rank * local_packed : (rank + 1) * local_packed]
            _, packed_rank_pcc = functional_tests.comp_pcc(
                packed_rank_ref[:, :, :1, :], packed_actual_shards[rank][:, :, :1, :], 0.0
            )
            packed_rank_pccs.append(float(packed_rank_pcc))
            down_input_rank_ref = down_input_ref[
                ..., rank * local_intermediate : (rank + 1) * local_intermediate
            ].unsqueeze(0)
            _, down_input_rank_pcc = functional_tests.comp_pcc(
                down_input_rank_ref[:, :, :1, :], down_input_actual_shards[rank][:, :, :1, :], 0.0
            )
            down_input_rank_pccs.append(float(down_input_rank_pcc))
        packed_ok, packed_pcc = functional_tests.comp_pcc(packed_ref[:, :, :1, :], packed_actual[:, :, :1, :], 0.99)
        output_ok, output_pcc = functional_tests.comp_pcc(output_ref[:, :, :1, :], output_actual[:, :, :1, :], 0.99)
        qkv_ok, qkv_pcc = functional_tests.comp_pcc(qkv_ref[:, :, :1, :], qkv_actual[:, :, :1, :], 0.99)
        artifact_dir = Path(
            "models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/fractured_sparse"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        test_path = Path(__file__)
        artifact = {
            "gate": "A_contract_and_correctness",
            "status": "pass" if packed_ok and output_ok and qkv_ok else "fail",
            "hardware": "four P300C devices as P150x4 proxy",
            "mesh_shape": [1, TP_SIZE],
            "layer_idx": 0,
            "real_weights": True,
            "weight_dtype": "BFLOAT8_B",
            "activation_dtype": "BFLOAT16",
            "active_ids": active_ids,
            "packed_partial_local_shapes": packed_partial_shapes,
            "packed_reduce_scatter_local_shapes": packed_fractured_shapes,
            "down_partial_local_shapes": down_partial_shapes,
            "final_local_shapes": [list(shard.shape) for shard in ttnn.get_device_tensors(fractured_output_tt)],
            "packed_reduce_scatter_pcc": float(packed_pcc),
            "packed_reduce_scatter_per_rank_pcc": packed_rank_pccs,
            "local_geglu_per_rank_pcc": down_input_rank_pccs,
            "fractured_expert_output_pcc": float(output_pcc),
            "next_sliding_qkv_pcc": float(qkv_pcc),
            "next_sliding_qkv_local_shapes": [list(shard.shape) for shard in ttnn.get_device_tensors(qkv_output_tt)],
            "pcc_threshold": 0.99,
            "fallback_policy": "TTNN throw_exception_on_fallback=true",
            "persistent_buffers": False,
            "trace": False,
            "test_sha256_at_run": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        }
        artifact_suffix = _measurement_artifact_suffix()
        (artifact_dir / f"gate_a_contract{artifact_suffix}.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n"
        )
        torch.save(
            {
                "packed_actual_first_token": packed_actual[:, :, 0, :],
                "packed_reference_first_token": packed_ref[:, :, 0, :],
            },
            artifact_dir / "gate_a_packed_diagnostic.pt",
        )
        assert packed_ok, f"logical-row packed indexed sparse + RS PCC={packed_pcc}"
        assert output_ok, f"fractured routed expert output PCC={output_pcc}"
        assert qkv_ok, f"fractured distributed norm + sliding QKV boundary PCC={qkv_pcc}"

        if os.getenv("GEMMA4_MULTICHIP_FRACTURED_SPARSE_TRACE") != "1":
            return

        # Gate B: bind every CCL output/staging tensor, then capture the exact
        # routed expert + next-QKV boundary.  The input/indices/score buffers
        # remain caller-owned so a replay can select a different expert list.
        (
            packed_intermediate_tt,
            packed_penult_tt,
        ) = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
            packed_partial_tt,
            dim=3,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,
            compute_kernel_config=compute_kernel_config,
        )
        (
            down_intermediate_tt,
            down_penult_tt,
        ) = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
            down_summed_tt,
            dim=3,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,
            compute_kernel_config=compute_kernel_config,
        )
        packed_persistent = [packed_intermediate_tt, packed_fractured_tt, packed_penult_tt]
        down_persistent = [down_intermediate_tt, fractured_output_tt, down_penult_tt]
        qkv_intermediate_tt, qkv_penult_tt = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
            qkv_partial_tt,
            dim=3,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,
            compute_kernel_config=compute_kernel_config,
        )
        qkv_persistent = [qkv_intermediate_tt, qkv_output_tt, qkv_penult_tt]
        qkv_trace_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(3)]

        def run_fractured_candidate():
            packed_partial = ttnn.sparse_matmul(
                hidden_tt,
                packed_weight_tt,
                sparsity=routing_tt,
                indices=indices_tt,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=ttnn.Tile([32, 32]),
                program_config=packed_config,
                dtype=ttnn.bfloat16,
                compute_kernel_config=compute_kernel_config,
            )
            packed_partial = ttnn.reshape(
                packed_partial,
                (1, TOP_K_EXPERTS, 32, 2 * PADDED_MOE_INTERMEDIATE_SIZE),
            )
            packed_local = ttnn.experimental.reduce_scatter_minimal_async(
                packed_partial,
                persistent_output_buffers=packed_persistent,
                dim=3,
                multi_device_global_semaphore=packed_rs_semaphores,
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                subdevice_id=worker_sub_device_id,
                cluster_axis=1,
                compute_kernel_config=compute_kernel_config,
            )
            up_local = ttnn.slice(
                packed_local,
                [0, 0, 0, 0],
                [1, TOP_K_EXPERTS, 32, local_intermediate],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate_local = ttnn.slice(
                packed_local,
                [0, 0, 0, local_intermediate],
                [1, TOP_K_EXPERTS, 32, local_packed],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            down_input = ttnn.mul(
                gate_local,
                up_local,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            down_partial = ttnn.sparse_matmul(
                down_input,
                down_weight_tt,
                sparsity=routing_tt,
                indices=indices_tt,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=ttnn.Tile([32, 32]),
                program_config=down_config,
                dtype=ttnn.bfloat16,
                compute_kernel_config=compute_kernel_config,
            )
            weighted = ttnn.mul(down_partial, compact_scores_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            down_summed = ttnn.sum(weighted, dim=1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            fractured_output = ttnn.experimental.reduce_scatter_minimal_async(
                down_summed,
                persistent_output_buffers=down_persistent,
                dim=3,
                multi_device_global_semaphore=down_rs_semaphores,
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                subdevice_id=worker_sub_device_id,
                cluster_axis=1,
                compute_kernel_config=compute_kernel_config,
            )
            fractured_residual = ttnn.add(fractured_output, hidden_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            local_stats = ttnn.rms_norm_pre_all_gather(fractured_residual, dtype=ttnn.bfloat16)
            global_stats = ttnn.experimental.all_gather_async(
                local_stats,
                persistent_output_buffer=global_stats_tt,
                dim=3,
                multi_device_global_semaphore=norm_semaphores,
                num_links=2,
                topology=ttnn.Topology.Ring,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                subdevice_id=worker_sub_device_id,
                cluster_axis=1,
            )
            normalized = ttnn.rms_norm_post_all_gather(
                fractured_residual,
                global_stats,
                epsilon=cfg.rms_norm_eps,
                weight=norm_gamma_tt,
            )
            qkv_partial = ttnn.linear(
                normalized,
                qkv_k_weight_tt,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=qkv_k_shard_config,
                compute_kernel_config=compute_kernel_config,
            )
            qkv_output = ttnn.experimental.reduce_scatter_minimal_async(
                qkv_partial,
                persistent_output_buffers=qkv_persistent,
                dim=3,
                multi_device_global_semaphore=qkv_trace_semaphores,
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                subdevice_id=worker_sub_device_id,
                cluster_axis=1,
                compute_kernel_config=compute_kernel_config,
            )
            return qkv_output

        # Compile the persistent candidate before capture.
        run_fractured_candidate()
        ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
        candidate_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        candidate_trace_output_tt = run_fractured_candidate()
        ttnn.end_trace_capture(mesh_device, candidate_trace_id, cq_id=0)
        trace_warmups = int(os.getenv("GEMMA4_FRACTURED_TRACE_WARMUPS", "5"))
        candidate_iterations = int(os.getenv("GEMMA4_FRACTURED_TRACE_ITERATIONS", "30"))
        dynamic_route_iterations = int(os.getenv("GEMMA4_FRACTURED_ROUTE_STRESS_ITERATIONS", "100"))
        try:
            for _ in range(trace_warmups):
                ttnn.execute_trace(mesh_device, candidate_trace_id, cq_id=0, blocking=True)
            candidate_first = [
                ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(candidate_trace_output_tt.cpu())
            ]
            started = time.perf_counter()
            for _ in range(candidate_iterations):
                ttnn.execute_trace(mesh_device, candidate_trace_id, cq_id=0, blocking=True)
            candidate_trace_ms = (time.perf_counter() - started) * 1000 / candidate_iterations
            candidate_repeat = [
                ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(candidate_trace_output_tt.cpu())
            ]

            def refresh_mesh_tensor(source, target, *, shard_dim=None):
                mapper = (
                    ShardTensorToMesh(mesh_device, dim=shard_dim)
                    if shard_dim is not None
                    else ReplicateTensorToMesh(mesh_device)
                )
                host_mesh_tensor = ttnn.from_torch(
                    source.contiguous(),
                    dtype=target.dtype,
                    layout=target.layout,
                    mesh_mapper=mapper,
                )
                ttnn.copy_host_to_device_tensor(host_mesh_tensor, target)

            cache_entries_before_dynamic_route = mesh_device.num_program_cache_entries()
            refresh_mesh_tensor(hidden_b, hidden_tt, shard_dim=3)
            refresh_mesh_tensor(routing_b, routing_tt)
            refresh_mesh_tensor(indices_b, indices_tt)
            refresh_mesh_tensor(compact_scores_b, compact_scores_tt)
            ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])

            # Prove that the second route reached the exact caller-owned
            # buffers captured by the trace before attributing any mismatch
            # to sparse-matmul replay.  Score only the public token row; the
            # remaining tile rows are padding in this decode repro.
            refreshed_hidden_shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(hidden_tt.cpu())]
            refreshed_hidden = torch.cat(refreshed_hidden_shards, dim=3)
            refreshed_routing = ttnn.to_torch(ttnn.get_device_tensors(routing_tt.cpu())[0])
            refreshed_indices = ttnn.to_torch(ttnn.get_device_tensors(indices_tt.cpu())[0])
            refreshed_scores = ttnn.to_torch(ttnn.get_device_tensors(compact_scores_tt.cpu())[0])
            hidden_refresh_ok, hidden_refresh_pcc = functional_tests.comp_pcc(
                hidden_b[:, :, :1, :], refreshed_hidden[:, :, :1, :], 0.999
            )
            routing_refresh_exact = torch.equal(routing_b, refreshed_routing)
            indices_refresh_exact = torch.equal(indices_b.to(refreshed_indices.dtype), refreshed_indices)
            scores_refresh_exact = torch.equal(compact_scores_b, refreshed_scores)

            ttnn.execute_trace(mesh_device, candidate_trace_id, cq_id=0, blocking=True)
            dynamic_route_first = [
                ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(candidate_trace_output_tt.cpu())
            ]
            for _ in range(dynamic_route_iterations):
                ttnn.execute_trace(mesh_device, candidate_trace_id, cq_id=0, blocking=True)
            dynamic_route_repeat = [
                ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(candidate_trace_output_tt.cpu())
            ]
            cache_entries_after_dynamic_route = mesh_device.num_program_cache_entries()
        finally:
            ttnn.release_trace(mesh_device, candidate_trace_id)
        candidate_bit_exact = all(torch.equal(a, b) for a, b in zip(candidate_first, candidate_repeat))
        dynamic_route_bit_exact = all(torch.equal(a, b) for a, b in zip(dynamic_route_first, dynamic_route_repeat))
        dynamic_route_changed_output = any(not torch.equal(a, b) for a, b in zip(candidate_repeat, dynamic_route_first))
        dynamic_route_concat = torch.cat(
            [shard[..., : SLIDING_KIND.qkv_width // TP_SIZE] for shard in dynamic_route_first], -1
        )
        dynamic_route_ok, dynamic_route_pcc = functional_tests.comp_pcc(
            qkv_ref_b[:, :, :1, :], dynamic_route_concat[:, :, :1, :], 0.99
        )

        # Incumbent: replicated H, N-sharded packed gate/up, the same K-sharded
        # down weights, expanded sparse output, and persistent all-reduce.
        from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import _width_sharded_memory_config

        incumbent_packed_weight_tt = ttnn.from_torch(
            packed_gate_up,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
        )
        incumbent_hidden_tt = ttnn.from_torch(
            hidden,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        incumbent_routing_tt = ttnn.from_torch(
            routing,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        dense_scores = torch.zeros((1, NUM_EXPERTS, 32, 1), dtype=torch.bfloat16)
        dense_scores[:, active_ids, 0, 0] = top_scores.to(torch.bfloat16)
        incumbent_scores_tt = ttnn.from_torch(
            dense_scores,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        incumbent_gamma_tt = ttnn.from_torch(
            norm_gamma.reshape(1, 1, 1, HIDDEN_SIZE),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        incumbent_packed_config = _optimized_sparse_decode_config(
            mesh_device,
            n=local_packed,
            per_core_n=2,
            in0_block_w=44,
            out_subblock_w=2,
        )
        ar_memory_config = _width_sharded_memory_config(HIDDEN_SIZE, ttnn.CoreGrid(x=11, y=8))
        ar_buffer_memory_config = _width_sharded_memory_config(HIDDEN_SIZE * TP_SIZE, ttnn.CoreGrid(x=11, y=8))
        ar_buffer_tt = ttnn.from_torch(
            torch.zeros((1, 1, 32, HIDDEN_SIZE * TP_SIZE), dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ar_buffer_memory_config,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        ar_semaphore = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)

        def run_incumbent():
            packed_expanded = ttnn.sparse_matmul(
                incumbent_hidden_tt,
                incumbent_packed_weight_tt,
                sparsity=incumbent_routing_tt,
                nnz=TOP_K_EXPERTS,
                is_input_b_sparse=True,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=ttnn.Tile([32, 32]),
                program_config=incumbent_packed_config,
                dtype=ttnn.bfloat16,
                compute_kernel_config=compute_kernel_config,
            )
            packed_expanded = ttnn.reshape(packed_expanded, (1, NUM_EXPERTS, 32, local_packed))
            up_expanded = ttnn.slice(
                packed_expanded,
                [0, 0, 0, 0],
                [1, NUM_EXPERTS, 32, local_intermediate],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            gate_expanded = ttnn.slice(
                packed_expanded,
                [0, 0, 0, local_intermediate],
                [1, NUM_EXPERTS, 32, local_packed],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            down_input = ttnn.mul(
                gate_expanded,
                up_expanded,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            down_expanded = ttnn.sparse_matmul(
                down_input,
                down_weight_tt,
                sparsity=incumbent_routing_tt,
                nnz=TOP_K_EXPERTS,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=ttnn.Tile([32, 32]),
                program_config=down_config,
                dtype=ttnn.bfloat16,
                compute_kernel_config=compute_kernel_config,
            )
            down_expanded = ttnn.reshape(down_expanded, (1, NUM_EXPERTS, 32, HIDDEN_SIZE))
            weighted = ttnn.mul(down_expanded, incumbent_scores_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
            partial = ttnn.sum(weighted, dim=1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            partial = ttnn.to_memory_config(partial, ar_memory_config, dtype=ttnn.bfloat16)
            replicated = ttnn.experimental.all_reduce_async(
                partial,
                ar_buffer_tt,
                cluster_axis=1,
                mesh_device=mesh_device,
                multi_device_global_semaphore=ar_semaphore,
                num_links=2,
                topology=ttnn.Topology.Ring,
                memory_config=ar_memory_config,
                subdevice_id=worker_sub_device_id,
            )
            replicated = ttnn.to_memory_config(replicated, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            residual = ttnn.add(replicated, incumbent_hidden_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            normalized = ttnn.rms_norm(
                residual,
                epsilon=cfg.rms_norm_eps,
                weight=incumbent_gamma_tt,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return ttnn.linear(
                normalized,
                qkv_n_weight_tt,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=qkv_config,
                compute_kernel_config=compute_kernel_config,
            )

        incumbent_eager_tt = run_incumbent()
        ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
        incumbent_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        incumbent_trace_output_tt = run_incumbent()
        ttnn.end_trace_capture(mesh_device, incumbent_trace_id, cq_id=0)
        try:
            for _ in range(trace_warmups):
                ttnn.execute_trace(mesh_device, incumbent_trace_id, cq_id=0, blocking=True)
            incumbent_first = [
                ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(incumbent_trace_output_tt.cpu())
            ]
            incumbent_iterations = candidate_iterations
            started = time.perf_counter()
            for _ in range(incumbent_iterations):
                ttnn.execute_trace(mesh_device, incumbent_trace_id, cq_id=0, blocking=True)
            incumbent_trace_ms = (time.perf_counter() - started) * 1000 / incumbent_iterations
            incumbent_repeat = [
                ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(incumbent_trace_output_tt.cpu())
            ]
        finally:
            ttnn.release_trace(mesh_device, incumbent_trace_id)
        incumbent_eager = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(incumbent_eager_tt.cpu())]
        incumbent_bit_exact = all(torch.equal(a, b) for a, b in zip(incumbent_first, incumbent_repeat))
        incumbent_eager_trace_bit_exact = all(torch.equal(a, b) for a, b in zip(incumbent_eager, incumbent_first))
        candidate_concat = torch.cat([shard[..., : SLIDING_KIND.qkv_width // TP_SIZE] for shard in candidate_first], -1)
        incumbent_concat = torch.cat([shard[..., : SLIDING_KIND.qkv_width // TP_SIZE] for shard in incumbent_first], -1)
        cross_ok, cross_pcc = functional_tests.comp_pcc(
            incumbent_concat[:, :, :1, :], candidate_concat[:, :, :1, :], 0.99
        )
        candidate_trace_torch_ok, candidate_trace_torch_pcc = functional_tests.comp_pcc(
            qkv_ref[:, :, :1, :], candidate_concat[:, :, :1, :], 0.99
        )
        incumbent_trace_torch_ok, incumbent_trace_torch_pcc = functional_tests.comp_pcc(
            qkv_ref[:, :, :1, :], incumbent_concat[:, :, :1, :], 0.99
        )
        speedup = incumbent_trace_ms / candidate_trace_ms
        gate_b = {
            "gate": "B_persistent_trace_and_boundary_performance",
            "status": (
                "accepted"
                if speedup > 1.0 and cross_ok and dynamic_route_ok and dynamic_route_bit_exact
                else "rejected"
            ),
            "candidate": "K-fractured indexed top-8 packed expert with I/H reduce-scatter",
            "incumbent": "replicated H expanded top-8 sparse expert with persistent all-reduce",
            "real_weights": True,
            "active_ids": active_ids,
            "dynamic_active_ids": active_ids_b,
            "persistent_candidate_buffers": {
                "packed_rs": [list(shard.shape) for shard in ttnn.get_device_tensors(packed_fractured_tt)],
                "down_rs": [list(shard.shape) for shard in ttnn.get_device_tensors(fractured_output_tt)],
                "norm_all_gather": [list(shard.shape) for shard in ttnn.get_device_tensors(global_stats_tt)],
                "qkv_reduce_scatter": [list(shard.shape) for shard in ttnn.get_device_tensors(qkv_output_tt)],
            },
            "warmups": trace_warmups,
            "iterations": candidate_iterations,
            "candidate_trace_ms": candidate_trace_ms,
            "incumbent_trace_ms": incumbent_trace_ms,
            "speedup": speedup,
            "candidate_repeat_bit_exact": candidate_bit_exact,
            "dynamic_route_stress_iterations": dynamic_route_iterations,
            "dynamic_route_repeat_bit_exact": dynamic_route_bit_exact,
            "dynamic_route_changed_output": dynamic_route_changed_output,
            "dynamic_route_vs_torch_pcc": float(dynamic_route_pcc),
            "refreshed_inputs": {
                "hidden_pcc": float(hidden_refresh_pcc),
                "routing_exact": routing_refresh_exact,
                "indices_exact": indices_refresh_exact,
                "compact_scores_exact": scores_refresh_exact,
            },
            "program_cache_entries_before_dynamic_route": cache_entries_before_dynamic_route,
            "program_cache_entries_after_dynamic_route": cache_entries_after_dynamic_route,
            "incumbent_repeat_bit_exact": incumbent_bit_exact,
            "incumbent_eager_trace_bit_exact": incumbent_eager_trace_bit_exact,
            "candidate_vs_incumbent_pcc": float(cross_pcc),
            "candidate_vs_torch_pcc": float(candidate_trace_torch_pcc),
            "incumbent_vs_torch_pcc": float(incumbent_trace_torch_pcc),
            "pcc_threshold": 0.99,
            "fallback_policy": "TTNN throw_exception_on_fallback=true",
        }
        (artifact_dir / f"gate_b_trace_perf{artifact_suffix}.json").write_text(
            json.dumps(gate_b, indent=2, sort_keys=True) + "\n"
        )
        torch.save(
            {
                "trace_first_token": dynamic_route_concat[:, :, :1, :],
                "torch_first_token": qkv_ref_b[:, :, :1, :],
            },
            artifact_dir / f"gate_b_dynamic_diagnostic{artifact_suffix}.pt",
        )
        assert hidden_refresh_ok, f"refreshed hidden PCC={hidden_refresh_pcc}"
        assert routing_refresh_exact
        assert indices_refresh_exact
        assert scores_refresh_exact
        assert candidate_bit_exact
        assert dynamic_route_bit_exact
        assert dynamic_route_changed_output
        assert dynamic_route_ok, f"changed routed indices vs Torch next-boundary PCC={dynamic_route_pcc}"
        assert cache_entries_after_dynamic_route == cache_entries_before_dynamic_route
        assert incumbent_bit_exact
        assert incumbent_eager_trace_bit_exact
        assert cross_ok, f"candidate vs incumbent next-boundary PCC={cross_pcc}"
        assert candidate_trace_torch_ok, f"candidate trace vs Torch next-boundary PCC={candidate_trace_torch_pcc}"
        assert incumbent_trace_torch_ok, f"incumbent trace vs Torch next-boundary PCC={incumbent_trace_torch_pcc}"
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_tp4_fractured_product_two_layer_boundary(mesh_device, device_params, monkeypatch):
    """Validate the opt-in layer-0 expert -> layer-5 QKV product API."""

    if os.getenv("GEMMA4_MULTICHIP_FRACTURED_PRODUCT_REPRO") != "1":
        pytest.skip("set GEMMA4_MULTICHIP_FRACTURED_PRODUCT_REPRO=1 for the serialized product repro")

    import hashlib

    import torch
    import torch.nn.functional as F

    from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
        MOE_INTERMEDIATE_SIZE,
        NUM_EXPERTS,
        TOP_K_EXPERTS,
    )
    from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh, ShardTensorToMesh

    assert tuple(mesh_device.shape) == (1, TP_SIZE)
    assert ttnn.CONFIG.throw_exception_on_fallback, "run this repro with fallback-throw enabled"
    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    monkeypatch.setenv("GEMMA4_MULTICHIP_FRACTURED_RESIDUAL_CANDIDATE", "1")
    cfg = functional_tests._load_text_config()
    source_state = functional_tests._load_layer_state(0)
    next_state = functional_tests._load_layer_state(5)
    source = MultichipDecoder.from_state_dict(
        source_state,
        hf_config=cfg,
        layer_idx=0,
        mesh_device=mesh_device,
    )
    next_decoder = MultichipDecoder.from_state_dict(
        next_state,
        hf_config=cfg,
        layer_idx=5,
        mesh_device=mesh_device,
    )
    assert source.layer_kind is SLIDING_KIND
    assert next_decoder.layer_kind is FULL_KIND
    assert source.fractured_residual_candidate_enabled
    assert next_decoder.fractured_residual_candidate_enabled
    assert tuple(ttnn.get_device_tensors(source.fractured_packed_expert_gate_up)[0].shape) == (
        1,
        NUM_EXPERTS,
        HIDDEN_SIZE // TP_SIZE,
        2 * PADDED_MOE_INTERMEDIATE_SIZE,
    )
    physical_full_qkv_width = FULL_KIND.q_width + 2 * FULL_KIND.kv_width * 2
    assert tuple(ttnn.get_device_tensors(next_decoder.fractured_qkv)[0].shape) == (
        1,
        1,
        HIDDEN_SIZE // TP_SIZE,
        physical_full_qkv_width,
    )

    torch.manual_seed(4904)
    hidden = torch.zeros((1, 1, 32, HIDDEN_SIZE), dtype=torch.bfloat16)
    hidden[..., 0, :] = torch.randn((HIDDEN_SIZE,), dtype=torch.bfloat16)
    hidden_replicated = ttnn.from_torch(
        hidden[..., :1, :],
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    hidden_fractured = ttnn.from_torch(
        hidden,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
    )
    routing = source._router_weights(hidden_replicated)
    sparsity, indices, compact_scores = source.fractured_route_metadata(routing)
    first_residual, first_qkv = source.fractured_expert_to_next_qkv(
        hidden_fractured,
        sparsity,
        indices,
        compact_scores,
        next_decoder=next_decoder,
    )
    resources = source.fractured_boundary_resources
    assert resources is not None and resources["initialized"]
    for name in ("packed", "down", "qkv"):
        assert len(resources[f"{name}_buffers"]) == 3
        assert resources[f"{name}_buffers"][1] is resources[f"{name}_output"]
    assert resources["norm_output"] is not None

    # Run once with all persistent addresses before trace capture.
    persistent_residual, persistent_qkv = source.fractured_expert_to_next_qkv(
        hidden_fractured,
        sparsity,
        indices,
        compact_scores,
        next_decoder=next_decoder,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_residual, trace_qkv = source.fractured_expert_to_next_qkv(
        hidden_fractured,
        sparsity,
        indices,
        compact_scores,
        next_decoder=next_decoder,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    warmups = int(os.getenv("GEMMA4_FRACTURED_PRODUCT_WARMUPS", "5"))
    iterations = int(os.getenv("GEMMA4_FRACTURED_PRODUCT_ITERATIONS", "20"))
    try:
        for _ in range(warmups):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        trace_first = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(trace_qkv.cpu())]
        cache_before = mesh_device.num_program_cache_entries()
        started = time.perf_counter()
        for _ in range(iterations):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        trace_ms = (time.perf_counter() - started) * 1000 / iterations
        trace_repeat = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(trace_qkv.cpu())]
        cache_after = mesh_device.num_program_cache_entries()
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    residual_actual = ttnn.to_torch(
        ttnn.from_device(trace_residual),
        mesh_composer=ConcatMeshToTensor(mesh_device, dim=3),
    )
    qkv_actual = torch.cat(trace_first, dim=3)[..., :physical_full_qkv_width]
    first_qkv_hosts = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(first_qkv.cpu())]
    persistent_qkv_hosts = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(persistent_qkv.cpu())]
    route_indices = ttnn.to_torch(ttnn.get_device_tensors(indices.cpu())[0]).to(torch.int64).reshape(-1).tolist()
    route_scores = ttnn.to_torch(ttnn.get_device_tensors(compact_scores.cpu())[0])[0, :, 0, 0].float()
    assert len(set(route_indices)) == TOP_K_EXPERTS
    assert route_indices != sorted(route_indices)

    source_defaults, _ = _multichip_optimized_defaults(TP_SIZE, SLIDING_KIND)
    folded_source = _prepare_folded_state_dict(
        source_state,
        layer_idx=0,
        folded_router_projection=source_defaults["folded_router_projection"],
        shared_ffn_norm=source_defaults["shared_ffn_norm"],
        folded_expert_scale=source_defaults["folded_expert_scale"],
    )
    source_prefix = "model.language_model.layers.0."
    gate_up = folded_source[source_prefix + "experts.gate_up_proj"]
    gate = _pad_last(
        gate_up[:, :MOE_INTERMEDIATE_SIZE, :].transpose(-2, -1).contiguous(),
        PADDED_MOE_INTERMEDIATE_SIZE,
    )
    up = _pad_last(
        gate_up[:, MOE_INTERMEDIATE_SIZE:, :].transpose(-2, -1).contiguous(),
        PADDED_MOE_INTERMEDIATE_SIZE,
    )
    down = _pad_penultimate(
        folded_source[source_prefix + "experts.down_proj"].transpose(-2, -1).contiguous(),
        PADDED_MOE_INTERMEDIATE_SIZE,
    )
    hidden_matrix = hidden[0, 0].float()
    gate_ref = torch.einsum("mk,eki->emi", hidden_matrix, gate[route_indices].float())
    up_ref = torch.einsum("mk,eki->emi", hidden_matrix, up[route_indices].float())
    activated_ref = F.gelu(gate_ref, approximate="none") * up_ref
    branch_ref = torch.einsum("emi,eih->emh", activated_ref, down[route_indices].float())
    branch_ref = (branch_ref * route_scores.reshape(TOP_K_EXPERTS, 1, 1)).sum(dim=0, keepdim=True).unsqueeze(0)
    residual_ref = hidden.float() + branch_ref

    next_prefix = "model.language_model.layers.5."
    gamma = next_state[next_prefix + "input_layernorm.weight"].float()
    normalized_ref = residual_ref * torch.rsqrt(residual_ref.pow(2).mean(dim=-1, keepdim=True) + cfg.rms_norm_eps)
    normalized_ref = normalized_ref * gamma
    q = next_state[next_prefix + "self_attn.q_proj.weight"].transpose(-2, -1).contiguous()
    k = next_state[next_prefix + "self_attn.k_proj.weight"].transpose(-2, -1).contiguous()
    q_shards = q.chunk(TP_SIZE, dim=-1)
    k_heads = k.chunk(2, dim=-1)
    full_qkv = (
        torch.cat(
            [torch.cat((q_shards[rank], k_heads[rank // 2], k_heads[rank // 2]), dim=-1) for rank in range(TP_SIZE)],
            dim=-1,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    qkv_ref = torch.matmul(normalized_ref, full_qkv.float())
    residual_ok, residual_pcc = functional_tests.comp_pcc(residual_ref[:, :, :1, :], residual_actual[:, :, :1, :], 0.99)
    qkv_ok, qkv_pcc = functional_tests.comp_pcc(qkv_ref[:, :, :1, :], qkv_actual[:, :, :1, :], 0.99)
    trace_bit_exact = all(torch.equal(first, repeat) for first, repeat in zip(trace_first, trace_repeat))
    first_persistent_bit_exact = all(
        torch.equal(first, persistent) for first, persistent in zip(first_qkv_hosts, persistent_qkv_hosts)
    )

    # The next decoder's existing attention splitter accepts the candidate's
    # local physical QKV ordering without an all-gather or host reshape.
    q_heads, k_heads_tt, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
        trace_qkv,
        num_heads=next_decoder.local_q_heads,
        num_kv_heads=next_decoder.local_full_kv_heads,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert tuple(q_heads.shape)[-2:] == (next_decoder.local_q_heads, FULL_KIND.head_dim)
    assert tuple(k_heads_tt.shape)[-2:] == (next_decoder.local_full_kv_heads, FULL_KIND.head_dim)
    assert tuple(v_heads.shape)[-2:] == (next_decoder.local_full_kv_heads, FULL_KIND.head_dim)
    ttnn.synchronize_device(mesh_device)

    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/fractured_sparse")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_suffix = _measurement_artifact_suffix()
    artifact = {
        "gate": "C_product_two_layer_boundary",
        "status": "pass" if residual_ok and qkv_ok and trace_bit_exact else "fail",
        "source_layer": 0,
        "next_layer": 5,
        "active_ids": route_indices,
        "real_weights": True,
        "router": "actual source decoder router plus on-device compact top-k",
        "residual_local_shapes": [list(shard.shape) for shard in ttnn.get_device_tensors(trace_residual)],
        "qkv_local_shapes": [list(shard.shape) for shard in ttnn.get_device_tensors(trace_qkv)],
        "residual_vs_torch_pcc": float(residual_pcc),
        "qkv_vs_torch_pcc": float(qkv_pcc),
        "first_to_persistent_bit_exact": first_persistent_bit_exact,
        "trace_repeat_bit_exact": trace_bit_exact,
        "trace_warmups": warmups,
        "trace_iterations": iterations,
        "trace_ms": trace_ms,
        "program_cache_entries_before": cache_before,
        "program_cache_entries_after": cache_after,
        "persistent_resources": {
            "packed_reduce_scatter": len(resources["packed_buffers"]),
            "down_reduce_scatter": len(resources["down_buffers"]),
            "norm_all_gather": resources["norm_output"] is not None,
            "qkv_reduce_scatter": len(resources["qkv_buffers"]),
        },
        "fallback_policy": "TTNN throw_exception_on_fallback=true",
        "source_sha256_at_run": hashlib.sha256(
            Path("models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py").read_bytes()
        ).hexdigest(),
        "test_sha256_at_run": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (artifact_dir / f"gate_c_product_boundary{artifact_suffix}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    assert first_persistent_bit_exact
    assert trace_bit_exact
    assert cache_before == cache_after
    assert residual_ok, f"fractured product residual PCC={residual_pcc}"
    assert qkv_ok, f"fractured product next-layer QKV PCC={qkv_pcc}"


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "role,global_output_width",
    [
        pytest.param("sliding_qkv", SLIDING_KIND.qkv_width, id="sliding_qkv_n8192"),
        # TP4 duplicates both full-attention KV heads across rank pairs, so
        # physical column sharding carries 8192 Q + 2 * (2 * 1024) KV columns.
        pytest.param(
            "full_qkv_physical",
            FULL_KIND.q_width + 2 * FULL_KIND.kv_width * 2,
            id="full_qkv_physical_n12288",
        ),
        pytest.param("dense_gate_up", 2 * PADDED_MLP_INTERMEDIATE_SIZE, id="dense_gate_up_n4352"),
        pytest.param("router", 128, id="router_n128"),
        pytest.param("fixed_selected_expert_gate", PADDED_MOE_INTERMEDIATE_SIZE, id="fixed_expert_n768"),
    ],
)
def test_tp4_fused_all_gather_matmul_exact_residual_consumer_repro(mesh_device, role, global_output_width):
    """Exercise AG+local-column-matmul at Gemma's fractured residual boundary.

    Each rank begins with its exact 704-wide slice of the 2816-wide residual.
    The fused op must gather K while consuming a column-sharded projection, so
    no standalone gather or restoration to a replicated residual is hidden in
    this feasibility test.
    """
    import torch

    from ttnn import ConcatMeshToTensor, ShardTensorToMesh

    if os.getenv("GEMMA4_MULTICHIP_FUSED_AGMM_REPRO") != "1":
        pytest.skip("set GEMMA4_MULTICHIP_FUSED_AGMM_REPRO=1 for the serialized fused-AGMM hardware repro")
    if tuple(mesh_device.shape) != (1, TP_SIZE):
        pytest.skip(f"requires target 1x{TP_SIZE} mesh, got {tuple(mesh_device.shape)}")

    torch.manual_seed(1701 + global_output_width)
    residual = torch.randn(1, 1, 32, HIDDEN_SIZE, dtype=torch.bfloat16)
    weight = torch.randn(1, 1, HIDDEN_SIZE, global_output_width, dtype=torch.bfloat16)
    gamma = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16)
    normalized = residual.float() * torch.rsqrt(residual.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    normalized = (normalized * gamma.float()).bfloat16()
    expected = torch.matmul(normalized, weight)
    residual_tt = ttnn.from_torch(
        residual,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
    )
    replicate_output = role == "router"
    weight_tt = ttnn.from_torch(
        weight,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=(
            ttnn.ReplicateTensorToMesh(mesh_device) if replicate_output else ShardTensorToMesh(mesh_device, dim=3)
        ),
    )
    gamma_tt = ttnn.from_torch(
        gamma.reshape(TP_SIZE, 1, HIDDEN_SIZE // TP_SIZE // 32, 32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),
    )

    compute_grid = mesh_device.compute_with_storage_grid_size()
    worker_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([worker_cores])], 0)
    mesh_device.load_sub_device_manager(manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])
    norm_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(2)]
    agmm_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(3)]
    local_n_tiles = global_output_width // (1 if replicate_output else TP_SIZE) // 32
    grid_x = min(11, local_n_tiles)
    per_core_n = (local_n_tiles + grid_x - 1) // grid_x
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, 4),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    try:
        local_stats = ttnn.rms_norm_pre_all_gather(residual_tt, dtype=ttnn.bfloat16)
        global_stats = ttnn.experimental.all_gather_async(
            local_stats,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=norm_semaphores,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=worker_sub_device_id,
        )
        normalized_tt = ttnn.rms_norm_post_all_gather(
            residual_tt,
            global_stats,
            epsilon=1e-6,
            weight=gamma_tt,
        )
        _, output_tt = ttnn.experimental.all_gather_matmul_async(
            normalized_tt,
            weight_tt,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=agmm_semaphores,
            all_gather_core_grid_offset=(0, 6),
            num_links=2,
            topology=ttnn.Topology.Ring,
            subdevice_id=worker_sub_device_id,
            memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
        if replicate_output:
            replicated = ttnn.to_torch(
                ttnn.from_device(output_tt), mesh_composer=ConcatMeshToTensor(mesh_device, dim=0)
            )
            for rank_output in replicated.chunk(TP_SIZE, dim=0):
                assert_with_pcc(expected, rank_output[..., :global_output_width], 0.99)
        else:
            actual = ttnn.to_torch(ttnn.from_device(output_tt), mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))[
                ..., :global_output_width
            ]
            assert_with_pcc(expected, actual, 0.99)
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_capture_optimized_single_chip_reference(mesh_device, monkeypatch, layer_idx):
    """Explicit opt-in producer for the checked-in direct-comparison reference."""
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    if os.environ.get("GEMMA4_CAPTURE_SINGLE_CHIP_REFERENCE") != "1":
        pytest.skip("reference capture is an explicit hardware evidence step")
    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    cfg = functional_tests._load_text_config()
    state = functional_tests._load_layer_state(layer_idx)
    layer_type = cfg.layer_types[layer_idx]
    seq_len = 32
    torch.manual_seed(4200 + layer_idx)
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, torch.arange(seq_len).unsqueeze(0), layer_type=layer_type)
    decode_cos, decode_sin = rotary(decode_hidden, torch.tensor([[seq_len]]), layer_type=layer_type)

    def run(decoder_cls, target_mesh, local):
        decoder = decoder_cls.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=target_mesh)
        if layer_type == "full_attention":
            blocks, heads, block, dim = 2, (1 if local else 2), 128, 512
        else:
            blocks, heads, block, dim = 4, (2 if local else 8), 64, 256
        page_table = functional_tests._as_tt(
            target_mesh,
            torch.arange(blocks, dtype=torch.int32).view(1, blocks),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cache = tuple(
            functional_tests._as_tt(target_mesh, torch.zeros((blocks, heads, block, dim), dtype=torch.bfloat16))
            for _ in range(2)
        )
        prefill = decoder.prefill_forward(
            functional_tests._as_tt(target_mesh, hidden.unsqueeze(1)),
            position_cos=functional_tests._as_tt(target_mesh, cos.unsqueeze(1)),
            position_sin=functional_tests._as_tt(target_mesh, sin.unsqueeze(1)),
            page_table=page_table,
            kv_cache=cache,
        )
        decode = decoder.decode_forward(
            hidden_states=functional_tests._as_tt(target_mesh, decode_hidden.unsqueeze(1)),
            position_cos=functional_tests._as_tt(target_mesh, decode_cos.unsqueeze(1)),
            position_sin=functional_tests._as_tt(target_mesh, decode_sin.unsqueeze(1)),
            current_pos=functional_tests._as_tt(
                target_mesh,
                torch.tensor([seq_len], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            page_table=page_table,
            kv_cache=cache,
        )
        return tuple(ttnn.to_torch(ttnn.get_device_tensors(result.cpu())[0]) for result in (prefill, decode))

    baseline_prefill, baseline_decode = run(OptimizedDecoder, mesh_device, False)
    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"prefill": baseline_prefill, "decode": baseline_decode},
        artifact_dir / f"optimized_reference_layer{layer_idx}.pt",
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_multichip_matches_optimized_single_chip(mesh_device, device_params, monkeypatch, layer_idx):
    """Compare TP output directly with separately captured optimized TTNN output."""
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    cfg = functional_tests._load_text_config()
    state = functional_tests._load_layer_state(layer_idx)
    layer_type = cfg.layer_types[layer_idx]
    seq_len = 32
    torch.manual_seed(4200 + layer_idx)
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, torch.arange(seq_len).unsqueeze(0), layer_type=layer_type)
    decode_cos, decode_sin = rotary(decode_hidden, torch.tensor([[seq_len]]), layer_type=layer_type)
    decoder = MultichipDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    if layer_type == "full_attention":
        blocks, heads, block, dim = 2, decoder.local_full_kv_heads, 128, 512
    else:
        blocks, heads, block, dim = 4, decoder.local_sliding_kv_heads, 64, 256
    page_table = functional_tests._as_tt(
        mesh_device,
        torch.arange(blocks, dtype=torch.int32).view(1, blocks),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache = tuple(
        functional_tests._as_tt(mesh_device, torch.zeros((blocks, heads, block, dim), dtype=torch.bfloat16))
        for _ in range(2)
    )
    prefill = decoder.prefill_forward(
        functional_tests._as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=functional_tests._as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=functional_tests._as_tt(mesh_device, sin.unsqueeze(1)),
        page_table=page_table,
        kv_cache=cache,
    )
    decode = decoder.decode_forward(
        hidden_states=functional_tests._as_tt(mesh_device, decode_hidden.unsqueeze(1)),
        position_cos=functional_tests._as_tt(mesh_device, decode_cos.unsqueeze(1)),
        position_sin=functional_tests._as_tt(mesh_device, decode_sin.unsqueeze(1)),
        current_pos=functional_tests._as_tt(
            mesh_device,
            torch.tensor([seq_len], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        page_table=page_table,
        kv_cache=cache,
    )
    multichip_prefill, multichip_decode = (
        ttnn.to_torch(ttnn.get_device_tensors(result.cpu())[0]) for result in (prefill, decode)
    )
    reference = torch.load(
        Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
        / f"optimized_reference_layer{layer_idx}.pt",
        weights_only=True,
    )
    prefill_ok, prefill_pcc = functional_tests.comp_pcc(reference["prefill"], multichip_prefill, 0.995)
    decode_ok, decode_pcc = functional_tests.comp_pcc(reference["decode"], multichip_decode, 0.995)
    assert prefill_ok, f"TP{decoder.tp_size} layer {layer_idx} prefill PCC={prefill_pcc}"
    assert decode_ok, f"TP{decoder.tp_size} layer {layer_idx} decode PCC={decode_pcc}"
    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    suffix = _measurement_artifact_suffix()
    (artifact_dir / f"pcc_tp{decoder.tp_size}_layer{layer_idx}{suffix}.json").write_text(
        json.dumps(
            {
                "tp_size": decoder.tp_size,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "prefill_pcc": float(prefill_pcc),
                "decode_pcc": float(decode_pcc),
                "threshold": 0.995,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_p150_proxy_matches_optimized_single_chip(mesh_device, monkeypatch, layer_idx):
    """Validate the one-chip P150 proxy without fabric."""
    test_multichip_matches_optimized_single_chip(mesh_device, {}, monkeypatch, layer_idx)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_p150x2_proxy_matches_optimized_single_chip(mesh_device, device_params, monkeypatch, layer_idx):
    """Validate two adjacent P300C compute chips on the QB2 FABRIC_2D control plane."""
    proxy_mesh = mesh_device.create_submesh(ttnn.MeshShape((1, 2)), offset=ttnn.MeshCoordinate(0, 0))
    test_multichip_matches_optimized_single_chip(proxy_mesh, device_params, monkeypatch, layer_idx)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_p150x2_proxy_all_reduce_smoke(mesh_device, device_params):
    """Prove the exact TP2 hidden-width contraction collective independently."""
    import torch

    proxy_mesh = mesh_device.create_submesh(ttnn.MeshShape((1, 2)), offset=ttnn.MeshCoordinate(0, 0))
    source = torch.cat(
        [
            torch.ones((1, 1, 32, HIDDEN_SIZE), dtype=torch.bfloat16),
            torch.full((1, 1, 32, HIDDEN_SIZE), 2.0, dtype=torch.bfloat16),
        ],
        dim=0,
    )
    partial = ttnn.from_torch(
        source,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=proxy_mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(proxy_mesh, dim=0),
    )
    reduced = ttnn.all_reduce(
        partial,
        cluster_axis=1,
        num_links=1,
        topology=ttnn.Topology.Linear,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(proxy_mesh)
    for shard in ttnn.get_device_tensors(reduced.cpu()):
        assert torch.equal(ttnn.to_torch(shard), torch.full((1, 1, 32, HIDDEN_SIZE), 3.0, dtype=torch.bfloat16))


def _run_traced_batch32(decoder_cls, mesh_device, cfg, state, layer_idx, *, local_cache):
    """Run an identical batch-32 TTNN decode regime for baseline and TP4."""
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    batch, current_position = 32, 32
    layer_type = cfg.layer_types[layer_idx]
    torch.manual_seed(4300 + layer_idx)
    decode_hidden = torch.randn(batch, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    positions = torch.full((batch, 1), current_position, dtype=torch.long)
    cos, sin = rotary(decode_hidden, positions, layer_type=layer_type)
    decoder = decoder_cls.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    if layer_type == "full_attention":
        blocks_per_user, heads, block, dim = 2, (1 if local_cache else 2), 128, 512
    else:
        blocks_per_user, heads, block, dim = 4, (2 if local_cache else 8), 64, 256
    page_table = functional_tests._as_tt(
        mesh_device,
        torch.arange(batch * blocks_per_user, dtype=torch.int32).view(batch, blocks_per_user),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = (batch * blocks_per_user, heads, block, dim)
    kv_cache = tuple(
        functional_tests._as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)) for _ in range(2)
    )
    if layer_type == "full_attention":
        torch.manual_seed(4400 + layer_idx)
        prefix = torch.randn(1, current_position, HIDDEN_SIZE, dtype=torch.bfloat16).expand(batch, -1, -1).clone()
        prefix_positions = torch.arange(current_position).view(1, -1).expand(batch, -1)
        prefix_cos, prefix_sin = rotary(prefix, prefix_positions, layer_type=layer_type)
        for user_id in range(batch):
            decoder.prefill_forward(
                functional_tests._as_tt(mesh_device, prefix[user_id : user_id + 1].unsqueeze(1)),
                position_cos=functional_tests._as_tt(mesh_device, prefix_cos[user_id : user_id + 1].unsqueeze(1)),
                position_sin=functional_tests._as_tt(mesh_device, prefix_sin[user_id : user_id + 1].unsqueeze(1)),
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user_id,
            )
    tt_cos = cos.unsqueeze(0) if layer_type == "sliding_attention" else cos.transpose(0, 1).unsqueeze(0)
    tt_sin = sin.unsqueeze(0) if layer_type == "sliding_attention" else sin.transpose(0, 1).unsqueeze(0)
    decode_args = {
        "hidden_states": functional_tests._as_tt(mesh_device, decode_hidden.transpose(0, 1).unsqueeze(0)),
        "position_cos": functional_tests._as_tt(mesh_device, tt_cos),
        "position_sin": functional_tests._as_tt(mesh_device, tt_sin),
        "current_pos": functional_tests._as_tt(
            mesh_device,
            torch.full((batch,), current_position, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    eager = decoder.decode_forward(**decode_args)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = decoder.decode_forward(**decode_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for _ in range(5):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    first = ttnn.to_torch(ttnn.get_device_tensors(traced.cpu())[0])
    iterations = 30
    started = time.perf_counter()
    for _ in range(iterations):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    elapsed_ms = (time.perf_counter() - started) * 1000 / iterations
    second = ttnn.to_torch(ttnn.get_device_tensors(traced.cpu())[0])
    eager_host = ttnn.to_torch(ttnn.get_device_tensors(eager.cpu())[0])
    ttnn.release_trace(mesh_device, trace_id)
    assert torch.equal(first, second)
    assert torch.equal(eager_host, second)
    return second, elapsed_ms, cache_shape


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_capture_optimized_batch32_reference(mesh_device, device_params, monkeypatch, layer_idx):
    import torch

    if os.environ.get("GEMMA4_CAPTURE_SINGLE_CHIP_REFERENCE") != "1":
        pytest.skip("reference capture is an explicit hardware evidence step")
    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    cfg = functional_tests._load_text_config()
    state = functional_tests._load_layer_state(layer_idx)
    output, latency_ms, cache_shape = _run_traced_batch32(
        OptimizedDecoder, mesh_device, cfg, state, layer_idx, local_cache=False
    )
    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    torch.save(output, artifact_dir / f"optimized_batch32_layer{layer_idx}.pt")
    (artifact_dir / f"optimized_batch32_layer{layer_idx}.json").write_text(
        json.dumps(
            {
                "layer_idx": layer_idx,
                "batch": 32,
                "decode_current_position": 32,
                "prefill_sequence_length": 32 if cfg.layer_types[layer_idx] == "full_attention" else 0,
                "cache_capacity_tokens_per_user": 256,
                "trace_replay_ms": latency_ms,
                "cache_shape": cache_shape,
                "repeat_bit_exact": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_multichip_batch32_trace_and_optimized_pcc(mesh_device, device_params, monkeypatch, layer_idx):
    import torch

    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    cfg = functional_tests._load_text_config()
    state = functional_tests._load_layer_state(layer_idx)
    output, latency_ms, cache_shape = _run_traced_batch32(
        MultichipDecoder, mesh_device, cfg, state, layer_idx, local_cache=True
    )
    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    reference = torch.load(artifact_dir / f"optimized_batch32_layer{layer_idx}.pt", weights_only=True)
    pcc_ok, pcc = functional_tests.comp_pcc(reference, output, 0.995)
    assert pcc_ok, f"TP4 batch-32 layer {layer_idx} PCC={pcc}"
    baseline = json.loads((artifact_dir / f"optimized_batch32_layer{layer_idx}.json").read_text())
    suffix = _measurement_artifact_suffix()
    (artifact_dir / f"multichip_batch32_layer{layer_idx}{suffix}.json").write_text(
        json.dumps(
            {
                "layer_idx": layer_idx,
                "layer_type": cfg.layer_types[layer_idx],
                "batch": 32,
                "decode_current_position": 32,
                "prefill_sequence_length": 32 if cfg.layer_types[layer_idx] == "full_attention" else 0,
                "cache_capacity_tokens_per_user": 256,
                "trace_replay_ms": latency_ms,
                "single_chip_trace_replay_ms": baseline["trace_replay_ms"],
                "speedup": baseline["trace_replay_ms"] / latency_ms,
                "tp_efficiency": baseline["trace_replay_ms"] / latency_ms / TP_SIZE,
                "local_cache_shape": cache_shape,
                "repeat_bit_exact": True,
                "optimized_pcc_threshold": 0.995,
                "optimized_pcc": float(pcc),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical,decode_pcc",
    [
        pytest.param(0, False, 0.995, id="sliding_attention"),
        pytest.param(5, False, 0.995, id="full_attention"),
    ],
)
def test_multichip_real_weights_prefill_decode(
    mesh_device, device_params, monkeypatch, layer_idx, shared_physical, decode_pcc
):
    """Reuse the established HF oracle with TP-local cache geometry."""

    def local_cache_shape(layer_type, *, shared_physical, token_capacity=None):
        if layer_type == "full_attention":
            block_size, heads, head_dim, default_blocks = 128, 1, 512, 2
        else:
            block_size, heads, head_dim, default_blocks = 64, 2, 256, 4
        blocks = default_blocks if token_capacity is None else (token_capacity + block_size - 1) // block_size
        return blocks, heads, block_size, head_dim

    def replicated_output(_mesh, tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor.cpu())[0])

    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", MultichipDecoder)
    monkeypatch.setattr(functional_tests, "_cache_shape", local_cache_shape)
    monkeypatch.setattr(functional_tests, "_to_torch", replicated_output)
    monkeypatch.setattr(
        functional_tests,
        "ARTIFACT_DIR",
        Path(
            os.getenv(
                "GEMMA4_MULTICHIP_ARTIFACT_DIR",
                "models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts",
            )
        ),
    )
    functional_tests.test_functional_decoder_real_weights_prefill_decode(
        mesh_device, device_params, layer_idx, shared_physical, decode_pcc
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_tp4_real_weight_attention_decode_precision_probe(mesh_device, device_params, monkeypatch, layer_idx):
    """Compare one decode-only attention precision policy with the optimized baseline."""
    import hashlib

    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    mode = os.getenv("GEMMA4_MULTICHIP_ATTENTION_PROBE")
    allowed_modes = {"selected", "qkv_control", "qkv_bfp4", "o_proj_bfp4"}
    if mode not in allowed_modes:
        pytest.skip("set GEMMA4_MULTICHIP_ATTENTION_PROBE to selected, qkv_control, qkv_bfp4, or o_proj_bfp4")

    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    monkeypatch.delenv("GEMMA4_MULTICHIP_DECODE_QKV_WEIGHT_DTYPE", raising=False)
    monkeypatch.delenv("GEMMA4_MULTICHIP_DECODE_O_WEIGHT_DTYPE", raising=False)
    selected_dram_roles = "o_proj,packed_mlp_gate_up,mlp_down"
    if mode.startswith("qkv_"):
        monkeypatch.setenv("GEMMA4_MULTICHIP_DRAM_SHARDED_ROLES", f"qkv,{selected_dram_roles}")
    else:
        monkeypatch.setenv("GEMMA4_MULTICHIP_DRAM_SHARDED_ROLES", selected_dram_roles)
    if mode == "qkv_bfp4":
        monkeypatch.setenv("GEMMA4_MULTICHIP_DECODE_QKV_WEIGHT_DTYPE", "bfp4")
    elif mode == "o_proj_bfp4":
        monkeypatch.setenv("GEMMA4_MULTICHIP_DECODE_O_WEIGHT_DTYPE", "bfp4")

    cfg = functional_tests._load_text_config()
    state = functional_tests._load_layer_state(layer_idx)
    layer_type = cfg.layer_types[layer_idx]
    seq_len = 32
    torch.manual_seed(4200 + layer_idx)
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, torch.arange(seq_len).unsqueeze(0), layer_type=layer_type)
    decode_cos, decode_sin = rotary(decode_hidden, torch.tensor([[seq_len]]), layer_type=layer_type)
    decoder = MultichipDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    full_qkv_bypass_overridden_in_probe = mode.startswith("qkv_") and layer_type == "full_attention"
    if full_qkv_bypass_overridden_in_probe:
        selected_use_decode_dram_weight = decoder._use_decode_dram_weight

        def use_decode_qkv_candidate(x, weight_name):
            if weight_name == "qkv":
                return decoder.multichip_execution_phase == "decode" and weight_name in decoder.decode_dram_weights
            return selected_use_decode_dram_weight(x, weight_name)

        decoder._use_decode_dram_weight = use_decode_qkv_candidate

    if layer_type == "full_attention":
        blocks, heads, block, dim = 2, decoder.local_full_kv_heads, 128, 512
    else:
        blocks, heads, block, dim = 4, decoder.local_sliding_kv_heads, 64, 256
    page_table = functional_tests._as_tt(
        mesh_device,
        torch.arange(blocks, dtype=torch.int32).view(1, blocks),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_cache = tuple(
        functional_tests._as_tt(mesh_device, torch.zeros((blocks, heads, block, dim), dtype=torch.bfloat16))
        for _ in range(2)
    )

    observed_attention_weights = []
    original_linear = decoder._linear

    def observed_linear(x, weight_name, **kwargs):
        if weight_name in {"qkv", "o_proj"}:
            decode_candidate = decoder._use_decode_dram_weight(x, weight_name)
            weight = (
                decoder.decode_dram_weights[weight_name] if decode_candidate else getattr(decoder.weights, weight_name)
            )
            observed_attention_weights.append(
                {
                    "phase": decoder.multichip_execution_phase,
                    "role": weight_name,
                    "decode_candidate": decode_candidate,
                    "weight_dtype": str(weight.dtype),
                    "weight_memory_layout": str(weight.memory_config().memory_layout),
                    "weight_buffer_type": str(weight.memory_config().buffer_type),
                    "input_shape": list(x.shape),
                }
            )
        return original_linear(x, weight_name, **kwargs)

    decoder._linear = observed_linear
    prefill = decoder.prefill_forward(
        functional_tests._as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=functional_tests._as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=functional_tests._as_tt(mesh_device, sin.unsqueeze(1)),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    decode_args = {
        "hidden_states": functional_tests._as_tt(mesh_device, decode_hidden.unsqueeze(1)),
        "position_cos": functional_tests._as_tt(mesh_device, decode_cos.unsqueeze(1)),
        "position_sin": functional_tests._as_tt(mesh_device, decode_sin.unsqueeze(1)),
        "current_pos": functional_tests._as_tt(
            mesh_device,
            torch.tensor([seq_len], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    eager = decoder.decode_forward(**decode_args)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = decoder.decode_forward(**decode_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for _ in range(5):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        first = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(traced.cpu())]
        iterations = 30
        started = time.perf_counter()
        for _ in range(iterations):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        trace_replay_ms = (time.perf_counter() - started) * 1000 / iterations
        second = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(traced.cpu())]
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    eager_shards = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(eager.cpu())]
    prefill_shards = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(prefill.cpu())]
    cache_shards = [[ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(cache.cpu())] for cache in kv_cache]
    reference_path = (
        Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
        / f"optimized_reference_layer{layer_idx}.pt"
    )
    reference = torch.load(reference_path, weights_only=True)
    prefill_ok, prefill_pcc = functional_tests.comp_pcc(reference["prefill"], prefill_shards[0], 0.995)
    decode_ok, decode_pcc = functional_tests.comp_pcc(reference["decode"], second[0], 0.995)
    prefill_calls = [entry for entry in observed_attention_weights if entry["phase"] == "prefill"]
    decode_calls = [entry for entry in observed_attention_weights if entry["phase"] == "decode"]
    candidate_role = "qkv" if mode == "qkv_bfp4" else "o_proj" if mode == "o_proj_bfp4" else None
    role_reached_runtime = candidate_role is None or any(
        entry["role"] == candidate_role and entry["decode_candidate"] and entry["weight_dtype"] == "DataType.BFLOAT4_B"
        for entry in decode_calls
    )
    prefill_preserved = all(not entry["decode_candidate"] for entry in prefill_calls)
    other_role_preserved = candidate_role is None or all(
        entry["weight_dtype"] != "DataType.BFLOAT4_B" for entry in decode_calls if entry["role"] != candidate_role
    )
    eager_trace_bit_exact = all(
        torch.equal(eager_rank, trace_rank) for eager_rank, trace_rank in zip(eager_shards, first)
    )
    trace_repeat_bit_exact = all(torch.equal(first_rank, second_rank) for first_rank, second_rank in zip(first, second))
    replicas_bit_exact = all(torch.equal(second[0], rank) for rank in second[1:])
    cache_nonzero_on_all_ranks = all(torch.count_nonzero(rank).item() > 0 for cache in cache_shards for rank in cache)

    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    source_path = Path("models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py")
    test_path = Path("models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py")
    artifact = {
        "mode": mode,
        "candidate_role": candidate_role,
        "tp_size": decoder.tp_size,
        "mesh_shape": [1, 4],
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "real_weights": True,
        "logical_prefill_length": seq_len,
        "decode_current_position": seq_len,
        "decode_batch": 1,
        "page_table_shape": list(page_table.shape),
        "local_cache_shape": [blocks, heads, block, dim],
        "cache_dtype": str(kv_cache[0].dtype),
        "cache_nonzero_on_all_ranks": cache_nonzero_on_all_ranks,
        "prefill_pcc": float(prefill_pcc),
        "decode_pcc": float(decode_pcc),
        "pcc_threshold": 0.995,
        "warmups": 5,
        "trace_iterations": iterations,
        "trace_replay_ms": trace_replay_ms,
        "eager_vs_trace_bit_exact": eager_trace_bit_exact,
        "trace_repeat_bit_exact": trace_repeat_bit_exact,
        "replicas_bit_exact": replicas_bit_exact,
        "prefill_weights_preserved": prefill_preserved,
        "other_attention_role_preserved": other_role_preserved,
        "candidate_role_reached_runtime": role_reached_runtime,
        "full_qkv_bypass_overridden_in_probe": full_qkv_bypass_overridden_in_probe,
        "attention_weight_calls": observed_attention_weights,
        "decode_weight_sources": dict(decoder.decode_weight_sources),
        "decode_attention_math_fidelity": str(decoder.decode_attention_compute_config.math_fidelity),
        "sdpa_program_config": str(decoder.sdpa_program_config),
        "fallback_policy": "TTNN throw_exception_on_fallback=true",
        "optimized_reference_artifact": reference_path.name,
        "optimized_reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
    }
    artifact_path = artifact_dir / f"bfp4_attention_{mode}_layer{layer_idx}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    assert prefill_ok, f"{mode} layer {layer_idx} prefill PCC={prefill_pcc}"
    assert decode_ok, f"{mode} layer {layer_idx} cache-consuming decode PCC={decode_pcc}"
    assert prefill_preserved
    assert other_role_preserved
    assert role_reached_runtime
    assert cache_nonzero_on_all_ranks
    assert eager_trace_bit_exact
    assert trace_repeat_bit_exact
    assert replicas_bit_exact


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_p150x2_proxy_real_weights_prefill_decode(mesh_device, device_params, monkeypatch, layer_idx):
    """Validate the TP2 selected policy directly against HF for both layer kinds."""

    def local_cache_shape(layer_type, *, shared_physical, token_capacity=None):
        del shared_physical
        if layer_type == "full_attention":
            block_size, heads, head_dim, default_blocks = 128, 1, 512, 2
        else:
            block_size, heads, head_dim, default_blocks = 64, 4, 256, 4
        blocks = default_blocks if token_capacity is None else (token_capacity + block_size - 1) // block_size
        return blocks, heads, block_size, head_dim

    def replicated_output(_mesh, tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor.cpu())[0])

    proxy_mesh = mesh_device.create_submesh(ttnn.MeshShape((1, 2)), offset=ttnn.MeshCoordinate(0, 0))
    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", MultichipDecoder)
    monkeypatch.setattr(functional_tests, "_cache_shape", local_cache_shape)
    monkeypatch.setattr(functional_tests, "_to_torch", replicated_output)
    monkeypatch.setattr(
        functional_tests,
        "ARTIFACT_DIR",
        Path(
            os.getenv(
                "GEMMA4_MULTICHIP_ARTIFACT_DIR",
                "models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/tp2_hf_oracle",
            )
        ),
    )
    functional_tests.test_functional_decoder_real_weights_prefill_decode(
        proxy_mesh,
        device_params,
        layer_idx,
        False,
        0.995,
    )


def _install_multichip_functional_harness(monkeypatch, mesh_device):
    """Adapt capacity tests to TP-local cache geometry and first replicated rank."""
    profile = _profile_for_tp(tuple(mesh_device.shape)[1])
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", MultichipDecoder)
    monkeypatch.setattr(functional_tests, "SLIDING_NUM_KV_HEADS", profile.local_sliding_kv_heads)
    monkeypatch.setattr(functional_tests, "FULL_NUM_KV_HEADS", profile.local_full_kv_heads)
    monkeypatch.setattr(
        functional_tests,
        "_to_torch",
        lambda _mesh, tensor: ttnn.to_torch(ttnn.get_device_tensors(tensor.cpu())[0]),
    )
    monkeypatch.setattr(
        functional_tests,
        "ARTIFACT_DIR",
        Path(
            os.getenv(
                "GEMMA4_MULTICHIP_ARTIFACT_DIR",
                "models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts",
            )
        ),
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_multichip_advertised_context_traced_decode(mesh_device, device_params, monkeypatch, layer_idx):
    _install_multichip_functional_harness(monkeypatch, mesh_device)
    functional_tests.test_advertised_context_traced_decode(mesh_device, device_params, layer_idx)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_multichip_prefill_capacity(mesh_device, device_params, monkeypatch, layer_idx):
    _install_multichip_functional_harness(monkeypatch, mesh_device)
    functional_tests.test_prefill_capacity_probe(mesh_device, device_params, layer_idx)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_p150_prefill_capacity_boundary(mesh_device, device_params, monkeypatch, layer_idx):
    """Probe a requested real-weight TP1 length at the analytical capacity boundary."""
    _install_multichip_functional_harness(monkeypatch, mesh_device)
    functional_tests.test_prefill_capacity_probe(mesh_device, device_params, layer_idx)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_multichip_bounded_modulo_tail_integrity(mesh_device, device_params, monkeypatch):
    _install_multichip_functional_harness(monkeypatch, mesh_device)
    functional_tests.test_bounded_modulo_prefill_tail_cache_integrity(mesh_device, device_params)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
@pytest.mark.parametrize(
    "layer_idx,shared_physical", [(0, True), (5, False)], ids=["sliding_attention", "full_attention"]
)
def test_multichip_perf_profile(mesh_device, device_params, monkeypatch, layer_idx, shared_physical, batch):
    _install_multichip_functional_harness(monkeypatch, mesh_device)
    functional_tests.test_functional_decoder_perf_profile(mesh_device, device_params, layer_idx, shared_physical, batch)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_capture_optimized_batch1_reference(mesh_device, device_params, monkeypatch, layer_idx):
    """Capture the optimized B1 baseline under the TP4 test's S=33 regime."""
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    if os.environ.get("GEMMA4_CAPTURE_SINGLE_CHIP_REFERENCE") != "1":
        pytest.skip("reference capture is an explicit hardware evidence step")
    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    cfg = functional_tests._load_text_config()
    state = functional_tests._load_layer_state(layer_idx)
    layer_type = cfg.layer_types[layer_idx]
    seq_len = 33
    torch.manual_seed(4100 + layer_idx)
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, torch.arange(seq_len).unsqueeze(0), layer_type=layer_type)
    decode_cos, decode_sin = rotary(decode_hidden, torch.tensor([[seq_len]]), layer_type=layer_type)
    decoder = OptimizedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    if layer_type == "full_attention":
        blocks, heads, block, dim = 2, 2, 128, 512
    else:
        blocks, heads, block, dim = 4, 8, 64, 256
    page_table = functional_tests._as_tt(
        mesh_device,
        torch.arange(blocks, dtype=torch.int32).view(1, blocks),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_cache = tuple(
        functional_tests._as_tt(mesh_device, torch.zeros((blocks, heads, block, dim), dtype=torch.bfloat16))
        for _ in range(2)
    )
    decoder.prefill_forward(
        functional_tests._as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=functional_tests._as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=functional_tests._as_tt(mesh_device, sin.unsqueeze(1)),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    decode_args = {
        "hidden_states": functional_tests._as_tt(mesh_device, decode_hidden.unsqueeze(1)),
        "position_cos": functional_tests._as_tt(mesh_device, decode_cos.unsqueeze(1)),
        "position_sin": functional_tests._as_tt(mesh_device, decode_sin.unsqueeze(1)),
        "current_pos": functional_tests._as_tt(
            mesh_device,
            torch.tensor([seq_len], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    eager = decoder.decode_forward(**decode_args)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = decoder.decode_forward(**decode_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for _ in range(5):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    first = ttnn.to_torch(ttnn.get_device_tensors(traced.cpu())[0])
    iterations = 30
    started = time.perf_counter()
    for _ in range(iterations):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    latency_ms = (time.perf_counter() - started) * 1000 / iterations
    second = ttnn.to_torch(ttnn.get_device_tensors(traced.cpu())[0])
    eager_host = ttnn.to_torch(ttnn.get_device_tensors(eager.cpu())[0])
    ttnn.release_trace(mesh_device, trace_id)
    assert torch.equal(first, second)
    assert torch.equal(eager_host, second)
    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    (artifact_dir / f"optimized_batch1_layer{layer_idx}.json").write_text(
        json.dumps(
            {
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "logical_prefill_length": seq_len,
                "decode_batch": 1,
                "warmups": 5,
                "iterations": iterations,
                "trace_replay_ms": latency_ms,
                "repeat_bit_exact": True,
                "cache_shape": [blocks, heads, block, dim],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _run_multichip_non_aligned_prefill_and_decode_trace(mesh_device, monkeypatch, layer_idx):
    """Exercise one profile's S=33 local-cache and warmed trace contract."""
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    cfg = functional_tests._load_text_config()
    state = functional_tests._load_layer_state(layer_idx)
    layer_type = cfg.layer_types[layer_idx]
    seq_len = 33
    torch.manual_seed(4100 + layer_idx)
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, torch.arange(seq_len).unsqueeze(0), layer_type=layer_type)
    decode_cos, decode_sin = rotary(decode_hidden, torch.tensor([[seq_len]]), layer_type=layer_type)
    decoder = MultichipDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    tp_size = decoder.tp_size
    if layer_type == "full_attention":
        blocks, heads, block, dim = 2, decoder.local_full_kv_heads, 128, 512
    else:
        blocks, heads, block, dim = 4, decoder.local_sliding_kv_heads, 64, 256
    page_table = functional_tests._as_tt(
        mesh_device,
        torch.arange(blocks, dtype=torch.int32).view(1, blocks),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_cache = tuple(
        functional_tests._as_tt(mesh_device, torch.zeros((blocks, heads, block, dim), dtype=torch.bfloat16))
        for _ in range(2)
    )
    prefill = decoder.prefill_forward(
        functional_tests._as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=functional_tests._as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=functional_tests._as_tt(mesh_device, sin.unsqueeze(1)),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    assert tuple(prefill.shape) == (1, 1, seq_len, HIDDEN_SIZE)
    decode_args = {
        "hidden_states": functional_tests._as_tt(mesh_device, decode_hidden.unsqueeze(1)),
        "position_cos": functional_tests._as_tt(mesh_device, decode_cos.unsqueeze(1)),
        "position_sin": functional_tests._as_tt(mesh_device, decode_sin.unsqueeze(1)),
        "current_pos": functional_tests._as_tt(
            mesh_device,
            torch.tensor([seq_len], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    eager = decoder.decode_forward(**decode_args)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = decoder.decode_forward(**decode_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    first = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(traced.cpu())]
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    second = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(traced.cpu())]
    eager_shards = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(eager.cpu())]
    for rank in range(tp_size):
        assert torch.equal(first[rank], second[rank])
        assert torch.equal(first[0], first[rank]), "replicated residual diverged across TP ranks"
        assert torch.equal(eager_shards[rank], first[rank])
    page_shards = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(page_table.cpu())]
    pos_shards = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(decode_args["current_pos"].cpu())]
    for rank in range(1, tp_size):
        assert torch.equal(page_shards[0], page_shards[rank])
        assert torch.equal(pos_shards[0], pos_shards[rank])
    cache_shards = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(kv_cache[0].cpu())]
    assert all(tuple(shard.shape) == (blocks, heads, block, dim) for shard in cache_shards)
    if layer_type == "full_attention" and tp_size == 4:
        assert torch.equal(cache_shards[0], cache_shards[1]), "full KV head 0 pair diverged"
        assert torch.equal(cache_shards[2], cache_shards[3]), "full KV head 1 pair diverged"
        assert not torch.equal(cache_shards[0], cache_shards[2]), "distinct full KV heads collapsed"
    assert decoder.multichip_path_counters["attention_tp"] >= 3
    assert decoder.multichip_path_counters["expert_tp"] >= 3
    if tp_size == 1:
        assert decoder.multichip_path_counters["all_reduce"] == 0
    else:
        assert decoder.multichip_path_counters["all_reduce"] >= 9
    for _ in range(5):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    iterations = 30
    started = time.perf_counter()
    for _ in range(iterations):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    multi_ms = (time.perf_counter() - started) * 1000 / iterations
    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((artifact_dir / f"optimized_batch1_layer{layer_idx}.json").read_text())
    single_ms = baseline["trace_replay_ms"]
    suffix = _measurement_artifact_suffix()
    artifact_stem = f"trace_{layer_type}_batch1" if tp_size == 4 else f"trace_tp{tp_size}_{layer_type}_batch1"
    (artifact_dir / f"{artifact_stem}{suffix}.json").write_text(
        json.dumps(
            {
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "profile": {1: "P150", 2: "P150x2", 4: "P150x4"}[tp_size],
                "tp_size": tp_size,
                "logical_prefill_length": seq_len,
                "decode_batch": 1,
                "warmups": 5,
                "iterations": iterations,
                "single_chip_optimized_baseline_ms": single_ms,
                "single_chip_source": f"optimized_batch1_layer{layer_idx}.json",
                "multichip_trace_replay_ms": multi_ms,
                "speedup": single_ms / multi_ms,
                "tp_efficiency": single_ms / multi_ms / tp_size,
                "repeat_bit_exact": True,
                "replicas_bit_exact": True,
                "page_table_replicated": True,
                "current_position_replicated": True,
                "local_cache_shape": [blocks, heads, block, dim],
                "full_kv_pair_duplication_verified": layer_type == "full_attention" and tp_size == 4,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_multichip_non_aligned_prefill_and_decode_trace(mesh_device, device_params, monkeypatch, layer_idx):
    """Validate and time the mandatory P150x4 proxy trace path."""
    _run_multichip_non_aligned_prefill_and_decode_trace(mesh_device, monkeypatch, layer_idx)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("tp_size", [1, 2], ids=["p150_tp1", "p150x2_tp2"])
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_required_profile_warmed_trace_latency(mesh_device, device_params, monkeypatch, tp_size, layer_idx):
    """Validate and time selected P150/P150x2 policies against one baseline."""
    target_mesh = mesh_device.create_submesh(ttnn.MeshShape((1, tp_size)), offset=ttnn.MeshCoordinate(0, 0))
    _run_multichip_non_aligned_prefill_and_decode_trace(target_mesh, monkeypatch, layer_idx)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 128 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_tp2_stacked_mixed_attention_shared_persistent_ccl_trace(mesh_device, device_params, monkeypatch):
    """Exercise shared rotating CCL resources across a real two-layer chain."""
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    monkeypatch.setenv("GEMMA4_MULTICHIP_PERSISTENT_ALL_REDUCE", "1")
    reference_policy = "selected_optimized"
    reference_kwargs = {}
    if os.getenv("GEMMA4_MULTICHIP_STACKED_REFERENCE_R0") == "1":
        reference_policy = "r0_raw_equivalent"
        monkeypatch.setenv("GEMMA4_OPT_ROUTING_ROW_MAJOR", "0")
        reference_kwargs = {
            "residual_shard_cores": 0,
            "packed_expert_decode_gate_up": False,
            "packed_expert_prefill_gate_up": False,
            "folded_router_projection": False,
            "shared_ffn_norm": False,
            "folded_expert_scale": False,
            "fused_final_scalar": False,
            "expert_gate_weight_dtype": ttnn.bfloat8_b,
        }
    cfg = functional_tests._load_text_config()
    layer_indices = (0, 5)
    states = {layer_idx: functional_tests._load_layer_state(layer_idx) for layer_idx in layer_indices}
    tp_mesh = mesh_device.create_submesh(ttnn.MeshShape((1, 2)), offset=ttnn.MeshCoordinate(0, 0))
    reference_mesh = mesh_device.create_submesh(ttnn.MeshShape((1, 1)), offset=ttnn.MeshCoordinate(1, 0))

    reference_layers = [
        OptimizedDecoder.from_state_dict(
            states[layer_idx],
            hf_config=cfg,
            layer_idx=layer_idx,
            mesh_device=reference_mesh,
            **reference_kwargs,
        )
        for layer_idx in layer_indices
    ]
    multichip_layers = [MultichipDecoder.from_state_dict(states[0], hf_config=cfg, layer_idx=0, mesh_device=tp_mesh)]
    shared_resources = multichip_layers[0].persistent_all_reduce_resources
    assert shared_resources is not None
    multichip_layers.append(
        MultichipDecoder.from_state_dict(
            states[5],
            hf_config=cfg,
            layer_idx=5,
            mesh_device=tp_mesh,
            persistent_all_reduce_resources=shared_resources,
        )
    )
    for decoder in multichip_layers:
        assert decoder.persistent_all_reduce_resources is shared_resources
        assert decoder.persistent_all_reduce_buffers is shared_resources["buffers"]
        assert decoder.persistent_all_reduce_semaphores is shared_resources["semaphores"]
    assert len(shared_resources["buffers"]) == len(shared_resources["semaphores"]) == 3

    seq_len = 32
    torch.manual_seed(4505)
    prefill_hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    positions = torch.arange(seq_len).unsqueeze(0)
    decode_positions = torch.tensor([[seq_len]])
    rotary = Gemma4TextRotaryEmbedding(cfg)
    host_rope = {}
    for layer_idx in layer_indices:
        layer_type = cfg.layer_types[layer_idx]
        host_rope[layer_idx] = (
            *rotary(prefill_hidden, positions, layer_type=layer_type),
            *rotary(decode_hidden, decode_positions, layer_type=layer_type),
        )

    def make_layer_calls(target_mesh, decoders, *, local_cache):
        calls = []
        for layer_idx, decoder in zip(layer_indices, decoders):
            kind = SLIDING_KIND if cfg.layer_types[layer_idx] == "sliding_attention" else FULL_KIND
            blocks = 4 if kind is SLIDING_KIND else 2
            if local_cache:
                heads = decoder.local_sliding_kv_heads if kind is SLIDING_KIND else decoder.local_full_kv_heads
            else:
                heads = kind.num_kv_heads
            page_table = functional_tests._as_tt(
                target_mesh,
                torch.arange(blocks, dtype=torch.int32).view(1, blocks),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            cache_shape = (blocks, heads, kind.block_size, kind.head_dim)
            kv_cache = tuple(
                functional_tests._as_tt(target_mesh, torch.zeros(cache_shape, dtype=torch.bfloat16)) for _ in range(2)
            )
            prefill_cos, prefill_sin, decode_cos, decode_sin = host_rope[layer_idx]
            calls.append(
                {
                    "prefill": {
                        "position_cos": functional_tests._as_tt(target_mesh, prefill_cos.unsqueeze(1)),
                        "position_sin": functional_tests._as_tt(target_mesh, prefill_sin.unsqueeze(1)),
                        "page_table": page_table,
                        "kv_cache": kv_cache,
                    },
                    "decode": {
                        "position_cos": functional_tests._as_tt(target_mesh, decode_cos.unsqueeze(1)),
                        "position_sin": functional_tests._as_tt(target_mesh, decode_sin.unsqueeze(1)),
                        "current_pos": functional_tests._as_tt(
                            target_mesh,
                            torch.tensor([seq_len], dtype=torch.int32),
                            dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                        ),
                        "page_table": page_table,
                        "kv_cache": kv_cache,
                    },
                    "cache_shape": cache_shape,
                }
            )
        return calls

    reference_calls = make_layer_calls(reference_mesh, reference_layers, local_cache=False)
    multichip_calls = make_layer_calls(tp_mesh, multichip_layers, local_cache=True)

    def run_chain(decoders, hidden_states, calls, phase):
        output = hidden_states
        layer_outputs = []
        for decoder, layer_call in zip(decoders, calls):
            if phase == "prefill":
                output = decoder.prefill_forward(output, **layer_call[phase])
            else:
                output = decoder.decode_forward(hidden_states=output, **layer_call[phase])
            layer_outputs.append(output)
        return layer_outputs

    reference_prefill_layers = run_chain(
        reference_layers,
        functional_tests._as_tt(reference_mesh, prefill_hidden.unsqueeze(1)),
        reference_calls,
        "prefill",
    )
    multichip_prefill_layers = run_chain(
        multichip_layers,
        functional_tests._as_tt(tp_mesh, prefill_hidden.unsqueeze(1)),
        multichip_calls,
        "prefill",
    )
    reference_prefill_hosts = [
        ttnn.to_torch(ttnn.get_device_tensors(output.cpu())[0]) for output in reference_prefill_layers
    ]
    multichip_prefill_shards_by_layer = [
        [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(output.cpu())]
        for output in multichip_prefill_layers
    ]

    # The chained layer-5 comparison below intentionally gives the two paths
    # different layer-0 outputs. Run layer 5 independently on identical inputs
    # and fresh caches so every meaningful layer kind retains a strict
    # selected-policy acceptance boundary.
    same_input_reference_call = make_layer_calls(reference_mesh, reference_layers, local_cache=False)[1]
    same_input_multichip_call = make_layer_calls(tp_mesh, multichip_layers, local_cache=True)[1]
    same_input_reference_prefill = reference_layers[1].prefill_forward(
        functional_tests._as_tt(reference_mesh, prefill_hidden.unsqueeze(1)),
        **same_input_reference_call["prefill"],
    )
    same_input_multichip_prefill = multichip_layers[1].prefill_forward(
        functional_tests._as_tt(tp_mesh, prefill_hidden.unsqueeze(1)),
        **same_input_multichip_call["prefill"],
    )
    same_input_reference_decode = reference_layers[1].decode_forward(
        hidden_states=functional_tests._as_tt(reference_mesh, decode_hidden.unsqueeze(1)),
        **same_input_reference_call["decode"],
    )
    same_input_multichip_decode = multichip_layers[1].decode_forward(
        hidden_states=functional_tests._as_tt(tp_mesh, decode_hidden.unsqueeze(1)),
        **same_input_multichip_call["decode"],
    )
    ttnn.synchronize_device(reference_mesh)
    ttnn.synchronize_device(tp_mesh)
    same_input_reference_layer5_hosts = [
        ttnn.to_torch(ttnn.get_device_tensors(output.cpu())[0])
        for output in (same_input_reference_prefill, same_input_reference_decode)
    ]
    same_input_multichip_layer5_hosts = [
        ttnn.to_torch(ttnn.get_device_tensors(output.cpu())[0])
        for output in (same_input_multichip_prefill, same_input_multichip_decode)
    ]

    routing_captures = {"reference": {}, "multichip": {}}
    original_routing_methods = []
    for path, decoders in (("reference", reference_layers), ("multichip", multichip_layers)):
        for layer_idx, decoder in zip(layer_indices, decoders):
            original = decoder._router_weights_from_normalized

            def capture_routing(router_in, *, original=original, path=path, layer_idx=layer_idx):
                routing = original(router_in)
                routing_captures[path][layer_idx] = routing
                return routing

            decoder._router_weights_from_normalized = capture_routing
            original_routing_methods.append((decoder, original))

    reference_decode_layers = run_chain(
        reference_layers,
        functional_tests._as_tt(reference_mesh, decode_hidden.unsqueeze(1)),
        reference_calls,
        "decode",
    )
    multichip_decode_layers = run_chain(
        multichip_layers,
        functional_tests._as_tt(tp_mesh, decode_hidden.unsqueeze(1)),
        multichip_calls,
        "decode",
    )
    ttnn.synchronize_device(reference_mesh)
    ttnn.synchronize_device(tp_mesh)
    reference_decode_hosts = [
        ttnn.to_torch(ttnn.get_device_tensors(output.cpu())[0]) for output in reference_decode_layers
    ]
    multichip_decode_shards_by_layer = [
        [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(output.cpu())]
        for output in multichip_decode_layers
    ]
    reference_routing_hosts = {
        layer_idx: ttnn.to_torch(ttnn.get_device_tensors(routing.cpu())[0])
        for layer_idx, routing in routing_captures["reference"].items()
    }
    multichip_routing_hosts = {
        layer_idx: ttnn.to_torch(ttnn.get_device_tensors(routing.cpu())[0])
        for layer_idx, routing in routing_captures["multichip"].items()
    }
    for decoder, original in original_routing_methods:
        decoder._router_weights_from_normalized = original

    trace_input = functional_tests._as_tt(tp_mesh, decode_hidden.unsqueeze(1))
    trace_id = ttnn.begin_trace_capture(tp_mesh, cq_id=0)
    traced_decode = run_chain(multichip_layers, trace_input, multichip_calls, "decode")[-1]
    ttnn.end_trace_capture(tp_mesh, trace_id, cq_id=0)
    stress_replays = 20
    replay_shards = []
    try:
        for _ in range(stress_replays):
            ttnn.execute_trace(tp_mesh, trace_id, cq_id=0, blocking=True)
            replay_shards.append([ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(traced_decode.cpu())])
    finally:
        ttnn.release_trace(tp_mesh, trace_id)

    # Require >=0.99 for both layer kinds when their inputs match. The final
    # chained decode remains a separately labelled discontinuity stress: its
    # layer-5 inputs differ because TP and baseline layer 0 are approximate,
    # and that perturbation changes one gate-selected expert.
    same_input_threshold = 0.99
    chained_prefill_threshold = 0.99
    chained_decode_thresholds = (same_input_threshold, 0.98)
    same_input_prefill_results = [
        functional_tests.comp_pcc(
            reference_prefill_hosts[0], multichip_prefill_shards_by_layer[0][0], same_input_threshold
        ),
        functional_tests.comp_pcc(
            same_input_reference_layer5_hosts[0], same_input_multichip_layer5_hosts[0], same_input_threshold
        ),
    ]
    same_input_eager_results = [
        functional_tests.comp_pcc(
            reference_decode_hosts[0], multichip_decode_shards_by_layer[0][0], same_input_threshold
        ),
        functional_tests.comp_pcc(
            same_input_reference_layer5_hosts[1], same_input_multichip_layer5_hosts[1], same_input_threshold
        ),
    ]
    chained_prefill_results = [
        functional_tests.comp_pcc(reference, multichip[0], chained_prefill_threshold)
        for reference, multichip in zip(reference_prefill_hosts, multichip_prefill_shards_by_layer)
    ]
    chained_eager_results = [
        functional_tests.comp_pcc(reference, multichip[0], threshold)
        for reference, multichip, threshold in zip(
            reference_decode_hosts,
            multichip_decode_shards_by_layer,
            chained_decode_thresholds,
        )
    ]
    trace_ok, trace_pcc = functional_tests.comp_pcc(
        reference_decode_hosts[-1], replay_shards[-1][0], chained_decode_thresholds[-1]
    )
    active_expert_indices = {
        path: {
            str(layer_idx): sorted(torch.topk(routing.abs().reshape(-1), k=8).indices.tolist())
            for layer_idx, routing in hosts.items()
        }
        for path, hosts in (("reference", reference_routing_hosts), ("multichip", multichip_routing_hosts))
    }
    active_expert_agreement = {
        str(layer_idx): len(
            set(active_expert_indices["reference"][str(layer_idx)])
            & set(active_expert_indices["multichip"][str(layer_idx)])
        )
        for layer_idx in layer_indices
    }
    replicas_bit_exact = all(
        torch.equal(rank_zero, rank_one)
        for shards in (
            *multichip_prefill_shards_by_layer,
            *multichip_decode_shards_by_layer,
            *replay_shards,
        )
        for rank_zero, rank_one in [shards]
    )
    eager_trace_bit_exact = torch.equal(multichip_decode_shards_by_layer[-1][0], replay_shards[0][0])
    repeat_bit_exact = all(torch.equal(replay_shards[0][0], replay[0]) for replay in replay_shards[1:])
    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    suffix = _measurement_artifact_suffix()
    (artifact_dir / f"stacked_tp2_mixed_trace{suffix}.json").write_text(
        json.dumps(
            {
                "tp_size": 2,
                "layer_indices": layer_indices,
                "layer_types": [cfg.layer_types[layer_idx] for layer_idx in layer_indices],
                "reference_policy": reference_policy,
                "sequence_length": seq_len,
                "shared_persistent_resource_slots": len(shared_resources["buffers"]),
                "persistent_resource_index_after_capture": shared_resources["index"],
                "stress_trace_replays": stress_replays,
                "same_input_selected_policy_threshold": same_input_threshold,
                "same_input_prefill_pcc_by_layer": [float(result[1]) for result in same_input_prefill_results],
                "same_input_eager_decode_pcc_by_layer": [float(result[1]) for result in same_input_eager_results],
                "chained_prefill_pcc_by_layer": [float(result[1]) for result in chained_prefill_results],
                "chained_prefill_threshold": chained_prefill_threshold,
                "chained_eager_decode_pcc_by_layer": [float(result[1]) for result in chained_eager_results],
                "chained_decode_threshold_by_layer": chained_decode_thresholds,
                "chained_trace_decode_pcc": float(trace_pcc),
                "chained_trace_decode_threshold": chained_decode_thresholds[-1],
                "chained_inputs_identical_by_layer": [True, False],
                "chained_discontinuity": "layer-5 inputs differ after approximate layer-0 outputs",
                "active_expert_indices": active_expert_indices,
                "active_expert_agreement_out_of_8": active_expert_agreement,
                "eager_vs_trace_bit_exact": eager_trace_bit_exact,
                "trace_replays_bit_exact": repeat_bit_exact,
                "replicas_bit_exact": replicas_bit_exact,
                "local_cache_shapes": [call["cache_shape"] for call in multichip_calls],
                "all_reduce_dispatches_per_layer": [
                    decoder.multichip_path_counters["all_reduce"] for decoder in multichip_layers
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert all(
        result[0] for result in same_input_prefill_results
    ), f"TP2 same-input prefill PCCs={same_input_prefill_results}"
    assert all(
        result[0] for result in same_input_eager_results
    ), f"TP2 same-input eager decode PCCs={same_input_eager_results}"
    assert all(result[0] for result in chained_prefill_results), f"TP2 stacked prefill PCCs={chained_prefill_results}"
    assert all(result[0] for result in chained_eager_results), f"TP2 stacked eager decode PCCs={chained_eager_results}"
    assert trace_ok, f"TP2 stacked trace decode PCC={trace_pcc}"
    assert eager_trace_bit_exact
    assert repeat_bit_exact
    assert replicas_bit_exact
    assert all(decoder.multichip_path_counters["all_reduce"] >= 9 for decoder in multichip_layers)


def _tp4_stacked_inputs_and_rope(cfg):
    """Create the deterministic inputs shared by the separate TP1/TP4 runs."""
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    layer_indices = (0, 5)
    seq_len = 32
    torch.manual_seed(4604)
    prefill_hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    positions = torch.arange(seq_len).unsqueeze(0)
    decode_positions = torch.tensor([[seq_len]])
    host_rope = {}
    for layer_idx in layer_indices:
        layer_type = cfg.layer_types[layer_idx]
        host_rope[layer_idx] = (
            *rotary(prefill_hidden, positions, layer_type=layer_type),
            *rotary(decode_hidden, decode_positions, layer_type=layer_type),
        )
    return layer_indices, seq_len, prefill_hidden, decode_hidden, host_rope


def _tp4_stacked_layer_calls(target_mesh, decoders, layer_indices, seq_len, host_rope, *, local_cache):
    """Build per-layer cache/control tensors with replicated control metadata."""
    import torch

    calls = []
    for layer_idx, decoder in zip(layer_indices, decoders):
        kind = SLIDING_KIND if decoder.layer_kind is SLIDING_KIND else FULL_KIND
        blocks = 4 if kind is SLIDING_KIND else 2
        heads = (
            (decoder.local_sliding_kv_heads if kind is SLIDING_KIND else decoder.local_full_kv_heads)
            if local_cache
            else kind.num_kv_heads
        )
        page_table = functional_tests._as_tt(
            target_mesh,
            torch.arange(blocks, dtype=torch.int32).view(1, blocks),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        current_pos = functional_tests._as_tt(
            target_mesh,
            torch.tensor([seq_len], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cache_shape = (blocks, heads, kind.block_size, kind.head_dim)
        kv_cache = tuple(
            functional_tests._as_tt(target_mesh, torch.zeros(cache_shape, dtype=torch.bfloat16)) for _ in range(2)
        )
        prefill_cos, prefill_sin, decode_cos, decode_sin = host_rope[layer_idx]
        calls.append(
            {
                "prefill": {
                    "position_cos": functional_tests._as_tt(target_mesh, prefill_cos.unsqueeze(1)),
                    "position_sin": functional_tests._as_tt(target_mesh, prefill_sin.unsqueeze(1)),
                    "page_table": page_table,
                    "kv_cache": kv_cache,
                },
                "decode": {
                    "position_cos": functional_tests._as_tt(target_mesh, decode_cos.unsqueeze(1)),
                    "position_sin": functional_tests._as_tt(target_mesh, decode_sin.unsqueeze(1)),
                    "current_pos": current_pos,
                    "page_table": page_table,
                    "kv_cache": kv_cache,
                },
                "cache_shape": cache_shape,
            }
        )
    return calls


def _run_tp4_stacked_chain(decoders, hidden_states, calls, phase):
    output = hidden_states
    layer_outputs = []
    for decoder, layer_call in zip(decoders, calls):
        if phase == "prefill":
            output = decoder.prefill_forward(output, **layer_call[phase])
        else:
            output = decoder.decode_forward(hidden_states=output, **layer_call[phase])
        layer_outputs.append(output)
    return layer_outputs


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 128 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_tp4_fractured_complete_sliding_layer(mesh_device, device_params, monkeypatch):
    """Compare one complete fractured sliding layer with the incumbent TP4 path."""

    if os.getenv("GEMMA4_MULTICHIP_FRACTURED_LAYER_REPRO") != "1":
        pytest.skip("set GEMMA4_MULTICHIP_FRACTURED_LAYER_REPRO=1 for the serialized whole-layer repro")

    import hashlib
    import math
    import statistics

    import torch

    from ttnn import ReplicateTensorToMesh, ShardTensorToMesh

    assert ttnn.CONFIG.throw_exception_on_fallback
    source_path = Path("models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert source_sha256 == "d769d045af22574934b49a5bfab10c9ec84816e7ce53b825b36d8c7b63776f42"
    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    monkeypatch.setenv("GEMMA4_MULTICHIP_FRACTURED_RESIDUAL_CANDIDATE", "1")
    cfg = functional_tests._load_text_config()
    _, seq_len, _, decode_hidden, host_rope = _tp4_stacked_inputs_and_rope(cfg)
    state = functional_tests._load_layer_state(0)
    decoder = MultichipDecoder.from_state_dict(state, hf_config=cfg, layer_idx=0, mesh_device=mesh_device)
    assert decoder.layer_kind is SLIDING_KIND
    baseline_call = _tp4_stacked_layer_calls(
        mesh_device,
        [decoder],
        (0,),
        seq_len,
        host_rope,
        local_cache=True,
    )[
        0
    ]["decode"]
    candidate_call = _tp4_stacked_layer_calls(
        mesh_device,
        [decoder],
        (0,),
        seq_len,
        host_rope,
        local_cache=True,
    )[
        0
    ]["decode"]
    baseline_input = ttnn.from_torch(
        decode_hidden.unsqueeze(1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    candidate_input = ttnn.from_torch(
        decode_hidden.unsqueeze(1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
    )

    # Compile both persistent paths before either capture.
    decoder.decode_forward(hidden_states=baseline_input, **baseline_call)
    decoder.decode_forward_fractured(candidate_input, **candidate_call)
    decoder.decode_forward_fractured(candidate_input, **candidate_call)
    ttnn.synchronize_device(mesh_device)
    resources = decoder.fractured_layer_resources
    assert resources is not None
    assert set(resources["reduce_buffers"]) == {
        "qkv",
        "o_proj",
        "dense_packed",
        "dense_down",
        "expert_packed",
        "expert_down",
    }
    assert set(resources["norm_outputs"]) == {
        "input",
        "post_attention",
        "shared_ffn",
        "post_ff_1",
        "post_ff_2",
        "post_ff",
    }

    warmups = int(os.getenv("GEMMA4_FRACTURED_LAYER_WARMUPS", "5"))
    iterations = int(os.getenv("GEMMA4_FRACTURED_LAYER_ITERATIONS", "30"))
    stress_replays = int(os.getenv("GEMMA4_FRACTURED_LAYER_STRESS_REPLAYS", "20"))
    if warmups < 5 or iterations < 30 or stress_replays < 20:
        raise ValueError("Gate D requires at least 5 warmups, 30 timed replays, and 20 stress replays")

    def run_timed_trace(forward):
        # Preallocate host timing storage before trace capture.  While the
        # trace exists, only blocking replay and host-clock sampling occur.
        samples_ms = [0.0] * iterations
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output = forward()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        try:
            for _ in range(warmups):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            for sample_idx in range(iterations):
                started = time.perf_counter()
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                samples_ms[sample_idx] = (time.perf_counter() - started) * 1000
        finally:
            ttnn.release_trace(mesh_device, trace_id)
        return output, samples_ms

    def timing_summary(samples_ms):
        ordered = sorted(samples_ms)
        return {
            "mean_ms": statistics.fmean(samples_ms),
            "median_ms": statistics.median(samples_ms),
            "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "min_ms": ordered[0],
            "max_ms": ordered[-1],
            "samples_ms": samples_ms,
        }

    # Run each path under an identical, non-overlapping trace lifecycle.  A
    # trace is always released before any device-to-host readback or before
    # the other path starts allocating its trace output tensors.
    baseline_output, baseline_samples_ms = run_timed_trace(
        lambda: decoder.decode_forward(hidden_states=baseline_input, **baseline_call)
    )
    baseline_first = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(baseline_output.cpu())]

    candidate_cache_before = mesh_device.num_program_cache_entries()
    candidate_output, candidate_samples_ms = run_timed_trace(
        lambda: decoder.decode_forward_fractured(candidate_input, **candidate_call)
    )
    candidate_first = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(candidate_output.cpu())]

    # Determinism uses a fresh trace.  Its already-allocated output is replayed
    # twenty times, the trace is released, and only then is the result read.
    stress_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    candidate_stress_output = decoder.decode_forward_fractured(candidate_input, **candidate_call)
    ttnn.end_trace_capture(mesh_device, stress_trace_id, cq_id=0)
    try:
        for _ in range(stress_replays):
            ttnn.execute_trace(mesh_device, stress_trace_id, cq_id=0, blocking=True)
    finally:
        ttnn.release_trace(mesh_device, stress_trace_id)
    candidate_repeat = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(candidate_stress_output.cpu())]
    candidate_cache_after = mesh_device.num_program_cache_entries()

    baseline_host = baseline_first[0]
    assert all(torch.equal(baseline_host, shard) for shard in baseline_first[1:])
    candidate_host = torch.cat(candidate_first, dim=3)
    candidate_ok, candidate_pcc = functional_tests.comp_pcc(
        baseline_host,
        candidate_host,
        0.99,
    )
    repeat_bit_exact = all(torch.equal(first, repeat) for first, repeat in zip(candidate_first, candidate_repeat))
    baseline_timing = timing_summary(baseline_samples_ms)
    candidate_timing = timing_summary(candidate_samples_ms)
    speedup = baseline_timing["median_ms"] / candidate_timing["median_ms"]
    p95_speedup = baseline_timing["p95_ms"] / candidate_timing["p95_ms"]
    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/fractured_sparse")
    artifact_suffix = _measurement_artifact_suffix()
    artifact = {
        "gate": "D_complete_sliding_layer",
        "status": "accepted" if candidate_ok and repeat_bit_exact and speedup > 1.0 else "rejected",
        "real_weights": True,
        "paged_kv_cache": True,
        "current_position": seq_len,
        "logical_batch": 1,
        "candidate_local_shapes": [list(shard.shape) for shard in ttnn.get_device_tensors(candidate_output)],
        "candidate_vs_incumbent_pcc": float(candidate_pcc),
        "candidate_repeat_bit_exact": repeat_bit_exact,
        "stress_replays": stress_replays,
        "program_cache_entries_before": candidate_cache_before,
        "program_cache_entries_after": candidate_cache_after,
        "warmups": warmups,
        "iterations": iterations,
        "timing_method": "individual host wall-clock samples around blocking trace replay",
        "trace_lifecycle": "sequential capture/replay/release; all readback occurs after release",
        "incumbent_trace_ms": baseline_timing["median_ms"],
        "candidate_trace_ms": candidate_timing["median_ms"],
        "incumbent_timing": baseline_timing,
        "candidate_timing": candidate_timing,
        "speedup": speedup,
        "p95_speedup": p95_speedup,
        "collective_counts": {
            "reduce_scatter": len(resources["reduce_buffers"]),
            "stats_all_gather": len(resources["norm_outputs"]),
            "router_all_reduce": 1,
        },
        "fallback_policy": "TTNN throw_exception_on_fallback=true",
        "source_sha256_at_run": source_sha256,
        "test_sha256_at_run": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (artifact_dir / f"gate_d_complete_sliding_layer{artifact_suffix}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    assert candidate_ok, f"complete fractured layer vs incumbent PCC={candidate_pcc}"
    assert repeat_bit_exact
    assert candidate_cache_before == candidate_cache_after


@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_capture_optimized_tp4_stacked_mixed_reference(mesh_device, device_params, monkeypatch):
    """Capture the exact optimized TP1 oracle used by the 1x4 stacked gate."""
    import gzip

    import torch

    if os.getenv("GEMMA4_CAPTURE_TP4_STACKED_REFERENCE") != "1":
        pytest.skip("reference capture is an explicit serialized hardware evidence step")
    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    cfg = functional_tests._load_text_config()
    layer_indices, seq_len, prefill_hidden, decode_hidden, host_rope = _tp4_stacked_inputs_and_rope(cfg)
    states = {layer_idx: functional_tests._load_layer_state(layer_idx) for layer_idx in layer_indices}
    decoders = [
        OptimizedDecoder.from_state_dict(states[layer_idx], hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
        for layer_idx in layer_indices
    ]
    calls = _tp4_stacked_layer_calls(mesh_device, decoders, layer_indices, seq_len, host_rope, local_cache=False)
    prefill_layers = _run_tp4_stacked_chain(
        decoders,
        functional_tests._as_tt(mesh_device, prefill_hidden.unsqueeze(1)),
        calls,
        "prefill",
    )
    same_input_call = _tp4_stacked_layer_calls(
        mesh_device, decoders, layer_indices, seq_len, host_rope, local_cache=False
    )[1]
    same_input_prefill = decoders[1].prefill_forward(
        functional_tests._as_tt(mesh_device, prefill_hidden.unsqueeze(1)),
        **same_input_call["prefill"],
    )
    same_input_decode = decoders[1].decode_forward(
        hidden_states=functional_tests._as_tt(mesh_device, decode_hidden.unsqueeze(1)),
        **same_input_call["decode"],
    )
    decode_layers = _run_tp4_stacked_chain(
        decoders,
        functional_tests._as_tt(mesh_device, decode_hidden.unsqueeze(1)),
        calls,
        "decode",
    )
    ttnn.synchronize_device(mesh_device)

    def first_host(tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor.cpu())[0])

    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(artifact_dir / "stacked_tp4_optimized_reference.pt.gz", "wb") as output_file:
        torch.save(
            {
                "layer_indices": layer_indices,
                "layer_types": [cfg.layer_types[layer_idx] for layer_idx in layer_indices],
                "sequence_length": seq_len,
                "prefill_by_layer": [first_host(output) for output in prefill_layers],
                "decode_by_layer": [first_host(output) for output in decode_layers],
                "same_input_layer5_prefill": first_host(same_input_prefill),
                "same_input_layer5_decode": first_host(same_input_decode),
                "policy_by_layer": [
                    {
                        "residual_shard_cores": decoder.residual_shard_cores,
                        "folded_router_projection": decoder.folded_router_projection,
                        "shared_ffn_norm": decoder.shared_ffn_norm,
                        "folded_expert_scale": decoder.folded_expert_scale,
                        "fused_final_scalar": decoder.fused_final_scalar,
                        "packed_expert_decode_gate_up": decoder.packed_expert_decode_gate_up,
                        "routing_row_major": decoder.routing_row_major,
                    }
                    for decoder in decoders
                ],
            },
            output_file,
        )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 128 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_tp4_stacked_mixed_attention_shared_persistent_ccl_trace(mesh_device, device_params, monkeypatch):
    """Validate the selected R22-sliding -> R0-full TP4 stack boundary."""
    import gzip
    import hashlib

    import torch

    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    monkeypatch.setenv("GEMMA4_MULTICHIP_PERSISTENT_ALL_REDUCE", "1")
    cfg = functional_tests._load_text_config()
    layer_indices, seq_len, prefill_hidden, decode_hidden, host_rope = _tp4_stacked_inputs_and_rope(cfg)
    states = {layer_idx: functional_tests._load_layer_state(layer_idx) for layer_idx in layer_indices}
    decoders = [MultichipDecoder.from_state_dict(states[0], hf_config=cfg, layer_idx=0, mesh_device=mesh_device)]
    shared_resources = decoders[0].persistent_all_reduce_resources
    assert shared_resources is not None
    decoders.append(
        MultichipDecoder.from_state_dict(
            states[5],
            hf_config=cfg,
            layer_idx=5,
            mesh_device=mesh_device,
            persistent_all_reduce_resources=shared_resources,
        )
    )
    assert len(shared_resources["buffers"]) == len(shared_resources["semaphores"]) == 3
    for buffer in shared_resources["buffers"]:
        assert tuple(buffer.shape) == (1, 1, 32, HIDDEN_SIZE * 4)
        assert buffer.dtype == ttnn.bfloat16
        assert buffer.layout == ttnn.TILE_LAYOUT
    for decoder in decoders:
        assert decoder.tp_size == 4
        assert decoder.persistent_all_reduce_resources is shared_resources
        assert decoder.persistent_all_reduce_buffers is shared_resources["buffers"]
        assert decoder.persistent_all_reduce_semaphores is shared_resources["semaphores"]
        assert decoder.topology == ttnn.Topology.Ring
    selected_policies = [
        {
            "residual_shard_cores": decoder.residual_shard_cores,
            "folded_router_projection": decoder.folded_router_projection,
            "shared_ffn_norm": decoder.shared_ffn_norm,
            "folded_expert_scale": decoder.folded_expert_scale,
            "fused_final_scalar": decoder.fused_final_scalar,
            "packed_expert_decode_gate_up": decoder.packed_expert_decode_gate_up,
            "routing_row_major": decoder.routing_row_major,
            "attention_weight_dtype": str(decoder.weights.qkv.dtype),
        }
        for decoder in decoders
    ]
    assert selected_policies[0] == {
        "residual_shard_cores": 22,
        "folded_router_projection": True,
        "shared_ffn_norm": True,
        "folded_expert_scale": True,
        "fused_final_scalar": True,
        "packed_expert_decode_gate_up": True,
        "routing_row_major": True,
        "attention_weight_dtype": str(ttnn.bfloat16),
    }
    assert selected_policies[1] == {
        "residual_shard_cores": 0,
        "folded_router_projection": False,
        "shared_ffn_norm": False,
        "folded_expert_scale": True,
        "fused_final_scalar": True,
        "packed_expert_decode_gate_up": True,
        "routing_row_major": False,
        "attention_weight_dtype": str(ttnn.bfloat8_b),
    }

    calls = _tp4_stacked_layer_calls(mesh_device, decoders, layer_indices, seq_len, host_rope, local_cache=True)
    prefill_input = functional_tests._as_tt(mesh_device, prefill_hidden.unsqueeze(1))
    decode_input = functional_tests._as_tt(mesh_device, decode_hidden.unsqueeze(1))

    def device_hosts(tensor):
        return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor.cpu())]

    def replicas_are_exact(tensor):
        shards = device_hosts(tensor)
        return len(shards) == 4 and all(torch.equal(shards[0], shard) for shard in shards[1:])

    def is_dram_interleaved(tensor):
        memory_config = tensor.memory_config()
        return (
            tensor.layout == ttnn.TILE_LAYOUT
            and memory_config.buffer_type == ttnn.BufferType.DRAM
            and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        )

    assert replicas_are_exact(prefill_input)
    assert replicas_are_exact(decode_input)
    assert is_dram_interleaved(prefill_input)
    assert is_dram_interleaved(decode_input)

    resource_index_history = {"initial": shared_resources["index"]}
    assert resource_index_history["initial"] == 0
    prefill_layers = _run_tp4_stacked_chain(decoders, prefill_input, calls, "prefill")
    resource_index_history["after_prefill_chain"] = shared_resources["index"]
    prefill_shards_by_layer = [device_hosts(output) for output in prefill_layers]
    assert all(is_dram_interleaved(output) for output in prefill_layers)

    # The full-attention layer gets an independent same-input cache so its
    # strict PCC gate cannot be weakened by the approximate layer-0 output.
    same_input_call = _tp4_stacked_layer_calls(
        mesh_device, decoders, layer_indices, seq_len, host_rope, local_cache=True
    )[1]
    same_input_prefill = decoders[1].prefill_forward(
        functional_tests._as_tt(mesh_device, prefill_hidden.unsqueeze(1)),
        **same_input_call["prefill"],
    )
    same_input_decode = decoders[1].decode_forward(
        hidden_states=functional_tests._as_tt(mesh_device, decode_hidden.unsqueeze(1)),
        **same_input_call["decode"],
    )
    resource_index_history["after_same_input_layer5_prefill_decode"] = shared_resources["index"]
    same_input_layer5_shards = [device_hosts(same_input_prefill), device_hosts(same_input_decode)]

    # Observe one controlled eager chain.  Three rotating slots must be used
    # by layer 0 and then reused by layer 5, proving actual shared-resource
    # reuse instead of object identity alone.
    assert shared_resources["index"] == 0
    resource_slot_sequence = []
    original_all_reduce_methods = []
    for layer_idx, decoder in zip(layer_indices, decoders):
        original = decoder._all_reduce_hidden

        def capture_slot(partial, *, original=original, layer_idx=layer_idx):
            resource_slot_sequence.append((layer_idx, shared_resources["index"]))
            return original(partial)

        decoder._all_reduce_hidden = capture_slot
        original_all_reduce_methods.append((decoder, original))
    try:
        decode_layers = _run_tp4_stacked_chain(decoders, decode_input, calls, "decode")
        ttnn.synchronize_device(mesh_device)
    finally:
        for decoder, original in original_all_reduce_methods:
            decoder._all_reduce_hidden = original
    assert resource_slot_sequence == [(0, 0), (0, 1), (0, 2), (5, 0), (5, 1), (5, 2)]
    assert shared_resources["index"] == 0
    resource_index_history["after_eager_chain"] = shared_resources["index"]
    decode_shards_by_layer = [device_hosts(output) for output in decode_layers]
    assert all(is_dram_interleaved(output) for output in decode_layers)

    trace_input = functional_tests._as_tt(mesh_device, decode_hidden.unsqueeze(1))
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_layers = _run_tp4_stacked_chain(decoders, trace_input, calls, "decode")
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    resource_index_history["after_trace_capture"] = shared_resources["index"]
    path_counters_after_capture = [dict(decoder.multichip_path_counters) for decoder in decoders]
    stress_replays = 20
    replay_shards_by_layer = []
    try:
        for _ in range(stress_replays):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            replay_shards_by_layer.append([device_hosts(output) for output in traced_layers])
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    resource_index_history["after_trace_replays"] = shared_resources["index"]
    path_counters_after_replays = [dict(decoder.multichip_path_counters) for decoder in decoders]
    assert path_counters_after_replays == path_counters_after_capture
    assert all(index == 0 for index in resource_index_history.values())

    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    reference_path = artifact_dir / "stacked_tp4_optimized_reference.pt.gz"
    assert reference_path.is_file(), "run test_capture_optimized_tp4_stacked_mixed_reference first"
    with gzip.open(reference_path, "rb") as input_file:
        reference = torch.load(input_file, weights_only=True)
    assert tuple(reference["layer_indices"]) == layer_indices
    assert reference["layer_types"] == [cfg.layer_types[layer_idx] for layer_idx in layer_indices]
    assert reference["sequence_length"] == seq_len

    same_input_threshold = 0.99
    chained_prefill_threshold = 0.99
    chained_decode_thresholds = (0.99, 0.98)
    same_input_prefill_results = [
        functional_tests.comp_pcc(reference["prefill_by_layer"][0], prefill_shards_by_layer[0][0], 0.99),
        functional_tests.comp_pcc(
            reference["same_input_layer5_prefill"], same_input_layer5_shards[0][0], same_input_threshold
        ),
    ]
    same_input_decode_results = [
        functional_tests.comp_pcc(reference["decode_by_layer"][0], decode_shards_by_layer[0][0], 0.99),
        functional_tests.comp_pcc(
            reference["same_input_layer5_decode"], same_input_layer5_shards[1][0], same_input_threshold
        ),
    ]
    chained_prefill_results = [
        functional_tests.comp_pcc(reference_output, shards[0], chained_prefill_threshold)
        for reference_output, shards in zip(reference["prefill_by_layer"], prefill_shards_by_layer)
    ]
    chained_decode_results = [
        functional_tests.comp_pcc(reference_output, shards[0], threshold)
        for reference_output, shards, threshold in zip(
            reference["decode_by_layer"], decode_shards_by_layer, chained_decode_thresholds
        )
    ]
    trace_ok, trace_pcc = functional_tests.comp_pcc(
        reference["decode_by_layer"][-1], replay_shards_by_layer[-1][-1][0], chained_decode_thresholds[-1]
    )

    all_output_groups = (
        *prefill_shards_by_layer,
        *decode_shards_by_layer,
        *same_input_layer5_shards,
        *(shards for replay in replay_shards_by_layer for shards in replay),
    )
    replicas_bit_exact = all(
        len(shards) == 4 and all(torch.equal(shards[0], shard) for shard in shards[1:]) for shards in all_output_groups
    )
    eager_trace_bit_exact = all(
        torch.equal(eager, replay)
        for eager_shards, replay_shards in zip(decode_shards_by_layer, replay_shards_by_layer[0])
        for eager, replay in zip(eager_shards, replay_shards)
    )
    trace_replays_bit_exact = all(
        all(
            torch.equal(first, current)
            for first_shards, current_shards in zip(replay_shards_by_layer[0], replay)
            for first, current in zip(first_shards, current_shards)
        )
        for replay in replay_shards_by_layer[1:]
    )

    control_replicated = True
    cache_contract = []
    for layer_idx, decoder, call in zip(layer_indices, decoders, calls):
        expected_page_table = torch.arange(call["cache_shape"][0], dtype=torch.int32).view(1, -1)
        expected_current_pos = torch.tensor([seq_len], dtype=torch.int32)
        page_table_shards = device_hosts(call["decode"]["page_table"])
        current_pos_shards = device_hosts(call["decode"]["current_pos"])
        assert call["decode"]["page_table"].dtype == ttnn.int32
        assert call["decode"]["page_table"].layout == ttnn.ROW_MAJOR_LAYOUT
        assert call["decode"]["current_pos"].dtype == ttnn.int32
        assert call["decode"]["current_pos"].layout == ttnn.ROW_MAJOR_LAYOUT
        page_table_replicated = all(torch.equal(expected_page_table, shard) for shard in page_table_shards)
        current_pos_replicated = all(torch.equal(expected_current_pos, shard) for shard in current_pos_shards)
        control_replicated &= page_table_replicated and current_pos_replicated
        cache_pair_evidence = []
        for cache_tensor in call["decode"]["kv_cache"]:
            assert cache_tensor.dtype == ttnn.bfloat16
            assert cache_tensor.layout == ttnn.TILE_LAYOUT
            assert cache_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
            assert cache_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
            cache_shards = device_hosts(cache_tensor)
            assert all(tuple(shard.shape) == call["cache_shape"] for shard in cache_shards)
            assert all(torch.count_nonzero(shard).item() > 0 for shard in cache_shards)
            assert all(torch.count_nonzero(shard[1:]).item() == 0 for shard in cache_shards)
            if layer_idx == 5:
                assert torch.equal(cache_shards[0], cache_shards[1])
                assert torch.equal(cache_shards[2], cache_shards[3])
                assert not torch.equal(cache_shards[0], cache_shards[2])
            else:
                assert any(not torch.equal(cache_shards[0], shard) for shard in cache_shards[1:])
            cache_pair_evidence.append(
                {
                    "rank_0_1_bit_exact": torch.equal(cache_shards[0], cache_shards[1]),
                    "rank_2_3_bit_exact": torch.equal(cache_shards[2], cache_shards[3]),
                }
            )
        cache_contract.append(
            {
                "layer_idx": layer_idx,
                "layer_type": cfg.layer_types[layer_idx],
                "local_cache_shape": call["cache_shape"],
                "local_kv_heads": (decoder.local_sliding_kv_heads if layer_idx == 0 else decoder.local_full_kv_heads),
                "page_table_shape": tuple(page_table_shards[0].shape),
                "page_table_replicated": page_table_replicated,
                "current_position_shape": tuple(current_pos_shards[0].shape),
                "current_position_replicated": current_pos_replicated,
                "full_kv_rank_pair_duplication": cache_pair_evidence if layer_idx == 5 else None,
            }
        )

    source_path = Path("models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py")
    test_path = Path("models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py")
    exact_command = (
        "GEMMA4_RANGE_DOWNLOAD=1 TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\":true}' "
        "timeout 1800 python_env/bin/python -m pytest -q -s "
        "models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py "
        "-k test_tp4_stacked_mixed_attention_shared_persistent_ccl_trace "
        "--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/"
        "stacked_tp4_mixed_trace.xml"
    )
    artifact = {
        "tp_size": 4,
        "mesh_shape": [1, 4],
        "layer_indices": layer_indices,
        "layer_types": [cfg.layer_types[layer_idx] for layer_idx in layer_indices],
        "sequence_length": seq_len,
        "optimized_reference_artifact": reference_path.name,
        "optimized_reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "exact_command": exact_command,
        "fallback_policy": "TTNN throw_exception_on_fallback=true",
        "selected_policy_by_layer": selected_policies,
        "public_input_layout": "replicated TILE DRAM INTERLEAVED",
        "public_output_layout": "replicated TILE DRAM INTERLEAVED",
        "same_input_selected_policy_threshold": same_input_threshold,
        "same_input_prefill_pcc_by_layer": [float(result[1]) for result in same_input_prefill_results],
        "same_input_eager_decode_pcc_by_layer": [float(result[1]) for result in same_input_decode_results],
        "chained_prefill_threshold": chained_prefill_threshold,
        "chained_prefill_pcc_by_layer": [float(result[1]) for result in chained_prefill_results],
        "chained_decode_threshold_by_layer": chained_decode_thresholds,
        "chained_eager_decode_pcc_by_layer": [float(result[1]) for result in chained_decode_results],
        "chained_trace_decode_threshold": chained_decode_thresholds[-1],
        "chained_trace_decode_pcc": float(trace_pcc),
        "chained_inputs_identical_by_layer": [True, False],
        "chained_discontinuity": "layer-5 inputs differ after approximate TP4 layer-0 output",
        "shared_persistent_resource_slots": len(shared_resources["buffers"]),
        "shared_persistent_buffer_shape": [1, 1, 32, HIDDEN_SIZE * 4],
        "shared_resource_slot_sequence": resource_slot_sequence,
        "persistent_resource_index_history": resource_index_history,
        "persistent_resource_index_after_trace_capture": shared_resources["index"],
        "path_counters_after_capture": path_counters_after_capture,
        "path_counters_unchanged_by_replay": path_counters_after_replays == path_counters_after_capture,
        "all_reduce_dispatches_per_layer": [counters["all_reduce"] for counters in path_counters_after_capture],
        "cache_contract": cache_contract,
        "page_table_and_current_position_replicated": control_replicated,
        "stress_trace_replays": stress_replays,
        "eager_vs_trace_bit_exact_all_ranks": eager_trace_bit_exact,
        "trace_replays_bit_exact_all_ranks": trace_replays_bit_exact,
        "replicas_bit_exact": replicas_bit_exact,
    }
    artifact_path = artifact_dir / "stacked_tp4_mixed_trace.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    assert all(result[0] for result in same_input_prefill_results), same_input_prefill_results
    assert all(result[0] for result in same_input_decode_results), same_input_decode_results
    assert all(result[0] for result in chained_prefill_results), chained_prefill_results
    assert all(result[0] for result in chained_decode_results), chained_decode_results
    assert trace_ok, f"TP4 stacked trace decode PCC={trace_pcc}"
    assert control_replicated
    assert replicas_bit_exact
    assert eager_trace_bit_exact
    assert trace_replays_bit_exact
    assert [counters["all_reduce"] for counters in path_counters_after_capture] == [9, 15]
    assert [counters["attention_tp"] for counters in path_counters_after_capture] == [3, 5]
    assert [counters["dense_tp"] for counters in path_counters_after_capture] == [3, 5]
    assert [counters["expert_tp"] for counters in path_counters_after_capture] == [3, 5]

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Correctness and performance gates for the optimized decoder path."""

import gc
import inspect
import json
import os
import time
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from tracy import signpost
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.openai_gpt_oss_120b.tests import test_functional_decoder as accepted
from models.autoports.openai_gpt_oss_120b.tt.optimized_decoder import (
    _BFP4_LOFI_DRAM_SEPARATE_QKV_CONTROL,
    _BFP8_DRAM_SEPARATE_QKV_CONTROL,
    _EXPERIMENTAL_DRAM10_OUTPUT_PROJECTION_POLICIES,
    _EXPERIMENTAL_DRAM15_OUTPUT_PROJECTION_POLICIES,
    _EXPERIMENTAL_DRAM_QKV_POLICIES,
    _EXPERIMENTAL_OUTPUT_PROJECTION_POLICIES,
    _FULL_LOCAL_CHECKPOINT_REVISION,
    _PRODUCTION_OUTPUT_PROJECTION_GEOMETRY,
    ATTENTION_BF16_CONTROL,
    ATTENTION_BFP4_HIFI2_OUTACT_POLICY,
    ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY,
    ATTENTION_BFP4_LOFI_OUTACT_POLICY,
    ATTENTION_BFP4_LOFI_POLICY,
    ATTENTION_BFP4_LOFI_SEPARATE_QKV_POLICY,
    ATTENTION_BFP4_POLICY,
    ATTENTION_BFP8_DRAM_SHARDED_POLICY,
    ATTENTION_BFP8_LOFI_OUTACT_POLICY,
    ATTENTION_BFP8_LOFI_POLICY,
    ATTENTION_BFP8_OUTACT_POLICY,
    ATTENTION_BFP8_POLICY,
    ATTENTION_BFP8_SEPARATE_QKV_POLICY,
    DEFAULT_OPTIMIZED_POLICY,
    OptimizedDecoder,
    _attention_weight_cache_namespace,
    _DecodeShardedRMSNorm,
    _DRAMQKVGeometry,
    _L1Router,
    _make_output_projection_program_config,
    _OptimizedAttention,
    _OptimizedMLP,
    _OutputProjectionGeometry,
    _resolve_policy,
    _validate_dram_qkv_geometry,
    _validate_output_projection_geometry,
)
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

# Two ordinary o200k-compatible text sequences, encoded once and pinned here so
# the real-activation A/B does not require tokenizer assets beyond the model
# checkpoint. Both rows are exactly the batch gate's 33-token prefill length.
_BATCH_TWO_REAL_TOKEN_IDS = (
    (
        176289,
        4436,
        164009,
        68074,
        19854,
        1261,
        139562,
        110684,
        91643,
        11,
        3463,
        4432,
        295,
        8684,
        326,
        16435,
        27303,
        12741,
        13,
        7649,
        19230,
        15652,
        326,
        12886,
        6960,
        14090,
        32057,
        13,
        115474,
        4436,
        164009,
        68074,
        19854,
    ),
    (
        24973,
        1920,
        25707,
        316,
        102328,
        261,
        59595,
        53790,
        2049,
        74564,
        132761,
        5251,
        47517,
        5385,
        11,
        4432,
        295,
        2140,
        1432,
        118890,
        11,
        326,
        1701,
        2893,
        13254,
        3626,
        67568,
        13,
        38200,
        1920,
        25707,
        316,
        102328,
    ),
)


def test_optimized_implementation_is_a_distinct_runtime_path():
    source = inspect.getsource(inspect.getmodule(OptimizedDecoder))
    constructor = inspect.getsource(OptimizedDecoder.from_state_dict.__func__)
    assert "class OptimizedDecoder" in source
    assert "super().from_state_dict" not in constructor
    assert "FunctionalDecoder" not in source
    assert DEFAULT_OPTIMIZED_POLICY is ATTENTION_BFP8_POLICY
    assert ATTENTION_BFP8_POLICY.attention_weight_dtype == ttnn.bfloat8_b
    assert ATTENTION_BFP8_POLICY.output_projection_geometry is _PRODUCTION_OUTPUT_PROJECTION_GEOMETRY
    assert ATTENTION_BF16_CONTROL.output_projection_geometry is _PRODUCTION_OUTPUT_PROJECTION_GEOMETRY
    assert _PRODUCTION_OUTPUT_PROJECTION_GEOMETRY.name == "32c_ibw4_pcn3_sb3"
    assert ATTENTION_BFP8_LOFI_POLICY.decode_math_fidelity == ttnn.MathFidelity.LoFi
    assert ATTENTION_BFP4_POLICY.attention_weight_dtype == ttnn.bfloat4_b
    assert ATTENTION_BFP4_LOFI_POLICY.attention_weight_dtype == ttnn.bfloat4_b
    assert ATTENTION_BFP4_LOFI_POLICY.decode_math_fidelity == ttnn.MathFidelity.LoFi
    assert ATTENTION_BFP4_LOFI_POLICY.output_activation_dtype is None
    assert ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY.dram_sharded_qkv
    assert ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY.dram_qkv_geometry.name == "15c_ibw6_pcn10"
    assert ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY.output_activation_dtype is None
    for candidate in (
        ATTENTION_BFP8_LOFI_POLICY,
        ATTENTION_BFP8_OUTACT_POLICY,
        ATTENTION_BFP8_LOFI_OUTACT_POLICY,
        ATTENTION_BFP4_POLICY,
        ATTENTION_BFP4_HIFI2_OUTACT_POLICY,
        ATTENTION_BFP4_LOFI_POLICY,
        ATTENTION_BFP4_LOFI_OUTACT_POLICY,
    ):
        assert candidate.output_projection_geometry is _PRODUCTION_OUTPUT_PROJECTION_GEOMETRY
    assert ATTENTION_BFP8_SEPARATE_QKV_POLICY.separate_qkv
    assert ATTENTION_BFP8_SEPARATE_QKV_POLICY.output_projection_geometry is _PRODUCTION_OUTPUT_PROJECTION_GEOMETRY
    assert ATTENTION_BFP4_LOFI_SEPARATE_QKV_POLICY.separate_qkv
    assert ATTENTION_BFP4_LOFI_SEPARATE_QKV_POLICY.output_projection_geometry is _PRODUCTION_OUTPUT_PROJECTION_GEOMETRY
    for control, dtype, fidelity in (
        (_BFP8_DRAM_SEPARATE_QKV_CONTROL, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
        (_BFP4_LOFI_DRAM_SEPARATE_QKV_CONTROL, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
    ):
        assert control.separate_qkv and control.dram_sharded_qkv
        assert control.attention_weight_dtype == dtype
        assert control.decode_math_fidelity == fidelity
        assert control.dram_qkv_geometry.name == "15c_ibw6_pcn10"
        assert control.dram_separate_qkv_per_core_n == (9, 2, 2)
        assert control.output_projection_geometry is _PRODUCTION_OUTPUT_PROJECTION_GEOMETRY
        for width, per_core_n in zip((4096, 512, 512), control.dram_separate_qkv_per_core_n, strict=True):
            _validate_dram_qkv_geometry(
                _DRAMQKVGeometry("separate", 15, 6, per_core_n),
                k=2880,
                n=width,
            )
            assert width % (8 * ttnn.TILE_SIZE) == 0
    assert "full_local_bfp4_lofi_experts" in OptimizedDecoder.optimization_manifest
    assert "decode_32core_width_sharded_output_projection" in OptimizedDecoder.optimization_manifest
    exported_names = inspect.getmodule(OptimizedDecoder).__all__
    for non_exported_name in (
        "ATTENTION_BFP4_POLICY",
        "ATTENTION_BFP4_LOFI_OUTACT_POLICY",
        "ATTENTION_BFP4_HIFI2_OUTACT_POLICY",
        "ATTENTION_BFP8_DRAM_SHARDED_POLICY",
        "ATTENTION_BFP8_LOFI_POLICY",
        "ATTENTION_BFP8_LOFI_OUTACT_POLICY",
        "ATTENTION_BFP8_OUTACT_POLICY",
        "ATTENTION_BFP8_SEPARATE_QKV_POLICY",
        "_EXPERIMENTAL_DRAM_QKV_POLICIES",
        "_EXPERIMENTAL_OUTPUT_PROJECTION_POLICIES",
    ):
        assert non_exported_name not in exported_names
    assert "ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY" in exported_names
    assert "ATTENTION_BFP4_LOFI_POLICY" in exported_names
    assert inspect.signature(OptimizedDecoder.from_state_dict).parameters["policy"].default is None


def test_optimized_automatic_policy_boundary_is_unambiguous(expect_error):
    assert _resolve_policy(None, 1) is ATTENTION_BFP8_POLICY
    assert _resolve_policy(None, 2) is ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY
    assert _resolve_policy(None, 32) is ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY
    assert _resolve_policy(None, 1).output_projection_geometry is _PRODUCTION_OUTPUT_PROJECTION_GEOMETRY
    assert _resolve_policy(None, 32).output_projection_geometry is _PRODUCTION_OUTPUT_PROJECTION_GEOMETRY
    assert _resolve_policy(ATTENTION_BFP8_POLICY, 2) is ATTENTION_BFP8_POLICY
    with expect_error(ValueError, "unsupported optimized decoder policy"):
        _resolve_policy(object(), 1)


def test_experimental_projection_geometries_are_complete_and_legal(expect_error):
    assert len(_EXPERIMENTAL_DRAM_QKV_POLICIES) == 20
    assert len(_EXPERIMENTAL_OUTPUT_PROJECTION_POLICIES) == 4
    assert len(_EXPERIMENTAL_DRAM10_OUTPUT_PROJECTION_POLICIES) == 4
    assert len(_EXPERIMENTAL_DRAM15_OUTPUT_PROJECTION_POLICIES) == 4
    expected_dram_cross_product = {
        (dtype, fidelity, geometry.name)
        for dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b)
        for fidelity in (ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi)
        for geometry in (
            _DRAMQKVGeometry("90c_ibw1_pcn2", 90, 1, 2),
            _DRAMQKVGeometry("45c_ibw2_pcn4", 45, 2, 4),
            _DRAMQKVGeometry("30c_ibw3_pcn5", 30, 3, 5),
            _DRAMQKVGeometry("15c_ibw6_pcn10", 15, 6, 10),
            _DRAMQKVGeometry("10c_ibw9_pcn16", 10, 9, 16),
        )
    }
    actual_dram_cross_product = set()
    for policy in _EXPERIMENTAL_DRAM_QKV_POLICIES:
        assert policy.dram_sharded_qkv
        assert policy.dram_qkv_geometry is not None
        assert policy.output_projection_geometry is _PRODUCTION_OUTPUT_PROJECTION_GEOMETRY
        _validate_dram_qkv_geometry(policy.dram_qkv_geometry, k=2880, n=5120)
        actual_dram_cross_product.add(
            (policy.attention_weight_dtype, policy.decode_math_fidelity, policy.dram_qkv_geometry.name)
        )
    assert actual_dram_cross_product == expected_dram_cross_product
    for policy in _EXPERIMENTAL_OUTPUT_PROJECTION_POLICIES:
        geometry = policy.output_projection_geometry
        assert geometry is not None
        _validate_output_projection_geometry(geometry, k=4096, n=2880)
        config = _make_output_projection_program_config(geometry)
        assert config.per_core_N * geometry.core_grid[0] * geometry.core_grid[1] >= 90
        assert config.fuse_batch
    for policy in _EXPERIMENTAL_DRAM10_OUTPUT_PROJECTION_POLICIES:
        assert policy.dram_sharded_qkv
        assert policy.dram_qkv_geometry.name == "10c_ibw9_pcn16"
        geometry = policy.output_projection_geometry
        assert geometry is not None
        _validate_output_projection_geometry(geometry, k=4096, n=2880)
    for policy in _EXPERIMENTAL_DRAM15_OUTPUT_PROJECTION_POLICIES:
        assert policy.dram_sharded_qkv
        assert policy.dram_qkv_geometry.name == "15c_ibw6_pcn10"
        geometry = policy.output_projection_geometry
        assert geometry is not None
        _validate_output_projection_geometry(geometry, k=4096, n=2880)

    with expect_error(ValueError, "covers 64 N tiles, requires 90"):
        _validate_output_projection_geometry(
            _OutputProjectionGeometry("invalid_floor_coverage", (8, 8), 2, 1, 1), k=4096, n=2880
        )
    with expect_error(ValueError, "must divide shard K"):
        _validate_dram_qkv_geometry(_DRAMQKVGeometry("invalid_block", 10, 2, 16), k=2880, n=5120)


_EXPERIMENTAL_PROJECTION_POLICIES_BY_NAME = {
    policy.name: policy
    for policy in (
        _EXPERIMENTAL_DRAM_QKV_POLICIES
        + _EXPERIMENTAL_OUTPUT_PROJECTION_POLICIES
        + _EXPERIMENTAL_DRAM10_OUTPUT_PROJECTION_POLICIES
        + _EXPERIMENTAL_DRAM15_OUTPUT_PROJECTION_POLICIES
    )
}
_SELECTED_EXPERIMENTAL_PROJECTION_POLICY = _EXPERIMENTAL_PROJECTION_POLICIES_BY_NAME.get(
    os.environ.get("GPT_OSS_120B_OPTIMIZED_PROJECTION_CANDIDATE", "")
)
_CAPACITY2_DRAM_QKV_10_POLICY = _EXPERIMENTAL_PROJECTION_POLICIES_BY_NAME["attention_bfp4_lofi_dram_qkv_10c_ibw9_pcn16"]


def _assert_optimized_runtime_components(decoder):
    assert type(decoder.input_layernorm) is _DecodeShardedRMSNorm
    assert type(decoder.post_attention_layernorm) is _DecodeShardedRMSNorm
    assert type(decoder.self_attn) is _OptimizedAttention
    assert type(decoder.mlp) is _OptimizedMLP
    assert type(decoder.mlp.router) is _L1Router
    assert decoder.input_layernorm.enable_decode_sharding is (decoder.max_batch_size < ttnn.TILE_SIZE)
    assert decoder.post_attention_layernorm.enable_decode_sharding is (decoder.max_batch_size < ttnn.TILE_SIZE)
    if decoder.policy.output_projection_geometry is not None:
        assert type(decoder.self_attn.decode_output_program_config) is ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig
        assert decoder.self_attn.decode_output_input_memory_config is not None


def _run_real_gate(monkeypatch, gate, mesh_device, device_params, layer_idx, reset_seeds, *, policy=None):
    class GateOptimizedDecoder(OptimizedDecoder):
        @classmethod
        def from_state_dict(cls, *args, **kwargs):
            kwargs["calibrated_checkpoint_revision"] = _FULL_LOCAL_CHECKPOINT_REVISION
            if policy is not None:
                kwargs["policy"] = policy
            decoder = super().from_state_dict(*args, **kwargs)
            assert type(decoder) is cls
            _assert_optimized_runtime_components(decoder)
            assert decoder.mlp.decode_uses_full_local
            assert decoder.policy is _resolve_policy(policy, decoder.max_batch_size)
            assert decoder.requested_policy is policy
            return decoder

    monkeypatch.setattr(accepted, "FunctionalDecoder", GateOptimizedDecoder)
    gate(mesh_device, device_params, layer_idx, reset_seeds)


def _run_synthetic_gate(
    monkeypatch,
    gate,
    mesh_device,
    device_params,
    layer_idx,
    reset_seeds,
    *,
    policy=None,
    configured_capacity=None,
    logical_batch=None,
    finite_only=False,
    audit_paged_cache=False,
):
    cache_audited = False

    def cache_block_to_torch(cache, block_idx):
        shape = tuple(cache.shape)
        block = ttnn.slice(
            cache,
            starts=[block_idx, 0, 0, 0],
            ends=[block_idx + 1, shape[1], shape[2], shape[3]],
            steps=[1, 1, 1, 1],
        )
        host = ttnn.to_torch(block).clone()
        block.deallocate(True)
        return host

    class GateOptimizedDecoder(OptimizedDecoder):
        @classmethod
        def from_state_dict(cls, *args, **kwargs):
            if configured_capacity is not None:
                kwargs["max_batch_size"] = configured_capacity
            if policy is not None:
                kwargs["policy"] = policy
            decoder = super().from_state_dict(*args, **kwargs)
            assert type(decoder) is cls
            assert not decoder.mlp.decode_uses_full_local
            _assert_optimized_runtime_components(decoder)
            expected_policy = _resolve_policy(policy, decoder.max_batch_size)
            assert decoder.policy is expected_policy
            assert decoder.requested_policy is policy
            print(
                "OPTIMIZED_POLICY_AUDIT "
                f"requested_policy={policy.name if policy is not None else 'automatic'} "
                f"effective_policy={expected_policy.name} configured_capacity={decoder.max_batch_size}"
            )
            return decoder

        def prefill_forward(self, *args, **kwargs):
            if logical_batch is not None:
                kwargs["batch_size"] = logical_batch
            return super().prefill_forward(*args, **kwargs)

        def decode_forward(self, *args, **kwargs):
            nonlocal cache_audited
            if logical_batch is not None:
                kwargs["batch_size"] = logical_batch
            if not audit_paged_cache or cache_audited:
                return super().decode_forward(*args, **kwargs)

            page_table = kwargs["page_table"]
            current_position = kwargs["current_position"]
            current = int(ttnn.to_torch(current_position).reshape(-1)[0].item())
            page_table_host = ttnn.to_torch(page_table).reshape(logical_batch, -1)
            logical_block = current // accepted.PAGE_SIZE
            selected_block = int(page_table_host[0, logical_block].item())
            unselected_block = int(page_table_host[0, 0].item())
            caches = tuple(self.self_attn.kv_cache)
            selected_before = tuple(cache_block_to_torch(cache, selected_block) for cache in caches)
            unselected_before = tuple(cache_block_to_torch(cache, unselected_block) for cache in caches)
            result = super().decode_forward(*args, **kwargs)
            ttnn.synchronize_device(mesh_device)
            selected_after = tuple(cache_block_to_torch(cache, selected_block) for cache in caches)
            unselected_after = tuple(cache_block_to_torch(cache, unselected_block) for cache in caches)
            selected_changed = [
                not torch.equal(before, after) for before, after in zip(selected_before, selected_after)
            ]
            unselected_equal = [
                torch.equal(before, after) for before, after in zip(unselected_before, unselected_after)
            ]
            assert all(selected_changed), "decode did not mutate the selected K/V cache page"
            assert all(unselected_equal), "decode mutated an unselected K/V cache page"
            cache_audited = True
            print(
                "LOGICAL_BATCH_CACHE_AUDIT "
                f"logical_batch={logical_batch} configured_capacity={self.max_batch_size} "
                f"current={current} selected_block={selected_block} unselected_block={unselected_block} "
                f"selected_changed={selected_changed} unselected_equal={unselected_equal}"
            )
            return result

    monkeypatch.setattr(accepted, "FunctionalDecoder", GateOptimizedDecoder)
    if finite_only:

        def assert_finite_shape(actual, expected, label, threshold=None):
            del threshold
            assert actual.shape == expected.shape, f"{label} shape mismatch: {actual.shape} != {expected.shape}"
            assert torch.isfinite(actual).all(), f"{label} contains non-finite values"
            return f"finite=True shape={tuple(actual.shape)} random_pcc_not_acceptance"

        monkeypatch.setattr(accepted, "_assert_pcc", assert_finite_shape)
    if os.environ.get("GPT_OSS_120B_RUN_BATCH32") == "1":
        torch_equal = torch.equal

        def audited_equal(actual, expected):
            equal = torch_equal(actual, expected)
            actual_float = actual.float()
            expected_float = expected.float()
            different = actual != expected
            rows = different.reshape(-1, different.shape[-1]).any(dim=-1)
            actual_centered = actual_float.flatten() - actual_float.mean()
            expected_centered = expected_float.flatten() - expected_float.mean()
            pcc = torch.dot(actual_centered, expected_centered) / (
                torch.linalg.vector_norm(actual_centered) * torch.linalg.vector_norm(expected_centered)
            )
            print(
                "AUTOFIX_REPLAY_AUDIT "
                f"equal={equal} different_elements={different.sum().item()} "
                f"different_rows={rows.sum().item()} max_abs_delta="
                f"{(actual_float - expected_float).abs().max().item():.9f} pcc={pcc.item():.9f}"
            )
            return equal

        monkeypatch.setattr(torch, "equal", audited_equal)
    gate(mesh_device, device_params, layer_idx, reset_seeds)
    if audit_paged_cache:
        assert cache_audited


def _run_real_checkpoint_gate(
    monkeypatch,
    gate,
    mesh_device,
    device_params,
    layer_idx,
    reset_seeds,
    *,
    expect_full_local,
    policy=None,
):
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_full_local_real_ab import (
        _load_sweep_layer_state_dict,
    )

    state_dict = _load_sweep_layer_state_dict(accepted.REAL_WEIGHT_SNAPSHOT, layer_idx)
    make_reference = accepted._reference_layer

    def make_real_reference(config, requested_layer_idx, state_dict=None):
        assert state_dict is None
        assert requested_layer_idx == layer_idx
        return make_reference(config, requested_layer_idx, state_dict=state_dict_for_layer)

    state_dict_for_layer = state_dict
    monkeypatch.setattr(accepted, "_reference_layer", make_real_reference)
    torch_randn = torch.randn
    hidden_size = accepted._config().hidden_size

    def checkpoint_randn(*args, **kwargs):
        value = torch_randn(*args, **kwargs)
        if value.ndim and value.shape[-1] == hidden_size:
            value = value.to(torch.bfloat16)
        return value

    monkeypatch.setattr(torch, "randn", checkpoint_randn)

    class GateOptimizedDecoder(OptimizedDecoder):
        @classmethod
        def from_state_dict(cls, *args, **kwargs):
            kwargs["calibrated_checkpoint_revision"] = _FULL_LOCAL_CHECKPOINT_REVISION
            if policy is not None:
                kwargs["policy"] = policy
            kwargs["tensor_cache_path"] = (
                Path(os.environ.get("GPT_OSS_120B_TENSOR_CACHE", "/tmp/gpt_oss_120b_functional_decoder_tensor_cache"))
                / f"layer_{layer_idx}"
            )
            decoder = super().from_state_dict(*args, **kwargs)
            assert type(decoder) is cls
            _assert_optimized_runtime_components(decoder)
            assert decoder.mlp.decode_uses_full_local is expect_full_local
            expected_policy = _resolve_policy(policy, decoder.max_batch_size)
            assert decoder.policy is expected_policy
            assert decoder.requested_policy is policy
            return decoder

    monkeypatch.setattr(accepted, "FunctionalDecoder", GateOptimizedDecoder)
    gate(mesh_device, device_params, layer_idx, reset_seeds)


def _real_batch_two_activations(config, layer_idx):
    """Generate exact checkpoint-derived inputs for representative layers."""
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_full_local_real_ab import (
        _load_sweep_layer_state_dict,
    )

    snapshot = Path(accepted.REAL_WEIGHT_SNAPSHOT)
    assert snapshot.name == _FULL_LOCAL_CHECKPOINT_REVISION
    with (snapshot / "model.safetensors.index.json").open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]
    embedding_key = "model.embed_tokens.weight"
    with safe_open(snapshot / weight_map[embedding_key], framework="pt", device="cpu") as checkpoint:
        embedding_slice = checkpoint.get_slice(embedding_key)
        token_ids = torch.tensor(_BATCH_TWO_REAL_TOKEN_IDS, dtype=torch.long)
        assert token_ids.max().item() < embedding_slice.get_shape()[0]
        embedded = torch.stack([embedding_slice[int(token)] for token in token_ids.flatten()]).reshape(
            *token_ids.shape, config.hidden_size
        )

    positions = torch.arange(token_ids.shape[1], dtype=torch.long)
    position_ids = positions.unsqueeze(0).expand(token_ids.shape[0], -1)
    if layer_idx == 0:
        layer_prefill = embedded
        activation_source = "exact_embedding"
    elif layer_idx == 1:
        layer_zero_state = _load_sweep_layer_state_dict(snapshot, 0)
        layer_zero = accepted._reference_layer(config, 0, state_dict=layer_zero_state)
        with torch.no_grad():
            layer_prefill = layer_zero(
                embedded,
                attention_mask=accepted._prefill_mask(config, 0, token_ids.shape[1]),
                position_embeddings=accepted.GptOssRotaryEmbedding(config)(embedded, position_ids),
                position_ids=position_ids,
                use_cache=False,
            ).to(torch.bfloat16)
        del embedded, layer_zero, layer_zero_state
        gc.collect()
        activation_source = "exact_embedding_plus_exact_layer0"
    else:
        raise ValueError(f"real activation A/B covers representative layers 0/1, got {layer_idx}")

    decode_generator = torch.Generator().manual_seed(211 + layer_idx)
    decode_positions = (torch.randperm(token_ids.shape[1] - 1, generator=decode_generator) + 2)[: token_ids.shape[0]]
    layer_decode = torch.stack(
        [layer_prefill[user, position] for user, position in enumerate(decode_positions.tolist())]
    ).unsqueeze(1)
    print(
        "BATCH2_REAL_ACTIVATIONS "
        f"revision={snapshot.name} target_layer={layer_idx} activation_source={activation_source} "
        f"embedding_rows=exact tokens={token_ids.shape[1]} "
        f"decode_positions={decode_positions.tolist()} "
        f"prefill_mean={layer_prefill.float().mean().item():.9f} "
        f"prefill_std={layer_prefill.float().std().item():.9f}"
    )
    return layer_prefill.contiguous(), layer_decode.contiguous(), decode_positions, activation_source


def _real_batch_two_reference(config, state_dict, prefill_hidden, decode_hidden, decode_positions, layer_idx):
    """Compute an exact-layer oracle for the fixed real-activation batch."""
    reference = accepted._reference_layer(config, layer_idx, state_dict=state_dict)
    batch_size, sequence_length, _ = prefill_hidden.shape
    positions = torch.arange(sequence_length, dtype=torch.long)
    position_ids = positions.unsqueeze(0).expand(batch_size, -1)
    with torch.no_grad():
        expected_prefill = reference(
            prefill_hidden,
            attention_mask=accepted._prefill_mask(config, layer_idx, sequence_length),
            position_embeddings=accepted.GptOssRotaryEmbedding(config)(prefill_hidden, position_ids),
            position_ids=position_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        expected_decode = []
        for user, current_position in enumerate(decode_positions.tolist()):
            user_cache = DynamicCache()
            user_positions = torch.arange(current_position, dtype=torch.long)
            user_prefill = prefill_hidden[user : user + 1, :current_position]
            reference(
                user_prefill,
                attention_mask=accepted._prefill_mask(config, layer_idx, current_position),
                position_embeddings=accepted._hf_position_embeddings(config, user_prefill, user_positions),
                position_ids=user_positions.unsqueeze(0),
                past_key_values=user_cache,
                use_cache=True,
            )
            user_position_ids = decode_positions[user : user + 1].unsqueeze(1)
            user_decode = decode_hidden[user : user + 1]
            expected_decode.append(
                reference(
                    user_decode,
                    attention_mask=accepted._decode_mask(config, layer_idx, current_position),
                    position_embeddings=accepted.GptOssRotaryEmbedding(config)(user_decode, user_position_ids),
                    position_ids=user_position_ids,
                    past_key_values=user_cache,
                    use_cache=True,
                )[0, 0]
            )
    del reference
    gc.collect()
    return expected_prefill, torch.stack(expected_decode)


def _measure_exact_activation_policy(
    mesh_device,
    config,
    state_dict,
    layer_idx,
    prefill_hidden,
    decode_hidden,
    decode_positions,
    expected_prefill,
    expected_decode,
    activation_source,
    requested_policy,
    repeats,
    *,
    label,
    configured_capacity=None,
):
    """Measure one policy on the fixed exact-checkpoint activation contract."""
    batch_size, sequence_length, hidden_size = prefill_hidden.shape
    max_batch_size = configured_capacity if configured_capacity is not None else batch_size
    assert max_batch_size >= batch_size
    effective_policy = _resolve_policy(requested_policy, max_batch_size)
    constructor_kwargs = {
        "hf_config": config,
        "layer_idx": layer_idx,
        "mesh_device": mesh_device,
        "max_batch_size": max_batch_size,
        "max_context_length": config.max_position_embeddings,
        "page_size": accepted.PAGE_SIZE,
        "tensor_cache_path": Path(
            os.environ.get(
                "GPT_OSS_120B_TENSOR_CACHE",
                "/tmp/gpt_oss_120b_functional_decoder_tensor_cache",
            )
        )
        / f"layer_{layer_idx}",
        "calibrated_checkpoint_revision": _FULL_LOCAL_CHECKPOINT_REVISION,
    }
    if requested_policy is not None:
        constructor_kwargs["policy"] = requested_policy
    decoder = OptimizedDecoder.from_state_dict(state_dict, **constructor_kwargs)
    assert decoder.policy is effective_policy
    assert decoder.requested_policy is requested_policy
    assert effective_policy.output_projection_geometry is not None
    assert decoder.self_attn.weights.wqkv.dtype == effective_policy.attention_weight_dtype
    assert decoder.self_attn.weights.o_proj.dtype == effective_policy.attention_weight_dtype
    assert (
        decoder.self_attn.decode_projection_compute_kernel_config.math_fidelity == effective_policy.decode_math_fidelity
    )
    assert decoder.mlp.decode_uses_full_local
    _assert_optimized_runtime_components(decoder)
    output_config = decoder.self_attn.decode_output_program_config
    output_geometry = effective_policy.output_projection_geometry
    assert output_config.compute_with_storage_grid_size == ttnn.CoreCoord(*output_geometry.core_grid)
    assert output_config.in0_block_w == output_geometry.in0_block_w
    assert output_config.per_core_N == output_geometry.per_core_n
    assert output_config.out_subblock_w == output_geometry.out_subblock_w
    assert output_config.fuse_batch
    if effective_policy.dram_sharded_qkv and not effective_policy.separate_qkv:
        assert effective_policy.dram_qkv_geometry is not None
        assert (
            type(decoder.self_attn.decode_qkv_program_config)
            is ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
        )
        assert decoder.self_attn.decode_qkv_program_config.in0_block_w == effective_policy.dram_qkv_geometry.in0_block_w
        assert decoder.self_attn.decode_qkv_program_config.per_core_N == effective_policy.dram_qkv_geometry.per_core_n
        assert decoder.self_attn.decode_qkv_input_memory_config is not None
        assert decoder.self_attn.decode_qkv_to_interleaved
        assert not decoder.self_attn.decode_qkv_fuse_bias
    if effective_policy.separate_qkv:
        assert decoder.self_attn.decode_separate_qkv_weights is not None
        assert len(decoder.self_attn.decode_separate_qkv_weights) == 3
        assert decoder.self_attn.decode_qkv_program_config is None
        assert decoder.self_attn.decode_separate_qkv_program_configs is not None
        if effective_policy.dram_sharded_qkv:
            assert effective_policy.dram_separate_qkv_per_core_n == (9, 2, 2)
            assert decoder.self_attn.decode_separate_qkv_to_interleaved
            assert not decoder.self_attn.decode_qkv_to_interleaved
            assert not decoder.self_attn.decode_qkv_fuse_bias
            for (weight, bias), program_config, per_core_n, shard_width in zip(
                decoder.self_attn.decode_separate_qkv_weights,
                decoder.self_attn.decode_separate_qkv_program_configs,
                effective_policy.dram_separate_qkv_per_core_n,
                (512, 64, 64),
                strict=True,
            ):
                assert bias is None
                assert type(program_config) is ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
                assert program_config.in0_block_w == 6
                assert program_config.per_core_M == 1
                assert program_config.per_core_N == per_core_n
                memory_config = weight.memory_config()
                assert memory_config.buffer_type == ttnn.BufferType.DRAM
                assert memory_config.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
                assert memory_config.shard_spec.shape == (2880, shard_width)
        else:
            assert decoder.self_attn.decode_separate_qkv_program_configs == (None, None, None)
            assert not decoder.self_attn.decode_separate_qkv_to_interleaved

    page_table = accepted._page_table(
        mesh_device,
        config.max_position_embeddings,
        batch_size=batch_size,
        seed=71 + layer_idx,
    )
    positions = torch.arange(sequence_length, dtype=torch.long)
    prefill_rope = accepted._rope_tensors(config, mesh_device, positions, decode=False)
    tt_prefill_hidden = ttnn.from_torch(
        prefill_hidden.reshape(1, batch_size, sequence_length, hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_decode_hidden = None
    tt_current = None
    decode_rope = []
    warm_prefill = None
    measured_prefill = None
    eager_decode = None
    traced_decode = None
    trace_id = None
    trace_released = False
    try:
        warm_prefill = decoder.prefill_forward(
            tt_prefill_hidden,
            position_embeddings=prefill_rope,
            page_table=page_table,
            batch_size=batch_size,
        )
        ttnn.synchronize_device(mesh_device)
        warm_prefill.deallocate(True)
        warm_prefill = None

        started = time.perf_counter()
        measured_prefill = decoder.prefill_forward(
            tt_prefill_hidden,
            position_embeddings=prefill_rope,
            page_table=page_table,
            batch_size=batch_size,
        )
        ttnn.synchronize_device(mesh_device)
        prefill_wall_ms = 1000 * (time.perf_counter() - started)
        prefill_actual = accepted._to_host_batch(measured_prefill)
        _, prefill_pcc = comp_pcc(expected_prefill.float(), prefill_actual.float(), 0.0)

        tt_decode_hidden = ttnn.from_torch(
            decode_hidden.reshape(1, 1, batch_size, hidden_size),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_current = ttnn.from_torch(
            decode_positions.to(torch.int32),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        decode_rope = accepted._rope_tensors(config, mesh_device, decode_positions, decode=True)
        eager_decode = decoder.decode_forward(
            tt_decode_hidden,
            position_embeddings=decode_rope,
            current_position=tt_current,
            page_table=page_table,
            batch_size=batch_size,
        )
        ttnn.synchronize_device(mesh_device)
        eager_decode.deallocate(True)
        eager_decode = None

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            traced_decode = decoder.decode_forward(
                tt_decode_hidden,
                position_embeddings=decode_rope,
                current_position=tt_current,
                page_table=page_table,
                batch_size=batch_size,
            )
        finally:
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        first_replay = accepted._to_host(traced_decode)[:batch_size].clone()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        second_replay = accepted._to_host(traced_decode)[:batch_size].clone()
        first_second_deterministic = torch.equal(second_replay, first_replay)

        started = time.perf_counter()
        for _ in range(repeats):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        traced_decode_wall_ms = 1000 * (time.perf_counter() - started) / repeats
        after_replays = accepted._to_host(traced_decode)[:batch_size].clone()
        after_replays_deterministic = torch.equal(after_replays, first_replay)
        _, decode_pcc = comp_pcc(expected_decode.float(), after_replays.float(), 0.0)
        result = {
            "prefill_pcc": prefill_pcc,
            "decode_pcc": decode_pcc,
            "prefill_wall_ms": prefill_wall_ms,
            "traced_decode_wall_ms": traced_decode_wall_ms,
            "first_second_deterministic": first_second_deterministic,
            "after_replays_deterministic": after_replays_deterministic,
        }
        geometry_name = (
            effective_policy.dram_qkv_geometry.name if effective_policy.dram_qkv_geometry is not None else "none"
        )
        print(
            "EXACT_ACTIVATION_POLICY_MEASURE "
            f"label={label} revision={_FULL_LOCAL_CHECKPOINT_REVISION} layer={layer_idx} "
            f"type={config.layer_types[layer_idx]} logical_batch={batch_size} configured_capacity={max_batch_size} "
            f"requested_policy={requested_policy.name if requested_policy is not None else 'automatic'} "
            f"effective_policy={effective_policy.name} activation_source={activation_source} "
            f"attention_weight_dtype={effective_policy.attention_weight_dtype} "
            f"decode_projection_fidelity={effective_policy.decode_math_fidelity} "
            f"dram_qkv_geometry={geometry_name} "
            f"output_projection_geometry={effective_policy.output_projection_geometry.name} "
            f"sequence={sequence_length} decode_positions={decode_positions.tolist()} "
            f"prefill_pcc={prefill_pcc:.9f} decode_pcc={decode_pcc:.9f} "
            f"warmed_prefill_wall_ms={prefill_wall_ms:.9f} repeats={repeats} "
            f"traced_decode_wall_ms={traced_decode_wall_ms:.9f} "
            f"first_second_deterministic={first_second_deterministic} "
            f"after_replays_deterministic={after_replays_deterministic}"
        )
        return result
    finally:
        if trace_id is not None and not trace_released:
            ttnn.release_trace(mesh_device, trace_id)
        for tensor in (warm_prefill, measured_prefill, eager_decode, traced_decode):
            if tensor is not None:
                tensor.deallocate(True)
        for tensor in (*prefill_rope, *decode_rope, tt_prefill_hidden, tt_decode_hidden, tt_current, page_table):
            if tensor is not None:
                tensor.deallocate(True)
        del decoder
        gc.collect()


@pytest.mark.skipif(not accepted.REAL_WEIGHT_SNAPSHOT, reason="set GPT_OSS_120B_SNAPSHOT for real-weight gates")
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_real_weight_paged_prefill_and_traced_decode(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_real_gate(
        monkeypatch,
        accepted.test_real_weight_paged_prefill_and_traced_decode,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        # This inherited random-activation robustness control is deliberately
        # explicit. Production omission is covered with prompt-derived inputs.
        policy=ATTENTION_BFP8_POLICY,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_OPTIMIZED_BFP4") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_OPTIMIZED_BFP4=1 and GPT_OSS_120B_SNAPSHOT for the BFP4 attention trial",
)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_real_weight_bfp4_attention_candidate(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_real_gate(
        monkeypatch,
        accepted.test_real_weight_paged_prefill_and_traced_decode,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        policy=ATTENTION_BFP4_POLICY,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_OPTIMIZED_LOFI") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_OPTIMIZED_LOFI=1 and GPT_OSS_120B_SNAPSHOT for the LoFi projection trial",
)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_real_weight_lofi_projection_candidate(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_real_gate(
        monkeypatch,
        accepted.test_real_weight_paged_prefill_and_traced_decode,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        policy=ATTENTION_BFP8_LOFI_POLICY,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_OPTIMIZED_BFP8_OUTACT") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_OPTIMIZED_BFP8_OUTACT=1 for the BFP8 output-activation trial",
)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_real_weight_bfp8_output_activation_candidate(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_real_gate(
        monkeypatch,
        accepted.test_real_weight_paged_prefill_and_traced_decode,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        policy=ATTENTION_BFP8_OUTACT_POLICY,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_OPTIMIZED_DRAM_QKV") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_OPTIMIZED_DRAM_QKV=1 for the DRAM-sharded QKV trial",
)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_real_weight_dram_sharded_qkv_candidate(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_real_gate(
        monkeypatch,
        accepted.test_real_weight_paged_prefill_and_traced_decode,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        policy=ATTENTION_BFP8_DRAM_SHARDED_POLICY,
    )


@pytest.mark.skipif(
    _SELECTED_EXPERIMENTAL_PROJECTION_POLICY is None or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_OPTIMIZED_PROJECTION_CANDIDATE to a named experimental policy",
)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_real_weight_selected_projection_candidate(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_real_gate(
        monkeypatch,
        accepted.test_real_weight_paged_prefill_and_traced_decode,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        policy=_SELECTED_EXPERIMENTAL_PROJECTION_POLICY,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_OPTIMIZED_SEPARATE_QKV") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_OPTIMIZED_SEPARATE_QKV=1 for the separate Q/K/V projection trial",
)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_real_weight_separate_qkv_candidate(monkeypatch, mesh_device, device_params, layer_idx, reset_seeds):
    _run_real_gate(
        monkeypatch,
        accepted.test_real_weight_paged_prefill_and_traced_decode,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        policy=ATTENTION_BFP8_SEPARATE_QKV_POLICY,
    )


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_non_aligned_prefill_boundaries(monkeypatch, mesh_device, device_params, layer_idx, reset_seeds):
    _run_synthetic_gate(
        monkeypatch,
        accepted.test_prefill_tile_page_and_window_boundaries,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_BATCH2_SYNTHETIC_BFP8_DIAGNOSTIC") != "1",
    reason="set GPT_OSS_120B_BATCH2_SYNTHETIC_BFP8_DIAGNOSTIC=1 for the rejected random-input BFP8 diagnostic",
)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_rejected_synthetic_bfp8_batch_two_diagnostic(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds, expect_error
):
    def run():
        _run_synthetic_gate(
            monkeypatch,
            accepted.test_batch_two_paged_prefill_and_traced_decode,
            mesh_device,
            device_params,
            layer_idx,
            reset_seeds,
            policy=ATTENTION_BFP8_POLICY,
        )

    if layer_idx == 1:
        with expect_error(AssertionError, "batch-two traced decode failed"):
            run()
    else:
        run()


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_BATCH2_RANDOM_BFP8_DIAGNOSTIC") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_BATCH2_RANDOM_BFP8_DIAGNOSTIC=1 and GPT_OSS_120B_SNAPSHOT for the rejected exact-weight/random-input BFP8 diagnostic",
)
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_rejected_real_weight_random_activation_bfp8_batch_two_diagnostic(
    monkeypatch, mesh_device, device_params, reset_seeds, expect_error
):
    with expect_error(AssertionError, "batch-two traced decode failed"):
        _run_real_checkpoint_gate(
            monkeypatch,
            accepted.test_batch_two_paged_prefill_and_traced_decode,
            mesh_device,
            device_params,
            1,
            reset_seeds,
            expect_full_local=True,
            policy=ATTENTION_BFP8_POLICY,
        )


@pytest.mark.skipif(
    not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_SNAPSHOT for the production real-activation batch-2 gate",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_real_activation_batch_two_precision_and_latency_ab(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    """Compare one attention policy with the automatic DRAM-QKV policy on exact activations."""
    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder import _install_profile_constructor_drains
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_full_local_real_ab import (
        _load_sweep_layer_state_dict,
    )

    _install_profile_constructor_drains(monkeypatch, mesh_device)
    config = accepted._config()
    prefill_hidden, decode_hidden, decode_positions, activation_source = _real_batch_two_activations(config, layer_idx)
    state_dict = _load_sweep_layer_state_dict(accepted.REAL_WEIGHT_SNAPSHOT, layer_idx)
    expected_prefill, expected_decode = _real_batch_two_reference(
        config, state_dict, prefill_hidden, decode_hidden, decode_positions, layer_idx
    )
    batch_size, sequence_length, hidden_size = prefill_hidden.shape
    repeats = int(os.environ.get("GPT_OSS_120B_BATCH2_POLICY_AB_REPEATS", "1000"))
    assert repeats >= 1000
    assert _resolve_policy(None, batch_size) is ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY

    def measure(requested_policy):
        effective_policy = _resolve_policy(requested_policy, batch_size)
        constructor_kwargs = {
            "hf_config": config,
            "layer_idx": layer_idx,
            "mesh_device": mesh_device,
            "max_batch_size": batch_size,
            "max_context_length": config.max_position_embeddings,
            "page_size": accepted.PAGE_SIZE,
            "tensor_cache_path": Path(
                os.environ.get(
                    "GPT_OSS_120B_TENSOR_CACHE",
                    "/tmp/gpt_oss_120b_functional_decoder_tensor_cache",
                )
            )
            / f"layer_{layer_idx}",
            "calibrated_checkpoint_revision": _FULL_LOCAL_CHECKPOINT_REVISION,
        }
        if requested_policy is not None:
            constructor_kwargs["policy"] = requested_policy
        decoder = OptimizedDecoder.from_state_dict(state_dict, **constructor_kwargs)
        assert decoder.policy is effective_policy
        assert decoder.requested_policy is requested_policy
        if requested_policy is None:
            assert effective_policy is ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY
        else:
            assert effective_policy is requested_policy
        assert effective_policy.output_projection_geometry is _PRODUCTION_OUTPUT_PROJECTION_GEOMETRY
        assert decoder.self_attn.weights.wqkv.dtype == effective_policy.attention_weight_dtype
        assert decoder.self_attn.weights.o_proj.dtype == effective_policy.attention_weight_dtype
        assert (
            decoder.self_attn.decode_projection_compute_kernel_config.math_fidelity
            == effective_policy.decode_math_fidelity
        )
        assert decoder.mlp.decode_uses_full_local
        _assert_optimized_runtime_components(decoder)

        page_table = accepted._page_table(
            mesh_device,
            config.max_position_embeddings,
            batch_size=batch_size,
            seed=71 + layer_idx,
        )
        positions = torch.arange(sequence_length, dtype=torch.long)
        prefill_rope = accepted._rope_tensors(config, mesh_device, positions, decode=False)
        tt_prefill_hidden = ttnn.from_torch(
            prefill_hidden.reshape(1, batch_size, sequence_length, hidden_size),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_decode_hidden = None
        tt_current = None
        decode_rope = []
        warm_prefill = None
        measured_prefill = None
        eager_decode = None
        traced_decode = None
        trace_id = None
        trace_released = False
        try:
            warm_prefill = decoder.prefill_forward(
                tt_prefill_hidden,
                position_embeddings=prefill_rope,
                page_table=page_table,
                batch_size=batch_size,
            )
            ttnn.synchronize_device(mesh_device)
            warm_prefill.deallocate(True)
            warm_prefill = None

            started = time.perf_counter()
            measured_prefill = decoder.prefill_forward(
                tt_prefill_hidden,
                position_embeddings=prefill_rope,
                page_table=page_table,
                batch_size=batch_size,
            )
            ttnn.synchronize_device(mesh_device)
            prefill_wall_ms = 1000 * (time.perf_counter() - started)
            prefill_actual = accepted._to_host_batch(measured_prefill)
            _, prefill_pcc = comp_pcc(expected_prefill.float(), prefill_actual.float(), 0.0)

            tt_decode_hidden = ttnn.from_torch(
                decode_hidden.reshape(1, 1, batch_size, hidden_size),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            tt_current = ttnn.from_torch(
                decode_positions.to(torch.int32),
                device=mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            decode_rope = accepted._rope_tensors(config, mesh_device, decode_positions, decode=True)
            eager_decode = decoder.decode_forward(
                tt_decode_hidden,
                position_embeddings=decode_rope,
                current_position=tt_current,
                page_table=page_table,
                batch_size=batch_size,
            )
            ttnn.synchronize_device(mesh_device)
            eager_decode.deallocate(True)
            eager_decode = None

            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            try:
                traced_decode = decoder.decode_forward(
                    tt_decode_hidden,
                    position_embeddings=decode_rope,
                    current_position=tt_current,
                    page_table=page_table,
                    batch_size=batch_size,
                )
            finally:
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            first_replay = accepted._to_host(traced_decode)[:batch_size].clone()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            second_replay = accepted._to_host(traced_decode)[:batch_size].clone()
            first_second_deterministic = torch.equal(second_replay, first_replay)

            started = time.perf_counter()
            for _ in range(repeats):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            traced_decode_wall_ms = 1000 * (time.perf_counter() - started) / repeats
            after_replays = accepted._to_host(traced_decode)[:batch_size].clone()
            after_1000_deterministic = torch.equal(after_replays, first_replay)
            _, decode_pcc = comp_pcc(expected_decode.float(), after_replays.float(), 0.0)
            ttnn.release_trace(mesh_device, trace_id)
            trace_released = True
            result = {
                "prefill_pcc": prefill_pcc,
                "decode_pcc": decode_pcc,
                "prefill_wall_ms": prefill_wall_ms,
                "traced_decode_wall_ms": traced_decode_wall_ms,
                "first_second_deterministic": first_second_deterministic,
                "after_1000_deterministic": after_1000_deterministic,
            }
            print(
                "BATCH2_REAL_POLICY_AB "
                f"revision={_FULL_LOCAL_CHECKPOINT_REVISION} layer={layer_idx} "
                f"type={config.layer_types[layer_idx]} "
                f"requested_policy={requested_policy.name if requested_policy is not None else 'automatic'} "
                f"effective_policy={effective_policy.name} activation_source={activation_source} "
                f"attention_weight_dtype={effective_policy.attention_weight_dtype} "
                f"decode_projection_fidelity={effective_policy.decode_math_fidelity} "
                f"output_activation_dtype={effective_policy.output_activation_dtype} "
                f"output_projection_geometry={effective_policy.output_projection_geometry.name} "
                f"configured_capacity={batch_size} automatic_policy={_resolve_policy(None, batch_size).name} "
                f"sequence={sequence_length} decode_positions={decode_positions.tolist()} "
                f"prefill_pcc={prefill_pcc:.9f} decode_pcc={decode_pcc:.9f} "
                f"warmed_prefill_wall_ms={prefill_wall_ms:.9f} repeats={repeats} "
                f"traced_decode_wall_ms={traced_decode_wall_ms:.9f} "
                f"first_second_deterministic={first_second_deterministic} "
                f"after_1000_deterministic={after_1000_deterministic}"
            )
            return result
        finally:
            if trace_id is not None and not trace_released:
                ttnn.release_trace(mesh_device, trace_id)
            for tensor in (warm_prefill, measured_prefill, eager_decode, traced_decode):
                if tensor is not None:
                    tensor.deallocate(True)
            for tensor in (*prefill_rope, *decode_rope, tt_prefill_hidden, tt_decode_hidden, tt_current, page_table):
                if tensor is not None:
                    tensor.deallocate(True)
            del decoder
            gc.collect()

    order_name = os.environ.get("GPT_OSS_120B_BATCH2_POLICY_AB_ORDER", "automatic,bfp8")
    requested_by_name = {
        "automatic": None,
        "bfp8": ATTENTION_BFP8_POLICY,
        "bf16": ATTENTION_BF16_CONTROL,
        "bfp8_lofi": ATTENTION_BFP8_LOFI_POLICY,
        "bfp8_outact": ATTENTION_BFP8_OUTACT_POLICY,
        "bfp8_lofi_outact": ATTENTION_BFP8_LOFI_OUTACT_POLICY,
        "bfp4_hifi2": ATTENTION_BFP4_POLICY,
        "bfp4_hifi2_outact": ATTENTION_BFP4_HIFI2_OUTACT_POLICY,
        "bfp4_lofi": ATTENTION_BFP4_LOFI_POLICY,
        "bfp4_lofi_outact": ATTENTION_BFP4_LOFI_OUTACT_POLICY,
    }
    order = order_name.split(",")
    if (
        len(order) != 2
        or len(set(order)) != 2
        or "automatic" not in order
        or any(name not in requested_by_name for name in order)
    ):
        raise ValueError(
            "GPT_OSS_120B_BATCH2_POLICY_AB_ORDER must compare 'automatic' with exactly one of "
            f"{tuple(name for name in requested_by_name if name != 'automatic')}, got {order_name!r}"
        )
    results = {}
    for requested_name in order:
        requested_policy = requested_by_name[requested_name]
        results[requested_name] = measure(requested_policy)
    automatic = results["automatic"]
    candidate_name = next(name for name in order if name != "automatic")
    candidate = results[candidate_name]
    print(
        "BATCH2_REAL_POLICY_AB_SUMMARY "
        f"layer={layer_idx} type={config.layer_types[layer_idx]} order={order_name} candidate={candidate_name} "
        f"automatic_prefill_pcc={automatic['prefill_pcc']:.9f} "
        f"candidate_prefill_pcc={candidate['prefill_pcc']:.9f} "
        f"automatic_decode_pcc={automatic['decode_pcc']:.9f} "
        f"candidate_decode_pcc={candidate['decode_pcc']:.9f} "
        f"automatic_warmed_prefill_wall_ms={automatic['prefill_wall_ms']:.9f} "
        f"candidate_warmed_prefill_wall_ms={candidate['prefill_wall_ms']:.9f} "
        f"automatic_traced_decode_wall_ms={automatic['traced_decode_wall_ms']:.9f} "
        f"candidate_traced_decode_wall_ms={candidate['traced_decode_wall_ms']:.9f} "
        f"candidate_vs_automatic_prefill_pct="
        f"{100 * (candidate['prefill_wall_ms'] / automatic['prefill_wall_ms'] - 1):.3f} "
        f"candidate_vs_automatic_decode_pct="
        f"{100 * (candidate['traced_decode_wall_ms'] / automatic['traced_decode_wall_ms'] - 1):.3f}"
    )
    for result in results.values():
        assert result["first_second_deterministic"]
        assert result["after_1000_deterministic"]
        assert result["prefill_pcc"] >= accepted.PREFILL_PCC_THRESHOLD
        assert result["decode_pcc"] >= accepted.BATCH_DECODE_PCC_THRESHOLD
    if candidate_name == "bfp8":
        assert automatic["traced_decode_wall_ms"] < candidate["traced_decode_wall_ms"]


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_DRAM_QKV_EXACT_MATRIX") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_DRAM_QKV_EXACT_MATRIX=1 and GPT_OSS_120B_SNAPSHOT for the DRAM-QKV matrix",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("configured_capacity", [1, 2], ids=["batch1", "batch2"])
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@pytest.mark.parametrize("candidate_policy", _EXPERIMENTAL_DRAM_QKV_POLICIES, ids=lambda policy: policy.name)
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_exact_activation_dram_qkv_cross_product(
    monkeypatch,
    mesh_device,
    device_params,
    candidate_policy,
    layer_idx,
    configured_capacity,
    reset_seeds,
):
    """Compare each cumulative DRAM-QKV row with the capacity default."""
    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder import _install_profile_constructor_drains
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_full_local_real_ab import (
        _load_sweep_layer_state_dict,
    )

    _install_profile_constructor_drains(monkeypatch, mesh_device)
    repeats = int(os.environ.get("GPT_OSS_120B_DRAM_QKV_EXACT_MATRIX_REPEATS", "1000"))
    assert repeats >= 1000
    assert candidate_policy.dram_sharded_qkv
    assert candidate_policy.dram_qkv_geometry is not None
    assert candidate_policy.output_projection_geometry is _PRODUCTION_OUTPUT_PROJECTION_GEOMETRY

    config = accepted._config()
    prefill_hidden, decode_hidden, decode_positions, activation_source = _real_batch_two_activations(config, layer_idx)
    prefill_hidden = prefill_hidden[:configured_capacity].contiguous()
    decode_hidden = decode_hidden[:configured_capacity].contiguous()
    decode_positions = decode_positions[:configured_capacity].contiguous()
    state_dict = _load_sweep_layer_state_dict(accepted.REAL_WEIGHT_SNAPSHOT, layer_idx)
    expected_prefill, expected_decode = _real_batch_two_reference(
        config, state_dict, prefill_hidden, decode_hidden, decode_positions, layer_idx
    )

    order_name = os.environ.get("GPT_OSS_120B_DRAM_QKV_EXACT_MATRIX_ORDER", "automatic,candidate")
    order = order_name.split(",")
    if sorted(order) != ["automatic", "candidate"]:
        raise ValueError(
            "GPT_OSS_120B_DRAM_QKV_EXACT_MATRIX_ORDER must be "
            f"'automatic,candidate' or 'candidate,automatic', got {order_name!r}"
        )
    requested_by_name = {"automatic": None, "candidate": candidate_policy}
    results = {}
    for name in order:
        results[name] = _measure_exact_activation_policy(
            mesh_device,
            config,
            state_dict,
            layer_idx,
            prefill_hidden,
            decode_hidden,
            decode_positions,
            expected_prefill,
            expected_decode,
            activation_source,
            requested_by_name[name],
            repeats,
            label=f"dram_qkv_matrix_{name}",
        )

    automatic = results["automatic"]
    candidate = results["candidate"]
    decode_delta_pct = 100 * (candidate["traced_decode_wall_ms"] / automatic["traced_decode_wall_ms"] - 1)
    print(
        "DRAM_QKV_EXACT_MATRIX_SUMMARY "
        f"revision={_FULL_LOCAL_CHECKPOINT_REVISION} layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"configured_capacity={configured_capacity} order={order_name} candidate={candidate_policy.name} "
        f"geometry={candidate_policy.dram_qkv_geometry.name} "
        f"dtype={candidate_policy.attention_weight_dtype} fidelity={candidate_policy.decode_math_fidelity} "
        f"output_projection_geometry={candidate_policy.output_projection_geometry.name} "
        f"automatic_policy={_resolve_policy(None, configured_capacity).name} "
        f"automatic_prefill_pcc={automatic['prefill_pcc']:.9f} "
        f"candidate_prefill_pcc={candidate['prefill_pcc']:.9f} "
        f"automatic_decode_pcc={automatic['decode_pcc']:.9f} "
        f"candidate_decode_pcc={candidate['decode_pcc']:.9f} "
        f"automatic_warmed_prefill_wall_ms={automatic['prefill_wall_ms']:.9f} "
        f"candidate_warmed_prefill_wall_ms={candidate['prefill_wall_ms']:.9f} "
        f"automatic_traced_decode_wall_ms={automatic['traced_decode_wall_ms']:.9f} "
        f"candidate_traced_decode_wall_ms={candidate['traced_decode_wall_ms']:.9f} "
        f"candidate_vs_automatic_decode_pct={decode_delta_pct:.3f}"
    )
    for result in results.values():
        assert result["first_second_deterministic"]
        assert result["after_replays_deterministic"]
        assert result["prefill_pcc"] >= accepted.PREFILL_PCC_THRESHOLD
        assert result["decode_pcc"] >= accepted.BATCH_DECODE_PCC_THRESHOLD


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_DRAM_QKV_10_VS_15") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_DRAM_QKV_10_VS_15=1 and GPT_OSS_120B_SNAPSHOT for the focused capacity-2 A/B",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_exact_activation_capacity2_dram_qkv_10_vs_15(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    """Compare capacity-2 DRAM10 directly with DRAM15 on identical topology."""
    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder import _install_profile_constructor_drains
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_full_local_real_ab import (
        _load_sweep_layer_state_dict,
    )

    _install_profile_constructor_drains(monkeypatch, mesh_device)
    repeats = int(os.environ.get("GPT_OSS_120B_DRAM_QKV_10_VS_15_REPEATS", "1000"))
    assert repeats >= 1000
    config = accepted._config()
    prefill_hidden, decode_hidden, decode_positions, activation_source = _real_batch_two_activations(config, layer_idx)
    state_dict = _load_sweep_layer_state_dict(accepted.REAL_WEIGHT_SNAPSHOT, layer_idx)
    expected_prefill, expected_decode = _real_batch_two_reference(
        config, state_dict, prefill_hidden, decode_hidden, decode_positions, layer_idx
    )
    policies = {
        "dram15": ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY,
        "dram10": _CAPACITY2_DRAM_QKV_10_POLICY,
    }
    order_name = os.environ.get("GPT_OSS_120B_DRAM_QKV_10_VS_15_ORDER", "dram15,dram10")
    order = order_name.split(",")
    if sorted(order) != ["dram10", "dram15"]:
        raise ValueError(
            "GPT_OSS_120B_DRAM_QKV_10_VS_15_ORDER must be 'dram15,dram10' or 'dram10,dram15', " f"got {order_name!r}"
        )
    results = {
        name: _measure_exact_activation_policy(
            mesh_device,
            config,
            state_dict,
            layer_idx,
            prefill_hidden,
            decode_hidden,
            decode_positions,
            expected_prefill,
            expected_decode,
            activation_source,
            policies[name],
            repeats,
            label=f"capacity2_dram_qkv_10_vs_15_{name}",
        )
        for name in order
    }
    dram15, dram10 = results["dram15"], results["dram10"]
    print(
        "CAPACITY2_DRAM_QKV_10_VS_15_SUMMARY "
        f"revision={_FULL_LOCAL_CHECKPOINT_REVISION} layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"order={order_name} repeats={repeats} "
        f"dram15_decode_pcc={dram15['decode_pcc']:.9f} dram10_decode_pcc={dram10['decode_pcc']:.9f} "
        f"dram15_traced_decode_wall_ms={dram15['traced_decode_wall_ms']:.9f} "
        f"dram10_traced_decode_wall_ms={dram10['traced_decode_wall_ms']:.9f} "
        f"dram10_vs_dram15_decode_pct="
        f"{100 * (dram10['traced_decode_wall_ms'] / dram15['traced_decode_wall_ms'] - 1):.3f}"
    )
    for result in results.values():
        assert result["first_second_deterministic"]
        assert result["after_replays_deterministic"]
        assert result["prefill_pcc"] >= accepted.PREFILL_PCC_THRESHOLD
        assert result["decode_pcc"] >= accepted.BATCH_DECODE_PCC_THRESHOLD


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_DRAM_QKV_LOGICAL_BATCH1") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_DRAM_QKV_LOGICAL_BATCH1=1 for configured-capacity logical-batch coverage",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("configured_capacity", [2, 32], ids=["capacity2", "capacity32"])
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_exact_activation_dram_qkv_logical_batch1_at_larger_capacity(
    monkeypatch, mesh_device, device_params, configured_capacity, layer_idx, reset_seeds
):
    """Require DRAM10/15 to preserve logical-batch1 semantics at larger capacity."""
    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder import _install_profile_constructor_drains
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_full_local_real_ab import (
        _load_sweep_layer_state_dict,
    )

    _install_profile_constructor_drains(monkeypatch, mesh_device)
    repeats = int(os.environ.get("GPT_OSS_120B_DRAM_QKV_LOGICAL_BATCH1_REPEATS", "1000"))
    assert repeats >= 1000
    config = accepted._config()
    prefill_hidden, decode_hidden, decode_positions, activation_source = _real_batch_two_activations(config, layer_idx)
    prefill_hidden = prefill_hidden[:1].contiguous()
    decode_hidden = decode_hidden[:1].contiguous()
    decode_positions = decode_positions[:1].contiguous()
    state_dict = _load_sweep_layer_state_dict(accepted.REAL_WEIGHT_SNAPSHOT, layer_idx)
    expected_prefill, expected_decode = _real_batch_two_reference(
        config, state_dict, prefill_hidden, decode_hidden, decode_positions, layer_idx
    )
    policies = {
        "dram15": ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY,
        "dram10": _CAPACITY2_DRAM_QKV_10_POLICY,
    }
    selected_names = os.environ.get("GPT_OSS_120B_DRAM_QKV_LOGICAL_BATCH1_POLICIES", "dram15,dram10").split(",")
    if (
        not selected_names
        or len(set(selected_names)) != len(selected_names)
        or any(name not in policies for name in selected_names)
    ):
        raise ValueError(f"invalid logical-batch1 policy selection: {selected_names}")
    results = {}
    for name in selected_names:
        policy = policies[name]
        results[name] = _measure_exact_activation_policy(
            mesh_device,
            config,
            state_dict,
            layer_idx,
            prefill_hidden,
            decode_hidden,
            decode_positions,
            expected_prefill,
            expected_decode,
            activation_source,
            policy,
            repeats,
            label=f"logical_batch1_{name}",
            configured_capacity=configured_capacity,
        )
    summary_fields = " ".join(
        f"{name}_decode_pcc={results[name]['decode_pcc']:.9f} "
        f"{name}_traced_decode_wall_ms={results[name]['traced_decode_wall_ms']:.9f}"
        for name in selected_names
    )
    print(
        "DRAM_QKV_LOGICAL_BATCH1_SUMMARY "
        f"revision={_FULL_LOCAL_CHECKPOINT_REVISION} layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"logical_batch=1 configured_capacity={configured_capacity} policies={','.join(selected_names)} "
        f"{summary_fields}"
    )
    for result in results.values():
        assert result["first_second_deterministic"]
        assert result["after_replays_deterministic"]
        assert result["prefill_pcc"] >= accepted.PREFILL_PCC_THRESHOLD
        assert result["decode_pcc"] >= accepted.BATCH_DECODE_PCC_THRESHOLD


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_DRAM_OUTPUT_GEOMETRY_SWEEP") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_DRAM_OUTPUT_GEOMETRY_SWEEP=1 for the cumulative output sweep",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@pytest.mark.parametrize(
    "candidate_policy",
    _EXPERIMENTAL_DRAM10_OUTPUT_PROJECTION_POLICIES + _EXPERIMENTAL_DRAM15_OUTPUT_PROJECTION_POLICIES,
    ids=lambda policy: policy.name,
)
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_exact_activation_dram_output_geometry_sweep(
    monkeypatch, mesh_device, device_params, candidate_policy, layer_idx, reset_seeds
):
    """Sweep legal output projections cumulatively on capacity-2 DRAM QKV."""
    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder import _install_profile_constructor_drains
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_full_local_real_ab import (
        _load_sweep_layer_state_dict,
    )

    _install_profile_constructor_drains(monkeypatch, mesh_device)
    repeats = int(os.environ.get("GPT_OSS_120B_DRAM_OUTPUT_GEOMETRY_SWEEP_REPEATS", "1000"))
    assert repeats >= 1000
    config = accepted._config()
    prefill_hidden, decode_hidden, decode_positions, activation_source = _real_batch_two_activations(config, layer_idx)
    state_dict = _load_sweep_layer_state_dict(accepted.REAL_WEIGHT_SNAPSHOT, layer_idx)
    expected_prefill, expected_decode = _real_batch_two_reference(
        config, state_dict, prefill_hidden, decode_hidden, decode_positions, layer_idx
    )
    base_policy = (
        _CAPACITY2_DRAM_QKV_10_POLICY
        if candidate_policy.dram_qkv_geometry.name == "10c_ibw9_pcn16"
        else ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY
    )
    order_name = os.environ.get("GPT_OSS_120B_DRAM_OUTPUT_GEOMETRY_SWEEP_ORDER", "reference,candidate")
    order = order_name.split(",")
    if sorted(order) != ["candidate", "reference"]:
        raise ValueError(
            "GPT_OSS_120B_DRAM_OUTPUT_GEOMETRY_SWEEP_ORDER must be 'reference,candidate' or "
            f"'candidate,reference', got {order_name!r}"
        )
    requested = {"reference": base_policy, "candidate": candidate_policy}
    results = {
        name: _measure_exact_activation_policy(
            mesh_device,
            config,
            state_dict,
            layer_idx,
            prefill_hidden,
            decode_hidden,
            decode_positions,
            expected_prefill,
            expected_decode,
            activation_source,
            requested[name],
            repeats,
            label=f"dram_output_geometry_{name}",
        )
        for name in order
    }
    reference, candidate = results["reference"], results["candidate"]
    print(
        "DRAM_OUTPUT_GEOMETRY_SUMMARY "
        f"revision={_FULL_LOCAL_CHECKPOINT_REVISION} layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"order={order_name} base_qkv_geometry={base_policy.dram_qkv_geometry.name} "
        f"candidate={candidate_policy.name} "
        f"output_geometry={candidate_policy.output_projection_geometry.name} "
        f"reference_traced_decode_wall_ms={reference['traced_decode_wall_ms']:.9f} "
        f"candidate_traced_decode_wall_ms={candidate['traced_decode_wall_ms']:.9f} "
        f"candidate_vs_reference_decode_pct="
        f"{100 * (candidate['traced_decode_wall_ms'] / reference['traced_decode_wall_ms'] - 1):.3f}"
    )
    for result in results.values():
        assert result["first_second_deterministic"]
        assert result["after_replays_deterministic"]
        assert result["prefill_pcc"] >= accepted.PREFILL_PCC_THRESHOLD
        assert result["decode_pcc"] >= accepted.BATCH_DECODE_PCC_THRESHOLD


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_PACKED_VS_SEPARATE_FINAL") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_PACKED_VS_SEPARATE_FINAL=1 for the cumulative packed/separate A/B",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("configured_capacity", [1, 2], ids=["batch1", "batch2"])
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_exact_activation_packed_vs_separate_final_topology(
    monkeypatch, mesh_device, device_params, configured_capacity, layer_idx, reset_seeds
):
    """Compare packed QKV with the closest legal separate final-topology path."""
    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder import _install_profile_constructor_drains
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_full_local_real_ab import (
        _load_sweep_layer_state_dict,
    )

    _install_profile_constructor_drains(monkeypatch, mesh_device)
    repeats = int(os.environ.get("GPT_OSS_120B_PACKED_VS_SEPARATE_FINAL_REPEATS", "1000"))
    assert repeats >= 1000
    config = accepted._config()
    prefill_hidden, decode_hidden, decode_positions, activation_source = _real_batch_two_activations(config, layer_idx)
    prefill_hidden = prefill_hidden[:configured_capacity].contiguous()
    decode_hidden = decode_hidden[:configured_capacity].contiguous()
    decode_positions = decode_positions[:configured_capacity].contiguous()
    state_dict = _load_sweep_layer_state_dict(accepted.REAL_WEIGHT_SNAPSHOT, layer_idx)
    expected_prefill, expected_decode = _real_batch_two_reference(
        config, state_dict, prefill_hidden, decode_hidden, decode_positions, layer_idx
    )
    separate_policy = (
        ATTENTION_BFP8_SEPARATE_QKV_POLICY if configured_capacity == 1 else ATTENTION_BFP4_LOFI_SEPARATE_QKV_POLICY
    )
    order_name = os.environ.get("GPT_OSS_120B_PACKED_VS_SEPARATE_FINAL_ORDER", "packed,separate")
    order = order_name.split(",")
    if sorted(order) != ["packed", "separate"]:
        raise ValueError(
            "GPT_OSS_120B_PACKED_VS_SEPARATE_FINAL_ORDER must be 'packed,separate' or 'separate,packed', "
            f"got {order_name!r}"
        )
    requested = {"packed": None, "separate": separate_policy}
    results = {
        name: _measure_exact_activation_policy(
            mesh_device,
            config,
            state_dict,
            layer_idx,
            prefill_hidden,
            decode_hidden,
            decode_positions,
            expected_prefill,
            expected_decode,
            activation_source,
            requested[name],
            repeats,
            label=f"packed_vs_separate_{name}",
            configured_capacity=configured_capacity,
        )
        for name in order
    }
    packed, separate = results["packed"], results["separate"]
    print(
        "PACKED_VS_SEPARATE_FINAL_SUMMARY "
        f"revision={_FULL_LOCAL_CHECKPOINT_REVISION} layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"logical_batch={configured_capacity} configured_capacity={configured_capacity} order={order_name} "
        f"packed_policy={_resolve_policy(None, configured_capacity).name} "
        f"separate_policy={separate_policy.name} output_geometry={separate_policy.output_projection_geometry.name} "
        f"packed_decode_pcc={packed['decode_pcc']:.9f} separate_decode_pcc={separate['decode_pcc']:.9f} "
        f"packed_traced_decode_wall_ms={packed['traced_decode_wall_ms']:.9f} "
        f"separate_traced_decode_wall_ms={separate['traced_decode_wall_ms']:.9f} "
        f"separate_vs_packed_decode_pct="
        f"{100 * (separate['traced_decode_wall_ms'] / packed['traced_decode_wall_ms'] - 1):.3f}"
    )
    for result in results.values():
        assert result["first_second_deterministic"]
        assert result["after_replays_deterministic"]
        assert result["prefill_pcc"] >= accepted.PREFILL_PCC_THRESHOLD
        assert result["decode_pcc"] >= accepted.BATCH_DECODE_PCC_THRESHOLD


_PACKED_WHOLE_TRACE_BASELINE_MS = {
    # Existing exact-activation, 1000-replay packed measurements from
    # optimized_autofix_stage_review_packed_vs_separate_final_packed_first.log.gz.
    (1, 0): 0.541222647,
    (1, 1): 0.541824298,
    (2, 0): 0.778223644,
    (2, 1): 0.727147289,
}


class _StatelessZeroMLP:
    """Common low-memory shell component; attention is the only A/B variable."""

    def __call__(self, hidden_states, *, is_decode):
        del is_decode
        return ttnn.zeros_like(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _load_exact_attention_topology_state(snapshot_path, layer_idx):
    """Load exact attention/norm tensors without materializing any expert tensor."""
    snapshot = Path(snapshot_path)
    assert snapshot.name == _FULL_LOCAL_CHECKPOINT_REVISION
    with (snapshot / "model.safetensors.index.json").open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]
    prefix = f"model.layers.{layer_idx}."
    local_names = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "self_attn.o_proj.bias",
        "self_attn.sinks",
    )
    state_dict = {}
    for local_name in local_names:
        checkpoint_name = prefix + local_name
        with safe_open(snapshot / weight_map[checkpoint_name], framework="pt", device="cpu") as checkpoint:
            state_dict[checkpoint_name] = checkpoint.get_tensor(checkpoint_name)

    # _local_layer_state_dict validates the full public decoder contract before
    # _OptimizedMLP is constructed. These sentinels are never read because the
    # test injects one shared stateless MLP object into both decoder shells.
    sentinel = torch.empty(0, dtype=torch.bfloat16)
    for local_name in (
        "mlp.router.weight",
        "mlp.router.bias",
        "mlp.experts.gate_up_proj",
        "mlp.experts.gate_up_proj_bias",
        "mlp.experts.down_proj",
        "mlp.experts.down_proj_bias",
    ):
        state_dict[prefix + local_name] = sentinel
    return state_dict


def _exact_prompt_embedding_activations(config, configured_capacity, layer_idx):
    """Use checkpoint embedding rows directly, including as the layer-1 proxy."""
    snapshot = Path(accepted.REAL_WEIGHT_SNAPSHOT)
    with (snapshot / "model.safetensors.index.json").open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]
    embedding_key = "model.embed_tokens.weight"
    token_ids = torch.tensor(_BATCH_TWO_REAL_TOKEN_IDS[:configured_capacity], dtype=torch.long)
    with safe_open(snapshot / weight_map[embedding_key], framework="pt", device="cpu") as checkpoint:
        embedding_slice = checkpoint.get_slice(embedding_key)
        embedded = torch.stack([embedding_slice[int(token)] for token in token_ids.flatten()]).reshape(
            *token_ids.shape, config.hidden_size
        )
    # Decode immediately beyond the populated prompt so the paged-cache audit
    # observes a real mutation instead of rewriting an identical prompt slot.
    decode_positions = torch.arange(token_ids.shape[1], token_ids.shape[1] + configured_capacity, dtype=torch.long)
    decode_hidden = torch.stack([embedded[user, -(user + 1)] for user in range(configured_capacity)]).unsqueeze(1)
    activation_source = (
        "exact_prompt_embedding" if layer_idx == 0 else "exact_prompt_embedding_layer1_proxy_not_post_layer0"
    )
    print(
        "LOW_MEMORY_EXACT_ATTENTION_ACTIVATIONS "
        f"revision={snapshot.name} layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"activation_source={activation_source} logical_batch={configured_capacity} "
        f"sequence={token_ids.shape[1]} decode_positions={decode_positions.tolist()}"
    )
    return embedded.contiguous(), decode_hidden.contiguous(), decode_positions, activation_source


def _cache_block_to_torch(cache, block_idx):
    shape = tuple(cache.shape)
    block = ttnn.slice(
        cache,
        starts=[block_idx, 0, 0, 0],
        ends=[block_idx + 1, shape[1], shape[2], shape[3]],
        steps=[1, 1, 1, 1],
    )
    host = ttnn.to_torch(block).clone()
    block.deallocate(True)
    return host


def _measure_exact_attention_topology_shell(
    mesh_device,
    config,
    state_dict,
    layer_idx,
    prefill_hidden,
    decode_hidden,
    decode_positions,
    activation_source,
    requested_policy,
    shared_mlp,
    repeats,
    *,
    label,
):
    """Measure the exact attention/norm graph inside an identical zero-MLP decoder shell."""
    from models.autoports.openai_gpt_oss_120b.tt import optimized_decoder as optimized_decoder_module

    batch_size, sequence_length, hidden_size = prefill_hidden.shape
    effective_policy = _resolve_policy(requested_policy, batch_size)
    original_mlp = optimized_decoder_module._OptimizedMLP
    optimized_decoder_module._OptimizedMLP = lambda *args, **kwargs: shared_mlp
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_batch_size=batch_size,
            max_context_length=config.max_position_embeddings,
            page_size=accepted.PAGE_SIZE,
            tensor_cache_path=Path(
                os.environ.get(
                    "GPT_OSS_120B_TENSOR_CACHE",
                    "/tmp/gpt_oss_120b_functional_decoder_tensor_cache",
                )
            )
            / f"layer_{layer_idx}",
            calibrated_checkpoint_revision=_FULL_LOCAL_CHECKPOINT_REVISION,
            policy=requested_policy,
        )
    finally:
        optimized_decoder_module._OptimizedMLP = original_mlp
    assert decoder.mlp is shared_mlp
    assert decoder.policy is effective_policy
    assert decoder.self_attn.weights.wqkv.dtype == effective_policy.attention_weight_dtype
    assert decoder.self_attn.weights.o_proj.dtype == effective_policy.attention_weight_dtype
    assert (
        decoder.self_attn.decode_projection_compute_kernel_config.math_fidelity == effective_policy.decode_math_fidelity
    )
    assert decoder.self_attn.decode_output_program_config is not None
    if effective_policy.separate_qkv:
        assert effective_policy.dram_sharded_qkv
        assert decoder.self_attn.decode_separate_qkv_to_interleaved
        assert decoder.self_attn.decode_separate_qkv_program_configs is not None
        assert tuple(config.per_core_N for config in decoder.self_attn.decode_separate_qkv_program_configs) == (9, 2, 2)

    page_table = accepted._page_table(
        mesh_device,
        config.max_position_embeddings,
        batch_size=batch_size,
        seed=71 + layer_idx,
    )
    page_table_host = ttnn.to_torch(page_table).reshape(batch_size, -1)
    selected_blocks = [
        int(page_table_host[user, int(position) // accepted.PAGE_SIZE].item())
        for user, position in enumerate(decode_positions.tolist())
    ]
    selected_set = set(selected_blocks)
    unselected_block = next(
        int(block) for block in page_table_host.flatten().tolist() if int(block) not in selected_set
    )
    positions = torch.arange(sequence_length, dtype=torch.long)
    prefill_rope = accepted._rope_tensors(config, mesh_device, positions, decode=False)
    decode_rope = accepted._rope_tensors(config, mesh_device, decode_positions, decode=True)
    tt_prefill_hidden = ttnn.from_torch(
        prefill_hidden.reshape(1, batch_size, sequence_length, hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_decode_hidden = ttnn.from_torch(
        decode_hidden.reshape(1, 1, batch_size, hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_current = ttnn.from_torch(
        decode_positions.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    warm_prefill = measured_prefill = eager_decode = traced_decode = None
    trace_id = None
    try:
        warm_prefill = decoder.prefill_forward(
            tt_prefill_hidden,
            position_embeddings=prefill_rope,
            page_table=page_table,
            batch_size=batch_size,
        )
        ttnn.synchronize_device(mesh_device)
        warm_prefill.deallocate(True)
        warm_prefill = None

        started = time.perf_counter()
        measured_prefill = decoder.prefill_forward(
            tt_prefill_hidden,
            position_embeddings=prefill_rope,
            page_table=page_table,
            batch_size=batch_size,
        )
        ttnn.synchronize_device(mesh_device)
        prefill_wall_ms = 1000 * (time.perf_counter() - started)
        prefill_output = accepted._to_host_batch(measured_prefill).clone()

        caches = tuple(decoder.self_attn.kv_cache)
        selected_before = {
            block: tuple(_cache_block_to_torch(cache, block) for cache in caches) for block in selected_blocks
        }
        unselected_before = tuple(_cache_block_to_torch(cache, unselected_block) for cache in caches)
        eager_decode = decoder.decode_forward(
            tt_decode_hidden,
            position_embeddings=decode_rope,
            current_position=tt_current,
            page_table=page_table,
            batch_size=batch_size,
        )
        ttnn.synchronize_device(mesh_device)
        selected_after = {
            block: tuple(_cache_block_to_torch(cache, block) for cache in caches) for block in selected_blocks
        }
        unselected_after = tuple(_cache_block_to_torch(cache, unselected_block) for cache in caches)
        selected_changed = {
            block: tuple(
                not torch.equal(before, after) for before, after in zip(selected_before[block], selected_after[block])
            )
            for block in selected_blocks
        }
        unselected_equal = tuple(
            torch.equal(before, after) for before, after in zip(unselected_before, unselected_after)
        )
        assert all(all(changed) for changed in selected_changed.values())
        assert all(unselected_equal)
        eager_decode.deallocate(True)
        eager_decode = None

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            traced_decode = decoder.decode_forward(
                tt_decode_hidden,
                position_embeddings=decode_rope,
                current_position=tt_current,
                page_table=page_table,
                batch_size=batch_size,
            )
        finally:
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        first_replay = accepted._to_host(traced_decode)[:batch_size].clone()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        second_replay = accepted._to_host(traced_decode)[:batch_size].clone()
        started = time.perf_counter()
        for _ in range(repeats):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        traced_decode_wall_ms = 1000 * (time.perf_counter() - started) / repeats
        after_replays = accepted._to_host(traced_decode)[:batch_size].clone()
        result = {
            "prefill_output": prefill_output,
            "decode_output": after_replays,
            "prefill_wall_ms": prefill_wall_ms,
            "traced_decode_wall_ms": traced_decode_wall_ms,
            "first_second_deterministic": torch.equal(first_replay, second_replay),
            "after_replays_deterministic": torch.equal(first_replay, after_replays),
            "selected_changed": selected_changed,
            "unselected_equal": unselected_equal,
        }
        print(
            "LOW_MEMORY_EXACT_ATTENTION_MEASURE "
            f"label={label} revision={_FULL_LOCAL_CHECKPOINT_REVISION} layer={layer_idx} "
            f"type={config.layer_types[layer_idx]} activation_source={activation_source} "
            f"logical_batch={batch_size} configured_capacity={batch_size} sequence={sequence_length} "
            f"policy={effective_policy.name} repeats={repeats} "
            f"warmed_prefill_wall_ms={prefill_wall_ms:.9f} "
            f"traced_decode_wall_ms={traced_decode_wall_ms:.9f} "
            f"first_second_deterministic={result['first_second_deterministic']} "
            f"after_replays_deterministic={result['after_replays_deterministic']} "
            f"selected_blocks={selected_blocks} selected_changed={selected_changed} "
            f"unselected_block={unselected_block} unselected_equal={unselected_equal}"
        )
        return result
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)
        for tensor in (warm_prefill, measured_prefill, eager_decode, traced_decode):
            if tensor is not None:
                tensor.deallocate(True)
        for tensor in (*prefill_rope, *decode_rope, tt_prefill_hidden, tt_decode_hidden, tt_current, page_table):
            if tensor is not None:
                tensor.deallocate(True)
        del decoder
        gc.collect()


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_DRAM_SEPARATE_QKV_LOW_MEMORY") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_DRAM_SEPARATE_QKV_LOW_MEMORY=1 for the low-memory exact-attention A/B",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("configured_capacity", [1, 2], ids=["batch1", "batch2"])
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_low_memory_exact_attention_packed_vs_dram_separate_qkv(
    monkeypatch, mesh_device, device_params, configured_capacity, layer_idx, reset_seeds
):
    """Compare cumulative packed/DRAM-separate QKV without allocating experts."""
    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder import _install_profile_constructor_drains

    _install_profile_constructor_drains(monkeypatch, mesh_device)
    repeats = int(os.environ.get("GPT_OSS_120B_DRAM_SEPARATE_QKV_LOW_MEMORY_REPEATS", "1000"))
    assert repeats >= 1000
    config = accepted._config()
    state_dict = _load_exact_attention_topology_state(accepted.REAL_WEIGHT_SNAPSHOT, layer_idx)
    prefill_hidden, decode_hidden, decode_positions, activation_source = _exact_prompt_embedding_activations(
        config, configured_capacity, layer_idx
    )
    separate_policy = (
        _BFP8_DRAM_SEPARATE_QKV_CONTROL if configured_capacity == 1 else _BFP4_LOFI_DRAM_SEPARATE_QKV_CONTROL
    )
    order_name = os.environ.get("GPT_OSS_120B_DRAM_SEPARATE_QKV_LOW_MEMORY_ORDER", "packed,separate")
    order = order_name.split(",")
    if sorted(order) != ["packed", "separate"]:
        raise ValueError(
            "GPT_OSS_120B_DRAM_SEPARATE_QKV_LOW_MEMORY_ORDER must be 'packed,separate' or "
            f"'separate,packed', got {order_name!r}"
        )
    requested = {"packed": None, "separate": separate_policy}
    shared_mlp = _StatelessZeroMLP()
    results = {
        name: _measure_exact_attention_topology_shell(
            mesh_device,
            config,
            state_dict,
            layer_idx,
            prefill_hidden,
            decode_hidden,
            decode_positions,
            activation_source,
            requested[name],
            shared_mlp,
            repeats,
            label=name,
        )
        for name in order
    }
    packed, separate = results["packed"], results["separate"]
    _, prefill_pcc = comp_pcc(packed["prefill_output"].float(), separate["prefill_output"].float(), 0.0)
    _, decode_pcc = comp_pcc(packed["decode_output"].float(), separate["decode_output"].float(), 0.0)
    decode_delta_ms = separate["traced_decode_wall_ms"] - packed["traced_decode_wall_ms"]
    packed_whole_trace_ms = _PACKED_WHOLE_TRACE_BASELINE_MS[(configured_capacity, layer_idx)]
    estimated_separate_whole_trace_ms = packed_whole_trace_ms + decode_delta_ms
    print(
        "LOW_MEMORY_EXACT_ATTENTION_SUMMARY "
        f"revision={_FULL_LOCAL_CHECKPOINT_REVISION} layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"activation_source={activation_source} logical_batch={configured_capacity} "
        f"configured_capacity={configured_capacity} order={order_name} "
        f"packed_policy={_resolve_policy(None, configured_capacity).name} separate_policy={separate_policy.name} "
        f"prefill_pcc_vs_packed={prefill_pcc:.9f} decode_pcc_vs_packed={decode_pcc:.9f} "
        f"packed_shell_decode_ms={packed['traced_decode_wall_ms']:.9f} "
        f"separate_shell_decode_ms={separate['traced_decode_wall_ms']:.9f} "
        f"attention_topology_delta_ms={decode_delta_ms:.9f} "
        f"existing_packed_whole_trace_ms={packed_whole_trace_ms:.9f} "
        f"estimated_separate_whole_trace_ms={estimated_separate_whole_trace_ms:.9f} "
        f"estimate_only=True decision_evidence=measured_shell_delta_and_direct_pcc "
        f"estimated_separate_vs_packed_pct={100 * decode_delta_ms / packed_whole_trace_ms:.3f}"
    )
    assert prefill_pcc >= accepted.PREFILL_PCC_THRESHOLD
    assert decode_pcc >= accepted.BATCH_DECODE_PCC_THRESHOLD
    for result in results.values():
        assert result["first_second_deterministic"]
        assert result["after_replays_deterministic"]
        assert all(all(changed) for changed in result["selected_changed"].values())
        assert all(result["unselected_equal"])


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_RUN_BATCH32") != "1",
    reason="set GPT_OSS_120B_RUN_BATCH32=1 for the batch-32 capacity gate",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_batch_32_paged_prefill_and_traced_decode_capacity(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_synthetic_gate(
        monkeypatch,
        accepted.test_batch_32_paged_prefill_and_traced_decode_capacity,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_DRAM_QKV_10_BATCH32") != "1",
    reason="set GPT_OSS_120B_DRAM_QKV_10_BATCH32=1 and GPT_OSS_120B_RUN_BATCH32=1 for DRAM10 qualification",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_dram_qkv_10_batch_32_capacity_qualification(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_synthetic_gate(
        monkeypatch,
        accepted.test_batch_32_paged_prefill_and_traced_decode_capacity,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        policy=_CAPACITY2_DRAM_QKV_10_POLICY,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_LOGICAL_BATCH1_CAPACITY32_LIGHTWEIGHT") != "1",
    reason="set GPT_OSS_120B_LOGICAL_BATCH1_CAPACITY32_LIGHTWEIGHT=1 for the bounded runtime contract gate",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_logical_batch1_at_configured_capacity32_lightweight(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_synthetic_gate(
        monkeypatch,
        accepted.test_paged_prefill_and_traced_decode,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        policy=ATTENTION_BFP4_LOFI_DRAM_SHARDED_POLICY,
        configured_capacity=32,
        logical_batch=1,
        finite_only=True,
        audit_paged_cache=True,
    )


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_traced_decode_at_advertised_context_limit(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_synthetic_gate(
        monkeypatch,
        accepted.test_traced_decode_at_advertised_context_limit,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_RUN_CHUNK_BOUNDARIES") != "1",
    reason="set GPT_OSS_120B_RUN_CHUNK_BOUNDARIES=1 for the 4095/4096/4097 gate",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_prefill_chunk_boundaries(monkeypatch, mesh_device, device_params, layer_idx, reset_seeds):
    _run_synthetic_gate(
        monkeypatch,
        accepted.test_prefill_chunk_boundaries,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_RUN_MAX_PREFILL") != "1",
    reason="set GPT_OSS_120B_RUN_MAX_PREFILL=1 for the 131071/131072 gate",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_prefill_at_advertised_context_limit(monkeypatch, mesh_device, device_params, layer_idx, reset_seeds):
    _run_synthetic_gate(
        monkeypatch,
        accepted.test_prefill_at_advertised_context_limit,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
    )


def test_optimized_attention_weight_cache_is_dtype_specific():
    assert _attention_weight_cache_namespace(_resolve_policy(None, 1)) == "attention_bfp8"
    assert _attention_weight_cache_namespace(_resolve_policy(None, 2)) == "attention_bfp4"
    assert _attention_weight_cache_namespace(_resolve_policy(ATTENTION_BFP8_POLICY, 2)) == "attention_bfp8"
    assert _resolve_policy(ATTENTION_BF16_CONTROL, 2) is ATTENTION_BF16_CONTROL
    assert _attention_weight_cache_namespace(ATTENTION_BF16_CONTROL) == "attention_bf16"
    assert _attention_weight_cache_namespace(ATTENTION_BFP4_POLICY) == "attention_bfp4"


_PERFORMANCE_POLICIES = [
    None,
    ATTENTION_BF16_CONTROL,
    ATTENTION_BFP8_POLICY,
    ATTENTION_BFP8_LOFI_POLICY,
    ATTENTION_BFP8_LOFI_OUTACT_POLICY,
    ATTENTION_BFP8_OUTACT_POLICY,
    ATTENTION_BFP8_DRAM_SHARDED_POLICY,
    ATTENTION_BFP8_SEPARATE_QKV_POLICY,
    ATTENTION_BFP4_POLICY,
    ATTENTION_BFP4_HIFI2_OUTACT_POLICY,
    ATTENTION_BFP4_LOFI_POLICY,
    ATTENTION_BFP4_LOFI_OUTACT_POLICY,
]
if (
    _SELECTED_EXPERIMENTAL_PROJECTION_POLICY is not None
    and _SELECTED_EXPERIMENTAL_PROJECTION_POLICY not in _PERFORMANCE_POLICIES
):
    _PERFORMANCE_POLICIES.append(_SELECTED_EXPERIMENTAL_PROJECTION_POLICY)


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_OPTIMIZED_PERF") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_OPTIMIZED_PERF=1 and GPT_OSS_120B_SNAPSHOT for the exact-checkpoint A/B",
)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@pytest.mark.parametrize("policy", _PERFORMANCE_POLICIES, ids=lambda policy: policy.name if policy else "automatic")
@pytest.mark.parametrize("configured_capacity", [1, 2], ids=["capacity1", "capacity2"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_optimized_real_weight_warmed_prefill_and_traced_decode_performance(
    monkeypatch, mesh_device, device_params, configured_capacity, layer_idx, policy, reset_seeds
):
    """Measure whole-decoder precision candidates under one real-weight trace harness."""
    del device_params, reset_seeds
    if configured_capacity != 1 and policy is not None:
        pytest.skip("capacity-2 profiling is the final automatic policy, not the historical candidate sweep")
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder import _install_profile_constructor_drains
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_full_local_real_ab import (
        _load_sweep_layer_state_dict,
    )

    _install_profile_constructor_drains(monkeypatch, mesh_device)
    config = accepted._config()
    constructor_kwargs = dict(
        state_dict=_load_sweep_layer_state_dict(accepted.REAL_WEIGHT_SNAPSHOT, layer_idx),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch_size=configured_capacity,
        max_context_length=config.max_position_embeddings,
        page_size=accepted.PAGE_SIZE,
        tensor_cache_path=Path(
            os.environ.get("GPT_OSS_120B_TENSOR_CACHE", "/tmp/gpt_oss_120b_functional_decoder_tensor_cache")
        )
        / f"layer_{layer_idx}",
        calibrated_checkpoint_revision=_FULL_LOCAL_CHECKPOINT_REVISION,
    )
    if policy is not None:
        constructor_kwargs["policy"] = policy
    decoder = OptimizedDecoder.from_state_dict(**constructor_kwargs)
    effective_policy = _resolve_policy(policy, configured_capacity)
    assert decoder.policy is effective_policy
    assert decoder.requested_policy is policy
    assert decoder.mlp.decode_uses_full_local
    assert decoder.self_attn.weights.wqkv.dtype == effective_policy.attention_weight_dtype
    assert decoder.self_attn.weights.o_proj.dtype == effective_policy.attention_weight_dtype
    assert (
        decoder.self_attn.decode_projection_compute_kernel_config.math_fidelity == effective_policy.decode_math_fidelity
    )
    assert effective_policy.output_projection_geometry is not None
    output_geometry = effective_policy.output_projection_geometry
    output_config = decoder.self_attn.decode_output_program_config
    assert output_config.compute_with_storage_grid_size == ttnn.CoreCoord(*output_geometry.core_grid)
    assert output_config.in0_block_w == output_geometry.in0_block_w
    assert output_config.per_core_N == output_geometry.per_core_n
    assert output_config.out_subblock_w == output_geometry.out_subblock_w
    if configured_capacity == 2 and policy is None:
        assert effective_policy.dram_sharded_qkv
        assert effective_policy.dram_qkv_geometry is not None
        assert (
            type(decoder.self_attn.decode_qkv_program_config)
            is ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
        )
        assert decoder.self_attn.decode_qkv_input_memory_config is not None
        assert decoder.self_attn.decode_qkv_to_interleaved

    sequence_length = 128
    page_table = accepted._page_table(
        mesh_device,
        config.max_position_embeddings,
        batch_size=configured_capacity,
        seed=12_001 + layer_idx,
    )
    generator = torch.Generator().manual_seed(120_812 + layer_idx)
    positions = torch.arange(sequence_length, dtype=torch.long)
    hidden = (torch.randn((configured_capacity, sequence_length, config.hidden_size), generator=generator) * 0.02).to(
        torch.bfloat16
    )
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, configured_capacity, sequence_length, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    prefill_rope = accepted._rope_tensors(config, mesh_device, positions, decode=False)
    warm_prefill = decoder.prefill_forward(
        tt_hidden,
        position_embeddings=prefill_rope,
        page_table=page_table,
        batch_size=configured_capacity,
    )
    ttnn.synchronize_device(mesh_device)
    assert torch.isfinite(accepted._to_host(warm_prefill)[:sequence_length]).all()
    warm_prefill.deallocate(True)
    ttnn.ReadDeviceProfiler(mesh_device)

    signpost("PERF_PREFILL")
    started = time.perf_counter()
    measured_prefill = decoder.prefill_forward(
        tt_hidden,
        position_embeddings=prefill_rope,
        page_table=page_table,
        batch_size=configured_capacity,
    )
    ttnn.synchronize_device(mesh_device)
    prefill_wall_ms = 1000 * (time.perf_counter() - started)
    signpost("PERF_PREFILL_END")
    assert torch.isfinite(accepted._to_host(measured_prefill)[:sequence_length]).all()
    measured_prefill.deallocate(True)
    ttnn.ReadDeviceProfiler(mesh_device)

    decode_positions = torch.full((configured_capacity,), sequence_length, dtype=torch.long)
    decode_hidden = (torch.randn((configured_capacity, 1, config.hidden_size), generator=generator) * 0.02).to(
        torch.bfloat16
    )
    tt_decode_hidden = ttnn.from_torch(
        decode_hidden.reshape(1, 1, configured_capacity, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_current = ttnn.from_torch(
        decode_positions.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    decode_rope = accepted._rope_tensors(config, mesh_device, decode_positions, decode=True)
    eager = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
        batch_size=configured_capacity,
    )
    ttnn.synchronize_device(mesh_device)
    eager.deallocate(True)
    ttnn.ReadDeviceProfiler(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        output = decoder.decode_forward(
            tt_decode_hidden,
            position_embeddings=decode_rope,
            current_position=tt_current,
            page_table=page_table,
            batch_size=configured_capacity,
        )
    finally:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    first = accepted._to_host(output).clone()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    assert torch.equal(accepted._to_host(output), first)

    repeats = int(os.environ.get("GPT_OSS_120B_OPTIMIZED_PERF_REPEATS", "1000"))
    signpost("PERF_DECODE")
    started = time.perf_counter()
    for _ in range(repeats):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    traced_decode_wall_ms = 1000 * (time.perf_counter() - started) / repeats
    signpost("PERF_DECODE_END")
    ttnn.ReadDeviceProfiler(mesh_device)
    print(
        "OPTIMIZED_WHOLE_PERF "
        f"layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"requested_policy={policy.name if policy is not None else 'automatic'} "
        f"effective_policy={effective_policy.name} "
        f"logical_batch={configured_capacity} configured_capacity={configured_capacity} "
        f"attention_weight_dtype={effective_policy.attention_weight_dtype} sequence={sequence_length} "
        f"decode_projection_fidelity={effective_policy.decode_math_fidelity} "
        f"dram_qkv_geometry={effective_policy.dram_qkv_geometry.name if effective_policy.dram_qkv_geometry else 'none'} "
        f"output_projection_geometry={output_geometry.name} "
        f"warmed_prefill_wall_ms={prefill_wall_ms:.9f} repeats={repeats} "
        f"traced_decode_wall_ms={traced_decode_wall_ms:.9f} deterministic=True"
    )
    ttnn.release_trace(mesh_device, trace_id)
    output.deallocate(True)

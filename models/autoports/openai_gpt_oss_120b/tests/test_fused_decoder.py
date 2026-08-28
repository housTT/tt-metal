# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fused-path variants of the functional decoder's accepted hardware gates.

The functional test module owns the HF/page/cache/trace oracle machinery.  The
small adapter below substitutes :class:`FusedDecoder` at the construction
boundary and then runs those exact gates.  The implementation-source test makes
that substitution auditable and prevents a future functional runtime fallback.
"""

import gc
import inspect
import os
import time
from pathlib import Path

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.openai_gpt_oss_120b.tests import test_functional_decoder as accepted
from models.autoports.openai_gpt_oss_120b.tt import fused_decoder as fused_impl
from models.autoports.openai_gpt_oss_120b.tt.fused_decoder import (
    _FULL_LOCAL_CALIBRATED_RING_SIZE,
    _FULL_LOCAL_CHECKPOINT_REVISION,
    _FULL_LOCAL_DECODE_LAYERS,
    _FULL_LOCAL_MAX_BATCH_SIZE,
    _FULL_LOCAL_REDUCE_OUTPUT_MEMORY_CONFIG,
    FusedDecoder,
    TtRoutedExpert,
    _FusedMLP,
    _use_full_local_decode,
)
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric


def _run_accepted_gate(
    monkeypatch,
    gate,
    mesh_device,
    device_params,
    layer_idx,
    reset_seeds,
    *,
    calibrated_checkpoint_revision=None,
    expect_full_local=None,
    full_local_reduce_output_memory_config=None,
):
    decoder_type = FusedDecoder
    if (
        calibrated_checkpoint_revision is not None
        or expect_full_local is not None
        or full_local_reduce_output_memory_config is not None
    ):

        class RevisionQualifiedFusedDecoder(FusedDecoder):
            @classmethod
            def from_state_dict(cls, *args, **kwargs):
                if calibrated_checkpoint_revision is not None:
                    kwargs["calibrated_checkpoint_revision"] = calibrated_checkpoint_revision
                decoder = super().from_state_dict(*args, **kwargs)
                if expect_full_local is not None:
                    assert decoder.mlp.decode_uses_full_local is expect_full_local
                    assert hasattr(decoder.mlp, "decode_full_local_w0_w1") is expect_full_local
                    assert hasattr(decoder.mlp, "decode_packed_gate_up") is not expect_full_local
                if full_local_reduce_output_memory_config is not None:
                    assert decoder.mlp.decode_uses_full_local
                    decoder.mlp.decode_full_local_reduce_output_memory_config = full_local_reduce_output_memory_config
                return decoder

            def decode_forward(self, *args, **kwargs):
                output = super().decode_forward(*args, **kwargs)
                assert output.layout == ttnn.TILE_LAYOUT
                return output

        decoder_type = RevisionQualifiedFusedDecoder
    monkeypatch.setattr(accepted, "FunctionalDecoder", decoder_type)
    gate(mesh_device, device_params, layer_idx, reset_seeds)


def _run_real_checkpoint_accepted_gate(
    monkeypatch,
    gate,
    mesh_device,
    device_params,
    layer_idx,
    reset_seeds,
    *,
    expect_full_local=True,
    full_local_reduce_output_memory_config=None,
):
    """Run an accepted multi-batch gate with the calibrated checkpoint tensors."""
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
    # The accepted synthetic batch-2 gate creates float32 activations.  Real
    # GPT-OSS checkpoint modules are BF16, so preserve the real-weight gate's
    # activation contract without converting the checkpoint tensors.
    torch_randn = torch.randn
    hidden_size = accepted._config().hidden_size

    def checkpoint_randn(*args, **kwargs):
        value = torch_randn(*args, **kwargs)
        if value.ndim and value.shape[-1] == hidden_size:
            value = value.to(torch.bfloat16)
        return value

    monkeypatch.setattr(torch, "randn", checkpoint_randn)
    _run_accepted_gate(
        monkeypatch,
        gate,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        calibrated_checkpoint_revision=_FULL_LOCAL_CHECKPOINT_REVISION,
        expect_full_local=expect_full_local,
        full_local_reduce_output_memory_config=full_local_reduce_output_memory_config,
    )


def _install_profile_constructor_drains(monkeypatch, mesh_device, *, experts_per_drain=8):
    """Drain setup-only expert conversions before a profiler buffer can fill."""
    original_weights = TtRoutedExpert._convert_and_cache_expert_weights
    original_biases = TtRoutedExpert._convert_expert_biases

    def convert_weights(
        torch_weights,
        experts_per_chip,
        target_mesh,
        weights_dtype,
        cache_path,
        cache_name_prefix,
        device=None,
        **dimensions,
    ):
        if torch_weights is None or device is None:
            return original_weights(
                torch_weights,
                experts_per_chip,
                target_mesh,
                weights_dtype,
                cache_path,
                cache_name_prefix,
                device=device,
                **dimensions,
            )

        converted = ([], [], [])
        for start in range(0, experts_per_chip, experts_per_drain):
            chunk = torch_weights[start : start + experts_per_drain]
            result = original_weights(
                chunk,
                len(chunk),
                target_mesh,
                weights_dtype,
                cache_path,
                f"{cache_name_prefix}.profile_chunk_{start}",
                device=device,
                **dimensions,
            )
            for destination, source in zip(converted, result):
                destination.extend(source)
            ttnn.ReadDeviceProfiler(mesh_device)
        return converted

    def convert_biases(torch_biases, experts_per_chip, target_mesh, bias_dtype=ttnn.bfloat16):
        converted = ([], [], [])
        for start in range(0, experts_per_chip, experts_per_drain):
            chunk = torch_biases[start : start + experts_per_drain]
            result = original_biases(chunk, len(chunk), target_mesh, bias_dtype)
            for destination, source in zip(converted, result):
                destination.extend(source)
            ttnn.ReadDeviceProfiler(mesh_device)
        return converted

    monkeypatch.setattr(TtRoutedExpert, "_convert_and_cache_expert_weights", staticmethod(convert_weights))
    monkeypatch.setattr(TtRoutedExpert, "_convert_expert_biases", staticmethod(convert_biases))


def test_profile_constructor_drains_use_128_distinct_expert_cache_inputs(monkeypatch):
    cache_inputs = []
    forwarded_dimensions = []
    profiler_drains = []
    target_mesh = object()
    device = object()

    def record_weight_conversion(
        torch_weights,
        experts_per_chip,
        requested_mesh,
        weights_dtype,
        cache_path,
        cache_name_prefix,
        device=None,
        **dimensions,
    ):
        del weights_dtype, cache_path
        assert requested_mesh is target_mesh
        assert device is not None
        assert experts_per_chip == len(torch_weights) == 8
        forwarded_dimensions.append(dimensions)
        cache_inputs.extend(
            (f"{cache_name_prefix}.local_{local_expert_idx}", global_expert_idx)
            for local_expert_idx, global_expert_idx in enumerate(torch_weights)
        )
        return tuple(
            [(projection, global_expert_idx) for global_expert_idx in torch_weights]
            for projection in ("gate", "up", "down")
        )

    monkeypatch.setattr(
        TtRoutedExpert,
        "_convert_and_cache_expert_weights",
        staticmethod(record_weight_conversion),
    )
    monkeypatch.setattr(ttnn, "ReadDeviceProfiler", lambda mesh: profiler_drains.append(mesh))
    _install_profile_constructor_drains(monkeypatch, target_mesh)

    converted = TtRoutedExpert._convert_and_cache_expert_weights(
        list(range(128)),
        128,
        target_mesh,
        ttnn.bfloat16,
        Path("/tmp/gpt_oss_120b_profile_cache_identity_test"),
        "fused_experts",
        device=device,
        emb_dim=2880,
        hidden_dim=2880,
    )

    assert len(cache_inputs) == len({cache_input for cache_input, _ in cache_inputs}) == 128
    assert [global_expert_idx for _, global_expert_idx in cache_inputs] == list(range(128))
    assert all(dimensions == {"emb_dim": 2880, "hidden_dim": 2880} for dimensions in forwarded_dimensions)
    assert len(forwarded_dimensions) == len(profiler_drains) == 16
    assert all(mesh is target_mesh for mesh in profiler_drains)
    for projection_idx, projection in enumerate(("gate", "up", "down")):
        assert converted[projection_idx] == [(projection, global_expert_idx) for global_expert_idx in range(128)]


def test_fused_implementation_has_no_functional_runtime_fallback():
    source = inspect.getsource(inspect.getmodule(FusedDecoder))
    assert "class FusedDecoder" in source
    assert "unified_routed_expert_moe" in source
    assert "local_sort_count_regroup" in source
    assert "TtDispatchModule" not in source
    assert "TtCombineModule" not in source
    assert "from models.autoports.openai_gpt_oss_120b.tt.functional_decoder" not in source
    assert "FunctionalDecoder." not in source
    assert "from tests" not in source
    assert _FULL_LOCAL_CHECKPOINT_REVISION == "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
    assert _FULL_LOCAL_CALIBRATED_RING_SIZE == 8
    assert _FULL_LOCAL_MAX_BATCH_SIZE == 2
    assert _FULL_LOCAL_DECODE_LAYERS == {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        21,
        23,
        24,
        25,
        26,
        27,
        29,
        31,
        33,
        34,
        35,
    }
    assert _FULL_LOCAL_REDUCE_OUTPUT_MEMORY_CONFIG == ttnn.DRAM_MEMORY_CONFIG
    profile_drain_source = inspect.getsource(_install_profile_constructor_drains)
    assert "profile_chunk_{start}" in profile_drain_source
    assert "**dimensions" in profile_drain_source
    revision_parameter = inspect.signature(FusedDecoder.from_state_dict).parameters["calibrated_checkpoint_revision"]
    assert revision_parameter.default is None
    assert "calibrated_checkpoint_revision == _FULL_LOCAL_CHECKPOINT_REVISION" in source
    assert "effective_matmul_ring_size(mesh_device) == _FULL_LOCAL_CALIBRATED_RING_SIZE" in source
    assert "quantize_weights_via_host" in source
    assert "ttnn.typecast(bf16, dtype=ttnn.bfloat4_b)" not in source
    assert "checkpoint_allowlisted_full_local_moe_compute" in FusedDecoder.fusion_manifest
    assert "deepseek_moe_fast_reduce_nc_fused" in FusedDecoder.fusion_manifest
    assert "indexed_packed_gate_up_sparse_matmul" in FusedDecoder.fusion_manifest
    assert "indexed_sparse_down_matmul" in FusedDecoder.fusion_manifest
    assert "indexed_bias_embedding" in FusedDecoder.fusion_manifest
    assert "unified_routed_expert_moe" in FusedDecoder.fusion_manifest


def test_full_local_activation_requires_exact_revision_layer_and_ring(monkeypatch):
    fake_mesh = object()
    monkeypatch.setattr(fused_impl, "effective_matmul_ring_size", lambda _: 8)
    assert _use_full_local_decode(fake_mesh, 1, _FULL_LOCAL_CHECKPOINT_REVISION, 1)
    assert _use_full_local_decode(fake_mesh, 1, _FULL_LOCAL_CHECKPOINT_REVISION, 2)
    assert not _use_full_local_decode(fake_mesh, 1, _FULL_LOCAL_CHECKPOINT_REVISION, 0)
    assert not _use_full_local_decode(fake_mesh, 1, _FULL_LOCAL_CHECKPOINT_REVISION, 8)
    assert not _use_full_local_decode(fake_mesh, 1, _FULL_LOCAL_CHECKPOINT_REVISION, 32)
    assert not _use_full_local_decode(fake_mesh, 14, _FULL_LOCAL_CHECKPOINT_REVISION, 1)
    assert not _use_full_local_decode(fake_mesh, 1, "different-revision", 1)
    assert not _use_full_local_decode(fake_mesh, 1, None, 1)
    monkeypatch.setattr(fused_impl, "effective_matmul_ring_size", lambda _: 7)
    assert not _use_full_local_decode(fake_mesh, 1, _FULL_LOCAL_CHECKPOINT_REVISION, 1)


@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_prefill_mlp_non_aligned_deterministic_and_traceable(mesh_device, device_params, reset_seeds):
    """Exercise the fabric-free fused prefill path at a non-aligned length."""
    del device_params, reset_seeds
    config = accepted._config()
    reference = accepted._reference_layer(config, layer_idx=0)
    mlp = _FusedMLP(mesh_device, config, reference.mlp.state_dict(), tensor_cache_path=None)

    logical_tokens = 129
    hidden = torch.randn(1, logical_tokens, config.hidden_size) * 0.02
    with torch.no_grad():
        expected = reference.mlp(hidden)[0]
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, 1, logical_tokens, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    eager = mlp(tt_hidden, is_decode=False)
    ttnn.synchronize_device(mesh_device)
    eager_host = ttnn.to_torch(eager)[0, 0]
    passing, detail = comp_pcc(expected.float(), eager_host.float(), accepted.PREFILL_PCC_THRESHOLD)
    assert passing, f"fused standalone prefill MLP failed: {detail}"
    assert eager.shape[-2] == logical_tokens

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = mlp(tt_hidden, is_decode=False)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    traced_host = ttnn.to_torch(traced)[0, 0]
    ttnn.release_trace(mesh_device, trace_id)
    assert torch.equal(eager_host, traced_host)


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_paged_prefill_and_traced_decode(monkeypatch, mesh_device, device_params, layer_idx, reset_seeds):
    _run_accepted_gate(
        monkeypatch,
        accepted.test_paged_prefill_and_traced_decode,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_PROFILE") != "1",
    reason="set GPT_OSS_120B_PROFILE=1 for warmed Tracy signpost collection",
)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_warmed_prefill_and_traced_decode_performance(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    """Collect fused signpost windows, draining profiler buffers between phases."""
    del device_params, reset_seeds
    _install_profile_constructor_drains(monkeypatch, mesh_device)
    config = accepted._config()
    reference = accepted._reference_layer(config, layer_idx)
    decoder = FusedDecoder.from_state_dict(
        reference.state_dict(),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_context_length=config.max_position_embeddings,
        page_size=accepted.PAGE_SIZE,
    )
    # This legacy synthetic profiler remains the revision-unknown indexed
    # baseline. Exact-revision FullLocal timing is collected by the real path
    # A/B below so random weights are never falsely attested as calibrated.
    assert not decoder.mlp.decode_uses_full_local
    page_table = accepted._page_table(mesh_device, config.max_position_embeddings, seed=53 + layer_idx)
    sequence_length = 128
    positions = torch.arange(sequence_length, dtype=torch.long)
    hidden = torch.randn(1, sequence_length, config.hidden_size) * 0.02
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, 1, sequence_length, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    prefill_rope = accepted._rope_tensors(config, mesh_device, positions, decode=False)

    warm_prefill = decoder.prefill_forward(tt_hidden, position_embeddings=prefill_rope, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    warm_prefill.deallocate(True)
    ttnn.ReadDeviceProfiler(mesh_device)

    signpost("PERF_PREFILL")
    started = time.perf_counter()
    measured_prefill = decoder.prefill_forward(tt_hidden, position_embeddings=prefill_rope, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    prefill_seconds = time.perf_counter() - started
    signpost("PERF_PREFILL_END")
    ttnn.ReadDeviceProfiler(mesh_device)

    decode_positions = torch.tensor([sequence_length], dtype=torch.long)
    decode_hidden = torch.randn(1, 1, config.hidden_size) * 0.02
    tt_decode_hidden = ttnn.from_torch(
        decode_hidden.reshape(1, 1, 1, config.hidden_size),
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
    _ = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    signpost("PERF_DECODE")
    started = time.perf_counter()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    decode_seconds = time.perf_counter() - started
    signpost("PERF_DECODE_END")
    ttnn.ReadDeviceProfiler(mesh_device)
    assert torch.isfinite(accepted._to_host(traced_output)).all()
    ttnn.release_trace(mesh_device, trace_id)
    print(
        f"WARMED_PERF layer={layer_idx} type={config.layer_types[layer_idx]} sequence={sequence_length} "
        f"prefill_wall_seconds={prefill_seconds:.9f} traced_decode_wall_seconds={decode_seconds:.9f}"
    )
    measured_prefill.deallocate(True)


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_LOCAL_WHOLE_PERF") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_FULL_LOCAL_WHOLE_PERF=1 and GPT_OSS_120B_SNAPSHOT for the real path A/B",
)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@pytest.mark.parametrize("decode_path", ["full_local", "indexed"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_real_weight_whole_traced_decode_path_performance(
    monkeypatch, mesh_device, device_params, layer_idx, decode_path, reset_seeds
):
    """Profile exact-checkpoint whole-decoder FullLocal against indexed decode."""
    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_full_local_real_ab import (
        _load_sweep_layer_state_dict,
    )

    _install_profile_constructor_drains(monkeypatch, mesh_device)
    config = accepted._config()
    revision = _FULL_LOCAL_CHECKPOINT_REVISION
    if decode_path == "indexed":
        # Keep the checkpoint/revision inputs identical while forcing only the
        # decode implementation under A/B. The assertions below attest which
        # representation was actually constructed.
        monkeypatch.setattr(fused_impl, "_use_full_local_decode", lambda *args, **kwargs: False)
    decoder = FusedDecoder.from_state_dict(
        _load_sweep_layer_state_dict(accepted.REAL_WEIGHT_SNAPSHOT, layer_idx),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_context_length=config.max_position_embeddings,
        page_size=accepted.PAGE_SIZE,
        tensor_cache_path=Path("/tmp/gpt_oss_120b_full_local_whole_perf") / decode_path / f"layer_{layer_idx}",
        calibrated_checkpoint_revision=revision,
    )
    expect_full_local = decode_path == "full_local"
    assert decoder.calibrated_checkpoint_revision == _FULL_LOCAL_CHECKPOINT_REVISION
    assert decoder.mlp.decode_uses_full_local is expect_full_local
    assert decoder.mlp.decode_full_local_checkpoint_revision == (
        _FULL_LOCAL_CHECKPOINT_REVISION if expect_full_local else None
    )
    assert hasattr(decoder.mlp, "decode_full_local_w0_w1") is expect_full_local
    assert hasattr(decoder.mlp, "decode_packed_gate_up") is not expect_full_local

    page_table = accepted._page_table(mesh_device, config.max_position_embeddings, seed=901 + layer_idx)
    sequence_length = 128
    positions = torch.arange(sequence_length, dtype=torch.long)
    generator = torch.Generator().manual_seed(812_000 + layer_idx)
    hidden = (torch.randn((1, sequence_length, config.hidden_size), generator=generator) * 0.02).to(torch.bfloat16)
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, 1, sequence_length, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    prefill_rope = accepted._rope_tensors(config, mesh_device, positions, decode=False)
    warm_prefill = decoder.prefill_forward(
        tt_hidden,
        position_embeddings=prefill_rope,
        page_table=page_table,
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
    )
    ttnn.synchronize_device(mesh_device)
    prefill_wall_ms = 1000 * (time.perf_counter() - started)
    signpost("PERF_PREFILL_END")
    assert torch.isfinite(accepted._to_host(measured_prefill)[:sequence_length]).all()
    measured_prefill.deallocate(True)
    ttnn.ReadDeviceProfiler(mesh_device)

    decode_positions = torch.tensor([sequence_length], dtype=torch.long)
    decode_hidden = (torch.randn((1, 1, config.hidden_size), generator=generator) * 0.02).to(torch.bfloat16)
    tt_decode_hidden = ttnn.from_torch(
        decode_hidden.reshape(1, 1, 1, config.hidden_size),
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
    )
    ttnn.synchronize_device(mesh_device)
    assert eager.layout == ttnn.TILE_LAYOUT
    eager.deallocate(True)
    ttnn.ReadDeviceProfiler(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        output = decoder.decode_forward(
            tt_decode_hidden,
            position_embeddings=decode_rope,
            current_position=tt_current,
            page_table=page_table,
        )
    finally:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    first = accepted._to_host(output).clone()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    assert torch.equal(accepted._to_host(output), first)

    repeats = int(os.environ.get("GPT_OSS_120B_FULL_LOCAL_WHOLE_PERF_REPEATS", "1000"))
    signpost("PERF_DECODE")
    started = time.perf_counter()
    for _ in range(repeats):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    wall_ms = 1000 * (time.perf_counter() - started) / repeats
    signpost("PERF_DECODE_END")
    ttnn.ReadDeviceProfiler(mesh_device)
    print(
        "FULL_LOCAL_WHOLE_PERF "
        f"checkpoint={accepted.REAL_WEIGHT_SNAPSHOT} layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"checkpoint_revision={revision} path={decode_path} path_attested=True sequence={sequence_length} "
        f"warmed_prefill_wall_ms={prefill_wall_ms:.9f} repeats={repeats} "
        f"traced_decode_wall_ms={wall_ms:.9f} deterministic=True"
    )
    ttnn.release_trace(mesh_device, trace_id)
    output.deallocate(True)


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_batch_two_paged_prefill_and_traced_decode(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_accepted_gate(
        monkeypatch,
        accepted.test_batch_two_paged_prefill_and_traced_decode,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        expect_full_local=False,
    )


@pytest.mark.skipif(
    not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_SNAPSHOT to run the real-checkpoint batch-2 FullLocal gate",
)
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_real_weight_batch_two_full_local_paged_prefill_and_traced_decode(
    monkeypatch, mesh_device, device_params, reset_seeds
):
    _run_real_checkpoint_accepted_gate(
        monkeypatch,
        accepted.test_batch_two_paged_prefill_and_traced_decode,
        mesh_device,
        device_params,
        1,
        reset_seeds,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_RUN_BATCH32") != "1",
    reason="set GPT_OSS_120B_RUN_BATCH32=1 for the batch-32 capacity gate",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_batch_32_paged_prefill_and_traced_decode_capacity(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_accepted_gate(
        monkeypatch,
        accepted.test_batch_32_paged_prefill_and_traced_decode_capacity,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        expect_full_local=False,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_RUN_BATCH32") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_RUN_BATCH32=1 and GPT_OSS_120B_SNAPSHOT for the exact-revision indexed-fallback gate",
)
@pytest.mark.timeout(1200)
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_real_weight_batch_32_indexed_fallback_paged_prefill_and_traced_decode_capacity(
    monkeypatch, mesh_device, device_params, reset_seeds
):
    _run_real_checkpoint_accepted_gate(
        monkeypatch,
        accepted.test_batch_32_paged_prefill_and_traced_decode_capacity,
        mesh_device,
        device_params,
        1,
        reset_seeds,
        expect_full_local=False,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_LOCAL_BATCH_BOUNDARY_SWEEP") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_FULL_LOCAL_BATCH_BOUNDARY_SWEEP=1 and GPT_OSS_120B_SNAPSHOT for the candidate sweep",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "batch_size",
    [int(value) for value in os.environ.get("GPT_OSS_120B_FULL_LOCAL_BATCH_BOUNDARY_BATCHES", "4,8,16").split(",")],
)
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_real_weight_full_local_whole_trace_batch_boundary_assessment(
    monkeypatch, mesh_device, device_params, batch_size, reset_seeds
):
    """Assess the rejected FullLocal construction range above the proven B2 cap."""
    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_full_local_real_ab import (
        _load_sweep_layer_state_dict,
    )

    # Intentionally bypass the production cap only inside this candidate test.
    monkeypatch.setattr(fused_impl, "_FULL_LOCAL_MAX_BATCH_SIZE", 32)
    config = accepted._config()
    layer_idx = 1
    decoder = FusedDecoder.from_state_dict(
        _load_sweep_layer_state_dict(accepted.REAL_WEIGHT_SNAPSHOT, layer_idx),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch_size=batch_size,
        max_context_length=config.max_position_embeddings,
        page_size=accepted.PAGE_SIZE,
        calibrated_checkpoint_revision=_FULL_LOCAL_CHECKPOINT_REVISION,
    )
    assert decoder.mlp.decode_uses_full_local
    page_table = accepted._page_table(
        mesh_device,
        config.max_position_embeddings,
        batch_size=batch_size,
        seed=97 + layer_idx,
    )
    sequence_length = 33
    positions = torch.arange(sequence_length, dtype=torch.long)
    hidden = torch.randn(batch_size, sequence_length, config.hidden_size, dtype=torch.bfloat16) * 0.02
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, batch_size, sequence_length, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    prefill = decoder.prefill_forward(
        tt_hidden,
        position_embeddings=accepted._rope_tensors(config, mesh_device, positions, decode=False),
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.synchronize_device(mesh_device)
    assert torch.isfinite(ttnn.to_torch(prefill)[0, :, -1]).all()

    generator = torch.Generator().manual_seed(307 + layer_idx)
    decode_positions = (torch.randperm(sequence_length, generator=generator) + 1)[:batch_size]
    decode_hidden = torch.randn(batch_size, 1, config.hidden_size, generator=generator, dtype=torch.bfloat16) * 0.02
    tt_decode_hidden = ttnn.from_torch(
        decode_hidden.reshape(1, 1, batch_size, config.hidden_size),
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
        batch_size=batch_size,
    )
    ttnn.synchronize_device(mesh_device)
    eager.deallocate(True)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        output = decoder.decode_forward(
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
    first = accepted._to_host(output)[:batch_size].clone()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    second = accepted._to_host(output)[:batch_size]
    deterministic = torch.equal(first, second)
    delta = (first.float() - second.float()).abs()
    _, pcc = comp_pcc(first.float(), second.float(), 0.999999)
    print(
        f"FULL_LOCAL_WHOLE_BATCH_BOUNDARY batch={batch_size} deterministic={deterministic} "
        f"replay_pcc={pcc} differing_elements={torch.count_nonzero(delta).item()} max_abs={delta.max().item()}"
    )
    ttnn.release_trace(mesh_device, trace_id)
    output.deallocate(True)


@pytest.mark.skipif(
    not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_SNAPSHOT to run the real-weight acceptance gate",
)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_real_weight_paged_prefill_and_traced_decode(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_accepted_gate(
        monkeypatch,
        accepted.test_real_weight_paged_prefill_and_traced_decode,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        calibrated_checkpoint_revision=_FULL_LOCAL_CHECKPOINT_REVISION,
        expect_full_local=layer_idx in _FULL_LOCAL_DECODE_LAYERS,
    )


@pytest.mark.skipif(
    not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_SNAPSHOT to run the arbitrary-layer real-weight gate",
)
@pytest.mark.parametrize("layer_idx", [2], ids=["enabled-sliding"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_real_weight_arbitrary_layer_path_qualification(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    """Qualify an enabled sliding-attention layer beyond the primary layer-1 gate."""
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_full_local_real_ab import (
        _load_sweep_layer_state_dict,
    )

    monkeypatch.setattr(accepted, "load_real_layer_state_dict", _load_sweep_layer_state_dict)
    use_full_local = layer_idx in _FULL_LOCAL_DECODE_LAYERS
    _run_accepted_gate(
        monkeypatch,
        accepted.test_real_weight_paged_prefill_and_traced_decode,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
        calibrated_checkpoint_revision=_FULL_LOCAL_CHECKPOINT_REVISION if use_full_local else None,
        expect_full_local=use_full_local,
    )


@pytest.mark.skipif(
    not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_SNAPSHOT to run the two-layer FullLocal compatibility gate",
)
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_real_weight_two_layer_decode_compatibility_and_output_memory_ab(mesh_device, device_params, reset_seeds):
    """Compose enabled full/sliding decoders and compare L1/DRAM reduce outputs."""
    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_full_local_real_ab import (
        _load_sweep_layer_state_dict,
    )

    config = accepted._config()
    hidden = torch.randn((1, 1, config.hidden_size), generator=torch.Generator().manual_seed(120_812)) * 0.02
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, 1, 1, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    # Position zero is a valid cache-independent decode.  Using position one
    # without a preceding prefill would read an unwritten KV-cache entry.
    position = torch.tensor([0], dtype=torch.int32)
    tt_position = ttnn.from_torch(
        position,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    rope = accepted._rope_tensors(config, mesh_device, position.to(torch.long), decode=True)
    cache_root = Path("/tmp/gpt_oss_120b_fused_two_layer_cache")

    def run_variant(reduce_output_memory_config, label):
        decoders = []
        page_tables = []
        for layer_idx in (1, 2):
            decoder = FusedDecoder.from_state_dict(
                _load_sweep_layer_state_dict(accepted.REAL_WEIGHT_SNAPSHOT, layer_idx),
                hf_config=config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                max_batch_size=1,
                max_context_length=128,
                page_size=accepted.PAGE_SIZE,
                tensor_cache_path=cache_root / label / f"layer_{layer_idx}",
                calibrated_checkpoint_revision=_FULL_LOCAL_CHECKPOINT_REVISION,
            )
            assert decoder.mlp.decode_uses_full_local
            decoder.mlp.decode_full_local_reduce_output_memory_config = reduce_output_memory_config
            decoders.append(decoder)
            page_tables.append(accepted._page_table(mesh_device, 128, seed=812 + layer_idx))

        def chain():
            first = decoders[0].decode_forward(
                tt_hidden,
                position_embeddings=rope,
                current_position=tt_position,
                page_table=page_tables[0],
            )
            assert first.layout == ttnn.TILE_LAYOUT
            assert first.memory_config() == reduce_output_memory_config
            second = decoders[1].decode_forward(
                first,
                position_embeddings=rope,
                current_position=tt_position,
                page_table=page_tables[1],
            )
            first.deallocate(True)
            assert second.layout == ttnn.TILE_LAYOUT
            assert second.memory_config() == reduce_output_memory_config
            return second

        eager = chain()
        ttnn.synchronize_device(mesh_device)
        eager_host = accepted._to_host(eager).clone()
        eager.deallocate(True)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            output = chain()
        finally:
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        first = accepted._to_host(output).clone()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        second = accepted._to_host(output)
        assert torch.equal(first, second)
        assert torch.equal(eager_host, first)

        repeats = 100
        started = time.perf_counter()
        for _ in range(repeats):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        wall_ms = 1000 * (time.perf_counter() - started) / repeats
        ttnn.release_trace(mesh_device, trace_id)
        output.deallocate(True)
        print(
            "TWO_LAYER_FULL_LOCAL_DECODE "
            f"layers=1,2 reduce_output={label} trace_repeats={repeats} "
            f"trace_wall_ms={wall_ms:.6f} layout=TILE deterministic=True"
        )
        del decoders, page_tables
        gc.collect()
        return eager_host, wall_ms

    l1_host, l1_wall_ms = run_variant(ttnn.L1_MEMORY_CONFIG, "l1")
    dram_host, dram_wall_ms = run_variant(ttnn.DRAM_MEMORY_CONFIG, "dram")
    matching, detail = comp_pcc(l1_host.float(), dram_host.float(), 0.999999)
    assert matching, f"two-layer L1/DRAM output mismatch: {detail}"
    assert torch.equal(l1_host, dram_host), "two-layer L1/DRAM outputs were not bitwise equal"
    print(
        "TWO_LAYER_FULL_LOCAL_OUTPUT_AB "
        f"l1_trace_wall_ms={l1_wall_ms:.6f} dram_trace_wall_ms={dram_wall_ms:.6f} "
        f"bitwise_equal=True pcc={detail}"
    )


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_prefill_tile_page_and_window_boundaries(monkeypatch, mesh_device, device_params, layer_idx, reset_seeds):
    _run_accepted_gate(
        monkeypatch,
        accepted.test_prefill_tile_page_and_window_boundaries,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
    )


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_traced_decode_at_advertised_context_limit(
    monkeypatch, mesh_device, device_params, layer_idx, reset_seeds
):
    _run_accepted_gate(
        monkeypatch,
        accepted.test_traced_decode_at_advertised_context_limit,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_RUN_CHUNK_BOUNDARIES") != "1",
    reason="set GPT_OSS_120B_RUN_CHUNK_BOUNDARIES=1 for the 4095/4096/4097-token gate",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_prefill_chunk_boundaries(monkeypatch, mesh_device, device_params, layer_idx, reset_seeds):
    _run_accepted_gate(
        monkeypatch,
        accepted.test_prefill_chunk_boundaries,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_RUN_MAX_PREFILL") != "1",
    reason="set GPT_OSS_120B_RUN_MAX_PREFILL=1 for the 131071/131072-token gate",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_prefill_at_advertised_context_limit(monkeypatch, mesh_device, device_params, layer_idx, reset_seeds):
    _run_accepted_gate(
        monkeypatch,
        accepted.test_prefill_at_advertised_context_limit,
        mesh_device,
        device_params,
        layer_idx,
        reset_seeds,
    )

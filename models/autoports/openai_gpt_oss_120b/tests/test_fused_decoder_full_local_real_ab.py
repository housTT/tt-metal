# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-checkpoint assessment gate for FullLocal ``moe_compute`` decode.

The FullLocal kernel is an attractive graph-fusion candidate because it joins
selective tilize, both expert projections, exact GPT-OSS SwiGLU, and local
combine in one trace-safe operation.  Its public weight preparation forces the
expert tensors to BF4, however. This opt-in A/B fixes the CPU oracle to the
exact TT router choices and scores, and proves whether the fused expert compute
clears the 0.995 direct-fusion bar. Every real decoder layer is swept because
attention kind does not itself determine expert-weight quantization sensitivity.
"""

import gc
import json
import os
import time
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from ttnn.experimental.moe_compute_utils import (
    auto_output_width_shard_dim,
    effective_matmul_ring_size,
    get_weight_core_shard_maps,
    get_weight_mem_configs,
    prepare_w0_w1_tensor_with_bias,
    prepare_w2_tensor_with_bias,
)
from ttnn.operations.ccl import MoEActivationFunction

import ttnn
from models.autoports.openai_gpt_oss_120b.tests import test_functional_decoder as accepted
from models.autoports.openai_gpt_oss_120b.tests.real_weight_utils import _EXPECTED_LOCAL_SHAPES, _PACKED_EXPERT_SHAPES
from models.autoports.openai_gpt_oss_120b.tt.fused_decoder import (
    _FULL_LOCAL_DECODE_LAYERS,
    _FULL_LOCAL_REDUCE_OUTPUT_MEMORY_CONFIG,
    FusedDecoder,
)
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gpt_oss.tt.topk import TopKRouter
from models.demos.gpt_oss.utils.substate import substate
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import create_sharded_memory_config, gen_expert_mapping
from tests.ttnn.nightly.unit_tests.operations.experimental.test_moe_compute_single_card import (
    _build_quantized_weight_tensors_cpu_prepare,
)

_DIRECT_FUSION_PCC = 0.995
_TOKENS = int(os.environ.get("GPT_OSS_120B_FULL_LOCAL_REAL_TOKENS", "8"))
if not 1 <= _TOKENS <= 32:
    raise ValueError("GPT_OSS_120B_FULL_LOCAL_REAL_TOKENS must be in [1, 32]")
_HIDDEN_SEED_BASE = int(os.environ.get("GPT_OSS_120B_FULL_LOCAL_REAL_SEED_BASE", "120511"))
_ASSERT_ALLOWLIST_PCC = os.environ.get("GPT_OSS_120B_FULL_LOCAL_ASSERT_ALLOWLIST_PCC") == "1"
_WEIGHT_QUANTIZER = os.environ.get("GPT_OSS_120B_FULL_LOCAL_WEIGHT_QUANTIZER", "host_quant")
if _WEIGHT_QUANTIZER not in {"device_typecast", "host_quant"}:
    raise ValueError(
        "GPT_OSS_120B_FULL_LOCAL_WEIGHT_QUANTIZER must be 'device_typecast' or 'host_quant', "
        f"got {_WEIGHT_QUANTIZER!r}"
    )
_NUM_LAYERS = 36


def _build_host_quantized_weight_tensors_cpu_prepare(
    mesh_device,
    torch_w0,
    torch_w1,
    torch_w2,
    torch_b0,
    torch_b1,
    torch_b2,
    num_layers,
    num_experts,
    hidden_size,
    intermediate_size,
    w0_w1_shard_map,
    w2_shard_map,
    w0_w1_mem_config,
    w2_mem_config,
):
    """CPU-pack and host-quantize one biased FullLocal layer, one projection at a time."""

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def upload_bf16(packed, memory_config):
        return ttnn.from_torch(
            packed,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            mesh_mapper=replicate,
        )

    def quantize_and_release(bf16, memory_config):
        try:
            return ttnn.experimental.quantize_weights_via_host(
                bf16,
                dtype=ttnn.bfloat4_b,
                memory_config=memory_config,
            )
        finally:
            bf16.deallocate(True)

    packed_w0_w1 = prepare_w0_w1_tensor_with_bias(
        torch_w0,
        torch_w1,
        torch_b0,
        torch_b1,
        num_layers,
        num_experts,
        hidden_size,
        intermediate_size,
        w0_w1_shard_map,
    )
    bf16_w0_w1 = upload_bf16(packed_w0_w1, w0_w1_mem_config)
    del packed_w0_w1
    tt_w0_w1 = quantize_and_release(bf16_w0_w1, w0_w1_mem_config)
    del bf16_w0_w1

    try:
        packed_w2 = prepare_w2_tensor_with_bias(
            torch_w2,
            torch_b2,
            num_layers,
            num_experts,
            intermediate_size,
            hidden_size,
            w2_shard_map,
            w0_w1_shard_map,
        )
        bf16_w2 = upload_bf16(packed_w2, w2_mem_config)
        del packed_w2
        tt_w2 = quantize_and_release(bf16_w2, w2_mem_config)
        del bf16_w2
    except Exception:
        tt_w0_w1.deallocate(True)
        raise
    return tt_w0_w1, tt_w2


def _load_sweep_layer_state_dict(snapshot_path, layer_idx):
    """Load any real layer without widening the functional gate's 0/1 API."""
    snapshot_path = Path(snapshot_path)
    with (snapshot_path / "model.safetensors.index.json").open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]
    prefix = f"model.layers.{layer_idx}."
    locations = {
        checkpoint_key[len(prefix) :]: shard_name
        for checkpoint_key, shard_name in weight_map.items()
        if checkpoint_key.startswith(prefix)
    }
    expected_raw_keys = set(_EXPECTED_LOCAL_SHAPES) - {
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    } | set(_PACKED_EXPERT_SHAPES)
    assert locations.keys() == expected_raw_keys, (
        f"layer {layer_idx} checkpoint keys differ: "
        f"missing={sorted(expected_raw_keys - locations.keys())}, "
        f"unexpected={sorted(locations.keys() - expected_raw_keys)}"
    )

    handles = {}
    state_dict = {}
    try:
        for local_key, shard_name in locations.items():
            handle = handles.setdefault(
                shard_name,
                safe_open(snapshot_path / shard_name, framework="pt", device="cpu"),
            )
            state_dict[local_key] = handle.get_tensor(prefix + local_key)
    finally:
        handles.clear()

    for projection in ("gate_up_proj", "down_proj"):
        packed_prefix = f"mlp.experts.{projection}"
        blocks = state_dict.pop(f"{packed_prefix}_blocks")
        scales = state_dict.pop(f"{packed_prefix}_scales")
        state_dict[packed_prefix] = convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)
    assert {key: tuple(value.shape) for key, value in state_dict.items()} == _EXPECTED_LOCAL_SHAPES
    return state_dict


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_LOCAL_FUNCTIONAL_FALLBACK_BASELINE") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set the functional-fallback baseline opt-in and GPT_OSS_120B_SNAPSHOT",
)
@parametrize_mesh_with_fabric([(1, 1)])
def test_functional_real_weight_failing_full_local_layer5_baseline(
    monkeypatch, mesh_device, device_params, reset_seeds
):
    """Isolate whether layer-5 whole-layer PCC is specific to fused indexed decode."""
    monkeypatch.setattr(accepted, "load_real_layer_state_dict", _load_sweep_layer_state_dict)
    accepted.test_real_weight_paged_prefill_and_traced_decode(
        mesh_device,
        device_params,
        layer_idx=5,
        reset_seeds=reset_seeds,
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_LOCAL_FALLBACK_EQUIVALENCE") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set the fallback-equivalence opt-in and GPT_OSS_120B_SNAPSHOT",
)
@parametrize_mesh_with_fabric([(1, 1)])
def test_layer5_indexed_fallback_preserves_functional_real_outputs(
    monkeypatch, mesh_device, device_params, reset_seeds
):
    """Prove the production-reachable indexed fallback matches functional."""
    monkeypatch.setattr(accepted, "load_real_layer_state_dict", _load_sweep_layer_state_dict)
    original_decoder = accepted.FunctionalDecoder
    original_to_host = accepted._to_host
    current_outputs = []
    hf_details = {}
    current_run_label = None

    def capture_to_host(tensor):
        host = original_to_host(tensor)
        current_outputs.append(host.clone())
        return host

    def record_hf_pcc(actual, expected, label, threshold=accepted.DEFAULT_PCC_THRESHOLD):
        del threshold
        _, detail = comp_pcc(expected.float(), actual.float(), 0.0)
        hf_details[f"{current_run_label}: {label}"] = detail
        return detail

    monkeypatch.setattr(accepted, "_to_host", capture_to_host)
    monkeypatch.setattr(accepted, "_assert_pcc", record_hf_pcc)

    def run(decoder_type, run_label):
        nonlocal current_run_label
        current_run_label = run_label
        current_outputs.clear()
        monkeypatch.setattr(accepted, "FunctionalDecoder", decoder_type)
        accepted.test_real_weight_paged_prefill_and_traced_decode(
            mesh_device,
            device_params,
            layer_idx=5,
            reset_seeds=reset_seeds,
        )
        outputs = [output.clone() for output in current_outputs]
        assert len(outputs) == 3
        gc.collect()
        return outputs

    functional = run(original_decoder, "functional")
    # These independently-authored paths share low-level op program hashes but
    # use different persistent weight layouts.  Clear setup programs between
    # them so the A/B compares tensor semantics instead of stale cache state.
    mesh_device.disable_and_clear_program_cache()
    mesh_device.enable_program_cache()

    class IndexedFusedDecoder(FusedDecoder):
        @classmethod
        def from_state_dict(cls, *args, **kwargs):
            kwargs["calibrated_checkpoint_revision"] = "unqualified-revision"
            decoder = super().from_state_dict(*args, **kwargs)
            assert not decoder.mlp.decode_uses_full_local
            assert hasattr(decoder.mlp, "decode_packed_gate_up")
            assert not hasattr(decoder.mlp, "decode_full_local_w0_w1")
            return decoder

    indexed = run(IndexedFusedDecoder, "indexed")
    prefill_matching, prefill_detail = comp_pcc(functional[0].float(), indexed[0].float(), 0.995)
    decode_matching, decode_detail = comp_pcc(functional[1].float(), indexed[1].float(), 0.995)
    assert prefill_matching, f"layer-5 fused/functional prefill mismatch: {prefill_detail}"
    assert decode_matching, f"layer-5 indexed/functional decode mismatch: {decode_detail}"
    assert torch.equal(functional[1], indexed[1])
    assert torch.equal(functional[2], indexed[2])
    print(
        "FULL_LOCAL_LAYER5_FALLBACK_EQUIVALENCE "
        f"prefill_pcc={prefill_detail} decode_pcc={decode_detail} decode_bitwise_equal=True "
        f"hf_details={hf_details}"
    )


@pytest.mark.parametrize(
    "layer_idx",
    range(_NUM_LAYERS),
    ids=[f"layer-{index:02d}-{'sliding' if index % 2 == 0 else 'full'}" for index in range(_NUM_LAYERS)],
)
@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_LOCAL_REAL_AB") != "1" or not accepted.REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_FULL_LOCAL_REAL_AB=1 and GPT_OSS_120B_SNAPSHOT for the real FullLocal A/B",
)
@parametrize_mesh_with_fabric([(1, 1)])
def test_full_local_moe_compute_real_weight_assessment(mesh_device, device_params, layer_idx, reset_seeds):
    """Compare BF4 FullLocal expert compute with an exact-route BF16 Torch oracle."""
    del device_params, reset_seeds
    config = accepted._config()

    dequant_started = time.perf_counter()
    state_dict = _load_sweep_layer_state_dict(accepted.REAL_WEIGHT_SNAPSHOT, layer_idx)
    dequant_seconds = time.perf_counter() - dequant_started
    reference = accepted._reference_layer(config, layer_idx, state_dict=state_dict)
    mlp_state = substate(state_dict, "mlp")
    expert_state = substate(mlp_state, "experts")
    router = TopKRouter(mesh_device, config, substate(mlp_state, "router"))
    # Keep the candidate's routing graph identical for 1..31 logical decode
    # users and avoid the B=32-only topk_router_gpt specialization.
    router.use_fused_op = False

    # Public FullLocal packers consume [L,E,H,N] gate/up and [L,E,N,H]
    # down tensors. GPT-OSS stores gate/up interleaved in the last dimension.
    gate_up = expert_state["gate_up_proj"]
    gate_up_bias = expert_state["gate_up_proj_bias"]
    torch_w0 = gate_up[..., ::2].unsqueeze(0).contiguous()
    torch_w1 = gate_up[..., 1::2].unsqueeze(0).contiguous()
    torch_w2 = expert_state["down_proj"].unsqueeze(0).contiguous()
    torch_b0 = gate_up_bias[..., ::2].unsqueeze(0).contiguous()
    torch_b1 = gate_up_bias[..., 1::2].unsqueeze(0).contiguous()
    torch_b2 = expert_state["down_proj_bias"].unsqueeze(0).contiguous()

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_experts = config.num_local_experts
    top_k = config.num_experts_per_tok
    output_height_shard_dim = 4
    output_width_shard_dim = auto_output_width_shard_dim(
        hidden_size,
        matmul_ring_size=effective_matmul_ring_size(mesh_device),
    )
    w0_w1_shard_map, w2_shard_map, dram_core_range_set = get_weight_core_shard_maps(
        mesh_device, hidden_size, intermediate_size
    )
    w0_w1_mem_config, w2_mem_config, _, _ = get_weight_mem_configs(
        1,
        num_experts,
        hidden_size,
        intermediate_size,
        w0_w1_shard_map,
        w2_shard_map,
        dram_core_range_set,
        has_bias=True,
    )
    print(f"FULL_LOCAL_WEIGHT_PREP layer={layer_idx} weight_quantizer={_WEIGHT_QUANTIZER}")
    if _WEIGHT_QUANTIZER == "host_quant":
        tt_w0_w1, tt_w2 = _build_host_quantized_weight_tensors_cpu_prepare(
            mesh_device,
            torch_w0,
            torch_w1,
            torch_w2,
            torch_b0,
            torch_b1,
            torch_b2,
            1,
            num_experts,
            hidden_size,
            intermediate_size,
            w0_w1_shard_map,
            w2_shard_map,
            w0_w1_mem_config,
            w2_mem_config,
        )
    else:
        tt_w0_w1, tt_w2 = _build_quantized_weight_tensors_cpu_prepare(
            mesh_device,
            torch_w0,
            torch_w1,
            torch_w2,
            torch_b0,
            torch_b1,
            torch_b2,
            1,
            num_experts,
            hidden_size,
            intermediate_size,
            True,
            w0_w1_shard_map,
            w2_shard_map,
            w0_w1_mem_config,
            w2_mem_config,
        )
    del torch_w0, torch_w1, torch_w2, torch_b0, torch_b1, torch_b2, gate_up, gate_up_bias, expert_state

    drain = ttnn.experimental.get_moe_tilize_drain_core(
        mesh_device,
        output_height_shard_dim,
        output_width_shard_dim,
        hidden_size,
    )
    drain_core = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(drain.x, drain.y),
                ttnn.CoreCoord(drain.x, drain.y),
            )
        }
    )
    indices_mem_config = create_sharded_memory_config(drain_core, [_TOKENS, top_k], ttnn.uint16)
    scores_mem_config = create_sharded_memory_config(drain_core, [_TOKENS, top_k], ttnn.bfloat16)
    torch_expert_mapping = gen_expert_mapping(1, 1, None, num_experts, num_experts, num_experts)
    tt_expert_mapping = ttnn.from_torch(
        torch_expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_reduce_expert_mapping = ttnn.from_torch(
        torch_expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_combine_output = ttnn.from_torch(
        torch.zeros((top_k, _TOKENS, hidden_size), dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )

    hidden_seed = _HIDDEN_SEED_BASE + 1000 * _TOKENS + layer_idx
    generator = torch.Generator().manual_seed(hidden_seed)
    hidden = (torch.randn((_TOKENS, 1, hidden_size), generator=generator) * 0.02).to(torch.bfloat16)
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, _TOKENS, 1, hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    def route(candidate_input):
        indices, scores = router(candidate_input, use_throughput_experts=True)
        return indices, scores

    def full_local_from_route(candidate_input, indices, scores):
        sparse_input = ttnn.reshape(candidate_input, (1, _TOKENS, hidden_size))
        sparse_input = ttnn.to_layout(
            sparse_input,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        candidate_input.deallocate(True)

        indices = ttnn.reshape(indices, (1, _TOKENS, top_k))
        indices_rm = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_indices = ttnn.to_memory_config(indices_rm, indices_mem_config)
        indices.deallocate(True)

        scores = ttnn.reshape(scores, (1, _TOKENS, top_k))
        scores_rm = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_scores = ttnn.to_memory_config(scores_rm, scores_mem_config)
        scores.deallocate(True)

        outputs = ttnn.experimental.moe_compute(
            sparse_input,
            tt_indices,
            tt_scores,
            tt_expert_mapping,
            tt_w0_w1,
            tt_w2,
            layer_id=0,
            output_height_shard_dim=output_height_shard_dim,
            intermediate_size=intermediate_size,
            has_bias=True,
            cluster_axis=None,
            topology=None,
            num_links=None,
            mux_core_range_set=None,
            optional_output_tensor=tt_combine_output,
            optional_cross_device_semaphore=None,
            # On Blackhole/P150 this enum selects the exact GPT-OSS formula:
            # clamp limit 7.0 and alpha 1.702, not ordinary SwiGLU.
            activation_type=MoEActivationFunction.SWIGLU,
            compute_only=False,
        )
        sparse_input.deallocate(True)
        tt_indices.deallocate(True)
        tt_scores.deallocate(True)
        for tensor in (outputs[0], outputs[1], outputs[2], outputs[4]):
            tensor.deallocate(True)

        slots4d = ttnn.reshape(outputs[5], (top_k, 1, _TOKENS, hidden_size))
        padded_slots = ttnn.tilize_with_val_padding(
            slots4d,
            output_tensor_shape=(top_k, 1, 32, hidden_size),
            pad_value=0.0,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        indices4d = ttnn.reshape(indices_rm, (_TOKENS, 1, 1, top_k))
        scores4d = ttnn.reshape(scores_rm, (_TOKENS, 1, 1, top_k))
        reduced = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            padded_slots,
            indices4d,
            tt_reduce_expert_mapping,
            reduce_dim=0,
            split_size=hidden_size,
            cluster_axis=0,
            output_memory_config=_FULL_LOCAL_REDUCE_OUTPUT_MEMORY_CONFIG,
            scores_tensor=scores4d,
            num_shared_experts=0,
        )[0]
        padded_slots.deallocate(True)
        indices_rm.deallocate(True)
        scores_rm.deallocate(True)
        return reduced

    def candidate():
        candidate_input = ttnn.clone(tt_hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        indices, scores = route(candidate_input)
        return full_local_from_route(candidate_input, indices, scores)

    # PCC uses the exact TT-selected route. This separates BF4 expert-compute
    # error from any top-k discontinuity between the HF and TT router graphs.
    eager_input = ttnn.clone(tt_hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    eager_indices, eager_scores = route(eager_input)
    ttnn.synchronize_device(mesh_device)
    route_indices = ttnn.to_torch(eager_indices).reshape(_TOKENS, top_k).to(torch.long)
    route_scores = ttnn.to_torch(eager_scores).reshape(_TOKENS, top_k).to(torch.bfloat16)
    unique_experts = torch.unique(route_indices).numel()
    with torch.no_grad():
        fixed_route_oracle = reference.mlp.experts(
            hidden.reshape(_TOKENS, hidden_size),
            route_indices,
            route_scores,
        )

    eager_output = full_local_from_route(eager_input, eager_indices, eager_scores)
    ttnn.synchronize_device(mesh_device)
    actual = ttnn.to_torch(eager_output)[0, 0]
    eager_output.deallocate(True)
    passed, detail = comp_pcc(fixed_route_oracle.float(), actual.float(), _DIRECT_FUSION_PCC)
    print(
        "FULL_LOCAL_REAL_PCC "
        f"layer={layer_idx} type={config.layer_types[layer_idx]} tokens={_TOKENS} hidden_seed={hidden_seed} "
        f"weight_quantizer={_WEIGHT_QUANTIZER} routed_experts={unique_experts} "
        f"fixed_route_torch_pcc={detail} threshold={_DIRECT_FUSION_PCC}"
    )

    # All static buffers and programs now exist before capture. Only this one
    # trace remains alive, so no candidate allocations overlap another trace.
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        traced_output = candidate()
    finally:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    first = ttnn.to_torch(traced_output)[0, 0].clone()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    second = ttnn.to_torch(traced_output)[0, 0]
    deterministic = torch.equal(first, second)

    repeats = int(os.environ.get("GPT_OSS_120B_FULL_LOCAL_REAL_REPEATS", "20"))
    started = time.perf_counter()
    for _ in range(repeats):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    wall_ms = 1000 * (time.perf_counter() - started) / repeats
    ttnn.release_trace(mesh_device, trace_id)

    print(
        "FULL_LOCAL_REAL_ASSESSMENT "
        f"layer={layer_idx} type={config.layer_types[layer_idx]} checkpoint={accepted.REAL_WEIGHT_SNAPSHOT} "
        f"tokens={_TOKENS} hidden_seed={hidden_seed} weight_quantizer={_WEIGHT_QUANTIZER} "
        f"routed_experts={unique_experts} "
        f"dequant_seconds={dequant_seconds:.3f} "
        f"fixed_route_torch_pcc={detail} threshold={_DIRECT_FUSION_PCC} "
        f"meets_bar={passed} trace_repeats={repeats} trace_wall_ms={wall_ms:.6f} "
        f"deterministic={deterministic}"
    )
    assert deterministic, "FullLocal real-weight trace replay was not bitwise deterministic"
    assert torch.isfinite(actual).all(), "FullLocal real-weight output contains non-finite values"
    if _ASSERT_ALLOWLIST_PCC and layer_idx in _FULL_LOCAL_DECODE_LAYERS:
        assert (
            passed
        ), f"delivered FullLocal layer {layer_idx} missed the {_DIRECT_FUSION_PCC} exact-route PCC bar: {detail}"

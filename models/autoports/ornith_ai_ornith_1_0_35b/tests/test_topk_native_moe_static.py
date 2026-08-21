# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host/static contract tests for Ornith's router-index-native MoE path.

The device correctness cases live beside the TTNN op. These tests are cheap
enough for every host run and pin the graph/counter/cache contracts that are
easy to regress without changing numerical output.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC
from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as OD

REPO = Path(OD.__file__).resolve().parents[4]
OP = REPO / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/topk_routed_expert_moe"
PROMOTION_TEST = REPO / ("models/autoports/ornith_ai_ornith_1_0_35b/tests/test_topk_native_moe_promotion.py")


def test_native_chunk_decomposition_covers_b1_and_b4_prefill():
    assert OD._topk_native_chunk_ranges(2048) == ((0, 1024), (1024, 2048))
    assert OD._topk_native_chunk_ranges(8192) == tuple((i, i + 1024) for i in range(0, 8192, 1024))


def test_native_chunk_decomposition_rejects_ragged_tail():
    try:
        OD._topk_native_chunk_ranges(1152)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("ragged native sub-chunk was admitted")


def test_promotion_topology_gate_requires_exact_replicate_placement_rank():
    source = PROMOTION_TEST.read_text()
    record_source = source[source.index("def _mesh_tensor_record(") : source.index("\ndef _native_input_mesh_record(")]
    assertion_source = source[
        source.index("def _assert_native_input_mesh_contract(") : source.index("\ndef _assert_native_ready(")
    ]

    assert "len(topology_placements) == len(distribution_shape)" in record_source
    assert "isinstance(placement, ttnn.PlacementReplicate)" in record_source
    assert "ttnn.PlacementShard" not in record_source
    assert 'stage["placement_count_matches_distribution_rank"]' in assertion_source
    assert 'stage["all_placements_are_replicate"]' in assertion_source


def test_global_to_local_map_is_shared_contiguous_ep_semantics():
    maps = MC._global_to_local_expert_maps(256, 64, 4)
    assert maps.shape == (4, 256)
    for device in range(4):
        lo, hi = device * 64, (device + 1) * 64
        assert torch.equal(maps[device, lo:hi], torch.arange(64, dtype=torch.int32))
        assert torch.all(maps[device, :lo] == -1)
        assert torch.all(maps[device, hi:] == -1)

    decode_maps = MC._global_to_local_expert_maps(256, 64, 4, nonlocal_value=64)
    assert torch.all(decode_maps[0, 64:] == 64)


def test_selected_python_path_calls_exact_composite_and_has_no_dense_glue():
    source = inspect.getsource(OD.OptimizedMoE._topk_native_routed_experts)
    assert (
        "self.topk_native_calls += 1\n            part = ttnn.experimental.deepseek_prefill.topk_routed_expert_moe("
        in source
    )
    for forbidden in (
        "routing_weights(",
        "ttnn.scatter(",
        "ttnn.sort(",
        "ttnn.embedding(",
        "ttnn.where(",
        "deepseek_moe_fast_reduce_nc",
    ):
        assert forbidden not in source


def test_cpp_composite_orders_dispatch_experts_combine_without_dense_routing():
    source = (OP / "topk_routed_expert_moe.cpp").read_text()
    dispatch = source.index("topk_local_dispatch(", source.index("ttnn::Tensor topk_routed_expert_moe("))
    experts = source.index("unified_routed_expert_moe(", dispatch)
    combine = source.index("topk_local_combine(", experts)
    assert dispatch < experts < combine
    assert "ttnn::scatter(" not in source
    assert "ttnn::sort(" not in source
    assert "ttnn::embedding(" not in source
    assert "deepseek_moe_fast_reduce_nc" not in source


def test_public_composite_has_unambiguous_compute_policy_and_replacement_output():
    header = (OP / "topk_routed_expert_moe.hpp").read_text()
    source = (OP / "topk_routed_expert_moe.cpp").read_text()
    binding = (OP / "topk_routed_expert_moe_nanobind.cpp").read_text()
    combine_types = (OP / "device/topk_local_combine_types.hpp").read_text()
    assert "DeviceComputeKernelConfig" not in header
    assert "std::nullopt,\n        activation" in source
    assert 'nb::arg("compute_kernel_config")' not in binding
    assert binding.count('nb::arg("output") = nb::none()') == 2
    assert "std::optional<Tensor> optional_output" in combine_types


def test_production_composite_uses_indexed_x_and_assignment_slots_under_40_mib():
    source = (OP / "topk_routed_expert_moe.cpp").read_text()
    unified = (
        REPO / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/"
        "unified_routed_expert_ffn.cpp"
    ).read_text()
    compute = (
        REPO / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/"
        "compute/fused_swiglu.cpp"
    ).read_text()
    writer = (
        REPO / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/"
        "dataflow/unified_routed_expert_ffn_writer.cpp"
    ).read_text()

    assert "/*materialize_x=*/false" in source
    assert "dispatch[4],\n        topk" in source
    assert "/*assignment_addressed=*/true" in source
    assert "auto y_bf16" not in source
    assert "auto y_rm" not in source
    assert "pack_untilize_block<out_subblock_w, out_subblock_w>" in compute
    assert ".page_id = assignment, .offset_bytes = output_offset_bytes" in writer
    assert "DataType::BFLOAT16" in unified and "Layout::ROW_MAJOR" in unified

    # T=1024, K=8, H=2048: x_rm + assignment slots + final BF8 + integer
    # metadata. This is the active sub-chunk workspace, not caller-owned x.
    x_rm = 1024 * 2048 * 2
    slots_rm = 1024 * 8 * 2048 * 2
    final_bf8 = 1024 * 2048 + (1024 // 32) * (2048 // 32) * 64
    metadata = (2 * 256 + 64 + (1024 * 8 + 31 * 64) + 2 * 1024 * 8) * 4
    total = x_rm + slots_rm + final_bf8 + metadata
    assert slots_rm == 33_554_432
    assert total == 40_085_504
    assert total < 40 * 1024 * 1024


def test_native_router_makes_uint32_index_contract_explicit():
    for cls in (OD.OptimizedMoE, MC.MultichipMoE):
        source = inspect.getsource(cls._routing_pairs)
        assert "ttnn.typecast(raw_indices, ttnn.uint32)" in source
        assert "ttnn.deallocate(raw_indices)" in source


def test_fp32_topk_weight_rounding_and_k8_page_geometry_are_explicit():
    composite = (OP / "topk_routed_expert_moe.cpp").read_text()
    validation = (OP / "device/topk_local_combine_device_operation.cpp").read_text()
    writer = (OP / "device/kernels/dataflow/topk_local_combine_writer.cpp").read_text()
    assert "topk_weights.dtype() == tt::tt_metal::DataType::FLOAT32" in composite
    assert "ttnn::typecast(topk_weights, tt::tt_metal::DataType::BFLOAT16)" in composite
    assert "buffer()->num_pages() == op.tokens" in validation
    assert ".page_id = token_start + token" in writer
    assert "token_weights[slot]" in writer


def test_combine_batches_k8_rows_and_chunk_weights_without_reordering_math():
    factory = (OP / "device/topk_local_combine_program_factory.cpp").read_text()
    reader = (OP / "device/kernels/dataflow/topk_local_combine_reader.cpp").read_text()
    writer = (OP / "device/kernels/dataflow/topk_local_combine_writer.cpp").read_text()
    compute = (OP / "device/kernels/compute/topk_local_combine.cpp").read_text()

    assert "constexpr uint32_t PACKED_ROW_BATCH_CAP = 8;" in factory
    assert "const uint64_t packed_row_cb_bytes = op.topk * row_bytes;" in factory
    assert "const uint64_t weight_scratch_bytes = TOKENS_PER_CHUNK * weight_page_size;" in factory
    assert "CB_PACKED_ROW, packed_row_cb_bytes" in factory
    assert "CB_WEIGHT_SCRATCH, weight_scratch_bytes, weight_page_size" in factory

    zero_batch = reader.index("rows_in_batch * row_storage_bytes")
    zero_barrier = reader.index("noc.write_zeros_l1_barrier();", zero_batch)
    packed_read = reader.index("CoreLocalMem<uint32_t>(batch_l1 + batch_row * row_storage_bytes)")
    read_barrier = reader.index("noc.async_read_barrier();", packed_read)
    batch_push = reader.index("cb_packed_row.push_back(tiles_in_batch);", read_barrier)
    assert zero_batch < zero_barrier < packed_read < read_barrier < batch_push

    weight_read_loop = writer.index("for (uint32_t token = 0; token < tokens_per_chunk; ++token)")
    weight_offset = writer.index("weight_scratch_l1 + token * weight_page_size", weight_read_loop)
    weight_barrier = writer.index("noc.async_read_barrier();", weight_offset)
    weight_consume_loop = writer.index("for (uint32_t token = 0; token < tokens_per_chunk; ++token)", weight_barrier)
    assert weight_read_loop < weight_offset < weight_barrier < weight_consume_loop

    # Reader and writer still publish token-major/slot-major streams, and the
    # compute kernel retains its exact K reduction order.
    compute_token = compute.index("for (uint32_t token = 0; token < tokens_per_chunk; ++token)")
    compute_slot = compute.index("for (uint32_t slot = 0; slot < topk; ++slot)", compute_token)
    assert compute_token < compute_slot


def test_cache_hit_callbacks_patch_every_runtime_tensor_address_and_valid_tokens():
    dispatch = (OP / "device/topk_local_dispatch_program_factory.cpp").read_text()
    combine = (OP / "device/topk_local_combine_program_factory.cpp").read_text()
    dispatch_override = dispatch[dispatch.index("override_runtime_arguments(") :]
    for expression in (
        "tensors.topk_indices.buffer()->address()",
        "tensors.global_to_local_expert.buffer()->address()",
        "tensors.x.buffer()->address()",
        "outputs[output_index].buffer()->address()",
        "op.valid_tokens",
    ):
        assert expression in dispatch_override
    combine_override = combine[combine.index("override_runtime_arguments(") :]
    for expression in (
        "tensors.packed_y.buffer()->address()",
        "tensors.topk_weights.buffer()->address()",
        "tensors.slot_to_packed_row.buffer()->address()",
        "tensors.slot_is_local.buffer()->address()",
        "output.buffer()->address()",
    ):
        assert expression in combine_override
    assert "tensors.optional_output" in (OP / "device/topk_local_combine_device_operation.cpp").read_text()


def test_custom_topologies_use_the_device_operation_vector_callback_contract():
    dispatch_header = (OP / "device/topk_local_dispatch_device_operation.hpp").read_text()
    combine_header = (OP / "device/topk_local_combine_device_operation.hpp").read_text()
    callback_type = "using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;"
    assert callback_type in dispatch_header
    assert callback_type in combine_header


def test_assignment_plan_capacity_is_validated_before_kernel_launch():
    validation = (
        REPO / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/"
        "unified_routed_expert_ffn_device_operation.cpp"
    ).read_text()
    assert "required_assignment_capacity = tokens * op.topk + (tile_height - 1) * idx_table_size" in validation
    assert "assignment_shape[-1]) >= required_assignment_capacity" in validation


def test_native_multichunk_returns_routed_and_frees_concat_parts():
    source = inspect.getsource(OD.OptimizedMoE._topk_native_routed_experts)
    assert "routed = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)" in source
    assert "for part in parts:\n                ttnn.deallocate(part)" in source
    assert source.rstrip().endswith("return routed")


def test_dispatch_and_combine_never_reuse_posted_write_or_read_invalid_row():
    dispatch = (OP / "device/kernels/dataflow/topk_local_dispatch.cpp").read_text()
    reader = (OP / "device/kernels/dataflow/topk_local_combine_reader.cpp").read_text()
    posted = dispatch.index("noc.async_write(CoreLocalMem<uint32_t>(row_l1)")
    next_barrier = dispatch.index("noc.async_write_barrier();", posted)
    assert next_barrier > posted
    zero_batch = reader.index("noc.async_write_zeros")
    zero_barrier = reader.index("noc.write_zeros_l1_barrier();", zero_batch)
    guard = reader.index("if (is_local[map_index] == 1 && packed_row < capacity)")
    packed_read = reader.index("packed_y,", guard)
    assert zero_batch < zero_barrier < guard < packed_read


def test_planner_rejects_duplicate_expert_ids_without_host_readback():
    planner = (OP / "device/kernels/dataflow/topk_local_plan.cpp").read_text()
    assert "for (uint32_t prior_slot = 0; prior_slot < slot; ++prior_slot)" in planner
    assert "if (duplicate_expert)" in planner
    assert "plan_ok = false;" in planner

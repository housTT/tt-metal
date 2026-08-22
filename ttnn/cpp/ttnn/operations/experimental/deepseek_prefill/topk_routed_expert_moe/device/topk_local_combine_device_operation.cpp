// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_local_combine_device_operation.hpp"

#include <cstdint>
#include <initializer_list>
#include <utility>
#include <variant>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe {

namespace {

bool combine_is_dram_interleaved(const Tensor& tensor) {
    const auto& memory_config = tensor.memory_config();
    return memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

bool combine_is_fully_replicated(const tt::tt_metal::TensorTopology& topology) {
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    for (const auto& placement : topology.placements()) {
        if (std::holds_alternative<Shard>(placement)) {
            return false;
        }
    }
    return true;
}

bool combine_has_same_mesh_footprint(const tt::tt_metal::TensorTopology& lhs, const tt::tt_metal::TensorTopology& rhs) {
    return lhs.distribution_shape() == rhs.distribution_shape() && lhs.mesh_coords() == rhs.mesh_coords();
}

bool combine_is_device_unique(const tt::tt_metal::TensorTopology& topology) {
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    if (topology.mesh_coords().size() <= 1) {
        return true;
    }
    const auto& shape = topology.distribution_shape();
    const auto& placements = topology.placements();
    if (placements.size() != shape.dims()) {
        return false;
    }
    for (size_t axis = 0; axis < shape.dims(); ++axis) {
        if (shape[axis] > 1 && !std::holds_alternative<Shard>(placements[axis])) {
            return false;
        }
    }
    return true;
}

}  // namespace

void TopkLocalCombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& op, const tensor_args_t& tensors) {
    for (const auto& [name, tensor] : std::initializer_list<std::pair<const char*, const Tensor&>>{
             {"packed_y", tensors.packed_y},
             {"topk_weights", tensors.topk_weights},
             {"slot_to_packed_row", tensors.slot_to_packed_row},
             {"slot_is_local", tensors.slot_is_local}}) {
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", name);
        TT_FATAL(tensor.buffer() != nullptr, "{} must have a device buffer", name);
        TT_FATAL(tensor.device() == tensors.packed_y.device(), "{} must be on the same device as packed_y", name);
        TT_FATAL(combine_is_dram_interleaved(tensor), "{} must be DRAM-interleaved", name);
    }

    const auto& plan_topology = tensors.slot_to_packed_row.tensor_topology();
    TT_FATAL(
        combine_is_device_unique(plan_topology),
        "slot_to_packed_row must be device-unique (sharded) on a multi-device mesh");
    TT_FATAL(
        tensors.slot_is_local.tensor_topology() == plan_topology,
        "slot_is_local topology must exactly match slot_to_packed_row");
    TT_FATAL(
        tensors.packed_y.tensor_topology() == plan_topology,
        "packed_y topology must exactly match the device-local slot maps");
    const auto& weight_topology = tensors.topk_weights.tensor_topology();
    TT_FATAL(
        combine_is_fully_replicated(weight_topology),
        "topk_weights must be fully replicated across the expert-parallel mesh");
    TT_FATAL(
        combine_has_same_mesh_footprint(weight_topology, plan_topology),
        "topk_weights must cover the same mesh coordinates as the device-local slot maps");

    if (tensors.optional_output.has_value()) {
        const auto& output = *tensors.optional_output;
        TT_FATAL(output.storage_type() == StorageType::DEVICE, "optional_output must be on device");
        TT_FATAL(output.buffer() != nullptr, "optional_output must have a device buffer");
        TT_FATAL(
            output.device() == tensors.packed_y.device(), "optional_output must be on the same device as packed_y");
        TT_FATAL(combine_is_dram_interleaved(output), "optional_output must be DRAM-interleaved");
        TT_FATAL(
            output.tensor_topology() == plan_topology,
            "optional_output topology must exactly match the device-local slot maps");
    }

    TT_FATAL(
        tensors.packed_y.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "packed_y must be BFLOAT16, got {}",
        tensors.packed_y.dtype());
    TT_FATAL(
        tensors.packed_y.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "packed_y must be ROW_MAJOR, got {}",
        tensors.packed_y.layout());
    const auto& y_shape = tensors.packed_y.logical_shape();
    TT_FATAL(
        y_shape.rank() == 4 && y_shape[0] == 1 && y_shape[1] == 1,
        "packed_y must have shape [1,1,C,H], got {}",
        y_shape);
    TT_FATAL(y_shape[-2] > 0 && y_shape[-1] > 0, "packed_y capacity/hidden must be positive, got {}", y_shape);
    TT_FATAL(
        y_shape[-1] % tt::constants::TILE_WIDTH == 0,
        "packed_y hidden size ({}) must be tile-width aligned",
        y_shape[-1]);
    TT_FATAL(
        y_shape[-1] % tt::constants::TILE_HW == 0,
        "packed_y hidden size ({}) must be divisible by 1024 for the fused row-block tilizer",
        y_shape[-1]);
    TT_FATAL(
        y_shape[-1] / tt::constants::TILE_HW <= 8,
        "packed_y hidden size ({}) needs more than eight accumulator tiles per row",
        y_shape[-1]);
    TT_FATAL(
        tensors.packed_y.buffer()->num_pages() == y_shape[-2],
        "packed_y must expose one ROW_MAJOR page per packed row (expected {}, got {})",
        y_shape[-2],
        tensors.packed_y.buffer()->num_pages());

    constexpr uint32_t max_tokens = 2048;
    TT_FATAL(op.tokens > 0 && op.tokens <= max_tokens, "tokens ({}) must be in [1,{}]", op.tokens, max_tokens);
    TT_FATAL(
        op.tokens % tt::constants::TILE_HEIGHT == 0,
        "tokens ({}) must be divisible by {}",
        op.tokens,
        tt::constants::TILE_HEIGHT);
    TT_FATAL(op.topk > 0 && op.topk <= 16, "topk ({}) must be in [1,16]", op.topk);
    const uint64_t slots = static_cast<uint64_t>(op.tokens) * op.topk;
    if (op.assignment_addressed) {
        TT_FATAL(
            y_shape[-2] == slots,
            "assignment-addressed packed_y rows ({}) must equal tokens*topk ({})",
            y_shape[-2],
            slots);
    }

    if (tensors.optional_output.has_value()) {
        const auto& output = *tensors.optional_output;
        const auto expected_shape = ttnn::Shape({1, 1, op.tokens, static_cast<uint32_t>(y_shape[-1])});
        TT_FATAL(
            output.logical_shape() == expected_shape,
            "optional_output shape must be {}, got {}",
            expected_shape,
            output.logical_shape());
        TT_FATAL(
            output.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
            "optional_output must be BFLOAT8_B, got {}",
            output.dtype());
        TT_FATAL(
            output.layout() == tt::tt_metal::Layout::TILE, "optional_output must be TILE, got {}", output.layout());
    }

    TT_FATAL(
        tensors.topk_weights.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "topk_weights must be BFLOAT16, got {}",
        tensors.topk_weights.dtype());
    TT_FATAL(
        tensors.topk_weights.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "topk_weights must be ROW_MAJOR, got {}",
        tensors.topk_weights.layout());
    const auto& weight_shape = tensors.topk_weights.logical_shape();
    TT_FATAL(
        weight_shape.rank() == 4 && weight_shape[0] == 1 && weight_shape[1] == 1 && weight_shape[2] == op.tokens &&
            weight_shape[3] == op.topk,
        "topk_weights must have shape [1,1,tokens,topk] == [1,1,{},{}], got {}",
        op.tokens,
        op.topk,
        weight_shape);
    TT_FATAL(
        tensors.topk_weights.buffer()->num_pages() == op.tokens,
        "topk_weights must expose one ROW_MAJOR page per token (expected {}, got {})",
        op.tokens,
        tensors.topk_weights.buffer()->num_pages());

    for (const auto& [name, tensor] : std::initializer_list<std::pair<const char*, const Tensor&>>{
             {"slot_to_packed_row", tensors.slot_to_packed_row}, {"slot_is_local", tensors.slot_is_local}}) {
        TT_FATAL(tensor.dtype() == tt::tt_metal::DataType::UINT32, "{} must be UINT32", name);
        TT_FATAL(tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "{} must be ROW_MAJOR", name);
        const auto& shape = tensor.logical_shape();
        TT_FATAL(
            shape.rank() == 2 && shape[0] == 1 && shape[-1] == slots,
            "{} must have shape [1,tokens*topk] == [1,{}], got {}",
            name,
            slots,
            shape);
        TT_FATAL(tensor.buffer()->num_pages() == 1, "{} must fit in one ROW_MAJOR page", name);
    }
}

void TopkLocalCombineDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& op, const tensor_args_t& tensors) {
    // Tensor addresses and every data tensor are runtime-patched. Revalidate
    // their full contract before a cached program can see them.
    validate_on_program_cache_miss(op, tensors);
}

TopkLocalCombineDeviceOperation::spec_return_value_t TopkLocalCombineDeviceOperation::compute_output_specs(
    const operation_attributes_t& op, const tensor_args_t& tensors) {
    if (tensors.optional_output.has_value()) {
        return tensors.optional_output->tensor_spec();
    }
    const uint32_t hidden = tensors.packed_y.logical_shape()[-1];
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, op.tokens, hidden}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT8_B,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

TopkLocalCombineDeviceOperation::topology_return_value_t TopkLocalCombineDeviceOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& tensors) {
    if (tensors.optional_output.has_value()) {
        return {tensors.optional_output->tensor_topology()};
    }
    return {tensors.slot_to_packed_row.tensor_topology()};
}

TopkLocalCombineDeviceOperation::tensor_return_value_t TopkLocalCombineDeviceOperation::create_output_tensors(
    const operation_attributes_t& op, const tensor_args_t& tensors) {
    if (tensors.optional_output.has_value()) {
        return *tensors.optional_output;
    }
    return create_device_tensor(compute_output_specs(op, tensors), tensors.packed_y.device());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe

namespace ttnn::prim {

ttnn::Tensor topk_local_combine(
    const ttnn::Tensor& packed_y,
    const ttnn::Tensor& topk_weights,
    const ttnn::Tensor& slot_to_packed_row,
    const ttnn::Tensor& slot_is_local,
    uint32_t tokens,
    uint32_t topk,
    bool assignment_addressed,
    const std::optional<ttnn::Tensor>& optional_output) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe::TopkLocalCombineDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .tokens = tokens, .topk = topk, .assignment_addressed = assignment_addressed},
        OperationType::tensor_args_t{
            .packed_y = packed_y,
            .topk_weights = topk_weights,
            .slot_to_packed_row = slot_to_packed_row,
            .slot_is_local = slot_is_local,
            .optional_output = optional_output,
        });
}

}  // namespace ttnn::prim

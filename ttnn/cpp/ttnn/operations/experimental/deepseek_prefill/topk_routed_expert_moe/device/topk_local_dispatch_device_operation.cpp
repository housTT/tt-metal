// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_local_dispatch_device_operation.hpp"

#include <algorithm>
#include <cstdint>
#include <variant>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe {

namespace {

bool is_dram_interleaved(const ttnn::Tensor& tensor) {
    const auto& memory_config = tensor.memory_config();
    return memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

bool is_fully_replicated(const tt::tt_metal::TensorTopology& topology) {
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    for (const auto& placement : topology.placements()) {
        if (std::holds_alternative<Shard>(placement)) {
            return false;
        }
    }
    return true;
}

bool has_same_mesh_footprint(const tt::tt_metal::TensorTopology& lhs, const tt::tt_metal::TensorTopology& rhs) {
    return lhs.distribution_shape() == rhs.distribution_shape() && lhs.mesh_coords() == rhs.mesh_coords();
}

bool is_device_unique(const tt::tt_metal::TensorTopology& topology) {
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    // Replication is harmless on a unit mesh and keeps the primitive directly
    // testable there. On a real expert-parallel mesh, every non-unit mesh axis
    // must shard so no pair of chips can silently consume the same mapping.
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

void validate_device_tensor(const ttnn::Tensor& tensor, const char* name, const ttnn::Tensor& reference) {
    TT_FATAL(tensor.storage_type() == ttnn::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a device buffer", name);
    TT_FATAL(tensor.device() == reference.device(), "{} must be on the same device as x", name);
    TT_FATAL(is_dram_interleaved(tensor), "{} must be DRAM-interleaved", name);
}

tt::tt_metal::TensorSpec row_major_spec(const ttnn::Shape& shape, tt::tt_metal::DataType dtype) {
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            dtype,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

}  // namespace

void TopkLocalDispatchDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& op, const tensor_args_t& tensors) {
    validate_device_tensor(tensors.x, "x", tensors.x);
    validate_device_tensor(tensors.topk_indices, "topk_indices", tensors.x);
    validate_device_tensor(tensors.global_to_local_expert, "global_to_local_expert", tensors.x);

    const auto& x_topology = tensors.x.tensor_topology();
    const auto& index_topology = tensors.topk_indices.tensor_topology();
    const auto& mapping_topology = tensors.global_to_local_expert.tensor_topology();
    TT_FATAL(is_fully_replicated(x_topology), "x must be fully replicated across the expert-parallel mesh");
    TT_FATAL(index_topology == x_topology, "topk_indices must have the same fully replicated topology as x");
    TT_FATAL(
        has_same_mesh_footprint(mapping_topology, x_topology),
        "global_to_local_expert must cover the same mesh coordinates as x");
    TT_FATAL(
        is_device_unique(mapping_topology),
        "global_to_local_expert must be device-unique (sharded) on a multi-device mesh");

    TT_FATAL(tensors.x.dtype() == tt::tt_metal::DataType::BFLOAT16, "x must be BFLOAT16, got {}", tensors.x.dtype());
    TT_FATAL(
        tensors.x.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "x must be ROW_MAJOR (untilized exactly once before dispatch), got {}",
        tensors.x.layout());
    TT_FATAL(tensors.x.logical_shape().rank() == 4, "x must have shape [1,1,T,H], got {}", tensors.x.logical_shape());
    TT_FATAL(
        tensors.x.logical_shape()[0] == 1 && tensors.x.logical_shape()[1] == 1,
        "x leading dimensions must both be 1, got {}",
        tensors.x.logical_shape());

    TT_FATAL(
        tensors.topk_indices.dtype() == tt::tt_metal::DataType::UINT32,
        "topk_indices must be UINT32 straight from ttnn.topk, got {}",
        tensors.topk_indices.dtype());
    TT_FATAL(
        tensors.topk_indices.layout() == tt::tt_metal::Layout::TILE,
        "topk_indices must be TILE layout, got {}",
        tensors.topk_indices.layout());
    TT_FATAL(
        tensors.topk_indices.logical_shape().rank() == 4,
        "topk_indices must have shape [1,1,T,K], got {}",
        tensors.topk_indices.logical_shape());
    TT_FATAL(
        tensors.topk_indices.logical_shape()[0] == 1 && tensors.topk_indices.logical_shape()[1] == 1,
        "topk_indices leading dimensions must both be 1, got {}",
        tensors.topk_indices.logical_shape());

    const uint32_t tokens = tensors.x.logical_shape()[-2];
    const uint32_t hidden = tensors.x.logical_shape()[-1];
    const uint32_t topk_tokens = tensors.topk_indices.logical_shape()[-2];
    const uint32_t topk = tensors.topk_indices.logical_shape()[-1];
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    constexpr uint32_t face_width = 16;
    constexpr uint32_t max_tokens = 1024;
    TT_FATAL(tokens == topk_tokens, "x token count ({}) must equal topk_indices token count ({})", tokens, topk_tokens);
    TT_FATAL(tokens > 0 && tokens % tile_height == 0, "token count ({}) must be positive and tile-aligned", tokens);
    TT_FATAL(
        tokens <= max_tokens,
        "token count ({}) exceeds the top-k-native planner limit ({}); sub-chunk to {} tokens or use gathered fallback",
        tokens,
        max_tokens,
        max_tokens);
    TT_FATAL(hidden > 0 && hidden % tile_height == 0, "hidden size ({}) must be positive and tile-aligned", hidden);
    TT_FATAL(
        topk > 0 && topk <= face_width,
        "top-k ({}) must be in [1, {}]; the planner reads the left tile faces directly",
        topk,
        face_width);
    const auto& index_tile = tensors.topk_indices.tensor_spec().page_config().get_tile();
    TT_FATAL(
        index_tile.get_height() == tile_height && index_tile.get_width() == tile_height,
        "topk_indices must use 32x32 tiles, got {}x{}",
        index_tile.get_height(),
        index_tile.get_width());
    TT_FATAL(
        tensors.topk_indices.padded_shape()[-1] == tile_height,
        "topk_indices padded width must be one tile ({}), got {}",
        tile_height,
        tensors.topk_indices.padded_shape()[-1]);

    TT_FATAL(
        tensors.global_to_local_expert.dtype() == tt::tt_metal::DataType::UINT32,
        "global_to_local_expert must be UINT32, got {}",
        tensors.global_to_local_expert.dtype());
    TT_FATAL(
        tensors.global_to_local_expert.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "global_to_local_expert must be ROW_MAJOR, got {}",
        tensors.global_to_local_expert.layout());
    const auto& mapping_shape = tensors.global_to_local_expert.logical_shape();
    TT_FATAL(
        mapping_shape.rank() == 2 && mapping_shape[0] == 1,
        "global_to_local_expert must have shape [1,E_global], got {}",
        mapping_shape);
    const uint32_t num_global_experts = mapping_shape[-1];
    TT_FATAL(
        num_global_experts > 0 && num_global_experts <= 1024, "E_global ({}) must be in [1,1024]", num_global_experts);
    TT_FATAL(
        op.num_local_experts > 0 && op.num_local_experts <= 64, "E_local ({}) must be in [1,64]", op.num_local_experts);
    TT_FATAL(
        op.num_local_experts <= num_global_experts,
        "E_local ({}) cannot exceed E_global ({})",
        op.num_local_experts,
        num_global_experts);
    TT_FATAL(topk <= num_global_experts, "top-k ({}) cannot exceed E_global ({})", topk, num_global_experts);
    TT_FATAL(
        op.valid_tokens > 0 && op.valid_tokens <= tokens,
        "valid_tokens ({}) must be in [1,{}]",
        op.valid_tokens,
        tokens);

    // The planner reads each vector through page 0 and the dispatcher copies
    // whole activation sticks. Pin those addressing assumptions here.
    TT_FATAL(
        tensors.global_to_local_expert.buffer()->num_pages() == 1,
        "global_to_local_expert must fit in one ROW_MAJOR page, got {} pages",
        tensors.global_to_local_expert.buffer()->num_pages());
    TT_FATAL(
        tensors.x.buffer()->num_pages() == tokens,
        "x must expose one ROW_MAJOR page per token (expected {}, got {})",
        tokens,
        tensors.x.buffer()->num_pages());
}

void TopkLocalDispatchDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& op, const tensor_args_t& tensors) {
    // Addresses, mapping contents and valid_tokens are runtime data. Re-run
    // every shape/layout/device check on hits before patching those values.
    validate_on_program_cache_miss(op, tensors);
}

TopkLocalDispatchDeviceOperation::spec_return_value_t TopkLocalDispatchDeviceOperation::compute_output_specs(
    const operation_attributes_t& op, const tensor_args_t& tensors) {
    const uint32_t tokens = tensors.x.logical_shape()[-2];
    const uint32_t hidden = tensors.x.logical_shape()[-1];
    const uint32_t topk = tensors.topk_indices.logical_shape()[-1];
    const uint32_t num_global_experts = tensors.global_to_local_expert.logical_shape()[-1];
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    const uint32_t capacity = tokens * topk + (tile_height - 1) * op.num_local_experts;
    const uint32_t slots = tokens * topk;

    return {
        row_major_spec(ttnn::Shape({1, 1, op.materialize_x ? capacity : 1, hidden}), tt::tt_metal::DataType::BFLOAT16),
        row_major_spec(ttnn::Shape({1, num_global_experts}), tt::tt_metal::DataType::UINT32),
        row_major_spec(ttnn::Shape({1, num_global_experts}), tt::tt_metal::DataType::UINT32),
        row_major_spec(ttnn::Shape({1, op.num_local_experts}), tt::tt_metal::DataType::UINT32),
        row_major_spec(ttnn::Shape({1, capacity}), tt::tt_metal::DataType::UINT32),
        row_major_spec(ttnn::Shape({1, slots}), tt::tt_metal::DataType::UINT32),
        row_major_spec(ttnn::Shape({1, slots}), tt::tt_metal::DataType::UINT32),
    };
}

TopkLocalDispatchDeviceOperation::topology_return_value_t TopkLocalDispatchDeviceOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& tensors) {
    // The global->local map is deliberately mesh-sharded and differs on every
    // expert-parallel device. Mark every output as unique on every mesh axis;
    // inheriting x's replicated topology would incorrectly claim equal plans.
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    const auto& mapping_topology = tensors.global_to_local_expert.tensor_topology();
    const auto& distribution_shape = mapping_topology.distribution_shape();
    ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements;
    for (size_t axis = 0; axis < distribution_shape.dims(); ++axis) {
        placements.push_back(Shard{static_cast<int>(axis)});
    }
    const auto topology = tt::tt_metal::TensorTopology(distribution_shape, placements, mapping_topology.mesh_coords());
    return {topology, topology, topology, topology, topology, topology, topology};
}

TopkLocalDispatchDeviceOperation::tensor_return_value_t TopkLocalDispatchDeviceOperation::create_output_tensors(
    const operation_attributes_t& op, const tensor_args_t& tensors) {
    const auto specs = compute_output_specs(op, tensors);
    return {
        create_device_tensor(specs[0], tensors.x.device()),
        create_device_tensor(specs[1], tensors.x.device()),
        create_device_tensor(specs[2], tensors.x.device()),
        create_device_tensor(specs[3], tensors.x.device()),
        create_device_tensor(specs[4], tensors.x.device()),
        create_device_tensor(specs[5], tensors.x.device()),
        create_device_tensor(specs[6], tensors.x.device()),
    };
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe

namespace ttnn::prim {

ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe::TopkLocalDispatchTensors topk_local_dispatch(
    const ttnn::Tensor& x,
    const ttnn::Tensor& topk_indices,
    const ttnn::Tensor& global_to_local_expert,
    uint32_t num_local_experts,
    uint32_t valid_tokens,
    bool materialize_x) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe::TopkLocalDispatchDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .num_local_experts = num_local_experts,
            .materialize_x = materialize_x,
            .valid_tokens = valid_tokens,
        },
        OperationType::tensor_args_t{
            .x = x,
            .topk_indices = topk_indices,
            .global_to_local_expert = global_to_local_expert,
        });
}

}  // namespace ttnn::prim

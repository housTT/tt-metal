// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_sort_slabs_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {
constexpr const char* sort_op_name = "moe_sort_slabs";
}  // namespace

MoeSortSlabsOperation::program_factory_t MoeSortSlabsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoeSortSlabsProgramFactory{};
}

void MoeSortSlabsOperation::validate_on_program_cache_miss(const operation_attributes_t& a, const tensor_args_t& in) {
    TT_FATAL(a.k > 0 && a.k <= tt::constants::TILE_WIDTH, "{}: k must be in 1..32", sort_op_name);
    TT_FATAL(
        a.rows > 0 && a.rows % tt::constants::TILE_HEIGHT == 0 && a.rows <= 1024,
        "{}: rows must be a tile multiple <= 1024",
        sort_op_name);
    TT_FATAL(a.local_experts > 0 && a.local_experts <= 1024, "{}: local_experts out of range", sort_op_name);
    const uint32_t needed =
        a.local_experts + (a.rows * a.k + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT + 1;
    TT_FATAL(
        a.slab_capacity >= needed,
        "{}: slab_capacity {} must be >= local_experts + ceil(rows*k/32) + 1 = {}",
        sort_op_name,
        a.slab_capacity,
        needed);
    for (const auto& [tensor, name, dtype] : std::array{
             std::tuple{&in.indices, "indices", DataType::UINT16},
             std::tuple{&in.scores, "scores", DataType::BFLOAT16}}) {
        kda_factory_detail::check_allocated_device_tensor(*tensor, sort_op_name, name);
        kda_factory_detail::check_layout(*tensor, Layout::TILE, sort_op_name, name);
        kda_factory_detail::check_dtype(*tensor, dtype, sort_op_name, name);
        kda_factory_detail::check_interleaved(*tensor, sort_op_name, name);
        TT_FATAL(
            tensor->logical_shape() == Shape({1, 1, a.rows, a.k}), "{}: {} must be [1,1,rows,k]", sort_op_name, name);
    }
    kda_factory_detail::check_allocated_device_tensor(in.rank_base, sort_op_name, "rank_base");
    kda_factory_detail::check_dtype(in.rank_base, DataType::INT32, sort_op_name, "rank_base");
    kda_factory_detail::check_same_device(in.indices, in.scores, sort_op_name, "scores");
    kda_factory_detail::check_same_device(in.indices, in.rank_base, sort_op_name, "rank_base");
    kda_factory_detail::check_output_interleaved(a.output_mem_config, sort_op_name);
}

MoeSortSlabsOperation::spec_return_value_t MoeSortSlabsOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    const auto tile_bf16 = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), a.output_mem_config);
    return {
        // slab_rows: row id of each slab position (row-major, one stick)
        TensorSpec(
            Shape({1, 1, 1, a.slab_capacity * tt::constants::TILE_HEIGHT}),
            TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), a.output_mem_config)),
        // slab_experts: local bank slot of each slab (row-major, one stick)
        TensorSpec(
            Shape({1, 1, 1, a.slab_capacity}),
            TensorLayout(DataType::UINT16, PageConfig(Layout::ROW_MAJOR), a.output_mem_config)),
        // slab_pos: slab position of every (row, k) entry (dummy last position for non-local)
        TensorSpec(
            Shape({1, 1, a.rows, a.k}), TensorLayout(DataType::INT32, PageConfig(Layout::TILE), a.output_mem_config)),
        // local_scores: routing weight masked to local experts
        TensorSpec(Shape({1, 1, a.rows, a.k}), tile_bf16),
    };
}

MoeSortSlabsOperation::tensor_return_value_t MoeSortSlabsOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    auto specs = compute_output_specs(a, in);
    tensor_return_value_t out;
    for (const auto& spec : specs) {
        out.push_back(create_device_tensor(spec, in.indices.device()));
    }
    return out;
}

std::vector<Tensor> moe_sort_slabs(
    const Tensor& indices,
    const Tensor& scores,
    const Tensor& rank_base,
    uint32_t local_experts,
    uint32_t slab_capacity,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    const auto& shape = indices.logical_shape();
    TT_FATAL(shape.rank() == 4, "{}: indices must be rank 4", sort_op_name);
    return ttnn::device_operation::launch<MoeSortSlabsOperation>(
        MoeSortSlabsParams{
            .rows = shape[2],
            .k = shape[3],
            .local_experts = local_experts,
            .slab_capacity = slab_capacity,
            .output_mem_config = output_mem_config},
        MoeSortSlabsInputs{.indices = indices, .scores = scores, .rank_base = rank_base});
}

}  // namespace ttnn::experimental::prim

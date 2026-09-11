// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_weighted_sum_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {
constexpr const char* wsum_op_name = "moe_weighted_sum";
}  // namespace

MoeWeightedSumOperation::program_factory_t MoeWeightedSumOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoeWeightedSumProgramFactory{};
}

void MoeWeightedSumOperation::validate_on_program_cache_miss(const operation_attributes_t& a, const tensor_args_t& in) {
    TT_FATAL(a.k > 0 && a.k <= 16, "{}: k must be in 1..16", wsum_op_name);
    TT_FATAL(a.width % tt::constants::TILE_WIDTH == 0, "{}: width must be tile aligned", wsum_op_name);
    for (const auto& [tensor, name] : std::array{std::pair{&in.groups, "groups"}, std::pair{&in.scores, "scores"}}) {
        kda_factory_detail::check_allocated_device_tensor(*tensor, wsum_op_name, name);
        kda_factory_detail::check_layout(*tensor, Layout::TILE, wsum_op_name, name);
        kda_factory_detail::check_dtype(*tensor, DataType::BFLOAT16, wsum_op_name, name);
        kda_factory_detail::check_interleaved(*tensor, wsum_op_name, name);
    }
    const auto& g = in.groups.logical_shape();
    TT_FATAL(
        g.rank() == 4 && g[0] == 1 && g[1] == a.k && g[2] >= 1 && g[2] <= tt::constants::TILE_HEIGHT && g[3] == a.width,
        "{}: groups must be [1,k,rows<=32,N], got {}",
        wsum_op_name,
        g);
    TT_FATAL(in.scores.logical_shape() == Shape({1, 1, 1, a.k}), "{}: scores must be [1,1,1,k]", wsum_op_name);
    kda_factory_detail::check_same_device(in.groups, in.scores, wsum_op_name, "scores");
    kda_factory_detail::check_output_interleaved(a.output_mem_config, wsum_op_name);
    kda_factory_detail::check_compute_config(a.compute_kernel_config, wsum_op_name);
}

MoeWeightedSumOperation::spec_return_value_t MoeWeightedSumOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    return TensorSpec(
        Shape({1, 1, a.rows, a.width}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), a.output_mem_config));
}

MoeWeightedSumOperation::tensor_return_value_t MoeWeightedSumOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(a, in), in.groups.device());
}

Tensor moe_weighted_sum(
    const Tensor& groups,
    const Tensor& scores,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& shape = groups.logical_shape();
    TT_FATAL(shape.rank() == 4, "{}: groups must be rank 4", wsum_op_name);
    return ttnn::device_operation::launch<MoeWeightedSumOperation>(
        MoeWeightedSumParams{
            .k = shape[1],
            .rows = shape[2],
            .width = shape[3],
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        MoeWeightedSumInputs{.groups = groups, .scores = scores});
}

}  // namespace ttnn::experimental::prim

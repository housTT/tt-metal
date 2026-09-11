// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_route_topk_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {
constexpr const char* route_op_name = "moe_route_topk";
}  // namespace

MoeRouteTopkOperation::program_factory_t MoeRouteTopkOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoeRouteTopkProgramFactory{};
}

void MoeRouteTopkOperation::validate_on_program_cache_miss(const operation_attributes_t& a, const tensor_args_t& in) {
    TT_FATAL(a.k > 0 && a.k <= 16, "{}: k must be in 1..16", route_op_name);
    TT_FATAL(a.num_experts % tt::constants::TILE_WIDTH == 0, "{}: experts must be tile aligned", route_op_name);
    TT_FATAL(a.local_experts > 0 && a.local_experts <= a.num_experts, "{}: local_experts out of range", route_op_name);
    kda_factory_detail::check_allocated_device_tensor(in.logits, route_op_name, "logits");
    kda_factory_detail::check_layout(in.logits, Layout::TILE, route_op_name, "logits");
    kda_factory_detail::check_dtype(in.logits, DataType::BFLOAT16, route_op_name, "logits");
    kda_factory_detail::check_interleaved(in.logits, route_op_name, "logits");
    const auto& shape = in.logits.logical_shape();
    TT_FATAL(shape.rank() == 4 && shape[0] == 1 && shape[1] == 1, "{}: logits must be [1,1,R,E]", route_op_name);
    TT_FATAL(shape[3] == a.num_experts, "{}: logits width {} != experts {}", route_op_name, shape[3], a.num_experts);
    kda_factory_detail::check_allocated_device_tensor(in.rank_base, route_op_name, "rank_base");
    kda_factory_detail::check_dtype(in.rank_base, DataType::INT32, route_op_name, "rank_base");
    kda_factory_detail::check_same_device(in.logits, in.rank_base, route_op_name, "rank_base");
    kda_factory_detail::check_output_interleaved(a.output_mem_config, route_op_name);
}

MoeRouteTopkOperation::spec_return_value_t MoeRouteTopkOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    return {
        TensorSpec(
            Shape({1, 1, 1, a.k}), TensorLayout(DataType::UINT16, PageConfig(Layout::ROW_MAJOR), a.output_mem_config)),
        TensorSpec(
            Shape({1, 1, 1, a.k}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), a.output_mem_config)),
    };
}

MoeRouteTopkOperation::tensor_return_value_t MoeRouteTopkOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    auto specs = compute_output_specs(a, in);
    return {create_device_tensor(specs[0], in.logits.device()), create_device_tensor(specs[1], in.logits.device())};
}

std::tuple<Tensor, Tensor> moe_route_topk(
    const Tensor& logits,
    const Tensor& rank_base,
    uint32_t k,
    uint32_t local_experts,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    const auto& shape = logits.logical_shape();
    TT_FATAL(shape.rank() == 4, "{}: logits must be rank 4", route_op_name);
    auto outputs = ttnn::device_operation::launch<MoeRouteTopkOperation>(
        MoeRouteTopkParams{
            .k = k, .num_experts = shape[3], .local_experts = local_experts, .output_mem_config = output_mem_config},
        MoeRouteTopkInputs{.logits = logits, .rank_base = rank_base});
    return {outputs[0], outputs[1]};
}

}  // namespace ttnn::experimental::prim

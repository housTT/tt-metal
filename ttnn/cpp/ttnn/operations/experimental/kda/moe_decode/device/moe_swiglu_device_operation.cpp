// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_swiglu_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {
constexpr const char* swiglu_op_name = "moe_swiglu";
}  // namespace

MoeSwigluOperation::program_factory_t MoeSwigluOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoeSwigluProgramFactory{};
}

void MoeSwigluOperation::validate_on_program_cache_miss(const operation_attributes_t& a, const tensor_args_t& in) {
    TT_FATAL(a.groups > 0, "{}: groups must be positive", swiglu_op_name);
    TT_FATAL(a.rows > 0 && a.rows <= tt::constants::TILE_HEIGHT, "{}: rows must be in 1..32", swiglu_op_name);
    TT_FATAL(a.intermediate % tt::constants::TILE_WIDTH == 0, "{}: intermediate must be tile aligned", swiglu_op_name);
    kda_factory_detail::check_allocated_device_tensor(in.gate_up, swiglu_op_name, "gate_up");
    kda_factory_detail::check_layout(in.gate_up, Layout::TILE, swiglu_op_name, "gate_up");
    kda_factory_detail::check_dtype(in.gate_up, DataType::BFLOAT16, swiglu_op_name, "gate_up");
    kda_factory_detail::check_interleaved(in.gate_up, swiglu_op_name, "gate_up");
    const auto& g = in.gate_up.logical_shape();
    TT_FATAL(
        g.rank() == 4 && g[0] == 1 && g[1] == a.groups && g[2] == a.rows && g[3] == 2 * a.intermediate,
        "{}: gate_up must be [1,G,rows,2*I], got {}",
        swiglu_op_name,
        g);
    kda_factory_detail::check_output_interleaved(a.output_mem_config, swiglu_op_name);
    kda_factory_detail::check_compute_config(a.compute_kernel_config, swiglu_op_name);
}

MoeSwigluOperation::spec_return_value_t MoeSwigluOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    return TensorSpec(
        Shape({1, a.groups, a.rows, a.intermediate}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), a.output_mem_config));
}

MoeSwigluOperation::tensor_return_value_t MoeSwigluOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(a, in), in.gate_up.device());
}

Tensor moe_swiglu(
    const Tensor& gate_up,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& shape = gate_up.logical_shape();
    TT_FATAL(shape.rank() == 4, "{}: gate_up must be rank 4", swiglu_op_name);
    return ttnn::device_operation::launch<MoeSwigluOperation>(
        MoeSwigluParams{
            .groups = shape[1],
            .rows = shape[2],
            .intermediate = shape[3] / 2,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        MoeSwigluInputs{.gate_up = gate_up});
}

}  // namespace ttnn::experimental::prim

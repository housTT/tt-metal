// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hc_inject_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {
constexpr const char* inject_op_name = "hc_inject";

void inject_check_tile_input(const Tensor& tensor, const char* name) {
    kda_factory_detail::check_allocated_device_tensor(tensor, inject_op_name, name);
    kda_factory_detail::check_layout(tensor, Layout::TILE, inject_op_name, name);
    kda_factory_detail::check_dtype(tensor, DataType::BFLOAT16, inject_op_name, name);
    kda_factory_detail::check_interleaved(tensor, inject_op_name, name);
}

void inject_check_shape(const Tensor& tensor, const Shape& expected, const char* name) {
    TT_FATAL(
        tensor.logical_shape() == expected,
        "{}: {} shape must be {}, got {}",
        inject_op_name,
        name,
        expected,
        tensor.logical_shape());
}
}  // namespace

HcInjectOperation::program_factory_t HcInjectOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return HcInjectProgramFactory{};
}

void HcInjectOperation::validate_on_program_cache_miss(const operation_attributes_t& a, const tensor_args_t& in) {
    TT_FATAL(a.streams > 0 && a.streams <= 16, "{}: streams must be in 1..16", inject_op_name);
    TT_FATAL(a.width > 0 && a.width % tt::constants::TILE_WIDTH == 0, "{}: width must be tile aligned", inject_op_name);
    inject_check_tile_input(in.hyper, "hyper");
    inject_check_tile_input(in.block, "block");
    inject_check_tile_input(in.injection, "injection");
    inject_check_shape(in.hyper, Shape({1, 1, a.streams, a.width}), "hyper");
    inject_check_shape(in.block, Shape({1, 1, 1, a.width}), "block");
    inject_check_shape(in.injection, Shape({1, 1, 1, a.streams}), "injection");
    kda_factory_detail::check_same_device(in.hyper, in.block, inject_op_name, "block");
    kda_factory_detail::check_same_device(in.hyper, in.injection, inject_op_name, "injection");
    kda_factory_detail::check_output_interleaved(a.output_mem_config, inject_op_name);
    kda_factory_detail::check_compute_config(a.compute_kernel_config, inject_op_name);
}

HcInjectOperation::spec_return_value_t HcInjectOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    return TensorSpec(
        Shape({1, 1, a.streams, a.width}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), a.output_mem_config));
}

HcInjectOperation::tensor_return_value_t HcInjectOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(a, in), in.hyper.device());
}

Tensor hc_inject(
    const Tensor& hyper,
    const Tensor& block,
    const Tensor& injection,
    uint32_t streams,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& shape = hyper.logical_shape();
    TT_FATAL(shape.rank() == 4, "{}: hyper must be rank 4", inject_op_name);
    return ttnn::device_operation::launch<HcInjectOperation>(
        HcInjectParams{
            .streams = streams,
            .width = shape[3],
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        HcInjectInputs{.hyper = hyper, .block = block, .injection = injection});
}

}  // namespace ttnn::experimental::prim

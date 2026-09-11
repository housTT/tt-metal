// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gdn_decode_step_device_operation.hpp"

#include <array>
#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {
namespace {

constexpr auto op_name = "gdn_decode_step";

void check_tile_input(const Tensor& tensor, DataType dtype, const char* name) {
    kda_factory_detail::check_allocated_device_tensor(tensor, op_name, name);
    kda_factory_detail::check_layout(tensor, Layout::TILE, op_name, name);
    kda_factory_detail::check_dtype(tensor, dtype, op_name, name);
    kda_factory_detail::check_interleaved(tensor, op_name, name);
}

void check_shape(const Tensor& tensor, const Shape& expected, const char* name) {
    TT_FATAL(
        tensor.logical_shape() == expected,
        "{}: {} shape must be {}, got {}",
        op_name,
        name,
        expected,
        tensor.logical_shape());
}

}  // namespace

GdnDecodeStepOperation::program_factory_t GdnDecodeStepOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GdnDecodeStepProgramFactory{};
}

void GdnDecodeStepOperation::validate_on_program_cache_miss(const operation_attributes_t& a, const tensor_args_t& in) {
    TT_FATAL(a.batch > 0 && a.num_heads > 0, "{}: batch and heads must be positive", op_name);
    TT_FATAL(
        a.qk_head_repeat > 0 && a.num_heads % a.qk_head_repeat == 0, "{}: qk_head_repeat must divide heads", op_name);
    TT_FATAL(
        a.key_dim % tt::constants::TILE_WIDTH == 0 && a.value_dim % tt::constants::TILE_WIDTH == 0,
        "{}: K/V must be tile aligned",
        op_name);
    const uint32_t qk_heads = a.num_heads / a.qk_head_repeat;
    TT_FATAL(
        a.qkv_width == 2 * qk_heads * a.key_dim + a.num_heads * a.value_dim,
        "{}: qkv width {} must equal 2*Hk*K + H*V",
        op_name,
        a.qkv_width);
    TT_FATAL(a.qk_norm_epsilon > 0.0f && a.norm_epsilon > 0.0f, "{}: epsilons must be positive", op_name);
    TT_FATAL(
        a.output_dtype == DataType::FLOAT32 || a.output_dtype == DataType::BFLOAT16,
        "{}: output dtype must be FLOAT32 or BFLOAT16",
        op_name);
    const Shape row({a.batch, 1, 1, a.qkv_width});
    for (const auto& [tensor, name] : std::array{
             std::pair{&in.x, "x"},
             std::pair{&in.tap0, "tap0"},
             std::pair{&in.tap1, "tap1"},
             std::pair{&in.tap2, "tap2"}}) {
        check_tile_input(*tensor, DataType::FLOAT32, name);
        check_shape(*tensor, row, name);
        kda_factory_detail::check_same_device(in.x, *tensor, op_name, name);
    }
    const Shape weight_row({1, 1, 1, a.qkv_width});
    for (const auto& [tensor, name] : std::array{
             std::pair{&in.conv_w0, "conv_w0"},
             std::pair{&in.conv_w1, "conv_w1"},
             std::pair{&in.conv_w2, "conv_w2"},
             std::pair{&in.conv_w3, "conv_w3"}}) {
        kda_factory_detail::check_allocated_device_tensor(*tensor, op_name, name);
        kda_factory_detail::check_layout(*tensor, Layout::TILE, op_name, name);
        kda_factory_detail::check_interleaved(*tensor, op_name, name);
        TT_FATAL(
            tensor->dtype() == DataType::FLOAT32 || tensor->dtype() == DataType::BFLOAT16,
            "{}: conv weights must be FLOAT32 or BFLOAT16",
            op_name);
        TT_FATAL(tensor->dtype() == in.conv_w0.dtype(), "{}: conv weights must share one dtype", op_name);
        check_shape(*tensor, weight_row, name);
        kda_factory_detail::check_same_device(in.x, *tensor, op_name, name);
    }
    check_tile_input(in.beta, DataType::FLOAT32, "beta");
    check_tile_input(in.log_decay, DataType::FLOAT32, "log_decay");
    check_tile_input(in.state, DataType::FLOAT32, "state");
    check_tile_input(in.gate, DataType::BFLOAT16, "gate");
    check_tile_input(in.norm_weight, DataType::BFLOAT16, "norm_weight");
    check_shape(in.beta, Shape({a.batch, a.num_heads, 1, 1}), "beta");
    check_shape(in.log_decay, Shape({a.batch, a.num_heads, 1, 1}), "log_decay");
    check_shape(in.state, Shape({a.batch, a.num_heads, a.key_dim, a.value_dim}), "state");
    check_shape(in.gate, Shape({a.batch, 1, 1, a.num_heads * a.value_dim}), "gate");
    check_shape(in.norm_weight, Shape({1, 1, 1, a.value_dim}), "norm_weight");
    for (const auto* tensor : std::array{&in.beta, &in.log_decay, &in.state, &in.gate, &in.norm_weight}) {
        kda_factory_detail::check_same_device(in.x, *tensor, op_name, "input");
    }
    kda_factory_detail::check_output_interleaved(a.output_mem_config, op_name);
    kda_factory_detail::check_compute_config(a.compute_kernel_config, op_name);
    if (in.state_output.has_value()) {
        check_tile_input(*in.state_output, DataType::FLOAT32, "state_output");
        check_shape(*in.state_output, Shape({a.batch, a.num_heads, a.key_dim, a.value_dim}), "state_output");
        kda_factory_detail::check_same_device(in.x, *in.state_output, op_name, "state_output");
    }
}

GdnDecodeStepOperation::spec_return_value_t GdnDecodeStepOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return {
        TensorSpec(
            Shape({a.batch, 1, 1, a.num_heads * a.value_dim}),
            TensorLayout(a.output_dtype, PageConfig(Layout::TILE), a.output_mem_config)),
        in.state_output.has_value()
            ? in.state_output->tensor_spec()
            : TensorSpec(
                  Shape({a.batch, a.num_heads, a.key_dim, a.value_dim}),
                  TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), a.output_mem_config)),
    };
}

GdnDecodeStepOperation::tensor_return_value_t GdnDecodeStepOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    auto specs = compute_output_specs(a, in);
    return {
        create_device_tensor(specs[0], in.x.device()),
        in.state_output.has_value() ? *in.state_output : create_device_tensor(specs[1], in.x.device())};
}

std::tuple<Tensor, Tensor> gdn_decode_step(
    const Tensor& x,
    const Tensor& tap0,
    const Tensor& tap1,
    const Tensor& tap2,
    const Tensor& conv_w0,
    const Tensor& conv_w1,
    const Tensor& conv_w2,
    const Tensor& conv_w3,
    const Tensor& beta,
    const Tensor& log_decay,
    const Tensor& state,
    const Tensor& gate,
    const Tensor& norm_weight,
    uint32_t num_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    const std::optional<Tensor>& state_output,
    uint32_t qk_head_repeat,
    float qk_norm_epsilon,
    float norm_epsilon,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& x_shape = x.logical_shape();
    TT_FATAL(x_shape.rank() == 4, "{}: x must be rank 4", op_name);
    auto outputs = ttnn::device_operation::launch<GdnDecodeStepOperation>(
        GdnDecodeStepParams{
            .batch = x_shape[0],
            .num_heads = num_heads,
            .key_dim = key_dim,
            .value_dim = value_dim,
            .qkv_width = x_shape[3],
            .qk_head_repeat = qk_head_repeat,
            .qk_norm_epsilon = qk_norm_epsilon,
            .norm_epsilon = norm_epsilon,
            .output_dtype = output_dtype,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        GdnDecodeStepInputs{
            .x = x,
            .tap0 = tap0,
            .tap1 = tap1,
            .tap2 = tap2,
            .conv_w0 = conv_w0,
            .conv_w1 = conv_w1,
            .conv_w2 = conv_w2,
            .conv_w3 = conv_w3,
            .beta = beta,
            .log_decay = log_decay,
            .state = state,
            .gate = gate,
            .norm_weight = norm_weight,
            .state_output = state_output});
    return {outputs[0], outputs[1]};
}

}  // namespace ttnn::experimental::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_gated_delta_rule_device_operation.hpp"

#include <array>
#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {
namespace {

constexpr auto op_name = "recurrent_gated_delta_rule";

void check_input(const Tensor& tensor, const char* name) {
    kda_factory_detail::check_allocated_device_tensor(tensor, op_name, name);
    kda_factory_detail::check_layout(tensor, Layout::TILE, op_name, name);
    kda_factory_detail::check_dtype(tensor, DataType::FLOAT32, op_name, name);
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

RecurrentGatedDeltaRuleOperation::program_factory_t RecurrentGatedDeltaRuleOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RecurrentGatedDeltaRuleProgramFactory{};
}

void RecurrentGatedDeltaRuleOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    check_input(in.query, "query");
    check_input(in.key, "key");
    check_input(in.value, "value");
    check_input(in.beta, "beta");
    check_input(in.log_decay, "log_decay");
    check_input(in.state, "state");
    for (const auto* tensor : std::array{&in.key, &in.value, &in.beta, &in.log_decay, &in.state}) {
        kda_factory_detail::check_same_device(in.query, *tensor, op_name, "input");
    }
    kda_factory_detail::check_output_interleaved(attrs.output_mem_config, op_name);
    kda_factory_detail::check_compute_config(attrs.compute_kernel_config, op_name);
    TT_FATAL(attrs.batch > 0 && attrs.num_heads > 0, "{}: batch and heads must be positive", op_name);
    TT_FATAL(
        attrs.key_dim > 0 && attrs.key_dim % tt::constants::TILE_WIDTH == 0,
        "{}: key_dim must be positive and tile aligned",
        op_name);
    TT_FATAL(
        attrs.value_dim > 0 && attrs.value_dim % tt::constants::TILE_WIDTH == 0,
        "{}: value_dim must be positive and tile aligned",
        op_name);
    TT_FATAL(
        attrs.qk_head_repeat > 0 && attrs.num_heads % attrs.qk_head_repeat == 0,
        "{}: qk_head_repeat must divide the value head count",
        op_name);
    const uint32_t qk_heads = attrs.num_heads / attrs.qk_head_repeat;
    check_shape(in.query, Shape({attrs.batch, qk_heads, 1, attrs.key_dim}), "query");
    check_shape(in.key, Shape({attrs.batch, qk_heads, 1, attrs.key_dim}), "key");
    check_shape(in.value, Shape({attrs.batch, attrs.num_heads, 1, attrs.value_dim}), "value");
    check_shape(in.beta, Shape({attrs.batch, attrs.num_heads, 1, 1}), "beta");
    check_shape(in.log_decay, Shape({attrs.batch, attrs.num_heads, 1, 1}), "log_decay");
    check_shape(in.state, Shape({attrs.batch, attrs.num_heads, attrs.key_dim, attrs.value_dim}), "state");
    if (in.state_output.has_value()) {
        check_input(*in.state_output, "state_output");
        kda_factory_detail::check_same_device(in.query, *in.state_output, op_name, "state_output");
        check_shape(
            *in.state_output, Shape({attrs.batch, attrs.num_heads, attrs.key_dim, attrs.value_dim}), "state_output");
        const TensorSpec expected_state_spec(
            Shape({attrs.batch, attrs.num_heads, attrs.key_dim, attrs.value_dim}),
            TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config));
        TT_FATAL(
            in.state_output->tensor_spec() == expected_state_spec,
            "{}: state_output spec must match the requested recurrent-state output spec",
            op_name);
    }
}

RecurrentGatedDeltaRuleOperation::spec_return_value_t RecurrentGatedDeltaRuleOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return {
        TensorSpec(
            Shape({attrs.batch, attrs.num_heads, 1, attrs.value_dim}),
            TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config)),
        in.state_output.has_value()
            ? in.state_output->tensor_spec()
            : TensorSpec(
                  Shape({attrs.batch, attrs.num_heads, attrs.key_dim, attrs.value_dim}),
                  TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config)),
    };
}

RecurrentGatedDeltaRuleOperation::tensor_return_value_t RecurrentGatedDeltaRuleOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {
        create_device_tensor(specs[0], in.query.device()),
        in.state_output.has_value() ? *in.state_output : create_device_tensor(specs[1], in.query.device())};
}

std::tuple<Tensor, Tensor> recurrent_gated_delta_rule(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& beta,
    const Tensor& log_decay,
    const Tensor& state,
    const std::optional<Tensor>& state_output,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t qk_head_repeat,
    float qk_norm_epsilon) {
    const auto& q_shape = query.logical_shape();
    TT_FATAL(q_shape.rank() == 4, "{}: query must be rank 4", op_name);
    const uint32_t batch = q_shape[0];
    // Value heads define the recurrence; query/key may carry fewer heads that
    // the reader repeats ``qk_head_repeat`` times.
    const uint32_t heads = value.logical_shape()[1];
    const uint32_t key_dim = q_shape[3];
    const uint32_t value_dim = value.logical_shape()[-1];
    auto outputs = ttnn::device_operation::launch<RecurrentGatedDeltaRuleOperation>(
        RecurrentGatedDeltaRuleParams{
            .batch = batch,
            .num_heads = heads,
            .key_dim = key_dim,
            .value_dim = value_dim,
            .qk_head_repeat = qk_head_repeat,
            .qk_norm_epsilon = qk_norm_epsilon,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        RecurrentGatedDeltaRuleInputs{
            .query = query,
            .key = key,
            .value = value,
            .beta = beta,
            .log_decay = log_decay,
            .state = state,
            .state_output = state_output});
    return {outputs[0], outputs[1]};
}

}  // namespace ttnn::experimental::prim

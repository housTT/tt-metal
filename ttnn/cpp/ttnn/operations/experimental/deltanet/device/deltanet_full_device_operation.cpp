// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_full_device_operation.hpp"

#include <array>

#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::deltanet {

DeltaNetDecodeFullDeviceOperation::program_factory_t DeltaNetDecodeFullDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*inputs*/) {
    return DeltaNetDecodeFullProgramFactory{};
}

void DeltaNetDecodeFullDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    TT_FATAL(
        inputs.recurrent_state.storage_type() == StorageType::DEVICE,
        "DeltaNet decode full: recurrent_state must be on device");
    TT_FATAL(inputs.q.storage_type() == StorageType::DEVICE, "DeltaNet decode full: q must be on device");
    TT_FATAL(inputs.k.storage_type() == StorageType::DEVICE, "DeltaNet decode full: k must be on device");
    TT_FATAL(inputs.v.storage_type() == StorageType::DEVICE, "DeltaNet decode full: v must be on device");
    TT_FATAL(inputs.beta.storage_type() == StorageType::DEVICE, "DeltaNet decode full: beta must be on device");
    TT_FATAL(inputs.decay.storage_type() == StorageType::DEVICE, "DeltaNet decode full: decay must be on device");
    if (attrs.preprocess_ab) {
        TT_FATAL(
            inputs.decay_scale.storage_type() == StorageType::DEVICE &&
                inputs.dt_bias.storage_type() == StorageType::DEVICE,
            "DeltaNet decode full: decay_scale and dt_bias must be on device");
    }
    TT_FATAL(inputs.q.layout() == Layout::TILE, "DeltaNet decode full: q must be TILE layout");
    TT_FATAL(inputs.k.layout() == Layout::TILE, "DeltaNet decode full: k must be TILE layout");
    TT_FATAL(inputs.v.layout() == Layout::TILE, "DeltaNet decode full: v must be TILE layout");
    TT_FATAL(inputs.beta.layout() == Layout::TILE, "DeltaNet decode full: beta must be TILE layout");
    TT_FATAL(inputs.decay.layout() == Layout::TILE, "DeltaNet decode full: decay must be TILE layout");
    if (attrs.preprocess_ab) {
        TT_FATAL(
            inputs.decay_scale.layout() == Layout::TILE && inputs.dt_bias.layout() == Layout::TILE,
            "DeltaNet decode full: decay_scale and dt_bias must be TILE layout");
    }
    TT_FATAL(
        inputs.recurrent_state.layout() == Layout::TILE, "DeltaNet decode full: recurrent_state must be TILE layout");
    TT_FATAL(
        inputs.q.dtype() == DataType::BFLOAT16 && inputs.k.dtype() == DataType::BFLOAT16 &&
            inputs.v.dtype() == DataType::BFLOAT16 && inputs.beta.dtype() == DataType::BFLOAT16 &&
            inputs.decay.dtype() == DataType::BFLOAT16 && inputs.recurrent_state.dtype() == DataType::BFLOAT16,
        "DeltaNet decode full currently requires BFLOAT16 inputs");
    if (attrs.preprocess_ab) {
        TT_FATAL(
            inputs.decay_scale.dtype() == DataType::BFLOAT16 && inputs.dt_bias.dtype() == DataType::BFLOAT16,
            "DeltaNet decode full: decay_scale and dt_bias must be BFLOAT16");
    }
    TT_FATAL(
        attrs.k_head_dim % 32 == 0 && attrs.v_head_dim % 32 == 0,
        "DeltaNet decode full: head dims must be multiples of 32");
    TT_FATAL(
        attrs.num_heads == attrs.num_k_heads * attrs.head_expand_ratio,
        "DeltaNet decode full: num_heads must equal num_k_heads * head_expand_ratio");
    TT_FATAL(inputs.q.logical_shape().rank() == 3, "DeltaNet decode full: q must be rank-3");
    const uint32_t batch_size =
        attrs.packed_qkv ? inputs.q.logical_shape()[-2] : inputs.q.logical_shape()[-3];
    TT_FATAL(
        batch_size > 0 && batch_size <= 32 && attrs.num_heads % batch_size == 0 &&
            attrs.num_k_heads % batch_size == 0,
        "DeltaNet decode full: flattened head counts must be divisible by the batch size");
    const uint32_t heads_per_batch = attrs.num_heads / batch_size;
    const uint32_t k_heads_per_batch = attrs.num_k_heads / batch_size;
    if (attrs.packed_qkv) {
        const uint32_t packed_width =
            2 * k_heads_per_batch * attrs.k_head_dim + heads_per_batch * attrs.v_head_dim;
        TT_FATAL(
            inputs.q.logical_shape()[0] == 1 && inputs.q.logical_shape()[-1] == packed_width,
            "DeltaNet decode full: packed q must have shape [1,B,2*Hk*Dk+H*Dv]");
    } else {
        TT_FATAL(
            inputs.k.logical_shape() == inputs.q.logical_shape() && inputs.v.logical_shape().rank() == 3,
            "DeltaNet decode full: q, k, and v must be rank-3 and q/k shapes must match");
        TT_FATAL(
            inputs.q.logical_shape()[-2] == k_heads_per_batch && inputs.q.logical_shape()[-1] == attrs.k_head_dim,
            "DeltaNet decode full: q/k shape does not match the supplied key-head dimensions");
        TT_FATAL(
            inputs.v.logical_shape()[-3] == batch_size && inputs.v.logical_shape()[-2] == heads_per_batch &&
                inputs.v.logical_shape()[-1] == attrs.v_head_dim,
            "DeltaNet decode full: v shape does not match the supplied value-head dimensions");
    }
    TT_FATAL(
        heads_per_batch <= 32 && k_heads_per_batch <= 32,
        "DeltaNet decode full currently supports at most 32 value and key heads per batch item");
    TT_FATAL(
        inputs.beta.logical_shape().rank() == 3 && inputs.beta.logical_shape()[-2] == batch_size &&
            inputs.beta.logical_shape()[-1] == heads_per_batch,
        "DeltaNet decode full: beta shape does not match batch and value heads");
    TT_FATAL(
        inputs.decay.logical_shape() == inputs.beta.logical_shape(),
        "DeltaNet decode full: decay shape must match beta");
    if (attrs.preprocess_ab) {
        TT_FATAL(
            inputs.decay_scale.logical_shape().rank() == 3 && inputs.decay_scale.logical_shape()[-2] == 1 &&
                inputs.decay_scale.logical_shape()[-1] == heads_per_batch &&
                inputs.dt_bias.logical_shape() == inputs.decay_scale.logical_shape(),
            "DeltaNet decode full: decay_scale/dt_bias must have shape [1,1,heads_per_batch]");
    }
    TT_FATAL(
        inputs.recurrent_state.logical_shape().rank() == 4 &&
            inputs.recurrent_state.logical_shape()[-4] == batch_size &&
            inputs.recurrent_state.logical_shape()[-3] == heads_per_batch &&
            inputs.recurrent_state.logical_shape()[-2] == attrs.k_head_dim &&
            inputs.recurrent_state.logical_shape()[-1] == attrs.v_head_dim,
        "DeltaNet decode full: recurrent_state shape does not match the supplied head dimensions");
}

void DeltaNetDecodeFullDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    validate_on_program_cache_miss(attrs, inputs);
}

DeltaNetDecodeFullDeviceOperation::spec_return_value_t DeltaNetDecodeFullDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    // output: [1, 1, 1, num_heads * v_dim] — flat raw q @ S_new output
    auto output_shape = Shape({1, 1, 1, attrs.num_heads * attrs.v_head_dim});
    auto output_spec =
        TensorSpec(output_shape, TensorLayout(inputs.q.dtype(), Layout::TILE, attrs.output_memory_config));

    // Keep the large persistent state where the caller placed it; raw output can stay in L1.
    auto state_spec = TensorSpec(
        inputs.recurrent_state.logical_shape(),
        TensorLayout(inputs.recurrent_state.dtype(), Layout::TILE, inputs.recurrent_state.memory_config()));

    return {output_spec, state_spec};
}

DeltaNetDecodeFullDeviceOperation::tensor_return_value_t DeltaNetDecodeFullDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    auto* device = inputs.recurrent_state.device();
    auto output_specs = compute_output_specs(attrs, inputs);
    return {
        create_device_tensor(output_specs[0], device),
        create_device_tensor(output_specs[1], device),
    };
}

DeltaNetConv1dDecodeDeviceOperation::program_factory_t DeltaNetConv1dDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*inputs*/) {
    return DeltaNetConv1dDecodeProgramFactory{};
}

void DeltaNetConv1dDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    const std::array<const Tensor*, 9> tensors = {
        &inputs.input,
        &inputs.state0,
        &inputs.state1,
        &inputs.state2,
        &inputs.state3,
        &inputs.tap0,
        &inputs.tap1,
        &inputs.tap2,
        &inputs.tap3,
    };
    for (const auto* tensor : tensors) {
        TT_FATAL(
            tensor->storage_type() == StorageType::DEVICE && tensor->buffer() != nullptr,
            "DeltaNet conv1d decode: all inputs must be allocated device tensors");
        TT_FATAL(tensor->layout() == Layout::TILE, "DeltaNet conv1d decode: all inputs must use TILE layout");
        TT_FATAL(tensor->dtype() == DataType::BFLOAT16, "DeltaNet conv1d decode: all inputs must be BFLOAT16");
        TT_FATAL(!tensor->is_sharded(), "DeltaNet conv1d decode: sharded inputs are not supported");
        TT_FATAL(
            tensor->device() == inputs.input.device(), "DeltaNet conv1d decode: all inputs must be on one device");
    }
    TT_FATAL(
        !attrs.output_memory_config.is_sharded(), "DeltaNet conv1d decode: sharded output is not supported");
    TT_FATAL(
        attrs.q_width > 0 && attrs.k_width > 0 && attrs.v_width > 0 && attrs.q_width % 32 == 0 &&
            attrs.k_width % 32 == 0 && attrs.v_width % 32 == 0,
        "DeltaNet conv1d decode: Q/K/V widths must be positive and tile aligned");
    const uint32_t channels = attrs.q_width + attrs.k_width + attrs.v_width;
    const auto& input_shape = inputs.input.logical_shape();
    TT_FATAL(
        input_shape.rank() == 3 && input_shape[0] == 1 && input_shape[1] > 0 && input_shape[1] <= 32 &&
            input_shape[2] == channels,
        "DeltaNet conv1d decode: input must have shape [1,B,Q+K+V] with 1 <= B <= 32");
    const auto& state_shape = inputs.state0.logical_shape();
    TT_FATAL(
        state_shape.rank() == 3 && state_shape[0] == 1 && state_shape[1] >= input_shape[1] &&
            state_shape[1] <= 32 && state_shape[2] == channels,
        "DeltaNet conv1d decode: state must have shape [1,Bmax,Q+K+V] with B <= Bmax <= 32");
    TT_FATAL(
        inputs.state1.logical_shape() == state_shape && inputs.state2.logical_shape() == state_shape &&
            inputs.state3.logical_shape() == state_shape,
        "DeltaNet conv1d decode: all state shapes must match");
    for (const auto* tap : {&inputs.tap0, &inputs.tap1, &inputs.tap2, &inputs.tap3}) {
        TT_FATAL(
            tap->logical_shape().rank() == 3 && tap->logical_shape()[0] == 1 && tap->logical_shape()[1] == 1 &&
                tap->logical_shape()[2] == channels,
            "DeltaNet conv1d decode: taps must have shape [1,1,Q+K+V]");
    }
}

void DeltaNetConv1dDecodeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    validate_on_program_cache_miss(attrs, inputs);
}

DeltaNetConv1dDecodeDeviceOperation::spec_return_value_t DeltaNetConv1dDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    return {TensorSpec(
        inputs.input.logical_shape(), TensorLayout(inputs.input.dtype(), Layout::TILE, attrs.output_memory_config))};
}

DeltaNetConv1dDecodeDeviceOperation::tensor_return_value_t
DeltaNetConv1dDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    auto specs = compute_output_specs(attrs, inputs);
    return {create_device_tensor(specs[0], inputs.input.device())};
}

}  // namespace ttnn::operations::experimental::deltanet

namespace ttnn::prim {

std::vector<Tensor> deltanet_decode_full(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& beta,
    const Tensor& decay,
    const Tensor& recurrent_state,
    uint32_t num_heads,
    uint32_t num_k_heads,
    uint32_t k_head_dim,
    uint32_t v_head_dim,
    uint32_t head_expand_ratio,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<const Tensor>& decay_scale,
    const std::optional<const Tensor>& dt_bias,
    bool packed_qkv) {
    using Op = ttnn::operations::experimental::deltanet::DeltaNetDecodeFullDeviceOperation;

    auto mem_config = output_memory_config.value_or(q.memory_config());
    const bool preprocess_ab = decay_scale.has_value();
    TT_FATAL(
        preprocess_ab == dt_bias.has_value(),
        "DeltaNet decode full: decay_scale and dt_bias must be provided together");

    auto operation_attributes = Op::operation_attributes_t{
        .num_heads = num_heads,
        .num_k_heads = num_k_heads,
        .k_head_dim = k_head_dim,
        .v_head_dim = v_head_dim,
        .head_expand_ratio = head_expand_ratio,
        .preprocess_ab = preprocess_ab,
        .packed_qkv = packed_qkv,
        .output_memory_config = mem_config,
    };

    auto tensor_args = Op::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .beta = beta,
        .decay = decay,
        .decay_scale = decay_scale.value_or(q),
        .dt_bias = dt_bias.value_or(q),
        .recurrent_state = recurrent_state,
    };

    return ttnn::device_operation::launch<Op>(operation_attributes, tensor_args);
}

std::vector<Tensor> deltanet_conv1d_decode(
    const Tensor& input,
    const Tensor& state0,
    const Tensor& state1,
    const Tensor& state2,
    const Tensor& state3,
    const Tensor& tap0,
    const Tensor& tap1,
    const Tensor& tap2,
    const Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const std::optional<MemoryConfig>& output_memory_config) {
    using Op = ttnn::operations::experimental::deltanet::DeltaNetConv1dDecodeDeviceOperation;
    auto attrs = Op::operation_attributes_t{
        .q_width = q_width,
        .k_width = k_width,
        .v_width = v_width,
        .output_memory_config = output_memory_config.value_or(input.memory_config()),
    };
    auto inputs = Op::tensor_args_t{
        .input = input,
        .state0 = state0,
        .state1 = state1,
        .state2 = state2,
        .state3 = state3,
        .tap0 = tap0,
        .tap1 = tap1,
        .tap2 = tap2,
        .tap3 = tap3,
    };
    return ttnn::device_operation::launch<Op>(attrs, inputs);
}

}  // namespace ttnn::prim

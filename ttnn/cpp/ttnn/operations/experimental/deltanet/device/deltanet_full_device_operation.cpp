// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_full_device_operation.hpp"

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
    TT_FATAL(inputs.q.layout() == Layout::TILE, "DeltaNet decode full: q must be TILE layout");
    TT_FATAL(inputs.k.layout() == Layout::TILE, "DeltaNet decode full: k must be TILE layout");
    TT_FATAL(inputs.v.layout() == Layout::TILE, "DeltaNet decode full: v must be TILE layout");
    TT_FATAL(inputs.beta.layout() == Layout::TILE, "DeltaNet decode full: beta must be TILE layout");
    TT_FATAL(inputs.decay.layout() == Layout::TILE, "DeltaNet decode full: decay must be TILE layout");
    TT_FATAL(
        inputs.recurrent_state.layout() == Layout::TILE, "DeltaNet decode full: recurrent_state must be TILE layout");
    TT_FATAL(
        inputs.q.dtype() == DataType::BFLOAT16 && inputs.k.dtype() == DataType::BFLOAT16 &&
            inputs.v.dtype() == DataType::BFLOAT16 && inputs.beta.dtype() == DataType::BFLOAT16 &&
            inputs.decay.dtype() == DataType::BFLOAT16 && inputs.recurrent_state.dtype() == DataType::BFLOAT16,
        "DeltaNet decode full currently requires BFLOAT16 inputs");
    TT_FATAL(
        attrs.k_head_dim % 32 == 0 && attrs.v_head_dim % 32 == 0,
        "DeltaNet decode full: head dims must be multiples of 32");
    TT_FATAL(
        attrs.num_heads == attrs.num_k_heads * attrs.head_expand_ratio,
        "DeltaNet decode full: num_heads must equal num_k_heads * head_expand_ratio");
    TT_FATAL(
        inputs.q.logical_shape().rank() == 3 && inputs.k.logical_shape() == inputs.q.logical_shape() &&
            inputs.v.logical_shape().rank() == 3,
        "DeltaNet decode full: q, k, and v must be rank-3 and q/k shapes must match");
    const uint32_t batch_size = inputs.q.logical_shape()[-3];
    TT_FATAL(
        batch_size > 0 && attrs.num_heads % batch_size == 0 && attrs.num_k_heads % batch_size == 0,
        "DeltaNet decode full: flattened head counts must be divisible by the batch size");
    const uint32_t heads_per_batch = attrs.num_heads / batch_size;
    const uint32_t k_heads_per_batch = attrs.num_k_heads / batch_size;
    TT_FATAL(
        inputs.q.logical_shape()[-2] == k_heads_per_batch && inputs.q.logical_shape()[-1] == attrs.k_head_dim,
        "DeltaNet decode full: q/k shape does not match the supplied key-head dimensions");
    TT_FATAL(
        inputs.v.logical_shape()[-3] == batch_size && inputs.v.logical_shape()[-2] == heads_per_batch &&
            inputs.v.logical_shape()[-1] == attrs.v_head_dim,
        "DeltaNet decode full: v shape does not match the supplied value-head dimensions");
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
    const std::optional<MemoryConfig>& output_memory_config) {
    using Op = ttnn::operations::experimental::deltanet::DeltaNetDecodeFullDeviceOperation;

    auto mem_config = output_memory_config.value_or(q.memory_config());

    auto operation_attributes = Op::operation_attributes_t{
        .num_heads = num_heads,
        .num_k_heads = num_k_heads,
        .k_head_dim = k_head_dim,
        .v_head_dim = v_head_dim,
        .head_expand_ratio = head_expand_ratio,
        .output_memory_config = mem_config,
    };

    auto tensor_args = Op::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .beta = beta,
        .decay = decay,
        .recurrent_state = recurrent_state,
    };

    return ttnn::device_operation::launch<Op>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "deltanet_full_program_factory.hpp"
#include "deltanet_full_device_operation_types.hpp"

namespace ttnn::operations::experimental::deltanet {

struct DeltaNetDecodeFullDeviceOperation {
    using operation_attributes_t = DeltaNetDecodeFullParams;
    using tensor_args_t = DeltaNetDecodeFullInputs;

    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    using program_factory_t = std::variant<DeltaNetDecodeFullProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

struct DeltaNetConv1dDecodeDeviceOperation {
    using operation_attributes_t = DeltaNetConv1dDecodeParams;
    using tensor_args_t = DeltaNetConv1dDecodeInputs;

    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    using program_factory_t = std::variant<DeltaNetConv1dDecodeProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

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
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<const Tensor>& decay_scale = std::nullopt,
    const std::optional<const Tensor>& dt_bias = std::nullopt,
    bool packed_qkv = false);

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
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace ttnn::prim

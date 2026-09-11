// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gdn_decode_step_device_operation_types.hpp"
#include "gdn_decode_step_program_factory.hpp"

namespace ttnn::experimental::prim {

struct GdnDecodeStepOperation {
    using operation_attributes_t = GdnDecodeStepParams;
    using tensor_args_t = GdnDecodeStepInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<GdnDecodeStepProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

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
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::experimental::prim

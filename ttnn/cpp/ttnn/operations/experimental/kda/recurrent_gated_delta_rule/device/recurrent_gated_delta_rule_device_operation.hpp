// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "recurrent_gated_delta_rule_device_operation_types.hpp"
#include "recurrent_gated_delta_rule_program_factory.hpp"

namespace ttnn::experimental::prim {

struct RecurrentGatedDeltaRuleOperation {
    using operation_attributes_t = RecurrentGatedDeltaRuleParams;
    using tensor_args_t = RecurrentGatedDeltaRuleInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<RecurrentGatedDeltaRuleProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

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
    uint32_t qk_head_repeat = 1,
    float qk_norm_epsilon = 0.0f);

}  // namespace ttnn::experimental::prim

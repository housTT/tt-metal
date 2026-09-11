// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_weighted_sum_device_operation_types.hpp"
#include "moe_weighted_sum_program_factory.hpp"

namespace ttnn::experimental::prim {

struct MoeWeightedSumOperation {
    using operation_attributes_t = MoeWeightedSumParams;
    using tensor_args_t = MoeWeightedSumInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MoeWeightedSumProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor moe_weighted_sum(
    const Tensor& groups,
    const Tensor& scores,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::experimental::prim

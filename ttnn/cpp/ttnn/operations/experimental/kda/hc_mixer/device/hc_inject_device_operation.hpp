// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hc_inject_device_operation_types.hpp"
#include "hc_inject_program_factory.hpp"

namespace ttnn::experimental::prim {

struct HcInjectOperation {
    using operation_attributes_t = HcInjectParams;
    using tensor_args_t = HcInjectInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<HcInjectProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor hc_inject(
    const Tensor& hyper,
    const Tensor& block,
    const Tensor& injection,
    uint32_t streams,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::experimental::prim

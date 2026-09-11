// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_sort_slabs_device_operation_types.hpp"
#include "moe_sort_slabs_program_factory.hpp"

namespace ttnn::experimental::prim {

struct MoeSortSlabsOperation {
    using operation_attributes_t = MoeSortSlabsParams;
    using tensor_args_t = MoeSortSlabsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<MoeSortSlabsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> moe_sort_slabs(
    const Tensor& indices,
    const Tensor& scores,
    const Tensor& rank_base,
    uint32_t local_experts,
    uint32_t slab_capacity,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::experimental::prim

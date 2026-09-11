// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_route_topk_device_operation_types.hpp"
#include "moe_route_topk_program_factory.hpp"

namespace ttnn::experimental::prim {

struct MoeRouteTopkOperation {
    using operation_attributes_t = MoeRouteTopkParams;
    using tensor_args_t = MoeRouteTopkInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<MoeRouteTopkProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::tuple<Tensor, Tensor> moe_route_topk(
    const Tensor& logits,
    const Tensor& rank_base,
    uint32_t k,
    uint32_t local_experts,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::experimental::prim

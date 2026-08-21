// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "topk_local_dispatch_program_factory.hpp"
#include "topk_local_dispatch_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe {

struct TopkLocalDispatchDeviceOperation {
    using operation_attributes_t = TopkLocalDispatchParams;
    using tensor_args_t = TopkLocalDispatchInputs;
    using spec_return_value_t = TopkLocalDispatchSpecs;
    using tensor_return_value_t = TopkLocalDispatchTensors;
    // device_operation::launch recognizes custom output topologies only through
    // this exact vector return type (see ttnn/api/ttnn/device_operation.hpp).
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using program_factory_t = std::variant<TopkLocalDispatchProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe

namespace ttnn::prim {

ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe::TopkLocalDispatchTensors topk_local_dispatch(
    const ttnn::Tensor& x,
    const ttnn::Tensor& topk_indices,
    const ttnn::Tensor& global_to_local_expert,
    uint32_t num_local_experts,
    uint32_t valid_tokens,
    bool materialize_x = true);

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "topk_local_combine_program_factory.hpp"
#include "topk_local_combine_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe {

struct TopkLocalCombineDeviceOperation {
    using operation_attributes_t = TopkLocalCombineParams;
    using tensor_args_t = TopkLocalCombineInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    // device_operation::launch recognizes custom output topologies only through
    // this exact vector return type (see ttnn/api/ttnn/device_operation.hpp).
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using program_factory_t = std::variant<TopkLocalCombineProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe

namespace ttnn::prim {

ttnn::Tensor topk_local_combine(
    const ttnn::Tensor& packed_y,
    const ttnn::Tensor& topk_weights,
    const ttnn::Tensor& slot_to_packed_row,
    const ttnn::Tensor& slot_is_local,
    uint32_t tokens,
    uint32_t topk,
    bool assignment_addressed = false,
    const std::optional<ttnn::Tensor>& optional_output = std::nullopt);

}  // namespace ttnn::prim

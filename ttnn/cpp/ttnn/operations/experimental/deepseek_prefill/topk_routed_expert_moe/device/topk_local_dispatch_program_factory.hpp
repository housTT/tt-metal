// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "topk_local_dispatch_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe {

struct TopkLocalDispatchSharedVariables {
    tt::tt_metal::KernelHandle planner_kernel = 0;
    tt::tt_metal::KernelHandle dispatch_kernel = 0;
    std::vector<CoreCoord> dispatch_cores;
    bool materialize_x = true;
};

struct TopkLocalDispatchProgramFactory {
    using shared_variables_t = TopkLocalDispatchSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const TopkLocalDispatchParams& operation_attributes,
        const TopkLocalDispatchInputs& tensor_args,
        TopkLocalDispatchTensors& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TopkLocalDispatchParams& operation_attributes,
        const TopkLocalDispatchInputs& tensor_args,
        TopkLocalDispatchTensors& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe

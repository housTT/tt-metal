// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "topk_local_combine_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe {

struct TopkLocalCombineSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel = 0;
    tt::tt_metal::KernelHandle writer_kernel = 0;
    std::vector<CoreCoord> cores;
};

struct TopkLocalCombineProgramFactory {
    using shared_variables_t = TopkLocalCombineSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const TopkLocalCombineParams& operation_attributes,
        const TopkLocalCombineInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TopkLocalCombineParams& operation_attributes,
        const TopkLocalCombineInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe

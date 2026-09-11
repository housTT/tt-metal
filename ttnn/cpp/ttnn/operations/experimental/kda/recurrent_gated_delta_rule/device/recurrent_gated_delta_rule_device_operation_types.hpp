// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct RecurrentGatedDeltaRuleParams {
    uint32_t batch;
    uint32_t num_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    // Value heads per key/query head (GQA-style expansion done in the reader).
    uint32_t qk_head_repeat = 1;
    // > 0: L2-normalize query/key inside the kernel (x / sqrt(sum x^2 + eps)),
    // and scale the query by 1/sqrt(key_dim); 0: inputs arrive normalized.
    float qk_norm_epsilon = 0.0f;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct RecurrentGatedDeltaRuleInputs {
    Tensor query;
    Tensor key;
    Tensor value;
    Tensor beta;
    Tensor log_decay;
    Tensor state;
    std::optional<Tensor> state_output;
};

}  // namespace ttnn::experimental::prim

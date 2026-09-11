// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

std::tuple<ttnn::Tensor, ttnn::Tensor> recurrent_gated_delta_rule(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& log_decay,
    const ttnn::Tensor& state,
    const std::optional<ttnn::Tensor>& state_output = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    uint32_t qk_head_repeat = 1,
    float qk_norm_epsilon = 0.0f);

}  // namespace ttnn::experimental::kda

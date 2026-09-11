// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

// One fused gated-delta-net decode step: 4-tap causal conv + SiLU on the
// head's q/k/v channels, in-kernel q/k L2 norm (q scaled by 1/sqrt(K)),
// gated delta-rule recurrence with the state updated in place, and the
// per-head RMSNorm x weight x sigmoid(gate) epilogue laid out time-first.
std::tuple<ttnn::Tensor, ttnn::Tensor> gdn_decode_step(
    const ttnn::Tensor& x,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& conv_w0,
    const ttnn::Tensor& conv_w1,
    const ttnn::Tensor& conv_w2,
    const ttnn::Tensor& conv_w3,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& log_decay,
    const ttnn::Tensor& state,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& norm_weight,
    uint32_t num_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    const std::optional<ttnn::Tensor>& state_output = std::nullopt,
    uint32_t qk_head_repeat = 1,
    float qk_norm_epsilon = 1e-6f,
    float norm_epsilon = 1e-6f,
    const std::optional<ttnn::DataType>& output_dtype = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::kda

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

// Decode tail of the hyper-connection input mixer after the all-reduced
// ``packed = [low | inject]`` row: ``mix_s = sigmoid(silu(low) @ up_s)`` per
// stream, ``mixed = mean_s(weighted_s * mix_s)`` and the pass-through
// ``injection`` gate row.
std::tuple<ttnn::Tensor, ttnn::Tensor> hc_mix_post(
    const ttnn::Tensor& packed,
    const ttnn::Tensor& weighted,
    const ttnn::Tensor& up,
    uint32_t lowrank,
    uint32_t streams,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

// ``out[s, :] = hyper[s, :] + 2 * sigmoid(injection[s]) * block`` for one
// decode row over ``streams`` residual streams.
ttnn::Tensor hc_inject(
    const ttnn::Tensor& hyper,
    const ttnn::Tensor& block,
    const ttnn::Tensor& injection,
    uint32_t streams,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::kda

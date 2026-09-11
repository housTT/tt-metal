// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

// Batch-one MoE routing for one rank: top-k over the router logits row,
// softmax over the k selected logits, and the rank-local bank slot ids
// (ids outside [rank_base, rank_base + local_experts) become slot 0 with
// weight 0).  Returns (slot ids UINT16 row-major [1,1,1,k], local scores
// BFLOAT16 tile [1,1,1,k]).
std::tuple<ttnn::Tensor, ttnn::Tensor> moe_route_topk(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& rank_base,
    uint32_t k,
    uint32_t local_experts,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

// out[0, :] = sum_g scores[g] * groups[g, 0, :] for groups = [1, k, 32, N]:
// the weighted expert sum of the batch-one decode row (other rows are zero).
ttnn::Tensor moe_weighted_sum(
    const ttnn::Tensor& groups,
    const ttnn::Tensor& scores,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

// hidden = silu(gate) * up for gate_up = [1, G, rows, 2*I] (gate | up along the last dim).
ttnn::Tensor moe_swiglu(
    const ttnn::Tensor& gate_up,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

// Prefill routing sort for one rank: groups the (row, k) entries of the rank's
// experts into 32-row slabs (one expert per slab).  Returns [slab_rows UINT32 RM
// [1,1,1,P*32], slab_experts UINT16 RM [1,1,1,P], slab_pos INT32 tile [1,1,rows,k],
// local_scores BFLOAT16 tile [1,1,rows,k]].
std::vector<ttnn::Tensor> moe_sort_slabs(
    const ttnn::Tensor& indices,
    const ttnn::Tensor& scores,
    const ttnn::Tensor& rank_base,
    uint32_t local_experts,
    uint32_t slab_capacity,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental::kda

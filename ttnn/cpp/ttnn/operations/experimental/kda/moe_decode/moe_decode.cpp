// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_decode.hpp"

#include "device/moe_route_topk_device_operation.hpp"
#include "device/moe_swiglu_device_operation.hpp"
#include "device/moe_sort_slabs_device_operation.hpp"
#include "device/moe_weighted_sum_device_operation.hpp"

namespace ttnn::experimental::kda {

std::tuple<ttnn::Tensor, ttnn::Tensor> moe_route_topk(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& rank_base,
    uint32_t k,
    uint32_t local_experts,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    TT_FATAL(
        logits.storage_type() == StorageType::DEVICE && logits.buffer() != nullptr,
        "moe_route_topk: logits must be an allocated device tensor");
    return ttnn::experimental::prim::moe_route_topk(
        logits, rank_base, k, local_experts, memory_config.value_or(logits.memory_config()));
}

ttnn::Tensor moe_weighted_sum(
    const ttnn::Tensor& groups,
    const ttnn::Tensor& scores,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        groups.storage_type() == StorageType::DEVICE && groups.buffer() != nullptr,
        "moe_weighted_sum: groups must be an allocated device tensor");
    const auto kernel_config = init_device_compute_kernel_config(
        groups.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false,
        /*default_dst_full_sync_en=*/false,
        ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE);
    return ttnn::experimental::prim::moe_weighted_sum(
        groups, scores, memory_config.value_or(groups.memory_config()), kernel_config);
}

ttnn::Tensor moe_swiglu(
    const ttnn::Tensor& gate_up,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        gate_up.storage_type() == StorageType::DEVICE && gate_up.buffer() != nullptr,
        "moe_swiglu: gate_up must be an allocated device tensor");
    const auto kernel_config = init_device_compute_kernel_config(
        gate_up.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false,
        /*default_dst_full_sync_en=*/false,
        ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE);
    return ttnn::experimental::prim::moe_swiglu(
        gate_up, memory_config.value_or(gate_up.memory_config()), kernel_config);
}

std::vector<ttnn::Tensor> moe_sort_slabs(
    const ttnn::Tensor& indices,
    const ttnn::Tensor& scores,
    const ttnn::Tensor& rank_base,
    uint32_t local_experts,
    uint32_t slab_capacity,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    TT_FATAL(
        indices.storage_type() == StorageType::DEVICE && indices.buffer() != nullptr,
        "moe_sort_slabs: indices must be an allocated device tensor");
    return ttnn::experimental::prim::moe_sort_slabs(
        indices, scores, rank_base, local_experts, slab_capacity, memory_config.value_or(indices.memory_config()));
}

}  // namespace ttnn::experimental::kda

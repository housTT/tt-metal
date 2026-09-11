// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_gated_delta_rule.hpp"

#include "device/recurrent_gated_delta_rule_device_operation.hpp"

namespace ttnn::experimental::kda {

std::tuple<ttnn::Tensor, ttnn::Tensor> recurrent_gated_delta_rule(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& log_decay,
    const ttnn::Tensor& state,
    const std::optional<ttnn::Tensor>& state_output,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    uint32_t qk_head_repeat,
    float qk_norm_epsilon) {
    TT_FATAL(
        query.storage_type() == StorageType::DEVICE && query.buffer() != nullptr,
        "recurrent_gated_delta_rule: query must be an allocated device tensor");
    const auto output_memory_config =
        memory_config.value_or(state_output.has_value() ? state_output->memory_config() : state.memory_config());
    const auto kernel_config = init_device_compute_kernel_config(
        query.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false,
        /*default_dst_full_sync_en=*/false,
        ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE);
    return ttnn::experimental::prim::recurrent_gated_delta_rule(
        query,
        key,
        value,
        beta,
        log_decay,
        state,
        state_output,
        output_memory_config,
        kernel_config,
        qk_head_repeat,
        qk_norm_epsilon);
}

}  // namespace ttnn::experimental::kda

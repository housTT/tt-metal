// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gdn_decode_step.hpp"

#include "device/gdn_decode_step_device_operation.hpp"

namespace ttnn::experimental::kda {

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
    const std::optional<ttnn::Tensor>& state_output,
    uint32_t qk_head_repeat,
    float qk_norm_epsilon,
    float norm_epsilon,
    const std::optional<ttnn::DataType>& output_dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        x.storage_type() == StorageType::DEVICE && x.buffer() != nullptr,
        "gdn_decode_step: x must be an allocated device tensor");
    const auto output_memory_config = memory_config.value_or(state.memory_config());
    const auto kernel_config = init_device_compute_kernel_config(
        x.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false,
        /*default_dst_full_sync_en=*/false,
        ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE);
    return ttnn::experimental::prim::gdn_decode_step(
        x,
        tap0,
        tap1,
        tap2,
        conv_w0,
        conv_w1,
        conv_w2,
        conv_w3,
        beta,
        log_decay,
        state,
        gate,
        norm_weight,
        num_heads,
        key_dim,
        value_dim,
        state_output,
        qk_head_repeat,
        qk_norm_epsilon,
        norm_epsilon,
        output_dtype.value_or(DataType::FLOAT32),
        output_memory_config,
        kernel_config);
}

}  // namespace ttnn::experimental::kda

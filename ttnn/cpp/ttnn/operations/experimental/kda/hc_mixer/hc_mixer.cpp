// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hc_mixer.hpp"

#include "device/hc_inject_device_operation.hpp"
#include "device/hc_mix_post_device_operation.hpp"

namespace ttnn::experimental::kda {

namespace {
DeviceComputeKernelConfig default_config(
    const ttnn::Tensor& reference, const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return init_device_compute_kernel_config(
        reference.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false,
        /*default_dst_full_sync_en=*/false,
        ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE);
}
}  // namespace

std::tuple<ttnn::Tensor, ttnn::Tensor> hc_mix_post(
    const ttnn::Tensor& packed,
    const ttnn::Tensor& weighted,
    const ttnn::Tensor& up,
    uint32_t lowrank,
    uint32_t streams,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        packed.storage_type() == StorageType::DEVICE && packed.buffer() != nullptr,
        "hc_mix_post: packed must be an allocated device tensor");
    return ttnn::experimental::prim::hc_mix_post(
        packed,
        weighted,
        up,
        lowrank,
        streams,
        memory_config.value_or(weighted.memory_config()),
        default_config(packed, compute_kernel_config));
}

ttnn::Tensor hc_inject(
    const ttnn::Tensor& hyper,
    const ttnn::Tensor& block,
    const ttnn::Tensor& injection,
    uint32_t streams,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        hyper.storage_type() == StorageType::DEVICE && hyper.buffer() != nullptr,
        "hc_inject: hyper must be an allocated device tensor");
    return ttnn::experimental::prim::hc_inject(
        hyper,
        block,
        injection,
        streams,
        memory_config.value_or(hyper.memory_config()),
        default_config(hyper, compute_kernel_config));
}

}  // namespace ttnn::experimental::kda

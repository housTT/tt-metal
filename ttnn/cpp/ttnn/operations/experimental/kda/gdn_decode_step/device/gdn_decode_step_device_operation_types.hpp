// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct GdnDecodeStepParams {
    uint32_t batch;
    uint32_t num_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t qkv_width;
    uint32_t qk_head_repeat;
    float qk_norm_epsilon;
    float norm_epsilon;
    tt::tt_metal::DataType output_dtype;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct GdnDecodeStepInputs {
    Tensor x;
    Tensor tap0;
    Tensor tap1;
    Tensor tap2;
    Tensor conv_w0;
    Tensor conv_w1;
    Tensor conv_w2;
    Tensor conv_w3;
    Tensor beta;
    Tensor log_decay;
    Tensor state;
    Tensor gate;
    Tensor norm_weight;
    std::optional<Tensor> state_output;
};

}  // namespace ttnn::experimental::prim

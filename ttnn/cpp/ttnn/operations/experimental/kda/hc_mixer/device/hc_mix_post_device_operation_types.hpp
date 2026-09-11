// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct HcMixPostParams {
    uint32_t streams;
    uint32_t lowrank;
    uint32_t width;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct HcMixPostInputs {
    Tensor packed;
    Tensor weighted;
    Tensor up;
};

}  // namespace ttnn::experimental::prim

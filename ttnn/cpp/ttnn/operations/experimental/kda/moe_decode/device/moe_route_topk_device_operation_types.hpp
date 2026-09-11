// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MoeRouteTopkParams {
    uint32_t k;
    uint32_t num_experts;
    uint32_t local_experts;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct MoeRouteTopkInputs {
    Tensor logits;
    Tensor rank_base;
};

}  // namespace ttnn::experimental::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MoeSortSlabsParams {
    uint32_t rows;
    uint32_t k;
    uint32_t local_experts;
    uint32_t slab_capacity;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct MoeSortSlabsInputs {
    Tensor indices;
    Tensor scores;
    Tensor rank_base;
};

}  // namespace ttnn::experimental::prim

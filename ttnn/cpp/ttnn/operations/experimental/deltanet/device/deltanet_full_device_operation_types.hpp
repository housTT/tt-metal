// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/base_types.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace ttnn::operations::experimental::deltanet {

struct DeltaNetDecodeFullParams {
    uint32_t num_heads;
    uint32_t num_k_heads;
    uint32_t k_head_dim;
    uint32_t v_head_dim;
    uint32_t head_expand_ratio;
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct DeltaNetDecodeFullInputs {
    const Tensor& qkv_proj;         // [1,1,1, 2*Hk*Dk + H*Dv], normalized q/k and v
    const Tensor& beta;             // [1,1,1,H]
    const Tensor& decay;            // [1,1,1,H]
    const Tensor& recurrent_state;  // [1,H,Dk,Dv]
};

}  // namespace ttnn::operations::experimental::deltanet

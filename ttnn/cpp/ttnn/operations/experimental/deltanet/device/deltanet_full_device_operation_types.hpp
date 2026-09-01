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
    const Tensor& q;                // [B,Hk,Dk], raw q
    const Tensor& k;                // [B,Hk,Dk], raw k
    const Tensor& v;                // [B,H,Dv]
    const Tensor& beta;             // [1,B,H]
    const Tensor& decay;            // [1,B,H]
    const Tensor& recurrent_state;  // [B,H,Dk,Dv]
};

}  // namespace ttnn::operations::experimental::deltanet

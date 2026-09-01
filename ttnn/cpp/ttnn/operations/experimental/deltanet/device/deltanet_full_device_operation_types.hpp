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
    bool preprocess_ab;
    bool fused_epilogue;
    bool packed_qkv;
    float norm_epsilon;
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct DeltaNetDecodeFullInputs {
    const Tensor& q;                // [B,Hk,Dk], raw q
    const Tensor& k;                // [B,Hk,Dk], raw k
    const Tensor& v;                // [B,H,Dv]
    const Tensor& beta;             // [1,B,H]
    const Tensor& decay;            // [1,B,H]
    const Tensor& decay_scale;      // [1,1,H], used when preprocess_ab
    const Tensor& dt_bias;          // [1,1,H], used when preprocess_ab
    const Tensor& gate;             // [1,B,H*Dv], used when fused_epilogue
    const Tensor& norm_weight;      // [1,1,Dv], used when fused_epilogue
    const Tensor& recurrent_state;  // [B,H,Dk,Dv]
};

struct DeltaNetConv1dDecodeParams {
    uint32_t q_width;
    uint32_t k_width;
    uint32_t v_width;
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct DeltaNetConv1dDecodeInputs {
    const Tensor& input;   // [1,B,Q+K+V]
    const Tensor& state0;  // [1,Bmax,Q+K+V], oldest state (updated in place)
    const Tensor& state1;
    const Tensor& state2;
    const Tensor& state3;  // newest state (updated in place with input)
    const Tensor& tap0;    // [1,1,Q+K+V]
    const Tensor& tap1;
    const Tensor& tap2;
    const Tensor& tap3;
};

}  // namespace ttnn::operations::experimental::deltanet

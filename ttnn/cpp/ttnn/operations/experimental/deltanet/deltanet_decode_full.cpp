// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_decode_full.hpp"
#include "device/deltanet_full_device_operation.hpp"

namespace ttnn::experimental {

std::vector<Tensor> deltanet_decode_full(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& beta,
    const Tensor& decay,
    const Tensor& recurrent_state,
    uint32_t num_heads,
    uint32_t num_k_heads,
    uint32_t k_head_dim,
    uint32_t v_head_dim,
    uint32_t head_expand_ratio,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const Tensor>& decay_scale,
    const std::optional<const Tensor>& dt_bias,
    bool packed_qkv) {
    TT_FATAL(
        decay_scale.has_value() == dt_bias.has_value(),
        "DeltaNet decode full: decay_scale and dt_bias must be provided together");
    return ttnn::prim::deltanet_decode_full(
        q,
        k,
        v,
        beta,
        decay,
        recurrent_state,
        num_heads,
        num_k_heads,
        k_head_dim,
        v_head_dim,
        head_expand_ratio,
        memory_config,
        decay_scale,
        dt_bias,
        packed_qkv);
}

Tensor deltanet_conv1d_decode(
    const Tensor& input,
    const Tensor& state0,
    const Tensor& state1,
    const Tensor& state2,
    const Tensor& state3,
    const Tensor& tap0,
    const Tensor& tap1,
    const Tensor& tap2,
    const Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::deltanet_conv1d_decode(
               input,
               state0,
               state1,
               state2,
               state3,
               tap0,
               tap1,
               tap2,
               tap3,
               q_width,
               k_width,
               v_width,
               memory_config)[0];
}

}  // namespace ttnn::experimental

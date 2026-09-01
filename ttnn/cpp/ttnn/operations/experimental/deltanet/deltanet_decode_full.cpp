// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_decode_full.hpp"
#include "device/deltanet_full_device_operation.hpp"

namespace ttnn::experimental {

std::vector<Tensor> deltanet_decode_full(
    const Tensor& qkv_proj,
    const Tensor& beta,
    const Tensor& decay,
    const Tensor& recurrent_state,
    uint32_t num_heads,
    uint32_t num_k_heads,
    uint32_t k_head_dim,
    uint32_t v_head_dim,
    uint32_t head_expand_ratio,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::deltanet_decode_full(
        qkv_proj,
        beta,
        decay,
        recurrent_state,
        num_heads,
        num_k_heads,
        k_head_dim,
        v_head_dim,
        head_expand_ratio,
        memory_config);
}

}  // namespace ttnn::experimental

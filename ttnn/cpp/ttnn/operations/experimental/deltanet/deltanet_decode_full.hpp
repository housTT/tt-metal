// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

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
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const Tensor>& decay_scale = std::nullopt,
    const std::optional<const Tensor>& dt_bias = std::nullopt);

}  // namespace ttnn::experimental

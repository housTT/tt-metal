// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe {

struct TopkLocalCombineParams {
    uint32_t tokens;
    uint32_t topk;
    // True when packed_y is the production token-major [T*K,H] slot buffer.
    // False retains the stage-1 expert-region inverse-map bring-up contract.
    bool assignment_addressed = false;

    static constexpr auto attribute_names = std::forward_as_tuple("tokens", "topk", "assignment_addressed");
    auto attribute_values() const { return std::forward_as_tuple(tokens, topk, assignment_addressed); }
};

struct TopkLocalCombineInputs {
    // [1,1,C,H], ROW_MAJOR BF16 widened/untilized fused-FFN output.
    Tensor packed_y;
    // [1,1,T,K], ROW_MAJOR BF16. One page per token; its K BF16 values
    // remain contiguous at byte offsets slot*sizeof(bfloat16).
    Tensor topk_weights;
    // [1,K*T], ROW_MAJOR UINT32, slot-major packed-row lookup.
    Tensor slot_to_packed_row;
    // [1,K*T], ROW_MAJOR UINT32 exact 0/1. A zero prevents the packed
    // activation row from being read at all, so uninitialized FFN padding
    // can never enter arithmetic (0 * NaN is not relied upon).
    Tensor slot_is_local;
    // Optional caller-owned replacement for the BF8 TILE output. Its address
    // is runtime-patched on cache hits; its spec is validated on both miss and
    // hit so a cached program cannot write through a stale/incompatible tensor.
    std::optional<Tensor> optional_output;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe

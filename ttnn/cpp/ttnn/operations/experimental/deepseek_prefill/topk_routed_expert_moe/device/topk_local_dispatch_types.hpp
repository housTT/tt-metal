// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe {

inline constexpr uint32_t TOPK_LOCAL_DISPATCH_OUTPUTS = 7;

struct TopkLocalDispatchParams {
    // Structural and therefore part of the program-cache key.
    uint32_t num_local_experts;

    // Stage-2 production mode emits only the integer routing plan. The
    // original activation stays in its one-copy ROW_MAJOR buffer and the
    // unified FFN resolves packed assignment ids while reading x. Keeping
    // this in the cache key prevents a planner-only program from aliasing the
    // bring-up program that also launches activation-copy workers.
    bool materialize_x = true;

    // Runtime scalar. Deliberately omitted from attribute_names/attribute_values:
    // cache hits patch it into both planner and dispatch runtime arguments.
    uint32_t valid_tokens;

    static constexpr auto attribute_names = std::forward_as_tuple("num_local_experts", "materialize_x");
    auto attribute_values() const { return std::forward_as_tuple(num_local_experts, materialize_x); }
};

struct TopkLocalDispatchInputs {
    // [1, 1, T, H], ROW_MAJOR BF16. The caller untilizes the original
    // activation exactly once before entering this primitive.
    Tensor x;
    // [1, 1, T, K], TILE UINT32, straight from ttnn.topk.
    Tensor topk_indices;
    // [1, E_global], ROW_MAJOR UINT32. Values 0..E_local-1 select a
    // device-local expert; every value >= E_local is the non-local sentinel.
    Tensor global_to_local_expert;
};

using TopkLocalDispatchSpecs = std::array<tt::tt_metal::TensorSpec, TOPK_LOCAL_DISPATCH_OUTPUTS>;
using TopkLocalDispatchTensors = std::array<Tensor, TOPK_LOCAL_DISPATCH_OUTPUTS>;

// Output order is public and pinned by the nanobind documentation/tests:
//   0 compact_x             [1,1,C,H] ROW_MAJOR BF16 in bring-up mode;
//                           [1,1,1,H] unused sentinel in planner-only mode
//   1 global_counts         [1,E_global] UINT32
//   2 global_region_offsets [1,E_global] UINT32
//   3 local_to_global       [1,E_local] UINT32
//   4 packed_assignment_ids [1,C] UINT32 (token*K + slot; UINT32_MAX in padding)
//   5 slot_to_packed_row    [1,K*T] UINT32, slot-major (slot*T + token)
//   6 slot_is_local         [1,K*T] UINT32, exact 0/1

}  // namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_routed_expert_moe.hpp"

#include "device/topk_local_dispatch_device_operation.hpp"
#include "device/topk_local_combine_device_operation.hpp"

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/unified_routed_expert_ffn.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe {

std::array<ttnn::Tensor, 7> topk_local_dispatch(
    const ttnn::Tensor& x,
    const ttnn::Tensor& topk_indices,
    const ttnn::Tensor& global_to_local_expert,
    uint32_t num_local_experts,
    uint32_t valid_tokens,
    bool materialize_x) {
    return ttnn::prim::topk_local_dispatch(
        x, topk_indices, global_to_local_expert, num_local_experts, valid_tokens, materialize_x);
}

ttnn::Tensor topk_local_combine(
    const ttnn::Tensor& packed_y,
    const ttnn::Tensor& topk_weights,
    const ttnn::Tensor& slot_to_packed_row,
    const ttnn::Tensor& slot_is_local,
    uint32_t tokens,
    uint32_t topk,
    bool assignment_addressed,
    const std::optional<ttnn::Tensor>& optional_output) {
    return ttnn::prim::topk_local_combine(
        packed_y, topk_weights, slot_to_packed_row, slot_is_local, tokens, topk, assignment_addressed, optional_output);
}

ttnn::Tensor topk_routed_expert_moe(
    const ttnn::Tensor& x,
    const ttnn::Tensor& topk_indices,
    const ttnn::Tensor& topk_weights,
    const ttnn::Tensor& global_to_local_expert,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    uint32_t num_local_experts,
    uint32_t valid_tokens,
    uint32_t max_dispatched_tokens_per_expert,
    ttnn::RoutedExpertActivation activation,
    const std::optional<ttnn::Tensor>& optional_output) {
    const uint32_t tokens = x.logical_shape()[-2];
    const uint32_t topk = topk_indices.logical_shape()[-1];
    TT_FATAL(
        topk_weights.logical_shape() == topk_indices.logical_shape(),
        "topk_weights shape ({}) must match topk_indices shape ({})",
        topk_weights.logical_shape(),
        topk_indices.logical_shape());
    TT_FATAL(
        topk_weights.dtype() == tt::tt_metal::DataType::FLOAT32 && topk_weights.layout() == tt::tt_metal::Layout::TILE,
        "topk_weights must be TILE FLOAT32 straight from router softmax, got {}/{}",
        topk_weights.dtype(),
        topk_weights.layout());
    TT_FATAL(
        max_dispatched_tokens_per_expert == tokens,
        "max_dispatched_tokens_per_expert ({}) must equal this native sub-chunk's token count ({})",
        max_dispatched_tokens_per_expert,
        tokens);

    auto x_rm = ttnn::to_layout(x, tt::tt_metal::Layout::ROW_MAJOR, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
    auto dispatch = topk_local_dispatch(
        x_rm,
        topk_indices,
        global_to_local_expert,
        num_local_experts,
        valid_tokens,
        /*materialize_x=*/false);

    // Production stage 2/3: the FFN reader resolves each expert-region row
    // through packed_assignment_ids and gathers x_rm[token] directly. Its
    // writer packs the down result to BF8, widens that rounded value, and
    // scatters disjoint hidden slices into token-major [T*K,H] BF16 slots.
    // No activation-sized compact_x or widened expert-region output exists.
    auto y = unified_routed_expert_ffn::unified_routed_expert_moe(
        x_rm,
        dispatch[2],
        dispatch[1],
        dispatch[3],
        gate_projs,
        up_projs,
        down_projs,
        max_dispatched_tokens_per_expert,
        std::nullopt,
        activation,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        dispatch[4],
        topk);

    // The router produces FP32 TILE scores. Narrow exactly once to the same
    // BF16 score precision as the accepted gathered path. ROW_MAJOR keeps one
    // page per token with all K weights contiguous; combine selects the slot by
    // an in-page byte offset rather than pretending a metadata reshape changed
    // the underlying buffer page geometry.
    auto weights_bf16 = ttnn::typecast(topk_weights, tt::tt_metal::DataType::BFLOAT16);
    auto weights_rm =
        ttnn::to_layout(weights_bf16, tt::tt_metal::Layout::ROW_MAJOR, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);

    return topk_local_combine(
        y,
        weights_rm,
        dispatch[5],
        dispatch[6],
        tokens,
        topk,
        /*assignment_addressed=*/true,
        optional_output);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/unified_routed_expert_ffn.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe {

// Native local planning + activation dispatch for router top-k pairs. This is
// exposed independently so correctness/cache tests can inspect every plan
// tensor and prove the selected path never materializes the post-topk dense
// [T,E] scattered routing matrix. The router's unavoidable [T,E] logits and
// its ordinary sorted top-k selection remain outside this operation.
std::array<ttnn::Tensor, 7> topk_local_dispatch(
    const ttnn::Tensor& x,
    const ttnn::Tensor& topk_indices,
    const ttnn::Tensor& global_to_local_expert,
    uint32_t num_local_experts,
    uint32_t valid_tokens,
    bool materialize_x = true);

// Native reverse permutation + weighted top-k reduction. packed_y is the
// widened/untilized shared FFN buffer; the op reads only rows whose slot mask
// is valid, accumulates K locally and emits the BF8 [1,1,T,H] partial that the
// expert-parallel collective consumes. No [K,T,H] intermediate is allocated.
ttnn::Tensor topk_local_combine(
    const ttnn::Tensor& packed_y,
    const ttnn::Tensor& topk_weights,
    const ttnn::Tensor& slot_to_packed_row,
    const ttnn::Tensor& slot_is_local,
    uint32_t tokens,
    uint32_t topk,
    bool assignment_addressed = false,
    const std::optional<ttnn::Tensor>& optional_output = std::nullopt);

// Production composite. It owns the one-time layout/dtype bridges around the
// two native primitives and the fused expert FFN, so the caller cannot
// accidentally reintroduce dense routing or a K-wide activation tensor.
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
    ttnn::RoutedExpertActivation activation = ttnn::RoutedExpertActivation::Silu,
    const std::optional<ttnn::Tensor>& optional_output = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe

namespace ttnn {
using operations::experimental::deepseek_prefill::topk_routed_expert_moe::topk_local_combine;
using operations::experimental::deepseek_prefill::topk_routed_expert_moe::topk_local_dispatch;
using operations::experimental::deepseek_prefill::topk_routed_expert_moe::topk_routed_expert_moe;
}  // namespace ttnn

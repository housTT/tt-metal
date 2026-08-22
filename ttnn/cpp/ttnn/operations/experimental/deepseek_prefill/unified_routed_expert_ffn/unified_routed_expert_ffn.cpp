// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

#include "device/unified_routed_expert_ffn_device_operation.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/routed_expert_ffn.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

ttnn::Tensor unified_routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t local_expert_id,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& output,
    const std::optional<ttnn::Tensor>& expert_region_offsets,
    const std::optional<uint32_t>& input_m_tiles,
    bool read_x_at_offset,
    bool x_is_row_major,
    RoutedExpertActivation activation,
    const std::optional<ttnn::Tensor>& gate_bias,
    const std::optional<ttnn::Tensor>& up_bias,
    const std::optional<ttnn::Tensor>& down_bias) {
    // Single-op fused per-expert FFN. One device Program runs gate matmul,
    // up matmul, silu, multiply, down matmul as four phases inside the same
    // kernel. The kernels read the local count range device-side at entry and
    // pick one shared runtime chunk_M_tiles/per_core_M from its hottest expert
    // (adaptive_chunk.hpp). Per-expert chunk counts still come from the actual
    // counts; zero-count experts are skipped entirely.
    //
    // The host only sets the CB-sized MAXIMUM chunk (kMaxChunkMTiles => per_core_M
    // 8). The program factory's L1 guard may lower it for large models; the
    // shared device geometry never exceeds whatever max the CBs were sized to.
    constexpr uint32_t kMaxChunkMTiles = 64;  // per_core_M_max = 8 (L1 cap)
    // This expert's M in tiles. Defaults to x's allocated M; a caller passing a
    // shared x buffer (wider than one region) supplies the per-expert value.
    const uint32_t M_tiles_full = input_m_tiles.value_or(x.padded_shape()[-2] / 32);

    return ttnn::prim::unified_routed_expert_ffn(
        x,
        gate_proj,
        up_proj,
        down_proj,
        counts,
        global_expert_idx_table,
        local_expert_id,
        kMaxChunkMTiles,
        M_tiles_full,
        read_x_at_offset,
        x_is_row_major,
        compute_kernel_config.has_value() ? std::optional<ttnn::DeviceComputeKernelConfig>(*compute_kernel_config)
                                          : std::nullopt,
        output,
        expert_region_offsets,
        activation,
        gate_bias,
        up_bias,
        down_bias);
}

ttnn::Tensor unified_routed_expert_moe(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    uint32_t max_dispatched_tokens_per_expert,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    RoutedExpertActivation activation,
    const std::optional<std::vector<ttnn::Tensor>>& gate_biases,
    const std::optional<std::vector<ttnn::Tensor>>& up_biases,
    const std::optional<std::vector<ttnn::Tensor>>& down_biases,
    const std::optional<ttnn::Tensor>& packed_assignment_ids,
    uint32_t topk) {
    TT_FATAL(
        gate_projs.size() == up_projs.size() && gate_projs.size() == down_projs.size(),
        "gate/up/down projection lists must have the same length (got {}, {}, {})",
        gate_projs.size(),
        up_projs.size(),
        down_projs.size());
    TT_FATAL(
        gate_projs.size() <= std::numeric_limits<uint32_t>::max(),
        "local expert count ({}) exceeds the uint32 API limit",
        gate_projs.size());
    const uint32_t experts_per_chip = static_cast<uint32_t>(gate_projs.size());
    TT_FATAL(experts_per_chip > 0, "Need at least one expert per chip");

    // Optional per-expert biases (gpt-oss): all three lists together or none,
    // each the same length as the weight lists (one bias per local expert).
    const int bias_lists = static_cast<int>(gate_biases.has_value()) + static_cast<int>(up_biases.has_value()) +
                           static_cast<int>(down_biases.has_value());
    TT_FATAL(
        bias_lists == 0 || bias_lists == 3,
        "gate/up/down bias lists must all be provided together or all omitted (got {} of 3)",
        bias_lists);
    const bool has_bias = bias_lists == 3;
    const bool assignment_indexed = packed_assignment_ids.has_value();
    TT_FATAL(
        assignment_indexed == (topk > 0),
        "packed_assignment_ids presence ({}) must exactly match a positive topk ({})",
        assignment_indexed,
        topk);
    TT_FATAL(!assignment_indexed || !has_bias, "assignment-indexed MoE does not support projection biases");
    TT_FATAL(!assignment_indexed || topk <= 16, "assignment-indexed topk ({}) exceeds 16", topk);
    if (has_bias) {
        TT_FATAL(
            gate_biases->size() == experts_per_chip && up_biases->size() == experts_per_chip &&
                down_biases->size() == experts_per_chip,
            "bias lists must have one entry per local expert ({}), got ({}, {}, {})",
            experts_per_chip,
            gate_biases->size(),
            up_biases->size(),
            down_biases->size());
    }

    // Bias-free weights run consecutive groups of up to
    // MAX_FUSED_LOCAL_EXPERTS in one device program each. The three kernels
    // iterate each group and switch projection base addresses at expert
    // boundaries, amortizing launch and counts/start-table setup while keeping
    // the public composite valid for larger expert lists. The gpt-oss bias path
    // remains on the proven single-expert program: its bias CBs are resident for
    // the duration of a kernel and intentionally are not multiplexed yet.
    //
    // x is the whole shared buffer, so pass this expert's row count
    // (max_dispatched_tokens_per_expert in tiles) as input_m_tiles — the op sizes
    // its grid/chunks to one expert, not the buffer.
    //
    // The input layout selects the output strategy:
    //   * TILE buffer -> write IN PLACE (output == dispatched_buffer). The reader
    //     reads x in phase 1 and the writer drains cb_out only after compute
    //     consumes it, so a row's write is ordered after its read via the CB
    //     chain; chunks cover disjoint rows and experts touch disjoint regions,
    //     so no expert can disturb another. No allocation, no up-front fill.
    //   * ROW_MAJOR bf16 buffer -> the FFN tilizes x and packs bf8 internally, so
    //     input and output differ in both layout and dtype and cannot alias. One
    //     shared TILE bf8 output is allocated once for all experts; each writes
    //     its own region. Left uninitialized (no fill): downstream `combine`
    //     reads only written rows (bounded per expert to
    //     [offset, offset + ceil_tile(count))).
    const bool x_is_row_major = dispatched_buffer.layout() == tt::tt_metal::Layout::ROW_MAJOR;
    TT_FATAL(!assignment_indexed || x_is_row_major, "assignment-indexed MoE requires ROW_MAJOR BF16 x");
    const auto assignment_output = [&]() {
        const auto shape =
            ttnn::Shape({1, 1, dispatched_buffer.logical_shape()[-2] * topk, dispatched_buffer.logical_shape()[-1]});
        const auto spec = tt::tt_metal::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig{
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
        // Every expert-parallel device owns a different assignment plan and
        // therefore different slot contents. Preserve the planner topology;
        // creating this buffer with the default replicated topology would
        // mislabel distinct per-device data and make combine unsafe.
        return ttnn::create_device_tensor(spec, dispatched_buffer.device(), expert_region_offsets.tensor_topology());
    };
    const ttnn::Tensor output =
        assignment_indexed ? assignment_output()
        : x_is_row_major   ? ttnn::empty(
                               dispatched_buffer.logical_shape(),
                               tt::tt_metal::DataType::BFLOAT8_B,
                               tt::tt_metal::Layout::TILE,
                               dispatched_buffer.device(),
                               tt::tt_metal::MemoryConfig{
                                   tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM})
                         : dispatched_buffer;
    const uint32_t m_tiles = (max_dispatched_tokens_per_expert + 31) / 32;
    if (!has_bias) {
        constexpr uint32_t kMaxChunkMTiles = 64;
        for (uint32_t first_local_expert = 0; first_local_expert < experts_per_chip;
             first_local_expert += MAX_FUSED_LOCAL_EXPERTS) {
            const uint32_t group_size = std::min(MAX_FUSED_LOCAL_EXPERTS, experts_per_chip - first_local_expert);
            const auto first = static_cast<std::ptrdiff_t>(first_local_expert);
            const auto last = static_cast<std::ptrdiff_t>(first_local_expert + group_size);
            const std::vector<ttnn::Tensor> gate_group(gate_projs.begin() + first, gate_projs.begin() + last);
            const std::vector<ttnn::Tensor> up_group(up_projs.begin() + first, up_projs.begin() + last);
            const std::vector<ttnn::Tensor> down_group(down_projs.begin() + first, down_projs.begin() + last);
            ttnn::prim::unified_routed_expert_ffn(
                dispatched_buffer,
                gate_group,
                up_group,
                down_group,
                expert_token_counts,
                global_expert_idx_table,
                first_local_expert,
                kMaxChunkMTiles,
                m_tiles,
                /*read_x_at_offset=*/!assignment_indexed,
                x_is_row_major,
                compute_kernel_config.has_value()
                    ? std::optional<ttnn::DeviceComputeKernelConfig>(*compute_kernel_config)
                    : std::nullopt,
                output,
                expert_region_offsets,
                activation,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                packed_assignment_ids,
                topk);
        }
        return output;
    }

    // Biased fallback: one device program per expert.
    for (uint32_t local_expert = 0; local_expert < experts_per_chip; ++local_expert) {
        unified_routed_expert_ffn(
            dispatched_buffer,
            gate_projs[local_expert],
            up_projs[local_expert],
            down_projs[local_expert],
            expert_token_counts,
            global_expert_idx_table,
            local_expert,
            compute_kernel_config,
            output,
            expert_region_offsets,
            m_tiles,
            /*read_x_at_offset=*/true,
            x_is_row_major,
            activation,
            std::optional<ttnn::Tensor>((*gate_biases)[local_expert]),
            std::optional<ttnn::Tensor>((*up_biases)[local_expert]),
            std::optional<ttnn::Tensor>((*down_biases)[local_expert]));
    }
    return output;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn

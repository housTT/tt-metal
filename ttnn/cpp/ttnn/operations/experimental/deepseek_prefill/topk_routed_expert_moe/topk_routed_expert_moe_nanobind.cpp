// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_routed_expert_moe_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "topk_routed_expert_moe.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_topk_routed_expert_moe(::nanobind::module_& mod) {
    ttnn::bind_function<"topk_local_dispatch", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Build a device-local expert plan directly from UINT32 top-k indices and
        dispatch the selected activation rows into one compact expert-major
        ROW_MAJOR BF16 buffer. No dense ``[tokens, experts]`` tensor is made.

        ``global_to_local_expert`` is device-unique/mesh-sharded: values below
        ``num_local_experts`` are local slots and every other value is exactly
        ``UINT32_MAX``, the non-local sentinel. Replicated mapping tables are
        rejected on a multi-device mesh. ``valid_tokens`` is runtime-patched on program-cache
        hits; it does not multiply cache entries. Each valid token's top-k row
        must contain unique expert ids, as produced by ``ttnn.topk``; malformed
        duplicate rows publish an unusable, all-zero plan rather than overflowing
        an expert's bounded workspace.

        Returns ``(compact_x, global_counts, global_region_offsets,
        local_to_global, packed_assignment_ids, slot_to_packed_row,
        slot_is_local)``. The final two vectors are slot-major
        (``slot * tokens + token``), suitable for native combine.
        )doc",
        &topk_routed_expert_moe::topk_local_dispatch,
        nb::arg("x").noconvert(),
        nb::arg("topk_indices").noconvert(),
        nb::arg("global_to_local_expert").noconvert(),
        nb::arg("num_local_experts"),
        nb::kw_only(),
        nb::arg("valid_tokens"),
        nb::arg("materialize_x") = true);

    ttnn::bind_function<"topk_local_combine", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Reverse the native local top-k dispatch without constructing a dense
        routing tensor or a ``[topk, tokens, hidden]`` activation tensor.

        ``packed_y`` is ``[1,1,C,H]`` ROW_MAJOR BF16. ``topk_weights`` is
        ``[1,1,tokens,topk]`` ROW_MAJOR BF16 (one page per token). The two UINT32 slot maps are
        ``[1,topk*tokens]`` and slot-major. Non-local slots are rejected before
        their packed row is read, then the op applies the preserved routing
        weights, reduces top-k and packs the result directly to
        ``[1,1,tokens,H]`` TILE BFLOAT8_B for the expert-parallel collective.
        )doc",
        &topk_routed_expert_moe::topk_local_combine,
        nb::arg("packed_y").noconvert(),
        nb::arg("topk_weights").noconvert(),
        nb::arg("slot_to_packed_row").noconvert(),
        nb::arg("slot_is_local").noconvert(),
        nb::kw_only(),
        nb::arg("tokens"),
        nb::arg("topk"),
        nb::arg("assignment_addressed") = false,
        nb::arg("output") = nb::none());

    ttnn::bind_function<"topk_routed_expert_moe", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Router-index-native local MoE composite. It consumes the preserved
        FLOAT32 top-k weights and UINT32 global indices directly, dispatches
        only this device's selected rows, executes all local fused experts,
        then combines the local slots into a BFLOAT8_B partial output.

        The composite never constructs the post-topk dense
        ``[tokens, experts]`` scattered routing matrix,
        never allocates ``[topk, tokens, hidden]``, and does not use sort,
        embedding, where, or a standalone reduce after top-k. Router logits and
        the router's normal ``topk(sorted=True)`` are intentionally preserved.
        Consequently, every valid token's index row must contain unique expert
        ids; malformed duplicate rows fail closed in the device planner.
        The admitted primitive span
        is at most 2048 tokens; callers sub-chunk a larger flattened prefill and
        preserve token order when concatenating its returned parts.
        )doc",
        &topk_routed_expert_moe::topk_routed_expert_moe,
        nb::arg("x").noconvert(),
        nb::arg("topk_indices").noconvert(),
        nb::arg("topk_weights").noconvert(),
        nb::arg("global_to_local_expert").noconvert(),
        nb::arg("gate_projs").noconvert(),
        nb::arg("up_projs").noconvert(),
        nb::arg("down_projs").noconvert(),
        nb::kw_only(),
        nb::arg("num_local_experts"),
        nb::arg("valid_tokens"),
        nb::arg("max_dispatched_tokens_per_expert"),
        nb::arg("activation") = ttnn::RoutedExpertActivation::Silu,
        nb::arg("output") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail

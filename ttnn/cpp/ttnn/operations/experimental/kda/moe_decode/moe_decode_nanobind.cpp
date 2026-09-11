// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_decode_nanobind.hpp"

#include "moe_decode.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::moe_decode::detail {

void bind_moe_decode(nb::module_& mod) {
    ttnn::bind_function<"moe_route_topk", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Batch-one MoE routing on one rank.  ``logits = [1,1,R,E]`` BFLOAT16
        tile layout (row 0 is the token), ``rank_base`` an INT32 tile tensor
        whose first element is this rank's first expert id.  Returns the
        rank-local bank slot ids (UINT16 row-major ``[1,1,1,k]``, descending
        logit order, non-local ids as slot 0) and the softmax scores of the k
        selected logits masked to the local ids (BFLOAT16 tile ``[1,1,1,k]``).
        )doc",
        &ttnn::experimental::kda::moe_route_topk,
        nb::arg("logits").noconvert(),
        nb::arg("rank_base").noconvert(),
        nb::arg("k"),
        nb::arg("local_experts"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
    ttnn::bind_function<"moe_weighted_sum", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        ``out[0, :] = sum_g scores[0, g] * groups[0, g, 0, :]`` for
        ``groups = [1,k,32,N]`` and ``scores = [1,1,1,k]`` (BFLOAT16 tiles);
        the output is ``[1,1,32,N]`` BFLOAT16 with rows 1..31 zero.
        )doc",
        &ttnn::experimental::kda::moe_weighted_sum,
        nb::arg("groups").noconvert(),
        nb::arg("scores").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
    ttnn::bind_function<"moe_swiglu", "ttnn.experimental.kda.">(
        mod,
        R"doc(``silu(gate) * up`` for ``gate_up = [1,G,rows,2*I]`` BFLOAT16 tiles (gate | up along the last dim); returns ``[1,G,rows,I]``.)doc",
        &ttnn::experimental::kda::moe_swiglu,
        nb::arg("gate_up").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
    ttnn::bind_function<"moe_sort_slabs", "ttnn.experimental.kda.">(
        mod,
        R"doc(Prefill routing sort for one rank: groups the (row, k) top-k entries that hit this rank's experts into 32-row slabs, one expert per slab (``slab_capacity`` >= local_experts + ceil(rows*k/32) + 1).  Returns ``[slab_rows, slab_experts, slab_pos, local_scores]``; non-local entries get weight 0 and the dummy position ``P*32-1``.)doc",
        &ttnn::experimental::kda::moe_sort_slabs,
        nb::arg("indices").noconvert(),
        nb::arg("scores").noconvert(),
        nb::arg("rank_base").noconvert(),
        nb::arg("local_experts"),
        nb::arg("slab_capacity"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::moe_decode::detail

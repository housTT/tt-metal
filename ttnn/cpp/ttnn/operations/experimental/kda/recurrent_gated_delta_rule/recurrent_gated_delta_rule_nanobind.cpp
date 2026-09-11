// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_gated_delta_rule_nanobind.hpp"
#include "recurrent_gated_delta_rule.hpp"

#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::recurrent_gated_delta_rule::detail {

void bind_recurrent_gated_delta_rule(nb::module_& mod) {
    ttnn::bind_function<"recurrent_gated_delta_rule", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Execute one normalized recurrent gated-delta-rule step.

        ``query`` and ``key`` are already L2-normalized (and query-scaled).
        The operation computes ``S=exp(g)*S``, ``d=v-k@S``,
        ``S=S+beta*(k.T@d)``, and ``o=q@S`` in one device program.

        Inputs are interleaved FLOAT32 TILE tensors with shapes
        ``q,k=[B,H,1,K]``, ``v=[B,H,1,V]``, ``beta,g=[B,H,1,1]``, and
        ``state=[B,H,K,V]``. ``K`` and ``V`` must be tile aligned.

        ``state_output`` may alias ``state`` to update a persistent recurrent
        state buffer in place and avoid a separate device copy.
        ``qk_head_repeat`` lets ``q``/``k`` carry ``H / qk_head_repeat`` heads
        (``[B,H/r,1,K]``); the reader reuses key head ``h / r`` for value head
        ``h`` (grouped-query expansion without a repeat_interleave op).
        ``qk_norm_epsilon > 0`` L2-normalizes ``q`` and ``k`` inside the kernel
        (``x / sqrt(sum(x^2) + eps)``) and scales ``q`` by ``1/sqrt(K)``, so the
        raw projections can be passed directly.
        )doc",
        &ttnn::experimental::kda::recurrent_gated_delta_rule,
        nb::arg("query").noconvert(),
        nb::arg("key").noconvert(),
        nb::arg("value").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("log_decay").noconvert(),
        nb::arg("state").noconvert(),
        nb::kw_only(),
        nb::arg("state_output").noconvert() = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("qk_head_repeat") = 1,
        nb::arg("qk_norm_epsilon") = 0.0f);
}

}  // namespace ttnn::operations::experimental::kda::recurrent_gated_delta_rule::detail

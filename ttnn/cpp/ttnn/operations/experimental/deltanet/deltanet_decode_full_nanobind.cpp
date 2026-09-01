// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_decode_full_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deltanet/deltanet_decode_full.hpp"

namespace ttnn::operations::experimental::deltanet::detail {

void bind_deltanet_decode_full(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Fused DeltaNet single-token recurrence on device.

        Updates the recurrent state and returns the raw q @ S_new output. The
        caller applies gated RMSNorm and silu(z), which is faster as a parallel
        TTNN tail than serializing those operations on one core per head.

        Args:
            q (ttnn.Tensor): [B,Hk,Dk] raw Q. Normalization and scaling are fused.
            k (ttnn.Tensor): [B,Hk,Dk] raw K. Normalization is fused.
            v (ttnn.Tensor): [B,H,Dv] value vectors.
            beta (ttnn.Tensor): [1,B,H] update coefficient, or raw b when
                decay_scale and dt_bias are supplied.
            decay (ttnn.Tensor): [1,B,H] recurrent-state decay, or raw a when
                decay_scale and dt_bias are supplied.
            recurrent_state (ttnn.Tensor): [B,H,Dk,Dv] recurrent state

        Keyword Args:
            num_heads (int): Number of value/output heads.
            num_k_heads (int): Number of key heads (before expansion).
            k_head_dim (int): Key head dimension.
            v_head_dim (int): Value head dimension.
            head_expand_ratio (int): num_heads / num_k_heads.
            memory_config (ttnn.MemoryConfig, optional): output memory config.
            decay_scale (ttnn.Tensor, optional): [1,1,H] negative exp(A).
                Supplying this and dt_bias fuses sigmoid(b) and
                exp(decay_scale * softplus(a + dt_bias)).
            dt_bias (ttnn.Tensor, optional): [1,1,H] decay bias.

        Returns:
            list[ttnn.Tensor]: [raw_output, new_recurrent_state]
        )doc";

    ttnn::bind_function<"deltanet_decode_full", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::deltanet_decode_full,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("v"),
        nb::arg("beta"),
        nb::arg("decay"),
        nb::arg("recurrent_state"),
        nb::kw_only(),
        nb::arg("num_heads"),
        nb::arg("num_k_heads"),
        nb::arg("k_head_dim"),
        nb::arg("v_head_dim"),
        nb::arg("head_expand_ratio"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("decay_scale") = nb::none(),
        nb::arg("dt_bias") = nb::none());
}

}  // namespace ttnn::operations::experimental::deltanet::detail

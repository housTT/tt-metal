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

        Updates the recurrent state and returns q @ S_new. When gate and
        norm_weight are supplied, per-head RMSNorm and SiLU gating are fused
        and the returned activation is ready for the output projection.

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
            gate (ttnn.Tensor, optional): [1,B,H*Dv] SiLU gate. Supplying this
                and norm_weight fuses the decode epilogue.
            norm_weight (ttnn.Tensor, optional): [1,1,Dv] RMSNorm weight.
            norm_epsilon (float): RMSNorm epsilon.
            packed_qkv (bool): Interpret q as packed [1,B,Q|K|V] and ignore
                k/v. This removes decode head-split and reshape operations.
            packed_projection (bool): Interpret beta, decay, and gate as the shared
                packed [Q|K|V|Z|A|B] projection. This removes their decode slices.

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
        nb::arg("dt_bias") = nb::none(),
        nb::arg("gate") = nb::none(),
        nb::arg("norm_weight") = nb::none(),
        nb::arg("norm_epsilon") = 1e-6F,
        nb::arg("packed_qkv") = false,
        nb::arg("packed_projection") = false);

    const auto* conv_doc =
        R"doc(
        Fused four-tap causal convolution and persistent decode-state update.

        Computes silu(state1*tap0 + state2*tap1 + state3*tap2 + input*tap3)
        while shifting state1->state0, state2->state1, state3->state2, and
        input->state3 in the same device operation. The returned tensor keeps
        the packed [Q|K|V] layout.
        )doc";

    ttnn::bind_function<"deltanet_conv1d_decode", "ttnn.experimental.">(
        mod,
        conv_doc,
        &ttnn::experimental::deltanet_conv1d_decode,
        nb::arg("input"),
        nb::arg("state0"),
        nb::arg("state1"),
        nb::arg("state2"),
        nb::arg("state3"),
        nb::arg("tap0"),
        nb::arg("tap1"),
        nb::arg("tap2"),
        nb::arg("tap3"),
        nb::kw_only(),
        nb::arg("q_width"),
        nb::arg("k_width"),
        nb::arg("v_width"),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::deltanet::detail

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
            qkv_proj (ttnn.Tensor): [1,1,1,2*Hk*Dk+H*Dv] normalized Q/K and V.
            beta (ttnn.Tensor): [1,1,1,H] update coefficient.
            decay (ttnn.Tensor): [1,1,1,H] recurrent-state decay.
            recurrent_state (ttnn.Tensor): [1,H,Dk,Dv] recurrent state

        Keyword Args:
            num_heads (int): Number of value/output heads.
            num_k_heads (int): Number of key heads (before expansion).
            k_head_dim (int): Key head dimension.
            v_head_dim (int): Value head dimension.
            head_expand_ratio (int): num_heads / num_k_heads.
            memory_config (ttnn.MemoryConfig, optional): output memory config.

        Returns:
            list[ttnn.Tensor]: [raw_output, new_recurrent_state]
        )doc";

    ttnn::bind_function<"deltanet_decode_full", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::deltanet_decode_full,
        nb::arg("qkv_proj"),
        nb::arg("beta"),
        nb::arg("decay"),
        nb::arg("recurrent_state"),
        nb::kw_only(),
        nb::arg("num_heads"),
        nb::arg("num_k_heads"),
        nb::arg("k_head_dim"),
        nb::arg("v_head_dim"),
        nb::arg("head_expand_ratio"),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::deltanet::detail

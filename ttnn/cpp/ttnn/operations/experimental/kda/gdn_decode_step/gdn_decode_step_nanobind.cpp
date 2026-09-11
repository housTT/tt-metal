// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gdn_decode_step_nanobind.hpp"

#include "gdn_decode_step.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::gdn_decode_step::detail {

void bind_gdn_decode_step(nb::module_& mod) {
    ttnn::bind_function<"gdn_decode_step", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        One fused gated-delta-net decode step for ``B`` single-token rows.

        Per value head ``h`` (key head ``h / qk_head_repeat``) the kernel
        computes the 4-tap causal convolution ``silu(tap0*w0 + tap1*w1 +
        tap2*w2 + x*w3)`` on the head's q/k/v channels, L2-normalizes ``q`` and
        ``k`` (``x / sqrt(sum x^2 + qk_norm_epsilon)``, ``q`` scaled by
        ``1/sqrt(K)``), runs ``S = exp(g) S; d = v - k S; S += beta k^T d;
        o = q S`` with ``state`` updated in place (``state_output`` may alias
        it), and emits ``rmsnorm(o) * norm_weight * sigmoid(gate_h)`` into the
        time-first ``[B, 1, 1, H*V]`` output.

        Shapes: ``x, tap0..2 = [B,1,1,W]`` FLOAT32 with ``W = 2*Hk*K + H*V``
        (q | k | v channels), ``conv_w0..3 = [1,1,1,W]``, ``beta, log_decay =
        [B,H,1,1]`` FLOAT32, ``state = [B,H,K,V]`` FLOAT32, ``gate = [B,1,1,H*V]``
        BFLOAT16, ``norm_weight = [1,1,1,V]`` BFLOAT16.  The caller rotates the
        taps (``tap0 <- tap1 <- tap2 <- x``) after the step.
        )doc",
        &ttnn::experimental::kda::gdn_decode_step,
        nb::arg("x").noconvert(),
        nb::arg("tap0").noconvert(),
        nb::arg("tap1").noconvert(),
        nb::arg("tap2").noconvert(),
        nb::arg("conv_w0").noconvert(),
        nb::arg("conv_w1").noconvert(),
        nb::arg("conv_w2").noconvert(),
        nb::arg("conv_w3").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("log_decay").noconvert(),
        nb::arg("state").noconvert(),
        nb::arg("gate").noconvert(),
        nb::arg("norm_weight").noconvert(),
        nb::arg("num_heads"),
        nb::arg("key_dim"),
        nb::arg("value_dim"),
        nb::kw_only(),
        nb::arg("state_output").noconvert() = nb::none(),
        nb::arg("qk_head_repeat") = 1,
        nb::arg("qk_norm_epsilon") = 1e-6f,
        nb::arg("norm_epsilon") = 1e-6f,
        nb::arg("output_dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::gdn_decode_step::detail

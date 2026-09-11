// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hc_mixer_nanobind.hpp"

#include "hc_mixer.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::hc_mixer::detail {

void bind_hc_mixer(nb::module_& mod) {
    ttnn::bind_function<"hc_mix_post", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Decode tail of the hyper-connection input mixer.

        ``packed`` BFLOAT16 holds partial ``[low | inject]`` rows (``L``
        tile-aligned columns + ``S`` gates) that the kernel sums: either
        ``[1,1,R,L+S]`` (R partial rows of one product, e.g. all-gathered rank
        partials) or ``[1,S,R*S,L+S]`` (batch ``s`` = stream ``s``'s
        ``weighted @ down`` rows where only row ``r*S+s`` of rank ``r``
        belongs to the product, as produced by a batched matmul over the
        stream-major down weight);
        ``weighted = [1,1,S,N]`` BFLOAT16 is the normalized, weighted residual
        with one stream per row; ``up = [1,1,L,S*N]`` (BFLOAT16 or BFLOAT8_B)
        is the mixer's up projection with the streams concatenated along
        columns.  Returns ``mixed = [1,1,1,N]`` BFLOAT16 =
        ``mean_s(weighted[s] * sigmoid(silu(low) @ up[:, s*N:(s+1)*N]))`` and
        ``injection = [1,1,1,S]`` BFLOAT16 (the trailing gate columns).
        )doc",
        &ttnn::experimental::kda::hc_mix_post,
        nb::arg("packed").noconvert(),
        nb::arg("weighted").noconvert(),
        nb::arg("up").noconvert(),
        nb::arg("lowrank"),
        nb::arg("streams"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
    ttnn::bind_function<"hc_inject", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        ``out[s, :] = hyper[s, :] + 2 * sigmoid(injection[s]) * block`` for
        ``hyper = [1,1,S,N]``, ``block = [1,1,1,N]`` and ``injection =
        [1,1,1,S]`` (all BFLOAT16, tile layout).
        )doc",
        &ttnn::experimental::kda::hc_inject,
        nb::arg("hyper").noconvert(),
        nb::arg("block").noconvert(),
        nb::arg("injection").noconvert(),
        nb::arg("streams"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::hc_mixer::detail

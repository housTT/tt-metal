// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/qkv_causal_conv1d_silu_nanobind.hpp"
#include "ttnn/operations/experimental/kda/reduce_affine_transforms/reduce_affine_transforms_nanobind.hpp"
#include "ttnn/operations/experimental/kda/gdn_decode_step/gdn_decode_step_nanobind.hpp"
#include "ttnn/operations/experimental/kda/hc_mixer/hc_mixer_nanobind.hpp"
#include "ttnn/operations/experimental/kda/moe_decode/moe_decode_nanobind.hpp"
#include "ttnn/operations/experimental/kda/recurrent_gated_delta_rule/recurrent_gated_delta_rule_nanobind.hpp"
#include "ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/sigmoid_gated_rms_norm_nanobind.hpp"

namespace ttnn::operations::experimental::kda::detail {

void bind_kda(nb::module_& mod) {
    auto kda_module = mod.def_submodule("kda", "Experimental KDA operations");
    qkv_causal_conv1d_silu::detail::bind_qkv_causal_conv1d_silu(kda_module);
    reduce_affine_transforms::detail::bind_reduce_affine_transforms(kda_module);
    recurrent_gated_delta_rule::detail::bind_recurrent_gated_delta_rule(kda_module);
    gdn_decode_step::detail::bind_gdn_decode_step(kda_module);
    hc_mixer::detail::bind_hc_mixer(kda_module);
    moe_decode::detail::bind_moe_decode(kda_module);
    sigmoid_gated_rms_norm::detail::bind_sigmoid_gated_rms_norm(kda_module);
}

}  // namespace ttnn::operations::experimental::kda::detail

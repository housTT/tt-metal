// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hc_mix_post_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct HcMixPostProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const HcMixPostParams&, const HcMixPostInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim

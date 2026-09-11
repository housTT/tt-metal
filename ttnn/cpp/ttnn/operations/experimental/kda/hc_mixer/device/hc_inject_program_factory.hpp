// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hc_inject_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct HcInjectProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const HcInjectParams&, const HcInjectInputs&, Tensor&);
};

}  // namespace ttnn::experimental::prim

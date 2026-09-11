// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_weighted_sum_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct MoeWeightedSumProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MoeWeightedSumParams&, const MoeWeightedSumInputs&, Tensor&);
};

}  // namespace ttnn::experimental::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "recurrent_gated_delta_rule_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct RecurrentGatedDeltaRuleProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RecurrentGatedDeltaRuleParams&, const RecurrentGatedDeltaRuleInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim

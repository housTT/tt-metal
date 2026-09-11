// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_sort_slabs_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct MoeSortSlabsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MoeSortSlabsParams&, const MoeSortSlabsInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim

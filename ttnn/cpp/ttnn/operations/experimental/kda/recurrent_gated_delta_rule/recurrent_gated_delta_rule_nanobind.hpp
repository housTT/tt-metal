// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::kda::recurrent_gated_delta_rule::detail {
void bind_recurrent_gated_delta_rule(nanobind::module_& mod);
}

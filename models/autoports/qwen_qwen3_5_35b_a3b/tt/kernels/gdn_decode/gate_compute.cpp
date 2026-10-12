// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common.hpp"

void kernel_main() {
    constexpr uint32_t inv_width_bits = get_compile_time_arg_val(0);
    constexpr uint32_t norm_eps_bits = get_compile_time_arg_val(1);
    compute_kernel_hw_startup(CB_O, CB_OUT);
    norm_gate<inv_width_bits, norm_eps_bits>();
}

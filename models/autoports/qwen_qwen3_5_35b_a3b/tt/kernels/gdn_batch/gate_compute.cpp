// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "../gdn_decode/common.hpp"

void kernel_main() {
    constexpr uint32_t inv_width_bits = get_compile_time_arg_val(0);
    constexpr uint32_t norm_eps_bits = get_compile_time_arg_val(1);
    compute_kernel_hw_startup(CB_O, CB_OUT);
    cb_wait_front(CB_O, 4);
    begin();
    load(CB_O, 0, 0);
    mul(0, 0, 0);
    for (uint32_t c = 1; c < 4; ++c) {
        load(CB_O, c, 1);
        mul(1, 1, 1);
        add(0, 1, 0);
    }
    row_sum(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_width_bits);
    add_unary_tile(0, norm_eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    finish();
    pack_to(0, CB_SC);
    release();
    cb_wait_front(CB_SC, 1);
    cb_wait_front(CB_W, 4);
    cb_wait_front(CB_Z, 4);
    for (uint32_t c = 0; c < 4; ++c) {
        begin();
        load(CB_O, c, 0);
        bcast<BroadcastType::COL>(CB_SC, 0, 1);
        mul(0, 1, 0);
        bcast<BroadcastType::ROW>(CB_W, c, 1);
        mul(0, 1, 0);
        load(CB_Z, c, 1);
        silu_tile_init();
        silu_tile(1);
        mul(0, 1, 0);
        finish();
        pack_to(0, CB_OUT);
        release();
    }
    cb_pop_front(CB_SC, 1);
    cb_pop_front(CB_W, 4);
    cb_pop_front(CB_Z, 4);
    cb_pop_front(CB_O, 4);
}

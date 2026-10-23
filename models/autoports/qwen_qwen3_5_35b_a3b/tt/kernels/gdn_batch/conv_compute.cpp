// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "../gdn_decode/common.hpp"

constexpr uint32_t CONV_X = 0;
constexpr uint32_t CONV_H = 1;
constexpr uint32_t CONV_W = 2;
constexpr uint32_t CONV_SUM = 3;
constexpr uint32_t CONV_OUT = 4;

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CONV_H, CONV_OUT);
    for (uint32_t tile = 0; tile < count; ++tile) {
        cb_wait_front(CONV_W, 4);
        cb_wait_front(CONV_H, 3);
        cb_wait_front(CONV_X, 1);
        begin();
        load(CONV_H, 0, 0);
        bcast<BroadcastType::ROW>(CONV_W, 0, 1);
        mul(0, 1, 0);
        for (uint32_t r = 1; r < 3; ++r) {
            load(CONV_H, r, 2);
            bcast<BroadcastType::ROW>(CONV_W, r, 1);
            mul(2, 1, 2);
            add(0, 2, 0);
        }
        load(CONV_X, 0, 2);
        bcast<BroadcastType::ROW>(CONV_W, 3, 1);
        mul(2, 1, 2);
        add(0, 2, 0);
        finish();
        pack_to(0, CONV_SUM);
        release();
        cb_pop_front(CONV_W, 4);
        cb_pop_front(CONV_H, 3);
        cb_pop_front(CONV_X, 1);
        cb_wait_front(CONV_SUM, 1);
        begin();
        load(CONV_SUM, 0, 0);
        silu_tile_init();
        silu_tile(0);
        finish();
        pack_to(0, CONV_OUT);
        release();
        cb_pop_front(CONV_SUM, 1);
    }
}

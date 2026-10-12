// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"

constexpr uint32_t CB_X = 0;
constexpr uint32_t CB_W = 1;
constexpr uint32_t CB_OUT = 16;

void kernel_main() {
    constexpr uint32_t kt = get_compile_time_arg_val(0);
    constexpr uint32_t nper = get_compile_time_arg_val(1);
    constexpr uint32_t kb = get_compile_time_arg_val(2);
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_W, CB_OUT);
    matmul_init(CB_X, CB_W);
    tile_regs_acquire();
    for (uint32_t k0 = 0; k0 < kt; k0 += kb) {
        cb_wait_front(CB_X, kb);
        cb_wait_front(CB_W, kb * nper * 2);
        for (uint32_t k = 0; k < kb; ++k) {
            for (uint32_t j = 0; j < nper; ++j) {
                matmul_tiles(CB_X, CB_W, k, (k * nper + j) * 2, 2 * j);
                matmul_tiles(CB_X, CB_W, k, (k * nper + j) * 2 + 1, 2 * j + 1);
            }
        }
        cb_pop_front(CB_X, kb);
        cb_pop_front(CB_W, kb * nper * 2);
    }
    silu_tile_init();
    for (uint32_t j = 0; j < nper; ++j) {
        silu_tile(2 * j + 1);
    }
    mul_binary_tile_init();
    for (uint32_t j = 0; j < nper; ++j) {
        mul_binary_tile(2 * j, 2 * j + 1, 2 * j);
    }
    tile_regs_commit();
    cb_reserve_back(CB_OUT, nper);
    tile_regs_wait();
    for (uint32_t j = 0; j < nper; ++j) {
        pack_tile(2 * j, CB_OUT);
    }
    tile_regs_release();
    cb_push_back(CB_OUT, nper);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Four-tap depthwise causal convolution followed by precise SiLU. Taps are
// stored in row zero and broadcast over the active decode batch rows.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t cb_input = get_compile_time_arg_val(0);
constexpr uint32_t cb_state1 = get_compile_time_arg_val(1);
constexpr uint32_t cb_state2 = get_compile_time_arg_val(2);
constexpr uint32_t cb_state3 = get_compile_time_arg_val(3);
constexpr uint32_t cb_tap0 = get_compile_time_arg_val(4);
constexpr uint32_t cb_tap1 = get_compile_time_arg_val(5);
constexpr uint32_t cb_tap2 = get_compile_time_arg_val(6);
constexpr uint32_t cb_tap3 = get_compile_time_arg_val(7);
constexpr uint32_t cb_partial = get_compile_time_arg_val(8);
constexpr uint32_t cb_output = get_compile_time_arg_val(9);

inline void first_tap(uint32_t activation_cb, uint32_t tap_cb) {
    cb_wait_front(activation_cb, 1);
    cb_wait_front(tap_cb, 1);
    cb_reserve_back(cb_partial, 1);
    reconfig_data_format(activation_cb, tap_cb);
    mul_bcast_rows_init(activation_cb, tap_cb);
    tile_regs_acquire();
    mul_tiles_bcast_rows(activation_cb, tap_cb, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_partial);
    tile_regs_release();
    cb_push_back(cb_partial, 1);
    cb_pop_front(activation_cb, 1);
    cb_pop_front(tap_cb, 1);
}

inline void accumulate_tap(uint32_t activation_cb, uint32_t tap_cb, bool final_tap) {
    cb_wait_front(activation_cb, 1);
    cb_wait_front(tap_cb, 1);
    cb_wait_front(cb_partial, 1);
    const uint32_t output_cb = final_tap ? cb_output : cb_partial;
    cb_reserve_back(output_cb, 1);

    reconfig_data_format(activation_cb, tap_cb);
    mul_bcast_rows_init(activation_cb, tap_cb);
    tile_regs_acquire();
    mul_tiles_bcast_rows(activation_cb, tap_cb, 0, 0, 0);
    reconfig_data_format_srca(cb_partial);
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_partial);
    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_partial, 0, 0);
    if (final_tap) {
        silu_tile(0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, output_cb);
    tile_regs_release();

    cb_push_back(output_cb, 1);
    cb_pop_front(cb_partial, 1);
    cb_pop_front(activation_cb, 1);
    cb_pop_front(tap_cb, 1);
}

void kernel_main() {
    compute_kernel_hw_startup(cb_state1, cb_tap0, cb_output);
    silu_tile_init();
    first_tap(cb_state1, cb_tap0);
    accumulate_tap(cb_state2, cb_tap1, false);
    accumulate_tap(cb_state3, cb_tap2, false);
    accumulate_tap(cb_input, cb_tap3, true);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Each core owns one QKV channel tile. Read the three live history tiles and
// current projection once, enqueue them for convolution, and shift the
// persistent four-tap state in place before the next decode step.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_state1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_state2 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_state3 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_tap0 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_tap1 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_tap2 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_tap3 = get_compile_time_arg_val(7);
    constexpr auto input_args = TensorAccessorArgs<8>();
    constexpr auto state0_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto state1_args = TensorAccessorArgs<state0_args.next_compile_time_args_offset()>();
    constexpr auto state2_args = TensorAccessorArgs<state1_args.next_compile_time_args_offset()>();
    constexpr auto state3_args = TensorAccessorArgs<state2_args.next_compile_time_args_offset()>();
    constexpr auto tap0_args = TensorAccessorArgs<state3_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t state0_addr = get_arg_val<uint32_t>(1);
    const uint32_t state1_addr = get_arg_val<uint32_t>(2);
    const uint32_t state2_addr = get_arg_val<uint32_t>(3);
    const uint32_t state3_addr = get_arg_val<uint32_t>(4);
    const uint32_t tap0_addr = get_arg_val<uint32_t>(5);
    const uint32_t tap1_addr = get_arg_val<uint32_t>(6);
    const uint32_t tap2_addr = get_arg_val<uint32_t>(7);
    const uint32_t tap3_addr = get_arg_val<uint32_t>(8);
    const uint32_t tile = get_arg_val<uint32_t>(9);

    const uint32_t tile_bytes = get_tile_size(cb_input);
    const auto input_accessor = TensorAccessor(input_args, input_addr, tile_bytes);
    const auto state0_accessor = TensorAccessor(state0_args, state0_addr, tile_bytes);
    const auto state1_accessor = TensorAccessor(state1_args, state1_addr, tile_bytes);
    const auto state2_accessor = TensorAccessor(state2_args, state2_addr, tile_bytes);
    const auto state3_accessor = TensorAccessor(state3_args, state3_addr, tile_bytes);
    const auto tap0_accessor = TensorAccessor(tap0_args, tap0_addr, tile_bytes);
    const auto tap1_accessor = TensorAccessor(tap1_args, tap1_addr, tile_bytes);
    const auto tap2_accessor = TensorAccessor(tap2_args, tap2_addr, tile_bytes);
    const auto tap3_accessor = TensorAccessor(tap3_args, tap3_addr, tile_bytes);

    cb_reserve_back(cb_input, 1);
    cb_reserve_back(cb_state1, 1);
    cb_reserve_back(cb_state2, 1);
    cb_reserve_back(cb_state3, 1);
    cb_reserve_back(cb_tap0, 1);
    cb_reserve_back(cb_tap1, 1);
    cb_reserve_back(cb_tap2, 1);
    cb_reserve_back(cb_tap3, 1);

    const uint32_t input_l1 = get_write_ptr(cb_input);
    const uint32_t state1_l1 = get_write_ptr(cb_state1);
    const uint32_t state2_l1 = get_write_ptr(cb_state2);
    const uint32_t state3_l1 = get_write_ptr(cb_state3);
    const uint32_t tap0_l1 = get_write_ptr(cb_tap0);
    const uint32_t tap1_l1 = get_write_ptr(cb_tap1);
    const uint32_t tap2_l1 = get_write_ptr(cb_tap2);
    const uint32_t tap3_l1 = get_write_ptr(cb_tap3);

    noc_async_read_tile(tile, input_accessor, input_l1);
    noc_async_read_tile(tile, state1_accessor, state1_l1);
    noc_async_read_tile(tile, state2_accessor, state2_l1);
    noc_async_read_tile(tile, state3_accessor, state3_l1);
    noc_async_read_tile(tile, tap0_accessor, tap0_l1);
    noc_async_read_tile(tile, tap1_accessor, tap1_l1);
    noc_async_read_tile(tile, tap2_accessor, tap2_l1);
    noc_async_read_tile(tile, tap3_accessor, tap3_l1);
    noc_async_read_barrier();

    noc_async_write_tile(tile, state0_accessor, state1_l1);
    noc_async_write_tile(tile, state1_accessor, state2_l1);
    noc_async_write_tile(tile, state2_accessor, state3_l1);
    noc_async_write_tile(tile, state3_accessor, input_l1);
    noc_async_write_barrier();

    cb_push_back(cb_input, 1);
    cb_push_back(cb_state1, 1);
    cb_push_back(cb_state2, 1);
    cb_push_back(cb_state3, 1);
    cb_push_back(cb_tap0, 1);
    cb_push_back(cb_tap1, 1);
    cb_push_back(cb_tap2, 1);
    cb_push_back(cb_tap3, 1);
}

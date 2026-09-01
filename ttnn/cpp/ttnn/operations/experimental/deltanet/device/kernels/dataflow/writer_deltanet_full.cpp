// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_state_output = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t k_head_dim_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t v_head_dim_tiles = get_compile_time_arg_val(3);
    constexpr auto state_args = TensorAccessorArgs<4>();
    constexpr auto output_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();

    const uint32_t state_output_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t state_start_tile = get_arg_val<uint32_t>(2);
    const uint32_t output_start_tile = get_arg_val<uint32_t>(3);

    constexpr uint32_t state_tiles = k_head_dim_tiles * v_head_dim_tiles;
    const uint32_t state_tile_bytes = get_tile_size(cb_state_output);
    const uint32_t output_tile_bytes = get_tile_size(cb_output);
    const auto state_accessor = TensorAccessor(state_args, state_output_addr, state_tile_bytes);
    const auto output_accessor = TensorAccessor(output_args, output_addr, output_tile_bytes);

    cb_wait_front(cb_state_output, state_tiles);
    uint32_t state_l1 = get_read_ptr(cb_state_output);
    for (uint32_t tile = 0; tile < state_tiles; ++tile) {
        noc_async_write_tile(state_start_tile + tile, state_accessor, state_l1);
        state_l1 += state_tile_bytes;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_state_output, state_tiles);

    cb_wait_front(cb_output, v_head_dim_tiles);
    uint32_t output_l1 = get_read_ptr(cb_output);
    for (uint32_t tile = 0; tile < v_head_dim_tiles; ++tile) {
        noc_async_write_tile(output_start_tile + tile, output_accessor, output_l1);
        output_l1 += output_tile_bytes;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_output, v_head_dim_tiles);
}

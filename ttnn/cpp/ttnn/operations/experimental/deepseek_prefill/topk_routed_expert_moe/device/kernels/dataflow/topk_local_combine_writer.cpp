// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stream one scalar routing weight for every token/slot. Invalid slots get an
// exact zero generated in L1. After compute has reduced K and fused the BF8
// pack, write this core's one 32-token tile row to the final output.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {
    constexpr uint32_t cb_weight_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_valid_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weight_scratch_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output_id = get_compile_time_arg_val(3);
    constexpr uint32_t map_chunk_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t weight_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t weight_tile_size = get_compile_time_arg_val(6);
    constexpr uint32_t output_tile_size = get_compile_time_arg_val(7);
    constexpr uint32_t emb_dim_out_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t tokens = get_compile_time_arg_val(9);
    constexpr uint32_t topk = get_compile_time_arg_val(10);

    constexpr auto weights_args = TensorAccessorArgs<11>();
    constexpr auto valid_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<valid_args.next_compile_time_args_offset()>();

    const uint32_t weights_addr = get_arg_val<uint32_t>(0);
    const uint32_t valid_addr = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_arg_val<uint32_t>(2);
    const uint32_t token_start = get_arg_val<uint32_t>(3);

    const auto weights = TensorAccessor(weights_args, weights_addr);
    const auto valid = TensorAccessor(valid_args, valid_addr);
    const auto output = TensorAccessor(output_args, output_addr);
    CircularBuffer cb_weight(cb_weight_id);
    CircularBuffer cb_valid(cb_valid_id);
    CircularBuffer cb_weight_scratch(cb_weight_scratch_id);
    CircularBuffer cb_output(cb_output_id);

    constexpr uint32_t tokens_per_chunk = 32;
    const uint32_t valid_l1 = cb_valid.get_write_ptr();
    const uint32_t weight_scratch_l1 = cb_weight_scratch.get_write_ptr();

    Noc noc;
    for (uint32_t slot = 0; slot < topk; ++slot) {
        const uint32_t source_word = slot * tokens + token_start;
        const uint32_t destination_offset = slot * tokens_per_chunk * sizeof(uint32_t);
        noc.async_read(
            valid,
            CoreLocalMem<uint32_t>(valid_l1 + destination_offset),
            tokens_per_chunk * sizeof(uint32_t),
            {.page_id = 0, .offset_bytes = source_word * sizeof(uint32_t)},
            {});
    }
    noc.async_read_barrier();

    volatile tt_l1_ptr uint32_t* is_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_l1);
    // All token pages in this chunk are independent DRAM reads. Queue them
    // together so one barrier replaces the former per-token serialization.
    for (uint32_t token = 0; token < tokens_per_chunk; ++token) {
        noc.async_read(
            weights,
            CoreLocalMem<uint32_t>(weight_scratch_l1 + token * weight_page_size),
            weight_page_size,
            {.page_id = token_start + token},
            {});
    }
    noc.async_read_barrier();

    for (uint32_t token = 0; token < tokens_per_chunk; ++token) {
        volatile tt_l1_ptr uint16_t* token_weights =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(weight_scratch_l1 + token * weight_page_size);
        for (uint32_t slot = 0; slot < topk; ++slot) {
            const uint32_t map_index = slot * tokens_per_chunk + token;
            const uint32_t weight = is_local[map_index] == 1 ? token_weights[slot] : 0;
            generate_bcast_unary_scalar(cb_weight, weight | (weight << 16));
        }
    }

    cb_output.wait_front(emb_dim_out_tiles);
    const uint32_t first_output_tile = (token_start / tokens_per_chunk) * emb_dim_out_tiles;
    for (uint32_t tile = 0; tile < emb_dim_out_tiles; ++tile) {
        noc.async_write(
            cb_output,
            output,
            output_tile_size,
            {.offset_bytes = tile * output_tile_size},
            {.page_id = first_output_tile + tile});
    }
    noc.async_write_barrier();
    cb_output.pop_front(emb_dim_out_tiles);

    (void)map_chunk_bytes;
    (void)weight_tile_size;
}

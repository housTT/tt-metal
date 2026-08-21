// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gather only device-local packed rows for one 32-token block. Invalid slots
// are zero-generated in L1; the possibly-uninitialized fused-FFN row they name
// is never read, so neither NaN propagation nor an out-of-range address can be
// hidden behind a zero routing weight.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    constexpr uint32_t cb_packed_row_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_inverse_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_valid_id = get_compile_time_arg_val(2);
    constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t emb_dim_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t input_tile_size = get_compile_time_arg_val(5);
    constexpr uint32_t map_chunk_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t tokens = get_compile_time_arg_val(7);
    constexpr uint32_t topk = get_compile_time_arg_val(8);
    constexpr uint32_t capacity = get_compile_time_arg_val(9);
    constexpr uint32_t assignment_addressed = get_compile_time_arg_val(10);
    constexpr uint32_t rows_per_batch = get_compile_time_arg_val(11);
    static_assert(rows_per_batch > 0);
    static_assert(rows_per_batch <= topk);

    constexpr auto packed_y_args = TensorAccessorArgs<12>();
    constexpr auto inverse_args = TensorAccessorArgs<packed_y_args.next_compile_time_args_offset()>();
    constexpr auto valid_args = TensorAccessorArgs<inverse_args.next_compile_time_args_offset()>();

    const uint32_t packed_y_addr = get_arg_val<uint32_t>(0);
    const uint32_t inverse_addr = get_arg_val<uint32_t>(1);
    const uint32_t valid_addr = get_arg_val<uint32_t>(2);
    const uint32_t token_start = get_arg_val<uint32_t>(3);

    const auto packed_y = TensorAccessor(packed_y_args, packed_y_addr);
    const auto inverse = TensorAccessor(inverse_args, inverse_addr);
    const auto valid = TensorAccessor(valid_args, valid_addr);
    CircularBuffer cb_packed_row(cb_packed_row_id);
    CircularBuffer cb_inverse(cb_inverse_id);
    CircularBuffer cb_valid(cb_valid_id);

    constexpr uint32_t tokens_per_chunk = 32;
    constexpr uint32_t words_per_slot_chunk = tokens_per_chunk;
    const uint32_t inverse_l1 = cb_inverse.get_write_ptr();
    const uint32_t valid_l1 = cb_valid.get_write_ptr();

    Noc noc;
    for (uint32_t slot = 0; slot < topk; ++slot) {
        const uint32_t source_word = slot * tokens + token_start;
        const uint32_t destination_offset = slot * tokens_per_chunk * sizeof(uint32_t);
        if constexpr (assignment_addressed == 0) {
            noc.async_read(
                inverse,
                CoreLocalMem<uint32_t>(inverse_l1 + destination_offset),
                tokens_per_chunk * sizeof(uint32_t),
                {.page_id = 0, .offset_bytes = source_word * sizeof(uint32_t)},
                {});
        }
        noc.async_read(
            valid,
            CoreLocalMem<uint32_t>(valid_l1 + destination_offset),
            tokens_per_chunk * sizeof(uint32_t),
            {.page_id = 0, .offset_bytes = source_word * sizeof(uint32_t)},
            {});
    }
    noc.async_read_barrier();

    volatile tt_l1_ptr uint32_t* packed_rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inverse_l1);
    volatile tt_l1_ptr uint32_t* is_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_l1);
    constexpr uint32_t row_storage_bytes = emb_dim_cb_tiles * input_tile_size;

    for (uint32_t token = 0; token < tokens_per_chunk; ++token) {
        for (uint32_t slot_base = 0; slot_base < topk; slot_base += rows_per_batch) {
            const uint32_t rows_in_batch = (topk - slot_base) < rows_per_batch ? topk - slot_base : rows_per_batch;
            const uint32_t tiles_in_batch = rows_in_batch * emb_dim_cb_tiles;
            cb_packed_row.reserve_back(tiles_in_batch);

            // Initialize every slot before selectively overwriting local rows. This
            // keeps invalid/uninitialized expert output out of arithmetic while
            // reducing K independent zero/read barriers to one pair per batch.
            noc.async_write_zeros(cb_packed_row, rows_in_batch * row_storage_bytes, {.offset_bytes = 0});
            noc.write_zeros_l1_barrier();

            const uint32_t batch_l1 = cb_packed_row.get_write_ptr();
            bool has_local_read = false;
            for (uint32_t batch_row = 0; batch_row < rows_in_batch; ++batch_row) {
                const uint32_t slot = slot_base + batch_row;
                const uint32_t map_index = slot * words_per_slot_chunk + token;
                uint32_t packed_row;
                if constexpr (assignment_addressed != 0) {
                    packed_row = (token_start + token) * topk + slot;
                } else {
                    packed_row = packed_rows[map_index];
                }
                if (is_local[map_index] == 1 && packed_row < capacity) {
                    noc.async_read(
                        packed_y,
                        CoreLocalMem<uint32_t>(batch_l1 + batch_row * row_storage_bytes),
                        emb_dim_bytes,
                        {.page_id = packed_row},
                        {});
                    has_local_read = true;
                }
            }
            if (has_local_read) {
                noc.async_read_barrier();
            }
            cb_packed_row.push_back(tiles_in_batch);
        }
    }

    (void)map_chunk_bytes;
}

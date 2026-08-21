// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Parallel compact activation dispatch. Each core owns a contiguous range of
// packed rows, reads its assignment-id slice once, and copies the selected
// ROW_MAJOR BF16 activation sticks. Tile-alignment padding is written as exact
// zero so the fused FFN may safely evaluate ceil(count/32) rows.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t cb_assignments_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_row_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_zero_id = get_compile_time_arg_val(2);
    constexpr uint32_t assignment_chunk_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t x_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t topk = get_compile_time_arg_val(6);

    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto assignments_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<assignments_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t assignments_addr = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_arg_val<uint32_t>(2);
    const uint32_t valid_tokens = get_arg_val<uint32_t>(3);
    const uint32_t row_start = get_arg_val<uint32_t>(4);
    const uint32_t row_count = get_arg_val<uint32_t>(5);
    const uint32_t ready_semaphore_id = get_arg_val<uint32_t>(6);

    const auto x = TensorAccessor(x_args, x_addr);
    const auto assignments = TensorAccessor(assignments_args, assignments_addr);
    const auto output = TensorAccessor(output_args, output_addr);

    CircularBuffer cb_assignments(cb_assignments_id);
    CircularBuffer cb_row(cb_row_id);
    CircularBuffer cb_zero(cb_zero_id);
    const uint32_t assignments_l1 = cb_assignments.get_write_ptr();
    const uint32_t row_l1 = cb_row.get_write_ptr();
    const uint32_t zero_l1 = cb_zero.get_write_ptr();

    volatile tt_l1_ptr uint32_t* zero_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_l1);
    for (uint32_t word = 0; word < output_page_size / sizeof(uint32_t); ++word) {
        zero_words[word] = 0;
    }

    Semaphore<> ready(ready_semaphore_id);
    ready.wait_min(1);

    Noc noc;
    if (row_count > 0) {
        noc.async_read(
            assignments,
            CoreLocalMem<uint32_t>(assignments_l1),
            row_count * sizeof(uint32_t),
            {.page_id = 0, .offset_bytes = row_start * sizeof(uint32_t)},
            {});
        noc.async_read_barrier();
    }

    volatile tt_l1_ptr uint32_t* assignment_ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(assignments_l1);
    const uint32_t valid_assignment_limit = valid_tokens * topk;
    for (uint32_t local_row = 0; local_row < row_count; ++local_row) {
        const uint32_t assignment = assignment_ids[local_row];
        const uint32_t output_row = row_start + local_row;
        if (assignment < valid_assignment_limit) {
            const uint32_t token = assignment / topk;
            noc.async_read(x, CoreLocalMem<uint32_t>(row_l1), x_page_size, {.page_id = token}, {});
            noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(row_l1), output, output_page_size, {}, {.page_id = output_row});
            // row_l1 is the source of the posted write. It cannot be reused by
            // the next token read until that write has drained.
            noc.async_write_barrier();
        } else {
            noc.async_write(CoreLocalMem<uint32_t>(zero_l1), output, output_page_size, {}, {.page_id = output_row});
        }
    }
    noc.async_write_barrier();

    // Re-arm the cache-resident program for the next invocation.
    volatile tt_l1_ptr uint32_t* ready_value =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_semaphore_id));
    *ready_value = 0;
    (void)assignment_chunk_bytes;
}

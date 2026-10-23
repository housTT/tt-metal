// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_X = 0;
constexpr uint32_t CB_H = 1;
constexpr uint32_t CB_W = 2;
constexpr uint32_t CB_ROWS = 6;
constexpr uint32_t SPAN = 64;

FORCE_INLINE uint32_t row_index(uint32_t row, uint32_t col) {
    return ((row >= 16) ? 512 : 0) + ((col >= 16) ? 256 : 0) + (row % 16) * 16 + (col % 16);
}

FORCE_INLINE void put_row(uint32_t tile, uint32_t row, uint32_t src) {
    volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile);
    volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src);
    for (uint32_t c = 0; c < 32; ++c) {
        t[row_index(row, c)] = s[c];
    }
}

FORCE_INLINE void get_row(uint32_t tile, uint32_t row, uint32_t dst) {
    volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile);
    volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst);
    for (uint32_t c = 0; c < 32; ++c) {
        d[c] = t[row_index(row, c)];
    }
}

FORCE_INLINE void zero_tile(uint32_t tile) {
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
    for (uint32_t i = 0; i < 512; ++i) {
        t[i] = 0;
    }
}

void kernel_main() {
    constexpr uint32_t users = get_compile_time_arg_val(0);
    constexpr auto x_args = TensorAccessorArgs<1>();
    constexpr auto w0_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w1_args = TensorAccessorArgs<w0_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<w1_args.next_compile_time_args_offset()>();
    constexpr auto w3_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<w3_args.next_compile_time_args_offset()>();

    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto w0 = TensorAccessor(w0_args, get_arg_val<uint32_t>(1));
    const auto w1 = TensorAccessor(w1_args, get_arg_val<uint32_t>(2));
    const auto w2 = TensorAccessor(w2_args, get_arg_val<uint32_t>(3));
    const auto w3 = TensorAccessor(w3_args, get_arg_val<uint32_t>(4));
    const uint32_t first = get_arg_val<uint32_t>(5);
    const uint32_t count = get_arg_val<uint32_t>(6);
    constexpr uint32_t state_arg = 7;
    constexpr uint32_t tile_bytes = get_tile_size(CB_X);
    Noc noc;
    CircularBuffer rows_cb(CB_ROWS);
    const uint32_t rows = rows_cb.get_write_ptr();

    for (uint32_t tile = first; tile < first + count; ++tile) {
        CircularBuffer wcb(CB_W);
        wcb.reserve_back(4);
        uint32_t wdst = wcb.get_write_ptr();
        noc.async_read(w0, CoreLocalMem<uint32_t>(wdst), tile_bytes, {.page_id = tile}, {});
        noc.async_read(w1, CoreLocalMem<uint32_t>(wdst + tile_bytes), tile_bytes, {.page_id = tile}, {});
        noc.async_read(w2, CoreLocalMem<uint32_t>(wdst + 2 * tile_bytes), tile_bytes, {.page_id = tile}, {});
        noc.async_read(w3, CoreLocalMem<uint32_t>(wdst + 3 * tile_bytes), tile_bytes, {.page_id = tile}, {});

        CircularBuffer xcb(CB_X);
        xcb.reserve_back(1);
        const uint32_t xtile = xcb.get_write_ptr();
        noc.async_read(x, CoreLocalMem<uint32_t>(xtile), tile_bytes, {.page_id = tile}, {});

        CircularBuffer hcb(CB_H);
        hcb.reserve_back(3);
        const uint32_t htile = hcb.get_write_ptr();
        zero_tile(htile);
        zero_tile(htile + tile_bytes);
        zero_tile(htile + 2 * tile_bytes);
        for (uint32_t u = 0; u < users; ++u) {
            const auto state = TensorAccessor(s_args, get_arg_val<uint32_t>(state_arg + u));
            for (uint32_t r = 0; r < 3; ++r) {
                noc.async_read(state, CoreLocalMem<uint32_t>(rows + (u * 3 + r) * SPAN), SPAN,
                               {.page_id = r, .offset_bytes = tile * SPAN}, {});
            }
        }
        noc.async_read_barrier();
        for (uint32_t u = 0; u < users; ++u) {
            for (uint32_t r = 0; r < 3; ++r) {
                put_row(htile + r * tile_bytes, u, rows + (u * 3 + r) * SPAN);
            }
        }
        wcb.push_back(4);
        hcb.push_back(3);
        for (uint32_t u = 0; u < users; ++u) {
            get_row(xtile, u, rows + (u * 3) * SPAN);
        }
        xcb.push_back(1);
        for (uint32_t u = 0; u < users; ++u) {
            const auto state = TensorAccessor(s_args, get_arg_val<uint32_t>(state_arg + u));
            const uint32_t base = rows + u * 3 * SPAN;
            noc.async_write(CoreLocalMem<uint32_t>(base + SPAN), state, SPAN, {}, {.page_id = 0, .offset_bytes = tile * SPAN});
            noc.async_write(CoreLocalMem<uint32_t>(base + 2 * SPAN), state, SPAN, {}, {.page_id = 1, .offset_bytes = tile * SPAN});
            noc.async_write(CoreLocalMem<uint32_t>(base), state, SPAN, {}, {.page_id = 2, .offset_bytes = tile * SPAN});
        }
        noc.async_write_barrier();
    }
}

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
constexpr uint32_t CB_W = 1;
constexpr uint32_t CB_IDS = 2;

void kernel_main() {
    constexpr uint32_t kt = get_compile_time_arg_val(0);
    constexpr uint32_t nt = get_compile_time_arg_val(1);
    constexpr uint32_t nper = get_compile_time_arg_val(2);
    constexpr uint32_t kb = get_compile_time_arg_val(3);
    constexpr uint32_t ids_bytes = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto i_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto ids = TensorAccessor(i_args, get_arg_val<uint32_t>(1));
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(2));
    const auto sw = TensorAccessor(s_args, get_arg_val<uint32_t>(5));
    const uint32_t slot = get_arg_val<uint32_t>(3);
    const uint32_t first = get_arg_val<uint32_t>(4);
    const bool shared = get_arg_val<uint32_t>(6);
    Noc noc;
    uint32_t expert = 0;
    if (!shared) {
        CircularBuffer ids_cb(CB_IDS);
        uint32_t ids_ptr = ids_cb.get_write_ptr();
        noc.async_read(ids, CoreLocalMem<uint32_t>(ids_ptr), ids_bytes, {.page_id = 0}, {});
        noc.async_read_barrier();
        expert = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ids_ptr)[slot];
    }
    constexpr uint32_t x_bytes = get_tile_size(CB_X);
    constexpr uint32_t w_bytes = get_tile_size(CB_W);
    constexpr uint32_t half = nt / 2;
    CircularBuffer xc(CB_X);
    CircularBuffer wc(CB_W);
    for (uint32_t k0 = 0; k0 < kt; k0 += kb) {
        xc.reserve_back(kb);
        wc.reserve_back(kb * nper * 2);
        uint32_t xd = xc.get_write_ptr();
        uint32_t wd = wc.get_write_ptr();
        for (uint32_t k = k0; k < k0 + kb; ++k) {
            noc.async_read(x, CoreLocalMem<uint32_t>(xd), x_bytes, {.page_id = k}, {});
            xd += x_bytes;
            const uint32_t row = (expert * kt + k) * nt;
            for (uint32_t j = first; j < first + nper; ++j) {
                if (shared) {
                    noc.async_read(sw, CoreLocalMem<uint32_t>(wd), w_bytes, {.page_id = k * nt + 2 * j + 1}, {});
                    wd += w_bytes;
                    noc.async_read(sw, CoreLocalMem<uint32_t>(wd), w_bytes, {.page_id = k * nt + 2 * j}, {});
                    wd += w_bytes;
                } else {
                    noc.async_read(w, CoreLocalMem<uint32_t>(wd), w_bytes, {.page_id = row + j}, {});
                    wd += w_bytes;
                    noc.async_read(w, CoreLocalMem<uint32_t>(wd), w_bytes, {.page_id = row + half + j}, {});
                    wd += w_bytes;
                }
            }
        }
        noc.async_read_barrier();
        xc.push_back(kb);
        wc.push_back(kb * nper * 2);
    }
}

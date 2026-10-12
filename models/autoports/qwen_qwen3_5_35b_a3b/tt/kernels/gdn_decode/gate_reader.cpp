// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_Z = 3;
constexpr uint32_t CB_W = 7;
constexpr uint32_t CB_O = 13;
constexpr uint32_t CB_SCRATCH = 18;

template <uint32_t cb_id, typename Acc>
FORCE_INLINE void read_tiles(const Noc& noc, const Acc& acc, uint32_t first, uint32_t n) {
    constexpr uint32_t bytes = get_tile_size(cb_id);
    CircularBuffer cb(cb_id);
    cb.reserve_back(n);
    uint32_t dst = cb.get_write_ptr();
    for (uint32_t i = 0; i < n; ++i) {
        noc.async_read(acc, CoreLocalMem<uint32_t>(dst), bytes, {.page_id = first + i}, {});
        dst += bytes;
    }
    noc.async_read_barrier();
    cb.push_back(n);
}

void kernel_main() {
    constexpr auto o_args = TensorAccessorArgs<0>();
    constexpr auto z_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    const auto o = TensorAccessor(o_args, get_arg_val<uint32_t>(0));
    const auto z = TensorAccessor(z_args, get_arg_val<uint32_t>(1));
    const auto weight = TensorAccessor(w_args, get_arg_val<uint32_t>(2));
    const uint32_t head = get_arg_val<uint32_t>(3);
    const uint32_t z_offset = get_arg_val<uint32_t>(4);
    Noc noc;
    read_tiles<CB_O>(noc, o, head * 4, 4);
    read_tiles<CB_Z>(noc, z, z_offset + head * 4, 4);
    CircularBuffer scratch(CB_SCRATCH);
    uint32_t src = scratch.get_write_ptr();
    for (uint32_t i = 0; i < 4; ++i) {
        noc.async_read(weight, CoreLocalMem<uint32_t>(src + i * 2048), 2048, {.page_id = i}, {});
    }
    noc.async_read_barrier();
    CircularBuffer cb(CB_W);
    cb.reserve_back(4);
    uint32_t dst = cb.get_write_ptr();
    for (uint32_t i = 0; i < 4; ++i) {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + i * 4096);
        volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src + i * 2048);
        for (uint32_t c = 0; c < 32; ++c) {
            uint32_t index = c < 16 ? c : 256 + c - 16;
            t[index] = static_cast<uint32_t>(s[index]) << 16;
        }
    }
    cb.push_back(4);
}

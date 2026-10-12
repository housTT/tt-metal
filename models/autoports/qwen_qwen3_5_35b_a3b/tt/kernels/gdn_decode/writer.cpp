// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_O = 13;
constexpr uint32_t CB_SOUT = 16;
constexpr uint32_t CB_OUT = 17;

void kernel_main() {
    constexpr uint32_t blocks = get_compile_time_arg_val(0);
    constexpr auto s_args = TensorAccessorArgs<1>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto state = TensorAccessor(s_args, get_arg_val<uint32_t>(0));
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(1));
    const uint32_t head = get_arg_val<uint32_t>(2);
    const uint32_t first_block = get_arg_val<uint32_t>(3);
    Noc noc;
    CircularBuffer sout(CB_SOUT);
    constexpr uint32_t s_bytes = get_tile_size(CB_SOUT);
    for (uint32_t c = first_block; c < first_block + blocks; ++c) {
        for (uint32_t r = 0; r < 4; ++r) {
            sout.wait_front(1);
            noc.async_write(sout, state, s_bytes, {.offset_bytes = 0}, {.page_id = head * 16 + r * 4 + c});
            noc.async_writes_flushed();
            sout.pop_front(1);
        }
    }
    constexpr uint32_t cb_id = blocks == 4 ? CB_OUT : CB_O;
    CircularBuffer cb(cb_id);
    constexpr uint32_t o_bytes = get_tile_size(cb_id);
    cb.wait_front(blocks);
    for (uint32_t c = 0; c < blocks; ++c) {
        noc.async_write(cb, out, o_bytes, {.offset_bytes = c * o_bytes}, {.page_id = head * 4 + first_block + c});
    }
    noc.async_write_barrier();
    cb.pop_front(blocks);
}

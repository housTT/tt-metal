// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_OUT = 17;
constexpr uint32_t CB_SHIFT = 19;

void kernel_main() {
    constexpr auto o_args = TensorAccessorArgs<0>();
    constexpr auto s_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(0));
    const uint32_t head = get_arg_val<uint32_t>(1);
    const auto state = TensorAccessor(s_args, get_arg_val<uint32_t>(2));
    const auto current = TensorAccessor(c_args, get_arg_val<uint32_t>(3));
    const uint32_t offset = get_arg_val<uint32_t>(4);
    const uint32_t span = get_arg_val<uint32_t>(5);
    Noc noc;
    CircularBuffer shift(CB_SHIFT);
    const uint32_t base = shift.get_write_ptr();
    noc.async_read(state, CoreLocalMem<uint32_t>(base), span, {.page_id = 1, .offset_bytes = offset}, {});
    noc.async_read(state, CoreLocalMem<uint32_t>(base + span), span, {.page_id = 2, .offset_bytes = offset}, {});
    noc.async_read(current, CoreLocalMem<uint32_t>(base + 2 * span), span, {.page_id = 0, .offset_bytes = offset}, {});
    noc.async_read_barrier();
    for (uint32_t r = 0; r < 3; ++r) {
        noc.async_write(CoreLocalMem<uint32_t>(base + r * span), state, span, {}, {.page_id = r, .offset_bytes = offset});
    }
    CircularBuffer cb(CB_OUT);
    constexpr uint32_t bytes = get_tile_size(CB_OUT);
    cb.wait_front(4);
    for (uint32_t c = 0; c < 4; ++c) {
        noc.async_write(cb, out, bytes, {.offset_bytes = c * bytes}, {.page_id = head * 4 + c});
    }
    noc.async_write_barrier();
    cb.pop_front(4);
}

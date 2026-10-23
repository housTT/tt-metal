// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_O = 13;
constexpr uint32_t CB_SOUT = 16;
constexpr uint32_t ROW_BYTES = 64;

void kernel_main() {
    constexpr auto s_args = TensorAccessorArgs<0>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(0));
    const uint32_t head = get_arg_val<uint32_t>(1);
    const uint32_t block = get_arg_val<uint32_t>(2);
    const uint32_t first_user = get_arg_val<uint32_t>(3);
    const uint32_t users = get_arg_val<uint32_t>(4);
    constexpr uint32_t state_arg = 5;
    Noc noc;
    CircularBuffer sout(CB_SOUT);
    CircularBuffer o(CB_O);
    constexpr uint32_t s_bytes = get_tile_size(CB_SOUT);
    for (uint32_t u = first_user; u < first_user + users; ++u) {
        const auto state = TensorAccessor(s_args, get_arg_val<uint32_t>(state_arg + u - first_user));
        for (uint32_t r = 0; r < 4; ++r) {
            sout.wait_front(1);
            noc.async_write(sout, state, s_bytes, {.offset_bytes = 0}, {.page_id = head * 16 + r * 4 + block});
            noc.async_writes_flushed();
            sout.pop_front(1);
        }
        o.wait_front(1);
        const uint32_t face = u >= 16 ? 2 : 0;
        const uint32_t row = (u % 16) * 16 * 4;
        noc.async_write(o, out, ROW_BYTES, {.offset_bytes = 0},
                        {.page_id = head * 4 + block, .offset_bytes = face * 1024 + row});
        noc.async_write(o, out, ROW_BYTES, {.offset_bytes = 1024},
                        {.page_id = head * 4 + block, .offset_bytes = (face + 1) * 1024 + row});
        noc.async_write_barrier();
        o.pop_front(1);
    }
}

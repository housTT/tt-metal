// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_OUT = 16;

void kernel_main() {
    constexpr uint32_t nper = get_compile_time_arg_val(0);
    constexpr uint32_t half = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    constexpr auto s_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(0));
    const auto shared_out = TensorAccessor(s_args, get_arg_val<uint32_t>(3));
    const uint32_t slot = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const bool shared = get_arg_val<uint32_t>(4);
    Noc noc;
    CircularBuffer cb(CB_OUT);
    constexpr uint32_t bytes = get_tile_size(CB_OUT);
    cb.wait_front(nper);
    for (uint32_t j = 0; j < nper; ++j) {
        if (shared) {
            noc.async_write(cb, shared_out, bytes, {.offset_bytes = j * bytes}, {.page_id = first + j});
        } else {
            noc.async_write(cb, out, bytes, {.offset_bytes = j * bytes}, {.page_id = slot * half + first + j});
        }
    }
    noc.async_write_barrier();
    cb.pop_front(nper);
}

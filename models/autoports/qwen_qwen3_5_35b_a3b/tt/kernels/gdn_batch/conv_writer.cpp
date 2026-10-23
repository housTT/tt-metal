// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_OUT = 4;

void kernel_main() {
    constexpr auto o_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(0));
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    Noc noc;
    CircularBuffer cb(CB_OUT);
    constexpr uint32_t bytes = get_tile_size(CB_OUT);
    for (uint32_t tile = first; tile < first + count; ++tile) {
        cb.wait_front(1);
        noc.async_write(cb, out, bytes, {.offset_bytes = 0}, {.page_id = tile});
        noc.async_writes_flushed();
        cb.pop_front(1);
    }
    noc.async_write_barrier();
}

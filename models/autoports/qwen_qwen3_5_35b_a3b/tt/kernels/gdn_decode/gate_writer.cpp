// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_OUT = 17;

void kernel_main() {
    constexpr auto o_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(0));
    const uint32_t head = get_arg_val<uint32_t>(1);
    Noc noc;
    CircularBuffer cb(CB_OUT);
    constexpr uint32_t bytes = get_tile_size(CB_OUT);
    cb.wait_front(4);
    for (uint32_t c = 0; c < 4; ++c) {
        noc.async_write(cb, out, bytes, {.offset_bytes = c * bytes}, {.page_id = head * 4 + c});
    }
    noc.async_write_barrier();
    cb.pop_front(4);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

TT_KERNEL void writer(uint32_t tile_start, uint32_t tile_count) {
    const auto out_acc = TensorAccessor(tensor::out);
    DataflowBuffer out(dfb::out);
    Noc noc;
    for (uint32_t offset = 0; offset < tile_count; ++offset) {
        out.wait_front(1);
        noc.async_write(out, out_acc, out.get_entry_size(), {.offset_bytes = 0}, {.page_id = tile_start + offset});
        noc.async_write_barrier();
        out.pop_front(1);
    }
}

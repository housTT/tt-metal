// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
template <typename Accessor>
void write_tile(Noc& noc, DataflowBuffer& buffer, const Accessor& accessor, uint32_t page) {
    buffer.wait_front(1);
    noc.async_write(buffer, accessor, buffer.get_entry_size(), {.offset_bytes = 0}, {.page_id = page});
    noc.async_write_barrier();
    buffer.pop_front(1);
}
}  // namespace

TT_KERNEL void writer(uint32_t col_start, uint32_t col_count, uint32_t emit_injection) {
    const auto mixed_acc = TensorAccessor(tensor::mixed);
    const auto inj_acc = TensorAccessor(tensor::injection);
    DataflowBuffer out(dfb::out), inj(dfb::inj_out);
    Noc noc;
    if (emit_injection) {
        write_tile(noc, inj, inj_acc, 0);
    }
    for (uint32_t offset = 0; offset < col_count; ++offset) {
        write_tile(noc, out, mixed_acc, col_start + offset);
    }
}

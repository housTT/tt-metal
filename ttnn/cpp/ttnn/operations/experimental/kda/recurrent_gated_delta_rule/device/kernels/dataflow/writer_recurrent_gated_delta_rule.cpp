// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {

template <typename Accessor>
void write_tiles(Noc& noc, DataflowBuffer& buffer, const Accessor& accessor, uint32_t first, uint32_t count) {
    buffer.wait_front(count);
    for (uint32_t tile = 0; tile < count; ++tile) {
        noc.async_write(
            buffer,
            accessor,
            buffer.get_entry_size(),
            {.offset_bytes = tile * buffer.get_entry_size()},
            {.page_id = first + tile});
    }
    noc.async_write_barrier();
    buffer.pop_front(count);
}

}  // namespace

template <uint32_t Kt, uint32_t Vt>
TT_KERNEL void writer(uint32_t head_start, uint32_t head_count) {
    const auto core_acc = TensorAccessor(tensor::core_out);
    const auto state_acc = TensorAccessor(tensor::state_out);
    DataflowBuffer core(dfb::core_out), state(dfb::state_out);
    Noc noc;
    constexpr uint32_t kv = Kt * Vt;
    for (uint32_t offset = 0; offset < head_count; ++offset) {
        const uint32_t head = head_start + offset;
        write_tiles(noc, core, core_acc, head * Vt, Vt);
        write_tiles(noc, state, state_acc, head * kv, kv);
    }
}

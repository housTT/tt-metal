// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Output tile i = (group g = i / It, column c = i % It); the gate tile is (g, c) and
// the up tile (g, It + c) of the [1,G,rows,2I] input, i.e. pages g*2It + c and g*2It + It + c.
template <uint32_t It>
TT_KERNEL void reader(uint32_t tile_start, uint32_t tile_count) {
    const auto acc = TensorAccessor(tensor::gate_up);
    DataflowBuffer gate(dfb::gate), up(dfb::up);
    Noc noc;
    for (uint32_t offset = 0; offset < tile_count; ++offset) {
        const uint32_t i = tile_start + offset;
        const uint32_t g = i / It;
        const uint32_t c = i - g * It;
        gate.reserve_back(1);
        up.reserve_back(1);
        noc.async_read(acc, gate, gate.get_entry_size(), {.page_id = g * 2 * It + c}, {.offset_bytes = 0});
        noc.async_read(acc, up, up.get_entry_size(), {.page_id = g * 2 * It + It + c}, {.offset_bytes = 0});
        noc.async_read_barrier();
        gate.push_back(1);
        up.push_back(1);
    }
}

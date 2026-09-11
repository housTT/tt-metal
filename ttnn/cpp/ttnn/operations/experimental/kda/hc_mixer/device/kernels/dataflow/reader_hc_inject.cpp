// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
template <typename Accessor>
void read_tile(Noc& noc, const Accessor& accessor, DataflowBuffer& buffer, uint32_t page) {
    buffer.reserve_back(1);
    noc.async_read(accessor, buffer, buffer.get_entry_size(), {.page_id = page}, {.offset_bytes = 0});
    noc.async_read_barrier();
    buffer.push_back(1);
}

// Rows 0..S-1 of column 0 hold 1.0: P4 @ X replicates X's row 0 into rows 0..S-1.
void generate_replicate_tile(DataflowBuffer& buffer, uint32_t streams) {
    buffer.reserve_back(1);
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buffer.get_write_ptr());
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = 0;
    }
    for (uint32_t s = 0; s < streams; ++s) {
        ptr[s * 16] = 0x3F80;
    }
    buffer.push_back(1);
}
}  // namespace

template <uint32_t S>
TT_KERNEL void reader(uint32_t col_start, uint32_t col_count) {
    const auto hyper_acc = TensorAccessor(tensor::hyper);
    const auto block_acc = TensorAccessor(tensor::block);
    const auto inj_acc = TensorAccessor(tensor::injection);
    DataflowBuffer inj(dfb::inj), p4(dfb::p4), block(dfb::block), hyper(dfb::hyper);
    Noc noc;
    read_tile(noc, inj_acc, inj, 0);
    generate_replicate_tile(p4, S);
    for (uint32_t offset = 0; offset < col_count; ++offset) {
        const uint32_t c = col_start + offset;
        read_tile(noc, block_acc, block, c);
        read_tile(noc, hyper_acc, hyper, c);
    }
}

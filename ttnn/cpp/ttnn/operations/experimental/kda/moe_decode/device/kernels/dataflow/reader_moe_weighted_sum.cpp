// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
// P_g: 1.0 at (row g, column 0): P_g @ T moves T's row 0 into row g.
void generate_row_select_tiles(DataflowBuffer& buffer, uint32_t count) {
    buffer.reserve_back(count);
    auto* base = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buffer.get_write_ptr());
    for (uint32_t g = 0; g < count; ++g) {
        auto* ptr = base + g * 1024;
        for (uint32_t i = 0; i < 1024; ++i) {
            ptr[i] = 0;
        }
        ptr[g * 16] = 0x3F80;
    }
    buffer.push_back(count);
}
}  // namespace

template <uint32_t K, uint32_t Nt>
TT_KERNEL void reader(uint32_t col_start, uint32_t col_count) {
    const auto groups_acc = TensorAccessor(tensor::groups);
    const auto scores_acc = TensorAccessor(tensor::scores);
    DataflowBuffer p(dfb::p), w(dfb::w), g(dfb::g);
    Noc noc;
    generate_row_select_tiles(p, K);
    w.reserve_back(1);
    noc.async_read(scores_acc, w, w.get_entry_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    w.push_back(1);
    for (uint32_t offset = 0; offset < col_count; ++offset) {
        const uint32_t c = col_start + offset;
        g.reserve_back(K);
        for (uint32_t group = 0; group < K; ++group) {
            noc.async_read(
                groups_acc,
                g,
                g.get_entry_size(),
                {.page_id = group * Nt + c},
                {.offset_bytes = group * g.get_entry_size()});
        }
        noc.async_read_barrier();
        g.push_back(K);
    }
}

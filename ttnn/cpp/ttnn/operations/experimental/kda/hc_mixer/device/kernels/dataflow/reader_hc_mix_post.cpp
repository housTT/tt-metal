// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {

template <typename Accessor>
void read_tiles(
    Noc& noc, const Accessor& accessor, DataflowBuffer& buffer, uint32_t first, uint32_t count, uint32_t stride = 1) {
    buffer.reserve_back(count);
    for (uint32_t tile = 0; tile < count; ++tile) {
        noc.async_read(
            accessor,
            buffer,
            buffer.get_entry_size(),
            {.page_id = first + tile * stride},
            {.offset_bytes = tile * buffer.get_entry_size()});
    }
    noc.async_read_barrier();
    buffer.push_back(count);
}

void zero_bf16_tile(volatile tt_l1_ptr uint16_t* ptr) {
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = 0;
    }
}

// P_s: 1.0 at (row s, column 0) so that P_s @ M moves M's row 0 into row s.
void generate_row_select_tiles(DataflowBuffer& buffer, uint32_t streams) {
    buffer.reserve_back(streams);
    auto* base = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buffer.get_write_ptr());
    for (uint32_t s = 0; s < streams; ++s) {
        auto* ptr = base + s * 1024;
        zero_bf16_tile(ptr);
        ptr[s * 16] = 0x3F80;  // face 0, row s, column 0
    }
    buffer.push_back(streams);
}

// R: 1/S in row 0, columns 0..S-1 (face 0) so that R @ X averages X's first S rows.
void generate_average_row_tile(DataflowBuffer& buffer, uint32_t streams, uint16_t inv_streams) {
    buffer.reserve_back(1);
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buffer.get_write_ptr());
    zero_bf16_tile(ptr);
    for (uint32_t s = 0; s < streams; ++s) {
        ptr[s] = inv_streams;
    }
    buffer.push_back(1);
}

// Sel_b: row 0 has 1.0 at every column c < rows with c % PB == b, so Sel_b @ X
// sums the rows of X that belong to batch b's stream (all rows when PB == 1).
void generate_row_sum_selectors(DataflowBuffer& buffer, uint32_t pb, uint32_t rows) {
    buffer.reserve_back(pb);
    auto* base = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buffer.get_write_ptr());
    for (uint32_t b = 0; b < pb; ++b) {
        auto* ptr = base + b * 1024;
        zero_bf16_tile(ptr);
        for (uint32_t c = 0; c < rows; ++c) {
            if (c % pb == b) {
                ptr[c < 16 ? c : 256 + c - 16] = 0x3F80;  // row 0, column c
            }
        }
    }
    buffer.push_back(pb);
}

}  // namespace

template <uint32_t Lt, uint32_t Nt, uint32_t S, uint32_t PB, uint32_t PR, uint32_t PRows, uint32_t InvStreamsBf16>
TT_KERNEL void reader(uint32_t col_start, uint32_t col_count, uint32_t emit_injection) {
    const auto packed_acc = TensorAccessor(tensor::packed);
    const auto weighted_acc = TensorAccessor(tensor::weighted);
    const auto up_acc = TensorAccessor(tensor::up);
    DataflowBuffer packed(dfb::packed), sel(dfb::sel), p(dfb::p), r(dfb::r), w(dfb::w), wgt(dfb::wgt);
    Noc noc;
    // packed tile (block b, tile t): one tile row, so page = b * (Lt + 1) + t.
    read_tiles(noc, packed_acc, packed, 0, PR * PB * (Lt + 1));  // page = r*PB*(Lt+1) + b*(Lt+1) + t
    generate_row_sum_selectors(sel, PB, PRows);
    generate_row_select_tiles(p, S);
    generate_average_row_tile(r, S, static_cast<uint16_t>(InvStreamsBf16));
    for (uint32_t offset = 0; offset < col_count; ++offset) {
        const uint32_t c = col_start + offset;
        // up tile (k, s*Nt + c) for s in [0, S), k in [0, Lt): pushed stream-major.
        w.reserve_back(S * Lt);
        for (uint32_t s = 0; s < S; ++s) {
            for (uint32_t k = 0; k < Lt; ++k) {
                noc.async_read(
                    up_acc,
                    w,
                    w.get_entry_size(),
                    {.page_id = k * (S * Nt) + s * Nt + c},
                    {.offset_bytes = (s * Lt + k) * w.get_entry_size()});
            }
        }
        noc.async_read_barrier();
        w.push_back(S * Lt);
        read_tiles(noc, weighted_acc, wgt, c, 1);
    }
}

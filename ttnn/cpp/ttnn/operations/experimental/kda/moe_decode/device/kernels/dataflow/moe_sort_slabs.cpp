// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {

// Element (r, c) of a 32x32 tile with 16x16 faces: face = (r/16)*2 + c/16.
inline uint32_t tile_index(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

}  // namespace

// Groups the (row, k) routing entries of this rank's experts into 32-row slabs, one
// expert per slab, for an indexed sparse matmul whose A side is a gather of the
// slab rows.  Non-local entries get weight 0 and the dummy position P*32-1 (a
// position that the capacity bound guarantees is never assigned).
template <uint32_t Rt, uint32_t K, uint32_t LocalExperts, uint32_t P, uint32_t RowsBytes, uint32_t ExpertsBytes>
TT_KERNEL void reader() {
    constexpr uint32_t rows = Rt * 32;
    const auto idx_acc = TensorAccessor(tensor::indices);
    const auto scr_acc = TensorAccessor(tensor::scores);
    const auto base_acc = TensorAccessor(tensor::rank_base);
    const auto rows_acc = TensorAccessor(tensor::slab_rows);
    const auto experts_acc = TensorAccessor(tensor::slab_experts);
    const auto pos_acc = TensorAccessor(tensor::slab_pos);
    const auto lsc_acc = TensorAccessor(tensor::local_scores);
    DataflowBuffer idx(dfb::idx), scr(dfb::scr), base(dfb::base), rows_out(dfb::rows_out),
        experts_out(dfb::experts_out), pos_out(dfb::pos_out), scores_out(dfb::scores_out), scratch(dfb::scratch);
    Noc noc;

    idx.reserve_back(Rt);
    scr.reserve_back(Rt);
    for (uint32_t t = 0; t < Rt; ++t) {
        noc.async_read(idx_acc, idx, idx.get_entry_size(), {.page_id = t}, {.offset_bytes = t * idx.get_entry_size()});
        noc.async_read(scr_acc, scr, scr.get_entry_size(), {.page_id = t}, {.offset_bytes = t * scr.get_entry_size()});
    }
    base.reserve_back(1);
    noc.async_read(base_acc, base, 64, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const auto* idx_tiles = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx.get_write_ptr());
    const auto* scr_tiles = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr.get_write_ptr());
    const int32_t rank_base = *reinterpret_cast<volatile tt_l1_ptr int32_t*>(base.get_write_ptr());

    scratch.reserve_back(1);
    auto* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch.get_write_ptr());
    auto* offset = count + LocalExperts;
    auto* entry_expert = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(offset + LocalExperts);
    for (uint32_t e = 0; e < LocalExperts; ++e) {
        count[e] = 0;
    }
    // pass 1: local expert of every entry and per-expert hit counts
    for (uint32_t r = 0; r < rows; ++r) {
        const volatile tt_l1_ptr uint16_t* idx_tile = idx_tiles + (r >> 5) * 1024;
        for (uint32_t j = 0; j < K; ++j) {
            const int32_t rel = static_cast<int32_t>(idx_tile[tile_index(r & 31, j)]) - rank_base;
            const bool local = rel >= 0 && rel < static_cast<int32_t>(LocalExperts);
            entry_expert[r * K + j] = local ? static_cast<uint16_t>(rel) : 0xFFFFu;
            if (local) {
                ++count[rel];
            }
        }
    }
    // slab-aligned prefix offsets and slab expert ids
    experts_out.reserve_back(1);
    rows_out.reserve_back(1);
    auto* slab_expert = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(experts_out.get_write_ptr());
    auto* slab_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_out.get_write_ptr());
    for (uint32_t p = 0; p < P; ++p) {
        slab_expert[p] = 0;
    }
    for (uint32_t i = 0; i < P * 32; ++i) {
        slab_row[i] = 0;
    }
    uint32_t cursor = 0;
    for (uint32_t e = 0; e < LocalExperts; ++e) {
        offset[e] = cursor;
        const uint32_t slabs = (count[e] + 31) >> 5;
        for (uint32_t s = 0; s < slabs; ++s) {
            slab_expert[(cursor >> 5) + s] = static_cast<uint16_t>(e);
        }
        cursor += slabs << 5;
        count[e] = 0;  // reused as the fill cursor in pass 2
    }
    // pass 2: assign positions
    pos_out.reserve_back(Rt);
    scores_out.reserve_back(Rt);
    auto* pos_tiles = reinterpret_cast<volatile tt_l1_ptr int32_t*>(pos_out.get_write_ptr());
    auto* lsc_tiles = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_out.get_write_ptr());
    for (uint32_t i = 0; i < Rt * 1024; ++i) {
        pos_tiles[i] = static_cast<int32_t>(P * 32 - 1);
        lsc_tiles[i] = 0;
    }
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t t = r >> 5;
        for (uint32_t j = 0; j < K; ++j) {
            const uint16_t e = entry_expert[r * K + j];
            const uint32_t ti = t * 1024 + tile_index(r & 31, j);
            if (e == 0xFFFFu) {
                continue;
            }
            const uint32_t position = offset[e] + count[e];
            ++count[e];
            slab_row[position] = r;
            pos_tiles[ti] = static_cast<int32_t>(position);
            lsc_tiles[ti] = scr_tiles[ti];
        }
    }
    noc.async_write(rows_out, rows_acc, RowsBytes, {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write(experts_out, experts_acc, ExpertsBytes, {.offset_bytes = 0}, {.page_id = 0});
    for (uint32_t t = 0; t < Rt; ++t) {
        noc.async_write(
            pos_out, pos_acc, pos_out.get_entry_size(), {.offset_bytes = t * pos_out.get_entry_size()}, {.page_id = t});
        noc.async_write(
            scores_out,
            lsc_acc,
            scores_out.get_entry_size(),
            {.offset_bytes = t * scores_out.get_entry_size()},
            {.page_id = t});
    }
    noc.async_write_barrier();
}

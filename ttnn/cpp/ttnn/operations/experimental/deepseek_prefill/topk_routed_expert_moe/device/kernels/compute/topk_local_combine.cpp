// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Weight and reduce K packed expert rows in BF16, then tilize and pack the
// complete 32-token output block directly to BF8. No [K,T,H] tensor exists in
// either L1 or DRAM; the only full-width live state is one output row block.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_packed_row_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_row_major_accum_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output_id = get_compile_time_arg_val(3);
    constexpr uint32_t topk = get_compile_time_arg_val(4);
    constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t emb_dim_out_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t tokens_per_chunk = 32;
    constexpr uint32_t total_row_major_tiles = tokens_per_chunk * emb_dim_cb_tiles;
    static_assert(total_row_major_tiles == emb_dim_out_tiles);

    CircularBuffer cb_packed_row(cb_packed_row_id);
    CircularBuffer cb_weight(cb_weight_id);
    CircularBuffer cb_row_major_accum(cb_row_major_accum_id);

    // The multiply/reduce phase packs BF16 rows into the row-major scratch.
    // tilize() reconfigures PACK to the final BF8 output format afterwards.
    compute_kernel_hw_startup(cb_packed_row_id, cb_weight_id, cb_row_major_accum_id);
    cb_row_major_accum.reserve_back(total_row_major_tiles);

    for (uint32_t token = 0; token < tokens_per_chunk; ++token) {
        mul_bcast_scalar_init(cb_packed_row_id, cb_weight_id);
        for (uint32_t slot = 0; slot < topk; ++slot) {
            cb_packed_row.wait_front(emb_dim_cb_tiles);
            cb_weight.wait_front(1);
            pack_reconfig_l1_acc(slot == 0 ? 0 : 1);

            tile_regs_acquire();
            for (uint32_t tile = 0; tile < emb_dim_cb_tiles; ++tile) {
                mul_tiles_bcast<BroadcastType::SCALAR>(cb_packed_row_id, cb_weight_id, tile, 0, tile);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t tile = 0; tile < emb_dim_cb_tiles; ++tile) {
                pack_tile<true>(tile, cb_row_major_accum_id, token * emb_dim_cb_tiles + tile);
            }
            tile_regs_release();

            cb_packed_row.pop_front(emb_dim_cb_tiles);
            cb_weight.pop_front(1);
        }
        pack_reconfig_l1_acc(0);
    }
    cb_row_major_accum.push_back(total_row_major_tiles);

    // This pack reconfiguration performs the policy-boundary BF16 -> BF8
    // conversion while tilizing. The collective sees BF8 and no standalone
    // K-axis multiply/reduce/cast graph is emitted.
    compute_kernel_lib::tilize<total_row_major_tiles, cb_row_major_accum_id, cb_output_id>(1);
}

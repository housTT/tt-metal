// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"

namespace {
inline void wait(uint32_t cb, uint32_t count) { CircularBuffer(cb).wait_front(count); }
inline void pop(uint32_t cb, uint32_t count) { CircularBuffer(cb).pop_front(count); }
}  // namespace

template <uint32_t TwoBits>
TT_KERNEL void compute(uint32_t col_count) {
    compute_kernel_hw_startup(dfb::hyper, dfb::block, dfb::out);
    wait(dfb::inj, 1);
    wait(dfb::p4, 1);
    // gcol[s, 0] = 2 * sigmoid(injection[0, s]).
    cb_reserve_back(dfb::gcol, 1);
    pack_reconfig_data_format(dfb::gcol);
    reconfig_data_format_srca(dfb::inj);
    transpose_init(dfb::inj);
    tile_regs_acquire();
    transpose_tile(dfb::inj, 0, 0);
    sigmoid_tile_init();
    sigmoid_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, TwoBits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::gcol, 0);
    tile_regs_release();
    cb_push_back(dfb::gcol, 1);
    pop(dfb::inj, 1);
    wait(dfb::gcol, 1);
    for (uint32_t col = 0; col < col_count; ++col) {
        wait(dfb::block, 1);
        wait(dfb::hyper, 1);
        // b4 = P4 @ block: block row replicated into rows 0..S-1.
        cb_reserve_back(dfb::b4, 1);
        pack_reconfig_data_format(dfb::b4);
        reconfig_data_format(dfb::block, dfb::p4);
        matmul_init(dfb::p4, dfb::block);
        tile_regs_acquire();
        matmul_tiles(dfb::p4, dfb::block, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::b4, 0);
        tile_regs_release();
        cb_push_back(dfb::b4, 1);
        pop(dfb::block, 1);
        wait(dfb::b4, 1);
        // scaled[s, :] = b4[s, :] * gcol[s, 0].
        cb_reserve_back(dfb::scaled, 1);
        pack_reconfig_data_format(dfb::scaled);
        reconfig_data_format(dfb::b4, dfb::gcol);
        mul_bcast_cols_init(dfb::b4, dfb::gcol);
        tile_regs_acquire();
        mul_tiles_bcast_cols(dfb::b4, dfb::gcol, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::scaled, 0);
        tile_regs_release();
        cb_push_back(dfb::scaled, 1);
        pop(dfb::b4, 1);
        wait(dfb::scaled, 1);
        // out = hyper + scaled.
        cb_reserve_back(dfb::out, 1);
        pack_reconfig_data_format(dfb::out);
        reconfig_data_format(dfb::scaled, dfb::hyper);
        add_init(dfb::scaled, dfb::hyper);
        tile_regs_acquire();
        add_tiles(dfb::scaled, dfb::hyper, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out, 0);
        tile_regs_release();
        cb_push_back(dfb::out, 1);
        pop(dfb::scaled, 1);
        pop(dfb::hyper, 1);
    }
    pop(dfb::gcol, 1);
    pop(dfb::p4, 1);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"

namespace {
inline void wait(uint32_t cb, uint32_t count) { CircularBuffer(cb).wait_front(count); }
inline void pop(uint32_t cb, uint32_t count) { CircularBuffer(cb).pop_front(count); }
}  // namespace

// out = silu(gate) * up, one tile per iteration: gate -> dest 0 (silu), up -> dest 1, SFPU multiply.
TT_KERNEL void compute(uint32_t tile_count) {
    compute_kernel_hw_startup(dfb::gate, dfb::up, dfb::out);
    for (uint32_t i = 0; i < tile_count; ++i) {
        wait(dfb::gate, 1);
        wait(dfb::up, 1);
        cb_reserve_back(dfb::out, 1);
        pack_reconfig_data_format(dfb::out);
        tile_regs_acquire();
        reconfig_data_format_srca(dfb::gate);
        copy_tile_to_dst_init_short(dfb::gate);
        copy_tile(dfb::gate, 0, 0);
        silu_tile_init();
        silu_tile(0);
        reconfig_data_format_srca(dfb::up);
        copy_tile_to_dst_init_short(dfb::up);
        copy_tile(dfb::up, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out, 0);
        tile_regs_release();
        cb_push_back(dfb::out, 1);
        pop(dfb::gate, 1);
        pop(dfb::up, 1);
    }
}

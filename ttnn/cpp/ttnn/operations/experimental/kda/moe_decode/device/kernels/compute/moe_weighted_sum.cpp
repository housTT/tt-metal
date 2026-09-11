// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"

namespace {
inline void wait(uint32_t cb, uint32_t count) { CircularBuffer(cb).wait_front(count); }
inline void pop(uint32_t cb, uint32_t count) { CircularBuffer(cb).pop_front(count); }
}  // namespace

template <uint32_t K>
TT_KERNEL void compute(uint32_t col_count) {
    compute_kernel_hw_startup(dfb::p, dfb::g, dfb::out);
    wait(dfb::p, K);
    wait(dfb::w, 1);
    for (uint32_t col = 0; col < col_count; ++col) {
        wait(dfb::g, K);
        // M = sum_g P_g @ T_g: row g of M is row 0 of group g.
        cb_reserve_back(dfb::m, 1);
        pack_reconfig_data_format(dfb::m);
        reconfig_data_format(dfb::g, dfb::p);
        matmul_init(dfb::p, dfb::g);
        tile_regs_acquire();
        for (uint32_t group = 0; group < K; ++group) {
            matmul_tiles(dfb::p, dfb::g, group, group, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::m, 0);
        tile_regs_release();
        cb_push_back(dfb::m, 1);
        pop(dfb::g, K);
        wait(dfb::m, 1);
        // out = scores_row @ M.
        cb_reserve_back(dfb::out, 1);
        pack_reconfig_data_format(dfb::out);
        reconfig_data_format(dfb::m, dfb::w);
        matmul_init(dfb::w, dfb::m);
        tile_regs_acquire();
        matmul_tiles(dfb::w, dfb::m, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out, 0);
        tile_regs_release();
        cb_push_back(dfb::out, 1);
        pop(dfb::m, 1);
    }
    pop(dfb::p, K);
    pop(dfb::w, 1);
}

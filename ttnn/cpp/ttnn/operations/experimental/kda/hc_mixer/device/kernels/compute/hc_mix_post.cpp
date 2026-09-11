// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"

namespace {

inline void wait(uint32_t cb, uint32_t count) { CircularBuffer(cb).wait_front(count); }
inline void pop(uint32_t cb, uint32_t count) { CircularBuffer(cb).pop_front(count); }

void copy_tiles(uint32_t input, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(input);
    copy_tile_to_dst_init_short(input);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        copy_tile(input, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, tile);
        tile_regs_release();
    }
    cb_push_back(out, count);
}

void copy_tiles_from(uint32_t input, uint32_t first, uint32_t out) {
    cb_reserve_back(out, 1);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(input);
    copy_tile_to_dst_init_short(input);
    tile_regs_acquire();
    copy_tile(input, first, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    cb_push_back(out, 1);
}

void silu_tiles(uint32_t input, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(input);
    copy_tile_to_dst_init_short(input);
    silu_tile_init();
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        copy_tile(input, tile, 0);
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, tile);
        tile_regs_release();
    }
    cb_push_back(out, count);
}

}  // namespace

template <uint32_t Lt, uint32_t S, uint32_t PB, uint32_t PR>
TT_KERNEL void compute(uint32_t col_count, uint32_t emit_injection) {
    compute_kernel_hw_startup(dfb::packed, dfb::w, dfb::out);
    wait(dfb::packed, PR * PB * (Lt + 1));
    wait(dfb::sel, PB);
    // low_sum[t] = sum_b Sel_b @ packed[b, t]: row 0 is the fully reduced packed row.
    cb_reserve_back(dfb::low_sum, Lt + 1);
    pack_reconfig_data_format(dfb::low_sum);
    reconfig_data_format(dfb::packed, dfb::sel);
    matmul_init(dfb::sel, dfb::packed);
    for (uint32_t t = 0; t < Lt + 1; ++t) {
        tile_regs_acquire();
        for (uint32_t r = 0; r < PR; ++r) {
            for (uint32_t b = 0; b < PB; ++b) {
                matmul_tiles(dfb::sel, dfb::packed, b, (r * PB + b) * (Lt + 1) + t, 0);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::low_sum, t);
        tile_regs_release();
    }
    cb_push_back(dfb::low_sum, Lt + 1);
    pop(dfb::packed, PR * PB * (Lt + 1));
    pop(dfb::sel, PB);
    wait(dfb::low_sum, Lt + 1);
    silu_tiles(dfb::low_sum, dfb::low_act, Lt);
    wait(dfb::low_act, Lt);
    wait(dfb::p, S);
    wait(dfb::r, 1);
    if (emit_injection) {
        // the injection gates are the trailing tile of the reduced row
        copy_tiles_from(dfb::low_sum, Lt, dfb::inj_out);
    }
    pop(dfb::low_sum, Lt + 1);
    for (uint32_t col = 0; col < col_count; ++col) {
        wait(dfb::w, S * Lt);
        wait(dfb::wgt, 1);
        // M_s = silu(low) @ up_s  (row 0 valid), one tile per stream.
        cb_reserve_back(dfb::m, S);
        pack_reconfig_data_format(dfb::m);
        reconfig_data_format(dfb::w, dfb::low_act);
        matmul_init(dfb::low_act, dfb::w);
        for (uint32_t s = 0; s < S; ++s) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Lt; ++k) {
                matmul_tiles(dfb::low_act, dfb::w, k, s * Lt + k, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::m, s);
            tile_regs_release();
        }
        cb_push_back(dfb::m, S);
        pop(dfb::w, S * Lt);
        wait(dfb::m, S);
        // gate4 = sigmoid(sum_s P_s @ M_s): row s holds stream s's mix.
        cb_reserve_back(dfb::gate4, 1);
        pack_reconfig_data_format(dfb::gate4);
        reconfig_data_format(dfb::m, dfb::p);
        matmul_init(dfb::p, dfb::m);
        tile_regs_acquire();
        for (uint32_t s = 0; s < S; ++s) {
            matmul_tiles(dfb::p, dfb::m, s, s, 0);
        }
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::gate4, 0);
        tile_regs_release();
        cb_push_back(dfb::gate4, 1);
        pop(dfb::m, S);
        wait(dfb::gate4, 1);
        // prod = weighted * gate4 (rows 0..S-1).
        cb_reserve_back(dfb::prod, 1);
        pack_reconfig_data_format(dfb::prod);
        reconfig_data_format(dfb::wgt, dfb::gate4);
        mul_init(dfb::wgt, dfb::gate4, false);
        tile_regs_acquire();
        mul_tiles(dfb::wgt, dfb::gate4, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::prod, 0);
        tile_regs_release();
        cb_push_back(dfb::prod, 1);
        pop(dfb::wgt, 1);
        pop(dfb::gate4, 1);
        wait(dfb::prod, 1);
        // out = R @ prod: row 0 = mean over the S streams.
        cb_reserve_back(dfb::out, 1);
        pack_reconfig_data_format(dfb::out);
        reconfig_data_format(dfb::prod, dfb::r);
        matmul_init(dfb::r, dfb::prod);
        tile_regs_acquire();
        matmul_tiles(dfb::r, dfb::prod, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out, 0);
        tile_regs_release();
        cb_push_back(dfb::out, 1);
        pop(dfb::prod, 1);
    }
    pop(dfb::low_act, Lt);
    pop(dfb::p, S);
    pop(dfb::r, 1);
}

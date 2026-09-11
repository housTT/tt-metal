// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {

inline void wait(uint32_t cb, uint32_t count) { CircularBuffer(cb).wait_front(count); }
inline void pop(uint32_t cb, uint32_t count) { CircularBuffer(cb).pop_front(count); }

void mm(uint32_t a, uint32_t b, uint32_t out, uint32_t mt, uint32_t kt, uint32_t nt) {
    cb_reserve_back(out, mt * nt);
    pack_reconfig_data_format(out);
    reconfig_data_format(b, a);
    matmul_init(a, b);
    for (uint32_t mi = 0; mi < mt; ++mi) {
        for (uint32_t ni = 0; ni < nt; ++ni) {
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < kt; ++ki) {
                matmul_tiles(a, b, mi * kt + ki, ki * nt + ni, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out, mi * nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(out, mt * nt);
}

void ew(uint32_t a, uint32_t b, uint32_t out, uint32_t count, bool subtract) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    if (subtract) {
        sub_init(a, b);
    } else {
        add_init(a, b);
    }
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        if (subtract) {
            sub_tiles(a, b, tile, tile, 0);
        } else {
            add_tiles(a, b, tile, tile, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, tile);
        tile_regs_release();
    }
    cb_push_back(out, count);
}

void scalar_mul(uint32_t input, uint32_t scalar, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format(input, scalar);
    mul_bcast_scalar_init(input, scalar);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(input, scalar, tile, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, tile);
        tile_regs_release();
    }
    cb_push_back(out, count);
}

void exp_one(uint32_t input, uint32_t out) {
    cb_reserve_back(out, 1);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(input);
    copy_tile_to_dst_init_short(input);
    exp_tile_init();
    tile_regs_acquire();
    copy_tile(input, 0, 0);
    exp_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    cb_push_back(out, 1);
}

void transpose_tiles(uint32_t input, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(input);
    transpose_init(input);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        transpose_tile(input, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, tile);
        tile_regs_release();
    }
    cb_push_back(out, count);
}

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

void square_tiles(uint32_t input, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format(input, input);
    mul_init(input, input, false);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        mul_tiles(input, input, tile, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, tile);
        tile_regs_release();
    }
    cb_push_back(out, count);
}

// inv = (stats + eps)^-1/2 [* scale]: one tile whose column 0 holds the row values.
void inverse_norm(uint32_t stats, uint32_t eps, uint32_t out, uint32_t scale_bits) {
    cb_reserve_back(out, 1);
    pack_reconfig_data_format(out);
    reconfig_data_format(stats, eps);
    add_init(stats, eps);
    tile_regs_acquire();
    add_tiles(stats, eps, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    if (scale_bits != 0) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale_bits);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    cb_push_back(out, 1);
}

void scale_rows(uint32_t input, uint32_t inv, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format(input, inv);
    mul_bcast_cols_init(input, inv);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(input, inv, tile, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, tile);
        tile_regs_release();
    }
    cb_push_back(out, count);
}

// L2-normalize the single valid row of ``input`` (Kt tiles): out = x / sqrt(sum x^2 + eps) [* scale].
template <uint32_t Kt, uint32_t sq, uint32_t stats, uint32_t inv>
void l2_normalize(uint32_t input, uint32_t out, uint32_t scale_bits) {
    square_tiles(input, sq, Kt);
    wait(sq, Kt);
    compute_kernel_lib::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, sq, dfb::scaler, stats>(
        compute_kernel_lib::ReduceInputBlockShape::of(1, Kt));
    wait(stats, 1);
    inverse_norm(stats, dfb::eps, inv, scale_bits);
    wait(inv, 1);
    pop(stats, 1);
    scale_rows(input, inv, out, Kt);
    pop(inv, 1);
}

}  // namespace

template <uint32_t Kt, uint32_t Vt, uint32_t NormQK, uint32_t EpsBits, uint32_t QScaleBits>
TT_KERNEL void compute(uint32_t head_count) {
    constexpr uint32_t kv = Kt * Vt;
    compute_kernel_hw_startup(dfb::k, dfb::state, dfb::core_out);
    // Buffers feeding the recurrence: raw (pre-normalized) inputs, or the
    // in-kernel normalized copies.
    constexpr uint32_t q_in = NormQK ? dfb::qn_out : dfb::q;
    constexpr uint32_t k_in = NormQK ? dfb::kn_out : dfb::k;
    constexpr uint32_t kt_in = NormQK ? dfb::kn_out_t : dfb::k_trans;
    if constexpr (NormQK) {
        wait(dfb::scaler, 1);
        wait(dfb::eps, 1);
    }
    for (uint32_t head = 0; head < head_count; ++head) {
        wait(dfb::q, Kt);
        wait(dfb::k, Kt);
        if constexpr (NormQK) {
            l2_normalize<Kt, dfb::qn_sq, dfb::qn_stats, dfb::qn_inv>(dfb::q, dfb::qn_out, QScaleBits);
            wait(dfb::qn_out, Kt);
            pop(dfb::q, Kt);
            l2_normalize<Kt, dfb::kn_sq, dfb::kn_stats, dfb::kn_inv>(dfb::k, dfb::kn_out, 0);
            wait(dfb::kn_out, Kt);
            copy_tiles(dfb::kn_out, dfb::kn_out_t, Kt);
            wait(dfb::kn_out_t, Kt);
            pop(dfb::k, Kt);
        } else {
            wait(dfb::k_trans, Kt);
        }
        wait(dfb::v, Vt);
        wait(dfb::beta, 1);
        wait(dfb::g, 1);
        wait(dfb::state, kv);

        exp_one(dfb::g, dfb::decay);
        wait(dfb::decay, 1);
        pop(dfb::g, 1);
        scalar_mul(dfb::state, dfb::decay, dfb::s_decay, kv);
        wait(dfb::s_decay, kv);
        pop(dfb::state, kv);
        pop(dfb::decay, 1);

        mm(k_in, dfb::s_decay, dfb::memory, 1, Kt, Vt);
        wait(dfb::memory, Vt);
        pop(k_in, Kt);
        ew(dfb::v, dfb::memory, dfb::delta, Vt, true);
        wait(dfb::delta, Vt);
        pop(dfb::v, Vt);
        pop(dfb::memory, Vt);

        // Match the model's reference ordering: beta scales the V-vector
        // before the outer product.  Besides reducing rounding drift, this
        // scales Vt tiles instead of Kt*Vt tiles.
        scalar_mul(dfb::delta, dfb::beta, dfb::scaled, Vt);
        wait(dfb::scaled, Vt);
        pop(dfb::delta, Vt);
        pop(dfb::beta, 1);

        transpose_tiles(kt_in, dfb::k_col, Kt);
        wait(dfb::k_col, Kt);
        pop(kt_in, Kt);
        mm(dfb::k_col, dfb::scaled, dfb::outer, Kt, 1, Vt);
        wait(dfb::outer, kv);
        pop(dfb::k_col, Kt);
        pop(dfb::scaled, Vt);

        ew(dfb::s_decay, dfb::outer, dfb::s_new, kv, false);
        wait(dfb::s_new, kv);
        pop(dfb::s_decay, kv);
        pop(dfb::outer, kv);
        mm(q_in, dfb::s_new, dfb::core_out, 1, Kt, Vt);
        wait(dfb::core_out, Vt);
        pop(q_in, Kt);
        copy_tiles(dfb::s_new, dfb::state_out, kv);
        pop(dfb::s_new, kv);
    }
}

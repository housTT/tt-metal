// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
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

void mul_ew(uint32_t a, uint32_t b, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    mul_init(a, b, false);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        mul_tiles(a, b, tile, tile, 0);
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

// out = silu(t0*w0 + t1*w1 + t2*w2 + x*w3) for `count` tiles, then pop the FIR inputs.
void fir_silu(uint32_t out, uint32_t count) {
    wait(dfb::fir_t0, count);
    wait(dfb::fir_w0, count);
    wait(dfb::fir_t1, count);
    wait(dfb::fir_w1, count);
    wait(dfb::fir_t2, count);
    wait(dfb::fir_w2, count);
    wait(dfb::fir_x, count);
    wait(dfb::fir_w3, count);
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        reconfig_data_format(dfb::fir_t0, dfb::fir_w0);
        mul_init(dfb::fir_t0, dfb::fir_w0, false);
        mul_tiles(dfb::fir_t0, dfb::fir_w0, tile, tile, 0);
        reconfig_data_format(dfb::fir_t1, dfb::fir_w1);
        mul_init(dfb::fir_t1, dfb::fir_w1, true);
        mul_tiles(dfb::fir_t1, dfb::fir_w1, tile, tile, 0);
        reconfig_data_format(dfb::fir_t2, dfb::fir_w2);
        mul_init(dfb::fir_t2, dfb::fir_w2, true);
        mul_tiles(dfb::fir_t2, dfb::fir_w2, tile, tile, 0);
        reconfig_data_format(dfb::fir_x, dfb::fir_w3);
        mul_init(dfb::fir_x, dfb::fir_w3, true);
        mul_tiles(dfb::fir_x, dfb::fir_w3, tile, tile, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, tile);
        tile_regs_release();
    }
    cb_push_back(out, count);
    pop(dfb::fir_t0, count);
    pop(dfb::fir_w0, count);
    pop(dfb::fir_t1, count);
    pop(dfb::fir_w1, count);
    pop(dfb::fir_t2, count);
    pop(dfb::fir_w2, count);
    pop(dfb::fir_x, count);
    pop(dfb::fir_w3, count);
}

void square_tiles(uint32_t input, uint32_t out, uint32_t count) { mul_ew(input, input, out, count); }

// inv = (stats + eps)^-1/2 [* scale]; column 0 holds the row values.
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

void scale_by_weight_row(uint32_t input, uint32_t weight, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format(input, weight);
    mul_bcast_rows_init(input, weight);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(input, weight, tile, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, tile);
        tile_regs_release();
    }
    cb_push_back(out, count);
}

void sigmoid_tiles(uint32_t input, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(input);
    copy_tile_to_dst_init_short(input);
    sigmoid_tile_init();
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        copy_tile(input, tile, 0);
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, tile);
        tile_regs_release();
    }
    cb_push_back(out, count);
}

// L2-normalize the valid row of `input` (n tiles): out = x / sqrt(sum x^2 + eps) [* scale].
// The row mask zeroes the tile's padding rows so they cannot leak into the
// rank-one state update or the output.
template <uint32_t N, uint32_t sq, uint32_t stats, uint32_t inv, uint32_t invm>
void l2_normalize(uint32_t input, uint32_t out, uint32_t scale_bits) {
    square_tiles(input, sq, N);
    wait(sq, N);
    compute_kernel_lib::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, sq, dfb::scaler_sum, stats>(
        compute_kernel_lib::ReduceInputBlockShape::of(1, N));
    wait(stats, 1);
    inverse_norm(stats, dfb::eps_qk, inv, scale_bits);
    wait(inv, 1);
    pop(stats, 1);
    mul_ew(inv, dfb::mask, invm, 1);
    wait(invm, 1);
    pop(inv, 1);
    scale_rows(input, invm, out, N);
    pop(invm, 1);
}

}  // namespace

template <uint32_t Kt, uint32_t Vt, uint32_t QScaleBits>
TT_KERNEL void compute(uint32_t head_count) {
    constexpr uint32_t kv = Kt * Vt;
    compute_kernel_hw_startup(dfb::fir_t0, dfb::fir_w0, dfb::out);
    wait(dfb::scaler_sum, 1);
    wait(dfb::scaler_avg, 1);
    wait(dfb::eps_qk, 1);
    wait(dfb::eps_norm, 1);
    wait(dfb::mask, 1);
    wait(dfb::norm_w, Vt);
    for (uint32_t head = 0; head < head_count; ++head) {
        // 1. causal conv + SiLU per channel segment.
        fir_silu(dfb::q_raw, Kt);
        fir_silu(dfb::k_raw, Kt);
        fir_silu(dfb::v, Vt);
        // 2. q / k L2 norm (q scaled by 1/sqrt(K)).
        wait(dfb::q_raw, Kt);
        l2_normalize<Kt, dfb::qn_sq, dfb::qn_stats, dfb::qn_inv, dfb::qn_invm>(dfb::q_raw, dfb::qn_out, QScaleBits);
        wait(dfb::qn_out, Kt);
        pop(dfb::q_raw, Kt);
        wait(dfb::k_raw, Kt);
        l2_normalize<Kt, dfb::kn_sq, dfb::kn_stats, dfb::kn_inv, dfb::kn_invm>(dfb::k_raw, dfb::kn_out, 0);
        wait(dfb::kn_out, Kt);
        copy_tiles(dfb::kn_out, dfb::kn_out_t, Kt);
        wait(dfb::kn_out_t, Kt);
        pop(dfb::k_raw, Kt);
        // 3. gated delta-rule recurrence.
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
        mm(dfb::kn_out, dfb::s_decay, dfb::memory, 1, Kt, Vt);
        wait(dfb::memory, Vt);
        pop(dfb::kn_out, Kt);
        ew(dfb::v, dfb::memory, dfb::delta, Vt, true);
        wait(dfb::delta, Vt);
        pop(dfb::v, Vt);
        pop(dfb::memory, Vt);
        scalar_mul(dfb::delta, dfb::beta, dfb::scaled, Vt);
        wait(dfb::scaled, Vt);
        pop(dfb::delta, Vt);
        pop(dfb::beta, 1);
        transpose_tiles(dfb::kn_out_t, dfb::k_col, Kt);
        wait(dfb::k_col, Kt);
        pop(dfb::kn_out_t, Kt);
        mm(dfb::k_col, dfb::scaled, dfb::outer, Kt, 1, Vt);
        wait(dfb::outer, kv);
        pop(dfb::k_col, Kt);
        pop(dfb::scaled, Vt);
        ew(dfb::s_decay, dfb::outer, dfb::s_new, kv, false);
        wait(dfb::s_new, kv);
        pop(dfb::s_decay, kv);
        pop(dfb::outer, kv);
        mm(dfb::qn_out, dfb::s_new, dfb::core, 1, Kt, Vt);
        wait(dfb::core, Vt);
        pop(dfb::qn_out, Kt);
        copy_tiles(dfb::s_new, dfb::state_out, kv);
        pop(dfb::s_new, kv);
        // 4. epilogue: rmsnorm(o) * weight * sigmoid(gate).
        square_tiles(dfb::core, dfb::ep_sq, Vt);
        wait(dfb::ep_sq, Vt);
        compute_kernel_lib::
            reduce<ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW, dfb::ep_sq, dfb::scaler_avg, dfb::ep_stats>(
                compute_kernel_lib::ReduceInputBlockShape::of(1, Vt));
        wait(dfb::ep_stats, 1);
        inverse_norm(dfb::ep_stats, dfb::eps_norm, dfb::ep_inv, 0);
        wait(dfb::ep_inv, 1);
        pop(dfb::ep_stats, 1);
        scale_rows(dfb::core, dfb::ep_inv, dfb::ep_norm, Vt);
        wait(dfb::ep_norm, Vt);
        pop(dfb::core, Vt);
        pop(dfb::ep_inv, 1);
        scale_by_weight_row(dfb::ep_norm, dfb::norm_w, dfb::ep_tmp, Vt);
        wait(dfb::ep_tmp, Vt);
        pop(dfb::ep_norm, Vt);
        wait(dfb::gate, Vt);
        sigmoid_tiles(dfb::gate, dfb::gate_act, Vt);
        wait(dfb::gate_act, Vt);
        pop(dfb::gate, Vt);
        mul_ew(dfb::ep_tmp, dfb::gate_act, dfb::out, Vt);
        pop(dfb::ep_tmp, Vt);
        pop(dfb::gate_act, Vt);
    }
}

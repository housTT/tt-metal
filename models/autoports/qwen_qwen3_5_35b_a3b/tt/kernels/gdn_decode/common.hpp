// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"

constexpr uint32_t CB_QC = 0;
constexpr uint32_t CB_KC = 1;
constexpr uint32_t CB_V = 2;
constexpr uint32_t CB_Z = 3;
constexpr uint32_t CB_DECAY = 4;
constexpr uint32_t CB_BETA = 5;
constexpr uint32_t CB_S = 6;
constexpr uint32_t CB_W = 7;
constexpr uint32_t CB_QN = 8;
constexpr uint32_t CB_KN = 9;
constexpr uint32_t CB_KD = 10;
constexpr uint32_t CB_SC = 11;
constexpr uint32_t CB_DEL = 12;
constexpr uint32_t CB_O = 13;
constexpr uint32_t CB_DEC = 14;
constexpr uint32_t CB_OP = 15;
constexpr uint32_t CB_SOUT = 16;
constexpr uint32_t CB_OUT = 17;

FORCE_INLINE void load(uint32_t cb, uint32_t index, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, index, dst);
}

template <BroadcastType type>
FORCE_INLINE void bcast(uint32_t cb, uint32_t index, uint32_t dst) {
    reconfig_data_format_srca(cb);
    unary_bcast_init<type>(cb);
    unary_bcast<type>(cb, index, dst);
    unary_bcast_uninit<type>(cb);
}

FORCE_INLINE void mul(uint32_t a, uint32_t b, uint32_t out) {
    mul_binary_tile_init();
    mul_binary_tile(a, b, out);
}

FORCE_INLINE void add(uint32_t a, uint32_t b, uint32_t out) {
    add_binary_tile_init();
    add_binary_tile(a, b, out);
}

FORCE_INLINE void sub(uint32_t a, uint32_t b, uint32_t out) {
    sub_binary_tile_init();
    sub_binary_tile(a, b, out);
}

FORCE_INLINE void column_sum(uint32_t dst) {
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(dst);
}

FORCE_INLINE void row_sum(uint32_t dst) {
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(dst);
}

FORCE_INLINE void begin() { tile_regs_acquire(); }

FORCE_INLINE void finish() {
    tile_regs_commit();
    tile_regs_wait();
}

FORCE_INLINE void pack_to(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}

FORCE_INLINE void release() { tile_regs_release(); }

template <uint32_t inv_width_bits, uint32_t norm_eps_bits>
FORCE_INLINE void norm_gate() {
    cb_wait_front(CB_O, 4);
    begin();
    load(CB_O, 0, 0);
    mul(0, 0, 0);
    for (uint32_t c = 1; c < 4; ++c) {
        load(CB_O, c, 1);
        mul(1, 1, 1);
        add(0, 1, 0);
    }
    row_sum(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_width_bits);
    add_unary_tile(0, norm_eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    finish();
    pack_to(0, CB_SC);
    release();
    cb_wait_front(CB_SC, 1);
    cb_wait_front(CB_W, 4);
    cb_wait_front(CB_Z, 4);
    for (uint32_t c = 0; c < 4; ++c) {
        begin();
        load(CB_O, c, 0);
        bcast<BroadcastType::SCALAR>(CB_SC, 0, 1);
        mul(0, 1, 0);
        bcast<BroadcastType::ROW>(CB_W, c, 1);
        mul(0, 1, 0);
        load(CB_Z, c, 1);
        silu_tile_init();
        silu_tile(1);
        mul(0, 1, 0);
        finish();
        pack_to(0, CB_OUT);
        release();
    }
    cb_pop_front(CB_SC, 1);
    cb_pop_front(CB_W, 4);
    cb_pop_front(CB_Z, 4);
    cb_pop_front(CB_O, 4);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common.hpp"

template <uint32_t eps_bits, uint32_t scale_bits, bool scale>
FORCE_INLINE void inverse_norm(uint32_t cb_in, uint32_t cb_out) {
    cb_wait_front(cb_in, 4);
    begin();
    load(cb_in, 0, 0);
    mul(0, 0, 0);
    for (uint32_t r = 1; r < 4; ++r) {
        load(cb_in, r, 1);
        mul(1, 1, 1);
        add(0, 1, 0);
    }
    column_sum(0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    if constexpr (scale) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale_bits);
    }
    finish();
    pack_to(0, cb_out);
    release();
}

void kernel_main() {
    constexpr uint32_t l2_eps_bits = get_compile_time_arg_val(0);
    constexpr uint32_t q_scale_bits = get_compile_time_arg_val(1);
    constexpr uint32_t inv_width_bits = get_compile_time_arg_val(2);
    constexpr uint32_t norm_eps_bits = get_compile_time_arg_val(3);
    constexpr uint32_t blocks = get_compile_time_arg_val(4);
    constexpr uint32_t gates = get_compile_time_arg_val(5);

    compute_kernel_hw_startup(CB_S, CB_SOUT);

    inverse_norm<l2_eps_bits, q_scale_bits, true>(CB_QC, CB_QN);
    inverse_norm<l2_eps_bits, 0, false>(CB_KC, CB_KN);
    cb_wait_front(CB_QN, 1);
    cb_wait_front(CB_KN, 1);

    constexpr uint32_t decay_tiles = gates ? 3 : 1;
    cb_wait_front(CB_DECAY, decay_tiles);
    cb_wait_front(CB_BETA, 1);
    begin();
    load(CB_DECAY, 0, 0);
    if constexpr (gates) {
        load(CB_DECAY, 1, 1);
        add(0, 1, 0);
        softplus_tile_init();
        softplus_tile(0, 0x3f800000u, 0x3f800000u, 0x41a00000u);
        load(CB_DECAY, 2, 1);
        mul(0, 1, 0);
    }
    exp_tile_init();
    exp_tile(0);
    bcast<BroadcastType::SCALAR>(CB_KN, 0, 1);
    mul(0, 1, 2);
    load(CB_BETA, 0, 3);
    if constexpr (gates) {
        sigmoid_tile_init();
        sigmoid_tile(3);
    }
    mul(3, 1, 3);
    finish();
    pack_to(0, CB_DEC);
    pack_to(2, CB_KD);
    pack_to(3, CB_SC);
    release();
    cb_pop_front(CB_DECAY, decay_tiles);
    cb_pop_front(CB_BETA, 1);
    cb_pop_front(CB_KN, 1);
    cb_wait_front(CB_DEC, 1);
    cb_wait_front(CB_KD, 1);
    cb_wait_front(CB_SC, 1);
    cb_wait_front(CB_S, blocks * 4);
    cb_wait_front(CB_V, blocks);

    for (uint32_t c = 0; c < blocks; ++c) {
        begin();
        bcast<BroadcastType::COL>(CB_KC, 0, 1);
        load(CB_S, c * 4, 0);
        mul(0, 1, 0);
        for (uint32_t r = 1; r < 4; ++r) {
            bcast<BroadcastType::COL>(CB_KC, r, 1);
            load(CB_S, c * 4 + r, 2);
            mul(2, 1, 2);
            add(0, 2, 0);
        }
        column_sum(0);
        bcast<BroadcastType::SCALAR>(CB_KD, 0, 1);
        mul(0, 1, 0);
        load(CB_V, c, 1);
        sub(1, 0, 1);
        bcast<BroadcastType::SCALAR>(CB_SC, 0, 2);
        mul(1, 2, 1);
        finish();
        pack_to(1, CB_DEL);
        release();
        cb_wait_front(CB_DEL, 1);

        for (uint32_t r = 0; r < 4; ++r) {
            begin();
            bcast<BroadcastType::ROW>(CB_DEL, 0, 3);
            bcast<BroadcastType::COL>(CB_KC, r, 1);
            mul(1, 3, 1);
            bcast<BroadcastType::SCALAR>(CB_DEC, 0, 2);
            load(CB_S, c * 4 + r, 0);
            mul(0, 2, 0);
            add(0, 1, 0);
            bcast<BroadcastType::COL>(CB_QC, r, 1);
            mul(0, 1, 1);
            finish();
            pack_to(0, CB_SOUT);
            pack_to(1, CB_OP);
            release();
        }
        cb_pop_front(CB_DEL, 1);

        cb_wait_front(CB_OP, 4);
        begin();
        load(CB_OP, 0, 0);
        for (uint32_t r = 1; r < 4; ++r) {
            load(CB_OP, r, 1);
            add(0, 1, 0);
        }
        column_sum(0);
        bcast<BroadcastType::SCALAR>(CB_QN, 0, 1);
        mul(0, 1, 0);
        finish();
        pack_to(0, CB_O);
        release();
        cb_pop_front(CB_OP, 4);
    }
    cb_pop_front(CB_S, blocks * 4);
    cb_pop_front(CB_V, blocks);
    cb_pop_front(CB_KC, 4);
    cb_pop_front(CB_QC, 4);
    cb_pop_front(CB_QN, 1);
    cb_pop_front(CB_KD, 1);
    cb_pop_front(CB_SC, 1);

    if constexpr (blocks == 4) {
        norm_gate<inv_width_bits, norm_eps_bits>();
    }
    cb_pop_front(CB_DEC, 1);
}

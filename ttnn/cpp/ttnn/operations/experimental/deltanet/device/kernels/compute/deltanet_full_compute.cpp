// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for one DeltaNet decode recurrence. Inputs and persisted
// state are bf16; fp32_dest_acc_en keeps intermediate DST math in fp32.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

constexpr uint32_t cb_state_in = get_compile_time_arg_val(0);
constexpr uint32_t cb_q = get_compile_time_arg_val(1);
constexpr uint32_t cb_k = get_compile_time_arg_val(2);
constexpr uint32_t cb_v = get_compile_time_arg_val(3);
constexpr uint32_t cb_decay = get_compile_time_arg_val(4);
constexpr uint32_t cb_beta = get_compile_time_arg_val(5);
constexpr uint32_t cb_output = get_compile_time_arg_val(6);
constexpr uint32_t cb_state_out = get_compile_time_arg_val(7);
constexpr uint32_t cb_tmp0 = get_compile_time_arg_val(8);
constexpr uint32_t cb_tmp1 = get_compile_time_arg_val(9);
constexpr uint32_t cb_acc = get_compile_time_arg_val(10);
constexpr uint32_t Dk_tiles = get_compile_time_arg_val(11);
constexpr uint32_t Dv_tiles = get_compile_time_arg_val(12);
constexpr uint32_t cb_state_mid = get_compile_time_arg_val(13);
constexpr uint32_t cb_k_T = get_compile_time_arg_val(14);
constexpr uint32_t cb_raw_out = get_compile_time_arg_val(15);
constexpr bool fused_epilogue = get_compile_time_arg_val(16) == 1;
constexpr uint32_t cb_gate = get_compile_time_arg_val(17);
constexpr uint32_t cb_norm_weight = get_compile_time_arg_val(18);
constexpr uint32_t cb_norm_scaler = get_compile_time_arg_val(19);
constexpr uint32_t cb_norm_epsilon = get_compile_time_arg_val(20);
constexpr uint32_t cb_norm_tmp = get_compile_time_arg_val(21);
constexpr uint32_t cb_norm_stats = get_compile_time_arg_val(22);
constexpr uint32_t cb_norm_inv = get_compile_time_arg_val(23);
constexpr uint32_t cb_norm = get_compile_time_arg_val(24);

constexpr uint32_t state_tiles = Dk_tiles * Dv_tiles;

inline void matmul_reconfig_and_init(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb) {
    reconfig_data_format<SrcOrder::Reverse>(in0_cb, in1_cb);
    matmul_init(in0_cb, in1_cb);
    pack_reconfig_data_format(out_cb);
}

inline void binary_reconfig(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb) {
    reconfig_data_format(in0_cb, in1_cb);
    pack_reconfig_data_format(out_cb);
}

inline void square_raw_output(DataflowBuffer& tmp) {
    tmp.reserve_back(Dv_tiles);
    pack_reconfig_data_format(cb_norm_tmp);
    reconfig_data_format(cb_raw_out, cb_raw_out);
    mul_init(cb_raw_out, cb_raw_out, false);
    for (uint32_t tile = 0; tile < Dv_tiles; ++tile) {
        tile_regs_acquire();
        mul_tiles(cb_raw_out, cb_raw_out, tile, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_norm_tmp, tile);
        tile_regs_release();
    }
    tmp.push_back(Dv_tiles);
}

inline void calculate_inverse_rms(DataflowBuffer& inv) {
    inv.reserve_back(1);
    pack_reconfig_data_format(cb_norm_inv);
    reconfig_data_format(cb_norm_stats, cb_norm_epsilon);
    add_init(cb_norm_stats, cb_norm_epsilon);
    tile_regs_acquire();
    add_tiles(cb_norm_stats, cb_norm_epsilon, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_norm_inv, 0);
    tile_regs_release();
    inv.push_back(1);
}

inline void scale_by_inverse_rms(DataflowBuffer& norm) {
    norm.reserve_back(Dv_tiles);
    pack_reconfig_data_format(cb_norm);
    reconfig_data_format(cb_raw_out, cb_norm_inv);
    mul_bcast_cols_init(cb_raw_out, cb_norm_inv);
    for (uint32_t tile = 0; tile < Dv_tiles; ++tile) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_raw_out, cb_norm_inv, tile, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_norm, tile);
        tile_regs_release();
    }
    norm.push_back(Dv_tiles);
}

inline void apply_norm_weight(DataflowBuffer& tmp) {
    tmp.reserve_back(Dv_tiles);
    pack_reconfig_data_format(cb_norm_tmp);
    reconfig_data_format(cb_norm, cb_norm_weight);
    mul_bcast_rows_init(cb_norm, cb_norm_weight);
    for (uint32_t tile = 0; tile < Dv_tiles; ++tile) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_norm, cb_norm_weight, tile, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_norm_tmp, tile);
        tile_regs_release();
    }
    tmp.push_back(Dv_tiles);
}

inline void activate_silu_gate(DataflowBuffer& norm) {
    norm.reserve_back(Dv_tiles);
    pack_reconfig_data_format(cb_norm);
    reconfig_data_format_srca(cb_gate);
    copy_tile_to_dst_init_short(cb_gate);
    silu_tile_init();
    for (uint32_t tile = 0; tile < Dv_tiles; ++tile) {
        tile_regs_acquire();
        copy_tile(cb_gate, tile, 0);
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_norm, tile);
        tile_regs_release();
    }
    norm.push_back(Dv_tiles);
}

inline void multiply_norm_and_gate(DataflowBuffer& output) {
    output.reserve_back(Dv_tiles);
    pack_reconfig_data_format(cb_output);
    reconfig_data_format(cb_norm_tmp, cb_norm);
    mul_init(cb_norm_tmp, cb_norm);
    for (uint32_t tile = 0; tile < Dv_tiles; ++tile) {
        tile_regs_acquire();
        mul_tiles(cb_norm_tmp, cb_norm, tile, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_output, tile);
        tile_regs_release();
    }
    output.push_back(Dv_tiles);
}

void kernel_main() {
    // Configure the compute engines once; per-op init below only switches the
    // source ordering and output pack format as the fused pipeline advances.
    compute_kernel_hw_startup(cb_state_in, cb_decay, cb_state_mid);

    // Step 1: S_mid = S * decay (broadcast scalar multiply)
    {
        mul_bcast_scalar_init(cb_state_in, cb_decay);
        cb_wait_front(cb_state_in, state_tiles);
        cb_wait_front(cb_decay, 1);
        cb_reserve_back(cb_state_mid, state_tiles);
        for (uint32_t t = 0; t < state_tiles; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_state_in, cb_decay, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state_mid);
            tile_regs_release();
        }
        cb_push_back(cb_state_mid, state_tiles);
        cb_pop_front(cb_state_in, state_tiles);
    }

    // Step 2: mem = k @ S_mid → cb_tmp0 [Dv_tiles]
    {
        matmul_reconfig_and_init(cb_k, cb_state_mid, cb_tmp0);
        cb_wait_front(cb_state_mid, state_tiles);
        cb_wait_front(cb_k, Dk_tiles);
        cb_reserve_back(cb_tmp0, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < Dk_tiles; i++) {
                matmul_tiles(cb_k, cb_state_mid, i, i * Dv_tiles + j, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp0);
            tile_regs_release();
        }
        cb_push_back(cb_tmp0, Dv_tiles);
    }

    // Step 3: delta = (v - mem) * beta
    {
        // v - mem → cb_tmp1
        binary_reconfig(cb_v, cb_tmp0, cb_tmp1);
        sub_init(cb_v, cb_tmp0);
        cb_wait_front(cb_v, Dv_tiles);
        cb_wait_front(cb_tmp0, Dv_tiles);
        cb_reserve_back(cb_tmp1, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire();
            sub_tiles(cb_v, cb_tmp0, j, j, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp1);
            tile_regs_release();
        }
        cb_push_back(cb_tmp1, Dv_tiles);
        cb_pop_front(cb_tmp0, Dv_tiles);

        // (v - mem) * beta → cb_acc
        binary_reconfig(cb_tmp1, cb_beta, cb_acc);
        mul_bcast_scalar_init(cb_tmp1, cb_beta);
        cb_wait_front(cb_tmp1, Dv_tiles);
        cb_wait_front(cb_beta, 1);
        cb_reserve_back(cb_acc, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_tmp1, cb_beta, j, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_acc);
            tile_regs_release();
        }
        cb_push_back(cb_acc, Dv_tiles);
        cb_pop_front(cb_tmp1, Dv_tiles);
    }

    // Step 4: S_new = S_mid + outer(k_T, delta)
    {
        cb_wait_front(cb_state_mid, state_tiles);
        cb_wait_front(cb_k_T, Dk_tiles);
        cb_wait_front(cb_acc, Dv_tiles);
        cb_reserve_back(cb_state_out, state_tiles);

        for (uint32_t i = 0; i < Dk_tiles; i++) {
            for (uint32_t j = 0; j < Dv_tiles; j++) {
                uint32_t state_tile_idx = i * Dv_tiles + j;

                // outer product: k_T @ delta → cb_tmp1 (one tile)
                cb_reserve_back(cb_tmp1, 1);
                matmul_reconfig_and_init(cb_k_T, cb_acc, cb_tmp1);
                tile_regs_acquire();
                matmul_tiles(cb_k_T, cb_acc, i, j, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp1);
                tile_regs_release();
                cb_push_back(cb_tmp1, 1);

                // S_mid + outer_tile → S_out
                cb_wait_front(cb_tmp1, 1);
                binary_reconfig(cb_state_mid, cb_tmp1, cb_state_out);
                add_init(cb_state_mid, cb_tmp1);
                tile_regs_acquire();
                add_tiles(cb_state_mid, cb_tmp1, state_tile_idx, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_state_out);
                tile_regs_release();
                cb_pop_front(cb_tmp1, 1);
            }
        }

        cb_push_back(cb_state_out, state_tiles);
        cb_pop_front(cb_state_mid, state_tiles);
        cb_pop_front(cb_k_T, Dk_tiles);
        cb_pop_front(cb_acc, Dv_tiles);
    }

    // Step 5: raw_out = q @ S_new → cb_raw_out [Dv_tiles]
    {
        matmul_reconfig_and_init(cb_q, cb_state_out, cb_raw_out);
        cb_wait_front(cb_state_out, state_tiles);
        cb_wait_front(cb_q, Dk_tiles);
        cb_reserve_back(cb_raw_out, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < Dk_tiles; i++) {
                matmul_tiles(cb_q, cb_state_out, i, i * Dv_tiles + j, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_raw_out);
            tile_regs_release();
        }
        cb_push_back(cb_raw_out, Dv_tiles);
    }

    // Pop consumed input CBs
    cb_pop_front(cb_q, Dk_tiles);
    cb_pop_front(cb_k, Dk_tiles);
    cb_pop_front(cb_v, Dv_tiles);
    cb_pop_front(cb_decay, 1);
    cb_pop_front(cb_beta, 1);

    if constexpr (fused_epilogue) {
        cb_wait_front(cb_raw_out, Dv_tiles);
        cb_wait_front(cb_gate, Dv_tiles);
        cb_wait_front(cb_norm_weight, Dv_tiles);
        cb_wait_front(cb_norm_scaler, 1);
        cb_wait_front(cb_norm_epsilon, 1);

        DataflowBuffer tmp(cb_norm_tmp);
        DataflowBuffer stats(cb_norm_stats);
        DataflowBuffer inv(cb_norm_inv);
        DataflowBuffer norm(cb_norm);
        DataflowBuffer output(cb_output);

        square_raw_output(tmp);
        compute_kernel_lib::
            reduce<ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW, cb_norm_tmp, cb_norm_scaler, cb_norm_stats>(
                compute_kernel_lib::ReduceInputBlockShape::of(1, Dv_tiles));
        stats.wait_front(1);
        calculate_inverse_rms(inv);
        inv.wait_front(1);
        scale_by_inverse_rms(norm);
        norm.wait_front(Dv_tiles);
        cb_pop_front(cb_raw_out, Dv_tiles);
        inv.pop_front(1);
        stats.pop_front(1);
        apply_norm_weight(tmp);
        tmp.wait_front(Dv_tiles);
        norm.pop_front(Dv_tiles);
        activate_silu_gate(norm);
        norm.wait_front(Dv_tiles);
        cb_pop_front(cb_gate, Dv_tiles);
        multiply_norm_and_gate(output);
        tmp.pop_front(Dv_tiles);
        norm.pop_front(Dv_tiles);
    } else {
        // Preserve the raw-output contract when the optional epilogue tensors are absent.
        cb_wait_front(cb_raw_out, Dv_tiles);
        cb_reserve_back(cb_output, Dv_tiles);
        copy_tile_to_dst_init_short(cb_raw_out);
        for (uint32_t t = 0; t < Dv_tiles; t++) {
            tile_regs_acquire();
            copy_tile(cb_raw_out, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);
            tile_regs_release();
        }
        cb_push_back(cb_output, Dv_tiles);
        cb_pop_front(cb_raw_out, Dv_tiles);
    }
}

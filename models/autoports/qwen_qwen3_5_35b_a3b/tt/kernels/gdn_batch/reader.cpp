// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_QC = 0;
constexpr uint32_t CB_KC = 1;
constexpr uint32_t CB_V = 2;
constexpr uint32_t CB_DECAY = 4;
constexpr uint32_t CB_BETA = 5;
constexpr uint32_t CB_S = 6;
constexpr uint32_t CB_SCRATCH = 18;
constexpr uint32_t CB_INPUTS = 20;
constexpr uint32_t BF16_TILE = 2048;

FORCE_INLINE uint32_t at(uint32_t row, uint32_t col) {
    return ((row >= 16) ? 512 : 0) + ((col >= 16) ? 256 : 0) + (row % 16) * 16 + (col % 16);
}

FORCE_INLINE uint32_t bf16_at(uint32_t tile, uint32_t row, uint32_t col) {
    return static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile)[at(row, col)]) << 16;
}

FORCE_INLINE uint32_t f32_at(uint32_t tile, uint32_t row, uint32_t col) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile)[at(row, col)];
}

FORCE_INLINE void zero_f32_tile(uint32_t tile) {
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
    for (uint32_t i = 0; i < 1024; ++i) {
        t[i] = 0;
    }
}

template <uint32_t cb_id>
FORCE_INLINE void push_columns(uint32_t src, uint32_t user, uint32_t n) {
    constexpr uint32_t dst_bytes = get_tile_size(cb_id);
    CircularBuffer cb(cb_id);
    cb.reserve_back(n);
    uint32_t dst = cb.get_write_ptr();
    for (uint32_t i = 0; i < n; ++i) {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + i * dst_bytes);
        for (uint32_t r = 0; r < 32; ++r) {
            t[at(r, 0)] = bf16_at(src + i * BF16_TILE, user, r);
        }
    }
    cb.push_back(n);
}

FORCE_INLINE void push_row(uint32_t src, uint32_t user) {
    CircularBuffer cb(CB_V);
    cb.reserve_back(1);
    volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_write_ptr());
    volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src);
    for (uint32_t c = 0; c < 32; ++c) {
        t[at(0, c)] = s[at(user, c)];
    }
    cb.push_back(1);
}

template <uint32_t cb_id>
FORCE_INLINE void push_scalars(const uint32_t* values, uint32_t n) {
    constexpr uint32_t bytes = get_tile_size(cb_id);
    CircularBuffer cb(cb_id);
    cb.reserve_back(n);
    uint32_t dst = cb.get_write_ptr();
    for (uint32_t i = 0; i < n; ++i) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + i * bytes)[0] = values[i];
    }
    cb.push_back(n);
}

void kernel_main() {
    constexpr uint32_t groups = get_compile_time_arg_val(0);
    constexpr uint32_t q_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(2);
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto m_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr auto n_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<n_args.next_compile_time_args_offset()>();

    const auto qkv = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto mega = TensorAccessor(m_args, get_arg_val<uint32_t>(1));
    const auto dt_bias = TensorAccessor(d_args, get_arg_val<uint32_t>(2));
    const auto negative = TensorAccessor(n_args, get_arg_val<uint32_t>(3));
    const uint32_t head = get_arg_val<uint32_t>(4);
    const uint32_t block = get_arg_val<uint32_t>(5);
    const uint32_t a_column = get_arg_val<uint32_t>(6);
    const uint32_t b_column = get_arg_val<uint32_t>(7);
    const uint32_t first_user = get_arg_val<uint32_t>(8);
    const uint32_t users = get_arg_val<uint32_t>(9);
    constexpr uint32_t state_arg = 10;
    const uint32_t key_head = head / groups;
    Noc noc;

    CircularBuffer inputs(CB_INPUTS);
    const uint32_t q_src = inputs.get_write_ptr();
    const uint32_t k_src = q_src + 4 * BF16_TILE;
    const uint32_t v_src = k_src + 4 * BF16_TILE;
    const uint32_t a_src = v_src + BF16_TILE;
    const uint32_t b_src = a_src + BF16_TILE;
    CircularBuffer scratch(CB_SCRATCH);
    const uint32_t d_src = scratch.get_write_ptr();
    const uint32_t n_src = d_src + 4096;
    for (uint32_t i = 0; i < 4; ++i) {
        noc.async_read(qkv, CoreLocalMem<uint32_t>(q_src + i * BF16_TILE), BF16_TILE, {.page_id = key_head * 4 + i}, {});
        noc.async_read(qkv, CoreLocalMem<uint32_t>(k_src + i * BF16_TILE), BF16_TILE,
                       {.page_id = q_tiles + key_head * 4 + i}, {});
    }
    noc.async_read(qkv, CoreLocalMem<uint32_t>(v_src), BF16_TILE, {.page_id = q_tiles + k_tiles + head * 4 + block}, {});
    noc.async_read(mega, CoreLocalMem<uint32_t>(a_src), BF16_TILE, {.page_id = a_column / 32}, {});
    noc.async_read(mega, CoreLocalMem<uint32_t>(b_src), BF16_TILE, {.page_id = b_column / 32}, {});
    noc.async_read(dt_bias, CoreLocalMem<uint32_t>(d_src), 4096, {.page_id = 0}, {});
    noc.async_read(negative, CoreLocalMem<uint32_t>(n_src), 4096, {.page_id = 0}, {});
    noc.async_read_barrier();
    const uint32_t bias = f32_at(d_src, 0, head);
    const uint32_t neg = f32_at(n_src, 0, head);

    for (uint32_t u = first_user; u < first_user + users; ++u) {
        push_columns<CB_QC>(q_src, u, 4);
        push_columns<CB_KC>(k_src, u, 4);
        const uint32_t decay_inputs[3] = {bf16_at(a_src, u, a_column % 32), bias, neg};
        push_scalars<CB_DECAY>(decay_inputs, 3);
        const uint32_t beta_input = bf16_at(b_src, u, b_column % 32);
        push_scalars<CB_BETA>(&beta_input, 1);
        const auto state = TensorAccessor(s_args, get_arg_val<uint32_t>(state_arg + u - first_user));
        CircularBuffer s(CB_S);
        constexpr uint32_t s_bytes = get_tile_size(CB_S);
        s.reserve_back(4);
        uint32_t dst = s.get_write_ptr();
        for (uint32_t row = 0; row < 4; ++row) {
            noc.async_read(state, CoreLocalMem<uint32_t>(dst + row * s_bytes), s_bytes,
                           {.page_id = head * 16 + row * 4 + block}, {});
        }
        noc.async_read_barrier();
        s.push_back(4);
        push_row(v_src, u);
    }
}

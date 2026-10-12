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
constexpr uint32_t CB_Z = 3;
constexpr uint32_t CB_DECAY = 4;
constexpr uint32_t CB_BETA = 5;
constexpr uint32_t CB_S = 6;
constexpr uint32_t CB_W = 7;
constexpr uint32_t CB_SCRATCH = 18;

template <uint32_t cb_id, typename Acc>
FORCE_INLINE void read_tiles(const Noc& noc, const Acc& acc, uint32_t first, uint32_t n) {
    constexpr uint32_t bytes = get_tile_size(cb_id);
    CircularBuffer cb(cb_id);
    cb.reserve_back(n);
    uint32_t dst = cb.get_write_ptr();
    for (uint32_t i = 0; i < n; ++i) {
        noc.async_read(acc, CoreLocalMem<uint32_t>(dst), bytes, {.page_id = first + i}, {});
        dst += bytes;
    }
    noc.async_read_barrier();
    cb.push_back(n);
}

FORCE_INLINE uint32_t row0_bf16(uint32_t tile_addr, uint32_t col) {
    volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr);
    uint32_t index = col < 16 ? col : 256 + col - 16;
    return static_cast<uint32_t>(t[index]) << 16;
}

FORCE_INLINE uint32_t row0_f32(uint32_t tile_addr, uint32_t col) {
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
    uint32_t index = col < 16 ? col : 256 + col - 16;
    return t[index];
}

FORCE_INLINE void zero_tile(volatile tt_l1_ptr uint32_t* t) {
    for (uint32_t i = 0; i < 1024; ++i) {
        t[i] = 0;
    }
}

template <uint32_t cb_id, typename Acc>
FORCE_INLINE void read_column_tiles(const Noc& noc, const Acc& acc, uint32_t first, uint32_t n) {
    constexpr uint32_t src_bytes = 2048;
    constexpr uint32_t dst_bytes = get_tile_size(cb_id);
    CircularBuffer scratch(CB_SCRATCH);
    uint32_t src = scratch.get_write_ptr();
    for (uint32_t i = 0; i < n; ++i) {
        noc.async_read(acc, CoreLocalMem<uint32_t>(src + i * src_bytes), src_bytes, {.page_id = first + i}, {});
    }
    noc.async_read_barrier();
    CircularBuffer cb(cb_id);
    cb.reserve_back(n);
    uint32_t dst = cb.get_write_ptr();
    for (uint32_t i = 0; i < n; ++i) {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + i * dst_bytes);
        for (uint32_t r = 0; r < 32; ++r) {
            uint32_t index = (r < 16 ? 0 : 512) + (r % 16) * 16;
            t[index] = row0_bf16(src + i * src_bytes, r);
        }
    }
    cb.push_back(n);
}

template <uint32_t cb_id, typename Acc>
FORCE_INLINE void read_row_tiles(const Noc& noc, const Acc& acc, uint32_t first, uint32_t n) {
    constexpr uint32_t src_bytes = 2048;
    constexpr uint32_t dst_bytes = get_tile_size(cb_id);
    CircularBuffer scratch(CB_SCRATCH);
    uint32_t src = scratch.get_write_ptr();
    for (uint32_t i = 0; i < n; ++i) {
        noc.async_read(acc, CoreLocalMem<uint32_t>(src + i * src_bytes), src_bytes, {.page_id = first + i}, {});
    }
    noc.async_read_barrier();
    CircularBuffer cb(cb_id);
    cb.reserve_back(n);
    uint32_t dst = cb.get_write_ptr();
    for (uint32_t i = 0; i < n; ++i) {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + i * dst_bytes);
        for (uint32_t c = 0; c < 32; ++c) {
            t[c < 16 ? c : 256 + c - 16] = row0_bf16(src + i * src_bytes, c);
        }
    }
    cb.push_back(n);
}

template <uint32_t cb_id, bool f32, typename Acc>
FORCE_INLINE void read_scalar(const Noc& noc, const Acc& acc, uint32_t col) {
    CircularBuffer scratch(CB_SCRATCH);
    uint32_t src = scratch.get_write_ptr();
    noc.async_read(acc, CoreLocalMem<uint32_t>(src), f32 ? 4096 : 2048, {.page_id = 0}, {});
    noc.async_read_barrier();
    CircularBuffer cb(cb_id);
    cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
    t[0] = f32 ? row0_f32(src, col) : row0_bf16(src, col);
    cb.push_back(1);
}

template <bool f32, typename Acc>
FORCE_INLINE uint32_t scalar_at(const Noc& noc, const Acc& acc, uint32_t column) {
    CircularBuffer scratch(CB_SCRATCH);
    uint32_t src = scratch.get_write_ptr();
    noc.async_read(acc, CoreLocalMem<uint32_t>(src), f32 ? 4096 : 2048, {.page_id = column / 32}, {});
    noc.async_read_barrier();
    return f32 ? row0_f32(src, column % 32) : row0_bf16(src, column % 32);
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
    constexpr uint32_t decay_f32 = get_compile_time_arg_val(1);
    constexpr uint32_t beta_f32 = get_compile_time_arg_val(2);
    constexpr uint32_t blocks = get_compile_time_arg_val(3);
    constexpr uint32_t gates = get_compile_time_arg_val(4);
    constexpr auto q_args = TensorAccessorArgs<5>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto z_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto n_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    const auto q = TensorAccessor(q_args, get_arg_val<uint32_t>(0));
    const auto k = TensorAccessor(k_args, get_arg_val<uint32_t>(1));
    const auto v = TensorAccessor(v_args, get_arg_val<uint32_t>(2));
    const auto z = TensorAccessor(z_args, get_arg_val<uint32_t>(3));
    const auto decay = TensorAccessor(d_args, get_arg_val<uint32_t>(4));
    const auto beta = TensorAccessor(b_args, get_arg_val<uint32_t>(5));
    const auto state = TensorAccessor(s_args, get_arg_val<uint32_t>(6));
    const auto weight = TensorAccessor(w_args, get_arg_val<uint32_t>(7));
    const uint32_t head = get_arg_val<uint32_t>(8);
    const uint32_t first_block = get_arg_val<uint32_t>(9);
    const auto negative = TensorAccessor(n_args, get_arg_val<uint32_t>(10));
    const uint32_t a_column = get_arg_val<uint32_t>(11);
    const uint32_t b_column = get_arg_val<uint32_t>(12);
    const uint32_t z_offset = get_arg_val<uint32_t>(13);
    const uint32_t key_head = head / groups;
    Noc noc;

    read_column_tiles<CB_QC>(noc, q, key_head * 4, 4);
    read_column_tiles<CB_KC>(noc, k, key_head * 4, 4);
    if constexpr (gates) {
        const uint32_t decay_inputs[3] = {
            scalar_at<false>(noc, decay, a_column), scalar_at<true>(noc, beta, head), scalar_at<true>(noc, negative, head)};
        push_scalars<CB_DECAY>(decay_inputs, 3);
        const uint32_t beta_input = scalar_at<false>(noc, decay, b_column);
        push_scalars<CB_BETA>(&beta_input, 1);
    } else {
        read_scalar<CB_DECAY, decay_f32 != 0>(noc, decay, head);
        read_scalar<CB_BETA, beta_f32 != 0>(noc, beta, head);
    }
    for (uint32_t c = first_block; c < first_block + blocks; ++c) {
        for (uint32_t row = 0; row < 4; ++row) {
            read_tiles<CB_S>(noc, state, head * 16 + row * 4 + c, 1);
        }
    }
    read_tiles<CB_V>(noc, v, head * 4 + first_block, blocks);
    if constexpr (blocks == 4) {
        read_tiles<CB_Z>(noc, z, z_offset + head * 4, 4);
        read_row_tiles<CB_W>(noc, weight, 0, 4);
    }
}

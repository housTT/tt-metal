// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the single-token DeltaNet recurrence. One Tensix core owns each
// flattened (batch, value-head) and normalizes/scales its Q/K vectors in L1.

#include <cmath>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

FORCE_INLINE float bf16_to_f32(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

FORCE_INLINE uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(uint32_t));
    return static_cast<uint16_t>((bits + 0x8000) >> 16);
}

FORCE_INLINE float extract_vector_element(uint32_t tile_l1_addr, uint32_t index) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);
    const uint32_t face = index / 16;
    const uint32_t position = face * 256 + index % 16;
    return bf16_to_f32(tile[position]);
}

FORCE_INLINE float extract_tile_element(uint32_t tile_l1_addr, uint32_t row, uint32_t column) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);
    const uint32_t face = (row / 16) * 2 + column / 16;
    const uint32_t position = face * 256 + (row % 16) * 16 + column % 16;
    return bf16_to_f32(tile[position]);
}

FORCE_INLINE void write_vector_element(uint32_t tile_l1_addr, uint32_t index, uint16_t value) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);
    const uint32_t face = index / 16;
    const uint32_t position = face * 256 + index % 16;
    tile[position] = value;
}

FORCE_INLINE void select_tile_row(uint32_t tile_l1_addr, uint32_t row) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);
    for (uint32_t column = 0; column < 32; ++column) {
        const uint32_t face = (row / 16) * 2 + column / 16;
        const uint32_t position = face * 256 + (row % 16) * 16 + column % 16;
        write_vector_element(tile_l1_addr, column, tile[position]);
    }
}

FORCE_INLINE void normalize_vector(uint32_t vector_l1, uint32_t num_tiles, uint32_t tile_bytes, float scale) {
    float sum_squares = 0.0f;
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        const uint32_t tile_l1 = vector_l1 + tile * tile_bytes;
        for (uint32_t element = 0; element < 32; ++element) {
            const float value = extract_vector_element(tile_l1, element);
            sum_squares += value * value;
        }
    }
    const float multiplier = scale / sqrtf(sum_squares + 1.0e-6f);
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        const uint32_t tile_l1 = vector_l1 + tile * tile_bytes;
        for (uint32_t element = 0; element < 32; ++element) {
            const float value = extract_vector_element(tile_l1, element);
            write_vector_element(tile_l1, element, f32_to_bf16(value * multiplier));
        }
    }
}

FORCE_INLINE void make_broadcast_scalar(uint32_t tile_l1_addr, float value) {
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);
    for (uint32_t index = 0; index < 1024; ++index) {
        tile[index] = 0;
    }
    const uint16_t bf16_value = f32_to_bf16(value);
    tile[0] = bf16_value;
    tile[256] = bf16_value;
    tile[512] = bf16_value;
    tile[768] = bf16_value;
}

void kernel_main() {
    constexpr uint32_t cb_state = get_compile_time_arg_val(0);
    constexpr uint32_t cb_q = get_compile_time_arg_val(1);
    constexpr uint32_t cb_k = get_compile_time_arg_val(2);
    constexpr uint32_t cb_v = get_compile_time_arg_val(3);
    constexpr uint32_t cb_decay = get_compile_time_arg_val(4);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(5);
    constexpr uint32_t cb_k_transposed = get_compile_time_arg_val(6);
    constexpr uint32_t k_head_dim_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t v_head_dim_tiles = get_compile_time_arg_val(8);
    constexpr bool preprocess_ab = get_compile_time_arg_val(9) == 1;
    constexpr auto state_args = TensorAccessorArgs<10>();
    constexpr auto q_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto decay_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();
    constexpr auto decay_scale_args = TensorAccessorArgs<decay_args.next_compile_time_args_offset()>();
    constexpr auto dt_bias_args = TensorAccessorArgs<decay_scale_args.next_compile_time_args_offset()>();

    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t decay_addr = get_arg_val<uint32_t>(5);
    const uint32_t decay_scale_addr = get_arg_val<uint32_t>(6);
    const uint32_t dt_bias_addr = get_arg_val<uint32_t>(7);
    const uint32_t state_start_tile = get_arg_val<uint32_t>(8);
    const uint32_t scalar_tile = get_arg_val<uint32_t>(9);
    const uint32_t scalar_row = get_arg_val<uint32_t>(10);
    const uint32_t scalar_column = get_arg_val<uint32_t>(11);
    const uint32_t q_start_tile = get_arg_val<uint32_t>(12);
    const uint32_t k_start_tile = get_arg_val<uint32_t>(13);
    const uint32_t v_start_tile = get_arg_val<uint32_t>(14);
    const uint32_t key_head_row = get_arg_val<uint32_t>(15);
    const uint32_t value_head_row = get_arg_val<uint32_t>(16);

    constexpr uint32_t state_tiles = k_head_dim_tiles * v_head_dim_tiles;
    const uint32_t tile_bytes = get_tile_size(cb_q);
    const auto state_accessor = TensorAccessor(state_args, state_addr, tile_bytes);
    const auto q_accessor = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_accessor = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_accessor = TensorAccessor(v_args, v_addr, tile_bytes);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr, tile_bytes);
    const auto decay_accessor = TensorAccessor(decay_args, decay_addr, tile_bytes);
    const auto decay_scale_accessor = TensorAccessor(decay_scale_args, decay_scale_addr, tile_bytes);
    const auto dt_bias_accessor = TensorAccessor(dt_bias_args, dt_bias_addr, tile_bytes);

    cb_reserve_back(cb_state, state_tiles);
    uint32_t state_l1 = get_write_ptr(cb_state);
    for (uint32_t tile = 0; tile < state_tiles; ++tile) {
        noc_async_read_tile(state_start_tile + tile, state_accessor, state_l1);
        state_l1 += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_state, state_tiles);

    cb_reserve_back(cb_q, k_head_dim_tiles);
    const uint32_t q_l1_start = get_write_ptr(cb_q);
    uint32_t q_l1 = q_l1_start;
    for (uint32_t tile = 0; tile < k_head_dim_tiles; ++tile) {
        noc_async_read_tile(q_start_tile + tile, q_accessor, q_l1);
        q_l1 += tile_bytes;
    }
    noc_async_read_barrier();
    for (uint32_t tile = 0; tile < k_head_dim_tiles; ++tile) {
        select_tile_row(q_l1_start + tile * tile_bytes, key_head_row);
    }
    normalize_vector(q_l1_start, k_head_dim_tiles, tile_bytes, 1.0f / sqrtf(static_cast<float>(k_head_dim_tiles * 32)));
    cb_push_back(cb_q, k_head_dim_tiles);

    cb_reserve_back(cb_k, k_head_dim_tiles);
    const uint32_t k_l1_start = get_write_ptr(cb_k);
    uint32_t k_l1 = k_l1_start;
    for (uint32_t tile = 0; tile < k_head_dim_tiles; ++tile) {
        noc_async_read_tile(k_start_tile + tile, k_accessor, k_l1);
        k_l1 += tile_bytes;
    }
    noc_async_read_barrier();
    for (uint32_t tile = 0; tile < k_head_dim_tiles; ++tile) {
        select_tile_row(k_l1_start + tile * tile_bytes, key_head_row);
    }
    normalize_vector(k_l1_start, k_head_dim_tiles, tile_bytes, 1.0f);
    cb_push_back(cb_k, k_head_dim_tiles);

    cb_reserve_back(cb_v, v_head_dim_tiles);
    const uint32_t v_l1_start = get_write_ptr(cb_v);
    uint32_t v_l1 = v_l1_start;
    for (uint32_t tile = 0; tile < v_head_dim_tiles; ++tile) {
        noc_async_read_tile(v_start_tile + tile, v_accessor, v_l1);
        v_l1 += tile_bytes;
    }
    noc_async_read_barrier();
    for (uint32_t tile = 0; tile < v_head_dim_tiles; ++tile) {
        select_tile_row(v_l1_start + tile * tile_bytes, value_head_row);
    }
    cb_push_back(cb_v, v_head_dim_tiles);

    cb_reserve_back(cb_k_transposed, k_head_dim_tiles);
    uint32_t k_source = get_read_ptr(cb_k);
    uint32_t k_transposed = get_write_ptr(cb_k_transposed);
    for (uint32_t tile_index = 0; tile_index < k_head_dim_tiles; ++tile_index) {
        volatile tt_l1_ptr uint16_t* source = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(k_source);
        volatile tt_l1_ptr uint16_t* destination = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(k_transposed);
        for (uint32_t index = 0; index < 1024; ++index) {
            destination[index] = 0;
        }
        for (uint32_t index = 0; index < 16; ++index) {
            destination[index * 16] = source[index];
            destination[512 + index * 16] = source[256 + index];
        }
        k_source += tile_bytes;
        k_transposed += tile_bytes;
    }
    cb_push_back(cb_k_transposed, k_head_dim_tiles);

    cb_reserve_back(cb_beta, 1);
    uint32_t beta_l1 = get_write_ptr(cb_beta);
    noc_async_read_tile(scalar_tile, beta_accessor, beta_l1);
    noc_async_read_barrier();
    float beta = extract_tile_element(beta_l1, scalar_row, scalar_column);
    if constexpr (preprocess_ab) {
        beta = 1.0f / (1.0f + expf(-beta));
    }
    make_broadcast_scalar(beta_l1, beta);
    cb_push_back(cb_beta, 1);

    cb_reserve_back(cb_decay, 1);
    uint32_t decay_l1 = get_write_ptr(cb_decay);
    noc_async_read_tile(scalar_tile, decay_accessor, decay_l1);
    noc_async_read_barrier();
    float decay = extract_tile_element(decay_l1, scalar_row, scalar_column);
    if constexpr (preprocess_ab) {
        const uint32_t constant_tile = value_head_row / 32;
        noc_async_read_tile(constant_tile, decay_scale_accessor, decay_l1);
        noc_async_read_barrier();
        const float decay_scale = extract_tile_element(decay_l1, 0, scalar_column);
        noc_async_read_tile(constant_tile, dt_bias_accessor, decay_l1);
        noc_async_read_barrier();
        const float biased_a = decay + extract_tile_element(decay_l1, 0, scalar_column);
        const float softplus = biased_a > 20.0f ? biased_a : log1pf(expf(biased_a));
        decay = expf(decay_scale * softplus);
    }
    make_broadcast_scalar(decay_l1, decay);
    cb_push_back(cb_decay, 1);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the single-token DeltaNet recurrence. Q/K arrive normalized and
// Q is already scaled. One Tensix core owns each flattened (batch, value-head).

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
    constexpr auto state_args = TensorAccessorArgs<9>();
    constexpr auto qkv_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto decay_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();

    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t qkv_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t decay_addr = get_arg_val<uint32_t>(3);
    const uint32_t state_start_tile = get_arg_val<uint32_t>(4);
    const uint32_t head = get_arg_val<uint32_t>(5);
    const uint32_t q_start_tile = get_arg_val<uint32_t>(6);
    const uint32_t k_start_tile = get_arg_val<uint32_t>(7);
    const uint32_t v_start_tile = get_arg_val<uint32_t>(8);

    constexpr uint32_t state_tiles = k_head_dim_tiles * v_head_dim_tiles;
    const uint32_t tile_bytes = get_tile_size(cb_q);
    const auto state_accessor = TensorAccessor(state_args, state_addr, tile_bytes);
    const auto qkv_accessor = TensorAccessor(qkv_args, qkv_addr, tile_bytes);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr, tile_bytes);
    const auto decay_accessor = TensorAccessor(decay_args, decay_addr, tile_bytes);

    cb_reserve_back(cb_state, state_tiles);
    uint32_t state_l1 = get_write_ptr(cb_state);
    for (uint32_t tile = 0; tile < state_tiles; ++tile) {
        noc_async_read_tile(state_start_tile + tile, state_accessor, state_l1);
        state_l1 += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_state, state_tiles);

    const uint32_t starts[3] = {q_start_tile, k_start_tile, v_start_tile};
    const uint32_t counts[3] = {k_head_dim_tiles, k_head_dim_tiles, v_head_dim_tiles};
    const uint32_t buffers[3] = {cb_q, cb_k, cb_v};
    for (uint32_t component = 0; component < 3; ++component) {
        cb_reserve_back(buffers[component], counts[component]);
        uint32_t component_l1 = get_write_ptr(buffers[component]);
        for (uint32_t tile = 0; tile < counts[component]; ++tile) {
            noc_async_read_tile(starts[component] + tile, qkv_accessor, component_l1);
            component_l1 += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(buffers[component], counts[component]);
    }

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

    const uint32_t scalar_tile = head / 32;
    const uint32_t scalar_element = head % 32;
    cb_reserve_back(cb_beta, 1);
    uint32_t beta_l1 = get_write_ptr(cb_beta);
    noc_async_read_tile(scalar_tile, beta_accessor, beta_l1);
    noc_async_read_barrier();
    const float beta = extract_vector_element(beta_l1, scalar_element);
    make_broadcast_scalar(beta_l1, beta);
    cb_push_back(cb_beta, 1);

    cb_reserve_back(cb_decay, 1);
    uint32_t decay_l1 = get_write_ptr(cb_decay);
    noc_async_read_tile(scalar_tile, decay_accessor, decay_l1);
    noc_async_read_barrier();
    const float decay = extract_vector_element(decay_l1, scalar_element);
    make_broadcast_scalar(decay_l1, decay);
    cb_push_back(cb_decay, 1);
}

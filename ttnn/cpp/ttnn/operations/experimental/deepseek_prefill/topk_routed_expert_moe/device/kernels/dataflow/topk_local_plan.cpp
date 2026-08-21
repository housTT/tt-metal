// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One-core router-index-native planner for the local expert shard.
//
// It consumes ttnn.topk's TILE UINT32 global expert indices directly. No
// [T,E] routing tensor is created. The per-device global->local table uses the
// same convention as Ornith's fused local router gate: 0..E_local-1 are owned,
// every value >= E_local is non-local. The kernel makes two deterministic
// token-major passes: histogram/offset planning, then assignment placement.
// Malformed mappings, duplicate expert ids within one token, out-of-range
// indices or capacity overflow fail closed by publishing all-zero
// counts/validity and safe zero indices.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t cb_mapping_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_counts_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_local_to_global_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_assignments_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_inverse_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_valid_id = get_compile_time_arg_val(7);
    constexpr uint32_t mapping_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t counts_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t local_to_global_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t assignments_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t inverse_page_size = get_compile_time_arg_val(14);
    constexpr uint32_t valid_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t tokens = get_compile_time_arg_val(16);
    constexpr uint32_t topk = get_compile_time_arg_val(17);
    constexpr uint32_t num_global_experts = get_compile_time_arg_val(18);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(19);
    constexpr uint32_t capacity = get_compile_time_arg_val(20);
    constexpr uint32_t slots = get_compile_time_arg_val(21);

    constexpr auto indices_args = TensorAccessorArgs<22>();
    constexpr auto mapping_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto counts_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr auto local_to_global_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto assignments_args = TensorAccessorArgs<local_to_global_args.next_compile_time_args_offset()>();
    constexpr auto inverse_args = TensorAccessorArgs<assignments_args.next_compile_time_args_offset()>();
    constexpr auto valid_args = TensorAccessorArgs<inverse_args.next_compile_time_args_offset()>();

    uint32_t runtime_index = 0;
    const uint32_t indices_addr = get_arg_val<uint32_t>(runtime_index++);
    const uint32_t mapping_addr = get_arg_val<uint32_t>(runtime_index++);
    const uint32_t counts_addr = get_arg_val<uint32_t>(runtime_index++);
    const uint32_t offsets_addr = get_arg_val<uint32_t>(runtime_index++);
    const uint32_t local_to_global_addr = get_arg_val<uint32_t>(runtime_index++);
    const uint32_t assignments_addr = get_arg_val<uint32_t>(runtime_index++);
    const uint32_t inverse_addr = get_arg_val<uint32_t>(runtime_index++);
    const uint32_t valid_addr = get_arg_val<uint32_t>(runtime_index++);
    const uint32_t valid_tokens = get_arg_val<uint32_t>(runtime_index++);
    const uint32_t ready_semaphore_id = get_arg_val<uint32_t>(runtime_index++);
    const uint32_t num_dispatch_cores = get_arg_val<uint32_t>(runtime_index++);

    const auto indices = TensorAccessor(indices_args, indices_addr);
    const auto mapping = TensorAccessor(mapping_args, mapping_addr);
    const auto counts_output = TensorAccessor(counts_args, counts_addr);
    const auto offsets_output = TensorAccessor(offsets_args, offsets_addr);
    const auto local_to_global_output = TensorAccessor(local_to_global_args, local_to_global_addr);
    const auto assignments_output = TensorAccessor(assignments_args, assignments_addr);
    const auto inverse_output = TensorAccessor(inverse_args, inverse_addr);
    const auto valid_output = TensorAccessor(valid_args, valid_addr);

    CircularBuffer cb_mapping(cb_mapping_id);
    CircularBuffer cb_indices(cb_indices_id);
    CircularBuffer cb_counts(cb_counts_id);
    CircularBuffer cb_offsets(cb_offsets_id);
    CircularBuffer cb_local_to_global(cb_local_to_global_id);
    CircularBuffer cb_assignments(cb_assignments_id);
    CircularBuffer cb_inverse(cb_inverse_id);
    CircularBuffer cb_valid(cb_valid_id);

    const uint32_t mapping_l1 = cb_mapping.get_write_ptr();
    const uint32_t indices_l1 = cb_indices.get_write_ptr();
    const uint32_t counts_l1 = cb_counts.get_write_ptr();
    const uint32_t offsets_l1 = cb_offsets.get_write_ptr();
    const uint32_t local_to_global_l1 = cb_local_to_global.get_write_ptr();
    const uint32_t assignments_l1 = cb_assignments.get_write_ptr();
    const uint32_t inverse_l1 = cb_inverse.get_write_ptr();
    const uint32_t valid_l1 = cb_valid.get_write_ptr();

    volatile tt_l1_ptr uint32_t* global_to_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mapping_l1);
    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
    volatile tt_l1_ptr uint32_t* offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(offsets_l1);
    volatile tt_l1_ptr uint32_t* local_to_global = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_to_global_l1);
    volatile tt_l1_ptr uint32_t* assignments = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(assignments_l1);
    volatile tt_l1_ptr uint32_t* inverse = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inverse_l1);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_l1);

    Noc noc;
    noc.async_read(mapping, CoreLocalMem<uint32_t>(mapping_l1), mapping_page_size, {.page_id = 0}, {});
    noc.async_read_barrier();

    for (uint32_t word = 0; word < counts_page_size / sizeof(uint32_t); ++word) {
        counts[word] = 0;
    }
    for (uint32_t word = 0; word < offsets_page_size / sizeof(uint32_t); ++word) {
        offsets[word] = 0;
    }
    for (uint32_t word = 0; word < local_to_global_page_size / sizeof(uint32_t); ++word) {
        local_to_global[word] = 0xFFFFFFFFu;
    }
    // Padding uses an explicit out-of-range sentinel. The dispatch kernel
    // checks it before reading x and writes an exact zero activation row.
    for (uint32_t word = 0; word < assignments_page_size / sizeof(uint32_t); ++word) {
        assignments[word] = 0xFFFFFFFFu;
    }
    for (uint32_t word = 0; word < inverse_page_size / sizeof(uint32_t); ++word) {
        inverse[word] = 0;
    }
    for (uint32_t word = 0; word < valid_page_size / sizeof(uint32_t); ++word) {
        valid[word] = 0;
    }

    bool plan_ok = valid_tokens > 0 && valid_tokens <= tokens;
    for (uint32_t global_expert = 0; global_expert < num_global_experts; ++global_expert) {
        const uint32_t local_expert = global_to_local[global_expert];
        if (local_expert < num_local_experts) {
            if (local_to_global[local_expert] != 0xFFFFFFFFu) {
                plan_ok = false;  // duplicate local id
            } else {
                local_to_global[local_expert] = global_expert;
            }
        }
    }
    for (uint32_t local_expert = 0; local_expert < num_local_experts; ++local_expert) {
        if (local_to_global[local_expert] == 0xFFFFFFFFu) {
            plan_ok = false;  // missing local id
        }
    }

    // A 32x32 UINT32 tile stores four 16x16 faces. K<=16 lives entirely in
    // the left faces, so no untilize or dtype conversion is required.
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t face_dim = 16;
    constexpr uint32_t face_size = face_dim * face_dim;
    const uint32_t tile_words = indices_page_size / sizeof(uint32_t);
    for (uint32_t tile_row = 0; tile_row < tokens / tile_height; ++tile_row) {
        noc.async_read(indices, CoreLocalMem<uint32_t>(indices_l1), indices_page_size, {.page_id = tile_row}, {});
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(indices_l1);
        for (uint32_t within = 0; within < tile_height; ++within) {
            const uint32_t token = tile_row * tile_height + within;
            if (token >= valid_tokens) {
                continue;
            }
            const uint32_t row_base = (within / face_dim) * 2 * face_size + (within % face_dim) * face_dim;
            for (uint32_t slot = 0; slot < topk; ++slot) {
                const uint32_t global_expert = tile[row_base + slot];
                // Router top-k rows contain unique expert ids. Enforce that
                // contract on device so a malformed direct caller cannot give
                // one expert more than `tokens` assignments and leave the
                // unified FFN's final valid slot unwritten.
                bool duplicate_expert = false;
                for (uint32_t prior_slot = 0; prior_slot < slot; ++prior_slot) {
                    if (tile[row_base + prior_slot] == global_expert) {
                        duplicate_expert = true;
                        break;
                    }
                }
                if (duplicate_expert) {
                    plan_ok = false;
                    continue;
                }
                if (global_expert >= num_global_experts) {
                    plan_ok = false;
                    continue;
                }
                if (global_to_local[global_expert] < num_local_experts) {
                    counts[global_expert] += 1;
                }
            }
        }
        (void)tile_words;
    }

    uint32_t region_end = 0;
    if (plan_ok) {
        for (uint32_t local_expert = 0; local_expert < num_local_experts; ++local_expert) {
            const uint32_t global_expert = local_to_global[local_expert];
            offsets[global_expert] = region_end;
            region_end += ((counts[global_expert] + tile_height - 1) / tile_height) * tile_height;
            if (region_end > capacity) {
                plan_ok = false;
                break;
            }
        }
    }

    uint32_t cursors[num_local_experts];
    for (uint32_t local_expert = 0; local_expert < num_local_experts; ++local_expert) {
        cursors[local_expert] = 0;
    }

    if (plan_ok) {
        for (uint32_t tile_row = 0; tile_row < tokens / tile_height; ++tile_row) {
            noc.async_read(indices, CoreLocalMem<uint32_t>(indices_l1), indices_page_size, {.page_id = tile_row}, {});
            noc.async_read_barrier();
            volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(indices_l1);
            for (uint32_t within = 0; within < tile_height; ++within) {
                const uint32_t token = tile_row * tile_height + within;
                if (token >= valid_tokens) {
                    continue;
                }
                const uint32_t row_base = (within / face_dim) * 2 * face_size + (within % face_dim) * face_dim;
                for (uint32_t slot = 0; slot < topk; ++slot) {
                    const uint32_t global_expert = tile[row_base + slot];
                    const uint32_t local_expert = global_to_local[global_expert];
                    if (local_expert >= num_local_experts) {
                        continue;
                    }
                    const uint32_t packed_row = offsets[global_expert] + cursors[local_expert]++;
                    if (packed_row >= capacity) {
                        plan_ok = false;
                        break;
                    }
                    const uint32_t flat_assignment = token * topk + slot;
                    assignments[packed_row] = flat_assignment;
                    const uint32_t inverse_index = slot * tokens + token;
                    inverse[inverse_index] = packed_row;
                    valid[inverse_index] = 1;
                }
                if (!plan_ok) {
                    break;
                }
            }
            if (!plan_ok) {
                break;
            }
        }
    }

    if (!plan_ok) {
        for (uint32_t word = 0; word < counts_page_size / sizeof(uint32_t); ++word) {
            counts[word] = 0;
        }
        for (uint32_t word = 0; word < offsets_page_size / sizeof(uint32_t); ++word) {
            offsets[word] = 0;
        }
        for (uint32_t word = 0; word < local_to_global_page_size / sizeof(uint32_t); ++word) {
            local_to_global[word] = 0;
        }
        for (uint32_t word = 0; word < assignments_page_size / sizeof(uint32_t); ++word) {
            assignments[word] = 0xFFFFFFFFu;
        }
        for (uint32_t word = 0; word < inverse_page_size / sizeof(uint32_t); ++word) {
            inverse[word] = 0;
        }
        for (uint32_t word = 0; word < valid_page_size / sizeof(uint32_t); ++word) {
            valid[word] = 0;
        }
    }

    noc.async_write(CoreLocalMem<uint32_t>(counts_l1), counts_output, counts_page_size, {}, {.page_id = 0});
    noc.async_write(CoreLocalMem<uint32_t>(offsets_l1), offsets_output, offsets_page_size, {}, {.page_id = 0});
    noc.async_write(
        CoreLocalMem<uint32_t>(local_to_global_l1),
        local_to_global_output,
        local_to_global_page_size,
        {},
        {.page_id = 0});
    noc.async_write(
        CoreLocalMem<uint32_t>(assignments_l1), assignments_output, assignments_page_size, {}, {.page_id = 0});
    noc.async_write(CoreLocalMem<uint32_t>(inverse_l1), inverse_output, inverse_page_size, {}, {.page_id = 0});
    noc.async_write(CoreLocalMem<uint32_t>(valid_l1), valid_output, valid_page_size, {}, {.page_id = 0});
    noc.async_write_barrier();

    // The dispatch kernels consume packed assignments from DRAM. Publish only
    // after all six plan outputs are globally visible.
    Semaphore<> ready(ready_semaphore_id);
    for (uint32_t core = 0; core < num_dispatch_cores; ++core) {
        const uint32_t noc_x = get_arg_val<uint32_t>(runtime_index++);
        const uint32_t noc_y = get_arg_val<uint32_t>(runtime_index++);
        ready.up(noc, noc_x, noc_y, 1);
    }
    noc.async_atomic_barrier();
}

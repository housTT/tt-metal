// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_local_combine_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe {

namespace {

constexpr uint32_t CB_PACKED_ROW = tt::CBIndex::c_0;
constexpr uint32_t CB_WEIGHT = tt::CBIndex::c_1;
constexpr uint32_t CB_READER_INVERSE = tt::CBIndex::c_2;
constexpr uint32_t CB_READER_VALID = tt::CBIndex::c_3;
constexpr uint32_t CB_WRITER_VALID = tt::CBIndex::c_4;
constexpr uint32_t CB_WEIGHT_SCRATCH = tt::CBIndex::c_5;
constexpr uint32_t CB_OUTPUT = tt::CBIndex::c_16;
constexpr uint32_t CB_ROW_MAJOR_ACCUM = tt::CBIndex::c_17;
constexpr uint32_t TOKENS_PER_CHUNK = tt::constants::TILE_HEIGHT;
constexpr uint32_t PACKED_ROW_BATCH_CAP = 8;

void create_combine_cb(
    tt::tt_metal::Program& program,
    const CoreRangeSet& cores,
    uint32_t cb_id,
    uint32_t total_size,
    uint32_t page_size,
    tt::DataFormat data_format) {
    auto config =
        tt::tt_metal::CircularBufferConfig(total_size, {{cb_id, data_format}}).set_page_size(cb_id, page_size);
    tt::tt_metal::CreateCircularBuffer(program, cores, config);
}

std::vector<uint32_t> reader_runtime_args(const TopkLocalCombineInputs& tensors, uint32_t token_start) {
    return {
        tensors.packed_y.buffer()->address(),
        tensors.slot_to_packed_row.buffer()->address(),
        tensors.slot_is_local.buffer()->address(),
        token_start,
    };
}

std::vector<uint32_t> writer_runtime_args(
    const TopkLocalCombineInputs& tensors, const Tensor& output, uint32_t token_start) {
    return {
        tensors.topk_weights.buffer()->address(),
        tensors.slot_is_local.buffer()->address(),
        output.buffer()->address(),
        token_start,
    };
}

}  // namespace

TopkLocalCombineProgramFactory::cached_program_t TopkLocalCombineProgramFactory::create(
    const TopkLocalCombineParams& op, const TopkLocalCombineInputs& tensors, Tensor& output) {
    tt::tt_metal::Program program;

    const uint32_t hidden = tensors.packed_y.logical_shape()[-1];
    const uint32_t capacity = tensors.packed_y.logical_shape()[-2];
    const uint32_t emb_dim_cb_tiles = hidden / tt::constants::TILE_HW;
    const uint32_t emb_dim_bytes = hidden * sizeof(uint16_t);
    const uint32_t emb_dim_out_tiles = hidden / tt::constants::TILE_WIDTH;
    const uint32_t map_chunk_bytes = op.topk * TOKENS_PER_CHUNK * sizeof(uint32_t);
    const uint32_t weight_page_size = tensors.topk_weights.buffer()->aligned_page_size();
    const uint32_t rows_per_batch = std::min(op.topk, PACKED_ROW_BATCH_CAP);

    const auto input_format = tt::tt_metal::datatype_to_dataformat_converter(tensors.packed_y.dtype());
    const auto weight_format = tt::tt_metal::datatype_to_dataformat_converter(tensors.topk_weights.dtype());
    const auto uint32_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::UINT32);
    const auto output_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t input_tile_size = tt::tile_size(input_format);
    const uint32_t weight_tile_size = tt::tile_size(weight_format);
    const uint32_t output_tile_size = tt::tile_size(output_format);

    auto* device = tensors.packed_y.device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_chunks = op.tokens / TOKENS_PER_CHUNK;
    const CoreRangeSet all_cores = tt::tt_metal::num_cores_to_corerangeset(num_chunks, grid, true);
    auto cores = tt::tt_metal::corerange_to_cores(all_cores, num_chunks, true);
    TT_FATAL(cores.size() == num_chunks, "fused top-k combine needs one core per 32-token chunk");

    const uint64_t row_bytes = static_cast<uint64_t>(emb_dim_cb_tiles) * input_tile_size;
    // Keep one complete token's K rows in the ring. The reader transfers at
    // most rows_per_batch at a time, but a full-K capacity makes every batch
    // physically contiguous even when K is not divisible by the batch cap.
    const uint64_t packed_row_cb_bytes = op.topk * row_bytes;
    const uint64_t weight_scratch_bytes = TOKENS_PER_CHUNK * weight_page_size;
    const uint64_t output_bytes = static_cast<uint64_t>(emb_dim_out_tiles) * output_tile_size;
    const uint64_t row_major_accum_bytes = static_cast<uint64_t>(TOKENS_PER_CHUNK) * emb_dim_cb_tiles * input_tile_size;
    const uint64_t cb_bytes = packed_row_cb_bytes + weight_tile_size + 3ull * map_chunk_bytes + weight_scratch_bytes +
                              output_bytes + row_major_accum_bytes;
    const uint32_t l1_reserved = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    constexpr uint32_t l1_margin = 32 * 1024;
    TT_FATAL(
        device->l1_size_per_core() > l1_reserved + l1_margin &&
            cb_bytes <= device->l1_size_per_core() - l1_reserved - l1_margin,
        "top-k combine CB footprint ({} bytes) exceeds per-core L1 budget (size {}, reserved {}, margin {})",
        cb_bytes,
        device->l1_size_per_core(),
        l1_reserved,
        l1_margin);

    create_combine_cb(program, all_cores, CB_PACKED_ROW, packed_row_cb_bytes, input_tile_size, input_format);
    create_combine_cb(program, all_cores, CB_WEIGHT, weight_tile_size, weight_tile_size, weight_format);
    create_combine_cb(program, all_cores, CB_READER_INVERSE, map_chunk_bytes, map_chunk_bytes, uint32_format);
    create_combine_cb(program, all_cores, CB_READER_VALID, map_chunk_bytes, map_chunk_bytes, uint32_format);
    create_combine_cb(program, all_cores, CB_WRITER_VALID, map_chunk_bytes, map_chunk_bytes, uint32_format);
    create_combine_cb(program, all_cores, CB_WEIGHT_SCRATCH, weight_scratch_bytes, weight_page_size, weight_format);
    create_combine_cb(program, all_cores, CB_OUTPUT, output_bytes, output_tile_size, output_format);
    create_combine_cb(program, all_cores, CB_ROW_MAJOR_ACCUM, row_major_accum_bytes, input_tile_size, input_format);

    std::vector<uint32_t> reader_compile_args = {
        CB_PACKED_ROW,
        CB_READER_INVERSE,
        CB_READER_VALID,
        emb_dim_cb_tiles,
        emb_dim_bytes,
        input_tile_size,
        map_chunk_bytes,
        op.tokens,
        op.topk,
        capacity,
        static_cast<uint32_t>(op.assignment_addressed),
        rows_per_batch,
    };
    tt::tt_metal::TensorAccessorArgs(tensors.packed_y.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensors.slot_to_packed_row.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensors.slot_is_local.buffer()).append_to(reader_compile_args);
    const auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/topk_routed_expert_moe/device/kernels/"
        "dataflow/topk_local_combine_reader.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    std::vector<uint32_t> writer_compile_args = {
        CB_WEIGHT,
        CB_WRITER_VALID,
        CB_WEIGHT_SCRATCH,
        CB_OUTPUT,
        map_chunk_bytes,
        weight_page_size,
        weight_tile_size,
        output_tile_size,
        emb_dim_out_tiles,
        op.tokens,
        op.topk,
    };
    tt::tt_metal::TensorAccessorArgs(tensors.topk_weights.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensors.slot_is_local.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_args);
    const auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/topk_routed_expert_moe/device/kernels/"
        "dataflow/topk_local_combine_writer.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    const std::vector<uint32_t> compute_compile_args = {
        CB_PACKED_ROW,
        CB_WEIGHT,
        CB_ROW_MAJOR_ACCUM,
        CB_OUTPUT,
        op.topk,
        emb_dim_cb_tiles,
        emb_dim_out_tiles,
    };
    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/topk_routed_expert_moe/device/kernels/"
        "compute/topk_local_combine.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_args});

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        const uint32_t token_start = chunk * TOKENS_PER_CHUNK;
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, cores[chunk], reader_runtime_args(tensors, token_start));
        tt::tt_metal::SetRuntimeArgs(
            program, writer_kernel, cores[chunk], writer_runtime_args(tensors, output, token_start));
    }

    return cached_program_t{
        std::move(program),
        TopkLocalCombineSharedVariables{
            .reader_kernel = reader_kernel,
            .writer_kernel = writer_kernel,
            .cores = std::move(cores),
        }};
}

void TopkLocalCombineProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TopkLocalCombineParams&,
    const TopkLocalCombineInputs& tensors,
    Tensor& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    for (const auto& core : shared.cores) {
        auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel, core);
        reader_args[0] = tensors.packed_y.buffer()->address();
        reader_args[1] = tensors.slot_to_packed_row.buffer()->address();
        reader_args[2] = tensors.slot_is_local.buffer()->address();

        auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel, core);
        writer_args[0] = tensors.topk_weights.buffer()->address();
        writer_args[1] = tensors.slot_is_local.buffer()->address();
        writer_args[2] = output.buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe

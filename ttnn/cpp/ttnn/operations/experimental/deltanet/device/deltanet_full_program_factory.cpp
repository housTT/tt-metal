// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_full_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "deltanet_full_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deltanet {
namespace full_factory {

constexpr uint32_t kTileSize = tt::constants::TILE_WIDTH;
constexpr auto kReaderPath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/reader_deltanet_full.cpp";
constexpr auto kComputePath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/compute/deltanet_full_compute.cpp";
constexpr auto kWriterPath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/writer_deltanet_full.cpp";

constexpr auto kCbStateIn = tt::CBIndex::c_0;
constexpr auto kCbQ = tt::CBIndex::c_1;
constexpr auto kCbK = tt::CBIndex::c_2;
constexpr auto kCbV = tt::CBIndex::c_3;
constexpr auto kCbDecay = tt::CBIndex::c_4;
constexpr auto kCbBeta = tt::CBIndex::c_5;
constexpr auto kCbOutput = tt::CBIndex::c_6;
constexpr auto kCbStateOut = tt::CBIndex::c_7;
constexpr auto kCbStateMid = tt::CBIndex::c_16;
constexpr auto kCbKT = tt::CBIndex::c_17;
constexpr auto kCbRawOut = tt::CBIndex::c_21;
constexpr auto kCbTmp0 = tt::CBIndex::c_24;
constexpr auto kCbTmp1 = tt::CBIndex::c_25;
constexpr auto kCbAcc = tt::CBIndex::c_26;

tt::tt_metal::CBHandle make_cb(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& cores,
    uint32_t cb_index,
    tt::DataFormat format,
    uint32_t num_tiles) {
    const uint32_t tile_size = tt::tile_size(format);
    auto config = tt::tt_metal::CircularBufferConfig(num_tiles * tile_size, {{cb_index, format}})
                      .set_page_size(cb_index, tile_size);
    return tt::tt_metal::CreateCircularBuffer(program, cores, config);
}

}  // namespace full_factory

DeltaNetDecodeFullProgramFactory::cached_program_t DeltaNetDecodeFullProgramFactory::create(
    const operation_attributes_t& attrs, const tensor_args_t& inputs, tensor_return_value_t& outputs) {
    using namespace tt::tt_metal;
    namespace ff = full_factory;

    const uint32_t num_heads = attrs.num_heads;
    const uint32_t k_head_dim_tiles = attrs.k_head_dim / ff::kTileSize;
    const uint32_t v_head_dim_tiles = attrs.v_head_dim / ff::kTileSize;
    const uint32_t state_tiles = k_head_dim_tiles * v_head_dim_tiles;

    auto* device = inputs.recurrent_state.device();
    Program program{};
    const auto data_format = datatype_to_dataformat_converter(inputs.qkv_proj.dtype());
    const auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(
        num_heads <= grid.x * grid.y,
        "DeltaNet decode needs {} cores for {} heads, grid is {}x{}",
        num_heads,
        num_heads,
        grid.x,
        grid.y);

    std::vector<CoreRange> core_ranges;
    core_ranges.reserve(num_heads);
    for (uint32_t head = 0; head < num_heads; ++head) {
        const CoreCoord core = {head % grid.x, head / grid.x};
        core_ranges.emplace_back(core, core);
    }
    const CoreRangeSet all_cores(core_ranges);

    ff::make_cb(program, all_cores, ff::kCbStateIn, data_format, state_tiles);
    ff::make_cb(program, all_cores, ff::kCbQ, data_format, k_head_dim_tiles);
    ff::make_cb(program, all_cores, ff::kCbK, data_format, k_head_dim_tiles);
    ff::make_cb(program, all_cores, ff::kCbV, data_format, v_head_dim_tiles);
    ff::make_cb(program, all_cores, ff::kCbDecay, data_format, 1);
    ff::make_cb(program, all_cores, ff::kCbBeta, data_format, 1);
    ff::make_cb(program, all_cores, ff::kCbOutput, data_format, v_head_dim_tiles);
    ff::make_cb(program, all_cores, ff::kCbStateOut, data_format, state_tiles);
    ff::make_cb(program, all_cores, ff::kCbStateMid, data_format, state_tiles);
    ff::make_cb(program, all_cores, ff::kCbKT, data_format, k_head_dim_tiles);
    ff::make_cb(program, all_cores, ff::kCbRawOut, data_format, v_head_dim_tiles);
    ff::make_cb(program, all_cores, ff::kCbTmp0, data_format, v_head_dim_tiles);
    ff::make_cb(program, all_cores, ff::kCbTmp1, data_format, std::max(k_head_dim_tiles, v_head_dim_tiles));
    ff::make_cb(program, all_cores, ff::kCbAcc, data_format, v_head_dim_tiles);

    auto* state_buffer = inputs.recurrent_state.buffer();
    auto* qkv_buffer = inputs.qkv_proj.buffer();
    auto* beta_buffer = inputs.beta.buffer();
    auto* decay_buffer = inputs.decay.buffer();

    std::vector<uint32_t> reader_compile_args = {
        static_cast<uint32_t>(ff::kCbStateIn),
        static_cast<uint32_t>(ff::kCbQ),
        static_cast<uint32_t>(ff::kCbK),
        static_cast<uint32_t>(ff::kCbV),
        static_cast<uint32_t>(ff::kCbDecay),
        static_cast<uint32_t>(ff::kCbBeta),
        static_cast<uint32_t>(ff::kCbKT),
        k_head_dim_tiles,
        v_head_dim_tiles,
    };
    TensorAccessorArgs(state_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(qkv_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(beta_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(decay_buffer).append_to(reader_compile_args);
    const auto reader_kernel =
        CreateKernel(program, ff::kReaderPath, all_cores, ReaderDataMovementConfig(reader_compile_args));

    const std::vector<uint32_t> compute_compile_args = {
        static_cast<uint32_t>(ff::kCbStateIn),
        static_cast<uint32_t>(ff::kCbQ),
        static_cast<uint32_t>(ff::kCbK),
        static_cast<uint32_t>(ff::kCbV),
        static_cast<uint32_t>(ff::kCbDecay),
        static_cast<uint32_t>(ff::kCbBeta),
        static_cast<uint32_t>(ff::kCbOutput),
        static_cast<uint32_t>(ff::kCbStateOut),
        static_cast<uint32_t>(ff::kCbTmp0),
        static_cast<uint32_t>(ff::kCbTmp1),
        static_cast<uint32_t>(ff::kCbAcc),
        k_head_dim_tiles,
        v_head_dim_tiles,
        static_cast<uint32_t>(ff::kCbStateMid),
        static_cast<uint32_t>(ff::kCbKT),
        static_cast<uint32_t>(ff::kCbRawOut),
    };
    const auto compute_kernel = CreateKernel(
        program,
        ff::kComputePath,
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .math_approx_mode = true,
            .compile_args = compute_compile_args,
        });

    auto* state_output_buffer = outputs[1].buffer();
    auto* output_buffer = outputs[0].buffer();
    std::vector<uint32_t> writer_compile_args = {
        static_cast<uint32_t>(ff::kCbStateOut),
        static_cast<uint32_t>(ff::kCbOutput),
        k_head_dim_tiles,
        v_head_dim_tiles,
    };
    TensorAccessorArgs(state_output_buffer).append_to(writer_compile_args);
    TensorAccessorArgs(output_buffer).append_to(writer_compile_args);
    const auto writer_kernel =
        CreateKernel(program, ff::kWriterPath, all_cores, WriterDataMovementConfig(writer_compile_args));

    const uint32_t key_dim_tiles = attrs.num_k_heads * k_head_dim_tiles;
    for (uint32_t head = 0; head < num_heads; ++head) {
        const CoreCoord core = {head % grid.x, head / grid.x};
        const uint32_t key_head = head / attrs.head_expand_ratio;
        const uint32_t q_tile = key_head * k_head_dim_tiles;
        const uint32_t k_tile = key_dim_tiles + key_head * k_head_dim_tiles;
        const uint32_t v_tile = 2 * key_dim_tiles + head * v_head_dim_tiles;

        SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {
                state_buffer->address(),
                qkv_buffer->address(),
                beta_buffer->address(),
                decay_buffer->address(),
                head * state_tiles,
                head,
                q_tile,
                k_tile,
                v_tile,
            });
        SetRuntimeArgs(
            program,
            writer_kernel,
            core,
            {
                state_output_buffer->address(),
                output_buffer->address(),
                head * state_tiles,
                head * v_head_dim_tiles,
            });
    }

    return cached_program_t{
        std::move(program),
        {
            .reader_kernel_id = reader_kernel,
            .compute_kernel_id = compute_kernel,
            .writer_kernel_id = writer_kernel,
            .all_cores = all_cores,
        }};
}

void DeltaNetDecodeFullProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attrs,
    const tensor_args_t& inputs,
    tensor_return_value_t& outputs) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;
    const auto grid = inputs.recurrent_state.device()->compute_with_storage_grid_size();
    auto& reader_runtime_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, shared.writer_kernel_id);

    for (uint32_t head = 0; head < attrs.num_heads; ++head) {
        const CoreCoord core = {head % grid.x, head / grid.x};
        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[0] = inputs.recurrent_state.buffer()->address();
        reader_args[1] = inputs.qkv_proj.buffer()->address();
        reader_args[2] = inputs.beta.buffer()->address();
        reader_args[3] = inputs.decay.buffer()->address();

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[0] = outputs[1].buffer()->address();
        writer_args[1] = outputs[0].buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deltanet

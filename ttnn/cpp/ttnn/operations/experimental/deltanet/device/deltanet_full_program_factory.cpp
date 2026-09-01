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
    const auto data_format = datatype_to_dataformat_converter(inputs.q.dtype());
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
    auto* q_buffer = inputs.q.buffer();
    auto* k_buffer = inputs.k.buffer();
    auto* v_buffer = inputs.v.buffer();
    auto* beta_buffer = inputs.beta.buffer();
    auto* decay_buffer = inputs.decay.buffer();
    auto* decay_scale_buffer = inputs.decay_scale.buffer();
    auto* dt_bias_buffer = inputs.dt_bias.buffer();

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
        static_cast<uint32_t>(attrs.preprocess_ab),
    };
    TensorAccessorArgs(state_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(q_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(k_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(v_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(beta_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(decay_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(decay_scale_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(dt_bias_buffer).append_to(reader_compile_args);
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

    const uint32_t batch_size =
        attrs.packed_qkv ? inputs.q.logical_shape()[-2] : inputs.q.logical_shape()[-3];
    const uint32_t heads_per_batch = attrs.num_heads / batch_size;
    const uint32_t k_heads_per_batch = attrs.num_k_heads / batch_size;
    const uint32_t packed_width_tiles =
        2 * k_heads_per_batch * k_head_dim_tiles + heads_per_batch * v_head_dim_tiles;
    for (uint32_t head = 0; head < num_heads; ++head) {
        const CoreCoord core = {head % grid.x, head / grid.x};
        const uint32_t batch = head / heads_per_batch;
        const uint32_t value_head = head % heads_per_batch;
        const uint32_t key_head = value_head / attrs.head_expand_ratio;
        const uint32_t packed_row_start = (batch / ff::kTileSize) * packed_width_tiles;
        const uint32_t q_tile = attrs.packed_qkv
                                    ? packed_row_start + key_head * k_head_dim_tiles
                                    : batch * k_head_dim_tiles;
        const uint32_t k_tile = attrs.packed_qkv
                                    ? packed_row_start + k_heads_per_batch * k_head_dim_tiles +
                                          key_head * k_head_dim_tiles
                                    : batch * k_head_dim_tiles;
        const uint32_t v_tile = attrs.packed_qkv
                                    ? packed_row_start + 2 * k_heads_per_batch * k_head_dim_tiles +
                                          value_head * v_head_dim_tiles
                                    : batch * v_head_dim_tiles;
        const uint32_t key_row = attrs.packed_qkv ? batch % ff::kTileSize : key_head;
        const uint32_t value_row = attrs.packed_qkv ? batch % ff::kTileSize : value_head;
        const uint32_t scalar_tile = (batch / ff::kTileSize) * ((heads_per_batch + ff::kTileSize - 1) / ff::kTileSize) +
                                     value_head / ff::kTileSize;

        SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {
                state_buffer->address(),
                q_buffer->address(),
                k_buffer->address(),
                v_buffer->address(),
                beta_buffer->address(),
                decay_buffer->address(),
                decay_scale_buffer->address(),
                dt_bias_buffer->address(),
                head * state_tiles,
                scalar_tile,
                batch % ff::kTileSize,
                value_head % ff::kTileSize,
                q_tile,
                k_tile,
                v_tile,
                key_row,
                value_row,
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
        reader_args[1] = inputs.q.buffer()->address();
        reader_args[2] = inputs.k.buffer()->address();
        reader_args[3] = inputs.v.buffer()->address();
        reader_args[4] = inputs.beta.buffer()->address();
        reader_args[5] = inputs.decay.buffer()->address();
        reader_args[6] = inputs.decay_scale.buffer()->address();
        reader_args[7] = inputs.dt_bias.buffer()->address();

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[0] = outputs[1].buffer()->address();
        writer_args[1] = outputs[0].buffer()->address();
    }
}

DeltaNetConv1dDecodeProgramFactory::cached_program_t DeltaNetConv1dDecodeProgramFactory::create(
    const operation_attributes_t& attrs, const tensor_args_t& inputs, tensor_return_value_t& outputs) {
    using namespace tt::tt_metal;
    namespace ff = full_factory;

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_state1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_state2 = tt::CBIndex::c_2;
    constexpr uint32_t cb_state3 = tt::CBIndex::c_3;
    constexpr uint32_t cb_tap0 = tt::CBIndex::c_4;
    constexpr uint32_t cb_tap1 = tt::CBIndex::c_5;
    constexpr uint32_t cb_tap2 = tt::CBIndex::c_6;
    constexpr uint32_t cb_tap3 = tt::CBIndex::c_7;
    constexpr uint32_t cb_output = tt::CBIndex::c_8;
    constexpr uint32_t cb_partial = tt::CBIndex::c_16;
    constexpr auto reader_path =
        "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/reader_deltanet_conv1d.cpp";
    constexpr auto compute_path =
        "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/compute/deltanet_conv1d_compute.cpp";
    constexpr auto writer_path =
        "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/writer_deltanet_conv1d.cpp";

    auto* device = inputs.input.device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_tiles = (attrs.q_width + attrs.k_width + attrs.v_width) / ff::kTileSize;
    TT_FATAL(
        num_tiles <= grid.x * grid.y,
        "DeltaNet conv1d decode needs {} cores for {} channel tiles, grid is {}x{}",
        num_tiles,
        num_tiles,
        grid.x,
        grid.y);

    std::vector<CoreRange> core_ranges;
    core_ranges.reserve(num_tiles);
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        const CoreCoord core = {tile % grid.x, tile / grid.x};
        core_ranges.emplace_back(core, core);
    }
    const CoreRangeSet all_cores(core_ranges);
    Program program{};
    const auto data_format = datatype_to_dataformat_converter(inputs.input.dtype());

    ff::make_cb(program, all_cores, cb_input, data_format, 1);
    ff::make_cb(program, all_cores, cb_state1, data_format, 1);
    ff::make_cb(program, all_cores, cb_state2, data_format, 1);
    ff::make_cb(program, all_cores, cb_state3, data_format, 1);
    ff::make_cb(program, all_cores, cb_tap0, data_format, 1);
    ff::make_cb(program, all_cores, cb_tap1, data_format, 1);
    ff::make_cb(program, all_cores, cb_tap2, data_format, 1);
    ff::make_cb(program, all_cores, cb_tap3, data_format, 1);
    ff::make_cb(program, all_cores, cb_output, data_format, 1);
    ff::make_cb(program, all_cores, cb_partial, data_format, 2);

    auto* input_buffer = inputs.input.buffer();
    auto* state0_buffer = inputs.state0.buffer();
    auto* state1_buffer = inputs.state1.buffer();
    auto* state2_buffer = inputs.state2.buffer();
    auto* state3_buffer = inputs.state3.buffer();
    auto* tap0_buffer = inputs.tap0.buffer();
    auto* tap1_buffer = inputs.tap1.buffer();
    auto* tap2_buffer = inputs.tap2.buffer();
    auto* tap3_buffer = inputs.tap3.buffer();
    auto* output_buffer = outputs[0].buffer();

    std::vector<uint32_t> reader_compile_args = {
        cb_input, cb_state1, cb_state2, cb_state3, cb_tap0, cb_tap1, cb_tap2, cb_tap3};
    TensorAccessorArgs(input_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(state0_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(state1_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(state2_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(state3_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(tap0_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(tap1_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(tap2_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(tap3_buffer).append_to(reader_compile_args);
    const auto reader_kernel =
        CreateKernel(program, reader_path, all_cores, ReaderDataMovementConfig(reader_compile_args));

    const std::vector<uint32_t> compute_compile_args = {
        cb_input, cb_state1, cb_state2, cb_state3, cb_tap0, cb_tap1, cb_tap2, cb_tap3, cb_partial, cb_output};
    const auto compute_kernel = CreateKernel(
        program,
        compute_path,
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_args,
        });

    std::vector<uint32_t> writer_compile_args = {cb_output};
    TensorAccessorArgs(output_buffer).append_to(writer_compile_args);
    const auto writer_kernel =
        CreateKernel(program, writer_path, all_cores, WriterDataMovementConfig(writer_compile_args));

    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        const CoreCoord core = {tile % grid.x, tile / grid.x};
        SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {
                input_buffer->address(),
                state0_buffer->address(),
                state1_buffer->address(),
                state2_buffer->address(),
                state3_buffer->address(),
                tap0_buffer->address(),
                tap1_buffer->address(),
                tap2_buffer->address(),
                tap3_buffer->address(),
                tile,
            });
        SetRuntimeArgs(program, writer_kernel, core, {output_buffer->address(), tile});
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

void DeltaNetConv1dDecodeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attrs,
    const tensor_args_t& inputs,
    tensor_return_value_t& outputs) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;
    const auto grid = inputs.input.device()->compute_with_storage_grid_size();
    const uint32_t num_tiles = (attrs.q_width + attrs.k_width + attrs.v_width) / full_factory::kTileSize;
    auto& reader_runtime_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, shared.writer_kernel_id);

    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        const CoreCoord core = {tile % grid.x, tile / grid.x};
        auto& reader_args = reader_runtime_args[core.x][core.y];
        reader_args[0] = inputs.input.buffer()->address();
        reader_args[1] = inputs.state0.buffer()->address();
        reader_args[2] = inputs.state1.buffer()->address();
        reader_args[3] = inputs.state2.buffer()->address();
        reader_args[4] = inputs.state3.buffer()->address();
        reader_args[5] = inputs.tap0.buffer()->address();
        reader_args[6] = inputs.tap1.buffer()->address();
        reader_args[7] = inputs.tap2.buffer()->address();
        reader_args[8] = inputs.tap3.buffer()->address();

        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[0] = outputs[0].buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deltanet

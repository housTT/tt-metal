// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hc_mix_post_device_operation.hpp"

#include <array>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {
constexpr const char* mix_op_name = "hc_mix_post";

void mix_check_tile_input(const Tensor& tensor, const char* name) {
    kda_factory_detail::check_allocated_device_tensor(tensor, mix_op_name, name);
    kda_factory_detail::check_layout(tensor, Layout::TILE, mix_op_name, name);
    kda_factory_detail::check_interleaved(tensor, mix_op_name, name);
}

void mix_check_shape(const Tensor& tensor, const Shape& expected, const char* name) {
    TT_FATAL(
        tensor.logical_shape() == expected,
        "{}: {} shape must be {}, got {}",
        mix_op_name,
        name,
        expected,
        tensor.logical_shape());
}
}  // namespace

HcMixPostOperation::program_factory_t HcMixPostOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return HcMixPostProgramFactory{};
}

void HcMixPostOperation::validate_on_program_cache_miss(const operation_attributes_t& a, const tensor_args_t& in) {
    TT_FATAL(a.streams > 0 && a.streams <= 16, "{}: streams must be in 1..16", mix_op_name);
    TT_FATAL(
        a.lowrank > 0 && a.lowrank % tt::constants::TILE_WIDTH == 0, "{}: lowrank must be tile aligned", mix_op_name);
    TT_FATAL(a.width > 0 && a.width % tt::constants::TILE_WIDTH == 0, "{}: width must be tile aligned", mix_op_name);
    mix_check_tile_input(in.packed, "packed");
    mix_check_tile_input(in.weighted, "weighted");
    mix_check_tile_input(in.up, "up");
    kda_factory_detail::check_dtype(in.packed, DataType::BFLOAT16, mix_op_name, "packed");
    kda_factory_detail::check_dtype(in.weighted, DataType::BFLOAT16, mix_op_name, "weighted");
    TT_FATAL(
        in.up.dtype() == DataType::BFLOAT16 || in.up.dtype() == DataType::BFLOAT8_B,
        "{}: up must be BFLOAT16 or BFLOAT8_B",
        mix_op_name);
    // packed is [1,1,rows,L+S] (rows = partial copies of one row, summed) or the
    // stream-blocked [1,1,rows,S*Lp] with Lp = tile-padded L+S (column block s =
    // stream s, row r belongs to stream r % S), as produced by one matmul against
    // the stream-major down weight followed by an all-gather over ranks.
    const auto& packed_shape = in.packed.logical_shape();
    const uint32_t lp = tt::round_up(a.lowrank + a.streams, tt::constants::TILE_WIDTH);
    TT_FATAL(
        packed_shape.rank() == 4 && packed_shape[0] >= 1 && packed_shape[0] <= 8 && packed_shape[1] == 1 &&
            packed_shape[2] >= 1 && packed_shape[2] <= tt::constants::TILE_HEIGHT &&
            (packed_shape[3] == a.lowrank + a.streams || packed_shape[3] == a.streams * lp),
        "{}: packed must be [R<=8,1,rows<=32,L+S] or [R<=8,1,rows<=32,S*Lp], got {}",
        mix_op_name,
        packed_shape);
    if (packed_shape[3] == a.streams * lp) {
        TT_FATAL(
            packed_shape[2] % a.streams == 0, "{}: stream-blocked packed rows must be a multiple of S", mix_op_name);
    }
    mix_check_shape(in.weighted, Shape({1, 1, a.streams, a.width}), "weighted");
    mix_check_shape(in.up, Shape({1, 1, a.lowrank, a.streams * a.width}), "up");
    kda_factory_detail::check_same_device(in.packed, in.weighted, mix_op_name, "weighted");
    kda_factory_detail::check_same_device(in.packed, in.up, mix_op_name, "up");
    kda_factory_detail::check_output_interleaved(a.output_mem_config, mix_op_name);
    kda_factory_detail::check_compute_config(a.compute_kernel_config, mix_op_name);
}

HcMixPostOperation::spec_return_value_t HcMixPostOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    return {
        TensorSpec(
            Shape({1, 1, 1, a.width}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), a.output_mem_config)),
        TensorSpec(
            Shape({1, 1, 1, a.streams}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), a.output_mem_config)),
    };
}

HcMixPostOperation::tensor_return_value_t HcMixPostOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    auto specs = compute_output_specs(a, in);
    return {create_device_tensor(specs[0], in.packed.device()), create_device_tensor(specs[1], in.packed.device())};
}

std::tuple<Tensor, Tensor> hc_mix_post(
    const Tensor& packed,
    const Tensor& weighted,
    const Tensor& up,
    uint32_t lowrank,
    uint32_t streams,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& shape = weighted.logical_shape();
    TT_FATAL(shape.rank() == 4, "{}: weighted must be rank 4", mix_op_name);
    auto outputs = ttnn::device_operation::launch<HcMixPostOperation>(
        HcMixPostParams{
            .streams = streams,
            .lowrank = lowrank,
            .width = shape[3],
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        HcMixPostInputs{.packed = packed, .weighted = weighted, .up = up});
    return {outputs[0], outputs[1]};
}

}  // namespace ttnn::experimental::prim

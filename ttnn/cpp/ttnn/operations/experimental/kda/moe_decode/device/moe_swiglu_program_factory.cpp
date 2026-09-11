// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_swiglu_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {
namespace m2 = tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts MoeSwigluProgramFactory::create_program_artifacts(
    const MoeSwigluParams& attrs, const MoeSwigluInputs& in, Tensor& output) {
    const auto& gate_up = in.gate_up.mesh_tensor();
    const auto& out = output.mesh_tensor();
    const auto& device = gate_up.device();
    const auto arch = device.arch();
    const uint32_t It = attrs.intermediate / TILE_WIDTH;
    const uint32_t total = attrs.groups * It;  // one work item per output tile
    auto dist = kda_factory_detail::distribute_prep(device.compute_with_storage_grid_size(), total, total);

    const m2::KernelSpecName READER{"reader"}, WRITER{"writer"}, COMPUTE{"compute"};
    const m2::DFBSpecName GATE{"gate"}, UP{"up"}, OUT{"out"};
    const m2::TensorParamName IN_T{"gate_up"}, OUT_T{"out"};
    const auto BF16 = tt::DataFormat::Float16_b;
    auto dfb = [](const m2::DFBSpecName& name, uint32_t tiles, tt::DataFormat format) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = tiles,
            .data_format_metadata = format};
    };
    m2::Group<m2::DataflowBufferSpec> dfbs = {dfb(GATE, 4, BF16), dfb(UP, 4, BF16), dfb(OUT, 4, BF16)};
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/moe_decode/device/kernels/dataflow/reader_moe_swiglu.cpp",
        .dfb_bindings = {{GATE, "gate", m2::DFBEndpointType::PRODUCER}, {UP, "up", m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {{IN_T, "gate_up"}},
        .compile_time_args = {{"It", It}},
        .runtime_arg_schema = {.runtime_arg_names = {"tile_start", "tile_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/moe_decode/device/kernels/dataflow/writer_moe_swiglu.cpp",
        .dfb_bindings = {{OUT, "out", m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {{OUT_T, "out"}},
        .compile_time_args = {},
        .runtime_arg_schema = {.runtime_arg_names = {"tile_start", "tile_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };
    auto compute_hw = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);
    auto& unpack_modes = m2::unpack_modes(compute_hw);
    for (const auto& name : {GATE, UP, OUT}) {
        unpack_modes[name] = UnpackMode::UnpackToSrc;
    }
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/moe_decode/device/kernels/compute/moe_swiglu.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {{GATE, "gate", m2::DFBEndpointType::CONSUMER},
             {UP, "up", m2::DFBEndpointType::CONSUMER},
             {OUT, "out", m2::DFBEndpointType::PRODUCER}},
        .compile_time_args = {},
        .runtime_arg_schema = {.runtime_arg_names = {"tile_count"}},
        .hw_config = std::move(compute_hw),
    };
    m2::KernelRunArgs reader_args{.kernel = READER}, writer_args{.kernel = WRITER}, compute_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        m2::AddRuntimeArgsForNode(
            reader_args.runtime_arg_values, core, {{"tile_start", dist.wi_start[i]}, {"tile_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(
            writer_args.runtime_arg_values, core, {{"tile_start", dist.wi_start[i]}, {"tile_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"tile_count", dist.wi_count[i]}});
    }
    m2::ProgramSpec spec{
        .name = "moe_swiglu",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {{.unique_id = IN_T, .spec = gate_up.tensor_spec()}, {.unique_id = OUT_T, .spec = out.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = dist.core_set}},
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args), std::move(compute_args)};
    run_args.tensor_args = {{IN_T, gate_up}, {OUT_T, out}};
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim

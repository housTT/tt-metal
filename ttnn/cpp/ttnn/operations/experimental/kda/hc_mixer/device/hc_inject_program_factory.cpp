// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hc_inject_program_factory.hpp"

#include <cstring>

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

ttnn::device_operation::ProgramArtifacts HcInjectProgramFactory::create_program_artifacts(
    const HcInjectParams& attrs, const HcInjectInputs& in, Tensor& output) {
    const auto& hyper = in.hyper.mesh_tensor();
    const auto& block = in.block.mesh_tensor();
    const auto& injection = in.injection.mesh_tensor();
    const auto& out = output.mesh_tensor();
    const auto& device = hyper.device();
    const auto arch = device.arch();

    const uint32_t S = attrs.streams;
    const uint32_t Nt = attrs.width / TILE_WIDTH;
    auto dist = kda_factory_detail::distribute_prep(device.compute_with_storage_grid_size(), Nt, Nt);
    const float two = 2.0f;
    uint32_t two_bits = 0;
    std::memcpy(&two_bits, &two, sizeof(float));

    const m2::KernelSpecName READER{"reader"}, WRITER{"writer"}, COMPUTE{"compute"};
    const m2::DFBSpecName INJ{"inj"}, P4{"p4"}, GCOL{"gcol"}, BLOCK{"block"}, HYPER{"hyper"}, B4{"b4"},
        SCALED{"scaled"}, OUT{"out"};
    const m2::TensorParamName HYPER_T{"hyper"}, BLOCK_T{"block"}, INJ_T{"injection"}, OUT_T{"out"};
    auto dfb = [](const m2::DFBSpecName& name, uint32_t tiles, tt::DataFormat format) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = tiles,
            .data_format_metadata = format};
    };
    const auto BF16 = tt::DataFormat::Float16_b;
    const auto F32 = tt::DataFormat::Float32;
    m2::Group<m2::DataflowBufferSpec> dfbs = {
        dfb(INJ, 1, BF16),
        dfb(P4, 1, BF16),
        dfb(GCOL, 1, F32),
        dfb(BLOCK, 2, BF16),
        dfb(HYPER, 2, BF16),
        dfb(B4, 1, F32),
        dfb(SCALED, 1, F32),
        dfb(OUT, 2, BF16),
    };
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/hc_mixer/device/kernels/dataflow/reader_hc_inject.cpp",
        .dfb_bindings =
            {{INJ, "inj", m2::DFBEndpointType::PRODUCER},
             {P4, "p4", m2::DFBEndpointType::PRODUCER},
             {BLOCK, "block", m2::DFBEndpointType::PRODUCER},
             {HYPER, "hyper", m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {{HYPER_T, "hyper"}, {BLOCK_T, "block"}, {INJ_T, "injection"}},
        .compile_time_args = {{"S", S}},
        .runtime_arg_schema = {.runtime_arg_names = {"col_start", "col_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/hc_mixer/device/kernels/dataflow/writer_hc_inject.cpp",
        .dfb_bindings = {{OUT, "out", m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {{OUT_T, "out"}},
        .compile_time_args = {},
        .runtime_arg_schema = {.runtime_arg_names = {"col_start", "col_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };
    auto compute_hw = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);
    auto& unpack_modes = m2::unpack_modes(compute_hw);
    for (const auto& name : {INJ, P4, GCOL, BLOCK, HYPER, B4, SCALED, OUT}) {
        unpack_modes[name] = UnpackMode::UnpackToSrc;
    }
    std::vector<m2::DFBBinding> compute_bindings = {
        {INJ, "inj", m2::DFBEndpointType::CONSUMER},
        {P4, "p4", m2::DFBEndpointType::CONSUMER},
        {BLOCK, "block", m2::DFBEndpointType::CONSUMER},
        {HYPER, "hyper", m2::DFBEndpointType::CONSUMER},
        {OUT, "out", m2::DFBEndpointType::PRODUCER}};
    for (const auto& [name, id] :
         std::vector<std::pair<m2::DFBSpecName, const char*>>{{GCOL, "gcol"}, {B4, "b4"}, {SCALED, "scaled"}}) {
        compute_bindings.push_back({name, id, m2::DFBEndpointType::PRODUCER});
        compute_bindings.push_back({name, id, m2::DFBEndpointType::CONSUMER});
    }
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/hc_mixer/device/kernels/compute/hc_inject.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_bindings),
        .compile_time_args = {{"TwoBits", two_bits}},
        .runtime_arg_schema = {.runtime_arg_names = {"col_count"}},
        .hw_config = std::move(compute_hw),
    };
    m2::KernelRunArgs reader_args{.kernel = READER}, writer_args{.kernel = WRITER}, compute_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        m2::AddRuntimeArgsForNode(
            reader_args.runtime_arg_values, core, {{"col_start", dist.wi_start[i]}, {"col_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(
            writer_args.runtime_arg_values, core, {{"col_start", dist.wi_start[i]}, {"col_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"col_count", dist.wi_count[i]}});
    }
    m2::ProgramSpec spec{
        .name = "hc_inject",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {{.unique_id = HYPER_T, .spec = hyper.tensor_spec()},
             {.unique_id = BLOCK_T, .spec = block.tensor_spec()},
             {.unique_id = INJ_T, .spec = injection.tensor_spec()},
             {.unique_id = OUT_T, .spec = out.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = dist.core_set}},
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args), std::move(compute_args)};
    run_args.tensor_args = {{HYPER_T, hyper}, {BLOCK_T, block}, {INJ_T, injection}, {OUT_T, out}};
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim

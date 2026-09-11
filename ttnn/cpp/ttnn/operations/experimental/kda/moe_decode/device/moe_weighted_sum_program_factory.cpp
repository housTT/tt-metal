// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_weighted_sum_program_factory.hpp"

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

ttnn::device_operation::ProgramArtifacts MoeWeightedSumProgramFactory::create_program_artifacts(
    const MoeWeightedSumParams& attrs, const MoeWeightedSumInputs& in, Tensor& output) {
    const auto& groups = in.groups.mesh_tensor();
    const auto& scores = in.scores.mesh_tensor();
    const auto& out = output.mesh_tensor();
    const auto& device = groups.device();
    const auto arch = device.arch();
    const uint32_t K = attrs.k;
    const uint32_t Nt = attrs.width / TILE_WIDTH;
    auto dist = kda_factory_detail::distribute_prep(device.compute_with_storage_grid_size(), Nt, Nt);

    const m2::KernelSpecName READER{"reader"}, WRITER{"writer"}, COMPUTE{"compute"};
    const m2::DFBSpecName P{"p"}, W{"w"}, G{"g"}, M{"m"}, OUT{"out"};
    const m2::TensorParamName GROUPS_T{"groups"}, SCORES_T{"scores"}, OUT_T{"out"};
    const auto BF16 = tt::DataFormat::Float16_b;
    const auto F32 = tt::DataFormat::Float32;
    auto dfb = [](const m2::DFBSpecName& name, uint32_t tiles, tt::DataFormat format) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = tiles,
            .data_format_metadata = format};
    };
    m2::Group<m2::DataflowBufferSpec> dfbs = {
        dfb(P, K, BF16), dfb(W, 1, BF16), dfb(G, 2 * K, BF16), dfb(M, 1, F32), dfb(OUT, 2, BF16)};
    m2::KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/moe_decode/device/kernels/dataflow/reader_moe_weighted_sum.cpp",
        .dfb_bindings =
            {{P, "p", m2::DFBEndpointType::PRODUCER},
             {W, "w", m2::DFBEndpointType::PRODUCER},
             {G, "g", m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {{GROUPS_T, "groups"}, {SCORES_T, "scores"}},
        .compile_time_args = {{"K", K}, {"Nt", Nt}},
        .runtime_arg_schema = {.runtime_arg_names = {"col_start", "col_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/moe_decode/device/kernels/dataflow/writer_moe_weighted_sum.cpp",
        .dfb_bindings = {{OUT, "out", m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {{OUT_T, "out"}},
        .compile_time_args = {},
        .runtime_arg_schema = {.runtime_arg_names = {"col_start", "col_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };
    auto compute_hw = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);
    auto& unpack_modes = m2::unpack_modes(compute_hw);
    for (const auto& name : {P, W, G, M, OUT}) {
        unpack_modes[name] = UnpackMode::UnpackToSrc;
    }
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/moe_decode/device/kernels/compute/moe_weighted_sum.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {{P, "p", m2::DFBEndpointType::CONSUMER},
             {W, "w", m2::DFBEndpointType::CONSUMER},
             {G, "g", m2::DFBEndpointType::CONSUMER},
             {M, "m", m2::DFBEndpointType::PRODUCER},
             {M, "m", m2::DFBEndpointType::CONSUMER},
             {OUT, "out", m2::DFBEndpointType::PRODUCER}},
        .compile_time_args = {{"K", K}},
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
        .name = "moe_weighted_sum",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {{.unique_id = GROUPS_T, .spec = groups.tensor_spec()},
             {.unique_id = SCORES_T, .spec = scores.tensor_spec()},
             {.unique_id = OUT_T, .spec = out.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = dist.core_set}},
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args), std::move(compute_args)};
    run_args.tensor_args = {{GROUPS_T, groups}, {SCORES_T, scores}, {OUT_T, out}};
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim

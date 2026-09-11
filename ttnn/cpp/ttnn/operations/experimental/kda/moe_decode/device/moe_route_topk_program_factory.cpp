// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_route_topk_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {
namespace m2 = tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts MoeRouteTopkProgramFactory::create_program_artifacts(
    const MoeRouteTopkParams& attrs, const MoeRouteTopkInputs& in, std::vector<Tensor>& outputs) {
    const auto& logits = in.logits.mesh_tensor();
    const auto& rank_base = in.rank_base.mesh_tensor();
    const auto& slots = outputs[0].mesh_tensor();
    const auto& scores = outputs[1].mesh_tensor();
    const auto& device = logits.device();
    const auto arch = device.arch();
    const uint32_t Et = attrs.num_experts / TILE_WIDTH;
    auto dist = kda_factory_detail::distribute_prep(device.compute_with_storage_grid_size(), 1, 1);

    const m2::KernelSpecName READER{"reader"};
    const m2::DFBSpecName LOGITS{"logits"}, BASE{"base"}, SLOTS{"slots"}, SCORES{"scores"};
    const m2::TensorParamName LOGITS_T{"logits"}, BASE_T{"rank_base"}, SLOTS_T{"slots"}, SCORES_T{"scores"};
    m2::Group<m2::DataflowBufferSpec> dfbs = {
        m2::DataflowBufferSpec{
            .unique_id = LOGITS,
            .entry_size = tt::tile_size(tt::DataFormat::Float16_b),
            .num_entries = Et,
            .data_format_metadata = tt::DataFormat::Float16_b},
        m2::DataflowBufferSpec{
            .unique_id = BASE, .entry_size = 64, .num_entries = 1, .data_format_metadata = tt::DataFormat::Int32},
        m2::DataflowBufferSpec{
            .unique_id = SLOTS, .entry_size = 64, .num_entries = 1, .data_format_metadata = tt::DataFormat::UInt16},
        m2::DataflowBufferSpec{
            .unique_id = SCORES,
            .entry_size = tt::tile_size(tt::DataFormat::Float16_b),
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::Float16_b},
    };
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/moe_decode/device/kernels/dataflow/moe_route_topk.cpp",
        // single-kernel program: the kernel is both producer and consumer of its scratch buffers
        .dfb_bindings =
            {{LOGITS, "logits", m2::DFBEndpointType::PRODUCER},
             {LOGITS, "logits", m2::DFBEndpointType::CONSUMER},
             {BASE, "base", m2::DFBEndpointType::PRODUCER},
             {BASE, "base", m2::DFBEndpointType::CONSUMER},
             {SLOTS, "slots", m2::DFBEndpointType::PRODUCER},
             {SLOTS, "slots", m2::DFBEndpointType::CONSUMER},
             {SCORES, "scores", m2::DFBEndpointType::PRODUCER},
             {SCORES, "scores", m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {{LOGITS_T, "logits"}, {BASE_T, "rank_base"}, {SLOTS_T, "slots"}, {SCORES_T, "scores"}},
        .compile_time_args =
            {{"K", attrs.k},
             {"Et", Et},
             {"LocalExperts", attrs.local_experts},
             {"SlotBytes", static_cast<uint32_t>(slots.tensor_spec().compute_page_size_bytes())}},
        .runtime_arg_schema = {.runtime_arg_names = {}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };
    m2::KernelRunArgs reader_args{.kernel = READER};
    for (const auto& core : dist.cores) {
        m2::AddRuntimeArgsForNode(reader_args.runtime_arg_values, core, {});
    }
    m2::ProgramSpec spec{
        .name = "moe_route_topk",
        .kernels = {std::move(reader)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {{.unique_id = LOGITS_T, .spec = logits.tensor_spec()},
             {.unique_id = BASE_T, .spec = rank_base.tensor_spec()},
             {.unique_id = SLOTS_T, .spec = slots.tensor_spec()},
             {.unique_id = SCORES_T, .spec = scores.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {READER}, .target_nodes = dist.core_set}},
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_args)};
    run_args.tensor_args = {{LOGITS_T, logits}, {BASE_T, rank_base}, {SLOTS_T, slots}, {SCORES_T, scores}};
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim

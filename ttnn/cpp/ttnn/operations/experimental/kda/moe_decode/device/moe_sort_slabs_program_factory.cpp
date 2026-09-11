// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_sort_slabs_program_factory.hpp"

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

ttnn::device_operation::ProgramArtifacts MoeSortSlabsProgramFactory::create_program_artifacts(
    const MoeSortSlabsParams& attrs, const MoeSortSlabsInputs& in, std::vector<Tensor>& outputs) {
    const auto& indices = in.indices.mesh_tensor();
    const auto& scores = in.scores.mesh_tensor();
    const auto& rank_base = in.rank_base.mesh_tensor();
    const auto& slab_rows = outputs[0].mesh_tensor();
    const auto& slab_experts = outputs[1].mesh_tensor();
    const auto& slab_pos = outputs[2].mesh_tensor();
    const auto& local_scores = outputs[3].mesh_tensor();
    const auto& device = indices.device();
    const auto arch = device.arch();
    const uint32_t Rt = attrs.rows / TILE_HEIGHT;
    const uint32_t P = attrs.slab_capacity;
    auto dist = kda_factory_detail::distribute_prep(device.compute_with_storage_grid_size(), 1, 1);

    const m2::KernelSpecName READER{"reader"};
    const m2::DFBSpecName IDX{"idx"}, SCR{"scr"}, BASE{"base"}, ROWS{"rows_out"}, EXP{"experts_out"}, POS{"pos_out"},
        LSC{"scores_out"}, SCRATCH{"scratch"};
    const m2::TensorParamName IDX_T{"indices"}, SCR_T{"scores"}, BASE_T{"rank_base"}, ROWS_T{"slab_rows"},
        EXP_T{"slab_experts"}, POS_T{"slab_pos"}, LSC_T{"local_scores"};
    const uint32_t entries = attrs.rows * attrs.k;
    m2::Group<m2::DataflowBufferSpec> dfbs = {
        m2::DataflowBufferSpec{
            .unique_id = IDX,
            .entry_size = tt::tile_size(tt::DataFormat::UInt16),
            .num_entries = Rt,
            .data_format_metadata = tt::DataFormat::UInt16},
        m2::DataflowBufferSpec{
            .unique_id = SCR,
            .entry_size = tt::tile_size(tt::DataFormat::Float16_b),
            .num_entries = Rt,
            .data_format_metadata = tt::DataFormat::Float16_b},
        m2::DataflowBufferSpec{
            .unique_id = BASE, .entry_size = 64, .num_entries = 1, .data_format_metadata = tt::DataFormat::Int32},
        m2::DataflowBufferSpec{
            .unique_id = ROWS,
            .entry_size = tt::round_up(P * TILE_HEIGHT * 4u, 64u),
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt32},
        m2::DataflowBufferSpec{
            .unique_id = EXP,
            .entry_size = tt::round_up(P * 2u, 64u),
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt16},
        m2::DataflowBufferSpec{
            .unique_id = POS,
            .entry_size = tt::tile_size(tt::DataFormat::Int32),
            .num_entries = Rt,
            .data_format_metadata = tt::DataFormat::Int32},
        m2::DataflowBufferSpec{
            .unique_id = LSC,
            .entry_size = tt::tile_size(tt::DataFormat::Float16_b),
            .num_entries = Rt,
            .data_format_metadata = tt::DataFormat::Float16_b},
        // per-expert counters/offsets (2 x local_experts u32) + per-entry expert ids (entries u16)
        m2::DataflowBufferSpec{
            .unique_id = SCRATCH,
            .entry_size = tt::round_up(attrs.local_experts * 8u + entries * 2u, 64u),
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt32},
    };
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/moe_decode/device/kernels/dataflow/moe_sort_slabs.cpp",
        // single-kernel program: the kernel is both producer and consumer of its scratch buffers
        .dfb_bindings =
            {{IDX, "idx", m2::DFBEndpointType::PRODUCER},
             {IDX, "idx", m2::DFBEndpointType::CONSUMER},
             {SCR, "scr", m2::DFBEndpointType::PRODUCER},
             {SCR, "scr", m2::DFBEndpointType::CONSUMER},
             {BASE, "base", m2::DFBEndpointType::PRODUCER},
             {BASE, "base", m2::DFBEndpointType::CONSUMER},
             {ROWS, "rows_out", m2::DFBEndpointType::PRODUCER},
             {ROWS, "rows_out", m2::DFBEndpointType::CONSUMER},
             {EXP, "experts_out", m2::DFBEndpointType::PRODUCER},
             {EXP, "experts_out", m2::DFBEndpointType::CONSUMER},
             {POS, "pos_out", m2::DFBEndpointType::PRODUCER},
             {POS, "pos_out", m2::DFBEndpointType::CONSUMER},
             {LSC, "scores_out", m2::DFBEndpointType::PRODUCER},
             {LSC, "scores_out", m2::DFBEndpointType::CONSUMER},
             {SCRATCH, "scratch", m2::DFBEndpointType::PRODUCER},
             {SCRATCH, "scratch", m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings =
            {{IDX_T, "indices"},
             {SCR_T, "scores"},
             {BASE_T, "rank_base"},
             {ROWS_T, "slab_rows"},
             {EXP_T, "slab_experts"},
             {POS_T, "slab_pos"},
             {LSC_T, "local_scores"}},
        .compile_time_args =
            {{"Rt", Rt},
             {"K", attrs.k},
             {"LocalExperts", attrs.local_experts},
             {"P", P},
             {"RowsBytes", static_cast<uint32_t>(slab_rows.tensor_spec().compute_page_size_bytes())},
             {"ExpertsBytes", static_cast<uint32_t>(slab_experts.tensor_spec().compute_page_size_bytes())}},
        .runtime_arg_schema = {.runtime_arg_names = {}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };
    m2::KernelRunArgs reader_args{.kernel = READER};
    for (const auto& core : dist.cores) {
        m2::AddRuntimeArgsForNode(reader_args.runtime_arg_values, core, {});
    }
    m2::ProgramSpec spec{
        .name = "moe_sort_slabs",
        .kernels = {std::move(reader)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {{.unique_id = IDX_T, .spec = indices.tensor_spec()},
             {.unique_id = SCR_T, .spec = scores.tensor_spec()},
             {.unique_id = BASE_T, .spec = rank_base.tensor_spec()},
             {.unique_id = ROWS_T, .spec = slab_rows.tensor_spec()},
             {.unique_id = EXP_T, .spec = slab_experts.tensor_spec()},
             {.unique_id = POS_T, .spec = slab_pos.tensor_spec()},
             {.unique_id = LSC_T, .spec = local_scores.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {READER}, .target_nodes = dist.core_set}},
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_args)};
    run_args.tensor_args = {
        {IDX_T, indices},
        {SCR_T, scores},
        {BASE_T, rank_base},
        {ROWS_T, slab_rows},
        {EXP_T, slab_experts},
        {POS_T, slab_pos},
        {LSC_T, local_scores}};
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim

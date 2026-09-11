// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hc_mix_post_program_factory.hpp"

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

namespace {
uint32_t bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    return bits >> 16;
}
}  // namespace

ttnn::device_operation::ProgramArtifacts HcMixPostProgramFactory::create_program_artifacts(
    const HcMixPostParams& attrs, const HcMixPostInputs& in, std::vector<Tensor>& outputs) {
    const auto& packed = in.packed.mesh_tensor();
    const auto& weighted = in.weighted.mesh_tensor();
    const auto& up = in.up.mesh_tensor();
    const auto& mixed = outputs[0].mesh_tensor();
    const auto& injection = outputs[1].mesh_tensor();
    const auto& device = packed.device();
    const auto arch = device.arch();

    const uint32_t S = attrs.streams;
    // packed = [1, 1, Rows, PB * Lp] with PB == S for the stream-blocked form (column block
    // b = stream b, row r belongs to stream r % S) or PB == 1 when every row is a partial of
    // the same row.  The kernel reduces the rows with selector matmuls.
    const uint32_t Lt = attrs.lowrank / TILE_WIDTH;
    const uint32_t PR = in.packed.logical_shape()[0];  // rank batches (tile-aligned all-gather along dim 0)
    const uint32_t PRows = in.packed.logical_shape()[2];
    const uint32_t PB = in.packed.logical_shape()[3] == attrs.lowrank + attrs.streams ? 1u : S;
    const uint32_t Nt = attrs.width / TILE_WIDTH;
    const auto up_format = datatype_to_dataformat_converter(in.up.dtype());
    auto dist = kda_factory_detail::distribute_prep(device.compute_with_storage_grid_size(), Nt, Nt);

    const m2::KernelSpecName READER{"reader"}, WRITER{"writer"}, COMPUTE{"compute"};
    const m2::DFBSpecName LOW{"low"}, LOW_ACT{"low_act"}, P{"p"}, R{"r"}, W{"w"}, WGT{"wgt"}, M{"m"}, GATE4{"gate4"},
        PROD{"prod"}, OUT{"out"}, INJ_IN{"inj_in"}, INJ_OUT{"inj_out"}, PACKED{"packed"}, SEL{"sel"},
        LOW_SUM{"low_sum"};
    const m2::TensorParamName PACKED_T{"packed"}, WEIGHTED_T{"weighted"}, UP_T{"up"}, MIXED_T{"mixed"},
        INJ_T{"injection"};

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
        dfb(LOW, Lt, BF16),
        dfb(LOW_ACT, Lt, F32),
        dfb(P, S, BF16),
        dfb(R, 1, BF16),
        dfb(W, S * Lt, up_format),
        dfb(WGT, 1, BF16),
        dfb(M, S, F32),
        dfb(GATE4, 1, F32),
        dfb(PROD, 1, F32),
        dfb(OUT, 1, BF16),
        dfb(INJ_IN, 1, BF16),
        dfb(INJ_OUT, 1, BF16),
        dfb(PACKED, PR * PB * (Lt + 1), BF16),
        dfb(SEL, PB, BF16),
        dfb(LOW_SUM, Lt + 1, BF16),
    };

    m2::KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/hc_mixer/device/kernels/dataflow/reader_hc_mix_post.cpp",
        .dfb_bindings =
            {{PACKED, "packed", m2::DFBEndpointType::PRODUCER},
             {SEL, "sel", m2::DFBEndpointType::PRODUCER},
             {P, "p", m2::DFBEndpointType::PRODUCER},
             {R, "r", m2::DFBEndpointType::PRODUCER},
             {W, "w", m2::DFBEndpointType::PRODUCER},
             {WGT, "wgt", m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {{PACKED_T, "packed"}, {WEIGHTED_T, "weighted"}, {UP_T, "up"}},
        .compile_time_args =
            {{"Lt", Lt},
             {"Nt", Nt},
             {"S", S},
             {"PB", PB},
             {"PR", PR},
             {"PRows", PRows},
             {"InvStreamsBf16", bf16_bits(1.0f / static_cast<float>(S))}},
        .runtime_arg_schema = {.runtime_arg_names = {"col_start", "col_count", "emit_injection"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/hc_mixer/device/kernels/dataflow/writer_hc_mix_post.cpp",
        .dfb_bindings =
            {{OUT, "out", m2::DFBEndpointType::CONSUMER}, {INJ_OUT, "inj_out", m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {{MIXED_T, "mixed"}, {INJ_T, "injection"}},
        .compile_time_args = {},
        .runtime_arg_schema = {.runtime_arg_names = {"col_start", "col_count", "emit_injection"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };
    auto compute_hw = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);
    auto& unpack_modes = m2::unpack_modes(compute_hw);
    for (const auto& name : {LOW, LOW_ACT, P, R, W, WGT, M, GATE4, PROD, OUT, INJ_IN, INJ_OUT, PACKED, SEL, LOW_SUM}) {
        unpack_modes[name] = UnpackMode::UnpackToSrc;
    }
    std::vector<m2::DFBBinding> compute_bindings = {
        {PACKED, "packed", m2::DFBEndpointType::CONSUMER},
        {SEL, "sel", m2::DFBEndpointType::CONSUMER},
        {P, "p", m2::DFBEndpointType::CONSUMER},
        {R, "r", m2::DFBEndpointType::CONSUMER},
        {W, "w", m2::DFBEndpointType::CONSUMER},
        {WGT, "wgt", m2::DFBEndpointType::CONSUMER},
        {OUT, "out", m2::DFBEndpointType::PRODUCER},
        {INJ_OUT, "inj_out", m2::DFBEndpointType::PRODUCER}};
    for (const auto& [name, id] : std::vector<std::pair<m2::DFBSpecName, const char*>>{
             {LOW_SUM, "low_sum"},
             {LOW, "low"},
             {INJ_IN, "inj_in"},
             {LOW_ACT, "low_act"},
             {M, "m"},
             {GATE4, "gate4"},
             {PROD, "prod"}}) {
        compute_bindings.push_back({name, id, m2::DFBEndpointType::PRODUCER});
        compute_bindings.push_back({name, id, m2::DFBEndpointType::CONSUMER});
    }
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/hc_mixer/device/kernels/compute/hc_mix_post.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_bindings),
        .compile_time_args = {{"Lt", Lt}, {"S", S}, {"PB", PB}, {"PR", PR}},
        .runtime_arg_schema = {.runtime_arg_names = {"col_count", "emit_injection"}},
        .hw_config = std::move(compute_hw),
    };

    m2::KernelRunArgs reader_args{.kernel = READER}, writer_args{.kernel = WRITER}, compute_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        const uint32_t emit = dist.wi_start[i] == 0 && dist.wi_count[i] > 0 ? 1u : 0u;
        m2::AddRuntimeArgsForNode(
            reader_args.runtime_arg_values,
            core,
            {{"col_start", dist.wi_start[i]}, {"col_count", dist.wi_count[i]}, {"emit_injection", emit}});
        m2::AddRuntimeArgsForNode(
            writer_args.runtime_arg_values,
            core,
            {{"col_start", dist.wi_start[i]}, {"col_count", dist.wi_count[i]}, {"emit_injection", emit}});
        m2::AddRuntimeArgsForNode(
            compute_args.runtime_arg_values, core, {{"col_count", dist.wi_count[i]}, {"emit_injection", emit}});
    }

    m2::ProgramSpec spec{
        .name = "hc_mix_post",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {{.unique_id = PACKED_T, .spec = packed.tensor_spec()},
             {.unique_id = WEIGHTED_T, .spec = weighted.tensor_spec()},
             {.unique_id = UP_T, .spec = up.tensor_spec()},
             {.unique_id = MIXED_T, .spec = mixed.tensor_spec()},
             {.unique_id = INJ_T, .spec = injection.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = dist.core_set}},
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args), std::move(compute_args)};
    run_args.tensor_args = {
        {PACKED_T, packed}, {WEIGHTED_T, weighted}, {UP_T, up}, {MIXED_T, mixed}, {INJ_T, injection}};
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim

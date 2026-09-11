// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_gated_delta_rule_program_factory.hpp"

#include <cmath>
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

ttnn::device_operation::ProgramArtifacts RecurrentGatedDeltaRuleProgramFactory::create_program_artifacts(
    const RecurrentGatedDeltaRuleParams& attrs, const RecurrentGatedDeltaRuleInputs& in, std::vector<Tensor>& outputs) {
    const auto& query = in.query.mesh_tensor();
    const auto& key = in.key.mesh_tensor();
    const auto& value = in.value.mesh_tensor();
    const auto& beta = in.beta.mesh_tensor();
    const auto& log_decay = in.log_decay.mesh_tensor();
    const auto& state = in.state.mesh_tensor();
    const auto& core_out = outputs[0].mesh_tensor();
    const auto& state_out = outputs[1].mesh_tensor();
    const auto& device = query.device();
    const auto arch = device.arch();
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.value_dim / TILE_WIDTH;
    const uint32_t heads = attrs.batch * attrs.num_heads;
    auto dist = kda_factory_detail::distribute_prep(device.compute_with_storage_grid_size(), heads, heads);

    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    const m2::DFBSpecName Q{"q"}, K{"k"}, K_TRANS{"k_trans"}, V{"v"}, BETA{"beta"}, G{"g"}, STATE{"state"};
    const m2::DFBSpecName DECAY{"decay"}, S_DECAY{"s_decay"}, MEMORY{"memory"}, DELTA{"delta"};
    const m2::DFBSpecName K_COL{"k_col"}, OUTER{"outer"}, SCALED{"scaled"}, S_NEW{"s_new"};
    const m2::DFBSpecName CORE_OUT{"core_out"}, STATE_OUT{"state_out"};
    // In-kernel q/k L2 normalization (used when attrs.qk_norm_epsilon > 0).
    const m2::DFBSpecName SCALER{"scaler"}, EPS{"eps"};
    const m2::DFBSpecName QN_SQ{"qn_sq"}, QN_STATS{"qn_stats"}, QN_INV{"qn_inv"}, QN_OUT{"qn_out"};
    const m2::DFBSpecName KN_SQ{"kn_sq"}, KN_STATS{"kn_stats"}, KN_INV{"kn_inv"}, KN_OUT{"kn_out"},
        KN_OUT_T{"kn_out_t"};
    const bool norm_qk = attrs.qk_norm_epsilon > 0.0f;
    uint32_t eps_bits = 0;
    std::memcpy(&eps_bits, &attrs.qk_norm_epsilon, sizeof(float));
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(attrs.key_dim));
    uint32_t q_scale_bits = 0;
    std::memcpy(&q_scale_bits, &q_scale, sizeof(float));

    const m2::TensorParamName QUERY_T{"query"}, KEY_T{"key"}, VALUE_T{"value"}, BETA_T{"beta"};
    const m2::TensorParamName G_T{"log_decay"}, STATE_T{"state"}, CORE_OUT_T{"core_out"}, STATE_OUT_T{"state_out"};

    auto dfb = [](const m2::DFBSpecName& name, uint32_t tiles) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(tt::DataFormat::Float32),
            .num_entries = tiles,
            .data_format_metadata = tt::DataFormat::Float32,
        };
    };
    const uint32_t kv = Kt * Vt;
    m2::Group<m2::DataflowBufferSpec> dfbs = {
        dfb(Q, Kt),
        dfb(K, Kt),
        dfb(K_TRANS, Kt),
        dfb(V, Vt),
        dfb(BETA, 1),
        dfb(G, 1),
        dfb(STATE, kv),
        dfb(DECAY, 1),
        dfb(S_DECAY, kv),
        dfb(MEMORY, Vt),
        dfb(DELTA, Vt),
        dfb(K_COL, Kt),
        dfb(OUTER, kv),
        dfb(SCALED, kv),
        dfb(S_NEW, kv),
        dfb(CORE_OUT, Vt),
        dfb(STATE_OUT, kv),
        dfb(SCALER, 1),
        // The scalar generator writes 16-bit values: keep the epsilon tile bf16.
        m2::DataflowBufferSpec{
            .unique_id = EPS,
            .entry_size = tt::tile_size(tt::DataFormat::Float16_b),
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::Float16_b,
        },
        dfb(QN_SQ, Kt),
        dfb(QN_STATS, 1),
        dfb(QN_INV, 1),
        dfb(QN_OUT, Kt),
        dfb(KN_SQ, Kt),
        dfb(KN_STATS, 1),
        dfb(KN_INV, 1),
        dfb(KN_OUT, Kt),
        dfb(KN_OUT_T, Kt),
    };

    m2::KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_gated_delta_rule/device/kernels/dataflow/"
            "reader_recurrent_gated_delta_rule.cpp",
        .dfb_bindings =
            {{Q, "q", m2::DFBEndpointType::PRODUCER},
             {K, "k", m2::DFBEndpointType::PRODUCER},
             {K_TRANS, "k_trans", m2::DFBEndpointType::PRODUCER},
             {V, "v", m2::DFBEndpointType::PRODUCER},
             {BETA, "beta", m2::DFBEndpointType::PRODUCER},
             {G, "g", m2::DFBEndpointType::PRODUCER},
             {STATE, "state", m2::DFBEndpointType::PRODUCER},
             {SCALER, "scaler", m2::DFBEndpointType::PRODUCER},
             {EPS, "eps", m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {{QUERY_T, "query"},
             {KEY_T, "key"},
             {VALUE_T, "value"},
             {BETA_T, "beta"},
             {G_T, "log_decay"},
             {STATE_T, "state"}},
        .compile_time_args =
            {{"Kt", Kt},
             {"Vt", Vt},
             {"H", attrs.num_heads},
             {"QkRepeat", attrs.qk_head_repeat},
             {"NormQK", norm_qk ? 1u : 0u},
             {"EpsBits", eps_bits}},
        .runtime_arg_schema = {.runtime_arg_names = {"head_start", "head_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_gated_delta_rule/device/kernels/dataflow/"
            "writer_recurrent_gated_delta_rule.cpp",
        .dfb_bindings =
            {{CORE_OUT, "core_out", m2::DFBEndpointType::CONSUMER},
             {STATE_OUT, "state_out", m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {{CORE_OUT_T, "core_out"}, {STATE_OUT_T, "state_out"}},
        .compile_time_args = {{"Kt", Kt}, {"Vt", Vt}},
        .runtime_arg_schema = {.runtime_arg_names = {"head_start", "head_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    auto compute_hw = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);
    auto& unpack_modes = m2::unpack_modes(compute_hw);
    for (const auto& name :
         {Q,     K,        V,         BETA,   G,   STATE, DECAY,    S_DECAY, MEMORY, DELTA, K_COL,    OUTER,  SCALED,
          S_NEW, CORE_OUT, STATE_OUT, SCALER, EPS, QN_SQ, QN_STATS, QN_INV,  QN_OUT, KN_SQ, KN_STATS, KN_INV, KN_OUT}) {
        unpack_modes[name] = UnpackMode::UnpackToSrc;
    }
    unpack_modes[K_TRANS] = UnpackMode::UnpackToDest;
    unpack_modes[KN_OUT_T] = UnpackMode::UnpackToDest;
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_gated_delta_rule/device/kernels/compute/"
            "recurrent_gated_delta_rule.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {{Q, "q", m2::DFBEndpointType::CONSUMER},
             {K, "k", m2::DFBEndpointType::CONSUMER},
             {K_TRANS, "k_trans", m2::DFBEndpointType::CONSUMER},
             {V, "v", m2::DFBEndpointType::CONSUMER},
             {BETA, "beta", m2::DFBEndpointType::CONSUMER},
             {G, "g", m2::DFBEndpointType::CONSUMER},
             {STATE, "state", m2::DFBEndpointType::CONSUMER},
             {DECAY, "decay", m2::DFBEndpointType::PRODUCER},
             {DECAY, "decay", m2::DFBEndpointType::CONSUMER},
             {S_DECAY, "s_decay", m2::DFBEndpointType::PRODUCER},
             {S_DECAY, "s_decay", m2::DFBEndpointType::CONSUMER},
             {MEMORY, "memory", m2::DFBEndpointType::PRODUCER},
             {MEMORY, "memory", m2::DFBEndpointType::CONSUMER},
             {DELTA, "delta", m2::DFBEndpointType::PRODUCER},
             {DELTA, "delta", m2::DFBEndpointType::CONSUMER},
             {K_COL, "k_col", m2::DFBEndpointType::PRODUCER},
             {K_COL, "k_col", m2::DFBEndpointType::CONSUMER},
             {OUTER, "outer", m2::DFBEndpointType::PRODUCER},
             {OUTER, "outer", m2::DFBEndpointType::CONSUMER},
             {SCALED, "scaled", m2::DFBEndpointType::PRODUCER},
             {SCALED, "scaled", m2::DFBEndpointType::CONSUMER},
             {S_NEW, "s_new", m2::DFBEndpointType::PRODUCER},
             {S_NEW, "s_new", m2::DFBEndpointType::CONSUMER},
             {CORE_OUT, "core_out", m2::DFBEndpointType::PRODUCER},
             {STATE_OUT, "state_out", m2::DFBEndpointType::PRODUCER},
             {SCALER, "scaler", m2::DFBEndpointType::CONSUMER},
             {EPS, "eps", m2::DFBEndpointType::CONSUMER},
             {QN_SQ, "qn_sq", m2::DFBEndpointType::PRODUCER},
             {QN_SQ, "qn_sq", m2::DFBEndpointType::CONSUMER},
             {QN_STATS, "qn_stats", m2::DFBEndpointType::PRODUCER},
             {QN_STATS, "qn_stats", m2::DFBEndpointType::CONSUMER},
             {QN_INV, "qn_inv", m2::DFBEndpointType::PRODUCER},
             {QN_INV, "qn_inv", m2::DFBEndpointType::CONSUMER},
             {QN_OUT, "qn_out", m2::DFBEndpointType::PRODUCER},
             {QN_OUT, "qn_out", m2::DFBEndpointType::CONSUMER},
             {KN_SQ, "kn_sq", m2::DFBEndpointType::PRODUCER},
             {KN_SQ, "kn_sq", m2::DFBEndpointType::CONSUMER},
             {KN_STATS, "kn_stats", m2::DFBEndpointType::PRODUCER},
             {KN_STATS, "kn_stats", m2::DFBEndpointType::CONSUMER},
             {KN_INV, "kn_inv", m2::DFBEndpointType::PRODUCER},
             {KN_INV, "kn_inv", m2::DFBEndpointType::CONSUMER},
             {KN_OUT, "kn_out", m2::DFBEndpointType::PRODUCER},
             {KN_OUT, "kn_out", m2::DFBEndpointType::CONSUMER},
             {KN_OUT_T, "kn_out_t", m2::DFBEndpointType::PRODUCER},
             {KN_OUT_T, "kn_out_t", m2::DFBEndpointType::CONSUMER}},
        .compile_time_args =
            {{"Kt", Kt},
             {"Vt", Vt},
             {"NormQK", norm_qk ? 1u : 0u},
             {"EpsBits", eps_bits},
             {"QScaleBits", q_scale_bits}},
        .runtime_arg_schema = {.runtime_arg_names = {"head_count"}},
        .hw_config = std::move(compute_hw),
    };

    m2::KernelRunArgs reader_args{.kernel = READER}, writer_args{.kernel = WRITER}, compute_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        m2::AddRuntimeArgsForNode(
            reader_args.runtime_arg_values, core, {{"head_start", dist.wi_start[i]}, {"head_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(
            writer_args.runtime_arg_values, core, {{"head_start", dist.wi_start[i]}, {"head_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"head_count", dist.wi_count[i]}});
    }

    m2::ProgramSpec spec{
        .name = "recurrent_gated_delta_rule",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {{.unique_id = QUERY_T, .spec = query.tensor_spec()},
             {.unique_id = KEY_T, .spec = key.tensor_spec()},
             {.unique_id = VALUE_T, .spec = value.tensor_spec()},
             {.unique_id = BETA_T, .spec = beta.tensor_spec()},
             {.unique_id = G_T, .spec = log_decay.tensor_spec()},
             {.unique_id = STATE_T, .spec = state.tensor_spec()},
             {.unique_id = CORE_OUT_T, .spec = core_out.tensor_spec()},
             {.unique_id = STATE_OUT_T, .spec = state_out.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = dist.core_set}},
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args), std::move(compute_args)};
    run_args.tensor_args = {
        {QUERY_T, query},
        {KEY_T, key},
        {VALUE_T, value},
        {BETA_T, beta},
        {G_T, log_decay},
        {STATE_T, state},
        {CORE_OUT_T, core_out},
        {STATE_OUT_T, state_out}};
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim

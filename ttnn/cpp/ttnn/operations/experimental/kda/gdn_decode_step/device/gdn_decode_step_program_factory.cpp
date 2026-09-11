// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gdn_decode_step_program_factory.hpp"

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

ttnn::device_operation::ProgramArtifacts GdnDecodeStepProgramFactory::create_program_artifacts(
    const GdnDecodeStepParams& attrs, const GdnDecodeStepInputs& in, std::vector<Tensor>& outputs) {
    const auto& x = in.x.mesh_tensor();
    const auto& tap0 = in.tap0.mesh_tensor();
    const auto& tap1 = in.tap1.mesh_tensor();
    const auto& tap2 = in.tap2.mesh_tensor();
    const auto& w0 = in.conv_w0.mesh_tensor();
    const auto& w1 = in.conv_w1.mesh_tensor();
    const auto& w2 = in.conv_w2.mesh_tensor();
    const auto& w3 = in.conv_w3.mesh_tensor();
    const auto& beta = in.beta.mesh_tensor();
    const auto& log_decay = in.log_decay.mesh_tensor();
    const auto& state = in.state.mesh_tensor();
    const auto& gate = in.gate.mesh_tensor();
    const auto& norm_weight = in.norm_weight.mesh_tensor();
    const auto& out = outputs[0].mesh_tensor();
    const auto& state_out = outputs[1].mesh_tensor();
    const auto& device = x.device();
    const auto arch = device.arch();

    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.value_dim / TILE_WIDTH;
    const uint32_t H = attrs.num_heads;
    const uint32_t Hk = H / attrs.qk_head_repeat;
    const uint32_t Wt = attrs.qkv_width / TILE_WIDTH;
    const uint32_t heads = attrs.batch * H;
    const uint32_t kv = Kt * Vt;
    const uint32_t fir_tiles = std::max(Kt, Vt);
    auto dist = kda_factory_detail::distribute_prep(device.compute_with_storage_grid_size(), heads, heads);

    const auto conv_w_format = datatype_to_dataformat_converter(in.conv_w0.dtype());
    const auto out_format = datatype_to_dataformat_converter(attrs.output_dtype);

    uint32_t qk_eps_bits = 0, norm_eps_bits = 0, q_scale_bits = 0;
    std::memcpy(&qk_eps_bits, &attrs.qk_norm_epsilon, sizeof(float));
    std::memcpy(&norm_eps_bits, &attrs.norm_epsilon, sizeof(float));
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(attrs.key_dim));
    std::memcpy(&q_scale_bits, &q_scale, sizeof(float));

    const m2::KernelSpecName READER{"reader"}, WRITER{"writer"}, COMPUTE{"compute"};

    // Dataflow buffers.  FIR inputs are shared by the q/k/v segments.
    const m2::DFBSpecName FIR_X{"fir_x"}, FIR_T0{"fir_t0"}, FIR_T1{"fir_t1"}, FIR_T2{"fir_t2"};
    const m2::DFBSpecName FIR_W0{"fir_w0"}, FIR_W1{"fir_w1"}, FIR_W2{"fir_w2"}, FIR_W3{"fir_w3"};
    const m2::DFBSpecName Q_RAW{"q_raw"}, K_RAW{"k_raw"}, V{"v"}, BETA{"beta"}, G{"g"}, STATE{"state"};
    const m2::DFBSpecName SCALER_SUM{"scaler_sum"}, SCALER_AVG{"scaler_avg"}, EPS_QK{"eps_qk"}, EPS_NORM{"eps_norm"};
    const m2::DFBSpecName QN_SQ{"qn_sq"}, QN_STATS{"qn_stats"}, QN_INV{"qn_inv"}, QN_OUT{"qn_out"};
    const m2::DFBSpecName KN_SQ{"kn_sq"}, KN_STATS{"kn_stats"}, KN_INV{"kn_inv"}, KN_OUT{"kn_out"},
        KN_OUT_T{"kn_out_t"};
    const m2::DFBSpecName DECAY{"decay"}, S_DECAY{"s_decay"}, MEMORY{"memory"}, DELTA{"delta"};
    const m2::DFBSpecName K_COL{"k_col"}, OUTER{"outer"}, SCALED{"scaled"}, S_NEW{"s_new"}, CORE{"core"};
    const m2::DFBSpecName EP_SQ{"ep_sq"}, EP_STATS{"ep_stats"}, EP_INV{"ep_inv"}, EP_NORM{"ep_norm"}, EP_TMP{"ep_tmp"};
    const m2::DFBSpecName GATE{"gate"}, GATE_ACT{"gate_act"}, NORM_W{"norm_w"}, OUT{"out"}, STATE_OUT{"state_out"};
    const m2::DFBSpecName MASK{"mask"}, QN_INVM{"qn_invm"}, KN_INVM{"kn_invm"};

    const m2::TensorParamName X_T{"x"}, T0_T{"tap0"}, T1_T{"tap1"}, T2_T{"tap2"};
    const m2::TensorParamName W0_T{"conv_w0"}, W1_T{"conv_w1"}, W2_T{"conv_w2"}, W3_T{"conv_w3"};
    const m2::TensorParamName BETA_T{"beta"}, G_T{"log_decay"}, STATE_T{"state"}, GATE_T{"gate"},
        NORM_W_T{"norm_weight"};
    const m2::TensorParamName OUT_T{"out"}, STATE_OUT_T{"state_out"};

    auto dfb = [](const m2::DFBSpecName& name, uint32_t tiles, tt::DataFormat format = tt::DataFormat::Float32) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = tiles,
            .data_format_metadata = format,
        };
    };
    m2::Group<m2::DataflowBufferSpec> dfbs = {
        dfb(FIR_X, fir_tiles),
        dfb(FIR_T0, fir_tiles),
        dfb(FIR_T1, fir_tiles),
        dfb(FIR_T2, fir_tiles),
        dfb(FIR_W0, fir_tiles, conv_w_format),
        dfb(FIR_W1, fir_tiles, conv_w_format),
        dfb(FIR_W2, fir_tiles, conv_w_format),
        dfb(FIR_W3, fir_tiles, conv_w_format),
        dfb(Q_RAW, Kt),
        dfb(K_RAW, Kt),
        dfb(V, Vt),
        dfb(BETA, 1),
        dfb(G, 1),
        dfb(STATE, kv),
        dfb(SCALER_SUM, 1),
        dfb(SCALER_AVG, 1),
        dfb(EPS_QK, 1, tt::DataFormat::Float16_b),
        dfb(EPS_NORM, 1, tt::DataFormat::Float16_b),
        dfb(QN_SQ, Kt),
        dfb(QN_STATS, 1),
        dfb(QN_INV, 1),
        dfb(QN_OUT, Kt),
        dfb(KN_SQ, Kt),
        dfb(KN_STATS, 1),
        dfb(KN_INV, 1),
        dfb(KN_OUT, Kt),
        dfb(KN_OUT_T, Kt),
        dfb(DECAY, 1),
        dfb(S_DECAY, kv),
        dfb(MEMORY, Vt),
        dfb(DELTA, Vt),
        dfb(K_COL, Kt),
        dfb(OUTER, kv),
        dfb(SCALED, Vt),
        dfb(S_NEW, kv),
        dfb(CORE, Vt),
        dfb(EP_SQ, Vt),
        dfb(EP_STATS, 1),
        dfb(EP_INV, 1),
        dfb(EP_NORM, Vt),
        dfb(EP_TMP, Vt),
        dfb(GATE, Vt, tt::DataFormat::Float16_b),
        dfb(GATE_ACT, Vt),
        dfb(NORM_W, Vt, tt::DataFormat::Float16_b),
        dfb(OUT, Vt, out_format),
        dfb(STATE_OUT, kv),
        dfb(MASK, 1),
        dfb(QN_INVM, 1),
        dfb(KN_INVM, 1),
    };

    const std::vector<std::pair<std::string, uint32_t>> common_ct = {
        {"Kt", Kt}, {"Vt", Vt}, {"H", H}, {"Hk", Hk}, {"Wt", Wt}, {"QkRepeat", attrs.qk_head_repeat}};

    m2::KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/"
            "reader_gdn_decode_step.cpp",
        .dfb_bindings =
            {{FIR_X, "fir_x", m2::DFBEndpointType::PRODUCER},
             {FIR_T0, "fir_t0", m2::DFBEndpointType::PRODUCER},
             {FIR_T1, "fir_t1", m2::DFBEndpointType::PRODUCER},
             {FIR_T2, "fir_t2", m2::DFBEndpointType::PRODUCER},
             {FIR_W0, "fir_w0", m2::DFBEndpointType::PRODUCER},
             {FIR_W1, "fir_w1", m2::DFBEndpointType::PRODUCER},
             {FIR_W2, "fir_w2", m2::DFBEndpointType::PRODUCER},
             {FIR_W3, "fir_w3", m2::DFBEndpointType::PRODUCER},
             {BETA, "beta", m2::DFBEndpointType::PRODUCER},
             {G, "g", m2::DFBEndpointType::PRODUCER},
             {STATE, "state", m2::DFBEndpointType::PRODUCER},
             {GATE, "gate", m2::DFBEndpointType::PRODUCER},
             {NORM_W, "norm_w", m2::DFBEndpointType::PRODUCER},
             {SCALER_SUM, "scaler_sum", m2::DFBEndpointType::PRODUCER},
             {SCALER_AVG, "scaler_avg", m2::DFBEndpointType::PRODUCER},
             {EPS_QK, "eps_qk", m2::DFBEndpointType::PRODUCER},
             {EPS_NORM, "eps_norm", m2::DFBEndpointType::PRODUCER},
             {MASK, "mask", m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {{X_T, "x"},
             {T0_T, "tap0"},
             {T1_T, "tap1"},
             {T2_T, "tap2"},
             {W0_T, "conv_w0"},
             {W1_T, "conv_w1"},
             {W2_T, "conv_w2"},
             {W3_T, "conv_w3"},
             {BETA_T, "beta"},
             {G_T, "log_decay"},
             {STATE_T, "state"},
             {GATE_T, "gate"},
             {NORM_W_T, "norm_weight"}},
        .compile_time_args =
            {{"Kt", Kt},
             {"Vt", Vt},
             {"H", H},
             {"Hk", Hk},
             {"Wt", Wt},
             {"QkRepeat", attrs.qk_head_repeat},
             {"QkEpsBits", qk_eps_bits},
             {"NormEpsBits", norm_eps_bits}},
        .runtime_arg_schema = {.runtime_arg_names = {"head_start", "head_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/"
            "writer_gdn_decode_step.cpp",
        .dfb_bindings =
            {{OUT, "out", m2::DFBEndpointType::CONSUMER}, {STATE_OUT, "state_out", m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {{OUT_T, "out"}, {STATE_OUT_T, "state_out"}},
        .compile_time_args = {{"Kt", Kt}, {"Vt", Vt}, {"H", H}},
        .runtime_arg_schema = {.runtime_arg_names = {"head_start", "head_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    auto compute_hw = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);
    auto& unpack_modes = m2::unpack_modes(compute_hw);
    for (const auto& name :
         {FIR_X,  FIR_T0,   FIR_T1, FIR_T2,   FIR_W0,     FIR_W1,     FIR_W2,   FIR_W3,   Q_RAW,   K_RAW,
          V,      BETA,     G,      STATE,    SCALER_SUM, SCALER_AVG, EPS_QK,   EPS_NORM, QN_SQ,   QN_STATS,
          QN_INV, QN_OUT,   KN_SQ,  KN_STATS, KN_INV,     KN_OUT,     DECAY,    S_DECAY,  MEMORY,  DELTA,
          K_COL,  OUTER,    SCALED, S_NEW,    CORE,       EP_SQ,      EP_STATS, EP_INV,   EP_NORM, EP_TMP,
          GATE,   GATE_ACT, NORM_W, OUT,      STATE_OUT,  MASK,       QN_INVM,  KN_INVM}) {
        unpack_modes[name] = UnpackMode::UnpackToSrc;
    }
    unpack_modes[KN_OUT_T] = UnpackMode::UnpackToDest;

    auto both = [](const m2::DFBSpecName& name, const char* id) {
        return std::vector<m2::DFBBinding>{
            {name, id, m2::DFBEndpointType::PRODUCER}, {name, id, m2::DFBEndpointType::CONSUMER}};
    };
    std::vector<m2::DFBBinding> compute_bindings = {
        {FIR_X, "fir_x", m2::DFBEndpointType::CONSUMER},
        {FIR_T0, "fir_t0", m2::DFBEndpointType::CONSUMER},
        {FIR_T1, "fir_t1", m2::DFBEndpointType::CONSUMER},
        {FIR_T2, "fir_t2", m2::DFBEndpointType::CONSUMER},
        {FIR_W0, "fir_w0", m2::DFBEndpointType::CONSUMER},
        {FIR_W1, "fir_w1", m2::DFBEndpointType::CONSUMER},
        {FIR_W2, "fir_w2", m2::DFBEndpointType::CONSUMER},
        {FIR_W3, "fir_w3", m2::DFBEndpointType::CONSUMER},
        {BETA, "beta", m2::DFBEndpointType::CONSUMER},
        {G, "g", m2::DFBEndpointType::CONSUMER},
        {STATE, "state", m2::DFBEndpointType::CONSUMER},
        {GATE, "gate", m2::DFBEndpointType::CONSUMER},
        {NORM_W, "norm_w", m2::DFBEndpointType::CONSUMER},
        {SCALER_SUM, "scaler_sum", m2::DFBEndpointType::CONSUMER},
        {SCALER_AVG, "scaler_avg", m2::DFBEndpointType::CONSUMER},
        {EPS_QK, "eps_qk", m2::DFBEndpointType::CONSUMER},
        {EPS_NORM, "eps_norm", m2::DFBEndpointType::CONSUMER},
        {MASK, "mask", m2::DFBEndpointType::CONSUMER},
        {OUT, "out", m2::DFBEndpointType::PRODUCER},
        {STATE_OUT, "state_out", m2::DFBEndpointType::PRODUCER}};
    for (const auto& [name, id] : std::vector<std::pair<m2::DFBSpecName, const char*>>{
             {Q_RAW, "q_raw"},     {K_RAW, "k_raw"},       {V, "v"},
             {QN_SQ, "qn_sq"},     {QN_STATS, "qn_stats"}, {QN_INV, "qn_inv"},
             {QN_OUT, "qn_out"},   {KN_SQ, "kn_sq"},       {KN_STATS, "kn_stats"},
             {KN_INV, "kn_inv"},   {KN_OUT, "kn_out"},     {KN_OUT_T, "kn_out_t"},
             {DECAY, "decay"},     {S_DECAY, "s_decay"},   {MEMORY, "memory"},
             {DELTA, "delta"},     {K_COL, "k_col"},       {OUTER, "outer"},
             {SCALED, "scaled"},   {S_NEW, "s_new"},       {CORE, "core"},
             {EP_SQ, "ep_sq"},     {EP_STATS, "ep_stats"}, {EP_INV, "ep_inv"},
             {EP_NORM, "ep_norm"}, {EP_TMP, "ep_tmp"},     {GATE_ACT, "gate_act"},
             {QN_INVM, "qn_invm"}, {KN_INVM, "kn_invm"}}) {
        for (auto& binding : both(name, id)) {
            compute_bindings.push_back(binding);
        }
    }
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/compute/gdn_decode_step.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_bindings),
        .compile_time_args = {{"Kt", Kt}, {"Vt", Vt}, {"QScaleBits", q_scale_bits}},
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
        .name = "gdn_decode_step",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {{.unique_id = X_T, .spec = x.tensor_spec()},
             {.unique_id = T0_T, .spec = tap0.tensor_spec()},
             {.unique_id = T1_T, .spec = tap1.tensor_spec()},
             {.unique_id = T2_T, .spec = tap2.tensor_spec()},
             {.unique_id = W0_T, .spec = w0.tensor_spec()},
             {.unique_id = W1_T, .spec = w1.tensor_spec()},
             {.unique_id = W2_T, .spec = w2.tensor_spec()},
             {.unique_id = W3_T, .spec = w3.tensor_spec()},
             {.unique_id = BETA_T, .spec = beta.tensor_spec()},
             {.unique_id = G_T, .spec = log_decay.tensor_spec()},
             {.unique_id = STATE_T, .spec = state.tensor_spec()},
             {.unique_id = GATE_T, .spec = gate.tensor_spec()},
             {.unique_id = NORM_W_T, .spec = norm_weight.tensor_spec()},
             {.unique_id = OUT_T, .spec = out.tensor_spec()},
             {.unique_id = STATE_OUT_T, .spec = state_out.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = dist.core_set}},
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args), std::move(compute_args)};
    run_args.tensor_args = {
        {X_T, x},
        {T0_T, tap0},
        {T1_T, tap1},
        {T2_T, tap2},
        {W0_T, w0},
        {W1_T, w1},
        {W2_T, w2},
        {W3_T, w3},
        {BETA_T, beta},
        {G_T, log_decay},
        {STATE_T, state},
        {GATE_T, gate},
        {NORM_W_T, norm_weight},
        {OUT_T, out},
        {STATE_OUT_T, state_out}};
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim

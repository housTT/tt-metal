// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"

namespace {

template <typename Accessor>
void read_tiles(Noc& noc, const Accessor& accessor, DataflowBuffer& buffer, uint32_t first, uint32_t count) {
    buffer.reserve_back(count);
    for (uint32_t tile = 0; tile < count; ++tile) {
        noc.async_read(
            accessor,
            buffer,
            buffer.get_entry_size(),
            {.page_id = first + tile},
            {.offset_bytes = tile * buffer.get_entry_size()});
    }
    noc.async_read_barrier();
    buffer.push_back(count);
}

}  // namespace

template <uint32_t Kt, uint32_t Vt, uint32_t H, uint32_t QkRepeat, uint32_t NormQK, uint32_t EpsBits>
TT_KERNEL void reader(uint32_t head_start, uint32_t head_count) {
    const auto q_acc = TensorAccessor(tensor::query);
    const auto k_acc = TensorAccessor(tensor::key);
    const auto v_acc = TensorAccessor(tensor::value);
    const auto beta_acc = TensorAccessor(tensor::beta);
    const auto g_acc = TensorAccessor(tensor::log_decay);
    const auto state_acc = TensorAccessor(tensor::state);
    DataflowBuffer q(dfb::q), k(dfb::k), k_trans(dfb::k_trans), v(dfb::v);
    DataflowBuffer beta(dfb::beta), g(dfb::g), state(dfb::state);
    Noc noc;
    constexpr uint32_t kv = Kt * Vt;
    if constexpr (NormQK) {
        // Row-sum scaler (1.0) and the epsilon column tile for the in-kernel
        // q/k L2 normalization; generated once per core.
        dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            1.0f);
        DataflowBuffer eps(dfb::eps);
        generate_bcast_col_scalar(eps, EpsBits);
    }
    for (uint32_t offset = 0; offset < head_count; ++offset) {
        const uint32_t head = head_start + offset;
        // Flattened (batch, value head) -> (batch, key head): value head h uses
        // key/query head h / QkRepeat within the same batch row.
        const uint32_t batch_row = head / H;
        const uint32_t value_head = head - batch_row * H;
        const uint32_t qk_head = batch_row * (H / QkRepeat) + value_head / QkRepeat;
        read_tiles(noc, q_acc, q, qk_head * Kt, Kt);
        read_tiles(noc, k_acc, k, qk_head * Kt, Kt);
        if constexpr (!NormQK) {
            read_tiles(noc, k_acc, k_trans, qk_head * Kt, Kt);
        }
        read_tiles(noc, v_acc, v, head * Vt, Vt);
        read_tiles(noc, beta_acc, beta, head, 1);
        read_tiles(noc, g_acc, g, head, 1);
        read_tiles(noc, state_acc, state, head * kv, kv);
    }
}

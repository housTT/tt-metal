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

// FP32 tile with 1.0 at row 0 / column 0 and zeros elsewhere.  Broadcast over
// columns it keeps row 0 (the valid decode row) and zeroes the tile padding.
void generate_row0_mask(DataflowBuffer& buffer) {
    buffer.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer.get_write_ptr());
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = 0;
    }
    ptr[0] = 0x3F800000u;
    buffer.push_back(1);
}

}  // namespace

// One work item = one (batch row, value head).  For each of the head's q, k
// and v channel segments the reader streams the x/tap tiles and conv weights
// (the compute kernel runs the FIR and routes the result), then the gates,
// the recurrent state and the epilogue gate slice.
template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t H,
    uint32_t Hk,
    uint32_t Wt,
    uint32_t QkRepeat,
    uint32_t QkEpsBits,
    uint32_t NormEpsBits>
TT_KERNEL void reader(uint32_t head_start, uint32_t head_count) {
    const auto x_acc = TensorAccessor(tensor::x);
    const auto t0_acc = TensorAccessor(tensor::tap0);
    const auto t1_acc = TensorAccessor(tensor::tap1);
    const auto t2_acc = TensorAccessor(tensor::tap2);
    const auto w0_acc = TensorAccessor(tensor::conv_w0);
    const auto w1_acc = TensorAccessor(tensor::conv_w1);
    const auto w2_acc = TensorAccessor(tensor::conv_w2);
    const auto w3_acc = TensorAccessor(tensor::conv_w3);
    const auto beta_acc = TensorAccessor(tensor::beta);
    const auto g_acc = TensorAccessor(tensor::log_decay);
    const auto state_acc = TensorAccessor(tensor::state);
    const auto gate_acc = TensorAccessor(tensor::gate);
    const auto norm_w_acc = TensorAccessor(tensor::norm_weight);
    DataflowBuffer fir_x(dfb::fir_x), fir_t0(dfb::fir_t0), fir_t1(dfb::fir_t1), fir_t2(dfb::fir_t2);
    DataflowBuffer fir_w0(dfb::fir_w0), fir_w1(dfb::fir_w1), fir_w2(dfb::fir_w2), fir_w3(dfb::fir_w3);
    DataflowBuffer beta(dfb::beta), g(dfb::g), state(dfb::state), gate(dfb::gate), norm_w(dfb::norm_w);
    Noc noc;
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t q_col0 = 0;
    constexpr uint32_t k_col0 = Hk * Kt;
    constexpr uint32_t v_col0 = 2 * Hk * Kt;

    dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        1.0f);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        dfb::scaler_avg,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        Vt * tt::constants::TILE_WIDTH>();
    {
        DataflowBuffer eps_qk(dfb::eps_qk), eps_norm(dfb::eps_norm);
        generate_bcast_col_scalar(eps_qk, QkEpsBits);
        generate_bcast_col_scalar(eps_norm, NormEpsBits);
        DataflowBuffer mask(dfb::mask);
        generate_row0_mask(mask);
    }
    read_tiles(noc, norm_w_acc, norm_w, 0, Vt);

    for (uint32_t offset = 0; offset < head_count; ++offset) {
        const uint32_t head = head_start + offset;
        const uint32_t b = head / H;
        const uint32_t h = head - b * H;
        const uint32_t hk = h / QkRepeat;
        const uint32_t row_base = b * Wt;
        // q, k, v channel segments of this head.
        const uint32_t seg_col[3] = {q_col0 + hk * Kt, k_col0 + hk * Kt, v_col0 + h * Vt};
        const uint32_t seg_tiles[3] = {Kt, Kt, Vt};
        for (uint32_t seg = 0; seg < 3; ++seg) {
            const uint32_t col = seg_col[seg];
            const uint32_t n = seg_tiles[seg];
            read_tiles(noc, t0_acc, fir_t0, row_base + col, n);
            read_tiles(noc, w0_acc, fir_w0, col, n);
            read_tiles(noc, t1_acc, fir_t1, row_base + col, n);
            read_tiles(noc, w1_acc, fir_w1, col, n);
            read_tiles(noc, t2_acc, fir_t2, row_base + col, n);
            read_tiles(noc, w2_acc, fir_w2, col, n);
            read_tiles(noc, x_acc, fir_x, row_base + col, n);
            read_tiles(noc, w3_acc, fir_w3, col, n);
        }
        read_tiles(noc, beta_acc, beta, head, 1);
        read_tiles(noc, g_acc, g, head, 1);
        read_tiles(noc, state_acc, state, head * kv, kv);
        read_tiles(noc, gate_acc, gate, b * H * Vt + h * Vt, Vt);
    }
}

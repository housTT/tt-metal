// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {

// Monotonic unsigned key for a bf16 bit pattern (larger key <=> larger value).
inline uint32_t order_key(uint16_t bits) {
    return (bits & 0x8000u) ? static_cast<uint32_t>(~bits & 0xFFFFu) : static_cast<uint32_t>(bits | 0x8000u);
}

inline float bf16_to_float(uint16_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = static_cast<uint32_t>(bits) << 16;
    return v.f;
}

inline uint16_t float_to_bf16(float value) {
    union {
        uint32_t u;
        float f;
    } v;
    v.f = value;
    // round to nearest even
    const uint32_t lsb = (v.u >> 16) & 1u;
    v.u += 0x7FFFu + lsb;
    return static_cast<uint16_t>(v.u >> 16);
}

// exp(x) for x <= 0 (softmax after max subtraction), soft-float friendly.
inline float exp_neg(float x) {
    if (x < -80.0f) {
        return 0.0f;
    }
    // exp(x) = 2^(x * log2 e) = 2^n * 2^f
    const float y = x * 1.4426950408889634f;
    int n = static_cast<int>(y);
    if (static_cast<float>(n) > y) {
        --n;  // floor
    }
    const float f = y - static_cast<float>(n);  // in [0, 1)
    // 2^f polynomial (degree 5, max rel err ~2e-7)
    float p = 1.8775767e-3f;
    p = p * f + 8.9893397e-3f;
    p = p * f + 5.5826318e-2f;
    p = p * f + 2.4015361e-1f;
    p = p * f + 6.9315308e-1f;
    p = p * f + 1.0f;
    union {
        uint32_t u;
        float f;
    } s;
    s.u = static_cast<uint32_t>(n + 127) << 23;
    return p * s.f;
}

}  // namespace

template <uint32_t K, uint32_t Et, uint32_t LocalExperts, uint32_t SlotBytes>
TT_KERNEL void reader() {
    const auto logits_acc = TensorAccessor(tensor::logits);
    const auto base_acc = TensorAccessor(tensor::rank_base);
    const auto slots_acc = TensorAccessor(tensor::slots);
    const auto scores_acc = TensorAccessor(tensor::scores);
    DataflowBuffer logits(dfb::logits), base(dfb::base), slots(dfb::slots), scores(dfb::scores);
    Noc noc;

    logits.reserve_back(Et);
    for (uint32_t t = 0; t < Et; ++t) {
        noc.async_read(
            logits_acc, logits, logits.get_entry_size(), {.page_id = t}, {.offset_bytes = t * logits.get_entry_size()});
    }
    base.reserve_back(1);
    noc.async_read(base_acc, base, 64, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const auto* tiles = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(logits.get_write_ptr());
    const int32_t rank_base = *reinterpret_cast<volatile tt_l1_ptr int32_t*>(base.get_write_ptr());

    // Row 0 of tile t: columns 0..15 at face 0 (index c), 16..31 at face 1 (index 256 + c - 16).
    uint32_t top_key[K];
    uint32_t top_idx[K];
    uint32_t count = 0;
    for (uint32_t t = 0; t < Et; ++t) {
        const volatile tt_l1_ptr uint16_t* tile = tiles + t * 1024;
        for (uint32_t c = 0; c < 32; ++c) {
            const uint16_t bits = tile[c < 16 ? c : 256 + c - 16];
            const uint32_t key = order_key(bits);
            const uint32_t idx = t * 32 + c;
            if (count == K && key <= top_key[K - 1]) {
                continue;
            }
            // insertion into the descending list; ties keep the earlier (lower) index first
            uint32_t pos = count < K ? count : K - 1;
            while (pos > 0 && top_key[pos - 1] < key) {
                top_key[pos] = top_key[pos - 1];
                top_idx[pos] = top_idx[pos - 1];
                --pos;
            }
            top_key[pos] = key;
            top_idx[pos] = idx;
            if (count < K) {
                ++count;
            }
        }
    }
    // softmax over the K selected logits
    float vals[K];
    float max_v = -3.0e38f;
    for (uint32_t i = 0; i < K; ++i) {
        const uint32_t idx = top_idx[i];
        const volatile tt_l1_ptr uint16_t* tile = tiles + (idx / 32) * 1024;
        const uint32_t c = idx % 32;
        vals[i] = bf16_to_float(tile[c < 16 ? c : 256 + c - 16]);
        if (vals[i] > max_v) {
            max_v = vals[i];
        }
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < K; ++i) {
        vals[i] = exp_neg(vals[i] - max_v);
        sum += vals[i];
    }
    const float inv = 1.0f / sum;

    slots.reserve_back(1);
    scores.reserve_back(1);
    auto* slot_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(slots.get_write_ptr());
    auto* score_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores.get_write_ptr());
    for (uint32_t i = 0; i < 32; ++i) {
        slot_ptr[i] = 0;
    }
    for (uint32_t i = 0; i < 1024; ++i) {
        score_ptr[i] = 0;
    }
    for (uint32_t i = 0; i < K; ++i) {
        const int32_t rel = static_cast<int32_t>(top_idx[i]) - rank_base;
        const bool local = rel >= 0 && rel < static_cast<int32_t>(LocalExperts);
        slot_ptr[i] = local ? static_cast<uint16_t>(rel) : 0;
        score_ptr[i] = local ? float_to_bf16(vals[i] * inv) : 0;  // row 0, column i (face 0)
    }
    noc.async_write(slots, slots_acc, SlotBytes, {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write(scores, scores_acc, scores.get_entry_size(), {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
}

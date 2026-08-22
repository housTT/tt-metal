// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Runtime chunk-size picker, shared by the reader, writer, and compute kernels.
//
// A fused program executes several local experts through the SAME circular
// buffers. TT circular-buffer producers and consumers may wrap only exactly at
// the FIFO limit; a block may not straddle it. Consequently the program must
// use one per_core_M for every active expert and every chunk in the launch.
// Picking a different divisor-sized block for each expert is not sufficient:
// e.g. a max-sized block, then a 1/8-sized block, then another max-sized block
// leaves the FIFO pointer at +1/8 and makes the last block straddle the limit.
//
// Each kernel therefore scans the same local count vector once, calls
// shared_runtime_geometry() with the maximum count, and uses the returned
// per_core_M/chunk_M_tiles for the entire launch. The selected per_core_M is a
// divisor of the CB-sized maximum, so the fixed block cadence always lands on a
// legal ring boundary. Small launches remain adaptive; a launch containing a
// hot expert promotes its small tail experts to the hot expert's geometry.
//
// The three kernels MUST derive identical geometry and per-expert chunk counts,
// or the reader/compute/writer disagree on CB cadence and row mapping. This is
// the single source of truth. Pure integer arithmetic (no NoC/CB/reg APIs) keeps
// the header valid in BRISC, NCRISC, and TRISC translation units alike.
namespace adaptive_chunk {

constexpr uint32_t kGridY = 8;  // M-row cores; a chunk spans per_core_M * kGridY tile-rows

struct RuntimeGeometry {
    uint32_t per_core_M;
    uint32_t chunk_M_tiles;
};

// Number of fixed-geometry chunks needed for one expert. A zero-count expert is
// skipped; a nonzero tail runs one full runtime chunk and its phantom rows are
// dropped by the existing row<count guards.
inline uint32_t num_chunks(uint32_t count_tiles, uint32_t chunk_M_tiles) {
    if (count_tiles < 1) {
        return 0;
    }
    const uint32_t num_full = count_tiles / chunk_M_tiles;
    const uint32_t tail = count_tiles - num_full * chunk_M_tiles;
    return num_full + ((tail > 0) ? 1u : 0u);
}

// Pick one launch-wide geometry from the maximum active local-expert count. The
// smallest divisor of per_core_M_max that covers that count minimizes phantom
// M-work while preserving a single legal CB block cadence. Once the maximum
// needs at least one full max chunk, use the max geometry for all chunks.
inline RuntimeGeometry shared_runtime_geometry(uint32_t max_count_tiles, uint32_t max_chunk) {
    const uint32_t per_core_M_max = max_chunk / kGridY;
    const uint32_t bounded_count = max_count_tiles < max_chunk ? max_count_tiles : max_chunk;
    uint32_t need = (bounded_count + kGridY - 1) / kGridY;
    if (need < 1) {
        need = 1;
    }
    for (uint32_t d = need; d <= per_core_M_max; ++d) {
        if ((per_core_M_max % d) == 0) {
            return RuntimeGeometry{d, d * kGridY};
        }
    }
    return RuntimeGeometry{per_core_M_max, max_chunk};
}

}  // namespace adaptive_chunk

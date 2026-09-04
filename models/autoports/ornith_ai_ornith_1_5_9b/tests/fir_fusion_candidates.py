# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Share row-major conversion across FIR shifts while preserving BF16 arithmetic."""

import ttnn

from .linear_fusion_candidates import CombinedRMTailGDN, HybridArithmetic


class SharedFIRRows(HybridArithmetic):
    _write_conv_state = CombinedRMTailGDN._write_conv_state

    def _causal_conv(self, qkv, logical_len):
        batch, seq, width = list(qkv.shape)
        dram = ttnn.DRAM_MEMORY_CONFIG
        rows = [ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram) for value in (*self.conv_state, qkv)]
        padded = ttnn.concat(rows, dim=1)
        for value in rows:
            ttnn.deallocate(value)
        acc = None
        for tap, weight in enumerate(self.w["conv_taps"]):
            piece = ttnn.slice(padded, [0, tap, 0], [batch, tap + seq, width])
            tiled = ttnn.to_layout(piece, ttnn.TILE_LAYOUT, memory_config=dram)
            ttnn.deallocate(piece)
            if acc is None:
                acc = ttnn.multiply(tiled, weight, memory_config=dram)
            else:
                result = ttnn.addcmul(acc, tiled, weight, memory_config=dram)
                ttnn.deallocate(acc)
                acc = result
            ttnn.deallocate(tiled)
        activated = ttnn.silu(acc, memory_config=dram)
        ttnn.deallocate(acc)
        tail = ttnn.slice(padded, [0, logical_len, 0], [batch, logical_len + 3, width])
        ttnn.deallocate(padded)
        return activated, tail

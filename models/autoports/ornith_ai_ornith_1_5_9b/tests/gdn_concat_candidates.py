# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dedicated FP32 GDN decode head concatenation on the current fused graph.

The ordinary concat operator accepts tiled, interleaved FP32 heads with logical
sequence length one. Unlike concat_heads_decode, it requires no sharded input
handoff. The factory assigns one output row tile per core, so the B1 workload
uses one core and a 1 MiB double-buffered circular buffer for 32 x 128 FP32
heads. Device correctness and timing must decide whether that tradeoff wins.
"""

import ttnn

from ..tt.fused_decoder import FusedDecoder


class ConcatGDN(FusedDecoder):
    """Replace only the decode output head permutation with dedicated concat."""

    def _gdn_out_head_major(self, core, z, batch, seq):
        if seq > 1:
            return super()._gdn_out_head_major(core, z, batch, seq)
        normed = ttnn.rms_norm(core, weight=self.w["gdn_norm"], epsilon=self.cfg.norm_eps)
        ttnn.deallocate(core)
        heads = ttnn.reshape(normed, [batch, self.cfg.linear_num_value_heads, 1, self.cfg.linear_value_head_dim])
        combined = ttnn.experimental.nlp_concat_heads(heads)
        ttnn.deallocate(normed)
        merged = ttnn.reshape(combined, [batch, 1, self.cfg.linear_v_dim])
        gated = ttnn.multiply(merged, z)
        ttnn.deallocate(combined)
        result = ttnn.linear(gated, self.w["gdn_out"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(gated)
        return result

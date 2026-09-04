# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Final-graph outer-product transpose fusion and matched-program control.

The default multicore matmul manually transposes K even with transpose_a=True.
Reuse supports a native transpose. Its smallest full-N output strip is one
32x128 block: per_core_M=1, per_core_N=4, one padded K tile, and four FP32
destination tiles. No dtype, arithmetic order, state ownership, or other
projection program is changed. Device validation is owned by the coordinator.
"""

import ttnn

from ..tt.fused_decoder import FusedDecoder


class ReuseOuterGDN(FusedDecoder):
    """Request a native K transpose in the existing FP32 outer-product matmul."""

    native_outer_transpose = True
    outer_rows_per_core = 1

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            cfg = decoder.cfg
            if cfg.linear_key_head_dim != 128 or cfg.linear_value_head_dim != 128:
                raise ValueError("the outer-product reuse probe requires 128-wide Q/K/V heads")
            grid = decoder.device.compute_with_storage_grid_size()
            decoder.outer_program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=decoder.outer_rows_per_core,
                per_core_N=4,
            )
        return decoder

    def _delta_rule_step(self, q, k, v, beta, g):
        cfg = self.cfg
        batch, nv, dk = q.shape[0], cfg.linear_num_value_heads, cfg.linear_key_head_dim
        dram = ttnn.DRAM_MEMORY_CONFIG
        q_row = ttnn.multiply(q, dk**-1.0, dtype=ttnn.float32, memory_config=dram)
        k_row = ttnn.unary_chain(k, self.key_scale_chain, memory_config=ttnn.L1_MEMORY_CONFIG)
        beta_view = ttnn.reshape(beta, [batch, nv, 1, 1])
        g_view = ttnn.reshape(g, [batch, nv, 1, 1])
        state = self.recurrent_state
        ttnn.multiply(state, g_view, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], output_tensor=state)
        read = ttnn.matmul(k_row, state, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        difference = ttnn.subtract(v, read, dtype=ttnn.float32, memory_config=dram)
        delta = ttnn.multiply(difference, beta_view, memory_config=dram)
        for tensor in (read, difference):
            ttnn.deallocate(tensor)
        k_operand = k_row if self.native_outer_transpose else ttnn.transpose(k_row, -1, -2)
        outer = ttnn.matmul(
            k_operand,
            delta,
            transpose_a=self.native_outer_transpose,
            program_config=self.outer_program_config,
            memory_config=dram,
            compute_kernel_config=self.compute_kernel_config,
        )
        if not self.native_outer_transpose:
            ttnn.deallocate(k_operand)
        ttnn.deallocate(k_row)
        ttnn.deallocate(delta)
        ttnn.add(state, outer, output_tensor=state)
        ttnn.deallocate(outer)
        result = ttnn.matmul(q_row, state, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(q_row)
        return result


class ReuseOuterControlGDN(ReuseOuterGDN):
    """Use the same reuse program after a materialized transpose of K."""

    native_outer_transpose = False


class DefaultOuterGDN(ReuseOuterGDN):
    """Preserve the pre-selection default program and its materialized transpose."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.outer_program_config = None
        return decoder


class ReuseOuterWholeGDN(ReuseOuterGDN):
    """Keep each head's full M matrix together for the reuse readers' batch stride.

    Strip mode can assign several M strips to one core, but the reuse readers
    advance each loop by a whole head matrix. Full-M blocks make their stride
    match the factory work split for every supported request batch.
    """

    outer_rows_per_core = 4


class ReuseOuterWholeControlGDN(ReuseOuterWholeGDN):
    """Whole-head reuse block with a separate transpose, for a matched control."""

    native_outer_transpose = False

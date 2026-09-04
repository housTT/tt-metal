# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unverified decode cast-elision probes; hardware and promotion owned by main.

The recurrence is copied from the integrated fused decoder. Every candidate
inherits the frozen coherent ConvFinalCombination and changes only one
explicit dtype handoff. BF16 weights/activations, K's BF16 scale rounding,
FP32 recurrence/gates, compute configuration and state ownership stay fixed.
"""

import ttnn

from . import final_fusion_candidates as F


class _DecodeCastGDN(F.ConvFinalCombination):
    fold_query_cast = False
    fold_value_cast = False

    def _delta_rule_step(self, q, k, v, beta, g):
        cfg = self.cfg
        batch, nv, dk = q.shape[0], cfg.linear_num_value_heads, cfg.linear_key_head_dim
        dram = ttnn.DRAM_MEMORY_CONFIG
        # Q/K are the existing BF16 joint-RMSNorm outputs. Keep K's rounded
        # BF16 scale result; the two flags isolate only lossless handoffs.
        if self.fold_query_cast:
            if dk != 128:
                raise ValueError("query cast probe requires Ornith's exact1/128 scale")
            q_row = ttnn.multiply(q, dk**-1.0, dtype=ttnn.float32, memory_config=dram)
        else:
            q_scaled = ttnn.multiply(q, dk**-1.0, memory_config=dram)
            q_row = ttnn.typecast(q_scaled, ttnn.float32)
            ttnn.deallocate(q_scaled)
        k_scaled = ttnn.multiply(k, dk**-0.5, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_row = ttnn.typecast(k_scaled, ttnn.float32)
        ttnn.deallocate(k_scaled)
        # v is caller-owned. The mixed subtract borrows it directly and must
        # not release it with the temporaries below.
        v_row = None if self.fold_value_cast else ttnn.typecast(v, ttnn.float32)
        beta_view = ttnn.reshape(beta, [batch, nv, 1, 1])
        g_view = ttnn.reshape(g, [batch, nv, 1, 1])
        state = self.recurrent_state
        ttnn.multiply(state, g_view, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], output_tensor=state)
        read = ttnn.matmul(k_row, state, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        if self.fold_value_cast:
            difference = ttnn.subtract(v, read, dtype=ttnn.float32, memory_config=dram)
        else:
            difference = ttnn.subtract(v_row, read, memory_config=dram)
        delta = ttnn.multiply(difference, beta_view, memory_config=dram)
        for tensor in (read, difference):
            ttnn.deallocate(tensor)
        if v_row is not None:
            ttnn.deallocate(v_row)
        outer = ttnn.matmul(
            k_row, delta, transpose_a=True, memory_config=dram, compute_kernel_config=self.compute_kernel_config
        )
        ttnn.deallocate(k_row)
        ttnn.deallocate(delta)
        ttnn.add(state, outer, output_tensor=state)
        ttnn.deallocate(outer)
        result = ttnn.matmul(q_row, state, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(q_row)
        return result


class QueryCastGDN(_DecodeCastGDN):
    """Emit FP32 directly from BF16 Q times the exact power-of-two1/128 scale."""

    fold_query_cast = True


class ValueCastGDN(_DecodeCastGDN):
    """Subtract FP32 state read directly from BF16 V with explicit FP32 output."""

    fold_value_cast = True


class CombinedCastGDN(_DecodeCastGDN):
    """Combine query/value cast elisions only after both isolated controls."""

    fold_query_cast = True
    fold_value_cast = True


class KeyCastChainGDN(_DecodeCastGDN):
    """Scale K, preserve its BF16 coefficient/product rounding, emit FP32 once."""

    def _delta_rule_step(self, q, k, v, beta, g):
        cfg = self.cfg
        batch, nv, dk = q.shape[0], cfg.linear_num_value_heads, cfg.linear_key_head_dim
        dram = ttnn.DRAM_MEMORY_CONFIG
        # Q/K are the existing BF16 joint-RMSNorm outputs. Keep K's rounded
        # BF16 scale result; the two flags isolate only lossless handoffs.
        if self.fold_query_cast:
            if dk != 128:
                raise ValueError("query cast probe requires Ornith's exact1/128 scale")
            q_row = ttnn.multiply(q, dk**-1.0, dtype=ttnn.float32, memory_config=dram)
        else:
            q_scaled = ttnn.multiply(q, dk**-1.0, memory_config=dram)
            q_row = ttnn.typecast(q_scaled, ttnn.float32)
            ttnn.deallocate(q_scaled)
        if dk != 128 or k.dtype != ttnn.bfloat16:
            raise ValueError("key chain probe requires the Ornith BF16 key norm and width128")
        # BinaryNG's BF16 scalar input rounds1/sqrt128 to181/2048. Unary
        # MUL_UNARY_SFPU instead embeds an FP32 immediate, so preserve that
        # original rounded coefficient explicitly before rounding the product.
        k_row = ttnn.unary_chain(
            k,
            [
                ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 181.0 / 2048.0),
                ttnn.UnaryWithParam(
                    ttnn.UnaryOpType.TYPECAST, ttnn.DataType.FLOAT32.value, ttnn.DataType.BFLOAT16.value
                ),
                ttnn.UnaryWithParam(
                    ttnn.UnaryOpType.TYPECAST, ttnn.DataType.BFLOAT16.value, ttnn.DataType.FLOAT32.value
                ),
            ],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # v is caller-owned. The mixed subtract borrows it directly and must
        # not release it with the temporaries below.
        v_row = None if self.fold_value_cast else ttnn.typecast(v, ttnn.float32)
        beta_view = ttnn.reshape(beta, [batch, nv, 1, 1])
        g_view = ttnn.reshape(g, [batch, nv, 1, 1])
        state = self.recurrent_state
        ttnn.multiply(state, g_view, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], output_tensor=state)
        read = ttnn.matmul(k_row, state, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        if self.fold_value_cast:
            difference = ttnn.subtract(v, read, dtype=ttnn.float32, memory_config=dram)
        else:
            difference = ttnn.subtract(v_row, read, memory_config=dram)
        delta = ttnn.multiply(difference, beta_view, memory_config=dram)
        for tensor in (read, difference):
            ttnn.deallocate(tensor)
        if v_row is not None:
            ttnn.deallocate(v_row)
        outer = ttnn.matmul(
            k_row, delta, transpose_a=True, memory_config=dram, compute_kernel_config=self.compute_kernel_config
        )
        ttnn.deallocate(k_row)
        ttnn.deallocate(delta)
        ttnn.add(state, outer, output_tensor=state)
        ttnn.deallocate(outer)
        result = ttnn.matmul(q_row, state, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(q_row)
        return result


class AllCastGDN(KeyCastChainGDN):
    """Query/value cast elisions plus the independently tested rounded key chain."""

    fold_query_cast = True
    fold_value_cast = True

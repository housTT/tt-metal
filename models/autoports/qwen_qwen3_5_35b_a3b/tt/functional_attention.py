# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Paged attention adapter for request capacities ending on a 32-token tile.

The shared decode helper uses 64-token KV chunks. If the logical page table
covers, for example, three 32-token pages, its last chunk can request a fourth
page-table entry before causal masking. This adapter selects 32-token chunks
for those capacities. Other calls retain Qwen36GatedAttention.forward.

FunctionalDecoder selects one request per call and supplies the original,
unpadded page-table row. Padding needed by chunked prefill belongs only to
that prefill SDPA call, never to paged cache updates or this decode path.
"""

import ttnn
from models.demos.blackhole.qwen36.tt.attention.gated_attention import Qwen36GatedAttention
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import apply_rotary_pos_emb_ttnn


class FunctionalGatedAttention(Qwen36GatedAttention):
    """Reuse existing weights, cache binding, reset behavior, and common paths."""

    @classmethod
    def from_existing(cls, attention):
        """Adapt a loaded attention object without loading or copying tensors."""
        if not isinstance(attention, Qwen36GatedAttention):
            raise TypeError("Expected a loaded Qwen36GatedAttention")
        self = cls.__new__(cls)
        self.__dict__ = attention.__dict__.copy()
        # Same grid, query chunk, and precision as the reused decode helper;
        # only the KV chunk changes to fit all tile-aligned request capacities.
        self._tile_decode_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )
        return self

    def forward(
        self,
        x,
        cos,
        sin,
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
    ):
        use_tile_chunks = (
            self.use_paged_attention
            and x.shape[1] == 1
            and page_table is not None
            and self.paged_kv_cache_key is not None
            and (page_table.shape[1] * self.paged_kv_cache_key.shape[2]) % 64 != 0
        )
        if not use_tile_chunks:
            return super().forward(
                x,
                cos,
                sin,
                position_tensor=position_tensor,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
            )
        return self._decode_tile_chunks(x, cos, sin, position_tensor, page_table)

    def _decode_tile_chunks(self, x, cos, sin, position_tensor, page_table):
        """Shared paged-decode math with only k_chunk_size changed from 64 to 32.

        Projection fidelity, BF16 norm weights, partial RoPE, cache update
        layout, sigmoid gate, and output projection match
        experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py.
        No custom SDPA compute config is supplied, matching its Blackhole path.
        """
        batch, length = x.shape[0], x.shape[1]
        if batch != 1:
            raise ValueError("FunctionalDecoder must select one request for paged attention")
        heads = self.config.num_heads
        kv_heads = self.config.num_kv_heads
        dim = self.config.head_dim
        ckc = self.compute_kernel_config_decode
        mc = ttnn.L1_MEMORY_CONFIG

        qg = ttnn.linear(x, self.weights.q_proj, compute_kernel_config=ckc, memory_config=mc)
        qg = ttnn.reshape(qg, [batch, length, heads, dim * 2])
        query, gate = ttnn.chunk(qg, 2, dim=-1)
        ttnn.deallocate(qg)
        gate = ttnn.reshape(gate, [batch, length, heads * dim])
        query = ttnn.rms_norm(query, weight=self.weights.q_norm, epsilon=self.config.norm_eps)
        query = ttnn.transpose(query, 1, 2)

        key = ttnn.linear(x, self.weights.k_proj, compute_kernel_config=ckc, memory_config=mc)
        key = ttnn.reshape(key, [batch, length, kv_heads, dim])
        key = ttnn.rms_norm(key, weight=self.weights.k_norm, epsilon=self.config.norm_eps)
        key = ttnn.transpose(key, 1, 2)
        value = ttnn.linear(x, self.weights.v_proj, compute_kernel_config=ckc, memory_config=mc)
        value = ttnn.reshape(value, [batch, length, kv_heads, dim])
        value = ttnn.transpose(value, 1, 2)

        if len(cos.shape) == 3:
            cos = ttnn.reshape(cos, [cos.shape[0], 1, 1, cos.shape[-1]])
            sin = ttnn.reshape(sin, [sin.shape[0], 1, 1, sin.shape[-1]])
        query, key = apply_rotary_pos_emb_ttnn(query, key, cos, sin)

        key = ttnn.reshape(key, [1, batch, kv_heads, dim])
        value = ttnn.reshape(value, [1, batch, kv_heads, dim])
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(batch - 1, 0))})
        spec = ttnn.ShardSpec(grid, [32, dim], ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
        key = ttnn.to_memory_config(key, sharded_mc)
        value = ttnn.to_memory_config(value, sharded_mc)
        ttnn.experimental.paged_update_cache(
            self.paged_kv_cache_key,
            key,
            update_idxs_tensor=position_tensor,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            self.paged_kv_cache_value,
            value,
            update_idxs_tensor=position_tensor,
            page_table=page_table,
        )

        q_decode = ttnn.to_memory_config(ttnn.transpose(query, 1, 2), ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_decode,
            self.paged_kv_cache_key,
            self.paged_kv_cache_value,
            cur_pos_tensor=position_tensor,
            page_table_tensor=page_table,
            is_causal=True,
            scale=dim**-0.5,
            program_config=self._tile_decode_program_config,
        )
        output = ttnn.transpose(output, 1, 2)
        ttnn.deallocate(query)
        output = ttnn.transformer.concatenate_heads(output)
        gate = ttnn.sigmoid(gate)
        output = ttnn.multiply(output, gate)
        ttnn.deallocate(gate)
        return ttnn.linear(output, self.weights.o_proj, compute_kernel_config=ckc, memory_config=mc)

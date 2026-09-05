# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in SDPA decode buffer-lifetime experiment for the TP4 production family."""

import ttnn

from ..tt.fused_decoder import _batch_grid, _height_memory


class SdpaLowLiveBuffers:
    """Mixin: release completed decode intermediates and read SDPA Q from DRAM.

    Compose before a production candidate in the MRO. Chunk size and grid come
    from the existing local config, so chunk256 and chunk1024 share this path.
    The native default reduction width is preserved unless a separate trial
    subclass explicitly overrides ``sdpa_max_cores_per_head_batch``.
    """

    sdpa_max_cores_per_head_batch = 16

    def _attention_decode(self, x, current_pos, rot_idxs, page_table):
        if self.k_cache is None or page_table is None:
            raise ValueError("full_attention requires allocated cache and page table")
        batch = int(x.shape[0])
        cfg = self.cfg
        qw = cfg.n_heads * cfg.head_dim
        kvw = 2 * cfg.n_kv_heads * cfg.head_dim
        packed = self._linear(x, "qkvg")
        packed = ttnn.reshape(packed, [1, 1, batch, 2 * qw + kvw])
        qkv = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, batch, qw + kvw], memory_config=ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.slice(
            packed,
            [0, 0, 0, qw + kvw],
            [1, 1, batch, 2 * qw + kvw],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.reshape(gate, [batch, 1, qw])
        ttnn.deallocate(packed)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=cfg.n_heads,
            num_kv_heads=cfg.n_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q_norm = self._norm(q, self.w["q_norm"])
        k_norm = self._norm(k, self.w["k_norm"])
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        q_rot, k_rot = self._decode_rotary(q_norm, k_norm, rot_idxs, batch)
        ttnn.deallocate(q_norm)
        ttnn.deallocate(k_norm)

        # Preserve the existing disjoint K/V update shards and persistent cache.
        kmem = _height_memory(_batch_grid(self.device, batch, offset=batch), cfg.head_dim)
        k_update = ttnn.to_memory_config(k_rot, kmem)
        if k_rot.memory_config() != kmem:
            ttnn.deallocate(k_rot)
        ttnn.experimental.paged_fused_update_cache(
            self.k_cache,
            k_update,
            self.v_cache,
            v,
            update_idxs_tensor=current_pos,
            page_table=page_table,
        )
        ttnn.deallocate(k_update)
        ttnn.deallocate(v)

        # Prime batches can already return DRAM Q; retain that allocation.
        query = ttnn.to_memory_config(q_rot, ttnn.DRAM_MEMORY_CONFIG)
        if q_rot.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            ttnn.deallocate(q_rot)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            is_causal=True,
            scale=cfg.head_dim**-0.5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.decode_sdpa_compute,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*self.optimization.sdpa_grid),
                q_chunk_size=32,
                k_chunk_size=self.optimization.sdpa_chunk,
                exp_approx_mode=False,
                max_cores_per_head_batch=self.sdpa_max_cores_per_head_batch,
            ),
        )
        ttnn.deallocate(query)
        return self._decode_finish(attn, gate, batch)

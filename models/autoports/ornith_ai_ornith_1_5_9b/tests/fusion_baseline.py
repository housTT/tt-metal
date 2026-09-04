# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Frozen experimental parent: accepted attention graph, functional GDN, fused MLP SiLU.

Kept in tests so rejected variants remain reproducible as the runtime evolves.
"""

import ttnn

from ..tt.functional_decoder import FunctionalDecoder, _align_up, _slice_owned


class FusionBaseline(FunctionalDecoder):
    """Device-only fused computation with the functional cache and chunk orchestration."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            return decoder
        cfg = decoder.cfg
        qg = state_dict["self_attn.q_proj.weight"].reshape(cfg.n_heads, 2, cfg.head_dim, cfg.dim)
        q = qg[:, 0].reshape(-1, cfg.dim)
        gate = qg[:, 1].reshape(-1, cfg.dim)
        packed = torch.cat([q, state_dict["self_attn.k_proj.weight"], state_dict["self_attn.v_proj.weight"], gate])
        decoder.w["qkvg"] = ttnn.from_torch(
            packed.T.contiguous(),
            device=decoder.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
        )
        decoder.w["rope_cos_tiled"] = ttnn.to_layout(decoder.rope.cos_table, ttnn.TILE_LAYOUT)
        decoder.w["rope_sin_tiled"] = ttnn.to_layout(decoder.rope.sin_table, ttnn.TILE_LAYOUT)
        decoder.w["batch_idxs"] = ttnn.from_torch(
            torch.arange(32, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=decoder.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
        )
        return decoder

    def _project_qkv(self, x):
        b, t, _ = x.shape
        cfg = self.cfg
        qwidth = cfg.n_heads * cfg.head_dim
        kvwidth = cfg.n_kv_heads * cfg.head_dim
        end = qwidth + 2 * kvwidth
        packed = ttnn.linear(x, self.w["qkvg"], compute_kernel_config=self.compute_kernel_config)
        qkv = ttnn.slice(packed, [0, 0, 0], [b, t, end])
        gate = ttnn.slice(packed, [0, 0, end], [b, t, end + qwidth])
        ttnn.deallocate(packed)
        if t == 1:
            qkv = ttnn.reshape(qkv, [1, 1, b, end])
            qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv,
                num_heads=cfg.n_heads,
                num_kv_heads=cfg.n_kv_heads,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
            q, k, v = [ttnn.permute(ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG), (1, 2, 0, 3)) for a in (q, k, v)]
        else:
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv, num_heads=cfg.n_heads, num_kv_heads=cfg.n_kv_heads, transpose_key=False
            )
        ttnn.deallocate(qkv)
        q = self._norm(q, self.w["q_norm"])
        k = self._norm(k, self.w["k_norm"])
        return q, k, v, gate

    def _apply_partial_rope(self, x, cos, sin):
        b, h, t, d = x.shape
        rd = self.cfg.rope_dim
        rotated = ttnn.slice(x, [0, 0, 0, 0], [b, h, t, rd])
        # Prefill uses one shared table; decode tables can have a distinct row per user.
        if cos.shape[0] == 1:
            out = ttnn.experimental.rotary_embedding(rotated, cos, sin, token_index=0 if t == 1 else None)
        else:
            pieces = []
            for user in range(b):
                xu = ttnn.slice(rotated, [user, 0, 0, 0], [user + 1, h, t, rd])
                cu = ttnn.slice(cos, [user, 0, 0, 0], [user + 1, 1, 1, rd])
                su = ttnn.slice(sin, [user, 0, 0, 0], [user + 1, 1, 1, rd])
                pieces.append(ttnn.experimental.rotary_embedding(xu, cu, su, token_index=0))
            out = ttnn.concat(pieces, dim=0)
        out = ttnn.slice(out, [0, 0, 0, 0], [b, h, t, rd])
        tail = ttnn.slice(x, [0, 0, 0, rd], [b, h, t, d])
        return ttnn.concat([out, tail], dim=-1)

    def _attention_output(self, attn, gate):
        gated = ttnn.multiply(gate, attn, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(attn)
        ttnn.deallocate(gate)
        return ttnn.linear(gated, self.w["o_proj"], compute_kernel_config=self.compute_kernel_config)

    def _block(self, x, *, mode, logical_len=None, page_table=None, chunk_start_idx=0, current_pos=None, rot_idxs=None):
        """One decoder block: norm → mixer → residual → norm → SwiGLU → residual."""
        b, t = x.shape[0], x.shape[1]
        attn_in = self._norm(x, self.w["attn_norm"])
        if self.is_full_attention:
            if mode == "prefill":
                mixed = self._attention_prefill(attn_in, page_table, chunk_start_idx)
            else:
                mixed = self._attention_decode(attn_in, current_pos, rot_idxs, page_table)
        else:
            if mode == "prefill":
                mixed = self._gdn_prefill(attn_in, logical_len)
            else:
                mixed = self._gdn_decode(attn_in)
        ttnn.deallocate(attn_in)

        h = ttnn.add(x, mixed)
        ttnn.deallocate(mixed)

        ff_in = self._norm(h, self.w["ff_norm"])
        gate = ttnn.linear(ff_in, self.w["gate_proj"], compute_kernel_config=self.compute_kernel_config)
        up = ttnn.linear(ff_in, self.w["up_proj"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(ff_in)
        activated = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        ff_out = ttnn.linear(activated, self.w["down_proj"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(activated)
        out = ttnn.add(h, ff_out)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out

    def _rope_decode(self, x, cos, sin):
        b, h, d = list(x.shape)[1:]
        rd = self.cfg.rope_dim
        source = ttnn.slice(x, [0, 0, 0, 0], [1, b, h, rd])
        swapped = ttnn.transpose(source, 1, 2)
        roped = ttnn.experimental.rotary_embedding_hf(
            swapped, cos, sin, compute_kernel_config=self.compute_kernel_config
        )
        out = ttnn.transpose(roped, 1, 2)
        tail = ttnn.slice(x, [0, 0, 0, rd], [1, b, h, d])
        return ttnn.concat([out, tail], dim=-1)

    def _attention_decode(self, x, current_pos, rot_idxs, page_table):
        if self.k_cache is None or page_table is None:
            raise ValueError("full_attention requires allocated cache and page table")
        b = x.shape[0]
        cfg = self.cfg
        qw, kvw = cfg.n_heads * cfg.head_dim, 2 * cfg.n_kv_heads * cfg.head_dim
        packed = ttnn.linear(x, self.w["qkvg"], compute_kernel_config=self.compute_kernel_config)
        packed = ttnn.reshape(packed, [1, 1, b, qw * 2 + kvw])
        qkv = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, b, qw + kvw])
        qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.slice(packed, [0, 0, 0, qw + kvw], [1, 1, b, qw * 2 + kvw])
        gate = ttnn.reshape(gate, [b, 1, qw])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=cfg.n_heads, num_kv_heads=cfg.n_kv_heads, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        q, k = [ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG) for a in (q, k)]
        q = self._norm(q, self.w["q_norm"])
        k = self._norm(k, self.w["k_norm"])
        cos, sin = self.rope.decode_forward(rot_idxs)
        cos, sin = [ttnn.reshape(a, [1, 1, b, int(a.shape[-1])]) for a in (cos, sin)]
        q = self._rope_decode(q, cos, sin)
        k = self._rope_decode(k, cos, sin)
        grid = self.device.compute_with_storage_grid_size()
        whole = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        kgrid = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(b % grid.x, b // grid.x), b, whole, True)
        spec = ttnn.ShardSpec(kgrid, [32, cfg.head_dim], ttnn.ShardOrientation.ROW_MAJOR)
        kmem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
        k = ttnn.to_memory_config(k, kmem)
        ttnn.experimental.paged_fused_update_cache(
            self.k_cache, k, self.v_cache, v, update_idxs_tensor=current_pos, page_table=page_table
        )
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            is_causal=True,
            scale=cfg.head_dim**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=32,
                k_chunk_size=64,
                exp_approx_mode=False,
            ),
        )
        attn = ttnn.reshape(attn, [b, 1, qw])
        return self._attention_output(attn, gate)

    def _attention_prefill(self, x, page_table, chunk_start_idx):
        if self.k_cache is None:
            raise RuntimeError("call allocate_kv_cache()/attach_kv_cache() before prefill")
        if page_table is None:
            raise ValueError("full_attention prefill requires a page_table")
        b, t = x.shape[0], x.shape[1]
        cos, sin = [
            ttnn.slice(self.w[key], [0, 0, chunk_start_idx, 0], [1, 1, chunk_start_idx + t, self.cfg.rope_dim])
            for key in ("rope_cos_tiled", "rope_sin_tiled")
        ]
        cos = ttnn.reshape(cos, [1, 1, t, self.cfg.rope_dim])
        sin = ttnn.reshape(sin, [1, 1, t, self.cfg.rope_dim])

        q, k, v, gate = self._project_qkv(x)
        q = self._apply_partial_rope(q, cos, sin)
        k = self._apply_partial_rope(k, cos, sin)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

        blk0 = chunk_start_idx // self.page_block_size
        blk_n = _align_up(chunk_start_idx + t, self.page_block_size) // self.page_block_size
        chunk_page_table, pt_owned = _slice_owned(page_table, [0, blk0], [int(page_table.shape[0]), blk_n])
        indices, owned = _slice_owned(self.w["batch_idxs"], [0], [b])
        ttnn.experimental.paged_fill_cache(self.k_cache, k, chunk_page_table, batch_idx_tensor=indices)
        ttnn.experimental.paged_fill_cache(self.v_cache, v, chunk_page_table, batch_idx_tensor=indices)
        if owned:
            ttnn.deallocate(indices)
        if pt_owned:
            ttnn.deallocate(chunk_page_table)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            self.k_cache,
            self.v_cache,
            page_table,
            chunk_start_idx,
            scale=self.cfg.head_dim**-0.5,
            program_config=self._prefill_sdpa_config(chunk_start_idx, t),
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        ttnn.deallocate(q)
        attn = ttnn.reshape(ttnn.experimental.nlp_concat_heads(attn), [b, t, self.cfg.n_heads * self.cfg.head_dim])
        return self._attention_output(attn, gate)

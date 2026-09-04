# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated attention graph hypotheses; device results decide whether to keep them.

Every class uses the frozen accepted baseline. The llama candidate deliberately
changes the internal Q/K basis at setup, including prefill and the K cache.
``cache_permutation`` lets the paired harness compare that cache in HF order.
"""

import ttnn

from .fusion_baseline import FusionBaseline as FusedDecoder


def _whole_grid(device):
    size = device.compute_with_storage_grid_size()
    return size, ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(size.x - 1, size.y - 1))])


def _batch_grid(device, batch, offset=0):
    size, whole = _whole_grid(device)
    return ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(offset % size.x, offset // size.x), batch, whole, True
    )


def _height_memory(grid, width, users_per_core=1):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32 * users_per_core, width], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _rectangular_rope_grid(device, batch):
    """HF's native factory executes the bounding rectangle; never leave holes."""
    size = device.compute_with_storage_grid_size()
    choices = [
        (width * height, width, height)
        for height in range(1, size.y + 1)
        for width in range(1, size.x + 1)
        if batch % (width * height) == 0
    ]
    cores, width, height = max(choices)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, height - 1))])
    return grid, batch // cores


class _AttentionDecodeCandidate(FusedDecoder):
    """Frozen decode pipeline with narrowly overridable RoPE and output handoffs."""

    def _decode_rotary(self, q, k, rot_idxs, batch):
        cos, sin = self.rope.decode_forward(rot_idxs)
        cos, sin = [ttnn.reshape(a, [1, 1, batch, self.cfg.rope_dim]) for a in (cos, sin)]
        return self._rope_decode(q, cos, sin), self._rope_decode(k, cos, sin)

    def _decode_finish(self, attn, gate, batch):
        return self._attention_output(ttnn.reshape(attn, [batch, 1, self.cfg.n_heads * self.cfg.head_dim]), gate)

    def _attention_decode(self, x, current_pos, rot_idxs, page_table):
        if self.k_cache is None or page_table is None:
            raise ValueError("full_attention requires allocated cache and page table")
        batch = int(x.shape[0])
        cfg = self.cfg
        qw = cfg.n_heads * cfg.head_dim
        kvw = 2 * cfg.n_kv_heads * cfg.head_dim
        packed = ttnn.linear(x, self.w["qkvg"], compute_kernel_config=self.compute_kernel_config)
        packed = ttnn.reshape(packed, [1, 1, batch, 2 * qw + kvw])
        qkv = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, batch, qw + kvw])
        qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.slice(packed, [0, 0, 0, qw + kvw], [1, 1, batch, 2 * qw + kvw])
        gate = ttnn.reshape(gate, [batch, 1, qw])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=cfg.n_heads,
            num_kv_heads=cfg.n_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        q, k = [ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG) for a in (q, k)]
        q = self._norm(q, self.w["q_norm"])
        k = self._norm(k, self.w["k_norm"])
        q, k = self._decode_rotary(q, k, rot_idxs, batch)
        # V retains the head-split shard on the first B cores. K must be disjoint.
        kmem = _height_memory(_batch_grid(self.device, batch, offset=batch), cfg.head_dim)
        k = ttnn.to_memory_config(k, kmem)
        ttnn.experimental.paged_fused_update_cache(
            self.k_cache,
            k,
            self.v_cache,
            v,
            update_idxs_tensor=current_pos,
            page_table=page_table,
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
        return self._decode_finish(attn, gate, batch)


class NativeShardedHFRope(_AttentionDecodeCandidate):
    """Slice directly into native HF64 shards; keep the learned norm unchanged."""

    def _decode_rotary(self, q, k, rot_idxs, batch):
        grid, users_per_core = _rectangular_rope_grid(self.device, batch)
        rd, dim = self.cfg.rope_dim, self.cfg.head_dim
        rotated_mem = _height_memory(grid, rd, users_per_core)
        tail_mem = _height_memory(grid, dim - rd, users_per_core)
        output_mem = _height_memory(grid, dim, users_per_core)
        cos, sin = self.rope.decode_forward(rot_idxs)
        cos, sin = [
            ttnn.transpose(ttnn.reshape(a, [1, 1, batch, rd]), 1, 2, memory_config=rotated_mem) for a in (cos, sin)
        ]
        outputs = []
        for value in (q, k):
            heads = int(value.shape[2])
            rotated = ttnn.slice(value, [0, 0, 0, 0], [1, batch, heads, rd], memory_config=rotated_mem)
            rotated = ttnn.experimental.rotary_embedding_hf(
                rotated,
                cos,
                sin,
                is_decode_mode=True,
                memory_config=rotated_mem,
                compute_kernel_config=self.compute_kernel_config,
            )
            tail = ttnn.slice(value, [0, 0, 0, rd], [1, batch, heads, dim], memory_config=tail_mem)
            outputs.append(ttnn.concat([rotated, tail], dim=-1, memory_config=output_mem))
        q, k = outputs
        # SDPA's usual sharded-Q geometry has one user per core. Grouped-user
        # rectangles keep native HF RoPE safe for arbitrary B, then interleave Q.
        if users_per_core > 1:
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        return q, k


class FusedLlamaQKRope(_AttentionDecodeCandidate):
    """One decode Q/K64 rotation in the interleaved-pair basis, with BF16 cache."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            return decoder
        cfg = decoder.cfg
        rd, dim = cfg.rope_dim, cfg.head_dim
        rotary_perm = [index for i in range(rd // 2) for index in (i, i + rd // 2)]
        perm = rotary_perm + list(range(rd, dim))
        decoder.cache_permutation = perm

        def upload(value, *, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(
                value.contiguous(),
                device=decoder.device,
                dtype=ttnn.bfloat16,
                layout=layout,
                memory_config=memory_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
            )

        qg = state_dict["self_attn.q_proj.weight"].reshape(cfg.n_heads, 2, dim, cfg.dim)
        q = qg[:, 0][:, perm].reshape(-1, cfg.dim)
        k = state_dict["self_attn.k_proj.weight"].reshape(cfg.n_kv_heads, dim, cfg.dim)
        k = k[:, perm].reshape(-1, cfg.dim)
        gate = qg[:, 1].reshape(-1, cfg.dim)
        packed = torch.cat([q, k, state_dict["self_attn.v_proj.weight"], gate])
        decoder.w["qkvg"] = upload(packed.T)
        for kind in ("q", "k"):
            norm = (state_dict[f"self_attn.{kind}_norm.weight"].float() + 1)[perm]
            decoder.w[f"{kind}_norm"] = upload(norm.reshape(1, 1, 1, dim))

        # Transform the already-pinned tables, including any setup-time scaling.
        # Download is confined to weight setup; all runtime positions stay on device.
        for kind in ("cos", "sin"):
            original = getattr(decoder.rope, f"{kind}_table")
            table = ttnn.to_torch(original)[..., rotary_perm]
            decoder.w[f"llama_{kind}_row"] = upload(table, layout=ttnn.ROW_MAJOR_LAYOUT)
            decoder.w[f"rope_{kind}_tiled"] = upload(table)

        transform = torch.zeros(1, 1, 32, 32)
        even = torch.arange(0, 32, 2)
        transform[..., even, even + 1] = 1
        transform[..., even + 1, even] = -1
        decoder.w["llama_transform_prefill"] = upload(transform)
        # The fused validator permits extra transform cores. Resident identical
        # tiles cover every possible Q/K union for B=1..32, without runtime copies.
        transform_mem = _height_memory(_batch_grid(decoder.device, 64), 32)
        decoder.w["llama_transform_decode"] = upload(transform.repeat(1, 1, 64, 1), memory_config=transform_mem)
        return decoder

    def _apply_partial_rope(self, value, cos, sin):
        batch, heads, length, dim = list(value.shape)
        rd = self.cfg.rope_dim
        rotated = ttnn.slice(value, [0, 0, 0, 0], [batch, heads, length, rd])
        rotated = ttnn.experimental.rotary_embedding_llama(
            rotated,
            cos,
            sin,
            self.w["llama_transform_prefill"],
            is_decode_mode=False,
            compute_kernel_config=self.compute_kernel_config,
        )
        tail = ttnn.slice(value, [0, 0, 0, rd], [batch, heads, length, dim])
        return ttnn.concat([rotated, tail], dim=-1)

    def _decode_rotary(self, q, k, rot_idxs, batch):
        rd, dim = self.cfg.rope_dim, self.cfg.head_dim
        qgrid = _batch_grid(self.device, batch)
        kgrid = _batch_grid(self.device, batch, offset=batch)
        trig_mem = _height_memory(_batch_grid(self.device, 2 * batch), rd)
        # Repeat positions on device. No hard-coded position or per-user host read.
        doubled_idxs = ttnn.concat([rot_idxs, rot_idxs], dim=-1)
        cos, sin = [
            ttnn.embedding(doubled_idxs, self.w[f"llama_{kind}_row"], layout=ttnn.TILE_LAYOUT)
            for kind in ("cos", "sin")
        ]
        cos, sin = [
            ttnn.transpose(ttnn.reshape(a, [1, 1, 2 * batch, rd]), 1, 2, memory_config=trig_mem) for a in (cos, sin)
        ]
        rotated, tails = [], []
        for value, grid in ((q, qgrid), (k, kgrid)):
            heads = int(value.shape[2])
            rotated.append(
                ttnn.slice(
                    value,
                    [0, 0, 0, 0],
                    [1, batch, heads, rd],
                    memory_config=_height_memory(grid, rd),
                )
            )
            tails.append(
                ttnn.slice(
                    value,
                    [0, 0, 0, rd],
                    [1, batch, heads, dim],
                    memory_config=_height_memory(grid, dim - rd),
                )
            )
        q_rot, k_rot = ttnn.experimental.rotary_embedding_llama_fused_qk(
            rotated[0],
            rotated[1],
            cos,
            sin,
            self.w["llama_transform_decode"],
            compute_kernel_config=self.compute_kernel_config,
        )
        return tuple(
            ttnn.concat([rotary, tail], dim=-1, memory_config=_height_memory(grid, dim))
            for rotary, tail, grid in ((q_rot, tails[0], qgrid), (k_rot, tails[1], kgrid))
        )


class ConcatDecode(_AttentionDecodeCandidate):
    """Dedicated head merge including restore to the existing DRAM gate boundary."""

    def _concat_decode(self, attn, batch):
        mem = _height_memory(_batch_grid(self.device, batch), self.cfg.head_dim)
        attn = ttnn.to_memory_config(attn, mem)
        _, whole = _whole_grid(self.device)
        return ttnn.experimental.nlp_concat_heads_decode(attn, num_heads=self.cfg.n_heads, sub_core_grids=whole)

    def _decode_finish(self, attn, gate, batch):
        width = self.cfg.n_heads * self.cfg.head_dim
        merged = self._concat_decode(attn, batch)
        merged = ttnn.to_memory_config(merged, ttnn.DRAM_MEMORY_CONFIG)
        if batch < 32:
            merged = ttnn.slice(merged, [0, 0, 0, 0], [1, 1, batch, width])
        return self._attention_output(ttnn.reshape(merged, [batch, 1, width]), gate)


class ConcatDecodeSharded(ConcatDecode):
    """Keep the merge's width shard through the gate and default output linear."""

    def _decode_finish(self, attn, gate, batch):
        width = self.cfg.n_heads * self.cfg.head_dim
        merged = self._concat_decode(attn, batch)
        gate = ttnn.reshape(gate, [1, 1, batch, width])
        if batch < 32:
            gate = ttnn.pad(gate, [(0, 0), (0, 0), (0, 32 - batch), (0, 0)], value=0.0)
        gate = ttnn.to_memory_config(gate, merged.memory_config())
        out = self._attention_output(merged, gate)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        if batch < 32:
            out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, batch, self.cfg.dim])
        return ttnn.reshape(out, [batch, 1, self.cfg.dim])


class JointHFRope(_AttentionDecodeCandidate):
    """Peer-merge Q/K in the head axis for one interleaved HF64 rotation."""

    def _decode_rotary(self, q, k, rot_idxs, batch):
        cfg = self.cfg
        rd, dim = cfg.rope_dim, cfg.head_dim
        total_heads = cfg.n_heads + cfg.n_kv_heads
        cos, sin = self.rope.decode_forward(rot_idxs)
        cos, sin = [ttnn.reshape(a, [1, 1, batch, rd]) for a in (cos, sin)]
        # Head axis is an outer dimension here, so concatenate does not need
        # to splice the padded head rows of the native decode layout.
        q, k = [ttnn.transpose(a, 1, 2) for a in (q, k)]
        joined = ttnn.concat([q, k], dim=1)
        rotated = ttnn.slice(joined, [0, 0, 0, 0], [1, total_heads, batch, rd])
        rotated = ttnn.experimental.rotary_embedding_hf(
            rotated, cos, sin, compute_kernel_config=self.compute_kernel_config
        )
        tail = ttnn.slice(joined, [0, 0, 0, rd], [1, total_heads, batch, dim])
        joined = ttnn.concat([rotated, tail], dim=-1)
        q = ttnn.slice(joined, [0, 0, 0, 0], [1, cfg.n_heads, batch, dim])
        k = ttnn.slice(joined, [0, cfg.n_heads, 0, 0], [1, total_heads, batch, dim])
        return ttnn.transpose(q, 1, 2), ttnn.transpose(k, 1, 2)


class NativeShardedDirectK(NativeShardedHFRope):
    """Write HF64 K and its tail on a rectangle disjoint from the V-cache cores.

    Unlike the frozen baseline, the cache handoff accepts that rectangle
    directly. If B has no legal rectangle (e.g. B13 on an 11x10 device), native
    HF uses grouped users and still needs the cache's one-user-per-core reshard.
    """

    def _direct_k_grid(self, batch):
        size = self.device.compute_with_storage_grid_size()
        # V occupies the first B row-major cores. Find a filled B-core rectangle
        # entirely after that prefix; no changes to V or the cache representation.
        choices = []
        for width in range(1, size.x + 1):
            if batch % width:
                continue
            height = batch // width
            if height > size.y:
                continue
            for row in range(size.y - height + 1):
                for column in range(size.x - width + 1):
                    if row * size.x + column >= batch:
                        choices.append((row * size.x + column, -width, row, column, height))
        if not choices:
            return None
        _, minus_width, row, column, height = min(choices)
        width = -minus_width
        return ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(column, row), ttnn.CoreCoord(column + width - 1, row + height - 1))]
        )

    def _decode_rotary(self, q, k, rot_idxs, batch):
        kgrid = self._direct_k_grid(batch)
        if kgrid is None:
            return super()._decode_rotary(q, k, rot_idxs, batch)
        qgrid, q_users = _rectangular_rope_grid(self.device, batch)
        rd, dim = self.cfg.rope_dim, self.cfg.head_dim
        qrot_mem = _height_memory(qgrid, rd, q_users)
        cos, sin = self.rope.decode_forward(rot_idxs)
        cos, sin = [
            ttnn.transpose(ttnn.reshape(a, [1, 1, batch, rd]), 1, 2, memory_config=qrot_mem) for a in (cos, sin)
        ]
        outputs = []
        for value, grid, users in ((q, qgrid, q_users), (k, kgrid, 1)):
            heads = int(value.shape[2])
            rotated_mem = _height_memory(grid, rd, users)
            local_cos, local_sin = [ttnn.to_memory_config(a, rotated_mem) for a in (cos, sin)]
            rotated = ttnn.slice(value, [0, 0, 0, 0], [1, batch, heads, rd], memory_config=rotated_mem)
            rotated = ttnn.experimental.rotary_embedding_hf(
                rotated,
                local_cos,
                local_sin,
                is_decode_mode=True,
                compute_kernel_config=self.compute_kernel_config,
            )
            tail = ttnn.slice(
                value, [0, 0, 0, rd], [1, batch, heads, dim], memory_config=_height_memory(grid, dim - rd, users)
            )
            outputs.append(ttnn.concat([rotated, tail], dim=-1, memory_config=_height_memory(grid, dim, users)))
        q, k = outputs
        if q_users > 1:
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        return q, k

    def _attention_decode(self, x, current_pos, rot_idxs, page_table):
        if self.k_cache is None or page_table is None:
            raise ValueError("full_attention requires allocated cache and page table")
        batch = int(x.shape[0])
        cfg = self.cfg
        qw, kvw = cfg.n_heads * cfg.head_dim, 2 * cfg.n_kv_heads * cfg.head_dim
        packed = ttnn.linear(x, self.w["qkvg"], compute_kernel_config=self.compute_kernel_config)
        packed = ttnn.reshape(packed, [1, 1, batch, 2 * qw + kvw])
        qkv = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, batch, qw + kvw])
        qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.slice(packed, [0, 0, 0, qw + kvw], [1, 1, batch, 2 * qw + kvw])
        gate = ttnn.reshape(gate, [batch, 1, qw])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=cfg.n_heads,
            num_kv_heads=cfg.n_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        q, k = [ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG) for a in (q, k)]
        q = self._norm(q, self.w["q_norm"])
        k = self._norm(k, self.w["k_norm"])
        q, k = self._decode_rotary(q, k, rot_idxs, batch)
        if self._direct_k_grid(batch) is None:
            k = ttnn.to_memory_config(k, _height_memory(_batch_grid(self.device, batch, offset=batch), cfg.head_dim))
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
        return self._decode_finish(attn, gate, batch)


class HFPrefillRope(NativeShardedHFRope):
    """HF partial64 prefill kernel with the decoder's explicit compute policy."""

    def _apply_partial_rope(self, value, cos, sin):
        batch, heads, seq, dim = list(value.shape)
        rd = self.cfg.rope_dim
        rotated = ttnn.slice(value, [0, 0, 0, 0], [batch, heads, seq, rd])
        rotated = ttnn.experimental.rotary_embedding_hf(
            rotated,
            cos,
            sin,
            is_decode_mode=False,
            compute_kernel_config=self.compute_kernel_config,
        )
        tail = ttnn.slice(value, [0, 0, 0, rd], [batch, heads, seq, dim])
        return ttnn.concat([rotated, tail], dim=-1)


class SliceL1Attention(NativeShardedHFRope):
    """Write the packed QKV slice directly into the decode head reader L1 input."""

    def _attention_decode(self, x, current_pos, rot_idxs, page_table):
        if self.k_cache is None or page_table is None:
            raise ValueError("full_attention requires allocated cache and page table")
        batch = int(x.shape[0])
        cfg = self.cfg
        qw = cfg.n_heads * cfg.head_dim
        kvw = 2 * cfg.n_kv_heads * cfg.head_dim
        packed = ttnn.linear(x, self.w["qkvg"], compute_kernel_config=self.compute_kernel_config)
        packed = ttnn.reshape(packed, [1, 1, batch, 2 * qw + kvw])
        qkv = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, batch, qw + kvw], memory_config=ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.slice(packed, [0, 0, 0, qw + kvw], [1, 1, batch, 2 * qw + kvw])
        gate = ttnn.reshape(gate, [batch, 1, qw])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=cfg.n_heads,
            num_kv_heads=cfg.n_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        q, k = [ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG) for a in (q, k)]
        q = self._norm(q, self.w["q_norm"])
        k = self._norm(k, self.w["k_norm"])
        q, k = self._decode_rotary(q, k, rot_idxs, batch)
        # V retains the head-split shard on the first B cores. K must be disjoint.
        kmem = _height_memory(_batch_grid(self.device, batch, offset=batch), cfg.head_dim)
        k = ttnn.to_memory_config(k, kmem)
        ttnn.experimental.paged_fused_update_cache(
            self.k_cache,
            k,
            self.v_cache,
            v,
            update_idxs_tensor=current_pos,
            page_table=page_table,
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
        return self._decode_finish(attn, gate, batch)

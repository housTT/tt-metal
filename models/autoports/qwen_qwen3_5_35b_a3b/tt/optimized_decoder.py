# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-device optimized Qwen3.5 decoder, retaining fused request semantics."""

from dataclasses import dataclass, field

from .fused_decoder import FusedDecoder, _CompactExperts, _PackedPrefillExperts


@dataclass(frozen=True)
class OptimizationPolicy:
    """Explicit, serializable per-role tuning controls; no global dtype switch."""

    expert_dtype: str = "bfp8"
    expert_grid_y: int = 2
    expert_grid_x: int = 8
    expert_fidelity: str = "LoFi"
    expert_input_dtype: str = "bf16"
    expert_backend: str = "compact"
    sparse_gate_cores: int = 32
    sparse_gate_k: int = 8
    sparse_gate_subblock: int = 1
    sparse_down_cores: int = 64
    sparse_down_k: int = 8
    sparse_down_subblock: int = 1
    sparse_split: bool = False
    sparse_indexed: bool = False
    sparse_fused_reduce: bool = False
    compact_router_dtype: bool = False
    sparse_down_dtype: str = "bfp8"
    prefill_down_cores: int = 64
    prefill_down_k: int = 1
    prefill_down_subblock: int = 1
    prefill_backend: str = "packed"
    dispatch_dtype: str = "bfp8"
    dispatch_input_dtype: str = "bf16"
    dispatch_grid: tuple = (8, 2)
    dispatch_grid_large: tuple = ()
    dispatch_m_tiles: int = 0
    dispatch_ffn: str = "fused"
    dispatch_hybrid_threshold: int = 0
    dispatch_weight_nd: bool = False
    dispatch_weights: str = "lists"
    plan_cost: tuple = ()
    dispatch_rank: str = "sort"
    prefill_tail_bucket: int = 0
    prefill_buckets: tuple = ()
    decode_input_fill: bool = False
    dispatch_combine: str = "post_combine"
    dispatch_tiles_from: int = 0
    prefill_moe_batch: bool = False
    prefill_moe_pieces: int = 1
    prefill_masked_tail: bool = False
    prefill_sparse_chunk: int = 32
    prefill_sparse_l1: bool = False
    prefill_gate_cores: int = 32
    prefill_gate_k: int = 16
    prefill_gate_subblock: int = 1
    projections: dict = field(default_factory=dict)
    sdpa: dict = field(default_factory=dict)
    kv_dtype: str = "bf16"
    residual_cores: int = 0
    decode_norm_cores: int = 0
    decode_norm_fp32_input: bool = True
    decode_norm_output: str = "l1"
    decode_norm_fill: bool = False
    attention_decode_memory: str = "l1"
    residual_output_sharded: bool = False
    norm_fused_weight: bool = False
    shared_split: bool = False
    gate_pack: bool = False
    gate_pack_grid: tuple = ()
    gdn_l1: bool = False
    gdn_state_l1: bool = False
    gdn_l1_slots: int = 32
    conv_chunk: int = 512
    gdn_heads_decode: bool = False
    conv_state_rm: bool = False
    gdn_ab_split: dict = field(default_factory=dict)
    gdn_flat_qk: bool = False
    release_unused_weights: bool = False
    prefill_tiled: bool = False
    prefill_chunk: int = 0
    shared_minimal_config: dict = field(default_factory=dict)
    shared_decode_minimal_config: dict = field(default_factory=dict)
    prefill_minimal_config: dict = field(default_factory=dict)


from typing import Any, Optional, Sequence


_COMMON_DEFAULTS = {
    "expert_backend": "sparse",
    "sparse_indexed": True,
    "expert_dtype": "bfp4",
    "sparse_gate_cores": 32,
    "sparse_gate_k": 32,
    "sparse_down_cores": 32,
    "sparse_down_k": 16,
    "prefill_down_cores": 32,
    "prefill_down_k": 16,
    "prefill_down_subblock": 2,
    "sparse_fused_reduce": True,
    "compact_router_dtype": True,
    "decode_norm_fp32_input": False,
    "decode_norm_output": "dram",
    "prefill_backend": "dispatch",
    "dispatch_dtype": "bfp8",
    "dispatch_grid": [4, 4],
    "dispatch_grid_large": [2048, 8, 8],
    "dispatch_tiles_from": 2048,
    "prefill_tiled": True,
    "prefill_chunk": 4096,
    "release_unused_weights": True,
    "dispatch_rank": "matmul",
    "prefill_tail_bucket": 32,
    "prefill_buckets": [2048, 1024, 256, 64],
    "decode_input_fill": True,
    "prefill_moe_batch": True,
    "prefill_masked_tail": True,
    "gate_pack": True,
    "gate_pack_grid": [8, 2, 32],
    "sparse_down_dtype": "bfp8",
}

DEFAULT_POLICIES = {
    "full_attention": {
        **_COMMON_DEFAULTS,
        "projections": {
            "attention_in": {"dtype": "bfp8", "fidelity": "LoFi", "prefill_minimal": {"grid": [11, 10], "M_block_size": 4, "K_block_size": 8, "N_block_size": 16, "subblock_h": 1, "subblock_w": 4}},
            "attention_out": {"dtype": "bfp8", "fidelity": "LoFi", "decode_grid": [8, 4], "k_block": 8, "per_n": 2, "prefill_minimal": {"grid": [8, 8], "M_block_size": 8, "K_block_size": 8, "N_block_size": 8, "subblock_h": 2, "subblock_w": 2}},
            "shared_down": {"dtype": "bfp8", "fidelity": "LoFi", "dram": True, "cores": 8, "k_block": 8, "readers": 1, "prefill_grid": [8, 8], "prefill_k": 4},
        },
        "decode_norm_cores": 16,
        "kv_dtype": "bfp8",
        "decode_norm_fill": True,
    },
    "linear_attention": {
        **_COMMON_DEFAULTS,
        "gdn_l1": True,
        "projections": {
            "attention_in": {"dtype": "bf16", "fidelity": "HiFi4", "prefill_minimal": {"grid": [11, 10], "M_block_size": 4, "K_block_size": 8, "N_block_size": 16, "subblock_h": 1, "subblock_w": 4}},
            "attention_out": {"dtype": "bfp8", "fidelity": "HiFi2", "decode_grid": [8, 4], "k_block": 8, "per_n": 2},
            "shared_down": {"dtype": "bfp8", "fidelity": "LoFi", "dram": True, "cores": 8, "k_block": 8, "readers": 1, "prefill_grid": [8, 8], "prefill_k": 4},
        },
        "decode_norm_cores": 0,
        "conv_chunk": 128,
        "conv_state_rm": True,
    },
}

class OptimizedDecoder(FusedDecoder):
    def prefill_forward(
        self,
        hidden_states,
        *,
        valid_len = None,
        page_table = None,
        start_pos: int = 0,
        request_slots = None,
    ):
        """Consume [B,T,H] device activations; return all logical rows [B,T,H].

        start_pos is the absolute cached-prefix length; state is retained until an
        explicit reset. Full attention requires an owned page table [B,blocks].
        Bounded chunks and token-wise tails never write padding into live state.
        Partial text RoPE and chunk planning are owned by this decoder.
        """
        # Fixed aligned inputs use tiled assembly. Padded logical tails retain
        # the fused device-index planner: a new valid_len in a warmed physical
        # bucket must not compile offset-specific slices/variable-size concats
        # while another request's decode trace is alive.
        logical = hidden_states.shape[1] if valid_len is None else int(valid_len)
        if (not self.optimization.prefill_tiled or logical != hidden_states.shape[1]
                or logical % 128 or start_pos % 128):
            return self._planned_prefill(hidden_states,valid_len=valid_len,page_table=page_table,
                                         start_pos=start_pos,request_slots=request_slots)
        import ttnn

        if len(hidden_states.shape) != 3:
            raise ValueError("hidden_states must have shape [B,T,H]")
        batch, physical_len, hidden = hidden_states.shape
        length = physical_len if valid_len is None else int(valid_len)
        if not 0 < length <= physical_len or hidden != self.config.dim:
            raise ValueError("invalid logical length or hidden dimension")
        if start_pos < 0 or start_pos + length > self.config.max_seq_len:
            raise ValueError("prefill exceeds advertised context")
        full = self.config.is_full_attention_layer(self.layer_idx)
        if full and (page_table is None or not self.layer.attention.use_paged_attention):
            raise ValueError("full attention requires bound paged KV and page_table")
        if full and (
            page_table.shape[0] != batch
            or page_table.shape[1] * self.layer.attention.paged_kv_cache_key.shape[2] < start_pos + length
        ):
            raise ValueError("page table does not cover the requested batch/context")
        block = self.layer.attention.paged_kv_cache_key.shape[2] if full else 128
        buckets = self._planner_buckets(full)
        aligned_buckets = tuple(n for n in buckets if n % block == 0 and n % 128 == 0)
        remainder = length
        for count in aligned_buckets:
            remainder %= count
        single_chunks = all(len(group) == 1 and group[0][1] == group[0][2] > 1 for group in self._chunk_plan(
            full, start_pos, length, page_table.shape[1] * block if full else None))
        if not aligned_buckets or start_pos % block or remainder or not single_chunks:
            return self._planned_prefill(hidden_states,valid_len=valid_len,page_table=page_table,
                                         start_pos=start_pos,request_slots=request_slots)
        read_page_table = page_table
        if full and page_table.shape[1] % 8:
            # Chunk SDPA reads 32-byte sticks. Extra entries are outside the
            # logical causal prefix; pad on device without changing ownership.
            read_page_table = ttnn.pad(page_table, ((0, 0), (0, -page_table.shape[1] % 8)), 0)
        # Tiled orchestration removes RM flatten/embedding/untilize/write/tilize
        # crossings while preserving the inherited causal chunk and tail planner.
        rows = []
        for row, slot in enumerate(self._slots(batch, request_slots)):
            self._select_state(slot)
            pt = page_table[row : row + 1, :] if full else None
            read_pt = read_page_table[row : row + 1, :] if full else None
            offset = 0
            pieces = []
            while offset < length:
                absolute = start_pos + offset
                count = next(
                    (
                        n
                        for n in aligned_buckets
                        if n % block == 0 and absolute % block == 0 and absolute % 128 == 0 and length - offset >= n
                    ),
                    1,
                )
                x = hidden_states[row:row+1,offset:offset+count,:]
                kw = {}
                if full:
                    pos = self._indices(absolute, count)
                    c, s = self._rotary(pos, count)
                    if count == 1:
                        kw = dict(
                            cos=c,
                            sin=s,
                            position_tensor=ttnn.reshape(ttnn.typecast(pos, ttnn.int32), (1,)),
                            page_table=pt,
                        )
                    else:
                        kw = dict(
                            cos=ttnn.unsqueeze(c, 1),
                            sin=ttnn.unsqueeze(s, 1),
                            page_table=read_pt,
                            chunk_page_table=ttnn.gather(pt, 1, self._indices(absolute // block, count // block)),
                            chunk_start_idx_tensor=ttnn.reshape(
                                ttnn.typecast(self._indices(absolute, 1), ttnn.int32), (1,)
                            ),
                        )
                y = self.layer.forward(x, mode="decode" if count == 1 else "prefill", chunk_size=128, **kw)
                pieces.append(y)
                offset += count
            rows.append(ttnn.concat(pieces,dim=1) if len(pieces)>1 else pieces[0])
        return ttnn.concat(rows,dim=0) if len(rows)>1 else rows[0]

    def _chunk_plan(self, full, start_pos, length, capacity=None):
        block = self.layer.attention.paged_kv_cache_key.shape[2] if full else 32
        buckets = self._planner_buckets(full)
        costs = tuple(self.optimization.plan_cost or _PLAN_COST["full" if full else "linear"])
        chunk_cost, row_cost, step_cost, group_cost = costs[:4]
        whole_step_cost = costs[4] if len(costs) > 4 else step_cost

        def fits(absolute, n):
            return n % block == 0 and absolute % block == 0 and absolute % min(n, 128) == 0

        def masked(absolute, n):
            return (self.optimization.prefill_masked_tail and fits(absolute, n)
                    and absolute + n <= self.config.max_seq_len
                    and (capacity is None or absolute + n <= capacity))

        plan, offset = [], 0
        while offset < length and (start_pos + offset) % max(block, 32):
            plan.append((offset, 1, 1))
            offset += 1
        while length - offset >= buckets[0] and fits(start_pos + offset, buckets[0]):
            plan.append((offset, buckets[0], buckets[0]))
            offset += buckets[0]
        best = {length: (0.0, ())}
        for p in reversed(range(offset, length, 32)):
            rem, absolute = length - p, start_pos + p
            options = []
            if rem <= 8:
                per_step = whole_step_cost if p == offset == 0 else step_cost
                options.append((rem * per_step - (group_cost if p == offset else 0.0),
                                tuple((p + i, 1, 1) for i in range(rem))))
            for n in buckets:
                if n <= rem and fits(absolute, n) and p + n in best:
                    cost, tail = best[p + n]
                    options.append((chunk_cost + row_cost * n + cost, ((p, n, n),) + tail))
                elif n > rem > 1 and masked(absolute, n):
                    options.append((chunk_cost + row_cost * n, ((p, n, rem),)))
            if not options and rem < 32:
                options.append((rem * step_cost, tuple((p + i, 1, 1) for i in range(rem))))
            if options:
                best[p] = min(options)
        if offset < length:
            if offset not in best:
                while offset < length:
                    plan.append((offset, 1, 1))
                    offset += 1
            else:
                plan.extend(best[offset][1])
        groups, current, tokens = [], [], 0
        for offset, count, valid in plan:
            if current and (tokens + count > buckets[0] or count == 1 or current[-1][1] == 1):
                groups.append(current)
                current, tokens = [], 0
            current.append((offset, count, valid))
            tokens += count
        groups.append(current)
        return groups

    def _attention_step(self, x, count, kw, valid=None):
        import ttnn
        from models.tt_transformers.tt.common import Mode

        layer = self.layer
        if valid is not None and valid != count and not self.config.is_full_attention_layer(self.layer_idx):
            row = ttnn.lt(self._index_offsets[count], float(valid))
            layer.attention._tail_mask = ttnn.transpose(ttnn.reshape(row, (1, 1, count)), -2, -1)
            layer.attention._tail_rows = self._indices(valid, 3)
        else:
            layer.attention._tail_mask = layer.attention._tail_rows = None
        decode = count == 1
        if decode and self.optimization.decode_input_fill:
            x = ttnn.fill_implicit_tile_padding(x, 0.0)
        attn_input = layer.attention_norm(x, mode=Mode.DECODE if decode else Mode.PREFILL,
                                          norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG} if decode else None)
        if self.config.is_full_attention_layer(self.layer_idx):
            attn = layer.attention.forward(attn_input, kw["cos"], kw["sin"], position_tensor=kw.get("position_tensor"),
                                           page_table=kw.get("page_table"), chunk_page_table=kw.get("chunk_page_table"),
                                           chunk_start_idx_tensor=kw.get("chunk_start_idx_tensor"))
        else:
            attn = layer.attention.forward(attn_input, mode="recurrent" if decode else "chunk", chunk_size=128,
                                           valid_len=valid if valid is not None and valid != count else None)
            layer.attention._tail_mask = layer.attention._tail_rows = None
        return ttnn.add(x, attn)

    def _batched_prefill(self, hidden_states, *, valid_len=None, page_table=None, start_pos=0, request_slots=None):
        import ttnn
        from models.tt_transformers.tt.common import Mode

        if len(hidden_states.shape) != 3:
            raise ValueError("hidden_states must have shape [B,T,H]")
        batch, physical_len, hidden = hidden_states.shape
        length = physical_len if valid_len is None else int(valid_len)
        if not 0 < length <= physical_len or hidden != self.config.dim:
            raise ValueError("invalid logical length or hidden dimension")
        if start_pos < 0 or start_pos + length > self.config.max_seq_len:
            raise ValueError("prefill exceeds advertised context")
        full = self.config.is_full_attention_layer(self.layer_idx)
        if full and (page_table is None or not self.layer.attention.use_paged_attention):
            raise ValueError("full attention requires bound paged KV and page_table")
        if full and (
            page_table.shape[0] != batch
            or page_table.shape[1] * self.layer.attention.paged_kv_cache_key.shape[2] < start_pos + length
        ):
            raise ValueError("page table does not cover the requested batch/context")
        block = self.layer.attention.paged_kv_cache_key.shape[2] if full else 32
        scratch = getattr(self, "kv_scratch_block", None) if full else None
        decode_page_table = page_table
        if scratch is not None:
            extra = max(self._planner_buckets(full)) // block
            page_table = ttnn.concat([page_table, ttnn.full((batch, extra), scratch, dtype=ttnn.int32,
                                                             layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device_ref)],
                                     dim=1)
        read_page_table = page_table
        if full and page_table.shape[1] % 8:
            read_page_table = ttnn.pad(page_table, ((0, 0), (0, -page_table.shape[1] % 8)), 0)
        groups = self._chunk_plan(full, start_pos, length,
                                  page_table.shape[1] * block if full else None)
        planned = max(offset + count for group in groups for offset, count, _ in group)
        sizes = sorted(set(_bucket_sizes(self.optimization)) | {128, 512})
        rows_alloc = -(-max(planned, physical_len) // 32) * 32
        source = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        if rows_alloc > physical_len:
            source = ttnn.pad(source, ((0, 0), (0, rows_alloc - physical_len), (0, 0)), 0.0)
        flat_input = ttnn.reshape(ttnn.to_layout(source, ttnn.ROW_MAJOR_LAYOUT), (batch * rows_alloc, hidden))
        tail = 0
        while tail < len(groups) and groups[-1 - tail][0][1] == 1:
            tail += 1
        tiled_only = 0 < tail <= 31 or tail == 0
        tiled_only = tiled_only and all(count % 32 == 0 for group in groups[: len(groups) - tail]
                                        for _, count, _ in group)
        output = None if tiled_only else ttnn.zeros((batch, rows_alloc, hidden), dtype=ttnn.bfloat16,
                                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device_ref,
                                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rows = []
        for row, slot in enumerate(self._slots(batch, request_slots)):
            self._select_state(slot)
            pt = page_table[row : row + 1, :] if full else None
            read_pt = read_page_table[row : row + 1, :] if full else None
            decode_pt = decode_page_table[row : row + 1, :] if full else None

            def step_input(offset, count):
                absolute = start_pos + offset
                x = ttnn.embedding(self._indices(row * rows_alloc + offset, count), flat_input,
                                   layout=ttnn.TILE_LAYOUT)
                if not full:
                    return x, {}
                pos = self._indices(absolute, count)
                c, s_ = self._rotary(pos, count)
                if count == 1:
                    return x, dict(cos=c, sin=s_, page_table=decode_pt,
                                   position_tensor=ttnn.reshape(ttnn.typecast(pos, ttnn.int32), (1,)))
                return x, dict(cos=ttnn.unsqueeze(c, 1), sin=ttnn.unsqueeze(s_, 1), page_table=read_pt,
                               chunk_page_table=ttnn.gather(pt, 1, self._indices(absolute // block, count // block)),
                               chunk_start_idx_tensor=ttnn.reshape(
                                   ttnn.typecast(self._indices(absolute, 1), ttnn.int32), (1,)))

            pieces, tail_rows = [], []
            for group in groups:
                group_start = group[0][0]
                if len(group) == 1 and group[0][1] == 1:
                    x, kw = step_input(group_start, 1)
                    y = self.layer.forward(x, mode="decode", chunk_size=128, **kw)
                    if tiled_only:
                        tail_rows.append(ttnn.reshape(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), (1, 1, hidden)))
                    else:
                        ttnn.experimental.slice_write(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), output,
                                                      (row, group_start, 0), (row + 1, group_start + 1, hidden),
                                                      (1, 1, 1))
                    continue
                hs = []
                for offset, count, valid in group:
                    x, kw = step_input(offset, count)
                    hs.append(self._attention_step(x, count, kw, valid))
                residual = ttnn.concat(hs, dim=1) if len(hs) > 1 else hs[0]
                padded = sum(count for _, count, _ in group)
                parts, start = [], 0
                for moe_rows in _moe_pieces(padded, sizes, self.optimization.prefill_moe_pieces):
                    rows_here = min(moe_rows, padded - start)
                    part = residual if rows_here == padded else ttnn.slice(residual, (0, start, 0),
                                                                            (1, start + rows_here, hidden))
                    if moe_rows > rows_here:
                        part = ttnn.pad(part, ((0, 0), (0, moe_rows - rows_here), (0, 0)), 0.0)
                    ff_input = self.layer.ffn_norm(part, mode=Mode.PREFILL, norm_config=None)
                    dispatch = getattr(self, "expert_dispatch", None)
                    if dispatch is not None:
                        dispatch.valid_rows = rows_here
                    try:
                        part = ttnn.add(part, self.layer.feed_forward.forward(ff_input, mode="prefill"))
                    finally:
                        if dispatch is not None:
                            dispatch.valid_rows = None
                    if isinstance(getattr(self, "moe_pieces_log", None), list):
                        self.moe_pieces_log.append((moe_rows, rows_here))
                    if moe_rows > rows_here:
                        part = ttnn.slice(part, (0, 0, 0), (1, rows_here, hidden))
                    parts.append(part)
                    start += rows_here
                result = ttnn.concat(parts, dim=1) if len(parts) > 1 else parts[0]
                if tiled_only:
                    pieces.append(result)
                else:
                    ttnn.experimental.slice_write(ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT), output,
                                                  (row, group_start, 0), (row + 1, group_start + padded, hidden),
                                                  (1, 1, 1))
            if tiled_only:
                if tail_rows:
                    tail_piece = ttnn.concat(tail_rows, dim=1) if len(tail_rows) > 1 else tail_rows[0]
                    tail_piece = ttnn.to_layout(ttnn.pad(tail_piece, ((0, 0), (0, 32 - len(tail_rows)), (0, 0)), 0.0),
                                                ttnn.TILE_LAYOUT)
                    pieces.append(tail_piece)
                rows.append(ttnn.concat(pieces, dim=1) if len(pieces) > 1 else pieces[0])
        tiled = (ttnn.to_layout(output, ttnn.TILE_LAYOUT) if not tiled_only
                 else ttnn.concat(rows, dim=0) if len(rows) > 1 else rows[0])
        if len(tiled.shape) == 4:
            tiled = ttnn.reshape(tiled, (batch, tiled.shape[-2], hidden))
        return ttnn.reshape(tiled, (batch, length, hidden), tiled.padded_shape)

    def _planner_buckets(self, full):
        base = {512, 128} if full else {128}
        return tuple(sorted(base | set(_bucket_sizes(self.optimization)), reverse=True))

    def _planned_prefill(
        self,
        hidden_states: Any,
        *,
        valid_len: Optional[int] = None,
        page_table: Any = None,
        start_pos: int = 0,
        request_slots: Optional[Sequence[int]] = None,
    ) -> Any:
        import ttnn

        if self.optimization.prefill_moe_batch and (hidden_states.shape[1] if valid_len is None else valid_len) > 1:
            return self._batched_prefill(hidden_states, valid_len=valid_len, page_table=page_table,
                                         start_pos=start_pos, request_slots=request_slots)
        if len(hidden_states.shape) != 3:
            raise ValueError("hidden_states must have shape [B,T,H]")
        batch, physical_len, hidden = hidden_states.shape
        length = physical_len if valid_len is None else int(valid_len)
        if not 0 < length <= physical_len or hidden != self.config.dim:
            raise ValueError("invalid logical length or hidden dimension")
        if start_pos < 0 or start_pos + length > self.config.max_seq_len:
            raise ValueError("prefill exceeds advertised context")
        full = self.config.is_full_attention_layer(self.layer_idx)
        if full and (page_table is None or not self.layer.attention.use_paged_attention):
            raise ValueError("full attention requires bound paged KV and page_table")
        if full and (
            page_table.shape[0] != batch
            or page_table.shape[1] * self.layer.attention.paged_kv_cache_key.shape[2] < start_pos + length
        ):
            raise ValueError("page table does not cover the requested batch/context")
        read_page_table = page_table
        if full and page_table.shape[1] % 8:
            read_page_table = ttnn.pad(page_table, ((0, 0), (0, -page_table.shape[1] % 8)), 0)
        flat_input = ttnn.reshape(ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT), (batch * physical_len, hidden))
        output = ttnn.to_layout(
            ttnn.zeros_like(ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT), memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        for row, slot in enumerate(self._slots(batch, request_slots)):
            self._select_state(slot)
            pt = page_table[row : row + 1, :] if full else None
            read_pt = read_page_table[row : row + 1, :] if full else None
            block = self.layer.attention.paged_kv_cache_key.shape[2] if full else 32
            offset = 0
            while offset < length:
                absolute = start_pos + offset
                buckets = self._planner_buckets(full)
                count = next(
                    (
                        n
                        for n in buckets
                        if n % block == 0 and absolute % block == 0 and absolute % min(n, 128) == 0 and length - offset >= n
                    ),
                    1,
                )
                x = ttnn.embedding(
                    self._indices(row * physical_len + offset, count), flat_input, layout=ttnn.TILE_LAYOUT
                )
                kw = {}
                if full:
                    pos = self._indices(absolute, count)
                    c, s = self._rotary(pos, count)
                    if count == 1:
                        kw = dict(
                            cos=c,
                            sin=s,
                            position_tensor=ttnn.reshape(ttnn.typecast(pos, ttnn.int32), (1,)),
                            page_table=pt,
                        )
                    else:
                        kw = dict(
                            cos=ttnn.unsqueeze(c, 1),
                            sin=ttnn.unsqueeze(s, 1),
                            page_table=read_pt,
                            chunk_page_table=ttnn.gather(pt, 1, self._indices(absolute // block, count // block)),
                            chunk_start_idx_tensor=ttnn.reshape(
                                ttnn.typecast(self._indices(absolute, 1), ttnn.int32), (1,)
                            ),
                        )
                y = self.layer.forward(x, mode="decode" if count == 1 else "prefill", chunk_size=128, **kw)
                ttnn.experimental.slice_write(
                    ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT),
                    output,
                    (row, offset, 0),
                    (row + 1, offset + count, hidden),
                    (1, 1, 1),
                )
                offset += count
        tiled = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        return ttnn.reshape(tiled, (batch, length, hidden), tiled.padded_shape)

    def bind_paged_kv_cache(self, key_cache, value_cache, scratch_block=None):
        import ttnn

        if not self.config.is_full_attention_layer(self.layer_idx):
            raise TypeError("linear-attention layers use recurrent state, not paged KV")
        if tuple(key_cache.shape) != tuple(value_cache.shape) or len(key_cache.shape) != 4:
            raise ValueError("K/V caches must share [pages,kv_heads,page_size,head_dim]")
        if key_cache.shape[1] != self.config.n_kv_heads or key_cache.shape[3] != self.config.head_dim:
            raise ValueError("K/V cache head geometry differs from target config")
        if key_cache.shape[2] < 32 or key_cache.shape[2] % 32:
            raise ValueError("K/V page size must be a positive multiple of32")
        if key_cache.dtype != value_cache.dtype or any(
            t.dtype not in (ttnn.bfloat16,ttnn.bfloat8_b) or t.layout != ttnn.TILE_LAYOUT
            or t.memory_config() != ttnn.DRAM_MEMORY_CONFIG for t in (key_cache,value_cache)
        ):
            raise ValueError("Optimized K/V caches require matching BF16/BFP8 tiled DRAM")
        if scratch_block is not None and not 0 <= scratch_block < key_cache.shape[0]:
            raise ValueError("scratch_block must index a page of the bound K/V pool")
        self.layer.attention.set_paged_kv_cache(key_cache,value_cache)
        self.kv_scratch_block = scratch_block

    @classmethod
    def from_state_dict(cls, *args, optimization=None, **kwargs):
        result = super().from_state_dict(*args, **kwargs)
        if optimization is None:
            kind = "full_attention" if result.config.is_full_attention_layer(result.layer_idx) else "linear_attention"
            optimization = DEFAULT_POLICIES[kind]
        policy = OptimizationPolicy(**optimization)
        result.optimization = policy
        result.mesh_device_ref = kwargs["mesh_device"]
        if _bucket_sizes(policy):
            import torch
            import ttnn

            counts = {c for n in _bucket_sizes(policy) for c in (n, n // 32, n // 128) if c} | {3}
            for count in counts - result._index_offsets.keys():
                result._index_offsets[count] = ttnn.from_torch(torch.arange(count,dtype=torch.float32)[None],
                    device=kwargs["mesh_device"],dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(kwargs["mesh_device"]))
        moe = result.layer.feed_forward.module
        if policy.prefill_chunk and policy.prefill_chunk not in moe.router.zero_templates:
            import torch
            import ttnn

            moe.router.zero_templates[policy.prefill_chunk] = ttnn.from_torch(
                torch.zeros((1,1,policy.prefill_chunk,256),dtype=torch.bfloat16),
                device=kwargs["mesh_device"],dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(kwargs["mesh_device"]))
        packed = moe.experts
        if not isinstance(packed, _PackedPrefillExperts) or not isinstance(packed.original, _CompactExperts):
            raise ValueError("Optimized decoder requires the qualified packed-prefill/compact-decode graph")
        compact = _OptimizedCompactExperts.__new__(_OptimizedCompactExperts)
        compact.__dict__.update(packed.original.__dict__)
        compact.policy = policy
        import ttnn

        if policy.expert_dtype not in ("bfp4", "bfp8"):
            raise ValueError("expert_dtype must be bfp4 or bfp8")
        if policy.expert_dtype == "bfp4":
            for role in ("gate", "up", "down"):
                setattr(compact, role, [ttnn.typecast(t, ttnn.bfloat4_b) for t in getattr(compact, role)])
        compact.compute = ttnn.init_device_compute_kernel_config(
            kwargs["mesh_device"].arch(),
            math_fidelity=getattr(ttnn.MathFidelity, policy.expert_fidelity),
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=False,
        )
        if policy.expert_backend == "sparse":
            sparse = _OptimizedSparseExperts.__new__(_OptimizedSparseExperts)
            sparse.__dict__.update(compact.__dict__)
            sparse.prepare(kwargs["mesh_device"])
            compact = sparse
        elif policy.expert_backend != "compact":
            raise ValueError("expert_backend must be compact or sparse")
        experts = _OptimizedPrefillExperts.__new__(_OptimizedPrefillExperts)
        experts.__dict__.update(packed.__dict__)
        experts.original = compact
        experts.policy = policy
        experts.down_compute = ttnn.init_device_compute_kernel_config(
            kwargs["mesh_device"].arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        if policy.prefill_backend == "dispatch":
            if policy.dispatch_weights == "lists":
                experts.dispatch = _LocalDispatch(packed.original,kwargs["mesh_device"],policy)
            elif policy.dispatch_weights in ("stacked", "expert_parallel") and policy.expert_backend == "sparse":
                experts.dispatch = _StackedDispatch(compact,kwargs["mesh_device"],policy)
            else:
                raise ValueError("dispatch_weights must be lists, or stacked/expert_parallel with sparse decode experts")
        moe.experts = experts
        if (policy.shared_split or policy.shared_minimal_config or policy.shared_decode_minimal_config or policy.gate_pack
                or any(k.startswith("shared_") for k in policy.projections)):
            shared = _OptimizedShared.__new__(_OptimizedShared)
            shared.__dict__.update(moe.shared.__dict__)
            shared.policy = policy
            shared.projections = {}
            weights = {"shared_gate":shared.gate_weight, "shared_down":shared.mlp.weights.w2,
                       "shared_gate_proj":shared.mlp.weights.w1, "shared_up_proj":shared.mlp.weights.w3}
            for role, weight in weights.items():
                if role in policy.projections:
                    shared.projections[role] = _Projection(weight,kwargs["mesh_device"],policy.projections[role])
            moe.shared = shared
        if policy.compact_router_dtype or policy.prefill_backend == "dispatch" or policy.gate_pack:
            router = _OptimizedRouter.__new__(_OptimizedRouter)
            router.__dict__.update(moe.router.__dict__)
            router.tuple_prefill = policy.prefill_backend == "dispatch"
            moe.router = router
        if "router" in policy.projections or policy.gate_pack:
            gate = _OptimizedGateModule(moe.router.module,kwargs["mesh_device"],
                                        policy.projections.get("router",{"dtype":"bf16","fidelity":"HiFi4"}))
            moe.router.module = gate
        if policy.gate_pack:
            result.layer.feed_forward.module = _PackedGateMoE(moe,kwargs["mesh_device"],policy)
        attention_class = _OptimizedAttention if result.config.is_full_attention_layer(result.layer_idx) else _OptimizedGDN
        attention = attention_class.__new__(attention_class)
        attention.__dict__.update(result.layer.attention.__dict__)
        attention.optimized_projections = {}
        role_weights = ({"attention_in": attention.qkvg_weight, "attention_out": attention.weights.o_proj}
                        if result.config.is_full_attention_layer(result.layer_idx)
                        else {"attention_in": attention.mega_weight, "attention_out": attention.output_weight})
        for role, weight in role_weights.items():
            if role in policy.projections:
                attention.optimized_projections[role] = _Projection(weight, kwargs["mesh_device"], policy.projections[role])
        if policy.gdn_ab_split and not result.config.is_full_attention_layer(result.layer_idx):
            attention.split_projection = _GdnSplitProjection(attention, kwargs["mesh_device"], policy.gdn_ab_split)
        attention.sdpa_policy = policy.sdpa
        attention.optimization = policy
        if policy.gdn_state_l1 and not result.config.is_full_attention_layer(result.layer_idx):
            result._states = [(ttnn.to_memory_config(state,ttnn.L1_MEMORY_CONFIG),conv)
                              if slot < policy.gdn_l1_slots else (state,conv)
                              for slot,(state,conv) in enumerate(result._states)]
            attention.recurrent_state,attention.fused_conv_state = result._states[0]
        if policy.conv_state_rm and not result.config.is_full_attention_layer(result.layer_idx):
            rm = lambda t: ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            result._states = [(state, rm(conv)) for state, conv in result._states]
            attention._zero_conv = rm(attention._zero_conv)
            attention.recurrent_state, attention.fused_conv_state = result._states[0]
        result.layer.attention = attention
        if policy.decode_norm_cores:
            result.layer.attention_norm = _DecodeNorm(result.layer.attention_norm, policy)
            result.layer.ffn_norm = _DecodeNorm(result.layer.ffn_norm, policy)
        if policy.residual_cores:
            result.layer = _ShardedLayer(result.layer, kwargs["mesh_device"], policy)
        if policy.decode_input_fill:
            result.layer = _ZeroPaddedDecodeLayer(result.layer)
        if policy.release_unused_weights:
            result.weight_memory = {"before_cleanup":_memory_bytes(kwargs["mesh_device"])}
            # Keep only live consumers. Drop Python owners, never force-deallocate
            # a potentially aliased TT buffer (weights are shared across phases).
            live = {"original","policy","weights","config","down_compute","compute","dispatch"}
            if policy.prefill_backend == "packed":
                live.update(("packed_groups","prefill_sparsity"))
            experts.__dict__ = {k:v for k,v in experts.__dict__.items() if k in live}
            if policy.expert_backend == "sparse":
                live = {"policy","compute","gate_up_weight","down_weight","up_weight","gate_weight",
                        "ones","zeros","placeholder","mapping"}
            else:
                live = {"policy","compute","gate","up","down","idx_template","slot_labels",
                        "compact_counts","compact_offsets","mapping"}
            compact.__dict__ = {k:v for k,v in compact.__dict__.items() if k in live}
            packed.__dict__.clear()
            out_projection = attention.optimized_projections.get("attention_out")
            if (not result.config.is_full_attention_layer(result.layer_idx) and out_projection is not None
                    and not out_projection.parts and out_projection.weight is not attention.output_weight):
                attention.output_weight = None
                role_weights.clear()
            if (result.config.is_full_attention_layer(result.layer_idx)
                    and {"qkv", "heads", "decode_heads"}.intersection(attention.fusions)):
                from dataclasses import replace

                attention.weights = replace(attention.weights, q_proj=None, k_proj=None, v_proj=None)
            import gc

            gc.collect()
            result.weight_memory["after_cleanup"] = _memory_bytes(kwargs["mesh_device"])
        return result


class _PackedGateMoE:
    def __init__(self, moe, mesh, policy):
        import torch
        import ttnn

        self.moe, self.policy = moe, policy
        router = ttnn.to_torch(ttnn.get_device_tensors(moe.router.module.original.tt_gate_weight)[0]).reshape(2048, -1)[:, :256]
        shared = ttnn.to_torch(ttnn.get_device_tensors(moe.shared.gate_weight)[0]).reshape(2048, 1)
        packed = torch.cat((router, torch.nn.functional.pad(shared, (0, 31))), dim=-1)
        self.weight = ttnn.from_torch(packed.reshape(1, 1, 2048, 288).to(torch.bfloat16), device=mesh,
                                      dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                      mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        self.program = None
        if policy.gate_pack_grid:
            gx, gy, k_block = policy.gate_pack_grid
            per_n = -(-9 // (gx * gy))
            self.program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(gx, gy), in0_block_w=k_block, per_core_M=1, per_core_N=per_n,
                out_subblock_h=1, out_subblock_w=per_n, out_block_h=1, out_block_w=per_n, fuse_batch=True,
                fused_activation=None, mcast_in0=True)

    def __getattr__(self, name):
        return getattr(self.__dict__["moe"], name)

    def forward(self, x, mode="decode"):
        import ttnn

        if x.shape[2] != 1:
            return self.moe.forward(x, mode=mode)
        logits = ttnn.linear(x, self.weight, compute_kernel_config=self.compute, program_config=self.program,
                             memory_config=ttnn.L1_MEMORY_CONFIG)
        values, indices = self.moe.router.module.route(logits[:, :, :, :256])
        routing = (ttnn.permute(values, (1, 2, 0, 3)), ttnn.permute(indices, (1, 2, 0, 3)))
        out = self.moe.experts(x, routing, mode=mode)
        return ttnn.add(out, self.moe.shared.forward(x, gate=logits[:, :, :, 256:257]))


class _OptimizedShared:
    def forward(self, x, gate=None):
        import ttnn

        def project(role, weight):
            if role in self.projections:
                return self.projections[role](x,ttnn.L1_MEMORY_CONFIG)
            return ttnn.linear(x,weight,compute_kernel_config=self.mlp.compute_kernel_config_decode,
                               memory_config=ttnn.L1_MEMORY_CONFIG)
        if gate is None:
            gate = project("shared_gate",self.gate_weight)
        if self.policy.shared_split:
            ff_gate = project("shared_gate_proj",self.mlp.weights.w1)
            ff_up = project("shared_up_proj",self.mlp.weights.w3)
            activated = ttnn.multiply(ff_gate,ff_up,input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                                     memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            config = self.policy.shared_minimal_config
            if x.shape[-2] == 1 and self.policy.shared_decode_minimal_config:
                config = self.policy.shared_decode_minimal_config
            activated = ttnn.experimental.minimal_matmul(x,self.packed,fuse_swiglu=True,
                config=_minimal_config(config),
                compute_kernel_config=self.mlp.compute_kernel_config_decode,memory_config=ttnn.L1_MEMORY_CONFIG)
        if "shared_down" in self.projections:
            output = self.projections["shared_down"](activated,ttnn.L1_MEMORY_CONFIG)
        else:
            output = ttnn.linear(activated,self.mlp.weights.w2,
                compute_kernel_config=self.mlp.compute_kernel_config_decode,memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.multiply(gate,output[:,:,:x.shape[2],:],input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])


class _OptimizedGateModule:
    def __init__(self, original, mesh, options):
        self.original = original
        self.projection = _Projection(original.tt_gate_weight,mesh,options)
        if original.n_group != 1 or original.num_blocks != 1 or original.k != 8 or original.score_transform:
            raise ValueError("Optimized router is the target256-expert top8 softmax-post contract")

    def forward(self, x):
        import ttnn

        return self.route(self.projection(x,ttnn.L1_MEMORY_CONFIG))

    def route(self, logits):
        import ttnn

        base = self.original
        total = logits.shape[2]
        chunk = min(base.num_device_cores,base._buffer_rows)
        iterations = (total+chunk-1)//chunk
        padding = (iterations-total%iterations)%iterations
        width = (total+padding)//iterations
        if padding:
            logits = ttnn.pad(logits,((0,0),(0,0),(0,padding),(0,0)),0)
        mem_in = base._sharded_mem_config(width,(32,32))
        bias = ttnn.slice(base.tt_bias,[0,0,0],[width,16,16],memory_config=mem_in)
        indices = ttnn.slice(base.tt_input_indices,[0,0,0],[width,16,16],memory_config=mem_in)
        out = ttnn.slice(base.tt_output,[0,0,0],[width,32,32],memory_config=mem_in)
        out_idx = ttnn.slice(base.tt_output_indices,[0,0,0],[width,32,32],memory_config=mem_in)
        values_list, indices_list = [], []
        for start in range(0,total+padding,width):
            cur = ttnn.to_memory_config(ttnn.reshape(logits[:,:,start:start+width,:],(width,16,16)),mem_in)
            values, idx = ttnn.experimental.deepseek.moe.generalized_moe_gate(cur,bias_tensor=bias,
                input_indices_tensor=indices,output_tensor=out,output_indices_tensor=out_idx,
                eps=base.eps,scaling_factor=base.scaling_factor,enable_sigmoid=base.enable_sigmoid,
                topk=8,output_softmax=base.output_softmax)
            values_list.append(ttnn.to_memory_config(values,ttnn.L1_MEMORY_CONFIG))
            indices_list.append(ttnn.to_memory_config(idx,ttnn.L1_MEMORY_CONFIG))
        values = values_list[0] if iterations==1 else ttnn.concat(values_list,dim=0)
        indices = indices_list[0] if iterations==1 else ttnn.concat(indices_list,dim=0)
        return (ttnn.view(values[:total,0,:8],(total,1,1,8)),
                ttnn.view(indices[:total,0,:8],(total,1,1,8)))


class _ShardedNorm:
    def __init__(self, original, memory, policy):
        import ttnn

        self.original, self.memory, self.policy = original, memory, policy
        cores = policy.residual_cores
        block = 64 // cores
        self.program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(min(8,cores),(cores+7)//8),
            subblock_w=next(x for x in (4,2,1) if block%x==0), block_h=1, block_w=block, inplace=False,
        )

    def __call__(self, x):
        import ttnn

        x = ttnn.typecast(x, ttnn.float32)
        out = ttnn.rms_norm(x, epsilon=self.original.eps, compute_kernel_config=self.original.compute,
                           program_config=self.program, memory_config=self.memory,
                           weight=self.original.weight if self.policy.norm_fused_weight else None)
        if not self.policy.norm_fused_weight:
            out = ttnn.multiply(out,self.original.weight,memory_config=self.memory)
        return ttnn.typecast(out,ttnn.bfloat16)


class _DecodeNorm:
    def __init__(self, original, policy):
        import ttnn

        self.original, self.policy = original, policy
        cores = policy.decode_norm_cores
        block = 64 // cores
        self.memory = ttnn.create_sharded_memory_config(
            (32, 2048 // cores), core_grid=ttnn.CoreGrid(x=min(8, cores), y=(cores + 7) // 8),
            strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True)
        self.program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(min(8, cores), (cores + 7) // 8),
            subblock_w=next(x for x in (4, 2, 1) if block % x == 0), block_h=1, block_w=block, inplace=False)

    def __call__(self, x, *, mode=None, norm_config=None):
        import ttnn

        if x.shape[-2] != 1:
            return self.original(x, mode=mode, norm_config=norm_config)
        if self.policy.decode_norm_fill:
            x = ttnn.fill_implicit_tile_padding(x, 0.0)
        y = ttnn.to_memory_config(x, self.memory)
        if self.policy.decode_norm_fp32_input:
            y = ttnn.typecast(y, ttnn.float32)
        y = ttnn.rms_norm(y, epsilon=self.original.eps, weight=self.original.weight,
                          compute_kernel_config=self.original.compute, program_config=self.program,
                          memory_config=self.memory)
        if self.policy.decode_norm_fp32_input:
            y = ttnn.typecast(y, ttnn.bfloat16)
        output = ttnn.DRAM_MEMORY_CONFIG if self.policy.decode_norm_output == "dram" else ttnn.L1_MEMORY_CONFIG
        return ttnn.to_memory_config(y, output)


class _ShardedLayer:
    def __init__(self, original, mesh, policy):
        import ttnn

        self.__dict__.update(original.__dict__)
        self.original = original
        self.output_sharded = policy.residual_output_sharded
        cores = policy.residual_cores
        self.residual_memory = ttnn.create_sharded_memory_config(
            (32,2048//cores), core_grid=ttnn.CoreGrid(x=min(8,cores),y=(cores+7)//8),
            strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True)
        self.sharded_attention_norm = _ShardedNorm(self.attention_norm,self.residual_memory,policy)
        self.sharded_ffn_norm = _ShardedNorm(self.ffn_norm,self.residual_memory,policy)

    def forward(self, x, *, mode="decode", **kwargs):
        if mode != "decode":
            return self.original.forward(x,mode=mode,**kwargs)
        import ttnn

        residual = ttnn.to_memory_config(x,self.residual_memory)
        normed = self.sharded_attention_norm(residual)
        if self.is_full_attention:
            attn = self.attention.forward(normed, **{k:v for k,v in kwargs.items() if k in
                ("cos","sin","position_tensor","page_table","chunk_page_table","chunk_start_idx_tensor")})
        else:
            attn = self.attention.forward(normed,mode="recurrent")
        mixed = ttnn.add(residual,ttnn.to_memory_config(attn,self.residual_memory),memory_config=self.residual_memory)
        ff = self.sharded_ffn_norm(mixed)
        # The native router and compact expert reader consume interleaved tensors.
        ff = ttnn.to_memory_config(ff,ttnn.L1_MEMORY_CONFIG)
        ffout = self.feed_forward.forward(ff,mode="decode")
        result = ttnn.add(mixed,ttnn.to_memory_config(ffout,self.residual_memory),memory_config=self.residual_memory)
        return result if self.output_sharded else ttnn.to_memory_config(result,ttnn.L1_MEMORY_CONFIG)


class _OptimizedCompactExperts(_CompactExperts):
    def __call__(self, hidden, routing, mode="decode"):
        if mode != "decode":
            raise ValueError("Compact expert adapter accepts decode only")
        import ttnn

        values, indices = routing
        indices = ttnn.typecast(indices, ttnn.uint32)
        table = ttnn.scatter(self.idx_template, dim=-1, index=indices, src=self.slot_labels)
        table = ttnn.reshape(ttnn.to_layout(table, ttnn.ROW_MAJOR_LAYOUT), (256,))
        inputs = ttnn.pad(ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT), ((0, 0), (0, 0), (0, 255), (0, 0)), 0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.policy.expert_input_dtype == "bfp8":
            inputs = ttnn.to_layout(inputs,ttnn.TILE_LAYOUT,dtype=ttnn.bfloat8_b,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.empty(
            (1, 1, 256, 2048), device=hidden.device(), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            inputs, self.gate, self.up, self.down, self.compact_counts, table,
            input_m_tiles=1, core_grid=ttnn.CoreCoord(self.policy.expert_grid_x, self.policy.expert_grid_y),
            dtype=ttnn.bfloat16, output=out, expert_region_offsets=self.compact_offsets,
            read_x_at_offset=False, compute_kernel_config=self.compute,
        )
        out = ttnn.reshape(out, (8, 1, 32, 2048))[:, :, :1, :]
        return ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            out, ttnn.to_layout(ttnn.typecast(indices, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT),
            self.mapping, reduce_dim=0, split_size=2048, cluster_axis=0,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            scores_tensor=ttnn.to_layout(values, ttnn.ROW_MAJOR_LAYOUT),
        )[0]


class _OptimizedSparseExperts:
    def prepare(self, mesh):
        import torch
        import ttnn

        self.ones = ttnn.from_torch(torch.ones((1,1,1,8), dtype=torch.bfloat16), device=mesh,
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        self.zeros = ttnn.from_torch(torch.zeros((1,1,1,256), dtype=torch.bfloat16), device=mesh,
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        self.placeholder = ttnn.to_layout(self.zeros,ttnn.ROW_MAJOR_LAYOUT)
        dt = {"bfp4":ttnn.bfloat4_b, "bfp8":ttnn.bfloat8_b}
        self.gate_up_weight = _as_dtype(self.weights.gate_up_proj, dt[self.policy.expert_dtype])
        self.down_weight = _as_dtype(self.weights.down_proj, dt[self.policy.sparse_down_dtype])
        if self.policy.sparse_split:
            self.up_weight = self.gate_up_weight[..., :512]
            self.gate_weight = self.gate_up_weight[..., 512:]

    def program_config(self, role, n, activation=None):
        import ttnn

        cores = getattr(self.policy, f"sparse_{role}_cores")
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(min(8,cores),(cores+7)//8),
            in0_block_w=getattr(self.policy,f"sparse_{role}_k"),
            out_subblock_h=1, out_subblock_w=getattr(self.policy,f"sparse_{role}_subblock"),
            out_block_h=1, out_block_w=n//32//cores, per_core_M=1, per_core_N=n//32//cores,
            fuse_batch=False, fused_activation=activation, mcast_in0=True,
        )

    def __call__(self, hidden, routing, mode="decode"):
        import ttnn

        values, indices = routing
        if self.policy.sparse_indexed:
            ids = ttnn.to_layout(indices if indices.dtype==ttnn.uint16 else ttnn.typecast(indices,ttnn.uint16),ttnn.ROW_MAJOR_LAYOUT)
            sparsity, nnz, active = self.placeholder, None, 8
        else:
            ids, nnz, active = None, 8, 256
            mask = ttnn.scatter(self.zeros, dim=-1, index=indices, src=self.ones)
            # Boolean ones establish count8 even if router scores underflow.
            sparsity = ttnn.to_layout(mask, ttnn.ROW_MAJOR_LAYOUT)
        def project(x, weight, role, n, sparse_input=False, activation=None):
            return ttnn.sparse_matmul(x, weight, sparsity=sparsity, nnz=nnz, indices=ids,
                program_config=self.program_config(role,n,activation), compute_kernel_config=self.compute,
                memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                is_input_a_sparse=sparse_input, output_tile=ttnn.Tile([32,32]))
        if self.policy.sparse_split:
            up = ttnn.reshape(project(hidden,self.up_weight,"gate",512), (1,active,1,512))
            gate = ttnn.reshape(project(hidden,self.gate_weight,"gate",512), (1,active,1,512))
            # The sparse factory does not emit SFPU_ACTIVATION from the program config.
            # Fuse SiLU into the binary consumer instead (see sparse_activation_diagnosis.md).
            activated = ttnn.multiply(gate,up,input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                                      memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            width = self.gate_up_weight.shape[-1]
            both = ttnn.reshape(project(hidden,self.gate_up_weight,"gate",width), (1,active,1,width))
            activated = ttnn.swiglu(both, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        down = project(activated,self.down_weight,"down",2048,True)
        if self.policy.sparse_fused_reduce:
            if not self.policy.sparse_indexed:
                raise ValueError("Fused sparse reduction requires compact indexed output")
            return ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                ttnn.reshape(down[:,:,:1,:],(8,1,1,2048)),ids,self.mapping,reduce_dim=0,split_size=2048,
                cluster_axis=getattr(self,"reduce_cluster_axis",0),
                scores_tensor=ttnn.to_layout(values,ttnn.ROW_MAJOR_LAYOUT),
                output_memory_config=ttnn.L1_MEMORY_CONFIG)[0]
        scores = values if self.policy.sparse_indexed else ttnn.scatter(self.zeros, dim=-1, index=indices, src=values)
        weighted = ttnn.multiply(down,ttnn.permute(scores,(0,3,2,1)),memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.experimental.fast_reduce_nc(weighted,dims=[1],memory_config=ttnn.L1_MEMORY_CONFIG)[:,:,:1,:]


class _OptimizedPrefillExperts(_PackedPrefillExperts):
    def sparse_prefill(self, hidden, routing):
        import ttnn

        memory = ttnn.L1_MEMORY_CONFIG if self.policy.prefill_sparse_l1 else ttnn.DRAM_MEMORY_CONFIG
        pieces = []
        length = hidden.shape[2]
        for offset in range(0,length,self.policy.prefill_sparse_chunk):
            stop = min(length,offset+self.policy.prefill_sparse_chunk)
            x = hidden[:,:,offset:stop,:]
            scores = routing[:,:,offset:stop,:]
            mask = ttnn.to_layout(ttnn.max(scores,dim=2,keepdim=True),ttnn.ROW_MAJOR_LAYOUT)
            def pc(role,n):
                cores = getattr(self.policy,f"prefill_{role}_cores")
                return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(min(8,cores),(cores+7)//8),
                    in0_block_w=getattr(self.policy,f"prefill_{role}_k"),
                    out_subblock_h=1,out_subblock_w=getattr(self.policy,f"prefill_{role}_subblock"),
                    out_block_h=1,out_block_w=n//32//cores,
                    per_core_M=(stop-offset)//32,per_core_N=n//32//cores,
                    fuse_batch=False,fused_activation=None,mcast_in0=True)
            both = ttnn.sparse_matmul(x,self.weights.gate_up_proj,sparsity=mask,nnz=None,
                program_config=pc("gate",1024),compute_kernel_config=self.down_compute,
                memory_config=memory,dtype=ttnn.bfloat16)
            both = ttnn.reshape(both,(1,256,stop-offset,1024))
            act = ttnn.swiglu(both,dim=-1,memory_config=memory)
            del both
            down = ttnn.sparse_matmul(act,self.weights.down_proj,sparsity=mask,nnz=None,
                program_config=pc("down",2048),compute_kernel_config=self.down_compute,
                is_input_a_sparse=True,memory_config=memory,dtype=ttnn.bfloat16)
            del act
            weighted = ttnn.multiply(down,ttnn.permute(scores,(0,3,2,1)),memory_config=memory)
            del down
            pieces.append(ttnn.experimental.fast_reduce_nc(weighted,dims=[1]))
            del weighted
        return ttnn.concat(pieces,dim=2) if len(pieces)>1 else pieces[0]

    def __call__(self, hidden, routing, mode="decode"):
        if mode == "decode":
            return self.original(hidden, routing, mode=mode)
        if self.policy.prefill_backend == "dispatch":
            return self.dispatch(hidden,routing)
        if self.policy.prefill_backend == "sparse":
            return self.sparse_prefill(hidden,routing)
        import ttnn

        length, experts = hidden.shape[2], self.config.num_experts
        groups = []
        for weight in self.packed_groups:
            parts = ttnn.experimental.minimal_matmul_split(
                hidden, weight, chunks=32, fuse_swiglu=True, dtype=ttnn.bfloat16,
                config=_minimal_config(self.policy.prefill_minimal_config),
                compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            groups.append(ttnn.concat(parts, dim=1))
        out = ttnn.concat(groups, dim=1)
        cores = self.policy.prefill_down_cores
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(min(8, cores), (cores + 7) // 8),
            in0_block_w=self.policy.prefill_down_k,
            out_subblock_h=1, out_subblock_w=self.policy.prefill_down_subblock,
            out_block_h=1, out_block_w=64 // cores,
            per_core_M=length // 32, per_core_N=64 // cores,
            fuse_batch=False, fused_activation=None, mcast_in0=True,
        )
        down = ttnn.sparse_matmul(
            out, self.weights.down_proj, sparsity=self.prefill_sparsity, nnz=experts,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tile=ttnn.Tile([32, 32]),
            program_config=pc, compute_kernel_config=self.down_compute,
            is_input_a_sparse=True, dtype=ttnn.bfloat16,
        )
        weighted = ttnn.multiply(down, ttnn.permute(routing, (0, 3, 2, 1)))
        return ttnn.experimental.fast_reduce_nc(weighted, dims=[1])


class _ProjectionMixin:
    def project(self, role, x, weight, compute, memory):
        import ttnn

        projection = self.optimized_projections.get(role)
        if projection is None:
            return ttnn.linear(x, weight, compute_kernel_config=compute, memory_config=memory)
        return projection(x, memory)


class _Projection:
    """Phase-specific matmul with setup-only weight packing and inert N padding."""

    def __init__(self, weight, mesh, options):
        import math
        import torch
        import ttnn

        self.options = options
        self.k, self.n = tuple(weight.shape)[-2:]
        self.parts = []
        if options.get("split_widths"):
            offset = 0
            for width, part_options in zip(options["split_widths"],options["split_options"],strict=True):
                self.parts.append(_Projection(weight[...,offset:offset+width],mesh,part_options))
                offset += width
            if offset != self.n:
                raise ValueError("Packed projection splits must preserve all columns")
            return
        dtype = {"bf16":ttnn.bfloat16, "bfp8":ttnn.bfloat8_b, "bfp4":ttnn.bfloat4_b}.get(options.get("dtype"), weight.dtype)
        self.weight = _as_dtype(weight, dtype)
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=getattr(ttnn.MathFidelity, options.get("fidelity", "HiFi2")),
            math_approx_mode=options.get("approx",False), fp32_dest_acc_en=options.get("fp32", True),
            packer_l1_acc=options.get("packer",True),
        )
        self.dram = options.get("dram", False)
        if self.dram:
            readers, cores = options.get("readers", 1), options.get("cores", 8)
            banks = mesh.dram_grid_size().x
            padded_n = math.ceil(self.n / (32 * banks * readers)) * 32 * banks * readers
            value = ttnn.to_torch(ttnn.get_device_tensors(self.weight)[0]).reshape(self.k, self.n)
            value = torch.nn.functional.pad(value, (0, padded_n - self.n))
            weight_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
                              [self.k, padded_n // banks], ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.decode_weight = ttnn.from_torch(value.reshape(1,1,self.k,padded_n), device=mesh,
                dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=weight_memory,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            self.input_memory = ttnn.create_sharded_memory_config(
                (32, self.k // cores), core_grid=ttnn.CoreGrid(x=min(8,cores), y=(cores+7)//8),
                strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=options.get("k_block", self.k // cores // 32), per_core_M=1,
                per_core_N=padded_n // banks // readers // 32, num_workers_per_dram_bank=readers,
            )

    def __call__(self, x, memory):
        import math
        import ttnn

        length = x.shape[-2]
        if self.parts:
            return ttnn.concat([part(x,memory) for part in self.parts],dim=-1,memory_config=memory)
        if length == 1 and self.dram:
            source = ttnn.to_memory_config(ttnn.reshape(x, (1,1,1,self.k)), self.input_memory)
            y = ttnn.linear(source, self.decode_weight, program_config=self.program,
                compute_kernel_config=self.compute, dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
            y = ttnn.to_memory_config(y, memory)[..., :self.n]
            return ttnn.reshape(y, (*tuple(x.shape)[:-1], self.n))
        pc = None
        if length == 1 and self.options.get("decode_grid"):
            gx, gy = self.options["decode_grid"]
            pn = self.options.get("per_n",math.ceil(self.n/32/(gx*gy)))
            pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(gx,gy),in0_block_w=self.options.get("k_block",8),
                per_core_M=1,per_core_N=pn,out_subblock_h=1,
                out_subblock_w=self.options.get("subblock",next(n for n in (4,3,2,1) if pn%n==0)),
                out_block_h=1,out_block_w=pn,fuse_batch=True,fused_activation=None,mcast_in0=True)
        if self.options.get("input_l1"):
            x = ttnn.to_memory_config(x,ttnn.L1_MEMORY_CONFIG)
        if length > 1 and self.options.get("prefill_minimal"):
            return ttnn.experimental.minimal_matmul(x, self.weight, config=_minimal_config(self.options["prefill_minimal"]),
                compute_kernel_config=self.compute, memory_config=memory, dtype=ttnn.bfloat16)
        if length > 1 and self.options.get("prefill_grid"):
            gx, gy = self.options["prefill_grid"]
            pn = math.ceil(self.n / 32 / gx)
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx,gy), in0_block_w=self.options.get("prefill_k",8),
                per_core_M=math.ceil(length/32/gy), per_core_N=pn,
                out_subblock_h=1, out_subblock_w=next(n for n in (4,2,1) if pn%n==0),
                transpose_mcast=False, fused_activation=None, fuse_batch=False,
            )
        return ttnn.linear(x, self.weight, memory_config=memory, program_config=pc,
                           compute_kernel_config=self.compute, dtype=ttnn.bfloat16)


from .fused_decoder import _FusedAttention, _FusedGDN


class _OptimizedAttention(_ProjectionMixin, _FusedAttention):
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
        residual=None,
    ):
        import ttnn
        from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
            _get_flexible_sdpa_program_config,
            _get_paged_sdpa_decode_program_config,
            _get_sdpa_compute_kernel_config,
            apply_rotary_pos_emb_ttnn,
        )

        batch, length = x.shape[0], x.shape[1]
        assert batch == 1 and self.use_paged_attention
        h, kv, d = self.config.num_heads, self.config.num_kv_heads, self.config.head_dim
        decode = length == 1
        ckc = self.compute_kernel_config_decode if decode else self.compute_kernel_config
        mc = (ttnn.L1_MEMORY_CONFIG if decode and self.optimization.attention_decode_memory == "l1"
              else ttnn.DRAM_MEMORY_CONFIG)
        heads_in_decode = decode and "decode_heads" in self.fusions
        if {"qkv", "heads", "decode_heads"}.intersection(self.fusions):
            packed = self.project("attention_in", x, self.qkvg_weight, ckc, mc)
            qw, kw = h * d, kv * d
            gate = packed[..., qw + 2 * kw :]
            if heads_in_decode:
                query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
                    ttnn.reshape(packed[..., : qw + 2 * kw], (1, 1, 1, qw + 2 * kw)),
                    num_heads=h,
                    num_kv_heads=kv,
                    memory_config=self.cache_mem,
                )
                query, key, value = (ttnn.to_memory_config(t, mc) for t in (query, key, value))
            elif "heads" in self.fusions:
                query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
                    packed[..., : qw + 2 * kw],
                    num_heads=h,
                    num_kv_heads=kv,
                    transpose_key=False,
                )
            else:
                query = ttnn.transpose(ttnn.reshape(packed[..., :qw], (1, length, h, d)), 1, 2)
                key = ttnn.transpose(ttnn.reshape(packed[..., qw : qw + kw], (1, length, kv, d)), 1, 2)
                value = ttnn.transpose(ttnn.reshape(packed[..., qw + kw : qw + 2 * kw], (1, length, kv, d)), 1, 2)
            query = ttnn.rms_norm(query, weight=self.weights.q_norm, epsilon=self.config.norm_eps)
            key = ttnn.rms_norm(key, weight=self.weights.k_norm, epsilon=self.config.norm_eps)
        else:
            qg = ttnn.linear(x, self.weights.q_proj, compute_kernel_config=ckc, memory_config=mc)
            query, gate = ttnn.chunk(ttnn.reshape(qg, (1, length, h, 2 * d)), 2, dim=-1)
            gate = ttnn.reshape(gate, (1, length, h * d))
            query = ttnn.transpose(ttnn.rms_norm(query, weight=self.weights.q_norm, epsilon=self.config.norm_eps), 1, 2)
            key = ttnn.linear(x, self.weights.k_proj, compute_kernel_config=ckc, memory_config=mc)
            key = ttnn.reshape(key, (1, length, kv, d))
            key = ttnn.transpose(ttnn.rms_norm(key, weight=self.weights.k_norm, epsilon=self.config.norm_eps), 1, 2)
            value = ttnn.linear(x, self.weights.v_proj, compute_kernel_config=ckc, memory_config=mc)
            value = ttnn.transpose(ttnn.reshape(value, (1, length, kv, d)), 1, 2)
        cos = ttnn.reshape(cos, (1, 1, length, cos.shape[-1]))
        sin = ttnn.reshape(sin, (1, 1, length, sin.shape[-1]))
        if "rope" in self.fusions:
            if decode and not heads_in_decode and "indexed_rope" not in self.fusions:
                query = ttnn.transpose(query, 1, 2)
                key = ttnn.transpose(key, 1, 2)
            query, key = self._rope(query, cos, sin, decode), self._rope(key, cos, sin, decode)
        else:
            query, key = apply_rotary_pos_emb_ttnn(query, key, cos, sin)
            if decode and not heads_in_decode:
                query = ttnn.transpose(query, 1, 2)
                key = ttnn.transpose(key, 1, 2)
        if decode:
            key = ttnn.to_memory_config(key, self.cache_mem)
            value = ttnn.to_memory_config(
                value if heads_in_decode else ttnn.transpose(value, 1, 2),
                self.value_cache_mem if "cache" in self.fusions else self.cache_mem,
            )
            if "cache" in self.fusions:
                ttnn.experimental.paged_fused_update_cache(
                    self.paged_kv_cache_key,
                    key,
                    self.paged_kv_cache_value,
                    value,
                    update_idxs_tensor=position_tensor,
                    page_table=page_table,
                )
            else:
                for cache, tensor in ((self.paged_kv_cache_key, key), (self.paged_kv_cache_value, value)):
                    ttnn.experimental.paged_update_cache(
                        cache, tensor, update_idxs_tensor=position_tensor, page_table=page_table
                    )
            pc = (
                self._tile_decode_program_config
                if (page_table.shape[1] * self.paged_kv_cache_key.shape[2]) % 64
                else _get_paged_sdpa_decode_program_config(
                    self.device, self.paged_kv_cache_key.shape[0] * self.paged_kv_cache_key.shape[2]
                )
            )
            sdpa_options = dict(self.sdpa_policy)
            prefill_options = sdpa_options.pop("prefill",None)
            fidelity = sdpa_options.pop("fidelity",None)
            if sdpa_options:
                grid = sdpa_options.pop("grid",[8,8])
                chunk = sdpa_options.get("k_chunk_size",64)
                coverage = page_table.shape[1]*self.paged_kv_cache_key.shape[2]
                while coverage%chunk:
                    chunk //= 2
                sdpa_options["k_chunk_size"] = chunk
                pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(*grid),**sdpa_options)
            sdpa_compute = ttnn.init_device_compute_kernel_config(self.device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity,fidelity),fp32_dest_acc_en=True,
                packer_l1_acc=False,math_approx_mode=False) if fidelity else None
            output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG),
                self.paged_kv_cache_key,
                self.paged_kv_cache_value,
                cur_pos_tensor=position_tensor,
                page_table_tensor=page_table,
                is_causal=True,
                scale=d**-0.5,
                program_config=pc,
                compute_kernel_config=sdpa_compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if "decode_concat" not in self.fusions:
                output = ttnn.transpose(output, 1, 2)
        else:
            ttnn.experimental.paged_fill_cache(self.paged_kv_cache_key, ttnn.typecast(key, self.paged_kv_cache_key.dtype), chunk_page_table, batch_idx=0)
            ttnn.experimental.paged_fill_cache(self.paged_kv_cache_value, ttnn.typecast(value, self.paged_kv_cache_value.dtype), chunk_page_table, batch_idx=0)
            output = ttnn.transformer.chunked_scaled_dot_product_attention(
                query,
                self.paged_kv_cache_key,
                self.paged_kv_cache_value,
                page_table,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                scale=d**-0.5,
                program_config=(ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                    q_chunk_size=32, k_chunk_size=32, exp_approx_mode=False) if length < 128
                    else ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                    **self.sdpa_policy["prefill"]) if "prefill" in self.sdpa_policy
                    else _get_flexible_sdpa_program_config(self.device)),
                compute_kernel_config=_get_sdpa_compute_kernel_config(),
            )
        if decode and "decode_concat" in self.fusions:
            output = ttnn.experimental.nlp_concat_heads_decode(
                ttnn.to_memory_config(output, self.cache_mem), num_heads=h
            )
            output = ttnn.to_memory_config(output, mc)[:, :, :1, :]
            output = ttnn.reshape(output, (1, 1, h * d))
        else:
            output = ttnn.transformer.concatenate_heads(output)
        if "gate" in self.fusions:
            output = ttnn.multiply(gate, output, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        else:
            output = ttnn.multiply(output, ttnn.sigmoid(gate))
        if residual is not None:
            return self.residual_projection(output, residual, ckc, mc)
        return self.project("attention_out", output, self.weights.o_proj, ckc, mc)


class _OptimizedGDN(_ProjectionMixin, _FusedGDN):
    def _convolve(self, qkv):
        if "kda_conv" not in self.fusions or (self.optimization.conv_chunk == 512 and not self.optimization.conv_state_rm):
            return super()._convolve(qkv)
        import ttnn

        length = qkv.shape[1]
        rm_state = self.optimization.conv_state_rm
        history = self.fused_conv_state if rm_state else ttnn.to_layout(self.fused_conv_state, ttnn.ROW_MAJOR_LAYOUT)
        if length == 1 and rm_state:
            qkv = ttnn.fill_implicit_tile_padding(qkv, 0.0)
            current = ttnn.to_layout(ttnn.reshape(qkv, (1, 32, qkv.shape[-1]), (1, 32, qkv.shape[-1])),
                                     ttnn.ROW_MAJOR_LAYOUT)
        else:
            current = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)
            if length == 1:
                current = ttnn.pad(current, ((0, 0), (0, 31), (0, 0)), 0)
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            current, history, *self.kda_taps, self.cfg.q_dim, self.cfg.k_dim, self.cfg.v_dim,
            actual_start=self.kda_start, predecessor_carry=history,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=self.optimization.conv_chunk),
            compute_kernel_config=self.compute_kernel_config, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if length > 1 and getattr(self, "_tail_rows", None) is not None:
            table = ttnn.reshape(ttnn.concat((history, current[:, :length, :]), dim=1), (length + 3, qkv.shape[-1]))
            tail = ttnn.reshape(ttnn.embedding(ttnn.reshape(self._tail_rows, (1, 3)), table,
                                               layout=ttnn.ROW_MAJOR_LAYOUT), (1, 3, qkv.shape[-1]))
        elif length >= 3 and "kda_tail" in self.fusions:
            tail = current[:, length - 3 : length, :]
        elif rm_state and length == 1:
            tail = ttnn.concat((history[:, 1:3, :], current[:, :1, :]), dim=1)
        else:
            tail = ttnn.concat((history, current[:, :length, :]), dim=1)[:, length : length + 3, :]
        ttnn.copy(tail if rm_state else ttnn.to_layout(tail, ttnn.TILE_LAYOUT), self.fused_conv_state)
        parts = (q[:, :length, :], k[:, :length, :], v[:, :length, :])
        return parts if "kda_tuple" in self.fusions else ttnn.concat(parts, dim=-1)

    def _recurrent_step_fp32(self, q, k, v, decay_log, beta):
        if not self.optimization.gdn_l1:
            return super()._recurrent_step_fp32(q,k,v,decay_log,beta)
        import ttnn

        memory = ttnn.L1_MEMORY_CONFIG
        heads,groups = self.num_heads,self.num_v_heads//self.num_heads
        def column(x):
            x = ttnn.typecast(x,ttnn.float32,memory_config=memory)
            x = ttnn.permute(x,(0,2,3,1),memory_config=memory)
            return ttnn.reshape(x,(heads,1,self.head_k_dim,1))
        def scalar(x):
            return ttnn.reshape(ttnn.typecast(x,ttnn.float32,memory_config=memory),(heads,groups,1,1))
        def reduce(x):
            return ttnn.sum(x,dim=-2,keepdim=True,compute_kernel_config=self.compute_kernel_config,
                            fast_and_approximate_mode=False,memory_config=memory)
        q = ttnn.multiply(column(q),self.head_k_dim**-0.5,memory_config=memory)
        k = column(k)
        v = ttnn.reshape(ttnn.typecast(v,ttnn.float32,memory_config=memory),(heads,groups,1,self.head_v_dim))
        decay = scalar(ttnn.exp(decay_log,fast_and_approximate_mode=False,memory_config=memory))
        beta = scalar(beta)
        state = ttnn.reshape(self.recurrent_state,(heads,groups,self.head_k_dim,self.head_v_dim))
        decayed = ttnn.multiply(state,decay,memory_config=memory)
        stored = reduce(ttnn.multiply(decayed,k,memory_config=memory))
        delta = ttnn.multiply(ttnn.subtract(v,stored,memory_config=memory),beta,memory_config=memory)
        updated = ttnn.addcmul(decayed,k,delta,output_tensor=state)
        output = reduce(ttnn.multiply(updated,q,memory_config=memory))
        return (ttnn.reshape(output,(1,1,self.num_v_heads,self.head_v_dim)),
                ttnn.reshape(updated,(1,self.num_v_heads,self.head_k_dim,self.head_v_dim)))

    def _decode_heads(self, mega, q, k, v, decay_log, beta, z_start, mc, residual):
        import ttnn

        if not hasattr(self, "heads_mem"):
            self.heads_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                               [32, self.head_v_dim], ttnn.ShardOrientation.ROW_MAJOR))
        heads, groups, dk = self.num_heads, self.num_v_heads // self.num_heads, self.head_k_dim
        z = mega[:, :, z_start : z_start + self.cfg.v_dim]
        packed = ttnn.concat((q, k, v, z), dim=-1, memory_config=mc)
        qk, vh, zh = ttnn.experimental.nlp_create_qkv_heads_decode(
            ttnn.reshape(packed, (1, 1, 1, packed.shape[-1])), num_heads=2 * heads, num_kv_heads=self.num_v_heads,
            memory_config=self.heads_mem)
        qk, vh, zh = (ttnn.to_memory_config(t, mc) for t in (qk, vh, zh))
        qk = ttnn.multiply(ttnn.rms_norm(qk, epsilon=1e-6 / dk, compute_kernel_config=self.compute_kernel_config),
                           dk**-0.5, memory_config=mc)
        cols = ttnn.permute(ttnn.typecast(qk, ttnn.float32, memory_config=mc), (0, 2, 3, 1), memory_config=mc)
        cols = ttnn.reshape(cols, (2 * heads, 1, dk, 1))
        qc = ttnn.multiply(cols[:heads], dk**-0.5, memory_config=mc)
        kc = cols[heads:]
        values = ttnn.reshape(ttnn.typecast(vh, ttnn.float32, memory_config=mc), (heads, groups, 1, self.head_v_dim))
        decay = ttnn.reshape(ttnn.typecast(ttnn.exp(decay_log, fast_and_approximate_mode=False, memory_config=mc),
                                           ttnn.float32, memory_config=mc), (heads, groups, 1, 1))
        beta = ttnn.reshape(ttnn.typecast(beta, ttnn.float32, memory_config=mc), (heads, groups, 1, 1))

        def reduce(t):
            return ttnn.sum(t, dim=-2, keepdim=True, compute_kernel_config=self.compute_kernel_config,
                            fast_and_approximate_mode=False, memory_config=mc)

        state = ttnn.reshape(self.recurrent_state, (heads, groups, dk, self.head_v_dim))
        decayed = ttnn.multiply(state, decay, memory_config=mc)
        stored = reduce(ttnn.multiply(decayed, kc, memory_config=mc))
        delta = ttnn.multiply(ttnn.subtract(values, stored, memory_config=mc), beta, memory_config=mc)
        updated = ttnn.addcmul(decayed, kc, delta, output_tensor=state)
        if "gdn_inplace" not in self.fusions:
            ttnn.copy(ttnn.reshape(updated, tuple(self.recurrent_state.shape)), self.recurrent_state)
        output = reduce(ttnn.multiply(updated, qc, memory_config=mc))
        output = ttnn.typecast(ttnn.reshape(output, (1, 1, self.num_v_heads, self.head_v_dim)), ttnn.bfloat16)
        normalized = ttnn.rms_norm(ttnn.typecast(output, ttnn.float32), epsilon=self.norm_eps, memory_config=mc,
                                   compute_kernel_config=self.compute_kernel_config)
        normalized = ttnn.multiply(ttnn.typecast(normalized, ttnn.bfloat16), self.norm_weight, memory_config=mc)
        gated = ttnn.multiply(ttnn.typecast(zh, ttnn.float32), ttnn.typecast(normalized, ttnn.float32),
                              input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mc)
        gated = ttnn.to_memory_config(ttnn.typecast(gated, ttnn.bfloat16), self.heads_mem)
        flat = ttnn.experimental.nlp_concat_heads_decode(gated, num_heads=self.num_v_heads)
        flat = ttnn.reshape(ttnn.to_memory_config(flat, mc)[:, :, :1, :], (1, 1, self.cfg.v_dim))
        if residual is not None:
            return self.residual_projection(flat, residual, self.compute_kernel_config, mc)
        return self.project("attention_out", flat, self.output_weight, self.compute_kernel_config, mc)

    def forward(self, x, mode="recurrent", chunk_size=None, valid_len=None, residual=None):
        """Consume one real token or an aligned real-token chunk, retaining state."""
        import ttnn

        if len(x.shape) != 3 or x.shape[0] != 1 or x.shape[2] != self.hidden_size:
            raise ValueError("GDN expects a selected request [1,T,2048]")
        length = x.shape[1]
        if mode not in ("recurrent", "chunk") or length < 1:
            raise ValueError("GDN mode must be recurrent or chunk, with at least one token")
        masked = valid_len is not None and valid_len != length and getattr(self, "_tail_mask", None) is not None
        if valid_len is not None and valid_len != length and not masked:
            raise ValueError("FunctionalDecoder must remove padding before the GDN state update")
        if length > 1 and (mode != "chunk" or length % 32):
            raise ValueError("The outer decoder must plan aligned chunks and single-token tails")
        mc = ttnn.L1_MEMORY_CONFIG if length==1 and self.optimization.gdn_l1 else ttnn.DRAM_MEMORY_CONFIG
        if self.optimization.gdn_ab_split:
            mega = self.split_projection(x, mc)
        else:
            mega = self.project("attention_in", x, self.mega_weight, self.compute_kernel_config, mc)
        qkv_width = self.cfg.q_dim + self.cfg.k_dim + self.cfg.v_dim
        stride = getattr(self, "ab_stride", self.num_v_heads)
        a_start, b_start, z_start = qkv_width, qkv_width + stride, qkv_width + 2 * stride

        def columns(tensor, start, end):
            return ttnn.to_layout(tensor[:, :, start:end], ttnn.TILE_LAYOUT, memory_config=mc)

        qkv = self._convolve(columns(mega, 0, qkv_width))
        if isinstance(qkv, tuple):
            q, k, v = qkv
        else:
            q = columns(qkv, 0, self.cfg.q_dim)
            k = columns(qkv, self.cfg.q_dim, self.cfg.q_dim + self.cfg.k_dim)
            v = columns(qkv, self.cfg.q_dim + self.cfg.k_dim, qkv_width)
        fused_norm = "kda_norm_decode" in self.fusions or (length > 1 and "kda_norm" in self.fusions)
        a = ttnn.typecast(columns(mega, a_start, a_start + self.num_v_heads), ttnn.float32)
        beta = ttnn.typecast(
            ttnn.sigmoid(ttnn.typecast(columns(mega, b_start, b_start + self.num_v_heads), ttnn.float32),
                         memory_config=mc), ttnn.bfloat16
        )
        decay_log = ttnn.multiply(
            self.a_neg,
            (
                ttnn.add(a, self.dt_bias, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])
                if "gdn_eltwise" in self.fusions
                else ttnn.softplus(ttnn.add(a, self.dt_bias, memory_config=mc), memory_config=mc)
            ),
            memory_config=mc,
        )
        if masked:
            decay_log = ttnn.multiply(decay_log, ttnn.typecast(self._tail_mask, decay_log.dtype))
            beta = ttnn.multiply(beta, ttnn.typecast(self._tail_mask, beta.dtype))
        if length == 1 and self.optimization.gdn_heads_decode and isinstance(qkv, tuple):
            return self._decode_heads(mega, q, k, v, decay_log, beta, z_start, mc, residual)
        if length > 1:
            eye, tril, ones, masks = self.chunk_constants
            # Explicit head-split inputs bypass the native flat FP32 normalizer.
            # The exact chunk op expands the 16 key heads to 32 value heads.
            if not self.optimization.gdn_flat_qk:
                q = self._normalize_key(self._heads(q, self.num_heads, self.head_k_dim))
                k = self._normalize_key(self._heads(k, self.num_heads, self.head_k_dim))
            if "gdn_flat_v" not in self.fusions:
                v = self._heads(v, self.num_v_heads, self.head_v_dim)
            output, state = ttnn.transformer.chunk_gated_delta_rule(
                q,
                k,
                v,
                decay_log,
                beta,
                initial_state=self.recurrent_state,
                output_final_state=True,
                chunk_size=32,
                output_head_major="gdn_head_major" in self.fusions,
                program_config=self.chunk_program_config,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=mc,
                eye=eye,
                tril=tril,
                ones=ones,
                masks=masks,
            )
            if "gdn_head_major" in self.fusions and not fused_norm:
                output = ttnn.reshape(ttnn.permute(output, (1, 0, 2)), (1, length, self.num_v_heads, self.head_v_dim))
            else:
                output = ttnn.to_layout(output, ttnn.TILE_LAYOUT, memory_config=mc)
        else:
            q = self._normalize_key(self._heads(q, self.num_heads, self.head_k_dim))
            k = self._normalize_key(self._heads(k, self.num_heads, self.head_k_dim))
            v = self._heads(v, self.num_v_heads, self.head_v_dim)
            output, state = self._recurrent_step_fp32(q, k, v, decay_log, beta)
        if length > 1 or "gdn_inplace" not in self.fusions:
            ttnn.copy(state, self.recurrent_state)
        # HF recurrence returns its activation dtype before the gated RMSNorm.
        output = ttnn.typecast(output, ttnn.bfloat16)
        if fused_norm:
            gate_flat = columns(mega, z_start, z_start + self.cfg.v_dim)
            if length == 1:
                output = ttnn.reshape(output, (self.num_v_heads, 1, self.head_v_dim))
                output = ttnn.pad(output, ((0, 0), (0, 31), (0, 0)), 0)
                gate_padded = ttnn.pad(gate_flat, ((0, 0), (0, 31), (0, 0)), 0)
            else:
                if "gdn_head_major" not in self.fusions:
                    output = ttnn.reshape(
                        ttnn.permute(output, (0, 2, 1, 3)), (self.num_v_heads, length, self.head_v_dim)
                    )
                gate_padded = gate_flat
            gated = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                output,
                gate_padded,
                self.kda_norm_weight,
                self.num_v_heads,
                epsilon=self.norm_eps,
                compute_kernel_config=self.kda_norm_compute,
                memory_config=mc,
                output_dtype=ttnn.float32,
            )
            gated = ttnn.multiply(gated[:, :length, :], ttnn.typecast(gate_flat, ttnn.float32))
            gated = ttnn.typecast(gated, ttnn.bfloat16)
            if residual is not None:
                return self.residual_projection(gated, residual, self.compute_kernel_config, mc)
            return self.project("attention_out", gated, self.output_weight, self.compute_kernel_config, mc)
        normalized = ttnn.rms_norm(
            ttnn.typecast(output, ttnn.float32),
            epsilon=self.norm_eps,
            memory_config=mc,
            compute_kernel_config=self.compute_kernel_config,
        )
        normalized = ttnn.multiply(ttnn.typecast(normalized, ttnn.bfloat16), self.norm_weight, memory_config=mc)
        gate = self._heads(columns(mega, z_start, z_start + self.cfg.v_dim), self.num_v_heads, self.head_v_dim)
        if "gdn_eltwise" in self.fusions:
            gated = ttnn.multiply(
                ttnn.typecast(gate, ttnn.float32),
                ttnn.typecast(normalized, ttnn.float32),
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=mc,
            )
        else:
            gated = ttnn.multiply(
                ttnn.typecast(normalized, ttnn.float32),
                ttnn.silu(ttnn.typecast(gate, ttnn.float32), memory_config=mc),
                memory_config=mc,
            )
        gated = ttnn.typecast(gated, ttnn.bfloat16)
        gated = ttnn.reshape(gated, (1, length, self.cfg.v_dim))
        if residual is not None:
            return self.residual_projection(gated, residual, self.compute_kernel_config, mc)
        return self.project("attention_out", gated, self.output_weight, self.compute_kernel_config, mc)


class _GdnSplitProjection:
    def __init__(self, attention, mesh, options):
        import ttnn

        self.qkv = attention.cfg.q_dim + attention.cfg.k_dim + attention.cfg.v_dim
        self.gates = 2 * attention.num_v_heads
        weight = attention.mega_weight
        main = ttnn.concat((weight[..., : self.qkv], weight[..., self.qkv + self.gates :]), dim=-1)
        self.main = _Projection(main, mesh, options.get("main", {"dtype": "bfp8", "fidelity": "LoFi"}))
        self.gate = _Projection(weight[..., self.qkv : self.qkv + self.gates], mesh,
                                options.get("gates", {"dtype": "bf16", "fidelity": "HiFi4"}))

    def __call__(self, x, memory):
        return _SplitColumns(self.main(x, memory), self.gate(x, memory), self.qkv, self.gates)


class _SplitColumns:
    def __init__(self, main, gates, qkv, width):
        self.main, self.gates, self.qkv, self.width = main, gates, qkv, width

    def __getitem__(self, key):
        start, stop = key[-1].start, key[-1].stop
        if start >= self.qkv + self.width:
            return self.main[:, :, start - self.width : stop - self.width]
        if start >= self.qkv:
            if stop > self.qkv + self.width:
                raise ValueError("Split GDN projection slices cannot cross the gate columns")
            return self.gates[:, :, start - self.qkv : stop - self.qkv]
        if stop > self.qkv:
            raise ValueError("Split GDN projection slices cannot cross the gate columns")
        return self.main[:, :, start:stop]


_PLAN_COST = {"full": (0.8, 0.0014, 0.8, 3.0, 1.05), "linear": (0.5, 0.003, 1.1, 3.0, 1.6)}


def _moe_pieces(rows, sizes, limit):
    single = next(n for n in sizes if n >= rows)
    pieces, remaining = [], rows
    while remaining > 0:
        if len(pieces) == limit - 1 or remaining < sizes[0]:
            pieces.append(next(n for n in sizes if n >= remaining))
            break
        piece = max(n for n in sizes if n <= remaining)
        pieces.append(piece)
        remaining -= piece
    if sum(pieces) >= single or (sum(pieces) > rows and single < sizes[-1]):
        return [single]
    return pieces


def _bucket_sizes(policy):
    return tuple(sorted({policy.prefill_chunk, policy.prefill_tail_bucket, *policy.prefill_buckets} - {0}))


class _ZeroPaddedDecodeLayer:
    def __init__(self, original):
        object.__setattr__(self, "original", original)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "original"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "original"), name, value)

    def forward(self, x, *args, mode="decode", **kwargs):
        import ttnn

        if mode == "decode":
            x = ttnn.fill_implicit_tile_padding(x, 0.0)
        return object.__getattribute__(self, "original").forward(x, *args, mode=mode, **kwargs)


def _as_dtype(tensor, dtype):
    import ttnn

    return tensor if tensor.dtype == dtype else ttnn.typecast(tensor, dtype)


def _minimal_config(options):
    import ttnn

    if not options:
        return None
    values = dict(options)
    grid = values.pop("grid", [8,8])
    return ttnn.MinimalMatmulConfig(compute_with_storage_grid_size=ttnn.CoreCoord(*grid),**values)


def _memory_bytes(mesh):
    import ttnn

    result = {}
    for name, kind in (("dram",ttnn.BufferType.DRAM),("l1",ttnn.BufferType.L1)):
        view = ttnn.get_memory_view(mesh,kind)
        result[name] = dict(allocated_bytes=view.total_bytes_allocated_per_bank*view.num_banks,
                           free_bytes=view.total_bytes_free_per_bank*view.num_banks,
                           banks=view.num_banks,
                           largest_free_bytes_per_bank=view.largest_contiguous_bytes_free_per_bank)
    return result


class _OptimizedRouter:
    def __call__(self,x):
        import ttnn

        values,indices = self.module.forward(x)
        values = ttnn.permute(values,(1,2,0,3))
        if x.shape[2]==1 or self.tuple_prefill:
            return values,ttnn.permute(indices,(1,2,0,3))
        indices = ttnn.permute(ttnn.typecast(indices,ttnn.uint32),(1,2,0,3))
        return ttnn.scatter(ttnn.zeros_like(self.zero_templates[x.shape[2]]),dim=-1,index=indices,src=values)


_PREFIX_MATRICES = {}


class _LocalDispatch:
    """Sort/gather local routes without entering the multi-device fabric APIs."""

    def __init__(self, original, mesh, policy):
        import torch
        import ttnn

        self.policy = policy
        dtype = {"bfp8":ttnn.bfloat8_b,"bfp4":ttnn.bfloat4_b}[policy.dispatch_dtype]
        for role in ("gate","up","down"):
            weights = [ttnn.typecast(t,dtype) for t in getattr(original,role)]
            if policy.dispatch_weight_nd:
                n_tiles = weights[0].shape[-1] // 32
                grid_x = ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID.x
                dram = mesh.dram_grid_size()
                memory = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=ttnn.NdShardSpec(
                    shard_shape=ttnn.Shape([32, -(-n_tiles // grid_x) * 32]),
                    grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                           ttnn.CoreCoord(dram.x - 1, dram.y - 1))]),
                    orientation=ttnn.ShardOrientation.ROW_MAJOR))
                weights = [ttnn.to_memory_config(t, memory) for t in weights]
            setattr(self,role,weights)
        self.expert_ids = original.expert_ids
        if policy.dispatch_ffn != "fused":
            self.expert_table = ttnn.reshape(ttnn.typecast(original.expert_ids, ttnn.uint32), (256,))
        self.compute = ttnn.init_device_compute_kernel_config(mesh.arch(),math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,packer_l1_acc=False,math_approx_mode=False)
        self.prepare_maps(mesh)

    def prepare_maps(self,mesh):
        import torch
        import ttnn

        policy = self.policy
        self.prefix_compute = ttnn.init_device_compute_kernel_config(mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,fp32_dest_acc_en=True,packer_l1_acc=False,math_approx_mode=False)
        def put(x,dtype,layout):
            return ttnn.from_torch(x,device=mesh,dtype=dtype,layout=layout,
                                  mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        self.mask = put(torch.zeros((1,256),dtype=torch.int32),ttnn.int32,ttnn.ROW_MAJOR_LAYOUT)
        self.region_compute = ttnn.init_device_compute_kernel_config(mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=False,math_approx_mode=False)
        self.region_matrix = put(torch.ones((256,256)).triu(1).reshape(1,1,256,256),ttnn.bfloat16,ttnn.TILE_LAYOUT)
        self.buffers = {}
        self.rank_buffers = {}
        for length in sorted(set(_bucket_sizes(policy)) | {128, 512}):
            routes = length*8
            capacity = ((routes+256*31+31)//32)*32
            if capacity>=65536:
                raise ValueError("Internal local-dispatch chunk must fit UINT16 route maps")
            self.buffers[length] = (
                put(torch.arange(routes,dtype=torch.float32).reshape(1,1,1,routes),ttnn.float32,ttnn.TILE_LAYOUT),
                put(torch.zeros((1,1,1,capacity),dtype=torch.int32),ttnn.uint32,ttnn.ROW_MAJOR_LAYOUT),
                put(torch.zeros((1,1,1,routes),dtype=torch.int32),ttnn.uint32,ttnn.ROW_MAJOR_LAYOUT),capacity)
            if policy.dispatch_rank == "matmul":
                key = (mesh, length)
                if key not in _PREFIX_MATRICES:
                    _PREFIX_MATRICES[key] = put(torch.ones((length,length)).tril(-1).reshape(1,1,length,length),
                                                ttnn.bfloat8_b,ttnn.TILE_LAYOUT)
                self.rank_buffers[length] = (
                    put(torch.zeros((1,1,length,256),dtype=torch.bfloat16),ttnn.bfloat16,ttnn.TILE_LAYOUT),
                    put(torch.ones((1,1,length,8),dtype=torch.bfloat16),ttnn.bfloat16,ttnn.TILE_LAYOUT),
                    put((torch.arange(routes,dtype=torch.int32) % length if self.tiles(length)
                         else torch.arange(routes,dtype=torch.int32)//8).reshape(1,1,1,routes),ttnn.uint32,
                        ttnn.ROW_MAJOR_LAYOUT), _PREFIX_MATRICES[key],
                    put(torch.ones((1,1,1,length)),ttnn.bfloat16,ttnn.TILE_LAYOUT))
            if policy.dispatch_rank == "cumsum":
                self.rank_buffers[length] = (
                    put(torch.zeros((1,1,length,256),dtype=torch.bfloat16),ttnn.bfloat16,ttnn.TILE_LAYOUT),
                    put(torch.ones((1,1,length,8),dtype=torch.bfloat16),ttnn.bfloat16,ttnn.TILE_LAYOUT),
                    put((torch.arange(routes,dtype=torch.int32)//8).reshape(1,1,1,routes),ttnn.uint32,
                        ttnn.ROW_MAJOR_LAYOUT))

    def tiles(self, length):
        return self.policy.dispatch_combine == "tiles" or 0 < self.policy.dispatch_tiles_from <= length

    def restore_table(self, result, capacity):
        import ttnn

        return ttnn.reshape(ttnn.to_layout(result,ttnn.ROW_MAJOR_LAYOUT),(capacity,2048))

    def route_maps(self,routing):
        import ttnn

        values,ids = routing
        length = ids.shape[2]
        positions,map_zeros,inverse_zeros,capacity = self.buffers[length]
        def tiled_float(x):
            if x.layout==ttnn.ROW_MAJOR_LAYOUT and x.dtype==ttnn.uint32:
                x = ttnn.typecast(x,ttnn.uint16)
            return ttnn.typecast(ttnn.to_layout(x,ttnn.TILE_LAYOUT),ttnn.float32)
        def rm_uint32(x):
            # All route indices/capacities fit16 bits. Avoid long-row UINT32 untilize.
            return ttnn.typecast(ttnn.to_layout(ttnn.typecast(x,ttnn.uint16),ttnn.ROW_MAJOR_LAYOUT),ttnn.uint32)
        if self.policy.dispatch_rank == "matmul":
            zeros, ones, tokens, prefix, count_row = self.rank_buffers[length]
            index = ttnn.typecast(ttnn.to_layout(ttnn.reshape(ids,(1,1,length,8)),ttnn.TILE_LAYOUT),ttnn.uint32)
            chosen = ttnn.scatter(zeros,dim=-1,index=index,src=ones)
            count = ttnn.matmul(count_row,chosen,dtype=ttnn.float32,compute_kernel_config=self.prefix_compute)
            counts = ttnn.reshape(rm_uint32(count),(256,))
            aligned = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(count,31.0),0.03125)),32.0)
            tiles = ttnn.typecast(ttnn.multiply(aligned,0.03125),ttnn.bfloat16)
            region_start = ttnn.multiply(ttnn.matmul(tiles,self.region_matrix,dtype=ttnn.float32,
                                                     compute_kernel_config=self.region_compute),32.0)
            earlier = ttnn.matmul(prefix,chosen,dtype=ttnn.float32,compute_kernel_config=self.prefix_compute)
            slots = ttnn.gather(ttnn.add(earlier,region_start),dim=-1,index=index)
            if self.tiles(length):
                slots = ttnn.transpose(slots,-2,-1)
            target = ttnn.reshape(rm_uint32(slots),(1,1,1,length*8))
            dispatch_map = ttnn.scatter(map_zeros,dim=-1,index=target,src=tokens)
            offsets = ttnn.reshape(rm_uint32(region_start),(256,))
            return counts,offsets,dispatch_map,target,capacity
        counts = ttnn.experimental.deepseek_prefill.masked_bincount(
            ttnn.to_layout(ttnn.reshape(ids,(length,8)),ttnn.TILE_LAYOUT),self.mask,
            n_routed_experts=256,num_experts_per_token=8)
        count = ttnn.reshape(tiled_float(counts),(1,1,1,256))
        aligned = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(count,31.0),0.03125)),32.0)
        raw_start = ttnn.subtract(ttnn.cumsum(count,dim=-1),count)
        region_start = ttnn.subtract(ttnn.cumsum(aligned,dim=-1),aligned)
        if self.policy.dispatch_rank == "cumsum":
            zeros, ones, tokens = self.rank_buffers[length]
            index = ttnn.typecast(ttnn.to_layout(ttnn.reshape(ids,(1,1,length,8)),ttnn.TILE_LAYOUT),ttnn.uint32)
            chosen = ttnn.typecast(ttnn.scatter(zeros,dim=-1,index=index,src=ones),ttnn.float32)
            earlier = ttnn.subtract(ttnn.cumsum(chosen,dim=2),chosen)
            slots = ttnn.gather(ttnn.add(earlier,region_start),dim=-1,index=index)
            target = ttnn.reshape(rm_uint32(slots),(1,1,1,length*8))
            dispatch_map = ttnn.scatter(map_zeros,dim=-1,index=target,src=tokens)
            offsets = ttnn.reshape(rm_uint32(region_start),(256,))
            return counts,offsets,dispatch_map,target,capacity
        flat_ids = ttnn.typecast(ttnn.reshape(ttnn.typecast(ttnn.to_layout(ids,ttnn.ROW_MAJOR_LAYOUT),ttnn.uint32),
            (1,1,1,length*8)),ttnn.uint16)
        sorted_ids,permutation = ttnn.sort(flat_ids,dim=-1,descending=False,stable=True)
        shift = ttnn.gather(ttnn.subtract(region_start,raw_start),dim=-1,
                           index=ttnn.to_layout(sorted_ids,ttnn.TILE_LAYOUT))
        target = rm_uint32(ttnn.add(positions,shift))
        token = rm_uint32(ttnn.floor(ttnn.multiply(tiled_float(permutation),0.125)))
        dispatch_map = ttnn.scatter(map_zeros,dim=-1,index=target,src=token)
        inverse = ttnn.scatter(inverse_zeros,dim=-1,index=ttnn.typecast(permutation,ttnn.uint32),src=target)
        offsets = ttnn.reshape(rm_uint32(region_start),(256,))
        return counts,offsets,dispatch_map,inverse,capacity

    def __call__(self,hidden,routing):
        import ttnn

        values,ids = routing
        length = hidden.shape[2]
        counts,offsets,dispatch_map,inverse,capacity = self.route_maps(routing)
        source = ttnn.reshape(ttnn.to_layout(hidden,ttnn.ROW_MAJOR_LAYOUT),(length,2048))
        dispatched = ttnn.embedding(ttnn.reshape(dispatch_map,(1,capacity)),source,layout=ttnn.ROW_MAJOR_LAYOUT)
        dispatched = ttnn.reshape(dispatched,(1,1,capacity,2048))
        if self.policy.dispatch_input_dtype == "bfp8":
            dispatched = ttnn.to_layout(dispatched,ttnn.TILE_LAYOUT,dtype=ttnn.bfloat8_b,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.empty((1,1,capacity,2048),device=hidden.device(),dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.policy.dispatch_ffn == "fused":
            result = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
                dispatched,self.gate,self.up,self.down,counts,self.expert_ids,**getattr(self,"weight_options",{}),
                input_m_tiles=self.policy.dispatch_m_tiles or length//32,
                core_grid=ttnn.CoreCoord(*(self.policy.dispatch_grid_large[1:]
                                           if self.policy.dispatch_grid_large and length >= self.policy.dispatch_grid_large[0]
                                           else self.policy.dispatch_grid)),dtype=ttnn.bfloat16,output=output,
                expert_region_offsets=offsets,read_x_at_offset=True,
                compute_kernel_config=self.compute)
        else:
            routed = (ttnn.experimental.deepseek_prefill.unified_routed_expert_moe
                      if self.policy.dispatch_ffn == "unified" else
                      ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe)
            extra = ({} if self.policy.dispatch_ffn == "unified"
                     else {"hybrid_token_threshold": self.policy.dispatch_hybrid_threshold})
            result = routed(dispatched, ttnn.reshape(offsets, (offsets.shape[-1],)),
                            ttnn.reshape(counts, (counts.shape[-1],)), self.expert_table,
                            self.gate, self.up, self.down, length, compute_kernel_config=self.compute, **extra)
            if result.dtype != ttnn.bfloat16:
                result = ttnn.typecast(result, ttnn.bfloat16)
        restored = ttnn.embedding(ttnn.reshape(inverse,(1,length*8)),self.restore_table(result,capacity),
                                  layout=ttnn.ROW_MAJOR_LAYOUT)
        if self.tiles(length):
            experts = ttnn.to_layout(ttnn.reshape(restored,(8,1,length,2048)),ttnn.TILE_LAYOUT)
            scores = ttnn.transpose(ttnn.reshape(ttnn.transpose(ttnn.to_layout(values,ttnn.TILE_LAYOUT),-2,-1),
                                                 (8,1,1,length)),-2,-1)
            return ttnn.experimental.fast_reduce_nc(ttnn.multiply(experts,scores),dims=[0])
        return ttnn.experimental.deepseek_prefill.post_combine_reduce(
            ttnn.reshape(restored,(1,1,length,8,2048)),
            ttnn.reshape(ttnn.to_layout(values,ttnn.ROW_MAJOR_LAYOUT),(1,1,length,8,1)),expert_dim=3,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG)


class _StackedDispatch(_LocalDispatch):

    def __init__(self, experts, mesh, policy):
        import torch
        import ttnn

        if policy.dispatch_ffn != "fused" or policy.dispatch_weight_nd:
            raise ValueError("Stacked dispatch weights implement the fused-FFN interleaved policy")
        self.policy = policy
        self.bind_weights(experts.gate_up_weight, experts.down_weight)
        local = experts.gate_up_weight.shape[1]
        self.expert_ids = ttnn.from_torch(torch.arange(local, dtype=torch.int32), device=mesh, dtype=ttnn.uint32,
                                          layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        self.compute = ttnn.init_device_compute_kernel_config(mesh.arch(),math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,packer_l1_acc=False,math_approx_mode=False)
        self.prepare_maps(mesh)

    def bind_weights(self, gate_up, down):
        width = down.shape[-2]
        self.gate, self.up, self.down = [gate_up], [gate_up], [down]
        self.weight_options = dict(stacked_intermediate=width, gate_column_offset=width // 32, up_column_offset=0)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused Ornith decoder with the functional public context and persistent-state contract.

Prefill uses packed projections, depthwise Conv1d, head-major DeltaNet output,
and gated RMSNorm. Decode uses native paged attention and merged recurrent
operations. All packing and convolution preparation happen during setup.
"""

import ttnn

from .functional_decoder import FunctionalDecoder, _align_up, _slice_owned


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


def _field(tensor, start, end):
    return ttnn.slice(tensor, [0, 0, start], [tensor.shape[0], tensor.shape[1], end])


class FusedDecoder(FunctionalDecoder):
    """Fused device computation; inherited orchestration preserves arbitrary logical lengths."""

    conv1d_channel_chunk = 1024

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        cfg = decoder.cfg
        if kwargs.get("dtype", ttnn.bfloat16) != ttnn.bfloat16:
            raise ValueError("the validated fused decoder uses BF16 weights and activations")

        def upload(value, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                value.contiguous(),
                device=decoder.device,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
            )

        # Prefill packing wins substantially; independent decode projections are faster.
        gate_up = torch.cat([state_dict["mlp.gate_proj.weight"], state_dict["mlp.up_proj.weight"]])
        decoder.w["gate_up"] = upload(gate_up.T)
        if decoder.is_full_attention:
            qg = state_dict["self_attn.q_proj.weight"].reshape(cfg.n_heads, 2, cfg.head_dim, cfg.dim)
            q, gate = qg[:, 0].reshape(-1, cfg.dim), qg[:, 1].reshape(-1, cfg.dim)
            packed = torch.cat([q, state_dict["self_attn.k_proj.weight"], state_dict["self_attn.v_proj.weight"], gate])
            decoder.w["qkvg"] = upload(packed.T)
            for name in ("q_proj", "k_proj", "v_proj"):
                ttnn.deallocate(decoder.w.pop(name))
            decoder.w["rope_cos_tiled"] = ttnn.to_layout(decoder.rope.cos_table, ttnn.TILE_LAYOUT)
            decoder.w["rope_sin_tiled"] = ttnn.to_layout(decoder.rope.sin_table, ttnn.TILE_LAYOUT)
            decoder.w["batch_idxs"] = upload(
                torch.arange(32, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            return decoder

        if cfg.linear_conv_kernel_dim != 4 or cfg.linear_key_head_dim != cfg.linear_value_head_dim:
            raise ValueError("fused DeltaNet requires four convolution taps and equal Q/K/V head widths")
        packed = torch.cat([state_dict[f"linear_attn.in_proj_{name}.weight"] for name in ("qkv", "a", "b")])
        decoder.w["gdn_packed"] = upload(packed.T)
        decoder.w["gdn_z_epilogue"] = decoder.w.pop("gdn_z")
        for name in ("gdn_qkv", "gdn_a", "gdn_b"):
            ttnn.deallocate(decoder.w.pop(name))
        decoder.w["kda_norm_vector"] = upload(state_dict["linear_attn.norm.weight"].float().reshape(-1))
        decoder.beta_chain = [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, ttnn.DataType.FLOAT32.value, ttnn.DataType.BFLOAT16.value),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, ttnn.DataType.BFLOAT16.value, ttnn.DataType.FLOAT32.value),
        ]
        if cfg.linear_key_head_dim != 128:
            raise ValueError("the validated key-scaling chain requires 128-wide heads")
        decoder.key_scale_chain = [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 181.0 / 2048.0),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, ttnn.DataType.FLOAT32.value, ttnn.DataType.BFLOAT16.value),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, ttnn.DataType.BFLOAT16.value, ttnn.DataType.FLOAT32.value),
        ]
        grid = decoder.device.compute_with_storage_grid_size()
        # Native transpose requires reuse. Give each work block a whole head:
        # the reuse reader advances whole matrices between its batch iterations.
        decoder.outer_program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=4,
            per_core_N=4,
        )
        # Ordinary depthwise convolution retains a BF16 output before SiLU.
        # The KDA convolution's fused activation failed the batch32 HF accuracy gate.
        decoder.conv1d_host_weights = [
            ttnn.from_torch(
                state_dict["linear_attn.conv1d.weight"][start : start + decoder.conv1d_channel_chunk]
                .float()
                .unsqueeze(2)
                .contiguous(),
                dtype=ttnn.bfloat16,
            )
            for start in range(0, cfg.conv_dim, decoder.conv1d_channel_chunk)
        ]
        decoder.conv1d_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=False,
            act_block_h_override=32,
            output_layout=ttnn.TILE_LAYOUT,
        )
        return decoder

    def allocate_state(self, batch_size):
        super().allocate_state(batch_size)
        if self.is_full_attention:
            return
        width = self.conv1d_channel_chunk
        self.conv1d_weights = {}
        # Public prefill physical chunks are multiples128 and cannot exceed
        # the already allocated padding-neutrality position ramp.
        for seq in range(128, self.w["pos_ramp"].shape[1] + 1, 128):
            for channel_group, weight in enumerate(self.conv1d_host_weights):
                self.conv1d_weights[seq, channel_group] = ttnn.prepare_conv_weights(
                    weight_tensor=weight,
                    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                    weights_format="OIHW",
                    in_channels=width,
                    out_channels=width,
                    batch_size=batch_size,
                    input_height=1,
                    input_width=seq + 3,
                    kernel_size=(1, 4),
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, 1),
                    has_bias=False,
                    groups=width,
                    device=self.device,
                    input_dtype=ttnn.bfloat16,
                    output_dtype=ttnn.bfloat16,
                    conv_config=self.conv1d_config,
                    compute_config=self.compute_kernel_config,
                )

    def _block(self, x, *, mode, logical_len=None, page_table=None, chunk_start_idx=0, current_pos=None, rot_idxs=None):
        attn_in = self._norm(x, self.w["attn_norm"])
        if self.is_full_attention:
            if mode == "prefill":
                mixed = self._attention_prefill(attn_in, page_table, chunk_start_idx)
            else:
                mixed = self._attention_decode(attn_in, current_pos, rot_idxs, page_table)
        elif mode == "prefill":
            mixed = self._gdn_prefill(attn_in, logical_len)
        else:
            mixed = self._gdn_decode(attn_in)
        ttnn.deallocate(attn_in)
        h = ttnn.add(x, mixed)
        ttnn.deallocate(mixed)
        ff_in = self._norm(h, self.w["ff_norm"])
        activated = self._activate_mlp(ff_in, mode)
        ttnn.deallocate(ff_in)
        ff_out = ttnn.linear(activated, self.w["down_proj"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(activated)
        out = ttnn.add(h, ff_out)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out

    def _activate_mlp(self, ff_in, mode):
        if mode == "prefill":
            gate_up = ttnn.linear(ff_in, self.w["gate_up"], compute_kernel_config=self.compute_kernel_config)
            gate, up = ttnn.chunk(gate_up, 2, dim=-1)
            ttnn.deallocate(gate_up)
        else:
            gate = ttnn.linear(ff_in, self.w["gate_proj"], compute_kernel_config=self.compute_kernel_config)
            up = ttnn.linear(ff_in, self.w["up_proj"], compute_kernel_config=self.compute_kernel_config)
        result = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return result

    def _project_qkv(self, x):
        batch, seq, _ = list(x.shape)
        cfg = self.cfg
        qw = cfg.n_heads * cfg.head_dim
        end = qw + 2 * cfg.n_kv_heads * cfg.head_dim
        packed = ttnn.linear(x, self.w["qkvg"], compute_kernel_config=self.compute_kernel_config)
        qkv = ttnn.slice(packed, [0, 0, 0], [batch, seq, end])
        gate = ttnn.slice(packed, [0, 0, end], [batch, seq, end + qw])
        ttnn.deallocate(packed)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=cfg.n_heads, num_kv_heads=cfg.n_kv_heads, transpose_key=False
        )
        ttnn.deallocate(qkv)
        q_norm, k_norm = self._norm(q, self.w["q_norm"]), self._norm(k, self.w["k_norm"])
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        return q_norm, k_norm, v, gate

    def _apply_partial_rope(self, value, cos, sin):
        batch, heads, seq, dim = list(value.shape)
        rd = self.cfg.rope_dim
        partial = ttnn.slice(value, [0, 0, 0, 0], [batch, heads, seq, rd])
        rotated = ttnn.experimental.rotary_embedding(partial, cos, sin)
        tail = ttnn.slice(value, [0, 0, 0, rd], [batch, heads, seq, dim])
        result = ttnn.concat([rotated, tail], dim=-1)
        for tensor in (value, partial, rotated, tail):
            ttnn.deallocate(tensor)
        return result

    def _attention_output(self, attn, gate):
        gated = ttnn.multiply(gate, attn, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(attn)
        ttnn.deallocate(gate)
        result = ttnn.linear(gated, self.w["o_proj"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(gated)
        return result

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

    def _decode_finish(self, attn, gate, batch):
        return self._attention_output(ttnn.reshape(attn, [batch, 1, self.cfg.n_heads * self.cfg.head_dim]), gate)

    def _gdn_project(self, x):
        cfg = self.cfg
        # QKV/A/B share a projection; Z has its own activation epilogue.
        packed = ttnn.linear(x, self.w["gdn_packed"], compute_kernel_config=self.compute_kernel_config)
        qkv = _field(packed, 0, cfg.conv_dim)
        a = _field(packed, cfg.conv_dim, cfg.conv_dim + cfg.linear_num_value_heads)
        b = _field(packed, cfg.conv_dim + cfg.linear_num_value_heads, cfg.conv_dim + 2 * cfg.linear_num_value_heads)
        ttnn.deallocate(packed)
        # Both modes use the same full-grid Z projection geometry.
        grid = self.device.compute_with_storage_grid_size()
        z_args = {"core_grid": ttnn.CoreGrid(x=grid.x, y=grid.y)}
        # Public prefill physically pads every chunk to at least128 tokens.
        # Only decode reaches this boundary with time1; no runtime data is read.
        if int(x.shape[1]) == 1:
            z_args["activation"] = "silu"
        z = ttnn.linear(
            x,
            self.w["gdn_z_epilogue"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            **z_args,
        )
        return qkv, z, a, b

    def _gdn_prefill_conv_fields(self, qkv, logical_len):
        cfg = self.cfg
        batch, seq = qkv.shape[0], qkv.shape[1]
        width = self.conv1d_channel_chunk
        rows_rm = [
            ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG) for row in self.conv_state
        ]
        history_rm = ttnn.concat(rows_rm, dim=1)
        for tensor in rows_rm:
            ttnn.deallocate(tensor)
        tokens_rm = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        padded = ttnn.concat([history_rm, tokens_rm], dim=1)
        ttnn.deallocate(history_rm)
        ttnn.deallocate(tokens_rm)
        parts = []
        for channel_group, start in enumerate(range(0, cfg.conv_dim, width)):
            piece = ttnn.slice(padded, [0, 0, start], [batch, seq + 3, start + width])
            conv = ttnn.conv1d(
                input_tensor=piece,
                weight_tensor=self.conv1d_weights[seq, channel_group],
                device=self.device,
                in_channels=width,
                out_channels=width,
                batch_size=batch,
                input_length=seq + 3,
                kernel_size=4,
                stride=1,
                padding=0,
                dilation=1,
                groups=width,
                dtype=ttnn.bfloat16,
                conv_config=self.conv1d_config,
                compute_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                return_output_dim=False,
                return_weights_and_bias=False,
            )
            ttnn.deallocate(piece)
            flattened = ttnn.reshape(conv, [batch, seq, width])
            activated = ttnn.silu(flattened, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(conv)
            parts.append(activated)
        fields = []
        group_start = 0
        for field_width in (cfg.linear_q_dim, cfg.linear_k_dim, cfg.linear_v_dim):
            if field_width % width:
                raise ValueError("Conv1d channel chunk must divide each Q/K/V field")
            group_end = group_start + field_width // width
            field_parts = parts[group_start:group_end]
            if len(field_parts) == 1:
                fields.append(field_parts[0])
            else:
                fields.append(ttnn.concat(field_parts, dim=2))
                for tensor in field_parts:
                    ttnn.deallocate(tensor)
            group_start = group_end
        q, k, v = fields
        tail_rm = ttnn.slice(padded, [0, logical_len, 0], [batch, logical_len + 3, cfg.conv_dim])
        tail = tail_rm
        ttnn.deallocate(padded)
        return q, k, v, tail

    def _write_conv_state(self, tail_rm):
        for index, buffer in enumerate(self.conv_state):
            row_rm = ttnn.slice(tail_rm, [0, index, 0], [tail_rm.shape[0], index + 1, tail_rm.shape[2]])
            row = ttnn.to_layout(row_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.copy(row, buffer)
            ttnn.deallocate(row)
            ttnn.deallocate(row_rm)

    def _gdn_prefill(self, x, logical_len):
        batch, seq = x.shape[0], x.shape[1]
        qkv, z, a, b = self._gdn_project(x)
        q, k, v, tail = self._gdn_prefill_conv_fields(qkv, logical_len)
        ttnn.deallocate(qkv)
        beta, g = self._gdn_gates_projected(a, b, logical_len, seq)
        for tensor in (a, b):
            ttnn.deallocate(tensor)
        core, final = self._chunk_delta_rule(q, k, v, g, beta)
        for tensor in (q, k, v, g, beta):
            ttnn.deallocate(tensor)
        ttnn.copy(final, self.recurrent_state)
        ttnn.deallocate(final)
        self._write_conv_state(tail)
        ttnn.deallocate(tail)
        result = self._gdn_out_head_major(core, z, batch, seq)
        ttnn.deallocate(z)
        return result

    def _chunk_delta_rule(self, q, k, v, g, beta):
        cfg = self.cfg
        batch, seq = q.shape[0], q.shape[1]
        nk, dk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
        nv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
        eye, tril, ones, masks = self.w["gdn_const_tiles"]
        normalized = []
        for tensor in (q, k):
            heads = self._split_heads(tensor, nk, dk)
            rms = ttnn.rms_norm(heads, epsilon=1e-6 / dk)
            norm = ttnn.multiply(
                rms,
                dk**-0.5,
                memory_config=(ttnn.L1_MEMORY_CONFIG if batch * seq <= 512 else ttnn.DRAM_MEMORY_CONFIG),
            )
            ttnn.deallocate(heads)
            ttnn.deallocate(rms)
            normalized.append(norm)
        q_norm, k_norm = normalized

        def launch(q_, k_, v_, g_, beta_, state_):
            return ttnn.transformer.chunk_gated_delta_rule(
                q_,
                k_,
                v_,
                g_,
                beta_,
                initial_state=state_,
                output_final_state=True,
                chunk_size=self.w["gdn_chunk_size"],
                use_qk_l2norm=False,
                output_head_major=True,
                eye=eye,
                tril=tril,
                ones=ones,
                masks=masks,
            )

        step = self.max_gdn_prefill_batch()
        if batch <= step:
            result = launch(q_norm, k_norm, v, g, beta, self.recurrent_state)
        else:
            cores, states = [], []
            for start in range(0, batch, step):
                end = min(start + step, batch)
                q_s = ttnn.slice(q_norm, [start, 0, 0, 0], [end, seq, nk, dk])
                k_s = ttnn.slice(k_norm, [start, 0, 0, 0], [end, seq, nk, dk])
                v_s = ttnn.slice(v, [start, 0, 0], [end, seq, nv * dv])
                g_s = ttnn.slice(g, [start, 0, 0], [end, seq, nv])
                beta_s = ttnn.slice(beta, [start, 0, 0], [end, seq, nv])
                state_s = ttnn.slice(self.recurrent_state, [start, 0, 0, 0], [end, nv, dk, dv])
                core_s, final_s = launch(q_s, k_s, v_s, g_s, beta_s, state_s)
                for tensor in (q_s, k_s, v_s, g_s, beta_s, state_s):
                    ttnn.deallocate(tensor)
                cores.append(core_s)
                states.append(final_s)
            result = ttnn.concat(cores, dim=0), ttnn.concat(states, dim=0)
            for tensor in (*cores, *states):
                ttnn.deallocate(tensor)
        for tensor in normalized:
            ttnn.deallocate(tensor)
        return result

    def _gdn_gates_projected(self, a_raw, b_raw, logical_len, seq_len):
        # Preserve the functional rounding boundaries: sigmoid(BF16) -> FP32,
        # while A is converted to FP32 before bias and softplus.
        if seq_len == 1:
            beta = ttnn.unary_chain(b_raw, self.beta_chain)
            biased = ttnn.add(a_raw, self.w["dt_bias"], dtype=ttnn.float32)
        else:
            beta16 = ttnn.sigmoid(b_raw)
            beta = ttnn.typecast(beta16, ttnn.float32)
            ttnn.deallocate(beta16)
            a32 = ttnn.typecast(a_raw, ttnn.float32)
            biased = ttnn.add(a32, self.w["dt_bias"])
            ttnn.deallocate(a32)
        soft = ttnn.softplus(biased)
        ttnn.deallocate(biased)
        g = ttnn.multiply(self.w["A_neg"], soft)
        ttnn.deallocate(soft)
        if logical_len < seq_len:
            ramp, owned = _slice_owned(self.w["pos_ramp"], [0, 0, 0], [1, seq_len, 1])
            keep = ttnn.typecast(ttnn.lt(ramp, float(logical_len)), ttnn.float32)
            if owned:
                ttnn.deallocate(ramp)
            masked_beta = ttnn.multiply(beta, keep)
            masked_g = ttnn.multiply(g, keep)
            for tensor in (beta, g, keep):
                ttnn.deallocate(tensor)
            beta, g = masked_beta, masked_g
        return beta, g

    def _gdn_conv_decode(self, qkv):
        kernel = self.cfg.linear_conv_kernel_dim
        acc = ttnn.multiply(qkv, self.w["conv_taps"][kernel - 1])
        for tap in range(kernel - 1):
            previous = acc
            acc = ttnn.addcmul(previous, self.conv_state[tap], self.w["conv_taps"][tap])
            ttnn.deallocate(previous)
        activated = ttnn.silu(acc)
        ttnn.deallocate(acc)
        for idx in range(kernel - 2):
            ttnn.copy(self.conv_state[idx + 1], self.conv_state[idx])
        ttnn.copy(qkv, self.conv_state[kernel - 2])
        return activated

    def _gdn_decode_heads(self, activated):
        cfg = self.cfg
        batch = activated.shape[0]
        nk, nv, dk = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim
        rows = ttnn.reshape(activated, [batch, 1, 2 * nk + nv, dk])
        heads = ttnn.permute(rows, (0, 2, 1, 3))
        v = ttnn.slice(heads, [0, 2 * nk, 0, 0], [batch, 2 * nk + nv, 1, dk])
        qk = ttnn.slice(heads, [0, 0, 0, 0], [batch, 2 * nk, 1, dk])
        ttnn.deallocate(heads)
        normed = ttnn.rms_norm(qk, epsilon=1e-6 / dk)
        ttnn.deallocate(qk)
        if nv // nk > 1:
            expanded = ttnn.repeat_interleave(normed, nv // nk, dim=1)
            ttnn.deallocate(normed)
            normed = expanded
        q = ttnn.slice(normed, [0, 0, 0, 0], [batch, nv, 1, dk])
        k = ttnn.slice(normed, [0, nv, 0, 0], [batch, 2 * nv, 1, dk])
        ttnn.deallocate(normed)
        return q, k, v

    def _delta_rule_step(self, q, k, v, beta, g):
        cfg = self.cfg
        batch, nv, dk = q.shape[0], cfg.linear_num_value_heads, cfg.linear_key_head_dim
        dram = ttnn.DRAM_MEMORY_CONFIG
        # q/k already contain the same BF16 RMSNorm outputs produced by the
        # separate per-head norms; scalar/cast rounding points remain intact.
        q_row = ttnn.multiply(q, dk**-1.0, dtype=ttnn.float32, memory_config=dram)
        # Preserve BinaryNG's BF16 scalar and BF16 product rounding before FP32 output.
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
        outer = ttnn.matmul(
            k_row,
            delta,
            transpose_a=True,
            program_config=self.outer_program_config,
            memory_config=dram,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(k_row)
        ttnn.deallocate(delta)
        ttnn.add(state, outer, output_tensor=state)
        ttnn.deallocate(outer)
        result = ttnn.matmul(q_row, state, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(q_row)
        return result

    def _gdn_decode(self, x):
        qkv, z, a, b = self._gdn_project(x)
        activated = self._gdn_conv_decode(qkv)
        ttnn.deallocate(qkv)
        q, k, v = self._gdn_decode_heads(activated)
        ttnn.deallocate(activated)
        beta, g = self._gdn_gates_projected(a, b, 1, 1)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        core = self._delta_rule_step(q, k, v, beta, g)
        for tensor in (q, k, v, beta, g):
            ttnn.deallocate(tensor)
        result = self._gdn_out_head_major(core, z, x.shape[0], 1)
        ttnn.deallocate(z)
        return result

    def _gdn_out_head_major(self, core, z, batch, seq):
        if seq > 1:
            normed = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                core,
                z,
                self.w["kda_norm_vector"],
                self.cfg.linear_num_value_heads,
                epsilon=self.cfg.norm_eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                output_dtype=ttnn.float32,
            )
            ttnn.deallocate(core)
            gated = ttnn.multiply(normed, z)
            ttnn.deallocate(normed)
        else:
            normed = ttnn.rms_norm(core, weight=self.w["gdn_norm"], epsilon=self.cfg.norm_eps)
            ttnn.deallocate(core)
            heads = ttnn.reshape(normed, [batch, self.cfg.linear_num_value_heads, 1, self.cfg.linear_value_head_dim])
            combined = ttnn.permute(heads, (0, 2, 1, 3))
            ttnn.deallocate(normed)
            merged = ttnn.reshape(combined, [batch, 1, self.cfg.linear_v_dim])
            # Z already contains its matmul SiLU epilogue in decode.
            gated = ttnn.multiply(merged, z)
            ttnn.deallocate(combined)
        result = ttnn.linear(gated, self.w["gdn_out"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(gated)
        return result

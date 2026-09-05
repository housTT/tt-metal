# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Focused recurrent-state and RoPE movement candidates; experiments only."""

import math
import os

import ttnn

from ..tt.fused_decoder import _align_up, _batch_grid, _height_memory, _rectangular_rope_grid, _slice_owned
from .optimization_baseline import OptimizedDecoder


class RecurrentConfigCandidate(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        g = decoder.device.compute_with_storage_grid_size()
        grid = tuple(int(v) for v in os.environ.get("ORNITH_STATE_GRID", f"{g.x},{g.y}").split(","))
        decoder.state_read_program = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=int(os.environ.get("ORNITH_STATE_BLOCK", "4")),
            out_subblock_h=1,
            out_subblock_w=int(os.environ.get("ORNITH_STATE_SUBBLOCK", "4")),
            per_core_M=1,
            per_core_N=4,
        )
        decoder.state_compute = ttnn.init_device_compute_kernel_config(
            decoder.device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("ORNITH_STATE_FIDELITY", "HiFi4")),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        decoder.requested_l1_intermediates = os.environ.get("ORNITH_STATE_INTERMEDIATES", "dram") == "l1"
        decoder.recurrent_l1_intermediates = decoder.requested_l1_intermediates
        return decoder

    def allocate_state(self, batch_size):
        super().allocate_state(batch_size)
        if not self.is_full_attention:
            # Validated with borrowed width-sharded public inputs: persistent
            # state fits through 300 KiB/bank (B16 on 110-bank P300c), but the
            # outer-product working set needs a 224 KiB/bank bound (B12).
            # B13 L1 outer collides with its CBs; B31 L1 state collides with
            # prefill RMSNorm. Select both policies before any trace capture.
            grid = self.device.compute_with_storage_grid_size()
            state_tiles = math.prod(self.recurrent_state.padded_shape) // 1024
            state_bytes_per_bank = math.ceil(state_tiles / (grid.x * grid.y)) * 4096
            fits_l1 = state_bytes_per_bank <= 300 * 1024
            self.recurrent_l1_intermediates = self.requested_l1_intermediates and state_bytes_per_bank <= 224 * 1024
            if os.environ.get("ORNITH_STATE_MEMORY", "dram") == "l1" and fits_l1:
                old = self.recurrent_state
                self.recurrent_state = ttnn.to_memory_config(old, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(old)

    def _delta_rule_step(self, q, k, v, beta, g):
        cfg = self.cfg
        batch, nv, dk = q.shape[0], cfg.linear_num_value_heads, cfg.linear_key_head_dim
        dram = ttnn.L1_MEMORY_CONFIG if self.recurrent_l1_intermediates else ttnn.DRAM_MEMORY_CONFIG
        # q/k already contain the same BF16 RMSNorm outputs produced by the
        # separate per-head norms; scalar/cast rounding points remain intact.
        q_row = ttnn.multiply(q, dk**-1.0, dtype=ttnn.float32, memory_config=dram)
        # Preserve BinaryNG's BF16 scalar and BF16 product rounding before FP32 output.
        k_row = ttnn.unary_chain(k, self.key_scale_chain, memory_config=ttnn.L1_MEMORY_CONFIG)
        beta_view = ttnn.reshape(beta, [batch, nv, 1, 1])
        g_view = ttnn.reshape(g, [batch, nv, 1, 1])
        state = self.recurrent_state
        ttnn.multiply(state, g_view, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], output_tensor=state)
        read = ttnn.matmul(
            k_row,
            state,
            memory_config=dram,
            program_config=self.state_read_program,
            compute_kernel_config=self.state_compute,
        )
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
        result = ttnn.matmul(
            q_row,
            state,
            memory_config=dram,
            program_config=self.state_read_program,
            compute_kernel_config=self.state_compute,
        )
        ttnn.deallocate(q_row)
        return result


class TiledRotaryIndicesCandidate(OptimizedDecoder):
    def _decode_rotary(self, q, k, rot_idxs, batch):
        grid, users_per_core = _rectangular_rope_grid(self.device, batch)
        rd, dim = self.cfg.rope_dim, self.cfg.head_dim
        rotated_mem = _height_memory(grid, rd, users_per_core)
        tail_mem = _height_memory(grid, dim - rd, users_per_core)
        output_mem = _height_memory(grid, dim, users_per_core)
        indices = ttnn.to_layout(rot_idxs, ttnn.TILE_LAYOUT)
        cos = ttnn.embedding(indices, self.rope.cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(indices, self.rope.sin_table, layout=ttnn.TILE_LAYOUT)
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


class LargePrefillCandidate(OptimizedDecoder):
    """Explicit two-dimensional work blocks for every material prefill role."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import math

        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.prefill_weights = {}
        multiple = int(os.environ.get("ORNITH_PREFILL_PAD_N", "0"))
        if multiple:
            for role in decoder.projection_compute:
                weight = decoder.w[role]
                width = weight.shape[-1]
                padding = math.ceil(width / multiple) * multiple - width
                if padding:
                    host = torch.nn.functional.pad(ttnn.to_torch(weight), (0, padding))
                    decoder.prefill_weights[role] = ttnn.from_torch(
                        host,
                        device=decoder.device,
                        dtype=weight.dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
                    )
        return decoder

    def _linear(self, x, role, **kwargs):
        if int(x.shape[1]) == 1:
            return super()._linear(x, role, **kwargs)
        import math

        grid = tuple(int(v) for v in os.environ.get("ORNITH_PREFILL_GRID", "8,8").split(","))
        weight = self.prefill_weights.get(role, self.w[role])
        k, n = weight.shape[-2], weight.shape[-1]
        m = math.prod(list(x.padded_shape)[:-1]) // 32
        pm, pn = math.ceil(m / grid[1]), math.ceil(n / 32 / grid[0])
        block_w = int(os.environ.get("ORNITH_PREFILL_BLOCK", "4"))
        block_m = max(v for v in range(1, min(pm, int(os.environ.get("ORNITH_PREFILL_OUT_M", "4"))) + 1) if pm % v == 0)
        block_n = max(
            v for v in range(1, min(pn, int(os.environ.get("ORNITH_PREFILL_OUT_N", "32"))) + 1) if pn % v == 0
        )
        subblock_h = max(
            v for v in range(1, min(block_m, int(os.environ.get("ORNITH_PREFILL_SUB_H", "1"))) + 1) if block_m % v == 0
        )
        subblock = max(
            v
            for v in range(1, min(8 // subblock_h, int(os.environ.get("ORNITH_PREFILL_SUB_W", "4"))) + 1)
            if block_n % v == 0
        )
        kwargs.pop("core_grid", None)
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=block_w,
            per_core_M=pm,
            per_core_N=pn,
            out_block_h=block_m,
            out_block_w=block_n,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock,
            transpose_mcast=False,
            fuse_batch=True,
        )
        out = ttnn.linear(
            x,
            weight,
            program_config=program,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.projection_compute[role],
            **kwargs,
        )
        if n != self.w[role].shape[-1]:
            out = ttnn.slice(out, [0, 0, 0], [x.shape[0], x.shape[1], self.w[role].shape[-1]])
        return out


class CacheSDPACandidate(OptimizedDecoder):
    def allocate_kv_cache(self, num_blocks, dtype=None):
        return super().allocate_kv_cache(
            num_blocks, dtype=dtype or getattr(ttnn, os.environ.get("ORNITH_KV_DTYPE", "bfloat8_b"))
        )

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
        k = ttnn.typecast(k, self.k_cache.dtype)
        v = ttnn.typecast(v, self.v_cache.dtype)
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
        packed = self._linear(x, "qkvg")
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
        sdpa_grid = ttnn.CoreCoord(*[int(v) for v in os.environ.get("ORNITH_SDPA_GRID", "8,8").split(",")])
        q = self._sdpa_query(q, batch, sdpa_grid)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            is_causal=True,
            scale=cfg.head_dim**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(
                    *[int(v) for v in os.environ.get("ORNITH_SDPA_GRID", "8,8").split(",")]
                ),
                q_chunk_size=32,
                k_chunk_size=int(os.environ.get("ORNITH_SDPA_CHUNK", "64")),
                exp_approx_mode=False,
            ),
        )
        return self._decode_finish(attn, gate, batch)


class ProjectionTopologyCandidate(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import math

        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.topology = os.environ.get("ORNITH_TOPOLOGY", "gate_epilogue")
        weights = {}
        source_role = "qkvg" if decoder.is_full_attention else "gdn_packed"
        if decoder.topology == "separate_attention" and decoder.is_full_attention:
            cfg = decoder.cfg
            qg = state_dict["self_attn.q_proj.weight"].reshape(cfg.n_heads, 2, cfg.head_dim, cfg.dim)
            weights = {
                "separate_q": qg[:, 0].reshape(-1, cfg.dim).T,
                "separate_gate": qg[:, 1].reshape(-1, cfg.dim).T,
                "separate_k": state_dict["self_attn.k_proj.weight"].T,
                "separate_v": state_dict["self_attn.v_proj.weight"].T,
            }
        if decoder.topology == "packed_gdn" and not decoder.is_full_attention:
            weights["gdn_all"] = torch.cat(
                [state_dict[f"linear_attn.in_proj_{n}.weight"] for n in ("qkv", "a", "b", "z")]
            ).T
        if decoder.topology == "separate_gdn" and not decoder.is_full_attention:
            weights["separate_gdn_qkv"] = state_dict["linear_attn.in_proj_qkv.weight"].T
            weights["separate_gdn_ab"] = torch.cat(
                [state_dict[f"linear_attn.in_proj_{n}.weight"] for n in ("a", "b")]
            ).T
        dg = decoder.device.dram_grid_size()
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))])
        for role, value in weights.items():
            decoder.w[role] = ttnn.from_torch(
                value.contiguous(),
                device=decoder.device,
                dtype=decoder.w[source_role].dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
            )
            decoder.projection_compute[role] = decoder.projection_compute[source_role]
            if role in decoder.optimization.dram_roles:
                k, n = value.shape
                readers = decoder._role_config(role)[2]
                mem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(
                        grid,
                        [k, math.ceil(n / (32 * dg.x * dg.y * readers)) * 32 * readers],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
                decoder.decode_weights[role] = ttnn.from_torch(
                    value.contiguous(),
                    device=decoder.device,
                    dtype=decoder.w[role].dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
                )
        return decoder

    def _linear(self, x, role, **kwargs):
        if role == "qkvg" and self.topology == "separate_attention":
            parts = [
                super(ProjectionTopologyCandidate, self)._linear(x, n)
                for n in ("separate_q", "separate_k", "separate_v", "separate_gate")
            ]
            result = ttnn.concat(
                parts, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG if x.shape[1] == 1 else ttnn.DRAM_MEMORY_CONFIG
            )
            for part in parts:
                ttnn.deallocate(part)
            return result
        return super()._linear(x, role, **kwargs)

    def _activate_mlp(self, ff_in, mode):
        if mode != "decode" or self.topology != "gate_epilogue":
            return super()._activate_mlp(ff_in, mode)
        gate = self._linear(ff_in, "gate_proj", activation="silu")
        up = self._linear(ff_in, "up_proj")
        result = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return result

    def _gdn_project(self, x):
        if self.topology == "separate_gdn":
            from ..tt.fused_decoder import _field

            qkv = self._linear(x, "separate_gdn_qkv")
            ab = self._linear(x, "separate_gdn_ab")
            heads = self.cfg.linear_num_value_heads
            a, b = _field(ab, 0, heads), _field(ab, heads, 2 * heads)
            ttnn.deallocate(ab)
            grid = self.device.compute_with_storage_grid_size()
            kwargs = {"core_grid": ttnn.CoreGrid(x=grid.x, y=grid.y)}
            if x.shape[1] == 1:
                kwargs["activation"] = "silu"
            z = self._linear(x, "gdn_z_epilogue", **kwargs)
            return qkv, z, a, b
        if self.topology != "packed_gdn":
            return super()._gdn_project(x)
        from ..tt.fused_decoder import _field

        cfg = self.cfg
        p = self._linear(x, "gdn_all")
        qkv = _field(p, 0, cfg.conv_dim)
        a = _field(p, cfg.conv_dim, cfg.conv_dim + cfg.linear_num_value_heads)
        b = _field(p, cfg.conv_dim + cfg.linear_num_value_heads, cfg.conv_dim + 2 * cfg.linear_num_value_heads)
        z = _field(
            p,
            cfg.conv_dim + 2 * cfg.linear_num_value_heads,
            cfg.conv_dim + 2 * cfg.linear_num_value_heads + cfg.linear_v_dim,
        )
        ttnn.deallocate(p)
        if x.shape[1] == 1:
            z = ttnn.silu(z)
        return qkv, z, a, b


class ActivationCandidate(OptimizedDecoder):
    def _linear(self, x, role, **kwargs):
        roles = os.environ.get("ORNITH_ACTIVATION_ROLES", "gate_proj,up_proj,down_proj").split(",")
        if x.shape[1] == 1 and role in roles:
            kwargs.setdefault("dtype", x.dtype)
            x = ttnn.typecast(x, ttnn.bfloat8_b)
        return super()._linear(x, role, **kwargs)


class ShardedHeadNormCandidate(OptimizedDecoder):
    def _norm(self, x, weight):
        if x.is_sharded() and x.shape[-1] == self.cfg.head_dim:
            pass

            batch = int(x.shape[1])
            cores = int(os.environ.get("ORNITH_HEAD_NORM_CORES", "8"))
            mem = self._width_memory(self.cfg.head_dim, cores, rows=batch * 32)
            x = ttnn.to_memory_config(x, mem)
            program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(cores, 1),
                block_h=batch,
                block_w=self.cfg.head_dim // 32 // cores,
                subblock_w=1,
                inplace=False,
            )
            return ttnn.rms_norm(x, weight=weight, epsilon=self.cfg.norm_eps, program_config=program, memory_config=mem)
        return super()._norm(x, weight)

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
        gate = ttnn.slice(packed, [0, 0, 0, qw + kvw], [1, 1, batch, 2 * qw + kvw])
        gate = ttnn.reshape(gate, [batch, 1, qw])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=cfg.n_heads,
            num_kv_heads=cfg.n_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
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
        sdpa_grid = ttnn.CoreCoord(*[int(v) for v in os.environ.get("ORNITH_SDPA_GRID", "8,8").split(",")])
        q = self._sdpa_query(q, batch, sdpa_grid)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            is_causal=True,
            scale=cfg.head_dim**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=sdpa_grid,
                q_chunk_size=32,
                k_chunk_size=int(os.environ.get("ORNITH_SDPA_CHUNK", "64")),
                exp_approx_mode=False,
            ),
        )
        return self._decode_finish(attn, gate, batch)


class SharedMLPInputCandidate(OptimizedDecoder):
    def _activate_mlp(self, ff_in, mode):
        if mode == "decode" and ff_in.shape[0] == 1:
            cores = self._role_config("gate_proj")[0]
            ff_in = ttnn.to_memory_config(ff_in, self._width_memory(self.cfg.dim, cores))
        return super()._activate_mlp(ff_in, mode)


class KDAConvCandidate(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.kda_conv_program = ttnn.QkvCausalConv1dSiluProgramConfig(
            channel_chunk_size=int(os.environ.get("ORNITH_CONV_CHUNK", "256"))
        )
        return decoder

    def _gdn_prefill_conv_fields(self, qkv, logical_len):
        from .linear_fusion_candidates import _kda_prefill_fields_with_rm_tail

        return _kda_prefill_fields_with_rm_tail(self, qkv, logical_len)


class PackedAlignedMLPCandidate(OptimizedDecoder):
    def _activate_mlp(self, ff_in, mode):
        if mode != "decode":
            return super()._activate_mlp(ff_in, mode)
        packed = self._linear(ff_in, "gate_up")
        b, _, width = packed.shape
        half = width // 2
        mem = self._width_memory(half, self._role_config("down_proj")[0])
        gate = ttnn.slice(packed, [0, 0, 0], [b, 1, half], memory_config=mem)
        up = ttnn.slice(packed, [0, 0, half], [b, 1, width], memory_config=mem)
        ttnn.deallocate(packed)
        result = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return result


class LowerMovementCandidate(OptimizedDecoder):
    def _linear(self, x, role, **kwargs):
        if x.shape[1] != 1:
            return super()._linear(x, role, **kwargs)
        if x.shape[1] == 1 and role in self.decode_weights:
            batch, _, k = x.shape
            cores, block_w, readers = self._role_config(role)
            n = self.w[role].shape[-1]
            folded = ttnn.reshape(x, [1, batch, k])
            mem = self._width_memory(k, cores)
            working = ttnn.to_memory_config(folded, mem)
            activation = kwargs.pop("activation", None)
            kwargs.pop("core_grid", None)
            program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block_w,
                per_core_M=1,
                per_core_N=math.ceil(n / (32 * cores)),
                num_workers_per_dram_bank=readers,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation == "silu" else None,
            )
            out = ttnn.linear(
                working,
                self.decode_weights[role],
                program_config=program,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.projection_compute[role],
                **kwargs,
            )
            # Mixer helpers use head-shaped tensors. MLP consumes the shared width shards directly.
            if role not in ("gate_proj", "up_proj", "down_proj", "gate_up", "gdn_out", "o_proj") or batch > 1:
                out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
            return ttnn.reshape(out, [batch, 1, n])
        if x.is_sharded():
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        return ttnn.linear(
            x,
            self.w[role],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.projection_compute[role],
            **kwargs,
        )

    def _block(self, x, *, mode, logical_len=None, page_table=None, chunk_start_idx=0, current_pos=None, rot_idxs=None):
        residual_mem = (
            self._width_memory(self.cfg.dim, self.optimization.residual_cores, 32)
            if mode == "decode" and self.optimization.residual_cores
            else ttnn.DRAM_MEMORY_CONFIG
        )
        compact_residual = mode == "decode" and x.shape[0] > 1 and self.optimization.residual_cores
        if mode == "decode" and not compact_residual:
            x = ttnn.to_memory_config(x, residual_mem)
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
        if not compact_residual:
            mixed = ttnn.to_memory_config(mixed, residual_mem)
        h = self._residual_add(x, mixed, residual_mem)
        ttnn.deallocate(mixed)
        ff_in = self._norm(h, self.w["ff_norm"])
        activated = self._activate_mlp(ff_in, mode)
        ttnn.deallocate(ff_in)
        ff_out = self._linear(activated, "down_proj")
        ttnn.deallocate(activated)
        if not compact_residual:
            ff_out = ttnn.to_memory_config(ff_out, residual_mem)
        out = self._residual_add(h, ff_out, residual_mem)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out


class CombinedDecodeCandidate(
    LowerMovementCandidate,
    ShardedHeadNormCandidate,
    CacheSDPACandidate,
    RecurrentConfigCandidate,
    SharedMLPInputCandidate,
):
    """Combine independently measured decode/cache improvements for integration."""


class CombinedPrefillCandidate(CombinedDecodeCandidate, LargePrefillCandidate):
    """Also use explicit prefill programs."""


class CombinedKDAPrefillCandidate(CombinedPrefillCandidate, KDAConvCandidate):
    """Also use KDA's prefill convolution."""


class ProjectionOutputCandidate(OptimizedDecoder):
    def _linear(self, x, role, **kwargs):
        if role == "gdn_out":
            kwargs["dtype"] = ttnn.bfloat16
        return super()._linear(x, role, **kwargs)


class CombinedOutputCandidate(ProjectionOutputCandidate, CombinedDecodeCandidate):
    """Keep the GDN output projection and following residual in BF16."""

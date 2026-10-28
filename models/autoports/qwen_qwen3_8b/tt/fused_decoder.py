# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused TTNN decoder layer for Qwen/Qwen3-8B on one Blackhole chip.

``FusedDecoder`` keeps the public contract, weights, paged KV-cache behaviour,
chunk planning and padding rules of ``FunctionalDecoder`` (see that module's
docstring) and replaces the op graph of the measured prefill and decode paths
with a fused, numerically equivalent one.

Prefill (per physical chunk) relative to the functional graph:

* ``ttnn.experimental.minimal_matmul`` replaces ``ttnn.linear`` for the QKV,
  gate and up projections once a chunk has ``PREFILL_MINIMAL_MATMUL_MIN_ROWS``
  rows or more.
* ``ttnn.experimental.dit_minimal_matmul_addcmul_fused`` computes the O and
  down projections with the residual add folded into the matmul kernel
  (``residual + 1.0 * (x @ W) * ones``), removing both ``ttnn.add`` ops.

Decode relative to the functional graph:

* The per-head Q/K RMSNorm writes its output straight into the height-sharded
  layout the rotary op needs (one reshard per head tensor instead of two).
* The RoPE cos/sin gather uses a row-major ``ttnn.embedding`` whose output is
  reshaped (a view) and tilized, replacing the tiled gather plus transpose.
* ``ttnn.experimental.paged_fused_update_cache`` writes K and V in one op; V is
  moved to a core rectangle disjoint from Q/K first because the op requires it.
* The O projection reads the width-sharded concat-heads output directly.
* For a logical batch of one, the residual adds after the O and down
  projections are folded into ``ttnn.linear`` as the matmul bias.

Everything else (chunk and cache-write planning, SDPA chunk selection,
decode batch padding, the 8-wide decode core rule) is inherited unchanged.
"""

from __future__ import annotations

from typing import Any

import ttnn
from models.autoports.qwen_qwen3_8b.tt.functional_decoder import (
    TILE_SIZE,
    DecoderWeights,
    FunctionalDecoder,
    PrefillChunk,
    _decode_height_sharded_memory_config,
    _same_buffer,
    decode_physical_batch,
    prefill_sdpa_chunk_size,
)

PREFILL_MINIMAL_MATMUL_MIN_ROWS = 256


def _decode_v_update_memory_config(mesh_device: ttnn.MeshDevice, batch: int, width: int) -> ttnn.MemoryConfig:
    """Height-sharded layout for V on the core rectangle directly below the Q/K rectangle.

    ``paged_fused_update_cache`` requires its two inputs on disjoint core sets with the
    same core count, so V uses a rectangle of the same shape as the Q/K rectangle from
    ``decode_physical_batch`` shifted down by that rectangle's height.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    physical, x, y = decode_physical_batch(batch, grid.x, grid.y)
    if physical != batch:
        raise ValueError(f"decode batch {batch} must be padded to {physical} users before sharding")
    if 2 * y > grid.y:
        raise ValueError(f"decode batch {batch} needs a {x}x{2 * y} core area for the fused cache update")
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(x - 1, 2 * y - 1))])
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, width),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class FusedDecoder(FunctionalDecoder):
    """One Qwen3-8B decoder layer with the fused prefill and decode op graphs."""

    def __init__(
        self,
        *,
        geometry,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        weights: DecoderWeights,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        prefill_chunk_size: int = 4096,
        prefill_minimal_matmul_min_rows: int = PREFILL_MINIMAL_MATMUL_MIN_ROWS,
    ) -> None:
        super().__init__(
            geometry=geometry,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            weights=weights,
            activation_dtype=activation_dtype,
            prefill_chunk_size=prefill_chunk_size,
        )
        if prefill_minimal_matmul_min_rows % TILE_SIZE != 0 or prefill_minimal_matmul_min_rows < TILE_SIZE:
            raise ValueError("prefill_minimal_matmul_min_rows must be a positive multiple of 32")
        self.prefill_minimal_matmul_min_rows = prefill_minimal_matmul_min_rows
        self.ones_row = ttnn.moreh_full(
            (1, 1, 1, geometry.hidden_size),
            1.0,
            mesh_device,
            dtype=activation_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @classmethod
    def from_state_dict(cls, state_dict: dict, *, prefill_minimal_matmul_min_rows: int | None = None, **kwargs: Any):
        decoder = super().from_state_dict(state_dict, **kwargs)
        if prefill_minimal_matmul_min_rows is not None:
            decoder.prefill_minimal_matmul_min_rows = prefill_minimal_matmul_min_rows
        return decoder

    # ----------------------------------------------------------------- prefill

    def _use_minimal_matmul(self, x: ttnn.Tensor) -> bool:
        return x.shape[-2] >= self.prefill_minimal_matmul_min_rows

    def _prefill_matmul(self, x: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        if not self._use_minimal_matmul(x):
            return self._linear(x, weight)
        return ttnn.experimental.minimal_matmul(
            x,
            weight,
            dtype=self.activation_dtype,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _prefill_matmul_residual(self, x: ttnn.Tensor, weight: ttnn.Tensor, residual: ttnn.Tensor) -> ttnn.Tensor:
        """``residual + x @ weight`` with the add folded into the matmul kernel."""
        if not self._use_minimal_matmul(x):
            y = self._linear(x, weight)
            out = ttnn.add(residual, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            y.deallocate(True)
            return out
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            x,
            weight,
            1.0,
            residual,
            self.ones_row,
            dtype=self.activation_dtype,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _prefill_chunk(
        self,
        x: ttnn.Tensor,
        chunk: PrefillChunk,
        user_page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        attn_in = self._rms_norm(x, self.weights.input_ln)
        h = self._attention_prefill(attn_in, chunk, user_page_table, kv_cache, residual=x)
        attn_in.deallocate(True)
        mlp_in = self._rms_norm(h, self.weights.post_attn_ln)
        out = self._mlp_prefill(mlp_in, residual=h)
        mlp_in.deallocate(True)
        h.deallocate(True)
        return out

    def _mlp_prefill(self, x: ttnn.Tensor, residual: ttnn.Tensor) -> ttnn.Tensor:
        gate = self._prefill_matmul(x, self.weights.gate_proj)
        up = self._prefill_matmul(x, self.weights.up_proj)
        act = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate.deallocate(True)
        up.deallocate(True)
        out = self._prefill_matmul_residual(act, self.weights.down_proj, residual)
        act.deallocate(True)
        return out

    def _attention_prefill(
        self,
        x: ttnn.Tensor,
        chunk: PrefillChunk,
        user_page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        residual: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        g = self.geometry
        xqkv = self._prefill_matmul(x, self.weights.qkv)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=g.num_heads,
            num_kv_heads=g.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        q_normed = self._rms_norm(q_heads, self.weights.q_norm)
        q_heads.deallocate(True)
        k_normed = self._rms_norm(k_heads, self.weights.k_norm)
        k_heads.deallocate(True)
        cos = ttnn.slice(
            self.weights.rope_cos,
            [0, 0, chunk.phys_start, 0],
            [1, 1, chunk.phys_start + chunk.phys_len, g.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin = ttnn.slice(
            self.weights.rope_sin,
            [0, 0, chunk.phys_start, 0],
            [1, 1, chunk.phys_start + chunk.phys_len, g.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_rot = ttnn.experimental.rotary_embedding_hf(
            q_normed, cos, sin, is_decode_mode=False, compute_kernel_config=self.compute_config
        )
        q_normed.deallocate(True)
        k_rot = ttnn.experimental.rotary_embedding_hf(
            k_normed, cos, sin, is_decode_mode=False, compute_kernel_config=self.compute_config
        )
        k_normed.deallocate(True)
        cos.deallocate(True)
        sin.deallocate(True)

        key_cache, value_cache = kv_cache
        self._write_prefill_cache(key_cache, value_cache, k_rot, v_heads, chunk, user_page_table)
        k_rot.deallocate(True)
        v_heads.deallocate(True)

        kv_extent = user_page_table.shape[1] * key_cache.shape[2]
        chunk_size = prefill_sdpa_chunk_size(chunk.phys_start, chunk.phys_len, kv_extent)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.prefill_sdpa_grid,
            q_chunk_size=chunk_size,
            k_chunk_size=chunk_size,
            exp_approx_mode=False,
        )
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q_rot,
            key_cache,
            value_cache,
            user_page_table,
            chunk_start_idx=chunk.phys_start,
            scale=g.scale,
            program_config=program_config,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_rot.deallocate(True)
        concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        if residual is None:
            out = self._prefill_matmul(concat, self.weights.o_proj)
        else:
            out = self._prefill_matmul_residual(concat, self.weights.o_proj, residual)
        concat.deallocate(True)
        return out

    # ------------------------------------------------------------------ decode

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        batch = hidden_states.shape[-2]
        grid = self.mesh_device.compute_with_storage_grid_size()
        physical, _, _ = decode_physical_batch(batch, grid.x, grid.y)
        owned = []
        if physical != batch:
            extra = physical - batch
            cleaned = ttnn.fill_implicit_tile_padding(hidden_states, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if not _same_buffer(cleaned, hidden_states):
                owned.append(cleaned)
            hidden_states = ttnn.pad(cleaned, [(0, 0), (0, 0), (0, extra), (0, 0)], 0.0)
            if not _same_buffer(hidden_states, cleaned):
                owned.append(hidden_states)
            rope_pos = ttnn.pad(current_pos, [(0, extra)], 0)
            padded_pos = ttnn.pad(current_pos, [(0, extra)], -1)
            padded_table = ttnn.pad(page_table, [(0, extra), (0, 0)], 0)
            for padded, source in ((rope_pos, current_pos), (padded_pos, current_pos), (padded_table, page_table)):
                if not _same_buffer(padded, source):
                    owned.append(padded)
            current_pos = padded_pos
            page_table = padded_table
        else:
            rope_pos = current_pos
        attn_in = self._rms_norm(hidden_states, self.weights.input_ln)
        h = self._attention_decode(
            attn_in, current_pos, page_table, kv_cache, rope_pos=rope_pos, residual=hidden_states
        )
        attn_in.deallocate(True)
        mlp_in = self._rms_norm(h, self.weights.post_attn_ln)
        out = self._mlp_decode(mlp_in, residual=h)
        mlp_in.deallocate(True)
        h.deallocate(True)
        if physical != batch:
            sliced = ttnn.slice(
                out, [0, 0, 0, 0], [1, 1, batch, self.geometry.hidden_size], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            out.deallocate(True)
            out = sliced
            for tensor in owned:
                tensor.deallocate(True)
        return out

    def _decode_matmul_residual(self, x: ttnn.Tensor, weight: ttnn.Tensor, residual: ttnn.Tensor) -> ttnn.Tensor:
        """``residual + x @ weight`` for decode, sliced to the residual's rows.

        With one logical residual row the add is the matmul's row-broadcast bias; the
        padded rows of the matmul output then carry garbage and are sliced away.
        """
        rows = residual.shape[-2]
        if rows == 1:
            y = ttnn.linear(
                x,
                weight,
                bias=residual,
                dtype=self.activation_dtype,
                compute_kernel_config=self.compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if y.shape[-2] == rows:
                return y
            out = ttnn.slice(
                y, [0, 0, 0, 0], [1, 1, rows, self.geometry.hidden_size], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            y.deallocate(True)
            return out
        y = self._linear(x, weight)
        if y.shape[-2] != rows:
            sliced = ttnn.slice(
                y, [0, 0, 0, 0], [1, 1, rows, self.geometry.hidden_size], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            y.deallocate(True)
            y = sliced
        out = ttnn.add(residual, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y.deallocate(True)
        return out

    def _mlp_decode(self, x: ttnn.Tensor, residual: ttnn.Tensor) -> ttnn.Tensor:
        gate = self._linear(x, self.weights.gate_proj)
        up = self._linear(x, self.weights.up_proj)
        act = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate.deallocate(True)
        up.deallocate(True)
        out = self._decode_matmul_residual(act, self.weights.down_proj, residual)
        act.deallocate(True)
        return out

    def _decode_head_norm(self, heads: ttnn.Tensor, weight: ttnn.Tensor, heads_mem_config) -> ttnn.Tensor:
        """Per-head RMSNorm of a height-sharded heads tensor, written back height-sharded."""
        interleaved = ttnn.to_memory_config(heads, ttnn.L1_MEMORY_CONFIG, dtype=heads.dtype)
        heads.deallocate(True)
        normed = ttnn.rms_norm(
            interleaved,
            epsilon=self.eps,
            weight=weight,
            compute_kernel_config=self.compute_config,
            memory_config=heads_mem_config,
        )
        interleaved.deallocate(True)
        return normed

    def _decode_rope(self, current_pos: ttnn.Tensor, batch: int, heads_mem_config) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        g = self.geometry
        idx = ttnn.reshape(current_pos, (1, batch))
        idx = ttnn.typecast(idx, ttnn.uint32)
        outs = []
        for table in (self.weights.rope_cos_table, self.weights.rope_sin_table):
            gathered = ttnn.embedding(idx, table, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            rows = ttnn.reshape(gathered, (1, batch, 1, g.head_dim))
            tiled = ttnn.to_layout(rows, ttnn.TILE_LAYOUT)
            gathered.deallocate(True)
            sharded = ttnn.interleaved_to_sharded(tiled, heads_mem_config)
            tiled.deallocate(True)
            outs.append(sharded)
        idx.deallocate(True)
        return outs[0], outs[1]

    def _attention_decode(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        rope_pos: ttnn.Tensor | None = None,
        residual: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        g = self.geometry
        batch = x.shape[-2]
        if rope_pos is None:
            rope_pos = current_pos
        xqkv = self._linear(x, self.weights.qkv)
        heads_mem_config = _decode_height_sharded_memory_config(self.mesh_device, batch, g.head_dim)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=g.num_heads,
            num_kv_heads=g.num_kv_heads,
            memory_config=heads_mem_config,
        )
        xqkv.deallocate(True)
        q_sh = self._decode_head_norm(q_heads, self.weights.q_norm, heads_mem_config)
        k_sh = self._decode_head_norm(k_heads, self.weights.k_norm, heads_mem_config)

        cos, sin = self._decode_rope(rope_pos, batch, heads_mem_config)
        q_rot = ttnn.experimental.rotary_embedding_hf(
            q_sh, cos, sin, is_decode_mode=True, compute_kernel_config=self.compute_config
        )
        q_sh.deallocate(True)
        k_rot = ttnn.experimental.rotary_embedding_hf(
            k_sh, cos, sin, is_decode_mode=True, compute_kernel_config=self.compute_config
        )
        k_sh.deallocate(True)
        cos.deallocate(True)
        sin.deallocate(True)

        key_cache, value_cache = kv_cache
        v_moved = ttnn.to_memory_config(v_heads, _decode_v_update_memory_config(self.mesh_device, batch, g.head_dim))
        v_heads.deallocate(True)
        ttnn.experimental.paged_fused_update_cache(
            key_cache, k_rot, value_cache, v_moved, update_idxs_tensor=current_pos, page_table=page_table
        )
        k_rot.deallocate(True)
        v_moved.deallocate(True)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_rot,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=g.scale,
            program_config=self._decode_sdpa_program_config(page_table.shape[1] * key_cache.shape[2]),
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_rot.deallocate(True)
        attn_sh = ttnn.to_memory_config(attn, heads_mem_config, dtype=attn.dtype)
        attn.deallocate(True)
        concat = ttnn.experimental.nlp_concat_heads_decode(attn_sh, num_heads=g.num_heads)
        attn_sh.deallocate(True)
        concat_dram = ttnn.sharded_to_interleaved(concat, ttnn.DRAM_MEMORY_CONFIG)
        concat.deallocate(True)
        if residual is None:
            out = self._linear(concat_dram, self.weights.o_proj)
            concat_dram.deallocate(True)
            if out.shape[-2] != batch:
                sliced = ttnn.slice(
                    out, [0, 0, 0, 0], [1, 1, batch, g.hidden_size], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                out.deallocate(True)
                out = sliced
            return out
        out = self._decode_matmul_residual(concat_dram, self.weights.o_proj, residual)
        concat_dram.deallocate(True)
        return out

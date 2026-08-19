# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional single-device decoder layer for Qwen/Qwen3.6-27B.

The checkpoint uses the Transformers Qwen3.5 text architecture.  It has two
meaningful layer kinds:

* ``linear_attention`` is a Gated DeltaNet layer.  Its cache is a caller-owned
  ``(conv_state, recurrent_state)`` pair.  Prefill and decode update both
  tensors in place.  This state is request/slot state, not a paged KV cache.
* ``full_attention`` is gated GQA.  It uses a caller-owned paged
  ``(key_cache, value_cache)`` pair plus a device page table.  Decode also
  consumes a device ``current_positions`` tensor and is trace safe.

Public tensor shapes are deliberately the standard TT LLM shapes:

* prefill hidden states: ``[batch, 1, sequence, 5120]``;
* decode hidden states: ``[1, 1, padded_batch, 5120]``;
* full-attention paged cache: ``[num_blocks, 4, block_size, 256]``;
* linear convolution state: ``[batch, 1, 4, 10240]``;
* linear recurrent state: ``[batch, 48, 128, 128]``.

Weights and fixed constants are converted in :meth:`from_state_dict`.  The
forward methods contain TTNN operations only: no Torch conversion or host
fallback is used inside a pass.
"""

from __future__ import annotations

import math

import ttnn

from models.common.lightweightmodule import LightweightModule


TILE_SIZE = 32
LINEAR_CHUNK_SIZE = 64
FULL_PREFILL_SDPA_LIMIT = 32768
FULL_PREFILL_CHUNK_SIZE = 4096
FULL_PREFILL_Q_CHUNK_SIZE = 128


def _round_up(value: int, multiple: int = TILE_SIZE) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _state_key(state_dict, layer_idx: int, suffix: str) -> str:
    candidates = (
        f"model.language_model.layers.{layer_idx}.{suffix}",
        f"language_model.layers.{layer_idx}.{suffix}",
        f"model.layers.{layer_idx}.{suffix}",
        f"layers.{layer_idx}.{suffix}",
        suffix,
    )
    for key in candidates:
        if key in state_dict:
            return key
    raise KeyError(f"Missing layer {layer_idx} tensor {suffix!r}; tried {candidates}")


def _as_weight(
    tensor, mesh_device, *, transpose: bool = False, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
):
    if transpose:
        tensor = tensor.transpose(-2, -1).contiguous()
    return ttnn.as_tensor(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _linear_weight(state_dict, layer_idx, suffix, mesh_device):
    return _as_weight(state_dict[_state_key(state_dict, layer_idx, suffix)], mesh_device, transpose=True)


def _norm_weight(state_dict, layer_idx, suffix, mesh_device, *, offset: bool):
    source = state_dict[_state_key(state_dict, layer_idx, suffix)].float()
    if offset:
        source = source + 1.0
    source = source.reshape(1, 1, 1, -1)
    return _as_weight(source, mesh_device)


def _slice_last(x, start: int, end: int):
    shape = list(x.shape)
    begin = [0] * len(shape)
    finish = shape.copy()
    begin[-1] = start
    finish[-1] = end
    return ttnn.slice(x, begin, finish)


def _rotate_half_partial(x, cos, sin, rotary_dim: int):
    half = rotary_dim // 2
    x_rotary = _slice_last(x, 0, rotary_dim)
    x_first = _slice_last(x, 0, half)
    x_second = _slice_last(x, half, rotary_dim)
    x_pass = _slice_last(x, rotary_dim, x.shape[-1])
    rotated = ttnn.concat([ttnn.neg(x_second), x_first], dim=-1)
    embedded = ttnn.add(ttnn.multiply(x_rotary, cos), ttnn.multiply(rotated, sin))
    return ttnn.concat([embedded, x_pass], dim=-1)


class FunctionalDecoder(LightweightModule):
    """Qwen3.6 decoder layer with TTNN-only prefill and decode paths."""

    def __init__(self, **entries):
        self.__dict__.update(entries)

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        page_block_size: int = 64,
        **kwargs,
    ):
        """Load one real-shape Qwen3.6 text decoder layer onto a 1x1 mesh.

        ``hf_config`` may be the outer multimodal config or its text config.
        Runtime caches are intentionally caller-owned; use
        :meth:`allocate_paged_kv_cache` or :meth:`allocate_linear_state`.
        """
        config = getattr(hf_config, "text_config", hf_config)
        if mesh_device.get_num_devices() != 1:
            raise ValueError("FunctionalDecoder is a single-device (1x1 mesh) stage")
        layer_kind = config.layer_types[layer_idx]
        if layer_kind not in ("linear_attention", "full_attention"):
            raise ValueError(f"Unsupported Qwen3.6 layer kind: {layer_kind}")
        if (
            config.hidden_size != 5120
            or config.intermediate_size != 17408
            or config.num_attention_heads != 24
            or config.num_key_value_heads != 4
            or config.head_dim != 256
        ):
            raise ValueError("FunctionalDecoder expects the real Qwen/Qwen3.6-27B text configuration")

        common = dict(
            mesh_device=mesh_device,
            config=config,
            layer_idx=layer_idx,
            layer_kind=layer_kind,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            rms_norm_eps=config.rms_norm_eps,
            max_context=config.max_position_embeddings,
            page_block_size=page_block_size,
            full_prefill_sdpa_limit=FULL_PREFILL_SDPA_LIMIT,
            full_prefill_chunk_size=FULL_PREFILL_CHUNK_SIZE,
            full_prefill_q_chunk_size=FULL_PREFILL_Q_CHUNK_SIZE,
            input_norm=_norm_weight(
                state_dict, layer_idx, "input_layernorm.weight", mesh_device, offset=True
            ),
            post_attention_norm=_norm_weight(
                state_dict, layer_idx, "post_attention_layernorm.weight", mesh_device, offset=True
            ),
            mlp_gate=_linear_weight(state_dict, layer_idx, "mlp.gate_proj.weight", mesh_device),
            mlp_up=_linear_weight(state_dict, layer_idx, "mlp.up_proj.weight", mesh_device),
            mlp_down=_linear_weight(state_dict, layer_idx, "mlp.down_proj.weight", mesh_device),
        )

        if layer_kind == "full_attention":
            q_norm = _norm_weight(state_dict, layer_idx, "self_attn.q_norm.weight", mesh_device, offset=True)
            k_norm = _norm_weight(state_dict, layer_idx, "self_attn.k_norm.weight", mesh_device, offset=True)
            common.update(
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                rotary_dim=int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 0.25)),
                attention_scale=config.head_dim**-0.5,
                q_proj=_linear_weight(state_dict, layer_idx, "self_attn.q_proj.weight", mesh_device),
                k_proj=_linear_weight(state_dict, layer_idx, "self_attn.k_proj.weight", mesh_device),
                v_proj=_linear_weight(state_dict, layer_idx, "self_attn.v_proj.weight", mesh_device),
                o_proj=_linear_weight(state_dict, layer_idx, "self_attn.o_proj.weight", mesh_device),
                q_norm=ttnn.reshape(q_norm, [1, 1, 1, config.head_dim]),
                k_norm=ttnn.reshape(k_norm, [1, 1, 1, config.head_dim]),
            )
        else:
            conv_dim = (
                2 * config.linear_num_key_heads * config.linear_key_head_dim
                + config.linear_num_value_heads * config.linear_value_head_dim
            )
            conv = state_dict[_state_key(state_dict, layer_idx, "linear_attn.conv1d.weight")]
            conv = conv[:, 0, :].transpose(0, 1).contiguous().reshape(1, 1, config.linear_conv_kernel_dim, conv_dim)
            a_log = state_dict[_state_key(state_dict, layer_idx, "linear_attn.A_log")]
            a_decay = (-a_log.float().exp()).reshape(1, config.linear_num_value_heads, 1, 1)
            dt_bias = state_dict[_state_key(state_dict, layer_idx, "linear_attn.dt_bias")].float().reshape(
                1, config.linear_num_value_heads, 1, 1
            )
            chunk_ones = a_log.float().new_ones((1, 1, LINEAR_CHUNK_SIZE, LINEAR_CHUNK_SIZE))
            chunk_lower = chunk_ones.tril()
            chunk_strict_lower = chunk_ones.tril(diagonal=-1)
            common.update(
                linear_num_key_heads=config.linear_num_key_heads,
                linear_num_value_heads=config.linear_num_value_heads,
                linear_key_head_dim=config.linear_key_head_dim,
                linear_value_head_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                conv_dim=conv_dim,
                linear_chunk_size=LINEAR_CHUNK_SIZE,
                in_proj_qkv=_linear_weight(state_dict, layer_idx, "linear_attn.in_proj_qkv.weight", mesh_device),
                in_proj_z=_linear_weight(state_dict, layer_idx, "linear_attn.in_proj_z.weight", mesh_device),
                in_proj_b=_linear_weight(state_dict, layer_idx, "linear_attn.in_proj_b.weight", mesh_device),
                in_proj_a=_linear_weight(state_dict, layer_idx, "linear_attn.in_proj_a.weight", mesh_device),
                linear_out_proj=_linear_weight(state_dict, layer_idx, "linear_attn.out_proj.weight", mesh_device),
                conv_weight=_as_weight(conv, mesh_device),
                a_decay=_as_weight(a_decay, mesh_device, dtype=ttnn.float32),
                dt_bias=_as_weight(dt_bias, mesh_device, dtype=ttnn.float32),
                linear_chunk_lower=_as_weight(chunk_lower, mesh_device, dtype=ttnn.float32),
                linear_chunk_strict_lower=_as_weight(
                    chunk_strict_lower, mesh_device, dtype=ttnn.float32
                ),
                linear_chunk_identity=_as_weight(
                    chunk_lower - chunk_strict_lower, mesh_device, dtype=ttnn.float32
                ),
                linear_norm=_norm_weight(
                    state_dict, layer_idx, "linear_attn.norm.weight", mesh_device, offset=False
                ),
            )
        return cls(**common)

    def allocate_paged_kv_cache(self, *, num_blocks: int, dtype=ttnn.bfloat16):
        if self.layer_kind != "full_attention":
            raise ValueError("Paged KV cache is only meaningful for full_attention layers")
        shape = [num_blocks, self.num_kv_heads, self.page_block_size, self.head_dim]
        return (
            ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device),
            ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device),
        )

    def allocate_linear_state(self, *, batch_size: int, dtype=ttnn.bfloat16):
        if self.layer_kind != "linear_attention":
            raise ValueError("DeltaNet state is only meaningful for linear_attention layers")
        return (
            ttnn.zeros(
                [batch_size, 1, self.conv_kernel_size, self.conv_dim],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
            ),
            ttnn.zeros(
                [batch_size, self.linear_num_value_heads, self.linear_key_head_dim, self.linear_value_head_dim],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
            ),
        )

    def _mlp(self, hidden_states):
        gate = ttnn.linear(hidden_states, self.mlp_gate, dtype=ttnn.bfloat16)
        up = ttnn.linear(hidden_states, self.mlp_up, dtype=ttnn.bfloat16)
        activated = ttnn.multiply(ttnn.silu(gate), up)
        return ttnn.linear(activated, self.mlp_down, dtype=ttnn.bfloat16)

    def _finish_layer(self, residual, mixed):
        hidden_states = ttnn.add(residual, mixed)
        mlp_input = ttnn.rms_norm(
            hidden_states, epsilon=self.rms_norm_eps, weight=self.post_attention_norm
        )
        return ttnn.add(hidden_states, self._mlp(mlp_input))

    def _finish_layer_chunked(self, residual, mixed, *, logical_seq_len: int):
        """Apply the residual/MLP tail without materializing full-context MLP intermediates."""
        outputs = []
        for start in range(0, logical_seq_len, self.linear_chunk_size):
            end = min(start + self.linear_chunk_size, logical_seq_len)
            residual_chunk = ttnn.slice(
                residual,
                [0, 0, start, 0],
                [residual.shape[0], 1, end, self.hidden_size],
            )
            mixed_chunk = ttnn.slice(
                mixed,
                [0, 0, start, 0],
                [mixed.shape[0], 1, end, self.hidden_size],
            )
            outputs.append(self._finish_layer(residual_chunk, mixed_chunk))
        if len(outputs) == 1:
            return outputs[0]
        result = ttnn.concat(outputs, dim=2)
        for output in outputs:
            ttnn.deallocate(output)
        return result

    def _full_qkv_prefill(self, hidden_states, cos, sin):
        batch, _, seq_len, _ = hidden_states.shape
        q_and_gate = ttnn.linear(hidden_states, self.q_proj, dtype=ttnn.bfloat16)
        q_and_gate = ttnn.reshape(q_and_gate, [batch, seq_len, self.num_heads, self.head_dim * 2])
        query = _slice_last(q_and_gate, 0, self.head_dim)
        gate = _slice_last(q_and_gate, self.head_dim, self.head_dim * 2)
        gate = ttnn.reshape(gate, [batch, 1, seq_len, self.num_heads * self.head_dim])
        query = ttnn.permute(query, [0, 2, 1, 3])

        key = ttnn.linear(hidden_states, self.k_proj, dtype=ttnn.bfloat16)
        value = ttnn.linear(hidden_states, self.v_proj, dtype=ttnn.bfloat16)
        key = ttnn.permute(ttnn.reshape(key, [batch, seq_len, self.num_kv_heads, self.head_dim]), [0, 2, 1, 3])
        value = ttnn.permute(
            ttnn.reshape(value, [batch, seq_len, self.num_kv_heads, self.head_dim]), [0, 2, 1, 3]
        )
        query = ttnn.rms_norm(query, epsilon=self.rms_norm_eps, weight=self.q_norm)
        key = ttnn.rms_norm(key, epsilon=self.rms_norm_eps, weight=self.k_norm)
        query = _rotate_half_partial(query, cos, sin, self.rotary_dim)
        key = _rotate_half_partial(key, cos, sin, self.rotary_dim)
        return query, key, value, gate

    def _full_qkv_decode(self, hidden_states, cos, sin):
        _, _, padded_batch, _ = hidden_states.shape
        q_and_gate = ttnn.linear(hidden_states, self.q_proj, dtype=ttnn.bfloat16)
        q_and_gate = ttnn.reshape(q_and_gate, [1, padded_batch, self.num_heads, self.head_dim * 2])
        query = _slice_last(q_and_gate, 0, self.head_dim)
        gate = _slice_last(q_and_gate, self.head_dim, self.head_dim * 2)
        gate = ttnn.reshape(gate, [1, 1, padded_batch, self.num_heads * self.head_dim])

        key = ttnn.linear(hidden_states, self.k_proj, dtype=ttnn.bfloat16)
        value = ttnn.linear(hidden_states, self.v_proj, dtype=ttnn.bfloat16)
        key = ttnn.reshape(key, [1, padded_batch, self.num_kv_heads, self.head_dim])
        value = ttnn.reshape(value, [1, padded_batch, self.num_kv_heads, self.head_dim])
        query = ttnn.rms_norm(query, epsilon=self.rms_norm_eps, weight=self.q_norm)
        key = ttnn.rms_norm(key, epsilon=self.rms_norm_eps, weight=self.k_norm)
        query = _rotate_half_partial(query, cos, sin, self.rotary_dim)
        key = _rotate_half_partial(key, cos, sin, self.rotary_dim)
        return query, key, value, gate

    def _full_prefill(
        self,
        hidden_states,
        *,
        cos,
        sin,
        page_table,
        kv_cache,
        logical_seq_len: int,
        cache_batch_idx: int = 0,
    ):
        query, key, value, gate = self._full_qkv_prefill(hidden_states, cos, sin)
        key_cache, value_cache = kv_cache
        batch = hidden_states.shape[0]
        for batch_idx in range(batch):
            key_user = ttnn.slice(key, [batch_idx, 0, 0, 0], [batch_idx + 1, key.shape[1], key.shape[2], key.shape[3]])
            value_user = ttnn.slice(
                value, [batch_idx, 0, 0, 0], [batch_idx + 1, value.shape[1], value.shape[2], value.shape[3]]
            )
            ttnn.experimental.paged_fill_cache(
                key_cache, key_user, page_table, batch_idx=cache_batch_idx + batch_idx
            )
            ttnn.experimental.paged_fill_cache(
                value_cache, value_user, page_table, batch_idx=cache_batch_idx + batch_idx
            )

        attention = ttnn.transformer.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=self.attention_scale
        )
        attention = ttnn.permute(attention, [0, 2, 1, 3])
        attention = ttnn.reshape(attention, [batch, 1, hidden_states.shape[2], self.num_heads * self.head_dim])
        attention = ttnn.multiply(attention, ttnn.sigmoid(gate))
        output = ttnn.linear(attention, self.o_proj, dtype=ttnn.bfloat16)
        if logical_seq_len != hidden_states.shape[2]:
            output = ttnn.slice(output, [0, 0, 0, 0], [batch, 1, logical_seq_len, self.hidden_size])
        return output

    def _full_prefill_chunked_layer(
        self,
        hidden_states,
        *,
        cos,
        sin,
        page_table,
        kv_cache,
        logical_seq_len: int,
    ):
        """Run long paged prefill without quadratic/full-MLP intermediates.

        Each 4096-token chunk is projected and written into its corresponding
        page-table slice.  Chunked SDPA reads the complete logical prefix from
        the paged cache, while the residual and MLP tail are also completed per
        chunk before advancing.  The public output is sliced to the logical
        sequence length even when the final physical tile contains padding.
        """
        batch = hidden_states.shape[0]
        physical_seq_len = hidden_states.shape[2]
        key_cache, value_cache = kv_cache
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=self.full_prefill_q_chunk_size,
            k_chunk_size=self.full_prefill_q_chunk_size,
            exp_approx_mode=False,
        )
        outputs = []

        for chunk_start in range(0, physical_seq_len, self.full_prefill_chunk_size):
            chunk_end = min(chunk_start + self.full_prefill_chunk_size, physical_seq_len)
            chunk_len = chunk_end - chunk_start
            logical_chunk_len = min(chunk_len, max(0, logical_seq_len - chunk_start))
            residual = ttnn.slice(
                hidden_states,
                [0, 0, chunk_start, 0],
                [batch, 1, chunk_end, self.hidden_size],
            )
            normalized = ttnn.rms_norm(
                residual, epsilon=self.rms_norm_eps, weight=self.input_norm
            )
            cos_chunk = ttnn.slice(
                cos,
                [0, 0, chunk_start, 0],
                [cos.shape[0], cos.shape[1], chunk_end, cos.shape[3]],
            )
            sin_chunk = ttnn.slice(
                sin,
                [0, 0, chunk_start, 0],
                [sin.shape[0], sin.shape[1], chunk_end, sin.shape[3]],
            )
            query, key, value, gate = self._full_qkv_prefill(normalized, cos_chunk, sin_chunk)

            first_page = chunk_start // self.page_block_size
            chunk_pages = (chunk_len + self.page_block_size - 1) // self.page_block_size
            fill_page_table = ttnn.slice(
                page_table,
                [0, first_page],
                [page_table.shape[0], first_page + chunk_pages],
            )
            for batch_idx in range(batch):
                key_user = ttnn.slice(
                    key,
                    [batch_idx, 0, 0, 0],
                    [batch_idx + 1, key.shape[1], key.shape[2], key.shape[3]],
                )
                value_user = ttnn.slice(
                    value,
                    [batch_idx, 0, 0, 0],
                    [batch_idx + 1, value.shape[1], value.shape[2], value.shape[3]],
                )
                ttnn.experimental.paged_fill_cache(
                    key_cache, key_user, fill_page_table, batch_idx=batch_idx
                )
                ttnn.experimental.paged_fill_cache(
                    value_cache, value_user, fill_page_table, batch_idx=batch_idx
                )
                ttnn.deallocate(key_user)
                ttnn.deallocate(value_user)
            ttnn.deallocate(fill_page_table)
            ttnn.deallocate(key)
            ttnn.deallocate(value)
            ttnn.deallocate(normalized)
            ttnn.deallocate(cos_chunk)
            ttnn.deallocate(sin_chunk)

            query_pad = (-chunk_len) % self.full_prefill_q_chunk_size
            if query_pad:
                query_unpadded = query
                query = ttnn.pad(
                    query_unpadded,
                    padding=[(0, 0), (0, 0), (0, query_pad), (0, 0)],
                    value=0.0,
                )
                ttnn.deallocate(query_unpadded)
            attention = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=query,
                input_tensor_k=key_cache,
                input_tensor_v=value_cache,
                page_table_tensor=page_table,
                chunk_start_idx=chunk_start,
                scale=self.attention_scale,
                program_config=program_config,
            )
            ttnn.deallocate(query)
            if query_pad:
                padded_attention = attention
                attention = ttnn.slice(
                    padded_attention,
                    [0, 0, 0, 0],
                    [batch, self.num_heads, chunk_len, self.head_dim],
                )
                ttnn.deallocate(padded_attention)

            attention = ttnn.permute(attention, [0, 2, 1, 3])
            attention = ttnn.reshape(
                attention, [batch, 1, chunk_len, self.num_heads * self.head_dim]
            )
            gated_attention = ttnn.multiply(attention, ttnn.sigmoid(gate))
            mixed = ttnn.linear(gated_attention, self.o_proj, dtype=ttnn.bfloat16)
            ttnn.deallocate(attention)
            ttnn.deallocate(gated_attention)
            ttnn.deallocate(gate)

            if logical_chunk_len:
                if logical_chunk_len != chunk_len:
                    logical_residual = ttnn.slice(
                        residual,
                        [0, 0, 0, 0],
                        [batch, 1, logical_chunk_len, self.hidden_size],
                    )
                    logical_mixed = ttnn.slice(
                        mixed,
                        [0, 0, 0, 0],
                        [batch, 1, logical_chunk_len, self.hidden_size],
                    )
                    ttnn.deallocate(residual)
                    ttnn.deallocate(mixed)
                    residual = logical_residual
                    mixed = logical_mixed
                output = self._finish_layer(residual, mixed)
                ttnn.deallocate(residual)
                ttnn.deallocate(mixed)
                outputs.append(output)
            else:
                ttnn.deallocate(residual)
                ttnn.deallocate(mixed)

        if len(outputs) == 1:
            return outputs[0]
        result = ttnn.concat(outputs, dim=2)
        for output in outputs:
            ttnn.deallocate(output)
        return result

    def _full_decode(self, hidden_states, *, cos, sin, page_table, current_positions, kv_cache):
        padded_batch = hidden_states.shape[2]
        query, key, value, gate = self._full_qkv_decode(hidden_states, cos, sin)
        batch_grid = ttnn.num_cores_to_corerangeset(
            padded_batch, ttnn.CoreCoord(8, 8), row_wise=True
        )
        decode_head_memcfg = ttnn.create_sharded_memory_config(
            shape=(TILE_SIZE, self.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        query = ttnn.to_memory_config(query, decode_head_memcfg)
        key = ttnn.to_memory_config(key, decode_head_memcfg)
        value = ttnn.to_memory_config(value, decode_head_memcfg)
        key_cache, value_cache = kv_cache
        ttnn.experimental.paged_update_cache(
            key_cache, key, update_idxs_tensor=current_positions, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            value_cache, value, update_idxs_tensor=current_positions, page_table=page_table
        )
        attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            cur_pos_tensor=current_positions,
            page_table_tensor=page_table,
            scale=self.attention_scale,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=0,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.reshape(attention, [1, 1, hidden_states.shape[2], self.num_heads * self.head_dim])
        attention = ttnn.multiply(attention, ttnn.sigmoid(gate))
        return ttnn.linear(attention, self.o_proj, dtype=ttnn.bfloat16)

    def _repeat_linear_qk(self, tensor):
        batch = tensor.shape[0]
        tensor = ttnn.reshape(tensor, [batch, self.linear_num_key_heads, 1, 1, self.linear_key_head_dim])
        tensor = ttnn.concat([tensor, tensor, tensor], dim=3)
        return ttnn.reshape(
            tensor, [batch, self.linear_num_value_heads, 1, self.linear_key_head_dim]
        )

    def _repeat_linear_qk_chunk(self, tensor, *, sequence_length: int):
        """Repeat each of the 16 Q/K heads three times, matching HF repeat_interleave."""
        batch = tensor.shape[0]
        tensor = ttnn.reshape(
            tensor,
            [batch, sequence_length, self.linear_num_key_heads, 1, self.linear_key_head_dim],
        )
        tensor = ttnn.concat([tensor, tensor, tensor], dim=3)
        tensor = ttnn.reshape(
            tensor,
            [batch, sequence_length, self.linear_num_value_heads, self.linear_key_head_dim],
        )
        return ttnn.permute(tensor, [0, 2, 1, 3])

    def _pad_linear_chunk(self, tensor, *, valid_tokens: int, sequence_dim: int):
        if valid_tokens == self.linear_chunk_size:
            return tensor
        padding = [(0, 0)] * len(tensor.shape)
        padding[sequence_dim] = (0, self.linear_chunk_size - valid_tokens)
        return ttnn.pad(tensor, padding=padding, value=0.0)

    def _linear_causal_conv_chunk(self, mixed_qkv, *, conv_state, valid_tokens: int):
        """Vectorized depthwise causal convolution for one logical chunk.

        The four cached projection rows precede the chunk.  Dropping the first
        convolution window gives exactly HF's cached causal-conv alignment.
        """
        batch = mixed_qkv.shape[0]
        context = ttnn.concat([conv_state, mixed_qkv], dim=2)
        new_conv_state = ttnn.slice(
            context,
            [0, 0, valid_tokens, 0],
            [batch, 1, valid_tokens + self.conv_kernel_size, self.conv_dim],
        )

        terms = []
        for kernel_idx in range(self.conv_kernel_size):
            window = ttnn.slice(
                context,
                [0, 0, kernel_idx + 1, 0],
                [batch, 1, kernel_idx + 1 + valid_tokens, self.conv_dim],
            )
            weight = ttnn.slice(
                self.conv_weight,
                [0, 0, kernel_idx, 0],
                [1, 1, kernel_idx + 1, self.conv_dim],
            )
            terms.append(ttnn.multiply(window, weight))
        convolved = terms[0]
        for term in terms[1:]:
            convolved = ttnn.add(convolved, term)
        ttnn.copy(new_conv_state, conv_state)
        return ttnn.silu(convolved)

    def _linear_chunk_inverse(self, base_attention):
        """Evaluate HF's triangular recurrence with five matrix-squaring steps.

        If ``B`` is the strict-lower base attention, HF's row recurrence is
        ``I + B + ... + B**63``.  Because ``B**64 == 0``, the factorization
        ``(I+B)(I+B**2)...(I+B**32)`` is exact and avoids 63 row dispatches.
        """
        power = base_attention
        inverse = ttnn.add(self.linear_chunk_identity, power)
        for _ in range(1, int(math.log2(self.linear_chunk_size))):
            power = ttnn.matmul(power, power, dtype=ttnn.float32)
            inverse = ttnn.matmul(
                inverse,
                ttnn.add(self.linear_chunk_identity, power),
                dtype=ttnn.float32,
            )
        return inverse

    def _linear_gated_delta_chunk(self, query, key, value, beta, g, *, recurrent_state):
        """TTNN translation of the HF 64-token chunk gated-delta rule."""
        query = ttnn.typecast(query, ttnn.float32)
        key = ttnn.typecast(key, ttnn.float32)
        value = ttnn.typecast(value, ttnn.float32)
        beta = ttnn.typecast(beta, ttnn.float32)
        g = ttnn.typecast(g, ttnn.float32)

        query = ttnn.multiply(query, 1.0 / math.sqrt(self.linear_key_head_dim))
        cumulative_g = ttnn.cumsum(g, dim=2, dtype=ttnn.float32)
        decay_exponent = ttnn.subtract(cumulative_g, ttnn.transpose(cumulative_g, -2, -1))
        # Mask before exp: the upper triangle contains large positive values.
        # HF applies tril() first; exp-then-mask would create inf*0 -> NaN.
        decay_exponent = ttnn.multiply(decay_exponent, self.linear_chunk_lower)
        decay = ttnn.exp(decay_exponent)
        decay = ttnn.multiply(decay, self.linear_chunk_lower)

        value_beta = ttnn.multiply(value, beta)
        key_beta = ttnn.multiply(key, beta)
        base_attention = ttnn.matmul(key_beta, ttnn.transpose(key, -2, -1), dtype=ttnn.float32)
        base_attention = ttnn.multiply(ttnn.neg(base_attention), decay)
        base_attention = ttnn.multiply(base_attention, self.linear_chunk_strict_lower)
        inverse_attention = self._linear_chunk_inverse(base_attention)

        transformed_value = ttnn.matmul(inverse_attention, value_beta, dtype=ttnn.float32)
        cumulative_exp = ttnn.exp(cumulative_g)
        cumulative_key = ttnn.matmul(
            inverse_attention,
            ttnn.multiply(key_beta, cumulative_exp),
            dtype=ttnn.float32,
        )
        state_projection = ttnn.matmul(cumulative_key, recurrent_state, dtype=ttnn.float32)
        adjusted_value = ttnn.subtract(transformed_value, state_projection)

        within_chunk = ttnn.matmul(query, ttnn.transpose(key, -2, -1), dtype=ttnn.float32)
        within_chunk = ttnn.multiply(ttnn.multiply(within_chunk, decay), self.linear_chunk_lower)
        from_initial_state = ttnn.matmul(
            ttnn.multiply(query, cumulative_exp), recurrent_state, dtype=ttnn.float32
        )
        core = ttnn.add(
            from_initial_state,
            ttnn.matmul(within_chunk, adjusted_value, dtype=ttnn.float32),
        )

        final_g = ttnn.slice(
            cumulative_g,
            [0, 0, self.linear_chunk_size - 1, 0],
            [cumulative_g.shape[0], cumulative_g.shape[1], self.linear_chunk_size, 1],
        )
        state_key = ttnn.multiply(key, ttnn.exp(ttnn.subtract(final_g, cumulative_g)))
        state_update = ttnn.matmul(
            ttnn.transpose(state_key, -2, -1), adjusted_value, dtype=ttnn.float32
        )
        new_recurrent_state = ttnn.add(
            ttnn.multiply(recurrent_state, ttnn.exp(final_g)), state_update
        )
        ttnn.copy(new_recurrent_state, recurrent_state)
        return ttnn.typecast(core, ttnn.bfloat16)

    def _linear_chunk(self, hidden_states, *, conv_state, recurrent_state, valid_tokens: int):
        batch = hidden_states.shape[0]
        mixed_qkv = ttnn.linear(hidden_states, self.in_proj_qkv, dtype=ttnn.bfloat16)
        z = ttnn.linear(hidden_states, self.in_proj_z, dtype=ttnn.bfloat16)
        beta = ttnn.sigmoid(ttnn.linear(hidden_states, self.in_proj_b, dtype=ttnn.bfloat16))
        a = ttnn.typecast(ttnn.linear(hidden_states, self.in_proj_a, dtype=ttnn.bfloat16), ttnn.float32)

        convolved = self._linear_causal_conv_chunk(
            mixed_qkv, conv_state=conv_state, valid_tokens=valid_tokens
        )
        key_width = self.linear_num_key_heads * self.linear_key_head_dim
        query = _slice_last(convolved, 0, key_width)
        key = _slice_last(convolved, key_width, key_width * 2)
        value = _slice_last(convolved, key_width * 2, self.conv_dim)

        query = ttnn.reshape(
            query, [batch, valid_tokens, self.linear_num_key_heads, self.linear_key_head_dim]
        )
        key = ttnn.reshape(
            key, [batch, valid_tokens, self.linear_num_key_heads, self.linear_key_head_dim]
        )
        query = self._repeat_linear_qk_chunk(query, sequence_length=valid_tokens)
        key = self._repeat_linear_qk_chunk(key, sequence_length=valid_tokens)
        value = ttnn.permute(
            ttnn.reshape(
                value,
                [batch, valid_tokens, self.linear_num_value_heads, self.linear_value_head_dim],
            ),
            [0, 2, 1, 3],
        )

        query_norm = ttnn.rsqrt(
            ttnn.add(ttnn.sum(ttnn.multiply(query, query), dim=-1, keepdim=True), 1e-6)
        )
        key_norm = ttnn.rsqrt(
            ttnn.add(ttnn.sum(ttnn.multiply(key, key), dim=-1, keepdim=True), 1e-6)
        )
        query = ttnn.multiply(query, query_norm)
        key = ttnn.multiply(key, key_norm)
        query = self._pad_linear_chunk(query, valid_tokens=valid_tokens, sequence_dim=2)
        key = self._pad_linear_chunk(key, valid_tokens=valid_tokens, sequence_dim=2)
        value = self._pad_linear_chunk(value, valid_tokens=valid_tokens, sequence_dim=2)

        beta = ttnn.permute(
            ttnn.reshape(beta, [batch, valid_tokens, self.linear_num_value_heads, 1]),
            [0, 2, 1, 3],
        )
        a = ttnn.permute(
            ttnn.reshape(a, [batch, valid_tokens, self.linear_num_value_heads, 1]),
            [0, 2, 1, 3],
        )
        g = ttnn.multiply(self.a_decay, ttnn.softplus(ttnn.add(a, self.dt_bias)))
        beta = self._pad_linear_chunk(beta, valid_tokens=valid_tokens, sequence_dim=2)
        g = self._pad_linear_chunk(g, valid_tokens=valid_tokens, sequence_dim=2)
        core = self._linear_gated_delta_chunk(
            query, key, value, beta, g, recurrent_state=recurrent_state
        )

        core = ttnn.permute(core, [0, 2, 1, 3])
        core = ttnn.reshape(
            core,
            [batch, 1, self.linear_chunk_size, self.linear_num_value_heads, self.linear_value_head_dim],
        )
        z = self._pad_linear_chunk(z, valid_tokens=valid_tokens, sequence_dim=2)
        z = ttnn.reshape(
            z,
            [batch, 1, self.linear_chunk_size, self.linear_num_value_heads, self.linear_value_head_dim],
        )
        norm = ttnn.rsqrt(
            ttnn.add(ttnn.mean(ttnn.multiply(core, core), dim=-1, keepdim=True), self.rms_norm_eps)
        )
        core = ttnn.multiply(ttnn.multiply(core, norm), self.linear_norm)
        core = ttnn.multiply(core, ttnn.silu(z))
        core = ttnn.reshape(
            core,
            [batch, 1, self.linear_chunk_size, self.linear_num_value_heads * self.linear_value_head_dim],
        )
        output = ttnn.linear(core, self.linear_out_proj, dtype=ttnn.bfloat16)
        if valid_tokens != self.linear_chunk_size:
            output = ttnn.slice(
                output, [0, 0, 0, 0], [batch, 1, valid_tokens, self.hidden_size]
            )
        return output

    def _linear_token(self, hidden_states, *, conv_state, recurrent_state):
        batch = hidden_states.shape[0]
        mixed_qkv = ttnn.linear(hidden_states, self.in_proj_qkv, dtype=ttnn.bfloat16)
        z = ttnn.linear(hidden_states, self.in_proj_z, dtype=ttnn.bfloat16)
        beta = ttnn.sigmoid(ttnn.linear(hidden_states, self.in_proj_b, dtype=ttnn.bfloat16))
        a = ttnn.linear(hidden_states, self.in_proj_a, dtype=ttnn.bfloat16)

        old_tail = ttnn.slice(
            conv_state,
            [0, 0, 1, 0],
            [batch, 1, self.conv_kernel_size, self.conv_dim],
        )
        mixed_row = ttnn.reshape(mixed_qkv, [batch, 1, 1, self.conv_dim])
        new_conv_state = ttnn.concat([old_tail, mixed_row], dim=2)
        ttnn.copy(new_conv_state, conv_state)
        convolved = ttnn.sum(ttnn.multiply(new_conv_state, self.conv_weight), dim=2, keepdim=True)
        convolved = ttnn.silu(convolved)

        query = _slice_last(convolved, 0, self.linear_num_key_heads * self.linear_key_head_dim)
        key_start = self.linear_num_key_heads * self.linear_key_head_dim
        key = _slice_last(convolved, key_start, key_start * 2)
        value = _slice_last(convolved, key_start * 2, self.conv_dim)
        query = self._repeat_linear_qk(ttnn.reshape(query, [batch, self.linear_num_key_heads, 1, self.linear_key_head_dim]))
        key = self._repeat_linear_qk(ttnn.reshape(key, [batch, self.linear_num_key_heads, 1, self.linear_key_head_dim]))
        value = ttnn.reshape(value, [batch, self.linear_num_value_heads, 1, self.linear_value_head_dim])

        query_norm = ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(query, query), dim=-1, keepdim=True), 1e-6))
        key_norm = ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(key, key), dim=-1, keepdim=True), 1e-6))
        query = ttnn.multiply(ttnn.multiply(query, query_norm), 1.0 / math.sqrt(self.linear_key_head_dim))
        key = ttnn.multiply(key, key_norm)
        beta = ttnn.reshape(beta, [batch, self.linear_num_value_heads, 1, 1])
        a = ttnn.reshape(a, [batch, self.linear_num_value_heads, 1, 1])
        a = ttnn.typecast(a, ttnn.float32)
        decay = ttnn.exp(ttnn.multiply(self.a_decay, ttnn.softplus(ttnn.add(a, self.dt_bias))))
        decayed_state = ttnn.multiply(recurrent_state, decay)
        recalled = ttnn.matmul(key, decayed_state, dtype=ttnn.float32)
        delta = ttnn.multiply(ttnn.subtract(value, recalled), beta)
        update = ttnn.matmul(ttnn.transpose(key, -2, -1), delta, dtype=ttnn.float32)
        new_recurrent_state = ttnn.add(decayed_state, update)
        ttnn.copy(new_recurrent_state, recurrent_state)
        core = ttnn.matmul(query, new_recurrent_state, dtype=ttnn.bfloat16)

        core = ttnn.reshape(core, [batch, 1, 1, self.linear_num_value_heads, self.linear_value_head_dim])
        z = ttnn.reshape(z, [batch, 1, 1, self.linear_num_value_heads, self.linear_value_head_dim])
        norm = ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.multiply(core, core), dim=-1, keepdim=True), self.rms_norm_eps))
        core = ttnn.multiply(ttnn.multiply(core, norm), self.linear_norm)
        core = ttnn.multiply(core, ttnn.silu(z))
        core = ttnn.reshape(core, [batch, 1, 1, self.linear_num_value_heads * self.linear_value_head_dim])
        return ttnn.linear(core, self.linear_out_proj, dtype=ttnn.bfloat16)

    def _linear_prefill(self, hidden_states, *, linear_state, logical_seq_len: int):
        conv_state, recurrent_state = linear_state
        outputs = []
        for chunk_start in range(0, logical_seq_len, self.linear_chunk_size):
            valid_tokens = min(self.linear_chunk_size, logical_seq_len - chunk_start)
            chunk = ttnn.slice(
                hidden_states,
                [0, 0, chunk_start, 0],
                [hidden_states.shape[0], 1, chunk_start + valid_tokens, self.hidden_size],
            )
            outputs.append(
                self._linear_chunk(
                    chunk,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                    valid_tokens=valid_tokens,
                )
            )
        return ttnn.concat(outputs, dim=2) if len(outputs) > 1 else outputs[0]

    def _linear_decode(self, hidden_states, *, linear_state):
        conv_state, recurrent_state = linear_state
        token = ttnn.permute(hidden_states, [2, 1, 0, 3])
        output = self._linear_token(token, conv_state=conv_state, recurrent_state=recurrent_state)
        return ttnn.permute(output, [2, 1, 0, 3])

    def _prefill_users_independently(
        self,
        hidden_states,
        *,
        logical_seq_len: int,
        cos=None,
        sin=None,
        page_table=None,
        kv_cache=None,
        linear_state=None,
    ):
        """Preserve user isolation for functional multi-user prefill.

        Several installed outer-batch kernels lose accuracy at batch 32.  This
        functional stage therefore builds one device-only user graph per row
        and concatenates the device outputs.  Caller-owned state/cache remains
        authoritative and no host conversion is introduced.
        """
        batch = hidden_states.shape[0]
        outputs = []
        conv_users = []
        recurrent_users = []
        for user in range(batch):
            hidden_user = ttnn.slice(
                hidden_states,
                [user, 0, 0, 0],
                [user + 1, 1, hidden_states.shape[2], self.hidden_size],
            )
            if self.layer_kind == "full_attention":
                cos_user = ttnn.slice(
                    cos, [user, 0, 0, 0], [user + 1, cos.shape[1], cos.shape[2], cos.shape[3]]
                )
                sin_user = ttnn.slice(
                    sin, [user, 0, 0, 0], [user + 1, sin.shape[1], sin.shape[2], sin.shape[3]]
                )
                normalized_user = ttnn.rms_norm(
                    hidden_user, epsilon=self.rms_norm_eps, weight=self.input_norm
                )
                mixed_user = self._full_prefill(
                    normalized_user,
                    cos=cos_user,
                    sin=sin_user,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    logical_seq_len=logical_seq_len,
                    cache_batch_idx=user,
                )
                residual_user = hidden_user
                if logical_seq_len != hidden_user.shape[2]:
                    residual_user = ttnn.slice(
                        hidden_user,
                        [0, 0, 0, 0],
                        [1, 1, logical_seq_len, self.hidden_size],
                    )
                outputs.append(self._finish_layer(residual_user, mixed_user))
            else:
                conv_state, recurrent_state = linear_state
                conv_user = ttnn.slice(
                    conv_state,
                    [user, 0, 0, 0],
                    [user + 1, 1, self.conv_kernel_size, self.conv_dim],
                )
                recurrent_user = ttnn.slice(
                    recurrent_state,
                    [user, 0, 0, 0],
                    [
                        user + 1,
                        self.linear_num_value_heads,
                        self.linear_key_head_dim,
                        self.linear_value_head_dim,
                    ],
                )
                outputs.append(
                    self.prefill_forward(
                        hidden_user,
                        logical_seq_len=logical_seq_len,
                        linear_state=(conv_user, recurrent_user),
                    )
                )
                conv_users.append(conv_user)
                recurrent_users.append(recurrent_user)
        if self.layer_kind == "linear_attention":
            ttnn.copy(ttnn.concat(conv_users, dim=0), linear_state[0])
            ttnn.copy(ttnn.concat(recurrent_users, dim=0), linear_state[1])
        return ttnn.concat(outputs, dim=0)

    def _linear_decode_users_independently(
        self, hidden_states, *, current_positions, linear_state
    ):
        outputs = []
        conv_users = []
        recurrent_users = []
        conv_state, recurrent_state = linear_state
        for user in range(hidden_states.shape[2]):
            hidden_user = ttnn.slice(
                hidden_states,
                [0, 0, user, 0],
                [1, 1, user + 1, self.hidden_size],
            )
            position_user = ttnn.slice(current_positions, [user], [user + 1])
            conv_user = ttnn.slice(
                conv_state,
                [user, 0, 0, 0],
                [user + 1, 1, self.conv_kernel_size, self.conv_dim],
            )
            recurrent_user = ttnn.slice(
                recurrent_state,
                [user, 0, 0, 0],
                [
                    user + 1,
                    self.linear_num_value_heads,
                    self.linear_key_head_dim,
                    self.linear_value_head_dim,
                ],
            )
            outputs.append(
                self.decode_forward(
                    hidden_user,
                    current_positions=position_user,
                    linear_state=(conv_user, recurrent_user),
                )
            )
            conv_users.append(conv_user)
            recurrent_users.append(recurrent_user)
        ttnn.copy(ttnn.concat(conv_users, dim=0), conv_state)
        ttnn.copy(ttnn.concat(recurrent_users, dim=0), recurrent_state)
        return ttnn.concat(outputs, dim=2)

    def prefill_forward(
        self,
        hidden_states,
        *,
        logical_seq_len: int,
        cos=None,
        sin=None,
        page_table=None,
        kv_cache=None,
        linear_state=None,
    ):
        """Run paged prefill (full attention) or stateful prefill (DeltaNet)."""
        if not 0 < logical_seq_len <= self.max_context:
            raise ValueError(f"logical_seq_len must be in [1, {self.max_context}]")
        if logical_seq_len > hidden_states.shape[2]:
            raise ValueError("logical_seq_len cannot exceed the physical hidden-state sequence")
        residual = hidden_states
        if self.layer_kind == "full_attention":
            if any(value is None for value in (cos, sin, page_table, kv_cache)):
                raise ValueError("full_attention prefill requires cos, sin, page_table, and kv_cache")
            if cos.shape[2] < hidden_states.shape[2] or sin.shape[2] < hidden_states.shape[2]:
                raise ValueError("RoPE tensors must cover the physical prefill sequence")
            if page_table.shape[0] < hidden_states.shape[0]:
                raise ValueError("page table needs one row per prefill user")
            if page_table.shape[1] * self.page_block_size < hidden_states.shape[2]:
                raise ValueError("page table does not cover the physical prefill sequence")
            if (
                hidden_states.shape[0] > 1
                and hidden_states.shape[2] <= self.full_prefill_sdpa_limit
            ):
                return self._prefill_users_independently(
                    hidden_states,
                    logical_seq_len=logical_seq_len,
                    cos=cos,
                    sin=sin,
                    page_table=page_table,
                    kv_cache=kv_cache,
                )
            if hidden_states.shape[2] > self.full_prefill_sdpa_limit:
                padded_sdpa_len = _round_up(
                    hidden_states.shape[2], self.full_prefill_q_chunk_size
                )
                if page_table.shape[1] * self.page_block_size < padded_sdpa_len:
                    raise ValueError(
                        "long prefill page table must cover internal 128-token SDPA padding"
                    )
                return self._full_prefill_chunked_layer(
                    hidden_states,
                    cos=cos,
                    sin=sin,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    logical_seq_len=logical_seq_len,
                )
            normalized = ttnn.rms_norm(
                hidden_states, epsilon=self.rms_norm_eps, weight=self.input_norm
            )
            mixed = self._full_prefill(
                normalized,
                cos=cos,
                sin=sin,
                page_table=page_table,
                kv_cache=kv_cache,
                logical_seq_len=logical_seq_len,
            )
            if logical_seq_len != residual.shape[2]:
                residual = ttnn.slice(
                    residual, [0, 0, 0, 0], [residual.shape[0], 1, logical_seq_len, self.hidden_size]
                )
        else:
            if linear_state is None:
                raise ValueError("linear_attention prefill requires linear_state")
            if hidden_states.shape[0] > 1:
                return self._prefill_users_independently(
                    hidden_states,
                    logical_seq_len=logical_seq_len,
                    linear_state=linear_state,
                )
            normalized = ttnn.rms_norm(
                hidden_states, epsilon=self.rms_norm_eps, weight=self.input_norm
            )
            mixed = self._linear_prefill(normalized, linear_state=linear_state, logical_seq_len=logical_seq_len)
            if logical_seq_len != residual.shape[2]:
                residual = ttnn.slice(
                    residual, [0, 0, 0, 0], [residual.shape[0], 1, logical_seq_len, self.hidden_size]
                )
        if self.layer_kind == "linear_attention":
            return self._finish_layer_chunked(residual, mixed, logical_seq_len=logical_seq_len)
        return self._finish_layer(residual, mixed)

    def decode_forward(
        self,
        hidden_states,
        *,
        current_positions,
        cos=None,
        sin=None,
        page_table=None,
        kv_cache=None,
        linear_state=None,
    ):
        """Run one trace-safe decode token with all mutable state on device."""
        residual = hidden_states
        normalized = ttnn.rms_norm(hidden_states, epsilon=self.rms_norm_eps, weight=self.input_norm)
        if self.layer_kind == "full_attention":
            if any(value is None for value in (cos, sin, page_table, kv_cache, current_positions)):
                raise ValueError(
                    "full_attention decode requires cos, sin, page_table, current_positions, and kv_cache"
                )
            mixed = self._full_decode(
                normalized,
                cos=cos,
                sin=sin,
                page_table=page_table,
                current_positions=current_positions,
                kv_cache=kv_cache,
            )
        else:
            if linear_state is None or current_positions is None:
                raise ValueError("linear_attention decode requires current_positions and linear_state")
            if hidden_states.shape[2] > 1:
                return self._linear_decode_users_independently(
                    hidden_states,
                    current_positions=current_positions,
                    linear_state=linear_state,
                )
            mixed = self._linear_decode(normalized, linear_state=linear_state)
        return self._finish_layer(residual, mixed)

    def forward(self, hidden_states, *, mode: str, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")

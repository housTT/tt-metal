# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Graph-fused TTNN decoder layer for ``Qwen/Qwen3.8-Flash-Next``.

This stage keeps the functional decoder's public contract and persistent-state
implementation, but replaces runtime subgraphs with TTNN fused operations.  It
has no functional-path switch: construction returns :class:`FusedDecoder`, and
the inherited public entry points dynamically dispatch every rewritten method
below.

The setup-only :meth:`from_state_dict` packs weights for shared-LHS matmuls and
folds constant scales.  Measured prefill/decode contains no host conversion.
"""

from __future__ import annotations

import math

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import chunk_gated_delta_rule_fused_adapter
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import l2_norm_ttnn

from .functional_decoder import (
    QSA_BLOCK_TOPK,
    FunctionalDecoder,
    _embedding_tiled_output,
    _free,
    _pad_seq,
    _round_up,
    _shape,
)
from .model_config import LINEAR_ATTENTION, QWEN_SPARSE_ATTENTION


class FusedDecoder(FunctionalDecoder):
    """Functional decoder with device-proven graph fusions applied."""

    # A source-visible manifest lets tests and stage review prove that the
    # delivered class is the fused path rather than a functional alias.
    FUSION_MANIFEST = (
        "shared_lhs_hyper_down_inject",
        "shared_lhs_moe_input",
        "group_batched_sparse_moe_down",
        "shared_lhs_gdn_qkv_beta_decay",
        "shared_lhs_qsa_qkv_gate_index",
        "shared_lhs_ple_key_value",
        "dedicated_qkv_head_split",
        "dedicated_decode_qkv_head_split",
        "retained_decode_value_sharding",
        "dedicated_hf_rotary_embedding",
        "structural_concat_heads_elimination",
        "dedicated_scaled_dot_product_attention",
        "dedicated_scaled_dot_product_attention_decode",
        "dedicated_paged_fused_kv_update",
        "dedicated_kda_qkv_causal_conv1d_silu_prefill",
        "native_sdpa_grouped_query_heads",
        "per_kv_head_paged_gather",
        "index_matmul_transpose_relu_and_scale",
        "integer_address_ternary_addcmul",
        "dedicated_rms_norm_l2",
        "rms_norm_scale_weight_fold",
        "persistent_recurrent_gdn_decode_l1_exp_and_mac",
        "packed_gdn_projection_bias",
        "fir_ternary_mac",
        "hyper_inject_scalar_mac",
        "normalized_topk_router",
        "static_compressed_address_precompute",
        "persistent_static_block_rope_rows",
        "persistent_compressed_index_key_cache",
        "binary_input_gate_activations",
        "consumer_fused_silu",
        "scaled_sum_to_mean",
    )

    @classmethod
    def from_state_dict(cls, *args, **kwargs) -> "FusedDecoder":
        """Build the proven functional state, then perform setup-only packing."""

        layer = super().from_state_dict(*args, **kwargs)
        w = layer.w
        s = layer.shapes

        def pack(name: str, parts: tuple[str, ...], scales: tuple[float, ...] | None = None):
            values = []
            for index, part in enumerate(parts):
                value = w[part]
                if scales is not None and scales[index] != 1.0:
                    value = ttnn.multiply(value, scales[index])
                values.append(value)
            w[name] = ttnn.concat(values, dim=-1)
            for value, part in zip(values, parts):
                _free(value, w[part], w[name])
                ttnn.deallocate(w.pop(part))

        # Both hyperconnection projections share the group-normalized input.
        # The 1/hc_count factors are folded into the setup weights.
        for prefix in ("attn_hc", "mlp_hc"):
            pack(
                f"{prefix}_down_inject",
                (f"{prefix}_down", f"{prefix}_inject"),
                (1.0 / s.hc_count, 1.0 / s.hc_count),
            )

        # Router, shared SwiGLU inputs and shared scalar gate all consume x.
        pack(
            "moe_input",
            ("router", "shared_gate_proj", "shared_up_proj", "shared_gate"),
        )
        layer.expert_gate_up = ttnn.concat([layer.experts.gate, layer.experts.up], dim=-1)
        ttnn.deallocate(layer.experts.gate)
        ttnn.deallocate(layer.experts.up)

        if s.layer_type == LINEAR_ATTENTION:
            # These three projections all require FP32 outputs.  A measured
            # qkv/b/a/z FP32 pack lost on layer-1 and required a z typecast.
            pack("gdn_qkv_b_a", ("in_proj_qkv", "in_proj_b", "in_proj_a"))
            bias_prefix = ttnn.zeros(
                (1, 1, 1, s.linear_qkv_width + s.linear_num_value_heads),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=layer.mesh_device,
            )
            dt_bias = ttnn.reshape(w["dt_bias"], (1, 1, 1, s.linear_num_value_heads))
            w["gdn_qkv_b_a_bias"] = ttnn.concat([bias_prefix, dt_bias], dim=-1)
            ttnn.deallocate(bias_prefix)
            _free(dt_bias, w["dt_bias"], w["gdn_qkv_b_a_bias"])
            ttnn.deallocate(w.pop("dt_bias"))
            # Preserve the original BF16 taps for the dedicated prefill FIR;
            # decode's FP32 recurrent state still needs widened copies.
            layer.fused_prefill_conv_taps = tuple(w[f"conv_tap_{tap}"] for tap in range(s.linear_conv_kernel_dim))
            # Ternary MAC requires homogeneous floating operands.  The GDN
            # projection/conv state is FP32, so widen its BF16 tap constants
            # once during setup (the represented values remain exact).
            for tap in range(s.linear_conv_kernel_dim):
                name = f"conv_tap_{tap}"
                widened = ttnn.typecast(w[name], ttnn.float32)
                w[name] = widened
            # Split persistent decode taps remove concat/slice state movement.
            # Canonical state stays in DRAM so a live L1 prefill residual
            # cannot overlap it; multichip decode hydrates its declared shared
            # L1 workspace immediately before executing a layer.
            tap_memory = ttnn.DRAM_MEMORY_CONFIG
            layer.fused_conv_state = tuple(
                ttnn.zeros(
                    (layer.max_batch, 1, 1, s.linear_qkv_width),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=layer.mesh_device,
                    memory_config=tap_memory,
                )
                for _ in range(s.linear_conv_kernel_dim - 1)
            )
            # Keep the large recurrent matrix in DRAM.  The batch-one L1
            # placement used by the original fused path corrupts repeated
            # decode transitions once temporary L1 allocations overlap the
            # persistent interleaved tensor; the progressing-HF gate catches
            # the resulting abrupt state drift after a few tokens.
            norm_one = ttnn.ones(
                (1, 1, 1, s.linear_key_head_dim),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=layer.mesh_device,
            )
            layer.fused_q_norm_weight = ttnn.multiply(norm_one, 1.0 / s.linear_key_head_dim)
            layer.fused_k_norm_weight = ttnn.multiply(norm_one, 1.0 / math.sqrt(s.linear_key_head_dim))
            ttnn.deallocate(norm_one)

        if s.layer_type == QWEN_SPARSE_ATTENTION:
            # q/k/v are contiguous so a dedicated head-split op can consume the
            # first slice of this single projection.
            pack(
                "qsa_input",
                ("q_proj", "k_proj", "v_proj", "attn_gate", "index_q", "index_k"),
            )
            # The post-matmul score scale is positive and therefore commutes
            # with ReLU.  Folding it into the query norm weight saves a large
            # full-context eltwise pass.  Static block positions are likewise
            # setup-only constants.
            scaled = ttnn.multiply(w["index_q_norm"], 1.0 / math.sqrt(s.indexer_head_dim))
            ttnn.deallocate(w["index_q_norm"])
            w["index_q_norm"] = scaled
            layer.fused_index_block_positions = ttnn.multiply(
                layer.const["index_block_ids"],
                s.indexer_compress_ratio,
            )
            compressed_per_page = layer.block_size // s.indexer_compress_ratio
            layer.fused_index_virtual_pages = ttnn.bitwise_right_shift(
                layer.const["index_block_ids"],
                int(math.log2(compressed_per_page)),
            )
            layer.fused_index_in_page = ttnn.bitwise_and(
                layer.const["index_block_ids"],
                compressed_per_page - 1,
            )
            layer.fused_index_scalar_one = ttnn.ones(
                (1, 1, 1, 1),
                dtype=ttnn.int32,
                layout=ttnn.TILE_LAYOUT,
                device=layer.mesh_device,
            )
            layer.fused_index_offset_ones = ttnn.ones_like(layer.const["index_token_offsets"])
            layer.fused_block_rot_source = None
            layer.fused_block_cos = None
            layer.fused_block_sin = None
            layer.fused_index_key_cache = ttnn.zeros(
                (layer.max_num_blocks, 1, compressed_per_page, s.indexer_head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=layer.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # ``paged_fused_update_cache`` requires its K and V updates on
            # disjoint height-sharded core grids.  Keep K on the functional
            # decoder's first ``max_batch`` cores and place V on the adjacent
            # row-major cores in the same 8x8 decode grid.
            value_ranges = set()
            remaining = layer.max_batch
            linear_core = layer.max_batch
            while remaining:
                core_x = linear_core % 8
                core_y = linear_core // 8
                count = min(remaining, 8 - core_x)
                value_ranges.add(
                    ttnn.CoreRange(
                        ttnn.CoreCoord(core_x, core_y),
                        ttnn.CoreCoord(core_x + count - 1, core_y),
                    )
                )
                remaining -= count
                linear_core += count
            layer.fused_decode_value_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(32, s.head_dim),
                core_grid=ttnn.CoreRangeSet(value_ranges),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        if s.has_ple:
            pack("ple_key_value", ("ple_key", "ple_value"))
            # dot(key, query) / sqrt(hidden) == dot(key, query / sqrt(hidden)).
            scaled = ttnn.multiply(w["ple_norm_query"], 1.0 / math.sqrt(s.hidden_size))
            ttnn.deallocate(w["ple_norm_query"])
            w["ple_norm_query"] = scaled
            ple_state_memory = ttnn.DRAM_MEMORY_CONFIG
            layer.fused_ple_conv_state = tuple(
                ttnn.zeros(
                    (layer.max_batch, 1, 1, s.hc_hidden_size),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=layer.mesh_device,
                    memory_config=ple_state_memory,
                )
                for _ in range(s.ple_conv_state_len)
            )

        return layer

    @staticmethod
    def _slice_last(tensor, start: int, end: int):
        lead = _shape(tensor)[:-1]
        return ttnn.slice(tensor, [0] * len(lead) + [start], lead + [end])

    @staticmethod
    def _l2norm(x, epsilon: float = 1e-6):
        """Dedicated RMSNorm form of exact L2 normalization."""

        return l2_norm_ttnn(x, dim=-1, eps=epsilon)

    # -------------------------------------------------------- hyperconnection

    def _hyper_mix(self, hyper_input, prefix: str):
        s = self.shapes
        normed = self._rms_norm(
            hyper_input,
            self.w[f"{prefix}_norm"],
            s.rms_norm_eps,
            group_count=s.hc_count,
        )
        packed = self._linear(normed, self.w[f"{prefix}_down_inject"])
        low = self._slice_last(packed, 0, s.hc_lowrank)
        injection = self._slice_last(packed, s.hc_lowrank, s.hc_lowrank + s.hc_count)
        ttnn.deallocate(packed)
        low = ttnn.silu(low)
        mix = self._linear(low, self.w[f"{prefix}_up"])
        ttnn.deallocate(low)

        rows = math.prod(_shape(normed)[:-1])
        norm_groups = ttnn.reshape(normed, (rows, s.hc_count, s.hidden_size))
        mix_groups = ttnn.reshape(mix, (rows, s.hc_count, s.hidden_size))
        mixed = ttnn.multiply(
            norm_groups,
            mix_groups,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        _free(norm_groups, normed, mixed)
        _free(mix_groups, mix, mixed)
        ttnn.deallocate(mix)
        mixed = ttnn.mean(mixed, dim=1, keepdim=True)
        mixed = ttnn.reshape(mixed, (*_shape(hyper_input)[:-1], s.hidden_size))
        ttnn.deallocate(normed)
        return mixed, hyper_input, injection

    def _hyper_inject(self, hyper_input, block_output, injection):
        s = self.shapes
        rows = math.prod(_shape(block_output)[:-1])
        value = ttnn.reshape(block_output, (rows, 1, s.hidden_size))
        gate = ttnn.reshape(injection, (rows, s.hc_count, 1))
        projected = ttnn.multiply(
            value,
            gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        _free(value, block_output, projected)
        _free(gate, injection, projected)
        ttnn.deallocate(block_output)
        ttnn.deallocate(injection)
        projected = ttnn.reshape(projected, _shape(hyper_input))
        out = ttnn.mac(projected, 2.0, hyper_input)
        ttnn.deallocate(projected)
        return out

    # --------------------------------------------------------- mixture of experts

    def _routing_from_logits(self, logits):
        s = self.shapes
        zeros = ttnn.zeros_like(logits)
        values, indices = ttnn.topk(logits, k=s.num_experts_per_tok, dim=-1, sorted=True)
        ttnn.deallocate(logits)
        selected_values = values
        values = ttnn.softmax(selected_values, dim=-1)
        _free(selected_values, values)
        routing = ttnn.scatter(zeros, dim=-1, index=indices, src=values)
        ttnn.deallocate(zeros)
        ttnn.deallocate(values)
        ttnn.deallocate(indices)
        return routing

    def _routed_experts(self, x, routing):
        """Evaluate gate/up with one packed sparse matmul, then sparse down."""

        s = self.shapes
        tokens = int(x.shape[-2])
        if tokens % 32:
            raise ValueError(f"expert input must be tile padded, got {tokens}")
        groups = tokens // 32
        grouped_x = ttnn.reshape(x, (1, groups, 32, s.hidden_size))
        routing_groups = ttnn.reshape(routing, (1, groups, 32, s.num_experts))
        sparsity = ttnn.max(routing_groups, dim=2, keepdim=True)
        sparsity = ttnn.reshape(sparsity, (1, 1, groups, s.num_experts))
        sparsity = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([32, 32])
        gate_up_cfg = self._sparse_matmul_config(
            32,
            2 * s.moe_intermediate_size,
            s.hidden_size,
        )
        down_cfg = self._sparse_matmul_config(32, s.hidden_size, s.moe_intermediate_size)

        gate_up_sparse = ttnn.sparse_matmul(
            grouped_x,
            self.expert_gate_up,
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_cfg,
            compute_kernel_config=self.compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _free(grouped_x, x, gate_up_sparse)
        # Dense-A/sparse-B output is logically
        # [1, groups, 1, experts, 32, 2 * intermediate].  Keep the group-major
        # ordering instead of transposing it to expert-major: A-sparse/B-dense
        # sparse_matmul can consume [groups, experts, 32, intermediate] in one
        # call with the existing [1, 1, groups, experts] sparsity tensor.
        gate_up = ttnn.reshape(
            gate_up_sparse,
            (groups, s.num_experts, 32, 2 * s.moe_intermediate_size),
        )
        _free(gate_up_sparse, gate_up)
        gate = self._slice_last(gate_up, 0, s.moe_intermediate_size)
        up = self._slice_last(gate_up, s.moe_intermediate_size, 2 * s.moe_intermediate_size)
        ttnn.deallocate(gate_up)
        hidden = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        token_weights = ttnn.permute(routing_groups, (1, 3, 2, 0))
        weighted_hidden = ttnn.multiply(hidden, token_weights)
        ttnn.deallocate(hidden)
        _free(token_weights, routing, routing_groups, weighted_hidden)

        down = ttnn.sparse_matmul(
            weighted_hidden,
            self.experts.down,
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            program_config=down_cfg,
            compute_kernel_config=self.compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _free(weighted_hidden, down)
        ttnn.deallocate(sparsity)
        out = ttnn.experimental.fast_reduce_nc(down, dims=[1])
        ttnn.deallocate(down)
        return ttnn.reshape(ttnn.unsqueeze_to_4D(out), (1, 1, tokens, s.hidden_size))

    def _moe(self, x):
        s = self.shapes
        logical = int(x.shape[-2])
        padded = _round_up(logical, 32)
        work = _pad_seq(x, padded, x) if padded != logical else x

        packed = self._linear(work, self.w["moe_input"])
        cursor = 0
        logits = self._slice_last(packed, cursor, cursor + s.num_experts)
        cursor += s.num_experts
        gate = self._slice_last(packed, cursor, cursor + s.shared_expert_intermediate_size)
        cursor += s.shared_expert_intermediate_size
        up = self._slice_last(packed, cursor, cursor + s.shared_expert_intermediate_size)
        cursor += s.shared_expert_intermediate_size
        scalar = self._slice_last(packed, cursor, cursor + 1)
        ttnn.deallocate(packed)

        routing = self._routing_from_logits(logits)
        routed = self._routed_experts(work, routing)
        ttnn.deallocate(routing)

        hidden = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        gated_hidden = ttnn.multiply(
            hidden,
            scalar,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        ttnn.deallocate(hidden)
        ttnn.deallocate(scalar)
        shared = self._linear(gated_hidden, self.w["shared_down_proj"])
        ttnn.deallocate(gated_hidden)

        out = ttnn.add(routed, shared)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        if padded != logical:
            trimmed = ttnn.slice(out, [0, 0, 0, 0], [1, 1, logical, s.hidden_size])
            _free(out, trimmed)
            out = trimmed
        return out

    # --------------------------------------------------- projected lexical embed

    def _ple_gate(self, hyper_input, embeddings):
        s = self.shapes
        packed = self._linear(embeddings, self.w["ple_key_value"])
        key = self._slice_last(packed, 0, s.hc_hidden_size)
        value = self._slice_last(packed, s.hc_hidden_size, s.hc_hidden_size + s.hidden_size)
        ttnn.deallocate(packed)
        key = self._rms_norm(key, self.w["ple_norm_key"], s.rms_norm_eps, group_count=s.hc_count)
        query = self._rms_norm(
            hyper_input,
            self.w["ple_norm_query"],
            s.rms_norm_eps,
            group_count=s.hc_count,
        )
        rows = math.prod(_shape(hyper_input)[:-1])
        key = ttnn.reshape(key, (rows, s.hc_count, s.hidden_size))
        query = ttnn.reshape(query, (rows, s.hc_count, s.hidden_size))
        product = ttnn.multiply(key, query)
        ttnn.deallocate(key)
        ttnn.deallocate(query)
        score = ttnn.sum(product, dim=-1, keepdim=True)
        ttnn.deallocate(product)
        sign = ttnn.sign(score)
        magnitude = ttnn.sqrt(ttnn.clamp(ttnn.abs(score), min=1e-6, max=None))
        ttnn.deallocate(score)
        score = ttnn.multiply(sign, magnitude)
        ttnn.deallocate(sign)
        ttnn.deallocate(magnitude)
        value = ttnn.reshape(value, (rows, 1, s.hidden_size))
        gated = ttnn.multiply(
            score,
            value,
            input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        ttnn.deallocate(score)
        ttnn.deallocate(value)
        return ttnn.reshape(gated, _shape(hyper_input))

    @staticmethod
    def _fir_term(piece, weight, accumulator):
        if accumulator is None:
            return ttnn.multiply(piece, weight)
        updated = ttnn.mac(piece, weight, accumulator)
        ttnn.deallocate(accumulator)
        return updated

    def _ple_prefill(self, hyper_input, embeddings, *, user_id: int, logical: int):
        s = self.shapes
        gated = self._ple_gate(hyper_input, embeddings)
        normed = self._rms_norm(
            gated,
            self.w["ple_norm_conv"],
            s.rms_norm_eps,
            group_count=s.hc_count,
        )
        window = ttnn.concat([self.user_ple_conv_state[user_id], normed], dim=-2)
        length = int(normed.shape[-2])
        acc = None
        for tap in range(s.ple_conv_kernel_size):
            start = tap * s.ple_conv_dilation
            piece = ttnn.slice(window, [0, 0, start, 0], [1, 1, start + length, s.hc_hidden_size])
            acc = self._fir_term(piece, self.w[f"ple_conv_tap_{tap}"], acc)
            _free(piece, window, acc)
        new_state = ttnn.slice(
            window,
            [0, 0, logical, 0],
            [1, 1, logical + s.ple_conv_state_len, s.hc_hidden_size],
        )
        _free(window, normed, new_state)
        ttnn.deallocate(normed)
        ttnn.deallocate(self.user_ple_conv_state[user_id])
        self.user_ple_conv_state[user_id] = ttnn.clone(new_state, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _free(new_state, self.user_ple_conv_state[user_id])
        out = ttnn.add(
            gated,
            acc,
            input_tensor_b_activations=[ttnn.UnaryOpType.SILU],
        )
        ttnn.deallocate(gated)
        ttnn.deallocate(acc)
        return out

    def _ple_decode(self, hyper_input, embeddings):
        s = self.shapes
        gated_public = self._ple_gate(hyper_input, embeddings)
        normed_public = self._rms_norm(
            gated_public,
            self.w["ple_norm_conv"],
            s.rms_norm_eps,
            group_count=s.hc_count,
        )
        normed = ttnn.reshape(normed_public, (self.max_batch, 1, 1, s.hc_hidden_size))
        acc = None
        for tap in range(s.ple_conv_kernel_size):
            index = tap * s.ple_conv_dilation
            piece = normed if index == s.ple_conv_state_len else self.fused_ple_conv_state[index]
            acc = self._fir_term(piece, self.w[f"ple_conv_tap_{tap}"], acc)
        for tap in range(s.ple_conv_state_len - 1):
            ttnn.copy(self.fused_ple_conv_state[tap + 1], self.fused_ple_conv_state[tap])
        ttnn.copy(normed, self.fused_ple_conv_state[-1])
        _free(normed, normed_public, *self.fused_ple_conv_state)
        ttnn.deallocate(normed_public)
        reshaped = ttnn.reshape(acc, _shape(gated_public))
        out = ttnn.add(
            gated_public,
            reshaped,
            input_tensor_b_activations=[ttnn.UnaryOpType.SILU],
        )
        ttnn.deallocate(gated_public)
        _free(reshaped, acc, out)
        ttnn.deallocate(acc)
        return out

    # ----------------------------------------------------- gated DeltaNet mixer

    def _gdn_inputs(self, x):
        s = self.shapes
        packed = ttnn.linear(
            x,
            self.w["gdn_qkv_b_a"],
            bias=self.w["gdn_qkv_b_a_bias"],
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_cfg,
        )
        mixed = self._slice_last(packed, 0, s.linear_qkv_width)
        b_start = s.linear_qkv_width
        b = self._slice_last(packed, b_start, b_start + s.linear_num_value_heads)
        a = self._slice_last(
            packed,
            b_start + s.linear_num_value_heads,
            b_start + 2 * s.linear_num_value_heads,
        )
        ttnn.deallocate(packed)
        z = self._linear(x, self.w["in_proj_z"])
        beta = ttnn.sigmoid(b)
        ttnn.deallocate(b)
        soft = ttnn.softplus(a, beta=1.0, threshold=20.0)
        ttnn.deallocate(a)
        g = ttnn.multiply(soft, self.w["neg_exp_A"])
        ttnn.deallocate(soft)
        return mixed, z, beta, g

    def _causal_conv_prefill(self, mixed, *, user_id: int, logical: int):
        s = self.shapes
        window = ttnn.concat([self.user_conv_state[user_id], mixed], dim=-2)
        length = int(mixed.shape[-2])
        acc = None
        for tap in range(s.linear_conv_kernel_dim):
            piece = ttnn.slice(window, [0, 0, tap, 0], [1, 1, tap + length, s.linear_qkv_width])
            acc = self._fir_term(piece, self.w[f"conv_tap_{tap}"], acc)
            _free(piece, window, acc)
        new_state = ttnn.slice(
            window,
            [0, 0, logical, 0],
            [1, 1, logical + s.linear_conv_kernel_dim - 1, s.linear_qkv_width],
        )
        _free(window, mixed, new_state)
        ttnn.deallocate(self.user_conv_state[user_id])
        self.user_conv_state[user_id] = ttnn.clone(new_state, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _free(new_state, self.user_conv_state[user_id])
        out = ttnn.silu(acc)
        ttnn.deallocate(acc)
        return out

    def _gdn_prefill(self, x, *, user_id: int, logical: int):
        """Run the dedicated QKV causal-convolution/SILU prefill kernel."""

        s = self.shapes
        padded = int(x.shape[-2])
        mixed, z, beta, g = self._gdn_inputs(x)

        # The fused kernel is BF16 row-major at its input boundary.  Keep the
        # persistent functional state FP32 and update it from the original
        # projection so decode semantics remain unchanged.
        history = self.user_conv_state[user_id]
        window = ttnn.concat([history, mixed], dim=-2)
        new_state = ttnn.slice(
            window,
            [0, 0, logical, 0],
            [1, 1, logical + s.linear_conv_kernel_dim - 1, s.linear_qkv_width],
        )
        _free(window, mixed, new_state)
        mixed_bf16 = ttnn.typecast(mixed, ttnn.bfloat16)
        history_bf16 = ttnn.typecast(history, ttnn.bfloat16)
        ttnn.deallocate(history)
        self.user_conv_state[user_id] = ttnn.clone(new_state, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _free(new_state, self.user_conv_state[user_id])
        mixed_rm = ttnn.to_layout(mixed_bf16, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        history_rm = ttnn.to_layout(history_bf16, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mixed_bf16)
        ttnn.deallocate(history_bf16)
        mixed_input = ttnn.reshape(mixed_rm, (1, padded, s.linear_qkv_width))
        history_input = ttnn.reshape(history_rm, (1, s.linear_conv_kernel_dim - 1, s.linear_qkv_width))
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            mixed_input,
            history_input,
            *self.fused_prefill_conv_taps,
            s.linear_qk_width,
            s.linear_qk_width,
            s.linear_value_width,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=640),
        )
        _free(mixed_input, mixed_rm, q, k, v)
        _free(history_input, history_rm, q, k, v)
        ttnn.deallocate(mixed_rm)
        ttnn.deallocate(history_rm)
        ttnn.deallocate(mixed)

        # Padded positions must be identity recurrence updates (beta=g=0).
        if logical != padded:
            positions = ttnn.arange(0, padded, 1, device=self.mesh_device, dtype=ttnn.int32)
            valid = ttnn.lt(positions, logical)
            valid = ttnn.reshape(valid, (1, 1, padded, 1))
            beta = ttnn.multiply(beta, valid)
            g = ttnn.multiply(g, valid)
            ttnn.deallocate(positions)
            ttnn.deallocate(valid)

        beta = ttnn.reshape(beta, (1, padded, s.linear_num_value_heads))
        g = ttnn.reshape(g, (1, padded, s.linear_num_value_heads))
        core, state = chunk_gated_delta_rule_fused_adapter(
            q,
            k,
            v,
            beta,
            g,
            initial_state=self.user_recurrent_state[user_id],
            device=self.mesh_device,
            qkv_head_dims=(
                s.linear_num_key_heads,
                s.linear_key_head_dim,
                s.linear_num_value_heads,
                s.linear_value_head_dim,
            ),
            const_tiles=self.const["gdn_tiles"],
        )
        ttnn.deallocate(self.user_recurrent_state[user_id])
        self.user_recurrent_state[user_id] = state
        core = ttnn.reshape(
            core,
            (1, padded, s.linear_num_value_heads, s.linear_value_head_dim),
        )
        return self._gdn_epilogue(core, z, public_shape=_shape(x))

    def _gdn_decode(self, x):
        s = self.shapes
        batch = self.max_batch
        mixed_public, z, beta, g = self._gdn_inputs(x)
        mixed = ttnn.reshape(mixed_public, (batch, 1, 1, s.linear_qkv_width))
        ttnn.deallocate(mixed_public)
        acc = ttnn.multiply(self.fused_conv_state[0], self.w["conv_tap_0"])
        for tap in range(1, s.linear_conv_kernel_dim - 1):
            acc = self._fir_term(self.fused_conv_state[tap], self.w[f"conv_tap_{tap}"], acc)
        acc = self._fir_term(mixed, self.w[f"conv_tap_{s.linear_conv_kernel_dim - 1}"], acc)
        for tap in range(s.linear_conv_kernel_dim - 2):
            ttnn.copy(self.fused_conv_state[tap + 1], self.fused_conv_state[tap])
        ttnn.copy(mixed, self.fused_conv_state[-1])
        _free(mixed, mixed_public, *self.fused_conv_state)
        conv = ttnn.silu(acc)
        ttnn.deallocate(acc)
        q, k, v = self._split_gdn(conv)
        ttnn.deallocate(conv)
        q = ttnn.reshape(q, (batch, 1, s.linear_num_key_heads, s.linear_key_head_dim))
        k = ttnn.reshape(k, (batch, 1, s.linear_num_key_heads, s.linear_key_head_dim))
        v = ttnn.reshape(v, (batch, 1, s.linear_num_value_heads, s.linear_value_head_dim))
        repeat = s.linear_num_value_heads // s.linear_num_key_heads
        q = ttnn.repeat_interleave(q, repeat, dim=2)
        k = ttnn.repeat_interleave(k, repeat, dim=2)
        beta = ttnn.reshape(beta, (batch, 1, s.linear_num_value_heads))
        g = ttnn.reshape(g, (batch, 1, s.linear_num_value_heads))
        core, state = self._recurrent_gdn_decode(q, k, v, beta, g)
        for tensor in (q, k, v, beta, g):
            ttnn.deallocate(tensor)
        ttnn.copy(state, self.recurrent_state)
        _free(state, self.recurrent_state)
        return self._gdn_epilogue(core, z, public_shape=_shape(x))

    def prepare_decode_state(self) -> None:
        s = self.shapes
        if self.max_batch != 1:
            super().prepare_decode_state()
            conv_source = self.conv_state
        else:
            # The common batch-one traced path restores the large persistent
            # states directly, avoiding the base implementation's concat.
            if s.layer_type == LINEAR_ATTENTION:
                ttnn.copy(self.user_recurrent_state[0], self.recurrent_state)
                conv_source = self.user_conv_state[0]
            if s.has_ple:
                ttnn.copy(self.user_ple_conv_state[0], self.ple_conv_state)
        if s.layer_type == LINEAR_ATTENTION:
            for tap, target in enumerate(self.fused_conv_state):
                piece = ttnn.slice(
                    conv_source,
                    [0, 0, tap, 0],
                    [self.max_batch, 1, tap + 1, s.linear_qkv_width],
                )
                ttnn.copy(piece, target)
                _free(piece, conv_source, target)
        if s.has_ple:
            for tap, target in enumerate(self.fused_ple_conv_state):
                piece = ttnn.slice(
                    self.ple_conv_state,
                    [0, 0, tap, 0],
                    [self.max_batch, 1, tap + 1, s.hc_hidden_size],
                )
                ttnn.copy(piece, target)
                _free(piece, self.ple_conv_state, target)

    def _recurrent_gdn_decode(self, q, k, v, beta, g):
        """One FP32 recurrent step with adjacent eltwise operations folded."""

        s = self.shapes
        batch = self.max_batch
        heads = s.linear_num_value_heads
        key_dim = s.linear_key_head_dim
        value_dim = s.linear_value_head_dim
        # Standalone fused decode keeps its canonical state in DRAM; the
        # multichip wrapper temporarily binds the same field to its shared L1
        # workspace.  Follow that active owner so multichip recurrence remains
        # entirely L1 while standalone updates cannot silently migrate the
        # canonical state back to the unsafe persistent-L1 placement.
        state_memory = self.recurrent_state.memory_config()

        # l2_norm(q) / sqrt(K) == rms_norm(q, eps/K) / K.  This
        # collapses the functional five-op normalization plus scale to two ops.
        q_row = ttnn.rms_norm(
            q,
            epsilon=1e-6 / key_dim,
            weight=self.fused_q_norm_weight,
            memory_config=state_memory,
        )
        q_row = ttnn.reshape(q_row, (batch, heads, 1, key_dim), memory_config=state_memory)
        k_row = ttnn.rms_norm(
            k,
            epsilon=1e-6 / key_dim,
            weight=self.fused_k_norm_weight,
            memory_config=state_memory,
        )
        k_row = ttnn.reshape(k_row, (batch, heads, 1, key_dim), memory_config=state_memory)
        v_row = ttnn.reshape(v, (batch, heads, value_dim), memory_config=state_memory)
        beta_row = ttnn.reshape(beta, (batch, heads, 1, 1), memory_config=state_memory)
        g_row = ttnn.reshape(g, (batch, heads, 1, 1), memory_config=state_memory)

        state = self.recurrent_state
        state = ttnn.multiply(
            state,
            g_row,
            input_tensor_b_activations=[ttnn.UnaryOpType.EXP],
            memory_config=state_memory,
        )
        read_cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            in0_block_w=key_dim // 32,
            out_subblock_h=1,
            out_subblock_w=min(2, value_dim // 32),
            per_core_M=1,
            per_core_N=value_dim // 32,
        )
        read_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        memory = ttnn.matmul(
            k_row,
            state,
            memory_config=state_memory,
            program_config=read_cfg,
            compute_kernel_config=read_compute,
        )
        memory = ttnn.reshape(memory, (batch, heads, value_dim), memory_config=state_memory)
        delta = ttnn.subtract(v_row, memory, memory_config=state_memory)
        ttnn.deallocate(memory)

        k_col = ttnn.reshape(k_row, (batch, heads, key_dim, 1), memory_config=state_memory)
        delta_row = ttnn.reshape(delta, (batch, heads, 1, value_dim), memory_config=state_memory)
        outer_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        outer = ttnn.matmul(
            k_col,
            delta_row,
            memory_config=state_memory,
            compute_kernel_config=outer_compute,
        )
        # h + beta * outer in one ternary kernel.
        updated_state = ttnn.mac(outer, beta_row, state)
        ttnn.deallocate(outer)
        ttnn.deallocate(state)
        core = ttnn.matmul(
            q_row,
            updated_state,
            memory_config=state_memory,
            program_config=read_cfg,
            compute_kernel_config=read_compute,
        )
        core = ttnn.reshape(core, (batch, 1, heads, value_dim), memory_config=state_memory)
        return core, updated_state

    # ---------------------------------------------------- Qwen sparse attention

    @staticmethod
    def _apply_rope(x, cos, sin, rotary_dim: int):
        head_dim = int(x.shape[-1])
        lead = _shape(x)[:-1]
        rotary = ttnn.slice(x, [0] * len(lead) + [0], lead + [rotary_dim])
        batch, heads, tokens = lead
        if batch == 1:
            embedded = ttnn.experimental.rotary_embedding_hf(
                rotary,
                cos,
                sin,
                is_decode_mode=False,
            )
            ttnn.deallocate(rotary)
        else:
            # The dedicated kernel requires cache leading dimensions [1, 1].
            # Flatten batch into the token axis after moving heads first, then
            # restore the public layout entirely on device.
            head_first = ttnn.permute(rotary, (1, 0, 2, 3))
            ttnn.deallocate(rotary)
            flat = ttnn.reshape(head_first, (1, heads, batch * tokens, rotary_dim))
            _free(head_first, flat)
            cos_first = ttnn.permute(cos, (1, 0, 2, 3))
            sin_first = ttnn.permute(sin, (1, 0, 2, 3))
            flat_cos = ttnn.reshape(cos_first, (1, 1, batch * tokens, rotary_dim))
            flat_sin = ttnn.reshape(sin_first, (1, 1, batch * tokens, rotary_dim))
            _free(cos_first, cos, flat_cos)
            _free(sin_first, sin, flat_sin)
            embedded = ttnn.experimental.rotary_embedding_hf(
                flat,
                flat_cos,
                flat_sin,
                is_decode_mode=False,
            )
            ttnn.deallocate(flat)
            ttnn.deallocate(flat_cos)
            ttnn.deallocate(flat_sin)
            if int(embedded.shape[-2]) != batch * tokens:
                padded = embedded
                embedded = ttnn.slice(
                    padded,
                    [0, 0, 0, 0],
                    [1, heads, batch * tokens, rotary_dim],
                )
                _free(padded, embedded)
            restored = ttnn.reshape(embedded, (heads, batch, tokens, rotary_dim))
            _free(embedded, restored)
            embedded = ttnn.permute(restored, (1, 0, 2, 3))
            _free(restored, embedded)
        # The rotary kernel exposes its tile-padded token axis for decode
        # lengths below 32.  Restore the caller's logical shape before the
        # partial-RoPE suffix concat.
        if _shape(embedded)[:-1] != lead:
            padded = embedded
            embedded = ttnn.slice(padded, [0] * len(lead) + [0], lead + [rotary_dim])
            _free(padded, embedded)
        if rotary_dim == head_dim:
            return embedded
        rest = ttnn.slice(x, [0] * len(lead) + [rotary_dim], lead + [head_dim])
        out = ttnn.concat([embedded, rest], dim=-1)
        ttnn.deallocate(embedded)
        ttnn.deallocate(rest)
        return out

    def _qsa_projections(self, x, positions, rot_mats, *, decode: bool):
        s = self.shapes
        batch = self.max_batch if decode else 1
        tokens = 1 if decode else int(x.shape[-2])
        packed = self._linear(x, self.w["qsa_input"])
        qkv_width = s.q_width + 2 * s.kv_width
        qkv = self._slice_last(packed, 0, qkv_width)
        gate = self._slice_last(packed, qkv_width, qkv_width + s.q_width)
        index_start = qkv_width + s.q_width
        index_q = self._slice_last(packed, index_start, index_start + s.indexer_n_heads * s.indexer_head_dim)
        raw_index_k = self._slice_last(
            packed,
            index_start + s.indexer_n_heads * s.indexer_head_dim,
            index_start + (s.indexer_n_heads + 1) * s.indexer_head_dim,
        )
        ttnn.deallocate(packed)

        if decode:
            # Height-sharded RMSNorm is binding-illegal, so Q/K cross to the
            # interleaved norm/RoPE path.  For batch one, V has no intervening
            # consumer and remains sharded through cache-update preparation.
            # The sharded pad kernel only supports one batch slice, so larger
            # contract batches take the legal interleaved fallback.
            qkv_l1 = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)
            q_sharded, k_sharded, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv_l1,
                num_heads=s.num_attention_heads,
                num_kv_heads=s.num_key_value_heads,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv_l1)
            q = ttnn.sharded_to_interleaved(q_sharded, ttnn.L1_MEMORY_CONFIG)
            k = ttnn.sharded_to_interleaved(k_sharded, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(q_sharded)
            ttnn.deallocate(k_sharded)
            if self.max_batch > 1:
                v_interleaved = ttnn.sharded_to_interleaved(v, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(v)
                v = v_interleaved
            q = ttnn.permute(q, (0, 2, 1, 3))
            k = ttnn.permute(k, (0, 2, 1, 3))
            index_q = ttnn.reshape(index_q, (batch, 1, s.indexer_n_heads, s.indexer_head_dim))
            index_q = ttnn.permute(index_q, (0, 2, 1, 3))
            raw_index_k = ttnn.reshape(raw_index_k, (1, batch, 1, s.indexer_head_dim))
        else:
            qkv_3d = ttnn.reshape(qkv, (1, tokens, qkv_width))
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv_3d,
                num_heads=s.num_attention_heads,
                num_kv_heads=s.num_key_value_heads,
                transpose_key=False,
            )
            _free(qkv_3d, qkv, q, k, v)
            index_q = ttnn.reshape(index_q, (1, tokens, s.indexer_n_heads, s.indexer_head_dim))
            index_q = ttnn.permute(index_q, (0, 2, 1, 3))
            raw_index_k = ttnn.reshape(raw_index_k, (1, tokens, 1, s.indexer_head_dim))
        ttnn.deallocate(qkv)

        q = self._rms_norm(q, self.w["q_norm"], s.rms_norm_eps)
        k = self._rms_norm(k, self.w["k_norm"], s.rms_norm_eps)
        index_q = self._rms_norm(index_q, self.w["index_q_norm"], s.rms_norm_eps)
        cos, sin = self._rotation_rows(rot_mats, positions, batch=batch, tokens=tokens)
        if decode:
            decode_cos = ttnn.reshape(cos, (1, 1, batch, s.rotary_dim))
            decode_sin = ttnn.reshape(sin, (1, 1, batch, s.rotary_dim))
            q = self._apply_rope(q, decode_cos, decode_sin, s.rotary_dim)
            k = self._apply_rope(k, decode_cos, decode_sin, s.rotary_dim)
            q = ttnn.permute(q, (2, 1, 0, 3))
            k = ttnn.permute(k, (0, 2, 1, 3))
        else:
            q = self._apply_rope(q, cos, sin, s.rotary_dim)
            k = self._apply_rope(k, cos, sin, s.rotary_dim)
        index_q = self._apply_rope(index_q, cos, sin, s.rotary_dim)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        return q, k, v, gate, index_q, raw_index_k

    def _ensure_static_block_rope(self, rot_mats):
        """Materialize invariant compressed-block RoPE rows once per table pair."""

        s = self.shapes
        count = self.const["compressed_blocks"]
        rot_source = tuple(tensor.buffer_address() for tensor in rot_mats)
        if self.fused_block_rot_source == rot_source:
            return
        freed_addresses = set()
        for cached in (
            self.fused_block_cos,
            self.fused_block_sin,
        ):
            if cached is not None and cached.is_allocated():
                address = cached.buffer_address()
                if address not in freed_addresses:
                    ttnn.deallocate(cached)
                    freed_addresses.add(address)
        block_positions = ttnn.reshape(self.fused_index_block_positions, (1, count))
        block_cos, block_sin = self._rotation_rows(
            rot_mats,
            block_positions,
            batch=1,
            tokens=count,
        )
        self.fused_block_cos = ttnn.to_memory_config(block_cos, ttnn.DRAM_MEMORY_CONFIG)
        self.fused_block_sin = ttnn.to_memory_config(block_sin, ttnn.DRAM_MEMORY_CONFIG)
        _free(block_cos, self.fused_block_cos)
        _free(block_sin, self.fused_block_sin)
        self.fused_block_rot_source = rot_source

    def _compressed_index_prefill(self, raw_heads, chunk_page_table, *, chunk_start: int, rot_mats):
        """Commit normalized/RoPE-applied four-token keys during prefill."""

        s = self.shapes
        length = int(raw_heads.shape[-2])
        cache_tokens = int(chunk_page_table.shape[-1]) * self.block_size
        if not 0 < cache_tokens <= length or cache_tokens % s.indexer_compress_ratio:
            raise ValueError(f"compressed QSA cache fill length {cache_tokens} is invalid for padded chunk {length}")
        cache_heads = raw_heads
        if cache_tokens != length:
            cache_heads = ttnn.slice(
                raw_heads,
                [0, 0, 0, 0],
                [int(raw_heads.shape[0]), int(raw_heads.shape[1]), cache_tokens, int(raw_heads.shape[-1])],
            )
        groups = cache_tokens // s.indexer_compress_ratio
        grouped = ttnn.reshape(cache_heads, (1, groups, s.indexer_compress_ratio, s.indexer_head_dim))
        keys = ttnn.mean(grouped, dim=2, keepdim=False)
        _free(grouped, cache_heads, keys)
        keys = ttnn.reshape(keys, (1, 1, groups, s.indexer_head_dim))
        keys = self._rms_norm(keys, self.w["index_k_norm"], s.rms_norm_eps)
        self._ensure_static_block_rope(rot_mats)
        first_group = chunk_start // s.indexer_compress_ratio
        cos = ttnn.slice(
            self.fused_block_cos,
            [0, 0, first_group, 0],
            [1, 1, first_group + groups, s.rotary_dim],
        )
        sin = ttnn.slice(
            self.fused_block_sin,
            [0, 0, first_group, 0],
            [1, 1, first_group + groups, s.rotary_dim],
        )
        keys = self._apply_rope(keys, cos, sin, s.rotary_dim)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        fill = ttnn.typecast(keys, self.fused_index_key_cache.dtype)
        ttnn.experimental.paged_fill_cache(
            self.fused_index_key_cache,
            fill,
            chunk_page_table,
            batch_idx=0,
        )
        _free(fill, keys)
        ttnn.deallocate(keys)
        _free(cache_heads, raw_heads)

    def _qsa_prefill(
        self,
        x,
        *,
        page_table,
        chunk_page_table,
        chunk_start: int,
        rot_mats,
    ):
        length = int(x.shape[-2])
        positions = ttnn.arange(chunk_start, chunk_start + length, 1, device=self.mesh_device, dtype=ttnn.int32)
        positions = ttnn.minimum(positions, self.max_seq_len - 1)
        positions = ttnn.reshape(positions, (1, length))
        q, k, v, gate, index_q, raw_index = self._qsa_projections(x, positions, rot_mats, decode=False)
        raw_heads = ttnn.permute(raw_index, (0, 2, 1, 3))
        cache_tokens = int(chunk_page_table.shape[-1]) * self.block_size
        if not 0 < cache_tokens <= length:
            raise ValueError(f"QSA cache fill length {cache_tokens} is outside padded chunk [1, {length}]")
        for cache, value in (
            (self.kv_cache[0], k),
            (self.kv_cache[1], v),
            (self.indexer_cache, raw_heads),
        ):
            cache_value = value
            if cache_tokens != length:
                cache_value = ttnn.slice(
                    value,
                    [0, 0, 0, 0],
                    [int(value.shape[0]), int(value.shape[1]), cache_tokens, int(value.shape[-1])],
                )
            fill = ttnn.typecast(cache_value, cache.dtype)
            ttnn.experimental.paged_fill_cache(cache, fill, chunk_page_table, batch_idx=0)
            _free(fill, cache_value)
            _free(cache_value, value)
        self._compressed_index_prefill(raw_heads, chunk_page_table, chunk_start=chunk_start, rot_mats=rot_mats)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        ttnn.deallocate(raw_index)
        _free(raw_heads, raw_index)
        selected, valid = self._selected_virtual_tokens(index_q, page_table, positions, rot_mats)
        ttnn.deallocate(index_q)
        attention = self._gathered_qsa_attention(q, selected, valid, page_table)
        ttnn.deallocate(q)
        ttnn.deallocate(positions)
        return self._qsa_epilogue(attention, gate, decode=False)

    def _compressed_index_decode(self, current_pos, page_table, rot_mats):
        """Update one rolling compressed key per user from the four raw taps."""

        s = self.shapes
        batch = self.max_batch
        group_ids = ttnn.bitwise_right_shift(current_pos, int(math.log2(s.indexer_compress_ratio)))
        group_starts = ttnn.multiply(group_ids, s.indexer_compress_ratio)
        group_starts = ttnn.reshape(group_starts, (batch, 1, 1, 1))
        group_offsets = ttnn.reshape(self.const["index_token_offsets"], (1, 1, 1, s.indexer_compress_ratio))
        virtual = ttnn.add(group_starts, group_offsets)
        ttnn.deallocate(group_starts)
        _free(group_offsets, self.const["index_token_offsets"], virtual)
        physical = self._virtual_to_physical_tokens(virtual, page_table)
        ttnn.deallocate(virtual)
        physical = ttnn.reshape(physical, (1, batch * s.indexer_compress_ratio))
        raw_weight = ttnn.reshape(
            self.indexer_cache,
            (self.max_num_blocks * self.block_size, s.indexer_head_dim),
        )
        raw = _embedding_tiled_output(physical, raw_weight)
        _free(raw_weight, self.indexer_cache, raw)
        ttnn.deallocate(physical)
        raw = ttnn.reshape(raw, (batch, 1, s.indexer_compress_ratio, s.indexer_head_dim))
        keys = ttnn.mean(raw, dim=2, keepdim=False)
        ttnn.deallocate(raw)
        keys = ttnn.reshape(keys, (batch, 1, 1, s.indexer_head_dim))
        keys = self._rms_norm(keys, self.w["index_k_norm"], s.rms_norm_eps)
        self._ensure_static_block_rope(rot_mats)
        block_indices = ttnn.typecast(group_ids, ttnn.uint32)
        block_indices = ttnn.reshape(block_indices, (1, batch))
        cos_weight = ttnn.reshape(self.fused_block_cos, (self.const["compressed_blocks"], s.rotary_dim))
        sin_weight = ttnn.reshape(self.fused_block_sin, (self.const["compressed_blocks"], s.rotary_dim))
        cos = _embedding_tiled_output(block_indices, cos_weight)
        sin = _embedding_tiled_output(block_indices, sin_weight)
        _free(cos_weight, self.fused_block_cos, cos)
        _free(sin_weight, self.fused_block_sin, sin)
        ttnn.deallocate(block_indices)
        cos = ttnn.reshape(cos, (batch, 1, 1, s.rotary_dim))
        sin = ttnn.reshape(sin, (batch, 1, 1, s.rotary_dim))
        keys = self._apply_rope(keys, cos, sin, s.rotary_dim)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        update = ttnn.permute(keys, (2, 0, 1, 3))
        ttnn.deallocate(keys)
        update = ttnn.pad(update, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
        update = ttnn.to_memory_config(update, self.decode_index_mem_cfg)
        ttnn.experimental.paged_update_cache(
            self.fused_index_key_cache,
            update,
            update_idxs_tensor=group_ids,
            page_table=page_table,
        )
        ttnn.deallocate(update)
        ttnn.deallocate(group_ids)

    def _qsa_decode(self, x, *, current_pos, page_table, rot_mats):
        s = self.shapes
        positions = ttnn.reshape(current_pos, (self.max_batch, 1))
        q, k, v, gate, index_q, raw_index = self._qsa_projections(x, positions, rot_mats, decode=True)

        def padded_heads(value, mem_cfg):
            heads = int(value.shape[2])
            if heads < 32:
                value = ttnn.pad(value, [(0, 0), (0, 0), (0, 32 - heads), (0, 0)], 0.0)
            return ttnn.to_memory_config(value, mem_cfg)

        k_update = padded_heads(k, self.decode_head_mem_cfg)
        v_update = padded_heads(v, self.fused_decode_value_mem_cfg)
        raw_update = padded_heads(raw_index, self.decode_index_mem_cfg)
        ttnn.experimental.paged_fused_update_cache(
            self.kv_cache[0],
            k_update,
            self.kv_cache[1],
            v_update,
            update_idxs_tensor=current_pos,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            self.indexer_cache,
            raw_update,
            update_idxs_tensor=current_pos,
            page_table=page_table,
        )
        ttnn.deallocate(k_update)
        ttnn.deallocate(v_update)
        ttnn.deallocate(raw_update)
        for tensor in (k, v, raw_index):
            _free(tensor, k_update, v_update, raw_update)

        self._compressed_index_decode(current_pos, page_table, rot_mats)

        selected, valid = self._selected_virtual_tokens(index_q, page_table, positions, rot_mats)
        ttnn.deallocate(index_q)
        attention = self._gathered_qsa_attention(q, selected, valid, page_table)
        ttnn.deallocate(q)
        return self._qsa_epilogue(attention, gate, decode=True)

    def _physical_compressed_ids(self, page_table, *, batch: int, tokens: int):
        """Map compressed blocks with a single integer ternary affine op."""

        s = self.shapes
        count = self.const["compressed_blocks"]
        per_page = self.block_size // s.indexer_compress_ratio
        virtual_pages = self.fused_index_virtual_pages
        in_page = self.fused_index_in_page
        pages = ttnn.reshape(page_table, (batch, 1, 1, int(page_table.shape[-1])))
        if tokens > 1:
            pages = ttnn.repeat(pages, (1, 1, 32, 1))
        pages = ttnn.to_layout(pages, ttnn.TILE_LAYOUT)
        if batch > 1:
            virtual_pages = ttnn.repeat(virtual_pages, (batch, 1, 1, 1))
            in_page = ttnn.repeat(in_page, (batch, 1, 1, 1))
        index = ttnn.typecast(virtual_pages, ttnn.uint32)
        physical_pages = ttnn.gather(pages, dim=-1, index=index)
        _free(pages, page_table, physical_pages)
        _free(virtual_pages, self.fused_index_virtual_pages)
        ttnn.deallocate(index)
        physical = ttnn.addcmul(
            in_page,
            physical_pages,
            self.fused_index_scalar_one,
            value=per_page,
        )
        ttnn.deallocate(physical_pages)
        _free(in_page, self.fused_index_in_page)
        return ttnn.reshape(physical, (batch, count))

    def _selected_virtual_tokens(self, index_q, page_table, current_pos, rot_mats):
        """QSA indexer with its score projection subgraph fused."""

        s = self.shapes
        batch, _, tokens, _ = _shape(index_q)
        count = self.const["compressed_blocks"]
        per_page = self.block_size // s.indexer_compress_ratio

        key_weight = ttnn.reshape(
            self.fused_index_key_cache,
            (self.max_num_blocks * per_page, s.indexer_head_dim),
        )
        physical_ids = self._physical_compressed_ids(page_table, batch=batch, tokens=tokens)
        physical_ids = ttnn.typecast(physical_ids, ttnn.uint32)
        physical_ids = ttnn.reshape(physical_ids, (1, batch * count))
        keys = _embedding_tiled_output(physical_ids, key_weight)
        _free(key_weight, self.fused_index_key_cache, keys)
        ttnn.deallocate(physical_ids)
        keys = ttnn.reshape(keys, (batch, 1, count, s.indexer_head_dim))
        if batch > 1:
            # The default matmul broadcasts the one key head only for the
            # batch-one fast path. With a real batch prefix it requires an
            # explicit matching index-head dimension.
            repeated_keys = ttnn.repeat(keys, (1, s.indexer_n_heads, 1, 1))
            _free(keys, repeated_keys)
            keys = repeated_keys
        scores = ttnn.matmul(
            index_q,
            keys,
            transpose_b=True,
            activation="relu",
            compute_kernel_config=self.compute_cfg,
        )
        ttnn.deallocate(keys)
        scores = ttnn.sum(scores, dim=1, keepdim=True)

        pos = ttnn.reshape(current_pos, (batch, 1, tokens, 1))
        complete = ttnn.bitwise_right_shift(
            ttnn.add(pos, 1),
            int(math.log2(s.indexer_compress_ratio)),
        )
        block_ids = self.const["index_block_ids"]
        visible = ttnn.lt(block_ids, complete)
        scores = ttnn.where(visible, scores, -1.0e4)
        ttnn.deallocate(visible)
        selected_scores, selected_blocks = ttnn.topk(scores, k=QSA_BLOCK_TOPK, dim=-1, sorted=True)
        ttnn.deallocate(selected_scores)
        ttnn.deallocate(scores)

        narrow_selected_blocks = selected_blocks
        selected_blocks = ttnn.typecast(narrow_selected_blocks, ttnn.int32)
        ttnn.deallocate(narrow_selected_blocks)
        selected_blocks = ttnn.reshape(selected_blocks, (batch, 1, tokens, QSA_BLOCK_TOPK, 1))
        offsets = self.const["index_token_offsets"]
        selected = ttnn.addcmul(
            offsets,
            selected_blocks,
            self.fused_index_offset_ones,
            value=s.indexer_compress_ratio,
        )
        ttnn.deallocate(selected_blocks)
        selected = ttnn.reshape(selected, (batch, 1, tokens, s.indexer_budget))
        complete_tokens = ttnn.multiply(complete, s.indexer_compress_ratio)
        selected_valid = ttnn.lt(selected, complete_tokens, dtype=ttnn.bfloat16)
        tail_base = ttnn.reshape(complete_tokens, (batch, 1, tokens, 1, 1))
        tail = ttnn.add(tail_base, offsets)
        _free(tail_base, complete_tokens, tail)
        ttnn.deallocate(complete_tokens)
        ttnn.deallocate(complete)
        tail = ttnn.reshape(tail, (batch, 1, tokens, s.indexer_compress_ratio))
        tail_valid = ttnn.le(tail, pos, dtype=ttnn.bfloat16)
        complete_selected = selected
        selected = ttnn.concat([selected, tail], dim=-1)
        _free(complete_selected, selected)
        valid = ttnn.concat([selected_valid, tail_valid], dim=-1)
        ttnn.deallocate(selected_valid)
        ttnn.deallocate(tail_valid)
        ttnn.deallocate(tail)
        if int(selected.shape[-1]) != self.const["gathered_width"]:
            unpadded_selected = selected
            selected = ttnn.pad(
                unpadded_selected,
                [(0, 0), (0, 0), (0, 0), (0, self.const["gathered_width"] - int(selected.shape[-1]))],
                0,
            )
            _free(unpadded_selected, selected)
            unpadded_valid = valid
            valid = ttnn.pad(
                unpadded_valid,
                [(0, 0), (0, 0), (0, 0), (0, self.const["gathered_width"] - int(valid.shape[-1]))],
                0,
            )
            _free(unpadded_valid, valid)
        unclamped_selected = selected
        selected = ttnn.minimum(unclamped_selected, self.max_seq_len - 1)
        _free(unclamped_selected, selected)
        return selected, valid

    def _virtual_to_physical_tokens(self, virtual_tokens, page_table):
        """Map gathered virtual tokens with one integer ternary affine op."""

        batch, _, tokens, width = _shape(virtual_tokens)
        virtual_page = ttnn.bitwise_right_shift(virtual_tokens, int(math.log2(self.block_size)))
        in_page = ttnn.bitwise_and(virtual_tokens, self.block_size - 1)
        pages = ttnn.reshape(page_table, (batch, 1, 1, int(page_table.shape[-1])))
        if tokens > 1:
            pages = ttnn.repeat(pages, (1, 1, _round_up(tokens, 32), 1))
        pages = ttnn.to_layout(pages, ttnn.TILE_LAYOUT)
        index = ttnn.typecast(virtual_page, ttnn.uint32)
        physical_page = ttnn.gather(pages, dim=-1, index=index)
        _free(pages, page_table, physical_page)
        ttnn.deallocate(virtual_page)
        ttnn.deallocate(index)
        physical = ttnn.addcmul(
            in_page,
            physical_page,
            self.fused_index_scalar_one,
            value=self.block_size,
        )
        ttnn.deallocate(physical_page)
        ttnn.deallocate(in_page)
        return ttnn.typecast(ttnn.reshape(physical, (batch, tokens, width)), ttnn.uint32)

    def _gathered_qsa_attention(self, q, selected, valid, page_table):
        """Gather token-specific pages, then use one dedicated SDPA op."""

        s = self.shapes
        batch, _, tokens, _ = _shape(q)
        width = self.const["gathered_width"]
        physical = self._virtual_to_physical_tokens(selected, page_table)
        ttnn.deallocate(selected)
        physical = ttnn.reshape(physical, (1, batch * tokens * width))

        gathered = []
        for cache in self.kv_cache:
            # Gathering the packed [2, 256] KV value and then reshaping puts
            # the two-head dimension in a tiled Y axis.  TTNN pads that axis
            # to 32 and copies a 16x larger intermediate.  Gather each cache
            # head independently, keeping width/head_dim as the tiled axes,
            # then concatenate heads in a non-tiled dimension.
            gathered_heads = []
            cache_shape = _shape(cache)
            for head in range(s.num_key_value_heads):
                cache_head = ttnn.slice(
                    cache,
                    [0, head, 0, 0],
                    [cache_shape[0], head + 1, cache_shape[2], cache_shape[3]],
                )
                flattened = ttnn.reshape(cache_head, (self.max_num_blocks * self.block_size, s.head_dim))
                _free(cache_head, cache, flattened)
                value = _embedding_tiled_output(physical, flattened)
                _free(flattened, cache, value)
                expected_volume = batch * tokens * width * s.head_dim
                if math.prod(_shape(value)) != expected_volume:
                    raise RuntimeError(
                        f"paged cache embedding returned {_shape(value)} (volume {math.prod(_shape(value))}); "
                        f"expected logical volume {expected_volume} for "
                        f"[{batch}, {tokens}, 1, {width}, {s.head_dim}]"
                    )
                head_rows = ttnn.reshape(value, (batch, tokens, 1, width, s.head_dim))
                _free(value, head_rows)
                gathered_heads.append(head_rows)
            gathered_value = ttnn.concat(gathered_heads, dim=2)
            for head_rows in gathered_heads:
                _free(head_rows, gathered_value)
            gathered.append(gathered_value)
        ttnn.deallocate(physical)
        k, v = gathered

        query_batches = batch * tokens
        if tokens == 1:
            q_rows = ttnn.reshape(q, (query_batches, s.num_attention_heads, 1, s.head_dim))
        else:
            query_tokens = ttnn.permute(q, (0, 2, 1, 3))
            q_rows = ttnn.reshape(query_tokens, (query_batches, s.num_attention_heads, 1, s.head_dim))
            _free(query_tokens, q, q_rows)
        k_rows = ttnn.reshape(k, (query_batches, s.num_key_value_heads, width, s.head_dim))
        v_rows = ttnn.reshape(v, (query_batches, s.num_key_value_heads, width, s.head_dim))
        additive = ttnn.where(valid, 0.0, -1.0e4)
        ttnn.deallocate(valid)
        if tokens > 1:
            token_mask = ttnn.permute(additive, (0, 2, 1, 3))
            ttnn.deallocate(additive)
            additive = ttnn.reshape(token_mask, (query_batches, 1, 1, width))
            _free(token_mask, additive)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=0 if tokens == 1 else 32,
            k_chunk_size=32 if tokens == 1 else 64,
            exp_approx_mode=False,
        )
        if tokens == 1:
            decode_q = ttnn.permute(q_rows, (2, 0, 1, 3))
            decode_mask = ttnn.repeat(additive, (1, 1, s.num_attention_heads, 1))
            out = ttnn.transformer.scaled_dot_product_attention_decode(
                decode_q,
                k_rows,
                v_rows,
                is_causal=False,
                attn_mask=decode_mask,
                scale=1.0 / math.sqrt(s.head_dim),
                program_config=sdpa_cfg,
                compute_kernel_config=self.compute_cfg,
            )
            ttnn.deallocate(decode_q)
            ttnn.deallocate(decode_mask)
        else:
            out = ttnn.transformer.scaled_dot_product_attention(
                q_rows,
                k_rows,
                v_rows,
                is_causal=False,
                attn_mask=additive,
                scale=1.0 / math.sqrt(s.head_dim),
                program_config=sdpa_cfg,
                compute_kernel_config=self.compute_cfg,
            )
        _free(q_rows, q, out)
        ttnn.deallocate(k_rows)
        ttnn.deallocate(v_rows)
        ttnn.deallocate(additive)
        token_out = ttnn.reshape(out, (batch, tokens, s.num_attention_heads, s.head_dim))
        _free(out, token_out)
        return token_out

    def _qsa_epilogue(self, attention, gate, *, decode: bool):
        s = self.shapes
        rows = self.max_batch if decode else int(attention.shape[1])
        flattened = ttnn.reshape(attention, (1, 1, rows, s.q_width))
        _free(attention, flattened)
        attention = flattened
        gated = ttnn.multiply(
            attention,
            gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        ttnn.deallocate(attention)
        ttnn.deallocate(gate)
        out = self._linear(gated, self.w["attn_out"])
        ttnn.deallocate(gated)
        return out


__all__ = ["FusedDecoder"]

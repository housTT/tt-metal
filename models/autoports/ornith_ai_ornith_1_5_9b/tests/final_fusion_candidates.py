# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Final coherent graph and direct normalization-to-DRAM handoff control."""

import ttnn

from . import linear_fusion_candidates as L
from .attention_fusion_candidates import SliceL1Attention
from .mlp_fusion_candidates import PackedPrefillSeparateDecode
from .mode_fusion_candidates import ModeNormGDN


class NormDRAMGDN(ModeNormGDN):
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
            norm = ttnn.multiply(rms, dk**-0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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


class FinalCombination(NormDRAMGDN, SliceL1Attention, PackedPrefillSeparateDecode):
    """Selected linear/attention paths plus packed-prefill/separate-decode MLP."""


class ConvFinalCombination(FinalCombination, L.Conv1d1024GDN):
    """Ordinary Conv1d1024 plus gated norm, decode fusions and MLP packing."""

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


class NormL1InputGDN(NormDRAMGDN):
    """Let the existing TensorAccessor consume small normalized Q/K directly in L1."""

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


class PostConcatSiluConv(ConvFinalCombination):
    """Merge peer SiLU launches after concatenating each Q/K/V channel field."""

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
            parts.append(flattened)
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
        activated = [ttnn.silu(field, memory_config=ttnn.DRAM_MEMORY_CONFIG) for field in fields]
        for field in fields:
            ttnn.deallocate(field)
        q, k, v = activated
        tail_rm = ttnn.slice(padded, [0, logical_len, 0], [batch, logical_len + 3, cfg.conv_dim])
        tail = tail_rm
        ttnn.deallocate(padded)
        return q, k, v, tail

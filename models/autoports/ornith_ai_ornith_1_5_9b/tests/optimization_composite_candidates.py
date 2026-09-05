# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native composite-op probes, isolated from production and resource repairs."""

import os

import torch

import ttnn

from .optimization_extra_candidates import (
    CombinedKDAPrefillCandidate,
    CombinedOutputCandidate,
    CombinedPrefillCandidate,
    ProjectionOutputCandidate,
)


class NativeDeltaDecodeCandidate(CombinedOutputCandidate):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            constants = [torch.eye(32), torch.tril(torch.ones(32, 32)), torch.ones(32, 32)]
            decoder.decode_constants = [
                ttnn.from_torch(
                    x.reshape(1, 1, 32, 32),
                    device=decoder.device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
                )
                for x in constants
            ]
        return decoder

    def _delta_rule_step(self, q, k, v, beta, g):
        batch, nv, dk = q.shape[0], self.cfg.linear_num_value_heads, self.cfg.linear_key_head_dim
        q = ttnn.multiply(q, dk**-1.0, dtype=ttnn.float32)
        k = ttnn.unary_chain(k, self.key_scale_chain, memory_config=ttnn.L1_MEMORY_CONFIG)
        fields = []
        for value in (q, k, v):
            token_major = ttnn.permute(value, (0, 2, 1, 3))
            fields.append(ttnn.pad(token_major, [(0, 0), (0, 31), (0, 0), (0, 0)], 0.0))
        beta = ttnn.pad(beta, [(0, 0), (0, 31), (0, 0)], 0.0)
        g = ttnn.pad(g, [(0, 0), (0, 31), (0, 0)], 0.0)
        eye, tril, ones = self.decode_constants
        out, final = ttnn.transformer.chunk_gated_delta_rule(
            *fields,
            g,
            beta,
            initial_state=self.recurrent_state,
            output_final_state=True,
            chunk_size=32,
            scale=1.0,
            use_qk_l2norm=False,
            output_head_major=True,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=self.w["gdn_const_tiles"][3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy(final, self.recurrent_state)
        ttnn.deallocate(final)
        first = ttnn.slice(out, [0, 0, 0], [batch * nv, 1, dk])
        ttnn.deallocate(out)
        return ttnn.reshape(first, [batch, nv, 1, dk])


class PrefillChunkComputeCandidate(CombinedOutputCandidate):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.prefill_chunk_compute = ttnn.init_device_compute_kernel_config(
            decoder.device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("ORNITH_CHUNK_FIDELITY", "HiFi2")),
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            packer_l1_acc=False,
        )
        decoder.prefill_chunk_memory = (
            ttnn.L1_MEMORY_CONFIG if os.environ.get("ORNITH_CHUNK_MEMORY", "dram") == "l1" else ttnn.DRAM_MEMORY_CONFIG
        )
        return decoder

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
                compute_kernel_config=self.prefill_chunk_compute,
                memory_config=self.prefill_chunk_memory,
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


class CombinedOutputPrefillCandidate(ProjectionOutputCandidate, CombinedPrefillCandidate):
    pass


class CombinedOutputKDAPrefillCandidate(ProjectionOutputCandidate, CombinedKDAPrefillCandidate):
    pass


class PrefillInputL1Candidate(CombinedOutputPrefillCandidate):
    def _linear(self, x, role, **kwargs):
        roles = os.environ.get(
            "ORNITH_PREFILL_L1_ROLES", "gdn_packed,gdn_z_epilogue,gdn_out,qkvg,o_proj,gate_up,down_proj"
        ).split(",")
        if x.shape[1] > 1 and role in roles:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        return super()._linear(x, role, **kwargs)


class RecurrentOuterBroadcastCandidate(CombinedOutputCandidate):
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
        key_column = ttnn.transpose(k_row, -2, -1)
        outer = ttnn.multiply(key_column, delta, memory_config=dram)
        ttnn.deallocate(key_column)
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


class TiledHeadSplitCandidate(CombinedOutputCandidate):
    def _split_heads(self, tensor, heads, head_dim):
        return ttnn.reshape(tensor, [tensor.shape[0], tensor.shape[1], heads, head_dim])


class NativeFlatPrefillCandidate(CombinedOutputCandidate):
    def _chunk_delta_rule(self, q, k, v, g, beta):
        from .linear_fusion_candidates import FlatGDN

        assert self.w["gdn_chunk_size"] == 32
        return FlatGDN._chunk_delta_rule(self, q, k, v, g, beta)


class ChunkL1AdaptedCandidate(PrefillChunkComputeCandidate):
    def prefill_forward(self, x, *, chunk_size=None, **kwargs):
        return super().prefill_forward(
            x, chunk_size=chunk_size or int(os.environ.get("ORNITH_CHUNK_PHYSICAL", "512")), **kwargs
        )


class FlatOutputPrefillCandidate(NativeFlatPrefillCandidate, CombinedOutputPrefillCandidate):
    pass


class FlatOutputKDAPrefillCandidate(NativeFlatPrefillCandidate, CombinedOutputKDAPrefillCandidate):
    pass


class SeparatePrefillMLPCandidate(FlatOutputPrefillCandidate):
    def _activate_mlp(self, ff_in, mode):
        if mode != "prefill":
            return super()._activate_mlp(ff_in, mode)
        gate = self._linear(ff_in, "gate_proj")
        up = self._linear(ff_in, "up_proj")
        result = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return result


class SeparateMLPKDAPrefillCandidate(SeparatePrefillMLPCandidate):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.kda_conv_program = ttnn.QkvCausalConv1dSiluProgramConfig(
            channel_chunk_size=int(os.environ.get("ORNITH_CONV_CHUNK", "512"))
        )
        return decoder

    def _gdn_prefill_conv_fields(self, qkv, logical_len):
        from .linear_fusion_candidates import _kda_prefill_fields_with_rm_tail

        return _kda_prefill_fields_with_rm_tail(self, qkv, logical_len)

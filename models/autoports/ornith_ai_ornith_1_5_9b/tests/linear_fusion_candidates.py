# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unverified linear-attention fusion experiments; no production selection logic.

Each class adds one hypothesis to the named parent. Device experiments and
promotion are owned by the stage coordinator. No candidate changes precision
policy or matmul program configuration.
"""

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import l2_norm_ttnn

from ..tt.functional_decoder import _slice_owned
from .fusion_baseline import FusionBaseline as FusedDecoder


def _field(tensor, start, end):
    return ttnn.slice(tensor, [0, 0, start], [tensor.shape[0], tensor.shape[1], end])


class FlatGDN(FusedDecoder):
    """H1/H2: raw flat prefill QKV and head-major core/output normalization."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            if decoder.cfg.linear_key_head_dim != decoder.cfg.linear_value_head_dim:
                raise ValueError("flat GDN requires equal key and value head dimensions")
            if decoder.w["gdn_chunk_size"] != 32:
                raise ValueError("flat GDN requires the existing 32-token constant tiles")
        return decoder

    def _gdn_project(self, x):
        return tuple(
            ttnn.linear(x, self.w[name], compute_kernel_config=self.compute_kernel_config)
            for name in ("gdn_qkv", "gdn_z", "gdn_a", "gdn_b")
        )

    def _gdn_gates_projected(self, a_raw, b_raw, logical_len, seq_len):
        # Preserve the functional rounding boundaries: sigmoid(BF16) -> FP32,
        # while A is converted to FP32 before bias and softplus.
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

    def _gdn_prefill_conv_fields(self, qkv, logical_len):
        activated, tail = self._causal_conv(qkv, logical_len)
        cfg = self.cfg
        q = _field(activated, 0, cfg.linear_q_dim)
        k = _field(activated, cfg.linear_q_dim, cfg.linear_q_dim + cfg.linear_k_dim)
        v = _field(activated, cfg.linear_q_dim + cfg.linear_k_dim, cfg.conv_dim)
        ttnn.deallocate(activated)
        return q, k, v, tail

    def _chunk_delta_rule(self, q, k, v, g, beta):
        cfg = self.cfg
        batch, seq = q.shape[0], q.shape[1]
        if seq % 32:
            raise ValueError("flat GDN requires a physically tile-aligned sequence")
        eye, tril, ones, masks = self.w["gdn_const_tiles"]

        def launch(q_, k_, v_, g_, beta_, state_):
            return ttnn.transformer.chunk_gated_delta_rule(
                q_,
                k_,
                v_,
                g_,
                beta_,
                initial_state=state_,
                output_final_state=True,
                chunk_size=32,
                use_qk_l2norm=False,
                output_head_major=True,
                eye=eye,
                tril=tril,
                ones=ones,
                masks=masks,
            )

        step = self.max_gdn_prefill_batch()
        if batch <= step:
            return launch(q, k, v, g, beta, self.recurrent_state)
        cores, states = [], []
        for start in range(0, batch, step):
            end = min(start + step, batch)
            args = [ttnn.slice(tensor, [start, 0, 0], [end, seq, tensor.shape[-1]]) for tensor in (q, k, v, g, beta)]
            state = ttnn.slice(
                self.recurrent_state,
                [start, 0, 0, 0],
                [end, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim],
            )
            core, final = launch(*args, state)
            for tensor in (*args, state):
                ttnn.deallocate(tensor)
            cores.append(core)
            states.append(final)
        output = ttnn.concat(cores, dim=0)
        final = ttnn.concat(states, dim=0)
        for tensor in cores + states:
            ttnn.deallocate(tensor)
        return output, final

    def _gdn_out_head_major(self, core, z, batch, seq):
        cfg = self.cfg
        normed = ttnn.rms_norm(core, weight=self.w["gdn_norm"], epsilon=cfg.norm_eps)
        ttnn.deallocate(core)
        heads = ttnn.reshape(normed, [batch, cfg.linear_num_value_heads, seq, cfg.linear_value_head_dim])
        if seq > 1:
            combined = ttnn.experimental.nlp_concat_heads(heads)
        else:
            combined = ttnn.permute(heads, (0, 2, 1, 3))
        # heads may alias normed; release the allocation only once.
        ttnn.deallocate(normed)
        merged = ttnn.reshape(combined, [batch, seq, cfg.linear_v_dim])
        activated = ttnn.silu(z)
        gated = ttnn.multiply(merged, activated)
        ttnn.deallocate(activated)
        ttnn.deallocate(combined)
        result = ttnn.linear(gated, self.w["gdn_out"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(gated)
        return result

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


class PackedGDN(FlatGDN):
    """H4: merge QKV/Z/A/B projections; decode layout stays functional."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        if decoder.is_full_attention:
            return decoder
        dtype = kwargs.get("dtype", ttnn.bfloat16)
        packed = torch.cat([state_dict[f"linear_attn.in_proj_{field}.weight"] for field in ("qkv", "z", "a", "b")])
        decoder.w["gdn_packed"] = ttnn.from_torch(
            packed.T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=decoder.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
        )
        # All packed methods below consume gdn_packed, including decode.
        for key in ("gdn_qkv", "gdn_z", "gdn_a", "gdn_b"):
            ttnn.deallocate(decoder.w.pop(key))
        return decoder

    def _gdn_project(self, x):
        cfg = self.cfg
        packed = ttnn.linear(x, self.w["gdn_packed"], compute_kernel_config=self.compute_kernel_config)
        ends = [0, cfg.conv_dim, cfg.conv_dim + cfg.linear_v_dim]
        ends.extend([ends[-1] + cfg.linear_num_value_heads, ends[-1] + 2 * cfg.linear_num_value_heads])
        fields = tuple(_field(packed, start, end) for start, end in zip(ends, ends[1:]))
        ttnn.deallocate(packed)
        return fields

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

    def _gdn_out_token_major(self, core, z):
        # Keep the functional decode layout in this projection-only experiment.
        z_heads = self._split_heads(z, self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim)
        normed = ttnn.rms_norm(core, weight=self.w["gdn_norm"], epsilon=self.cfg.norm_eps)
        ttnn.deallocate(core)
        activated = ttnn.silu(z_heads)
        gated = ttnn.multiply(normed, activated)
        for tensor in (normed, activated, z_heads):
            ttnn.deallocate(tensor)
        merged = self._merge_heads(gated)
        ttnn.deallocate(gated)
        result = ttnn.linear(merged, self.w["gdn_out"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(merged)
        return result

    def _gdn_decode(self, x):
        cfg = self.cfg
        qkv, z, a, b = self._gdn_project(x)
        activated = self._gdn_conv_decode(qkv)
        ttnn.deallocate(qkv)
        q, k, v = self._gdn_split_qkv(activated)
        ttnn.deallocate(activated)
        repeats = cfg.linear_num_value_heads // cfg.linear_num_key_heads
        if repeats > 1:
            q_expanded = ttnn.repeat_interleave(q, repeats, dim=2)
            k_expanded = ttnn.repeat_interleave(k, repeats, dim=2)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            q, k = q_expanded, k_expanded
        beta, g = self._gdn_gates_projected(a, b, 1, 1)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        core = self._delta_rule_step(q, k, v, beta, g)
        for tensor in (q, k, v, beta, g):
            ttnn.deallocate(tensor)
        core = ttnn.reshape(core, [x.shape[0], 1, cfg.linear_num_value_heads, cfg.linear_value_head_dim])
        result = self._gdn_out_token_major(core, z)
        ttnn.deallocate(z)
        return result


class SplitPackedGDN(PackedGDN):
    """H4 alternative: QKV+Z width 12288 and A+B width 64."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        # Build this subclass through FlatGDN, skipping PackedGDN's full-width
        # allocation and deletion of the individual projection weights.
        decoder = FlatGDN.from_state_dict.__func__(cls, state_dict, **kwargs)
        if decoder.is_full_attention:
            return decoder
        dtype = kwargs.get("dtype", ttnn.bfloat16)
        for key, fields in (("gdn_qkvz_packed", ("qkv", "z")), ("gdn_ab_packed", ("a", "b"))):
            packed = torch.cat([state_dict[f"linear_attn.in_proj_{field}.weight"] for field in fields])
            decoder.w[key] = ttnn.from_torch(
                packed.T.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=decoder.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
            )
        for key in ("gdn_qkv", "gdn_z", "gdn_a", "gdn_b"):
            ttnn.deallocate(decoder.w.pop(key))
        return decoder

    def _gdn_project(self, x):
        cfg = self.cfg
        qkvz = ttnn.linear(x, self.w["gdn_qkvz_packed"], compute_kernel_config=self.compute_kernel_config)
        ab = ttnn.linear(x, self.w["gdn_ab_packed"], compute_kernel_config=self.compute_kernel_config)
        qkv = _field(qkvz, 0, cfg.conv_dim)
        z = _field(qkvz, cfg.conv_dim, cfg.conv_dim + cfg.linear_v_dim)
        a = _field(ab, 0, cfg.linear_num_value_heads)
        b = _field(ab, cfg.linear_num_value_heads, 2 * cfg.linear_num_value_heads)
        ttnn.deallocate(qkvz)
        ttnn.deallocate(ab)
        return qkv, z, a, b


class DecodeLayoutGDN(PackedGDN):
    """H5: one packed decode relayout and adjacent Q/K head expansion."""

    fuse_exp = False
    fuse_transpose = False
    fuse_query_scale = False

    def _gdn_decode_heads(self, activated):
        cfg = self.cfg
        batch = activated.shape[0]
        nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        dk = cfg.linear_key_head_dim
        rows = ttnn.reshape(activated, [batch, 1, 2 * nk + nv, dk])
        heads = ttnn.permute(rows, (0, 2, 1, 3))
        # rows can share the input allocation; the caller owns activated.
        v = ttnn.slice(heads, [0, 2 * nk, 0, 0], [batch, 2 * nk + nv, 1, dk])
        qk = ttnn.slice(heads, [0, 0, 0, 0], [batch, 2 * nk, 1, dk])
        ttnn.deallocate(heads)
        repeats = nv // nk
        if repeats > 1:
            expanded = ttnn.repeat_interleave(qk, repeats, dim=1)
            ttnn.deallocate(qk)
            qk = expanded
        q = ttnn.slice(qk, [0, 0, 0, 0], [batch, nv, 1, dk])
        k = ttnn.slice(qk, [0, nv, 0, 0], [batch, 2 * nv, 1, dk])
        ttnn.deallocate(qk)
        return q, k, v

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

    def _delta_rule_step(self, q, k, v, beta, g):
        """Functional arithmetic on head-major tensors; class flags isolate H6."""
        cfg = self.cfg
        batch, nv, dk = q.shape[0], cfg.linear_num_value_heads, cfg.linear_key_head_dim
        dram = ttnn.DRAM_MEMORY_CONFIG
        if self.fuse_query_scale:
            q_norm = ttnn.rms_norm(q, epsilon=1e-6 / dk)
            q_scaled = ttnn.multiply(q_norm, dk**-1.0, memory_config=dram)
            ttnn.deallocate(q_norm)
            q_row = ttnn.typecast(q_scaled, ttnn.float32)
            ttnn.deallocate(q_scaled)
        else:
            q_norm = l2_norm_ttnn(q)
            q32 = ttnn.typecast(q_norm, ttnn.float32)
            ttnn.deallocate(q_norm)
            q_row = ttnn.multiply(q32, dk**-0.5, memory_config=dram)
            ttnn.deallocate(q32)
        k_norm = l2_norm_ttnn(k)
        k_row = ttnn.typecast(k_norm, ttnn.float32)
        ttnn.deallocate(k_norm)
        v_row = ttnn.typecast(v, ttnn.float32)
        beta_view = ttnn.reshape(beta, [batch, nv, 1, 1])
        g_view = ttnn.reshape(g, [batch, nv, 1, 1])
        state = self.recurrent_state
        if self.fuse_exp:
            ttnn.multiply(state, g_view, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], output_tensor=state)
        else:
            decay = ttnn.exp(g_view, memory_config=dram)
            ttnn.multiply(state, decay, output_tensor=state)
            ttnn.deallocate(decay)
        read = ttnn.matmul(k_row, state, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        difference = ttnn.subtract(v_row, read, memory_config=dram)
        delta = ttnn.multiply(difference, beta_view, memory_config=dram)
        for tensor in (read, difference, v_row):
            ttnn.deallocate(tensor)
        if self.fuse_transpose:
            outer = ttnn.matmul(
                k_row, delta, transpose_a=True, memory_config=dram, compute_kernel_config=self.compute_kernel_config
            )
        else:
            k_col = ttnn.transpose(k_row, -1, -2)
            outer = ttnn.matmul(k_col, delta, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
            ttnn.deallocate(k_col)
        ttnn.deallocate(k_row)
        ttnn.deallocate(delta)
        ttnn.add(state, outer, output_tensor=state)
        ttnn.deallocate(outer)
        result = ttnn.matmul(q_row, state, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(q_row)
        # beta_view and g_view may alias caller-owned gate buffers.
        return result


class ExpGDN(DecodeLayoutGDN):
    """H6a alone: EXP in the FP32 state multiply."""

    fuse_exp = True


class TransposeGDN(DecodeLayoutGDN):
    """H6b alone: transpose_a on the outer-product matmul."""

    fuse_transpose = True


class QueryScaleGDN(DecodeLayoutGDN):
    """H6c alone: fold the additional query scale into the L2 multiply."""

    fuse_query_scale = True


class ArithmeticGDN(DecodeLayoutGDN):
    """Combined H6 control; validate its three individual parents first."""

    fuse_exp = True
    fuse_transpose = True
    fuse_query_scale = True


class KDAConvGDN(PackedGDN):
    """H3: four-tap causal conv/SiLU/QKV split, one launch per user."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            if decoder.cfg.linear_conv_kernel_dim != 4:
                raise ValueError("KDA convolution requires four taps")
            if kwargs.get("dtype", ttnn.bfloat16) != ttnn.bfloat16:
                raise ValueError("KDA convolution requires the stage BF16 projection policy")
            decoder.kda_conv_program = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256)
        return decoder

    def _gdn_prefill_conv_fields(self, qkv, logical_len):
        cfg = self.cfg
        batch, seq = qkv.shape[0], qkv.shape[1]
        dram = ttnn.DRAM_MEMORY_CONFIG
        history_rows = [ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram) for row in self.conv_state]
        history = ttnn.concat(history_rows, dim=1)
        for row in history_rows:
            ttnn.deallocate(row)
        tokens = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        outputs = [[], [], []]
        for user in range(batch):
            user_tokens, tokens_owned = _slice_owned(tokens, [user, 0, 0], [user + 1, seq, cfg.conv_dim])
            user_history, history_owned = _slice_owned(history, [user, 0, 0], [user + 1, 3, cfg.conv_dim])
            fields = ttnn.experimental.kda.qkv_causal_conv1d_silu(
                user_tokens,
                user_history,
                *self.w["conv_taps"],
                cfg.linear_q_dim,
                cfg.linear_k_dim,
                cfg.linear_v_dim,
                program_config=self.kda_conv_program,
                memory_config=dram,
                compute_kernel_config=self.compute_kernel_config,
            )
            for destination, tensor in zip(outputs, fields):
                destination.append(tensor)
            if tokens_owned:
                ttnn.deallocate(user_tokens)
            if history_owned:
                ttnn.deallocate(user_history)
        if batch == 1:
            q, k, v = [parts[0] for parts in outputs]
        else:
            q, k, v = [ttnn.concat(parts, dim=0) for parts in outputs]
            for parts in outputs:
                for tensor in parts:
                    ttnn.deallocate(tensor)

        # Do not append padded prompt rows to history. Avoid concatenating the
        # complete prompt merely to recover its last three real rows.
        if logical_len >= 3:
            tail_rm = ttnn.slice(tokens, [0, logical_len - 3, 0], [batch, logical_len, cfg.conv_dim])
        else:
            old = ttnn.slice(history, [0, logical_len, 0], [batch, 3, cfg.conv_dim])
            new = ttnn.slice(tokens, [0, 0, 0], [batch, logical_len, cfg.conv_dim])
            tail_rm = ttnn.concat([old, new], dim=1)
            ttnn.deallocate(old)
            ttnn.deallocate(new)
        tail = ttnn.to_layout(tail_rm, ttnn.TILE_LAYOUT, memory_config=dram)
        for tensor in (tail_rm, history, tokens):
            ttnn.deallocate(tensor)
        return q, k, v, tail


def probe_linear_core_equivalence(functional, candidate, prompt, tokens, *, pcc_bar=0.995, amplitude_rtol=0.05):
    """Untimed real-weight H1/H6 probe, including every recurrent-state update.

    Pass two already-built linear decoders, a host prompt tensor, and at least
    32 distinct host token tensors. The coordinator owns the hardware lock.
    This diagnostic temporarily intercepts the pre-norm output boundary and
    performs host reads there; never invoke it inside capture or a benchmark.
    It returns per-step metrics and restores all intercepted methods in finally.
    State is reset initially and left advanced through the provided sequence.
    """
    import torch

    from . import test_functional_decoder as harness

    if functional.is_full_attention or candidate.is_full_attention:
        raise ValueError("this probe is only for linear attention")
    if len(tokens) < 32:
        raise ValueError("provide at least 32 decode tokens to test accumulated recurrence")
    captured = {"functional": [], "candidate": []}
    restored = []

    def install(decoder, label, method_name, head_major):
        original = getattr(decoder, method_name)
        had_override = method_name in decoder.__dict__

        def capture(core, *args, **kwargs):
            value = ttnn.to_torch(core).float().clone()
            if head_major:
                batch, seq = args[1], args[2]  # (z, batch, seq)
                value = value.reshape(batch, decoder.cfg.linear_num_value_heads, seq, decoder.cfg.linear_value_head_dim)
                value = value.permute(0, 2, 1, 3).contiguous()
            captured[label].append(value)
            return original(core, *args, **kwargs)

        setattr(decoder, method_name, capture)
        restored.append((decoder, method_name, original, had_override))

    def metrics(expected, actual):
        expected, actual = expected.double(), actual.double()
        assert expected.shape == actual.shape, (expected.shape, actual.shape)
        assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
        reference_norm = expected.norm().item()
        error_norm = (expected - actual).norm().item()
        equal = torch.equal(expected, actual)
        correlation = 1.0 if equal else harness.pcc(expected, actual)
        assert correlation >= pcc_bar, correlation
        ratio = actual.norm().item() / max(reference_norm, 1e-30)
        relative_l2 = error_norm / max(reference_norm, 1e-30)
        if reference_norm > 0:
            assert abs(ratio - 1.0) <= amplitude_rtol, ("amplitude mismatch", ratio)
        else:
            assert equal, "zero reference has nonzero candidate values"
        return {
            "pcc": correlation,
            "relative_l2": relative_l2,
            "norm_ratio": ratio if reference_norm > 0 else 1.0,
            "max_abs_error": (expected - actual).abs().max().item(),
        }

    results = []
    try:
        for label, decoder in (("functional", functional), ("candidate", candidate)):
            install(decoder, label, "_gdn_out", False)
            for method, head_major in (("_gdn_out_token_major", False), ("_gdn_out_head_major", True)):
                if hasattr(decoder, method):
                    install(decoder, label, method, head_major)
            decoder.reset_state()
        for step, host_input in enumerate([prompt, *tokens]):
            states = {}
            outputs = {}
            for label, decoder in (("functional", functional), ("candidate", candidate)):
                captured[label].clear()
                device_input = harness.to_device(decoder.device, host_input)
                output = decoder.prefill_forward(device_input) if step == 0 else decoder.decode_forward(device_input)
                outputs[label] = ttnn.to_torch(output)
                states[label] = harness._snapshot_state(decoder)
                ttnn.deallocate(output)
                ttnn.deallocate(device_input)
            assert len(captured["functional"]) == len(captured["candidate"]) > 0
            core_results = [metrics(a, b) for a, b in zip(captured["functional"], captured["candidate"])]
            state_results = {"rec": metrics(states["functional"]["rec"], states["candidate"]["rec"])}
            state_results["conv"] = [
                metrics(a, b) for a, b in zip(states["functional"]["conv"], states["candidate"]["conv"])
            ]
            results.append(
                {
                    "mode": "prefill" if step == 0 else "decode",
                    "step": step,
                    "cores": core_results,
                    "state": state_results,
                    "output": metrics(outputs["functional"], outputs["candidate"]),
                }
            )
    finally:
        for decoder, name, original, had_override in reversed(restored):
            if had_override:
                setattr(decoder, name, original)
            else:
                delattr(decoder, name)
    return {
        "candidate": type(candidate).__name__,
        "pcc_bar": pcc_bar,
        "amplitude_rtol": amplitude_rtol,
        "steps": results,
    }


class SoftplusGDN(ArithmeticGDN):
    """H8: FP32 bias add emits the default softplus activation."""

    def _gdn_gates_projected(self, a_raw, b_raw, logical_len, seq_len):
        beta16 = ttnn.sigmoid(b_raw)
        beta = ttnn.typecast(beta16, ttnn.float32)
        ttnn.deallocate(beta16)
        a32 = ttnn.typecast(a_raw, ttnn.float32)
        soft = ttnn.add(a32, self.w["dt_bias"], activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])
        ttnn.deallocate(a32)
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


class KdaNormGDN(ArithmeticGDN):
    """KDA sigmoid-gated norm plus flat multiply by z restores SiLU gating."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            decoder.w["kda_norm_vector"] = ttnn.from_torch(
                state_dict["linear_attn.norm.weight"].float().reshape(-1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=decoder.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
            )
        return decoder

    def _gdn_out_head_major(self, core, z, batch, seq):
        if seq == 1:
            return super()._gdn_out_head_major(core, z, batch, seq)
        sigmoid_gated = ttnn.experimental.kda.sigmoid_gated_rms_norm(
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
        gated = ttnn.multiply(sigmoid_gated, z)
        ttnn.deallocate(sigmoid_gated)
        result = ttnn.linear(gated, self.w["gdn_out"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(gated)
        return result


class MixedSiluGDN(ArithmeticGDN):
    """Recheck mixed FP32 norm output times input-activated BF16 z (operand B)."""

    gate_is_operand_a = False

    def _gdn_out_head_major(self, core, z, batch, seq):
        cfg = self.cfg
        normed = ttnn.rms_norm(core, weight=self.w["gdn_norm"], epsilon=cfg.norm_eps)
        ttnn.deallocate(core)
        heads = ttnn.reshape(normed, [batch, cfg.linear_num_value_heads, seq, cfg.linear_value_head_dim])
        combined = ttnn.experimental.nlp_concat_heads(heads) if seq > 1 else ttnn.permute(heads, (0, 2, 1, 3))
        ttnn.deallocate(normed)
        merged = ttnn.reshape(combined, [batch, seq, cfg.linear_v_dim])
        if self.gate_is_operand_a:
            gated = ttnn.multiply(z, merged, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.float32)
        else:
            gated = ttnn.multiply(merged, z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.float32)
        ttnn.deallocate(combined)
        result = ttnn.linear(gated, self.w["gdn_out"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(gated)
        return result


class MixedSiluAGDN(MixedSiluGDN):
    """Same dtype contract with activated BF16 z placed in operand A."""

    gate_is_operand_a = True


def _additional_arithmetic_step(decoder, q, k, v, beta, g, *, weighted_norm=False, rank_one=False):
    """ArithmeticGDN control with only the selected additional graph rewrite."""
    cfg = decoder.cfg
    batch, nv, dk = q.shape[0], cfg.linear_num_value_heads, cfg.linear_key_head_dim
    dram = ttnn.DRAM_MEMORY_CONFIG
    if weighted_norm:
        q16 = ttnn.rms_norm(q, epsilon=1e-6 / dk, weight=decoder.w["q_l2_scale"])
        k16 = ttnn.rms_norm(k, epsilon=1e-6 / dk, weight=decoder.w["k_l2_scale"])
    else:
        q_norm = ttnn.rms_norm(q, epsilon=1e-6 / dk)
        q16 = ttnn.multiply(q_norm, dk**-1.0, memory_config=dram)
        ttnn.deallocate(q_norm)
        k16 = l2_norm_ttnn(k)
    q_row = ttnn.typecast(q16, ttnn.float32)
    k_row = ttnn.typecast(k16, ttnn.float32)
    ttnn.deallocate(q16)
    ttnn.deallocate(k16)
    v_row = ttnn.typecast(v, ttnn.float32)
    beta_view = ttnn.reshape(beta, [batch, nv, 1, 1])
    g_view = ttnn.reshape(g, [batch, nv, 1, 1])
    state = decoder.recurrent_state
    ttnn.multiply(state, g_view, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], output_tensor=state)
    read = ttnn.matmul(k_row, state, memory_config=dram, compute_kernel_config=decoder.compute_kernel_config)
    difference = ttnn.subtract(v_row, read, memory_config=dram)
    delta = ttnn.multiply(difference, beta_view, memory_config=dram)
    for tensor in (read, difference, v_row):
        ttnn.deallocate(tensor)
    if rank_one:
        # Logical contraction dim is one: [B,H,K,1] * [B,H,1,V].
        # Ternary ROW_COL_BCAST supports this exact interleaved FP32 pattern;
        # unlike an unsupported composite fallback it honors output_tensor.
        k_col = ttnn.transpose(k_row, -1, -2)
        ttnn.addcmul(state, k_col, delta, output_tensor=state, memory_config=dram)
        ttnn.deallocate(k_col)
    else:
        outer = ttnn.matmul(
            k_row, delta, transpose_a=True, memory_config=dram, compute_kernel_config=decoder.compute_kernel_config
        )
        ttnn.add(state, outer, output_tensor=state)
        ttnn.deallocate(outer)
    ttnn.deallocate(k_row)
    ttnn.deallocate(delta)
    result = ttnn.matmul(q_row, state, memory_config=dram, compute_kernel_config=decoder.compute_kernel_config)
    ttnn.deallocate(q_row)
    return result


class RankOneGDN(ArithmeticGDN):
    """H6: broadcast rank-one addcmul updates the existing FP32 state buffer."""

    def _delta_rule_step(self, q, k, v, beta, g):
        return _additional_arithmetic_step(self, q, k, v, beta, g, rank_one=True)


class NormWeightGDN(ArithmeticGDN):
    """Fold q/k L2 scalar multiplication into fixed BF16 RMSNorm weights."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            dk = decoder.cfg.linear_key_head_dim
            # Query 1/K is exactly representable; key 1/sqrt(K) is rounded
            # by the BF16 weight format, so raw-state equivalence is required.
            for name, scale in (("q_l2_scale", dk**-1.0), ("k_l2_scale", dk**-0.5)):
                decoder.w[name] = ttnn.from_torch(
                    torch.full((1, 1, 1, dk), scale, dtype=torch.float32),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=decoder.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
                )
        return decoder

    def _delta_rule_step(self, q, k, v, beta, g):
        return _additional_arithmetic_step(self, q, k, v, beta, g, weighted_norm=True)


class MixedSiluFP32GDN(MixedSiluGDN):
    """Same-dtype control: cast the BF16 z projection to FP32 before folding SiLU."""

    def _gdn_out_head_major(self, core, z, batch, seq):
        cfg = self.cfg
        normed = ttnn.rms_norm(core, weight=self.w["gdn_norm"], epsilon=cfg.norm_eps)
        ttnn.deallocate(core)
        heads = ttnn.reshape(normed, [batch, cfg.linear_num_value_heads, seq, cfg.linear_value_head_dim])
        combined = ttnn.experimental.nlp_concat_heads(heads) if seq > 1 else ttnn.permute(heads, (0, 2, 1, 3))
        ttnn.deallocate(normed)
        merged = ttnn.reshape(combined, [batch, seq, cfg.linear_v_dim])
        z32 = ttnn.typecast(z, ttnn.float32)
        if self.gate_is_operand_a:
            gated = ttnn.multiply(z32, merged, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.float32)
        else:
            gated = ttnn.multiply(merged, z32, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.float32)
        ttnn.deallocate(z32)
        ttnn.deallocate(combined)
        result = ttnn.linear(gated, self.w["gdn_out"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(gated)
        return result


class MixedSiluFP32AGDN(MixedSiluFP32GDN):
    """Same FP32 control with the activated gate in operand A."""

    gate_is_operand_a = True


def probe_softplus_bias_fusion(decoder, a_raw):
    """Untimed operator localization on an actual BF16 projected A tensor.

    Compare standalone/fused softplus at identical FP32 a+bias, and report
    negative-tail zeros. For a boundary control supply a BF16 tensor chosen
    so a+dt_bias spans -12 through 12, especially both sides of +/-5.
    """
    import torch

    from . import test_functional_decoder as harness

    if a_raw.dtype != ttnn.bfloat16:
        raise ValueError("pass the real projection's BF16 A tensor")
    a32 = ttnn.typecast(a_raw, ttnn.float32)
    biased = ttnn.add(a32, decoder.w["dt_bias"])
    standalone = ttnn.softplus(biased)
    fused = ttnn.add(a32, decoder.w["dt_bias"], activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])
    x, separate, combined = [ttnn.to_torch(tensor).float() for tensor in (biased, standalone, fused)]
    for tensor in (a32, biased, standalone, fused):
        ttnn.deallocate(tensor)
    oracle = torch.nn.functional.softplus(x)
    assert torch.isfinite(separate).all() and torch.isfinite(combined).all()

    def metrics(value):
        relative_l2 = ((value - oracle).double().norm() / oracle.double().norm().clamp_min(1e-30)).item()
        return {
            "oracle_pcc": harness.pcc(oracle, value),
            "relative_l2": relative_l2,
            "max_abs_error": (value - oracle).abs().max().item(),
        }

    negative_tail = x < -5
    errors = (separate - combined).abs().flatten()
    indices = errors.topk(min(8, errors.numel())).indices
    return {
        "biased_min": x.min().item(),
        "biased_max": x.max().item(),
        "standalone": metrics(separate),
        "fused": metrics(combined),
        "fused_vs_standalone_pcc": harness.pcc(separate, combined),
        "negative_tail_count": int(negative_tail.sum()),
        "standalone_negative_tail_zeros": int((separate[negative_tail] == 0).sum()),
        "fused_negative_tail_zeros": int((combined[negative_tail] == 0).sum()),
        "largest_differences": [
            {
                "biased": x.flatten()[index].item(),
                "standalone": separate.flatten()[index].item(),
                "fused": combined.flatten()[index].item(),
                "torch": oracle.flatten()[index].item(),
            }
            for index in indices
        ],
    }


class CombinedGDN(KDAConvGDN, KdaNormGDN):
    """Combine verified controls: KDA conv/norm prefill and Arithmetic decode.

    Cooperative construction adds both KDA setup objects. C3 method order uses
    KDAConvGDN's prefill fields, KdaNormGDN's output method, and the inherited
    ArithmeticGDN/DecodeLayoutGDN recurrence and decode methods.
    """


class SeparatePrefillCombinedGDN(CombinedGDN):
    """Independent original projections for prefill, packed projection for decode."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            for field in ("qkv", "z", "a", "b"):
                decoder.w[f"gdn_prefill_{field}"] = ttnn.from_torch(
                    state_dict[f"linear_attn.in_proj_{field}.weight"].T.contiguous(),
                    dtype=kwargs.get("dtype", ttnn.bfloat16),
                    layout=ttnn.TILE_LAYOUT,
                    device=decoder.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
                )
        return decoder

    def _gdn_project(self, x):
        # Public prefill pads every physical chunk to at least128 tokens;
        # decode alone has logical/physical time1 at this internal boundary.
        if x.shape[1] == 1:
            return super()._gdn_project(x)
        return tuple(
            ttnn.linear(x, self.w[f"gdn_prefill_{field}"], compute_kernel_config=self.compute_kernel_config)
            for field in ("qkv", "z", "a", "b")
        )


def _kda_prefill_fields_with_rm_tail(decoder, qkv, logical_len):
    cfg = decoder.cfg
    batch, seq = qkv.shape[0], qkv.shape[1]
    dram = ttnn.DRAM_MEMORY_CONFIG
    history_rows = [ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram) for row in decoder.conv_state]
    history = ttnn.concat(history_rows, dim=1)
    for row in history_rows:
        ttnn.deallocate(row)
    tokens = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    outputs = [[], [], []]
    for user in range(batch):
        user_tokens, tokens_owned = _slice_owned(tokens, [user, 0, 0], [user + 1, seq, cfg.conv_dim])
        user_history, history_owned = _slice_owned(history, [user, 0, 0], [user + 1, 3, cfg.conv_dim])
        fields = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            user_tokens,
            user_history,
            *decoder.w["conv_taps"],
            cfg.linear_q_dim,
            cfg.linear_k_dim,
            cfg.linear_v_dim,
            program_config=decoder.kda_conv_program,
            memory_config=dram,
            compute_kernel_config=decoder.compute_kernel_config,
        )
        for destination, tensor in zip(outputs, fields):
            destination.append(tensor)
        if tokens_owned:
            ttnn.deallocate(user_tokens)
        if history_owned:
            ttnn.deallocate(user_history)
    if batch == 1:
        q, k, v = [parts[0] for parts in outputs]
    else:
        q, k, v = [ttnn.concat(parts, dim=0) for parts in outputs]
        for parts in outputs:
            for tensor in parts:
                ttnn.deallocate(tensor)
    if logical_len >= 3:
        tail_rm = ttnn.slice(tokens, [0, logical_len - 3, 0], [batch, logical_len, cfg.conv_dim])
    else:
        old = ttnn.slice(history, [0, logical_len, 0], [batch, 3, cfg.conv_dim])
        new = ttnn.slice(tokens, [0, 0, 0], [batch, logical_len, cfg.conv_dim])
        tail_rm = ttnn.concat([old, new], dim=1)
        ttnn.deallocate(old)
        ttnn.deallocate(new)
    ttnn.deallocate(history)
    ttnn.deallocate(tokens)
    return q, k, v, tail_rm


class CombinedRMTailGDN(CombinedGDN):
    """Remove the three-row tail's TILE round-trip before individual writes."""

    def _gdn_prefill_conv_fields(self, qkv, logical_len):
        return _kda_prefill_fields_with_rm_tail(self, qkv, logical_len)

    def _write_conv_state(self, tail_rm):
        for index, buffer in enumerate(self.conv_state):
            row_rm = ttnn.slice(tail_rm, [0, index, 0], [tail_rm.shape[0], index + 1, tail_rm.shape[2]])
            row = ttnn.to_layout(row_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.copy(row, buffer)
            ttnn.deallocate(row)
            ttnn.deallocate(row_rm)


class SeparatePrefillCombinedRMTailGDN(SeparatePrefillCombinedGDN, CombinedRMTailGDN):
    """Independent-prefill projection control plus direct row-major tail writes."""


class KDAConvDecodeGDN(CombinedRMTailGDN):
    """T=1 adaptation: pad conv input to T32, retain only the first output row.

    This measures the dedicated operator's minimum legal sequence. The 31 dummy
    rows never reach recurrence or persistent convolution history. The padding
    tensor is allocated once with the other state, before trace capture.
    """

    def allocate_state(self, batch_size):
        super().allocate_state(batch_size)
        if not self.is_full_attention:
            self.kda_decode_padding = ttnn.zeros(
                [batch_size, 31, self.cfg.conv_dim],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def _gdn_conv_decode(self, qkv):
        padded = ttnn.concat([qkv, self.kda_decode_padding], dim=1)
        fields = _kda_prefill_fields_with_rm_tail(self, padded, logical_len=1)
        ttnn.deallocate(padded)
        first_rows = [ttnn.slice(field, [0, 0, 0], [qkv.shape[0], 1, field.shape[2]]) for field in fields[:3]]
        activated = ttnn.concat(first_rows, dim=2)
        for tensor in (*fields[:3], *first_rows):
            ttnn.deallocate(tensor)
        tail_rm = fields[3]
        self._write_conv_state(tail_rm)
        ttnn.deallocate(tail_rm)
        return activated


def _decode_gate_fusion(decoder, a_raw, b_raw, *, fold_a_cast=False, beta_chain=False, round_beta=True):
    """One-token gate probes only; preserve the tested prefill padding path."""
    if beta_chain:
        operations = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)]
        if round_beta:
            # Explicit BF16 RNE in FP32 Dest preserves standalone sigmoid's
            # BF16 materialization before conversion to an FP32 gate tensor.
            operations.append(
                ttnn.UnaryWithParam(
                    ttnn.UnaryOpType.TYPECAST, ttnn.DataType.FLOAT32.value, ttnn.DataType.BFLOAT16.value
                )
            )
        operations.append(
            ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, ttnn.DataType.BFLOAT16.value, ttnn.DataType.FLOAT32.value)
        )
        beta = ttnn.unary_chain(b_raw, operations)
    else:
        beta16 = ttnn.sigmoid(b_raw)
        beta = ttnn.typecast(beta16, ttnn.float32)
        ttnn.deallocate(beta16)
    if fold_a_cast:
        biased = ttnn.add(a_raw, decoder.w["dt_bias"], dtype=ttnn.float32)
    else:
        a32 = ttnn.typecast(a_raw, ttnn.float32)
        biased = ttnn.add(a32, decoder.w["dt_bias"])
        ttnn.deallocate(a32)
    soft = ttnn.softplus(biased)
    ttnn.deallocate(biased)
    g = ttnn.multiply(decoder.w["A_neg"], soft)
    ttnn.deallocate(soft)
    return beta, g


class GateAddGDN(ArithmeticGDN):
    """Fold A's lossless BF16→FP32 conversion into the FP32 bias add."""

    def _gdn_gates_projected(self, a_raw, b_raw, logical_len, seq_len):
        if seq_len != 1:
            return super()._gdn_gates_projected(a_raw, b_raw, logical_len, seq_len)
        return _decode_gate_fusion(self, a_raw, b_raw, fold_a_cast=True)


class BetaChainGDN(ArithmeticGDN):
    """Fold sigmoid, explicit BF16 rounding, and FP32 output into one unary op."""

    def _gdn_gates_projected(self, a_raw, b_raw, logical_len, seq_len):
        if seq_len != 1:
            return super()._gdn_gates_projected(a_raw, b_raw, logical_len, seq_len)
        return _decode_gate_fusion(self, a_raw, b_raw, beta_chain=True)


class UnroundedBetaChainGDN(ArithmeticGDN):
    """Diagnostic control omitting the original BF16 sigmoid materialization."""

    def _gdn_gates_projected(self, a_raw, b_raw, logical_len, seq_len):
        if seq_len != 1:
            return super()._gdn_gates_projected(a_raw, b_raw, logical_len, seq_len)
        return _decode_gate_fusion(self, a_raw, b_raw, beta_chain=True, round_beta=False)


class GateChainGDN(ArithmeticGDN):
    """Combine GateAddGDN and rounded BetaChainGDN after each separate control."""

    def _gdn_gates_projected(self, a_raw, b_raw, logical_len, seq_len):
        if seq_len != 1:
            return super()._gdn_gates_projected(a_raw, b_raw, logical_len, seq_len)
        return _decode_gate_fusion(self, a_raw, b_raw, fold_a_cast=True, beta_chain=True)


class HybridNormGDN(FlatGDN):
    """Localize H1: functional rank4 Q/K normalization, flat V/head-major core."""

    def _chunk_delta_rule(self, q, k, v, g, beta):
        cfg = self.cfg
        batch, seq = q.shape[0], q.shape[1]
        nk, dk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
        nv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
        eye, tril, ones, masks = self.w["gdn_const_tiles"]
        normalized = []
        for tensor in (q, k):
            heads = self._split_heads(tensor, nk, dk)
            norm = l2_norm_ttnn(heads)
            ttnn.deallocate(heads)
            if norm.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                in_dram = ttnn.to_memory_config(norm, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(norm)
                norm = in_dram
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


class HybridCombinedGDN(HybridNormGDN, CombinedGDN):
    """Apply the HybridNormGDN normalization control to combined conv/norm/decode."""


class JointQKNormGDN(ArithmeticGDN):
    """Normalize adjacent expanded Q/K once, then retain Arithmetic's scalars."""

    def _gdn_decode_heads(self, activated):
        cfg = self.cfg
        batch = activated.shape[0]
        nk, nv, dk = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim
        rows = ttnn.reshape(activated, [batch, 1, 2 * nk + nv, dk])
        heads = ttnn.permute(rows, (0, 2, 1, 3))
        v = ttnn.slice(heads, [0, 2 * nk, 0, 0], [batch, 2 * nk + nv, 1, dk])
        qk = ttnn.slice(heads, [0, 0, 0, 0], [batch, 2 * nk, 1, dk])
        ttnn.deallocate(heads)
        if nv // nk > 1:
            expanded = ttnn.repeat_interleave(qk, nv // nk, dim=1)
            ttnn.deallocate(qk)
            qk = expanded
        normed = ttnn.rms_norm(qk, epsilon=1e-6 / dk)
        ttnn.deallocate(qk)
        q = ttnn.slice(normed, [0, 0, 0, 0], [batch, nv, 1, dk])
        k = ttnn.slice(normed, [0, nv, 0, 0], [batch, 2 * nv, 1, dk])
        ttnn.deallocate(normed)
        return q, k, v

    def _delta_rule_step(self, q, k, v, beta, g):
        cfg = self.cfg
        batch, nv, dk = q.shape[0], cfg.linear_num_value_heads, cfg.linear_key_head_dim
        dram = ttnn.DRAM_MEMORY_CONFIG
        # q/k already contain the same BF16 RMSNorm outputs produced by the
        # two ArithmeticGDN calls; scalar/cast rounding points remain intact.
        q_scaled = ttnn.multiply(q, dk**-1.0, memory_config=dram)
        q_row = ttnn.typecast(q_scaled, ttnn.float32)
        ttnn.deallocate(q_scaled)
        k_scaled = ttnn.multiply(k, dk**-0.5, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_row = ttnn.typecast(k_scaled, ttnn.float32)
        ttnn.deallocate(k_scaled)
        v_row = ttnn.typecast(v, ttnn.float32)
        beta_view = ttnn.reshape(beta, [batch, nv, 1, 1])
        g_view = ttnn.reshape(g, [batch, nv, 1, 1])
        state = self.recurrent_state
        ttnn.multiply(state, g_view, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], output_tensor=state)
        read = ttnn.matmul(k_row, state, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        difference = ttnn.subtract(v_row, read, memory_config=dram)
        delta = ttnn.multiply(difference, beta_view, memory_config=dram)
        for tensor in (read, difference, v_row):
            ttnn.deallocate(tensor)
        outer = ttnn.matmul(
            k_row, delta, transpose_a=True, memory_config=dram, compute_kernel_config=self.compute_kernel_config
        )
        ttnn.deallocate(k_row)
        ttnn.deallocate(delta)
        ttnn.add(state, outer, output_tensor=state)
        ttnn.deallocate(outer)
        result = ttnn.matmul(q_row, state, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(q_row)
        return result


class HybridConvOnly(HybridNormGDN):
    """Hybrid Q/K with KDA prefill conv only; original functional decode."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            if decoder.cfg.linear_conv_kernel_dim != 4 or kwargs.get("dtype", ttnn.bfloat16) != ttnn.bfloat16:
                raise ValueError("KDA conv requires four BF16 taps")
            decoder.kda_conv_program = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256)
        return decoder

    _gdn_prefill_conv_fields = KDAConvGDN._gdn_prefill_conv_fields


class HybridArithmetic(ArithmeticGDN):
    """Functional Q/K prefill normalization with Arithmetic decode and FIR."""

    _chunk_delta_rule = HybridNormGDN._chunk_delta_rule


class HybridKdaNorm(HybridNormGDN):
    """Hybrid Q/K plus KDA prefill output norm only; functional decode/FIR."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            decoder.w["kda_norm_vector"] = ttnn.from_torch(
                state_dict["linear_attn.norm.weight"].float().reshape(-1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=decoder.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
            )
        return decoder

    def _gdn_out_head_major(self, core, z, batch, seq):
        if seq == 1:
            return super()._gdn_out_head_major(core, z, batch, seq)
        # The positive prefill branch of KdaNormGDN has no super() dispatch.
        return KdaNormGDN._gdn_out_head_major(self, core, z, batch, seq)


class JointQKNormBeforeRepeat(JointQKNormGDN):
    """Move the same per-head RMSNorm before Q/K repetition; halve norm rows."""

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


class BiasProjectionGDN(ArithmeticGDN):
    """Fold dt_bias into the A columns of the existing packed linear.

    This probes the unavoidable movement of A's BF16 rounding point across
    the FP32 bias add. Other packed columns have exactly zero bias. Both this
    default BF16 output and the FP32 adaptation need raw gate/core checks.
    """

    biased_projection_dtype = ttnn.bfloat16

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            cfg = decoder.cfg
            start = cfg.conv_dim + cfg.linear_v_dim
            width = start + 2 * cfg.linear_num_value_heads
            bias = torch.zeros((1, 1, width), dtype=torch.float32)
            bias[0, 0, start : start + cfg.linear_num_value_heads] = state_dict["linear_attn.dt_bias"].float()
            decoder.w["gdn_packed_bias"] = ttnn.from_torch(
                bias,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=decoder.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
            )
        return decoder

    def _gdn_project(self, x):
        cfg = self.cfg
        packed = ttnn.linear(
            x,
            self.w["gdn_packed"],
            bias=self.w["gdn_packed_bias"],
            dtype=self.biased_projection_dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        ends = [0, cfg.conv_dim, cfg.conv_dim + cfg.linear_v_dim]
        ends.extend([ends[-1] + cfg.linear_num_value_heads, ends[-1] + 2 * cfg.linear_num_value_heads])
        fields = []
        for index, (start, end) in enumerate(zip(ends, ends[1:])):
            field = _field(packed, start, end)
            # The FP32 control changes only the bias boundary: restore the
            # original BF16 outputs for zero-biased QKV/Z/B fields.
            if self.biased_projection_dtype == ttnn.float32 and index != 2:
                bf16 = ttnn.typecast(field, ttnn.bfloat16)
                ttnn.deallocate(field)
                field = bf16
            fields.append(field)
        ttnn.deallocate(packed)
        return tuple(fields)

    def _gdn_gates_projected(self, a_raw, b_raw, logical_len, seq_len):
        beta16 = ttnn.sigmoid(b_raw)
        beta = ttnn.typecast(beta16, ttnn.float32)
        ttnn.deallocate(beta16)
        if a_raw.dtype == ttnn.float32:
            biased = a_raw
        else:
            biased = ttnn.typecast(a_raw, ttnn.float32)
        soft = ttnn.softplus(biased)
        if biased is not a_raw:
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


class BiasProjectionFP32GDN(BiasProjectionGDN):
    """Keep the fused A+bias result FP32, cast only zero-biased fields to BF16."""

    biased_projection_dtype = ttnn.float32


def probe_linear_conv_equivalence(functional, kda_candidate, prompt, *, compute_kernel_config=None):
    """Untimed conv localization using one shared real-weight QKV projection.

    Recommended input: H.make_activations(32,63,seed=31), before prefill.
    Existing functional history is read but never changed. Torch's FP32 oracle
    starts at identical BF16 QKV/taps/history, isolating convolution from earlier
    norm/projection and later recurrence. An optional KDA-only compute config
    enables a controlled accumulator probe without changing either decoder.
    """
    import torch

    from . import test_functional_decoder as harness

    if functional.is_full_attention or kda_candidate.is_full_attention:
        raise ValueError("conv probe requires two linear decoders")
    cfg = functional.cfg
    batch, logical_len = prompt.shape[0], prompt.shape[1]
    physical_len = ((logical_len + 127) // 128) * 128
    if batch != functional.batch_size or cfg.linear_conv_kernel_dim != 4:
        raise ValueError("probe requires the allocated batch and four convolution taps")
    host_padded = torch.nn.functional.pad(prompt, (0, 0, 0, physical_len - logical_len))
    device_input = harness.to_device(functional.device, host_padded)
    normed = functional._norm(device_input, functional.w["attn_norm"])
    qkv = ttnn.linear(normed, functional.w["gdn_qkv"], compute_kernel_config=functional.compute_kernel_config)
    ttnn.deallocate(normed)
    ttnn.deallocate(device_input)
    qkv_host = ttnn.to_torch(qkv).float()
    history_host = torch.cat([ttnn.to_torch(row).float() for row in functional.conv_state], dim=1)
    taps_host = [ttnn.to_torch(tap).float().reshape(-1) for tap in functional.w["conv_taps"]]
    weight_host = torch.stack(taps_host, dim=-1).unsqueeze(1)
    oracle_acc = torch.nn.functional.conv1d(
        torch.cat([history_host, qkv_host], dim=1).transpose(1, 2), weight_host, groups=cfg.conv_dim
    ).transpose(1, 2)
    oracle = torch.nn.functional.silu(oracle_acc)

    history = ttnn.concat(functional.conv_state, dim=1)
    padded = ttnn.concat([history, qkv], dim=1)
    ttnn.deallocate(history)
    acc = None
    for index, tap in enumerate(functional.w["conv_taps"]):
        piece = padded[:, index : index + physical_len, :]
        piece = ttnn.to_layout(piece, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if acc is None:
            acc = ttnn.multiply(piece, tap, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            previous = acc
            acc = ttnn.addcmul(previous, piece, tap, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(previous)
        ttnn.deallocate(piece)
    ttnn.deallocate(padded)
    functional_acc = ttnn.to_torch(acc).float()
    activated = ttnn.silu(acc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(acc)
    functional_out = ttnn.to_torch(activated).float()
    ttnn.deallocate(activated)

    tokens_rm = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(qkv)
    rows_rm = [
        ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for row in functional.conv_state
    ]
    history_rm = ttnn.concat(rows_rm, dim=1)
    for tensor in rows_rm:
        ttnn.deallocate(tensor)
    candidate_users = []
    for user in range(batch):
        user_tokens, owns_tokens = _slice_owned(tokens_rm, [user, 0, 0], [user + 1, physical_len, cfg.conv_dim])
        user_history, owns_history = _slice_owned(history_rm, [user, 0, 0], [user + 1, 3, cfg.conv_dim])
        fields = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            user_tokens,
            user_history,
            *functional.w["conv_taps"],
            cfg.linear_q_dim,
            cfg.linear_k_dim,
            cfg.linear_v_dim,
            program_config=kda_candidate.kda_conv_program,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=(
                compute_kernel_config if compute_kernel_config is not None else kda_candidate.compute_kernel_config
            ),
        )
        candidate_users.append(torch.cat([ttnn.to_torch(field).float() for field in fields], dim=-1))
        for tensor in fields:
            ttnn.deallocate(tensor)
        if owns_tokens:
            ttnn.deallocate(user_tokens)
        if owns_history:
            ttnn.deallocate(user_history)
    ttnn.deallocate(tokens_rm)
    ttnn.deallocate(history_rm)
    candidate_out = torch.cat(candidate_users, dim=0)

    def metrics(reference, actual):
        reference, actual = reference[:, :logical_len], actual[:, :logical_len]
        finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
        return {
            "finite": finite,
            "pcc": (1.0 if torch.equal(reference, actual) else harness.pcc(reference, actual)) if finite else None,
            "max_abs_error": (reference - actual).abs().max().item(),
            "relative_l2": ((reference - actual).double().norm() / reference.double().norm().clamp_min(1e-30)).item(),
        }

    fields = (
        ("q", 0, cfg.linear_q_dim),
        ("k", cfg.linear_q_dim, cfg.linear_q_dim + cfg.linear_k_dim),
        ("v", cfg.linear_q_dim + cfg.linear_k_dim, cfg.conv_dim),
    )
    per_user = []
    for user in range(batch):
        take = slice(user, user + 1)
        per_user.append(
            {
                "user": user,
                "kda_vs_functional": metrics(functional_out[take], candidate_out[take]),
                "functional_vs_fp32_oracle": metrics(oracle[take], functional_out[take]),
                "kda_vs_fp32_oracle": metrics(oracle[take], candidate_out[take]),
                "functional_acc_vs_fp32_oracle": metrics(oracle_acc[take], functional_acc[take]),
                "functional_silu_vs_torch_on_bf16_acc": metrics(
                    torch.nn.functional.silu(functional_acc[take]), functional_out[take]
                ),
                "fields": {
                    name: metrics(functional_out[take, :, start:end], candidate_out[take, :, start:end])
                    for name, start, end in fields
                },
            }
        )
    return {
        "logical_len": logical_len,
        "physical_len": physical_len,
        "batch": batch,
        "kda_vs_functional": metrics(functional_out, candidate_out),
        "functional_vs_fp32_oracle": metrics(oracle, functional_out),
        "kda_vs_fp32_oracle": metrics(oracle, candidate_out),
        "worst_user": min(
            per_user,
            key=lambda item: item["kda_vs_functional"]["pcc"] if item["kda_vs_functional"]["pcc"] is not None else -1,
        )["user"],
        "per_user": per_user,
    }


class Conv1dGDN(HybridNormGDN):
    """Ordinary depthwise Conv1d prefill, functional SiLU/QK/decode semantics.

    Four independent channel groups bound the height-sharded kernel's L1
    footprint at BF16/HiFi4. All physical prefill lengths are prepared during
    state setup; forward never uploads weights. The ordinary depthwise kernel
    does not execute the generic factory's activation macro, so SiLU remains
    explicitly separate.
    """

    conv1d_channel_chunk = 2048

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            cfg = decoder.cfg
            if cfg.linear_conv_kernel_dim != 4 or kwargs.get("dtype", ttnn.bfloat16) != ttnn.bfloat16:
                raise ValueError("Conv1d probe requires four BF16 taps")
            if (cfg.linear_q_dim, cfg.linear_k_dim, cfg.linear_v_dim) != (2048, 2048, 4096):
                raise ValueError("Conv1d probe requires the Ornith 2048/2048/4096 Q/K/V widths")
            # prepare_conv_weights expects host OIHW, and Conv1d is a height1
            # Conv2d wrapper. These are real weights, not a dense diagonal GEMM.
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
        q, k = parts[0], parts[1]
        v = ttnn.concat(parts[2:], dim=2)
        for tensor in parts[2:]:
            ttnn.deallocate(tensor)
        tail_rm = ttnn.slice(padded, [0, logical_len, 0], [batch, logical_len + 3, cfg.conv_dim])
        tail = ttnn.to_layout(tail_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tail_rm)
        ttnn.deallocate(padded)
        return q, k, v, tail


class Conv1dNarrowGDN(Conv1dGDN):
    """Retry the measured L1 failure with independent512-channel conv groups.

    Reuses setup, BF16/HiFi4,32-row activation blocks and auto DRAM slicing.
    Field assembly uses model widths so the sixteen groups form Q4/K4/V8.
    """

    conv1d_channel_chunk = 512

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
        tail = ttnn.to_layout(tail_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tail_rm)
        ttnn.deallocate(padded)
        return q, k, v, tail


class Conv1dNarrow256GDN(Conv1dNarrowGDN):
    """Secondary L1-footprint control: Q8/K8/V16 independent256-channel groups."""

    conv1d_channel_chunk = 256


class Conv1d1024GDN(Conv1dNarrowGDN):
    """Largest intermediate width between the failed2048 and valid512 adaptations."""

    conv1d_channel_chunk = 1024

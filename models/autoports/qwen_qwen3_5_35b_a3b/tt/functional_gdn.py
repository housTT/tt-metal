# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact-recurrence, explicit-precision GDN for the functional decoder.

This module owns one request's stable state buffers. FunctionalDecoder selects
per-request buffers before calling it. Input/output are device tensors [1,T,2048].
The outer decoder plans arbitrary logical lengths as bounded aligned chunks and
single-token tails; the native prefill core uses internal chunks of 32 tokens.
Construction may use torch. Forward and reset use device operations only.
"""


class FunctionalGatedDeltaNet:
    """Single-device Qwen3.5-35B-A3B GDN with FP32 recurrent state.

    Projection weights are BF16 from the original checkpoint, with HiFi4/FP32
    accumulation. Decay and recurrent state stay FP32 across prefill/decode.
    The native chunk op uses an exact unit diagonal, independent of environment
    variables. Its BF16 Q/K/V inputs and FP32 state match its explicit API.
    """

    @classmethod
    def from_existing(cls, gdn, *, state_dict):
        """Replace a Qwen36 GDN using its config and unquantized remapped substate.

        Keys must have the ``layers.N.linear_attn.`` prefix removed. Reconstruct
        weights from this mapping, never by upcasting the existing BFP8 tensors.
        Complete this setup before allocating request slots or recording traces.
        """
        import torch

        import ttnn

        self = cls()
        self.device, self.cfg = gdn.device, gdn.cfg
        self.num_heads = self.cfg.num_heads
        self.num_v_heads = self.cfg.num_v_heads
        self.head_k_dim = self.cfg.head_k_dim
        self.head_v_dim = self.cfg.head_v_dim
        self.conv_kernel_size = self.cfg.conv_kernel_size
        self.norm_eps = self.cfg.norm_eps
        self.long_prefill_chunk_size = 32
        actual = (self.num_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim, self.conv_kernel_size)
        if self.num_v_heads % self.num_heads or actual[2:] != (128, 128, 4) or 32 % self.num_v_heads:
            raise ValueError(f"Qwen3.5-35B-A3B GDN dimensions {actual} are unsupported")
        self.hidden_size = int(state_dict["qkv_proj.weight"].shape[1])
        if self.hidden_size != 2048:
            raise ValueError("This functional GDN requires the target hidden width 2048")

        def put(tensor, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                tensor.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )

        projection_keys = ("qkv_proj.weight", "in_proj_a.weight", "in_proj_b.weight", "in_proj_z.weight")
        mega = torch.cat([state_dict[key] for key in projection_keys], dim=0).T
        self.mega_weight = put(mega)
        self.output_weight = put(state_dict["out_proj.weight"].T)
        self.norm_weight = put(state_dict["norm.weight"].reshape(1, 1, 1, self.head_v_dim))
        self.dt_bias = put(state_dict["dt_bias"].float().reshape(1, 1, self.num_v_heads), ttnn.float32)
        self.a_neg = put(-state_dict["A_log"].float().exp().reshape(1, 1, self.num_v_heads), ttnn.float32)

        conv = torch.cat([state_dict[f"{part}_conv.weight"] for part in ("q", "k", "v")], dim=0)
        self.conv_taps = tuple(
            put(conv[:, 0, tap].float().reshape(1, 1, -1), ttnn.float32) for tap in range(self.conv_kernel_size)
        )
        if any(f"{part}_conv.bias" in state_dict for part in ("q", "k", "v")):
            raise ValueError("The target checkpoint has bias-free GDN convolution")

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Caller-owned constants prevent the native op's eager host-upload fallback.
        eye = torch.eye(32, dtype=torch.float32)
        ones = torch.ones(32, 32, dtype=torch.float32)
        ii, jj = torch.arange(32)[:, None], torch.arange(32)[None, :]
        quadrants = torch.cat(((ii < 16) & (jj < 16), (ii >= 16) & (jj >= 16), (ii >= 16) & (jj < 16)), dim=1).float()
        self.chunk_constants = tuple(
            put(t.reshape(1, 1, *t.shape), ttnn.float32) for t in (eye, torch.tril(ones), ones, quadrants)
        )
        self.chunk_program_config = ttnn.ChunkGdnPhasedProgramConfig()

        def zeros(shape, dtype):
            return ttnn.zeros(
                shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        rec_shape = (1, self.num_v_heads, self.head_k_dim, self.head_v_dim)
        conv_shape = (1, self.conv_kernel_size - 1, self.cfg.q_dim + self.cfg.k_dim + self.cfg.v_dim)
        self.recurrent_state = zeros(rec_shape, ttnn.float32)
        self._zero_recurrent = zeros(rec_shape, ttnn.float32)
        self.fused_conv_state = zeros(conv_shape, ttnn.bfloat16)
        self._zero_conv = zeros(conv_shape, ttnn.bfloat16)
        return self

    def reset_state_inplace(self):
        import ttnn

        ttnn.copy(self._zero_recurrent, self.recurrent_state)
        ttnn.copy(self._zero_conv, self.fused_conv_state)

    def _convolve(self, qkv):
        """Accumulate four FIR taps in FP32, preserving checkpoint BF16 boundaries."""
        import ttnn

        mc = ttnn.DRAM_MEMORY_CONFIG
        length = qkv.shape[1]
        history = ttnn.concat([self.fused_conv_state, qkv], dim=1, memory_config=mc)
        accumulated = None
        for tap, weight in enumerate(self.conv_taps):
            x = ttnn.to_layout(history[:, tap : tap + length, :], ttnn.TILE_LAYOUT, memory_config=mc)
            product = ttnn.multiply(ttnn.typecast(x, ttnn.float32), weight, memory_config=mc)
            accumulated = product if accumulated is None else ttnn.add(accumulated, product, memory_config=mc)
        # In [old three tokens, new tokens], the new three-token tail starts at T.
        tail = ttnn.to_layout(
            history[:, length : length + self.conv_kernel_size - 1, :], ttnn.TILE_LAYOUT, memory_config=mc
        )
        ttnn.copy(tail, self.fused_conv_state)
        rounded = ttnn.typecast(accumulated, ttnn.bfloat16)
        return ttnn.typecast(ttnn.silu(ttnn.typecast(rounded, ttnn.float32), memory_config=mc), ttnn.bfloat16)

    def _heads(self, x, heads, dim):
        import ttnn

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.reshape(x, (1, x.shape[1], heads, dim))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _normalize_key(self, x):
        import ttnn

        # HF's unfused reference normalizes in its BF16 activation dtype,
        # before converting the recurrence operands/state to FP32.
        square = ttnn.multiply(x, x)
        total = ttnn.sum(square, dim=-1, keepdim=True, compute_kernel_config=self.compute_kernel_config)
        inverse = ttnn.rsqrt(ttnn.add(total, 1e-6), fast_and_approximate_mode=False)
        return ttnn.multiply(x, inverse)

    def _recurrent_step_fp32(self, q, k, v, decay_log, beta):
        """One HF-ordered recurrence using FP32 SFPU arithmetic and reductions.

        Q/K are BF16-normalized [1,1,16,128], V is [1,1,32,128].
        Adjacent value heads share each key head. State stays [1,32,128,128]
        and the returned output is [1,1,32,128], both FP32. The caller copies
        the returned state into its persistent request buffer.

        Native recurrent matmuls carry FP32 state but truncate operands in
        FPU source registers. Matching FP32 operands route TTNN binary ops
        through SFPU; accurate FP32 sum also uses SFPU on this target. Keep
        each HF multiply/add/reduction boundary explicit.
        """
        import ttnn

        mc = ttnn.DRAM_MEMORY_CONFIG
        repeats = self.num_v_heads // self.num_heads

        def key_column(tensor):
            tensor = ttnn.typecast(tensor, ttnn.float32)
            tensor = ttnn.repeat_interleave(tensor, repeats, dim=2, memory_config=mc)
            return ttnn.permute(tensor, (0, 2, 3, 1), memory_config=mc)

        def head_scalar(tensor):
            tensor = ttnn.typecast(tensor, ttnn.float32)
            tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
            tensor = ttnn.reshape(tensor, (1, self.num_v_heads, 1, 1))
            return ttnn.to_layout(tensor, ttnn.TILE_LAYOUT, memory_config=mc)

        def sum_keys(tensor):
            return ttnn.sum(
                tensor,
                dim=-2,
                keepdim=True,
                memory_config=mc,
                compute_kernel_config=self.compute_kernel_config,
                fast_and_approximate_mode=False,
            )

        # HF scales the normalized query only after conversion to FP32.
        q_column = ttnn.multiply(key_column(q), self.head_k_dim**-0.5, memory_config=mc)
        k_column = key_column(k)
        v_row = ttnn.permute(ttnn.typecast(v, ttnn.float32), (0, 2, 1, 3), memory_config=mc)
        decay = head_scalar(ttnn.exp(decay_log, fast_and_approximate_mode=False, memory_config=mc))
        beta = head_scalar(beta)

        decayed_state = ttnn.multiply(self.recurrent_state, decay, memory_config=mc)
        memory = sum_keys(ttnn.multiply(decayed_state, k_column, memory_config=mc))
        delta = ttnn.multiply(ttnn.subtract(v_row, memory, memory_config=mc), beta, memory_config=mc)
        state = ttnn.add(decayed_state, ttnn.multiply(k_column, delta, memory_config=mc), memory_config=mc)
        output = sum_keys(ttnn.multiply(state, q_column, memory_config=mc))
        return ttnn.permute(output, (0, 2, 1, 3), memory_config=mc), state

    def forward(self, x, mode="recurrent", chunk_size=None, valid_len=None):
        """Consume one real token or an aligned real-token chunk, retaining state."""
        import ttnn

        if len(x.shape) != 3 or x.shape[0] != 1 or x.shape[2] != self.hidden_size:
            raise ValueError("GDN expects a selected request [1,T,2048]")
        length = x.shape[1]
        if mode not in ("recurrent", "chunk") or length < 1:
            raise ValueError("GDN mode must be recurrent or chunk, with at least one token")
        if valid_len is not None and valid_len != length:
            raise ValueError("FunctionalDecoder must remove padding before the GDN state update")
        if length > 1 and (mode != "chunk" or length % 32):
            raise ValueError("The outer decoder must plan aligned chunks and single-token tails")
        mc = ttnn.DRAM_MEMORY_CONFIG
        mega = ttnn.linear(x, self.mega_weight, memory_config=mc, compute_kernel_config=self.compute_kernel_config)
        qkv_width = self.cfg.q_dim + self.cfg.k_dim + self.cfg.v_dim
        a_start, b_start, z_start = qkv_width, qkv_width + self.num_v_heads, qkv_width + 2 * self.num_v_heads

        def columns(tensor, start, end):
            return ttnn.to_layout(tensor[:, :, start:end], ttnn.TILE_LAYOUT, memory_config=mc)

        qkv = self._convolve(columns(mega, 0, qkv_width))
        q = columns(qkv, 0, self.cfg.q_dim)
        k = columns(qkv, self.cfg.q_dim, self.cfg.q_dim + self.cfg.k_dim)
        v = columns(qkv, self.cfg.q_dim + self.cfg.k_dim, qkv_width)
        a = ttnn.typecast(columns(mega, a_start, b_start), ttnn.float32)
        beta = ttnn.typecast(
            ttnn.sigmoid(ttnn.typecast(columns(mega, b_start, z_start), ttnn.float32), memory_config=mc), ttnn.bfloat16
        )
        decay_log = ttnn.multiply(
            self.a_neg, ttnn.softplus(ttnn.add(a, self.dt_bias, memory_config=mc), memory_config=mc), memory_config=mc
        )
        if length > 1:
            eye, tril, ones, masks = self.chunk_constants
            # Explicit head-split inputs bypass the native flat FP32 normalizer.
            # The exact chunk op expands the 16 key heads to 32 value heads.
            q = self._normalize_key(self._heads(q, self.num_heads, self.head_k_dim))
            k = self._normalize_key(self._heads(k, self.num_heads, self.head_k_dim))
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
                program_config=self.chunk_program_config,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=mc,
                eye=eye,
                tril=tril,
                ones=ones,
                masks=masks,
            )
            output = ttnn.to_layout(output, ttnn.TILE_LAYOUT, memory_config=mc)
        else:
            q = self._normalize_key(self._heads(q, self.num_heads, self.head_k_dim))
            k = self._normalize_key(self._heads(k, self.num_heads, self.head_k_dim))
            v = self._heads(v, self.num_v_heads, self.head_v_dim)
            output, state = self._recurrent_step_fp32(q, k, v, decay_log, beta)
        ttnn.copy(state, self.recurrent_state)
        # HF recurrence returns its activation dtype before the gated RMSNorm.
        output = ttnn.typecast(output, ttnn.bfloat16)
        normalized = ttnn.rms_norm(
            ttnn.typecast(output, ttnn.float32),
            epsilon=self.norm_eps,
            memory_config=mc,
            compute_kernel_config=self.compute_kernel_config,
        )
        normalized = ttnn.multiply(ttnn.typecast(normalized, ttnn.bfloat16), self.norm_weight, memory_config=mc)
        gate = self._heads(columns(mega, z_start, z_start + self.cfg.v_dim), self.num_v_heads, self.head_v_dim)
        gated = ttnn.multiply(
            ttnn.typecast(normalized, ttnn.float32),
            ttnn.silu(ttnn.typecast(gate, ttnn.float32), memory_config=mc),
            memory_config=mc,
        )
        gated = ttnn.typecast(gated, ttnn.bfloat16)
        gated = ttnn.reshape(gated, (1, length, self.cfg.v_dim))
        return ttnn.linear(
            gated, self.output_weight, memory_config=mc, compute_kernel_config=self.compute_kernel_config
        )

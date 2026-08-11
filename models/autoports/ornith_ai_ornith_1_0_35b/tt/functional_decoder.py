# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Functional TTNN decoder layer for ornith-ai/Ornith-1.0-35B (``Qwen3_5MoeForConditionalGeneration``).

Ornith's text decoder interleaves two layer kinds, selected per index by
``text_config.layer_types`` (30 × ``linear_attention`` + 10 × ``full_attention``, period 4):

``linear_attention``
    Gated DeltaNet token mixer: fused QKVZAB projections, a depthwise causal conv1d
    (kernel 4) over the QKV stream, and the gated delta rule. State is a fixed-size
    recurrent matrix ``[B, 32, 128, 128]`` plus a ``[B, 3, 8192]`` conv history — there is
    no KV cache and no RoPE.

``full_attention``
    Gated GQA softmax attention: 16 query heads / 2 KV heads / head_dim 256, per-head
    zero-centered RMSNorm on Q and K, **partial** RoPE over the first 64 of 256 head dims,
    a sigmoid output gate carried in the 2×-wide ``q_proj``, and a paged KV cache.

Both kinds share the residual/norm order and the same MoE feed-forward block
(:mod:`~models.autoports.ornith_ai_ornith_1_0_35b.tt.moe`)::

    h = x + mixer(input_layernorm(x))
    y = h + moe(post_attention_layernorm(h))

with zero-centered RMSNorm everywhere in the residual stream (HF applies ``x_normed *
(1 + weight)``; the ``+1`` is folded into the weight at load time).

Prefill / decode contract
-------------------------
``prefill_forward(x, start_pos=0, page_table=None, chunk_size=None)``
    ``x``: ``[batch, seq_len, hidden]``, bfloat16, TILE, DRAM. ``seq_len`` may be **any**
    value in ``[1, supported_context - start_pos]`` — it does not need to be a multiple of
    the tile, page, or internal chunk size. The layer chunks the sequence internally into
    ``chunk_size`` blocks (default :data:`DEFAULT_PREFILL_CHUNK`), pads the physical length
    of each block up to :data:`PREFILL_ALIGN`, masks the padded tail so it cannot perturb
    the recurrent/conv state, and slices the output back to ``seq_len``.
    ``start_pos`` is the absolute position of ``x[:, 0]`` and must be a multiple of
    ``chunk_size`` (0 for a fresh sequence) so the internal chunk starts stay aligned to
    the chunked-SDPA program config.
    ``page_table``: ``[batch, num_blocks]`` int32 ROW_MAJOR device tensor; required for
    ``full_attention`` layers, ignored by ``linear_attention`` layers. Row ``u`` owns
    user ``u``'s cache blocks. All users in a call share one ``seq_len``.
    Returns ``[batch, seq_len, hidden]``. Prefill leaves the KV cache filled for
    ``[start_pos, start_pos + seq_len)`` and the DeltaNet state advanced to
    ``start_pos + seq_len``.

``decode_forward(x, current_pos, rot_idxs, page_table=None)``
    ``x``: ``[batch, 1, hidden]``. ``current_pos``: ``[batch]`` int32 ROW_MAJOR **device**
    tensor holding each user's absolute KV position (the slot this token writes).
    ``rot_idxs``: ``[1, batch]`` uint32 ROW_MAJOR device tensor of RoPE row indices —
    normally the same values as ``current_pos``. Both are device tensors so a captured
    trace only needs its input buffers refreshed. ``page_table`` as above.
    Returns ``[batch, 1, hidden]``. Updates the paged KV cache in place at
    ``current_pos`` and advances the DeltaNet state by one step, in place.

Neither forward path calls ``torch``, ``ttnn.from_torch``, ``ttnn.to_torch``, or any host
fallback: every weight, constant, cache and state buffer is created in
:meth:`FunctionalDecoder.from_state_dict` / :meth:`FunctionalDecoder.allocate_state`.
"""

from __future__ import annotations

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import OrnithDecoderConfig
from models.autoports.ornith_ai_ornith_1_0_35b.tt.moe import OrnithMoE, load_moe_weights
from models.autoports.ornith_ai_ornith_1_0_35b.tt.rope import OrnithRope
from models.common.lightweightmodule import LightweightModule
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import l2_norm_ttnn

TILE = 32

#: Physical alignment of a prefill block. The chunked-SDPA program config uses 64-token
#: q/k chunks, and 128 keeps every block start a legal multiple for both the SDPA chunking
#: and the paged cache's 64-token blocks.
PREFILL_ALIGN = 128

#: Default logical tokens per internal prefill block. Bounds peak activation memory so the
#: advertised 262144-token context fits; every block start is a multiple of this value.
DEFAULT_PREFILL_CHUNK = 2048

#: Paged KV cache block size (tokens per block).
DEFAULT_PAGE_BLOCK_SIZE = 64


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_dim(tensor, dim: int, amount: int):
    """Zero-pad ``amount`` elements onto the high end of ``dim``.

    Aliasing warning: when the requested logical padding already fits inside the tensor's
    physical tile padding (e.g. 1 → 32 rows), the result can share ``tensor``'s buffer. Callers
    must therefore never free the pre-pad tensor separately — free the padded one and let the
    source's Python reference drop.
    """
    padding = [(0, 0)] * len(tensor.shape)
    padding[dim] = (0, amount)
    return ttnn.pad(tensor, padding, 0.0)


def _slice_owned(tensor, begins, ends):
    """``(slice, owned)`` where ``owned`` says whether the result is a fresh buffer.

    A slice covering the whole tensor can come back aliasing its input, so freeing it would free
    the caller's tensor. Returning ownership explicitly keeps the deallocations honest.
    """
    if all(b == 0 for b in begins) and list(ends) == [int(d) for d in tensor.shape]:
        return tensor, False
    return ttnn.slice(tensor, list(begins), list(ends)), True


def num_blocks_for_context(context: int, block_size: int = DEFAULT_PAGE_BLOCK_SIZE) -> int:
    """Paged blocks needed to hold ``context`` tokens.

    Rounded up to a multiple of 32 blocks: the chunked/paged SDPA kernels require the page
    table's row stick size to be a multiple of 32 entries.
    """
    blocks = _align_up(context, block_size) // block_size
    return _align_up(blocks, 32)


class FunctionalDecoder(LightweightModule):
    """One Ornith decoder layer on a TTNN mesh device.

    Construct with :meth:`from_state_dict`; call :meth:`allocate_state` (and, for
    ``full_attention`` layers, :meth:`attach_kv_cache`) before the first forward.
    """

    def __init__(
        self,
        mesh_device,
        config: OrnithDecoderConfig,
        layer_idx: int,
        *,
        weights: dict,
        moe,
        rope,
        max_context: int,
        page_block_size: int,
        prefill_chunk: int,
    ):
        self.device = mesh_device
        self.cfg = config
        self.layer_idx = layer_idx
        self.kind = config.layer_kind(layer_idx)
        self.is_full_attention = self.kind == "full_attention"
        self.w = weights
        self.moe = moe
        self.rope = rope
        self.max_context = max_context
        self.page_block_size = page_block_size
        self.prefill_chunk = prefill_chunk

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Paged KV cache (full_attention only) and DeltaNet state (linear_attention only).
        self.k_cache = None
        self.v_cache = None
        self.recurrent_state = None
        self.conv_state = None  # list of ``conv_kernel_dim - 1`` buffers, each [B, 1, conv_dim]
        self.batch_size = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        max_context: int | None = None,
        page_block_size: int = DEFAULT_PAGE_BLOCK_SIZE,
        prefill_chunk: int = DEFAULT_PREFILL_CHUNK,
        dtype=ttnn.bfloat16,
        **kwargs,
    ) -> "FunctionalDecoder":
        """Build a layer from an HF decoder-layer state dict.

        ``state_dict`` keys are module-relative, exactly as ``transformers`` holds them for a
        ``Qwen3_5MoeDecoderLayer`` (see
        :func:`models.autoports.ornith_ai_ornith_1_0_35b.reference.hf_reference.load_layer_state_dict`).

        ``max_context`` defaults to the HF-advertised ``max_position_embeddings``.
        """
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
        import torch

        config = OrnithDecoderConfig.from_hf_config(hf_config)
        max_context = int(max_context or config.max_position_embeddings)
        if prefill_chunk % PREFILL_ALIGN:
            raise ValueError(f"prefill_chunk {prefill_chunk} must be a multiple of {PREFILL_ALIGN}")
        kind = config.layer_kind(layer_idx)

        def upload(t, tensor_dtype=dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.as_tensor(
                t.to(torch.bfloat16).contiguous() if tensor_dtype == ttnn.bfloat16 else t.float().contiguous(),
                dtype=tensor_dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        def linear_weight(name):
            """HF stores ``[out, in]``; ``ttnn.linear`` wants ``[in, out]``."""
            return upload(state_dict[name].transpose(0, 1))

        weights: dict = {}
        # Residual-stream norms: fold HF's zero-centering (`x_normed * (1 + w)`) into the weight.
        for key, dst in (("input_layernorm.weight", "attn_norm"), ("post_attention_layernorm.weight", "ff_norm")):
            weights[dst] = upload((state_dict[key].float() + 1.0).reshape(1, 1, 1, -1))

        rope = None
        if kind == "full_attention":
            weights["q_proj"] = linear_weight("self_attn.q_proj.weight")
            weights["k_proj"] = linear_weight("self_attn.k_proj.weight")
            weights["v_proj"] = linear_weight("self_attn.v_proj.weight")
            weights["o_proj"] = linear_weight("self_attn.o_proj.weight")
            # Q/K norms are also zero-centered and act over head_dim only.
            weights["q_norm"] = upload((state_dict["self_attn.q_norm.weight"].float() + 1.0).reshape(1, 1, 1, -1))
            weights["k_norm"] = upload((state_dict["self_attn.k_norm.weight"].float() + 1.0).reshape(1, 1, 1, -1))
            # The table is built one prefill alignment longer than the logical context: the last
            # prefill block's *physical* window can end past max_context (e.g. max_context=8000,
            # seq_len=8000 -> final block offset 6144, physical length 1920, window end 8064).
            # Those rows only ever rotate zero-padded activations, but the slice must be in range.
            rope = OrnithRope(
                mesh_device,
                config,
                max_context=max_context,
                table_context=_align_up(max_context, prefill_chunk) + prefill_chunk,
            )
        elif kind == "linear_attention":
            prefix = "linear_attn."
            weights["gdn_qkv"] = linear_weight(prefix + "in_proj_qkv.weight")
            weights["gdn_z"] = linear_weight(prefix + "in_proj_z.weight")
            weights["gdn_a"] = linear_weight(prefix + "in_proj_a.weight")
            weights["gdn_b"] = linear_weight(prefix + "in_proj_b.weight")
            weights["gdn_out"] = linear_weight(prefix + "out_proj.weight")
            # Gated-DeltaNet output norm is a *standard* RMSNorm (weights ≈ 1), not zero-centered.
            weights["gdn_norm"] = upload(state_dict[prefix + "norm.weight"].float().reshape(1, 1, 1, -1))
            # Depthwise causal conv: pre-slice [conv_dim, 1, K] into K per-tap rows [1, 1, conv_dim].
            conv_w = state_dict[prefix + "conv1d.weight"].float()
            # 3-D so they broadcast against the [B, T, conv_dim] activation stream.
            weights["conv_taps"] = [
                upload(conv_w[:, 0, k].reshape(1, 1, -1)) for k in range(config.linear_conv_kernel_dim)
            ]
            # g = -exp(A_log) * softplus(a + dt_bias); -exp(A_log) is constant per layer.
            weights["A_neg"] = upload(
                (-state_dict[prefix + "A_log"].float().exp()).reshape(1, 1, -1), tensor_dtype=ttnn.float32
            )
            weights["dt_bias"] = upload(
                state_dict[prefix + "dt_bias"].float().reshape(1, 1, -1), tensor_dtype=ttnn.float32
            )
            # Constant tiles for ttnn.transformer.chunk_gated_delta_rule. Passed explicitly so they
            # are device-resident before any trace capture (the op's internal build uploads them).
            from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE, build_fused_const_tiles

            weights["gdn_chunk_size"] = _FUSED_CHUNK_SIZE
            weights["gdn_const_tiles"] = build_fused_const_tiles(mesh_device, _FUSED_CHUNK_SIZE)
            # Device-resident position ramp used to mask the padded tail of a prefill block
            # without a host round trip.
            weights["pos_ramp"] = upload(
                torch.arange(prefill_chunk, dtype=torch.float32).reshape(1, prefill_chunk, 1),
                tensor_dtype=ttnn.float32,
            )
        else:
            raise ValueError(f"unsupported layer kind {kind!r}")

        moe = OrnithMoE(mesh_device, config, load_moe_weights(mesh_device, config, state_dict, dtype=dtype))

        return cls(
            mesh_device,
            config,
            layer_idx,
            weights=weights,
            moe=moe,
            rope=rope,
            max_context=max_context,
            page_block_size=page_block_size,
            prefill_chunk=prefill_chunk,
        )

    # ------------------------------------------------------------------ state
    def allocate_kv_cache(self, num_blocks: int, dtype=ttnn.bfloat16):
        """Allocate and attach a paged KV cache with ``num_blocks`` blocks.

        Shape ``[num_blocks, n_kv_heads, page_block_size, head_dim]``. No-op for
        ``linear_attention`` layers, which hold recurrent state instead.
        """
        if not self.is_full_attention:
            return None
        shape = [num_blocks, self.cfg.n_kv_heads, self.page_block_size, self.cfg.head_dim]
        self.k_cache = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        self.v_cache = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        return self.k_cache, self.v_cache

    def attach_kv_cache(self, k_cache, v_cache):
        """Point the layer at an externally allocated paged KV cache."""
        if not self.is_full_attention:
            raise TypeError("linear_attention layers have no KV cache")
        self.k_cache = k_cache
        self.v_cache = v_cache

    def allocate_state(self, batch_size: int):
        """Allocate the per-batch DeltaNet state (zeroed). No-op for ``full_attention``.

        The buffers are persistent and only ever written in place, so their device addresses
        stay valid across trace replays.
        """
        self.batch_size = batch_size
        if self.is_full_attention:
            return
        self.recurrent_state = ttnn.zeros(
            [batch_size, self.cfg.linear_num_value_heads, self.cfg.linear_key_head_dim, self.cfg.linear_value_head_dim],
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.conv_state = [
            ttnn.zeros(
                [batch_size, 1, self.cfg.conv_dim],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            for _ in range(self.cfg.linear_conv_kernel_dim - 1)
        ]

    def reset_state(self):
        """Zero the DeltaNet state in place (addresses preserved for trace replay)."""
        if self.is_full_attention or self.recurrent_state is None:
            return
        ttnn.multiply(self.recurrent_state, 0.0, output_tensor=self.recurrent_state)
        for buf in self.conv_state:
            ttnn.multiply(buf, 0.0, output_tensor=buf)

    # ------------------------------------------------------------------ small helpers
    def _norm(self, x, weight):
        """Zero-centered RMSNorm — the ``+1`` is already folded into ``weight``."""
        return ttnn.rms_norm(x, weight=weight, epsilon=self.cfg.norm_eps)

    def _split_heads(self, t, heads, head_dim):
        """``[B, T, heads*head_dim] -> [B, T, heads, head_dim]``.

        Goes through ROW_MAJOR: a direct TILE reshape that splits the last dim pads the new
        second-to-last dim instead of relabelling it.
        """
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        t = ttnn.reshape(t, [t.shape[0], t.shape[1], heads, head_dim])
        return ttnn.to_layout(t, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _merge_heads(self, t):
        """``[B, T, heads, head_dim] -> [B, T, heads*head_dim]`` (inverse of :meth:`_split_heads`)."""
        b, s, h, d = t.shape
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        t = ttnn.reshape(t, [b, s, h * d])
        return ttnn.to_layout(t, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ------------------------------------------------------------------ full attention
    def _apply_partial_rope(self, x, cos, sin):
        """Rotate the first ``rope_dim`` of each head; pass the rest through.

        ``x``: ``[B, heads, T, head_dim]``; ``cos``/``sin``: ``[B or 1, 1, T, rope_dim]``.
        """
        rope_dim = self.cfg.rope_dim
        head_dim = x.shape[-1]
        b, h, t = x.shape[0], x.shape[1], x.shape[2]
        half = rope_dim // 2
        x_rot, rot_owned = _slice_owned(x, [0, 0, 0, 0], [b, h, t, rope_dim])
        x1 = ttnn.slice(x_rot, [0, 0, 0, 0], [b, h, t, half])
        x2 = ttnn.slice(x_rot, [0, 0, 0, half], [b, h, t, rope_dim])
        rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        out = ttnn.add(ttnn.multiply(x_rot, cos), ttnn.multiply(rotated, sin))
        if rot_owned:
            ttnn.deallocate(x_rot)
        ttnn.deallocate(rotated)
        if rope_dim < head_dim:
            passthrough = ttnn.slice(x, [0, 0, 0, rope_dim], [b, h, t, head_dim])
            merged = ttnn.concat([out, passthrough], dim=-1)
            ttnn.deallocate(out)
            ttnn.deallocate(passthrough)
            return merged
        return out

    def _project_qkv(self, x):
        """Q/K/V plus the sigmoid output gate carried in the 2×-wide ``q_proj``.

        Returns ``(q, k, v, gate)`` with ``q``: ``[B, n_heads, T, head_dim]``,
        ``k``/``v``: ``[B, n_kv_heads, T, head_dim]``, ``gate``: ``[B, T, n_heads*head_dim]``.
        """
        n_heads, n_kv, head_dim = self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim

        qg = ttnn.linear(x, self.w["q_proj"], compute_kernel_config=self.compute_kernel_config)
        qg = self._split_heads(qg, n_heads, head_dim * 2)
        q, gate = ttnn.chunk(qg, 2, dim=-1)
        ttnn.deallocate(qg)
        gate = self._merge_heads(gate)
        q = self._norm(q, self.w["q_norm"])
        q = ttnn.transpose(q, 1, 2)

        k = ttnn.linear(x, self.w["k_proj"], compute_kernel_config=self.compute_kernel_config)
        k = self._split_heads(k, n_kv, head_dim)
        k = self._norm(k, self.w["k_norm"])
        k = ttnn.transpose(k, 1, 2)

        v = ttnn.linear(x, self.w["v_proj"], compute_kernel_config=self.compute_kernel_config)
        v = self._split_heads(v, n_kv, head_dim)
        v = ttnn.transpose(v, 1, 2)
        return q, k, v, gate

    def _attention_output(self, attn, gate):
        """``sigmoid`` gate + output projection. ``attn``: ``[B, T, n_heads*head_dim]``."""
        gated = ttnn.multiply(attn, ttnn.sigmoid(gate))
        ttnn.deallocate(attn)
        ttnn.deallocate(gate)
        out = ttnn.linear(gated, self.w["o_proj"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(gated)
        return out

    def _prefill_sdpa_config(self, chunk_start_idx: int, phys_len: int):
        """Chunked-SDPA tiling. ``q_chunk`` must divide ``chunk_start_idx`` when it is non-zero."""
        qk = 64
        if chunk_start_idx:
            qk = min(qk, chunk_start_idx & -chunk_start_idx)
        qk = min(qk, phys_len)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=qk,
            k_chunk_size=qk,
            exp_approx_mode=False,
        )

    def _attention_prefill(self, x, page_table, chunk_start_idx):
        if self.k_cache is None:
            raise RuntimeError("call allocate_kv_cache()/attach_kv_cache() before prefill")
        if page_table is None:
            raise ValueError("full_attention prefill requires a page_table")
        b, t = x.shape[0], x.shape[1]
        cos, sin = self.rope.prefill_forward(chunk_start_idx, t)
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
        for user in range(b):
            k_user = ttnn.slice(k, [user, 0, 0, 0], [user + 1, self.cfg.n_kv_heads, t, self.cfg.head_dim])
            v_user = ttnn.slice(v, [user, 0, 0, 0], [user + 1, self.cfg.n_kv_heads, t, self.cfg.head_dim])
            ttnn.experimental.paged_fill_cache(self.k_cache, k_user, chunk_page_table, batch_idx=user)
            ttnn.experimental.paged_fill_cache(self.v_cache, v_user, chunk_page_table, batch_idx=user)
            ttnn.deallocate(k_user)
            ttnn.deallocate(v_user)
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
        attn = self._merge_heads(ttnn.transpose(attn, 1, 2))
        return self._attention_output(attn, gate)

    def _kv_update_memory_config(self, batch_size):
        """Height-sharded L1 config for ``paged_update_cache``: one user per core.

        Uses a row-wise core *range set* of exactly ``batch_size`` cores rather than a rectangular
        ``CoreGrid``. A rectangle would restrict the batch to values with a factor pair fitting the
        grid in both axes — batch 13 has none on an 11×10 grid — whereas any batch up to the core
        count is a legal row-wise set.
        """
        grid = self.device.compute_with_storage_grid_size()
        cores = grid.x * grid.y
        if batch_size > cores:
            raise ValueError(
                f"decode batch {batch_size} exceeds the {cores} cores available for the "
                f"one-user-per-core paged_update_cache shard on a {grid.x}x{grid.y} grid"
            )
        shard_grid = ttnn.num_cores_to_corerangeset(batch_size, grid, row_wise=True)
        shard_spec = ttnn.ShardSpec(shard_grid, [TILE, self.cfg.head_dim], ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    def _attention_decode(self, x, current_pos, rot_idxs, page_table):
        if self.k_cache is None:
            raise RuntimeError("call allocate_kv_cache()/attach_kv_cache() before decode")
        if page_table is None:
            raise ValueError("full_attention decode requires a page_table")
        b = x.shape[0]
        n_heads, n_kv, head_dim = self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim

        cos, sin = self.rope.decode_forward(rot_idxs)  # [1, B, rope_dim]
        cos = ttnn.reshape(cos, [b, 1, 1, self.cfg.rope_dim])
        sin = ttnn.reshape(sin, [b, 1, 1, self.cfg.rope_dim])

        q, k, v, gate = self._project_qkv(x)
        q = self._apply_partial_rope(q, cos, sin)
        k = self._apply_partial_rope(k, cos, sin)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

        # paged_update_cache wants [1, B, kv_heads(pad 32), head_dim] height-sharded on B cores.
        shard_cfg = self._kv_update_memory_config(b)
        for cache, tensor in ((self.k_cache, k), (self.v_cache, v)):
            upd = ttnn.reshape(ttnn.transpose(tensor, 1, 2), [1, b, n_kv, head_dim])
            upd = _pad_dim(upd, 2, TILE - n_kv)
            upd = ttnn.to_memory_config(upd, shard_cfg)
            ttnn.experimental.paged_update_cache(cache, upd, update_idxs_tensor=current_pos, page_table=page_table)
            ttnn.deallocate(upd)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # paged SDPA decode wants Q as [1, batch, n_heads, head_dim] (and in DRAM when not
        # sharded). transpose gives [batch, 1, n_heads, head_dim]; the reshape swapping the two
        # leading dims is a pure relabel because the sequence extent is 1.
        q_decode = ttnn.reshape(ttnn.transpose(q, 1, 2), [1, b, n_heads, head_dim])
        q_decode = ttnn.to_memory_config(q_decode, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_decode,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            is_causal=True,
            scale=head_dim**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=32,
                k_chunk_size=64,
                exp_approx_mode=False,
            ),
        )
        ttnn.deallocate(q_decode)
        # [1, B, n_heads, D] -> [B, 1, n_heads*D]: memory order already (B, head, dim).
        attn = ttnn.reshape(attn, [b, 1, n_heads * head_dim])
        return self._attention_output(attn, gate)

    # ------------------------------------------------------------------ gated deltanet
    def _causal_conv(self, qkv, logical_len):
        """Depthwise causal conv1d + SiLU over the fused QKV stream.

        ``qkv``: ``[B, T, conv_dim]``. Returns ``(activated, tail)`` where ``tail`` is the
        ``kernel-1`` real inputs ending at ``logical_len`` — the conv history the next block
        or the next decode step must start from.
        """
        t = int(qkv.shape[1])
        kernel = self.cfg.linear_conv_kernel_dim
        history = ttnn.concat(self.conv_state, dim=1) if len(self.conv_state) > 1 else self.conv_state[0]
        padded = ttnn.concat([history, qkv], dim=1)
        if len(self.conv_state) > 1:
            ttnn.deallocate(history)

        acc = None
        for tap in range(kernel):
            piece = padded[:, tap : tap + t, :]
            piece = ttnn.to_layout(piece, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if acc is None:
                acc = ttnn.multiply(piece, self.w["conv_taps"][tap], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                acc = ttnn.addcmul(acc, piece, self.w["conv_taps"][tap], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(piece)
        activated = ttnn.silu(acc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(acc)

        tail = padded[:, logical_len : logical_len + kernel - 1, :]
        tail = ttnn.to_layout(tail, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(padded)
        return activated, tail

    def _write_conv_state(self, tail):
        """Copy the new conv history into the persistent buffers, preserving addresses."""
        for idx, buf in enumerate(self.conv_state):
            row = tail[:, idx : idx + 1, :]
            row = ttnn.to_layout(row, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.copy(row, buf)
            ttnn.deallocate(row)

    def _gdn_gates(self, x, logical_len):
        """``(beta, g)`` as float32 ``[B, T, num_v_heads]``, with the padded tail neutralised.

        HF: ``beta = sigmoid(in_proj_b(x))`` and
        ``g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)`` (computed in float32).

        Zeroing both past ``logical_len`` makes each padded step an exact identity on the
        recurrent state: ``beta = 0`` kills the delta write and ``g = 0`` kills the decay.
        """
        t = x.shape[1]
        beta = ttnn.sigmoid(ttnn.linear(x, self.w["gdn_b"], compute_kernel_config=self.compute_kernel_config))
        beta = ttnn.typecast(beta, ttnn.float32)
        a = ttnn.typecast(
            ttnn.linear(x, self.w["gdn_a"], compute_kernel_config=self.compute_kernel_config), ttnn.float32
        )
        a = ttnn.add(a, self.w["dt_bias"])
        g = ttnn.multiply(self.w["A_neg"], ttnn.softplus(a))
        ttnn.deallocate(a)

        if logical_len < t:
            ramp = ttnn.slice(self.w["pos_ramp"], [0, 0, 0], [1, t, 1])
            keep = ttnn.typecast(ttnn.lt(ramp, float(logical_len)), ttnn.float32)
            ttnn.deallocate(ramp)
            beta_masked = ttnn.multiply(beta, keep)
            g_masked = ttnn.multiply(g, keep)
            ttnn.deallocate(keep)
            ttnn.deallocate(beta)
            ttnn.deallocate(g)
            beta, g = beta_masked, g_masked
        return beta, g

    def _gdn_out(self, core, x):
        """Gated output norm + out projection. ``core``: ``[B, T, num_v_heads, head_v_dim]``.

        HF ``Qwen3_5MoeRMSNormGated``: plain RMSNorm over ``head_v_dim`` (weights ≈ 1, *not*
        zero-centered) then multiplied by ``silu(z)``.
        """
        z = ttnn.linear(x, self.w["gdn_z"], compute_kernel_config=self.compute_kernel_config)
        z = self._split_heads(z, self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim)
        normed = ttnn.rms_norm(core, weight=self.w["gdn_norm"], epsilon=self.cfg.norm_eps)
        ttnn.deallocate(core)
        out = ttnn.multiply(normed, ttnn.silu(z))
        ttnn.deallocate(normed)
        ttnn.deallocate(z)
        out = self._merge_heads(out)
        return ttnn.linear(out, self.w["gdn_out"], compute_kernel_config=self.compute_kernel_config)

    def _gdn_split_qkv(self, activated):
        """Split the conv output into per-head Q/K/V."""
        cfg = self.cfg
        q_end = cfg.linear_q_dim
        k_end = q_end + cfg.linear_k_dim
        q = ttnn.slice(activated, [0, 0, 0], [activated.shape[0], activated.shape[1], q_end])
        k = ttnn.slice(activated, [0, 0, q_end], [activated.shape[0], activated.shape[1], k_end])
        v = ttnn.slice(activated, [0, 0, k_end], [activated.shape[0], activated.shape[1], cfg.conv_dim])
        q = self._split_heads(q, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        k = self._split_heads(k, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        v = self._split_heads(v, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
        return q, k, v

    def max_gdn_prefill_batch(self) -> int:
        """Largest batch one ``chunk_gated_delta_rule`` launch can serve.

        The op's phased scan gives one core per ``(batch, value-head)`` pair and asserts
        ``batch * num_value_heads <= compute_cores`` (``chunk_gdn_phased_program_factory.cpp``:
        ``BH <= ncores``). On an 11×10 Blackhole grid with 32 value heads that is 3 users per
        launch, so :meth:`_chunk_delta_rule` splits larger batches and stitches the results —
        which is exact, because the recurrence factorises over the batch axis.
        """
        grid = self.device.compute_with_storage_grid_size()
        return max(1, (grid.x * grid.y) // self.cfg.linear_num_value_heads)

    def _chunk_delta_rule(self, q, k, v, g, beta):
        """Chunk-parallel gated delta rule over the whole block, sub-batched if needed.

        Returns ``(o, final_state)`` for the full batch.
        """
        cfg = self.cfg
        eye, tril, ones, masks = self.w["gdn_const_tiles"]
        batch, seq = q.shape[0], q.shape[1]
        nk, dk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
        nv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim

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
            q_s = ttnn.slice(q, [start, 0, 0, 0], [end, seq, nk, dk])
            k_s = ttnn.slice(k, [start, 0, 0, 0], [end, seq, nk, dk])
            v_s = ttnn.slice(v, [start, 0, 0, 0], [end, seq, nv, dv])
            g_s = ttnn.slice(g, [start, 0, 0], [end, seq, nv])
            beta_s = ttnn.slice(beta, [start, 0, 0], [end, seq, nv])
            state_s = ttnn.slice(self.recurrent_state, [start, 0, 0, 0], [end, nv, dk, dv])
            core_s, final_s = launch(q_s, k_s, v_s, g_s, beta_s, state_s)
            for tensor in (q_s, k_s, v_s, g_s, beta_s, state_s):
                ttnn.deallocate(tensor)
            cores.append(core_s)
            states.append(final_s)
        core = ttnn.concat(cores, dim=0)
        final_state = ttnn.concat(states, dim=0)
        for tensor in cores + states:
            ttnn.deallocate(tensor)
        return core, final_state

    def _gdn_prefill(self, x, logical_len):
        """Chunk-parallel gated delta rule over a whole prefill block."""
        qkv = ttnn.linear(x, self.w["gdn_qkv"], compute_kernel_config=self.compute_kernel_config)
        activated, tail = self._causal_conv(qkv, logical_len)
        ttnn.deallocate(qkv)
        q, k, v = self._gdn_split_qkv(activated)
        ttnn.deallocate(activated)
        beta, g = self._gdn_gates(x, logical_len)

        # The op does not L2-normalise Q/K (its contract asserts use_qk_l2norm is off) and it
        # expands the 16 K-heads to the 32 V-heads internally. Force the normalised Q/K back to
        # DRAM: l2_norm_ttnn lands short sequences in L1, whose per-core footprint clashes with
        # the chunk-GDN kernel's static circular buffers once the batch grows.
        q = ttnn.to_memory_config(l2_norm_ttnn(q), ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(l2_norm_ttnn(k), ttnn.DRAM_MEMORY_CONFIG)
        core, final_state = self._chunk_delta_rule(q, k, v, g, beta)
        for tensor in (q, k, v, g, beta):
            ttnn.deallocate(tensor)
        ttnn.copy(final_state, self.recurrent_state)
        ttnn.deallocate(final_state)
        self._write_conv_state(tail)
        ttnn.deallocate(tail)

        core = ttnn.to_layout(core, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._gdn_out(core, x)

    def _gdn_decode(self, x):
        """One recurrent gated-delta-rule step, updating conv + recurrent state in place."""
        cfg = self.cfg
        b = x.shape[0]
        nv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim

        qkv = ttnn.linear(x, self.w["gdn_qkv"], compute_kernel_config=self.compute_kernel_config)
        kernel = cfg.linear_conv_kernel_dim
        acc = ttnn.multiply(qkv, self.w["conv_taps"][kernel - 1])
        for tap in range(kernel - 1):
            acc = ttnn.addcmul(acc, self.conv_state[tap], self.w["conv_taps"][tap])
        activated = ttnn.silu(acc)
        ttnn.deallocate(acc)
        # Shift the history: oldest out, this step's pre-activation input in. Read-before-write
        # order makes the in-place chain correct and keeps buffer addresses stable.
        for idx in range(kernel - 2):
            ttnn.copy(self.conv_state[idx + 1], self.conv_state[idx])
        ttnn.copy(qkv, self.conv_state[kernel - 2])
        ttnn.deallocate(qkv)

        q, k, v = self._gdn_split_qkv(activated)
        ttnn.deallocate(activated)
        beta, g = self._gdn_gates(x, 1)

        repeats = nv // cfg.linear_num_key_heads
        if repeats > 1:
            q = ttnn.repeat_interleave(q, repeats, dim=2)
            k = ttnn.repeat_interleave(k, repeats, dim=2)
        core = self._delta_rule_step(q, k, v, beta, g)
        for tensor in (q, k, v, beta, g):
            ttnn.deallocate(tensor)
        core = ttnn.reshape(core, [b, 1, nv, dv])
        return self._gdn_out(core, x)

    def _delta_rule_step(self, q, k, v, beta, g):
        """Single gated-delta-rule step on the persistent recurrent state.

        ``q``/``k``: ``[B, 1, num_v_heads, head_k_dim]`` (already GVA-expanded), ``v``:
        ``[B, 1, num_v_heads, head_v_dim]``, ``beta``/``g``: ``[B, 1, num_v_heads]`` float32.

        Mirrors HF ``torch_recurrent_gated_delta_rule``: L2-normalise Q/K, scale Q by
        ``head_k_dim ** -0.5``, decay the state, read ``k @ h``, write the ``beta``-weighted
        delta outer product, then read ``q @ h``. Everything stays in DRAM float32 so the
        state is exact for any batch size.
        """
        cfg = self.cfg
        b = q.shape[0]
        nv, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        dram = ttnn.DRAM_MEMORY_CONFIG

        q = ttnn.typecast(l2_norm_ttnn(q), ttnn.float32)
        k = ttnn.typecast(l2_norm_ttnn(k), ttnn.float32)
        # HF's torch_recurrent_gated_delta_rule applies the same `scale = 1/sqrt(head_k_dim)` to q
        # (its `query = query * scale`), so this matches the reference. Note the layer's PCC cannot
        # discriminate it either way: `Qwen3_5MoeRMSNormGated` immediately normalises the output,
        # which cancels any uniform per-(token, head) factor. Do not "simplify" it away — the
        # chunked prefill op applies the same scale internally, so dropping it here would make
        # prefill and decode disagree on the intermediate `o`, only masked by the following norm.
        q = ttnn.multiply(q, dk**-0.5, memory_config=dram)

        q_row = ttnn.reshape(q, [b, nv, 1, dk])
        k_row = ttnn.reshape(k, [b, nv, 1, dk])
        v_row = ttnn.reshape(ttnn.typecast(v, ttnn.float32), [b, nv, 1, dv])
        beta_b = ttnn.reshape(beta, [b, nv, 1, 1])
        decay = ttnn.exp(ttnn.reshape(g, [b, nv, 1, 1]), memory_config=dram)

        state = self.recurrent_state
        ttnn.multiply(state, decay, output_tensor=state)
        ttnn.deallocate(decay)

        v_read = ttnn.matmul(k_row, state, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        delta = ttnn.multiply(ttnn.subtract(v_row, v_read, memory_config=dram), beta_b, memory_config=dram)
        ttnn.deallocate(v_read)
        k_col = ttnn.transpose(k_row, -1, -2)
        outer = ttnn.matmul(k_col, delta, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(k_col)
        ttnn.deallocate(delta)
        ttnn.add(state, outer, output_tensor=state)
        ttnn.deallocate(outer)

        out = ttnn.matmul(q_row, state, memory_config=dram, compute_kernel_config=self.compute_kernel_config)
        return out

    # ------------------------------------------------------------------ public forwards
    def _block(self, x, *, mode, logical_len=None, page_table=None, chunk_start_idx=0, current_pos=None, rot_idxs=None):
        """One decoder block: norm → mixer → residual → norm → MoE → residual."""
        b, t = x.shape[0], x.shape[1]
        attn_in = self._norm(x, self.w["attn_norm"])
        if self.is_full_attention:
            if mode == "prefill":
                mixed = self._attention_prefill(attn_in, page_table, chunk_start_idx)
            else:
                mixed = self._attention_decode(attn_in, current_pos, rot_idxs, page_table)
        else:
            if mode == "prefill":
                mixed = self._gdn_prefill(attn_in, logical_len)
            else:
                mixed = self._gdn_decode(attn_in)
        ttnn.deallocate(attn_in)

        h = ttnn.add(x, mixed)
        ttnn.deallocate(mixed)

        # The MoE is token-parallel: fold batch and sequence into one token axis. Its sparsity
        # granularity is one 32-token tile, so pad the token axis up and slice the result back
        # (decode with batch < 32 is the case that needs this).
        tokens = b * t
        padded_tokens = _align_up(tokens, TILE)
        ff_in = ttnn.reshape(self._norm(h, self.w["ff_norm"]), [1, 1, tokens, self.cfg.dim])
        if padded_tokens != tokens:
            # Do not free the pre-pad tensor: ttnn.pad may alias it (see _pad_dim). One
            # deallocation of the final tensor releases the shared buffer exactly once.
            ff_in = _pad_dim(ff_in, 2, padded_tokens - tokens)
        ff_out = self.moe.forward(ff_in)
        ttnn.deallocate(ff_in)
        if padded_tokens != tokens:
            trimmed = ttnn.slice(ff_out, [0, 0, 0, 0], [1, 1, tokens, self.cfg.dim])
            ttnn.deallocate(ff_out)
            ff_out = trimmed
        ff_out = ttnn.reshape(ff_out, [b, t, self.cfg.dim])
        out = ttnn.add(h, ff_out)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out

    def prefill_forward(self, x, *, start_pos: int = 0, page_table=None, chunk_size: int | None = None):
        """See the module docstring for the full contract."""
        b, seq_len, dim = x.shape[0], x.shape[1], x.shape[2]
        if dim != self.cfg.dim:
            raise ValueError(f"hidden size {dim} != {self.cfg.dim}")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if start_pos + seq_len > self.max_context:
            raise ValueError(
                f"prefill window [{start_pos}, {start_pos + seq_len}) exceeds supported context {self.max_context}"
            )
        chunk_size = chunk_size or self.prefill_chunk
        if chunk_size % PREFILL_ALIGN:
            raise ValueError(f"chunk_size {chunk_size} must be a multiple of {PREFILL_ALIGN}")
        if start_pos % chunk_size:
            raise ValueError(f"start_pos {start_pos} must be a multiple of chunk_size {chunk_size}")
        if not self.is_full_attention and chunk_size > self.w["pos_ramp"].shape[1]:
            raise ValueError(
                f"chunk_size {chunk_size} exceeds the position ramp built for {self.w['pos_ramp'].shape[1]}"
            )
        if self.batch_size is None:
            self.allocate_state(b)
        elif not self.is_full_attention and b != self.batch_size:
            raise ValueError(f"batch {b} != allocated state batch {self.batch_size}")

        outputs = []
        for offset in range(0, seq_len, chunk_size):
            logical = min(chunk_size, seq_len - offset)
            phys = min(chunk_size, _align_up(logical, PREFILL_ALIGN))
            block, owned = _slice_owned(x, [0, offset, 0], [b, offset + logical, dim])
            if phys > logical:
                # ttnn.pad may alias `block` (see _pad_dim), so ownership does not change: the
                # padded tensor is only ours to free when the slice above allocated a new buffer.
                block = _pad_dim(block, 1, phys - logical)
            out = self._block(
                block,
                mode="prefill",
                logical_len=logical,
                page_table=page_table,
                chunk_start_idx=start_pos + offset,
            )
            if owned:
                ttnn.deallocate(block)
            if phys > logical:
                trimmed = ttnn.slice(out, [0, 0, 0], [b, logical, dim])
                ttnn.deallocate(out)
                out = trimmed
            outputs.append(out)

        if len(outputs) == 1:
            return outputs[0]
        merged = ttnn.concat(outputs, dim=1)
        for out in outputs:
            ttnn.deallocate(out)
        return merged

    def decode_forward(self, x, *, current_pos=None, rot_idxs=None, page_table=None):
        """See the module docstring for the full contract."""
        b, t, dim = x.shape[0], x.shape[1], x.shape[2]
        if t != 1:
            raise ValueError(f"decode expects seq_len 1, got {t}")
        if dim != self.cfg.dim:
            raise ValueError(f"hidden size {dim} != {self.cfg.dim}")
        if self.is_full_attention and (current_pos is None or rot_idxs is None):
            raise ValueError("full_attention decode requires current_pos and rot_idxs device tensors")
        if self.batch_size is None:
            self.allocate_state(b)
        elif not self.is_full_attention and b != self.batch_size:
            raise ValueError(f"batch {b} != allocated state batch {self.batch_size}")
        return self._block(x, mode="decode", current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused TTNN decoder for Qwen/Qwen3.6-27B (HF ``model_type: qwen3_5``).

This is the graph-fused sibling of :mod:`.functional_decoder`.  Same math, same public
contract (with the one documented exception below), fewer and larger ops.  The functional
module stays as the unfused reference the equivalence tests compare against.

What was fused, and why
-----------------------

**Dedicated fused ops** (highest priority — a hand-written kernel replaces a spelled-out
primitive sequence):

``F1`` SwiGLU MLP
    ``slice → slice → silu → multiply`` becomes :func:`ttnn.swiglu`.  ``ttnn.swiglu(x)`` is
    ``x[..., :n] * silu(x[..., n:])``, so the gate/up weights are concatenated **up first**
    (the functional module concatenates gate first).
``F2`` Partial rotary embedding
    ``slice ×3 → neg → concat → mul → mul → add → slice → concat`` (ten ops per tensor)
    becomes a single :func:`ttnn.experimental.rotary_embedding_hf`.  That op applies the
    HF *rotate-half* over the **whole** head, while Qwen3.5 rotates only the leading
    ``rotary_dim`` (64 of 256) channels.  The two are made to agree by permuting the head
    channels of ``q``/``k`` (and of ``q_norm``/``k_norm``) **host-side, at load time** so the
    rotary pair ``(j, j + rotary_dim/2)`` lands on ``(j, j + head_dim/2)`` — exactly the pair
    a full-width rotate-half touches — and by giving the non-rotary channels ``cos = 1``,
    ``sin = 0``.  See :func:`rope_channel_permutation`.  A permutation applied identically to
    ``q`` and ``k`` leaves ``q · k`` unchanged, so attention is unaffected; ``v``, the output
    gate and ``o_proj`` are untouched.
``F3`` L2 normalisation of the gated-delta-net ``q``/``k``
    ``multiply → sum → add → rsqrt → multiply`` (plus a scale multiply for ``q``) becomes one
    :func:`ttnn.rms_norm`.  ``rms_norm(x, eps') = x·sqrt(D)/sqrt(sum(x²) + D·eps')``, so with
    ``eps' = 1e-6/D`` it *is* the HF ``l2norm(x, eps=1e-6)`` up to the constant ``sqrt(D)``,
    which is folded into the norm weight together with the ``1/sqrt(head_k_dim)`` query scale.
``F4`` Fused paged cache update — **assessed and rejected**
    :func:`ttnn.experimental.paged_fused_update_cache` would merge the two decode
    ``paged_update_cache`` dispatches, but it requires its two inputs on disjoint core ranges
    while ``nlp_create_qkv_heads_decode`` puts K and V on the same batch cores; the reshard
    that would fix it costs the dispatch the fusion saves.  See ``_full_attention_decode``.

**Graph rewrites** (structural / algebraic, no new kernel):

``F5`` Shared-LHS projections
    ``in_proj_b`` and ``in_proj_a`` share their LHS, so they become one matmul over the
    concatenated weight (padded so both slices start on a tile boundary).
``F6`` L1 residency for the gated-delta-rule small tensors
    Not an op-count change but the single largest measured win: a batched ``32x32x32``
    matmul costs ~1.06 µs per batch element out of DRAM and ~0.043 µs out of L1 (24x —
    ``doc/fused_decoder/probes/probe_fused_ops2.py``).  The triangular inverse is thousands of
    such matmuls, so the recursion, the decay masks and the per-chunk loop run in L1 whenever
    the chunk's footprint fits :data:`L1_BUDGET_BYTES`.
``F7`` ``TRI_INV_BASE`` 16 → 32
    One recursion level fewer, and the base case's ``32x32`` blocks fill a whole tile instead
    of wasting three quarters of one.
``F8`` decode conv state kept as ``conv_kernel_size`` separate row buffers
    removes the ``slice`` + ``concat`` that rebuilt the window every step.

**Op merging** (fold a neighbour into an op that is already running):

``F9``  ``matmul → sigmoid`` on the attention output gate → ``ttnn.linear(activation="sigmoid")``.
``F10`` ``matmul → silu`` on the gated-delta-net ``z`` → ``ttnn.linear(activation="silu")``.
``F11`` ``matmul → add(dt_bias)`` → ``ttnn.linear(bias=...)`` on the fused ``b|a`` projection.
``F12`` ``transpose → matmul`` → ``ttnn.matmul(..., transpose_b=True)``.
``F13`` ``tril → exp → tril`` for the decay mask → ``add(triu_neg_inf) → exp``.
``F14`` ``multiply → tril → neg`` for ``attn0`` → ``multiply(..., activation) → tril``.

Contract differences from :class:`~.functional_decoder.FunctionalDecoder`
------------------------------------------------------------------------

Exactly one, and it is a consequence of ``F2``:

* ``rot_mats`` are ``head_dim``-wide, not ``rotary_dim``-wide, and carry ``cos = 1`` /
  ``sin = 0`` in the non-rotary channels, in the permuted channel order.
  :func:`rope_channel_permutation` and :data:`ROT_MAT_WIDTH_IS_HEAD_DIM` describe the layout;
  the test harness builds them with ``harness.expand_rot_mats``.
* Consequently the ``full_attention`` **paged K cache holds permuted head channels**.  ``V``
  does not (it never sees RoPE).  :attr:`FusedDecoder.kv_channel_permutation` exposes the
  permutation so a reader can invert it; nothing inside the layer needs to.

Everything else — prefill/decode signatures, arbitrary ``1 <= seq_len <= max_seq_len``, the
paged page-table protocol, per-user state, ``prepare_decode_state`` semantics, determinism and
the "no torch after ``from_state_dict``" rule — is unchanged.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import ttnn

from models.common.lightweightmodule import LightweightModule

from .model_config import DELTA_CHUNK, FULL_ATTENTION, LINEAR_ATTENTION, DecoderShapes, decoder_shapes

#: Tokens processed per prefill chunk.
PREFILL_CHUNK = 2048

#: Padding granularity of a ``full_attention`` prefill chunk, in tokens.
SDPA_CHUNK = 256

#: Cap on the number of k chunks one ``chunked_scaled_dot_product_attention`` may merge; the
#: kernel keeps its running softmax denominator in ``Float16_b`` regardless of
#: ``fp32_dest_acc_en``, so merging more than this silently inflates the output.  Carried over
#: unchanged from the functional decoder, where it was derived and measured.
SDPA_MAX_K_CHUNKS = 512

#: q chunk size to use for each k chunk size (bounded by L1, see the functional decoder).
_SDPA_Q_FOR_K = {256: 256, 512: 64}

#: Decode head padding: TTNN decode attention ops operate on tile-padded head counts.
PADDED_HEADS = 32

#: Default paged-KV block size (tokens per page).
DEFAULT_BLOCK_SIZE = 64

#: Block size at which :meth:`FusedDecoder._unit_tri_inverse` stops recursing.  ``F7``: 32
#: rather than the functional decoder's 16.  The recursion stores its blocks in 32x32 tiles, so
#: a base below 32 pays for three quarters of every tile it touches; 32 is also one recursion
#: level shallower.  The functional stage measured identical layer PCC at bases 8/16/32 and a
#: recurrent-state PCC of 0.999986 at 32 versus 0.999991 at 16, both far above the 0.995 bar.
TRI_INV_BASE = 32

#: Ceiling, in bytes, on the *estimated peak live footprint* a region of the gated-delta-rule
#: path may place in L1 (``F6``).  A Blackhole worker grid exposes 110 banks x 1.43 MB = 157 MB
#: of interleaved L1; the estimates below are peak-live, not per-tensor, and this leaves a
#: comfortable margin for the DRAM-resident tensors' circular buffers.  A region whose estimate
#: exceeds this falls back to DRAM rather than failing to allocate.
L1_BUDGET_BYTES = 96 * 1024 * 1024

#: Marker for readers of the module docstring: ``rot_mats`` are ``head_dim``-wide here.
ROT_MAT_WIDTH_IS_HEAD_DIM = True

#: Value added to the strictly-upper triangle before ``exp`` so it comes out as 0 (``F13``).
_NEG_INF_MASK = -1.0e9


def rope_channel_permutation(head_dim: int, rotary_dim: int) -> list[int]:
    """Head-channel permutation that turns Qwen3.5's *partial* RoPE into a full-width one.

    Returns ``perm`` such that ``permuted[..., j] == original[..., perm[j]]``.

    HF rotates only the leading ``rotary_dim`` channels and pairs channel ``j`` with
    ``j + rotary_dim/2``.  ``ttnn.experimental.rotary_embedding_hf`` rotates the whole head and
    pairs ``j`` with ``j + head_dim/2``.  Moving the second half of the rotary block from
    ``[rotary_dim/2, rotary_dim)`` to ``[head_dim/2, head_dim/2 + rotary_dim/2)`` — and pushing
    the pass-through channels into the gaps — makes the two pairings identical.  The remaining
    channels are neutralised by ``cos = 1``, ``sin = 0``.

    The permutation is applied to the ``q``/``k`` projection weights and to the ``q_norm`` /
    ``k_norm`` weights at load time, so nothing moves at runtime.
    """
    assert head_dim % 2 == 0 and rotary_dim % 2 == 0, "head_dim and rotary_dim must be even"
    assert rotary_dim <= head_dim // 2, (
        f"the permutation needs rotary_dim ({rotary_dim}) <= head_dim/2 ({head_dim // 2}): the "
        "rotary block's two halves must fit either side of the rotate-half midpoint"
    )
    half = rotary_dim // 2
    mid = head_dim // 2
    passthrough = list(range(rotary_dim, head_dim))
    lead = passthrough[: mid - half]
    trail = passthrough[mid - half :]
    return list(range(0, half)) + lead + list(range(half, rotary_dim)) + trail


#: Core grids tried, in order, for the sharded decode RMSNorm (``F15``).  The first whose core
#: count divides the hidden size in tiles wins.  Measured on this Blackhole part for
#: ``[1, 1, 32, 5120]`` (``doc/fused_decoder/logs/probe_sharded_norm.log``): the default
#: interleaved kernel runs the whole norm on **one** core at 103.5 us, while these width-sharded
#: configurations take 20.3 us (5x2), 27.1 us (8x4) and 32.6 us (8x5).  The interleaved kernel
#: parallelises over tile *rows*, and a decode norm has exactly one, which is why it is so slow.
_DECODE_NORM_GRIDS = ((5, 2), (8, 4), (8, 5), (10, 4), (8, 8))


def _decode_norm_config(hidden_size: int, rows: int):
    """``(memory_config_shape, grid, block_h, block_w, subblock_w)`` for the decode RMSNorm.

    Returns ``None`` when no candidate grid divides the hidden size evenly, in which case the
    caller falls back to the interleaved kernel.
    """
    tiles = hidden_size // ttnn.TILE_SIZE
    block_h = _round_up(rows, ttnn.TILE_SIZE) // ttnn.TILE_SIZE
    for grid_x, grid_y in _DECODE_NORM_GRIDS:
        cores = grid_x * grid_y
        if tiles % cores:
            continue
        block_w = tiles // cores
        subblock_w = next(w for w in (4, 2, 1) if block_w % w == 0)
        return (grid_x, grid_y), block_h, block_w, subblock_w
    return None


def _prefill_alignment(layer_type: str, block_size: int) -> int:
    """Padding granularity of a prefill chunk."""
    if layer_type == FULL_ATTENTION:
        return _lcm(SDPA_CHUNK, block_size)
    return DELTA_CHUNK


def _shape(tensor) -> list:
    """``ttnn.Shape`` is not python-sliceable; normalise to a list of ints."""
    return [int(d) for d in tensor.shape]


def _free(tensor, *live) -> None:
    """``ttnn.deallocate`` that is a no-op when ``tensor`` shares a buffer with ``live``.

    Several TTNN ops return a **view** rather than a copy when no data has to move
    (``reshape``/``typecast``/full-range ``slice``/single-tensor ``concat``/no-op
    ``to_memory_config``/``pad`` that fits existing tile padding).  Freeing the input of such
    an op frees the result with it.
    """
    if not tensor.is_allocated():
        return
    address = tensor.buffer_address()
    for other in live:
        if other is not None and other.is_allocated() and other.buffer_address() == address:
            return
    ttnn.deallocate(tensor)


def _hifi4(fp32_dest_acc_en: bool = True) -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )


def _sdpa_program_config(chunk_start_idx: int, kv_len: int) -> ttnn.SDPAProgramConfig:
    """SDPA config for a prefill chunk starting at ``chunk_start_idx`` over ``kv_len`` keys."""
    k_chunk = SDPA_CHUNK
    while kv_len > k_chunk * SDPA_MAX_K_CHUNKS and 2 * k_chunk in _SDPA_Q_FOR_K:
        k_chunk *= 2
    q_chunk = _SDPA_Q_FOR_K[k_chunk]
    assert chunk_start_idx % k_chunk == 0 and chunk_start_idx % q_chunk == 0, (
        f"chunk_start_idx {chunk_start_idx} must be a multiple of the SDPA q chunk {q_chunk} "
        f"and k chunk {k_chunk}"
    )
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


class FusedDecoder(LightweightModule):
    """One graph-fused Qwen3.5/3.6 decoder layer on a TTNN mesh device."""

    def __init__(
        self,
        *,
        shapes: DecoderShapes,
        mesh_device,
        weights: dict,
        max_batch: int,
        max_seq_len: int,
        block_size: int,
        max_num_blocks: int,
        kv_cache: Optional[tuple] = None,
        conv_state=None,
        recurrent_state=None,
        user_conv_state=None,
        user_recurrent_state=None,
        constants: Optional[dict] = None,
    ):
        super().__init__()
        self.shapes = shapes
        self.mesh_device = mesh_device
        self.w = weights
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.kv_cache = kv_cache
        self.conv_state = conv_state
        self.recurrent_state = recurrent_state
        self.user_conv_state: list = list(user_conv_state or [])
        self.user_recurrent_state: list = list(user_recurrent_state or [])
        self.const = constants or {}
        self.compute_cfg = _hifi4(fp32_dest_acc_en=True)
        self.sdpa_compute_cfg = _hifi4(fp32_dest_acc_en=True)
        self.decode_head_mem_cfg = None
        self.decode_rot_mem_cfg = None
        self.kv_channel_permutation = None
        # F15: width-sharded decode RMSNorm.  Both layer kinds use it for the two full-width
        # norms of a decode step, which the interleaved kernel would run single-core.
        self.decode_norm_mem_cfg = None
        self.decode_norm_prog_cfg = None
        norm_plan = _decode_norm_config(shapes.hidden_size, max_batch)
        if norm_plan is not None:
            (grid_x, grid_y), block_h, block_w, subblock_w = norm_plan
            self.decode_norm_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(block_h * ttnn.TILE_SIZE, block_w * ttnn.TILE_SIZE),
                core_grid=ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
                ),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.decode_norm_prog_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(grid_x, grid_y),
                subblock_w=subblock_w,
                block_h=block_h,
                block_w=block_w,
                inplace=False,
            )
        if shapes.layer_type == FULL_ATTENTION:
            self.kv_channel_permutation = rope_channel_permutation(shapes.head_dim, shapes.rotary_dim)
            grid = ttnn.num_cores_to_corerangeset(max_batch, ttnn.CoreCoord(8, 8), row_wise=True)
            # One user per core.  Q/K/V come out of nlp_create_qkv_heads_decode in exactly this
            # layout, and rotary_embedding_hf's decode mode and paged_update_cache both want it.
            self.decode_head_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(PADDED_HEADS, shapes.head_dim),
                core_grid=grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            # cos/sin are logically [1, batch, 1, head_dim]; the tile-padded height is 32, which
            # is the shard height rotary_embedding_hf's decode kernel expects.
            self.decode_rot_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, shapes.head_dim),
                core_grid=grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        max_batch: int = 1,
        max_seq_len: Optional[int] = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_num_blocks: Optional[int] = None,
        weight_dtype=ttnn.bfloat16,
        cache_dtype=ttnn.bfloat16,
        state_dtype=ttnn.float32,
    ) -> "FusedDecoder":
        """Build the layer from an HF **submodule-relative** state dict.

        Identical signature and key set to
        :meth:`~.functional_decoder.FunctionalDecoder.from_state_dict`; the only place
        ``torch`` is used.  All the host-side folding the fused graph needs — the SwiGLU
        weight order, the RoPE channel permutation, the ``b|a`` weight concatenation and its
        ``dt_bias``, the L2-norm scale weights and the decay masks — happens here.
        """
        import torch  # setup-time only; never on the prefill/decode path

        shapes = decoder_shapes(hf_config, layer_idx)
        max_seq_len = max_seq_len or shapes.max_position_embeddings
        if max_num_blocks is None:
            padded_context = _round_up(max_seq_len, _prefill_alignment(shapes.layer_type, block_size))
            max_num_blocks = max_batch * (padded_context // block_size)

        def _get(name: str) -> "torch.Tensor":
            if name not in state_dict:
                raise KeyError(f"missing weight {name!r} for layer {layer_idx} ({shapes.layer_type})")
            return state_dict[name].to(torch.float32)

        def _tt(tensor: "torch.Tensor", dtype=weight_dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                tensor,
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        def _linear_w(tensor: "torch.Tensor", dtype=weight_dtype):
            """torch ``nn.Linear`` weight ``[out, in]`` → ttnn ``[1, 1, in, out]``."""
            return _tt(tensor.t().contiguous().reshape(1, 1, tensor.shape[1], tensor.shape[0]), dtype)

        def _norm_w(tensor: "torch.Tensor", one_centred: bool, dtype=ttnn.bfloat16):
            """Qwen3.5 RMSNorm multiplies by ``1 + weight``; fold the +1 in at load time."""
            value = tensor + 1.0 if one_centred else tensor
            return _tt(value.reshape(1, 1, 1, -1), dtype)

        weights = {
            "input_layernorm": _norm_w(_get("input_layernorm.weight"), one_centred=True),
            "post_attention_layernorm": _norm_w(_get("post_attention_layernorm.weight"), one_centred=True),
        }

        # F1: ttnn.swiglu computes ``first * silu(second)``, so **up** goes first.
        gate_w = _get("mlp.gate_proj.weight")
        up_w = _get("mlp.up_proj.weight")
        weights["mlp_up_gate"] = _linear_w(torch.cat([up_w, gate_w], dim=0))
        weights["mlp_down"] = _linear_w(_get("mlp.down_proj.weight"))

        constants: dict = {}

        if shapes.layer_type == FULL_ATTENTION:
            n_heads, n_kv, head_dim = shapes.num_attention_heads, shapes.num_key_value_heads, shapes.head_dim
            perm = torch.tensor(rope_channel_permutation(head_dim, shapes.rotary_dim), dtype=torch.long)
            # q_proj emits [num_heads, 2 * head_dim]; the per-head second half is the gate.
            q_full = _get("self_attn.q_proj.weight").reshape(n_heads, 2 * head_dim, -1)
            # F2: permute the head channels of q and k (rows of the projection weight) so a
            # full-width rotate-half reproduces Qwen3.5's partial RoPE.  The gate half of
            # q_proj, v and o_proj keep their original channel order.
            q_only = q_full[:, :head_dim, :][:, perm, :].reshape(n_heads * head_dim, -1)
            gate_only = q_full[:, head_dim:, :].reshape(n_heads * head_dim, -1)
            k_w = _get("self_attn.k_proj.weight").reshape(n_kv, head_dim, -1)[:, perm, :]
            k_w = k_w.reshape(n_kv * head_dim, -1)
            v_w = _get("self_attn.v_proj.weight")
            weights["wqkv"] = _linear_w(torch.cat([q_only, k_w, v_w], dim=0))
            weights["wgate"] = _linear_w(gate_only)
            weights["o_proj"] = _linear_w(_get("self_attn.o_proj.weight"))
            weights["q_norm"] = _norm_w(_get("self_attn.q_norm.weight")[perm], one_centred=True)
            weights["k_norm"] = _norm_w(_get("self_attn.k_norm.weight")[perm], one_centred=True)
        else:
            nv, dk = shapes.num_v_heads, shapes.head_k_dim
            weights["in_proj_qkv"] = _linear_w(_get("linear_attn.in_proj_qkv.weight"))
            weights["in_proj_z"] = _linear_w(_get("linear_attn.in_proj_z.weight"))
            weights["out_proj"] = _linear_w(_get("linear_attn.out_proj.weight"))
            weights["gated_norm"] = _norm_w(_get("linear_attn.norm.weight"), one_centred=False)

            # F5 + F11: b and a share their LHS, so one matmul emits both, with dt_bias folded
            # in as the bias of the ``a`` half.  The ``b`` half is padded up to a tile so both
            # output slices start on a tile boundary.
            b_w = _get("linear_attn.in_proj_b.weight")
            a_w = _get("linear_attn.in_proj_a.weight")
            pad = _round_up(nv, ttnn.TILE_SIZE) - nv
            hidden = b_w.shape[1]
            weights["in_proj_ba"] = _linear_w(
                torch.cat([b_w, torch.zeros(pad, hidden), a_w], dim=0), ttnn.float32
            )
            weights["ba_bias"] = _tt(
                torch.cat([torch.zeros(nv + pad), _get("linear_attn.dt_bias")]).reshape(1, 1, 1, -1),
                ttnn.float32,
            )

            # conv1d weight [conv_dim, 1, K] → K taps of shape [1, 1, 1, conv_dim]
            conv_w = _get("linear_attn.conv1d.weight").squeeze(1)
            weights["conv_taps"] = [
                _tt(conv_w[:, j].reshape(1, 1, 1, -1), ttnn.float32) for j in range(shapes.conv_kernel_size)
            ]

            weights["neg_exp_A"] = _tt(
                (-torch.exp(_get("linear_attn.A_log"))).reshape(1, 1, 1, -1), ttnn.float32
            )

            # F3: L2 norm as an rms_norm.  rms_norm(x, eps') = x*sqrt(D)/sqrt(sum(x^2)+D*eps'),
            # so eps' = 1e-6/D reproduces HF's l2norm(x, eps=1e-6) up to sqrt(D).  q additionally
            # carries the 1/sqrt(D) attention scale, so its weight is 1/D.
            weights["q_l2_scale"] = _tt(torch.full((1, 1, 1, dk), 1.0 / dk), ttnn.float32)
            weights["k_l2_scale"] = _tt(torch.full((1, 1, 1, dk), 1.0 / math.sqrt(dk)), ttnn.float32)

            constants["eye_base"] = _tt(
                torch.eye(TRI_INV_BASE).reshape(1, 1, TRI_INV_BASE, TRI_INV_BASE), ttnn.float32
            )
            constants["conv_zero_prefix"] = _tt(
                torch.zeros(1, 1, shapes.conv_kernel_size - 1, shapes.conv_dim), ttnn.float32
            )
            # F13: strictly-upper -inf mask, so one `exp` replaces `tril -> exp -> tril`.
            chunk = DELTA_CHUNK
            triu = torch.triu(torch.full((chunk, chunk), _NEG_INF_MASK), diagonal=1)
            constants["triu_neg_inf"] = _tt(triu.reshape(1, 1, chunk, chunk), ttnn.float32)
            # F14: -1 strictly below the diagonal, 0 elsewhere - folds the `neg` and the `tril`
            # of `attn0` into the multiply that was already there.
            strict = -torch.tril(torch.ones(chunk, chunk), diagonal=-1)
            constants["neg_strict_lower"] = _tt(strict.reshape(1, 1, chunk, chunk), ttnn.float32)

        kv_cache = None
        conv_state = None
        recurrent_state = None
        user_conv_state = None
        user_recurrent_state = None
        if shapes.layer_type == FULL_ATTENTION:
            cache_shape = (max_num_blocks, shapes.num_key_value_heads, block_size, shapes.head_dim)
            kv_cache = tuple(_tt(torch.zeros(cache_shape), cache_dtype) for _ in range(2))
        else:
            conv_state = _tt(
                torch.zeros(1, max_batch, shapes.conv_kernel_size, shapes.conv_dim), state_dtype
            )
            recurrent_state = _tt(
                torch.zeros(1, max_batch * shapes.num_v_heads, shapes.head_k_dim, shapes.head_v_dim),
                state_dtype,
            )
            user_conv_state = [
                _tt(torch.zeros(1, 1, shapes.conv_kernel_size, shapes.conv_dim), state_dtype)
                for _ in range(max_batch)
            ]
            user_recurrent_state = [
                _tt(torch.zeros(1, shapes.num_v_heads, shapes.head_k_dim, shapes.head_v_dim), state_dtype)
                for _ in range(max_batch)
            ]

        return cls(
            shapes=shapes,
            mesh_device=mesh_device,
            weights=weights,
            max_batch=max_batch,
            max_seq_len=max_seq_len,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            kv_cache=kv_cache,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            user_conv_state=user_conv_state,
            user_recurrent_state=user_recurrent_state,
            constants=constants,
        )

    # ------------------------------------------------------------- primitives

    @staticmethod
    def _mem(nbytes: int):
        """L1 for tensors that comfortably fit, DRAM otherwise (``F6``)."""
        return ttnn.L1_MEMORY_CONFIG if nbytes <= L1_BUDGET_BYTES else ttnn.DRAM_MEMORY_CONFIG

    def _tri_mem(self, a, size: int):
        """Memory config for one level of :meth:`_unit_tri_inverse`.

        A level keeps about five tensors of its own block size live at once (the running inverse,
        the running power, the squared power, the ``I + power`` factor and the product), so the
        estimate is five times one block.  Blocks narrower than a tile still occupy a full tile.
        """
        lead, heads = _shape(a)[0], _shape(a)[1]
        tile = max(size, ttnn.TILE_SIZE)
        return self._mem(5 * lead * heads * tile * tile * 4)

    def _rms_norm(self, x, weight, epsilon=None, memory_config=None):
        return ttnn.rms_norm(
            x,
            epsilon=self.shapes.rms_norm_eps if epsilon is None else epsilon,
            weight=weight,
            compute_kernel_config=self.compute_cfg,
            memory_config=memory_config,
        )

    def _decode_rms_norm(self, x, weight):
        """F15: the two full-width decode norms, width-sharded across a core grid.

        A decode activation is one tile row tall, and the interleaved layernorm kernel
        parallelises over rows, so it lands on a single core (measured 102 us of a 2.4 ms
        step, twice).  Sharding along the *width* instead spreads it over the grid; the two
        extra resharding dispatches cost ~1 us each.
        """
        if self.decode_norm_mem_cfg is None:
            return self._rms_norm(x, weight)
        sharded = ttnn.to_memory_config(x, self.decode_norm_mem_cfg)
        normed = ttnn.rms_norm(
            sharded,
            epsilon=self.shapes.rms_norm_eps,
            weight=weight,
            compute_kernel_config=self.compute_cfg,
            program_config=self.decode_norm_prog_cfg,
            memory_config=self.decode_norm_mem_cfg,
        )
        _free(sharded, x)
        out = ttnn.to_memory_config(normed, ttnn.DRAM_MEMORY_CONFIG)
        _free(normed, out)
        return out

    def _mlp(self, x):
        """F1: fused up|gate matmul + ``ttnn.swiglu`` + down projection — three ops."""
        rows = int(x.shape[-2])
        up_gate = ttnn.linear(
            x, self.w["mlp_up_gate"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg
        )
        activated = ttnn.swiglu(up_gate)
        ttnn.deallocate(up_gate)
        # ttnn.swiglu reports the *tile-padded* height as its logical height, so a decode pass
        # with batch < 32 comes back 32 rows tall and would broadcast against the residual.
        # Prefill chunks are always a multiple of the tile, so this only fires on decode.
        if int(activated.shape[-2]) != rows:
            trimmed = ttnn.slice(
                activated, [0, 0, 0, 0], [1, 1, rows, self.shapes.intermediate_size]
            )
            _free(activated, trimmed)
            activated = trimmed
        out = ttnn.linear(
            activated, self.w["mlp_down"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg
        )
        ttnn.deallocate(activated)
        return out

    # ------------------------------------------------------- full attention

    def _attn_projections(self, x, *, decode: bool):
        """Return ``(q, k, v, gate)`` with heads split out.

        ``F9``: the sigmoid of the output gate is the projection's fused activation.
        """
        s = self.shapes
        qkv = ttnn.linear(x, self.w["wqkv"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        gate = ttnn.linear(
            x,
            self.w["wgate"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_cfg,
            activation="sigmoid",
        )
        if decode:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv, num_heads=s.num_attention_heads, num_kv_heads=s.num_key_value_heads
            )
        else:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=s.num_attention_heads,
                num_kv_heads=s.num_key_value_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ttnn.deallocate(qkv)
        return q, k, v, gate

    def _attn_epilogue(self, attn_out, gate):
        """``gate`` already carries its sigmoid (``F9``)."""
        gated = ttnn.multiply(attn_out, gate)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(gate)
        out = ttnn.linear(
            gated, self.w["o_proj"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg
        )
        ttnn.deallocate(gated)
        return out

    def _full_attention_prefill(self, x, *, user_id, page_table, chunk_page_table, rot_mats, chunk_start):
        """Attention over one prefill chunk; ``x`` is ``[1, 1, L_padded, hidden]``."""
        s = self.shapes
        cos, sin = rot_mats
        q, k, v, gate = self._attn_projections(x, decode=False)

        q = self._rms_norm(q, self.w["q_norm"])
        k = self._rms_norm(k, self.w["k_norm"])
        # F2: one op per tensor instead of ten.
        q_rot = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
        ttnn.deallocate(q)
        k_rot = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)
        ttnn.deallocate(k)
        q, k = q_rot, k_rot

        k_cache, v_cache = self.kv_cache
        k_fill = ttnn.typecast(k, k_cache.dtype)
        v_fill = ttnn.typecast(v, v_cache.dtype)
        _free(k, k_fill)
        _free(v, v_fill)
        page_len = int(chunk_page_table.shape[-1]) * self.block_size
        assert page_len == int(k_fill.shape[2]), (
            f"chunk page table covers {page_len} tokens but the padded chunk is "
            f"{int(k_fill.shape[2])}; use prefill_chunk_plan() to size the per-chunk slices"
        )
        ttnn.experimental.paged_fill_cache(k_cache, k_fill, chunk_page_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(v_cache, v_fill, chunk_page_table, batch_idx=0)
        ttnn.deallocate(k_fill)
        ttnn.deallocate(v_fill)

        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            k_cache,
            v_cache,
            page_table,
            chunk_start,
            scale=s.attn_scaling,
            program_config=_sdpa_program_config(chunk_start, chunk_start + page_len),
            compute_kernel_config=self.sdpa_compute_cfg,
        )
        ttnn.deallocate(q)
        concat = ttnn.experimental.nlp_concat_heads(attn)
        ttnn.deallocate(attn)
        return self._attn_epilogue(concat, gate)

    def _full_attention_decode(self, x, *, current_pos, page_table, rot_mats):
        """Decode attention.

        ``F2`` + ``F4``: ``q``/``k`` leave ``nlp_create_qkv_heads_decode`` height-sharded, take a
        detour through DRAM only for the two head norms (``rms_norm`` rejects height-sharded
        inputs), come back sharded for one ``rotary_embedding_hf`` each, and ``k``/``v`` then go
        straight into a single ``paged_fused_update_cache``.  ``v`` never leaves L1.
        """
        s = self.shapes
        cos_in, sin_in = rot_mats
        # The decode rotary kernel reads cos/sin height-sharded, one user per core.  This is a
        # no-op view when the caller already handed them over sharded that way.
        cos = ttnn.to_memory_config(cos_in, self.decode_rot_mem_cfg)
        sin = ttnn.to_memory_config(sin_in, self.decode_rot_mem_cfg)
        q, k, v, gate = self._attn_projections(x, decode=True)

        q_i = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        _free(q, q_i)
        k_i = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        _free(k, k_i)
        q = self._rms_norm(q_i, self.w["q_norm"])
        ttnn.deallocate(q_i)
        k = self._rms_norm(k_i, self.w["k_norm"])
        ttnn.deallocate(k_i)

        q_sh = ttnn.to_memory_config(q, self.decode_head_mem_cfg)
        _free(q, q_sh)
        k_sh = ttnn.to_memory_config(k, self.decode_head_mem_cfg)
        _free(k, k_sh)
        q = ttnn.experimental.rotary_embedding_hf(q_sh, cos, sin, is_decode_mode=True)
        ttnn.deallocate(q_sh)
        k = ttnn.experimental.rotary_embedding_hf(k_sh, cos, sin, is_decode_mode=True)
        ttnn.deallocate(k_sh)
        _free(cos, cos_in)
        _free(sin, sin_in)

        k_cache, v_cache = self.kv_cache
        # v comes out of nlp_create_qkv_heads_decode already in this layout, so this is a view.
        v_sh = ttnn.to_memory_config(v, self.decode_head_mem_cfg)
        _free(v, v_sh)
        # `F4` was tried and rejected here: ttnn.experimental.paged_fused_update_cache would
        # merge these two dispatches into one, but it requires its two inputs to sit on
        # *disjoint* core ranges (paged_fused_update_cache_device_operation.cpp:227), and
        # nlp_create_qkv_heads_decode emits K and V on the same batch cores.  Moving V to a
        # second core range costs exactly the reshard dispatch the fusion would have saved.
        ttnn.experimental.paged_update_cache(
            k_cache, k, update_idxs_tensor=current_pos, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            v_cache, v_sh, update_idxs_tensor=current_pos, page_table=page_table
        )
        ttnn.deallocate(k)
        ttnn.deallocate(v_sh)

        # q has to come back to DRAM for the decode SDPA.  Feeding it the sharded rotary output
        # instead is wrong, not just slower: that kernel derives each user's core from the
        # device grid width (11 on this part), so an 8-wide 32-core shard makes it read user 8
        # onwards off the wrong core - measured PCC 0.999 for users 0-7 and ~0.02 for 8-31.
        # Asking the rotary op for an interleaved output instead does not work either
        # (circular_buffer_config.cpp:222), so the reshard stays.  k keeps its sharded layout,
        # which is what paged_update_cache wants.
        q_dram = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        _free(q, q_dram)
        q = q_dram
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            page_table,
            cur_pos_tensor=current_pos,
            scale=s.attn_scaling,
            # A sharded output would feed nlp_concat_heads_decode directly, but the decode SDPA
            # kernel rejects it for GQA (sdpa_decode_device_operation.cpp:405), so the reshard
            # stays.
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.sdpa_compute_cfg,
        )
        ttnn.deallocate(q)
        attn_sharded = ttnn.to_memory_config(attn, self.decode_head_mem_cfg)
        _free(attn, attn_sharded)
        concat = ttnn.experimental.nlp_concat_heads_decode(attn_sharded, num_heads=s.num_attention_heads)
        ttnn.deallocate(attn_sharded)
        if int(concat.shape[2]) != self.max_batch:
            trimmed = ttnn.slice(
                concat, [0, 0, 0, 0], [1, 1, self.max_batch, s.num_attention_heads * s.head_dim]
            )
            _free(concat, trimmed)
            concat = trimmed
        concat = ttnn.to_memory_config(concat, ttnn.DRAM_MEMORY_CONFIG)
        return self._attn_epilogue(concat, gate)

    # ----------------------------------------------------- linear attention

    def _gdn_inputs(self, x):
        """Shared GatedDeltaNet input projections.

        ``F5`` + ``F11``: one fp32 matmul with a fused bias emits both ``b`` and ``a``.
        ``F10``: ``z``'s SiLU is the projection's fused activation.
        """
        s = self.shapes
        # The ``b`` half is padded up to a tile so ``a`` starts on a tile boundary (F5).
        offset = _round_up(s.num_v_heads, ttnn.TILE_SIZE)
        mixed_qkv = ttnn.linear(
            x, self.w["in_proj_qkv"], dtype=ttnn.float32, compute_kernel_config=self.compute_cfg
        )
        z = ttnn.linear(
            x,
            self.w["in_proj_z"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_cfg,
            activation="silu",
        )
        ba = ttnn.linear(
            x,
            self.w["in_proj_ba"],
            bias=self.w["ba_bias"],
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_cfg,
        )
        lead = _shape(ba)[:-1]
        starts = [0] * len(lead)
        b_part = ttnn.slice(ba, [*starts, 0], [*lead, s.num_v_heads])
        a_part = ttnn.slice(ba, [*starts, offset], [*lead, offset + s.num_v_heads])
        _free(ba, b_part, a_part)
        beta = ttnn.sigmoid(b_part)
        ttnn.deallocate(b_part)
        soft = ttnn.softplus(a_part, beta=1.0, threshold=20.0)
        ttnn.deallocate(a_part)
        g = ttnn.multiply(soft, self.w["neg_exp_A"])
        ttnn.deallocate(soft)
        return mixed_qkv, z, beta, g

    def _causal_conv(self, mixed_qkv, prefix, logical: int):
        """Depthwise causal conv1d (width ``K``) + SiLU, channels-last.

        ``mixed_qkv``: ``[1, 1, L, conv_dim]``; ``prefix``: ``[1, 1, K-1, conv_dim]`` left
        context.  Returns ``(activations, new_state [1, 1, K, conv_dim])``.  The state is
        sliced at the *logical* end, not the padded end, so a zero-padded chunk still stores
        the last ``K`` real rows.
        """
        s = self.shapes
        k = s.conv_kernel_size
        length = int(mixed_qkv.shape[-2])
        window = ttnn.concat([prefix, mixed_qkv], dim=-2)
        # F16: each tap after the first is `acc + tap * w`, which ttnn.addcmul does in one pass
        # instead of a multiply into a temporary followed by an add.  Measured on the real
        # shape ([1, 1, 2051, 10240] fp32, 4 taps): 15.38 ms -> 13.74 ms, three fewer ops and
        # three fewer 84 MB temporaries (doc/fused_decoder/logs/probe_conv.log).
        acc = None
        for j in range(k):
            tap = ttnn.slice(window, [0, 0, j, 0], [1, 1, j + length, s.conv_dim])
            if acc is None:
                acc = ttnn.multiply(tap, self.w["conv_taps"][j])
            else:
                updated = ttnn.addcmul(acc, tap, self.w["conv_taps"][j], value=1.0)
                ttnn.deallocate(acc)
                acc = updated
            ttnn.deallocate(tap)
        new_state = ttnn.slice(window, [0, 0, logical - 1, 0], [1, 1, logical - 1 + k, s.conv_dim])
        _free(window, new_state)
        activations = ttnn.silu(acc)
        ttnn.deallocate(acc)
        return activations, new_state

    def _split_qkv(self, conv_out):
        s = self.shapes
        lead = _shape(conv_out)[:-1]
        starts = [0] * len(lead)
        q = ttnn.slice(conv_out, [*starts, 0], [*lead, s.key_dim])
        k = ttnn.slice(conv_out, [*starts, s.key_dim], [*lead, 2 * s.key_dim])
        v = ttnn.slice(conv_out, [*starts, 2 * s.key_dim], [*lead, s.conv_dim])
        return q, k, v

    def _mm(self, a, b, memory_config=None, transpose_b=False):
        return ttnn.matmul(
            a,
            b,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_cfg,
            memory_config=memory_config,
            transpose_b=transpose_b,
        )

    def _unit_tri_inverse(self, a, size: int):
        """``(I - a)**-1`` for strictly-lower-triangular ``a`` of shape ``[n, heads, size, size]``.

        Recursive 2x2 block inversion (the Neumann doubling product alone cancels catastrophically
        at these magnitudes — see the functional decoder), with both diagonal blocks inverted in
        one batched call.  ``F7``: the recursion bottoms out at :data:`TRI_INV_BASE` = 32, where
        a block fills a whole tile instead of a quarter of one.

        ``F6``: each recursion level picks its own memory config from its own tensor size, so the
        base case — thousands of ``32x32x32`` matmuls, where L1 is 24x faster than DRAM — runs in
        L1 while the wider top level, whose few matmuls are far larger, stays in DRAM.
        """
        mem = self._tri_mem(a, size)
        if size <= TRI_INV_BASE:
            eye = self.const["eye_base"]
            inv = ttnn.add(a, eye, memory_config=mem)
            power = a
            for _ in range(int(math.log2(size)) - 1):
                squared = self._mm(power, power, mem)
                _free(power, a)
                power = squared
                factor = ttnn.add(power, eye, memory_config=mem)
                updated = self._mm(inv, factor, mem)
                ttnn.deallocate(factor)
                ttnn.deallocate(inv)
                inv = updated
            _free(power, a)
            return inv

        half = size // 2
        lead, heads = _shape(a)[0], _shape(a)[1]
        a11 = ttnn.slice(a, [0, 0, 0, 0], [lead, heads, half, half], memory_config=mem)
        a22 = ttnn.slice(a, [0, 0, half, half], [lead, heads, size, size], memory_config=mem)
        a21 = ttnn.slice(a, [0, 0, half, 0], [lead, heads, size, half], memory_config=mem)
        diagonal = ttnn.concat([a11, a22], dim=0, memory_config=mem)
        _free(a11, diagonal)
        _free(a22, diagonal)

        inverted = self._unit_tri_inverse(diagonal, half)
        _free(diagonal, inverted)
        x11 = ttnn.slice(inverted, [0, 0, 0, 0], [lead, heads, half, half], memory_config=mem)
        x22 = ttnn.slice(inverted, [lead, 0, 0, 0], [2 * lead, heads, half, half], memory_config=mem)
        _free(inverted, x11, x22)

        scaled = self._mm(a21, x11, mem)
        ttnn.deallocate(a21)
        x21 = self._mm(x22, scaled, mem)
        ttnn.deallocate(scaled)

        zero = ttnn.zeros(
            (lead, heads, half, half),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=mem,
        )
        top = ttnn.concat([x11, zero], dim=-1, memory_config=mem)
        bottom = ttnn.concat([x21, x22], dim=-1, memory_config=mem)
        ttnn.deallocate(zero)
        _free(x11, top)
        _free(x21, bottom)
        _free(x22, bottom)
        out = ttnn.concat([top, bottom], dim=-2, memory_config=mem)
        _free(top, out)
        _free(bottom, out)
        return out

    def _linear_attention_prefill_chunk(self, x, *, state, conv_prefix, length):
        """One prefill chunk of the gated delta rule.

        ``x``: ``[1, 1, L, hidden]`` (``L`` may exceed ``length`` after zero padding).
        ``state``: ``[1, num_v_heads, head_k_dim, head_v_dim]`` float32 recurrent state.
        Returns ``(out [1, 1, L, value_dim], new_state, new_conv_state)``.
        """
        s = self.shapes
        chunk = DELTA_CHUNK
        mixed_qkv, z, beta, g = self._gdn_inputs(x)
        conv_out, new_conv_state = self._causal_conv(mixed_qkv, conv_prefix, length)
        ttnn.deallocate(mixed_qkv)
        q_flat, k_flat, v_flat = self._split_qkv(conv_out)
        ttnn.deallocate(conv_out)

        padded = int(x.shape[-2])
        assert padded % chunk == 0, f"prefill chunk length {padded} must be a multiple of {chunk}"
        nc = padded // chunk
        nv = s.num_v_heads
        # F6: L1 is 24x faster than DRAM for the many small batched matmuls, but a Blackhole
        # worker grid only exposes ~160 MB of it, so only the two hot regions go there: the
        # triangular inverse (thousands of 32x32 matmuls) and the per-chunk recurrence loop
        # (one chunk's slices at a time).  The full-length tensors - decay, kk, k_beta, the
        # projections - stay in DRAM, where their one big streaming pass costs nothing extra.
        loop_mem = self._mem(16 * nv * chunk * max(s.head_k_dim, s.head_v_dim) * 4)
        mem = None

        # HF pads q/k/v/beta/g with zeros up to the delta-chunk multiple, so a padded position is
        # an exact no-op.  Here the *hidden states* were padded instead, which leaves
        # beta = sigmoid(0) = 0.5 and g != 0, so the state would be decayed and updated by up to
        # chunk - 1 phantom tokens.  Mask them out.
        beta = _zero_after_seq(beta, length, padded)
        g = _zero_after_seq(g, length, padded)

        def to_heads(flat, num_heads, head_dim, repeat: int):
            t = ttnn.reshape(flat, (1, padded, num_heads, head_dim))
            t = ttnn.permute(t, (0, 2, 1, 3))  # [1, H, L, D]
            if repeat > 1:
                rep = ttnn.repeat_interleave(t, repeat, dim=1)
                _free(t, flat, rep)
                t = rep
            t = ttnn.reshape(t, (nv, nc, chunk, head_dim))
            return ttnn.permute(t, (1, 0, 2, 3))  # [nc, H, chunk, D]

        q = to_heads(q_flat, s.num_k_heads, s.head_k_dim, s.v_per_k)
        k = to_heads(k_flat, s.num_k_heads, s.head_k_dim, s.v_per_k)
        v = to_heads(v_flat, nv, s.head_v_dim, 1)
        _free(q_flat, q)
        _free(k_flat, k)
        _free(v_flat, v)

        # F3: l2norm + query scale as a single rms_norm each.
        q = self._l2_rms(q, "q_l2_scale", mem)
        k = self._l2_rms(k, "k_l2_scale", mem)

        def to_scalar_heads(flat):
            t = ttnn.permute(flat, (0, 3, 2, 1))  # [1, nv, L, 1]
            t = ttnn.reshape(t, (nv, nc, chunk, 1))
            return ttnn.permute(t, (1, 0, 2, 3))

        beta_h = to_scalar_heads(beta)
        g_h = to_scalar_heads(g)
        ttnn.deallocate(beta)
        ttnn.deallocate(g)

        g_cum = ttnn.cumsum(g_h, dim=-2)
        ttnn.deallocate(g_h)

        # decay[i, j] = exp(g_cum[i] - g_cum[j]) for i >= j else 0.  F13: the strictly-upper
        # -inf mask does the job of the tril before *and* the tril after the exp.
        g_row = ttnn.transpose(g_cum, -2, -1, memory_config=mem)  # [nc, nv, 1, chunk]
        diff = ttnn.subtract(g_cum, g_row, memory_config=mem)
        ttnn.deallocate(g_row)
        masked = ttnn.add(diff, self.const["triu_neg_inf"], memory_config=mem)
        ttnn.deallocate(diff)
        decay = ttnn.exp(masked, memory_config=mem)
        ttnn.deallocate(masked)

        k_beta = ttnn.multiply(k, beta_h, memory_config=mem)
        v_beta = ttnn.multiply(v, beta_h, memory_config=mem)
        ttnn.deallocate(beta_h)
        ttnn.deallocate(v)

        # F12: transpose folded into the matmul.
        kk = ttnn.matmul(
            k_beta,
            k,
            transpose_b=True,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_cfg,
            memory_config=mem,
        )
        # F14: `neg_strict_lower` carries both the sign and the strict-lower mask.
        scaled_decay = ttnn.multiply(decay, self.const["neg_strict_lower"], memory_config=mem)
        attn0 = ttnn.multiply(kk, scaled_decay, memory_config=mem)
        ttnn.deallocate(scaled_decay)
        ttnn.deallocate(kk)

        inv = self._unit_tri_inverse(attn0, chunk)
        ttnn.deallocate(attn0)

        value = ttnn.matmul(
            inv, v_beta, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg, memory_config=mem
        )
        ttnn.deallocate(v_beta)
        exp_gcum = ttnn.exp(g_cum, memory_config=mem)
        k_beta_decayed = ttnn.multiply(k_beta, exp_gcum, memory_config=mem)
        k_cumdecay = ttnn.matmul(
            inv,
            k_beta_decayed,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_cfg,
            memory_config=mem,
        )
        ttnn.deallocate(k_beta_decayed)
        ttnn.deallocate(inv)
        ttnn.deallocate(k_beta)

        g_last = ttnn.slice(g_cum, [0, 0, chunk - 1, 0], [nc, nv, chunk, 1], memory_config=mem)
        decay_to_end = ttnn.exp(ttnn.subtract(g_last, g_cum, memory_config=mem), memory_config=mem)
        exp_g_last = ttnn.exp(g_last, memory_config=mem)
        ttnn.deallocate(g_last)
        k_decayed = ttnn.multiply(k, decay_to_end, memory_config=mem)
        ttnn.deallocate(decay_to_end)
        q_decayed = ttnn.multiply(q, exp_gcum, memory_config=mem)
        ttnn.deallocate(exp_gcum)

        outputs = []
        for i in range(nc):
            sl = lambda t, w: ttnn.slice(t, [i, 0, 0, 0], [i + 1, nv, chunk, w], memory_config=loop_mem)  # noqa: E731
            q_i = sl(q, s.head_k_dim)
            k_i = sl(k, s.head_k_dim)
            v_i = sl(value, s.head_v_dim)
            d_i = sl(decay, chunk)
            kc_i = sl(k_cumdecay, s.head_k_dim)
            qd_i = sl(q_decayed, s.head_k_dim)
            kd_i = sl(k_decayed, s.head_k_dim)
            gl_i = ttnn.slice(exp_g_last, [i, 0, 0, 0], [i + 1, nv, 1, 1], memory_config=loop_mem)

            intra = ttnn.multiply(
                self._mm(q_i, k_i, loop_mem, transpose_b=True), d_i, memory_config=loop_mem
            )
            v_prime = self._mm(kc_i, state, loop_mem)
            v_new = ttnn.subtract(v_i, v_prime, memory_config=loop_mem)
            ttnn.deallocate(v_prime)
            inter = self._mm(qd_i, state, loop_mem)
            out_i = ttnn.add(inter, self._mm(intra, v_new, loop_mem), memory_config=mem)
            ttnn.deallocate(inter)
            ttnn.deallocate(intra)
            outputs.append(out_i)

            decayed_state = ttnn.multiply(state, gl_i)
            update = ttnn.matmul(
                kd_i,
                v_new,
                transpose_a=True,
                dtype=ttnn.float32,
                compute_kernel_config=self.compute_cfg,
            )
            ttnn.deallocate(v_new)
            new_state = ttnn.add(decayed_state, update)
            ttnn.deallocate(decayed_state)
            ttnn.deallocate(update)
            ttnn.deallocate(state)
            state = new_state
            for tensor in (q_i, k_i, v_i, d_i, kc_i, qd_i, kd_i, gl_i):
                _free(tensor, q, k, value, decay, k_cumdecay, q_decayed, k_decayed, exp_g_last)

        for tensor in (q, k, value, decay, k_cumdecay, q_decayed, k_decayed, exp_g_last, g_cum):
            ttnn.deallocate(tensor)

        if len(outputs) == 1:
            core = outputs[0]
        else:
            core = ttnn.concat(outputs, dim=0, memory_config=mem)  # [nc, nv, chunk, head_v_dim]
            for tensor in outputs:
                ttnn.deallocate(tensor)
        core = ttnn.permute(core, (1, 0, 2, 3))  # [nv, nc, chunk, Dv]
        core = ttnn.reshape(core, (1, nv, padded, s.head_v_dim))
        core = ttnn.permute(core, (0, 2, 1, 3))  # [1, L, nv, Dv]

        z_heads = ttnn.reshape(z, (1, padded, nv, s.head_v_dim))
        normed = self._rms_norm(ttnn.typecast(core, ttnn.bfloat16), self.w["gated_norm"])
        ttnn.deallocate(core)
        gated = ttnn.multiply(normed, z_heads)  # F10: z already carries its SiLU
        ttnn.deallocate(normed)
        ttnn.deallocate(z_heads)
        flat = ttnn.reshape(gated, (1, 1, padded, s.value_dim))
        _free(gated, flat)
        out = ttnn.linear(
            flat, self.w["out_proj"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg
        )
        ttnn.deallocate(flat)
        return out, state, new_conv_state

    def _l2_rms(self, x, weight_key: str, mem=None):
        """F3: HF ``l2norm(x, eps=1e-6)`` (times a constant folded into the weight) as one op."""
        dk = self.shapes.head_k_dim
        out = self._rms_norm(x, self.w[weight_key], epsilon=1e-6 / dk, memory_config=mem)
        ttnn.deallocate(x)
        return out

    def _linear_attention_decode(self, x):
        """Single-token gated delta rule for all ``max_batch`` users at once."""
        s = self.shapes
        batch = self.max_batch
        nv = s.num_v_heads
        # F6: the decode working set is dominated by the recurrent state, batch * nv * dk * dv
        # floats.  At batch 1 that is 3 MB and belongs in L1; at batch 32 it is 100 MB and does
        # not.  Everything here follows the same decision so a step never mixes the two.
        mem = self._mem(3 * batch * nv * s.head_k_dim * s.head_v_dim * 4)
        x_rows = ttnn.reshape(x, (1, batch, 1, s.hidden_size))
        mixed_qkv, z, beta, g = self._gdn_inputs(x_rows)
        _free(x_rows, x)

        # F8: the conv window is `conv_state` rows 1..K-1 followed by this token.  Slicing the
        # state and concatenating is two ops on a K-row tensor; taking the taps straight out of
        # the state buffer and writing the shifted window back is the same arithmetic without
        # the concat.
        acc = ttnn.multiply(mixed_qkv, self.w["conv_taps"][s.conv_kernel_size - 1], memory_config=mem)
        for j in range(s.conv_kernel_size - 1):
            row = ttnn.slice(
                self.conv_state, [0, 0, j + 1, 0], [1, batch, j + 2, s.conv_dim], memory_config=mem
            )
            # F16: one addcmul instead of multiply-then-add.
            updated = ttnn.addcmul(acc, row, self.w["conv_taps"][j], value=1.0, memory_config=mem)
            ttnn.deallocate(acc)
            ttnn.deallocate(row)
            acc = updated
        window = ttnn.concat(
            [
                ttnn.slice(self.conv_state, [0, 0, 1, 0], [1, batch, s.conv_kernel_size, s.conv_dim]),
                mixed_qkv,
            ],
            dim=-2,
        )
        ttnn.deallocate(mixed_qkv)
        ttnn.copy(window, self.conv_state)
        ttnn.deallocate(window)
        conv_out = ttnn.silu(acc, memory_config=mem)
        ttnn.deallocate(acc)

        q_flat, k_flat, v_flat = self._split_qkv(conv_out)
        ttnn.deallocate(conv_out)

        def to_heads(flat, num_heads, head_dim, repeat: int):
            t = ttnn.reshape(flat, (1, batch, num_heads, head_dim))
            if repeat > 1:
                rep = ttnn.repeat_interleave(t, repeat, dim=2)
                _free(t, flat, rep)
                t = rep
            return ttnn.reshape(t, (1, batch * nv, 1, head_dim))

        q = to_heads(q_flat, s.num_k_heads, s.head_k_dim, s.v_per_k)
        k = to_heads(k_flat, s.num_k_heads, s.head_k_dim, s.v_per_k)
        v = to_heads(v_flat, nv, s.head_v_dim, 1)
        _free(q_flat, q)
        _free(k_flat, k)
        _free(v_flat, v)

        q = self._l2_rms(q, "q_l2_scale", mem)
        k = self._l2_rms(k, "k_l2_scale", mem)

        beta_h = ttnn.reshape(beta, (1, batch * nv, 1, 1))
        g_h = ttnn.reshape(g, (1, batch * nv, 1, 1))
        decay = ttnn.exp(g_h, memory_config=mem)
        ttnn.deallocate(g_h)

        state = ttnn.multiply(self.recurrent_state, decay, memory_config=mem)
        ttnn.deallocate(decay)
        kv_mem = self._mm(k, state, mem)
        delta = ttnn.multiply(ttnn.subtract(v, kv_mem, memory_config=mem), beta_h, memory_config=mem)
        ttnn.deallocate(kv_mem)
        ttnn.deallocate(v)
        ttnn.deallocate(beta_h)
        update = ttnn.matmul(
            k,
            delta,
            transpose_a=True,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_cfg,
            memory_config=mem,
        )
        ttnn.deallocate(delta)
        ttnn.deallocate(k)
        new_state = ttnn.add(state, update, memory_config=mem)
        ttnn.deallocate(state)
        ttnn.deallocate(update)
        out = self._mm(q, new_state, mem)
        ttnn.deallocate(q)
        ttnn.copy(new_state, self.recurrent_state)
        ttnn.deallocate(new_state)

        core = ttnn.reshape(out, (1, batch, nv, s.head_v_dim))
        _free(out, core)
        z_heads = ttnn.reshape(z, (1, batch, nv, s.head_v_dim))
        normed = self._rms_norm(ttnn.typecast(core, ttnn.bfloat16), self.w["gated_norm"])
        ttnn.deallocate(core)
        gated = ttnn.multiply(normed, z_heads)  # F10: z already carries its SiLU
        ttnn.deallocate(normed)
        ttnn.deallocate(z_heads)
        flat = ttnn.reshape(gated, (1, 1, batch, s.value_dim))
        _free(gated, flat)
        result = ttnn.linear(
            flat, self.w["out_proj"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg
        )
        ttnn.deallocate(flat)
        return result

    # ------------------------------------------------------------- forwards

    def prefill_forward(
        self,
        hidden_states,
        *,
        user_id: int = 0,
        page_table=None,
        page_tables_per_chunk: Optional[Sequence] = None,
        rot_mats=None,
    ):
        s = self.shapes
        assert len(hidden_states.shape) == 4 and int(hidden_states.shape[0]) == 1 and int(hidden_states.shape[1]) == 1, (
            f"prefill expects [1, 1, seq_len, hidden]; got {hidden_states.shape}"
        )
        seq_len = int(hidden_states.shape[2])
        assert 1 <= seq_len <= self.max_seq_len, f"seq_len {seq_len} outside [1, {self.max_seq_len}]"
        assert user_id < self.max_batch, f"user_id {user_id} >= max_batch {self.max_batch}"

        if s.layer_type == FULL_ATTENTION:
            assert page_table is not None and page_tables_per_chunk is not None, (
                "full_attention prefill requires a paged KV cache: pass page_table and page_tables_per_chunk"
            )
            assert rot_mats is not None, "full_attention prefill requires rot_mats=(cos, sin)"
            assert int(rot_mats[0].shape[-1]) == s.head_dim, (
                f"the fused decoder's rot_mats are head_dim ({s.head_dim}) wide in the permuted "
                f"channel order, not rotary_dim wide; got {int(rot_mats[0].shape[-1])}"
            )
        else:
            self._reset_linear_state(user_id)

        conv_prefix = None
        state = None
        if s.layer_type == LINEAR_ATTENTION:
            conv_prefix = self.const["conv_zero_prefix"]
            state = self.user_recurrent_state[user_id]
            self.user_recurrent_state[user_id] = None

        pieces = []
        alignment = _prefill_alignment(s.layer_type, self.block_size)
        for chunk_idx, chunk_start in enumerate(range(0, seq_len, PREFILL_CHUNK)):
            logical = min(PREFILL_CHUNK, seq_len - chunk_start)
            padded = _round_up(logical, alignment)
            x_chunk = ttnn.slice(
                hidden_states, [0, 0, chunk_start, 0], [1, 1, chunk_start + logical, s.hidden_size]
            )
            if padded != logical:
                x_chunk = _pad_seq(x_chunk, padded, hidden_states)

            residual = x_chunk
            normed = self._rms_norm(x_chunk, self.w["input_layernorm"])
            if s.layer_type == FULL_ATTENTION:
                cos, sin = rot_mats
                cos_chunk = _slice_pad_seq(cos, chunk_start, logical, padded)
                sin_chunk = _slice_pad_seq(sin, chunk_start, logical, padded)
                mixed = self._full_attention_prefill(
                    normed,
                    user_id=user_id,
                    page_table=page_table,
                    chunk_page_table=page_tables_per_chunk[chunk_idx],
                    rot_mats=(cos_chunk, sin_chunk),
                    chunk_start=chunk_start,
                )
                _free(cos_chunk, cos)
                _free(sin_chunk, sin)
            else:
                mixed, state, new_conv_state = self._linear_attention_prefill_chunk(
                    normed, state=state, conv_prefix=conv_prefix, length=logical
                )
                if conv_prefix is not self.const["conv_zero_prefix"]:
                    ttnn.deallocate(conv_prefix)
                conv_prefix = ttnn.slice(
                    new_conv_state, [0, 0, 1, 0], [1, 1, s.conv_kernel_size, s.conv_dim]
                )
                ttnn.deallocate(self.user_conv_state[user_id])
                self.user_conv_state[user_id] = new_conv_state
            ttnn.deallocate(normed)

            hidden = ttnn.add(residual, mixed)
            _free(residual, hidden_states)
            ttnn.deallocate(mixed)
            normed2 = self._rms_norm(hidden, self.w["post_attention_layernorm"])
            mlp_out = self._mlp(normed2)
            ttnn.deallocate(normed2)
            out_chunk = ttnn.add(hidden, mlp_out)
            ttnn.deallocate(hidden)
            ttnn.deallocate(mlp_out)
            if padded != logical:
                trimmed = ttnn.slice(out_chunk, [0, 0, 0, 0], [1, 1, logical, s.hidden_size])
                _free(out_chunk, trimmed)
                out_chunk = trimmed
            pieces.append(out_chunk)

        if s.layer_type == LINEAR_ATTENTION:
            self.user_recurrent_state[user_id] = state
            if conv_prefix is not self.const["conv_zero_prefix"]:
                ttnn.deallocate(conv_prefix)

        if len(pieces) == 1:
            return pieces[0]
        out = ttnn.concat(pieces, dim=2)
        for piece in pieces:
            ttnn.deallocate(piece)
        return out

    def decode_forward(self, hidden_states, *, current_pos=None, page_table=None, rot_mats=None):
        s = self.shapes
        assert len(hidden_states.shape) == 4, f"decode expects [1, 1, batch, hidden]; got {hidden_states.shape}"
        assert int(hidden_states.shape[2]) == self.max_batch, (
            f"decode batch {int(hidden_states.shape[2])} != max_batch {self.max_batch}"
        )
        residual = hidden_states
        normed = self._decode_rms_norm(hidden_states, self.w["input_layernorm"])
        if s.layer_type == FULL_ATTENTION:
            assert current_pos is not None and page_table is not None and rot_mats is not None
            assert int(rot_mats[0].shape[-1]) == s.head_dim, (
                f"the fused decoder's rot_mats are head_dim ({s.head_dim}) wide in the permuted "
                f"channel order, not rotary_dim wide; got {int(rot_mats[0].shape[-1])}"
            )
            mixed = self._full_attention_decode(
                normed, current_pos=current_pos, page_table=page_table, rot_mats=rot_mats
            )
        else:
            mixed = self._linear_attention_decode(normed)
        ttnn.deallocate(normed)
        hidden = ttnn.add(residual, mixed)
        ttnn.deallocate(mixed)
        normed2 = self._decode_rms_norm(hidden, self.w["post_attention_layernorm"])
        mlp_out = self._mlp(normed2)
        ttnn.deallocate(normed2)
        out = ttnn.add(hidden, mlp_out)
        ttnn.deallocate(hidden)
        ttnn.deallocate(mlp_out)
        return out

    # ------------------------------------------------------------- helpers

    def prefill_chunk_plan(self, seq_len: int) -> list[tuple[int, int, int]]:
        """``[(chunk_start, logical_len, padded_len)]`` for a prefill of ``seq_len`` tokens."""
        plan = []
        alignment = _prefill_alignment(self.shapes.layer_type, self.block_size)
        for chunk_start in range(0, seq_len, PREFILL_CHUNK):
            logical = min(PREFILL_CHUNK, seq_len - chunk_start)
            plan.append((chunk_start, logical, _round_up(logical, alignment)))
        return plan

    def _reset_linear_state(self, user_id: int) -> None:
        s = self.shapes
        old_state = self.user_recurrent_state[user_id]
        if old_state is not None:
            ttnn.deallocate(old_state)
        self.user_recurrent_state[user_id] = ttnn.zeros(
            (1, s.num_v_heads, s.head_k_dim, s.head_v_dim),
            dtype=self.recurrent_state.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        ttnn.deallocate(self.user_conv_state[user_id])
        self.user_conv_state[user_id] = ttnn.zeros(
            (1, 1, s.conv_kernel_size, s.conv_dim),
            dtype=self.conv_state.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )

    def prepare_decode_state(self) -> None:
        """Fold the per-user prefill state into the batch-wide decode buffers.

        Same semantics and same limitation as the functional decoder: it rewrites *every* slot
        from that user's post-prefill snapshot, so continuous batching needs a per-slot fold.
        """
        if self.shapes.layer_type != LINEAR_ATTENTION:
            return
        assert all(t is not None for t in self.user_recurrent_state), "prefill every user first"
        merged_recurrent = ttnn.concat(self.user_recurrent_state, dim=1)
        ttnn.copy(merged_recurrent, self.recurrent_state)
        _free(merged_recurrent, *self.user_recurrent_state)
        merged_conv = ttnn.concat(self.user_conv_state, dim=1)
        ttnn.copy(merged_conv, self.conv_state)
        _free(merged_conv, *self.user_conv_state)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def _pad_seq(tensor, padded: int, *keep):
    """Zero-pad ``tensor`` along the sequence axis, freeing it unless it aliases ``keep``."""
    length = int(tensor.shape[-2])
    if length == padded:
        return tensor
    out = ttnn.pad(tensor, [(0, 0), (0, 0), (0, padded - length), (0, 0)], 0.0)
    _free(tensor, out, *keep)
    return out


def _slice_pad_seq(tensor, start: int, logical: int, padded: int):
    """``tensor[start : start + logical]`` zero-padded to ``padded``; ``tensor`` stays alive."""
    piece = ttnn.slice(tensor, [0, 0, start, 0], [1, 1, start + logical, int(tensor.shape[-1])])
    return _pad_seq(piece, padded, tensor)


def _zero_after_seq(tensor, logical: int, padded: int):
    """Zero rows ``[logical, padded)`` of a ``[1, 1, padded, W]`` sequence tensor."""
    if logical == padded:
        return tensor
    kept = ttnn.slice(tensor, [0, 0, 0, 0], [1, 1, logical, int(tensor.shape[-1])])
    out = _pad_seq(kept, padded, tensor)
    _free(tensor, out, kept)
    return out

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN functional decoder for Qwen/Qwen3.6-27B (HF ``model_type: qwen3_5``).

The HF text stack has **two** decoder layer kinds, selected per layer by
``config.layer_types[layer_idx]``:

``full_attention``
    ``Qwen3_5Attention`` — GQA (24 query heads / 4 KV heads, ``head_dim`` 256) with an
    **output gate** (``q_proj`` emits ``2 * num_heads * head_dim`` and the second half of
    every head becomes a sigmoid gate on the attention output), per-head Q/K RMS norms and
    **partial** rotary embeddings (``partial_rotary_factor`` 0.25 ⇒ only the first 64 of
    256 head channels are rotated).

``linear_attention``
    ``Qwen3_5GatedDeltaNet`` — a gated delta-rule linear-attention mixer: four input
    projections (qkv / z / b / a), a depthwise causal conv1d of width 4 over the fused
    qkv channels, L2-normalised Q/K, a per-head decay ``g = -exp(A_log)*softplus(a+dt_bias)``
    and gate ``beta = sigmoid(b)``, the gated delta rule itself, and a z-gated RMS norm.

Both kinds share the residual structure, the two RMS norms and the SwiGLU MLP.  Note the
Qwen3.5 RMS norm is **1-centred**: ``y = normed * (1 + weight)``.

Prefill / decode contract
-------------------------

``prefill_forward(hidden_states, user_id=..., page_table=..., page_tables_per_chunk=...,
rot_mats=...)``
    * ``hidden_states``: ``[1, 1, seq_len, hidden_size]`` — one user's **entire** prompt.
      Any ``1 <= seq_len <= max_seq_len`` is accepted (``max_seq_len`` defaults to the HF
      ``max_position_embeddings``); the layer owns all internal padding, chunking and tiling
      and slices the result back to ``seq_len``.  No divisibility requirement is imposed on
      the caller.
    * Prefill is single-user by construction: TTNN's chunked paged SDPA requires the page
      table batch to equal the query batch, and ``paged_fill_cache`` addresses one user per
      call.  Batched prefill = loop over ``user_id``.
    * Sequences longer than :data:`PREFILL_CHUNK` are processed in chunks that carry the
      full recurrent/conv state (``linear_attention``) or fill the paged KV cache
      incrementally and attend with ``chunked_scaled_dot_product_attention``
      (``full_attention``).
    * Calling ``prefill_forward`` resets the layer state of ``user_id`` — it always starts
      the user at absolute position 0.  For ``full_attention`` that means the user's cache
      blocks are rewritten over ``[0, seq_len)``; blocks beyond the new prompt keep whatever
      an earlier prefill left there and are never read, because decode only attends up to
      ``current_pos``.
    * ``page_table`` (``full_attention`` only): int32 ``[1, blocks_per_user]``, this user's
      whole virtual→physical block map, used by the SDPA read.
    * ``page_tables_per_chunk`` (``full_attention`` only): one int32 ``[1, n]`` slice of
      ``page_table`` per prefill chunk, covering exactly that chunk's padded block span.
      Use :meth:`FunctionalDecoder.prefill_chunk_plan` to compute the slices host-side.
    * ``rot_mats`` (``full_attention`` only): ``(cos, sin)``, each
      ``[1, 1, seq_len, rotary_dim]``, ``bfloat16``, TILE.
    * Returns ``[1, 1, seq_len, hidden_size]``.
    * ``linear_attention`` prefill leaves the per-user conv/recurrent state in
      ``user_conv_state[user_id]`` / ``user_recurrent_state[user_id]``; call
      :meth:`prepare_decode_state` once, after every user is prefilled, to fold them into the
      batch-wide buffers the traced decode updates in place.

``decode_forward(hidden_states, current_pos=..., page_table=..., rot_mats=...)``
    * ``hidden_states``: ``[1, 1, batch, hidden_size]`` (decode layout: batch on the
      second-to-last axis), ``batch == max_batch``.
    * ``current_pos``: ``ttnn`` int32 tensor of shape ``[batch]`` holding each user's
      absolute position.  A device tensor (not a python list) so decode is traceable.
      ``full_attention`` uses it for both the paged cache update and the SDPA read window;
      ``linear_attention`` needs no position at all — the gated-delta-net recurrence carries
      the whole history in ``conv_state``/``recurrent_state``, so ``current_pos``,
      ``page_table`` and ``rot_mats`` are all ignored for that layer kind.
    * ``rot_mats`` (``full_attention`` only): ``(cos, sin)``, each
      ``[1, batch, 1, rotary_dim]``, ``bfloat16``, TILE — one row per user, broadcast over
      the head axis on device so no host work happens inside the traced region.
    * Returns ``[1, 1, batch, hidden_size]``.

Everything that touches ``torch`` (weight transposition, dtype selection, cache
construction) happens in :meth:`FunctionalDecoder.from_state_dict`; a single prefill or
decode pass runs entirely on device.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import ttnn
from models.common.lightweightmodule import LightweightModule

from .model_config import DELTA_CHUNK, FULL_ATTENTION, LINEAR_ATTENTION, DecoderShapes, decoder_shapes

#: Tokens processed per prefill chunk.  Must be a multiple of the KV block size, of the
#: SDPA q/k chunk size, and of :data:`~.model_config.DELTA_CHUNK`.
PREFILL_CHUNK = 2048

#: Padding granularity of a ``full_attention`` prefill chunk, in tokens, and the smallest SDPA k
#: chunk this layer asks for.  A padded chunk is *not* always a whole number of k chunks - once
#: :func:`_sdpa_program_config` grows the k chunk to 512 past 131072 keys, a final chunk padded
#: to 256, 768, 1280 or 1792 is not - and it does not need to be: the op rounds its own key
#: extent up to the k chunk internally (``sdpa_program_factory.cpp``: ``padded_Sk = ceil(Sk /
#: k_chunk) * k_chunk``) and only requires ``k_chunk % TILE_WIDTH == 0``.
SDPA_CHUNK = 256

#: Cap on the number of k chunks one ``chunked_scaled_dot_product_attention`` call may merge.
#:
#: The kernel merges its flash-attention state - running max, running softmax denominator and
#: running output accumulator - once per k chunk, and loses a little of the denominator every
#: time.  The denominator is a sum of positive terms, so the loss is one-sided: the normalised
#: output comes out uniformly **too large**, by a factor that grows with the *number of merges*
#: rather than with the context length.  PCC is scale-invariant and cannot see it, but the
#: scale survives ``o_proj`` and the residual and does wreck the layer output.
#:
#: Measured on this checkout with synthetic Q/K/V
#: (``doc/functional_decoder/probes/probe_sdpa_synthetic.py``,
#: ``doc/functional_decoder/logs/sdpa_long_sweep_v2.log``), as ``alpha``, the ratio between the
#: device output and a float32 torch attention on bit-identical inputs:
#:
#:     k chunks | 262144 keys | 131072 keys | 8192 keys
#:     16       | -           | -           | 1.0009
#:     32       | -           | -           | 1.0037
#:     64       | -           | -           | 1.0091
#:     256      | -           | 1.045       | -
#:     512      | 1.091       | 1.090       | -
#:     1024     | 1.204       | -           | -
#:
#: 8192 keys in 64 chunks and 262144 keys in 512 chunks sit on the same curve, so the chunk
#: count is the variable.  Capping it at 512 is the most this op allows: 512 is also the
#: largest k chunk that fits L1 at ``head_dim`` 256 (see :data:`_SDPA_Q_FOR_K`), so a
#: 262144-key call cannot merge fewer than 512 chunks.
#:
#: This is a tt-metal SDPA defect, not a model-code one; ``probe_sdpa_synthetic.py`` is the
#: model-free reproducer.  Note the synthetic probe is the worst case for it - random Q/K/V
#: give a maximally flat softmax, so every chunk contributes equally.  Real attention is
#: peakier and the layer-level effect is far smaller; the layer numbers are in the README.
SDPA_MAX_K_CHUNKS = 512

#: q chunk size to use for each k chunk size.  The q chunk does not change the *result* - the
#: k-chunk merge is what the flash statistics see - it is bounded only by L1.
#:
#: Measured on this checkout (``doc/functional_decoder/logs/sdpa_fit_sweep_v2.log``), at
#: ``head_dim`` 256 with ``fp32_dest_acc_en``, against Blackhole's 1572864 B L1::
#:
#:     q 256 / k 256 -> 1676672 B  (rejected)      q 128 / k 256 -> fits
#:     q 512 / k 128 -> 2184576 B  (rejected)      q 128 / k 128 -> fits
#:     q 128 / k 512 -> 1815936 B  (rejected)      q  64 / k 512 -> fits
#:     q  64 / k 1024 -> 2672000 B (rejected)      -> 512 is the largest usable k chunk
#:
#: The q 256 entries this table used to hold were sized against an older tt-metal whose SDPA
#: kept every intermediate in bfloat16; current main promotes the QK and row-sum circular
#: buffers to fp32 under ``fp32_dest_acc_en`` (``sdpa_program_factory.cpp``
#: ``fp32_dest_intermediate_dataformat``), which is what pushed q 256 / k 256 over L1.
_SDPA_Q_FOR_K = {256: 128, 512: 64}

#: k chunk and core count for ``paged_scaled_dot_product_attention_decode``.
#:
#: The decode kernel has two independent accuracy problems at long positions, both measured on
#: this checkout with ``doc/functional_decoder/probes/probe_sdpa_decode_synthetic.py`` (a
#: model-free reproducer) and both invisible to PCC on the attention output, because they are
#: pure *scale* errors that only the residual and ``o_proj`` expose:
#:
#: 1. **The flash statistics were bfloat16.**  With the stock kernel the device/float32-golden
#:    scale at position 262143 is **37.7** with the default program config, and 5.2 with one
#:    core per head.  ``ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/
#:    sdpa_decode_program_factory.cpp`` now promotes the core-local running max, softmax
#:    denominator and output accumulator to fp32 under ``fp32_dest_acc_en`` when there is one
#:    core per head; see ``doc/functional_decoder/work_log.md``.
#: 2. **The cross-core tree reduction is wrong for most positions.**  With more than one core
#:    per head the result is correct only when ``num_k_chunks`` is 1 or a multiple of
#:    ``2 * cores_per_head``, and ``num_k_chunks = ceil((cur_pos + 1) / k_chunk)`` is a
#:    *runtime* quantity while the program config is compile-time.  Violations are not small:
#:    position 1023 returns a 3705x scale and position 261887 returns NaN
#:    (``doc/functional_decoder/logs/sdpa_decode_cfg_sweep_v2.log``).  No core count above 1
#:    is correct at every position, so this layer pins ``max_cores_per_head_batch = 1``, which
#:    takes the reduction out of the picture entirely.  That is a real decode-latency cost -
#:    4 active cores instead of 64 - and it is a correctness-first choice, not a tuned one.
#:
#: With both in place the scale is 0.995-1.017 over positions 1023, 4095, 12287, 16383, 65535,
#: 131071, 261887 and 262143 (``doc/functional_decoder/logs/sdpa_decode_fp32acc_v2.log``).
#: 512 is the largest k chunk that fits L1 here and gives the fewest sequential merges.
SDPA_DECODE_K_CHUNK = 512
SDPA_DECODE_CORES_PER_HEAD = 1

#: Decode head padding: TTNN decode attention ops operate on tile-padded head counts.
PADDED_HEADS = 32

#: Default paged-KV block size (tokens per page).
DEFAULT_BLOCK_SIZE = 64

#: Block size at which :meth:`FunctionalDecoder._unit_tri_inverse` stops recursing and falls
#: back to the Neumann doubling product.  Chosen by measurement on **real checkpoint weights**
#: (``doc/functional_decoder/probes/probe_tri_inv_base.py``,
#: ``doc/functional_decoder/logs/tri_inv_base_sweep.log``), because the recursion's blocks are
#: stored in 32x32 tiles and run single-core, so a smaller base costs real time:
#:
#:     base | prefill PCC | recurrent state PCC | decode PCC | warmed 2048-token prefill
#:     8    | 0.999970    | 0.999993            | 0.999985   | 204.7 ms
#:     16   | 0.999969    | 0.999992            | 0.999986   | 161.1 ms
#:     32   | 0.999970    | 0.999988            | 0.999986   | 136.4 ms
#:
#: All three clear the bar with room to spare, and the measurement does **not** by itself select
#: 16 - base 32 is 15 % faster again and still clears every bar.  16 is the *correctness-stage*
#: default: it is 21 % faster than 8 while its recurrent-state PCC stays within 1e-6 of base 8,
#: and 32 is the first base whose recurrent state moves by more than that.  Taking the extra
#: 15 % is a precision-for-speed trade, so it is handed to the optimization stage with the
#: numbers attached rather than taken here.
TRI_INV_BASE = 16


def _prefill_alignment(layer_type: str, block_size: int) -> int:
    """Padding granularity of a prefill chunk.

    ``full_attention`` needs the padded length to be a multiple of the SDPA q/k chunk *and* of
    the KV block size, so a chunk always covers a whole number of pages; ``linear_attention``
    needs a multiple of the gated-delta-rule chunk.
    """
    if layer_type == FULL_ATTENTION:
        return _lcm(SDPA_CHUNK, block_size)
    return DELTA_CHUNK


def _shape(tensor) -> list:
    """``ttnn.Shape`` is not python-sliceable; normalise to a list of ints."""
    return [int(d) for d in tensor.shape]


def _free(tensor, *live) -> None:
    """``ttnn.deallocate`` that is a no-op when ``tensor`` shares a buffer with ``live``.

    Several TTNN ops return a **view** instead of a copy whenever no data has to move:
    ``ttnn.reshape`` and ``ttnn.typecast`` on an unchanged layout/dtype, ``ttnn.slice`` over
    the full range, ``ttnn.concat`` of a single tensor, ``ttnn.to_memory_config`` into the
    config the tensor already has, and ``ttnn.pad`` when the padding fits inside the existing
    tile padding.  Deallocating the input of such an op frees the result along with it, and
    the next allocation silently overwrites live data.  Every deallocation whose target can
    alias a still-needed tensor goes through here.
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
    """SDPA config for a prefill chunk starting at ``chunk_start_idx`` over ``kv_len`` keys.

    The k chunk is grown from :data:`SDPA_CHUNK` until the call merges at most
    :data:`SDPA_MAX_K_CHUNKS` of them, which is what keeps the kernel's bfloat16 running
    softmax denominator honest; see :data:`SDPA_MAX_K_CHUNKS`.

    ``chunked_scaled_dot_product_attention`` requires ``chunk_start_idx`` to be a multiple of
    both chunk sizes.  Prefill chunks start at multiples of :data:`PREFILL_CHUNK`, itself a
    multiple of every k chunk size we can select, so that always holds.
    """
    k_chunk = SDPA_CHUNK
    while kv_len > k_chunk * SDPA_MAX_K_CHUNKS and 2 * k_chunk in _SDPA_Q_FOR_K:
        k_chunk *= 2
    q_chunk = _SDPA_Q_FOR_K[k_chunk]
    assert chunk_start_idx % k_chunk == 0 and chunk_start_idx % q_chunk == 0, (
        f"chunk_start_idx {chunk_start_idx} must be a multiple of the SDPA q chunk {q_chunk} " f"and k chunk {k_chunk}"
    )
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


def _sdpa_decode_program_config() -> ttnn.SDPAProgramConfig:
    """Program config for ``paged_scaled_dot_product_attention_decode``.

    Position-independent by construction - see :data:`SDPA_DECODE_K_CHUNK` for why the core
    count is pinned to one per head and why the default (auto) config cannot be used.  The q
    chunk is irrelevant for decode (one query row) and is set to a single tile.
    """
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=32,
        k_chunk_size=SDPA_DECODE_K_CHUNK,
        exp_approx_mode=False,
        max_cores_per_head_batch=SDPA_DECODE_CORES_PER_HEAD,
    )


class FunctionalDecoder(LightweightModule):
    """One Qwen3.5/3.6 decoder layer on a TTNN mesh device."""

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
        # Per-user prefill state.  Prefill rebinds these (cheap, host-side); a single
        # ``prepare_decode_state()`` folds them into the batch-wide buffers above, which the
        # traced decode then updates in place with ``ttnn.copy``.
        self.user_conv_state: list = list(user_conv_state or [])
        self.user_recurrent_state: list = list(user_recurrent_state or [])
        self.const = constants or {}
        self.compute_cfg = _hifi4(fp32_dest_acc_en=True)
        # SDPA gets fp32 destination accumulation too.  Clearing it also switches the op to the
        # streaming compute kernel (`can_use_streaming_compute(fp32_dest_acc_en)` is
        # `!fp32_dest_acc_en`), whose k-chunk merge is measurably worse: at 262144 keys in 512
        # chunks the device/golden scale is 1.176 without fp32 destination accumulation against
        # 1.091 with it, and 1.692 vs 1.204 at 1024 chunks
        # (doc/functional_decoder/logs/sdpa_long_sweep_v2.log).
        self.sdpa_compute_cfg = _hifi4(fp32_dest_acc_en=True)
        self.sdpa_decode_program_cfg = _sdpa_decode_program_config()
        self.decode_head_mem_cfg = None
        if shapes.layer_type == FULL_ATTENTION:
            # Decode K/V (for paged_update_cache) and the SDPA output (for
            # nlp_concat_heads_decode) are height-sharded one user per core.
            self.decode_head_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(PADDED_HEADS, shapes.head_dim),
                core_grid=ttnn.num_cores_to_corerangeset(max_batch, ttnn.CoreCoord(8, 8), row_wise=True),
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
    ) -> "FunctionalDecoder":
        """Build the layer from an HF **submodule-relative** state dict.

        Keys are exactly those of ``Qwen3_5DecoderLayer`` (``input_layernorm.weight``,
        ``self_attn.q_proj.weight`` / ``linear_attn.in_proj_qkv.weight``, ...).  This is the
        only place ``torch`` is used.
        """
        import torch  # setup-time only; never on the prefill/decode path

        shapes = decoder_shapes(hf_config, layer_idx)
        max_seq_len = max_seq_len or shapes.max_position_embeddings
        # PREFILL_CHUNK must be a whole number of pages *and* a whole number of padded chunks,
        # or chunk N's page-table slice starts inside chunk N-1's tokens and paged_fill_cache
        # silently overwrites the tail of the previous chunk. block_size = 96, for example, is
        # accepted by paged_update_cache (it only needs block_size % TILE_HEIGHT == 0) and gives
        # _prefill_alignment = lcm(256, 96) = 768, so chunk 1 would start writing at token 2016
        # instead of 2048 while every existing length assertion still passed. Fail here instead.
        alignment = _prefill_alignment(shapes.layer_type, block_size)
        if PREFILL_CHUNK % block_size or PREFILL_CHUNK % alignment:
            raise ValueError(
                f"block_size {block_size} is incompatible with PREFILL_CHUNK {PREFILL_CHUNK}: "
                f"the prefill chunk must be a whole number of pages and of padded chunks "
                f"(alignment {alignment})"
            )
        if max_num_blocks is None:
            # The final prefill chunk is padded up to the layer's alignment and the caller's
            # page table has to cover that padded span, so size the cache from the padded
            # context rather than from max_seq_len itself.
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

        def _norm_w(tensor: "torch.Tensor", one_centred: bool):
            """Qwen3.5 RMSNorm multiplies by ``1 + weight``; fold the +1 in at load time."""
            value = tensor + 1.0 if one_centred else tensor
            return _tt(value.reshape(1, 1, 1, -1), ttnn.bfloat16)

        weights = {
            "input_layernorm": _norm_w(_get("input_layernorm.weight"), one_centred=True),
            "post_attention_layernorm": _norm_w(_get("post_attention_layernorm.weight"), one_centred=True),
        }

        # SwiGLU MLP with gate/up fused into a single matmul.
        gate_w = _get("mlp.gate_proj.weight")
        up_w = _get("mlp.up_proj.weight")
        weights["mlp_gate_up"] = _linear_w(torch.cat([gate_w, up_w], dim=0))
        weights["mlp_down"] = _linear_w(_get("mlp.down_proj.weight"))

        constants: dict = {}

        if shapes.layer_type == FULL_ATTENTION:
            n_heads, head_dim = shapes.num_attention_heads, shapes.head_dim
            # q_proj emits [num_heads, 2 * head_dim]; the per-head second half is the gate.
            q_full = _get("self_attn.q_proj.weight").reshape(n_heads, 2 * head_dim, -1)
            q_only = q_full[:, :head_dim, :].reshape(n_heads * head_dim, -1)
            gate_only = q_full[:, head_dim:, :].reshape(n_heads * head_dim, -1)
            k_w = _get("self_attn.k_proj.weight")
            v_w = _get("self_attn.v_proj.weight")
            weights["wqkv"] = _linear_w(torch.cat([q_only, k_w, v_w], dim=0))
            weights["wgate"] = _linear_w(gate_only)
            weights["o_proj"] = _linear_w(_get("self_attn.o_proj.weight"))
            weights["q_norm"] = _norm_w(_get("self_attn.q_norm.weight"), one_centred=True)
            weights["k_norm"] = _norm_w(_get("self_attn.k_norm.weight"), one_centred=True)
        else:
            weights["in_proj_qkv"] = _linear_w(_get("linear_attn.in_proj_qkv.weight"))
            weights["in_proj_z"] = _linear_w(_get("linear_attn.in_proj_z.weight"))
            weights["in_proj_b"] = _linear_w(_get("linear_attn.in_proj_b.weight"), ttnn.float32)
            weights["in_proj_a"] = _linear_w(_get("linear_attn.in_proj_a.weight"), ttnn.float32)
            weights["out_proj"] = _linear_w(_get("linear_attn.out_proj.weight"))
            weights["gated_norm"] = _norm_w(_get("linear_attn.norm.weight"), one_centred=False)

            # conv1d weight [conv_dim, 1, K] → K taps of shape [1, 1, 1, conv_dim]
            conv_w = _get("linear_attn.conv1d.weight").squeeze(1)
            weights["conv_taps"] = [
                _tt(conv_w[:, j].reshape(1, 1, 1, -1), ttnn.float32) for j in range(shapes.conv_kernel_size)
            ]

            # g = -exp(A_log) * softplus(a + dt_bias); both factors are per v-head.
            weights["neg_exp_A"] = _tt((-torch.exp(_get("linear_attn.A_log"))).reshape(1, 1, 1, -1), ttnn.float32)
            weights["dt_bias"] = _tt(_get("linear_attn.dt_bias").reshape(1, 1, 1, -1), ttnn.float32)

            constants["eye_base"] = _tt(torch.eye(TRI_INV_BASE).reshape(1, 1, TRI_INV_BASE, TRI_INV_BASE), ttnn.float32)
            constants["conv_zero_prefix"] = _tt(
                torch.zeros(1, 1, shapes.conv_kernel_size - 1, shapes.conv_dim), ttnn.float32
            )

        kv_cache = None
        conv_state = None
        recurrent_state = None
        user_conv_state = None
        user_recurrent_state = None
        if shapes.layer_type == FULL_ATTENTION:
            cache_shape = (max_num_blocks, shapes.num_key_value_heads, block_size, shapes.head_dim)
            kv_cache = tuple(_tt(torch.zeros(cache_shape), cache_dtype) for _ in range(2))
        else:
            conv_state = _tt(torch.zeros(1, max_batch, shapes.conv_kernel_size, shapes.conv_dim), state_dtype)
            recurrent_state = _tt(
                torch.zeros(1, max_batch * shapes.num_v_heads, shapes.head_k_dim, shapes.head_v_dim),
                state_dtype,
            )
            user_conv_state = [
                _tt(torch.zeros(1, 1, shapes.conv_kernel_size, shapes.conv_dim), state_dtype) for _ in range(max_batch)
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

    def _rms_norm(self, x, weight):
        return ttnn.rms_norm(
            x,
            epsilon=self.shapes.rms_norm_eps,
            weight=weight,
            compute_kernel_config=self.compute_cfg,
        )

    def _mlp(self, x):
        gate_up = ttnn.linear(x, self.w["mlp_gate_up"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        inter = self.shapes.intermediate_size
        lead = _shape(gate_up)[:3]
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [*lead, inter])
        up = ttnn.slice(gate_up, [0, 0, 0, inter], [*lead, 2 * inter])
        _free(gate_up, gate, up)
        activated = ttnn.silu(gate)
        ttnn.deallocate(gate)
        prod = ttnn.multiply(activated, up)
        ttnn.deallocate(activated)
        ttnn.deallocate(up)
        out = ttnn.linear(prod, self.w["mlp_down"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        ttnn.deallocate(prod)
        return out

    def _apply_rope(self, x, cos, sin):
        """Partial rotary embedding over the leading ``rotary_dim`` channels of each head."""
        rd = self.shapes.rotary_dim
        head_dim = self.shapes.head_dim
        lead = _shape(x)[:-1]
        starts = [0] * len(lead)
        rot = ttnn.slice(x, [*starts, 0], [*lead, rd])
        first = ttnn.slice(rot, [*starts, 0], [*lead, rd // 2])
        second = ttnn.slice(rot, [*starts, rd // 2], [*lead, rd])
        neg_second = ttnn.neg(second)
        ttnn.deallocate(second)
        rotated = ttnn.concat([neg_second, first], dim=-1)
        ttnn.deallocate(neg_second)
        ttnn.deallocate(first)
        embedded = ttnn.add(ttnn.multiply(rot, cos), ttnn.multiply(rotated, sin))
        ttnn.deallocate(rot)
        ttnn.deallocate(rotated)
        if rd == head_dim:
            return embedded
        passthrough = ttnn.slice(x, [*starts, rd], [*lead, head_dim])
        out = ttnn.concat([embedded, passthrough], dim=-1)
        ttnn.deallocate(embedded)
        ttnn.deallocate(passthrough)
        return out

    @staticmethod
    def _l2norm(x, eps: float = 1e-6):
        square_sum = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        inv = ttnn.rsqrt(ttnn.add(square_sum, eps))
        ttnn.deallocate(square_sum)
        out = ttnn.multiply(x, inv)
        ttnn.deallocate(inv)
        return out

    # ------------------------------------------------------- full attention

    def _attn_projections(self, x, *, decode: bool):
        """Return ``(q, k, v, gate)`` with heads split out.

        prefill: q ``[1, n_heads, S, D]``, k/v ``[1, n_kv, S, D]``, gate ``[1, 1, S, n_heads*D]``
        decode:  q ``[1, B, PADDED_HEADS, D]``, k/v ``[1, B, PADDED_HEADS, D]``,
                 gate ``[1, 1, B, n_heads*D]``
        """
        s = self.shapes
        qkv = ttnn.linear(x, self.w["wqkv"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        gate = ttnn.linear(x, self.w["wgate"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        if decode:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv,
                num_heads=s.num_attention_heads,
                num_kv_heads=s.num_key_value_heads,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        gated = ttnn.multiply(attn_out, ttnn.sigmoid(gate))
        ttnn.deallocate(attn_out)
        ttnn.deallocate(gate)
        out = ttnn.linear(gated, self.w["o_proj"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        ttnn.deallocate(gated)
        return out

    def _full_attention_prefill(self, x, *, user_id, page_table, chunk_page_table, rot_mats, chunk_start):
        """Attention over one prefill chunk; ``x`` is ``[1, 1, L_padded, hidden]``."""
        s = self.shapes
        cos, sin = rot_mats
        q, k, v, gate = self._attn_projections(x, decode=False)

        q = self._rms_norm(q, self.w["q_norm"])
        k = self._rms_norm(k, self.w["k_norm"])
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        k_cache, v_cache = self.kv_cache
        k_fill = ttnn.typecast(k, k_cache.dtype)
        v_fill = ttnn.typecast(v, v_cache.dtype)
        _free(k, k_fill)
        _free(v, v_fill)
        # The per-chunk page table must cover the whole padded chunk.  Trimming the K/V to a
        # short page table instead would silently drop cache writes for the tail of the
        # prompt, which only surfaces much later as a wrong decode; fail directly.
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
            # Causality bounds the keys this call actually merges at the end of the chunk.
            program_config=_sdpa_program_config(chunk_start, chunk_start + page_len),
            compute_kernel_config=self.sdpa_compute_cfg,
        )
        ttnn.deallocate(q)
        concat = ttnn.experimental.nlp_concat_heads(attn)
        ttnn.deallocate(attn)
        return self._attn_epilogue(concat, gate)

    def _full_attention_decode(self, x, *, current_pos, page_table, rot_mats):
        s = self.shapes
        cos, sin = rot_mats
        q, k, v, gate = self._attn_projections(x, decode=True)
        # nlp_create_qkv_heads_decode always emits height-sharded tensors; the norm and the
        # partial-RoPE slices below need interleaved inputs.
        q, k, v = (ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v))
        q = self._rms_norm(q, self.w["q_norm"])
        k = self._rms_norm(k, self.w["k_norm"])
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        k_cache, v_cache = self.kv_cache
        k = ttnn.to_memory_config(k, self.decode_head_mem_cfg)
        v = ttnn.to_memory_config(v, self.decode_head_mem_cfg)
        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            page_table,
            cur_pos_tensor=current_pos,
            scale=s.attn_scaling,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.sdpa_decode_program_cfg,
            compute_kernel_config=self.sdpa_compute_cfg,
        )
        ttnn.deallocate(q)
        attn_sharded = ttnn.to_memory_config(attn, self.decode_head_mem_cfg)
        _free(attn, attn_sharded)
        concat = ttnn.experimental.nlp_concat_heads_decode(attn_sharded, num_heads=s.num_attention_heads)
        ttnn.deallocate(attn_sharded)
        # nlp_concat_heads_decode pads the batch axis up to a tile; drop the padding so the
        # output gate (and the residual) keep the logical batch.
        if int(concat.shape[2]) != self.max_batch:
            trimmed = ttnn.slice(concat, [0, 0, 0, 0], [1, 1, self.max_batch, s.num_attention_heads * s.head_dim])
            _free(concat, trimmed)
            concat = trimmed
        concat = ttnn.to_memory_config(concat, ttnn.DRAM_MEMORY_CONFIG)
        return self._attn_epilogue(concat, gate)

    # ----------------------------------------------------- linear attention

    def _gdn_inputs(self, x):
        """Shared GatedDeltaNet input projections.

        ``x`` is ``[..., N, hidden]`` where ``N`` is the sequence (prefill) or batch
        (decode) axis.  Returns ``(mixed_qkv, z, beta, g)``: ``z`` in bfloat16, the rest in
        float32 (the recurrence and the softplus/exp gating are numerically sensitive).
        """
        mixed_qkv = ttnn.linear(x, self.w["in_proj_qkv"], dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)
        z = ttnn.linear(x, self.w["in_proj_z"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        b = ttnn.linear(x, self.w["in_proj_b"], dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)
        a = ttnn.linear(x, self.w["in_proj_a"], dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)
        beta = ttnn.sigmoid(b)
        ttnn.deallocate(b)
        biased = ttnn.add(a, self.w["dt_bias"])
        ttnn.deallocate(a)
        soft = ttnn.softplus(biased, beta=1.0, threshold=20.0)
        ttnn.deallocate(biased)
        g = ttnn.multiply(soft, self.w["neg_exp_A"])
        ttnn.deallocate(soft)
        return mixed_qkv, z, beta, g

    def _causal_conv(self, mixed_qkv, prefix, logical: int):
        """Depthwise causal conv1d (width ``K``) + SiLU, channels-last.

        ``mixed_qkv``: ``[1, 1, L, conv_dim]``; ``prefix``: ``[1, 1, K-1, conv_dim]`` left
        context.  Returns ``(activations [1, 1, L, conv_dim], new_state [1, 1, K, conv_dim])``.

        ``L`` may exceed the logical token count when the chunk was zero-padded up to the
        delta-rule chunk length.  The activations of the padded rows are discarded by the
        caller, but the conv state must be the last ``K`` *raw* ``mixed_qkv`` rows ending at
        the last logical token (HF: ``F.pad(mixed_qkv, (K - L, 0))`` over the real prompt),
        so it is sliced at the logical end rather than at the padded end.
        """
        s = self.shapes
        k = s.conv_kernel_size
        length = int(mixed_qkv.shape[-2])
        window = ttnn.concat([prefix, mixed_qkv], dim=-2)
        acc = None
        for j in range(k):
            tap = ttnn.slice(window, [0, 0, j, 0], [1, 1, j + length, s.conv_dim])
            term = ttnn.multiply(tap, self.w["conv_taps"][j])
            ttnn.deallocate(tap)
            if acc is None:
                acc = term
            else:
                acc = ttnn.add(acc, term)
                ttnn.deallocate(term)
        # window row ``K - 1 + t`` is token ``t``; the K rows ending at token ``logical - 1``
        # start at row ``logical - 1``.
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

    def _mm(self, a, b):
        return ttnn.matmul(a, b, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)

    def _unit_tri_inverse(self, a, size: int):
        """``(I - a)**-1`` for strictly-lower-triangular ``a`` of shape ``[n, heads, size, size]``.

        The obvious closed form for a nilpotent ``a`` is the Neumann doubling product
        ``prod_j (I + a**(2**j))``.  It is mathematically exact but numerically unusable on
        this model: with the real Qwen3.6 gated-delta-net weights the intermediates peak at
        ``|a**8| ~ 1e3`` and then cancel back down to ``|inv| = 1``, so TTNN matmul precision
        leaves an absolute error of ~3 on a result of magnitude 1 (measured on captured
        ``attn0``; the recurrent state then diverged to ~1e18 and prefill PCC fell to 0.937).

        Recursive 2x2 block inversion has no such cancellation, because every intermediate is
        itself a well-conditioned unit-triangular inverse::

            [[L11, 0], [L21, L22]]**-1 == [[X11, 0], [X22 @ a21 @ X11, X22]]

        Both diagonal blocks are inverted in a single batched call.  Measured max absolute
        error on the captured real-weight ``attn0``: 3.3e-2 at base 32, 8.8e-3 at
        :data:`TRI_INV_BASE` = 16 and 1.8e-3 at base 8, against 3.3 for the plain doubling
        product.  See :data:`TRI_INV_BASE` for why 16 is the default.
        """
        if size <= TRI_INV_BASE:
            eye = self.const["eye_base"]
            inv = ttnn.add(a, eye)
            power = a
            for _ in range(int(math.log2(size)) - 1):
                squared = self._mm(power, power)
                _free(power, a)
                power = squared
                factor = ttnn.add(power, eye)
                updated = self._mm(inv, factor)
                ttnn.deallocate(factor)
                ttnn.deallocate(inv)
                inv = updated
            _free(power, a)
            return inv

        half = size // 2
        lead, heads = _shape(a)[0], _shape(a)[1]
        a11 = ttnn.slice(a, [0, 0, 0, 0], [lead, heads, half, half])
        a22 = ttnn.slice(a, [0, 0, half, half], [lead, heads, size, size])
        a21 = ttnn.slice(a, [0, 0, half, 0], [lead, heads, size, half])
        diagonal = ttnn.concat([a11, a22], dim=0)
        _free(a11, diagonal)
        _free(a22, diagonal)

        inverted = self._unit_tri_inverse(diagonal, half)
        _free(diagonal, inverted)
        x11 = ttnn.slice(inverted, [0, 0, 0, 0], [lead, heads, half, half])
        x22 = ttnn.slice(inverted, [lead, 0, 0, 0], [2 * lead, heads, half, half])
        _free(inverted, x11, x22)

        scaled = self._mm(a21, x11)
        ttnn.deallocate(a21)
        x21 = self._mm(x22, scaled)
        ttnn.deallocate(scaled)

        zero = ttnn.zeros(
            (lead, heads, half, half), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
        )
        top = ttnn.concat([x11, zero], dim=-1)
        bottom = ttnn.concat([x21, x22], dim=-1)
        ttnn.deallocate(zero)
        _free(x11, top)
        _free(x21, bottom)
        _free(x22, bottom)
        out = ttnn.concat([top, bottom], dim=-2)
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

        # HF pads q/k/v/beta/g with **zeros** up to the delta-chunk multiple, so a padded
        # position is an exact no-op for the recurrence: ``beta = 0`` kills its contribution
        # and ``g = 0`` keeps the decay at 1.  Here the *hidden states* were padded instead,
        # which leaves ``beta = sigmoid(0) = 0.5`` and ``g = -exp(A_log) * softplus(dt_bias)
        # != 0``; without this mask the recurrent state is decayed and updated by up to
        # ``chunk - 1`` phantom tokens (observed: state underflowing to ~1e-29).
        beta = _zero_after_seq(beta, length, padded)
        g = _zero_after_seq(g, length, padded)

        def to_heads(flat, num_heads, head_dim, repeat: int):
            t = ttnn.reshape(flat, (1, padded, num_heads, head_dim))
            t = ttnn.permute(t, (0, 2, 1, 3))  # [1, H, L, D]
            if repeat > 1:
                rep = ttnn.repeat_interleave(t, repeat, dim=1)
                _free(t, flat, rep)
                t = rep
            # [1, H, L, D] -> [H, nc, chunk, D] -> [nc, H, chunk, D]
            t = ttnn.reshape(t, (nv, nc, chunk, head_dim))
            t = ttnn.permute(t, (1, 0, 2, 3))
            return t

        q = to_heads(q_flat, s.num_k_heads, s.head_k_dim, s.v_per_k)
        k = to_heads(k_flat, s.num_k_heads, s.head_k_dim, s.v_per_k)
        v = to_heads(v_flat, nv, s.head_v_dim, 1)
        _free(q_flat, q)
        _free(k_flat, k)
        _free(v_flat, v)

        q = self._l2norm(q)
        k = self._l2norm(k)
        q = ttnn.multiply(q, 1.0 / math.sqrt(s.head_k_dim))

        def to_scalar_heads(flat):
            # [1, 1, L, nv] -> [1, nv, L, 1] -> [nc, nv, chunk, 1]
            t = ttnn.permute(flat, (0, 3, 2, 1))
            t = ttnn.reshape(t, (nv, nc, chunk, 1))
            return ttnn.permute(t, (1, 0, 2, 3))

        beta_h = to_scalar_heads(beta)
        g_h = to_scalar_heads(g)
        ttnn.deallocate(beta)
        ttnn.deallocate(g)

        g_cum = ttnn.cumsum(g_h, dim=-2)
        ttnn.deallocate(g_h)

        # decay_mask[i, j] = exp(g_cum[i] - g_cum[j]) for i >= j else 0.
        # tril BEFORE exp so the (discarded) upper triangle cannot overflow.
        g_row = ttnn.transpose(g_cum, -2, -1)  # [nc, nv, 1, chunk]
        diff = ttnn.subtract(g_cum, g_row)
        ttnn.deallocate(g_row)
        diff = ttnn.tril(diff)
        decay = ttnn.exp(diff)
        ttnn.deallocate(diff)
        decay = ttnn.tril(decay)

        k_beta = ttnn.multiply(k, beta_h)
        v_beta = ttnn.multiply(v, beta_h)
        ttnn.deallocate(beta_h)
        ttnn.deallocate(v)

        kk = ttnn.matmul(k_beta, ttnn.transpose(k, -2, -1), dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)
        attn0 = ttnn.neg(ttnn.tril(ttnn.multiply(kk, decay), diagonal=-1))
        ttnn.deallocate(kk)

        inv = self._unit_tri_inverse(attn0, chunk)
        ttnn.deallocate(attn0)

        value = ttnn.matmul(inv, v_beta, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)
        ttnn.deallocate(v_beta)
        exp_gcum = ttnn.exp(g_cum)
        k_cumdecay = ttnn.matmul(
            inv,
            ttnn.multiply(k_beta, exp_gcum),
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_cfg,
        )
        ttnn.deallocate(inv)
        ttnn.deallocate(k_beta)

        g_last = ttnn.slice(g_cum, [0, 0, chunk - 1, 0], [nc, nv, chunk, 1])
        decay_to_end = ttnn.exp(ttnn.subtract(g_last, g_cum))
        exp_g_last = ttnn.exp(g_last)
        ttnn.deallocate(g_last)
        k_decayed = ttnn.multiply(k, decay_to_end)
        ttnn.deallocate(decay_to_end)
        q_decayed = ttnn.multiply(q, exp_gcum)
        ttnn.deallocate(exp_gcum)

        outputs = []
        for i in range(nc):
            sl = lambda t, w: ttnn.slice(t, [i, 0, 0, 0], [i + 1, nv, chunk, w])  # noqa: E731
            q_i = sl(q, s.head_k_dim)
            k_i = sl(k, s.head_k_dim)
            v_i = sl(value, s.head_v_dim)
            d_i = sl(decay, chunk)
            kc_i = sl(k_cumdecay, s.head_k_dim)
            qd_i = sl(q_decayed, s.head_k_dim)
            kd_i = sl(k_decayed, s.head_k_dim)
            gl_i = ttnn.slice(exp_g_last, [i, 0, 0, 0], [i + 1, nv, 1, 1])

            intra = ttnn.multiply(
                ttnn.matmul(
                    q_i,
                    ttnn.transpose(k_i, -2, -1),
                    dtype=ttnn.float32,
                    compute_kernel_config=self.compute_cfg,
                ),
                d_i,
            )
            v_prime = ttnn.matmul(kc_i, state, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)
            v_new = ttnn.subtract(v_i, v_prime)
            ttnn.deallocate(v_prime)
            inter = ttnn.matmul(qd_i, state, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)
            out_i = ttnn.add(
                inter,
                ttnn.matmul(intra, v_new, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg),
            )
            ttnn.deallocate(inter)
            ttnn.deallocate(intra)
            outputs.append(out_i)

            decayed_state = ttnn.multiply(state, gl_i)
            update = ttnn.matmul(
                ttnn.transpose(kd_i, -2, -1),
                v_new,
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

        # ttnn.concat of a single tensor returns that tensor, so only free the pieces when
        # a genuinely new buffer was produced.
        if len(outputs) == 1:
            core = outputs[0]
        else:
            core = ttnn.concat(outputs, dim=0)  # [nc, nv, chunk, head_v_dim]
            for tensor in outputs:
                ttnn.deallocate(tensor)
        core = ttnn.permute(core, (1, 0, 2, 3))  # [nv, nc, chunk, Dv]
        core = ttnn.reshape(core, (1, nv, padded, s.head_v_dim))
        core = ttnn.permute(core, (0, 2, 1, 3))  # [1, L, nv, Dv]

        z_heads = ttnn.reshape(z, (1, padded, nv, s.head_v_dim))
        normed = self._rms_norm(ttnn.typecast(core, ttnn.bfloat16), self.w["gated_norm"])
        ttnn.deallocate(core)
        gated = ttnn.multiply(normed, ttnn.silu(z_heads))
        ttnn.deallocate(normed)
        ttnn.deallocate(z_heads)
        flat = ttnn.reshape(gated, (1, 1, padded, s.value_dim))
        _free(gated, flat)
        out = ttnn.linear(flat, self.w["out_proj"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        ttnn.deallocate(flat)
        return out, state, new_conv_state

    def _linear_attention_decode(self, x):
        """Single-token gated delta rule for all ``max_batch`` users at once."""
        s = self.shapes
        batch = self.max_batch
        nv = s.num_v_heads
        # decode layout [1, 1, B, hidden] -> per-user rows [1, B, 1, hidden]
        x_rows = ttnn.reshape(x, (1, batch, 1, s.hidden_size))
        mixed_qkv, z, beta, g = self._gdn_inputs(x_rows)
        _free(x_rows, x)

        prefix = ttnn.slice(self.conv_state, [0, 0, 1, 0], [1, batch, s.conv_kernel_size, s.conv_dim])
        window = ttnn.concat([prefix, mixed_qkv], dim=-2)  # [1, B, K, conv_dim]
        ttnn.deallocate(prefix)
        ttnn.deallocate(mixed_qkv)
        acc = None
        for j in range(s.conv_kernel_size):
            tap = ttnn.slice(window, [0, 0, j, 0], [1, batch, j + 1, s.conv_dim])
            term = ttnn.multiply(tap, self.w["conv_taps"][j])
            ttnn.deallocate(tap)
            acc = term if acc is None else ttnn.add(acc, term)
        ttnn.copy(window, self.conv_state)
        ttnn.deallocate(window)
        conv_out = ttnn.silu(acc)
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

        q = self._l2norm(q)
        k = self._l2norm(k)
        q = ttnn.multiply(q, 1.0 / math.sqrt(s.head_k_dim))

        beta_h = ttnn.reshape(beta, (1, batch * nv, 1, 1))
        g_h = ttnn.reshape(g, (1, batch * nv, 1, 1))
        decay = ttnn.exp(g_h)
        ttnn.deallocate(g_h)

        state = ttnn.multiply(self.recurrent_state, decay)
        ttnn.deallocate(decay)
        kv_mem = ttnn.matmul(k, state, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)
        delta = ttnn.multiply(ttnn.subtract(v, kv_mem), beta_h)
        ttnn.deallocate(kv_mem)
        ttnn.deallocate(v)
        ttnn.deallocate(beta_h)
        update = ttnn.matmul(
            ttnn.transpose(k, -2, -1), delta, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg
        )
        ttnn.deallocate(delta)
        ttnn.deallocate(k)
        new_state = ttnn.add(state, update)
        ttnn.deallocate(state)
        ttnn.deallocate(update)
        out = ttnn.matmul(q, new_state, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)
        ttnn.deallocate(q)
        ttnn.copy(new_state, self.recurrent_state)
        ttnn.deallocate(new_state)

        core = ttnn.reshape(out, (1, batch, nv, s.head_v_dim))
        _free(out, core)
        z_heads = ttnn.reshape(z, (1, batch, nv, s.head_v_dim))
        normed = self._rms_norm(ttnn.typecast(core, ttnn.bfloat16), self.w["gated_norm"])
        ttnn.deallocate(core)
        gated = ttnn.multiply(normed, ttnn.silu(z_heads))
        ttnn.deallocate(normed)
        ttnn.deallocate(z_heads)
        flat = ttnn.reshape(gated, (1, 1, batch, s.value_dim))
        _free(gated, flat)
        result = ttnn.linear(flat, self.w["out_proj"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
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
        assert (
            len(hidden_states.shape) == 4 and int(hidden_states.shape[0]) == 1 and int(hidden_states.shape[1]) == 1
        ), f"prefill expects [1, 1, seq_len, hidden]; got {hidden_states.shape}"
        seq_len = int(hidden_states.shape[2])
        assert 1 <= seq_len <= self.max_seq_len, f"seq_len {seq_len} outside [1, {self.max_seq_len}]"
        assert user_id < self.max_batch, f"user_id {user_id} >= max_batch {self.max_batch}"

        if s.layer_type == FULL_ATTENTION:
            assert (
                page_table is not None and page_tables_per_chunk is not None
            ), "full_attention prefill requires a paged KV cache: pass page_table and page_tables_per_chunk"
            assert rot_mats is not None, "full_attention prefill requires rot_mats=(cos, sin)"
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
            x_chunk = ttnn.slice(hidden_states, [0, 0, chunk_start, 0], [1, 1, chunk_start + logical, s.hidden_size])
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
                conv_prefix = ttnn.slice(new_conv_state, [0, 0, 1, 0], [1, 1, s.conv_kernel_size, s.conv_dim])
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
        assert (
            int(hidden_states.shape[2]) == self.max_batch
        ), f"decode batch {int(hidden_states.shape[2])} != max_batch {self.max_batch}"
        residual = hidden_states
        normed = self._rms_norm(hidden_states, self.w["input_layernorm"])
        if s.layer_type == FULL_ATTENTION:
            assert current_pos is not None and page_table is not None and rot_mats is not None
            mixed = self._full_attention_decode(
                normed, current_pos=current_pos, page_table=page_table, rot_mats=rot_mats
            )
        else:
            mixed = self._linear_attention_decode(normed)
        ttnn.deallocate(normed)
        hidden = ttnn.add(residual, mixed)
        ttnn.deallocate(mixed)
        normed2 = self._rms_norm(hidden, self.w["post_attention_layernorm"])
        mlp_out = self._mlp(normed2)
        ttnn.deallocate(normed2)
        out = ttnn.add(hidden, mlp_out)
        ttnn.deallocate(hidden)
        ttnn.deallocate(mlp_out)
        return out

    # ------------------------------------------------------------- helpers

    def prefill_chunk_plan(self, seq_len: int) -> list[tuple[int, int, int]]:
        """``[(chunk_start, logical_len, padded_len)]`` for a prefill of ``seq_len`` tokens.

        Callers use this (host-side, outside the measured pass) to slice the per-chunk page
        tables that :meth:`prefill_forward` needs: chunk ``i`` covers page-table blocks
        ``[chunk_start // block_size, (chunk_start + padded_len) // block_size)``.
        """
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

        Call once after prefilling every user and before capturing the decode trace: the
        decode buffers keep stable addresses and are updated in place, so trace replay never
        reallocates.

        **Limitation.** This rewrites *every* slot from that user's post-prefill snapshot, so
        calling it again after prefilling one new user also rewinds all the other users'
        recurrent state to where their own prefill left it. That is fine for the
        prefill-all-then-decode pattern this stage tests, but continuous batching needs a
        per-slot fold instead; the batch-wide buffers are laid out user-major
        (``[1, batch * num_v_heads, head_k_dim, head_v_dim]``), so a single-slot variant is a
        strided ``ttnn.copy`` into one slice. Left for the serving stage, which is where the
        slot-eviction policy lives.
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
    """``tensor[start : start + logical]`` zero-padded to ``padded``; ``tensor`` stays alive.

    A full-range ``ttnn.slice`` returns a view, so the intermediate must never be freed
    against the caller's tensor.
    """
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

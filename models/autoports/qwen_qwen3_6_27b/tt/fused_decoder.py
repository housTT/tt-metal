# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Graph-fused TTNN decoder for Qwen/Qwen3.6-27B (HF ``model_type: qwen3_5``).

:class:`FusedDecoder` is a drop-in replacement for
:class:`~.functional_decoder.FunctionalDecoder`: same constructor, same
``prefill_forward`` / ``decode_forward`` / ``prefill_chunk_plan`` /
``prepare_decode_state`` contract, same paged KV cache, same per-user linear-attention state,
same acceptance bar.  Only the *graph* changes - every rewrite below is a
numerically-equivalent replacement of a primitive op sequence, proven against the same HF
reference at the same PCC bar (see ``doc/fused_decoder/``).

What is fused, and why (the three kinds of rewrite, in the skill's priority order)
--------------------------------------------------------------------------------

**Dedicated fused ops** (highest priority - one hand-written kernel replaces a spelled-out
primitive sequence):

``ttnn.transformer.chunk_gated_delta_rule``
    Replaces the entire ``linear_attention`` prefill delta-rule core: the head split, the
    L2 norms, the GQA head expansion, the ``1/sqrt(Dk)`` scale, the decay cumsum and mask, the
    recursive unit-triangular (WY) inverse and the python loop over sub-chunks - roughly 700
    device ops for one 2048-token chunk - become **one** op.  It is called on the *flat*
    token-major path (rank-3 ``[1, T, H*D]`` q/k/v, ``chunk_size=32``), which additionally
    folds the Q/K L2 norm and the scale into the kernel, so no ``l2norm``,
    ``repeat_interleave`` or head permute survives on the prefill path at all.
``ttnn.experimental.rotary_embedding_hf``
    Replaces the prefill partial-RoPE ``slice/slice/neg/concat/mul/mul/add`` with one op on the
    rotary slice.  Only the rotary/passthrough split remains, because the rotary factor is 0.25
    (64 of 256 head channels) and the op rotates its whole input width.
``ttnn.addcmul``
    Replaces the decode recurrent-state update's ``multiply`` + ``add`` with one pass over the
    carried state, written in place at the persistent buffer's address (work log section 3.21).
    ``ttnn.experimental.rotate_half`` was used here too and was **reverted on measurement**: it is
    single-core by construction, so the four ops it replaced are faster under trace at the
    advertised batch (work log section 3.22).
``ttnn.rms_norm``
    Replaces the decode GatedDeltaNet Q/K L2 norm's ``mul/sum/rsqrt/mul`` (plus the query's
    scale multiply), by the identity ``l2norm(x) == rms_norm(x, eps/D) / sqrt(D)``.

**Graph rewrites** (structural / algebraic / peer merge):

* ``in_proj_b`` and ``in_proj_a`` share their LHS and are both float32-weighted, so they become
  **one** matmul over the concatenated weight; both were dispatch-bound, not bandwidth-bound.
  The two unpacked weights are freed rather than kept on device.
* The z-gated per-head RMS norm becomes a **group reduction against two constant matrices**, so
  it runs on the flat token-major layout that both the delta-rule output and the ``in_proj_z``
  matmul already produce; the ``[L, H*D]`` <-> ``[L, H, D]`` tile relayouts either side of the
  functional layer's ``ttnn.rms_norm`` are the two most expensive ops in the fused prefill
  otherwise.
* The prefill causal conv1d runs its FIR in bfloat16 while its carried state stays float32: a
  float32 *height-broadcast* multiply is the one binary-op shape on this device that runs far
  below bandwidth, and nothing downstream can use more than bfloat16.
* The decode recurrence matmuls get an explicit core grid; the default batched program factory
  puts 48 independent per-head problems on 4 and 16 cores.
* The decode causal conv keeps its ``K`` packed state rows as ``K`` separate batch-major buffers
  instead of one ``[1, B, K, conv_dim]`` tensor.  Slicing a row out of the packed tensor puts
  the shift on the *tile-height* axis, which costs an ``untilize_with_unpadding`` +
  ``tilize_with_val_padding`` pair per tap; batch-major buffers make every tap a plain
  elementwise read and the shift an in-place ``ttnn.copy``.
* The decode RMS norms run **width-sharded across the grid**.  Interleaved
  ``[1, 1, batch, 5120]`` has one tile row, so the interleaved kernel parallelises over one
  core; sharding the *width* uses :data:`NORM_SHARD_CORES`.
* Decode K/V go straight from ``nlp_create_qkv_heads_decode`` into ``paged_update_cache`` in the
  memory config the head op already produced, instead of a
  sharded->interleaved->sharded round trip.

**Op merging** (fold a neighbour into an op that was already running):

* The MLP's ``silu`` folds into the gate*up multiply as an input activation.
* The decode recurrence's outer-product transpose folds into its matmul as ``transpose_a=True``.
* The gated delta net's ``dt_bias`` folds into the packed ``a``/``b`` matmul as its bias row.
* The gated norm's ``rsqrt`` folds into the epsilon add as an output activation.
* The attention output gate's ``sigmoid`` folds into the gate multiply the same way.
* The gated delta net's ``silu(z)`` folds into the z-gate multiply the same way.
* The causal conv's trailing ``silu`` folds into the last tap's ``add`` as an *output*
  activation, in both prefill and decode.
* The decode recurrent-state update writes straight into the persistent state buffer through
  the binary op's ``output_tensor``, instead of a separate ``ttnn.copy``.
* The prefill KV-cache typecasts are guarded on dtype, so the default bfloat16 cache does not
  dispatch two ``bfloat16 -> bfloat16`` no-ops per chunk.

Everything that touches ``torch`` still happens in :meth:`FusedDecoder.from_state_dict`.
"""

from __future__ import annotations

import math
import os

import ttnn

from .functional_decoder import FunctionalDecoder, _free, _round_up, _sdpa_program_config, _shape, _zero_after_seq
from .model_config import FULL_ATTENTION, LINEAR_ATTENTION

#: Chunk length ``ttnn.transformer.chunk_gated_delta_rule`` runs the recurrence at.
#:
#: This is an internal tiling choice of the op, not a model parameter: the delta rule is exact
#: for any chunk length, and 32 gives the same result as the 64 of
#: :data:`~.model_config.DELTA_CHUNK`.  32 is the only value this layer may use, for two
#: independent reasons:
#:
#: 1. The *flat* (rank-3, token-major) input path - which is what lets the op L2-normalise Q/K
#:    and fold the ``1/sqrt(Dk)`` scale in-kernel, and skip the head split entirely - is gated
#:    on ``chunk_size == 32`` in ``chunk_gated_delta_rule.cpp`` (``qk_norm = flat_qk && C ==
#:    32``), because the in-kernel norm reuses circular buffers that are only free at 32.  That
#:    same flat path additionally requires the op's *phased* prep/scan implementation, which is
#:    its default but is switchable off through the ``QWEN_GDN_PHASED`` environment variable
#:    (``TT_FATAL(!flat_qk || (phased && qk_norm), ...)``); :meth:`FusedDecoder.__init__` asserts
#:    it is not disabled rather than letting a later stage hit that fatal.
#: 2. At 64 each per-chunk WY matrix is a 2x2 tile block whose bottom-right 32x32 sub-block can
#:    be ill-conditioned enough for the float32 block inverse to lose precision.  Measured
#:    model-free on this checkout with real-range synthetic inputs
#:    (``doc/fused_decoder/probes/probe_chunk_gdr.py``, log
#:    ``doc/fused_decoder/logs/probe_chunk_gdr.log``): at 2048 tokens the output PCC against
#:    HF's ``torch_chunk_gated_delta_rule`` is **0.999994 at chunk 32** and **0.903635 at
#:    chunk 64**.
FUSED_DELTA_CHUNK = 32

#: Cores the decode RMS norms are width-sharded over.
#:
#: The decode hidden state is ``[1, 1, batch, 5120]`` - a single tile row - so the interleaved
#: ``ttnn.rms_norm`` parallelises over exactly one core, which in the stage-1 decode report is
#: the third-largest op of the step, twice over.  Width sharding needs a core count that divides
#: ``hidden_size / TILE_WIDTH`` (160 here); measured on this checkout
#: (``doc/fused_decoder/probes/probe_small_ops.py``, log
#: ``doc/fused_decoder/logs/probe_small_ops.log``), wall time for
#: interleaved->shard->rms_norm->interleaved:
#:
#: The measured curve is ``work_log.md`` section 3.3, generated from that log so the two cannot
#: drift.  Its shape: sharding is several times faster than the interleaved norm, flat between 16
#: and 20 cores - they are inside each other's run-to-run spread, which the probe now reports as a
#: stdev column rather than asserting - and it rises from 32 upwards as the shard/unshard overhead
#: starts to dominate.  20 is used, and which of 16 and 20 the log calls faster has changed
#: between runs, which is the point: ``test_selected_constants_are_the_measured_best`` holds this
#: one to the same within-the-combined-spread rule as every other shipped constant, rather than to
#: a claim about which is nominally ahead.
NORM_SHARD_CORES = 20

#: Column stride of the packed ``b``/``a`` gate projection.  ``in_proj_b`` and ``in_proj_a`` are
#: both ``num_v_heads`` (48) wide and share their input, so they are packed into one matmul; the
#: second block starts at a tile-aligned column so both halves come back out with a plain slice.
_AB_STRIDE = 64

#: Core grids for the single-token gated-delta-rule recurrence matmuls.
#:
#: At batch 1 these are 48 independent ``[1,128] x [128,128]`` state reads and one
#: ``[128,1] x [1,128]`` outer product per head.  ``ttnn.matmul``'s default batched program
#: factory spreads them over 4 and 16 cores respectively.  Measured on this checkout
#: (``doc/fused_decoder/probes/probe_decode_recurrence.py``, log
#: ``doc/fused_decoder/logs/probe_decode_recurrence.log``), median and spread over 30 repeats.
#:
#: The measured table lives in ``doc/fused_decoder/work_log.md`` section 3.6, generated from that
#: log, and is not transcribed here so the two cannot drift.  What it shows: the default program
#: factory is the slowest row of every sweep, the explicit grids differ from each other by more
#: than their spreads at 1536 head problems and by less at 48, and the caption names which is
#: which per family and per regime.  That is why the state-read grid is keyed by regime and the
#: outer product's is not, and ``test_selected_grids_are_the_measured_best`` re-derives both from
#: the log.  Every grid is exact (PCC 1.000000 against torch) - a grid only changes how the
#: independent per-head problems are distributed.  ``ttnn.experimental.group_attn_matmul`` was tried for the
#: state read and rejected: its contract ties the batch dim to the number of users
#: ("Num of users must match!") - and a stage review showed that first attempt had mapped the op's
#: batch axis onto the flattened ``batch * num_v_heads`` axis.  Mapped the way the op wants
#: (``a = [1, num_v_heads, batch, head_dim]``, ``b = [batch, num_v_heads, head_k_dim, head_v_dim]``,
#: which the user-major state provides as a free view) every shape assertion passes at
#: ``max_batch`` 32 - and the op then overflows L1: its circular buffers come to 6484864 B in
#: float32 and 3298176 B in bfloat16 against 1572864 B of L1.  That is the real blocker, it is
#: quantified, and it is a factor of two out even at half precision.  See ``work_log.md`` §3.6.
#: ``core_grid`` for the two recurrence matmuls.  Swept over eighteen grids at **both** decode
#: regimes - 48 head problems at batch 1 and 1536 at the advertised ``max_batch`` - by
#: ``doc/fused_decoder/probes/probe_decode_recurrence.py``; the tables are ``work_log.md``
#: section 3.6.  The outer product's grid is the fastest measured at 1536 head problems and inside the
#: run-to-run spread of the fastest at 48, which is the rule the whole stage uses when a lever
#: is a tie in one regime and decisive in the other.  The sweep runs to the edge of this
#: device's grid in both axes: a stage review pointed out that the previous tuple stopped at
#: ``y = 6`` while the trend was still improving, and widening it moved the state read.
#:
#: ``tests/test_fused_decoder_docs.py::test_selected_grids_are_the_measured_best`` re-derives
#: both of these from the probe log and fails if a shipped grid is not within a stdev of the
#: fastest measured one, at every regime measured - the claim above used to be a sentence.
#:
#: The state read is the one lever where no single grid wins at both regimes once the sweep runs
#: to the edge of the device grid: ``6x4`` is fastest at 48 head problems and ``10x4`` at 1536,
#: each distinguishably faster than the other in its own regime.  So it is keyed by regime, the
#: way the small-N matmul grids are keyed by phase, and :meth:`FusedDecoder.__init__` picks by
#: this layer's own head-problem count.  The outer product has one grid that holds at both.
_RECURRENCE_READ_GRID = {"small": (6, 4), "large": (10, 4)}
_RECURRENCE_OUTER_GRID = (2, 11)
#: Head problems (``max_batch * num_v_heads``) at and above which the *large* recurrence-read grid
#: is used.  1536 is the count the probe's second regime measures, i.e. ``max_batch`` 32.
#:
#: The sweep measures 48 and 1536, so every ``max_batch`` from 2 to 31 takes the small grid on the
#: strength of the 48-head measurement rather than one of its own.  That is a real gap and it is
#: recorded as a limitation in ``doc/fused_decoder/README.md``; the two grids differ by about a
#: tenth of the state read at 1536, so the exposure is small and bounded by the two measured ends.
_RECURRENCE_LARGE_HEADS = 1536

#: ``core_grid`` for the three matmuls this stage created whose N is a handful of tiles: the
#: packed ``a``/``b`` projection (N = 4 tiles) and the two gated-norm constant matmuls (N = 2 and
#: N = 192 tiles).  The default 1D program config spreads output columns over the whole grid, so a
#: row with 2 or 4 output tiles pays a full-grid broadcast of its activation to fill a handful of
#: cores.  Naming a smaller grid is the same lever §3.6 used on the decode recurrence, and it is
#: worth 2-3x on the two-and-four-tile rows at decode.  The winning grid is a function of both the
#: shape *and* the row count, so prefill and decode are separate entries; every grid from the full
#: device down to 1x2 was measured at both row counts by
#: ``doc/fused_decoder/probes/probe_matmul_bound.py``, and the tables are ``work_log.md``
#: section 3.18, generated from that probe's log.  ``None`` means the default won.
_AB_MATMUL_GRID = {"prefill": (4, 8), "decode": (1, 4)}
_GROUP_SUM_GRID = {"prefill": (8, 8), "decode": (1, 4)}
_GROUP_EXPAND_GRID = {"prefill": None, "decode": (2, 8)}

#: ``max_batch`` at and above which the decode z-gated norm uses the same group reduction as
#: prefill (:meth:`FusedDecoder._gated_norm_and_project`) instead of the reshape-and-``rms_norm``
#: form.  The two are the same arithmetic; which is cheaper is a pure function of the row count,
#: because the group form is two skinny constant matmuls that barely move with it while the
#: reshape form's two tile relayouts grow with it.  Measured at the real decode shapes over five
#: batch sizes by ``doc/fused_decoder/probes/probe_gated_norm_batch.py``; the table is
#: ``work_log.md`` section 3.17, generated from that probe's log so the two cannot drift.  The
#: two forms cross between **16 and 32**: the reshape form wins every measured batch up to and
#: including 16, and the group form wins at 32.  Which batch that is, and which form wins each
#: one, are derived from the log by the caption in §3.17 rather than asserted here, because round
#: 21 wrote a verdict into this comment that its own log denied.  The threshold has been at
#: 32, then 16, and is 32 again - each move followed the measurement of the day, and
#: ``test_selected_constants_are_the_measured_best`` now binds it to the probe log so it cannot
#: drift from it silently.  Their outputs agree to PCC 0.99999 or better at every batch measured,
#: and ``test_batched_users`` covers 4, 16 and 32, i.e. both sides of the boundary and the
#: boundary itself.
_GATED_NORM_GROUP_BATCH = 32

#: The *decode* causal-conv FIR stays float32, and this constant is why it is not a knob.
#:
#: ``doc/fused_decoder/probes/probe_decode_conv_dtype.py`` measures the bfloat16 form as clearly
#: faster from batch 4 up (the table is ``work_log.md`` §3.25) - and it was built, shipped behind
#: this threshold, and **reverted**, because the suite caught what that probe cannot see: batched
#: traced decode fell below the 0.995 PCC bar against HF for the shortest-prefill user in the
#: batch.  That run is committed (``doc/fused_decoder/logs/rejected_bf16_decode_fir.log``) and
#: §3.25 reads it: the failures track the *size of the carried state* rather than the number of
#: steps - a per-step compounding probe does not reproduce them - so the mechanism is recorded as
#: open rather than guessed.  The prefill FIR is unaffected: its output feeds
#: ``chunk_gated_delta_rule``, which casts to bfloat16 anyway and accumulates the state in kernel.
_DECODE_CONV_BF16_BATCH = None

#: Epsilon of the GatedDeltaNet Q/K L2 norm, matching HF's ``l2norm(x, dim=-1, eps=1e-6)`` and
#: the functional layer's ``FunctionalDecoder._l2norm``.
_L2NORM_EPS = 1e-6

#: Tile-padded head count of the per-head gated RMS norm's group reduction.  The two 0/1-style
#: constant matrices are built at this width with explicit zero rows/columns rather than relying
#: on tile padding, so the padded lanes provably contribute nothing.
_GDN_GROUP_PAD = 64


def _gdn_phased_disabled() -> bool:
    """Whether ``QWEN_GDN_PHASED`` reads as "off" to ``chunk_gated_delta_rule``.

    Matched to the op's own test (``chunk_gated_delta_rule.cpp``: ``e == nullptr || e[0] != '0'``),
    so ``0``, ``00`` and ``0anything`` all count as disabled here exactly as they do there.
    """
    value = os.environ.get("QWEN_GDN_PHASED")
    return value is not None and value.startswith("0")


def _norm_shard_cores(hidden_size: int, grid) -> int:
    """Largest usable width-shard core count for a norm over ``hidden_size`` channels.

    :data:`NORM_SHARD_CORES` unless it does not divide the tile-width count or does not fit the
    grid, in which case the largest divisor that does.
    """
    tiles = hidden_size // ttnn.TILE_SIZE
    limit = min(grid.x * grid.y, tiles)
    if tiles % NORM_SHARD_CORES == 0 and NORM_SHARD_CORES <= limit:
        return NORM_SHARD_CORES
    for cores in range(limit, 0, -1):
        if tiles % cores == 0:
            return cores
    return 1


class FusedDecoder(FunctionalDecoder):
    """One Qwen3.5/3.6 decoder layer on a TTNN mesh device, graph-fused.

    Public behaviour is identical to :class:`~.functional_decoder.FunctionalDecoder`; see the
    module docstring for the list of rewrites.
    """

    #: Dedicated tt-metal ops this layer must actually dispatch.  ``tests/test_fused_decoder.py``
    #: asserts each one is called during a real prefill/decode pass, so a silent fall back to the
    #: functional graph is a test failure rather than a quiet performance regression.
    FUSED_OPS = (
        "ttnn.transformer.chunk_gated_delta_rule",
        "ttnn.experimental.rotary_embedding_hf",
        "ttnn.addcmul",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        s = self.shapes
        if s.layer_type == LINEAR_ATTENTION and _gdn_phased_disabled():
            raise ValueError(
                "QWEN_GDN_PHASED is set to a value the op reads as 'off', which disables the "
                "phased prep/scan implementation of ttnn.transformer.chunk_gated_delta_rule - "
                "the one the flat rank-3 input path this layer uses requires. Unset it or set it "
                "to 1. Note the op re-reads it per call, so a later stage that sets it after "
                "construction still reaches the op's own TT_FATAL; this check only catches the "
                "common case of it being set up front."
            )
        grid = self.mesh_device.compute_with_storage_grid_size()
        # Width-sharded decode norm: one tile row per 32 users, the full hidden width split
        # across the grid.
        rows = _round_up(self.max_batch, ttnn.TILE_SIZE)
        cores = _norm_shard_cores(s.hidden_size, grid)
        block_w = s.hidden_size // cores // ttnn.TILE_SIZE
        self.decode_norm_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(rows, s.hidden_size // cores),
            core_grid=ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_norm_prgm_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=max(w for w in range(1, 5) if block_w % w == 0),
            block_h=rows // ttnn.TILE_SIZE,
            block_w=block_w,
            inplace=False,
        )
        # Recurrence matmul grids, clamped to whatever grid this device actually has.  The read
        # grid is chosen by this layer's head-problem count, which is the axis the sweep varies.
        read_regime = "large" if self.max_batch * s.num_v_heads >= _RECURRENCE_LARGE_HEADS else "small"
        read_spec = _RECURRENCE_READ_GRID[read_regime]
        self.recurrence_read_grid = ttnn.CoreGrid(y=min(read_spec[0], grid.y), x=min(read_spec[1], grid.x))
        self.recurrence_outer_grid = ttnn.CoreGrid(
            y=min(_RECURRENCE_OUTER_GRID[0], grid.y), x=min(_RECURRENCE_OUTER_GRID[1], grid.x)
        )

        def _clamped(spec):
            """``{phase: (y, x) or None}`` -> ``{phase: CoreGrid or None}`` for this device."""
            return {
                phase: None if value is None else ttnn.CoreGrid(y=min(value[0], grid.y), x=min(value[1], grid.x))
                for phase, value in spec.items()
            }

        self.ab_matmul_grid = _clamped(_AB_MATMUL_GRID)
        self.group_sum_grid = _clamped(_GROUP_SUM_GRID)
        self.group_expand_grid = _clamped(_GROUP_EXPAND_GRID)
        # Decode causal-conv state, one batch-major buffer per tap: buffer ``j`` holds token
        # ``t - (K - 1) + j`` of every user, which is row ``j + 1`` of the functional layer's
        # packed ``[1, batch, K, conv_dim]`` state.  A decode step therefore reads each buffer
        # whole and the shift is a copy chain, rather than a slice out of a tile-height axis.
        #
        # The inherited packed ``conv_state`` is **not** read by the fused decode, and a decode
        # step does not write it either, so between steps it holds whatever the last
        # :meth:`prepare_decode_state` or :meth:`current_conv_state` put there.  That is a real
        # divergence from the functional layer, which rewrites the packed buffer every step, so
        # the tap buffers are laid out to make it recoverable: there is one buffer per *packed
        # row*, K of them rather than the K - 1 the FIR reads, and :meth:`current_conv_state`
        # folds them back into the packed buffer for any caller that wants it (a serving stage
        # reading state mid-generation, or the state test).  The extra row costs one copy per
        # step and one buffer per layer, both measured in the stage documents.
        self.conv_state_split: list = []
        if s.layer_type == LINEAR_ATTENTION:
            self.conv_state_split = [
                ttnn.zeros(
                    (1, 1, self.max_batch, s.conv_dim),
                    dtype=self.conv_state.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                )
                for _ in range(s.conv_kernel_size)
            ]

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx: int, mesh_device, **kwargs) -> "FusedDecoder":
        """Build the fused layer from an HF submodule-relative state dict.

        Delegates the shared weight preparation to
        :meth:`~.functional_decoder.FunctionalDecoder.from_state_dict` and then adds the
        artifacts only the fused graph needs.  This is the only place ``torch`` is used.
        """
        import torch  # setup-time only; never on the prefill/decode path

        layer = super().from_state_dict(
            state_dict, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device, **kwargs
        )
        s = layer.shapes

        def _tt(tensor, dtype):
            return ttnn.from_torch(
                tensor,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        if s.layer_type == LINEAR_ATTENTION:
            # Shared-LHS merge: in_proj_b and in_proj_a both read the normed hidden state and are
            # both float32-weighted, so pack them into one [hidden, 2 * _AB_STRIDE] weight with the
            # second block starting on a tile boundary.
            if s.num_v_heads > _AB_STRIDE:
                raise ValueError(
                    f"num_v_heads {s.num_v_heads} exceeds the packed a/b column stride {_AB_STRIDE}; "
                    "raise _AB_STRIDE to the next tile multiple"
                )
            b_w = state_dict["linear_attn.in_proj_b.weight"].to(torch.float32)  # [nv, hidden]
            a_w = state_dict["linear_attn.in_proj_a.weight"].to(torch.float32)
            packed = torch.zeros(s.hidden_size, 2 * _AB_STRIDE, dtype=torch.float32)
            packed[:, : s.num_v_heads] = b_w.t()
            packed[:, _AB_STRIDE : _AB_STRIDE + s.num_v_heads] = a_w.t()
            layer.w["in_proj_ab"] = _tt(packed.reshape(1, 1, s.hidden_size, 2 * _AB_STRIDE), ttnn.float32)
            # ``dt_bias`` is added to ``a`` right after that matmul, and a bias row is what
            # ``ttnn.linear`` already takes - the skill's "matmul + bias -> linear" merge.  Pack it
            # into the ``a`` block's columns, zeros under ``b``.
            packed_bias = torch.zeros(2 * _AB_STRIDE, dtype=torch.float32)
            packed_bias[_AB_STRIDE : _AB_STRIDE + s.num_v_heads] = (
                state_dict["linear_attn.dt_bias"].to(torch.float32).reshape(-1)
            )
            layer.w["in_proj_ab_bias"] = _tt(packed_bias.reshape(1, 1, 1, 2 * _AB_STRIDE), ttnn.float32)
            # The two unpacked weights are dead in this graph; free them rather than hold 2.6 MB
            # of device DRAM per layer that nothing reads.
            for dead in ("in_proj_b", "in_proj_a"):
                ttnn.deallocate(layer.w.pop(dead))

            # Constant tiles for chunk_gated_delta_rule.  The op builds these itself when they are
            # not supplied, but that build is a host upload and so is illegal under trace capture;
            # owning them here keeps them device-resident and device-lifetime-scoped.
            chunk = FUSED_DELTA_CHUNK
            index = torch.arange(32)
            low_i = (index < 16).reshape(32, 1)
            low_j = (index < 16).reshape(1, 32)
            layer.const["gdn_eye"] = _tt(torch.eye(chunk).reshape(1, 1, chunk, chunk), ttnn.float32)
            layer.const["gdn_tril"] = _tt(
                torch.tril(torch.ones(chunk, chunk)).reshape(1, 1, chunk, chunk), ttnn.float32
            )
            layer.const["gdn_ones"] = _tt(torch.ones(1, 1, chunk, chunk), ttnn.float32)
            layer.const["gdn_masks"] = _tt(
                torch.cat(
                    [
                        (low_i & low_j).float(),
                        (~low_i & ~low_j).float(),
                        (~low_i & low_j).float(),
                    ],
                    dim=1,
                ).reshape(1, 1, 32, 96),
                ttnn.float32,
            )

            # bfloat16 conv taps.  The prefill FIR runs in bfloat16 (see _causal_conv), and the
            # checkpoint itself stores conv1d.weight in bfloat16, so these are the taps at their
            # native precision rather than a reduction of it.
            layer.w["conv_taps_bf16"] = [ttnn.typecast(tap, ttnn.bfloat16) for tap in layer.w["conv_taps"]]

            # Per-head gated RMS norm as a group reduction, so it can run on the flat token-major
            # [1, 1, L, value_dim] tensor and no [L, H*D] <-> [L, H, D] tile relayout is needed:
            #
            #   mean_h(core^2)        == core^2 @ gdn_group_mean        (1/head_v_dim folded in)
            #   rsqrt(...) * weight   == rsqrt(...) @ gdn_scale_expand  (the norm weight folded in)
            #
            # Both matrices are exactly representable in bfloat16: the mean factor is
            # 1/head_v_dim (a power of two) and the scale entries are the norm weight itself,
            # which is already bfloat16.
            nv, dv = s.num_v_heads, s.head_v_dim
            norm_w = (state_dict["linear_attn.norm.weight"].to(torch.float32)).reshape(dv)
            group_mean = torch.zeros(s.value_dim, _GDN_GROUP_PAD, dtype=torch.float32)
            scale_expand = torch.zeros(_GDN_GROUP_PAD, s.value_dim, dtype=torch.float32)
            for head in range(nv):
                group_mean[head * dv : (head + 1) * dv, head] = 1.0 / dv
                scale_expand[head, head * dv : (head + 1) * dv] = norm_w
            layer.const["gdn_group_mean"] = _tt(group_mean.reshape(1, 1, s.value_dim, _GDN_GROUP_PAD), ttnn.bfloat16)
            layer.const["gdn_scale_expand"] = _tt(
                scale_expand.reshape(1, 1, _GDN_GROUP_PAD, s.value_dim), ttnn.bfloat16
            )
        return layer

    # ------------------------------------------------------------- primitives

    def _decode_rms_norm(self, x, weight):
        """RMS norm of a decode-shaped ``[1, 1, batch, hidden]`` tensor, width-sharded."""
        sharded = ttnn.to_memory_config(x, self.decode_norm_mem_cfg)
        normed = ttnn.rms_norm(
            sharded,
            epsilon=self.shapes.rms_norm_eps,
            weight=weight,
            program_config=self.decode_norm_prgm_cfg,
            memory_config=self.decode_norm_mem_cfg,
            compute_kernel_config=self.compute_cfg,
        )
        ttnn.deallocate(sharded)
        out = ttnn.sharded_to_interleaved(normed, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(normed)
        return out

    def _mlp(self, x):
        """SwiGLU MLP with the SiLU folded into the gate*up multiply.

        The gate/up projection stays a single matmul: measured on this checkout
        (``doc/fused_decoder/probes/probe_mlp_variants.py``) splitting it into two
        ``[hidden, intermediate]`` matmuls so the SiLU could ride the gate matmul's ``activation=``
        epilogue is clearly **slower** for a 2048-token chunk - two narrower matmuls lose more than the
        slices cost - and a wash at decode.  The measured table is ``work_log.md`` section 3.8,
        generated from the probe log so the two cannot drift.
        """
        gate_up = ttnn.linear(x, self.w["mlp_gate_up"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        inter = self.shapes.intermediate_size
        lead = _shape(gate_up)[:3]
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [*lead, inter])
        up = ttnn.slice(gate_up, [0, 0, 0, inter], [*lead, 2 * inter])
        _free(gate_up, gate, up)
        out = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        result = ttnn.linear(out, self.w["mlp_down"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        ttnn.deallocate(out)
        return result

    def _apply_rope_prefill(self, x, cos, sin):
        """Partial rotary embedding over the leading ``rotary_dim`` channels, prefill layout.

        ``x`` is ``[1, heads, seq, head_dim]`` and ``cos``/``sin`` are ``[1, 1, seq, rotary_dim]``,
        which is exactly ``rotary_embedding_hf``'s prefill contract for the rotary slice.
        """
        rd = self.shapes.rotary_dim
        head_dim = self.shapes.head_dim
        lead = _shape(x)[:-1]
        starts = [0] * len(lead)
        rot = ttnn.slice(x, [*starts, 0], [*lead, rd])
        embedded = ttnn.experimental.rotary_embedding_hf(rot, cos, sin, compute_kernel_config=self.compute_cfg)
        _free(rot, x)
        if rd == head_dim:
            return embedded
        passthrough = ttnn.slice(x, [*starts, rd], [*lead, head_dim])
        out = ttnn.concat([embedded, passthrough], dim=-1)
        ttnn.deallocate(embedded)
        ttnn.deallocate(passthrough)
        return out

    def _apply_rope_decode(self, x, cos, sin):
        """Partial rotary embedding, decode layout ``[1, batch, heads, head_dim]``.

        ``rotary_embedding_hf``'s decode mode needs a HEIGHT_SHARDED input *and* sharded per-user
        ``cos``/``sin``, and its prefill mode broadcasts ``cos``/``sin`` over dim 1 - the batch
        axis in this layout - so it cannot serve per-user positions here.  The rotate-half is
        spelled out rather than dedicated, for the measured reason in the body.
        """
        rd = self.shapes.rotary_dim
        head_dim = self.shapes.head_dim
        lead = _shape(x)[:-1]
        starts = [0] * len(lead)
        rot = ttnn.slice(x, [*starts, 0], [*lead, rd])
        # The rotate-half here is *not* ``ttnn.experimental.rotate_half``, and that is a
        # measurement rather than an oversight (§3.22).  That op's program factory pins
        # ``CoreCoord({0, 0})``, so it is single-core by construction: in the committed reports it
        # is the largest layout-ish row of the batch-32 ``full_attention`` decode, against a few
        # microseconds for the four ops below, which run on 64 to 110 cores.  Swapping it out
        # moved that whole traced pass by about one and a half percent and left batch 1 unchanged.
        # Prefill still uses the dedicated ``rotary_embedding_hf``, a different op and a measured
        # win (§3.2).
        half = rd // 2
        low = ttnn.slice(rot, [*starts, 0], [*lead, half])
        high = ttnn.slice(rot, [*starts, half], [*lead, rd])
        negated = ttnn.neg(high)
        ttnn.deallocate(high)
        rotated = ttnn.concat([negated, low], dim=-1)
        ttnn.deallocate(negated)
        ttnn.deallocate(low)
        direct = ttnn.multiply(rot, cos)
        crossed = ttnn.multiply(rotated, sin)
        embedded = ttnn.add(direct, crossed)
        ttnn.deallocate(direct)
        ttnn.deallocate(crossed)
        _free(rot, x)
        ttnn.deallocate(rotated)
        if rd == head_dim:
            return embedded
        passthrough = ttnn.slice(x, [*starts, rd], [*lead, head_dim])
        out = ttnn.concat([embedded, passthrough], dim=-1)
        ttnn.deallocate(embedded)
        ttnn.deallocate(passthrough)
        return out

    # ------------------------------------------------------- full attention

    def _attn_epilogue(self, attn_out, gate):
        """Sigmoid output gate + ``o_proj``, with the sigmoid folded into the gate multiply.

        ``Qwen3_5Attention``'s gate is ``attn_out * sigmoid(gate)``.  The separate ``sigmoid``
        dispatch reads and writes the whole ``[1, 1, seq, heads*head_dim]`` tensor for nothing:
        it is an input activation of the multiply that follows.  Measured device time for one
        2048-token prefill chunk: the functional report's two ops
        (``tracy/functional/full_attention/prefill_perf_report.csv``) become one in the fused
        report, and the ``elementwise`` bucket of ``perf_summary.json`` carries the difference.
        """
        gated = ttnn.multiply(attn_out, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
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
        q = self._apply_rope_prefill(q, cos, sin)
        k = self._apply_rope_prefill(k, cos, sin)

        k_cache, v_cache = self.kv_cache
        # Only cast when the cache dtype really differs.  ``ttnn.typecast`` dispatches a full
        # bfloat16 -> bfloat16 pass otherwise, which at the default cache dtype was two ops per
        # prefill chunk for nothing; ``test_bfloat8_kv_cache`` covers the branch that casts.
        k_fill = ttnn.typecast(k, k_cache.dtype) if k.dtype != k_cache.dtype else k
        v_fill = ttnn.typecast(v, v_cache.dtype) if v.dtype != v_cache.dtype else v
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
        s = self.shapes
        cos, sin = rot_mats
        q, k, v, gate = self._attn_projections(x, decode=True)
        # nlp_create_qkv_heads_decode already emits exactly the height-sharded layout
        # paged_update_cache wants, so V never leaves it.  Q and K still need the interleaved
        # norm and partial-RoPE slices.
        v_sharded = v
        q_interleaved = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        _free(q, q_interleaved)
        q = q_interleaved
        k_interleaved = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        _free(k, k_interleaved)
        q = self._rms_norm(q, self.w["q_norm"])
        k_interleaved = self._rms_norm(k_interleaved, self.w["k_norm"])
        q = self._apply_rope_decode(q, cos, sin)
        k_interleaved = self._apply_rope_decode(k_interleaved, cos, sin)

        k_cache, v_cache = self.kv_cache
        k_sharded = ttnn.to_memory_config(k_interleaved, v_sharded.memory_config())
        _free(k_interleaved, k_sharded)
        ttnn.experimental.paged_update_cache(k_cache, k_sharded, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_cache, v_sharded, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.deallocate(k_sharded)
        ttnn.deallocate(v_sharded)

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
        if int(concat.shape[2]) != self.max_batch:
            trimmed = ttnn.slice(concat, [0, 0, 0, 0], [1, 1, self.max_batch, s.num_attention_heads * s.head_dim])
            _free(concat, trimmed)
            concat = trimmed
        concat = ttnn.to_memory_config(concat, ttnn.DRAM_MEMORY_CONFIG)
        return self._attn_epilogue(concat, gate)

    # ----------------------------------------------------- linear attention

    def _gdn_inputs(self, x, *, raw_beta: bool = False, phase: str = "prefill"):
        """Shared GatedDeltaNet input projections, with ``b`` and ``a`` packed into one matmul.

        ``raw_beta`` returns ``b`` itself instead of ``sigmoid(b)``, for the decode path, which
        carries the sigmoid on the multiply that consumes it (see :meth:`_linear_attention_decode`).
        The prefill path cannot: ``beta`` is an *input* of ``chunk_gated_delta_rule``, and an op
        input has no activation slot to ride on.
        """
        s = self.shapes
        mixed_qkv = ttnn.linear(x, self.w["in_proj_qkv"], dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)
        z = ttnn.linear(x, self.w["in_proj_z"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        ab = ttnn.linear(
            x,
            self.w["in_proj_ab"],
            bias=self.w["in_proj_ab_bias"],
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_cfg,
            core_grid=self.ab_matmul_grid[phase],
        )
        lead = _shape(ab)[:-1]
        starts = [0] * len(lead)
        b = ttnn.slice(ab, [*starts, 0], [*lead, s.num_v_heads])
        a = ttnn.slice(ab, [*starts, _AB_STRIDE], [*lead, _AB_STRIDE + s.num_v_heads])
        ttnn.deallocate(ab)
        if raw_beta:
            beta = b
        else:
            beta = ttnn.sigmoid(b)
            ttnn.deallocate(b)
        # ``a`` already carries ``dt_bias``: it is the packed matmul's bias row.
        soft = ttnn.softplus(a, beta=1.0, threshold=20.0)
        ttnn.deallocate(a)
        g = ttnn.multiply(soft, self.w["neg_exp_A"])
        ttnn.deallocate(soft)
        return mixed_qkv, z, beta, g

    def _causal_conv(self, mixed_qkv, prefix, logical: int):
        """Depthwise causal conv1d (width ``K``) + SiLU over a prefill chunk, in bfloat16.

        Same FIR as the functional layer, with two changes:

        * The **arithmetic** runs in bfloat16.  Each tap is a *height-broadcast* binary op, and
          on this checkout a float32 height-broadcast multiply reaches a small fraction of the
          bandwidth the same op gets on same-shape float32 operands, while the bfloat16 broadcast
          does not - the measured table is ``work_log.md`` section 3.7, generated from
          ``doc/fused_decoder/probes/probe_causal_conv.py``'s log; the whole FIR is about 3x faster in
          bfloat16 than in float32 at 2048 tokens, for a conv-output PCC of 0.999990, and
          ``work_log.md`` section 3.7 carries the generated table.  Nothing downstream can use the extra precision either - the conv output
          feeds ``chunk_gated_delta_rule``, whose contract casts q/k/v to bfloat16 - and the
          checkpoint stores ``conv1d.weight`` in bfloat16 to begin with.
        * The carried conv **state** stays float32 and is taken from the float32 inputs, not
          from the bfloat16 window, so the recurrence's carried precision is unchanged.

        ``mixed_qkv``: ``[1, 1, L, conv_dim]`` float32; ``prefix``: ``[1, 1, K-1, conv_dim]``
        float32 left context.  Returns ``(activations [1, 1, L, conv_dim] bfloat16, new_state
        [1, 1, K, conv_dim] float32)``.
        """
        s = self.shapes
        k = s.conv_kernel_size

        length = int(mixed_qkv.shape[-2])

        # State: the K raw rows of the conceptual window ending at the last *logical* token.
        # Row r of that window is prefix[r] for r < K-1 and mixed_qkv[r - (K - 1)] otherwise, and
        # the rows wanted are logical-1 .. logical+K-2 - which are always the last K rows of
        # ``prefix ++ mixed_qkv[max(0, logical-K) : logical]``, for every logical >= 1.
        #
        # Cutting those K rows straight out of ``mixed_qkv`` costs an untilize of the *whole*
        # tensor - it showed up as a whole-tensor untilize in the profile - because neither end is on
        # a tile boundary, so
        # take a tile-aligned two-tile block around them first - that slice is a plain tile copy -
        # and do the ragged cut inside the block.  ``length`` is the padded chunk length, always a
        # multiple of the tile height, so both block ends are tile-aligned.
        # ...and the ragged cut, the concat with the left context and the final K-row cut all
        # happen in ROW_MAJOR, so the block is untilized once and the K-row result tilized once.
        # Doing them on TILE tensors instead makes every one of them its own untilize/tilize
        # sandwich, and leaves a tilize immediately undone by the next op - which is what
        # ``tests/test_fused_decoder_docs.py::test_no_layout_round_trip_in_the_measured_pass``
        # reads out of the committed report.
        tile = ttnn.TILE_SIZE
        block_start = max(0, ((logical - k) // tile) * tile)
        block_end = min(length, block_start + 2 * tile)
        block = ttnn.slice(mixed_qkv, [0, 0, block_start, 0], [1, 1, block_end, s.conv_dim])
        block_rows = ttnn.to_layout(block, ttnn.ROW_MAJOR_LAYOUT)
        # A full-range ttnn.slice returns a view, so ``block`` can alias ``mixed_qkv`` on a short
        # chunk; ``_free`` has to see every tensor that is still live.
        _free(block, block_rows, mixed_qkv)
        tail = ttnn.slice(
            block_rows,
            [0, 0, max(0, logical - k) - block_start, 0],
            [1, 1, logical - block_start, s.conv_dim],
        )
        _free(block_rows, tail, mixed_qkv)
        prefix_rows = ttnn.to_layout(prefix, ttnn.ROW_MAJOR_LAYOUT)
        merged = ttnn.concat([prefix_rows, tail], dim=-2)
        _free(prefix_rows, merged, prefix)
        _free(tail, merged, mixed_qkv)
        rows_in = int(merged.shape[-2])
        state_rows = ttnn.slice(merged, [0, 0, rows_in - k, 0], [1, 1, rows_in, s.conv_dim])
        _free(merged, state_rows)
        new_state = ttnn.to_layout(state_rows, ttnn.TILE_LAYOUT)
        _free(state_rows, new_state)

        taps = self.w["conv_taps_bf16"]
        # Build the window in ROW_MAJOR and take each tap's one-row-shifted view there, where a row
        # range is contiguous, instead of letting every TILE slice pay its own untilize+tilize
        # pair.  Concatenating in ROW_MAJOR matters as much as slicing there: ``ttnn.concat`` on
        # TILE operands untilizes them, concatenates, and re-tilizes - and the tap loop would throw
        # that tilize straight away again, so the TILE concat was a tilize/untilize round trip over
        # the whole ~42 MB window.  Measured over 12 repeats at 2048 tokens
        # (``doc/fused_decoder/logs/probe_causal_conv.log``, which reports median and stdev):
        # faster than either all-TILE form by several times the run-to-run spread, and
        # bit-identical to them.  ``work_log.md`` section 3.7 has the table.
        prefix_bf16 = ttnn.typecast(prefix, ttnn.bfloat16)
        prefix_rows = ttnn.to_layout(prefix_bf16, ttnn.ROW_MAJOR_LAYOUT)
        _free(prefix_bf16, prefix_rows, prefix)
        input_bf16 = ttnn.typecast(mixed_qkv, ttnn.bfloat16)
        input_rows = ttnn.to_layout(input_bf16, ttnn.ROW_MAJOR_LAYOUT)
        _free(input_bf16, input_rows, mixed_qkv)
        rows = ttnn.concat([prefix_rows, input_rows], dim=-2)
        ttnn.deallocate(prefix_rows)
        ttnn.deallocate(input_rows)
        acc = None
        for j in range(k):
            piece = ttnn.slice(rows, [0, 0, j, 0], [1, 1, j + length, s.conv_dim])
            tap = ttnn.to_layout(piece, ttnn.TILE_LAYOUT)
            ttnn.deallocate(piece)
            if acc is None:
                acc = ttnn.multiply(tap, taps[j])
                ttnn.deallocate(tap)
                continue
            if j != k - 1:
                # ``acc + window_j * w_j`` in one op.  Only the *last* tap's add carries the SiLU,
                # so only it needs to stay a separate multiply and add - §3.23.
                merged = ttnn.addcmul(acc, tap, taps[j])
                ttnn.deallocate(tap)
                ttnn.deallocate(acc)
                acc = merged
                continue
            term = ttnn.multiply(tap, taps[j])
            ttnn.deallocate(tap)
            # The FIR's trailing SiLU is an *output* activation of the last tap's add, not a
            # separate pass over a ~42 MB tensor: folding it there removed a standalone
            # ``UnaryDeviceOperation`` from the 2048-token prefill entirely.  Same merge as the MLP's SiLU,
            # the output gate's sigmoid and the z-gate's SiLU, and the same one the in-tree
            # reference makes (``ttnn_gated_deltanet.py``'s ``ttnn.add(out, bias, activations=...)``).
            merged = ttnn.add(acc, term, activations=[ttnn.UnaryOpType.SILU] if j == k - 1 else [])
            ttnn.deallocate(term)
            ttnn.deallocate(acc)
            acc = merged
        ttnn.deallocate(rows)
        return acc, new_state

    def _gated_norm_and_project(self, core, z, rows: int, *, phase: str = "prefill"):
        """z-gated per-head RMS norm + ``out_proj``, on the flat token-major layout.

        ``core`` is ``[1, 1, rows, value_dim]`` (all ``num_v_heads`` heads laid out along the
        last axis) and ``z`` is the same shape, which is what both the delta-rule output and the
        ``in_proj_z`` matmul naturally produce.  Doing the per-head norm with ``ttnn.rms_norm``
        would need the tensor reshaped to ``[1, rows, num_v_heads, head_v_dim]`` and back, and in
        TILE layout each of those is a full relayout, and in the stage-1 profile they were the two most
        expensive ops of the whole pass.  The identical arithmetic as a group reduction is two skinny matmuls
        against constant matrices (:data:`_GDN_GROUP_PAD`-wide), with the ``1/head_v_dim`` mean
        factor and the norm weight folded into them, and no relayout at all.
        """
        s = self.shapes
        squares = ttnn.multiply(core, core)
        mean_square = ttnn.matmul(
            squares,
            self.const["gdn_group_mean"],
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_cfg,
            core_grid=self.group_sum_grid[phase],
        )
        ttnn.deallocate(squares)
        # rsqrt is the add's *output* activation, not an op after it.
        inv = ttnn.add(mean_square, s.rms_norm_eps, activations=[ttnn.UnaryOpType.RSQRT])
        ttnn.deallocate(mean_square)
        inv16 = ttnn.typecast(inv, ttnn.bfloat16)
        _free(inv, inv16)
        scale = ttnn.matmul(
            inv16,
            self.const["gdn_scale_expand"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_cfg,
            core_grid=self.group_expand_grid[phase],
        )
        ttnn.deallocate(inv16)
        normed = ttnn.multiply(core, scale)
        ttnn.deallocate(scale)
        gated = ttnn.multiply(normed, z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(normed)
        assert int(gated.shape[-2]) == rows and int(gated.shape[-1]) == s.value_dim
        out = ttnn.linear(gated, self.w["out_proj"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        ttnn.deallocate(gated)
        return out

    def _linear_attention_prefill_chunk(self, x, *, state, conv_prefix, length):
        """One prefill chunk of the gated delta rule, as a single dedicated op.

        ``x``: ``[1, 1, L, hidden]`` (``L`` may exceed ``length`` after zero padding).
        ``state``: ``[1, num_v_heads, head_k_dim, head_v_dim]`` float32 recurrent state.
        Returns ``(out [1, 1, L, value_dim], new_state, new_conv_state)``.
        """
        s = self.shapes
        mixed_qkv, z, beta, g = self._gdn_inputs(x)
        conv_out, new_conv_state = self._causal_conv(mixed_qkv, conv_prefix, length)
        ttnn.deallocate(mixed_qkv)

        padded = int(x.shape[-2])
        assert (
            padded % FUSED_DELTA_CHUNK == 0
        ), f"prefill chunk length {padded} must be a multiple of the fused delta chunk {FUSED_DELTA_CHUNK}"
        # The hidden states were zero-padded, not q/k/v/beta/g, so a padded row still produces
        # beta = sigmoid(0) = 0.5 and a non-zero g.  Zeroing both makes every padded position an
        # exact identity update of the recurrent state, which is what HF's own zero padding does.
        beta = _zero_after_seq(beta, length, padded)
        g = _zero_after_seq(g, length, padded)

        q_flat, k_flat, v_flat = self._split_qkv(conv_out)
        ttnn.deallocate(conv_out)
        # Flat token-major rank-3 inputs: the op does the head split, the Q/K L2 norm, the
        # 1/sqrt(head_k_dim) scale and the GQA head expansion itself.
        q3 = ttnn.reshape(q_flat, (1, padded, s.key_dim))
        k3 = ttnn.reshape(k_flat, (1, padded, s.key_dim))
        v3 = ttnn.reshape(v_flat, (1, padded, s.value_dim))
        _free(q_flat, q3)
        _free(k_flat, k3)
        _free(v_flat, v3)
        beta3 = ttnn.reshape(beta, (1, padded, s.num_v_heads))
        g3 = ttnn.reshape(g, (1, padded, s.num_v_heads))
        _free(beta, beta3)
        _free(g, g3)

        core, new_state = ttnn.transformer.chunk_gated_delta_rule(
            q3,
            k3,
            v3,
            g3,
            beta3,
            initial_state=state,
            output_final_state=True,
            chunk_size=FUSED_DELTA_CHUNK,
            eye=self.const["gdn_eye"],
            tril=self.const["gdn_tril"],
            ones=self.const["gdn_ones"],
            masks=self.const["gdn_masks"],
            compute_kernel_config=self.compute_cfg,
        )
        for tensor in (q3, k3, v3, beta3, g3):
            ttnn.deallocate(tensor)
        ttnn.deallocate(state)

        # ``core`` is token-major ROW_MAJOR [1, L, num_v_heads, head_v_dim].  Merging the trailing
        # two axes is contiguous in ROW_MAJOR.  That is still a real op in the report - a
        # ROW_MAJOR page-size change is a copy - but it is one copy instead of the two full TILE
        # [L, H*D] <-> [L, H, D] relayouts it replaces, and the tilize that follows lands on
        # tile-aligned dims with no head-axis padding.  Both costs are rows of the committed
        # prefill report; work_log.md section 3.4 has the comparison.
        #
        # ``output_head_major=True`` would skip the op's own untilize+permute epilogue, but the
        # consumer chain (per-head norm, z gate, out_proj) is token-major-flat, so it buys the
        # epilogue back as relayouts of ``z`` and of the gated result - measured about 2x this
        # path's cost at PCC 0.999994 between the two, in
        # ``doc/fused_decoder/probes/probe_output_paths.py``; the numbers are work_log.md
        # section 3.13, generated from that probe's log.
        flat = ttnn.reshape(core, (1, 1, padded, s.value_dim))
        # ``ttnn.tilize`` takes the output dtype, so the tilize and the float32 -> bfloat16 cast
        # the group-reduction norm wants are one op rather than two passes over the same tensor.
        tiled = ttnn.tilize(flat, dtype=ttnn.bfloat16)
        _free(flat, tiled)
        _free(core, tiled)
        core = tiled
        out = self._gated_norm_and_project(core, z, padded)
        ttnn.deallocate(core)
        ttnn.deallocate(z)
        return out, new_state, new_conv_state

    def _linear_attention_decode(self, x):
        """Single-token gated delta rule for all ``max_batch`` users at once."""
        s = self.shapes
        batch = self.max_batch
        nv = s.num_v_heads
        k_size = s.conv_kernel_size
        # §3.25: float32, always - the bfloat16 form is faster and loses PCC.
        bf16_fir = _DECODE_CONV_BF16_BATCH is not None and batch >= _DECODE_CONV_BF16_BATCH
        taps = self.w["conv_taps_bf16"] if bf16_fir else self.w["conv_taps"]

        # ``raw_beta``: the sigmoid rides on the ``delta`` multiply below instead of being its own
        # op, which the prefill path cannot do because ``beta`` is an op *input* there.
        mixed_qkv, z, b_raw, g = self._gdn_inputs(x, raw_beta=True, phase="decode")  # [1, 1, batch, *]

        # Depthwise causal conv over the batch-major tap buffers: tap j reads a whole buffer, the
        # newest tap reads this token, and the shift is an in-place copy chain.
        token = ttnn.typecast(mixed_qkv, ttnn.bfloat16) if bf16_fir else mixed_qkv
        acc = ttnn.multiply(token, taps[k_size - 1])
        if bf16_fir:
            ttnn.deallocate(token)
        cast_rows = []
        for j in range(k_size - 1):
            # Buffer ``j`` is packed row ``j``; the FIR's tap ``j`` reads the *previous* tokens,
            # which are rows 1..K-1.  Row 0 is the token that falls out of the window this step,
            # and it is kept only so :meth:`current_conv_state` can rebuild the packed state the
            # functional layer maintains - see that method.
            row = self.conv_state_split[j + 1]
            if bf16_fir:
                # Cast a *copy* of the carried row; the buffer itself stays float32, so the
                # recurrence's carried precision is unchanged (§3.25).
                row = ttnn.typecast(row, ttnn.bfloat16)
                cast_rows.append(row)
            if j == k_size - 2:
                # The last tap keeps its own multiply, because the SiLU rides on this add and
                # ``addcmul`` has no activation slot (§3.14).
                term = ttnn.multiply(row, taps[j])
                merged = ttnn.add(acc, term, activations=[ttnn.UnaryOpType.SILU])
                ttnn.deallocate(term)
            else:
                # ``acc + state * tap`` in one op.  Nothing rides on these adds, so §3.14's
                # blocker does not apply to them - §3.23.
                merged = ttnn.addcmul(acc, row, taps[j])
            ttnn.deallocate(acc)
            acc = merged
        for row in cast_rows:
            ttnn.deallocate(row)
        for j in range(k_size - 1):
            ttnn.copy(self.conv_state_split[j + 1], self.conv_state_split[j])
        ttnn.copy(mixed_qkv, self.conv_state_split[k_size - 1])
        ttnn.deallocate(mixed_qkv)
        conv_out = acc

        q_flat, k_flat, v_flat = self._split_qkv(conv_out)
        ttnn.deallocate(conv_out)

        def to_heads(flat, num_heads, head_dim, repeat: int, scale=None, dense: bool = False):
            """``dense`` keeps ``[1, batch, heads, dim]`` instead of one padded row per head.

            The three matmuls need the per-head-row layout - a per-head batch dimension is what
            makes them batched - but the *transients* around them do not, and in that layout a
            logical height of 1 is padded to a 32-row tile, so 31 of every 32 bytes moved are
            padding (§3.24).
            """
            t = ttnn.reshape(flat, (1, batch, num_heads, head_dim))
            if repeat > 1:
                rep = ttnn.repeat_interleave(t, repeat, dim=2)
                _free(t, flat, rep)
                t = rep
            if scale is not None:
                # L2 norm as one dedicated op instead of mul/sum/rsqrt/mul.  With
                # ``mean(x^2) = sum(x^2)/D``,
                #     rms_norm(x, eps/D) == x * sqrt(D) / sqrt(sum(x^2) + eps) == sqrt(D) * l2norm(x)
                # so ``l2norm(x) == rms_norm(x, eps/D) / sqrt(D)``, and the query's extra
                # ``1/sqrt(head_k_dim)`` folds into the same constant.  Done here, on
                # ``[1, batch, heads, head_dim]``, because the last axis is the norm axis and the
                # head axis is the tile height - the shape the norm kernel wants.
                normed = ttnn.rms_norm(t, epsilon=_L2NORM_EPS / head_dim, compute_kernel_config=self.compute_cfg)
                _free(t, flat, normed)
                t = ttnn.multiply(normed, scale)
                ttnn.deallocate(normed)
            if dense:
                return t
            return ttnn.reshape(t, (1, batch * nv, 1, head_dim))

        root_dk = math.sqrt(s.head_k_dim)
        q = to_heads(q_flat, s.num_k_heads, s.head_k_dim, s.v_per_k, scale=1.0 / (root_dk * root_dk))
        k = to_heads(k_flat, s.num_k_heads, s.head_k_dim, s.v_per_k, scale=1.0 / root_dk)
        v = to_heads(v_flat, nv, s.head_v_dim, 1, dense=True)
        _free(q_flat, q)
        _free(k_flat, k)
        _free(v_flat, v)

        # ``b`` and ``g`` stay dense - ``[1, batch, num_v_heads, 1]`` - because the ops that read
        # them are the dense transient chain below; only the decay the ``addcmul`` needs is turned
        # into per-head rows, and that is a ``[1, BH, 1, 1]`` tensor either way (§3.24).
        b_h = ttnn.reshape(b_raw, (1, batch, nv, 1))
        g_h = ttnn.reshape(g, (1, batch, nv, 1))
        _free(b_raw, b_h)
        _free(g, g_h)

        # The state update is ``state * exp(g) + update``, which is one ``ttnn.addcmul`` - a single
        # LLK ternary op on this checkout, not the composite an earlier round recorded (§3.21).  It
        # replaces a full-size multiply *and* a full-size add with one pass over the carried state:
        # at the advertised ``max_batch`` that state is 100 MB of float32, and the pair measured
        # about three fifths of the pair's time for the fused form, bit-exact in place
        # (the generated table is ``work_log.md`` section 3.21, from
        # ``doc/fused_decoder/logs/probe_addcmul_state.log``).
        #
        # Two consequences.  ``exp(g)`` becomes its own tiny op again instead of riding on the
        # multiply that no longer exists.  And the state read now consumes the *undecayed* state,
        # with the decay applied to its ``[1, BH, 1, head_v_dim]`` result instead: ``g`` is one
        # scalar per head, so ``k @ (state * g) == (k @ state) * g`` exactly, and the moved multiply
        # is four orders of magnitude smaller than the one it came from.
        decay = ttnn.exp(g_h)
        ttnn.deallocate(g_h)
        decay_rows = ttnn.reshape(decay, (1, batch * nv, 1, 1))
        kv_raw = ttnn.matmul(
            k,
            self.recurrent_state,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_cfg,
            core_grid=self.recurrence_read_grid,
        )
        # §3.24: the matmul hands back one padded row per head; the arithmetic that follows runs
        # dense, on a twenty-fourth of the bytes at the advertised batch, and only ``delta`` is
        # turned back into rows for the outer product.
        kv_dense = ttnn.reshape(kv_raw, (1, batch, nv, s.head_v_dim))
        _free(kv_raw, kv_dense)
        kv_mem = ttnn.multiply(kv_dense, decay)
        ttnn.deallocate(kv_dense)
        residual = ttnn.subtract(v, kv_mem)
        gated = ttnn.multiply(residual, b_h, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(residual)
        ttnn.deallocate(kv_mem)
        ttnn.deallocate(v)
        ttnn.deallocate(b_h)
        delta = ttnn.reshape(gated, (1, batch * nv, 1, s.head_v_dim))
        _free(gated, delta)
        # The outer product's transpose is an argument of the matmul, not an op before it - the
        # skill's "permute/transpose + matmul" merge.  Exact (PCC 1.000000 against torch) and one
        # dispatch fewer; measured in ``doc/fused_decoder/logs/probe_decode_recurrence.log``.
        update = ttnn.matmul(
            k,
            delta,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_cfg,
            core_grid=self.recurrence_outer_grid,
            transpose_a=True,
        )
        ttnn.deallocate(delta)
        ttnn.deallocate(k)
        # One pass: decay the carried state and add this step's update, straight into the
        # persistent buffer.  ``output_tensor`` aliasing an input is what the traced decode needs
        # (the state must land at the persistent address) and is bit-exact here - the probe checks
        # the in-place form against torch as well as against the two-op form.
        ttnn.addcmul(update, self.recurrent_state, decay_rows, output_tensor=self.recurrent_state)
        ttnn.deallocate(decay)
        if decay_rows.is_allocated():
            ttnn.deallocate(decay_rows)
        ttnn.deallocate(update)
        out = ttnn.matmul(
            q,
            self.recurrent_state,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_cfg,
            core_grid=self.recurrence_read_grid,
        )
        ttnn.deallocate(q)

        if batch >= _GATED_NORM_GROUP_BATCH:
            # Enough rows that the two tile relayouts cost more than the group reduction; see
            # :data:`_GATED_NORM_GROUP_BATCH`.  This is the prefill path, reused verbatim.
            flat = ttnn.reshape(out, (1, 1, batch, s.value_dim))
            _free(out, flat)
            core = ttnn.typecast(flat, ttnn.bfloat16)
            _free(flat, core)
            result = self._gated_norm_and_project(core, z, batch, phase="decode")
            ttnn.deallocate(core)
            ttnn.deallocate(z)
            return result

        core = ttnn.reshape(out, (1, batch, nv, s.head_v_dim))
        _free(out, core)
        z_heads = ttnn.reshape(z, (1, batch, nv, s.head_v_dim))
        core16 = ttnn.typecast(core, ttnn.bfloat16)
        normed = self._rms_norm(core16, self.w["gated_norm"])
        ttnn.deallocate(core16)
        ttnn.deallocate(core)
        gated = ttnn.multiply(normed, z_heads, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(normed)
        _free(z_heads, z)
        ttnn.deallocate(z)
        flat = ttnn.reshape(gated, (1, 1, batch, s.value_dim))
        _free(gated, flat)
        result = ttnn.linear(flat, self.w["out_proj"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        ttnn.deallocate(flat)
        return result

    # ------------------------------------------------------------- forwards

    def decode_forward(self, hidden_states, *, current_pos=None, page_table=None, rot_mats=None):
        s = self.shapes
        assert len(hidden_states.shape) == 4, f"decode expects [1, 1, batch, hidden]; got {hidden_states.shape}"
        assert (
            int(hidden_states.shape[2]) == self.max_batch
        ), f"decode batch {int(hidden_states.shape[2])} != max_batch {self.max_batch}"
        residual = hidden_states
        normed = self._decode_rms_norm(hidden_states, self.w["input_layernorm"])
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
        normed2 = self._decode_rms_norm(hidden, self.w["post_attention_layernorm"])
        mlp_out = self._mlp(normed2)
        ttnn.deallocate(normed2)
        out = ttnn.add(hidden, mlp_out)
        ttnn.deallocate(hidden)
        ttnn.deallocate(mlp_out)
        return out

    # ------------------------------------------------------------- helpers

    def prepare_decode_state(self) -> None:
        """Fold the per-user prefill state into the batch-wide decode buffers.

        Same contract and same limitation as the functional layer, plus the batch-major conv tap
        buffers the fused decode conv reads.
        """
        if self.shapes.layer_type != LINEAR_ATTENTION:
            return
        super().prepare_decode_state()
        s = self.shapes
        # conv_state is [1, batch, K, conv_dim]; tap buffer j is the whole batch's row j of it.
        # There is one buffer per packed row, not one per FIR tap, so the packed state the
        # functional layer maintains stays derivable after a decode step - :meth:`current_conv_state`.
        for j, buffer in enumerate(self.conv_state_split):
            rows = [
                ttnn.slice(self.user_conv_state[user], [0, 0, j, 0], [1, 1, j + 1, s.conv_dim])
                for user in range(self.max_batch)
            ]
            if len(rows) > 1:
                merged = ttnn.concat(rows, dim=2)
                for row in rows:
                    ttnn.deallocate(row)
            else:
                merged = rows[0]
            reshaped = ttnn.reshape(merged, (1, 1, self.max_batch, s.conv_dim))
            ttnn.copy(reshaped, buffer)
            _free(reshaped, merged)
            ttnn.deallocate(merged)

    def current_conv_state(self):
        """Fold the decode tap buffers back into the packed ``conv_state`` and return it.

        The functional layer rewrites ``conv_state`` on every decode step; the fused layer keeps
        the same state as ``max_batch``-wide per-row buffers instead, because a decode step then
        reads whole buffers rather than slicing a tile-height axis (§3.11).  Anything that wants
        the packed view - a serving stage inspecting state mid-generation, or
        ``test_conv_state_after_decode_matches_reference`` - calls this, which is exact: buffer
        ``j`` *is* packed row ``j``.

        Allocates (the concat needs a new buffer), so it is not callable inside a captured
        trace; call it between steps.
        """
        if self.shapes.layer_type != LINEAR_ATTENTION:
            return None
        s = self.shapes
        rows = [ttnn.reshape(buffer, (1, self.max_batch, 1, s.conv_dim)) for buffer in self.conv_state_split]
        packed = ttnn.concat(rows, dim=2)
        for row, buffer in zip(rows, self.conv_state_split):
            if row.buffer_address() != buffer.buffer_address():
                ttnn.deallocate(row)
        ttnn.copy(packed, self.conv_state)
        ttnn.deallocate(packed)
        return self.conv_state

    def released_tensors(self) -> list:
        """Every device tensor this layer owns, for the test harness to free."""
        tensors: list = []
        for value in self.w.values():
            tensors.extend(value if isinstance(value, list) else [value])
        tensors.extend(self.const.values())
        tensors.extend(self.kv_cache or ())
        for value in (self.conv_state, self.recurrent_state):
            if value is not None:
                tensors.append(value)
        tensors.extend(self.conv_state_split)
        tensors.extend(t for t in self.user_conv_state if t is not None)
        tensors.extend(t for t in self.user_recurrent_state if t is not None)
        return tensors

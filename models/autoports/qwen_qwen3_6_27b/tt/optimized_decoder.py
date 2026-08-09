# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Performance-optimized TTNN decoder for Qwen/Qwen3.6-27B (HF ``model_type: qwen3_5``).

This is the tuned sibling of :mod:`.fused_decoder`.  The graph topology it inherits from that
stage is already close to minimal, so this stage optimizes what a fusing pass does not touch:
**precision and math fidelity per tensor group, decode activation/weight sharding, matmul
program configs, and the gated-delta-net prefill kernel**.  The fused module stays as the
unoptimized reference the equivalence tests compare against.

Everything from the fused stage's ``F*`` catalogue is still here and still documented at its
call site.  What this stage adds is catalogued as ``O*`` and derived, measured and (where
rejected) refuted in ``doc/optimized_decoder/work_log.md``.

Contract
--------

Identical to :class:`~.fused_decoder.FusedDecoder`: same ``prefill_forward`` /
``decode_forward`` signatures, arbitrary ``1 <= seq_len <= max_seq_len`` with no divisibility
requirement, the same paged page-table protocol, the same per-user state,
``prepare_decode_state`` semantics, determinism, and the "no torch after ``from_state_dict``"
rule.  ``rot_mats`` are ``head_dim``-wide in the permuted channel order and the paged K cache
holds permuted head channels, exactly as the fused decoder documents; the gated-delta-net
state is stored in permuted value-head order.  :func:`rope_channel_permutation` and
:func:`value_head_permutation` expose both orders.

The one *new* knob is :class:`PrecisionPolicy`, the named per-tensor-group precision and
fidelity contract.  It is a constructor argument with a documented default, not a global
switch, so a caller can pin any group without touching the rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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

#: Number of DRAM banks a Blackhole chip exposes; the width a DRAM-sharded weight is cut into.
DRAM_BANKS = 8

#: Narrow-K grid for the batched delta-rule matmuls (``O10``).  48 cores for 48 value heads, so
#: each core owns one head's matmul.  Measured best for ``K = 64``; see
#: :meth:`OptimizedDecoder._batched_program_config` and ``logs/batched_matmul_sweep.log``.
BATCHED_MATMUL_GRID = (8, 6)

#: The whole Blackhole worker grid, measured best for the batched delta-rule matmuls with
#: ``K >= 128``.
FULL_WORKER_GRID = (11, 10)

#: ``(x, y)`` compute grid for the 2D prefill matmul program configs (``O6``).  Blackhole's
#: worker grid is 11x10; the reference Qwen3.6 implementation
#: (``models/demos/blackhole/qwen36/tt/tp_common.py:prefill_grid_default``) uses 8x10 and warns
#: that ``grid_x = 10`` garbles the regular matmul kernel, so the wide axis is y.
PREFILL_GRID = (8, 10)

#: Core range the DRAM-sharded weight memory configs use.  ``ttnn``'s DRAM-sharded matmul reads
#: the shard grid as *bank* coordinates, so this is one row of ``DRAM_BANKS`` cores, not a
#: worker-grid rectangle.
_DRAM_SHARD_GRID = ttnn.CoreRangeSet(
    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_BANKS - 1, 0))}
)


# ------------------------------------------------------------------------- topology


@dataclass(frozen=True)
class TopologyOptions:
    """Structural choices that are *not* precision (``O2``-``O5``).

    Every field here is something the candidate table in
    ``doc/optimized_decoder/work_log.md`` measured both ways.  They are options rather than
    hard-coded decisions so a rejected family can be re-measured on demand instead of being
    re-derived from prose.
    """

    name: str = "default"

    #: ``O2``: run the dominant decode projections as DRAM-sharded matmuls (weights width
    #: sharded across the DRAM banks, activations and outputs width-sharded in L1) rather than
    #: as interleaved matmuls with a ttnn-chosen program.
    dram_sharded_decode: bool = True

    #: ``O3``: carry the decode residual stream width-sharded in L1 across both norms, the
    #: mixer boundary and both residual adds instead of round-tripping it through DRAM.
    sharded_decode_residual: bool = True

    #: ``O4``: keep the MLP's gate and up projections as one packed matmul (the fused stage's
    #: choice) or as two separate matmuls.  Packed saves a launch; separate saves the two
    #: output slices and halves ``per_core_N``.
    pack_gate_up: bool = False

    #: ``O6``: 2D ``MatmulMultiCoreReuseMultiCast`` program configs for large prefill matmuls
    #: whose weight is **interleaved**.  This flag is only consulted in that case: a
    #: DRAM-width-sharded weight (which is what ``dram_sharded_decode`` produces, and the
    #: default) *requires* an explicit config, because ttnn's auto-selected prefill program
    #: rejects a sharded ``input_tensor_b`` outright.  It is off by default because the only
    #: configuration that reaches it is the ``dram_sharded_decode=False`` fallback, where it is
    #: measured separately - 50.51 -> 42.98 ms and 17.53 -> 10.32 ms of prefill, so it helps
    #: there too.  See work_log.md O6.
    prefill_program_configs: bool = False

    #: Target core count for the width-sharded decode activation grids.  ``0`` asks
    #: :func:`_find_grid` for its own default (half of an 8x8 grid).
    decode_cores: int = 0


DEFAULT_TOPOLOGY = TopologyOptions(name="default")


#: Ceiling used when sizing a prefill matmul's per-core circular buffers.  A Blackhole worker's
#: L1 is 1,572,864 B; this leaves a small margin.  The estimate below is calibrated against the
#: measured allocate/overflow boundary in ``logs/prefill_sweep.log``, which is tight: the
#: 5120x6144 projection allocates 1,523,712 B at ``out_block_h=7, in0_block_w=8`` and overflows
#: at 1,637,581 B one step up.
_PREFILL_CB_BUDGET = 1_565_000

#: Multiplier on the output block's tiles.  The kernel keeps more than the output block itself
#: alongside it (packer intermediate plus scratch under ``fp32_dest_acc_en``); 3.283 is the
#: exact value implied by the one overflow whose byte count ttnn reported - 1,630,976 B for
#: ``per_core_M=8, out_block_h=8, out_block_w=24, in0_block_w=4`` on a BFP8 weight - and 3.3
#: reproduces every observed allocate/overflow outcome in ``logs/prefill_sweep.log``.
_PREFILL_OUT_CB_FACTOR = 3.3

#: Bytes in one 32x32 tile of each dtype these buffers hold.  Block-float tiles carry one
#: shared exponent per 16-value face, hence the +64.
_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.float32: 4096,
    ttnn.bfloat8_b: 1024 + 64,
    ttnn.bfloat4_b: 512 + 64,
}


def _prefill_cb_bytes(per_core_M, out_block_h, out_block_w, in0_block_w, weight_dtype) -> int:
    """Per-core circular-buffer estimate for a 2D prefill matmul, in bytes.

    ``in0`` (bfloat16 activations) and ``in1`` (the weight, at its own tile size) are
    double-buffered; the output block is counted at :data:`_PREFILL_OUT_CB_FACTOR` because the
    kernel keeps a packer intermediate and scratch alongside it.
    """
    in0 = 2 * per_core_M * in0_block_w * _TILE_BYTES[ttnn.bfloat16]
    in1 = 2 * in0_block_w * out_block_w * _TILE_BYTES.get(weight_dtype, 2048)
    out = _PREFILL_OUT_CB_FACTOR * out_block_h * out_block_w * _TILE_BYTES[ttnn.bfloat16]
    return int(in0 + in1 + out)


def _largest_divisor(value: int, cap: int = 8) -> int:
    """Largest ``d <= cap`` dividing ``value`` (``in0_block_w`` selection)."""
    for d in range(min(cap, value), 0, -1):
        if value % d == 0:
            return d
    return 1


def _find_grid(n_tiles: int, target: int = 32, max_rows: int = 8, max_cols: int = 8):
    """``(rows, cols)`` whose core count divides ``n_tiles`` and is closest to ``target``.

    Same selection rule as ``models/common/modules/mlp/mlp_1d.py``: an activation shard grid
    whose core count divides the tiled K dimension is what lets the DRAM-sharded matmul pick a
    legal ``in0_block_w`` at all.
    """
    max_cores = max_rows * max_cols
    candidates = [c for c in range(1, max_cores + 1) if n_tiles % c == 0]
    candidates.sort(key=lambda c: (abs(c - target), -c))
    for cores in candidates:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0 and cores // rows <= max_cols:
                return rows, cores // rows
    raise AssertionError(f"no grid divides {n_tiles} tiles within {max_rows}x{max_cols}")


@dataclass(frozen=True)
class DecodeMatmulPlan:
    """Everything one DRAM-sharded decode projection needs (``O2``).

    ``weight_memory_config`` is applied at load time, the other two at every decode step.  The
    three have to agree: the DRAM-sharded matmul reads its weight straight out of the banks and
    its activation straight out of L1, so a mismatch is a validation error rather than a slow
    path.
    """

    key: str
    k: int
    n: int
    cores: int
    grid: object
    input_memory_config: object
    program_config: object
    weight_memory_config: object


def _dram_sharded_weight_config(k: int, n: int) -> ttnn.MemoryConfig:
    """WIDTH_SHARDED DRAM config for a ``[k, n]`` weight, one shard per DRAM bank."""
    stride = ttnn.TILE_SIZE * DRAM_BANKS
    padded_n = _round_up(n, stride)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(_DRAM_SHARD_GRID, (k, padded_n // DRAM_BANKS), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _width_sharded_l1(rows: int, width: int, grid) -> ttnn.MemoryConfig:
    """Width-sharded L1 config on a **rectangular** core grid (``O11``).

    ttnn's DRAM-sharded matmul does not use a rectangle.  For 32 cores on an 11x10 Blackhole it
    computes ``{0,0}-{10,1}`` plus ``{0,2}-{9,2}`` - row-wise across the whole device grid - and
    logs ``Mismatch between computed MemoryConfig ... Using computed config`` for the output
    config this layer asks for (555 times in one suite run).  Matching that core set was tried
    and is **not legal for this decoder**: the sharded layernorm that carries the residual
    rejects it outright,

        ``Sharded layernorm does not support non-rectangular core grids. The shard spec grid
        has 32 cores but its bounding box spans 33 cores (11 x 3).``
        (``layernorm_device_operation.cpp:187``)

    and a rectangular core count that is also row-wise-rectangular on an 11-wide grid would have
    to be a multiple of 11 or at most 11, while the count also has to divide the tiled K of every
    role (160 for the 5120-wide projections, 544 for the down projection) - and no such count
    exists.  So the rectangle stays, and the cost is the one ``ReshardDeviceOperation`` per decode
    step recorded in the boundary table of work_log.md section 4.
    """
    return ttnn.create_sharded_memory_config(
        shape=(rows, width // grid.num_cores),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _decode_matmul_plan(key: str, k: int, n: int, rows: int, target_cores: int) -> DecodeMatmulPlan:
    """Plan one decode projection: activation grid from K, program config from both."""
    grid_rows, grid_cols = _find_grid(k // ttnn.TILE_SIZE, target=target_cores or 32)
    cores = grid_rows * grid_cols
    grid = ttnn.CoreGrid(x=grid_cols, y=grid_rows)
    return DecodeMatmulPlan(
        key=key,
        k=k,
        n=n,
        cores=cores,
        grid=grid,
        input_memory_config=_width_sharded_l1(rows, k, grid),
        program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            # in0_block_w counts K tiles per core, so it is bounded by K / (tile * cores).
            in0_block_w=_largest_divisor(k // (ttnn.TILE_SIZE * cores), cap=k // (ttnn.TILE_SIZE * cores)),
            per_core_M=_round_up(rows, ttnn.TILE_SIZE) // ttnn.TILE_SIZE,
            per_core_N=math.ceil(n / (ttnn.TILE_SIZE * cores)),
            fused_activation=None,
        ),
        weight_memory_config=_dram_sharded_weight_config(k, n),
    )


def decode_matmul_plans(shapes: DecoderShapes, max_batch: int, topology: TopologyOptions) -> dict:
    """The DRAM-sharded decode projection plans for one layer kind (``O2``).

    Shared by ``from_state_dict`` (which needs the weight memory configs at load time) and the
    runtime (which needs the activation configs and program configs), so the two cannot drift.
    """
    if not topology.dram_sharded_decode:
        return {}
    rows = _round_up(max_batch, ttnn.TILE_SIZE)
    hidden, inter = shapes.hidden_size, shapes.intermediate_size
    plan = {}

    def add(key, k, n):
        plan[key] = _decode_matmul_plan(key, k, n, rows, topology.decode_cores)

    if topology.pack_gate_up:
        add("mlp_gate_up", hidden, 2 * inter)
    else:
        add("mlp_gate", hidden, inter)
        add("mlp_up", hidden, inter)
    add("mlp_down", inter, hidden)

    if shapes.layer_type == FULL_ATTENTION:
        add("wqkv", hidden, (shapes.num_attention_heads + 2 * shapes.num_key_value_heads) * shapes.head_dim)
        add("wgate", hidden, shapes.num_attention_heads * shapes.head_dim)
        add("o_proj", shapes.num_attention_heads * shapes.head_dim, hidden)
    else:
        add("in_proj_qkv", hidden, shapes.conv_dim)
        add("in_proj_z", hidden, shapes.value_dim)
        add("out_proj", shapes.value_dim, hidden)
    return plan


# --------------------------------------------------------------------------- precision


@dataclass(frozen=True)
class PrecisionPolicy:
    """Per-tensor-group weight dtype and math fidelity (``O1``).

    A decoder-wide dtype switch is not a precision policy: the delta-rule state recurrence
    needs float32 accumulation for reasons the functional stage measured (catastrophic
    cancellation in the triangular inverse), while the MLP's 356 MB gate/up projection is pure
    DRAM bandwidth and tolerates BFP4.  Each group is therefore named and set independently so
    a regression can be assigned to one of them.

    ``doc/optimized_decoder/work_log.md`` §2 has the per-group sweep this default came from.
    """

    name: str

    #: SwiGLU MLP.  ``gate_up`` is the single packed ``[hidden, 2 * intermediate]`` weight.
    mlp_gate_up: object = ttnn.bfloat4_b
    mlp_down: object = ttnn.bfloat8_b

    #: ``full_attention`` projections.  ``qkv`` is the packed Q|K|V weight, ``gate`` the
    #: per-head output gate half of ``q_proj``, ``out`` the output projection.
    attn_qkv: object = ttnn.bfloat8_b
    #: ``O14`` tried bfloat4_b here and it is **rejected**.  It is worth 6.7 us of every
    #: ``full_attention`` decode step and clears the bar on the suite's own decode-token draw
    #: (0.996131 at seq 17), but sweeping only the draw puts its worst at **0.994316** - below
    #: the 0.995 real-weight bar - where the shipped bfloat8_b reads 0.996181 on the same six
    #: draws.  work_log.md §23; the rule it fails is stated in §9.
    attn_gate: object = ttnn.bfloat8_b
    attn_out: object = ttnn.bfloat8_b

    #: ``linear_attention`` (gated delta net) projections.
    gdn_qkv: object = ttnn.bfloat8_b
    gdn_z: object = ttnn.bfloat8_b
    #: bfloat4_b here is worth 7.3 us of decode and clears the bar on every *single* real-weight
    #: measurement (0.998507 at 2048, 0.998054 at seq 17, 0.996838 at seq 743) - but over six
    #: decode-token draws at seq 743 its worst is 0.994194 and three of the six are below 0.995.
    #: An earlier revision rejected it by quoting ``attn_out`` numbers taken on
    #: ``full_attention``, where this weight does not exist; work_log.md §9 now has its own.
    gdn_out: object = ttnn.bfloat8_b
    #: The packed ``b|a`` projection feeds ``sigmoid``/``softplus`` gates of the state
    #: recurrence; it is 2.6 MB, so its dtype is an accuracy choice, not a bandwidth one.
    gdn_ba: object = ttnn.float32

    #: Paged K/V cache storage dtype (``full_attention`` only).
    kv_cache: object = ttnn.bfloat8_b

    #: ``O8``: dtype of the gated-delta-net causal-conv stage - the projection output, the four
    #: tap weights and the conv state.  At prefill these are ``[1, 1, L, conv_dim]`` tensors,
    #: 84 MB each in float32 at L = 2048, and the conv is three ``addcmul`` passes plus a
    #: multiply and a concat over them; that stage alone is 16 ms of a 42 ms prefill.  bfloat16
    #: halves every one of those tensors and measured 41.93 -> 32.16 ms of ``linear_attention``
    #: prefill for a real-weight prefill PCC of 0.999432 against 0.999460 - see work_log.md O8.
    #: The delta-rule *state* math downstream keeps float32 regardless; the conv output is cast
    #: back to it at the boundary in :meth:`OptimizedDecoder._causal_conv`.
    gdn_conv: object = ttnn.bfloat16

    #: Math fidelity of the weight matmuls above.  Independent of dtype: BFP8 does not imply
    #: HiFi2, and LoFi is legal (and usually faster) for both BFP8 and BFP4 weights.
    proj_fidelity: object = ttnn.MathFidelity.LoFi
    #: ``fp32_dest_acc_en`` for the weight matmuls.  Kept on: turning it off is worth 5.4-5.5 us
    #: of decode, 0.5 %, and costs real-weight accuracy that a single sweep row hides.  Over six
    #: decode-token draws at seq 743 on real weights it takes ``linear_attention``'s worst-case
    #: decode PCC from 0.996689 to 0.995077 - onto the 0.995 bar, which the shipped
    #: ``test_real_weight_pcc_at_disputed_lengths`` catches because it happens to draw the worst
    #: one.  It also puts two synthetic structural cases at 0.9789 against the 0.98 bar.
    #: work_log.md §9 has the per-draw table and both rejections.
    proj_fp32_acc: bool = True

    #: Fidelity of prefill / decode SDPA.
    sdpa_fidelity: object = ttnn.MathFidelity.HiFi2
    sdpa_fp32_acc: bool = True

    #: Activation dtype of the decode residual stream and the MLP/attention intermediates.
    activation: object = ttnn.bfloat16

    def with_(self, **changes) -> "PrecisionPolicy":
        """A copy with some groups overridden - the sweep harness's only mutation point."""
        return replace(self, **changes)


#: The policy the optimized decoder uses unless a caller passes another one.  Selected by the
#: sweep in ``doc/optimized_decoder/work_log.md`` §2 and matched against the reference Blackhole
#: Qwen3.6 implementation in ``models/demos/blackhole/qwen36`` (BFP4 gate/up, BFP8 down, BFP8
#: attention and gated-delta-net projections, LoFi throughout).
DEFAULT_PRECISION = PrecisionPolicy(name="default")

#: The fused decoder's precision, kept as a named policy so the optimized module can reproduce
#: the previous stage's numbers under the *same* code path for before/after comparison.
FUSED_BASELINE_PRECISION = PrecisionPolicy(
    name="fused_baseline",
    mlp_gate_up=ttnn.bfloat16,
    mlp_down=ttnn.bfloat16,
    attn_qkv=ttnn.bfloat16,
    attn_gate=ttnn.bfloat16,
    attn_out=ttnn.bfloat16,
    gdn_qkv=ttnn.bfloat16,
    gdn_z=ttnn.bfloat16,
    gdn_out=ttnn.bfloat16,
    gdn_ba=ttnn.float32,
    gdn_conv=ttnn.float32,
    kv_cache=ttnn.bfloat16,
    proj_fidelity=ttnn.MathFidelity.HiFi4,
    proj_fp32_acc=True,
    sdpa_fidelity=ttnn.MathFidelity.HiFi4,
    sdpa_fp32_acc=True,
)


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
#: ``[1, 1, 32, 5120]`` (``doc/fused_decoder/logs/probe_norm.log`` — the inherited comment cited a
#: ``probe_sharded_norm.log`` that the fused stage never wrote): the default interleaved kernel
#: runs the whole norm on **one** core at 103.8 us, while these width-sharded configurations take
#: 23.7 us (5x2), 30.8 us (8x4) and 31.9 us (8x5).  The interleaved kernel
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


def value_head_permutation(num_v_heads: int, num_k_heads: int) -> list[int]:
    """Value-head order that turns ``repeat_interleave`` into a plain ``concat`` (``F24``).

    The gated delta net has ``num_v_heads`` value heads sharing ``num_k_heads`` key heads, value
    head ``j`` using key head ``j // v_per_k``.  Matching them therefore needs
    ``repeat_interleave(q, v_per_k)`` — and ``ttnn.repeat_interleave`` is a **composite**:
    ``typecast to bf16 -> untilize -> concat -> tilize -> typecast back``
    (``repeat_interleave.cpp:34-68``), which both costs five dispatches and silently rounds the
    float32 ``q``/``k`` to bfloat16 and back.

    Re-ordering the value heads to ``j' -> v_per_k * (j' % num_k_heads) + j' // num_k_heads``
    makes value head ``j'`` use key head ``j' % num_k_heads`` instead, which is exactly what a
    plain ``ttnn.concat([q] * v_per_k, dim=head_axis)`` produces — one tile-resident copy, no
    layout change and no dtype round trip.  Returns ``perm`` with
    ``permuted[j'] == original[perm[j']]``; it is applied at load time to every weight indexed by
    a value head, so nothing moves at runtime.
    """
    per_key = num_v_heads // num_k_heads
    return [per_key * (j % num_k_heads) + j // num_k_heads for j in range(num_v_heads)]


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


def _compute_cfg(fidelity, fp32_dest_acc_en: bool = True) -> ttnn.WormholeComputeKernelConfig:
    """``WormholeComputeKernelConfig`` is the class ttnn exposes for Blackhole too."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )


def _hifi4(fp32_dest_acc_en: bool = True) -> ttnn.WormholeComputeKernelConfig:
    return _compute_cfg(ttnn.MathFidelity.HiFi4, fp32_dest_acc_en)


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


class OptimizedDecoder(LightweightModule):
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
        precision: PrecisionPolicy = DEFAULT_PRECISION,
        topology: TopologyOptions = DEFAULT_TOPOLOGY,
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
        self.precision = precision
        self.topology = topology
        self._decode_plans = decode_matmul_plans(shapes, max_batch, topology)
        #: Prefill program configs proven to allocate, keyed by (weight, M tiles).
        self._prefill_pc_cache: dict = {}
        #: ``O10`` program configs for the batched delta-rule matmuls, keyed by shape.  ``None``
        #: records a shape the reuse program refused, so it is tried once and not again.
        self._batched_pc_cache: dict = {}
        #: ``O1``: the state math of the gated delta rule keeps HiFi4 + fp32 accumulation.  The
        #: functional stage measured real catastrophic cancellation in the triangular inverse,
        #: and this path is a few percent of decode time, so there is nothing to win by
        #: lowering it.  Every *weight* matmul uses ``proj_cfg`` instead.
        self.compute_cfg = _hifi4(fp32_dest_acc_en=True)
        self.proj_cfg = _compute_cfg(precision.proj_fidelity, precision.proj_fp32_acc)
        self.sdpa_compute_cfg = _compute_cfg(precision.sdpa_fidelity, precision.sdpa_fp32_acc)
        self.decode_head_mem_cfg = None
        self.decode_rot_mem_cfg = None
        self.kv_channel_permutation = None
        #: F24's value-head order, or ``None`` for ``full_attention``.  The recurrent state is
        #: stored in this order; a reader comparing it against HF must invert the permutation.
        self.value_head_permutation = (
            None
            if shapes.layer_type == FULL_ATTENTION
            else value_head_permutation(shapes.num_v_heads, shapes.num_k_heads)
        )
        # F15 + ``O3``: the two full-width decode norms are width-sharded (the interleaved
        # layernorm kernel parallelises over tile *rows*, and a decode activation has exactly
        # one, so it lands on a single core).  What ``O3`` adds is that the shard is now the
        # *residual* shard: the same memory config carries the hidden stream across both norms,
        # the mixer boundary and both residual adds, so nothing round-trips through DRAM and the
        # projections consume the norm output directly.
        rows = _round_up(max_batch, ttnn.TILE_SIZE)
        self.decode_residual_mem_cfg = None
        self.decode_norm_mem_cfg = None
        self.decode_norm_prog_cfg = None
        residual_plan = self._decode_plans.get("mlp_gate") or self._decode_plans.get("mlp_gate_up")
        if topology.sharded_decode_residual and residual_plan is not None:
            self.decode_residual_mem_cfg = residual_plan.input_memory_config
            grid = residual_plan.grid
            block_w = shapes.hidden_size // residual_plan.cores // ttnn.TILE_SIZE
            self.decode_norm_mem_cfg = self.decode_residual_mem_cfg
            self.decode_norm_prog_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                subblock_w=next(w for w in (4, 2, 1) if block_w % w == 0),
                block_h=rows // ttnn.TILE_SIZE,
                block_w=block_w,
                inplace=False,
            )
        else:
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
        weight_dtype=None,
        cache_dtype=None,
        state_dtype=ttnn.float32,
        precision: PrecisionPolicy = DEFAULT_PRECISION,
        topology: TopologyOptions = DEFAULT_TOPOLOGY,
    ) -> "OptimizedDecoder":
        """Build the layer from an HF **submodule-relative** state dict.

        Identical signature and key set to
        :meth:`~.functional_decoder.FunctionalDecoder.from_state_dict`; the only place
        ``torch`` is used.  All the host-side folding the fused graph needs — the SwiGLU
        weight order, the RoPE channel permutation, the ``b|a`` weight concatenation and its
        ``dt_bias``, the L2-norm scale weights and the decay masks — happens here.
        """
        import torch  # setup-time only; never on the prefill/decode path

        # ``weight_dtype`` / ``cache_dtype`` stay in the signature so this class is a drop-in
        # for the fused decoder, but the *policy* is the real contract: passing either one
        # explicitly pins every weight / the cache to it and overrides the policy's group.
        if weight_dtype is not None:
            precision = precision.with_(
                mlp_gate_up=weight_dtype,
                mlp_down=weight_dtype,
                attn_qkv=weight_dtype,
                attn_gate=weight_dtype,
                attn_out=weight_dtype,
                gdn_qkv=weight_dtype,
                gdn_z=weight_dtype,
                gdn_out=weight_dtype,
            )
        if cache_dtype is not None:
            precision = precision.with_(kv_cache=cache_dtype)
        cache_dtype = precision.kv_cache

        shapes = decoder_shapes(hf_config, layer_idx)
        plans = decode_matmul_plans(shapes, max_batch, topology)
        max_seq_len = max_seq_len or shapes.max_position_embeddings
        if max_num_blocks is None:
            padded_context = _round_up(max_seq_len, _prefill_alignment(shapes.layer_type, block_size))
            max_num_blocks = max_batch * (padded_context // block_size)

        def _get(name: str) -> "torch.Tensor":
            if name not in state_dict:
                raise KeyError(f"missing weight {name!r} for layer {layer_idx} ({shapes.layer_type})")
            return state_dict[name].to(torch.float32)

        def _tt(tensor: "torch.Tensor", dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                tensor,
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        def _linear_w(tensor: "torch.Tensor", dtype=ttnn.bfloat16, key: Optional[str] = None):
            """torch ``nn.Linear`` weight ``[out, in]`` → ttnn ``[1, 1, in, out]``.

            ``O2``: a weight with a decode plan is stored **width-sharded across the DRAM
            banks**, which is what lets the decode matmul use the DRAM-sharded program.  The
            same tensor still serves prefill, where a 2D program config reads it as usual.
            """
            out_features, in_features = tensor.shape
            value = tensor.t().contiguous().reshape(1, 1, in_features, out_features)
            plan = plans.get(key) if key else None
            if plan is None:
                return _tt(value, dtype)
            assert (plan.k, plan.n) == (in_features, out_features), (
                f"decode plan for {key} is {plan.k}x{plan.n}, weight is {in_features}x{out_features}"
            )
            stride = ttnn.TILE_SIZE * DRAM_BANKS
            assert out_features % stride == 0, (
                f"{key}: a DRAM-sharded weight needs its output width ({out_features}) to be a "
                f"multiple of {stride}; pad it before sharding"
            )
            return ttnn.from_torch(
                value,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=plan.weight_memory_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        def _norm_w(tensor: "torch.Tensor", one_centred: bool, dtype=ttnn.bfloat16):
            """Qwen3.5 RMSNorm multiplies by ``1 + weight``; fold the +1 in at load time."""
            value = tensor + 1.0 if one_centred else tensor
            return _tt(value.reshape(1, 1, 1, -1), dtype)

        weights = {
            "input_layernorm": _norm_w(_get("input_layernorm.weight"), one_centred=True),
            "post_attention_layernorm": _norm_w(_get("post_attention_layernorm.weight"), one_centred=True),
        }

        # ``O4``: SwiGLU gate/up as one packed matmul or as two.  Packed is one launch and one
        # weight read of the same total bytes; separate removes the two output slices the packed
        # form needs and halves ``per_core_N``.  Measured both ways - see work_log.md O4.
        gate_w = _get("mlp.gate_proj.weight")
        up_w = _get("mlp.up_proj.weight")
        if topology.pack_gate_up:
            weights["mlp_gate_up"] = _linear_w(
                torch.cat([gate_w, up_w], dim=0), precision.mlp_gate_up, "mlp_gate_up"
            )
        else:
            weights["mlp_gate"] = _linear_w(gate_w, precision.mlp_gate_up, "mlp_gate")
            weights["mlp_up"] = _linear_w(up_w, precision.mlp_gate_up, "mlp_up")
        weights["mlp_down"] = _linear_w(_get("mlp.down_proj.weight"), precision.mlp_down, "mlp_down")

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
            weights["wqkv"] = _linear_w(torch.cat([q_only, k_w, v_w], dim=0), precision.attn_qkv, "wqkv")
            weights["wgate"] = _linear_w(gate_only, precision.attn_gate, "wgate")
            weights["o_proj"] = _linear_w(_get("self_attn.o_proj.weight"), precision.attn_out, "o_proj")
            weights["q_norm"] = _norm_w(_get("self_attn.q_norm.weight")[perm], one_centred=True)
            weights["k_norm"] = _norm_w(_get("self_attn.k_norm.weight")[perm], one_centred=True)
        else:
            nv, dk, dv = shapes.num_v_heads, shapes.head_k_dim, shapes.head_v_dim
            # F24: reorder the value heads so the key-head match is a concat, not an interleave.
            v_perm = torch.tensor(value_head_permutation(nv, shapes.num_k_heads), dtype=torch.long)

            def _perm_v_rows(tensor: "torch.Tensor", width: int) -> "torch.Tensor":
                """Permute the value-head blocks of a ``[nv * width, ...]`` leading axis."""
                rest = tensor.shape[1:]
                return tensor.reshape(nv, width, *rest)[v_perm].reshape(nv * width, *rest)

            qkv_w = _get("linear_attn.in_proj_qkv.weight")
            v_start = 2 * shapes.key_dim
            qkv_w = torch.cat([qkv_w[:v_start], _perm_v_rows(qkv_w[v_start:], dv)], dim=0)
            weights["in_proj_qkv"] = _linear_w(qkv_w, precision.gdn_qkv, "in_proj_qkv")
            weights["in_proj_z"] = _linear_w(
                _perm_v_rows(_get("linear_attn.in_proj_z.weight"), dv), precision.gdn_z, "in_proj_z"
            )
            # out_proj consumes the value dim, so its *columns* carry the permutation.
            out_w = _get("linear_attn.out_proj.weight")
            weights["out_proj"] = _linear_w(
                _perm_v_rows(out_w.t().contiguous(), dv).t().contiguous(), precision.gdn_out, "out_proj"
            )
            weights["gated_norm"] = _norm_w(_get("linear_attn.norm.weight"), one_centred=False)

            # F5 + F11: b and a share their LHS, so one matmul emits both, with dt_bias folded
            # in as the bias of the ``a`` half.  The ``b`` half is padded up to a tile so both
            # output slices start on a tile boundary.
            b_w = _perm_v_rows(_get("linear_attn.in_proj_b.weight"), 1)
            a_w = _perm_v_rows(_get("linear_attn.in_proj_a.weight"), 1)
            pad = _round_up(nv, ttnn.TILE_SIZE) - nv
            hidden = b_w.shape[1]
            weights["in_proj_ba"] = _linear_w(
                torch.cat([b_w, torch.zeros(pad, hidden), a_w], dim=0), precision.gdn_ba
            )
            weights["ba_bias"] = _tt(
                torch.cat([torch.zeros(nv + pad), _get("linear_attn.dt_bias")[v_perm]]).reshape(1, 1, 1, -1),
                ttnn.float32,
            )

            # conv1d weight [conv_dim, 1, K] → K taps of shape [1, 1, 1, conv_dim]
            conv_w = _get("linear_attn.conv1d.weight").squeeze(1)
            conv_w = torch.cat([conv_w[:v_start], _perm_v_rows(conv_w[v_start:], dv)], dim=0)
            weights["conv_taps"] = [
                _tt(conv_w[:, j].reshape(1, 1, 1, -1), precision.gdn_conv)
                for j in range(shapes.conv_kernel_size)
            ]

            weights["neg_exp_A"] = _tt(
                (-torch.exp(_get("linear_attn.A_log")))[v_perm].reshape(1, 1, 1, -1), ttnn.float32
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
                torch.zeros(1, 1, shapes.conv_kernel_size - 1, shapes.conv_dim), precision.gdn_conv
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
            # F8: the decode conv window is `conv_kernel_size - 1` history rows plus the current
            # token.  Holding them as separate `[1, 1, max_batch, conv_dim]` buffers (batch on the
            # tile height, one buffer per tap) means a decode step reads each tap directly, with
            # no slice at a non-tile-aligned row and therefore no untilize/retilize round trip -
            # which is what a single `[1, batch, K, conv_dim]` buffer cost (measured 15 us of
            # untilize+tilize per tap per step).  Separate buffers also keep stable addresses for
            # trace replay.
            conv_state = [
                _tt(torch.zeros(1, 1, max_batch, shapes.conv_dim), precision.gdn_conv)
                for _ in range(shapes.conv_kernel_size - 1)
            ]
            recurrent_state = _tt(
                torch.zeros(1, max_batch * shapes.num_v_heads, shapes.head_k_dim, shapes.head_v_dim),
                state_dtype,
            )
            user_conv_state = [
                _tt(torch.zeros(1, 1, shapes.conv_kernel_size, shapes.conv_dim), precision.gdn_conv)
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
            precision=precision,
            topology=topology,
        )

    # ------------------------------------------------------------- primitives

    def decode_matmul_plans(self) -> dict:
        """The DRAM-sharded decode projection plans this layer was built with (``O2``).

        Public because it is the only honest way for a test to assert that the optimization
        actually reached the device: a program config is not visible in a ``ttnn.graph`` capture
        and a policy object proves nothing about the tensors.
        """
        return dict(self._decode_plans)

    def _prefill_program_configs(self, m: int, k: int, weight) -> list:
        """2D matmul program configs for a prefill matmul, best first (``O6``).

        Two reasons this exists.  The first is the ordinary one: prefill matmuls are
        compute-bound with large M and N, which is the ``MatmulMultiCoreReuseMultiCast`` case,
        and an explicit config beats the ttnn-chosen program by 1.7-2.6x on every role here
        (``logs/prefill_sweep.log``).

        The second is a hard requirement rather than a tuning choice.  ``O2`` stores the
        dominant weights **width-sharded across the DRAM banks** so decode can use the
        DRAM-sharded matmul, and ttnn's auto-selected prefill program rejects a sharded
        ``input_tensor_b`` outright (``matmul_device_operation.cpp:1805``).  The 2D multicast
        program does accept one, but only when ``per_core_N`` equals the weight's DRAM shard
        width in tiles and the shard grid is a single row - so the grid's x extent is pinned to
        :data:`DRAM_BANKS` and ``per_core_N`` is read off the shard spec.

        ``out_block_w`` is then pinned to ``per_core_N``.  That is a **correctness** constraint:
        with a DRAM-width-sharded weight and an interleaved output, ttnn accepts
        ``out_block_w < per_core_N`` and silently computes the wrong answer - PCC 0.240 for the
        5120x17408 gate projection and 0.496 for 5120x8192, against 0.9995 for the pinned form
        (``probes/probe_matmul_correctness.py``).  The guard in
        ``matmul_device_operation.cpp:1659`` only runs when the *output* is sharded, so nothing
        rejects the bad configuration.  See work_log.md O6.

        What is left to choose is the grid's y extent, ``out_block_h`` and ``in0_block_w``,
        bounded by the per-core circular buffers.  Candidates are ordered by
        ``out_block_h * in0_block_w`` - the tiles of work per buffer refill - with ties broken
        toward the larger ``out_block_h``.  That order puts the measured optimum first for the
        gate (1620 us at 8x8/obh 2/in0_block_w 4), the down projection (1221 us at
        8x10/obh 7/in0_block_w 8) and QKV (764 us at 8x8/obh 4/in0_block_w 4).

        The list is returned rather than a single choice because the L1 bound is **not
        predictable from the shapes**.  A model calibrated on the exact byte counts ttnn reports
        for full-length chunks does not extrapolate to short ones: a 64-row chunk allocates
        1,585,920 B where the same arithmetic predicts 1,119,027 B, so the kernel's circular
        buffer composition changes with ``per_core_M``.  :meth:`_linear` therefore tries the
        candidates in order and remembers the first that allocates, which is a compile-time
        search rather than a calibration that can silently go stale.
        """
        memory_config = weight.memory_config()
        sharded = memory_config.is_sharded()
        if not sharded and not self.topology.prefill_program_configs:
            return []
        n = int(weight.shape[-1])
        grid_x = DRAM_BANKS if sharded else PREFILL_GRID[0]
        if sharded:
            per_core_N = memory_config.shard_spec.shape[1] // ttnn.TILE_SIZE
        else:
            per_core_N = max(1, math.ceil(n / ttnn.TILE_SIZE / grid_x))
        m_tiles = math.ceil(m / ttnn.TILE_SIZE)
        k_tiles = math.ceil(k / ttnn.TILE_SIZE)
        out_subblock_w = next((w for w in (4, 3, 2, 1) if per_core_N % w == 0), 1)

        candidates = []
        for grid_y in (10, 8, 5, 4, 2, 1):
            per_core_M = math.ceil(m_tiles / grid_y)
            if grid_y > 1 and per_core_M * (grid_y - 1) >= m_tiles:
                continue  # a row of the grid would have no work at all
            for out_block_h in (d for d in range(per_core_M, 0, -1) if per_core_M % d == 0):
                for in0_block_w in (8, 4, 2, 1):
                    if k_tiles % in0_block_w:
                        continue
                    estimate = _prefill_cb_bytes(
                        per_core_M, out_block_h, per_core_N, in0_block_w, weight.dtype
                    )
                    candidates.append(
                        (
                            (out_block_h * in0_block_w, out_block_h, -estimate),
                            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                                compute_with_storage_grid_size=(grid_x, grid_y),
                                in0_block_w=in0_block_w,
                                out_subblock_h=1,
                                out_subblock_w=out_subblock_w,
                                out_block_h=out_block_h,
                                out_block_w=per_core_N,
                                per_core_M=per_core_M,
                                per_core_N=per_core_N,
                                transpose_mcast=False,
                                fused_activation=None,
                                fuse_batch=False,
                            ),
                        )
                    )
        # Anything whose estimate is already over the budget goes to the back of the queue
        # rather than being dropped: the estimate is a good ordering heuristic and a poor
        # predicate, and a shape where *nothing* fits the estimate still has to run.
        candidates.sort(key=lambda item: (-item[0][2] <= _PREFILL_CB_BUDGET, item[0]), reverse=True)
        return [config for _key, config in candidates]

    def _prefill_linear(self, x, weight, key: str, *, dtype, bias=None, out_memory_config=None):
        """Run a prefill matmul under the best program config that actually allocates.

        The candidate order is the measured one (see :meth:`_prefill_program_configs`); the first
        call for a given ``(weight, M tiles)`` searches down it and the result is cached, so this
        is a compile-time search, not per-call work.  It exists because the kernel's per-core
        circular-buffer footprint is not predictable from the shapes across the whole range this
        decoder supports - 1 to 2048 tokens, no divisibility requirement, and any weight dtype a
        caller pins through ``from_state_dict`` - and a config that overflows L1 raises rather
        than degrading.

        There are **two** failure modes and both are handled.  One is shape-only
        (``circular buffers ... grow to N B which is beyond max L1 size``) and is deterministic.
        The other depends on what else is live in L1 at that moment
        (``circular buffers in program N clash with L1 buffers on core range ...``), so a config
        that allocated once can fail later - which is why the cached config is re-searched on
        failure instead of being trusted forever.  A BF16 weight policy at short chunk lengths
        reaches the second one, and did before this loop was written that way.
        """
        cache_key = (key, math.ceil(int(x.shape[-2]) / ttnn.TILE_SIZE))

        def attempt(config):
            return ttnn.linear(
                x, weight, bias=bias, dtype=dtype, compute_kernel_config=self.proj_cfg,
                program_config=config, memory_config=out_memory_config,
            )

        cached = self._prefill_pc_cache.get(cache_key)
        if cached is not None:
            try:
                return attempt(cached)
            except RuntimeError:
                self._prefill_pc_cache.pop(cache_key, None)  # L1 occupancy moved under it

        candidates = self._prefill_program_configs(int(x.shape[-2]), int(x.shape[-1]), weight)
        if not candidates:
            return ttnn.linear(
                x, weight, bias=bias, dtype=dtype, compute_kernel_config=self.proj_cfg,
                memory_config=out_memory_config,
            )
        errors = []
        for program_config in candidates:
            try:
                out = attempt(program_config)
            except RuntimeError as exc:  # an L1 or validation rejection, not a device fault
                lines = str(exc).splitlines()
                errors.append(lines[2] if len(lines) > 2 else str(exc))
                continue
            self._prefill_pc_cache[cache_key] = program_config
            return out
        raise RuntimeError(
            f"no legal prefill program config for {key} at M={int(x.shape[-2])}; tried "
            f"{len(candidates)} candidates, last errors: {errors[-3:]}"
        )

    def _linear(self, x, key: str, *, decode: bool, dtype, out_memory_config=None, bias=None):
        """One weight matmul, with the phase-appropriate program config (``O2`` / ``O6``).

        Decode uses the DRAM-sharded program with a width-sharded L1 activation; prefill uses a
        2D program config over DRAM-interleaved activations.  ``out_memory_config`` is how the
        caller hands the *consumer's* shard contract to the producer, so the decode path does
        not reshard between projections.
        """
        weight = self.w[key]
        plan = self._decode_plans.get(key) if decode else None
        if plan is None:
            if decode:
                return ttnn.linear(
                    x, weight, bias=bias, dtype=dtype, compute_kernel_config=self.proj_cfg,
                    memory_config=out_memory_config,
                )
            return self._prefill_linear(x, weight, key, dtype=dtype, bias=bias,
                                        out_memory_config=out_memory_config)
        sharded_in = ttnn.to_memory_config(x, plan.input_memory_config)
        out = ttnn.linear(
            sharded_in,
            weight,
            bias=bias,
            dtype=dtype,
            compute_kernel_config=self.proj_cfg,
            program_config=plan.program_config,
            memory_config=out_memory_config or ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        _free(sharded_in, x)
        return out

    @staticmethod
    def _mem(nbytes: int):
        """L1 for tensors that comfortably fit, DRAM otherwise (``F6``)."""
        return ttnn.L1_MEMORY_CONFIG if nbytes <= L1_BUDGET_BYTES else ttnn.DRAM_MEMORY_CONFIG

    @staticmethod
    def _l1_groups(leading: int, bytes_per_unit: int) -> int:
        """Fewest equal leading-dim groups whose working set fits comfortably in L1 (``F21``).

        A batched matmul over a DRAM-resident operand costs ~1.06 us per batch element; the same
        matmul out of L1 costs ~0.043 us.  The gated delta rule's whole-prefill operands are too
        big for L1 as one piece, but they are indexed by chunk, so running them a few chunks at a
        time puts every one of those matmuls in L1.  Half of :data:`L1_BUDGET_BYTES` is used here
        because two of these regions can be live at once.
        """
        # Every divisor, ascending - not just powers of two, so a ragged final prefill chunk
        # (e.g. nc = 15 for seq_len 5000) is split 3 or 5 ways instead of falling all the way
        # through to one chunk per group.
        for groups in range(1, leading + 1):
            if leading % groups:
                continue
            if bytes_per_unit * (leading // groups) <= L1_BUDGET_BYTES // 2:
                return groups
        return leading

    def _grouped_matmul(self, a, b, leading: int, heads: int, rows: int, width: int, *, transpose_b=False):
        """``a @ b`` over the leading chunk axis, a few chunks at a time out of L1 (``F21``)."""
        groups = self._l1_groups(leading, 3 * heads * rows * max(width, rows) * 4)
        if groups == 1:
            return ttnn.matmul(
                a, b, transpose_b=transpose_b, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg
            )
        per_group = leading // groups
        pieces = []
        for g in range(groups):
            lo, hi = g * per_group, (g + 1) * per_group
            a_g = ttnn.slice(a, [lo, 0, 0, 0], [hi, heads, rows, width], memory_config=ttnn.L1_MEMORY_CONFIG)
            b_g = ttnn.slice(b, [lo, 0, 0, 0], [hi, heads, rows, width], memory_config=ttnn.L1_MEMORY_CONFIG)
            pieces.append(
                ttnn.matmul(
                    a_g,
                    b_g,
                    transpose_b=transpose_b,
                    dtype=ttnn.float32,
                    compute_kernel_config=self.compute_cfg,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            )
            ttnn.deallocate(a_g)
            ttnn.deallocate(b_g)
        out = ttnn.concat(pieces, dim=0)
        for piece in pieces:
            _free(piece, out)
        return out

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
        if self.decode_residual_mem_cfg is not None:
            # ``O3``: the norm output *is* the projections' activation shard, so it goes
            # straight into the DRAM-sharded matmuls with no layout op in between.
            return normed
        out = ttnn.to_memory_config(normed, ttnn.DRAM_MEMORY_CONFIG)
        _free(normed, out)
        return out

    def _mlp(self, x, *, decode: bool):
        """SwiGLU MLP.  ``O4`` chooses between packed and separate gate/up projections.

        ``F1`` is still here: the SiLU is an *input activation of the multiply the graph
        already contains*, not a separate unary.  ``ttnn.swiglu`` was tried in the fusing stage
        and is a composite that dispatches ``split -> swish -> multiply``, and
        ``ttnn.linear(activation=...)`` applies its activation as a separate ``unary_chain``
        unless a program config is given; both left the op stream unchanged.

        ``O4``: with a DRAM-sharded decode matmul the packed form's two output slices are no
        longer free - they are width slices of a width-sharded tensor - while the separate form
        halves ``per_core_N`` and lands the product straight in the down projection's shard
        contract.  Both are measured; see work_log.md O4.
        """
        inter = self.shapes.intermediate_size
        activation = self.precision.activation
        # The down projection's activation shard is the contract the gate/up outputs should
        # already satisfy, so the product needs no reshard before it.
        down_plan = self._decode_plans.get("mlp_down") if decode else None
        mid_mem_cfg = down_plan.input_memory_config if down_plan is not None else None
        out_mem_cfg = self.decode_residual_mem_cfg if decode else None

        if self.topology.pack_gate_up:
            gate_up = self._linear(x, "mlp_gate_up", decode=decode, dtype=activation)
            lead = _shape(gate_up)[:3]
            interleaved = ttnn.to_memory_config(gate_up, ttnn.L1_MEMORY_CONFIG) if decode else gate_up
            _free(gate_up, interleaved)
            gate = ttnn.slice(interleaved, [0, 0, 0, 0], [*lead, inter])
            up = ttnn.slice(interleaved, [0, 0, 0, inter], [*lead, 2 * inter])
            _free(interleaved, gate, up)
        else:
            gate = self._linear(x, "mlp_gate", decode=decode, dtype=activation, out_memory_config=mid_mem_cfg)
            up = self._linear(x, "mlp_up", decode=decode, dtype=activation, out_memory_config=mid_mem_cfg)

        activated = ttnn.multiply(
            up,
            gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SILU],
            memory_config=mid_mem_cfg,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = self._linear(
            activated, "mlp_down", decode=decode, dtype=activation, out_memory_config=out_mem_cfg
        )
        ttnn.deallocate(activated)
        return out

    # ------------------------------------------------------- full attention

    def _attn_projections(self, x, *, decode: bool):
        """Return ``(q, k, v, gate)`` with heads split out; ``gate`` is pre-sigmoid (``F9``).

        ``O2``: at decode both projections are DRAM-sharded matmuls off the residual shard.
        ``qkv`` is handed to ``nlp_create_qkv_heads_decode``, which wants an interleaved input,
        so it lands in L1 interleaved; ``gate`` lands directly in the output projection's shard
        contract, where the gating multiply and ``o_proj`` consume it without a reshard.
        """
        s = self.shapes
        gate_mem_cfg = None
        if decode and "o_proj" in self._decode_plans:
            gate_mem_cfg = self._decode_plans["o_proj"].input_memory_config
        qkv = self._linear(x, "wqkv", decode=decode, dtype=ttnn.bfloat16)
        if decode and self._decode_plans:
            # ``nlp_create_qkv_heads_decode`` takes an interleaved input, and a DRAM-sharded
            # matmul must write a sharded one, so the conversion happens here rather than as
            # the matmul's output memory config.
            interleaved = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)
            _free(qkv, interleaved)
            qkv = interleaved
        gate = self._linear(x, "wgate", decode=decode, dtype=ttnn.bfloat16, out_memory_config=gate_mem_cfg)
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

    def _attn_epilogue(self, attn_out, gate, *, decode: bool):
        """F9: the output gate's sigmoid is an input activation of the multiply already here.

        ``ttnn.linear(activation="sigmoid")`` was tried first and does **not** fuse unless a
        program config / ``core_grid`` is given - ``matmul.cpp:295`` applies the activation as a
        separate ``ttnn::unary_chain`` dispatch, which the profiler confirmed (a 192 us
        ``UnaryDeviceOperation`` right after the gate matmul, identical to the unfused one).

        ``O2`` + ``O3``: at decode the multiply runs in the output projection's activation
        shard and the projection writes straight into the residual shard.  One layout op remains
        here - a 1.6-1.8 us ``ReshardDeviceOperation`` - because the DRAM-sharded matmul writes
        its own row-wise core set rather than the rectangle this layer asks for, and the
        rectangle cannot be given up (the sharded layernorm rejects a non-rectangular grid; see
        :func:`_width_sharded_l1`).  It is in the boundary table of work_log.md section 4.
        """
        plan = self._decode_plans.get("o_proj") if decode else None
        mem_cfg = plan.input_memory_config if plan is not None else None
        if mem_cfg is not None:
            resharded = ttnn.to_memory_config(attn_out, mem_cfg)
            _free(attn_out, resharded)
            attn_out = resharded
        gated = ttnn.multiply(
            attn_out, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=mem_cfg
        )
        ttnn.deallocate(attn_out)
        ttnn.deallocate(gate)
        out = self._linear(
            gated,
            "o_proj",
            decode=decode,
            dtype=ttnn.bfloat16,
            out_memory_config=self.decode_residual_mem_cfg if decode else None,
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
        return self._attn_epilogue(concat, gate, decode=False)

    def _full_attention_decode(self, x, *, current_pos, page_table, rot_mats):
        """Decode attention.

        ``F2``: ``q``/``k`` leave ``nlp_create_qkv_heads_decode`` height-sharded, take a detour
        through DRAM only for the two head norms (``rms_norm`` rejects height-sharded inputs) and
        come back sharded for one ``rotary_embedding_hf`` each.  ``k`` and ``v`` then go into two
        ``paged_update_cache`` calls - ``paged_fused_update_cache`` would merge them but cannot be
        used here, see the comment at the call site.  ``v`` never leaves L1.
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
        return self._attn_epilogue(concat, gate, decode=True)

    # ----------------------------------------------------- linear attention

    def _gdn_inputs(self, x, *, decode: bool):
        """Shared GatedDeltaNet input projections.

        ``F5`` + ``F11``: one fp32 matmul with a genuinely fused bias emits both ``b`` and ``a``
        (the profiler shows no separate add after it).  ``z`` is returned pre-SiLU; ``F10``
        folds that SiLU into the gated-norm multiply instead of into the matmul, which does
        not fuse it (see :meth:`_attn_epilogue`).
        """
        s = self.shapes
        # The ``b`` half is padded up to a tile so ``a`` starts on a tile boundary (F5).
        offset = _round_up(s.num_v_heads, ttnn.TILE_SIZE)
        decode = decode and bool(self._decode_plans)
        z_mem_cfg = self._decode_plans["out_proj"].input_memory_config if decode else None
        mixed_qkv = self._linear(x, "in_proj_qkv", decode=decode, dtype=self.precision.gdn_conv)
        if decode:
            # The causal conv and the head split downstream work on interleaved tensors, so the
            # width-sharded projection output is converted once here rather than teaching the
            # whole gated-delta-net decode path a shard contract for a 1.3 MB tensor.
            interleaved = ttnn.to_memory_config(mixed_qkv, ttnn.L1_MEMORY_CONFIG)
            _free(mixed_qkv, interleaved)
            mixed_qkv = interleaved
        z = self._linear(x, "in_proj_z", decode=decode, dtype=ttnn.bfloat16, out_memory_config=z_mem_cfg)
        # F18: at decode the b|a projection is 32 x 5120 x 128 - only four output tiles, so the
        # default program picks four cores and runs at 9.5 % of DRAM roofline (61 us, flagged
        # SLOW by tt-perf-report).  An explicit 4x8 core grid takes it to 35 us; wider padded N
        # and an 8x8 grid were both measured and are worse (doc/fused_decoder/logs/probe_review_followups.log).
        # Prefill has 2048 rows and does not want the restriction.
        ba_in = x
        if decode:
            # ``in_proj_ba`` is a 2.6 MB float32 weight feeding the two gates of the state
            # recurrence, so it stays out of the DRAM-sharded family (its 112-column output is
            # smaller than one shard row) and keeps HiFi4.  Its input comes off the residual
            # shard once.
            ba_in = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        ba = ttnn.linear(
            ba_in,
            self.w["in_proj_ba"],
            bias=self.w["ba_bias"],
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_cfg,
            **({"core_grid": ttnn.CoreGrid(y=4, x=8)} if int(x.shape[-2]) <= ttnn.TILE_SIZE else {}),
        )
        _free(ba_in, x)
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
        # F22: `window` is `prefix` (K-1 rows) followed by `mixed_qkv`, so tap K-1 covers exactly
        # `mixed_qkv` - taking it directly saves a slice at a non-tile-aligned row, which TTNN
        # implements as untilize -> slice -> retilize on an 84 MB tensor.
        acc = ttnn.multiply(mixed_qkv, self.w["conv_taps"][k - 1])
        for j in range(k - 1):
            tap = ttnn.slice(window, [0, 0, j, 0], [1, 1, j + length, s.conv_dim])
            updated = ttnn.addcmul(acc, tap, self.w["conv_taps"][j], value=1.0)
            ttnn.deallocate(acc)
            ttnn.deallocate(tap)
            acc = updated
        new_state = ttnn.slice(window, [0, 0, logical - 1, 0], [1, 1, logical - 1 + k, s.conv_dim])
        _free(window, new_state)
        activations = ttnn.silu(acc)
        ttnn.deallocate(acc)
        # ``O8``: the conv may run at a narrower dtype than the delta-rule state math; the
        # boundary is here, so everything downstream sees exactly the float32 it did before.
        if activations.dtype != ttnn.float32:
            wide = ttnn.typecast(activations, ttnn.float32)
            _free(activations, wide)
            activations = wide
        return activations, new_state

    def _split_qkv(self, conv_out):
        s = self.shapes
        lead = _shape(conv_out)[:-1]
        starts = [0] * len(lead)
        q = ttnn.slice(conv_out, [*starts, 0], [*lead, s.key_dim])
        k = ttnn.slice(conv_out, [*starts, s.key_dim], [*lead, 2 * s.key_dim])
        v = ttnn.slice(conv_out, [*starts, 2 * s.key_dim], [*lead, s.conv_dim])
        return q, k, v

    def _batched_program_config(self, a, b, transpose_a: bool, transpose_b: bool):
        """``O10``: an explicit program config for the batched delta-rule matmuls, or ``None``.

        ``tt-perf-report`` flags these with "No program_config specified, try using one to
        override in0_block_w and out_subblock_h/w".  The first implementation of this method
        measured two shapes and then applied one fixed 8x6 config to *every* tile-aligned
        batched matmul, which was too broad: the triangular inverse's single-tile base case got
        **slower** (``b={768} x 32 x 32 x 32``, 271.5 -> 535.9 us), because a 48-core grid is
        narrower than what the default program picks for a 768-deep batch of one-tile matmuls.

        So every shape the prefill actually dispatches was swept - grid x ``in0_block_w`` x
        ``out_subblock_h``, PCC-gated, ``probes/batched_matmul_sweep.py``,
        ``logs/batched_matmul_sweep.log``:

        =========================  =========  ========  ========  ========
        shape                      default    11x10     8x6       8x8
        =========================  =========  ========  ========  ========
        ``b=48 x 64 x 128 x 128``   20.3 us   **9.8**    12.7      10.5
        ``b=384 x 64 x 128 x 64``   57.0 us   **25.4**   34.6      29.8
        ``b=48 x 64 x 64 x 128``    14.5 us    13.6     **9.4**     9.7
        ``b=768 x 32 x 32 x 32``   **11.9**    15.6      17.8      14.8
        ``b=384 x 32 x 32 x 32``    11.6 us    12.0     **10.0**   21.7
        =========================  =========  ========  ========  ========

        Two rules fall out and both are in the code below.  A matmul only one tile tall has
        nothing to spread over a fixed grid, and the default program beats every explicit
        candidate for it at the batch depth that matters - so single-M-tile shapes are left
        alone.  Above that, the full 11x10 worker grid wins whenever K is 4 tiles or more, and
        the 48-core grid (one per value head) wins for the narrower K.

        ``in0_block_w = 1`` and ``out_subblock_h = 1`` are the measured optimum for every shape
        here, not a floor left unexplored: 2 and 4 were swept and are 3-10 % slower
        (``11x10_ibw2`` 10.1 us and ``ibw4`` 10.4 us against 9.8 us for the largest shape).
        That is the answer to the ``in0_block_w=1 is small`` advice this group carries.

        Shapes the reuse program refuses are cached as ``None`` after one failed dispatch.
        """
        shape_a, shape_b = _shape(a), _shape(b)
        if len(shape_a) != 4 or len(shape_b) != 4:
            return None
        m = shape_a[-1] if transpose_a else shape_a[-2]
        n = shape_b[-2] if transpose_b else shape_b[-1]
        k = shape_a[-2] if transpose_a else shape_a[-1]
        if m % ttnn.TILE_SIZE or n % ttnn.TILE_SIZE or k % ttnn.TILE_SIZE:
            return None
        m_tiles, k_tiles, n_tiles = (
            m // ttnn.TILE_SIZE,
            k // ttnn.TILE_SIZE,
            n // ttnn.TILE_SIZE,
        )
        if m_tiles < 2:
            return None  # the default program wins for one-tile-tall matmuls; see above
        key = self._batched_key(a, b, transpose_a, transpose_b)
        if key in self._batched_pc_cache:
            return self._batched_pc_cache[key]
        grid = FULL_WORKER_GRID if k_tiles >= 4 else BATCHED_MATMUL_GRID
        config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=next(w for w in (4, 3, 2, 1) if n_tiles % w == 0),
            per_core_M=m_tiles,
            per_core_N=n_tiles,
        )
        self._batched_pc_cache[key] = config
        return config

    @staticmethod
    def _batched_key(a, b, transpose_a: bool, transpose_b: bool):
        """``(M, K, N, transpose_a, transpose_b)`` after transposition - the program-config key.

        The **batch** dimension is deliberately not part of the key.  Every field
        :meth:`_batched_program_config` sets is a function of ``M``, ``K`` and ``N`` only
        (``in0_block_w``, ``out_subblock_h/w``, ``per_core_M``, ``per_core_N``, and a grid chosen
        from ``k_tiles``); ``MatmulMultiCoreReuse`` spreads ``batch * M`` over the grid itself.
        So ``b={48} x 64 x 128 x 128`` and ``b={384} x 64 x 128 x 128`` want the same config, and
        keying on batch would only split the cache.  If a future field does depend on batch, it
        has to be added here.
        """
        shape_a, shape_b = _shape(a), _shape(b)
        return (
            shape_a[-1] if transpose_a else shape_a[-2],
            shape_a[-2] if transpose_a else shape_a[-1],
            shape_b[-2] if transpose_b else shape_b[-1],
            transpose_a,
            transpose_b,
        )

    def _mm(self, a, b, memory_config=None, transpose_b=False, transpose_a=False):
        """One delta-rule matmul, with ``O10``'s program config where the shape allows it."""
        program_config = self._batched_program_config(a, b, transpose_a, transpose_b)

        def run(config):
            return ttnn.matmul(
                a,
                b,
                dtype=ttnn.float32,
                compute_kernel_config=self.compute_cfg,
                memory_config=memory_config,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                **({} if config is None else {"program_config": config}),
            )

        if program_config is None:
            return run(None)
        try:
            return run(program_config)
        except RuntimeError:
            # The reuse program refused this shape.  Remember that so it is tried once, and
            # fall back to the ttnn-chosen program.
            self._batched_pc_cache[self._batched_key(a, b, transpose_a, transpose_b)] = None
            return run(None)

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

    def _prefill_heads(self, conv_out, padded: int):
        """F17: split the fused conv output into per-head ``q``/``k``/``v`` with a dedicated op.

        ``conv_out`` is ``[1, 1, L, 2*key_dim + value_dim]`` = ``[q(16*128) | k(16*128) |
        v(48*128)]``.  The obvious ``reshape → permute`` head split is a last-dim change, which
        TTNN implements as untilize + retilize: measured 11.25 ms for the three tensors at
        ``L`` = 2048, 14 % of the whole layer.

        ``ttnn.experimental.nlp_create_qkv_heads`` is the dedicated kernel for exactly this, but
        it requires K and V to have the same head count, and this mixer has 16 key heads and 48
        value heads.  Two overlapping calls express it anyway, because 48 value heads are three
        consecutive groups of 16:

            call A over columns [0, 6144)      -> q,      k,       v heads 0-15
            call B over columns [4096, 10240)  -> (v0),   v 16-31, v 32-47

        Measured 11.25 ms -> 3.14 ms, **bit-identical** to the reshape path
        (``doc/fused_decoder/logs/probe_headsplit_narrow.log``).  The duplicated ``v0`` output of call B is the
        only waste and is one third of one of the two calls.
        """
        s = self.shapes
        group = s.key_dim  # 16 heads x head_k_dim, and the width of one third of v
        assert s.value_dim == 3 * group and s.key_dim == s.num_k_heads * s.head_k_dim, (
            "the two-call head split assumes value_dim == 3 * key_dim (48 v-heads / 16 k-heads)"
        )
        first = ttnn.slice(conv_out, [0, 0, 0, 0], [1, 1, padded, 3 * group])
        q, k, v0 = ttnn.experimental.nlp_create_qkv_heads(
            first,
            num_heads=s.num_k_heads,
            num_kv_heads=s.num_k_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(first)
        second = ttnn.slice(conv_out, [0, 0, 0, 2 * group], [1, 1, padded, s.conv_dim])
        v0_again, v1, v2 = ttnn.experimental.nlp_create_qkv_heads(
            second,
            num_heads=s.num_k_heads,
            num_kv_heads=s.num_k_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(second)
        ttnn.deallocate(v0_again)
        v = ttnn.concat([v0, v1, v2], dim=1)
        for tensor in (v0, v1, v2):
            _free(tensor, v)
        # F24: with the value heads reordered, matching key heads to value heads is a plain
        # concat along the head axis instead of ttnn.repeat_interleave, which is a composite
        # (typecast -> untilize -> concat -> tilize -> typecast, repeat_interleave.cpp:34-68)
        # that cost 1.82 ms here and round-tripped the float32 q/k through bfloat16.
        q_rep = ttnn.concat([q] * s.v_per_k, dim=1)
        _free(q, q_rep)
        k_rep = ttnn.concat([k] * s.v_per_k, dim=1)
        _free(k, k_rep)
        return q_rep, k_rep, v

    def _linear_attention_prefill_chunk(self, x, *, state, conv_prefix, length):
        """One prefill chunk of the gated delta rule.

        ``x``: ``[1, 1, L, hidden]`` (``L`` may exceed ``length`` after zero padding).
        ``state``: ``[1, num_v_heads, head_k_dim, head_v_dim]`` float32 recurrent state.
        Returns ``(out [1, 1, L, value_dim], new_state, new_conv_state)``.
        """
        s = self.shapes
        chunk = DELTA_CHUNK
        mixed_qkv, z, beta, g = self._gdn_inputs(x, decode=False)
        conv_out, new_conv_state = self._causal_conv(mixed_qkv, conv_prefix, length)
        ttnn.deallocate(mixed_qkv)
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

        q, k, v = self._prefill_heads(conv_out, padded)
        ttnn.deallocate(conv_out)

        def to_chunks(t, head_dim):
            t = ttnn.reshape(t, (nv, nc, chunk, head_dim))
            return ttnn.permute(t, (1, 0, 2, 3))  # [nc, H, chunk, D]

        q, k, v = (to_chunks(t, d) for t, d in ((q, s.head_k_dim), (k, s.head_k_dim), (v, s.head_v_dim)))

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
        # ``O13``: the per-chunk total decay is the *last row* of ``g_cum``, and reading it as
        # ``g_cum[:, :, chunk - 1 : chunk, :]`` is a row-shifted slice of a tiled tensor - the
        # §18 audit found it costing 308.7 us of untilize/slice/tilize (op IDs 1089-1091 of the
        # pre-``O13`` profile), the largest layout cost outside the causal conv.  A cumulative
        # sum's last element is the plain sum, so a reduction over the chunk axis gives the same
        # value while staying tiled.  ``keepdim`` preserves the ``[nc, nv, 1, 1]`` shape the two
        # consumers broadcast against.
        g_last = ttnn.sum(g_h, dim=-2, keepdim=True, memory_config=mem,
                          compute_kernel_config=self.compute_cfg)
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

        # F21: `kk` a few chunks at a time in L1.  As one [nc, nv, chunk, dk] x [.., dk, chunk]
        # batched matmul out of DRAM it ran on 4 of 110 cores at 5.4 % of roofline (4.57 ms).
        kk = self._grouped_matmul(k_beta, k, nc, nv, chunk, s.head_k_dim, transpose_b=True)
        # F14: `neg_strict_lower` carries both the sign and the strict-lower mask.
        scaled_decay = ttnn.multiply(decay, self.const["neg_strict_lower"], memory_config=mem)
        attn0 = ttnn.multiply(kk, scaled_decay, memory_config=mem)
        ttnn.deallocate(scaled_decay)
        ttnn.deallocate(kk)

        # F21: the whole triangular inverse in L1, a few chunks at a time.  As one piece its top
        # level needs ~126 MB and falls back to DRAM, where its two matmuls ran on 1 core at 2.3 %
        # of DRAM roofline (3.2 ms); split into groups it fits, and `_tri_mem` then picks L1 at
        # every level by itself.
        groups = self._l1_groups(nc, 5 * nv * chunk * chunk * 4)
        if groups == 1:
            inv = self._unit_tri_inverse(attn0, chunk)
        else:
            per_group = nc // groups
            pieces = []
            for g in range(groups):
                part = ttnn.slice(
                    attn0,
                    [g * per_group, 0, 0, 0],
                    [(g + 1) * per_group, nv, chunk, chunk],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                pieces.append(self._unit_tri_inverse(part, chunk))
                _free(part, pieces[-1])
            inv = pieces[0] if groups == 1 else ttnn.concat(pieces, dim=0)
            for piece in pieces:
                _free(piece, inv)
        ttnn.deallocate(attn0)

        exp_gcum = ttnn.exp(g_cum, memory_config=mem)
        k_beta_decayed = ttnn.multiply(k_beta, exp_gcum, memory_config=mem)
        ttnn.deallocate(k_beta)
        # F21: `value = inv @ v_beta` and `k_cumdecay = inv @ k_beta_decayed` used to be two whole
        # -prefill batched matmuls out of DRAM, 8 of 110 cores at 8.6 % of roofline (2.87 ms each).
        # The per-chunk loop below already slices both results one chunk at a time, so the matmuls
        # move into it and run out of L1 on the slices instead.

        decay_to_end = ttnn.exp(ttnn.subtract(g_last, g_cum, memory_config=mem), memory_config=mem)
        exp_g_last = ttnn.exp(g_last, memory_config=mem)
        ttnn.deallocate(g_last)
        k_decayed = ttnn.multiply(k, decay_to_end, memory_config=mem)
        ttnn.deallocate(decay_to_end)
        q_decayed = ttnn.multiply(q, exp_gcum, memory_config=mem)
        ttnn.deallocate(exp_gcum)

        # ``O12``: two of the loop's matmuls read the recurrent state, and it arrives DRAM
        # interleaved.  This module's own ``_l1_groups`` docstring records the cost: a batched
        # matmul over a DRAM-resident operand is ~1.06 us per batch element against ~0.043 us out
        # of L1, and the profile showed exactly that - the two ``@ state`` dispatches ran at
        # 20.6 us against 9.8 us for the same shape with both operands in L1.  The state is
        # 3.1 MB at batch 1, so it goes to L1 for the duration of the loop and back to DRAM once.
        state_home = state.memory_config()
        if loop_mem is ttnn.L1_MEMORY_CONFIG:
            state_l1 = ttnn.to_memory_config(state, ttnn.L1_MEMORY_CONFIG)
            _free(state, state_l1)
            state = state_l1

        outputs = []
        for i in range(nc):
            sl = lambda t, w: ttnn.slice(t, [i, 0, 0, 0], [i + 1, nv, chunk, w], memory_config=loop_mem)  # noqa: E731
            q_i = sl(q, s.head_k_dim)
            k_i = sl(k, s.head_k_dim)
            inv_i = sl(inv, chunk)
            v_i = self._mm(inv_i, sl(v_beta, s.head_v_dim), loop_mem)
            kc_i = self._mm(inv_i, sl(k_beta_decayed, s.head_k_dim), loop_mem)
            _free(inv_i, inv)
            d_i = sl(decay, chunk)
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

            decayed_state = ttnn.multiply(state, gl_i, memory_config=loop_mem)
            update = self._mm(kd_i, v_new, memory_config=loop_mem, transpose_a=True)
            ttnn.deallocate(v_new)
            new_state = ttnn.add(decayed_state, update, memory_config=loop_mem)
            ttnn.deallocate(decayed_state)
            ttnn.deallocate(update)
            ttnn.deallocate(state)
            state = new_state
            ttnn.deallocate(v_i)
            ttnn.deallocate(kc_i)
            for tensor in (q_i, k_i, d_i, qd_i, kd_i, gl_i):
                _free(tensor, q, k, decay, q_decayed, k_decayed, exp_g_last)

        for tensor in (q, k, decay, q_decayed, k_decayed, exp_g_last, g_cum, inv, v_beta, k_beta_decayed):
            ttnn.deallocate(tensor)
        if state.memory_config() != state_home:  # ``O12``: hand the state back where it came from
            state_dram = ttnn.to_memory_config(state, state_home)
            _free(state, state_dram)
            state = state_dram

        # F19: the per-chunk outputs are [1, nv, chunk, Dv] and chunk i holds tokens
        # [i*chunk, (i+1)*chunk), so concatenating along the *sequence* axis lands directly in
        # [1, nv, L, Dv].  Concatenating along the chunk axis instead, as the unfused code did,
        # then needs a permute and a reshape to get there - and that reshape was 2.3 ms.
        core = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2, memory_config=mem)
        if len(outputs) > 1:
            for tensor in outputs:
                ttnn.deallocate(tensor)
        # core is [1, nv, L, Dv] - head-major, which is what the gated norm wants (it reduces
        # over Dv) *and* what the dedicated head-concat op wants.
        normed = self._rms_norm(ttnn.typecast(core, ttnn.bfloat16), self.w["gated_norm"])
        ttnn.deallocate(core)
        # F20: `permute([0, 2, 1, 3]) -> reshape` back to [1, 1, L, value_dim] is the graph-fusing
        # skill's prefill head-concat pattern, and `nlp_concat_heads` is its dedicated op.  The
        # permute was 625 us and the reshape 2.93 ms (a last-dim change, so untilize + retilize);
        # the op does both in 0.12 ms.  Reshaping `z` into head shape instead was measured at
        # 3.1 ms.  F10 keeps z's SiLU as an input activation of the multiply that follows.
        normed_flat = ttnn.experimental.nlp_concat_heads(normed)
        _free(normed, normed_flat)
        gated = ttnn.multiply(normed_flat, z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(normed_flat)
        ttnn.deallocate(z)
        out = self._linear(gated, "out_proj", decode=False, dtype=ttnn.bfloat16)
        ttnn.deallocate(gated)
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
        # Everything here stays in the decode layout [1, 1, batch, ...].  The unfused decoder
        # reshaped to [1, batch, 1, hidden] so that its conv window concat had the window on the
        # height axis; F8's per-tap buffers removed that need, and keeping the batch on the
        # height axis is what makes each conv-state buffer one tile row tall instead of `batch`
        # of them (3.9 MB for the three buffers at batch 32 instead of 126 MB).
        mixed_qkv, z, beta, g = self._gdn_inputs(x, decode=True)

        # F8: tap j reads conv_state[j] directly - no slice, so no untilize/retilize - and
        # F16 folds each tap's multiply and add into one addcmul.  The write-back shifts the
        # buffers by one token in place, which keeps their addresses stable for trace replay.
        acc = ttnn.multiply(mixed_qkv, self.w["conv_taps"][s.conv_kernel_size - 1], memory_config=mem)
        for j in range(s.conv_kernel_size - 1):
            updated = ttnn.addcmul(
                acc, self.conv_state[j], self.w["conv_taps"][j], value=1.0, memory_config=mem
            )
            ttnn.deallocate(acc)
            acc = updated
        for j in range(s.conv_kernel_size - 2):
            ttnn.copy(self.conv_state[j + 1], self.conv_state[j])
        ttnn.copy(mixed_qkv, self.conv_state[s.conv_kernel_size - 2])
        ttnn.deallocate(mixed_qkv)
        conv_out = ttnn.silu(acc, memory_config=mem)
        ttnn.deallocate(acc)
        if conv_out.dtype != ttnn.float32:  # ``O8``, as in prefill
            wide = ttnn.typecast(conv_out, ttnn.float32, memory_config=mem)
            _free(conv_out, wide)
            conv_out = wide

        q_flat, k_flat, v_flat = self._split_qkv(conv_out)
        ttnn.deallocate(conv_out)

        def to_heads(flat, head_dim, repeat: int):
            # F24: repeat on the *flat* tensor's last axis, which is tile-aligned, rather than on
            # the head axis of [1, batch, 16, 128] - a 16-row concat is not tile-aligned and
            # TTNN pays for it with an untilize/retilize.  Concatenating [q | q | q] column-wise
            # gives exactly the head order the reordered value heads want.
            if repeat > 1:
                wide = ttnn.concat([flat] * repeat, dim=-1)
                _free(flat, wide)
                flat = wide
            # ``O9``: one reshape, not two.  Both are row-major views of the same flat buffer,
            # so the intermediate ``[1, batch, heads, head_dim]`` step is redundant - and it is
            # not free: this is a last-dim change, which TTNN implements as untilize + retilize.
            # The ten decode reshapes were 103.5 us of a 1132 us step before this.
            return ttnn.reshape(flat, (1, batch * nv, 1, head_dim))

        q = to_heads(q_flat, s.head_k_dim, s.v_per_k)
        k = to_heads(k_flat, s.head_k_dim, s.v_per_k)
        v = to_heads(v_flat, s.head_v_dim, 1)
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
        update = self._mm(k, delta, memory_config=mem, transpose_a=True)
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
        normed = self._rms_norm(ttnn.typecast(core, ttnn.bfloat16), self.w["gated_norm"])
        ttnn.deallocate(core)
        # F20 + F10, as in prefill: flatten the norm output rather than reshaping z into heads.
        normed_flat = ttnn.reshape(normed, (1, 1, batch, s.value_dim))
        _free(normed, normed_flat)
        # ``O2`` + ``O3``: ``z`` already sits in the output projection's activation shard, so the
        # gating multiply happens there and ``out_proj`` writes straight into the residual shard.
        z_mem_cfg = z.memory_config() if self._decode_plans else None
        if self._decode_plans:
            resharded = ttnn.to_memory_config(normed_flat, z_mem_cfg)
            _free(normed_flat, resharded)
            normed_flat = resharded
        gated = ttnn.multiply(
            normed_flat, z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=z_mem_cfg
        )
        ttnn.deallocate(normed_flat)
        ttnn.deallocate(z)
        result = self._linear(
            gated,
            "out_proj",
            decode=True,
            dtype=ttnn.bfloat16,
            out_memory_config=self.decode_residual_mem_cfg,
        )
        ttnn.deallocate(gated)
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
            mlp_out = self._mlp(normed2, decode=False)
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
        # ``O3``: the residual stream is width-sharded in L1 for the whole step.  ``residual``
        # may therefore be a fresh tensor rather than the caller's, so it is freed below.
        residual = hidden_states
        if self.decode_residual_mem_cfg is not None:
            residual = ttnn.to_memory_config(hidden_states, self.decode_residual_mem_cfg)
        normed = self._decode_rms_norm(residual, self.w["input_layernorm"])
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
        hidden = ttnn.add(residual, mixed, memory_config=self.decode_residual_mem_cfg)
        _free(residual, hidden_states)
        ttnn.deallocate(mixed)
        normed2 = self._decode_rms_norm(hidden, self.w["post_attention_layernorm"])
        mlp_out = self._mlp(normed2, decode=True)
        ttnn.deallocate(normed2)
        out = ttnn.add(hidden, mlp_out, memory_config=self.decode_residual_mem_cfg)
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
            dtype=self.conv_state[0].dtype,
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
        # F8: the per-user prefill state is [1, 1, K, conv_dim] with the window on the height
        # axis; the decode buffers are one per tap with the batch on the height axis, so the
        # fold transposes.  This runs once, outside any traced region.
        s = self.shapes
        for tap in range(s.conv_kernel_size - 1):
            rows = [
                ttnn.slice(state, [0, 0, tap + 1, 0], [1, 1, tap + 2, s.conv_dim])
                for state in self.user_conv_state
            ]
            plane = rows[0] if len(rows) == 1 else ttnn.concat(rows, dim=2)
            ttnn.copy(plane, self.conv_state[tap])
            _free(plane, *rows)
            for row in rows:
                _free(row, *self.user_conv_state)


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

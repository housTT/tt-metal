# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Performance-optimized TTNN decoder for Qwen/Qwen3.6-27B (HF ``model_type: qwen3_5``).

:class:`OptimizedDecoder` is a drop-in replacement for :class:`~.fused_decoder.FusedDecoder`:
same constructor, same ``prefill_forward`` / ``decode_forward`` / ``prefill_chunk_plan`` /
``prepare_decode_state`` / ``current_conv_state`` contract, same paged KV-cache geometry, same
per-user linear-attention state, same public "any ``1 <= seq_len <= max_seq_len``, no
divisibility requirement" prefill API, same acceptance bar.  Stage 2 fused the *op graph* at a
fixed precision policy and a fixed memory layout; this stage changes the **precision policy, the
math fidelity, the memory layout and the matmul program configs** of that graph.

Where the fused stage's time actually goes, read out of its own committed reports
(``doc/fused_decoder/tracy/fused/*/``), is the whole justification for what this file does:

* **traced decode, batch 1** - 87 % (``full_attention``) and 80 % (``linear_attention``) of the
  step is five DRAM-bound projection matmuls reading **bfloat16** weights at ~415 GB/s, i.e. at
  ~81 % of this chip's DRAM roofline.  No graph rewrite is left there: the only way to move fewer
  than 2 bytes per weight is to store fewer.
* **prefill, 2048 tokens** - 76 % (``full_attention``) and 54 % (``linear_attention``) of the pass
  is the same five matmuls, and every one of them is ``Bound=FLOP`` at **HiFi4**, i.e. at 70-79 %
  of the *HiFi4* FLOP roofline, which is a quarter of the LoFi one.  Prefill is not short of
  bandwidth (16-18 % DRAM); it is paying four passes per multiply for precision the model does
  not need.

So the levers, in the order of their measured effect:

1. **Precision and fidelity, per tensor group** (:class:`PrecisionPolicy`).  Attention and MLP
   weights move to block-float and math fidelity drops to the lowest each group tolerates.  Norms,
   the carried recurrent/conv state, the gated-delta-rule core and the recurrence matmuls keep
   their float32/HiFi4 contract - they are state, not weights.
2. **DRAM-sharded decode matmuls over a single width-sharded L1 activation stream**
   (:class:`DecodeGeometry`).  Every decode projection becomes
   ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` reading a width-sharded L1
   activation and writing one, over **one** core count for the whole layer, so the residual
   stream, both RMS norms, the attention epilogue and the MLP never leave L1 and no reshard is
   paid between them.
3. **A split decode gate/up MLP and explicit prefill program configs** - the two knobs that only
   pay off once (1) and (2) have changed which resource is scarce.

Everything measured, kept and rejected is in ``doc/optimized_decoder/work_log.md``; the
before/after tables and the ``tt-perf-report`` conclusions are in
``doc/optimized_decoder/README.md``.

Nothing here changes the model's capability: the context contract, the paged-KV geometry, the
public non-aligned sequence-length API, the logical batch semantics and the trace path are the
fused stage's, unchanged.  The two capacity changes are recorded in ``doc/context_contract.json``:
block-float weights and a ``bfloat8_b`` KV cache make a layer much *smaller*, and the split decode
MLP holds one extra copy of the gate/up weight so each phase can run its measured-best form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import ttnn

from .functional_decoder import SDPA_DECODE_K_CHUNK, _free, _round_up, _shape
from .fused_decoder import _AB_STRIDE, _GATED_NORM_GROUP_BATCH, _L2NORM_EPS, FusedDecoder
from .model_config import FULL_ATTENTION

# --------------------------------------------------------------------------- roles

#: The projection roles this stage tunes independently.  Each is a ``(K, N)`` matmul that appears
#: in the measured prefill and/or decode window, and each ``self.w`` key equals its role name.
#:
#: ``mlp_gate`` / ``mlp_up`` are the two halves of ``mlp_gate_up``.  The fused stage measured the
#: packed form faster at prefill (its work log section 3.8) and this stage re-measures both at the
#: new dtype/fidelity for both phases, because packing trades a wider output - and therefore a
#: smaller legal ``in0_block_w`` and two slices of a 34816-wide tensor - for one fewer dispatch;
#: which side of that wins is a function of the dtype and the layout, not of the graph (OPT-010).
ROLES = (
    "wqkv",
    "wgate",
    "o_proj",
    "mlp_gate_up",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_ab",
    "out_proj",
)

LOFI = ttnn.MathFidelity.LoFi
HIFI2 = ttnn.MathFidelity.HiFi2
HIFI4 = ttnn.MathFidelity.HiFi4


def _kernel_cfg(fidelity, fp32_dest_acc_en: bool = False) -> ttnn.WormholeComputeKernelConfig:
    """Compute-kernel config for one tensor group.

    ``packer_l1_acc`` is on everywhere: it is free and strictly reduces the DRAM round trips of a
    partial sum.  ``fp32_dest_acc_en`` is *not* free - it halves destination-register capacity and
    for these shapes roughly halves matmul throughput - so it is set only where the consumer is a
    float32 tensor the recurrence carries (see :meth:`PrecisionPolicy.fp32_acc`).
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )


# --------------------------------------------------------------------- precision policy


@dataclass(frozen=True)
class PrecisionPolicy:
    """Per-tensor-group weight dtype, math fidelity and KV-cache dtype.

    One field per group rather than one global switch, because the groups have measurably
    different tolerances and measurably different payoffs; ``doc/optimized_decoder/work_log.md``
    moves them one at a time.  ``dataclasses.replace(policy, mlp_weight=...)`` is how the probes
    and the tests sweep it.

    Provenance of the shipped default, group by group:

    ``attn_weight`` / ``attn_fidelity``
        ``wqkv``, ``wgate`` and ``o_proj`` - pure linear projections of the normed hidden state
        with no per-projection dtype, bias, adapter or device-placement contract, so they take the
        lowest precision that keeps the layer at the acceptance bar on **real checkpoint weights**
        (OPT-007 makes this trial mandatory rather than optional).
    ``mlp_weight`` / ``mlp_down_weight`` / ``mlp_fidelity``
        The largest decode cost in both layer kinds, so the BFP4/LoFi trial is mandatory.
        ``mlp_down_weight`` is a separate field because the down projection is the numerically
        sensitive half of the pair and the skill expects it to be able to differ.
    ``gdn_qkv_weight`` / ``gdn_qkv_fidelity``
        ``in_proj_qkv``.  Its **output** is float32 and is what the causal conv carries as state,
        so this role keeps ``fp32_dest_acc_en`` and a fidelity high enough that the carried
        ``conv_state`` still matches HF's cache object.  Weight dtype and accumulation precision
        are independent levers and are measured independently.
    ``gdn_state_fidelity``
        The gated-delta-rule recurrence matmuls, the two gated-norm constant matmuls,
        ``chunk_gated_delta_rule`` and every norm/RoPE op: float32 state arithmetic, kept at HiFi4
        with float32 destination accumulation.  The fused stage's own sweeps show the levers there
        are core grids, not precision.
    ``kv_cache``
        The paged K/V cache.  ``bfloat8_b`` halves both the cache footprint and every SDPA read
        (OPT-002 makes the reduced-cache trial mandatory when SDPA is material).
    """

    name: str = "opt-v1"

    #: Attention projections stay **BFP8**, and that is a measured rejection of BFP4 rather than a
    #: default: on the real checkpoint a BFP4 attention policy takes ``linear_attention`` decode to
    #: PCC 0.952900 and ``full_attention`` traced decode to 0.989013, both below the 0.995 bar, while
    #: BFP8 sits at 0.999310 / 0.999562 (``logs/probe_real_weight_policy.log``).  OPT-007 asks for
    #: exactly this trial on real weights with a cache-consuming follow-on, and it is what rejects it.
    attn_weight: object = ttnn.bfloat8_b
    attn_fidelity: object = LOFI

    #: The gate/up pair is **BFP4** and the down projection is **BFP8**, which is the split the real
    #: checkpoint asks for: BFP4 on gate/up holds at 0.999411 / 0.997708 / 0.998741
    #: (prefill / decode / traced, ``linear_attention``) and 0.999081 / 0.999253 / 0.998238
    #: (``full_attention``), while extending BFP4 to the down projection drops ``full_attention`` to
    #: 0.992892 / 0.992573 / 0.991344 - below the bar.  The skill predicts exactly this asymmetry
    #: ("try BFP4 for FF2/down-projection, but expect it to be more sensitive"), and this is the
    #: evidence rather than the expectation.
    mlp_weight: object = ttnn.bfloat4_b
    mlp_down_weight: object = ttnn.bfloat8_b
    mlp_fidelity: object = LOFI

    gdn_qkv_weight: object = ttnn.bfloat8_b
    gdn_qkv_fidelity: object = HIFI2
    gdn_z_weight: object = ttnn.bfloat8_b
    gdn_out_weight: object = ttnn.bfloat8_b
    gdn_proj_fidelity: object = LOFI

    #: ``in_proj_ab`` stays float32: four output tiles feeding ``softplus``/``exp`` into the
    #: recurrence's decay, i.e. state arithmetic, and at 15 us of a ~2 ms step there is nothing to
    #: win by reducing it.
    gdn_ab_weight: object = ttnn.float32
    gdn_state_fidelity: object = HIFI4

    kv_cache: object = ttnn.bfloat8_b

    #: Math fidelity every projection uses **at decode**, overriding the per-group fields above.
    #: ``None`` - the shipped value - uses the per-group fidelity in both phases.
    #:
    #: This field exists because the obvious assumption is wrong and the measurement says so.  The
    #: audit found every decode projection ``Bound=DRAM`` at 79-84 % of the roofline *at bfloat16*,
    #: which suggests a higher decode fidelity should be free.  It is not, once the weights are
    #: block-float: raising decode fidelity from LoFi to HiFi2 at the same BFP8 weights costs
    #: **+29 %** of the traced ``linear_attention`` step (1.5325 ms -> 1.9773 ms) and **+43 %** of the
    #: ``full_attention`` one (1.1705 -> 1.6694), because halving the weight bytes moved those
    #: matmuls off the bandwidth ceiling and onto the compute one.  The accuracy it buys is small in
    #: comparison (``full_attention`` decode PCC 0.995928 -> 0.996567).  So the shipped policy is
    #: LoFi in both phases, and this field is the measured lever that says why - ``work_log.md``
    #: section 3.1.
    decode_fidelity: object = None

    #: Force float32 destination accumulation on **every** projection.  The fused stage did this
    #: implicitly - it used one ``HiFi4 + fp32_dest_acc_en`` config for every matmul in the layer -
    #: so the baseline arm of the candidate table sets it, and the shipped policy does not: it costs
    #: roughly half of matmul throughput and only the roles below need it.
    fp32_dest_acc_all: bool = False

    #: Roles that accumulate in float32 destination registers **at prefill only**.
    #:
    #: ``wqkv`` is here because of a full-context failure this stage found and traced, and the trace
    #: is worth keeping: with float32 destination accumulation dropped everywhere except the two
    #: float32-output roles, the 262143-token ``full_attention`` prefill tail came back at PCC
    #: 0.972423 against stage 2's 0.998030 - and it was **not** a precision-policy effect.  Reverting
    #: the weight dtypes and raising the fidelity did not fix it (HiFi4 everywhere is *worse*, 0.947617),
    #: while the paged K/V cache PCC moved from stage 2's 0.999989 to 0.999924.  ``wqkv``'s reduction is
    #: 160 K tiles deep and its output *is* what the cache stores, so accumulating it in bfloat16
    #: destination registers costs the cache about 6x its error - invisible at 2049 keys, and amplified
    #: by an attention over 262144 of them.  ``logs/probe_long_context_precision.log`` and
    #: ``logs/probe_long_context_mlp.log`` are the attribution.
    #:
    #: Prefill only, because prefill is what fills the cache: a decode step writes one token's K/V into
    #: a 262144-entry cache, so its accumulation precision cannot move the cache PCC, and decode is
    #: where float32 destination accumulation costs the most relative to the work done.
    prefill_fp32_acc_roles: tuple = ("wqkv",)

    _WEIGHT_FIELD = {
        "wqkv": "attn_weight",
        "wgate": "attn_weight",
        "o_proj": "attn_weight",
        "mlp_gate_up": "mlp_weight",
        "mlp_gate": "mlp_weight",
        "mlp_up": "mlp_weight",
        "mlp_down": "mlp_down_weight",
        "in_proj_qkv": "gdn_qkv_weight",
        "in_proj_z": "gdn_z_weight",
        "in_proj_ab": "gdn_ab_weight",
        "out_proj": "gdn_out_weight",
    }

    _FIDELITY_FIELD = {
        "wqkv": "attn_fidelity",
        "wgate": "attn_fidelity",
        "o_proj": "attn_fidelity",
        "mlp_gate_up": "mlp_fidelity",
        "mlp_gate": "mlp_fidelity",
        "mlp_up": "mlp_fidelity",
        "mlp_down": "mlp_fidelity",
        "in_proj_qkv": "gdn_qkv_fidelity",
        "in_proj_z": "gdn_proj_fidelity",
        "in_proj_ab": "gdn_state_fidelity",
        "out_proj": "gdn_proj_fidelity",
    }

    def weight_dtype(self, role: str):
        return getattr(self, self._WEIGHT_FIELD[role])

    def fidelity(self, role: str, decode: bool = False):
        if decode and self.decode_fidelity is not None:
            # The state-arithmetic roles keep their own fidelity: ``in_proj_ab`` feeds the
            # recurrence's decay and is four output tiles, so there is nothing to trade.
            if role != "in_proj_ab":
                return self.decode_fidelity
        return getattr(self, self._FIDELITY_FIELD[role])

    def fp32_acc(self, role: str, decode: bool = False) -> bool:
        """Whether this role's matmul accumulates in float32 destination registers.

        The two roles whose *output* is a float32 tensor the recurrence carries always do; the roles
        in :attr:`prefill_fp32_acc_roles` do at prefill only; :attr:`fp32_dest_acc_all` restores the
        fused stage's blanket setting for the baseline arm.
        """
        if self.fp32_dest_acc_all or role in ("in_proj_qkv", "in_proj_ab"):
            return True
        return not decode and role in self.prefill_fp32_acc_roles


#: The policy every construction uses unless the caller passes another one.
DEFAULT_POLICY = PrecisionPolicy()

#: The fused stage's policy, expressed in this stage's vocabulary.  This is the baseline the
#: candidate table compares against, and ``tests/test_optimized_decoder.py`` uses it to prove the
#: optimized code reproduces the fused numbers when the policy is put back - so a win cannot be a
#: measurement artefact of a different harness.
FUSED_BASELINE_POLICY = PrecisionPolicy(
    name="fused-baseline",
    attn_weight=ttnn.bfloat16,
    attn_fidelity=HIFI4,
    mlp_weight=ttnn.bfloat16,
    mlp_down_weight=ttnn.bfloat16,
    mlp_fidelity=HIFI4,
    gdn_qkv_weight=ttnn.bfloat16,
    gdn_qkv_fidelity=HIFI4,
    gdn_z_weight=ttnn.bfloat16,
    gdn_out_weight=ttnn.bfloat16,
    gdn_proj_fidelity=HIFI4,
    gdn_state_fidelity=HIFI4,
    kv_cache=ttnn.bfloat16,
    decode_fidelity=None,
    fp32_dest_acc_all=True,
    prefill_fp32_acc_roles=(),
)

#: BFP8 everywhere a block-float weight is legal - the conservative fallback, and the arm the
#: shipped policy's one BFP4 group is measured against.  It is a *supported* configuration, not a
#: hypothetical: ``test_bfp8_policy_also_passes`` runs it on real weights, so a later stage that
#: wants more accuracy headroom can select it without rediscovering whether it works.  It costs
#: about 5 % of traced decode and 3-6 % of prefill against the shipped policy.
BFP8_POLICY = PrecisionPolicy(
    name="bfp8-all-lofi",
    mlp_weight=ttnn.bfloat8_b,
)


# ------------------------------------------------------------------------ geometry


#: Per-role ``in0_block_w`` overrides.  Empty on purpose: the shipped value is **computed** at
#: construction as the largest legal divisor of the role's ``K_tiles / cores`` whose weight and
#: activation blocks fit the L1 budget, and :meth:`OptimizedDecoder._decode_program_cfg` is where
#: that lives.  A hard-coded table would go stale the moment the core count or a weight dtype moved -
#: it did, once, and the sweep that was supposed to validate it silently measured a configuration the
#: layer never ran, because the constructor clamped the table down and the sweep did not.
#:
#: ``doc/optimized_decoder/probes/probe_optimized.py geometry`` sweeps every legal divisor for every
#: role, one role at a time, *starting from the computed baseline read off a built layer*, and
#: ``work_log.md`` section 3.2 is that table.  An entry here is only needed when the measurement
#: disagrees with the computed choice, and the work log says so when it does.
DEFAULT_IN0_BLOCK_W: dict = {}


@dataclass(frozen=True)
class DecodeGeometry:
    """Layout and program-config geometry of the decode path.

    ``cores`` is the single width-shard core count the whole decode path runs on: the residual
    stream, both RMS norms, every projection's activation and output, the attention epilogue and
    the MLP intermediate.  One number rather than one per role, because a per-role optimum buys a
    reshard between every pair of ops while the measured difference between the best per-role core
    count and a shared one is inside the run-to-run spread (``work_log.md`` section 3.2).

    It must divide the tile count of **every** ``K`` *and* every ``N`` the layer projects over -
    ``hidden_size`` (160 tiles), ``num_heads * head_dim`` / ``value_dim`` (192), the packed QKV or
    conv width (256 / 320), ``intermediate_size`` (544) and ``2 * intermediate_size`` (1088) -
    whose greatest common divisor is 32, so the legal candidates are 8, 16 and 32 on this device.
    :meth:`OptimizedDecoder._legal_stream_cores` recomputes that set from the real shapes and
    rejects an illegal value rather than silently resharding.

    ``in0_block_w`` is per role because the legal maximum is ``K_tiles / cores`` *and* what fits
    L1 alongside ``in0_block_w * per_core_N`` tiles of weight, which depends on ``N`` and on the
    weight dtype (OPT-004 / OPT-014).
    """

    cores: int = 32
    #: ``None`` means "use the measured :data:`DEFAULT_IN0_BLOCK_W`"; a dict overrides it per role
    #: and a role missing from a non-``None`` dict falls back to the conservative computed value.
    #: An empty dict therefore means "compute every role", which is what a ``cores`` sweep wants -
    #: the measured defaults are for ``cores=16`` and are not legal at every core count.
    in0_block_w: Optional[dict] = None
    #: Run the decode MLP as two ``[hidden, intermediate]`` matmuls (the gate with its SiLU fused
    #: into the matmul's own epilogue, then the up projection) instead of one packed
    #: ``[hidden, 2 * intermediate]`` matmul plus two slices.  Decode only; prefill keeps the
    #: packed form unless :attr:`split_gate_up_prefill` is set (OPT-010).
    split_gate_up_decode: bool = True
    #: Run the *prefill* MLP split as well.  When both are set the packed weight is never built.
    #:
    #: This **reverses** the fused stage's finding, and the reversal is the point of re-measuring it
    #: (OPT-010): stage 2 measured the packed form 51 % faster at 2048 rows, with a DRAM-interleaved
    #: bfloat16 weight and ``ttnn.linear``'s own heuristic on both sides.  At this stage's dtypes and
    #: layout - BFP4 weights width-sharded across the DRAM banks, an explicit 2D program config on
    #: both sides - the split form wins: 19.385 ms against 19.621 (``linear_attention``) and 9.959
    #: against 10.151 (``full_attention``).  The two slices of a 34816-wide tensor did not get
    #: cheaper; the matmul they were amortising did, so they stopped paying for themselves.  It also
    #: makes the layer *smaller*: with both phases split, the packed ``[hidden, 2*intermediate]``
    #: weight is never built, which is about 89 MB per layer at BFP4.
    split_gate_up_prefill: bool = True
    #: Use the DRAM-sharded program config at all.  ``False`` restores the fused stage's
    #: interleaved decode matmuls, which is how the candidate table isolates layout from dtype.
    dram_sharded: bool = True
    #: Keep the decode residual stream width-sharded in L1 across both norms and both residual
    #: adds.  ``False`` restores the fused stage's DRAM-interleaved residual (OPT-003).
    sharded_stream: bool = True
    #: Fuse the SiLU into the split gate matmul's epilogue.  Turned off automatically if this
    #: build's program config rejects a fused activation, and then the SiLU rides the following
    #: multiply exactly as it does in the packed form.
    fuse_gate_silu: bool = False
    #: Cores per head-batch for ``paged_scaled_dot_product_attention_decode``.
    #:
    #: Stage 1 pinned this to **1** and said so in as many words: "a real decode-latency cost - 4
    #: active cores instead of 64 - and it is a correctness-first choice, not a tuned one", because
    #: the kernel's cross-core tree reduction was wrong for most positions and it handed the kernel
    #: fix to this stage.  This stage took the hand-off: the defect is a DEST-register bounds
    #: violation in the fused SFPU softmax correction under float32 destination accumulation, it is
    #: fixed in
    #: ``ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp``,
    #: and the whole diagnosis, correctness sweep and blast-radius run are in
    #: ``doc/optimized_decoder/sdpa/AUTOFIX_SDPA.md``.
    #:
    #: 8 is the measured choice, not the largest legal one.  At ``k_chunk`` 512 over the eight
    #: positions stage 1 characterised (1023 ... 262143) the device/float32-golden scale is
    #: 0.9929-1.0011 at 8 cores and 0.9933-0.9999 at 4, against 0.9951-1.0168 at 1 core - so both
    #: multi-core settings are *more* accurate than the pinned one as well as 3.5x faster on the op
    #: (9774 us -> 2795 us at position 262143), and in the whole traced ``full_attention`` decode step
    #: 8 cores measures 0.9794 ms against 0.9907 at 4 and 1.0344 at 1
    #: (``logs/probe_optimized_geometry.log``).  16 cores exceeds L1 at this k chunk, which is a
    #: pre-existing limit identical on the stock build.  2 cores is *avoided*: with more than one core
    #: the program factory's float32 flash accumulators stay off and the residual bf16 merge error
    #: tracks the **per-core** merge count, which at 2 cores is 256 and lands at 0.980 - inside the
    #: bar but on its edge, where 8 cores' 64 merges are not.
    #:
    #: At the advertised ``max_batch`` of 32 this setting is inert by construction: the op allocates
    #: cores per head-batch out of a fixed grid, and ``batch * num_kv_heads`` = 128 already exceeds
    #: the grid, so every head-batch gets one core whatever is requested.  The batch-32 SDPA win in
    #: this stage comes from the ``bfloat8_b`` cache halving the bytes it reads, not from this field.
    sdpa_cores_per_head: int = 8
    #: Compute grid the decode SDPA program config is given.  ``None`` keeps stage 1's 8x8, which is
    #: what the correctness sweep above was measured on; a tuple asks for a larger grid, which
    #: matters at the advertised ``max_batch`` where ``batch * num_kv_heads`` exceeds 64.
    sdpa_grid: Optional[tuple] = None

    def block_w(self, role: str, computed: int) -> int:
        """The ``in0_block_w`` for ``role``: an explicit override, else the computed value."""
        if self.in0_block_w is None:
            return DEFAULT_IN0_BLOCK_W.get(role, computed)
        return self.in0_block_w.get(role, computed)


DEFAULT_GEOMETRY = DecodeGeometry()

#: The fused stage's decode layout, for the candidate table's "dtype only" arm.  Includes stage 1's
#: one-core-per-head SDPA, so this arm really is the old decode path.
FUSED_BASELINE_GEOMETRY = DecodeGeometry(
    split_gate_up_decode=False,
    dram_sharded=False,
    sharded_stream=False,
    fuse_gate_silu=False,
    sdpa_cores_per_head=1,
)


@dataclass(frozen=True)
class PrefillGeometry:
    """Explicit 2D matmul program configs for the large prefill projections.

    ``grids`` maps a role to a ``(x, y)`` compute grid; a role that is absent (or an empty map)
    keeps ``ttnn.linear``'s own heuristic, which on this device already spreads these matmuls over
    109-110 of 110 cores.  The candidate table is ``work_log.md`` section 3.4.
    """

    grids: Optional[dict] = None
    in0_block_w: Optional[dict] = None

    def grid(self, role: str):
        return (self.grids or {}).get(role)

    def block_w(self, role: str):
        return (self.in0_block_w or {}).get(role)


DEFAULT_PREFILL_GEOMETRY = PrefillGeometry()


# ------------------------------------------------------------------------ helpers


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _divisors(value: int) -> list:
    return [d for d in range(1, value + 1) if value % d == 0]


def _mem_config_eq(left, right) -> bool:
    """``==`` on two memory configs, tolerating a binding that does not implement it."""
    try:
        return bool(left == right)
    except Exception:  # pragma: no cover - defensive; ttnn implements __eq__ today
        return False


#: Ceiling on ``out_block_h * out_block_w`` for a prefill 2D matmul, in output tiles.
#:
#: The output circular buffer holds one output block, double-buffered, and this device has 1.5 MB of
#: L1 per core.  160 bfloat16 tiles is 328 KB and 160 float32 tiles is 655 KB, which leaves room for
#: the double-buffered ``in0``/``in1`` blocks alongside; the ``2048 x 5120 x 34816`` prefill matmul's
#: natural ``per_core_M x per_core_N`` is 8 x 136 = 1088 tiles, which is 2.2 MB and does not fit, so
#: this bound is what makes the 2D config legal rather than a nicety.
_PREFILL_OUT_BLOCK_TILES = 160

#: Largest ``in0_block_w`` a prefill 2D matmul may take, and the L1 budget its block circular
#: buffers must fit.  8 is where the measured curve flattens: 2 is 5 % slower and 16 is not a divisor
#: of every ``K`` here.
_PREFILL_MAX_BLOCK_W = 8
_PREFILL_L1_BUDGET = 1_100_000

#: Bytes one tile occupies in each dtype, used only to size the *fallback* ``in0_block_w``
#: estimate.  A shipped value comes from :data:`DEFAULT_IN0_BLOCK_W`, which is measured.
_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}

#: Construction-time hand-off for :meth:`OptimizedDecoder.from_state_dict`.  The inherited loader
#: calls ``cls(...)`` with a fixed keyword set, so the policy and geometry reach ``__init__``
#: through here rather than through a widened base-class signature.  Setup is single-threaded and
#: the hand-off is cleared in a ``finally``.
_CONSTRUCTION: dict = {}


class OptimizedDecoder(FusedDecoder):
    """One Qwen3.5/3.6 decoder layer on a TTNN mesh device, optimized for this hardware.

    Public behaviour is identical to :class:`~.fused_decoder.FusedDecoder`; see the module
    docstring for what changed underneath.
    """

    #: Ops the optimized path must really dispatch on top of :attr:`FusedDecoder.FUSED_OPS`, so a
    #: silent fall back to the fused layout (an interleaved decode matmul, a DRAM residual) is a
    #: test failure rather than a quiet regression.  ``tests/test_optimized_decoder.py`` asserts
    #: these from the *device* report as well as from the python call trace, because a python-level
    #: trap cannot see which program config a ``ttnn.linear`` chose.
    OPTIMIZED_OPS = ("ttnn.linear", "ttnn.rms_norm")

    WEIGHT_KEY = {role: role for role in ROLES}

    # ------------------------------------------------------------------ setup

    def __init__(self, **kwargs):
        self.policy = kwargs.pop("policy", None) or _CONSTRUCTION.get("policy") or DEFAULT_POLICY
        self.decode_geometry = (
            kwargs.pop("decode_geometry", None) or _CONSTRUCTION.get("decode_geometry") or DEFAULT_GEOMETRY
        )
        self.prefill_geometry = (
            kwargs.pop("prefill_geometry", None) or _CONSTRUCTION.get("prefill_geometry") or DEFAULT_PREFILL_GEOMETRY
        )
        #: ``True`` only inside :meth:`decode_forward`.  Phase is passed explicitly rather than
        #: inferred from a row count, because a prefill chunk of 32 padded rows has the same row
        #: count as a decode step at ``max_batch`` 32.
        self._decoding = False
        super().__init__(**kwargs)
        self._build_runtime_configs()

    # -- configuration ----------------------------------------------------

    def _role_shapes(self) -> dict:
        """``role -> (K, N)`` for every role this layer actually runs."""
        s = self.shapes
        shapes = {
            "mlp_down": (s.intermediate_size, s.hidden_size),
        }
        if self.decode_geometry.split_gate_up_decode or self.decode_geometry.split_gate_up_prefill:
            shapes["mlp_gate"] = (s.hidden_size, s.intermediate_size)
            shapes["mlp_up"] = (s.hidden_size, s.intermediate_size)
        if not (self.decode_geometry.split_gate_up_decode and self.decode_geometry.split_gate_up_prefill):
            shapes["mlp_gate_up"] = (s.hidden_size, 2 * s.intermediate_size)
        if s.layer_type == FULL_ATTENTION:
            qkv_n = s.num_attention_heads * s.head_dim + 2 * s.num_key_value_heads * s.head_dim
            shapes.update(
                {
                    "wqkv": (s.hidden_size, qkv_n),
                    "wgate": (s.hidden_size, s.num_attention_heads * s.head_dim),
                    "o_proj": (s.num_attention_heads * s.head_dim, s.hidden_size),
                }
            )
        else:
            shapes.update(
                {
                    "in_proj_qkv": (s.hidden_size, s.conv_dim),
                    "in_proj_z": (s.hidden_size, s.value_dim),
                    "in_proj_ab": (s.hidden_size, 2 * _AB_STRIDE),
                    "out_proj": (s.value_dim, s.hidden_size),
                }
            )
        return shapes

    def _build_runtime_configs(self) -> None:
        """Derive every memory config, program config and compute-kernel config from the policy.

        Called from :meth:`__init__`, so a directly-constructed layer and one built by
        :meth:`from_state_dict` agree.
        """
        s = self.shapes
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.grid = grid
        self.role_shapes = self._role_shapes()
        self.role_kernel_cfg = {
            role: _kernel_cfg(self.policy.fidelity(role), self.policy.fp32_acc(role)) for role in self.role_shapes
        }
        self.role_kernel_cfg_decode = {
            role: _kernel_cfg(self.policy.fidelity(role, decode=True), self.policy.fp32_acc(role, decode=True))
            for role in self.role_shapes
        }
        # Everything that is not a weight projection - the recurrence matmuls, the gated-norm
        # constant matmuls, chunk_gated_delta_rule, the norms, RoPE - keeps the fused stage's
        # HiFi4 + float32-destination contract.  ``self.sdpa_compute_cfg`` is left strictly alone:
        # the ``sdpa_decode`` float32-accumulator fix this branch carries is gated on
        # ``fp32_dest_acc_en``, so lowering it would silently reintroduce a 37x scale error.
        self.compute_cfg = _kernel_cfg(self.policy.gdn_state_fidelity, fp32_dest_acc_en=True)

        # Decode SDPA: same q/k chunk and the same fp32-destination compute config stage 1 pinned,
        # with the core count this stage's kernel fix made usable (see
        # ``DecodeGeometry.sdpa_cores_per_head``).
        if s.layer_type == FULL_ATTENTION:
            sdpa_grid = self.decode_geometry.sdpa_grid or (8, 8)
            self.sdpa_decode_program_cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*sdpa_grid),
                q_chunk_size=32,
                k_chunk_size=SDPA_DECODE_K_CHUNK,
                exp_approx_mode=False,
                max_cores_per_head_batch=self.decode_geometry.sdpa_cores_per_head,
            )

        self.decode_rows = _round_up(self.max_batch, ttnn.TILE_SIZE)
        cores = self.decode_geometry.cores
        legal = self._legal_stream_cores()
        if cores not in legal:
            raise ValueError(
                f"decode_geometry.cores={cores} does not divide the tile count of every width "
                f"this {s.layer_type} layer's decode path shards; legal values here are {legal}"
            )
        self.decode_cores = cores
        self.decode_core_range = ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True)

        self._stream_cfgs: dict = {}
        self.decode_stream_mem_cfg = self._stream_cfg(s.hidden_size)

        # The decode RMS norms run on the stream's own shard grid, so the norm reshards neither
        # its input nor its output (OPT-003).
        block_w = s.hidden_size // cores // ttnn.TILE_SIZE
        self.decode_norm_mem_cfg = self.decode_stream_mem_cfg
        self.decode_norm_prgm_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=_largest_divisor_at_most(block_w, 4),
            block_h=self.decode_rows // ttnn.TILE_SIZE,
            block_w=block_w,
            inplace=False,
        )

        dram = self.mesh_device.dram_grid_size()
        self.dram_banks = dram.x
        self.dram_shard_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))}
        )
        self.weight_mem_cfg: dict = {}
        self.decode_program_cfg: dict = {}
        self.decode_in_cfg: dict = {}
        self.decode_out_cfg: dict = {}
        for role, (k, n) in self.role_shapes.items():
            if not self._role_is_dram_sharded(role):
                continue
            self.weight_mem_cfg[role] = self._dram_sharded_weight_cfg(k, n)
            self.decode_in_cfg[role] = self._stream_cfg(k)
            self.decode_out_cfg[role] = self._stream_cfg(n)
            self.decode_program_cfg[role] = self._decode_program_cfg(role, k, n)

        # The split gate matmul's SiLU epilogue, if this build's program config accepts one.
        self.gate_silu_program_cfg = None
        if self.decode_geometry.fuse_gate_silu and "mlp_gate" in self.decode_program_cfg:
            self.gate_silu_program_cfg = self._with_activation(
                self.decode_program_cfg["mlp_gate"], ttnn.UnaryOpType.SILU
            )

        self._prefill_pc_cache: dict = {}

    def _legal_stream_cores(self) -> list:
        """Core counts that width-shard every activation width the decode path carries.

        A shared activation stream is only free if the *output* of one projection is a legal
        *input* shard of the next, so both sides of every role have to divide.
        """
        widths = {k for k, _ in self.role_shapes.values()} | {n for _, n in self.role_shapes.values()}
        # ``in_proj_ab``'s 4-tile output never joins the stream: it is sliced into two 48-column
        # tensors and consumed by the recurrence's decay.
        widths.discard(2 * _AB_STRIDE)
        limit = min(self.grid.x * self.grid.y, min(widths) // ttnn.TILE_SIZE)
        return [c for c in range(1, limit + 1) if all((w // ttnn.TILE_SIZE) % c == 0 for w in widths)]

    def _role_is_dram_sharded(self, role: str) -> bool:
        if not self.decode_geometry.dram_sharded:
            return False
        if not self.decode_geometry.sharded_stream:
            # The DRAM-sharded matmul *requires* a width-sharded L1 activation and produces one, so
            # "DRAM-interleaved residual with DRAM-sharded matmuls" is not a configuration - the
            # projections would hand a sharded tensor straight back into the residual add.  The two
            # flags are therefore one lever in the off direction: turning the sharded stream off
            # restores the fused stage's interleaved decode matmuls with it.  (Turning only the
            # matmuls off *is* a coherent arm - a sharded residual with interleaved projections - and
            # the candidate table measures it.)
            return False
        if role == "in_proj_ab":
            # Four output tiles with a fused bias row.  The DRAM-sharded matmul has no bias slot to
            # fold ``dt_bias`` into, so this role keeps the fused stage's measured ``core_grid``
            # form; the alternative is measured in ``work_log.md`` section 3.5.
            return False
        if role == "mlp_gate_up" and self.decode_geometry.split_gate_up_decode:
            return False
        if role in ("mlp_gate", "mlp_up") and not self.decode_geometry.split_gate_up_decode:
            return False
        return True

    def _stream_cfg(self, width: int) -> ttnn.MemoryConfig:
        """Width-sharded L1 config for a ``[1, 1, decode_rows, width]`` decode tensor."""
        if width not in self._stream_cfgs:
            self._stream_cfgs[width] = ttnn.create_sharded_memory_config(
                shape=(self.decode_rows, width // self.decode_cores),
                core_grid=self.decode_core_range,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        return self._stream_cfgs[width]

    def _dram_sharded_weight_cfg(self, k: int, n: int) -> ttnn.MemoryConfig:
        """Width-shard a ``[K, N]`` weight across this chip's DRAM banks."""
        padded = math.ceil(n / (ttnn.TILE_SIZE * self.dram_banks)) * ttnn.TILE_SIZE * self.dram_banks
        spec = ttnn.ShardSpec(self.dram_shard_grid, (k, padded // self.dram_banks), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)

    def _decode_program_cfg(self, role: str, k: int, n: int):
        """DRAM-sharded decode program config for one role.

        ``in0_block_w`` comes from :data:`DEFAULT_IN0_BLOCK_W` (measured) when the role is listed
        there, and otherwise from the largest legal divisor of ``K_tiles / cores`` whose weight
        block still fits a conservative L1 estimate - a fallback, so an unmeasured shape gets a
        value that allocates rather than one that is fast.
        """
        cores = self.decode_cores
        k_tiles = k // ttnn.TILE_SIZE
        per_core_m = self.decode_rows // ttnn.TILE_SIZE
        per_core_n = (n // ttnn.TILE_SIZE) // cores
        weight_bytes = _TILE_BYTES.get(self.policy.weight_dtype(role), 2048)
        act_bytes = _TILE_BYTES[ttnn.bfloat16]
        # The circular buffers have to fit L1 *alongside* the width-sharded activations that live
        # there for the whole decode step - the measured failure is
        # "Statically allocated circular buffers ... clash with L1 buffers", not a bare CB-size
        # overflow - so the budget is well below this device's 1.5 MB of L1 per core.
        budget = 350_000
        fallback = 1
        for candidate in _divisors(k_tiles // cores):
            l1 = 2 * candidate * per_core_m * act_bytes + 2 * candidate * per_core_n * weight_bytes
            l1 += per_core_m * per_core_n * act_bytes
            if l1 <= budget:
                fallback = candidate
        block_w = self.decode_geometry.block_w(role, fallback)
        if not self.decode_geometry.in0_block_w and block_w != fallback:
            # No explicit override, so the computed value is authoritative: it already accounts for
            # this role's weight dtype, whose tile size sets the weight block's L1 cost.  An explicit
            # override is honoured verbatim instead, because a sweep wants the allocation failure
            # reported rather than silently replaced by something else.
            block_w = fallback
        if (k_tiles // cores) % block_w:
            raise ValueError(
                f"in0_block_w={block_w} for role {role!r} does not divide the input shard's "
                f"{k_tiles // cores} K tiles at cores={cores}"
            )
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=block_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            fused_activation=None,
        )

    @staticmethod
    def _with_activation(program_cfg, activation):
        """Copy a DRAM-sharded program config with a fused activation, or ``None`` if unsupported.

        The ``fused_activation`` field's accepted python type differs between tt-metal revisions,
        so the two forms are tried and a build that accepts neither simply keeps the activation on
        the op that follows.
        """
        for value in (activation, [activation]):
            try:
                return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=program_cfg.in0_block_w,
                    per_core_M=program_cfg.per_core_M,
                    per_core_N=program_cfg.per_core_N,
                    fused_activation=value,
                )
            except Exception:  # noqa: BLE001 - a rejected type is the signal, not an error
                continue
        return None

    def _prefill_program_cfg(self, role: str, rows: int):
        """Explicit 2D program config for a prefill projection, or ``None`` for the heuristic.

        A config is **required**, not optional, for every role whose weight lives in a DRAM
        width-sharded memory config, because ``ttnn.linear``'s own heuristic falls back to
        ``MatmulMultiCoreProgramConfig`` for a sharded ``in1`` and that config rejects a sharded B
        outright (``matmul_device_operation.cpp``: "Input B memory layout must be INTERLEAVED").
        ``MatmulMultiCoreReuseMultiCastProgramConfig`` is the config that accepts a DRAM
        width-sharded B, so the decode-side DRAM-sharded weight and the prefill-side 2D matmul are
        two halves of one decision: it is what lets both phases read **one** copy of each weight
        rather than a DRAM-interleaved copy for prefill and a DRAM-sharded copy for decode, which at
        this model's weight sizes would be about 210 MB per layer of duplication.

        The geometry is derived rather than tabulated, because prefill runs at every padded chunk
        length the public API allows, not just at 2048 rows:

        * the column count divides the output tile count so ``per_core_N`` is exact;
        * the row count divides the input tile count so ``per_core_M`` is exact - at short lengths
          that means fewer rows of cores, which is correct: there is no work for them;
        * ``out_block_h`` is the largest divisor of ``per_core_M`` whose output block stays under
          :data:`_PREFILL_OUT_BLOCK_TILES`, which is what keeps the output circular buffer inside L1
          for a 34816-wide or float32 output;
        * ``out_subblock_h * out_subblock_w`` stays within the destination-register budget, halved
          for the two roles that accumulate in float32.

        :class:`PrefillGeometry` overrides the grid and ``in0_block_w`` per role, which is how the
        candidate table in ``work_log.md`` section 3.4 sweeps them.
        """
        if role not in self.weight_mem_cfg and self.prefill_geometry.grid(role) is None:
            return None
        key = (role, rows)
        if key in self._prefill_pc_cache:
            return self._prefill_pc_cache[key]
        k, n = self.role_shapes[role]
        k_tiles, n_tiles = k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        m_tiles = max(1, math.ceil(rows / ttnn.TILE_SIZE))
        override = self.prefill_geometry.grid(role)
        if override is not None:
            x, y = override
        elif role in self.weight_mem_cfg:
            # The weight is width-sharded across this chip's DRAM banks, so one column of compute
            # cores must line up with exactly one bank's shard.  Measured, not assumed: with the
            # column count set to any other divisor of the output tile count the matmul returns
            # non-finite values rather than failing validation - `x = 10` on the 160-tile-wide
            # `o_proj` / `mlp_down` / `out_proj` and the 320-tile-wide `in_proj_qkv` all produced
            # NaN, while `x = 8` (= the bank count) is correct to PCC 0.99937 against the
            # heuristic's own output on the same weights
            # (``logs/probe_prefill_grid_alignment.log``).  Every ``N`` this model projects to is a
            # multiple of 8 tiles, so ``per_core_N`` stays exact.
            x = self.dram_banks
            y = _largest_divisor_at_most(m_tiles, self.grid.y)
            if n_tiles % x:
                raise ValueError(
                    f"role {role!r} has {n_tiles} output tiles, which is not a multiple of this "
                    f"chip's {x} DRAM banks; a DRAM width-sharded weight needs one bank per compute "
                    "column"
                )
        else:
            x = _largest_divisor_at_most(n_tiles, self.grid.x)
            y = _largest_divisor_at_most(m_tiles, self.grid.y)
        per_core_m = math.ceil(m_tiles / y)
        per_core_n = math.ceil(n_tiles / x)
        out_block_w = per_core_n
        out_block_h = 1
        for candidate in _divisors(per_core_m):
            if candidate * out_block_w <= _PREFILL_OUT_BLOCK_TILES:
                out_block_h = candidate
        # ``in0_block_w`` up to :data:`_PREFILL_MAX_BLOCK_W`, largest first, subject to the block
        # circular buffers fitting L1.  Prefill has no resident width-sharded activations competing
        # for L1, so the budget is much larger than the decode one - but not unlimited: measured,
        # ``in0_block_w=8`` on every projection is 1.5 % faster for ``full_attention`` and raises
        # "Statically allocated circular buffers ... beyond max L1 size" for ``linear_attention``,
        # whose ``in_proj_qkv`` has a float32 output block and a BFP8 weight
        # (``logs/probe_optimized_prefill.log``).  That is the one role this bound holds back, and it
        # is the reason the bound is computed per role rather than set globally.
        weight_bytes = _TILE_BYTES.get(self.policy.weight_dtype(role), 2048)
        out_bytes = _TILE_BYTES[ttnn.float32 if self.policy.fp32_acc(role) else ttnn.bfloat16]
        block_w = 1
        for candidate in _divisors(k_tiles):
            if candidate > _PREFILL_MAX_BLOCK_W:
                break
            l1 = 2 * candidate * out_block_h * _TILE_BYTES[ttnn.bfloat16]
            l1 += 2 * candidate * out_block_w * weight_bytes
            l1 += out_block_h * out_block_w * out_bytes
            if l1 <= _PREFILL_L1_BUDGET:
                block_w = candidate
        block_w = self.prefill_geometry.block_w(role) or block_w
        budget = 2 if self.policy.fp32_acc(role) else 4
        subblock_w = _largest_divisor_at_most(out_block_w, budget)
        subblock_h = _largest_divisor_at_most(out_block_h, max(1, budget // subblock_w))
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(x, y),
            in0_block_w=block_w,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock_w,
            out_block_h=out_block_h,
            out_block_w=out_block_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
        )
        self._prefill_pc_cache[key] = cfg
        return cfg

    # -- construction -----------------------------------------------------

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        policy: Optional[PrecisionPolicy] = None,
        decode_geometry: Optional[DecodeGeometry] = None,
        prefill_geometry: Optional[PrefillGeometry] = None,
        **kwargs,
    ) -> "OptimizedDecoder":
        """Build the optimized layer from an HF submodule-relative state dict.

        The projection weights are re-derived from ``state_dict`` at the policy's dtypes and in the
        DRAM width-sharded memory config the decode program configs need, replacing the bfloat16
        DRAM-interleaved tensors the inherited loader built.  Doing it this way, rather than
        threading a dtype map through the base loader, keeps one source of truth for *how* each
        weight is assembled (the ``q``/gate split, the ``gate``/``up`` concat, the ``a``/``b`` pack)
        in the stage that introduced it.

        This is the only place ``torch`` is used.
        """
        import torch  # setup-time only; never on the prefill/decode path

        policy = policy or DEFAULT_POLICY
        kwargs.setdefault("cache_dtype", policy.kv_cache)
        _CONSTRUCTION.update(policy=policy, decode_geometry=decode_geometry, prefill_geometry=prefill_geometry)
        try:
            layer = super().from_state_dict(
                state_dict, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device, **kwargs
            )
        finally:
            _CONSTRUCTION.clear()
        layer._retype_projection_weights(state_dict, torch)
        return layer

    def _retype_projection_weights(self, state_dict, torch) -> None:
        """Replace every projection weight with a policy-dtype, DRAM-sharded one.

        Each replacement is built, installed and the old tensor freed before the next one starts,
        so the extra device DRAM this costs at load is one weight rather than all of them.
        """
        s = self.shapes

        def get(name):
            if name not in state_dict:
                raise KeyError(f"missing weight {name!r} for layer {s.layer_idx} ({s.layer_type})")
            return state_dict[name].to(torch.float32)

        def install(role, tensor):
            if role not in self.role_shapes:
                return
            expected = self.role_shapes[role]
            assert tuple(tensor.shape) == expected, f"{role}: built {tuple(tensor.shape)}, expected {expected}"
            new = ttnn.from_torch(
                tensor.contiguous().reshape(1, 1, *tensor.shape),
                dtype=self.policy.weight_dtype(role),
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=self.weight_mem_cfg.get(role, ttnn.DRAM_MEMORY_CONFIG),
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            old = self.w.pop(role, None)
            self.w[role] = new
            if old is not None and old.is_allocated():
                ttnn.deallocate(old)

        gate_w = get("mlp.gate_proj.weight")
        up_w = get("mlp.up_proj.weight")
        install("mlp_gate", gate_w.t())
        install("mlp_up", up_w.t())
        if "mlp_gate_up" in self.role_shapes:
            install("mlp_gate_up", torch.cat([gate_w, up_w], dim=0).t())
        elif "mlp_gate_up" in self.w:
            # Both phases run the split form, so the packed weight the inherited loader built is
            # dead: free it rather than hold a second copy of the MLP's first projection - about
            # 89 MB per layer at BFP4 - that nothing reads.  The concatenation is not even built.
            ttnn.deallocate(self.w.pop("mlp_gate_up"))
        install("mlp_down", get("mlp.down_proj.weight").t())

        if s.layer_type == FULL_ATTENTION:
            n_heads, head_dim = s.num_attention_heads, s.head_dim
            q_full = get("self_attn.q_proj.weight").reshape(n_heads, 2 * head_dim, -1)
            q_only = q_full[:, :head_dim, :].reshape(n_heads * head_dim, -1)
            gate_only = q_full[:, head_dim:, :].reshape(n_heads * head_dim, -1)
            install(
                "wqkv",
                torch.cat([q_only, get("self_attn.k_proj.weight"), get("self_attn.v_proj.weight")], dim=0).t(),
            )
            install("wgate", gate_only.t())
            install("o_proj", get("self_attn.o_proj.weight").t())
        else:
            install("in_proj_qkv", get("linear_attn.in_proj_qkv.weight").t())
            install("in_proj_z", get("linear_attn.in_proj_z.weight").t())
            install("out_proj", get("linear_attn.out_proj.weight").t())

    # ------------------------------------------------------------- primitives

    def _project(self, x, role: str, *, decode: bool, dtype, program_cfg=None):
        """One projection, with this stage's memory/program/compute config for ``role``.

        Decode takes the DRAM-sharded path when the role has one, resharding its input only if the
        caller did not already hand it the stream layout - which, on the shipped configuration, it
        always does.  Prefill takes the explicit 2D program config when one is configured and
        ``ttnn.linear``'s heuristic otherwise.
        """
        weight = self.w[self.WEIGHT_KEY[role]]
        kernel_cfg = self.role_kernel_cfg_decode[role] if decode else self.role_kernel_cfg[role]
        if decode and role in self.decode_program_cfg:
            wanted = self.decode_in_cfg[role]
            sharded = x if _mem_config_eq(x.memory_config(), wanted) else ttnn.to_memory_config(x, wanted)
            out = ttnn.linear(
                sharded,
                weight,
                program_config=program_cfg or self.decode_program_cfg[role],
                memory_config=self.decode_out_cfg[role],
                dtype=dtype,
                compute_kernel_config=kernel_cfg,
            )
            _free(sharded, x, out)
            return out
        return ttnn.linear(
            x,
            weight,
            dtype=dtype,
            compute_kernel_config=kernel_cfg,
            program_config=None if decode else self._prefill_program_cfg(role, int(x.shape[-2])),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _decode_rms_norm(self, x, weight):
        """RMS norm of a width-sharded decode stream tensor: sharded in, sharded out.

        The fused stage brackets its norm with an ``InterleavedToSharded`` and a
        ``ShardedToInterleaved`` because its residual lives in DRAM.  Here the residual is already
        on the norm's own shard grid, so both disappear (OPT-003).  ``sharded_stream=False``
        restores the fused behaviour, which is how the candidate table isolates this change.
        """
        if not self.decode_geometry.sharded_stream:
            return super()._decode_rms_norm(x, weight)
        sharded = (
            x
            if _mem_config_eq(x.memory_config(), self.decode_norm_mem_cfg)
            else ttnn.to_memory_config(x, self.decode_norm_mem_cfg)
        )
        normed = ttnn.rms_norm(
            sharded,
            epsilon=self.shapes.rms_norm_eps,
            weight=weight,
            program_config=self.decode_norm_prgm_cfg,
            memory_config=self.decode_norm_mem_cfg,
            compute_kernel_config=self.compute_cfg,
        )
        _free(sharded, x, normed)
        return normed

    def _mlp(self, x):
        """SwiGLU MLP, split into two ``[hidden, intermediate]`` matmuls in **both** phases.

        This reverses the fused stage's choice, and the reversal is measured rather than assumed
        (``work_log.md`` section 3.3, OPT-010).  Stage 2 found the packed
        ``[hidden, 2 * intermediate]`` matmul 51 % faster at 2048 rows with a DRAM-interleaved
        bfloat16 weight and ``ttnn.linear``'s heuristic; at this stage's BFP4 weights, DRAM
        width-sharded layout and explicit program configs the split form wins at prefill
        (19.415 vs 19.634 ms and 9.908 vs 10.151 ms) and at decode it is the only legal form at the
        winning 32-core shard grid - the packed output is twice as wide, so its ``per_core_N``
        doubles and the circular buffers clash with the resident sharded activations.  At 16 cores,
        where both allocate, the split form still wins by 2-3 %.

        Splitting both phases also means the packed weight is never built, which makes the layer
        about 89 MB smaller per layer at BFP4.

        The SiLU stays on the multiply that consumes the gate, not on the gate matmul's own
        epilogue: fusing it there measured 3.5 % / 4.7 % *slower*
        (:attr:`DecodeGeometry.fuse_gate_silu`).  Both packed forms remain runnable, because the
        candidate table needs an arm that runs.
        """
        decode = self._decoding
        split = self.decode_geometry.split_gate_up_prefill if not decode else self.decode_geometry.split_gate_up_decode
        if split:
            gate = self._project(
                x,
                "mlp_gate",
                decode=decode,
                dtype=ttnn.bfloat16,
                program_cfg=self.gate_silu_program_cfg if decode else None,
            )
            up = self._project(x, "mlp_up", decode=decode, dtype=ttnn.bfloat16)
            fused_silu = decode and self.gate_silu_program_cfg is not None
            inner = ttnn.multiply(
                gate,
                up,
                input_tensor_a_activations=[] if fused_silu else [ttnn.UnaryOpType.SILU],
                memory_config=gate.memory_config(),
            )
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            out = self._project(inner, "mlp_down", decode=decode, dtype=ttnn.bfloat16)
            ttnn.deallocate(inner)
            return out

        gate_up = self._project(x, "mlp_gate_up", decode=decode, dtype=ttnn.bfloat16)
        inter = self.shapes.intermediate_size
        lead = _shape(gate_up)[:3]
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [*lead, inter])
        up = ttnn.slice(gate_up, [0, 0, 0, inter], [*lead, 2 * inter])
        _free(gate_up, gate, up)
        inner = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = self._project(inner, "mlp_down", decode=decode, dtype=ttnn.bfloat16)
        ttnn.deallocate(inner)
        return out

    # ------------------------------------------------------- full attention

    def _attn_projections(self, x, *, decode: bool):
        """``(q, k, v, gate)``, with the two projections on this stage's configs.

        ``nlp_create_qkv_heads_decode`` needs an interleaved bfloat16 input, so the decode ``wqkv``
        output is un-sharded into **L1** (not DRAM), exactly as
        ``models/common/modules/attention/attention_1d.py`` does.  The ``gate`` output stays on the
        stream grid, because the epilogue's multiply and ``o_proj`` both consume it there.
        """
        s = self.shapes
        qkv = self._project(x, "wqkv", decode=decode, dtype=ttnn.bfloat16)
        gate = self._project(x, "wgate", decode=decode, dtype=ttnn.bfloat16)
        if decode:
            if qkv.memory_config().is_sharded():
                interleaved = ttnn.sharded_to_interleaved(qkv, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(qkv)
                qkv = interleaved
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
        """Sigmoid output gate + ``o_proj``, on the stream grid when the gate is already on it.

        The fused stage runs the gate multiply on DRAM-interleaved tensors and then ``o_proj`` from
        DRAM.  Here, when ``gate`` came back width-sharded, the attention output is brought onto
        the same shard grid, the multiply runs in L1 and ``o_proj`` reads that shard directly - so
        the whole epilogue costs one reshard of the attention output instead of two DRAM round
        trips.
        """
        if gate.memory_config().is_sharded():
            wanted = self._stream_cfg(int(gate.shape[-1]))
            sharded = (
                attn_out
                if _mem_config_eq(attn_out.memory_config(), wanted)
                else ttnn.to_memory_config(attn_out, wanted)
            )
            _free(attn_out, sharded)
            gated = ttnn.multiply(
                sharded, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=wanted
            )
            ttnn.deallocate(sharded)
            ttnn.deallocate(gate)
            out = self._project(gated, "o_proj", decode=True, dtype=ttnn.bfloat16)
            ttnn.deallocate(gated)
            return out
        gated = ttnn.multiply(attn_out, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(attn_out)
        ttnn.deallocate(gate)
        out = self._project(gated, "o_proj", decode=False, dtype=ttnn.bfloat16)
        ttnn.deallocate(gated)
        return out

    def _full_attention_decode(self, x, *, current_pos, page_table, rot_mats):
        """Attention for one decode step; ``x`` is the width-sharded normed stream tensor.

        Structurally the fused stage's - ``nlp_create_qkv_heads_decode``, interleaved Q/K norm and
        partial RoPE, ``paged_update_cache`` in the head op's own memory config,
        ``paged_scaled_dot_product_attention_decode``, ``nlp_concat_heads_decode`` - with the
        projections and the epilogue on this stage's configs, and with the concat result handed to
        the epilogue on the stream grid instead of through DRAM.
        """
        s = self.shapes
        cos, sin = rot_mats
        q, k, v, gate = self._attn_projections(x, decode=True)
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
        return self._attn_epilogue(concat, gate)

    # ----------------------------------------------------- linear attention

    def _gdn_inputs(self, x, *, raw_beta: bool = False, phase: str = "prefill"):
        """Shared GatedDeltaNet input projections on this stage's configs.

        ``in_proj_ab`` keeps the fused stage's interleaved, bias-folded, measured-``core_grid``
        form: it is four output tiles of float32 state arithmetic and the DRAM-sharded matmul has
        no bias slot for ``dt_bias``.  On the decode path that costs one ``ShardedToInterleaved``
        of the normed stream tensor - a 320 KB copy - and the alternative (a DRAM-sharded ``ab``
        plus a separate bias add) is measured in ``work_log.md`` section 3.5.
        """
        s = self.shapes
        decode = phase == "decode"
        mixed_qkv = self._project(x, "in_proj_qkv", decode=decode, dtype=ttnn.float32)
        z = self._project(x, "in_proj_z", decode=decode, dtype=ttnn.bfloat16)
        if mixed_qkv.memory_config().is_sharded():
            # The causal conv slices, typecasts and concatenates this tensor in ROW_MAJOR; it is
            # an interleaved-tensor chain by construction.
            interleaved = ttnn.sharded_to_interleaved(mixed_qkv, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(mixed_qkv)
            mixed_qkv = interleaved
        ab_in = x if not x.memory_config().is_sharded() else ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        ab = ttnn.linear(
            ab_in,
            self.w["in_proj_ab"],
            bias=self.w["in_proj_ab_bias"],
            dtype=ttnn.float32,
            compute_kernel_config=self.role_kernel_cfg["in_proj_ab"],
            core_grid=self.ab_matmul_grid[phase],
        )
        _free(ab_in, x, ab)
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
        soft = ttnn.softplus(a, beta=1.0, threshold=20.0)
        ttnn.deallocate(a)
        g = ttnn.multiply(soft, self.w["neg_exp_A"])
        ttnn.deallocate(soft)
        return mixed_qkv, z, beta, g

    def _gated_norm_and_project(self, core, z, rows: int, *, phase: str = "prefill"):
        """z-gated per-head RMS norm + ``out_proj``, on the flat token-major layout.

        Arithmetic identical to the fused stage's group reduction; only ``out_proj`` changes, to
        this stage's role config, which on the decode path puts the gated result on the stream grid
        first so the projection reads a shard rather than DRAM.
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
        out = self._project(gated, "out_proj", decode=phase == "decode", dtype=ttnn.bfloat16)
        ttnn.deallocate(gated)
        return out

    def _linear_attention_decode(self, x):
        """Single-token gated delta rule for all ``max_batch`` users at once.

        The recurrence is the fused stage's, op for op: float32 state arithmetic whose levers are
        core grids, already swept there and re-verified unchanged here.  This override exists for
        one reason - the fused version's **small-batch terminal projection** is a bare
        ``ttnn.linear`` on a DRAM-interleaved tensor, and it is the only place in either layer kind
        where a dominant decode matmul cannot be reached by overriding a helper.  Everything from
        the input projections down to ``out`` is delegated by calling the same helpers the fused
        layer calls (:meth:`_gdn_inputs`, :meth:`_gated_norm_and_project`), so the two
        implementations cannot drift silently; ``test_optimized_matches_fused`` proves it by
        running both at the fused baseline policy and comparing outputs.
        """
        s = self.shapes
        batch = self.max_batch
        nv = s.num_v_heads
        k_size = s.conv_kernel_size
        taps = self.w["conv_taps"]

        mixed_qkv, z, b_raw, g = self._gdn_inputs(x, raw_beta=True, phase="decode")

        acc = ttnn.multiply(mixed_qkv, taps[k_size - 1])
        for j in range(k_size - 1):
            row = self.conv_state_split[j + 1]
            if j == k_size - 2:
                term = ttnn.multiply(row, taps[j])
                merged = ttnn.add(acc, term, activations=[ttnn.UnaryOpType.SILU])
                ttnn.deallocate(term)
            else:
                merged = ttnn.addcmul(acc, row, taps[j])
            ttnn.deallocate(acc)
            acc = merged
        for j in range(k_size - 1):
            ttnn.copy(self.conv_state_split[j + 1], self.conv_state_split[j])
        ttnn.copy(mixed_qkv, self.conv_state_split[k_size - 1])
        ttnn.deallocate(mixed_qkv)
        conv_out = acc

        q_flat, k_flat, v_flat = self._split_qkv(conv_out)
        ttnn.deallocate(conv_out)

        def to_heads(flat, num_heads, head_dim, repeat: int, scale=None, dense: bool = False):
            t = ttnn.reshape(flat, (1, batch, num_heads, head_dim))
            if repeat > 1:
                rep = ttnn.repeat_interleave(t, repeat, dim=2)
                _free(t, flat, rep)
                t = rep
            if scale is not None:
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

        b_h = ttnn.reshape(b_raw, (1, batch, nv, 1))
        g_h = ttnn.reshape(g, (1, batch, nv, 1))
        _free(b_raw, b_h)
        _free(g, g_h)

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
        # The one changed line against the fused layer: the terminal projection takes this stage's
        # role config, so at decode it is a DRAM-sharded matmul reading a width-sharded L1 shard.
        result = self._project(flat, "out_proj", decode=True, dtype=ttnn.bfloat16)
        ttnn.deallocate(flat)
        return result

    # ------------------------------------------------------------- forwards

    def prefill_forward(self, *args, **kwargs):
        """Prefill, with the phase flag set so :meth:`_mlp` knows which config to use."""
        self._decoding = False
        return super().prefill_forward(*args, **kwargs)

    def decode_forward(self, hidden_states, *, current_pos=None, page_table=None, rot_mats=None):
        """One decode step for all ``max_batch`` users.

        Public contract unchanged: ``[1, 1, batch, hidden]`` DRAM-interleaved in and out.  Inside,
        the residual stream is width-sharded in L1 from the first norm to the last residual add, so
        neither norm, neither residual add and none of the projections touches DRAM for an
        activation (OPT-003).  The two conversions at the boundary are the price of keeping the
        stage-1/stage-2 public tensor contract; a stacked-model stage that wants the sharded stream
        to cross the layer boundary can read :attr:`decode_stream_mem_cfg` and skip them.
        """
        s = self.shapes
        assert len(hidden_states.shape) == 4, f"decode expects [1, 1, batch, hidden]; got {hidden_states.shape}"
        assert (
            int(hidden_states.shape[2]) == self.max_batch
        ), f"decode batch {int(hidden_states.shape[2])} != max_batch {self.max_batch}"
        self._decoding = True
        try:
            if not self.decode_geometry.sharded_stream:
                return super().decode_forward(
                    hidden_states, current_pos=current_pos, page_table=page_table, rot_mats=rot_mats
                )

            residual = ttnn.to_memory_config(hidden_states, self.decode_stream_mem_cfg)
            normed = self._decode_rms_norm(residual, self.w["input_layernorm"])
            if s.layer_type == FULL_ATTENTION:
                assert current_pos is not None and page_table is not None and rot_mats is not None
                mixed = self._full_attention_decode(
                    normed, current_pos=current_pos, page_table=page_table, rot_mats=rot_mats
                )
            else:
                mixed = self._linear_attention_decode(normed)
            _free(normed, mixed, residual)
            mixed = self._to_stream(mixed, s.hidden_size)
            hidden = ttnn.add(residual, mixed, memory_config=self.decode_stream_mem_cfg)
            _free(residual, hidden_states, hidden)
            ttnn.deallocate(mixed)
            normed2 = self._decode_rms_norm(hidden, self.w["post_attention_layernorm"])
            mlp_out = self._mlp(normed2)
            _free(normed2, mlp_out, hidden)
            mlp_out = self._to_stream(mlp_out, s.hidden_size)
            out = ttnn.add(hidden, mlp_out, memory_config=self.decode_stream_mem_cfg)
            ttnn.deallocate(hidden)
            ttnn.deallocate(mlp_out)
            interleaved = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
            _free(out, interleaved)
            return interleaved
        finally:
            self._decoding = False

    def _to_stream(self, tensor, width: int):
        """Put a decode-shaped tensor on the residual stream's shard grid, if it is not already."""
        wanted = self._stream_cfg(width)
        if _mem_config_eq(tensor.memory_config(), wanted):
            return tensor
        moved = ttnn.to_memory_config(tensor, wanted)
        _free(tensor, moved)
        return moved

    # ------------------------------------------------------------- reporting

    def config_summary(self) -> dict:
        """Machine-readable record of every config this layer actually runs.

        The optimized-stage tests and the ``doc/optimized_decoder/`` artifacts are generated from
        this rather than from prose, so a policy or geometry change cannot leave the documents
        claiming something the code does not do.  The *proof* that a dtype reached the measured op
        is the ``tt-perf-report`` Math-Fidelity column, not this dict (OPT-013); this is the
        intent, and the tests cross-check the two.
        """
        summary = {
            "layer_kind": self.shapes.layer_type,
            "policy": self.policy.name,
            "max_batch": self.max_batch,
            "kv_cache_dtype": str(self.kv_cache[0].dtype) if self.kv_cache else None,
            "state_fidelity": str(self.policy.gdn_state_fidelity),
            "decode": {
                "cores": self.decode_cores,
                "legal_cores": self._legal_stream_cores(),
                "rows": self.decode_rows,
                "sharded_stream": self.decode_geometry.sharded_stream,
                "dram_sharded": self.decode_geometry.dram_sharded,
                "split_gate_up": self.decode_geometry.split_gate_up_decode,
                "gate_silu_fused": self.gate_silu_program_cfg is not None,
                "norm_block_w": self.decode_norm_prgm_cfg.block_w,
                "norm_subblock_w": self.decode_norm_prgm_cfg.subblock_w,
                "sdpa": (
                    {
                        "cores_per_head_batch": self.sdpa_decode_program_cfg.max_cores_per_head_batch,
                        "k_chunk_size": self.sdpa_decode_program_cfg.k_chunk_size,
                        "q_chunk_size": self.sdpa_decode_program_cfg.q_chunk_size,
                        "exp_approx_mode": self.sdpa_decode_program_cfg.exp_approx_mode,
                        "grid": list(self.decode_geometry.sdpa_grid or (8, 8)),
                    }
                    if self.shapes.layer_type == FULL_ATTENTION
                    else None
                ),
            },
            "prefill": {"split_gate_up": self.decode_geometry.split_gate_up_prefill},
            "roles": {},
        }
        for role, (k, n) in sorted(self.role_shapes.items()):
            if role not in self.w:
                continue
            entry = {
                "K": k,
                "N": n,
                "weight_dtype": str(self.w[role].dtype),
                "policy_dtype": str(self.policy.weight_dtype(role)),
                "fidelity": str(self.policy.fidelity(role)),
                "fidelity_decode": str(self.policy.fidelity(role, decode=True)),
                "fp32_dest_acc": self.policy.fp32_acc(role),
                "fp32_dest_acc_decode": self.policy.fp32_acc(role, decode=True),
                "weight_memory": "dram_width_sharded" if role in self.weight_mem_cfg else "dram_interleaved",
            }
            pc = self.decode_program_cfg.get(role)
            if pc is not None:
                entry["decode_program_config"] = {
                    "class": "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
                    "in0_block_w": pc.in0_block_w,
                    "per_core_M": pc.per_core_M,
                    "per_core_N": pc.per_core_N,
                    "cores": self.decode_cores,
                    "input_shard_k_tiles": (k // ttnn.TILE_SIZE) // self.decode_cores,
                    "input_memory": "l1_width_sharded",
                    "output_memory": "l1_width_sharded",
                }
            grid = self.prefill_geometry.grid(role)
            if grid is not None:
                entry["prefill_program_config"] = {
                    "class": "MatmulMultiCoreReuseMultiCastProgramConfig",
                    "grid": list(grid),
                    "in0_block_w": self.prefill_geometry.block_w(role),
                }
            summary["roles"][role] = entry
        return summary

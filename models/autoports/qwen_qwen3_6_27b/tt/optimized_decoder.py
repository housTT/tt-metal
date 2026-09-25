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
from dataclasses import dataclass, field
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
    #: ``in_proj_qkv``'s math fidelity.  **HiFi2**, and the reason is the advertised context.
    #:
    #: This role is about 14 % of the traced ``linear_attention`` decode step and its output *is* the
    #: float32 state the causal convolution carries, which is why the higher fidelity was inherited
    #: without being swept - "the state deserves accuracy" is a plausible reason and it was the only
    #: one on the record.  Swept, LoFi is worth **4.9 % of the traced decode step and 3.2 % of prefill**
    #: (1.2810 vs 1.3469 ms, 18.553 vs 19.160 ms) and it clears the acceptance bar on the real
    #: checkpoint with margin: prefill 0.999309, decode 0.997146, traced decode 0.998837, and - the two
    #: numbers that actually matter for a state producer - conv state 0.999869 and recurrent state
    #: 0.999714, against 0.999958 / 0.999849 at HiFi2.  HiFi4 costs 11 % of the step for 4e-4 of PCC.
    #:
    #: BFP4 *weights* on this role are rejected on **real-weight** evidence rather than synthetic, which
    #: is the distinction OPT-012 insists on: decode 0.962928 and conv state 0.992137, both below the
    #: bar, even though BFP4+LoFi would have been 6.1 % faster than the shipped configuration.
    #:
    #: **And LoFi is rejected, by the one measurement that could see it.**  Every 2049-token and
    #: real-weight number above says LoFi is free.  At the advertised context it is not: this role builds
    #: the recurrent state over all 262143 tokens, and there the full-context decode *scale* comes back
    #: at 0.959272 on real weights and 0.978149 on the stand-in, against a (0.98, 1.02) tolerance - the
    #: real-weight arm is the one that failed ``test_full_advertised_context`` outright.  The carried
    #: recurrent state's own scale goes with it, 0.923884 against HiFi2's 0.938657.  HiFi2 holds both
    #: gates (0.980911 real, 0.982941 synthetic) and the whole difference is invisible to PCC, which is
    #: 0.9993 vs 0.9995 either way.
    #:
    #: A gain error in a carried state is exactly what a scale-ratio gate exists to catch, and this is why
    #: the fidelity of a *state producer* cannot be chosen on short-context evidence however good that
    #: evidence looks.
    #:
    #: ``logs/probe_optimized_policy.log`` has the latencies, ``logs/probe_real_weight_policy.log`` the
    #: real-weight PCCs including the state, and ``logs/probe_long_context_linear_real.log`` is the arm
    #: that decided it.
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

    #: Per-role math fidelity that applies at **prefill only**, overriding the role's own field.
    #:
    #: The mirror of :attr:`decode_fidelity`, and it exists for the same reason
    #: :attr:`prefill_fp32_acc_roles` does: the two phases have different accuracy contracts.  Prefill
    #: is what fills the KV cache and builds the recurrent state, and those are read hundreds of
    #: thousands of times afterwards, so a systematic error there is amplified in a way the same error
    #: at decode is not.  Decode is where fidelity costs the most relative to the work done.
    #:
    #: **``mlp_down`` at HiFi4**, and it is the whole fix for §3.8.2's full-context scale failure.
    #:
    #: LoFi truncates the operand mantissa toward zero, which is a systematic *gain loss* rather than
    #: symmetric noise (§3.8.2 establishes this model-free).  A per-element truncation bias accumulates
    #: over the reduction, so the role that shows it worst is the one that reduces deepest - ``mlp_down``
    #: over 17408 elements, three times ``mlp_gate``/``mlp_up``'s depth - and raising *only* that one role
    #: is both the most accurate and nearly the cheapest option measured:
    #:
    #:     arm                                    tail scale   linear prefill   full prefill
    #:     shipped LoFi                           0.968952       19.146 ms        9.794 ms   (FAILS 0.98)
    #:     HiFi2 on wqkv + all three MLP matmuls  0.980595       22.261 ms       13.508 ms
    #:     **HiFi4 on mlp_down only**             **0.983246**   22.345 ms       12.989 ms
    #:     HiFi4 on wqkv + mlp_down               0.983887       22.346 ms       14.572 ms
    #:     HiFi2 wqkv+gate/up, HiFi4 mlp_down     0.983698       24.400 ms       15.697 ms
    #:     HiFi4 on all three MLP matmuls         0.985473       28.394 ms       19.195 ms   (see below)
    #:
    #: Uniform HiFi4 is the most accurate and is **not shippable**: at 28.394 and 19.195 ms it is slower
    #: than the stage-2 baseline this stage is measured against (25.830 and 17.780), so it would give back
    #: more than the whole layout change won.  Adding ``wqkv`` buys 6e-4 of tail scale and a tidier paged V
    #: cache for 1.58 ms of ``full_attention`` prefill, and makes the *decode* scale slightly worse
    #: (0.991326 against 0.993796), so the V-cache scale error it fixes was not the thing that mattered.
    #: ``o_proj`` is marginally harmful.  ``logs/probe_prefill_fidelity_{roles,cost,mixed}.log``.
    #:
    #: Prefill-only, for the reason :attr:`prefill_fp32_acc_roles` is: prefill fills the cache and builds
    #: the state that the next 262144 reads all depend on, and decode - where fidelity costs the most per
    #: unit of work, and which this stage optimises hardest - writes one row.  Decode keeps LoFi.
    #:
    #: A dict makes this frozen dataclass unhashable, which is fine here - policies are only ever dict
    #: *values* (``test_optimized_decoder_perf.POLICIES``) and are never hashed - but a future caller that
    #: wants a policy in a set or as a key has to normalise this field to a tuple of pairs first.
    prefill_fidelity_roles: Optional[dict] = field(default_factory=lambda: {"mlp_down": HIFI4})

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
        if not decode and role in (self.prefill_fidelity_roles or {}):
            return self.prefill_fidelity_roles[role]
        return getattr(self, self._FIDELITY_FIELD[role])

    #: Roles whose *output* is a float32 tensor the gated-delta-rule recurrence carries, and which
    #: therefore accumulate in float32 destination registers at prefill.
    STATE_ROLES = ("in_proj_qkv", "in_proj_ab")

    #: The subset of :data:`STATE_ROLES` that :attr:`state_fp32_acc_decode` applies to.
    #:
    #: ``in_proj_ab`` is deliberately *not* here.  It is four output tiles of state arithmetic feeding the
    #: recurrence's decay (§3.5), so its accumulation precision is worth nothing in either direction, and
    #: it is the one role whose decode matmul shares the prefill compute-kernel config.  Excluding it
    #: keeps the policy a description of what the code does rather than a claim the code does not honour.
    STATE_ROLES_DECODE_OVERRIDE = ("in_proj_qkv",)

    #: Keep float32 destination accumulation on :data:`STATE_ROLES` at **decode**.
    #:
    #: Separable from the prefill setting because the two phases write different things: at prefill
    #: ``in_proj_qkv`` produces 2048 rows that become the conv window and the initial recurrent state,
    #: while at decode it produces one row convolved into an existing state.  And worth separating on
    #: cost, because float32 destination accumulation halves matmul throughput and this role is about
    #: 14 % of the traced ``linear_attention`` decode step.
    #:
    #: **Off**, and the reason is the advertised context rather than the 0.13 % of decode it saves.
    #:
    #: On short-context evidence this knob looks worthless in both directions: dropping the accumulation
    #: is worth 0.13 % of the traced step (1.3452 vs 1.3469 ms) and costs 8e-5 of real-checkpoint decode
    #: PCC at 2049 tokens (0.998419 -> 0.998339).  At BFP8 weights this matmul sits on the bandwidth
    #: ceiling rather than the compute one, so halving its FLOP throughput costs almost nothing - and
    #: buying it back gains almost nothing.  The first pass through this stage rejected it on exactly
    #: that reasoning.
    #:
    #: The full-context arm says otherwise.  At 262143 tokens on the real checkpoint the
    #: ``linear_attention`` decode *scale* is 0.980911 with float32 accumulation on and **0.996676** with
    #: it off, against a (0.98, 1.02) tolerance - a gate passing by 0.0009 becomes one passing by 0.0167,
    #: and the off setting beats even the fused control's 0.988000.  A carried state is read for every one
    #: of the next 262144 steps, so its gain error is the thing worth optimising here, not 0.13 % of one
    #: step.  ``logs/probe_long_context_linear_real.log`` is the arm; §3.8.2 is the write-up.
    state_fp32_acc_decode: bool = False

    def fp32_acc(self, role: str, decode: bool = False) -> bool:
        """Whether this role's matmul accumulates in float32 destination registers.

        The two roles whose *output* is a float32 tensor the recurrence carries do in both phases,
        subject to :attr:`state_fp32_acc_decode` at decode; the roles in
        :attr:`prefill_fp32_acc_roles` do at prefill only; :attr:`fp32_dest_acc_all` restores the fused
        stage's blanket setting for the baseline arm.
        """
        if self.fp32_dest_acc_all:
            return True
        if role in self.STATE_ROLES:
            if decode and role in self.STATE_ROLES_DECODE_OVERRIDE:
                return self.state_fp32_acc_decode
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
    # Stage 2 had no per-role prefill fidelity at all, so the baseline arm must opt out of this stage's
    # ``mlp_down`` override explicitly or it would not be stage 2's configuration.
    prefill_fidelity_roles=None,
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
    #: Build the decode stream's shard grid as a rectangle instead of ``num_cores_to_corerangeset``'s
    #: row-wise fill.  **Measured and not taken**, and the measurement is the point:
    #:
    #: 32 cores on this device's 11x10 grid fill row-wise as two ragged ranges whose bounding box is
    #: 33, and ops that check rectangularity rather than core count degrade on that - `ttnn.reshape`
    #: on a width-sharded tensor falls back to INTERLEAVED.  A rectangular 8x4 grid fixes that in
    #: principle and loses in practice, because
    #: ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` computes its *own* output grid
    #: row-wise and overrides whatever the caller provides ("Mismatch between computed MemoryConfig
    #: ... Using computed config", six times per step).  Every DRAM-sharded matmul therefore lands on
    #: the ragged grid anyway and the next op reshards it back: 9 and 11 memory-config conversions per
    #: step against 6 and 8, for no time (1.3522 vs 1.3480 ms and 0.9762 vs 0.9799 ms - a wash both
    #: ways).  And the fallbacks it was meant to remove are already gone: the explicit ``z`` unshard in
    #: :meth:`_linear_attention_decode` took the INTERLEAVED reshape count to **zero on both grids**.
    #: ``logs/probe_stream_grid.log`` is the table.
    rectangular_stream: bool = False
    #: Give ``in_proj_ab`` the DRAM-sharded decode matmul too, with ``dt_bias`` added as a separate
    #: elementwise op instead of the matmul's bias row.  ``False`` keeps stage 2's interleaved,
    #: bias-folded, measured-``core_grid`` form.  This is the arm that makes §3.5's claim measurable
    #: rather than self-referential.
    dram_sharded_ab: bool = False
    #: Normalise the gated-delta-net ``q``/``k`` heads **before** expanding them to the value-head
    #: count, and expand along a batch axis instead of a tile axis.  **Measured and not taken.**
    #:
    #: The shipped order is stage 2's: reshape the 16 key heads to ``[1, B, 16, dk]``,
    #: ``repeat_interleave`` to 48 on ``dim=2``, then normalise.  ``dim=2`` is a *tile* axis, so that
    #: repeat runs as ``untilize_with_unpadding`` on 1 core -> ``concat`` of 48 pieces ->
    #: ``tilize_with_val_padding`` on 2 cores, and the device report shows the pair twice per step at
    #: about 2 % of the traced ``linear_attention`` decode step.  Removing it looks free: the norm is
    #: per-head over the last dim and the expanded copies are identical, so
    #: ``norm(repeat(x)) == repeat(norm(x))`` exactly, which makes "normalise 16 heads, then expand on
    #: ``dim=1`` of ``[1, B*16, 1, dk]``" an equivalent graph with no layout change in it at all.
    #:
    #: It is equivalent - the PCC is identical to six decimals at both regimes - and it is **slower
    #: where it matters**: a wash at batch 1 (1.3479 vs 1.3487 ms, inside the spread) and 1.9 % worse
    #: at the advertised ``max_batch`` of 32 (4.2317 vs 4.1519 ms), because the batch-axis repeat over
    #: 512 -> 1536 entries costs more than the tile-axis one plus its layout round-trip.  So the two
    #: one-and-two-core layout ops stay, now with a measured reason rather than as an unexplained row
    #: in the report.  ``logs/probe_norm_repeat_order.log`` is the table; ``True`` runs the arm.
    norm_before_repeat: bool = False
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

    #: Ceiling on the per-role ``in0_block_w`` search, or ``None`` for :data:`_PREFILL_MAX_BLOCK_W`.
    #: A knob rather than a constant because the shipped ceiling has to be *measured*: with the L1
    #: model exact, whether a deeper block is faster or slower is a question about the op, not about
    #: what fits.
    max_block_w: Optional[int] = None

    def grid(self, role: str):
        return (self.grids or {}).get(role)

    def block_w(self, role: str):
        return (self.in0_block_w or {}).get(role)

    def cap(self) -> int:
        return self.max_block_w or _PREFILL_MAX_BLOCK_W


DEFAULT_PREFILL_GEOMETRY = PrefillGeometry()


# ------------------------------------------------------------------------ helpers


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _divisors(value: int) -> list:
    return [d for d in range(1, value + 1) if value % d == 0]


def _rectangular_core_range(cores: int, grid) -> ttnn.CoreRangeSet:
    """A **rectangular** ``CoreRangeSet`` of exactly ``cores`` cores, widest first.

    Reached only through :attr:`DecodeGeometry.rectangular_stream`, which is **off** by default; see
    that attribute for the measurement that decided it.  The short version: this exists because
    ``ttnn.num_cores_to_corerangeset`` fills rows and leaves a ragged last row - 32 cores on this
    device's 11x10 grid come back as ``{[0-0 - 10-1], [0-2 - 9-2]}``, whose bounding box is 33 - and
    ops that check *rectangularity* rather than core count degrade on that, ``ttnn.reshape`` by falling
    back to INTERLEAVED.  It turned out to be the wrong fix for that problem: the DRAM-sharded matmul
    overrides the output grid with its own row-wise one, so a rectangular stream buys three extra
    reshards per step, and the fallbacks were removed instead by not reshaping a width-sharded tensor
    at all.  It stays because a measured-and-rejected arm has to remain runnable.
    """
    best = None
    for width in range(min(cores, grid.x), 0, -1):
        if cores % width:
            continue
        height = cores // width
        if height <= grid.y:
            best = (width, height)
            break
    if best is None:
        raise ValueError(f"no rectangle of {cores} cores fits a {grid.x}x{grid.y} grid")
    width, height = best
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, height - 1))})


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

#: Default ceiling on the per-role ``in0_block_w`` search at prefill; :attr:`PrefillGeometry.cap`
#: overrides it, which is how ``probe_optimized.py prefill`` sweeps it.
#:
#: Every ``K`` this model reduces over is 160, 192 or 544 tiles, and 16 divides all three, so the
#: ceiling is a *measured* choice and not an arithmetic one - the L1 model below already rejects
#: whatever does not fit, and at 16 that is every role except ``wgate`` and ``in_proj_z``.  See
#: ``work_log.md`` section 3.4 for the sweep this value comes from.
_PREFILL_MAX_BLOCK_W = 8

#: Row count :meth:`OptimizedDecoder.config_summary` reports its derived prefill configs at.  Prefill
#: runs at every padded chunk length the public API allows, so a summary has to name one; 2048 is the
#: length every perf table in ``doc/optimized_decoder/`` is measured at.
_PREFILL_SUMMARY_ROWS = 2048

#: Fixed L1 a prefill 2D multicast matmul allocates on top of the four blocks modelled below.
#:
#: This replaced a flat 1.1 MB budget, which was a guess in both directions: it held
#: ``linear_attention``'s ``in_proj_qkv`` at 4 when 5 fits, and it would have allowed configurations
#: that do not.  The four block circular buffers - double-buffered ``in0`` and ``in1``, the output
#: block in the output dtype, and a float32 accumulation intermediate when the role accumulates in
#: float32 but does *not* output it - are exactly modelled; what is left over is the sender-side
#: ``in1`` buffer and the multicast semaphores, which do not depend on ``in0_block_w``.
#:
#: That leftover is **measured, twice, on two roles whose modelled totals differ by 270 KB**, and it
#: is the same both times.  Forcing ``in0_block_w = 8``:
#:
#: * ``in_proj_qkv`` models 1,482,752 B and the op reports "Statically allocated circular buffers ...
#:   grow to 1594240 B which is beyond max L1 size of 1572864 B" -> 111,488 B over;
#: * ``wqkv`` models 1,474,560 B and the op reports "grow to 1586048 B" -> 111,488 B over.
#:
#: So the model plus this constant reproduces what the op allocates *exactly*, and the comparison is
#: against ``ttnn.get_max_worker_l1_unreserved_size()`` (1,532,032 B here, 40 KB under the 1,572,864 B
#: the op checks), which is the margin.  No fudge factor: if this bound ever rejects a block size that
#: would have fit, the arithmetic is wrong and can be corrected rather than re-tuned.
_PREFILL_L1_FIXED = 111_488

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
        self.decode_core_range = (
            _rectangular_core_range(cores, grid)
            if self.decode_geometry.rectangular_stream
            else ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True)
        )

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
        #: Unreserved L1 per core, read from the device rather than assumed: it is what the prefill
        #: ``in0_block_w`` search is allowed to fill (see :data:`_PREFILL_L1_MARGIN`).
        self.l1_size = ttnn.get_max_worker_l1_unreserved_size()
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
            if (n // ttnn.TILE_SIZE) % cores or (k // ttnn.TILE_SIZE) % cores:
                raise ValueError(
                    f"role {role!r} ({k} x {n}) cannot be DRAM-sharded at cores={cores}: its "
                    f"{k // ttnn.TILE_SIZE} x {n // ttnn.TILE_SIZE} tile shape does not divide the "
                    "stream's core count, so neither its activation nor its output has a legal "
                    "width shard"
                )
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
            # fold ``dt_bias`` into, so by default this role keeps the fused stage's measured
            # ``core_grid`` form; :attr:`DecodeGeometry.dram_sharded_ab` is the arm that measures the
            # alternative (DRAM-sharded matmul plus a separate bias add).
            return self.decode_geometry.dram_sharded_ab
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
            # The weight is width-sharded across this chip's DRAM banks, and the compute grid may not
            # have more columns than there are banks to feed them.  Measured, not assumed, and the
            # measurement is narrower than it first looked: `logs/probe_prefill_grid_alignment.log`
            # runs this config at *every* legal column count for every DRAM width-sharded role, and
            # column counts of 2, 4, 5, 6 and 8 are all correct to the same PCC against
            # ``ttnn.linear``'s heuristic on an interleaved copy of the weight (0.99937-0.99997,
            # identical across column counts), while **10** returns non-finite values rather than
            # failing validation - on the 160-tile-wide ``o_proj`` / ``mlp_down`` / ``out_proj`` and on
            # the 320-tile-wide ``in_proj_qkv`` alike.  So the rule is a *bound*, not an equality, and
            # the bound is this chip's 8 DRAM banks.
            #
            # The largest legal column count is taken because it is the widest grid: every ``N`` this
            # model projects to is a multiple of 8 tiles, so that is 8 here, and ``per_core_N`` stays
            # exact.  Writing it as a bound rather than pinning it to ``dram_banks`` keeps a role whose
            # ``N`` is *not* a multiple of the bank count legal instead of unbuildable.
            x = _largest_divisor_at_most(n_tiles, min(self.grid.x, self.dram_banks))
            y = _largest_divisor_at_most(m_tiles, self.grid.y)
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
        # circular buffers fitting this core's real L1 with :data:`_PREFILL_L1_MARGIN` for the buffers
        # the estimate does not model.  Prefill has no resident width-sharded activations competing for
        # L1, so this is a much larger allowance than the decode one, and it is per role rather than
        # global because only two roles are anywhere near it - the two with a float32 output block.
        # ``linear_attention``'s ``in_proj_qkv`` is the one role the bound actually holds back, and it
        # is held back by a *measured* overflow rather than a chosen budget.
        weight_bytes = _TILE_BYTES.get(self.policy.weight_dtype(role), 2048)
        out_dtype = self.role_out_dtype(role)
        out_bytes = _TILE_BYTES[out_dtype]
        # A role that accumulates in float32 destination registers but packs a narrower output needs a
        # separate float32 intermediate to carry partials across ``in0_block_w`` blocks; one that
        # already outputs float32 accumulates into the output block itself.
        interm_bytes = _TILE_BYTES[ttnn.float32] if self.policy.fp32_acc(role) and out_dtype != ttnn.float32 else 0
        block_w = 1
        for candidate in _divisors(k_tiles):
            if candidate > self.prefill_geometry.cap():
                break
            l1 = 2 * candidate * out_block_h * _TILE_BYTES[ttnn.bfloat16]
            l1 += 2 * candidate * out_block_w * weight_bytes
            l1 += out_block_h * out_block_w * (out_bytes + interm_bytes)
            if l1 + _PREFILL_L1_FIXED <= self.l1_size:
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

    #: Output dtype per role, as the forward passes request it.  This is a *declaration* rather than a
    #: second copy: :meth:`_project` asserts the caller's ``dtype`` against it, so the prefill L1 model
    #: in :meth:`_prefill_program_cfg` - which has only the role name to work from - cannot silently
    #: disagree with what the op is actually asked to pack.  Only the two roles that feed the
    #: recurrence's float32 state carry a float32 output.
    ROLE_OUT_DTYPE = {"in_proj_qkv": ttnn.float32, "in_proj_ab": ttnn.float32}

    def role_out_dtype(self, role: str):
        return self.ROLE_OUT_DTYPE.get(role, ttnn.bfloat16)

    def _project(self, x, role: str, *, decode: bool, dtype, program_cfg=None):
        """One projection, with this stage's memory/program/compute config for ``role``.

        Decode takes the DRAM-sharded path when the role has one, resharding its input only if the
        caller did not already hand it the stream layout - which, on the shipped configuration, it
        always does.  Prefill takes the explicit 2D program config when one is configured and
        ``ttnn.linear``'s heuristic otherwise.
        """
        assert dtype == self.role_out_dtype(role), (
            f"role {role!r} is declared to output {self.role_out_dtype(role)} in ROLE_OUT_DTYPE but is "
            f"being asked for {dtype}; the prefill L1 model reads the declaration, so the two must agree"
        )
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
        if decode and "in_proj_ab" in self.decode_program_cfg:
            # The DRAM-sharded arm: no bias slot, so ``dt_bias`` becomes its own elementwise add.
            ab_sharded = self._project(x, "in_proj_ab", decode=True, dtype=ttnn.float32)
            ab = ttnn.add(ab_sharded, self.w["in_proj_ab_bias"])
            ttnn.deallocate(ab_sharded)
            if ab.memory_config().is_sharded():
                interleaved = ttnn.sharded_to_interleaved(ab, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(ab)
                ab = interleaved
        else:
            ab_in = x if not x.memory_config().is_sharded() else ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
            ab = ttnn.linear(
                ab_in,
                self.w["in_proj_ab"],
                bias=self.w["in_proj_ab_bias"],
                dtype=ttnn.float32,
                # Phase-appropriate, not always the prefill config: this call is reached from both
                # phases (``phase`` already selects the core grid), so taking ``role_kernel_cfg``
                # unconditionally would silently ignore any decode-side policy for this role.  The two
                # configs are identical for ``in_proj_ab`` today - see ``STATE_ROLES_DECODE_OVERRIDE`` -
                # and this keeps them identical *because* the policy says so rather than by accident.
                compute_kernel_config=(
                    self.role_kernel_cfg_decode["in_proj_ab"]
                    if phase == "decode"
                    else self.role_kernel_cfg["in_proj_ab"]
                ),
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
            if repeat > 1 and not self.decode_geometry.norm_before_repeat:
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
            if repeat > 1 and self.decode_geometry.norm_before_repeat:
                # ``dim=1`` of ``[1, B*num_heads, 1, head_dim]`` is a batch axis: entry ``b*num_heads+h``
                # becomes ``b*num_heads*repeat + h*repeat + r``, which is exactly the head order the
                # ``dim=2`` repeat produced, and no tile has to be rebuilt to get it.
                rows = ttnn.reshape(t, (1, batch * num_heads, 1, head_dim))
                _free(t, flat, rows)
                out = ttnn.repeat_interleave(rows, repeat, dim=1)
                _free(rows, flat, out)
                return out
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
        # ``z`` came back from a DRAM-sharded matmul, and that op picks its **own** output shard grid
        # rather than the one it is handed: the runtime says so - "Mismatch between computed
        # MemoryConfig(... grid=[{0,0}-{10,1}, {0,2}-{9,2}] ...) and provided MemoryConfig(...
        # grid=[{0,0}-{7,3}] ...). Using computed config" - and the grid it computes is the ragged
        # row-wise one whose bounding box is 33 cores for 32 shards.  ``ttnn.reshape`` requires a
        # *rectangular* grid and silently falls back to DRAM interleaved when it does not get one, so
        # the rank change below would be an uncounted host-invisible DRAM round trip.  Doing the
        # conversion explicitly makes it one counted op instead of a silent fallback; the small-batch
        # branch consumes ``z`` as an interleaved rank-4 tensor anyway.
        if z.memory_config().is_sharded():
            z_interleaved = ttnn.sharded_to_interleaved(z, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(z)
            z = z_interleaved
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
            # The *derived* prefill config at the canonical measured length, not the override that
            # produced it.  Recording the override was how a stale hand-written table once described a
            # configuration the layer never ran; this records what the op is actually handed.
            prefill_pc = self._prefill_program_cfg(role, _PREFILL_SUMMARY_ROWS)
            if prefill_pc is not None:
                entry["prefill_program_config"] = {
                    "class": "MatmulMultiCoreReuseMultiCastProgramConfig",
                    "rows": _PREFILL_SUMMARY_ROWS,
                    "grid": [prefill_pc.compute_with_storage_grid_size.x, prefill_pc.compute_with_storage_grid_size.y],
                    "in0_block_w": prefill_pc.in0_block_w,
                    "in0_block_w_ceiling": self.prefill_geometry.cap(),
                    "out_block_h": prefill_pc.out_block_h,
                    "out_block_w": prefill_pc.out_block_w,
                    "out_subblock_h": prefill_pc.out_subblock_h,
                    "out_subblock_w": prefill_pc.out_subblock_w,
                    "per_core_M": prefill_pc.per_core_M,
                    "per_core_N": prefill_pc.per_core_N,
                    "out_dtype": str(self.role_out_dtype(role)),
                }
            summary["roles"][role] = entry
        return summary

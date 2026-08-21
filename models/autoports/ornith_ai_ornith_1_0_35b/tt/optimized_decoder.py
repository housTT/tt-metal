# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Performance-optimized TTNN decoder layer for ornith-ai/Ornith-1.0-35B.

This is the optimized successor to
:mod:`~models.autoports.ornith_ai_ornith_1_0_35b.tt.fused_decoder`, which is in turn the graph-fused
successor to the functional decoder. **Semantics, public contract and supported capability are
identical to the fused decoder's** — same prefill/decode signature, same paged KV cache geometry,
same DeltaNet state, same determinism, same support for any non-aligned ``seq_len``, same advertised
262144-token context — and every fused rewrite is preserved. What changes is how fast it runs, and
each change is measured in ``doc/optimized_decoder/`` with the candidate it beat.

Measured on one Blackhole ``p300c``, real checkpoint weights, batch 1, against the fused decoder
built in the same process on the same device (``doc/optimized_decoder/logs/bench.py``): roughly
**2.5x** warmed 2048-token prefill and **1.9-2.1x** warmed traced decode on both layer kinds.

**No absolute run-varying timing is quoted anywhere in this module or its tests, on purpose.** Four
review rounds of this stage closed on "the figures are re-derived" and each next re-run made a
hand-written microsecond figure in a docstring stale again — round 5 still found six here, three of them
disagreeing with each other about a single measurement — so the numbers live only where a generator or an
audit keeps them true: the README's headline table is spliced from
``doc/optimized_decoder/logs/ab_fused_vs_optimized.txt`` by ``logs/make_readme.py``, and
``doc/optimized_decoder/audit_figures.py`` asserts that every figure quoted in any of this stage's
documents exists in a committed artifact. What the comments below carry instead is the *decision*,
the shipped configuration, and the artifact rows to read - which do not drift.

What was changed, largest measured effect first (``doc/optimized_decoder/work_log.md`` §3 has the
cumulative table, one row per step)

* **Every routed-expert intermediate moved from DRAM to L1** — the packed gate/up output, its two
  unpacking slices, the SwiGLU product, the scored activation, the down output and the expert
  reduction. They are ``num_experts`` wide but small per call, and the fused decoder put all of them
  in ``ttnn.DRAM_MEMORY_CONFIG``. :meth:`OptimizedMoE._expert_mem` falls back to DRAM for expert
  group sizes whose intermediates do not fit an L1 budget, so a large ``moe_group_tokens`` is slower
  rather than broken.
* **Sparse-matmul geometry chosen from the call's active-expert bound**: 8 cores with a wide output
  block for a decode step, 32 cores with ``per_core_N`` 1 for a prefill group. The fused decoder
  used the largest core count dividing ``Nt``, which is right for prefill and ~40 % off for decode.
* **Explicit program configs for every dense projection**, tuned per role and per phase: a 1D
  ``mcast_in0`` config for the skinny decode shapes, a 2D config with a large inner block for
  prefill. ttnn's heuristic was measured for all of them and is the loser in every role.
* **A precision policy per tensor group** (:class:`PrecisionPolicy`): BFP4 + LoFi routed-expert
  weights, BFP8 + HiFi2 dense projections and shared expert, BFP8 expert activations, BFP8 KV cache;
  router weight, norms, RoPE tables and the DeltaNet float32 state deliberately unchanged.
  ``POLICIES["fused-parity"]`` reproduces the fused decoder's dtypes exactly for A/B.
* **The decode MoE no longer routes its tile padding.** A batch-1 decode step runs the MoE on a
  32-row tile with one real row; the padding rows have exactly-zero router logits, so ``topk``
  returns a full expert set for each and the old whole-tile reduction unioned them into the
  sparsity — 16 active experts where the model asks for 8. See
  :meth:`OptimizedMoE._active_expert_mask`.
* **Width-sharded decode RMSNorms.** ``ttnn.rms_norm`` parallelises over rows and a decode
  activation is one tile of rows, so the interleaved form ran the whole 2048-wide norm on one core.
* **The decode token mixer runs on the folded residual** ``[1, 1, batch, dim]`` rather than
  ``[batch, 1, dim]`` — see :data:`DECODE_COMPACT_ROWS`. A tile row is 32 rows whatever the batch,
  so the unfolded form moved ``32 * batch`` physical rows through every mixer op to carry ``batch``
  of them; folded it is one tile row up to batch 32. Nothing about the public contract changes, the
  block's reshape count does not change, and at batch 1 the graph is unchanged op for op.

Prefill / decode contract
-------------------------
Unchanged from the fused decoder — see that module's docstring. In particular ``prefill_forward``
accepts **any** logical ``seq_len``; the 128-token physical alignment and the 2048-token internal
chunk are internal. The only added constructor argument is ``policy``.

Neither forward path calls ``torch``, ``ttnn.from_torch``, ``ttnn.to_torch`` or any host fallback
**on an allocated layer**, exactly as in the fused decoder, including the same single lazy-allocation
divergence: a forward on a layer whose state was never allocated calls ``allocate_state`` itself and
that *is* host work. It happens at most once per layer per batch, never inside a captured trace, and
never inside anything measured here.
``tests/test_optimized_decoder.py::test_lazy_allocation_is_the_only_host_call`` pins both halves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace

from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import OrnithDecoderConfig
from models.common.lightweightmodule import LightweightModule

TILE = 32


# ---------------------------------------------------------------------------- prefill inputs
@dataclass(frozen=True)
class PrefillChunkInputs:
    """Offset-stable device metadata shared by every full-attention layer in one prefill chunk.

    The legacy decoder surface accepts a page-table tensor plus a Python ``start_pos``.  That is a
    useful low-level fallback, but three operations then bake the absolute offset into their program
    key: RoPE's static table slice, the fill page-table slice, and chunked SDPA's scalar start.  The
    serving path stages these four tensors once per scheduler chunk instead:

    ``full_page_table``
        The request's complete page-table row, still needed by flexible chunked SDPA.
    ``fill_page_table``
        A fixed-width row whose first entries are the physical blocks for this chunk.  Paged fill
        consumes only as many entries as the physical token row needs, so the unused suffix is zero.
    ``position_idxs``
        A ``[1, physical_len]`` uint32 row used to gather RoPE values with ``ttnn.embedding``.
    ``chunk_start_idx_tensor``
        A device ``[1]`` int32 value read by flexible chunked SDPA at runtime.

    All four device tensors have persistent addresses allocated before decode trace capture.  Only
    their contents are refreshed between eager prefill chunks; every layer reads the same buffers.
    ``start_pos`` and ``physical_len`` are host metadata used solely to reject a stale/mismatched
    bundle and to select the already-existing shape-dependent SDPA program config.
    """

    full_page_table: object
    fill_page_table: object
    position_idxs: object
    chunk_start_idx_tensor: object
    start_pos: int
    physical_len: int

    def validate_for(self, *, start_pos: int, physical_len: int, batch: int, page_block_size: int) -> None:
        """Reject reuse after a caller refreshed the wrong staging shape or chunk."""

        if int(self.start_pos) != int(start_pos) or int(self.physical_len) != int(physical_len):
            raise ValueError(
                "staged prefill metadata is for "
                f"[{self.start_pos}, {self.start_pos + self.physical_len}), not "
                f"[{start_pos}, {start_pos + physical_len})"
            )

        full_shape = tuple(int(d) for d in self.full_page_table.shape)
        fill_shape = tuple(int(d) for d in self.fill_page_table.shape)
        pos_shape = tuple(int(d) for d in self.position_idxs.shape)
        start_shape = tuple(int(d) for d in self.chunk_start_idx_tensor.shape)
        if len(full_shape) != 2 or full_shape[0] != batch:
            raise ValueError(f"full prefill page table shape {full_shape} != [{batch}, blocks]")
        if len(fill_shape) != 2 or fill_shape[0] != batch or fill_shape[1] * page_block_size < physical_len:
            raise ValueError(
                f"fill prefill page table shape {fill_shape} cannot address {batch} x {physical_len} tokens "
                f"at page_block_size {page_block_size}"
            )
        if pos_shape != (1, physical_len):
            raise ValueError(f"prefill position row shape {pos_shape} != [1, {physical_len}]")
        if start_shape != (1,):
            raise ValueError(f"prefill chunk-start tensor shape {start_shape} != [1]")
        if full_shape[1] * page_block_size < start_pos + physical_len:
            raise ValueError(
                f"full prefill page table has {full_shape[1]} blocks, but chunk end "
                f"{start_pos + physical_len} needs more at page_block_size {page_block_size}"
            )


# ---------------------------------------------------------------------------- precision policy
@dataclass(frozen=True)
class PrecisionPolicy:
    """Weight dtype and math fidelity, **per tensor group**.

    The fused stage deliberately left every weight ``bfloat16`` and every dense matmul ``HiFi4``
    so its graph-rewrite measurements stayed interpretable. That is this stage's starting point and
    its correctness floor; each field below is swept separately (one tensor group at a time) so a
    regression can be assigned, and the selected values are recorded with their measurement in
    ``doc/optimized_decoder/README.md``.

    Field groups, in decode-cost order (see the operation-topology audit):

    ``expert_*``
        The two routed-expert ``ttnn.sparse_matmul`` projections. Together they are the largest
        single item in both prefill and decode.
    ``proj_*``
        The dense token-mixer projections: the packed attention in-projection and ``o_proj`` on
        ``full_attention`` layers, the packed DeltaNet in-projection and ``out_proj`` on
        ``linear_attention`` ones.
    ``shared_*``
        The shared expert's packed ``gate|up|router`` matmul and its down projection.
    ``router_*``
        The 256-way routing matmul. Expert *selection* is a discrete decision, so this group stays
        high precision by default: a rounding change here swaps an expert rather than perturbing a
        value.
    ``state_*``
        The ``linear_attention`` recurrent-state matmuls, which run in float32 on the persistent
        state.
    ``kv_cache_dtype``
        The paged K/V cache. Prefill fill tensors are cast to it explicitly; decode
        ``paged_update_cache`` inputs stay bfloat16, which is what that op accepts.
    ``expert_act_dtype``
        Output dtype of the routed-expert ``sparse_matmul`` calls, i.e. the dtype of the
        ``num_experts``-wide intermediate the SwiGLU, the score multiply, the zero-fill and the
        expert reduction all pay for.
    ``residual_dtype``
        The inter-layer residual stream, i.e. the dtype of both ``ttnn.add`` results in
        :meth:`MultichipDecoder._block` and therefore of what the next layer's first RMSNorm reads.
    ``ccl_dtype``
        What the two per-layer collectives carry. ``None`` means "whatever the producer emitted",
        which is the inherited behaviour: bfloat16 out of the token mixer and ``expert_act_dtype``
        out of the MoE. A dtype casts both operands to it before :meth:`MultichipDecoder._all_reduce`
        and is therefore the CCL payload width.
    ``lm_head_dtype`` / ``lm_head_fidelity`` / ``logits_dtype``
        The terminal projection: weight dtype, math fidelity, and the dtype of the logits tensor the
        split sampler's local top-k reads. ``None`` on the first two means "follow the dense
        projection group", which is what the LM head inherits as the model's one extra dense
        projection.
    ``layer_exceptions``
        Per-layer overrides, as ``((layer_idx, field_name, value), ...)``. :meth:`for_layer` resolves
        them, and ``OrnithModel.from_pretrained`` calls it once per layer, so an exception is a
        property of the built layer rather than a note in a document. The usual use is keeping the
        first and last decoder layer off the most aggressive weight dtype.

    Datatype-sweep note: the sweep stage's selected values live in
    ``doc/datatype_sweep/selected_precision_config.json`` and are loaded by
    ``tt/precision_config.py``, which is what ``OrnithModel.from_pretrained`` resolves by default.
    The dataclass defaults below stay the decoder stage's own selections so a policy constructed in
    code is still the decoder stage's contract.
    """

    name: str = "unnamed"

    expert_gate_up_dtype: object = ttnn.bfloat8_b
    expert_down_dtype: object = ttnn.bfloat8_b
    expert_fidelity: object = ttnn.MathFidelity.LoFi
    expert_fp32_acc: bool = False
    expert_packer_l1_acc: bool = False
    expert_act_dtype: object = ttnn.bfloat16

    proj_dtype: object = ttnn.bfloat8_b
    proj_fidelity: object = ttnn.MathFidelity.HiFi2
    proj_fp32_acc: bool = False
    proj_packer_l1_acc: bool = True

    shared_dtype: object = ttnn.bfloat8_b
    shared_fidelity: object = ttnn.MathFidelity.HiFi2
    shared_fp32_acc: bool = False
    shared_packer_l1_acc: bool = True

    router_dtype: object = ttnn.bfloat16
    router_fidelity: object = ttnn.MathFidelity.HiFi4
    router_fp32_acc: bool = True

    #: float32 recurrent-state matmuls: fidelity only, the state dtype is not a knob (the state is
    #: the model's exact carry between steps).
    state_fidelity: object = ttnn.MathFidelity.HiFi4
    state_fp32_acc: bool = True

    kv_cache_dtype: object = ttnn.bfloat8_b
    sdpa_fidelity: object = ttnn.MathFidelity.HiFi2
    sdpa_fp32_acc: bool = True

    #: The inter-layer residual stream. Passed explicitly to both residual adds, so changing it
    #: changes the tensor the next norm reads rather than only this document.
    residual_dtype: object = ttnn.bfloat16

    #: CCL payload dtype. ``None`` = no cast, which is what the multichip stage measured and ships.
    ccl_dtype: object = None

    #: Terminal projection. ``None`` follows ``proj_dtype`` / ``proj_fidelity``.
    lm_head_dtype: object = None
    lm_head_fidelity: object = None
    #: Dtype of the logits tensor the split sampler's local top-k consumes.
    logits_dtype: object = ttnn.bfloat16

    #: ``((layer_idx, field, value), ...)``; see :meth:`for_layer`.
    layer_exceptions: tuple = ()

    #: Measured chunked-SDPA prefill ``q_chunk``/``k_chunk`` for this policy. ``None`` falls back to
    #: the name-keyed :data:`PREFILL_SDPA_CHUNK` table, which is where the three code-defined
    #: policies keep theirs. A policy loaded from JSON carries its own measured value instead, so a
    #: new policy name cannot silently take :data:`PREFILL_SDPA_CHUNK_DEFAULT`.
    prefill_sdpa_chunk: int = None

    def replace(self, **kwargs) -> "PrecisionPolicy":
        return replace(self, **kwargs)

    def for_layer(self, layer_idx: int) -> "PrecisionPolicy":
        """This policy as the layer at ``layer_idx`` sees it, with :attr:`layer_exceptions` applied.

        Returns ``self`` when the layer has no exception, so ``layer.policy is DEFAULT_POLICY``
        stays true for an unexceptional build and the identity check in the delivered suite keeps
        meaning what it meant.
        """
        overrides = {field: value for idx, field, value in self.layer_exceptions if int(idx) == int(layer_idx)}
        if not overrides:
            return self
        unknown = set(overrides) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"layer_exceptions names unknown policy field(s) {sorted(unknown)}")
        if "layer_exceptions" in overrides or "name" in overrides:
            raise ValueError("layer_exceptions may not override 'name' or 'layer_exceptions'")
        # The resolved per-layer policy keeps its own name so a capability dump distinguishes it.
        return replace(self, name=f"{self.name}@L{int(layer_idx)}", layer_exceptions=(), **overrides)

    def exception_layers(self) -> tuple:
        return tuple(sorted({int(idx) for idx, _, _ in self.layer_exceptions}))

    @property
    def base_name(self) -> str:
        """The policy name with any :meth:`for_layer` suffix removed.

        Name-keyed tables (:data:`PREFILL_SDPA_CHUNK`) must resolve the same value for a layer that
        carries an exception as for one that does not, because the exception is a weight dtype and
        the table is about SDPA's L1 legality. Keying them on ``name`` directly would silently drop
        an exceptional layer onto :data:`PREFILL_SDPA_CHUNK_DEFAULT`, which is exactly the dead-table
        failure review round 15 found.
        """
        return self.name.split("@L")[0]

    @property
    def resolved_lm_head_dtype(self):
        return self.proj_dtype if self.lm_head_dtype is None else self.lm_head_dtype

    @property
    def resolved_lm_head_fidelity(self):
        return self.proj_fidelity if self.lm_head_fidelity is None else self.lm_head_fidelity


#: The fused decoder's policy, exactly: bfloat16 everywhere, HiFi4 with fp32 accumulation on the
#: dense matmuls, HiFi4 without it on the expert matmuls, HiFi2 SDPA, bfloat16 KV cache. This is the
#: correctness floor and the "before" column of every measurement in this stage.
FUSED_PARITY_POLICY = PrecisionPolicy(
    name="fused-parity",
    expert_gate_up_dtype=ttnn.bfloat16,
    expert_down_dtype=ttnn.bfloat16,
    expert_fidelity=ttnn.MathFidelity.HiFi4,
    expert_fp32_acc=False,
    expert_packer_l1_acc=False,
    expert_act_dtype=ttnn.bfloat16,
    proj_dtype=ttnn.bfloat16,
    proj_fidelity=ttnn.MathFidelity.HiFi4,
    proj_fp32_acc=True,
    proj_packer_l1_acc=False,
    shared_dtype=ttnn.bfloat16,
    shared_fidelity=ttnn.MathFidelity.HiFi4,
    shared_fp32_acc=True,
    shared_packer_l1_acc=False,
    router_dtype=ttnn.bfloat16,
    router_fidelity=ttnn.MathFidelity.HiFi4,
    router_fp32_acc=True,
    state_fidelity=ttnn.MathFidelity.HiFi4,
    state_fp32_acc=True,
    kv_cache_dtype=ttnn.bfloat16,
    sdpa_fidelity=ttnn.MathFidelity.HiFi2,
    sdpa_fp32_acc=True,
)

#: Selected policy — see ``doc/optimized_decoder/README.md`` §Precision for the per-group sweep that
#: chose each field.
DEFAULT_POLICY = PrecisionPolicy(
    name="optimized",
    expert_gate_up_dtype=ttnn.bfloat4_b,
    expert_down_dtype=ttnn.bfloat4_b,
    expert_fidelity=ttnn.MathFidelity.LoFi,
    expert_act_dtype=ttnn.bfloat8_b,
    proj_dtype=ttnn.bfloat8_b,
    proj_fidelity=ttnn.MathFidelity.HiFi2,
    shared_dtype=ttnn.bfloat8_b,
    shared_fidelity=ttnn.MathFidelity.HiFi2,
    kv_cache_dtype=ttnn.bfloat8_b,
)

#: The BFP4 dense-projection candidate OPT-007 requires, kept as a named policy so the comparison
#: stays reproducible and a later datatype-sweep stage can take it without rediscovering it. It is
#: **faster** on traced decode at unchanged prefill — README §4.2's generated policy-sweep table
#: times it against the selected policy — and it is **not** selected, because on the same real-weight
#: HF-golden ladder the delivered suite runs it costs an order of magnitude more layer error, leaving
#: a third of the selected policy's margin above the 0.995 bar, in one layer of a 40-layer stack, for
#: a low-single-digit percentage of one decode step. README §4.3 states the trade with both figures;
#: `doc/optimized_decoder/logs/probe_projection_dtype.txt` is the whole ladder, both arms.
BFP4_PROJECTION_POLICY = DEFAULT_POLICY.replace(name="bfp4-projections", proj_dtype=ttnn.bfloat4_b)

#: The LoFi half of the same candidate. ``$datatype-sweep`` requires a BFP4+LoFi arm for **every**
#: material BFP4 matmul group, and the decoder stage only ever measured BFP4 projections at the
#: dense group's inherited HiFi2. Named here rather than only in the sweep's JSON so the pair
#: ``bfp4-projections`` / ``bfp4-projections-lofi`` is constructible from code and the comparison
#: stays reproducible without the sweep artifacts.
BFP4_PROJECTION_LOFI_POLICY = BFP4_PROJECTION_POLICY.replace(
    name="bfp4-projections-lofi", proj_fidelity=ttnn.MathFidelity.LoFi
)

POLICIES = {
    p.name: p for p in (FUSED_PARITY_POLICY, DEFAULT_POLICY, BFP4_PROJECTION_POLICY, BFP4_PROJECTION_LOFI_POLICY)
}

#: Chunked-SDPA `q_chunk`/`k_chunk` for **prefill**, keyed by **policy name**, because the bound is L1 legality
#: and legality depends on the whole policy rather than on any one field of it.
#: `logs/probe_prefill_sdpa.txt` sweeps the ladder and work_log §4.19 reads it; the figures live there because a
#: comment cannot be regenerated when the sweep re-runs. What matters here is the shape of the result: the
#: inherited 64 is far from the winner, the winner is bounded above by program placement rather than by
#: diminishing returns, and the entries below are *measured legality*, not tuning.
#:
#: Review round 14 keyed this on `(kv_cache_dtype, sdpa_fp32_acc)` and round 15 found the table **dead**: every
#: shipped policy sets `sdpa_fp32_acc=True`, both keys carried `False`, so every lookup missed and the layer
#: silently kept the inherited 64 while three documents claimed otherwise. Two lessons are baked in here. The key
#: is the policy's own name, so a miss is a *new* policy rather than a field nobody checked; and
#: `test_every_shipped_policy_prefills_at_the_shipped_chunk` asserts the **resolved** value per policy, because a
#: legality table whose fallback is universally legal cannot be gated by a build-and-run test - which is exactly
#: how a dead table survived a full sweep, a 123-case suite and the figure audit.
#:
#: `fused-parity` takes the inherited 64 on measurement, not on caution: at 128 it throws
#: `TT_THROW: Statically allocated circular buffers ... grow to ... beyond max L1`, because bfloat16 weights and a
#: bfloat16 cache leave less room than the same chunk needs under the shipped BFP8 policy.
PREFILL_SDPA_CHUNK = {
    "optimized": 256,
    "bfp4-projections": 256,
    "bfp4-projections-lofi": 256,
    "fused-parity": 64,
}

#: What an unlisted policy gets: the fused stage's value, the only one every policy here has been shown to build.
#: Deliberately conservative - a too-large chunk is not slow, it is a `TT_THROW` at program construction.
PREFILL_SDPA_CHUNK_DEFAULT = 64

#: Device bytes per element, including the shared exponent the block-float dtypes carry (one
#: bfloat16 exponent per 16-element face row: 1 + 32*bits/8 bytes per 32 elements). Used to order
#: cache dtypes by width rather than by identity, so a *narrower*-than-BFP8 cache is not clamped by
#: :data:`PREFILL_SDPA_CHUNK_WIDE_CACHE`, which exists for the wider ones.
DTYPE_BYTES = {
    ttnn.bfloat4_b: 0.5625,
    ttnn.bfloat8_b: 1.0625,
    ttnn.bfloat16: 2.0,
    ttnn.float32: 4.0,
    ttnn.uint32: 4.0,
    ttnn.int32: 4.0,
}

#: Ceiling for any policy whose KV cache is wider than BFP8. The entries above are keyed by policy name, which is
#: what makes a miss visible - but a `--set kv_cache_dtype=...` override changes the dtypes *without* changing the
#: name, and the phase-3 policy sweep is what caught that: `optimized` with a bfloat16 cache resolved 256 and threw
#: at program construction. So the name gives the measured value and the cache dtype clamps it. 128 is the largest
#: that builds with a bfloat16 cache under the shipped weight dtypes; `fused-parity` is below this ceiling anyway
#: because its own entry is 64.
PREFILL_SDPA_CHUNK_WIDE_CACHE = 128

#: Where the routed gate/up matmul's `in0` lives. L1 is the shipped choice and the measured one
#: (`logs/ab_routed_in0.txt`); DRAM is what the slice inherited before review round 14 surfaced
#: `tt-perf-report`'s advice on that row. A module constant rather than a literal so the A/B can flip it.
ROUTED_IN0_MEMORY = ttnn.L1_MEMORY_CONFIG

#: Where the `full_attention` output projection's `in0` lives at **decode**. Same `tt-perf-report` item
#: as `ROUTED_IN0_MEMORY` and the same shape of fix: the gated multiply that produces it named no
#: placement, so it inherited DRAM from the one decode op whose output has to be in DRAM. Review round 26
#: found the advice still raised on this row while README §5.5 said it was raised nowhere in decode.
#: A module constant rather than a literal so the A/B can flip it (`logs/ab_attn_out_in0.txt`).
ATTN_OUT_IN0_MEMORY = ttnn.L1_MEMORY_CONFIG

#: Whether the decode V head-split output is written to the paged cache **as produced**, keeping its
#: height shard, instead of being interleaved to DRAM and re-sharded into an identical config.
#: `nlp_create_qkv_heads_decode` already emits V as HEIGHT_SHARDED L1 with shard `[32, head_dim]`, which
#: is bit-for-bit the config `_kv_update_memory_configs` used to rebuild, and `paged_fused_update_cache`
#: accepts it. Until review round 27 the layer threw that shard away and paid a `sharded_to_interleaved`,
#: a `FillPad` and an `InterleavedToSharded` per decode step to rebuild it, and the work log called all
#: three "required by an op contract". A module constant rather than a literal so the A/B can flip it
#: (`logs/ab_v_shard_passthrough.txt`).
DECODE_V_SHARD_PASSTHROUGH = True

#: Whether the K cache-write input skips its kv-head tile pad. Same reasoning as
#: :data:`DECODE_V_SHARD_PASSTHROUGH` and the same op: `paged_fused_update_cache` takes the head count
#: from the **cache**, and its writer kernel reads exactly that many rows of the input, so rows past the
#: real kv heads are never read and need not be zeroed. Review round 27 removed this pad for V without
#: noticing K still paid it; review round 28 pointed out the shipped tree already contained the
#: counter-example, since V reaches the same call unpadded. A module constant so the A/B can flip it
#: (`logs/ab_kv_pad_free_write.txt`).
DECODE_KV_PAD_FREE_WRITE = True

#: Physical alignment of a prefill block. 128 keeps every block start legal for the chunked-SDPA
#: 64-token q/k chunks, the paged cache's 64-token blocks, and the flat ``chunk_gated_delta_rule``
#: contract (which requires the block length to be a multiple of its 32-token internal chunk).
PREFILL_ALIGN = 128

#: Default logical tokens per internal prefill block.
DEFAULT_PREFILL_CHUNK = 2048

#: Paged KV cache block size (tokens per block).
DEFAULT_PAGE_BLOCK_SIZE = 64

#: Tokens per ``sparse_matmul`` call in prefill.
#:
#: One sparsity entry covers one 32-row batch entry, so a call's *down*-projection mask is the
#: union of the experts chosen by **all** its tokens. At 256 tokens/call that union is essentially
#: all 256 experts and the down projection does no skipping at all; at 32 tokens/call it is the
#: same partial expert union the gate/up projections already get. 32 is both the finest legal value
#: (one sparsity entry covers exactly one 32-row batch entry) and the fastest measured one — the sweep
#: over 32/64/128/256/512 on a 2048-token prefill is in ``doc/fused_decoder/logs/ab_moe_group_tokens.txt``
#: and tabulated in that stage's README. It also cuts the transient expert-activation footprint 8x.
#: Decode is unaffected: its token count is one 32-row tile either way.
DEFAULT_MOE_GROUP_TOKENS = 32

#: Environment variable that selects the gathered routed-expert path. See :data:`MOE_GATHER_EXPERTS`.
MOE_GATHER_ENV_VAR = "ORNITH_MOE_GATHER"


def _read_bool_env(name: str) -> bool:
    """Read a strict boolean environment variable without enabling on a typo."""

    value = os.environ.get(name, "").strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return False
    if value in ("1", "true", "on", "yes"):
        return True
    raise ValueError(f"{name} must be one of 1/true/on/yes or 0/false/off/no; got {os.environ.get(name)!r}")


#: Select the router-index-native prefill MoE. Unlike :data:`MOE_GATHER_EXPERTS`, this path never
#: materializes dense ``[tokens, experts]`` routing: it preserves top-k's FP32 scores and UINT32
#: indices through device-native dispatch, fused experts and fused weighted combine. It is still
#: prefill-only; decode retains the sparse path.
MOE_TOPK_NATIVE_ENV_VAR = "ORNITH_MOE_TOPK_NATIVE"
MOE_TOPK_NATIVE = _read_bool_env(MOE_TOPK_NATIVE_ENV_VAR)

#: Largest native planner/combine span. Real flattened prefill calls are split on this boundary:
#: B1x2048 -> two composite invocations and B4x2048 -> eight, always in token order.
MOE_TOPK_NATIVE_SUB_CHUNK = 1024


def _topk_native_chunk_ranges(tokens: int, sub_chunk: int = MOE_TOPK_NATIVE_SUB_CHUNK) -> tuple[tuple[int, int], ...]:
    """Exact token-order decomposition for the native primitive's bounded L1 planner.

    Production calls are deliberately all-native or not-native: a ragged tail would mix routing
    representations and make the invocation proof ambiguous, so this helper rejects it and the
    caller uses the accepted gathered fallback for the whole layer call.
    """

    tokens, sub_chunk = int(tokens), int(sub_chunk)
    if tokens <= 0 or sub_chunk <= 0:
        raise ValueError(f"native top-k chunking needs positive tokens/sub_chunk; got {tokens}/{sub_chunk}")
    if tokens % sub_chunk:
        raise ValueError(f"native top-k tokens ({tokens}) must be divisible by sub-chunk ({sub_chunk})")
    return tuple((start, start + sub_chunk) for start in range(0, tokens, sub_chunk))


#: Replace the two routed ``ttnn.sparse_matmul`` calls with a **gathered** per-expert FFN.
#:
#: ``ttnn.sparse_matmul`` is the single largest item of the prefill window: it computes, for every
#: 32-token group and every expert that group activates, the *whole* 32-row product, so at the
#: measured 41 active local experts per group it evaluates 64 x 41 x 32 = 83 968 row-expert products
#: per layer per 2048-token chunk where the routing needs 2048 x 8 / 4 = 4096 — a 20.5x arithmetic
#: overhead, and it runs on 32 of the 110 worker cores. The gathered path instead materialises, per
#: expert, only the rows that expert was actually routed, and runs
#: ``ttnn.experimental.deepseek_prefill.unified_routed_expert_moe`` over them: one device program for
#: the 64 local experts that fuses gate+up+silu+down, switches weight addresses at expert boundaries,
#: and reads token counts *on device*, so the arithmetic is exactly the 4096 rows routing asks for.
#:
#: **Off by default and selected by one environment variable**, :data:`MOE_GATHER_ENV_VAR`
#: (``ORNITH_MOE_GATHER=1``). The value is read once at import into this module global, which is then
#: read through the module at call time — the same contract :data:`DECODE_COMPACT_ROWS` documents, so
#: a test or probe can flip it with ``monkeypatch.setattr(optimized_decoder, "MOE_GATHER_EXPERTS", …)``.
#: With it cleared, every line below is unreachable and the sparse path runs byte for byte.
#:
#: **It is read at weight-load time as well as at forward time.** The gathered path needs per-expert
#: ``gate_proj``/``up_proj``/``down_proj`` (the fused kernel's layout) while the sparse path needs the
#: packed ``expert_gate_up``/``expert_down``, and the same bytes in two arrangements cannot alias, so
#: the flag being set at load time makes :meth:`OptimizedDecoder._load_moe_weights` upload **both**.
#: That is deliberate rather than wasteful: this switch is prefill-only, so decode still runs the
#: sparse path and its weights have to be there. The cost is the routed-expert weight term twice —
#: 113 MB per device per layer, i.e. about 4.5 GB per device across 40 layers, against the 25.9 GB
#: ``doc/optimized_full_model/footprint.json`` records as free for activations. Sizeable, and the one
#: number a device run should check first.
#:
#: :attr:`OptimizedMoE.gather_experts` records whether the per-expert layout was actually uploaded, so
#: a flag flipped *after* construction can turn the path off (the sparse weights are always there) but
#: never on (the gathered ones may not be). An in-process A/B — build once with the flag set, then
#: flip it between forwards — is therefore valid in one direction only, which is the direction
#: ``test_ccl_modes_agree`` uses: the shipped default is the reference arm.
#:
#: **Validated only with expert parallelism, and enforced.** The compact gathered buffer has the exact
#: static capacity ``top_k * sub_chunk + (TILE - 1) * num_experts_local``: every real assignment plus
#: the worst possible tile-rounding slack for every local expert. That removes the old
#: ``num_experts_local * sub_chunk`` row movement. The fused operator can partition longer lists, but
#: the surrounding permutation and duplicate-weight footprint have only been measured at 64 local
#: experts. Above :data:`MOE_GATHER_MAX_LOCAL_EXPERTS` the path refuses — loudly, at setup, with the
#: validated bound named — and the sparse path runs; see :func:`_gather_support_reason`.
#:
#: **Prefill only, deliberately.** At decode the whole call is one 32-row tile, so a gathered expert
#: region is one tile whatever its real count: the gathered path would evaluate 64 x 32 = 2048 rows
#: for the 256 the routing asks (an 8x overhead, *worse* than the sparse path's 5x at that shape), plus
#: the dispatch/reverse-permutation graph. Decode is dispatch-bound even though the FFN itself is one
#: fused program, so :data:`MOE_GATHER_MIN_TOKENS` is the floor; below it, and for every
#: ``decode=True`` call, the sparse path runs unchanged.
MOE_GATHER_EXPERTS = _read_bool_env(MOE_GATHER_ENV_VAR)

#: Tokens per gathered-expert call.
#:
#: The gathered buffer tight-packs tile-aligned expert regions into
#: ``top_k * sub_chunk + (TILE - 1) * num_experts_local`` rows. This is an exact upper bound, not a
#: capacity factor: the local assignment count is at most ``top_k * sub_chunk`` and rounding each of
#: ``num_experts_local`` counts to a tile adds at most ``TILE - 1`` rows. Device-side counts and an
#: exclusive cumsum provide the dynamic region starts; no assignment is dropped and no host sync is
#: needed. At the shipped 1024-token / 64-expert / top-8 shape this is 10,176 rows instead of 65,536.
#:
#: 1024 is the smallest value at which the *arithmetic* is already optimal: the mean rows per expert
#: is ``sub_chunk * top_k / num_experts`` = ``sub_chunk / 32``, so at 1024 that is exactly one 32-row
#: tile and the FFN's tile rounding costs nothing, while at 512 it is 16 rows rounded to 32 (a 2x
#: overhead). Larger values do not improve the arithmetic and do grow the compact transient linearly
#: in ``top_k * sub_chunk``. Its bfloat16 input is 39.8 MiB at the admitted 1024-token / 64-expert
#: shape, before the fused FFN's block-float output.
MOE_GATHER_SUB_CHUNK = 1024

#: Smallest whole-call token count that takes the gathered path. Below it the sparse path runs.
#:
#: Two reasons for a floor rather than "any prefill". Arithmetically, 1024 rows give each of 64 local
#: experts one full 32-row tile on average; shorter spans increasingly pay for padded expert tiles.
#: Operationally, 1024 is the only profitable sub-chunk measured in the real layer and full stack.
#: Smaller aligned tails stay sparse instead of silently broadening the production shape contract.
MOE_GATHER_MIN_TOKENS = 1024

#: Most experts **per device** the gathered path will accept. Above this it refuses, loudly.
#:
#: The compact buffer is no longer linear in ``num_experts_local * sub_chunk`` and the fused operator
#: runs 64 experts in one program, but the duplicate per-expert weight layout and permutation tables
#: still scale with the expert count. EP=4 divides 256 experts into 64 per device, the only geometry
#: this experimental path has run.
#:
#: Without EP — the single-device mesh the unit tests use, where ``num_experts`` is the full 256 — the
#: superseded full-region permutation failed to launch on Blackhole 2026-08-20 with a 2,225,360-byte
#: circular-buffer request against 1,572,864 bytes of L1. Compact dispatch removes that exact forward
#: sort/buffer shape, so the old failure is no longer proof that 256 cannot launch; nevertheless four
#: fused operator partitions and the surrounding glue plus duplicate weight layout remain unmeasured.
#: The guard therefore stays at the only geometry with correctness evidence instead of silently
#: broadening the experiment.
#:
#: 64 rather than a larger round number because 64 is the only per-device expert count that has been
#: run: it is what EP=4 produces on the shipped ``(1, 4)`` mesh, and the value the equivalence,
#: permutation and zero-local-expert tests all measure. A tp=2 deployment (128 per device) may well be
#: fine and is not permitted here, because nothing has measured it.
MOE_GATHER_MAX_LOCAL_EXPERTS = 64

#: Largest gathered sub-chunk the path will accept, i.e. the cap on :data:`MOE_GATHER_SUB_CHUNK`.
#:
#: Equal to the shipped sub-chunk, so the shipped configuration *is* the largest permitted one. That is
#: deliberate: the compact scatter operands and buffer still grow with the sub-chunk, and 1024 is the
#: only value the surrounding fused-FFN path has launched on device. Raising it is a re-validation, not
#: a tuning knob, which is why it is enforced rather than merely documented.
MOE_GATHER_MAX_SUB_CHUNK = 1024


def _gather_support_reason(num_experts: int, sub_chunk: int | None = None) -> str | None:
    """Why the gathered routed-expert path cannot run at this geometry, or ``None`` if it can.

    One predicate, three callers — the weight loader (which skips uploading the per-expert layout when
    the path cannot run, so an unsupported build does not pay 453 MB per layer for weights nothing will
    read), :meth:`OptimizedMoE.prepare_gather_experts` (which logs the reason once at setup) and
    :meth:`OptimizedMoE._gather_ok` (which carries it per call). Keeping it in one place is what stops
    the three from disagreeing and turning a refusal into a ``KeyError`` mid-forward.

    Returns a sentence, not a flag, because every one of these refusals is a fallback to a *slower*
    path and the log has to say which bound was hit and what the measured consequence of exceeding it
    was. ``sub_chunk=None`` checks only the geometry the weight loader knows about.
    """
    if num_experts > MOE_GATHER_MAX_LOCAL_EXPERTS:
        return (
            f"{num_experts} experts per device exceeds MOE_GATHER_MAX_LOCAL_EXPERTS "
            f"({MOE_GATHER_MAX_LOCAL_EXPERTS}); only that EP=4 geometry has run. The fused operator "
            f"can partition a longer list, but the surrounding permutation and duplicate weight "
            f"layout have not been validated above this local-expert count"
        )
    if num_experts % 64:
        # The reverse permutation sorts the expert axis, and `ttnn.sort` needs that axis to be a
        # multiple of 64. True at both 256 and the 64 of an EP shard, but checked, not assumed.
        return f"num_experts ({num_experts}) must be a multiple of 64 for ttnn.sort's sorted axis"
    if sub_chunk is None:
        return None
    if sub_chunk > MOE_GATHER_MAX_SUB_CHUNK:
        return (
            f"sub-chunk {sub_chunk} exceeds MOE_GATHER_MAX_SUB_CHUNK ({MOE_GATHER_MAX_SUB_CHUNK}); "
            f"every permutation-glue L1 footprint grows with it and only {MOE_GATHER_MAX_SUB_CHUNK} "
            f"has been launched on device"
        )
    if sub_chunk % PREFILL_ALIGN or sub_chunk < MOE_GATHER_MIN_TOKENS:
        return (
            f"sub-chunk {sub_chunk} must be a multiple of PREFILL_ALIGN ({PREFILL_ALIGN}) and at "
            f"least MOE_GATHER_MIN_TOKENS ({MOE_GATHER_MIN_TOKENS})"
        )
    return None


def _gather_compact_capacity(tokens: int, num_experts: int, top_k: int) -> int:
    """Exact static row capacity for tile-aligned, tightly packed expert regions.

    A device can own at most ``tokens * top_k`` assignments. Expert ``e`` occupies
    ``TILE * ceil(count[e] / TILE)`` rows, whose padding is at most ``TILE - 1``. Summing the two
    bounds gives this capacity; unlike a probabilistic capacity factor it covers adversarial routing.

    The gathered path additionally requires tile-aligned ``tokens`` and an expert count divisible by
    64, so the returned row count is tile-aligned at every admitted geometry. Keep the assertion here:
    ``unified_routed_expert_moe`` rejects an input whose M dimension is not tile-aligned.
    """
    tokens = int(tokens)
    num_experts = int(num_experts)
    top_k = int(top_k)
    if tokens <= 0 or num_experts <= 0 or top_k <= 0:
        raise ValueError(
            f"compact gathered capacity requires positive tokens/experts/top_k; got " f"{tokens}/{num_experts}/{top_k}"
        )
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")
    capacity = tokens * top_k + (TILE - 1) * num_experts
    if capacity % TILE:
        raise ValueError(
            f"compact gathered capacity ({capacity}) must be tile-aligned; "
            f"tokens={tokens}, num_experts={num_experts}, top_k={top_k}"
        )
    return capacity


#: ``chunk_gated_delta_rule`` internal chunk. 32 is the only value that enables the flat
#: (rank-3 token-major) q/k contract, which is what carries the in-kernel L2 norm.
GDN_CHUNK = 32

#: Channels per ``ttnn.conv1d`` call for the depthwise causal conv.
#:
#: The op cannot serve Ornith's full ``conv_dim = 8192``: height-sharded it runs out of L1 at every
#: legal slice count, at 128 slices and above it refuses the slice count itself, and block-sharded it
#: fails at kernel compile — three distinct blockers, all in
#: ``doc/fused_decoder/logs/probe_conv1d_and_norm.txt`` and catalogued in
#: ``doc/fused_decoder/work_log.md`` §4.4. But a
#: depthwise conv is separable over
#: channels, and 4096 is the widest split that runs. It also happens to be the natural split point:
#: channels ``[0, 4096)`` are exactly Q and K, and ``[4096, 8192)`` are exactly V, so the two halves
#: feed the delta rule directly with no output concatenation.
CONV1D_CHANNELS = 4096

#: How partial RoPE is lowered. See :class:`OrnithFusedRope`. Both modes were measured
#: (``doc/fused_decoder/logs/ab_rope_mode.txt``): ``"full"`` is one op instead of four in prefill
#: (six in decode, which adds two transposes), but it is
#: slower on traced decode — the wider cos/sin table costs more in the gather's tilize than the
#: slice and concat it removes — and indistinguishable on prefill, while needing a 4x larger table
#: (head_dim-wide instead of rope_dim-wide) at the full context. So the measured default is
#: ``"partial"``; ``"full"`` is kept because it is the fewer-op graph, because the knob is what
#: makes that comparison reproducible, and because ``test_rope_mode_equivalence`` uses it to prove
#: the head-dim permutation is self-consistent across Q, K and the KV cache.
DEFAULT_ROPE_MODE = "partial"

#: What every RMSNorm in the layer must hand to its consumer. ``ttnn.rms_norm`` takes no ``dtype``,
#: so its output is its input's dtype and a block-float ``residual_dtype`` would propagate into the
#: token-mixer projection and then into ``nlp_create_qkv_heads_decode``, which accepts only FLOAT32
#: and BFLOAT16. :meth:`OptimizedDecoder._widen_norm_output` restores this, and dispatches nothing
#: at all under the shipped bfloat16 residual.
NORM_OUTPUT_DTYPE = ttnn.bfloat16
_NORM_SAFE_DTYPES = (ttnn.bfloat16, ttnn.float32)

#: What ``ttnn.experimental.deepseek_moe_fast_reduce_nc`` accepts, and what a narrower routed-expert
#: activation is widened to before it. See :meth:`OptimizedMoE._routed_experts`.
_MOE_REDUCE_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b)
MOE_REDUCE_FALLBACK_DTYPE = ttnn.bfloat8_b

_SILU = [ttnn.UnaryOpType.SILU]
_SIGMOID = [ttnn.UnaryOpType.SIGMOID]
_SOFTPLUS = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)]


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_dim(tensor, dim: int, amount: int):
    """Zero-pad ``amount`` elements onto the high end of ``dim``.

    Aliasing warning: when the requested logical padding already fits inside the tensor's
    physical tile padding (e.g. 1 → 32 rows), the result can share ``tensor``'s buffer. Callers
    must therefore never free the pre-pad tensor separately.
    """
    padding = [(0, 0)] * len(tensor.shape)
    padding[dim] = (0, amount)
    return ttnn.pad(tensor, padding, 0.0)


def _slice_owned(tensor, begins, ends):
    """``(slice, owned)`` where ``owned`` says whether the result is a fresh buffer."""
    if all(b == 0 for b in begins) and list(ends) == [int(d) for d in tensor.shape]:
        return tensor, False
    return ttnn.slice(tensor, list(begins), list(ends)), True


def _slice_last(tensor, start: int, end: int):
    """Slice ``[start, end)`` of the last dim, keeping every leading dim whole.

    Rejects a whole-dim range rather than returning one: ``ttnn.slice`` short-circuits a full cover
    to the input tensor itself, so a caller that deallocates the result would free its own input.
    Every shipped call site slices a strict sub-range; the ones that can legitimately want the whole
    tensor branch around this (``_rope_prefill``/``_rope_decode`` at ``rope_dim == head_dim``), and
    anything that cannot know statically should use ``_slice_owned``.
    """
    shape = [int(d) for d in tensor.shape]
    if start == 0 and end == shape[-1]:
        raise ValueError(
            f"_slice_last({start}, {end}) covers the whole last dim of {shape}: ttnn.slice would "
            "alias the input and the caller's deallocate would free it — use _slice_owned"
        )
    begins = [0] * len(shape)
    ends = list(shape)
    begins[-1] = start
    ends[-1] = end
    return ttnn.slice(tensor, begins, ends)


def num_blocks_for_context(context: int, block_size: int = DEFAULT_PAGE_BLOCK_SIZE) -> int:
    """Paged blocks needed to hold ``context`` tokens, rounded to the 32-entry page-table stick."""
    blocks = _align_up(context, block_size) // block_size
    return _align_up(blocks, 32)


def _rope_head_permutation(head_dim: int, rope_dim: int) -> list[int]:
    """Head-dim reordering that turns partial RoPE into a full-width rotate-half.

    Ornith rotates only the first ``rope_dim`` (64) of each ``head_dim`` (256) head, with the
    rotate-half pairing ``(i, i + rope_dim // 2)``. ``ttnn.experimental.rotary_embedding_hf``
    only rotates the *whole* width it is given, pairing ``(j, j + head_dim // 2)``.

    Reordering the head dims so that the rotated pair ``(i, i + rope_dim//2)`` lands on
    ``(i, i + head_dim//2)`` makes the full-width op compute exactly the partial rotation, as long
    as the cos/sin table is 1 / 0 on the pass-through positions (:meth:`OrnithFusedRope`). The
    returned list maps **new position → old position**::

        new[0 : 32]        = old[0 : 32]            # rotate-half "first half" of the 64-wide RoPE
        new[32 : 128]      = old[64 : 160]          # pass-through
        new[128 : 160]     = old[32 : 64]           # rotate-half "second half"
        new[160 : 256]     = old[160 : 256]         # pass-through

    Q and K share the permutation, so ``q·k`` is unchanged; V and ``o_proj`` are untouched, so the
    layer output is bit-for-bit the same math. The KV cache stores permuted K, which is an
    internal detail of this layer.
    """
    half_rope = rope_dim // 2
    half_head = head_dim // 2
    if rope_dim == head_dim:
        return list(range(head_dim))
    if half_rope > half_head:
        raise ValueError(f"rope_dim {rope_dim} cannot exceed head_dim {head_dim}")
    lo = list(range(0, half_rope))
    lo_pass = list(range(rope_dim, rope_dim + (half_head - half_rope)))
    hi = list(range(half_rope, rope_dim))
    hi_pass = list(range(rope_dim + (half_head - half_rope), head_dim))
    perm = lo + lo_pass + hi + hi_pass
    assert sorted(perm) == list(range(head_dim)), "head permutation must be a bijection"
    return perm


class OrnithFusedRope(LightweightModule):
    """Device-resident RoPE tables for :func:`ttnn.experimental.rotary_embedding_hf`.

    Two modes, both of which replace the functional decoder's 10-op hand-written partial rotation
    with the dedicated kernel; ``doc/fused_decoder/README.md`` records the measurement that picks
    the default.

    ``"full"``
        The table is widened to ``head_dim`` with ``cos = 1`` / ``sin = 0`` on the pass-through
        dims. Combined with :func:`_rope_head_permutation` on the Q/K projection and norm weights,
        one full-width rotate-half *is* the partial rotation — **one op, no slice, no concat**.
    ``"partial"``
        The ordinary ``rope_dim``-wide table; the layer slices the rotated head prefix, rotates it
        and concatenates the pass-through tail back on (4 ops in prefill; 6 in decode, which adds two
        transposes to reach the op's expected axis order).
    """

    def __init__(
        self,
        mesh_device,
        config,
        *,
        max_context: int,
        table_context: int | None = None,
        # DEFAULT_ROPE_MODE, not "full": constructing this directly used to silently get the
        # measured-slower head_dim-wide table, while OptimizedDecoder.from_state_dict passed the fast one.
        mode: str = DEFAULT_ROPE_MODE,
        dtype=ttnn.bfloat16,
    ):
        import torch

        if mode not in ("full", "partial"):
            raise ValueError(f"unknown rope mode {mode!r}")
        self.device = mesh_device
        self.mode = mode
        self.head_dim = config.head_dim
        self.rope_dim = config.rope_dim
        self.width = config.head_dim if mode == "full" else config.rope_dim
        # ttnn.experimental.rotary_embedding_hf requires the rotated width to put rotate_half's
        # midpoint on a tile boundary: `width == TILE_WIDTH || width % 64 == 0`
        # (rotary_embedding_hf_device_operation.cpp). The functional decoder's hand-rolled rotation
        # had no such constraint, so this is a narrowing the dedicated op introduces — Ornith's
        # rope_dim of 64 satisfies it, but a config that did not would otherwise reach a raw TT_FATAL.
        if self.width != 32 and self.width % 64:
            raise ValueError(
                f"rotary_embedding_hf cannot rotate a width of {self.width} (rope_dim={self.rope_dim}, "
                f"head_dim={self.head_dim}, mode={mode!r}): it must be 32 or a multiple of 64"
            )
        self.table_rows = int(table_context or max_context)

        half_rope = self.rope_dim // 2
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.rope_dim, 2, dtype=torch.int64).float() / self.rope_dim)
        )
        positions = torch.arange(self.table_rows, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [rows, rope_dim // 2]

        if mode == "full":
            cos = torch.ones(self.table_rows, self.width, dtype=torch.float32)
            sin = torch.zeros(self.table_rows, self.width, dtype=torch.float32)
            slots = ((0, half_rope), (self.head_dim // 2, self.head_dim // 2 + half_rope))
        else:
            cos = torch.empty(self.table_rows, self.width, dtype=torch.float32)
            sin = torch.empty(self.table_rows, self.width, dtype=torch.float32)
            slots = ((0, half_rope), (half_rope, self.rope_dim))
        for dst_lo, dst_hi in slots:
            cos[:, dst_lo:dst_hi] = freqs.cos()
            sin[:, dst_lo:dst_hi] = freqs.sin()

        def upload(t):
            return ttnn.from_torch(
                t.reshape(1, 1, self.table_rows, self.width),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        self.cos_table = upload(cos)
        self.sin_table = upload(sin)

    def prefill_forward(self, start_pos: int, seq_len: int):
        """cos/sin ``[1, 1, seq_len, width]`` for ``[start_pos, start_pos+seq_len)``."""
        end = start_pos + seq_len
        if end > self.table_rows:
            raise ValueError(f"rope window [{start_pos}, {end}) exceeds the {self.table_rows}-row table")
        out = []
        for table in (self.cos_table, self.sin_table):
            t = ttnn.slice(table, [0, 0, start_pos, 0], [1, 1, end, self.width])
            out.append(ttnn.to_layout(t, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        return out[0], out[1]

    def indexed_forward(self, position_idxs):
        """Gather cos/sin at device-resident positions, with no value in the program key.

        ``position_idxs`` is a ROW_MAJOR uint32 ``[1, rows]`` tensor. Decode refreshes one row per
        user; the serving prefill path refreshes one row per physical token. Both use the same
        embedding programs, keyed by tensor shape rather than by the position values.
        """

        cos = ttnn.embedding(position_idxs, self.cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(position_idxs, self.sin_table, layout=ttnn.TILE_LAYOUT)
        rows = int(position_idxs.shape[-1])
        cos = ttnn.reshape(cos, [1, 1, rows, self.width])
        sin = ttnn.reshape(sin, [1, 1, rows, self.width])
        return cos, sin

    def decode_forward(self, rot_idxs):
        """cos/sin ``[1, 1, batch, width]`` gathered at the per-user positions in ``rot_idxs``.

        ``rot_idxs`` is a ``[1, batch]`` uint32 device tensor, so a captured trace only needs its
        contents refreshed. The batch axis takes the place of the sequence axis, which is what lets
        ``rotary_embedding_hf`` run in its (interleaved) prefill mode on a decode step. Its native decode
        mode requires a height-sharded *input* (`rotary_embedding_hf_device_operation.cpp` asserts
        `is_sharded()` and HEIGHT_SHARDED); it asks only that cos/sin be sharded, not height-sharded, so
        the reason the mode is unreachable here is neither of those. See :meth:`_rope_decode`.
        """
        return self.indexed_forward(rot_idxs)


#: How many cores each routed-expert sparse matmul should run on, as a function of how many experts
#: the call can actually activate.
#:
#: The fused stage took the *largest* core count that divided ``Nt`` — 32 for the packed gate/up
#: projection, 64 for the down projection — which pins ``per_core_N`` to 1 and therefore the output
#: block and subblock to 1x1 as well. That is the right shape for a prefill group and the wrong one
#: for a decode step, and the difference is large in both directions. The op's work is
#: ``active_experts`` K-sweeps per output block, so the useful core count tracks the *active* expert
#: count, not ``Nt``: at 8 active experts (batch-1 decode) eight cores with ``per_core_N`` 4/8 and
#: matching output blocks beat the 32/64-core 1x1 form by ~40 %, while at ~162 (a 32-token prefill
#: group, the expected distinct union of 256 draws from 256 experts) the 8-core form loses decisively. The
#: ratio lives in work_log §4.14 beside the artifact that measures it, not here: `check_source_magnitude_words`
#: refuses a spelled-out ratio in this file for the same reason the figure audit refuses a numeral.
#:
#: ``doc/optimized_decoder/logs/probe_sparse_matmul.txt`` sweeps grid, ``in0_block_w``, output
#: block/subblock width and output placement at 8 / 32 / 64 / 162 active experts under the selected
#: BFP4/LoFi policy; the two divisors below reproduce the measured winner at every one of those
#: points. §Sparse in the README tabulates the sweep.
SPARSE_CORES_PER_ACTIVE = {"gate_up": 2, "down": 4}
SPARSE_MIN_CORES = 8
SPARSE_MAX_CORES = 32

#: ``in0_block_w`` cap for the packed routed gate/up matmul, keyed by whether the call got more than the
#: minimum core count — i.e. by the same active-expert bound that chooses the core count.
#:
#: The two tuned points want different values and the margins are not close, so one cap cannot serve both
#: (``doc/optimized_decoder/logs/probe_sparse_matmul.txt``, and README §5.4's generated table checks the
#: shipped row of every active-expert point against the sweep):
#:
#: * batch-1 decode, 8 active experts, 8 cores: a 32-tile inner block beats the whole tiled ``K`` by a
#:   couple of percent, several times the measured spread.
#: * a 32-token prefill group, ~162 active, 32 cores: the whole tiled ``K`` beats 32 tiles by roughly ten
#:   percent — the opposite direction, and by a much larger margin.
#:
#: README §5.4's generated table carries both figures. Shipping one value therefore costs either a fraction
#: of a percent of every decode step or several percent of every prefill window. Review round 6 found the
#: single-cap version paying the decode side of that; keying the cap off the bound the layer already
#: computes costs nothing and pays neither.
SPARSE_GATE_UP_IN0_BLOCK_W = {False: 32, True: 64}


def _sparse_cores(role: str, active_experts: int) -> int:
    """Target core count for ``role`` at an upper bound of ``active_experts`` active experts."""
    per = SPARSE_CORES_PER_ACTIVE[role]
    return max(SPARSE_MIN_CORES, min(SPARSE_MAX_CORES, active_experts // per))


def _sparse_n_tiles(config, role: str) -> int:
    """Tiles of ``N`` the routed matmul of ``role`` produces, which is what a core count must divide.

    Shared with :meth:`OptimizedMoE._sparse_cfg` so the realised core count can be computed before the
    config is built, and named rather than inlined because README §5.4's generated table mirrors the same
    reduction and ``audit_figures.check_mirrored_constants`` compares the two.
    """
    import math

    width = 2 * config.moe_intermediate_size if role == "gate_up" else config.dim
    return max(1, int(math.ceil(width / TILE)))


def _sparse_matmul_config(
    m: int,
    n: int,
    k: int,
    *,
    cores: int = SPARSE_MIN_CORES,
    in0_block_w: int | None = None,
    grid=None,
):
    """``MatmulMultiCoreReuseMultiCast1DProgramConfig`` for one sparse matmul shape.

    The sparse factory requires ``mcast_in0``, ``Kt % in0_block_w == 0``, and that the output blocks
    exactly tile a rectangle of the chosen grid (``num_blocks_total == bounding box``), so the core
    count has to divide ``Nt`` exactly and the rectangle has to be full.

    ``cores`` is a *target*: it is reduced to the largest divisor of ``Nt`` that is no larger, so a
    shape whose ``Nt`` is not a multiple of the target still gets a legal config. ``in0_block_w``
    likewise falls back to the largest legal divisor of ``Kt`` at or below the request.
    """
    import math

    n_tiles = max(1, int(math.ceil(n / TILE)))
    k_tiles = max(1, int(math.ceil(k / TILE)))
    m_tiles = max(1, int(math.ceil(m / TILE)))

    cores = _largest_divisor_at_most(n_tiles, max(1, cores))
    per_core_n = n_tiles // cores
    # Orientation: fill one grid axis first (a column, 1 x cores, widening to a rectangle when the target
    # exceeds the axis), with the axis length taken from the device rather than from a Blackhole constant.
    # This is the shipped rule for every role and core count, and it is the *measured* choice, but only the
    # end-to-end A/B below establishes that — the isolated op disagrees with the layer here, so the op rows
    # alone would ship the wrong rectangle.
    #
    # At the op (`probe_sparse_matmul.txt`, both rectangles back to back at the same in0_block_w and output
    # placement, per-row `spread=`; README §5.4's generated table re-derives all of this from the artifact):
    #
    # Each bullet's `[ladder <active>/<role>]` tag names the generated-ladder rows it describes, and
    # `audit_figures.check_orientation_claims` re-derives the winning rectangle from those rows on every run.
    #
    # These bullets deliberately carry no percentages. Rounds 18-20 each found a ratio here that had been true
    # of an older artifact; round 20's own repair then overstated the `down` gap two-fold; and when round 21
    # added a check to verify the magnitudes rather than ban them, two consecutive sweeps falsified three more
    # bullets without a line of shipped code changing - the `162/gate_up` gap shrank by a fifth of itself, and
    # `64/gate_up` crossed its spread boundary in both directions. A magnitude here has no generator, so it is
    # stale the moment the sweep is re-run. The direction is stable and is what the shipped rule turns on; the
    # figures live in README §5.4's generated table and work_log §4.14's ladder, regenerated from the artifact.
    #
    # * batch-1 decode, 8 active experts, both roles (the tuned decode point) [ladder 8/gate_up 8/down]: the
    #   column wins, and by the largest margin anywhere in the sweep - decisively.
    # * a 32-token prefill group, ~162 active, `gate_up` [ladder 162/gate_up]: the column wins again, by less.
    # * the same group's `down` [ladder 162/down]: the *row* wins - the narrowest of the ladder's decisive
    #   gaps, and the reason this rule was worth testing end to end at all. Among the *tuned* geometries this
    #   is the only point where `down` reaches 32 cores - the largest tuned decode batch, 8, gives a 64-expert
    #   bound and realises 16 - though an untuned decode batch of 32 or more saturates the active bound at 256
    #   and reaches 32 cores too. Review round 12 corrected this comment, which said 64 active was the largest
    #   supported decode batch rather than the largest tuned one.
    # * `gate_up` at 32 cores, 64 active [ladder 64/gate_up]: the column leads here too. This is the ladder's
    #   least stable row - its margin has moved between sweeps, in both directions across the spread boundary,
    #   with no shipped code changing - so the direction is all this comment asserts about it. Counting the
    #   sweeps would be the same mistake one layer up: that count goes stale the next time one runs. The row
    #   leads it only at inner `in0_block_w` blocks the wide phase never selects, since that phase caps at 64,
    #   so none of those is a shipped comparison at all. Which blocks they are is another run-varying list, so
    #   it is left to the probe rather than spelled here. Review round 20 corrected this bullet, which claimed
    #   the row led by reading exactly those unshipped arms; round 22 corrected it again, for calling a
    #   then-decisive row noise; round 24 found the enumeration itself had drifted.
    #
    # So the op rows do argue for a `("down", 32) -> row` rule, and review round 9 asked for one. Taken and
    # measured end to end (`logs/ab_sdpa_decode_grid.txt`, arms alternating build-by-build, three timed builds
    # each), it **loses**: warmed prefill is slower on both layer kinds, every timed build of the row arm behind
    # every build of the column arm, while traced decode is unchanged because decode never builds that grid. The
    # op-level gap does not shrink at the layer, it reverses, and into a share of prefill that the expert groups
    # of one prefill cannot explain by that op alone. The mechanism is not measured and is therefore not
    # claimed; the plausible reading is that the isolated probe holds `in0` still while the layer feeds `down`
    # from the preceding `gate_up`'s output, and the two rectangles do not place that input the same way.
    # Rejected on the layer measurement, which is the number that ships. Exact figures: README §5.4's generated
    # table for the op rows, work_log §4.14 for the A/B - deliberately not repeated here, because a comment
    # cannot be regenerated when the sweep re-runs.
    #
    # Reviews 5, 8 and 9 all landed on this comment: it claimed a universal column win (round 5), then had
    # the `down` sign inverted (round 8), then corrected past the artifact (round 9). The figures above are
    # transcribed from README §5.4's generated table, which checks every row against the artifact including
    # its orientation and prints the gap wherever the shipped choice is not the op-level winner.
    cy = min(cores, int(grid.y) if grid is not None else cores)
    while cores % cy:
        cy -= 1
    cx = cores // cy

    block_w = per_core_n
    # out_subblock_h * out_subblock_w must fit the dest register budget (8 half-tiles without fp32
    # dest accumulation, which the expert compute-kernel config does not enable).
    sub_w = _largest_divisor_at_most(block_w, 8)
    sub_h = _largest_divisor_at_most(m_tiles, max(1, 8 // sub_w))
    requested = in0_block_w if in0_block_w else k_tiles
    in0_block_w = _largest_divisor_at_most(k_tiles, max(1, requested))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=m_tiles,
        out_block_w=block_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


#: Decode program-config geometry per dense projection role: ``(target cores, in0_block_w cap)``.
#:
#: Every dense decode matmul in this layer is a *skinny* one — a single tile of rows against a large
#: weight — so ttnn's own heuristic, which the fused stage relied on, picks a core count from the
#: output width and lands far from the roofline on the narrow-output roles. The values below are the
#: measured winner for each role in ``doc/optimized_decoder/logs/probe_dense_matmul.txt``, which
#: sweeps three families (ttnn heuristic, explicit 1D ``mcast_in0``, and DRAM-sharded with a
#: DRAM width-sharded weight plus L1 width-sharded activation) across core counts, ``in0_block_w``
#: and output placement, under each role's own weight dtype and math fidelity.
#:
#: The DRAM-sharded family lost on every one of the seven roles here, even measured without its
#: activation-reshard cost (README §5.4 has the per-role table, generated from the probe).
#: That matches what ``models/demos/blackhole/qwen36`` reports for Blackhole decode matmuls: the op
#: pins the compute grid to the 8 DRAM banks, and 8 wide-shard cores lose to a large mcast grid on
#: shapes this skinny. §Dense in the README carries the whole table.
#: Which policy field holds each role's **weight** dtype. The 2D prefill config's L1 model sizes `in1` from it,
#: so the model has to know that the shared expert's weights are `shared_dtype` and the router's are
#: `router_dtype`, not `proj_dtype`. Review round 14 made the model dtype-aware but wired one dtype for all seven
#: roles; round 15 pointed out that leaves two groups mis-sized under two of the three shipped policies - latent
#: today because those roles' modelled totals stay well inside L1, and exactly the defect round 14 fixed for the
#: dense projections, left in place for the rest.
DECODE_MATMUL_WEIGHT_FIELD = {
    "attn_in": "proj_dtype",
    "o_proj": "proj_dtype",
    "gdn_in": "proj_dtype",
    "gdn_out": "proj_dtype",
    "shared_in": "shared_dtype",
    "shared_down": "shared_dtype",
    "router": "router_dtype",
}

DECODE_MATMUL_GEOMETRY = {
    # 32 cores / `in0_block_w` 2, not 96 / 8: both were retuned in review round 26, because round 25 moved
    # this role onto a width-sharded `in0` and its geometry had been selected under the DRAM-interleaved
    # family it no longer runs. README §5.4 now ranks it in the family it ships, and `logs/ab_dense_in0_block_w.txt`
    # measures the candidate at the layer - never slower than the old cap on any timed build, faster on most.
    "attn_in": (32, 2),
    "o_proj": (16, 16),
    # `in0_block_w` 2 at the same core count, for the same reason as `attn_in` above.
    "gdn_in": (110, 2),
    "gdn_out": (24, 8),
    # 32, not 80, and the reason is structural rather than measured. `Nt` is 33 tiles, so a target of 80 names an
    # 88-core grid in which 55 cores never receive an output tile, while 32 realises 11x3 = 33 - exactly one core
    # per tile. At the shipped `in0_block_w` the op ladder is **flat within noise** across every target from 24 to
    # 110, and a whole-layer A/B moved nothing measurable, so this is not a latency claim. No figures here: README
    # §1 states as an invariant that this file quotes no run-varying absolute timing, and round 18's replacement
    # text broke that invariant within one round of the round that existed to enforce it. The ladder is in
    # `logs/probe_dense_matmul.txt` and README §5.4's generated table reads it.
    #
    # Review round 17 changed this entry on a claimed op-level win, and round 18 found that claim wrong:
    # the slower band it cited is the `in0_block_w=8` arm, which varies little across core count itself, not a
    # core-count split. Both targets are defensible on the measurements; this one is kept because naming exactly
    # `Nt` cores is the honest spelling of what the op can use, and because README §5.4's generated table then has
    # no row whose shipped geometry differs from the sweep's winner by more than that row's own spread.
    "shared_in": (32, 32),
    "shared_down": (48, 16),
    "router": (32, 32),
}


def _decode_1d_matmul_config(grid, cores: int, m_rows: int, k: int, n: int, *, fp32_acc: bool, in0_cap: int):
    """1D ``mcast_in0`` program config for a skinny decode matmul, or ``None`` if illegal here.

    ``cores`` is a target: the grid is the widest legal rectangle at or below it, so a device with a
    different worker grid still gets a coherent config rather than a validation error.
    """
    import math

    cols = min(grid.x, cores)
    rows = math.ceil(cores / cols)
    if rows > grid.y:
        rows = grid.y
    m_tiles = max(1, int(math.ceil(m_rows / TILE)))
    k_tiles = max(1, int(math.ceil(k / TILE)))
    n_tiles = max(1, int(math.ceil(n / TILE)))
    per_core_n = int(math.ceil(n_tiles / (cols * rows)))
    cap = 4 if fp32_acc else 8
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_tiles % i == 0 and i * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        # mcast_in0: every core streams the whole K, so in0_block_w has to divide the full Kt.
        in0_block_w=_largest_divisor_at_most(k_tiles, in0_cap),
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


#: Bytes per element on device, including the shared exponent block-float dtypes carry (one
#: exponent byte per 16 data).
_DTYPE_BYTES = {
    ttnn.bfloat16: 2.0,
    ttnn.bfloat8_b: 1.0625,
    ttnn.bfloat4_b: 0.5625,
    ttnn.float32: 4.0,
}

#: What an unknown weight dtype is modelled at when sizing `in1` circular buffers. 4.0 rather than bfloat16's 2.0
#: because the only direction that matters is *under*-modelling: too small a number declares a program legal that
#: then throws at construction, which is how `POLICIES["fused-parity"]` was unable to prefill for the whole stage
#: until review round 14. Every shipped policy's weight dtypes are in the table above; this is the guard for a
#: policy someone adds later.
_UNKNOWN_DTYPE_BYTES = 4.0

#: Fraction of the device's total worker L1 the routed-expert intermediates may occupy at their
#: peak. They are ``num_experts`` wide, so their size scales with the MoE call's token count: the
#: shipped 32-token expert group peaks at ~22 MB and belongs in L1, while a 256-token group would
#: ask for a single 136 MB buffer and the allocator refuses it outright. Above the budget the layer
#: keeps them in DRAM, which is exactly what the fused decoder did, so a large group size is slower
#: rather than broken.
#: 0.32 rather than a rounder number for a reason: ``get_max_worker_l1_unreserved_size()`` reports
#: 1 532 032 B, but the allocator's own refusals quote "bank size is 1436800 B" — about 6.5 % less,
#: which is the per-bank reservation it keeps back. Buffer budgets are therefore taken against a
#: fraction that already absorbs that difference, while the *circular-buffer* model in
#: ``_prefill_2d_matmul_config`` uses the unreserved figure directly, because the program build
#: compares against the larger 1 572 864 B ("... beyond max L1 size of 1572864").
EXPERT_L1_BUDGET_FRACTION = 0.32

#: Largest whole-MoE-call token count that still puts the expert intermediates in L1.
#:
#: The per-group buffers are the same size whatever the call's token count is, so the group-size
#: budget above is necessary but not sufficient: what changes with the call is how much L1 the
#: *surrounding* activations are already holding. Measured: batch-1 prefill at the shipped 2048-token
#: chunk and every decode batch fit; a batch-32 prefill does not — the gate/up output's 64 MiB
#: allocation is refused with 622 592 B per bank already in use. One prefill chunk is therefore the
#: bound, and a larger call keeps the fused decoder's DRAM placement: slower, never wrong.
EXPERT_L1_MAX_CALL_TOKENS = DEFAULT_PREFILL_CHUNK


def _prefill_2d_matmul_config(
    grid, m_rows: int, k: int, n: int, *, fp32_acc: bool, l1_per_core: int, in1_bytes: float = 1.0625
):
    """2D ``MatmulMultiCoreReuseMultiCastProgramConfig`` for a large prefill projection.

    ttnn's heuristic already picks this family for these shapes and fills the whole grid, but it
    leaves ``in0_block_w`` at 1, which ``tt-perf-report`` flags on every dense prefill row. Naming
    the config lets the inner block grow to the largest value the ``in1`` circular buffer can hold.
    """
    import math

    gx, gy = grid.x, grid.y
    m_tiles = max(1, int(math.ceil(m_rows / TILE)))
    k_tiles = max(1, int(math.ceil(k / TILE)))
    n_tiles = max(1, int(math.ceil(n / TILE)))
    per_core_m = int(math.ceil(m_tiles / gy))
    per_core_n = int(math.ceil(n_tiles / gx))
    cap = 4 if fp32_acc else 8
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if per_core_m % i == 0 and i * sub_w <= cap)
    in0_cap = max(1, min(PREFILL_MATMUL_IN0_BLOCK_CAP, PREFILL_MATMUL_IN1_TILE_BUDGET // per_core_n))
    in0_block_w = _largest_divisor_at_most(k_tiles, in0_cap)
    # A named config fixes the whole block geometry, so its circular buffers are fixed too, and at a
    # large prefill batch the *output* block alone can exceed L1 (per_core_M 13 x per_core_N 27 at
    # batch 32). Rather than shrink the block until it is no longer the measured winner, hand those
    # shapes back to ttnn's heuristic, which sizes itself: this config exists for the single-user
    # prefill it was swept at, and batch-32 prefill is a correctness path, not a latency target.
    #
    # The model is the 2D factory's own arithmetic
    # (`matmul_multicore_reuse_mcast_2d_program_factory.cpp`, `in0_CB_size` / `in1_CB_size` /
    # `out_CB_size`): in0 and in1 are double-buffered at MCAST_INPUT_BUFFERING_DEPTH, the output CB
    # is `out_block_h * out_block_w` and is *not* double-buffered, and interm0 is in-place with the
    # output whenever the output is interleaved (`do_not_inplace_interm0_out_CB` is false), which it
    # always is here. `out_block_h`/`out_block_w` default to `per_core_M`/`per_core_N`.
    # `in1_bytes` is the WEIGHT dtype's bytes per element, from the policy, not a constant. It was hardcoded
    # at BFP8's 1.0625 until review round 14's chain turned this up: under `POLICIES["fused-parity"]` the
    # weights are bfloat16, so the real `in1` circular buffers are ~1.9x what the model predicted, the
    # model said the program fit, and program construction threw
    # `Statically allocated circular buffers ... grow to ... beyond max L1`. A model that only holds for
    # one dtype silently mis-sizes every other policy.
    tile_bytes_in0 = 2 * TILE * TILE
    tile_bytes_in1 = in1_bytes * TILE * TILE
    tile_bytes_out = 2 * TILE * TILE
    estimate = MATMUL_CB_MODEL_OVERHEAD * (
        2 * per_core_m * in0_block_w * tile_bytes_in0
        + 2 * in0_block_w * per_core_n * tile_bytes_in1
        + per_core_m * per_core_n * tile_bytes_out
    )
    if estimate > l1_per_core:
        return None
    if m_rows > DEFAULT_PREFILL_CHUNK:
        # Swept at batch-1 prefill, one chunk of rows. Above that the modelled circular-buffer total
        # stops predicting the build: at batch 32 (per_core_M 13) the program asks for 1 684 416 B
        # against a 1 572 864 B limit while the model says 1 281 638. Rather than tune a fudge factor
        # into correctness, larger prefill activations keep ttnn's heuristic, which sizes itself.
        return None
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


def _largest_divisor_at_most(value: int, cap: int) -> int:
    for candidate in range(min(cap, value), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


#: Largest activation height, in tiles, that still takes a decode program config. Above it the
#: matmul is a prefill-shaped one and ttnn's 2D heuristic is the right family. 8 tiles covers every
#: decode batch up to 8 at one tile per batch entry.
DECODE_MATMUL_MAX_M_TILES = 8

#: Cap on ``per_core_M * in0_block_w``, i.e. the tiles of activation a core buffers per K block.
#: The ``in0`` circular buffer is that many tiles, double-buffered, and it is charged against the
#: same L1 the interleaved activations and expert intermediates live in. At ``per_core_M`` 1 it is
#: never binding; at ``per_core_M`` 8 an uncapped ``in0_block_w`` of 32 asks for 256 tiles — half a
#: megabyte per core before double-buffering — and the program build fails with "statically
#: allocated circular buffers ... clash with L1 buffers".
DECODE_MATMUL_IN0_TILE_BUDGET = 64

#: Float32 ``[batch, num_v_heads, 1, head_dim]`` vectors :meth:`OptimizedDecoder._delta_rule_step`
#: holds in L1 at its peak: ``q_row``, ``k_row``, ``v_row``, ``v_read``, the ``v_row - v_read``
#: temporary and ``delta``. Each is ``batch * num_v_heads`` **tile rows** — the token axis is 1 and
#: pads to a whole tile per (batch, head) — so the step's L1 working set scales with the batch even
#: though :data:`DECODE_COMPACT_ROWS` makes the residual and the projections one tile row at every
#: batch. :meth:`OptimizedDecoder._gdn_decode` sizes the *bf16* head-major relayout's placement
#: against this working set and :attr:`OptimizedDecoder._l1_budget`, because that is what shares the
#: L1 with it. At 32 value heads, a 128 head dim and the shipped 51.4 MiB budget the head tensors
#: stay in L1 through batch 17 and go to DRAM above it, which is where they were before the fold.
_STATE_STEP_L1_VECTORS = 6

#: Cap on ``in0_block_w * per_core_N`` for the 2D prefill program configs — the ``in1`` circular
#: buffer, in tiles. ``in0_block_w`` 16 is the best value for the narrow-output prefill roles but
#: fails to build for the wide ones (``attn_in`` at ``per_core_N`` 27, ``gdn_in`` at 36) with
#: "statically allocated circular buffers ... clash with L1 buffers"; this budget picks 8 for those
#: two and 16 for the rest, which is exactly the measured winner for all six roles in
#: ``doc/optimized_decoder/logs/probe_prefill_matmul.txt`` — read the ``in0=DRAM`` rows, which are the
#: arm whose L1 state matches the shipped graph. (Review round 4 caught the ``in0=L1`` arm of an
#: earlier run of that probe holding its 8 MB activation copy resident across *both* arms, which made
#: ``gdn_in`` at ``in0_block_w`` 8 — the geometry this budget selects and the layer runs — appear to
#: fail to build. The probe now allocates that copy only during its own pass.)
PREFILL_MATMUL_IN1_TILE_BUDGET = 320
PREFILL_MATMUL_IN0_BLOCK_CAP = 16

#: Fudge factor on the modelled circular-buffer total for a 2D matmul. The model below is the
#: factory's own arithmetic, but it omits the small per-core CBs (bias, sharded-in0 staging,
#: alignment) that the build also charges. Calibrated against the one geometry that is known to
#: fail: ``per_core_M`` 13 x ``per_core_N`` 27 x ``in0_block_w`` 8 at prefill batch 32, where the
#: build reports 1 684 416 B and the model gives 1 614 848 B — 4.3 % low. 1.05 keeps the two
#: measured-good geometries (94 % of L1) on the legal side of the two measured-bad ones (111 %).
MATMUL_CB_MODEL_OVERHEAD = 1.05


def _worker_l1_bytes() -> int:
    """Usable worker L1 per core, in bytes.

    ``ttnn.get_max_worker_l1_unreserved_size()`` is the authoritative value (1 532 032 B on
    Blackhole: 1536 KiB minus the reserved region). It is deliberately *not* read off the mesh
    device: ``ttnn.MeshDevice`` exposes no L1-size attribute, and an earlier revision of this file
    asked for one through ``getattr(mesh_device, "l1_size_per_core", lambda: 1 << 20)()`` — which
    silently returned the 1 MiB fallback on every call, i.e. 67 % of the real value. That is what
    review round 1 of this stage caught: the understated budget turned the shipped 2D prefill config
    off on the two widest projections while the docs claimed it was on.
    """
    try:
        return int(ttnn.get_max_worker_l1_unreserved_size())
    except Exception:  # noqa: BLE001 - keep a conservative floor rather than crashing at import
        logger.warning("ttnn.get_max_worker_l1_unreserved_size() unavailable; assuming 1 MiB of worker L1")
        return 1 << 20


def _physical_rows(shape) -> int:
    """Rows a TILE-layout tensor of this shape actually occupies.

    Not the logical row count: the second-to-last dim is padded up to a tile and every leading dim
    multiplies whole tiles of it, so a rank-3 decode activation ``[batch, 1, dim]`` is ``batch * 32``
    rows, not ``batch``. Both the matmul's ``per_core_M`` and the sharded norm's shard height are
    counted in these rows, and getting it wrong fails at op validation
    (``num_blocks_total <= num_cores`` / ``!shard_grid_fit_error.has_value()``) rather than silently.
    """
    dims = [int(d) for d in shape]
    if len(dims) < 2:
        return TILE
    rows = _align_up(dims[-2], TILE)
    for dim in dims[:-2]:
        rows *= dim
    return rows


#: Fold the batch axis of a **decode** activation into the tile row axis for the whole token-mixer
#: path: the residual stream, the four dense projections (``attn_in``, ``o_proj``, ``gdn_in``,
#: ``gdn_out``), their slices, the conv taps, the persistent conv history and the two residual
#: RMSNorms all run on ``[1, 1, batch, dim]`` instead of ``[batch, 1, dim]``.
#:
#: A tile's row axis is the *sequence* axis, so ``[batch, 1, dim]`` occupies ``batch`` whole tile
#: rows — ``batch * 32`` physical rows — of which ``batch`` carry data. At batch 8 that is 256
#: physical rows for 8 useful ones, and every elementwise op, norm and matmul in the mixer pays for
#: all of them; folded it is one tile row at every batch up to 32. This is the layout the MoE call
#: site has always used (:meth:`OptimizedDecoder._block` reshapes to ``[1, 1, tokens, dim]`` before
#: the FF norm) and the layout ``MultichipDecoder._all_reduce``'s ``CCL_COMPACT_ROWS`` produced
#: around the collective *only*; this extends the same fold to the whole decode mixer.
#:
#: The fold is a pure logical reshape — ``[batch, 1, dim]`` and ``[1, 1, batch, dim]`` enumerate the
#: same elements in the same row-major order — so no arithmetic changes and no PCC should move. At
#: batch 1 the two shapes describe the same tile, ``ttnn.reshape`` returns a view, and the fold
#: costs nothing; that is why extending it past the collective was worth nothing until this stage
#: measured a batch above 1, and it is also why the batch-1 graph is unchanged op for op.
#:
#: It does **not** add to the block's reshape count, it subtracts: the fold at the top of the block
#: replaces the ``ff_in`` reshape the MoE needed and the unfold at the bottom replaces the ``ff_out``
#: one, one for one, and on top of that a ``full_attention`` decode step loses the 9216-wide ``fused``
#: reshape and the gate reshape inside :meth:`OptimizedDecoder._attention_decode`, and a multichip
#: layer of either kind loses ``CCL_COMPACT_ROWS``' fold/view-back **pair** on the mixer collective
#: (the MoE collective's operand was already rank-4, so that guard never fired there). Net per decode
#: layer at batch > 1: 2 reshapes fewer on a single-chip ``full_attention`` layer, 4 on a multichip
#: one, 2 on a multichip ``linear_attention`` layer, 0 on a single-chip ``linear_attention`` layer.
#:
#: Two knock-on effects, both intended, both from rules that already existed:
#:
#: * :attr:`OptimizedDecoder.NORM_SHARD_MAX_M_TILES` (4 tile rows) admits the two residual norms at
#:   *every* servable batch, where unfolded they fell out above batch 4. No call site changes —
#:   :meth:`OptimizedDecoder._norm` keys the decision on ``_physical_rows`` — and the shard carry
#:   into the in-projection keeps working because ``per_core_M`` is then 1, which is exactly the
#:   condition :meth:`OptimizedDecoder._shard_feeds_projection` checks.
#: * :data:`DECODE_MATMUL_MAX_M_TILES` (8) selects the ``per_core_M`` 1 config for the four
#:   token-mixer roles at every batch instead of one config class per batch, so ``o_proj``'s
#:   ``in0_block_w`` stops narrowing with the batch (:data:`DECODE_MATMUL_IN0_TILE_BUDGET` is
#:   divided by the M tiles) and batches 9..32 stop falling back to ttnn's heuristic entirely.
#:
#: One limitation, and it is about *dtype* rather than shape. The fold moves rows across tile
#: boundaries, so with the shipped ``residual_dtype=bfloat16`` it is bit-exact (bfloat16 carries no
#: shared exponent, so a relayout is a copy) but with a **block-float** residual - the bfloat8_b
#: candidate ``doc/datatype_sweep`` rejected on cost - a batch above 1 would re-tile BFP8 data and
#: could re-derive its shared exponents. At batch 1 the fold is a view, so that arm is unaffected
#: where it is actually exercised (``test_the_residual_and_logits_dtypes_reach_the_ops`` runs it at
#: batch 1). A block-float residual at batch > 1 is neither shipped nor measured; it would need this
#: fold assessed before it could be.
#:
#: Read through the module global at call time, never captured at import, so a probe can flip it the
#: way ``doc/multichip_decoder/logs/probe_decode_batch.py`` flips ``CCL_COMPACT_ROWS``. With it off
#: the layer runs the pre-fold graph exactly, including the unfolded conv history.
DECODE_COMPACT_ROWS = True


def _free_unless_aliased(tensor, keep) -> None:
    """``ttnn.deallocate(tensor)``, unless it shares a device buffer with ``keep``.

    ``ttnn.reshape`` returns a *new* tensor when the request needs a relayout and may return a view
    over the same buffer when it does not. Freeing one side of such a view frees the survivor's
    storage, which is a use-after-free rather than a leak, and the two cases are exactly the two
    batches :data:`DECODE_COMPACT_ROWS` folds: at batch > 1 the fold changes the physical row count
    and a relayout is dispatched, at batch 1 it does not and the view is the caller's own tensor.
    Buffer addresses are what ``deallocate`` operates on, so they are what is compared — an
    object-identity check would not notice the aliasing case (review round 3 of the multichip
    stage, where this helper started life as ``MultichipDecoder._all_reduce``'s guard).
    """
    if tensor is keep:
        return
    # `buffer_address` raises for three different reasons (`pytensor.cpp`): not a device tensor, not
    # allocated, and **per-core allocated** ("do not have a single address"). Only the first two mean
    # "nothing to alias". On the third the addresses simply cannot be compared, and falling through to
    # the free is the use-after-free this helper exists to prevent — so that case returns instead.
    # Nothing here allocates per-core today; the helper is written for the case where something does.
    if any(t.is_per_core_allocated() for t in (tensor, keep)):
        return
    try:
        if tensor.buffer_address() == keep.buffer_address():
            return
    except RuntimeError:  # host or unallocated tensor: no device buffer to alias
        pass
    ttnn.deallocate(tensor)


class _ProjectionConfigs:
    """Per-role decode matmul program configs, built once and cached by (role, M tiles).

    Construction is pure Python — no device work and no ``torch`` — so a first call inside a forward
    pass is not a host round trip, and a captured trace only ever replays the resulting matmul.
    """

    def __init__(self, mesh_device, policy=None):
        self.grid = mesh_device.compute_with_storage_grid_size()
        self.l1_per_core = _worker_l1_bytes()
        #: Bytes per element of each role's **weight**, so the 2D prefill config's L1 model sizes that role's
        #: `in1` circular buffers for the tensor it actually reads under the policy that is running.
        self.in1_bytes = {
            role: _DTYPE_BYTES.get(getattr(policy, field, None), _UNKNOWN_DTYPE_BYTES)
            if policy is not None
            else _UNKNOWN_DTYPE_BYTES
            for role, field in DECODE_MATMUL_WEIGHT_FIELD.items()
        }
        self._cache: dict[tuple, object] = {}

    def get(self, role: str, rows: int, k: int, n: int, *, fp32_acc: bool, decode: bool = True):
        """Program config for ``role`` at ``rows`` activation rows, or ``None`` to use the default.

        ``decode=False`` returns the 2D prefill config for the same role instead. The shape alone
        cannot separate the two phases — a 128-token prefill block is four tile rows, exactly like a
        batch-4 decode step — and the two families are tuned at, and only measured at, their own
        shapes.
        """
        if role not in DECODE_MATMUL_GEOMETRY:
            return None
        if not decode:
            key = ("prefill", role, int(rows), int(n))
            if key not in self._cache:
                self._cache[key] = _prefill_2d_matmul_config(
                    self.grid,
                    int(rows),
                    int(k),
                    int(n),
                    fp32_acc=fp32_acc,
                    l1_per_core=self.l1_per_core,
                    in1_bytes=self.in1_bytes.get(role, _UNKNOWN_DTYPE_BYTES),
                )
            return self._cache[key]
        m_tiles = max(1, (int(rows) + TILE - 1) // TILE)
        if m_tiles > DECODE_MATMUL_MAX_M_TILES:
            return None
        key = (role, m_tiles)
        if key not in self._cache:
            cores, in0_cap = DECODE_MATMUL_GEOMETRY[role]
            in0_cap = max(1, min(in0_cap, DECODE_MATMUL_IN0_TILE_BUDGET // m_tiles))
            self._cache[key] = _decode_1d_matmul_config(
                self.grid, cores, int(rows), int(k), int(n), fp32_acc=fp32_acc, in0_cap=in0_cap
            )
        return self._cache[key]


def _conv_compute_config(arch):
    """Compute-kernel config for the depthwise conv, shared by weight preparation and the forward.

    ``prepare_conv_weights`` bakes the compute config into the prepared layout, so the config used at
    setup and the one passed to :func:`ttnn.conv1d` must be the same object shape — hence one
    definition rather than two matching literals.
    """
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _conv1d_host_weights(conv_w, config):
    """Host-side ``ttnn.conv1d`` weight tensors, one per :data:`CONV1D_CHANNELS` block.

    ``ttnn.prepare_conv_weights`` rejects a device tensor, so the host copies are kept and the
    per-(batch, length) preparation happens in :meth:`OptimizedDecoder.allocate_state`.
    """
    channels = CONV1D_CHANNELS
    kernel = config.linear_conv_kernel_dim
    if config.conv_dim % channels:
        return []
    return [
        ttnn.from_torch(
            conv_w[idx * channels : (idx + 1) * channels].reshape(channels, 1, 1, kernel),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        for idx in range(config.conv_dim // channels)
    ]


def _drop_prepared(prepared, phys):
    """Discard one length's prepared weights, freeing the device buffers rather than dropping the ref.

    ``prepare_conv_weights`` always moves its result to the device, so a dropped entry is device
    memory. At batch 32 every one of the 16 candidate lengths is dropped, and the same argument the
    probe input is freed under applies here.
    """
    for weight in prepared.pop(phys, ()):
        ttnn.deallocate(weight)


def _prepare_conv1d_weights(mesh_device, host_weights, config, prefill_chunk, batch, compute_config):
    """Pre-prepare ``ttnn.conv1d`` weights for every physical block length the layer can produce.

    ``ttnn.prepare_conv_weights`` is a **host** call whose output layout depends on the batch and the
    input length, so it cannot happen inside a forward pass without reintroducing a host round trip.
    Every internal prefill block is a multiple of :data:`PREFILL_ALIGN` and at most ``prefill_chunk``
    tokens, and the conv sees ``kernel-1`` extra history rows, so the whole set of input lengths is
    known at ``allocate_state`` time and is small (16 for the default 2048-token chunk).

    Returns ``{physical_block_length: [prepared weight per CONV1D_CHANNELS block]}``. A length that
    could not be prepared is simply absent and the layer falls back to the FIR form for it, so a
    ttnn change that breaks weight preparation or the validating execution below costs performance
    rather than correctness.

    That guarantee covers *this* function only. ``_conv1d_halves`` calls ``ttnn.conv1d`` unguarded, so
    a length that passes the probe here and then fails in a forward raises rather than falling back.
    It is not a theoretical gap: the dominant refusal class is a pressure-dependent per-bank L1
    allocation failure, and this probe runs at ``allocate_state`` time when the large prefill
    activations are not yet resident. README §9 item 8 records it.
    """
    channels = CONV1D_CHANNELS
    kernel = config.linear_conv_kernel_dim
    if not host_weights:
        return {}
    import torch

    conv_cfg = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)

    prepared: dict[int, list] = {}
    for phys in range(PREFILL_ALIGN, prefill_chunk + 1, PREFILL_ALIGN):
        length = phys + kernel - 1
        try:
            prepared[phys] = [
                ttnn.prepare_conv_weights(
                    weight_tensor=host,
                    weights_format="OIHW",
                    in_channels=channels,
                    out_channels=channels,
                    batch_size=batch,
                    input_height=1,
                    input_width=length,
                    kernel_size=(1, kernel),
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, 1),
                    has_bias=False,
                    groups=channels,
                    device=mesh_device,
                    input_dtype=ttnn.bfloat16,
                    conv_config=conv_cfg,
                    compute_config=compute_config,
                    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                for host in host_weights
            ]
        except Exception:  # noqa: BLE001 - an unpreparable length falls back to the FIR path
            _drop_prepared(prepared, phys)
            continue
        # Preparing the weights is not proof the op will run: the conv's circular buffers are sized
        # at program build and at large batches exceed L1 ("Statically allocated circular buffers ...
        # grow to N B which is beyond max L1 size"), and — more often, above batch 4 — the sharded
        # input's per-bank allocation is refused outright (bank_manager.cpp:462). Neither is
        # predictable from the shape alone, so each (batch, length) is executed
        # once here, on a zero input, and dropped if it throws. That keeps the decision at setup
        # time and out of the forward path, and it is exact rather than a guessed bound.
        probe = None
        try:
            probe = ttnn.from_torch(
                torch.zeros(batch, length, 1, channels, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            out = ttnn.conv1d(
                input_tensor=probe,
                weight_tensor=prepared[phys][0],
                device=mesh_device,
                in_channels=channels,
                out_channels=channels,
                batch_size=batch,
                input_length=length,
                kernel_size=kernel,
                stride=1,
                padding=0,
                dilation=1,
                groups=channels,
                dtype=ttnn.bfloat16,
                conv_config=conv_cfg,
                compute_config=compute_config,
                slice_config=ttnn.Conv2dL1FullSliceConfig,
                return_output_dim=False,
                return_weights_and_bias=False,
            )
            ttnn.deallocate(out)
        except Exception:  # noqa: BLE001 - this (batch, length) does not fit; use the FIR path
            _drop_prepared(prepared, phys)
        finally:
            # Freed on both paths and explicitly: at batch 32 this input is ~0.5 GB, and the failing
            # (batch, length) pairs are exactly the large ones, so leaving it to refcounting would
            # hold the largest probes longest and across the next iteration's allocation.
            if probe is not None:
                ttnn.deallocate(probe)
    return prepared


class OptimizedMoE:
    """Router + 256 routed experts + gated shared expert, graph-fused.

    Same math as :class:`~models.autoports.ornith_ai_ornith_1_0_35b.tt.moe.OrnithMoE`; five
    rewrites:

    1. **Shared-LHS packing of the shared expert.** Its ``gate``/``up`` projections and its sigmoid
       router share the input, so they run as one matmul over a concatenated weight and are sliced
       apart (the router column is zero-padded to a tile so the slice offsets stay tile-aligned).
    2. **Shared-LHS packing of the routed experts.** ``gate`` and ``up`` become one ``2·I``-wide
       sparse matmul. The win is core count: the sparse factory gives each core one N-tile output
       block and cannot split M (a group is a single 32-row tile), so ``N = I`` is pinned to
       ``I/32`` cores and ``N = 2·I`` doubles that.
    3. **Score placement.** ``down_proj`` is linear, so scaling its *input* by the router score is
       identical to scaling its output — and the input is ``moe_intermediate`` (512) wide rather
       than ``hidden`` (2048) wide, i.e. 4× less elementwise traffic over the 256-expert axis.
    4. **One expert-axis permute.** ``silu(gate) * up`` is evaluated in the sparse matmul's native
       ``[1, groups, 1, E, 32, N]`` layout (with the SiLU folded into the multiply), so only the
       product is relaid out to the expert-major ``[1, E, tokens, N]`` the down projection wants.
       With a single 32-token group — which, at the shipped ``moe_group_tokens = 32``, is every
       call, prefill included — that relayout is a reshape rather than a permute.
    5. **One router per call.** ``topk`` and the scatter chain are single-core on a 256-wide last
       dim and barely scale with the token count, so routing runs once for the whole call and each
       expert group takes a tile-aligned slice of the dense score vector.
    """

    def __init__(
        self,
        mesh_device,
        config,
        weights,
        *,
        group_tokens: int = DEFAULT_MOE_GROUP_TOKENS,
        policy: PrecisionPolicy = DEFAULT_POLICY,
        expert_mem_config=ttnn.L1_MEMORY_CONFIG,
    ):
        self.device = mesh_device
        self.cfg = config
        self.w = weights
        self.group_tokens = group_tokens
        self.policy = policy
        self.proj_cfgs = _ProjectionConfigs(mesh_device, policy)
        #: See :attr:`OptimizedDecoder._decode_phase`.
        self._decode_phase = False

        # Expert matmuls. `fp32_dest_acc_en` halves the matmul dest register budget and is a known
        # Blackhole corruption source for this op family, so it stays off by default.
        self.expert_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.expert_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=policy.expert_fp32_acc,
            packer_l1_acc=policy.expert_packer_l1_acc,
        )
        #: Shared expert (dense) matmuls.
        self.shared_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.shared_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=policy.shared_fp32_acc,
            packer_l1_acc=policy.shared_packer_l1_acc,
        )
        #: Router matmul + the softmax over its kept top-k.
        self.dense_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.router_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=policy.router_fp32_acc,
            packer_l1_acc=False,
        )

        #: Where the ``num_experts``-wide expert intermediates live *when they fit*. The fused stage
        #: put every one of them in DRAM; at the shipped 32-token expert group they are ~22 MB at
        #: their peak, so L1 keeps the zero-fill, the SwiGLU, the score multiply, the unpacking
        #: slices and the expert reduction off the DRAM bus entirely.
        #: :meth:`_expert_mem` falls back to DRAM for group sizes whose intermediates do not fit.
        self.expert_mem_config = expert_mem_config
        grid = mesh_device.compute_with_storage_grid_size()
        self.grid = grid
        self.expert_l1_budget = int(EXPERT_L1_BUDGET_FRACTION * grid.x * grid.y * _worker_l1_bytes())
        self._expert_mem_cache: dict[tuple, object] = {}
        #: Token count of the whole MoE call in flight; see EXPERT_L1_MAX_CALL_TOKENS.
        self._call_tokens = 0
        # `in0_block_w`, swept over the full divisor ladder of Kt under the selected BFP4/LoFi policy.
        #
        # `down` takes the largest legal divisor (16): it improves monotonically with the inner block and
        # then flattens. `gate/up` does NOT have one best value — the cap depends on the realised core
        # count exactly as the geometry does, see SPARSE_GATE_UP_IN0_BLOCK_W. Review round 6 found the
        # single-cap version costing 2.2 % of the largest decode op, beyond the measured spread, because
        # the value that wins a 32-token prefill group loses a batch-1 decode step. Round 7 then found
        # three stale copies of the old single-cap claim, including two stacked here; this is the one.
        self.gate_up_in0_block_w = {
            bound: _largest_divisor_at_most(config.dim // TILE, cap)
            for bound, cap in SPARSE_GATE_UP_IN0_BLOCK_W.items()
        }
        self.down_in0_block_w = _largest_divisor_at_most(config.moe_intermediate_size // TILE, 16)
        self._sparse_cfg_cache: dict[tuple, object] = {}
        #: Persistent all-zero scatter targets for the router, keyed by decode shape. See `_router_zeros_for`.
        self._router_zeros: dict[tuple, object] = {}
        self.output_tile = ttnn.Tile([TILE, TILE])
        #: Whether the gathered routed-expert path is *available* — i.e. whether the weight loader
        #: built the per-expert ``gate``/``up``/``down`` triples the fused kernel needs. Recorded here,
        #: at construction, rather than re-read from :data:`MOE_GATHER_EXPERTS` in the forward: the two
        #: paths carry different weights and only one set exists, so flipping the module flag after the
        #: layer is built must not move the forward onto weights that were never uploaded.
        self.gather_experts = all(
            isinstance(weights.get(key), (list, tuple)) and len(weights[key]) == config.num_experts
            for key in ("expert_gate_list", "expert_up_list", "expert_down_list")
        )
        #: Per-token-count constants for the gathered path (token ramp, expert ramp, compact-scatter
        #: zero base, local-expert index table). Built on the host, so built at *setup* — the first
        #: gathered call for a given sub-chunk size allocates them and every later call reuses them.
        #: Dynamic expert region offsets are deliberately *not* here: they are derived on device from
        #: this call's counts. Prefill is not trace-captured, so a first-use build is legal here in a
        #: way it would not be at decode; the dict is keyed by row count and the tensors are held for
        #: the layer's lifetime.
        self._gather_const: dict[int, tuple] = {}
        #: Whether :meth:`prepare_gather_experts` has run and populated the sub-chunk ladder.
        self._gather_ready = False
        #: The setup-time reason the gathered path is unavailable, or ``None`` when it is available.
        #: Set by :meth:`prepare_gather_experts`; read by :meth:`_gather_reason` and by the tests.
        self.gather_refusal: str | None = "prepare_gather_experts has not run for this layer"

        # Router-index-native prefill capability and proof counters. `calls` counts actual composite
        # invocations (one per <=1024-token sub-chunk), while `layer_calls` counts completed public
        # MoE calls. The serving capability aggregates `calls` per layer, so B1/B4 produce uniform
        # deltas of 2/8 across all 40 layers and cannot pass on an environment selection alone.
        self.topk_native_weights_loaded = self.gather_experts
        self.topk_native_ready = False
        self.topk_native_refusal: str | None = "prepare_topk_native has not run for this layer"
        self.topk_native_global_to_local = None
        self.topk_native_calls = 0
        self.topk_native_layer_calls = 0
        self.topk_native_subchunks = 0
        # Only eligible prefill attempts that cannot enter the selected native path increment this.
        # Decode and deliberately ineligible short/padded calls do not.
        self.topk_native_fallbacks = 0

    @property
    def topk_native_selected(self) -> bool:
        """Live selection state; tests may flip the module flag after import."""

        return bool(MOE_TOPK_NATIVE)

    @property
    def topk_native_enabled(self) -> bool:
        """Whether this concrete layer has every object needed for native execution."""

        return bool(
            self.topk_native_selected
            and self.topk_native_weights_loaded
            and self.topk_native_ready
            and self.topk_native_refusal is None
            and self.topk_native_global_to_local is not None
        )

    def topk_native_status(self) -> dict:
        """Per-layer status consumed by the serving capability aggregator."""

        return {
            "selected": self.topk_native_selected,
            "weights_loaded": bool(self.topk_native_weights_loaded),
            "ready": bool(self.topk_native_ready),
            "enabled": self.topk_native_enabled,
            "refusal": self.topk_native_refusal,
            "calls": int(self.topk_native_calls),
            "fallbacks": int(self.topk_native_fallbacks),
            "layer_calls": int(self.topk_native_layer_calls),
            "subchunks": int(self.topk_native_subchunks),
        }

    def _expert_mem(self, tokens: int):
        """Memory config for this call's ``num_experts``-wide intermediates.

        Peak concurrent footprint, in the order :meth:`_routed_experts` allocates: the packed
        gate/up output alongside its two slices (``2 x E x tokens x 2I``), and later the down
        output alongside the scored activation (``E x tokens x (H + I)``). Whichever is larger has
        to fit :attr:`expert_l1_budget` or the whole chain stays in DRAM — a partial split would
        pay a DRAM round trip in the middle of the chain, which is the cost the move to L1 removed.
        """
        key = (tokens, self._call_tokens)
        cached = self._expert_mem_cache.get(key)
        if cached is None:
            e, h, i = self.cfg.num_experts, self.cfg.dim, self.cfg.moe_intermediate_size
            width = _DTYPE_BYTES.get(self.policy.expert_act_dtype, 2.0)
            peak = max(2 * e * tokens * 2 * i, e * tokens * (h + i)) * width
            fits = peak <= self.expert_l1_budget and self._call_tokens <= EXPERT_L1_MAX_CALL_TOKENS
            cached = self.expert_mem_config if fits else ttnn.DRAM_MEMORY_CONFIG
            self._expert_mem_cache[key] = cached
            logger.debug(
                f"MoE expert intermediates for a {tokens}-token group of a {self._call_tokens}-token "
                f"call: peak {peak / 2**20:.1f} MiB against a {self.expert_l1_budget / 2**20:.1f} MiB "
                f"L1 budget -> {cached.buffer_type}"
            )
        return cached

    def _fits_l1(self, nbytes: float):
        """``L1_MEMORY_CONFIG`` when ``nbytes`` fits the expert L1 budget, else DRAM.

        Both conditions, same as :meth:`_expert_mem`: the tensor's own size *and* the whole MoE
        call's token count. The second is not redundant — what changes with the call is how much L1
        the surrounding activations are already holding, which is why an 8-token-per-group budget
        that fits in isolation still refuses at a batch-32 prefill.
        """
        fits = nbytes <= self.expert_l1_budget and self._call_tokens <= EXPERT_L1_MAX_CALL_TOKENS
        return self.expert_mem_config if fits else ttnn.DRAM_MEMORY_CONFIG

    def _active_expert_bound(self, valid_tokens) -> int:
        """Upper bound on the experts one group can activate, from its real token count."""
        rows = self.group_tokens if valid_tokens is None else max(1, int(valid_tokens))
        return min(self.cfg.num_experts, rows * self.cfg.num_experts_per_tok)

    def _sparse_cfg(self, role: str, tokens: int, active_bound: int):
        """Cached sparse-matmul program config for ``role`` at this M and active-expert bound."""
        cores = _sparse_cores(role, active_bound)
        # The cap keys off the **realised** core count, not the target. `_sparse_matmul_config` reduces the
        # target to the largest divisor of `Nt`, so the two differ — and review round 7 found the target-keyed
        # version handing batch-3 decode the one combination round 6 had just removed: a bound of 24 gives a
        # target of 12, which is above SPARSE_MIN_CORES, while the grid it builds is still 8 cores. Reducing
        # here means the cap always describes the geometry that actually runs.
        realised_cores = _largest_divisor_at_most(_sparse_n_tiles(self.cfg, role), max(1, cores))
        gate_up_block_w = self.gate_up_in0_block_w[realised_cores > SPARSE_MIN_CORES]
        key = (role, tokens, cores)
        cfg = self._sparse_cfg_cache.get(key)
        if cfg is None:
            if role == "gate_up":
                cfg = _sparse_matmul_config(
                    tokens,
                    2 * self.cfg.moe_intermediate_size,
                    self.cfg.dim,
                    cores=cores,
                    in0_block_w=gate_up_block_w,
                    grid=self.grid,
                )
            else:
                cfg = _sparse_matmul_config(
                    tokens,
                    self.cfg.dim,
                    self.cfg.moe_intermediate_size,
                    cores=cores,
                    in0_block_w=self.down_in0_block_w,
                    grid=self.grid,
                )
            self._sparse_cfg_cache[key] = cfg
        return cfg

    def _router_zeros_for(self, logits):
        """The all-zero bfloat16 scatter target for :meth:`routing_weights`, persistent in decode.

        Decode replays a fixed shape inside a trace region, so the target is allocated once and reused: no
        `zeros_like`, no `typecast`, and nothing written from the host inside the trace. Prefill's shape varies
        per call, so it keeps the per-call spelling. `ttnn.scatter` is out-of-place, so reuse is safe - the tensor
        is read as the base and never mutated.
        """
        shape = tuple(int(d) for d in logits.shape)
        if not self._decode_phase:
            return ttnn.typecast(ttnn.zeros_like(logits), ttnn.bfloat16)
        # Keyed by shape and never freed. The first version replaced the buffer when the shape changed, which is
        # a trace hazard rather than a leak: the logits shape is `[1, 1, align_up(batch, 32), num_experts]`, so it
        # is stable for every batch up to 32 but changes at the supported 40 and 56, and a decode trace captured
        # at one of those shapes would replay a scatter into a freed buffer - silently wrong routing weights, not
        # an error. Review round 17 raised it before any test could. One tensor per distinct decode shape is at
        # most a few kilobytes and the decoder holds them for its lifetime.
        if shape not in self._router_zeros:
            self._router_zeros[shape] = ttnn.typecast(ttnn.zeros_like(logits), ttnn.bfloat16)
        return self._router_zeros[shape]

    # ---------------- router ----------------
    def _routing_pairs(self, x):
        """Preserved ``(FLOAT32 top-k weights, UINT32 global indices)`` for native MoE.

        This is intentionally separate from :meth:`routing_weights`: calling this method commits to
        the native representation and no scatter target or dense ``[T,E]`` tensor is allocated.
        """

        logits = ttnn.linear(
            x,
            self.w["router"],
            dtype=ttnn.float32,
            compute_kernel_config=self.dense_ckc,
            program_config=self.proj_cfgs.get(
                "router",
                _physical_rows(x.shape),
                x.shape[-1],
                self.cfg.num_experts,
                fp32_acc=self.policy.router_fp32_acc,
                decode=self._decode_phase,
            ),
        )
        values, raw_indices = ttnn.topk(logits, k=self.cfg.num_experts_per_tok, dim=-1, sorted=True)
        # The native planner consumes UINT32 tile faces. FP32 topk currently
        # produces that dtype, but make the API boundary explicit so a future
        # topk default change cannot route through a mis-sized index accessor.
        indices = ttnn.typecast(raw_indices, ttnn.uint32)
        if indices is not raw_indices:
            ttnn.deallocate(raw_indices)
        weights = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=self.dense_ckc)
        ttnn.deallocate(logits)
        ttnn.deallocate(values)
        return weights, indices

    def routing_weights(self, x):
        """Dense routing weights ``[1, 1, tokens, num_experts]`` (bfloat16, zeros off-selection).

        Unchanged from the functional decoder: float32 logits (expert *selection* is a discrete
        decision, so logit noise near the top-8/top-9 boundary swaps an expert rather than
        perturbing a value), ``topk`` on the raw logits, ``softmax`` over the kept 8, scatter into a
        dense vector.

        Two replacements were assessed and rejected, both recorded in ``doc/fused_decoder``:
        ``ttnn.experimental.deepseek.moe.generalized_moe_gate`` — one kernel for softmax + top-k +
        normalize, but bfloat16-only, rejected on that ground alone and **never timed**; the
        99.8 %/95.5 % top-8 set agreement it is rejected against is the functional stage's measured
        bfloat16-vs-float32 router A/B, applied to it because it is bfloat16-only — and a
        ``ge(kth) → where → softmax(256)`` threshold rewrite, which *was* timed and is slower at these
        shapes at identical accuracy (doc/fused_decoder/work_log.md §4.3, §4.12).
        """
        logits = ttnn.linear(
            x,
            self.w["router"],
            dtype=ttnn.float32,
            compute_kernel_config=self.dense_ckc,
            program_config=self.proj_cfgs.get(
                "router",
                _physical_rows(x.shape),
                x.shape[-1],
                self.cfg.num_experts,
                fp32_acc=self.policy.router_fp32_acc,
                decode=self._decode_phase,
            ),
        )
        values, indices = ttnn.topk(logits, k=self.cfg.num_experts_per_tok, dim=-1, sorted=True)
        weights = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=self.dense_ckc)
        # The scatter target. `ttnn.scatter` is out-of-place, decode shapes are fixed inside a trace, and this
        # tensor is all zeros every step - so it is allocated ONCE at build time and reused, which removes both
        # the `zeros_like` and the `typecast` from the step entirely. Review round 14 tried the obvious spelling
        # (`zeros_like(logits, dtype=...)`) and hit `TT_FATAL: Writes are not supported during trace capture`,
        # because with a dtype argument the op materialises via a host write; round 15 pointed out - correctly -
        # that a first API error is not a rejection, and that the persistent-tensor pattern this file already
        # uses for the RoPE tables, `batch_idxs` and `pos_ramp` is the adaptation that was never tried.
        #
        # Only the decode path can use it: prefill's token count varies per call, so its target is built per
        # call at the shape that call needs. `_router_zeros` is None until a decode shape is first seen.
        zeros = self._router_zeros_for(logits)
        dense = ttnn.scatter(zeros, dim=-1, index=indices, src=ttnn.typecast(weights, ttnn.bfloat16))
        ttnn.deallocate(logits)
        ttnn.deallocate(values)
        ttnn.deallocate(indices)
        ttnn.deallocate(weights)
        return dense

    def _active_expert_mask(self, dense_routing, groups, valid_tokens):
        """``[1, groups, 1, E]`` ROW_MAJOR sparsity: which experts this group's *real* tokens chose.

        The MoE runs on whole 32-row tiles, so a decode step for batch ``b < 32`` hands the routed
        experts a group whose trailing ``32 - b`` rows are the zero padding
        :meth:`OptimizedDecoder._block` added. Those rows are not free: their router logits are
        exactly zero (a zero activation row times the routing weight), so ``topk`` returns *some*
        deterministic set of ``num_experts_per_tok`` experts for each of them and ``softmax`` gives
        each an equal non-zero weight. Reducing the mask over all 32 rows therefore unions the real
        token's experts with the padding's, and ``ttnn.sparse_matmul`` computes both sets: at
        batch 1 that is 8 real experts plus up to 8 padding ones, i.e. close to **twice** the expert
        work the model actually asks for, in the largest single item of the decode window.

        Restricting the reduction to the ``valid_tokens`` real rows costs one slice on a
        ``[1, 1, 32, 256]`` tensor and removes the padding experts from the sparsity. The padded
        rows' *outputs* were already zero (their activation row is zero, so every expert block they
        produce is zero and the score multiply keeps it zero), so this changes which experts run,
        never the result. ``tests/test_optimized_decoder.py::test_padded_rows_do_not_route`` pins
        both halves: the mask holds exactly ``num_experts_per_tok`` experts at decode batch 1, and
        the layer output is unchanged.
        """
        E = self.cfg.num_experts
        tokens = int(dense_routing.shape[-2])
        rows = tokens if valid_tokens is None else min(int(valid_tokens), tokens)
        if rows < tokens and groups != 1:
            # Reachable only for a decode batch above 32, where the padded token count spans more
            # than one 32-row group and only the last group is partially valid. Restricting the
            # reduction there would need a per-group valid count; instead that case keeps the whole
            # -tile reduction, which is what the fused decoder always did and is *correct* — the
            # padding rows can only add experts, never drop a real token's, and an added expert
            # contributes exactly zero because its routing score for every real token is zero. It
            # loses the optimization, not the result, and only for batches past the primary
            # single-user decode target. `active_bound` below is unaffected: a multi-group call
            # already assumes a full group.
            rows = tokens
        source, owned = dense_routing, False
        if rows < tokens:
            source = ttnn.slice(dense_routing, [0, 0, 0, 0], [1, 1, rows, E])
            owned = True
        elif groups > 1:
            source = ttnn.reshape(dense_routing, [1, groups, tokens // groups, E])
        mask = ttnn.to_layout(ttnn.gtz(ttnn.sum(source, dim=-2, keepdim=True)), ttnn.ROW_MAJOR_LAYOUT)
        if owned:
            ttnn.deallocate(source)
        return mask

    # ---------------- experts ----------------
    def _routed_experts(self, x, dense_routing, tokens, *, group_mask=None, scores=None, valid_tokens=None):
        """Routed-expert output ``[1, 1, tokens, hidden]``.

        ``group_mask`` (``[1, groups, 1, E]``, ROW_MAJOR) and ``scores`` (``[1, E, tokens, 1]``) may
        be supplied by the caller. Both are per-call quantities that this method would otherwise
        rebuild from ``dense_routing`` on every expert group, which is the same redundancy the
        router hoist removed one level up; :meth:`forward` computes them once and passes each
        group its slice. Anything the caller supplies is borrowed, not owned, and is not freed here.

        ``valid_tokens`` is how many leading rows of the group are real; see
        :meth:`_active_expert_mask`.
        """
        E = self.cfg.num_experts
        H = self.cfg.dim
        I = self.cfg.moe_intermediate_size
        groups = tokens // TILE
        # Which sparse-matmul geometry this call gets. `valid_tokens` is the real row count of the
        # *group*, so for the multi-group prefill path each group is a full 32 rows.
        active_bound = self._active_expert_bound(valid_tokens if groups == 1 else TILE)
        expert_mem = self._expert_mem(tokens)

        mask_owned = group_mask is None
        if mask_owned:
            group_mask = self._active_expert_mask(dense_routing, groups, valid_tokens)  # [1, groups, 1, E]
        if groups == 1:
            # The per-group union over the single group *is* the whole-call union; only the shape
            # differs, and [1,1,1,E] is what the group mask already is.
            call_mask = ttnn.reshape(group_mask, [1, 1, 1, E])
            call_owned = False
        else:
            call_mask = self._active_expert_mask(dense_routing, 1, valid_tokens)  # [1, 1, 1, E]
            call_owned = True

        a = ttnn.reshape(x, [1, groups, TILE, H])
        # Shared-LHS packing of gate and up. The motivation is core count, not dispatch: the sparse
        # factory hands each core one N-tile block and cannot split M (a group is a single 32-row
        # tile), so an N=512 matmul is pinned to 512/32 = 16 cores while the packed N=1024 one uses
        # 32. Measured at the decode shapes in doc/fused_decoder/logs/probe_gate_up_pack.txt, which
        # also checks the packed form against the two-matmul one with `torch.equal`, not PCC alone.
        packed = ttnn.sparse_matmul(
            a,
            self.w["expert_gate_up"],
            sparsity=group_mask,
            nnz=None,
            memory_config=expert_mem,
            output_tile=self.output_tile,
            program_config=self._sparse_cfg("gate_up", TILE, active_bound),
            compute_kernel_config=self.expert_ckc,
            dtype=self.policy.expert_act_dtype,
        )
        gate = _slice_last(packed, 0, I)
        up = _slice_last(packed, I, 2 * I)
        ttnn.deallocate(packed)
        # SwiGLU folded into one binary op, still in the sparse matmul's native
        # [1, groups, 1, E, TILE, I] layout so only the product needs a relayout.
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=_SILU, memory_config=expert_mem)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        if groups == 1:
            # Swapping a pair of extent-1 axes is a pure relabel: drop straight to expert-major.
            hidden = ttnn.reshape(hidden, [1, E, tokens, I])
        else:
            swapped = ttnn.transpose(hidden, 1, 3)
            ttnn.deallocate(hidden)
            hidden = ttnn.reshape(swapped, [1, E, tokens, I])
            del swapped  # the reshape is a view; keeping the handle would outlive the buffer's free

        # Score the *input* of the down projection instead of its output: identical by linearity,
        # and moe_intermediate (512) wide instead of hidden (2048) wide.
        scores_owned = scores is None
        if scores_owned:
            scores = ttnn.permute(dense_routing, (0, 3, 2, 1))  # [1, E, tokens, 1]
        scaled = ttnn.multiply(hidden, scores, memory_config=expert_mem)
        ttnn.deallocate(hidden)
        if scores_owned:
            ttnn.deallocate(scores)

        down = ttnn.sparse_matmul(
            scaled,
            self.w["expert_down"],
            sparsity=call_mask,
            nnz=None,
            memory_config=expert_mem,
            output_tile=self.output_tile,
            program_config=self._sparse_cfg("down", tokens, active_bound),
            is_input_a_sparse=True,
            compute_kernel_config=self.expert_ckc,
            dtype=self.policy.expert_act_dtype,
        )  # [1, E, tokens, H]
        ttnn.deallocate(scaled)
        if call_owned:
            ttnn.deallocate(call_mask)
        if mask_owned:
            ttnn.deallocate(group_mask)

        # deepseek_moe_fast_reduce_nc over ttnn.experimental.fast_reduce_nc: same latency at these
        # shapes but a materially more accurate accumulation (PCC 0.999999 vs 0.999409 against a
        # float32 sum of 256 bfloat16 expert blocks — doc/fused_decoder/logs/probe_router_and_reduce.txt).
        #
        # The op accepts only BFLOAT16 and BFLOAT8_B ("DeepseekMoEFastReduceNC input only supports
        # specific data types", `moreh_helper_functions.cpp:285`), so a bfloat4_b `expert_act_dtype`
        # has to be widened first. That is the *adapted* form of the datatype-sweep's C19 candidate:
        # a first API error is not a rejection. Under every shipped policy `expert_act_dtype` is
        # already one of the two accepted dtypes and this dispatches nothing.
        #
        # It is kept, and it is correct — `doc/datatype_sweep/AUTOFIX_C19.md` measured this exact
        # typecast at the decode shape and it is trace-capturable in both L1 and DRAM — but it does
        # NOT make C19 measurable. The blocker is one op *upstream*: `ttnn.sparse_matmul` cannot
        # produce a BFLOAT4_B output inside a trace at all, because its device operation zero-fills
        # the output with `ttnn::zeros_like` (the kernel never writes inactive experts' blocks) and
        # `full_like_impl` has no device-fill fast path for BFLOAT4_B, so the fill is a host write.
        # C19 stays blocked on that, not on this.
        if down.dtype not in _MOE_REDUCE_DTYPES:
            widened = ttnn.typecast(down, MOE_REDUCE_FALLBACK_DTYPE, memory_config=down.memory_config())
            ttnn.deallocate(down)
            down = widened
        reduced = ttnn.experimental.deepseek_moe_fast_reduce_nc(down, dim=1, split_size=H)[0]
        ttnn.deallocate(down)
        return ttnn.reshape(ttnn.unsqueeze_to_4D(reduced), [1, 1, tokens, H])

    # ---------------- router-index-native experts ----------------
    def prepare_topk_native(self, prefill_chunk: int) -> bool:
        """Finalize setup-time native readiness for this concrete layer.

        The base single-chip geometry owns all 256 experts and is intentionally refused by the
        validated 64-local-expert bound. :class:`MultichipMoE` installs the mesh-sharded
        global-to-local table first, then delegates the remaining checks here.
        """

        if not self.topk_native_selected:
            self.topk_native_ready = False
            self.topk_native_refusal = f"{MOE_TOPK_NATIVE_ENV_VAR} is not set"
            return False
        reasons = []
        if self.cfg.num_experts > MOE_GATHER_MAX_LOCAL_EXPERTS:
            reasons.append(
                f"{self.cfg.num_experts} local experts exceeds the validated native bound "
                f"{MOE_GATHER_MAX_LOCAL_EXPERTS}"
            )
        if self.cfg.num_experts_per_tok > 16:
            reasons.append(f"top-k {self.cfg.num_experts_per_tok} exceeds the native kernel bound 16")
        if self.cfg.dim % 1024:
            reasons.append(f"hidden size {self.cfg.dim} is not divisible by the fused combine block 1024")
        if self.policy.expert_act_dtype != ttnn.bfloat8_b:
            reasons.append(
                f"expert output policy {self.policy.expert_act_dtype} is not BFLOAT8_B at the collective boundary"
            )
        if int(prefill_chunk) < MOE_TOPK_NATIVE_SUB_CHUNK:
            reasons.append(f"prefill chunk {prefill_chunk} is below native sub-chunk {MOE_TOPK_NATIVE_SUB_CHUNK}")
        if not self.topk_native_weights_loaded:
            reasons.append("the per-expert gate/up/down weight lists were not uploaded")
        if self.topk_native_global_to_local is None:
            reasons.append("the mesh-sharded global-to-local expert table was not prepared")
        if reasons:
            self.topk_native_ready = False
            self.topk_native_refusal = "; ".join(reasons)
            logger.warning(f"top-k-native routed experts unavailable: {self.topk_native_refusal}")
            return False
        self.topk_native_refusal = None
        self.topk_native_ready = True
        return True

    def _topk_native_geometry_reason(self, tokens: int, decode: bool, valid_tokens) -> str | None:
        """Shape-only eligibility; readiness failures on eligible calls count as fallbacks."""

        tokens = int(tokens)
        if decode:
            return "decode intentionally keeps sparse MoE"
        if valid_tokens is not None:
            return "tile-padded calls intentionally keep the accepted fallback"
        if tokens < MOE_TOPK_NATIVE_SUB_CHUNK:
            return f"{tokens} tokens is below {MOE_TOPK_NATIVE_SUB_CHUNK}"
        if tokens % MOE_TOPK_NATIVE_SUB_CHUNK:
            return f"{tokens} tokens has a ragged native sub-chunk"
        if self.cfg.num_experts > MOE_GATHER_MAX_LOCAL_EXPERTS:
            return f"{self.cfg.num_experts} local experts exceeds {MOE_GATHER_MAX_LOCAL_EXPERTS}"
        if self.cfg.num_experts_per_tok > 16:
            return f"top-k {self.cfg.num_experts_per_tok} exceeds 16"
        if self.cfg.dim % 1024:
            return f"hidden size {self.cfg.dim} is not divisible by 1024"
        if self.policy.expert_act_dtype != ttnn.bfloat8_b:
            return "the selected precision policy does not emit BFLOAT8_B expert partials"
        return None

    def _topk_native_routed_experts(self, x, topk_weights, topk_indices, tokens: int):
        """Run the exact public native composite over <=1024-token slices in token order."""

        parts = []
        for start, end in _topk_native_chunk_ranges(tokens):
            whole = start == 0 and end == tokens
            chunk = x if whole else ttnn.slice(x, [0, 0, start, 0], [1, 1, end, self.cfg.dim])
            indices = (
                topk_indices
                if whole
                else ttnn.slice(topk_indices, [0, 0, start, 0], [1, 1, end, self.cfg.num_experts_per_tok])
            )
            weights = (
                topk_weights
                if whole
                else ttnn.slice(topk_weights, [0, 0, start, 0], [1, 1, end, self.cfg.num_experts_per_tok])
            )
            span = end - start
            # `calls` is actual composite invocation count, not an env-selected intent counter.
            self.topk_native_calls += 1
            part = ttnn.experimental.deepseek_prefill.topk_routed_expert_moe(
                chunk,
                indices,
                weights,
                self.topk_native_global_to_local,
                self.w["expert_gate_list"],
                self.w["expert_up_list"],
                self.w["expert_down_list"],
                num_local_experts=self.cfg.num_experts,
                valid_tokens=span,
                max_dispatched_tokens_per_expert=span,
                activation=ttnn.RoutedExpertActivation.Silu,
            )
            self.topk_native_subchunks += 1
            parts.append(part)
            if chunk is not x:
                ttnn.deallocate(chunk)
            if indices is not topk_indices:
                ttnn.deallocate(indices)
            if weights is not topk_weights:
                ttnn.deallocate(weights)

        routed = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
        if len(parts) > 1:
            for part in parts:
                ttnn.deallocate(part)
        self.topk_native_layer_calls += 1
        return routed

    # ---------------- gathered experts ----------------
    def prepare_gather_experts(self, prefill_chunk: int) -> bool:
        """Build the gathered path's host-side constants. Setup, not forward.

        Returns whether the path is available. Called from
        :meth:`OptimizedDecoder.allocate_state`, which is the module's one documented not-host-free
        entry point, because :meth:`_gather_consts` needs ``ttnn.from_torch`` and the module docstring
        promises the forward paths make no host call on an allocated layer.

        A call whose token count was never prepared keeps the sparse path — slower and correct, the
        same rule ``MultichipMoE.prepare_decode_gate`` states for an unprepared decode row count.
        """
        if not (MOE_GATHER_EXPERTS or MOE_TOPK_NATIVE):
            self.gather_refusal = f"neither {MOE_GATHER_ENV_VAR} nor {MOE_TOPK_NATIVE_ENV_VAR} is set"
            return False
        largest = min(MOE_GATHER_SUB_CHUNK, int(prefill_chunk))
        reason = _gather_support_reason(self.cfg.num_experts, largest)
        if reason is None and not self.gather_experts:
            reason = "the per-expert weight layout was not uploaded (the switch was clear at load time)"
        if reason is not None:
            # Loud, once, at setup — never a silent fallback. The sparse path is correct and slower, so
            # a run that meant to measure the gathered path and quietly did not must be visible in the
            # log rather than only in the numbers.
            logger.warning(f"gathered routed experts unavailable: {reason}; keeping ttnn.sparse_matmul")
            self.gather_refusal = reason
            return False
        self.gather_refusal = None
        # Every admitted gathered sub-chunk length, not just the full one. A chunk's token count is a
        # multiple of :data:`PREFILL_ALIGN`, so :meth:`forward`'s ragged last sub-chunk is too. A tail
        # below :data:`MOE_GATHER_MIN_TOKENS` stays on sparse MoE, because one-tile gathered regions
        # lose the arithmetic advantage. The largest compact scatter base is about 40 KiB.
        for rows in range(MOE_GATHER_MIN_TOKENS, largest + 1, PREFILL_ALIGN):
            self._gather_consts(rows)
        self._gather_ready = True
        return True

    def _gather_consts(self, rows: int):
        """Host-built constants the gathered path needs at a ``rows``-token sub-chunk.

        Four tensors, none of them data-dependent, so all four are built once per distinct sub-chunk
        size and then never touched by the host again:

        * ``ramp`` — ``arange(rows)`` as ``[1, 1, 1, rows]`` float32. Transposed to
          ``[1, 1, rows, 1]``, it supplies token ids to the compact scatter.
        * ``eramp`` — ``arange(num_experts)`` as ``[1, 1, 1, E]`` float32, the same trick on the other
          axis for the per-token expert list.
        * ``scatter_zeros`` — persistent UINT32 ROW_MAJOR zeros of length ``capacity + 1``. The first
          ``capacity`` entries are the compact dispatch table and the last entry is a dump slot for
          inactive ``(token, expert)`` pairs. Region starts are data-dependent and computed on device;
          the capacity itself is the exact static bound from :func:`_gather_compact_capacity`.
        * ``idx_table`` — the identity ``arange(E)``. ``unified_routed_expert_ffn`` reads
          ``counts[idx_table[local_expert_id]]``, i.e. it indirects a *local* expert id into a global
          count vector. This model's routing is already narrowed to the device's own experts before it
          gets here, so the local and global index spaces coincide and the table is the identity; the
          op asserts only ``local_expert_id < len(idx_table)``.
        """
        cached = self._gather_const.get(rows)
        if cached is None:
            # Inside the miss branch, not at the top: a cache hit must reach no `torch` call at all,
            # which is what `test_no_host_fallback_in_forward`'s TorchFunctionMode guard checks when the
            # suite is run with the switch set.
            import torch

            if rows % PREFILL_ALIGN or rows < MOE_GATHER_MIN_TOKENS:
                # Only measured, profitable gathered shapes are prepared. A smaller or unaligned
                # caller is a bug in sub-chunking, not a shape to allocate during forward.
                raise ValueError(
                    f"gathered-expert sub-chunk {rows} must be a multiple of PREFILL_ALIGN "
                    f"({PREFILL_ALIGN}) and at least MOE_GATHER_MIN_TOKENS ({MOE_GATHER_MIN_TOKENS})"
                )
            E = self.cfg.num_experts
            capacity = _gather_compact_capacity(rows, E, self.cfg.num_experts_per_tok)

            def upload(host, dtype, layout):
                return ttnn.from_torch(
                    host,
                    dtype=dtype,
                    layout=layout,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(self.device),
                )

            cached = (
                upload(torch.arange(rows, dtype=torch.float32).reshape(1, 1, 1, rows), ttnn.float32, ttnn.TILE_LAYOUT),
                upload(torch.arange(E, dtype=torch.float32).reshape(1, 1, 1, E), ttnn.float32, ttnn.TILE_LAYOUT),
                upload(
                    torch.zeros(1, 1, 1, capacity + 1, dtype=torch.int32),
                    ttnn.uint32,
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
                upload(torch.arange(E, dtype=torch.int32).reshape(1, E), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            )
            self._gather_const[rows] = cached
        return cached

    def _gather_routed_experts(self, x, dense_routing, tokens):
        """Routed-expert output ``[1, 1, tokens, hidden]``, gathered per expert.

        Same quantity as :meth:`_routed_experts` — the router-weighted sum over each token's selected
        experts, restricted to the experts this device owns — computed the other way round. Instead of
        evaluating every activated expert against every token of a 32-row group and letting the score
        zero the ones it did not select, this materialises each expert's *own* rows and evaluates only
        those.

        The permutation, built entirely on device:

        1. ``mask[t, e]`` is 1 where token ``t`` selected local expert ``e`` (the router already wrote
           exact zeros everywhere else, so ``gtz`` is the selection).
        2. ``counts[e] = Σ_t mask[t, e]`` is the expert's real token count — the number the fused FFN
           reads on device to size its work.
        3. ``rank[t, e]``, the exclusive prefix sum of ``mask`` down the token axis, is *where in
           expert e's region token t goes*: the number of earlier tokens that also chose ``e``.
        4. ``aligned[e] = 32 * ceil(counts[e] / 32)`` and an exclusive cumsum over ``aligned`` give
           expert ``e``'s tight-packed, tile-aligned ``region_offset[e]``. Their sum is bounded exactly
           by ``tokens * top_k + 31 * num_experts``; the persistent buffer is sized to that bound.
        5. A UINT32 ROW_MAJOR ``ttnn.scatter`` writes token id ``t`` at
           ``region_offset[e] + rank[t, e]``. Inactive pairs write zero to a one-element dump slot, so
           all real destinations are unique and no reduction is needed. ``ttnn.embedding`` then
           gathers exactly the compact buffer's token rows from ``x``.

        The reverse is the same identity read backwards. Sorting ``mask[t, e] ? e : BIG`` along the
        *expert* axis gives each token its selected local experts in increasing order — at most
        ``num_experts_per_tok`` of them, because a token has that many experts in total — and a second
        ``embedding`` gathers row ``region_offset[e] + rank[t, e]`` of the FFN output back to slot
        ``j`` of token ``t``. The router score multiplies there. The reverse slot is the same
        ``region_offset[e] + rank[t, e]`` built for dispatch; no second token-axis sort is needed.

        Contract notes, each pinned by a test in ``tests/test_multichip_decoder.py``:

        * **Only the selected experts contribute.** A token has at most ``num_experts_per_tok`` valid
          slots; the trailing ones hold the sort's ``BIG`` sentinel. Their gathered FFN rows are
          selected to an exact zero with ``where`` *before arithmetic*, and their score is separately
          multiplied by the ``valid`` indicator. This matters because the fused FFN deliberately
          leaves undispatched output rows uninitialized: ``0 * NaN`` would not be zero.
        * **Score placement.** The score multiplies the FFN's *output* row rather than the down
          projection's input. Identical by linearity — the same argument :meth:`_routed_experts` uses
          to move the multiply the other way — and here it is also 8 rows per token instead of 64.
        * **Region padding cannot reach the output.** The FFN evaluates ``ceil(counts[e] / 32)`` tiles,
          so rows in ``[counts[e], 32·ceil(counts[e]/32))`` are computed from the scatter base's exact
          zeros; rows past that are never written at all. Neither is ever read back, because every
          valid reverse slot has ``rank[t, e] < counts[e]``.
        * **No collective.** Every op here is per-device. The result is this device's partial sum over
          its own experts, exactly what :meth:`_routed_experts` returns, so ``MultichipDecoder._block``
          closes it with the same single all-reduce and nothing about the expert-parallel invariant
          moves. A device that owns none of the call's selected experts gets an all-zero ``mask``: zero
          counts (the FFN launches and evaluates nothing), an all-sentinel expert list, every score
          zeroed, every uninitialized gathered row selected away, and an exactly-zero result. It needs
          no floored expert — there is no sparsity argument to keep non-empty.
        """
        E = self.cfg.num_experts
        H = self.cfg.dim
        K = self.cfg.num_experts_per_tok
        ramp, eramp, scatter_zeros, idx_table = self._gather_consts(tokens)
        capacity = _gather_compact_capacity(tokens, E, K)
        # Where the two per-slot tensors live. They are the only ones small enough to consider for L1:
        # the compact gathered buffer is still much larger and belongs in DRAM at every shape. The
        # factor of two is because the gathered slots and the scored copy are live at the same time,
        # which is the same peak rule :meth:`_expert_mem` applies to the sparse chain. Both are
        # bfloat16: the slots come out of `ttnn.embedding`, whose table dtype the op fixes at bfloat16.
        expert_mem = self._fits_l1(2.0 * K * tokens * H * 2.0)
        # Above every token index and every expert index, and exactly representable in float32 (which
        # also represents every index below 2**24 exactly, so `rank` and `sorted` are integers, not
        # roundings). `ttnn.sort` takes float32 natively, which is why the keys never go through a
        # narrow integer dtype: bfloat16 stops being integral at 256, well inside a 2048-token chunk.
        big = float(1 << 20)

        mask = ttnn.typecast(ttnn.gtz(dense_routing), ttnn.float32)  # [1, 1, tokens, E]
        # `counts` is what the FFN reads on device to bound each expert's work. float32 sums exactly at
        # these magnitudes; the op wants UINT32 ROW_MAJOR DRAM, so untilize then narrow.
        counts_f = ttnn.sum(mask, dim=-2, keepdim=True)  # [1, 1, 1, E]
        counts_rm = ttnn.to_layout(counts_f, ttnn.ROW_MAJOR_LAYOUT)
        counts = ttnn.typecast(
            ttnn.reshape(counts_rm, [1, E]),
            ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(counts_rm)

        # Tile-align each expert's real count, then exclusive-cumsum the aligned sizes. All values are
        # exact in float32 at the admitted shapes: counts <= 1024, 1/32 is an exact binary fraction,
        # and the final capacity is 10,176 rows. The UINT32 copy is what the fused FFN's reader/writer
        # consumes; the tiled float copy broadcasts into `packed_slot` below without a round trip.
        scaled_counts = ttnn.multiply(counts_f, 1.0 / TILE)
        count_tiles = ttnn.ceil(scaled_counts)
        ttnn.deallocate(scaled_counts)
        aligned_counts = ttnn.multiply(count_tiles, float(TILE))  # [1, 1, 1, E]
        ttnn.deallocate(count_tiles)
        ttnn.deallocate(counts_f)
        region_ends = ttnn.cumsum(aligned_counts, -1, dtype=ttnn.float32)
        region_offsets_f = ttnn.subtract(region_ends, aligned_counts)  # exclusive cumsum
        ttnn.deallocate(region_ends)
        ttnn.deallocate(aligned_counts)
        region_offsets_rm = ttnn.to_layout(region_offsets_f, ttnn.ROW_MAJOR_LAYOUT)
        region_offsets = ttnn.typecast(
            ttnn.reshape(region_offsets_rm, [1, E]),
            ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(region_offsets_rm)

        # Exclusive prefix sum down the token axis: rank[t, e] = #{t' < t : mask[t', e]}.
        inclusive = ttnn.cumsum(mask, -2, dtype=ttnn.float32)
        rank = ttnn.subtract(inclusive, mask)  # [1, 1, tokens, E]
        ttnn.deallocate(inclusive)
        packed_slot = ttnn.add(rank, region_offsets_f)  # [1, 1, tokens, E]
        ttnn.deallocate(rank)
        ttnn.deallocate(region_offsets_f)

        # ---- forward permutation: each expert's tokens, in order ----
        # Scatter token id `t` directly into its unique compact slot. Inactive pairs all target the
        # extra dump element at `capacity` and all write zero, so their write/write collision is benign;
        # selected pairs have unique (expert, rank) and never collide. ROW_MAJOR is the scatter op's
        # native layout and, importantly, avoids its >256-element restriction for *tiled* int32.
        slot_from_dump = ttnn.subtract(packed_slot, float(capacity))
        selected_dst = ttnn.multiply(mask, slot_from_dump)
        ttnn.deallocate(slot_from_dump)
        dispatch_dst_f = ttnn.add(selected_dst, float(capacity))
        ttnn.deallocate(selected_dst)
        token_ramp = ttnn.transpose(ramp, -2, -1)  # [1, 1, tokens, 1]
        dispatch_src_f = ttnn.multiply(mask, token_ramp)  # inactive pairs write zero to the dump slot
        ttnn.deallocate(token_ramp)

        dispatch_dst_rm = ttnn.to_layout(dispatch_dst_f, ttnn.ROW_MAJOR_LAYOUT)
        dispatch_dst = ttnn.typecast(
            ttnn.reshape(dispatch_dst_rm, [1, 1, 1, E * tokens]),
            ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(dispatch_dst_rm)
        ttnn.deallocate(dispatch_dst_f)
        dispatch_src_rm = ttnn.to_layout(dispatch_src_f, ttnn.ROW_MAJOR_LAYOUT)
        dispatch_src = ttnn.typecast(
            ttnn.reshape(dispatch_src_rm, [1, 1, 1, E * tokens]),
            ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(dispatch_src_rm)
        ttnn.deallocate(dispatch_src_f)

        dispatched_with_dump = ttnn.scatter(
            scatter_zeros,
            dim=-1,
            index=dispatch_dst,
            src=dispatch_src,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, 1, capacity + 1] ROW_MAJOR uint32
        ttnn.deallocate(dispatch_dst)
        ttnn.deallocate(dispatch_src)
        gather_idx = ttnn.slice(
            dispatched_with_dump,
            [0, 0, 0, 0],
            [1, 1, 1, capacity],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _free_unless_aliased(dispatched_with_dump, gather_idx)

        # `ttnn.embedding`'s table must be ROW_MAJOR bfloat16 with two leading unit dims; the MoE input
        # is TILE bfloat16 (`NORM_OUTPUT_DTYPE`), so this is an untilize and not a cast.
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_buf = ttnn.embedding(
            gather_idx, x_rm, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1, 1, capacity, H] ROW_MAJOR bf16
        ttnn.deallocate(gather_idx)
        ttnn.deallocate(x_rm)

        # ---- the experts ----
        # One device program per local expert, each fusing gate+up+silu+down and reading counts[e] to
        # skip the chunks past its real token count. `max_dispatched_tokens_per_expert` is the region
        # size, i.e. the worst case; it sizes the grid, not the work.
        y_buf = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            x_buf,
            region_offsets,
            counts,
            idx_table,
            self.w["expert_gate_list"],
            self.w["expert_up_list"],
            self.w["expert_down_list"],
            max_dispatched_tokens_per_expert=tokens,
            compute_kernel_config=self.expert_ckc,
            activation=ttnn.RoutedExpertActivation.Silu,
        )  # [1, 1, capacity, H], TILE block-float
        ttnn.deallocate(x_buf)
        ttnn.deallocate(counts)
        ttnn.deallocate(region_offsets)

        # ---- reverse permutation: each token's slots ----
        # keys[t, e] = mask ? e : big, sorted along the expert axis -> this token's local experts.
        shifted_e = ttnn.subtract(eramp, big)  # [1, 1, 1, E]
        keys_t = ttnn.add(ttnn.multiply(mask, shifted_e), big)
        ttnn.deallocate(shifted_e)
        expert_of_slot, sort_aux = ttnn.sort(keys_t, dim=-1)  # [1, 1, tokens, E], ascending
        ttnn.deallocate(sort_aux)
        ttnn.deallocate(keys_t)
        # A slot is real iff its sorted key is a genuine expert index. `ltz(x - E)` is `x < E`.
        # Narrowed to the score dtype here so the score multiply below has one dtype on both sides.
        valid = ttnn.typecast(ttnn.ltz(ttnn.subtract(expert_of_slot, float(E))), dense_routing.dtype)
        expert_sel = ttnn.clamp(expert_of_slot, 0.0, float(E - 1))
        ttnn.deallocate(expert_of_slot)
        # `ttnn.gather` wants index and input at the same rank and layout, and every dim but the
        # gathered one no larger than the input's, so both stay [1, 1, tokens, E] and the top-k prefix
        # is sliced off the *results*.
        expert_idx = ttnn.typecast(expert_sel, ttnn.uint32)
        gathered_slot = ttnn.gather(packed_slot, -1, expert_idx)
        # An invalid sentinel clamps its expert id to E-1. If all of that expert's real occurrences
        # precede this token, its exclusive rank can equal the expert count and point one row past its
        # region (equal to `capacity` in the tight adversarial case). Embedding requires an in-range
        # index, so clamp those dead slots before the top-k slice; their possibly-uninitialized values
        # are selected away with `where` below rather than relying on unsafe `0 * NaN` arithmetic.
        slot_sel = ttnn.clamp(gathered_slot, 0.0, float(capacity - 1))
        ttnn.deallocate(gathered_slot)
        score_sel = ttnn.gather(dense_routing, -1, expert_idx)
        ttnn.deallocate(expert_idx)
        ttnn.deallocate(packed_slot)
        ttnn.deallocate(mask)

        # `slot_sel[t, j]` is the exact compact row used by the forward scatter for this
        # (token, expert) pair. Sentinel expert slots gather an arbitrary in-range expert's slot and are
        # selected to zero by `valid` below, so no read can escape the compact buffer or propagate an
        # uninitialized value from the fused FFN's output.
        ttnn.deallocate(expert_sel)
        slot_k = ttnn.slice(slot_sel, [0, 0, 0, 0], [1, 1, tokens, K])
        ttnn.deallocate(slot_sel)
        score_top = ttnn.slice(score_sel, [0, 0, 0, 0], [1, 1, tokens, K])
        valid_top = ttnn.slice(valid, [0, 0, 0, 0], [1, 1, tokens, K])
        ttnn.deallocate(score_sel)
        ttnn.deallocate(valid)
        # The score of a sentinel slot is multiplied by exactly zero here. This keeps arbitrary
        # non-selected scores out of the sum; a separate `where` below protects the value side from
        # uninitialized non-finite rows before any arithmetic touches them.
        score_k = ttnn.multiply(score_top, valid_top)
        valid_rows = ttnn.permute(valid_top, (0, 3, 2, 1))  # [1, K, tokens, 1]
        ttnn.deallocate(score_top)
        ttnn.deallocate(valid_top)

        # Slot-major so the gathered rows reshape into [1, K, tokens, H] as a view: the reduction below
        # is over the slot axis, and in TILE layout only the last two axes are the tile axes, so a
        # token-major [1, 1, tokens, K] would put K on a tile axis and pad 8 rows out to 32.
        slot_kt = ttnn.transpose(slot_k, -2, -1)  # [1, 1, K, tokens]
        ttnn.deallocate(slot_k)
        slot_flat = ttnn.typecast(
            ttnn.reshape(ttnn.to_layout(slot_kt, ttnn.ROW_MAJOR_LAYOUT), [1, 1, 1, K * tokens]),
            ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(slot_kt)
        # The fused FFN emits a TILE block-float buffer; `ttnn.embedding`'s table must be ROW_MAJOR
        # bfloat16, and a block-float tensor has no row-major form, so the widen and the untilize are
        # two separate ops rather than one `to_layout(..., dtype=...)`.
        y_wide = ttnn.typecast(y_buf, ttnn.bfloat16)
        ttnn.deallocate(y_buf)
        y_rm = ttnn.to_layout(y_wide, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(y_wide)
        y_sel = ttnn.embedding(slot_flat, y_rm, layout=ttnn.TILE_LAYOUT, memory_config=expert_mem)
        ttnn.deallocate(slot_flat)
        ttnn.deallocate(y_rm)
        y_sel = ttnn.reshape(ttnn.unsqueeze_to_4D(y_sel), [1, K, tokens, H])

        # The shared TILE output allocated by `unified_routed_expert_moe` is intentionally uninitialized
        # outside the expert tiles it writes. Invalid reverse slots can legally point at those rows.
        # Select them away before scoring: multiplying by the zero score is insufficient when the read
        # happens to be NaN or Inf (`0 * NaN` and `0 * Inf` are NaN).
        selected_y = ttnn.where(valid_rows, y_sel, 0.0, memory_config=expert_mem)
        ttnn.deallocate(valid_rows)
        ttnn.deallocate(y_sel)

        # Score placement, and the reduction over the slot axis. `deepseek_moe_fast_reduce_nc` is the
        # same op `_routed_experts` reduces its expert axis with, and takes the same dtypes.
        scores = ttnn.permute(score_k, (0, 3, 2, 1))  # [1, K, tokens, 1]
        ttnn.deallocate(score_k)
        scaled = ttnn.multiply(selected_y, scores, memory_config=expert_mem)
        ttnn.deallocate(selected_y)
        ttnn.deallocate(scores)
        if scaled.dtype not in _MOE_REDUCE_DTYPES:
            widened = ttnn.typecast(scaled, MOE_REDUCE_FALLBACK_DTYPE, memory_config=scaled.memory_config())
            ttnn.deallocate(scaled)
            scaled = widened
        reduced = ttnn.experimental.deepseek_moe_fast_reduce_nc(scaled, dim=1, split_size=H)[0]
        ttnn.deallocate(scaled)
        return ttnn.reshape(ttnn.unsqueeze_to_4D(reduced), [1, 1, tokens, H])

    def _gather_ok(self, tokens: int, decode: bool, valid_tokens) -> bool:
        """Whether this call takes the gathered path. See :meth:`_gather_reason` for why it does not."""
        return self._gather_reason(tokens, decode, valid_tokens) is None

    def _gather_reason(self, tokens: int, decode: bool, valid_tokens) -> str | None:
        """Why this call keeps ``sparse_matmul``, or ``None`` if it takes the gathered path.

        A reason rather than a bool so the refusal is never silent: :meth:`prepare_gather_experts` logs
        the setup-time ones and a test can assert on the exact clause that fired, which is what stops a
        "the gathered arm agreed with the sparse arm" result from passing because both arms were sparse.
        """
        if not (MOE_GATHER_EXPERTS or MOE_TOPK_NATIVE):
            return f"neither {MOE_GATHER_ENV_VAR} nor {MOE_TOPK_NATIVE_ENV_VAR} is set"
        if not self.gather_experts:
            return "the per-expert weight layout was not uploaded (the switch was clear at load time)"
        if self.gather_refusal is not None:
            return self.gather_refusal
        if decode:
            return (
                "decode keeps sparse_matmul: a 32-row call gives every expert a one-tile region, so the "
                "gathered path would evaluate num_experts_local x 32 rows for the 32 x top_k / tp the "
                "routing asks and add the dispatch/reverse-permutation graph; the FFN launch is fused, "
                "but the arithmetic and glue remain worse than sparse decode"
            )
        if tokens < MOE_GATHER_MIN_TOKENS:
            return f"{tokens} tokens is below MOE_GATHER_MIN_TOKENS ({MOE_GATHER_MIN_TOKENS})"
        if valid_tokens is not None:
            return "a tile-padded call (valid_tokens set) is refused rather than handled"
        if not self._gather_ready:
            return "prepare_gather_experts has not run for this layer"
        if tokens % PREFILL_ALIGN:
            return f"{tokens} tokens is not a multiple of PREFILL_ALIGN ({PREFILL_ALIGN})"
        return _gather_support_reason(self.cfg.num_experts, min(MOE_GATHER_SUB_CHUNK, tokens))

    # ---------------- shared expert ----------------
    def _shared_expert(self, x):
        """``sigmoid(router(x)) * down(silu(gate(x)) * up(x))`` in one packed matmul + two ops."""
        inter = self.cfg.shared_expert_intermediate_size
        rows = _physical_rows(x.shape)
        # `tt-perf-report` flags the shared expert's *down* projection with "place input 0 in L1", and
        # its in0 is the SwiGLU product below. Both it and the packed matmul it comes from are
        # size-gated rather than assumed: at batch-1 decode the product is 32 KB and at the shipped
        # 2048-token prefill chunk 2 MB, but a batch-32 prefill would ask for 64 MB.
        wide_mem = self._fits_l1(rows * (2 * inter + TILE) * 2)
        act_mem = self._fits_l1(rows * inter * 2)
        fused = ttnn.linear(
            x,
            self.w["shared_in"],
            memory_config=wide_mem,
            compute_kernel_config=self.shared_ckc,
            program_config=self.proj_cfgs.get(
                "shared_in",
                _physical_rows(x.shape),
                x.shape[-1],
                self.w["shared_in"].shape[-1],
                fp32_acc=self.policy.shared_fp32_acc,
                decode=self._decode_phase,
            ),
        )
        gate = _slice_last(fused, 0, inter)
        up = _slice_last(fused, inter, 2 * inter)
        router = _slice_last(fused, 2 * inter, 2 * inter + 1)
        ttnn.deallocate(fused)
        act = ttnn.multiply(gate, up, input_tensor_a_activations=_SILU, memory_config=act_mem)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(
            act,
            self.w["shared_down"],
            memory_config=self._fits_l1(rows * self.cfg.dim * 2),
            compute_kernel_config=self.shared_ckc,
            program_config=self.proj_cfgs.get(
                "shared_down",
                _physical_rows(act.shape),
                act.shape[-1],
                self.cfg.dim,
                fp32_acc=self.policy.shared_fp32_acc,
                decode=self._decode_phase,
            ),
        )
        ttnn.deallocate(act)
        gated = ttnn.multiply(out, router, input_tensor_b_activations=_SIGMOID)
        ttnn.deallocate(out)
        ttnn.deallocate(router)
        return gated

    # ---------------- public ----------------
    def forward(self, x, *, valid_tokens=None, decode: bool = False):
        """``x``: ``[1, 1, tokens, hidden]`` with ``tokens % 32 == 0``. Returns the same shape.

        ``valid_tokens`` names how many leading rows are real when the caller tile-padded ``x``;
        rows past it are exactly zero and must not contribute experts to the routing sparsity
        (:meth:`_active_expert_mask`). ``None`` means every row is real.
        """
        tokens = x.shape[2]
        if tokens % TILE:
            raise ValueError(f"MoE token count must be a multiple of {TILE}; got {tokens}")
        if valid_tokens is not None and not 1 <= int(valid_tokens) <= tokens:
            raise ValueError(f"valid_tokens {valid_tokens} outside [1, {tokens}]")
        self._decode_phase = bool(decode)
        self._call_tokens = tokens

        shared = self._shared_expert(x)

        # Routing runs **once** for the whole call, not once per expert group: the router's topk and
        # its scatter chain barely scale with the token count (they are single-core on a 256-wide
        # last dim), so 64 group-sized routers cost 64x what one whole-call router does. Slicing the
        # dense score vector per group is tile-aligned and nearly free.
        native_geometry_reason = self._topk_native_geometry_reason(tokens, self._decode_phase, valid_tokens)
        if self.topk_native_selected and native_geometry_reason is None and self.topk_native_enabled:
            topk_weights, topk_indices = self._routing_pairs(x)
            routed = self._topk_native_routed_experts(x, topk_weights, topk_indices, tokens)
            ttnn.deallocate(topk_weights)
            ttnn.deallocate(topk_indices)
            out = ttnn.add(routed, shared)
            ttnn.deallocate(routed)
            ttnn.deallocate(shared)
            return out
        if self.topk_native_selected and native_geometry_reason is None:
            # This was a supported native prefill shape but the live layer lacked required objects.
            # It is the only case counted as fallback; decode, short and padded calls are intentional
            # sparse/gather choices and must not perturb the serving proof counter.
            self.topk_native_fallbacks += 1

        dense = self.routing_weights(x)
        if self._gather_ok(tokens, self._decode_phase, valid_tokens):
            # The gathered path (:data:`MOE_GATHER_EXPERTS`). It replaces the whole expert-group loop
            # below, not one group of it: the permutation it builds is over the sub-chunk's tokens, and
            # a 32-token group would give every expert a one-tile region — which is the sparse path's
            # geometry with much worse arithmetic and glue. Its own sub-chunking is
            # `MOE_GATHER_SUB_CHUNK`.
            E = self.cfg.num_experts
            parts = []
            start = 0
            while start < tokens:
                # A ragged last sub-chunk is allowed: chunk token counts are multiples of PREFILL_ALIGN,
                # so the remainder is too. Tails below MOE_GATHER_MIN_TOKENS use sparse MoE; that avoids
                # paying a one-tile region for every expert while retaining gathered execution for the
                # profitable prefix. Slicing is skipped for a call that fits one sub-chunk.
                span = min(MOE_GATHER_SUB_CHUNK, tokens - start)
                whole = span == tokens
                chunk = x if whole else ttnn.slice(x, [0, 0, start, 0], [1, 1, start + span, self.cfg.dim])
                routing = dense if whole else ttnn.slice(dense, [0, 0, start, 0], [1, 1, start + span, E])
                if span < MOE_GATHER_MIN_TOKENS:
                    part = self._routed_experts(chunk, routing, span)
                    # The sparse path returns the policy's block-float expert-output dtype; gathered
                    # combine is bfloat16 because `ttnn.embedding` requires a row-major bfloat16 table.
                    # Match the profitable prefix before concat, using the same memory placement too.
                    prefix = parts[0]
                    if part.dtype != prefix.dtype:
                        converted = ttnn.typecast(part, prefix.dtype, memory_config=prefix.memory_config())
                        ttnn.deallocate(part)
                        part = converted
                    elif part.memory_config() != prefix.memory_config():
                        converted = ttnn.to_memory_config(part, prefix.memory_config())
                        ttnn.deallocate(part)
                        part = converted
                else:
                    part = self._gather_routed_experts(chunk, routing, span)
                parts.append(part)
                if chunk is not x:
                    ttnn.deallocate(chunk)
                if routing is not dense:
                    ttnn.deallocate(routing)
                start += span
            ttnn.deallocate(dense)
            routed = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
            if len(parts) > 1:
                for part in parts:
                    ttnn.deallocate(part)
            # Reverse gather needs a ROW_MAJOR bfloat16 embedding table, so its reduction naturally
            # returns bfloat16. Restore the precision policy at the branch boundary: otherwise the
            # gathered experiment silently doubles the expert-parallel all-reduce payload and changes
            # the selected routed-output contract. One cast after concat is cheaper than one per
            # gathered sub-chunk and also normalizes a mixed sparse tail.
            if routed.dtype != self.policy.expert_act_dtype:
                contracted = ttnn.typecast(
                    routed,
                    self.policy.expert_act_dtype,
                    memory_config=routed.memory_config(),
                )
                ttnn.deallocate(routed)
                routed = contracted
        elif tokens <= self.group_tokens:
            routed = self._routed_experts(x, dense, tokens, valid_tokens=valid_tokens)
            ttnn.deallocate(dense)
        else:
            # Hoist the two per-call quantities out of the expert-group loop, exactly as the router
            # above them already is: the sparsity mask and the down projection's score operand are
            # both functions of the whole-call `dense`, so rebuilding them per group repeats the
            # same reduce/relayout/permute once per 32 tokens. Computed once here, each group takes
            # a slice. Measured at the shipped 2048-token prefill shape in
            # doc/fused_decoder/logs/probe_router_and_reduce.txt (`MASKHOIST` rows), where the
            # hoisted spelling is roughly half the cost of the per-group one.
            E = self.cfg.num_experts
            n_groups = tokens // TILE
            # Multi-group calls are prefill only, where the physical alignment makes every row real;
            # `forward` rejects a partially valid multi-group call above via _active_expert_mask.
            all_masks = self._active_expert_mask(dense, n_groups, valid_tokens)  # [1, n_groups, 1, E]
            all_scores = ttnn.permute(dense, (0, 3, 2, 1))  # [1, E, tokens, 1]
            parts = []
            for start in range(0, tokens, self.group_tokens):
                span = min(self.group_tokens, tokens - start)
                # `memory_config`: this slice is the routed gate/up matmul's `in0`, and that matmul is the
                # largest op of the prefill window. `tt-perf-report`'s advice on it - visible only once the
                # report is given `--active-experts`, which review round 14 found missing - is "place input 0
                # in L1". It costs no extra op here because the slice already dispatches; one group of
                # `group_tokens x dim` at bfloat16 is ~131 KB, far inside the budget the expert intermediates
                # already fit. `ROUTED_IN0_MEMORY` is a module constant so `logs/ab_routed_in0.py` can measure
                # the shipped choice against the alternative at the layer rather than by hand-editing; work_log
                # §4.19 reads its rows.
                chunk = ttnn.slice(
                    x, [0, 0, start, 0], [1, 1, start + span, self.cfg.dim], memory_config=ROUTED_IN0_MEMORY
                )
                # `_routed_experts` reads `dense_routing` only to build what it was not given, and
                # for the whole-call mask on its `groups > 1` branch. With both kwargs supplied that
                # leaves exactly one reader — the `span > TILE` branch — so slicing it at the shipped
                # 32-token group would dispatch a device op nothing reads. The fused stage's review found 64 of them
                # per 2048-token prefill.
                scores = ttnn.slice(dense, [0, 0, start, 0], [1, 1, start + span, E]) if span > TILE else None
                group_mask = ttnn.slice(all_masks, [0, start // TILE, 0, 0], [1, (start + span) // TILE, 1, E])
                group_scores = ttnn.slice(all_scores, [0, 0, start, 0], [1, E, start + span, 1])
                part = self._routed_experts(
                    chunk, scores, span, group_mask=group_mask, scores=group_scores, valid_tokens=None
                )
                ttnn.deallocate(chunk)
                if scores is not None:
                    ttnn.deallocate(scores)
                ttnn.deallocate(group_mask)
                ttnn.deallocate(group_scores)
                parts.append(part)
            ttnn.deallocate(all_masks)
            ttnn.deallocate(all_scores)
            ttnn.deallocate(dense)
            routed = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
            if len(parts) > 1:
                for part in parts:
                    ttnn.deallocate(part)

        out = ttnn.add(routed, shared)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out


class OptimizedDecoder(LightweightModule):
    """One graph-fused Ornith decoder layer on a TTNN mesh device.

    Construct with :meth:`from_state_dict`; call :meth:`allocate_state` (and, for
    ``full_attention`` layers, :meth:`allocate_kv_cache` / :meth:`attach_kv_cache`) before the
    first forward. The public API matches
    :class:`~models.autoports.ornith_ai_ornith_1_0_35b.tt.functional_decoder.FunctionalDecoder`.
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
        policy: PrecisionPolicy = DEFAULT_POLICY,
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

        self.policy = policy
        #: Dense token-mixer projections: the packed in-projection and the output projection.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.proj_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=policy.proj_fp32_acc,
            packer_l1_acc=policy.proj_packer_l1_acc,
        )
        #: The three float32 recurrent-state matmuls. Kept separate from the projection group: the
        #: state is the model's exact carry between decode steps, so its fidelity is not swept with
        #: the projections' weight dtype.
        self.state_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.state_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=policy.state_fp32_acc,
            packer_l1_acc=False,
        )
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.sdpa_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=policy.sdpa_fp32_acc,
            packer_l1_acc=False,
        )
        #: Paged flash-decode program config. Explicit, and it matters far more than a knob usually
        #: does: at Ornith's cache geometry the op *default* is more than an order of magnitude slower
        #: than this one, which is why the fused stage's explicit config is kept rather than replaced
        #: by the default. README §5.5 quotes both times.
        #:
        #: `doc/optimized_decoder/logs/probe_decode_micro.txt` (`SDPA` rows) sweeps four grids x five
        #: chunk pairs at an 8192-token context under the shipped BFP8 paged cache, at this exact
        #: compute-kernel contract (none: see below). The k-chunk is
        #: the only live axis there — a 128-token or unbounded chunk is a few microseconds faster than
        #: the shipped 64 in isolation, and 32 is markedly slower — but a k-chunk **larger than the
        #: 64-token paged block size is wrong**, not just risky: the isolated op agrees with itself
        #: (the probe's reference is the op default on the same page table, so it cannot see this),
        #: while the layer's decode PCC against the HF golden collapses far below the bar at the paged
        #: contexts the tests use (`logs/ab_sdpa_decode_contract.txt`). So 128 is rejected on
        #: correctness with that evidence, and 64 — one k-chunk per page — stays.
        #:
        #: The grid is the one axis of this config that is **not** a latency knob, and both the sweep and
        #: review round 9 misread it. `8x4` is the measured winner across 8x4 / 8x8 / 4x8 / 11x10, a microsecond
        #: ahead of this 8x8 at identical PCC, and at the layer the two are a dead heat because SDPA is ~2 % of
        #: a decode step (`logs/ab_sdpa_decode_grid.txt`). Round 9 found nothing recorded that gap, which was
        #: fair. Taking it was still wrong, and the suite is what said so: **flash-decode assigns at least one
        #: core per batch row** (`TT_FATAL(num_cores_available >= B)`,
        #: `sdpa_decode_program_factory.cpp:191`), so a 32-core grid caps decode at batch 32 and the supported
        #: batch-40 and batch-56 cases die inside the op. The grid therefore encodes the largest decode batch
        #: the layer can serve, and 8x8's 64 cores are chosen to cover the 56 the tests exercise - not for
        #: latency, which is why that microsecond goes unclaimed. 11x10 clears the bound too and is slower, so this
        #: is also the fastest *legal* grid. `test_decode_runs_the_tuned_program_configs` asserts the relation
        #: rather than the literal, so the "free win" cannot be re-taken by inspection.
        self.decode_sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=32,
            k_chunk_size=self.page_block_size,
            exp_approx_mode=False,
        )
        self.conv_compute_kernel_config = _conv_compute_config(mesh_device.arch())
        # rotary_embedding_hf defaults to math_approx_mode=True; the layer is accuracy-graded
        # against an HF float32 golden, so the exact SFPU path is used instead.
        self.rope_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.k_cache = None
        self.v_cache = None
        self.recurrent_state = None
        # list of ``conv_kernel_dim - 1`` buffers, each `_conv_state_shape(B)`: [1, 1, B, conv_dim]
        # folded (DECODE_COMPACT_ROWS) or [B, 1, conv_dim] unfolded
        self.conv_state = None
        self.conv1d_lengths = []  # prefill block lengths ttnn.conv1d accepted at the allocated batch
        self.batch_size = None
        self.batch_idxs = None  # [batch] int32 device tensor for the batched paged_fill_cache
        # The recurrent read `q @ state` is a rank-4 [B, 32, 1, 128] x [B, 32, 128, 128] batched
        # matmul, i.e. B*32 per-head matmuls.
        # Left to ttnn's default heuristic it lands on 4 cores; naming the whole grid puts the 32
        # per-head matmuls across it (measured in doc/fused_decoder/logs/probe_decode_micro.txt).
        grid = mesh_device.compute_with_storage_grid_size()
        self.full_core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        self.proj_cfgs = _ProjectionConfigs(mesh_device, policy)
        self._norm_shard_cache: dict[tuple, tuple] = {}
        #: Bytes of worker L1 a single decode-path intermediate may occupy. Same fraction and same
        #: source as the MoE's expert budget.
        self._l1_budget = int(EXPERT_L1_BUDGET_FRACTION * grid.x * grid.y * _worker_l1_bytes())
        self._state_cfg_cache: dict[tuple, object] = {}
        #: Set by :meth:`_block` for the duration of one forward. The decode-tuned program configs
        #: and the width-sharded norms are selected by *phase*, not by shape: a 128-token prefill
        #: block has the same tile height as a batch-4 decode step, and applying a decode config to
        #: it asks for an ``in0`` circular buffer sized for a prefill activation.
        self._decode_phase = False

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
        moe_group_tokens: int = DEFAULT_MOE_GROUP_TOKENS,
        rope_mode: str = DEFAULT_ROPE_MODE,
        dtype=ttnn.bfloat16,
        policy: PrecisionPolicy | str = DEFAULT_POLICY,
        **kwargs,
    ) -> "OptimizedDecoder":
        """Build a layer from an HF decoder-layer state dict (same keys as the functional decoder).

        ``policy`` selects the per-tensor-group weight dtype and math fidelity
        (:class:`PrecisionPolicy`); ``dtype`` remains the dtype of everything the policy does not
        name (norm weights, RoPE tables, conv taps and the like). Passing
        ``policy="fused-parity"`` reproduces the fused decoder's dtypes exactly, which is what the
        before/after tables in ``doc/optimized_decoder/README.md`` are measured against.
        """
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
        if isinstance(policy, str):
            if policy not in POLICIES:
                raise ValueError(f"unknown precision policy {policy!r}; known: {sorted(POLICIES)}")
            policy = POLICIES[policy]
        import torch

        config = OrnithDecoderConfig.from_hf_config(hf_config)
        max_context = int(max_context or config.max_position_embeddings)
        if prefill_chunk % PREFILL_ALIGN:
            raise ValueError(f"prefill_chunk {prefill_chunk} must be a multiple of {PREFILL_ALIGN}")
        if moe_group_tokens % TILE:
            raise ValueError(f"moe_group_tokens {moe_group_tokens} must be a multiple of {TILE}")
        if prefill_chunk % page_block_size:
            # _attention_prefill computes blk0 = chunk_start_idx // page_block_size and lets
            # paged_fill_cache fill from offset 0 of the sliced page table, so a chunk boundary that
            # is not also a page boundary would write the chunk at the wrong offset within its block.
            # chunk_start_idx is always a multiple of the chunk size, so this is the condition.
            raise ValueError(
                f"prefill_chunk {prefill_chunk} must be a multiple of page_block_size "
                f"{page_block_size}: chunk boundaries have to land on page boundaries"
            )
        kind = config.layer_kind(layer_idx)

        def upload(t, tensor_dtype=dtype, layout=ttnn.TILE_LAYOUT):
            # Block-float targets (bfloat8_b / bfloat4_b) are packed by ttnn from the host tensor:
            # the exponent is shared per face, so rounding the host copy to bfloat16 first would
            # discard mantissa bits before that packing sees them. Only a plain bfloat16 target
            # takes the cheap host cast.
            return ttnn.as_tensor(
                t.to(torch.bfloat16).contiguous() if tensor_dtype == ttnn.bfloat16 else t.float().contiguous(),
                dtype=tensor_dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        weights: dict = {}
        for key, dst in (("input_layernorm.weight", "attn_norm"), ("post_attention_layernorm.weight", "ff_norm")):
            weights[dst] = upload((state_dict[key].float() + 1.0).reshape(1, 1, 1, -1))

        rope = None
        if kind == "full_attention":
            n_heads, n_kv, head_dim = config.n_heads, config.n_kv_heads, config.head_dim
            perm = torch.tensor(
                _rope_head_permutation(head_dim, config.rope_dim) if rope_mode == "full" else range(head_dim),
                dtype=torch.long,
            )

            # q_proj is 2x wide: per head, [query(head_dim) | output-gate(head_dim)].
            qg = state_dict["self_attn.q_proj.weight"].float().reshape(n_heads, 2 * head_dim, config.dim)
            q_rows = qg[:, :head_dim, :][:, perm, :].reshape(n_heads * head_dim, config.dim)
            gate_rows = qg[:, head_dim:, :].reshape(n_heads * head_dim, config.dim)
            k_rows = state_dict["self_attn.k_proj.weight"].float().reshape(n_kv, head_dim, config.dim)[:, perm, :]
            k_rows = k_rows.reshape(n_kv * head_dim, config.dim)
            v_rows = state_dict["self_attn.v_proj.weight"].float()
            # Shared-LHS packing, in the order nlp_create_qkv_heads wants: [q | k | v | gate].
            weights["attn_in"] = upload(
                torch.cat([q_rows, k_rows, v_rows, gate_rows], dim=0).transpose(0, 1), policy.proj_dtype
            )
            weights["o_proj"] = upload(state_dict["self_attn.o_proj.weight"].transpose(0, 1), policy.proj_dtype)
            # Q/K norms act over head_dim only and follow the same head-dim permutation. RMSNorm is
            # permutation-equivariant, so permuting the weight reproduces the unpermuted result.
            for src, dst in (("self_attn.q_norm.weight", "q_norm"), ("self_attn.k_norm.weight", "k_norm")):
                weights[dst] = upload(((state_dict[src].float() + 1.0)[perm]).reshape(1, 1, 1, -1))
            rope = OrnithFusedRope(
                mesh_device,
                config,
                max_context=max_context,
                # Rounded up to a whole prefill chunk and then given one more chunk of headroom:
                # the final block's *physical* window can end past max_context, and those extra rows
                # only ever rotate zero-padded activations.
                table_context=_align_up(max_context, prefill_chunk) + prefill_chunk,
                mode=rope_mode,
            )
        elif kind == "linear_attention":
            prefix = "linear_attn."
            # Shared-LHS packing of the four in-projections. `a` and `b` are num_value_heads (32)
            # wide, i.e. exactly one tile, so every slice offset stays tile-aligned.
            weights["gdn_in"] = upload(
                torch.cat(
                    [
                        state_dict[prefix + "in_proj_qkv.weight"].float(),
                        state_dict[prefix + "in_proj_z.weight"].float(),
                        state_dict[prefix + "in_proj_a.weight"].float(),
                        state_dict[prefix + "in_proj_b.weight"].float(),
                    ],
                    dim=0,
                ).transpose(0, 1),
                policy.proj_dtype,
            )
            weights["gdn_out"] = upload(state_dict[prefix + "out_proj.weight"].transpose(0, 1), policy.proj_dtype)
            # Gated-DeltaNet output norm is a *standard* RMSNorm (weights ~ 1), not zero-centered.
            weights["gdn_norm"] = upload(state_dict[prefix + "norm.weight"].float().reshape(1, 1, 1, -1))
            conv_w = state_dict[prefix + "conv1d.weight"].float()
            weights["conv_taps"] = [
                upload(conv_w[:, 0, k].reshape(1, 1, -1)) for k in range(config.linear_conv_kernel_dim)
            ]
            # ttnn.conv1d cannot take all conv_dim channels at once, but a depthwise conv is
            # separable over channels and CONV1D_CHANNELS-wide calls do run — several times faster
            # than the FIR form at 2048 tokens; the two `CONV1DTIME` lines of
            # doc/fused_decoder/logs/probe_conv1d_and_norm.txt carry the measurement, and are cited
            # rather than copied here because they move a little on every re-run. The
            # weights depend on the input length, so every length the layer can produce is prepared
            # here, on the host, once.
            weights["conv1d_host"] = _conv1d_host_weights(conv_w, config)
            weights["conv1d_compute_config"] = _conv_compute_config(mesh_device.arch())
            weights["conv1d_weights"] = {}
            weights["A_neg"] = upload(
                (-state_dict[prefix + "A_log"].float().exp()).reshape(1, 1, -1), tensor_dtype=ttnn.float32
            )
            weights["dt_bias"] = upload(
                state_dict[prefix + "dt_bias"].float().reshape(1, 1, -1), tensor_dtype=ttnn.float32
            )
            from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import build_fused_const_tiles

            weights["gdn_const_tiles"] = build_fused_const_tiles(mesh_device, GDN_CHUNK)
            weights["pos_ramp"] = upload(
                torch.arange(prefill_chunk, dtype=torch.float32).reshape(1, prefill_chunk, 1),
                tensor_dtype=ttnn.float32,
            )
        else:
            raise ValueError(f"unsupported layer kind {kind!r}")

        moe_weights = cls._load_moe_weights(mesh_device, config, state_dict, upload, policy=policy)
        moe = OptimizedMoE(mesh_device, config, moe_weights, group_tokens=moe_group_tokens, policy=policy)

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
            policy=policy,
        )

    @staticmethod
    def _load_moe_weights(mesh_device, config, state_dict, upload, prefix="mlp.", *, policy=DEFAULT_POLICY):
        """Device-resident MoE weights, with the shared expert's three projections packed.

        ``experts.gate_up_proj`` is ``[E, 2*moe_intermediate, hidden]`` with the gate rows first
        (``transformers``' registered fusion for ``qwen3_5_moe_text``).
        """
        import torch

        def get(name):
            return state_dict[f"{prefix}{name}"]

        inter = config.moe_intermediate_size
        fused = get("experts.gate_up_proj")
        if fused.shape[1] != 2 * inter:
            raise ValueError(f"experts.gate_up_proj dim1 {fused.shape[1]} != 2*{inter}")

        shared_inter = config.shared_expert_intermediate_size
        # [gate | up | shared-expert sigmoid router], the router zero-padded out to one tile so the
        # slice offsets are tile-aligned; the padding columns are exact zeros and never read.
        router_col = get("shared_expert_gate.weight").float().transpose(0, 1).reshape(config.dim, 1)
        shared_in = torch.cat(
            [
                get("shared_expert.gate_proj.weight").float().transpose(0, 1),
                get("shared_expert.up_proj.weight").float().transpose(0, 1),
                router_col,
                torch.zeros(config.dim, TILE - 1, dtype=torch.float32),
            ],
            dim=1,
        )

        weights = {
            "router": upload(
                get("gate.weight").float().transpose(0, 1).reshape(1, 1, config.dim, config.num_experts),
                policy.router_dtype,
            ),
            "shared_in": upload(shared_in.reshape(1, 1, config.dim, 2 * shared_inter + TILE), policy.shared_dtype),
            "shared_down": upload(
                get("shared_expert.down_proj.weight").float().transpose(0, 1).reshape(1, 1, shared_inter, config.dim),
                policy.shared_dtype,
            ),
        }
        # [1, E, hidden, 2*moe_intermediate] with the gate half first: one shared-LHS sparse
        # matmul per group instead of two half-width ones (see OptimizedMoE._routed_experts).
        weights["expert_gate_up"] = upload(fused.transpose(-2, -1).unsqueeze(0), policy.expert_gate_up_dtype)
        weights["expert_down"] = upload(
            get("experts.down_proj").transpose(-2, -1).unsqueeze(0), policy.expert_down_dtype
        )
        # `_gather_support_reason`: skip the upload entirely when the path cannot run at this geometry
        # (a single-device mesh, where `num_experts` is the full 256). Otherwise an unsupported build
        # would pay 453 MB per layer for a weight layout nothing will ever read, and
        # `OptimizedMoE.gather_experts` would claim the path is available when it is not.
        if (MOE_GATHER_EXPERTS or MOE_TOPK_NATIVE) and _gather_support_reason(config.num_experts) is None:
            # The gathered path's layout: one 2D tensor per expert per projection, oriented the way
            # `unified_routed_expert_ffn` reads them — gate/up as (K=hidden, N=moe_intermediate) and
            # down as (K=moe_intermediate, N=hidden), i.e. the transpose of the HF `[out, in]` rows.
            # Uploaded *alongside* the packed pair above, because the switch is prefill-only and decode
            # keeps the sparse path; see MOE_GATHER_EXPERTS for what that costs.
            down = get("experts.down_proj")
            weights["expert_gate_list"] = [
                upload(fused[e, :inter].float().transpose(0, 1), policy.expert_gate_up_dtype)
                for e in range(config.num_experts)
            ]
            weights["expert_up_list"] = [
                upload(fused[e, inter : 2 * inter].float().transpose(0, 1), policy.expert_gate_up_dtype)
                for e in range(config.num_experts)
            ]
            weights["expert_down_list"] = [
                upload(down[e].float().transpose(0, 1), policy.expert_down_dtype) for e in range(config.num_experts)
            ]
        return weights

    # ------------------------------------------------------------------ state
    def allocate_kv_cache(self, num_blocks: int, dtype=None):
        """Allocate and attach a paged KV cache ``[num_blocks, n_kv_heads, page_block_size, head_dim]``.

        ``dtype`` defaults to the precision policy's ``kv_cache_dtype`` rather than to bfloat16, so a
        caller that does not name one gets the cache the measured path was tuned for. Naming one still wins, and
        it then **diverges from the policy** - which is why `_prefill_sdpa_config` resolves its chunk from the
        attached cache rather than from `policy.kv_cache_dtype`. §4.2's cache-dtype A/B does *not* come through
        here: it goes through `bench.py --set kv_cache_dtype=...`, which replaces the policy, so this argument had
        no coverage at all until review round 16 pointed out that the two mechanisms disagree about which dtype
        the layer is running.
        """
        dtype = self.policy.kv_cache_dtype if dtype is None else dtype
        if not self.is_full_attention:
            return None
        shape = [num_blocks, self.cfg.n_kv_heads, self.page_block_size, self.cfg.head_dim]
        self.k_cache = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        self.v_cache = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        return self.k_cache, self.v_cache

    def attach_kv_cache(self, k_cache, v_cache):
        if not self.is_full_attention:
            raise TypeError("linear_attention layers have no KV cache")
        self.k_cache = k_cache
        self.v_cache = v_cache

    def allocate_state(self, batch_size: int):
        """Allocate the per-batch state. Buffers are persistent and only ever written in place, so
        their device addresses stay valid across trace replays.

        **This is setup, and it is not host-free**: it builds the paged-fill row indices and, for
        ``linear_attention``, prepares and probes the ``ttnn.conv1d`` weights. The forward paths call
        it lazily if the caller never did, so a *first* forward on an unallocated layer performs host
        calls and every subsequent one does not. Call it explicitly before capturing a trace or before
        any measurement, as this stage's tests and benchmarks do.
        """
        # The gathered routed-expert path's constants (:data:`MOE_GATHER_EXPERTS`). Here rather than in
        # the forward for the same reason as everything else in this method: they need
        # ``ttnn.from_torch``. A no-op when the path is off, and idempotent across re-entry.
        self.moe.prepare_gather_experts(self.prefill_chunk)
        self.moe.prepare_topk_native(self.prefill_chunk)
        # Freed before being replaced: prefill_forward/decode_forward re-enter here when a
        # full_attention layer is handed a larger batch than it was allocated for.
        if self.batch_idxs is not None:
            ttnn.deallocate(self.batch_idxs)
            self.batch_idxs = None
        for prepared in self.w.get("conv1d_weights", {}).values():
            for weight in prepared:
                ttnn.deallocate(weight)
        self.w["conv1d_weights"] = {}

        import torch

        self.batch_size = batch_size
        if self.is_full_attention:
            # Row indices for the single batched paged_fill_cache call.
            self.batch_idxs = ttnn.from_torch(
                torch.arange(batch_size, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
            return
        # ttnn.conv1d weights depend on the batch as well as the input length, so they are prepared
        # here rather than in from_state_dict: allocate_state is setup, and this keeps every host
        # call out of the forward paths.
        self.w["conv1d_weights"] = _prepare_conv1d_weights(
            self.device,
            self.w.get("conv1d_host", []),
            self.cfg,
            self.prefill_chunk,
            batch_size,
            self.w["conv1d_compute_config"],
        )
        # Which block lengths actually got a working conv1d program at this batch. Coverage shrinks
        # as the batch grows (the conv's circular buffers are sized per batch), so it is logged
        # rather than assumed: at batch 1 every length works, at batch 32 none do and the whole
        # prefill runs on the FIR fallback. doc/fused_decoder/README.md tabulates the measured set and
        # the per-batch split between the two refusal classes.
        self.conv1d_lengths = sorted(self.w["conv1d_weights"])
        total = self.prefill_chunk // PREFILL_ALIGN
        logger.info(
            f"layer {self.layer_idx} batch {batch_size}: ttnn.conv1d accepted "
            f"{len(self.conv1d_lengths)}/{total} prefill block lengths {self.conv1d_lengths}"
        )
        self.recurrent_state = ttnn.zeros(
            [batch_size, self.cfg.linear_num_value_heads, self.cfg.linear_key_head_dim, self.cfg.linear_value_head_dim],
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.conv_state = [
            ttnn.zeros(
                self._conv_state_shape(batch_size),
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
    def _decode_rows(self, x):
        """``(batch, folded)`` for a decode activation in either residual convention.

        The folded convention is ``[1, 1, batch, dim]`` (:data:`DECODE_COMPACT_ROWS`) and the
        unfolded one is ``[batch, 1, dim]``, so the batch is dim -2 or dim 0 and the rank tells the
        two apart: only :meth:`_block`'s fold produces a rank-4 activation at decode. Every decode
        helper that needs the user count reads it here rather than from ``x.shape[0]``, which is 1
        for a folded step.
        """
        dims = [int(d) for d in x.shape]
        return (dims[-2], True) if len(dims) == 4 else (dims[0], False)

    def _fold_decode_rows(self, x, tokens: int):
        """``x`` folded to ``[1, 1, tokens, dim]`` at decode, or ``x`` unchanged.

        The single reader of :data:`DECODE_COMPACT_ROWS` on the forward path, and it reads the module
        global at call time rather than a captured import, so flipping the knob flips both this class
        and :class:`~models.autoports.ornith_ai_ornith_1_0_35b.tt.multichip_decoder.MultichipDecoder`
        — whose ``_block`` calls this method for exactly that reason.

        The result may be a *view* of ``x`` (at batch 1 the two shapes describe the same tile), so
        the caller frees it through :func:`_free_unless_aliased` rather than unconditionally.
        """
        if not (self._decode_phase and DECODE_COMPACT_ROWS):
            return x
        return ttnn.reshape(x, [1, 1, int(tokens), self.cfg.dim])

    def _conv_state_shape(self, batch_size: int):
        """Shape of one persistent conv-history buffer.

        Folded to ``[1, 1, batch, conv_dim]`` under :data:`DECODE_COMPACT_ROWS`, because the decode
        step's four tap ops and three history copies all run on these buffers and unfolded they are
        ``batch`` tile rows of which ``batch`` carry data. The slot axis moves from dim 0 to dim -2
        with the fold, which is why :meth:`OrnithModel._slot_axis` derives it from the buffer's shape
        instead of assuming dim 0 — a per-slot write on the wrong axis would hand one request another
        request's DeltaNet history with no shape error to catch it.
        """
        if DECODE_COMPACT_ROWS:
            return [1, 1, int(batch_size), self.cfg.conv_dim]
        return [int(batch_size), 1, self.cfg.conv_dim]

    def _shard_feeds_projection(self, role, shape, norm_cfg) -> bool:
        """Can ``role``'s tuned decode config consume this norm's width-sharded output directly?

        Every condition here mirrors the ``mcast_in0`` sharded-``in0`` validation in
        ``ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp``: WIDTH_SHARDED and
        ROW_MAJOR (which :meth:`_norm_shard` always builds), ``fuse_batch`` (always set here),
        ``per_core_M == shard_shape[0] / tile_h``, and ``block_w % in0_block_w == 0`` where
        ``block_w`` is the per-core shard width in tiles. The last is the binding one: the residual
        norms shard ``dim`` over :attr:`NORM_SHARD_CORES`, so a role whose ``in0_block_w`` exceeds
        that per-core width cannot take the shard. That is why the MoE roles are excluded — their
        tuned ``in0_block_w`` is wider than the norm's per-core shard, and lowering it to fit is
        measurably slower (`logs/probe_dense_matmul.txt`).
        """
        if not self._decode_phase:
            return False
        weight = self.w.get(role)
        if weight is None:
            return False
        rows = _physical_rows(shape)
        cfg = self.proj_cfgs.get(
            role,
            rows,
            shape[-1],
            int(weight.shape[-1]),
            fp32_acc=self.policy.proj_fp32_acc,
            decode=True,
        )
        if cfg is None or not getattr(cfg, "mcast_in0", False) or not getattr(cfg, "fuse_batch", False):
            return False
        if int(cfg.per_core_M) != rows // TILE:
            return False
        if norm_cfg.block_w % int(cfg.in0_block_w) != 0:
            return False
        # The shard's cores must be the ones the matmul will actually read from, and that is **not** implied by
        # the validation above. `matmul_multicore_reuse_mcast_1d_program_factory.cpp` takes only the *count* of
        # in0 sender cores from the shard spec and then lays those cores out inside the **matmul's** own rect in
        # row-major order, so a shard grid that is not the matmul rect's first `n` row-major cores passes every
        # `TT_FATAL` and reads the wrong cores. The layer's norm shard is a single row, so requiring the matmul
        # grid to be at least that wide makes the two coincide exactly.
        #
        # This is not hypothetical. Review round 28 predicted it as unreachable-but-silent; review round 32
        # reached it: a 16-core norm shard is 8x2 on this device, it is ~8 us a step *faster* at the layer, and
        # it drops `full_attention` decode PCC far below the acceptance bar while raising no error. It is correct with the shard
        # carry disabled, which is what pins the cause here rather than on the norm. A latency-only A/B cannot
        # see that, which is why `ab_norm_shard_cores.txt` now records PCC beside every arm.
        norm_grid = norm_cfg.compute_with_storage_grid_size
        return int(norm_grid.y if hasattr(norm_grid, "y") else norm_grid[1]) == 1 and int(
            norm_grid.x if hasattr(norm_grid, "x") else norm_grid[0]
        ) <= int(
            cfg.compute_with_storage_grid_size.x
            if hasattr(cfg.compute_with_storage_grid_size, "x")
            else cfg.compute_with_storage_grid_size[0]
        )

    def _proj_linear(self, x, weight, role):
        """Dense token-mixer projection under this role's tuned decode program config.

        Prefill-shaped activations get ``None`` back from :class:`_ProjectionConfigs` and fall
        through to ttnn's 2D heuristic with a DRAM output, which is the right family for them; only
        the skinny decode shapes take the explicit 1D config and the L1 output.
        """
        shape = [int(d) for d in x.shape]
        rows = _physical_rows(shape)
        cfg = self.proj_cfgs.get(
            role,
            rows,
            shape[-1],
            int(weight.shape[-1]),
            fp32_acc=self.policy.proj_fp32_acc,
            decode=self._decode_phase,
        )
        return ttnn.linear(
            x,
            weight,
            compute_kernel_config=self.compute_kernel_config,
            program_config=cfg,
            # L1 only for the decode shapes: a prefill projection output is tens of megabytes.
            memory_config=ttnn.L1_MEMORY_CONFIG
            if (cfg is not None and self._decode_phase)
            else ttnn.DRAM_MEMORY_CONFIG,
        )

    #: Cores the decode residual RMSNorms are width-sharded over. ``ttnn.rms_norm`` parallelises
    #: over *rows*, and a decode activation is a single tile of rows, so the interleaved form the
    #: fused stage used lands the whole 2048-wide norm on one core. Width-sharding the input and
    #: output and naming a ``LayerNormShardedMultiCoreProgramConfig`` moves it onto ``cores`` cores,
    #: which cuts the isolated norm's time by around a third.
    #:
    #: On the **isolated op** 8 cores is fastest (``logs/probe_decode_micro.txt``, ``NORM`` rows, min-of-three
    #: with a reported ``spread=``), then 4, then 16/32/64 as the per-core block shrinks. The ladder is not
    #: monotonic in the shard count and this comment claimed it was until review round 32.
    #:
    #: At the **whole layer** 8 is fastest or tied at every arm measured, with the arms alternating
    #: build-by-build (``logs/ab_norm_shard_cores.txt``): on ``full_attention`` 8/16/32 tie, and on
    #: ``linear_attention`` 16 costs about 11 µs and 32 about 12. So 8 ships on the layer measurement, not
    #: merely on consistency with the rest of the norm evidence.
    #:
    #: **Review round 32 found something sharper here, and it is the reason this block is long.** The
    #: artifact then on disk showed the 16-core arm ~8 µs a step *faster* on ``full_attention``, reproducibly,
    #: and five places in this stage - including this comment - were calling the arms indistinguishable. The
    #: arm was faster because it was **wrong**: a 16-core shard of a 2048-wide norm is 8x2 on this device, and
    #: :meth:`_shard_feeds_projection` let it through to a matmul that takes only the *core count* from the
    #: shard spec and then lays those cores out inside its own rect, so it read the wrong cores. Layer decode
    #: PCC fell far below the acceptance bar, with no error raised, and it is correct again with the shard carry
    #: disabled - which is what pins the cause on the interaction rather than on the norm. Review round 28 had
    #: predicted exactly this as unreachable-but-silent; round 32's latency measurement reached it.
    #:
    #: What ships as a result: the single-row guard in :meth:`_shard_feeds_projection`, and a
    #: ``pcc_vs_shipped`` column in ``ab_norm_shard_cores.txt`` so a latency-only arm can never again read as
    #: a candidate. With the guard in place the 16-core arm is correct, and it is no longer faster - the whole
    #: gap was the corrupted path.
    #:
    #: Review round 6 found an earlier version of this comment claiming the isolated ladder was monotonic the
    #: other way and citing ``ab_norm_shard_width.txt``, which varies a different knob (which norms shard, not
    #: over how many cores); ``ab_norm_shard_cores.txt`` exists because of that finding, and review round 11
    #: corrected a "marginally best on both kinds" claim the artifact reversed.
    NORM_SHARD_CORES = 8

    #: Narrowest activation that takes the sharded norm path. 0 means every decode-shaped norm
    #: does, including the 256-wide Q/K head-dim norms. Restricting it to the 2048-wide residual
    #: norms was measured as well, on the theory that two conversions would cost more than the narrow
    #: norm saves; it is reproducibly *worse* on ``full_attention`` and identical on
    #: ``linear_attention`` (which has no Q/K norm), so the simpler contract is also the faster one.
    #: ``doc/optimized_decoder/logs/ab_norm_shard_width.txt`` has both arms, two runs each.
    NORM_SHARD_MIN_WIDTH = 0

    #: Largest activation height, in *tile rows*, that takes the sharded norm path. Above it the
    #: interleaved form already spreads over enough cores by row and the shard would be large.
    #:
    #: What reaches it changed with :data:`DECODE_COMPACT_ROWS` and the *bound* deliberately did not.
    #: Folded, the two residual norms are one tile row at every batch up to 32 and two up to 64, so
    #: they now take this path at every servable batch instead of falling out above batch 4 — that is
    #: the fold widening the reach of a rule that was already measured, not a new claim about 4. The
    #: narrow Q/K head-dim norms are unaffected: they are head-shaped (``[1, batch, heads, dim]``),
    #: not residual-shaped, so they still leave this path above batch 4.
    NORM_SHARD_MAX_M_TILES = 4

    def _norm_shard(self, rows: int, width: int):
        """``(program_config, memory_config)`` for a width-sharded decode RMSNorm, or ``(None, None)``.

        ``rows`` is the tensor's **physical** row count, not its logical one: a rank-3 decode
        activation ``[batch, 1, dim]`` in TILE layout pads each batch entry to a whole 32-row tile,
        so a batch-4 step is 128 physical rows, not 4. Getting that wrong builds a shard spec whose
        height covers a quarter of the tensor and the op rejects it with
        ``!shard_grid_fit_error.has_value()``.

        That physical count is also why :data:`DECODE_COMPACT_ROWS` widens this path's reach without
        touching it: the folded residual ``[1, 1, batch, dim]`` is 32 rows at every batch up to 32,
        so the two residual norms clear :attr:`NORM_SHARD_MAX_M_TILES` at every servable batch where
        unfolded they fell out above batch 4.

        Only decode-shaped activations take this path. A prefill activation already has enough rows
        to fill the grid the ordinary way, and width-sharding thousands of rows would need a
        block-sharded layout rather than this one.
        """
        if rows > TILE * self.NORM_SHARD_MAX_M_TILES:
            return None, None
        if width < self.NORM_SHARD_MIN_WIDTH:
            return None, None
        n_tiles = width // TILE
        cores = _largest_divisor_at_most(n_tiles, self.NORM_SHARD_CORES)
        if cores < 2 or width % cores:
            return None, None
        key = (rows, width, cores)
        cached = self._norm_shard_cache.get(key)
        if cached is None:
            grid = self.device.compute_with_storage_grid_size()
            cols = max((c for c in range(1, grid.x + 1) if cores % c == 0 and cores // c <= grid.y), default=0)
            if not cols:
                self._norm_shard_cache[key] = (None, None)
                return None, None
            rows_of_cores = cores // cols
            block_w = n_tiles // cores
            mem = ttnn.create_sharded_memory_config(
                shape=(rows, width // cores),
                core_grid=ttnn.CoreGrid(x=cols, y=rows_of_cores),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(cols, rows_of_cores),
                subblock_w=_largest_divisor_at_most(block_w, 8),
                block_h=rows // TILE,
                block_w=block_w,
                inplace=False,
            )
            cached = (cfg, mem)
            self._norm_shard_cache[key] = cached
        return cached

    def _widen_norm_output(self, out):
        """Restore :data:`NORM_OUTPUT_DTYPE` when a block-float residual made the norm produce one.

        ``ttnn.rms_norm`` has no ``dtype`` argument, so its output takes the residual's dtype. With
        the shipped ``residual_dtype=bfloat16`` this method is a no-op and no op is dispatched. With
        a block-float residual it is what makes the arm *buildable*: the norm feeds the token-mixer
        in-projection, whose output reaches
        ``nlp_create_qkv_heads_decode``, which asserts
        ``input_tensor.dtype() == FLOAT32 || input_tensor.dtype() == BFLOAT16``
        (``nlp_create_qkv_heads_decode_device_operation.cpp:41``) and raises
        ``TT_FATAL: Unsupported data format`` otherwise.

        The cast is therefore the *adapted* form of the bfloat8_b-residual candidate rather than a
        reason to reject it unmeasured - and its cost is why that candidate loses: it adds two
        ``typecast`` dispatches per layer to a decode step the stage has measured as launch-bound,
        against a residual stream that is one 2048-wide tile row per token. See
        ``doc/datatype_sweep/README.md``'s rejected-candidate table for the measurement.
        """
        if out.dtype in _NORM_SAFE_DTYPES:
            return out
        wide = ttnn.typecast(out, NORM_OUTPUT_DTYPE, memory_config=out.memory_config())
        ttnn.deallocate(out)
        return wide

    def _norm(self, x, weight, *, keep_sharded_for=None):
        """Zero-centered RMSNorm — the ``+1`` is already folded into ``weight``.

        Decode-shaped activations take a width-sharded multi-core program config; everything else
        keeps the interleaved form. README §5.5's generated knob table has the interleaved and
        width-sharded times, and `logs/ab_norm_shard_cores.txt` has the whole-layer comparison.

        ``keep_sharded_for`` names the projection role that consumes this norm. When the role's tuned
        config can take the shard directly, the result is returned **still width-sharded** and the
        ``sharded_to_interleaved`` between norm and projection disappears. Until review round 25 this
        stage asserted that no such case existed — that ``mcast_in0`` requires an interleaved ``in0``
        — and rejected the whole sharded-residual family on it. The assertion was false, and it came
        from the probe rather than the op: every ``mcast1d`` row in `probe_dense_matmul.py` handed the
        op a DRAM-interleaved activation, so no row could contradict it.
        ``matmul_device_operation.cpp`` validates a sharded ``in0`` for ``mcast_in0`` explicitly, and
        :meth:`_shard_feeds_projection` re-derives each of its conditions here.
        """
        shape = [int(d) for d in x.shape]
        cfg, mem = self._norm_shard(_physical_rows(shape), shape[-1]) if self._decode_phase else (None, None)
        if cfg is None:
            return self._widen_norm_output(ttnn.rms_norm(x, weight=weight, epsilon=self.cfg.norm_eps))
        x_sh = ttnn.to_memory_config(x, mem)
        out = ttnn.rms_norm(x_sh, weight=weight, epsilon=self.cfg.norm_eps, program_config=cfg, memory_config=mem)
        ttnn.deallocate(x_sh)
        out = self._widen_norm_output(out)
        if keep_sharded_for is not None and self._shard_feeds_projection(keep_sharded_for, shape, cfg):
            return out
        # Back to interleaved for the consumers that cannot take the shard - which is *not* a property of
        # `mcast_in0`, and saying so here is what closed a whole family until round 25 (see the docstring
        # above and §4.21). The binding rule is per role: a projection whose `in0_block_w` exceeds this
        # norm's per-core shard width cannot consume it, and
        # `paged_scaled_dot_product_attention_decode` rejects a non-sharded Q that is not in DRAM.
        # `tt-perf-report` advises placing a matmul's in0 in L1, and the two residual norms feed
        # matmuls, so they interleave into L1. The narrow head-dim norms do not: their result reaches
        # `paged_scaled_dot_product_attention_decode`, which rejects a non-sharded Q that is not in
        # DRAM ("Q tensor buffer type must be DRAM when not sharded"). That op *does* accept a
        # height-sharded Q in L1, so the constraint is on the interleaved form the intervening rotary and
        # concat ops produce here rather than on L1 as such - review round 26 asked for the
        # distinction, having just seen a whole family closed by an op-contract claim that was too broad.
        target = ttnn.L1_MEMORY_CONFIG if shape[-1] >= self.cfg.dim else ttnn.DRAM_MEMORY_CONFIG
        interleaved = ttnn.sharded_to_interleaved(out, target)
        ttnn.deallocate(out)
        return interleaved

    def max_gdn_prefill_batch(self) -> int:
        """Largest batch one ``chunk_gated_delta_rule`` launch can serve (``B*HV <= cores``)."""
        grid = self.device.compute_with_storage_grid_size()
        return max(1, (grid.x * grid.y) // self.cfg.linear_num_value_heads)

    # ------------------------------------------------------------------ full attention
    def _attention_prefill(self, x, page_table, chunk_start_idx):
        if self.k_cache is None:
            raise RuntimeError("call allocate_kv_cache()/attach_kv_cache() before prefill")
        if page_table is None:
            raise ValueError("full_attention prefill requires a page_table")
        cfg = self.cfg
        b, t = x.shape[0], x.shape[1]
        staged = page_table if isinstance(page_table, PrefillChunkInputs) else None
        if staged is not None:
            staged.validate_for(
                start_pos=chunk_start_idx,
                physical_len=t,
                batch=b,
                page_block_size=self.page_block_size,
            )
            full_page_table = staged.full_page_table
        else:
            # Public low-level compatibility path: static RoPE/page-table slices and the scalar
            # chunked-SDPA overload retain their original behaviour and program keys.
            full_page_table = page_table
        q_width = cfg.n_heads * cfg.head_dim
        kv_width = 2 * cfg.n_kv_heads * cfg.head_dim

        fused = self._proj_linear(x, self.w["attn_in"], "attn_in")
        fused = ttnn.reshape(fused, [b, 1, t, int(fused.shape[-1])])
        q_flat = _slice_last(fused, 0, q_width)
        kv_flat = _slice_last(fused, q_width, q_width + kv_width)
        gate = _slice_last(fused, q_width + kv_width, int(fused.shape[-1]))
        ttnn.deallocate(fused)
        gate = ttnn.reshape(gate, [b, t, q_width])

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            q_flat, kv_flat, num_heads=cfg.n_heads, num_kv_heads=cfg.n_kv_heads, transpose_k_heads=False
        )
        ttnn.deallocate(q_flat)
        ttnn.deallocate(kv_flat)

        q = self._norm(q, self.w["q_norm"])
        k = self._norm(k, self.w["k_norm"])
        cos, sin = (
            self.rope.indexed_forward(staged.position_idxs)
            if staged is not None
            else self.rope.prefill_forward(chunk_start_idx, t)
        )
        q = self._rope_prefill(q, cos, sin)
        k = self._rope_prefill(k, cos, sin)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

        if staged is not None:
            chunk_page_table, pt_owned = staged.fill_page_table, False
        else:
            blk0 = chunk_start_idx // self.page_block_size
            blk_n = _align_up(chunk_start_idx + t, self.page_block_size) // self.page_block_size
            chunk_page_table, pt_owned = _slice_owned(
                full_page_table, [0, blk0], [int(full_page_table.shape[0]), blk_n]
            )
        # One batched fill per cache: `batch_idx_tensor` writes row u of the input into
        # page_table[batch_idxs[u]], so a batch-32 prefill costs 2 launches, not 64.
        batch_idxs, idx_owned = _slice_owned(self.batch_idxs, [0], [b])
        # paged_fill_cache requires the fill tensor's dtype to match the cache's. Under a reduced
        # cache dtype the projection output is still bfloat16, so the cast is explicit here rather
        # than left to the op (which rejects the mismatch). Decode's paged_update_cache is the
        # opposite contract and keeps its bfloat16 input — see _attention_decode.
        k_fill, k_cast = self._cache_fill_tensor(k)
        v_fill, v_cast = self._cache_fill_tensor(v)
        ttnn.experimental.paged_fill_cache(self.k_cache, k_fill, chunk_page_table, batch_idx_tensor=batch_idxs)
        ttnn.experimental.paged_fill_cache(self.v_cache, v_fill, chunk_page_table, batch_idx_tensor=batch_idxs)
        if k_cast:
            ttnn.deallocate(k_fill)
        if v_cast:
            ttnn.deallocate(v_fill)
        if idx_owned:
            ttnn.deallocate(batch_idxs)
        if pt_owned:
            ttnn.deallocate(chunk_page_table)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        sdpa_kwargs = dict(
            scale=cfg.head_dim**-0.5,
            program_config=self._prefill_sdpa_config(chunk_start_idx, t),
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        if staged is not None:
            # Flexible chunked SDPA reads the offset from this persistent device scalar, so one
            # program serves every 2048-token scheduler boundary. It compiles against the complete
            # page-table width; the serving A/B must therefore price that max-prefix geometry.
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                self.k_cache,
                self.v_cache,
                full_page_table,
                chunk_start_idx_tensor=staged.chunk_start_idx_tensor,
                **sdpa_kwargs,
            )
        else:
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                self.k_cache,
                self.v_cache,
                full_page_table,
                chunk_start_idx,
                **sdpa_kwargs,
            )
        ttnn.deallocate(q)
        merged = ttnn.experimental.nlp_concat_heads(attn)
        ttnn.deallocate(attn)
        merged = ttnn.reshape(merged, [b, t, q_width])
        return self._attention_output(merged, gate)

    def _cache_fill_tensor(self, t):
        """``(tensor, owned)`` cast to the KV-cache dtype for ``paged_fill_cache``."""
        cache_dtype = self.k_cache.dtype
        if t.dtype == cache_dtype:
            return t, False
        return ttnn.typecast(t, cache_dtype), True

    def _prefill_sdpa_config(self, chunk_start_idx: int, phys_len: int):
        """Chunked-SDPA tiling. ``q_chunk`` must divide ``chunk_start_idx`` when it is non-zero.

        The cap is `PREFILL_SDPA_CHUNK`, measured rather than inherited. This config was the one knob the stage
        shipped unswept - README §9 item 7 disclosed it as a real gap and review round 14 called that deferred
        work, correctly - and sweeping it (`logs/probe_prefill_sdpa.txt`) found the fused stage's 64 markedly
        slower than the 256 shipped here; work_log §4.19 reads the ladder, which is where a ratio belongs because
        a re-run moves it and a comment cannot be regenerated. The clamps below are contract, not tuning:
        `q_chunk` has to divide a non-zero resume offset, and neither chunk may exceed the physical length.

        The wide-cache clamp reads the **attached cache's own dtype**, not `policy.kv_cache_dtype`. Those two can
        differ: `allocate_kv_cache(dtype=...)` and `attach_kv_cache` set the real cache without touching the
        policy, which is a documented, supported route - the contract says a caller who wants the old cache can
        name it. Review round 15 keyed this on the policy field and round 16 pointed out that left the public API
        resolving 256 against a bfloat16 cache, which is the combination work_log §4.19 records throwing at
        program construction. `_cache_fill_tensor` already read the cache, so the two halves of the same
        question now answer it the same way.
        """
        cache_dtype = self.k_cache.dtype if self.k_cache is not None else self.policy.kv_cache_dtype
        qk = self.policy.prefill_sdpa_chunk
        if qk is None:
            qk = PREFILL_SDPA_CHUNK.get(self.policy.base_name, PREFILL_SDPA_CHUNK_DEFAULT)
        if DTYPE_BYTES.get(cache_dtype, 4.0) > DTYPE_BYTES[ttnn.bfloat8_b]:
            qk = min(qk, PREFILL_SDPA_CHUNK_WIDE_CACHE)
        if chunk_start_idx:
            qk = min(qk, chunk_start_idx & -chunk_start_idx)
        qk = min(qk, phys_len)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=qk,
            k_chunk_size=qk,
            exp_approx_mode=False,
        )

    def _attention_output(self, attn, gate):
        """Sigmoid output gate folded into the multiply, then the output projection.

        The gated tensor is the output projection's ``in0``, and in **decode** it names L1 rather than
        inheriting SDPA's DRAM output. `tt-perf-report` raises "If possible place input 0 in L1" on this
        row, and the stage took that advice everywhere else in decode; this row kept DRAM only because
        the multiply named no placement and its producer is the one decode op whose output must be in
        DRAM. `linear_attention`'s identically shaped `gdn_out` already ran from L1 for the same reason
        in reverse — its producer happens to be an L1 op — which is what made the gap visible to review
        round 26. Prefill keeps DRAM: there the same tensor is tens of megabytes.
        """
        gated = ttnn.multiply(
            attn,
            gate,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ATTN_OUT_IN0_MEMORY if self._decode_phase else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(gate)
        out = self._proj_linear(gated, self.w["o_proj"], "o_proj")
        ttnn.deallocate(gated)
        return out

    def _kv_update_memory_configs(self, batch_size):
        """``(k_cfg, v_cfg, fused)`` height-sharded L1 configs for the paged cache update.

        One user per core, as a row-wise core *range set* of exactly ``batch_size`` cores rather
        than a rectangular ``CoreGrid`` — a rectangle would restrict the batch to values with a
        factor pair fitting the grid in both axes (batch 13 has none on an 11×10 grid).

        ``paged_fused_update_cache`` writes both caches in one launch but requires the two inputs
        to live on **disjoint** cores, so K takes the first ``batch_size`` cores and V the next
        ``batch_size``. That needs ``2 * batch_size`` cores; above that the caller falls back to two
        separate ``paged_update_cache`` launches (``fused=False``), which is what the functional
        decoder always did.
        """
        grid = self.device.compute_with_storage_grid_size()
        cores = grid.x * grid.y
        if batch_size > cores:
            raise ValueError(
                f"decode batch {batch_size} exceeds the {cores} cores available for the "
                f"one-user-per-core paged_update_cache shard on a {grid.x}x{grid.y} grid"
            )

        def config(shard_grid):
            spec = ttnn.ShardSpec(shard_grid, [TILE, self.cfg.head_dim], ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)

        first = ttnn.num_cores_to_corerangeset(batch_size, grid, row_wise=True)
        if 2 * batch_size > cores:
            # Two separate launches, so there is no non-overlap rule to satisfy and both take the
            # natural first-`batch` range. This branch **is** reached - it needs `2 * batch > cores`, i.e.
            # batch > 55 on this grid, and `test_decode_batch_above_head_split_limit[56]` covers it. What
            # does not apply here is the V passthrough: the head split caps at
            # `DECODE_HEAD_SPLIT_MAX_BATCH`, so above 32 users V arrives interleaved from the generic
            # fallback and is resharded like K. Review round 28 wrote "unreachable" here while replacing a
            # different wrong comment, and round 29 caught it against the test that covers the branch.
            return config(first), config(first), False
        whole = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        second = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            ttnn.CoreCoord(batch_size % grid.x, batch_size // grid.x), batch_size, whole, True
        )
        # **V takes the first range, K the second.** Review round 27: V arrives from the head split
        # already sharded on the first `batch` cores, and `paged_fused_update_cache` only cares that the
        # two inputs are disjoint - so moving K is free and moving V would cost the reshard the
        # passthrough exists to remove. Before that round these were the other way round.
        return config(second), config(first), True

    def _rope_prefill(self, x, cos, sin):
        """Rotate ``[batch, heads, seq, head_dim]`` with cos/sin ``[1, 1, seq, width]``."""
        if self.rope.mode == "full":
            out = ttnn.experimental.rotary_embedding_hf(
                x, cos, sin, compute_kernel_config=self.rope_compute_kernel_config
            )
            ttnn.deallocate(x)
            return out
        rope_dim, head_dim = self.cfg.rope_dim, self.cfg.head_dim
        # `rope_dim == head_dim` (partial_rotary_factor 1.0) means there is nothing to pass through,
        # and the slice would alias `x` — the functional decoder guards the same way. Ornith is 0.25,
        # so this is about not narrowing the supported range, not about a shape reached here.
        source = _slice_last(x, 0, rope_dim) if rope_dim < head_dim else x
        rotated = ttnn.experimental.rotary_embedding_hf(
            source, cos, sin, compute_kernel_config=self.rope_compute_kernel_config
        )
        if rope_dim >= head_dim:
            ttnn.deallocate(x)
            return rotated
        ttnn.deallocate(source)
        passthrough = _slice_last(x, rope_dim, head_dim)
        ttnn.deallocate(x)
        out = ttnn.concat([rotated, passthrough], dim=-1)
        ttnn.deallocate(rotated)
        ttnn.deallocate(passthrough)
        return out

    def _rope_decode(self, x, cos, sin):
        """Rotate ``[1, batch, heads, head_dim]`` with per-user cos/sin ``[1, 1, batch, width]``.

        ``rotary_embedding_hf``'s native decode mode wants a height-sharded input, and the head split does
        produce one - but the Q/K head-dim RMSNorm sits between them, and `layernorm`'s sharded program
        config refuses a HEIGHT_SHARDED input *and* a HEIGHT_SHARDED output, so Q and K cannot reach the
        rope still height-sharded. Transposing the batch and head axes lets the interleaved prefill kernel
        do the same work with no reshard, which is how ``models/demos/blackhole/qwen36`` drives it too.
        Review rounds 27 and 28 assessed the native path: it is the layernorm line that blocks it, not the
        cos/sin layout this comment used to cite, and re-spelling it would trade the four transposes for a
        comparable number of shard conversions - a wash inside the harness band, before the partial-rotary
        width slice that would also have to be re-expressed.
        """
        rope_dim, head_dim = self.cfg.rope_dim, self.cfg.head_dim
        # As in _rope_prefill: a full-width rotation has no pass-through half to re-concatenate,
        # whether that is because the mode is "full" or because rope_dim already covers head_dim.
        partial = self.rope.mode == "partial" and rope_dim < head_dim
        source = _slice_last(x, 0, rope_dim) if partial else x
        swapped = ttnn.transpose(source, 1, 2)
        if partial:
            ttnn.deallocate(source)
        roped = ttnn.experimental.rotary_embedding_hf(
            swapped, cos, sin, compute_kernel_config=self.rope_compute_kernel_config
        )
        ttnn.deallocate(swapped)
        out = ttnn.transpose(roped, 1, 2)
        ttnn.deallocate(roped)
        if partial:
            passthrough = _slice_last(x, rope_dim, head_dim)
            ttnn.deallocate(x)
            merged = ttnn.concat([out, passthrough], dim=-1)
            ttnn.deallocate(out)
            ttnn.deallocate(passthrough)
            return merged
        ttnn.deallocate(x)
        return out

    #: Largest decode batch ``ttnn.experimental.nlp_create_qkv_heads_decode`` accepts
    #: (``nlp_create_qkv_heads_decode_device_operation.cpp``: ``num_users <= 32``). Above it the
    #: layer falls back to the functional decoder's generic split, so the supported batch is
    #: unchanged — only the op used to reach it is.
    DECODE_HEAD_SPLIT_MAX_BATCH = 32

    def _decode_qkv_heads(self, qkv, batch):
        """Split the packed ``[1, 1, batch, (n_heads + 2*n_kv) * head_dim]`` into decode-layout heads.

        Returns ``q`` ``[1, batch, n_heads, head_dim]`` and ``k``/``v``
        ``[1, batch, n_kv_heads, head_dim]``, DRAM-interleaved.

        The dedicated op is used whenever it can be — it does the whole split in one launch — and
        the functional decoder's generic slice/reshape/permute is kept as the fallback for batches
        past the op's hard ``num_users <= 32`` limit, so this stage narrows nothing.
        """
        cfg = self.cfg
        n_heads, n_kv, head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        if batch <= self.DECODE_HEAD_SPLIT_MAX_BATCH:
            # `paged_fused_update_cache` requires its two inputs on disjoint cores, and that comes
            # entirely from `_kv_update_memory_configs`: V takes the range the head split emits it on and
            # K is resharded onto the next one. Not from `overlap_qk_coregrid` - round 27's comment here
            # claimed it was, and round 28 found the claim inert. `nlp_create_qkv_heads_decode`'s wrapper
            # forces that flag to `True` whenever its input is not sharded, and this `qkv` is L1
            # *interleaved*, so passing `False` would have changed nothing. Q, K and V all come off the
            # split on one range; only K's write grid moves. That is the same defect class as §4.21, §4.22 and §4.23,
            # caught this time in a claim the stage had just written about its own change.
            heads = ttnn.experimental.nlp_create_qkv_heads_decode(qkv, num_heads=n_heads, num_kv_heads=n_kv)
            q, k, v = heads
            if DECODE_V_SHARD_PASSTHROUGH:
                # Q feeds the norm/rope chain and SDPA; K feeds the norm/rope chain. Only V reaches the
                # cache write untouched, so only V can keep the shard.
                return (
                    ttnn.sharded_to_interleaved(q, ttnn.DRAM_MEMORY_CONFIG),
                    ttnn.sharded_to_interleaved(k, ttnn.DRAM_MEMORY_CONFIG),
                    v,
                )
            return tuple(ttnn.sharded_to_interleaved(t, ttnn.DRAM_MEMORY_CONFIG) for t in heads)

        width = int(qkv.shape[-1])
        flat = ttnn.reshape(qkv, [batch, 1, width])
        out = []
        start = 0
        for heads in (n_heads, n_kv, n_kv):
            part = _slice_last(flat, start, start + heads * head_dim)
            start += heads * head_dim
            part = ttnn.to_layout(part, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            part = ttnn.reshape(part, [batch, 1, heads, head_dim])
            part = ttnn.to_layout(part, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # [batch, 1, heads, head_dim] -> [1, batch, heads, head_dim]: a relabel, seq extent is 1.
            out.append(ttnn.reshape(part, [1, batch, heads, head_dim]))
        ttnn.deallocate(flat)
        return tuple(out)

    def _attention_decode(self, x, current_pos, rot_idxs, page_table):
        if self.k_cache is None:
            raise RuntimeError("call allocate_kv_cache()/attach_kv_cache() before decode")
        if page_table is None:
            raise ValueError("full_attention decode requires a page_table")
        cfg = self.cfg
        b, folded = self._decode_rows(x)
        n_heads, n_kv, head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        q_width = n_heads * head_dim
        kv_width = 2 * n_kv * head_dim

        fused = self._proj_linear(x, self.w["attn_in"], "attn_in")
        width = int(fused.shape[-1])
        if not folded:
            # [b, 1, width] -> [1, 1, b, width], the layout `nlp_create_qkv_heads_decode` wants. Not a
            # metadata view — ttnn.reshape only returns one when the last dim matches and the
            # second-to-last dims are equal or both tile multiples, and 1 is neither — so this
            # dispatches a tiled reshape of the 9216-wide projection output that moves b from the
            # batch axis into the tile height. Under :data:`DECODE_COMPACT_ROWS` the activation was
            # folded before the projection ran, so the projection itself is 32 rows instead of
            # `32 * b` and this reshape does not exist at all.
            fused = ttnn.reshape(fused, [1, 1, b, width])
        qkv = _slice_last(fused, 0, q_width + kv_width)
        gate = _slice_last(fused, q_width + kv_width, width)
        ttnn.deallocate(fused)
        if not folded:
            # The gate is the output projection's other operand, so it has to match whichever
            # convention `_attention_output` will be handed `attn` in.
            gate = ttnn.reshape(gate, [b, 1, q_width])

        q, k, v = self._decode_qkv_heads(qkv, b)
        ttnn.deallocate(qkv)

        q = self._norm(q, self.w["q_norm"])
        k = self._norm(k, self.w["k_norm"])
        cos, sin = self.rope.decode_forward(rot_idxs)
        q = self._rope_decode(q, cos, sin)
        k = self._rope_decode(k, cos, sin)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

        # paged_fused_update_cache writes both caches in one launch. Both inputs must be
        # height-sharded [1, B, kv_heads padded to 32, head_dim].
        k_cfg, v_cfg, fused_update = self._kv_update_memory_configs(b)
        k_upd = ttnn.to_memory_config(k if DECODE_KV_PAD_FREE_WRITE else _pad_dim(k, 2, TILE - n_kv), k_cfg)
        # V is already in `v_cfg` when the head split handed it over sharded, so this is a no-op check
        # rather than a conversion. Compared rather than assumed: a grid or shard-shape mismatch has to
        # fall back to the rebuild, not be written to the cache in the wrong layout.
        v_passthrough = v.memory_config() == v_cfg
        v_upd = v if v_passthrough else ttnn.to_memory_config(_pad_dim(v, 2, TILE - n_kv), v_cfg)
        if fused_update:
            ttnn.experimental.paged_fused_update_cache(
                self.k_cache, k_upd, self.v_cache, v_upd, update_idxs_tensor=current_pos, page_table=page_table
            )
        else:
            ttnn.experimental.paged_update_cache(
                self.k_cache, k_upd, update_idxs_tensor=current_pos, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                self.v_cache, v_upd, update_idxs_tensor=current_pos, page_table=page_table
            )
        ttnn.deallocate(k_upd)
        if not v_passthrough:
            ttnn.deallocate(v_upd)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            is_causal=True,
            scale=head_dim**-0.5,
            program_config=self.decode_sdpa_config,
            # No compute_kernel_config, deliberately, and this is measured rather than inherited:
            # passing the prefill SDPA's HiFi2 + fp32-dest-accumulate config here collapses decode
            # PCC against the HF golden to 0.33 on a BFP8 paged cache, which is a correctness
            # failure rather than a precision drift. The op's own default is what the fused decoder
            # used and what every PCC number in this stage is measured with.
        )
        ttnn.deallocate(q)
        # [1, B, n_heads, D] -> [B, 1, n_heads*D], or the folded [1, 1, B, n_heads*D]: memory order is
        # already (B, head, dim), so both are the same logical flatten of the last two axes and the
        # folded one lands on 32 physical rows instead of `32 * B`.
        attn = ttnn.reshape(attn, [1, 1, b, n_heads * head_dim] if folded else [b, 1, n_heads * head_dim])
        return self._attention_output(attn, gate)

    # ------------------------------------------------------------------ gated deltanet
    def _gdn_project(self, x):
        """One packed in-projection, sliced into ``(qkv, z, a, b)``."""
        cfg = self.cfg
        nv = cfg.linear_num_value_heads
        fused = self._proj_linear(x, self.w["gdn_in"], "gdn_in")
        qkv_end = cfg.conv_dim
        z_end = qkv_end + cfg.linear_v_dim
        a_end = z_end + nv
        qkv = _slice_last(fused, 0, qkv_end)
        z = _slice_last(fused, qkv_end, z_end)
        a = _slice_last(fused, z_end, a_end)
        b = _slice_last(fused, a_end, a_end + nv)
        ttnn.deallocate(fused)
        return qkv, z, a, b

    def _gdn_gates(self, a, b_raw, logical_len, seq_len):
        """``(beta, g)`` as float32 ``[B, T, num_v_heads]``, with the padded tail neutralised.

        ``beta = sigmoid(in_proj_b(x))`` and ``g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)``
        (float32). ``softplus`` is folded into the bias add. Zeroing both past ``logical_len``
        makes every padded step an exact identity on the recurrent state.
        """
        beta = ttnn.typecast(ttnn.sigmoid(b_raw), ttnn.float32)
        a32 = ttnn.typecast(a, ttnn.float32)
        softplus = ttnn.add(a32, self.w["dt_bias"], activations=_SOFTPLUS)
        ttnn.deallocate(a32)
        g = ttnn.multiply(self.w["A_neg"], softplus)
        ttnn.deallocate(softplus)

        if logical_len < seq_len:
            # _slice_owned, not ttnn.slice: pos_ramp is [1, prefill_chunk, 1], so at
            # seq_len == prefill_chunk this is a whole-tensor slice, which ttnn.slice short-circuits
            # to the input itself (slice.cpp's `no_step && starts_zero && ends_max` no-op check).
            # Deallocating that would free the layer's persistent ramp weight and fatal the *next*
            # masked prefill. Reachable whenever a chunk's logical length lands in
            # (chunk_size - PREFILL_ALIGN, chunk_size), e.g. seq_len=2000 at the shipped 2048.
            ramp, ramp_owned = _slice_owned(self.w["pos_ramp"], [0, 0, 0], [1, seq_len, 1])
            keep = ttnn.typecast(ttnn.lt(ramp, float(logical_len)), ttnn.float32)
            if ramp_owned:
                ttnn.deallocate(ramp)
            beta_masked = ttnn.multiply(beta, keep)
            g_masked = ttnn.multiply(g, keep)
            ttnn.deallocate(keep)
            ttnn.deallocate(beta)
            ttnn.deallocate(g)
            beta, g = beta_masked, g_masked
        return beta, g

    def _conv1d_halves(self, padded_rm, phys_len, prepared):
        """Depthwise conv1d over the padded stream, one call per :data:`CONV1D_CHANNELS` block.

        Returns the SiLU-activated halves as a **list**, one per :data:`CONV1D_CHANNELS` block,
        *not* a concatenated stream: at 4096 channels the split point is exactly the Q/K | V
        boundary, so ``_gdn_prefill`` consumes ``[q|k]`` and ``[v]`` directly and neither a concat
        nor a 4096-wide re-slice is needed. The SiLU is applied separately rather than through
        ``Conv2dConfig(activation=...)``: folding it is faster but not correct on this depthwise
        conv. Measured at Ornith's own shapes in ``doc/fused_decoder/logs/probe_conv1d_and_norm.txt``
        (``CONV1DACT`` rows) and recorded in ``doc/fused_decoder/work_log.md`` §4.15;
        ``models/demos/blackhole/qwen36/tt/gdn/tp.py:367`` reports the same for its conv.
        """
        rm = ttnn.DRAM_MEMORY_CONFIG
        channels = CONV1D_CHANNELS
        batch = int(padded_rm.shape[0])
        length = int(padded_rm.shape[1])
        cfg = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)

        outs = []
        for idx, weight in enumerate(prepared):
            part = _slice_last(padded_rm, idx * channels, (idx + 1) * channels)
            part = ttnn.reshape(part, [batch, length, 1, channels])
            out = ttnn.conv1d(
                input_tensor=part,
                weight_tensor=weight,
                device=self.device,
                in_channels=channels,
                out_channels=channels,
                batch_size=batch,
                input_length=length,
                kernel_size=self.cfg.linear_conv_kernel_dim,
                stride=1,
                padding=0,
                dilation=1,
                groups=channels,
                dtype=ttnn.bfloat16,
                conv_config=cfg,
                compute_config=self.conv_compute_kernel_config,
                # L1-full slicing: the DRAM-slice path does host reads, which a later traced caller
                # could not use. Prefill is not traced today, but the constraint costs nothing here.
                slice_config=ttnn.Conv2dL1FullSliceConfig,
                return_output_dim=False,
                return_weights_and_bias=False,
            )
            ttnn.deallocate(part)
            out = ttnn.sharded_to_interleaved(out, rm)
            out = ttnn.to_layout(ttnn.reshape(out, [batch, phys_len, channels]), ttnn.TILE_LAYOUT, memory_config=rm)
            activated = ttnn.silu(out, memory_config=rm)
            ttnn.deallocate(out)
            outs.append(activated)

        return outs

    def _conv1d_weights(self, phys_len):
        """Prepared ``ttnn.conv1d`` weights for a block of ``phys_len`` tokens, or ``None``.

        ``ttnn.prepare_conv_weights`` is a host call whose output layout depends on the input
        length, so the weights for every physical block length the layer can produce are prepared at
        load time (``from_state_dict``) and looked up here. A caller that asks for a ``chunk_size``
        the layer was not built for simply gets ``None`` and the FIR path, rather than a host call
        inside a measured forward pass.
        """
        return self.w.get("conv1d_weights", {}).get(int(phys_len))

    def _causal_conv_prefill(self, qkv, logical_len):
        """Depthwise causal conv1d + SiLU over the fused QKV stream.

        Returns ``(activated, tail)``. ``activated`` is the whole ``[batch, phys_len, conv_dim]``
        stream when the FIR path runs, or the list of :data:`CONV1D_CHANNELS`-wide halves when
        ``ttnn.conv1d`` does; :meth:`_gdn_split_conv_output` consumes either. ``tail`` is the
        ``kernel-1`` real inputs ending at ``logical_len`` — the conv history the next block or the
        next decode step starts from.

        The FIR fallback takes its shifted taps from one ROW_MAJOR concatenation rather than
        untilizing the whole stream per tap. It accumulates with :func:`ttnn.addcmul`, as the
        functional decoder did: this stage briefly used :func:`ttnn.mac` instead, on the belief that
        ``addcmul`` was a three-op composite. It is not — ``ternary.cpp`` dispatches a single
        ``prim::ternary(ADDCMUL)`` LLK whenever the broadcast is valid and no input is block-float,
        which holds for both of these call sites, while ``mac`` is unconditionally
        ``add(multiply(...))``. The swap cost one extra device op per tap on the *decode* conv, which
        is the traced path, so it was reverted. ``doc/fused_decoder/work_log.md`` §4.14.
        """
        t = int(qkv.shape[1])
        kernel = self.cfg.linear_conv_kernel_dim
        # Concatenate in ROW_MAJOR: `ttnn.concat` on the token axis of TILE tensors lowers to
        # untilize -> concat -> tilize, and that tilize of the whole 8192-wide stream would be thrown
        # away immediately by the shifted-window slicing below.
        rm = ttnn.DRAM_MEMORY_CONFIG
        pieces = []
        for buf in self.conv_state:
            piece = ttnn.to_layout(buf, ttnn.ROW_MAJOR_LAYOUT, memory_config=rm)
            if len(piece.shape) == 4:
                # The history is stored folded (`[1, 1, batch, conv_dim]`,
                # :data:`DECODE_COMPACT_ROWS`); the concat below is on the *token* axis of
                # `[batch, t, conv_dim]`, so it is unfolded back here. ROW_MAJOR with the last dim
                # unchanged, so this is a view rather than a relayout — prefill pays nothing for the
                # decode step's fold.
                unfolded = ttnn.reshape(piece, [int(piece.shape[-2]), 1, int(piece.shape[-1])])
                _free_unless_aliased(piece, unfolded)
                piece = unfolded
            pieces.append(piece)
        qkv_rm = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=rm)
        padded_rm = ttnn.concat(pieces + [qkv_rm], dim=1)
        for piece in pieces:
            ttnn.deallocate(piece)
        ttnn.deallocate(qkv_rm)

        prepared = self._conv1d_weights(t)
        if prepared is not None:
            # A list of CONV1D_CHANNELS-wide halves; _gdn_prefill splits Q/K/V out of them without
            # ever materialising the concatenated stream.
            activated = self._conv1d_halves(padded_rm, t, prepared)
        else:
            acc = None
            for tap in range(kernel):
                if tap == kernel - 1:
                    # The last tap's window is exactly the (already TILE) input: no slice, no tilize.
                    piece, owned = qkv, False
                else:
                    piece, owned = (
                        ttnn.to_layout(padded_rm[:, tap : tap + t, :], ttnn.TILE_LAYOUT, memory_config=rm),
                        True,
                    )
                if acc is None:
                    acc = ttnn.multiply(piece, self.w["conv_taps"][tap], memory_config=rm)
                else:
                    acc = ttnn.addcmul(acc, piece, self.w["conv_taps"][tap], memory_config=rm)
                if owned:
                    ttnn.deallocate(piece)
            activated = ttnn.silu(acc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(acc)

        # The tail stays ROW_MAJOR: _write_conv_state wants one row at a time and rows 0..kernel-2
        # are not tile-aligned, so tilizing the whole tail here would only be undone immediately.
        # The alternative — tilize the whole tail once here and row-slice it in TILE — is timed
        # against this one on the real decoder at the 2048-token prefill shape by
        # doc/fused_decoder/logs/probe_conv_tail.py, which monkeypatches that variant in. The two
        # produce identical output (`torch.equal` in the probe); this one additionally dispatches one
        # layout conversion fewer, which
        # test_no_layout_churn_in_measured_forward pins at 256 and 2048 tokens. The slice is a
        # strict sub-range, so it owns a fresh buffer and `padded_rm` can be freed.
        tail = padded_rm[:, logical_len : logical_len + kernel - 1, :]
        ttnn.deallocate(padded_rm)
        return activated, tail

    def _gdn_split_conv_output(self, activated):
        """Split the causal-conv output into ``(q, k, v)``, consuming ``activated``.

        ``_causal_conv_prefill`` hands back either one ``[B, T, conv_dim]`` stream (the FIR
        fallback) or the list of :data:`CONV1D_CHANNELS`-wide blocks ``ttnn.conv1d`` produced. In the
        list case the blocks are *not* concatenated first: a block that lies entirely inside one of
        q/k/v is handed over as-is, so on Ornith — ``conv_dim`` 8192 split 2048/2048/4096, blocks
        4096 wide — v is exactly block 1 and needs no slice at all. That removes both the
        8192-wide concat of the conv output and the 4096-wide v slice the concatenated spelling
        needed; doc/fused_decoder/work_log.md §4.4 has the ConcatDeviceOperation row it cost.

        The block/field boundaries are only guaranteed to line up that neatly for this config, so the
        general case is still handled: a field spanning two blocks is concatenated from its pieces.
        """
        cfg = self.cfg
        bounds = [0, cfg.linear_q_dim, cfg.linear_q_dim + cfg.linear_k_dim, cfg.conv_dim]
        if not isinstance(activated, list):
            fields = [_slice_last(activated, bounds[i], bounds[i + 1]) for i in range(3)]
            ttnn.deallocate(activated)
            return fields

        channels = CONV1D_CHANNELS
        consumed = [False] * len(activated)
        fields = []
        for i in range(3):
            lo, hi = bounds[i], bounds[i + 1]
            pieces = []
            for blk in range(lo // channels, (hi - 1) // channels + 1):
                base = blk * channels
                start, end = max(lo, base) - base, min(hi, base + channels) - base
                if (start, end) == (0, channels):
                    pieces.append(activated[blk])
                    consumed[blk] = True
                else:
                    pieces.append(_slice_last(activated[blk], start, end))
            if len(pieces) == 1:
                fields.append(pieces[0])
            else:
                fields.append(ttnn.concat(pieces, dim=-1))
                # Every piece is now a copy inside the concat output, including any whole block
                # (already marked consumed, so the sweep below will not double-free it).
                for piece in pieces:
                    ttnn.deallocate(piece)
        for blk, block in enumerate(activated):
            if not consumed[blk]:
                ttnn.deallocate(block)
        return fields

    def _write_conv_state(self, tail_rm):
        """Copy the new conv history into the persistent buffers, preserving addresses.

        ``tail_rm`` is ROW_MAJOR ``[batch, kernel-1, conv_dim]``; each row is tilized on its own.

        The buffers may be folded (``[1, 1, batch, conv_dim]``, :data:`DECODE_COMPACT_ROWS`), in
        which case each row is folded *before* it is tilized. That reshape is ROW_MAJOR with the last
        dim unchanged, so ttnn hands back a view and no op is dispatched — the same `to_layout` count
        this method always had, which is what ``test_no_layout_churn_in_measured_forward`` budgets.
        """
        for idx, buf in enumerate(self.conv_state):
            row = tail_rm[:, idx : idx + 1, :]
            if len(buf.shape) == 4:
                row = ttnn.reshape(row, [1, 1, int(row.shape[0]), int(row.shape[-1])])
            tiled = ttnn.to_layout(row, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.copy(tiled, buf)
            ttnn.deallocate(tiled)

    def _chunk_delta_rule(self, q, k, v, g, beta):
        """Chunk-parallel gated delta rule over a whole block, sub-batched if needed.

        ``q``/``k``/``v`` are **flat** rank-3 token-major tensors. That contract lets the op's prep
        kernel L2-normalise Q/K over the head dim and fold the ``K**-0.5`` scale into that norm, so
        the three head-split relayouts, the two explicit device-side L2 norms and the scale multiply all
        disappear into the op's prep kernel.
        It requires ``chunk_size == 32`` and ``T % 32 == 0``, both guaranteed by the 128-token
        physical prefill alignment.

        Two couplings the rank-4 spelling did not have, both documented in
        ``doc/fused_decoder/README.md`` §8:

        * the op infers the key head dim **from the value head dim** on this path
          (``chunk_gated_delta_rule.cpp``: ``K = flat_qk ? V : qs[3]``), so an unequal pair is guarded
          below rather than silently miscomputed;
        * the flat path is only legal on the op's phased branch, which that file selects from the
          ``QWEN_GDN_PHASED`` environment variable. With it set to ``0`` this call raises where the
          functional decoder's rank-4 call still runs.
        """
        # See the docstring: guarded rather than assumed. _gdn_decode raises the same way.
        dk, dv = self.cfg.linear_key_head_dim, self.cfg.linear_value_head_dim
        if dk != dv:
            raise ValueError(
                "the flat rank-3 chunk_gated_delta_rule contract infers the key head dim from the "
                f"value head dim, which assumes linear_key_head_dim == linear_value_head_dim; got "
                f"{dk} != {dv}. Use the functional decoder's rank-4 call for such a config."
            )
        cfg = self.cfg
        eye, tril, ones, masks = self.w["gdn_const_tiles"]
        batch, seq = q.shape[0], q.shape[1]
        nv = cfg.linear_num_value_heads

        def launch(q_, k_, v_, g_, beta_, state_):
            return ttnn.transformer.chunk_gated_delta_rule(
                q_,
                k_,
                v_,
                g_,
                beta_,
                initial_state=state_,
                output_final_state=True,
                chunk_size=GDN_CHUNK,
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
            q_s = ttnn.slice(q, [start, 0, 0], [end, seq, cfg.linear_q_dim])
            k_s = ttnn.slice(k, [start, 0, 0], [end, seq, cfg.linear_k_dim])
            v_s = ttnn.slice(v, [start, 0, 0], [end, seq, cfg.linear_v_dim])
            g_s = ttnn.slice(g, [start, 0, 0], [end, seq, nv])
            beta_s = ttnn.slice(beta, [start, 0, 0], [end, seq, nv])
            state_s = ttnn.slice(
                self.recurrent_state, [start, 0, 0, 0], [end, nv, cfg.linear_key_head_dim, cfg.linear_value_head_dim]
            )
            core_s, final_s = launch(q_s, k_s, v_s, g_s, beta_s, state_s)
            for tensor in (q_s, k_s, v_s, g_s, beta_s, state_s):
                ttnn.deallocate(tensor)
            cores.append(core_s)
            states.append(final_s)
        # head-major output is [B*HV, T, V]; concatenating on dim 0 restitches the batch.
        core = ttnn.concat(cores, dim=0)
        final_state = ttnn.concat(states, dim=0)
        for tensor in cores + states:
            ttnn.deallocate(tensor)
        return core, final_state

    def _gdn_out(self, core_head_major, z, batch, seq):
        """Gated output norm + out projection.

        ``core_head_major``: ``[B*num_v_heads, T, head_v_dim]`` (prefill) — the op's head-major
        output — or ``[B, num_v_heads, 1, head_v_dim]`` (decode). ``ttnn.rms_norm`` normalises over
        the last dim either way, and ``nlp_concat_heads`` does the head→token relayout in one op.
        """
        cfg = self.cfg
        nv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
        normed = ttnn.rms_norm(core_head_major, weight=self.w["gdn_norm"], epsilon=cfg.norm_eps)
        ttnn.deallocate(core_head_major)
        normed = ttnn.reshape(normed, [batch, nv, seq, dv])
        # The head->token relayout lands in `z`'s convention, because `z` is the gate's other
        # operand: `[batch, seq, nv * dv]` in prefill and in an unfolded decode step, and the folded
        # `[1, 1, batch, nv * dv]` under :data:`DECODE_COMPACT_ROWS`. Both are the same logical
        # flatten of the permuted tensor's last two axes, and taking the shape from `z` rather than
        # from a flag keeps the two operands of that multiply impossible to mismatch.
        # ttnn.Shape indexes by int only - it has no slice overload - so walk it by rank.
        merged_shape = [*(int(z.shape[i]) for i in range(len(z.shape) - 1)), nv * dv]
        if seq > 1:
            merged = ttnn.experimental.nlp_concat_heads(normed)
            ttnn.deallocate(normed)
            merged = ttnn.reshape(merged, merged_shape)
        else:
            # All three spellings of this head->token relayout were measured from a captured trace
            # (doc/fused_decoder/logs/probe_decode_micro.txt). nlp_concat_heads wins by more than an
            # order of magnitude for a prefill block but collapses onto one core at seq 1, where the
            # plain permute is several times cheaper. All three agree exactly at seq 1
            # (`torch.equal`); above it
            # only nlp_concat_heads and permute+reshape are equivalent (the flat relayout does not
            # transpose head<->token), which the same probe records.
            swapped = ttnn.permute(normed, (0, 2, 1, 3))
            ttnn.deallocate(normed)
            merged = ttnn.reshape(swapped, merged_shape)
            ttnn.deallocate(swapped)
        # Deliberately unfused, and the reason is the DTYPES, not the magnitudes.
        # `models/demos/blackhole/qwen36/tt/gdn/tp.py:31-34` reports that folding the SiLU here
        # "overflows to NaN in the real layer for large-magnitude z (op-level PCC hid it - small
        # inputs)". The NaN is real; the attribution to magnitude is not what reproduces here.
        # This gate is a MIXED-DTYPE binary: `merged` is FLOAT32 (chunk_gated_delta_rule's output
        # dtype) and `z` is BFLOAT16. The GATEFOLD rows in doc/fused_decoder/logs/probe_fused_ops.txt
        # run both pairings at this gate's own prefill and decode shapes: with matching bfloat16
        # operands the folded and separate forms agree to PCC 0.999996 with zero non-finite outputs
        # at every |z| tested, and with the real float32 x bfloat16 pairing the folded form emits
        # non-finite values at EVERY magnitude, including |z| < 4. So an op-level A/B on matched
        # dtypes - which is what a naive probe writes - passes while the model breaks.
        # doc/fused_decoder/work_log.md §4.8 records both arms and the real-weight control.
        # `ttnn.silu(z)` first keeps the activation in bfloat16 and the multiply mixed-but-unfused,
        # which is exact.
        # bfloat16 out, not the float32 the mixed-dtype multiply would default to: the only consumer
        # is the output projection, whose weight is BFP8, and a float32 activation there costs both
        # the multiply's write and the matmul's read. Measured on the shipped path in
        # doc/optimized_decoder/logs/ab_gdn_out_activation.txt: the `gdn_out` row falls from the
        # FP32 x BFP8 form to the BF16 x BFP8 one that `o_proj` already runs.
        gated = ttnn.multiply(merged, ttnn.silu(z), dtype=ttnn.bfloat16)
        ttnn.deallocate(merged)
        ttnn.deallocate(z)
        out = self._proj_linear(gated, self.w["gdn_out"], "gdn_out")
        ttnn.deallocate(gated)
        return out

    def _gdn_prefill(self, x, logical_len):
        b, t = x.shape[0], x.shape[1]
        qkv, z, a, b_raw = self._gdn_project(x)
        activated, tail = self._causal_conv_prefill(qkv, logical_len)
        ttnn.deallocate(qkv)

        q, k, v = self._gdn_split_conv_output(activated)

        beta, g = self._gdn_gates(a, b_raw, logical_len, t)
        ttnn.deallocate(a)
        ttnn.deallocate(b_raw)

        core, final_state = self._chunk_delta_rule(q, k, v, g, beta)
        for tensor in (q, k, v, g, beta):
            ttnn.deallocate(tensor)
        ttnn.copy(final_state, self.recurrent_state)
        ttnn.deallocate(final_state)
        self._write_conv_state(tail)
        ttnn.deallocate(tail)
        return self._gdn_out(core, z, b, t)

    def _gdn_decode(self, x):
        """One recurrent gated-delta-rule step, updating conv + recurrent state in place.

        ``x`` follows :meth:`_block`'s residual convention, and so does everything derived from it:
        the packed projection, the four tap ops below and the persistent conv history are all
        ``[1, 1, batch, ...]`` under :data:`DECODE_COMPACT_ROWS` and ``[batch, 1, ...]`` without it.
        :meth:`allocate_state` allocates the history in the same convention, which is what keeps
        ``ttnn.copy`` and ``ttnn.addcmul`` operand-shaped here; a caller that reached this method
        with the other convention would meet a shape error at the first tap, not a wrong answer.
        """
        cfg = self.cfg
        b, _ = self._decode_rows(x)  # every shape below follows `x`'s own convention
        nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        kernel = cfg.linear_conv_kernel_dim

        qkv, z, a, b_raw = self._gdn_project(x)

        acc = ttnn.multiply(qkv, self.w["conv_taps"][kernel - 1])
        for tap in range(kernel - 1):
            acc = ttnn.addcmul(acc, self.conv_state[tap], self.w["conv_taps"][tap])
        activated = ttnn.silu(acc)
        ttnn.deallocate(acc)
        # Shift the history: oldest out, this step's pre-activation input in. Read-before-write
        # order keeps the in-place chain correct and the buffer addresses stable.
        for idx in range(kernel - 2):
            ttnn.copy(self.conv_state[idx + 1], self.conv_state[idx])
        ttnn.copy(qkv, self.conv_state[kernel - 2])
        ttnn.deallocate(qkv)

        # One relayout for the whole conv output into head-major [B, heads, 1, dim] — the shape both
        # the recurrent step and the output norm want. It is a reshape + permute rather than an
        # explicit ROW_MAJOR round trip; the reshape splits the last dim, so ttnn dispatches a tiled
        # reshape kernel rather than returning a view, but that is one device op against the
        # functional decoder's untilize/reshape/tilize. It is not a layout
        # conversion. Slicing heads off dim 1 is then a cheap batch-dim slice, where the functional
        # decoder needed three separate untilize/reshape/tilize round trips (one per Q/K/V).
        if dk != dv:
            raise ValueError(
                f"the decode head split folds Q/K/V into one {2 * nk + nv}-head relayout, which "
                f"assumes linear_key_head_dim == linear_value_head_dim; got {dk} != {dv}"
            )
        # Where the head-major tensors live is *decided* here, not inherited, and that is not a
        # style preference. The permute below moves the head axis to dim 1, which leaves a token
        # axis of 1 that pads to a whole tile per (batch, head): `heads` is `batch * heads` tile
        # rows — 16 MiB at batch 32 — out of a `rows` tensor that is 512 KiB, and q/k/v are half
        # that each. Under :data:`DECODE_COMPACT_ROWS` the conv output they come from is one tile
        # row at every batch, so `_proj_linear` gives it an L1 memory config at every batch and all
        # of this inherited L1; unfolded the same inheritance reached DRAM above
        # :data:`DECODE_MATMUL_MAX_M_TILES` because the projection's own row count fell out of
        # `_ProjectionConfigs`' band. Nothing about the arithmetic changes either way and no PCC bar
        # can see it, but ~28 MiB of extra L1 held live across the recurrent step at batch 32 leaves
        # ttnn too little free worker L1 to *build* a legal program config for `_state_matmul`'s
        # `core_grid` fallback: `create_simple_matmul_program_config` sizes `per_core_M` from
        # `get_max_l1_space()`, lands on 2, and then its own `per_core_M % out_subblock_h` check
        # rejects the pair it built (`matmul_program_config.cpp:1291` hardcodes `out_subblock_h = 4`
        # and only resets it when `out_subblock_w != per_core_N`). So size-gate the relayout the way
        # :meth:`_delta_rule_step` already size-gates the delta outer product, against the float32
        # working set it will be sharing that L1 with (:data:`_STATE_STEP_L1_VECTORS`).
        step_l1_bytes = _STATE_STEP_L1_VECTORS * b * nv * TILE * dk * 4
        head_mem = ttnn.L1_MEMORY_CONFIG if step_l1_bytes <= self._l1_budget else ttnn.DRAM_MEMORY_CONFIG
        # Unchanged by the fold: `[batch, 1, conv_dim]` and `[1, 1, batch, conv_dim]` enumerate the
        # same elements in the same order, so this is the same logical reshape either way. No
        # `memory_config` on the reshape: the head axis is still dim -2 here, so `rows` is two tile
        # rows per batch entry rather than 32, and forcing a placement on it could cost a copy where
        # ttnn would otherwise hand back a view.
        rows = ttnn.reshape(activated, [b, 1, 2 * nk + nv, dk])
        ttnn.deallocate(activated)
        heads = ttnn.permute(rows, (0, 2, 1, 3), memory_config=head_mem)
        ttnn.deallocate(rows)
        v = ttnn.slice(heads, [0, 2 * nk, 0, 0], [b, 2 * nk + nv, 1, dv], memory_config=head_mem)

        # Q and K are adjacent on the head axis, so one repeat_interleave over the pair does both
        # GQA expansions: [q0..q15, k0..k15] -> [q0,q0,...,q15,q15, k0,k0,...,k15,k15]. That halves
        # the untilize/concat/tilize this op lowers to.
        repeats = nv // nk
        qk = ttnn.slice(heads, [0, 0, 0, 0], [b, 2 * nk, 1, dk], memory_config=head_mem)
        ttnn.deallocate(heads)
        if repeats > 1:
            # L1 wherever the relayout is in L1, not the op's default: `repeat_interleave` lowers to
            # untilize -> concat -> tilize, and the profiler shows the op-to-op stall in the traced replay
            # before that tilize as the single largest gap in the linear decode window. README §7's
            # generated itemisation carries its size for the shipped L1 path; the DRAM intermediate this
            # replaced was worse. It is `head_mem` rather than a flat `L1_MEMORY_CONFIG` because after the
            # GQA expansion this tensor is `2 * num_v_heads` heads wide - as wide as `heads` itself, 16 MiB
            # at batch 32 - so exempting it from the gate above would largely defeat the gate. The batches
            # that give the L1 spelling up are the ones the gate has already found do not fit, and they are
            # the supported-but-untuned band README §9 item 5 records.
            expanded = ttnn.repeat_interleave(qk, repeats, dim=1, memory_config=head_mem)
            ttnn.deallocate(qk)
            qk = expanded
        q = ttnn.slice(qk, [0, 0, 0, 0], [b, nv, 1, dk], memory_config=head_mem)
        k = ttnn.slice(qk, [0, nv, 0, 0], [b, 2 * nv, 1, dk], memory_config=head_mem)
        ttnn.deallocate(qk)

        beta, g = self._gdn_gates(a, b_raw, 1, 1)
        ttnn.deallocate(a)
        ttnn.deallocate(b_raw)
        core = self._delta_rule_step(q, k, v, beta, g)
        for tensor in (q, k, v, beta, g):
            ttnn.deallocate(tensor)
        return self._gdn_out(core, z, b, 1)

    def _state_matmul_config(self, role: str, batch: int):
        """``MatmulMultiCoreReuseProgramConfig`` for one recurrent-state matmul, or ``None``.

        These three are the ``linear_attention`` decode step's float32 matmuls, and
        ``tt-perf-report`` flags each of them ``SLOW`` with ``in0_block_w=1 is small``. The batched
        non-mcast family does take an ``in0_block_w`` — the sibling ``core_grid`` spelling the fused
        stage used does not expose one — so the advice is actionable after all, which is what review
        round 2 found:

        * ``read`` (``q @ h`` and ``k @ h``): ``Mt`` 1, ``Kt`` 4, ``Nt`` 4. ``in0_block_w`` 2 is the
          measured winner, a little under a microsecond ahead of the ``core_grid`` spelling and of
          ``in0_block_w`` 1 and 4.
        * ``outer`` (``k^T @ delta``, ``transpose_a=True``): ``Mt`` 4, ``Kt`` **1**, ``Nt`` 4, so 1 is
          the only legal ``in0_block_w`` — 2 and 4 are rejected by the op. This is the larger win of
          the two, roughly a third off the ``core_grid`` spelling.

        The op parallelises over ``batch * M-blocks * N-blocks`` and that product must fit the worker
        grid. With ``num_value_heads`` 32 blocks per batch row that means **decode batch 3 or less** on
        an 11x10 grid; above it the config is dropped and the ``core_grid`` spelling the fused decoder
        used is what runs. Batch 1 is the tuned single-user target and larger batches stay correct
        (tested to 56) but untuned, which README §9 item 5 records alongside the other two thresholds. ``doc/optimized_decoder/logs/probe_decode_micro.txt`` (``STATE``
        rows) has both arms.
        """
        cfg = self.cfg
        dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        m_tiles = 1 if role == "read" else dk // TILE
        k_tiles = dk // TILE if role == "read" else 1
        n_tiles = dv // TILE
        grid = self.device.compute_with_storage_grid_size()
        blocks = batch * cfg.linear_num_value_heads
        if blocks > grid.x * grid.y or dk % TILE or dv % TILE:
            return None
        key = (role, batch)
        if key not in self._state_cfg_cache:
            self._state_cfg_cache[key] = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                in0_block_w=_largest_divisor_at_most(k_tiles, 2),
                out_subblock_h=1,
                out_subblock_w=_largest_divisor_at_most(n_tiles, 4),
                per_core_M=m_tiles,
                per_core_N=n_tiles,
            )
        return self._state_cfg_cache[key]

    def _state_matmul(self, a, b, role: str, batch: int, *, transpose_a: bool = False, memory_config=None):
        """One recurrent-state matmul under its tuned program config, or the ``core_grid`` fallback.

        The fallback is not shape-determined: ttnn sizes it from the worker L1 that happens to be
        *free* when the op is built (``get_max_l1_space()`` -> ``get_per_core_factor()``), and one of
        the values it can land on — ``per_core_M`` 2 — makes it reject its own program config with
        "per_core_M must be divisible by out_subblock_h"
        (``matmul_program_config.cpp:1291`` hardcodes ``out_subblock_h = 4`` and only resets it when
        ``out_subblock_w != per_core_N``). So how much L1 the decode step is holding when it gets
        here is part of this call's contract, which is what :meth:`_gdn_decode`'s ``head_mem`` gate
        exists to bound. Nothing here can repair it locally: the tuned config is dropped because the
        op's ``batch * M-blocks * N-blocks`` no longer fits the worker grid, not because the operands
        changed shape.
        """
        program_config = self._state_matmul_config(role, batch)
        return ttnn.matmul(
            a,
            b,
            transpose_a=transpose_a,
            memory_config=memory_config,
            compute_kernel_config=self.state_compute_kernel_config,
            program_config=program_config,
            **({} if program_config is not None else {"core_grid": self.full_core_grid}),
        )

    def _delta_rule_step(self, q, k, v, beta, g):
        """Single gated-delta-rule step on the persistent recurrent state.

        ``q``/``k``: ``[B, num_v_heads, 1, head_k_dim]`` (already GVA-expanded), ``v``:
        ``[B, num_v_heads, 1, head_v_dim]``, ``beta``/``g``: ``[B, 1, num_v_heads]`` float32.

        Mirrors HF ``torch_recurrent_gated_delta_rule``: L2-normalise Q/K, scale Q by
        ``head_k_dim ** -0.5``, decay the state, read ``k @ h``, write the ``beta``-weighted delta
        outer product, then read ``q @ h``. The state stays float32 in DRAM so the carry between steps is exact; the per-step vectors
        live in L1 (see the placement comment in the body).

        The L2 norm is ``rms_norm(x, eps/K) * K**-0.5`` (the idiom from
        ``models/experimental/gated_attention_gated_deltanet``), and Q's ``K**-0.5`` scale is folded
        into that same multiply, so Q costs two ops rather than four.
        """
        cfg = self.cfg
        b = q.shape[0]
        nv, dk = cfg.linear_num_value_heads, cfg.linear_key_head_dim
        # The per-step *vectors* are small ([B, HV, 1, D] float32 = 16 KB at batch 1) and every one
        # of them is consumed immediately, so they live in L1; `tt-perf-report` asks for exactly this
        # on the three state matmul rows ("place input 0 in L1"), and it is the larger of that A/B's two
        # wins (`doc/optimized_decoder/logs/ab_state_l1.txt` has both arms).
        # `self.recurrent_state` stays in DRAM: it is the persistent per-batch carry that trace
        # replay writes in place, and its address has to survive the whole capture.
        step_mem = ttnn.L1_MEMORY_CONFIG
        # The delta outer product is *state-shaped*, not vector-shaped: [B, HV, DK, DV] float32 is
        # 2 MiB at batch 1 but 64 MiB at batch 32, which L1 refuses outright
        # (`bank_manager.cpp:462`). Size-gate it rather than assume the batch.
        outer_bytes = b * nv * dk * cfg.linear_value_head_dim * 4
        outer_mem = step_mem if outer_bytes <= self._l1_budget else ttnn.DRAM_MEMORY_CONFIG
        eps = 1e-6

        # HF's torch_recurrent_gated_delta_rule applies `scale = 1/sqrt(head_k_dim)` to q. The
        # layer PCC cannot discriminate it (Qwen3_5MoeRMSNormGated cancels any uniform per-(token,
        # head) factor), but the chunked prefill op applies the same scale internally, so dropping
        # it would make prefill and decode disagree on the intermediate `o`.
        q_n = ttnn.rms_norm(q, epsilon=eps / dk)
        q_row = ttnn.multiply(q_n, dk**-1.0, memory_config=step_mem, dtype=ttnn.float32)
        ttnn.deallocate(q_n)
        k_n = ttnn.rms_norm(k, epsilon=eps / dk)
        k_row = ttnn.multiply(k_n, dk**-0.5, memory_config=step_mem, dtype=ttnn.float32)
        ttnn.deallocate(k_n)

        v_row = ttnn.typecast(v, ttnn.float32)
        beta_b = ttnn.reshape(beta, [b, nv, 1, 1])
        decay = ttnn.exp(ttnn.reshape(g, [b, nv, 1, 1]), memory_config=step_mem)

        state = self.recurrent_state
        ttnn.multiply(state, decay, output_tensor=state)
        ttnn.deallocate(decay)

        v_read = self._state_matmul(k_row, state, "read", b, memory_config=step_mem)
        delta = ttnn.multiply(ttnn.subtract(v_row, v_read, memory_config=step_mem), beta_b, memory_config=step_mem)
        ttnn.deallocate(v_read)
        ttnn.deallocate(v_row)
        # `beta_b` is deliberately not freed here: it is a reshape of the caller's `beta`, and
        # ttnn.reshape may hand back a view, in which case freeing it would free a tensor the caller
        # frees again. Dropping the Python reference is enough.
        # transpose folded into the matmul: k_row^T @ delta is the rank-1 delta outer product.
        outer = self._state_matmul(k_row, delta, "outer", b, transpose_a=True, memory_config=outer_mem)
        ttnn.deallocate(k_row)
        ttnn.deallocate(delta)
        ttnn.add(state, outer, output_tensor=state)
        ttnn.deallocate(outer)

        out = self._state_matmul(q_row, state, "read", b, memory_config=step_mem)
        ttnn.deallocate(q_row)
        return out

    # ------------------------------------------------------------------ public forwards
    def _block(self, x, *, mode, logical_len=None, page_table=None, chunk_start_idx=0, current_pos=None, rot_idxs=None):
        """One decoder block: norm → mixer → residual → norm → MoE → residual."""
        self._decode_phase = mode == "decode"
        b, t = int(x.shape[0]), int(x.shape[1])
        tokens = b * t
        # The whole decode mixer runs on the folded residual (:data:`DECODE_COMPACT_ROWS`). The fold
        # is here rather than at each call site so that exactly one shape convention reaches the
        # norms, the projections, the slices and the state: `[1, 1, tokens, dim]`, which is what the
        # MoE half of this block has always used. `x` belongs to the caller, so the folded copy is
        # freed below through `_free_unless_aliased` — at batch 1 the reshape is a view of `x` itself
        # and freeing it would be a use-after-free.
        residual = self._fold_decode_rows(x, tokens)
        folded = residual is not x
        # The token-mixer norm hands its shard straight to the in-projection when that projection's
        # tuned config can take it (§4.21). The MoE norm below cannot: its consumer is `shared_in`,
        # whose `in0_block_w` is wider than the norm's per-core shard.
        attn_in = self._norm(
            residual,
            self.w["attn_norm"],
            keep_sharded_for=("attn_in" if self.is_full_attention else "gdn_in") if mode == "decode" else None,
        )
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

        h = ttnn.add(residual, mixed, dtype=self.policy.residual_dtype)
        ttnn.deallocate(mixed)
        # Frees the fold, no-ops on `x` itself and on the batch-1 view of it.
        _free_unless_aliased(residual, x)

        padded_tokens = _align_up(tokens, TILE)
        ff_norm = self._norm(h, self.w["ff_norm"])
        # Already `[1, 1, tokens, dim]` when the mixer ran folded, so the reshape the MoE used to
        # need is the fold at the top of this method instead of an extra op.
        ff_in = ff_norm if folded else ttnn.reshape(ff_norm, [1, 1, tokens, self.cfg.dim])
        if padded_tokens != tokens:
            # Do not free the pre-pad tensor: ttnn.pad may alias it.
            ff_in = _pad_dim(ff_in, 2, padded_tokens - tokens)
        ff_out = self.moe.forward(
            ff_in, valid_tokens=tokens if padded_tokens != tokens else None, decode=self._decode_phase
        )
        ttnn.deallocate(ff_in)
        if padded_tokens != tokens:
            trimmed = ttnn.slice(ff_out, [0, 0, 0, 0], [1, 1, tokens, self.cfg.dim])
            ttnn.deallocate(ff_out)
            ff_out = trimmed
        if not folded:
            ff_out = ttnn.reshape(ff_out, [b, t, self.cfg.dim])
        out = ttnn.add(h, ff_out, dtype=self.policy.residual_dtype)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        if folded:
            # Back to the layer's public `[b, t, dim]` contract. This is the same op the unfolded
            # spelling spent on `ff_out` above, moved one add later, so the block's reshape count is
            # unchanged; a caller that wanted the folded residual carried between layers would save
            # this one and the fold at the top.
            unfolded = ttnn.reshape(out, [b, t, self.cfg.dim])
            _free_unless_aliased(out, unfolded)
            out = unfolded
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
        # Same rule as the functional decoder: the DeltaNet state is per-row and cannot be resized
        # under a live sequence, so linear_attention pins the batch; full_attention has no such
        # state and may run any batch up to what its buffers were allocated for.
        if self.batch_size is None or (self.is_full_attention and b > self.batch_size):
            self.allocate_state(b)
        elif not self.is_full_attention and b != self.batch_size:
            raise ValueError(f"batch {b} != allocated state batch {self.batch_size}")

        outputs = []
        for offset in range(0, seq_len, chunk_size):
            logical = min(chunk_size, seq_len - offset)
            phys = min(chunk_size, _align_up(logical, PREFILL_ALIGN))
            block, owned = _slice_owned(x, [0, offset, 0], [b, offset + logical, dim])
            if phys > logical:
                # ttnn.pad may alias `block`, so ownership does not change.
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
        if self.batch_size is None or (self.is_full_attention and b > self.batch_size):
            self.allocate_state(b)
        elif not self.is_full_attention and b != self.batch_size:
            raise ValueError(f"batch {b} != allocated state batch {self.batch_size}")
        return self._block(x, mode="decode", current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table)

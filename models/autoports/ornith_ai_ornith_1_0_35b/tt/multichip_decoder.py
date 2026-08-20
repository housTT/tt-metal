# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Multi-chip TTNN decoder layer for ornith-ai/Ornith-1.0-35B.

Target hardware: the 4-chip Blackhole ``p300c`` ring on this host (2 x p300 dual-ASIC cards,
``ClusterType.P300_X2``, every chip degree 2, 2 ethernet links per hop). The mesh is opened as
``ttnn.MeshShape(1, 4)`` under ``FabricConfig.FABRIC_1D_RING``, which is what actually selects the
ring for every collective. ``ttnn.Topology.Ring`` is additionally passed to the ops that still accept
it (``all_reduce``, ``reduce_scatter``); ``ttnn.all_gather`` deprecated and **ignores** both
``topology`` and ``num_links`` (``all_gather_nanobind.cpp``), so the shipped decode collective takes
the ring from the fabric config alone. The fabric config and the ``topology`` argument are measured
**separately**: ``probe_ccl.py --fabric line`` reconfigures the fabric in its own process and tags
its rows ``CCLFAB`` (the ring fabric is ahead on every traced row of both shipped spellings, by about
14% on the ``stack_sum`` the layer picks at the decode tile), while the ``CCL`` rows vary only the
ops' argument under the ring fabric. Rounds 0-3 of this
stage quoted the second comparison as if it were the first.

Baseline
--------
:class:`~models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder.OptimizedDecoder` is the
single-chip baseline and this class **subclasses it**, unmodified. Every program config, precision
policy, sharded-norm decision, sparse-matmul geometry rule, paged-cache contract and trace-safety
property of that stage is inherited rather than re-derived; what this module adds is

1. a **local** :class:`OrnithDecoderConfig` (:func:`local_decoder_config`) whose head counts, expert
   count and shared-expert width are the *per-device* ones, so every shape the inherited code
   computes from ``self.cfg`` is already the sharded shape;
2. **setup-time weight sharding** (:meth:`MultichipDecoder.from_state_dict`) that hands each device
   exactly the slice its local config describes;
3. **two collectives per layer** (:meth:`MultichipDecoder._all_reduce`), one after the token mixer
   and one after the MoE;
4. a **globally correct router that emits device-local routing** (:meth:`MultichipMoE.routing_weights`).

Parallelisation
---------------
``TP = 4`` for every dense tensor, ``EP = 4`` for the 256 routed experts, on the same four chips.
The residual stream stays **replicated**, which makes both RMSNorms exact and local (no distributed
norm, no stats all-gather) and makes the MoE input available in full width for expert parallelism.

======================  =====================================  ===============================
tensor / activation     global                                 per device (TP=4 / EP=4)
======================  =====================================  ===============================
residual ``x``          ``[b, t, 2048]``                       replicated ``[b, t, 2048]``
``attn_in``             ``[2048, 9216]``                       ``[2048, 2560]``  (col-parallel)
q heads                 16                                     4
kv heads                2                                      1 (kv head ``d // 2``)
``o_proj``              ``[4096, 2048]``                       ``[1024, 2048]``  (row-parallel)
paged K/V cache         ``[nb, 2, 64, 256]``                   ``[nb, 1, 64, 256]``
``gdn_in``              ``[2048, 12352]``                      ``[2048, 3136]``  (col-parallel)
gdn key/value heads     16 / 32                                4 / 8
``gdn_out``             ``[4096, 2048]``                       ``[1024, 2048]``  (row-parallel)
DeltaNet state          ``[b, 32, 128, 128]``                  ``[b, 8, 128, 128]``
conv taps               ``[8192, 1, 4]``                       ``[2048, 1, 4]``
``expert_gate_up``      ``[1, 256, 2048, 1024]``               ``[1, 64, 2048, 1024]`` (EP)
``expert_down``         ``[1, 256, 512, 2048]``                ``[1, 64, 512, 2048]``  (EP)
``shared_in``           ``[1, 1, 2048, 1056]``                 ``[1, 1, 2048, 288]``  (col-parallel)
``shared_down``         ``[1, 1, 512, 2048]``                  ``[1, 1, 128, 2048]``  (row-parallel)
``router``              ``[1, 1, 2048, 256]``                  replicated (global top-8 is global)
======================  =====================================  ===============================

Two dims do not divide by 4 and are handled explicitly rather than by rounding the model down:

* **``n_kv_heads = 2`` over 4 devices.** Devices 0,1 own kv head 0 and devices 2,3 own kv head 1,
  which is exactly the GQA grouping the 16 query heads already impose (query head ``h`` uses kv head
  ``h // 8``). Each device stores **one** kv head, i.e. half the single-chip cache, and the k/v
  projection rows are duplicated across the pair — 256 extra weight columns in ``attn_in`` per layer,
  557056 B at bfloat8_b, against a halved per-device KV cache.
* **``a`` / ``b`` DeltaNet gates, 32 wide over 4 devices.** 8 columns per device is a quarter tile,
  so each gate gets its own 32-column block in the packed ``gdn_in`` weight whose trailing 24
  columns are exact zeros, and :meth:`MultichipDecoder._gdn_project` slices the 8 real ones back
  out. The padding is internal and never reaches the delta rule.

Each collective is spelled by :data:`CCL_MODE`: ``all_gather`` onto a new axis plus a local sum at
the decode tile, ``ttnn.all_reduce`` above it and for prefill. The alternatives — the sharded
residual, the tuned ``all_reduce_async``, and the two fused matmul+CCL ops — are measured in
``doc/multichip_decoder/logs/probe_ccl.txt`` and ``probe_fused_ccl.txt`` and rejected there, each on
a number or an exact op-contract blocker rather than on an API error.

Contract
--------
Identical to the optimized decoder's, including that ``seq_len`` may be **any** value in
``[1, max_context - start_pos]``: the 4-way sharding adds no divisibility requirement to any public
argument. The one deliberate difference is the *distribution* of the input and output tensors:
inputs are replicated across the mesh and outputs are identical on every device, which is the layout
a stack of these layers passes between them with no boundary conversion.
"""

from __future__ import annotations

from dataclasses import replace

from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import OrnithDecoderConfig
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import (
    _DTYPE_BYTES,
    _UNKNOWN_DTYPE_BYTES,
    DECODE_MATMUL_GEOMETRY,
    DECODE_MATMUL_IN0_TILE_BUDGET,
    DECODE_MATMUL_MAX_M_TILES,
    DECODE_MATMUL_WEIGHT_FIELD,
    DEFAULT_MOE_GROUP_TOKENS,
    DEFAULT_PAGE_BLOCK_SIZE,
    DEFAULT_POLICY,
    DEFAULT_PREFILL_CHUNK,
    DEFAULT_ROPE_MODE,
    POLICIES,
    PREFILL_ALIGN,
    SPARSE_CORES_PER_ACTIVE,
    TILE,
    OptimizedDecoder,
    OptimizedMoE,
    OrnithFusedRope,
    PrecisionPolicy,
    _align_up,
    _conv_compute_config,
    _decode_1d_matmul_config,
    _drop_prepared,
    _free_unless_aliased,
    _pad_dim,
    _physical_rows,
    _prefill_2d_matmul_config,
    _ProjectionConfigs,
    _rope_head_permutation,
    _slice_last,
    num_blocks_for_context,
)

__all__ = [
    "DEFAULT_MESH_SHAPE",
    "DEFAULT_FABRIC_CONFIG",
    "DEFAULT_FABRIC_PACKET_BYTES",
    "fabric_router_config",
    "DEFAULT_CCL_TOPOLOGY",
    "MultichipDecoder",
    "MultichipMoE",
    "local_decoder_config",
    "num_blocks_for_context",
]

# --------------------------------------------------------------------------------------
# target mesh
# --------------------------------------------------------------------------------------

#: Mesh this module is built for. Four Blackhole ``p300c`` chips wired as a ring
#: (0-1-2-3-0, 2 ethernet links per hop, ``ClusterType.P300_X2``). Opened 1x4 rather than 2x2
#: because every collective here spans **all four** devices: a 2x2 mesh would only let a CCL address
#: one axis at a time (2 devices) unless it is flattened anyway, and the physical ring is 1D.
DEFAULT_MESH_SHAPE = (1, 4)

#: Fabric to configure **before** ``ttnn.open_mesh_device``. The ring is physical, so the 1D ring
#: fabric gives both directions. Both comparisons are in ``probe_ccl.txt``, and they are different
#: comparisons: the ``CCL`` rows are ``Topology.Ring`` against ``Topology.Linear`` **as the
#: collectives' argument under this fabric**, and the ``CCLFAB`` rows are this fabric against
#: ``FABRIC_1D`` (a second process, since the fabric is set before the mesh is opened). The fabrics
#: are close on ``all_reduce`` below 512 rows, but the ring is ahead on **every** traced row of both
#: spellings, and on the ``stack_sum`` spelling this layer actually picks at the decode tile it is
#: ahead by about 14% there and more as rows grow. The fabric config is load-bearing at decode.
DEFAULT_FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_1D_RING

#: Topology passed to the collectives that still accept one. ``ttnn.all_reduce`` and
#: ``ttnn.reduce_scatter`` take it; ``ttnn.all_gather`` marks both ``topology`` and ``num_links``
#: deprecated and ignored, so the ``stack_sum`` path takes the ring from :data:`DEFAULT_FABRIC_CONFIG`
#: instead. That is not a gap: the fabric config is what selects the ring, and it is set before
#: ``ttnn.open_mesh_device`` for every process in this stage.
DEFAULT_CCL_TOPOLOGY = ttnn.Topology.Ring

#: Ethernet links per hop on this machine (both directions of the ring are 2-wide). Passed to the
#: same ops as :data:`DEFAULT_CCL_TOPOLOGY`, and ignored by ``ttnn.all_gather`` for the same reason.
DEFAULT_CCL_NUM_LINKS = 2

#: Fabric max packet payload, in bytes, to configure alongside :data:`DEFAULT_FABRIC_CONFIG`.
#:
#: The build default on this machine is 4352 B. The two per-layer collectives carry **different
#: dtypes** — the token mixer's is ``bfloat16`` and the MoE's is ``bfloat8_b`` — so their tile pages
#: are 2048 B and 1088 B, and ``ccl_common.cpp:60-61`` derives a different ideal packet for each (and warns from ``:64-70``):
#: ``min(15232/page, 4) * page`` is 8192 B for the first and 4352 B for the second. One fabric
#: setting serves both, so the runtime warns about whichever it is not; the warning cannot be driven
#: to zero, and rounds 5 and 6 each found a version of this file treating it as if it could.
#:
#: Measured on both dtypes (``probe_ccl.txt``: ``CCL``/``CCLPKT`` for bf16, ``CCLBF8``/``CCLBF8PKT``
#: for block-float). README section 2.1 generates the census; it counts rows rather than
#: characterising them, because three review rounds each found a characterisation of this comparison
#: overstated. bf16 favours 8192 B on the large majority of its 72 traced rows, by up to 18%, and
#: gives up under a percent on the handful that fall the other way; block-float splits almost evenly,
#: with extremes of a few percent in both directions and no shape favouring either size consistently.
#: So 8192 B is taken on the bf16 rows and costs nothing measurable on the others. ``ab_single_vs_multichip.txt`` carries the
#: layer-level arms, where the difference is at or below the build-to-build spread.
#:
#: This is a **fabric** setting, so it is applied before ``ttnn.open_mesh_device`` by whoever opens
#: the mesh, not by this module: :func:`fabric_router_config` builds the object, the suite passes it
#: through ``device_params["fabric_router_config"]``, and every probe and benchmark here sets it the
#: same way. A caller that forgets it gets a correct but slightly slower layer.
DEFAULT_FABRIC_PACKET_BYTES = 8192


def fabric_router_config(packet_bytes: int = DEFAULT_FABRIC_PACKET_BYTES):
    """``ttnn.FabricRouterConfig`` carrying :data:`DEFAULT_FABRIC_PACKET_BYTES`.

    Pass to ``ttnn.set_fabric_config(..., router_config=...)`` before opening the mesh, or through
    the pytest ``device_params`` fixture's ``fabric_router_config`` key.
    """
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = packet_bytes
    return router


#: Tensor-parallel / expert-parallel factor. One number, because the same four devices carry both.
DEFAULT_TP = 4


# --------------------------------------------------------------------------------------
# knobs the A/B probes flip
# --------------------------------------------------------------------------------------

#: How the two per-layer collectives are spelled.
#:
#: ``"auto"`` — the shipped default — is ``"stack_sum"`` at or below
#: :data:`CCL_STACK_SUM_MAX_ROWS` physical activation rows and ``"all_reduce"`` above.
#: ``doc/multichip_decoder/logs/probe_ccl.txt`` measures every spelling at every shape the layer
#: produces, **inside a captured trace**, which is where a decode step actually pays:
#:
#:   * ``ttnn.all_reduce`` and an explicit ``reduce_scatter`` + ``all_gather`` are the same number to
#:     the probe's repeatability at every shape — the stable all-reduce lowers to exactly that pair — so
#:     ``"rs_ag"`` exists to make that identity checkable, not because it is a separate candidate.
#:   * ``Topology.Ring`` beats ``Topology.Linear`` as the ops' argument at every traced shape, which
#:     is the physical ring being real; the ``CCLFAB`` rows say the same of the fabric config itself
#:     above 64 rows. ``FABRIC_1D_RING`` + ``Ring`` is therefore what the layer configures.
#:   * ``ttnn.experimental.all_reduce_async`` — the tuned experimental-tier op — runs correctly here
#:     and is **1.5-1.9x slower than the shipped arm at every measured shape** — roughly double at the
#:     decode tile and 1.7x at the 2048-token prefill chunk; ``probe_ccl.txt`` has the row. Rounds 0-3
#:     of this stage recorded it as refusing Blackhole DRAM; it does not, and the arm had simply been
#:     calling it with one barrier semaphore and no ``cluster_axis``. Rejected on measurement.
#:   * ``"stack_sum"`` (``all_gather`` onto a new leading axis, then a local ``ttnn.sum``) moves 4x
#:     the bytes and is the **winner below the crossover** anyway: at the batch-1 decode tile it is
#:     roughly half the all-reduce, because at that size both are latency-bound and it is one fabric
#:     phase instead of two. It loses from 96 rows up.
#:   * ``"stack_sum_async"`` is ``"stack_sum"`` with ``ttnn.experimental.all_gather_async`` in place of
#:     the deprecated ``ttnn.all_gather``, given two persistent global semaphores and a persistent
#:     **barrier** semaphore built at construction. It exists because the deprecated op takes no
#:     semaphore at all, and this stage reproduced a cross-device divergence under sustained traced
#:     replay with it — on BOTH router modes, i.e. on the path stage 4 shipped
#:     (``logs/probe_replay_divergence.txt``, ``doc/optimized_multichip_decoder/work_log.md`` §11).
#:     the correctness evidence and the price are in work_log §11.
#:
#: **This stage ships ``"all_reduce"``, not ``"auto"``.** The multichip stage's crossover was measured
#: against the deprecated ``ttnn.all_gather`` spelling of ``stack_sum``, and that spelling is the one
#: that diverges across devices under sustained traced replay (§11 of
#: ``doc/optimized_multichip_decoder/work_log.md``, whose generated table carries the counts and the
#: magnitudes; they are not repeated here, because a figure in a docstring is a figure nothing
#: regenerates).
#: Removing it leaves two correct candidates below the crossover, and ``ttnn.all_reduce`` wins both
#: comparisons: zero diverged rounds against the deprecated op's non-zero count
#: (``logs/probe_replay_divergence*.txt``, tabulated in README section 4.1) and the faster of the two
#: at the layer (``logs/ab_layer_knobs.txt``, ``collective`` arm). So the crossover is void rather
#: than moved: the stable op is the right answer at every shape now, and ``"auto"`` is kept only as a
#: measurement arm.
CCL_MODE = "all_reduce"

#: Physical activation rows at or below which ``"auto"`` picks ``"stack_sum"``. The crossover is
#: measured, not modelled (``probe_ccl.txt``, ``trace`` rows, which amortise the fixed per-replay
#: dispatch over 8 captured copies): ``stack_sum`` wins clearly at 32 rows, is a tie at 64, and loses
#: by a widening margin from 96 rows up. 64 rows is decode batch 2; every larger batch and all of
#: prefill take the all-reduce. The ``ccl`` arms of ``doc/multichip_decoder/logs/ab_layer_knobs.txt``
#: measure the layer-level effect of this switch end to end, and README section 5.5 tabulates it;
#: no figure is quoted here, because this file states no run-varying absolute timing.
CCL_STACK_SUM_MAX_ROWS = 64

#: What ``CCL_MODE="auto"`` resolves to at or below :data:`CCL_STACK_SUM_MAX_ROWS`. **Neither value
#: ships**: :data:`CCL_MODE` is ``"all_reduce"``, so ``"auto"`` is a measurement arm only.
#: ``"stack_sum"`` is the multichip stage's choice and the faster of the two, and it is the spelling
#: that diverges across devices - on this stage's router and on the multichip stage's alike
#: (``doc/optimized_multichip_decoder/work_log.md`` §11);
#: ``"stack_sum_async"`` is the barrier-semaphore repair for it, which is correct but materially
#: behind ``ttnn.all_reduce`` at the decode tile (README §4.1's table has the figures). Separate from :data:`CCL_MODE` so the A/B harness
#: can vary the two independently.
AUTO_STACK_SUM_MODE = "stack_sum_async"

#: Whether a block-float activation is cast to ``bfloat16`` before the collective. **Off**, measured.
#:
#: The MoE half of the layer produces ``bfloat8_b`` (the optimized stage's routed-expert activation
#: dtype), so the second per-layer collective is handed a block-float tensor while the first gets
#: ``bfloat16``. In the prefill profile those two are the same logical shape and wildly different
#: cost: ``tracy/full_attention/prefill_perf_report.txt`` has the BF16 reduce-scatter at ~100 us on 20
#: cores and the BFP8 one **an order of magnitude above it** on 12 cores — a few percent of the whole
#: prefill window, and the largest non-sparse item in it. (No absolute figure is quoted here, for the
#: reason :data:`CCL_STACK_SUM_MAX_ROWS` gives: this file states no run-varying timing. README §5.8
#: carries the row, regenerated from the profile by ``logs/make_tables.py``.) That reads as an obvious
#: win, and it is not one.
#:
#: The ``cast`` arm of ``doc/multichip_decoder/logs/ab_layer_knobs.txt`` measures it at the layer:
#: casting up costs a few microseconds on every decode step (README section 5.5's generated table has
#: the pair; it is 3-4 us depending on the layer kind in the committed sweep) and moves warmed prefill by
#: **nothing** — the arms'
#: three-build ranges overlap on both layer kinds. So that row is **not** data movement this layer
#: pays: removing the block-float operand changes its cost and not the layer's. What
#: it *is* remains open — a collective barrier absorbing device skew is the candidate — and README
#: limitation 9 records it as a candidate rather than a finding.
#:
#: Kept as a knob rather than deleted because that null result is the control for the anomaly, and
#: because a future dtype policy could move the boundary.
CCL_CAST_BLOCKFLOAT = False

#: Whether the routed sparse matmuls' core target is rescaled for expert parallelism.
#:
#: ``OptimizedMoE`` targets ``clamp(bound // SPARSE_CORES_PER_ACTIVE[role], 8, 32)`` cores, where
#: ``bound`` is ``_active_expert_bound`` = ``min(num_experts_local, rows * top_k)``. The divisor means
#: "work per core, per active expert" and was calibrated at ``E = 256``, where that bound saturates
#: at 256 and a 32-token prefill group activates ~162 experts. Expert parallelism cuts the bound's
#: ceiling to 64 without changing ``Nt`` (``moe_intermediate_size`` and ``dim`` are not sharded, so
#: the op still produces 32 and 64 output tiles), so inheriting the divisor shrinks the core target
#: exactly where the available parallelism did not change.
#:
#: On means the bound is scaled back by the same divisor, i.e. the net rule becomes
#: ``cores = clamp(bound, 8, 32)``. Read off the shipped constants rather than the intended
#: narrative — review round 2 found an earlier version of this comment describing an effect one
#: divisor and one operating point away from the real one:
#:
#:   ==============  =====  ==========================  =======================
#:   call            bound  ``gate_up`` realised, off->on  ``down`` realised, off->on
#:   ==============  =====  ==========================  =======================
#:   decode b=1          8  8 -> 8                      8 -> 8
#:   decode b=2         16  8 -> 16                     8 -> 16
#:   decode b=4         32  16 -> 32                    8 -> 32
#:   decode b>=8        64  32 -> 32                    16 -> 32
#:   prefill group      64  32 -> 32                    16 -> 32
#:   ==============  =====  ==========================  =======================
#:
#: So at prefill only ``down`` moves, and decode geometry changes at **every batch above 1** — the
#: opposite of what that earlier comment claimed. Both are measured rather than argued:
#: ``ab_layer_knobs.txt``'s ``sparse`` arm has the prefill effect (several milliseconds a layer on
#: both layer kinds) and ``probe_decode_batch.txt``'s ``SPARSEB`` rows have the decode effect at batch
#: 1..32 (a tie at batch 1 by construction, and tens of microseconds a step at every batch above it).
#: ``probe_sparse_matmul_local.txt`` is the isolated ladder the rule is checked against, and
#: ``test_sparse_cores_match_the_local_sweep`` pins the realised grid at every advertised batch.
#: The rescale is additionally gated on ``tp > 1``, so a layer built on a 1-device mesh reproduces
#: ``OptimizedMoE`` exactly rather than only when this flag is cleared.
SPARSE_SCALE_CORES_BY_TP = True

#: Whether a rank-3 ``[b, t, dim]`` activation is folded to ``[1, 1, b * t, dim]`` before the
#: collective and viewed back after.
#:
#: A tile's row axis is the *sequence* axis, so a decode activation ``[b, 1, dim]`` occupies ``b``
#: tile rows — ``b * 32`` physical rows — of which ``b`` carry data. The collective moves the padding
#: too. At the advertised batch bound of 32 that is 1024 rows to reduce 32, and the layer's other
#: collective already runs on the compacted ``[1, 1, tokens, dim]`` form that ``_block`` builds for
#: the MoE, so without this the same logical reduction costs 32x more at one call site than the
#: other in the same forward.
#:
#: Off at batch 1, where ``[1, 1, dim]`` is already one tile row and the reshape would be pure
#: overhead: the guard compares the physical row count against ``align_up(b * t, 32)`` and only folds
#: when it is strictly larger. It is therefore a **decode-only** path in practice: ``prefill_forward``
#: pads every chunk to ``PREFILL_ALIGN`` before ``_block`` runs, so a prefill operand's physical row
#: count already equals ``align_up(b * t, 32)`` and the guard is false at every prefill shape. Round
#: 8's correctness audit found an earlier comment claiming a batched non-aligned prefill reached it. ``doc/multichip_decoder/logs/probe_decode_batch.txt`` records both the
#: measured shapes at each call site and the traced-decode A/B at batch 1/4/13/32.
#:
#: **Superseded on the shipped path, and kept as its own fallback.** With
#: ``optimized_decoder.DECODE_COMPACT_ROWS`` on, ``_block`` folds the residual before the mixer runs,
#: so both operands reach :meth:`MultichipDecoder._all_reduce` already rank-4 and this guard is false
#: at every call in a decode forward — the collectives move the same rows they moved with the fold
#: here, and the fold/view-back pair it used to dispatch on the mixer operand is gone (the MoE
#: operand was already rank-4, so this guard never fired on it). This flag is what keeps the
#: collective's payload compact when that outer fold is switched off, and it still covers any rank-3
#: caller from outside ``_block``. The measured value of the fold *as a collective-only fold* is
#: unchanged and still recorded in ``probe_decode_batch.txt``; it is now the floor of what the outer
#: fold is worth, not the whole of it.
CCL_COMPACT_ROWS = True

#: How the globally-routed dense score vector is narrowed to this device's expert block.
#: ``"select_matmul"`` multiplies the replicated ``[1, 1, tokens, 256]`` vector by a mesh-sharded
#: one-hot ``[256, 64]`` (the four shards concatenate to ``I_256``); ``"gather"`` uses
#: ``ttnn.gather`` with a mesh-sharded index tensor. Both are exact — the selection matmul sums 255
#: exact zeros and one bfloat16 value, and ``test_routing_select_modes_agree`` asserts the two give
#: identical layer output. Measured end to end by the ``routing`` arms of
#: ``doc/multichip_decoder/logs/ab_layer_knobs.txt``.
#:
#: ``"gather"`` is a **measurement arm, not a shippable path, and it is not trace-safe**: its index
#: tensor is built with ``torch`` and uploaded on first sight of each routed shape, i.e. inside the
#: forward. That is a host write, illegal under trace capture and exactly what the stage's
#: no-host-fallback guard forbids — which is why ``test_no_host_fallback_in_forward`` runs the
#: shipped mode only. Round 8's correctness audit asked for that to be said here rather than implied
#: by the default. It is kept because a rejected alternative has to stay runnable to stay honest.
ROUTING_SELECT_MODE = "select_matmul"

#: Per-role ``(core target, in0_block_w cap)`` for the dense decode matmuls, **re-swept at the
#: per-device TP=4 shapes**. The single-chip table was tuned at the unsharded widths and does not
#: transfer: ``mcast_in0`` streams the whole ``K`` through each core in ``in0_block_w``-tile blocks,
#: so the cap that wins is a function of how much ``N`` each core owns, and TP=4 cuts ``N`` by four
#: on the two wide in-projections. ``doc/multichip_decoder/logs/probe_dense_matmul.txt`` has the whole
#: ladder — 9 core targets x 6 caps for every dense decode role, at its per-device shape.
#:
#: The sweep's core column is the *realised* grid, not the requested target: ``_decode_1d_matmul_config``
#: lays a target out as ``ceil(target / 11)`` rows of ``min(11, target)`` columns, so the inherited
#: targets 16/24/32/48 realise as 22/33/33/55 cores and the inherited ``in0_block_w`` is additionally
#: clamped to ``Kt``. Every inherited entry is therefore compared to the local winner **at its
#: realised point**, and the comparison is judged against this probe's own repeatability rather than
#: against zero: the file measures several realised configs more than once (distinct core targets
#: collapse onto the same grid), and those repeats disagree by up to ~0.4 us. No figures are quoted
#: here — README section 5.6 has the generated table and this file states no run-varying absolute
#: timing, an invariant inherited from the single-chip stage.
#:
#: Roles kept at the inherited entry — ``o_proj``, ``gdn_out``, ``shared_in``, ``router`` — sit within
#: that repeatability band of the local winner, and which of the two is ahead is not stable across
#: independent sweeps of the same binary. Roles retuned below are outside it by a wide and stable
#: margin:
#:
#:   * ``attn_in`` and ``gdn_in`` are the two wide in-projections, where the inherited cap of 2 is the
#:     *worst* legal value for the local shape and costs a large factor on the op;
#:   * ``shared_down`` because TP=4 cuts its ``K`` from 512 to 128 — 4 tiles — and the inherited
#:     entry's 55-core grid for a 4-tile ``K`` and a 64-tile ``N`` is launch overhead rather than
#:     parallelism. 8 cores rather than the 4 an earlier round shipped: 4 is the faster of the two on
#:     some sweeps and 1.4 us slower on others, while 8 has never read above 9 us on any sweep, so 8 is
#:     both the stable choice and never behind;
#:   * ``expert_select`` is a role this stage introduces and the single-chip table has no entry for.
#:
#: The ``geometry`` arm of ``doc/multichip_decoder/logs/ab_layer_knobs.txt`` measures the whole
#: retuned set against the inherited one at the layer.
#:
#: ``in0_block_w`` is additionally bounded above by the residual norm's per-core shard width when the
#: shard is carried into the projection (``_shard_feeds_projection`` re-derives ``mcast_in0``'s
#: ``block_w % in0_block_w == 0`` rule): the 2048-wide norm over ``NORM_SHARD_CORES`` = 8 cores gives
#: 8 tiles per core, so 8 is the largest cap that keeps the carry legal. Caps of 16 and 32 do build
#: against an interleaved ``in0`` and are measured — they are simply slower — and the only two
#: ``FAIL`` rows in the sweep are a different constraint, ``cores=4, in0_block_w=32``.
MULTICHIP_DECODE_MATMUL_GEOMETRY = {
    "attn_in": (110, 8),
    "gdn_in": (110, 8),
    "shared_down": (8, 4),
    "expert_select": (8, 8),
}

#: How the router turns 256 logits into the per-token top-8 and its softmax weights, at **decode**.
#:
#: ``"topk"`` is the chain every earlier stage shipped: ``ttnn.topk(k=8)`` on float32 logits, then
#: ``ttnn.softmax`` over the kept 8, then ``ttnn.scatter`` into a persistent all-zero 256-wide vector.
#: In the multichip decode profile its ``TopKDeviceOperation`` row alone is 48 us on **one core** —
#: 9.5% of the traced decode window, the largest single non-sparse op — and the whole chain is ~17%.
#:
#: ``"fused_gate"`` — the shipped default — replaces the ``topk`` + ``softmax`` half with the single
#: ``ttnn.experimental.deepseek.moe.generalized_moe_gate`` kernel: score, top-``k`` and
#: softmax-over-selected in one op over a 16x16 expert face, one token per core, writing into
#: **preallocated** output buffers. The optimized stage rejected this op as "bfloat16-only" and, in
#: its own words, never timed it. ``doc/optimized_multichip_decoder/logs/probe_gate.txt`` times it,
#: and it is an order of magnitude cheaper than the two ops it replaces in that same untraced
#: harness; the decode capture puts it at about 2 us of device time. The scatter is unchanged. No
#: absolute timing is quoted in this file - that invariant is inherited from the single-chip stage
#: and it is why the figures live in the generated tables of README section 2 and work log section 3.
#:
#: The op takes bfloat16 logits, so this **is** a router precision change and it is measured as one:
#: ``logs/probe_gate.txt`` reports selected-set agreement against the float32 chain at the decode and
#: prefill row counts, and ``tests/test_multichip_decoder.py::test_router_modes_agree`` pins layer
#: output PCC between the two modes on real weights. Prefill keeps the float32 ``topk`` chain: the
#: gate op is one token per core, a 2048-token chunk would need 19 sequential calls, and ``TopK`` is
#: 0.17% of the prefill window, so there is nothing to win there.
ROUTER_MODE = "fused_gate"

#: Memory config for the two residual adds **at decode**. ``None`` — the shipped value — lets
#: ``ttnn.add`` take the operand's config, which is DRAM interleaved.
#:
#: This is a knob because OPT-003 says a decode residual should not sit in DRAM merely because it is
#: convenient, and because the decode profile puts a tenth to a sixth of the window in ``BinaryNg``
#: and about a fifth in layout (``TM``) — the generated share table in README §5.2 — with several
#: rows marked ``in0:dram_interleaved`` on tensors that are one 32-row tile: 4 KiB at batch 1,
#: 128 KiB at the advertised bound. It was the largest lever this stage had not tried.
#:
#: Measured, and it is a **tie**: the ``residual`` arm of ``logs/ab_layer_knobs.py`` reads the same
#: for both arms on both layer kinds, three builds each, inside the harness's spread. The step is
#: launch-bound, and moving a 4 KiB tensor's home does not change how many ops there are. The
#: inherited spelling ships because a tie is not a reason to change anything; the knob and its
#: measurement stay so the next stage does not re-run the experiment blind. See README section 5.1
#: and work_log section 6.4.
DECODE_RESIDUAL_MEMORY = None

#: Math fidelity for the router matmul **at decode**. ``None`` keeps the inherited policy, which is
#: HiFi4 with float32 accumulation — chosen by the functional stage because expert selection is a
#: discrete decision and a rounding change swaps an expert rather than perturbing a value.
#:
#: It is a knob because that argument weakened when this stage's fused gate started reading
#: **bfloat16** logits (:data:`ROUTER_MODE`): a HiFi4 float32-accumulate matmul whose result is then
#: rounded to bfloat16 is paying for precision the consumer throws away, and ``tt-perf-report`` flags
#: the row with "HiFi2 may also work and has 2x the throughput of HiFi4". The row is 8 us/step.
#:
#: Measured by the ``router_fidelity`` arm of ``logs/ab_layer_knobs.py``, and gated on the quantity
#: that matters by ``test_router_modes_agree``, which asserts the selected expert set is unchanged.
ROUTER_DECODE_FIDELITY = None

#: Whether the routed-expert sparsity mask always keeps at least one local expert active.
#:
#: With 8 experts drawn from 256 and 64 experts per device, a device gets **zero** active experts
#: with probability ``(3/4)**8`` ~ 10% per token, per layer. Rather than depend on
#: ``ttnn.sparse_matmul`` tolerating an all-zero sparsity (it infers ``nnz`` at runtime, and the
#: optimized stage already recorded a device wedge inside this op when a count was wrong), the mask
#: is floored at local expert 0. That expert's routing **score** stays exactly zero, and the score
#: multiplies the down projection's *input*, so its contribution is exactly zero — the same argument
#: the single-chip stage uses for its tile-padding rows.
MOE_MASK_FLOOR = True


def local_decoder_config(config: OrnithDecoderConfig, tp: int = DEFAULT_TP) -> OrnithDecoderConfig:
    """The per-device view of ``config`` under ``tp``-way tensor/expert parallelism.

    Everything the inherited single-chip code derives from ``self.cfg`` — packed projection widths,
    head splits, GQA repeat factors, cache shapes, sparse-matmul ``Nt``, expert-group bounds — is
    then already the *local* shape, so the forward paths need no per-op sharding arithmetic.

    ``dim`` (2048), ``head_dim``, ``moe_intermediate_size`` and ``num_experts_per_tok`` are
    deliberately **not** divided: the hidden size is the replicated residual, the head dim is not
    split, expert parallelism keeps each expert whole, and the top-k is a global decision.
    """
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")
    for name, value in (
        ("n_heads", config.n_heads),
        ("linear_num_key_heads", config.linear_num_key_heads),
        ("linear_num_value_heads", config.linear_num_value_heads),
        ("num_experts", config.num_experts),
        ("shared_expert_intermediate_size", config.shared_expert_intermediate_size),
    ):
        if value % tp:
            raise ValueError(f"{name}={value} is not divisible by tp={tp}")
    if config.n_kv_heads > tp and config.n_kv_heads % tp:
        raise ValueError(f"n_kv_heads={config.n_kv_heads} is neither <= tp nor divisible by tp={tp}")
    # And the other direction: fewer kv heads than devices means a *group* of devices shares one head,
    # so `kv_head_owner` divides `tp` by the head count and the count has to divide `tp` evenly.
    # Without this, `n_kv_heads=3, tp=4` passes every check above and then hands device 3 an empty
    # k/v slice in `from_state_dict` — which `torch.cat` accepts and the shard mapper splits wrong,
    # silently. Unreachable for this model (2 over 4) but `local_decoder_config` is general in `tp`;
    # found by round 8's correctness audit and, when the first fix did not land, again by round 9.
    if config.n_kv_heads < tp and tp % config.n_kv_heads:
        raise ValueError(f"n_kv_heads={config.n_kv_heads} is below tp={tp} and does not divide it")
    if config.n_heads % config.n_kv_heads:
        raise ValueError(f"n_heads={config.n_heads} is not a multiple of n_kv_heads={config.n_kv_heads}")
    return replace(
        config,
        n_heads=config.n_heads // tp,
        # `n_kv_heads < tp` is the Ornith case: 2 kv heads over 4 devices, so each device owns one
        # and the pair sharing a kv head duplicates it. That duplication is what keeps the GQA
        # grouping intact without any cross-device attention traffic.
        n_kv_heads=max(1, config.n_kv_heads // tp),
        linear_num_key_heads=config.linear_num_key_heads // tp,
        linear_num_value_heads=config.linear_num_value_heads // tp,
        num_experts=config.num_experts // tp,
        shared_expert_intermediate_size=config.shared_expert_intermediate_size // tp,
    )


class _MultichipProjectionConfigs(_ProjectionConfigs):
    """:class:`_ProjectionConfigs` reading :data:`MULTICHIP_DECODE_MATMUL_GEOMETRY` first.

    Same construction, same cache, same prefill 2D family; only the per-role decode
    ``(cores, in0_block_w cap)`` lookup changes, and only for the roles the local sweep moved.
    """

    GEOMETRY = {**DECODE_MATMUL_GEOMETRY, **MULTICHIP_DECODE_MATMUL_GEOMETRY}
    WEIGHT_FIELD = {**DECODE_MATMUL_WEIGHT_FIELD, "expert_select": "router_dtype"}

    def __init__(self, mesh_device, policy=None):
        super().__init__(mesh_device, policy)
        self.in1_bytes = {
            role: _DTYPE_BYTES.get(getattr(policy, field, None), _UNKNOWN_DTYPE_BYTES)
            if policy is not None
            else _UNKNOWN_DTYPE_BYTES
            for role, field in self.WEIGHT_FIELD.items()
        }

    def get(self, role: str, rows: int, k: int, n: int, *, fp32_acc: bool, decode: bool = True):
        if role not in self.GEOMETRY:
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
            cores, in0_cap = self.GEOMETRY[role]
            in0_cap = max(1, min(in0_cap, DECODE_MATMUL_IN0_TILE_BUDGET // m_tiles))
            self._cache[key] = _decode_1d_matmul_config(
                self.grid, cores, int(rows), int(k), int(n), fp32_acc=fp32_acc, in0_cap=in0_cap
            )
        return self._cache[key]


#: Block-float dtypes, which pack a shared exponent per 16 datums. A collective on one of these is
#: measurably slower than on ``bfloat16`` at the same logical shape; see :data:`CCL_CAST_BLOCKFLOAT`.
_BLOCK_FLOAT_DTYPES = (ttnn.bfloat8_b, ttnn.bfloat4_b)

#: ``generalized_moe_gate`` lays one token's experts out in the top-left 16x16 face of a 32x32 tile.
_GATE_FACE = 16
#: Top-k values the gate kernel's finalize rank-mask handles (its own ``TT_FATAL``).
_GATE_LEGAL_TOPK = (4, 6, 8)
#: Denominator stabilisation for the gate's normalisation; irrelevant under ``output_softmax=True``.
_GATE_EPS = 1e-20
#: :data:`ROUTER_MODE` values that run the fused kernel.
_FUSED_GATE_MODES = ("fused_gate", "fused_gate_local")


def kv_head_owner(kv_heads: int, tp: int, device: int) -> int:
    """Which global kv head ``device`` owns the *first* of, for either sharding direction.

    ``kv_heads < tp`` (this model: 2 over 4) is the sharing case — a group of ``tp // kv_heads``
    devices owns one head each. ``kv_heads >= tp`` is the ordinary split, where each device owns
    ``kv_heads // tp`` consecutive heads and this returns the first. The old form divided ``tp`` by
    the head count unconditionally and raised ``ZeroDivisionError`` on the second case, which
    ``local_decoder_config`` explicitly accepts; round 10 found the gap next to the one round 8 fixed.
    """
    if kv_heads >= tp:
        return device * (kv_heads // tp)
    return device // (tp // kv_heads)


def _shard_mapper(mesh_device, dim: int):
    return ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=dim)


def _replicate_mapper(mesh_device):
    return ttnn.replicate_tensor_to_mesh_mapper(mesh_device)


# `_free_unless_aliased` now lives in `optimized_decoder` and is imported above: the decode-residual
# fold this module's `_all_reduce` introduced was extended to the whole decode mixer path there
# (`DECODE_COMPACT_ROWS`), and both call sites need the same aliasing guard.


# --------------------------------------------------------------------------------------
# MoE
# --------------------------------------------------------------------------------------
class MultichipMoE(OptimizedMoE):
    """The optimized MoE with the 256 routed experts split 64-per-device (expert parallelism).

    Everything below the router is unchanged: ``self.cfg`` is the *local* config, so
    ``num_experts`` is 64 and the inherited ``_routed_experts`` chain — packed gate/up sparse
    matmul, SwiGLU in the sparse layout, score-on-the-down-input, ``deepseek_moe_fast_reduce_nc``
    over the expert axis — runs on this device's 64 experts and produces this device's **partial**
    sum over experts. The caller all-reduces it.

    Why expert parallelism rather than sharding the 512-wide expert intermediate:

    * ``ttnn.sparse_matmul`` loops once per *active* expert and its parallelism is capped by the
      output tile count. Sharding the intermediate keeps 8 loop iterations and cuts ``Nt`` from 32
      to 8 tiles, i.e. it removes parallelism the op was already short of. Expert parallelism keeps
      the full ``Nt`` and the tuned geometry, and cuts the loop count to a mean of 2 (an *expected
      maximum* over the four devices of 3.512 at 8 active experts, which is what the step waits for)
      — the single largest item of the decode window.
    * every ``num_experts``-wide intermediate — the packed gate/up output, its two unpacking slices,
      the SwiGLU product, the scored activation, the down output and the expert reduction — becomes
      4x narrower, which the intermediate-sharded variant only achieves for the ones that are also
      ``moe_intermediate`` wide. The ``[1, E, tokens, 2048]`` down output, the largest of them, does
      not shrink at all under intermediate sharding.

    ``doc/multichip_decoder/work_log.md`` has both arms measured.
    """

    def __init__(self, mesh_device, config, weights, *, global_config, tp, **kwargs):
        super().__init__(mesh_device, config, weights, **kwargs)
        self.proj_cfgs = _MultichipProjectionConfigs(mesh_device, self.policy)
        #: The *unsharded* config. Only the router needs it: the top-8 is over all 256 experts.
        self.global_cfg = global_config
        self.tp = tp
        #: One-hot on local expert 0, ``[1, 1, 1, E_local]``, built **here** rather than on first use.
        #: It broadcasts over the group axis, so one tensor serves every ``[1, groups, 1, E_local]``
        #: mask the layer can produce, and building it at construction keeps ``torch`` out of the
        #: forward paths entirely — which ``test_no_host_fallback_in_forward`` checks and which a
        #: lazily-built buffer would break the first time a shape was seen inside a measured pass.
        self._mask_floor = self._build_mask_floor() if MOE_MASK_FLOOR and tp > 1 else None
        #: Persistent ``generalized_moe_gate`` tensors, keyed by the decode row count. Built by
        #: :meth:`prepare_decode_gate` from :meth:`MultichipDecoder.allocate_state`, i.e. at setup,
        #: because they need ``ttnn.from_torch`` and a decode forward may be under trace capture.
        self._gate_buffers: dict[int, tuple] = {}
        #: Persistent ROW_MAJOR all-zero scatter bases for the fused gate, keyed by the same row count.
        self._gate_zeros: dict[int, object] = {}

    # ---------------- fused router gate ----------------
    def prepare_decode_gate(self, rows: int) -> bool:
        """Build the persistent fused-gate tensors for a decode call of ``rows`` tile rows.

        Returns whether the fused gate is available for that row count. Five tensors, allocated once
        here rather than per step; the first four are height-sharded one token per core, as the op
        requires:

        * ``bias`` — the op adds a score-correction bias before ranking. Ornith's router has none, so
          this is exact zeros over all 256 experts; a constant shift changes neither the selection nor
          the normalized weights. It exists because the op's signature requires it.
        * ``in_idx`` — what the op returns for each selected slot. ``arange(256)`` (global expert id)
          under ``"fused_gate"``; the device-local mapping with a dump column under
          ``"fused_gate_local"``, which makes that tensor mesh-sharded rather than replicated. Laid
          out as the op wants it: a 16x16 face, transposed within the face.
        * ``out`` / ``out_idx`` — the **preallocated output buffers** the op writes into and returns.
          Reusing them is what keeps the gate free of per-step allocation inside the trace.
        * the ROW_MAJOR all-zero scatter base, which plays the same role for the fused path that
          :meth:`OptimizedMoE._router_zeros_for` plays for the ``topk`` one.

        Returns ``False`` — and the caller keeps the ``topk`` chain, which is correct and slower —
        when :data:`ROUTER_MODE` does not name a fused mode, when the row count exceeds the worker
        core count (the op is strictly one token per core; that is decode batch > 110 after this
        model's tile padding, far above the advertised bound of 32), or when the model's expert count
        or top-k is outside what the kernel handles. Guards rather than assumptions.
        """
        import torch

        rows = int(rows)
        if ROUTER_MODE not in _FUSED_GATE_MODES:
            return False
        if self._gate_buffers.get(rows, (None,))[0] == ROUTER_MODE:
            return True
        # Rebuilding at the same row count under a different mode replaces these; free the old ones
        # rather than leaking them. Only the A/B harness and the mode-agreement test reach this.
        for stale in self._gate_buffers.pop(rows, ())[1:5]:
            ttnn.deallocate(stale)
        if rows in self._gate_zeros:
            ttnn.deallocate(self._gate_zeros.pop(rows))
        grid = self.grid
        if rows < 1 or rows > grid.x * grid.y:
            return False
        e_global = self.global_cfg.num_experts
        if e_global != _GATE_FACE * _GATE_FACE or self.global_cfg.num_experts_per_tok not in _GATE_LEGAL_TOPK:
            return False
        core_grid = ttnn.num_cores_to_corerangeset(rows, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, (TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR),
        )

        def upload(host, dtype):
            return ttnn.from_torch(
                host,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=mem,
                mesh_mapper=_replicate_mapper(self.device),
            )

        # The op asserts `bias_shape == in_shape`, and the input it sees is the [rows, 16, 16] face
        # view of the logits. Both metadata tensors are allocated at the full (32, 32) shard - that is
        # what the op reads - and sliced to the face shape once, here.
        face = (_GATE_FACE, _GATE_FACE)
        zeros_face = torch.zeros(1, TILE, TILE, dtype=torch.float32)
        bias = upload(zeros_face.repeat(rows, 1, 1), ttnn.bfloat16)
        local = ROUTER_MODE == "fused_gate_local"
        e_local = self.cfg.num_experts
        if local:
            # The op returns, for each selected slot, whatever `input_indices_tensor` holds at that
            # expert position. Feeding it the DEVICE-LOCAL index instead of the global one makes the
            # scatter land straight in this device's 64-wide block: expert `e` maps to `e - d*E_local`
            # when it belongs to device `d`, and to the dump column `E_local` when it does not. The
            # dump column is sliced off after the scatter, so the non-local selections cost a write
            # nobody reads - and the `expert_select` one-hot matmul that used to do this narrowing
            # disappears. Per-device values, so this tensor is mesh-sharded, not replicated.
            per_device = []
            for d in range(self.tp):
                mapped = torch.full((e_global,), e_local, dtype=torch.int32)
                lo = d * e_local
                mapped[lo : lo + e_local] = torch.arange(e_local, dtype=torch.int32)
                per_device.append(mapped.reshape(1, *face).transpose(1, 2).repeat(rows, 1, 1))
            ids_host = torch.cat(per_device, dim=0)
            mapper = _shard_mapper(self.device, dim=0)
        else:
            ids_host = torch.arange(e_global, dtype=torch.int32).reshape(1, *face).transpose(1, 2).repeat(rows, 1, 1)
            mapper = _replicate_mapper(self.device)
        in_idx = ttnn.from_torch(
            torch.nn.functional.pad(ids_host, (0, TILE - _GATE_FACE, 0, TILE - _GATE_FACE)),
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        out = upload(zeros_face.repeat(rows, 1, 1), ttnn.bfloat16)
        out_idx = upload(torch.zeros(rows, TILE, TILE, dtype=torch.int32), ttnn.uint16)
        # The op reads the (32, 32) shard but asserts the FACE logical shape, so the uploads are
        # sliced down and the pre-slice tensors freed - they are setup-only and small, but
        # `allocate_state` should not leak one pair per prepared row count.
        bias_full, in_idx_full = bias, in_idx
        bias = ttnn.slice(bias, [0, 0, 0], [rows, *face], memory_config=mem)
        in_idx = ttnn.slice(in_idx, [0, 0, 0], [rows, *face], memory_config=mem)
        # Each upload is checked against ITS OWN survivor. Review round 6 found the first version
        # comparing `in_idx_full` against `bias`, which disarms the alias guard for exactly the case
        # the guard exists for: if `ttnn.slice` ever returns a view for this metadata-only narrowing,
        # freeing the upload would free the buffer `self._gate_buffers[rows]` then hands every decode
        # step.
        for stale_upload, survivor in ((bias_full, bias), (in_idx_full, in_idx)):
            _free_unless_aliased(stale_upload, survivor)
        self._gate_buffers[rows] = (ROUTER_MODE, bias, in_idx, out, out_idx, mem)
        # The scatter base. ROW_MAJOR, because `ttnn.scatter` untilizes a tiled base on the way in
        # and re-tilizes the result on the way out (scatter.cpp:164/233), and every operand this path
        # hands it is already ROW_MAJOR. `+1` in the local mode is the dump column.
        self._gate_zeros[rows] = ttnn.from_torch(
            torch.zeros(1, 1, rows, (e_local + 1) if local else e_global, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(self.device),
        )
        return True

    def _fused_gate_dense(self, logits):
        """``[1, 1, rows, 256]`` bfloat16 logits -> the same dense routing vector the topk chain builds.

        One ``generalized_moe_gate`` call replaces ``topk`` + ``softmax``; the scatter that turns the
        selected ``(index, weight)`` pairs into the dense vector is the shipped one, unchanged.
        """
        rows = int(logits.shape[-2])
        mode, bias, in_idx, out_buf, out_idx_buf, mem = self._gate_buffers[rows]
        faces = ttnn.to_memory_config(ttnn.reshape(logits, (rows, _GATE_FACE, _GATE_FACE)), memory_config=mem)
        weights, indices = ttnn.experimental.deepseek.moe.generalized_moe_gate(
            faces,
            bias_tensor=bias,
            input_indices_tensor=in_idx,
            output_tensor=out_buf,
            output_indices_tensor=out_idx_buf,
            eps=_GATE_EPS,
            scaling_factor=1.0,
            enable_sigmoid=False,
            topk=self.global_cfg.num_experts_per_tok,
            output_softmax=True,
        )
        ttnn.deallocate(faces)
        k = self.global_cfg.num_experts_per_tok
        # Only row 0 of each token's (32, 32) tile is valid, and only its first k columns.
        #
        # Both selected tensors are taken to ROW_MAJOR before the slice, for two reasons that are the
        # same reason. `ttnn.scatter` converts every non-ROW_MAJOR operand to ROW_MAJOR itself
        # (scatter.cpp:164/193), so handing it tiled index/src buys two `UntilizeWithUnpadding` rows
        # this path would otherwise pay inside the op; and the `[rows, 1, k] -> [1, 1, rows, k]`
        # reshape the scatter's operand contract needs is a *view* in ROW_MAJOR - identical memory
        # order - where in TILE it is a 13.8 us `ReshapeView` that gathers `rows` one-row tiles into
        # one. The first working version of this method did it in TILE and gave back most of what the
        # gate op saved: two ReshapeView rows plus three untilizes, against the `topk` + `softmax`
        # it removed. README section 2.3 has the figures and the profile they come from.
        wide = ttnn.to_layout(weights, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        idx = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        sel_w = ttnn.reshape(ttnn.slice(wide, [0, 0, 0], [rows, 1, k]), (1, 1, rows, k))
        # `ttnn.scatter` wants a uint32 index; the gate op emits uint16.
        sel_i = ttnn.typecast(ttnn.reshape(ttnn.slice(idx, [0, 0, 0], [rows, 1, k]), (1, 1, rows, k)), ttnn.uint32)
        # ROW_MAJOR scatter base, for the same reason the operands are ROW_MAJOR: `ttnn.scatter`
        # untilizes a tiled base on the way in and re-tilizes the result on the way out
        # (scatter.cpp:164/233). With every operand already ROW_MAJOR the op runs on its native
        # layout and this path pays exactly one tilize - the one below - instead of an untilize of
        # the 256-wide base plus that tilize.
        scattered = ttnn.scatter(self._gate_zeros[rows], dim=-1, index=sel_i, src=sel_w)
        trimmed = scattered
        if mode == "fused_gate_local":
            # Drop the dump column the non-local selections were written to. In ROW_MAJOR this is a
            # contiguous prefix of every row.
            trimmed = ttnn.slice(scattered, [0, 0, 0, 0], [1, 1, rows, self.cfg.num_experts])
        dense = ttnn.to_layout(trimmed, ttnn.TILE_LAYOUT)
        for tensor in (wide, idx, sel_w, sel_i, scattered):
            ttnn.deallocate(tensor)
        if trimmed is not scattered:
            ttnn.deallocate(trimmed)
        return dense

    # ---------------- sparse geometry ----------------
    def _sparse_cfg(self, role: str, tokens: int, active_bound: int):
        """The inherited sparse-matmul config, with the core target rescaled for expert parallelism.

        ``OptimizedMoE`` picks the routed matmul's core target as
        ``clamp(active_bound // SPARSE_CORES_PER_ACTIVE[role], SPARSE_MIN_CORES, SPARSE_MAX_CORES)``,
        and ``active_bound`` is ``min(num_experts_local, rows * top_k)``. The divisor encodes "how
        much work one core should be given per active expert" and was calibrated at ``E = 256``.
        Expert parallelism does not change ``Nt`` — the op still produces 32 and 64 output tiles — but
        it cuts the bound's ceiling from 256 to 64, which shrinks the core target exactly where the
        available parallelism did not change.

        Scaling the bound rather than re-declaring the rule keeps the parent's clamping, its
        realised-core reduction and its ``in0_block_w`` cap running exactly once, and in the parent.
        :data:`SPARSE_SCALE_CORES_BY_TP` has the resulting per-batch table and the measurements; the
        short version is that only ``down`` moves at prefill, both roles move at decode batch 2 and
        4, and batch 1 is unchanged.
        """
        # `self.tp > 1` as well as the flag: at tp=1 there is no expert parallelism to compensate
        # for, and a MultichipMoE built on a 1-device mesh must reproduce `OptimizedMoE` exactly.
        scale = SPARSE_CORES_PER_ACTIVE[role] if SPARSE_SCALE_CORES_BY_TP and self.tp > 1 else 1
        return super()._sparse_cfg(role, tokens, active_bound * scale)

    # ---------------- router ----------------
    def routing_weights(self, x):
        """Device-local dense routing weights ``[1, 1, tokens, num_experts_local]``.

        The top-8 selection is a **global** decision over all 256 experts, so the router weight is
        replicated and every device computes the identical 256-wide logits, top-k, softmax and
        scatter that the single-chip decoder computes. Nothing is communicated: replicating a
        ``[2048, 256]`` matmul is cheaper than an all-gather of its output plus the synchronisation
        it would impose, and it keeps the routing decision bit-identical across the mesh by
        construction rather than by a collective's accumulation order.

        Only the final narrowing is mesh-aware. ``self.w["expert_select"]`` is the ``[1, 1, 256, 64]``
        one-hot block of the 256x256 identity that this device's expert range occupies — the four
        shards concatenate to the identity — so one matmul turns the replicated 256-wide score
        vector into this device's 64-wide one. It is exact: each output sums 255 structural zeros
        and one bfloat16 score.
        """
        cfg = self.global_cfg
        rows = _physical_rows(x.shape)
        # `generalized_moe_gate` reads bfloat16 logits. The matmul that produces them keeps its HiFi4 /
        # float32-accumulate compute config either way - only the stored logits change dtype - so this
        # is a rounding of the router's OUTPUT, not a lower-fidelity router matmul.
        # A dict lookup, never a build: :meth:`prepare_decode_gate` uploads from host, which is
        # illegal under trace capture and is what `test_no_host_fallback_in_forward` forbids. Setup
        # (:meth:`MultichipDecoder.allocate_state`) prepares the row count the advertised batch needs;
        # a decode at an unprepared row count keeps the ``topk`` chain, which is slower and correct.
        fused = (
            ROUTER_MODE in _FUSED_GATE_MODES
            and self._decode_phase
            and self._gate_buffers.get(rows, (None,))[0] == ROUTER_MODE
        )
        router_ckc = self.dense_ckc
        if fused and ROUTER_DECODE_FIDELITY is not None:
            router_ckc = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ROUTER_DECODE_FIDELITY,
                math_approx_mode=False,
                fp32_dest_acc_en=self.policy.router_fp32_acc,
                packer_l1_acc=False,
            )
        logits = ttnn.linear(
            x,
            self.w["router"],
            dtype=ttnn.bfloat16 if fused else ttnn.float32,
            compute_kernel_config=router_ckc,
            program_config=self.proj_cfgs.get(
                "router",
                _physical_rows(x.shape),
                x.shape[-1],
                cfg.num_experts,
                fp32_acc=self.policy.router_fp32_acc,
                decode=self._decode_phase,
            ),
        )
        if fused:
            dense = self._fused_gate_dense(logits)
            ttnn.deallocate(logits)
            if ROUTER_MODE == "fused_gate_local":
                # Already device-local: the gate's index tensor carried the local mapping, so the
                # scatter wrote straight into this device's 64-wide block and there is nothing left
                # for `_select_local_experts` to narrow.
                return dense
        else:
            values, indices = ttnn.topk(logits, k=cfg.num_experts_per_tok, dim=-1, sorted=True)
            weights = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=self.dense_ckc)
            zeros = self._router_zeros_for(logits)
            dense = ttnn.scatter(zeros, dim=-1, index=indices, src=ttnn.typecast(weights, ttnn.bfloat16))
            ttnn.deallocate(logits)
            ttnn.deallocate(values)
            ttnn.deallocate(indices)
            ttnn.deallocate(weights)

        local = self._select_local_experts(dense)
        ttnn.deallocate(dense)
        return local

    def _select_local_experts(self, dense):
        """``[1, 1, tokens, 256]`` replicated -> ``[1, 1, tokens, 64]`` device-local."""
        if ROUTING_SELECT_MODE == "gather":
            index = self._select_index_for(dense)
            return ttnn.gather(dense, dim=-1, index=index)
        if ROUTING_SELECT_MODE != "select_matmul":
            raise ValueError(f"unknown ROUTING_SELECT_MODE {ROUTING_SELECT_MODE!r}")
        return ttnn.linear(
            dense,
            self.w["expert_select"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.dense_ckc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=self.proj_cfgs.get(
                "expert_select",
                _physical_rows(dense.shape),
                int(dense.shape[-1]),
                self.cfg.num_experts,
                fp32_acc=self.policy.router_fp32_acc,
                decode=self._decode_phase,
            ),
        )

    def _select_index_for(self, dense):
        """Persistent ``ttnn.gather`` index for the ``"gather"`` arm, keyed by the routed shape."""
        import torch

        shape = tuple(int(d) for d in dense.shape)
        key = shape
        cached = self.w.setdefault("expert_select_index", {}).get(key)
        if cached is None:
            e_local = self.cfg.num_experts
            base = torch.arange(e_local, dtype=torch.int32).reshape(1, 1, 1, e_local)
            per_device = [base + d * e_local for d in range(self.tp)]
            rows = shape[2]
            stacked = torch.cat([t.expand(1, 1, rows, e_local).contiguous() for t in per_device], dim=-1)
            cached = ttnn.from_torch(
                stacked,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_shard_mapper(self.device, dim=-1),
            )
            self.w["expert_select_index"][key] = cached
        return cached

    # ---------------- sparsity ----------------
    def _active_expert_mask(self, dense_routing, groups, valid_tokens):
        """``[1, groups, 1, E_local]`` ROW_MAJOR sparsity, floored so one local expert is always on.

        The body is :meth:`OptimizedMoE._active_expert_mask` — the same restriction of the reduction
        to the group's real rows, for the same reason — with :data:`MOE_MASK_FLOOR` adding a one-hot
        on local expert 0 to the *pre-threshold* sum. That changes which experts the sparse matmul
        visits and nothing else: the floored expert's routing score is whatever the router gave it,
        which is exactly zero when the global top-8 did not select it, and the score multiplies the
        down projection's input, so its contribution is exactly zero.
        """
        E = self.cfg.num_experts
        tokens = int(dense_routing.shape[-2])
        rows = tokens if valid_tokens is None else min(int(valid_tokens), tokens)
        if rows < tokens and groups != 1:
            rows = tokens
        source, owned = dense_routing, False
        if rows < tokens:
            source = ttnn.slice(dense_routing, [0, 0, 0, 0], [1, 1, rows, E])
            owned = True
        elif groups > 1:
            source = ttnn.reshape(dense_routing, [1, groups, tokens // groups, E])
        totals = ttnn.sum(source, dim=-2, keepdim=True)
        if owned:
            ttnn.deallocate(source)
        if self._mask_floor is not None:
            floored = ttnn.add(totals, self._mask_floor)
            ttnn.deallocate(totals)
            totals = floored
        mask = ttnn.to_layout(ttnn.gtz(totals), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(totals)
        return mask

    def _build_mask_floor(self):
        """The persistent ``[1, 1, 1, E_local]`` one-hot on local expert 0. Setup, not forward."""
        import torch

        host = torch.zeros(1, 1, 1, self.cfg.num_experts, dtype=torch.float32)
        host[..., 0] = 1.0
        return ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(self.device),
        )


# --------------------------------------------------------------------------------------
# decoder
# --------------------------------------------------------------------------------------
class MultichipDecoder(OptimizedDecoder):
    """One Ornith decoder layer sharded over the 4-chip Blackhole ring.

    Public contract, argument shapes and semantics are the optimized decoder's. The tensors handed
    in are **replicated** across the mesh and the tensor handed back is identical on every device,
    which is the layout a stack of these layers passes between them.
    """

    #: Conv1d channels per ``ttnn.conv1d`` call, per device. The single-chip stage uses 4096, the
    #: widest split its 8192-channel depthwise conv accepts; a quarter of that conv is 2048 channels
    #: and ``local_q_dim + local_k_dim`` is 1024, so 1024 keeps the same property the single-chip
    #: split had — block 0 is exactly ``[q | k]`` and block 1 is exactly ``v``, so ``v`` needs no
    #: slice — at the same two calls per forward.
    conv1d_channels: int = 1024

    def __init__(self, mesh_device, config, layer_idx, *, global_config, tp, **kwargs):
        super().__init__(mesh_device, config, layer_idx, **kwargs)
        self.proj_cfgs = _MultichipProjectionConfigs(mesh_device, self.policy)
        self.global_cfg = global_config
        self.tp = tp
        self.ccl_topology = DEFAULT_CCL_TOPOLOGY
        self.ccl_num_links = DEFAULT_CCL_NUM_LINKS
        #: Lazily built, then held for the layer's lifetime. See :meth:`_ccl_semaphores`.
        self._ccl_sems = None
        if mesh_device.get_num_devices() != tp:
            raise ValueError(
                f"MultichipDecoder was built for tp={tp} but the mesh has " f"{mesh_device.get_num_devices()} devices"
            )

    # ------------------------------------------------------------------ collectives
    def _all_reduce(self, tensor):
        """Sum ``tensor`` across the mesh, returning the full-width result on every device.

        Called exactly twice per layer — once on the token mixer's row-parallel output and once on
        the MoE's (routed partial sum + shared-expert row-parallel partial). Both operands are the
        residual-shaped ``[b, t, dim]``, which is the smallest tensor either half can be reduced on:
        reducing earlier would carry the un-projected 4096-wide head/intermediate stream instead.

        ``[b, t, dim]`` is not the smallest *physical* tensor, though, and at decode the difference
        is the whole cost. A tile's row axis is the sequence axis, so a decode activation
        ``[b, 1, dim]`` occupies ``b`` tile rows — ``b * 32`` physical rows — of which ``b`` carry
        data. :data:`CCL_COMPACT_ROWS` folds the batch into the row axis as ``[1, 1, b * t, dim]``
        first, which is exactly the layout the MoE call site already uses (``_block`` reshapes to
        ``[1, 1, tokens, dim]`` before the FF norm), so at batch 32 the two collectives in one
        forward stop differing by 32x in the rows they move.

        On the shipped path that fold now happens **upstream**: ``optimized_decoder``'s
        ``DECODE_COMPACT_ROWS`` folds the residual before the mixer runs, so the mixer operand arrives
        rank-4 like the MoE one always did and the guard below is false for both — same payload, two
        reshapes fewer per layer. Measured in
        ``doc/multichip_decoder/logs/probe_decode_batch.txt``, which also records the shapes each
        call site is handed rather than inferring them from this docstring.
        """
        if self.tp == 1:
            return tensor
        # The CCL payload dtype. `policy.ccl_dtype` is the datatype-sweep knob and takes precedence:
        # it casts *both* operands to one width, so this layer's two collectives stop differing in
        # dtype (bfloat16 out of the token mixer, `expert_act_dtype` out of the MoE). `None` - the
        # inherited and shipped value - leaves each as its producer emitted it, and then
        # :data:`CCL_CAST_BLOCKFLOAT` is the multichip stage's own narrower "widen block-float to
        # bfloat16" arm, kept because README limitation 9's null result is its control.
        ccl_dtype = self.policy.ccl_dtype
        if ccl_dtype is None and CCL_CAST_BLOCKFLOAT and tensor.dtype in _BLOCK_FLOAT_DTYPES:
            ccl_dtype = ttnn.bfloat16
        if ccl_dtype is not None and tensor.dtype != ccl_dtype:
            recast = ttnn.typecast(tensor, ccl_dtype, memory_config=tensor.memory_config())
            ttnn.deallocate(tensor)
            tensor = recast
        dims = [int(d) for d in tensor.shape]
        compact = (
            CCL_COMPACT_ROWS and len(dims) == 3 and _physical_rows(tensor.shape) > _align_up(dims[0] * dims[1], TILE)
        )
        if compact:
            packed = ttnn.reshape(tensor, [1, 1, dims[0] * dims[1], dims[2]])
            _free_unless_aliased(tensor, packed)
            reduced = self._all_reduce(packed)
            out = ttnn.reshape(reduced, dims)
            _free_unless_aliased(reduced, out)
            return out
        mode = CCL_MODE
        if mode == "auto":
            mode = AUTO_STACK_SUM_MODE if _physical_rows(tensor.shape) <= CCL_STACK_SUM_MAX_ROWS else "all_reduce"
        if mode == "all_reduce":
            out = ttnn.all_reduce(
                tensor,
                topology=self.ccl_topology,
                num_links=self.ccl_num_links,
                memory_config=tensor.memory_config(),
            )
        elif mode == "rs_ag":
            scattered = ttnn.reduce_scatter(
                tensor,
                dim=len(tensor.shape) - 1,
                topology=self.ccl_topology,
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = ttnn.all_gather(scattered, dim=len(tensor.shape) - 1, memory_config=tensor.memory_config())
            ttnn.deallocate(scattered)
        elif mode == "stack_sum":
            out = self._stack_sum(tensor)
        elif mode == "stack_sum_async":
            out = self._stack_sum(tensor, asynchronous=True)
        else:
            raise ValueError(f"unknown CCL_MODE {CCL_MODE!r}")
        ttnn.deallocate(tensor)
        return out

    def _stack_sum(self, tensor, *, asynchronous: bool = False):
        """All-reduce as ``all_gather`` onto a new leading axis plus a local ``ttnn.sum``.

        ``asynchronous`` spells the gather with ``ttnn.experimental.all_gather_async`` and the
        persistent semaphores :meth:`_ccl_semaphores` builds, instead of the deprecated
        ``ttnn.all_gather``. See :data:`CCL_MODE`: the deprecated op takes no semaphore, and this
        stage reproduced cross-device divergence under sustained traced replay with it.

        The gather axis has to be a **new** one: gathering on the existing leading dim of a rank-3
        ``[b, t, dim]`` activation would concatenate the batch entries and the sum would then reduce
        across users. A rank-3 input is therefore viewed as ``[1, b, t, dim]`` first — adding a
        leading extent-1 dim leaves the last two dims untouched, so ``ttnn.reshape`` returns a view
        rather than dispatching a relayout — and the result is viewed back.
        """
        dims = [int(d) for d in tensor.shape]
        rank3 = len(dims) == 3
        staged = ttnn.reshape(tensor, [1, *dims]) if rank3 else tensor
        if asynchronous:
            gather_sems, barrier_sem = self._ccl_semaphores()
            gathered = ttnn.experimental.all_gather_async(
                staged,
                dim=0,
                multi_device_global_semaphore=gather_sems,
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_topology,
                barrier_semaphore=barrier_sem,
            )
        else:
            gathered = ttnn.all_gather(staged, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        reduced = ttnn.sum(gathered, dim=0, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gathered)
        return ttnn.reshape(reduced, dims) if rank3 else reduced

    def _ccl_semaphores(self):
        """Persistent global semaphores for :meth:`_stack_sum`'s async spelling.

        Built by :meth:`allocate_state` — i.e. at setup, before any trace can be captured — and held
        for the layer's lifetime. ``ttnn.create_global_semaphore`` is a device-side allocation rather
        than a host write, but creating one inside a captured decode forward is the same class of
        trace-lifecycle hazard that :meth:`MultichipMoE.prepare_decode_gate` exists to avoid, and
        review round 2 pointed out that the first version of this method had exactly that shape: it
        built them lazily from ``_stack_sum``, which only worked because every harness happens to run
        an eager warm-up first. Two gather semaphores because
        ``all_gather_async_device_operation.cpp:57`` asserts exactly that, plus one barrier semaphore
        — which is the whole point of the arm.
        """
        if self._ccl_sems is None:
            grid = self.device.compute_with_storage_grid_size()
            crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
            self._ccl_sems = (
                [ttnn.create_global_semaphore(self.device, crs, 0) for _ in range(2)],
                ttnn.create_global_semaphore(self.device, crs, 0),
            )
        return self._ccl_sems

    # ------------------------------------------------------------------ block
    def _block(self, x, *, mode, logical_len=None, page_table=None, chunk_start_idx=0, current_pos=None, rot_idxs=None):
        """The optimized block with the two collectives inserted.

        Structurally identical to :meth:`OptimizedDecoder._block`; the only additions are the two
        :meth:`_all_reduce` calls. They sit **before** each residual add, so the residual stream
        that the next norm reads is the full-width replicated one and both RMSNorms stay local and
        exact. Placing them after the add instead would reduce the residual four times over.

        The inherited decode-residual fold (``DECODE_COMPACT_ROWS``) applies here too, and it
        *subsumes* :data:`CCL_COMPACT_ROWS`: the mixer's operand is now already rank-4, like the MoE's
        always was, so the fold-and-view-back pair inside :meth:`_all_reduce` fires for neither and the
        two collectives move the same 32 rows they moved before, two reshapes fewer.
        :data:`CCL_COMPACT_ROWS` is kept because it is what makes that true when the outer fold is
        switched off, and because a rank-3 operand from any other caller still needs it.
        """
        self._decode_phase = mode == "decode"
        b, t = int(x.shape[0]), int(x.shape[1])
        tokens = b * t
        residual = self._fold_decode_rows(x, tokens)
        folded = residual is not x
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

        # Row-parallel `o_proj` / `gdn_out` produce a partial sum over this device's heads.
        mixed = self._all_reduce(mixed)
        # `DECODE_RESIDUAL_MEMORY` only at decode: a prefill chunk's residual is 2048 x 2048 x 2 B and
        # belongs in DRAM, while a decode step's is one tile row for the whole batch once folded.
        residual_mem = DECODE_RESIDUAL_MEMORY if mode == "decode" else None
        h = ttnn.add(residual, mixed, memory_config=residual_mem, dtype=self.policy.residual_dtype)
        ttnn.deallocate(mixed)
        # Frees the fold, no-ops on `x` itself and on the batch-1 view of it.
        _free_unless_aliased(residual, x)

        padded_tokens = _align_up(tokens, TILE)
        ff_norm = self._norm(h, self.w["ff_norm"])
        ff_in = ff_norm if folded else ttnn.reshape(ff_norm, [1, 1, tokens, self.cfg.dim])
        if padded_tokens != tokens:
            ff_in = _pad_dim(ff_in, 2, padded_tokens - tokens)
        ff_out = self.moe.forward(
            ff_in, valid_tokens=tokens if padded_tokens != tokens else None, decode=self._decode_phase
        )
        ttnn.deallocate(ff_in)
        # `ff_out` is this device's 64 routed experts plus its shard of the shared expert; both are
        # partial sums of the same quantity, so one collective closes both.
        ff_out = self._all_reduce(ff_out)
        if padded_tokens != tokens:
            trimmed = ttnn.slice(ff_out, [0, 0, 0, 0], [1, 1, tokens, self.cfg.dim])
            ttnn.deallocate(ff_out)
            ff_out = trimmed
        if not folded:
            ff_out = ttnn.reshape(ff_out, [b, t, self.cfg.dim])
        out = ttnn.add(h, ff_out, memory_config=residual_mem, dtype=self.policy.residual_dtype)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        if folded:
            # Back to the layer's public `[b, t, dim]` contract; see `OptimizedDecoder._block`.
            unfolded = ttnn.reshape(out, [b, t, self.cfg.dim])
            _free_unless_aliased(out, unfolded)
            out = unfolded
        return out

    # ------------------------------------------------------------------ gated deltanet
    def _gdn_project(self, x):
        """The packed in-projection, sliced into ``(qkv, z, a, b)`` at the **local** widths.

        The per-device packing is ``[q | k | v | z | a | pad | b | pad]``: ``a`` and ``b`` are
        ``num_value_heads / tp`` = 8 columns each, a quarter tile, so each gets its own 32-column
        block whose trailing 24 columns are exact zeros in the weight. That keeps every slice offset
        tile-aligned; only the two gate slices end mid-tile, which ``ttnn.slice`` expresses as a
        logical width with tile padding.
        """
        cfg = self.cfg
        nv = cfg.linear_num_value_heads
        fused = self._proj_linear(x, self.w["gdn_in"], "gdn_in")
        qkv_end = cfg.conv_dim
        z_end = qkv_end + cfg.linear_v_dim
        a_start = z_end
        b_start = a_start + TILE
        qkv = _slice_last(fused, 0, qkv_end)
        z = _slice_last(fused, qkv_end, z_end)
        a = _slice_last(fused, a_start, a_start + nv)
        b = _slice_last(fused, b_start, b_start + nv)
        ttnn.deallocate(fused)
        return qkv, z, a, b

    def _conv1d_halves(self, padded_rm, phys_len, prepared):
        """:meth:`OptimizedDecoder._conv1d_halves` at :attr:`conv1d_channels`."""
        rm = ttnn.DRAM_MEMORY_CONFIG
        channels = self.conv1d_channels
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

    def _gdn_split_conv_output(self, activated):
        """:meth:`OptimizedDecoder._gdn_split_conv_output` at :attr:`conv1d_channels`."""
        cfg = self.cfg
        bounds = [0, cfg.linear_q_dim, cfg.linear_q_dim + cfg.linear_k_dim, cfg.conv_dim]
        if not isinstance(activated, list):
            fields = [_slice_last(activated, bounds[i], bounds[i + 1]) for i in range(3)]
            ttnn.deallocate(activated)
            return fields

        channels = self.conv1d_channels
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
                for piece in pieces:
                    ttnn.deallocate(piece)
        for blk, block in enumerate(activated):
            if not consumed[blk]:
                ttnn.deallocate(block)
        return fields

    def allocate_state(self, batch_size: int):
        """The inherited setup, with the conv1d weights prepared at :attr:`conv1d_channels`.

        Also the point where the fused router gate's persistent buffers are built: they need
        ``ttnn.from_torch``, so they cannot be created inside a decode forward that may be under trace
        capture, and this is the one setup entry point that knows the batch.
        """
        self.moe.prepare_decode_gate(_align_up(batch_size, TILE))
        # Only when a mode that uses them can be selected: three global semaphores per layer is
        # nothing here, but a 40-layer stack should not allocate 120 of them for a path it never
        # takes. The shipped CCL_MODE is `all_reduce`, which needs none.
        if CCL_MODE == "stack_sum_async" or (CCL_MODE == "auto" and AUTO_STACK_SUM_MODE == "stack_sum_async"):
            self._ccl_semaphores()
        if self.is_full_attention:
            return super().allocate_state(batch_size)
        if self.batch_idxs is not None:
            ttnn.deallocate(self.batch_idxs)
            self.batch_idxs = None
        for prepared in self.w.get("conv1d_weights", {}).values():
            for weight in prepared:
                ttnn.deallocate(weight)
        self.w["conv1d_weights"] = {}
        self.batch_size = batch_size
        self.w["conv1d_weights"] = _prepare_conv1d_weights_local(
            self.device,
            self.w.get("conv1d_host", []),
            self.cfg,
            self.prefill_chunk,
            batch_size,
            self.w["conv1d_compute_config"],
            self.conv1d_channels,
        )
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
        tp: int | None = None,
        **kwargs,
    ) -> "MultichipDecoder":
        """Build the layer from the same HF state dict, sharding every weight at upload time.

        ``tp`` defaults to the mesh's device count. The host does all the slicing once, here; the
        forward paths see only whole per-device tensors.
        """
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
        if isinstance(policy, str):
            if policy not in POLICIES:
                raise ValueError(f"unknown precision policy {policy!r}; known: {sorted(POLICIES)}")
            policy = POLICIES[policy]
        import torch

        tp = int(tp or mesh_device.get_num_devices())
        gcfg = OrnithDecoderConfig.from_hf_config(hf_config)
        cfg = local_decoder_config(gcfg, tp)
        max_context = int(max_context or gcfg.max_position_embeddings)
        if prefill_chunk % PREFILL_ALIGN:
            raise ValueError(f"prefill_chunk {prefill_chunk} must be a multiple of {PREFILL_ALIGN}")
        if moe_group_tokens % TILE:
            raise ValueError(f"moe_group_tokens {moe_group_tokens} must be a multiple of {TILE}")
        if prefill_chunk % page_block_size:
            raise ValueError(f"prefill_chunk {prefill_chunk} must be a multiple of page_block_size {page_block_size}")
        kind = gcfg.layer_kind(layer_idx)

        def upload(t, tensor_dtype=dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.as_tensor(
                t.to(torch.bfloat16).contiguous() if tensor_dtype == ttnn.bfloat16 else t.float().contiguous(),
                dtype=tensor_dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate_mapper(mesh_device),
            )

        def upload_sharded(parts, dim, tensor_dtype=dtype, layout=ttnn.TILE_LAYOUT):
            """Upload one per-device tensor each, concatenated along ``dim`` and sharded on it."""
            if len(parts) != tp:
                raise ValueError(f"expected {tp} shards, got {len(parts)}")
            joined = torch.cat([p.contiguous() for p in parts], dim=dim)
            return ttnn.as_tensor(
                joined.to(torch.bfloat16).contiguous()
                if tensor_dtype == ttnn.bfloat16
                else joined.float().contiguous(),
                dtype=tensor_dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_shard_mapper(mesh_device, dim=dim),
            )

        weights: dict = {}
        for key, dst in (("input_layernorm.weight", "attn_norm"), ("post_attention_layernorm.weight", "ff_norm")):
            weights[dst] = upload((state_dict[key].float() + 1.0).reshape(1, 1, 1, -1))

        rope = None
        if kind == "full_attention":
            n_heads, n_kv, head_dim = gcfg.n_heads, gcfg.n_kv_heads, gcfg.head_dim
            hpd = cfg.n_heads  # query heads per device
            perm = torch.tensor(
                _rope_head_permutation(head_dim, gcfg.rope_dim) if rope_mode == "full" else range(head_dim),
                dtype=torch.long,
            )
            qg = state_dict["self_attn.q_proj.weight"].float().reshape(n_heads, 2 * head_dim, gcfg.dim)
            q_heads = qg[:, :head_dim, :][:, perm, :]  # [n_heads, head_dim, dim]
            gate_heads = qg[:, head_dim:, :]
            k_heads = state_dict["self_attn.k_proj.weight"].float().reshape(n_kv, head_dim, gcfg.dim)[:, perm, :]
            v_heads = state_dict["self_attn.v_proj.weight"].float().reshape(n_kv, head_dim, gcfg.dim)

            attn_in_parts = []
            for d in range(tp):
                kv = kv_head_owner(n_kv, tp, d)
                rows = torch.cat(
                    [
                        q_heads[d * hpd : (d + 1) * hpd].reshape(hpd * head_dim, gcfg.dim),
                        k_heads[kv : kv + cfg.n_kv_heads].reshape(cfg.n_kv_heads * head_dim, gcfg.dim),
                        v_heads[kv : kv + cfg.n_kv_heads].reshape(cfg.n_kv_heads * head_dim, gcfg.dim),
                        gate_heads[d * hpd : (d + 1) * hpd].reshape(hpd * head_dim, gcfg.dim),
                    ],
                    dim=0,
                )
                attn_in_parts.append(rows.transpose(0, 1))
            weights["attn_in"] = upload_sharded(attn_in_parts, dim=1, tensor_dtype=policy.proj_dtype)
            # o_proj is row-parallel over the concatenated query heads, and the query heads are
            # contiguous per device, so this is exactly a dim-0 shard of the transposed weight.
            o_proj = state_dict["self_attn.o_proj.weight"].float().transpose(0, 1)
            weights["o_proj"] = upload_sharded(
                [o_proj[d * hpd * head_dim : (d + 1) * hpd * head_dim] for d in range(tp)],
                dim=0,
                tensor_dtype=policy.proj_dtype,
            )
            for src, dst in (("self_attn.q_norm.weight", "q_norm"), ("self_attn.k_norm.weight", "k_norm")):
                weights[dst] = upload(((state_dict[src].float() + 1.0)[perm]).reshape(1, 1, 1, -1))
            rope = OrnithFusedRope(
                mesh_device,
                gcfg,
                max_context=max_context,
                table_context=_align_up(max_context, prefill_chunk) + prefill_chunk,
                mode=rope_mode,
            )
        elif kind == "linear_attention":
            prefix = "linear_attn."
            nk, nv = gcfg.linear_num_key_heads, gcfg.linear_num_value_heads
            dk, dv = gcfg.linear_key_head_dim, gcfg.linear_value_head_dim
            nk_l, nv_l = cfg.linear_num_key_heads, cfg.linear_num_value_heads

            qkv_w = state_dict[prefix + "in_proj_qkv.weight"].float()
            z_w = state_dict[prefix + "in_proj_z.weight"].float()
            a_w = state_dict[prefix + "in_proj_a.weight"].float()
            b_w = state_dict[prefix + "in_proj_b.weight"].float()
            conv_w = state_dict[prefix + "conv1d.weight"].float()
            q_off, k_off, v_off = 0, nk * dk, 2 * nk * dk

            def head_rows(tensor, offset, heads_per_device, head_dim, device):
                lo = offset + device * heads_per_device * head_dim
                return tensor[lo : lo + heads_per_device * head_dim]

            gdn_in_parts, conv_parts, a_neg_parts, dt_parts = [], [], [], []
            a_neg = -state_dict[prefix + "A_log"].float().exp()
            dt_bias = state_dict[prefix + "dt_bias"].float()
            pad = torch.zeros(TILE - nv_l, gcfg.dim, dtype=torch.float32)
            for d in range(tp):
                local_qkv = torch.cat(
                    [
                        head_rows(qkv_w, q_off, nk_l, dk, d),
                        head_rows(qkv_w, k_off, nk_l, dk, d),
                        head_rows(qkv_w, v_off, nv_l, dv, d),
                    ],
                    dim=0,
                )
                rows = torch.cat(
                    [
                        local_qkv,
                        z_w[d * nv_l * dv : (d + 1) * nv_l * dv],
                        a_w[d * nv_l : (d + 1) * nv_l],
                        pad,
                        b_w[d * nv_l : (d + 1) * nv_l],
                        pad,
                    ],
                    dim=0,
                )
                gdn_in_parts.append(rows.transpose(0, 1))
                conv_parts.append(
                    torch.cat(
                        [
                            head_rows(conv_w, q_off, nk_l, dk, d),
                            head_rows(conv_w, k_off, nk_l, dk, d),
                            head_rows(conv_w, v_off, nv_l, dv, d),
                        ],
                        dim=0,
                    )
                )
                a_neg_parts.append(a_neg[d * nv_l : (d + 1) * nv_l])
                dt_parts.append(dt_bias[d * nv_l : (d + 1) * nv_l])

            weights["gdn_in"] = upload_sharded(gdn_in_parts, dim=1, tensor_dtype=policy.proj_dtype)
            out_proj = state_dict[prefix + "out_proj.weight"].float().transpose(0, 1)
            weights["gdn_out"] = upload_sharded(
                [out_proj[d * nv_l * dv : (d + 1) * nv_l * dv] for d in range(tp)],
                dim=0,
                tensor_dtype=policy.proj_dtype,
            )
            weights["gdn_norm"] = upload(state_dict[prefix + "norm.weight"].float().reshape(1, 1, 1, -1))
            local_conv = torch.cat(conv_parts, dim=0)  # [tp * conv_dim_local, 1, kernel]
            weights["conv_taps"] = [
                upload_sharded(
                    [conv_parts[d][:, 0, k].reshape(1, 1, -1) for d in range(tp)],
                    dim=-1,
                )
                for k in range(gcfg.linear_conv_kernel_dim)
            ]
            weights["conv1d_host"] = _conv1d_host_weights_local(local_conv, cfg, tp, cls.conv1d_channels)
            weights["conv1d_compute_config"] = _conv_compute_config(mesh_device.arch())
            weights["conv1d_weights"] = {}
            weights["A_neg"] = upload_sharded(
                [t.reshape(1, 1, -1) for t in a_neg_parts], dim=-1, tensor_dtype=ttnn.float32
            )
            weights["dt_bias"] = upload_sharded(
                [t.reshape(1, 1, -1) for t in dt_parts], dim=-1, tensor_dtype=ttnn.float32
            )
            from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import build_fused_const_tiles

            weights["gdn_const_tiles"] = build_fused_const_tiles(mesh_device, 32)
            weights["pos_ramp"] = upload(
                torch.arange(prefill_chunk, dtype=torch.float32).reshape(1, prefill_chunk, 1),
                tensor_dtype=ttnn.float32,
            )
        else:
            raise ValueError(f"unsupported layer kind {kind!r}")

        moe_weights = cls._load_moe_weights_sharded(
            mesh_device, gcfg, cfg, tp, state_dict, upload, upload_sharded, policy=policy
        )
        moe = MultichipMoE(
            mesh_device,
            cfg,
            moe_weights,
            global_config=gcfg,
            tp=tp,
            group_tokens=moe_group_tokens,
            policy=policy,
        )

        return cls(
            mesh_device,
            cfg,
            layer_idx,
            global_config=gcfg,
            tp=tp,
            weights=weights,
            moe=moe,
            rope=rope,
            max_context=max_context,
            page_block_size=page_block_size,
            prefill_chunk=prefill_chunk,
            policy=policy,
        )

    @staticmethod
    def _load_moe_weights_sharded(
        mesh_device, gcfg, cfg, tp, state_dict, upload, upload_sharded, prefix="mlp.", *, policy=DEFAULT_POLICY
    ):
        """MoE weights: 64 experts per device (EP), shared expert column/row-parallel (TP).

        The router stays replicated — the top-8 is a global decision — and ``expert_select`` is the
        256x256 identity sharded on its columns, i.e. exactly this device's expert block.
        """
        import torch

        def get(name):
            return state_dict[f"{prefix}{name}"]

        inter = gcfg.moe_intermediate_size
        fused = get("experts.gate_up_proj")
        if fused.shape[1] != 2 * inter:
            raise ValueError(f"experts.gate_up_proj dim1 {fused.shape[1]} != 2*{inter}")

        shared_inter = gcfg.shared_expert_intermediate_size
        local_shared = cfg.shared_expert_intermediate_size
        gate_w = get("shared_expert.gate_proj.weight").float().transpose(0, 1)  # [dim, shared_inter]
        up_w = get("shared_expert.up_proj.weight").float().transpose(0, 1)
        router_col = get("shared_expert_gate.weight").float().transpose(0, 1).reshape(gcfg.dim, 1)
        # The sigmoid gate is a scalar per token computed from the full-width replicated activation,
        # so it is replicated on every device and multiplies each device's partial output. That is
        # exact by linearity: sum_d sigmoid(r) * y_d == sigmoid(r) * sum_d y_d.
        shared_parts = [
            torch.cat(
                [
                    gate_w[:, d * local_shared : (d + 1) * local_shared],
                    up_w[:, d * local_shared : (d + 1) * local_shared],
                    router_col,
                    torch.zeros(gcfg.dim, TILE - 1, dtype=torch.float32),
                ],
                dim=1,
            )
            for d in range(tp)
        ]
        shared_down = get("shared_expert.down_proj.weight").float().transpose(0, 1)  # [shared_inter, dim]

        e_local = cfg.num_experts
        select = torch.eye(gcfg.num_experts, dtype=torch.float32).reshape(1, 1, gcfg.num_experts, gcfg.num_experts)

        return {
            "router": upload(
                get("gate.weight").float().transpose(0, 1).reshape(1, 1, gcfg.dim, gcfg.num_experts),
                policy.router_dtype,
            ),
            "expert_select": upload_sharded(
                [select[:, :, :, d * e_local : (d + 1) * e_local] for d in range(tp)],
                dim=-1,
                tensor_dtype=ttnn.bfloat16,
            ),
            "expert_gate_up": upload_sharded(
                [fused[d * e_local : (d + 1) * e_local].transpose(-2, -1).unsqueeze(0).float() for d in range(tp)],
                dim=1,
                tensor_dtype=policy.expert_gate_up_dtype,
            ),
            "expert_down": upload_sharded(
                [
                    get("experts.down_proj")[d * e_local : (d + 1) * e_local].transpose(-2, -1).unsqueeze(0).float()
                    for d in range(tp)
                ],
                dim=1,
                tensor_dtype=policy.expert_down_dtype,
            ),
            "shared_in": upload_sharded(
                [p.reshape(1, 1, gcfg.dim, 2 * local_shared + TILE) for p in shared_parts],
                dim=-1,
                tensor_dtype=policy.shared_dtype,
            ),
            "shared_down": upload_sharded(
                [
                    shared_down[d * local_shared : (d + 1) * local_shared].reshape(1, 1, local_shared, gcfg.dim)
                    for d in range(tp)
                ],
                dim=2,
                tensor_dtype=policy.shared_dtype,
            ),
        }


# --------------------------------------------------------------------------------------
# conv1d weight preparation at the local channel count
# --------------------------------------------------------------------------------------
def _conv1d_host_weights_local(conv_w, cfg, tp, channels):
    """Host ``ttnn.conv1d`` weights per ``channels`` block, **per device**.

    ``conv_w`` is every device's conv taps concatenated on the channel axis
    (``[tp * conv_dim_local, 1, kernel]``). Returns ``[block][device]`` host tensors, because
    ``ttnn.prepare_conv_weights`` takes no mesh mapper and has to be given one device's taps at a
    time; :func:`_prepare_conv1d_weights_local` reassembles the prepared buffers into one
    mesh-sharded tensor per block.
    """
    kernel = cfg.linear_conv_kernel_dim
    local = cfg.conv_dim
    if local % channels:
        return []
    blocks = local // channels
    return [
        [
            ttnn.from_torch(
                conv_w[d * local + idx * channels : d * local + (idx + 1) * channels].reshape(channels, 1, 1, kernel),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            for d in range(tp)
        ]
        for idx in range(blocks)
    ]


def _prepare_conv1d_weights_local(mesh_device, host_weights, cfg, prefill_chunk, batch, compute_config, channels):
    """:func:`~...optimized_decoder._prepare_conv1d_weights` at ``channels``, sharded across the mesh.

    ``ttnn.prepare_conv_weights`` accepts no mesh mapper (``conv2d_nanobind.cpp`` binds
    ``device`` and no distribution argument), and a depthwise conv's taps differ per device here, so
    each device's block is prepared on its own — which lands it *replicated* — read back, and the
    four host layouts are concatenated on dim 0 and re-uploaded sharded on dim 0. That is
    layout-agnostic: whatever shape ``S`` the op produces, ``cat`` gives ``[tp*S0, S1...]`` and the
    dim-0 shard hands device ``d`` exactly the buffer a single device running that conv would use.
    The round trip is lossless because ``conv_config.weights_dtype`` is bfloat16.

    Every failure mode still degrades to the FIR path rather than raising, exactly as the
    single-chip version does: a length absent from the returned dict is one the layer will run
    through the FIR form.
    """
    kernel = cfg.linear_conv_kernel_dim
    if not host_weights:
        return {}
    import torch

    conv_cfg = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)

    def prepare_block(per_device_hosts, length):
        """One mesh-sharded prepared weight for one ``channels`` block."""
        shards = []
        layout = dtype = None
        for host in per_device_hosts:
            replicated = ttnn.prepare_conv_weights(
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
            layout, dtype = replicated.layout, replicated.dtype
            lead = int(replicated.shape[0])
            whole = ttnn.to_torch(replicated, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0))
            shards.append(whole[:lead].clone())
            ttnn.deallocate(replicated)
        return ttnn.from_torch(
            torch.cat(shards, dim=0),
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_shard_mapper(mesh_device, dim=0),
        )

    prepared: dict[int, list] = {}
    for phys in range(PREFILL_ALIGN, prefill_chunk + 1, PREFILL_ALIGN):
        length = phys + kernel - 1
        try:
            prepared[phys] = [prepare_block(block, length) for block in host_weights]
        except Exception:  # noqa: BLE001 - an unpreparable length falls back to the FIR path
            _drop_prepared(prepared, phys)
            continue
        probe = None
        try:
            probe = ttnn.from_torch(
                torch.zeros(batch, length, 1, channels, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate_mapper(mesh_device),
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
            if probe is not None:
                ttnn.deallocate(probe)
    return prepared

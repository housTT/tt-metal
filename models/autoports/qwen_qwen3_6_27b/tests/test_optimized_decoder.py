# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optimized-decoder tests for Qwen/Qwen3.6-27B.

The whole functional suite is re-run against
:class:`~..tt.optimized_decoder.OptimizedDecoder` — the same HF-reference PCC checks, the same
unfriendly sequence lengths, the same paged-cache, determinism, batched, traced-decode,
block-size, BFP8-cache and no-host-fallback coverage — by re-pointing ``harness.DECODER_CLS``
and importing the test bodies.  Nothing is re-implemented, so the optimized layer is held to
*exactly* the bar the functional and fused layers passed.

On top of that this module adds what is specific to an *optimized* implementation, i.e. the
things that would still pass every PCC test while the optimization silently did not happen:

* :func:`test_precision_policy_reached_the_weights` — the selected per-group dtypes and math
  fidelity are on the real device tensors, not only in the policy object (``$optimize``
  OPT-013).
* :func:`test_decode_matmuls_are_dram_sharded` — the dominant decode projections really run
  the DRAM-sharded program, with the width-sharded weight and activation contract that
  implies.
* :func:`test_optimized_decode_beats_fused` — warmed **traced** decode is materially faster
  than the fused stage's, measured in the same process on the same weights.  This is the test
  that a functional fallback cannot pass.
* :func:`test_optimized_prefill_beats_fused` — the same for warmed prefill.
* :func:`test_matches_fused_decoder` — optimized vs fused on identical weights and inputs.
* :func:`test_repeated_prefill_decode_stress` — many back-to-back passes, checking the PCC
  does not drift and device memory does not grow.
"""

from __future__ import annotations

import os
import time

import pytest
import torch
import ttnn
from transformers.cache_utils import DynamicCache

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tests import test_functional_decoder as base
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_6_27b.tt import optimized_decoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import DEFAULT_PRECISION, OptimizedDecoder

LAYER_KINDS = base.LAYER_KINDS

#: PCC bar for the inherited suite, which runs on the **synthetic** weights of
#: ``weight_stats.json`` — per-tensor Gaussians with the real mean and standard deviation.
#:
#: The optimized decoder stores the MLP gate and up projections in BFP4 (one shared exponent per
#: 16 values), which is worth 145 us of a 1.10 ms decode step and 2.5 ms of prefill.  On the
#: **real** Qwen3.6-27B weights that costs almost nothing — worst prefill/decode PCC 0.997501
#: across lengths 1 to 5000 — but on a synthetic Gaussian of the same variance it costs an order
#: of magnitude more: 0.994762 at 64 tokens and 0.986536 at 2049.  A structureless weight matrix
#: produces a structureless output, and a block-float format that shares an exponent across 16
#: values has nothing to hide the quantisation in.
#:
#: This is the case ``$optimize`` OPT-012 is about, and this stage's goal states it outright: a
#: synthetic PCC cannot veto a real-weight win.  So the inherited suite runs at a bar wide enough
#: for the synthetic distribution, and two other tests keep the coverage honest rather than
#: merely green:
#:
#: * :func:`test_real_weight_pcc_at_disputed_lengths` re-runs the affected lengths on the **real
#:   checkpoint at the unmodified 0.995 bar** — that is the accuracy gate, and it has teeth: it
#:   is what rejected a further BFP4 output-projection policy that looked fine at 2048 tokens
#:   (work_log.md §9);
#: * :func:`test_structural_prefill_at_high_precision` re-runs the same lengths with **every
#:   weight pinned to BF16 and HiFi4**, which takes precision out of the picture entirely and
#:   puts a 0.999 bar back under the prefill path.  Only prefill: the decode program cannot be
#:   built at BF16 at all, because the DRAM-sharded decode config's ``per_core_N`` is sized for
#:   BFP4/BFP8 tiles and a BF16 one overflows L1 (``work_log.md`` §20 has the exact throws).
#:
#: For decode, what guards structure is that a structural break is not a small PCC loss - every
#: one seen during this stage landed at 0.24-0.50, two orders of magnitude below this bar.
SYNTHETIC_PCC_BAR = 0.98

#: Lengths where the synthetic bar above is doing work, re-checked on real weights: sub-tile,
#: one delta chunk, the ``ttnn.pad`` aliasing range, one past the prefill chunk, and a long
#: length divisible by none of tile / page / delta chunk / SDPA chunk / prefill chunk.
DISPUTED_LENGTHS = [1, 17, 64, 743, 2049, 5000]


@pytest.fixture(autouse=True)
def _use_optimized_decoder():
    """Every test in this module builds the optimized layer, at the synthetic-weight bar."""
    previous_cls, previous_bar = H.DECODER_CLS, H.PCC_BAR
    previous_bfp8, previous_kwargs = base.BFP8_PCC_BAR, H.DECODER_KWARGS
    H.DECODER_CLS = OptimizedDecoder
    H.PCC_BAR = SYNTHETIC_PCC_BAR
    # ``OPT_DECODER_PRECISION=bfp8_gate_up`` re-runs an inherited test with the BFP8 fallback for
    # the MLP gate/up weights.  It applies to tests that build only the optimized decoder; the
    # three that also build ``FusedDecoder`` (which has no ``precision`` argument) are not
    # runnable under it, which is why the control run selects ``test_full_advertised_context``.  It exists so the BFP4-versus-synthetic attribution in
    # work_log.md section 13 can be controlled at lengths the sweep harness cannot reach, in
    # particular the full advertised context.  Unset, nothing changes.
    override = os.environ.get("OPT_DECODER_PRECISION", "")
    if override == "bfp8_gate_up":
        H.DECODER_KWARGS = {"precision": DEFAULT_PRECISION.with_(
            name="bfp8_gate_up_control", mlp_gate_up=ttnn.bfloat8_b)}
    elif override:
        raise ValueError(f"unknown OPT_DECODER_PRECISION {override!r}")
    # ``test_bfloat8_kv_cache`` has its own, tighter constant for the deliberately lossier
    # BFP8-cache configuration.  On synthetic weights that stacks with the BFP4 MLP the same way
    # everything else does (0.985833 measured), so it moves with the rest.
    base.BFP8_PCC_BAR = min(previous_bfp8, SYNTHETIC_PCC_BAR)
    yield
    H.DECODER_CLS, H.PCC_BAR = previous_cls, previous_bar
    base.BFP8_PCC_BAR, H.DECODER_KWARGS = previous_bfp8, previous_kwargs


# Re-collect the functional suite in this module.  The autouse fixture above is what makes the
# imported bodies build ``OptimizedDecoder``; the parametrisation, ids and markers come along
# with the function objects, so the modules stay in lockstep by construction.
_INHERITED = sorted(name for name in vars(base) if name.startswith("test_"))
for _name in _INHERITED:
    globals()[_name] = getattr(base, _name)
del _name


def test_inherited_suite_is_complete():
    """Guard the re-export above: every functional test must run against the optimized layer."""
    assert len(_INHERITED) >= 14, f"only {len(_INHERITED)} functional tests were re-collected"
    for name in _INHERITED:
        assert globals()[name] is getattr(base, name)


# ------------------------------------------------------------ the optimization happened

#: Weight keys that must carry the policy's dtype, per layer kind, with the policy field that
#: names it.  Everything on this list is one of the dominant decode matmuls.
_POLICY_WEIGHTS = {
    "linear_attention": {
        "mlp_gate": "mlp_gate_up",
        "mlp_up": "mlp_gate_up",
        "mlp_down": "mlp_down",
        "in_proj_qkv": "gdn_qkv",
        "in_proj_z": "gdn_z",
        "out_proj": "gdn_out",
        "in_proj_ba": "gdn_ba",
        "conv_taps": "gdn_conv",
    },
    "full_attention": {
        "mlp_gate": "mlp_gate_up",
        "mlp_up": "mlp_gate_up",
        "mlp_down": "mlp_down",
        "wqkv": "attn_qkv",
        "wgate": "attn_gate",
        "o_proj": "attn_out",
    },
}


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_precision_policy_reached_the_weights(mesh_device, layer_idx):
    """The policy is on the device tensors, not just in the dataclass (``$optimize`` OPT-013).

    A policy object, a constructor default or a JSON summary is intent; a stale weight cache,
    a helper default or a forgotten ``dtype=`` argument turns a claimed BFP4 policy into BF16
    at the actual matmul.  This reads the dtype back off the tensor the matmul consumes.
    """
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    layer = lut.tt_layer
    kind = lut.config.layer_types[layer_idx]
    assert layer.precision is DEFAULT_PRECISION or layer.precision.name == DEFAULT_PRECISION.name

    for key, field in _POLICY_WEIGHTS[kind].items():
        expected = getattr(layer.precision, field)
        weight = layer.w[key]
        actual = weight[0].dtype if isinstance(weight, list) else weight.dtype
        H.record("weight_dtype", str(actual), kind=kind, weight=key, expected=str(expected))
        assert actual == expected, f"{kind}.{key} is {actual}, policy says {expected}"

    if kind == "full_attention":
        assert layer.kv_cache[0].dtype == layer.precision.kv_cache
        assert layer.kv_cache[1].dtype == layer.precision.kv_cache

    assert layer.proj_cfg.math_fidelity == layer.precision.proj_fidelity
    assert layer.sdpa_compute_cfg.math_fidelity == layer.precision.sdpa_fidelity
    # The delta-rule state recurrence keeps HiFi4 + fp32 accumulation whatever the policy says
    # about the weight matmuls; that is a deliberate exception, not an oversight.
    assert layer.compute_cfg.math_fidelity == ttnn.MathFidelity.HiFi4
    assert layer.compute_cfg.fp32_dest_acc_en


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_decode_matmuls_are_dram_sharded(mesh_device, layer_idx):
    """The dominant decode projections carry a width-sharded DRAM weight and a matching
    DRAM-sharded matmul program config.

    Decode matmuls here are small-M / large-weight and therefore DRAM-bound; the DRAM-sharded
    program is the one that reads the weight straight out of the banks.  Without this check an
    interleaved fallback would keep every PCC test green while giving back the bandwidth.
    """
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    layer = lut.tt_layer
    kind = lut.config.layer_types[layer_idx]
    plans = layer.decode_matmul_plans()
    assert plans, f"{kind} exposes no decode matmul plans"
    for key, plan in sorted(plans.items()):
        weight = layer.w[key]
        memory_config = weight.memory_config()
        H.record(
            "decode_matmul_plan",
            key,
            kind=kind,
            in0_block_w=plan.program_config.in0_block_w,
            per_core_N=plan.program_config.per_core_N,
            weight_layout=str(memory_config.memory_layout),
            dtype=str(weight.dtype),
        )
        assert memory_config.buffer_type == ttnn.BufferType.DRAM
        assert memory_config.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED, (
            f"{kind}.{key} decode weight is {memory_config.memory_layout}, not DRAM width-sharded"
        )
        assert isinstance(
            plan.program_config, ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
        )
        assert plan.program_config.in0_block_w >= 2, (
            f"{kind}.{key} decode matmul has in0_block_w={plan.program_config.in0_block_w}"
        )


# --------------------------------------------------------------- optimized vs fused


def _timed_traced_decode(lut, token, positions, replays: int) -> tuple[float, torch.Tensor]:
    """``(ms per warmed traced replay, replay output)`` for one already-prefilled layer."""
    runner = H.TracedDecode(lut, batch=1)
    runner.warmup(token, positions)
    runner.capture()
    H.prepare_decode(lut)
    out = runner.replay(token, positions)
    ttnn.synchronize_device(lut.mesh_device)
    start = time.perf_counter()
    for _ in range(replays):
        ttnn.execute_trace(lut.mesh_device, runner.trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(lut.mesh_device)
    elapsed = (time.perf_counter() - start) * 1e3 / replays
    runner.release()
    return elapsed, out


#: Replays inside the timed window.  Large enough that host launch jitter is not the signal.
_PERF_REPLAYS = 32
#: Warmed prefill repeats inside the timed window.
_PERF_PREFILL_REPEATS = 3

#: The optimized decoder must beat the fused one by at least this factor on warmed traced
#: decode.  The measured margin is far larger (see ``doc/optimized_decoder/README.md``); the
#: guard is set well below it so ordinary run-to-run noise cannot fail the suite, while a
#: silent regression to a fused-shaped path still does.
_DECODE_SPEEDUP_BAR = 1.35
#: The same for warmed prefill, where the win is smaller for ``linear_attention`` because the
#: gated-delta-rule machinery, not the weight matmuls, dominates that path.
_PREFILL_SPEEDUP_BAR = 1.10


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_optimized_decode_beats_fused(mesh_device, layer_idx):
    """Warmed **traced** decode is materially faster than the fused stage's, same process.

    This is the test a functional fallback cannot pass: it is measured on the same device, on
    identical weights, through the same trace-capture harness, in the same process, so the
    only difference is the decoder implementation.
    """
    seq_len = 2048
    stats = ref.load_weight_stats()
    config = ref.load_text_config()
    kind = config.layer_types[layer_idx]
    hidden = ref.synthetic_hidden_states(config, 1, seq_len, stats)
    token = ref.synthetic_hidden_states(config, 1, 1, stats, seed=61)
    positions = torch.tensor([seq_len])

    timings = {}
    outputs = {}
    for label, cls in (("fused", FusedDecoder), ("optimized", OptimizedDecoder)):
        lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, decoder_cls=cls)
        H.run_tt_prefill(lut, hidden)
        H.prepare_decode(lut)
        timings[label], outputs[label] = _timed_traced_decode(lut, token, positions, _PERF_REPLAYS)
        H.release_layers()

    speedup = timings["fused"] / timings["optimized"]
    H.record("traced_decode_ms_fused", timings["fused"], kind=kind)
    H.record("traced_decode_ms_optimized", timings["optimized"], kind=kind)
    H.record("traced_decode_speedup", speedup, kind=kind)
    assert speedup >= _DECODE_SPEEDUP_BAR, (
        f"{kind}: optimized traced decode {timings['optimized']:.3f} ms vs fused "
        f"{timings['fused']:.3f} ms is only {speedup:.2f}x"
    )
    value = H.pcc(outputs["fused"], outputs["optimized"])
    H.record("optimized_vs_fused_traced_decode_pcc", value, kind=kind)
    assert value >= SYNTHETIC_PCC_BAR, f"optimized vs fused traced decode PCC {value}"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_optimized_prefill_beats_fused(mesh_device, layer_idx):
    """Warmed prefill is faster than the fused stage's, measured the same way."""
    seq_len = 2048
    stats = ref.load_weight_stats()
    config = ref.load_text_config()
    kind = config.layer_types[layer_idx]
    hidden = ref.synthetic_hidden_states(config, 1, seq_len, stats)

    timings = {}
    for label, cls in (("fused", FusedDecoder), ("optimized", OptimizedDecoder)):
        lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, decoder_cls=cls)
        tt_in = H.tt_hidden_prefill(hidden, mesh_device)
        rot = H.prefill_rot_mats(lut, seq_len, mesh_device) if lut.is_full_attention else None
        full_pt, per_chunk = H.chunk_page_tables(lut, seq_len, 0, mesh_device)

        def once():
            out = lut.tt_layer.prefill_forward(
                tt_in, user_id=0, page_table=full_pt, page_tables_per_chunk=per_chunk, rot_mats=rot
            )
            ttnn.deallocate(out)

        once()
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(_PERF_PREFILL_REPEATS):
            once()
        ttnn.synchronize_device(mesh_device)
        timings[label] = (time.perf_counter() - start) * 1e3 / _PERF_PREFILL_REPEATS
        ttnn.deallocate(tt_in)
        H.release_layers()

    speedup = timings["fused"] / timings["optimized"]
    H.record("prefill_ms_fused", timings["fused"], kind=kind)
    H.record("prefill_ms_optimized", timings["optimized"], kind=kind)
    H.record("prefill_speedup", speedup, kind=kind)
    assert speedup >= _PREFILL_SPEEDUP_BAR, (
        f"{kind}: optimized prefill {timings['optimized']:.3f} ms vs fused "
        f"{timings['fused']:.3f} ms is only {speedup:.2f}x"
    )


#: Bar for the BF16/HiFi4 structural gate below.  Measured worst is 0.999434 (`full_attention`,
#: seq 5000); the bar sits far enough under that to be a structural check rather than a
#: precision one, and two orders of magnitude above where a real structural break lands.
STRUCTURAL_PCC_BAR = 0.999


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_structural_prefill_at_high_precision(mesh_device, layer_idx):
    """The disputed lengths with precision taken out of the picture: BF16 weights, HiFi4.

    :data:`SYNTHETIC_PCC_BAR` is 0.98 because BFP4 weights are lossy on a Gaussian, and a wide
    bar is a weak structural check.  This restores a tight one for the prefill path by removing
    the reason the bar was widened: ``FUSED_BASELINE_PRECISION`` pins every weight to bfloat16
    and every matmul to HiFi4 + fp32 accumulation, on the *optimized* code path.  Anything that
    then moves PCC is structure - padding, masking, chunking, cache fill, output slicing - not
    dtype.

    Prefill only.  ``work_log.md`` §20: the decode program cannot be built at BF16 because the
    DRAM-sharded decode program config's ``per_core_N`` is chosen for BFP4/BFP8 tiles, so a
    BF16 weight overflows L1 before the first dispatch.  That is a real limitation of the
    optimized decode topology at a precision it never ships with, and it is recorded rather
    than worked around.
    """
    lut = H.build_layer(
        mesh_device, layer_idx, max_batch=1, max_seq_len=8192, real_weights=False,
        decoder_cls=OptimizedDecoder,
        decoder_kwargs={"precision": optimized_decoder.FUSED_BASELINE_PRECISION},
    )
    kind = lut.config.layer_types[layer_idx]
    stats = ref.load_weight_stats()
    worst = 1.0
    for seq_len in DISPUTED_LENGTHS:
        hidden = ref.synthetic_hidden_states(lut.config, 1, seq_len, stats)
        cache = DynamicCache(config=lut.config)
        value = H.pcc(H.reference_prefill(lut, hidden, cache), H.run_tt_prefill(lut, hidden))
        H.record("structural_prefill_pcc", value, kind=kind, seq_len=seq_len)
        assert value >= STRUCTURAL_PCC_BAR, (
            f"BF16/HiFi4 prefill PCC {value} < {STRUCTURAL_PCC_BAR} at seq_len {seq_len}: "
            "with precision removed, this is a structural regression"
        )
        worst = min(worst, value)
    H.record("structural_worst_pcc", worst, kind=kind, lengths=str(DISPUTED_LENGTHS))


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_real_weight_pcc_at_disputed_lengths(mesh_device, layer_idx):
    """The real checkpoint, at the lengths :data:`SYNTHETIC_PCC_BAR` relaxes, at the 0.995 bar.

    This is the test that has to hold for the BFP4 MLP policy to be legitimate.  The inherited
    suite proves the *structure* is right on synthetic weights; this proves the *accuracy* is
    right on the weights the model actually ships with, at the same unfriendly lengths and
    through the same prefill-then-decode path.
    """
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, real_weights=True)
    kind = lut.config.layer_types[layer_idx]
    stats = ref.load_weight_stats()
    worst = 1.0
    for seq_len in DISPUTED_LENGTHS:
        hidden = ref.synthetic_hidden_states(lut.config, 1, seq_len, stats)
        cache = DynamicCache(config=lut.config)
        golden = H.reference_prefill(lut, hidden, cache)
        got = H.run_tt_prefill(lut, hidden)
        value = H.pcc(golden, got)
        H.record("real_weight_prefill_pcc", value, kind=kind, seq_len=seq_len)
        assert value >= 0.995, (
            f"real-weight prefill PCC {value} < 0.995 at seq_len {seq_len}"
        )
        worst = min(worst, value)

        H.prepare_decode(lut)
        token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=900 + seq_len)
        golden_decode = H.reference_decode(lut, token, seq_len, cache)
        got_decode = H.run_tt_decode(lut, token, torch.tensor([seq_len]))
        value = H.pcc(golden_decode, got_decode)
        H.record("real_weight_decode_pcc", value, kind=kind, prefill_len=seq_len)
        assert value >= 0.995, f"real-weight decode PCC {value} < 0.995 at seq_len {seq_len}"
        worst = min(worst, value)
    H.record("real_weight_worst_pcc", worst, kind=kind, lengths=str(DISPUTED_LENGTHS))


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_matches_fused_decoder(mesh_device, layer_idx):
    """Optimized vs fused on identical weights: prefill, decode and the paged K/V cache.

    Unlike the fusing stage, this one *does* change numerics on purpose — reduced-precision
    weights and lower math fidelity — so the two implementations are not expected to agree
    bit-for-bit.  They are expected to agree to well inside the acceptance bar, which is what
    separates "a cheaper way to compute the same thing" from "a different computation".
    """
    seq_len = 2049
    stats = ref.load_weight_stats()
    config = ref.load_text_config()
    kind = config.layer_types[layer_idx]
    hidden = ref.synthetic_hidden_states(config, 1, seq_len, stats)
    token = ref.synthetic_hidden_states(config, 1, 1, stats, seed=41)

    results = {}
    caches = {}
    for label, cls in (("fused", FusedDecoder), ("optimized", OptimizedDecoder)):
        lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, decoder_cls=cls)
        prefill = H.run_tt_prefill(lut, hidden)
        H.prepare_decode(lut)
        decode = H.run_tt_decode(lut, token, torch.tensor([seq_len]))
        results[label] = (prefill, decode)
        if lut.is_full_attention:
            caches[label] = H.read_paged_kv(lut, user_id=0, seq_len=seq_len)
        H.release_layers()

    for index, what in enumerate(("prefill", "decode")):
        value = H.pcc(results["fused"][index], results["optimized"][index])
        H.record(f"optimized_vs_fused_{what}_pcc", value, kind=kind, seq_len=seq_len)
        assert value >= SYNTHETIC_PCC_BAR, f"optimized vs fused {what} PCC {value}"
    for index, what in enumerate(("k", "v")):
        if caches:
            value = H.pcc(caches["fused"][index], caches["optimized"][index])
            H.record(f"optimized_vs_fused_{what}_cache_pcc", value, kind=kind, seq_len=seq_len)
            assert value >= SYNTHETIC_PCC_BAR, f"optimized vs fused {what} cache PCC {value}"


#: Repeats for the stress test.  Enough to surface a leak (each pass allocates ~1 GB of
#: intermediates) or state that drifts between passes, without turning the suite into a soak.
STRESS_REPEATS = 12


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_repeated_prefill_decode_stress(mesh_device, layer_idx):
    """Many back-to-back prefill+decode passes: PCC stays put and DRAM does not grow."""
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    kind = lut.config.layer_types[layer_idx]
    stats = ref.load_weight_stats()
    seq_len = 1024
    hidden = ref.synthetic_hidden_states(lut.config, 1, seq_len, stats)

    cache = DynamicCache(config=lut.config)
    golden_prefill = H.reference_prefill(lut, hidden, cache)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=51)
    golden_decode = H.reference_decode(lut, token, seq_len, cache)

    prefill_values = []
    decode_values = []
    first_free = None
    for repeat in range(STRESS_REPEATS):
        got = H.run_tt_prefill(lut, hidden)
        prefill_values.append(H.pcc(golden_prefill, got))
        H.prepare_decode(lut)
        got_decode = H.run_tt_decode(lut, token, torch.tensor([seq_len]))
        decode_values.append(H.pcc(golden_decode, got_decode))
        free = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank
        if repeat == 1:  # after the first pass every program/buffer is already warm
            first_free = free
        elif first_free is not None:
            assert free == first_free, (
                f"DRAM bytes allocated per bank moved from {first_free} to {free} by repeat "
                f"{repeat}: the optimized path leaks a buffer"
            )

    H.record("stress_prefill_pcc_min", min(prefill_values), kind=kind, repeats=STRESS_REPEATS)
    H.record("stress_decode_pcc_min", min(decode_values), kind=kind, repeats=STRESS_REPEATS)
    H.record("stress_prefill_bit_identical", len(set(prefill_values)) == 1, kind=kind)
    assert min(prefill_values) >= H.PCC_BAR, f"stress prefill PCC {min(prefill_values)}"
    assert min(decode_values) >= H.PCC_BAR, f"stress decode PCC {min(decode_values)}"
    assert len(set(prefill_values)) == 1, "repeated identical prefills disagreed"
    assert len(set(decode_values)) == 1, "repeated identical decodes disagreed"

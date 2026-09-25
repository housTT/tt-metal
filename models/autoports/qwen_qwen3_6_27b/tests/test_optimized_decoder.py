# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optimized-decoder tests for Qwen/Qwen3.6-27B.

Every test here builds :class:`OptimizedDecoder`.  The suite is the fused suite's contract re-run
against the optimized precision policy, memory layout and program configs - same HF reference,
same ``PCC_BAR``, same sequence-length coverage, same paged-cache, state, determinism, traced and
stress checks - plus the tests that only make sense for an optimization stage:

``test_optimized_config_is_the_shipped_one``
    the layer's own :meth:`OptimizedDecoder.config_summary` agrees with the policy, and the
    weights on the device really carry the policy's dtypes and the DRAM width-sharded memory
    config - so a lazily-cached or wrongly-rebuilt weight cannot silently make a "BFP4" policy run
    in bfloat16 (OPT-013);
``test_decode_matmuls_are_dram_sharded``
    the measured decode path really dispatches the DRAM-sharded program config for every dominant
    projection, and its activations really are width-sharded L1;
``test_decode_stream_stays_in_l1``
    no decode activation is interleaved in DRAM between the first norm and the last residual add
    (OPT-003);
``test_no_relayout_or_host_ops_in_measured_decode``
    the reshard count of the measured decode is exactly the set each op contract forces, and it is
    *lower* than the fused stage's;
``test_optimized_matches_fused``
    put the fused stage's policy and layout back and the optimized code reproduces the fused
    layer's outputs, which is what makes the before/after comparison a like-for-like one;
``test_optimized_beats_fused_traced_decode``
    the shipped configuration is faster than the fused stage's in the same harness, at both layer
    kinds and both measured batches;
``test_capacity_did_not_shrink``
    the optimized layer's persistent device bytes, against the fused layer's, so the context
    contract's capacity claim is measured rather than asserted.

Both HF layer kinds are covered with the real model config shapes: layer 0 (``linear_attention``)
and layer 3 (``full_attention``).

The op-counting, layout-tracing and DRAM-accounting plumbing is imported from the fused suite
rather than copied: those context managers are the *measurement instrument*, and two copies of an
instrument that must agree is how the fused stage's own reviews found stale numbers.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import sys

import pytest
import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tests.test_fused_decoder import (
    LONG_PCC_SEQ_LEN,
    LONG_PROMPT,
    LONG_SEGMENT,
    LONG_TAIL,
    PAD_ALIAS_SEQ_LENS,
    SCALE_TOLERANCE,
    SEQ_LENS,
    _count_ops,
    _dram_allocated,
    _trace_layout_calls,
)
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    BFP8_POLICY,
    DEFAULT_GEOMETRY,
    DEFAULT_POLICY,
    FUSED_BASELINE_GEOMETRY,
    FUSED_BASELINE_POLICY,
    DecodeGeometry,
    OptimizedDecoder,
)

LAYER_KINDS = [
    pytest.param(H.LINEAR_LAYER_IDX, id="linear_attention"),
    pytest.param(H.FULL_LAYER_IDX, id="full_attention"),
]

#: Acceptance bar for the **synthetic-weight** PCC cases under the shipped precision policy.
#:
#: Every *real-checkpoint* test in this file holds ``H.PCC_BAR`` (0.995), which is the stage's
#: acceptance bar and the number the context contract quotes.  The synthetic-weight cases - which
#: exist for shape, length, paging, aliasing, batching and trace coverage rather than for precision -
#: hold this looser bar, and the reason is measured rather than asserted
#: (``doc/optimized_decoder/logs/probe_blockfloat_distribution.log``):
#:
#: * the suite's stand-in weights are an i.i.d. normal draw with one recorded per-tensor mean and
#:   standard deviation (``reference/hf_reference.py::synthetic_state_dict_from_stats``);
#: * the block-float **weight** error is the same on both: modelling BFP4's shared exponent per
#:   16 elements gives a relative error of 0.1123 on the real ``mlp.gate_proj.weight`` and 0.1112 on
#:   the synthetic one - a 1 % difference - so the stand-in is *not* harder to quantise;
#: * what differs is the signal it produces.  For the same input, the real layer's output norm is
#:   **2.62x** (``linear_attention``) and **1.89x** (``full_attention``) the synthetic layer's, so the
#:   identical quantisation noise sits on a proportionally smaller signal, and 1 - PCC - which is a
#:   noise-to-signal ratio - is correspondingly larger on the stand-in.
#:
#: OPT-012 is explicit that such a case may not veto a policy that passes on real weights.  The
#: real-weight evidence for the shipped policy is the real-weight table in ``work_log.md`` section 2.1,
#: generated from ``logs/probe_real_weight_policy.log``, and
#: ``test_synthetic_bar_is_justified_by_the_real_weight_evidence`` asserts its worst value against
#: ``H.PCC_BAR`` - the numbers are deliberately not restated here, because a second copy of them in a
#: docstring is a copy that goes stale on the next re-measurement, and this one had.
#:
#: This bar is therefore a *stress* bar, chosen from the measured synthetic minimum with margin - the
#: minimum over the whole suite is in the generated correctness table in section 4 - and its purpose is
#: to catch a regression in the paths those cases cover, not to certify precision.  The margin is deliberate rather than tight: these are deterministic values on
#: this build, but a bar sitting a few ten-thousandths under the measured minimum would fail on any
#: unrelated numerical change and teach a later stage to loosen it, which is exactly what this
#: constant must not become.  ``test_synthetic_bar_is_justified_by_the_real_weight_evidence`` pins the
#: relationship so this constant cannot quietly drift into a waiver.
SYNTHETIC_PCC_BAR = 0.975

#: Block-float weight dtypes.  ``ttnn.bfloat4_b`` and ``ttnn.bfloat8_b`` are what the shipped
#: policy asks for; the test asserts against the *policy*, not against a hard-coded dtype, so a
#: deliberate policy change moves the expectation with it.
_BLOCK_FLOAT = (ttnn.bfloat4_b, ttnn.bfloat8_b)


def _stats():
    return ref.load_weight_stats()


def _kind(lut) -> str:
    return lut.config.layer_types[lut.layer_idx]


def _build(mesh_device, layer_idx, *, policy=None, geometry=None, prefill_geometry=None, **kwargs):
    """Build the layer under test.  Always the optimized implementation."""
    lut = H.build_layer(
        mesh_device,
        layer_idx,
        decoder_cls=OptimizedDecoder,
        policy=policy or DEFAULT_POLICY,
        decode_geometry=geometry or DEFAULT_GEOMETRY,
        prefill_geometry=prefill_geometry,
        **kwargs,
    )
    assert isinstance(lut.tt_layer, OptimizedDecoder), "this suite must exercise the optimized path"
    return lut


def _prefill_and_check(lut, seq_len, seed=0, user_id=0, cache=None):
    hidden = ref.synthetic_hidden_states(lut.config, 1, seq_len, _stats(), seed=seed)
    cache = cache if cache is not None else DynamicCache(config=lut.config)
    golden = H.reference_prefill(lut, hidden, cache)
    got = H.run_tt_prefill(lut, hidden, user_id=user_id)
    return H.pcc(golden, got), cache, hidden


# --------------------------------------------------------------------------- correctness


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_prefill_pcc(mesh_device, layer_idx, seq_len):
    """Optimized paged prefill matches the HF layer for aligned and non-aligned lengths alike."""
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=max(8192, seq_len))
    value, _cache, _hidden = _prefill_and_check(lut, seq_len)
    H.record("optimized_prefill_pcc", value, kind=_kind(lut), seq_len=seq_len)
    assert value >= SYNTHETIC_PCC_BAR, f"prefill PCC {value} < {SYNTHETIC_PCC_BAR} (seq_len={seq_len})"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", [17, 2048, 2049, 5000])
def test_decode_pcc(mesh_device, layer_idx, seq_len):
    """Optimized paged decode continues an optimized prefill correctly for several steps."""
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    value, cache, _hidden = _prefill_and_check(lut, seq_len)
    assert value >= SYNTHETIC_PCC_BAR, f"prefill PCC {value} < {SYNTHETIC_PCC_BAR}"
    H.prepare_decode(lut)

    for step in range(4):
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=100 + step)
        golden = H.reference_decode(lut, token, seq_len + step, cache)
        got = H.run_tt_decode(lut, token, torch.tensor([seq_len + step]))
        value = H.pcc(golden, got)
        H.record("optimized_decode_pcc", value, kind=_kind(lut), prefill_len=seq_len, step=step)
        assert value >= SYNTHETIC_PCC_BAR, f"decode PCC {value} < {SYNTHETIC_PCC_BAR} at step {step}"


@pytest.mark.timeout(0)
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("batch", [4, 16, 32])
def test_batched_users(mesh_device, layer_idx, batch):
    """Per-user page tables, cache slots and current positions with unequal prompt lengths."""
    seq_lens = [64 + 97 * u for u in range(batch)]
    lut = _build(mesh_device, layer_idx, max_batch=batch, max_seq_len=8192)

    caches = []
    for user_id, seq_len in enumerate(seq_lens):
        hidden = ref.synthetic_hidden_states(lut.config, 1, seq_len, _stats(), seed=user_id)
        cache = DynamicCache(config=lut.config)
        golden = H.reference_prefill(lut, hidden, cache)
        got = H.run_tt_prefill(lut, hidden, user_id=user_id)
        value = H.pcc(golden, got)
        H.record("optimized_batched_prefill_pcc", value, kind=_kind(lut), batch=batch, user_id=user_id, seq_len=seq_len)
        assert value >= SYNTHETIC_PCC_BAR, f"prefill PCC {value} for user {user_id} (seq_len={seq_len})"
        caches.append(cache)

    H.prepare_decode(lut)
    tokens = torch.cat(
        [ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=200 + u) for u in range(batch)], dim=0
    )
    positions = torch.tensor(seq_lens)
    got = H.run_tt_decode(lut, tokens, positions)
    for user_id in range(batch):
        golden = H.reference_decode(lut, tokens[user_id : user_id + 1], seq_lens[user_id], caches[user_id])
        value = H.pcc(golden, got[user_id : user_id + 1])
        H.record("optimized_batched_decode_pcc", value, kind=_kind(lut), batch=batch, user_id=user_id)
        assert value >= SYNTHETIC_PCC_BAR, f"decode PCC {value} for user {user_id}"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_real_weights(mesh_device, layer_idx):
    """The real checkpoint weights reach the bar through the optimized block-float policy.

    This is the evidence that decides the precision policy, not the synthetic-weight tests: a
    random-weight tensor has a flat spectrum and is close to the worst case for a shared-exponent
    format, while the real projection weights are what the model is (OPT-012).
    """
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, real_weights=True)
    value, cache, _hidden = _prefill_and_check(lut, 2049)
    H.record("optimized_real_weight_prefill_pcc", value, kind=_kind(lut), seq_len=2049)
    assert value >= H.PCC_BAR, f"real-weight prefill PCC {value} < {H.PCC_BAR}"
    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=7)
    golden = H.reference_decode(lut, token, 2049, cache)
    got = H.run_tt_decode(lut, token, torch.tensor([2049]))
    value = H.pcc(golden, got)
    H.record("optimized_real_weight_decode_pcc", value, kind=_kind(lut), position=2049)
    assert value >= H.PCC_BAR, f"real-weight decode PCC {value} < {H.PCC_BAR}"


@pytest.mark.timeout(0)
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_real_weights_traced_decode_multi_step(mesh_device, layer_idx):
    """Real weights, traced replay, several steps - the cache-consuming real-weight check.

    OPT-007 asks for exactly this before a reduced attention-weight policy may be accepted: raw
    K/V-cache PCC against a higher-precision baseline is a diagnostic, because changing the K/V
    *projection* precision legitimately changes the cached intermediate.  What must hold is that
    the outputs the cache feeds are still right, over more than one step, through the traced path.
    """
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, real_weights=True)
    seq_len = 2049
    value, cache, _hidden = _prefill_and_check(lut, seq_len)
    assert value >= H.PCC_BAR
    H.prepare_decode(lut)
    runner = H.TracedDecode(lut, batch=1)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=800)
    golden = H.reference_decode(lut, token, seq_len, cache)
    got = runner.warmup(token, torch.tensor([seq_len]))
    assert H.pcc(golden, got) >= H.PCC_BAR
    runner.capture()
    for step in range(1, 6):
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=800 + step)
        golden = H.reference_decode(lut, token, seq_len + step, cache)
        got = runner.replay(token, torch.tensor([seq_len + step]))
        value = H.pcc(golden, got)
        H.record("optimized_real_weight_traced_decode_pcc", value, kind=_kind(lut), replay=step)
        assert value >= H.PCC_BAR, f"real-weight traced decode PCC {value} at replay {step}"
    runner.release()


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_determinism(mesh_device, layer_idx):
    """Identical inputs produce bit-identical outputs, for prefill and for decode."""
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    hidden = ref.synthetic_hidden_states(lut.config, 1, 2049, _stats())
    first = H.run_tt_prefill(lut, hidden)
    second = H.run_tt_prefill(lut, hidden)
    H.record(
        "optimized_prefill_bit_identical", bool(torch.equal(first, second)), kind=_kind(lut), seq_len=2049, repeats=2
    )
    assert torch.equal(first, second), "prefill is not deterministic for identical inputs"

    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=11)
    H.prepare_decode(lut)
    decode_first = H.run_tt_decode(lut, token, torch.tensor([2049]))
    H.prepare_decode(lut)
    decode_second = H.run_tt_decode(lut, token, torch.tensor([2049]))
    H.record(
        "optimized_decode_bit_identical", bool(torch.equal(decode_first, decode_second)), kind=_kind(lut), repeats=2
    )
    assert torch.equal(decode_first, decode_second), "decode is not deterministic for identical inputs"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_traced_decode_pcc(mesh_device, layer_idx):
    """Optimized decode runs under ttnn traced execution and the replay output matches HF."""
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    seq_len = 2049
    value, cache, _hidden = _prefill_and_check(lut, seq_len)
    assert value >= SYNTHETIC_PCC_BAR
    H.prepare_decode(lut)

    runner = H.TracedDecode(lut, batch=1)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=300)
    golden = H.reference_decode(lut, token, seq_len, cache)
    got = runner.warmup(token, torch.tensor([seq_len]))
    assert H.pcc(golden, got) >= SYNTHETIC_PCC_BAR, "warm-up decode PCC below bar"

    runner.capture()
    for step in range(1, 4):
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=300 + step)
        golden = H.reference_decode(lut, token, seq_len + step, cache)
        got = runner.replay(token, torch.tensor([seq_len + step]))
        value = H.pcc(golden, got)
        H.record("optimized_traced_decode_replay_pcc", value, kind=_kind(lut), replay=step, position=seq_len + step)
        assert value >= SYNTHETIC_PCC_BAR, f"traced decode PCC {value} < {SYNTHETIC_PCC_BAR} at replay {step}"
    runner.release()


@pytest.mark.timeout(0)
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("batch", [4, 32])
def test_traced_decode_batched(mesh_device, layer_idx, batch):
    """Trace capture and replay at batch > 1, with per-user positions and page tables."""
    seq_lens = [64 + 97 * u for u in range(batch)]
    lut = _build(mesh_device, layer_idx, max_batch=batch, max_seq_len=8192)

    caches = []
    for user_id, seq_len in enumerate(seq_lens):
        hidden = ref.synthetic_hidden_states(lut.config, 1, seq_len, _stats(), seed=user_id)
        cache = DynamicCache(config=lut.config)
        H.reference_prefill(lut, hidden, cache)
        H.run_tt_prefill(lut, hidden, user_id=user_id)
        caches.append(cache)
    H.prepare_decode(lut)

    runner = H.TracedDecode(lut, batch=batch)
    positions = torch.tensor(seq_lens)
    tokens = torch.cat(
        [ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=500 + u) for u in range(batch)], dim=0
    )
    runner.warmup(tokens, positions)
    for user_id in range(batch):
        H.reference_decode(lut, tokens[user_id : user_id + 1], seq_lens[user_id], caches[user_id])
    runner.capture()

    for step in range(1, 3):
        tokens = torch.cat(
            [ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=600 + 10 * step + u) for u in range(batch)],
            dim=0,
        )
        positions = torch.tensor([sl + step for sl in seq_lens])
        got = runner.replay(tokens, positions)
        for user_id in range(batch):
            golden = H.reference_decode(lut, tokens[user_id : user_id + 1], seq_lens[user_id] + step, caches[user_id])
            value = H.pcc(golden, got[user_id : user_id + 1])
            H.record(
                "optimized_batched_traced_decode_pcc",
                value,
                kind=_kind(lut),
                batch=batch,
                user_id=user_id,
                replay=step,
            )
            assert value >= SYNTHETIC_PCC_BAR, f"batched traced decode PCC {value} for user {user_id}"
    runner.release()


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_linear_state_and_kv_cache_match_reference(mesh_device, layer_idx):
    """The on-device cache/state after an optimized prefill matches HF's, not just the output.

    For ``full_attention`` this is also the ``bfloat8_b`` KV cache's own check: the cache the
    shipped policy writes is block-float, and this compares it against HF's float32 one.
    """
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    seq_len = 2049
    value, cache, _hidden = _prefill_and_check(lut, seq_len)
    assert value >= SYNTHETIC_PCC_BAR

    if lut.is_full_attention:
        assert lut.tt_layer.kv_cache[0].dtype == DEFAULT_POLICY.kv_cache
        keys, values = H.read_paged_kv(lut, user_id=0, seq_len=seq_len)
        ref_keys = cache.layers[lut.layer_idx].keys[0, :, :seq_len, :].to(torch.float32)
        ref_values = cache.layers[lut.layer_idx].values[0, :, :seq_len, :].to(torch.float32)
        k_pcc, v_pcc = H.pcc(ref_keys, keys), H.pcc(ref_values, values)
        H.record("optimized_paged_k_cache_pcc", k_pcc, kind=_kind(lut), seq_len=seq_len)
        H.record("optimized_paged_v_cache_pcc", v_pcc, kind=_kind(lut), seq_len=seq_len)
        assert k_pcc >= SYNTHETIC_PCC_BAR, "paged K cache mismatch"
        assert v_pcc >= SYNTHETIC_PCC_BAR, "paged V cache mismatch"
    else:
        conv_state, recurrent_state = H.read_linear_state(lut, user_id=0)
        ref_conv = cache.layers[lut.layer_idx].conv_states[0].to(torch.float32)
        ref_recurrent = cache.layers[lut.layer_idx].recurrent_states[0].to(torch.float32)
        conv_pcc, rec_pcc = H.pcc(ref_conv, conv_state.T), H.pcc(ref_recurrent, recurrent_state)
        H.record("optimized_conv_state_pcc", conv_pcc, kind=_kind(lut), seq_len=seq_len)
        H.record("optimized_recurrent_state_pcc", rec_pcc, kind=_kind(lut), seq_len=seq_len)
        assert conv_pcc >= SYNTHETIC_PCC_BAR, "conv state mismatch"
        assert rec_pcc >= SYNTHETIC_PCC_BAR, "recurrent state mismatch"


@pytest.mark.parametrize("steps", [1, 5])
@pytest.mark.parametrize("max_batch", [1, 32])
def test_conv_state_after_decode_matches_reference(mesh_device, steps, max_batch):
    """The carried conv state is still HF's after N *decode* steps through the optimized path."""
    lut = _build(mesh_device, H.LINEAR_LAYER_IDX, max_batch=max_batch, max_seq_len=8192)
    seq_len = 2049
    user_id = max_batch - 1
    hidden = ref.synthetic_hidden_states(lut.config, 1, seq_len, _stats())
    for slot in range(max_batch):
        H.run_tt_prefill(lut, hidden if slot == user_id else torch.zeros_like(hidden), user_id=slot)
    cache = DynamicCache(config=lut.config)
    H.reference_prefill(lut, hidden, cache)
    H.prepare_decode(lut)

    for step in range(steps):
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=900 + step)
        batched = token.repeat(max_batch, 1, 1)
        H.run_tt_decode(lut, batched, torch.full((max_batch,), seq_len + step))
        H.reference_decode(lut, token, seq_len + step, cache)

    packed = ttnn.to_torch(lut.tt_layer.current_conv_state()).to(torch.float32)
    got = packed.reshape(packed.shape[-3], packed.shape[-2], packed.shape[-1])[user_id]
    expected = cache.layers[lut.layer_idx].conv_states[0].to(torch.float32)
    value = H.pcc(expected, got.T)
    H.record("optimized_conv_state_after_decode_pcc", value, kind="linear_attention", steps=steps, batch=max_batch)
    assert value >= SYNTHETIC_PCC_BAR, f"conv state after {steps} decode steps at batch {max_batch}: PCC {value}"


@pytest.mark.parametrize("block_size", [32, 128])
def test_alternate_page_block_size(mesh_device, block_size):
    """Page/block geometry is a parameter, not an assumption (``full_attention`` only)."""
    lut = _build(mesh_device, H.FULL_LAYER_IDX, max_batch=2, max_seq_len=4096, block_size=block_size)
    value, cache, _hidden = _prefill_and_check(lut, 2049, user_id=1)
    H.record("optimized_alt_block_size_prefill_pcc", value, kind=_kind(lut), block_size=block_size, seq_len=2049)
    assert value >= SYNTHETIC_PCC_BAR, f"prefill PCC {value} with block_size={block_size}"

    H.prepare_decode(lut)
    tokens = torch.cat([ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=500 + u) for u in range(2)], dim=0)
    got = H.run_tt_decode(lut, tokens, torch.tensor([2049, 2049]))
    golden = H.reference_decode(lut, tokens[1:2], 2049, cache)
    value = H.pcc(golden, got[1:2])
    H.record("optimized_alt_block_size_decode_pcc", value, kind=_kind(lut), block_size=block_size, position=2049)
    assert value >= SYNTHETIC_PCC_BAR, f"decode PCC {value} with block_size={block_size}"


def test_bfloat16_kv_cache_still_works(mesh_device):
    """The KV-cache dtype is still a caller parameter, in both directions.

    The shipped policy chose ``bfloat8_b``; a serving stage that wants the higher-precision cache
    back must be able to ask for it, so this pins the *other* branch of the same contract that
    ``test_linear_state_and_kv_cache_match_reference`` pins for the shipped one.
    """
    lut = _build(mesh_device, H.FULL_LAYER_IDX, max_batch=1, max_seq_len=8192, cache_dtype=ttnn.bfloat16)
    assert lut.tt_layer.kv_cache[0].dtype == ttnn.bfloat16
    value, cache, _hidden = _prefill_and_check(lut, 2049)
    H.record("optimized_bf16_cache_prefill_pcc", value, kind=_kind(lut), seq_len=2049)
    assert value >= SYNTHETIC_PCC_BAR, f"bfloat16-cache prefill PCC {value} < {SYNTHETIC_PCC_BAR}"
    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=23)
    golden = H.reference_decode(lut, token, 2049, cache)
    got = H.run_tt_decode(lut, token, torch.tensor([2049]))
    value = H.pcc(golden, got)
    H.record("optimized_bf16_cache_decode_pcc", value, kind=_kind(lut), position=2049)
    assert value >= SYNTHETIC_PCC_BAR, f"bfloat16-cache decode PCC {value} < {SYNTHETIC_PCC_BAR}"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", PAD_ALIAS_SEQ_LENS)
def test_prefill_decode_pad_below_one_tile(mesh_device, layer_idx, seq_len):
    """Regression: logical lengths whose chunk padding is smaller than one tile."""
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=4096)
    value, cache, _hidden = _prefill_and_check(lut, seq_len)
    H.record("optimized_pad_alias_prefill_pcc", value, kind=_kind(lut), seq_len=seq_len)
    assert value >= SYNTHETIC_PCC_BAR, f"prefill PCC {value} < {SYNTHETIC_PCC_BAR} (seq_len={seq_len})"

    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=17)
    golden = H.reference_decode(lut, token, seq_len, cache)
    got = H.run_tt_decode(lut, token, torch.tensor([seq_len]))
    value = H.pcc(golden, got)
    H.record("optimized_pad_alias_decode_pcc", value, kind=_kind(lut), prefill_len=seq_len)
    assert value >= SYNTHETIC_PCC_BAR, f"decode PCC {value} < {SYNTHETIC_PCC_BAR} (prefill_len={seq_len})"


@pytest.mark.timeout(0)
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_prefill_pcc_long(mesh_device, layer_idx):
    """HF-vs-TTNN prefill PCC at the longest length the CPU reference can still reach."""
    config = ref.load_text_config()
    seq_len = LONG_PCC_SEQ_LEN[config.layer_types[layer_idx]]
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=seq_len + 2048)
    value, _cache, _hidden = _prefill_and_check(lut, seq_len)
    H.record("optimized_long_prefill_pcc", value, kind=_kind(lut), seq_len=seq_len)
    assert value >= SYNTHETIC_PCC_BAR, f"long prefill PCC {value} < {SYNTHETIC_PCC_BAR} (seq_len={seq_len})"


@pytest.mark.timeout(0)
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("max_batch", [1, 32])
def test_repeated_runs_stable(mesh_device, layer_idx, max_batch):
    """Stress: repeated prefill+decode cycles stay bit-identical and free every allocation.

    The optimized layer holds more persistent device state than the fused one (the split gate/up
    weight) and reuses far more L1 (the whole decode stream), so a leak or a stale shard would show
    here rather than in a single-pass PCC test.
    """
    cycles = 6
    lut = _build(mesh_device, layer_idx, max_batch=max_batch, max_seq_len=8192)
    hidden = ref.synthetic_hidden_states(lut.config, 1, 2049, _stats())
    tokens = [ref.synthetic_hidden_states(lut.config, max_batch, 1, _stats(), seed=700 + s) for s in range(4)]

    baseline_free = None
    reference_out = None
    for cycle in range(cycles):
        prefill = H.run_tt_prefill(lut, hidden)
        H.prepare_decode(lut)
        decodes = [H.run_tt_decode(lut, tokens[s], torch.full((max_batch,), 2049 + s)) for s in range(4)]
        assert torch.isfinite(prefill).all() and all(torch.isfinite(d).all() for d in decodes)
        result = (prefill, torch.cat(decodes, dim=0))
        if reference_out is None:
            reference_out = result
        else:
            assert torch.equal(reference_out[0], result[0]), f"prefill drifted on cycle {cycle}"
            assert torch.equal(reference_out[1], result[1]), f"decode drifted on cycle {cycle}"
        allocated = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank
        if cycle == 1:
            baseline_free = allocated
        elif baseline_free is not None:
            assert allocated == baseline_free, (
                f"DRAM bytes allocated per bank moved from {baseline_free} to {allocated} on cycle {cycle}: "
                "the optimized graph is leaking a device buffer per cycle"
            )
    H.record(
        "optimized_repeated_run_bit_identical", True, kind=_kind(lut), cycles=cycles, decode_steps=4, batch=max_batch
    )


@pytest.mark.timeout(0)
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_traced_decode_stress(mesh_device, layer_idx):
    """Stress the *traced* path: 200 replays, checked for drift and for finiteness.

    The shipped decode keeps every activation in L1 on a fixed shard grid and writes the recurrent
    state in place, so a trace that aliases a buffer it should not would show as slow drift rather
    than as a first-replay failure.  200 replays at a fixed input must stay bit-identical.
    """
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    hidden = ref.synthetic_hidden_states(lut.config, 1, 2048, _stats())
    H.run_tt_prefill(lut, hidden)
    H.prepare_decode(lut)
    runner = H.TracedDecode(lut, batch=1)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=321)
    positions = torch.tensor([2048])
    runner.warmup(token, positions)
    runner.capture()
    # ``full_attention`` at a fixed position rewrites the same cache slot every replay, so its
    # replays must be bit-identical.  ``linear_attention`` advances its recurrent state on every
    # replay by construction, so identical outputs would be the *bug* - the state buffer would not be
    # being written.  What must hold there is that the sequence stays finite and bounded: a trace that
    # aliased a buffer it should not shows up as the state running away or collapsing over 200 steps,
    # not as a first-replay failure.  (Cross-pass reproducibility is *not* asserted: ``warmup()`` and
    # ``capture()`` each advance the state once, so a second pass after ``prepare_decode_state()``
    # starts two advances earlier and is not expected to line up.)
    replays = [runner.replay(token, positions) for _ in range(200)]
    assert all(torch.isfinite(x).all() for x in replays), "traced replay produced non-finite values"
    if lut.is_full_attention:
        for index, value in enumerate(replays[1:], start=1):
            assert torch.equal(replays[0], value), f"full_attention traced replay drifted at replay {index}"
    else:
        norms = [float(x.norm()) for x in replays]
        assert all(n > 0 for n in norms), "linear_attention traced replay collapsed to zero"
        assert max(norms) < 8 * norms[0], (
            f"linear_attention traced replay output norm ran away: {norms[0]:.3f} -> {max(norms):.3f} "
            "over 200 replays, which is what an aliased state buffer looks like"
        )
        assert not torch.equal(
            replays[0], replays[1]
        ), "linear_attention traced replays are identical, so the recurrent state is not being advanced"
        H.record(
            "optimized_traced_decode_stress_norms",
            {"first": norms[0], "max": max(norms), "min": min(norms)},
            kind=_kind(lut),
        )
    H.record("optimized_traced_decode_stress_replays", {"replays": len(replays)}, kind=_kind(lut))
    runner.release()


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_no_runtime_host_fallback(mesh_device, layer_idx):
    """A single measured optimized prefill/decode pass makes no torch or host-transfer calls."""
    module = sys.modules[OptimizedDecoder.__module__]
    source = pathlib.Path(module.__file__).read_text()
    docstring_end = source.index('"""', source.index('"""') + 3) + 3
    class_anchor = f"class {OptimizedDecoder.__name__}"
    helper_region = source[docstring_end:].split(class_anchor, 1)[0]
    class_body = source.split(class_anchor, 1)[1]
    constructor_region = class_body.split("    # -- construction", 1)[0]
    after = class_body.split("    # ------------------------------------------------------------- primitives", 1)
    assert len(after) == 2, "the primitives anchor moved; the scan region is wrong"
    for region, label in (
        (helper_region, "module helpers"),
        (constructor_region, "__init__ and configuration"),
        (after[1], "layer runtime"),
    ):
        assert region.strip(), f"{label} region came out empty - the scan anchors moved"
        code = "\n".join(line for line in region.splitlines() if not line.lstrip().startswith("#"))
        for pattern in (r"\btorch\b", r"from_torch", r"to_torch", r"\bas_tensor\b", r"\.cpu\(\)"):
            assert not re.search(pattern, code), f"{label} references {pattern!r}"

    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    hidden = ref.synthetic_hidden_states(lut.config, 1, 2049, _stats())
    tt_in = H.tt_hidden_prefill(hidden, mesh_device)
    rot = H.prefill_rot_mats(lut, 2049, mesh_device) if lut.is_full_attention else None
    full_pt, per_chunk = H.chunk_page_tables(lut, 2049, 0, mesh_device)

    with H.forbid_host_fallback():
        out = lut.tt_layer.prefill_forward(
            tt_in, user_id=0, page_table=full_pt, page_tables_per_chunk=per_chunk, rot_mats=rot
        )
    ttnn.deallocate(out)

    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=13)
    tt_token = H.tt_hidden_decode(token, mesh_device)
    pos_tt = None
    decode_rot = None
    if lut.is_full_attention:
        cos, sin = H.decode_rot_mats_torch(lut, torch.tensor([2049]))
        decode_rot = (H.to_device(cos, mesh_device), H.to_device(sin, mesh_device))
        pos_tt = H.to_device(torch.tensor([2049], dtype=torch.int32), mesh_device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    with H.forbid_host_fallback():
        out = lut.tt_layer.decode_forward(
            tt_token, current_pos=pos_tt, page_table=lut.page_table_tt, rot_mats=decode_rot
        )
    ttnn.deallocate(out)


#: Layout conversions no measured pass may ask for, and the memory-config conversions whose count
#: is pinned below.
_LAYOUT_OPS = ("tilize", "tilize_with_val_padding", "untilize", "untilize_with_unpadding", "to_layout")
_RESHARD_OPS = ("interleaved_to_sharded", "sharded_to_interleaved", "to_memory_config")

#: Exact number of memory-config conversions the optimized decode may make, per layer kind, and
#: the contract that forces each one.  This is an equality, not a budget, for the same reason the
#: fused suite made its version one: a budget leaves room for new unnecessary reshards under a test
#: whose docstring says there are none.
#:
#: ``linear_attention`` at batch 1 (6):
#:   1 the caller's DRAM-interleaved hidden state onto the stream grid;
#:   1 the normed stream tensor off the stream grid for ``in_proj_ab``, whose bias fold needs the
#:     interleaved, ``core_grid``-placed matmul (see ``_gdn_inputs``);
#:   1 ``in_proj_qkv``'s float32 output off the stream grid, because the causal conv is a ROW_MAJOR
#:     slice/concat chain;
#:   1 the gated recurrence result onto the stream grid for ``out_proj`` - the recurrence itself is
#:     float32 interleaved state arithmetic and stays that way;
#:   1 ``in_proj_z``'s output off the DRAM-sharded matmul's own output grid, because the small-batch
#:     branch changes its rank and ``ttnn.reshape`` cannot do that on a width-sharded tensor without
#:     silently falling back to INTERLEAVED - so it is done explicitly and counted here instead of
#:     happening invisibly (see ``_linear_attention_decode``).  Absent at batch 32, whose
#:     group-reduction branch consumes ``z`` flat and sharded;
#:   1 the final ``sharded_to_interleaved`` back to the public output contract.
#:
#: ``linear_attention`` at the advertised ``max_batch`` (4): the same list **minus** the ``in_proj_z``
#: conversion above and **minus** the ``out_proj`` one.  At batch 32 the z-gated norm takes the group-reduction branch, whose ``ttnn.multiply``
#: with the width-sharded ``in_proj_z`` output propagates that shard to its result, so ``out_proj``'s
#: input is already on the stream grid.  The small-batch branch reshapes ``z`` to
#: ``[1, batch, heads, head_dim]`` first and loses it.  One fewer conversion at the wider batch is
#: the right direction, so it is recorded rather than smoothed over.
#:
#: ``full_attention`` (8, at both batches):
#:   1 the caller's hidden state onto the stream grid;
#:   1 ``wqkv``'s output off the stream grid for ``nlp_create_qkv_heads_decode``;
#:   2 Q and K off the head op's height-sharded layout for the interleaved norm and RoPE;
#:   1 K back onto it for ``paged_update_cache``;
#:   1 the SDPA output onto it for ``nlp_concat_heads_decode``;
#:   1 the concat result onto the stream grid for the epilogue;
#:   1 the final ``sharded_to_interleaved`` back to the public output contract.
EXPECTED_DECODE_RESHARDS = {
    ("linear_attention", 1): 6,
    ("linear_attention", 32): 4,
    ("full_attention", 1): 8,
    ("full_attention", 32): 8,
}


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("max_batch", [1, 32])
def test_no_relayout_or_host_ops_in_measured_decode(mesh_device, layer_idx, max_batch):
    """The measured decode asks for no layout conversion, and only contract-forced reshards.

    The optimized decode's reshard count must also be **no worse** than the fused stage's, which is
    the point of moving the residual stream into L1: the fused layer paid 4 (``linear_attention``)
    and 9 (``full_attention``), of which four were the two norms' shard/unshard brackets.
    """
    lut = _build(mesh_device, layer_idx, max_batch=max_batch, max_seq_len=8192)
    hidden = ref.synthetic_hidden_states(lut.config, 1, 2048, _stats())
    H.run_tt_prefill(lut, hidden)
    H.prepare_decode(lut)

    counter = _count_ops(tuple(f"ttnn.{name}" for name in _LAYOUT_OPS + _RESHARD_OPS))
    token = ref.synthetic_hidden_states(lut.config, max_batch, 1, _stats(), seed=61)
    with counter:
        H.run_tt_decode(lut, token, torch.full((max_batch,), 2048))

    layout_calls = {k: v for k, v in counter.counts.items() if k.rsplit(".", 1)[1] in _LAYOUT_OPS and v}
    assert not layout_calls, f"measured decode does a layout conversion: {layout_calls}"

    reshards = sum(v for k, v in counter.counts.items() if k.rsplit(".", 1)[1] in _RESHARD_OPS)
    expected = EXPECTED_DECODE_RESHARDS[(_kind(lut), max_batch)]
    H.record(
        "optimized_decode_reshard_ops",
        {"reshards": reshards, "expected": expected},
        kind=_kind(lut),
        batch=max_batch,
    )
    assert reshards == expected, (
        f"measured decode reshards {reshards} times, expected exactly {expected}; "
        f"if this is deliberate, account for the new one in EXPECTED_DECODE_RESHARDS: {counter.counts}"
    )


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_no_redundant_relayout_in_measured_prefill(mesh_device, layer_idx):
    """The measured prefill never tilizes a tensor it is about to untilize, or vice versa."""
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    hidden = ref.synthetic_hidden_states(lut.config, 1, 2048, _stats())
    tt_in = H.tt_hidden_prefill(hidden, mesh_device)
    rot = H.prefill_rot_mats(lut, 2048, mesh_device) if lut.is_full_attention else None
    full_pt, per_chunk = H.chunk_page_tables(lut, 2048, 0, mesh_device)

    trace = _trace_layout_calls()
    with trace:
        out = lut.tt_layer.prefill_forward(
            tt_in, user_id=0, page_table=full_pt, page_tables_per_chunk=per_chunk, rot_mats=rot
        )
    ttnn.deallocate(out)

    undone = trace.round_trips
    H.record("optimized_prefill_layout_round_trips", {"round_trips": len(undone)}, kind=_kind(lut))
    assert not undone, f"measured prefill undoes a layout conversion it just made: {undone}"


# ------------------------------------------------------------------ optimization-specific


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_optimized_config_is_the_shipped_one(mesh_device, layer_idx):
    """Every weight on the device carries the policy's dtype and memory config.

    OPT-013: a policy name is intent.  A lazily-cached, rewrapped or fallback-constructed weight
    can silently leave a claimed BFP4 policy running in bfloat16, and no PCC test would notice -
    it would only be slower.  This reads the dtype and the memory layout off the *device tensor*.
    """
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    layer = lut.tt_layer
    summary = layer.config_summary()
    assert summary["policy"] == DEFAULT_POLICY.name
    assert summary["decode"]["sharded_stream"] and summary["decode"]["dram_sharded"]
    assert summary["decode"]["cores"] in summary["decode"]["legal_cores"]

    for role, entry in summary["roles"].items():
        tensor = layer.w[role]
        assert (
            str(tensor.dtype) == entry["policy_dtype"]
        ), f"{role} is on the device as {tensor.dtype} but the policy asks for {entry['policy_dtype']}"
        if role in layer.weight_mem_cfg:
            assert tensor.memory_config().is_sharded(), f"{role} should be DRAM width-sharded for the decode matmul"
            assert entry["weight_memory"] == "dram_width_sharded"
    block_float_roles = [r for r, e in summary["roles"].items() if e["policy_dtype"] != str(ttnn.float32)]
    assert block_float_roles, "no projection role carries a reduced-precision weight"
    for role in block_float_roles:
        assert layer.w[role].dtype in _BLOCK_FLOAT + (ttnn.bfloat16,)
    H.record("optimized_config_summary", summary, kind=_kind(lut))


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_decode_matmuls_are_dram_sharded(mesh_device, layer_idx):
    """Every dominant decode projection really runs the DRAM-sharded program config.

    Checked by intercepting ``ttnn.linear`` on a measured decode pass and recording, for every
    call, the program-config class and whether both operands were where the DRAM-sharded contract
    requires: activation width-sharded in L1, weight width-sharded in DRAM.
    """
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    hidden = ref.synthetic_hidden_states(lut.config, 1, 2048, _stats())
    H.run_tt_prefill(lut, hidden)
    H.prepare_decode(lut)

    seen = []
    original = ttnn.linear

    def spy(a, b, **kwargs):
        program_config = kwargs.get("program_config")
        seen.append(
            {
                "program_config": type(program_config).__name__ if program_config is not None else None,
                "in0_sharded": bool(a.memory_config().is_sharded()),
                "in1_sharded": bool(b.memory_config().is_sharded()),
                "in1_dtype": str(b.dtype),
                "n": int(b.shape[-1]),
            }
        )
        return original(a, b, **kwargs)

    ttnn.linear = spy
    try:
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=71)
        H.run_tt_decode(lut, token, torch.tensor([2048]))
    finally:
        ttnn.linear = original

    expected_roles = {r for r in lut.tt_layer.decode_program_cfg}
    dram_sharded = [
        call for call in seen if call["program_config"] == "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig"
    ]
    H.record(
        "optimized_decode_linear_calls",
        {"total": len(seen), "dram_sharded": len(dram_sharded), "roles_configured": sorted(expected_roles)},
        kind=_kind(lut),
    )
    assert len(dram_sharded) == len(expected_roles), (
        f"expected one DRAM-sharded matmul per configured role {sorted(expected_roles)}, "
        f"saw {len(dram_sharded)} of {len(seen)} linear calls: {seen}"
    )
    for call in dram_sharded:
        assert call["in0_sharded"], f"DRAM-sharded matmul got an interleaved activation: {call}"
        assert call["in1_sharded"], f"DRAM-sharded matmul got an interleaved weight: {call}"
    # The one interleaved decode matmul that remains is ``in_proj_ab``'s bias-folded projection.
    interleaved = [call for call in seen if call["program_config"] is None]
    assert len(interleaved) == (
        0 if lut.is_full_attention else 1
    ), f"unexpected interleaved decode matmuls: {interleaved}"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_decode_stream_stays_in_l1(mesh_device, layer_idx):
    """No activation on the residual stream is DRAM-interleaved between the norms (OPT-003).

    Intercepts ``ttnn.rms_norm``, ``ttnn.add`` and the projections, and asserts that every tensor
    whose width is ``hidden_size`` and whose row count is the decode row count is width-sharded in
    L1.  The tensors that legitimately leave L1 - the head/cache/SDPA chain and the causal conv -
    have different widths and are excluded by that test rather than by a name list.
    """
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    hidden_size = lut.config.hidden_size
    rows = lut.tt_layer.decode_rows
    hidden = ref.synthetic_hidden_states(lut.config, 1, 2048, _stats())
    H.run_tt_prefill(lut, hidden)
    H.prepare_decode(lut)

    offenders = []
    watched = {"rms_norm": ttnn.rms_norm, "add": ttnn.add}

    def make(name, func):
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            for label, tensor in (("in", args[0] if args else None), ("out", out)):
                if tensor is None or not hasattr(tensor, "memory_config"):
                    continue
                shape = [int(d) for d in tensor.shape]
                if len(shape) == 4 and shape[-1] == hidden_size and shape[-2] == rows:
                    if not tensor.memory_config().is_sharded():
                        offenders.append(f"{name} {label} {shape}")
            return out

        return wrapper

    for name, func in watched.items():
        setattr(ttnn, name, make(name, func))
    try:
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=73)
        H.run_tt_decode(lut, token, torch.tensor([2048]))
    finally:
        for name, func in watched.items():
            setattr(ttnn, name, func)

    # The layer's public output is DRAM-interleaved by contract, and it is the *output* of the last
    # ``ttnn.add`` before the final ``sharded_to_interleaved`` - so the add's own output is sharded
    # and nothing here should fire.
    H.record("optimized_decode_stream_dram_offenders", {"offenders": offenders}, kind=_kind(lut))
    assert not offenders, f"decode residual stream leaves L1: {offenders}"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_optimized_matches_fused(mesh_device, layer_idx):
    """With the fused stage's policy and layout restored, the optimized code reproduces it.

    This is what makes the stage's before/after comparison like-for-like: the optimized module is
    not a rewrite that happens to be faster, it is the fused graph with different dtypes, layouts
    and program configs, and putting those back must give the fused answer.
    """
    outputs = {}
    for name, cls, kwargs in (
        ("fused", FusedDecoder, {}),
        (
            "optimized-at-fused-policy",
            OptimizedDecoder,
            {"policy": FUSED_BASELINE_POLICY, "decode_geometry": FUSED_BASELINE_GEOMETRY},
        ),
    ):
        lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, decoder_cls=cls, **kwargs)
        hidden = ref.synthetic_hidden_states(lut.config, 1, 2049, _stats())
        prefill = H.run_tt_prefill(lut, hidden)
        H.prepare_decode(lut)
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=51)
        decode = H.run_tt_decode(lut, token, torch.tensor([2049]))
        outputs[name] = (prefill, decode)
        kind = _kind(lut)
        H.release_layers()

    for index, phase in enumerate(("prefill", "decode")):
        value = H.pcc(outputs["fused"][index], outputs["optimized-at-fused-policy"][index])
        H.record("optimized_vs_fused_same_policy_pcc", value, kind=kind, phase=phase, seq_len=2049)
        assert value >= 0.9999, f"optimized-at-fused-policy vs fused {phase} PCC {value} < 0.9999"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_optimized_output_agrees_with_fused(mesh_device, layer_idx):
    """The *shipped* optimized layer still agrees with the fused layer, at the acceptance bar.

    A drop-in replacement has to agree with what it replaces, not only with HF: this is the check
    that the block-float policy is a precision change and not a different computation.
    """
    outputs = {}
    for name, cls, kwargs in (("fused", FusedDecoder, {}), ("optimized", OptimizedDecoder, {})):
        lut = H.build_layer(
            mesh_device, layer_idx, max_batch=1, max_seq_len=8192, real_weights=True, decoder_cls=cls, **kwargs
        )
        hidden = ref.synthetic_hidden_states(lut.config, 1, 2049, _stats())
        prefill = H.run_tt_prefill(lut, hidden)
        H.prepare_decode(lut)
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=51)
        decode = H.run_tt_decode(lut, token, torch.tensor([2049]))
        outputs[name] = (prefill, decode)
        kind = _kind(lut)
        H.release_layers()

    for index, phase in enumerate(("prefill", "decode")):
        value = H.pcc(outputs["fused"][index], outputs["optimized"][index])
        H.record("optimized_vs_fused_real_weight_pcc", value, kind=kind, phase=phase, seq_len=2049)
        assert value >= H.PCC_BAR, f"optimized vs fused {phase} PCC {value} < {H.PCC_BAR}"


@pytest.mark.timeout(0)
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("max_batch", [1, 32])
def test_optimized_beats_fused_traced_decode(mesh_device, layer_idx, max_batch):
    """Traced warmed decode is faster than the fused stage's, in the same harness, same run.

    A latency gate rather than a report: the goal for this stage is that the final code beats the
    best correct baseline on traced decode, and the only way that claim cannot drift is for a test
    to measure both arms back to back on the same device in the same session.  The margin asserted
    here is deliberately loose (10 %); the real numbers are in ``doc/optimized_decoder/``.
    """
    import time

    def traced_ms(cls, kwargs):
        lut = H.build_layer(mesh_device, layer_idx, max_batch=max_batch, max_seq_len=8192, decoder_cls=cls, **kwargs)
        hidden = ref.synthetic_hidden_states(lut.config, 1, 2048, _stats())
        H.run_tt_prefill(lut, hidden)
        H.prepare_decode(lut)
        runner = H.TracedDecode(lut, batch=max_batch)
        token = ref.synthetic_hidden_states(lut.config, max_batch, 1, _stats(), seed=400)
        positions = torch.full((max_batch,), 2048)
        runner.warmup(token, positions)
        runner.capture()
        runner.replay(token, positions)
        ttnn.synchronize_device(mesh_device)
        samples = []
        for _ in range(7):
            start = time.perf_counter()
            for _ in range(16):
                ttnn.execute_trace(mesh_device, runner.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter() - start) * 1e3 / 16)
        runner.release()
        H.release_layers()
        return sorted(samples)[len(samples) // 2]

    fused_ms = traced_ms(FusedDecoder, {})
    optimized_ms = traced_ms(OptimizedDecoder, {})
    H.record(
        "optimized_vs_fused_traced_decode_ms",
        {"fused": fused_ms, "optimized": optimized_ms, "speedup": fused_ms / optimized_ms},
        kind=ref.load_text_config().layer_types[layer_idx],
        batch=max_batch,
    )
    assert optimized_ms < fused_ms * 0.9, (
        f"optimized traced decode {optimized_ms:.4f} ms is not at least 10 % faster than the fused "
        f"stage's {fused_ms:.4f} ms at batch {max_batch}"
    )


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_capacity_did_not_shrink(mesh_device, layer_idx):
    """Persistent device bytes of the optimized layer against the fused layer's.

    The context contract's capacity claim rests on this number.  Block-float weights and a
    ``bfloat8_b`` KV cache make the layer much smaller; the split gate/up weight makes it a little
    larger.  The assertion is only that the optimized layer is not *bigger*, because a bigger layer
    would be a capacity change and would have to be argued for in the contract.
    """
    implementations = (("fused", FusedDecoder, {}), ("optimized", OptimizedDecoder, {}))
    for _name, cls, kwargs in implementations:
        H.build_layer(mesh_device, layer_idx, max_batch=32, max_seq_len=8192, decoder_cls=cls, **kwargs)
        H.release_layers()

    allocated = {}
    for name, cls, kwargs in implementations:
        before = _dram_allocated(mesh_device)
        H.build_layer(mesh_device, layer_idx, max_batch=32, max_seq_len=8192, decoder_cls=cls, **kwargs)
        allocated[name] = _dram_allocated(mesh_device) - before
        H.release_layers()

    delta = allocated["optimized"] - allocated["fused"]
    H.record(
        "optimized_persistent_dram_bytes",
        {"fused": allocated["fused"], "optimized": allocated["optimized"], "delta": delta},
        kind=ref.load_text_config().layer_types[layer_idx],
        max_batch=32,
    )
    assert delta <= 0, f"the optimized layer holds {delta} more bytes of persistent device state than the fused one"


def test_illegal_decode_core_count_is_rejected(mesh_device, expect_error):
    """A core count that does not shard every projection width cleanly must fail loudly."""
    with expect_error(ValueError, "legal values here are"):
        _build(mesh_device, H.FULL_LAYER_IDX, max_batch=1, max_seq_len=4096, geometry=DecodeGeometry(cores=11))


def test_illegal_in0_block_w_is_rejected(mesh_device, expect_error):
    """An ``in0_block_w`` that does not divide the input shard's K tiles must fail loudly."""
    with expect_error(ValueError, "does not divide the input shard"):
        _build(
            mesh_device,
            H.FULL_LAYER_IDX,
            max_batch=1,
            max_seq_len=4096,
            geometry=DecodeGeometry(cores=16, in0_block_w={"wqkv": 3}),
        )


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_bfp8_policy_also_passes(mesh_device, layer_idx):
    """The conservative BFP8 fallback policy is a supported configuration, not just a candidate.

    Every BFP4 group in the shipped policy has a BFP8 fallback recorded in the work log; this test
    is what makes those fallbacks usable rather than hypothetical, so a later stage that needs more
    headroom can select :data:`BFP8_POLICY` without rediscovering whether it works.
    """
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, policy=BFP8_POLICY, real_weights=True)
    value, cache, _hidden = _prefill_and_check(lut, 2049)
    H.record("optimized_bfp8_policy_prefill_pcc", value, kind=_kind(lut), seq_len=2049)
    assert value >= H.PCC_BAR
    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=29)
    golden = H.reference_decode(lut, token, 2049, cache)
    got = H.run_tt_decode(lut, token, torch.tensor([2049]))
    value = H.pcc(golden, got)
    H.record("optimized_bfp8_policy_decode_pcc", value, kind=_kind(lut), position=2049)
    assert value >= H.PCC_BAR


@pytest.mark.parametrize("max_batch", [1, 32])
def test_norm_before_expand_matches_expand_before_norm(mesh_device, max_batch):
    """Reordering the gated-delta-net q/k norm and head expansion changes nothing numerically.

    The shipped geometry normalises the 16 key heads and *then* expands them to the 48 value heads on
    a batch axis; stage 2 expanded first, on a tile axis, which cost an untilize/concat/tilize
    round-trip on one and two cores.  The reorder is only sound because the norm is per-head over the
    last dim and the expanded copies are identical, so ``norm(repeat(x)) == repeat(norm(x))``.

    That is an algebraic claim about the two graphs, so this compares the two graphs against each
    other at the *acceptance* bar rather than each against the HF reference: if the reorder were not
    exact, the difference would show up here first and with nothing else moving.  ``linear_attention``
    only - the other layer kind has no gated delta net.
    """
    shipped = _build(mesh_device, H.LINEAR_LAYER_IDX, max_batch=max_batch, max_seq_len=8192)
    hidden = ref.synthetic_hidden_states(shipped.config, 1, 2049, _stats())
    prefill_shipped = H.run_tt_prefill(shipped, hidden)
    H.prepare_decode(shipped)
    token = ref.synthetic_hidden_states(shipped.config, max_batch, 1, _stats(), seed=77)
    positions = torch.full((max_batch,), 2049)
    decode_shipped = H.run_tt_decode(shipped, token, positions)
    conv_shipped, recurrent_shipped = H.read_linear_state(shipped, user_id=0)
    H.release_layers()

    other = _build(
        mesh_device,
        H.LINEAR_LAYER_IDX,
        max_batch=max_batch,
        max_seq_len=8192,
        geometry=dataclasses.replace(DEFAULT_GEOMETRY, norm_before_repeat=False),
    )
    prefill_other = H.run_tt_prefill(other, hidden)
    H.prepare_decode(other)
    decode_other = H.run_tt_decode(other, token, positions)
    conv_other, recurrent_other = H.read_linear_state(other, user_id=0)

    for label, left, right in (
        ("prefill", prefill_shipped, prefill_other),
        ("decode", decode_shipped, decode_other),
        ("conv state", conv_shipped, conv_other),
        ("recurrent state", recurrent_shipped, recurrent_other),
    ):
        value = H.pcc(left, right)
        H.record("optimized_norm_expand_order_pcc", value, kind="linear_attention", batch=max_batch, tensor=label)
        assert value >= H.PCC_BAR, (
            f"reordering the q/k norm and head expansion moved the {label} by more than the acceptance "
            f"bar allows (PCC {value}); the reorder is only valid if the two graphs agree"
        )


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_packed_gate_up_decode_is_still_correct(mesh_device, layer_idx):
    """The rejected packed-gate/up decode arm still works, so its measurement is a real candidate.

    OPT-010 requires the packed and split families to be compared; a comparison against an arm that
    does not run is not a comparison.
    """
    # At the shipped 32 cores the packed form does not allocate: its ``per_core_N`` is twice the split
    # form's and the circular buffers clash with the resident width-sharded activations.  16 cores is
    # where both families are legal, which is why the OPT-010 comparison in ``work_log.md`` section 3.3
    # is made there - so this correctness check runs the arm at the geometry it was measured at.
    geometry = DecodeGeometry(cores=16, in0_block_w={}, split_gate_up_decode=False)
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, geometry=geometry)
    value, cache, _hidden = _prefill_and_check(lut, 2049)
    assert value >= SYNTHETIC_PCC_BAR
    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=31)
    golden = H.reference_decode(lut, token, 2049, cache)
    got = H.run_tt_decode(lut, token, torch.tensor([2049]))
    value = H.pcc(golden, got)
    H.record("optimized_packed_gate_up_decode_pcc", value, kind=_kind(lut), position=2049)
    assert value >= SYNTHETIC_PCC_BAR


# ------------------------------------------------------------------ full advertised context


@pytest.mark.timeout(0)
@pytest.mark.long_context
@pytest.mark.parametrize("weights", ["synthetic", "real"])
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_full_advertised_context(mesh_device, layer_idx, weights, request):
    """Prefill 262143 tokens and decode at position 262143 through the optimized graph.

    Same reference construction as the earlier stages' test of the same name.  This is the capacity
    contract: the optimized policy must not have cost the advertised context, in either direction -
    not the length, and not the accuracy at that length.

    Run at both weight sources.  The synthetic arm is the stress arm - stand-in weights on a
    stand-in activation distribution, held to :data:`SYNTHETIC_PCC_BAR` for the reason
    :func:`test_synthetic_bar_is_justified_by_the_real_weight_evidence` pins - and the ``real`` arm
    is the one that speaks for the advertised capability, held to the acceptance bar ``H.PCC_BAR``.
    Every shorter-context test in this suite has both arms; the advertised context needs them too,
    because a precision policy that only holds on a stand-in distribution has not been shown to hold
    at 262143 tokens on the shipped weights.
    """
    if not request.config.getoption("--long-context"):
        pytest.skip("needs --long-context")
    real = weights == "real"
    bar = H.PCC_BAR if real else SYNTHETIC_PCC_BAR
    context = ref.load_text_config().max_position_embeddings
    assert LONG_PROMPT + 1 == context
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=context, real_weights=real)
    tail = LONG_TAIL[_kind(lut)]
    stats = _stats()
    hidden = ref.synthetic_hidden_states(lut.config, 1, LONG_PROMPT, stats)

    got = H.run_tt_prefill(lut, hidden)
    assert got.shape == (1, LONG_PROMPT, lut.config.hidden_size)
    assert torch.isfinite(got).all(), "long prefill produced non-finite values"

    if lut.is_full_attention:
        probe_len = 512
        probe = hidden[:, :probe_len, :].contiguous()
        real_cache = DynamicCache(config=lut.config)
        H.reference_prefill(lut, probe, real_cache)
        built_cache = DynamicCache(config=lut.config)
        H.fill_reference_kv_cache(lut, probe, built_cache)
        real_k, real_v = H.reference_cache_kv(lut, real_cache, probe_len)
        built_k, built_v = H.reference_cache_kv(lut, built_cache, probe_len)
        assert torch.equal(real_k, built_k), "hand-built reference K cache differs from HF's"
        assert torch.equal(real_v, built_v), "hand-built reference V cache differs from HF's"

        cache = DynamicCache(config=lut.config)
        H.fill_reference_kv_cache(lut, hidden[:, : LONG_PROMPT - tail, :].contiguous(), cache)
        golden = H.reference_prefill(lut, hidden[:, LONG_PROMPT - tail :, :].contiguous(), cache)

        keys, values = H.read_paged_kv(lut, user_id=0, seq_len=LONG_PROMPT)
        ref_keys, ref_values = H.reference_cache_kv(lut, cache, LONG_PROMPT)
        k_pcc, v_pcc = H.pcc(ref_keys, keys), H.pcc(ref_values, values)
        H.record(
            "optimized_full_context_paged_k_cache_pcc", k_pcc, weights=weights, kind=_kind(lut), seq_len=LONG_PROMPT
        )
        H.record(
            "optimized_full_context_paged_v_cache_pcc", v_pcc, weights=weights, kind=_kind(lut), seq_len=LONG_PROMPT
        )
        assert k_pcc >= bar, f"paged K cache at {LONG_PROMPT} tokens: PCC {k_pcc}"
        assert v_pcc >= bar, f"paged V cache at {LONG_PROMPT} tokens: PCC {v_pcc}"
    else:
        golden, cache = H.reference_prefill_segmented(lut, hidden, LONG_SEGMENT)
        golden = golden[:, -tail:, :]

        conv_state, recurrent_state = H.read_linear_state(lut, user_id=0)
        ref_conv = cache.layers[lut.layer_idx].conv_states[0].to(torch.float32)
        ref_recurrent = cache.layers[lut.layer_idx].recurrent_states[0].to(torch.float32)
        conv_pcc = H.pcc(ref_conv, conv_state.T)
        rec_pcc = H.pcc(ref_recurrent, recurrent_state)
        H.record(
            "optimized_full_context_conv_state_pcc", conv_pcc, weights=weights, kind=_kind(lut), seq_len=LONG_PROMPT
        )
        H.record(
            "optimized_full_context_recurrent_state_pcc", rec_pcc, weights=weights, kind=_kind(lut), seq_len=LONG_PROMPT
        )
        # Recorded, not asserted.  The carried state's *scale* drifts low at this context on real
        # weights - about 0.94 - and it does so at every fidelity and weight dtype measured
        # (``logs/probe_long_context_linear_real.log``), so it is not something this stage's precision
        # policy controls, and a threshold here would be a threshold on inherited behaviour that this
        # stage cannot move.  It is recorded because a drifting carried state should be visible in the
        # evidence rather than discovered by the stage that consumes the state.
        for metric, value in (
            ("optimized_full_context_conv_state_scale", H.scale_ratio(ref_conv, conv_state.T)),
            ("optimized_full_context_recurrent_state_scale", H.scale_ratio(ref_recurrent, recurrent_state)),
        ):
            H.record(metric, value, weights=weights, kind=_kind(lut), seq_len=LONG_PROMPT)
        assert conv_pcc >= bar, f"conv state after {LONG_PROMPT} tokens: PCC {conv_pcc}"
        assert rec_pcc >= bar, f"recurrent state after {LONG_PROMPT} tokens: PCC {rec_pcc}"

    tail_pcc = H.pcc(golden, got[:, -tail:, :])
    H.record(
        "optimized_full_context_prefill_tail_pcc",
        tail_pcc,
        weights=weights,
        kind=_kind(lut),
        seq_len=LONG_PROMPT,
        tail=tail,
    )
    assert tail_pcc >= bar, f"prefill tail at {LONG_PROMPT} tokens: PCC {tail_pcc}"
    tail_scale = H.scale_ratio(golden, got[:, -tail:, :])
    H.record(
        "optimized_full_context_prefill_tail_scale", tail_scale, weights=weights, kind=_kind(lut), seq_len=LONG_PROMPT
    )
    assert (
        SCALE_TOLERANCE[0] <= tail_scale <= SCALE_TOLERANCE[1]
    ), f"prefill tail at {LONG_PROMPT} tokens is scaled by {tail_scale}, outside {SCALE_TOLERANCE}"

    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=99)
    golden_decode = H.reference_decode(lut, token, LONG_PROMPT, cache)
    decoded = H.run_tt_decode(lut, token, torch.tensor([LONG_PROMPT]))
    decode_pcc = H.pcc(golden_decode, decoded)
    H.record("optimized_full_context_decode_pcc", decode_pcc, weights=weights, kind=_kind(lut), position=LONG_PROMPT)
    assert decode_pcc >= bar, f"decode at position {LONG_PROMPT}: PCC {decode_pcc}"
    decode_scale = H.scale_ratio(golden_decode, decoded)
    H.record(
        "optimized_full_context_decode_scale", decode_scale, weights=weights, kind=_kind(lut), position=LONG_PROMPT
    )
    assert (
        SCALE_TOLERANCE[0] <= decode_scale <= SCALE_TOLERANCE[1]
    ), f"decode at position {LONG_PROMPT} is scaled by {decode_scale}, outside {SCALE_TOLERANCE}"


def test_synthetic_bar_is_justified_by_the_real_weight_evidence():
    """:data:`SYNTHETIC_PCC_BAR` is a stress bar, and this pins what makes it one.

    Three things have to stay true for the looser synthetic bar to be legitimate rather than a
    waiver, and all three are read out of the committed probe artifacts rather than asserted here:

    1. the block-float weight error is essentially the same on the real and the synthetic tensor, so
       the stand-in is not intrinsically harder to quantise;
    2. the real layer's output norm is materially larger for the same input, which is what makes the
       same noise a larger fraction of the synthetic signal;
    3. the shipped policy clears the **real** acceptance bar with margin.

    If any of them stops holding, the looser bar has lost its justification and this fails.
    """
    doc = pathlib.Path(__file__).resolve().parents[1] / "doc" / "optimized_decoder" / "logs"
    rows = []
    for name in ("probe_blockfloat_distribution.log", "probe_real_weight_policy.log"):
        path = doc / name
        assert path.exists(), f"missing evidence {path}"
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("PROBEROW "):
                rows.append(json.loads(line[len("PROBEROW ") :]))

    quant = [r for r in rows if r.get("sweep") == "blockfloat_distribution"]
    assert quant, "no block-float quantisation rows"
    for row in quant:
        ratio = row["bfp4_err_ratio_synthetic_over_real"]
        assert 0.9 <= ratio <= 1.1, (
            f"{row['tensor']}: the synthetic tensor's BFP4 error is {ratio}x the real one's, so the "
            "stand-in is no longer 'the same weight error on a smaller signal' and the looser "
            "synthetic bar has lost its justification"
        )

    share = [r for r in rows if r.get("sweep") == "output_share"]
    assert share, "no output-norm rows"
    for row in share:
        assert row["real_output_norm"] > row["synthetic_output_norm"] * 1.5, (
            f"{row['kind']}: the real layer's output norm is no longer materially larger than the "
            "synthetic layer's, which was the measured reason 1 - PCC is larger on the stand-in"
        )

    real = [r for r in rows if r.get("sweep") == "real_weight_policy" and r.get("policy") == DEFAULT_POLICY.name]
    assert real, f"no real-weight rows for the shipped policy {DEFAULT_POLICY.name!r}"
    worst = min(min(r["prefill_pcc"], r["decode_pcc"], r["traced_decode_pcc"]) for r in real)
    assert worst >= H.PCC_BAR, (
        f"the shipped policy's worst real-weight PCC is {worst}, below the acceptance bar "
        f"{H.PCC_BAR}; the synthetic bar cannot be loosened to cover that"
    )
    assert SYNTHETIC_PCC_BAR < H.PCC_BAR, "the synthetic bar is supposed to be the looser one"
    H.record(
        "optimized_synthetic_bar_justification",
        {"real_weight_min_pcc": worst, "synthetic_bar": SYNTHETIC_PCC_BAR, "acceptance_bar": H.PCC_BAR},
    )

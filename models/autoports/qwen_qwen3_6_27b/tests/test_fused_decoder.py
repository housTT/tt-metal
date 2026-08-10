# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused-decoder tests for Qwen/Qwen3.6-27B.

Every test here builds :class:`FusedDecoder`, never :class:`FunctionalDecoder`.  The suite is
the functional suite's contract re-run against the fused graph - same HF reference, same
``PCC_BAR``, same sequence-length coverage, same paged-cache and state checks - plus four tests
that only make sense for a fused stage:

``test_fused_ops_are_dispatched``
    every op in ``FusedDecoder.FUSED_OPS`` is actually called during a real prefill/decode pass,
    so a silent fall back to the functional graph fails rather than quietly costing performance;
``test_fused_graph_is_smaller``
    the fused prefill/decode dispatch *fewer* device ops than the functional one on the same
    inputs, counted at the ``ttnn`` boundary;
``test_fused_matches_functional``
    fused and functional outputs agree with each other, not just with HF, from identical
    weights and inputs;
``test_repeated_runs_stable``
    repeated prefill+decode cycles stay bit-identical and leak no device memory.

Both HF layer kinds are covered with the real model config shapes: layer 0
(``linear_attention``) and layer 3 (``full_attention``).
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest
import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder

LAYER_KINDS = [
    pytest.param(H.LINEAR_LAYER_IDX, id="linear_attention"),
    pytest.param(H.FULL_LAYER_IDX, id="full_attention"),
]

# 1/17: sub-tile.  2048: exactly PREFILL_CHUNK.  2049: one past it.  4096: two whole chunks.
# 5000: long and divisible by none of {32 tile, 64 page/delta chunk, 256 SDPA chunk, 2048}.
SEQ_LENS = [1, 17, 128, 2048, 2049, 4096, 5000]

#: See LONG_TEST_TIMEOUT_NOTE in ``test_functional_decoder.py``: the repo-root ``pytest.ini``
#: sets a 300 s per-test timeout, which is far too short for the tests whose HF *reference* is
#: the slow part.  Those carry ``@pytest.mark.timeout(0)``.


def _stats():
    return ref.load_weight_stats()


def _kind(lut) -> str:
    return lut.config.layer_types[lut.layer_idx]


def _build(mesh_device, layer_idx, **kwargs) -> H.LayerUnderTest:
    """Build the layer under test.  Always the fused implementation."""
    lut = H.build_layer(mesh_device, layer_idx, decoder_cls=FusedDecoder, **kwargs)
    assert isinstance(lut.tt_layer, FusedDecoder), "this suite must exercise the fused path"
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
    """Fused paged prefill matches the HF layer for aligned and non-aligned lengths alike."""
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=max(8192, seq_len))
    value, _cache, _hidden = _prefill_and_check(lut, seq_len)
    H.record("fused_prefill_pcc", value, kind=_kind(lut), seq_len=seq_len)
    assert value >= H.PCC_BAR, f"prefill PCC {value} < {H.PCC_BAR} (seq_len={seq_len})"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", [17, 2048, 2049, 5000])
def test_decode_pcc(mesh_device, layer_idx, seq_len):
    """Fused paged decode continues a fused prefill correctly for several steps."""
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    value, cache, _hidden = _prefill_and_check(lut, seq_len)
    assert value >= H.PCC_BAR, f"prefill PCC {value} < {H.PCC_BAR}"
    H.prepare_decode(lut)

    for step in range(4):
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=100 + step)
        golden = H.reference_decode(lut, token, seq_len + step, cache)
        got = H.run_tt_decode(lut, token, torch.tensor([seq_len + step]))
        value = H.pcc(golden, got)
        H.record("fused_decode_pcc", value, kind=_kind(lut), prefill_len=seq_len, step=step)
        assert value >= H.PCC_BAR, f"decode PCC {value} < {H.PCC_BAR} at step {step}"


@pytest.mark.timeout(0)
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("batch", [4, 32])
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
        H.record("fused_batched_prefill_pcc", value, kind=_kind(lut), batch=batch, user_id=user_id, seq_len=seq_len)
        assert value >= H.PCC_BAR, f"prefill PCC {value} for user {user_id} (seq_len={seq_len})"
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
        H.record("fused_batched_decode_pcc", value, kind=_kind(lut), batch=batch, user_id=user_id)
        assert value >= H.PCC_BAR, f"decode PCC {value} for user {user_id}"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_real_weights(mesh_device, layer_idx):
    """The real Qwen3.6-27B checkpoint weights load into the fused layer and reach the bar."""
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, real_weights=True)
    value, cache, _hidden = _prefill_and_check(lut, 2049)
    H.record("fused_real_weight_prefill_pcc", value, kind=_kind(lut), seq_len=2049)
    assert value >= H.PCC_BAR, f"real-weight prefill PCC {value} < {H.PCC_BAR}"
    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=7)
    golden = H.reference_decode(lut, token, 2049, cache)
    got = H.run_tt_decode(lut, token, torch.tensor([2049]))
    value = H.pcc(golden, got)
    H.record("fused_real_weight_decode_pcc", value, kind=_kind(lut), position=2049)
    assert value >= H.PCC_BAR, f"real-weight decode PCC {value} < {H.PCC_BAR}"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_determinism(mesh_device, layer_idx):
    """Identical inputs produce bit-identical outputs, for prefill and for decode."""
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    hidden = ref.synthetic_hidden_states(lut.config, 1, 2049, _stats())
    first = H.run_tt_prefill(lut, hidden)
    second = H.run_tt_prefill(lut, hidden)
    H.record("fused_prefill_bit_identical", bool(torch.equal(first, second)), kind=_kind(lut), seq_len=2049, repeats=2)
    assert torch.equal(first, second), "prefill is not deterministic for identical inputs"

    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=11)
    H.prepare_decode(lut)
    decode_first = H.run_tt_decode(lut, token, torch.tensor([2049]))
    # prepare_decode_state() restores the batch-wide buffers - including the fused layer's
    # batch-major conv tap buffers - from the untouched per-user prefill state.
    H.prepare_decode(lut)
    decode_second = H.run_tt_decode(lut, token, torch.tensor([2049]))
    H.record("fused_decode_bit_identical", bool(torch.equal(decode_first, decode_second)), kind=_kind(lut), repeats=2)
    assert torch.equal(decode_first, decode_second), "decode is not deterministic for identical inputs"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_traced_decode_pcc(mesh_device, layer_idx):
    """Fused decode runs under ttnn traced execution and the replay output matches HF."""
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    seq_len = 2049
    value, cache, _hidden = _prefill_and_check(lut, seq_len)
    assert value >= H.PCC_BAR
    H.prepare_decode(lut)

    runner = H.TracedDecode(lut, batch=1)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=300)
    golden = H.reference_decode(lut, token, seq_len, cache)
    got = runner.warmup(token, torch.tensor([seq_len]))
    assert H.pcc(golden, got) >= H.PCC_BAR, "warm-up decode PCC below bar"

    runner.capture()
    for step in range(1, 4):
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=300 + step)
        golden = H.reference_decode(lut, token, seq_len + step, cache)
        got = runner.replay(token, torch.tensor([seq_len + step]))
        value = H.pcc(golden, got)
        H.record("fused_traced_decode_replay_pcc", value, kind=_kind(lut), replay=step, position=seq_len + step)
        assert value >= H.PCC_BAR, f"traced decode PCC {value} < {H.PCC_BAR} at replay {step}"
    runner.release()


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_traced_decode_batched(mesh_device, layer_idx):
    """Trace capture and replay at batch > 1, with per-user positions and page tables."""
    batch = 4
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
                "fused_batched_traced_decode_pcc", value, kind=_kind(lut), batch=batch, user_id=user_id, replay=step
            )
            assert value >= H.PCC_BAR, f"batched traced decode PCC {value} for user {user_id}"
    runner.release()


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_linear_state_and_kv_cache_match_reference(mesh_device, layer_idx):
    """The on-device cache/state after a fused prefill matches the HF cache, not just the output."""
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    seq_len = 2049
    value, cache, _hidden = _prefill_and_check(lut, seq_len)
    assert value >= H.PCC_BAR

    if lut.is_full_attention:
        keys, values = H.read_paged_kv(lut, user_id=0, seq_len=seq_len)
        ref_keys = cache.layers[lut.layer_idx].keys[0, :, :seq_len, :].to(torch.float32)
        ref_values = cache.layers[lut.layer_idx].values[0, :, :seq_len, :].to(torch.float32)
        k_pcc, v_pcc = H.pcc(ref_keys, keys), H.pcc(ref_values, values)
        H.record("fused_paged_k_cache_pcc", k_pcc, kind=_kind(lut), seq_len=seq_len)
        H.record("fused_paged_v_cache_pcc", v_pcc, kind=_kind(lut), seq_len=seq_len)
        assert k_pcc >= H.PCC_BAR, "paged K cache mismatch"
        assert v_pcc >= H.PCC_BAR, "paged V cache mismatch"
    else:
        conv_state, recurrent_state = H.read_linear_state(lut, user_id=0)
        ref_conv = cache.layers[lut.layer_idx].conv_states[0].to(torch.float32)
        ref_recurrent = cache.layers[lut.layer_idx].recurrent_states[0].to(torch.float32)
        conv_pcc, rec_pcc = H.pcc(ref_conv, conv_state.T), H.pcc(ref_recurrent, recurrent_state)
        H.record("fused_conv_state_pcc", conv_pcc, kind=_kind(lut), seq_len=seq_len)
        H.record("fused_recurrent_state_pcc", rec_pcc, kind=_kind(lut), seq_len=seq_len)
        assert conv_pcc >= H.PCC_BAR, "conv state mismatch"
        assert rec_pcc >= H.PCC_BAR, "recurrent state mismatch"


@pytest.mark.parametrize("block_size", [32, 128])
def test_alternate_page_block_size(mesh_device, block_size):
    """Page/block geometry is a parameter, not an assumption (``full_attention`` only)."""
    lut = _build(mesh_device, H.FULL_LAYER_IDX, max_batch=2, max_seq_len=4096, block_size=block_size)
    value, cache, _hidden = _prefill_and_check(lut, 2049, user_id=1)
    H.record("fused_alt_block_size_prefill_pcc", value, kind=_kind(lut), block_size=block_size, seq_len=2049)
    assert value >= H.PCC_BAR, f"prefill PCC {value} with block_size={block_size}"

    H.prepare_decode(lut)
    tokens = torch.cat([ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=500 + u) for u in range(2)], dim=0)
    got = H.run_tt_decode(lut, tokens, torch.tensor([2049, 2049]))
    golden = H.reference_decode(lut, tokens[1:2], 2049, cache)
    value = H.pcc(golden, got[1:2])
    H.record("fused_alt_block_size_decode_pcc", value, kind=_kind(lut), block_size=block_size, position=2049)
    assert value >= H.PCC_BAR, f"decode PCC {value} with block_size={block_size}"


@pytest.mark.parametrize("block_size", [96, 192])
def test_block_size_incompatible_with_prefill_chunk_is_rejected(mesh_device, block_size, expect_error):
    """A block size that does not divide the prefill chunk must fail loudly, not silently."""
    with expect_error(ValueError, "incompatible with PREFILL_CHUNK"):
        _build(mesh_device, H.FULL_LAYER_IDX, max_batch=1, max_seq_len=4096, block_size=block_size)


def test_bfloat8_kv_cache(mesh_device):
    """``cache_dtype=ttnn.bfloat8_b`` is a real supported configuration for the fused layer too."""
    lut = _build(mesh_device, H.FULL_LAYER_IDX, max_batch=1, max_seq_len=8192, cache_dtype=ttnn.bfloat8_b)
    assert lut.tt_layer.kv_cache[0].dtype == ttnn.bfloat8_b
    value, cache, _hidden = _prefill_and_check(lut, 2049)
    H.record("fused_bfp8_cache_prefill_pcc", value, kind=_kind(lut), seq_len=2049)
    assert value >= H.PCC_BAR, f"BFP8-cache prefill PCC {value} < {H.PCC_BAR}"

    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=23)
    golden = H.reference_decode(lut, token, 2049, cache)
    got = H.run_tt_decode(lut, token, torch.tensor([2049]))
    value = H.pcc(golden, got)
    H.record("fused_bfp8_cache_decode_pcc", value, kind=_kind(lut), position=2049)
    assert value >= H.PCC_BAR, f"BFP8-cache decode PCC {value} < {H.PCC_BAR}"


#: Lengths whose padded chunk exceeds the logical length by less than one tile - the ``ttnn.pad``
#: aliasing regression range inherited from the functional stage.  The fused conv's own aliasing
#: hazard - its state cut takes a tile-aligned block out of ``mixed_qkv`` first, and when the
#: padded chunk is at most two tiles that block is a *full-range* slice, i.e. a view of
#: ``mixed_qkv`` itself - is covered by the ``1`` and ``17`` cases of ``test_prefill_pcc``
#: instead, because it only arises for ``seq_len <= 64``.  Both ranges matter; neither may be
#: dropped.
PAD_ALIAS_SEQ_LENS = [735, 736, 737, 743, 767, 768]


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", PAD_ALIAS_SEQ_LENS)
def test_prefill_decode_pad_below_one_tile(mesh_device, layer_idx, seq_len):
    """Regression: logical lengths whose chunk padding is smaller than one tile."""
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=4096)
    value, cache, _hidden = _prefill_and_check(lut, seq_len)
    H.record("fused_pad_alias_prefill_pcc", value, kind=_kind(lut), seq_len=seq_len)
    assert value >= H.PCC_BAR, f"prefill PCC {value} < {H.PCC_BAR} (seq_len={seq_len})"

    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=17)
    golden = H.reference_decode(lut, token, seq_len, cache)
    got = H.run_tt_decode(lut, token, torch.tensor([seq_len]))
    value = H.pcc(golden, got)
    H.record("fused_pad_alias_decode_pcc", value, kind=_kind(lut), prefill_len=seq_len)
    assert value >= H.PCC_BAR, f"decode PCC {value} < {H.PCC_BAR} (prefill_len={seq_len})"


LONG_PCC_SEQ_LEN = {"linear_attention": 16385, "full_attention": 8191}


@pytest.mark.timeout(0)
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_prefill_pcc_long(mesh_device, layer_idx):
    """HF-vs-TTNN prefill PCC at the longest length the CPU reference can still reach."""
    config = ref.load_text_config()
    seq_len = LONG_PCC_SEQ_LEN[config.layer_types[layer_idx]]
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=seq_len + 2048)
    value, _cache, _hidden = _prefill_and_check(lut, seq_len)
    H.record("fused_long_prefill_pcc", value, kind=_kind(lut), seq_len=seq_len)
    assert value >= H.PCC_BAR, f"long prefill PCC {value} < {H.PCC_BAR} (seq_len={seq_len})"


# ------------------------------------------------------------------------- fusing-specific


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_fused_ops_are_dispatched(mesh_device, layer_idx):
    """Every dedicated op the fused graph claims is really called on a prefill/decode pass.

    Without this, a shape or dtype that silently missed a fused path would still pass every PCC
    test in this file - at the functional layer's op count.
    """
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    calls = _count_ops(FusedDecoder.FUSED_OPS)
    with calls:
        hidden = ref.synthetic_hidden_states(lut.config, 1, 2049, _stats())
        H.run_tt_prefill(lut, hidden)
        H.prepare_decode(lut)
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=31)
        H.run_tt_decode(lut, token, torch.tensor([2049]))

    kind = _kind(lut)
    # Which dedicated ops a pass must dispatch depends on the layer kind: the gated delta rule is
    # linear_attention's mixer, RoPE is full_attention's.
    expected = {
        "linear_attention": ["ttnn.transformer.chunk_gated_delta_rule"],
        "full_attention": ["ttnn.experimental.rotary_embedding_hf", "ttnn.experimental.rotate_half"],
    }[kind]
    for name in expected:
        # Recorded as a dict, not a bare int: ``scripts/collect_evidence.py`` folds every *numeric*
        # record that is not a ``_scale`` into the PCC minimum, and an op-call count of 2 is not a
        # PCC. Non-numeric values are carried through untouched.
        H.record("fused_op_calls", {"calls": calls.counts[name]}, kind=kind, op=name)
        assert calls.counts[name] > 0, f"{name} was never dispatched on a {kind} pass: {calls.counts}"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_fused_graph_is_smaller(mesh_device, layer_idx):
    """The fused graph dispatches strictly fewer ttnn ops than the functional one.

    Counted at the ``ttnn`` python boundary over one identical prefill and one identical decode.
    This is a topology check, not a performance claim - the latency evidence is the
    ``tt-perf-report`` tables in ``doc/fused_decoder/`` - but it is what catches a rewrite that
    was reverted or never taken.
    """
    counts = {}
    for name, cls in (("functional", FunctionalDecoder), ("fused", FusedDecoder)):
        lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, decoder_cls=cls)
        hidden = ref.synthetic_hidden_states(lut.config, 1, 2048, _stats())
        counter = _count_all_ops()
        with counter:
            H.run_tt_prefill(lut, hidden)
        prefill = counter.total
        H.prepare_decode(lut)
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=41)
        counter = _count_all_ops()
        with counter:
            H.run_tt_decode(lut, token, torch.tensor([2048]))
        counts[name] = (prefill, counter.total)
        H.release_layers()

    H.record(
        "fused_op_counts",
        {"functional": list(counts["functional"]), "fused": list(counts["fused"])},
        kind=_kind(lut),
    )
    assert counts["fused"][0] < counts["functional"][0], f"prefill op count did not fall: {counts}"
    assert counts["fused"][1] < counts["functional"][1], f"decode op count did not fall: {counts}"


def _dram_allocated(mesh_device) -> int:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return view.total_bytes_allocated_per_bank * view.num_banks


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_fused_persistent_state_delta(mesh_device, layer_idx):
    """How much persistent device memory fusing costs, measured rather than asserted.

    The capacity claim in ``doc/context_contract.json`` rests on this: the fused layer adds
    per-tap conv-state buffers, the delta-rule constant tiles, the two gated-norm constant
    matrices, bfloat16 conv taps and the packed ``a``/``b`` weight, and it frees the two
    unpacked ``a``/``b`` weights the packed one replaces.  The number recorded here is the whole
    of that, read off the device allocator around a real ``from_state_dict``.
    """
    implementations = (("functional", FunctionalDecoder), ("fused", FusedDecoder))
    # Warm-up build/release of each: the *first* layer built in a session also allocates one-time
    # buffers that never come back, which would otherwise land on whichever implementation ran
    # first (measured at 114688 bytes, enough to make the full_attention delta come out negative).
    for _name, cls in implementations:
        H.build_layer(mesh_device, layer_idx, max_batch=32, max_seq_len=8192, decoder_cls=cls)
        H.release_layers()

    allocated = {}
    for name, cls in implementations:
        before = _dram_allocated(mesh_device)
        H.build_layer(mesh_device, layer_idx, max_batch=32, max_seq_len=8192, decoder_cls=cls)
        allocated[name] = _dram_allocated(mesh_device) - before
        H.release_layers()

    delta = allocated["fused"] - allocated["functional"]
    H.record(
        "fused_persistent_dram_bytes",
        {"functional": allocated["functional"], "fused": allocated["fused"], "delta": delta},
        kind=ref.load_text_config().layer_types[layer_idx],
        max_batch=32,
    )
    # A hard ceiling, not a target: anything approaching the KV cache of a single full-context
    # layer would be a capacity change and belongs in the context contract as one.
    assert delta < 64 * 1024 * 1024, f"fusing added {delta} bytes of persistent device state"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_fused_matches_functional(mesh_device, layer_idx):
    """Fused and functional outputs agree with each other from identical weights and inputs."""
    outputs = {}
    for name, cls in (("functional", FunctionalDecoder), ("fused", FusedDecoder)):
        lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, decoder_cls=cls)
        hidden = ref.synthetic_hidden_states(lut.config, 1, 2049, _stats())
        prefill = H.run_tt_prefill(lut, hidden)
        H.prepare_decode(lut)
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=51)
        decode = H.run_tt_decode(lut, token, torch.tensor([2049]))
        outputs[name] = (prefill, decode)
        H.release_layers()

    for index, phase in enumerate(("prefill", "decode")):
        value = H.pcc(outputs["functional"][index], outputs["fused"][index])
        H.record("fused_vs_functional_pcc", value, kind=_kind(lut), phase=phase, seq_len=2049)
        assert value >= H.PCC_BAR, f"fused vs functional {phase} PCC {value} < {H.PCC_BAR}"


@pytest.mark.timeout(0)
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_repeated_runs_stable(mesh_device, layer_idx):
    """Stress: repeated prefill+decode cycles stay bit-identical and free every allocation.

    Six cycles of a 2049-token prefill plus four decode steps.  Bit-identical output across
    cycles rules out state that leaks from one cycle into the next (the fused layer carries more
    persistent buffers than the functional one - the batch-major conv taps and the in-place
    recurrent-state write-back), and the device allocation statistics rule out a per-cycle leak
    in the fused graph's ``_free`` aliasing rules.
    """
    cycles = 6
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    hidden = ref.synthetic_hidden_states(lut.config, 1, 2049, _stats())
    tokens = [ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=700 + s) for s in range(4)]

    baseline_free = None
    reference_out = None
    for cycle in range(cycles):
        prefill = H.run_tt_prefill(lut, hidden)
        H.prepare_decode(lut)
        decodes = [H.run_tt_decode(lut, tokens[s], torch.tensor([2049 + s])) for s in range(4)]
        assert torch.isfinite(prefill).all() and all(torch.isfinite(d).all() for d in decodes)
        result = (prefill, torch.cat(decodes, dim=0))
        if reference_out is None:
            reference_out = result
        else:
            assert torch.equal(reference_out[0], result[0]), f"prefill drifted on cycle {cycle}"
            assert torch.equal(reference_out[1], result[1]), f"decode drifted on cycle {cycle}"
        allocated = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank
        if cycle == 1:  # cycle 0 still allocates first-use scratch and compiled program buffers
            baseline_free = allocated
        elif baseline_free is not None:
            assert allocated == baseline_free, (
                f"DRAM bytes allocated per bank moved from {baseline_free} to {allocated} on cycle {cycle}: "
                "the fused graph is leaking a device buffer per cycle"
            )
    H.record("fused_repeated_run_bit_identical", True, kind=_kind(lut), cycles=cycles, decode_steps=4)


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_no_runtime_host_fallback(mesh_device, layer_idx):
    """A single measured fused prefill/decode pass makes no torch or host-transfer calls."""
    module = sys.modules[FusedDecoder.__module__]
    source = pathlib.Path(module.__file__).read_text()
    # Scan every line of the implementation except the module docstring and the body of
    # ``from_state_dict``, the documented setup-time torch boundary.  Same anchors and same rule
    # as the functional suite's scan.
    docstring_end = source.index('"""', source.index('"""') + 3) + 3
    class_anchor = f"class {FusedDecoder.__name__}"
    helper_region = source[docstring_end:].split(class_anchor, 1)[0]
    class_body = source.split(class_anchor, 1)[1]
    constructor_region = class_body.split("    @classmethod", 1)[0]
    body = source.split("def from_state_dict", 1)
    assert len(body) == 2, "from_state_dict is the documented torch boundary"
    after = body[1].split("    # ------------------------------------------------------------- primitives", 1)[1]
    for region, label in (
        (helper_region, "module helpers"),
        (constructor_region, "__init__"),
        (after, "layer runtime"),
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


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_no_relayout_or_host_ops_in_measured_decode(mesh_device, layer_idx):
    """The measured decode path asks for no layout conversion, and only contract-forced reshards.

    ``ttnn.tilize*`` / ``ttnn.untilize*`` / ``ttnn.to_layout`` are how a graph pays for a layout
    it should not have needed, and ``interleaved_to_sharded`` / ``sharded_to_interleaved`` /
    ``to_memory_config`` for a memory config it should not have needed.  This counts what the
    *layer itself* asks for; relayouts a dedicated op performs internally (``repeat_interleave``
    still untilizes to expand the GQA head axis) are visible in the
    ``tt-perf-report`` tables and are discussed in ``doc/fused_decoder/README.md``.

    The reshards that remain are each forced by a dedicated op's contract - the width-sharded RMS
    norm program config, and ``paged_update_cache`` / ``nlp_concat_heads_decode`` wanting
    height-sharded heads - and this test pins that set so a new unnecessary one cannot appear
    unnoticed.
    """
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    hidden = ref.synthetic_hidden_states(lut.config, 1, 2048, _stats())
    H.run_tt_prefill(lut, hidden)
    H.prepare_decode(lut)

    watched = ("tilize", "tilize_with_val_padding", "untilize", "untilize_with_unpadding", "to_layout")
    resharding = ("interleaved_to_sharded", "sharded_to_interleaved", "to_memory_config")
    counter = _count_ops(tuple(f"ttnn.{name}" for name in watched + resharding))
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=61)
    with counter:
        H.run_tt_decode(lut, token, torch.tensor([2048]))

    layout_calls = {k: v for k, v in counter.counts.items() if k.rsplit(".", 1)[1] in watched and v}
    assert not layout_calls, f"measured decode does a layout conversion: {layout_calls}"

    # Reshards that remain, and the op contract that forces each one:
    #   2 x width-sharded RMS norm       -> 2 to_memory_config + 2 sharded_to_interleaved
    #   full_attention only: Q and K off the head op, K back on, the SDPA output on for
    #                        nlp_concat_heads_decode, and its result back off
    reshards = sum(v for k, v in counter.counts.items() if k.rsplit(".", 1)[1] in resharding)
    budget = 6 if not lut.is_full_attention else 12
    # A dict, not a bare int, for the same reason as ``fused_op_calls`` above.
    H.record("fused_decode_reshard_ops", {"reshards": reshards, "budget": budget}, kind=_kind(lut))
    assert reshards <= budget, f"measured decode reshards {reshards} times (budget {budget}): {counter.counts}"


# ------------------------------------------------------------------ full advertised context

LONG_TAIL = {"linear_attention": 8192, "full_attention": 256}
LONG_SEGMENT = 16384
LONG_PROMPT = 262143
SCALE_TOLERANCE = (0.98, 1.02)


@pytest.mark.timeout(0)
@pytest.mark.long_context
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_full_advertised_context(mesh_device, layer_idx, request):
    """Prefill 262143 tokens and decode at position 262143 through the fused graph.

    Same reference construction as the functional stage's test of the same name: a segmented HF
    run for ``linear_attention``, and for ``full_attention`` a K/V cache built from
    ``k_proj``/``v_proj`` + ``k_norm`` + RoPE (validated ``torch.equal`` against a genuine short
    reference prefill) with the real HF layer run over the last :data:`LONG_TAIL` queries.
    """
    if not request.config.getoption("--long-context"):
        pytest.skip("needs --long-context")
    context = ref.load_text_config().max_position_embeddings
    assert LONG_PROMPT + 1 == context
    lut = _build(mesh_device, layer_idx, max_batch=1, max_seq_len=context)
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
        H.record("fused_full_context_paged_k_cache_pcc", k_pcc, kind=_kind(lut), seq_len=LONG_PROMPT)
        H.record("fused_full_context_paged_v_cache_pcc", v_pcc, kind=_kind(lut), seq_len=LONG_PROMPT)
        assert k_pcc >= H.PCC_BAR, f"paged K cache at {LONG_PROMPT} tokens: PCC {k_pcc}"
        assert v_pcc >= H.PCC_BAR, f"paged V cache at {LONG_PROMPT} tokens: PCC {v_pcc}"
    else:
        golden, cache = H.reference_prefill_segmented(lut, hidden, LONG_SEGMENT)
        golden = golden[:, -tail:, :]

        conv_state, recurrent_state = H.read_linear_state(lut, user_id=0)
        ref_conv = cache.layers[lut.layer_idx].conv_states[0].to(torch.float32)
        ref_recurrent = cache.layers[lut.layer_idx].recurrent_states[0].to(torch.float32)
        conv_pcc = H.pcc(ref_conv, conv_state.T)
        rec_pcc = H.pcc(ref_recurrent, recurrent_state)
        H.record("fused_full_context_conv_state_pcc", conv_pcc, kind=_kind(lut), seq_len=LONG_PROMPT)
        H.record("fused_full_context_recurrent_state_pcc", rec_pcc, kind=_kind(lut), seq_len=LONG_PROMPT)
        assert conv_pcc >= H.PCC_BAR, f"conv state after {LONG_PROMPT} tokens: PCC {conv_pcc}"
        assert rec_pcc >= H.PCC_BAR, f"recurrent state after {LONG_PROMPT} tokens: PCC {rec_pcc}"

    tail_pcc = H.pcc(golden, got[:, -tail:, :])
    H.record("fused_full_context_prefill_tail_pcc", tail_pcc, kind=_kind(lut), seq_len=LONG_PROMPT, tail=tail)
    assert tail_pcc >= H.PCC_BAR, f"prefill tail at {LONG_PROMPT} tokens: PCC {tail_pcc}"
    tail_scale = H.scale_ratio(golden, got[:, -tail:, :])
    H.record("fused_full_context_prefill_tail_scale", tail_scale, kind=_kind(lut), seq_len=LONG_PROMPT)
    assert (
        SCALE_TOLERANCE[0] <= tail_scale <= SCALE_TOLERANCE[1]
    ), f"prefill tail at {LONG_PROMPT} tokens is scaled by {tail_scale}, outside {SCALE_TOLERANCE}"

    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=99)
    golden_decode = H.reference_decode(lut, token, LONG_PROMPT, cache)
    decoded = H.run_tt_decode(lut, token, torch.tensor([LONG_PROMPT]))
    decode_pcc = H.pcc(golden_decode, decoded)
    H.record("fused_full_context_decode_pcc", decode_pcc, kind=_kind(lut), position=LONG_PROMPT)
    assert decode_pcc >= H.PCC_BAR, f"decode at position {LONG_PROMPT}: PCC {decode_pcc}"
    decode_scale = H.scale_ratio(golden_decode, decoded)
    H.record("fused_full_context_decode_scale", decode_scale, kind=_kind(lut), position=LONG_PROMPT)
    assert (
        SCALE_TOLERANCE[0] <= decode_scale <= SCALE_TOLERANCE[1]
    ), f"decode at position {LONG_PROMPT} is scaled by {decode_scale}, outside {SCALE_TOLERANCE}"


# ------------------------------------------------------------------------------- plumbing


class _OpCounter:
    """Context manager counting calls to a set of dotted ``ttnn`` op names."""

    def __init__(self, names):
        self.names = tuple(names)
        self.counts = {name: 0 for name in self.names}
        self._saved = {}

    def _resolve(self, dotted):
        parts = dotted.split(".")
        owner = ttnn
        for part in parts[1:-1]:
            owner = getattr(owner, part)
        return owner, parts[-1]

    def __enter__(self):
        for name in self.names:
            owner, attr = self._resolve(name)
            original = getattr(owner, attr)
            self._saved[name] = (owner, attr, original)

            def make(op_name, func):
                def wrapper(*args, **kwargs):
                    self.counts[op_name] += 1
                    return func(*args, **kwargs)

                return wrapper

            setattr(owner, attr, make(name, original))
        return self

    def __exit__(self, *exc):
        for owner, attr, original in self._saved.values():
            setattr(owner, attr, original)
        return False


class _AllOpCounter:
    """Context manager counting every ``ttnn`` op dispatched from python."""

    #: ttnn namespaces whose callables are wrapped.  Keeping the set explicit avoids counting
    #: python-level helpers (``ttnn.Shape``, config objects, ...) as device ops.
    _SKIP = {
        "deallocate",
        "synchronize_device",
        "to_torch",
        "from_torch",
        "get_memory_view",
        "num_cores_to_corerangeset",
        "create_sharded_memory_config",
    }

    def __init__(self):
        self.total = 0
        self._saved = []

    def __enter__(self):
        import ttnn as _ttnn

        owners = [_ttnn, _ttnn.experimental, _ttnn.transformer]
        for owner in owners:
            for attr in dir(owner):
                if attr.startswith("_") or attr in self._SKIP:
                    continue
                value = getattr(owner, attr, None)
                if not callable(value) or isinstance(value, type):
                    continue
                if type(value).__name__ not in ("FastOperation", "Operation", "builtin_function_or_method", "nb_func"):
                    continue
                self._saved.append((owner, attr, value))

                def make(func):
                    def wrapper(*args, **kwargs):
                        self.total += 1
                        return func(*args, **kwargs)

                    return wrapper

                setattr(owner, attr, make(value))
        return self

    def __exit__(self, *exc):
        for owner, attr, original in self._saved:
            setattr(owner, attr, original)
        return False


def _count_ops(names):
    return _OpCounter(names)


def _count_all_ops():
    return _AllOpCounter()

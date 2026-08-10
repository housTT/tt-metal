# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Functional-decoder tests for Qwen/Qwen3.6-27B.

Both HF layer kinds are covered with the **real** model config shapes:

* layer 0 — ``linear_attention`` (``Qwen3_5GatedDeltaNet``)
* layer 3 — ``full_attention``   (``Qwen3_5Attention``, gated, partial RoPE, paged KV)

Sequence-length coverage is deliberately un-friendly: ``1`` and ``17`` (sub-tile), ``2048``
(exactly the prefill chunk), ``2049`` (just across it), ``4096`` (two whole chunks) and
``5000`` (long, divisible by neither the chunk, the page, the delta chunk nor the tile).
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

LAYER_KINDS = [
    pytest.param(H.LINEAR_LAYER_IDX, id="linear_attention"),
    pytest.param(H.FULL_LAYER_IDX, id="full_attention"),
]

# 1/17: sub-tile.  2048: exactly PREFILL_CHUNK.  2049: one past it.  4096: two whole chunks.
# 5000: long and divisible by none of {32 tile, 64 page/delta chunk, 256 SDPA chunk, 2048}.
SEQ_LENS = [1, 17, 128, 2048, 2049, 4096, 5000]

#: LONG_TEST_TIMEOUT_NOTE: the repo-root ``pytest.ini`` sets a 300 s per-test timeout.  That is
#: right for ordinary tests and far too short for the tests whose HF *reference* is the slow
#: part - a 262143-token segmented reference, 32 users of eager attention, or a 16385-token
#: single-shot reference.  Those three carry ``@pytest.mark.timeout(0)`` so the global setting
#: keeps guarding everything else.
#:


def _stats():
    return ref.load_weight_stats()


def _kind(lut) -> str:
    return lut.config.layer_types[lut.layer_idx]


def _prefill_and_check(lut, seq_len, seed=0, user_id=0, cache=None):
    hidden = ref.synthetic_hidden_states(lut.config, 1, seq_len, _stats(), seed=seed)
    cache = cache if cache is not None else DynamicCache(config=lut.config)
    golden = H.reference_prefill(lut, hidden, cache)
    got = H.run_tt_prefill(lut, hidden, user_id=user_id)
    return H.pcc(golden, got), cache, hidden


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_prefill_pcc(mesh_device, layer_idx, seq_len):
    """Paged prefill matches the HF layer for aligned and non-aligned lengths alike."""
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=max(8192, seq_len))
    value, _cache, _hidden = _prefill_and_check(lut, seq_len)
    H.record("prefill_pcc", value, kind=_kind(lut), seq_len=seq_len)
    assert value >= H.PCC_BAR, f"prefill PCC {value} < {H.PCC_BAR} (seq_len={seq_len})"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", [17, 2048, 2049, 5000])
def test_decode_pcc(mesh_device, layer_idx, seq_len):
    """Paged decode continues a prefill correctly for several steps."""
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    value, cache, _hidden = _prefill_and_check(lut, seq_len)
    assert value >= H.PCC_BAR, f"prefill PCC {value} < {H.PCC_BAR}"
    H.prepare_decode(lut)

    for step in range(4):
        token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=100 + step)
        golden = H.reference_decode(lut, token, seq_len + step, cache)
        got = H.run_tt_decode(lut, token, torch.tensor([seq_len + step]))
        value = H.pcc(golden, got)
        H.record("decode_pcc", value, kind=_kind(lut), prefill_len=seq_len, step=step)
        assert value >= H.PCC_BAR, f"decode PCC {value} < {H.PCC_BAR} at step {step}"


@pytest.mark.timeout(0)  # see LONG_TEST_TIMEOUT_NOTE
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("batch", [4, 32])
def test_batched_users(mesh_device, layer_idx, batch):
    """Per-user page tables, cache slots and current positions with unequal prompt lengths."""
    seq_lens = [64 + 97 * u for u in range(batch)]
    lut = H.build_layer(mesh_device, layer_idx, max_batch=batch, max_seq_len=8192)

    caches = []
    for user_id, seq_len in enumerate(seq_lens):
        hidden = ref.synthetic_hidden_states(lut.config, 1, seq_len, _stats(), seed=user_id)
        cache = DynamicCache(config=lut.config)
        golden = H.reference_prefill(lut, hidden, cache)
        got = H.run_tt_prefill(lut, hidden, user_id=user_id)
        value = H.pcc(golden, got)
        H.record("batched_prefill_pcc", value, kind=_kind(lut), batch=batch, user_id=user_id, seq_len=seq_len)
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
        H.record("batched_decode_pcc", value, kind=_kind(lut), batch=batch, user_id=user_id)
        assert value >= H.PCC_BAR, f"decode PCC {value} for user {user_id}"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_real_weights(mesh_device, layer_idx):
    """The real Qwen3.6-27B checkpoint weights load and reach the PCC bar."""
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, real_weights=True)
    value, cache, _hidden = _prefill_and_check(lut, 2049)
    H.record("real_weight_prefill_pcc", value, kind=_kind(lut), seq_len=2049)
    assert value >= H.PCC_BAR, f"real-weight prefill PCC {value} < {H.PCC_BAR}"
    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=7)
    golden = H.reference_decode(lut, token, 2049, cache)
    got = H.run_tt_decode(lut, token, torch.tensor([2049]))
    value = H.pcc(golden, got)
    H.record("real_weight_decode_pcc", value, kind=_kind(lut), position=2049)
    assert value >= H.PCC_BAR, f"real-weight decode PCC {value} < {H.PCC_BAR}"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_determinism(mesh_device, layer_idx):
    """Identical inputs produce bit-identical outputs, for prefill and for decode."""
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    hidden = ref.synthetic_hidden_states(lut.config, 1, 2049, _stats())
    first = H.run_tt_prefill(lut, hidden)
    second = H.run_tt_prefill(lut, hidden)
    H.record("prefill_bit_identical", bool(torch.equal(first, second)), kind=_kind(lut), seq_len=2049, repeats=2)
    assert torch.equal(first, second), "prefill is not deterministic for identical inputs"

    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=11)
    H.prepare_decode(lut)
    decode_first = H.run_tt_decode(lut, token, torch.tensor([2049]))
    # prepare_decode_state() restores the batch-wide buffers from the untouched per-user
    # prefill state, so the second call starts from exactly the same state as the first.
    H.prepare_decode(lut)
    decode_second = H.run_tt_decode(lut, token, torch.tensor([2049]))
    H.record("decode_bit_identical", bool(torch.equal(decode_first, decode_second)), kind=_kind(lut), repeats=2)
    assert torch.equal(decode_first, decode_second), "decode is not deterministic for identical inputs"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_traced_decode_pcc(mesh_device, layer_idx):
    """Decode runs under ttnn traced execution and the replay output matches HF."""
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    seq_len = 2049
    value, cache, _hidden = _prefill_and_check(lut, seq_len)
    assert value >= H.PCC_BAR
    H.prepare_decode(lut)

    runner = H.TracedDecode(lut, batch=1)
    # Warm-up / compile step: executes eagerly and advances the layer state by one token.
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
        H.record("traced_decode_replay_pcc", value, kind=_kind(lut), replay=step, position=seq_len + step)
        assert value >= H.PCC_BAR, f"traced decode PCC {value} < {H.PCC_BAR} at replay {step}"
    runner.release()


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_no_runtime_host_fallback(mesh_device, layer_idx):
    """A single measured prefill/decode pass makes no torch or host-transfer calls."""
    decoder_cls = FunctionalDecoder
    module = sys.modules[decoder_cls.__module__]
    class_anchor = f"class {decoder_cls.__name__}"
    source = pathlib.Path(module.__file__).read_text()
    # Scan every line of the implementation except the module docstring and the body of
    # ``from_state_dict``, the documented setup-time torch boundary.  That leaves the
    # module-level helpers, ``__init__`` and everything after the setup section.
    # Everything after the module docstring and before the class: this deliberately starts at the
    # end of the docstring rather than at a named function, so a helper added above the first one
    # is still scanned.
    docstring_end = source.index('"""', source.index('"""') + 3) + 3
    helper_region = source[docstring_end:].split(class_anchor, 1)[0]
    class_body = source.split(class_anchor, 1)[1]
    constructor_region = class_body.split("    @classmethod", 1)[0]
    body = source.split("def from_state_dict", 1)
    assert len(body) == 2, "from_state_dict is the documented torch boundary"
    after = body[1].split("# ------------------------------------------------------------- primitives", 1)[1]
    regions = (
        (helper_region, "module helpers"),
        (constructor_region, "__init__"),
        (after, "layer runtime"),
    )
    for region, label in regions:
        assert region.strip(), f"{label} region came out empty - the scan anchors moved"
        # Comment lines are prose - both modules discuss torch in their commentary (e.g. "a
        # float32 torch attention on identical inputs" in the SDPA k-chunk note) - so the scan
        # looks at code only.
        code = "\n".join(l for l in region.splitlines() if not l.lstrip().startswith("#"))
        for pattern in (r"\btorch\b", r"from_torch", r"to_torch", r"\bas_tensor\b", r"\.cpu\(\)"):
            assert not re.search(pattern, code), f"{label} references {pattern!r}"

    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
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
def test_linear_state_and_kv_cache_match_reference(mesh_device, layer_idx):
    """The on-device cache/state after prefill matches the HF cache, not just the output."""
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    seq_len = 2049
    value, cache, _hidden = _prefill_and_check(lut, seq_len)
    assert value >= H.PCC_BAR

    if lut.is_full_attention:
        keys, values = H.read_paged_kv(lut, user_id=0, seq_len=seq_len)
        ref_keys = cache.layers[lut.layer_idx].keys[0, :, :seq_len, :].to(torch.float32)
        ref_values = cache.layers[lut.layer_idx].values[0, :, :seq_len, :].to(torch.float32)
        k_pcc, v_pcc = H.pcc(ref_keys, keys), H.pcc(ref_values, values)
        H.record("paged_k_cache_pcc", k_pcc, kind=_kind(lut), seq_len=seq_len)
        H.record("paged_v_cache_pcc", v_pcc, kind=_kind(lut), seq_len=seq_len)
        assert k_pcc >= H.PCC_BAR, "paged K cache mismatch"
        assert v_pcc >= H.PCC_BAR, "paged V cache mismatch"
    else:
        conv_state, recurrent_state = H.read_linear_state(lut, user_id=0)
        ref_conv = cache.layers[lut.layer_idx].conv_states[0].to(torch.float32)  # [conv_dim, K]
        ref_recurrent = cache.layers[lut.layer_idx].recurrent_states[0].to(torch.float32)
        conv_pcc, rec_pcc = H.pcc(ref_conv, conv_state.T), H.pcc(ref_recurrent, recurrent_state)
        H.record("conv_state_pcc", conv_pcc, kind=_kind(lut), seq_len=seq_len)
        H.record("recurrent_state_pcc", rec_pcc, kind=_kind(lut), seq_len=seq_len)
        assert conv_pcc >= H.PCC_BAR, "conv state mismatch"
        assert rec_pcc >= H.PCC_BAR, "recurrent state mismatch"


@pytest.mark.parametrize("block_size", [32, 128])
def test_alternate_page_block_size(mesh_device, block_size):
    """Page/block geometry is a parameter, not an assumption.

    ``full_attention`` only: ``linear_attention`` has no paged cache, so a block-size
    parametrisation there would assert nothing.
    """
    layer_idx = H.FULL_LAYER_IDX
    lut = H.build_layer(mesh_device, layer_idx, max_batch=2, max_seq_len=4096, block_size=block_size)
    value, cache, _hidden = _prefill_and_check(lut, 2049, user_id=1)
    H.record("alt_block_size_prefill_pcc", value, kind=_kind(lut), block_size=block_size, seq_len=2049)
    assert value >= H.PCC_BAR, f"prefill PCC {value} with block_size={block_size}"

    # Prefill alone would not catch a block-size bug in the *decode* path: paged_update_cache
    # and paged_scaled_dot_product_attention_decode index the page table themselves, and the
    # decode SDPA k chunk is a fixed 512 tokens regardless of the block size.
    H.prepare_decode(lut)
    tokens = torch.cat([ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=500 + u) for u in range(2)], dim=0)
    positions = torch.tensor([2049, 2049])
    got = H.run_tt_decode(lut, tokens, positions)
    golden = H.reference_decode(lut, tokens[1:2], 2049, cache)
    value = H.pcc(golden, got[1:2])
    H.record("alt_block_size_decode_pcc", value, kind=_kind(lut), block_size=block_size, position=2049)
    assert value >= H.PCC_BAR, f"decode PCC {value} with block_size={block_size}"


@pytest.mark.parametrize("block_size", [96, 192])
def test_block_size_incompatible_with_prefill_chunk_is_rejected(mesh_device, block_size, expect_error):
    """A block size that does not divide the prefill chunk must fail loudly, not silently.

    ``paged_update_cache`` only needs ``block_size % TILE_HEIGHT == 0``, so 96 is structurally
    legal, but ``lcm(SDPA_CHUNK, 96) = 768`` does not divide ``PREFILL_CHUNK`` and chunk N's
    page-table slice would start inside chunk N-1's tokens - a cache corruption that every
    length assertion in this file would still pass.
    """
    with expect_error(ValueError, "incompatible with PREFILL_CHUNK"):
        H.build_layer(mesh_device, H.FULL_LAYER_IDX, max_batch=1, max_seq_len=4096, block_size=block_size)


# Lengths whose padded chunk exceeds the logical length by **less than one tile**, i.e.
# ceil(len/32)*32 == round_up(len, alignment).  These are exactly the lengths that the
# ttnn.pad aliasing hazard corrupted: pad returns a view of its input when the padding fits
# inside the existing tile padding, so freeing the input frees the result too.  The hazard and
# the _free() guard that fixes it are in tt/functional_decoder.py; this range is its regression
# test (doc/functional_decoder/work_log.md section 0 records which pass found it).  735/736 are the
# neighbouring lengths that stayed correct and pin the boundary.
PAD_ALIAS_SEQ_LENS = [735, 736, 737, 743, 767, 768]


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", PAD_ALIAS_SEQ_LENS)
def test_prefill_decode_pad_below_one_tile(mesh_device, layer_idx, seq_len):
    """Regression: logical lengths whose chunk padding is smaller than one tile.

    Both halves matter. Prefill output catches a corrupted *input* chunk, and the decode step
    catches a corrupted *state*: the earlier gated-delta-net padding bug left prefill output
    correct while the recurrent state was wrong.
    """
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=4096)
    value, cache, _hidden = _prefill_and_check(lut, seq_len)
    H.record("pad_alias_prefill_pcc", value, kind=_kind(lut), seq_len=seq_len)
    assert value >= H.PCC_BAR, f"prefill PCC {value} < {H.PCC_BAR} (seq_len={seq_len})"

    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=17)
    golden = H.reference_decode(lut, token, seq_len, cache)
    got = H.run_tt_decode(lut, token, torch.tensor([seq_len]))
    value = H.pcc(golden, got)
    H.record("pad_alias_decode_pcc", value, kind=_kind(lut), prefill_len=seq_len)
    assert value >= H.PCC_BAR, f"decode PCC {value} < {H.PCC_BAR} (prefill_len={seq_len})"


def test_bfloat8_kv_cache(mesh_device):
    """``cache_dtype=ttnn.bfloat8_b`` is a real supported configuration, not just a kwarg.

    Prefill must cast K/V to the cache dtype before ``paged_fill_cache``, while decode must
    keep the ``paged_update_cache`` update tensors in a supported compute dtype. BFP8 has a
    shared exponent per 16 values, so it is a lossier storage format than the BF16 default; it
    still clears the stage bar, and is asserted at that bar like everything else.
    """
    lut = H.build_layer(mesh_device, H.FULL_LAYER_IDX, max_batch=1, max_seq_len=8192, cache_dtype=ttnn.bfloat8_b)
    assert lut.tt_layer.kv_cache[0].dtype == ttnn.bfloat8_b
    value, cache, _hidden = _prefill_and_check(lut, 2049)
    H.record("bfp8_cache_prefill_pcc", value, kind=_kind(lut), seq_len=2049)
    assert value >= H.PCC_BAR, f"BFP8-cache prefill PCC {value} < {H.PCC_BAR}"

    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, _stats(), seed=23)
    golden = H.reference_decode(lut, token, 2049, cache)
    got = H.run_tt_decode(lut, token, torch.tensor([2049]))
    value = H.pcc(golden, got)
    H.record("bfp8_cache_decode_pcc", value, kind=_kind(lut), position=2049)
    assert value >= H.PCC_BAR, f"BFP8-cache decode PCC {value} < {H.PCC_BAR}"


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_traced_decode_batched(mesh_device, layer_idx):
    """Trace capture and replay at batch > 1, with per-user positions and page tables."""
    batch = 4
    seq_lens = [64 + 97 * u for u in range(batch)]
    lut = H.build_layer(mesh_device, layer_idx, max_batch=batch, max_seq_len=8192)

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
    runner.warmup(tokens, positions)  # eager warm-up advances every user by one token
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
            H.record("batched_traced_decode_pcc", value, kind=_kind(lut), batch=batch, user_id=user_id, replay=step)
            assert value >= H.PCC_BAR, f"batched traced decode PCC {value} for user {user_id}"
    runner.release()


#: Longest length at which the HF CPU reference is still runnable per layer kind.  The
#: ``full_attention`` reference uses eager attention, whose score matrix is
#: ``num_heads * seq_len**2 * 4`` bytes (6.4 GB at 8191); ``linear_attention`` is O(seq_len).
LONG_PCC_SEQ_LEN = {"linear_attention": 16385, "full_attention": 8191}


@pytest.mark.timeout(0)  # see LONG_TEST_TIMEOUT_NOTE
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_prefill_pcc_long(mesh_device, layer_idx):
    """HF-vs-TTNN prefill PCC at the longest length the CPU reference can still reach.

    Both lengths are non-divisible by the prefill chunk (2048), the delta chunk / page (64),
    the SDPA chunk (256) and the tile (32), so they exercise multi-chunk prefill together
    with the internal pad/mask/slice path.
    """
    config = ref.load_text_config()
    seq_len = LONG_PCC_SEQ_LEN[config.layer_types[layer_idx]]
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=seq_len + 2048)
    value, _cache, _hidden = _prefill_and_check(lut, seq_len)
    H.record("long_prefill_pcc", value, kind=_kind(lut), seq_len=seq_len)
    assert value >= H.PCC_BAR, f"long prefill PCC {value} < {H.PCC_BAR} (seq_len={seq_len})"


#: Tokens of the full-context prefill whose output is compared against the HF reference,
#: per layer kind.  ``linear_attention``'s reference is O(seq_len), so it can afford four
#: whole prefill chunks including the ragged final one.  ``full_attention``'s reference uses
#: eager attention against the full 262143-key cache, costing
#: ``num_heads * tail * context * 4`` bytes for the scores alone: 6.4 GB at 256, 206 GB at
#: 8192.  256 still spans the ragged tail of the last prefill chunk, which is what matters.
LONG_TAIL = {"linear_attention": 8192, "full_attention": 256}
#: Segment length of the segmented ``linear_attention`` reference.
LONG_SEGMENT = 16384
#: Prompt length of the full-context test.  Prompt + one decoded token occupy exactly the
#: advertised 262144 positions, so decode runs at the last addressable position.
LONG_PROMPT = 262143
#: Accepted range for the best-fit scale of the device output onto the HF reference at the full
#: context.  Both SDPA defects the layer works around are one-sided scale errors that PCC cannot
#: see, so the full-context tests bound the magnitude as well as the correlation.  The shipped
#: configuration measures 0.9949-0.9985 here.  For scale, the stock decode kernel's op-level
#: probe ``alpha`` is 1.29 at position 262143
#: (``doc/functional_decoder/logs/controls/sdpa_decode_stock_baseline.log``); its layer-level
#: scale is not a recorded number because the same control fails the decode *PCC* assertion
#: first.
SCALE_TOLERANCE = (0.98, 1.02)


@pytest.mark.timeout(0)  # see LONG_TEST_TIMEOUT_NOTE
@pytest.mark.long_context
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_full_advertised_context(mesh_device, layer_idx, request):
    """Prefill 262143 tokens and decode at position 262143, checked against the HF reference.

    A whole-prompt HF forward is impossible at this length, so the reference is built the only
    way that stays O(seq_len):

    * ``linear_attention`` — the HF layer runs in :data:`LONG_SEGMENT`-token segments with a
      carried ``DynamicCache``. ``Qwen3_5GatedDeltaNet`` continues exactly from a populated
      cache (it prepends ``conv_state`` and passes ``recurrent_state`` as ``initial_state``),
      so this is the same computation as one long call. The last segment's output, the final
      conv/recurrent state and a following decode step are all compared.
    * ``full_attention`` — the K/V cache is built from ``k_proj``/``v_proj`` + ``k_norm`` +
      RoPE only (no attention), then the real HF layer runs on the last :data:`LONG_TAIL`
      queries against that cache, followed by a decode step. The construction is validated
      against a genuine short ``reference_prefill`` first, so it cannot silently diverge from
      what ``Qwen3_5Attention.forward`` writes.

    This is what the earlier prefix-consistency version could not check: the first outputs of
    a causal prefill do not depend on anything the later 250k tokens do, so comparing them
    proved only that the run completed.
    """
    if not request.config.getoption("--long-context"):
        pytest.skip("needs --long-context")
    context = ref.load_text_config().max_position_embeddings
    assert LONG_PROMPT + 1 == context
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=context)
    tail = LONG_TAIL[_kind(lut)]
    stats = _stats()
    hidden = ref.synthetic_hidden_states(lut.config, 1, LONG_PROMPT, stats)

    got = H.run_tt_prefill(lut, hidden)
    assert got.shape == (1, LONG_PROMPT, lut.config.hidden_size)
    assert torch.isfinite(got).all(), "long prefill produced non-finite values"

    if lut.is_full_attention:
        # Control: the O(L) cache construction must reproduce a real short reference prefill.
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
        H.record("full_context_paged_k_cache_pcc", k_pcc, kind=_kind(lut), seq_len=LONG_PROMPT)
        H.record("full_context_paged_v_cache_pcc", v_pcc, kind=_kind(lut), seq_len=LONG_PROMPT)
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
        H.record("full_context_conv_state_pcc", conv_pcc, kind=_kind(lut), seq_len=LONG_PROMPT)
        H.record("full_context_recurrent_state_pcc", rec_pcc, kind=_kind(lut), seq_len=LONG_PROMPT)
        assert conv_pcc >= H.PCC_BAR, f"conv state after {LONG_PROMPT} tokens: PCC {conv_pcc}"
        assert rec_pcc >= H.PCC_BAR, f"recurrent state after {LONG_PROMPT} tokens: PCC {rec_pcc}"

    tail_pcc = H.pcc(golden, got[:, -tail:, :])
    H.record("full_context_prefill_tail_pcc", tail_pcc, kind=_kind(lut), seq_len=LONG_PROMPT, tail=tail)
    assert tail_pcc >= H.PCC_BAR, f"prefill tail at {LONG_PROMPT} tokens: PCC {tail_pcc}"
    # PCC is scale-invariant and both SDPA defects this layer works around are pure scale
    # errors, so assert the magnitude too: this bounds a quantity PCC cannot see. It is an
    # independent check, not a rescue - the reverted-build control fails on decode PCC alone
    # (0.977888, below the 0.995 bar) before any scale assertion is reached.
    tail_scale = H.scale_ratio(golden, got[:, -tail:, :])
    H.record("full_context_prefill_tail_scale", tail_scale, kind=_kind(lut), seq_len=LONG_PROMPT)
    assert (
        SCALE_TOLERANCE[0] <= tail_scale <= SCALE_TOLERANCE[1]
    ), f"prefill tail at {LONG_PROMPT} tokens is scaled by {tail_scale}, outside {SCALE_TOLERANCE}"

    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=99)
    golden_decode = H.reference_decode(lut, token, LONG_PROMPT, cache)
    decoded = H.run_tt_decode(lut, token, torch.tensor([LONG_PROMPT]))
    decode_pcc = H.pcc(golden_decode, decoded)
    H.record("full_context_decode_pcc", decode_pcc, kind=_kind(lut), position=LONG_PROMPT)
    assert decode_pcc >= H.PCC_BAR, f"decode at position {LONG_PROMPT}: PCC {decode_pcc}"
    decode_scale = H.scale_ratio(golden_decode, decoded)
    H.record("full_context_decode_scale", decode_scale, kind=_kind(lut), position=LONG_PROMPT)
    assert (
        SCALE_TOLERANCE[0] <= decode_scale <= SCALE_TOLERANCE[1]
    ), f"decode at position {LONG_PROMPT} is scaled by {decode_scale}, outside {SCALE_TOLERANCE}"

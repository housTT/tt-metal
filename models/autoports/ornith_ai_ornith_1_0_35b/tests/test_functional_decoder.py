# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Correctness, capability and performance tests for the Ornith-1.0-35B functional decoder.

Both Ornith decoder-layer kinds are covered with the real HF config shapes:

* ``layer 0``  — ``linear_attention`` (Gated DeltaNet + MoE)
* ``layer 3``  — ``full_attention``  (gated GQA + paged KV cache + MoE)

Weight source is selected by ``ORNITH_WEIGHTS``:

``real`` (default when the checkpoint snapshot is present)
    weights read straight out of the ``ornith-ai/Ornith-1.0-35B`` safetensors.
``synthetic``
    deterministic weights regenerated from the recorded real-weight statistics in
    ``doc/functional_decoder/weight_stats_layer{0,3}.json`` — real shapes and moments, no
    77 GB download. Used by ``test_*_synthetic_weights`` regardless of the env var.

Run everything on a single Blackhole device::

    pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_functional_decoder.py -v

Long-context and performance cases are marked and opt-in:

    pytest ... -m "long" -v          # full advertised 262144-token context
    pytest ... -k "perf" -v          # warmed prefill / traced warmed decode timing
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt.functional_decoder import (
    DEFAULT_PREFILL_CHUNK,
    FunctionalDecoder,
    num_blocks_for_context,
)

DOC_DIR = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder"

LINEAR_LAYER = 0
FULL_LAYER = 3
LAYERS = (LINEAR_LAYER, FULL_LAYER)
LAYER_IDS = {LINEAR_LAYER: "linear_attention", FULL_LAYER: "full_attention"}

#: Functional-decoder acceptance bar (skill default). Both layer kinds clear it on real weights.
PCC_BAR = 0.995

#: Context used by the non-long tests. Sized so the paged KV cache and RoPE tables stay small
#: while still exercising multi-chunk prefill; the advertised 262144 context is covered by the
#: ``long`` tests.
TEST_CONTEXT = 8192

#: HF advertised context (``text_config.max_position_embeddings``).
ADVERTISED_CONTEXT = 262144

DEVICE_PARAMS = [{"l1_small_size": 24576, "trace_region_size": 0}]

pytestmark = [
    pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True),
    pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True),
]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    """Pearson correlation, accumulated in float64.

    float32 accumulation over multi-million-element tensors drifts enough to report values
    slightly above 1.0, which makes the recorded evidence untrustworthy.
    """
    a = golden.double().flatten()
    b = actual.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def _snapshot_available() -> bool:
    try:
        return (R.resolve_model_path() / "model.safetensors.index.json").is_file()
    except Exception:  # noqa: BLE001 - offline / not downloaded
        return False


def default_weight_source() -> str:
    src = os.environ.get("ORNITH_WEIGHTS")
    if src:
        return src
    return "real" if _snapshot_available() else "synthetic"


_CACHE: dict = {}


def hf_config():
    if "cfg" not in _CACHE:
        _CACHE["cfg"] = R.load_text_config()
    return _CACHE["cfg"]


def stats_path(layer_idx: int) -> Path:
    return DOC_DIR / f"weight_stats_layer{layer_idx}.json"


def layer_state_dict(layer_idx: int, source: str) -> dict:
    key = ("sd", layer_idx, source)
    if key in _CACHE:
        return _CACHE[key]
    if source == "real":
        if not _snapshot_available():
            pytest.skip("Ornith-1.0-35B checkpoint snapshot not available")
        sd = R.load_layer_state_dict(layer_idx)
    elif source == "synthetic":
        path = stats_path(layer_idx)
        if not path.is_file():
            pytest.skip(f"missing recorded weight statistics: {path}")
        with open(path) as f:
            sd = R.synthetic_state_dict(json.load(f)["tensors"], seed=1234 + layer_idx)
    else:
        raise ValueError(f"unknown weight source {source!r}")
    _CACHE[key] = sd
    return sd


def reference_layer(layer_idx: int, source: str):
    key = ("ref", layer_idx, source)
    if key not in _CACHE:
        _CACHE[key] = R.build_reference_layer(hf_config(), layer_idx, layer_state_dict(layer_idx, source))
    return _CACHE[key]


def make_activations(batch: int, seq_len: int, *, seed: int = 0) -> torch.Tensor:
    """Activations approximating a post-embedding / post-residual hidden state.

    Ornith's residual stream after the embedding has unit-ish scale; ``std = 0.5`` keeps the
    router logits and the DeltaNet gates in their trained range rather than saturating them.
    """
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, seq_len, hf_config().hidden_size, generator=gen) * 0.5).to(torch.bfloat16)


def build_decoder(
    mesh_device,
    layer_idx: int,
    source: str,
    *,
    batch: int = 1,
    max_context: int = TEST_CONTEXT,
    prefill_chunk: int = DEFAULT_PREFILL_CHUNK,
    num_blocks: int | None = None,
):
    """Construct the layer, allocate its paged cache / recurrent state, and build a page table.

    Each user gets a **disjoint** span of physical blocks (``blocks_per_user`` of them), so a
    batched run cannot silently pass by having every user alias the same cache blocks.
    """
    sd = layer_state_dict(layer_idx, source)
    decoder = FunctionalDecoder.from_state_dict(
        sd,
        hf_config=hf_config(),
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_context=max_context,
        prefill_chunk=prefill_chunk,
    )
    blocks_per_user = num_blocks if num_blocks is not None else num_blocks_for_context(max_context)
    total_blocks = blocks_per_user * batch
    decoder.allocate_kv_cache(total_blocks)
    decoder.allocate_state(batch)
    page_table = None
    if decoder.is_full_attention:
        table = torch.arange(total_blocks, dtype=torch.int32).reshape(batch, blocks_per_user)
        page_table = to_device(mesh_device, table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return decoder, page_table, blocks_per_user


def to_device(mesh_device, tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def decode_inputs(mesh_device, positions: torch.Tensor):
    """``(current_pos, rot_idxs)`` device tensors for one decode step."""
    current_pos = to_device(mesh_device, positions.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    rot_idxs = to_device(
        mesh_device, positions.to(torch.int32).reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    return current_pos, rot_idxs


def run_reference(layer_idx, source, x, *, decode_x=None, decode_steps=0, start_pos=0):
    """HF golden prefill (and optional decode steps) for the same inputs."""
    ref = reference_layer(layer_idx, source)
    cfg = hf_config()
    with torch.no_grad():
        prefill_out, cache = R.reference_prefill(ref, cfg, x.float(), start_pos=start_pos)
        decode_outs = []
        pos = start_pos + x.shape[1]
        for step in range(decode_steps):
            positions = torch.full((x.shape[0],), pos + step, dtype=torch.long)
            decode_outs.append(R.reference_decode(ref, cfg, decode_x[step].float(), positions, cache))
    return prefill_out, decode_outs


# --------------------------------------------------------------------------------------
# RoPE contract
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len", [1, 64, 4096])
def test_rope_matches_hf(mesh_device, seq_len):
    """Ornith's interleaved M-RoPE reduces to 1-D partial RoPE for text-only positions.

    The TTNN layer builds a plain ``[context, rope_dim]`` table. This asserts that choice against
    the real ``Qwen3_5MoeTextRotaryEmbedding`` rather than assuming it.
    """
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import OrnithDecoderConfig
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.rope import OrnithRope

    cfg_hf = hf_config()
    cfg = OrnithDecoderConfig.from_hf_config(cfg_hf)
    start = 12345 if seq_len < 4096 else 0
    positions = torch.arange(start, start + seq_len).unsqueeze(0)
    cos_ref, sin_ref = R.reference_position_embeddings(cfg_hf, positions)
    assert cos_ref.shape[-1] == cfg.rope_dim

    rope = OrnithRope(mesh_device, cfg, max_context=start + seq_len)
    cos, sin = rope.prefill_forward(start, seq_len)
    cos_t = ttnn.to_torch(cos).reshape(1, seq_len, cfg.rope_dim)
    sin_t = ttnn.to_torch(sin).reshape(1, seq_len, cfg.rope_dim)
    cos_value, sin_value = pcc(cos_ref, cos_t), pcc(sin_ref, sin_t)

    # Decode gathers the same rows through ttnn.embedding from a device index tensor.
    probe = torch.tensor([[start, start + seq_len - 1]])
    cos_d, sin_d = rope.decode_forward(
        to_device(mesh_device, probe.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    )
    cos_dref, sin_dref = R.reference_position_embeddings(cfg_hf, probe)
    cos_dvalue = pcc(cos_dref, ttnn.to_torch(cos_d).reshape(1, 2, cfg.rope_dim))
    sin_dvalue = pcc(sin_dref, ttnn.to_torch(sin_d).reshape(1, 2, cfg.rope_dim))
    logger.info(
        f"rope vs HF seq_len={seq_len} start={start} prefill cos PCC={cos_value:.6f} "
        f"sin PCC={sin_value:.6f}; decode-gather cos PCC={cos_dvalue:.6f} sin PCC={sin_dvalue:.6f}"
    )
    for name, value in (
        ("prefill cos", cos_value),
        ("prefill sin", sin_value),
        ("decode cos", cos_dvalue),
        ("decode sin", sin_dvalue),
    ):
        assert value > 0.9999, f"{name} PCC {value} <= 0.9999"


# --------------------------------------------------------------------------------------
# prefill: sequence-length coverage
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize(
    "seq_len",
    [
        1,  # single token
        7,  # sub-tile, sub-page
        32,  # exactly one tile
        64,  # exactly one page block
        128,  # exactly the prefill physical alignment
        129,  # one past the alignment boundary
        250,  # tile-padded height (256) already equals the physical alignment
        2048,  # exactly one internal prefill chunk
        2049,  # one past a chunk boundary
        3000,  # multi-chunk, divisible by nothing relevant
    ],
)
def test_prefill_pcc(mesh_device, layer_idx, seq_len):
    """HF-vs-TTNN prefill PCC across aligned and deliberately non-aligned logical lengths."""
    source = default_weight_source()
    x = make_activations(1, seq_len, seed=seq_len)
    ref_out, _ = run_reference(layer_idx, source, x)

    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    x_tt = to_device(mesh_device, x)
    out = decoder.prefill_forward(x_tt, page_table=page_table)
    assert list(out.shape) == [1, seq_len, hf_config().hidden_size]
    got = ttnn.to_torch(out)
    value = pcc(ref_out, got)
    logger.info(f"prefill layer={layer_idx} ({LAYER_IDS[layer_idx]}) seq_len={seq_len} PCC={value:.6f}")
    assert torch.isfinite(got.float()).all()
    assert value > PCC_BAR, f"prefill PCC {value} <= {PCC_BAR} (layer {layer_idx}, seq_len {seq_len})"


# --------------------------------------------------------------------------------------
# decode: paged cache, current position, multi-step
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("prefill_len", [130, 2048])
def test_decode_pcc(mesh_device, layer_idx, prefill_len):
    """Prefill then four decode steps, comparing every step against HF.

    Exercises paged-cache writes at a growing ``current_pos`` (crossing a 64-token page boundary
    for ``prefill_len=130``) and the DeltaNet recurrent/conv state handoff from prefill to decode.
    """
    source = default_weight_source()
    steps = 4
    x = make_activations(1, prefill_len, seed=prefill_len)
    decode_x = [make_activations(1, 1, seed=1000 + i) for i in range(steps)]
    ref_prefill, ref_decode = run_reference(layer_idx, source, x, decode_x=decode_x, decode_steps=steps)

    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    prefill_value = pcc(ref_prefill, ttnn.to_torch(out))
    assert prefill_value > PCC_BAR, f"prefill PCC {prefill_value}"
    ttnn.deallocate(out)

    for step in range(steps):
        position = torch.tensor([prefill_len + step])
        current_pos, rot_idxs = decode_inputs(mesh_device, position)
        out = decoder.decode_forward(
            to_device(mesh_device, decode_x[step]),
            current_pos=current_pos,
            rot_idxs=rot_idxs,
            page_table=page_table,
        )
        got = ttnn.to_torch(out)
        value = pcc(ref_decode[step], got)
        logger.info(
            f"decode layer={layer_idx} ({LAYER_IDS[layer_idx]}) prefill_len={prefill_len} "
            f"step={step} pos={prefill_len + step} PCC={value:.6f}"
        )
        assert torch.isfinite(got.float()).all()
        assert value > PCC_BAR, f"decode step {step} PCC {value} <= {PCC_BAR}"
        ttnn.deallocate(out)


# --------------------------------------------------------------------------------------
# batch > 1
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("batch", [4, 32])
def test_batched_prefill_decode_pcc(mesh_device, layer_idx, batch):
    """Batched prefill + batched decode with **per-user** current positions.

    Users are prefilled with the same length but decode from different positions, so a layer that
    hard-coded batch 1 into the page table, cache indexing, current position, or DeltaNet state
    row would fail here.
    """
    source = default_weight_source()
    prefill_len = 96
    x = make_activations(batch, prefill_len, seed=7 + batch)
    decode_x = make_activations(batch, 1, seed=99 + batch)

    ref = reference_layer(layer_idx, source)
    cfg = hf_config()
    with torch.no_grad():
        ref_prefill, cache = R.reference_prefill(ref, cfg, x.float(), start_pos=0)

    # Small per-user context keeps batch-32 paged caches modest; the long tests cover capacity.
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, batch=batch, max_context=1024)
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    value = pcc(ref_prefill, ttnn.to_torch(out))
    logger.info(f"batched prefill layer={layer_idx} batch={batch} PCC={value:.6f}")
    assert value > PCC_BAR, f"batched prefill PCC {value}"
    ttnn.deallocate(out)

    positions = torch.full((batch,), prefill_len, dtype=torch.long)
    with torch.no_grad():
        ref_dec = R.reference_decode(ref, cfg, decode_x.float(), positions, cache)
    current_pos, rot_idxs = decode_inputs(mesh_device, positions)
    out = decoder.decode_forward(
        to_device(mesh_device, decode_x), current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table
    )
    got = ttnn.to_torch(out)
    value = pcc(ref_dec, got)
    logger.info(f"batched decode layer={layer_idx} batch={batch} PCC={value:.6f}")
    assert torch.isfinite(got.float()).all()
    assert value > PCC_BAR, f"batched decode PCC {value}"


@pytest.mark.parametrize("layer_idx", [FULL_LAYER], ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("batch", [4, 13])
def test_batched_decode_ragged_positions(mesh_device, layer_idx, batch):
    """Batched decode where every user sits at a **different** absolute position.

    This is the serving case, and it is the one that distinguishes a genuinely per-user
    ``current_pos`` / ``rot_idxs`` from a path that only ever reads element 0: with equal positions
    such a bug is invisible. Only ``full_attention`` is covered because it is the only kind that
    consumes a position at all — ``linear_attention`` has no KV cache and no RoPE, so its decode
    step takes no position input.

    Construction: prefill all users to ``max(lengths)``, then decode user ``u`` at
    ``current_pos = lengths[u]``. Paged SDPA decode reads slots ``[0, current_pos]``, and the
    token this step writes lands on slot ``current_pos``, so user ``u`` attends to exactly the
    first ``lengths[u]`` prefill tokens plus its new token — identical to a standalone run whose
    prompt was ``x[u, :lengths[u]]``. That gives an exact per-user HF golden without needing a
    ragged-batch reference (HF's ``DynamicCache`` cannot express one).

    ``batch=13`` also exercises a decode batch with no factor pair that fits an 8-wide shard grid.
    """
    source = default_weight_source()
    generator = torch.Generator().manual_seed(500 + batch)
    lengths = sorted(set((torch.randperm(200, generator=generator)[:batch] + 65).tolist()))
    while len(lengths) < batch:  # keep them distinct
        lengths.append(max(lengths) + 1)
    lengths = lengths[:batch]
    assert len(set(lengths)) == batch, "positions must be distinct for this test to mean anything"
    longest = max(lengths)

    x = make_activations(batch, longest, seed=600 + batch)
    decode_x = make_activations(batch, 1, seed=700 + batch)

    ref = reference_layer(layer_idx, source)
    cfg = hf_config()
    goldens = []
    with torch.no_grad():
        for user, length in enumerate(lengths):
            _, cache = R.reference_prefill(ref, cfg, x[user : user + 1, :length].float(), start_pos=0)
            goldens.append(
                R.reference_decode(ref, cfg, decode_x[user : user + 1].float(), torch.tensor([length]), cache)
            )

    # Shuffled, disjoint per-user block spans: a batched run must not depend on identity paging.
    blocks_per_user = num_blocks_for_context(1024)
    total_blocks = blocks_per_user * batch
    decoder = FunctionalDecoder.from_state_dict(
        layer_state_dict(layer_idx, source),
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_context=1024,
    )
    decoder.allocate_kv_cache(total_blocks)
    decoder.allocate_state(batch)
    table = torch.randperm(total_blocks, generator=generator).to(torch.int32).reshape(batch, blocks_per_user)
    page_table = to_device(mesh_device, table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    ttnn.deallocate(decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table))
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor(lengths))
    out = ttnn.to_torch(
        decoder.decode_forward(
            to_device(mesh_device, decode_x), current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table
        )
    )
    for user, length in enumerate(lengths):
        value = pcc(goldens[user], out[user : user + 1])
        logger.info(f"ragged-position decode batch={batch} user={user} pos={length} PCC={value:.6f}")
        assert value > PCC_BAR, f"user {user} at position {length}: PCC {value} <= {PCC_BAR}"


# --------------------------------------------------------------------------------------
# page table handling
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", [FULL_LAYER], ids=lambda i: LAYER_IDS[i])
def test_permuted_page_table(mesh_device, layer_idx):
    """A shuffled, offset page table must give bit-comparable results to the identity table.

    Catches address/indexing bugs that an identity page table hides: the logical positions are
    unchanged, only the physical block each one lands in.
    """
    source = default_weight_source()
    prefill_len = 200
    steps = 2
    x = make_activations(1, prefill_len, seed=5)
    decode_x = [make_activations(1, 1, seed=500 + i) for i in range(steps)]
    ref_prefill, ref_decode = run_reference(layer_idx, source, x, decode_x=decode_x, decode_steps=steps)

    blocks = num_blocks_for_context(TEST_CONTEXT)
    decoder, _, _ = build_decoder(mesh_device, layer_idx, source, num_blocks=blocks)

    generator = torch.Generator().manual_seed(3)
    permutation = torch.randperm(blocks, generator=generator).to(torch.int32).reshape(1, blocks)
    assert permutation[0, 0].item() != 0, "test wants a non-zero first physical slot"
    page_table = to_device(mesh_device, permutation, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    value = pcc(ref_prefill, ttnn.to_torch(out))
    logger.info(f"permuted-page-table prefill PCC={value:.6f} first_slot={permutation[0, 0].item()}")
    assert value > PCC_BAR, f"permuted page table prefill PCC {value}"
    ttnn.deallocate(out)

    for step in range(steps):
        position = torch.tensor([prefill_len + step])
        current_pos, rot_idxs = decode_inputs(mesh_device, position)
        out = decoder.decode_forward(
            to_device(mesh_device, decode_x[step]),
            current_pos=current_pos,
            rot_idxs=rot_idxs,
            page_table=page_table,
        )
        value = pcc(ref_decode[step], ttnn.to_torch(out))
        logger.info(f"permuted-page-table decode step={step} PCC={value:.6f}")
        assert value > PCC_BAR, f"permuted page table decode PCC {value}"
        ttnn.deallocate(out)


# --------------------------------------------------------------------------------------
# chunked prefill continuation (start_pos > 0)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_prefill_continuation(mesh_device, layer_idx):
    """Two prefill calls over one sequence must equal a single call over the concatenation.

    This is the contract every caller that streams a prompt in pieces relies on: the paged cache
    keeps growing and the DeltaNet state carries.
    """
    source = default_weight_source()
    chunk = 128
    total = 3 * chunk
    x = make_activations(1, total, seed=21)
    ref_prefill, _ = run_reference(layer_idx, source, x)

    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, prefill_chunk=chunk)
    x_tt = to_device(mesh_device, x)
    first = decoder.prefill_forward(
        ttnn.slice(x_tt, [0, 0, 0], [1, chunk, hf_config().hidden_size]),
        start_pos=0,
        page_table=page_table,
        chunk_size=chunk,
    )
    second = decoder.prefill_forward(
        ttnn.slice(x_tt, [0, chunk, 0], [1, total, hf_config().hidden_size]),
        start_pos=chunk,
        page_table=page_table,
        chunk_size=chunk,
    )
    got = torch.cat([ttnn.to_torch(first), ttnn.to_torch(second)], dim=1)
    value = pcc(ref_prefill, got)
    logger.info(f"prefill continuation layer={layer_idx} PCC={value:.6f}")
    assert value > PCC_BAR, f"continuation PCC {value}"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_unaligned_max_context(mesh_device, layer_idx):
    """A ``max_context`` that is not a multiple of the internal prefill alignment must work.

    The documented contract accepts any ``seq_len <= max_context``, and the layer pads a block's
    *physical* length up to 128. With ``max_context = 5000`` the final block of a 5000-token
    prefill spans physical positions [4096, 6016) — past the last logical position — so anything
    sized exactly to ``max_context`` (the RoPE table in particular) has to tolerate that.
    """
    source = default_weight_source()
    max_context = 5000  # not a multiple of 128, and not a multiple of the 2048 chunk
    seq_len = max_context
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=max_context)
    out = decoder.prefill_forward(to_device(mesh_device, make_activations(1, seq_len, seed=77)), page_table=page_table)
    assert list(out.shape) == [1, seq_len, hf_config().hidden_size]
    tail = ttnn.to_torch(ttnn.slice(out, [0, seq_len - 64, 0], [1, seq_len, hf_config().hidden_size])).float()
    ttnn.deallocate(out)
    assert torch.isfinite(tail).all()
    assert tail.std() > 0.01
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([seq_len - 1]))
    got = ttnn.to_torch(
        decoder.decode_forward(
            to_device(mesh_device, make_activations(1, 1, seed=78)),
            current_pos=current_pos,
            rot_idxs=rot_idxs,
            page_table=page_table,
        )
    ).float()
    assert torch.isfinite(got).all()
    logger.info(f"unaligned max_context={max_context} layer={layer_idx} prefill+decode ok, tail std={tail.std():.4f}")


# --------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_determinism_repeated_inputs(mesh_device, layer_idx):
    """Identical inputs must give bit-identical outputs, in prefill and in decode.

    Decode is checked by replaying the *same* step from a re-zeroed state, which also proves the
    state reset is complete (a leaked residual would show up as a mismatch).
    """
    source = default_weight_source()
    prefill_len = 256
    x = make_activations(1, prefill_len, seed=11)
    decode_x = make_activations(1, 1, seed=12)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    x_tt = to_device(mesh_device, x)
    d_tt = to_device(mesh_device, decode_x)
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([prefill_len]))

    results = []
    for _ in range(3):
        decoder.reset_state()
        prefill = ttnn.to_torch(decoder.prefill_forward(x_tt, page_table=page_table))
        decode = ttnn.to_torch(
            decoder.decode_forward(d_tt, current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table)
        )
        results.append((prefill, decode))

    for idx in (1, 2):
        assert torch.equal(results[0][0], results[idx][0]), f"prefill run {idx} differs bitwise"
        assert torch.equal(results[0][1], results[idx][1]), f"decode run {idx} differs bitwise"
    logger.info(f"determinism layer={layer_idx}: 3/3 runs bit-identical for prefill and decode")


# --------------------------------------------------------------------------------------
# use-after-free / aliasing regression
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("seq_len", [1, 250, 300])
def test_forward_with_poisoned_free_pool(mesh_device, layer_idx, seq_len):
    """Correctness must not depend on what freed DRAM happens to contain.

    A large tensor of near-``bfloat16``-max values is allocated and freed immediately before the
    forward pass, so any buffer the layer frees too early and then reads back returns huge
    garbage instead of the benign leftovers that make such a bug look intermittent. This is the
    regression guard for the `ttnn.slice` / `ttnn.pad` aliasing hazard: both can return a tensor
    sharing their input's buffer, so freeing the pre-slice/pre-pad tensor frees the live one.

    The chosen lengths are the aliasing-prone ones: ``1`` (decode's token axis padded 1 → 32) and
    ``250`` (tile-padded height 256 already equals the 128-aligned physical length 256).
    """
    source = default_weight_source()
    x = make_activations(1, seq_len, seed=seq_len + 7)
    decode_x = make_activations(1, 1, seed=seq_len + 8)
    ref_prefill, ref_decode = run_reference(layer_idx, source, x, decode_x=[decode_x], decode_steps=1)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    x_tt = to_device(mesh_device, x)
    d_tt = to_device(mesh_device, decode_x)
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([seq_len]))

    def poison():
        junk = to_device(mesh_device, torch.full((1, 1, 4096, 4096), 3.0e38, dtype=torch.bfloat16))
        ttnn.deallocate(junk)

    poison()
    out = decoder.prefill_forward(x_tt, page_table=page_table)
    prefill_value = pcc(ref_prefill, ttnn.to_torch(out))
    ttnn.deallocate(out)
    poison()
    out = decoder.decode_forward(d_tt, current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table)
    got = ttnn.to_torch(out)
    decode_value = pcc(ref_decode[0], got)
    logger.info(
        f"poisoned-pool layer={layer_idx} seq_len={seq_len} prefill PCC={prefill_value:.6f} "
        f"decode PCC={decode_value:.6f}"
    )
    assert torch.isfinite(got.float()).all(), "decode produced non-finite values after a poisoned free"
    assert prefill_value > PCC_BAR, f"prefill PCC {prefill_value} after poisoned free pool"
    assert decode_value > PCC_BAR, f"decode PCC {decode_value} after poisoned free pool"


# --------------------------------------------------------------------------------------
# runtime fallback audit
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_no_host_fallback_in_forward(mesh_device, layer_idx, monkeypatch, expect_error):
    """A single prefill and a single decode pass must not touch the host.

    Two independent guards run together for the duration of the measured passes:

    * every ``ttnn`` host↔device entry point is replaced with a raising stub;
    * a ``TorchFunctionMode`` raises on *any* dispatched ``torch`` operation.

    So a hidden host round trip anywhere in the layer or in the helpers it reuses fails the test
    instead of quietly costing latency.
    """
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    x_tt = to_device(mesh_device, make_activations(1, 256, seed=13))
    d_tt = to_device(mesh_device, make_activations(1, 1, seed=14))
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([256]))

    banned = ["from_torch", "to_torch", "as_tensor", "from_device", "to_device"]

    def _raise(name):
        def stub(*args, **kwargs):
            raise AssertionError(f"host fallback: ttnn.{name} called inside a measured forward pass")

        return stub

    for name in banned:
        monkeypatch.setattr(ttnn, name, _raise(name))

    class NoTorchOps(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            raise AssertionError(f"host fallback: torch op {getattr(func, '__name__', func)} in a measured pass")

    # Positive controls: prove both guards actually fire, so a clean pass below means "no host
    # fallback" rather than "the guards were inert".
    probe = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
    with expect_error(AssertionError, "host fallback"):
        ttnn.from_torch(probe)
    with NoTorchOps():
        with expect_error(AssertionError, "host fallback"):
            torch.add(probe, probe)

    with NoTorchOps():
        out = decoder.prefill_forward(x_tt, page_table=page_table)
        ttnn.deallocate(out)
        out = decoder.decode_forward(d_tt, current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table)
        ttnn.deallocate(out)
    logger.info(
        f"fallback audit layer={layer_idx}: both guards verified to fire; prefill+decode clean for "
        f"ttnn {banned} and all torch ops"
    )


# --------------------------------------------------------------------------------------
# traced decode
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_traced_decode_pcc(mesh_device, layer_idx):
    """Decode runs fully under ``ttnn`` traced execution, and the *replay* output matches HF.

    The trace bakes in buffer addresses, so this also proves the state buffers (DeltaNet
    recurrent + conv history, paged KV cache) are updated in place rather than reallocated.
    """
    source = default_weight_source()
    prefill_len = 128
    steps = 3
    x = make_activations(1, prefill_len, seed=31)
    decode_x = [make_activations(1, 1, seed=3100 + i) for i in range(steps)]
    _, ref_decode = run_reference(layer_idx, source, x, decode_x=decode_x, decode_steps=steps)

    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    ttnn.deallocate(decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table))

    # Stable trace inputs: only their contents change between replays.
    x_buf = to_device(mesh_device, decode_x[0])
    pos_buf, rot_buf = decode_inputs(mesh_device, torch.tensor([prefill_len]))

    def forward():
        return decoder.decode_forward(x_buf, current_pos=pos_buf, rot_idxs=rot_buf, page_table=page_table)

    # Compile + snapshot/restore: the capture pass itself advances the DeltaNet state and the KV
    # cache, so the state must be rewound before the first real replay.
    saved = _snapshot_state(decoder)
    ttnn.deallocate(forward())
    ttnn.synchronize_device(mesh_device)
    _restore_state(decoder, saved)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_out = forward()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    _restore_state(decoder, saved)

    for step in range(steps):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(decode_x[step], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), x_buf
        )
        position = torch.tensor([prefill_len + step], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(position, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), pos_buf
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(position.reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), rot_buf
        )
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        got = ttnn.to_torch(trace_out)
        value = pcc(ref_decode[step], got)
        logger.info(f"traced decode layer={layer_idx} step={step} replay PCC={value:.6f}")
        assert torch.isfinite(got.float()).all()
        assert value > PCC_BAR, f"traced decode replay step {step} PCC {value} <= {PCC_BAR}"

    ttnn.release_trace(mesh_device, trace_id)


def _snapshot_state(decoder):
    """Host copies of the mutable state, for rewinding around trace capture.

    A device-side ``ttnn.clone`` is deliberately avoided: clones draw from the same general pool
    the captured trace's baked intermediates use, so a replay can overwrite them.
    """
    snapshot = {}
    if decoder.is_full_attention:
        snapshot["k"] = ttnn.to_torch(decoder.k_cache)
        snapshot["v"] = ttnn.to_torch(decoder.v_cache)
    else:
        snapshot["rec"] = ttnn.to_torch(decoder.recurrent_state)
        snapshot["conv"] = [ttnn.to_torch(buf) for buf in decoder.conv_state]
    return snapshot


def _restore_state(decoder, snapshot):
    if decoder.is_full_attention:
        for key, cache in (("k", decoder.k_cache), ("v", decoder.v_cache)):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(snapshot[key], dtype=cache.dtype, layout=ttnn.TILE_LAYOUT), cache
            )
    else:
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(snapshot["rec"], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
            decoder.recurrent_state,
        )
        for saved, buf in zip(snapshot["conv"], decoder.conv_state):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(saved, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), buf)


# --------------------------------------------------------------------------------------
# explicit weight-source coverage
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_real_weights_pcc(mesh_device, layer_idx):
    """``from_state_dict`` loads a real checkpoint subtree and clears the PCC bar on it."""
    if not _snapshot_available():
        pytest.skip("Ornith-1.0-35B checkpoint snapshot not available")
    seq_len = 300
    x = make_activations(1, seq_len, seed=41)
    decode_x = [make_activations(1, 1, seed=42)]
    ref_prefill, ref_decode = run_reference(layer_idx, "real", x, decode_x=decode_x, decode_steps=1)

    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, "real")
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    prefill_value = pcc(ref_prefill, ttnn.to_torch(out))
    ttnn.deallocate(out)

    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([seq_len]))
    out = decoder.decode_forward(
        to_device(mesh_device, decode_x[0]), current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table
    )
    decode_value = pcc(ref_decode[0], ttnn.to_torch(out))
    logger.info(
        f"REAL WEIGHTS layer={layer_idx} ({LAYER_IDS[layer_idx]}) "
        f"prefill PCC={prefill_value:.6f} decode PCC={decode_value:.6f}"
    )
    assert prefill_value > PCC_BAR
    assert decode_value > PCC_BAR


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_synthetic_weights_pcc(mesh_device, layer_idx):
    """The CI weight path (synthetic from recorded real statistics) also clears the PCC bar."""
    seq_len = 300
    x = make_activations(1, seq_len, seed=43)
    decode_x = [make_activations(1, 1, seed=44)]
    ref_prefill, ref_decode = run_reference(layer_idx, "synthetic", x, decode_x=decode_x, decode_steps=1)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, "synthetic")
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    prefill_value = pcc(ref_prefill, ttnn.to_torch(out))
    ttnn.deallocate(out)
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([seq_len]))
    out = decoder.decode_forward(
        to_device(mesh_device, decode_x[0]), current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table
    )
    decode_value = pcc(ref_decode[0], ttnn.to_torch(out))
    logger.info(
        f"SYNTHETIC WEIGHTS layer={layer_idx} ({LAYER_IDS[layer_idx]}) "
        f"prefill PCC={prefill_value:.6f} decode PCC={decode_value:.6f}"
    )
    assert prefill_value > PCC_BAR
    assert decode_value > PCC_BAR


# --------------------------------------------------------------------------------------
# advertised-context capability
# --------------------------------------------------------------------------------------
@pytest.mark.long
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("seq_len", [ADVERTISED_CONTEXT - 1, ADVERTISED_CONTEXT])
def test_full_context_prefill_and_decode(mesh_device, layer_idx, seq_len):
    """Run prefill at the full advertised context, then decode at the last legal position.

    HF cannot serve as a golden here (the 262144-token reference is a multi-hour CPU run for the
    full-attention layer), so this is a capability + sanity test: the pass must complete on
    device, stay finite and non-degenerate, and leave a usable cache/state for a decode step.
    ``seq_len = ADVERTISED_CONTEXT - 1`` additionally proves a non-aligned logical length at the
    very top of the range.
    """
    source = default_weight_source()
    batch = 1
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, batch=batch, max_context=ADVERTISED_CONTEXT)
    x = make_activations(batch, seq_len, seed=97)
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    assert list(out.shape) == [batch, seq_len, hf_config().hidden_size]
    tail = ttnn.to_torch(ttnn.slice(out, [0, seq_len - 64, 0], [batch, seq_len, hf_config().hidden_size])).float()
    ttnn.deallocate(out)
    assert torch.isfinite(tail).all(), "long prefill produced non-finite outputs"
    assert tail.std() > 0.01, f"long prefill output is near-constant (std={tail.std():.4g})"

    if seq_len < ADVERTISED_CONTEXT:
        current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([seq_len]))
        dec = decoder.decode_forward(
            to_device(mesh_device, make_activations(batch, 1, seed=98)),
            current_pos=current_pos,
            rot_idxs=rot_idxs,
            page_table=page_table,
        )
        got = ttnn.to_torch(dec).float()
        assert torch.isfinite(got).all()
        assert got.std() > 0.01
    logger.info(
        f"full-context layer={layer_idx} ({LAYER_IDS[layer_idx]}) seq_len={seq_len} "
        f"prefill+decode completed, tail std={tail.std():.4f}"
    )


@pytest.mark.long
# Two full 262144-token prefills back to back; the repo-wide pytest.ini timeout is 300 s and the
# full_attention pair alone takes ~4 minutes of quadratic attention.
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_full_context_chunk_size_invariance(mesh_device, layer_idx):
    """At the full advertised context, the result must not depend on the internal chunking.

    HF cannot serve as a golden at 262144, so this is the control that gives the long-context
    claim teeth beyond "finite and non-degenerate": the same prefill is run with two different
    internal chunk sizes, which changes every block boundary, every paged-cache fill span, every
    chunked-SDPA offset and every DeltaNet state hand-off — but must not change the answer. A
    long-context indexing bug (wrong RoPE row, wrong page span, dropped state carry) would almost
    certainly move with the chunking; agreement is strong evidence it is absent.

    Only the last 128 positions are compared: they depend on the entire history, and holding the
    whole ``[1, 262144, 2048]`` output on the host twice is 2 GB.
    """
    source = default_weight_source()
    seq_len = ADVERTISED_CONTEXT
    tail_len = 128
    hidden = hf_config().hidden_size
    x = make_activations(1, seq_len, seed=131)

    tails = {}
    for chunk_size in (2048, 1024):
        decoder, page_table, _ = build_decoder(
            mesh_device, layer_idx, source, max_context=ADVERTISED_CONTEXT, prefill_chunk=2048
        )
        out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table, chunk_size=chunk_size)
        tails[chunk_size] = ttnn.to_torch(ttnn.slice(out, [0, seq_len - tail_len, 0], [1, seq_len, hidden])).float()
        ttnn.deallocate(out)
        assert torch.isfinite(tails[chunk_size]).all(), f"chunk_size={chunk_size} produced non-finite output"

    value = pcc(tails[2048], tails[1024])
    logger.info(
        f"chunk-size invariance layer={layer_idx} ({LAYER_IDS[layer_idx]}) seq_len={seq_len} "
        f"tail({tail_len}) PCC(chunk 2048 vs 1024)={value:.6f}"
    )
    # Not bit-exact: different block boundaries reassociate the bf16 accumulations.
    assert value > 0.999, f"chunking changed the result at the full context: PCC {value}"


@pytest.mark.long
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_long_context_pcc(mesh_device, layer_idx):
    """Largest context where the HF golden is tractable, at a non-aligned length."""
    source = default_weight_source()
    seq_len = 8000
    x = make_activations(1, seq_len, seed=61)
    ref_prefill, ref_decode = run_reference(
        layer_idx, source, x, decode_x=[make_activations(1, 1, seed=62)], decode_steps=1
    )
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=16384)
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    prefill_value = pcc(ref_prefill, ttnn.to_torch(out))
    ttnn.deallocate(out)
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([seq_len]))
    dec = decoder.decode_forward(
        to_device(mesh_device, make_activations(1, 1, seed=62)),
        current_pos=current_pos,
        rot_idxs=rot_idxs,
        page_table=page_table,
    )
    decode_value = pcc(ref_decode[0], ttnn.to_torch(dec))
    logger.info(
        f"long-context PCC layer={layer_idx} seq_len={seq_len} prefill={prefill_value:.6f} decode={decode_value:.6f}"
    )
    assert prefill_value > PCC_BAR
    assert decode_value > PCC_BAR


# --------------------------------------------------------------------------------------
# performance
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("seq_len", [2048])
def test_perf_prefill(mesh_device, layer_idx, seq_len):
    """Warmed prefill timing between ``PERF_PREFILL`` signposts."""
    from tracy import signpost

    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    x_tt = to_device(mesh_device, make_activations(1, seq_len, seed=71))

    for _ in range(2):  # compile + warm
        ttnn.deallocate(decoder.prefill_forward(x_tt, page_table=page_table))
    ttnn.synchronize_device(mesh_device)

    import time

    signpost("PERF_PREFILL")
    start = time.time()
    out = decoder.prefill_forward(x_tt, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.time() - start
    signpost("PERF_PREFILL_END")
    ttnn.deallocate(out)
    logger.info(
        f"PERF prefill layer={layer_idx} ({LAYER_IDS[layer_idx]}) seq_len={seq_len} "
        f"wall={elapsed * 1e3:.2f} ms  tok/s={seq_len / elapsed:.1f}"
    )


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_perf_decode_traced(mesh_device, layer_idx):
    """Warmed traced decode timing between ``PERF_DECODE`` signposts."""
    from tracy import signpost

    source = default_weight_source()
    iters = 32
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    ttnn.deallocate(
        decoder.prefill_forward(to_device(mesh_device, make_activations(1, 128, seed=81)), page_table=page_table)
    )

    x_buf = to_device(mesh_device, make_activations(1, 1, seed=82))
    pos_buf, rot_buf = decode_inputs(mesh_device, torch.tensor([128]))

    def forward():
        return decoder.decode_forward(x_buf, current_pos=pos_buf, rot_idxs=rot_buf, page_table=page_table)

    ttnn.deallocate(forward())
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_out = forward()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    for _ in range(4):  # warm the replay
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    import time

    signpost("PERF_DECODE")
    start = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.time() - start
    signpost("PERF_DECODE_END")
    logger.info(
        f"PERF decode(traced) layer={layer_idx} ({LAYER_IDS[layer_idx]}) iters={iters} "
        f"wall/iter={elapsed / iters * 1e3:.3f} ms  steps/s={iters / elapsed:.1f}"
    )
    assert torch.isfinite(ttnn.to_torch(trace_out).float()).all()
    ttnn.release_trace(mesh_device, trace_id)

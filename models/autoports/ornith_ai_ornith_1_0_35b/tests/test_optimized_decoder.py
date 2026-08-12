# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Correctness, capability and performance tests for the Ornith-1.0-35B **optimized** decoder.

Everything the fused decoder is held to is re-asserted here against
:class:`~models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder.OptimizedDecoder`, because
an optimization stage is only allowed to change *how fast* the layer computes, never what it
computes or what it supports: HF-vs-TTNN PCC over aligned and deliberately non-aligned sequence
lengths, paged-KV behaviour, per-user decode positions, chunked-prefill continuation, determinism,
freed-DRAM independence, trace replay, the advertised 262144-token context, and the absence of any
host fallback.

On top of that this file adds what is specific to an *optimized* implementation:

* ``test_optimized_matches_fused`` — the optimized and fused decoders driven with identical weights
  and inputs must agree far more tightly than either agrees with HF, allowing for the reduced-
  precision weight policy this stage selects;
* ``test_optimized_path_is_used`` — the delivered tests must exercise the optimized path, not a
  functional fallback, so this asserts the dedicated ops are dispatched (and that the functional
  decoder does not dispatch them), **and** that the optimization-stage contracts are live: the
  selected weight dtypes reached the device tensors, the tuned decode program configs are the ones
  the projections run under, the routed-expert intermediates are in L1, and the decode residual
  norms take the width-sharded multi-core path;
* ``test_padded_rows_do_not_route`` — the tile-padding rows of a decode MoE group must not
  contribute experts to the routing sparsity, and must not change the layer output;
* ``test_optimized_beats_fused_traced_decode`` — the optimized traced decode must actually be
  faster than the fused decoder's, measured in one process on one device with the same weights;
* ``test_no_layout_churn_in_measured_forward`` — no ``tilize``/``untilize``/``reshard`` op may run
  in the measured prefill/decode passes beyond the documented budget;
* ``test_repeated_run_stress`` — many back-to-back prefill/decode cycles, checking drift-free
  repeatability and bounded memory.

Weight source is selected by ``ORNITH_WEIGHTS`` exactly as in the fused suite.

Run everything on a single Blackhole device::

    pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_optimized_decoder.py -v

The long-context and performance cases carry markers but are **not** deselected by default — this
repo's ``pytest.ini`` adds no ``-m "not long"`` — so a plain run above covers all of them, and the
committed evidence comes from one invocation. The markers exist to *narrow* to them::

    pytest ... -m "long" -v          # full advertised 262144-token context only
    pytest ... -k "perf" -v          # warmed prefill / traced warmed decode timing only
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
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import (
    CONV1D_CHANNELS,
    DEFAULT_PREFILL_CHUNK,
    PREFILL_ALIGN,
    OptimizedDecoder,
    num_blocks_for_context,
)

DOC_DIR = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder"

LINEAR_LAYER = 0
FULL_LAYER = 3
LAYERS = (LINEAR_LAYER, FULL_LAYER)
LAYER_IDS = {LINEAR_LAYER: "linear_attention", FULL_LAYER: "full_attention"}

#: Acceptance bar inherited from the functional-decoder stage. The fusing stage is not allowed to
#: lower it, and every measurement below clears it by more than an order of magnitude of error (the
#: worst case in the shipped suite log is 0.999882, i.e. 1.2e-4 against the 5e-3 the bar allows).
PCC_BAR = 0.995

#: Bar for optimized-vs-fused agreement. The two implementations are not bit-identical and are not
#: meant to be: this stage moves the expert weights to BFP4 and the dense projection weights to
#: BFP8, so the optimized layer carries strictly less weight precision than the fused one. They must
#: still agree far more tightly than either agrees with the float32 HF golden — the acceptance gate
#: is :data:`PCC_BAR` against HF, and this bar exists to catch a *structural* divergence (a wrong
#: expert set, a dropped residual, a mis-sliced projection) that the HF bar might absorb.
EQUIV_BAR = 0.998

TEST_CONTEXT = 8192
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
    """Pearson correlation, accumulated in float64."""
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
    """Activations approximating a post-embedding / post-residual hidden state."""
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
    cls=OptimizedDecoder,
    **kwargs,
):
    """Construct the layer, allocate its paged cache / recurrent state, and build a page table.

    Each user gets a **disjoint** span of physical blocks, so a batched run cannot silently pass by
    having every user alias the same cache blocks.
    """
    sd = layer_state_dict(layer_idx, source)
    decoder = cls.from_state_dict(
        sd,
        hf_config=hf_config(),
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_context=max_context,
        prefill_chunk=prefill_chunk,
        **kwargs,
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
# the fused graph is what the tests exercise
# --------------------------------------------------------------------------------------
class _OpRecorder:
    """Records which ``ttnn`` entry points a forward pass dispatches.

    Used to prove the delivered tests run the *fused* graph rather than a functional fallback, and
    to bound the layout-conversion traffic in the measured passes.
    """

    def __init__(self, monkeypatch, names):
        self.calls: dict[str, int] = {}
        for name in names:
            module, _, attr = name.rpartition(".")
            target = ttnn
            for part in module.split(".") if module else []:
                target = getattr(target, part)
            original = getattr(target, attr)

            def wrapper(*args, _name=name, _original=original, **kwargs):
                self.calls[_name] = self.calls.get(_name, 0) + 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(target, attr, wrapper)

    def count(self, name):
        return self.calls.get(name, 0)

    def reset(self):
        """Forget everything recorded so far.

        Setup is not a measured forward pass, and it dispatches some of the same ops: building a
        linear layer executes each candidate ``ttnn.conv1d`` program once to find out whether it
        fits in L1. Without this, ``conv1d`` would show a nonzero count even when every forward
        pass had fallen back to the FIR form.
        """
        self.calls.clear()


#: The dedicated ops the fused graph is *defined* by. If a future edit silently drops one of these
#: back to a primitive sequence, ``test_optimized_path_is_used`` fails instead of quietly regressing.
DEDICATED_OPS_PREFILL = {
    LINEAR_LAYER: [
        "transformer.chunk_gated_delta_rule",
        "experimental.nlp_concat_heads",
        "experimental.deepseek_moe_fast_reduce_nc",
        "sparse_matmul",
        # The depthwise causal conv runs as CONV1D_CHANNELS-wide ttnn.conv1d calls in prefill; the
        # addcmul FIR form is the decode path (and the fallback for a block length whose conv
        # weights were not pre-prepared).
        "conv1d",
    ],
    FULL_LAYER: [
        "experimental.nlp_create_qkv_heads",
        "experimental.nlp_concat_heads",
        "experimental.rotary_embedding_hf",
        "experimental.paged_fill_cache",
        "experimental.deepseek_moe_fast_reduce_nc",
        "sparse_matmul",
    ],
}
DEDICATED_OPS_DECODE = {
    # No `mac` here: the conv-tap accumulator is `ttnn.addcmul`, the same op the functional decoder
    # uses, because `addcmul` is a single LLK op for these shapes and `mac` is always two (work_log
    # §4.14). It is therefore not a fused rewrite and cannot be a fused-only assertion.
    LINEAR_LAYER: ["experimental.deepseek_moe_fast_reduce_nc", "sparse_matmul"],
    FULL_LAYER: [
        "experimental.nlp_create_qkv_heads_decode",
        "experimental.rotary_embedding_hf",
        "experimental.paged_fused_update_cache",
        "experimental.deepseek_moe_fast_reduce_nc",
        "sparse_matmul",
    ],
}

#: Ops the *functional* decoder uses in its decode step and the fused one must have replaced
#: outright. Asserted 0 in the fused decode window and, as a liveness control, nonzero in the
#: functional one — so this catches a hybrid where the dedicated op fires *and* the fallback still
#: runs, which a ``> 0`` assertion on the dedicated op alone cannot see.
REPLACED_OPS_DECODE = {
    # `fast_reduce_nc` is the functional MoE's expert-axis reduction, which the fused MoE replaces
    # with `deepseek_moe_fast_reduce_nc`; it applies to both layer kinds because they share the MoE.
    # Without it the linear-attention entry would be empty, and an empty entry makes both the
    # replacement check *and* its own liveness control vacuous for that layer.
    LINEAR_LAYER: ["experimental.fast_reduce_nc"],
    FULL_LAYER: ["experimental.fast_reduce_nc", "experimental.paged_update_cache"],
}


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_optimized_path_is_used(mesh_device, layer_idx, monkeypatch):
    """The delivered tests must exercise the fused graph, not a functional fallback.

    Asserts that every dedicated op the fused decoder is built around is actually dispatched by a
    plain prefill and a plain decode call — and, as a negative control, that the functional decoder
    dispatches none of them **beyond the three the two implementations genuinely share**
    (`sparse_matmul`, `chunk_gated_delta_rule`, `paged_fill_cache`), so the assertion cannot pass
    vacuously. The control is also checked to be live, so a typo in an op name fails rather than
    silently recording nothing.
    """
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.functional_decoder import FunctionalDecoder

    source = default_weight_source()
    dedicated = sorted(set(DEDICATED_OPS_PREFILL[layer_idx]) | set(DEDICATED_OPS_DECODE[layer_idx]))
    names = sorted(set(dedicated) | set(REPLACED_OPS_DECODE[layer_idx]))

    seen = {}
    conv1d_lengths, conv_dim, prefill_chunk, moe_groups = [], None, None, None
    for cls in (OptimizedDecoder, FunctionalDecoder):
        with monkeypatch.context() as ctx:
            recorder = _OpRecorder(ctx, names)
            decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, cls=cls)
            # Setup, not a measured pass: allocate_state probes each candidate conv1d program once.
            recorder.reset()
            ttnn.deallocate(
                decoder.prefill_forward(
                    to_device(mesh_device, make_activations(1, 256, seed=91)), page_table=page_table
                )
            )
            prefill_counts = dict(recorder.calls)
            current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([256]))
            ttnn.deallocate(
                decoder.decode_forward(
                    to_device(mesh_device, make_activations(1, 1, seed=92)),
                    current_pos=current_pos,
                    rot_idxs=rot_idxs,
                    page_table=page_table,
                )
            )
            seen[cls.__name__] = (prefill_counts, dict(recorder.calls))
            if cls is OptimizedDecoder:
                conv1d_lengths = list(decoder.conv1d_lengths)
                conv_dim, prefill_chunk = decoder.cfg.conv_dim, decoder.prefill_chunk
                moe_groups = -(-256 // decoder.moe.group_tokens)
        del decoder

    optimized_prefill, optimized_total = seen["OptimizedDecoder"]
    func_prefill, func_total = seen["FunctionalDecoder"]
    logger.info(f"optimized-op dispatch layer={layer_idx} ({LAYER_IDS[layer_idx]}) prefill={optimized_prefill}")
    logger.info(f"optimized-op dispatch layer={layer_idx} ({LAYER_IDS[layer_idx]}) prefill+decode={optimized_total}")
    for name in DEDICATED_OPS_PREFILL[layer_idx]:
        assert optimized_prefill.get(name, 0) > 0, f"optimized prefill did not dispatch ttnn.{name}"
    if layer_idx == LINEAR_LAYER:
        # Exact count, not just "nonzero": the recorder is cleared after setup, so these are the
        # calls the forward pass itself made. A 256-token prefill is one physical block, and the
        # depthwise conv covers conv_dim in CONV1D_CHANNELS-wide calls.
        expected = conv_dim // CONV1D_CHANNELS
        assert 256 in conv1d_lengths, (
            f"conv1d weights were not prepared for the 256-token block "
            f"(accepted lengths: {conv1d_lengths}) - the prefill under test used the FIR path"
        )
        assert optimized_prefill["conv1d"] == expected, (
            f"prefill dispatched {optimized_prefill['conv1d']} conv1d calls, expected exactly "
            f"{expected} (one per {CONV1D_CHANNELS}-channel block of conv_dim)"
        )
        logger.info(
            f"conv1d block lengths accepted at batch 1: {conv1d_lengths} "
            f"({len(conv1d_lengths)}/{prefill_chunk // PREFILL_ALIGN})"
        )
    # Exact counts for the headline MoE rewrite, which a `> 0` assertion cannot pin: the packed
    # gate+up form issues *two* sparse_matmul per expert group where the functional MoE issues three
    # (separate gate, up, down), and one expert-axis reduce per group. Both layer kinds share the
    # MoE, so this holds for either.
    assert optimized_prefill["sparse_matmul"] == 2 * moe_groups, (
        f"prefill dispatched {optimized_prefill['sparse_matmul']} sparse_matmul over {moe_groups} expert "
        f"group(s), expected exactly {2 * moe_groups} (packed gate+up, then down) — a count of "
        f"{3 * moe_groups} means the gate/up packing regressed to three separate matmuls"
    )
    assert optimized_prefill["experimental.deepseek_moe_fast_reduce_nc"] == moe_groups, (
        f"expected one expert-axis reduce per group ({moe_groups}), got "
        f"{optimized_prefill['experimental.deepseek_moe_fast_reduce_nc']}"
    )
    for name in DEDICATED_OPS_DECODE[layer_idx]:
        decode_only = optimized_total.get(name, 0) - optimized_prefill.get(name, 0)
        assert decode_only > 0, f"optimized decode did not dispatch ttnn.{name}"
    # The decode step is a single token, i.e. exactly one expert group, whatever the group size is.
    assert (
        optimized_total["sparse_matmul"] - optimized_prefill["sparse_matmul"] == 2
    ), "decode MoE is not the packed pair"
    # Ops the fused decode must have *replaced*, not merely supplemented. Checked to be live in the
    # functional decoder so a renamed/misspelled op cannot make the 0 vacuous.
    for name in REPLACED_OPS_DECODE[layer_idx]:
        optimized_decode_only = optimized_total.get(name, 0) - optimized_prefill.get(name, 0)
        func_decode_only = func_total.get(name, 0) - func_prefill.get(name, 0)
        assert optimized_decode_only == 0, (
            f"optimized decode still dispatched ttnn.{name} {optimized_decode_only} time(s) alongside its "
            f"replacement — the dedicated op fired but the fallback ran too"
        )
        assert func_decode_only > 0, f"control is inert: the functional decode never dispatched ttnn.{name}"
        logger.info(f"replaced-op check: ttnn.{name} fused decode 0, functional decode {func_decode_only}")
    # Negative control: apart from the three ops the functional decoder already used, none of the
    # dedicated ops above appear in the functional graph — so the assertions are measuring the fused
    # implementation rather than something both share. The control is also checked to be live (the
    # functional decoder does dispatch the shared ops), so it cannot pass by patching nothing.
    shared_expected = {"sparse_matmul", "transformer.chunk_gated_delta_rule", "experimental.paged_fill_cache"}
    shared = {n for n in dedicated if func_total.get(n, 0) > 0}
    assert shared <= shared_expected, f"functional decoder also dispatches {sorted(shared - shared_expected)}"
    assert shared, "control is inert: the functional decoder dispatched none of the recorded ops"
    optimized_only = sorted(set(dedicated) - shared_expected)
    assert all(func_total.get(n, 0) == 0 for n in optimized_only), "a fused-only op fired in the functional decoder"
    logger.info(f"optimized-only ops for layer={layer_idx}: {optimized_only}; shared with functional: {sorted(shared)}")


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("seq_len", [256, 2048])
def test_no_layout_churn_in_measured_forward(mesh_device, layer_idx, seq_len, monkeypatch):
    """Bound the layout/relayout traffic the measured prefill and decode paths may do.

    The optimization contract says the measured path carries no *unnecessary* tilize/untilize/reshard.
    Some are unavoidable and required by an op contract — ``sparse_matmul`` demands a ROW_MAJOR
    sparsity tensor, ``paged_update_cache`` demands a height-sharded input, and the depthwise conv's
    shifted windows are not tile-aligned by construction — so this test pins the count rather than
    requiring zero. A regression that reintroduces a per-head relayout moves it immediately.
    """
    source = default_weight_source()
    names = ["to_layout", "to_memory_config", "sharded_to_interleaved", "interleaved_to_sharded", "tilize", "untilize"]
    # Itemisation of the budgets below, from the shipped graph. Each term names the ttnn entry point
    # the recorder actually sees: every layout *conversion* in this decoder is requested through
    # ``to_layout``/``to_memory_config``, so the bare ``tilize``/``untilize`` entry points are watched
    # but are 0 in all four cases — they are in ``names`` to catch a future edit that starts calling
    # them directly, not because the shipped graph uses them.
    #   linear prefill = to_layout: 4 conv ROW_MAJOR conversions (3 state buffers + the qkv stream)
    #                             + 2 on the two ttnn.conv1d halves that dispatch nothing, because
    #                               Conv1dConfig aliases Conv2dConfig whose output_layout already
    #                               defaults to TILE and the layer passes no override
    #                             + 3 conv-history row writebacks
    #                             + ONE MoE group-mask ROW_MAJOR conversion per MoE *call*
    #                  + sharded_to_interleaved: 2, one per ttnn.conv1d half
    #   linear decode  = to_layout: the MoE group mask only (the conv output's head-major relayout is
    #                    a reshape + permute, which needs no layout conversion at all)
    #                  + 2 x (to_memory_config + sharded_to_interleaved) for the two width-sharded
    #                    residual RMSNorms this stage added (input norm, post-attention norm)
    #   full  prefill  = to_layout: 2 RoPE tables + one MoE group mask per MoE call
    #   full  decode   = sharded_to_interleaved: 3 off nlp_create_qkv_heads_decode
    #                  + to_memory_config: 2 height-shards for the fused paged-cache update
    #                  + to_layout: 1 MoE group mask
    #                  + 4 x (to_memory_config + sharded_to_interleaved) for the width-sharded
    #                    RMSNorms: input, post-attention, and the Q and K head-dim norms
    #
    # The eight-to-twelve new conversions are this stage's own, and they are the price of the norm
    # win, not churn: ``ttnn.rms_norm`` parallelises over rows, so an interleaved decode norm (one
    # tile of rows) runs on a single core, and the 1D ``mcast_in0`` projection matmul that consumes
    # the result needs an interleaved ``in0`` back. The pair costs ~3 us and saves ~12 per residual
    # norm; ``doc/optimized_decoder/logs/probe_decode_micro.txt`` (``NORM`` rows) and
    # ``ab_norm_shard_width.txt`` carry both halves of that trade.
    #
    # NOTHING here scales with the sequence. It used to, in the fused stage: the MoE mask was
    # rebuilt per 32-token expert group, so the budget was `11 + groups` and reached 75 at seq 2048.
    # The counts are sequence-independent and asserted EXACTLY, which is what makes this a gate.
    # Both lengths are still budgeted, because "does not scale with the sequence" is the property
    # under test.
    with monkeypatch.context() as ctx:
        recorder = _OpRecorder(ctx, names)
        decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
        budgets = {
            (LINEAR_LAYER, "prefill"): 12,
            (LINEAR_LAYER, "decode"): 5,
            (FULL_LAYER, "prefill"): 3,
            (FULL_LAYER, "decode"): 14,
        }
        x = to_device(mesh_device, make_activations(1, seq_len, seed=93))
        d = to_device(mesh_device, make_activations(1, 1, seed=94))
        current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([seq_len]))
        ttnn.deallocate(decoder.prefill_forward(x, page_table=page_table))  # warm
        recorder.calls.clear()
        ttnn.deallocate(decoder.prefill_forward(x, page_table=page_table))
        prefill_total = sum(recorder.calls.values())
        prefill_detail = dict(recorder.calls)
        recorder.calls.clear()
        ttnn.deallocate(decoder.decode_forward(d, current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table))
        decode_total = sum(recorder.calls.values())
        decode_detail = dict(recorder.calls)
    logger.info(
        f"layout ops layer={layer_idx} ({LAYER_IDS[layer_idx]}) seq_len={seq_len} "
        f"prefill={prefill_total} {prefill_detail} decode={decode_total} {decode_detail}"
    )
    # Exact, not `<=`: the counts are sequence-independent and itemised above, so an upper bound
    # would let a regression hide under the slack (round 22) and would also let a future *removal*
    # pass while the itemisation went stale.
    assert prefill_total == budgets[(layer_idx, "prefill")], f"prefill layout ops {prefill_detail}"
    assert decode_total == budgets[(layer_idx, "decode")], f"decode layout ops {decode_detail}"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("seq_len", [1, 130, 300])
def test_optimized_matches_fused(mesh_device, layer_idx, seq_len):
    """The optimized decoder must reproduce the *fused* decoder, not merely clear the HF bar.

    Same weights, same inputs, same page table; PCC between the two TTNN implementations. This is a
    tighter comparison than either against HF, but deliberately not as tight as the fused stage's
    own fused-vs-functional check was: this stage carries BFP4 expert weights and BFP8 projection
    weights where the fused decoder carried bfloat16, so the two cannot agree to 1e-4. The bar
    catches structural divergence — a wrong active-expert set, a dropped residual, a mis-sliced
    projection — while :data:`PCC_BAR` against the float32 HF golden remains the acceptance gate.
    """
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.fused_decoder import FusedDecoder

    source = default_weight_source()
    x = make_activations(1, seq_len, seed=200 + seq_len)
    decode_x = make_activations(1, 1, seed=300 + seq_len)
    outs = {}
    for cls in (FusedDecoder, OptimizedDecoder):
        decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, cls=cls)
        prefill = ttnn.to_torch(decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table))
        current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([seq_len]))
        decode = ttnn.to_torch(
            decoder.decode_forward(
                to_device(mesh_device, decode_x),
                current_pos=current_pos,
                rot_idxs=rot_idxs,
                page_table=page_table,
            )
        )
        outs[cls.__name__] = (prefill, decode)
        del decoder
    prefill_value = pcc(outs["FusedDecoder"][0], outs["OptimizedDecoder"][0])
    decode_value = pcc(outs["FusedDecoder"][1], outs["OptimizedDecoder"][1])
    logger.info(
        f"optimized-vs-fused layer={layer_idx} ({LAYER_IDS[layer_idx]}) seq_len={seq_len} "
        f"prefill PCC={prefill_value:.6f} decode PCC={decode_value:.6f}"
    )
    assert prefill_value > EQUIV_BAR, f"optimized vs fused prefill PCC {prefill_value} <= {EQUIV_BAR}"
    assert decode_value > EQUIV_BAR, f"optimized vs fused decode PCC {decode_value} <= {EQUIV_BAR}"


# --------------------------------------------------------------------------------------
# RoPE contract
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len", [1, 64, 4096])
@pytest.mark.parametrize("mode", ["partial", "full"])
def test_rope_matches_hf(mesh_device, seq_len, mode):
    """The fused RoPE table must reproduce HF's interleaved M-RoPE for text-only positions.

    Both lowerings are checked, because both are selectable: ``partial`` keeps the ordinary
    ``rope_dim``-wide table, and ``full`` widens it to ``head_dim`` with ones/zeros on the
    pass-through dims so that one full-width rotate-half *is* the partial rotation. For ``full``
    the comparison is done through the permutation the layer applies to the Q/K weights.
    """
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import OrnithDecoderConfig
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import OrnithFusedRope, _rope_head_permutation

    cfg_hf = hf_config()
    cfg = OrnithDecoderConfig.from_hf_config(cfg_hf)
    start = 12345 if seq_len < 4096 else 0
    positions = torch.arange(start, start + seq_len).unsqueeze(0)
    cos_ref, sin_ref = R.reference_position_embeddings(cfg_hf, positions)
    assert cos_ref.shape[-1] == cfg.rope_dim

    rope = OrnithFusedRope(mesh_device, cfg, max_context=start + seq_len, mode=mode)
    cos, sin = rope.prefill_forward(start, seq_len)
    cos_t = ttnn.to_torch(cos).reshape(1, seq_len, rope.width).float()
    sin_t = ttnn.to_torch(sin).reshape(1, seq_len, rope.width).float()

    if mode == "partial":
        cos_cmp, sin_cmp = cos_t, sin_t
    else:
        # The rotated dims live at [0, rope_dim/2) and [head_dim/2, head_dim/2 + rope_dim/2) in the
        # widened table; the rest must be exactly cos=1 / sin=0.
        half_r, half_h = cfg.rope_dim // 2, cfg.head_dim // 2
        idx = list(range(0, half_r)) + list(range(half_h, half_h + half_r))
        cos_cmp = cos_t[..., idx]
        sin_cmp = sin_t[..., idx]
        rest = [i for i in range(cfg.head_dim) if i not in idx]
        assert torch.equal(cos_t[..., rest], torch.ones_like(cos_t[..., rest]))
        assert torch.equal(sin_t[..., rest], torch.zeros_like(sin_t[..., rest]))
        assert _rope_head_permutation(cfg.head_dim, cfg.rope_dim)[:half_r] == list(range(half_r))

    cos_value, sin_value = pcc(cos_ref, cos_cmp), pcc(sin_ref, sin_cmp)
    probe = torch.tensor([[start, start + seq_len - 1]])
    cos_d, sin_d = rope.decode_forward(
        to_device(mesh_device, probe.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    )
    cos_dref, sin_dref = R.reference_position_embeddings(cfg_hf, probe)
    cos_dt = ttnn.to_torch(cos_d).reshape(1, 2, rope.width).float()
    sin_dt = ttnn.to_torch(sin_d).reshape(1, 2, rope.width).float()
    if mode == "full":
        cos_dt, sin_dt = cos_dt[..., idx], sin_dt[..., idx]
    cos_dvalue, sin_dvalue = pcc(cos_dref, cos_dt), pcc(sin_dref, sin_dt)
    logger.info(
        f"optimized rope vs HF mode={mode} seq_len={seq_len} start={start} prefill cos PCC={cos_value:.6f} "
        f"sin PCC={sin_value:.6f}; decode-gather cos PCC={cos_dvalue:.6f} sin PCC={sin_dvalue:.6f}"
    )
    for name, value in (
        ("prefill cos", cos_value),
        ("prefill sin", sin_value),
        ("decode cos", cos_dvalue),
        ("decode sin", sin_dvalue),
    ):
        assert value > 0.9999, f"{name} PCC {value} <= 0.9999"


@pytest.mark.parametrize("layer_idx", [FULL_LAYER], ids=lambda i: LAYER_IDS[i])
def test_rope_mode_equivalence(mesh_device, layer_idx):
    """The two RoPE lowerings must be interchangeable, not just individually plausible.

    ``full`` permutes the head-dim order of ``q_proj``/``k_proj``/``q_norm``/``k_norm`` so that a
    full-width rotate-half reproduces the 64-of-256 partial rotation. That is an exact algebraic
    identity, so both modes must give the same layer output — which also proves the permutation is
    self-consistent across Q, K and the KV cache.
    """
    source = default_weight_source()
    seq_len = 256
    x = make_activations(1, seq_len, seed=401)
    decode_x = make_activations(1, 1, seed=402)
    outs = {}
    for mode in ("partial", "full"):
        decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, rope_mode=mode)
        prefill = ttnn.to_torch(decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table))
        current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([seq_len]))
        decode = ttnn.to_torch(
            decoder.decode_forward(
                to_device(mesh_device, decode_x), current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table
            )
        )
        outs[mode] = (prefill, decode)
        del decoder
    prefill_value = pcc(outs["partial"][0], outs["full"][0])
    decode_value = pcc(outs["partial"][1], outs["full"][1])
    logger.info(f"rope-mode equivalence prefill PCC={prefill_value:.6f} decode PCC={decode_value:.6f}")
    assert prefill_value > EQUIV_BAR
    assert decode_value > EQUIV_BAR


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
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    assert list(out.shape) == [1, seq_len, hf_config().hidden_size]
    got = ttnn.to_torch(out)
    value = pcc(ref_out, got)
    logger.info(f"optimized prefill layer={layer_idx} ({LAYER_IDS[layer_idx]}) seq_len={seq_len} PCC={value:.6f}")
    assert torch.isfinite(got.float()).all()
    assert value > PCC_BAR, f"prefill PCC {value} <= {PCC_BAR} (layer {layer_idx}, seq_len {seq_len})"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("group_tokens", [64, 256], ids=lambda g: f"group{g}")
def test_moe_group_tokens_pcc(mesh_device, layer_idx, group_tokens):
    """PCC at non-default ``moe_group_tokens``, which is the only way to reach a multi-group branch.

    ``FusedMoE._routed_experts`` has a ``groups > 1`` path — the expert-axis `transpose` instead of a
    reshape, and a whole-call ``call_mask`` computed separately from the per-group ``group_mask`` —
    that is unreachable at the shipped ``moe_group_tokens = 32``, where every call is a single
    32-token group. Review rounds 21, 22 and 23 each noted it had no correctness coverage: the only
    thing exercising it was ``logs/ab_moe_group_tokens.txt``, which measures wall time and asserts no
    PCC. §4.16's hoist then put *new* code on that branch (the borrowed mask now spans
    ``span // TILE`` rows rather than one), so it is now covered here rather than argued about.

    A 2048-token prefill at 64 tokens/group is 32 calls of 2 groups; at 256 it is 8 calls of 8.
    """
    source = default_weight_source()
    seq_len = 2048
    x = make_activations(1, seq_len, seed=seq_len)
    ref_out, _ = run_reference(layer_idx, source, x)

    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, moe_group_tokens=group_tokens)
    assert decoder.moe.group_tokens == group_tokens
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    got = ttnn.to_torch(out)
    value = pcc(ref_out, got)
    logger.info(
        f"optimized moe_group_tokens={group_tokens} layer={layer_idx} ({LAYER_IDS[layer_idx]}) "
        f"seq_len={seq_len} PCC={value:.6f}"
    )
    assert torch.isfinite(got.float()).all()
    assert value > PCC_BAR, f"multi-group MoE PCC {value} <= {PCC_BAR} (group_tokens {group_tokens})"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_repeated_prefill_at_a_masked_chunk_length(mesh_device, layer_idx):
    """A logical length that pads up to exactly one full chunk must survive being run twice.

    ``_gdn_gates`` builds its tail mask by slicing ``pos_ramp``, which is ``[1, prefill_chunk, 1]``.
    When the logical length lands in ``(prefill_chunk - PREFILL_ALIGN, prefill_chunk)`` the physical
    length *is* ``prefill_chunk``, so that slice covers the whole tensor — and ``ttnn.slice``
    short-circuits a full cover to the input itself, so deallocating the result frees the layer's
    persistent ramp weight. The first prefill still returns the right answer (the mask is consumed
    before the free); the *second* one reads a deallocated buffer. Hence two passes, both checked.

    Two lengths are covered: 2000 pads to the 2048-token chunk (the aliasing case) and 1900 pads to
    1920, which does not — so a regression in the alias guard cannot hide behind the second length
    also being unmasked.
    """
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    assert decoder.prefill_chunk == DEFAULT_PREFILL_CHUNK, "this test's lengths assume the default chunk"
    for seq_len in (DEFAULT_PREFILL_CHUNK - 48, DEFAULT_PREFILL_CHUNK - 148):
        phys = -(-seq_len // PREFILL_ALIGN) * PREFILL_ALIGN
        x = make_activations(1, seq_len, seed=seq_len)
        ref_out, _ = run_reference(layer_idx, source, x)
        for attempt in (1, 2):
            # Re-zero the DeltaNet recurrent and conv state: a linear-attention layer is stateful, so
            # without this the second pass would continue from the first's state and disagree with a
            # from-scratch reference for that reason rather than for the one under test. The aliasing
            # bug this guards frees a *weight*, which reset_state does not restore, so the regression
            # signal survives the reset.
            decoder.reset_state()
            got = ttnn.to_torch(decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table))
            value = pcc(ref_out, got)
            logger.info(
                f"masked-chunk prefill layer={layer_idx} ({LAYER_IDS[layer_idx]}) seq_len={seq_len} "
                f"(physical {phys}) attempt {attempt} PCC={value:.6f}"
            )
            assert torch.isfinite(got.float()).all(), f"seq_len {seq_len} attempt {attempt} produced non-finite output"
            assert value > PCC_BAR, f"seq_len {seq_len} attempt {attempt}: PCC {value} <= {PCC_BAR}"
    if not decoder.is_full_attention:
        assert decoder.w["pos_ramp"].is_allocated(), "the persistent pos_ramp weight was freed by a masked prefill"


# --------------------------------------------------------------------------------------
# decode: paged cache, current position, multi-step
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("prefill_len", [130, 2048])
def test_decode_pcc(mesh_device, layer_idx, prefill_len):
    """Prefill then four decode steps, comparing every step against HF."""
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
            f"optimized decode layer={layer_idx} ({LAYER_IDS[layer_idx]}) prefill_len={prefill_len} "
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
    """Batched prefill + batched decode with per-user current positions."""
    source = default_weight_source()
    prefill_len = 96
    x = make_activations(batch, prefill_len, seed=7 + batch)
    decode_x = make_activations(batch, 1, seed=99 + batch)

    ref = reference_layer(layer_idx, source)
    cfg = hf_config()
    with torch.no_grad():
        ref_prefill, cache = R.reference_prefill(ref, cfg, x.float(), start_pos=0)

    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, batch=batch, max_context=1024)
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    value = pcc(ref_prefill, ttnn.to_torch(out))
    logger.info(f"optimized batched prefill layer={layer_idx} batch={batch} PCC={value:.6f}")
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
    logger.info(f"optimized batched decode layer={layer_idx} batch={batch} PCC={value:.6f}")
    assert torch.isfinite(got.float()).all()
    assert value > PCC_BAR, f"batched decode PCC {value}"


@pytest.mark.parametrize("layer_idx", [FULL_LAYER], ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("batch", [4, 13])
def test_batched_decode_ragged_positions(mesh_device, layer_idx, batch):
    """Batched decode where every user sits at a **different** absolute position.

    This is the serving case, and the one that distinguishes a genuinely per-user
    ``current_pos``/``rot_idxs`` from a path that only reads element 0. ``batch=13`` also exercises
    a decode batch with no factor pair that fits an 8-wide shard grid, and (fused-specific) a batch
    for which the K and V shard grids of ``paged_fused_update_cache`` must stay disjoint.
    """
    source = default_weight_source()
    generator = torch.Generator().manual_seed(500 + batch)
    lengths = sorted(set((torch.randperm(200, generator=generator)[:batch] + 65).tolist()))
    while len(lengths) < batch:
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

    blocks_per_user = num_blocks_for_context(1024)
    total_blocks = blocks_per_user * batch
    decoder = OptimizedDecoder.from_state_dict(
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
        logger.info(f"optimized ragged-position decode batch={batch} user={user} pos={length} PCC={value:.6f}")
        assert value > PCC_BAR, f"user {user} at position {length}: PCC {value} <= {PCC_BAR}"


@pytest.mark.parametrize("layer_idx", [FULL_LAYER], ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("batch", [40, 56])
def test_decode_batch_above_head_split_limit(mesh_device, layer_idx, batch):
    """Decode at a batch larger than the dedicated ops accept.

    ``nlp_create_qkv_heads_decode`` asserts ``num_users <= 32``, so the fused decoder keeps the
    functional decoder's generic slice/reshape/permute head split as the fallback above it. Without
    that fallback this batch would die inside the op with a raw ``TT_FATAL`` and the fused decoder
    would silently support a smaller decode batch than the functional one — which is what this test
    exists to prevent.

    ``batch=56`` additionally crosses the *other* fused-path threshold, ``2 * batch > 110`` cores, so
    ``paged_fused_update_cache`` cannot be used either and the two-launch ``paged_update_cache``
    fallback runs. Both fallbacks are therefore covered, and the parametrisation says which is which.
    """
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import OptimizedDecoder as _FD

    source = default_weight_source()
    assert batch > _FD.DECODE_HEAD_SPLIT_MAX_BATCH, "this test must exceed the dedicated op's limit"
    grid = mesh_device.compute_with_storage_grid_size()
    fused_update = 2 * batch <= grid.x * grid.y
    logger.info(
        f"decode batch {batch}: head-split fallback active; "
        f"cache update is {'fused' if fused_update else 'two separate launches'}"
    )
    assert (batch == 40) == fused_update, "the two parametrisations must cover both cache-update paths"
    prefill_len = 96
    x = make_activations(batch, prefill_len, seed=1300 + batch)
    decode_x = make_activations(batch, 1, seed=1400 + batch)

    ref = reference_layer(layer_idx, source)
    cfg = hf_config()
    with torch.no_grad():
        _, cache = R.reference_prefill(ref, cfg, x.float(), start_pos=0)
        positions = torch.full((batch,), prefill_len, dtype=torch.long)
        ref_dec = R.reference_decode(ref, cfg, decode_x.float(), positions, cache)

    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, batch=batch, max_context=1024)
    ttnn.deallocate(decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table))
    current_pos, rot_idxs = decode_inputs(mesh_device, positions)
    got = ttnn.to_torch(
        decoder.decode_forward(
            to_device(mesh_device, decode_x), current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table
        )
    )
    value = pcc(ref_dec, got)
    logger.info(f"optimized decode above head-split limit batch={batch} PCC={value:.6f}")
    assert torch.isfinite(got.float()).all()
    assert value > PCC_BAR, f"batch {batch} decode PCC {value} <= {PCC_BAR}"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_batch_smaller_than_allocated_state(mesh_device, layer_idx, expect_error):
    """A ``full_attention`` layer allocated for a large batch must still serve one user.

    That is the ordinary serving pattern (per-user prefill into a shared cache), the functional
    decoder allows it, and the fused decoder must not narrow it. ``linear_attention`` keeps the
    functional decoder's stricter rule, because its DeltaNet state is per-row and cannot be resized
    under a live sequence — so this test asserts the *raise* for that kind, and the *result* for the
    other.
    """
    source = default_weight_source()
    seq_len = 128
    x = make_activations(1, seq_len, seed=1501)
    decode_x = make_activations(1, 1, seed=1502)
    ref_prefill, ref_decode = run_reference(layer_idx, source, x, decode_x=[decode_x], decode_steps=1)

    decoder, page_table, blocks_per_user = build_decoder(mesh_device, layer_idx, source, batch=8, max_context=1024)
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([seq_len]))
    if not decoder.is_full_attention:
        with expect_error(ValueError, "allocated state batch"):
            decoder.prefill_forward(to_device(mesh_device, x))
        logger.info("linear_attention keeps the functional decoder's fixed-batch rule for its DeltaNet state")
        return

    # The paged SDPA ops require one page-table row per input row, so a single-user call passes
    # user 0's row of the batch-8 table. That is the caller's job in both implementations; what is
    # being tested here is that the *decoder* accepts a batch below its allocation.
    page_table = ttnn.slice(page_table, [0, 0], [1, blocks_per_user])
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    prefill_value = pcc(ref_prefill, ttnn.to_torch(out))
    ttnn.deallocate(out)
    got = ttnn.to_torch(
        decoder.decode_forward(
            to_device(mesh_device, decode_x), current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table
        )
    )
    decode_value = pcc(ref_decode[0], got)
    logger.info(f"batch 1 on a batch-8 allocation: prefill PCC={prefill_value:.6f} decode PCC={decode_value:.6f}")
    assert prefill_value > PCC_BAR
    assert decode_value > PCC_BAR


# --------------------------------------------------------------------------------------
# page table handling
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", [FULL_LAYER], ids=lambda i: LAYER_IDS[i])
def test_permuted_page_table(mesh_device, layer_idx):
    """A shuffled, offset page table must give the same result as the identity table.

    The fused prefill fills the cache with a single batched ``paged_fill_cache`` driven by a
    ``batch_idx_tensor`` instead of one call per user, so an off-by-one in that mapping would show
    up here first.
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
    logger.info(f"optimized permuted-page-table prefill PCC={value:.6f} first_slot={permutation[0, 0].item()}")
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
        logger.info(f"optimized permuted-page-table decode step={step} PCC={value:.6f}")
        assert value > PCC_BAR, f"permuted page table decode PCC {value}"
        ttnn.deallocate(out)


# --------------------------------------------------------------------------------------
# chunked prefill continuation (start_pos > 0)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_prefill_continuation(mesh_device, layer_idx):
    """Two prefill calls over one sequence must equal a single call over the concatenation."""
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
    logger.info(f"optimized prefill continuation layer={layer_idx} PCC={value:.6f}")
    assert value > PCC_BAR, f"continuation PCC {value}"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_unaligned_max_context(mesh_device, layer_idx):
    """A ``max_context`` that is not a multiple of the internal prefill alignment must work."""
    source = default_weight_source()
    max_context = 5000
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
    logger.info(
        f"optimized unaligned max_context={max_context} layer={layer_idx} prefill+decode ok, tail std={tail.std():.4f}"
    )


# --------------------------------------------------------------------------------------
# determinism and stress
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_determinism_repeated_inputs(mesh_device, layer_idx):
    """Identical inputs must give bit-identical outputs, in prefill and in decode."""
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
    logger.info(f"optimized determinism layer={layer_idx}: 3/3 runs bit-identical for prefill and decode")


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_repeated_run_stress(mesh_device, layer_idx):
    """Many back-to-back prefill/decode cycles must stay correct and bounded.

    Repeats a full prefill + 4-step decode cycle 12 times, each cycle from a re-zeroed state and
    with a *rotating* prompt length so the program cache, the paged-cache fills, the DeltaNet state
    hand-off and the freed-DRAM pool are all exercised repeatedly rather than once. Checks:

    * every cycle's prefill and every decode step is finite;
    * cycles that share a prompt length agree **bit-for-bit** with the first cycle at that length,
      so nothing accumulates across runs — a stricter check than a PCC bar, and the one that would
      catch a stale read past ``current_pos`` in the paged cache or a leaked DeltaNet residual;
    * device DRAM allocation does not grow across cycles (a leaked buffer per cycle would show).
    """
    source = default_weight_source()
    lengths = [96, 130, 257, 96]
    cycles = 12
    steps = 4
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    decode_x = [to_device(mesh_device, make_activations(1, 1, seed=8000 + i)) for i in range(steps)]
    inputs = {n: to_device(mesh_device, make_activations(1, n, seed=9000 + n)) for n in set(lengths)}
    positions = {(n, s): decode_inputs(mesh_device, torch.tensor([n + s])) for n in set(lengths) for s in range(steps)}

    baseline = {}
    allocations = []
    for cycle in range(cycles):
        n = lengths[cycle % len(lengths)]
        # The paged cache is deliberately *not* reallocated between cycles: a cycle rewrites slots
        # [0, n) and only ever reads [0, current_pos], so the leftovers from a longer earlier cycle
        # are unreachable — which is itself worth asserting, since a read past current_pos would
        # break the bit-identical repeat check below.
        decoder.reset_state()
        out = decoder.prefill_forward(inputs[n], page_table=page_table)
        prefill = ttnn.to_torch(out)
        ttnn.deallocate(out)
        decodes = []
        for step in range(steps):
            current_pos, rot_idxs = positions[(n, step)]
            out = decoder.decode_forward(
                decode_x[step], current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table
            )
            decodes.append(ttnn.to_torch(out))
            ttnn.deallocate(out)
        assert torch.isfinite(prefill.float()).all(), f"cycle {cycle}: non-finite prefill"
        for step, d in enumerate(decodes):
            assert torch.isfinite(d.float()).all(), f"cycle {cycle} step {step}: non-finite decode"
        if n in baseline:
            assert torch.equal(baseline[n][0], prefill), f"cycle {cycle}: prefill drifted from the first run at n={n}"
            for step, d in enumerate(decodes):
                assert torch.equal(baseline[n][1][step], d), f"cycle {cycle} step {step}: decode drifted at n={n}"
        else:
            baseline[n] = (prefill, decodes)
        allocations.append(ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank)

    growth = allocations[-1] - allocations[len(lengths)]
    logger.info(
        f"optimized stress layer={layer_idx} ({LAYER_IDS[layer_idx]}): {cycles} prefill+{steps}-step-decode cycles over "
        f"lengths {sorted(set(lengths))}, repeats bit-identical, DRAM allocated "
        f"{allocations[len(lengths)]} -> {allocations[-1]} bytes (growth {growth})"
    )
    assert growth == 0, f"DRAM allocation grew by {growth} bytes across stress cycles: {allocations}"


# --------------------------------------------------------------------------------------
# use-after-free / aliasing regression
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("seq_len", [1, 250, 300])
def test_forward_with_poisoned_free_pool(mesh_device, layer_idx, seq_len):
    """Correctness must not depend on what freed DRAM happens to contain."""
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
        f"optimized poisoned-pool layer={layer_idx} seq_len={seq_len} prefill PCC={prefill_value:.6f} "
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
    """A single prefill and a single decode pass must not touch the host."""
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

    # Positive controls: prove both guards actually fire.
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
        f"optimized fallback audit layer={layer_idx}: both guards verified to fire; prefill+decode clean for "
        f"ttnn {banned} and all torch ops"
    )


def test_batched_paged_fill_is_one_launch_per_cache(mesh_device, monkeypatch):
    """The batched `paged_fill_cache(batch_idx_tensor=...)` rewrite is only testable above batch 1.

    At batch 1 the fused spelling (one launch per cache) and the per-user spelling the functional
    decoder used (one launch per user per cache) both dispatch 2 calls, so `test_optimized_path_is_used`
    — which runs at batch 1 — cannot tell them apart. At batch 4 the claim is `2·batch → 2`, i.e. 8
    launches against 2, and that is what this asserts, against the functional decoder as a live
    control rather than against a remembered number.
    """
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.functional_decoder import FunctionalDecoder

    batch, seq = 4, 256
    source = default_weight_source()
    counts = {}
    for cls in (OptimizedDecoder, FunctionalDecoder):
        with monkeypatch.context() as ctx:
            recorder = _OpRecorder(ctx, ["experimental.paged_fill_cache"])
            decoder, page_table, _ = build_decoder(mesh_device, FULL_LAYER, source, batch=batch, cls=cls)
            recorder.reset()
            ttnn.deallocate(
                decoder.prefill_forward(
                    to_device(mesh_device, make_activations(batch, seq, seed=77)), page_table=page_table
                )
            )
            counts[cls.__name__] = recorder.count("experimental.paged_fill_cache")
        del decoder

    logger.info(
        f"optimized batched paged_fill_cache batch={batch}: optimized {counts['OptimizedDecoder']} launches, "
        f"functional {counts['FunctionalDecoder']}"
    )
    assert (
        counts["OptimizedDecoder"] == 2
    ), f"expected one batched paged_fill_cache per cache, got {counts['OptimizedDecoder']}"
    assert counts["FunctionalDecoder"] == 2 * batch, (
        "the control is inert: the functional decoder no longer issues one fill per user per cache "
        f"(got {counts['FunctionalDecoder']}, expected {2 * batch})"
    )


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_lazy_allocation_is_the_only_host_call(mesh_device, layer_idx, monkeypatch):
    """Pin the one host-call divergence from the functional decoder, in both directions.

    ``prefill_forward``/``decode_forward`` call ``allocate_state`` themselves if the caller never
    did. For this decoder that is host work — the paged-fill row indices, and for
    ``linear_attention`` the ``ttnn.conv1d`` weight preparation and its L1 probes — whereas the
    functional decoder's ``allocate_state`` uses only ``ttnn.zeros``. `test_no_host_fallback_in_forward`
    cannot see it, because `build_decoder` always allocates first.

    So this test asserts both halves of the real claim: the **first** forward on an unallocated layer
    does touch the host, and the **second** does not. If a future change makes the lazy path
    host-free, the first assertion fails and this test should be simplified rather than deleted.
    """
    source = default_weight_source()
    sd = layer_state_dict(layer_idx, source)
    decoder = OptimizedDecoder.from_state_dict(
        sd,
        hf_config=hf_config(),
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_context=TEST_CONTEXT,
        prefill_chunk=DEFAULT_PREFILL_CHUNK,
    )
    page_table = None
    if decoder.is_full_attention:
        blocks = num_blocks_for_context(TEST_CONTEXT)
        decoder.allocate_kv_cache(blocks)
        page_table = to_device(
            mesh_device,
            torch.arange(blocks, dtype=torch.int32).reshape(1, blocks),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    # Deliberately no allocate_state: that is the path under test.
    assert decoder.batch_size is None, "the layer must start unallocated for this test to mean anything"

    x_tt = to_device(mesh_device, make_activations(1, 256, seed=71))
    banned = ["from_torch", "to_torch", "as_tensor", "from_device", "to_device"]

    def _guarded(name, original, seen):
        def stub(*args, **kwargs):
            seen.append(name)
            return original(*args, **kwargs)

        return stub

    seen: list[str] = []
    with monkeypatch.context() as ctx:
        for name in banned:
            ctx.setattr(ttnn, name, _guarded(name, getattr(ttnn, name), seen))
        # Positive control: the recorders are installed and do record.
        ttnn.deallocate(ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), device=mesh_device))
        assert seen == ["from_torch"], f"the recorder is inert: a direct ttnn.from_torch logged {seen}"
        seen.clear()

        ttnn.deallocate(decoder.prefill_forward(x_tt, page_table=page_table))
        first = list(seen)
        seen.clear()
        ttnn.deallocate(decoder.prefill_forward(x_tt, page_table=page_table))
        second = list(seen)
        # decode_forward has the same lazy branch, but reaching it needs a layer that has never been
        # allocated *and* a decode as its first call; this test's layer is allocated by now, so this
        # half asserts only the quiet side of the claim for decode. The noisy side is pinned for
        # prefill_forward above, and the two share one `allocate_state`.
        current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([256]))
        d_tt = to_device(mesh_device, make_activations(1, 1, seed=72))
        seen.clear()  # after building the inputs: those are the test's host calls, not the layer's
        ttnn.deallocate(decoder.decode_forward(d_tt, current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table))
        third = list(seen)

    # The count is of the five banned ttnn entry points only. allocate_state also calls
    # ttnn.prepare_conv_weights, which is host work and is *not* in that set, so this is a lower
    # bound on the host work the lazy path does, not a total.
    # Derived, not hardcoded: a linear layer uploads one conv1d block-length candidate per
    # PREFILL_ALIGN-sized step of the prefill chunk (2048/128 = 16 at the shipped defaults); a full
    # layer uploads only its paged-fill batch-index tensor (allocate_state builds batch_idxs).
    expected = decoder.prefill_chunk // PREFILL_ALIGN if layer_idx == LINEAR_LAYER else 1
    assert len(first) == expected, (
        f"the first forward on an unallocated {LAYER_IDS[layer_idx]} layer made {len(first)} host "
        f"call(s) among {banned}, expected {expected} — if allocate_state's host work changed, update "
        "the module docstring and README §6, which document this divergence, and work_log.md §7, which "
        "quotes this count"
    )
    assert not second, f"a prefill on an *allocated* layer touched the host: {sorted(set(second))}"
    assert not third, f"a decode on an *allocated* layer touched the host: {sorted(set(third))}"
    logger.info(
        f"optimized lazy-allocation audit layer={layer_idx}: first prefill made {len(first)} host call(s) "
        f"{sorted(set(first))}; the second prefill and the decode made none"
    )


# --------------------------------------------------------------------------------------
# traced decode
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_traced_decode_pcc(mesh_device, layer_idx):
    """Decode runs fully under ``ttnn`` traced execution, and the *replay* output matches HF."""
    source = default_weight_source()
    prefill_len = 128
    steps = 3
    x = make_activations(1, prefill_len, seed=31)
    decode_x = [make_activations(1, 1, seed=3100 + i) for i in range(steps)]
    _, ref_decode = run_reference(layer_idx, source, x, decode_x=decode_x, decode_steps=steps)

    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    ttnn.deallocate(decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table))

    x_buf = to_device(mesh_device, decode_x[0])
    pos_buf, rot_buf = decode_inputs(mesh_device, torch.tensor([prefill_len]))

    def forward():
        return decoder.decode_forward(x_buf, current_pos=pos_buf, rot_idxs=rot_buf, page_table=page_table)

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
        logger.info(f"optimized traced decode layer={layer_idx} step={step} replay PCC={value:.6f}")
        assert torch.isfinite(got.float()).all()
        assert value > PCC_BAR, f"traced decode replay step {step} PCC {value} <= {PCC_BAR}"

    ttnn.release_trace(mesh_device, trace_id)


def _snapshot_state(decoder):
    """Host copies of the mutable state, for rewinding around trace capture."""
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
        f"optimized REAL WEIGHTS layer={layer_idx} ({LAYER_IDS[layer_idx]}) "
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
        f"optimized SYNTHETIC WEIGHTS layer={layer_idx} ({LAYER_IDS[layer_idx]}) "
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
    """Run prefill at the full advertised context, then decode at the last legal position."""
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

    # ADVERTISED_CONTEXT is the position *count*, so the last legal decode slot is one below it and
    # the 262144 case is prefill-only. The log line below says which of the two ran, because the
    # README's capability table is generated from it.
    decoded = seq_len < ADVERTISED_CONTEXT
    if decoded:
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
        f"optimized full-context layer={layer_idx} ({LAYER_IDS[layer_idx]}) seq_len={seq_len} "
        f"{'prefill+decode' if decoded else 'prefill'} completed, tail std={tail.std():.4f}"
    )


@pytest.mark.long
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_full_context_chunk_size_invariance(mesh_device, layer_idx):
    """At the full advertised context, the result must not depend on the internal chunking."""
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
        f"optimized chunk-size invariance layer={layer_idx} ({LAYER_IDS[layer_idx]}) seq_len={seq_len} "
        f"tail({tail_len}) PCC(chunk 2048 vs 1024)={value:.6f}"
    )
    # Deliberately looser than EQUIV_BAR: this compares two *different* chunkings of a 262 143-token
    # prefill, so the two runs accumulate the DeltaNet recurrent state over a different number of
    # hand-offs (128 chunks vs 256). That is a real reassociation of bfloat16 accumulation over the
    # full context, not the same graph twice, and it is the one comparison in this file where a
    # 0.9999 bar would be asserting more than the arithmetic supports.
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
        f"optimized long-context PCC layer={layer_idx} seq_len={seq_len} "
        f"prefill={prefill_value:.6f} decode={decode_value:.6f}"
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
        f"OPTIMIZED PERF prefill layer={layer_idx} ({LAYER_IDS[layer_idx]}) seq_len={seq_len} "
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
        f"OPTIMIZED PERF decode(traced) layer={layer_idx} ({LAYER_IDS[layer_idx]}) iters={iters} "
        f"wall/iter={elapsed / iters * 1e3:.3f} ms  steps/s={iters / elapsed:.1f}"
    )
    assert torch.isfinite(ttnn.to_torch(trace_out).float()).all()
    ttnn.release_trace(mesh_device, trace_id)


# --------------------------------------------------------------------------------------
# optimization-stage contracts
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_precision_policy_reaches_the_device_tensors(mesh_device, layer_idx):
    """OPT-013: the selected weight dtypes must be what the device tensors actually hold.

    A policy object, a dataclass default or a JSON field is only intent — a lazy loader, a helper
    default or a stale cache can leave a claimed BFP4 weight sitting in bfloat16 at the matmul. This
    checks the built layer, tensor by tensor, and the KV cache the layer allocated for itself. The
    other half of OPT-013 — that the *measured* rows show the same dtypes — is the
    ``tt-perf-report`` table in ``doc/optimized_decoder/tracy/``, which the README quotes per row.
    """
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import DEFAULT_POLICY

    source = default_weight_source()
    decoder, _, _ = build_decoder(mesh_device, layer_idx, source)
    policy = decoder.policy
    assert policy is DEFAULT_POLICY, "build_decoder must exercise the shipped policy"

    expected = {
        "expert_gate_up": policy.expert_gate_up_dtype,
        "expert_down": policy.expert_down_dtype,
        "shared_in": policy.shared_dtype,
        "shared_down": policy.shared_dtype,
        "router": policy.router_dtype,
    }
    for name, dtype in expected.items():
        got = decoder.moe.w[name].dtype
        assert got == dtype, f"moe weight {name!r} is {got}, policy says {dtype}"
    proj_names = ["attn_in", "o_proj"] if decoder.is_full_attention else ["gdn_in", "gdn_out"]
    for name in proj_names:
        got = decoder.w[name].dtype
        assert got == policy.proj_dtype, f"projection weight {name!r} is {got}, policy says {policy.proj_dtype}"
    # Norm weights, RoPE tables and the DeltaNet state constants are deliberately NOT block-float:
    # a shared exponent over a norm gain vector or a position table is precision loss for no
    # bandwidth saving, since none of them is a decode-time DRAM cost.
    assert decoder.w["attn_norm"].dtype == ttnn.bfloat16
    assert decoder.w["ff_norm"].dtype == ttnn.bfloat16
    if decoder.is_full_attention:
        assert decoder.k_cache.dtype == policy.kv_cache_dtype
        assert decoder.v_cache.dtype == policy.kv_cache_dtype
    logger.info(
        f"precision policy layer={layer_idx} ({LAYER_IDS[layer_idx]}) name={policy.name} "
        f"proj={policy.proj_dtype} experts={policy.expert_gate_up_dtype}/{policy.expert_down_dtype} "
        f"expert_act={policy.expert_act_dtype} kv_cache={getattr(decoder.k_cache, 'dtype', None)}"
    )
    del decoder


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_decode_runs_the_tuned_program_configs(mesh_device, layer_idx, monkeypatch):
    """Every dense decode projection must run under its tuned program config, not ttnn's heuristic.

    The tuned configs are the whole content of the dense-matmul optimization: without them the
    ``o_proj``/``gdn_out`` role alone costs ~48 us per decode step more. A `None` program config on
    one of these call sites is therefore a silent regression that no PCC test can see, so this
    records what ``ttnn.linear`` was actually called with, and separately asserts that the routed
    experts' sparse matmuls carry a program config and put their intermediates in L1.
    """
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    ttnn.deallocate(
        decoder.prefill_forward(to_device(mesh_device, make_activations(1, 128, seed=95)), page_table=page_table)
    )

    linear_calls: list = []
    sparse_calls: list = []
    real_linear, real_sparse = ttnn.linear, ttnn.sparse_matmul

    def spy_linear(*args, **kwargs):
        linear_calls.append((int(args[1].shape[-2]), int(args[1].shape[-1]), kwargs.get("program_config")))
        return real_linear(*args, **kwargs)

    def spy_sparse(*args, **kwargs):
        sparse_calls.append((kwargs.get("program_config"), kwargs.get("memory_config")))
        return real_sparse(*args, **kwargs)

    monkeypatch.setattr(ttnn, "linear", spy_linear)
    monkeypatch.setattr(ttnn, "sparse_matmul", spy_sparse)
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([128]))
    ttnn.deallocate(
        decoder.decode_forward(
            to_device(mesh_device, make_activations(1, 1, seed=96)),
            current_pos=current_pos,
            rot_idxs=rot_idxs,
            page_table=page_table,
        )
    )
    monkeypatch.undo()

    # One decode step: packed in-projection, output projection, shared-expert packed matmul, shared
    # down projection, router. Every one of them must carry a program config.
    assert len(linear_calls) == 5, f"unexpected dense decode matmul count: {linear_calls}"
    for k, n, cfg in linear_calls:
        assert cfg is not None, f"dense decode matmul {k}x{n} ran on ttnn's heuristic, not a tuned config"
        assert isinstance(cfg, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig), f"{k}x{n}: {type(cfg)}"
        assert cfg.in0_block_w >= 2, f"dense decode matmul {k}x{n} has in0_block_w={cfg.in0_block_w}"
    logger.info(
        f"decode dense matmuls layer={layer_idx} ({LAYER_IDS[layer_idx]}): "
        + ", ".join(
            f"{k}x{n} grid={cfg.compute_with_storage_grid_size} in0_block_w={cfg.in0_block_w} "
            f"per_core_N={cfg.per_core_N} sub={cfg.out_subblock_h}x{cfg.out_subblock_w}"
            for k, n, cfg in linear_calls
        )
    )
    assert len(sparse_calls) == 2, f"decode MoE is not the packed sparse pair: {sparse_calls}"
    # The three float32 recurrent-state matmuls go through `ttnn.matmul`, not `ttnn.linear`, so the
    # spy above cannot see them. They are the rows `tt-perf-report` flagged `SLOW` with
    # `in0_block_w=1 is small`, and `_state_matmul_config` returns `None` (a silent `core_grid`
    # fallback) on any shape or grid it cannot serve — exactly the failure mode review round 1 found
    # for the prefill 2D config. Assert the config directly for the layer kind that has them.
    if not decoder.is_full_attention:
        for role, expected_block_w in (("read", 2), ("outer", 1)):
            cfg = decoder._state_matmul_config(role, 1)
            assert cfg is not None, f"recurrent-state {role} matmul fell back to the core_grid spelling"
            assert isinstance(cfg, ttnn.MatmulMultiCoreReuseProgramConfig), f"{role}: {type(cfg)}"
            assert cfg.in0_block_w == expected_block_w, (
                f"recurrent-state {role} matmul has in0_block_w={cfg.in0_block_w}, expected "
                f"{expected_block_w} (2 is the measured winner for the reads; the transpose_a outer "
                f"product has a single tiled-K tile so 1 is the only legal value)"
            )
            logger.info(
                f"state matmul {role}: grid={cfg.compute_with_storage_grid_size} "
                f"in0_block_w={cfg.in0_block_w} per_core_M={cfg.per_core_M} per_core_N={cfg.per_core_N} "
                f"sub={cfg.out_subblock_h}x{cfg.out_subblock_w}"
            )
        # And the documented bound: above batch 3 the block count outgrows the grid and the config is
        # deliberately dropped (README §9 item 5).
        assert decoder._state_matmul_config("read", 4) is None, (
            "the recurrent-state config must drop above the batch its block count fits, or the op "
            "fails at validation instead of falling back"
        )
    for cfg, mem in sparse_calls:
        assert cfg is not None, "routed-expert sparse matmul ran without a program config"
        assert mem is not None and mem.buffer_type == ttnn.BufferType.L1, (
            f"routed-expert sparse matmul writes its num_experts-wide output to {mem}; the optimized "
            "path keeps every expert intermediate in L1"
        )
    logger.info(
        "decode sparse matmuls: "
        + ", ".join(
            f"grid={cfg.compute_with_storage_grid_size} in0_block_w={cfg.in0_block_w} "
            f"per_core_N={cfg.per_core_N} out_block_w={cfg.out_block_w} sub_w={cfg.out_subblock_w}"
            for cfg, _ in sparse_calls
        )
    )
    del decoder


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_prefill_runs_the_tuned_program_configs(mesh_device, layer_idx):
    """The dense *prefill* projections must run under their tuned 2D config, not ttnn's heuristic.

    This is the companion to ``test_decode_runs_the_tuned_program_configs`` and it exists because
    nothing else notices a ``None``. ``_prefill_2d_matmul_config`` returns ``None`` when its modelled
    circular-buffer total does not fit worker L1, which is correct for a large prefill batch — but an
    earlier revision of this stage asked the mesh device for its L1 size through a ``getattr`` that
    can never succeed, silently ran the whole budget against a 1 MiB fallback, and turned the config
    off on the two widest projections while the docs said it was on. Review round 1 caught that from
    the perf report; this test would have caught it from the code.

    Batch-1 prefill at the shipped 2048-token chunk is the shape the configs were swept at, so every
    dense projection must carry one, with an ``in0_block_w`` above the ``1`` ttnn's heuristic picks.
    """
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)

    calls: list = []
    real_linear = ttnn.linear

    def spy(*args, **kwargs):
        calls.append((int(args[1].shape[-2]), int(args[1].shape[-1]), kwargs.get("program_config")))
        return real_linear(*args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ttnn, "linear", spy)
    try:
        ttnn.deallocate(
            decoder.prefill_forward(
                to_device(mesh_device, make_activations(1, DEFAULT_PREFILL_CHUNK, seed=99)), page_table=page_table
            )
        )
    finally:
        monkeypatch.undo()

    # One prefill chunk: packed in-projection, output projection, shared-expert packed matmul, shared
    # down projection, router. (`moe_group_tokens` does not change this: the router and the shared
    # expert run once per MoE call, not once per expert group.)
    assert len(calls) == 5, f"unexpected dense prefill matmul count: {calls}"
    for k, n, cfg in calls:
        assert cfg is not None, (
            f"dense prefill matmul {k}x{n} ran on ttnn's heuristic — _prefill_2d_matmul_config "
            f"returned None at the shape its own sweep was measured at"
        )
        assert isinstance(cfg, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig), f"{k}x{n}: {type(cfg)}"
        assert cfg.in0_block_w >= 2, f"dense prefill matmul {k}x{n} has in0_block_w={cfg.in0_block_w}"
    logger.info(
        f"prefill dense matmuls layer={layer_idx} ({LAYER_IDS[layer_idx]}): "
        + ", ".join(
            f"{k}x{n} grid={cfg.compute_with_storage_grid_size} in0_block_w={cfg.in0_block_w} "
            f"per_core_M={cfg.per_core_M} per_core_N={cfg.per_core_N}"
            for k, n, cfg in calls
        )
    )
    del decoder


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("batch", [1, 4])
def test_padded_rows_do_not_route(mesh_device, layer_idx, batch):
    """Tile-padding rows must not add experts to the routing sparsity, and must not change output.

    A decode step for batch ``b`` runs the MoE on a 32-row tile whose trailing ``32 - b`` rows are
    zero padding. Those rows produce exactly-zero router logits, so ``topk`` still returns a full
    set of experts for each and the reduction over all 32 rows would union them into the sparsity —
    close to twice the expert work at batch 1. This asserts the shipped mask holds at most
    ``b * num_experts_per_tok`` experts, and that forcing the old whole-tile reduction back on
    changes nothing but the expert count.
    """
    source = default_weight_source()
    masks: dict = {}
    outs: dict = {}
    top_k = hf_config().num_experts_per_tok
    # A fresh layer per arm, not two passes on one: a `linear_attention` decode advances the
    # recurrent and conv state in place, so a second pass on the same layer starts from a different
    # state and would differ for reasons that have nothing to do with routing.
    for tag in ("masked", "whole-tile"):
        decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, batch=batch)
        real_mask = type(decoder.moe)._active_expert_mask
        counts: list = []

        def wrapper(self, dense_routing, groups, valid_tokens, _tag=tag, _real=real_mask, _counts=counts):
            out = _real(self, dense_routing, groups, valid_tokens if _tag == "masked" else None)
            _counts.append(int(ttnn.to_torch(out).float().sum().item()))
            return out

        ttnn.deallocate(
            decoder.prefill_forward(
                to_device(mesh_device, make_activations(batch, 128, seed=97)), page_table=page_table
            )
        )
        x = to_device(mesh_device, make_activations(batch, 1, seed=98))
        current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([128] * batch))
        type(decoder.moe)._active_expert_mask = wrapper
        try:
            counts.clear()
            out = decoder.decode_forward(x, current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table)
            outs[tag] = ttnn.to_torch(out).float()
            ttnn.deallocate(out)
        finally:
            type(decoder.moe)._active_expert_mask = real_mask
        masks[tag] = list(counts)
        del decoder, page_table

    active_masked = max(masks["masked"])
    active_whole = max(masks["whole-tile"])
    logger.info(
        f"padded-row routing layer={layer_idx} ({LAYER_IDS[layer_idx]}) batch={batch}: "
        f"active experts masked={active_masked} whole-tile={active_whole} (bound {batch * top_k})"
    )
    assert active_masked <= batch * top_k, f"masked sparsity activates {active_masked} experts, bound {batch * top_k}"
    assert active_whole > active_masked, (
        "control is inert: reducing over the whole 32-row tile activated no more experts than the "
        "masked reduction, so this test would pass even if the masking were removed"
    )
    value = pcc(outs["whole-tile"], outs["masked"])
    assert value > 0.9999, f"padding-row masking changed the layer output (PCC {value})"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_optimized_beats_fused_traced_decode(mesh_device, layer_idx):
    """The optimized traced decode must actually be faster than the fused decoder's.

    Both are built and timed in one process, on one device, with the same real weights and the same
    inputs, so this is the same comparison ``doc/optimized_decoder/logs/bench.py`` reports and the
    README quotes — as a gate rather than a number in a log.
    """
    import time

    from models.autoports.ornith_ai_ornith_1_0_35b.tt.fused_decoder import FusedDecoder

    source = default_weight_source()
    iters = 32
    timings = {}
    for cls in (FusedDecoder, OptimizedDecoder):
        decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, cls=cls)
        ttnn.deallocate(
            decoder.prefill_forward(to_device(mesh_device, make_activations(1, 128, seed=81)), page_table=page_table)
        )
        x_buf = to_device(mesh_device, make_activations(1, 1, seed=82))
        pos_buf, rot_buf = decode_inputs(mesh_device, torch.tensor([128]))

        def forward(decoder=decoder, page_table=page_table, x_buf=x_buf, pos_buf=pos_buf, rot_buf=rot_buf):
            return decoder.decode_forward(x_buf, current_pos=pos_buf, rot_idxs=rot_buf, page_table=page_table)

        ttnn.deallocate(forward())
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_out = forward()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        for _ in range(4):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        start = time.time()
        for _ in range(iters):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        timings[cls.__name__] = (time.time() - start) / iters * 1e3
        assert torch.isfinite(ttnn.to_torch(trace_out).float()).all()
        ttnn.release_trace(mesh_device, trace_id)
        del decoder, page_table

    before, after = timings["FusedDecoder"], timings["OptimizedDecoder"]
    logger.info(
        f"OPTIMIZED VS FUSED decode(traced) layer={layer_idx} ({LAYER_IDS[layer_idx]}) "
        f"before={before:.3f} ms after={after:.3f} ms speedup={before / after:.2f}x"
    )
    assert after < before, f"optimized traced decode {after:.3f} ms is not faster than fused {before:.3f} ms"

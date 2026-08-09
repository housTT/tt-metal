# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused-decoder tests for Qwen/Qwen3.6-27B.

The whole functional suite is re-run against :class:`~..tt.fused_decoder.FusedDecoder` — the
same HF-reference PCC checks, the same unfriendly sequence lengths, the same paged-cache,
determinism, batched, traced-decode, block-size, BFP8 and no-host-fallback coverage — by
re-pointing ``harness.DECODER_CLS`` and importing the test bodies.  Nothing is re-implemented,
so the fused layer is held to *exactly* the bar the functional layer passed.

On top of that this module adds what is specific to a fused implementation:

* :func:`test_fused_graph_is_smaller` — the measured path really dispatches fewer **device**
  ops than the unfused one, counted with ``ttnn.graph``.  Without it a silent fallback to a
  functional-shaped graph would still pass every PCC test, and a Python-level check would not
  help: several ttnn helpers are composites that lower back to the sequence they replace.
* :func:`test_matches_functional_decoder` — fused vs unfused on identical weights and inputs,
  which is the equivalence the graph-fusing transform actually promises.
* :func:`test_repeated_prefill_decode_stress` — many back-to-back passes, checking the PCC
  does not drift and device memory does not grow.
"""

from __future__ import annotations

import pytest
import torch
import ttnn
from transformers.cache_utils import DynamicCache

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tests import test_functional_decoder as base
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder

LAYER_KINDS = base.LAYER_KINDS


@pytest.fixture(autouse=True)
def _use_fused_decoder():
    """Every test in this module builds the fused layer, not the functional one."""
    previous = H.DECODER_CLS
    H.DECODER_CLS = FusedDecoder
    yield
    H.DECODER_CLS = previous


# Re-collect the functional suite in this module.  The autouse fixture above is what makes the
# imported bodies build ``FusedDecoder``; the parametrisation, ids and markers come along with
# the function objects, so the two modules stay in lockstep by construction.
_INHERITED = sorted(name for name in vars(base) if name.startswith("test_"))
for _name in _INHERITED:
    globals()[_name] = getattr(base, _name)
del _name


def test_inherited_suite_is_complete():
    """Guard the re-export above: every functional test must run against the fused layer too."""
    assert len(_INHERITED) >= 14, f"only {len(_INHERITED)} functional tests were re-collected"
    for name in _INHERITED:
        assert globals()[name] is getattr(base, name)


# --------------------------------------------------------------- fused-path assertions

#: Device ops the fused graph must dispatch that the unfused one never does.  These are the
#: names ``ttnn.graph`` reports, i.e. the ops that actually reach the device — not the Python
#: helper that was called.  That distinction matters: ``ttnn.swiglu`` and
#: ``ttnn.linear(activation=...)`` are *composites* on this build and lower back to the very
#: sequences they look like they replace, so a Python-level spy would have happily "proved" a
#: fusion that never happened (see ``doc/fused_decoder/work_log.md`` §5).
FUSED_DEVICE_OPS = {
    "linear_attention": {"TernaryDeviceOperation"},          # F16, ttnn.addcmul in the causal conv
    "full_attention": {"RotaryEmbeddingHfDeviceOperation"},  # F2, the partial-RoPE fusion
}


def _device_ops(fn) -> list[str]:
    """Device-op names dispatched by ``fn``, in order, via ``ttnn.graph`` capture."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        fn()
    finally:
        graph = ttnn.graph.end_graph_capture()
    names = [
        node["params"].get("name", "")
        for node in graph
        if node.get("node_type") == "function_start"
    ]
    return [
        name
        for name in names
        if name.endswith("Operation") and "::" not in name and not name.startswith("ttnn.")
    ]


def _prefill_decode_ops(mesh_device, layer_idx, decoder_cls):
    """``(prefill_ops, decode_ops)`` for one decoder class, on identical inputs."""
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, decoder_cls=decoder_cls)
    stats = ref.load_weight_stats()
    hidden = ref.synthetic_hidden_states(lut.config, 1, 2049, stats)
    H.run_tt_prefill(lut, hidden)  # warm the program cache first
    prefill = _device_ops(lambda: H.run_tt_prefill(lut, hidden))
    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=31)
    H.run_tt_decode(lut, token, torch.tensor([2049]))
    decode = _device_ops(lambda: H.run_tt_decode(lut, token, torch.tensor([2049])))
    H.release_layers()
    return prefill, decode


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_fused_graph_is_smaller(mesh_device, layer_idx):
    """The fused layer dispatches strictly fewer **device** ops than the unfused one.

    This is the test that distinguishes "fused" from "functional with a different file name",
    and it counts what reaches the device rather than which Python helper was called — the
    two differ for every composite in ttnn.
    """
    kind = ref.load_text_config().layer_types[layer_idx]
    fused_prefill, fused_decode = _prefill_decode_ops(mesh_device, layer_idx, FusedDecoder)
    base_prefill, base_decode = _prefill_decode_ops(mesh_device, layer_idx, FunctionalDecoder)

    for phase, fused_ops, base_ops in (
        ("prefill", fused_prefill, base_prefill),
        ("decode", fused_decode, base_decode),
    ):
        H.record(f"device_ops_{phase}_functional", len(base_ops), kind=kind)
        H.record(f"device_ops_{phase}_fused", len(fused_ops), kind=kind)
        assert len(fused_ops) < len(base_ops), (
            f"{kind} {phase}: fused dispatches {len(fused_ops)} device ops, "
            f"unfused {len(base_ops)} - the rewrite removed nothing"
        )
        expected = FUSED_DEVICE_OPS[kind]
        assert expected <= set(fused_ops), f"{kind} {phase} missing {sorted(expected - set(fused_ops))}"
        assert not (expected & set(base_ops)), (
            f"{kind} {phase}: {sorted(expected & set(base_ops))} is not actually fusion-specific"
        )


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_matches_functional_decoder(mesh_device, layer_idx):
    """Fused vs unfused on identical weights: the equivalence the rewrite actually promises.

    HF PCC alone cannot tell a fused graph from a subtly different one that happens to stay
    inside the bar, so this compares the two implementations against *each other* — prefill
    output, decode output, and (``full_attention``) the paged K/V cache contents.
    """
    seq_len = 2049
    stats = ref.load_weight_stats()
    config = ref.load_text_config()
    kind = config.layer_types[layer_idx]
    hidden = ref.synthetic_hidden_states(config, 1, seq_len, stats)
    token = ref.synthetic_hidden_states(config, 1, 1, stats, seed=41)

    results = {}
    caches = {}
    for label, cls in (("functional", FunctionalDecoder), ("fused", FusedDecoder)):
        lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, decoder_cls=cls)
        prefill = H.run_tt_prefill(lut, hidden)
        H.prepare_decode(lut)
        decode = H.run_tt_decode(lut, token, torch.tensor([seq_len]))
        results[label] = (prefill, decode)
        if lut.is_full_attention:
            caches[label] = H.read_paged_kv(lut, user_id=0, seq_len=seq_len)
        H.release_layers()

    for index, what in enumerate(("prefill", "decode")):
        value = H.pcc(results["functional"][index], results["fused"][index])
        H.record(f"fused_vs_functional_{what}_pcc", value, kind=kind, seq_len=seq_len)
        assert value >= H.PCC_BAR, f"fused vs functional {what} PCC {value} < {H.PCC_BAR}"
    for index, what in enumerate(("k", "v")):
        if caches:
            value = H.pcc(caches["functional"][index], caches["fused"][index])
            H.record(f"fused_vs_functional_{what}_cache_pcc", value, kind=kind, seq_len=seq_len)
            assert value >= H.PCC_BAR, f"fused vs functional {what} cache PCC {value} < {H.PCC_BAR}"


#: Repeats for the stress test.  Enough to surface a leak (each pass allocates ~1 GB of
#: intermediates) or state that drifts between passes, without turning the suite into a soak.
STRESS_REPEATS = 12


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_repeated_prefill_decode_stress(mesh_device, layer_idx):
    """Many back-to-back prefill+decode passes: PCC stays put and DRAM does not grow.

    A fused graph that leaks a buffer, or whose L1-resident regions fragment, shows up here
    long before it shows up in a single-pass test.
    """
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
                f"{repeat}: the fused path leaks a buffer"
            )

    H.record("stress_prefill_pcc_min", min(prefill_values), kind=kind, repeats=STRESS_REPEATS)
    H.record("stress_decode_pcc_min", min(decode_values), kind=kind, repeats=STRESS_REPEATS)
    H.record("stress_prefill_bit_identical", len(set(prefill_values)) == 1, kind=kind)
    assert min(prefill_values) >= H.PCC_BAR, f"stress prefill PCC {min(prefill_values)}"
    assert min(decode_values) >= H.PCC_BAR, f"stress decode PCC {min(decode_values)}"
    assert len(set(prefill_values)) == 1, "repeated identical prefills disagreed"
    assert len(set(decode_values)) == 1, "repeated identical decodes disagreed"

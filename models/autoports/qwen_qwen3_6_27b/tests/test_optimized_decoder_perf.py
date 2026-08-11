# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill / traced-decode performance runs for the **optimized** Qwen3.6-27B decoder.

Same measured windows, same signposts and the same warm-up/profiler-drain discipline as
``test_fused_decoder_perf.py``, so a ``tt-perf-report`` table from this file is directly comparable
to the fused-stage one.  ``--impl`` selects the implementation and ``--policy`` / ``--geometry``
select a named configuration, so the whole before/after matrix is produced by one script:

    python -m tracy -r -p -v -m pytest \\
        models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder_perf.py::test_perf_decode_traced[linear_attention] \\
        -s -q --impl optimized --policy opt-v1 --geometry opt-v1 --perf-batch 1

* ``PERF_PREFILL`` / ``PERF_PREFILL_END`` - one warmed prefill of :data:`PERF_PREFILL_LEN` tokens;
  every input tensor is built and uploaded *before* the start signpost.
* ``PERF_DECODE`` / ``PERF_DECODE_END`` - :data:`PERF_DECODE_ITERS` replays of a captured decode
  trace with no host work at all inside the window.
"""

from __future__ import annotations

import dataclasses
import json
import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    BFP8_POLICY,
    DEFAULT_GEOMETRY,
    DEFAULT_POLICY,
    FUSED_BASELINE_GEOMETRY,
    FUSED_BASELINE_POLICY,
    OptimizedDecoder,
)

LAYER_KINDS = [
    pytest.param(H.LINEAR_LAYER_IDX, id="linear_attention"),
    pytest.param(H.FULL_LAYER_IDX, id="full_attention"),
]

IMPLS = {"optimized": OptimizedDecoder, "fused": FusedDecoder, "functional": FunctionalDecoder}

#: Named precision policies a perf run may select.  ``fused-baseline`` is the fused stage's policy
#: expressed in this stage's vocabulary, so ``--impl optimized --policy fused-baseline`` measures
#: the *same code* at the *old* precision - which is what isolates the precision change from the
#: layout change in the before/after tables.
POLICIES = {
    "opt-v1": DEFAULT_POLICY,
    "bfp8-all-lofi": BFP8_POLICY,
    "fused-baseline": FUSED_BASELINE_POLICY,
}

#: Named decode geometries.  ``fused-baseline`` restores the fused stage's interleaved decode
#: matmuls, DRAM residual and packed gate/up.
GEOMETRIES = {
    "opt-v1": DEFAULT_GEOMETRY,
    "fused-baseline": FUSED_BASELINE_GEOMETRY,
    "no-dram-sharding": dataclasses.replace(DEFAULT_GEOMETRY, dram_sharded=False),
    "dram-residual": dataclasses.replace(DEFAULT_GEOMETRY, sharded_stream=False),
    "packed-gate-up": dataclasses.replace(DEFAULT_GEOMETRY, split_gate_up_decode=False),
    "split-gate-up-prefill": dataclasses.replace(DEFAULT_GEOMETRY, split_gate_up_prefill=True),
}

#: Prefill length used for the perf window (one whole ``PREFILL_CHUNK``).
PERF_PREFILL_LEN = 2048
#: Absolute position the decode step runs at.
PERF_DECODE_POS = 2048
#: Trace replays inside the decode perf window.  Kept small on purpose: the profiler's per-core
#: DRAM marker buffer holds 12000 markers, so a larger count drops markers and silently truncates
#: the report.
PERF_DECODE_ITERS = 8


@pytest.fixture
def impl(request):
    return request.config.getoption("--impl")


@pytest.fixture
def decoder_kwargs(request, impl):
    """Extra ``from_state_dict`` keywords for the selected implementation."""
    if impl != "optimized":
        return {}
    policy_name = request.config.getoption("--policy")
    geometry_name = request.config.getoption("--geometry")
    assert policy_name in POLICIES, f"unknown --policy {policy_name!r}; known: {sorted(POLICIES)}"
    assert geometry_name in GEOMETRIES, f"unknown --geometry {geometry_name!r}; known: {sorted(GEOMETRIES)}"
    return {"policy": POLICIES[policy_name], "decode_geometry": GEOMETRIES[geometry_name]}


@pytest.fixture
def perf_batch(request):
    """Users in the decode window.

    Decode at ``max_batch`` 32 is not the batch-1 graph with a wider tensor: the z-gated norm
    switches to the group-reduction form and every recurrence op grows by 32x, so the
    advertised-batch path needs its own before/after pair rather than an extrapolation.
    """
    return int(request.config.getoption("--perf-batch"))


def _describe(lut, impl, decoder_kwargs) -> str:
    """One line naming exactly which configuration this window measured."""
    layer = lut.tt_layer
    payload = {"impl": impl}
    if hasattr(layer, "config_summary"):
        payload["config"] = layer.config_summary()
    return "PERFCONFIG " + json.dumps(payload, sort_keys=True, default=str)


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_perf_prefill(mesh_device, layer_idx, impl, decoder_kwargs):
    """Warmed prefill, signposted, with all host work outside the measured window."""
    lut = H.build_layer(
        mesh_device, layer_idx, max_batch=1, max_seq_len=8192, decoder_cls=IMPLS[impl], **decoder_kwargs
    )
    hidden = ref.synthetic_hidden_states(lut.config, 1, PERF_PREFILL_LEN, ref.load_weight_stats())

    tt_in = H.tt_hidden_prefill(hidden, mesh_device)
    rot = H.prefill_rot_mats(lut, PERF_PREFILL_LEN, mesh_device) if lut.is_full_attention else None
    full_pt, per_chunk = H.chunk_page_tables(lut, PERF_PREFILL_LEN, 0, mesh_device)

    def once():
        out = lut.tt_layer.prefill_forward(
            tt_in, user_id=0, page_table=full_pt, page_tables_per_chunk=per_chunk, rot_mats=rot
        )
        ttnn.deallocate(out)

    once()  # compile / warm up
    ttnn.synchronize_device(mesh_device)
    # Drain the device profiler's DRAM marker buffers before the measured pass; setup plus the
    # warm-up pass alone can overflow the 12000-marker-per-core buffer.
    ttnn.ReadDeviceProfiler(mesh_device)

    signpost("PERF_PREFILL")
    start = time.perf_counter()
    once()
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - start
    signpost("PERF_PREFILL_END")

    print(f"\n{_describe(lut, impl, decoder_kwargs)}")
    print(
        f"PERF prefill impl={impl} cls={IMPLS[impl].__name__} layer_idx={lut.layer_idx} "
        f"kind={lut.config.layer_types[lut.layer_idx]} seq_len={PERF_PREFILL_LEN} "
        f"wall_e2e_ms={elapsed * 1e3:.3f}"
    )


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_perf_decode_traced(mesh_device, layer_idx, impl, decoder_kwargs, perf_batch):
    """Warmed *traced* decode: capture once, then replay inside the signposted window."""
    lut = H.build_layer(
        mesh_device, layer_idx, max_batch=perf_batch, max_seq_len=8192, decoder_cls=IMPLS[impl], **decoder_kwargs
    )
    stats = ref.load_weight_stats()
    hidden = ref.synthetic_hidden_states(lut.config, 1, PERF_DECODE_POS, stats)
    H.run_tt_prefill(lut, hidden)
    H.prepare_decode(lut)
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    runner = H.TracedDecode(lut, batch=perf_batch)
    token = ref.synthetic_hidden_states(lut.config, perf_batch, 1, stats, seed=400)
    positions = torch.full((perf_batch,), PERF_DECODE_POS)
    runner.warmup(token, positions)
    runner.capture()
    runner.replay(token, positions)  # warm the replay path
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    signpost("PERF_DECODE")
    start = time.perf_counter()
    for _ in range(PERF_DECODE_ITERS):
        ttnn.execute_trace(mesh_device, runner.trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - start
    signpost("PERF_DECODE_END")

    print(f"\n{_describe(lut, impl, decoder_kwargs)}")
    print(
        f"PERF decode impl={impl} cls={IMPLS[impl].__name__} layer_idx={lut.layer_idx} "
        f"kind={lut.config.layer_types[lut.layer_idx]} batch={perf_batch} iters={PERF_DECODE_ITERS} "
        f"wall_total_ms={elapsed * 1e3:.3f} wall_per_iter_ms={elapsed * 1e3 / PERF_DECODE_ITERS:.3f}"
    )
    runner.release()

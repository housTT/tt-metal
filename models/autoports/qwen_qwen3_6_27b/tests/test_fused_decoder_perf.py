# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill / traced-decode performance runs for the **fused** Qwen3.6-27B decoder.

Same measured windows, same signposts and the same warm-up/profiler-drain discipline as
``test_functional_decoder_perf.py``, so a ``tt-perf-report`` table from this file is directly
comparable to the functional-stage one.  ``--impl`` selects which implementation runs, so the
before/after pair is produced by one script rather than two:

    python -m tracy -r -p -v -m pytest \\
        models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder_perf.py::test_perf_prefill[linear_attention] \\
        -s -q --impl fused

* ``PERF_PREFILL`` / ``PERF_PREFILL_END`` — one warmed prefill of :data:`PERF_PREFILL_LEN`
  tokens; every input tensor is built and uploaded *before* the start signpost.
* ``PERF_DECODE`` / ``PERF_DECODE_END`` — :data:`PERF_DECODE_ITERS` replays of a captured
  decode trace with no host work at all inside the window.
"""

from __future__ import annotations

import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder

LAYER_KINDS = [
    pytest.param(H.LINEAR_LAYER_IDX, id="linear_attention"),
    pytest.param(H.FULL_LAYER_IDX, id="full_attention"),
]

IMPLS = {"fused": FusedDecoder, "functional": FunctionalDecoder}

#: Prefill length used for the perf window (one whole ``PREFILL_CHUNK``).
PERF_PREFILL_LEN = 2048
#: Absolute position the decode step runs at.
PERF_DECODE_POS = 2048
#: Trace replays inside the decode perf window.  Kept small on purpose: the profiler's per-core
#: DRAM marker buffer holds 12000 markers, so a larger count drops markers and silently
#: truncates the report.
PERF_DECODE_ITERS = 8


@pytest.fixture
def decoder_cls(request):
    return IMPLS[request.config.getoption("--impl")]


@pytest.fixture
def perf_batch(request):
    """Users in the decode window.

    Decode at ``max_batch`` 32 is not the batch-1 graph with a wider tensor: the z-gated norm
    switches to the group-reduction form (:data:`~...tt.fused_decoder._GATED_NORM_GROUP_BATCH`)
    and every recurrence op grows by 32x, so the advertised-batch path needs its own before/after
    pair rather than an extrapolation of the batch-1 one.
    """
    return int(request.config.getoption("--perf-batch"))


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_perf_prefill(mesh_device, layer_idx, decoder_cls):
    """Warmed prefill, signposted, with all host work outside the measured window."""
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192, decoder_cls=decoder_cls)
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

    print(
        f"\nPERF prefill impl={decoder_cls.__name__} layer_idx={lut.layer_idx} "
        f"kind={lut.config.layer_types[lut.layer_idx]} seq_len={PERF_PREFILL_LEN} "
        f"wall_e2e_ms={elapsed * 1e3:.3f}"
    )


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_perf_decode_traced(mesh_device, layer_idx, decoder_cls, perf_batch):
    """Warmed *traced* decode: capture once, then replay inside the signposted window."""
    lut = H.build_layer(mesh_device, layer_idx, max_batch=perf_batch, max_seq_len=8192, decoder_cls=decoder_cls)
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

    print(
        f"\nPERF decode impl={decoder_cls.__name__} layer_idx={lut.layer_idx} "
        f"kind={lut.config.layer_types[lut.layer_idx]} batch={perf_batch} iters={PERF_DECODE_ITERS} "
        f"wall_total_ms={elapsed * 1e3:.3f} wall_per_iter_ms={elapsed * 1e3 / PERF_DECODE_ITERS:.3f}"
    )
    runner.release()

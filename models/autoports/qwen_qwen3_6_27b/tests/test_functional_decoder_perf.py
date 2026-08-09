# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill / traced-decode performance runs for the Qwen3.6-27B decoder layers.

These tests exist to produce Tracy device-profiler evidence, not to assert a latency
target.  Run them under the profiler, one layer kind at a time::

    python -m tracy -r -p -v -m pytest \\
        models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder_perf.py \\
        -k "prefill and linear_attention"

The measured window is delimited by Tracy signposts so ``tt-perf-report`` can filter it:

* ``PERF_PREFILL`` / ``PERF_PREFILL_END`` — one warmed prefill of :data:`PERF_PREFILL_LEN`
  tokens; every input tensor is built and uploaded *before* the start signpost.
* ``PERF_DECODE`` / ``PERF_DECODE_END`` — :data:`PERF_DECODE_ITERS` replays of a captured
  decode trace with no host work at all inside the window.
"""

from __future__ import annotations

import time

import pytest
import torch
import ttnn
from tracy import signpost

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H

LAYER_KINDS = [
    pytest.param(H.LINEAR_LAYER_IDX, id="linear_attention"),
    pytest.param(H.FULL_LAYER_IDX, id="full_attention"),
]

#: Prefill length used for the perf window (one whole :data:`~..tt.functional_decoder.PREFILL_CHUNK`).
PERF_PREFILL_LEN = 2048
#: Absolute position the decode step runs at.
PERF_DECODE_POS = 2048
#: Trace replays inside the decode perf window.  Kept small on purpose: a ``linear_attention``
#: decode step is ~55 device ops, and the profiler's per-core DRAM marker buffer holds 12000
#: markers, so a larger count drops markers ("Profiler DRAM buffers were full") and silently
#: truncates the report.
PERF_DECODE_ITERS = 8


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_perf_prefill(mesh_device, layer_idx):
    """Warmed prefill, signposted, with all host work outside the measured window."""
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
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
    # Drain the device profiler's DRAM marker buffers before the measured pass.  A whole
    # linear_attention prefill chunk is ~1e3 ops; setup plus the warm-up pass alone overflows
    # the 12000-marker-per-core buffer ("Profiler DRAM buffers were full, markers were
    # dropped!"), which makes post-processing drop ops and assert.
    ttnn.ReadDeviceProfiler(mesh_device)

    signpost("PERF_PREFILL")
    start = time.perf_counter()
    once()
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - start
    signpost("PERF_PREFILL_END")

    print(
        f"\nPERF prefill layer_idx={lut.layer_idx} kind={lut.config.layer_types[lut.layer_idx]} "
        f"seq_len={PERF_PREFILL_LEN} wall_e2e_ms={elapsed * 1e3:.3f}"
    )


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_perf_decode_traced(mesh_device, layer_idx):
    """Warmed *traced* decode: capture once, then replay inside the signposted window."""
    lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192)
    stats = ref.load_weight_stats()
    hidden = ref.synthetic_hidden_states(lut.config, 1, PERF_DECODE_POS, stats)
    H.run_tt_prefill(lut, hidden)
    H.prepare_decode(lut)

    runner = H.TracedDecode(lut, batch=1)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=400)
    positions = torch.tensor([PERF_DECODE_POS])
    runner.warmup(token, positions)
    runner.capture()
    runner.replay(token, positions)  # warm the replay path
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)  # see the note in test_perf_prefill

    signpost("PERF_DECODE")
    start = time.perf_counter()
    for _ in range(PERF_DECODE_ITERS):
        ttnn.execute_trace(mesh_device, runner.trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - start
    signpost("PERF_DECODE_END")

    print(
        f"\nPERF decode layer_idx={lut.layer_idx} kind={lut.config.layer_types[lut.layer_idx]} "
        f"iters={PERF_DECODE_ITERS} wall_total_ms={elapsed * 1e3:.3f} "
        f"wall_per_iter_ms={elapsed * 1e3 / PERF_DECODE_ITERS:.3f}"
    )
    runner.release()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profile the same real 2048-token input and decode position as paired timing."""

import json
import os
import time

import pytest
import torch
from tracy import signpost

import ttnn

from ..tt.fused_decoder import FusedDecoder
from . import test_functional_decoder as H
from .optimization_candidates import selected_candidate
from .test_optimization_experiments import recorded_activations

pytestmark = H.pytestmark


@pytest.mark.parametrize("layer_idx", H.LAYERS, ids=lambda i: H.LAYER_IDS[i])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_profile(mesh_device, monkeypatch, layer_idx, mode):
    baseline = os.environ.get("ORNITH_PROFILE_BASELINE", "0") == "1"
    monkeypatch.setattr(H, "FunctionalDecoder", FusedDecoder if baseline else selected_candidate())
    decoder, table, _ = H.build_decoder(mesh_device, layer_idx, "real")
    recorded = recorded_activations(layer_idx)
    prompt, token = recorded[:, :2048], recorded[:, 2048:2049]
    x, d = H.to_device(mesh_device, prompt), H.to_device(mesh_device, token)
    for _ in range(2):
        decoder.reset_state()
        ttnn.deallocate(decoder.prefill_forward(x, page_table=table))
    trace = None
    if mode == "prefill":
        decoder.reset_state()
        iterations = 1

        def forward():
            return decoder.prefill_forward(x, page_table=table)

    else:
        saved = H._snapshot_state(decoder)
        pos, rot = H.decode_inputs(mesh_device, torch.tensor([2048]))

        def forward():
            return decoder.decode_forward(d, current_pos=pos, rot_idxs=rot, page_table=table)

        ttnn.deallocate(forward())
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = forward()
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        for _ in range(4):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        H._restore_state(decoder, saved)
        iterations = int(os.environ.get("ORNITH_PERF_DECODE_ITERS", "4"))
    ttnn.synchronize_device(mesh_device)
    # Drain setup/warmup outside the measured window; verified collector fix.
    ttnn.ReadDeviceProfiler(mesh_device)
    signpost(f"PERF_{mode.upper()}")
    start = time.perf_counter()
    if mode == "prefill":
        out = forward()
    else:
        for _ in range(iterations):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    elapsed = (time.perf_counter() - start) * 1000 / iterations
    signpost(f"PERF_{mode.upper()}_END")
    ttnn.ReadDeviceProfiler(mesh_device)
    actual = ttnn.to_torch(out)
    steps = 1 if decoder.is_full_attention else iterations
    golden, decoded = H.run_reference(layer_idx, "real", prompt, decode_x=[token] * steps, decode_steps=steps)
    value = H.pcc(golden if mode == "prefill" else decoded[-1], actual)
    print(
        "PROFILE_MEASUREMENT "
        + json.dumps(
            dict(
                layer=layer_idx,
                mode=mode,
                baseline=baseline,
                sequence_length=2048 if mode == "prefill" else 1,
                decode_position=2048,
                iterations=iterations,
                host_wall_per_iteration_ms=elapsed,
                hf_pcc=value,
                activation_source="recorded pinned-HF layer input",
            )
        ),
        flush=True,
    )
    if trace is not None:
        ttnn.release_trace(mesh_device, trace)
    assert value >= H.PCC_BAR

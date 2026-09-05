# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Same-shape prefill tracing control for the final profiler's dispatch advice."""

import json
import statistics
import time

import pytest
import torch

import ttnn

from ..tt.fused_decoder import FusedDecoder
from ..tt.optimized_decoder import OptimizedDecoder
from . import test_functional_decoder as H
from .test_optimization_experiments import recorded_activations
from .test_optimized_trace_regression import all_equal, differences

pytestmark = H.pytestmark


@pytest.mark.parametrize("layer_idx", H.LAYERS, ids=lambda i: H.LAYER_IDS[i])
def test_prefill_trace_control(mesh_device, monkeypatch, layer_idx):
    prompt = recorded_activations(layer_idx)[:, :2048]
    golden, _ = H.run_reference(layer_idx, "real", prompt, decode_steps=0)
    refreshed_prompt = recorded_activations(layer_idx)[:, 1:2049]
    refreshed_golden, _ = H.run_reference(layer_idx, "real", refreshed_prompt, decode_steps=0)
    results = {}
    for name, cls in (("fused", FusedDecoder), ("optimized", OptimizedDecoder)):
        with monkeypatch.context() as patch:
            patch.setattr(H, "FunctionalDecoder", cls)
            decoder, table, _ = H.build_decoder(mesh_device, layer_idx, "real")
        x = H.to_device(mesh_device, prompt)
        eager_times = []
        for iteration in range(7):
            decoder.reset_state()
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            out = decoder.prefill_forward(x, page_table=table)
            ttnn.synchronize_device(mesh_device)
            elapsed = (time.perf_counter() - start) * 1000
            if iteration >= 2:
                eager_times.append(elapsed)
            if iteration == 6:
                expected = ttnn.to_torch(out)
                expected_state = H._snapshot_state(decoder)
            ttnn.deallocate(out)
        decoder.reset_state()
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = decoder.prefill_forward(x, page_table=table)
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        try:
            trace_times = []
            for iteration in range(7):
                decoder.reset_state()
                ttnn.synchronize_device(mesh_device)
                start = time.perf_counter()
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                elapsed = (time.perf_counter() - start) * 1000
                if iteration >= 2:
                    trace_times.append(elapsed)
            actual = ttnn.to_torch(out)
            state_check = differences(expected_state, H._snapshot_state(decoder))
            assert torch.equal(expected, actual)
            assert all_equal(state_check), state_check
            assert torch.equal(ttnn.to_torch(x), prompt.to(torch.bfloat16))
            value = H.pcc(golden, actual)
            assert value >= H.PCC_BAR
            # Reuse the captured input address with a different real prompt.
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(refreshed_prompt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), x
            )
            decoder.reset_state()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            refreshed = ttnn.to_torch(out)
            refreshed_state = H._snapshot_state(decoder)
            decoder.reset_state()
            eager = decoder.prefill_forward(x, page_table=table)
            assert torch.equal(refreshed, ttnn.to_torch(eager))
            assert all_equal(differences(refreshed_state, H._snapshot_state(decoder)))
            assert torch.equal(ttnn.to_torch(x), refreshed_prompt.to(torch.bfloat16))
            refreshed_pcc = H.pcc(refreshed_golden, refreshed)
            assert refreshed_pcc >= H.PCC_BAR
            ttnn.deallocate(eager)
            results[name] = dict(
                eager_prefill_ms=eager_times,
                traced_prefill_ms=trace_times,
                eager_median_ms=statistics.median(eager_times),
                traced_median_ms=statistics.median(trace_times),
                hf_pcc=value,
                exact_output=True,
                exact_state=state_check,
                immutable_input=True,
                refreshed_input_hf_pcc=refreshed_pcc,
                refreshed_input_exact_output_and_state=True,
            )
        finally:
            ttnn.release_trace(mesh_device, trace)
    print("PREFILL_TRACE_CONTROL " + json.dumps(dict(layer=layer_idx, sequence_length=2048, results=results)))

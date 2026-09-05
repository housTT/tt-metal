# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape TP4 prefill trace with refreshed inputs and all-rank state checks."""
import json
import statistics
import time

import pytest
import torch

import ttnn

from . import test_functional_decoder as H
from .test_multichip_decoder import multichip_contract, pytestmark, state_buffers  # noqa: F401
from .test_optimization_experiments import recorded_activations


def read_ranks(value):
    return [ttnn.to_torch(part) for part in ttnn.get_device_tensors(value)]


def read_state(decoder):
    return [read_ranks(value) for value in state_buffers(decoder)]


def equal_state(first, second):
    return all(torch.equal(a, b) for left, right in zip(first, second) for a, b in zip(left, right))


@pytest.mark.parametrize("layer_idx", H.LAYERS, ids=lambda i: H.LAYER_IDS[i])
def test_multichip_prefill_trace(mesh_device, layer_idx):
    prompt = recorded_activations(layer_idx)[:, :2048]
    refreshed = recorded_activations(layer_idx)[:, 1:2049]
    golden, _ = H.run_reference(layer_idx, "real", prompt, decode_steps=0)
    new_golden, _ = H.run_reference(layer_idx, "real", refreshed, decode_steps=0)
    decoder, table, _ = H.build_decoder(mesh_device, layer_idx, "real")
    x = H.to_device(mesh_device, prompt)
    eager_times = []
    for iteration in range(7):
        decoder.reset_state()
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        out = decoder.prefill_forward(x, page_table=table)
        ttnn.synchronize_device(mesh_device)
        if iteration >= 2:
            eager_times.append((time.perf_counter() - start) * 1000)
        if iteration == 6:
            expected, saved = read_ranks(out), read_state(decoder)
        ttnn.deallocate(out)
    decoder.reset_state()
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = decoder.prefill_forward(x, page_table=table)
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        traced_times = []
        for iteration in range(7):
            decoder.reset_state()
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            if iteration >= 2:
                traced_times.append((time.perf_counter() - start) * 1000)
        assert all(torch.equal(a, b) for a, b in zip(expected, read_ranks(out)))
        assert equal_state(saved, read_state(decoder))
        assert all(torch.equal(part, prompt.to(torch.bfloat16)) for part in read_ranks(x))
        scores = [H.pcc(golden, part) for part in read_ranks(out)]
        assert min(scores) >= H.PCC_BAR
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(refreshed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), x)
        decoder.reset_state()
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        changed, changed_state = read_ranks(out), read_state(decoder)
        decoder.reset_state()
        eager = decoder.prefill_forward(x, page_table=table)
        assert all(torch.equal(a, b) for a, b in zip(changed, read_ranks(eager)))
        assert equal_state(changed_state, read_state(decoder))
        new_scores = [H.pcc(new_golden, part) for part in changed]
        assert min(new_scores) >= H.PCC_BAR
        assert all(torch.equal(part, refreshed.to(torch.bfloat16)) for part in read_ranks(x))
        ttnn.deallocate(eager)
        print(
            "MULTICHIP_PREFILL_TRACE "
            + json.dumps(
                dict(
                    layer=layer_idx,
                    mesh=[1, 4],
                    logical_shape=list(prompt.shape),
                    eager_prefill_ms=statistics.median(eager_times),
                    traced_prefill_ms=statistics.median(traced_times),
                    eager_samples_ms=eager_times,
                    traced_samples_ms=traced_times,
                    hf_pcc=scores,
                    refreshed_hf_pcc=new_scores,
                    exact_output=True,
                    exact_all_rank_state=True,
                    immutable_inputs=True,
                    refreshed_input_exact=True,
                )
            ),
            flush=True,
        )
    finally:
        ttnn.release_trace(mesh_device, trace)
        ttnn.deallocate(out)

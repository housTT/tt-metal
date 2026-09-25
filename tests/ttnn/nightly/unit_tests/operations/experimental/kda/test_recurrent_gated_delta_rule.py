# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct correctness and trace coverage for the T=1 KDA recurrence."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576, "trace_region_size": 16_000_000}),
]


def _inputs(seed: int = 721):
    generator = torch.Generator().manual_seed(seed)
    batch, heads, key_dim, value_dim = 1, 48, 128, 128
    query = torch.randn(batch, heads, 1, key_dim, generator=generator)
    key = torch.randn(batch, heads, 1, key_dim, generator=generator)
    value = torch.randn(batch, heads, 1, value_dim, generator=generator)
    beta = torch.sigmoid(torch.randn(batch, heads, 1, 1, generator=generator))
    log_decay = -torch.rand(batch, heads, 1, 1, generator=generator)
    state = torch.randn(batch, heads, key_dim, value_dim, generator=generator) * 0.02
    return query, key, value, beta, log_decay, state


def _reference(inputs):
    query, key, value, beta, log_decay, state = inputs
    decayed = state * torch.exp(log_decay)
    delta = (value - key @ decayed) * beta
    updated = decayed + key.transpose(-2, -1) @ delta
    return query @ updated, updated


def _upload(device, tensor):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _pcc(expected, actual):
    return torch.corrcoef(torch.stack((expected.float().flatten(), actual.float().flatten())))[0, 1].item()


def _hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def test_recurrent_gated_delta_rule_is_accurate_and_traceable(device: ttnn.Device) -> None:
    host = _inputs()
    expected_core, expected_state = _reference(host)
    tensors = tuple(_upload(device, value) for value in host)

    core, state = ttnn.experimental.kda.recurrent_gated_delta_rule(*tensors, compute_kernel_config=_hifi4())
    actual_core = ttnn.to_torch(core)
    actual_state = ttnn.to_torch(state)
    core_pcc = _pcc(expected_core, actual_core)
    state_pcc = _pcc(expected_state, actual_state)
    head_pcc = min(_pcc(expected_core[:, head], actual_core[:, head]) for head in range(expected_core.shape[1]))
    print(f"KDA_RECURRENT_PCC core={core_pcc:.8f} state={state_pcc:.8f} min_head={head_pcc:.8f}")
    assert core_pcc >= 0.999
    assert state_pcc >= 0.999
    assert head_pcc >= 0.999

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced_core, traced_state = ttnn.experimental.kda.recurrent_gated_delta_rule(
        *tensors, compute_kernel_config=_hifi4()
    )
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    assert _pcc(expected_core, ttnn.to_torch(traced_core)) >= 0.999
    assert _pcc(expected_state, ttnn.to_torch(traced_state)) >= 0.999
    ttnn.release_trace(device, trace_id)


def test_recurrent_gated_delta_rule_updates_preallocated_state_in_place(device: ttnn.Device) -> None:
    host = _inputs(seed=722)
    expected_core, expected_state = _reference(host)
    tensors = tuple(_upload(device, value) for value in host)
    persistent_state = tensors[-1]
    initial_state = _upload(device, host[-1])
    address = persistent_state.buffer_address()

    # Compile once, then restore the input before both trace capture and replay;
    # an aliased state is intentionally mutated by every execution.
    ttnn.experimental.kda.recurrent_gated_delta_rule(
        *tensors,
        state_output=persistent_state,
        compute_kernel_config=_hifi4(),
    )
    ttnn.copy(initial_state, persistent_state)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    core, state = ttnn.experimental.kda.recurrent_gated_delta_rule(
        *tensors,
        state_output=persistent_state,
        compute_kernel_config=_hifi4(),
    )
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.copy(initial_state, persistent_state)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

    assert state.buffer_address() == address
    assert _pcc(expected_core, ttnn.to_torch(core)) >= 0.999
    assert _pcc(expected_state, ttnn.to_torch(state)) >= 0.999
    ttnn.release_trace(device, trace_id)

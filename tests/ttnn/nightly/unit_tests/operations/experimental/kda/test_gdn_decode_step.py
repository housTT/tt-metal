# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness of the fused T=1 gated-delta-net decode step."""

from __future__ import annotations

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576, "trace_region_size": 16_000_000}),
]

QK_EPS = 1e-6
NORM_EPS = 1e-6


def _inputs(*, batch: int, heads: int, key_heads: int, key_dim: int, value_dim: int, seed: int = 731):
    generator = torch.Generator().manual_seed(seed)
    width = 2 * key_heads * key_dim + heads * value_dim
    x = torch.randn(batch, 1, 1, width, generator=generator)
    taps = [torch.randn(batch, 1, 1, width, generator=generator) for _ in range(3)]
    weights = [torch.randn(1, 1, 1, width, generator=generator) * 0.5 for _ in range(4)]
    beta = torch.sigmoid(torch.randn(batch, heads, 1, 1, generator=generator))
    log_decay = -torch.rand(batch, heads, 1, 1, generator=generator)
    state = torch.randn(batch, heads, key_dim, value_dim, generator=generator) * 0.02
    gate = torch.randn(batch, 1, 1, heads * value_dim, generator=generator)
    norm_weight = 1.0 + 0.1 * torch.randn(1, 1, 1, value_dim, generator=generator)
    return x, taps, weights, beta, log_decay, state, gate, norm_weight


def _reference(inputs, *, heads: int, key_heads: int, key_dim: int, value_dim: int):
    x, taps, weights, beta, log_decay, state, gate, norm_weight = inputs
    conv = taps[0] * weights[0] + taps[1] * weights[1] + taps[2] * weights[2] + x * weights[3]
    conv = torch.nn.functional.silu(conv)
    batch = x.shape[0]
    qk = key_heads * key_dim
    q = conv[..., :qk].reshape(batch, key_heads, 1, key_dim)
    k = conv[..., qk : 2 * qk].reshape(batch, key_heads, 1, key_dim)
    v = conv[..., 2 * qk :].reshape(batch, heads, 1, value_dim)
    repeat = heads // key_heads
    q = q.repeat_interleave(repeat, dim=1)
    k = k.repeat_interleave(repeat, dim=1)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + QK_EPS) / math.sqrt(key_dim)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + QK_EPS)
    decayed = state * torch.exp(log_decay)
    delta = (v - k @ decayed) * beta
    updated = decayed + k.transpose(-2, -1) @ delta
    core = q @ updated  # [B, H, 1, V]
    o = core.reshape(batch, heads, value_dim)
    o = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + NORM_EPS) * norm_weight.reshape(1, 1, value_dim)
    out = o * torch.sigmoid(gate.reshape(batch, heads, value_dim))
    return out.reshape(batch, 1, 1, heads * value_dim), updated


def _upload(device, tensor, dtype=ttnn.float32):
    return ttnn.from_torch(
        tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _pcc(expected, actual):
    return torch.corrcoef(torch.stack((expected.float().flatten(), actual.float().flatten())))[0, 1].item()


def _hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


@pytest.mark.parametrize("heads,key_heads", [(12, 4), (48, 16)])
def test_gdn_decode_step_matches_reference(device: ttnn.Device, heads: int, key_heads: int) -> None:
    key_dim = value_dim = 128
    host = _inputs(batch=1, heads=heads, key_heads=key_heads, key_dim=key_dim, value_dim=value_dim)
    expected_out, expected_state = _reference(host, heads=heads, key_heads=key_heads, key_dim=key_dim, value_dim=value_dim)
    x, taps, weights, beta, log_decay, state, gate, norm_weight = host
    dev = dict(
        x=_upload(device, x),
        taps=[_upload(device, t) for t in taps],
        weights=[_upload(device, w) for w in weights],
        beta=_upload(device, beta),
        log_decay=_upload(device, log_decay),
        state=_upload(device, state),
        gate=_upload(device, gate, ttnn.bfloat16),
        norm_weight=_upload(device, norm_weight, ttnn.bfloat16),
    )
    out, new_state = ttnn.experimental.kda.gdn_decode_step(
        dev["x"],
        *dev["taps"],
        *dev["weights"],
        dev["beta"],
        dev["log_decay"],
        dev["state"],
        dev["gate"],
        dev["norm_weight"],
        heads,
        key_dim,
        value_dim,
        state_output=dev["state"],
        qk_head_repeat=heads // key_heads,
        qk_norm_epsilon=QK_EPS,
        norm_epsilon=NORM_EPS,
        output_dtype=ttnn.bfloat16,
        compute_kernel_config=_hifi4(),
    )
    actual_out = ttnn.to_torch(out).float()
    actual_state = ttnn.to_torch(new_state).float()
    assert new_state.buffer_address() == dev["state"].buffer_address()
    out_pcc = _pcc(expected_out, actual_out)
    state_pcc = _pcc(expected_state, actual_state)
    head_pcc = min(
        _pcc(expected_out[..., h * value_dim : (h + 1) * value_dim], actual_out[..., h * value_dim : (h + 1) * value_dim])
        for h in range(heads)
    )
    max_abs = (expected_out - actual_out).abs().max().item()
    print(f"GDN_DECODE_STEP_PCC out={out_pcc:.8f} state={state_pcc:.8f} min_head={head_pcc:.8f} max_abs={max_abs:.5f}")
    assert out_pcc >= 0.999, out_pcc
    assert state_pcc >= 0.999, state_pcc
    assert head_pcc >= 0.998, head_pcc
    assert torch.isfinite(actual_out).all()

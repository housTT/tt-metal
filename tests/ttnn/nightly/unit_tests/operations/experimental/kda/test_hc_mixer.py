# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness of the fused hyper-connection mixer tail and injection."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576, "trace_region_size": 16_000_000}),
]


def _upload(device, tensor, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _pcc(expected, actual):
    return torch.corrcoef(torch.stack((expected.float().flatten(), actual.float().flatten())))[0, 1].item()


def _hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


def test_hc_mix_post_matches_reference(device: ttnn.Device) -> None:
    streams, lowrank, width = 4, 320, 640
    generator = torch.Generator().manual_seed(811)
    packed = torch.randn(1, 1, 1, lowrank + streams, generator=generator).bfloat16()
    weighted = torch.randn(1, 1, streams, width, generator=generator).bfloat16()
    up = (torch.randn(1, 1, lowrank, streams * width, generator=generator) * 0.05).bfloat16()

    low = packed[..., :lowrank].float()
    inject = packed[..., lowrank:].float()
    mix = torch.sigmoid(torch.nn.functional.silu(low) @ up.float())  # [1,1,1,S*N]
    mix = mix.reshape(streams, width)
    expected = (weighted.float().reshape(streams, width) * mix).mean(0).reshape(1, 1, 1, width)

    mixed, injection = ttnn.experimental.kda.hc_mix_post(
        _upload(device, packed), _upload(device, weighted), _upload(device, up), lowrank, streams, compute_kernel_config=_hifi4()
    )
    actual = ttnn.to_torch(mixed).float()
    actual_inject = ttnn.to_torch(injection).float()
    pcc = _pcc(expected, actual)
    max_abs = (expected - actual).abs().max().item()
    print(f"HC_MIX_POST_PCC mixed={pcc:.8f} max_abs={max_abs:.5f} inject_ok={torch.equal(actual_inject, inject)}")
    assert list(mixed.shape) == [1, 1, 1, width]
    assert list(injection.shape) == [1, 1, 1, streams]
    assert pcc >= 0.999, pcc
    assert torch.equal(actual_inject, inject)


@pytest.mark.parametrize("ranks", [1, 4])
@pytest.mark.parametrize("batch_gather", [False, True])
def test_hc_mix_post_stream_blocked_input(device: ttnn.Device, ranks: int, batch_gather: bool) -> None:
    """Stream-blocked partial rows: [1,1,R*S, S*Lp], row r*S+s of block s belongs to stream s."""

    streams, lowrank, width = 4, 320, 640
    padded = 352
    generator = torch.Generator().manual_seed(813 + ranks)
    weighted = torch.randn(1, 1, streams, width, generator=generator).bfloat16()
    down = (torch.randn(1, 1, streams * width, lowrank + streams, generator=generator) * 0.05).bfloat16()
    up = (torch.randn(1, 1, lowrank, streams * width, generator=generator) * 0.05).bfloat16()
    # rank partials: split the K dim of each stream's down block over ranks
    full = torch.zeros(ranks, streams, streams, padded)
    per_rank = width // ranks
    for r in range(ranks):
        for s in range(streams):
            w_block = down.float()[0, 0, s * width : (s + 1) * width][r * per_rank : (r + 1) * per_rank]  # [K_r, L+S]
            x_rows = weighted.float()[0, 0, :, r * per_rank : (r + 1) * per_rank]  # [S rows, K_r]
            full[r, :, s, : lowrank + streams] = x_rows @ w_block  # all rows; only row s is used
    if batch_gather:
        # gathered over ranks along the batch dim: [R, 1, S, S*Lp]
        packed = full.reshape(ranks, 1, streams, streams * padded).bfloat16()
    else:
        # gathered over ranks along rows: row index = r*S + row
        blocked = full.reshape(ranks * streams, streams, padded)
        packed = blocked.reshape(1, 1, ranks * streams, streams * padded).bfloat16()
    # reference from the exact (unsplit) product
    flat = weighted.float().reshape(1, streams * width)
    packed_ref = flat @ down.float()[0, 0]  # [1, L+S]
    low = packed_ref[:, :lowrank]
    inject = packed_ref[:, lowrank:]
    mix = torch.sigmoid(torch.nn.functional.silu(low) @ up.float()[0, 0]).reshape(streams, width)
    expected = (weighted.float().reshape(streams, width) * mix).mean(0).reshape(1, 1, 1, width)

    mixed, injection = ttnn.experimental.kda.hc_mix_post(
        _upload(device, packed), _upload(device, weighted), _upload(device, up), lowrank, streams, compute_kernel_config=_hifi4()
    )
    actual = ttnn.to_torch(mixed).float()
    actual_inject = ttnn.to_torch(injection).float().reshape(-1)
    pcc = _pcc(expected, actual)
    inj_err = (actual_inject - inject.reshape(-1)).abs().max().item()
    print(f"HC_MIX_POST_PCC blocked ranks={ranks} batch_gather={batch_gather} mixed={pcc:.8f} inject_err={inj_err:.4f}")
    assert pcc >= 0.999, pcc
    assert inj_err < 0.1 + 0.02 * inject.abs().max().item(), (actual_inject, inject)


def test_hc_inject_matches_reference(device: ttnn.Device) -> None:
    streams, width = 4, 640
    generator = torch.Generator().manual_seed(812)
    hyper = torch.randn(1, 1, streams, width, generator=generator).bfloat16()
    block = torch.randn(1, 1, 1, width, generator=generator).bfloat16()
    injection = torch.randn(1, 1, 1, streams, generator=generator).bfloat16()
    gate = 2.0 * torch.sigmoid(injection.float()).reshape(streams, 1)
    expected = hyper.float() + (gate * block.float().reshape(1, width)).reshape(1, 1, streams, width)

    out = ttnn.experimental.kda.hc_inject(
        _upload(device, hyper), _upload(device, block), _upload(device, injection), streams, compute_kernel_config=_hifi4()
    )
    actual = ttnn.to_torch(out).float()
    pcc = _pcc(expected, actual)
    max_abs = (expected - actual).abs().max().item()
    print(f"HC_INJECT_PCC out={pcc:.8f} max_abs={max_abs:.5f}")
    assert list(out.shape) == [1, 1, streams, width]
    assert pcc >= 0.999, pcc
    assert max_abs < 0.05, max_abs

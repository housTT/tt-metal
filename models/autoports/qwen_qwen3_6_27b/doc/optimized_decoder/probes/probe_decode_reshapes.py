# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Is the 16 us `core` reshape in the gated-delta-net decode worth paying?

`_linear_attention_decode` produces the delta-rule output as ``[1, batch*nv, 1, head_v_dim]``
- one *row* per (user, value head) - and then reshapes it to ``[1, batch, nv, head_v_dim]``
before the gated RMSNorm.  That reshape is the single largest layout op in the decode profile
(16.0 us of a 44.6 us `ReshapeView` total, `tracy/linear_attention/decode_perf_report.csv`).

The norm itself does not need it: RMSNorm reduces over the last dimension, and both shapes have
``head_v_dim`` last with one logical row per (user, head).  So the reshape looks removable, and
the question is whether removing it is faster.  It should not be - a tiled ``Y = 1`` axis is
padded to 32 rows, so the un-reshaped tensor is ``batch*nv`` tile-rows of 1/32 occupancy against
the reshaped tensor's ``ceil(batch*nv / 32)`` dense ones - but "should not be" is not a
measurement, so this times both, model-free, at the real shapes.
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")

import torch  # noqa: E402
import ttnn  # noqa: E402

BATCH, NV, HEAD_V = 1, 48, 128
REPEATS = 200


def pcc(a, b):
    a = a.to(torch.float64).flatten() - a.to(torch.float64).mean()
    b = b.to(torch.float64).flatten() - b.to(torch.float64).mean()
    return float((a @ b) / (a.norm() * b.norm()))


def timed(fn, device):
    fn()
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(REPEATS):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - start) * 1e6 / REPEATS


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        torch.manual_seed(0)
        flat = torch.randn(1, BATCH * NV, 1, HEAD_V)
        weight = torch.randn(HEAD_V).abs()
        golden = torch.nn.functional.rms_norm(flat, (HEAD_V,), weight, eps=1e-6)

        w_tt = ttnn.from_torch(weight.reshape(1, 1, 1, HEAD_V), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=device)
        cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                               math_approx_mode=False, fp32_dest_acc_en=True,
                                               packer_l1_acc=True)

        def fresh():
            return ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

        # A: shipped - reshape [1, b*nv, 1, D] -> [1, b, nv, D], then norm the dense tensor.
        def shipped():
            x = fresh()
            core = ttnn.reshape(x, (1, BATCH, NV, HEAD_V))
            out = ttnn.rms_norm(core, weight=w_tt, epsilon=1e-6, compute_kernel_config=cfg)
            ttnn.deallocate(out)

        # B: candidate - norm the [1, b*nv, 1, D] tensor directly, no reshape.
        def no_reshape():
            x = fresh()
            out = ttnn.rms_norm(x, weight=w_tt, epsilon=1e-6, compute_kernel_config=cfg)
            ttnn.deallocate(out)

        x = fresh()
        core = ttnn.reshape(x, (1, BATCH, NV, HEAD_V))
        a = ttnn.to_torch(ttnn.rms_norm(core, weight=w_tt, epsilon=1e-6,
                                        compute_kernel_config=cfg)).float()
        b = ttnn.to_torch(ttnn.rms_norm(fresh(), weight=w_tt, epsilon=1e-6,
                                        compute_kernel_config=cfg)).float()
        row = {
            "shape_shipped": f"[1, {BATCH}, {NV}, {HEAD_V}] after reshape",
            "shape_candidate": f"[1, {BATCH * NV}, 1, {HEAD_V}] as produced",
            "pcc_shipped_vs_torch": pcc(golden.reshape(-1), a.reshape(-1)),
            "pcc_candidate_vs_torch": pcc(golden.reshape(-1), b.reshape(-1)),
            "us_reshape_plus_norm": timed(shipped, device),
            "us_norm_only": timed(no_reshape, device),
        }
        row["us_saved_by_removing_reshape"] = row["us_reshape_plus_norm"] - row["us_norm_only"]
        print("RESHAPE " + json.dumps(row), flush=True)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

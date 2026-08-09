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
    """Median-of-5 batches of ``REPEATS`` calls, with the spread, in microseconds per call.

    The differences here are single-digit microseconds, so a single mean is not enough for a
    reader to tell a result from noise; the spread is reported alongside it.
    """
    fn()
    ttnn.synchronize_device(device)
    batches = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(REPEATS):
            fn()
        ttnn.synchronize_device(device)
        batches.append((time.perf_counter() - start) * 1e6 / REPEATS)
    batches.sort()
    return batches[2], batches[-1] - batches[0]


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

        # The input is uploaded once and reused: the upload is ~280 us and would swamp a
        # single-digit-microsecond difference if it sat inside the timed loop.
        src = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        #: The delta rule's own output dtype, which is what the typecast consumes.
        src32 = ttnn.from_torch(flat, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        flat_shape = (1, 1, BATCH, NV * HEAD_V)

        # A: shipped - de-pad to [1, b, nv, D], typecast, norm the dense tensor, then flatten.
        # The typecast matters and an earlier revision of this probe left it out: the delta rule
        # produces float32 and the gated norm runs in bfloat16, so `_linear_attention_decode`
        # casts *between* the reshape and the norm.  On the un-reshaped tensor that cast would
        # run on batch*nv tile-rows of 1/32 occupancy instead of ceil(batch*nv/32) dense ones,
        # which is exactly the asymmetry the reshape exists to create.
        def shipped():
            core = ttnn.reshape(src32, (1, BATCH, NV, HEAD_V))
            cast = ttnn.typecast(core, ttnn.bfloat16)
            out = ttnn.rms_norm(cast, weight=w_tt, epsilon=1e-6, compute_kernel_config=cfg)
            flat_out = ttnn.reshape(out, flat_shape)
            ttnn.deallocate(flat_out)

        # B: norm the [1, b*nv, 1, D] tensor as produced, no reshape at all.  Not a candidate
        # for the layer - the output projection needs the flat shape - but it isolates what the
        # norm itself costs on the padded tensor.
        def norm_only():
            out = ttnn.rms_norm(src, weight=w_tt, epsilon=1e-6, compute_kernel_config=cfg)
            ttnn.deallocate(out)

        # C: the real alternative - norm first on the padded tensor, then one reshape straight
        # to the flat shape the output projection wants.  This is the arrangement the shipped
        # order is being compared against, and the fifth review round was right that it was
        # never timed.
        def norm_then_flatten():
            cast = ttnn.typecast(src32, ttnn.bfloat16)
            out = ttnn.rms_norm(cast, weight=w_tt, epsilon=1e-6, compute_kernel_config=cfg)
            flat_out = ttnn.reshape(out, flat_shape)
            ttnn.deallocate(flat_out)

        core = ttnn.reshape(src32, (1, BATCH, NV, HEAD_V))
        a = ttnn.to_torch(ttnn.reshape(
            ttnn.rms_norm(ttnn.typecast(core, ttnn.bfloat16), weight=w_tt, epsilon=1e-6,
                          compute_kernel_config=cfg), flat_shape)).float()
        c = ttnn.to_torch(ttnn.reshape(
            ttnn.rms_norm(ttnn.typecast(src32, ttnn.bfloat16), weight=w_tt, epsilon=1e-6,
                          compute_kernel_config=cfg), flat_shape)).float()
        shipped_us, shipped_spread = timed(shipped, device)
        norm_us, norm_spread = timed(norm_only, device)
        alt_us, alt_spread = timed(norm_then_flatten, device)
        row = {
            "shape_shipped": f"reshape to [1, {BATCH}, {NV}, {HEAD_V}] -> norm -> flatten",
            "shape_alternative": f"norm on [1, {BATCH * NV}, 1, {HEAD_V}] -> flatten",
            "pcc_shipped_vs_torch": pcc(golden.reshape(-1), a.reshape(-1)),
            "pcc_alternative_vs_torch": pcc(golden.reshape(-1), c.reshape(-1)),
            "pcc_shipped_vs_alternative": pcc(a.reshape(-1), c.reshape(-1)),
            "us_shipped": shipped_us, "spread_shipped": shipped_spread,
            "us_alternative": alt_us, "spread_alternative": alt_spread,
            "us_norm_only_no_flatten": norm_us, "spread_norm_only": norm_spread,
        }
        row["us_alternative_minus_shipped"] = alt_us - shipped_us
        print("RESHAPE " + json.dumps(row), flush=True)
        ttnn.deallocate(src)
        ttnn.deallocate(src32)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

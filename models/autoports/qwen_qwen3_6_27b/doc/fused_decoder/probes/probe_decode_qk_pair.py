# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Q and K are the same shape and take the same path - should they be one tensor at decode?

The fused decode turns Q and K into the recurrence's per-head layout separately: each is a
``reshape``, a ``repeat_interleave``, an ``rms_norm`` (the L2 norm, §3.12), a scalar multiply and
a second ``reshape``.  They have identical shapes - ``num_k_heads`` heads of ``head_k_dim`` - and
differ only in the constant they are scaled by, which is exactly the shape of a shared-work merge:
do all of it once on a ``[1, batch, 2 * num_k_heads, head_k_dim]`` tensor and scale with a
per-head constant column instead of a scalar.

That is worth measuring because at the advertised ``max_batch`` the rank changes are the
second-largest non-matmul cost of the step: nine ``ReshapeView`` rows, about a seventh of it.
Halving the Q/K half of them is the only part of that a graph rewrite can reach without changing
the recurrent state's format.

Both forms are timed here at batch 1 and 32 and checked against each other and against torch.

    python .../probes/probe_decode_qk_pair.py
"""

from __future__ import annotations

import math
import statistics
import time

import torch

import ttnn

NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM = 16, 48, 128
V_PER_K = NUM_V_HEADS // NUM_K_HEADS
KEY_DIM = NUM_K_HEADS * HEAD_DIM
BATCHES = (1, 32)
EPS = 1e-6


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def median_us(fn, device, iters=25):
    outs = fn()
    for tensor in outs:
        ttnn.deallocate(tensor)
    samples = []
    for _ in range(iters):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        outs = fn()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) * 1e6)
        for tensor in outs:
            ttnn.deallocate(tensor)
    return statistics.median(samples), statistics.stdev(samples)


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        torch.manual_seed(0)
        root_dk = math.sqrt(HEAD_DIM)
        q_scale, k_scale = 1.0 / (root_dk * root_dk), 1.0 / root_dk

        for batch in BATCHES:
            q_host = torch.randn(1, 1, batch, KEY_DIM)
            k_host = torch.randn(1, 1, batch, KEY_DIM)

            def dev(tensor, dtype=ttnn.float32):
                return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

            q_flat, k_flat = dev(q_host), dev(k_host)
            # The per-head scale column for the merged form: q heads first, then k heads.
            column = torch.cat([torch.full((NUM_K_HEADS, 1), q_scale), torch.full((NUM_K_HEADS, 1), k_scale)], dim=0)
            scale_column = dev(column.reshape(1, 1, 2 * NUM_K_HEADS, 1))

            def one_at_a_time():
                """What the fused decoder ships: Q and K through the same chain, separately."""
                out = []
                for flat, scale in ((q_flat, q_scale), (k_flat, k_scale)):
                    heads = ttnn.reshape(flat, (1, batch, NUM_K_HEADS, HEAD_DIM))
                    wide = ttnn.repeat_interleave(heads, V_PER_K, dim=2)
                    ttnn.deallocate(heads)
                    normed = ttnn.rms_norm(wide, epsilon=EPS / HEAD_DIM, compute_kernel_config=cfg)
                    ttnn.deallocate(wide)
                    scaled = ttnn.multiply(normed, scale)
                    ttnn.deallocate(normed)
                    rows = ttnn.reshape(scaled, (1, batch * NUM_V_HEADS, 1, HEAD_DIM))
                    if rows.buffer_address() != scaled.buffer_address():
                        ttnn.deallocate(scaled)
                    out.append(rows)
                return tuple(out)

            def merged():
                """One tensor: concat the head axes, one norm, one per-head scale, one cut."""
                pair = ttnn.concat([q_flat, k_flat], dim=-1)
                heads = ttnn.reshape(pair, (1, batch, 2 * NUM_K_HEADS, HEAD_DIM))
                ttnn.deallocate(pair)
                normed = ttnn.rms_norm(heads, epsilon=EPS / HEAD_DIM, compute_kernel_config=cfg)
                ttnn.deallocate(heads)
                scaled = ttnn.multiply(normed, scale_column)
                ttnn.deallocate(normed)
                out = []
                for index in range(2):
                    part = ttnn.slice(
                        scaled,
                        [0, 0, index * NUM_K_HEADS, 0],
                        [1, batch, (index + 1) * NUM_K_HEADS, HEAD_DIM],
                    )
                    wide = ttnn.repeat_interleave(part, V_PER_K, dim=2)
                    ttnn.deallocate(part)
                    rows = ttnn.reshape(wide, (1, batch * NUM_V_HEADS, 1, HEAD_DIM))
                    if rows.buffer_address() != wide.buffer_address():
                        ttnn.deallocate(wide)
                    out.append(rows)
                ttnn.deallocate(scaled)
                return tuple(out)

            got_a = [ttnn.to_torch(tensor).float() for tensor in one_at_a_time()]
            got_b = [ttnn.to_torch(tensor).float() for tensor in merged()]
            a_median, a_stdev = median_us(one_at_a_time, device)
            b_median, b_stdev = median_us(merged, device)
            print(
                f"decode_qk batch={batch:2d} separate_us={a_median:8.1f} ({a_stdev:5.1f}) "
                f"merged_us={b_median:8.1f} ({b_stdev:5.1f}) "
                f"pcc_q={pcc(got_a[0], got_b[0]):.6f} pcc_k={pcc(got_a[1], got_b[1]):.6f}",
                flush=True,
            )
            for tensor in (q_flat, k_flat, scale_column):
                ttnn.deallocate(tensor)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

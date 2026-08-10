# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The three batched matmuls of the single-token gated delta rule, and how wide they can run.

At batch 1 the decode recurrence is 48 independent ``[1,128] x [128,128]`` (and one
``[128,1] x [1,128]`` outer product) problems.  ``ttnn.matmul``'s default batched program
factory picks 4 and 16 cores for them, which is where ~260 us of a ~2.7 ms decode step goes.
This probe measures the default against an explicit ``MatmulMultiCoreReuseProgramConfig`` over
a wider grid, and against ``ttnn.experimental.group_attn_matmul`` for the two shapes whose
contract it can express.

    python .../probes/probe_decode_recurrence.py
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn

HEADS = 48
DK = DV = 128


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def bench(fn, device, iters=30):
    """Median/stdev over ``iters`` repeats; these shapes are dispatch-bound and noisy."""
    out = fn()
    ttnn.deallocate(out)
    samples = []
    for _ in range(iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1e6)
        got = ttnn.to_torch(out).float()
        ttnn.deallocate(out)
    return (statistics.median(samples), statistics.stdev(samples)), got


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        grid = device.compute_with_storage_grid_size()
        torch.manual_seed(0)
        k = torch.randn(1, HEADS, 1, DK)
        state = torch.randn(1, HEADS, DK, DV) * 0.1
        delta = torch.randn(1, HEADS, 1, DV)

        def dev(t):
            return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

        tk, ts, td = dev(k), dev(state), dev(delta)
        ref_read = (k @ state).float()
        ref_outer = (k.transpose(-2, -1) @ delta).float()

        (ms, sd), got = bench(lambda: ttnn.matmul(tk, ts, dtype=ttnn.float32, compute_kernel_config=cfg), device)
        print(
            f"read  default              median_us={ms:8.1f} stdev_us={sd:6.1f} pcc={pcc(ref_read, got):.6f}",
            flush=True,
        )
        for gy in (1, 2, 4, 6):
            for gx in (4, 8, 11):
                try:
                    (ms, sd), got = bench(
                        lambda: ttnn.matmul(
                            tk, ts, dtype=ttnn.float32, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=gy, x=gx)
                        ),
                        device,
                    )
                    print(
                        f"read  core_grid {gy}x{gx:<2d}         median_us={ms:8.1f} stdev_us={sd:6.1f} pcc={pcc(ref_read, got):.6f}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"read  core_grid {gy}x{gx:<2d}         FAILED {str(exc).splitlines()[0][:90]}", flush=True)

        # group_attn_matmul: [q_len=1, q_heads=1, batch=HEADS, DK] x [batch=HEADS, kv_heads=1, DK, DV]
        try:
            ga_a = ttnn.from_torch(
                k.reshape(1, 1, HEADS, DK), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            ga_b = ttnn.from_torch(
                state.reshape(HEADS, 1, DK, DV), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            (ms, sd), got = bench(
                lambda: ttnn.experimental.group_attn_matmul(
                    ga_a, ga_b, compute_with_storage_grid_size=grid, compute_kernel_config=cfg
                ),
                device,
            )
            print(
                f"read  group_attn_matmul    median_us={ms:8.1f} stdev_us={sd:6.1f} pcc={pcc(ref_read.reshape(1, 1, HEADS, DV), got):.6f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"read  group_attn_matmul    FAILED {str(exc).splitlines()[0][:110]}", flush=True)

        # The transpose can be an argument of the matmul instead of an op before it.  Swept over
        # the same grids as the spelled-out form, so the shipped variant is a row of the table
        # rather than a single point next to it.
        (ms, sd), got = bench(
            lambda: ttnn.matmul(tk, td, dtype=ttnn.float32, compute_kernel_config=cfg, transpose_a=True),
            device,
        )
        print(
            f"outer transpose_a default   median_us={ms:8.1f} stdev_us={sd:6.1f} pcc={pcc(ref_outer, got):.6f}",
            flush=True,
        )
        for gy in (1, 2, 4, 6):
            for gx in (4, 8, 11):
                try:
                    (ms, sd), got = bench(
                        lambda: ttnn.matmul(
                            tk,
                            td,
                            dtype=ttnn.float32,
                            compute_kernel_config=cfg,
                            core_grid=ttnn.CoreGrid(y=gy, x=gx),
                            transpose_a=True,
                        ),
                        device,
                    )
                    print(
                        f"outer transpose_a core_grid {gy}x{gx:<2d} median_us={ms:8.1f} "
                        f"stdev_us={sd:6.1f} pcc={pcc(ref_outer, got):.6f}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"outer transpose_a core_grid {gy}x{gx:<2d} FAILED {str(exc).splitlines()[0][:90]}",
                        flush=True,
                    )

        tk_t = ttnn.transpose(tk, -2, -1)
        (ms, sd), got = bench(lambda: ttnn.matmul(tk_t, td, dtype=ttnn.float32, compute_kernel_config=cfg), device)
        print(
            f"outer default              median_us={ms:8.1f} stdev_us={sd:6.1f} pcc={pcc(ref_outer, got):.6f}",
            flush=True,
        )
        for gy in (1, 2, 4, 6):
            for gx in (4, 8, 11):
                try:
                    (ms, sd), got = bench(
                        lambda: ttnn.matmul(
                            tk_t, td, dtype=ttnn.float32, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=gy, x=gx)
                        ),
                        device,
                    )
                    print(
                        f"outer core_grid {gy}x{gx:<2d}         median_us={ms:8.1f} stdev_us={sd:6.1f} pcc={pcc(ref_outer, got):.6f}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"outer core_grid {gy}x{gx:<2d}         FAILED {str(exc).splitlines()[0][:90]}", flush=True)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

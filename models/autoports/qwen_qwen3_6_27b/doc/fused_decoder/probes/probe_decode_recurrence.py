# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The three batched matmuls of the single-token gated delta rule, and how wide they can run.

At batch 1 the decode recurrence is 48 independent ``[1,128] x [128,128]`` (and one
``[128,1] x [1,128]`` outer product) problems.  ``ttnn.matmul``'s default batched program
factory picks 4 and 16 cores for them, which is where ~260 us of a ~2.7 ms decode step goes.
This probe measures the default against an explicit ``MatmulMultiCoreReuseProgramConfig`` over
a wider grid, and against ``ttnn.experimental.group_attn_matmul``, whose own mapping is measured by
``probe_group_attn_matmul.py`` - the call here is the mis-mapped one an early round tried, kept
because its error message is what §3.6 corrects.

    python .../probes/probe_decode_recurrence.py
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn

#: ``batch * num_v_heads`` - the number of independent head problems in one decode step.  Both
#: the batch-1 and the advertised-``max_batch`` regimes are swept, because the winning grid is a
#: function of the problem count and the shipped grids were once chosen at batch 1 only.
HEAD_COUNTS = (48, 48 * 32)
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
        for heads in HEAD_COUNTS:
            _sweep(device, grid, cfg, heads)
    finally:
        ttnn.close_mesh_device(device)


def _sweep(device, grid, cfg, HEADS: int) -> None:
    """One full grid sweep at ``HEADS`` independent head problems."""
    if True:
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
            f"read heads={HEADS}  default              median_us={ms:8.1f} stdev_us={sd:6.1f} pcc={pcc(ref_read, got):.6f}",
            flush=True,
        )
        for gy in (1, 2, 4, 6, 8, 10):
            for gx in (4, 8, 11):
                try:
                    (ms, sd), got = bench(
                        lambda: ttnn.matmul(
                            tk, ts, dtype=ttnn.float32, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=gy, x=gx)
                        ),
                        device,
                    )
                    print(
                        f"read heads={HEADS}  core_grid {gy}x{gx:<2d}         median_us={ms:8.1f} stdev_us={sd:6.1f} pcc={pcc(ref_read, got):.6f}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"read heads={HEADS}  core_grid {gy}x{gx:<2d}         FAILED {str(exc).splitlines()[0][:90]}",
                        flush=True,
                    )

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
                f"read heads={HEADS}  group_attn_matmul    median_us={ms:8.1f} stdev_us={sd:6.1f} pcc={pcc(ref_read.reshape(1, 1, HEADS, DV), got):.6f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"read heads={HEADS}  group_attn_matmul    FAILED {str(exc).splitlines()[0][:110]}", flush=True)

        # The transpose can be an argument of the matmul instead of an op before it.  Swept over
        # the same grids as the spelled-out form, so the shipped variant is a row of the table
        # rather than a single point next to it.
        (ms, sd), got = bench(
            lambda: ttnn.matmul(tk, td, dtype=ttnn.float32, compute_kernel_config=cfg, transpose_a=True),
            device,
        )
        print(
            f"outer heads={HEADS} transpose_a default   median_us={ms:8.1f} stdev_us={sd:6.1f} pcc={pcc(ref_outer, got):.6f}",
            flush=True,
        )
        for gy in (1, 2, 4, 6, 8, 10):
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
                        f"outer heads={HEADS} transpose_a core_grid {gy}x{gx:<2d} median_us={ms:8.1f} "
                        f"stdev_us={sd:6.1f} pcc={pcc(ref_outer, got):.6f}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"outer heads={HEADS} transpose_a core_grid {gy}x{gx:<2d} FAILED {str(exc).splitlines()[0][:90]}",
                        flush=True,
                    )

        tk_t = ttnn.transpose(tk, -2, -1)
        (ms, sd), got = bench(lambda: ttnn.matmul(tk_t, td, dtype=ttnn.float32, compute_kernel_config=cfg), device)
        print(
            f"outer heads={HEADS} default              median_us={ms:8.1f} stdev_us={sd:6.1f} pcc={pcc(ref_outer, got):.6f}",
            flush=True,
        )
        for gy in (1, 2, 4, 6, 8, 10):
            for gx in (4, 8, 11):
                try:
                    (ms, sd), got = bench(
                        lambda: ttnn.matmul(
                            tk_t, td, dtype=ttnn.float32, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=gy, x=gx)
                        ),
                        device,
                    )
                    print(
                        f"outer heads={HEADS} core_grid {gy}x{gx:<2d}         median_us={ms:8.1f} stdev_us={sd:6.1f} pcc={pcc(ref_outer, got):.6f}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"outer heads={HEADS} core_grid {gy}x{gx:<2d}         FAILED {str(exc).splitlines()[0][:90]}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()

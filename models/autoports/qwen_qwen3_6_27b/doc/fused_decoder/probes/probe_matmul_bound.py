# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Why the committed reports mark some matmul rows ``Bound=SLOW``, for each such row.

``tt-perf-report`` calls a matmul row ``SLOW`` when it reaches neither the DRAM nor the FLOP
roofline.  Four rows in the fused ``linear_attention`` prefill report and two in its decode report
carry that label, and a label is not a diagnosis: this probe measures each of them against the
levers that are graph properties, so ``work_log.md`` §6 can name the actual cause of each rather
than assert one.

The levers, in the order they matter for these shapes:

``out_dtype``   a float32 matmul output halves the packer's throughput per tile and, with
                ``fp32_dest_acc_en``, also halves the DEST register budget, which caps the output
                subblock.  Three of the six rows emit float32.
``fp32_dest``   the same DEST budget, without changing what the op emits.
``core_grid``   the lever §3.6 used on the decode recurrence.  A row whose N is a handful of tiles
                cannot use 110 cores no matter what, which is the shape of the a/b row's problem.

Shapes are the real ones, taken from the committed reports.  Model-free: synthetic tensors.

    python .../probes/probe_matmul_bound.py
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn

#: ``(label, M, K, N, shipped_out_dtype)`` - every ``Bound=SLOW`` matmul row in the committed
#: fused ``linear_attention`` reports, prefill first.
ROWS = (
    ("in_proj_qkv    prefill", 2048, 5120, 10240, ttnn.float32),
    ("gated_norm_sum prefill", 2048, 6144, 64, ttnn.float32),
    ("in_proj_ab     prefill", 2048, 5120, 128, ttnn.float32),
    ("gated_norm_exp prefill", 2048, 64, 6144, ttnn.bfloat16),
    ("in_proj_ab     decode ", 32, 5120, 128, ttnn.float32),
    # Not ``SLOW`` rows: the batch-32 decode shapes of the two gated-norm constant matmuls, swept
    # here because the prefill grid that wins at 2048 rows cannot be assumed to win at 32.
    ("gated_norm_sum decode ", 32, 6144, 64, ttnn.float32),
    ("gated_norm_exp decode ", 32, 64, 6144, ttnn.bfloat16),
)


def median_us(fn, device, iters):
    out = fn()
    ttnn.deallocate(out)
    samples = []
    for _ in range(iters):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) * 1e6)
        ttnn.deallocate(out)
    return statistics.median(samples), statistics.stdev(samples)


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    grid = device.compute_with_storage_grid_size()
    try:
        torch.manual_seed(0)

        def dev(tensor, dtype):
            return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        for label, rows, k_dim, n_dim, shipped in ROWS:
            x = dev(torch.randn(1, 1, rows, k_dim) * 0.05, ttnn.bfloat16)
            w = dev(torch.randn(1, 1, k_dim, n_dim) * 0.02, ttnn.bfloat16)
            #: 25 repeats even on the 2.4 ms row: the interesting deltas here are ~10 %, which
            #: a 9-sample median cannot separate from the ~25 us run-to-run spread.
            iters = 25

            def run(out_dtype, fp32_dest, core_grid=None):
                cfg = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=fp32_dest,
                    packer_l1_acc=True,
                )
                kwargs = {"dtype": out_dtype, "compute_kernel_config": cfg}
                if core_grid is not None:
                    kwargs["core_grid"] = core_grid
                return median_us(lambda: ttnn.matmul(x, w, **kwargs), device, iters)

            shipped_name = "fp32" if shipped == ttnn.float32 else "bf16"
            base, base_stdev = run(shipped, True)
            other = ttnn.bfloat16 if shipped == ttnn.float32 else ttnn.float32
            other_name = "bf16" if shipped == ttnn.float32 else "fp32"
            swapped, swapped_stdev = run(other, True)
            no_dest, no_dest_stdev = run(shipped, False)
            print(
                f"matmul {label} {rows:5d}x{k_dim:5d}x{n_dim:5d} "
                f"out={shipped_name}+fp32dest_us={base:9.1f} ({base_stdev:6.1f}) "
                f"out={other_name}+fp32dest_us={swapped:9.1f} ({swapped_stdev:6.1f}) "
                f"out={shipped_name}+bf16dest_us={no_dest:9.1f} ({no_dest_stdev:6.1f})",
                flush=True,
            )

            # core_grid: a row whose N is a handful of tiles cannot use the whole grid, and on this
            # checkout the default 1D program config still *asks* for it, so sweep down to 2 cores.
            n_tiles = n_dim // 32
            for cores_y, cores_x in ((grid.y, grid.x), (8, 8), (4, 8), (2, 8), (2, 4), (1, 4), (1, 2)):
                if cores_y > grid.y or cores_x > grid.x:
                    continue
                try:
                    value, spread = run(shipped, True, ttnn.CoreGrid(y=cores_y, x=cores_x))
                except Exception as error:  # noqa: BLE001 - the blocker text is the result
                    print(
                        f"matmul {label} core_grid {cores_y}x{cores_x} rejected: "
                        f"{type(error).__name__}: {str(error).splitlines()[0][:140]}",
                        flush=True,
                    )
                    continue
                print(
                    f"matmul {label} core_grid {cores_y}x{cores_x:2d} us={value:9.1f} ({spread:6.1f}) "
                    f"n_tiles={n_tiles}",
                    flush=True,
                )
            ttnn.deallocate(x)
            ttnn.deallocate(w)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

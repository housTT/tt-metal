# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Which form of the z-gated per-head RMS norm is cheaper at each decode batch size.

The norm is over ``head_v_dim`` = 128 channels of a ``[1, 1, batch, value_dim]`` tensor, and
there are two ways to say that:

``reshape``   view it as ``[1, batch, num_v_heads, head_v_dim]``, call ``ttnn.rms_norm``, gate,
              and view it back - two full TILE relayouts whose cost grows with the row count;
``group``     a group reduction against two constant matrices (the prefill path, §3.4) - two
              skinny matmuls whose cost barely moves with the row count.

They are the same arithmetic, so which is cheaper is a pure function of the batch size, and the
fused decoder picks between them at construction (:data:`~..tt.fused_decoder._GATED_NORM_GROUP_BATCH`).
This probe is where that threshold comes from.  Model-free: synthetic tensors at the real shapes.

    python .../probes/probe_gated_norm_batch.py
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn

NUM_V_HEADS, HEAD_V_DIM = 48, 128
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM
GROUP_PAD = 64
BATCHES = (1, 4, 8, 16, 32)
#: Filled in ``main`` from the device's grid; the shipped values are ``_GROUP_SUM_GRID["decode"]``
#: and ``_GROUP_EXPAND_GRID["decode"]``.
SUM_GRID = None
EXPAND_GRID = None


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def median_us(fn, device, iters=25):
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
    # The shipped decode grids for the two constant matmuls (§3.18).  A stage review found this
    # probe measuring the *default* program factory for both, which charges the group form about
    # 65 us it does not pay - and the threshold this log chooses was being chosen against that.
    global SUM_GRID, EXPAND_GRID
    grid = device.compute_with_storage_grid_size()
    SUM_GRID = ttnn.CoreGrid(y=min(1, grid.y), x=min(4, grid.x))
    EXPAND_GRID = ttnn.CoreGrid(y=min(2, grid.y), x=min(8, grid.x))
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        torch.manual_seed(0)
        weight = torch.randn(HEAD_V_DIM) * 0.02
        group_mean = torch.zeros(VALUE_DIM, GROUP_PAD)
        scale_expand = torch.zeros(GROUP_PAD, VALUE_DIM)
        for head in range(NUM_V_HEADS):
            group_mean[head * HEAD_V_DIM : (head + 1) * HEAD_V_DIM, head] = 1.0 / HEAD_V_DIM
            scale_expand[head, head * HEAD_V_DIM : (head + 1) * HEAD_V_DIM] = weight

        def dev(tensor):
            return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        head_weight = dev(weight.reshape(1, 1, 1, HEAD_V_DIM))
        t_group = dev(group_mean.reshape(1, 1, VALUE_DIM, GROUP_PAD))
        t_scale = dev(scale_expand.reshape(1, 1, GROUP_PAD, VALUE_DIM))

        for batch in BATCHES:
            core = dev(torch.randn(1, 1, batch, VALUE_DIM))
            z = dev(torch.randn(1, 1, batch, VALUE_DIM))

            def reshape_form():
                heads = ttnn.reshape(core, (1, batch, NUM_V_HEADS, HEAD_V_DIM))
                z_heads = ttnn.reshape(z, (1, batch, NUM_V_HEADS, HEAD_V_DIM))
                normed = ttnn.rms_norm(heads, epsilon=1e-6, weight=head_weight, compute_kernel_config=cfg)
                gated = ttnn.multiply(normed, z_heads, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
                ttnn.deallocate(normed)
                out = ttnn.reshape(gated, (1, 1, batch, VALUE_DIM))
                if out.buffer_address() != gated.buffer_address():
                    ttnn.deallocate(gated)
                return out

            def group_form():
                squares = ttnn.multiply(core, core)
                mean_square = ttnn.matmul(
                    squares,
                    t_group,
                    dtype=ttnn.float32,
                    compute_kernel_config=cfg,
                    core_grid=SUM_GRID,
                )
                ttnn.deallocate(squares)
                inv = ttnn.add(mean_square, 1e-6, activations=[ttnn.UnaryOpType.RSQRT])
                ttnn.deallocate(mean_square)
                inv16 = ttnn.typecast(inv, ttnn.bfloat16)
                ttnn.deallocate(inv)
                scale = ttnn.matmul(
                    inv16, t_scale, dtype=ttnn.bfloat16, compute_kernel_config=cfg, core_grid=EXPAND_GRID
                )
                ttnn.deallocate(inv16)
                normed = ttnn.multiply(core, scale)
                ttnn.deallocate(scale)
                out = ttnn.multiply(normed, z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
                ttnn.deallocate(normed)
                return out

            a, b = reshape_form(), group_form()
            agreement = pcc(ttnn.to_torch(a).float(), ttnn.to_torch(b).float())
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            reshape_median, reshape_stdev = median_us(reshape_form, device)
            group_median, group_stdev = median_us(group_form, device)
            print(
                f"gated_norm batch={batch:2d} reshape_us={reshape_median:8.1f} ({reshape_stdev:5.1f}) "
                f"group_us={group_median:8.1f} ({group_stdev:5.1f}) pcc_between={agreement:.6f}",
                flush=True,
            )
            ttnn.deallocate(core)
            ttnn.deallocate(z)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

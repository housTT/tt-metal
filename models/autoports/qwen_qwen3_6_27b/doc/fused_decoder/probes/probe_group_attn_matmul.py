# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``ttnn.experimental.group_attn_matmul`` for the decode state reads, mapped the way it wants.

§3.6 tried this op and recorded ``TT_FATAL: Num of users must match!`` as its contract blocker.  A
stage review showed the mapping was wrong: the probe had put the *flattened* ``batch *
num_v_heads`` axis in the op's **batch** slot.  What the op actually wants
(``group_attn_matmul_device_operation.cpp``) is

    a = [1, q_heads, batch, head_dim]      b = [batch, kv_heads, head_dim, kv_len]

with ``ashape[1] % bshape[1] == 0``, ``ashape[2] == bshape[0]`` and ``bshape[0] == 32``.  With
``q_heads = kv_heads = num_v_heads`` and ``batch`` in the *tile-height* slot, all three hold at the
advertised ``max_batch`` - and the batch axis then carries 32 real rows instead of one row padded
to 32, which is where a third of the shipped read's DRAM traffic goes.

So this measures the shipped batched matmul against the dedicated op *including* the permutes the
mapping needs, at the advertised batch, and checks both against torch.

    python .../probes/probe_group_attn_matmul.py
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn

NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM = 48, 128, 128
#: The op hard-requires ``bshape[0] == 32``, so this is the only batch it can serve.
BATCH = 32
SHIPPED_GRID = (10, 4)


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def median_us(fn, device, iters=15):
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
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        torch.manual_seed(0)
        heads = BATCH * NUM_V_HEADS
        # ``k`` per head, and the recurrent state, in the layouts the shipped decode carries.
        k_host = torch.randn(1, heads, 1, HEAD_K_DIM)
        state_host = torch.randn(1, heads, HEAD_K_DIM, HEAD_V_DIM) * 0.1
        reference = (k_host @ state_host).float()  # [1, heads, 1, dv]

        def dev(tensor, dtype=ttnn.float32):
            return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        k_rows = dev(k_host)
        state = dev(state_host)
        # The op's ``b``: the state is user-major already, so ``[batch, kv_heads, dk, dv]`` is a
        # free leading-dimension view of the same buffer.
        state_grouped = ttnn.reshape(state, (BATCH, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM))
        # The op's ``a``: ``[1, q_heads, batch, dk]``, i.e. the dense per-head activation with the
        # user axis in the tile height.  One permute of a 1.5 MB tensor builds it.
        k_dense = ttnn.reshape(k_rows, (1, BATCH, NUM_V_HEADS, HEAD_K_DIM))
        k_grouped = ttnn.permute(k_dense, (0, 2, 1, 3))

        def shipped():
            """The batched matmul the layer ships, on its measured grid."""
            return ttnn.matmul(
                k_rows,
                state,
                dtype=ttnn.float32,
                compute_kernel_config=cfg,
                core_grid=ttnn.CoreGrid(y=min(SHIPPED_GRID[0], grid.y), x=min(SHIPPED_GRID[1], grid.x)),
            )

        def grouped():
            """The dedicated op, plus the permute back to the layout the consumer wants."""
            out = ttnn.experimental.group_attn_matmul(
                k_grouped,
                state_grouped,
                compute_with_storage_grid_size=grid,
                compute_kernel_config=cfg,
            )
            dense = ttnn.permute(out, (0, 2, 1, 3))
            ttnn.deallocate(out)
            return dense

        got_shipped = ttnn.to_torch(shipped()).float()
        try:
            got_grouped = ttnn.to_torch(grouped()).float()
        except Exception as error:  # noqa: BLE001 - the blocker text is the result
            print(
                f"group_attn_matmul batch={BATCH} dtype=fp32 rejected: {type(error).__name__}: "
                f"{str(error).splitlines()[0][:170]}",
                flush=True,
            )
            # Half the bytes, in case the blocker is a circular-buffer budget rather than a
            # contract: if bfloat16 fits, the op is available at a precision cost and the
            # rejection has to be argued on precision instead.
            k16 = ttnn.typecast(k_grouped, ttnn.bfloat16)
            state16 = ttnn.typecast(state_grouped, ttnn.bfloat16)
            try:
                out = ttnn.experimental.group_attn_matmul(
                    k16, state16, compute_with_storage_grid_size=grid, compute_kernel_config=cfg
                )
                got16 = ttnn.to_torch(out).float()
                ttnn.deallocate(out)
                print(
                    f"group_attn_matmul batch={BATCH} dtype=bf16 accepted, "
                    f"pcc_vs_torch="
                    f"{pcc(got16.reshape(-1), reference.reshape(1, BATCH, NUM_V_HEADS, HEAD_V_DIM).permute(0, 2, 1, 3).reshape(-1)):.6f}",
                    flush=True,
                )
            except Exception as second:  # noqa: BLE001
                print(
                    f"group_attn_matmul batch={BATCH} dtype=bf16 rejected: {type(second).__name__}: "
                    f"{str(second).splitlines()[0][:170]}",
                    flush=True,
                )
            return
        shipped_median, shipped_stdev = median_us(shipped, device)
        grouped_median, grouped_stdev = median_us(grouped, device)
        print(
            f"group_attn_matmul batch={BATCH} shipped_us={shipped_median:9.1f} ({shipped_stdev:6.1f}) "
            f"grouped_us={grouped_median:9.1f} ({grouped_stdev:6.1f}) "
            f"pcc_shipped_vs_torch={pcc(got_shipped.reshape(-1), reference.reshape(-1)):.6f} "
            f"pcc_grouped_vs_torch={pcc(got_grouped.reshape(-1), reference.reshape(1, BATCH, NUM_V_HEADS, HEAD_V_DIM).reshape(-1)):.6f}",
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

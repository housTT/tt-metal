# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Should the recurrence's *transient* vectors be dense instead of one row per head?

The decode recurrence works in ``[1, batch * num_v_heads, 1, head_dim]``: one logical row per
head, padded to a 32-row tile, so about 31 of every 32 bytes moved are padding.  The three matmuls
need that layout - a per-head batch dimension is what makes them batched - but ``v``, ``b``,
``kv_mem``, ``residual`` and ``delta`` are transients that no matmul reads in that shape.

A stage review pointed out that §6 excluded this whole cost as "a state-format change", which is
true of the *state* and not of the transients.  The dense alternative keeps them as
``[1, batch, num_v_heads, head_v_dim]`` - 1/24th of the bytes at batch 32 - and pays two rank
changes at the matmul boundaries instead of the ones it removes:

``shipped``  ``v`` and ``b`` reshaped to per-head rows, then subtract/multiply there;
``dense``    ``kv_mem`` reshaped to dense, subtract/multiply there, ``delta`` reshaped back.

Both are timed at both decode regimes and checked against each other.

    python .../probes/probe_dense_recurrence.py
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn

NUM_V_HEADS, HEAD_V_DIM = 48, 128
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM
BATCHES = (1, 32)


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
    try:
        torch.manual_seed(0)
        for batch in BATCHES:
            heads = batch * NUM_V_HEADS

            def dev(tensor, dtype=ttnn.float32):
                return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

            # ``v_flat`` is what ``_split_qkv`` produces; ``kv_rows`` is what the state-read matmul
            # produces (per-head rows); ``b_flat`` is the raw ``b`` projection.
            v_flat = dev(torch.randn(1, 1, batch, VALUE_DIM) * 0.5)
            kv_rows = dev(torch.randn(1, heads, 1, HEAD_V_DIM) * 0.5)
            b_flat = dev(torch.randn(1, 1, batch, NUM_V_HEADS))

            def shipped():
                """Per-head rows throughout: what the fused decoder does today."""
                v_heads = ttnn.reshape(v_flat, (1, batch, NUM_V_HEADS, HEAD_V_DIM))
                v_rows = ttnn.reshape(v_heads, (1, heads, 1, HEAD_V_DIM))
                if v_rows.buffer_address() != v_heads.buffer_address():
                    ttnn.deallocate(v_heads)
                b_rows = ttnn.reshape(b_flat, (1, heads, 1, 1))
                residual = ttnn.subtract(v_rows, kv_rows)
                ttnn.deallocate(v_rows)
                delta = ttnn.multiply(residual, b_rows, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                ttnn.deallocate(residual)
                ttnn.deallocate(b_rows)
                return delta

            def dense():
                """Dense transients: one rank change in, one out, arithmetic on 1/24th the bytes."""
                v_heads = ttnn.reshape(v_flat, (1, batch, NUM_V_HEADS, HEAD_V_DIM))
                b_heads = ttnn.reshape(b_flat, (1, batch, NUM_V_HEADS, 1))
                kv_heads = ttnn.reshape(kv_rows, (1, batch, NUM_V_HEADS, HEAD_V_DIM))
                residual = ttnn.subtract(v_heads, kv_heads)
                ttnn.deallocate(v_heads)
                if kv_heads.buffer_address() != kv_rows.buffer_address():
                    ttnn.deallocate(kv_heads)
                gated = ttnn.multiply(residual, b_heads, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                ttnn.deallocate(residual)
                ttnn.deallocate(b_heads)
                delta = ttnn.reshape(gated, (1, heads, 1, HEAD_V_DIM))
                if delta.buffer_address() != gated.buffer_address():
                    ttnn.deallocate(gated)
                return delta

            got_a = ttnn.to_torch(shipped()).float()
            got_b = ttnn.to_torch(dense()).float()
            a_median, a_stdev = median_us(shipped, device)
            b_median, b_stdev = median_us(dense, device)
            print(
                f"dense_recurrence batch={batch:2d} rows_us={a_median:8.1f} ({a_stdev:5.1f}) "
                f"dense_us={b_median:8.1f} ({b_stdev:5.1f}) "
                f"pcc_between={pcc(got_a, got_b):.6f} "
                f"max_abs_diff={float((got_a - got_b).abs().max()):.3e}",
                flush=True,
            )
            for tensor in (v_flat, kv_rows, b_flat):
                ttnn.deallocate(tensor)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

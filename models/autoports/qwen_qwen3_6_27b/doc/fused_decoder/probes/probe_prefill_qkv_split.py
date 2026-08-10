# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The other direction of §6.2's question: should ``in_proj_qkv`` be one matmul, or three?

The GatedDeltaNet input projection is one ``[hidden, 2 * key_dim + value_dim]`` matmul whose
output the causal conv consumes whole and ``_split_qkv`` then cuts into Q, K and V.  §6.2 measured
the *inverse* rewrite on ``full_attention``'s ``wqkv``/``wgate`` pair and found that cutting a wide
TILE tensor apart costs more than the merge saves - which raises the same question here, where
those three slices are among the larger ``layout`` rows of the ``linear_attention`` prefill.

The catch is that the conv runs on the *packed* tensor: the FIR is one depthwise pass over all
``conv_dim`` channels.  Splitting the projection means either three convs or a concat before the
conv, so this measures the whole chain - projection, conv, split - both ways:

``packed``   one projection, one FIR, three slices (what ships);
``split``    three projections, three FIRs, no slices.

    python .../probes/probe_prefill_qkv_split.py
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn

HIDDEN = 5120
KEY_DIM, VALUE_DIM = 2048, 6144
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
K_SIZE = 4
SEQ = 2048


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def median_ms(fn, device, iters=9):
    outs = fn()
    for tensor in outs:
        ttnn.deallocate(tensor)
    samples = []
    for _ in range(iters):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        outs = fn()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) * 1e3)
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
        widths = (KEY_DIM, KEY_DIM, VALUE_DIM)

        def dev(tensor, dtype):
            return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        weight_host = torch.randn(HIDDEN, CONV_DIM) * 0.02
        packed_weight = dev(weight_host.reshape(1, 1, HIDDEN, CONV_DIM), ttnn.bfloat16)
        offsets = (0, KEY_DIM, 2 * KEY_DIM)
        split_weights = [
            dev(weight_host[:, start : start + width].reshape(1, 1, HIDDEN, width), ttnn.bfloat16)
            for start, width in zip(offsets, widths)
        ]
        taps_host = [torch.randn(1, 1, 1, CONV_DIM) * 0.3 for _ in range(K_SIZE)]
        packed_taps = [dev(tap, ttnn.bfloat16) for tap in taps_host]
        split_taps = [
            [dev(tap[:, :, :, start : start + width], ttnn.bfloat16) for tap in taps_host]
            for start, width in zip(offsets, widths)
        ]
        x = dev(torch.randn(1, 1, SEQ, HIDDEN), ttnn.bfloat16)

        def fir(activation, taps, width):
            """The shipped ROW_MAJOR-window FIR, in the form §3.7 ships, on ``width`` channels."""
            rows = ttnn.to_layout(activation, ttnn.ROW_MAJOR_LAYOUT)
            prefix = ttnn.zeros(
                (1, 1, K_SIZE - 1, width), dtype=rows.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            window = ttnn.concat([prefix, rows], dim=2)
            ttnn.deallocate(prefix)
            ttnn.deallocate(rows)
            acc = None
            for index, tap in enumerate(taps):
                piece = ttnn.slice(window, [0, 0, index, 0], [1, 1, index + SEQ, width])
                tiled = ttnn.to_layout(piece, ttnn.TILE_LAYOUT)
                ttnn.deallocate(piece)
                term = ttnn.multiply(tiled, tap)
                ttnn.deallocate(tiled)
                if acc is None:
                    acc = term
                    continue
                merged = ttnn.add(acc, term, activations=[ttnn.UnaryOpType.SILU] if index == K_SIZE - 1 else [])
                ttnn.deallocate(term)
                ttnn.deallocate(acc)
                acc = merged
            ttnn.deallocate(window)
            return acc

        def packed_form():
            mixed = ttnn.linear(x, packed_weight, dtype=ttnn.bfloat16, compute_kernel_config=cfg)
            conv = fir(mixed, packed_taps, CONV_DIM)
            ttnn.deallocate(mixed)
            parts = [
                ttnn.slice(conv, [0, 0, 0, start], [1, 1, SEQ, start + width]) for start, width in zip(offsets, widths)
            ]
            ttnn.deallocate(conv)
            return tuple(parts)

        def split_form():
            parts = []
            for weight, taps, width in zip(split_weights, split_taps, widths):
                projected = ttnn.linear(x, weight, dtype=ttnn.bfloat16, compute_kernel_config=cfg)
                parts.append(fir(projected, taps, width))
                ttnn.deallocate(projected)
            return tuple(parts)

        got_a = [ttnn.to_torch(tensor).float() for tensor in packed_form()]
        got_b = [ttnn.to_torch(tensor).float() for tensor in split_form()]
        a_median, a_stdev = median_ms(packed_form, device)
        b_median, b_stdev = median_ms(split_form, device)
        print(
            f"prefill_qkv seq={SEQ} packed_ms={a_median:8.3f} ({a_stdev:6.3f}) "
            f"split_ms={b_median:8.3f} ({b_stdev:6.3f}) "
            + " ".join(f"pcc_{name}={pcc(a, b):.6f}" for name, a, b in zip("qkv", got_a, got_b)),
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

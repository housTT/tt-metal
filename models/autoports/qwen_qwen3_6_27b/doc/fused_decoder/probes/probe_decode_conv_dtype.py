# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Should the *decode* causal-conv FIR run in bfloat16, as the prefill one does?

§3.7 measured a float32 height-broadcast multiply at about a fifth of the bandwidth the same op
reaches on same-shape float32 operands, and moved the prefill FIR to bfloat16 for it.  The decode
FIR was left in float32, and in the committed batch-32 report its four tap multiplies cost about
four times what the three adds beside them do, on the same shapes - the same gap.

The decode FIR is not the prefill one, though:

* its taps read **carried state buffers**, which are float32 because the recurrence's carried
  precision is a stage-1 decision, so a bfloat16 FIR has to cast them - ``K - 1`` casts per step
  that the prefill path does not pay;
* it is one token wide, so every tensor is a single tile row and the whole thing is small.

Whether the casts eat the gain is therefore a measurement, not an argument.  Both whole FIRs are
timed here at batch 1 and at the advertised ``max_batch``, against the same inputs, and checked
against each other and against torch.

    python .../probes/probe_decode_conv_dtype.py
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn

CONV_DIM = 10240
K_SIZE = 4
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
        taps_host = [torch.randn(1, 1, 1, CONV_DIM) * 0.3 for _ in range(K_SIZE)]

        def dev(tensor, dtype):
            return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        taps32 = [dev(tap, ttnn.float32) for tap in taps_host]
        taps16 = [dev(tap, ttnn.bfloat16) for tap in taps_host]

        for batch in BATCHES:
            token_host = torch.randn(1, 1, batch, CONV_DIM)
            state_host = [torch.randn(1, 1, batch, CONV_DIM) for _ in range(K_SIZE - 1)]
            token32 = dev(token_host, ttnn.float32)
            state32 = [dev(row, ttnn.float32) for row in state_host]

            def float32_form():
                """What the fused decoder ships: every tap multiply in float32."""
                acc = ttnn.multiply(token32, taps32[K_SIZE - 1])
                for j in range(K_SIZE - 1):
                    term = ttnn.multiply(state32[j], taps32[j])
                    merged = ttnn.add(acc, term, activations=[ttnn.UnaryOpType.SILU] if j == K_SIZE - 2 else [])
                    ttnn.deallocate(term)
                    ttnn.deallocate(acc)
                    acc = merged
                return acc

            def bfloat16_form():
                """The prefill FIR's arithmetic: cast the carried state, multiply in bfloat16."""
                token16 = ttnn.typecast(token32, ttnn.bfloat16)
                acc = ttnn.multiply(token16, taps16[K_SIZE - 1])
                ttnn.deallocate(token16)
                for j in range(K_SIZE - 1):
                    state16 = ttnn.typecast(state32[j], ttnn.bfloat16)
                    term = ttnn.multiply(state16, taps16[j])
                    ttnn.deallocate(state16)
                    merged = ttnn.add(acc, term, activations=[ttnn.UnaryOpType.SILU] if j == K_SIZE - 2 else [])
                    ttnn.deallocate(term)
                    ttnn.deallocate(acc)
                    acc = merged
                return acc

            got32 = ttnn.to_torch(float32_form()).float()
            got16 = ttnn.to_torch(bfloat16_form()).float()
            reference = token_host * taps_host[K_SIZE - 1]
            for j in range(K_SIZE - 1):
                reference = reference + state_host[j] * taps_host[j]
            reference = torch.nn.functional.silu(reference)
            float32_median, float32_stdev = median_us(float32_form, device)
            bfloat16_median, bfloat16_stdev = median_us(bfloat16_form, device)
            print(
                f"decode_conv batch={batch:2d} float32_us={float32_median:8.1f} ({float32_stdev:5.1f}) "
                f"bfloat16_us={bfloat16_median:8.1f} ({bfloat16_stdev:5.1f}) "
                f"pcc_float32_vs_torch={pcc(got32, reference):.6f} "
                f"pcc_bfloat16_vs_torch={pcc(got16, reference):.6f} "
                f"pcc_between={pcc(got32, got16):.6f}",
                flush=True,
            )
            ttnn.deallocate(token32)
            for row in state32:
                ttnn.deallocate(row)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

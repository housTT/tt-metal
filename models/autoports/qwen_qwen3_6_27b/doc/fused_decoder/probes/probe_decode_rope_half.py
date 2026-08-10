# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Is ``ttnn.experimental.rotate_half`` faster than the spelled-out rotate-half at *decode*?

§3.2 replaced the functional layer's ``slice`` / ``slice`` / ``neg`` / ``concat`` rotate-half with
the dedicated op on both paths, and recorded the substitution as an op-count win (10 ops to 7) with
no timing.  A stage review then read the committed reports and found the dedicated op costs about
27 us per call at the advertised ``max_batch`` against about 6 us for the four ops it replaced,
because ``rotate_half``'s program factory pins ``CoreCoord({0, 0})`` - it is single-core by
construction, so its cost scales with the batch and the spelled-out form parallelises.

This probe is the measurement that was missing, at both decode regimes and at the real shapes:
``[1, batch, padded_heads, rotary_dim]`` for the query and for the key.

    python .../probes/probe_decode_rope_half.py
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn

#: q heads and k heads at decode, padded to a tile row as ``nlp_create_qkv_heads_decode`` returns
#: them, and the rotary block width (this model rotates 64 of 256 head channels).
HEAD_COUNTS = (32, 32)
ROTARY_DIM = 64
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
            for heads in HEAD_COUNTS[:1]:
                rot_host = torch.randn(1, batch, heads, ROTARY_DIM)
                rot = ttnn.from_torch(rot_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

                def dedicated():
                    """What the fused decoder ships."""
                    return ttnn.experimental.rotate_half(rot)

                def spelled_out():
                    """What the functional layer does: two slices, a negate and a concat."""
                    half = ROTARY_DIM // 2
                    first = ttnn.slice(rot, [0, 0, 0, 0], [1, batch, heads, half])
                    second = ttnn.slice(rot, [0, 0, 0, half], [1, batch, heads, ROTARY_DIM])
                    negated = ttnn.neg(second)
                    ttnn.deallocate(second)
                    out = ttnn.concat([negated, first], dim=-1)
                    ttnn.deallocate(negated)
                    ttnn.deallocate(first)
                    return out

                got_a = ttnn.to_torch(dedicated()).float()
                got_b = ttnn.to_torch(spelled_out()).float()
                a_median, a_stdev = median_us(dedicated, device)
                b_median, b_stdev = median_us(spelled_out, device)
                print(
                    f"rope_half batch={batch:2d} heads={heads:3d} dedicated_us={a_median:8.1f} ({a_stdev:5.1f}) "
                    f"spelled_out_us={b_median:8.1f} ({b_stdev:5.1f}) "
                    f"pcc_between={pcc(got_a, got_b):.6f} "
                    f"max_abs_diff={float((got_a - got_b).abs().max()):.3e}",
                    flush=True,
                )
                ttnn.deallocate(rot)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

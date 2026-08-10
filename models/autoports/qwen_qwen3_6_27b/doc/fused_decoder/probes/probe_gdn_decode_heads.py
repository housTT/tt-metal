# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Where the Q/K L2 norm and the GQA expansion should sit relative to each other, at decode.

The fused decode turns each flat ``[1, 1, batch, key_dim]`` projection into the recurrence's
per-head layout ``[1, batch * num_v_heads, 1, head_dim]``, and on the way it L2-normalises Q and
K.  The order the layer shipped is *expand, then normalise*: ``repeat_interleave`` widens 16 k
heads to 48 v heads and the norm then runs over all 48.  The norm is per head and the expansion
copies whole heads, so *normalise, then expand* is the same arithmetic over a third of the rows.

That reordering also changes which reshapes are needed.  Expanding on the flattened
``[1, batch * heads, 1, dim]`` axis instead of the ``[1, batch, heads, dim]`` one means the rank
change happens once rather than on both sides of the expansion - and in the batch-32 report the
``ReshapeView`` rows are a seventh of the whole step, because a ``[1, N, 1, 128]`` TILE tensor
carries 32 padded rows for every real one.

Both orders are timed here at batch 1 and at the advertised ``max_batch``, and checked against
each other and against torch.

    python .../probes/probe_gdn_decode_heads.py
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
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        torch.manual_seed(0)
        scale = 1.0 / math.sqrt(HEAD_DIM)

        for batch in BATCHES:
            flat_host = torch.randn(1, 1, batch, KEY_DIM)
            flat = ttnn.from_torch(flat_host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

            def expand_then_norm():
                """What the fused decoder shipped: widen to 48 heads, then normalise all of them."""
                heads = ttnn.reshape(flat, (1, batch, NUM_K_HEADS, HEAD_DIM))
                wide = ttnn.repeat_interleave(heads, V_PER_K, dim=2)
                ttnn.deallocate(heads)
                normed = ttnn.rms_norm(wide, epsilon=EPS / HEAD_DIM, compute_kernel_config=cfg)
                ttnn.deallocate(wide)
                scaled = ttnn.multiply(normed, scale)
                ttnn.deallocate(normed)
                out = ttnn.reshape(scaled, (1, batch * NUM_V_HEADS, 1, HEAD_DIM))
                if out.buffer_address() != scaled.buffer_address():
                    ttnn.deallocate(scaled)
                return out

            def norm_then_expand():
                """Normalise 16 heads, take the rank change once, then widen."""
                heads = ttnn.reshape(flat, (1, batch, NUM_K_HEADS, HEAD_DIM))
                normed = ttnn.rms_norm(heads, epsilon=EPS / HEAD_DIM, compute_kernel_config=cfg)
                ttnn.deallocate(heads)
                scaled = ttnn.multiply(normed, scale)
                ttnn.deallocate(normed)
                narrow = ttnn.reshape(scaled, (1, batch * NUM_K_HEADS, 1, HEAD_DIM))
                if narrow.buffer_address() != scaled.buffer_address():
                    ttnn.deallocate(scaled)
                out = ttnn.repeat_interleave(narrow, V_PER_K, dim=1)
                ttnn.deallocate(narrow)
                return out

            got_a = ttnn.to_torch(expand_then_norm()).float()
            got_b = ttnn.to_torch(norm_then_expand()).float()
            reference = flat_host.reshape(batch, NUM_K_HEADS, HEAD_DIM)
            reference = reference / reference.pow(2).sum(-1, keepdim=True).add(EPS).sqrt() * scale * math.sqrt(HEAD_DIM)
            reference = reference.repeat_interleave(V_PER_K, dim=1).reshape(1, batch * NUM_V_HEADS, 1, HEAD_DIM)
            a_median, a_stdev = median_us(expand_then_norm, device)
            b_median, b_stdev = median_us(norm_then_expand, device)
            print(
                f"decode_heads batch={batch:2d} expand_then_norm_us={a_median:8.1f} ({a_stdev:5.1f}) "
                f"norm_then_expand_us={b_median:8.1f} ({b_stdev:5.1f}) "
                f"pcc_between={pcc(got_a, got_b):.6f} "
                f"pcc_shipped_vs_torch={pcc(got_a, reference):.6f} "
                f"pcc_reordered_vs_torch={pcc(got_b, reference):.6f}",
                flush=True,
            )
            ttnn.deallocate(flat)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

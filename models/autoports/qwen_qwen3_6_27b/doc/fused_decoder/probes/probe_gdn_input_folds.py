# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The remaining unary-into-binary folds and rank changes around the GatedDeltaNet inputs.

Three questions the earlier probes did not cover, all of them instances of patterns this stage
already applied elsewhere:

``decay``     decode does ``exp(g)`` and then multiplies the recurrent state by it.  ``exp`` is a
              legal ``input_tensor_b_activations`` entry, so the unary can ride on the multiply -
              but the multiply's ``b`` is *height-and-width broadcast* (``[1, BH, 1, 1]`` against
              ``[1, BH, dk, dv]``), and whether the activation is applied once per element of
              ``b`` or once per element of the broadcast output decides whether this is free.
``beta``      the same fold for ``sigmoid`` on the ``delta`` multiply, which needs ``_gdn_inputs``
              to hand the decode path the raw ``b`` instead of ``sigmoid(b)``.
``rank3``     prefill slices ``b``/``a`` out of the packed projection at rank 4 and then reshapes
              ``beta``/``g`` to rank 3 for ``chunk_gated_delta_rule``.  Doing the rank change once
              on the packed ``[1, 1, T, 128]`` output instead - before the slices, where the last
              axis is tile-aligned - should remove two float32 rank-changing reshapes of a
              48-wide (tile-padded to 64) tensor.

Model-free: synthetic tensors at the real shapes.

    python .../probes/probe_gdn_input_folds.py
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn

NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM = 48, 128, 128
AB_STRIDE = 64
SEQ = 2048
BATCHES = (1, 32)


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def median_us(fn, device, iters=25):
    out = fn()
    for tensor in out if isinstance(out, tuple) else (out,):
        ttnn.deallocate(tensor)
    samples = []
    for _ in range(iters):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) * 1e6)
        for tensor in out if isinstance(out, tuple) else (out,):
            ttnn.deallocate(tensor)
    return statistics.median(samples), statistics.stdev(samples)


def keep(fn, device):
    """Run once and return the torch value, for the agreement check."""
    out = fn()
    tensors = out if isinstance(out, tuple) else (out,)
    values = [ttnn.to_torch(tensor).float() for tensor in tensors]
    for tensor in tensors:
        ttnn.deallocate(tensor)
    return values


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        torch.manual_seed(0)

        def dev(tensor, dtype=ttnn.float32):
            return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        # ------------------------------------------------------------------ decay / beta folds
        for batch in BATCHES:
            heads = batch * NUM_V_HEADS
            state = dev(torch.randn(1, heads, HEAD_K_DIM, HEAD_V_DIM) * 0.1)
            g = dev(-torch.nn.functional.softplus(torch.randn(1, heads, 1, 1)) * 0.3)
            v = dev(torch.randn(1, heads, 1, HEAD_V_DIM) * 0.5)
            kv = dev(torch.randn(1, heads, 1, HEAD_V_DIM) * 0.5)
            b = dev(torch.randn(1, heads, 1, 1))

            def decay_split():
                decay = ttnn.exp(g)
                out = ttnn.multiply(state, decay)
                ttnn.deallocate(decay)
                return out

            def decay_folded():
                return ttnn.multiply(state, g, input_tensor_b_activations=[ttnn.UnaryOpType.EXP])

            def beta_split():
                beta = ttnn.sigmoid(b)
                out = ttnn.multiply(ttnn.subtract(v, kv), beta)
                ttnn.deallocate(beta)
                return out

            def beta_folded():
                return ttnn.multiply(ttnn.subtract(v, kv), b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])

            for name, split_fn, fold_fn in (("decay", decay_split, decay_folded), ("beta", beta_split, beta_folded)):
                try:
                    (folded_value,) = keep(fold_fn, device)
                except Exception as error:  # noqa: BLE001 - the blocker text is the result
                    print(
                        f"fold {name:5s} batch={batch:2d} rejected: "
                        f"{type(error).__name__}: {str(error).splitlines()[0][:140]}",
                        flush=True,
                    )
                    continue
                (split_value,) = keep(split_fn, device)
                split_median, split_stdev = median_us(split_fn, device)
                fold_median, fold_stdev = median_us(fold_fn, device)
                print(
                    f"fold {name:5s} batch={batch:2d} split_us={split_median:8.1f} ({split_stdev:5.1f}) "
                    f"folded_us={fold_median:8.1f} ({fold_stdev:5.1f}) "
                    f"pcc={pcc(split_value, folded_value):.6f} "
                    f"max_abs_diff={float((split_value - folded_value).abs().max()):.3e}",
                    flush=True,
                )
            for tensor in (state, g, v, kv, b):
                ttnn.deallocate(tensor)

        # ------------------------------------------------------------------- rank-3 slice order
        neg_exp_a = dev(torch.rand(1, 1, 1, NUM_V_HEADS) * -8.0)
        neg_exp_a3 = dev(torch.zeros(1, 1, NUM_V_HEADS))
        ttnn.deallocate(neg_exp_a3)
        ab_host = torch.randn(1, 1, SEQ, 2 * AB_STRIDE) * 0.3

        def rank4_first():
            ab = dev(ab_host)
            b = ttnn.slice(ab, [0, 0, 0, 0], [1, 1, SEQ, NUM_V_HEADS])
            a = ttnn.slice(ab, [0, 0, 0, AB_STRIDE], [1, 1, SEQ, AB_STRIDE + NUM_V_HEADS])
            ttnn.deallocate(ab)
            beta = ttnn.sigmoid(b)
            ttnn.deallocate(b)
            soft = ttnn.softplus(a, beta=1.0, threshold=20.0)
            ttnn.deallocate(a)
            g = ttnn.multiply(soft, neg_exp_a)
            ttnn.deallocate(soft)
            beta3 = ttnn.reshape(beta, (1, SEQ, NUM_V_HEADS))
            g3 = ttnn.reshape(g, (1, SEQ, NUM_V_HEADS))
            if beta3.buffer_address() != beta.buffer_address():
                ttnn.deallocate(beta)
            if g3.buffer_address() != g.buffer_address():
                ttnn.deallocate(g)
            return beta3, g3

        def rank3_first():
            ab = dev(ab_host)
            ab3 = ttnn.reshape(ab, (1, SEQ, 2 * AB_STRIDE))
            if ab3.buffer_address() != ab.buffer_address():
                ttnn.deallocate(ab)
            b = ttnn.slice(ab3, [0, 0, 0], [1, SEQ, NUM_V_HEADS])
            a = ttnn.slice(ab3, [0, 0, AB_STRIDE], [1, SEQ, AB_STRIDE + NUM_V_HEADS])
            ttnn.deallocate(ab3)
            beta3 = ttnn.sigmoid(b)
            ttnn.deallocate(b)
            soft = ttnn.softplus(a, beta=1.0, threshold=20.0)
            ttnn.deallocate(a)
            g3 = ttnn.multiply(soft, ttnn.reshape(neg_exp_a, (1, 1, NUM_V_HEADS)))
            ttnn.deallocate(soft)
            return beta3, g3

        beta_a, g_a = keep(rank4_first, device)
        beta_b, g_b = keep(rank3_first, device)
        rank4_median, rank4_stdev = median_us(rank4_first, device, iters=9)
        rank3_median, rank3_stdev = median_us(rank3_first, device, iters=9)
        print(
            f"rank3 seq={SEQ} rank4_first_us={rank4_median:8.1f} ({rank4_stdev:5.1f}) "
            f"rank3_first_us={rank3_median:8.1f} ({rank3_stdev:5.1f}) "
            f"pcc_beta={pcc(beta_a, beta_b):.6f} pcc_g={pcc(g_a, g_b):.6f} "
            f"max_abs_diff={max(float((beta_a - beta_b).abs().max()), float((g_a - g_b).abs().max())):.3e}",
            flush=True,
        )
        ttnn.deallocate(neg_exp_a)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

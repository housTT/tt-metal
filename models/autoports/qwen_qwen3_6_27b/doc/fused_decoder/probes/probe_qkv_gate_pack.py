# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The two largest shared-LHS matmul pairs left: ``wqkv``/``wgate`` and ``in_proj_qkv``/``in_proj_z``.

``q_proj`` in the checkpoint emits ``[num_heads, 2 * head_dim]`` and the per-head second half is
the output gate, so HF ships q and the gate as *one* weight.  Stage 1 split them into ``wqkv``
(``[hidden, 8192]``) and ``wgate`` (``[hidden, 6144]``) because the two halves take different
paths: ``wqkv`` feeds ``nlp_create_qkv_heads*``, which needs exactly ``q|k|v`` and nothing else.

The skill's shared-LHS merge says two matmuls over the same activation should be one matmul over
``concat`` of the weights.  Here that trades one re-read of the ``[S, hidden]`` activation and one
dispatch for **two slices** of the merged output, and the slices are on the last axis of a TILE
tensor, i.e. real copies.  §3.8's MLP probe found the same trade going the other way, so the sign
is not obvious - this measures it, at the real prefill and decode shapes, both directions.

The ``linear_attention`` pair is the same question without the dtype objection §6 raises against
a four-way merge: ``in_proj_qkv`` (``[hidden, 10240]``) and ``in_proj_z`` (``[hidden, 6144]``) read
the same normed hidden state and are both bfloat16-weighted, so nothing but the cut stands in the
way.  Both pairs are measured the same way, at prefill and decode shapes.

    python .../probes/probe_qkv_gate_pack.py
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn

HIDDEN = 5120
N_HEADS, N_KV, HEAD_DIM = 24, 4, 256
QKV = (N_HEADS + 2 * N_KV) * HEAD_DIM  # 8192
GATE = N_HEADS * HEAD_DIM  # 6144
MERGED = QKV + GATE  # 14336
#: prefill tokens, and the tile-padded decode row count at batch 1 and batch 32.
ROWS = (2048, 32)


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def median_us(fn, device, iters):
    outs = fn()
    for out in outs:
        ttnn.deallocate(out)
    samples = []
    for _ in range(iters):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        outs = fn()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) * 1e6)
        for out in outs:
            ttnn.deallocate(out)
    return statistics.median(samples), statistics.stdev(samples)


#: ``(label, hidden, first width, second width, first dtype, second dtype)`` - the two pairs, each
#: read by one LHS, at the output dtypes the layer actually ships.  ``in_proj_qkv`` emits float32
#: because the causal conv carries its state in float32; a packed matmul has *one* output dtype, so
#: the packed form has to emit float32 for both, which is the trade this measures rather than a
#: bfloat16-everywhere comparison that the layer could not use.
PAIRS = (
    ("qkv_gate", HIDDEN, QKV, GATE, ttnn.bfloat16, ttnn.bfloat16),
    ("qkv_z", HIDDEN, 10240, 6144, ttnn.float32, ttnn.bfloat16),
)


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        torch.manual_seed(0)

        def dev(tensor, dtype=ttnn.bfloat16):
            return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        for label, hidden, first, second, first_dtype, second_dtype in PAIRS:
            # The packed form must emit one dtype for both halves; float32 when either half needs
            # it, which is the honest cost of the merge.
            packed_dtype = ttnn.float32 if ttnn.float32 in (first_dtype, second_dtype) else ttnn.bfloat16
            merged_width = first + second
            w_first = torch.randn(hidden, first) * 0.02
            w_second = torch.randn(hidden, second) * 0.02
            t_first = dev(w_first.reshape(1, 1, hidden, first))
            t_second = dev(w_second.reshape(1, 1, hidden, second))
            t_merged = dev(torch.cat([w_first, w_second], dim=-1).reshape(1, 1, hidden, merged_width))

            for rows in ROWS:
                x = dev(torch.randn(1, 1, rows, hidden))

                def split_form():
                    one = ttnn.linear(x, t_first, dtype=first_dtype, compute_kernel_config=cfg)
                    two = ttnn.linear(x, t_second, dtype=second_dtype, compute_kernel_config=cfg)
                    return one, two

                def packed_form():
                    both = ttnn.linear(x, t_merged, dtype=packed_dtype, compute_kernel_config=cfg)
                    one = ttnn.slice(both, [0, 0, 0, 0], [1, 1, rows, first])
                    two = ttnn.slice(both, [0, 0, 0, first], [1, 1, rows, merged_width])
                    ttnn.deallocate(both)
                    return one, two

                a_one, a_two = split_form()
                b_one, b_two = packed_form()
                agree_one = pcc(ttnn.to_torch(a_one).float(), ttnn.to_torch(b_one).float())
                agree_two = pcc(ttnn.to_torch(a_two).float(), ttnn.to_torch(b_two).float())
                for out in (a_one, a_two, b_one, b_two):
                    ttnn.deallocate(out)

                iters = 9 if rows > 256 else 25
                split_median, split_stdev = median_us(split_form, device, iters)
                packed_median, packed_stdev = median_us(packed_form, device, iters)
                names = {ttnn.float32: "fp32", ttnn.bfloat16: "bf16"}
                print(
                    f"{label} rows={rows:5d} split_us={split_median:9.1f} ({split_stdev:6.1f}) "
                    f"packed_us={packed_median:9.1f} ({packed_stdev:6.1f}) "
                    f"split_dtypes={names[first_dtype]}/{names[second_dtype]} "
                    f"packed_dtype={names[packed_dtype]} "
                    f"pcc_first={agree_one:.6f} pcc_second={agree_two:.6f}",
                    flush=True,
                )
                ttnn.deallocate(x)
            for tensor in (t_first, t_second, t_merged):
                ttnn.deallocate(tensor)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

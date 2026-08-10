# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The last shared-LHS matmul pair in ``full_attention``: ``wqkv`` and ``wgate``.

``q_proj`` in the checkpoint emits ``[num_heads, 2 * head_dim]`` and the per-head second half is
the output gate, so HF ships q and the gate as *one* weight.  Stage 1 split them into ``wqkv``
(``[hidden, 8192]``) and ``wgate`` (``[hidden, 6144]``) because the two halves take different
paths: ``wqkv`` feeds ``nlp_create_qkv_heads*``, which needs exactly ``q|k|v`` and nothing else.

The skill's shared-LHS merge says two matmuls over the same activation should be one matmul over
``concat`` of the weights.  Here that trades one re-read of the ``[S, hidden]`` activation and one
dispatch for **two slices** of the merged output, and the slices are on the last axis of a TILE
tensor, i.e. real copies.  §3.8's MLP probe found the same trade going the other way, so the sign
is not obvious - this measures it, at the real prefill and decode shapes, both directions.

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


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        torch.manual_seed(0)
        w_qkv = torch.randn(HIDDEN, QKV) * 0.02
        w_gate = torch.randn(HIDDEN, GATE) * 0.02
        w_merged = torch.cat([w_qkv, w_gate], dim=-1)

        def dev(tensor, dtype=ttnn.bfloat16):
            return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        t_qkv = dev(w_qkv.reshape(1, 1, HIDDEN, QKV))
        t_gate = dev(w_gate.reshape(1, 1, HIDDEN, GATE))
        t_merged = dev(w_merged.reshape(1, 1, HIDDEN, MERGED))

        for rows in ROWS:
            x = dev(torch.randn(1, 1, rows, HIDDEN))

            def split_form():
                qkv = ttnn.linear(x, t_qkv, dtype=ttnn.bfloat16, compute_kernel_config=cfg)
                gate = ttnn.linear(x, t_gate, dtype=ttnn.bfloat16, compute_kernel_config=cfg)
                return qkv, gate

            def packed_form():
                both = ttnn.linear(x, t_merged, dtype=ttnn.bfloat16, compute_kernel_config=cfg)
                qkv = ttnn.slice(both, [0, 0, 0, 0], [1, 1, rows, QKV])
                gate = ttnn.slice(both, [0, 0, 0, QKV], [1, 1, rows, MERGED])
                ttnn.deallocate(both)
                return qkv, gate

            a_qkv, a_gate = split_form()
            b_qkv, b_gate = packed_form()
            agree_qkv = pcc(ttnn.to_torch(a_qkv).float(), ttnn.to_torch(b_qkv).float())
            agree_gate = pcc(ttnn.to_torch(a_gate).float(), ttnn.to_torch(b_gate).float())
            for out in (a_qkv, a_gate, b_qkv, b_gate):
                ttnn.deallocate(out)

            iters = 9 if rows > 256 else 25
            split_median, split_stdev = median_us(split_form, device, iters)
            packed_median, packed_stdev = median_us(packed_form, device, iters)
            print(
                f"qkv_gate rows={rows:5d} split_us={split_median:9.1f} ({split_stdev:6.1f}) "
                f"packed_us={packed_median:9.1f} ({packed_stdev:6.1f}) "
                f"pcc_qkv={agree_qkv:.6f} pcc_gate={agree_gate:.6f}",
                flush=True,
            )
            ttnn.deallocate(x)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

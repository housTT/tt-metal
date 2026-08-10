# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP shapes: which fusing of the gate/up matmul, the SiLU and the slice wins.

Three variants at the real Qwen3.6-27B MLP shape (hidden 5120, intermediate 17408), for one
prefill chunk (2048 tokens) and one decode row (batch 1, tile-padded to 32):

``fused_slice_silu``   one ``[H, 2I]`` matmul, two slices, ``silu``, ``multiply``  (functional)
``fused_slice_act``    same, but the SiLU folded into the multiply's input activation
``split_act``          two ``[H, I]`` matmuls, SiLU folded into the gate matmul, ``multiply``

    python .../probes/probe_mlp_variants.py
"""

from __future__ import annotations

import time

import torch

import ttnn

HIDDEN = 5120
INTER = 17408


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        torch.manual_seed(0)
        wg = torch.randn(HIDDEN, INTER) * 0.02
        wu = torch.randn(HIDDEN, INTER) * 0.02

        def dev(t):
            return ttnn.from_torch(
                t.reshape(1, 1, *t.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

        t_gate, t_up = dev(wg), dev(wu)
        t_fused = dev(torch.cat([wg, wu], dim=1))

        for rows, tag in ((2048, "prefill-2048"), (32, "decode-32")):
            x = torch.randn(1, 1, rows, HIDDEN) * 0.5
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            ref = torch.nn.functional.silu(x @ wg) * (x @ wu)

            def v_fused_slice_silu():
                gu = ttnn.linear(tx, t_fused, dtype=ttnn.bfloat16, compute_kernel_config=cfg)
                g = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, rows, INTER])
                u = ttnn.slice(gu, [0, 0, 0, INTER], [1, 1, rows, 2 * INTER])
                ttnn.deallocate(gu)
                a = ttnn.silu(g)
                ttnn.deallocate(g)
                out = ttnn.multiply(a, u)
                ttnn.deallocate(a)
                ttnn.deallocate(u)
                return out

            def v_fused_slice_act():
                gu = ttnn.linear(tx, t_fused, dtype=ttnn.bfloat16, compute_kernel_config=cfg)
                g = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, rows, INTER])
                u = ttnn.slice(gu, [0, 0, 0, INTER], [1, 1, rows, 2 * INTER])
                ttnn.deallocate(gu)
                out = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
                ttnn.deallocate(g)
                ttnn.deallocate(u)
                return out

            def v_split_act():
                g = ttnn.linear(tx, t_gate, dtype=ttnn.bfloat16, compute_kernel_config=cfg, activation="silu")
                u = ttnn.linear(tx, t_up, dtype=ttnn.bfloat16, compute_kernel_config=cfg)
                out = ttnn.multiply(g, u)
                ttnn.deallocate(g)
                ttnn.deallocate(u)
                return out

            for name, fn in (
                ("fused_slice_silu", v_fused_slice_silu),
                ("fused_slice_act", v_fused_slice_act),
                ("split_act", v_split_act),
            ):
                out = fn()
                ttnn.deallocate(out)
                best = None
                for _ in range(5):
                    ttnn.synchronize_device(device)
                    t0 = time.perf_counter()
                    out = fn()
                    ttnn.synchronize_device(device)
                    best = min(best or 1e9, (time.perf_counter() - t0) * 1e3)
                    got = ttnn.to_torch(out).float()
                    ttnn.deallocate(out)
                print(f"{tag:14s} {name:18s} best_ms={best:8.3f} pcc={pcc(ref, got):.6f}", flush=True)
            ttnn.deallocate(tx)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

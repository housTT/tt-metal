# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Op-level contract + accuracy + latency probe for ``ttnn.transformer.chunk_gated_delta_rule``.

Model-free: builds Qwen3.6-27B ``linear_attention`` shapes (Nk=16, Nv=48, Dk=Dv=128) with
synthetic q/k/v/beta/g in the ranges the real layer produces, then compares the device op
against HF's ``torch_chunk_gated_delta_rule`` (float32) for the output and the final
recurrent state.

Two call shapes are probed:

``flat``   rank-3 token-major q/k/v ``[B, T, H*D]`` — the op L2-normalises q/k and folds the
           ``1/sqrt(Dk)`` scale in-kernel, and does the GQA head expansion itself.  Requires
           ``chunk_size == 32`` and ``T % 32 == 0``.
``split``  rank-4 ``[B, T, H, D]`` — the caller must L2-normalise q/k first.

    python .../probes/probe_chunk_gdr.py --seq 64 2048 --mode flat split
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

import ttnn

NK, NV, DK, DV = 16, 48, 128, 128


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    if a.norm() == 0 or b.norm() == 0:
        return float(a.norm() == b.norm())
    return float((a @ b) / (a.norm() * b.norm()))


def build_const_tiles(device, chunk: int):
    eye = torch.eye(chunk, dtype=torch.float32)
    tril = torch.tril(torch.ones(chunk, chunk, dtype=torch.float32))
    ones = torch.ones(chunk, chunk, dtype=torch.float32)
    ii = torch.arange(32).unsqueeze(1)
    jj = torch.arange(32).unsqueeze(0)
    lo_i, lo_j = ii < 16, jj < 16
    masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)

    def up(t):
        return ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    return up(eye), up(tril), up(ones), up(masks)


def l2norm_tt(x):
    inv = ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True), 1e-6))
    return ttnn.multiply(x, inv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, nargs="+", default=[64, 2048])
    ap.add_argument("--mode", nargs="+", default=["flat", "split"])
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        for mode in args.mode:
            chunk = 32 if mode == "flat" else 64
            const = build_const_tiles(device, chunk)
            for seq in args.seq:
                torch.manual_seed(0)
                q = torch.randn(1, seq, NK, DK)
                k = torch.randn(1, seq, NK, DK)
                v = torch.randn(1, seq, NV, DV) * 0.5
                beta = torch.sigmoid(torch.randn(1, seq, NV))
                g = -torch.nn.functional.softplus(torch.randn(1, seq, NV)) * 0.3
                state0 = torch.randn(1, NV, DK, DV) * 0.1

                rep = NV // NK
                o_ref, s_ref = torch_chunk_gated_delta_rule(
                    q.repeat_interleave(rep, dim=2).to(torch.float32),
                    k.repeat_interleave(rep, dim=2).to(torch.float32),
                    v.to(torch.float32),
                    g.to(torch.float32),
                    beta.to(torch.float32),
                    chunk_size=64,
                    initial_state=state0.to(torch.float32),
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )

                def dev(t, dtype=ttnn.float32):
                    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

                tg, tb, ts = dev(g), dev(beta), dev(state0)
                if mode == "flat":
                    tq = dev(q.reshape(1, seq, NK * DK), ttnn.bfloat16)
                    tk = dev(k.reshape(1, seq, NK * DK), ttnn.bfloat16)
                    tv = dev(v.reshape(1, seq, NV * DV), ttnn.bfloat16)
                else:
                    tq = l2norm_tt(dev(q))
                    tk = l2norm_tt(dev(k))
                    tv = dev(v)

                best = None
                for _ in range(args.iters):
                    ttnn.synchronize_device(device)
                    start = time.perf_counter()
                    o, s = ttnn.transformer.chunk_gated_delta_rule(
                        tq,
                        tk,
                        tv,
                        tg,
                        tb,
                        initial_state=ts,
                        output_final_state=True,
                        chunk_size=chunk,
                        eye=const[0],
                        tril=const[1],
                        ones=const[2],
                        masks=const[3],
                    )
                    ttnn.synchronize_device(device)
                    elapsed = (time.perf_counter() - start) * 1e3
                    best = elapsed if best is None else min(best, elapsed)
                    o_keep, s_keep = o, s
                o_t = ttnn.to_torch(o_keep).to(torch.float32).reshape(o_ref.shape)
                s_t = ttnn.to_torch(s_keep).to(torch.float32).reshape(s_ref.shape)
                print(
                    f"mode={mode:5s} chunk={chunk:3d} seq={seq:6d} "
                    f"o_pcc={pcc(o_ref, o_t):.6f} state_pcc={pcc(s_ref, s_t):.6f} best_wall_ms={best:.2f}",
                    flush=True,
                )
            for t in const:
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

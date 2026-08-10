# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Two output-path choices this stage had to make, measured end to end.

**1. ``chunk_gated_delta_rule``'s ``output_head_major``.**  The op can return its result
token-major (``[B, T, HV, V]`` ROW_MAJOR, after an internal untilize + permute) or head-major
(``[B*HV, T, V]`` TILE, free — the kernel already produces that).  Head-major skips the op's own
epilogue, but the consumer is a *per-head* RMS norm gated by ``silu(z)`` and then ``out_proj``,
and ``z`` and ``out_proj`` are both token-major-flat, so head-major buys the epilogue back as
relayouts of ``z`` and of the gated result.  Both whole paths are timed here, from the op call to
the flat ``[1, 1, T, value_dim]`` tensor ``out_proj`` consumes, and checked against each other.

**2. partial RoPE width.**  This model rotates 64 of 256 head channels, so the fused decoder
slices the rotary block out, runs ``rotary_embedding_hf`` on it and concatenates the passthrough
back.  Permuting head channels host-side (rotary pair ``c`` at ``c`` and ``c+128``, ``cos=1``,
``sin=0`` elsewhere) would make it one op over the whole 256-wide head with no slice or concat —
at 4x the rotated data.  Both are timed at the real prefill shapes.

    python .../probes/probe_output_paths.py
"""

from __future__ import annotations

import time

import torch

import ttnn

# linear_attention
NK, NV, DK, DV = 16, 48, 128, 128
VALUE_DIM = NV * DV
SEQ = 2048
CHUNK = 32
# full_attention
N_HEADS, N_KV, HEAD_DIM, ROTARY_DIM = 24, 4, 256, 64


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def bench(fn, device, iters=3):
    out = fn()
    ttnn.deallocate(out)
    best = None
    for _ in range(iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        best = min(best or 1e9, (time.perf_counter() - t0) * 1e3)
        got = ttnn.to_torch(out).float()
        ttnn.deallocate(out)
    return best, got


def const_tiles(device):
    eye = torch.eye(CHUNK)
    tril = torch.tril(torch.ones(CHUNK, CHUNK))
    ones = torch.ones(CHUNK, CHUNK)
    ii, jj = torch.arange(32).unsqueeze(1), torch.arange(32).unsqueeze(0)
    lo_i, lo_j = ii < 16, jj < 16
    masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)

    def up(t):
        return ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    return up(eye), up(tril), up(ones), up(masks)


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        torch.manual_seed(0)
        eye, tril, ones, masks = const_tiles(device)

        def dev(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)

        q = dev(torch.randn(1, SEQ, NK * DK), ttnn.bfloat16)
        k = dev(torch.randn(1, SEQ, NK * DK), ttnn.bfloat16)
        v = dev(torch.randn(1, SEQ, NV * DV) * 0.5, ttnn.bfloat16)
        beta = dev(torch.sigmoid(torch.randn(1, SEQ, NV)))
        g = dev(-torch.nn.functional.softplus(torch.randn(1, SEQ, NV)) * 0.3)
        state = dev(torch.randn(1, NV, DK, DV) * 0.1)
        z = dev(torch.randn(1, 1, SEQ, VALUE_DIM), ttnn.bfloat16)
        norm_w = torch.randn(DV) * 0.02
        w_head = dev(norm_w.reshape(1, 1, 1, DV), ttnn.bfloat16)
        group_mean = torch.zeros(VALUE_DIM, 64)
        scale_expand = torch.zeros(64, VALUE_DIM)
        for head in range(NV):
            group_mean[head * DV : (head + 1) * DV, head] = 1.0 / DV
            scale_expand[head, head * DV : (head + 1) * DV] = norm_w
        t_group = dev(group_mean.reshape(1, 1, VALUE_DIM, 64), ttnn.bfloat16)
        t_scale = dev(scale_expand.reshape(1, 1, 64, VALUE_DIM), ttnn.bfloat16)

        def gdn(head_major):
            return ttnn.transformer.chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=state,
                output_final_state=True,
                chunk_size=CHUNK,
                output_head_major=head_major,
                eye=eye,
                tril=tril,
                ones=ones,
                masks=masks,
                compute_kernel_config=cfg,
            )

        def token_major_path():
            """What the fused decoder ships: flat token-major + the group-reduction norm."""
            core, final = gdn(False)
            ttnn.deallocate(final)
            flat = ttnn.reshape(core, (1, 1, SEQ, VALUE_DIM))
            tiled = ttnn.to_layout(flat, ttnn.TILE_LAYOUT)
            core16 = ttnn.typecast(tiled, ttnn.bfloat16)
            ttnn.deallocate(tiled)
            squares = ttnn.multiply(core16, core16)
            mean_sq = ttnn.matmul(squares, t_group, dtype=ttnn.float32, compute_kernel_config=cfg)
            ttnn.deallocate(squares)
            inv = ttnn.rsqrt(ttnn.add(mean_sq, 1e-6))
            ttnn.deallocate(mean_sq)
            inv16 = ttnn.typecast(inv, ttnn.bfloat16)
            scale = ttnn.matmul(inv16, t_scale, dtype=ttnn.bfloat16, compute_kernel_config=cfg)
            ttnn.deallocate(inv16)
            normed = ttnn.multiply(core16, scale)
            ttnn.deallocate(scale)
            ttnn.deallocate(core16)
            out = ttnn.multiply(normed, z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
            ttnn.deallocate(normed)
            return out

        def head_major_path():
            """Head-major output + a plain per-head rms_norm, paying for z's relayout instead."""
            core, final = gdn(True)  # [B*HV, T, V] TILE
            ttnn.deallocate(final)
            core16 = ttnn.typecast(core, ttnn.bfloat16)
            ttnn.deallocate(core)
            normed = ttnn.rms_norm(core16, epsilon=1e-6, weight=w_head, compute_kernel_config=cfg)
            ttnn.deallocate(core16)
            z_heads = ttnn.reshape(z, (1, SEQ, NV, DV))
            z_bh = ttnn.permute(z_heads, (0, 2, 1, 3))
            ttnn.deallocate(z_heads)
            z_bh = ttnn.reshape(z_bh, (NV, SEQ, DV))
            gated = ttnn.multiply(normed, z_bh, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
            ttnn.deallocate(normed)
            ttnn.deallocate(z_bh)
            back = ttnn.reshape(gated, (1, NV, SEQ, DV))
            ttnn.deallocate(gated)
            token = ttnn.permute(back, (0, 2, 1, 3))
            ttnn.deallocate(back)
            out = ttnn.reshape(token, (1, 1, SEQ, VALUE_DIM))
            if out.buffer_address() != token.buffer_address():
                ttnn.deallocate(token)
            return out

        ms_a, out_a = bench(token_major_path, device)
        ms_b, out_b = bench(head_major_path, device)
        print(
            f"gdn_epilogue token_major ms={ms_a:8.2f} | head_major ms={ms_b:8.2f} | "
            f"pcc(token,head)={pcc(out_a.reshape(-1), out_b.reshape(-1)):.6f}",
            flush=True,
        )
        for t in (q, k, v, beta, g, state, z, w_head, t_group, t_scale, eye, tril, ones, masks):
            if t.is_allocated():
                ttnn.deallocate(t)

        # ------------------------------------------------------------------ partial RoPE width
        for heads, tag in ((N_HEADS, "q"), (N_KV, "k")):
            x = torch.randn(1, heads, SEQ, HEAD_DIM)
            angle = torch.randn(1, 1, SEQ, ROTARY_DIM // 2)
            cos_s = torch.cat([angle.cos(), angle.cos()], dim=-1)
            sin_s = torch.cat([angle.sin(), angle.sin()], dim=-1)
            tx = dev(x, ttnn.bfloat16)
            tc, ts = dev(cos_s, ttnn.bfloat16), dev(sin_s, ttnn.bfloat16)

            def narrow():
                rot = ttnn.slice(tx, [0, 0, 0, 0], [1, heads, SEQ, ROTARY_DIM])
                emb = ttnn.experimental.rotary_embedding_hf(rot, tc, ts, compute_kernel_config=cfg)
                ttnn.deallocate(rot)
                keep = ttnn.slice(tx, [0, 0, 0, ROTARY_DIM], [1, heads, SEQ, HEAD_DIM])
                out = ttnn.concat([emb, keep], dim=-1)
                ttnn.deallocate(emb)
                ttnn.deallocate(keep)
                return out

            # Permuted-channel variant: rotary pair c lands at (c, c+128); cos=1/sin=0 elsewhere.
            cos_full = torch.ones(1, 1, SEQ, HEAD_DIM)
            sin_full = torch.zeros(1, 1, SEQ, HEAD_DIM)
            cos_full[..., :32] = angle.cos()
            cos_full[..., 128:160] = angle.cos()
            sin_full[..., :32] = angle.sin()
            sin_full[..., 128:160] = angle.sin()
            tcf, tsf = dev(cos_full, ttnn.bfloat16), dev(sin_full, ttnn.bfloat16)

            def wide():
                return ttnn.experimental.rotary_embedding_hf(tx, tcf, tsf, compute_kernel_config=cfg)

            ms_n, _ = bench(narrow, device, iters=5)
            ms_w, _ = bench(wide, device, iters=5)
            print(
                f"rope_{tag}({heads} heads) slice+64wide+concat ms={ms_n:7.3f} | " f"permuted 256wide ms={ms_w:7.3f}",
                flush=True,
            )
            for t in (tx, tc, ts, tcf, tsf):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

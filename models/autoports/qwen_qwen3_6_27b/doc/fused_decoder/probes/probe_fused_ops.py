# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model-free probes for the dedicated TTNN ops the fused decoder wants to use.

Each probe answers one question before the op is wired into the layer:

* ``swiglu``     - which half does ``ttnn.swiglu`` apply SiLU to?
* ``rope``       - can ``ttnn.experimental.rotary_embedding_hf`` express Qwen3.5's *partial*
                   rotary embedding with a host-side head-channel permutation?
* ``gdn``        - does ``ttnn.transformer.gated_delta_attn_seq`` reproduce HF's
                   ``torch_chunk_gated_delta_rule``?
* ``triinv``     - how many cores does the batched 32x32 matmul of the recursive triangular
                   inverse actually get, and does a different rank/memory config help?

Run:  python doc/fused_decoder/probes/probe_fused_ops.py [name ...]
"""

from __future__ import annotations

import sys
import time

import torch
import ttnn


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def tt(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# --------------------------------------------------------------------------- swiglu


def probe_swiglu(device):
    x = torch.randn(1, 1, 64, 256)
    out = ttnn.to_torch(ttnn.swiglu(tt(x, device))).float()
    a, b = x[..., :128], x[..., 128:]
    print(f"  swiglu vs first*silu(second): pcc={pcc(a * torch.nn.functional.silu(b), out):.6f}")
    print(f"  swiglu vs silu(first)*second: pcc={pcc(torch.nn.functional.silu(a) * b, out):.6f}")


# ----------------------------------------------------------------------------- rope

HEAD_DIM = 256
ROT_DIM = 64


def rope_permutation(head_dim=HEAD_DIM, rot_dim=ROT_DIM):
    """Old-channel index for each new channel (see fused_decoder.rope_channel_permutation)."""
    half = rot_dim // 2
    mid = head_dim // 2
    return (
        list(range(0, half))
        + list(range(rot_dim, rot_dim + (mid - half)))
        + list(range(half, rot_dim))
        + list(range(rot_dim + (mid - half), head_dim))
    )


def torch_partial_rope(x, cos, sin, rot_dim=ROT_DIM):
    rot, passthrough = x[..., :rot_dim], x[..., rot_dim:]
    half = rot_dim // 2
    rotated = torch.cat([-rot[..., half:], rot[..., :half]], dim=-1)
    return torch.cat([rot * cos + rotated * sin, passthrough], dim=-1)


def expand_rope_mats(cos, sin, head_dim=HEAD_DIM, rot_dim=ROT_DIM):
    """[..., rot_dim] -> [..., head_dim] cos/sin for the permuted full-width rotate-half."""
    half, mid = rot_dim // 2, head_dim // 2
    lead = cos.shape[:-1]
    cos_full = torch.ones(*lead, head_dim, dtype=cos.dtype)
    sin_full = torch.zeros(*lead, head_dim, dtype=sin.dtype)
    cos_full[..., :half] = cos[..., :half]
    cos_full[..., mid : mid + half] = cos[..., half:rot_dim]
    sin_full[..., :half] = sin[..., :half]
    sin_full[..., mid : mid + half] = sin[..., half:rot_dim]
    return cos_full, sin_full


def probe_rope(device):
    perm = rope_permutation()
    heads, seq = 8, 128
    x = torch.randn(1, heads, seq, HEAD_DIM)
    cos = torch.randn(1, 1, seq, ROT_DIM).clamp(-1, 1)
    sin = torch.randn(1, 1, seq, ROT_DIM).clamp(-1, 1)
    golden = torch_partial_rope(x, cos, sin)[..., perm]

    cos_full, sin_full = expand_rope_mats(cos, sin)
    out = ttnn.experimental.rotary_embedding_hf(
        tt(x[..., perm].contiguous(), device), tt(cos_full, device), tt(sin_full, device), is_decode_mode=False
    )
    print(f"  prefill permuted rotary_embedding_hf: pcc={pcc(golden, ttnn.to_torch(out).float()):.6f}")

    # decode: [1, batch, heads, head_dim], cos/sin [1, batch, 1, head_dim]
    batch = 8
    xd = torch.randn(1, batch, 32, HEAD_DIM)
    cosd = torch.randn(1, batch, 1, ROT_DIM).clamp(-1, 1)
    sind = torch.randn(1, batch, 1, ROT_DIM).clamp(-1, 1)
    golden_d = torch_partial_rope(xd, cosd, sind)[..., perm]
    cosd_full, sind_full = expand_rope_mats(cosd, sind)
    for decode_mode in (True, False):
        try:
            out = ttnn.experimental.rotary_embedding_hf(
                tt(xd[..., perm].contiguous(), device),
                tt(cosd_full, device),
                tt(sind_full, device),
                is_decode_mode=decode_mode,
            )
            got = ttnn.to_torch(out).float()
            print(f"  decode is_decode_mode={decode_mode}: pcc={pcc(golden_d, got):.6f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  decode is_decode_mode={decode_mode}: FAILED {type(exc).__name__}: {str(exc)[:200]}")


# ------------------------------------------------------------------------------ gdn


def probe_gdn(device):
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
        chunk_gated_delta_rule_seq_adapter,
        create_chunk_masks_seq,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    torch.manual_seed(0)
    batch, seq, heads, dk, dv = 1, 256, 4, 128, 128
    q = torch.randn(batch, seq, heads, dk)
    k = torch.randn(batch, seq, heads, dk)
    v = torch.randn(batch, seq, heads, dv)
    beta = torch.rand(batch, seq, heads)
    g = -torch.rand(batch, seq, heads) * 0.05
    state0 = torch.randn(batch, heads, dk, dv) * 0.1

    golden, golden_state = torch_chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=64, initial_state=state0.clone(),
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    masks = create_chunk_masks_seq(128, device)
    out, state = chunk_gated_delta_rule_seq_adapter(
        tt(q, device, ttnn.float32), tt(k, device, ttnn.float32), tt(v, device, ttnn.float32),
        tt(beta, device, ttnn.float32), tt(g, device, ttnn.float32),
        chunk_size=128, initial_state=tt(state0, device, ttnn.float32),
        device=device, cached_masks=masks,
    )
    print(f"  gated_delta_attn_seq out   pcc={pcc(golden, ttnn.to_torch(out).float()):.6f}")
    print(f"  gated_delta_attn_seq state pcc={pcc(golden_state, ttnn.to_torch(state).float()):.6f}")


# --------------------------------------------------------------------------- tri inv


def probe_triinv(device):
    """How wide does a batched 32x32x32 matmul actually run?"""
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    for shape in ([768, 1, 32, 32], [768, 32, 32], [6144, 1, 32, 32], [24, 32, 32, 32]):
        a = tt(torch.randn(*shape), device, ttnn.float32)
        b = tt(torch.randn(*shape), device, ttnn.float32)
        ttnn.matmul(a, b, dtype=ttnn.float32, compute_kernel_config=cfg)
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(5):
            out = ttnn.matmul(a, b, dtype=ttnn.float32, compute_kernel_config=cfg)
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)
        print(f"  matmul {shape}: {1e3 * (time.perf_counter() - start) / 5:.3f} ms/call")
        ttnn.deallocate(a)
        ttnn.deallocate(b)


PROBES = {"swiglu": probe_swiglu, "rope": probe_rope, "gdn": probe_gdn, "triinv": probe_triinv}


def main():
    names = sys.argv[1:] or list(PROBES)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        for name in names:
            print(f"== {name}", flush=True)
            PROBES[name](device)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

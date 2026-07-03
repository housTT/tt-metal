# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hand-written TT-NN modules for the DeepSeek-V4 port (Checkpoint 2, Path B).

Correctness-first: bf16 / DRAM-interleaved / HiFi4, no sharding or trace yet
(those are Checkpoint 3). Each function is torch-in / torch-out with the ttnn work
inside, matching the per-module PCC test idiom. Math follows docs/HF_REFERENCE_SPEC.md.

No simulator/hardware conditionals (GOAL rule 2). Runs identically on Blackhole
silicon and (if ever installed) libttsim.
"""
from __future__ import annotations

import torch

import ttnn

HIFI4 = ttnn.WormholeComputeKernelConfig  # placeholder; set per-op below


def _to_dev(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def linear(x: torch.Tensor, weight: torch.Tensor, device, bias: torch.Tensor | None = None, weight_dtype=ttnn.bfloat16):
    """TT-NN dense linear. `weight` is nn.Linear layout [out, in]; y = x @ weightᵀ (+bias)."""
    tx = _to_dev(x, device)
    tw = _to_dev(weight.t().contiguous(), device, dtype=weight_dtype)
    tb = _to_dev(bias, device) if bias is not None else None
    ty = ttnn.linear(tx, tw, bias=tb)
    out = ttnn.to_torch(ty)
    for t in (tx, tw, ty):
        ttnn.deallocate(t)
    if tb is not None:
        ttnn.deallocate(tb)
    return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, device, eps: float = 1e-6) -> torch.Tensor:
    """Weighted RMSNorm over the last dim (weight=None -> unweighted, V4 q_b_norm).

    y = x * rsqrt(mean(x^2, -1) + eps) * weight
    """
    tx = _to_dev(x, device)
    w = weight if weight is not None else torch.ones(x.shape[-1], dtype=torch.float32)
    tw = _to_dev(w, device)
    ty = ttnn.rms_norm(tx, epsilon=eps, weight=tw)
    out = ttnn.to_torch(ty)
    ttnn.deallocate(tx)
    ttnn.deallocate(ty)
    return out


def clamped_swiglu_mlp(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    device,
    limit: float = 10.0,
    weight_dtype=ttnn.bfloat16,
) -> torch.Tensor:
    """DeepSeek-V4 clamped SwiGLU (shared expert / DeepseekV4MLP).

    gate = clamp(x @ gate_wᵀ, max=limit)          # upper-clamp only
    up   = clamp(x @ up_wᵀ, min=-limit, max=limit) # two-sided
    y    = (silu(gate) * up) @ down_wᵀ
    (weights are nn.Linear [out,in]; matmul uses xᵀ convention via transpose.)
    `weight_dtype` selects the on-device weight precision (Checkpoint 3 precision sweep).
    """
    tx = _to_dev(x, device)
    tg = _to_dev(gate_w.t().contiguous(), device, dtype=weight_dtype)
    tu = _to_dev(up_w.t().contiguous(), device, dtype=weight_dtype)
    td = _to_dev(down_w.t().contiguous(), device, dtype=weight_dtype)

    gate = ttnn.linear(tx, tg)
    up = ttnn.linear(tx, tu)
    gate = ttnn.clamp(gate, max=limit)
    up = ttnn.clamp(up, min=-limit, max=limit)
    act = ttnn.multiply(ttnn.silu(gate), up)
    y = ttnn.linear(act, td)
    out = ttnn.to_torch(y)
    for t in (tx, tg, tu, td, gate, up, act, y):
        ttnn.deallocate(t)
    return out


def grouped_linear(x: torch.Tensor, weight: torch.Tensor, n_groups: int, device) -> torch.Tensor:
    """DeepseekV4GroupedLinear (o_lora block-diagonal bmm), HF ref lines 326-332.

    weight is nn.Linear [out = g*rank, in_per_group]; input x is [..., g*in_per_group].
      w = weight.view(g, rank, in_per_group).transpose(1,2)   -> [g, in_per_group, rank]
      x = x.reshape(-1, g, in_per_group).transpose(0,1)       -> [g, N, in_per_group]
      y = bmm(x, w).transpose(0,1)                            -> [N, g, rank]
    Returns [..., g, rank].
    """
    input_shape = x.shape[:-1]
    in_per_group = weight.shape[1]
    rank = weight.shape[0] // n_groups
    w = weight.view(n_groups, rank, in_per_group).transpose(1, 2).contiguous()  # [g, ipg, rank]
    xg = x.reshape(-1, n_groups, in_per_group).transpose(0, 1).contiguous()  # [g, N, ipg]

    tw = _to_dev(w, device)
    tx = _to_dev(xg, device)
    ty = ttnn.matmul(tx, tw)  # [g, N, rank]
    y = ttnn.to_torch(ty).transpose(0, 1)  # [N, g, rank]
    ttnn.deallocate(tw)
    ttnn.deallocate(tx)
    ttnn.deallocate(ty)
    return y.reshape(*input_shape, n_groups, rank)


def moe_routed_experts(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,  # [E, 2*interm, hidden]  (HF DeepseekV4Experts layout)
    down_proj: torch.Tensor,  # [E, hidden, interm]
    indices: torch.Tensor,  # [N, top_k] selected expert ids
    weights: torch.Tensor,  # [N, top_k] combine weights (already scaled/normalized)
    device,
    limit: float = 10.0,
) -> torch.Tensor:
    """MoE routed-expert compute (clamped SwiGLU per selected expert, weighted sum).

    For each token n and selected expert e=indices[n,k]:
      g,u = split(x_n @ gate_up[e]ᵀ ... )   -> here gate_up is [2*interm, hidden]
      out_n += weights[n,k] * (silu(clamp(g,max))*clamp(u,±)) @ down[e]ᵀ
    Matmuls run on device (ttnn); the per-token expert gather/scatter is host-side
    (data-dependent indexing — the documented torch-fallback pattern, TTNN-bringup §3.1).
    """
    N = x.shape[0]
    hidden = x.shape[-1]
    interm = down_proj.shape[-1]
    out = torch.zeros(N, hidden, dtype=torch.float32)
    tx_full = _to_dev(x, device)  # [N, hidden]
    for k in range(indices.shape[1]):
        # group tokens is data-dependent; process per token-slot k across all experts naively
        for n in range(N):
            e = int(indices[n, k])
            gu = gate_up_proj[e]  # [2*interm, hidden]
            dn = down_proj[e]  # [hidden, interm]
            txn = ttnn.from_torch(x[n : n + 1], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            tgu = _to_dev(gu.t().contiguous(), device)
            tdn = _to_dev(dn.t().contiguous(), device)
            proj = ttnn.to_torch(ttnn.linear(txn, tgu))  # [1, 2*interm]
            g, u = proj[..., :interm], proj[..., interm:]
            g = g.clamp(max=limit)
            u = u.clamp(min=-limit, max=limit)
            act = torch.nn.functional.silu(g) * u
            tact = ttnn.from_torch(act, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            y = ttnn.to_torch(ttnn.linear(tact, tdn))[0]  # [hidden]
            out[n] += float(weights[n, k]) * y.float()
    return out


def hyperconnection(streams: torch.Tensor, hc, device):
    """mHC HyperConnection (V4's defining novelty), HF ref lines 872-948.

    Returns (post, comb, collapsed). The heavy `input_norm` + `fn` matmul run on
    device (ttnn); the tiny fixed hc×hc Sinkhorn / sigmoid / softmax control math runs
    on host (non-tile-friendly size-`hc` dims; documented torch-fallback, GOAL anti-BS
    rule). `streams` is [B,S,hc,H].
    """
    B, S, HC, Hd = streams.shape
    eps = hc.hc_eps
    fn_w = hc.fn.data  # [(2+hc)*hc, hc*H]  (raw Parameter, not an nn.Linear)
    base = hc.base.data
    scale = hc.scale.data

    flat = streams.reshape(B, S, HC * Hd)
    # input_norm is an UnweightedRMSNorm over hc*H -> device
    normed = rms_norm(flat, None, device, eps=getattr(hc.input_norm, "variance_epsilon", 1e-6))
    # fn linear on device
    tnorm = _to_dev(normed, device)
    tfn = _to_dev(fn_w.t().contiguous(), device)
    proj = ttnn.to_torch(ttnn.linear(tnorm, tfn)).float()  # [B,S,(2+hc)*hc]
    ttnn.deallocate(tnorm)
    ttnn.deallocate(tfn)

    # --- tiny host-side control math (Sinkhorn etc.) ---
    pre_w, post_w, comb_w = proj.split([HC, HC, HC * HC], dim=-1)
    pre_b, post_b, comb_b = base.float().split([HC, HC, HC * HC])
    pre_scale, post_scale, comb_scale = scale.float().unbind(0)
    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(B, S, HC, HC) * comb_scale + comb_b.view(HC, HC)
    comb = torch.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(hc.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    collapsed = (pre.unsqueeze(-1) * streams.float()).sum(dim=2)  # [B,S,H]
    return post, comb, collapsed


def hyper_head(streams: torch.Tensor, hh, device):
    """mHC final collapse (DeepseekV4HyperHead, HF ref 951-967). streams [B,S,hc,H] -> [B,S,H].
    input_norm + `hc_fn` linear on device; the tiny sigmoid mix on host."""
    B, S, HC, Hd = streams.shape
    eps = getattr(hh, "hc_eps", 1e-6)
    flat = streams.reshape(B, S, HC * Hd)
    normed = rms_norm(flat, None, device, eps=getattr(hh.input_norm, "variance_epsilon", 1e-6))
    mixes = linear(normed, hh.hc_fn.data, device).float()  # [B,S,hc]
    pre = torch.sigmoid(mixes * hh.hc_scale.data.float() + hh.hc_base.data.float()) + eps
    return (pre.unsqueeze(-1) * streams.float()).sum(dim=2)  # [B,S,H]


def sqrtsoftplus_router_scores(x: torch.Tensor, gate_w: torch.Tensor, device) -> torch.Tensor:
    """MoE gate score front-end (the V4-specific part): scores = sqrt(softplus(x @ gate_wᵀ)).

    (Top-k selection + bias + normalization live in the router; this validates the
    score transform that the generalized-MoE gate applies externally.)
    """
    tx = _to_dev(x, device)
    tw = _to_dev(gate_w.t().contiguous(), device)
    logits = ttnn.linear(tx, tw)
    scores = ttnn.sqrt(ttnn.softplus(logits))
    out = ttnn.to_torch(scores)
    for t in (tx, tw, logits, scores):
        ttnn.deallocate(t)
    return out

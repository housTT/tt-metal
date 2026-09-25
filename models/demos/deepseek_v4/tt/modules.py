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


def _is_mesh(device):
    """True if `device` is a multi-device MeshDevice (needs mesh mappers/composers)."""
    n = getattr(device, "get_num_devices", None)
    try:
        return n is not None and n() > 1
    except Exception:
        return False


def _to_dev(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Host tensor -> device. On a MeshDevice the tensor is REPLICATED across all chips
    (each chip computes redundantly with resident replicated weights); on a single device
    it's a plain upload."""
    mapper = ttnn.ReplicateTensorToMesh(device) if _is_mesh(device) else None
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
    )


def _from_dev(ty, device):
    """Device -> host. On a MeshDevice every chip holds an identical (replicated) copy of the
    result, so concat along dim 0 and keep the first 1/num_devices rows (chip 0's answer);
    on a single device it's a plain read."""
    if _is_mesh(device):
        o = ttnn.to_torch(ty, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        return o[: o.shape[0] // device.get_num_devices()]
    return ttnn.to_torch(ty)


def linear(x: torch.Tensor, weight: torch.Tensor, device, bias: torch.Tensor | None = None, weight_dtype=ttnn.bfloat16):
    """TT-NN dense linear. `weight` is nn.Linear layout [out, in]; y = x @ weightᵀ (+bias)."""
    tx = _to_dev(x, device)
    tw = _to_dev(weight.t().contiguous(), device, dtype=weight_dtype)
    tb = _to_dev(bias, device) if bias is not None else None
    ty = ttnn.linear(tx, tw, bias=tb)
    out = _from_dev(ty, device)
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
    out = _from_dev(ty, device)
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
    out = _from_dev(y, device)
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
    y = _from_dev(ty, device).transpose(0, 1)  # [N, g, rank]
    ttnn.deallocate(tw)
    ttnn.deallocate(tx)
    ttnn.deallocate(ty)
    return y.reshape(*input_shape, n_groups, rank)


def fused_experts(x, gate_up_T, down_T, interm, device, limit=10.0):
    """Clamped-SwiGLU expert with a FUSED gate+up matmul. gate_up_T [H, 2I] and down_T [I, H]
    are pre-transposed+contiguous (cached), so this does 2 matmuls + no per-call transpose
    (vs 3 matmuls + 3 transposes for the generic MLP). x [n, H] -> [n, H]."""
    tx = _to_dev(x, device)
    tgu = _to_dev(gate_up_T, device)
    tdn = _to_dev(down_T, device)
    gu = ttnn.matmul(tx, tgu)  # [n, 2I]
    gate = ttnn.clamp(gu[..., :interm], max=limit)
    up = ttnn.clamp(gu[..., interm:], min=-limit, max=limit)
    act = ttnn.multiply(ttnn.silu(gate), up)  # [n, I]
    y = ttnn.matmul(act, tdn)  # [n, H]
    out = _from_dev(y, device)
    for t in (tx, tgu, tdn, gu, gate, up, act, y):
        ttnn.deallocate(t)
    return out


def grouped_experts_decode(x_row, gu_list, dn_list, interm, device, limit=10.0, weight_dtype=ttnn.bfloat16):
    """Batched decode MoE (N==1): run E experts on the single token via ONE grouped matmul each.

    (a) The E selected experts' weights are STACKED (gu [E,H,2I], dn [E,I,H]) and uploaded ONCE
    each, then a batched matmul runs all E experts together — collapsing the ~12 per-layer expert
    ops (upload+matmul ×E×2) into 2 uploads + 2 grouped matmuls. (b) Weights upload as bfloat8_b
    (½ the DMA + tilize of bf16). Returns per-expert outputs to HOST [E, H]; the caller does the
    fp32 weighted sum (on-device bf16 accumulation flips the argmax — must stay host-fp32).

    x_row: host [1, H]. gu_list: E host [H, 2I]. dn_list: E host [I, H]."""
    E = len(gu_list)
    GU = torch.stack(gu_list, 0)  # [E, H, 2I]
    DN = torch.stack(dn_list, 0)  # [E, I, H]
    tx = _to_dev(x_row.reshape(1, -1).expand(E, -1).reshape(E, 1, -1).contiguous(), device)  # [E,1,H]
    tgu = _to_dev(GU, device, dtype=weight_dtype)  # [E,H,2I]
    tdn = _to_dev(DN, device, dtype=weight_dtype)  # [E,I,H]
    gu = ttnn.matmul(tx, tgu)  # [E,1,2I]
    gate = ttnn.clamp(gu[..., :interm], max=limit)
    up = ttnn.clamp(gu[..., interm:], min=-limit, max=limit)
    act = ttnn.multiply(ttnn.silu(gate), up)  # [E,1,I]
    y = ttnn.matmul(act, tdn)  # [E,1,H]
    out = _from_dev(y, device)  # host [E,1,H] (single-chip decode)
    for t in (tx, tgu, tdn, gu, gate, up, act, y):
        ttnn.deallocate(t)
    return out.reshape(E, -1)  # [E, H]


def fused_expert_ondevice(tx, gate_up_T, down_T, interm, device, limit=10.0):
    """One clamped-SwiGLU expert that RETURNS ITS RESULT ON DEVICE (no read-back).

    `tx` is the activation ALREADY on device [n, H]; `gate_up_T` [H, 2I] and `down_T` [I, H] are
    host weights (uploaded + freed here). Returns a device tensor [n, H] — the caller accumulates
    on device and reads back ONCE per layer, instead of a host round-trip per expert (the ~258
    per-token round-trips that dominate decode; see tt/model.py::sparse_moe_streaming)."""
    tgu = _to_dev(gate_up_T, device)
    tdn = _to_dev(down_T, device)
    gu = ttnn.matmul(tx, tgu)  # [n, 2I]
    gate = ttnn.clamp(gu[..., :interm], max=limit)
    up = ttnn.clamp(gu[..., interm:], min=-limit, max=limit)
    act = ttnn.multiply(ttnn.silu(gate), up)  # [n, I]
    y = ttnn.matmul(act, tdn)  # [n, H] on device
    for t in (tgu, tdn, gu, gate, up, act):
        ttnn.deallocate(t)
    return y


def linear_dev(x, tw_T, device):
    """Linear with a RESIDENT device weight `tw_T` [in, out] (pre-transposed, on device). Only
    the activation crosses host->device. y = x @ tw_T."""
    tx = _to_dev(x, device)
    ty = ttnn.matmul(tx, tw_T)
    out = _from_dev(ty, device)
    ttnn.deallocate(tx)
    ttnn.deallocate(ty)
    return out


def grouped_linear_dev(x, tw_dev, n_groups, device):
    """DeepseekV4GroupedLinear with RESIDENT device weight `tw_dev` [g, in_per_group, rank].
    x is [..., g*in_per_group]. Returns [..., g, rank]."""
    input_shape = x.shape[:-1]
    ipg = tw_dev.shape[1]
    xg = x.reshape(-1, n_groups, ipg).transpose(0, 1).contiguous()  # [g, N, ipg]
    tx = _to_dev(xg, device)
    ty = ttnn.matmul(tx, tw_dev)  # [g, N, rank]
    y = _from_dev(ty, device).transpose(0, 1)  # [N, g, rank]
    ttnn.deallocate(tx)
    ttnn.deallocate(ty)
    return y.reshape(*input_shape, n_groups, tw_dev.shape[-1])


def fused_experts_dev(x, tgu, tdn, interm, device, limit=10.0):
    """Clamped-SwiGLU expert using RESIDENT device weights `tgu` [H,2I], `tdn` [I,H] (ttnn
    tensors already on device — NOT re-uploaded per token). Only the small activation `x`
    crosses host->device. 2 matmuls + no weight transfer."""
    tx = _to_dev(x, device)
    gu = ttnn.matmul(tx, tgu)  # [n, 2I]
    gate = ttnn.clamp(gu[..., :interm], max=limit)
    up = ttnn.clamp(gu[..., interm:], min=-limit, max=limit)
    act = ttnn.multiply(ttnn.silu(gate), up)  # [n, I]
    y = ttnn.matmul(act, tdn)  # [n, H]
    out = _from_dev(y, device)
    for t in (tx, gu, gate, up, act, y):
        ttnn.deallocate(t)
    return out


def _tp_reduce_read(y, device):
    """Read a row-parallel (down-projection) partial result off the mesh and SUM the per-chip
    partials — the tensor-parallel reduce. On a single device it's a plain read."""
    if _is_mesh(device):
        C = device.get_num_devices()
        parts = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))  # [C*N, H]
        N = parts.shape[0] // C
        return parts.reshape(C, N, parts.shape[-1]).sum(0)  # [N, H]
    return ttnn.to_torch(y)


def swiglu_sharded(x, tgate, tup, tdown, device, limit=10.0):
    """Clamped-SwiGLU with tensor-parallel-SHARDED RESIDENT weights: `tgate`/`tup` [H, I/C]
    (column-sharded) and `tdown` [I/C, H] (row-sharded). Each chip computes its I-shard; the
    row-parallel down output is summed across chips (TP reduce). x [N, H] -> [N, H]."""
    tx = _to_dev(x, device)  # replicated
    g = ttnn.clamp(ttnn.matmul(tx, tgate), max=limit)  # [N, I/C] per chip
    u = ttnn.clamp(ttnn.matmul(tx, tup), min=-limit, max=limit)
    act = ttnn.multiply(ttnn.silu(g), u)
    y = ttnn.matmul(act, tdown)  # [N, H] partial per chip
    out = _tp_reduce_read(y, device)
    for t in (tx, g, u, act, y):
        ttnn.deallocate(t)
    return out


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
    proj = _from_dev(ttnn.linear(tnorm, tfn), device).float()  # [B,S,(2+hc)*hc]
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
    out = _from_dev(scores, device)
    for t in (tx, tw, logits, scores):
        ttnn.deallocate(t)
    return out

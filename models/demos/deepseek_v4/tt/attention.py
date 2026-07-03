# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Full MLA-v4 attention (Path B), ported from HF DeepseekV4Attention (docs/HF_REFERENCE_SPEC.md §1).

All learned projections run on device (ttnn): q_a_proj, q_a_norm, q_b_proj, q_b_norm,
kv_proj, kv_norm, grouped o_a_proj (o_lora), o_b_proj. The small per-head control math —
interleaved partial RoPE, the sink softmax, and the conjugate −sin un-rotation — runs on
host (documented torch-fallback, GOAL anti-BS rule + TTNN-bringup §3.1; these are tiny
elementwise/softmax ops on [B,H,S,d], negligible FLOPs, and fiddly to tile-align).

For `sliding_attention` layers with seq_len ≤ sliding_window (the reduced-config demo),
there is no compressor and the sliding cache does not evict, so this reduces to standard
causal attention with attention sinks. CSA/HCA compressors + lightning indexer are ported
and PCC-validated separately in tt/compressors.py.
"""
from __future__ import annotations

import torch

from models.demos.deepseek_v4.tt import modules as M


def _rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x, cos, sin, unsqueeze_dim=1):
    """V4 interleaved partial RoPE on the trailing rope slice (host fallback)."""
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rd = cos.shape[-1]
    nope, rope = x[..., :-rd], x[..., -rd:]
    rotated = ((rope.float() * cos) + (_rotate_half(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


def mla_attention(hidden_states, attn, cos, sin, causal_mask, cfg, device, weight_dtype=None):
    """hidden_states: [B,S,H] (already input_layernorm'd). `attn` = HF DeepseekV4Attention
    (source of weights + `sinks`). Returns [B,S,H]. `weight_dtype` optionally sets on-device
    projection precision (Checkpoint-3 sweep); None -> bf16."""
    import ttnn

    wdt = weight_dtype or ttnn.bfloat16
    B, S, H = hidden_states.shape
    nh, hd = cfg.num_attention_heads, cfg.head_dim
    scaling = attn.scaling

    # --- Q path (device projections) ---
    q_res = M.linear(hidden_states, attn.q_a_proj.weight.data, device, weight_dtype=wdt)  # [B,S,q_lora]
    q_res = M.rms_norm(q_res, attn.q_a_norm.weight.data, device, eps=cfg.rms_norm_eps)
    q = M.linear(q_res, attn.q_b_proj.weight.data, device, weight_dtype=wdt)  # [B,S,nh*hd]
    q = q.view(B, S, nh, hd).transpose(1, 2)  # [B,nh,S,hd]
    q = M.rms_norm(q, None, device, eps=cfg.rms_norm_eps)  # unweighted q_b_norm over hd
    q = apply_rope(q, cos, sin)

    # --- KV path (single shared head, K==V) ---
    kv = M.linear(hidden_states, attn.kv_proj.weight.data, device, weight_dtype=wdt)  # [B,S,hd]
    kv = M.rms_norm(kv, attn.kv_norm.weight.data, device, eps=cfg.rms_norm_eps)
    kv = kv.view(B, S, 1, hd).transpose(1, 2)  # [B,1,S,hd]
    kv = apply_rope(kv, cos, sin)

    # --- attention core with sinks (host fallback: tiny softmax over [B,nh,S,S+1]) ---
    k = kv.expand(B, nh, S, hd)  # repeat_kv 1 -> nh
    v = k
    aw = torch.matmul(q.float(), k.float().transpose(2, 3)) * scaling  # [B,nh,S,S]
    if causal_mask is not None:
        aw = aw + causal_mask[..., :S, :S].float()
    sinks = attn.sinks.reshape(1, -1, 1, 1).expand(B, nh, S, 1).float()
    combined = torch.cat([aw, sinks], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = torch.softmax(combined, dim=-1)
    scores = probs[..., :-1]  # drop sink column
    attn_out = torch.matmul(scores, v.float()).to(hidden_states.dtype)  # [B,nh,S,hd]

    # conjugate −sin un-rotation of the output rope slice (K==V); attn_out is [B,nh,S,hd]
    # so apply_rope (unsqueeze_dim=1) broadcasts cos/sin over the head axis, matching HF.
    attn_out = apply_rope(attn_out, cos, -sin)  # [B,nh,S,hd]
    attn_out = attn_out.transpose(1, 2).contiguous()  # [B,S,nh,hd]
    grouped = attn_out.reshape(B, S, cfg.o_groups, -1)  # [B,S,g,nh*hd/g]
    grouped = M.grouped_linear(grouped.reshape(B, S, -1), attn.o_a_proj.weight.data, cfg.o_groups, device)
    grouped = grouped.reshape(B, S, -1)  # flatten(2) -> [B,S,g*rank]
    output = M.linear(grouped, attn.o_b_proj.weight.data, device, weight_dtype=wdt)  # [B,S,H]
    return output

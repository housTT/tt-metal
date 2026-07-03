# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end DeepSeek-V4 forward (Path B), mirroring HF DeepseekV4ForCausalLM.

Assembles the validated TT-NN modules through the mHC 4-stream residual:
  embed -> [ attn_hc -> MLA attention -> mix ; ffn_hc -> SparseMoE -> mix ] × L
        -> hc_head -> norm -> lm_head.

All learned matmuls/projections/norms/experts/o_lora/lm_head run on device (ttnn).
Documented host fallbacks (GOAL anti-BS rule): interleaved RoPE, sink softmax, the tiny
mHC Sinkhorn/mixing, hash-table lookup, top-k selection, causal mask, embedding gather.

Weights are read directly from an in-memory HF DeepseekV4 module (bring-up convenience;
the reference and the TT path therefore share identical weights, which is exactly what the
per-op/e2e PCC comparison requires). This is the prefill / single-forward path used for
the reduced-config correctness demo (seq_len ≤ sliding_window, sliding_attention layers).
"""
from __future__ import annotations

import torch

from models.demos.deepseek_v4.tt import attention as A
from models.demos.deepseek_v4.tt import modules as M


def _causal_mask(S, dtype=torch.float32):
    m = torch.full((S, S), float("-inf"), dtype=dtype)
    m = torch.triu(m, diagonal=1)
    return m.view(1, 1, S, S)


def sparse_moe(collapsed_ln, mlp, input_ids, cfg, device, weight_dtype=None):
    """DeepseekV4SparseMoeBlock: hash/topk router (scores on device) + routed experts
    (device matmuls) + shared expert (device), combined. Returns [B,S,H]."""
    B, S, H = collapsed_ln.shape
    flat = collapsed_ln.reshape(-1, H)
    gate = mlp.gate
    scores = M.sqrtsoftplus_router_scores(flat, gate.weight.data, device)  # [N, n_experts]
    if getattr(mlp, "is_hash", False):
        indices = gate.tid2eid[input_ids.reshape(-1)].long()  # frozen token->expert table
    else:
        indices = torch.topk(
            scores + gate.e_score_correction_bias, cfg.num_experts_per_tok, dim=-1, sorted=False
        ).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * cfg.routed_scaling_factor
    routed = M.moe_routed_experts(
        flat,
        mlp.experts.gate_up_proj.data,
        mlp.experts.down_proj.data,
        indices,
        weights,
        device,
        limit=cfg.swiglu_limit,
    ).reshape(B, S, H)
    se = mlp.shared_experts
    shared = M.clamped_swiglu_mlp(
        collapsed_ln,
        se.gate_proj.weight.data,
        se.up_proj.weight.data,
        se.down_proj.weight.data,
        device,
        limit=cfg.swiglu_limit,
        weight_dtype=weight_dtype or __import__("ttnn").bfloat16,
    )
    return routed + shared


def tt_forward(hf_model, input_ids, device, weight_dtype=None):
    """Full reduced-config forward. Returns logits [B,S,vocab]."""
    top = hf_model.model
    cfg = hf_model.config
    B, S = input_ids.shape
    dtype = torch.float32

    # embedding (gather, no FLOPs -> host) then expand to hc_mult streams
    embeds = top.embed_tokens(input_ids).to(dtype)  # [B,S,H]
    streams = embeds.unsqueeze(2).expand(B, S, cfg.hc_mult, embeds.shape[-1]).contiguous()

    position_ids = torch.arange(S).unsqueeze(0)
    rope = {
        "main": top.rotary_emb(embeds, position_ids=position_ids, layer_type="main"),
        "compress": top.rotary_emb(embeds, position_ids=position_ids, layer_type="compress"),
    }
    causal = _causal_mask(S, dtype)

    for layer in top.layers:
        # attention sublayer through attn_hc; sliding layers use "main" rope, CSA/HCA use "compress"
        cos, sin = rope[layer.self_attn.rope_layer_type]
        post, comb, collapsed = M.hyperconnection(streams, layer.attn_hc, device)
        collapsed_ln = M.rms_norm(collapsed, layer.input_layernorm.weight.data, device, eps=cfg.rms_norm_eps)
        attn_out = A.mla_attention(
            collapsed_ln, layer.self_attn, cos, sin, causal, cfg, device, weight_dtype, position_ids=position_ids
        )
        streams = post.to(dtype).unsqueeze(-1) * attn_out.to(dtype).unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), streams
        )
        # MLP sublayer through ffn_hc
        post, comb, collapsed = M.hyperconnection(streams, layer.ffn_hc, device)
        collapsed_ln = M.rms_norm(collapsed, layer.post_attention_layernorm.weight.data, device, eps=cfg.rms_norm_eps)
        mlp_out = sparse_moe(collapsed_ln, layer.mlp, input_ids, cfg, device, weight_dtype)
        streams = post.to(dtype).unsqueeze(-1) * mlp_out.to(dtype).unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), streams
        )

    hidden = M.hyper_head(streams, top.hc_head, device)  # [B,S,H]
    hidden = M.rms_norm(hidden, top.norm.weight.data, device, eps=cfg.rms_norm_eps)
    logits = M.linear(hidden, hf_model.lm_head.weight.data, device)  # [B,S,vocab]
    return logits

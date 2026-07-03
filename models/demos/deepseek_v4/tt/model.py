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


def sparse_moe_streaming(collapsed_ln, layer_idx, layer, store, cfg, device, input_ids):
    """MoE for the streaming real-weight path: route, then dequantize + run ONLY the routed
    experts (union across tokens) from `store` — avoids materializing all 256 experts. Batches
    all tokens for a given expert into one device matmul. Shared expert + gate are already
    loaded in `layer`."""

    from models.demos.deepseek_v4.reference import real_weights as RW

    B, S, H = collapsed_ln.shape
    flat = collapsed_ln.reshape(-1, H)
    N = flat.shape[0]
    gate = layer.mlp.gate
    scores = M.sqrtsoftplus_router_scores(flat, gate.weight.data, device)  # [N, 256]
    if getattr(layer.mlp, "is_hash", False):
        indices = gate.tid2eid[input_ids.reshape(-1)].long()  # [N, top_k]
    else:
        indices = torch.topk(
            scores + gate.e_score_correction_bias, cfg.num_experts_per_tok, dim=-1, sorted=False
        ).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor

    # Run only the routed experts (union across tokens). Expert weights are kept RESIDENT on
    # device in bfloat4_b (native fp4 precision), loaded on first touch and reused every token —
    # so warm tokens do NO per-token weight transfer/dequant (only the tiny activation crosses).
    interm = cfg.moe_intermediate_size
    uniq = torch.unique(indices).tolist()
    out = torch.zeros(N, H, dtype=torch.float32)
    for e in uniq:
        sel = indices == e
        rows = sel.any(dim=-1).nonzero().flatten()
        if rows.numel() == 0:
            continue
        tgu, tdn = RW.expert_fused_dev(store, device, layer_idx, e)  # resident on device (bf4)
        y = M.fused_experts_dev(flat[rows], tgu, tdn, interm, device, limit=cfg.swiglu_limit)
        w_e = (weights * sel).sum(dim=-1)[rows].float()
        out[rows] += w_e.unsqueeze(-1) * y.float()

    se = layer.mlp.shared_experts
    shared = M.clamped_swiglu_mlp(
        collapsed_ln,
        se.gate_proj.weight.data,
        se.up_proj.weight.data,
        se.down_proj.weight.data,
        device,
        limit=cfg.swiglu_limit,
    )
    return out.reshape(B, S, H) + shared


def run_decoder_layer(
    streams, layer, rope, causal, cfg, device, input_ids, position_ids, weight_dtype=None, moe_fn=None
):
    """One DeepSeek-V4 decoder layer through the mHC 4-stream residual. `layer` is an HF
    DecoderLayer module (weights read from it). Returns updated streams [B,S,hc,H]."""
    dtype = torch.float32
    cos, sin = rope[layer.self_attn.rope_layer_type]  # sliding->main, CSA/HCA->compress
    post, comb, collapsed = M.hyperconnection(streams, layer.attn_hc, device)
    collapsed_ln = M.rms_norm(collapsed, layer.input_layernorm.weight.data, device, eps=cfg.rms_norm_eps)
    attn_out = A.mla_attention(
        collapsed_ln, layer.self_attn, cos, sin, causal, cfg, device, weight_dtype, position_ids=position_ids
    )
    streams = post.to(dtype).unsqueeze(-1) * attn_out.to(dtype).unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), streams
    )
    post, comb, collapsed = M.hyperconnection(streams, layer.ffn_hc, device)
    collapsed_ln = M.rms_norm(collapsed, layer.post_attention_layernorm.weight.data, device, eps=cfg.rms_norm_eps)
    mlp_out = (
        moe_fn(collapsed_ln) if moe_fn else sparse_moe(collapsed_ln, layer.mlp, input_ids, cfg, device, weight_dtype)
    )
    streams = post.to(dtype).unsqueeze(-1) * mlp_out.to(dtype).unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), streams
    )
    return streams


def tt_forward_streaming(scratch, store, layer_types, mlp_types, input_ids, device, num_layers, log=None):
    """Full-model forward with the ACTUAL pretrained weights, streamed one layer at a time.

    `scratch` is a small HF model whose layers cover each (attn_type, mlp_type) combo; `store`
    is a reference.real_weights.RealWeightStore over the fp8/fp4 checkpoint. Globals (embed,
    norm, lm_head, hc_head) are loaded once; each real layer i is dequantized into the matching
    scratch layer just before it runs, so peak memory is ~one layer of bf16 weights. Returns
    logits [B,S,vocab]. This is the prefill / single-forward path (no KV cache)."""
    from models.demos.deepseek_v4.reference import real_weights as RW

    top = scratch.model
    cfg = scratch.config
    B, S = input_ids.shape
    dtype = torch.float32
    RW.load_globals(scratch, store)

    embeds = top.embed_tokens(input_ids).to(dtype)
    streams = embeds.unsqueeze(2).expand(B, S, cfg.hc_mult, embeds.shape[-1]).contiguous()
    position_ids = torch.arange(S).unsqueeze(0)
    rope = {
        "main": top.rotary_emb(embeds, position_ids=position_ids, layer_type="main"),
        "compress": top.rotary_emb(embeds, position_ids=position_ids, layer_type="compress"),
    }
    causal = _causal_mask(S, dtype)

    # map (attn_kind, mlp_kind) -> a scratch layer of that exact structure.
    # NB: CSA and HCA both use "compress" rope but have different compressor modules, so key
    # on the compressor class, not the rope type.
    def attn_kind(mod):
        c = getattr(mod, "compressor", None)
        return "sliding" if c is None else ("CSA" if "CSA" in type(c).__name__ else "HCA")

    type_of_layer = {
        "sliding_attention": "sliding",
        "compressed_sparse_attention": "CSA",
        "heavily_compressed_attention": "HCA",
    }
    scratch_by_type = {}
    for si, sl in enumerate(top.layers):
        key = (attn_kind(sl.self_attn), "hash" if getattr(sl.mlp, "is_hash", False) else "moe")
        scratch_by_type.setdefault(key, si)

    for i in range(num_layers):
        key = (type_of_layer[layer_types[i]], "hash" if mlp_types[i] == "hash_moe" else "moe")
        if key not in scratch_by_type:
            raise RuntimeError(f"scratch model lacks a layer of type {key}; increase its num_hidden_layers")
        sl = top.layers[scratch_by_type[key]]
        RW.load_layer(sl, i, store, skip_experts=True)  # everything but the 256 routed experts
        moe_fn = lambda cln, _i=i, _sl=sl: sparse_moe_streaming(cln, _i, _sl, store, cfg, device, input_ids)
        streams = run_decoder_layer(streams, sl, rope, causal, cfg, device, input_ids, position_ids, moe_fn=moe_fn)
        if log:
            log(f"  layer {i + 1}/{num_layers} ({layer_types[i]}, {mlp_types[i]})")

    hidden = M.hyper_head(streams, top.hc_head, device)
    hidden = M.rms_norm(hidden, top.norm.weight.data, device, eps=cfg.rms_norm_eps)
    logits = M.linear(hidden, scratch.lm_head.weight.data, device)
    return logits


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
        streams = run_decoder_layer(streams, layer, rope, causal, cfg, device, input_ids, position_ids, weight_dtype)

    hidden = M.hyper_head(streams, top.hc_head, device)  # [B,S,H]
    hidden = M.rms_norm(hidden, top.norm.weight.data, device, eps=cfg.rms_norm_eps)
    logits = M.linear(hidden, hf_model.lm_head.weight.data, device)  # [B,S,vocab]
    return logits

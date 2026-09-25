# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Incremental (KV-cached) MLA-v4 decode on the real weights.

The correctness path (tt/model.py::tt_forward_streaming) recomputes the whole sequence every
token, so each decode step touches the *union* of routed experts over all positions (up to all
256/layer -> streams ~the entire 149 GB MoE per token). This module keeps a per-layer KV cache
so a decode step processes only the ONE new token: it attends its query over the cached K==V
(+ compressor entries) and runs MoE on just that token (top-6 experts/layer). That collapses
per-token expert streaming ~40x, which is the dominant decode cost.

State per layer:
  * main_kv : [B,1,S,hd]  post-RoPE K==V (shared MQA head), appended one row per token
  * hbuf    : [B,S,H]     attention-input (input_layernorm'd collapsed stream), needed because
              the CSA/HCA compressors + lightning indexer pool over the whole sequence; the
              compressed entries are rebuilt from hbuf each step (cheap host math + small
              projections) — resident/cached compressors are a later optimization.

Math mirrors tt/attention.py + tt/compressors.py exactly (validated by test_kv_cache_decode.py
against tt_forward_streaming), so tokens are identical to the reference. Weights are still
streamed per layer (resident sharding is the next optimization); this step isolates the KV-cache
win. mHC mixing is per-position, so a new token's stream evolves independently of past streams.
"""
from __future__ import annotations

import torch

from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import attention as A
from models.demos.deepseek_v4.tt import compressors as C
from models.demos.deepseek_v4.tt import model as MODEL
from models.demos.deepseek_v4.tt import modules as M


def _is_hca(comp):
    return comp.__class__.__name__.startswith("DeepseekV4HCA")


def _csa_indexer_decode(hbuf, new_hidden, new_q_res, new_pos, indexer, cfg, device):
    """Lightning indexer for a single new query. Compressed cache built from the full buffer
    `hbuf`; query (q_b + weights_proj) from the new token only. Returns top_k indices [B,1,k]
    with -1 sentinels — mirrors compressors.csa_indexer with S=1 query."""
    B = hbuf.shape[0]
    hd, m = indexer.head_dim, indexer.compress_rate
    compressed, _ = C._ca_cb_pool(hbuf, indexer, cfg, device, hd, m)  # [B,T,hd]
    T = compressed.shape[1]
    q = M.linear(new_q_res, indexer.q_b_proj.weight.data, device)  # [B,1,H_idx*hd]
    q = q.view(B, 1, -1, hd).transpose(1, 2)  # [B,H,1,hd]
    pos = torch.tensor([[new_pos]], dtype=torch.long)
    cos_q, sin_q = indexer.rotary_emb(new_hidden, position_ids=pos, layer_type="compress")
    q = A.apply_rope(q, cos_q, sin_q).transpose(1, 2)  # [B,1,H,hd]
    scorer = indexer.scorer
    scores = torch.matmul(q.float(), compressed.transpose(-1, -2).float().unsqueeze(1))  # [B,1,H,T]
    scores = torch.relu(scores) * scorer.softmax_scale
    weights = M.linear(new_hidden, scorer.weights_proj.weight.data, device).float() * scorer.weights_scaling
    index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B,1,T]
    top_k = min(indexer.index_topk, T)
    if T == 0:
        return index_scores.topk(top_k, dim=-1).indices
    causal_threshold = (pos + 1) // m  # [B,1]
    entry_idx = torch.arange(T)
    future = entry_idx.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
    index_scores = index_scores.masked_fill(future, float("-inf"))
    tki = index_scores.topk(top_k, dim=-1).indices
    invalid = tki >= causal_threshold.unsqueeze(-1)
    return torch.where(invalid, torch.full_like(tki, -1), tki)


def _compressor_decode(hbuf, new_hidden, new_q_res, new_pos, comp, cfg, device):
    """Compressed long-range KV for the new query. Returns (ckv [B,1,T,hd], block_bias [1,1,1,T]|None)."""
    if _is_hca(comp):
        return C.hca_compressor(hbuf, comp, cfg, device), None  # [B,1,T,hd]; HCA has no indexer bias
    hd, m = comp.head_dim, comp.compress_rate
    compressed, _ = C._ca_cb_pool(hbuf, comp, cfg, device, hd, m)  # [B,T,hd]
    ckv = compressed.unsqueeze(1)  # [B,1,T,hd]
    T = ckv.shape[2]
    tki = _csa_indexer_decode(hbuf, new_hidden, new_q_res, new_pos, comp.indexer, cfg, device)  # [B,1,k]
    B = hbuf.shape[0]
    valid = tki >= 0
    safe = torch.where(valid, tki, torch.full_like(tki, T))
    block_bias = ckv.new_full((B, 1, 1, T + 1), float("-inf"))
    block_bias.scatter_(-1, safe.unsqueeze(1), 0.0)
    return ckv, block_bias[..., :T]


def mla_attention_decode(new_ln, hbuf, kv_cache, attn, cfg, device, cos_new, sin_new, new_pos, resident):
    """Incremental MLA-v4 attention for one new token.

    new_ln   : [B,1,H]  attention input (input_layernorm'd) for the new token
    hbuf     : [B,S,H]  full attention-input buffer INCLUDING the new token (compressor pooling)
    kv_cache : [B,1,S-1,hd] cached post-RoPE main K==V (past), or None on the first token
    cos_new/sin_new : main-RoPE cos/sin at the new position, shape [B,1,rope_dim]
    Returns (output [B,1,H], new_kv [B,1,1,hd]).  Mirrors tt/attention.py::mla_attention (S=1 query).
    """
    B, _, H = new_ln.shape
    nh, hd = cfg.num_attention_heads, cfg.head_dim
    scaling = attn.scaling
    store, li = resident

    def lin(x, w, tag):
        return M.linear_dev(x, RW.dev_linear(store, device, (li, tag), w), device)

    # Q (new token)
    q_res = lin(new_ln, attn.q_a_proj.weight.data, "q_a")  # [B,1,q_lora]
    q_res = M.rms_norm(q_res, attn.q_a_norm.weight.data, device, eps=cfg.rms_norm_eps)
    q = lin(q_res, attn.q_b_proj.weight.data, "q_b").view(B, 1, nh, hd).transpose(1, 2)  # [B,nh,1,hd]
    q = M.rms_norm(q, None, device, eps=cfg.rms_norm_eps)
    q = A.apply_rope(q, cos_new, sin_new)

    # KV (new token) -> append to cache
    kv_new = lin(new_ln, attn.kv_proj.weight.data, "kv")  # [B,1,hd]
    kv_new = M.rms_norm(kv_new, attn.kv_norm.weight.data, device, eps=cfg.rms_norm_eps)
    kv_new = kv_new.view(B, 1, 1, hd).transpose(1, 2)  # [B,1,1,hd]
    kv_new = A.apply_rope(kv_new, cos_new, sin_new)
    main_kv = kv_new if kv_cache is None else torch.cat([kv_cache, kv_new], dim=2)  # [B,1,S,hd]

    # new query sees all past main KV (causal; new token is last). matches the reference (no
    # sliding-window eviction — reference assumes S <= window; kept identical for correctness).
    kv = main_kv
    mask = torch.zeros(1, 1, 1, main_kv.shape[2])
    if getattr(attn, "compressor", None) is not None:
        ckv, block_bias = _compressor_decode(hbuf, new_ln, q_res, new_pos, attn.compressor, cfg, device)
        T = ckv.shape[2]
        if T > 0:
            kv = torch.cat([main_kv, ckv.to(main_kv.dtype)], dim=2)  # [B,1,S+T,hd]
            ext = block_bias.float() if block_bias is not None else torch.zeros(1, 1, 1, T)
            mask = torch.cat([mask, ext], dim=-1)

    # attention core with sinks (host)
    Lkv = kv.shape[2]
    k = kv.expand(B, nh, Lkv, hd)
    v = k
    aw = torch.matmul(q.float(), k.float().transpose(2, 3)) * scaling  # [B,nh,1,Lkv]
    aw = aw + mask.float()
    sinks = attn.sinks.reshape(1, -1, 1, 1).expand(B, nh, 1, 1).float()
    combined = torch.cat([aw, sinks], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = torch.softmax(combined, dim=-1)
    scores = probs[..., :-1]
    attn_out = torch.matmul(scores, v.float()).to(new_ln.dtype)  # [B,nh,1,hd]
    attn_out = A.apply_rope(attn_out, cos_new, -sin_new)
    attn_out = attn_out.transpose(1, 2).contiguous()  # [B,1,nh,hd]
    gflat = attn_out.reshape(B, 1, cfg.o_groups, -1).reshape(B, 1, -1)
    wd = RW.dev_grouped(store, device, (li, "o_a"), attn.o_a_proj.weight.data, cfg.o_groups)
    grouped = M.grouped_linear_dev(gflat, wd, cfg.o_groups, device).reshape(B, 1, -1)
    output = lin(grouped, attn.o_b_proj.weight.data, "o_b")  # [B,1,H]
    return output, kv_new


class KVDecoder:
    """Prefill then incremental single-token decode with a per-layer KV cache, real weights."""

    def __init__(self, scratch, store, layer_types, mlp_types, device, num_layers=43):
        self.scratch = scratch
        self.store = store
        self.device = device
        self.num_layers = num_layers
        self.layer_types = layer_types
        self.mlp_types = mlp_types
        self.cfg = scratch.config
        RW.load_globals(scratch, store)
        top = scratch.model
        self.type_of_layer = {
            "sliding_attention": "sliding",
            "compressed_sparse_attention": "CSA",
            "heavily_compressed_attention": "HCA",
        }

        def attn_kind(mod):
            c = getattr(mod, "compressor", None)
            return "sliding" if c is None else ("CSA" if "CSA" in type(c).__name__ else "HCA")

        self.scratch_by_type = {}
        for si, sl in enumerate(top.layers):
            key = (attn_kind(sl.self_attn), "hash" if getattr(sl.mlp, "is_hash", False) else "moe")
            self.scratch_by_type.setdefault(key, si)
        # caches (populated by prefill)
        self.kv = [None] * num_layers  # per-layer main KV [B,1,S,hd]
        self.hbuf = [None] * num_layers  # per-layer attention-input buffer [B,S,H]
        self.pos = 0  # next position index
        self.token_ids = None

    def _scratch_layer(self, i):
        key = (self.type_of_layer[self.layer_types[i]], "hash" if self.mlp_types[i] == "hash_moe" else "moe")
        sl = self.scratch.model.layers[self.scratch_by_type[key]]
        RW.load_layer(sl, i, self.store, skip_experts=True)
        return sl

    def _rope(self, positions):
        top = self.scratch.model
        dummy = torch.zeros(1, positions.shape[1], self.cfg.hidden_size)
        return {
            "main": top.rotary_emb(dummy, position_ids=positions, layer_type="main"),
            "compress": top.rotary_emb(dummy, position_ids=positions, layer_type="compress"),
        }

    def _logits_from_streams(self, streams):
        top = self.scratch.model
        hidden = M.hyper_head(streams, top.hc_head, self.device)
        hidden = M.rms_norm(hidden, top.norm.weight.data, self.device, eps=self.cfg.rms_norm_eps)
        return M.linear_dev(
            hidden, RW.dev_linear(self.store, self.device, ("lm_head",), self.scratch.lm_head.weight.data), self.device
        )

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the full prompt, seeding per-layer KV + hbuf caches. Returns next-token logits [1, vocab]."""
        cfg = self.cfg
        top = self.scratch.model
        B, S = input_ids.shape
        self.token_ids = input_ids.clone()
        embeds = top.embed_tokens(input_ids).to(torch.float32)
        streams = embeds.unsqueeze(2).expand(B, S, cfg.hc_mult, embeds.shape[-1]).contiguous()
        positions = torch.arange(S).unsqueeze(0)
        rope = self._rope(positions)
        causal = MODEL._causal_mask(S, torch.float32)
        for i in range(self.num_layers):
            sl = self._scratch_layer(i)
            cos, sin = rope[sl.self_attn.rope_layer_type]
            post, comb, collapsed = M.hyperconnection(streams, sl.attn_hc, self.device)
            collapsed_ln = M.rms_norm(collapsed, sl.input_layernorm.weight.data, self.device, eps=cfg.rms_norm_eps)
            attn_out, main_kv = A.mla_attention(
                collapsed_ln, sl.self_attn, cos, sin, causal, cfg, self.device,
                position_ids=positions, resident=(self.store, i), return_kv=True,
            )
            self.kv[i] = main_kv.detach().clone()  # [B,1,S,hd]
            self.hbuf[i] = collapsed_ln.detach().clone()  # [B,S,H]
            streams = post.unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), streams)
            post, comb, collapsed = M.hyperconnection(streams, sl.ffn_hc, self.device)
            collapsed_ln = M.rms_norm(collapsed, sl.post_attention_layernorm.weight.data, self.device, eps=cfg.rms_norm_eps)
            mlp_out = MODEL.sparse_moe_streaming(collapsed_ln, i, sl, self.store, cfg, self.device, input_ids)
            streams = post.unsqueeze(-1) * mlp_out.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), streams)
        self.pos = S
        logits = self._logits_from_streams(streams)  # [B,S,vocab]
        return logits[:, -1, :]

    def decode_step(self, token_id: int, profile: bool = False) -> torch.Tensor:
        """Advance one token using the KV cache. Returns next-token logits [1, vocab].
        If `profile`, accumulate a host-side time breakdown into self.prof."""
        import time as _t

        cfg = self.cfg
        top = self.scratch.model
        new_pos = self.pos
        prof = {"load": 0.0, "attn": 0.0, "moe": 0.0, "mhc": 0.0}
        tok = torch.tensor([[token_id]], dtype=torch.long)
        self.token_ids = torch.cat([self.token_ids, tok], dim=1)
        embed = top.embed_tokens(tok).to(torch.float32)  # [1,1,H]
        B = 1
        streams = embed.unsqueeze(2).expand(B, 1, cfg.hc_mult, embed.shape[-1]).contiguous()  # [1,1,hc,H]
        positions = torch.arange(new_pos + 1).unsqueeze(0)  # 0..new_pos (for compress-rope over buffer)
        full_rope = self._rope(positions)
        for i in range(self.num_layers):
            t0 = _t.perf_counter()
            sl = self._scratch_layer(i)
            prof["load"] += _t.perf_counter() - t0
            cos_all, sin_all = full_rope[sl.self_attn.rope_layer_type]
            cos_new = cos_all[:, new_pos : new_pos + 1, :]
            sin_new = sin_all[:, new_pos : new_pos + 1, :]
            t0 = _t.perf_counter()
            post, comb, collapsed = M.hyperconnection(streams, sl.attn_hc, self.device)
            collapsed_ln = M.rms_norm(collapsed, sl.input_layernorm.weight.data, self.device, eps=cfg.rms_norm_eps)
            self.hbuf[i] = torch.cat([self.hbuf[i], collapsed_ln.detach()], dim=1)  # [B,S+1,H]
            prof["mhc"] += _t.perf_counter() - t0
            t0 = _t.perf_counter()
            attn_out, kv_new = mla_attention_decode(
                collapsed_ln, self.hbuf[i], self.kv[i], sl.self_attn, cfg, self.device,
                cos_new, sin_new, new_pos, resident=(self.store, i),
            )
            self.kv[i] = torch.cat([self.kv[i], kv_new.detach()], dim=2)  # [B,1,S+1,hd]
            prof["attn"] += _t.perf_counter() - t0
            t0 = _t.perf_counter()
            streams = post.unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), streams)
            post, comb, collapsed = M.hyperconnection(streams, sl.ffn_hc, self.device)
            collapsed_ln = M.rms_norm(collapsed, sl.post_attention_layernorm.weight.data, self.device, eps=cfg.rms_norm_eps)
            prof["mhc"] += _t.perf_counter() - t0
            t0 = _t.perf_counter()
            mlp_out = MODEL.sparse_moe_streaming(collapsed_ln, i, sl, self.store, cfg, self.device, tok)
            prof["moe"] += _t.perf_counter() - t0
            streams = post.unsqueeze(-1) * mlp_out.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), streams)
        self.pos = new_pos + 1
        logits = self._logits_from_streams(streams)  # [B,1,vocab]
        if profile:
            self.prof = prof
            print(f"  [decode profile] load={prof['load']:.2f}s attn={prof['attn']:.2f}s "
                  f"moe={prof['moe']:.2f}s mhc={prof['mhc']:.2f}s", flush=True)
        return logits[:, -1, :]

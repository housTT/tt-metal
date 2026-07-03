# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CSA / HCA KV compressors + lightning indexer (HF DeepseekV4 §2.3.2 / §2.3.1),
ported for the STATELESS single-shot path (past_key_values=None) used in a prefill
forward. See docs/HF_REFERENCE_SPEC.md §3-4.

Learned projections (kv_proj, gate_proj, q_b_proj, weights_proj) and the indexer score
matmul run on device (ttnn). The windowed gated-softmax pooling, RoPE, top-k selection
and block-bias scatter run on host (documented torch-fallback — intricate control flow on
tiny non-tile-friendly window axes; negligible FLOPs).
"""
from __future__ import annotations

import torch

from models.demos.deepseek_v4.tt import attention as A
from models.demos.deepseek_v4.tt import modules as M


def _rope_compress(x_bthd, comp_module, positions):
    """Apply the compress-rope to a [B,T,hd] tensor (as a single head)."""
    cos, sin = comp_module.rotary_emb(x_bthd, position_ids=positions, layer_type="compress")
    return A.apply_rope(x_bthd.unsqueeze(1), cos, sin).squeeze(1)


def hca_compressor(hidden_states, hca, cfg, device):
    """Heavily-Compressed-Attention compressor, stateless. Returns compressed_kv [B,1,T,hd]."""
    B, S, _ = hidden_states.shape
    m = hca.compress_rate
    hd = hca.head_dim
    kv = M.linear(hidden_states, hca.kv_proj.weight.data, device)  # [B,S,hd]  (device)
    gate = M.linear(hidden_states, hca.gate_proj.weight.data, device)  # [B,S,hd]  (device)
    usable = (S // m) * m
    if usable == 0:
        return torch.zeros(B, 1, 0, hd)
    nw = usable // m
    ck = kv[:, :usable].view(B, nw, m, hd)
    cg = gate[:, :usable].view(B, nw, m, hd) + hca.position_bias.data
    pooled = (ck * cg.softmax(dim=2, dtype=torch.float32).to(ck.dtype)).sum(dim=2)  # [B,nw,hd]
    compressed = M.rms_norm(pooled, hca.kv_norm.weight.data, device, eps=cfg.rms_norm_eps)
    positions = (torch.arange(nw) * m).unsqueeze(0).expand(B, -1)
    compressed = _rope_compress(compressed, hca, positions)
    return compressed.unsqueeze(1)  # [B,1,nw,hd]


def _ca_cb_pool(hidden_states, comp, cfg, device, hd, m):
    """Shared CSA/indexer Ca/Cb overlapping-window gated pool (stateless). Returns compressed [B,T,hd]."""
    B, S, _ = hidden_states.shape
    kv = M.linear(hidden_states, comp.kv_proj.weight.data, device)  # [B,S,2*hd]
    gate = M.linear(hidden_states, comp.gate_proj.weight.data, device)  # [B,S,2*hd]
    usable = (S // m) * m
    if usable == 0:
        return torch.zeros(B, 0, hd), 0
    nw = usable // m
    ck = kv[:, :usable].view(B, nw, m, -1)
    cg = gate[:, :usable].view(B, nw, m, -1) + comp.position_bias.data
    new_kv = ck.new_zeros((B, nw, 2 * m, hd))
    new_gate = cg.new_full((B, nw, 2 * m, hd), float("-inf"))
    new_kv[:, :, m:] = ck[..., hd:]
    new_gate[:, :, m:] = cg[..., hd:]
    if nw > 1:
        new_kv[:, 1:, :m] = ck[:, :-1, :, :hd]
        new_gate[:, 1:, :m] = cg[:, :-1, :, :hd]
    pooled = (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
    compressed = M.rms_norm(pooled, comp.kv_norm.weight.data, device, eps=cfg.rms_norm_eps)
    positions = (torch.arange(nw) * m).unsqueeze(0).expand(B, -1)
    compressed = _rope_compress(compressed, comp, positions)
    return compressed, nw


def csa_indexer(hidden_states, q_residual, position_ids, indexer, cfg, device):
    """Lightning indexer, stateless. Returns top_k_indices [B,S,k] with -1 sentinels."""
    B, S, _ = hidden_states.shape
    hd = indexer.head_dim
    m = indexer.compress_rate
    compressed, nw = _ca_cb_pool(hidden_states, indexer, cfg, device, hd, m)  # [B,T,hd]
    T = compressed.shape[1]
    # queries: q_b_proj on device, RoPE on host
    q = M.linear(q_residual, indexer.q_b_proj.weight.data, device)  # [B,S,H_idx*hd]
    q = q.view(B, S, -1, hd).transpose(1, 2)  # [B,H,S,hd]
    cos_q, sin_q = indexer.rotary_emb(hidden_states, position_ids=position_ids, layer_type="compress")
    q = A.apply_rope(q, cos_q, sin_q).transpose(1, 2)  # [B,S,H,hd]
    # scorer: learned projections were on device; the small 4-D score matmul + relu/scale on host
    scorer = indexer.scorer
    scores = torch.matmul(q.float(), compressed.transpose(-1, -2).float().unsqueeze(1))  # [B,S,H,T]
    scores = torch.relu(scores) * scorer.softmax_scale
    weights = M.linear(hidden_states, scorer.weights_proj.weight.data, device).float() * scorer.weights_scaling
    index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B,S,T]
    top_k = min(indexer.index_topk, T)
    if T == 0:
        return index_scores.topk(top_k, dim=-1).indices
    causal_threshold = (position_ids + 1) // m
    entry_idx = torch.arange(T)
    future = entry_idx.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
    index_scores = index_scores.masked_fill(future, float("-inf"))
    tki = index_scores.topk(top_k, dim=-1).indices
    invalid = tki >= causal_threshold.unsqueeze(-1)
    return torch.where(invalid, torch.full_like(tki, -1), tki)


def csa_compressor(hidden_states, q_residual, position_ids, csa, cfg, device):
    """Compressed-Sparse-Attention compressor, stateless. Returns (compressed_kv [B,1,T,hd], block_bias)."""
    B, S, _ = hidden_states.shape
    hd = csa.head_dim
    m = csa.compress_rate
    compressed, nw = _ca_cb_pool(hidden_states, csa, cfg, device, hd, m)  # [B,T,hd]
    compressed_kv = compressed.unsqueeze(1)  # [B,1,T,hd]
    T = compressed_kv.shape[2]
    tki = csa_indexer(hidden_states, q_residual, position_ids, csa.indexer, cfg, device)  # [B,S,k]
    valid = tki >= 0
    safe = torch.where(valid, tki, torch.full_like(tki, T))
    block_bias = compressed_kv.new_full((B, 1, S, T + 1), float("-inf"))
    block_bias.scatter_(-1, safe.unsqueeze(1), 0.0)
    return compressed_kv, block_bias[..., :T]

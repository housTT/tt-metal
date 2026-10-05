# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

EPS = 1e-5


def layer_norm(x, w, b):
    return F.layer_norm(x, (x.shape[-1],), w, b, EPS)


def attention_block(p, i, x, key_pad, n_heads=16, activation="relu"):
    """One nn.TransformerEncoderLayer(norm_first=True) in explicit math; key_pad (B,S) True marks padding."""
    g = lambda k: p[f"layers.{i}.{k}"]
    B, S, D = x.shape
    hd = D // n_heads
    h = layer_norm(x, g("norm1.weight"), g("norm1.bias"))
    qkv = h @ g("self_attn.in_proj_weight").T + g("self_attn.in_proj_bias")
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, S, n_heads, hd).transpose(1, 2)
    k = k.view(B, S, n_heads, hd).transpose(1, 2)
    v = v.view(B, S, n_heads, hd).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) * (hd**-0.5)
    scores = scores.masked_fill(key_pad[:, None, None, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    ctx = (attn @ v).transpose(1, 2).reshape(B, S, D)
    x = x + ctx @ g("self_attn.out_proj.weight").T + g("self_attn.out_proj.bias")
    h2 = layer_norm(x, g("norm2.weight"), g("norm2.bias"))
    ff = h2 @ g("linear1.weight").T + g("linear1.bias")
    ff = F.relu(ff) if activation == "relu" else F.gelu(ff)
    ff = ff @ g("linear2.weight").T + g("linear2.bias")
    return x + ff


def head_layers(parts, h, attention_mask, activation="relu"):
    key_pad = attention_mask == 0
    for i in range(2):
        h = attention_block(parts["head"], i, h, key_pad, activation=activation)
    return h


def scorer(parts, h):
    sc = parts["scorer"]
    x = layer_norm(h, sc["0.weight"], sc["0.bias"])
    x = F.gelu(x @ sc["1.weight"].T + sc["1.bias"])
    return (x @ sc["3.weight"].T + sc["3.bias"]).squeeze(-1)


@torch.no_grad()
def decision_head(parts, encoder_out, attention_mask, qtype):
    """encoder_out (B,S,D) fp32 -> (logits_all (B,S), cls (B,D), h (B,S,D)) with the explicit-math head."""
    h = encoder_out + parts["type_emb"][qtype][:, None, :]
    h = head_layers(parts, h, attention_mask)
    return scorer(parts, h), h[:, 0], h


def gather_markers(logits_all, marker_pos, marker_mask):
    g = torch.gather(logits_all, 1, marker_pos.clamp(min=0))
    return g.masked_fill(~marker_mask, -1e4)

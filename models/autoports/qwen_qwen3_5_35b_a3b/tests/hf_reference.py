# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer-only HF references, including selected-output checks at native context.

The long reference retains every KV/recurrent contribution at real target shapes.
It avoids an O(context²) host attention matrix and MoE work on unobserved outputs.
TT still executes the complete layer for every input token.
"""

import itertools

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import apply_rotary_pos_emb, eager_attention_forward


@torch.no_grad()
def prefill_reference(layer, cfg, rope, x, layer_idx, *, sparse=False):
    batch, length, _ = x.shape
    cache = DynamicCache(config=cfg)
    if not sparse:
        pos = torch.arange(length).expand(batch, -1)
        mask = torch.full((length, length), float("-inf"), dtype=x.dtype).triu(1)[None, None]
        y = layer(
            x,
            position_embeddings=rope(x, pos),
            position_ids=pos,
            attention_mask=mask if layer_idx == 3 else None,
            past_key_values=cache,
        )
        return y, cache, None
    probes = sorted(
        {
            i
            for i in (
                0,
                1,
                30,
                31,
                32,
                126,
                127,
                128,
                510,
                511,
                512,
                length // 2,
                length - 129,
                length - 33,
                length - 2,
                length - 1,
            )
            if 0 <= i < length
        }
    )
    mixed = []
    if layer_idx == 0:
        for start in range(0, length, 512):
            part = x[:, start : start + 512]
            a = layer.linear_attn(layer.input_layernorm(part), cache_params=cache)
            take = [i - start for i in probes if start <= i < start + part.shape[1]]
            if take:
                mixed.append(a[:, take])
        attention = torch.cat(mixed, dim=1)
    else:
        a = layer.self_attn
        keys, values = [], []
        for start in range(0, length, 2048):
            part = layer.input_layernorm(x[:, start : start + 2048])
            b, t, _ = part.shape
            key = a.k_norm(a.k_proj(part).view(b, t, cfg.num_key_value_heads, cfg.head_dim)).transpose(1, 2)
            value = a.v_proj(part).view(b, t, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
            cs = rope(part, torch.arange(start, start + t).expand(batch, -1))
            _, key = apply_rotary_pos_emb(key, key, *cs)
            keys.append(key)
            values.append(value)
        key, value = torch.cat(keys, dim=2), torch.cat(values, dim=2)
        cache.update(key, value, layer_idx)
        part = layer.input_layernorm(x[:, probes])
        q, gate = a.q_proj(part).view(batch, len(probes), cfg.num_attention_heads, 2 * cfg.head_dim).chunk(2, dim=-1)
        q = a.q_norm(q).transpose(1, 2)
        cs = rope(part, torch.tensor(probes).expand(batch, -1))
        q, _ = apply_rotary_pos_emb(q, q, *cs)
        mask = torch.where(torch.arange(length)[None, :] <= torch.tensor(probes)[:, None], 0.0, float("-inf")).to(
            x.dtype
        )[None, None]
        attention, _ = eager_attention_forward(a, q, key, value, mask, scaling=a.scaling, dropout=0.0)
        attention = a.o_proj(attention.reshape(batch, len(probes), -1) * gate.reshape(batch, len(probes), -1).sigmoid())
    residual = x[:, probes] + attention
    y = residual + layer.mlp(layer.post_attention_layernorm(residual))
    return y, cache, probes


@torch.no_grad()
def decode_reference(layer, rope, cache, x, start):
    outputs = []
    for i in range(x.shape[1]):
        pos = torch.full((x.shape[0], 1), start + i, dtype=torch.long)
        token = x[:, i : i + 1]
        outputs.append(layer(token, position_embeddings=rope(token, pos), position_ids=pos, past_key_values=cache))
    return torch.cat(outputs, dim=1) if outputs else None


@torch.no_grad()
def decode_reference_with_ties(layer, rope, cache, x, start):
    mlp = layer.mlp
    captured = {}
    gate_forward, experts_forward = mlp.gate.forward, mlp.experts.forward

    def gate(hidden):
        result = gate_forward(hidden)
        captured["route"] = result
        return result

    def experts(hidden, selected, weights):
        captured["hidden"] = hidden
        return experts_forward(hidden, selected, weights)

    mlp.gate.forward, mlp.experts.forward = gate, experts
    outputs, alternatives = [], {}
    try:
        for i in range(x.shape[1]):
            pos = torch.full((x.shape[0], 1), start + i, dtype=torch.long)
            token = x[:, i : i + 1]
            outputs.append(layer(token, position_embeddings=rope(token, pos), position_ids=pos, past_key_values=cache))
            logits, weights, selected = captured["route"]
            probs = torch.softmax(logits, dtype=torch.float, dim=-1)
            hidden = captured["hidden"]
            for row in range(probs.shape[0]):
                boundary = logits[row, selected[row, -1]].float()
                ulp = 2.0 ** (torch.floor(torch.log2(boundary.abs().clamp_min(1e-30))) - 7)
                near = (logits[row].float() - boundary).abs() <= ulp
                tied = torch.nonzero(near).flatten().tolist()
                keep = [e for e in selected[row].tolist() if not near[e]]
                slots = len(selected[row]) - len(keep)
                if len(tied) <= slots:
                    continue
                exact = bool((probs[row, tied] == probs[row, selected[row, -1]]).all())
                base = experts_forward(hidden[row : row + 1], selected[row : row + 1], weights[row : row + 1])
                deltas = []
                for choice in itertools.combinations(tied, slots):
                    alternative = torch.tensor([keep + list(choice)], dtype=selected.dtype)
                    if sorted(alternative[0].tolist()) == sorted(selected[row].tolist()):
                        continue
                    chosen_probs = probs[row, alternative[0]]
                    alt_weights = (chosen_probs / chosen_probs.sum()).to(weights.dtype)
                    delta = experts_forward(hidden[row : row + 1], alternative, alt_weights[None]) - base
                    deltas.append(delta.reshape(-1))
                alternatives[(row, i)] = dict(tied=tied, slots=slots, deltas=deltas, exact=exact)
    finally:
        mlp.gate.forward, mlp.experts.forward = gate_forward, experts_forward
    return (torch.cat(outputs, dim=1) if outputs else None), alternatives

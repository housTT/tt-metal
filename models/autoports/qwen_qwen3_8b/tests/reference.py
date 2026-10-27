# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hugging Face reference paths for the Qwen3-8B decoder layer.

``HFLayerReference`` wraps the installed ``Qwen3DecoderLayer`` with a
``DynamicCache`` so prefill, continuation and decode steps share one cache.
``chunked_layer_reference`` is a plain torch implementation of the same layer
that bounds attention memory for 40K-token inputs where the HF eager path
would materialize a full score matrix.
"""

from __future__ import annotations

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

from models.autoports.qwen_qwen3_8b.tests.weights import MODEL_ID


def load_config(attn_implementation: str = "sdpa"):
    config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    config._attn_implementation = attn_implementation
    return config


class HFLayerReference:
    """Layer-only HF reference in float32 with an explicit KV cache."""

    def __init__(self, config, layer_idx: int, state_dict: dict[str, torch.Tensor]):
        self.config = config
        self.layer_idx = layer_idx
        self.layer = Qwen3DecoderLayer(config, layer_idx).eval()
        self.layer.load_state_dict({k: v.to(torch.float32) for k, v in state_dict.items()}, strict=True)
        self.rotary = Qwen3RotaryEmbedding(config)
        self.reset()

    def reset(self) -> None:
        self.cache = DynamicCache(config=self.config)

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Run ``hidden`` ``[1, S, H]`` at absolute positions ``[start_pos, start_pos + S)``."""
        hidden = hidden.to(torch.float32)
        seq_len = hidden.shape[1]
        position_ids = torch.arange(start_pos, start_pos + seq_len).unsqueeze(0)
        cos, sin = self.rotary(hidden, position_ids)
        mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=self.cache,
            position_ids=position_ids,
        )
        return self.layer(
            hidden,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
            position_embeddings=(cos, sin),
        )


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.no_grad()
def chunked_layer_reference(
    config,
    state_dict: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    *,
    query_rows: slice | None = None,
    query_chunk: int = 1024,
) -> torch.Tensor:
    """Float32 torch reference of one layer over ``hidden`` ``[1, S, H]`` from position 0.

    Attention is evaluated one query chunk at a time against all keys so a
    40960-token input needs a bounded score buffer. ``query_rows`` limits the
    returned rows (and the attention work) to that slice of positions.
    """
    w = {k: v.to(torch.float32) for k, v in state_dict.items()}
    eps = float(config.rms_norm_eps)
    head_dim = int(config.head_dim)
    n_heads = int(config.num_attention_heads)
    n_kv = int(config.num_key_value_heads)
    groups = n_heads // n_kv
    x = hidden.to(torch.float32)[0]
    seq_len = x.shape[0]
    rows = query_rows if query_rows is not None else slice(0, seq_len)
    row_start, row_stop, _ = rows.indices(seq_len)

    normed = _rms_norm(x, w["input_layernorm.weight"], eps)
    q = (normed @ w["self_attn.q_proj.weight"].T).view(seq_len, n_heads, head_dim)
    k = (normed @ w["self_attn.k_proj.weight"].T).view(seq_len, n_kv, head_dim)
    v = (normed @ w["self_attn.v_proj.weight"].T).view(seq_len, n_kv, head_dim)
    q = _rms_norm(q, w["self_attn.q_norm.weight"], eps)
    k = _rms_norm(k, w["self_attn.k_norm.weight"], eps)
    rope_theta = (
        float(config.rope_parameters["rope_theta"])
        if getattr(config, "rope_parameters", None)
        else float(config.rope_theta)
    )
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    k = k.repeat_interleave(groups, dim=1).transpose(0, 1)
    v = v.repeat_interleave(groups, dim=1).transpose(0, 1)
    q = q.transpose(0, 1)
    scale = head_dim**-0.5
    attn_rows = []
    for q0 in range(row_start, row_stop, query_chunk):
        q1 = min(q0 + query_chunk, row_stop)
        scores = torch.matmul(q[:, q0:q1], k[:, :q1].transpose(-1, -2)) * scale
        causal = torch.ones(q1 - q0, q1, dtype=torch.bool).tril(diagonal=q0)
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        attn_rows.append(torch.matmul(probs, v[:, :q1]))
    attn = torch.cat(attn_rows, dim=1).transpose(0, 1).reshape(row_stop - row_start, n_heads * head_dim)
    attn = attn @ w["self_attn.o_proj.weight"].T
    h = x[row_start:row_stop] + attn
    normed = _rms_norm(h, w["post_attention_layernorm.weight"], eps)
    gate = torch.nn.functional.silu(normed @ w["mlp.gate_proj.weight"].T)
    up = normed @ w["mlp.up_proj.weight"].T
    out = h + (gate * up) @ w["mlp.down_proj.weight"].T
    return out.unsqueeze(0)

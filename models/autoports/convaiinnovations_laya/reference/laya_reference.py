# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from models.autoports.convaiinnovations_laya import common
from models.autoports.convaiinnovations_laya.vendor import rl_common

NEG_LOGIT = -1e4
TEMP_MIN = 0.5
TEMP_MAX = 5.0
QTYPES = rl_common.QTYPES
QTYPE_NAMES = rl_common.QTYPE_NAMES


def strict_load(model: torch.nn.Module, state_dict: dict) -> dict:
    own = model.state_dict()
    expected, got = set(own.keys()), set(state_dict.keys())
    missing = sorted(expected - got)
    unexpected = sorted(got - expected)
    bad_shape = sorted(k for k in expected & got if tuple(own[k].shape) != tuple(state_dict[k].shape))
    if missing or unexpected or bad_shape:
        raise RuntimeError(
            f"state dict mismatch: missing {len(missing)} {missing[:8]}, unexpected {len(unexpected)} {unexpected[:8]}, "
            f"bad shapes {len(bad_shape)} {bad_shape[:8]}"
        )
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"load_state_dict: {result}")
    return {"n_tensors": len(got), "missing": missing, "unexpected": unexpected, "bad_shape": bad_shape}


def build_decision_model(model_dir: str | None = None, attn_implementation: str = "eager", dtype=torch.float32):
    from transformers import AutoModel

    d = model_dir or common.model_dir()
    cfg = common.load_rl_config(d)
    ecfg = common.load_config(d)
    enc = AutoModel.from_config(ecfg, attn_implementation=attn_implementation)
    model = rl_common.DecisionModel(enc, cfg["head_layers"], len(cfg["act_costs"]) + 1)
    raw = common.load_state_dict(d, dtype=None)
    source_dtypes = sorted({str(v.dtype) for v in raw.values()})
    sd = {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in raw.items()}
    report = strict_load(model, sd)
    report["source_dtypes"] = source_dtypes
    model = model.to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, report


def _as_long(x):
    return torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x).to(torch.long)


def _as_bool(x):
    return torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x).to(torch.bool)


class LayaReference:
    def __init__(self, model_dir: str | None = None, attn_implementation: str = "eager", threads: int | None = None, dtype=torch.float32):
        self.threads = common.configure_torch_threads(threads)
        self.model_dir = model_dir or common.model_dir()
        self.attn_implementation = attn_implementation
        self.dtype = dtype
        t0 = time.perf_counter()
        self.model, self.cfg, self.load_report = build_decision_model(self.model_dir, attn_implementation, dtype)
        self.load_seconds = time.perf_counter() - t0
        self.tok = common.load_tokenizer(self.model_dir)
        self.encoder_config = self.model.encoder.config
        self.max_len = int(self.cfg["max_len"])
        self.head_max_len = int(self.cfg["head_max_len"])
        self.temperature = [float(t) for t in self.cfg.get("temperature", [1.0, 1.0, 1.0])]
        self.temperature_by_options = {k: float(v) for k, v in self.cfg.get("temperature_by_options", {}).items()}
        self.temperature_clamped = [clamp_temperature(t) for t in self.temperature]
        self.temperature_by_options_clamped = {k: clamp_temperature(v) for k, v in self.temperature_by_options.items()}

    def shapes(self) -> dict:
        ec = self.encoder_config
        return {
            "hidden_size": int(ec.hidden_size),
            "num_hidden_layers": int(ec.num_hidden_layers),
            "num_attention_heads": int(ec.num_attention_heads),
            "head_dim": int(ec.hidden_size // ec.num_attention_heads),
            "intermediate_size": int(ec.intermediate_size),
            "vocab_size": int(ec.vocab_size),
            "pad_token_id": int(ec.pad_token_id),
            "head_layers": int(self.cfg["head_layers"]),
            "head_ffn": int(4 * ec.hidden_size),
            "n_act": int(len(self.cfg["act_costs"]) + 1),
            "max_len": self.max_len,
            "head_max_len": self.head_max_len,
            "inputs": {
                "input_ids": "[B, S] int64",
                "attention_mask": "[B, S] int64 (1 real, 0 pad)",
                "marker_pos": "[B, kmax] int64",
                "marker_mask": "[B, kmax] bool",
                "qtype": "[B] int64 (choice 0, score 1, noul 2)",
            },
            "outputs": {"logits": "[B, kmax] float32 (-1e4 where marker_mask is False)", "act_logits": "[B, 2] float32"},
            "checkpoint_dtypes": self.load_report["source_dtypes"],
            "compute_dtype": str(self.dtype),
            "attn_implementation": self.attn_implementation,
        }

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits, act = self.model(_as_long(input_ids), _as_long(attention_mask), _as_long(marker_pos), _as_bool(marker_mask), _as_long(qtype))
        return logits.float(), act.float()

    __call__ = forward

    @torch.inference_mode()
    def forward_with_hidden(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        input_ids, attention_mask = _as_long(input_ids), _as_long(attention_mask)
        marker_pos, marker_mask, qtype = _as_long(marker_pos), _as_bool(marker_mask), _as_long(qtype)
        m = self.model
        out = m.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = {"encoder": list(out.hidden_states)}
        h = out.last_hidden_state
        h = h + m.type_emb(qtype)[:, None, :]
        hidden["after_type_emb"] = h
        hidden["head"] = []
        pad = ~attention_mask.bool()
        for layer in m.head.layers:
            h = layer(h, src_key_padding_mask=pad)
            hidden["head"].append(h)
        logits, act = _tail(m, h, marker_pos, marker_mask)
        return logits.float(), act.float(), hidden

    @torch.inference_mode()
    def forward_explicit_head(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        input_ids, attention_mask = _as_long(input_ids), _as_long(attention_mask)
        marker_pos, marker_mask, qtype = _as_long(marker_pos), _as_bool(marker_mask), _as_long(qtype)
        m = self.model
        h = m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + m.type_emb(qtype)[:, None, :]
        for layer in m.head.layers:
            h = head_layer_explicit(layer, h, attention_mask)
        logits, act = _tail(m, h, marker_pos, marker_mask)
        return logits.float(), act.float()

    def encode(self, state, questions: dict, max_len: int | None = None, head_max_len: int | None = None) -> list:
        max_len = max_len or self.max_len
        head_max_len = head_max_len or self.head_max_len
        case = {"state": state, "questions": questions, "id": None, "index": None, "workflow": None}
        return common.case_items(self.tok, case, max_len=max_len, head_max_len=head_max_len)

    def collate(self, items: list, seq_len: int | None = None) -> dict:
        return common.collate(items, self.tok.pad_token_id, seq_len=seq_len)

    def decide(self, state, questions: dict, shape: str = "pip", clamp: bool = True) -> dict:
        items = self.encode(state, questions)
        b = self.collate(items)
        logits, act = self.forward(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        return decode_items(items, logits.numpy(), act_probs(act), self.temperature, self.temperature_by_options, shape=shape, clamp=clamp)


def _tail(m, h, marker_pos, marker_mask):
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    g = torch.gather(h, 1, idx)
    logits = m.scorer(g).squeeze(-1).float()
    logits = logits.masked_fill(~marker_mask, NEG_LOGIT)
    p = torch.softmax(logits.detach(), -1)
    k = marker_mask.sum(-1).clamp(min=2).float()
    ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
    top2 = p.topk(2, -1).values
    feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
    pooled = h[:, 0].float()
    act = m.act_head(torch.cat([pooled, feats], -1))
    return logits, act


def head_layer_explicit(layer: torch.nn.TransformerEncoderLayer, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    sa = layer.self_attn
    B, S, D = x.shape
    H = sa.num_heads
    hd = D // H
    h = F.layer_norm(x, (D,), layer.norm1.weight, layer.norm1.bias, layer.norm1.eps)
    qkv = F.linear(h, sa.in_proj_weight, sa.in_proj_bias)
    q, k, v = qkv.split(D, dim=-1)
    q = q.view(B, S, H, hd).transpose(1, 2)
    k = k.view(B, S, H, hd).transpose(1, 2)
    v = v.view(B, S, H, hd).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) * (hd**-0.5)
    key_pad = (_as_long(attention_mask) == 0)[:, None, None, :]
    scores = scores.masked_fill(key_pad, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, D)
    x = x + F.linear(ctx, sa.out_proj.weight, sa.out_proj.bias)
    h2 = F.layer_norm(x, (D,), layer.norm2.weight, layer.norm2.bias, layer.norm2.eps)
    ff = F.linear(F.relu(F.linear(h2, layer.linear1.weight, layer.linear1.bias)), layer.linear2.weight, layer.linear2.bias)
    return x + ff


def scorer_explicit(scorer: torch.nn.Sequential, h: torch.Tensor) -> torch.Tensor:
    ln, lin1, lin2 = scorer[0], scorer[1], scorer[3]
    y = F.layer_norm(h, (h.shape[-1],), ln.weight, ln.bias, ln.eps)
    y = F.gelu(F.linear(y, lin1.weight, lin1.bias))
    return F.linear(y, lin2.weight, lin2.bias).squeeze(-1)


def act_probs(act_logits) -> np.ndarray:
    return torch.softmax(torch.as_tensor(np.asarray(act_logits) if not torch.is_tensor(act_logits) else act_logits).float(), -1).cpu().numpy()


def temp_bucket(qtype: int, k: int) -> str:
    return rl_common.temp_bucket(qtype, k)


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    if isinstance(t, bool):
        return 1.0
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):
        return 1.0
    return min(hi, max(lo, t))


def temperature_for(temperature, temperature_by_options, qtype: int, k: int, clamp: bool = True) -> float:
    t = temperature_by_options.get(temp_bucket(qtype, k), temperature[int(qtype)])
    return clamp_temperature(t) if clamp else float(t)


def scaled_probs(logits_k, t: float) -> np.ndarray:
    z = np.asarray(logits_k, dtype=np.float32) / t
    p = np.exp(z - z.max())
    return p / p.sum()


def confidence_from_probs(p, k: int) -> float:
    return rl_common.confidence_from_probs(np.asarray(p), k)


def answer_confidence(p, k: int) -> float:
    if k < 1:
        return 1.0
    return float(np.clip(np.max(np.asarray(p)[:k]), 0.0, 1.0))


def expected_score(p) -> float:
    p = np.asarray(p)
    return float((np.arange(len(p)) * p).sum())


def p_true(p) -> float:
    return float(np.asarray(p)[1])


def decode_answer(q: dict, logits_k, act_row, temperature, temperature_by_options, shape: str = "pip", clamp: bool = True) -> dict:
    k = len(np.asarray(logits_k))
    qt = QTYPES[q["t"]]
    p = scaled_probs(logits_k, temperature_for(temperature, temperature_by_options, qt, k, clamp=clamp))
    if shape == "hub":
        ext = {"act_probability": float(act_row[0])}
        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            return {
                "type": "choice",
                "choice": keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                "confidence": round(confidence_from_probs(p, k), 4),
                "rl_agent": ext,
            }
        if q["t"] == "score":
            return {
                "type": "score",
                "score": round(expected_score(p), 4),
                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": round(confidence_from_probs(p, k), 4),
                "rl_agent": ext,
            }
        return {"type": "noul", "noul": round(p_true(p), 4), "rl_agent": ext}
    ext = {"act_probability": round(float(act_row[0]), 4)}
    ans_conf = round(answer_confidence(p, k), 4)
    if q["t"] == "choice":
        keys = list(q["crit"].keys())
        return {
            "type": "choice",
            "choice": keys[int(p.argmax())],
            "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
            "confidence": round(confidence_from_probs(p, k), 4),
            "answer_confidence": ans_conf,
            "action": ext,
        }
    if q["t"] == "score":
        return {
            "type": "score",
            "score": round(expected_score(p), 4),
            "legend": {str(i): (c if isinstance(c, str) else str(c)) for i, c in enumerate(q["crit"])},
            "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
            "confidence": round(confidence_from_probs(p, k), 4),
            "answer_confidence": ans_conf,
            "action": ext,
        }
    pt = p_true(p)
    return {
        "type": "noul",
        "noul": round(pt, 4),
        "confidence": round(max(pt, 1.0 - pt), 4),
        "answer_confidence": ans_conf,
        "action": ext,
    }


def decode_items(items: list, logits: np.ndarray, act_p: np.ndarray, temperature, temperature_by_options, shape: str = "pip", clamp: bool = True) -> dict:
    logits = np.asarray(logits, dtype=np.float32)
    answers = {}
    for r, it in enumerate(items):
        k = len(it["markers"])
        answers[it["qid"]] = decode_answer(it["q"], logits[r, :k], act_p[r], temperature, temperature_by_options, shape=shape, clamp=clamp)
    return answers

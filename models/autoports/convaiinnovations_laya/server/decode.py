# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..vendor.rl_common import QTYPES, confidence_from_probs, temp_bucket

MODEL_NAME = "laya-rl-agent"
TEMP_MIN = 0.5
TEMP_MAX = 5.0
MASKED_LOGIT = -1e4
GATE_PASSED = "passed"
GATE_ABSTAINED = "abstained"
GATE_UNEVALUATED = "unevaluated"


def clamp_temperature(t: Any, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    if isinstance(t, bool):
        return 1.0
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):
        return 1.0
    return min(hi, max(lo, t))


class Temperatures:
    def __init__(self, cfg: Dict[str, Any], clamp: bool = True):
        raw = cfg.get("temperature", [1.0, 1.0, 1.0])
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ValueError("rl_agent_config.json: temperature must be a list of 3 floats, got %r" % (raw,))
        self.clamp = bool(clamp)
        self.raw = [float(t) for t in raw]
        self.by_options_raw = {str(k): float(v) for k, v in (cfg.get("temperature_by_options") or {}).items()}
        if self.clamp:
            self.temperature = [clamp_temperature(t) for t in self.raw]
            self.by_options = {k: clamp_temperature(v) for k, v in self.by_options_raw.items()}
        else:
            self.temperature = list(self.raw)
            self.by_options = dict(self.by_options_raw)

    def for_question(self, qtype: int, k: int) -> float:
        return self.by_options.get(temp_bucket(qtype, k), self.temperature[int(qtype)])

    def describe(self) -> Dict[str, Any]:
        changed = sorted(k for k in self.by_options if self.by_options[k] != self.by_options_raw[k])
        return {
            "clamp": self.clamp,
            "range": [TEMP_MIN, TEMP_MAX],
            "temperature": self.temperature,
            "temperature_by_options": self.by_options,
            "clamped_buckets": changed,
        }


def render_criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def scaled_softmax(z: np.ndarray, t: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / float(t)
    p = np.exp(z - z.max())
    return p / p.sum()


def answer_confidence(p: np.ndarray, k: int) -> float:
    if k < 1:
        return 1.0
    return float(np.clip(np.max(p[:k]), 0.0, 1.0))


def unpermute_probs(p: np.ndarray, option_order: Optional[Sequence[int]]) -> np.ndarray:
    if option_order is None or len(option_order) != len(p):
        return p
    canonical = np.empty_like(p)
    canonical[np.asarray(option_order, dtype=int)] = p
    return canonical


def decode_answer(q: Dict[str, Any], logits_row: np.ndarray, act_prob_row: np.ndarray, k: int, temps: Temperatures) -> Dict[str, Any]:
    qt = QTYPES[q["t"]]
    t_scale = temps.for_question(qt, k)
    p = scaled_softmax(np.asarray(logits_row[:k], dtype=np.float32), t_scale)
    p = unpermute_probs(p, q.get("option_order"))
    ans_conf = round(answer_confidence(p, k), 4)
    ext = {"act_probability": round(float(act_prob_row[0]), 4)}
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
            "score": round(float((np.arange(k) * p).sum()), 4),
            "legend": {str(i): render_criterion(c) for i, c in enumerate(q["crit"])},
            "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
            "confidence": round(confidence_from_probs(p, k), 4),
            "answer_confidence": ans_conf,
            "action": ext,
        }
    return {
        "type": "noul",
        "noul": round(float(p[1]), 4),
        "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
        "answer_confidence": ans_conf,
        "action": ext,
    }


def decode_answers(
    ids: Sequence[str],
    internal: Dict[str, Dict[str, Any]],
    items: Sequence[Dict[str, Any]],
    logits: np.ndarray,
    act_probs: np.ndarray,
    temps: Temperatures,
) -> Dict[str, Dict[str, Any]]:
    answers: Dict[str, Dict[str, Any]] = {}
    for r, qid in enumerate(ids):
        k = len(items[r]["markers"])
        answers[qid] = decode_answer(internal[qid], logits[r], act_probs[r], k, temps)
    return answers


def collapsed_options(ids: Sequence[str], items: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[int]]]:
    out: Dict[str, Dict[str, Optional[int]]] = {}
    for qid, item in zip(ids, items):
        stats = item.get("options")
        if stats and stats["options_distinct"] < stats["options"]:
            out[qid] = {
                "total": stats["options"],
                "distinct": stats["options_distinct"],
                "tokens_per_option": stats["tokens_per_option"],
            }
    return out


def usage_for_state(ids: Sequence[str], items: Sequence[Dict[str, Any]], n_tokens: int) -> Dict[str, Any]:
    stats = [item["state_stats"] for item in items]
    dropped = max(s["state_tokens_dropped"] for s in stats)
    usage: Dict[str, Any] = {
        "input_tokens": int(n_tokens),
        "output_tokens": 0,
        "state_tokens": int(stats[0]["state_tokens"]),
        "state_tokens_dropped": int(dropped),
        "truncated": dropped > 0,
        "truncated_questions": [qid for qid, s in zip(ids, stats) if s["truncated"]],
    }
    collapsed = collapsed_options(ids, items)
    if collapsed:
        usage["options"] = collapsed
    return usage


def empty_result() -> Dict[str, Any]:
    return {"model": MODEL_NAME, "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}


def _check_one_threshold(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError("min_confidence must be a float in [0.0, 1.0], got %r" % (v,))
    return float(v)


def check_min_confidence(v: Any):
    if isinstance(v, dict):
        if not v:
            raise ValueError("a min_confidence map must be a non-empty dict of bucket -> float, got %r" % (v,))
        out: Dict[str, float] = {}
        for key, val in v.items():
            if not isinstance(key, str):
                raise ValueError("min_confidence map keys must be strings like 'choice:3-5', got %r" % (key,))
            out[key] = _check_one_threshold(val)
        return out
    return _check_one_threshold(v)


def _usable(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def gate_confidence(answer: Dict[str, Any]) -> Optional[float]:
    conf = _usable(answer.get("answer_confidence"))
    if conf is not None:
        return conf
    return _usable(answer.get("confidence"))


def _option_bucket(answer: Dict[str, Any]) -> Optional[str]:
    qt = answer.get("type")
    if qt not in ("choice", "score", "noul"):
        return None
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and probs:
        k = len(probs)
    elif qt == "noul":
        k = 2
    else:
        return None
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (qt, size)


def resolve_min_confidence(answer: Dict[str, Any], thresholds: Dict[str, float], default: float = 0.0) -> float:
    key = _option_bucket(answer)
    if key is not None and key in thresholds:
        return thresholds[key]
    return thresholds.get("default", default)


def flag_low_confidence(results: List[Dict[str, Any]], min_confidence: Any) -> None:
    is_map = isinstance(min_confidence, dict)
    if not is_map and min_confidence == 0.0:
        return
    for res in results:
        answers = res.get("answers") if isinstance(res, dict) else None
        if not isinstance(answers, dict):
            continue
        for a in answers.values():
            if not isinstance(a, dict):
                continue
            conf = gate_confidence(a)
            if conf is None:
                a.pop("low_confidence", None)
                continue
            thr = resolve_min_confidence(a, min_confidence) if is_map else min_confidence
            if conf < thr:
                a["low_confidence"] = True
            else:
                a.pop("low_confidence", None)


def apply_confidence_gate(results: List[Dict[str, Any]], min_confidence: Any = None) -> None:
    if min_confidence is None:
        return
    is_map = isinstance(min_confidence, dict)
    flag_low_confidence(results, min_confidence)
    for res in results:
        answers = res.get("answers") if isinstance(res, dict) else None
        if not isinstance(answers, dict):
            continue
        for a in answers.values():
            if not isinstance(a, dict):
                continue
            if not is_map and min_confidence == 0.0:
                a.pop("low_confidence", None)
            if a.get("low_confidence"):
                a["abstention"] = GATE_ABSTAINED
            elif gate_confidence(a) is None:
                a["abstention"] = GATE_UNEVALUATED
            else:
                a["abstention"] = GATE_PASSED
            a["abstention_threshold"] = float(resolve_min_confidence(a, min_confidence) if is_map else min_confidence)


def aggregate_usage(results: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    def total(key: str) -> int:
        return sum(int((r.get("usage") or {}).get(key, 0) or 0) for r in results)

    return {"input_tokens": total("input_tokens"), "output_tokens": total("output_tokens")}

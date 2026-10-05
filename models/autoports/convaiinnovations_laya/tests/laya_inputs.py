# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional

import torch

from models.autoports.convaiinnovations_laya.vendor.rl_common import QTYPES, build_sequence, collate_items

PARQUET = os.environ.get(
    "LAYA_TYPED_DECISIONS_PARQUET", "/home/hous/dev/laya/evals/typed_decisions/data/test-00000-of-00001.parquet"
)
MODEL_DIR = os.environ.get("LAYA_MODEL_DIR", "/home/hous/dev/laya/state/laya_models/laya")
MAX_LEN = 512
HEAD_MAX_LEN = 192


@lru_cache(maxsize=1)
def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "tokenizer"))


@lru_cache(maxsize=1)
def load_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(os.path.join(MODEL_DIR, "encoder"))


@lru_cache(maxsize=1)
def load_cases() -> List[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(PARQUET).to_pylist()
    for r in rows:
        r["questions"] = json.loads(r["questions"])
        r["gold"] = json.loads(r["gold"])
    return rows


def to_internal(qdef: dict) -> dict:
    crit = qdef.get("criteria")
    t = qdef["type"]
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"] if isinstance(qdef["instructions"], str) else json.dumps(qdef["instructions"])
    return {"t": t, "ins": ins, "crit": crit}


def question_items(n_questions: int, max_len: int = MAX_LEN, head_max_len: int = HEAD_MAX_LEN, offset: int = 0):
    tok = load_tokenizer()
    items = []
    for case in load_cases()[offset:]:
        for qid, qdef in case["questions"].items():
            q = to_internal(qdef)
            ids, markers = build_sequence(tok, case["state"], q, max_len, head_max_len)
            items.append(
                {
                    "ids": ids,
                    "markers": markers,
                    "qtype": QTYPES[q["t"]],
                    "target": [0.0] * len(markers),
                    "label": -1,
                    "episode": 0,
                    "ep_step": 0,
                    "ep_len": 1,
                    "src": case["id"],
                    "case_id": case["id"],
                    "qid": qid,
                    "n_options": len(markers),
                }
            )
            if len(items) >= n_questions:
                return items
    return items


def collate(items: List[dict], seq_len: int) -> Dict[str, torch.Tensor]:
    tok = load_tokenizer()
    b = collate_items([items], tok.pad_token_id)
    n, L = b["input_ids"].shape
    if L > seq_len:
        raise ValueError(f"longest sequence {L} exceeds the bucket {seq_len}")
    ids = torch.full((n, seq_len), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((n, seq_len), dtype=torch.long)
    ids[:, :L] = b["input_ids"]
    att[:, :L] = b["attention_mask"]
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": b["marker_pos"],
        "marker_mask": b["marker_mask"],
        "qtype": b["qtype"],
        "meta": b["meta"],
        "lengths": b["attention_mask"].sum(-1),
    }


def build_inputs(batch_size: int = 1, seq_len: int = MAX_LEN, offset: int = 0, fill: bool = False):
    """Real typed-decisions sequences, padded to `seq_len`. With `fill` the last row is extended with state text so the batch has one unpadded row."""
    items = question_items(batch_size, offset=offset)
    if fill:
        tok = load_tokenizer()
        filler = " ".join(json.loads(c["state"]).get("task", "") if c["state"].startswith("{") else c["state"] for c in load_cases()[:40])
        extra = tok(filler, add_special_tokens=False)["input_ids"]
        ids = items[-1]["ids"]
        room = seq_len - len(ids)
        if room > 0:
            items[-1]["ids"] = ids[:-1] + extra[: room] + [ids[-1]]
    return collate(items, seq_len)


def pad_row_host(attention_mask: torch.Tensor, neg: float = -1e30) -> torch.Tensor:
    pad = torch.zeros(attention_mask.shape, dtype=torch.float32)
    return pad.masked_fill(attention_mask == 0, neg)[:, None, None, :]

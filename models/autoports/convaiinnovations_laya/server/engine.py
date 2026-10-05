# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..vendor.rl_common import QTYPES, build_sequence, collate_items, render_options, serialize_state
from .decode import (
    MODEL_NAME,
    Temperatures,
    apply_confidence_gate,
    decode_answers,
    empty_result,
    usage_for_state,
)

log = logging.getLogger("laya.server")

DEFAULT_HF_MODEL = "convaiinnovations/laya"
DEFAULT_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
ALLOW_PATTERNS = ["*.json", "tokenizer/*", "encoder/*", "model.safetensors"]
REQUIRED_FILES = ("rl_agent_config.json", "model.safetensors", "encoder/config.json", "tokenizer/tokenizer.json")
DEFAULT_ROW_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_MAX_ROWS = 64
DEFAULT_MAX_BATCH_TOKENS = 65536
DEFAULT_MAX_BATCH_STATES = 64
DEFAULT_MAX_QUESTIONS = 64
DEFAULT_MAX_STATE_CHARS = 50000
DEFAULT_MAX_CHOICE_OPTIONS = 100
DEFAULT_MAX_SCORE_LEVELS = 32
DEFAULT_MAX_TOTAL_OPTIONS = 512
OPTION_TOKEN_CAP = 48
BACKENDS = ("tt", "cpu")


class BadRequest(ValueError):
    status = 400


class RequestError(ValueError):
    status = 422


class LimitError(ValueError):
    status = 413


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    try:
        n = int(str(v).strip())
    except ValueError:
        log.warning("invalid %s=%r; using %d", name, v, default)
        return default
    if n <= 0:
        log.warning("%s must be positive, got %d; using %d", name, n, default)
        return default
    return n


def env_int_list(name: str, default: Sequence[int]) -> List[int]:
    v = os.environ.get(name)
    if v in (None, ""):
        return sorted(set(int(x) for x in default))
    out = sorted({int(x.strip()) for x in str(v).split(",") if x.strip()})
    if not out or min(out) <= 0:
        raise ValueError("%s must be a comma list of positive integers, got %r" % (name, v))
    return out


def resolve_model_dir() -> str:
    model_dir = os.environ.get("LAYA_MODEL_DIR") or ""
    if not model_dir:
        from huggingface_hub import snapshot_download

        repo = os.environ.get("HF_MODEL") or DEFAULT_HF_MODEL
        revision = os.environ.get("LAYA_REVISION") or None
        try:
            model_dir = snapshot_download(repo, revision=revision, allow_patterns=ALLOW_PATTERNS, local_files_only=True)
        except Exception as e:
            raise FileNotFoundError(
                "no local snapshot of %s (revision %s) under HF_HOME=%s; set LAYA_MODEL_DIR or pull the weights first: %s"
                % (repo, revision or "default", os.environ.get("HF_HOME", "~/.cache/huggingface"), e)
            ) from e
    sub = os.environ.get("LAYA_SUBFOLDER") or ""
    if sub:
        model_dir = os.path.join(model_dir, sub)
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(model_dir, f))]
    if missing:
        raise FileNotFoundError("model dir %s lacks %s" % (model_dir, ", ".join(missing)))
    return model_dir


def load_cfg(model_dir: str) -> Dict[str, Any]:
    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))


def _option_count(qdef: Dict[str, Any]) -> int:
    if qdef.get("type") == "noul":
        return 2
    crit = qdef.get("criteria")
    return len(crit) if isinstance(crit, (dict, list, tuple)) else 0


def check_question(qid: Any, qdef: Any) -> None:
    if qid is None:
        raise RequestError("question id must not be None")
    if not isinstance(qid, (str, int)) or (isinstance(qid, str) and not qid.strip()):
        raise RequestError("question id must be a non-empty string, got %r" % (qid,))
    if not isinstance(qdef, dict):
        raise RequestError("question %r: definition must be a dict, got %s" % (qid, type(qdef).__name__))
    t = qdef.get("type")
    if not isinstance(t, str) or t not in QTYPES:
        raise RequestError("question %r: unknown type %r; use one of %s" % (qid, t, sorted(QTYPES)))
    if "instructions" not in qdef:
        raise RequestError("question %r: no 'instructions'; add the text the model should answer" % (qid,))
    ins = qdef["instructions"]
    if ins is None:
        raise RequestError("question %r: 'instructions' must not be None; add the text the model should answer" % (qid,))
    if isinstance(ins, str) and not ins.strip():
        raise RequestError("question %r: 'instructions' must not be empty; add the text the model should answer" % (qid,))
    if isinstance(ins, (list, dict)) and not ins:
        raise RequestError("question %r: 'instructions' must not be empty; add the text the model should answer" % (qid,))
    if not isinstance(ins, (str, dict, list, int, float)):
        raise RequestError("question %r: 'instructions' must be a string, dict, or list, got %s" % (qid, type(ins).__name__))
    crit = qdef.get("criteria")
    if t == "choice":
        if not isinstance(crit, (dict, list)):
            raise RequestError(
                "question %r: a choice question takes 'criteria' as a dict of label -> description, or a list of labels" % (qid,)
            )
        if not crit:
            raise RequestError("question %r: a choice question needs at least one criterion" % (qid,))
        for i, label in enumerate(crit if isinstance(crit, list) else crit.keys()):
            if label is not None and not isinstance(label, (str, int, float, bool)):
                raise RequestError(
                    "question %r: choice label %d is a %s; a label is rendered as option text and used as the answer key, "
                    "so it must be a scalar (a string, number or bool), got %r" % (qid, i, type(label).__name__, label)
                )
            if label is None:
                raise RequestError(
                    "question %r: choice label %d is null; a label is rendered as option text and used as the answer key, "
                    "so it must be a string, number or bool" % (qid, i)
                )
        if isinstance(crit, list):
            keys: Dict[Any, int] = {}
            for i, label in enumerate(crit):
                try:
                    first = keys[label]
                except TypeError as exc:
                    raise RequestError(
                        "question %r: choice label %d (%r) cannot be an answer key because it is unhashable" % (qid, i, label)
                    ) from exc
                except KeyError:
                    keys[label] = i
                else:
                    raise RequestError(
                        "question %r: choice label %d (%r) repeats label %d; the labels are the answer keys, so every option "
                        "needs its own (1, 1.0 and True are one key)" % (qid, i, label, first)
                    )
    elif t == "score":
        if not isinstance(crit, list):
            raise RequestError(
                "question %r: a score question takes 'criteria' as a list of level descriptions, index 0 first" % (qid,)
            )
        if not crit:
            raise RequestError("question %r: a score question needs at least one level" % (qid,))
        if None in crit:
            raise RequestError(
                "question %r: score level %d is null; give every level a description, index 0 first" % (qid, crit.index(None))
            )
    elif crit is not None and not isinstance(crit, dict):
        raise RequestError(
            "question %r: a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it"
            % (qid,)
        )
    elif isinstance(crit, dict):
        keys_set = {str(k).lower() for k in crit}
        if not keys_set <= {"true", "false"}:
            raise RequestError(
                "question %r: a noul question takes 'criteria' keyed only 'true'/'false' (either or both, and omitted is "
                "fine), got %s" % (qid, sorted(keys_set))
            )
    if "option_order" in qdef:
        order = qdef["option_order"]
        n = _option_count(qdef)
        if (
            not isinstance(order, (list, tuple))
            or len(order) != n
            or sorted(int(i) for i in order if isinstance(i, int) and not isinstance(i, bool)) != list(range(n))
        ):
            raise RequestError(
                "question %r: 'option_order' must be a permutation of range(%d), one slot per option, each option once, got %r"
                % (qid, n, order)
            )
    if "labels" in qdef:
        raise RequestError(
            "question %r: 'labels' is not supported by this server (the vendored sequence builder renders noul options as "
            "false/true); drop 'labels' and keep 'criteria'" % (qid,)
        )


def to_internal(qdef: Dict[str, Any]) -> Dict[str, Any]:
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    elif t == "noul" and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins, ensure_ascii=False)
    q: Dict[str, Any] = {"t": t, "ins": ins, "crit": crit}
    if "option_order" in qdef:
        q["option_order"] = [int(i) for i in qdef["option_order"]]
    return q


def state_length(state: Any) -> int:
    if isinstance(state, str):
        return len(state)
    try:
        return len(json.dumps(state, ensure_ascii=False))
    except (TypeError, ValueError, RecursionError):
        raise BadRequest("'state' must be JSON-serializable")


class Limits:
    def __init__(self):
        self.max_rows = env_int("LAYA_MAX_ROWS", DEFAULT_MAX_ROWS)
        self.max_batch_tokens = env_int("LAYA_MAX_BATCH_TOKENS", DEFAULT_MAX_BATCH_TOKENS)
        self.max_batch_states = env_int("LAYA_MAX_BATCH_STATES", DEFAULT_MAX_BATCH_STATES)
        self.max_questions = env_int("LAYA_MAX_QUESTIONS", DEFAULT_MAX_QUESTIONS)
        self.max_state_chars = env_int("LAYA_MAX_STATE_CHARS", DEFAULT_MAX_STATE_CHARS)
        self.max_choice_options = env_int("LAYA_MAX_CHOICE_OPTIONS", DEFAULT_MAX_CHOICE_OPTIONS)
        self.max_score_levels = env_int("LAYA_MAX_SCORE_LEVELS", DEFAULT_MAX_SCORE_LEVELS)
        self.max_total_options = env_int("LAYA_MAX_TOTAL_OPTIONS", DEFAULT_MAX_TOTAL_OPTIONS)
        self.token_budget = env_int("LAYA_MAX_TOKEN_BUDGET", 0) if os.environ.get("LAYA_MAX_TOKEN_BUDGET") else 0

    def describe(self) -> Dict[str, int]:
        return {
            "max_rows": self.max_rows,
            "max_batch_tokens": self.max_batch_tokens,
            "max_batch_states": self.max_batch_states,
            "max_questions": self.max_questions,
            "max_state_chars": self.max_state_chars,
            "max_choice_options": self.max_choice_options,
            "max_score_levels": self.max_score_levels,
            "max_total_options": self.max_total_options,
            "max_token_budget": self.token_budget,
        }


def check_request_limits(state: Any, questions: Any, limits: Limits) -> None:
    if state is None:
        raise BadRequest("'state' is required")
    if not isinstance(questions, dict):
        raise BadRequest("'questions' must be an object")
    if len(questions) > limits.max_questions:
        raise LimitError("too many questions (%d > %d)" % (len(questions), limits.max_questions))
    total = 0
    for qid, question in questions.items():
        if not isinstance(question, dict):
            continue
        crit = question.get("criteria")
        qtype = question.get("type")
        if qtype == "choice" and isinstance(crit, (dict, list)):
            total += len(crit)
            if len(crit) > limits.max_choice_options:
                raise LimitError("too many choice options for %r (%d > %d)" % (qid, len(crit), limits.max_choice_options))
        elif qtype == "score" and isinstance(crit, list):
            total += len(crit)
            if len(crit) > limits.max_score_levels:
                raise LimitError("too many score levels for %r (%d > %d)" % (qid, len(crit), limits.max_score_levels))
            if None in crit:
                raise RequestError(
                    "score question %r has a null level at index %d; give every level a description" % (qid, crit.index(None))
                )
    if total > limits.max_total_options:
        raise LimitError("too many answer options across questions (%d > %d)" % (total, limits.max_total_options))
    n = state_length(state)
    if n > limits.max_state_chars:
        raise LimitError("state too large (%d > %d chars)" % (n, limits.max_state_chars))


def check_batch_limits(states: Any, questions: Any, limits: Limits) -> None:
    if not isinstance(states, list) or len(states) == 0:
        raise BadRequest("'states' must be a non-empty list")
    if len(states) > limits.max_batch_states:
        raise LimitError("too many states in batch (%d > %d)" % (len(states), limits.max_batch_states))
    for state in states:
        check_request_limits(state, questions, limits)


def check_budget_param(body: Dict[str, Any], key: str, cap: int) -> Optional[int]:
    if key not in body or body[key] is None:
        return None
    val = body[key]
    if not isinstance(val, int) or isinstance(val, bool):
        raise RequestError("%s must be an integer" % key)
    if val <= 0:
        raise RequestError("%s must be a positive integer" % key)
    if val > cap:
        raise RequestError("%s exceeds server limit (%d > %d)" % (key, val, cap))
    return val


def head_stats(tok, q: Dict[str, Any], head_max_len: int) -> Dict[str, Any]:
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = q.get("option_order") or list(range(len(opts)))
    opt_ids = []
    for i in order:
        toks = tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:OPTION_TOKEN_CAP]
        opt_ids.append([tok.mask_token_id] + toks)
    per = None
    if head_max_len - sum(len(o) for o in opt_ids) < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
    return {"options": len(opt_ids), "options_distinct": len({tuple(o) for o in opt_ids}), "tokens_per_option": per}


def encode_state(tok, state: Any, ids: Sequence[str], internal: Dict[str, Dict[str, Any]], max_len: int, head_max_len: int) -> List[Dict[str, Any]]:
    truncate_left = isinstance(state, list)
    state_ids = tok(serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
    items = []
    for qid in ids:
        q = internal[qid]
        order = q.get("option_order")
        seq, markers = build_sequence(tok, state, q, max_len, head_max_len, option_order=order, truncate_left=truncate_left)
        n_opts = len(render_options(q))
        if len(markers) != n_opts:
            raise RequestError(
                "question %r: only %d of its %d option markers fit in max_len=%d with head_max_len=%d spent on the question; "
                "lower head_max_len, raise max_len, or use fewer options" % (qid, len(markers), n_opts, max_len, head_max_len)
            )
        head_only, _ = build_sequence(tok, "", q, max_len, head_max_len, option_order=order)
        head_len = len(head_only) - 1
        room = max(0, max_len - head_len - 1)
        used = min(len(state_ids), room)
        items.append(
            {
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES[q["t"]],
                "target": [0.0] * len(markers),
                "label": -1,
                "episode": 0,
                "ep_step": 0,
                "ep_len": 1,
                "src": "api",
                "options": head_stats(tok, q, head_max_len),
                "state_stats": {
                    "state_tokens": len(state_ids),
                    "state_tokens_used": used,
                    "state_tokens_dropped": len(state_ids) - used,
                    "truncated": used < len(state_ids),
                },
            }
        )
    return items


class Buckets:
    def __init__(self, rows: Sequence[int], seqs: Sequence[int]):
        self.rows = sorted({int(r) for r in rows})
        self.seqs = sorted({int(s) for s in seqs})
        if not self.rows or not self.seqs:
            raise ValueError("row and seq buckets must be non-empty")

    @property
    def max_rows(self) -> int:
        return self.rows[-1]

    @property
    def max_seq(self) -> int:
        return self.seqs[-1]

    def rows_for(self, n: int) -> int:
        for r in self.rows:
            if r >= n:
                return r
        raise LimitError("%d rows exceed the largest row bucket %d" % (n, self.max_rows))

    def seq_for(self, length: int) -> int:
        for s in self.seqs:
            if s >= length:
                return s
        raise RequestError("a sequence of %d tokens exceeds the served context of %d tokens" % (length, self.max_seq))

    @classmethod
    def from_shapes(cls, shapes: Dict[str, Any]) -> "Buckets":
        rows = shapes.get("row_buckets")
        seqs = shapes.get("seq_buckets")
        warm = shapes.get("warm_shapes") or []
        if not rows:
            rows = sorted({int(w[0]) for w in warm}) or list(DEFAULT_ROW_BUCKETS)
        if not seqs:
            seqs = sorted({int(w[1]) for w in warm})
        if not seqs:
            raise ValueError("backend.shapes() reports neither seq_buckets nor warm_shapes")
        return cls(rows, seqs)


def pad_batch(b: Dict[str, torch.Tensor], rows: int, seq: int, pad_id: int, keep_one_token: bool):
    n, length = b["input_ids"].shape
    kmax = b["marker_pos"].shape[1]
    ids = torch.full((rows, seq), pad_id, dtype=torch.long)
    ids[:n, :length] = b["input_ids"]
    att = torch.zeros((rows, seq), dtype=torch.long)
    att[:n, :length] = b["attention_mask"]
    if keep_one_token and rows > n:
        att[n:, 0] = 1
    mpos = torch.zeros((rows, kmax), dtype=torch.long)
    mpos[:n] = b["marker_pos"]
    mmask = torch.zeros((rows, kmax), dtype=torch.bool)
    mmask[:n] = b["marker_mask"]
    if rows > n:
        mmask[n:, 0] = True
    qt = torch.zeros((rows,), dtype=torch.long)
    qt[:n] = b["qtype"]
    return ids, att, mpos, mmask, qt


def plan_chunks(lengths: Sequence[int], buckets: Buckets, max_rows: int, max_batch_tokens: int) -> List[Tuple[List[int], int, int]]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    cap = min(max_rows, buckets.max_rows)
    plan: List[Tuple[List[int], int, int]] = []
    cur: List[int] = []
    cur_len = 0
    for i in order:
        new_len = max(cur_len, lengths[i])
        n = len(cur) + 1
        fits = n <= cap and buckets.rows_for(n) * buckets.seq_for(new_len) <= max_batch_tokens
        if cur and not fits:
            plan.append((cur, buckets.rows_for(len(cur)), buckets.seq_for(cur_len)))
            cur, cur_len = [], 0
            new_len = lengths[i]
        cur.append(i)
        cur_len = new_len
    if cur:
        plan.append((cur, buckets.rows_for(len(cur)), buckets.seq_for(cur_len)))
    return plan


class CpuBackend:
    name = "cpu"

    def __init__(self, model_dir: str, cfg: Dict[str, Any]):
        self.model_dir = model_dir
        self.cfg = cfg
        self.max_len = int(cfg.get("max_len", 512))
        self.last_device_ms = 0.0
        self.threads = int(os.environ["LAYA_CPU_THREADS"]) if os.environ.get("LAYA_CPU_THREADS") else None
        if self.threads:
            torch.set_num_threads(self.threads)
        self.impl_name = "vendored DecisionModel"
        self.impl = None if os.environ.get("LAYA_CPU_IMPL", "reference").lower() == "vendored" else self._try_reference(model_dir, self.threads)
        if self.impl is None:
            self.model = self._build_vendored(model_dir, cfg)
        else:
            self.model = None
        if self.threads:
            torch.set_num_threads(self.threads)
        self.threads = torch.get_num_threads()
        self.seq_buckets = env_int_list("LAYA_SEQ_BUCKETS", list(range(64, self.max_len + 1, 64)) or [self.max_len])
        self.row_buckets = env_int_list("LAYA_ROW_BUCKETS", DEFAULT_ROW_BUCKETS)

    @staticmethod
    def _try_reference(model_dir: str, threads: Optional[int] = None):
        try:
            from ..reference import laya_reference as ref
        except ImportError:
            return None
        cls = getattr(ref, "LayaReference", None)
        if cls is None:
            return None
        try:
            attn = os.environ.get("LAYA_CPU_ATTN") or "eager"
            impl = cls(model_dir, attn_implementation=attn, threads=threads)
        except Exception as e:
            log.warning("reference.laya_reference.LayaReference could not be built (%s); using the vendored DecisionModel", e)
            return None
        if not callable(getattr(impl, "forward", None)):
            return None
        return impl

    @staticmethod
    def _build_vendored(model_dir: str, cfg: Dict[str, Any]):
        from safetensors.torch import load_file

        from ..vendor.rl_common import build_model

        model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
        state = load_file(os.path.join(model_dir, "model.safetensors"))
        model.load_state_dict(state, strict=True)
        model.float().eval()
        model.encoder.config.reference_compile = False
        return model

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        t0 = time.perf_counter()
        if self.impl is not None:
            logits, act = self.impl.forward(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        else:
            logits, act = self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        self.last_device_ms = (time.perf_counter() - t0) * 1000.0
        return torch.as_tensor(logits).float(), torch.as_tensor(act).float()

    def shapes(self) -> Dict[str, Any]:
        return {
            "backend": "cpu",
            "device": "cpu",
            "mesh_shape": "1x1",
            "precision": "fp32",
            "implementation": self.impl_name if self.impl is None else type(self.impl).__name__,
            "threads": self.threads,
            "seq_buckets": list(self.seq_buckets),
            "row_buckets": list(self.row_buckets),
            "max_rows": self.row_buckets[-1],
            "warm_shapes": [],
            "pad_rows_keep_one_token": True,
            "trace": False,
        }

    def close(self) -> None:
        self.model = None
        self.impl = None


def make_backend(kind: str, model_dir: str, cfg: Dict[str, Any]):
    kind = (kind or "tt").lower()
    if kind == "cpu":
        return CpuBackend(model_dir, cfg)
    if kind == "tt":
        os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")
        os.environ.setdefault("LAYA_MODEL_DIR", model_dir)
        try:
            from ..tt.engine import LayaEngine
        except ModuleNotFoundError as e:
            raise ImportError(
                "LAYA_BACKEND=tt needs models.autoports.convaiinnovations_laya.tt.engine.LayaEngine.from_env(); "
                "set LAYA_BACKEND=cpu to serve the fp32 reference on the host (%s)" % e
            ) from e
        return LayaEngine.from_env()
    raise ValueError("LAYA_BACKEND must be one of %s, got %r" % (BACKENDS, kind))


class Engine:
    def __init__(self, backend, tokenizer, cfg: Dict[str, Any], model_dir: str, backend_kind: str, limits: Optional[Limits] = None):
        self.backend = backend
        self.tok = tokenizer
        self.cfg = cfg
        self.model_dir = model_dir
        self.backend_kind = backend_kind
        self.limits = limits or Limits()
        self.temps = Temperatures(cfg, clamp=env_flag("LAYA_TEMPERATURE_CLAMP", True))
        self.max_len = int(cfg.get("max_len", 512))
        self.head_max_len = int(cfg.get("head_max_len", 192))
        self.shapes = dict(backend.shapes())
        self.buckets = Buckets.from_shapes(self.shapes)
        self.keep_one_token = bool(self.shapes.get("pad_rows_keep_one_token", True))
        self.pad_id = int(tokenizer.pad_token_id)
        self.requests = 0
        self.rows_served = 0
        self.batch_histogram: Dict[str, int] = {}
        self.hf_model = os.environ.get("HF_MODEL") or DEFAULT_HF_MODEL
        self.revision = os.environ.get("LAYA_REVISION") or ""

    @classmethod
    def from_env(cls) -> "Engine":
        kind = (os.environ.get("LAYA_BACKEND") or "tt").lower()
        model_dir = resolve_model_dir()
        cfg = load_cfg(model_dir)
        tok = load_tokenizer(model_dir)
        backend = make_backend(kind, model_dir, cfg)
        return cls(backend, tok, cfg, model_dir, kind)

    @property
    def token_budget(self) -> int:
        return self.limits.token_budget or self.buckets.max_seq

    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_kind,
            "model": self.hf_model,
            "revision": self.revision,
            "model_dir": self.model_dir,
            "max_len": self.max_len,
            "head_max_len": self.head_max_len,
            "mesh_shape": self.shapes.get("mesh_shape"),
            "precision": self.shapes.get("precision"),
            "seq_buckets": self.buckets.seqs,
            "row_buckets": self.buckets.rows,
            "warm_shapes": self.shapes.get("warm_shapes", []),
            "shapes": self.shapes,
            "limits": self.limits.describe(),
            "token_budget": self.token_budget,
            "temperatures": self.temps.describe(),
            "requests": self.requests,
            "rows_served": self.rows_served,
            "batch_histogram": dict(self.batch_histogram),
        }

    def budget(self, body: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        max_len = check_budget_param(body, "max_len", self.token_budget)
        head_max_len = check_budget_param(body, "head_max_len", self.token_budget)
        return max_len, head_max_len

    def _run_chunk(self, b: Dict[str, torch.Tensor], rows: int, seq: int) -> Tuple[np.ndarray, np.ndarray, float]:
        n = b["input_ids"].shape[0]
        kmax = b["marker_pos"].shape[1]
        padded = pad_batch(b, rows, seq, self.pad_id, self.keep_one_token)
        t0 = time.perf_counter()
        logits, act = self.backend.forward(*padded)
        wall = (time.perf_counter() - t0) * 1000.0
        logits = torch.as_tensor(logits)
        act = torch.as_tensor(act)
        if logits.ndim != 2 or logits.shape[0] < n or logits.shape[1] < kmax:
            raise RuntimeError("backend logits shape %s does not cover %d rows x %d markers" % (tuple(logits.shape), n, kmax))
        if act.ndim != 2 or act.shape[0] < n or act.shape[1] != 2:
            raise RuntimeError("backend act_logits shape %s is not [rows, 2]" % (tuple(act.shape),))
        device_ms = getattr(self.backend, "last_device_ms", None)
        if not isinstance(device_ms, (int, float)):
            device_ms = wall
        logits_np = logits[:n, :kmax].float().cpu().numpy()
        act_np = torch.softmax(act[:n].float(), -1).cpu().numpy()
        key = "%dx%d" % (rows, seq)
        self.batch_histogram[key] = self.batch_histogram.get(key, 0) + 1
        self.rows_served += n
        return logits_np, act_np, float(device_ms)

    def predict_batch(
        self,
        states: Sequence[Any],
        questions: Dict[str, Dict[str, Any]],
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        min_confidence: Any = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self.requests += 1
        ids = list(questions.keys())
        meta: Dict[str, Any] = {"device_ms": 0.0, "batches": [], "rows": 0}
        if not ids:
            return [empty_result() for _ in states], meta
        for qid in ids:
            check_question(qid, questions[qid])
        internal = {qid: to_internal(questions[qid]) for qid in ids}
        max_len = int(max_len or self.max_len)
        head_max_len = int(head_max_len or self.head_max_len)
        if max_len > self.buckets.max_seq:
            raise RequestError("max_len exceeds the served context (%d > %d)" % (max_len, self.buckets.max_seq))
        encoded = [encode_state(self.tok, st, ids, internal, max_len, head_max_len) for st in states]
        flat = [item for items in encoded for item in items]
        lengths = [len(item["ids"]) for item in flat]
        plan = plan_chunks(lengths, self.buckets, self.limits.max_rows, self.limits.max_batch_tokens)
        logits_rows: List[Optional[np.ndarray]] = [None] * len(flat)
        act_rows: List[Optional[np.ndarray]] = [None] * len(flat)
        for idx, rows, seq in plan:
            b = collate_items([[flat[i] for i in idx]], self.pad_id)
            logits_np, act_np, device_ms = self._run_chunk(b, rows, seq)
            meta["device_ms"] += device_ms
            meta["batches"].append("%dx%d" % (rows, seq))
            for r, i in enumerate(idx):
                logits_rows[i] = logits_np[r]
                act_rows[i] = act_np[r]
        meta["rows"] = len(flat)
        results = []
        offset = 0
        for items in encoded:
            n = len(items)
            logits_state = logits_rows[offset : offset + n]
            act_state = act_rows[offset : offset + n]
            n_tokens = sum(len(item["ids"]) for item in items)
            answers = decode_answers(ids, internal, items, logits_state, act_state, self.temps)
            results.append({"model": MODEL_NAME, "answers": answers, "usage": usage_for_state(ids, items, n_tokens)})
            offset += n
        apply_confidence_gate(results, min_confidence)
        return results, meta

    def raw_forward(self, body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.requests += 1
        try:
            ids = torch.as_tensor(body["input_ids"], dtype=torch.long)
            att = torch.as_tensor(body["attention_mask"], dtype=torch.long)
            mpos = torch.as_tensor(body["marker_pos"], dtype=torch.long)
            mmask = torch.as_tensor(body["marker_mask"], dtype=torch.bool)
            qt = torch.as_tensor(body["qtype"], dtype=torch.long)
        except (KeyError, TypeError, ValueError) as e:
            raise RequestError("body must carry input_ids, attention_mask, marker_pos, marker_mask and qtype as nested integer lists: %s" % e)
        if ids.ndim != 2 or att.shape != ids.shape or mpos.ndim != 2 or mmask.shape != mpos.shape or qt.shape != (ids.shape[0],):
            raise RequestError("tensor shapes disagree: input_ids %s attention_mask %s marker_pos %s marker_mask %s qtype %s" % (tuple(ids.shape), tuple(att.shape), tuple(mpos.shape), tuple(mmask.shape), tuple(qt.shape)))
        if mpos.shape[0] != ids.shape[0]:
            raise RequestError("marker_pos rows %d differ from input_ids rows %d" % (mpos.shape[0], ids.shape[0]))
        n, length = ids.shape
        if n > min(self.limits.max_rows, self.buckets.max_rows):
            raise LimitError("too many rows (%d > %d)" % (n, min(self.limits.max_rows, self.buckets.max_rows)))
        if int(qt.max()) > 2 or int(qt.min()) < 0:
            raise RequestError("qtype values must be 0 (choice), 1 (score) or 2 (noul)")
        if int(mpos.max()) >= length or int(mpos.min()) < 0:
            raise RequestError("marker positions must lie inside the sequence")
        seq = self.buckets.seq_for(length)
        rows = self.buckets.rows_for(n)
        b = {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "qtype": qt}
        kmax = mpos.shape[1]
        padded = pad_batch(b, rows, seq, self.pad_id, self.keep_one_token)
        t0 = time.perf_counter()
        logits, act = self.backend.forward(*padded)
        wall = (time.perf_counter() - t0) * 1000.0
        logits = torch.as_tensor(logits)[:n, :kmax].float().cpu()
        act = torch.as_tensor(act)[:n].float().cpu()
        device_ms = getattr(self.backend, "last_device_ms", None)
        if not isinstance(device_ms, (int, float)):
            device_ms = wall
        key = "%dx%d" % (rows, seq)
        self.batch_histogram[key] = self.batch_histogram.get(key, 0) + 1
        self.rows_served += n
        out = {"logits": logits.tolist(), "act_logits": act.tolist(), "batch": key, "rows": n, "kmax": kmax}
        return out, {"device_ms": float(device_ms), "batches": [key], "rows": n}

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

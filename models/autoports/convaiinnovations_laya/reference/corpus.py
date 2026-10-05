# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np
import torch

from models.autoports.convaiinnovations_laya import common
from models.autoports.convaiinnovations_laya.reference import laya_reference as lr
from models.autoports.convaiinnovations_laya.vendor import rl_common

DEFAULT_CLONE_DIR = "/home/hous/dev/laya/evals/vendor/laya"
DEFAULT_OUT_DIR = "/home/hous/dev/laya/reference"
SOURCE_TYPED = 0
SOURCE_PARITY_FAST = 1


def clone_dir() -> str:
    return os.environ.get("LAYA_CLONE_DIR", DEFAULT_CLONE_DIR)


def _load_presets_module():
    path = os.path.join(clone_dir(), "laya", "presets.py")
    spec = importlib.util.spec_from_file_location("laya_clone_presets", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parity_fast_definitions() -> tuple:
    presets = _load_presets_module()
    path = os.path.join(clone_dir(), "benchmarks", "parity_fast.py")
    tree = ast.parse(open(path).read())
    ns = {name: getattr(presets, name) for name in ("triage_questions", "moderation_questions", "guard_questions", "router_questions", "email_questions")}
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) in ("TEXTS", "PRESETS") for t in n.targets)]
    if len(nodes) != 2:
        raise RuntimeError(f"expected TEXTS and PRESETS assignments in {path}, found {len(nodes)}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns["TEXTS"], ns["PRESETS"], path


def parity_fast_states() -> list:
    texts, presets, _ = parity_fast_definitions()
    out = []
    for pname, fn in presets.items():
        try:
            qs = fn()
        except TypeError:
            qs = fn
        for i, t in enumerate(texts):
            out.append((f"{pname}/{i}", {"subject": t[:60], "body": t}, dict(list(qs.items())[:8])))
    return out


def parity_fast_items(tok, max_len: int, head_max_len: int) -> list:
    groups = []
    for name, state, questions in parity_fast_states():
        case = {"id": name, "index": None, "workflow": name.split("/")[0], "state": state, "questions": questions}
        items = common.case_items(tok, case, max_len=max_len, head_max_len=head_max_len)
        for it in items:
            it["src"] = "parity_fast"
        groups.append((name, items, None))
    return groups


def typed_groups(tok, cases: list, max_len: int, head_max_len: int) -> list:
    groups = []
    for c in cases:
        items = common.case_items(tok, c, max_len=max_len, head_max_len=head_max_len)
        groups.append((c["id"], items, c))
    return groups


def gold_for(case: dict, qid: str, q: dict):
    if case is None or "gold" not in case:
        return None
    g = case["gold"][qid]
    if q["t"] == "choice":
        keys = list(q["crit"].keys())
        return {"label": str(g["label"]), "idx": keys.index(str(g["label"])), "soft": [float(g.get("probabilities", {}).get(k, 0.0)) for k in keys]}
    if q["t"] == "noul":
        pt = float(g.get("probabilities", {}).get("true", g.get("noul", 0.5)))
        return {"label": str(g["label"]), "idx": 1 if str(g["label"]).lower() == "true" else 0, "soft": [1 - pt, pt]}
    n = len(q["crit"])
    return {
        "label": str(g["label"]),
        "idx": int(g["label"]),
        "soft": [float(g.get("probabilities", {}).get(str(i), 0.0)) for i in range(n)],
        "gold_score": float(g.get("score", float(g["label"]))),
    }


def run_groups(ref: lr.LayaReference, groups: list, source: int, log=print) -> list:
    records = []
    t_start = time.perf_counter()
    for gi, (name, items, case) in enumerate(groups):
        b = ref.collate(items)
        t0 = time.perf_counter()
        logits, act = ref.forward(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        dt = time.perf_counter() - t0
        logits = logits.numpy()
        act = act.numpy()
        ap = lr.act_probs(act)
        stored_ms = getattr(ref, "last_forward_ms", None)
        for r, it in enumerate(items):
            k = len(it["markers"])
            qt = int(it["qtype"])
            t = lr.temperature_for(ref.temperature, ref.temperature_by_options, qt, k, clamp=True)
            t_raw = lr.temperature_for(ref.temperature, ref.temperature_by_options, qt, k, clamp=False)
            records.append(
                {
                    "source": source,
                    "group": name,
                    "group_index": gi,
                    "row_in_group": r,
                    "group_rows": len(items),
                    "group_seq_len": int(b["input_ids"].shape[1]),
                    "case_index": it.get("case_index"),
                    "workflow": it.get("workflow"),
                    "qid": it["qid"],
                    "type": rl_common.QTYPE_NAMES[qt],
                    "qtype": qt,
                    "k": k,
                    "ids": list(map(int, it["ids"])),
                    "markers": list(map(int, it["markers"])),
                    "seq_len": len(it["ids"]),
                    "option_keys": list(it["q"]["crit"].keys()) if it["q"]["t"] == "choice" else [str(i) for i in range(k)],
                    "temperature": t,
                    "temperature_raw": t_raw,
                    "temperature_bucket": lr.temp_bucket(qt, k),
                    "logits": [float(x) for x in logits[r, :k]],
                    "act_logits": [float(x) for x in act[r]],
                    "act_probs": [float(x) for x in ap[r]],
                    "probs": [float(x) for x in lr.scaled_probs(logits[r, :k], t)],
                    "probs_hub": [float(x) for x in lr.scaled_probs(logits[r, :k], t_raw)],
                    "probs_raw": [float(x) for x in lr.scaled_probs(logits[r, :k], 1.0)],
                    "gold": gold_for(case, it["qid"], it["q"]),
                    "answer_pip": lr.decode_answer(it["q"], logits[r, :k], ap[r], ref.temperature, ref.temperature_by_options, shape="pip", clamp=True),
                    "answer_hub": lr.decode_answer(it["q"], logits[r, :k], ap[r], ref.temperature, ref.temperature_by_options, shape="hub", clamp=False),
                    "forward_ms_group": stored_ms if stored_ms is not None else round(1000 * dt, 1),
                }
            )
        if (gi + 1) % 10 == 0 or gi + 1 == len(groups):
            log(f"  source {source}: {gi + 1}/{len(groups)} groups, {time.perf_counter() - t_start:.0f} s elapsed")
    return records


def pack_corpus(records: list, max_len: int) -> dict:
    n = len(records)
    kmax = max(r["k"] for r in records)
    ids = np.full((n, max_len), -1, dtype=np.int32)
    att = np.zeros((n, max_len), dtype=np.int8)
    mpos = np.zeros((n, kmax), dtype=np.int32)
    mmask = np.zeros((n, kmax), dtype=bool)
    logits = np.full((n, kmax), lr.NEG_LOGIT, dtype=np.float32)
    probs = np.zeros((n, kmax), dtype=np.float32)
    probs_hub = np.zeros((n, kmax), dtype=np.float32)
    probs_raw = np.zeros((n, kmax), dtype=np.float32)
    for i, r in enumerate(records):
        L, k = r["seq_len"], r["k"]
        ids[i, :L] = r["ids"]
        att[i, :L] = 1
        mpos[i, :k] = r["markers"]
        mmask[i, :k] = True
        logits[i, :k] = r["logits"]
        probs[i, :k] = r["probs"]
        probs_hub[i, :k] = r["probs_hub"]
        probs_raw[i, :k] = r["probs_raw"]
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": np.array([r["qtype"] for r in records], dtype=np.int8),
        "k": np.array([r["k"] for r in records], dtype=np.int16),
        "seq_len": np.array([r["seq_len"] for r in records], dtype=np.int16),
        "source": np.array([r["source"] for r in records], dtype=np.int8),
        "group_index": np.array([r["group_index"] for r in records], dtype=np.int32),
        "temperature": np.array([r["temperature"] for r in records], dtype=np.float32),
        "temperature_raw": np.array([r["temperature_raw"] for r in records], dtype=np.float32),
        "logits_fp32": logits,
        "act_logits_fp32": np.array([r["act_logits"] for r in records], dtype=np.float32),
        "act_probs": np.array([r["act_probs"] for r in records], dtype=np.float32),
        "probs": probs,
        "probs_hub": probs_hub,
        "probs_raw": probs_raw,
        "gold_idx": np.array([(-1 if r["gold"] is None else r["gold"]["idx"]) for r in records], dtype=np.int16),
    }


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_outputs(ref: lr.LayaReference, records: list, gate_cases: list, out_dir: str, max_len: int, head_max_len: int, seed: int, per_workflow: int, parity_path: str, timings: dict):
    import transformers

    os.makedirs(out_dir, exist_ok=True)
    arrays = pack_corpus(records, max_len)
    npz_path = os.path.join(out_dir, "parity_corpus.npz")
    np.savez_compressed(npz_path, **arrays)
    index = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hf_model": common.HF_MODEL,
        "revision": common.LAYA_REVISION,
        "model_dir": ref.model_dir,
        "model_safetensors_sha256": sha256_file(os.path.join(ref.model_dir, "model.safetensors")),
        "max_len": max_len,
        "head_max_len": head_max_len,
        "attn_implementation": ref.attn_implementation,
        "compute_dtype": str(ref.dtype),
        "checkpoint_dtypes": ref.load_report["source_dtypes"],
        "threads": ref.threads,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "typed_decisions_parquet": common.typed_decisions_path(),
        "typed_decisions_parquet_sha256": sha256_file(common.typed_decisions_path()),
        "gate_subset": {
            "rule": "numpy RandomState(seed).choice(per_workflow case indices per workflow without replacement), workflows in parquet order, result sorted by case index",
            "seed": seed,
            "per_workflow": per_workflow,
            "case_indices": [c["index"] for c in gate_cases],
            "case_ids": [c["id"] for c in gate_cases],
        },
        "parity_fast": {"source_file": parity_path, "clone_dir": clone_dir(), "rule": "TEXTS and PRESETS executed from the file's AST; states() replicated: 5 presets x 12 texts, up to 8 questions each, state {subject: text[:60], body: text}"},
        "temperature": ref.temperature,
        "temperature_by_options": ref.temperature_by_options,
        "temperature_rule": {
            "probs": "pip laya 0.3.27 rule: temperature clamped to [0.5, 5.0] (plan amendment A6); array temperature",
            "probs_hub": "Hub rl_agent_api.py rule: raw temperature from rl_agent_config.json; array temperature_raw",
            "probs_raw": "temperature 1.0 (parity_fast.py p_fp32)",
            "answer_pip": "pip shape, clamped rule",
            "answer_hub": "Hub shape, raw rule",
            "buckets_where_the_rules_differ": [k for k, v in ref.temperature_by_options.items() if lr.clamp_temperature(v) != float(v)],
        },
        "n_items": len(records),
        "n_typed": int(sum(1 for r in records if r["source"] == SOURCE_TYPED)),
        "n_parity_fast": int(sum(1 for r in records if r["source"] == SOURCE_PARITY_FAST)),
        "kmax": int(arrays["marker_pos"].shape[1]),
        "timings": timings,
        "arrays": {k: [list(v.shape), str(v.dtype)] for k, v in arrays.items()},
        "items": [{k: v for k, v in r.items() if k not in ("ids",)} for r in records],
    }
    with open(os.path.join(out_dir, "parity_corpus_index.json"), "w") as f:
        json.dump(index, f, indent=1)
    td_dir = os.path.join(out_dir, "typed_decisions_cpu")
    os.makedirs(td_dir, exist_ok=True)
    by_case = {}
    for r in records:
        if r["source"] == SOURCE_TYPED:
            by_case.setdefault(r["group"], []).append(r)
    agree = 0
    n_dec = 0
    with open(os.path.join(td_dir, "answers.jsonl"), "w") as f:
        for c in gate_cases:
            rows = by_case[c["id"]]
            answers = {r["qid"]: r["answer_pip"] for r in rows}
            answers_hub = {r["qid"]: r["answer_hub"] for r in rows}
            logits = {r["qid"]: r["logits"] for r in rows}
            probs = {r["qid"]: r["probs"] for r in rows}
            probs_hub = {r["qid"]: r["probs_hub"] for r in rows}
            gold = {r["qid"]: r["gold"] for r in rows}
            for r in rows:
                n_dec += 1
                agree += int(int(np.argmax(r["probs"])) == r["gold"]["idx"])
            f.write(json.dumps({"id": c["id"], "index": c["index"], "workflow": c["workflow"], "answers": answers, "answers_hub": answers_hub, "logits": logits, "probs": probs, "probs_hub": probs_hub, "gold": gold, "state": c["state"], "questions": c["questions"]}, ensure_ascii=False) + "\n")
    summary = {
        "n_cases": len(gate_cases),
        "n_decisions": n_dec,
        "argmax_equals_gold_label": agree,
        "argmax_equals_gold_label_fraction": round(agree / max(1, n_dec), 4),
        "note": "sanity count only; the E2 metrics come from the authors' bench_local.py metric block in the evals harness",
        "protocol": "one forward per case with its 5 questions, rows padded to the longest row of the case, max_len 512, head_max_len 192, fp32, eager attention",
        "temperature_rule": {"answers, probs": "pip laya 0.3.27 rule, temperatures clamped to [0.5, 5.0]", "answers_hub, probs_hub": "Hub rl_agent_api.py rule, raw temperatures", "note": "the two rules differ only for the choice:11+ bucket (0.1006 raw, 0.5 clamped), which no typed-decisions question uses"},
        "index_json": os.path.join(out_dir, "parity_corpus_index.json"),
    }
    with open(os.path.join(td_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return npz_path, summary


class StoredForward:
    def __init__(self, out_dir: str):
        self.index = json.load(open(os.path.join(out_dir, "parity_corpus_index.json")))
        self.items = self.index["items"]
        self.pos = 0
        self.model_dir = self.index["model_dir"]
        self.attn_implementation = self.index["attn_implementation"]
        self.dtype = torch.float32
        self.load_report = {"source_dtypes": self.index["checkpoint_dtypes"]}
        self.threads = self.index["threads"]
        self.load_seconds = self.index["timings"]["model_load_s"]
        self.tok = common.load_tokenizer(self.model_dir)
        cfg = common.load_rl_config(self.model_dir)
        self.temperature = [float(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
        self.temperature_by_options = {k: float(v) for k, v in cfg.get("temperature_by_options", {}).items()}
        self.last_forward_ms = None

    def collate(self, items: list, seq_len: int | None = None) -> dict:
        return common.collate(items, self.tok.pad_token_id, seq_len=seq_len)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        n, kmax = marker_mask.shape
        rows = self.items[self.pos : self.pos + n]
        self.pos += n
        logits = torch.full((n, kmax), lr.NEG_LOGIT, dtype=torch.float32)
        act = torch.zeros((n, 2), dtype=torch.float32)
        for r, it in enumerate(rows):
            if it["seq_len"] != int(attention_mask[r].sum()) or it["markers"] != marker_pos[r, : it["k"]].tolist() or it["qtype"] != int(qtype[r]):
                raise RuntimeError(f"stored item {self.pos - n + r} ({it['group']}/{it['qid']}) does not match the rebuilt sequence")
            logits[r, : it["k"]] = torch.tensor(it["logits"], dtype=torch.float32)
            act[r] = torch.tensor(it["act_logits"], dtype=torch.float32)
        self.last_forward_ms = rows[0]["forward_ms_group"]
        return logits, act


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--max-len", type=int, default=common.DEFAULT_MAX_LEN)
    ap.add_argument("--head-max-len", type=int, default=common.DEFAULT_HEAD_MAX_LEN)
    ap.add_argument("--seed", type=int, default=common.DEFAULT_GATE_SEED)
    ap.add_argument("--per-workflow", type=int, default=common.DEFAULT_GATE_PER_WORKFLOW)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--skip-parity-fast", action="store_true")
    ap.add_argument("--repack-from-index", action="store_true")
    a = ap.parse_args(argv)

    def log(msg):
        print(msg, flush=True)

    t_all = time.perf_counter()
    if a.repack_from_index:
        ref = StoredForward(a.out_dir)
        a.seed = ref.index["gate_subset"]["seed"]
        a.per_workflow = ref.index["gate_subset"]["per_workflow"]
        a.max_len, a.head_max_len = ref.index["max_len"], ref.index["head_max_len"]
        log(f"repacking {ref.index['n_items']} stored items from {a.out_dir} without a model forward")
    else:
        ref = lr.LayaReference(model_dir=a.model_dir, attn_implementation=a.attn)
    log(f"model loaded in {ref.load_seconds:.1f} s from {ref.model_dir}; threads {ref.threads}; load report {ref.load_report}")
    cases = common.load_typed_decisions()
    gate = common.gate_subset(cases, seed=a.seed, per_workflow=a.per_workflow)
    log(f"gate subset: {len(gate)} cases, indices {[c['index'] for c in gate]}")
    tg = typed_groups(ref.tok, gate, a.max_len, a.head_max_len)
    t0 = time.perf_counter()
    records = run_groups(ref, tg, SOURCE_TYPED, log=log)
    t_typed = time.perf_counter() - t0
    parity_path = None
    t_parity = 0.0
    if not a.skip_parity_fast:
        _, _, parity_path = parity_fast_definitions()
        pg = parity_fast_items(ref.tok, a.max_len, a.head_max_len)
        log(f"parity_fast: {len(pg)} states, {sum(len(g[1]) for g in pg)} questions")
        t0 = time.perf_counter()
        records += run_groups(ref, pg, SOURCE_PARITY_FAST, log=log)
        t_parity = time.perf_counter() - t0
    timings = {"typed_decisions_s": round(t_typed, 1), "parity_fast_s": round(t_parity, 1), "model_load_s": round(ref.load_seconds, 1)}
    if a.repack_from_index:
        timings = dict(ref.index["timings"], repacked_from_index_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), original_created_utc=ref.index["created_utc"])
        if ref.pos != len(ref.items):
            raise RuntimeError(f"repack consumed {ref.pos} of {len(ref.items)} stored items")
    npz_path, summary = write_outputs(ref, records, gate, a.out_dir, a.max_len, a.head_max_len, a.seed, a.per_workflow, parity_path, timings)
    log(f"wrote {npz_path} with {len(records)} items; typed_decisions_cpu summary {summary}")
    log(f"total {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()

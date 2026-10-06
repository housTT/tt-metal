# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import platform
import socket
import time
from datetime import datetime, timezone

import numpy as np
import torch

os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
AUTOPORT = os.path.dirname(HERE)
DOC_DIR = os.path.join(AUTOPORT, "doc", "full_model")
SUITES = "/home/hous/dev/laya/evals/apps/suites.json"
REFERENCE_APPS = "/home/hous/dev/laya/evals/results/cpu_reference_cpu_b0_20261005T210448Z/apps"
OUT_NPZ = "/home/hous/dev/laya/reference/parity_corpus_single.npz"
OUT_INDEX = "/home/hous/dev/laya/reference/parity_corpus_single_index.json"
SUITE_SOURCE = {"jev.emotion": 0, "jev.ag_news": 1}
MAX_LEN = 512
HEAD_MAX_LEN = 192
FLAGGED_EMOTION_CASES = (398, 381, 335, 209, 244, 286, 140)
FILLER_EMOTION_CASES = (0, 1, 2, 3)


def loadavg():
    with open("/proc/loadavg") as f:
        return [float(x) for x in f.read().split()[:3]]


def suite_cases(name):
    suites = json.load(open(SUITES))["suites"][name]
    return suites["cases"], suites["gold"], suites.get("meta", {})


def served_items(tok, state, questions):
    """The server's request-to-rows path: server/engine.py encode_state on the vendored builder."""
    from models.autoports.convaiinnovations_laya.server.engine import encode_state, to_internal

    ids = list(questions.keys())
    internal = {qid: to_internal(questions[qid]) for qid in ids}
    items = encode_state(tok, state, ids, internal, MAX_LEN, HEAD_MAX_LEN)
    for qid, it in zip(ids, items):
        it["qid"] = qid
        it["q"] = internal[qid]
    return items


def single_call_tensors(items, seq_len=None):
    from models.autoports.convaiinnovations_laya.vendor.rl_common import collate_items

    b = collate_items([items], 50283)
    out = {k: b[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}
    if seq_len is not None and out["input_ids"].shape[1] < seq_len:
        n, L = out["input_ids"].shape
        ids = torch.full((n, seq_len), 50283, dtype=torch.long)
        att = torch.zeros((n, seq_len), dtype=torch.long)
        ids[:, :L] = out["input_ids"]
        att[:, :L] = out["attention_mask"]
        out["input_ids"], out["attention_mask"] = ids, att
    return out


def cmd_corpus(a):
    """CPU fp32 single-row parity corpus of the two application suites (eager reference, SDPA control), parity_corpus format."""
    import transformers

    from models.autoports.convaiinnovations_laya import common
    from models.autoports.convaiinnovations_laya.reference import corpus as cp
    from models.autoports.convaiinnovations_laya.reference import laya_reference as lr
    from models.autoports.convaiinnovations_laya.vendor import rl_common

    torch.set_num_threads(a.threads)
    ref = lr.LayaReference(attn_implementation="eager", threads=a.threads)
    sdpa = lr.LayaReference(attn_implementation="sdpa", threads=a.threads) if not a.no_sdpa else None
    tok = ref.tok
    records = []
    calls = []
    stored_delta = {}
    t_start = time.perf_counter()
    for suite_name, source in SUITE_SOURCE.items():
        cases, gold, _meta = suite_cases(suite_name)
        ref_path = os.path.join(REFERENCE_APPS, f"{suite_name}.decisions.jsonl")
        stored = {r["case"]: r for r in (json.loads(l) for l in open(ref_path))} if os.path.exists(ref_path) else {}
        n_cases = len(cases) if a.limit <= 0 else min(len(cases), a.limit)
        deltas = []
        for ci in range(n_cases):
            state, questions = cases[ci]
            items = served_items(tok, state, questions)
            b = single_call_tensors(items)
            t0 = time.perf_counter()
            logits, act = ref.forward(
                b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"]
            )
            dt = time.perf_counter() - t0
            logits_s = None
            if sdpa is not None:
                logits_s, _ = sdpa.forward(
                    b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"]
                )
                logits_s = logits_s.numpy()
            logits = logits.numpy()
            ap = lr.act_probs(act.numpy())
            for r, it in enumerate(items):
                k = len(it["markers"])
                qt = int(it["qtype"])
                t = lr.temperature_for(ref.temperature, ref.temperature_by_options, qt, k, clamp=True)
                t_raw = lr.temperature_for(ref.temperature, ref.temperature_by_options, qt, k, clamp=False)
                gold_idx = int(gold[ci]) if gold is not None and ci < len(gold) else -1
                keys = list(it["q"]["crit"].keys())
                rec = {
                    "source": source,
                    "group": f"{suite_name}/{ci}",
                    "suite": suite_name,
                    "group_index": ci,
                    "row_in_group": r,
                    "group_rows": len(items),
                    "group_seq_len": int(b["input_ids"].shape[1]),
                    "case_index": ci,
                    "workflow": suite_name,
                    "qid": it["qid"],
                    "type": rl_common.QTYPE_NAMES[qt],
                    "qtype": qt,
                    "k": k,
                    "ids": list(map(int, it["ids"])),
                    "markers": list(map(int, it["markers"])),
                    "seq_len": len(it["ids"]),
                    "option_keys": keys,
                    "temperature": t,
                    "temperature_raw": t_raw,
                    "temperature_bucket": lr.temp_bucket(qt, k),
                    "logits": [float(x) for x in logits[r, :k]],
                    "logits_sdpa": [float(x) for x in logits_s[r, :k]] if logits_s is not None else None,
                    "act_logits": [float(x) for x in act.numpy()[r]],
                    "act_probs": [float(x) for x in ap[r]],
                    "probs": [float(x) for x in lr.scaled_probs(logits[r, :k], t)],
                    "probs_hub": [float(x) for x in lr.scaled_probs(logits[r, :k], t_raw)],
                    "probs_raw": [float(x) for x in lr.scaled_probs(logits[r, :k], 1.0)],
                    "gold": {"label": keys[gold_idx], "idx": gold_idx} if 0 <= gold_idx < len(keys) else None,
                    "text": state.get("text", state.get("article", "")) if isinstance(state, dict) else str(state),
                    "forward_ms_group": round(1000 * dt, 1),
                }
                if ci in stored:
                    d = float(np.abs(np.asarray(stored[ci]["logits"], dtype=np.float64) - logits[r, :k]).max())
                    rec["stored_reference_logit_delta"] = d
                    deltas.append(d)
                records.append(rec)
            calls.append(
                {
                    "call_index": ci,
                    "call_id": f"{suite_name}/{ci}",
                    "group": suite_name,
                    "state": state,
                    "questions": questions,
                    "qids": [it["qid"] for it in items],
                    "rows": [len(records) - len(items) + i for i in range(len(items))],
                }
            )
            if (ci + 1) % 50 == 0 or ci + 1 == n_cases:
                print(
                    f"corpus {suite_name}: {ci + 1}/{n_cases} cases, {time.perf_counter() - t_start:.0f} s, load {loadavg()[0]}",
                    flush=True,
                )
        stored_delta[suite_name] = {
            "n": len(deltas),
            "max_abs_logit_delta_vs_stored_reference": float(max(deltas)) if deltas else None,
        }
    arrays = cp.pack_corpus(records, MAX_LEN)
    kmax = arrays["marker_pos"].shape[1]
    if sdpa is not None:
        ls = np.full((len(records), kmax), lr.NEG_LOGIT, dtype=np.float32)
        for i, r in enumerate(records):
            ls[i, : r["k"]] = r["logits_sdpa"]
        arrays["logits_fp32_sdpa"] = ls
    arrays["group"] = np.array([r["suite"] for r in records])
    arrays["call_id"] = np.array([r["group"] for r in records])
    arrays["qid"] = np.array([r["qid"] for r in records])
    arrays["calls"] = np.array([json.dumps(calls)])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **arrays)
    pins = common.checkpoint_pins(ref.model_dir)
    eager_vs_sdpa = None
    if sdpa is not None:
        d = np.abs(arrays["logits_fp32"] - arrays["logits_fp32_sdpa"])[arrays["marker_mask"]]
        eager_vs_sdpa = {
            "max_abs_logit_delta": float(d.max()),
            "p99": float(np.percentile(d, 99)),
            "median": float(np.median(d)),
        }
    index = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "single-row parity corpus of the application suites (review R3)",
        "hf_model": pins["hf_model"],
        "revision": pins["revision"],
        "model_dir": ref.model_dir,
        "model_safetensors_sha256": cp.sha256_file(os.path.join(ref.model_dir, "model.safetensors")),
        "max_len": MAX_LEN,
        "head_max_len": HEAD_MAX_LEN,
        "attn_implementation": ref.attn_implementation,
        "sdpa_control": "logits_fp32_sdpa holds the same forwards with attn_implementation sdpa"
        if sdpa is not None
        else None,
        "eager_vs_sdpa": eager_vs_sdpa,
        "compute_dtype": str(ref.dtype),
        "checkpoint_dtypes": ref.load_report["source_dtypes"],
        "threads": ref.threads,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "suites_file": SUITES,
        "suites_file_sha256": cp.sha256_file(SUITES),
        "sources": {name: source for name, source in SUITE_SOURCE.items()},
        "rows_path": "every call is one row: the served E3 request shape (one state, one question)",
        "encoding": "server/engine.py encode_state on vendor/rl_common.build_sequence at 512 / 192, the path the server uses",
        "stored_reference_check": stored_delta,
        "temperature": ref.temperature,
        "temperature_by_options": ref.temperature_by_options,
        "temperature_rule": {
            "probs": "pip laya 0.3.27 rule, clamped to [0.5, 5.0]; array temperature",
            "probs_hub": "raw Hub rule; array temperature_raw",
            "probs_raw": "temperature 1.0",
        },
        "n_items": len(records),
        "kmax": kmax,
        "timings": {"total_s": round(time.perf_counter() - t_start, 1), "model_load_s": round(ref.load_seconds, 1)},
        "arrays": {k: [list(v.shape), str(v.dtype)] for k, v in arrays.items()},
        "items": [{k: v for k, v in r.items() if k != "ids"} for r in records],
    }
    with open(a.index, "w") as f:
        json.dump(index, f, indent=1)
    print(
        "CORPUS_DONE",
        json.dumps(
            {
                "npz": a.out,
                "index": a.index,
                "n": len(records),
                "stored_reference_check": stored_delta,
                "eager_vs_sdpa": eager_vs_sdpa,
            }
        ),
    )


def cmd_device(a):
    """Flagged emotion decisions on the device: alone at 1x128 (as served), alone at 1x256, inside 5x256 and 5x128 calls."""
    from models.autoports.convaiinnovations_laya.tests import laya_inputs as LI
    from models.autoports.convaiinnovations_laya.tests.run_fidelity import scaled_probs
    from models.autoports.convaiinnovations_laya.tt import model_config as mc
    from models.autoports.convaiinnovations_laya.tt.engine import LayaEngine, host_tail

    torch.set_num_threads(a.threads)
    policy = mc.policy_from_name(a.policy)
    z = np.load(a.corpus, allow_pickle=False)
    idx = json.load(open(a.index))
    items = idx["items"]
    cases, _gold, _meta = suite_cases("jev.emotion")
    tok = LI.load_tokenizer()
    rows_by_case = {int(it["case_index"]): i for i, it in enumerate(items) if it["suite"] == "jev.emotion"}
    flagged = [int(x) for x in a.cases.split(",")]
    fillers = [int(x) for x in a.fillers.split(",")]
    engine = LayaEngine(
        policy=policy,
        row_buckets=(1, 5),
        seq_buckets=(128, 256),
        warmup_shapes=[(1, 128), (1, 256), (5, 128), (5, 256)],
        threads=a.threads,
    )
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "policy": policy.describe(),
        "corpus": a.corpus,
        "cases": flagged,
        "fillers": fillers,
        "loadavg_start": loadavg(),
        "placements": {},
        "per_case": {},
    }

    def run(call_rows, bucket):
        t = {
            k: torch.as_tensor(np.stack([z[k][r] for r in call_rows]))
            for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
        }
        length = int(t["attention_mask"].sum(-1).max())
        ids = t["input_ids"].long().clamp(min=0)[:, :length]
        att = t["attention_mask"].long()[:, :length]
        out = engine.runner.run(ids, att, t["qtype"].long(), bucket=bucket)
        logits, act = host_tail(
            out["logits"], out["cls"], t["marker_pos"].long(), t["marker_mask"].bool(), engine.act_head
        )
        return logits.numpy(), act.numpy(), out["bucket"]

    def metrics(row, dev_logits):
        k = int(z["k"][row])
        t = float(z["temperature"][row])
        ref = np.asarray(z["logits_fp32"][row, :k], dtype=np.float32)
        sd = np.asarray(z["logits_fp32_sdpa"][row, :k], dtype=np.float32) if "logits_fp32_sdpa" in z.files else None
        dev = np.asarray(dev_logits[:k], dtype=np.float32)
        p_ref, p_dev = scaled_probs(ref, t), scaled_probs(dev, t)
        s = np.sort(p_ref)[::-1]
        return {
            "k": k,
            "temperature": t,
            "option_keys": items[row]["option_keys"],
            "ref_logits": ref.tolist(),
            "dev_logits": dev.tolist(),
            "logit_delta_per_option": (dev - ref).tolist(),
            "max_abs_logit_delta": float(np.abs(dev - ref).max()),
            "sdpa_ref_logits": sd.tolist() if sd is not None else None,
            "eager_vs_sdpa_max_abs_logit_delta": float(np.abs(ref - sd).max()) if sd is not None else None,
            "ref_probs": p_ref.tolist(),
            "dev_probs": p_dev.tolist(),
            "max_abs_dp": float(np.abs(p_dev - p_ref).max()),
            "argmax_ref": int(p_ref.argmax()),
            "argmax_dev": int(p_dev.argmax()),
            "argmax_agree": bool(p_ref.argmax() == p_dev.argmax()),
            "ref_margin": float(s[0] - s[1]),
            "confident": bool(s[0] - s[1] >= 0.10),
        }

    for ci in flagged:
        row = rows_by_case[ci]
        per = {"case": ci, "text": items[row]["text"], "seq_len": int(z["seq_len"][row]), "placements": {}}
        for name, call_rows, bucket in (
            ("alone_1x128_as_served", [row], (1, 128)),
            ("alone_1x256", [row], (1, 256)),
            ("in_5x256", [row] + [rows_by_case[f] for f in fillers], (5, 256)),
            ("in_5x128", [row] + [rows_by_case[f] for f in fillers], (5, 128)),
        ):
            lg, _act, bk = run(call_rows, bucket)
            m = metrics(row, lg[0])
            m["bucket"] = list(bk)
            per["placements"][name] = m
        a128 = per["placements"]["alone_1x128_as_served"]
        per["dev_vs_dev"] = {
            name: {
                "max_abs_logit_delta_vs_alone_1x128": float(
                    np.abs(np.asarray(p["dev_logits"]) - np.asarray(a128["dev_logits"])).max()
                ),
                "max_abs_dp_vs_alone_1x128": float(
                    np.abs(np.asarray(p["dev_probs"]) - np.asarray(a128["dev_probs"])).max()
                ),
            }
            for name, p in per["placements"].items()
            if name != "alone_1x128_as_served"
        }
        report["per_case"][str(ci)] = per
        print(
            "CASE",
            ci,
            json.dumps(
                {
                    n: {
                        "max_abs_dp": round(p["max_abs_dp"], 4),
                        "max_abs_logit_delta": round(p["max_abs_logit_delta"], 3),
                        "argmax": (p["argmax_ref"], p["argmax_dev"]),
                    }
                    for n, p in per["placements"].items()
                }
            ),
        )
    for name in ("alone_1x128_as_served", "alone_1x256", "in_5x256", "in_5x128"):
        dps = [report["per_case"][str(c)]["placements"][name]["max_abs_dp"] for c in flagged]
        dl = [report["per_case"][str(c)]["placements"][name]["max_abs_logit_delta"] for c in flagged]
        flips = [c for c in flagged if not report["per_case"][str(c)]["placements"][name]["argmax_agree"]]
        conf_flips = [c for c in flips if report["per_case"][str(c)]["placements"][name]["confident"]]
        report["placements"][name] = {
            "max_abs_dp": max(dps),
            "median_max_abs_dp": float(np.median(dps)),
            "max_abs_logit_delta": max(dl),
            "argmax_flips": flips,
            "confident_flips": conf_flips,
        }
    report["loadavg_end"] = loadavg()
    engine.close()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print("DEVICE_DONE", json.dumps(report["placements"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--out", default=OUT_NPZ)
    c.add_argument("--index", default=OUT_INDEX)
    c.add_argument("--threads", type=int, default=6)
    c.add_argument("--limit", type=int, default=0, help="cases per suite (0 = all)")
    c.add_argument("--no-sdpa", action="store_true")
    d = sub.add_parser("device")
    d.add_argument("--policy", default=None)
    d.add_argument("--corpus", default=OUT_NPZ)
    d.add_argument("--index", default=OUT_INDEX)
    d.add_argument("--cases", default=",".join(str(c) for c in FLAGGED_EMOTION_CASES))
    d.add_argument("--fillers", default=",".join(str(c) for c in FILLER_EMOTION_CASES))
    d.add_argument("--threads", type=int, default=6)
    d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"corpus": cmd_corpus, "device": cmd_device}[a.cmd](a)


if __name__ == "__main__":
    main()

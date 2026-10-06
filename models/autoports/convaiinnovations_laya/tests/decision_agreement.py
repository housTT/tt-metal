# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone

import numpy as np
import torch

os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")

from models.autoports.convaiinnovations_laya.tests.run_fidelity import (
    CORPUS,
    DOC_DIR,
    INDEX,
    SEQ,
    call_tensors,
    index_for,
    load_corpus,
    loadavg,
    pcc,
    row_buckets_by_seq_arg,
    scaled_probs,
)

N_QUESTIONS = 16
GATE_MAX_ABS_DP = 0.01
PLACEMENTS = ("alone", "b2", "b4", "mixed_b8", "b64")


def pick_questions(data, n=N_QUESTIONS):
    gate_rows = [r for r in range(len(data["items"])) if int(data["source"][r]) == 0]
    step = len(gate_rows) / n
    return [gate_rows[int(round(i * step))] for i in range(n)]


def run_rows(engine, data, rows, bucket=None):
    t = call_tensors(data, rows)
    res = engine.forward_detailed(t["input_ids"], t["attention_mask"], t["marker_pos"], t["marker_mask"], t["qtype"])
    return res["logits"].numpy(), res["act_logits"].numpy(), res["bucket"]


def probs_for(data, r, logits_row):
    k = int(data["k"][r])
    return scaled_probs(logits_row[:k], float(data["temperature"][r]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--mesh", default="1x1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--trace-region", type=int, default=512 * 1024 * 1024)
    ap.add_argument("--seq-buckets", default=None, help="comma list; default: 512 only (the stage 6 protocol)")
    ap.add_argument(
        "--row-buckets", default=None, help="comma list; default: the buckets the placements need from ROW_BUCKETS"
    )
    ap.add_argument("--corpus", default=CORPUS, help="reference corpus npz (default: the English corpus)")
    ap.add_argument("--index", default=None, help="corpus index JSON (default: derived from --corpus)")
    ap.add_argument(
        "--model-dir", default=None, help="checkpoint directory (default: LAYA_MODEL_DIR or the English checkpoint)"
    )
    ap.add_argument(
        "--row-buckets-by-seq", default=None, help='JSON of per-seq row lists, e.g. {"1024": "1,2,4,5,8,10,16"}'
    )
    ap.add_argument(
        "--largest-rows",
        type=int,
        default=64,
        help="rows of the largest placement (64: the 16 questions plus 48 filler rows; 16: the 16 questions alone in one call)",
    )
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    if a.model_dir:
        os.environ["LAYA_MODEL_DIR"] = a.model_dir
    index_path = a.index or (INDEX if a.corpus == CORPUS else index_for(a.corpus))
    by_seq = row_buckets_by_seq_arg(a.row_buckets_by_seq)
    largest = f"b{a.largest_rows}"
    placements = PLACEMENTS[:-1] + (largest,)

    from models.autoports.convaiinnovations_laya.tt import model_config as mc
    from models.autoports.convaiinnovations_laya.tt.engine import LayaEngine, parse_mesh_shape

    policy = mc.policy_from_name(a.policy)
    os.makedirs(DOC_DIR, exist_ok=True)
    out = a.out or os.path.join(DOC_DIR, "decision_agreement.json")
    seq_buckets = tuple(sorted(int(x) for x in a.seq_buckets.split(","))) if a.seq_buckets else (SEQ,)
    data = load_corpus(a.corpus, index_path)
    mesh = parse_mesh_shape(a.mesh)
    num_devices = mesh[0] * mesh[1]
    qrows = pick_questions(data)
    gate_rows = [r for r in range(len(data["items"])) if int(data["source"][r]) == 0]
    others = [r for r in gate_rows if r not in qrows]
    rng = np.random.RandomState(a.seed)
    filler = [int(x) for x in rng.choice(others, a.largest_rows - N_QUESTIONS, replace=False)]
    mixed_order = [int(x) for x in rng.permutation(qrows)]
    if a.row_buckets:
        per_device_buckets = tuple(sorted(int(x) for x in a.row_buckets.split(",")))
    else:
        per_device_buckets = tuple(
            sorted({mc.pick_bucket(-(-n // num_devices), mc.ROW_BUCKETS) for n in (1, 2, 4, 8, a.largest_rows)})
        )
    t0 = time.perf_counter()
    engine = LayaEngine(
        model_dir=a.model_dir,
        mesh_shape=mesh,
        policy=policy,
        row_buckets=per_device_buckets,
        seq_buckets=seq_buckets,
        trace=True,
        trace_region_size=a.trace_region,
        warmup_shapes=[(b, s) for s in seq_buckets for b in mc.rows_for_seq(s, per_device_buckets, by_seq)],
        threads=a.threads,
        row_buckets_by_seq=by_seq,
    )
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "policy": policy.describe(),
        "mesh": a.mesh,
        "corpus": a.corpus,
        "index": index_path,
        "model_dir": a.model_dir or os.environ.get("LAYA_MODEL_DIR"),
        "engine": engine.shapes(),
        "engine_load_seconds": round(time.perf_counter() - t0, 1),
        "question_rows": qrows,
        "mixed_b8_order": mixed_order,
        "largest_rows": a.largest_rows,
        "largest_placement": largest,
        f"{largest}_filler_rows": filler,
        "loadavg_start": loadavg(),
        "questions": [],
        "gates": {},
    }
    probs = {p: {} for p in placements}
    logits = {p: {} for p in placements}
    buckets = {}
    bucket_hist = {}

    def note(bk):
        key = f"{bk[0]}x{bk[1]}"
        bucket_hist[key] = bucket_hist.get(key, 0) + 1

    for r in qrows:
        lg, _, bk = run_rows(engine, data, [r])
        note(bk)
        probs["alone"][r] = probs_for(data, r, lg[0])
        logits["alone"][r] = lg[0, : int(data["k"][r])]
        buckets["alone"] = list(bk)
    for name, size in (("b2", 2), ("b4", 4), ("mixed_b8", 8)):
        for start in range(0, N_QUESTIONS, size):
            rows = mixed_order[start : start + size]
            lg, _, bk = run_rows(engine, data, rows)
            note(bk)
            for i, r in enumerate(rows):
                probs[name][r] = probs_for(data, r, lg[i])
                logits[name][r] = lg[i, : int(data["k"][r])]
            buckets[name] = list(bk)
    rows64 = list(qrows) + filler
    lg, _, bk = run_rows(engine, data, rows64)
    for i, r in enumerate(rows64[:N_QUESTIONS]):
        probs[largest][r] = probs_for(data, r, lg[i])
        logits[largest][r] = lg[i, : int(data["k"][r])]
    note(bk)
    buckets[largest] = list(bk)
    report["buckets"] = buckets
    report["bucket_histogram"] = dict(sorted(bucket_hist.items()))
    report["seq_buckets"] = list(seq_buckets)
    report["row_buckets"] = list(per_device_buckets)
    report[f"{largest}_real_rows"] = len(rows64)
    worst_dp = 0.0
    all_same = True
    for r in qrows:
        it = data["items"][r]
        k = int(data["k"][r])
        p_ref = scaled_probs(data["logits_fp32"][r, :k], float(data["temperature"][r]))
        q = {
            "row": int(r),
            "group": it["group"],
            "qid": it["qid"],
            "type": it["type"],
            "k": k,
            "ref_probs": p_ref.tolist(),
            "ref_argmax": int(p_ref.argmax()),
        }
        argmaxes = {}
        for p in placements:
            q[f"{p}_probs"] = probs[p][r].tolist()
            q[f"{p}_logits"] = [float(v) for v in logits[p][r]]
            q[f"{p}_vs_cpu_max_abs_dp"] = float(np.abs(probs[p][r] - p_ref).max())
            argmaxes[p] = int(probs[p][r].argmax())
        q["argmax"] = argmaxes
        q["same_argmax_across_placements"] = bool(len(set(argmaxes.values())) == 1)
        for p in placements[1:]:
            q[f"alone_vs_{p}_max_abs_dp"] = float(np.abs(probs["alone"][r] - probs[p][r]).max())
            q[f"alone_vs_{p}_max_abs_dlogit"] = float(np.abs(logits["alone"][r] - logits[p][r]).max())
            worst_dp = max(worst_dp, q[f"alone_vs_{p}_max_abs_dp"])
        q[f"mixed_b8_vs_{largest}_max_abs_dp"] = float(np.abs(probs["mixed_b8"][r] - probs[largest][r]).max())
        q["max_abs_dp_any_pair"] = float(
            max(np.abs(probs[x][r] - probs[y][r]).max() for x in placements for y in placements)
        )
        all_same = all_same and q["same_argmax_across_placements"]
        report["questions"].append(q)
    report["summary"] = {
        "questions": N_QUESTIONS,
        "placements": list(placements),
        "same_argmax_all_placements": int(sum(q["same_argmax_across_placements"] for q in report["questions"])),
        "max_abs_dp_alone_vs": {
            p: float(max(q[f"alone_vs_{p}_max_abs_dp"] for q in report["questions"])) for p in placements[1:]
        },
        "median_abs_dp_alone_vs": {
            p: float(np.median([q[f"alone_vs_{p}_max_abs_dp"] for q in report["questions"]])) for p in placements[1:]
        },
        "max_abs_dlogit_alone_vs": {
            p: float(max(q[f"alone_vs_{p}_max_abs_dlogit"] for q in report["questions"])) for p in placements[1:]
        },
        "pcc_logits_alone_vs": {
            p: pcc(np.concatenate([logits["alone"][r] for r in qrows]), np.concatenate([logits[p][r] for r in qrows]))
            for p in placements[1:]
        },
        f"max_abs_dp_mixed_b8_vs_{largest}": float(
            max(q[f"mixed_b8_vs_{largest}_max_abs_dp"] for q in report["questions"])
        ),
        "max_abs_dp_any_pair": float(max(q["max_abs_dp_any_pair"] for q in report["questions"])),
        "vs_cpu_max_abs_dp": {
            p: float(max(q[f"{p}_vs_cpu_max_abs_dp"] for q in report["questions"])) for p in placements
        },
        "vs_cpu_argmax_agree": {
            p: int(sum(q["argmax"][p] == q["ref_argmax"] for q in report["questions"])) for p in placements
        },
    }
    report["gates"] = {
        "same_argmax": {
            "value": report["summary"]["same_argmax_all_placements"],
            "of": N_QUESTIONS,
            "pass": bool(all_same),
        },
        "max_abs_dp_alone_vs_in_batch": {
            "value": worst_dp,
            "threshold": GATE_MAX_ABS_DP,
            "pass": bool(worst_dp <= GATE_MAX_ABS_DP),
        },
    }
    report["pass"] = all(v["pass"] for v in report["gates"].values())
    engine.close()
    report["loadavg_end"] = loadavg()
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("INVARIANCE_SUMMARY", json.dumps(report["summary"]))
    print("INVARIANCE_GATES", json.dumps(report["gates"]))
    print("INVARIANCE_DONE", out, "pass", report["pass"])


if __name__ == "__main__":
    main()

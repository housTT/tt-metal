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

CORPUS = "/home/hous/dev/laya/reference/parity_corpus.npz"
INDEX = "/home/hous/dev/laya/reference/parity_corpus_index.json"
DOC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "doc", "full_model")
PAD_ID = 50283
SEQ = 512
CONFIDENT_MARGIN = 0.10
GATES = {"confident_agreement": 0.98, "median_max_abs_dp": 0.02, "scorer_pcc": 0.99, "hidden_pcc": 0.99}
TYPE_NAMES = {0: "choice", 1: "score", 2: "noul"}


def loadavg():
    with open("/proc/loadavg") as f:
        return [float(x) for x in f.read().split()[:3]]


def pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def scaled_probs(logits_k, t):
    z = np.asarray(logits_k, dtype=np.float32) / float(t)
    e = np.exp(z - z.max())
    return e / e.sum()


def load_corpus(corpus=CORPUS, index=INDEX):
    z = np.load(corpus, allow_pickle=True)
    with open(index) as f:
        idx = json.load(f)
    items = idx["items"]
    data = {k: z[k] for k in z.files}
    data["items"] = items
    calls = {}
    for r in range(len(items)):
        key = (int(data["source"][r]), int(data["group_index"][r]))
        calls.setdefault(key, []).append(r)
    data["calls"] = [calls[k] for k in sorted(calls)]
    return data


def call_tensors(data, rows):
    ids = torch.as_tensor(data["input_ids"][rows]).long()
    att = torch.as_tensor(data["attention_mask"][rows]).long()
    length = int(att.sum(-1).max())
    ids = ids.masked_fill(ids < 0, PAD_ID)[:, :length]
    att = att[:, :length]
    kmax = int(max(int(k) for k in data["k"][rows]))
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": torch.as_tensor(data["marker_pos"][rows]).long()[:, :kmax],
        "marker_mask": torch.as_tensor(data["marker_mask"][rows]).bool()[:, :kmax],
        "qtype": torch.as_tensor(data["qtype"][rows]).long(),
    }


def item_metrics(data, r, tt_logits_row, tt_act_row):
    k = int(data["k"][r])
    t = float(data["temperature"][r])
    ref = np.asarray(data["logits_fp32"][r, :k], dtype=np.float32)
    tt = np.asarray(tt_logits_row[:k], dtype=np.float32)
    p_ref = scaled_probs(ref, t)
    p_tt = scaled_probs(tt, t)
    s = np.sort(p_ref)[::-1]
    margin = float(s[0] - s[1]) if k >= 2 else 1.0
    ref_act = np.asarray(data["act_logits_fp32"][r], dtype=np.float32)
    it = data["items"][r]
    return {
        "row": int(r),
        "source": int(data["source"][r]),
        "group": it["group"],
        "qid": it["qid"],
        "type": it["type"],
        "k": k,
        "temperature": t,
        "temperature_bucket": it.get("temperature_bucket"),
        "ref_logits": ref.tolist(),
        "tt_logits": tt.tolist(),
        "ref_probs": p_ref.tolist(),
        "tt_probs": p_tt.tolist(),
        "max_abs_dp": float(np.abs(p_tt - p_ref).max()),
        "max_abs_dlogit": float(np.abs(tt - ref).max()),
        "argmax_ref": int(p_ref.argmax()),
        "argmax_tt": int(p_tt.argmax()),
        "argmax_agree": bool(p_ref.argmax() == p_tt.argmax()),
        "ref_margin": margin,
        "confident": bool(margin >= CONFIDENT_MARGIN),
        "ref_act_logits": ref_act.tolist(),
        "tt_act_logits": [float(v) for v in tt_act_row],
        "act_argmax_agree": bool(int(ref_act.argmax()) == int(np.asarray(tt_act_row).argmax())),
        "nan": bool(np.isnan(tt).any() or np.isnan(np.asarray(tt_act_row)).any()),
    }


def aggregate(records, data, label, breakdown=True):
    if not records:
        return {"label": label, "n": 0}
    dps = np.array([x["max_abs_dp"] for x in records])
    conf = [x for x in records if x["confident"]]
    ref_all, tt_all = [], []
    for x in records:
        ref_all += x["ref_logits"]
        tt_all += x["tt_logits"]
    ref_act = np.array([x["ref_act_logits"] for x in records])
    tt_act = np.array([x["tt_act_logits"] for x in records])
    out = {
        "label": label,
        "n": len(records),
        "argmax_agree": int(sum(x["argmax_agree"] for x in records)),
        "argmax_agree_rate": float(sum(x["argmax_agree"] for x in records) / len(records)),
        "confident_n": len(conf),
        "confident_agree": int(sum(x["argmax_agree"] for x in conf)),
        "confident_agree_rate": float(sum(x["argmax_agree"] for x in conf) / len(conf)) if conf else float("nan"),
        "max_abs_dp_max": float(dps.max()),
        "max_abs_dp_mean": float(dps.mean()),
        "max_abs_dp_median": float(np.median(dps)),
        "max_abs_dp_p95": float(np.percentile(dps, 95)),
        "max_abs_dlogit": float(max(x["max_abs_dlogit"] for x in records)),
        "scorer_logit_pcc": pcc(ref_all, tt_all),
        "act_logit_pcc": pcc(ref_act, tt_act),
        "act_argmax_agree": int(sum(x["act_argmax_agree"] for x in records)),
        "act_argmax_agree_rate": float(sum(x["act_argmax_agree"] for x in records) / len(records)),
        "nan_rows": int(sum(x["nan"] for x in records)),
        "flips": [
            {
                "row": x["row"],
                "group": x["group"],
                "qid": x["qid"],
                "type": x["type"],
                "ref_probs": x["ref_probs"],
                "tt_probs": x["tt_probs"],
                "ref_margin": x["ref_margin"],
                "confident": x["confident"],
            }
            for x in records
            if not x["argmax_agree"]
        ],
        "worst_20": sorted(
            [
                {
                    "row": x["row"],
                    "group": x["group"],
                    "qid": x["qid"],
                    "type": x["type"],
                    "max_abs_dp": x["max_abs_dp"],
                    "max_abs_dlogit": x["max_abs_dlogit"],
                    "ref_margin": x["ref_margin"],
                }
                for x in records
            ],
            key=lambda d: -d["max_abs_dp"],
        )[:20],
    }
    if not breakdown:
        out.pop("flips", None)
        out.pop("worst_20", None)
        return out
    by_type = {}
    for name in ("choice", "score", "noul"):
        sub = [x for x in records if x["type"] == name]
        if sub:
            by_type[name] = aggregate(sub, data, f"{label}/{name}", breakdown=False)
    out["by_type"] = by_type
    by_bucket = {}
    for b in sorted({x["temperature_bucket"] for x in records if x["temperature_bucket"]}):
        sub = [x for x in records if x["temperature_bucket"] == b]
        a = aggregate(sub, data, f"{label}/{b}", breakdown=False)
        by_bucket[b] = {
            kk: a[kk]
            for kk in (
                "n",
                "argmax_agree_rate",
                "confident_n",
                "confident_agree_rate",
                "max_abs_dp_median",
                "max_abs_dp_max",
                "scorer_logit_pcc",
            )
        }
    out["by_temperature_bucket"] = by_bucket
    return out


def hidden_check(engine, reference, data, rows, cache):
    """Encoder output after the final norm and head output, device eager versus CPU fp32, real positions only."""
    t = call_tensors(data, rows)
    key = tuple(rows)
    if key not in cache:
        t0 = time.perf_counter()
        logits_ref, act_ref, hidden = reference.forward_with_hidden(
            t["input_ids"], t["attention_mask"], t["marker_pos"], t["marker_mask"], t["qtype"]
        )
        type_emb = reference.model.type_emb(t["qtype"])[:, None, :]
        enc_ref = (hidden["after_type_emb"] - type_emb).float()
        head_ref = hidden["head"][-1].float()
        cache[key] = {"enc": enc_ref, "head": head_ref, "logits": logits_ref, "cpu_seconds": time.perf_counter() - t0}
    c = cache[key]
    dev = engine.model.forward_with_hidden(t["input_ids"], t["attention_mask"], t["qtype"])
    real = t["attention_mask"] == 1
    n = t["input_ids"].shape[0]
    per_row_enc = [pcc(c["enc"][i][real[i]], dev["encoder"][i][real[i]]) for i in range(n)]
    per_row_head = [pcc(c["head"][i][real[i]], dev["head"][i][real[i]]) for i in range(n)]
    from models.autoports.convaiinnovations_laya.tt.engine import host_tail

    lg_eager, _ = host_tail(dev["logits"], dev["cls"], t["marker_pos"], t["marker_mask"], engine.act_head)
    return {
        "rows": [int(r) for r in rows],
        "group": data["items"][rows[0]]["group"],
        "encoder_pcc": pcc(c["enc"][real], dev["encoder"][real]),
        "head_pcc": pcc(c["head"][real], dev["head"][real]),
        "cls_pcc": pcc(c["head"][:, 0], dev["cls"]),
        "encoder_pcc_per_row": per_row_enc,
        "head_pcc_per_row": per_row_head,
        "encoder_max_abs": float(c["enc"][real].abs().max()),
        "head_max_abs": float(c["head"][real].abs().max()),
        "eager_marker_logits_vs_cpu_max_abs": float((lg_eager - c["logits"]).abs().max()),
        "cpu_seconds": c["cpu_seconds"],
        "bucket": list(dev["bucket"]),
        "pooled": {
            "enc_ref": c["enc"][real],
            "enc_dev": dev["encoder"][real],
            "head_ref": c["head"][real],
            "head_dev": dev["head"][real],
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--mesh", default="1x1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--items", choices=["gate", "all"], default="all")
    ap.add_argument("--hidden-cases", type=int, default=40)
    ap.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="re-split every call into chunks of this many rows (B 2 and B 4 end-to-end rows)",
    )
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--trace-region", type=int, default=512 * 1024 * 1024)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.environ.setdefault("LAYA_CPU_THREADS", str(a.threads))

    from models.autoports.convaiinnovations_laya.reference.laya_reference import LayaReference
    from models.autoports.convaiinnovations_laya.tt import model_config as mc
    from models.autoports.convaiinnovations_laya.tt.engine import LayaEngine, parse_mesh_shape

    policy = mc.policy_from_name(a.policy)
    os.makedirs(DOC_DIR, exist_ok=True)
    suffix = f"_b{a.chunk}" if a.chunk else ""
    out = a.out or os.path.join(DOC_DIR, f"fidelity_{policy.name}_{a.mesh}{suffix}.json")
    data = load_corpus()
    calls = data["calls"] if a.items == "all" else [c for c in data["calls"] if int(data["source"][c[0]]) == 0]
    if a.chunk:
        calls = [c[i : i + a.chunk] for c in calls for i in range(0, len(c), a.chunk)]
    mesh = parse_mesh_shape(a.mesh)
    sizes = sorted({len(c) for c in calls})
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "policy": policy.describe(),
        "mesh": a.mesh,
        "trace": not a.no_trace,
        "corpus": CORPUS,
        "index": INDEX,
        "calls": len(calls),
        "call_sizes": sizes,
        "chunk": a.chunk,
        "loadavg_start": loadavg(),
        "gates": {},
    }
    t0 = time.perf_counter()
    num_devices = mesh[0] * mesh[1]
    per_device = sorted({-(-s // num_devices) for s in sizes})
    row_buckets = tuple(sorted({mc.pick_bucket(p, mc.ROW_BUCKETS) for p in per_device}))
    engine = LayaEngine(
        mesh_shape=mesh,
        policy=policy,
        row_buckets=row_buckets,
        seq_buckets=(SEQ,),
        trace=not a.no_trace,
        trace_region_size=a.trace_region,
        warmup_shapes=[(b, SEQ) for b in row_buckets],
        threads=a.threads,
    )
    report["engine"] = engine.shapes()
    report["engine_load_seconds"] = round(time.perf_counter() - t0, 1)
    records = []
    device_ms = []
    for rows in calls:
        t = call_tensors(data, rows)
        res = engine.forward_detailed(
            t["input_ids"], t["attention_mask"], t["marker_pos"], t["marker_mask"], t["qtype"]
        )
        device_ms.append(res["device_ms"])
        lg = res["logits"].numpy()
        act = res["act_logits"].numpy()
        for i, r in enumerate(rows):
            records.append(item_metrics(data, r, lg[i], act[i]))
    report["device_ms_per_call_p50"] = float(np.median(device_ms))
    report["device_ms_per_call_max"] = float(np.max(device_ms))
    report["loadavg_after_device_pass"] = loadavg()
    gate_records = [x for x in records if x["source"] == 0]
    report["gate_subset"] = aggregate(gate_records, data, "gate subset (200 typed-decisions decisions)")
    if a.items == "all":
        report["all_items"] = aggregate(records, data, "all 488 corpus items")
        report["parity_fast"] = aggregate([x for x in records if x["source"] == 1], data, "parity_fast (288 items)")
    hidden = None
    if a.hidden_cases > 0:
        reference = LayaReference(threads=a.threads)
        report["reference_load_seconds"] = round(reference.load_seconds, 1)
        cache = {}
        gate_calls = [c for c in calls if int(data["source"][c[0]]) == 0][: a.hidden_cases]
        per_call = []
        pooled = {"enc_ref": [], "enc_dev": [], "head_ref": [], "head_dev": []}
        for rows in gate_calls:
            h = hidden_check(engine, reference, data, rows, cache)
            p = h.pop("pooled")
            for k in pooled:
                pooled[k].append(p[k])
            per_call.append(h)
            print(
                "HIDDEN",
                json.dumps({k: v for k, v in h.items() if k not in ("encoder_pcc_per_row", "head_pcc_per_row")}),
            )
        hidden = {
            "cases": len(per_call),
            "encoder_pcc_pooled": pcc(torch.cat(pooled["enc_ref"]), torch.cat(pooled["enc_dev"])),
            "head_pcc_pooled": pcc(torch.cat(pooled["head_ref"]), torch.cat(pooled["head_dev"])),
            "encoder_pcc_min_call": float(min(h["encoder_pcc"] for h in per_call)),
            "head_pcc_min_call": float(min(h["head_pcc"] for h in per_call)),
            "encoder_pcc_min_row": float(min(min(h["encoder_pcc_per_row"]) for h in per_call)),
            "head_pcc_min_row": float(min(min(h["head_pcc_per_row"]) for h in per_call)),
            "cls_pcc_min_call": float(min(h["cls_pcc"] for h in per_call)),
            "eager_vs_cpu_marker_logits_max_abs": float(max(h["eager_marker_logits_vs_cpu_max_abs"] for h in per_call)),
            "cpu_seconds_total": float(sum(h["cpu_seconds"] for h in per_call)),
            "per_call": per_call,
        }
        report["hidden_states"] = hidden
    g = report["gate_subset"]
    gates = {
        "confident_argmax_agreement": {
            "value": g["confident_agree_rate"],
            "agree": g["confident_agree"],
            "of": g["confident_n"],
            "threshold": GATES["confident_agreement"],
            "pass": bool(g["confident_agree_rate"] >= GATES["confident_agreement"]),
        },
        "median_max_abs_dp": {
            "value": g["max_abs_dp_median"],
            "threshold": GATES["median_max_abs_dp"],
            "pass": bool(g["max_abs_dp_median"] <= GATES["median_max_abs_dp"]),
        },
        "scorer_logit_pcc": {
            "value": g["scorer_logit_pcc"],
            "threshold": GATES["scorer_pcc"],
            "pass": bool(g["scorer_logit_pcc"] >= GATES["scorer_pcc"]),
        },
        "no_nan": {"value": g["nan_rows"], "threshold": 0, "pass": bool(g["nan_rows"] == 0)},
    }
    if hidden is not None:
        gates["hidden_state_pcc"] = {
            "encoder_pooled": hidden["encoder_pcc_pooled"],
            "head_pooled": hidden["head_pcc_pooled"],
            "encoder_min_call": hidden["encoder_pcc_min_call"],
            "head_min_call": hidden["head_pcc_min_call"],
            "threshold": GATES["hidden_pcc"],
            "pass": bool(min(hidden["encoder_pcc_pooled"], hidden["head_pcc_pooled"]) >= GATES["hidden_pcc"]),
        }
    report["gates"] = gates
    report["reported_not_gated"] = {
        "plain_argmax_agreement": g["argmax_agree_rate"],
        "act_argmax_agreement": g["act_argmax_agree_rate"],
        "act_logit_pcc": g["act_logit_pcc"],
        "nan_count": g["nan_rows"],
        "p95_max_abs_dp": g["max_abs_dp_p95"],
        "max_abs_dp": g["max_abs_dp_max"],
        "max_abs_dlogit": g["max_abs_dlogit"],
    }
    report["pass"] = all(v["pass"] for v in gates.values())
    report["records"] = records
    engine.close()
    report["loadavg_end"] = loadavg()
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("FIDELITY_GATES", json.dumps(gates))
    print(
        "FIDELITY_SUMMARY",
        json.dumps({k: v for k, v in g.items() if k not in ("flips", "worst_20", "by_type", "by_temperature_bucket")}),
    )
    print("FIDELITY_DONE", out, "pass", report["pass"])


if __name__ == "__main__":
    main()

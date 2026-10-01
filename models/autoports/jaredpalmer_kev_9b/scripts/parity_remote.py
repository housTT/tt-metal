import argparse
import json
import statistics
import time

import httpx
import torch

REF = "/home/hous/dev/kev/reports/reference"


def api_request(record):
    return {
        "state": record["state"],
        "questions": {
            qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
            for qid, q in record["questions"].items()
        },
    }


def vec(a, qtype, keys):
    if qtype == "noul":
        return torch.tensor([1 - a["noul"], a["noul"]])
    return torch.tensor([a["probabilities"][k] for k in keys])


def ref_vec(rq, qtype):
    if qtype == "noul":
        return torch.tensor([rq["probabilities"]["false"], rq["probabilities"]["true"]])
    return torch.tensor([rq["probabilities"][k] for k in rq["keys"]])


def agreement(pairs):
    dp = [float((p - q).abs().max()) for p, q in pairs]
    return {
        "max_dp": max(dp),
        "mean_dp": statistics.mean(dp),
        "argmax_flips": sum(int(p.argmax() != q.argmax()) for p, q in pairs),
    }


def main():
    ap = argparse.ArgumentParser(
        description="16-record parity of a System One server against the fp32 and bf16 CPU references, with a second pass that revisits every state after the others ran."
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8008")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    records = [json.loads(l) for l in open(f"{REF}/records.jsonl") if l.strip()]
    refs = {name: json.load(open(f"{REF}/probs_{name}.json"))["records"] for name in ("fp32", "bf16")}
    client = httpx.Client(base_url=a.base_url, timeout=600)
    pairs = {name: [] for name in refs}
    per_record = []
    first = {}
    revisit_equal = 0
    revisit_total = 0
    for p in range(a.passes):
        order = list(range(len(records))) if p == 0 else list(reversed(range(len(records))))
        for i in order:
            rec = records[i]
            t = time.perf_counter()
            r = client.post("/v1/systemone", json={**api_request(rec), "model": "kev-latest"})
            wall = (time.perf_counter() - t) * 1000
            r.raise_for_status()
            body = r.json()
            if p == 0:
                first[i] = body["answers"]
                row = {
                    "record": i,
                    "latency_ms": body["latency_ms"],
                    "wall_ms": round(wall, 1),
                    "input_tokens": body["usage"]["input_tokens"],
                    "questions": {},
                }
                for qid, ans in body["answers"].items():
                    qtype = rec["questions"][qid]["type"]
                    entry = {}
                    for name, ref in refs.items():
                        rq = ref[str(i)]["questions"][qid]
                        pv, qv = vec(ans, qtype, rq["keys"]), ref_vec(rq, qtype)
                        pairs[name].append((pv, qv))
                        entry[name] = {
                            "max_dp": float((pv - qv).abs().max()),
                            "argmax_tt": int(pv.argmax()),
                            "argmax_ref": int(qv.argmax()),
                        }
                    row["questions"][qid] = {"type": qtype, "served": ans, **entry}
                per_record.append(row)
                print(
                    p,
                    i,
                    row["input_tokens"],
                    row["latency_ms"],
                    {n: round(max(row["questions"][q][n]["max_dp"] for q in row["questions"]), 4) for n in refs},
                    flush=True,
                )
            else:
                revisit_total += 1
                same = body["answers"] == first[i]
                revisit_equal += int(same)
                per_record[i].setdefault("revisits", []).append(
                    {"pass": p, "latency_ms": body["latency_ms"], "equal": same}
                )
                print(p, i, body["latency_ms"], "equal" if same else "DIFFERENT", flush=True)
    ref_pair = [
        (ref_vec(refs["bf16"][k]["questions"][q], "choice"), ref_vec(refs["fp32"][k]["questions"][q], "choice"))
        for k in refs["fp32"]
        for q in refs["fp32"][k]["questions"]
    ]
    card = client.get("/v1/models").json()["models"][0]
    out = {
        "base_url": a.base_url,
        "served_model": card,
        "records": len(records),
        "passes": a.passes,
        "summary_vs": {name: {"questions": len(v), **agreement(v)} for name, v in pairs.items()},
        "revisits": {
            "total": revisit_total,
            "answers_equal_to_first_pass": revisit_equal,
            "prefix_cache": card["prefix_cache"],
        },
        "note": "served probabilities are rounded to 4 decimals by the API (kev round_prob); max_dp therefore includes up to 5e-5 rounding",
        "bf16_vs_fp32_reference": {"questions": len(ref_pair), **agreement(ref_pair)},
        "per_record": per_record,
        "latency_ms": {
            "median": statistics.median(r["latency_ms"] for r in per_record),
            "max": max(r["latency_ms"] for r in per_record),
            "sum": sum(r["latency_ms"] for r in per_record),
        },
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("summary_vs", "revisits", "bf16_vs_fp32_reference", "latency_ms")}, indent=1))


if __name__ == "__main__":
    main()

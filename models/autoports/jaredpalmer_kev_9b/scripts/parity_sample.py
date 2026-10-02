import argparse
import json
import statistics
import time
from pathlib import Path

KEV = Path("/home/hous/dev/kev/kev")
REF = Path("/home/hous/dev/kev/reports/reference/records.jsonl")
ADAPTER = (
    "/home/hous/.cache/huggingface/hub/models--jaredpalmer--kev-9b/snapshots/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0"
)
SUITES = ("hard-v1", "devtools-v1", "documents-v1")
SKIP, TAKE = 4, 16
MARGIN = 0.05


def api_request(record):
    return {
        "state": record["state"],
        "questions": {
            qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
            for qid, q in record["questions"].items()
        },
    }


def keys(q):
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(q["criteria"]))]


def placeholder_labels(record):
    for q in record["questions"].values():
        if "label" not in q:
            q["label"] = keys(q)[0] if q["type"] == "choice" else (False if q["type"] == "noul" else 0)
            q["src"] = "reference"
    return record


def build(out):
    records = []
    for suite in SUITES:
        lines = (KEV / "evals" / suite / "development.jsonl").read_text(encoding="utf-8").splitlines()
        for i in range(SKIP, SKIP + TAKE):
            r = json.loads(lines[i])
            r["_meta"] = {**r.get("_meta", {}), "parity_source": f"evals/{suite}/development.jsonl", "parity_index": i}
            records.append(r)
    for i, line in enumerate(REF.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            r = json.loads(line)
            r["_meta"] = {**r.get("_meta", {}), "parity_source": "reference", "parity_index": i}
            records.append(placeholder_labels(r))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(len(records), "records ->", out)


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def cpu(records, out):
    import torch
    from kev.predictors import LocalPredictor
    from kev.suite import SERVING_CONTEXT

    t = time.perf_counter()
    pred = LocalPredictor(ADAPTER, "cpu", context=SERVING_CONTEXT)
    result = {
        "adapter": ADAPTER,
        "dtype": pred.model.dtype,
        "temperature": pred.temperature,
        "environment": pred.environment,
        "torch_threads": torch.get_num_threads(),
        "load_s": round(time.perf_counter() - t, 1),
        "records": {},
    }
    print(time.strftime("%H:%M:%S"), "loaded", result["dtype"], "in", result["load_s"], "s", flush=True)
    for i, record in enumerate(records):
        res = pred(record)
        result["records"][str(i)] = {
            "source": record["_meta"]["parity_source"],
            "input_tokens": res["input_tokens"],
            "latency_ms": res["latency_ms"],
            "probabilities": res["probabilities"],
        }
        out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(time.strftime("%H:%M:%S"), i, res["input_tokens"], "tokens", round(res["latency_ms"]), "ms", flush=True)


def server(records, out, base_url):
    import httpx

    client = httpx.Client(base_url=base_url, timeout=600)
    result = {"base_url": base_url, "records": {}}
    for i, record in enumerate(records):
        t = time.perf_counter()
        r = client.post("/v1/systemone", json={**api_request(record), "model": "kev-latest"})
        r.raise_for_status()
        body = r.json()
        result["records"][str(i)] = {
            "source": record["_meta"]["parity_source"],
            "input_tokens": body["usage"]["input_tokens"],
            "latency_ms": body["latency_ms"],
            "wall_ms": round((time.perf_counter() - t) * 1000, 1),
            "answers": body["answers"],
        }
        print(i, body["usage"]["input_tokens"], body["latency_ms"], flush=True)
    result["served_model"] = client.get("/v1/models").json()["models"][0]
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


def served_vec(ans, ks):
    if ans["type"] == "noul":
        return [1 - ans["noul"], ans["noul"]]
    return [ans["probabilities"][k] for k in ks]


def agreement(pairs):
    dp = [max(abs(a - b) for a, b in zip(p, q)) for p, q in pairs]
    return {
        "questions": len(pairs),
        "max_dp": max(dp),
        "mean_dp": statistics.mean(dp),
        "argmax_flips": sum(int(argmax(p) != argmax(q)) for p, q in pairs),
    }


def argmax(v):
    return max(range(len(v)), key=lambda i: v[i])


def compare(records, cpu_path, server_path, out):
    ref = json.loads(Path(cpu_path).read_text(encoding="utf-8"))["records"]
    tt = json.loads(Path(server_path).read_text(encoding="utf-8"))["records"]
    pairs, by_source, flips = [], {}, []
    for i, record in enumerate(records):
        k = str(i)
        if k not in ref or k not in tt:
            continue
        for qid, q in record["questions"].items():
            ks = keys(q)
            p = served_vec(tt[k]["answers"][qid], ks)
            r = [ref[k]["probabilities"][qid][x] for x in ks]
            pairs.append((p, r))
            by_source.setdefault(record["_meta"]["parity_source"], []).append((p, r))
            if argmax(p) != argmax(r):
                top = sorted(r, reverse=True)
                margin = top[0] - top[1]
                flips.append(
                    {
                        "record": i,
                        "question": qid,
                        "type": q["type"],
                        "source": record["_meta"]["parity_source"],
                        "ref_top2_margin": round(margin, 4),
                        "kind": "flip" if margin >= MARGIN else "near_tie",
                        "probs_ref": [round(x, 4) for x in r],
                        "probs_tt": p,
                    }
                )
    result = {
        "records_compared": len({k for k in ref if k in tt}),
        "cpu": cpu_path,
        "server": server_path,
        "summary": agreement(pairs),
        "flips_margin_ge_0_05": sum(f["kind"] == "flip" for f in flips),
        "flips_near_tie": sum(f["kind"] == "near_tie" for f in flips),
        "flip_detail": flips,
        "by_source": {s: agreement(v) for s, v in sorted(by_source.items())},
        "note": "served probabilities are rounded to 4 decimals by the API; a flip counts only when the fp32 top-2 margin is >= 0.05",
    }
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "flip_detail"}, indent=1))
    for f in flips:
        print(f)


def main():
    ap = argparse.ArgumentParser(
        description="64-record parity sample: 16 development records from each of hard-v1, devtools-v1, documents-v1 plus the 16 reference records, through the server and through kev's CPU fp32 LocalPredictor."
    )
    ap.add_argument("cmd", choices=("build", "cpu", "server", "compare"))
    ap.add_argument("--records", default="/home/hous/dev/kev/reports/parity64/records.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8008")
    ap.add_argument("--cpu", default="/home/hous/dev/kev/reports/parity64/cpu_fp32.json")
    ap.add_argument("--server", default="/home/hous/dev/kev/reports/parity64/server_x4.json")
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd == "build":
        build(out)
        return
    records = load(a.records)
    if a.cmd == "cpu":
        cpu(records, out)
    elif a.cmd == "server":
        server(records, out, a.base_url)
    else:
        compare(records, a.cpu, a.server, out)


if __name__ == "__main__":
    main()

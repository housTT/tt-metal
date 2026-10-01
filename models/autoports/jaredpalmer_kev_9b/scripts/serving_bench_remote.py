import argparse
import hashlib
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "returns": "Exchanges, refunds, wrong or damaged items",
            "shipping": "Delivery status, delays, lost packages",
            "billing": "Charges, invoices, payment problems",
        },
    },
    "return_reason": {
        "type": "choice",
        "instructions": "If the customer wants to return something, why?",
        "criteria": {
            "wrong_size": "The item doesn't fit",
            "wrong_item": "A different product was delivered",
            "damaged": "The item arrived broken or faulty",
            "changed_mind": "The item is fine, the customer no longer wants it",
            "other": "A return reason that fits none of the above",
        },
    },
    "requested_resolution": {
        "type": "choice",
        "instructions": "What does the customer want to happen?",
        "criteria": {
            "exchange": "Swap the item for a different one",
            "refund": "Money back",
            "replacement": "The same item sent again",
            "information": "Just an answer, no action needed",
        },
    },
    "tone": {
        "type": "choice",
        "instructions": "What is the customer's tone?",
        "criteria": {"calm": None, "frustrated": None, "angry": None},
    },
    "escalate": {"type": "noul", "instructions": "Does this message require urgent human attention?"},
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
}
TICKET = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. What are you going to do about this?"
PARAGRAPH = (
    "I ordered a pair of running shoes on the first of the month and paid with my credit card. The confirmation email said "
    "delivery in three to five business days, but the tracking page did not update for over a week, and when the package "
    "finally arrived the box was crushed on one side. The shoes inside were a size ten instead of the size nine I ordered. "
)
FIVE = {k: QUESTIONS[k] for k in ("department", "return_reason", "requested_resolution", "escalate", "frustration")}
CASES = {
    "2 questions, short state": (TICKET, {k: QUESTIONS[k] for k in ("department", "escalate")}),
    "6 questions, short state": (TICKET, QUESTIONS),
    "5 questions, 370-token state": (PARAGRAPH * 5, FIVE),
    "5 questions, 2,200-token state": (PARAGRAPH * 30, FIVE),
}
SHORT, LONG = "6 questions, short state", "5 questions, 2,200-token state"
THROUGHPUT_KEY = "6 questions, new short state @ 64 clients"
DEFAULT_SUITE = "/home/hous/dev/kev/kev/evals/v7/decision-v7"


def request(case, i):
    state, questions = CASES[case]
    return {"state": state if i == 0 else f"Ticket {i}. {state}", "questions": questions}


class Client:
    def __init__(self, base_url, api_key, model, timeout):
        limits = httpx.Limits(max_connections=256, max_keepalive_connections=256)
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"authorization": f"Bearer {api_key}"},
            timeout=timeout,
            limits=limits,
        )
        self.model = model

    def answer(self, body):
        r = self.http.post("/v1/systemone", json={**body, "model": self.model})
        r.raise_for_status()
        return r.json()

    def models(self):
        r = self.http.get("/v1/models")
        r.raise_for_status()
        return r.json()["models"][0]


def latency(client, reps):
    out = {}
    for case in CASES:
        first = client.answer(request(case, 0))
        row = {"tokens": first["usage"]["input_tokens"], "first_ms": first["latency_ms"]}
        for mode in ("new", "cached"):
            ms = [client.answer(request(case, i if mode == "new" else 0))["latency_ms"] for i in range(1, reps + 3)][2:]
            row[f"{mode}_ms"] = statistics.median(ms)
        out[case] = row
        print(case, row, flush=True)
    return out


def api_request(record):
    return {
        "state": record["state"],
        "questions": {
            qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
            for qid, q in record["questions"].items()
        },
    }


def suite_records(suite, split="development"):
    suite = Path(suite)
    manifest = json.loads((suite / "manifest.json").read_text(encoding="utf-8"))
    path = suite / f"{split}.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["files"][path.name]["sha256"]:
        raise ValueError(f"suite checksum mismatch: {path}")
    return [api_request(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def throughput(client, suite, levels, n=256, n_long=64):
    samples = {
        "6 questions, new short state": [request(SHORT, 1000 + i) for i in range(n)],
        "decision-v7 development": random.Random(0).choices(suite_records(suite), k=n),
        LONG: [request(LONG, 1000 + i) for i in range(n_long)],
    }

    def run(recs, c):
        def one(rec):
            t = time.perf_counter()
            client.answer(rec)
            return time.perf_counter() - t

        with ThreadPoolExecutor(c) as pool:
            start = time.perf_counter()
            lat = sorted(pool.map(one, recs))
            wall = time.perf_counter() - start
        return {
            "p50_ms": round(1000 * statistics.median(lat), 1),
            "p99_ms": round(1000 * lat[int(0.99 * (len(lat) - 1))], 1),
            "requests_per_s": round(len(lat) / wall, 1),
        }

    out = {}
    for name, recs in samples.items():
        first = {c: run(recs, c) for c in levels}
        for c in levels:
            out[f"{name} @ {c} clients"] = {**run(recs, c), "first": first[c]}
            print(name, c, out[f"{name} @ {c} clients"], flush=True)
    return out


def card_row(label, lat, thr):
    s, l = lat[SHORT], lat[LONG]
    rps = thr.get(THROUGHPUT_KEY, {}).get("requests_per_s", "n/a")
    return f"| {label} | {s['new_ms']:.1f} / {s['cached_ms']:.1f} ms | {l['new_ms']:.1f} / {l['cached_ms']:.1f} ms | {rps} |"


def main():
    ap = argparse.ArgumentParser(
        description="Port of kev scripts/serving_bench.py that drives a remote System One server over HTTP."
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8008")
    ap.add_argument(
        "--suite",
        default=DEFAULT_SUITE,
        help="frozen suite directory whose development.jsonl feeds the throughput sample",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument(
        "--concurrency", default="1,8,32,64", help="comma-separated client counts for the throughput section"
    )
    ap.add_argument("--label", default="P150", help="device name for the model-card table row")
    ap.add_argument("--model", default="kev-latest")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="plumbing check: 32 / 8 throughput requests instead of 256 / 64; the report says quick: true",
    )
    a = ap.parse_args()
    levels = tuple(int(x) for x in a.concurrency.split(","))
    n, n_long = (32, 8) if a.quick else (256, 64)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    client = Client(a.base_url, a.api_key, a.model, a.timeout)
    report = {
        "base_url": a.base_url,
        "label": a.label,
        "served": client.models(),
        "suite": a.suite,
        "reps": a.reps,
        "levels": list(levels),
        "quick": a.quick,
    }

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    report["latency"] = latency(client, a.reps)
    save()
    report["throughput"] = {} if a.skip_throughput else throughput(client, a.suite, levels, n, n_long)
    report["records"] = n
    report["card_row"] = card_row(a.label, report["latency"], report["throughput"])
    save()
    print(json.dumps({k: v for k, v in report.items() if k != "served"}, indent=1))
    print("| Device | 6 questions, short state | 5 questions, 2,200-token state | Requests/s, 64 clients |")
    print("|---|---|---|---|")
    print(report["card_row"])


if __name__ == "__main__":
    main()

"""Serving benchmark of a running Clef server over HTTP, in the kev model-card format.

Port of /home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/scripts/serving_bench_remote.py
(itself a port of kev's scripts/serving_bench.py). The schema and states are kev's Apache-2.0 bench
constants (see NOTICE); the 2,200-token state is PARAGRAPH x 30. The blog triage request (3 questions,
the server's warmup record) is added as a Clef-specific row.

Latency section: per case, "new" sends a fresh state each time (Ticket i. prefix, a prefix-cache
miss), "cached" sends the same state (a hit); each column is the median of --reps requests after 2
warm-ups, of the server's latency_ms (model time on the worker, no queue wait, no HTTP).

Throughput section: per concurrency level, a thread pool of c clients sends the sample; the second
of two passes is timed. Samples: 256 requests over 64 distinct short states (6 questions), the 64
dev64 text records of /home/hous/dev/clef/reports/reference/dev64_text.jsonl (distinct states, 260 to
1,774 tokens), and 64 distinct 2,200-token states (5 questions).

Usage (host):
  python serving_bench_remote.py --out /home/hous/dev/clef/reports/bench/<run> [--reps 20]
      [--concurrency 1,8,32,64] [--label "P150x2 (TP=2)"] [--quick] [--skip-throughput]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import read_jsonl

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
        "criteria": {
            "calm": "Polite and composed",
            "frustrated": "Annoyed but civil",
            "angry": "Hostile or threatening",
        },
    },
    "escalate": {"type": "noul", "instructions": "Does this message require urgent human attention?"},
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
}
TRIAGE = {
    "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "Payments, invoices, and refunds",
            "technical": "Outages, errors, and configuration",
            "sales": "Plans and upgrades",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the customer impact?",
        "criteria": ["No impact", "Minor", "Major", "Critical"],
    },
}
TRIAGE_STATE = "Checkout has been failing for every customer for the last hour."
TICKET = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. What are you going to do about this?"
PARAGRAPH = (
    "I ordered a pair of running shoes on the first of the month and paid with my credit card. The confirmation email said "
    "delivery in three to five business days, but the tracking page did not update for over a week, and when the package "
    "finally arrived the box was crushed on one side. The shoes inside were a size ten instead of the size nine I ordered. "
)
FIVE = {k: QUESTIONS[k] for k in ("department", "return_reason", "requested_resolution", "escalate", "frustration")}
CASES = {
    "3 questions, blog triage state": (TRIAGE_STATE, TRIAGE),
    "2 questions, short state": (TICKET, {k: QUESTIONS[k] for k in ("department", "escalate")}),
    "6 questions, short state": (TICKET, QUESTIONS),
    "5 questions, 370-token state": (PARAGRAPH * 5, FIVE),
    "5 questions, 2,200-token state": (PARAGRAPH * 30, FIVE),
}
SHORT, LONG = "6 questions, short state", "5 questions, 2,200-token state"
THROUGHPUT_KEY = "6 questions, new short state @ 64 clients"
DEV64 = "/home/hous/dev/clef/reports/reference/dev64_text.jsonl"


def request(case, i):
    state, questions = CASES[case]
    return {"state": state if i == 0 else f"Ticket {i}. {state}", "questions": questions}


class Client:
    def __init__(self, base_url, api_key, model, timeout):
        limits = httpx.Limits(max_connections=256, max_keepalive_connections=256)
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, limits=limits)
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
            row[f"{mode}_p95_ms"] = sorted(ms)[int(0.95 * (len(ms) - 1))]
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


def throughput(client, levels, n=256, n_states=64, n_long=64):
    dev64 = [api_request(r) for r in read_jsonl(DEV64)]
    samples = {
        "6 questions, new short state": [request(SHORT, 1000 + (i % n_states)) for i in range(n)],
        "dev64 text records": random.Random(0).choices(dev64, k=min(n, len(dev64))) if n < len(dev64) else dev64,
        LONG: [request(LONG, 1000 + i) for i in range(n_long)],
    }

    def run(recs, c):
        def one(rec):
            t = time.perf_counter()
            body = client.answer(rec)
            return time.perf_counter() - t, body["latency_ms"]

        with ThreadPoolExecutor(c) as pool:
            start = time.perf_counter()
            results = list(pool.map(one, recs))
            wall = time.perf_counter() - start
        lat = sorted(r[0] for r in results)
        model_ms = sorted(r[1] for r in results)
        return {
            "p50_ms": round(1000 * statistics.median(lat), 1),
            "p99_ms": round(1000 * lat[int(0.99 * (len(lat) - 1))], 1),
            "model_p50_ms": round(statistics.median(model_ms), 1),
            "requests_per_s": round(len(lat) / wall, 2),
            "requests": len(lat),
            "wall_s": round(wall, 1),
        }

    out = {}
    for name, recs in samples.items():
        for c in levels:
            first = run(recs, c)
            out[f"{name} @ {c} clients"] = {**run(recs, c), "first": first}
            print(name, c, out[f"{name} @ {c} clients"], flush=True)
    return out


def card_row(label, lat, thr):
    s, l = lat[SHORT], lat[LONG]
    rps = thr.get(THROUGHPUT_KEY, {}).get("requests_per_s", "n/a")
    return f"| {label} | {s['new_ms']:.1f} / {s['cached_ms']:.1f} ms | {l['new_ms']:.1f} / {l['cached_ms']:.1f} ms | {rps} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8008")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--concurrency", default="1,8,32,64")
    ap.add_argument("--label", default="P150x2 (TP=2)")
    ap.add_argument("--model", default="clef")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument("--quick", action="store_true", help="32 / 8 throughput requests instead of 256 / 64")
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
        "reps": a.reps,
        "levels": list(levels),
        "quick": a.quick,
        "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    report["latency"] = latency(client, a.reps)
    save()
    report["throughput"] = {} if a.skip_throughput else throughput(client, levels, n, 64, n_long)
    report["records"] = n
    report["card_row"] = card_row(a.label, report["latency"], report["throughput"])
    report["served_after"] = client.models()
    report["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    save()
    print(json.dumps({k: v for k, v in report.items() if k not in ("served", "served_after")}, indent=1))
    print("| Device | 6 questions, short state | 5 questions, 2,200-token state | Requests/s, 64 clients |")
    print("|---|---|---|---|")
    print(report["card_row"])


if __name__ == "__main__":
    main()

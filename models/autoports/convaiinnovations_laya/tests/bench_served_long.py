# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone

import numpy as np
import requests

STATE_TEXT = "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. "
Q_NOUL = {"type": "noul", "instructions": "Does `ticket.messages[0].text` express urgency?"}
Q_CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "payments", "technical": "bugs and integrations", "sales": "pricing"},
}


def qs(n):
    return {("q%d" % i): (Q_NOUL if i % 2 else Q_CHOICE) for i in range(n)}


def long_state(repeats):
    """The bench_latency.py STATE_EN ticket with its message repeated `repeats` times (6 is the authors' value)."""
    return {"ticket": {"subject": "Payout failing", "messages": [{"from": "customer", "text": STATE_TEXT * repeats}]}}


def loadavg():
    with open("/proc/loadavg") as f:
        return [float(x) for x in f.read().split()[:3]]


def pct(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p)) if xs else None


def call(session, url, body):
    t0 = time.perf_counter()
    r = session.post(url, json=body, timeout=600)
    dt = (time.perf_counter() - t0) * 1000.0
    h = r.headers
    out = {
        "status": r.status_code,
        "client_ms": dt,
        "server_ms": float(h["X-Inference-Time-Ms"]) if "X-Inference-Time-Ms" in h else None,
        "device_ms": float(h["X-Laya-Device-Ms"]) if "X-Laya-Device-Ms" in h else None,
        "batch": h.get("X-Laya-Batch"),
    }
    if r.status_code == 200:
        body = r.json()
        usage = body.get("usage") or (body.get("total_usage") if "results" in body else None) or {}
        out["input_tokens"] = usage.get("input_tokens")
        out["truncated"] = usage.get("truncated")
        answers = body.get("answers") or (body["results"][0]["answers"] if "results" in body else {})
        out["first_answer"] = next(iter(answers.values()), None)
    else:
        out["error"] = r.text[:300]
    return out


def timed(session, url, body, warm, reps):
    for _ in range(warm):
        call(session, url, body)
    recs = [call(session, url, body) for _ in range(reps)]
    ok = [r for r in recs if r["status"] == 200]
    hist = {}
    for r in ok:
        for b in (r["batch"] or "").split(","):
            hist[b] = hist.get(b, 0) + 1
    return {
        "n": len(ok),
        "errors": len(recs) - len(ok),
        "client_ms_p50": pct([r["client_ms"] for r in ok], 50),
        "client_ms_p95": pct([r["client_ms"] for r in ok], 95),
        "server_ms_p50": pct([r["server_ms"] for r in ok if r["server_ms"] is not None], 50),
        "device_ms_p50": pct([r["device_ms"] for r in ok if r["device_ms"] is not None], 50),
        "batch_histogram": hist,
        "input_tokens": ok[0].get("input_tokens") if ok else None,
        "truncated": ok[0].get("truncated") if ok else None,
        "first_answer": ok[0].get("first_answer") if ok else None,
        "sample_error": next((r.get("error") for r in recs if r["status"] != 200), None),
        "loadavg": loadavg(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--repeats", type=int, default=30, help="message repeats of the long state (30 gives rows of about 860 tokens)"
    )
    ap.add_argument("--cells", default="1,5,10,16,50")
    ap.add_argument("--batch-states", default="8,16")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--batch-reps", type=int, default=10)
    a = ap.parse_args(argv)
    s = requests.Session()
    health = s.get(a.base_url + "/v1/health", timeout=60).json()
    state = long_state(a.repeats)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "base_url": a.base_url,
        "protocol": "bench_latency.py shapes (STATE_EN ticket, Q_NOUL and Q_CHOICE alternating) with the ticket message repeated "
        f"{a.repeats} times so every row lands in the 1024-token bucket; warm {a.warm}, p50 of {a.reps}; batch: states x qs(5), warm {a.warm}, p50 of {a.batch_reps}",
        "server": {
            k: health.get(k)
            for k in (
                "backend",
                "model",
                "revision",
                "precision",
                "seq_buckets",
                "row_buckets",
                "warm_shapes",
                "max_len",
                "head_max_len",
                "limits",
                "sanity",
            )
        },
        "state_repeats": a.repeats,
        "loadavg_start": loadavg(),
        "cells": {},
        "batch": {},
    }
    for n in [int(x) for x in a.cells.split(",") if x.strip()]:
        body = {"state": state, "questions": qs(n)}
        report["cells"][str(n)] = timed(s, a.base_url + "/v1/systemone", body, a.warm, a.reps)
        c = report["cells"][str(n)]
        print(
            f"LONG {n} questions: client p50 {c['client_ms_p50']:.1f} ms server {c['server_ms_p50']} device {c['device_ms_p50']} batch {c['batch_histogram']} tokens {c['input_tokens']} truncated {c['truncated']} errors {c['errors']}"
        )
    for b in [int(x) for x in a.batch_states.split(",") if x.strip()]:
        body = {"states": [state] * b, "questions": qs(5)}
        r = timed(s, a.base_url + "/v1/systemone/batch", body, a.warm, a.batch_reps)
        r["rows"] = 5 * b
        r["questions_per_s"] = (5 * b) / (r["client_ms_p50"] / 1000.0) if r["client_ms_p50"] else None
        report["batch"][f"{b}x5"] = r
        print(
            f"LONG batch {b} states x 5: client p50 {r['client_ms_p50']:.1f} ms device {r['device_ms_p50']} batch {r['batch_histogram']} q/s {r['questions_per_s']:.1f} errors {r['errors']}"
        )
    report["loadavg_end"] = loadavg()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print("LONG_DONE", a.out)


if __name__ == "__main__":
    main()

"""POST the reference records to a running Clef server and write the answers as parity rows.

Each output row has the CPU reference format of /home/hous/dev/clef/reports/reference/ref_text_bf16.jsonl
("id", "model", "answers", "usage", "probs", "input_tokens", "seconds", "_label" when the record
has one) plus "latency_ms" (the server's model time), "wall_ms" (client round trip), "status",
"cache_hit" (derived from the change of the prefix-cache hit counter in /v1/models between the
request and the previous one; exact only when nobody else sends) and "pass". "probs" is rebuilt from
the served "answers", so every probability carries the 4-decimal rounding of the release
systemone_answer. The file is a parity_compare.py candidate.

A second pass (--passes 2, the default) sends the same records in reverse order and records whether
the answers are byte-equal to the first pass and whether the state was a prefix-cache hit.

Usage (host, no device):
  python parity_remote.py --records R.jsonl --output OUT.jsonl [--records R2.jsonl --output OUT2.jsonl]
      [--base-url http://127.0.0.1:8008] [--passes 2] [--model clef] [--api-key KEY]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import read_jsonl, write_jsonl

REQUEST_KEYS = ("model", "state", "questions", "media_kwargs")


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), message, flush=True)


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def request_of(row, model):
    body = {key: row[key] for key in REQUEST_KEYS if key in row}
    body["model"] = model or body.get("model", "clef")
    if row.get("images"):
        body["images"] = [b64(p) if isinstance(p, str) else p for p in row["images"]]
    if row.get("videos"):
        body["videos"] = [[b64(f) if isinstance(f, str) else f for f in frames] for frames in row["videos"]]
    return body


def probs_of(answers):
    probs = {}
    for qid, answer in answers.items():
        kind = answer.get("type")
        if kind == "noul":
            p = float(answer["noul"])
            probs[qid] = {"true": p, "false": round(1.0 - p, 4)}
        else:
            probs[qid] = {str(k): float(v) for k, v in answer["probabilities"].items()}
    return probs


class Client:
    def __init__(self, base_url, api_key, timeout):
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def hits(self):
        card = self.http.get("/v1/models").json()["models"][0]
        return int(card["prefix_cache"]["hits"]), card

    def post(self, body):
        started = time.perf_counter()
        response = self.http.post("/v1/systemone", json=body)
        wall = (time.perf_counter() - started) * 1000
        return response, wall


def run_pass(client, rows, model, pass_index, first_answers):
    out = []
    order = range(len(rows)) if pass_index == 0 else reversed(range(len(rows)))
    hits_before, _ = client.hits()
    for index in order:
        row = rows[index]
        record_id = row.get("id", f"row{index}")
        result = {"id": record_id, "pass": pass_index}
        if "_label" in row:
            result["_label"] = row["_label"]
        response, wall = client.post(request_of(row, model))
        hits_after, _ = client.hits()
        result["status"] = response.status_code
        result["wall_ms"] = round(wall, 1)
        result["cache_hit"] = hits_after > hits_before
        hits_before = hits_after
        if response.status_code != 200:
            result["error"] = f"HTTP {response.status_code}: {response.text[:300]}"
            log(f"pass {pass_index} {index + 1}/{len(rows)} {record_id} FAILED {result['error']}")
            out.append(result)
            continue
        body = response.json()
        result.update(
            model=body["model"],
            answers=body["answers"],
            usage=body["usage"],
            probs=probs_of(body["answers"]),
            input_tokens=body["usage"]["input_tokens"],
            latency_ms=body["latency_ms"],
            seconds=round(wall / 1000, 3),
        )
        if pass_index == 0:
            first_answers[record_id] = body["answers"]
        else:
            result["equal_to_first_pass"] = body["answers"] == first_answers.get(record_id)
        log(
            f"pass {pass_index} {index + 1}/{len(rows)} {record_id} tokens={result['input_tokens']} "
            f"latency_ms={result['latency_ms']} wall_ms={result['wall_ms']} hit={result['cache_hit']}"
            + ("" if pass_index == 0 else f" equal={result['equal_to_first_pass']}")
        )
        out.append(result)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", action="append", required=True)
    parser.add_argument("--output", action="append", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--model", default="clef")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()
    if len(args.records) != len(args.output):
        raise SystemExit("--records and --output must be given the same number of times")
    client = Client(args.base_url, args.api_key, args.timeout)
    _, card = client.hits()
    summary = {"base_url": args.base_url, "served": card, "files": {}}
    for records_path, output in zip(args.records, args.output):
        rows = read_jsonl(records_path)
        first_answers = {}
        passes = [run_pass(client, rows, args.model, p, first_answers) for p in range(args.passes)]
        write_jsonl(output, passes[0])
        if args.passes > 1:
            write_jsonl(output.replace(".jsonl", "_pass2.jsonl"), passes[1])
        ok = [r for r in passes[0] if "error" not in r]
        latencies = sorted(r["latency_ms"] for r in ok)
        entry = {
            "records": len(rows),
            "ok": len(ok),
            "failed": len(rows) - len(ok),
            "latency_ms_median": latencies[len(latencies) // 2] if latencies else None,
            "latency_ms_max": max(latencies) if latencies else None,
            "latency_ms_sum": round(sum(latencies), 1),
            "wall_ms_median": sorted(r["wall_ms"] for r in ok)[len(ok) // 2] if ok else None,
            "output": output,
        }
        if args.passes > 1:
            second = [r for r in passes[1] if "error" not in r]
            entry["pass2"] = {
                "answers_equal_to_first_pass": sum(int(r.get("equal_to_first_pass", False)) for r in second),
                "cache_hits": sum(int(r["cache_hit"]) for r in second),
                "latency_ms_median": sorted(r["latency_ms"] for r in second)[len(second) // 2] if second else None,
                "latency_ms_on_hits_median": (
                    sorted(r["latency_ms"] for r in second if r["cache_hit"])[
                        sum(int(r["cache_hit"]) for r in second) // 2
                    ]
                    if any(r["cache_hit"] for r in second)
                    else None
                ),
            }
        summary["files"][records_path] = entry
        log(f"SUMMARY {records_path}: {json.dumps(entry)}")
    _, summary["served_after"] = client.hits()
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=1))
    log("PARITY_REMOTE_DONE")


if __name__ == "__main__":
    main()

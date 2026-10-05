"""POST SystemOne JSONL records to a running Clef server and write the answers as reference-format rows.

Input rows are the records of /home/hous/dev/clef/evals/*.jsonl and /home/hous/dev/clef/reports/reference/*.jsonl:
a SystemOne request body ("id", "model", "state", "questions", optional "images" as PNG paths, "videos" as
frame path lists) plus "_label" and "_source". Each output row has the CPU reference row format of
/home/hous/dev/clef/reports/reference/ref_text_bf16.jsonl ("id", "_label", "model", "answers", "usage",
"probs", "input_tokens", "correct", "seconds") plus "latency_ms" (the server's model time), "wall_ms"
(client round trip), "status" and "attempts". "probs" is rebuilt from the served "answers", so every
probability carries the 4-decimal rounding of the release systemone_answer. A record that fails after the
retries writes an "error" row. The output is a parity_compare.py candidate and an eval_metrics.py results file.

Resume: ids already present in the output with a 200 answer are skipped (--no-resume sends everything).
Rows are appended as they complete and the file is rewritten in input order at the end.
Retries: 408, 429, 5xx, connection errors and timeouts are retried with exponential backoff; any other 4xx is
recorded as an error row at once (a 422 is the server refusing the record).

Usage (host, no device):
  python eval_remote.py --records R.jsonl --output OUT.jsonl [--records R2.jsonl --output OUT2.jsonl]
      [--base-url http://127.0.0.1:8008] [--concurrency 4] [--retries 3] [--timeout 600] [--limit N]
      [--model clef] [--api-key KEY] [--summary S.json] [--no-resume] [--strict]
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import read_jsonl, write_jsonl

REQUEST_KEYS = ("model", "state", "questions", "media_kwargs")
RETRY_STATUSES = (408, 429)


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), message, flush=True)


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def request_of(row, model):
    body = {key: row[key] for key in REQUEST_KEYS if key in row}
    body["model"] = model or body.get("model", "clef")
    if row.get("images"):
        body["images"] = [
            b64(p) if isinstance(p, str) and not p.startswith(("http://", "https://", "data:")) else p
            for p in row["images"]
        ]
    if row.get("videos"):
        body["videos"] = [[b64(f) if isinstance(f, str) else f for f in frames] for frames in row["videos"]]
    return body


def probs_of(answers):
    probs = {}
    for qid, answer in answers.items():
        if answer.get("type") == "noul":
            p = float(answer["noul"])
            probs[qid] = {"true": p, "false": round(1.0 - p, 4)}
        else:
            probs[qid] = {str(k): float(v) for k, v in answer["probabilities"].items()}
    return probs


def correctness(questions, probs, labels):
    out = {}
    for qid, label in labels.items():
        dist = probs.get(qid)
        if dist is None:
            continue
        if questions[qid]["type"] == "noul":
            out[qid] = (dist["true"] >= 0.5) == bool(label)
        else:
            out[qid] = max(dist, key=dist.__getitem__) == str(label)
    return out


class Client:
    def __init__(self, base_url, api_key, timeout, retries):
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)
        self.retries = retries

    def card(self):
        return self.http.get("/v1/models").json()["models"][0]

    def post(self, body):
        last = None
        for attempt in range(1, self.retries + 1):
            started = time.perf_counter()
            try:
                response = self.http.post("/v1/systemone", json=body)
            except httpx.HTTPError as error:
                last = f"{type(error).__name__}: {error}"
            else:
                wall = (time.perf_counter() - started) * 1000
                if response.status_code == 200:
                    return response, wall, attempt, None
                if response.status_code in RETRY_STATUSES or response.status_code >= 500:
                    last = f"HTTP {response.status_code}: {response.text[:300]}"
                else:
                    return response, wall, attempt, f"HTTP {response.status_code}: {response.text[:300]}"
            if attempt < self.retries:
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
        return None, None, self.retries, f"failed after {self.retries} attempts: {last}"


def serve_one(client, row, index, model):
    record_id = row.get("id", f"row{index}")
    result = {"id": record_id}
    if "_label" in row:
        result["_label"] = row["_label"]
    started = time.perf_counter()
    response, wall, attempts, error = client.post(request_of(row, model))
    result["attempts"] = attempts
    result["seconds"] = round(time.perf_counter() - started, 3)
    if error is not None:
        result["status"] = response.status_code if response is not None else None
        result["error"] = error
        return result
    body = response.json()
    probs = probs_of(body["answers"])
    result.update(
        model=body["model"],
        answers=body["answers"],
        usage=body["usage"],
        probs=probs,
        input_tokens=body["usage"]["input_tokens"],
        latency_ms=body["latency_ms"],
        wall_ms=round(wall, 1),
        status=200,
    )
    if "_label" in row:
        result["correct"] = correctness(row["questions"], probs, row["_label"])
    return result


def run_file(client, records_path, output, model, concurrency, limit, resume):
    rows = read_jsonl(records_path)
    if limit:
        rows = rows[:limit]
    output = Path(output)
    done = {}
    if resume and output.exists():
        for r in read_jsonl(output):
            if "error" not in r:
                done[r["id"]] = r
    todo = [(i, r) for i, r in enumerate(rows) if r.get("id", f"row{i}") not in done]
    log(
        f"{records_path}: {len(rows)} records, {len(done)} already done, {len(todo)} to send, concurrency {concurrency}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    completed = 0
    started = time.perf_counter()
    with output.open("a", encoding="utf-8") as handle:

        def work(item):
            nonlocal completed
            index, row = item
            result = serve_one(client, row, index, model)
            with lock:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                done[result["id"]] = result
                completed += 1
                n = completed
            if "error" in result:
                log(f"{n}/{len(todo)} {result['id']} FAILED {result['error']}")
            elif n % 25 == 0 or n == len(todo):
                elapsed = time.perf_counter() - started
                log(
                    f"{n}/{len(todo)} {result['id']} tokens={result['input_tokens']} latency_ms={result['latency_ms']} "
                    f"wall_ms={result['wall_ms']} elapsed={elapsed:.0f}s rate={n / elapsed:.2f}/s"
                )
            return result

        if concurrency > 1 and len(todo) > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(work, todo))
        else:
            for item in todo:
                work(item)
    wall_s = time.perf_counter() - started
    ordered = [done[r.get("id", f"row{i}")] for i, r in enumerate(rows) if r.get("id", f"row{i}") in done]
    write_jsonl(output, ordered)
    ok = [r for r in ordered if "error" not in r]
    failed = len(rows) - len(ok)
    latencies = sorted(float(r["latency_ms"]) for r in ok)
    walls = sorted(float(r["wall_ms"]) for r in ok)
    tokens = sorted(int(r["input_tokens"]) for r in ok)
    entry = {
        "records": len(rows),
        "ok": len(ok),
        "failed": failed,
        "sent_this_run": len(todo),
        "run_wall_s": round(wall_s, 1),
        "records_per_s_this_run": round(len(todo) / wall_s, 3) if wall_s > 0 and todo else None,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "latency_ms_p95": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None,
        "latency_ms_max": max(latencies) if latencies else None,
        "latency_ms_sum": round(sum(latencies), 1),
        "wall_ms_median": statistics.median(walls) if walls else None,
        "tokens_median": statistics.median(tokens) if tokens else None,
        "tokens_max": max(tokens) if tokens else None,
        "errors": [{"id": r["id"], "error": r["error"]} for r in ordered if "error" in r][:20],
        "records_path": str(records_path),
        "output": str(output),
    }
    if any("correct" in r for r in ok):
        flags = [v for r in ok for v in (r.get("correct") or {}).values()]
        entry["labelled_questions"] = len(flags)
        entry["correct_questions"] = sum(flags)
        entry["accuracy"] = round(sum(flags) / len(flags), 4) if flags else None
    log(f"SUMMARY {records_path}: {json.dumps(entry)}")
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", action="append", required=True)
    parser.add_argument("--output", action="append", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="clef")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit 1 when any record failed")
    args = parser.parse_args()
    if len(args.records) != len(args.output):
        raise SystemExit("--records and --output must be given the same number of times")
    client = Client(args.base_url, args.api_key, args.timeout, args.retries)
    card = client.card()
    summary = {
        "base_url": args.base_url,
        "served": card,
        "concurrency": args.concurrency,
        "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "files": {},
    }
    log(
        f"server {args.base_url}: backend={card.get('backend')} device={card.get('device')} precision={card.get('precision')}"
    )
    failed = 0
    for records_path, output in zip(args.records, args.output):
        entry = run_file(client, records_path, output, args.model, args.concurrency, args.limit, not args.no_resume)
        summary["files"][records_path] = entry
        failed += entry["failed"]
    summary["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    summary["served_after"] = client.card()
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=1))
    log(f"EVAL_REMOTE_DONE failed={failed}")
    if args.strict and failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

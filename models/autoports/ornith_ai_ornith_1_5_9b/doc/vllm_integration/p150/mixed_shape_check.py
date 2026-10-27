# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mixed-shape trace reuse and request-lifecycle check against a live Ornith vLLM server.

Sends unseen logical prompt lengths, concurrent arrivals while other requests decode,
repeated lengths in new batch compositions and orders, sampling-mode switches, and a
prompt that crosses the internal 2048-token prefill chunk. Reads the generator's trace
capture/release events from the server log before and after the workload.
"""

import argparse
import bz2
import json
import re
import statistics
import threading
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

EVENT = re.compile(
    r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) \| INFO +\| .*Ornith trace (capture|release) (\d+): programs (\d+)"
)


def log_events(path):
    if path is None or not path.exists():
        return None
    events = []
    for line in path.read_text(errors="replace").splitlines():
        match = EVENT.search(line)
        if match:
            events.append(
                {
                    "time": match.group(1),
                    "kind": match.group(2),
                    "count": int(match.group(3)),
                    "programs": int(match.group(4)),
                    "line": line.strip()[:300],
                }
            )
    return events


def log_programs(path):
    if path is None or not path.exists():
        return None
    found = None
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(r"Ornith prefill bucket warmup: (\{.*\})", line)
        if match:
            found = json.loads(match.group(1))
    return found


class Client:
    def __init__(self, url, model, timeout=1800):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, ids, *, max_tokens, temperature=0.0, top_k=None, top_p=None, seed=None, label=""):
        body = {
            "model": self.model,
            "prompt": ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "ignore_eos": True,
            "stream": True,
            "return_token_ids": True,
        }
        if top_k is not None:
            body["top_k"] = top_k
        if top_p is not None:
            body["top_p"] = top_p
        if seed is not None:
            body["seed"] = seed
        start = time.perf_counter()
        chunk_times, text, token_ids = [], [], []
        status = None
        with requests.post(self.url + "/v1/completions", json=body, stream=True, timeout=self.timeout) as response:
            status = response.status_code
            if status != 200:
                return {"label": label, "status": status, "error": response.text[:500], "logical_length": len(ids)}
            for raw in response.iter_lines():
                if not raw:
                    continue
                line = raw.decode()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                choice = event["choices"][0]
                piece = choice.get("text", "")
                new_ids = choice.get("token_ids") or []
                if piece or new_ids:
                    chunk_times.append(time.perf_counter())
                    text.append(piece)
                    token_ids.extend(new_ids)
        end = time.perf_counter()
        intervals = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
        return {
            "label": label,
            "status": status,
            "logical_length": len(ids),
            "chunk_starts": list(range(0, len(ids), 2048)),
            "chunk_lengths": [min(2048, len(ids) - s) for s in range(0, len(ids), 2048)],
            "temperature": temperature,
            "top_k": top_k,
            "seed": seed,
            "max_tokens": max_tokens,
            "ttft_s": (chunk_times[0] - start) if chunk_times else None,
            "itl_max_s": max(intervals) if intervals else None,
            "itl_median_s": statistics.median(intervals) if intervals else None,
            "total_s": end - start,
            "completion_tokens": len(token_ids),
            "token_ids": token_ids,
            "text": "".join(text),
        }


def run_concurrent(client, jobs):
    results = [None] * len(jobs)

    def work(index, job):
        results[index] = client.complete(**job)

    threads = [threading.Thread(target=work, args=(i, job)) for i, job in enumerate(jobs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="/home/hous/dev/ornith-1.5-9b/upstream")
    parser.add_argument("--tokenizer", default="/home/hous/dev/ornith-1.5-9b/upstream")
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument("--text", type=Path, default=Path("models/tt_transformers/tests/tale-of-two-cities.txt.bz2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-length", type=int, default=4353)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    text = bz2.open(args.text, "rt").read() if args.text.suffix == ".bz2" else args.text.read_text()
    book = tokenizer(text, add_special_tokens=False)["input_ids"]

    def prompt(length, offset):
        ids = book[offset : offset + length]
        assert len(ids) == length
        return ids

    client = Client(args.url, args.model)
    report = {
        "server_url": args.url,
        "server_log": str(args.server_log) if args.server_log else None,
        "prefill_warmup": log_programs(args.server_log),
        "events_before": log_events(args.server_log),
        "phases": [],
    }
    greedy_by_key = {}

    def record(phase, results):
        report["phases"].append({"phase": phase, "results": results})
        for row in results:
            print(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in ("label", "status", "logical_length", "ttft_s", "itl_max_s", "completion_tokens")
                    }
                ),
                flush=True,
            )
            if row.get("status") == 200 and row["temperature"] == 0.0:
                key = (row["logical_length"], row["label"].split(":")[0])
                greedy_by_key.setdefault(key, []).append(row["token_ids"])
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    # Phase 1: unseen logical lengths, one at a time, first encounters.
    singles = [129, 130, 131, 64, 200, 2305, 1, 2, 5, 1000, 1024, 1025]
    results = [client.complete(prompt(n, 1000 + n), max_tokens=16, label=f"single{n}") for n in singles]
    record("1_single_first_encounters", results)
    # Phase 2: concurrent arrivals while a request decodes; unseen lengths and new combinations.
    jobs = [dict(ids=prompt(131, 5000), max_tokens=192, label="decoding131")]
    jobs += [dict(ids=prompt(n, 6000 + n), max_tokens=24, label=f"arrival{n}") for n in (65, 131, 2305, 129, 33, 257)]
    record("2_concurrent_arrivals_during_decode", run_concurrent(client, jobs))
    # Phase 3: repeated lengths in new batch compositions and row orders; identical greedy prompts must match phase 1.
    repeat_a = [dict(ids=prompt(n, 1000 + n), max_tokens=16, label=f"single{n}:batchA") for n in (131, 65, 129, 2305)]
    record("3a_repeat_batch", run_concurrent(client, repeat_a))
    repeat_b = [
        dict(ids=prompt(n, 1000 + n), max_tokens=16, label=f"single{n}:batchB") for n in (2305, 129, 1, 131, 200, 130)
    ]
    record("3b_repeat_batch_reordered", run_concurrent(client, repeat_b))
    # Phase 4: sampling-mode switches, then return to the greedy key.
    sampled = [
        dict(ids=prompt(n, 1000 + n), max_tokens=16, temperature=0.8, top_k=20, top_p=0.95, seed=7, label=f"sampled{n}")
        for n in (131, 129)
    ]
    record("4a_sampled", run_concurrent(client, sampled))
    back = [dict(ids=prompt(n, 1000 + n), max_tokens=16, label=f"single{n}:after_sampled") for n in (131, 129, 130)]
    record("4b_greedy_after_sampled", run_concurrent(client, back))
    # Phase 5: a prompt that crosses the 2048-token chunk twice (first chunk, full continuation chunk, partial final chunk) while others decode.
    jobs = [dict(ids=prompt(args.long_length, 20000), max_tokens=16, label=f"long{args.long_length}")]
    jobs += [dict(ids=prompt(n, 7000 + n), max_tokens=48, label=f"beside{n}") for n in (131, 65)]
    record("5_long_prompt_crossing_chunks", run_concurrent(client, jobs))
    # Phase 6: repeat the very first encounters once more, alone.
    record(
        "6_repeat_singles",
        [client.complete(prompt(n, 1000 + n), max_tokens=16, label=f"single{n}:final") for n in (129, 130, 131, 2305)],
    )

    # Verdict data.
    time.sleep(2)
    report["events_after"] = log_events(args.server_log)
    before = len(report["events_before"] or [])
    after = report["events_after"]
    report["events_during_workload"] = after[before:] if after is not None else None
    rows = [row for phase in report["phases"] for row in phase["results"]]
    ttfts = sorted(row["ttft_s"] for row in rows if row.get("ttft_s") is not None)

    def pct(p):
        return ttfts[min(len(ttfts) - 1, int(round((p / 100) * (len(ttfts) - 1))))] if ttfts else None

    mismatches = []
    for (length, base), outputs in greedy_by_key.items():
        for other in outputs[1:]:
            if other[:16] != outputs[0][:16]:
                mismatches.append({"length": length, "label": base, "first": outputs[0][:16], "other": other[:16]})
    report["summary"] = {
        "requests": len(rows),
        "completed": sum(
            1 for row in rows if row.get("status") == 200 and row.get("completion_tokens") == row.get("max_tokens")
        ),
        "ttft_p50_s": pct(50),
        "ttft_p95_s": pct(95),
        "ttft_p99_s": pct(99),
        "ttft_max_s": ttfts[-1] if ttfts else None,
        "itl_max_s": max((row["itl_max_s"] for row in rows if row.get("itl_max_s") is not None), default=None),
        "output_tokens": sum(row.get("completion_tokens", 0) for row in rows),
        "greedy_repeat_mismatches": mismatches,
        "captures_during_workload": sum(1 for e in (report["events_during_workload"] or []) if e["kind"] == "capture"),
        "releases_during_workload": sum(1 for e in (report["events_during_workload"] or []) if e["kind"] == "release"),
        "events_available": report["events_after"] is not None,
    }
    report["summary"]["status"] = (
        "pass"
        if report["summary"]["completed"] == len(rows)
        and not mismatches
        and report["summary"]["events_available"]
        and report["summary"]["captures_during_workload"] == 0
        and report["summary"]["releases_during_workload"] == 0
        else "fail"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

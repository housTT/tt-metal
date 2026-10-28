# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Send a 32 x 128-token burst and record wall-clock send/first-token/last-token times per request."""

import argparse
import bz2
import json
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="ornith-ai/Ornith-1.5-9B")
    parser.add_argument("--tokenizer", default="/home/hous/dev/ornith-1.5-9b/upstream")
    parser.add_argument("--text", type=Path, default=Path("models/tt_transformers/tests/tale-of-two-cities.txt.bz2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    book = tokenizer(bz2.open(args.text, "rt").read()[:300000], add_special_tokens=False)["input_ids"]
    prompts = [book[3000 + i * 131 : 3000 + i * 131 + args.length] for i in range(args.requests)]
    results = [None] * args.requests

    def work(index):
        body = {
            "model": args.model,
            "prompt": prompts[index],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
        }
        send = time.time()
        first = last = None
        tokens = 0
        with requests.post(args.url + "/v1/completions", json=body, stream=True, timeout=600) as response:
            for raw in response.iter_lines():
                if not raw or not raw.startswith(b"data:") or raw.strip() == b"data: [DONE]":
                    continue
                now = time.time()
                first = first or now
                last = now
                tokens += 1
        results[index] = {"index": index, "send": send, "first": first, "last": last, "tokens": tokens}

    threads = [threading.Thread(target=work, args=(i,)) for i in range(args.requests)]
    start = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    ttft = [r["first"] - r["send"] for r in results]
    stamp = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    summary = {
        "start_utc": stamp(start),
        "ttft_sorted_s": sorted(round(v, 3) for v in ttft),
        "ttft_median_s": statistics.median(ttft),
        "first_token_utc_sorted": sorted(stamp(r["first"]) for r in results),
        "send_utc_min_max": [stamp(min(r["send"] for r in results)), stamp(max(r["send"] for r in results))],
        "last_token_utc_max": stamp(max(r["last"] for r in results)),
        "requests": results,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "requests"}, indent=1))


if __name__ == "__main__":
    main()

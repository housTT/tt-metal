# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Serving-level check that a 32-slot batched prefill group equals solo prefill per row.

Sends 32 short chat prompts at once, then sends every prompt alone. The engine admits the first
arrival in a prefill step of its own, so a client burst of N forms a group of at most N - 1 rows; the
check requires a group of at least N - 1 rows padded to the 32-slot variant. Compares the first greedy
token per row (prefill equivalence) and records the 8-token greedy continuations, which may differ
between the batch-1 and 32-slot decode lanes. Reads the server log for the ``prefill_groups`` event of
the burst.

Usage:
    python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_vllm_batched_group_equivalence \
        --server-url http://127.0.0.1:8000 --hf-model google/gemma-4-26B-A4B-it \
        --server-log <server.log> --output <json>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

EVENT_PREFIX = "GEMMA4_TRACE_EVENT "
TOPICS = [
    "the ocean",
    "a red bicycle",
    "winter mornings",
    "the number seven",
    "a quiet library",
    "fresh bread",
    "a mountain trail",
    "an old clock",
    "city traffic at dawn",
    "a paper boat",
    "the smell of rain",
    "a lighthouse keeper",
    "chess openings",
    "a lost key",
    "the color blue",
    "a garden in spring",
    "a long train ride",
    "a cup of tea",
    "a desert night",
    "a violin",
    "a snowball fight",
    "a kite",
    "the first day of school",
    "a river delta",
    "a lantern",
    "an empty stadium",
    "a honeybee",
    "a sailor's knot",
    "a wooden bridge",
    "a telescope",
    "a rainy market",
]


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _events_since(path: Path, first_line: int) -> list[dict[str, Any]]:
    events = []
    with path.open("r", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index < first_line or EVENT_PREFIX not in line:
                continue
            try:
                events.append(json.loads(line.split(EVENT_PREFIX, 1)[1].strip()))
            except json.JSONDecodeError:
                continue
    return events


def _chat(server_url: str, hf_model: str, content: str, max_tokens: int) -> dict[str, Any]:
    began = time.perf_counter()
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": hf_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
        timeout=1800,
    )
    payload = response.json()
    text = payload["choices"][0]["message"].get("content") or ""
    return {"status": response.status_code, "text": text, "elapsed_ms": (time.perf_counter() - began) * 1000.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--padded-batch", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    prompts = []
    for index in range(args.rows):
        topic = TOPICS[index % len(TOPICS)]
        extra = " Keep it short." * (index % 3)
        prompts.append(f"Write one sentence about {topic}.{extra}")
    prompt_tokens = [
        len(
            tokenizer(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
                ),
                add_special_tokens=False,
            )["input_ids"]
        )
        for p in prompts
    ]
    log_start = _count_lines(args.server_log)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.rows) as pool:
        burst = list(pool.map(lambda p: _chat(args.server_url, args.hf_model, p, args.max_tokens), prompts))
    burst_events = _events_since(args.server_log, log_start)
    log_mid = _count_lines(args.server_log)
    solo = [_chat(args.server_url, args.hf_model, p, args.max_tokens) for p in prompts]
    solo_events = _events_since(args.server_log, log_mid)

    def first_token(text: str) -> int | None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if ids else None

    rows = []
    for index, (b, s) in enumerate(zip(burst, solo)):
        b_ids = tokenizer.encode(b["text"], add_special_tokens=False)
        s_ids = tokenizer.encode(s["text"], add_special_tokens=False)
        common = 0
        for x, y in zip(b_ids, s_ids):
            if x != y:
                break
            common += 1
        rows.append(
            {
                "row": index,
                "prompt_tokens": prompt_tokens[index],
                "burst_text": b["text"],
                "solo_text": s["text"],
                "first_token_equal": first_token(b["text"]) == first_token(s["text"]),
                "common_prefix_tokens": common,
                "burst_status": b["status"],
                "solo_status": s["status"],
            }
        )
    groups = [e for e in burst_events if e.get("event") == "prefill_groups" and e.get("groups")]
    group_shapes = [tuple(g) for e in groups for g in e["groups"]]
    largest_group = max(group_shapes, key=lambda g: len(g[0]), default=None)
    largest = len(largest_group[0]) if largest_group else 0
    largest_padded = largest_group[2] if largest_group else 0
    record = {
        "rows": args.rows,
        "max_tokens": args.max_tokens,
        "total_prompt_tokens": sum(prompt_tokens),
        "burst_prefill_groups": [e["groups"] for e in groups],
        "largest_group_rows": largest,
        "largest_group_padded_batch": largest_padded,
        "required_group_rows": args.rows - 1,
        "burst_releases": [e for e in burst_events if e.get("event") == "release"],
        "solo_releases": [e for e in solo_events if e.get("event") == "release"],
        "all_first_tokens_equal": all(r["first_token_equal"] for r in rows),
        "rows_identical_through_8": sum(1 for r in rows if r["common_prefix_tokens"] >= min(8, args.max_tokens)),
        "all_completed": all(r["burst_status"] == 200 and r["solo_status"] == 200 for r in rows),
        "per_row": rows,
    }
    record["status"] = (
        "pass"
        if record["all_first_tokens_equal"]
        and record["all_completed"]
        and largest >= args.rows - 1
        and largest_padded == args.padded_batch
        else "fail"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"batched group equivalence: {record['status']} largest_group_rows={largest} padded_batch={largest_padded} "
        f"first_tokens_equal={record['all_first_tokens_equal']} identical_through_8={record['rows_identical_through_8']}/{args.rows} -> {args.output}"
    )
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

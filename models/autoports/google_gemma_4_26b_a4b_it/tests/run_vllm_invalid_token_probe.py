# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Serving-level probe for out-of-vocabulary sampled token ids.

Sends the sampling suite's high-temperature workload ("Random letter: ", temperature 5, 10 tokens)
in several arrangements against a running server and counts the detokenizer's
``Encountered invalid token id`` lines that the server log gains during each phase, together with
the trace events of that window. Scenario ``arrangements``: concurrency 1 (single-row prefill, batch-1
decode lane, no slot remap), concurrency 2, and the suite's batch sizes 7, 10, 19, 32, 37 with every
request running to ``max_tokens``, plus the same batch sizes at temperature 1 with top_k 50 (the
suite's other sampled profile). Scenario ``endings``: the suite's own request shape (EOS allowed, so
rows leave the batch while others decode) and staggered request lengths with EOS ignored, at
temperature 5, at temperature 1 with top_k 50, and greedy.

Usage:
    python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_vllm_invalid_token_probe \
        --server-url http://127.0.0.1:8000 --hf-model google/gemma-4-26B-A4B-it \
        --server-log <server.log> --output <json>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

EVENT_PREFIX = "GEMMA4_TRACE_EVENT "
INVALID_RE = re.compile(r"Encountered invalid token id: (-?\d+)")


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _window(path: Path, first_line: int) -> tuple[list[int], dict[str, int]]:
    invalid: list[int] = []
    events: dict[str, int] = {}
    with path.open("r", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index < first_line:
                continue
            match = INVALID_RE.search(line)
            if match:
                invalid.append(int(match.group(1)))
            if EVENT_PREFIX in line:
                try:
                    event = json.loads(line.split(EVENT_PREFIX, 1)[1].strip())
                except json.JSONDecodeError:
                    continue
                name = event.get("event")
                if name == "release":
                    name = f"release:{event.get('reason')}"
                elif name == "capture":
                    name = f"capture:b{event.get('batch')}:{'greedy' if event.get('greedy') else 'sampled'}"
                elif name == "prefill_groups":
                    name = "prefill_groups:batched" if event.get("groups") else "prefill_groups:single"
                events[name] = events.get(name, 0) + 1
    return invalid, events


def _complete(
    server_url: str, hf_model: str, prompt: str, max_tokens: int, sampling: dict[str, Any], ignore_eos: bool = True
) -> dict[str, Any]:
    began = time.perf_counter()
    response = requests.post(
        f"{server_url}/v1/completions",
        json={"model": hf_model, "prompt": prompt, "max_tokens": max_tokens, "ignore_eos": ignore_eos, **sampling},
        timeout=600,
    )
    payload = response.json()
    choice = payload["choices"][0] if response.status_code == 200 else {}
    return {
        "status": response.status_code,
        "text": choice.get("text", ""),
        "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
        "elapsed_ms": (time.perf_counter() - began) * 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Random letter: ")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--scenario",
        choices=("arrangements", "endings"),
        default="arrangements",
        help="arrangements: the suite's batch sizes with every request running to max_tokens; "
        "endings: the suite's own request shape (EOS allowed) and staggered request lengths, "
        "so rows leave the batch while others still decode",
    )
    parser.add_argument("--top-k", type=int, default=None, help="explicit top_k for the temperature-5 requests")
    args = parser.parse_args()
    hot = {"temperature": 5} if args.top_k is None else {"temperature": 5, "top_k": args.top_k}
    cool = {"temperature": 1.0, "top_k": 50}
    greedy = {"temperature": 0.0}
    if args.scenario == "arrangements":
        phases = [("c1_t5", 1, hot, 40, True, False), ("c2_t5", 2, hot, 10, True, False)]
        phases += [(f"b{b}_t5", b, hot, args.rounds, True, False) for b in (7, 10, 19, 32, 37)]
        phases += [(f"b{b}_t1k50", b, cool, args.rounds, True, False) for b in (7, 19, 37)]
    else:
        phases = [
            ("b32_t5_eos", 32, hot, args.rounds, False, False),
            ("b37_t5_eos", 37, hot, args.rounds, False, False),
            ("b32_t5_stagger", 32, hot, args.rounds, True, True),
            ("b32_t1k50_stagger", 32, cool, args.rounds, True, True),
            ("b32_greedy_stagger", 32, greedy, args.rounds, True, True),
            ("b7_t5_eos", 7, hot, args.rounds, False, False),
        ]
    record: dict[str, Any] = {
        "server_url": args.server_url,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "scenario": args.scenario,
        "phases": [],
    }
    for name, concurrency, sampling, rounds, ignore_eos, stagger in phases:
        first_line = _count_lines(args.server_log)
        responses: list[dict[str, Any]] = []
        began = time.perf_counter()
        for _ in range(rounds):
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                responses += list(
                    pool.map(
                        lambda i: _complete(
                            args.server_url,
                            args.hf_model,
                            args.prompt,
                            (3 + (i % 10)) if stagger else args.max_tokens,
                            sampling,
                            ignore_eos,
                        ),
                        range(concurrency),
                    )
                )
        time.sleep(1.0)
        invalid, events = _window(args.server_log, first_line)
        samples = sum(r["completion_tokens"] or 0 for r in responses)
        phase = {
            "phase": name,
            "concurrency": concurrency,
            "rounds": rounds,
            "sampling": sampling,
            "ignore_eos": ignore_eos,
            "staggered_lengths": stagger,
            "requests": len(responses),
            "completed": sum(1 for r in responses if r["status"] == 200),
            "sampled_tokens": samples,
            "invalid_token_ids": invalid,
            "invalid_count": len(invalid),
            "invalid_rate": (len(invalid) / samples) if samples else None,
            "trace_events": events,
            "elapsed_s": time.perf_counter() - began,
        }
        record["phases"].append(phase)
        print(json.dumps({k: v for k, v in phase.items() if k != "invalid_token_ids"}), flush=True)
    record["total_invalid"] = sum(p["invalid_count"] for p in record["phases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"invalid token probe: total_invalid={record['total_invalid']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

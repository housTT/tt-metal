# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mixed-shape trace reuse check against a live Gemma 4 vLLM server.

Sends unseen prompt lengths, chunked long prompts while short requests decode,
a burst of eight short prompts while a long request decodes (so a batched
prefill group forms next to live decode rows), repeated lengths in changed
batch compositions, sampling-mode switches, and a prefix-cache resume. Then it reads the generator's ``GEMMA4_TRACE_EVENT`` lines
from the server log for the request window and maps captures, releases,
program-cache growth, prefill chunk starts, and decode shapes to the traffic.

Verdict: pass when every request completes, no decode trace is captured or
released during the workload, greedy repeats of one prompt are identical
across batch compositions, and no output is mechanically degenerate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

EVENT_PREFIX = "GEMMA4_TRACE_EVENT "


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _events_since(path: Path, first_line: int) -> list[dict[str, Any]]:
    events = []
    with path.open("r", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index < first_line or EVENT_PREFIX not in line:
                continue
            payload = line.split(EVENT_PREFIX, 1)[1].strip()
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
    return events


def _token_prompt(tokenizer: Any, length: int, seed: int) -> list[int]:
    corpus = (
        "The history of science is the study of the development of science and scientific knowledge, "
        "including both the natural and social sciences. Science is a body of empirical, theoretical, "
        "and practical knowledge about the natural world, produced by scientists who emphasize the "
        "observation, explanation, and prediction of real-world phenomena. "
    )
    ids = tokenizer(corpus * (length // 40 + 2), add_special_tokens=False)["input_ids"]
    offset = seed % 17
    body = ids[offset : offset + length - 1]
    return [tokenizer.bos_token_id] + body if tokenizer.bos_token_id is not None else ids[offset : offset + length]


def _completion(
    *, server_url: str, hf_model: str, prompt_ids: list[int], max_tokens: int, temperature: float, top_k: int
) -> dict[str, Any]:
    payload = {
        "model": hf_model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "ignore_eos": True,
    }
    if temperature > 0:
        payload["top_k"] = top_k
    began = time.perf_counter()
    first = None
    text_parts: list[str] = []
    with requests.post(f"{server_url}/v1/completions", json=payload, stream=True, timeout=1800) as response:
        status = response.status_code
        for raw in response.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            choice = chunk["choices"][0]
            if choice.get("text"):
                if first is None:
                    first = time.perf_counter()
                text_parts.append(choice["text"])
    finished = time.perf_counter()
    text = "".join(text_parts)
    return {
        "status": status,
        "prompt_len": len(prompt_ids),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "ttft_ms": None if first is None else (first - began) * 1000.0,
        "e2e_ms": (finished - began) * 1000.0,
        "text": text,
    }


def _degenerate(text: str) -> bool:
    words = text.split()
    if len(words) < 6:
        return False
    adjacent = sum(1 for a, b in zip(words, words[1:]) if a == b) / max(1, len(words) - 1)
    return adjacent > 0.5 or len(set(words)) <= 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=24)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    log_start = _count_lines(args.server_log)
    baseline_events = _events_since(args.server_log, 0)
    baseline_captures = [e for e in baseline_events if e.get("event") == "capture"]
    requests_log: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []

    def run(label: str, length: int, seed: int, *, max_tokens: int | None = None, temperature: float = 0.0):
        record = _completion(
            server_url=args.server_url,
            hf_model=args.hf_model,
            prompt_ids=_token_prompt(tokenizer, length, seed),
            max_tokens=max_tokens or args.max_tokens,
            temperature=temperature,
            top_k=32,
        )
        record["label"] = label
        record["seed"] = seed
        requests_log.append(record)
        return record

    def run_concurrent(label: str, specs: list[tuple[str, int, int, int, float]]):
        began = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures = [
                pool.submit(run, f"{label}:{name}", length, seed, max_tokens=max_tokens, temperature=temp)
                for name, length, seed, max_tokens, temp in specs
            ]
            results = [future.result() for future in futures]
        phases.append({"phase": label, "requests": len(results), "elapsed_s": time.perf_counter() - began})
        return results

    for length in (129, 130, 131, 257):
        run(f"unseen_{length}", length, seed=length)
    phases.append({"phase": "unseen_lengths", "requests": 4})

    run_concurrent(
        "chunked_while_decoding",
        [
            ("decode_a", 64, 1, 192, 0.0),
            ("decode_b", 33, 2, 192, 0.0),
            ("decode_c", 129, 3, 160, 0.0),
            ("decode_d", 300, 4, 160, 0.0),
            ("chunked_2200", 2200, 5, 16, 0.0),
            ("chunked_3100", 3100, 6, 16, 0.0),
            ("chunked_4500", 4500, 7, 16, 0.0),
        ],
    )

    began = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
        long_decode = pool.submit(run, "batched_group_while_decoding:decode_long", 64, 11, max_tokens=160)
        time.sleep(1.0)
        short_futures = [
            pool.submit(
                run, f"batched_group_while_decoding:short_{index}", 40 + index, 20 + index, max_tokens=args.max_tokens
            )
            for index in range(8)
        ]
        long_decode.result()
        for future in short_futures:
            future.result()
    phases.append({"phase": "batched_group_while_decoding", "requests": 9, "elapsed_s": time.perf_counter() - began})

    greedy_repeat_a = run("repeat_129_solo", 129, seed=129)
    run_concurrent(
        "repeat_mixed_order_1",
        [
            ("r129", 129, 129, args.max_tokens, 0.0),
            ("r33", 33, 2, args.max_tokens, 0.0),
            ("r257", 257, 257, args.max_tokens, 0.0),
        ],
    )
    run_concurrent(
        "sampled_switch",
        [
            ("s129", 129, 129, args.max_tokens, 0.8),
            ("s300", 300, 4, args.max_tokens, 0.8),
            ("s33", 33, 2, args.max_tokens, 0.8),
        ],
    )
    run_concurrent(
        "repeat_mixed_order_2",
        [
            ("r257", 257, 257, args.max_tokens, 0.0),
            ("r129", 129, 129, args.max_tokens, 0.0),
            ("r130", 130, 130, args.max_tokens, 0.0),
        ],
    )
    greedy_repeat_b = run("repeat_129_solo_again", 129, seed=129)

    resume = run("prefix_resume_2200", 2200, seed=5, max_tokens=16)
    phases.append({"phase": "prefix_resume", "requests": 1})

    events = _events_since(args.server_log, log_start)
    captures = [e for e in events if e.get("event") == "capture"]
    releases = [e for e in events if e.get("event") == "release"]
    growth = [e for e in events if e.get("event") == "program_cache_growth"]
    prefills = [e for e in events if e.get("event") == "prefill"]
    shapes = [e for e in events if e.get("event") == "decode_shape"]
    groups = [e for e in events if e.get("event") == "prefill_groups"]

    repeat_129 = [
        r
        for r in requests_log
        if r["prompt_len"] == 129 and r["temperature"] == 0.0 and r["max_tokens"] == args.max_tokens
    ]
    repeat_texts = {r["text"] for r in repeat_129}
    ttfts = [r["ttft_ms"] for r in requests_log if r["ttft_ms"] is not None]
    ttfts_sorted = sorted(ttfts)

    def percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, int(round(q * (len(values) - 1))))
        return values[index]

    reasons = []
    if any(r["status"] != 200 for r in requests_log):
        reasons.append("a request did not return HTTP 200")
    if captures:
        reasons.append(f"{len(captures)} decode trace capture(s) during the workload")
    if releases:
        reasons.append(f"{len(releases)} decode trace release(s) during the workload")
    if len(repeat_texts) != 1:
        reasons.append(f"greedy 129-token prompt produced {len(repeat_texts)} distinct texts across batch compositions")
    if any(_degenerate(r["text"]) for r in requests_log):
        reasons.append("a completion is mechanically degenerate")
    if not any(len(e.get("starts", [])) and any(s > 0 for s in e["starts"]) for e in prefills):
        reasons.append("no prefill call with a nonzero start position was recorded")
    if not any(e.get("decode_rows_active_before", 0) > 0 for e in prefills):
        reasons.append("no prefill call was recorded while other rows were decoding")

    record = {
        "server_url": args.server_url,
        "hf_model": args.hf_model,
        "server_log": str(args.server_log),
        "log_window": {"first_line": log_start, "events": len(events)},
        "baseline": {
            "captures_before_workload": [
                {k: v for k, v in e.items() if k in ("batch", "mode", "greedy", "model_trace_id", "sampling_trace_id")}
                for e in baseline_captures
            ],
            "releases_before_workload": len([e for e in baseline_events if e.get("event") == "release"]),
        },
        "workload": {
            "requests": len(requests_log),
            "phases": phases,
            "prompt_lengths": sorted({r["prompt_len"] for r in requests_log}),
            "completed": sum(1 for r in requests_log if r["status"] == 200),
            "ttft_ms": {
                "p50": statistics.median(ttfts) if ttfts else None,
                "p95": percentile(ttfts_sorted, 0.95),
                "p99": percentile(ttfts_sorted, 0.99),
                "max": max(ttfts) if ttfts else None,
            },
            "completed_request_throughput_per_s": len(requests_log)
            / max(1e-9, sum(p.get("elapsed_s", 0.0) for p in phases) or 1.0),
        },
        "trace_events": {
            "captures": captures,
            "releases": releases,
            "program_cache_growth": growth,
            "decode_shape_transitions": shapes,
            "prefill_calls": prefills,
            "prefill_groups": groups,
            "batched_prefill_calls": sum(1 for e in groups if e.get("groups")),
        },
        "greedy_repeat_129": {"texts": sorted(repeat_texts), "requests": len(repeat_129)},
        "prefix_resume": resume,
        "requests_log": requests_log,
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"mixed-shape check: {record['status']} -> {args.output}")
    for reason in reasons:
        print(f"  {reason}")
    print(
        f"  requests={len(requests_log)} captures={len(captures)} releases={len(releases)} "
        f"program_cache_growth_events={len(growth)} prefill_calls={len(prefills)} "
        f"batched_prefill_calls={record['trace_events']['batched_prefill_calls']} decode_shape_transitions={len(shapes)}"
    )
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())

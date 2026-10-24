# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Latency of a sampled request under the server's default sampling recipe, alone and beside a greedy user.

Phase ``sampled_alone``: one 128-token prompt, 128 output tokens, no sampling parameters other than
``temperature`` (so the server's ``generation_config.json`` defaults apply), streamed; records TTFT,
per-token latency, and the trace events of the window. Phase ``greedy_alone``: the same prompt,
greedy. Phase ``greedy_with_sampled``: a greedy 128/128 request that starts first and a sampled
request that arrives one second later; records both and the per-token latency of the greedy user
before and after the sampled arrival.

Usage:
    python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_vllm_default_recipe_latency \
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


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _events_since(path: Path, first_line: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("r", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index < first_line or EVENT_PREFIX not in line:
                continue
            try:
                event = json.loads(line.split(EVENT_PREFIX, 1)[1].strip())
            except json.JSONDecodeError:
                continue
            name = event.get("event")
            if name == "release":
                name = f"release:{event.get('reason')}"
            elif name == "capture":
                name = f"capture:b{event.get('batch')}:{'greedy' if event.get('greedy') else 'sampled'}"
            counts[name] = counts.get(name, 0) + 1
    return counts


def _stream(
    server_url: str, hf_model: str, prompt_ids: list[int], max_tokens: int, temperature: float
) -> dict[str, Any]:
    payload = {
        "model": hf_model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "ignore_eos": True,
    }
    began = time.perf_counter()
    arrivals: list[float] = []
    with requests.post(f"{server_url}/v1/completions", json=payload, stream=True, timeout=1800) as response:
        status = response.status_code
        for raw in response.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk["choices"][0].get("text"):
                arrivals.append(time.perf_counter())
    gaps = [(b - a) * 1000.0 for a, b in zip(arrivals, arrivals[1:])]
    return {
        "status": status,
        "temperature": temperature,
        "tokens": len(arrivals),
        "ttft_ms": (arrivals[0] - began) * 1000.0 if arrivals else None,
        "tpot_ms_mean": sum(gaps) / len(gaps) if gaps else None,
        "tpot_ms_first_half": sum(gaps[: len(gaps) // 2]) / max(1, len(gaps) // 2) if gaps else None,
        "tpot_ms_second_half": sum(gaps[len(gaps) // 2 :]) / max(1, len(gaps) - len(gaps) // 2) if gaps else None,
        "e2e_ms": (arrivals[-1] - began) * 1000.0 if arrivals else None,
        "started_at": began,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    text = "The history of computing is a story of ideas becoming machines. " * 40
    ids = tokenizer.encode(text, add_special_tokens=True)[: args.prompt_len]
    record: dict[str, Any] = {"prompt_len": len(ids), "max_tokens": args.max_tokens, "phases": {}}

    def phase(name: str, fn):
        first_line = _count_lines(args.server_log)
        result = fn()
        time.sleep(1.0)
        result["trace_events"] = _events_since(args.server_log, first_line)
        record["phases"][name] = result
        print(
            json.dumps({"phase": name, **{k: v for k, v in result.items() if k != "started_at"}}, default=str),
            flush=True,
        )

    phase("greedy_alone", lambda: _stream(args.server_url, args.hf_model, ids, args.max_tokens, 0.0))
    phase("sampled_alone_default_recipe", lambda: _stream(args.server_url, args.hf_model, ids, args.max_tokens, 1.0))
    phase("greedy_alone_again", lambda: _stream(args.server_url, args.hf_model, ids, args.max_tokens, 0.0))

    def concurrent_phase():
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            greedy = pool.submit(_stream, args.server_url, args.hf_model, ids, args.max_tokens, 0.0)
            time.sleep(1.0)
            sampled = pool.submit(_stream, args.server_url, args.hf_model, ids, args.max_tokens, 1.0)
            g, s = greedy.result(), sampled.result()
        return {
            "greedy": {k: v for k, v in g.items() if k != "started_at"},
            "sampled": {k: v for k, v in s.items() if k != "started_at"},
        }

    phase("greedy_with_sampled_arrival", concurrent_phase)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"default recipe latency -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

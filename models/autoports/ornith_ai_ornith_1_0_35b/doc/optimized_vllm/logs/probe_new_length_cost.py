# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What a *new prompt length* costs a running server, and what a repeat of it costs.

Every distinct logical prompt length compiles its own tail-chunk prefill programs, and those
programs' kernel binaries land on addresses the live decode trace writes - so the generator has to
re-capture the decode traces before the next replay
(``OrnithGenerator._ensure_traces_replay_safe``). This probe sends the same request twice at each of
several **non-aligned** lengths and records TTFT and the inter-token intervals for both, so the
first-at-a-length cost is separated from the steady state.

Prompts are sent as explicit token-id lists, which is the only way to control the logical length
exactly.

    python .../doc/optimized_vllm/logs/probe_new_length_cost.py --url http://localhost:8100 \
        --output .../doc/optimized_vllm/before/new_length_cost.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
from loguru import logger

MODEL = "ornith-ai/Ornith-1.0-35B"


def _one(url, tokens, max_tokens):
    """One streamed completion; returns TTFT and the inter-token intervals, in ms."""
    body = {
        "model": MODEL,
        "prompt": tokens,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    stamps = []
    start = time.perf_counter()
    with requests.post(f"{url}/v1/completions", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            if line == b"data: [DONE]":
                break
            stamps.append(time.perf_counter())
    if not stamps:
        raise RuntimeError("no streamed tokens")
    ttft = (stamps[0] - start) * 1000.0
    itls = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]
    return ttft, itls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8100")
    ap.add_argument("--lengths", default="211,347,613")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    lengths = [int(v) for v in args.lengths.split(",")]
    report = {"url": args.url, "max_tokens": args.max_tokens, "lengths": lengths, "rows": []}
    for length in lengths:
        tokens = [(1000 + (i * 7919) % 90000) for i in range(length)]
        row = {"prompt_len": length, "aligned_to_64": length % 64 == 0, "aligned_to_32": length % 32 == 0}
        for label in ("first", "repeat"):
            ttft, itls = _one(args.url, tokens, args.max_tokens)
            ordered = sorted(itls)
            row[label] = {
                "ttft_ms": ttft,
                "itl_max_ms": max(itls),
                "itl_median_ms": ordered[len(ordered) // 2],
                "itl_sum_ms": sum(itls),
                "intervals": len(itls),
            }
            logger.info(
                f"len {length:4d} {label:6s}: ttft {ttft:7.1f} ms, itl median "
                f"{row[label]['itl_median_ms']:6.2f} ms, itl max {row[label]['itl_max_ms']:7.1f} ms"
            )
        row["first_minus_repeat_ttft_ms"] = row["first"]["ttft_ms"] - row["repeat"]["ttft_ms"]
        row["first_minus_repeat_itl_sum_ms"] = row["first"]["itl_sum_ms"] - row["repeat"]["itl_sum_ms"]
        row["hidden_cost_ms"] = row["first_minus_repeat_ttft_ms"] + row["first_minus_repeat_itl_sum_ms"]
        report["rows"].append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    logger.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()

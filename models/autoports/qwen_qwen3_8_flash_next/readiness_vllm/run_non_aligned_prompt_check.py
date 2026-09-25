#!/usr/bin/env python3
"""Verify exact logical prompt lengths across internal page/tile boundaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8018")
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    endpoint = f"{args.server_url.rstrip('/')}/v1/completions"
    # page (64), 128-row tail, 512-row microchunk and 1,024-token vLLM chunk boundaries
    lengths = (1, 63, 64, 65, 67, 127, 129, 511, 512, 513, 1023, 1024, 1025, 1537)
    cases = []
    with httpx.Client(timeout=180.0) as client:
        for prompt_length in lengths:
            repeats = []
            payload = {
                "model": args.model,
                "prompt": [9707] * prompt_length,
                "max_tokens": 2,
                "temperature": 0.0,
                "ignore_eos": True,
            }
            for _ in range(2):
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                body = response.json()
                repeats.append(
                    {
                        "http_status": response.status_code,
                        "text": body["choices"][0]["text"],
                        "finish_reason": body["choices"][0]["finish_reason"],
                        "usage": body["usage"],
                    }
                )
            cases.append(
                {
                    "logical_prompt_tokens": prompt_length,
                    "repeats": repeats,
                    "exact_prompt_usage": all(row["usage"]["prompt_tokens"] == prompt_length for row in repeats),
                    "repeat_output_equal": repeats[0]["text"] == repeats[1]["text"],
                }
            )

    artifact = {
        "endpoint": endpoint,
        "model": args.model,
        "sampling_profile": {
            "path": "canonical on-device exact global argmax",
            "temperature": 0.0,
            "max_tokens": 2,
            "ignore_eos": True,
            "logprobs": False,
        },
        "internal_boundaries": {
            "attention_page_tokens": 64,
            # Model microchunk rows; the served default is 128 (QWEN38_PREFILL_CHUNK).
            "prefill_compute_chunk_tokens": int(os.getenv("QWEN38_PREFILL_CHUNK", "128")),
            "prefill_chunk_adaptive": os.getenv("QWEN38_PREFILL_CHUNK_ADAPTIVE", "0") == "1",
            "vllm_prefill_chunk_tokens": 1024,
        },
        "cases": cases,
    }
    artifact["verdict"] = (
        "pass" if all(row["exact_prompt_usage"] and row["repeat_output_equal"] for row in cases) else "fail"
    )
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    if artifact["verdict"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

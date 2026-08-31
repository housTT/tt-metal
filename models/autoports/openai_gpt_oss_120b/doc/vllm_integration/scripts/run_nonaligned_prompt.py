# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Send an exact-length, non-aligned token prompt to the live vLLM server."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from transformers import AutoTokenizer

MODEL_ID = "openai/gpt-oss-120b"


def main() -> None:
    script = Path(__file__).resolve()
    model_dir = script.parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=65)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=model_dir / "readiness_vllm" / "nonaligned_prompt_request.json",
    )
    args = parser.parse_args()

    if args.prompt_tokens <= 0 or args.prompt_tokens % 32 == 0 or args.prompt_tokens % 64 == 0:
        raise ValueError("prompt length must be positive and non-aligned to tile/page sizes")

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    seed_text = (
        "Explain why reliable systems validate boundary conditions with direct evidence. "
        "Keep the answer concise and practical. "
    )
    token_ids = tokenizer.encode(seed_text * 32, add_special_tokens=False)[: args.prompt_tokens]
    if len(token_ids) != args.prompt_tokens:
        raise RuntimeError(f"could only construct {len(token_ids)} prompt tokens")

    request = {
        "model": MODEL_ID,
        "prompt": token_ids,
        "max_tokens": args.output_tokens,
        "temperature": 0.0,
        "return_token_ids": True,
    }
    with httpx.Client(timeout=1800.0) as client:
        response = client.post(f"{args.server_url.rstrip('/')}/v1/completions", json=request)
    response.raise_for_status()
    body = response.json()
    usage = body.get("usage") or {}
    observed_prompt_tokens = usage.get("prompt_tokens")
    if observed_prompt_tokens != args.prompt_tokens:
        raise RuntimeError(f"server reported {observed_prompt_tokens} prompt tokens; expected {args.prompt_tokens}")

    artifact = {
        "model_id": MODEL_ID,
        "endpoint": "/v1/completions",
        "prompt_tokens": args.prompt_tokens,
        "output_tokens_requested": args.output_tokens,
        "alignment_checks": {
            "divisible_by_32": args.prompt_tokens % 32 == 0,
            "divisible_by_64": args.prompt_tokens % 64 == 0,
            "divisible_by_128": args.prompt_tokens % 128 == 0,
        },
        "request": request,
        "response": body,
        "http_status": response.status_code,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"PASS: HTTP {response.status_code}, prompt_tokens={observed_prompt_tokens}, "
        f"completion_tokens={usage.get('completion_tokens')}; wrote {args.output}"
    )


if __name__ == "__main__":
    main()

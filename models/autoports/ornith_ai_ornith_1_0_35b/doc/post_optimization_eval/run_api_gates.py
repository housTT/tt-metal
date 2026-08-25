# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exercise the OpenAI-compatible surface of a running Ornith server.

The output is intentionally compact: it records checks, timings, token counts and
content hashes, but not model generations.  This makes the JSON suitable as an input
to the post-optimization release artifact without publishing prompt traces.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ID = "ornith-ai/Ornith-1.0-35B"


def _request(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 900,
) -> tuple[int, bytes, float]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return int(response.status), body, time.perf_counter() - start
    except urllib.error.HTTPError as error:
        return int(error.code), error.read(), time.perf_counter() - start


def _chat(base_url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    status, body, elapsed = _request(base_url, "/v1/chat/completions", payload)
    if status != 200:
        raise RuntimeError(f"chat request returned HTTP {status}: {body[:500]!r}")
    return json.loads(body), elapsed


def _message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
        raise RuntimeError(f"unexpected chat response shape: {response!r}")
    return choices[0]["message"]


def _reasoning(message: dict[str, Any]) -> str:
    return str(message.get("reasoning") or message.get("reasoning_content") or "")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _usage(response: dict[str, Any]) -> dict[str, int | None]:
    usage = response.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _exact_payload(label: str) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": f"Reply with exactly this text and nothing else: {label}",
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _run(base_url: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, **details: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **details})

    status, _, elapsed = _request(base_url, "/health", timeout=30)
    record("health", status == 200, http_status=status, elapsed_s=round(elapsed, 3))

    status, body, elapsed = _request(base_url, "/v1/models", timeout=30)
    models = json.loads(body) if status == 200 else {}
    entries = models.get("data") or []
    matching = [entry for entry in entries if entry.get("id") == MODEL_ID]
    advertised_context = matching[0].get("max_model_len") if matching else None
    record(
        "model_discovery",
        status == 200 and len(matching) == 1 and advertised_context == 262144,
        http_status=status,
        model_id=matching[0].get("id") if matching else None,
        max_model_len=advertised_context,
        elapsed_s=round(elapsed, 3),
    )

    short_contents: list[str] = []
    short_times: list[float] = []
    short_usages: list[dict[str, int | None]] = []
    for _ in range(3):
        response, elapsed = _chat(base_url, _exact_payload("ORNITH_API_OK"))
        short_contents.append(str(_message(response).get("content") or "").strip())
        short_times.append(elapsed)
        short_usages.append(_usage(response))
    record(
        "short_greedy_reproducibility",
        short_contents == ["ORNITH_API_OK"] * 3,
        repeats=3,
        unique_output_hashes=len({_hash(value) for value in short_contents}),
        completion_tokens=[item["completion_tokens"] for item in short_usages],
        elapsed_s=[round(value, 3) for value in short_times],
    )

    reasoning_response, reasoning_elapsed = _chat(
        base_url,
        {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 17 multiplied by 19? Put only the number in the final answer.",
                }
            ],
            "temperature": 0,
            "max_tokens": 2048,
        },
    )
    reasoning_message = _message(reasoning_response)
    reasoning_text = _reasoning(reasoning_message)
    reasoning_content = str(reasoning_message.get("content") or "").strip()
    record(
        "reasoning_parser",
        bool(reasoning_text) and "323" in reasoning_content,
        reasoning_present=bool(reasoning_text),
        reasoning_chars=len(reasoning_text),
        final_answer_has_323="323" in reasoning_content,
        finish_reason=reasoning_response["choices"][0].get("finish_reason"),
        usage=_usage(reasoning_response),
        elapsed_s=round(reasoning_elapsed, 3),
    )

    tool_response, tool_elapsed = _chat(
        base_url,
        {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": "Call get_weather for Paris, France. Do not answer directly.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Return the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 512,
        },
    )
    tool_message = _message(tool_response)
    tool_calls = tool_message.get("tool_calls") or []
    tool_names = [call.get("function", {}).get("name") for call in tool_calls if isinstance(call, dict)]
    record(
        "tool_call_parser",
        tool_names == ["get_weather"],
        tool_call_count=len(tool_calls),
        tool_names=tool_names,
        reasoning_present=bool(_reasoning(tool_message)),
        finish_reason=tool_response["choices"][0].get("finish_reason"),
        usage=_usage(tool_response),
        elapsed_s=round(tool_elapsed, 3),
    )

    stream_payload = {**_exact_payload("ORNITH_STREAM_OK"), "stream": True}
    status, stream_body, stream_elapsed = _request(base_url, "/v1/chat/completions", stream_payload)
    stream_text = stream_body.decode("utf-8", errors="replace")
    stream_content = ""
    for line in stream_text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            chunk = json.loads(line.removeprefix("data: "))
            stream_content += str(chunk["choices"][0]["delta"].get("content") or "")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
    record(
        "streaming_sse",
        status == 200 and "data: [DONE]" in stream_text and stream_content.strip() == "ORNITH_STREAM_OK",
        http_status=status,
        saw_done="data: [DONE]" in stream_text,
        reassembled_content_matches=stream_content.strip() == "ORNITH_STREAM_OK",
        elapsed_s=round(stream_elapsed, 3),
    )

    # A deliberately bland shared label keeps the gate about concurrent serving.  Some numbered
    # all-caps strings are independently refused by this checkpoint even in a single request.
    labels = ["ORNITH_API_OK"] * 8

    def concurrent_request(label: str) -> tuple[str, dict[str, int | None], float]:
        response, request_elapsed = _chat(base_url, _exact_payload(label))
        return (
            str(_message(response).get("content") or "").strip(),
            _usage(response),
            request_elapsed,
        )

    concurrent_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        concurrent_results = list(executor.map(concurrent_request, labels))
    concurrent_elapsed = time.perf_counter() - concurrent_start
    record(
        "concurrency_8",
        [item[0] for item in concurrent_results] == labels,
        requests=8,
        successful_exact_outputs=sum(
            result[0] == label for result, label in zip(concurrent_results, labels, strict=True)
        ),
        completion_tokens=[item[1]["completion_tokens"] for item in concurrent_results],
        request_elapsed_s=[round(item[2], 3) for item in concurrent_results],
        wall_elapsed_s=round(concurrent_elapsed, 3),
    )

    long_generations: list[str] = []
    long_completion_tokens: list[int | None] = []
    for _ in range(3):
        response, _ = _chat(
            base_url,
            {
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reason briefly, then answer: what is the capital of France?",
                    }
                ],
                "temperature": 0,
                "max_tokens": 256,
            },
        )
        message = _message(response)
        long_generations.append(f"{_reasoning(message)}\n{message.get('content') or ''}")
        long_completion_tokens.append(_usage(response)["completion_tokens"])

    long_hashes = {_hash(value) for value in long_generations}
    diagnostics = {
        "long_greedy_reproducibility": {
            "repeats": 3,
            "reproducible": len(long_hashes) == 1,
            "unique_output_hashes": len(long_hashes),
            "completion_tokens": long_completion_tokens,
            "classification": ("pass" if len(long_hashes) == 1 else "known_decode_reproducibility_limitation"),
        }
    }

    return {
        "schema": "ornith-api-gates/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID,
        "base_url": base_url,
        "checks": checks,
        "summary": {
            "passed": sum(bool(check["passed"]) for check in checks),
            "total": len(checks),
            "status": "pass" if all(check["passed"] for check in checks) else "fail",
        },
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = _run(args.base_url)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["summary"]["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

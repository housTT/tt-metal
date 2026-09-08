# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Gemma 4 text, tool-call, and reasoning OpenAI API surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--expected-max-model-len", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    client = OpenAI(base_url=f"{args.server_url.rstrip('/')}/v1", api_key="dummy", timeout=600)
    model_list = client.models.list().model_dump()
    served = next(item for item in model_list["data"] if item["id"] == args.model)
    if served["max_model_len"] != args.expected_max_model_len:
        raise AssertionError(f"advertised max_model_len={served['max_model_len']}")

    text = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: nonaligned serving works. Extra padding words alpha beta gamma.",
            }
        ],
        max_tokens=32,
        temperature=0,
    ).model_dump()
    prompt_tokens = text["usage"]["prompt_tokens"]
    if prompt_tokens % 32 == 0 or prompt_tokens % 64 == 0:
        raise AssertionError(f"prompt length {prompt_tokens} unexpectedly aligns to an internal boundary")
    if "nonaligned serving works" not in (text["choices"][0]["message"]["content"] or ""):
        raise AssertionError("non-aligned text request produced an unexpected response")

    tool = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": "Use the get_weather tool to find the weather in Paris. Do not answer directly.",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="auto",
        max_tokens=96,
        temperature=0,
    ).model_dump()
    tool_calls = tool["choices"][0]["message"]["tool_calls"]
    if len(tool_calls) != 1 or tool_calls[0]["function"]["name"] != "get_weather":
        raise AssertionError(f"Gemma4 tool parser did not return get_weather: {tool_calls}")
    if json.loads(tool_calls[0]["function"]["arguments"]) != {"city": "Paris"}:
        raise AssertionError(f"unexpected tool arguments: {tool_calls[0]}")

    reasoning = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Think briefly, then answer: what is 2 + 3?"}],
        max_tokens=128,
        temperature=0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "skip_special_tokens": False,
        },
    ).model_dump()
    reasoning_message = reasoning["choices"][0]["message"]
    if not reasoning_message["reasoning"] or "5" not in (reasoning_message["content"] or ""):
        raise AssertionError(f"Gemma4 reasoning parser did not split reasoning/content: {reasoning_message}")

    result = {
        "status": "passed",
        "server_url": args.server_url,
        "model": args.model,
        "expected_max_model_len": args.expected_max_model_len,
        "non_aligned_prompt_tokens": prompt_tokens,
        "models_response": model_list,
        "text_response": text,
        "tool_call_response": tool,
        "reasoning_response": reasoning,
        "scope": "text-only; image and video inputs are excluded",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: wrote {args.output}")


if __name__ == "__main__":
    _main()

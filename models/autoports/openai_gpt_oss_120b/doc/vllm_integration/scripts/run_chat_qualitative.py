# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Collect prompt-correct GPT-OSS chat qualitative evidence from vLLM."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from transformers import AutoTokenizer

MODEL_ID = "openai/gpt-oss-120b"
MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"


def _read_prompts(path: Path) -> list[str]:
    prompts = [prompt.strip() for prompt in path.read_text(encoding="utf-8").split("\n\n") if prompt.strip()]
    if len(prompts) != 6:
        raise ValueError(f"expected six blank-line-separated prompts in {path}, got {len(prompts)}")
    return prompts


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


async def _request_one(client, *, server_url: str, prompt_id: int, prompt: str, profile: dict) -> dict:
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": profile.get("max_completion_tokens", 1024),
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "reasoning_effort": profile.get("reasoning_effort", "medium"),
        "include_reasoning": True,
        "return_token_ids": True,
    }
    if profile["seeded"]:
        body["seed"] = 1234 + prompt_id
    response = await client.post(f"{server_url.rstrip('/')}/v1/chat/completions", json=body)
    response.raise_for_status()
    raw = response.json()
    choice = raw["choices"][0]
    message = choice["message"]
    if profile.get("require_non_length") and choice.get("finish_reason") == "length":
        reasoning = message.get("reasoning_content", message.get("reasoning")) or ""
        raise RuntimeError(
            f"prompt {prompt_id} profile {profile['name']} exhausted its completion budget "
            f"after {len(reasoning)} reasoning characters"
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            f"prompt {prompt_id} profile {profile['name']} returned no parsed final-channel content; "
            "raw token detokenization is deliberately not accepted as chat qualitative evidence"
        )
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    return {
        "id": prompt_id,
        "content": content,
        "reasoning": reasoning,
        "reasoning_chars": len(reasoning or ""),
        "token_ids": choice.get("token_ids"),
        "finish_reason": choice.get("finish_reason"),
        "response_id": raw.get("id"),
        "usage": raw.get("usage"),
        "raw": raw,
    }


async def _request_arm(*, server_url: str, prompts: list[str], profile: dict) -> list[dict]:
    timeout = httpx.Timeout(1800.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await asyncio.gather(
            *(
                _request_one(
                    client,
                    server_url=server_url,
                    prompt_id=prompt_id,
                    prompt=prompt,
                    profile=profile,
                )
                for prompt_id, prompt in enumerate(prompts)
            )
        )


def _render_controls(tokenizer, prompts: list[str]) -> list[dict]:
    rendered = []
    for prompt_id, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(token_ids, "input_ids"):
            token_ids = token_ids.input_ids
        elif isinstance(token_ids, dict):
            token_ids = token_ids["input_ids"]
        token_ids = list(token_ids)
        rendered.append(
            {
                "id": prompt_id,
                "prompt": prompt,
                "rendered_prompt": text,
                "rendered_prompt_sha256": _sha256_bytes(text.encode("utf-8")),
                "prompt_token_ids": token_ids,
                "prompt_tokens": len(token_ids),
            }
        )
    return rendered


def main() -> None:
    script = Path(__file__).resolve()
    model_dir = script.parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=model_dir / "doc" / "full_model" / "prompts" / "shared_qualitative_prompts.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=model_dir / "readiness_vllm")
    args = parser.parse_args()

    prompts = _read_prompts(args.prompts)
    prompt_file_sha256 = _sha256_bytes(args.prompts.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    profiles = [
        {
            "name": "greedy",
            "temperature": 0.0,
            "top_p": 1.0,
            "seeded": False,
            "sampling_path": "traced_device_token_out",
        },
        {
            "name": "sampled",
            "temperature": 0.7,
            "top_p": 0.9,
            "seeded": False,
            "sampling_path": "traced_device_token_out",
        },
    ]

    results = {
        profile["name"]: asyncio.run(_request_arm(server_url=args.server_url, prompts=prompts, profile=profile))
        for profile in profiles
    }
    seeded_host_profile = {
        "name": "seeded_host_compatibility_sample",
        "temperature": 0.7,
        "top_p": 0.9,
        "seeded": True,
        "max_completion_tokens": 512,
        "reasoning_effort": "low",
        "require_non_length": True,
        "sampling_path": "explicit_seed_host_compatibility",
    }
    seeded_host_result = asyncio.run(
        _request_arm(
            server_url=args.server_url,
            prompts=prompts[:1],
            profile=seeded_host_profile,
        )
    )[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "vllm_qualitative_outputs.json"
    artifact = []
    raw_responses = {"greedy": [], "sampled": []}
    for prompt_id, prompt in enumerate(prompts):
        greedy = {key: value for key, value in results["greedy"][prompt_id].items() if key != "raw"}
        sampled = {key: value for key, value in results["sampled"][prompt_id].items() if key != "raw"}
        artifact.append(
            {
                "id": prompt_id,
                "prompt": prompt,
                "greedy_completion": greedy["content"],
                "sampled_completion": sampled["content"],
                "greedy": greedy,
                "sampled": sampled,
            }
        )
        raw_responses["greedy"].append(results["greedy"][prompt_id]["raw"])
        raw_responses["sampled"].append(results["sampled"][prompt_id]["raw"])

    generated_at = datetime.now(timezone.utc).isoformat()
    server_prompt_tokens = [len(result["raw"]["prompt_token_ids"]) for result in results["greedy"]]
    sampled_server_prompt_tokens = [len(result["raw"]["prompt_token_ids"]) for result in results["sampled"]]
    if sampled_server_prompt_tokens != server_prompt_tokens:
        raise RuntimeError("greedy and sampled arms were rendered with different server prompt lengths")
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "vllm_chat_qualitative_raw.json").write_text(
        json.dumps(raw_responses, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "seeded_host_compatibility_sample.json").write_text(
        json.dumps(
            {
                "profile": seeded_host_profile,
                "prompt": prompts[0],
                "response": seeded_host_result,
                "generated_at_utc": generated_at,
                "command": " ".join(sys.argv),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    control = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "snapshot": str(args.snapshot.resolve()),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_present": bool(tokenizer.chat_template),
        "prompt_mode": "chat",
        "endpoint": "/v1/chat/completions",
        "rendering_method": (
            "The local tokenizer.apply_chat_template(add_generation_prompt=true) render is a format control, "
            "not an assertion of byte-identical server rendering. The raw vLLM API artifacts preserve the actual "
            "server prompt_token_ids."
        ),
        "server_prompt_token_note": (
            "The live Harmony renderer may differ from the local chat-template control at the system-prompt boundary. "
            "Prompt text, chat endpoint, model revision, and prompt-file SHA match; no claim of exact token-id "
            "identity is made."
        ),
        "actual_server_prompt_tokens": server_prompt_tokens,
        "prompt_file": str(args.prompts.resolve()),
        "prompt_file_sha256": prompt_file_sha256,
        "rendered_prompts": _render_controls(tokenizer, prompts),
        "profiles": profiles,
        "seeded_host_compatibility_smoke_profile": seeded_host_profile,
        "max_completion_tokens": 1024,
        "reasoning_effort": "medium",
        "include_reasoning": True,
        "return_token_ids": True,
        "concurrency_per_arm": 6,
        "cross_request_isolation_probe": True,
        "raw_detokenize_fallback_accepted": False,
        "generated_at_utc": generated_at,
        "command": " ".join(sys.argv),
    }
    (args.output_dir / "qualitative_prompt_format.json").write_text(
        json.dumps(control, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path} with {len(artifact)} prompts x 2 chat profiles")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the shared chat-template qualitative suite through vLLM completions.

The input is the prior full-model/HF control artifact.  Its prompts were
rendered by the checkpoint tokenizer's declared chat template and include the
exact token ids.  Sending those ids directly to ``/v1/completions`` avoids a
second server-side formatting pass while preserving byte-for-byte prompt
provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import httpx
from transformers import AutoTokenizer


def _review(tokens: list[int], text: str) -> dict[str, object]:
    adjacent = sum(left == right for left, right in zip(tokens, tokens[1:]))
    four_grams = Counter(tuple(tokens[index : index + 4]) for index in range(max(0, len(tokens) - 3)))
    repeated_four_grams = sum(count - 1 for count in four_grams.values() if count > 1)
    dominant = max(Counter(tokens).values(), default=0) / max(1, len(tokens))
    letters = [character for character in text if character.isalpha()]
    latin = sum(("a" <= character.lower() <= "z") for character in letters) / max(1, len(letters))
    return {
        "adjacent_token_repeats": adjacent,
        "dominant_token_fraction": dominant,
        "repeated_four_grams": repeated_four_grams,
        "latin_letter_fraction": latin,
        "mechanically_degenerate": bool(dominant > 0.35 or adjacent > max(8, len(tokens) // 5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8018")
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    control = json.loads(args.control.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError("checkpoint tokenizer has no chat template")
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    if template_hash != control["metadata"]["chat_template_sha256"]:
        raise RuntimeError("checkpoint chat template differs from the HF control")

    endpoint = f"{args.server_url.rstrip('/')}/v1/completions"
    results = []
    with httpx.Client(timeout=240.0) as client:
        for prompt in control["prompts"]:
            rendered = tokenizer.apply_chat_template(prompt["messages"], tokenize=False, add_generation_prompt=True)
            encoded_prompt = tokenizer.apply_chat_template(
                prompt["messages"], tokenize=True, add_generation_prompt=True
            )
            prompt_tokens = encoded_prompt["input_ids"] if hasattr(encoded_prompt, "keys") else encoded_prompt
            if prompt_tokens and isinstance(prompt_tokens[0], list):
                prompt_tokens = prompt_tokens[0]
            if rendered != prompt["rendered_prompt"] or list(prompt_tokens) != prompt["prompt_tokens"]:
                raise RuntimeError(f"rendered prompt drift for {prompt['id']}")
            response = client.post(
                endpoint,
                json={
                    "model": args.model,
                    "prompt": list(prompt_tokens),
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["text"]
            tt_tokens = tokenizer.encode(text, add_special_tokens=False)
            hf_tokens = list(prompt["hf_tokens"])
            matching = 0
            for expected, observed in zip(hf_tokens, tt_tokens):
                if expected != observed:
                    break
                matching += 1
            results.append(
                {
                    "id": prompt["id"],
                    "messages": prompt["messages"],
                    "rendered_prompt": rendered,
                    "prompt_tokens": list(prompt_tokens),
                    "hf_control_completion": prompt["hf_completion"],
                    "hf_control_tokens": hf_tokens,
                    "tt_completion": text,
                    "tt_tokens": tt_tokens,
                    "matching_hf_prefix_tokens": matching,
                    "finish_reason": body["choices"][0]["finish_reason"],
                    "usage": body["usage"],
                    "http_status": response.status_code,
                    "tt_review": _review(tt_tokens, text),
                }
            )

    artifact = {
        "schema_version": 1,
        "model": args.model,
        "checkpoint_revision": control["metadata"]["checkpoint_revision"],
        "tokenizer_class": type(tokenizer).__name__,
        "prompt_mode": "chat",
        "rendering_method": "tokenizer.apply_chat_template(add_generation_prompt=True)",
        "transport": "exact rendered chat-template token ids through /v1/completions",
        "chat_template_sha256": template_hash,
        "prompt_suite_sha256": control["metadata"]["prompt_suite_sha256"],
        "control_source": str(args.control),
        "sampling": {
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "server_path": "canonical traced on-device exact global argmax",
            "logprobs": False,
        },
        "results": results,
    }
    artifact["verdict"] = (
        "pass"
        if all(
            result["http_status"] == 200
            and result["usage"]["prompt_tokens"] == len(result["prompt_tokens"])
            and not result["tt_review"]["mechanically_degenerate"]
            for result in results
        )
        else "fail"
    )
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    if artifact["verdict"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

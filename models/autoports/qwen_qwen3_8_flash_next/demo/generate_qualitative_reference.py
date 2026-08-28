# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate prompt-correct HF controls for the shared qualitative suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.demo.generate_hf_reference import _load_oracle
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tt.model import MODEL_ID, MODEL_REVISION

GENERATION_LENGTH = 128
QUALITATIVE_PROMPTS = (
    {
        "id": "explanation",
        "messages": (
            {
                "role": "user",
                "content": (
                    "Explain why the sky looks blue to a curious twelve-year-old. "
                    "Use one simple analogy and keep the answer concise."
                ),
            },
        ),
    },
    {
        "id": "coding",
        "messages": (
            {"role": "system", "content": "You are a careful Python programmer."},
            {
                "role": "user",
                "content": (
                    "Write a Python function named deduplicate_preserving_order that returns the unique "
                    "items from a list in first-seen order. Include one short example."
                ),
            },
        ),
    },
    {
        "id": "summarization",
        "messages": (
            {
                "role": "user",
                "content": (
                    "Summarize this passage in one sentence: A neighborhood library extended its weekend "
                    "hours after a six-month trial. Attendance rose, volunteers covered the additional "
                    "shifts, and the city approved permanent funding without reducing weekday services."
                ),
            },
        ),
    },
)


def _render(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    tokens = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
    return rendered, tokens.reshape(1, -1)


@torch.inference_mode()
def generate(output: Path, *, expert_cache_capacity: int = 32, threads: int | None = None) -> dict:
    if threads is not None:
        torch.set_num_threads(threads)
    H.import_target_transformers()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True)
    model, ple_store, experts = _load_oracle(H.MODEL_SNAPSHOT, expert_cache_capacity=expert_cache_capacity)
    suite_payload = json.dumps(QUALITATIVE_PROMPTS, sort_keys=True, separators=(",", ":")).encode()
    prompts = []
    total_started = time.perf_counter()
    for item in QUALITATIVE_PROMPTS:
        prompt_id = item["id"]
        messages = [dict(message) for message in item["messages"]]
        rendered, prompt_tokens = _render(tokenizer, messages)
        ple_store.reset_request("hf-aime24")
        generated = []
        step_seconds = []
        current = prompt_tokens
        past = None
        for step in range(GENERATION_LENGTH):
            started = time.perf_counter()
            result = model(input_ids=current, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = result.past_key_values
            token = result.logits[0, -1].float().argmax().reshape(1, 1)
            generated.append(int(token))
            current = token
            step_seconds.append(time.perf_counter() - started)
            print(
                f"hf-qualitative prompt={prompt_id} step={step + 1}/{GENERATION_LENGTH} "
                f"seconds={step_seconds[-1]:.3f} token={int(token)}",
                flush=True,
            )
        generated_tokens = torch.tensor(generated, dtype=torch.int64)
        prompts.append(
            {
                "id": prompt_id,
                "messages": messages,
                "rendered_prompt": rendered,
                "prompt_tokens": prompt_tokens.reshape(-1).cpu(),
                "reference_tokens": generated_tokens,
                "reference_text": tokenizer.decode(generated, skip_special_tokens=True),
                "step_seconds": step_seconds,
            }
        )

    artifact = {
        "metadata": {
            "schema_version": 1,
            "hf_model_id": MODEL_ID,
            "checkpoint_revision": MODEL_REVISION,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_name_or_path": str(tokenizer.name_or_path),
            "chat_template": bool(tokenizer.chat_template),
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "prompt_mode": "chat",
            "rendering_method": "tokenizer.apply_chat_template(add_generation_prompt=True)",
            "prompt_source": "Qwen3.8 shared qualitative suite",
            "prompt_suite_sha256": hashlib.sha256(suite_payload).hexdigest(),
            "prompt_ids": [item["id"] for item in QUALITATIVE_PROMPTS],
            "generation_length": GENERATION_LENGTH,
            "sampling": "HF greedy",
            "weight_policy": "persistent non-expert HF BF16 plus exact bounded mmap experts and exact mmap PLE",
            "expert_cache_capacity_per_layer": expert_cache_capacity,
            "generation_command": (
                "python -m models.autoports.qwen_qwen3_8_flash_next.demo.generate_qualitative_reference "
                f"--output {output} --expert-cache-capacity {expert_cache_capacity}"
            ),
        },
        "prompts": prompts,
        "total_generation_seconds": time.perf_counter() - total_started,
        "host_store_metrics": {
            "ple": ple_store.metrics(),
            "expert_reads": sum(expert.reads for expert in experts),
            "expert_hits": sum(expert.hits for expert in experts),
            "expert_read_seconds": sum(expert.read_seconds for expert in experts),
        },
    }
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "metadata": artifact["metadata"],
                "prompts": [
                    {
                        "id": item["id"],
                        "messages": item["messages"],
                        "rendered_prompt": item["rendered_prompt"],
                        "prompt_tokens": item["prompt_tokens"].tolist(),
                        "reference_text": item["reference_text"],
                    }
                    for item in prompts
                ],
                "total_generation_seconds": artifact["total_generation_seconds"],
                "host_store_metrics": artifact["host_store_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    ple_store.close()
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-cache-capacity", type=int, default=32)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate(args.output, expert_cache_capacity=args.expert_cache_capacity, threads=args.threads)


if __name__ == "__main__":
    main()

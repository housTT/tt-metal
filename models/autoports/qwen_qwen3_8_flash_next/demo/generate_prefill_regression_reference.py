# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate the exact HF first-token oracle for the 125-token chat regression.

This intentionally records the full rendered prompt and the top-100 logits.
The release regression returned ``<|im_end|>`` before decode, so a generated
text smoke alone is too weak: the gate must compare the prefill distribution.
"""

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


MESSAGES = (
    {
        "role": "user",
        "content": (
            "Exactly one of these statements is true: (A) B committed the theft. "
            "(B) D committed the theft. (C) Statement B is false. "
            "(D) C committed the theft. Assume exactly one person committed it. "
            "Determine every culprit consistent with the clues. Do not assume the "
            "answer is unique; show a compact truth table and explain your conclusion."
        ),
    },
)
EXPECTED_PROMPT_TOKENS = 125


def _render(tokenizer):
    messages = [dict(message) for message in MESSAGES]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    tokens = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
    tokens = tokens.reshape(1, -1)
    if int(tokens.shape[1]) != EXPECTED_PROMPT_TOKENS:
        raise RuntimeError(
            f"chat regression prompt changed: expected {EXPECTED_PROMPT_TOKENS} tokens, "
            f"got {tokens.shape[1]}"
        )
    return messages, rendered, tokens


@torch.inference_mode()
def generate(output: Path, *, expert_cache_capacity: int = 32, threads: int | None = None) -> dict:
    if threads is not None:
        torch.set_num_threads(threads)
    H.import_target_transformers()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True)
    messages, rendered, prompt = _render(tokenizer)
    load_started = time.perf_counter()
    model, ple_store, experts = _load_oracle(H.MODEL_SNAPSHOT, expert_cache_capacity=expert_cache_capacity)
    load_seconds = time.perf_counter() - load_started
    # MmapPLE uses this stable request identity internally.
    ple_store.reset_request("hf-aime24")
    started = time.perf_counter()
    result = model(input_ids=prompt, use_cache=False, logits_to_keep=1)
    prefill_seconds = time.perf_counter() - started
    logits = result.logits[0, -1].float().cpu()
    values, indices = torch.topk(logits, 100)
    reference_token = int(indices[0])
    artifact = {
        "metadata": {
            "schema_version": 1,
            "hf_model_id": MODEL_ID,
            "checkpoint_revision": MODEL_REVISION,
            "tokenizer_class": type(tokenizer).__name__,
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "purpose": "125-token prefill first-token correctness regression",
            "sampling": "HF greedy",
            "top_k": 100,
            "weight_policy": (
                "persistent non-expert HF BF16 plus exact bounded mmap experts and exact mmap PLE"
            ),
            "expert_cache_capacity_per_layer": expert_cache_capacity,
        },
        "messages": messages,
        "rendered_prompt": rendered,
        "prompt_tokens": prompt.reshape(-1).cpu(),
        "reference_token": reference_token,
        "reference_text": tokenizer.decode([reference_token], skip_special_tokens=False),
        "top100_tokens": indices,
        "top100_values": values,
        "timing": {"model_load_seconds": load_seconds, "prefill_seconds": prefill_seconds},
        "host_store_metrics": {
            "ple": ple_store.metrics(),
            "expert_reads": sum(expert.reads for expert in experts),
            "expert_hits": sum(expert.hits for expert in experts),
            "expert_read_seconds": sum(expert.read_seconds for expert in experts),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "metadata": artifact["metadata"],
                "messages": messages,
                "rendered_prompt": rendered,
                "prompt_tokens": prompt.reshape(-1).tolist(),
                "reference_token": reference_token,
                "reference_text": artifact["reference_text"],
                "top100_tokens": indices.tolist(),
                "top100_values": values.tolist(),
                "timing": artifact["timing"],
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
    artifact = generate(
        args.output,
        expert_cache_capacity=args.expert_cache_capacity,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "prompt_tokens": len(artifact["prompt_tokens"]),
                "reference_token": artifact["reference_token"],
                "reference_text": artifact["reference_text"],
                "timing": artifact["timing"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

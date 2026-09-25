# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate HF prefill oracles around the 128-token microchunk boundary.

Every variant retains the exact seven-token assistant suffix from the shipped
125-token regression.  Short variants remove user-content tokens immediately
before that suffix; variants above 125 insert newline tokens before it.  This
keeps the selected row semantically identical while exercising every relevant
logical-length/indexing path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.demo.generate_hf_reference import _load_oracle
from models.autoports.qwen_qwen3_8_flash_next.demo.generate_prefill_regression_reference import _render
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tt.model import MODEL_ID, MODEL_REVISION


PROMPT_LENGTHS = (96, 104, 110, 118, 124, 125, 126, 127, 128)
ASSISTANT_SUFFIX_TOKENS = 7
NEWLINE_TOKEN_ID = 198


def prompt_variants(prompt: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return deterministic suffix-preserving prompt variants."""

    source = torch.as_tensor(prompt, dtype=torch.int64).reshape(-1)
    if int(source.numel()) != 125:
        raise ValueError(f"expected the 125-token regression prompt, got {source.numel()}")
    suffix = source[-ASSISTANT_SUFFIX_TOKENS:]
    variants = []
    for length in PROMPT_LENGTHS:
        if length <= int(source.numel()):
            value = torch.cat((source[: length - ASSISTANT_SUFFIX_TOKENS], suffix))
        else:
            extension = torch.full(
                (length - int(source.numel()),),
                NEWLINE_TOKEN_ID,
                dtype=torch.int64,
            )
            value = torch.cat((source[:-ASSISTANT_SUFFIX_TOKENS], extension, suffix))
        if int(value.numel()) != length or not torch.equal(value[-ASSISTANT_SUFFIX_TOKENS:], suffix):
            raise RuntimeError(f"failed to construct suffix-preserving length {length}")
        variants.append(value.reshape(1, -1).contiguous())
    return tuple(variants)


@torch.inference_mode()
def generate(output: Path, *, expert_cache_capacity: int = 32, threads: int | None = None) -> dict:
    if threads is not None:
        torch.set_num_threads(threads)
    H.import_target_transformers()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True)
    _messages, _rendered, source_prompt = _render(tokenizer)
    variants = prompt_variants(source_prompt)
    suffix = source_prompt.reshape(-1)[-ASSISTANT_SUFFIX_TOKENS:].cpu()

    load_started = time.perf_counter()
    model, ple_store, experts = _load_oracle(H.MODEL_SNAPSHOT, expert_cache_capacity=expert_cache_capacity)
    load_seconds = time.perf_counter() - load_started
    captured: list[torch.Tensor] = []
    hook = model.model.hyper_connection_mixer.register_forward_hook(
        lambda _module, _inputs, value: captured.append(value.detach().cpu().to(torch.bfloat16).contiguous())
    )
    rows = []
    try:
        for prompt in variants:
            length = int(prompt.shape[1])
            captured.clear()
            ple_store.reset_request("hf-aime24")
            attention_mask = torch.ones_like(prompt, dtype=torch.int64)
            started = time.perf_counter()
            result = model(
                input_ids=prompt,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
            elapsed = time.perf_counter() - started
            if len(captured) != 1 or tuple(captured[0].shape) != (1, length, 2560):
                raise RuntimeError(f"unexpected final hidden capture for length {length}: {captured}")
            hidden = captured[0][0, -1].clone()
            logits = result.logits[0, -1].detach().cpu().float()
            values, tokens = torch.topk(logits, 100)
            rows.append(
                {
                    "length": length,
                    "prompt_tokens": prompt.reshape(-1).cpu(),
                    "attention_mask": attention_mask.reshape(-1).cpu(),
                    "selected_row": length - 1,
                    "selected_final_hidden": hidden,
                    "top100_tokens": tokens,
                    "top100_values": values,
                    "reference_token": int(tokens[0]),
                    "reference_text": tokenizer.decode([int(tokens[0])], skip_special_tokens=False),
                    "prefill_seconds": elapsed,
                }
            )
    finally:
        hook.remove()
        ple_store.close()

    artifact = {
        "metadata": {
            "schema_version": 1,
            "hf_model_id": MODEL_ID,
            "checkpoint_revision": MODEL_REVISION,
            "purpose": "suffix-preserving prefill length/index/mask regression around one 128-token microchunk",
            "prompt_lengths": list(PROMPT_LENGTHS),
            "assistant_suffix_tokens": suffix.tolist(),
            "assistant_suffix_sha256": hashlib.sha256(suffix.numpy().tobytes()).hexdigest(),
            "variant_policy": (
                "remove user-content tokens immediately before the exact seven-token assistant suffix below 125; "
                "insert token 198 newlines immediately before that suffix above 125"
            ),
            "selected_row_policy": "logical prompt length minus one; attention mask is exactly one for every token",
            "sampling": "HF greedy top-100 prefill logits",
            "weight_policy": "persistent non-expert HF BF16 plus exact bounded mmap experts and exact mmap PLE",
            "expert_cache_capacity_per_layer": expert_cache_capacity,
            "model_load_seconds": load_seconds,
        },
        "rows": rows,
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
                "rows": [
                    {
                        "length": row["length"],
                        "prompt_tokens": row["prompt_tokens"].tolist(),
                        "attention_mask": row["attention_mask"].tolist(),
                        "selected_row": row["selected_row"],
                        "reference_token": row["reference_token"],
                        "reference_text": row["reference_text"],
                        "top100_tokens": row["top100_tokens"].tolist(),
                        "top100_values": row["top100_values"].tolist(),
                        "prefill_seconds": row["prefill_seconds"],
                    }
                    for row in rows
                ],
                "host_store_metrics": artifact["host_store_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
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
                "prompt_lengths": artifact["metadata"]["prompt_lengths"],
                "reference_tokens": [row["reference_token"] for row in artifact["rows"]],
                "prefill_seconds": [row["prefill_seconds"] for row in artifact["rows"]],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

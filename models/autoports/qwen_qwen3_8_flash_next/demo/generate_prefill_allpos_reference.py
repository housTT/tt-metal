"""HF bf16 per-position top-100 logits for a teacher-set prompt (C0 prefill agreement).

One HF forward over the whole prompt; stores, for every prompt position, the top-100 token
ids and values of the next-token distribution.  The TT side compares
``prefill_forward(return_all_logits=True)`` against it, which isolates the prefill path
(today's weakest numerics) from decode.

    python -m models.autoports.qwen_qwen3_8_flash_next.demo.generate_prefill_allpos_reference \
        --manifest doc/correctness/teacher_set/manifest.json --prompt-id code_debug \
        --output doc/correctness/teacher_set/refs/code_debug.allpos.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.demo.generate_hf_reference import _load_oracle, _manifest_prompt
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tt.model import MODEL_ID, MODEL_REVISION


@torch.inference_mode()
def generate(output: Path, *, manifest: Path, prompt_id: str, expert_cache_capacity: int = 32,
             threads: int | None = None, top_k: int = 100) -> dict:
    if threads is not None:
        torch.set_num_threads(threads)
    H.import_target_transformers()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True)
    entry, messages, rendered, prompt = _manifest_prompt(tokenizer, manifest, prompt_id)
    model, ple_store, experts = _load_oracle(H.MODEL_SNAPSHOT, expert_cache_capacity=expert_cache_capacity)
    ple_store.reset_request(f"hf-allpos-{prompt_id}")
    started = time.perf_counter()
    result = model(input_ids=prompt, use_cache=False)
    logits = result.logits[0].float()  # [L, V]
    values, indices = torch.topk(logits, top_k, dim=-1)
    seconds = time.perf_counter() - started
    artifact = {
        "metadata": {
            "schema_version": 1,
            "kind": "prefill_all_positions",
            "hf_model_id": MODEL_ID,
            "checkpoint_revision": MODEL_REVISION,
            "prompt_id": prompt_id,
            "prompt_domain": entry.get("domain"),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "top_k": top_k,
            "prompt_tokens": int(prompt.numel()),
            "forward_seconds": seconds,
        },
        "messages": messages,
        "rendered_prompt": rendered,
        "prompt_tokens": prompt.reshape(-1).cpu(),
        "topk_tokens": indices.cpu(),   # [L, top_k]
        "topk_values": values.cpu(),    # [L, top_k]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(json.dumps(artifact["metadata"], indent=2, sort_keys=True) + "\n")
    print(f"allpos-reference {prompt_id}: {int(prompt.numel())} positions in {seconds:.1f}s -> {output}", flush=True)
    ple_store.close()
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-cache-capacity", type=int, default=32)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    generate(args.output, manifest=args.manifest, prompt_id=args.prompt_id,
             expert_cache_capacity=args.expert_cache_capacity, threads=args.threads, top_k=args.top_k)


if __name__ == "__main__":
    main()

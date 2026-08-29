# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generate exact, same-format HF controls for the shared qualitative suite.

The GPT-OSS 120B checkpoint dequantizes to BF16 on this CPU-only host.  Keep a
bounded resident prefix and disk-offload the remaining layers, then generate all
six controls as one greedy batch so every decode step streams offloaded weights
only once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from accelerate import __version__ as accelerate_version
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "openai/gpt-oss-120b"
MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
NUM_LAYERS = 36
CPU_LAYERS = 27
MAX_NEW_TOKENS = 128


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--tt-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--offload-folder", type=Path, required=True)
    return parser.parse_args()


def _device_map() -> dict[str, str]:
    mapping = {
        "model.embed_tokens": "cpu",
        "model.norm": "cpu",
        "model.rotary_emb": "cpu",
        "lm_head": "cpu",
    }
    for layer_idx in range(NUM_LAYERS):
        mapping[f"model.layers.{layer_idx}"] = "cpu" if layer_idx < CPU_LAYERS else "disk"
    return mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stop_ids(model, tokenizer) -> list[int]:
    ids = model.generation_config.eos_token_id
    if ids is None:
        ids = tokenizer.eos_token_id
    if isinstance(ids, int):
        ids = [ids]
    return [int(token_id) for token_id in ids]


def _chat_prompt_tokens(tokenizer, prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
    )
    if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
        encoded = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.reshape(-1).tolist()
    while isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def main() -> None:
    args = _parse_args()
    prompts = [text.strip() for text in args.prompts.read_text(encoding="utf-8").split("\n\n") if text.strip()]
    if len(prompts) != 6:
        raise ValueError(f"Expected six prompts, found {len(prompts)}")

    args.offload_folder.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot,
        trust_remote_code=True,
        local_files_only=True,
        device_map=_device_map(),
        offload_folder=args.offload_folder,
        offload_state_dict=True,
        offload_buffers=True,
        low_cpu_mem_usage=True,
    ).eval()

    prompt_token_ids = [_chat_prompt_tokens(tokenizer, prompt) for prompt in prompts]
    tt_controls = json.loads(args.tt_control.read_text(encoding="utf-8"))
    if [entry["prompt_token_ids"] for entry in tt_controls] != prompt_token_ids:
        raise RuntimeError("HF and TT prompt token IDs differ; controls would not be same-format")

    stop_ids = _stop_ids(model, tokenizer)
    pad_id = int(model.generation_config.pad_token_id or tokenizer.pad_token_id or stop_ids[0])
    max_prompt = max(len(tokens) for tokens in prompt_token_ids)
    input_ids = torch.full((len(prompts), max_prompt), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, tokens in enumerate(prompt_token_ids):
        input_ids[row, -len(tokens) :] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[row, -len(tokens) :] = 1

    with torch.inference_mode():
        sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            eos_token_id=stop_ids,
            pad_token_id=pad_id,
            use_cache=True,
        )

    generated = sequences[:, max_prompt:].tolist()
    controls = []
    comparisons = []
    for index, (prompt, prompt_tokens, output_tokens, tt_entry) in enumerate(
        zip(prompts, prompt_token_ids, generated, tt_controls, strict=True)
    ):
        for token_index, token in enumerate(output_tokens):
            if token in stop_ids:
                output_tokens = output_tokens[: token_index + 1]
                break
        tt_tokens = tt_entry["completion_token_ids"]
        rendered_prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        stop_reason = (
            f"eos:{output_tokens[-1]}" if output_tokens and output_tokens[-1] in stop_ids else "max_new_tokens"
        )
        common_prefix = 0
        for hf_token, tt_token in zip(output_tokens, tt_tokens):
            if hf_token != tt_token:
                break
            common_prefix += 1
        controls.append(
            {
                "id": index,
                "prompt": prompt,
                "rendered_prompt": rendered_prompt,
                "rendered_prompt_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
                "prompt_token_ids": prompt_tokens,
                "completion_token_ids": output_tokens,
                "completion": tokenizer.decode(output_tokens, skip_special_tokens=False),
                "stop_reason": stop_reason,
            }
        )
        comparisons.append(
            {
                "id": index,
                "prompt_tokens_identical": prompt_tokens == tt_entry["prompt_token_ids"],
                "hf_tokens": len(output_tokens),
                "tt_tokens": len(tt_tokens),
                "common_prefix_tokens": common_prefix,
                "equal_positions_over_overlap": sum(a == b for a, b in zip(output_tokens, tt_tokens)),
                "overlap_tokens": min(len(output_tokens), len(tt_tokens)),
            }
        )

    command = " ".join(shlex.quote(argument) for argument in [sys.executable, *sys.argv])
    provenance = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "snapshot": str(args.snapshot),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "accelerate_version": accelerate_version,
        "python_version": sys.version,
        "tokenizer_class": type(tokenizer).__name__,
        "script_sha256": _sha256(Path(__file__)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "prompt_file": str(args.prompts),
        "prompt_file_sha256": _sha256(args.prompts),
        "rendering": "tokenizer.apply_chat_template([{role: user, content: prompt}], add_generation_prompt=True)",
        "generation": {
            "batch_size": len(prompts),
            "greedy": True,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "eos_token_ids": stop_ids,
            "pad_token_id": pad_id,
        },
        "dispatch": {
            "cpu_layers": list(range(CPU_LAYERS)),
            "disk_layers": list(range(CPU_LAYERS, NUM_LAYERS)),
            "offload_folder": str(args.offload_folder),
        },
        "batching_rationale": "The six independent prompts share one greedy batch so each NVMe-offloaded layer is streamed once per decode step; attention masks isolate rows and prompt ids are asserted against the per-request TT controls.",
    }
    args.output.write_text(
        json.dumps({"provenance": provenance, "outputs": controls}, indent=2) + "\n", encoding="utf-8"
    )
    args.comparison_output.write_text(
        json.dumps({"provenance": provenance, "comparisons": comparisons}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

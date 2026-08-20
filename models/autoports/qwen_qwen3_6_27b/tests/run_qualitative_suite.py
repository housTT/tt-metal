# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prompt-correct HF/TT controls for the shared readiness qualitative suite."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.autoports.qwen_qwen3_6_27b.tt.generator import Generator
from models.autoports.qwen_qwen3_6_27b.tt.model import MODEL_ID, _resolve_checkpoint
from models.common.readiness_check.mesh_device import (
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)


DEFAULT_SUITE = Path(__file__).parents[3] / "common/readiness_check/vllm_prompts.txt"
DEFAULT_OUTPUT = Path(__file__).parents[1] / "doc/full_model/evidence/qualitative"


def _render_suite(tokenizer, source: Path) -> list[dict]:
    prompts = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    rendered = []
    for index, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        rendered.append(
            {
                "prompt_id": f"shared-{index}",
                "messages": messages,
                "rendered_prompt": text,
                "prompt_token_ids": tokenizer.encode(text, add_special_tokens=False),
            }
        )
    return rendered


def _hf_outputs(checkpoint: Path, tokenizer, items: list[dict], max_new_tokens: int):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.bfloat16
    ).eval()
    outputs = []
    for item in items:
        input_ids = torch.tensor([item["prompt_token_ids"]], dtype=torch.long)
        start = time.perf_counter()
        with torch.no_grad():
            result = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        token_ids = result[0, input_ids.shape[1] :].tolist()
        elapsed = time.perf_counter() - start
        outputs.append(
            item
            | {
                "completion_token_ids": token_ids,
                "completion": tokenizer.decode(token_ids, skip_special_tokens=False),
                "generation_seconds": elapsed,
            }
        )
    return outputs


def _tt_outputs(checkpoint: Path, tokenizer, items: list[dict], max_new_tokens: int):
    mesh = open_readiness_mesh_device("P300", "FABRIC_1D_RING", 1_500_000_000)
    try:
        generator = Generator(mesh_device=mesh, checkpoint_path=checkpoint)
        try:
            outputs = []
            for item in items:
                start = time.perf_counter()
                token_ids = generator.generate(
                    item["prompt_token_ids"],
                    max_new_tokens,
                    enable_trace=True,
                    stop_on_eos=True,
                )
                elapsed = time.perf_counter() - start
                outputs.append(
                    item
                    | {
                        "completion_token_ids": token_ids,
                        "completion": tokenizer.decode(token_ids, skip_special_tokens=False),
                        "generation_seconds": elapsed,
                    }
                )
            return outputs
        finally:
            generator.teardown()
    finally:
        close_readiness_mesh_device(mesh, "FABRIC_1D_RING")


def _repetition_metrics(token_ids: list[int]) -> dict:
    adjacent_duplicates = sum(a == b for a, b in zip(token_ids, token_ids[1:]))
    runs = []
    for token_id in token_ids:
        if runs and runs[-1][0] == token_id:
            runs[-1][1] += 1
        else:
            runs.append([token_id, 1])
    trigrams = [tuple(token_ids[index : index + 3]) for index in range(len(token_ids) - 2)]
    repeated_trigrams = sum(count - 1 for count in Counter(trigrams).values() if count > 1)
    return {
        "adjacent_duplicate_count": adjacent_duplicates,
        "maximum_identical_token_run": max((run[1] for run in runs), default=0),
        "repeated_trigram_fraction": repeated_trigrams / max(len(trigrams), 1),
    }


def _compare_outputs(output_dir: Path) -> dict:
    hf = json.loads((output_dir / "qualitative_hf_outputs.json").read_text())
    tt = json.loads((output_dir / "qualitative_tt_outputs.json").read_text())
    if [item["prompt_id"] for item in hf] != [item["prompt_id"] for item in tt]:
        raise RuntimeError("HF and TT qualitative prompt ids do not match")
    comparisons = []
    for hf_item, tt_item in zip(hf, tt):
        hf_ids = hf_item["completion_token_ids"]
        tt_ids = tt_item["completion_token_ids"]
        prefix = 0
        while prefix < min(len(hf_ids), len(tt_ids)) and hf_ids[prefix] == tt_ids[prefix]:
            prefix += 1
        metrics = _repetition_metrics(tt_ids)
        mechanical_failure = (
            metrics["maximum_identical_token_run"] > 3
            or metrics["repeated_trigram_fraction"] > 0.25
            or "<|" in tt_item["completion"]
            or "\ufffd" in tt_item["completion"]
        )
        comparisons.append(
            {
                "prompt_id": tt_item["prompt_id"],
                "hf_tt_matching_prefix_tokens": prefix,
                "first_divergent_token_index": prefix
                if prefix < min(len(hf_ids), len(tt_ids))
                else None,
                "tt_completion_tokens": len(tt_ids),
                **metrics,
                "control_token_or_replacement_leak": "<|" in tt_item["completion"]
                or "\ufffd" in tt_item["completion"],
                "mechanical_failure": mechanical_failure,
            }
        )
    return {
        "prompt_count": len(comparisons),
        "automatic_verdict": "pass"
        if all(not item["mechanical_failure"] for item in comparisons)
        else "fail",
        "checks": comparisons,
        "manual_review_required": [
            "coherence relative to the HF control",
            "wrong-language drift",
            "prompt echo or cross-request leakage",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("hf", "tt", "check"), required=True)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.backend == "check":
        report = _compare_outputs(args.output_dir)
        path = args.output_dir / "qualitative_degeneracy_report.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{report['automatic_verdict']}: wrote qualitative comparison to {path}")
        return

    checkpoint = _resolve_checkpoint(None)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError("Qwen3.6 readiness qualitative checks require its chat template")
    items = _render_suite(tokenizer, args.suite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_id": MODEL_ID,
        "checkpoint_revision": checkpoint.name,
        "tokenizer_class": tokenizer.__class__.__name__,
        "chat_template_present": True,
        "prompt_mode": "chat",
        "render_method": "tokenizer.apply_chat_template(add_generation_prompt=True)",
        "prompt_source": str(args.suite.resolve()),
        "prompt_count": len(items),
        "generation": {"greedy": True, "max_new_tokens": args.max_new_tokens},
    }
    (args.output_dir / "qualitative_prompt_format.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    outputs = (
        _hf_outputs(checkpoint, tokenizer, items, args.max_new_tokens)
        if args.backend == "hf"
        else _tt_outputs(checkpoint, tokenizer, items, args.max_new_tokens)
    )
    path = args.output_dir / f"qualitative_{args.backend}_outputs.json"
    path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(outputs)} {args.backend.upper()} controls to {path}")


if __name__ == "__main__":
    main()

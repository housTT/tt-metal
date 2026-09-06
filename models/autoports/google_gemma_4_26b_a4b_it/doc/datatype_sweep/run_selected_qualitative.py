# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run the shared prompt suite through HF and the selected TT precision policy."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(
    "/home/hous/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/"
    "snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
)
PROMPT_SOURCE = (
    ROOT.parent / "optimized_full_model" / "final" / "qualitative" / "shared_readiness_suite" / "vllm_prompts.txt"
)
OUTPUT = ROOT / "artifacts" / "selected_qualitative"
MAX_NEW_TOKENS = 64


def main():
    prompts = [prompt.strip() for prompt in PROMPT_SOURCE.read_text().split("\n\n") if prompt.strip()]
    case_ids = [
        "machine_learning_haiku",
        "supervised_unsupervised",
        "inventor_story",
        "thermodynamics_laws",
        "french_translation",
        "fibonacci_function",
    ]
    assert len(prompts) == len(case_ids) == 6
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    cases = {}
    for case_id, prompt in zip(case_ids, prompts):
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        cases[case_id] = {
            "prompt_text": prompt,
            "rendered_prompt": tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            ),
            "prompt_token_ids": encoded.reshape(-1).tolist(),
        }

    hf = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).eval()
    with torch.no_grad():
        for case in cases.values():
            prompt_ids = case["prompt_token_ids"]
            generated = hf.generate(
                torch.tensor([prompt_ids]),
                attention_mask=torch.ones((1, len(prompt_ids)), dtype=torch.long),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )[0, len(prompt_ids) :]
            case["hf_token_ids"] = generated.tolist()
    del hf
    gc.collect()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 4)), trace_region_size=134_217_728)
    try:
        generator = build_generator(
            model_dir=ROOT.parents[1],
            mesh_device=mesh,
            model_path=str(CHECKPOINT),
            max_seq_len=512,
        )
        assert generator.model.precision_policy["config_id"] == "selected_canonical_profile_policy"
        precision_summary = generator.model.precision_summary()
        for index, case in enumerate(cases.values()):
            if index:
                generator.reset()
            case["tt_token_ids"] = generator.generate(
                prompt_token_ids=case["prompt_token_ids"],
                max_new_tokens=MAX_NEW_TOKENS,
                enable_trace=True,
            )
            case["trace_counters"] = vars(generator.trace_counters).copy()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "vllm_prompts.txt").write_text(PROMPT_SOURCE.read_text())
    metadata = {
        "hf_model_id": "google/gemma-4-26B-A4B-it",
        "checkpoint_revision": CHECKPOINT.name,
        "selected_config_id": "selected_canonical_profile_policy",
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_present": bool(tokenizer.chat_template),
        "prompt_mode": "chat",
        "rendering_method": "tokenizer.apply_chat_template(add_generation_prompt=True)",
        "prompt_source": str(PROMPT_SOURCE),
        "retained_prompt_source": "vllm_prompts.txt",
        "prompt_source_sha256": hashlib.sha256(PROMPT_SOURCE.read_bytes()).hexdigest(),
        "generation": {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False},
        "trace_allocation_tracking": os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1",
        "command": os.environ.get("GEMMA4_SWEEP_COMMAND", "see work_log.md"),
        "reset_between_tt_cases": True,
        "precision_summary": precision_summary,
        "cases": {},
    }
    for case_id, case in cases.items():
        hf_text = tokenizer.decode(case["hf_token_ids"], skip_special_tokens=False)
        tt_text = tokenizer.decode(case["tt_token_ids"], skip_special_tokens=False)
        case_dir = OUTPUT / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "hf_completion.txt").write_text(hf_text)
        (case_dir / "tt_completion.txt").write_text(tt_text)
        (case_dir / "autoregressive_meta.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "hf": {"token_ids": case["hf_token_ids"]},
                    "tt": {"token_ids": case["tt_token_ids"]},
                },
                indent=2,
            )
            + "\n"
        )
        metadata["cases"][case_id] = case
        print(f"{case_id} HF:\n{hf_text}\n{case_id} TT:\n{tt_text}")
    (OUTPUT / "prompt_format_and_outputs.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

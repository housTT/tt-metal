#!/usr/bin/env python3
"""Stage-10 qualitative verdict: the served greedy and sampled completions (chat template, 256 tokens) against
the stage-9 served greedy text (same server configuration, pre-change code), the stage-9 generator control
and the HF greedy control, as token ids; plus the degeneracy statistics the stage-9 verdict used."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

M = Path("models/autoports/qwen_qwen3_8b")
SNAPSHOT = "/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"


def first_divergence(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def adjacent_dup(ids: list[int]) -> float:
    return sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1]) / max(1, len(ids) - 1)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    served = json.loads((M / "readiness_vllm" / "vllm_qualitative_outputs.json").read_text())
    stage9 = json.loads((M / "doc/optimized_vllm/served_stage9_final/vllm_qualitative_outputs.json").read_text())
    gen9 = {
        r["prompt"]: r["token_ids"]
        for r in json.loads((M / "doc/vllm_integration/qualitative_control/chip1/tt_outputs.json").read_text())
    }
    hf = {
        r["prompt"]: r["token_ids"]
        for r in json.loads((M / "doc/optimized_full_model/qualitative/hf_outputs.json").read_text())
    }
    rows = []
    for item in served:
        prompt = item["prompt"]
        greedy = tok.encode(item["greedy_completion"], add_special_tokens=False)
        sampled = tok.encode(item.get("sampled_completion", ""), add_special_tokens=False)
        prev = next((r for r in stage9 if r["prompt"] == prompt), None)
        prev_ids = tok.encode(prev["greedy_completion"], add_special_tokens=False) if prev else []
        text = item["greedy_completion"]
        rows.append(
            {
                "prompt": prompt[:60],
                "prompt_mode": item.get("prompt_mode"),
                "greedy_tokens": len(greedy),
                "sampled_tokens": len(sampled),
                "first_divergence_vs_stage9_served_greedy": first_divergence(greedy, prev_ids),
                "stage9_served_tokens": len(prev_ids),
                "first_divergence_vs_stage9_generator_control": first_divergence(greedy, gen9.get(prompt, [])),
                "first_divergence_vs_hf": first_divergence(greedy, hf.get(prompt, [])),
                "control_tokens": len(gen9.get(prompt, [])),
                "adjacent_dup_fraction": round(adjacent_dup(greedy), 4),
                "non_ascii_fraction": round(sum(1 for c in text if ord(c) > 127) / max(1, len(text)), 4),
                "has_think": text.lstrip().startswith("<think>"),
                "control_token_leak": any(t in text[8:] for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")),
                "greedy_head": text[:140].replace("\n", " "),
                "sampled_head": item.get("sampled_completion", "")[:100].replace("\n", " "),
            }
        )
    verdict = {
        "served": "readiness_vllm/vllm_qualitative_outputs.json (stage-10 server, chat completions, greedy 256 tokens and sampled)",
        "controls": {
            "stage9_served": "doc/optimized_vllm/served_stage9_final/vllm_qualitative_outputs.json (stage-9 final server, same launch line)",
            "stage9_generator_chip1": "doc/vllm_integration/qualitative_control/chip1/tt_outputs.json (stage-9 code, generator greedy, 128 tokens)",
            "hf": "doc/optimized_full_model/qualitative/hf_outputs.json (HF greedy, chat template, 128 tokens)",
            "prompt_format": "doc/optimized_full_model/qualitative/qualitative_prompt_format.json (chat template, stage 7; tokenizer and prompts unchanged)",
        },
        "rows": rows,
    }
    out = M / "doc/optimized_vllm/qualitative_verdict.json"
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    for r in rows:
        print({k: v for k, v in r.items() if k not in ("greedy_head", "sampled_head")})
    print(out)


if __name__ == "__main__":
    main()

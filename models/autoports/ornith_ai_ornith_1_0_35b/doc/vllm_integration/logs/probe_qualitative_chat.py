# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The shared qualitative prompts through serving in the checkpoint's own prompt format, with controls.

``run_vllm_server``'s qualitative stage posts the bare prompt to ``/v1/completions``. This checkpoint
has a chat template, so per ``$qualitative-check`` that run is continuation stress coverage, not a
quality verdict: the verdict needs the same prompts rendered the way the checkpoint declares. This
probe sends them through ``/v1/chat/completions`` - which makes the server apply
``tokenizer.apply_chat_template(..., add_generation_prompt=True)`` - and puts each completion beside the
two controls the full-model stage already produced from the *same* rendered prompts:

* the HF reference completion (``doc/full_model/readiness_qualitative.json`` -> ``hf``);
* the full-model TTNN completion on the same mesh (-> ``tt``).

Greedy, 128 new tokens, matching those controls. The artifact records the prompt-format decision, the
rendered prompt and its token ids, all three completions per prompt, and, per prompt, the token index
where serving first departs from each control - which is what separates "different near-tie" from
"different answer".

    python .../doc/vllm_integration/logs/probe_qualitative_chat.py --url http://localhost:8100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request as urlrequest

MODEL_DIR = Path(__file__).resolve().parents[3]
HF_MODEL = "ornith-ai/Ornith-1.0-35B"
PROMPTS_FILE = Path("models/common/readiness_check/vllm_prompts.txt")
CONTROL = MODEL_DIR / "doc" / "full_model" / "readiness_qualitative.json"


def chat(url, model, prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urlrequest.Request(
        url + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urlrequest.urlopen(req, timeout=1800) as response:
        body = json.load(response)
    choice = body["choices"][0]
    message = choice["message"]
    text = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    return {
        "content": text,
        "reasoning_content": reasoning,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": body["usage"]["completion_tokens"],
    }


def first_divergence(left: str, right: str):
    """Index of the first differing character, or None when one is a prefix of the other."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) or not left or not right else min(len(left), len(right))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8100")
    ap.add_argument("--model", default=HF_MODEL)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--output", default=str(MODEL_DIR / "doc" / "vllm_integration" / "qualitative_chat.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [p.strip() for p in PROMPTS_FILE.read_text().split("\n\n") if p.strip()]
    control = json.loads(CONTROL.read_text())
    control_by_prompt = {
        entry["prompt"]: {"hf": hf["completion"], "tt": entry["completion"]}
        for entry, hf in zip(control["tt"], control["hf"])
    }

    report = {
        "prompt_format": {
            "hf_model_id": args.model,
            "tokenizer_class": type(tokenizer).__name__,
            "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
            "prompt_mode": "chat",
            "rendering": "server-side tokenizer.apply_chat_template(add_generation_prompt=True) via /v1/chat/completions",
            "prompt_source": str(PROMPTS_FILE),
            "generation": {"greedy": True, "temperature": 0.0, "max_new_tokens": args.max_tokens},
            "controls": {
                "hf_reference": f"{CONTROL}::hf",
                "full_model_ttnn": f"{CONTROL}::tt",
                "note": "both controls were produced by the full-model stage from the same rendered chat prompts",
            },
        },
        "results": [],
    }

    for prompt in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        served = chat(args.url, args.model, prompt, args.max_tokens)
        # The chat endpoint may split the model's reasoning block out of `content`; the controls are raw
        # continuations of the rendered prompt, so compare against the concatenation.
        served_text = (served["reasoning_content"] + served["content"]).strip()
        controls = control_by_prompt.get(prompt, {})
        entry = {
            "prompt": prompt,
            "rendered": rendered,
            "token_ids": tokenizer(rendered, add_special_tokens=False)["input_ids"],
            "served": served,
            "served_text": served_text,
            "control_hf": controls.get("hf"),
            "control_full_model_tt": controls.get("tt"),
            "first_char_divergence_vs_hf": (
                first_divergence(served_text, controls["hf"].strip()) if "hf" in controls else None
            ),
            "first_char_divergence_vs_full_model": (
                first_divergence(served_text, controls["tt"].strip()) if "tt" in controls else None
            ),
        }
        report["results"].append(entry)
        print(f"--- {prompt[:60]}")
        print(f"    served : {served_text[:140]!r}")
        print(f"    hf     : {(controls.get('hf') or '')[:140]!r}")
        print(
            f"    diverges vs hf at char {entry['first_char_divergence_vs_hf']}, "
            f"vs full model at {entry['first_char_divergence_vs_full_model']}"
        )

    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

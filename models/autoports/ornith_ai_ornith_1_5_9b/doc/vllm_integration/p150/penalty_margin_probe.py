# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Control for an unchanged greedy output under frequency penalty: verify the device decision against raw logits.

Device path: greedy with the penalty, no logprobs. Raw logits: a prefill of prompt plus the generated
prefix with top-20 raw logprobs (host-compatibility path, no penalty). At every step the chosen token must
be the argmax of raw_logit - penalty * count_in_generated_prefix over the observable candidates.
"""

import argparse
import json
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="ornith-ai/Ornith-1.5-9B")
    parser.add_argument("--tokenizer", default="/home/hous/dev/ornith-1.5-9b/upstream")
    parser.add_argument("--prompt", default="After a while, she")
    parser.add_argument("--penalty", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompt_ids = tok(args.prompt, add_special_tokens=False)["input_ids"]
    url = args.url.rstrip("/") + "/v1/completions"

    def device(penalty):
        body = {
            "model": args.model,
            "prompt": prompt_ids,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "return_token_ids": True,
            "frequency_penalty": penalty,
        }
        raw = requests.post(url, json=body, timeout=600).json()
        return raw["choices"][0]["token_ids"], raw["choices"][0]["text"]

    with_pen, text_pen = device(args.penalty)
    without, text_plain = device(0.0)
    steps = []
    for k in range(len(with_pen)):
        body = {
            "model": args.model,
            "prompt": prompt_ids + with_pen[:k],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 20,
            "return_token_ids": True,
        }
        raw = requests.post(url, json=body, timeout=600).json()
        choice = raw["choices"][0]
        top = choice["logprobs"]["top_logprobs"][0]
        candidates = {}
        for key, value in top.items():
            ids = tok(key, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                candidates[ids[0]] = value
        emitted = choice["token_ids"][0]
        candidates.setdefault(emitted, choice["logprobs"]["token_logprobs"][0])
        counts = {}
        for t in with_pen[:k]:
            counts[t] = counts.get(t, 0) + 1
        penalized = {t: v - args.penalty * counts.get(t, 0) for t, v in candidates.items()}
        best = max(penalized, key=penalized.get)
        chosen = with_pen[k]
        ranked = sorted(penalized.items(), key=lambda kv: -kv[1])
        steps.append(
            {
                "step": k,
                "chosen": chosen,
                "chosen_text": tok.decode([chosen]),
                "chosen_count_before": counts.get(chosen, 0),
                "raw_argmax": emitted,
                "penalized_argmax": best,
                "agrees": best == chosen,
                "chosen_in_candidates": chosen in candidates,
                "margin_to_runner_up": (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else None,
                "top_penalized": [(t, tok.decode([t]), round(v, 3), counts.get(t, 0)) for t, v in ranked[:5]],
            }
        )
    report = {
        "prompt": args.prompt,
        "penalty": args.penalty,
        "device_with_penalty": {"ids": with_pen, "text": text_pen},
        "device_without_penalty": {"ids": without, "text": text_plain},
        "outputs_identical": with_pen == without,
        "steps": steps,
        "all_steps_agree": all(s["agrees"] for s in steps if s["chosen_in_candidates"]),
        "steps_with_repeated_chosen_token": [s["step"] for s in steps if s["chosen_count_before"] > 0],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: report[k] for k in ("outputs_identical", "all_steps_agree", "steps_with_repeated_chosen_token")})
    )
    for s in steps:
        print(
            s["step"],
            repr(s["chosen_text"]),
            "count",
            s["chosen_count_before"],
            "agree",
            s["agrees"],
            "margin",
            None if s["margin_to_runner_up"] is None else round(s["margin_to_runner_up"], 3),
            s["top_penalized"][:3],
        )


if __name__ == "__main__":
    main()

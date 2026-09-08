# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact numerical logprob repeat/permutation check; run apart from benchmarks.

These are raw prose continuation controls, not chat-quality evaluations.
Logprobs explicitly exercise the authorized host compatibility path.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
PROMPTS = {
    "A": (
        "On Saturday morning, Maya walked to the community library carrying a box of donated books. "
        "The old reading room had been closed for months while volunteers repaired the windows and painted the walls. "
        "Now sunlight reached every table, and the shelves smelled faintly of fresh wood. "
        "Children were gathering near the entrance with their families, eager to discover the new stories. "
        "Maya set down her box beside the librarian and looked around the room. "
        "A neighbor was arranging flowers on the counter, while two students tested a lamp in the quiet corner. "
        "Everyone had contributed something small, and together they had made a welcoming place to read. "
        "When the doors finally opened, the first visitor stepped inside"
    ),
    "B": (
        "Regular exercise can make daily life more comfortable by improving strength, balance, and endurance. "
        "A short walk after lunch offers a practical starting point for someone building a new routine. "
        "The aim is to choose an activity that feels enjoyable and can fit naturally into an ordinary week. "
        "With time, small consistent efforts can help people feel more energetic"
    ),
}
CASES = [("repeat_A_1", "A"), ("repeat_A_2", "A"), ("ABA", "ABA"), ("BAA", "BAA"), ("AAB", "AAB")]


def prompt_manifest(model_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=REVISION, local_files_only=True)
    prompts = {}
    for label, text in PROMPTS.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) == {"A": 131, "B": 65}[label], (label, len(ids))
        assert tokenizer.decode(ids) == text
        prompts[label] = {"text": text, "token_ids": ids, "token_count": len(ids)}
    return prompts


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def token_id(value):
    assert isinstance(value, str) and value.startswith("token_id:"), value
    return int(value.split(":", 1)[1])


def signature(choice, expected_prompt_ids, max_tokens, top_count):
    """Keep actual numerical values; equality must detect even a one-ULP change."""
    logprobs = choice["logprobs"]
    ids = choice["token_ids"]
    assert choice["prompt_token_ids"] == expected_prompt_ids
    assert len(ids) == len(logprobs["tokens"]) == len(logprobs["token_logprobs"]) == max_tokens
    assert [token_id(value) for value in logprobs["tokens"]] == ids
    assert len(logprobs["top_logprobs"]) == max_tokens
    steps = []
    for emitted, chosen, alternatives in zip(ids, logprobs["token_logprobs"], logprobs["top_logprobs"]):
        top = {str(token_id(key)): value for key, value in alternatives.items()}
        assert len(top) == top_count and str(emitted) in top
        assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in [chosen, *top.values()])
        assert chosen == top[str(emitted)]
        steps.append({"token_id": emitted, "logprob": chosen, "top_logprobs": top})
    return {"prompt_token_ids": expected_prompt_ids, "steps": steps}


def compare(left, right):
    """Record token, chosen-logprob and alternative-map comparisons separately."""
    assert len(left["steps"]) == len(right["steps"])
    differences = []
    for step, (a, b) in enumerate(zip(left["steps"], right["steps"])):
        if a != b:
            shared = a["top_logprobs"].keys() & b["top_logprobs"].keys()
            differences.append(
                {
                    "step": step,
                    "token_ids_equal": a["token_id"] == b["token_id"],
                    "chosen_logprobs": [a["logprob"], b["logprob"]],
                    "top_token_sets_equal": a["top_logprobs"].keys() == b["top_logprobs"].keys(),
                    "max_shared_top_logprob_abs_difference": max(
                        (abs(a["top_logprobs"][key] - b["top_logprobs"][key]) for key in shared), default=None
                    ),
                }
            )
    return {"exact": left == right, "differing_steps": differences}


def comparisons(cases):
    references, results = {}, []
    for case in cases:
        for position, (label, item) in enumerate(zip(case["labels"], case["signatures"])):
            location = f"{case['name']}[{position}]"
            if label not in references:
                references[label] = (location, item)
                continue
            source, reference = references[label]
            results.append({"reference": source, "candidate": location, **compare(reference, item)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="../upstream")
    parser.add_argument("--tokenizer", type=Path, default=Path("../upstream"))
    parser.add_argument("--server-max-num-seqs", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="Save exact prompts without making requests")
    args = parser.parse_args()
    assert 1 <= args.top_logprobs <= 20 and args.max_tokens > 0
    report = {
        "scope": "raw prose, greedy numeric logprob determinism; explicit host compatibility; no performance claim",
        "batch_position_scope": "API prompt-array positions; physical scheduler slots are not inferred from indices",
        "server_max_num_seqs": args.server_max_num_seqs,
        "checkpoint_revision": REVISION,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompts": prompt_manifest(args.tokenizer),
        "cases": [],
    }
    save(args.output, report)
    if args.prepare_only:
        print(json.dumps({"prepared": str(args.output), "prompt_lengths": [131, 65]}), flush=True)
        return

    import requests

    for name, labels in CASES:
        texts = [PROMPTS[label] for label in labels]
        body = {
            "model": args.model,
            "prompt": texts[0] if len(texts) == 1 else texts,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 7,
            "ignore_eos": True,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "logprobs": args.top_logprobs,
        }
        response = requests.post(args.url.rstrip("/") + "/v1/completions", json=body, timeout=300)
        result = response.json()
        case = {
            "name": name,
            "labels": list(labels),
            "request": body,
            "status": response.status_code,
            "response": result,
        }
        report["cases"].append(case)
        save(args.output, report)
        assert response.status_code == 200 and "error" not in result, result
        choices = sorted(result["choices"], key=lambda item: item["index"])
        assert [item["index"] for item in choices] == list(range(len(labels))), choices
        assert result["usage"]["completion_tokens"] == args.max_tokens * len(labels)
        assert result["usage"]["prompt_tokens"] == sum(report["prompts"][label]["token_count"] for label in labels)
        case["signatures"] = [
            signature(choice, report["prompts"][label]["token_ids"], args.max_tokens, args.top_logprobs)
            for label, choice in zip(labels, choices)
        ]
        save(args.output, report)
        print(
            json.dumps(
                {"case": name, "choices": [{"ids": item["token_ids"], "text": item["text"]} for item in choices]}
            ),
            flush=True,
        )
    report["comparisons"] = comparisons(report["cases"])
    report["passed"] = all(item["exact"] for item in report["comparisons"])
    save(args.output, report)
    print(json.dumps({"passed": report["passed"], "comparisons": report["comparisons"]}), flush=True)
    assert report["passed"], "Exact numerical logprob determinism failed; inspect saved comparisons and raw responses"


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Meaningful non-aligned completion requests using normal device sampling."""

import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="/home/hous/dev/ornith-1.5-9b/upstream")
    parser.add_argument("--server-manifest", required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = json.loads((Path(__file__).parent.parent / "vllm_integration/logit_determinism_vllm.json").read_text())
    prompts = control["prompts"]
    result = {
        "scope": "raw prose continuation and logical-length contract; normal device sampling, no logprobs",
        "server_manifest": args.server_manifest,
        "max_model_len": 262144,
        "max_num_seqs": args.max_num_seqs,
        "reduced_layers": args.reduced,
        "cases": [],
    }
    cases = [["A"], ["B"], ["A"]]
    if args.max_num_seqs > 1:
        cases.append(["A", "B", "A"])
    repeated = {}
    for labels in cases:
        ids = [prompts[label]["token_ids"] for label in labels]
        lengths = list(map(len, ids))
        assert all(length % 64 and length % 32 for length in lengths)
        body = {
            "model": args.model,
            "prompt": ids,
            "max_tokens": 8,
            "temperature": 0.0,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        response = requests.post(args.url.rstrip("/") + "/v1/completions", json=body, timeout=300)
        raw = response.json()
        case = {"labels": labels, "logical_lengths": lengths, "request": body, "response": raw}
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        assert response.status_code == 200, raw
        assert raw["usage"]["prompt_tokens"] == sum(lengths)
        assert raw["usage"]["completion_tokens"] == 8 * len(labels)
        choices = sorted(raw["choices"], key=lambda choice: choice["index"])
        assert len(choices) == len(labels)
        for label, choice in zip(labels, choices):
            assert len(choice["token_ids"]) == 8
            reference = next(
                [step["token_id"] for step in signature["steps"]]
                for entry in control["cases"]
                for name, signature in zip(entry["labels"], entry["signatures"])
                if name == label
            )
            if not args.reduced:
                assert choice["token_ids"][: len(reference)] == reference
            if label in repeated:
                assert choice["token_ids"] == repeated[label]
            repeated[label] = choice["token_ids"]
        case["matches_control_token_prefix"] = None if args.reduced else True
        print(json.dumps({"lengths": lengths, "texts": [choice["text"] for choice in choices]}), flush=True)
    result["passed"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

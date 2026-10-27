# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Non-aligned prompt lengths through the served OpenAI API on p150: exact usage, in-vocabulary ids, repeat agreement."""

import argparse
import bz2
import json
import threading
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="ornith-ai/Ornith-1.5-9B")
    parser.add_argument("--tokenizer", default="/home/hous/dev/ornith-1.5-9b/upstream")
    parser.add_argument("--text", type=Path, default=Path("models/tt_transformers/tests/tale-of-two-cities.txt.bz2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    book = tokenizer(bz2.open(args.text, "rt").read()[:300000], add_special_tokens=False)["input_ids"]
    vocab = tokenizer.vocab_size

    def post(ids, max_tokens=12):
        body = {
            "model": args.model,
            "prompt": ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        response = requests.post(args.url.rstrip("/") + "/v1/completions", json=body, timeout=600)
        raw = response.json()
        return response.status_code, raw

    lengths = {"A": 131, "B": 65, "C": 2305, "D": 129, "E": 33}
    prompts = {label: book[3000 + n : 3000 + 2 * n] for label, n in lengths.items()}
    result = {
        "model": args.model,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": 262144,
        "lengths": lengths,
        "cases": [],
    }
    greedy = {}

    def check(labels, status, raw):
        assert status == 200, raw
        assert raw["usage"]["prompt_tokens"] == sum(lengths[l] for l in labels), raw["usage"]
        choices = sorted(raw["choices"], key=lambda c: c["index"])
        assert len(choices) == len(labels)
        for label, choice in zip(labels, choices):
            ids = choice["token_ids"]
            assert len(ids) == 12 and all(0 <= t < vocab for t in ids), ids
            greedy.setdefault(label, ids)
            assert greedy[label] == ids, (label, greedy[label], ids)
        return [
            {"label": l, "logical_length": lengths[l], "token_ids": c["token_ids"], "text": c["text"]}
            for l, c in zip(labels, choices)
        ]

    for label in ("A", "B", "C", "D", "E"):
        status, raw = post(prompts[label])
        result["cases"].append({"kind": "single", "labels": [label], "rows": check([label], status, raw)})
    status, raw = post([prompts[l] for l in ("A", "B", "A")])
    result["cases"].append(
        {"kind": "batched_prompt_list", "labels": ["A", "B", "A"], "rows": check(["A", "B", "A"], status, raw)}
    )
    labels = ["A", "B", "C", "D", "E", "A"][: max(2, min(6, args.max_num_seqs))]
    outputs = [None] * len(labels)

    def work(i, label):
        outputs[i] = post(prompts[label])

    threads = [threading.Thread(target=work, args=(i, l)) for i, l in enumerate(labels)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = []
    for label, (status, raw) in zip(labels, outputs):
        rows.extend(check([label], status, raw))
    result["cases"].append({"kind": "concurrent_requests", "labels": labels, "rows": rows})
    messages = [
        {"role": "user", "content": "Write one sentence about why astronomers record the time of an observation."}
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    body = {
        "model": args.model,
        "messages": messages,
        "max_tokens": 48,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(args.url.rstrip("/") + "/v1/chat/completions", json=body, timeout=600)
    raw = response.json()
    assert response.status_code == 200, raw
    result["cases"].append(
        {
            "kind": "chat_nonaligned",
            "rendered_prompt_tokens": len(rendered),
            "usage": raw["usage"],
            "text": raw["choices"][0]["message"]["content"],
        }
    )
    assert raw["usage"]["prompt_tokens"] % 32, raw["usage"]
    result["cases"][-1]["local_render_matches_server"] = raw["usage"]["prompt_tokens"] == len(rendered)
    result["passed"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "cases": [(c["kind"], c.get("labels")) for c in result["cases"]],
                "chat": result["cases"][-1]["text"][:200],
            }
        )
    )


if __name__ == "__main__":
    main()

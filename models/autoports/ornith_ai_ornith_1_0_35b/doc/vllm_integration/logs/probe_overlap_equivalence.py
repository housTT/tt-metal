# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does `--async-scheduling` change what the model emits?

The async split lets vLLM submit decode step *N+1* before token *N* has been applied to its host state,
so the adapter must keep a continuing row on the device's own token/position pair instead of the host's
lagging one. The mechanical proof of that merge is ``serving_primitives.json``'s ``stale_inputs`` arms;
this probe is the serving-level check: the same greedy request, on a server with overlap on and a server
with it off, must emit the same text.

Run it against each server in turn and pass the other's file, or run it twice with ``--tag`` and compare
afterwards:

    python .../logs/probe_overlap_equivalence.py --tag sync  --url http://localhost:8100
    # relaunch the server with --additional-server-args="--async-scheduling"
    python .../logs/probe_overlap_equivalence.py --tag async --url http://localhost:8100 \
        --compare-with <the sync file>

Both arms request the same number of tokens, so the comparison is over the whole completion rather than
over a prefix. Repeats within one arm are compared too: at ``max_num_seqs=1`` this model is
bit-reproducible (``slot_reproducibility.json``), so a difference *within* an arm would mean the overlap
had introduced nondeterminism rather than a different-but-equal answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request as urlrequest

MODEL_DIR = Path(__file__).resolve().parents[3]
PROMPT = "The capital of France is"


def complete(url, model, prompt, *, max_tokens, temperature=0.0, seed=None, top_p=1.0):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if seed is not None:
        payload["seed"] = seed
    req = urlrequest.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlrequest.urlopen(req, timeout=900) as response:
        body = json.load(response)
    return body["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8100")
    ap.add_argument("--model", default="ornith-ai/Ornith-1.0-35B")
    ap.add_argument("--tag", required=True, help="sync | async — what this server is")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--compare-with", default=None, help="a file written by the other arm")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    out_path = Path(
        args.output or (MODEL_DIR / "doc" / "vllm_integration" / "async" / f"overlap_texts_{args.tag}.json")
    )

    greedy = [complete(args.url, args.model, PROMPT, max_tokens=args.max_tokens) for _ in range(args.repeats)]
    seeded = [
        complete(args.url, args.model, PROMPT, max_tokens=args.max_tokens, temperature=0.8, top_p=0.9, seed=4242)
        for _ in range(2)
    ]
    report = {
        "tag": args.tag,
        "url": args.url,
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "greedy_texts": greedy,
        "greedy_identical_within_arm": len(set(greedy)) == 1,
        "seeded_texts": seeded,
        "seeded_identical_within_arm": len(set(seeded)) == 1,
    }
    if args.compare_with:
        other = json.loads(Path(args.compare_with).read_text())
        report["compared_with"] = {
            "file": args.compare_with,
            "tag": other.get("tag"),
            "max_tokens": other.get("max_tokens"),
            "same_length_request": other.get("max_tokens") == args.max_tokens,
            "greedy_identical_across_arms": other["greedy_texts"][0] == greedy[0],
            "seeded_identical_across_arms": other["seeded_texts"][0] == seeded[0],
            "other_greedy_text": other["greedy_texts"][0],
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("greedy_texts", "seeded_texts")}, indent=1))
    print("greedy:", repr(greedy[0]))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

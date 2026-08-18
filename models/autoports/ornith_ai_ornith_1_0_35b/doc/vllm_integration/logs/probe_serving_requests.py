# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serving-path request checks against a live vLLM server, with the controls that make them mean
something.

    python .../doc/vllm_integration/logs/probe_serving_requests.py --url http://localhost:8100

1. **single-user determinism** - the same greedy request repeated must produce the same text. This is
   the determinism a deployment sees at ``max_concurrency=1``, and it is the one the headline
   benchmark measures;
2. **non-aligned prompt lengths** - lengths that divide none of the internal sizes (64-token pages,
   the 128-token prefill block alignment, the 2048-token prefill chunk, the 32-row tile) must be
   accepted at their true length. Each length is asked twice, with and without ``ignore_eos``: the
   ``ignore_eos`` request is the length evidence (it must return every requested token), and the other
   records what the model does with a random-token prompt, which for the longer ones is to emit
   end-of-text immediately - a `finish_reason` of ``stop``, not a truncation. 130 is the interesting one: its KV write pads to 256 tokens, which
   is one 64-token page *past* what vLLM allocated for it, and that page is vLLM's reserved
   ``BlockPool.null_block`` (block id 0, which the block table pads with and no request ever owns);
3. **null-block containment** - a baseline greedy request is answered alone, then again while several
   130-token requests run beside it. Identical text is the evidence that the padded write lands
   somewhere no request reads;
4. **concurrency** - a burst of N requests all complete, and a request repeated inside the burst
   agrees with itself;
5. **long prompt** - a prompt well past the prefill chunk is served at its full length.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request as urlrequest

MODEL_DIR = Path(__file__).resolve().parents[3]
NON_ALIGNED = [1, 3, 17, 65, 130, 257, 999, 2049, 4097]


def post(url, payload, timeout=1800):
    req = urlrequest.Request(
        url + "/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urlrequest.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def complete(url, model, prompt, *, max_tokens=16, temperature=0.0, **extra):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    payload.update(extra)
    out = post(url, payload)
    choice = out["choices"][0]
    return choice["text"], out["usage"], choice.get("finish_reason")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8100")
    ap.add_argument("--model", default="ornith-ai/Ornith-1.0-35B")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument(
        "--server-label",
        required=True,
        help="which server this is, e.g. 'max_num_seqs=1, no async scheduling'. Recorded in the report: "
        "the arms below read the same over the API whatever the server's batch limit is, so a file "
        "without this label cannot be attributed to a configuration afterwards.",
    )
    ap.add_argument("--output", default=str(MODEL_DIR / "doc" / "vllm_integration" / "serving_requests.json"))
    args = ap.parse_args()
    url, model = args.url.rstrip("/"), args.model
    report: dict = {"url": url, "model": model, "server": args.server_label}
    with urlrequest.urlopen(url + "/v1/models", timeout=60) as response:
        served = json.load(response)["data"][0]
    report["served_model"] = {"id": served["id"], "max_model_len": served.get("max_model_len")}

    # ---------------------------------------------------------------- 1. single-user determinism
    baseline_prompt = "The capital of France is"
    repeats = [complete(url, model, baseline_prompt, max_tokens=24)[0] for _ in range(3)]
    report["single_user_determinism"] = {
        "prompt": baseline_prompt,
        "texts": repeats,
        "identical": len(set(repeats)) == 1,
    }

    # ---------------------------------------------------------------- 2. non-aligned prompt lengths
    lengths = {}
    for length in NON_ALIGNED:
        ids = [1000 + (i % 50000) for i in range(length)]
        # `ignore_eos` because this arm is about *lengths*, not content: these are random token ids, and
        # the model quite reasonably answers some of them with an immediate end-of-text. Without the flag
        # a short completion looks like a truncation bug when it is the model finishing.
        # `ignore_eos` goes at the top level of the JSON body: `extra_body` is an OpenAI *client* concept,
        # and a raw POST that nests it there just sends a field the server ignores (which is how the first
        # run of this arm came back identical to the arm it was supposed to contrast with).
        text, usage, finish = complete(url, model, ids, max_tokens=8, ignore_eos=True)
        eos_text, eos_usage, eos_finish = complete(url, model, ids, max_tokens=8)
        lengths[str(length)] = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "finish_reason": finish,
            "text": text,
            "length_preserved": usage["prompt_tokens"] == length,
            "without_ignore_eos": {
                "completion_tokens": eos_usage["completion_tokens"],
                "finish_reason": eos_finish,
                "text": eos_text,
            },
        }
    report["non_aligned_prompt_lengths"] = {
        "results": lengths,
        "all_lengths_preserved": all(v["length_preserved"] for v in lengths.values()),
        "all_completed": all(v["completion_tokens"] == 8 for v in lengths.values()),
        "short_completions_without_ignore_eos_are_eos_stops": {
            k: (v["without_ignore_eos"]["completion_tokens"] < 8, v["without_ignore_eos"]["finish_reason"])
            for k, v in lengths.items()
            if v["without_ignore_eos"]["completion_tokens"] < 8
        },
    }

    # ---------------------------------------------------------------- 3. null-block containment
    alone = complete(url, model, baseline_prompt, max_tokens=24)[0]
    neighbour_ids = [1000 + (i % 50000) for i in range(130)]

    def _neighbour(_):
        return complete(url, model, neighbour_ids, max_tokens=24)[0]

    with ThreadPoolExecutor(max_workers=9) as pool:
        futures = [pool.submit(_neighbour, i) for i in range(8)]
        beside = pool.submit(complete, url, model, baseline_prompt, max_tokens=24).result()[0]
        neighbours = [f.result() for f in futures]
    report["null_block_containment"] = {
        "baseline_alone": alone,
        "baseline_beside_eight_130_token_requests": beside,
        "identical": alone == beside,
        "neighbours_completed": len(neighbours),
    }

    # ---------------------------------------------------------------- 4. concurrency
    prompts = [f"Question {i}: name one fact about the number {i}." for i in range(args.concurrency)]
    prompts[-1] = prompts[0]  # the same request twice inside the burst
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        burst = list(pool.map(lambda p: complete(url, model, p, max_tokens=32)[0], prompts))
    report["concurrency"] = {
        "requests": args.concurrency,
        "all_completed": all(isinstance(text, str) and text for text in burst),
        "duplicate_request_agrees": burst[0] == burst[-1],
        "first_two": burst[:2],
    }

    # ---------------------------------------------------------------- 5. long prompt
    long_ids = [1000 + (i % 50000) for i in range(9000)]
    text, usage, finish = complete(url, model, long_ids, max_tokens=8, ignore_eos=True)
    report["long_prompt"] = {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "finish_reason": finish,
        "text": text,
    }

    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "non_aligned_prompt_lengths"}, indent=1))
    print("non-aligned:", json.dumps(report["non_aligned_prompt_lengths"]["results"], indent=1)[:400])


if __name__ == "__main__":
    main()

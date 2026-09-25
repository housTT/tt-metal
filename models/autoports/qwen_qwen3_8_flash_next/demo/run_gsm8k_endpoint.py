"""GSM8K exact-match against a served Qwen3.8-Flash-Next endpoint (OpenAI chat API).

C0 task metric: a fixed 100-item GSM8K test subset (``doc/correctness/gsm8k_100.jsonl``,
seeded sample of openai/grade-school-math test.jsonl) sent greedily through
``/v1/chat/completions`` with thinking disabled, scored by exact match of the final
number.  Resumable: finished items are kept in the output JSON and skipped on rerun.

    python -m models.autoports.qwen_qwen3_8_flash_next.demo.run_gsm8k_endpoint \
        --server-url http://127.0.0.1:20000 --output doc/correctness/gsm8k_endpoint_<tag>.json [--limit 50]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "doc/correctness/gsm8k_100.jsonl"
PROMPT_SUFFIX = "\n\nSolve the problem step by step, then give the final numeric answer on the last line as: Final answer: <number>"
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _final_number(text: str) -> str | None:
    tail = text.split("Final answer:")[-1] if "Final answer:" in text else text
    hits = _NUM.findall(tail)
    if not hits:
        return None
    value = hits[0 if "Final answer:" in text else -1].replace(",", "").rstrip(".")
    return value


def _same(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:20000")
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true", help="default off: instruct mode, bounded output")
    # Stochastic / degeneracy mode (DEVSTACK-294): sample instead of greedy, repeat one
    # prompt, and flag repetition collapse in every reply.
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None, help="explicit request seed (routes to the host sampler)")
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help="text file with one user prompt; replaces the GSM8K dataset (no scoring)")
    parser.add_argument("--repeats", type=int, default=1, help="send each prompt this many times")
    parser.add_argument("--tokenizer", default=None,
                        help="HF tokenizer path for the degeneracy review (default: the pinned snapshot)")
    parser.add_argument("--no-degeneracy", action="store_true", help="skip the token-level degeneracy review")
    args = parser.parse_args()

    if args.prompt_file is not None:
        prompt_text = args.prompt_file.read_text()
        items = [{"gsm8k_test_index": f"prompt{r}", "question": prompt_text, "final": None} for r in range(args.repeats)]
        dataset_sha = hashlib.sha256(args.prompt_file.read_bytes()).hexdigest()
        suffix = ""
    else:
        base = [json.loads(l) for l in args.dataset.read_text().splitlines() if l.strip()][: args.limit]
        items = [dict(item, gsm8k_test_index=f"{item['gsm8k_test_index']}" + (f"r{r}" if r else ""))
                 for item in base for r in range(args.repeats)]
        dataset_sha = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
        suffix = PROMPT_SUFFIX
    sampling = {"temperature": args.temperature}
    for key, value in (("top_p", args.top_p), ("top_k", args.top_k), ("presence_penalty", args.presence_penalty), ("seed", args.seed)):
        if value is not None:
            sampling[key] = value
    review = None
    if not args.no_degeneracy:
        from transformers import AutoTokenizer

        from models.autoports.qwen_qwen3_8_flash_next.demo.full_model import _degeneracy
        from models.autoports.qwen_qwen3_8_flash_next.tt.model import DEFAULT_SNAPSHOT

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or DEFAULT_SNAPSHOT, local_files_only=True)

        def review(text: str) -> dict:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            return _degeneracy(ids, text)

    result = {"metadata": {"server_url": args.server_url, "model": args.model, "dataset": str(args.prompt_file or args.dataset),
                           "dataset_sha256": dataset_sha, "max_tokens": args.max_tokens, **sampling,
                           "repeats": args.repeats, "enable_thinking": bool(args.enable_thinking), "prompt_suffix": suffix},
              "items": {}}
    if args.output.is_file():
        prior = json.loads(args.output.read_text())
        if prior.get("metadata", {}).get("dataset_sha256") == dataset_sha:
            result["items"] = prior.get("items", {})

    endpoint = f"{args.server_url.rstrip('/')}/v1/chat/completions"
    with httpx.Client(timeout=1800.0) as client:
        for item in items:
            key = str(item["gsm8k_test_index"])
            if key in result["items"]:
                continue
            body = {"model": args.model, **sampling, "max_tokens": args.max_tokens,
                    "messages": [{"role": "user", "content": item["question"] + suffix}],
                    "chat_template_kwargs": {"enable_thinking": bool(args.enable_thinking)}}
            started = time.perf_counter()
            response = client.post(endpoint, json=body)
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            text = choice["message"].get("content") or ""
            reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning") or ""
            predicted = _final_number(text) if item["final"] is not None else None
            correct = _same(predicted, item["final"]) if item["final"] is not None else None
            result["items"][key] = {"predicted": predicted, "expected": item["final"], "correct": correct,
                                    "finish_reason": choice.get("finish_reason"),
                                    "completion_tokens": (payload.get("usage") or {}).get("completion_tokens"),
                                    "empty_content": text.strip() == "",
                                    "seconds": time.perf_counter() - started, "text_tail": text[-300:],
                                    "reasoning_tail": reasoning[-300:]}
            if review is not None:
                result["items"][key]["degeneracy"] = review(reasoning + text)
            done = len(result["items"]); acc = sum(bool(v["correct"]) for v in result["items"].values())
            print(f"[gsm8k] {done}/{len(items)} idx={key} correct={correct} pred={predicted} exp={item['final']} "
                  f"finish={choice.get('finish_reason')} {result['items'][key]['seconds']:.1f}s running_acc={acc}/{done}", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=1))
    scored = [v for k, v in result["items"].items() if any(str(i["gsm8k_test_index"]) == k for i in items)]
    acc = sum(bool(v["correct"]) for v in scored) / max(1, len(scored))
    result["summary"] = {"items": len(scored), "accuracy": acc if items and items[0]["final"] is not None else None,
                         "length_truncated": sum(v["finish_reason"] == "length" for v in scored),
                         "empty_content_rate": sum(v.get("empty_content", False) for v in scored) / max(1, len(scored)),
                         "degenerate_rate": (sum(v["degeneracy"]["mechanically_degenerate"] for v in scored if "degeneracy" in v)
                                             / max(1, len(scored))) if review is not None else None,
                         "mean_seconds": sum(v["seconds"] for v in scored) / max(1, len(scored))}
    args.output.write_text(json.dumps(result, indent=1))
    print({"gsm8k_endpoint": result["summary"]})


if __name__ == "__main__":
    main()

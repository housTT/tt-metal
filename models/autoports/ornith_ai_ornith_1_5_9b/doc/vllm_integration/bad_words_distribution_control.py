"""Parent-serialized bad-word distribution/quality investigation, no TTNN import.

This is an explicit host-sampling diagnostic. It is not a serving benchmark or
an automatic qualitative pass/fail judgment.
"""

import argparse
import ast
import asyncio
import hashlib
import json
import math
import runpy
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import openai
import torch
from transformers import AutoTokenizer

BAD_WORDS = ["hello", "Hello", "hi", "Hi", "hey", "Hey"]


def pinned_bad_word_policy(vllm_root, tokenizer):
    source = vllm_root / "vllm/sampling_params.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "SamplingParams"
    )
    update = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "update_from_tokenizer"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, update], type_ignores=[])), str(source), "exec"),
        namespace,
    )
    params = SimpleNamespace(bad_words=BAD_WORDS)
    namespace["update_from_tokenizer"](
        params,
        SimpleNamespace(encode=tokenizer.encode, max_token_id=max(tokenizer.get_vocab().values())),
    )
    mask_source = vllm_root / "vllm/v1/sample/ops/bad_words.py"
    apply = runpy.run_path(str(mask_source))["apply_bad_words"]
    sampler_source = vllm_root / "vllm/v1/sample/sampler.py"
    runner_source = vllm_root / "plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py"
    provenance = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source, mask_source, sampler_source, runner_source)
    }
    return params._bad_words_token_ids, apply, provenance


def token_id(label):
    assert label.startswith("token_id:"), f"Expected actual token-ID logprob metadata: {label!r}"
    return int(label.removeprefix("token_id:"))


def chat_step(raw, index):
    choice = raw["choices"][0]
    content = (choice.get("logprobs") or {}).get("content") or []
    if index >= len(content):
        return {"available": False, "reason": "step not present in chat logprobs"}
    entry = content[index]
    assert token_id(entry["token"]) == choice["token_ids"][index]
    return {
        "available": True,
        "sampled_token_id": token_id(entry["token"]),
        "sampled_raw_logprob": entry["logprob"],
        "top_raw_logprobs": {str(token_id(item["token"])): item["logprob"] for item in entry["top_logprobs"]},
    }


async def probe(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    sequences, apply_bad_words, hashes = pinned_bad_word_policy(args.vllm_root, tokenizer)
    assert all(len(sequence) == 1 for sequence in sequences), "This diagnostic expects the original singleton greetings"
    forbidden_ids = {sequence[0] for sequence in sequences}
    result = {
        "label": args.label,
        "model": args.model,
        "purpose": "same-seed host controls and first-divergence raw distribution; no automatic quality verdict",
        "sampling_path": "explicit optional host sampler for all requests via logprobs",
        "raw_logprobs_contract": "TTModelRunner constructs Sampler(); default raw_logprobs are computed before bad-word masking",
        "source_sha256": hashes,
        "bad_words": BAD_WORDS,
        "forbidden_token_sequences": sequences,
        "requests": [],
        "comparisons": [],
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    async with openai.AsyncOpenAI(base_url=f"{args.url.rstrip('/')}/v1", api_key="dummy", timeout=600.0) as client:

        async def collect(kind, seed, request, chat=True):
            request_id = f"{args.label}-seed{seed}-{kind}"
            request = {
                **request,
                "extra_body": {**request.get("extra_body", {}), "request_id": request_id},
            }
            endpoint = client.chat.completions if chat else client.completions
            raw = (await endpoint.create(**request)).model_dump()
            result["requests"].append(
                {"kind": kind, "seed": seed, "request_id": request_id, "request": request, "response": raw}
            )
            save()
            return raw

        for seed in (2, 3):
            shared = {
                "model": args.model,
                "messages": [{"role": "user", "content": "Say hello to me"}],
                "max_tokens": 100,
                "temperature": 1.0,
                "top_p": 1.0,
                "seed": seed,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "logprobs": True,
                "top_logprobs": 20,
                "extra_body": {"return_token_ids": True, "return_tokens_as_token_ids": True},
            }
            unbanned = await collect("unbanned", seed, shared)
            banned = await collect(
                "banned", seed, {**shared, "extra_body": {**shared["extra_body"], "bad_words": BAD_WORDS}}
            )
            unbanned_ids = unbanned["choices"][0]["token_ids"]
            banned_ids = banned["choices"][0]["token_ids"]
            prompt_ids = unbanned["prompt_token_ids"]
            assert (
                prompt_ids == banned["prompt_token_ids"]
            ), "The two controls must use the exact same rendered prompt IDs"
            assert unbanned_ids and banned_ids and prompt_ids
            common_length = 0
            for left, right in zip(unbanned_ids, banned_ids):
                if left != right:
                    break
                common_length += 1
            comparison = {
                "seed": seed,
                "unbanned_response_id": unbanned["id"],
                "banned_response_id": banned["id"],
                "unbanned_text": unbanned["choices"][0]["message"]["content"],
                "banned_text": banned["choices"][0]["message"]["content"],
                "common_generated_prefix_ids": unbanned_ids[:common_length],
                "banned_output_contains_forbidden_id": bool(forbidden_ids.intersection(banned_ids)),
            }
            if common_length == min(len(unbanned_ids), len(banned_ids)):
                comparison["first_divergence"] = None
                comparison[
                    "limitation"
                ] = "No differing token before one output ended; no forced divergence or retry performed"
                result["comparisons"].append(comparison)
                save()
                continue
            prefix_ids = prompt_ids + unbanned_ids[:common_length]
            comparison["first_divergence"] = {
                "output_index": common_length,
                "unbanned_id": unbanned_ids[common_length],
                "unbanned_piece": tokenizer.decode([unbanned_ids[common_length]]),
                "banned_id": banned_ids[common_length],
                "banned_piece": tokenizer.decode([banned_ids[common_length]]),
                "unbanned_divergent_id_is_forbidden": unbanned_ids[common_length] in forbidden_ids,
                "unbanned_step": chat_step(unbanned, common_length),
                "banned_step": chat_step(banned, common_length),
            }
            neutral = await collect(
                "neutral-prefix",
                seed,
                {
                    "model": args.model,
                    "prompt": prefix_ids,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": seed,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "logprobs": 20,
                    "extra_body": {"return_token_ids": True, "return_tokens_as_token_ids": True},
                },
                chat=False,
            )
            raw_top = {
                token_id(label): value for label, value in neutral["choices"][0]["logprobs"]["top_logprobs"][0].items()
            }
            vocab_extent = max(max(tokenizer.get_vocab().values()), max(raw_top), max(forbidden_ids)) + 1
            logits = torch.zeros((1, vocab_extent), dtype=torch.float32)
            for token, value in raw_top.items():
                logits[0, token] = value
            before = logits.clone()
            apply_bad_words(logits, {0: sequences}, [unbanned_ids[:common_length]])
            observed = [
                {
                    "token_id": token,
                    "piece": tokenizer.decode([token]),
                    "raw_logprob": value,
                    "raw_probability": math.exp(value),
                    "forbidden": token in forbidden_ids,
                    "masked_to_negative_infinity": bool(torch.isneginf(logits[0, token])),
                }
                for token, value in sorted(raw_top.items(), key=lambda item: item[1], reverse=True)
            ]
            comparison["neutral_exact_prefix_control"] = {
                "response_id": neutral["id"],
                "exact_prompt_ids": prefix_ids,
                "prefix_text_for_inspection": tokenizer.decode(prefix_ids, skip_special_tokens=False),
                "observed_top_raw_distribution": observed,
                "forbidden_probability_mass_lower_bound": sum(
                    item["raw_probability"] for item in observed if item["forbidden"]
                ),
                "all_forbidden_ids_masked_by_actual_pinned_processor": all(
                    bool(torch.isneginf(logits[0, token])) for token in forbidden_ids
                ),
                "observed_unbanned_logits_preserved_by_processor": all(
                    bool(logits[0, token] == before[0, token]) for token in raw_top if token not in forbidden_ids
                ),
                "limitations": "Top-20 distribution only; removed mass is a lower bound. Fresh exact-prefix prefill may differ numerically from original decode. CPU mask control is not an autoregressive sampling oracle.",
            }
            result["comparisons"].append(comparison)
            save()
    print(
        json.dumps(
            [
                {
                    "seed": comparison["seed"],
                    "first_divergence": comparison["first_divergence"],
                    "forbidden_probability_mass_lower_bound": comparison.get("neutral_exact_prefix_control", {}).get(
                        "forbidden_probability_mass_lower_bound"
                    ),
                }
                for comparison in result["comparisons"]
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/home/hous/dev/ornith-1.5-9b/upstream")
    parser.add_argument("--vllm-root", type=Path, default=Path("../vllm"))
    parser.add_argument("--label", default=f"bad-words-dist-{uuid4().hex[:10]}")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(probe(parser.parse_args()))

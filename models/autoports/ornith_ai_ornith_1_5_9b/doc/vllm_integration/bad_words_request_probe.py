"""Repeat the original bad-words gate while preserving generated token IDs.

Run only under the parent agent's serialized ownership of the active server.
"""

import argparse
import asyncio
import json
import string
from pathlib import Path

import openai
from transformers import AutoTokenizer


async def probe(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bad_words = ["hello", "Hello", "hi", "Hi", "hey", "Hey"]
    banned_sequences = [
        tokenizer.encode(prefix + word, add_special_tokens=False) for word in bad_words for prefix in ("", " ")
    ]
    assert all(len(seq) == 1 for seq in banned_sequences)
    banned_ids = {seq[0] for seq in banned_sequences}
    async with openai.AsyncOpenAI(base_url=f"{args.url.rstrip('/')}/v1", api_key="dummy", timeout=600.0) as client:
        responses = await asyncio.gather(
            *[
                client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": "Say hello to me"}],
                    max_tokens=100,
                    temperature=1.0,
                    top_p=1.0,
                    seed=seed,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    logprobs=False,
                    top_logprobs=None,
                    extra_body={"bad_words": bad_words, "return_token_ids": True},
                )
                for seed in range(5)
            ]
        )
    cases = []
    for seed, response in enumerate(responses):
        raw = response.model_dump()
        choice = raw["choices"][0]
        token_ids = choice.get("token_ids")
        text = choice["message"].get("content")
        words = [word.strip(string.punctuation.replace(">", "")) for word in text.split()] if text is not None else []
        cases.append(
            {
                "seed": seed,
                "raw_response": raw,
                "original_text_assertion_failures": [w for w in bad_words if w in words],
                "generated_token_ids_available": token_ids is not None,
                "banned_generated_ids": ([i for i in token_ids if i in banned_ids] if token_ids is not None else None),
                "generated_token_pieces": (
                    [tokenizer.decode([i]) for i in token_ids] if token_ids is not None else None
                ),
            }
        )
    result = {
        "model": args.model,
        "probe": "original five requests plus return_token_ids response metadata",
        "banned_sequences": banned_sequences,
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            [{k: c[k] for k in ("seed", "original_text_assertion_failures", "banned_generated_ids")} for c in cases],
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/home/hous/dev/ornith-1.5-9b/upstream")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(probe(parser.parse_args()))

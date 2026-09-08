"""One parent-serialized original seven-request first-token diagnostic."""

import argparse
import asyncio
import json
from pathlib import Path

import openai
from transformers import AutoTokenizer


async def probe(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    request = {
        "model": args.model,
        "prompt": "Random letter: ",
        "max_tokens": 10,
        "temperature": 5.0,
        "top_p": 1.0,
        "seed": None,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "logprobs": None,
        "extra_body": {"return_token_ids": True},
    }
    async with openai.AsyncOpenAI(base_url=f"{args.url.rstrip('/')}/v1", api_key="dummy", timeout=600.0) as client:
        responses = await asyncio.gather(*[client.completions.create(**request) for _ in range(7)])
    raw = [response.model_dump() for response in responses]
    choices = [response["choices"][0] for response in raw]
    texts = [choice["text"] for choice in choices]
    token_ids = [choice.get("token_ids") for choice in choices]
    assert len(raw) == 7 and all(texts), "expected seven nonempty completions"
    assert all(token_ids), "actual generated token-ID metadata is required"
    first_ids = [ids[0] for ids in token_ids]
    first_characters = [text[:1] for text in texts]
    result = {
        "request": request,
        "batch_size": 7,
        "original_first_character_values": first_characters,
        "actual_first_token_ids": first_ids,
        "actual_first_token_pieces": [tokenizer.decode([token]) for token in first_ids],
        "full_output_variety_passes": len(set(texts)) >= 2,
        "first_character_variety_passes": len(set(first_characters)) >= 2,
        "actual_first_token_variety_passes": len(set(first_ids)) >= 2,
        "raw_responses": raw,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "raw_responses"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/home/hous/dev/ornith-1.5-9b/upstream")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(probe(parser.parse_args()))

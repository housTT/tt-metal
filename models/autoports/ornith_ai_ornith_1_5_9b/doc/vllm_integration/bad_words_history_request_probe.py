"""Parent-serialized API control for neutral-penalty multi-token bad words."""

import argparse
import asyncio
import json
from pathlib import Path

import openai
from transformers import AutoTokenizer


def contains(tokens, sequences):
    return any(
        tokens[start : start + len(sequence)] == sequence
        for sequence in sequences
        for start in range(len(tokens) - len(sequence) + 1)
    )


async def probe(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    records = []
    selected = None
    async with openai.AsyncOpenAI(base_url=f"{args.url.rstrip('/')}/v1", api_key="dummy", timeout=600.0) as client:
        for phrase in ("New York", "machine learning"):
            sequences = [tokenizer.encode(prefix + phrase, add_special_tokens=False) for prefix in ("", " ")]
            assert all(len(sequence) == 2 for sequence in sequences)
            allowed_ids = sorted({token for sequence in sequences for token in sequence})
            for first_bias in (100.0, 98.0, 95.0):
                biases = {str(token): 100.0 for token in allowed_ids}
                for sequence in sequences:
                    biases[str(sequence[0])] = first_bias
                request = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": f"Reply with exactly these two words: {phrase}"}],
                    "max_tokens": 6,
                    "temperature": 0.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "extra_body": {
                        "return_token_ids": True,
                        "allowed_token_ids": allowed_ids,
                        "logit_bias": biases,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                }
                control = (await client.chat.completions.create(**request)).model_dump()
                ids = control["choices"][0]["token_ids"]
                records.append({"kind": "unbanned_control", "request": request, "response": control})
                if contains(ids, sequences):
                    selected = (phrase, sequences, request)
                    break
            if selected is not None:
                break
        if selected is None:
            result = {"status": "inconclusive_no_reachable_control", "records": records}
            code = 2
        else:
            phrase, sequences, request = selected
            blocked_request = {
                **request,
                "extra_body": {**request["extra_body"], "bad_words": [phrase]},
            }
            blocked = (await client.chat.completions.create(**blocked_request)).model_dump()
            blocked_ids = blocked["choices"][0]["token_ids"]
            records.append({"kind": "banned_phrase", "request": blocked_request, "response": blocked})
            # The final token is still legal without the preceding generated
            # prefix: multi-token blocking must not become a singleton ban.
            final_id = sequences[0][-1]
            standalone_request = {
                **request,
                "max_tokens": 1,
                "extra_body": {
                    **request["extra_body"],
                    "bad_words": [phrase],
                    "allowed_token_ids": [final_id],
                    "logit_bias": {str(final_id): 100.0},
                },
            }
            standalone = (await client.chat.completions.create(**standalone_request)).model_dump()
            records.append({"kind": "final_token_alone", "request": standalone_request, "response": standalone})
            standalone_ids = standalone["choices"][0]["token_ids"]
            banned_sequence_generated = contains(blocked_ids, sequences)
            final_allowed = standalone_ids == [final_id]
            code = int(banned_sequence_generated or not final_allowed)
            result = {
                "status": "pass" if code == 0 else "fail",
                "phrase": phrase,
                "banned_sequences": sequences,
                "banned_sequence_generated": banned_sequence_generated,
                "final_token_allowed_alone": final_allowed,
                "records": records,
            }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/home/hous/dev/ornith-1.5-9b/upstream")
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(asyncio.run(probe(parser.parse_args())))

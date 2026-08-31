# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check exact greedy token determinism through vLLM at B1 and B32."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import torch

MODEL_ID = "openai/gpt-oss-120b"
MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"


def _sha256_tokens(tokens: list[int]) -> str:
    payload = torch.tensor(tokens, dtype=torch.int64).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


async def _request_one(
    client: httpx.AsyncClient, server_url: str, prompt_tokens: list[int], output_tokens: int
) -> dict:
    response = await client.post(
        f"{server_url.rstrip('/')}/v1/completions",
        json={
            "model": MODEL_ID,
            "prompt": prompt_tokens,
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "return_token_ids": True,
        },
    )
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    token_ids = [int(token) for token in choice["token_ids"]]
    if len(token_ids) != output_tokens:
        raise RuntimeError(f"expected {output_tokens} generated tokens, got {len(token_ids)}")
    return {
        "http_status": response.status_code,
        "response_id": body.get("id"),
        "token_ids": token_ids,
        "token_sha256": _sha256_tokens(token_ids),
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
    }


async def _request_batch(server_url: str, prompt_tokens: list[int], output_tokens: int, batch_size: int) -> list[dict]:
    timeout = httpx.Timeout(1800.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await asyncio.gather(
            *(_request_one(client, server_url, prompt_tokens, output_tokens) for _ in range(batch_size))
        )


def main() -> None:
    script = Path(__file__).resolve()
    model_dir = script.parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--hf-reference",
        type=Path,
        default=model_dir / "doc" / "full_model" / "references" / "aime24_chat_100_top100.refpt",
    )
    parser.add_argument(
        "--standalone-tt-logit-reference",
        type=Path,
        default=model_dir / "doc" / "optimized_full_model" / "artifacts" / "batch2_logit_reproducibility.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=model_dir / "readiness_vllm" / "batch32_greedy_determinism.json",
    )
    args = parser.parse_args()
    if args.batch_size != 32:
        raise ValueError("the serving-width determinism gate requires batch-size=32")

    reference = torch.load(args.hf_reference, map_location="cpu", weights_only=False)
    entry = reference["entries"][0]
    prompt_tokens = [int(token) for token in entry["prompt_tokens"].reshape(-1).tolist()]
    hf_tokens = [int(token) for token in entry["generated_tokens"].reshape(-1).tolist()][: args.output_tokens]
    if len(hf_tokens) != args.output_tokens:
        raise RuntimeError("HF reference does not contain enough generated tokens")

    b1_runs = [asyncio.run(_request_batch(args.server_url, prompt_tokens, args.output_tokens, 1))[0] for _ in range(2)]
    b32_runs = [
        asyncio.run(_request_batch(args.server_url, prompt_tokens, args.output_tokens, args.batch_size))
        for _ in range(2)
    ]
    sequences = [run["token_ids"] for run in b1_runs]
    sequences.extend(row["token_ids"] for run in b32_runs for row in run)
    all_equal = all(tokens == hf_tokens for tokens in sequences)
    standalone_tt = json.loads(args.standalone_tt_logit_reference.read_text(encoding="utf-8"))
    standalone_tt_argmax = [
        int(standalone_tt["comparisons"]["prefill"]["argmax_tokens"][0][0]),
        int(standalone_tt["comparisons"]["decode"]["argmax_tokens"][0][0]),
    ]
    prompt_sha256 = _sha256_tokens(prompt_tokens)
    standalone_tt_match = standalone_tt["prompt_sha256"] == prompt_sha256 and all(
        tokens[:2] == standalone_tt_argmax for tokens in sequences
    )
    artifact = {
        "status": "pass" if all_equal and standalone_tt_match else "fail",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "endpoint": "/v1/completions",
        "sampling": {"temperature": 0.0, "ignore_eos": True},
        "prompt_tokens": len(prompt_tokens),
        "prompt_sha256": prompt_sha256,
        "output_tokens": args.output_tokens,
        "hf_reference": str(args.hf_reference.resolve()),
        "hf_token_ids": hf_tokens,
        "hf_token_sha256": _sha256_tokens(hf_tokens),
        "standalone_tt_logit_reference": str(args.standalone_tt_logit_reference.resolve()),
        "standalone_tt_first_two_argmax_tokens": standalone_tt_argmax,
        "standalone_tt_full_logits_exact_across_two_rows_and_runs": {
            "prefill": standalone_tt["comparisons"]["prefill"]["max_abs_diff_between_rows"] == 0.0
            and standalone_tt["comparisons"]["prefill"]["max_abs_diff_between_runs"] == 0.0,
            "decode": standalone_tt["comparisons"]["decode"]["max_abs_diff_between_rows"] == 0.0
            and standalone_tt["comparisons"]["decode"]["max_abs_diff_between_runs"] == 0.0,
        },
        "b1_runs": b1_runs,
        "b32_runs": b32_runs,
        "same_prompt_run_to_run_exact": b1_runs[0]["token_ids"] == b1_runs[1]["token_ids"],
        "each_b32_run_cross_position_exact": [len({row["token_sha256"] for row in run}) == 1 for run in b32_runs],
        "b32_run_to_run_exact": [row["token_ids"] for row in b32_runs[0]] == [row["token_ids"] for row in b32_runs[1]],
        "all_vllm_sequences_match_hf_reference": all_equal,
        "all_vllm_first_two_tokens_match_standalone_tt_argmax": standalone_tt_match,
        "scope": (
            "The HF artifact supplies the exact 32-token greedy reference. The TT standalone artifact uses the same "
            "prompt SHA and supplies the first prefill/decode argmax tokens plus independent cross-row/run full-logit "
            "SHA equality. The shared vLLM API does not expose raw full logits."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    if not all_equal or not standalone_tt_match:
        raise RuntimeError(f"vLLM greedy tokens did not match the HF/standalone-TT references; wrote {args.output}")
    print(
        f"PASS: 2xB1 and 2xB{args.batch_size} matched the exact HF {args.output_tokens}-token sequence "
        "and the standalone-TT prefill/decode argmax tokens; "
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()

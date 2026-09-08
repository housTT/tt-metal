# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Check numeric logit reproducibility through a live Gemma 4 vLLM server."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import requests

PROMPT = "Complete this factual sentence with only a few words: The capital of France is"
TOP_LOGPROBS = 10
MAX_TOKENS = 4
MAX_SELECTED_LOGPROB_DELTA = 1.0e-3
MAX_TOP10_TAIL_LOGPROB_DELTA = 1.0
MIN_TOP10_TOKEN_OVERLAP = 9


def _request(server_url: str, model: str, barrier: threading.Barrier | None = None) -> dict[str, Any]:
    if barrier is not None:
        barrier.wait()
    response = requests.post(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_k": 1,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            "return_tokens_as_token_ids": True,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def _completion_request(
    server_url: str,
    model: str,
    prompt_token_ids: list[int],
    barrier: threading.Barrier | None = None,
) -> dict[str, Any]:
    """Follow the exact two-token greedy trajectory retained by the B1 oracle."""
    if barrier is not None:
        barrier.wait()
    response = requests.post(
        f"{server_url.rstrip('/')}/v1/completions",
        json={
            "model": model,
            "prompt": prompt_token_ids,
            "max_tokens": 2,
            "temperature": 0.0,
            "top_k": 1,
            "ignore_eos": True,
            "logprobs": TOP_LOGPROBS,
            "return_tokens_as_token_ids": True,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def _compact(response: dict[str, Any]) -> dict[str, Any]:
    choice = response["choices"][0]
    positions = []
    for position in choice["logprobs"]["content"]:
        positions.append(
            {
                "selected_token": position["token"],
                "selected_logprob": position["logprob"],
                "top_logprobs": {entry["token"]: entry["logprob"] for entry in position["top_logprobs"]},
            }
        )
    return {"text": choice["message"]["content"], "positions": positions}


def _compact_completion(response: dict[str, Any]) -> dict[str, Any]:
    choice = response["choices"][0]
    logprobs = choice["logprobs"]
    return {
        "text": choice["text"],
        "positions": [
            {
                "selected_token": token,
                "selected_logprob": token_logprob,
                "top_logprobs": top_logprobs,
            }
            for token, token_logprob, top_logprobs in zip(
                logprobs["tokens"],
                logprobs["token_logprobs"],
                logprobs["top_logprobs"],
                strict=True,
            )
        ],
    }


def _compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    selected_match = [
        left["selected_token"] == right["selected_token"]
        for left, right in zip(reference["positions"], candidate["positions"], strict=True)
    ]
    overlaps = []
    deltas = []
    selected_logprob_deltas = []
    for left, right in zip(reference["positions"], candidate["positions"], strict=True):
        common = set(left["top_logprobs"]) & set(right["top_logprobs"])
        overlaps.append(len(common))
        deltas.extend(abs(left["top_logprobs"][token] - right["top_logprobs"][token]) for token in common)
        selected_logprob_deltas.append(abs(left["selected_logprob"] - right["selected_logprob"]))
    return {
        "text_exact_match": reference["text"] == candidate["text"],
        "selected_tokens_match": all(selected_match),
        "selected_token_match_by_position": selected_match,
        "maximum_selected_logprob_absolute_delta": max(selected_logprob_deltas),
        "minimum_top10_token_overlap": min(overlaps),
        "maximum_common_logprob_absolute_delta": max(deltas),
    }


def _load_standalone_baseline(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload["decode_input_token_id_by_group"] != payload["prefill_oracle_token_by_group"]:
        raise ValueError("standalone decode trajectory does not follow its prefill argmax tokens")
    return {
        "artifact": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "profile": payload["profile"],
        "real_weights": payload["real_weights"],
        "checkpoint_revision": payload["checkpoint_revision"],
        "mesh_shape": payload["mesh_shape"],
        "tp_size": payload["tp_size"],
        "verdict": payload["verdict"],
        "prompt_len": payload["prompt_len"],
        "evidence_kind": payload["evidence_kind"],
        "prompt_token_ids_by_group": payload["prompt_token_ids_by_group"],
        "prefill_oracle_token_by_group": payload["prefill_oracle_token_by_group"],
        "prefill_oracle_top10_by_group": payload["prefill_oracle_top10_by_group"],
        "decode_input_token_id_by_group": payload["decode_input_token_id_by_group"],
        "decode_oracle_token_by_group": payload["decode_oracle_token_by_group"],
        "decode_oracle_top10_by_group": payload["decode_oracle_top10_by_group"],
    }


def _load_standalone_batch_control(path: Path, checkpoint_revision: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    control = payload.get("batch32_control")
    if control is None:
        raise ValueError("standalone batch control does not contain batch32_control evidence")
    if payload.get("checkpoint_revision") != checkpoint_revision:
        raise ValueError("standalone B1 and B32 controls use different checkpoint revisions")
    comparisons = control["comparisons_to_b1_by_row"]
    if (
        payload.get("verdict") != "pass"
        or not payload.get("real_weights")
        or payload.get("evidence_kind") != "full_model_b1_b32_logit_oracle"
        or len(comparisons) != 32
        or not all(comparison["selected_token_match"] for comparison in comparisons)
    ):
        raise ValueError("standalone B1/B32 selected-token correctness control did not pass")
    return {
        "artifact": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "profile": payload["profile"],
        "mesh_shape": payload["mesh_shape"],
        "checkpoint_revision": payload["checkpoint_revision"],
        "real_weights": payload["real_weights"],
        "verdict": payload["verdict"],
        "evidence_kind": payload["evidence_kind"],
        "correctness_gate": control["correctness_gate"],
        "selected_token_gate_pass": all(comparison["selected_token_match"] for comparison in comparisons),
        "minimum_logits_cosine": control["minimum_logits_cosine"],
        "minimum_centered_logits_cosine": control["minimum_centered_logits_cosine"],
        "maximum_probability_total_variation": control["maximum_probability_total_variation"],
        "maximum_selected_logprob_absolute_delta": control["maximum_selected_logprob_absolute_delta"],
        "minimum_top10_token_overlap": control["minimum_top10_token_overlap"],
        "maximum_common_top10_logprob_absolute_delta": control["maximum_common_top10_logprob_absolute_delta"],
        "distribution_metrics": control["distribution_metrics"],
    }


def _oracle_position(token_id: int, oracle: dict[str, Any]) -> dict[str, Any]:
    top_logprobs = {
        f"token_id:{candidate_id}": logprob
        for candidate_id, logprob in zip(oracle["token_ids"], oracle["logprobs"], strict=True)
    }
    selected_token = f"token_id:{token_id}"
    return {
        "selected_token": selected_token,
        "selected_logprob": top_logprobs[selected_token],
        "top_logprobs": top_logprobs,
    }


def _standalone_response(standalone: dict[str, Any], group: int) -> dict[str, Any]:
    return {
        "text": None,
        "positions": [
            _oracle_position(
                standalone["prefill_oracle_token_by_group"][group],
                standalone["prefill_oracle_top10_by_group"][group],
            ),
            _oracle_position(
                standalone["decode_oracle_token_by_group"][group],
                standalone["decode_oracle_top10_by_group"][group],
            ),
        ],
    }


def _numeric_comparison_pass(comparison: dict[str, Any]) -> bool:
    return (
        comparison["selected_tokens_match"]
        and comparison["maximum_selected_logprob_absolute_delta"] <= MAX_SELECTED_LOGPROB_DELTA
        and comparison["minimum_top10_token_overlap"] >= MIN_TOP10_TOKEN_OVERLAP
        and comparison["maximum_common_logprob_absolute_delta"] <= MAX_TOP10_TAIL_LOGPROB_DELTA
    )


def _chat_comparison_pass(comparison: dict[str, Any]) -> bool:
    return comparison["text_exact_match"] and _numeric_comparison_pass(comparison)


def _selected_comparison_pass(comparison: dict[str, Any], *, require_text: bool = False) -> bool:
    return comparison["selected_tokens_match"] and (not require_text or comparison["text_exact_match"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--standalone-baseline", type=Path, required=True)
    parser.add_argument("--standalone-batch-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrent-requests", type=int, default=8)
    args = parser.parse_args()

    chat_sequential = [_compact(_request(args.server_url, args.model)) for _ in range(2)]
    chat_run_to_run = _compare(chat_sequential[0], chat_sequential[1])

    barrier = threading.Barrier(args.concurrent_requests)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent_requests) as pool:
        futures = [pool.submit(_request, args.server_url, args.model, barrier) for _ in range(args.concurrent_requests)]
        chat_concurrent_responses = [_compact(future.result()) for future in futures]
    chat_graph_shape = [_compare(chat_sequential[0], candidate) for candidate in chat_concurrent_responses]
    chat_cross_batch_position = [
        _compare(chat_concurrent_responses[0], candidate) for candidate in chat_concurrent_responses[1:]
    ]

    standalone = _load_standalone_baseline(args.standalone_baseline)
    standalone_batch_control = _load_standalone_batch_control(
        args.standalone_batch_control, standalone["checkpoint_revision"]
    )
    direct_prompts = standalone["prompt_token_ids_by_group"]
    direct_sequential = [
        [_compact_completion(_completion_request(args.server_url, args.model, prompt)) for _ in range(2)]
        for prompt in direct_prompts
    ]
    direct_run_to_run = [_compare(responses[0], responses[1]) for responses in direct_sequential]
    direct_standalone = [
        _compare(_standalone_response(standalone, group), responses[0])
        for group, responses in enumerate(direct_sequential)
    ]

    direct_group_by_request = [index % len(direct_prompts) for index in range(args.concurrent_requests)]
    direct_barrier = threading.Barrier(args.concurrent_requests)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent_requests) as pool:
        direct_futures = [
            pool.submit(
                _completion_request,
                args.server_url,
                args.model,
                direct_prompts[group],
                direct_barrier,
            )
            for group in direct_group_by_request
        ]
        direct_concurrent_responses = [_compact_completion(future.result()) for future in direct_futures]
    direct_graph_shape = [
        _compare(direct_sequential[group][0], candidate)
        for group, candidate in zip(direct_group_by_request, direct_concurrent_responses, strict=True)
    ]
    direct_concurrent_standalone = [
        _compare(_standalone_response(standalone, group), candidate)
        for group, candidate in zip(direct_group_by_request, direct_concurrent_responses, strict=True)
    ]
    concurrent_indices_by_group = {
        group: [index for index, candidate_group in enumerate(direct_group_by_request) if candidate_group == group]
        for group in range(len(direct_prompts))
    }
    if any(len(indices) < 2 for indices in concurrent_indices_by_group.values()):
        raise ValueError("concurrent request count must exercise every direct prompt in at least two batch positions")
    direct_cross_batch_position = {
        str(group): [
            _compare(
                direct_concurrent_responses[indices[0]],
                direct_concurrent_responses[candidate_index],
            )
            for candidate_index in indices[1:]
        ]
        for group, indices in concurrent_indices_by_group.items()
    }

    passed = (
        _chat_comparison_pass(chat_run_to_run)
        and all(_chat_comparison_pass(item) for item in chat_cross_batch_position)
        and all(_selected_comparison_pass(item, require_text=True) for item in chat_graph_shape)
        and all(_numeric_comparison_pass(item) for item in direct_run_to_run)
        and all(_numeric_comparison_pass(item) for item in direct_standalone)
        and all(
            _numeric_comparison_pass(item)
            for comparisons in direct_cross_batch_position.values()
            for item in comparisons
        )
        and all(_selected_comparison_pass(item, require_text=True) for item in direct_graph_shape)
        and all(_selected_comparison_pass(item) for item in direct_concurrent_standalone)
    )
    artifact = {
        "verdict": "pass" if passed else "fail",
        "scope": (
            "Numeric TT logits exposed through the optional vLLM logprobs diagnostic path; "
            "this host-readback check is not the serving performance sampling path."
        ),
        "request": {
            "prompt": PROMPT,
            "temperature": 0.0,
            "top_k": 1,
            "top_logprobs": TOP_LOGPROBS,
            "max_tokens": MAX_TOKENS,
        },
        "direct_oracle_request": {
            "prompt": "four retained 32-token synthetic prompts",
            "max_tokens": 2,
            "trajectory": "prefill-selected token followed by one decode-selected token",
            "temperature": 0.0,
            "top_k": 1,
            "ignore_eos": True,
            "top_logprobs": TOP_LOGPROBS,
        },
        "thresholds": {
            "applies_to": "same-graph run-to-run, standalone-B1, and concurrent cross-position comparisons",
            "maximum_selected_logprob_absolute_delta": MAX_SELECTED_LOGPROB_DELTA,
            "minimum_top10_token_overlap": MIN_TOP10_TOKEN_OVERLAP,
            "maximum_top10_tail_logprob_absolute_delta": MAX_TOP10_TAIL_LOGPROB_DELTA,
        },
        "graph_shape_transition_gate": (
            "B1-to-padded-B32 distribution deltas are diagnostic; exact selected tokens are required. "
            "A standalone real-weight B1/B32 control separately requires every B32 row to select its "
            "corresponding B1 global maximum and reports full-vocabulary distribution sensitivity."
        ),
        "standalone_full_model_baseline": standalone,
        "standalone_full_model_b1_b32_control": standalone_batch_control,
        "chat_vllm_sequential_reference": chat_sequential[0],
        "chat_vllm_run_to_run_comparison": chat_run_to_run,
        "chat_vllm_concurrent_batch_size": args.concurrent_requests,
        "chat_vllm_concurrent_responses": chat_concurrent_responses,
        "chat_vllm_cross_batch_position_comparisons": chat_cross_batch_position,
        "chat_vllm_b1_to_padded_b32_graph_shape_comparisons": chat_graph_shape,
        "direct_standalone_prompt_token_ids_by_group": direct_prompts,
        "direct_vllm_sequential_responses_by_group": direct_sequential,
        "direct_vllm_run_to_run_comparisons_by_group": direct_run_to_run,
        "direct_vllm_standalone_comparisons_by_group": direct_standalone,
        "direct_vllm_concurrent_group_by_request": direct_group_by_request,
        "direct_vllm_concurrent_responses": direct_concurrent_responses,
        "direct_vllm_cross_batch_position_comparisons_by_group": direct_cross_batch_position,
        "direct_vllm_b1_to_padded_b32_graph_shape_comparisons": direct_graph_shape,
        "direct_vllm_concurrent_standalone_comparisons": direct_concurrent_standalone,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2))
    print(
        json.dumps(
            {
                "verdict": artifact["verdict"],
                "chat_run_to_run": chat_run_to_run,
                "chat_cross_batch_position": chat_cross_batch_position,
                "chat_graph_shape": chat_graph_shape,
                "direct_run_to_run": direct_run_to_run,
                "direct_standalone": direct_standalone,
                "direct_cross_batch_position": direct_cross_batch_position,
                "direct_graph_shape": direct_graph_shape,
                "direct_concurrent_standalone": direct_concurrent_standalone,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

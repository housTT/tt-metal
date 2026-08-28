# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Accuracy and qualitative runners for the Qwen3.8 full-model stage.

The checkout does not yet contain the shared readiness generator harness; it
only carries the runner-side degeneracy checker.  These functions implement
the harness's three required surfaces against the same reference artifact
schema: one prompt, 100 HF greedy tokens, and the HF top-100 set for every
generation position.
"""

from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import torch


def load_reference(reference: str | Path | dict) -> dict:
    if isinstance(reference, dict):
        value = reference
    else:
        value = torch.load(Path(reference), map_location="cpu", weights_only=False)
    required = {"metadata", "prompt_tokens", "reference_tokens", "top100_tokens"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"reference is missing {sorted(missing)}")
    if int(value["metadata"]["generation_length"]) != 100:
        raise ValueError("the main readiness reference must contain exactly 100 generated tokens")
    if tuple(torch.as_tensor(value["top100_tokens"]).shape) != (100, 100):
        raise ValueError("top100_tokens must have shape [100, 100]")
    return value


def _membership(tt_logits: torch.Tensor, hf_top100: torch.Tensor) -> dict[str, object]:
    logits = torch.as_tensor(tt_logits).float().reshape(-1, tt_logits.shape[-1])
    reference = torch.as_tensor(hf_top100, dtype=torch.int64).reshape(-1, 100)
    if logits.shape[0] != reference.shape[0]:
        raise ValueError(f"logit/reference row mismatch: {logits.shape[0]} versus {reference.shape[0]}")
    predicted = logits.argmax(-1)
    result = {"rows": int(predicted.numel()), "tt_top1_tokens": predicted.tolist()}
    for k in (1, 5, 100):
        hits = (predicted[:, None] == reference[:, :k]).any(-1)
        result[f"top{k}_hits"] = int(hits.sum())
        result[f"top{k}_percent"] = 100.0 * float(hits.float().mean())
    return result


def run_prefill_check(generator, reference: str | Path | dict) -> dict[str, object]:
    """Run the public prefill API and report HF top-1/top-5/top-100 membership."""

    ref = load_reference(reference)
    prompt = torch.as_tensor(ref["prompt_tokens"], dtype=torch.int64).reshape(1, -1)
    logits = generator.prefill_forward(
        prompt,
        prompt_lens=[prompt.shape[1]],
        request_ids=("aime24-prefill",),
        read_from_device=True,
    )
    report = _membership(logits[:, -1:, :], torch.as_tensor(ref["top100_tokens"][:1]))
    report.update(prompt_tokens=int(prompt.shape[1]), reference_rows=1, phase="prefill")
    return report


def run_teacher_forcing(
    generator,
    reference: str | Path | dict,
    *,
    enable_trace: bool = True,
    decode_rows: int = 99,
) -> dict[str, object]:
    """Teacher-force the HF tokens and report repeated decode top-k membership.

    The first reference row is the prefill prediction.  Decode feeds HF token
    ``i`` and compares the resulting logits with HF row ``i + 1``.  The
    compatibility mode intentionally reads full logits; it is excluded from
    the optimized token-out performance path.
    """

    ref = load_reference(reference)
    prompt = torch.as_tensor(ref["prompt_tokens"], dtype=torch.int64).reshape(1, -1)
    teacher = torch.as_tensor(ref["reference_tokens"], dtype=torch.int64).reshape(1, 100)
    prefill_logits = generator.prefill_forward(
        prompt,
        prompt_lens=[prompt.shape[1]],
        request_ids=("aime24-teacher",),
        read_from_device=True,
    )
    prefill_report = _membership(prefill_logits[:, -1:, :], torch.as_tensor(ref["top100_tokens"][:1]))
    if not 1 <= decode_rows <= 99:
        raise ValueError("decode_rows must be in [1, 99]")
    rows = []
    decode_step_seconds = []
    for step in range(decode_rows):
        started = time.perf_counter()
        logits, _ = generator.decode_forward(
            teacher[:, step],
            state=generator.state,
            enable_trace=enable_trace,
            host_sampling_compatibility=True,
            read_from_device=True,
        )
        decode_step_seconds.append(time.perf_counter() - started)
        rows.append(logits[:, -1, :])
    stacked = torch.cat(rows, dim=0)
    report = _membership(stacked, torch.as_tensor(ref["top100_tokens"])[1 : decode_rows + 1])
    steady = decode_step_seconds[1:] if len(decode_step_seconds) > 1 else decode_step_seconds
    report.update(
        prompt_tokens=int(prompt.shape[1]),
        reference_rows=decode_rows,
        phase="decode",
        traced=bool(enable_trace),
        compatibility_mode="explicit_host_logits",
        trace_capture_seconds=generator.model.trace_capture_seconds,
        decode_measured_tokens=len(steady),
        decode_seconds=sum(steady),
        decode_seconds_per_token=(sum(steady) / len(steady) if steady else 0.0),
        decode_tokens_per_second_per_user=(len(steady) / sum(steady) if steady and sum(steady) else 0.0),
        prefill=prefill_report,
    )
    return report


def _degeneracy(tokens: torch.Tensor, text: str) -> dict[str, object]:
    values = [int(value) for value in torch.as_tensor(tokens).reshape(-1)]
    counts = collections.Counter(values)
    dominant = max(counts.values(), default=0) / max(len(values), 1)
    adjacent_repeats = sum(left == right for left, right in zip(values, values[1:]))
    four_gram_counts = collections.Counter(tuple(values[i : i + 4]) for i in range(max(0, len(values) - 3)))
    repeated_four_grams = sum(count - 1 for count in four_gram_counts.values() if count > 1)
    latin = sum(character.isascii() and character.isalpha() for character in text)
    other_letters = sum(character.isalpha() and not character.isascii() for character in text)
    return {
        "dominant_token_fraction": dominant,
        "adjacent_token_repeats": adjacent_repeats,
        "repeated_four_grams": repeated_four_grams,
        "latin_letter_fraction": latin / max(latin + other_letters, 1),
        "mechanically_degenerate": dominant > 0.35 or adjacent_repeats > len(values) // 4,
    }


def run_autoregressive(
    generator,
    reference: str | Path | dict,
    *,
    enable_trace: bool = True,
) -> dict[str, object]:
    """Generate 100 free-running TT tokens and compare with the HF completion."""

    ref = load_reference(reference)
    prompt = torch.as_tensor(ref["prompt_tokens"], dtype=torch.int64).reshape(1, -1)
    hf_tokens = torch.as_tensor(ref["reference_tokens"], dtype=torch.int64).reshape(-1)
    tt_tokens = generator.generate_batch(
        prompt,
        100,
        enable_trace=enable_trace,
        sampling_mode="device",
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        request_ids=("aime24-autoregressive",),
        stop_on_eos=False,
    )[0]
    hf_text = ref.get("reference_text", generator.tokenizer.decode(hf_tokens.tolist(), skip_special_tokens=True))
    tt_text = generator.tokenizer.decode(tt_tokens.tolist(), skip_special_tokens=True)
    mismatches = torch.nonzero(tt_tokens != hf_tokens, as_tuple=False).flatten()
    first_divergence = int(mismatches[0]) if len(mismatches) else None
    return {
        "prompt_tokens": int(prompt.shape[1]),
        "generation_tokens": 100,
        "first_divergence": first_divergence,
        "matching_prefix_tokens": 100 if first_divergence is None else first_divergence,
        "hf_tokens": hf_tokens.tolist(),
        "tt_tokens": tt_tokens.tolist(),
        "hf_completion": hf_text,
        "tt_completion": tt_text,
        "hf_review": _degeneracy(hf_tokens, hf_text),
        "tt_review": _degeneracy(tt_tokens, tt_text),
        "metrics": generator.last_metrics.report(),
        "runtime_fallback_audit": generator.model.runtime_fallback_audit(generator.state),
    }


def run_qualitative_suite(generator, reference: str | Path | dict, *, enable_trace: bool = True) -> dict[str, object]:
    """Run the fixed multi-prompt chat suite through the traced TT generator."""

    ref = (
        torch.load(Path(reference), map_location="cpu", weights_only=False)
        if not isinstance(reference, dict)
        else reference
    )
    metadata = ref.get("metadata", {})
    prompts = ref.get("prompts", ())
    if metadata.get("prompt_mode") != "chat" or not metadata.get("chat_template"):
        raise ValueError("qualitative reference must record chat-template rendering")
    if len(prompts) < 3:
        raise ValueError("the shared qualitative suite must contain at least three prompts")
    generation_length = int(metadata.get("generation_length", 0))
    if not 64 <= generation_length <= 128:
        raise ValueError("qualitative generations must contain 64-128 tokens")

    results = []
    for item in prompts:
        prompt_id = str(item["id"])
        prompt = torch.as_tensor(item["prompt_tokens"], dtype=torch.int64).reshape(1, -1)
        hf_tokens = torch.as_tensor(item["reference_tokens"], dtype=torch.int64).reshape(-1)
        if int(hf_tokens.numel()) != generation_length:
            raise ValueError(f"{prompt_id} reference length does not match suite metadata")
        tt_tokens = generator.generate_batch(
            prompt,
            generation_length,
            enable_trace=enable_trace,
            sampling_mode="device",
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            request_ids=(f"qualitative-{prompt_id}",),
            stop_on_eos=False,
        )[0]
        hf_text = str(item["reference_text"])
        tt_text = generator.tokenizer.decode(tt_tokens.tolist(), skip_special_tokens=True)
        mismatches = torch.nonzero(tt_tokens != hf_tokens, as_tuple=False).flatten()
        first_divergence = int(mismatches[0]) if len(mismatches) else None
        results.append(
            {
                "id": prompt_id,
                "messages": item["messages"],
                "rendered_prompt": item["rendered_prompt"],
                "prompt_tokens": prompt.reshape(-1).tolist(),
                "hf_tokens": hf_tokens.tolist(),
                "tt_tokens": tt_tokens.tolist(),
                "hf_completion": hf_text,
                "tt_completion": tt_text,
                "first_divergence": first_divergence,
                "matching_prefix_tokens": generation_length if first_divergence is None else first_divergence,
                "hf_review": _degeneracy(hf_tokens, hf_text),
                "tt_review": _degeneracy(tt_tokens, tt_text),
                "metrics": generator.last_metrics.report(),
                "runtime_fallback_audit": generator.model.runtime_fallback_audit(generator.state),
            }
        )
    return {"metadata": metadata, "prompts": results}


def write_report(report: dict, output: str | Path) -> None:
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")


def write_autoregressive_artifacts(report: dict, output_dir: str | Path) -> None:
    """Write the canonical readiness sidecars consumed by runner-side gates."""

    required = {"prompt_tokens", "generation_tokens", "hf_tokens", "tt_tokens", "hf_completion", "tt_completion"}
    missing = required - report.keys()
    if missing:
        raise ValueError(f"autoregressive report is missing {sorted(missing)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "source": "models.autoports.qwen_qwen3_8_flash_next.demo.full_model.run_autoregressive",
        "prompt_tokens": int(report["prompt_tokens"]),
        "generation_tokens": int(report["generation_tokens"]),
        "hf": {"token_ids": [int(token) for token in report["hf_tokens"]]},
        "tt": {"token_ids": [int(token) for token in report["tt_tokens"]]},
    }
    write_report(metadata, output_dir / "autoregressive_meta.json")
    (output_dir / "hf_completion.txt").write_text(str(report["hf_completion"]) + "\n", encoding="utf-8")
    (output_dir / "tt_completion.txt").write_text(str(report["tt_completion"]) + "\n", encoding="utf-8")


__all__ = [
    "load_reference",
    "run_autoregressive",
    "run_prefill_check",
    "run_qualitative_suite",
    "run_teacher_forcing",
    "write_autoregressive_artifacts",
    "write_report",
]

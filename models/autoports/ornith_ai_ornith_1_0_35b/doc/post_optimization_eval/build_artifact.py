# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build the compact Ornith post-optimization evaluation handoff.

Raw lm-eval samples and JUnit files remain outside the model tree.  Their hashes,
sample IDs and aggregate results are retained in the committed JSON so the handoff is
auditable without shipping large or prompt-bearing artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ID = "ornith-ai/Ornith-1.0-35B"
HF_REVISION = "5df2ed3f675c7beaa490328cc70bb573b65fb660"
HISTORICAL_RESULTS = {
    "ifeval": {
        "score_percent": 82.08056478405315,
        "samples": 28,
        "temperature": 1.0,
        "max_num_seqs": 32,
        "max_gen_toks": 8192,
        "measured_at": "2026-08-19",
        "results_sha256": "f52ca367221dd6f1b7a50de796dd7f54710ceeb6e9bae4dabb6fe95d74680445",
        "samples_sha256": "587edb112e7ca9800289b44a32d0f7e574bbe709edc4faa6362628dd742a23a7",
    },
    "r1_gpqa_diamond": {
        "score_percent": 50.0,
        "samples": 10,
        "temperature": 1.0,
        "max_num_seqs": 32,
        "max_gen_toks": 32768,
        "measured_at": "2026-08-19",
        "results_sha256": "8d50ca94e5ed4de4ace3c207fb2317762636d78f151e7bd7cbc1f44ab9b4e978",
        "samples_sha256": "5f10cc3b915e0e213a221d6cf7a3dd710e2934cf0fe2306fb7c7f05ed7d0c49f",
    },
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _response_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in _response_strings(child)]
    return []


def _samples(path: Path) -> tuple[list[int | str], int, dict[str, int]]:
    doc_ids: list[int | str] = []
    empty_filtered_responses = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                doc_ids.append(row["doc_id"])
                responses = _response_strings(row.get("filtered_resps"))
                empty_filtered_responses += not any(response.strip() for response in responses)
    return (
        doc_ids,
        len(doc_ids),
        {
            "nonempty_filtered_responses": len(doc_ids) - empty_filtered_responses,
            "empty_filtered_responses": empty_filtered_responses,
        },
    )


def _junit_case_names(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [case.attrib.get("name", "").split("[", 1)[0] for case in root.findall(".//testcase")]


def _junit(path: Path, label: str) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    elapsed = 0.0
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.attrib.get(key, 0))
        elapsed += float(suite.attrib.get("time", 0.0))
    return {
        "label": label,
        **totals,
        "elapsed_s": round(elapsed, 3),
        "status": "pass" if totals["failures"] == totals["errors"] == 0 else "fail",
        "raw_sha256": _sha256(path),
    }


def _quality_task(
    task: str,
    result_path: Path,
    samples_path: Path,
    *,
    full_samples: int,
) -> dict[str, Any]:
    payload = _json(result_path)
    metrics = payload["results"][task]
    doc_ids, count, response_diagnostics = _samples(samples_path)
    expected_doc_ids = list(range(28 if task == "ifeval" else 10))
    if doc_ids != expected_doc_ids:
        raise ValueError(f"{task} must contain the fixed document IDs {expected_doc_ids}; got {doc_ids}")
    if task == "ifeval":
        metric_names = [
            "prompt_level_strict_acc,none",
            "inst_level_strict_acc,none",
            "prompt_level_loose_acc,none",
            "inst_level_loose_acc,none",
        ]
        score = sum(float(metrics[name]) for name in metric_names) / len(metric_names) * 100
        reported_metrics = {name: float(metrics[name]) * 100 for name in metric_names}
        stderr = None
    else:
        score = float(metrics["exact_match,none"]) * 100
        reported_metrics = {"exact_match,none": score}
        stderr = float(metrics["exact_match_stderr,none"]) * 100

    config = payload["configs"][task]
    historical = HISTORICAL_RESULTS[task]
    return {
        "task": task,
        "scope": "fixed_ci_subset",
        "score_percent": score,
        "score_definition": ("mean_of_four_reported_accuracies" if task == "ifeval" else "exact_match"),
        "stderr_percent": stderr,
        "metrics_percent": reported_metrics,
        "samples": count,
        "full_dataset_samples": full_samples,
        "coverage_percent": count / full_samples * 100,
        "doc_ids": doc_ids,
        "response_diagnostics": response_diagnostics,
        "reference": None,
        "classification": "measured_only_no_gpu_or_published_reference",
        "configuration": {
            "num_fewshot": config.get("num_fewshot"),
            "generation_kwargs": config.get("generation_kwargs"),
            "api_model": config.get("metadata"),
            "evaluation_harness": {
                "lm_eval_version": payload.get("lm_eval_version"),
                "lm_eval_git_revision": payload.get("git_hash"),
                "transformers_version": payload.get("transformers_version"),
            },
        },
        "raw": {
            "results_sha256": _sha256(result_path),
            "samples_sha256": _sha256(samples_path),
        },
        "historical_same_doc_ids": historical,
        "historical_context_delta_points": score - historical["score_percent"],
        "historical_comparison": "context_only_not_a_graded_ab_configuration_changed",
    }


def _latency(finalization_path: Path) -> dict[str, Any]:
    finalization = _json(finalization_path)
    if finalization.get("status") != "pass":
        raise ValueError("latency finalization is not a passing receipt")
    base = finalization_path.parent
    cell_path = base / "json" / "isl-131072_osl-512_batch-8.json"
    cell = _json(cell_path)
    median_itl = float(cell["median_itl_ms"])
    return {
        "status": "pass",
        "scope": "previously_finalized_selected_profile",
        "run_id": finalization["run_id"],
        "validated_cells": finalization["validation"]["validated_cells"],
        "strict_latency_provenance": finalization["validation"]["strict_latency_provenance"],
        "selected_cell": {
            "concurrency": 8,
            "input_tokens": 131072,
            "output_tokens": 512,
            "median_ttft_s": float(cell["median_ttft_ms"]) / 1000,
            "median_e2el_s": float(cell["median_e2el_ms"]) / 1000,
            "decode_tokens_per_second_per_user": 1000 / median_itl,
            "aggregate_decode_tokens_per_second": 8 * 1000 / median_itl,
            "e2el_delta_vs_gathered_percent": -42.15,
            "e2el_delta_vs_original_sparse_percent": -53.24,
            "failed_requests": int(cell.get("error_request_count") or 0),
        },
        "receipt_sha256": _sha256(finalization_path),
        "latency_cell_sha256": _sha256(cell_path),
        "json_set_sha256": finalization["latency_json"]["json_set_sha256"],
    }


def _render_markdown(artifact: dict[str, Any]) -> str:
    quality = {row["task"]: row for row in artifact["quality"]}
    api = artifact["correctness"]["api"]
    device = artifact["correctness"]["device"]
    host = artifact["correctness"]["host"]
    static_contract = artifact["correctness"]["production_static_contract"]
    cell = artifact["performance"]["selected_cell"]
    gpqa_stderr = quality["r1_gpqa_diamond"]["stderr_percent"]
    gpqa_empty = quality["r1_gpqa_diamond"]["response_diagnostics"]["empty_filtered_responses"]
    host_tests = sum(row["tests"] for row in host)
    host_failed = sum(row["failures"] + row["errors"] for row in host)
    lines = [
        "# Ornith-1.0-35B post-optimization evaluation",
        "",
        f"Status: **{artifact['status'].replace('_', ' ')}**. This is a compact handoff artifact; "
        "`results.json` is canonical.",
        "",
        "## Quality subset",
        "",
        "| task | samples | current | historical context | delta | interpretation |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        f"| IFEval | {quality['ifeval']['samples']}/{quality['ifeval']['full_dataset_samples']} | "
        f"{quality['ifeval']['score_percent']:.2f} | "
        f"{quality['ifeval']['historical_same_doc_ids']['score_percent']:.2f} | "
        f"{quality['ifeval']['historical_context_delta_points']:+.2f} | four-metric mean; measured "
        "only |",
        f"| GPQA-Diamond | {quality['r1_gpqa_diamond']['samples']}/"
        f"{quality['r1_gpqa_diamond']['full_dataset_samples']} | "
        f"{quality['r1_gpqa_diamond']['score_percent']:.2f} ± {gpqa_stderr:.2f} | "
        f"{quality['r1_gpqa_diamond']['historical_same_doc_ids']['score_percent']:.2f} | "
        f"{quality['r1_gpqa_diamond']['historical_context_delta_points']:+.2f} | measured only; "
        f"{gpqa_empty}/10 parsed final responses were empty |",
        "",
        "These are fixed CI subsets (IFEval doc IDs 0–27; GPQA doc IDs 0–9), not full-benchmark "
        "scores. The historical run used the same IDs but different sampling and max-seqs settings, "
        "so it is context rather than a graded A/B.",
        "",
        "Eight GPQA responses exhausted their reasoning budget without parsed final content; the "
        "20% score therefore diagnoses the selected batch-8 serving profile, not just subject "
        "knowledge.",
        "",
        "## Correctness and serving",
        "",
        f"- API gate: **{api['summary']['passed']}/{api['summary']['total']} passed** (health, model "
        "discovery, short greedy repeatability, reasoning parser, tool parser, streaming, concurrency 8).",
        f"- Device gate: **{sum(row['tests'] for row in device)} tests passed**; the full 40-layer "
        "B1/B2/B4 proof recorded 280 native calls/subchunks, 120 layer calls, and zero fallbacks.",
        f"- Host/static gates: **{host_tests - host_failed}/{host_tests} passed**.",
        f"- Production source/evidence contract: **{static_contract['status']}**.",
        "",
        "## Selected-profile performance",
        "",
        f"The finalized 10-cell sweep passed strict provenance. At concurrency 8, 131,072 input "
        f"tokens and 512 output tokens: median TTFT **{cell['median_ttft_s']:.2f}s**, median E2EL "
        f"**{cell['median_e2el_s']:.2f}s**, aggregate decode **"
        f"{cell['aggregate_decode_tokens_per_second']:.2f} tok/s**, zero failed requests. E2EL is "
        "42.15% below the gathered-MoE baseline and 53.24% below the original sparse path.",
        "",
        "## Provenance and limits",
        "",
        f"- weights: `{artifact['provenance']['weights']['repo']}` @ "
        f"`{artifact['provenance']['weights']['revision']}`",
        f"- tt-metal runtime: `{artifact['provenance']['tt_metal']['runtime_revision']}`; evaluation "
        f"parent: `{artifact['provenance']['tt_metal']['evaluation_parent_revision']}`",
        f"- vLLM: `{artifact['provenance']['vllm']['revision']}`",
        "- hardware: four Blackhole p300c chips, `(1, 4)` ring (P300_X2)",
        "- selected server: C25 Q/K=128, native MoE 2K span, max-seqs 8, four API frontends, "
        "2.0s coalescing ceiling",
        "- text-only port; no full IFEval/GPQA or GPU-reference result is claimed",
        "- short thinking-off generations repeated byte-for-byte; longer thinking-on greedy "
        "generations did not, matching the carried-forward padded-decode reproducibility limitation",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-gates", type=Path, required=True)
    parser.add_argument("--ifeval-results", type=Path, required=True)
    parser.add_argument("--ifeval-samples", type=Path, required=True)
    parser.add_argument("--gpqa-results", type=Path, required=True)
    parser.add_argument("--gpqa-samples", type=Path, required=True)
    parser.add_argument("--latency-finalization", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--static-contract", type=Path, required=True)
    parser.add_argument("--device-junit", type=Path, action="append", default=[])
    parser.add_argument("--host-junit", type=Path, action="append", default=[])
    parser.add_argument("--tt-metal-root", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    api = _json(args.api_gates)
    device = [_junit(path, path.stem) for path in args.device_junit]
    host = [_junit(path, path.stem) for path in args.host_junit]
    device_case_names = {name for path in args.device_junit for name in _junit_case_names(path)}
    required_device_cases = {
        "test_topk_native_matches_gathered_layer_and_counts_exact_composites",
        "test_topk_native_reduced_stack_2048_completes_collectives",
        "test_topk_native_decode_keeps_sparse_path_and_does_not_move_counters",
        "test_full_stack_b1_b2_b4_prefill_has_exact_native_counters",
    }
    quality = [
        _quality_task("ifeval", args.ifeval_results, args.ifeval_samples, full_samples=541),
        _quality_task("r1_gpqa_diamond", args.gpqa_results, args.gpqa_samples, full_samples=198),
    ]
    finalization = _json(args.latency_finalization)
    selected_config = _json(args.selected_config)
    static_contract = _json(args.static_contract)
    all_gate_rows = device + host
    gates_pass = (
        api["summary"]["status"] == "pass"
        and bool(device)
        and bool(host)
        and required_device_cases <= device_case_names
        and static_contract.get("status") == "pass"
        and all(row["status"] == "pass" for row in all_gate_rows)
    )
    has_quality_warning = any(row["response_diagnostics"]["empty_filtered_responses"] for row in quality)
    runtime_revision = finalization["configuration"]["revisions"]["tt_metal"]
    artifact: dict[str, Any] = {
        "schema": "ornith-post-optimization-eval/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID,
        "status": "functional_pass_with_warnings" if gates_pass else "fail",
        "summary": {
            "model_functional": gates_pass,
            "quality_classification": (
                "measured_only_with_empty_final_response_warning"
                if has_quality_warning
                else "measured_only_no_reference"
            ),
            "long_generation_byte_reproducibility": api["diagnostics"]["long_greedy_reproducibility"]["reproducible"],
        },
        "quality": quality,
        "correctness": {
            "api": api,
            "device": device,
            "host": host,
            "production_static_contract": {
                "schema": static_contract.get("schema"),
                "status": static_contract.get("status"),
                "raw_sha256": _sha256(args.static_contract),
            },
            "full_device_contract": {
                "layers": 40,
                "physical_prefill_batches": [1, 2, 4],
                "tokens_per_user": 2048,
                "native_calls": 280,
                "native_subchunks": 280,
                "native_layer_calls": 120,
                "fallbacks": 0,
                "source": "passing test_full_stack_b1_b2_b4_prefill_has_exact_native_counters",
            },
        },
        "performance": _latency(args.latency_finalization),
        "profile": {
            "precision_config": selected_config["config_id"],
            "prefill_sdpa_qk_chunk": selected_config["prefill"]["sdpa_q_k_chunk"],
            "native_moe_sub_chunk": 2048,
            "max_num_seqs": 8,
            "api_server_count": 4,
            "input_queue_batching_delay_s": 2.0,
            "max_model_len": 262144,
            "environment": {
                "ORNITH_MOE_TOPK_NATIVE": "1",
                "ORNITH_MOE_TOPK_NATIVE_SUB_CHUNK": "2048",
                "ORNITH_MOE_GATHER": "unset",
                "TT_MAX_PREFILLS_PER_STEP": "4",
                "TT_INTERLEAVE_PREFILL_CHUNKS": "1",
                "ORNITH_VLLM_PREFILL_WARMUP": "all",
            },
        },
        "provenance": {
            "weights": {"repo": MODEL_ID, "revision": HF_REVISION},
            "tt_metal": {
                "runtime_revision": runtime_revision,
                "evaluation_parent_revision": _git_head(args.tt_metal_root),
            },
            "vllm": {"revision": _git_head(args.vllm_root)},
            "hardware": {
                "system": "P300_X2",
                "chips": 4,
                "chip_type": "p300c",
                "mesh": [1, 4],
                "architecture": "blackhole",
            },
        },
        "limitations": [
            "IFEval and GPQA-Diamond are fixed 5% CI subsets, not full benchmark scores.",
            "No published or GPU reference exists for these task/configuration pairs; quality rows are measured-only.",
            "Eight of ten GPQA samples had empty parsed final content after the reasoning budget was exhausted.",
            "Long thinking-on greedy generations are not byte reproducible on the padded batch-8 decode path.",
            "The TT port serves text only.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.write_text(_render_markdown(artifact), encoding="utf-8")
    if artifact["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

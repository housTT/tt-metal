#!/usr/bin/env python3
"""Merge validated GPQA, benchmark, and spec-test repairs and rerender."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

WORKSPACE_ROOT = Path("/home/ttuser/dev/gpt-oss-20b")
TTI_ROOT = WORKSPACE_ROOT / "tti-release" / "openai_gpt_oss_120b" / "tt-inference-server"
sys.path.insert(0, str(TTI_ROOT))

from report_module import ReportGenerator, ReportSchema, acceptance_criteria_check, build_acceptance_export

TASK_NAME = "gpqa_diamond_cot_zeroshot"
EXPECTED_EVAL_TASKS = frozenset({"aime25", TASK_NAME, "mmlu_generative"})
EXPECTED_BENCHMARK_ROWS = (
    (128, 128, 1, 8),
    (128, 128, 32, 256),
    (128, 1024, 1, 4),
    (128, 1024, 32, 128),
    (1024, 128, 1, 4),
    (1024, 128, 32, 128),
    (2048, 128, 1, 4),
    (2048, 128, 32, 128),
    (4096, 128, 1, 4),
    (4096, 128, 31, 124),
    (8192, 128, 1, 2),
    (8192, 128, 15, 30),
    (8192, 1024, 1, 2),
    (8192, 1024, 14, 28),
    (10000, 1024, 1, 2),
    (10000, 1024, 11, 22),
    (16384, 128, 1, 2),
    (16384, 128, 7, 14),
    (32768, 128, 1, 1),
    (32768, 128, 3, 3),
    (65536, 128, 1, 1),
)
AUTOPORT_PATH = "models/autoports/openai_gpt_oss_120b"
HANDOFF_PATH = f"{AUTOPORT_PATH}/doc/tti_release"
CONTEXT_CONTRACT_PATH = WORKSPACE_ROOT / "tt-metal" / AUTOPORT_PATH / "doc" / "context_contract.json"
SUPPORTED_CONTEXT = 131072
PUBLISHER_REVISION = "56686c06f5e19865c153de0fdb11be3890014df7"
ARCHIVE_SHA256 = "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc"
DIAMOND_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
ACCEPTANCE_KEYS = (
    "acceptance_criteria",
    "acceptance_blockers",
    "acceptance_criteria_metadata",
    "acceptance_summary_markdown",
)
RAW_BENCHMARK_FIELDS = frozenset(
    {
        "backend",
        "completed",
        "date",
        "errors",
        "failed",
        "input_lens",
        "max_concurrency",
        "model_id",
        "num_prompts",
        "output_lens",
        "total_input_tokens",
        "total_output_tokens",
    }
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class _JsonChars:
    """One-character pushback reader for projecting a top-level JSON object."""

    def __init__(self, handle: TextIO):
        self._handle = handle
        self._pending = ""

    def get(self) -> str:
        if self._pending:
            char, self._pending = self._pending, ""
            return char
        return self._handle.read(1)

    def unread(self, char: str) -> None:
        if not char or self._pending:
            raise ValueError("invalid JSON pushback")
        self._pending = char


def _next_non_whitespace(chars: _JsonChars) -> str:
    while (char := chars.get()) and char.isspace():
        pass
    return char


def _consume_json_value(chars: _JsonChars, *, capture: bool) -> str | None:
    """Consume one JSON value, retaining text only for allowlisted fields."""

    first = _next_non_whitespace(chars)
    if not first:
        raise ValueError("unexpected end of JSON value")
    pieces = [first] if capture else None

    if first == '"':
        escaped = False
        while True:
            char = chars.get()
            if not char:
                raise ValueError("unterminated JSON string")
            if pieces is not None:
                pieces.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                break
        return "".join(pieces) if pieces is not None else None

    if first in "[{":
        expected_closers = ["]" if first == "[" else "}"]
        in_string = False
        escaped = False
        while expected_closers:
            char = chars.get()
            if not char:
                raise ValueError("unterminated compound JSON value")
            if pieces is not None:
                pieces.append(char)
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "[":
                expected_closers.append("]")
            elif char == "{":
                expected_closers.append("}")
            elif char in "]}":
                if char != expected_closers.pop():
                    raise ValueError("mismatched JSON delimiter")
        return "".join(pieces) if pieces is not None else None

    while True:
        char = chars.get()
        if not char or char.isspace():
            break
        if char in ",]}":
            chars.unread(char)
            break
        if pieces is not None:
            pieces.append(char)
    return "".join(pieces) if pieces is not None else None


def _read_json_fields(path: Path, fields: frozenset[str]) -> dict:
    """Read selected top-level fields without materializing generated outputs."""

    selected = {}
    with path.open("r", encoding="utf-8") as handle:
        chars = _JsonChars(handle)
        if _next_non_whitespace(chars) != "{":
            raise ValueError(f"raw benchmark is not a JSON object: {path}")
        while True:
            char = _next_non_whitespace(chars)
            if char == "}":
                break
            if char != '"':
                raise ValueError(f"invalid JSON object key in {path}")
            chars.unread(char)
            key_text = _consume_json_value(chars, capture=True)
            key = json.loads(key_text)
            if _next_non_whitespace(chars) != ":":
                raise ValueError(f"missing JSON key separator in {path}")
            value_text = _consume_json_value(chars, capture=key in fields)
            if key in fields:
                if key in selected:
                    raise ValueError(f"duplicate JSON field {key!r} in {path}")
                selected[key] = json.loads(value_text)
            delimiter = _next_non_whitespace(chars)
            if delimiter == "}":
                break
            if delimiter != ",":
                raise ValueError(f"invalid JSON object delimiter in {path}")
        if _next_non_whitespace(chars):
            raise ValueError(f"trailing data in raw benchmark JSON: {path}")
    return selected


def _task_name(block) -> str | None:
    if block.kind != "evals" or not isinstance(block.data, dict):
        return None
    return block.data.get("task_name")


def _exact_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    result = int(value)
    if value != result:
        raise RuntimeError(f"{label} is not an integer: {value}")
    return result


def _normalized_timestamp(value: str, label: str) -> str:
    for fmt in ("%Y%m%d-%H%M%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y%m%d-%H%M%S")
        except ValueError:
            pass
    raise RuntimeError(f"invalid {label} timestamp: {value}")


def _raw_benchmark_projection(path: Path) -> dict:
    # Deliberately scan past unselected values without materializing them, so
    # generated model text is never loaded, logged, copied, or handed off.
    return _read_json_fields(path, RAW_BENCHMARK_FIELDS)


def _validate_raw_benchmarks(benchmark_blocks, raw_dir: Path) -> list[dict]:
    raw_dir = raw_dir.resolve()
    workspace_root = WORKSPACE_ROOT.resolve()
    if workspace_root != raw_dir and workspace_root not in raw_dir.parents:
        raise RuntimeError("benchmark raw directory is outside the authorized workspace")
    if not raw_dir.is_dir():
        raise RuntimeError(f"benchmark raw directory does not exist: {raw_dir}")
    by_timestamp = {}
    for path in sorted(raw_dir.glob("benchmark_*.json")):
        raw = _raw_benchmark_projection(path)
        timestamp = _normalized_timestamp(raw.get("date"), f"raw {path.name}")
        if timestamp in by_timestamp:
            raise RuntimeError(f"ambiguous raw benchmark timestamp: {timestamp}")
        by_timestamp[timestamp] = (path, raw)

    validated = []
    used_paths = set()
    for index, (block, expected) in enumerate(zip(benchmark_blocks, EXPECTED_BENCHMARK_ROWS), start=1):
        isl, osl, concurrency, num_prompts = expected
        aggregate_timestamp = _normalized_timestamp(block.targets.get("timestamp"), f"aggregate row {index}")
        match = by_timestamp.get(aggregate_timestamp)
        if match is None:
            raise RuntimeError(f"benchmark row {index} has no raw artifact at {aggregate_timestamp}")
        path, raw = match
        if path in used_paths:
            raise RuntimeError(f"raw benchmark artifact reused: {path.name}")
        used_paths.add(path)

        expected_scalars = {
            "max_concurrency": concurrency,
            "num_prompts": num_prompts,
            "completed": num_prompts,
            "failed": 0,
            "total_input_tokens": isl * num_prompts,
            "total_output_tokens": osl * num_prompts,
        }
        for key, expected_value in expected_scalars.items():
            actual = _exact_int(raw.get(key), f"raw row {index} {key}")
            if actual != expected_value:
                raise RuntimeError(f"raw row {index} {key}={actual}, expected {expected_value}")
        if raw.get("model_id") != "openai/gpt-oss-120b":
            raise RuntimeError(f"raw row {index} has unexpected model_id")
        if raw.get("backend") != "vllm":
            raise RuntimeError(f"raw row {index} has unexpected backend")

        errors = raw.get("errors")
        if not isinstance(errors, list) or len(errors) != num_prompts:
            raise RuntimeError(f"raw row {index} has malformed errors evidence")
        if any(errors):
            raise RuntimeError(f"raw row {index} contains request errors")
        for key, exact_length in (("input_lens", isl), ("output_lens", osl)):
            lengths = raw.get(key)
            if not isinstance(lengths, list) or len(lengths) != num_prompts:
                raise RuntimeError(f"raw row {index} has malformed {key} evidence")
            if any(_exact_int(value, f"raw row {index} {key}") != exact_length for value in lengths):
                raise RuntimeError(f"raw row {index} does not preserve exact {key}={exact_length}")
        validated.append(
            {
                "artifact": path.name,
                "timestamp": aggregate_timestamp,
                "input_sequence_length": isl,
                "output_sequence_length": osl,
                "concurrency": concurrency,
                "num_prompts": num_prompts,
                "completed": num_prompts,
                "failed": 0,
                "errors": 0,
            }
        )
    if len(used_paths) != len(EXPECTED_BENCHMARK_ROWS):
        raise RuntimeError("raw benchmark evidence did not bind uniquely to all rows")
    return validated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-report-json", required=True, type=Path)
    parser.add_argument("--repair-report-json", required=True, type=Path)
    parser.add_argument("--benchmark-repair-report-json", required=True, type=Path)
    parser.add_argument("--benchmark-raw-dir", required=True, type=Path)
    parser.add_argument("--benchmark-issue-waiver", required=True, type=Path)
    parser.add_argument("--spec-repair-report-json", required=True, type=Path)
    parser.add_argument("--runtime-model-spec", required=True, type=Path)
    parser.add_argument("--official-vllm-release-commit", required=True)
    parser.add_argument("--tti-release-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    runtime_wrapper = _read_json(args.runtime_model_spec)
    runtime_spec = runtime_wrapper.get("runtime_model_spec", runtime_wrapper)
    impl = runtime_spec.get("impl") or {}
    if impl.get("code_path") != AUTOPORT_PATH:
        raise RuntimeError("runtime spec does not identify the generated autoport")
    for label, commit in (
        ("official vLLM release", args.official_vllm_release_commit),
        ("TTI release", args.tti_release_commit),
    ):
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise RuntimeError(f"invalid {label} commit SHA")

    metadata = runtime_spec.get("metadata") or {}
    device_spec = runtime_spec.get("device_model_spec") or {}
    vllm_args = device_spec.get("vllm_args") or {}
    context_values = {
        "metadata.supported_context": metadata.get("supported_context"),
        "device_model_spec.max_context": device_spec.get("max_context"),
        "device_model_spec.max_tokens_all_users_override": device_spec.get("max_tokens_all_users_override"),
        "device_model_spec.vllm_args.max_model_len": vllm_args.get("max_model_len"),
        "device_model_spec.vllm_args.max_num_batched_tokens": vllm_args.get("max_num_batched_tokens"),
    }
    for label, value in context_values.items():
        if isinstance(value, str):
            try:
                exact_value = int(value)
            except ValueError as exc:
                raise RuntimeError(f"{label} is not an integer: {value}") from exc
            if str(exact_value) != value:
                raise RuntimeError(f"{label} is not an integer: {value}")
        else:
            exact_value = _exact_int(value, label)
        if exact_value != SUPPORTED_CONTEXT:
            raise RuntimeError(f"{label} does not preserve {SUPPORTED_CONTEXT}")
    context_contract = _read_json(CONTEXT_CONTRACT_PATH)
    if context_contract.get("current_supported_context") != SUPPORTED_CONTEXT:
        raise RuntimeError("context contract does not preserve supported context")
    context_contract_sha256 = hashlib.sha256(CONTEXT_CONTRACT_PATH.read_bytes()).hexdigest()

    release_payload = _read_json(args.release_report_json)
    repair_payload = _read_json(args.repair_report_json)
    benchmark_payload = _read_json(args.benchmark_repair_report_json)
    spec_payload = _read_json(args.spec_repair_report_json)
    release = ReportSchema.from_dict(release_payload)
    repair = ReportSchema.from_dict(repair_payload)
    benchmark_repair = ReportSchema.from_dict(benchmark_payload)
    spec_repair = ReportSchema.from_dict(spec_payload)

    release_eval_tasks = {task_name for block in release.sections if (task_name := _task_name(block)) is not None}
    if release_eval_tasks != EXPECTED_EVAL_TASKS:
        raise RuntimeError("main release report has unexpected eval coverage: " f"{sorted(release_eval_tasks)}")
    release_kinds = {block.kind for block in release.sections}
    missing_kinds = {"benchmarks", "spec_tests"} - release_kinds
    if missing_kinds:
        raise RuntimeError(f"main release report is missing sections: {sorted(missing_kinds)}")

    original_blockers = release_payload.get("acceptance_blockers") or {}
    recoverable_blocker_keys = {
        "spec.spec_tests:Vllm Chat Completions",
        "task:spec_tests",
        "task:llm_eval",
    }
    unrelated_blockers = {
        key: value
        for key, value in original_blockers.items()
        if key not in recoverable_blocker_keys and TASK_NAME not in key and TASK_NAME not in str(value)
    }
    if unrelated_blockers:
        raise RuntimeError(
            "main release has blockers unrelated to the recovered GPQA task: " f"{sorted(unrelated_blockers)}"
        )

    if repair_payload.get("acceptance_criteria") is not True:
        raise RuntimeError("GPQA repair workflow did not pass acceptance")
    if repair_payload.get("acceptance_blockers"):
        raise RuntimeError("GPQA repair workflow still has acceptance blockers")

    repair_blocks = [block for block in repair.sections if _task_name(block) == TASK_NAME]
    if len(repair_blocks) != 1:
        raise RuntimeError(f"expected one repair block, found {len(repair_blocks)}")
    repair_block = repair_blocks[0]
    if repair_block.data.get("score") is None:
        raise RuntimeError("repair block has no score")
    if int(repair_block.data.get("accuracy_check", 3)) == 3:
        raise RuntimeError("repair GPQA accuracy row did not pass")

    if benchmark_payload.get("acceptance_criteria") is not True:
        raise RuntimeError("benchmark repair workflow did not pass acceptance")
    if benchmark_payload.get("acceptance_blockers"):
        raise RuntimeError("benchmark repair workflow still has acceptance blockers")
    benchmark_blocks = [block for block in benchmark_repair.sections if block.kind == "benchmarks"]
    if len(benchmark_blocks) != 21:
        raise RuntimeError(f"expected 21 repaired benchmark blocks, found {len(benchmark_blocks)}")
    realized_rows = []
    for index, block in enumerate(benchmark_blocks, start=1):
        data = block.data
        # The vLLM aggregate parser represents zero failures as null. The raw
        # artifact validation below is authoritative for request failures.
        if data.get("error_request_count") not in (None, 0):
            raise RuntimeError(f"benchmark repair row {index} has request errors")
        for metric in (
            "mean_ttft_ms",
            "mean_tpot_ms",
            "tps_output_throughput",
            "input_sequence_length",
            "output_sequence_length",
        ):
            if data.get(metric) is None:
                raise RuntimeError(f"benchmark repair row {index} is missing metric {metric}")
        if data.get("status") == "fail":
            raise RuntimeError(f"benchmark repair row {index} is an unwaived failure")
        realized_rows.append(
            tuple(
                _exact_int(data[key], f"aggregate row {index} {key}")
                for key in (
                    "input_sequence_length",
                    "output_sequence_length",
                    "concurrency",
                    "num_requests",
                )
            )
        )
    if tuple(realized_rows) != EXPECTED_BENCHMARK_ROWS:
        raise RuntimeError("benchmark repair did not preserve the exact ordered workload matrix: " f"{realized_rows}")
    raw_benchmark_evidence = _validate_raw_benchmarks(benchmark_blocks, args.benchmark_raw_dir)
    issue_waiver = args.benchmark_issue_waiver.resolve()
    workspace_root = WORKSPACE_ROOT.resolve()
    if workspace_root != issue_waiver and workspace_root not in issue_waiver.parents:
        raise RuntimeError("benchmark issue waiver is outside the authorized workspace")
    if not issue_waiver.is_file():
        raise RuntimeError(f"benchmark issue waiver does not exist: {issue_waiver}")

    if spec_payload.get("acceptance_criteria") is not True:
        raise RuntimeError("spec-test repair workflow did not pass acceptance")
    if spec_payload.get("acceptance_blockers"):
        raise RuntimeError("spec-test repair workflow still has acceptance blockers")
    spec_blocks = [block for block in spec_repair.sections if block.kind == "spec_tests"]
    if len(spec_blocks) != 2:
        raise RuntimeError(f"expected two repaired spec blocks, found {len(spec_blocks)}")
    if any(block.data.get("status") != "pass" for block in spec_blocks):
        raise RuntimeError("not every repaired spec-test block passed")
    spec_identities = {(block.title, block.data.get("task_name")) for block in spec_blocks}
    if spec_identities != {
        ("Logger Fork Safety", None),
        ("Vllm Chat Completions", "vllm_chat_completions"),
    }:
        raise RuntimeError(f"unexpected repaired spec-test identities: {spec_identities}")
    for block in spec_blocks:
        detailed_results = block.data.get("detailed_test_results") or []
        for result in detailed_results:
            if "PASS" not in str(result.get("status", "")).upper():
                raise RuntimeError("spec-test detail contains a non-passing result")
            if result.get("message"):
                raise RuntimeError("passing spec-test detail contains unexpected output")
        block.data["detailed_test_results"] = [
            {
                "test_case": result.get("test_case"),
                "parametrization": result.get("parametrization"),
                "status": result.get("status"),
            }
            for result in detailed_results
        ]
        # Keep the customer handoff compact and response-free. The complete
        # passing workflow log remains in the bounded stage workspace.
        block.data["logs"] = []

    release_matches = [index for index, block in enumerate(release.sections) if _task_name(block) == TASK_NAME]
    if len(release_matches) != 1:
        raise RuntimeError(f"expected one release GPQA block, found {len(release_matches)}")
    release.sections[release_matches[0]] = repair_block

    merged_sections = []
    inserted_benchmarks = False
    inserted_specs = False
    for block in release.sections:
        if block.kind == "benchmarks":
            if not inserted_benchmarks:
                merged_sections.extend(benchmark_blocks)
                inserted_benchmarks = True
            continue
        if block.kind == "spec_tests":
            if not inserted_specs:
                merged_sections.extend(spec_blocks)
                inserted_specs = True
            continue
        merged_sections.append(block)
    if not inserted_benchmarks or not inserted_specs:
        raise RuntimeError("main release did not provide benchmark/spec insertion points")
    release = ReportSchema(metadata=release.metadata, sections=merged_sections)

    for key in ACCEPTANCE_KEYS:
        release.metadata.pop(key, None)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source_release_runtime_spec = release.metadata.get("runtime_model_spec_json")
    source_vllm_commit = release.metadata.get("vllm_commit")
    release.metadata.update(
        {
            "generated_at": generated_at,
            "report_id": f"{runtime_spec['model_id']}_release-repaired_{generated_at.replace(':', '')}",
            "workflow": "release",
            "release_readiness": "ci-nightly-subset-pass",
            "vllm_base_commit": source_vllm_commit,
            "vllm_commit": args.official_vllm_release_commit,
            "tti_commit": args.tti_release_commit,
            "source_release_report_json": str(args.release_report_json),
            "source_release_runtime_model_spec_json": source_release_runtime_spec,
            "validation_runtime_model_spec_json": str(args.runtime_model_spec),
            "handoff_runtime_model_spec_json": (f"{HANDOFF_PATH}/runtime_model_spec_validation.json"),
            "autoport_code_path": AUTOPORT_PATH,
            "release_code_commits": {
                "official_vllm": args.official_vllm_release_commit,
                "tti_client": args.tti_release_commit,
            },
            "context_contract": {
                "path": f"{AUTOPORT_PATH}/doc/context_contract.json",
                "sha256": context_contract_sha256,
                "supported_context": SUPPORTED_CONTEXT,
                "non_aligned_requests_preserved": True,
            },
            "gpqa_harness_recovery": {
                "task_name": TASK_NAME,
                "repair_report_json": str(args.repair_report_json),
                "publisher_revision": PUBLISHER_REVISION,
                "archive_sha256": ARCHIVE_SHA256,
                "diamond_csv_sha256": DIAMOND_SHA256,
                "sample_ids": list(range(7)),
                "raw_samples_copied": False,
            },
            "benchmark_harness_recovery": {
                "repair_report_json": str(args.benchmark_repair_report_json),
                "endpoint": "/v1/completions",
                "backend": "vllm",
                "temperature": 0,
                "expected_rows": 21,
                "raw_evidence_validation": raw_benchmark_evidence,
                "raw_outputs_copied": False,
                "issue_waiver": {
                    "classification": "issue-waived",
                    "scope": {
                        "input_sequence_length": 128,
                        "output_sequence_length": 128,
                        "concurrency": 1,
                        "num_prompts": 8,
                    },
                    "source": str(issue_waiver),
                    "handoff_path": (f"{HANDOFF_PATH}/benchmark_target_ISSUE_WAIVER.md"),
                    "reason": (
                        "the source target combines a concurrency-1 row with "
                        "an aggregate-throughput threshold scaled for a larger batch"
                    ),
                    "unrestricted_performance_readiness": False,
                },
            },
            "spec_test_harness_recovery": {
                "repair_report_json": str(args.spec_repair_report_json),
                "request_timeout_seconds": 300,
                "generated_token_checks_use_reasoning_fallback": True,
                "final_content_required_for_coherence": True,
                "raw_responses_copied": False,
            },
        }
    )

    known_issues = device_spec.get("known_issues") or None
    model_status = runtime_spec.get("status")
    accepted, blockers, categories = acceptance_criteria_check(
        release,
        known_issues=known_issues,
        model_status=model_status,
    )
    release.metadata.update(build_acceptance_export(accepted, blockers, categories, model_status))
    result = ReportGenerator().generate(release, args.output_dir)

    print(f"acceptance={'PASS' if accepted else 'FAIL'}")
    print(f"blocker_keys={','.join(sorted(blockers)) or 'none'}")
    print(f"markdown={result.markdown_path}")
    print(f"json={result.json_path}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

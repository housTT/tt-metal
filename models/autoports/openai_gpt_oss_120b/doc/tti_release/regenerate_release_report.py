#!/usr/bin/env python3
"""Merge validated GPQA, IFEval, benchmark, and spec-test repairs and rerender."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

WORKSPACE_ROOT = Path("/home/ttuser/dev/gpt-oss-20b")
TTI_ROOT = WORKSPACE_ROOT / "tti-release" / "openai_gpt_oss_120b" / "tt-inference-server"
sys.path.insert(0, str(TTI_ROOT))

from report_module import ReportGenerator, ReportSchema, acceptance_criteria_check, build_acceptance_export

TASK_NAME = "gpqa_diamond_cot_zeroshot"
GPQA_DATASET_PATH = "Idavidrein/gpqa"
GPQA_DATASET_NAME = "gpqa_diamond"
GPQA_SAMPLE_IDS = tuple(range(7))
GPQA_DATASET_SAMPLE_COUNT = 198
GPQA_SCORE_METRIC = "exact_match,flexible-extract"
GPQA_SERVICE_PORT = 8000
GPQA_REASONING_EFFORT = "high"
GPQA_MAX_GEN_TOKS = 122880
GPQA_SEED = 42
GPQA_REQUEST_TIMEOUT = 14400
GPQA_NUM_CONCURRENT = 1
EXPECTED_EVAL_TASKS = frozenset({"aime25", TASK_NAME, "mmlu_generative"})
IFEVAL_TASK_NAME = "meta_ifeval"
IFEVAL_LM_EVAL_TASK_NAME = "ifeval"
IFEVAL_DATASET_PATH = "google/IFEval"
IFEVAL_SAMPLE_COUNT = 541
IFEVAL_STRICT_PROMPT_METRIC = "prompt_level_strict_acc,none"
IFEVAL_REASONING_EFFORT = "medium"
IFEVAL_MAX_GEN_TOKS = 4096
IFEVAL_SEED = 42
RELEASE_READINESS_STATUS = "release-readiness-ci-subset-pass"
FINAL_EVAL_TASKS = EXPECTED_EVAL_TASKS | {IFEVAL_TASK_NAME}
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
BENCHMARK_SOURCE_TARGET_FAILURES = frozenset({"functional.tput", "complete.tput", "target.tput"})
BENCHMARK_UNMET_TIER_FAILURES = frozenset({"complete.ttft", "target.ttft"})
EXPECTED_WAIVED_BENCHMARK_FAILURES = BENCHMARK_SOURCE_TARGET_FAILURES | BENCHMARK_UNMET_TIER_FAILURES
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
RAW_IFEVAL_FIELDS = frozenset(
    {
        "config",
        "configs",
        "n-samples",
        "results",
        "total_evaluation_time_seconds",
    }
)
RAW_GPQA_FIELDS = frozenset(
    {
        "config",
        "configs",
        "model_name",
        "n-samples",
        "results",
        "total_evaluation_time_seconds",
    }
)


def _release_readiness_metadata() -> dict[str, str]:
    """Return the exact Stage 11 CI-subset readiness marker for report output."""

    return {"release_readiness": RELEASE_READINESS_STATUS}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _finite_float(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def _positive_timing_float(value, label: str) -> float:
    """Parse a positive duration, including lm-eval's numeric-string format."""

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise RuntimeError(f"{label} is not numeric")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    if result <= 0.0:
        raise RuntimeError(f"{label} must be positive")
    return result


def _workspace_file(path: Path, label: str, workspace_root: Path = WORKSPACE_ROOT) -> Path:
    resolved = path.resolve()
    resolved_root = workspace_root.resolve()
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise RuntimeError(f"{label} is outside the authorized workspace")
    if not resolved.is_file():
        raise RuntimeError(f"{label} does not exist: {resolved}")
    return resolved


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
        # The aggregate parser represents zero request errors as null. Once the
        # raw artifact proves every request completed without an error, retain
        # the authoritative numeric value in the customer-facing row.
        block.data["error_request_count"] = 0
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


def _validate_benchmark_waiver_failures(block) -> list[str]:
    target_checks = block.data.get("target_checks")
    if not isinstance(target_checks, dict):
        raise RuntimeError("waived benchmark row has malformed target checks")
    failed = set()
    for tier, checks in target_checks.items():
        if not isinstance(checks, dict):
            raise RuntimeError(f"waived benchmark {tier} checks are malformed")
        for key, value in checks.items():
            if not key.endswith("_check"):
                continue
            check = _exact_int(value, f"waived benchmark {tier}.{key}")
            if check == 3:
                failed.add(f"{tier}.{key.removesuffix('_check')}")
    if failed != EXPECTED_WAIVED_BENCHMARK_FAILURES:
        raise RuntimeError(
            "waived benchmark failure set does not match the five reviewed " f"subchecks: {sorted(failed)}"
        )
    return sorted(failed)


def _validate_gpqa_generation_config(config: object, label: str) -> None:
    if not isinstance(config, dict):
        raise RuntimeError(f"GPQA {label} is malformed")
    if config.get("reasoning_effort") != GPQA_REASONING_EFFORT:
        raise RuntimeError(f"GPQA {label} does not use high reasoning")
    max_gen_toks = _exact_int(config.get("max_gen_toks"), f"GPQA {label} max_gen_toks")
    if max_gen_toks != GPQA_MAX_GEN_TOKS:
        raise RuntimeError(f"GPQA {label} max_gen_toks={max_gen_toks}, " f"expected {GPQA_MAX_GEN_TOKS}")
    if config.get("do_sample") is not True:
        raise RuntimeError(f"GPQA {label} must set do_sample=true")
    temperature = _finite_float(config.get("temperature"), f"GPQA {label} temperature")
    if temperature != 1.0:
        raise RuntimeError(f"GPQA {label} must set temperature=1")
    if config.get("stream") is not False:
        raise RuntimeError(f"GPQA {label} must set stream=false")
    seed = _exact_int(config.get("seed"), f"GPQA {label} seed")
    if seed != GPQA_SEED:
        raise RuntimeError(f"GPQA {label} seed={seed}, expected {GPQA_SEED}")


def _validate_gpqa_recovery(
    report_path: Path,
    raw_result_path: Path,
    runtime_spec_path: Path,
    *,
    workspace_root: Path = WORKSPACE_ROOT,
):
    """Validate and project the explicit seven-sample GPQA repair run."""

    report_path = _workspace_file(report_path, "GPQA report", workspace_root)
    raw_result_path = _workspace_file(raw_result_path, "GPQA raw result", workspace_root)
    runtime_spec_path = _workspace_file(runtime_spec_path, "GPQA runtime model spec", workspace_root)

    payload = _read_json(report_path)
    if payload.get("acceptance_criteria") is not True:
        raise RuntimeError("GPQA repair workflow did not pass acceptance")
    blockers = payload.get("acceptance_blockers")
    if not isinstance(blockers, dict) or blockers:
        raise RuntimeError("GPQA repair workflow has malformed or nonempty blockers")
    report_metadata = payload.get("metadata")
    if not isinstance(report_metadata, dict):
        raise RuntimeError("GPQA report metadata is malformed")
    if report_metadata.get("workflow") != "evals":
        raise RuntimeError("GPQA report workflow is not evals")
    if report_metadata.get("server_mode") != "API":
        raise RuntimeError("GPQA report did not use an external API server")
    reported_runtime_spec = report_metadata.get("runtime_model_spec_json")
    if not isinstance(reported_runtime_spec, str) or Path(reported_runtime_spec).resolve() != runtime_spec_path:
        raise RuntimeError("GPQA report does not reference the supplied runtime spec")

    report = ReportSchema.from_dict(payload)
    eval_blocks = [block for block in report.sections if block.kind == "evals"]
    if len(eval_blocks) != 1:
        raise RuntimeError(f"expected one GPQA report block, found {len(eval_blocks)}")
    block = eval_blocks[0]
    if _task_name(block) != TASK_NAME or block.targets.get("task_name") != TASK_NAME:
        raise RuntimeError("GPQA report does not preserve canonical task identity")
    report_score = _finite_float(block.data.get("score"), "GPQA report score")
    accuracy_check = _exact_int(block.data.get("accuracy_check"), "GPQA accuracy_check")
    if accuracy_check != 2:
        raise RuntimeError("GPQA accuracy row is not PASS")
    recomputed_acceptance, recomputed_blockers, _ = acceptance_criteria_check(report)
    if not recomputed_acceptance or recomputed_blockers:
        raise RuntimeError("GPQA report block does not independently pass acceptance")

    runtime_wrapper = _read_json(runtime_spec_path)
    runtime_spec = runtime_wrapper.get("runtime_model_spec", runtime_wrapper)
    if not isinstance(runtime_spec, dict):
        raise RuntimeError("GPQA runtime model spec is malformed")
    expected_runtime_identity = {
        "model_id": "id_openai-gpt-oss-120b-autoport_p150x4",
        "model_name": "gpt-oss-120b",
        "inference_engine": "vLLM",
        "device_type": "P150X4",
    }
    for key, expected_value in expected_runtime_identity.items():
        if runtime_spec.get(key) != expected_value:
            raise RuntimeError(f"GPQA runtime {key} identity is incorrect")
    impl = runtime_spec.get("impl")
    if not isinstance(impl, dict) or impl.get("code_path") != AUTOPORT_PATH:
        raise RuntimeError("GPQA runtime spec does not identify the generated autoport")
    metadata = runtime_spec.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("GPQA runtime spec metadata is malformed")
    if metadata.get("autoport_code_path") != AUTOPORT_PATH:
        raise RuntimeError("GPQA runtime metadata does not identify the autoport path")
    if metadata.get("external_autoport_server") is not True:
        raise RuntimeError("GPQA runtime spec is not an external-server run")
    if metadata.get("docker_server_used") is not False:
        raise RuntimeError("GPQA runtime spec used Docker")

    device_spec = runtime_spec.get("device_model_spec")
    if not isinstance(device_spec, dict):
        raise RuntimeError("GPQA device model spec is malformed")
    vllm_args = device_spec.get("vllm_args")
    if not isinstance(vllm_args, dict):
        raise RuntimeError("GPQA vLLM args are malformed")
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
                raise RuntimeError(f"GPQA {label} is not an integer: {value}") from exc
            if str(exact_value) != value:
                raise RuntimeError(f"GPQA {label} is not an integer: {value}")
        else:
            exact_value = _exact_int(value, f"GPQA {label}")
        if exact_value != SUPPORTED_CONTEXT:
            raise RuntimeError(f"GPQA {label} does not preserve {SUPPORTED_CONTEXT}")

    cli_args = runtime_spec.get("cli_args")
    if not isinstance(cli_args, dict):
        raise RuntimeError("GPQA runtime cli_args are malformed")
    if cli_args.get("workflow") != "evals":
        raise RuntimeError("GPQA runtime workflow is not evals")
    if cli_args.get("model") != "gpt-oss-120b":
        raise RuntimeError("GPQA runtime model identity is incorrect")
    if cli_args.get("engine") != "vLLM":
        raise RuntimeError("GPQA runtime engine is not vLLM")
    if cli_args.get("device") != "p150x4" or cli_args.get("tt_device") != "p150x4":
        raise RuntimeError("GPQA runtime device identity is not p150x4")
    if cli_args.get("docker_server") is not False:
        raise RuntimeError("GPQA runtime docker_server must be false")
    if cli_args.get("local_server") is not False:
        raise RuntimeError("GPQA runtime local_server must be false")
    if cli_args.get("server_url") != f"http://127.0.0.1:{GPQA_SERVICE_PORT}":
        raise RuntimeError("GPQA runtime server_url is incorrect")
    if cli_args.get("service_port") not in (
        GPQA_SERVICE_PORT,
        str(GPQA_SERVICE_PORT),
    ):
        raise RuntimeError("GPQA runtime service_port is incorrect")
    if cli_args.get("limit_samples_mode") is not None:
        raise RuntimeError("GPQA runtime unexpectedly applied a sample limit")
    if cli_args.get("disable_trace_capture") is not True:
        raise RuntimeError("GPQA runtime did not disable trace capture")
    try:
        eval_samples = json.loads(cli_args.get("eval_samples"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("GPQA runtime eval_samples is malformed") from exc
    expected_eval_samples = {TASK_NAME: list(GPQA_SAMPLE_IDS)}
    if eval_samples != expected_eval_samples:
        raise RuntimeError("GPQA runtime eval_samples must be exactly IDs 0..6")

    raw = _read_json_fields(raw_result_path, RAW_GPQA_FIELDS)
    missing_fields = RAW_GPQA_FIELDS - raw.keys()
    if missing_fields:
        raise RuntimeError(f"GPQA raw result is missing fields: {sorted(missing_fields)}")
    if raw["model_name"] != "openai/gpt-oss-120b":
        raise RuntimeError("GPQA raw result has an unexpected model identity")
    run_config = raw["config"]
    if not isinstance(run_config, dict) or "limit" not in run_config or run_config["limit"] is not None:
        raise RuntimeError("GPQA raw result is not an explicit unbounded run")
    if run_config.get("model") != "local-chat-completions":
        raise RuntimeError("GPQA raw result did not use chat completions")
    model_args = run_config.get("model_args")
    if not isinstance(model_args, dict):
        raise RuntimeError("GPQA raw model_args are malformed")
    if model_args.get("model") != "openai/gpt-oss-120b":
        raise RuntimeError("GPQA raw model_args have an unexpected model")
    if model_args.get("base_url") != (f"http://127.0.0.1:{GPQA_SERVICE_PORT}/v1/chat/completions"):
        raise RuntimeError("GPQA raw result used an unexpected API endpoint")
    if _exact_int(model_args.get("timeout"), "GPQA raw request timeout") != (GPQA_REQUEST_TIMEOUT):
        raise RuntimeError("GPQA raw result used an unexpected request timeout")
    if _exact_int(model_args.get("num_concurrent"), "GPQA raw concurrency") != (GPQA_NUM_CONCURRENT):
        raise RuntimeError("GPQA raw result used unexpected concurrency")
    _validate_gpqa_generation_config(run_config.get("gen_kwargs"), "raw run gen_kwargs")

    expected_raw_keys = {TASK_NAME}
    for label, field in (
        ("configs", raw["configs"]),
        ("n-samples", raw["n-samples"]),
        ("results", raw["results"]),
    ):
        if not isinstance(field, dict) or set(field) != expected_raw_keys:
            raise RuntimeError(f"GPQA raw {label} must contain only canonical task {TASK_NAME}")
    canonical_config = raw["configs"][TASK_NAME]
    if not isinstance(canonical_config, dict):
        raise RuntimeError("GPQA canonical task config is malformed")
    if canonical_config.get("task") != TASK_NAME:
        raise RuntimeError("GPQA raw task identity is not canonical")
    if canonical_config.get("dataset_path") != GPQA_DATASET_PATH:
        raise RuntimeError("GPQA raw result has an unexpected dataset path")
    if canonical_config.get("dataset_name") != GPQA_DATASET_NAME:
        raise RuntimeError("GPQA raw result has an unexpected dataset name")
    _validate_gpqa_generation_config(
        canonical_config.get("generation_kwargs"),
        "canonical task generation_kwargs",
    )

    raw_sample_counts = raw["n-samples"][TASK_NAME]
    if not isinstance(raw_sample_counts, dict):
        raise RuntimeError("GPQA raw sample counts are malformed")
    for count_name in ("original", "effective"):
        count = _exact_int(raw_sample_counts.get(count_name), f"GPQA raw {count_name} samples")
        if count != GPQA_DATASET_SAMPLE_COUNT:
            raise RuntimeError(f"GPQA raw {count_name} samples={count}, " f"expected {GPQA_DATASET_SAMPLE_COUNT}")

    raw_results = raw["results"][TASK_NAME]
    if not isinstance(raw_results, dict):
        raise RuntimeError("GPQA raw metrics are malformed")
    raw_score = _finite_float(raw_results.get(GPQA_SCORE_METRIC), f"GPQA raw {GPQA_SCORE_METRIC}")
    if not 0.0 <= raw_score <= 1.0:
        raise RuntimeError("GPQA flexible-extract metric is outside [0, 1]")
    if not math.isclose(report_score, raw_score * 100.0, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("GPQA report score does not match the raw aggregate metric")
    total_evaluation_time_seconds = _positive_timing_float(
        raw["total_evaluation_time_seconds"],
        "GPQA total_evaluation_time_seconds",
    )
    mean_seconds_per_task = total_evaluation_time_seconds / len(GPQA_SAMPLE_IDS)
    block.data["mean_seconds_per_task"] = mean_seconds_per_task

    recovery_metadata = {
        "task_name": TASK_NAME,
        "canonical_task_name": TASK_NAME,
        "dataset_path": GPQA_DATASET_PATH,
        "dataset_name": GPQA_DATASET_NAME,
        "score_metric": GPQA_SCORE_METRIC,
        "publisher_revision": PUBLISHER_REVISION,
        "archive_sha256": ARCHIVE_SHA256,
        "diamond_csv_sha256": DIAMOND_SHA256,
        "sample_ids": list(GPQA_SAMPLE_IDS),
        "sample_count": len(GPQA_SAMPLE_IDS),
        "scope": {
            "selection_source": "runtime_model_spec.cli_args.eval_samples",
            "raw_original_samples": GPQA_DATASET_SAMPLE_COUNT,
            "raw_effective_samples": GPQA_DATASET_SAMPLE_COUNT,
            "raw_count_semantics": "dataset-size-not-selected-request-count",
            "generation_policy": {
                "reasoning_effort": GPQA_REASONING_EFFORT,
                "max_gen_toks": GPQA_MAX_GEN_TOKS,
                "do_sample": True,
                "temperature": 1.0,
                "stream": False,
                "seed": GPQA_SEED,
                "request_timeout_seconds": GPQA_REQUEST_TIMEOUT,
                "num_concurrent": GPQA_NUM_CONCURRENT,
            },
        },
        "server": {
            "mode": "external",
            "docker_server": False,
            "local_server": False,
            "service_port": GPQA_SERVICE_PORT,
            "workflow": "evals",
            "autoport_code_path": AUTOPORT_PATH,
            "supported_context": SUPPORTED_CONTEXT,
        },
        "timing": {
            "source": "raw_result.total_evaluation_time_seconds",
            "semantics": "lm-eval-evaluation-time-per-selected-request",
            "total_evaluation_time_seconds": total_evaluation_time_seconds,
            "mean_seconds_per_task": mean_seconds_per_task,
            "sample_count": len(GPQA_SAMPLE_IDS),
        },
        "report_paths": {
            "report_json": str(report_path),
            "raw_result_json": str(raw_result_path),
            "runtime_model_spec_json": str(runtime_spec_path),
        },
        "artifact_sha256": {
            "report_json": _sha256_file(report_path),
            "raw_result_json": _sha256_file(raw_result_path),
            "runtime_model_spec_json": _sha256_file(runtime_spec_path),
        },
        "raw_samples_copied": False,
    }
    return block, recovery_metadata


def _validate_ifeval_generation_config(config: object, label: str) -> None:
    """Require the deterministic GPT-OSS policy used for reference parity."""

    if not isinstance(config, dict):
        raise RuntimeError(f"IFEval {label} is malformed")
    if config.get("reasoning_effort") != IFEVAL_REASONING_EFFORT:
        raise RuntimeError(f"IFEval {label} does not use {IFEVAL_REASONING_EFFORT} reasoning")
    max_gen_toks = _exact_int(config.get("max_gen_toks"), f"IFEval {label} max_gen_toks")
    if max_gen_toks != IFEVAL_MAX_GEN_TOKS:
        raise RuntimeError(f"IFEval {label} max_gen_toks={max_gen_toks}, " f"expected {IFEVAL_MAX_GEN_TOKS}")
    if config.get("do_sample") is not False:
        raise RuntimeError(f"IFEval {label} must set do_sample=false")
    temperature = _finite_float(config.get("temperature"), f"IFEval {label} temperature")
    if temperature != 0.0:
        raise RuntimeError(f"IFEval {label} must set temperature=0")
    seed = _exact_int(config.get("seed"), f"IFEval {label} seed")
    if seed != IFEVAL_SEED:
        raise RuntimeError(f"IFEval {label} seed={seed}, expected {IFEVAL_SEED}")


def _validate_ifeval_recovery(
    report_path: Path,
    raw_result_path: Path,
    *,
    workspace_root: Path = WORKSPACE_ROOT,
):
    """Validate and project a separately completed canonical Google IFEval run."""

    report_path = _workspace_file(report_path, "IFEval report", workspace_root)
    raw_result_path = _workspace_file(raw_result_path, "IFEval raw result", workspace_root)

    payload = _read_json(report_path)
    if payload.get("acceptance_criteria") is not True:
        raise RuntimeError("IFEval workflow did not pass acceptance")
    blockers = payload.get("acceptance_blockers")
    if not isinstance(blockers, dict) or blockers:
        raise RuntimeError("IFEval workflow has malformed or nonempty blockers")

    report = ReportSchema.from_dict(payload)
    eval_blocks = [block for block in report.sections if block.kind == "evals"]
    if len(eval_blocks) != 1:
        raise RuntimeError(f"expected one IFEval report block, found {len(eval_blocks)}")
    block = eval_blocks[0]
    if _task_name(block) != IFEVAL_TASK_NAME:
        raise RuntimeError("IFEval report does not use logical task meta_ifeval")
    if block.data.get("lm_eval_task_name") != IFEVAL_LM_EVAL_TASK_NAME:
        raise RuntimeError("IFEval report does not prove canonical task provenance")
    if block.targets.get("task_name") != IFEVAL_TASK_NAME:
        raise RuntimeError("IFEval report target does not use logical task meta_ifeval")

    report_score = _finite_float(block.data.get("score"), "IFEval report score")
    accuracy_check = _exact_int(block.data.get("accuracy_check"), "IFEval accuracy_check")
    if accuracy_check != 2:
        raise RuntimeError("IFEval accuracy row is not PASS")
    recomputed_acceptance, recomputed_blockers, _ = acceptance_criteria_check(report)
    if not recomputed_acceptance or recomputed_blockers:
        raise RuntimeError("IFEval report block does not independently pass acceptance")

    raw = _read_json_fields(raw_result_path, RAW_IFEVAL_FIELDS)
    missing_fields = RAW_IFEVAL_FIELDS - raw.keys()
    if missing_fields:
        raise RuntimeError(f"IFEval raw result is missing fields: {sorted(missing_fields)}")
    total_evaluation_time_seconds = _positive_timing_float(
        raw["total_evaluation_time_seconds"],
        "IFEval total_evaluation_time_seconds",
    )

    run_config = raw["config"]
    if not isinstance(run_config, dict) or "limit" not in run_config:
        raise RuntimeError("IFEval raw result has no explicit run limit")
    if run_config["limit"] is not None:
        raise RuntimeError("IFEval raw result is not a full unbounded run")

    model_args = run_config.get("model_args")
    if not isinstance(model_args, dict):
        raise RuntimeError("IFEval raw run model_args are malformed")
    run_max_length = _exact_int(model_args.get("max_length"), "IFEval raw run max_length")
    if run_max_length != SUPPORTED_CONTEXT:
        raise RuntimeError(f"IFEval raw run max_length={run_max_length}, expected {SUPPORTED_CONTEXT}")
    _validate_ifeval_generation_config(run_config.get("gen_kwargs"), "raw run gen_kwargs")

    expected_raw_keys = {IFEVAL_LM_EVAL_TASK_NAME}
    for label, field in (
        ("configs", raw["configs"]),
        ("n-samples", raw["n-samples"]),
        ("results", raw["results"]),
    ):
        if not isinstance(field, dict) or set(field) != expected_raw_keys:
            raise RuntimeError(f"IFEval raw {label} must contain only canonical task ifeval")

    canonical_config = raw["configs"][IFEVAL_LM_EVAL_TASK_NAME]
    if not isinstance(canonical_config, dict):
        raise RuntimeError("IFEval canonical task config is malformed")
    if canonical_config.get("task") != IFEVAL_LM_EVAL_TASK_NAME:
        raise RuntimeError("IFEval raw task identity is not canonical ifeval")
    if canonical_config.get("dataset_path") != IFEVAL_DATASET_PATH:
        raise RuntimeError("IFEval raw result does not use google/IFEval")

    task_metadata = canonical_config.get("metadata")
    if not isinstance(task_metadata, dict):
        raise RuntimeError("IFEval canonical task metadata is malformed")
    task_max_length = _exact_int(task_metadata.get("max_length"), "IFEval canonical task max_length")
    if task_max_length != SUPPORTED_CONTEXT:
        raise RuntimeError("IFEval canonical task max_length=" f"{task_max_length}, expected {SUPPORTED_CONTEXT}")
    _validate_ifeval_generation_config(
        canonical_config.get("generation_kwargs"),
        "canonical task generation_kwargs",
    )

    sample_counts = raw["n-samples"][IFEVAL_LM_EVAL_TASK_NAME]
    if not isinstance(sample_counts, dict):
        raise RuntimeError("IFEval raw sample counts are malformed")
    for count_name in ("original", "effective"):
        count = _exact_int(sample_counts.get(count_name), f"IFEval {count_name} samples")
        if count != IFEVAL_SAMPLE_COUNT:
            raise RuntimeError(f"IFEval {count_name} samples={count}, expected {IFEVAL_SAMPLE_COUNT}")

    raw_results = raw["results"][IFEVAL_LM_EVAL_TASK_NAME]
    if not isinstance(raw_results, dict):
        raise RuntimeError("IFEval raw metrics are malformed")
    strict_prompt = _finite_float(
        raw_results.get(IFEVAL_STRICT_PROMPT_METRIC),
        f"IFEval raw {IFEVAL_STRICT_PROMPT_METRIC}",
    )
    if not 0.0 <= strict_prompt <= 1.0:
        raise RuntimeError("IFEval strict prompt metric is outside [0, 1]")
    expected_report_score = strict_prompt * 100.0
    if not math.isclose(report_score, expected_report_score, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("IFEval report score does not match the raw strict prompt metric")

    mean_seconds_per_task = total_evaluation_time_seconds / IFEVAL_SAMPLE_COUNT
    block.data["mean_seconds_per_task"] = mean_seconds_per_task

    recovery_metadata = {
        "logical_task_name": IFEVAL_TASK_NAME,
        "canonical_task_name": IFEVAL_LM_EVAL_TASK_NAME,
        "dataset_path": IFEVAL_DATASET_PATH,
        "scope": {
            "limit": None,
            "original_samples": IFEVAL_SAMPLE_COUNT,
            "effective_samples": IFEVAL_SAMPLE_COUNT,
            "strict_prompt_metric": IFEVAL_STRICT_PROMPT_METRIC,
            "max_length": SUPPORTED_CONTEXT,
            "generation_policy": {
                "reasoning_effort": IFEVAL_REASONING_EFFORT,
                "max_gen_toks": IFEVAL_MAX_GEN_TOKS,
                "do_sample": False,
                "temperature": 0.0,
                "seed": IFEVAL_SEED,
            },
        },
        "report_paths": {
            "report_json": str(report_path),
            "raw_result_json": str(raw_result_path),
        },
        "timing": {
            "total_evaluation_time_seconds": total_evaluation_time_seconds,
            "mean_seconds_per_task": mean_seconds_per_task,
            "sample_count": IFEVAL_SAMPLE_COUNT,
        },
        "raw_samples_copied": False,
    }
    return block, recovery_metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-report-json", required=True, type=Path)
    parser.add_argument("--repair-report-json", required=True, type=Path)
    parser.add_argument("--gpqa-raw-result-json", required=True, type=Path)
    parser.add_argument("--gpqa-runtime-model-spec", required=True, type=Path)
    parser.add_argument("--benchmark-repair-report-json", required=True, type=Path)
    parser.add_argument("--benchmark-raw-dir", required=True, type=Path)
    parser.add_argument("--benchmark-issue-waiver", required=True, type=Path)
    parser.add_argument("--spec-repair-report-json", required=True, type=Path)
    parser.add_argument("--ifeval-report-json", required=True, type=Path)
    parser.add_argument("--ifeval-raw-result-json", required=True, type=Path)
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
    benchmark_payload = _read_json(args.benchmark_repair_report_json)
    spec_payload = _read_json(args.spec_repair_report_json)
    release = ReportSchema.from_dict(release_payload)
    benchmark_repair = ReportSchema.from_dict(benchmark_payload)
    spec_repair = ReportSchema.from_dict(spec_payload)
    repair_block, gpqa_recovery = _validate_gpqa_recovery(
        args.repair_report_json,
        args.gpqa_raw_result_json,
        args.gpqa_runtime_model_spec,
    )
    ifeval_block, ifeval_recovery = _validate_ifeval_recovery(args.ifeval_report_json, args.ifeval_raw_result_json)

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
    waived_benchmark_failures = _validate_benchmark_waiver_failures(benchmark_blocks[0])
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
    last_eval_index = max(index for index, block in enumerate(release.sections) if block.kind == "evals")
    for index, block in enumerate(release.sections):
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
        if index == last_eval_index:
            merged_sections.append(ifeval_block)
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
            **_release_readiness_metadata(),
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
            "gpqa_harness_recovery": gpqa_recovery,
            "ifeval_harness_recovery": ifeval_recovery,
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
                        "three aggregate-throughput checks use a batch-scaled "
                        "source target, while complete/target TTFT are genuine "
                        "unmet higher performance tiers"
                    ),
                    "failed_subchecks": waived_benchmark_failures,
                    "disposition": {
                        "source_target_defect": sorted(BENCHMARK_SOURCE_TARGET_FAILURES),
                        "unmet_performance_tiers": sorted(BENCHMARK_UNMET_TIER_FAILURES),
                    },
                    "acceptance_basis": "informational-under-experimental-status",
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
    final_eval_tasks = {task_name for block in release.sections if (task_name := _task_name(block)) is not None}
    if final_eval_tasks != FINAL_EVAL_TASKS:
        raise RuntimeError("merged release report has unexpected eval coverage: " f"{sorted(final_eval_tasks)}")
    result = ReportGenerator().generate(release, args.output_dir)

    print(f"acceptance={'PASS' if accepted else 'FAIL'}")
    print(f"blocker_keys={','.join(sorted(blockers)) or 'none'}")
    print(f"markdown={result.markdown_path}")
    print(f"json={result.json_path}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

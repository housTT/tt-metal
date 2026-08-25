# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "doc" / "post_optimization_eval" / "build_artifact.py"
SPEC = importlib.util.spec_from_file_location("ornith_build_eval_artifact", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
artifact_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifact_builder)


def _ifeval_result(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "results": {
                    "ifeval": {
                        "prompt_level_strict_acc,none": 0.5,
                        "inst_level_strict_acc,none": 0.6,
                        "prompt_level_loose_acc,none": 0.7,
                        "inst_level_loose_acc,none": 0.8,
                    }
                },
                "configs": {
                    "ifeval": {
                        "num_fewshot": 0,
                        "generation_kwargs": {"temperature": 0.6},
                        "metadata": {"num_concurrent": 8},
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _samples(path: Path, doc_ids: list[int]) -> None:
    path.write_text(
        "".join(json.dumps({"doc_id": doc_id}) + "\n" for doc_id in doc_ids),
        encoding="utf-8",
    )


def test_quality_task_requires_and_records_the_fixed_subset(tmp_path: Path) -> None:
    result_path = tmp_path / "results.json"
    samples_path = tmp_path / "samples.jsonl"
    _ifeval_result(result_path)
    _samples(samples_path, list(range(28)))

    result = artifact_builder._quality_task("ifeval", result_path, samples_path, full_samples=541)

    assert result["doc_ids"] == list(range(28))
    assert result["score_percent"] == pytest.approx(65.0)
    assert result["score_definition"] == "mean_of_four_reported_accuracies"
    assert result["classification"] == "measured_only_no_gpu_or_published_reference"
    assert result["response_diagnostics"] == {
        "nonempty_filtered_responses": 0,
        "empty_filtered_responses": 28,
    }


def test_quality_task_rejects_a_different_subset(tmp_path: Path, expect_error) -> None:
    result_path = tmp_path / "results.json"
    samples_path = tmp_path / "samples.jsonl"
    _ifeval_result(result_path)
    _samples(samples_path, list(range(1, 29)))

    with expect_error(ValueError, "fixed document IDs"):
        artifact_builder._quality_task("ifeval", result_path, samples_path, full_samples=541)


def test_junit_summary_is_compact_and_hash_anchored(tmp_path: Path) -> None:
    junit_path = tmp_path / "tests.junit.xml"
    junit_path.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="1" time="1.25">'
        '<testcase name="one"/><testcase name="two"/></testsuite></testsuites>',
        encoding="utf-8",
    )

    result = artifact_builder._junit(junit_path, "host")

    assert result["status"] == "pass"
    assert result["tests"] == 2
    assert result["skipped"] == 1
    assert result["elapsed_s"] == 1.25
    assert "test_names" not in result
    assert len(result["raw_sha256"]) == 64

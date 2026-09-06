# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from models.common.readiness_check.check_degenerate_output import Report, check_completion, main


def _check_text(text: str) -> Report:
    report = Report()
    check_completion(report, artifact=Path("artifact.json"), label="completion", text=text)
    return report


def test_clean_text_has_no_findings():
    report = _check_text(" ".join(f"word{index}" for index in range(60)))

    assert report.exit_code == 0
    assert report.findings == []


def test_adjacent_token_duplication_is_critical():
    text = " ".join(word for index in range(20) for word in (f"word{index}", f"word{index}"))
    report = _check_text(text)

    assert report.exit_code == 2
    assert [(finding.severity, finding.metric) for finding in report.findings] == [("critical", "adjacent_duplication")]


def test_model_dir_discovers_standard_autoregressive_artifact(tmp_path):
    model_dir = tmp_path / "model"
    artifact_dir = model_dir / "doc" / "full_model" / "artifacts" / "qualitative"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "autoregressive_meta.json").write_text(
        json.dumps({"tt": {"token_ids": list(range(64))}}), encoding="utf-8"
    )

    assert main(["--model-dir", str(model_dir), "--scope", "autoregressive", "--missing-artifacts", "critical"]) == 0


def test_empty_autoregressive_payload_is_missing_evidence(tmp_path):
    model_dir = tmp_path / "model"
    artifact_dir = model_dir / "doc" / "full_model" / "artifacts" / "qualitative"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "autoregressive_meta.json").write_text(json.dumps({"tt": {"token_ids": []}}), encoding="utf-8")

    assert main(["--model-dir", str(model_dir), "--scope", "autoregressive", "--missing-artifacts", "critical"]) == 2


def test_token_collapse_is_checked_when_text_is_present(tmp_path):
    model_dir = tmp_path / "model"
    artifact_dir = model_dir / "doc" / "full_model" / "artifacts" / "qualitative"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "autoregressive_meta.json").write_text(
        json.dumps({"tt": {"token_ids": [7] * 64}}), encoding="utf-8"
    )
    (artifact_dir / "tt_completion.txt").write_text("onefusedwordwithoutspaces", encoding="utf-8")

    assert main(["--model-dir", str(model_dir), "--scope", "autoregressive", "--missing-artifacts", "critical"]) == 2


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"tt": {"token_ids": "not-a-list"}},
        {"tt": {"token_ids": {"0": 7}}},
        {"tt": {"token_ids": [None]}},
    ],
)
def test_malformed_autoregressive_payload_is_critical(tmp_path, payload):
    model_dir = tmp_path / "model"
    artifact_dir = model_dir / "doc" / "full_model" / "artifacts" / "qualitative"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "autoregressive_meta.json").write_text(json.dumps(payload), encoding="utf-8")

    assert main(["--model-dir", str(model_dir), "--scope", "autoregressive", "--missing-artifacts", "critical"]) == 2


@pytest.mark.parametrize(("severity", "expected_exit"), [("advisory", 1), ("critical", 2)])
def test_missing_artifact_policy_controls_exit_code(tmp_path, severity, expected_exit):
    assert main(["--root", str(tmp_path / "missing"), "--missing-artifacts", severity]) == expected_exit

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import models.common.readiness_check.run_teacher_forcing as runner

from models.common.readiness_check.run_teacher_forcing import _compute_perf_stats, _run_one_entry
from models.common.readiness_check.schema import Reference, ReferenceEntry
from models.common.readiness_check.teacher_forcing import TokenAccuracy


class _CallbackGenerator:
    def __init__(self, predictions):
        self.predictions = predictions
        self.generate_calls = 0
        self.reset_calls = 0
        self.teardown_calls = 0

    def generate(self, prompt_token_ids, max_new_tokens, *, next_input=None, enable_trace=True):
        assert enable_trace is True
        self.generate_calls += 1
        emitted = []
        for step, prediction in enumerate(self.predictions[:max_new_tokens]):
            emitted.append(prediction)
            if next_input is not None:
                next_input(step, prediction)
        return emitted

    def reset(self):
        self.reset_calls += 1

    def teardown(self):
        self.teardown_calls += 1


class _NoCallbackGenerator:
    def generate(self, prompt_token_ids, max_new_tokens, *, next_input=None, enable_trace=True):
        assert enable_trace is True
        return [11, 12, 13][:max_new_tokens]


def _make_accuracy() -> TokenAccuracy:
    reference = Reference(
        k=5,
        hf_model_id="unit-test",
        token_ids_meta={},
        entries=[
            ReferenceEntry(
                prompt_text="prompt",
                prompt_tokens=torch.tensor([[101, 102]], dtype=torch.int64),
                generated_tokens=torch.tensor([[11, 12, 13]], dtype=torch.int64),
                topk_tokens=torch.tensor(
                    [
                        [11, 1, 2, 3, 4],
                        [12, 1, 2, 3, 4],
                        [13, 1, 2, 3, 4],
                    ],
                    dtype=torch.int32,
                ),
                tf_prompt_len=2,
            )
        ],
    )
    return TokenAccuracy(reference)


def test_run_one_entry_scores_full_teacher_forcing_run():
    stats = _run_one_entry(generator=_CallbackGenerator([11, 12, 13]), acc=_make_accuracy(), entry_idx=0)

    assert stats["total"] == 3
    assert stats["matches_top1"] == 3
    assert stats["top1"] == 1.0


def test_run_one_entry_fails_when_generate_stops_before_reference_length():
    with pytest.raises(RuntimeError, match="produced 2/3 predictions"):
        _run_one_entry(generator=_CallbackGenerator([11, 12]), acc=_make_accuracy(), entry_idx=0)


def test_run_one_entry_fails_when_generate_never_calls_next_input():
    with pytest.raises(RuntimeError, match="produced 0/3 predictions"):
        _run_one_entry(generator=_NoCallbackGenerator(), acc=_make_accuracy(), entry_idx=0)


def test_perf_stats_separate_trace_capture_from_warmed_replay():
    stats = _compute_perf_stats(
        timing={
            "start_s": 0.0,
            "first_token_s": 10.0,
            "first_decode_token_s": 12.0,
            "last_decode_token_s": 15.0,
        },
        end_s=15.0,
        token_count=5,
    )

    assert stats["decode_setup_ms"] == 2000.0
    assert stats["decode_with_capture_tokens"] == 4.0
    assert stats["decode_with_capture_t/s/u"] == 0.8
    assert stats["decode_tokens"] == 3.0
    assert stats["decode_t/s/u"] == 1.0


def test_run_teacher_forcing_warms_and_resets_before_measured_pass(monkeypatch, tmp_path):
    generator = _CallbackGenerator([11, 12, 13])
    monkeypatch.setattr(runner, "_import_build_generator", lambda _: lambda **kwargs: generator)
    monkeypatch.setattr(runner, "TokenAccuracy", lambda _: _make_accuracy())

    result = runner.run_teacher_forcing(
        model_dir=tmp_path,
        reference_path=tmp_path / "reference.refpt",
        mesh_device=object(),
        warmup_repeats=1,
    )

    assert result[0]["top1"] == 1.0
    assert generator.generate_calls == 2
    assert generator.reset_calls == 1
    assert generator.teardown_calls == 1


def test_cli_forwards_machine_readable_output_and_runtime_metadata(monkeypatch, tmp_path):
    captured = {}
    mesh = object()
    monkeypatch.setattr(runner, "open_readiness_mesh_device", lambda *args: mesh)
    monkeypatch.setattr(runner, "close_readiness_mesh_device", lambda *args: None)

    def fake_run_teacher_forcing(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(runner, "run_teacher_forcing", fake_run_teacher_forcing)
    output = tmp_path / "teacher.json"
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run_teacher_forcing",
            "--model-dir",
            str(tmp_path),
            "--reference",
            str(tmp_path / "reference.refpt"),
            "--mesh-device",
            "P300",
            "--fabric-config",
            "FABRIC_1D_RING",
            "--trace-region-size",
            "1500000000",
            "--output-json",
            str(output),
        ],
    )
    runner._main()

    assert captured["output_json_path"] == output.resolve()
    assert captured["warmup_repeats"] == 0
    assert captured["runtime"] == {
        "mesh_device": "P300",
        "fabric_config": "FABRIC_1D_RING",
        "trace_region_size": 1_500_000_000,
        "decode_trace_enabled": True,
    }

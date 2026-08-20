# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import models.common.readiness_check.run_prefill_check as runner


def test_cli_forwards_machine_readable_output_and_runtime_metadata(monkeypatch, tmp_path):
    captured = {}
    mesh = object()
    monkeypatch.setattr(runner, "open_readiness_mesh_device", lambda *args: mesh)
    monkeypatch.setattr(runner, "close_readiness_mesh_device", lambda *args: None)

    def fake_run_prefill_check(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(runner, "run_prefill_check", fake_run_prefill_check)
    output = tmp_path / "prefill.json"
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run_prefill_check",
            "--model-dir",
            str(tmp_path),
            "--reference",
            str(tmp_path / "reference.refpt"),
            "--mesh-device",
            "P300",
            "--fabric-config",
            "FABRIC_1D_RING",
            "--output-json",
            str(output),
        ],
    )
    runner._main()

    assert captured["output_json_path"] == output.resolve()
    assert captured["runtime"] == {
        "mesh_device": "P300",
        "fabric_config": "FABRIC_1D_RING",
        "trace_region_size": None,
    }

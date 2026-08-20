# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

from models.common.readiness_check.metrics_json import aggregate_accuracy, runtime_metadata, write_metrics_json


def test_metrics_json_captures_accuracy_runtime_and_nested_output(tmp_path):
    entries = [
        {
            "top1": 0.5,
            "top5": 1.0,
            "top100": 1.0,
            "matches_top1": 1,
            "matches_top5": 2,
            "matches_top100": 2,
            "total": 2,
            "k": 100,
        },
        {
            "top1": 1.0,
            "top5": 1.0,
            "top100": 1.0,
            "matches_top1": 1,
            "matches_top5": 1,
            "matches_top100": 1,
            "total": 1,
            "k": 100,
        },
    ]
    aggregate = aggregate_accuracy(entries)
    assert aggregate == {
        "top1": 2 / 3,
        "top5": 1.0,
        "top100": 1.0,
        "matches_top1": 2,
        "matches_top5": 3,
        "matches_top100": 3,
        "total": 3,
        "k": 100,
    }

    mesh = SimpleNamespace(shape=(1, 4), get_num_devices=lambda: 4)
    output = tmp_path / "nested" / "metrics.json"
    write_metrics_json(
        output,
        {
            "runtime": runtime_metadata(mesh, cli={"fabric_config": "FABRIC_1D_RING"}),
            "entries": entries,
            "aggregate": aggregate,
        },
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["runtime"]["mesh_device_count"] == 4
    assert report["runtime"]["mesh_shape"] == [1, 4]
    assert report["runtime"]["fabric_config"] == "FABRIC_1D_RING"
    assert report["aggregate"]["top100"] == 1.0


def test_aggregate_accuracy_returns_none_for_empty_entries():
    assert aggregate_accuracy([]) is None

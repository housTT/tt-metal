# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable evidence helpers for readiness checks."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def aggregate_accuracy(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Aggregate the stable accuracy fields shared by readiness runners."""
    total = sum(int(entry["total"]) for entry in entries)
    if not total:
        return None
    return {
        "top1": sum(int(entry["matches_top1"]) for entry in entries) / total,
        "top5": sum(int(entry["matches_top5"]) for entry in entries) / total,
        "top100": sum(int(entry["matches_top100"]) for entry in entries) / total,
        "matches_top1": sum(int(entry["matches_top1"]) for entry in entries),
        "matches_top5": sum(int(entry["matches_top5"]) for entry in entries),
        "matches_top100": sum(int(entry["matches_top100"]) for entry in entries),
        "total": total,
        "k": int(entries[0]["k"]),
    }


def runtime_metadata(mesh_device: Any, *, cli: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return JSON-safe runtime and physical-topology metadata."""
    try:
        import ttnn

        ttnn_version = getattr(ttnn, "__version__", None)
    except ImportError:
        ttnn_version = None

    device_count = None
    get_num_devices = getattr(mesh_device, "get_num_devices", None)
    if callable(get_num_devices):
        device_count = int(get_num_devices())

    mesh_shape = getattr(mesh_device, "shape", None)
    metadata: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "ttnn_version": ttnn_version,
        "mesh_device_count": device_count,
        "mesh_shape": list(mesh_shape) if mesh_shape is not None else None,
    }
    if cli:
        metadata.update(cli)
    return metadata


def write_metrics_json(path: Path, report: Mapping[str, Any]) -> None:
    """Write a deterministic, human-readable JSON metrics artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

#!/usr/bin/env python3
"""Extract compact exact-host runtime proof from an autoport server log."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METRIC_MARKER = "QWEN38_VLLM_METRICS "


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_latest_metrics(log_path: Path) -> tuple[int, dict]:
    latest = None
    with log_path.open(errors="replace") as log:
        for line_number, line in enumerate(log, 1):
            if METRIC_MARKER in line:
                latest = (line_number, json.loads(line.split(METRIC_MARKER, 1)[1]))
    if latest is None:
        raise RuntimeError(f"No {METRIC_MARKER.strip()} record in {log_path}")
    return latest


def _select(mapping: dict, keys: tuple[str, ...]) -> dict:
    return {key: mapping[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--host-contract", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    line_number, metrics = _load_latest_metrics(args.server_log)
    gauges = metrics["host_gauges"]
    service = metrics["host_service"]
    declared = metrics["runtime_fallback"]["declared_host_work"]
    prohibited = metrics["runtime_fallback"]["prohibited_host_work"]

    expected_gauges = {
        "expert_layers": 48.0,
        "expert_packed_host_entries": 24576.0,
        "expert_packed_host_bytes": 68080435200.0,
        "expert_device_slot_capacity": 480.0,
        "expert_device_bytes_per_rank": 1459814400.0,
    }
    gauge_match = all(gauges.get(key) == value for key, value in expected_gauges.items())
    expert_miss_proven = all(
        service.get(key, 0) > 0
        for key in (
            "expert_misses",
            "expert_packed_host_hits",
            "expert_direct_slot_h2d_copies",
            "expert_direct_slot_h2d_bytes",
        )
    ) and service["expert_packed_host_hits"] == service["expert_misses"]
    ple_path_proven = all(
        service.get(key, 0) > 0
        for key in (
            "ple_lookup_calls",
            "ple_selected_rows",
            "ple_table_bytes_read",
            "ple_device_h2d_bytes",
        )
    )
    forbidden_clear = all(value is False for value in prohibited.values())

    evidence = {
        "schema_version": 1,
        "event": metrics.get("event"),
        "completed_requests": metrics.get("completed_requests"),
        "source": {
            "server_log": str(args.server_log),
            "server_log_sha256": _sha256(args.server_log),
            "metrics_line": line_number,
            "host_weight_contract": str(args.host_contract),
            "host_weight_contract_sha256": _sha256(args.host_contract),
            "release_manifest": str(args.release_manifest),
            "release_manifest_sha256": _sha256(args.release_manifest),
        },
        "host_gauges": _select(gauges, tuple(expected_gauges)),
        "runtime_defaults": {
            "expert_preload_seconds": gauges["expert_preload_seconds"],
            "ple_row_cache_entries": gauges["ple_row_cache_entries"],
        },
        "expert_exact_miss_path": _select(
            service,
            (
                "expert_requests",
                "expert_misses",
                "expert_hits",
                "expert_packed_host_hits",
                "expert_packed_host_misses",
                "expert_direct_slot_h2d_copies",
                "expert_direct_slot_h2d_bytes",
                "expert_owner_d2d_copies",
                "expert_owner_d2d_bytes",
                "expert_zero_d2d_resets",
                "expert_zero_d2d_skips",
            ),
        ),
        "ple_exact_lookup_path": _select(
            service,
            (
                "ple_lookup_calls",
                "ple_selected_rows",
                "ple_unique_rows",
                "ple_table_rows_read",
                "ple_table_bytes_read",
                "ple_device_h2d_bytes",
                "ple_device_logical_h2d_bytes",
                "ple_device_deferred_uploads",
                "ple_device_completion_syncs",
            ),
        ),
        "declared_host_work": declared,
        "prohibited_host_work": prohibited,
        "assertions": {
            "expected_host_gauges_match": gauge_match,
            "expert_exact_miss_and_direct_h2d_proven": expert_miss_proven,
            "ple_exact_mmap_lookup_and_h2d_proven": ple_path_proven,
            "prohibited_host_work_absent": forbidden_clear,
        },
    }
    if not all(evidence["assertions"].values()):
        raise RuntimeError(f"Host runtime assertions failed: {evidence['assertions']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

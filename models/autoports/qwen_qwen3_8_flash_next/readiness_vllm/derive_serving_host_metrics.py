#!/usr/bin/env python3
"""Derive exact benchmark-window host deltas from request-boundary markers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MARKER = "QWEN38_VLLM_METRICS "


def numeric_delta(end: dict[str, object], start: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value) - float(start.get(key, 0)) for key, value in end.items() if isinstance(value, (int, float))
    }


def window(label: str, shape: dict[str, object], start: dict[str, object], end: dict[str, object]):
    runtime_start = start["runtime_fallback"]
    runtime_end = end["runtime_fallback"]
    cumulative_runtime_counters = (
        "trace_replays",
        "model_only_trace_replays",
        "sampling_seed_host_copies",
    )
    return {
        "label": label,
        "workload": shape,
        "completed_requests_delta": int(end["completed_requests"]) - int(start["completed_requests"]),
        "host_service_delta": numeric_delta(end["host_service"], start["host_service"]),
        "decode_timing_delta": numeric_delta(end["decode_timing"], start["decode_timing"]),
        "host_gauges_start": start["host_gauges"],
        "host_gauges_end": end["host_gauges"],
        "attention_cache_start": start["attention_cache"],
        "attention_cache_end": end["attention_cache"],
        "runtime_fallback_end": runtime_end,
        # Token/position/page/readback counters live on a request state and
        # reset at admission; their end snapshot is recorded separately below.
        # Only model-lifetime counters have meaningful window deltas.
        "runtime_counter_delta": {
            name: float(runtime_end["counters"][name]) - float(runtime_start["counters"][name])
            for name in cumulative_runtime_counters
        },
        "host_sampling_compatibility_calls_delta": int(runtime_end["host_sampling_compatibility_calls"])
        - int(runtime_start["host_sampling_compatibility_calls"]),
        "request_counters_end": end["request_counters"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    markers = []
    for line in args.server_log.read_text(errors="replace").splitlines():
        if MARKER in line:
            markers.append(json.loads(line.split(MARKER, 1)[1]))
    if len(markers) < 34:
        raise SystemExit(f"need at least 34 final markers, found {len(markers)}")
    primary_start, primary_end = markers[-34], markers[-33]
    ci_start, ci_end = markers[-33], markers[-1]
    artifact = {
        "source": str(args.server_log),
        "marker_count": len(markers),
        "marker_selection": "last 34: primary start, 32 CI starts, post-CI sentinel",
        "primary_single_user": window(
            "headline single-user",
            {
                "prompt_tokens": 128,
                "output_tokens": 128,
                "requests": 1,
                "max_concurrency": 1,
                "temperature": 0.0,
            },
            primary_start,
            primary_end,
        ),
        "ci_serving_burst": window(
            "secondary CI serving burst",
            {
                "prompt_tokens": 100,
                "output_tokens": 100,
                "requests": 32,
                "max_concurrency": None,
                "temperature": 0.0,
            },
            ci_start,
            ci_end,
        ),
    }
    assert artifact["primary_single_user"]["completed_requests_delta"] == 1
    assert artifact["ci_serving_burst"]["completed_requests_delta"] == 32
    for section in (artifact["primary_single_user"], artifact["ci_serving_burst"]):
        runtime = section["runtime_fallback_end"]
        assert runtime["prohibited_host_work"] == {
            "activation_roundtrip": False,
            "expert_projection": False,
            "kv_or_recurrence": False,
            "optimized_sampling_or_argmax": False,
            "per_token_position_refresh": False,
            "ple_projection": False,
            "token_feedback_reconstruction": False,
            "unchanged_page_table_refresh": False,
        }
        assert runtime["ownership"]["kv_cache"] == "vllm"
        assert section["runtime_counter_delta"]["model_only_trace_replays"] == 0
        assert section["host_sampling_compatibility_calls_delta"] == 0
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

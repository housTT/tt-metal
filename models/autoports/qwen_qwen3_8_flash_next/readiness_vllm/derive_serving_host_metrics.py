#!/usr/bin/env python3
"""Derive exact benchmark-window host and virtual-state deltas.

The adapter emits one cumulative snapshot before every physical prefill call.
One call may admit multiple virtual requests, so marker *indices* do not identify
workload boundaries.  Cumulative logical-admission counts do.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

MARKER = "QWEN38_VLLM_METRICS "

EXPECTED_PROHIBITED_HOST_WORK = {
    "activation_roundtrip": False,
    "expert_projection": False,
    "kv_or_recurrence": False,
    "optimized_sampling_or_argmax": False,
    "per_token_position_refresh": False,
    "ple_projection": False,
    "token_feedback_reconstruction": False,
    "unchanged_page_table_refresh": False,
}
EXPECTED_OWNERSHIP = {
    "kv_cache": "vllm",
    "recurrence": "model",
    "page_table": "state with stable model buffer",
    "tokens_and_positions": "device feedback after request reset",
    "ple_store": "shared exact mmap table with request-isolated two-token history",
}
EXPECTED_DECLARED_HOST_WORK = {
    "model_load_exact_expert_prepack": True,
    "expert_route_id_read_and_exact_weight_dma": True,
    "ple_ngram_hash_row_lookup_and_dma": True,
    "caller_visible_compact_token_readback": True,
    "explicit_non_greedy_seed_control_h2d": False,
}
VIRTUAL_COUNTERS = (
    "assignments",
    "releases",
    "stale_rejections",
    "prefill_admissions_while_trace_live",
    "prefill_trace_invalidations",
    "sampling_trace_mode_switches",
)
VIRTUAL_BANK_COUNTERS = (
    "restores",
    "commits",
    "resets",
    "restore_logical_bytes",
    "commit_logical_bytes",
    "restore_submit_seconds",
    "commit_submit_seconds",
)
VIRTUAL_GAUGES = ("enabled", "physical_batch", "capacity", "active_slots", "resident_slot", "valid_slots")
VIRTUAL_BANK_GAUGES = (
    "enabled",
    "capacity",
    "closed",
    "allocated_logical_bytes",
    "logical_bytes_per_slot",
    "zero_template_logical_bytes",
)


def read_log_text(path: Path) -> str:
    """Read live text logs and archived ``.gz`` logs with the same API."""

    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as stream:
            return stream.read()
    return path.read_text(errors="replace")


def parse_markers(path: Path) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    text = read_log_text(path)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if MARKER in line:
            try:
                markers.append(json.loads(line.split(MARKER, 1)[1]))
            except json.JSONDecodeError:
                # A live EngineCore can be between writes while the lifecycle
                # poll reads the file.  Only the trailing partial line is safe
                # to retry; an earlier malformed marker is real corruption.
                if index != len(lines) - 1 or text.endswith(("\n", "\r")):
                    raise
    return markers


def numeric_delta(end: dict[str, object], start: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value) - float(start.get(key, 0)) for key, value in end.items() if isinstance(value, (int, float))
    }


def selected_numeric_delta(end: dict[str, object], start: dict[str, object], names: Iterable[str]) -> dict[str, float]:
    return {name: float(end.get(name, 0)) - float(start.get(name, 0)) for name in names}


def selected_gauges(source: dict[str, object], names: Iterable[str]) -> dict[str, object]:
    return {name: source.get(name) for name in names}


def runtime(marker: dict[str, Any]) -> dict[str, Any]:
    value = marker.get("runtime_fallback")
    if not isinstance(value, dict):
        raise AssertionError("selected request-boundary marker has no runtime fallback audit")
    return value


def virtual_slots(marker: dict[str, Any]) -> dict[str, Any]:
    counters = runtime(marker).get("counters")
    if not isinstance(counters, dict) or not isinstance(counters.get("virtual_decode_slots"), dict):
        raise AssertionError("selected marker has no virtual-decode-slot telemetry")
    return counters["virtual_decode_slots"]


def trace_replays(marker: dict[str, Any]) -> int | None:
    audit = marker.get("runtime_fallback")
    if not isinstance(audit, dict) or not isinstance(audit.get("counters"), dict):
        return None
    value = audit["counters"].get("trace_replays")
    return None if value is None else int(value)


def prefill_trace_invalidations(marker: dict[str, Any]) -> int | None:
    """Return trace invalidations that replace one replay with recapture.

    When prefill grows the TT program cache, the adapter releases the live
    decode trace before returning.  The first subsequent decode step captures
    its replacement and therefore is not counted by ``trace_replays``.  The
    sum of replay and invalidation deltas remains the canonical decode-step
    signature for a benchmark window.
    """

    try:
        value = virtual_slots(marker).get("prefill_trace_invalidations")
    except AssertionError:
        return None
    return None if value is None else int(value)


def traced_decode_steps(end: dict[str, Any], start: dict[str, Any]) -> int | None:
    start_replays = trace_replays(start)
    end_replays = trace_replays(end)
    start_invalidations = prefill_trace_invalidations(start)
    end_invalidations = prefill_trace_invalidations(end)
    if None in (start_replays, end_replays, start_invalidations, end_invalidations):
        return None
    replay_delta = end_replays - start_replays
    invalidation_delta = end_invalidations - start_invalidations
    if replay_delta < 0 or invalidation_delta < 0:
        return None
    return int(replay_delta + invalidation_delta)


def select_benchmark_windows(
    markers: list[dict[str, Any]],
    *,
    primary_requests: int,
    ci_requests: int,
    primary_trace_replays: int = 127,
    ci_trace_replays: int = 3168,
    primary_trace_captures: int = 0,
    ci_trace_captures: int = 0,
) -> tuple[int, int, int]:
    """Return primary-start, shared boundary, and CI-end marker indices.

    Logical request counts remain exact when a physical prefill groups two
    virtual rows and advances ``completed_requests`` by two at a time.  Replay
    Replay plus trace-invalidation deltas identify the canonical 128/128 and
    100/100/32 workloads even if prefill program-cache growth replaces one
    replay with a decode-trace capture, or lifecycle/qualitative probes append
    more markers after the benchmark.
    """

    if len(markers) < 3:
        raise ValueError(f"need at least three request-boundary markers, found {len(markers)}")
    by_count: dict[int, list[int]] = {}
    for index, marker in enumerate(markers):
        by_count.setdefault(int(marker["completed_requests"]), []).append(index)

    for ci_end_index in range(len(markers) - 1, 1, -1):
        ci_end_count = int(markers[ci_end_index]["completed_requests"])
        for ci_start_index in reversed(by_count.get(ci_end_count - ci_requests, ())):
            if ci_start_index >= ci_end_index:
                continue
            if (
                traced_decode_steps(markers[ci_end_index], markers[ci_start_index])
                != ci_trace_replays - ci_trace_captures
            ):
                continue
            primary_start_count = int(markers[ci_start_index]["completed_requests"]) - primary_requests
            for primary_start_index in reversed(by_count.get(primary_start_count, ())):
                if primary_start_index >= ci_start_index:
                    continue
                if (
                    traced_decode_steps(markers[ci_start_index], markers[primary_start_index])
                    == primary_trace_replays - primary_trace_captures
                ):
                    return primary_start_index, ci_start_index, ci_end_index
    raise ValueError(
        "no markers delimit the exact primary/CI logical-request and traced-decode signatures: "
        f"primary={primary_requests} requests/{primary_trace_replays} steps "
        f"({primary_trace_captures} initial captures), CI={ci_requests} requests/{ci_trace_replays} steps "
        f"({ci_trace_captures} initial captures)"
    )


def validate_runtime_contract(
    start: dict[str, Any],
    end: dict[str, Any],
    *,
    physical_batch: int,
    virtual_slot_capacity: int,
) -> list[str]:
    """Assert the exact optimized-serving ownership and virtual-bank contract."""

    checks: list[str] = []
    for boundary, marker in (("start", start), ("end", end)):
        audit = runtime(marker)
        assert marker["attention_cache_owner"] == "vllm"
        assert audit["prohibited_host_work"] == EXPECTED_PROHIBITED_HOST_WORK
        declared = audit["declared_host_work"]
        for name, expected in EXPECTED_DECLARED_HOST_WORK.items():
            if name == "explicit_non_greedy_seed_control_h2d":
                # The start marker is emitted before the measured prefill
                # installs its sampling parameters, so a warmed greedy window
                # may inherit ``True`` from the preceding sampled qualitative
                # request. The measured window separately proves zero seed
                # copies, and its end boundary must advertise greedy mode.
                assert isinstance(declared[name], bool)
                if boundary == "end":
                    assert declared[name] is False
            else:
                assert declared[name] == expected
        ownership = audit["ownership"]
        for name, expected in EXPECTED_OWNERSHIP.items():
            assert ownership[name] == expected
        assert ownership["expert_store"] == (
            "per-layer exact mmap source, model-load packed-host preload, and fixed TT slots"
        )

        slots = virtual_slots(marker)
        bank = slots["bank"]
        assert slots["enabled"] is True
        assert int(slots["physical_batch"]) == physical_batch
        assert int(slots["capacity"]) == virtual_slot_capacity
        assert bank["enabled"] is True
        assert int(bank["capacity"]) == virtual_slot_capacity
        assert bank["closed"] is False
        assert int(bank["logical_bytes_per_slot"]) > 0
        assert int(bank["allocated_logical_bytes"]) >= int(bank["logical_bytes_per_slot"]) * virtual_slot_capacity
        checks.append(
            f"{boundary}: vLLM KV ownership and model-owned virtual bank {physical_batch}/{virtual_slot_capacity}"
        )

    assert start["attention_cache"] == end["attention_cache"]
    cache = end["attention_cache"]
    assert int(cache["vllm_adoptions"]) == 1
    assert int(cache["standalone_allocations"]) == int(cache["standalone_tensors_released"])
    checks.append("attention-cache lifecycle stayed constant with one vLLM adoption and no model-owned residue")
    return checks


def window(
    label: str,
    shape: dict[str, object],
    start: dict[str, Any],
    end: dict[str, Any],
    *,
    physical_batch: int,
    virtual_slot_capacity: int,
) -> dict[str, object]:
    runtime_start = runtime(start)
    runtime_end = runtime(end)
    cumulative_runtime_counters = (
        "trace_replays",
        "model_only_trace_replays",
        "sampling_seed_host_copies",
        "async_feedback_host_reuses",
        "async_feedback_device_fallbacks",
    )
    slots_start = virtual_slots(start)
    slots_end = virtual_slots(end)
    bank_start = slots_start["bank"]
    bank_end = slots_end["bank"]
    host_service_delta = numeric_delta(end["host_service"], start["host_service"])
    decode_timing_delta = numeric_delta(end["decode_timing"], start["decode_timing"])
    artifact = {
        "label": label,
        "workload": shape,
        "completed_requests_delta": int(end["completed_requests"]) - int(start["completed_requests"]),
        "host_service_delta": host_service_delta,
        "decode_timing_delta": decode_timing_delta,
        "host_gauges_start": start["host_gauges"],
        "host_gauges_end": end["host_gauges"],
        "attention_cache_owner_start": start["attention_cache_owner"],
        "attention_cache_owner_end": end["attention_cache_owner"],
        "attention_cache_start": start["attention_cache"],
        "attention_cache_end": end["attention_cache"],
        "virtual_slot_delta": selected_numeric_delta(slots_end, slots_start, VIRTUAL_COUNTERS),
        "virtual_slot_gauges_start": selected_gauges(slots_start, VIRTUAL_GAUGES),
        "virtual_slot_gauges_end": selected_gauges(slots_end, VIRTUAL_GAUGES),
        "virtual_bank_delta": selected_numeric_delta(bank_end, bank_start, VIRTUAL_BANK_COUNTERS),
        "virtual_bank_gauges_start": selected_gauges(bank_start, VIRTUAL_BANK_GAUGES),
        "virtual_bank_gauges_end": selected_gauges(bank_end, VIRTUAL_BANK_GAUGES),
        "runtime_fallback_start": runtime_start,
        "runtime_fallback_end": runtime_end,
        # Token/position/page/readback counters are request-state gauges that
        # reset at admission.  Preserve the final boundary; do not mislabel it
        # as a model-lifetime delta.
        "runtime_counter_delta": selected_numeric_delta(
            runtime_end["counters"], runtime_start["counters"], cumulative_runtime_counters
        ),
        "host_sampling_compatibility_calls_delta": int(runtime_end["host_sampling_compatibility_calls"])
        - int(runtime_start["host_sampling_compatibility_calls"]),
        "request_counters_end": end["request_counters"],
        "runtime_ownership_assertions": validate_runtime_contract(
            start,
            end,
            physical_batch=physical_batch,
            virtual_slot_capacity=virtual_slot_capacity,
        ),
    }
    for family, deltas in (("host_service", host_service_delta), ("decode_timing", decode_timing_delta)):
        negative = {name: value for name, value in deltas.items() if value < 0}
        assert not negative, f"{family} counters decreased inside {label}: {negative}"
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-requests", type=int, default=1)
    parser.add_argument("--ci-requests", type=int, default=32)
    parser.add_argument("--primary-trace-replays", type=int, default=127)
    parser.add_argument("--ci-trace-replays", type=int, default=3168)
    parser.add_argument("--primary-trace-captures", type=int, default=0)
    parser.add_argument("--ci-trace-captures", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--physical-batch", type=int, default=1)
    parser.add_argument("--virtual-slot-capacity", type=int, default=2)
    args = parser.parse_args()
    if args.max_num_seqs != args.virtual_slot_capacity:
        raise SystemExit("served max-num-seqs must equal the declared virtual-slot capacity")
    if not 0 <= args.primary_trace_captures <= args.primary_trace_replays:
        raise SystemExit("primary trace captures must be between zero and the expected primary decode steps")
    if not 0 <= args.ci_trace_captures <= args.ci_trace_replays:
        raise SystemExit("CI trace captures must be between zero and the expected CI decode steps")

    markers = parse_markers(args.server_log)
    try:
        primary_start_index, ci_start_index, ci_end_index = select_benchmark_windows(
            markers,
            primary_requests=args.primary_requests,
            ci_requests=args.ci_requests,
            primary_trace_replays=args.primary_trace_replays,
            ci_trace_replays=args.ci_trace_replays,
            primary_trace_captures=args.primary_trace_captures,
            ci_trace_captures=args.ci_trace_captures,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    primary_start = markers[primary_start_index]
    ci_start = markers[ci_start_index]
    ci_end = markers[ci_end_index]
    artifact = {
        "schema_version": 2,
        "source": str(args.server_log),
        "marker_count": len(markers),
        "serving_capacity": {
            "max_num_seqs": args.max_num_seqs,
            "physical_decode_batch": args.physical_batch,
            "virtual_slot_capacity": args.virtual_slot_capacity,
        },
        "marker_selection": {
            "method": "latest exact logical-request and canonical replay-plus-invalidation-plus-declared-initial-capture signatures",
            "grouped_virtual_prefills_supported": True,
            "primary_start_index": primary_start_index,
            "primary_end_ci_start_index": ci_start_index,
            "ci_end_sentinel_index": ci_end_index,
            "primary_start_completed_requests": int(primary_start["completed_requests"]),
            "ci_start_completed_requests": int(ci_start["completed_requests"]),
            "ci_end_completed_requests": int(ci_end["completed_requests"]),
            "primary_expected_decode_steps": args.primary_trace_replays,
            "primary_initial_trace_captures": args.primary_trace_captures,
            "primary_trace_replays": int(trace_replays(ci_start)) - int(trace_replays(primary_start)),
            "primary_prefill_trace_invalidations": int(prefill_trace_invalidations(ci_start))
            - int(prefill_trace_invalidations(primary_start)),
            "ci_expected_decode_steps": args.ci_trace_replays,
            "ci_initial_trace_captures": args.ci_trace_captures,
            "ci_trace_replays": int(trace_replays(ci_end)) - int(trace_replays(ci_start)),
            "ci_prefill_trace_invalidations": int(prefill_trace_invalidations(ci_end))
            - int(prefill_trace_invalidations(ci_start)),
        },
        "primary_single_user": window(
            "headline single-user",
            {
                "prompt_tokens": 128,
                "output_tokens": 128,
                "requests": args.primary_requests,
                "max_concurrency": 1,
                "server_max_num_seqs": args.max_num_seqs,
                "temperature": 0.0,
            },
            primary_start,
            ci_start,
            physical_batch=args.physical_batch,
            virtual_slot_capacity=args.virtual_slot_capacity,
        ),
        "ci_serving_burst": window(
            "secondary CI serving burst",
            {
                "prompt_tokens": 100,
                "output_tokens": 100,
                "requests": args.ci_requests,
                "max_concurrency": None,
                "server_max_num_seqs": args.max_num_seqs,
                "temperature": 0.0,
            },
            ci_start,
            ci_end,
            physical_batch=args.physical_batch,
            virtual_slot_capacity=args.virtual_slot_capacity,
        ),
    }
    primary = artifact["primary_single_user"]
    burst = artifact["ci_serving_burst"]
    assert primary["completed_requests_delta"] == args.primary_requests
    assert burst["completed_requests_delta"] == args.ci_requests
    assert (
        primary["runtime_counter_delta"]["trace_replays"]
        + primary["virtual_slot_delta"]["prefill_trace_invalidations"]
        + args.primary_trace_captures
        == args.primary_trace_replays
    )
    assert (
        burst["runtime_counter_delta"]["trace_replays"]
        + burst["virtual_slot_delta"]["prefill_trace_invalidations"]
        + args.ci_trace_captures
        == args.ci_trace_replays
    )
    for section in (primary, burst):
        completed_requests = int(section["completed_requests_delta"])
        expected_feedback_steps = completed_requests * (int(section["workload"]["output_tokens"]) - 1)
        assert section["runtime_counter_delta"]["trace_replays"] > 0
        assert section["decode_timing_delta"]["replay_tokens"] == section["runtime_counter_delta"]["trace_replays"]
        assert section["runtime_counter_delta"]["model_only_trace_replays"] == 0
        assert section["runtime_counter_delta"]["sampling_seed_host_copies"] == 0
        assert section["host_sampling_compatibility_calls_delta"] == 0
        assert section["runtime_counter_delta"]["async_feedback_device_fallbacks"] == completed_requests
        assert (
            section["runtime_counter_delta"]["async_feedback_host_reuses"]
            == expected_feedback_steps - completed_requests
        )
        assert (
            section["runtime_counter_delta"]["async_feedback_host_reuses"]
            + section["runtime_counter_delta"]["async_feedback_device_fallbacks"]
            == expected_feedback_steps
        )
        assert section["virtual_slot_delta"]["assignments"] == section["completed_requests_delta"]
        assert section["virtual_slot_delta"]["releases"] == section["completed_requests_delta"]
        assert section["virtual_slot_delta"]["stale_rejections"] == 0
        assert section["virtual_slot_delta"]["prefill_admissions_while_trace_live"] > 0
        assert section["virtual_slot_delta"]["sampling_trace_mode_switches"] == 0
        assert section["virtual_slot_gauges_start"]["active_slots"] == 0
        assert section["virtual_slot_gauges_end"]["active_slots"] == 0
        assert section["virtual_slot_gauges_start"]["valid_slots"] == 0
        assert section["virtual_slot_gauges_end"]["valid_slots"] == 0
        assert section["host_service_delta"]["expert_requests"] > 0
        assert section["host_service_delta"]["expert_hits"] > 0
        assert section["host_service_delta"]["expert_misses"] > 0
        assert section["host_service_delta"]["expert_h2d_bytes"] > 0
        assert (
            section["host_service_delta"]["expert_direct_slot_h2d_bytes"]
            == section["host_service_delta"]["expert_h2d_bytes"]
        )
        assert section["host_service_delta"]["expert_owner_d2d_bytes"] == 0
        assert section["host_service_delta"]["expert_dma_completion_syncs"] == 0
        assert section["host_service_delta"]["ple_lookup_calls"] > 0
        assert section["host_service_delta"]["ple_selected_rows"] > 0
        assert section["host_service_delta"]["ple_device_h2d_bytes"] > 0
        assert section["host_service_delta"]["ple_device_completion_syncs"] == 0
    # The B1 path stays resident and must not pay virtual-bank copies.  The CI
    # burst actively overlaps rows on physical B1, so it must prove both sides
    # of the bank lifecycle rather than merely advertise capacity two.
    for name in ("commits", "restores", "commit_logical_bytes", "restore_logical_bytes"):
        assert primary["virtual_bank_delta"][name] == 0
        assert burst["virtual_bank_delta"][name] > 0
    assert burst["virtual_bank_delta"]["resets"] > 0

    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

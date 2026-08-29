#!/usr/bin/env python3
"""Exercise physical-B1/virtual-B2 overlap, cancellation, and isolation."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
from derive_serving_host_metrics import (
    EXPECTED_PROHIBITED_HOST_WORK,
    VIRTUAL_BANK_COUNTERS,
    VIRTUAL_COUNTERS,
    parse_markers,
    read_log_text,
    selected_numeric_delta,
    validate_runtime_contract,
    virtual_slots,
)


def abort_marker_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(re.findall(r"(?:aborted|aborting|abort) request", read_log_text(path), re.IGNORECASE))


def marker_with_count(
    markers: list[dict[str, Any]], completed_requests: int, *, start_index: int = 0
) -> dict[str, Any]:
    return next(
        marker
        for marker in markers[start_index:]
        if int(marker["completed_requests"]) == completed_requests and marker.get("runtime_fallback") is not None
    )


def lifecycle_metric_evidence(
    markers: list[dict[str, Any]],
    *,
    first_new_index: int,
    baseline_start_count: int,
    physical_batch: int,
    virtual_slot_capacity: int,
) -> dict[str, Any]:
    """Extract the four-request measured interval from its sentinel marker.

    The baseline request increments the logical admission counter once.  The
    measured interval then contains survivor, cancelled peer, and two
    deterministic followups.  The final one-token sentinel emits its snapshot
    after all four have released their state.
    """

    start_count = baseline_start_count + 1
    end_count = start_count + 4
    start = marker_with_count(markers, start_count, start_index=first_new_index)
    end = marker_with_count(markers, end_count, start_index=first_new_index)
    slots_start = virtual_slots(start)
    slots_end = virtual_slots(end)
    bank_start = slots_start["bank"]
    bank_end = slots_end["bank"]
    runtime_start = start["runtime_fallback"]
    runtime_end = end["runtime_fallback"]
    slot_delta = selected_numeric_delta(slots_end, slots_start, VIRTUAL_COUNTERS)
    bank_delta = selected_numeric_delta(bank_end, bank_start, VIRTUAL_BANK_COUNTERS)
    runtime_checks = validate_runtime_contract(
        start,
        end,
        physical_batch=physical_batch,
        virtual_slot_capacity=virtual_slot_capacity,
    )
    host_delta = selected_numeric_delta(
        end["host_service"],
        start["host_service"],
        end["host_service"].keys(),
    )
    evidence = {
        "marker_start_completed_requests": start_count,
        "marker_end_completed_requests": end_count,
        "measured_logical_requests": end_count - start_count,
        "virtual_slot_start": slots_start,
        "virtual_slot_end": slots_end,
        "virtual_slot_delta": slot_delta,
        "virtual_bank_delta": bank_delta,
        "runtime_counter_delta": selected_numeric_delta(
            runtime_end["counters"],
            runtime_start["counters"],
            ("trace_replays", "model_only_trace_replays", "sampling_seed_host_copies"),
        ),
        "host_service_delta": host_delta,
        "host_gauges_start": start["host_gauges"],
        "host_gauges_end": end["host_gauges"],
        "runtime_ownership_assertions": runtime_checks,
        "prohibited_host_work_end": runtime_end["prohibited_host_work"],
        "host_sampling_compatibility_calls_delta": int(runtime_end["host_sampling_compatibility_calls"])
        - int(runtime_start["host_sampling_compatibility_calls"]),
    }
    evidence["active_overlap_bank_evidence"] = (
        slot_delta["assignments"] == 4
        and slot_delta["releases"] == 4
        and slot_delta["stale_rejections"] == 0
        and bank_delta["commits"] > 0
        and bank_delta["restores"] > 0
        and bank_delta["commit_logical_bytes"] > 0
        and bank_delta["restore_logical_bytes"] > 0
    )
    evidence["finish_cancel_release_evidence"] = (
        slot_delta["releases"] == 4
        and slots_start["active_slots"] == 0
        and slots_end["active_slots"] == 0
        and slots_start["valid_slots"] == 0
        and slots_end["valid_slots"] == 0
        and start["host_gauges"]["ple_history_entries"] == 0
        and end["host_gauges"]["ple_history_entries"] == 0
    )
    evidence["runtime_contract_evidence"] = (
        evidence["prohibited_host_work_end"] == EXPECTED_PROHIBITED_HOST_WORK
        and evidence["host_sampling_compatibility_calls_delta"] == 0
        and evidence["runtime_counter_delta"]["trace_replays"] > 0
        and evidence["runtime_counter_delta"]["model_only_trace_replays"] == 0
        and evidence["runtime_counter_delta"]["sampling_seed_host_copies"] == 0
        and host_delta["expert_requests"] > 0
        and host_delta["expert_h2d_bytes"] > 0
        and host_delta["ple_lookup_calls"] > 0
        and host_delta["ple_device_h2d_bytes"] > 0
    )
    return evidence


async def wait_for_marker_count(path: Path, minimum: int, timeout_seconds: float = 10.0) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        markers = parse_markers(path)
        if len(markers) >= minimum:
            return markers
        if time.monotonic() >= deadline:
            raise RuntimeError(f"server log did not publish metric marker {minimum} within {timeout_seconds}s")
        await asyncio.sleep(0.1)


async def post_completion(client: httpx.AsyncClient, endpoint: str, payload: dict[str, object]) -> dict[str, object]:
    response = await client.post(endpoint, json=payload)
    body = response.json()
    return {
        "http_status": response.status_code,
        "text": body["choices"][0]["text"],
        "finish_reason": body["choices"][0]["finish_reason"],
        "usage": body["usage"],
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8018")
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--physical-batch", type=int, default=1)
    parser.add_argument("--virtual-slot-capacity", type=int, default=2)
    parser.add_argument("--cancel-after-first-token-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_num_seqs != args.virtual_slot_capacity:
        raise SystemExit("served max-num-seqs must equal the virtual-slot capacity")
    if args.physical_batch != 1 or args.virtual_slot_capacity < 2:
        raise SystemExit("this lifecycle probe requires physical batch 1 and at least two virtual slots")

    endpoint = f"{args.server_url.rstrip('/')}/v1/completions"
    timeline: list[dict[str, object]] = []
    abort_markers_before = abort_marker_count(args.server_log)
    initial_marker_count = len(parse_markers(args.server_log))

    baseline_payload = {
        "model": args.model,
        "prompt": "Virtual lifecycle baseline:",
        "max_tokens": 1,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        baseline = await post_completion(client, endpoint, baseline_payload)
    baseline_markers = await wait_for_marker_count(args.server_log, initial_marker_count + 1)
    baseline_marker = baseline_markers[initial_marker_count]
    baseline_start_count = int(baseline_marker["completed_requests"])

    first_token_events = {"survivor": asyncio.Event(), "cancelled_active_peer": asyncio.Event()}

    async def stream(label: str, prompt: str, max_tokens: int) -> dict[str, object]:
        started = time.time()
        record: dict[str, object] = {
            "label": label,
            "started_unix": started,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        text_parts: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                async with client.stream(
                    "POST",
                    endpoint,
                    json={
                        "model": args.model,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        "ignore_eos": True,
                        "stream": True,
                    },
                ) as response:
                    record["http_status"] = response.status_code
                    chunks = 0
                    token_events = 0
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        chunks += 1
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        event = json.loads(line.removeprefix("data: "))
                        delta = event.get("choices", [{}])[0].get("text", "")
                        text_parts.append(delta)
                        token_events += 1
                        if "first_token_unix" not in record:
                            record["first_token_unix"] = time.time()
                            first_token_events[label].set()
                    record["stream_chunks"] = chunks
                    record["token_events"] = token_events
                    record["text"] = "".join(text_parts)
                    record["completed"] = True
        except asyncio.CancelledError:
            record["cancelled"] = True
            raise
        finally:
            record["ended_unix"] = time.time()
            record["elapsed_seconds"] = float(record["ended_unix"]) - started
            record.setdefault("text", "".join(text_parts))
            timeline.append(record)
        return record

    survivor = asyncio.create_task(stream("survivor", "Write a numbered list of one hundred colors:", 128))
    cancelled = asyncio.create_task(
        stream("cancelled_active_peer", "Explain active request cancellation in detail:", 128)
    )
    await asyncio.wait_for(
        asyncio.gather(*(event.wait() for event in first_token_events.values())),
        timeout=180.0,
    )
    await asyncio.sleep(args.cancel_after_first_token_seconds)
    cancel_requested_unix = time.time()
    cancel_request_accepted = cancelled.cancel()
    try:
        await cancelled
    except asyncio.CancelledError:
        pass
    survivor_result = await survivor

    deterministic_payload = {
        "model": args.model,
        "prompt": "Isolation sentinel: the capital of France is",
        "max_tokens": 12,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    followups = []
    async with httpx.AsyncClient(timeout=180.0) as client:
        for _ in range(2):
            followups.append(await post_completion(client, endpoint, deterministic_payload))
        metric_sentinel = await post_completion(
            client,
            endpoint,
            {
                "model": args.model,
                "prompt": "Lifecycle metrics sentinel:",
                "max_tokens": 1,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        )

    all_markers = await wait_for_marker_count(args.server_log, len(baseline_markers) + 4)
    metrics = lifecycle_metric_evidence(
        all_markers,
        first_new_index=initial_marker_count + 1,
        baseline_start_count=baseline_start_count,
        physical_batch=args.physical_batch,
        virtual_slot_capacity=args.virtual_slot_capacity,
    )
    cancelled_records = [row for row in timeline if row["label"] == "cancelled_active_peer"]
    survivor_record = next(row for row in timeline if row["label"] == "survivor")
    cancelled_record = cancelled_records[0] if cancelled_records else {}
    abort_markers_after = abort_marker_count(args.server_log)
    active_stream_overlap = (
        "first_token_unix" in survivor_record
        and "first_token_unix" in cancelled_record
        and float(survivor_record["first_token_unix"]) < float(cancelled_record["ended_unix"])
        and float(cancelled_record["first_token_unix"]) < float(survivor_record["ended_unix"])
    )
    artifact = {
        "schema_version": 2,
        "server_url": args.server_url,
        "model": args.model,
        "serving_capacity": {
            "max_num_seqs": args.max_num_seqs,
            "physical_decode_batch": args.physical_batch,
            "virtual_slot_capacity": args.virtual_slot_capacity,
        },
        "workload": {
            "baseline": {"requests": 1, "max_tokens": 1, "temperature": 0.0},
            "survivor": {
                "prompt_tokens": "variable",
                "max_tokens": 128,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "cancelled_active_peer": {
                "prompt_tokens": "variable",
                "max_tokens": 128,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "cancel_after_both_first_tokens_seconds": args.cancel_after_first_token_seconds,
            "followup": {
                "requests": 2,
                "max_tokens": 12,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "metrics_sentinel": {"requests": 1, "max_tokens": 1, "temperature": 0.0},
        },
        "baseline": baseline,
        "timeline": sorted(timeline, key=lambda row: float(row["started_unix"])),
        "both_requests_streamed_before_cancel": all(
            "first_token_unix" in row for row in (survivor_record, cancelled_record)
        ),
        "active_stream_overlap": active_stream_overlap,
        "cancel_requested_unix": cancel_requested_unix,
        "cancel_request_accepted": cancel_request_accepted,
        "survivor_completed": bool(survivor_result.get("completed")),
        "active_peer_client_cancelled": bool(cancelled_record.get("cancelled")),
        "server_log": str(args.server_log),
        "server_abort_markers_delta": abort_markers_after - abort_markers_before,
        "server_log_abort_observed": abort_markers_after > abort_markers_before,
        "release_path_semantics": (
            "the live survivor and client-aborted peer use the finished-request release path; "
            "the plugin carries preemption through the same generation-tagged lifecycle hook"
        ),
        "virtual_lifecycle_metrics": metrics,
        "followups": followups,
        "followups_http_200": all(row["http_status"] == 200 for row in followups),
        "followups_nonempty": all(bool(str(row["text"]).strip()) for row in followups),
        "followups_identical": followups[0]["text"] == followups[1]["text"],
        "metrics_sentinel": metric_sentinel,
    }
    artifact["verdict"] = (
        "pass"
        if artifact["survivor_completed"]
        and artifact["active_peer_client_cancelled"]
        and artifact["cancel_request_accepted"]
        and artifact["both_requests_streamed_before_cancel"]
        and artifact["active_stream_overlap"]
        and artifact["virtual_lifecycle_metrics"]["active_overlap_bank_evidence"]
        and artifact["virtual_lifecycle_metrics"]["finish_cancel_release_evidence"]
        and artifact["virtual_lifecycle_metrics"]["runtime_contract_evidence"]
        and artifact["followups_http_200"]
        and artifact["followups_nonempty"]
        and artifact["followups_identical"]
        and artifact["metrics_sentinel"]["http_status"] == 200
        else "fail"
    )
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    if artifact["verdict"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

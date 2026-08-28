#!/usr/bin/env python3
"""Exercise queued cancellation and request-state isolation on a live server."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx


def abort_marker_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return len(re.findall(r"(?:aborted|aborting|abort) request", path.read_text(errors="replace"), re.IGNORECASE))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8018")
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path)
    args = parser.parse_args()
    endpoint = f"{args.server_url.rstrip('/')}/v1/completions"
    timeline: list[dict[str, object]] = []
    abort_markers_before = abort_marker_count(args.server_log)

    async def stream(label: str, prompt: str, max_tokens: int) -> dict[str, object]:
        started = time.time()
        record: dict[str, object] = {
            "label": label,
            "started_unix": started,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
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
                    async for line in response.aiter_lines():
                        if line:
                            chunks += 1
                    record["stream_chunks"] = chunks
                    record["completed"] = True
        except asyncio.CancelledError:
            record["cancelled"] = True
            raise
        finally:
            record["ended_unix"] = time.time()
            record["elapsed_seconds"] = float(record["ended_unix"]) - started
            timeline.append(record)
        return record

    survivor = asyncio.create_task(stream("survivor", "Write a numbered list of one hundred colors:", 128))
    await asyncio.sleep(0.75)
    cancelled = asyncio.create_task(
        stream("cancelled_while_queued", "Explain queued request cancellation in detail:", 128)
    )
    await asyncio.sleep(1.0)
    cancelled.cancel()
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
    }
    followups = []
    async with httpx.AsyncClient(timeout=180.0) as client:
        for _ in range(2):
            response = await client.post(endpoint, json=deterministic_payload)
            body = response.json()
            followups.append(
                {
                    "http_status": response.status_code,
                    "text": body["choices"][0]["text"],
                    "finish_reason": body["choices"][0]["finish_reason"],
                    "usage": body["usage"],
                }
            )

    cancelled_records = [row for row in timeline if row["label"] == "cancelled_while_queued"]
    abort_markers_after = abort_marker_count(args.server_log)
    artifact = {
        "server_url": args.server_url,
        "model": args.model,
        "physical_max_num_seqs": 1,
        "workload": {
            "survivor": {
                "prompt_tokens": "variable",
                "max_tokens": 128,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "cancelled": {
                "prompt_tokens": "variable",
                "max_tokens": 128,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "overlap_delay_seconds": 0.75,
            "cancel_delay_seconds": 1.0,
            "followup": {"requests": 2, "max_tokens": 12, "temperature": 0.0},
        },
        "timeline": sorted(timeline, key=lambda row: float(row["started_unix"])),
        "survivor_completed": bool(survivor_result.get("completed")),
        "queued_client_cancelled": bool(cancelled_records and cancelled_records[0].get("cancelled")),
        "server_log": None if args.server_log is None else str(args.server_log),
        "server_abort_markers_delta": abort_markers_after - abort_markers_before,
        "server_log_abort_observed": abort_markers_after > abort_markers_before,
        "followups": followups,
        "followups_http_200": all(row["http_status"] == 200 for row in followups),
        "followups_nonempty": all(bool(str(row["text"]).strip()) for row in followups),
        "followups_identical": followups[0]["text"] == followups[1]["text"],
    }
    artifact["verdict"] = (
        "pass"
        if artifact["survivor_completed"]
        and artifact["queued_client_cancelled"]
        and artifact["followups_http_200"]
        and artifact["followups_nonempty"]
        and artifact["followups_identical"]
        else "fail"
    )
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    if artifact["verdict"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exercise every GPT-OSS 120B serving trace/lifecycle transition.

Run this once with normal allocation warnings and once with
TT_METAL_TRACE_ALLOC_TRACKING=1 plus TT_METAL_TRACE_ALLOC_TRACEBACKS=1. The
companion server logs provide the allocation-safety evidence; this artifact
records the exact request sequence and HTTP outcomes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

MODEL_ID = "openai/gpt-oss-120b"


async def _request(
    client: httpx.AsyncClient,
    server_url: str,
    *,
    phase: str,
    index: int,
    output_tokens: int,
    greedy: bool,
    host_sampling: bool,
) -> dict:
    payload = {
        "model": MODEL_ID,
        "prompt": f"Trace allocation probe {phase} request {index}:",
        "max_tokens": output_tokens,
        "temperature": 0.0 if greedy else 0.7,
        "top_p": 1.0 if greedy else 0.9,
        "ignore_eos": True,
    }
    if host_sampling:
        # min_p is deliberately unsupported by the TT device sampler. It must
        # select the explicit optional host-compatibility route.
        payload["min_p"] = 0.1
    response = await client.post(f"{server_url.rstrip('/')}/v1/completions", json=payload)
    row = {
        "i": index,
        "status": response.status_code,
    }
    if response.status_code == 200:
        body = response.json()
        row.update(
            text=body["choices"][0]["text"],
            completion_tokens=body.get("usage", {}).get("completion_tokens"),
        )
    else:
        row["body"] = response.text
    return row


async def _phase(
    client: httpx.AsyncClient,
    server_url: str,
    *,
    name: str,
    count: int,
    output_tokens: int,
    greedy: bool = False,
    host_sampling: bool = False,
) -> dict:
    rows = await asyncio.gather(
        *(
            _request(
                client,
                server_url,
                phase=name,
                index=index,
                output_tokens=output_tokens,
                greedy=greedy,
                host_sampling=host_sampling,
            )
            for index in range(count)
        )
    )
    return {
        "name": name,
        "sampling_route": "host_min_p" if host_sampling else "device_greedy" if greedy else "device_sampled",
        "count": count,
        "output_tokens_per_request": output_tokens,
        "http200": sum(row["status"] == 200 for row in rows),
        "rows": rows,
    }


async def _run(server_url: str) -> list[dict]:
    timeout = httpx.Timeout(600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        phases = []
        phases.append(await _phase(client, server_url, name="device_sampled_b1_before", count=1, output_tokens=4))
        phases.append(await _phase(client, server_url, name="device_greedy_b1", count=1, output_tokens=4, greedy=True))
        phases.append(await _phase(client, server_url, name="device_sampled_b1_after_greedy", count=1, output_tokens=4))
        phases.append(await _phase(client, server_url, name="device_sampled_b32", count=32, output_tokens=4))
        phases.append(
            await _phase(client, server_url, name="device_greedy_b32", count=32, output_tokens=4, greedy=True)
        )
        phases.append(
            await _phase(client, server_url, name="device_sampled_b32_after_greedy", count=32, output_tokens=4)
        )
        phases.append(await _phase(client, server_url, name="device_sampled_b1_after_b32", count=1, output_tokens=4))
        phases.append(
            await _phase(
                client,
                server_url,
                name="host_min_p_b10",
                count=10,
                output_tokens=10,
                host_sampling=True,
            )
        )
        phases.append(
            await _phase(client, server_url, name="device_greedy_b1_after_host", count=1, output_tokens=4, greedy=True)
        )
        phases.append(
            await _phase(
                client,
                server_url,
                name="device_sampled_b1_after_host_and_greedy",
                count=1,
                output_tokens=4,
            )
        )
        return phases


def main() -> None:
    script = Path(__file__).resolve()
    model_dir = script.parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--output",
        type=Path,
        default=model_dir / "readiness_vllm" / "trace_allocation_autofix" / "probe.json",
    )
    args = parser.parse_args()

    phases = asyncio.run(_run(args.server_url))
    expected_requests = sum(phase["count"] for phase in phases)
    http200 = sum(phase["http200"] for phase in phases)
    artifact = {
        "status": "pass" if http200 == expected_requests else "fail",
        "model_id": MODEL_ID,
        "server_url": args.server_url,
        "expected_requests": expected_requests,
        "http200": http200,
        "phases": phases,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if artifact["status"] != "pass":
        raise RuntimeError(f"{http200}/{expected_requests} allocation-probe requests passed; wrote {args.output}")
    print(f"PASS: {http200}/{expected_requests} allocation-probe requests returned HTTP 200; wrote {args.output}")


if __name__ == "__main__":
    main()

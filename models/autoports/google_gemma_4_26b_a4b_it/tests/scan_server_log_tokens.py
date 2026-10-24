# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Scan a vLLM server log for out-of-vocabulary sampled token ids.

Counts the detokenizer's ``Encountered invalid token id`` lines and the adapter's
``GEMMA4_TRACE_EVENT invalid_sampled_token`` events, optionally from a start line, and writes a JSON
record with the counts, the offending ids and the rows they came from. It also counts how the window
sampled: device sampled and greedy trace captures versus ``host_sampling_prefill`` and
``host_sampling_decode`` releases, so a sampling verdict can show whether it exercised the device
sampler.

Usage:
    python -m models.autoports.google_gemma_4_26b_a4b_it.tests.scan_server_log_tokens \
        --server-log <server.log> --output <json> [--start-line N]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EVENT_PREFIX = "GEMMA4_TRACE_EVENT "
INVALID_RE = re.compile(r"Encountered invalid token id: (-?\d+)")


def scan(path: Path, start_line: int = 0) -> dict:
    detokenizer_ids: list[int] = []
    events: list[dict] = []
    modes = {
        "device_sampled_captures": 0,
        "device_greedy_captures": 0,
        "host_sampling_prefill_releases": 0,
        "host_sampling_decode_releases": 0,
    }
    total = 0
    with path.open("r", errors="replace") as handle:
        for index, line in enumerate(handle):
            total = index + 1
            if index < start_line:
                continue
            match = INVALID_RE.search(line)
            if match:
                detokenizer_ids.append(int(match.group(1)))
            if EVENT_PREFIX in line:
                try:
                    event = json.loads(line.split(EVENT_PREFIX, 1)[1].strip())
                except json.JSONDecodeError:
                    continue
                name = event.get("event")
                if name == "invalid_sampled_token":
                    events.append({k: event.get(k) for k in ("rows", "values", "execution_batch", "device_max_top_k")})
                elif name == "capture" and event.get("mode") == "device":
                    modes["device_greedy_captures" if event.get("greedy") else "device_sampled_captures"] += 1
                elif name == "release" and event.get("reason") in ("host_sampling_prefill", "host_sampling_decode"):
                    modes[f"{event['reason']}_releases"] += 1
    return {
        "server_log": str(path),
        "start_line": start_line,
        "lines_scanned": max(0, total - start_line),
        "detokenizer_invalid_count": len(detokenizer_ids),
        "detokenizer_invalid_ids": detokenizer_ids,
        "adapter_invalid_events": len(events),
        "adapter_invalid_rows": sorted({row for event in events for row in (event.get("rows") or [])}),
        "adapter_events": events,
        "sampling_modes": modes,
        "status": "pass" if not detokenizer_ids and not events else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-line", type=int, default=0)
    args = parser.parse_args()
    record = scan(args.server_log, args.start_line)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"token scan: {record['status']} detokenizer_invalid={record['detokenizer_invalid_count']} "
        f"adapter_events={record['adapter_invalid_events']} rows={record['adapter_invalid_rows']} "
        f"modes={record['sampling_modes']} -> {args.output}"
    )
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

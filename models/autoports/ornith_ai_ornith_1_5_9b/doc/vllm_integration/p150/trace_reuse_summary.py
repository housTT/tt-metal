# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Summarize trace capture/release events, warmup coverage and request signatures for one served run."""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--mixed-shape", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = args.server_log.read_text(errors="replace").splitlines()
    events, warmup, growth, prefills, counters = [], None, [], [], None
    for line in lines:
        m = re.search(
            r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) \| INFO +\| .*Ornith trace (capture|release) (\d+): programs (\d+)(.*)$",
            line,
        )
        if m:
            events.append(
                {
                    "time": m.group(1),
                    "kind": m.group(2),
                    "count": int(m.group(3)),
                    "programs": int(m.group(4)),
                    "detail": m.group(5).strip()[:600],
                }
            )
        m = re.search(r"Ornith prefill bucket warmup: (\{.*\})", line)
        if m:
            warmup = json.loads(m.group(1))
        m = re.search(
            r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) .*Ornith program growth (\d+) -> (\d+) before replay; chain (\S+); counters (\{.*\})",
            line,
        )
        if m:
            growth.append(
                {
                    "time": m.group(1),
                    "before": int(m.group(2)),
                    "after": int(m.group(3)),
                    "chain": m.group(4),
                    "counters": m.group(5),
                }
            )
        m = re.search(
            r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) .*Ornith eager prefill (\d+): lengths (\[.*?\]) starts (\[.*?\]) slots (\[.*?\]) programs (\d+) -> (\d+)",
            line,
        )
        if m:
            prefills.append(
                {
                    "time": m.group(1),
                    "call": int(m.group(2)),
                    "lengths": json.loads(m.group(3)),
                    "starts": json.loads(m.group(4)),
                    "slots": json.loads(m.group(5)),
                    "programs_before": int(m.group(6)),
                    "programs_after": int(m.group(7)),
                }
            )
        m = re.search(r"Serving execution counters: (\{.*\})", line)
        if m:
            counters = json.loads(m.group(1))
    startup_complete = next((i for i, l in enumerate(lines) if "Application startup complete" in l), None)
    startup_events = [e for e in events if e["count"] <= 2 or e["kind"] == "release" and e["count"] <= 1]
    mixed = json.loads(args.mixed_shape.read_text())
    rows = [row for phase in mixed["phases"] for row in phase["results"]]
    signatures = {}
    for row in rows:
        for start, length in zip(row["chunk_starts"], row["chunk_lengths"]):
            bucket = min(2048, ((length + 127) // 128) * 128)
            key = f"prefill bucket {bucket} at chunk start {start}"
            signatures.setdefault(key, {"logical_lengths": set(), "requests": 0})
            signatures[key]["logical_lengths"].add(row["logical_length"])
            signatures[key]["requests"] += 1
    for value in signatures.values():
        value["logical_lengths"] = sorted(value["logical_lengths"])
    summary = {
        "server_log": str(args.server_log),
        "mixed_shape_check": str(args.mixed_shape),
        "prefill_warmup": warmup,
        "trace_events": events,
        "program_growth_events": growth,
        "eager_prefills_after_startup": [p for p in prefills if p["programs_after"] != p["programs_before"]],
        "eager_prefill_calls": len(prefills),
        "startup_captures": sum(1 for e in events if e["kind"] == "capture" and e["count"] <= 2),
        "workload_captures": mixed["summary"]["captures_during_workload"],
        "workload_releases": mixed["summary"]["releases_during_workload"],
        "final_trace_ids": events[-1]["detail"] if events else None,
        "request_signatures": signatures,
        "decode_signature": "one B32 decode trace plus sampler/history traces per sampling graph key (penalties off, penalties on); token, position, RoPE and page-table inputs are persistent device tensors",
        "teardown_counters": counters,
        "mixed_shape_summary": mixed["summary"],
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("startup_captures", "workload_captures", "workload_releases", "eager_prefill_calls")
            }
        )
    )
    print(json.dumps(summary["request_signatures"], indent=1))


if __name__ == "__main__":
    main()

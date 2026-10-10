# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer device spans from signposted Tracy CSV; never sum matmul times."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def window(rows, start, end):
    matches = [i for i, r in enumerate(rows) if r["OP TYPE"] == "signpost" and r["OP CODE"] == start]
    assert len(matches) == 1, (start, len(matches))
    first = matches[0]
    last = next(
        i for i in range(first + 1, len(rows)) if rows[i]["OP TYPE"] == "signpost" and rows[i]["OP CODE"] == end
    )
    return rows[first + 1 : last]


def span(rows):
    expected_device_rows = [r for r in rows if r.get("DEVICE ID") or r.get("OP TYPE") == "tt_dnn_device"]
    device_rows = [r for r in rows if r.get("DEVICE FW START CYCLE") and r.get("DEVICE FW END CYCLE")]
    assert device_rows, "No matching device timestamps; host time is not a substitute"
    assert len(expected_device_rows) == len(device_rows), "Incomplete device timestamps in measured window"
    assert len({r["DEVICE ID"] for r in device_rows}) == 1, "This stage measures one ASIC"
    intervals = [(int(r["DEVICE FW START CYCLE"]), int(r["DEVICE FW END CYCLE"])) for r in device_rows]
    scales = [float(r["DEVICE FW DURATION [ns]"]) / (b - a) for r, (a, b) in zip(device_rows, intervals) if b > a]
    ns_per_cycle = median(scales)
    assert max(abs(s / ns_per_cycle - 1) for s in scales) < 0.02, "Inconsistent device clock units"
    start, end = min(a for a, b in intervals), max(b for a, b in intervals)
    return dict(
        device_us=(end - start) * ns_per_cycle / 1000,
        first_fw_cycle=start,
        last_fw_cycle=end,
        ns_per_cycle=ns_per_cycle,
        operations=len(device_rows),
        expected_device_operations=len(expected_device_rows),
        device_id=device_rows[0]["DEVICE ID"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with args.csv.open() as stream:
        rows = list(csv.DictReader(stream))
    prefill = span(window(rows, "PERF_PREFILL", "PERF_PREFILL_END"))
    decode = [span(window(rows, f"PERF_DECODE_STEP_{i}", f"PERF_DECODE_STEP_{i}_END")) for i in range(128)]
    result = dict(
        workload=dict(input_tokens=4096, output_tokens=128, batch=1, concurrency=1),
        source="single_layer",
        csv=str(args.csv),
        csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        denominator="First device firmware start to last device firmware end of the complete layer; includes every operation and internal device gap. Decode is mean of 128 whole-layer replay spans. Host input refresh/readback lies outside each replay window.",
        prefill_device_us=prefill["device_us"],
        decode_device_us=mean(d["device_us"] for d in decode),
        prefill=prefill,
        decode=decode,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("prefill_device_us", "decode_device_us")}))


if __name__ == "__main__":
    main()

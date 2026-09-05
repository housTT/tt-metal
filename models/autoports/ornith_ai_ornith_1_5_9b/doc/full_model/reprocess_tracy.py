# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only Tracy recovery with exact raw-to-report measurement coverage."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from tracy.process_ops_logs import process_ops


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def key(row):
    return tuple(
        row[field] for field in ("DEVICE ID", "GLOBAL CALL COUNT", "METAL TRACE ID", "METAL TRACE REPLAY SESSION ID")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    logs = args.capture / ".logs"
    report_path = args.capture / "reports" / args.name / f"ops_perf_results_{args.name}.csv"
    assert report_path.exists() if args.verify_existing else not report_path.exists(), report_path
    raw_hashes = {p.name: sha(p) for p in logs.iterdir() if p.is_file()}
    if not args.verify_existing:
        process_ops(args.capture, args.name, False)
    raw = list(csv.DictReader((logs / "cpp_device_perf_report.csv").open()))
    report = list(csv.DictReader(report_path.open()))
    measured = [row for row in report if row.get("DEVICE FW DURATION [ns]")]
    raw_by_id = {key(row): row for row in raw}
    measured_by_id = {key(row): row for row in measured}
    assert len(raw_by_id) == len(raw) == len(measured_by_id) == len(measured)
    assert raw_by_id.keys() == measured_by_id.keys()
    fields = [
        "DEVICE FW DURATION [ns]",
        "DEVICE KERNEL DURATION [ns]",
        "DEVICE FW START CYCLE",
        "DEVICE FW END CYCLE",
    ]
    for identity, raw_row in raw_by_id.items():
        assert all(raw_row[field] == measured_by_id[identity][field] for field in fields), identity
    signposts = [row for row in report if row.get("OP TYPE") == "signpost"]
    assert [row["OP CODE"] for row in signposts] == ["start", "stop"]
    start, stop = [int(row["HOST START TS"]) for row in signposts]
    # Replay rows retain capture-time HOST START TS. Canonical CSV row order is
    # instead sorted by replay timestamps and is the signpost slicing contract.
    marker_indices = [i for i, row in enumerate(report) if row.get("OP TYPE") == "signpost"]
    interval = [row for row in report[marker_indices[0] + 1 : marker_indices[1]] if row.get("DEVICE FW DURATION [ns]")]
    assert interval, "No measured device executions between signposts"
    counts = {}
    for row in interval:
        identity = (
            f"device{row['DEVICE ID']}/trace{row['METAL TRACE ID']}/session{row['METAL TRACE REPLAY SESSION ID']}"
        )
        counts[identity] = counts.get(identity, 0) + 1
    result = dict(
        capture=str(args.capture),
        report=str(report_path),
        raw_files_sha256=raw_hashes,
        report_sha256=sha(report_path),
        raw_device_measurements=len(raw),
        report_device_measurements=len(measured),
        exact_execution_identity_coverage=True,
        exact_timing_fields=fields,
        signposts=[{"name": r["OP CODE"], "host_timestamp_ns": int(r["HOST START TS"])} for r in signposts],
        measured_interval_rows=len(interval),
        measured_interval_execution_counts=counts,
        cpu_only=True,
    )
    assert all(sha(logs / name) == digest for name, digest in raw_hashes.items())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

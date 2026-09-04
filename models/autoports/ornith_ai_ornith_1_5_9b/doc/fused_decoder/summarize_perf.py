# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validate signposted trace coverage and summarize filtered kernel times."""

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "doc/fused_decoder"
parser = argparse.ArgumentParser()
parser.add_argument("--run-suffix", default="")
args = parser.parse_args()
measurements = []


def read_evidence(path):
    return (
        path.read_text()
        if path.exists()
        else gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes()).decode()
    )


for kind in ("linear_attention", "full_attention"):
    for mode in ("prefill", "decode"):
        folder = root / "tracy" / kind
        provenance = json.loads((folder / f"{mode}_provenance.json").read_text())
        iterations = provenance["measured_iterations"]
        log = root / "logs" / f"profile_final_{kind}_{mode}{args.run_suffix}.log"
        log_text = read_evidence(log)
        assert "markers were dropped" not in log_text, log
        assert "1 passed" in log_text, log
        raw_path = folder / f"{mode}_ops.csv"
        raw = list(csv.DictReader(read_evidence(raw_path).splitlines()))
        begin = next(i for i, row in enumerate(raw) if row["OP CODE"] == f"PERF_{mode.upper()}")
        end = next(i for i, row in enumerate(raw) if row["OP CODE"] == f"PERF_{mode.upper()}_END")
        window = raw[begin + 1 : end]
        assert window and all(row["DEVICE KERNEL DURATION [ns]"] for row in window)
        replay_counts = {}
        if mode == "decode":
            # Every measured replay must contain exactly the warm replay's op IDs.
            template = [row["GLOBAL CALL COUNT"] for row in raw if row["METAL TRACE REPLAY SESSION ID"] == "1"]
            assert template
            sessions = sorted({row["METAL TRACE REPLAY SESSION ID"] for row in window}, key=int)
            assert len(sessions) == iterations, (sessions, iterations)
            for session in sessions:
                ops = [row["GLOBAL CALL COUNT"] for row in window if row["METAL TRACE REPLAY SESSION ID"] == session]
                assert ops == template, (kind, session, len(ops), len(template))
                replay_counts[session] = len(ops)
        report_path = folder / f"{mode}_perf_report.csv"
        report = list(csv.DictReader(report_path.open()))
        assert len(report) == len(window), (kind, mode, len(report), len(window))
        total_us = sum(float(row["Device Time"]) for row in report)
        raw_us = sum(float(row["DEVICE KERNEL DURATION [ns]"]) for row in window) / 1000
        assert abs(total_us - raw_us) < 0.01
        measurements.append(
            dict(
                layer_kind=kind,
                mode=mode,
                batch=1,
                sequence_length=2048 if mode == "prefill" else 1,
                decode_context=129 if mode == "decode" else None,
                iterations=iterations,
                device_time_sum_us=total_us,
                device_time_per_forward_ms=total_us / iterations / 1000,
                measured_output_hf_pcc=float(re.search(r"measured-output HF PCC=([\d.]+)", log_text)[1]),
                source=str(report_path.relative_to(root)),
                source_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                unit="Device Time: microseconds",
                rows=len(report),
                measured_replay_op_counts=replay_counts,
                coverage="complete: signpost rows agree with report and each replay has the warm template's op IDs",
            )
        )
result = dict(
    hardware="one Blackhole chip on physical P300c boards",
    profile="p150 single-chip topology",
    tt_perf_report_version="1.2.9",
    metric="sum of filtered Device Time; excludes dispatch gaps; not end-to-end generation throughput",
    measurements=measurements,
)
(root / "performance.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

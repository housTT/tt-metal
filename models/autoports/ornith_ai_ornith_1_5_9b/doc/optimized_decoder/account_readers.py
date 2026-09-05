# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Extract single-replay device rows from alternating reader confirmations."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("kind")
parser.add_argument("--label", default="reader_confirmation")
parser.add_argument("--retain-failed", action="store_true", help="Archive a diagnostic capture; mark it incomplete")
args = parser.parse_args()
root = Path(__file__).resolve().parent
folder = root / "tracy" / args.kind
sources = list((folder / "raw" / args.label).rglob("ops_perf_results*.csv"))
assert len(sources) == 1, sources
source = sources[0]
folder = folder / args.label
folder.mkdir(exist_ok=True)
(folder / "reader_ops.csv.gz").write_bytes(gzip.compress(source.read_bytes(), mtime=0))
for filename in ("cpp_device_perf_report.csv", "tracy_ops_data.csv"):
    raw = root / "tracy" / args.kind / "raw" / args.label / ".logs" / filename
    if raw.exists():
        (folder / (filename + ".gz")).write_bytes(gzip.compress(raw.read_bytes(), mtime=0))
with source.open() as stream:
    rows = list(csv.DictReader(stream))
measurements = json.loads((root / f"{args.label}_{args.kind}.json").read_text())
failures = [row for row in measurements if row["status"] != "pass"]
assert args.retain_failed or not failures, [(row["role"], row["readers"], row["status"]) for row in failures]
for role in dict.fromkeys(row["role"] for row in measurements):
    group = [row for row in measurements if row["role"] == role]
    assert [row["readers"] for row in group] == [1, 2, 3, 3, 2, 1], role
    assert len({(row["cores"], row["block_w"], row["per_core_N"]) for row in group}) == 1, role
results = []
for measured in measurements:
    if measured["status"] != "pass":
        continue
    label = measured["profile_signpost"]
    starts = [i for i, row in enumerate(rows) if row["OP CODE"] == label]
    ends = [i for i, row in enumerate(rows) if row["OP CODE"] == label + "_END"]
    assert len(starts) == len(ends) == 1, label
    start, end = starts[0], ends[0]
    assert end > start, label
    operations = [row for row in rows[start + 1 : end] if row.get("DEVICE KERNEL DURATION [ns]")]
    matmuls = [row for row in operations if row["OP CODE"] == "MatmulDeviceOperation"]
    assert len(matmuls) == 1
    matmul = matmuls[0]
    replay_keys = {
        (row["DEVICE ID"], row["METAL TRACE ID"], row["METAL TRACE REPLAY SESSION ID"]) for row in operations
    }
    assert len(replay_keys) == 1 and all(replay_keys.pop()), label
    same_op = [
        row for row in rows if row.get("GLOBAL CALL COUNT") == matmul["GLOBAL CALL COUNT"] and row.get("CORE COUNT")
    ]
    assert float(matmul["CORE COUNT"]) == max(float(row["CORE COUNT"]) for row in same_op), label
    physical_bytes = (
        measured["per_bank_tiles"]
        * measured["dram_banks"]
        * (measured["shape"][1] // 32)
        * measured["weight_tile_bytes"]
    )
    logical_bytes = measured["shape"][1] * measured["shape"][2] / 2
    device_us = float(matmul["DEVICE KERNEL DURATION [ns]"]) / 1000
    result = {
        **measured,
        "matmul_device_us": device_us,
        "all_ops_device_us": sum(float(row["DEVICE KERNEL DURATION [ns]"]) for row in operations) / 1000,
        "device_physical_GBs": physical_bytes / device_us / 1000,
        "device_logical_GBs": logical_bytes / device_us / 1000,
        "device_percent_declared_peak": physical_bytes / device_us / 1000 / 512 * 100,
        "runtime_input_dtype": matmul["INPUT_0_DATATYPE"],
        "runtime_weight_dtype": matmul["INPUT_1_DATATYPE"],
        "runtime_fidelity": matmul["MATH FIDELITY"],
        "runtime_attributes": matmul["ATTRIBUTES"],
        "device_identity": {
            key: matmul[key]
            for key in (
                "DEVICE ID",
                "GLOBAL CALL COUNT",
                "METAL TRACE ID",
                "METAL TRACE REPLAY SESSION ID",
                "CORE COUNT",
                "COMPUTE KERNEL SOURCE",
                "COMPUTE KERNEL HASH",
                "DATA MOVEMENT KERNEL SOURCE",
                "DATA MOVEMENT KERNEL HASH",
            )
        },
    }
    assert "BFLOAT4" in result["runtime_weight_dtype"]
    assert result["runtime_fidelity"] == "LoFi"
    assert f"num_workers_per_dram_bank={measured['readers']}" in result["runtime_attributes"]
    results.append(result)
report = dict(
    status="incomplete_diagnostic" if failures else "complete",
    failed_measurements=failures,
    source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    physical_hardware="one Blackhole chip on P300c boards",
    declared_peak_DRAM_GBs=512,
    notes="Alternating 1/2/3/3/2/1 reader order at fixed per-role geometry. Device time from the signposted warmed replay; profiled host timing is reported separately in each row.",
    measurements=results,
)
(folder / "reader_accounting.json").write_text(json.dumps(report, indent=2) + "\n")
columns = [
    "layer",
    "role",
    "readers",
    "cores",
    "block_w",
    "per_core_N",
    "dram_banks",
    "weight_tile_bytes",
    "matmul_device_us",
    "all_ops_device_us",
    "device_physical_GBs",
    "device_logical_GBs",
    "device_percent_declared_peak",
    "per_bank_tiles",
    "per_reader_row_bytes",
    "trace_us",
    "pcc_vs_interleaved",
    "exact_eager_trace",
    "runtime_input_dtype",
    "runtime_weight_dtype",
    "runtime_fidelity",
    "profile_signpost",
]
with (folder / "reader_accounting.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
print(f"Accounted {len(results)} reader replays")

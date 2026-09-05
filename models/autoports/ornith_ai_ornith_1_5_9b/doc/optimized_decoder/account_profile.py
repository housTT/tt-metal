# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reconcile host/profile timing and a declared single-chip DRAM roofline."""

import argparse
import csv
import gzip
import json
import math
import re
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("kind", choices=("linear_attention", "full_attention"))
parser.add_argument("label")
parser.add_argument("log", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parent
folder = root / "tracy" / args.kind / args.label
with gzip.open(folder / "decode_ops.csv.gz", "rt") as f:
    all_rows = list(csv.DictReader(f))
start = next(i for i, x in enumerate(all_rows) if x["OP CODE"] == "PERF_DECODE")
end = next(i for i, x in enumerate(all_rows) if i > start and x["OP CODE"] == "PERF_DECODE_END")
rows = [x for x in all_rows[start + 1 : end] if x.get("DEVICE KERNEL DURATION [ns]")]
measurements = [
    json.loads(line.split("PROFILE_MEASUREMENT ", 1)[1])
    for line in args.log.read_text().splitlines()
    if "PROFILE_MEASUREMENT " in line
]
measurement = next(x for x in measurements if x["mode"] == "decode")
iters = measurement["iterations"]


def number(row, key):
    return float(row[key])


def dimension(row, key):
    return int(re.match(r"\d+", row[key]).group())


kernel_ms = sum(number(x, "DEVICE KERNEL DURATION [ns]") for x in rows) / iters / 1e6
gaps_ms = sum(number(x, "OP TO OP LATENCY [ns]") for x in rows[1:]) / iters / 1e6
rates = [
    (number(x, "DEVICE FW END CYCLE") - number(x, "DEVICE FW START CYCLE")) / number(x, "DEVICE FW DURATION [ns]")
    for x in rows
    if number(x, "DEVICE FW DURATION [ns]") > 0
]
clock = statistics.median(rates)
fw_span_ms = (
    (max(number(x, "DEVICE FW END CYCLE") for x in rows) - min(number(x, "DEVICE FW START CYCLE") for x in rows))
    / clock
    / iters
    / 1e6
)
roles = (
    ("gdn_packed", "gdn_z_epilogue", "gdn_out", "gate_proj", "up_proj", "down_proj")
    if args.kind == "linear_attention"
    else ("qkvg", "o_proj", "gate_proj", "up_proj", "down_proj")
)
matmuls = [
    x
    for x in rows
    if x["OP CODE"] == "MatmulDeviceOperation"
    and dimension(x, "INPUT_1_Y_PAD[LOGICAL]") >= 4096
    and dimension(x, "INPUT_1_X_PAD[LOGICAL]") >= 4096
]
assert len(matmuls) == len(roles) * iters, (len(matmuls), len(roles), iters)
projection_rows = []
for i, role in enumerate(roles):
    group = matmuls[i :: len(roles)]
    row = group[0]
    k, n = dimension(row, "INPUT_1_Y_PAD[LOGICAL]"), dimension(row, "INPUT_1_X_PAD[LOGICAL]")
    dtype = row["INPUT_1_DATATYPE"].upper()
    tile_bytes = 576 if "BFLOAT4" in dtype else 1088 if "BFLOAT8" in dtype else 2048 if "BFLOAT16" in dtype else 4096
    attributes = row["ATTRIBUTES"]
    match = re.search(r"num_workers_per_dram_bank\s*=\s*(\d+)", attributes)
    readers = int(match.group(1)) if match else None
    banks = 8
    weight_tiles = math.ceil(k / 32) * math.ceil(n / 32)
    if readers:
        weight_tiles = math.ceil(k / 32) * math.ceil(n / (32 * banks * readers)) * banks * readers
    weight_bytes = weight_tiles * tile_bytes
    device_us = statistics.mean(number(x, "DEVICE KERNEL DURATION [ns]") for x in group) / 1000
    projection_rows.append(
        dict(
            role=role,
            k=k,
            n=n,
            input_dtype=row["INPUT_0_DATATYPE"],
            weight_dtype=row["INPUT_1_DATATYPE"],
            output_dtype=row["OUTPUT_0_DATATYPE"],
            fidelity=row["MATH FIDELITY"],
            readers=readers,
            physical_weight_bytes=weight_bytes,
            device_us=device_us,
            physical_weight_GBs=weight_bytes / device_us / 1000,
            program_and_memory_attributes=attributes,
        )
    )
if not measurement["baseline"]:
    expected = json.loads((root / "best_geometry_config.json").read_text())["role_configs"]
    for row in projection_rows:
        assert row["weight_dtype"].upper() == "BFLOAT4_B", row
        assert row["fidelity"].lower() == "lofi", row
        assert row["readers"] == expected[row["role"]]["readers"], row
weight_bytes = sum(x["physical_weight_bytes"] for x in projection_rows)
kv_bytes = 0
if args.kind == "full_attention":
    sdpa = next(x for x in rows if "SDPA" in x["OP CODE"].upper() or "SCALED" in x["OP CODE"].upper())
    kv_tile_bytes = []
    for index in (1, 2):
        dtype = sdpa.get(f"INPUT_{index}_DATATYPE", "").upper()
        kv_tile_bytes.append(576 if "BFLOAT4" in dtype else 1088 if "BFLOAT8" in dtype else 2048)
    kv_bytes = 4 * math.ceil((measurement["decode_position"] + 1) / 32) * 8 * sum(kv_tile_bytes)
report = dict(
    measurement=measurement,
    kernel_sum_ms=kernel_ms,
    op_to_op_gaps_ms=gaps_ms,
    device_fw_span_ms=fw_span_ms,
    profile_clock_cycles_per_ns=clock,
    host_minus_device_fw_span_ms=measurement["host_wall_per_iteration_ms"] - fw_span_ms,
    declared_peak_DRAM_GBs=512,
    peak_source="tt-perf-report Blackhole model; one chip on P300c boards",
    projection_physical_weight_bytes=weight_bytes,
    minimum_kv_read_bytes=kv_bytes,
    weight_and_kv_bandwidth_lower_bound_ms=(weight_bytes + kv_bytes) / 512e6,
    roofline_limitations="Bandwidth-only lower bound; excludes residual/conv/state traffic, launches, synchronization, and compute. BFP tile exponent overhead and DRAM-reader padding are included. KV counts one read per stored K/V element; actual SDPA rereads may be higher.",
    profile_limitations="Host and device accounting come from this profiled run. Headline candidate latency is separately measured without profiling; profiler overhead is not silently subtracted.",
    projections=projection_rows,
)
(folder / "decode_accounting.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("projections",)}, indent=2))

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Copy an ops CSV and render the required signposted CSV and text reports."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("prefill", "decode"))
parser.add_argument("--source-csv", type=Path, help="Explicit canonical ops CSV when a capture has multiple reports")
parser.add_argument("--capture", help="Raw capture directory; defaults to mode")
parser.add_argument("--iterations", type=int)
parser.add_argument("--decode-context", type=int, default=129)
parser.add_argument("--label", default="")
args = parser.parse_args()
root = Path(__file__).resolve().parent / "tracy"
sources = (
    [args.source_csv]
    if args.source_csv
    else list((root / "raw" / (args.capture or args.mode)).rglob("ops_perf_results*.csv"))
)
if len(sources) != 1:
    raise RuntimeError(f"expected one ops CSV, found {sources}")
source = sources[0]
if args.label:
    root = root / args.label
    root.mkdir(parents=True, exist_ok=True)
ops = root / f"{args.mode}_ops.csv"
shutil.copyfile(source, ops)
ops.with_suffix(".csv.gz").write_bytes(gzip.compress(ops.read_bytes(), mtime=0))
base = [
    "tt-perf-report",
    str(ops),
    "--start-signpost",
    "start",
    "--end-signpost",
    "stop",
    "--no-summary",
    "--no-color",
]
if args.mode == "decode":
    base += ["--tracing-mode"]
report = root / f"{args.mode}_perf_report.csv"
commands = [base + ["--csv", str(report)], base]
for command, suffix in zip(commands, ("console.log", "txt")):
    with (root / f"{args.mode}_perf_report.{suffix}").open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
# Keep generated review artifacts compatible with the repository's whitespace checks.
# Raw ops CSV bytes remain preserved separately in the compressed capture archive.
for path in (report, root / f"{args.mode}_perf_report.txt"):
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
(root / f"{args.mode}_provenance.json").write_text(
    json.dumps(
        {
            "source": str(source),
            "ops_sha256": hashlib.sha256(ops.read_bytes()).hexdigest(),
            "commands": commands,
            "physical_hardware": "four Blackhole chips on two P300c boards",
            "profile": "p150x4 (1x4 mesh on P300c hardware)",
            "measured_iterations": args.iterations or (1 if args.mode == "prefill" else 4),
            "sequence_length": 128 if args.mode == "prefill" else 1,
            "decode_context": None if args.mode == "prefill" else args.decode_context,
            "batch": 1,
            "cache_context": 262144,
            "layers": [0, 3],
        },
        indent=2,
    )
    + "\n"
)
print(report)

# Device clocks are independent. Keep the original merged report above, then
# render each rank separately. The first device gap starts BEFORE the signpost
# (it includes ReadDeviceProfiler/setup); it is outside the measured window.
with ops.open() as stream:
    reader = csv.DictReader(stream)
    fields, rows = reader.fieldnames, list(reader)
start = next(i for i, row in enumerate(rows) if row["OP CODE"] == "start")
end = next(i for i, row in enumerate(rows) if row["OP CODE"] == "stop")
window = rows[start + 1 : end]
accounting = {}
for device in sorted({row["DEVICE ID"] for row in window if row["DEVICE ID"]}):
    selected = [dict(row) for row in window if row["DEVICE ID"] == device]
    selected.sort(key=lambda row: float(row["DEVICE FW START CYCLE"] or 0))
    first = selected[0]
    removed_gap = float(first["OP TO OP LATENCY [ns]"] or 0)
    first["OP TO OP LATENCY [ns]"] = "0"
    first["OP TO OP LATENCY BR/NRISC START [ns]"] = "0"
    rank_ops = root / f"{args.mode}_device{device}_window.csv"
    with rank_ops.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([rows[start], *selected, rows[end]])
    command = [arg if arg != str(ops) else str(rank_ops) for arg in base]
    for suffix, extra in [
        ("txt", []),
        ("console.log", ["--csv", str(root / f"{args.mode}_device{device}_report.csv")]),
    ]:
        with (root / f"{args.mode}_device{device}_report.{suffix}").open("w") as stream:
            subprocess.run(command + extra, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if suffix == "txt":
            text_report = root / f"{args.mode}_device{device}_report.txt"
            text_report.write_text(
                "\n".join(line.rstrip() for line in text_report.read_text().splitlines()).rstrip() + "\n"
            )
    iterations = args.iterations or (1 if args.mode == "prefill" else 4)
    groups = {}
    for row in selected:
        group = (
            "collective"
            if any(name in row["OP CODE"] for name in ("AllGather", "ReduceScatter", "AllReduce"))
            else (
                "matmul"
                if "Matmul" in row["OP CODE"]
                else (
                    "data_movement"
                    if any(
                        name in row["OP CODE"]
                        for name in ("Sharded", "Reshard", "Transpose", "Slice", "Copy", "Reshape", "Tilize", "Concat")
                    )
                    else "other_compute"
                )
            )
        )
        groups[group] = groups.get(group, 0) + float(row["DEVICE KERNEL DURATION [ns]"] or 0) / 1000 / iterations
    accounting[device] = dict(
        operations=len(selected),
        iterations=iterations,
        removed_pre_window_gap_ns=removed_gap,
        kernel_us_per_iteration=sum(float(row["DEVICE KERNEL DURATION [ns]"] or 0) for row in selected)
        / 1000
        / iterations,
        gap_us_per_iteration=sum(float(row["OP TO OP LATENCY [ns]"] or 0) for row in selected) / 1000 / iterations,
        kernel_groups_us_per_iteration=groups,
        normalized_ops_sha256=hashlib.sha256(rank_ops.read_bytes()).hexdigest(),
        command=command,
    )
(root / f"{args.mode}_rank_accounting.json").write_text(
    json.dumps(
        dict(
            correction="Remove only the first per-device gap, which begins before the signposted window; preserve every kernel duration and all intra-window gaps. Device times are reported independently, never summed across chips.",
            devices=accounting,
        ),
        indent=2,
    )
    + "\n"
)

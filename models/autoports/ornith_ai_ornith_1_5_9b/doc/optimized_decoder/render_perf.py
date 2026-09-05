# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Copy an ops CSV and render the required signposted CSV and text reports."""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("kind", choices=("linear_attention", "full_attention"))
parser.add_argument("mode", choices=("prefill", "decode"))
parser.add_argument("--capture", help="Raw capture directory; defaults to mode")
parser.add_argument("--iterations", type=int)
parser.add_argument("--decode-context", type=int, default=2049)
parser.add_argument("--label", default="")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2] / "doc/optimized_decoder/tracy" / args.kind
sources = list((root / "raw" / (args.capture or args.mode)).rglob("ops_perf_results*.csv"))
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
    f"PERF_{args.mode.upper()}",
    "--end-signpost",
    f"PERF_{args.mode.upper()}_END",
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
            "physical_hardware": "one Blackhole chip on P300c boards",
            "profile": "p150 (single-chip topology, not a measured P150 board)",
            "measured_iterations": args.iterations or (1 if args.mode == "prefill" else 32),
            "sequence_length": 2048 if args.mode == "prefill" else 1,
            "decode_context": None if args.mode == "prefill" else args.decode_context,
            "batch": 1,
        },
        indent=2,
    )
    + "\n"
)
print(report)

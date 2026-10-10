# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce performance tables and whole-layer estimates from existing captures."""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-type", choices=("linear_attention", "full_attention"), required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--policy", choices=("functional", "fused"), default="functional")
    args = parser.parse_args()
    evidence = args.evidence_dir or (Path(__file__).resolve().parents[1] / "doc" / "functional_decoder")
    directory = evidence / "tracy" / args.layer_type
    csv_path = directory / "ops.csv"
    if not csv_path.exists():
        captures = list(directory.rglob("ops_perf_results*.csv"))
        if captures:
            assert len(captures) == 1, captures
            shutil.copyfile(captures[0], csv_path)
        else:
            with gzip.open(directory / "ops.csv.gz", "rb") as src, csv_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    commands = []

    def run(command, log=None):
        commands.append(command)
        if log:
            with log.open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        else:
            subprocess.run(command, check=True)

    module = "models.autoports.qwen_qwen3_5_35b_a3b.tests."
    run(
        [
            sys.executable,
            "-m",
            module + "summarize_perf",
            str(csv_path),
            "--output",
            str(directory / "measurement.json"),
        ]
    )
    for phase in ("prefill", "decode"):
        command = [
            "tt-perf-report",
            str(csv_path),
            "--start-signpost",
            "PERF_" + phase.upper(),
            "--end-signpost",
            "PERF_" + phase.upper() + "_END",
            "--no-advice",
        ]
        run(
            command + ["--csv", str(directory / (phase + "_perf_report.csv"))],
            directory / (phase + "_perf_report.console.log"),
        )
        run(command + ["--no-summary"], directory / (phase + "_perf_report.txt"))
    run(
        [
            sys.executable,
            "-m",
            module + "estimate_roofline",
            "--policy",
            args.policy,
            str(directory / "measurement.json"),
            "--layer-type",
            args.layer_type,
            "--output",
            str(directory / "roofline.json"),
        ]
    )
    with csv_path.open("rb") as src, gzip.open(directory / "ops.csv.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    result = dict(
        commands=commands,
        ops_csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        archived_csv=str(directory / "ops.csv.gz"),
        tt_perf_report_version="1.4.1",
    )
    (directory / "processing_provenance.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

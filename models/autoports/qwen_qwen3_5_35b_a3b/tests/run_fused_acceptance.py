# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized fused correctness gates with exact commands and fresh artifacts."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--group", choices=("positions", "reuse", "context", "watcher", "pytest"), required=True)
    parser.add_argument("--fusions")
    parser.add_argument("--decoder", choices=("fused", "optimized"), default="fused")
    parser.add_argument("--layer", type=int, choices=(0, 3))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    module = "models.autoports.qwen_qwen3_5_35b_a3b.tests."
    tracking = dict(TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
    jobs = []

    def add(name, runner, arguments, watcher=False):
        command = [
            sys.executable,
            "-m",
            module + runner,
            "--decoder",
            args.decoder,
            *arguments,
            "--output",
            str(args.directory / (name + ".json")),
        ]
        if args.fusions is not None:
            command.extend(("--fusions", args.fusions))
        extra = dict(tracking)
        if watcher:
            extra.update(
                TT_METAL_WATCHER="10", TT_METAL_LOGS_PATH=str((args.directory / ("watcher_" + name)).resolve())
            )
        jobs.append((name, command, extra))

    for layer, kind in ((0, "linear"), (3, "full")):
        if args.layer is not None and layer != args.layer:
            continue
        if args.group == "positions":
            add(kind, "run_positions", ["--layer", str(layer)])
        elif args.group == "reuse":
            add(kind, "run_reuse", ["--layer", str(layer)])
            if layer == 3:
                add("full_page128", "run_reuse", ["--layer", "3", "--page-size", "128"])
        elif args.group == "context":
            for length, steps in ((262143, 1), (262144, 0)):
                add(
                    f"{kind}_{length}",
                    "run_layer",
                    [
                        "--layer",
                        str(layer),
                        "--length",
                        str(length),
                        "--steps",
                        str(steps),
                        "--trace",
                        "--audit",
                        "--sparse-reference",
                    ],
                )
        elif args.group == "watcher":
            add(
                kind,
                "run_layer",
                ["--layer", str(layer), "--length", "129", "--steps", "8", "--trace", "--audit"],
                watcher=True,
            )
    if args.group == "watcher" and args.layer != 0:
        for length, batch in ((257, 2), (65, 1)):
            add(
                f"small_pool_{length}_b{batch}",
                "run_layer",
                [
                    "--layer",
                    "3",
                    "--length",
                    str(length),
                    "--steps",
                    "31",
                    "--batch",
                    str(batch),
                    "--minimum-cache-pages",
                    "0",
                    "--cache-guard-pages",
                    "0",
                    "--trace",
                    "--audit",
                ],
                watcher=True,
            )
    if args.group == "pytest":
        jobs.append(
            (
                "device",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-s",
                    str(Path(__file__).with_name(f"test_{args.decoder}_decoder_device.py")),
                    "--basetemp",
                    str(args.directory / "pytest_artifacts"),
                ],
                tracking,
            )
        )
    records = []
    for name, command, extra in jobs:
        print("START", name, flush=True)
        env = dict(os.environ)
        for key in ("TT_METAL_WATCHER", "TT_METAL_LOGS_PATH", *tracking):
            env.pop(key, None)
        env.update(extra)
        started = time.time()
        with (args.directory / (name + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        records.append(
            dict(
                name=name,
                command=command,
                environment=extra,
                exit_code=result.returncode,
                elapsed_seconds=time.time() - started,
            )
        )
        (args.directory / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        print("END", name, "exit", result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

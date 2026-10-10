# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized functional-stage evidence commands; stop on the first failed gate."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "doc" / "functional_decoder"
    module = "models.autoports.qwen_qwen3_5_35b_a3b.tests."
    tracking = dict(TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
    jobs = []

    def add(name, runner, arguments, env=None):
        jobs.append(
            (
                name,
                [sys.executable, "-m", module + runner, *arguments, "--output", str(evidence / (name + ".json"))],
                env or tracking,
            )
        )

    add(
        "full_accept_4096_128",
        "run_layer",
        ["--layer", "3", "--length", "4096", "--steps", "128", "--trace", "--audit"],
    )
    add("positions_full_accept", "run_positions", ["--layer", "3"])
    for layer, kind in ((0, "linear"), (3, "full")):
        add(kind + "_reuse_accept", "run_reuse", ["--layer", str(layer)])
    add("full_reuse_page128", "run_reuse", ["--layer", "3", "--page-size", "128"])
    for layer, kind in ((0, "linear"), (3, "full")):
        for length, steps in ((262143, 1), (262144, 0)):
            add(
                f"{kind}_accept_context{length}",
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
    jobs.append(
        (
            "pytest_device_accept",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-s",
                str(root / "tests" / "test_functional_decoder_device.py"),
                "--basetemp",
                str(evidence / "pytest_device_artifacts"),
            ],
            tracking,
        )
    )
    for layer, kind in ((0, "linear"), (3, "full")):
        add(
            kind + "_watcher",
            "run_layer",
            ["--layer", str(layer), "--length", "129", "--steps", "8", "--trace", "--audit"],
            dict(tracking, TT_METAL_WATCHER="10", TT_METAL_LOGS_PATH=str(evidence / ("watcher_" + kind))),
        )
    results = []
    for name, command, extra_env in jobs:
        print("START", name, flush=True)
        start = time.time()
        with (evidence / (name + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, **extra_env))
        results.append(
            dict(
                name=name,
                command=command,
                environment=extra_env,
                exit_code=result.returncode,
                elapsed_seconds=time.time() - start,
                console_log=str(evidence / (name + ".log")),
            )
        )
        (evidence / "acceptance_commands.json").write_text(json.dumps(results, indent=2) + "\n")
        print("END", name, "exit", result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

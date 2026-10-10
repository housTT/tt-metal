# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Final review regressions and profiles, serialized with command provenance."""

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
    for length, batch in ((257, 2), (65, 1)):
        name = f"small_pool{(length+31)//32}_b{batch}"
        command = [
            sys.executable,
            "-m",
            module + "run_layer",
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
            "--output",
            str(evidence / (name + ".json")),
        ]
        env = dict(tracking, TT_METAL_WATCHER="10", TT_METAL_LOGS_PATH=str(evidence / ("watcher_" + name)))
        jobs.append((name, command, env))
    jobs.append(
        (
            "full_reuse_postreview",
            [
                sys.executable,
                "-m",
                module + "run_reuse",
                "--layer",
                "3",
                "--output",
                str(evidence / "full_reuse_postreview.json"),
            ],
            tracking,
        )
    )
    jobs.append(
        (
            "cpu_contract_postreview",
            [sys.executable, "-m", "pytest", "-q", str(root / "tests/test_functional_contract.py")],
            {},
        )
    )
    for layer, kind in ((0, "linear"), (3, "full")):
        name = kind + "_profile"
        jobs.append(
            (
                name,
                [
                    sys.executable,
                    "-m",
                    "tracy",
                    "-r",
                    "-p",
                    "-v",
                    "--no-web-server",
                    "--check-exit-code",
                    "--op-support-count",
                    "32768",
                    "--dump-device-data-mid-run",
                    "-o",
                    str(evidence / "tracy" / (kind + "_attention")),
                    "-m",
                    module + "run_layer",
                    "--layer",
                    str(layer),
                    "--length",
                    "4096",
                    "--steps",
                    "128",
                    "--trace",
                    "--profile",
                    "--output",
                    str(evidence / (name + ".json")),
                ],
                {},
            )
        )
    records = []
    for name, command, extra in jobs:
        print("START", name, flush=True)
        env = dict(os.environ)
        for key in ("TT_METAL_WATCHER", "TT_METAL_LOGS_PATH", *tracking):
            env.pop(key, None)
        env.update(extra)
        start = time.time()
        with (evidence / (name + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        records.append(
            dict(
                name=name,
                command=command,
                environment=extra,
                exit_code=result.returncode,
                elapsed_seconds=time.time() - start,
                console_log=str(evidence / (name + ".log")),
            )
        )
        (evidence / "completion_commands.json").write_text(json.dumps(records, indent=2) + "\n")
        print("END", name, "exit", result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

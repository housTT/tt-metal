# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize the two complete 4096/128 layer profiles with bounded profiler buffers."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-type", choices=("linear_attention", "full_attention"))
    parser.add_argument("--decoder", choices=("functional", "fused", "optimized", "multichip"), default="functional")
    parser.add_argument("--mesh", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--tracy-arg", action="append", default=[])
    parser.add_argument("--fusions")
    parser.add_argument("--optimization-json", help="Omit to profile delivered optimized defaults")
    parser.add_argument("--input-activations", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    evidence = args.evidence_dir or (Path(__file__).resolve().parents[1] / "doc" / (args.decoder + "_decoder"))
    evidence = evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    records = []
    for layer, kind in ((0, "linear"), (3, "full")):
        if args.layer_type and args.layer_type != kind + "_attention":
            continue
        name = kind + "_profile"
        command = [
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
            *args.tracy_arg,
            "-o",
            str(evidence / "tracy" / (kind + "_attention")),
            "-m",
            "models.autoports.qwen_qwen3_5_35b_a3b.tests.run_layer",
            "--layer",
            str(layer),
            "--checkpoint",
            str(repo_root.parent / "weights" / "Qwen3.5-35B-A3B"),
            "--length",
            "4096",
            "--steps",
            "128",
            "--trace",
            "--profile",
            "--output",
            str(evidence / (name + ".json")),
            "--decoder",
            args.decoder,
            "--mesh",
            str(args.mesh),
        ]
        if args.optimization_json is not None:
            command.extend(("--optimization-json", args.optimization_json))
        if args.input_activations:
            command.extend(("--input-activations", str(args.input_activations.resolve())))
        if args.fusions is not None:
            command.extend(("--fusions", args.fusions))
        env = dict(os.environ)
        for key in (
            "TT_METAL_WATCHER",
            "TT_METAL_LOGS_PATH",
            "TT_METAL_TRACE_ALLOC_TRACKING",
            "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
        ):
            env.pop(key, None)
        start = time.time()
        print("START", name, flush=True)
        with (evidence / (name + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=evidence)
        records.append(
            dict(
                name=name,
                command=command,
                exit_code=result.returncode,
                elapsed_seconds=time.time() - start,
                console_log=str(evidence / (name + ".log")),
                cwd=str(evidence),
                profiler_program_support_count=32768,
                explicit_between_pass_drains=True,
                environment={
                    key: env[key]
                    for key in ("TT_METAL_HOME", "TT_METAL_RUNTIME_ROOT", "TT_METAL_KERNEL_PATH", "TT_METAL_CACHE",
                                "TT_VISIBLE_DEVICES", "QDEV")
                    if key in env
                },
            )
        )
        (evidence / "profile_commands.json").write_text(json.dumps(records, indent=2) + "\n")
        print("END", name, "exit", result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

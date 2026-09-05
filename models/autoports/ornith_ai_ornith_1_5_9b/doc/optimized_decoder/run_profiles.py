# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serialized, advice-enabled real-input prefill/decode profile collection."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("label")
parser.add_argument("--baseline", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
doc = root / "doc/optimized_decoder"
env = os.environ.copy()
env.update(
    TORCHINDUCTOR_CACHE_DIR="/home/hous/dev/ornith-1.5-9b/state/torch-cache",
    OMP_NUM_THREADS="8",
    ORNITH_WEIGHTS="real",
    ORNITH_PROFILE_BASELINE="1" if args.baseline else "0",
    ORNITH_PERF_DECODE_ITERS="4",
)
assert not env.get("TT_METAL_WATCHER"), "Watcher and profiler runs must remain separate"
for kind in ("linear_attention", "full_attention"):
    for mode in ("prefill", "decode"):
        name = f"profile_{args.label}_{kind}_{mode}"
        capture = f"{args.label}_{mode}"
        command = [
            sys.executable,
            str(doc / "record_run.py"),
            name,
            sys.executable,
            "-m",
            "tracy",
            "-r",
            "-p",
            "-v",
            "--check-exit-code",
            "--op-support-count",
            "8000",
            "--web-app-port",
            "18940",
            "-o",
            str(doc / "tracy" / kind / "raw" / capture),
            "-m",
            "pytest",
            str(root / "tests/test_optimized_profile.py"),
            "-k",
            f"{mode}-{kind}",
            "-v",
            "-s",
        ]
        subprocess.run(command, env=env, check=True)
        subprocess.run(
            [
                sys.executable,
                str(doc / "render_perf.py"),
                kind,
                mode,
                "--capture",
                capture,
                "--label",
                args.label,
                "--iterations",
                "1" if mode == "prefill" else "4",
                "--decode-context",
                "2049",
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(doc / "account_profile.py"),
            kind,
            args.label,
            str(doc / "logs" / f"profile_{args.label}_{kind}_decode.log"),
        ],
        check=True,
    )

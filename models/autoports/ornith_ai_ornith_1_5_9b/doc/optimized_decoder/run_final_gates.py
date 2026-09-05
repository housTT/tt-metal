# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run final production-default gates serially with immutable per-run evidence."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("label")
parser.add_argument("gates", nargs="*", default=["short", "long", "batch", "watcher", "pair"])
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
doc = root / "doc/optimized_decoder"
env = os.environ.copy()
for key in list(env):
    if (
        key.startswith("ORNITH_")
        or key.startswith("TT_METAL_PROFILER")
        or key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER")
    ):
        env.pop(key)
env.update(
    TORCHINDUCTOR_CACHE_DIR="/home/hous/dev/ornith-1.5-9b/state/torch-cache", OMP_NUM_THREADS="8", ORNITH_WEIGHTS="real"
)
checks = {
    "short": ["test_optimized_decoder.py", "-m", "not long"],
    "long": ["test_optimized_decoder.py", "-m", "long"],
    "batch": ["test_optimized_batch_resources.py"],
    "watcher": ["test_optimized_decoder.py", "-m", "not long"],
    "pair": ["test_optimization_experiments.py"],
    "prefill_trace": ["test_optimized_prefill_trace.py"],
    "watcher_prefill_trace": ["test_optimized_prefill_trace.py"],
    "watcher_batch": [
        "test_optimized_batch_resources.py",
        "-k",
        "(test_batch_restored_trace and 12 and sharded) or (test_prime_batch_rope_trace and 31 and sharded)",
    ],
}
for gate in args.gates:
    check = checks[gate]
    run_env = dict(env)
    if gate.startswith("watcher"):
        run_env["TT_METAL_WATCHER"] = "10"
    command = [
        sys.executable,
        str(doc / "record_run.py"),
        f"final_{args.label}_{gate}",
        sys.executable,
        "-m",
        "pytest",
        str(root / "tests" / check[0]),
        *check[1:],
        "-x",
        "-v",
        "-s",
    ]
    subprocess.run(command, env=run_env, check=True)

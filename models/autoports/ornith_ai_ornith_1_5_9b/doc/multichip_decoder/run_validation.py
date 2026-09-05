# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serialize complete worker-watcher contracts, stack comparison and timings."""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("label")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
doc = root / "doc/multichip_decoder"
env = os.environ.copy()
assert not env.get("TT_METAL_DEVICE_PROFILER"), "Watcher and profiler must be separate"
env.update(TORCHINDUCTOR_CACHE_DIR="/home/hous/dev/ornith-1.5-9b/state/torch-cache", OMP_NUM_THREADS="8")
raw = doc / "watcher_raw" / args.label
raw.mkdir(parents=True, exist_ok=False)
watcher = dict(
    env,
    TT_METAL_WATCHER="10",
    TT_METAL_WATCHER_DISABLE_ETH="1",
    TT_METAL_WATCHER_APPEND="1",
    TT_METAL_LOGS_PATH=str(raw),
)


def run(name, command, environment):
    subprocess.run([sys.executable, str(doc / "record_run.py"), name, *command], env=environment, check=True)


try:
    run(
        f"{args.label}_watcher_contracts",
        [
            "timeout",
            "1800",
            sys.executable,
            "-m",
            "pytest",
            "-x",
            "-q",
            "-s",
            str(root / "tests/test_multichip_decoder.py"),
            str(root / "tests/test_multichip_capacity.py"),
            str(root / "tests/test_multichip_native_cache.py"),
        ],
        watcher,
    )
finally:
    manifest = []
    for source in sorted(raw.rglob("*")):
        if source.is_file():
            target = doc / "logs" / f"{args.label}_watcher_{str(source.relative_to(raw)).replace('/', '_')}.gz"
            data = source.read_bytes()
            target.write_bytes(gzip.compress(data, mtime=0))
            manifest.append(
                dict(
                    source=str(source),
                    archive=str(target.relative_to(doc)),
                    bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            )
    (doc / f"{args.label}_watcher_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

module = "models.autoports.ornith_ai_ornith_1_5_9b.tests."
run(f"{args.label}_stack", ["timeout", "600", sys.executable, "-m", module + "multichip_stack_probe"], env)
for layer in (0, 3):
    run(
        f"{args.label}_timing_layer{layer}",
        [
            "timeout",
            "600",
            sys.executable,
            "-m",
            module + "multichip_probe",
            "--layer",
            str(layer),
            "--length",
            "2048",
            "--prefill-iterations",
            "16",
        ],
        env,
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serial alternating-order reader timings and separate signposted profiles."""
import os
import subprocess
import sys
from pathlib import Path

doc = Path(__file__).resolve().parent
env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR="/home/hous/dev/ornith-1.5-9b/state/torch-cache", OMP_NUM_THREADS="8")
assert not env.get("TT_METAL_WATCHER"), "Reader profiler runs must exclude watcher"
module = "models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_reader_probe"
for profile in (False, True):
    for layer in (0, 3):
        name = f"reader_packed_{'profile' if profile else 'alternating'}_layer{layer}"
        command = [sys.executable]
        if profile:
            command += [
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
                str(doc / "tracy/reader_raw" / f"layer{layer}"),
            ]
        command += [
            "-m",
            module,
            "--layer",
            str(layer),
            "--roles",
            "gate_up",
            "--cores",
            "32",
            "--block",
            "4",
            "--output",
            str(doc / f"{name}.json"),
        ]
        if profile:
            command += ["--profile", "--windows", "3", "--iterations", "4"]
        subprocess.run(
            [sys.executable, str(doc / "record_run.py"), name, "timeout", "300", *command], env=env, check=True
        )

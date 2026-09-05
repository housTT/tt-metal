# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run an explicit candidate list serially; stop on the first failed gate."""
import json
import os
import subprocess
import sys
from pathlib import Path

doc = Path(__file__).resolve().parent
env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR="/home/hous/dev/ornith-1.5-9b/state/torch-cache", OMP_NUM_THREADS="8")
for case in json.loads(Path(sys.argv[1]).read_text()):
    name, arguments = case["name"], case["arguments"]
    if (doc / "logs" / f"{name}.provenance.json").exists():
        previous = json.loads((doc / "logs" / f"{name}.provenance.json").read_text())
        if previous.get("returncode") == 0:
            continue
        raise RuntimeError(f"{name} failed or remains running; inspect before retrying with a new name")
    command = [
        sys.executable,
        str(doc / "record_run.py"),
        name,
        "timeout",
        "240",
        sys.executable,
        "-m",
        "models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe",
        "--length",
        "2048",
        *arguments,
    ]
    subprocess.run(command, env=env, check=True)

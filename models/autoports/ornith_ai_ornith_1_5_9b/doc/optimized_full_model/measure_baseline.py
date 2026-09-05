# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Measure exact completed-stage implementation, restoring selected files even on failure."""

import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
repo = root.parents[2]
paths = [root / "tt/model.py", root / "tt/generator.py"]
selected = {p: p.read_bytes() for p in paths}
backup = Path("/home/hous/dev/ornith-1.5-9b/state/optimized-full-model-selected-source")
backup.mkdir(exist_ok=True)
for p, data in selected.items():
    (backup / p.name).write_bytes(data)
try:
    for p in paths:
        original = subprocess.check_output(["git", "show", f"2e4b8f828c:{p.relative_to(repo)}"], cwd=repo)
        p.write_bytes(original)
    command = [
        "python_env/bin/python",
        "-m",
        "models.autoports.ornith_ai_ornith_1_5_9b.doc.optimized_full_model.record_run",
        "baseline_repeated_v2",
        "timeout",
        "180",
        "python_env/bin/python",
        "-m",
        "models.autoports.ornith_ai_ornith_1_5_9b.doc.optimized_full_model.run_checks",
        "perf",
        "--warm-samples",
        "5",
        "--output",
        str(root / "doc/optimized_full_model/baseline_repeated_v2.json"),
    ]
    status = subprocess.run(command, cwd=repo, env=os.environ.copy()).returncode
finally:
    for p, data in selected.items():
        p.write_bytes(data)
    assert all(p.read_bytes() == data for p, data in selected.items())
raise SystemExit(status)

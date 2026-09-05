# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run a finite, serialized candidate plan, preserving each source snapshot."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("plan", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
base = json.loads((root / "doc/optimized_decoder/best_geometry_config.json").read_text())
policy = dict(
    attention="bfloat4_b", attention_fidelity="LoFi", mlp_gate_up="bfloat4_b", mlp_down="bfloat4_b", mlp_fidelity="LoFi"
)
results = []
for row in json.loads(args.plan.read_text()):
    config = {**base, **row.get("config", {})}
    env = os.environ.copy()
    env.update(
        TORCHINDUCTOR_CACHE_DIR="/home/hous/dev/ornith-1.5-9b/state/torch-cache",
        OMP_NUM_THREADS="8",
        ORNITH_WEIGHTS="real",
        ORNITH_OPT_POLICY=json.dumps({**policy, **row.get("policy", {})}),
        ORNITH_OPT_CONFIG=json.dumps(config),
        ORNITH_OPT_VARIANT="default",
    )
    env.update(row.get("env", {}))
    command = [
        sys.executable,
        str(root / "doc/optimized_decoder/record_run.py"),
        row["name"],
        sys.executable,
        "-m",
        "pytest",
        str(root / row.get("test", "tests/test_optimization_experiments.py")),
        "-v",
        "-s",
        "--maxfail=1",
    ]
    if row.get("layer"):
        command.extend(["-k", row["layer"]])
    result = subprocess.run(command, env=env)
    results.append(dict(name=row["name"], returncode=result.returncode))
    print(json.dumps(results[-1]), flush=True)
args.plan.with_suffix(".results.json").write_text(json.dumps(results, indent=2) + "\n")

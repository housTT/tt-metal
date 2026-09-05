# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serialize small profiler captures for the selected reader geometries."""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--label", default="reader_confirmation")
parser.add_argument("--kind", choices=("linear_attention", "full_attention"))
parser.add_argument("--roles")
parser.add_argument("--op-support-count", type=int)
parser.add_argument("--per-core-n", help="Integer or JSON role-to-per_core_N map for a controlled geometry probe")
parser.add_argument("--no-account", action="store_true")
parser.add_argument("--drain", action="store_true")
args = parser.parse_args()

doc = Path(__file__).resolve().parent
root = doc.parents[1]
config = json.loads((doc / "best_geometry_config.json").read_text())
policy = dict(
    attention="bfloat4_b", mlp_gate_up="bfloat4_b", mlp_down="bfloat4_b", attention_fidelity="LoFi", mlp_fidelity="LoFi"
)
for row in json.loads((doc / "reader_confirmation_plan.json").read_text()):
    kind = row["layer"]
    if args.kind and args.kind != kind:
        continue
    capture_root = doc / "tracy" / kind / "raw" / args.label
    assert not capture_root.exists(), f"Refusing to overwrite an existing capture: {capture_root}"
    assert not (doc / f"{args.label}_{kind}.json").exists(), "Measurement evidence already exists"
    env = os.environ.copy()
    assert not env.get("TT_METAL_WATCHER")
    env.update(
        TORCHINDUCTOR_CACHE_DIR="/home/hous/dev/ornith-1.5-9b/state/torch-cache",
        OMP_NUM_THREADS="8",
        ORNITH_WEIGHTS="real",
        ORNITH_OPT_POLICY=json.dumps(policy),
        ORNITH_OPT_CONFIG=json.dumps(config),
        ORNITH_OPT_VARIANT="default",
    )
    env.update(row["env"])
    env.update(ORNITH_READER_PROFILE="1", ORNITH_SWEEP_OUTPUT=f"{args.label}_{kind}.json")
    if args.drain:
        env["ORNITH_READER_DRAIN"] = "1"
    if args.roles:
        env["ORNITH_SWEEP_ROLES"] = args.roles
    if args.per_core_n:
        env["ORNITH_READER_PER_CORE_N"] = str(args.per_core_n)
    command = [
        sys.executable,
        str(doc / "record_run.py"),
        f"profile_{args.label}_{kind}",
        sys.executable,
        "-m",
        "tracy",
        "-r",
        "-p",
        "-v",
        "--check-exit-code",
        "--web-app-port",
        "18940",
        "-o",
        str(capture_root),
        "-m",
        "pytest",
        str(root / "tests/test_projection_geometry.py"),
        "-k",
        kind,
        "-v",
        "-s",
    ]
    if args.op_support_count:
        at = command.index("--check-exit-code")
        command[at:at] = ["--op-support-count", str(args.op_support_count)]
    provenance = doc / "logs" / f"profile_{args.label}_{kind}.collector.json"
    assert not provenance.exists(), provenance
    sources = [
        Path(__file__),
        doc / "account_readers.py",
        Path("tools/tracy/__main__.py"),
        Path("tools/tracy/process_ops_logs.py"),
    ]
    archive = provenance.with_suffix(".sources.json.gz")
    archive.write_bytes(gzip.compress(json.dumps({str(path): path.read_text() for path in sources}).encode(), mtime=0))
    provenance.write_text(
        json.dumps(
            {
                "command": command,
                "effective_child_environment": {
                    "TT_METAL_DEVICE_PROFILER": "1",
                    "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT": str(args.op_support_count or 1000),
                    "ORNITH_READER_DRAIN": env.get("ORNITH_READER_DRAIN"),
                },
                "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                "archive": str(archive),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    subprocess.run(command, env=env, check=True)
    if not args.no_account:
        subprocess.run([sys.executable, str(doc / "account_readers.py"), kind, "--label", args.label], check=True)

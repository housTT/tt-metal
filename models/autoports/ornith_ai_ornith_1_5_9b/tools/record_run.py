# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Record a serialized stage command with source hashes and exact provenance."""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
if not args.command:
    parser.error("a command is required")
log = root / "doc/functional_decoder/logs" / f"{args.name}.log"
provenance = log.with_suffix(".provenance.json")
if log.exists() or provenance.exists():
    parser.error(f"evidence already exists: {args.name}")
files = sorted(p for folder in ("tt", "reference", "tests") for p in (root / folder).rglob("*.py"))
record = {
    "command": args.command,
    "cwd": str(Path.cwd()),
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "environment": {
        key: os.environ.get(key)
        for key in (
            "TORCHINDUCTOR_CACHE_DIR",
            "OMP_NUM_THREADS",
            "ORNITH_WEIGHTS",
            "ORNITH_PERF_DECODE_ITERS",
            "TT_METAL_WATCHER",
            "TT_METAL_LOGS_PATH",
            "TT_METAL_DEVICE_PROFILER",
            "TT_METAL_HOME",
            "TT_METAL_CACHE",
            "VIRTUAL_ENV",
        )
    },
    "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
}
provenance.write_text(json.dumps(record, indent=2) + "\n")
with log.open("w") as stream:
    result = subprocess.run(args.command, stdout=stream, stderr=subprocess.STDOUT)
record.update(
    returncode=result.returncode,
    ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
)
provenance.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"name": args.name, "returncode": result.returncode, "log": str(log)}))
sys.exit(result.returncode)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Record a serialized stage command with source hashes and exact provenance."""

import argparse
import datetime
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
if not args.command:
    parser.error("a command is required")
log = root / "doc/multichip_decoder/logs" / f"{args.name}.log"
provenance = log.with_suffix(".provenance.json")
if log.exists() or provenance.exists():
    parser.error(f"evidence already exists: {args.name}")
files = sorted(p for folder in ("tt", "reference", "tests") for p in (root / folder).rglob("*.py"))
files += sorted((root / "doc/multichip_decoder").glob("*.py"))
files += [root / "doc/context_contract.json", root / "doc/multichip_decoder/memory_capacity_plan.json"]
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
            "ORNITH_OPT_POLICY",
            "ORNITH_OPT_INPUTS",
            "ORNITH_OPT_CONFIG",
            "ORNITH_OPT_VARIANT",
            "ORNITH_STATE_BLOCK",
            "ORNITH_STATE_GRID",
            "ORNITH_CHUNK_FIDELITY",
            "ORNITH_CHUNK_MEMORY",
            "ORNITH_CHUNK_PHYSICAL",
            "ORNITH_STATE_SUBBLOCK",
            "ORNITH_STATE_MEMORY",
            "ORNITH_STATE_INTERMEDIATES",
            "ORNITH_STATE_FIDELITY",
            "ORNITH_SWEEP_ROLES",
            "ORNITH_SWEEP_EXTRA",
            "ORNITH_READER_CONFIRM",
            "ORNITH_READER_PROFILE",
            "ORNITH_FINAL_PACKED_READERS",
            "ORNITH_FINAL_PREFILL_M",
            "ORNITH_READER_DRAIN",
            "ORNITH_READER_PER_CORE_N",
            "ORNITH_SWEEP_OUTPUT",
            "ORNITH_TOPOLOGY",
            "ORNITH_CONV_CHUNK",
            "ORNITH_HEAD_NORM_CORES",
            "ORNITH_ACTIVATION_ROLES",
            "ORNITH_SDPA_GRID",
            "ORNITH_SDPA_CHUNK",
            "ORNITH_KV_DTYPE",
            "ORNITH_PREFILL_GRID",
            "ORNITH_PREFILL_L1_ROLES",
            "ORNITH_PREFILL_OUT_M",
            "ORNITH_PREFILL_OUT_N",
            "ORNITH_PREFILL_SUB_H",
            "ORNITH_PREFILL_SUB_W",
            "ORNITH_PREFILL_PAD_N",
            "ORNITH_PREFILL_BLOCK",
            "ORNITH_PERF_DECODE_ITERS",
            "ORNITH_PROFILE_BASELINE",
            "ORNITH_TRACE_ALLOCATION_REPRO",
            "TT_METAL_WATCHER",
            "TT_METAL_WATCHER_DISABLE_ETH",
            "TT_METAL_WATCHER_NOINLINE",
            "TT_METAL_WATCHER_APPEND",
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
archive = log.with_suffix(".sources.json.gz")
archive.write_bytes(
    gzip.compress(json.dumps({str(p.relative_to(root)): p.read_text() for p in files}).encode(), mtime=0)
)
record["source_archive"] = str(archive)
record["source_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
provenance.write_text(json.dumps(record, indent=2) + "\n")
with log.open("w") as stream:
    result = subprocess.run(args.command, stdout=stream, stderr=subprocess.STDOUT)
log_archive = log.with_suffix(".log.gz")
log_archive.write_bytes(gzip.compress(log.read_bytes(), mtime=0))
record.update(
    log_archive=str(log_archive),
    log_archive_sha256=hashlib.sha256(log_archive.read_bytes()).hexdigest(),
    returncode=result.returncode,
    ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
)
provenance.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"name": args.name, "returncode": result.returncode, "log": str(log)}))
sys.exit(result.returncode)

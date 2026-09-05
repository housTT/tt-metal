# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Immutable command log and source snapshot for serialized full-model runs."""

import argparse
import datetime
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("command required")
    doc = Path(__file__).resolve().parent
    model = doc.parents[1]
    repo = model.parents[2]
    log = doc / "logs" / f"{args.name}.log"
    meta = log.with_suffix(".provenance.json")
    sources = log.with_suffix(".sources.json.gz")
    if any(p.exists() for p in (log, meta, sources)):
        parser.error("run label already exists")
    files = list((model / "tt").glob("*.py")) + list(doc.glob("*.py"))
    files += list((doc.parent / "full_model").glob("*.py"))
    files += list((repo / "models/common/sampling").glob("*.py"))
    files += list((repo / "models/common/modules/lm_head").glob("*.py"))
    files += [repo / "models/common/modules/lazy_weight.py"]
    files += list((repo / "models/common/readiness_check").glob("*.py"))
    files += [
        repo
        / "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp",
        repo
        / "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp",
        repo
        / "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp",
        repo / "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/layernorm_dataflow_utils.h",
        repo / "tests/ttnn/unit_tests/operations/fused/test_rms_norm_sharded.py",
        repo / "tools/tracy/process_ops_logs.py",
        repo / "tests/ttnn/tracy/test_process_ops_logs.py",
    ]
    payload = {str(p.relative_to(repo)): p.read_text() for p in files}
    with gzip.open(sources, "wt") as f:
        json.dump(payload, f)
    record = dict(
        command=args.command,
        cwd=str(Path.cwd()),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        environment={
            key: os.environ.get(key)
            for key in (
                "TORCHINDUCTOR_CACHE_DIR",
                "OMP_NUM_THREADS",
                "HF_HUB_OFFLINE",
                "TT_METAL_WATCHER",
                "TT_METAL_WATCHER_DISABLE_ETH",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_TRACEBACKS",
            )
        },
        sources_sha256={path: hashlib.sha256(text.encode()).hexdigest() for path, text in payload.items()},
        snapshot_sha256=sha(sources),
    )
    record["binaries_sha256"] = {
        str(p): sha(p) for p in (repo / "build/lib/_ttnncpp.so", repo / "build/lib/libtt_metal.so") if p.exists()
    }
    meta.write_text(json.dumps(record, indent=2) + "\n")
    with log.open("w") as out:
        status = subprocess.run(args.command, stdout=out, stderr=subprocess.STDOUT).returncode
    record.update(
        exit_code=status, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), log_sha256=sha(log)
    )
    meta.write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()

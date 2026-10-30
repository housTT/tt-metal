#!/usr/bin/env python3
"""Write a provenance sidecar for a readiness run: commit, chip, command, log and artifact paths."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chip", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--command", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--artifact", action="append", default=[])
    ap.add_argument("--note", default=None)
    ap.add_argument("--exit-code", type=int, default=None)
    ap.add_argument("--started-utc", default=None)
    ap.add_argument("--ended-utc", default=None)
    args = ap.parse_args()
    root = Path("/home/hous/dev/tt-metal-wt-qwen3")
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--short", "--", "models/autoports/qwen_qwen3_8b"], text=True
    ).strip()
    payload = {}
    if args.output.is_file():
        payload = json.loads(args.output.read_text())
    payload.setdefault("checkout_git_sha", sha)
    payload.setdefault("visible_devices", str(args.chip))
    payload.setdefault(
        "mesh", "one Blackhole P300 chip, 1x1 mesh, p150 mesh graph descriptor (runner argument --mesh-device N150)"
    )
    payload.setdefault("runs", {})
    payload["runs"][args.name] = {
        "checkout_git_sha": sha,
        "dirty_autoport_files_at_record_time": dirty.splitlines(),
        "chip": str(args.chip),
        "command": args.command,
        "log": args.log,
        "artifacts": args.artifact,
        "exit_code": args.exit_code,
        "started_utc": args.started_utc,
        "ended_utc": args.ended_utc,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": args.note,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

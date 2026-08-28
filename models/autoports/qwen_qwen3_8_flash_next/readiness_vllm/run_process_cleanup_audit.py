#!/usr/bin/env python3
"""Record that graceful vLLM shutdown released processes and TT devices."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def matching_processes(needles: tuple[str, ...]) -> list[dict[str, object]]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command and any(needle in command for needle in needles):
            matches.append({"pid": int(entry.name), "command": command})
    return sorted(matches, key=lambda row: int(row["pid"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    processes = matching_processes(("vllm.entrypoints.openai.api_server", "VLLM::EngineCore", "run_vllm_server.py"))
    holders = {}
    for device in ("/dev/tenstorrent/0", "/dev/tenstorrent/1"):
        result = subprocess.run(["fuser", device], text=True, capture_output=True, check=False)
        holders[device] = (result.stdout + result.stderr).strip().split()
    snapshot = json.loads(subprocess.run(["tt-smi", "-s"], text=True, capture_output=True, check=True).stdout)
    devices = [
        {
            "bus_id": row["board_info"]["bus_id"],
            "board_type": row["board_info"]["board_type"],
            "dram_status": row["board_info"]["dram_status"],
            "asic_temperature_c": float(row["telemetry"]["asic_temperature"]),
            "heartbeat": row["telemetry"]["heartbeat"],
        }
        for row in snapshot["device_info"][:2]
    ]
    log_text = args.server_log.read_text(errors="replace")
    artifact = {
        "shutdown_log": str(args.server_log),
        "shutdown_log_has_device_close": "Closing user mode device drivers" in log_text
        or "Closing user mode device driver" in log_text,
        "matching_processes": processes,
        "device_holders": holders,
        "devices": devices,
    }
    artifact["verdict"] = (
        "pass"
        if not processes and not any(holders.values()) and all(device["dram_status"] for device in devices)
        else "fail"
    )
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    if artifact["verdict"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

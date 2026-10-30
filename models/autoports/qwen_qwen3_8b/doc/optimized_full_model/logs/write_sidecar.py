"""Append one readiness run's provenance and headline numbers to the stage-7 readiness sidecar JSON."""
import json
import re
import subprocess
import sys
from pathlib import Path

sidecar, name, log, rc, started, ended = sys.argv[1:7]
text = Path(log).read_text(errors="replace")
entry = {"log": log, "exit": int(rc), "started_utc": started, "ended_utc": ended}
for key, pattern in (
    ("top1", r"top1=([0-9.]+)"),
    ("top5", r"top5=([0-9.]+)"),
    ("top100", r"top100=([0-9.]+)"),
    ("ttft_ms", r"TTFT=([0-9.]+)ms"),
    ("decode_tps_per_user", r"decode=([0-9.]+) t/s/u"),
    ("e2e_tps_per_user", r"e2e=([0-9.]+) t/s/u"),
    ("prepare_s", r"prepare done in ([0-9.]+) s"),
    ("weights_on_device_s", r"model weights on device in ([0-9.]+) s"),
    ("status", r"status=(\w+)"),
):
    found = re.findall(pattern, text)
    if found:
        value = found[-1]
        entry[key] = value if key == "status" else float(value)
path = Path(sidecar)
data = (
    json.loads(path.read_text())
    if path.exists()
    else {
        "checkout_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "visible_devices": "2",
        "chip": 2,
        "mesh": "N150 (1x1, p150 mesh graph descriptor)",
        "runs": {},
    }
)
data["checkout_git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
data[
    "mesh"
] = "one Blackhole P300 chip, TT_VISIBLE_DEVICES=2, 1x1 mesh, p150 mesh graph descriptor (runner argument --mesh-device N150)"
data["runs"][name] = entry
path.write_text(json.dumps(data, indent=2) + "\n")
print(name, json.dumps(entry))

"""Append one readiness run's provenance, headline numbers and runtime counters to a stage-8 sidecar JSON.

Usage: write_sidecar.py <sidecar.json> <run name> <log> <rc> <started> <ended> [<config id>] [<config path>]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sidecar, name, log, rc, started, ended = sys.argv[1:7]
config_id = sys.argv[7] if len(sys.argv) > 7 else None
config_path = sys.argv[8] if len(sys.argv) > 8 else None
chip = sys.argv[9] if len(sys.argv) > 9 else "2"
reference = sys.argv[10] if len(sys.argv) > 10 else "bringup/references/qwen3_8b_aime24_chat.refpt"
regime = sys.argv[11] if len(sys.argv) > 11 else "serialized: the only device job on the host during this run"
text = Path(log).read_text(errors="replace")
entry = {
    "log": log,
    "exit": int(rc),
    "started_utc": started,
    "ended_utc": ended,
    "chip": int(chip),
    "reference": reference,
    "timing_regime": regime,
}
if config_id:
    entry["config_id"] = config_id
    entry["config_path"] = config_path
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
counters = re.findall(r"runtime counters at teardown: (\{.*\})", text)
if counters:
    entry["runtime_counters"] = json.loads(counters[-1])
precision = re.findall(r"precision config (\S+) from (\S+?):", text)
if precision:
    entry["precision_config_logged"] = {"config_id": precision[-1][0], "source": precision[-1][1]}
path = Path(sidecar)
data = (
    json.loads(path.read_text())
    if path.exists()
    else {
        "visible_devices": chip,
        "chip": int(chip),
        "runs": {},
    }
)
data["checkout_git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
data["checkout_dirty"] = bool(
    subprocess.check_output(
        ["git", "status", "--porcelain", "--", "models/autoports/qwen_qwen3_8b/tt"], text=True
    ).strip()
)
data[
    "mesh"
] = f"one Blackhole P300 chip, TT_VISIBLE_DEVICES={chip}, 1x1 mesh, p150 mesh graph descriptor (runner argument --mesh-device N150)"
data["runs"][name] = entry
path.write_text(json.dumps(data, indent=2) + "\n")
print(name, json.dumps(entry))

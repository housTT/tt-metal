import json
import sys
from pathlib import Path

E = Path(sys.argv[1])
MIB = 2**20
path = E / "memory_handoff_recheck.json"
recheck = json.loads(path.read_text())
peaks = {}
for mesh in (2, 4):
    for layer in (0, 3):
        data = json.loads((E / f"inventory/replica_peaks/m{mesh}_layer{layer}.json").read_text())
        peaks[f"m{mesh}_layer{layer}"] = dict(
            peak_before_cleanup_mib=round(data["peak_before_cleanup_bytes"] / MIB, 1),
            per_replica_mib=[dict(before_cleanup=round(r["before_cleanup"] / MIB, 1),
                                  after_cleanup=round(r["after_cleanup"] / MIB, 1))
                             for r in data["replica_weight_memory"]])
for layer in (0, 3):
    memory = json.loads((E / f"inventory/m1_layer{layer}.json").read_text())["weight_memory"]
    peaks[f"m1_layer{layer}"] = dict(peak_before_cleanup_mib=round(memory["before_cleanup"]["dram"]["allocated_bytes"] / MIB, 1),
                                     after_cleanup_mib=round(memory["after_cleanup"]["dram"]["allocated_bytes"] / MIB, 1))
recheck["construction_transient_peaks"] = dict(
    method=("p150: weight_memory before_cleanup inside OptimizedDecoder.from_state_dict (inventory/m1_layer*.json). "
            "p150x2/p150x4: inventory/replica_peaks/replica_peaks.py wraps MultichipDecoder._replica and records each "
            "replica's before/after-cleanup allocator total (device-wide, so later replicas include earlier ones)."),
    peaks=peaks,
    full_model_note=("A layer-by-layer build adds at most this per-layer transient on top of the resident earlier "
                     "layers; replicas and setup copies are released before the next layer starts."))
path.write_text(json.dumps(recheck, indent=1) + "\n")
print({k: v["peak_before_cleanup_mib"] for k, v in peaks.items()})

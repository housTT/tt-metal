# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Summarize final-source validation, preserving links to complete runner evidence."""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--suffix", default="")
args = parser.parse_args()
model = root.parents[1]
runtime_hash = hashlib.sha256((model / "tt/fused_decoder.py").read_bytes()).hexdigest()


def read_log(name):
    path = root / "logs" / f"{name}.log"
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(".log.gz").read_bytes())


runs = []
for stem in ("final_short", "final_long", "watcher_final"):
    name = stem + args.suffix
    provenance = json.loads((root / "logs" / f"{name}.provenance.json").read_text())
    data = read_log(name)
    assert provenance["returncode"] == 0
    assert provenance["source_sha256"]["tt/fused_decoder.py"] == runtime_hash
    assert hashlib.sha256(data).hexdigest() == provenance["log_sha256"]
    summary = re.findall(r"=+ (\d+ passed[^\n]+) =+", data.decode())[-1]
    runs.append(dict(name=name, summary=summary, provenance=f"logs/{name}.provenance.json"))

short = read_log("final_short" + args.suffix).decode()
paired = [json.loads(line.split(" ", 1)[1]) for line in short.splitlines() if line.startswith("FUSION_PAIR ")]
probes = [json.loads(line.split(" ", 1)[1]) for line in short.splitlines() if line.startswith("LINEAR_CORE_PROBE ")]
assert len(paired) == 2 and len(probes) == 6
core_summary = []
for probe in probes:
    cores = [core for step in probe["steps"] for core in step["cores"]]
    states = [step["state"]["rec"] for step in probe["steps"]]
    outputs = [step["output"] for step in probe["steps"]]
    core_summary.append(
        dict(
            logical_prefill=probe["seq_len"],
            decode_steps=len(probe["steps"]) - 1,
            minimum_raw_core_pcc=min(item["pcc"] for item in cores),
            core_norm_ratio_range=[
                min(item["norm_ratio"] for item in cores),
                max(item["norm_ratio"] for item in cores),
            ],
            minimum_recurrent_state_pcc=min(item["pcc"] for item in states),
            minimum_layer_output_pcc=min(item["pcc"] for item in outputs),
        )
    )
result = dict(
    runtime_sha256=runtime_hash,
    runs=runs,
    paired_functional_comparison=paired,
    raw_linear_core_probes=core_summary,
    long_context_metric_lines=[
        line.split(" - ", 1)[-1]
        for line in read_log("final_long" + args.suffix).decode().splitlines()
        if "PCC=" in line or "PCC(" in line or "long-context PCC layer=" in line
    ],
)
(root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(runs=runs, raw_linear_core_probes=core_summary), indent=2))

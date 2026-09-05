# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Rebuild candidate inventory from immutable run records."""

import csv
import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
for provenance in sorted((root / "logs").glob("*.provenance.json")):
    info = json.loads(provenance.read_text())
    stem = provenance.name.removesuffix(".provenance.json")
    path = provenance.with_name(stem + ".log")
    if path.exists():
        lines = path.read_text(errors="replace").splitlines()
    elif path.with_suffix(".log.gz").exists():
        lines = gzip.decompress(path.with_suffix(".log.gz").read_bytes()).decode(errors="replace").splitlines()
    else:
        continue
    for line in lines:
        if not line.startswith("OPTIMIZATION_PAIR "):
            continue
        row = json.loads(line.split(" ", 1)[1])
        row.update(run=stem, returncode=info.get("returncode"), provenance=str(provenance.relative_to(root)))
        row["experiment_environment"] = info.get("environment", {})
        row.setdefault("variant", info.get("environment", {}).get("ORNITH_OPT_VARIANT") or "default")
        rows.append(row)
payload = (json.dumps(rows, separators=(",", ":")) + "\n").encode()
(root / "candidate_measurements.json").write_bytes(payload)
(root / "candidate_measurements.json.gz").write_bytes(gzip.compress(payload, mtime=0))
with (root / "candidate_measurements.csv").open("w", newline="") as f:
    fields = [
        "run",
        "layer",
        "variant",
        "returncode",
        "prefill_ms",
        "decode_ms",
        "hf_prefill_pcc",
        "hf_decode_pcc",
        "hf_stress_pcc",
        "policy",
        "config",
        "provenance",
    ]
    writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        out = {key: row[key] for key in ("run", "layer", "variant", "returncode", "policy", "config", "provenance")}
        timing = row["timings"]["optimized"]
        out.update(prefill_ms=timing["prefill_median_ms"], decode_ms=timing["decode_median_ms"])
        out.update({"hf_" + key + "_pcc": value for key, value in row["hf_pcc"].items()})
        writer.writerow(out)
print(f"Recorded {len(rows)} paired candidates")

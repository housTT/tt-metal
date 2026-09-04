# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Collect paired experiment data without mixing standard perf and paired regimes."""

import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
paths = sorted((root / "logs").glob("*.log"))
paths += sorted(p for p in (root / "logs").glob("*.log.gz") if not p.with_suffix("").exists())
for path in paths:
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    for line in data.decode().splitlines():
        if not line.startswith("FUSION_PAIR "):
            continue
        result = json.loads(line.removeprefix("FUSION_PAIR "))
        archive = path.with_suffix(".log.gz") if path.suffix == ".log" else path
        result.update(
            log=str((archive if archive.exists() else path).relative_to(root)),
            log_sha256=hashlib.sha256(data).hexdigest(),
        )
        rows.append(result)
(root / "candidate_measurements.json").write_text(json.dumps(rows, indent=2) + "\n")
lines = [
    "# Paired candidate measurements",
    "",
    "Real checkpoint weights; single Blackhole chip on P300c boards; B=1; prefill 2048, decode position 2048. Five measured windows after two warm windows; decode windows contain 32 replays. These are wall timings with synchronization. Experiments include rejected candidates and do not by themselves select the final runtime.",
    "",
    "| Log | Layer | Prefill baseline→candidate ms | Trace baseline→candidate ms | Minimum output PCC |",
    "| --- | --- | --- | --- | --- |",
]
for row in rows:
    before, after = row["timings"]["functional"], row["timings"]["fused"]
    lines.append(
        f"| [{row['log']}]({row['log']}) | {row['layer']} | {before['prefill_median_ms']:.6f}→{after['prefill_median_ms']:.6f} | {before['decode_median_ms']:.6f}→{after['decode_median_ms']:.6f} | {min(row['pcc'].values()):.8f} |"
    )
(root / "candidate_measurements.md").write_text("\n".join(lines) + "\n")
print(f"{len(rows)} paired rows")

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Rebuild compact measurement tables from immutable recorded-run artifacts."""

import csv
import gzip
import json
from pathlib import Path

doc = Path(__file__).resolve().parent
measurements = []
for provenance in sorted((doc / "logs").glob("*.provenance.json")):
    meta = json.loads(provenance.read_text())
    command = meta["command"]
    if not any("tests.multichip_probe" in part for part in command):
        continue
    archive = doc / "logs" / (provenance.name.removesuffix(".provenance.json") + ".log.gz")
    if not archive.exists():
        continue
    records = []
    for line in gzip.decompress(archive.read_bytes()).decode().splitlines():
        if line.startswith("{"):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    layer = int(command[command.index("--layer") + 1]) if "--layer" in command else 0
    length = int(command[command.index("--length") + 1]) if "--length" in command else 128
    baseline = {key: value for row in records if row.get("name") == "baseline" for key, value in row.items()}
    parallel = {key: value for row in records if row.get("name") == "tp4" for key, value in row.items()}
    pcc = {row["mode"]: min(row["pcc"]) for row in records if "mode" in row and "pcc" in row}
    for mode in ("prefill", "decode"):
        if f"{mode}_ms" not in baseline or f"{mode}_ms" not in parallel:
            continue
        single, multi = baseline[f"{mode}_ms"], parallel[f"{mode}_ms"]
        measurements.append(
            dict(
                run=provenance.name.removesuffix(".provenance.json"),
                layer=layer,
                logical_prefill=length,
                mode=mode,
                baseline_ms=single,
                tp4_ms=multi,
                speedup=single / multi,
                efficiency_percent=100 * single / multi / 4,
                baseline_pcc=pcc.get(mode),
                trace_exact=parallel.get("trace_exact"),
                returncode=meta["returncode"],
                command=json.dumps(command),
                source_archive_sha256=meta["source_archive_sha256"],
            )
        )
if measurements:
    with (doc / "candidate_measurements.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)
    lines = [
        "# Recorded paired measurements",
        "",
        "Historical candidates, including rejected policies. Exit0 proves only the probe's own gates;",
        "final acceptance also requires batch/cache/native-context/stress/watcher checks.",
        "",
        "| Run | Layer | Prefix | Mode | Single ms | TP4 ms | Speedup | Efficiency | PCC | Exit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in measurements:
        pcc = f"{row['baseline_pcc']:.8f}" if row["baseline_pcc"] is not None else "incomplete"
        lines.append(
            f"| {row['run']} | {row['layer']} | {row['logical_prefill']} | {row['mode']} | "
            f"{row['baseline_ms']:.4f} | {row['tp4_ms']:.4f} | {row['speedup']:.3f}× | "
            f"{row['efficiency_percent']:.1f}% | {pcc} | {row['returncode']} |"
        )
    (doc / "candidate_measurements.md").write_text("\n".join(lines) + "\n")

lines = [
    "# Local projection geometry search",
    "",
    "Real decoder-produced TP-local inputs; warmed trace includes the row projection's collective.",
    "DRAM-sharded family uses reader1: reader2/3 are blocked by the verified mesh API call.",
    "PCC is against the original local geometry, not a replacement for whole-layer HF gates.",
    "",
    "| Artifact | Family | Weight dtype | Layer | Role | K × N per device | Input/compute cores | K-block tiles | Median μs | Min PCC | Exact original | Exact replay |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
]
for path in sorted(doc.glob("geometry_*layer*.json")):
    for row in json.loads(path.read_text()):
        lines.append(
            f"| {path.name} | {row.get('family', 'dram')} | {row.get('weight_dtype', 'BFP4 (historical)')} | {row['layer']} | {row['role']} | {row['local_shape'][1]} × {row['local_shape'][2]} | "
            f"{row['cores']} | {row['block_w']} | {row['trace_us']:.3f} | {min(row['pcc']):.8f} | "
            f"{row['exact_reference']} | {row['trace_exact']} |"
        )
(doc / "geometry_search.md").write_text("\n".join(lines) + "\n")

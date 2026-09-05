# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Keep per-rank reader device times, physical storage rates and advice tables."""
import csv
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

doc = Path(__file__).resolve().parent
result = []
for layer in (0, 3):
    sources = list((doc / "tracy/reader_raw" / f"layer{layer}").rglob("ops_perf_results*.csv"))
    assert len(sources) == 1, sources
    data = sources[0].read_bytes()
    root = doc / "tracy/readers" / f"layer{layer}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "reader_ops.csv.gz").write_bytes(gzip.compress(data, mtime=0))
    reader = csv.DictReader(data.decode().splitlines())
    fields, rows = reader.fieldnames, list(reader)
    micro = json.loads((doc / f"reader_packed_profile_layer{layer}.json").read_text())
    for readers in (1, 2, 3):
        record = next(x for x in micro["results"] if x["readers"] == readers)
        start_name = f"PERF_READER_gate_up_R{readers}"
        start = next(i for i, r in enumerate(rows) if r["OP CODE"] == start_name)
        end = next(i for i, r in enumerate(rows) if r["OP CODE"] == start_name + "_END")
        window = rows[start + 1 : end]
        for rank in range(4):
            selected = [dict(r) for r in window if r["DEVICE ID"] == str(rank)]
            selected.sort(key=lambda r: float(r["DEVICE FW START CYCLE"] or 0))
            assert len(selected) == 4 and all("Matmul" in r["OP CODE"] for r in selected), (rank, selected)
            selected[0]["OP TO OP LATENCY [ns]"] = "0"
            selected[0]["OP TO OP LATENCY BR/NRISC START [ns]"] = "0"
            ops = root / f"r{readers}_device{rank}_window.csv"
            with ops.open("w") as out:
                writer = csv.DictWriter(out, fieldnames=fields)
                writer.writeheader()
                writer.writerows([rows[start], *selected, rows[end]])
            base = [
                "tt-perf-report",
                str(ops),
                "--start-signpost",
                start_name,
                "--end-signpost",
                start_name + "_END",
                "--tracing-mode",
                "--no-summary",
                "--no-color",
            ]
            commands = []
            for suffix, extra in (
                ("txt", []),
                ("console.log", ["--csv", str(root / f"r{readers}_device{rank}_report.csv")]),
            ):
                command = base + extra
                commands.append(command)
                with (root / f"r{readers}_device{rank}_report.{suffix}").open("w") as out:
                    subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True)
                if suffix == "txt":
                    text_report = root / f"r{readers}_device{rank}_report.txt"
                    text_report.write_text(
                        "\n".join(line.rstrip() for line in text_report.read_text().splitlines()).rstrip() + "\n"
                    )
            kernel_us = sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in selected) / 4000
            physical_gbs = record["physical_weight_bytes_per_chip"] / kernel_us / 1000
            result.append(
                dict(
                    layer=layer,
                    readers=readers,
                    rank=rank,
                    kernel_us=kernel_us,
                    interior_gap_us=sum(float(r["OP TO OP LATENCY [ns]"] or 0) for r in selected) / 4000,
                    same_run_host_us=record["profile_host_us"],
                    physical_weight_bytes=record["physical_weight_bytes_per_chip"],
                    weight_storage_gbs=physical_gbs,
                    weight_storage_fraction_of_512gbps_model=physical_gbs / 512,
                    per_reader_tile_row_bytes=record["per_reader_tile_row_bytes"],
                    input_cores=32,
                    k_block=4,
                    raw_sha256=hashlib.sha256(data).hexdigest(),
                    commands=commands,
                )
            )
(doc / "reader_packed_device_accounting.json").write_text(json.dumps(result, indent=2) + "\n")
columns = [k for k in result[0] if k != "commands"]
with (doc / "reader_packed_device_accounting.csv").open("w") as out:
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(result)
print(f"Rendered {len(result)} reader/rank records; rates use stored weight bytes, not measured traffic counters")

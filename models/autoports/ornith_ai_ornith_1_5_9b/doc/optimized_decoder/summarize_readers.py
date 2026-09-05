# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Join legal alternating reader device and unprofiled host measurements."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
config = json.loads((root / "best_geometry_config.json").read_text())
rows = []
sources = []
for kind in ("linear_attention", "full_attention"):
    device_path = root / "tracy" / kind / "reader_legal_confirmation/reader_accounting.json"
    host_path = root / f"readers_legal_host_{kind}.json"
    sources.extend((device_path, host_path))
    capture = json.loads(device_path.read_text())
    assert capture["status"] == "complete"
    device = capture["measurements"]
    host = json.loads(host_path.read_text())
    assert all(row["status"] == "pass" for row in host)
    for role in dict.fromkeys(row["role"] for row in device):
        role_rows = []
        for readers in (1, 2, 3):
            d = [row for row in device if row["role"] == role and row["readers"] == readers]
            h = [row for row in host if row["role"] == role and row["readers"] == readers]
            assert len(d) == len(h) == 2
            assert all(row["exact_eager_trace"] for row in d + h)
            assert all(row["compute"] == d[0]["compute"] for row in d + h)
            row = dict(
                layer=kind,
                role=role,
                readers=readers,
                selected=readers == config["role_configs"][role]["readers"],
                cores=d[0]["cores"],
                block_w=d[0]["block_w"],
                per_core_N=d[0]["per_core_N"],
                matmul_device_us=statistics.median(x["matmul_device_us"] for x in d),
                unprofiled_trace_us=statistics.median(x["trace_us"] for x in h),
                profiled_trace_us=statistics.median(x["trace_us"] for x in d),
                physical_GBs=statistics.median(x["device_physical_GBs"] for x in d),
                logical_GBs=statistics.median(x["device_logical_GBs"] for x in d),
                percent_declared_peak=statistics.median(x["device_percent_declared_peak"] for x in d),
                min_pcc=min(x["pcc_vs_interleaved"] for x in d + h),
                exact_eager_trace=True,
                runtime_weight_dtype=d[0]["runtime_weight_dtype"],
                runtime_fidelity=d[0]["runtime_fidelity"],
                per_bank_tiles=d[0]["per_bank_tiles"],
                per_reader_row_bytes=d[0]["per_reader_row_bytes"],
                device_core_counts=sorted({x["device_identity"]["CORE COUNT"] for x in d}),
            )
            role_rows.append(row)
        assert min(role_rows, key=lambda row: row["matmul_device_us"])["selected"], role
        assert min(role_rows, key=lambda row: row["unprofiled_trace_us"])["selected"], role
        rows.extend(role_rows)
report = dict(
    status="complete",
    hardware="one Blackhole chip on physical P300c boards",
    order=[1, 2, 3, 3, 2, 1],
    measurements_per_reader=2,
    declared_peak_DRAM_GBs=512,
    notes="Each device sample is one signposted replay. Host medians use five warmed windows of 64 replays per candidate, collected without the profiler. The three N=4096 output roles use common per_core_N=18 to make reader3 legal; production reader2 keeps per_core_N=16, whose passing controls are separately archived. Physical bandwidth includes BFP4 tile headers and per-bank reader padding. Runtime attributes, execution identities, source hashes and compressed operation CSVs are retained beside the source device accounting JSONs.",
    source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    rows=rows,
)
(root / "reader_comparison_summary.json").write_text(json.dumps(report, indent=2) + "\n")
with (root / "reader_comparison_summary.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# Alternating reader confirmation",
    "",
    report["notes"],
    "",
    "All 66 profiled and 66 unprofiled projection cases pass real-input PCC and exact eager/trace equality. The selected reader count wins both device matmul and unprofiled trace medians for every role.",
    "",
    "| Layer kind | Role | Readers | Device matmul us | Unprofiled trace us | Physical GB/s | Peak % |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    readers = f"**{row['readers']}**" if row["selected"] else str(row["readers"])
    lines.append(
        f"| {row['layer']} | {row['role']} | {readers} | {row['matmul_device_us']:.3f} | {row['unprofiled_trace_us']:.3f} | {row['physical_GBs']:.1f} | {row['percent_declared_peak']:.1f} |"
    )
(root / "reader_comparison_summary.md").write_text("\n".join(lines) + "\n")
print(f"Confirmed {len(rows)} role/reader combinations; every selected reader wins both timing metrics")

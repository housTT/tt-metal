# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Render immutable whole-layer run evidence, including failed candidates."""
import csv
import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
for source in sorted((root / "logs").glob("*.provenance.json")):
    provenance = json.loads(source.read_text())
    command = provenance["command"]
    if "models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe" not in command:
        continue
    name = source.name.removesuffix(".provenance.json")
    row = dict(
        name=name,
        returncode=provenance.get("returncode"),
        layer=command[command.index("--layer") + 1] if "--layer" in command else "0",
        prefill_ms=None,
        decode_ms=None,
        prefill_pcc=None,
        decode_pcc=None,
        trace_exact=False,
        trace_after_eager_exact=False,
    )
    log = root / "logs" / (name + ".log")
    contents = log.read_text() if log.exists() else gzip.decompress(log.with_suffix(".log.gz").read_bytes()).decode()
    for line in contents.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if value.get("name") == "tp4":
            for key in ("prefill_ms", "decode_ms", "trace_exact", "trace_after_eager_exact"):
                if key in value:
                    row[key] = value[key]
        if value.get("mode") in ("prefill", "decode") and "pcc" in value:
            row[value["mode"] + "_pcc"] = min(value["pcc"])
    row["probe_pass"] = row["returncode"] == 0 and row["trace_exact"] and row["decode_pcc"] is not None
    rows.append(row)
(root / "candidate_measurements.json").write_text(json.dumps(rows, indent=2) + "\n")
with (root / "candidate_measurements.csv").open("w") as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# Whole-layer candidate measurements",
    "",
    "Actual TP4 path, context2048 unless command provenance says otherwise. All decode latency is warmed trace replay. Probe pass records only this run’s gates; it does not establish stage eligibility. In particular, QKV4 candidates later fail the real batch32 trace gate; see AUTOFIX_qkv_trace.md. Failed rows remain failed even if a latency was emitted. The default-path release rows determine final numbers. Tests restore state outside timing; diagnostic single-chip comparison is excluded from these latencies.",
    "",
    "| Run | Exit | Layer | Prefill ms | Decode ms | Prefill PCC | Decode PCC | Probe pass |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
for row in rows:
    values = [f"[{row['name']}](logs/{row['name']}.provenance.json)", row["returncode"], row["layer"]]
    values += [
        "—" if row[k] is None else f"{row[k]:.8f}" for k in ("prefill_ms", "decode_ms", "prefill_pcc", "decode_pcc")
    ]
    values += [row["probe_pass"]]
    lines.append("| " + " | ".join(map(str, values)) + " |")
(root / "candidate_measurements.md").write_text("\n".join(lines) + "\n")
print(f"Rendered {len(rows)} whole-layer records")

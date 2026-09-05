# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Keep raw runtime policy, collective and trace-split evidence beside perf reports."""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

DOC = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    root = DOC / "tracy" / args.label
    for mode in ("decode", "prefill"):
        split = {}
        for rank in range(4):
            with (root / f"{mode}_device{rank}_window.csv").open() as stream:
                rows = [row for row in csv.DictReader(stream) if row["DEVICE ID"]]
            groups = defaultdict(list)
            has_traces = any(row["METAL TRACE ID"] for row in rows)
            for row in rows:
                groups[row["METAL TRACE ID"] or "eager"].append(row)
            split[str(rank)] = {}
            for trace, selected in groups.items():
                iterations = 4 if mode == "decode" else 1
                split[str(rank)][trace] = dict(
                    phase=(
                        ("prefill" if any("Embedding" in r["OP CODE"] for r in selected) else "sampling")
                        if mode == "prefill" and trace != "eager"
                        else (
                            ("request_boundaries" if has_traces else "prefill_and_sampling")
                            if mode == "prefill"
                            else "sampling_history"
                            if any("Sampling" in r["OP CODE"] for r in selected)
                            else "model"
                        )
                    ),
                    operations_per_iteration=len(selected) / iterations,
                    kernel_ms_per_iteration=sum(float(r["DEVICE KERNEL DURATION [ns]"] or 0) for r in selected)
                    / 1e6
                    / iterations,
                    interior_gap_ms_per_iteration=sum(float(r["OP TO OP LATENCY [ns]"] or 0) for r in selected)
                    / 1e6
                    / iterations,
                    op_codes=sorted({r["OP CODE"] for r in selected}),
                )
            if rank:
                continue
            programs = {}
            for row in rows:
                key = row["PROGRAM HASH"]
                if not key:
                    key = row["OP CODE"] + row["ATTRIBUTES"]
                if key not in programs:
                    programs[key] = dict(
                        opcode=row["OP CODE"],
                        program_hash=row["PROGRAM HASH"],
                        math_fidelity=row["MATH FIDELITY"],
                        attributes=row["ATTRIBUTES"],
                        tensors={
                            k: v for k, v in row.items() if v and (k.startswith("INPUT_") or k.startswith("OUTPUT_"))
                        },
                        compute_kernel_source=row["COMPUTE KERNEL SOURCE"],
                        data_movement_kernel_source=row["DATA MOVEMENT KERNEL SOURCE"],
                        trace_id=row["METAL TRACE ID"],
                        global_calls=[],
                        kernel_ns=[],
                    )
                programs[key]["global_calls"].append(row["GLOBAL CALL COUNT"])
                programs[key]["kernel_ns"].append(float(row["DEVICE KERNEL DURATION [ns]"] or 0))
            collectives = [
                p
                for p in programs.values()
                if any(name in p["opcode"] for name in ("AllGather", "ReduceScatter", "AllReduce"))
            ]
            matmuls = [p for p in programs.values() if "Matmul" in p["opcode"]]
            (root / f"{mode}_runtime_contracts.json").write_text(
                json.dumps(
                    dict(
                        source=f"{mode}_device0_window.csv.gz",
                        matmuls=matmuls,
                        collectives=collectives,
                        all_opcodes=sorted({r["OP CODE"] for r in rows}),
                        note="Exact rank0 runtime attributes/tensor policy for all unique matmul and collective programs; complete rows on all ranks are retained in signposted archives.",
                    ),
                    indent=2,
                )
                + "\n"
            )
        (root / f"{mode}_split_accounting.json").write_text(json.dumps(split, indent=2) + "\n")

    # The installed report assumes eight DRAM-sharded workers; the selected head
    # actually uses two workers per bank on eight banks. Preserve raw percentages.
    contract = json.loads((root / "decode_runtime_contracts.json").read_text())
    head = next(
        p for p in contract["matmuls"] if p["tensors"].get("INPUT_1_X_PAD[LOGICAL]", "").split("[")[0] == "32768"
    )
    ns = sum(head["kernel_ns"]) / len(head["kernel_ns"])
    report_source = Path("python_env/lib/python3.10/site-packages/tt_perf_report/perf_report.py")
    weight_bytes = 4096 * 32768 * 2
    measured_tflops = 2 * 32 * 4096 * 32768 / ns / 1000
    classification = dict(
        measured_head_program=head,
        kernel_mean_ns=ns,
        input_storage_cores=64,
        dram_banks=8,
        readers_per_bank=2,
        actual_compute_workers=16,
        installed_report_model_workers=8,
        report_source=str(report_source),
        report_source_sha256=hashlib.sha256(report_source.read_bytes()).hexdigest(),
        raw_modeled_flops_percent=measured_tflops / (8 * 1.3824) * 100,
        corrected_modeled_flops_percent=measured_tflops / (16 * 1.3824) * 100,
        weight_bytes=weight_bytes,
        weight_dram_GB_per_second=weight_bytes / ns,
        weight_bandwidth_roofline_percent=weight_bytes / ns / 512 * 100,
        ideal_weight_read_ns=weight_bytes / 512,
        classification="Reporting-model worker undercount, not physical utilization above100%. Native two readers/bank imply16 compute workers. Raw reports are preserved. Weight-bandwidth floor remains above corrected HiFi4 compute floor.",
        lower_fidelity_evidence="../head_fidelity_probe.json and optimization_evidence.md; real-hidden controls showed no material gain. Lower head dtypes remain rejected by full-model French controls.",
    )
    classification["all_dram_sharded_worker_corrections"] = [
        dict(
            program_hash=program["program_hash"],
            attributes=program["attributes"],
            workers_per_bank=int(re.search(r"num_workers_per_dram_bank=(\d+)", program["attributes"]).group(1)),
            report_compute_percent_divisor=int(
                re.search(r"num_workers_per_dram_bank=(\d+)", program["attributes"]).group(1)
            ),
        )
        for program in contract["matmuls"]
        if "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig" in program["attributes"]
    ]
    (root / "head_roofline_classification.json").write_text(json.dumps(classification, indent=2) + "\n")


if __name__ == "__main__":
    main()

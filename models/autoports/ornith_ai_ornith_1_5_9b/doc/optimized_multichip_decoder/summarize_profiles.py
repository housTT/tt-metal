# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reconcile paired TP4 profiles using preserved raw evidence; no device imports."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

KINDS = {"linear_attention": 0, "full_attention": 3}
MODES = ("prefill", "decode")
PM_COLUMNS = {"bandwidth": "PM BANDWIDTH [ns]", "compute": "PM COMPUTE [ns]", "ideal": "PM IDEAL [ns]"}
TILE_BYTES = {"BFLOAT4_B": 576, "BFLOAT8_B": 1088, "BFLOAT16": 2048, "FLOAT32": 4096}
EXPECTED_DECODE_PROJECTION_BYTES = {"linear_attention": 30_818_304, "full_attention": 34_734_080}
MODELED_DRAM_BYTES_PER_SECOND_PER_CHIP = 512_000_000_000


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def number(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"missing/invalid numeric column {key!r}: {row.get(key)!r}") from error
    require(math.isfinite(value), f"non-finite {key}: {value}")
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_asset(path, doc, compressed=False):
    candidates = [path.with_name(path.name + ".gz"), path] if compressed else [path]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    require(existing, f"required evidence is missing: {path}{'(.gz)' if compressed else ''}")
    contents = [(p, p.read_bytes()) for p in existing]
    decoded = [gzip.decompress(data) if p.suffix == ".gz" else data for p, data in contents]
    require(all(data == decoded[0] for data in decoded), f"plain/compressed evidence differs: {path}")
    selected, raw = contents[0]
    return decoded[0], {
        "path": str(selected.relative_to(doc)),
        "file_sha256": digest(raw),
        "uncompressed_sha256": digest(decoded[0]),
    }


def op_group(code):
    if any(name in code for name in ("AllGather", "ReduceScatter", "AllReduce")):
        return "collective"
    if "Matmul" in code:
        return "matmul"
    if any(
        name in code for name in ("Sharded", "Reshard", "Transpose", "Slice", "Copy", "Reshape", "Tilize", "Concat")
    ):
        return "data_movement"
    return "other_compute"


def pm_summary(rows, iterations):
    result = {}
    for name, column in PM_COLUMNS.items():
        valid = []
        rejected = Counter()
        for row in rows:
            try:
                value = float(row.get(column, ""))
            except (ValueError, TypeError):
                rejected["missing_or_invalid"] += 1
                continue
            if not math.isfinite(value) or value <= 1:
                rejected["nonfinite_or_at_most_1ns_placeholder"] += 1
                continue
            valid.append((value, number(row, "DEVICE KERNEL DURATION [ns]")))
        kernel_ns = sum(kernel for _, kernel in valid)
        result[name] = {
            "source_column": column,
            "valid_operations": len(valid),
            "excluded_operations": dict(rejected),
            "modeled_us_per_iteration": sum(value for value, _ in valid) / 1000 / iterations if valid else None,
            "covered_kernel_us_per_iteration": kernel_ns / 1000 / iterations if valid else None,
            "modeled_to_covered_kernel_ratio": sum(value for value, _ in valid) / kernel_ns if kernel_ns > 0 else None,
        }
    return result


def shape(row, prefix, logical=True):
    dimensions = []
    for axis in "WZYX":
        value = row.get(f"{prefix}_{axis}_PAD[LOGICAL]", "")
        match = re.fullmatch(r"(\d+)(?:\[(\d+)\])?", value)
        require(match, f"invalid tensor dimension {prefix}/{axis}: {value!r}")
        dimensions.append(int(match.group(2) if logical and match.group(2) else match.group(1)))
    return dimensions


def attribute(raw, key):
    match = re.search(r"'" + re.escape(key) + r"'\s*:\s*'([^']*)'", raw)
    return match.group(1) if match else None


def projection_metadata(row, role):
    raw = row.get("ATTRIBUTES", "")
    program = attribute(raw, "program_config")
    require(program, f"projection {role} has no raw program_config metadata")
    tensors = {}
    for prefix in ("INPUT_0", "INPUT_1", "OUTPUT_0"):
        tensors[prefix] = {
            "logical_shape": shape(row, prefix),
            "padded_shape": shape(row, prefix, logical=False),
            "dtype": row.get(f"{prefix}_DATATYPE") or None,
            "layout": row.get(f"{prefix}_LAYOUT") or None,
            "memory": row.get(f"{prefix}_MEMORY") or None,
        }
        require(tensors[prefix]["dtype"], f"projection {role} has no {prefix} dtype")
    reader = re.search(r"num_workers_per_dram_bank\s*=\s*(\d+)", program)
    return {
        "role": role,
        "role_identification": "local logical weight K/N and projection order; raw geometry retained",
        "op_code": row["OP CODE"],
        "tensors": tensors,
        "math_fidelity": row.get("MATH FIDELITY") or None,
        "compute_kernel_config": attribute(raw, "compute_kernel_config"),
        "program_config": program,
        "program_class": program.split("(", 1)[0],
        "dram_readers": int(reader.group(1)) if reader else None,
        "reported_core_count": int(number(row, "CORE COUNT")),
        "reported_core_count_semantics": "profiler program worker-core count; not the activation input shard-core count",
    }


def projections(rows, iterations, kind, registry):
    selected = [row for row in rows if "Matmul" in row["OP CODE"] and min(shape(row, "INPUT_1")[-2:]) >= 1024]
    require(selected and len(selected) % iterations == 0, "projection count does not divide measured iterations")
    slots = len(selected) // iterations
    roles = {
        (4096, 2112): "gdn_packed",
        (4096, 3136): "gdn_all",
        (4096, 1024): "gdn_z_epilogue",
        (1024, 4096): "gdn_out" if kind == "linear_attention" else "o_proj",
        (3072, 4096): "down_proj",
        (4096, 2560): "qkvg",
        (4096, 6144): "gate_up",
    }
    mlp_slot = 0
    result = []
    for slot in range(slots):
        group = selected[slot::slots]
        dimensions = tuple(shape(group[0], "INPUT_1")[-2:])
        if dimensions == (4096, 3072):
            require(mlp_slot < 2, "more than two separate gate/up projections per iteration")
            role = ("gate_proj", "up_proj")[mlp_slot]
            mlp_slot += 1
        else:
            role = roles.get(dimensions, f"projection_slot{slot}_{dimensions[0]}x{dimensions[1]}")
        metadata = projection_metadata(group[0], role)
        require(
            all(projection_metadata(row, role) == metadata for row in group),
            f"projection slot {slot} changes between replays",
        )
        key = digest(json.dumps(metadata, sort_keys=True).encode())[:16]
        registry[key] = metadata
        result.append(
            {
                "role": role,
                "metadata_id": key,
                "calls_per_iteration": len(group) / iterations,
                "kernel_us_per_iteration": sum(number(row, "DEVICE KERNEL DURATION [ns]") for row in group)
                / 1000
                / iterations,
                "program_hashes": sorted({row["PROGRAM HASH"] for row in group}),
                "performance_model": pm_summary(group, iterations),
            }
        )
    return result


def rank_summary(rows, saved, iterations, kind, registry):
    rows.sort(key=lambda row: int(row["DEVICE FW START CYCLE"]))
    require(len(rows) % iterations == 0, "rank operation count does not divide measured iterations")
    kernels = [number(row, "DEVICE KERNEL DURATION [ns]") for row in rows]
    require(min(kernels) >= 0, "negative kernel duration is invalid")
    gaps = [number(row, "OP TO OP LATENCY [ns]") for row in rows]
    clocks = [
        (int(row["DEVICE FW END CYCLE"]) - int(row["DEVICE FW START CYCLE"])) / number(row, "DEVICE FW DURATION [ns]")
        for row in rows
        if number(row, "DEVICE FW DURATION [ns]") > 0
    ]
    require(clocks and min(clocks) > 0, "cannot infer per-rank cycle/ns scale")
    clock = statistics.median(clocks)
    scale = 1000 * iterations
    kernel_us, gap_us = sum(kernels) / scale, sum(gaps[1:]) / scale
    span_cycles = max(int(row["DEVICE FW END CYCLE"]) for row in rows) - int(rows[0]["DEVICE FW START CYCLE"])
    span_us = span_cycles / clock / scale
    require(
        saved["operations"] == len(rows) and saved["iterations"] == iterations,
        "rank accounting count/iterations mismatch",
    )
    for key, computed in [
        ("kernel_us_per_iteration", kernel_us),
        ("gap_us_per_iteration", gap_us),
        ("removed_pre_window_gap_ns", gaps[0]),
    ]:
        require(
            math.isclose(saved[key], computed, rel_tol=1e-9, abs_tol=1e-6), f"raw/rank accounting mismatch for {key}"
        )
    groups, codes = defaultdict(float), defaultdict(list)
    for row, duration in zip(rows, kernels):
        groups[op_group(row["OP CODE"])] += duration / scale
        codes[row["OP CODE"]].append(row)
    return {
        "operations": len(rows),
        "operations_per_iteration": len(rows) / iterations,
        "kernel_us_per_iteration": kernel_us,
        "interior_gap_us_per_iteration": gap_us,
        "positive_interior_gap_us_per_iteration": sum(max(gap, 0) for gap in gaps[1:]) / scale,
        "negative_interior_gap_us_per_iteration": sum(min(gap, 0) for gap in gaps[1:]) / scale,
        "excluded_pre_window_gap_ns": gaps[0],
        "kernel_plus_gap_us_per_iteration": kernel_us + gap_us,
        "firmware_endpoint_span_us_per_iteration": span_us,
        "span_minus_kernel_and_gap_us_per_iteration": span_us - kernel_us - gap_us,
        "inferred_clock_cycles_per_ns": clock,
        "inferred_clock_sample_range": [min(clocks), max(clocks)],
        "trace_replay_sessions": sorted(
            {row["METAL TRACE REPLAY SESSION ID"] for row in rows if row["METAL TRACE REPLAY SESSION ID"]}
        ),
        "kernel_groups_us_per_iteration": dict(groups),
        "op_counts": {
            code: {
                "count": len(group),
                "per_iteration": len(group) / iterations,
                "kernel_us_per_iteration": sum(number(row, "DEVICE KERNEL DURATION [ns]") for row in group) / scale,
            }
            for code, group in sorted(codes.items())
        },
        "performance_model": pm_summary(rows, iterations),
        "projections": projections(rows, iterations, kind, registry),
    }


def norm_sdpa_contracts(rows, iterations):
    """Retain actual norm/SDPA tensor and memory contracts from the measured window."""
    groups = {}
    for row in rows:
        if not any(token in row["OP CODE"].lower() for token in ("norm", "sdpa")):
            continue
        prefixes = sorted(
            key.removesuffix("_DATATYPE")
            for key, value in row.items()
            if re.fullmatch(r"(?:INPUT|OUTPUT)_\d+_DATATYPE", key) and value
        )
        metadata = {
            "op_code": row["OP CODE"],
            "attributes": row.get("ATTRIBUTES"),
            "math_fidelity": row.get("MATH FIDELITY"),
            "reported_core_count": int(number(row, "CORE COUNT")),
            "tensors": {
                prefix: {
                    "logical_shape": shape(row, prefix),
                    "padded_shape": shape(row, prefix, logical=False),
                    "dtype": row[f"{prefix}_DATATYPE"],
                    "layout": row.get(f"{prefix}_LAYOUT"),
                    "memory": row.get(f"{prefix}_MEMORY"),
                }
                for prefix in prefixes
            },
        }
        key = digest(json.dumps(metadata, sort_keys=True).encode())[:16]
        entry = groups.setdefault(key, {"metadata": metadata, "ranks": {}})
        rank = entry["ranks"].setdefault(row["DEVICE ID"], {"count": 0, "kernel_ns": 0, "global_call_counts": []})
        rank["count"] += 1
        rank["kernel_ns"] += number(row, "DEVICE KERNEL DURATION [ns]")
        rank["global_call_counts"].append(row.get("GLOBAL CALL COUNT"))
    for entry in groups.values():
        for rank in entry["ranks"].values():
            rank["calls_per_iteration"] = rank.pop("count") / iterations
            rank["kernel_us_per_iteration"] = rank.pop("kernel_ns") / 1000 / iterations
    return groups


def storage_floor(profile, rows):
    """An optimistic storage-read model, never measured DRAM traffic or bandwidth."""
    if profile["mode"] != "decode":
        return {"available": False, "reason": "This context-specific weight-plus-KV read model covers decode only."}
    require(profile["context"]["batch"] == 1, "storage-floor contract expects logical batch 1")
    context = profile["context"]["decode_context"]
    require(context == 2049, f"storage-floor contract expects decode context 2049, got {context}")
    weights = {}
    for device, rank in profile["ranks"].items():
        weights[device] = {}
        for projection in rank["projections"]:
            require(projection["calls_per_iteration"] == 1, "storage-floor projection must run once per decode")
            tensor = profile["projection_metadata"][projection["metadata_id"]]["tensors"]["INPUT_1"]
            dimensions = tensor["padded_shape"]
            require(tensor["layout"] == "TILE", "storage-floor weight is not tiled")
            require(tensor["dtype"] in TILE_BYTES, f"unknown tile storage for {tensor['dtype']}")
            require(all(d % 32 == 0 for d in dimensions[-2:]), "weight dimensions are not whole tiles")
            tiles = math.prod(dimensions[:-2]) * (dimensions[-2] // 32) * (dimensions[-1] // 32)
            weights[device][projection["role"]] = tiles * TILE_BYTES[tensor["dtype"]]
        require(
            sum(weights[device].values()) == EXPECTED_DECODE_PROJECTION_BYTES[profile["kind"]],
            f"actual {profile['label']}/{profile['kind']} weight storage differs from the final policy contract: {weights[device]}",
        )
    cache_bytes = 0
    cache_geometry = None
    if profile["kind"] == "full_attention":
        sdpa = [row for row in rows if "sdpa" in row["OP CODE"].lower()]
        require(sdpa, "full-attention storage floor requires actual SDPA cache metadata")
        for row in sdpa:
            for prefix in ("INPUT_1", "INPUT_2"):
                require(shape(row, prefix)[1] == 1 and shape(row, prefix)[-1] == 256, "unexpected local KV geometry")
                require(row[f"{prefix}_DATATYPE"] == "BFLOAT8_B", "storage floor expects the recorded BFP8 KV cache")
        cache_geometry = {
            "logical_tokens": context,
            "token_tiles": math.ceil(context / 32),
            "head_width_tiles": 8,
            "local_kv_heads": 1,
            "bytes_per_tile": 1088,
            "k_and_v": 2,
        }
        cache_bytes = math.ceil(context / 32) * 8 * 1088 * 2
    projection_bytes = EXPECTED_DECODE_PROJECTION_BYTES[profile["kind"]]
    total_bytes = projection_bytes + cache_bytes
    floor_us = total_bytes / MODELED_DRAM_BYTES_PER_SECOND_PER_CHIP * 1_000_000
    host_us = profile["measurement"]["host_wall_per_iteration_ms"] * 1000
    rank_count = len(profile["ranks"])
    return {
        "available": True,
        "model": "optimistic active projection storage plus minimum tiled KV reads",
        "bandwidth_is_assumption_not_measurement": True,
        "assumed_bytes_per_second_per_chip": MODELED_DRAM_BYTES_PER_SECOND_PER_CHIP,
        "assumed_aggregate_bytes_per_second": rank_count * MODELED_DRAM_BYTES_PER_SECOND_PER_CHIP,
        "projection_bytes_per_rank": projection_bytes,
        "projection_bytes_by_role_and_rank": weights,
        "minimum_tiled_kv_read_bytes_per_rank": cache_bytes,
        "kv_geometry": cache_geometry,
        "total_modeled_read_bytes_per_rank": total_bytes,
        "aggregate_modeled_read_bytes": rank_count * total_bytes,
        "floor_us_per_iteration": floor_us,
        "aggregate_bytes_over_aggregate_bandwidth_us": rank_count
        * total_bytes
        / (rank_count * MODELED_DRAM_BYTES_PER_SECOND_PER_CHIP)
        * 1_000_000,
        "same_run_host_wall_us_per_iteration": host_us,
        "floor_to_same_run_host_wall_ratio": floor_us / host_us,
        "floor_to_max_independent_rank_span_ratio": floor_us / profile["max_independent_rank_span_us_per_iteration"],
        "ranks": {
            device: {
                "same_run_kernel_us_per_iteration": rank["kernel_us_per_iteration"],
                "same_run_endpoint_span_us_per_iteration": rank["firmware_endpoint_span_us_per_iteration"],
                "floor_to_kernel_ratio": floor_us / rank["kernel_us_per_iteration"],
                "floor_to_endpoint_span_ratio": floor_us / rank["firmware_endpoint_span_us_per_iteration"],
            }
            for device, rank in profile["ranks"].items()
        },
        "exclusions": [
            "unused persistent weight copies",
            "recurrent-state reads and writes",
            "KV-cache writes",
            "activation traffic",
            "collectives",
            "SDPA chunk over-read and repeated tile reads",
            "compute",
            "host and synchronization overhead",
        ],
    }


def load_profile(doc, label, kind, mode):
    folder = doc / "tracy" / kind / label
    assets = {}

    def asset(name, path, compressed=False):
        content, assets[name] = read_asset(path, doc, compressed)
        return content

    ops = asset("raw_ops", folder / f"{mode}_ops.csv", compressed=True)
    accounting = json.loads(asset("rank_accounting", folder / f"{mode}_rank_accounting.json"))
    provenance = json.loads(asset("profile_provenance", folder / f"{mode}_provenance.json"))
    stem = doc / "logs" / f"profile_{label}_{kind}_{mode}"
    log = asset("profile_log", stem.with_suffix(".log"), compressed=True).decode()
    run = json.loads(asset("run_provenance", stem.with_suffix(".provenance.json")))
    require(run["returncode"] == 0, f"profile command failed: {stem}")
    require(digest(ops) == provenance["ops_sha256"], f"ops hash does not match provenance: {folder}")
    require(digest(log.encode()) == run["log_sha256"], f"log hash does not match provenance: {stem}")
    measurements = [
        json.loads(line.split("PROFILE_MEASUREMENT ", 1)[1])
        for line in log.splitlines()
        if "PROFILE_MEASUREMENT " in line
    ]
    measurements = [item for item in measurements if item["mode"] == mode and item["layer"] == KINDS[kind]]
    require(len(measurements) == 1, f"expected one matching same-run PROFILE_MEASUREMENT: {stem}")
    measurement = measurements[0]
    iterations = measurement["iterations"]
    require(isinstance(iterations, int) and iterations > 0, "invalid measurement iterations")
    require(provenance["measured_iterations"] == iterations, "profile/host iteration count mismatch")
    require(number(measurement, "host_wall_per_iteration_ms") > 0, "host wall measurement must be positive")
    require(measurement["baseline"] is False, "expected the four-chip profile, not the single-chip baseline")
    require(measurement["sequence_length"] == provenance["sequence_length"], "host/profile sequence length mismatch")
    if mode == "decode":
        require(provenance["decode_context"] == measurement["decode_position"] + 1, "decode context/position mismatch")
    all_rows = list(csv.DictReader(io.StringIO(ops.decode())))
    starts = [i for i, row in enumerate(all_rows) if row["OP CODE"] == f"PERF_{mode.upper()}"]
    ends = [i for i, row in enumerate(all_rows) if row["OP CODE"] == f"PERF_{mode.upper()}_END"]
    require(len(starts) == len(ends) == 1 and starts[0] < ends[0], f"ambiguous signposted window: {folder}")
    start, end = starts[0], ends[0]
    window = [row for row in all_rows[start + 1 : end] if row["DEVICE ID"]]
    devices = sorted({row["DEVICE ID"] for row in window})
    require(len(devices) == 4 and set(devices) == set(accounting["devices"]), f"expected four matching ranks: {folder}")
    registry = {}
    ranks = {
        device: rank_summary(
            [row for row in window if row["DEVICE ID"] == device],
            accounting["devices"][device],
            iterations,
            kind,
            registry,
        )
        for device in devices
    }
    if mode == "decode":
        require(
            all(len(rank["trace_replay_sessions"]) == iterations for rank in ranks.values()),
            f"trace replay session count does not match measured iterations: {folder}",
        )
    max_span = max(rank["firmware_endpoint_span_us_per_iteration"] for rank in ranks.values())
    profile = {
        "label": label,
        "kind": kind,
        "mode": mode,
        "measurement": measurement,
        "host_signpost_span_us_per_iteration": (
            number(all_rows[end], "HOST START TS") - number(all_rows[start], "HOST START TS")
        )
        / 1000
        / iterations,
        "max_independent_rank_span_us_per_iteration": max_span,
        "host_minus_max_independent_rank_span_us_per_iteration": measurement["host_wall_per_iteration_ms"] * 1000
        - max_span,
        "ranks": ranks,
        "projection_metadata": registry,
        "norm_sdpa_runtime_contracts": norm_sdpa_contracts(window, iterations),
        "inputs": assets,
        "runtime_binary_sha256": run.get("runtime_binary_sha256", {}),
        "native_source_sha256": run.get("native_source_sha256", {}),
        "source_archive_sha256": run.get("source_archive_sha256"),
        "context": {
            key: provenance.get(key)
            for key in ("physical_hardware", "profile", "batch", "sequence_length", "decode_context")
        },
    }
    profile["optimistic_storage_floor"] = storage_floor(profile, window)
    return profile


def extent(values, decimals=3):
    values = list(values)
    lo, hi = min(values), max(values)
    return (
        f"{lo:.{decimals}f}"
        if math.isclose(lo, hi, abs_tol=10 ** (-decimals))
        else f"{lo:.{decimals}f}–{hi:.{decimals}f}"
    )


def comparisons(profiles, labels):
    if len(labels) != 2:
        return []
    results = []
    for kind in KINDS:
        for mode in MODES:
            pair = [
                next(p for p in profiles if p["label"] == label and p["kind"] == kind and p["mode"] == mode)
                for label in labels
            ]
            for field in ("sequence_length", "decode_position", "activation_source"):
                require(
                    pair[0]["measurement"][field] == pair[1]["measurement"][field],
                    f"before/after {kind}/{mode} differs in {field}",
                )
            require(pair[0]["context"]["batch"] == pair[1]["context"]["batch"], "before/after batch differs")
            wall = [p["measurement"]["host_wall_per_iteration_ms"] * 1000 for p in pair]
            spans = [p["max_independent_rank_span_us_per_iteration"] for p in pair]
            results.append(
                {
                    "kind": kind,
                    "mode": mode,
                    "before_host_us_per_iteration": wall[0],
                    "after_host_us_per_iteration": wall[1],
                    "host_latency_reduction_percent": 100 * (1 - wall[1] / wall[0]),
                    "before_max_rank_span_us_per_iteration": spans[0],
                    "after_max_rank_span_us_per_iteration": spans[1],
                    "max_rank_span_reduction_percent": 100 * (1 - spans[1] / spans[0]),
                }
            )
    return results


def projection_table(profiles):
    lines = [
        "",
        "## Projection runtime metadata",
        "",
        "Actual activation/weight/output dtype, fidelity, program family, K block, output tiles per core, and explicit DRAM reader count follow. A dash means the raw configuration does not expose that field. JSON retains complete program/memory configurations and per-rank model coverage. Its reported_core_count is the profiler's program worker-core count, not the input shard-core count (e.g. 110 or 80 program workers can use a 32-core input shard). Roles are inferred from local K/N and gate/up order.",
        "",
        "| Label / kind / mode | Role | A / B / output dtype | Fidelity | Program; K block; M×N; readers | Rank kernel µs/iter |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for profile in profiles:
        roles = sorted({projection["role"] for rank in profile["ranks"].values() for projection in rank["projections"]})
        for role in roles:
            selected = [
                projection
                for rank in profile["ranks"].values()
                for projection in rank["projections"]
                if projection["role"] == role
            ]
            metadata = [profile["projection_metadata"][projection["metadata_id"]] for projection in selected]
            dtypes = sorted(
                {
                    " / ".join(item["tensors"][prefix]["dtype"] for prefix in ("INPUT_0", "INPUT_1", "OUTPUT_0"))
                    for item in metadata
                }
            )
            programs = set()
            for item in metadata:
                values = {}
                for field in ("in0_block_w", "per_core_M", "per_core_N"):
                    match = re.search(r"\b" + field + r"=(\d+)", item["program_config"])
                    values[field] = match.group(1) if match else "—"
                programs.add(
                    f"{item['program_class']}; {values['in0_block_w']}; {values['per_core_M']}×{values['per_core_N']}; {item['dram_readers'] if item['dram_readers'] is not None else '—'}"
                )
            key = f"{profile['label']} / {profile['kind']} / {profile['mode']}"
            fidelity = ", ".join(sorted({item["math_fidelity"] or "unavailable" for item in metadata}))
            lines.append(
                f"| {key} | {role} | {', '.join(dtypes)} | {fidelity} | {', '.join(sorted(programs))} | {extent(p['kernel_us_per_iteration'] for p in selected)} |"
            )
    return lines


def markdown(document):
    lines = [
        "# TP4 performance accounting",
        "",
        document["status"],
        "",
        "Four Blackhole chips on physical P300c boards; native 1x4 ring. All timing is from the same profiled run as its raw ops. Profiler overhead is retained.",
        "",
        "Ranks have independent clocks. Ranges below are min–max across ranks, never sums. Endpoint spans divide the measured firmware cycle range by an independently inferred per-rank cycles/ns scale. Firmware durations are not added to kernel times.",
        "",
        "| Label / kind / mode | Host wall µs/iter | Rank kernel µs/iter | Rank interior gaps µs/iter | Rank endpoint span µs/iter | Span − kernel − gaps µs/iter | Ops/rank/iter |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in document["profiles"]:
        ranks = list(profile["ranks"].values())
        key = f"{profile['label']} / {profile['kind']} / {profile['mode']}"
        values = [
            extent(rank[field] for rank in ranks)
            for field in (
                "kernel_us_per_iteration",
                "interior_gap_us_per_iteration",
                "firmware_endpoint_span_us_per_iteration",
                "span_minus_kernel_and_gap_us_per_iteration",
                "operations_per_iteration",
            )
        ]
        lines.append(
            f"| {key} | {profile['measurement']['host_wall_per_iteration_ms'] * 1000:.3f} | "
            + " | ".join(values)
            + " |"
        )
    if document["comparisons"]:
        lines += [
            "",
            "| Kind / mode | Before host µs/iter | After host µs/iter | Host latency reduction | Max independent rank-span reduction |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for comparison in document["comparisons"]:
            lines.append(
                f"| {comparison['kind']} / {comparison['mode']} | {comparison['before_host_us_per_iteration']:.3f} | {comparison['after_host_us_per_iteration']:.3f} | {comparison['host_latency_reduction_percent']:.2f}% | {comparison['max_rank_span_reduction_percent']:.2f}% |"
            )
    lines += [
        "",
        "Only the first rank gap begins before the signpost and is excluded; its original value is retained in JSON/CSV. Every interior gap is preserved, including signed gaps and boundaries between trace replays.",
        "",
        "PM BANDWIDTH is a modeled time in nanoseconds, not measured traffic or bandwidth. Values ≤1 ns are excluded: the generic model initializes bandwidth/compute/ideal time to 1 ns (ttnn/api/ttnn/operation.hpp). Coverage and exclusions are explicit in JSON. These partial modeled totals are not a full-layer roofline.",
        "",
        "| Label / kind / mode | Valid bandwidth-model ops/rank/iter | Modeled bandwidth µs/iter | Kernel time covered by model µs/iter |",
        "| --- | ---: | ---: | ---: |",
    ]
    for profile in document["profiles"]:
        models = [rank["performance_model"]["bandwidth"] for rank in profile["ranks"].values()]
        valid = [item for item in models if item["modeled_us_per_iteration"] is not None]
        key = f"{profile['label']} / {profile['kind']} / {profile['mode']}"
        values = [extent(item["valid_operations"] / profile["measurement"]["iterations"] for item in models)]
        values += [
            extent(item[field] for item in valid) if valid else "unavailable"
            for field in ("modeled_us_per_iteration", "covered_kernel_us_per_iteration")
        ]
        lines.append(f"| {key} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Optimistic decode storage floor, paired with the same run",
        "",
        "This separate model reads each active projection weight once, including BFP tile exponent bytes, plus the minimum tiled K/V cache at 2049 logical tokens. Actual profiled weight dtype/shape and BFP8 cache metadata must match the final policy: 30,818,304 projection bytes/rank for linear attention; 34,734,080 for full attention, plus 65×8×1088×2 = 1,131,520 KV bytes/rank. Packed GDN and gate/up count their consumed packed weights once; unused persistent copies are excluded.",
        "",
        "The denominator is an explicit 512 GB/s per-chip model assumption (decimal GB), taken from the installed tt-perf-report Blackhole architecture model and the prior-stage PERF_ANALYSIS.md. It is not a measured P300c peak. Official [P150 specifications](https://docs.tenstorrent.com/aibs/blackhole/) list 512 GB/s; the [QuietBox P300c specifications](https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html) do not establish a measured per-chip DRAM value for this runner. TP4 aggregate modeled bytes divided by 4×512 GB/s gives the same floor, without adding times across chips.",
        "",
        "Recurrent-state traffic, cache writes, activation movement, collectives, chunk over-read/repeated tile reads, compute, and host/synchronization overhead are omitted. These optimistic modeled floors are not attainable end-to-end targets. Fractions below use the same profiled host wall and rank spans, including profiler overhead and all interior gaps.",
        "",
        "| Label / kind | Projection + KV bytes/rank | Modeled floor µs | Same-run host µs | Floor / host | Same-run rank span µs | Floor / rank span |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in document["profiles"]:
        floor = profile["optimistic_storage_floor"]
        if not floor["available"]:
            continue
        lines.append(
            f"| {profile['label']} / {profile['kind']} | {floor['projection_bytes_per_rank']:,} + {floor['minimum_tiled_kv_read_bytes_per_rank']:,} | "
            f"{floor['floor_us_per_iteration']:.3f} | {floor['same_run_host_wall_us_per_iteration']:.3f} | {floor['floor_to_same_run_host_wall_ratio']:.2%} | "
            f"{extent(rank['same_run_endpoint_span_us_per_iteration'] for rank in floor['ranks'].values())} | "
            f"{extent((100 * rank['floor_to_endpoint_span_ratio'] for rank in floor['ranks'].values()), 2)}% |"
        )
    lines += [
        "",
        "JSON also retains every measured norm/SDPA tensor's actual logical/padded shape, dtype, layout, memory location, complete attributes, call counts and per-rank kernel timing under `norm_sdpa_runtime_contracts`.",
    ]
    lines += projection_table(document["profiles"])
    lines += [
        "",
        "## Material operation counts",
        "",
        "Counts below are per rank per iteration. The table includes every changed opcode count, plus opcodes contributing at least 2% of a rank's kernel total in either run; JSON retains all opcodes.",
    ]
    for kind in KINDS:
        for mode in MODES:
            profiles = [p for p in document["profiles"] if p["kind"] == kind and p["mode"] == mode]
            codes = sorted({code for p in profiles for rank in p["ranks"].values() for code in rank["op_counts"]})
            lines += [
                "",
                f"### {kind} / {mode}",
                "",
                "| Opcode | " + " | ".join(p["label"] for p in profiles) + " |",
                "| --- | " + " | ".join("---:" for _ in profiles) + " |",
            ]
            for code in codes:
                values = [
                    extent(rank["op_counts"].get(code, {}).get("per_iteration", 0) for rank in p["ranks"].values())
                    for p in profiles
                ]
                material = len(set(values)) > 1 or any(
                    rank["op_counts"].get(code, {}).get("kernel_us_per_iteration", 0)
                    >= rank["kernel_us_per_iteration"] * 0.02
                    for p in profiles
                    for rank in p["ranks"].values()
                )
                if material:
                    lines.append(f"| {code} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_reports(document, output, label):
    rows = []
    for profile in document["profiles"]:
        floor = profile["optimistic_storage_floor"]
        for device, rank in profile["ranks"].items():
            rows.append(
                {
                    "label": profile["label"],
                    "kind": profile["kind"],
                    "mode": profile["mode"],
                    "rank": device,
                    "iterations": profile["measurement"]["iterations"],
                    "same_run_host_wall_us_per_iteration": profile["measurement"]["host_wall_per_iteration_ms"] * 1000,
                    **{
                        key: rank[key]
                        for key in (
                            "operations_per_iteration",
                            "kernel_us_per_iteration",
                            "interior_gap_us_per_iteration",
                            "positive_interior_gap_us_per_iteration",
                            "negative_interior_gap_us_per_iteration",
                            "excluded_pre_window_gap_ns",
                            "kernel_plus_gap_us_per_iteration",
                            "firmware_endpoint_span_us_per_iteration",
                            "span_minus_kernel_and_gap_us_per_iteration",
                            "inferred_clock_cycles_per_ns",
                        )
                    },
                    "modeled_bandwidth_us_per_iteration": rank["performance_model"]["bandwidth"][
                        "modeled_us_per_iteration"
                    ],
                    "bandwidth_model_valid_operations": rank["performance_model"]["bandwidth"]["valid_operations"],
                    "bandwidth_model_covered_kernel_us_per_iteration": rank["performance_model"]["bandwidth"][
                        "covered_kernel_us_per_iteration"
                    ],
                    "optimistic_storage_floor_us_per_iteration": floor.get("floor_us_per_iteration"),
                    "modeled_projection_bytes_per_rank": floor.get("projection_bytes_per_rank"),
                    "modeled_minimum_kv_read_bytes_per_rank": floor.get("minimum_tiled_kv_read_bytes_per_rank"),
                    "assumed_dram_bytes_per_second_per_chip": floor.get("assumed_bytes_per_second_per_chip"),
                    "floor_to_same_run_host_wall_ratio": floor.get("floor_to_same_run_host_wall_ratio"),
                    "floor_to_same_run_rank_span_ratio": floor.get("ranks", {})
                    .get(device, {})
                    .get("floor_to_endpoint_span_ratio"),
                    "raw_ops_sha256": profile["inputs"]["raw_ops"]["uncompressed_sha256"],
                    "profile_log_sha256": profile["inputs"]["profile_log"]["uncompressed_sha256"],
                }
            )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    outputs = {
        "json": json.dumps(document, separators=(",", ":"), allow_nan=False) + "\n",
        "md": markdown(document),
        "csv": stream.getvalue(),
    }
    output.mkdir(parents=True, exist_ok=True)
    for suffix, content in outputs.items():
        target = output / f"{label}.{suffix}"
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default="before")
    parser.add_argument("--after", default="after")
    parser.add_argument("--doc-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-label", default="performance_accounting")
    parser.add_argument(
        "--before-only", action="store_true", help="explicitly labeled sanity report; no final comparison"
    )
    args = parser.parse_args()
    output = args.output_dir or args.doc_dir
    if (
        args.before_only
        and output.resolve() == args.doc_dir.resolve()
        and args.report_label == "performance_accounting"
    ):
        parser.error("--before-only requires a temporary --output-dir or distinct --report-label")
    if not args.before_only and args.before == args.after:
        parser.error("before and after labels must differ")
    if Path(args.report_label).name != args.report_label:
        parser.error("--report-label must be a filename stem")
    labels = [args.before] if args.before_only else [args.before, args.after]
    try:
        profiles = [
            load_profile(args.doc_dir, label, kind, mode) for label in labels for kind in KINDS for mode in MODES
        ]
        document = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": (
                "BEFORE-ONLY SANITY; final after evidence was not requested"
                if args.before_only
                else "Complete paired before/after accounting; required raw files, logs, provenance and rank accounting verified"
            ),
            "labels": labels,
            "generator_source_sha256": digest(Path(__file__).read_bytes()),
            "method_sources": {
                "PM_default_placeholders": "ttnn/api/ttnn/operation.hpp:95-100",
                "raw_PM_column_units": "tools/tracy/process_ops_logs.py:1838-1840",
                "raw_host_timestamp_units": "tools/tracy/process_ops_logs.py:1577",
                "raw_kernel_to_kernel_gap": "tools/tracy/process_ops_logs.py:1666-1690",
                "optimistic_storage_floor_reference": "models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder/PERF_ANALYSIS.md:18-33",
                "bandwidth_assumption": "python_env/lib/python3.10/site-packages/tt_perf_report/perf_report.py:347-350 (Blackhole 512 GB/s model)",
                "official_P150_reference_not_runner_hardware": "https://docs.tenstorrent.com/aibs/blackhole/",
                "official_P300c_system_specs": "https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html",
            },
            "units": {
                "raw_durations": "ns",
                "host_measurement": "ms",
                "reported_durations": "us per iteration",
                "firmware_timestamps": "cycles",
                "clock_scale": "cycles/ns inferred independently per rank",
            },
            "interpretation": [
                "No timings are summed across chips.",
                "Only the first pre-window gap is excluded; every interior gap remains.",
                "Endpoint spans use independent rank clocks; their maximum is not a synchronized mesh span.",
                "Host minus maximum rank span retains profiling, launch and synchronization overhead; no overhead is subtracted.",
                "PM values at most1ns are placeholders; partial modeled coverage is not measured physical bandwidth or a complete layer roofline.",
                "The separate decode storage floor uses actual active weight tile storage and minimum tiled KV reads divided by an assumed 512 GB/s per chip; it is not measured P300c bandwidth.",
            ],
            "profiles": profiles,
            "comparisons": comparisons(profiles, labels),
        }
        write_reports(document, output, args.report_label)
    except (EvidenceError, OSError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"Profile accounting failed: {error}\n")
    print(f"Wrote {output / args.report_label}.[json,md,csv] ({len(profiles)} profiles)")


if __name__ == "__main__":
    main()

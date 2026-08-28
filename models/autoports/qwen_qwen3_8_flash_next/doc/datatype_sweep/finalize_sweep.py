# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build the machine-readable sweep ledger, selected policy, and Pareto plots."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
TOP1_GATE = 90.0
TOP5_GATE = 98.0
TOP100_GATE = 100.0
HARDWARE = "P300 Blackhole board, dies 0 and 1"
MESH = "1x2 FABRIC_1D, linear CCL, 2 links, 8192-byte packets"
REFERENCE = "../full_model/readiness_aime24_chat.refpt (chat template; prompt 201 tokens; 99 scored decode rows)"
FIXED_REGIME = {
    "batch": 1,
    "prompt_tokens": 201,
    "teacher_forcing_decode_rows": 99,
    "measured_trace_replays": 98,
    "expert_host_store": "all 512 experts/layer prepacked before the measured workload",
    "expert_device_cache": "10 exact top-k slots/layer/rank, reset/cold at candidate construction",
    "expert_staging": "one exact BFP4 TILE staging allocation/rank/layer",
    "ple_store": "BF16 row-major mmap table with 8192-row host cache, reset/cold at candidate construction",
    "ple_staging": "BF16 TILE, 128-row prefill chunk",
    "routing_and_tokens": "identical AIME24 readiness prompt and reference continuation",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_path(value: dict, dotted_path: str):
    current = value
    for part in dotted_path.split("."):
        current = current[part]
    return current


def _candidate_matches_result(config: dict, result: dict) -> bool:
    propagation = result.get("precision_propagation", {})
    if result.get("config_id") != config["config_id"] or propagation.get("config_id") != config["config_id"]:
        return False
    try:
        return all(
            _get_path(config, field) == check["expected"]
            for field, check in propagation.get("checks", {}).items()
        )
    except (KeyError, TypeError):
        return False


def _command(config_id: str) -> str:
    return (
        "source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh; "
        "TT_VISIBLE_DEVICES=0,1 "
        "TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto "
        "RUN_QWEN38_DATATYPE_SWEEP=1 "
        f"QWEN38_PRECISION_CONFIG=models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/candidates/{config_id}.json "
        f"QWEN38_EVIDENCE_DIR=models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/full_runs/{config_id} "
        "timeout 2400 pytest -q -s --tt-arch blackhole "
        "models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_model_datatype_sweep_candidate"
    )


def _anomaly_control_command(config_id: str) -> str:
    return _command(config_id).replace(
        f"doc/datatype_sweep/full_runs/{config_id}",
        f"doc/datatype_sweep/anomaly_controls/{config_id}",
    )


def _policy_summary(config: dict) -> tuple[dict, dict]:
    dtype_policy = {
        "weight_groups": {
            name: {key: value for key, value in policy.items() if key in {"dtype", "policy", "layout"}}
            for name, policy in config["weight_groups"].items()
        },
        "weight_exceptions": config["weight_exceptions"],
        "layer_exceptions": config["layer_exceptions"],
        "activations": config["activations"],
        "ccl": config["ccl"],
        "kv_cache": config["kv_cache"],
        "logits_sampling": config["logits_sampling"],
        "host_backed": config["host_backed"],
    }
    fidelity = {
        name: policy.get("compute_fidelity")
        for name, policy in config["weight_groups"].items()
        if "compute_fidelity" in policy
    }
    return dtype_policy, fidelity


def _row(config_id: str) -> dict:
    config_path = ROOT / "candidates" / f"{config_id}.json"
    result_path = ROOT / "full_runs" / config_id / "candidate_result.json"
    config = json.loads(config_path.read_text())
    dtype_policy, fidelity = _policy_summary(config)
    base = {
        "config_id": config_id,
        "dtype_policy": dtype_policy,
        "compute_fidelity_policy": fidelity,
        "top1_percent": None,
        "top5_percent": None,
        "top100_percent": None,
        "ttft_seconds": None,
        "teacher_forcing_decode_seconds_per_token": None,
        "teacher_forcing_decode_tokens_per_second_per_user": None,
        "trace_verified": False,
        "measurement_regime": FIXED_REGIME,
        "reference": REFERENCE,
        "command": _command(config_id),
        "hardware": HARDWARE,
        "mesh": MESH,
        "host_service_totals": None,
        "source_provenance": None,
        "precision_propagation_all_fields_consumed": None,
        "candidate_result_policy_match": None,
        "result_artifact": str(result_path.relative_to(ROOT)),
        "result_sha256": None,
        "raw_measurements": [],
        "selection_sample_count": 0,
        "selection_basis": None,
        "anomaly_resolution": None,
        "status": "runtime-fail",
        "passes_gate": False,
        "rejection_reason": "candidate did not produce candidate_result.json",
    }
    if not result_path.is_file():
        return base
    primary_paths = [result_path] + sorted(
        (ROOT / "replicates" / config_id).glob("*/candidate_result.json")
    )
    anomaly_path = ROOT / "anomaly_controls" / config_id / "candidate_result.json"
    anomaly_paths = ([anomaly_path] if anomaly_path.is_file() else []) + sorted(
        (ROOT / "anomaly_controls" / config_id / "replicates").glob("*/candidate_result.json")
    )
    selected_default_path = ROOT / "post_selection" / "teacher_forcing_selected" / "candidate_result.json"
    if anomaly_paths and selected_default_path.is_file():
        selected_default = json.loads(selected_default_path.read_text())
        if selected_default.get("config_id") == config_id:
            anomaly_paths.append(selected_default_path)
    selection_paths = anomaly_paths or primary_paths
    all_result_paths = primary_paths + anomaly_paths
    results = [json.loads(path.read_text()) for path in selection_paths]
    all_results = [json.loads(path.read_text()) for path in all_result_paths]
    result = results[0]
    trace_verified = all(
        bool(item.get("traced"))
        and int(item.get("model_only_trace_replays", -1)) == int(item.get("decode_measured_tokens", -2)) == 98
        for item in results
    )
    accuracy_pass = all(
        float(item["top1_percent"]) >= TOP1_GATE
        and float(item["top5_percent"]) >= TOP5_GATE
        and float(item["top100_percent"]) == TOP100_GATE
        for item in results
    )
    propagation_pass = all(
        bool(item.get("precision_propagation", {}).get("all_fields_consumed")) for item in results
    )
    candidate_result_policy_match = all(_candidate_matches_result(config, item) for item in all_results)
    source_digests = {item.get("source_provenance", {}).get("digest") for item in results}
    provenance_pass = len(source_digests) == 1 and None not in source_digests
    top1 = statistics.median(float(item["top1_percent"]) for item in results)
    top5 = statistics.median(float(item["top5_percent"]) for item in results)
    top100 = statistics.median(float(item["top100_percent"]) for item in results)
    ttft = statistics.median(float(item["ttft_seconds"]) for item in results)
    decode_seconds_per_token = statistics.median(float(item["decode_seconds_per_token"]) for item in results)
    decode_tps = statistics.median(float(item["decode_tokens_per_second_per_user"]) for item in results)
    passes = trace_verified and accuracy_pass and propagation_pass and provenance_pass and candidate_result_policy_match
    if not trace_verified:
        rejection = "decode result is not the required 98-replay traced teacher-forcing regime"
    elif not propagation_pass:
        rejection = "not every selected precision field was proven at runtime"
    elif not provenance_pass:
        rejection = "selection samples do not share one exact source digest"
    elif not candidate_result_policy_match:
        rejection = "candidate JSON does not exactly match the measured runtime precision-propagation policy"
    elif not accuracy_pass:
        rejection = (
            f"accuracy gate failed (requires top-1 >= {TOP1_GATE}, top-5 >= {TOP5_GATE}, "
            f"top-100 == {TOP100_GATE})"
        )
    else:
        rejection = None
    base.update(
        top1_percent=top1,
        top5_percent=top5,
        top100_percent=top100,
        ttft_seconds=ttft,
        teacher_forcing_decode_seconds_per_token=decode_seconds_per_token,
        teacher_forcing_decode_tokens_per_second_per_user=decode_tps,
        trace_verified=trace_verified,
        measurement_regime=result.get("measurement_regime", FIXED_REGIME),
        host_service_totals=result.get("host_service_totals"),
        source_provenance=result.get("source_provenance"),
        precision_propagation_all_fields_consumed=propagation_pass,
        candidate_result_policy_match=candidate_result_policy_match,
        result_artifact=str(selection_paths[0].relative_to(ROOT)),
        result_sha256=_sha256(selection_paths[0]),
        raw_measurements=[
            {
                "artifact": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "cohort": "current_source_anomaly_control" if path in anomaly_paths else "primary_sweep",
                "used_for_selection": path in selection_paths,
                "source_digest": item.get("source_provenance", {}).get("digest"),
                "top1_percent": float(item["top1_percent"]),
                "top5_percent": float(item["top5_percent"]),
                "top100_percent": float(item["top100_percent"]),
                "ttft_seconds": float(item["ttft_seconds"]),
                "teacher_forcing_decode_seconds_per_token": float(item["decode_seconds_per_token"]),
                "teacher_forcing_decode_tokens_per_second_per_user": float(
                    item["decode_tokens_per_second_per_user"]
                ),
                "expert_misses": float(item["host_service_totals"]["expert_misses"]),
                "expert_h2d_bytes": float(item["host_service_totals"]["expert_h2d_bytes"]),
                "expert_h2d_seconds": float(item["host_service_totals"]["expert_h2d_seconds"]),
                "ple_table_rows_read": float(item["host_service_totals"]["ple_table_rows_read"]),
                "ple_lookup_seconds": float(item["host_service_totals"]["ple_lookup_seconds"]),
            }
            for path, item in zip(all_result_paths, all_results)
        ],
        selection_sample_count=len(results),
        selection_basis=(
            "median_of_current_source_anomaly_controls"
            if anomaly_paths
            else "median_of_primary_and_same_source_replicates"
        ),
        anomaly_resolution=(
            {
                "reason": (
                    "The independent stage review required a current-source resolution of host-timing "
                    "variance and the close finalist comparison. This current-source cohort supersedes the "
                    "historical primary sample for selection while every measurement remains in raw_measurements."
                ),
                "command": _anomaly_control_command(config_id),
                "artifact": str(anomaly_path.relative_to(ROOT)),
                "primary_teacher_forcing_decode_tokens_per_second_per_user": statistics.median(
                    float(item["decode_tokens_per_second_per_user"])
                    for item in all_results[: len(primary_paths)]
                ),
                "control_teacher_forcing_decode_tokens_per_second_per_user": decode_tps,
                "primary_expert_h2d_seconds": statistics.median(
                    float(item["host_service_totals"]["expert_h2d_seconds"])
                    for item in all_results[: len(primary_paths)]
                ),
                "control_expert_h2d_seconds": statistics.median(
                    float(item["host_service_totals"]["expert_h2d_seconds"])
                    for item in results
                ),
            }
            if anomaly_paths
            else None
        ),
        status="pass" if passes else "accuracy-fail" if not accuracy_pass else "runtime-fail",
        passes_gate=passes,
        rejection_reason=rejection,
    )
    return base


def _pareto(rows: list[dict], accuracy_key: str) -> list[dict]:
    valid = [
        row
        for row in rows
        if row[accuracy_key] is not None and row["teacher_forcing_decode_tokens_per_second_per_user"] is not None
    ]
    frontier = []
    for row in valid:
        dominated = any(
            other[accuracy_key] >= row[accuracy_key]
            and other["teacher_forcing_decode_tokens_per_second_per_user"]
            >= row["teacher_forcing_decode_tokens_per_second_per_user"]
            and (
                other[accuracy_key] > row[accuracy_key]
                or other["teacher_forcing_decode_tokens_per_second_per_user"]
                > row["teacher_forcing_decode_tokens_per_second_per_user"]
            )
            for other in valid
            if other is not row
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda item: item[accuracy_key])


def _plot(rows: list[dict], selected: dict, accuracy_key: str, gate: float, output: Path) -> None:
    valid = [row for row in rows if row[accuracy_key] is not None]
    code_by_id = {row["config_id"]: f"C{index:02d}" for index, row in enumerate(valid, start=1)}
    fig, axis = plt.subplots(figsize=(16, 8))
    fig.subplots_adjust(left=0.08, right=0.69, bottom=0.11, top=0.92)
    for row in valid:
        x = row[accuracy_key]
        y = row["teacher_forcing_decode_tokens_per_second_per_user"]
        axis.scatter(x, y, color="#3572A5", s=38, alpha=0.85)
        axis.annotate(
            code_by_id[row["config_id"]],
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold" if row["config_id"] == selected["config_id"] else "normal",
        )
    frontier = _pareto(valid, accuracy_key)
    if frontier:
        axis.plot(
            [row[accuracy_key] for row in frontier],
            [row["teacher_forcing_decode_tokens_per_second_per_user"] for row in frontier],
            color="#222222",
            linewidth=1.5,
            label="Pareto frontier",
        )
    axis.scatter(
        selected[accuracy_key],
        selected["teacher_forcing_decode_tokens_per_second_per_user"],
        color="red",
        edgecolor="darkred",
        marker="*",
        s=220,
        zorder=5,
        label=f"selected: {code_by_id[selected['config_id']]}",
    )
    axis.axvline(gate, color="black", linestyle=":", linewidth=1.5, label=f"minimum accuracy {gate:.0f}%")
    axis.set_xlabel("Full-model accuracy (%)")
    axis.set_ylabel("Traced teacher-forcing decode (tokens/s/user)")
    axis.set_title(f"Qwen3.8-Flash-Next {accuracy_key.replace('_percent', '').upper()} / performance Pareto")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=8)
    mapping = "Evaluated configurations\n" + "\n".join(
        f"{code_by_id[row['config_id']]}  {row['config_id']}" for row in valid
    )
    fig.text(0.71, 0.92, mapping, ha="left", va="top", family="monospace", fontsize=7.2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _post_selection(selected_config_id: str) -> dict:
    teacher_path = ROOT / "post_selection" / "teacher_forcing_selected" / "candidate_result.json"
    transient_teacher_path = (
        ROOT
        / "anomaly_controls"
        / "qsa_bfp8_hifi2_shared_bfp8_hifi2_lm_head_bf16_hifi2"
        / "transient_h2d"
        / "candidate_result.json"
    )
    displaced_teacher_path = (
        ROOT
        / "anomaly_controls"
        / "qsa_bfp8_hifi2_shared_bfp8_hifi2_lm_head_bf16_hifi2"
        / "candidate_result.json"
    )
    token_out_path = ROOT / "post_selection" / "token_out" / "full_model_performance.json"
    baseline_control_path = ROOT / "post_selection" / "token_out_baseline_control" / "full_model_performance.json"
    propagation_path = ROOT / "post_selection" / "precision_smoke" / "precision_propagation_smoke.json"
    qualitative_path = ROOT / "post_selection" / "qualitative" / "qualitative_shared_suite_final.json"
    required = (
        teacher_path,
        transient_teacher_path,
        displaced_teacher_path,
        token_out_path,
        baseline_control_path,
        propagation_path,
        qualitative_path,
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"post-selection artifact(s) missing: {missing}")

    teacher = json.loads(teacher_path.read_text())
    transient_teacher = json.loads(transient_teacher_path.read_text())
    displaced_teacher = json.loads(displaced_teacher_path.read_text())
    token_out = json.loads(token_out_path.read_text())
    baseline_control = json.loads(baseline_control_path.read_text())
    propagation = json.loads(propagation_path.read_text())
    qualitative = json.loads(qualitative_path.read_text())
    artifacts = {
        "selected_teacher_forcing": teacher_path,
        "displaced_hifi2_shared_teacher_forcing": displaced_teacher_path,
        "displaced_hifi2_shared_teacher_forcing_transient_h2d": transient_teacher_path,
        "selected_token_out": token_out_path,
        "current_source_baseline_token_out_control": baseline_control_path,
        "selected_precision_propagation": propagation_path,
        "selected_qualitative": qualitative_path,
    }
    if any(payload.get("config_id") != selected_config_id for payload in (teacher, token_out, propagation, qualitative)):
        raise RuntimeError("post-selection artifact does not use the selected config through its normal construction path")
    if not all(
        payload.get("precision_propagation", payload).get("all_fields_consumed")
        for payload in (teacher, token_out, qualitative)
    ):
        raise RuntimeError("post-selection evidence does not consume every selected precision field")

    return {
        "selected_config_id": selected_config_id,
        "selected_teacher_forcing": {
            "top1_percent": teacher["top1_percent"],
            "top5_percent": teacher["top5_percent"],
            "top100_percent": teacher["top100_percent"],
            "ttft_seconds": teacher["ttft_seconds"],
            "decode_seconds_per_token": teacher["decode_seconds_per_token"],
            "decode_tokens_per_second_per_user": teacher["decode_tokens_per_second_per_user"],
            "decode_measured_tokens": teacher["decode_measured_tokens"],
            "model_only_trace_replays": teacher["model_only_trace_replays"],
            "traced": teacher["traced"],
            "source_digest": teacher["source_provenance"]["digest"],
        },
        "displaced_hifi2_shared_policy_host_timing_variance": {
            "config_id": displaced_teacher["config_id"],
            "normal_sample": {
                "decode_tokens_per_second_per_user": displaced_teacher["decode_tokens_per_second_per_user"],
                "expert_misses": displaced_teacher["host_service_totals"]["expert_misses"],
                "expert_h2d_bytes": displaced_teacher["host_service_totals"]["expert_h2d_bytes"],
                "expert_h2d_seconds": displaced_teacher["host_service_totals"]["expert_h2d_seconds"],
            },
            "preserved_transient_sample": {
                "decode_tokens_per_second_per_user": transient_teacher["decode_tokens_per_second_per_user"],
                "expert_misses": transient_teacher["host_service_totals"]["expert_misses"],
                "expert_h2d_bytes": transient_teacher["host_service_totals"]["expert_h2d_bytes"],
                "expert_h2d_seconds": transient_teacher["host_service_totals"]["expert_h2d_seconds"],
            },
            "interpretation": (
                "The exact miss count and bytes were unchanged; the slower sample accumulated transient host H2D "
                "submission time and is retained as variance evidence for the displaced HiFi2 shared-projection "
                "policy, not used for Pareto selection."
            ),
        },
        "selected_token_out": {
            "workload": token_out["workload"],
            "measurement_regime": "warmed token-out; no activation/logit readback; traced device sampling",
            **token_out["metrics"],
            "source_digest": token_out["source_provenance"]["digest"],
            "host_service_totals": token_out["host_service_all_totals"],
        },
        "baseline_token_out_control": {
            "config_id": baseline_control["config_id"],
            "workload": baseline_control["workload"],
            **baseline_control["metrics"],
            "source_digest": baseline_control["source_provenance"]["digest"],
            "host_service_totals": baseline_control["host_service_all_totals"],
        },
        "precision_propagation": {
            "all_fields_consumed": propagation["all_fields_consumed"],
            "consumed_leaf_count": propagation["consumed_leaf_count"],
        },
        "artifacts": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for name, path in artifacts.items()
        },
    }


def main() -> None:
    manifest = json.loads((ROOT / "candidate_matrix.json").read_text())
    rows = [_row(entry["config_id"]) for entry in manifest]
    missing = [row["config_id"] for row in rows if row["top1_percent"] is None]
    if missing:
        raise RuntimeError(f"full-model result missing for candidate(s): {missing}")
    passing = [row for row in rows if row["passes_gate"]]
    if not passing:
        raise RuntimeError("no candidate satisfies accuracy, trace, and propagation gates")
    selected = max(passing, key=lambda item: item["teacher_forcing_decode_tokens_per_second_per_user"])
    for row in rows:
        row["selected"] = row["config_id"] == selected["config_id"]
        row["pareto_top1"] = row in _pareto(rows, "top1_percent")
        row["pareto_top5"] = row in _pareto(rows, "top5_percent")

    ledger = {
        "schema_version": 1,
        "model_id": "Qwen/Qwen3.8-Flash-Next",
        "thresholds": {"top1_percent_min": TOP1_GATE, "top5_percent_min": TOP5_GATE, "top100_percent_exact": TOP100_GATE},
        "selection_rule": (
            "maximum median traced teacher-forcing decode tokens/s/user among full-model configs passing all "
            "accuracy, source-provenance, and precision-propagation gates; a single sample is its own median. "
            "For rows challenged by independent review because of elevated host H2D service time, the recorded "
            "current-source anomaly control supersedes the historical primary sample for selection."
        ),
        "selected_config_id": selected["config_id"],
        "hardware": HARDWARE,
        "mesh": MESH,
        "fixed_measurement_regime": FIXED_REGIME,
        "post_selection": _post_selection(selected["config_id"]),
        "results": rows,
    }
    (ROOT / "sweep_results.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    csv_fields = [
        "config_id",
        "status",
        "passes_gate",
        "selected",
        "pareto_top1",
        "pareto_top5",
        "top1_percent",
        "top5_percent",
        "top100_percent",
        "ttft_seconds",
        "teacher_forcing_decode_seconds_per_token",
        "teacher_forcing_decode_tokens_per_second_per_user",
        "trace_verified",
        "selection_sample_count",
        "selection_basis",
        "anomaly_resolution",
        "precision_propagation_all_fields_consumed",
        "candidate_result_policy_match",
        "dtype_policy",
        "compute_fidelity_policy",
        "measurement_regime",
        "host_service_totals",
        "reference",
        "command",
        "hardware",
        "mesh",
        "result_artifact",
        "result_sha256",
        "raw_measurements",
        "rejection_reason",
    ]
    with (ROOT / "sweep_results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in csv_fields}
            for key in (
                "dtype_policy",
                "compute_fidelity_policy",
                "measurement_regime",
                "host_service_totals",
                "raw_measurements",
                "anomaly_resolution",
            ):
                flat[key] = json.dumps(flat[key], sort_keys=True)
            writer.writerow(flat)

    selected_source = ROOT / "candidates" / f"{selected['config_id']}.json"
    shutil.copyfile(selected_source, ROOT / "selected_precision_config.json")
    _plot(rows, selected, "top1_percent", TOP1_GATE, ROOT / "top1_perf_pareto.png")
    _plot(rows, selected, "top5_percent", TOP5_GATE, ROOT / "top5_perf_pareto.png")
    print(json.dumps({"selected": selected["config_id"], "passing": len(passing), "evaluated": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()

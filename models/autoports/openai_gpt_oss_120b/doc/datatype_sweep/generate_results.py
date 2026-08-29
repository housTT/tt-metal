# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Build the checked-in datatype-sweep tables and Pareto plots from raw evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[4]
TOP1_MIN = 0.90
TOP5_MIN = 0.98
HARDWARE = "4x p300c (P150 semantic target)"
MESH = [1, 4]
TEST = "models/autoports/openai_gpt_oss_120b/tests/test_full_model.py"
REFERENCE_PATH = "models/autoports/openai_gpt_oss_120b/doc/full_model/references/aime24_chat_100_top100.refpt"
REFERENCE_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
PROMPT_TOKENS = 214
GENERATED_TOKENS = 100
MEASUREMENT_REGIME = (
    "full-model AIME24 chat-template, prompt=214, generated/teacher-forced tokens=100; one discarded "
    "same-process warmup; timed decode starts with captured model and sampling traces; no eager/untraced "
    "ranking data"
)
ENVIRONMENT_NOTES = (
    "Physical 4x p300c host used as the P150x4 semantic target; serialized with /tmp/tt-device.lock; "
    "P150/P150x2 fail resident 36-layer capacity before KV allocation and are documented in context_contract.json"
)


def semantic_json_sha256(payload: dict) -> str:
    """Match PrecisionConfig.source_sha256 (semantic JSON, not whitespace bytes)."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def command_for(path: Path, config_id: str) -> str:
    repeated_candidates = {"ds00_baseline", "ds11_lm_head_lofi", "ds12_attention_hifi2_lm_head_lofi"}
    repetitions = " GPT_OSS_120B_DATATYPE_SWEEP_REPETITIONS=2" if config_id in repeated_candidates else ""
    return (
        "env PATH=$PWD/python_env/bin:$PATH GPT_OSS_120B_FULL_MODEL_READINESS=1"
        f"{repetitions} GPT_OSS_120B_DATATYPE_SWEEP_CONFIG={repo_path(path)} "
        "LD_LIBRARY_PATH=/tmp/gptoss-libnsl2:"
        "/home/ttuser/dev/gpt-oss-20b/tt-metal/build/lib "
        f"scripts/run_safe_pytest.sh {TEST} -q -k run_teacher_forcing_aime24_top100 -s"
    )


def validate_trace_measurement(trace: dict) -> None:
    expected = GENERATED_TOKENS - 1
    required = {
        "requested": True,
        "warmed_before_timing": True,
        "expected_decode_steps": expected,
        "model_trace_handles_before_timing": 1,
        "sampling_trace_handles_before_timing": 1,
        "model_trace_handles_after_timing": 1,
        "sampling_trace_handles_after_timing": 1,
        "model_execute_submissions": expected,
        "sampling_execute_submissions": expected,
        "unclassified_execute_submissions": 0,
        "trace_verified": True,
    }
    if trace != required:
        raise RuntimeError(f"teacher-forcing trace evidence does not satisfy the ranking contract: {trace}")


def measurement_provenance(payload: dict) -> dict:
    source = payload["runtime_source_provenance"]
    required = {"source_branch", "source_commit", "runtime_source_state_sha256", "runtime_source_files"}
    missing = sorted(required - set(source))
    if missing:
        raise RuntimeError(f"measurement-time source provenance missing {missing}")
    return {
        "source_branch": source["source_branch"],
        "source_commit": source["source_commit"],
        "source_state_sha256": source["runtime_source_state_sha256"],
        "source_dirty": source["source_dirty"],
        "runtime_source_files": source["runtime_source_files"],
        "runner_provenance": payload["runner_provenance"],
    }


def validate_measured_row(stats: dict, config: dict, config_path: Path) -> dict:
    if stats["precision_config_id"] != config["config_id"]:
        raise RuntimeError(f"config id mismatch for {config['config_id']}")
    expected_hash = semantic_json_sha256(config)
    if stats["precision_config_sha256"] != expected_hash:
        raise RuntimeError(f"config hash mismatch for {config['config_id']}")
    if stats["precision_config_path"] != repo_path(config_path):
        raise RuntimeError(f"config path mismatch for {config['config_id']}")
    repetitions = stats.get("warmed_repetitions", [])
    expected_repetitions = (
        2 if config["config_id"] in {"ds00_baseline", "ds11_lm_head_lofi", "ds12_attention_hifi2_lm_head_lofi"} else 1
    )
    if len(repetitions) != expected_repetitions:
        raise RuntimeError(
            f"{config['config_id']} requires {expected_repetitions} warmed repetition(s), got {len(repetitions)}"
        )
    for repetition in repetitions:
        validate_trace_measurement(repetition["trace_measurement"])
        if repetition["runtime_source_provenance"] != stats["runtime_source_provenance"]:
            raise RuntimeError(f"source changed between repetitions for {config['config_id']}")
    validate_trace_measurement(stats["trace_measurement"])
    return measurement_provenance(stats)


def base_row(config: dict, config_path: Path) -> dict:
    config_id = config["config_id"]
    return {
        "schema_version": 2,
        "config_id": config_id,
        "config_path": repo_path(config_path),
        "config_sha256": semantic_json_sha256(config),
        "layer_exceptions": config["layer_exceptions"],
        "dtype_policy": {
            "weight_groups": config["weight_groups"],
            "activation_residual_dtype": config["activation_residual_dtype"],
            "ccl_dtype": config["ccl_dtype"],
            "kv_cache_dtype": config["kv_cache_dtype"],
            "logits_sampling_dtype_assumptions": config["logits_sampling_dtype_assumptions"],
        },
        "compute_fidelity_policy": config["compute_fidelities"],
        "prompt_tokens": PROMPT_TOKENS,
        "generated_tokens": GENERATED_TOKENS,
        "reference_path": REFERENCE_PATH,
        "reference_revision": REFERENCE_REVISION,
        "measurement_regime": MEASUREMENT_REGIME,
        "command": command_for(config_path, config_id),
        "hardware": HARDWARE,
        "mesh": MESH,
        "environment_notes": ENVIRONMENT_NOTES,
    }


def load_results() -> list[dict]:
    selected = json.loads((ROOT / "selected_precision_config.json").read_text(encoding="utf-8"))
    results = []
    for config_path in sorted((ROOT / "candidates").glob("*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_id = config["config_id"]
        artifact_dir = ROOT / "artifacts" / config_id
        readiness = artifact_dir / "teacher_forcing_readiness.json"
        failure = artifact_dir / "failure.json"
        row = base_row(config, config_path)
        row["selected"] = config_id == selected["config_id"]
        if readiness.exists():
            stats = json.loads(readiness.read_text(encoding="utf-8"))[0]
            provenance = validate_measured_row(stats, config, config_path)
            runtime_evidence = artifact_dir / "runtime_precision_evidence.json"
            if not runtime_evidence.exists():
                raise RuntimeError(f"missing runtime consumption evidence for {config_id}")
            runtime = json.loads(runtime_evidence.read_text(encoding="utf-8"))
            if runtime["config_id"] != config_id or runtime["config_sha256"] != row["config_sha256"]:
                raise RuntimeError(f"runtime evidence/config mismatch for {config_id}")
            passed = stats["top1"] >= TOP1_MIN and stats["top5"] >= TOP5_MIN
            row.update(
                {
                    **provenance,
                    "top1": stats["top1"],
                    "top5": stats["top5"],
                    "top100": stats["top100"],
                    "ttft_ms": stats["ttft_ms"],
                    "trace_verified_teacher_forcing_decode_t_s_u": stats["decode_t/s/u"],
                    "trace_verified": stats["trace_measurement"]["trace_verified"],
                    "trace_measurement": stats["trace_measurement"],
                    "warmed_repetitions": stats["warmed_repetitions"],
                    "status": "pass" if passed else "fail_accuracy_gate",
                    "pass": passed,
                    "rejection_reason": None if passed else "top-1 and/or top-5 below the required gate",
                    "evidence": str(readiness.relative_to(ROOT)),
                    "runtime_consumption_evidence": str(runtime_evidence.relative_to(ROOT)),
                }
            )
        elif failure.exists():
            failed = json.loads(failure.read_text(encoding="utf-8"))
            provenance = measurement_provenance(failed)
            if failed["precision_config_sha256"] != row["config_sha256"]:
                raise RuntimeError(f"failure config hash mismatch for {config_id}")
            row.update(
                {
                    **provenance,
                    "top1": None,
                    "top5": None,
                    "top100": None,
                    "ttft_ms": None,
                    "trace_verified_teacher_forcing_decode_t_s_u": None,
                    "trace_verified": False,
                    "trace_measurement": None,
                    "warmed_repetitions": [],
                    "status": failed["status"],
                    "pass": False,
                    "rejection_reason": failed["error"],
                    "evidence": str(failure.relative_to(ROOT)),
                    "runtime_consumption_evidence": failed.get("runtime_consumption_evidence"),
                }
            )
        else:
            raise RuntimeError(f"missing full-model result for {config_id}")
        results.append(row)
    return results


def pareto_frontier(rows: list[dict], accuracy_key: str) -> list[dict]:
    measured = [row for row in rows if row[accuracy_key] is not None]
    return sorted(
        [
            row
            for row in measured
            if not any(
                other[accuracy_key] >= row[accuracy_key]
                and other["trace_verified_teacher_forcing_decode_t_s_u"]
                >= row["trace_verified_teacher_forcing_decode_t_s_u"]
                and (
                    other[accuracy_key] > row[accuracy_key]
                    or other["trace_verified_teacher_forcing_decode_t_s_u"]
                    > row["trace_verified_teacher_forcing_decode_t_s_u"]
                )
                for other in measured
            )
        ],
        key=lambda row: row[accuracy_key],
    )


def plot(rows: list[dict], accuracy_key: str, threshold: float, output: Path) -> None:
    measured = [row for row in rows if row[accuracy_key] is not None]
    frontier = pareto_frontier(rows, accuracy_key)
    annotation_offsets = {
        "top1": {
            "ds00_baseline": (8, 18),
            "ds01_attention_bfp4_lofi": (-165, 8),
            "ds02_attention_bfp4_hifi2": (8, 8),
            "ds03_expert_bfp4_hifi2": (8, -18),
            "ds06_kv_bf16": (-120, 18),
            "ds10_attention_bfp8_hifi2": (-180, -2),
            "ds11_lm_head_lofi": (8, -14),
            "ds12_attention_hifi2_lm_head_lofi": (-245, -20),
        },
        "top5": {
            "ds00_baseline": (-120, 36),
            "ds01_attention_bfp4_lofi": (-235, 20),
            "ds02_attention_bfp4_hifi2": (-235, -10),
            "ds03_expert_bfp4_hifi2": (-175, -36),
            "ds06_kv_bf16": (8, 30),
            "ds10_attention_bfp8_hifi2": (8, 10),
            "ds11_lm_head_lofi": (-150, -18),
            "ds12_attention_hifi2_lm_head_lofi": (8, -38),
        },
    }
    fig, axis = plt.subplots(figsize=(11, 7))
    for row in measured:
        selected = row["selected"]
        axis.scatter(
            row[accuracy_key],
            row["trace_verified_teacher_forcing_decode_t_s_u"],
            c="red" if selected else "#3572A5",
            s=125 if selected else 65,
            marker="*" if selected else "o",
            zorder=3,
        )
        axis.annotate(
            row["config_id"],
            (row[accuracy_key], row["trace_verified_teacher_forcing_decode_t_s_u"]),
            xytext=annotation_offsets.get(accuracy_key, {}).get(row["config_id"], (6, 6)),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#888888", "linewidth": 0.5},
            bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    axis.plot(
        [row[accuracy_key] for row in frontier],
        [row["trace_verified_teacher_forcing_decode_t_s_u"] for row in frontier],
        color="#222222",
        linewidth=1.5,
        label="Pareto frontier",
    )
    axis.axvline(threshold, color="#555555", linestyle=":", linewidth=2, label=f"minimum {threshold:.0%}")
    axis.set_xlabel(accuracy_key.replace("top", "Top-") + " accuracy")
    axis.set_ylabel("Trace-verified teacher-forcing decode (tokens/s/user)")
    axis.set_title(f"GPT-OSS 120B datatype sweep: {accuracy_key.replace('top', 'Top-')} / performance")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_baseline_refresh() -> dict:
    path = ROOT.parent / "optimized_full_model/artifacts/teacher_forcing_readiness.json"
    stats = json.loads(path.read_text(encoding="utf-8"))[0]
    selected_path = ROOT / "selected_precision_config.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if stats["precision_config_id"] != selected["config_id"]:
        raise RuntimeError("default selected refresh config id does not match selected_precision_config.json")
    if stats["precision_config_sha256"] != semantic_json_sha256(selected):
        raise RuntimeError("default selected refresh config hash does not match selected_precision_config.json")
    if stats["precision_config_path"] != repo_path(selected_path):
        raise RuntimeError("default selected refresh did not use the normal selected-config construction path")
    validate_trace_measurement(stats["trace_measurement"])
    if len(stats["warmed_repetitions"]) != 2:
        raise RuntimeError("default selected refresh requires two warmed repetitions")
    return {
        "artifact": repo_path(path),
        "config_id": stats["precision_config_id"],
        "top1": stats["top1"],
        "top5": stats["top5"],
        "top100": stats["top100"],
        "ttft_ms": stats["ttft_ms"],
        "trace_verified_teacher_forcing_decode_t_s_u": stats["decode_t/s/u"],
        "trace_measurement": stats["trace_measurement"],
        "warmed_repetitions": stats["warmed_repetitions"],
        **measurement_provenance(stats),
    }


def main() -> None:
    rows = load_results()
    eligible = [row for row in rows if row["pass"] and row["trace_verified"]]
    winner = max(eligible, key=lambda row: row["trace_verified_teacher_forcing_decode_t_s_u"])
    if not winner["selected"]:
        raise RuntimeError(f"selected config is not fastest passing candidate: expected {winner['config_id']}")
    payload = {
        "schema_version": 2,
        "reference": REFERENCE_PATH,
        "reference_revision": REFERENCE_REVISION,
        "prompt_tokens": PROMPT_TOKENS,
        "generated_tokens": GENERATED_TOKENS,
        "thresholds": {"top1_min": TOP1_MIN, "top5_min": TOP5_MIN, "top100_recorded_not_gated": True},
        "ranking_metric": "maximum trace-verified warmed teacher-forcing decode tokens/s/user among accuracy-gated configs",
        "selected_config_id": winner["config_id"],
        "baseline_optimized_full_model_refresh": load_baseline_refresh(),
        "results": rows,
    }
    (ROOT / "sweep_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fields = [
        "schema_version",
        "config_id",
        "config_path",
        "config_sha256",
        "layer_exceptions",
        "dtype_policy",
        "compute_fidelity_policy",
        "top1",
        "top5",
        "top100",
        "ttft_ms",
        "trace_verified_teacher_forcing_decode_t_s_u",
        "trace_verified",
        "trace_measurement",
        "measurement_regime",
        "prompt_tokens",
        "generated_tokens",
        "reference_path",
        "reference_revision",
        "command",
        "hardware",
        "mesh",
        "source_branch",
        "source_commit",
        "source_state_sha256",
        "source_dirty",
        "runtime_source_files",
        "runner_provenance",
        "environment_notes",
        "status",
        "pass",
        "selected",
        "rejection_reason",
        "evidence",
        "runtime_consumption_evidence",
    ]
    with (ROOT / "sweep_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key in (
                "layer_exceptions",
                "dtype_policy",
                "compute_fidelity_policy",
                "trace_measurement",
                "runtime_source_files",
                "runner_provenance",
            ):
                flat[key] = json.dumps(flat[key], sort_keys=True) if flat.get(key) is not None else None
            flat["mesh"] = "x".join(map(str, flat["mesh"]))
            writer.writerow({field: flat.get(field) for field in fields})
    plot(rows, "top1", TOP1_MIN, ROOT / "top1_perf_pareto.png")
    plot(rows, "top5", TOP5_MIN, ROOT / "top5_perf_pareto.png")


if __name__ == "__main__":
    main()

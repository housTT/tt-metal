#!/usr/bin/env python3
"""Build the Qwen3.6 datatype-sweep ledger and Pareto plots from raw evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from models.autoports.qwen_qwen3_6_27b.tt.precision import load_precision_policy


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT.parents[1]
REFERENCE = MODEL_DIR / "readiness_aime24_chat.refpt"
THRESHOLDS = {"top1": 0.90, "top5": 0.98}
HARDWARE = "4x Blackhole P300c"
MESH = "1x4 TP physical Ring"
RUNTIME_GIT_BRANCH = "agentic-research/hous/qwen3.8-27b"
RUNTIME_GIT_BASE_COMMIT = "b7b52f83305e1e7c350bde15c7b648d43652e4e2"
BASELINE_MEASURED_SOURCE_SHA256 = "6ab2b2093de4652e7e358cc2afda422f327b4cccf8a0778e85d52671bd46f2a7"
RUNTIME_ENVIRONMENT = (
    "TT_METAL_HOME=/home/ttuser/dev/tt-metal; Python 3.12.3; torch 2.11.0+cpu; "
    "TTNN Python package from /home/ttuser/dev/tt-metal/ttnn; firmware 19.11.0"
)
RUNTIME_SOURCE_MANIFEST = ROOT / "source_provenance.json"
RUNTIME_SOURCE_FILES = (
    MODEL_DIR / "tt/precision.py",
    MODEL_DIR / "tt/model.py",
    MODEL_DIR / "tt/generator.py",
    MODEL_DIR / "tt/multichip_decoder.py",
    MODEL_DIR / "tt/optimized_decoder.py",
    Path("models/common/readiness_check/run_prefill_check.py"),
    Path("models/common/readiness_check/run_teacher_forcing.py"),
)
BASE_COMMAND = (
    "TT_METAL_HOME=/home/ttuser/dev/tt-metal "
    "PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal:/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal "
    "{policy_env}python -m models.common.readiness_check.run_teacher_forcing "
    "--model-dir models/autoports/qwen_qwen3_6_27b "
    "--reference models/autoports/qwen_qwen3_6_27b/readiness_aime24_chat.refpt "
    "--mesh-device P300 --fabric-config FABRIC_1D_RING --trace-region-size 1500000000 "
    "--warmup-repeats 1 --output-json {output}"
)


def command(policy_path: Path | None, output: Path) -> str:
    policy_env = "" if policy_path is None else f"QWEN36_PRECISION_CONFIG={policy_path} "
    return BASE_COMMAND.format(policy_env=policy_env, output=output)


def final_runtime_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in RUNTIME_SOURCE_FILES:
        resolved = path.resolve()
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_cohort(config_id: str) -> str:
    if config_id == "baseline_optimized_mixed":
        return "final_runtime_v3"
    if config_id == "all_mlp_bfp4_hifi2":
        return "strict_policy_v2"
    return "pre_strict_policy_v1"


def source_state(config_id: str) -> dict[str, str]:
    cohort = source_cohort(config_id)
    equivalence = {
        "pre_strict_policy_v1": (
            "Measured before host-only exact-schema/KV-block propagation fixes. "
            "All recorded policies use page block 64 and valid known keys; replacing "
            "PAGE_BLOCK_SIZE with the validated value 64 is algebraically identical. "
            "The later HiFi4 projection dispatch branch is unreachable because no row "
            "uses projection HiFi4. Exact root/nested-key rejection changes invalid-input "
            "handling only. Thus measured TT operations are unchanged by final source."
        ),
        "strict_policy_v2": (
            "Measured after KV layout/block and logits strict validation/propagation, "
            "before exact root/nested-key rejection. The candidate contains only known "
            "valid keys, so the final validation-only change does not alter construction "
            "or any TT operation."
        ),
        "final_runtime_v3": (
            "Measured after runtime and exact root/nested-schema remediation. It "
            "predates only canonical-lowercase rejection for datatype/fidelity strings; "
            "the embedded frozen baseline policy is entirely lowercase, so that "
            "validation-only change cannot alter construction or TT operations. The raw "
            "prefill and teacher-forcing metrics embed the resolved baseline policy."
        ),
    }[cohort]
    return {
        "measurement_source_cohort": cohort,
        "runtime_source_identifier": f"dirty-cohort:{cohort}@{RUNTIME_GIT_BASE_COMMIT}",
        "runtime_source_state": (
            "stage-owned runtime changes were uncommitted at measurement time; the exact "
            "historical dirty patch was not preserved. This limitation is explicit and "
            "the row-to-final behavior-equivalence basis is recorded"
        ),
        "runtime_source_equivalence": equivalence,
        "runtime_source_manifest": str(RUNTIME_SOURCE_MANIFEST),
    }


def result_from_metrics(metrics_path: Path, policy_path: Path | None) -> dict:
    report = json.loads(metrics_path.read_text())
    aggregate = report["aggregate"]
    repeat_paths = sorted(metrics_path.parent.glob("teacher_forcing_metrics_repeat*.json"))
    repeat_reports = [json.loads(path.read_text()) for path in repeat_paths]
    repeat_aggregates = [aggregate] + [item["aggregate"] for item in repeat_reports]
    summary = report.get("precision_summary")
    if summary is None:
        summary = load_precision_policy(policy_path).summary()
    trace_verified = report["runtime"].get("decode_trace_enabled") is True
    gate_pass = (
        aggregate["top1"] >= THRESHOLDS["top1"]
        and aggregate["top5"] >= THRESHOLDS["top5"]
        and trace_verified
    )
    return {
        "config_id": summary["config_id"],
        "precision_config_path": summary["path"],
        "dtype_policy": {
            "weight_groups": summary["weight_groups"],
            "layer_exceptions": summary["layer_exceptions"],
            "activation_residual": summary["activation_residual"],
            "ccl": summary["ccl"],
            "kv_cache": summary["kv_cache"],
            "logits_sampling": summary["logits_sampling"],
        },
        "compute_fidelity_policy": summary["compute_fidelities"],
        "top1": aggregate["top1"],
        "top5": aggregate["top5"],
        "top100": aggregate["top100"],
        "tokens": int(aggregate["total"]),
        "ttft_ms": sum(item["ttft_ms"] for item in repeat_aggregates) / len(repeat_aggregates),
        "teacher_forcing_decode_t_s_u": sum(
            item["decode_t/s/u"] for item in repeat_aggregates
        )
        / len(repeat_aggregates),
        "teacher_forcing_decode_t_s_u_repeats": [item["decode_t/s/u"] for item in repeat_aggregates],
        "full_model_measurement_runs": len(repeat_aggregates),
        "teacher_forcing_decode_tokens": int(aggregate["decode_tokens"]),
        "trace_verified": trace_verified,
        "measurement_regime": (
            "AIME24 chat-template prompt 0; 100-token full-model teacher forcing; "
            "one full-reference warmup per run; steady post-capture traced replay; "
            f"{len(repeat_aggregates)} measurement run(s)"
        ),
        "reference_path": str(REFERENCE),
        "command": command(policy_path, metrics_path),
        "hardware": HARDWARE,
        "mesh": MESH,
        "runtime_git_branch": RUNTIME_GIT_BRANCH,
        "runtime_git_base_commit": RUNTIME_GIT_BASE_COMMIT,
        **source_state(summary["config_id"]),
        "runtime_environment_notes": RUNTIME_ENVIRONMENT,
        "accuracy_gate": "pass" if gate_pass else "fail",
        "run_status": "completed",
        "evidence_path": str(metrics_path),
    }


def pareto_frontier(points: list[dict], accuracy_key: str) -> list[dict]:
    frontier = []
    for point in sorted(points, key=lambda p: (p[accuracy_key], p["teacher_forcing_decode_t_s_u"])):
        if any(
            other[accuracy_key] >= point[accuracy_key]
            and other["teacher_forcing_decode_t_s_u"] >= point["teacher_forcing_decode_t_s_u"]
            and (
                other[accuracy_key] > point[accuracy_key]
                or other["teacher_forcing_decode_t_s_u"] > point["teacher_forcing_decode_t_s_u"]
            )
            for other in points
        ):
            continue
        frontier.append(point)
    return sorted(frontier, key=lambda p: p[accuracy_key])


def plot(points: list[dict], accuracy_key: str, selected_id: str, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    x = [100 * point[accuracy_key] for point in points]
    y = [point["teacher_forcing_decode_t_s_u"] for point in points]
    axis.scatter(x, y, s=70, color="#2878B5", edgecolor="white", linewidth=0.8, zorder=3)
    frontier = pareto_frontier(points, accuracy_key)
    axis.plot(
        [100 * point[accuracy_key] for point in frontier],
        [point["teacher_forcing_decode_t_s_u"] for point in frontier],
        color="#2A9D8F",
        marker="o",
        linewidth=2.2,
        label="evaluated Pareto frontier",
        zorder=2,
    )
    label_offsets = (
        {
            "full_down_bfp4_lofi": (15, 16),
            "baseline_optimized_mixed": (15, 2),
            "ccl_all_bfp8": (15, -12),
            "kv_bf16_control": (15, -26),
            "full_down_bfp8_hifi2": (15, -24),
            "projection_hifi2": (15, -2),
            "canonical_runnable_bfp8_hifi2_kv_bf16": (15, 6),
        }
        if accuracy_key == "top5"
        else {
            "baseline_optimized_mixed": (8, 12),
            "ccl_all_bfp8": (8, -1),
            "kv_bf16_control": (8, -13),
            "full_down_bfp4_lofi": (8, 12),
        }
    )
    for point in points:
        selected = point["config_id"] == selected_id
        if selected:
            axis.scatter(
                [100 * point[accuracy_key]],
                [point["teacher_forcing_decode_t_s_u"]],
                s=150,
                color="red",
                edgecolor="white",
                linewidth=1.2,
                label="selected",
                zorder=5,
            )
        axis.annotate(
            point["config_id"],
            (100 * point[accuracy_key], point["teacher_forcing_decode_t_s_u"]),
            xytext=label_offsets.get(point["config_id"], (8, 5)),
            textcoords="offset points",
            fontsize=8,
        )
    threshold = 100 * THRESHOLDS[accuracy_key]
    axis.axvline(threshold, color="#555555", linestyle=":", linewidth=1.8, label=f"minimum {threshold:.0f}%")
    axis.set_xlabel(f"Full-model {accuracy_key.replace('top', 'top-')} accuracy (%)")
    axis.set_ylabel("Steady traced teacher-forcing decode (tokens/s/user)")
    axis.set_title(f"Qwen3.6-27B {accuracy_key.replace('top', 'Top-')} accuracy / traced decode Pareto")
    axis.margins(y=0.12)
    axis.grid(alpha=0.22)
    axis.legend(loc="best")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    baseline_metrics = ROOT / "evidence/baseline/teacher_forcing_metrics.json"
    baseline_policy = ROOT / "candidates/baseline_optimized_mixed.json"
    results = [result_from_metrics(baseline_metrics, baseline_policy)]
    candidate_root = ROOT / "evidence/candidates"
    for metrics_path in sorted(candidate_root.glob("*/teacher_forcing_metrics.json")):
        policy_path = ROOT / "candidates" / f"{metrics_path.parent.name}.json"
        results.append(result_from_metrics(metrics_path, policy_path))

    failed_results = []
    for failure_path in sorted(candidate_root.glob("*/failure.json")):
        failure = json.loads(failure_path.read_text())
        policy_path = ROOT / "candidates" / f"{failure_path.parent.name}.json"
        summary = load_precision_policy(policy_path).summary()
        failed_results.append(
            {
                "config_id": summary["config_id"],
                "precision_config_path": summary["path"],
                "dtype_policy": {
                    "weight_groups": summary["weight_groups"],
                    "layer_exceptions": summary["layer_exceptions"],
                    "activation_residual": summary["activation_residual"],
                    "ccl": summary["ccl"],
                    "kv_cache": summary["kv_cache"],
                    "logits_sampling": summary["logits_sampling"],
                },
                "compute_fidelity_policy": summary["compute_fidelities"],
                "top1": None,
                "top5": None,
                "top100": None,
                "tokens": 0,
                "ttft_ms": None,
                "teacher_forcing_decode_t_s_u": None,
                "teacher_forcing_decode_tokens": 0,
                "trace_verified": False,
                "measurement_regime": (
                    "AIME24 chat-template prompt 0; full-model traced teacher-forcing "
                    "warmup failed before accuracy measurement"
                ),
                "reference_path": str(REFERENCE),
                "command": failure["command"],
                "hardware": HARDWARE,
                "mesh": MESH,
                "runtime_git_branch": RUNTIME_GIT_BRANCH,
                "runtime_git_base_commit": RUNTIME_GIT_BASE_COMMIT,
                **source_state(summary["config_id"]),
                "runtime_environment_notes": RUNTIME_ENVIRONMENT,
                "accuracy_gate": "not_evaluated_runtime_failure",
                "run_status": "failed",
                "failure": failure["failure"],
                "selection_status": "rejected_runtime_failure",
                "evidence_path": str(failure_path),
            }
        )

    selected_id = max(
        (result for result in results if result["accuracy_gate"] == "pass"),
        key=lambda result: result["teacher_forcing_decode_t_s_u"],
    )["config_id"]
    for result in results:
        result["selection_status"] = (
            "selected" if result["config_id"] == selected_id else "rejected_or_control"
        )
        if result["config_id"] == selected_id:
            token_out_path = ROOT / "evidence/final/token_out_metrics.json"
            token_out = json.loads(token_out_path.read_text())
            result["post_selection_token_out"] = {
                "device_only_no_readback_t_s_u": token_out["selected"]["token_out_t/s/u"],
                "device_only_no_readback_ms": token_out["selected"]["combined_trace_ms"],
                "caller_visible_t_s_u": token_out["representative_token_out"]["token_out_t/s/u"],
                "caller_visible_decode_ms_per_token": token_out["representative_token_out"]["decode_ms_per_token"],
                "caller_visible_ttft_ms": token_out["representative_token_out"]["ttft_ms"],
                "prompt_tokens": token_out["representative_token_out"]["prompt_tokens"],
                "generated_tokens": token_out["representative_token_out"]["generated_tokens"],
                "measurement_regime": (
                    "normal default selected-config construction; warmed full-64-layer "
                    "model+distributed local-argmax sampler trace; no host readback in "
                    "device-only rank; separate caller-visible prompt-128/generate-128 run"
                ),
                "evidence_path": str(token_out_path),
            }

    all_results = results + failed_results

    final_source_hash = final_runtime_source_sha256()
    for result in all_results:
        result["final_runtime_source_sha256"] = final_source_hash
        if result["measurement_source_cohort"] == "final_runtime_v3":
            result["runtime_source_identifier"] = f"sha256:{BASELINE_MEASURED_SOURCE_SHA256}"
            result["runtime_source_state"] = (
                "the measured runtime source SHA-256 was recorded and independently "
                "recomputed before the later host-only canonical-spelling validation; "
                "the raw metric embeds the resolved precision policy"
            )

    cohorts = {}
    for result in all_results:
        cohort = result["measurement_source_cohort"]
        cohorts.setdefault(
            cohort,
            {
                "row_config_ids": [],
                "source_identifier": result["runtime_source_identifier"],
                "exact_historical_dirty_patch_preserved": False,
                "source_state": result["runtime_source_state"],
                "equivalence_to_final_runtime": result["runtime_source_equivalence"],
            },
        )["row_config_ids"].append(result["config_id"])
    source_manifest = {
        "schema_version": 1,
        "git_branch": RUNTIME_GIT_BRANCH,
        "git_base_commit": RUNTIME_GIT_BASE_COMMIT,
        "final_runtime_source_sha256": final_source_hash,
        "final_runtime_source_files": [str(path) for path in RUNTIME_SOURCE_FILES],
        "cohorts": cohorts,
        "unrelated_dirty_state_excluded": [
            "tt_metal/third_party/tracy",
            "tt_metal/third_party/umd",
            "tt_metal/third_party/tt-cluster-descriptors/",
        ],
        "limitation": (
            "The exact pre-final uncommitted patch objects were not preserved. This "
            "manifest does not claim otherwise; it records row cohorts and the exact "
            "behavior-equivalence argument for host-only changes made after measurement."
        ),
    }
    RUNTIME_SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2) + "\n")

    (ROOT / "sweep_results.json").write_text(json.dumps(all_results, indent=2) + "\n")
    csv_fields = [
        "config_id", "precision_config_path", "dtype_policy", "compute_fidelity_policy",
        "top1", "top5", "top100", "tokens", "ttft_ms",
        "teacher_forcing_decode_t_s_u", "teacher_forcing_decode_t_s_u_repeats",
        "full_model_measurement_runs",
        "teacher_forcing_decode_tokens", "trace_verified", "measurement_regime", "reference_path",
        "command", "hardware", "mesh", "runtime_git_branch",
        "runtime_git_base_commit", "runtime_source_state",
        "measurement_source_cohort", "runtime_source_identifier",
        "runtime_source_equivalence", "runtime_source_manifest",
        "final_runtime_source_sha256", "runtime_environment_notes",
        "accuracy_gate", "run_status",
        "selection_status", "post_selection_token_out", "failure", "evidence_path",
    ]
    with (ROOT / "sweep_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        for result in all_results:
            row = dict(result)
            row.setdefault("failure", "")
            row["dtype_policy"] = json.dumps(row["dtype_policy"], sort_keys=True)
            row["compute_fidelity_policy"] = json.dumps(row["compute_fidelity_policy"], sort_keys=True)
            if isinstance(row.get("teacher_forcing_decode_t_s_u_repeats"), list):
                row["teacher_forcing_decode_t_s_u_repeats"] = json.dumps(row["teacher_forcing_decode_t_s_u_repeats"])
            if isinstance(row.get("post_selection_token_out"), dict):
                row["post_selection_token_out"] = json.dumps(row["post_selection_token_out"], sort_keys=True)
            writer.writerow(row)
    plot(results, "top1", selected_id, ROOT / "top1_perf_pareto.png")
    plot(results, "top5", selected_id, ROOT / "top5_perf_pareto.png")
    print(
        f"wrote {len(results)} completed points and {len(failed_results)} runtime failures; "
        f"selected={selected_id}"
    )


if __name__ == "__main__":
    main()

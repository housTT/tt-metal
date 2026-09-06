# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build the checked-in Gemma-4 datatype-sweep tables and Pareto charts."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[4]
CONFIGS = ROOT / "configs"
ARTIFACTS = ROOT / "artifacts"
PROFILES = ("P150", "P150x2", "P150x4")
PROFILE_TP = {"P150": 1, "P150x2": 2, "P150x4": 4}
SELECTED_ID = "selected_canonical_profile_policy"
TP4_RANKING_SAMPLES = {
    SELECTED_ID: [
        ARTIFACTS / "selected_default_readiness" / "readiness_tp4.json",
        ARTIFACTS / "repeats" / "experts_bfp4_lofi_tp4_final1" / "readiness_tp4.json",
        ARTIFACTS / "repeats" / "experts_bfp4_lofi_tp4_final2" / "readiness_tp4.json",
    ],
    "experts_bfp4_lofi": [
        ARTIFACTS / "experts_bfp4_lofi" / "readiness_tp4.json",
        ARTIFACTS / "repeats" / "experts_bfp4_lofi_tp4_final1" / "readiness_tp4.json",
        ARTIFACTS / "repeats" / "experts_bfp4_lofi_tp4_final2" / "readiness_tp4.json",
    ],
    "decode_dense_gate_up_bfp4_lofi": [
        ARTIFACTS / "decode_dense_gate_up_bfp4_lofi" / "readiness_tp4.json",
        ARTIFACTS / "repeats" / "decode_dense_gate_up_bfp4_lofi_tp4_final" / "readiness_tp4.json",
        ARTIFACTS / "repeats" / "decode_dense_gate_up_bfp4_lofi_tp4_final2" / "readiness_tp4.json",
    ],
    "full_attention_decode_lofi": [
        ARTIFACTS / "full_attention_decode_lofi" / "readiness_tp4.json",
        ARTIFACTS / "repeats" / "full_attention_decode_lofi_tp4_final1" / "readiness_tp4.json",
        ARTIFACTS / "repeats" / "full_attention_decode_lofi_tp4_final2" / "readiness_tp4.json",
    ],
}


def exact_readiness_command(path, report, selected):
    profile_id = report["profile"].lower()
    test = (
        "models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::"
        f"test_reduced_real_weight_full_model_probe[blackhole-{profile_id}]"
    )
    output_dir = path.parent.relative_to(REPO_ROOT)
    config = "" if selected else f" GEMMA4_PRECISION_CONFIG={CONFIGS.relative_to(REPO_ROOT)}/{report['config_id']}.json"
    return (
        f"cd {REPO_ROOT} && env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 "
        "TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\":true}'"
        f"{config} GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_STACK_PROBE=1 "
        "GEMMA4_READINESS_REFERENCE=models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/artifacts/"
        f"gemma4_aime24_chat.refpt GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR={output_dir} "
        f"python_env/bin/pytest -q -s '{test}' --junitxml={output_dir}/junit.xml"
    )


def deep_merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_policy(path):
    policy = json.loads(path.read_text())
    if "extends" in policy:
        base = load_policy(path.parent / policy["extends"])
        policy = deep_merge(base, policy.get("overrides", {}))
        policy["config_id"] = json.loads(path.read_text())["config_id"]
    return policy


def profile_policy(policy, profile):
    resolved = deepcopy(policy)
    override = resolved.pop("profile_overrides", {}).get(profile, {})
    return deep_merge(resolved, override)


def compact_precision_summary(summary):
    compact = {key: value for key, value in summary.items() if key != "layers"}
    signatures = {}
    for layer in summary["layers"]:
        values = {key: value for key, value in layer.items() if key != "layer"}
        signature_key = json.dumps(values, sort_keys=True)
        signatures.setdefault(signature_key, []).append(layer["layer"])
    compact["layer_policy_signatures"] = [
        {"layers": layers, **json.loads(signature)} for signature, layers in signatures.items()
    ]
    return compact


def result_rows():
    paths = sorted(ARTIFACTS.glob("*/readiness_tp*.json"))
    rows = []
    selected_token_out = {
        json.loads(path.read_text())["profile"]: json.loads(path.read_text())
        for path in sorted((ARTIFACTS / "selected_token_out").glob("token_out_trace_tp*.json"))
    }
    selected_prefill = {
        json.loads(path.read_text())["profile"]: {
            **json.loads(path.read_text()),
            "artifact": str(path.relative_to(ROOT)),
        }
        for path in sorted((ARTIFACTS / "selected_token_out").glob("prefill_tp*.json"))
    }
    for path in paths:
        report = json.loads(path.read_text())
        config_id = report["config_id"]
        selected = path.parent.name == "selected_default_readiness"
        if selected:
            policy_path = ROOT / "selected_precision_config.json"
        else:
            policy_path = CONFIGS / f"{config_id}.json"
        policy = profile_policy(load_policy(policy_path), report["profile"])
        accuracy = report["teacher_forcing_accuracy"]
        perf = report["teacher_forcing_performance"]
        ranking_samples = []
        if report["profile"] == "P150x4" and config_id in TP4_RANKING_SAMPLES:
            ranking_samples = [
                json.loads(sample.read_text())["teacher_forcing_performance"]["decode_t/s/u"]
                for sample in TP4_RANKING_SAMPLES[config_id]
            ]
        ranking_perf = statistics.median(ranking_samples) if ranking_samples else perf["decode_t/s/u"]
        token_out = selected_token_out.get(report["profile"]) if selected else None
        prefill = selected_prefill.get(report["profile"]) if selected else None
        rows.append(
            {
                "config_id": config_id,
                "profile": report["profile"],
                "dtype_policy": {
                    key: policy.get(key, {})
                    for key in (
                        "weight_groups",
                        "decode_weight_groups",
                        "embedding",
                        "activation_residual",
                        "ccl",
                        "kv_cache",
                        "logits_sampling",
                        "layer_exceptions",
                    )
                    if key in policy
                },
                "compute_fidelity_policy": policy.get("compute_fidelities", {}),
                "top1": accuracy["top1"],
                "top5": accuracy["top5"],
                "top100": accuracy["top100"],
                "token_count": accuracy["total"],
                "prefill_top1": report["prefill_accuracy"]["top1"],
                "prefill_top5": report["prefill_accuracy"]["top5"],
                "prefill_top100": report["prefill_accuracy"]["top100"],
                "ttft_ms": perf["ttft_ms"],
                "teacher_forcing_decode_t_s_u": ranking_perf,
                "teacher_forcing_decode_raw_t_s_u": perf["decode_t/s/u"],
                "ranking_samples_t_s_u": ranking_samples,
                "teacher_forcing_warmed_decode_t_s_u": perf["warmed_decode_t/s/u"],
                "trace_replays": int(perf["trace_replays"]),
                "trace_verified": report["trace_verified"] and int(perf["trace_replays"]) == 99,
                "measurement_regime": report["measurement_regime"],
                "command": exact_readiness_command(path, report, selected),
                "hardware": report["hardware"],
                "mesh": report["mesh_shape"],
                "reference": report["reference"],
                "reference_sha256": report["reference_sha256"],
                "pass_fail": report["verdict"],
                "selected": selected,
                "runtime_consumption_evidence": compact_precision_summary(report["precision_summary"]),
                "post_selection_token_out": (
                    {
                        "prefill_initial_ttft_ms": prefill["initial_ttft_ms"],
                        "prefill_warmed_ttft_ms": prefill["warmed_ttft_ms"],
                        "prefill_artifact": prefill["artifact"],
                        "decode_t_s_u": token_out["decode_t_s_u"],
                        "warmups": token_out["warmups"],
                        "iterations": token_out["iterations"],
                        "trace_replays": token_out["trace_replays"],
                        "token_readbacks": token_out["token_readbacks"],
                        "artifact": str(
                            Path("artifacts")
                            / "selected_token_out"
                            / f"token_out_trace_tp{PROFILE_TP[report['profile']]}.json"
                        ),
                    }
                    if token_out
                    else None
                ),
                "artifact": str(path.relative_to(ROOT)),
            }
        )
    return rows


def write_csv(rows):
    fields = [
        "config_id",
        "profile",
        "dtype_policy",
        "compute_fidelity_policy",
        "top1",
        "top5",
        "top100",
        "token_count",
        "prefill_top1",
        "prefill_top5",
        "prefill_top100",
        "ttft_ms",
        "teacher_forcing_decode_t_s_u",
        "teacher_forcing_decode_raw_t_s_u",
        "ranking_samples_t_s_u",
        "teacher_forcing_warmed_decode_t_s_u",
        "trace_replays",
        "trace_verified",
        "measurement_regime",
        "command",
        "hardware",
        "mesh",
        "reference",
        "reference_sha256",
        "pass_fail",
        "selected",
        "runtime_consumption_evidence",
        "post_selection_token_out",
        "artifact",
    ]
    with (ROOT / "sweep_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def pareto(rows, accuracy_key, output):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for axis, profile in zip(axes, PROFILES):
        profile_rows = [row for row in rows if row["profile"] == profile]
        for row in profile_rows:
            axis.scatter(
                100 * row[accuracy_key],
                row["teacher_forcing_decode_t_s_u"],
                color="red" if row["selected"] else ("tab:blue" if row["pass_fail"] == "pass" else "gray"),
                marker="*" if row["selected"] else "o",
                s=180 if row["selected"] else 45,
                zorder=4 if row["selected"] else 2,
            )
            label = row["config_id"].replace("selected_canonical_profile_policy", "selected")
            offset = (4, 4)
            arrow = None
            tp4_labels = {
                "selected_canonical_profile_policy": ("selected: expert BFP4+LoFi", (-175, -8)),
                "canonical_baseline": ("canonical expert BFP8+LoFi", (-175, -30)),
                "full_dense_bfp8_lofi": ("full dense BFP8+LoFi", (-175, -52)),
            }
            annotate = row["selected"] or row["pass_fail"] == "fail"
            if profile == "P150x4" and row["config_id"] in tp4_labels:
                label, offset = tp4_labels[row["config_id"]]
                arrow = {"arrowstyle": "-", "color": "0.35", "linewidth": 0.6}
                annotate = not (row["config_id"] == "canonical_baseline" and row["selected"])
            if annotate:
                axis.annotate(
                    label,
                    (100 * row[accuracy_key], row["teacher_forcing_decode_t_s_u"]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=8,
                    arrowprops=arrow,
                )
        frontier = [
            row
            for row in profile_rows
            if not any(
                other[accuracy_key] >= row[accuracy_key]
                and other["teacher_forcing_decode_t_s_u"] >= row["teacher_forcing_decode_t_s_u"]
                and (
                    other[accuracy_key] > row[accuracy_key]
                    or other["teacher_forcing_decode_t_s_u"] > row["teacher_forcing_decode_t_s_u"]
                )
                for other in profile_rows
            )
        ]
        frontier.sort(key=lambda row: row[accuracy_key])
        axis.plot(
            [100 * row[accuracy_key] for row in frontier],
            [row["teacher_forcing_decode_t_s_u"] for row in frontier],
            color="black",
            linewidth=1.5,
            label="Pareto frontier",
        )
        threshold = 90 if accuracy_key == "top1" else 98
        axis.axvline(threshold, color="black", linestyle=":", linewidth=1.3, label=f"gate {threshold}%")
        axis.text(0.5, 1.01, profile, transform=axis.transAxes, ha="center", va="bottom", fontsize=12)
        axis.set_xlabel(f"{accuracy_key.replace('top', 'Top-')} accuracy (%)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Trace-verified teacher-forcing decode (tokens/s/user)")
    figure.tight_layout()
    figure.savefig(ROOT / output, dpi=180)
    plt.close(figure)


def main():
    rows = result_rows()
    repeats = [json.loads(path.read_text()) for path in sorted((ARTIFACTS / "repeats").glob("*/readiness_tp*.json"))]
    token_out_conflict = [
        {
            "artifact": str(path.relative_to(ROOT)),
            **json.loads(path.read_text()),
        }
        for path in sorted((ARTIFACTS / "token_out_compare").glob("*/token_out_trace_tp*.json"))
    ]
    payload = {
        "schema_version": 2,
        "model": "google/gemma-4-26B-A4B-it",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "accuracy_gates": {"top1": 0.90, "top5": 0.98, "top100": 1.0},
        "ranking_metric": "trace-verified teacher-forcing decode_t/s/u",
        "selected_config_id": SELECTED_ID,
        "results": rows,
        "supporting_repeats": repeats,
        "token_out_conflict_resolution": token_out_conflict,
    }
    # Consumption summaries make the aggregate large; compact serialization
    # preserves the complete schema while staying below the repository's
    # checked-in file-size limit.
    (ROOT / "sweep_results.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    write_csv(rows)
    pareto(rows, "top1", "top1_perf_pareto.png")
    pareto(rows, "top5", "top5_perf_pareto.png")


if __name__ == "__main__":
    main()

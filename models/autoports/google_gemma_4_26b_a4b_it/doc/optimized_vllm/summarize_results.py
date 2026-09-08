#!/usr/bin/env python3
"""Reconcile same-workload serving results without importing the device runtime."""

import json
from pathlib import Path

DOC = Path(__file__).resolve().parent
MODEL = DOC.parents[1]
PROFILES = ("P150", "P150x2", "P150x4")


def read(path):
    return json.loads(path.read_text())


def main():
    results = {}
    for tp, profile in zip((1, 2, 4), PROFILES):
        root = MODEL / "readiness_vllm" / profile / "optimized_vllm"
        before_manifest = read(root / "before_warmed/run_manifest.json")
        after_manifest = read(root / "after_warmed/run_manifest.json")
        for manifest in (before_manifest, after_manifest):
            assert manifest["server_runner_exit_code"] == 0, (profile, manifest["phase"])
            assert all(command["exit_code"] == 0 for command in manifest["commands"][1:])
        for field in (
            "mesh",
            "max_model_len",
            "max_num_seqs",
            "tt_config",
            "sampling_mode",
            "async_scheduling",
            "vllm_worker_sha256",
            "vllm_benchmark_sha256",
            "fabric_router_sha256",
        ):
            assert before_manifest[field] == after_manifest[field], (profile, field)
        for name in ("model.py", "generator.py"):
            assert before_manifest["source_hashes"][name] == after_manifest["source_hashes"][name], (profile, name)
        row = {
            "serving_config": {
                key: after_manifest[key]
                for key in ("mesh", "max_model_len", "max_num_seqs", "tt_config", "sampling_mode", "async_scheduling")
            }
        }
        for workload, filename in (
            ("primary_single_user", "vllm_benchmark.json"),
            ("ci_serving_burst", "vllm_ci_serving_benchmark.json"),
        ):
            phases = {phase: read(root / f"{phase}_warmed" / filename) for phase in ("before", "after")}
            assert phases["before"]["config"] == phases["after"]["config"], (profile, workload)
            for phase, metrics in phases.items():
                assert metrics["completed_requests"] == metrics["config"]["num_requests"]
                assert metrics["missing_output_tokens"] == 0
                warmup_index = metrics["command"].index("--num-warmups")
                assert metrics["command"][warmup_index + 1] == "1"
                log_name = (
                    "vllm_benchmark.log" if workload == "primary_single_user" else "vllm_ci_serving_benchmark.log"
                )
                warmup_log = (root / f"{phase}_warmed" / log_name).read_text()
                assert "Successful warmup requests: 1/1" in warmup_log, (profile, workload, phase)
                metrics["explicit_warmup_requests"] = 1
                metrics["evidence_file"] = str((root / f"{phase}_warmed" / filename).relative_to(MODEL))
            row[workload] = phases
        control_path = MODEL / f"doc/datatype_sweep/artifacts/selected_token_out/token_out_trace_tp{tp}.json"
        row["full_model_token_out_control"] = {
            key: value for key, value in read(control_path).items() if key != "precision_summary"
        }
        row["full_model_token_out_control"]["evidence_file"] = str(control_path.relative_to(MODEL))
        primary = row["primary_single_user"]["after"]
        full_model_ms = row["full_model_token_out_control"]["ms_per_token"]
        row["performance_accounting"] = {
            "workload": primary["config"],
            "ttft_ms": primary["ttft_ms"]["mean"],
            "decode_ms_per_token_e2e": primary["tpot_ms"]["mean"],
            "decode_ms_per_token_device": None,
            "roofline_ms_per_token_estimate": None,
            "device_profile_omission_reason": "vllm_serving_profiler_disabled_to_protect_hardware",
            "full_model_ms_per_token": full_model_ms,
            "serving_minus_full_model_ms_per_token": primary["tpot_ms"]["mean"] - full_model_ms,
            "serving_to_full_model_tpot_ratio": primary["tpot_ms"]["mean"] / full_model_ms,
            "comparison_boundary": (
                "Serving: warmed B1 128 prompt / 128 output, 127 inter-token intervals, token return via plugin/API. "
                "Full model: B1 prompt 128, five warmups, 128 timed tokens at positions 134..261, "
                "on-device feedback with no timed token readback. Comparable token-out work, distinct host boundaries."
            ),
        }
        quality_root = root / ("after" if profile == "P150" else "after_warmed")
        quality_manifest = read(quality_root / "run_manifest.json")
        for field in ("source_hashes", "vllm_worker_sha256", "fabric_router_sha256"):
            assert quality_manifest[field] == after_manifest[field], (profile, "quality source", field)
        assert quality_manifest["server_runner_exit_code"] == 0
        for command in quality_manifest["commands"][1:]:
            assert command["exit_code"] == 0, (profile, command)
        assert read(quality_root / "openai_feature_checks.json")["status"] == "passed"
        assert read(quality_root / "async_overlap_state_test.json")["verdict"] == "pass"
        assert read(quality_root / "logit_determinism.json")["verdict"] == "pass"
        assert "72 passed, 1 skipped" in (quality_root / "sampling_tests.log").read_text()
        qualitative_review = read(DOC / f"qualitative_{profile}_review.json")
        assert qualitative_review["verdict"].startswith("pass"), profile
        assert qualitative_review["summary"]["outputs_read"] == 12, profile
        row["quality_evidence"] = {
            "directory": str(quality_root.relative_to(MODEL)),
            "source_matches_measured_candidate": True,
            "sampling": "72 passed, 1 expected all-vocabulary logprobs skip",
            "feature_async_logit_checks": "pass",
            "qualitative_review": f"doc/optimized_vllm/qualitative_{profile}_review.json",
        }
        results[profile] = row
    summary = {
        "model": "google/gemma-4-26B-A4B-it",
        "measured_adapter": "tt/generator_vllm.py",
        "profiles": results,
        "device_time_ms": None,
        "roofline": None,
        "profiler": None,
        "device_profile_omission_reason": "vllm_serving_profiler_disabled_to_protect_hardware",
    }
    (DOC / "perf_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build the p150 stage 10 perf_summary.json from the archived runner artifacts."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[2]
STAGE9 = MODEL / "doc/vllm_integration/p150"


def metrics(path):
    data = json.loads(path.read_text())
    return {
        "artifact": str(path.relative_to(MODEL.parents[2])),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "workload": data["config"],
        "completed_requests": data["completed_requests"],
        "ttft_ms": data["ttft_ms"],
        "tpot_ms": data["tpot_ms"],
        "itl_ms": data["itl_ms"],
        "output_throughput_tok_per_s": data["output_throughput_tok_per_s"],
        "decode_t_s_u": data["vllm_mean_tpot_decode_tps"],
    }


def main():
    before_b1 = metrics(STAGE9 / "b1_primary/first/vllm_benchmark.json")
    after_b1 = metrics(HERE / "b1_final_v3/first/vllm_benchmark.json")
    after_b1_repeat = metrics(HERE / "b1_final_v3/repeat/vllm_benchmark.json")
    before_ci = metrics(STAGE9 / "b32_final/vllm_ci_serving_benchmark.json")
    serial_ci = metrics(HERE / "serial_b32_v4/vllm_ci_serving_benchmark.json")
    after_ci = metrics(HERE / "b32_final/vllm_ci_serving_benchmark.json")
    burst = json.loads((HERE / "b32_final/vllm_prefill_burst_benchmark.json").read_text())
    probe = json.loads((HERE / "batched_probe_full_v3_final.json").read_text())
    teacher = json.loads((STAGE9 / "teacher_forcing_p150.json").read_text())
    summary = {
        "stage": "optimized-vllm",
        "profile": "p150",
        "model": "ornith-ai/Ornith-1.5-9B",
        "implementation": "tt/generator_vllm.py",
        "hardware": "one Blackhole chip of a P300c board (readiness mesh label N150)",
        "mesh": [1, 1],
        "precision": "p150_head8_hifi2_kv8_policy.json",
        "max_model_len": 262144,
        "workload": {"profile": "single_user_decode", "prompt_len": 128, "gen_len": 128, "batch": 1},
        "ttft_ms": after_b1["ttft_ms"]["p50"],
        "decode_ms_per_token_e2e": after_b1["tpot_ms"]["mean"],
        "decode_ms_per_token_device": None,
        "roofline_ms_per_token_estimate": None,
        "device_time_reason": "vllm_serving_profiler_disabled_to_protect_hardware",
        "named_limitations": [
            "Batched prefill batches fresh rows whose 128-token bucket is at most 512 tokens; longer rows run one at a time (calibrated ratios 0.75 at 8x512 and 0.97 at 4x1024).",
            "A multi-row admission costs one 59 to 63 ms boundary decode step for the plugin slot remap (24 recurrent layers, two ops per moved row).",
            "The batched 32x128 pass is host-dispatch bound (about 7 000 ops, 0.75 s); the KDA causal convolution accepts one row per launch and the chunked delta rule three rows per launch.",
            "The scheduler admits the first burst request alone, so a 32-request burst runs as a one-row and a 31-row prefill step.",
        ],
        "primary_single_user": {
            "before": before_b1,
            "after": after_b1,
            "after_repeat": after_b1_repeat,
            "ttft_change_percent": 100 * (after_b1["ttft_ms"]["p50"] / before_b1["ttft_ms"]["p50"] - 1),
            "tpot_change_percent": 100 * (after_b1["tpot_ms"]["mean"] / before_b1["tpot_ms"]["mean"] - 1),
            "gate": "TTFT p50 and TPOT mean within 5 percent of the pre-change run",
        },
        "ci_serving_burst_secondary": {
            "before_stage9": before_ci,
            "serial_reference_final_code": serial_ci,
            "after": after_ci,
        },
        "prefill_burst": {
            "artifact": str((HERE / "b32_final/vllm_prefill_burst_benchmark.json").relative_to(MODEL.parents[2])),
            "ttft_ms": burst["ttft_ms"],
            "serial_reference": burst["serial_reference"],
            "gate": "p99 ratio <= 0.6 and mean ratio <= 0.5",
        },
        "batched_equals_solo": {
            "artifact": str((HERE / "batched_probe_full_v3_final.json").relative_to(MODEL.parents[2])),
            "git_head": probe["git_head"],
            "groups": [
                {
                    "rows": g["group_size"],
                    "order": g["order"],
                    "top1_equal": g["all_top1_equal"],
                    "min_logits_pcc": g["min_logits_pcc"],
                    "kv_pages_exact": g["all_kv_exact"],
                    "state_exact": g["all_state_exact"],
                    "greedy_agree_tokens": g["min_greedy_agree"],
                    "programs_added": g["batched_programs_added"],
                }
                for g in probe["groups"]
            ],
            "calibration": probe["calibration"],
            "programs_after_warmup": probe["programs_after_warmup"],
            "programs_end": probe["programs_end"],
        },
        "full_model_comparison": {
            "artifact": str((STAGE9 / "teacher_forcing_p150.json").relative_to(MODEL.parents[2])),
            "teacher_forcing_decode_t_s_u": sorted(row["decode_t/s/u"] for row in teacher["teacher_forcing"])[
                len(teacher["teacher_forcing"]) // 2
            ],
            "teacher_forcing_ttft_ms": sorted(row["ttft_ms"] for row in teacher["teacher_forcing"])[
                len(teacher["teacher_forcing"]) // 2
            ],
            "teacher_forcing_selection": teacher.get("repeat_selection"),
            "serving_decode_t_s_u": after_b1["decode_t_s_u"],
        },
        "no_profiler_reason": "vLLM stages skip Tracy, tt-perf-report and device profiler collection by skill rule; evidence is same-harness benchmark JSON, adapter timing logs and contract checks.",
    }
    (HERE / "perf_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("ttft_ms", "decode_ms_per_token_e2e")}))
    print(
        "primary change %",
        summary["primary_single_user"]["ttft_change_percent"],
        summary["primary_single_user"]["tpot_change_percent"],
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write doc/optimized_vllm/perf_summary.json from the served benchmark JSON (optimize skill shape; device-time
fields null because vLLM stages collect no profiler) and print the before/after table rows."""

from __future__ import annotations

import json
from pathlib import Path

M = Path("models/autoports/qwen_qwen3_8b")
RV = M / "readiness_vllm"
OV = M / "doc" / "optimized_vllm"
STAGE9 = OV / "served_stage9_final"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def row(b: dict) -> dict:
    return {
        "ttft_p50_ms": b["ttft_ms"]["p50"],
        "ttft_p99_ms": b["ttft_ms"]["p99"],
        "ttft_mean_ms": b["ttft_ms"]["mean"],
        "tpot_mean_ms": b["tpot_ms"]["mean"],
        "tpot_p99_ms": b["tpot_ms"]["p99"],
        "itl_p50_ms": b["itl_ms"]["p50"],
        "itl_p99_ms": b["itl_ms"]["p99"],
        "output_throughput_tok_per_s": b["output_throughput_tok_per_s"],
        "decode_tps_per_user_from_mean_tpot": 1000.0 / b["tpot_ms"]["mean"],
        "completed_requests": b["completed_requests"],
        "config": b["config"],
    }


def main() -> None:
    after = {
        "single_user": row(load(RV / "vllm_benchmark.json")),
        "ci_burst": row(load(RV / "vllm_ci_serving_benchmark.json")),
    }
    before = {
        "single_user": row(load(STAGE9 / "vllm_benchmark.json")),
        "ci_burst": row(load(STAGE9 / "vllm_ci_serving_benchmark.json")),
    }
    burst = load(RV / "vllm_prefill_burst_benchmark.json")
    serial = load(OV / "vllm_prefill_burst_serial_benchmark.stage9.json")
    su = after["single_user"]
    summary = {
        "workload": {
            "profile": "single_user_decode",
            "prompt_len": 128,
            "gen_len": 128,
            "batch": 1,
            "max_num_seqs": 32,
            "max_model_len": 40960,
            "sampling": "greedy (temperature 0.0, explicit)",
            "mesh": "N150 label, one Blackhole P300 chip",
        },
        "ttft_ms": su["ttft_p50_ms"],
        "decode_ms_per_token_e2e": su["tpot_mean_ms"],
        "decode_ms_per_token_device": None,
        "roofline_ms_per_token_estimate": 13.99,
        "device_time_note": "vllm_serving_profiler_disabled_to_protect_hardware; device decode time 20.73 ms/token from the stage-7 reduced-variant profile (doc/optimized_full_model/perf_summary.json), roofline from doc/context_contract.json byte accounting",
        "before_stage9": before,
        "after_stage10": after,
        "single_user_change_pct": {
            "ttft_p50": 100.0 * (su["ttft_p50_ms"] / before["single_user"]["ttft_p50_ms"] - 1.0),
            "tpot_mean": 100.0 * (su["tpot_mean_ms"] / before["single_user"]["tpot_mean_ms"] - 1.0),
        },
        "prefill_burst": {
            "workload": burst["config"],
            "ttft_ms": burst["ttft_ms"],
            "serial_reference": burst.get("serial_reference"),
            "serial_reference_ttft_ms": serial["ttft_ms"],
            "p99_ratio": burst["ttft_ms"]["p99"] / serial["ttft_ms"]["p99"],
            "mean_ratio": burst["ttft_ms"]["mean"] / serial["ttft_ms"]["mean"],
            "gate": "p99 ratio <= 0.6 and mean ratio <= 0.5",
        },
        "named_limitations": [
            "decode is device bound: 36 decoder layers carry the stage-3/8 precision policy; the next decode lever is precision, not orchestration",
            "batched prefill dense ops run on the 2D/1D multicast family for bit equality with the solo rows, about 0.8 ms per layer slower than minimal_matmul at 4096 rows",
            "gate and up projections (N 12288) run as 512-row pieces because one 2D block does not fit L1 above 512 rows",
            "prefill is eager (no prefill trace); the first sampler step after prefill runs eagerly",
        ],
    }
    (OV / "perf_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: summary[k] for k in ("ttft_ms", "decode_ms_per_token_e2e", "single_user_change_pct", "prefill_burst")},
            indent=1,
        )
    )
    for name in ("single_user", "ci_burst"):
        print(name, "before", {k: round(v, 2) for k, v in before[name].items() if isinstance(v, float)})
        print(name, "after ", {k: round(v, 2) for k, v in after[name].items() if isinstance(v, float)})


if __name__ == "__main__":
    main()

"""Select median-TTFT repeats without mixing metrics from different requests."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(root, filename):
    rows = []
    configs = []
    for directory in sorted(root.glob("repeat*")):
        path = directory / filename
        data = json.loads(path.read_text())
        assert data["completed_requests"] == data["config"]["num_requests"]
        assert data["missing_output_tokens"] == 0
        configs.append(data["config"])
        rows.append(
            {
                "artifact": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "ttft_ms": data["ttft_ms"],
                "tpot_ms": data["tpot_ms"],
                "itl_ms": data["itl_ms"],
                "output_throughput_tok_per_s": data["output_throughput_tok_per_s"],
                "decode_t_s_u": 1000 / data["tpot_ms"]["mean"],
            }
        )
    assert len(rows) == 3 and all(config == configs[0] for config in configs)
    return {
        "workload": configs[0],
        "selection": "median TTFT among three warmed repeats; all metrics from that same repeat",
        "headline": sorted(rows, key=lambda row: row["ttft_ms"]["p50"])[1],
        "repeats": rows,
        "warmup": str(root / "warmup" / filename),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name = "vllm_ci_serving_benchmark.json" if args.ci else "vllm_benchmark.json"
    before, after = (summarize(root, name) for root in (args.before, args.after))
    assert before["workload"] == after["workload"], "Workloads are incomparable"
    result = {"before": before, "after": after}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for label, data in result.items():
        row = data["headline"]
        print(label, "TTFT", row["ttft_ms"]["p50"], "decode t/s/u", row["decode_t_s_u"])


if __name__ == "__main__":
    main()

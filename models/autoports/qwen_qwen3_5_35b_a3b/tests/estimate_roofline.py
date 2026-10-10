# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit algorithmic roofline estimates against complete measured layer spans.

DRAM traffic is a compulsory-traffic lower bound, not a hardware counter. It
counts active MoE weights (8 of256), required KV, and the materialized FP32
recurrence state passes. Cache/reloads, tile padding, multicast duplication,
and additional temporary activation traffic can make physical traffic larger.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement", type=Path)
    parser.add_argument("--layer-type", choices=("linear_attention", "full_attention"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measured = json.loads(args.measurement.read_text())
    assert measured["workload"] == dict(input_tokens=4096, output_tokens=128, batch=1, concurrency=1)
    tokens, hidden, intermediate, active = 4096, 2048, 512, 8
    q_dim, kv_dim, linear_q, linear_v = 4096, 512, 2048, 4096
    # A multiply-accumulate is two useful FLOPs. No work for unselected experts,
    # padded tokens, padded matmul rows, or the prefill tile union is counted.
    flops = dict(
        moe_and_shared=6 * hidden * intermediate * (active + 1) * tokens,
        router_and_shared_gate=2 * hidden * (256 + 1) * tokens,
    )
    # Logical BF16 inputs/outputs and two FP32 norm weights, read once.
    traffic = dict(
        input_output_and_norm_weights=2 * hidden * 2 + 2 * hidden * 4, router=hidden * 256 * 2, shared_gate=hidden * 2
    )
    # Packed TT tiles contain64 exponent bytes in addition to their mantissas.
    bfp4, bfp8 = 576 / 1024, 1088 / 1024
    traffic["active_experts_and_shared_weights"] = (active + 1) * hidden * intermediate * (2 * bfp4 + bfp8)
    if args.layer_type == "linear_attention":
        mega = 2 * linear_q + linear_v + 2 * 32 + linear_v
        flops["attention_projections"] = 2 * hidden * (mega + linear_v) * tokens
        flops["delta_recurrence"] = 32 * (7 * 128 * 128 + 2 * 128) * tokens
        flops["causal_convolution"] = (2 * linear_q + linear_v) * 7 * tokens
        traffic["attention_projection_weights"] = hidden * (mega + linear_v) * 2
        traffic["convolution_weights_and_state"] = (2 * linear_q + linear_v) * (4 * 4 + 2 * 3 * 2)
        # State decay2 + key product2 + reduction1 + outer product1 + add3
        # + query product2 + reduction1 + persistent copy2 =14 full state volumes.
        traffic["materialized_recurrent_state"] = 14 * 32 * 128 * 128 * 4
    else:
        flops["attention_projections"] = 2 * hidden * (3 * q_dim + 2 * kv_dim) * tokens
        causal_pairs = tokens * (tokens + 1) // 2
        flops["causal_qk_pv"] = 4 * q_dim * causal_pairs
        traffic["attention_projection_weights"] = hidden * (3 * q_dim + 2 * kv_dim) * bfp8
        # Decode step0 sees4097 tokens; step127 sees4224, including new K/V.
        mean_context = tokens + (1 + 128) / 2
        traffic["required_kv_reads_and_new_writes"] = (mean_context + 1) * 2 * kv_dim * 2
    useful_flops, dram_bytes = sum(flops.values()), sum(traffic.values())
    # Full theoretical ASIC peak, not just the110 user cores exposed by dispatch.
    peak_flops = 120 * 4096 * 1.35e9
    peak_dram = 512e9
    result = dict(
        layer_type=args.layer_type,
        source="single_layer",
        evidence=str(args.measurement),
        workload=measured["workload"],
        prefill_device_us=measured["prefill_device_us"],
        decode_device_us=measured["decode_device_us"],
        prefill_useful_flops=useful_flops,
        decode_dram_bytes=dram_bytes,
        peak_flops_per_s=peak_flops,
        peak_dram_bytes_per_s=peak_dram,
        prefill_flops_pct=100 * useful_flops / (peak_flops * measured["prefill_device_us"] / 1e6),
        decode_dram_pct=100 * dram_bytes / (peak_dram * measured["decode_device_us"] / 1e6),
        peak_basis="One Blackhole ASIC:120 physical Tensix cores x4096 FLOPs/cycle x1.35GHz LoFi nominal peak;512GB/s DRAM. Actual mixed LoFi/HiFi2/HiFi4; HiFi4 peak is one quarter of this maximum.110 user cores are exposed. P300c spec is two ASICs; divided card DRAM bandwidth by2.",
        peak_sources=[
            "https://docs.tenstorrent.com/aibs/blackhole/p300.html",
            "tt_perf_report/perf_report.py ArchitectureSpec blackhole per-core phase rates",
        ],
        useful_flops_components=flops,
        estimated_compulsory_dram_bytes_components=traffic,
        assumptions=[
            "Whole-layer firmware spans, including normalization, movement and gaps, are the denominator for both ratios.",
            "Useful FLOPs count projections, active MoE, causal QK/PV or delta recurrence, and convolution. Small vector/transcendental operations are excluded from numerator; their full time remains in denominator.",
            "DRAM is a compulsory-traffic lower-bound estimate: each active weight once, KV once,14 materialized FP32 state volumes for token GDN. Additional temporary traffic, tile padding, multicast reloads and caching are not modeled. This is not measured DRAM utilization.",
            "Default experts/shared:gate-up BFP4,down BFP8; attention projections GDN BF16 HiFi4/full BFP8 LoFi; FP32 GDN state; full KV BF16. Sparse prefill tile-union over-computation is not useful FLOPs.",
            "No percentage is clamped. These nominal-peak ratios are not per-op utilization averages.",
        ],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("prefill_flops_pct", "decode_dram_pct")}))


if __name__ == "__main__":
    main()

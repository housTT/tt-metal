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
    parser.add_argument("--policy", choices=("functional", "fused", "optimized"), default="functional")
    parser.add_argument("--optimization-json", default="{}")
    args = parser.parse_args()
    optimization = json.loads(args.optimization_json)
    if args.policy == "optimized" and not optimization:
        from ..tt.optimized_decoder import DEFAULT_POLICIES

        optimization = DEFAULT_POLICIES[args.layer_type]
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
    traffic["active_experts_and_shared_weights"] = (
        hidden * intermediate * (active * (3 * bfp8 if args.policy == "fused" else 2 * bfp4 + bfp8) + 2 * bfp4 + bfp8)
    )
    state_volumes = 14 if args.policy == "functional" else 10
    if args.policy == "optimized" and optimization.get("gdn_l1"):
        # Only read old state, write updated state, and read it for Q reduction
        # cross DRAM when all transient recurrence tensors reside in L1.
        state_volumes = 0 if optimization.get("gdn_state_l1") and optimization.get("gdn_l1_slots",32)>0 else 3
    if args.layer_type == "linear_attention":
        mega = 2 * linear_q + linear_v + 2 * 32 + linear_v
        flops["attention_projections"] = 2 * hidden * (mega + linear_v) * tokens
        flops["delta_recurrence"] = 32 * (7 * 128 * 128 + 2 * 128) * tokens
        flops["causal_convolution"] = (2 * linear_q + linear_v) * 7 * tokens
        traffic["attention_projection_weights"] = hidden * (mega + linear_v) * 2
        tap_bytes = 4 if args.policy == "functional" else 2
        traffic["convolution_weights_and_state"] = (2 * linear_q + linear_v) * (4 * tap_bytes + 2 * 3 * 2)
        # State decay2 + key product2 + reduction1 + outer product1 + add3
        # + query product2 + reduction1 + persistent copy2 =14 full state volumes.
        traffic["materialized_recurrent_state"] = state_volumes * 32 * 128 * 128 * 4
    else:
        flops["attention_projections"] = 2 * hidden * (3 * q_dim + 2 * kv_dim) * tokens
        causal_pairs = tokens * (tokens + 1) // 2
        flops["causal_qk_pv"] = 4 * q_dim * causal_pairs
        traffic["attention_projection_weights"] = hidden * (3 * q_dim + 2 * kv_dim) * bfp8
        # Decode step0 sees4097 tokens; step127 sees4224, including new K/V.
        mean_context = tokens + (1 + 128) / 2
        traffic["required_kv_reads_and_new_writes"] = (mean_context + 1) * 2 * kv_dim * 2
    if args.policy == "optimized":
        byte_rates = {"bf16":2, "bfp8":bfp8, "bfp4":bfp4}
        def projection_bytes(role, k, n, default):
            options = optimization.get("projections",{}).get(role,{})
            if options.get("split_widths"):
                return sum(k*w*byte_rates[o.get("dtype",default)] for w,o in
                           zip(options["split_widths"],options["split_options"],strict=True))
            return k*n*byte_rates[options.get("dtype",default)]
        expert_rate = byte_rates[optimization.get("expert_dtype","bfp8")]
        down_rate = byte_rates[optimization.get("sparse_down_dtype","bfp8")] if optimization.get("expert_backend") == "sparse" else expert_rate
        traffic["active_experts_and_shared_weights"] = hidden*intermediate*active*(2*expert_rate+down_rate)
        traffic["active_experts_and_shared_weights"] += (
            projection_bytes("shared_gate_proj",hidden,intermediate,"bfp4") +
            projection_bytes("shared_up_proj",hidden,intermediate,"bfp4") +
            projection_bytes("shared_down",intermediate,hidden,"bfp8"))
        traffic["router"] = projection_bytes("router",hidden,256,"bf16")
        traffic["shared_gate"] = projection_bytes("shared_gate",hidden,1,"bf16")
        incoming = 2*linear_q+2*linear_v+64 if args.layer_type=="linear_attention" else 2*q_dim+2*kv_dim
        default = "bf16" if args.layer_type=="linear_attention" else "bfp8"
        traffic["attention_projection_weights"] = (projection_bytes("attention_in",hidden,incoming,default) +
            projection_bytes("attention_out",linear_v,hidden,default))
        if args.layer_type=="full_attention":
            traffic["required_kv_reads_and_new_writes"] = (mean_context+1)*2*kv_dim*byte_rates[optimization.get("kv_dtype","bf16")]
    useful_flops, dram_bytes = sum(flops.values()), sum(traffic.values())
    # Full theoretical ASIC peak, not just the110 user cores exposed by dispatch.
    peak_flops = 120 * 4096 * 1.35e9
    peak_dram = 512e9
    result = dict(
        layer_type=args.layer_type,
        policy=args.policy,
        optimization=optimization if args.policy=="optimized" else None,
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
            f"DRAM is a compulsory-traffic lower-bound estimate: each active weight once, KV once,{state_volumes} materialized FP32 state volumes for token GDN. Fused state uses addcmul into persistent storage, eliminating the outer-product intermediate and copy. Additional temporary traffic, tile padding, multicast reloads and caching are not modeled. This is not measured DRAM utilization.",
            (
                "Fused decode routed weights:all BFP8, gate/up preserve original BFP4 coefficients by upcasting because native FFN requires homogeneous dtype; internal activation/partials BFP8 and LoFi, output BF16. Prefill horizontal packed gate/up and shared gate/up BFP4; down BFP8. "
                if args.policy == "fused"
                else ("Optimized per-role precision/fidelity is recorded in optimization; verify against matching runtime profile rows. " if args.policy=="optimized" else "Default experts/shared:gate-up BFP4,down BFP8. ")
            )
            + ("Attention projections GDN BF16 HiFi4/full BFP8 LoFi; FP32 GDN state; full KV BF16. " if args.policy!="optimized" else "FP32 GDN state; projection/cache precision is the supplied per-role policy. ")
            + "Unselected expert work (dense fused prefill or sparse tile-union), padding and algorithmic chunk intermediates are not useful FLOPs.",
            "No percentage is clamped. These nominal-peak ratios are not per-op utilization averages.",
        ],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("prefill_flops_pct", "decode_dram_pct")}))


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reconcile final full timing with same-run reduced roofline/device/host evidence."""

import argparse
import json
import math
import statistics
from pathlib import Path

DOC = Path(__file__).resolve().parent


def read(name):
    return json.loads((DOC / name).read_text())


def floor(linear, full, contexts):
    # Stored projection bytes read by decode, per rank. Excludes prefill-only copies.
    weights = linear * 30818304 + full * 34734080 + 536870912
    kv = full * statistics.mean(2 * math.ceil(context / 32) * 8 * 1088 for context in contexts)
    return dict(
        projection_and_head_bytes_per_rank=weights,
        mean_kv_read_bytes_per_rank=kv,
        embedding_minimum_bytes_per_rank=2048,
        dram_bandwidth_bytes_per_second_per_rank=512e9,
        aggregate_dram_bandwidth_bytes_per_second=4 * 512e9,
        roofline_ms_per_token=(weights + kv + 2048) / 512e9 * 1000,
        formula="4 * per-rank bytes / (4 * 512e9 bytes/s). KV includes BFP8 exponent headers, rounded32-token read tiles.",
        exclusions="Optimistic weight/KV-read floor: excludes recurrence, conv, norms, RoPE, activations, cache writes, collectives, padding overreads and compute. Not a prediction of achievable whole-model latency.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--after", default="perf_release_v2.json")
    parser.add_argument("--context", default="perf_context2048_release_v2.json")
    parser.add_argument("--teacher", default="teacher_release_v2.json")
    parser.add_argument("--profile-label", default="")
    parser.add_argument("--profile-run", default="tracy_decode_release_v1")
    parser.add_argument("--host-profile", default="profile_decode_host.json")
    args = parser.parse_args()
    before = read("baseline_repeated_v2.json")
    after = read(args.after)
    context = read(args.context)
    ccl = read("ccl_persistence_v2.json")
    terminal = read("head_geometry_c64_k1_r2_v1.json")["terminal"]["candidate"]["median_trace_ms"]
    components = {"terminal_ms": terminal}
    for mode in ("sampling", "embedding"):
        components[mode + "_ms"] = statistics.mean(
            row["trace_ms"] for row in ccl if row["mode"] == mode and row["persistent"]
        )
    stack = 24 * 0.355672 + 8 * 0.268965
    budget = stack + sum(components.values())
    context_ms = context["perf"]["decode_s"] * 1000 / context["perf"]["decode_steps"]
    ranks = read(Path("tracy") / args.profile_label / "decode_rank_accounting.json")["devices"]
    host = read(args.host_profile)
    device_ms = {
        rank: (row["kernel_us_per_iteration"] + row["gap_us_per_iteration"]) / 1000 for rank, row in ranks.items()
    }
    reduced_host_ms = host["elapsed_s"] * 1000 / host["steps"]
    full_floor = floor(24, 8, range(129, 256))
    reduced_floor = floor(1, 1, range(130, 134))
    result = dict(
        workload=dict(
            profile="single_user_decode", prompt_len=128, gen_len=128, batch=1, layers=32, cache_context=262144
        ),
        hardware="four Blackhole chips on two P300c boards; TP4 mesh1x4; software profilep150x4",
        ttft_ms=after["perf"]["ttft_s"] * 1000,
        decode_ms_per_token_e2e=after["perf"]["decode_s"] * 1000 / after["perf"]["decode_steps"],
        decode_ms_per_token_device=None,
        device_time_reason="Full32-layer profiling prohibited by optimize skill; direct same-run reduced-path device/host/floor triplet is reported below. No synthetic full-stack device measurement is claimed.",
        roofline_ms_per_token_estimate=full_floor["roofline_ms_per_token"],
        full_floor=full_floor,
        baseline=dict(
            artifact="baseline_repeated_v2.json", ttft_ms=before["perf"]["ttft_s"] * 1000, perf=before["perf"]
        ),
        final=dict(artifact=args.after, perf=after["perf"], warm_runs=after["warm_runs"]),
        first_request_after_model_construction=dict(
            baseline=before["first_perf"],
            final=after["first_perf"],
            scope="Includes first request setup, warmup/capture and first token. Excludes model loading; not total cold-start latency. A new prefill shape rebuilds the bounded four-trace family.",
        ),
        traced_logits_only=after["logits_only_trace"],
        plain_token_out=after["token_out_no_readback"],
        readiness_teacher_forcing=read(args.teacher)["metrics"],
        stack_budget=dict(
            source="../optimized_multichip_decoder/measurements.json and optimization_evidence.md",
            linear_count=24,
            linear_ms=0.355672,
            full_count=8,
            full_ms=0.268965,
            layer_stack_ms=stack,
            layer_stack_tps=1000 / stack,
            terminal_components=components,
            budget_ms=budget,
            measured_context2048_ms=context_ms,
            measured_context2048_artifact=args.context,
            excess_over_budget_percent=100 * (context_ms / budget - 1),
            requires_gap_closure=context_ms > budget * 1.10,
            boundary_caveat="Standalone layers start from DRAM and each has a trace boundary; the full stack directly reuses selected L1 residuals. The additive budget is a planning/lower-bound estimate, not a strict physical equality or same-run profile.",
        ),
        reduced_same_run=dict(
            run=f"logs/{args.profile_run}.provenance.json",
            layers=[0, 3],
            prompt_len=128,
            decode_positions=[129, 130, 131, 132],
            cache_context=262144,
            steps=host["steps"],
            output_history=True,
            roofline=reduced_floor,
            decode_ms_per_token_device_by_rank=device_ms,
            decode_ms_per_token_device=max(device_ms.values()),
            decode_ms_per_token_e2e=reduced_host_ms,
            host_minus_slowest_device_ms=reduced_host_ms - max(device_ms.values()),
            scope="Same instrumented signposted four-replay run; full real shapes, embedding/two layer kinds/norm/head/split sampler/history/token feedback. Profiler timings are distinct from uninstrumented full-model headline.",
        ),
        named_limitations=[
            "Optimistic bandwidth floor omits recurrent/state/activation traffic, compute and communication; the measured path includes many small kernels and dispatch gaps.",
            "Native decoder coherent topology/program families remain selected from precision-locked predecessor evidence; no precision or residual-layout fallback.",
            "High-level generation collects fixed128-step windows and reads at window boundaries; plain token-out has no loop readback. First-token TTFT read is included.",
            "Native context validated atbatch1; batch32 validated at short context, not32 simultaneousnative-length requests.",
        ],
    )
    (DOC / "perf_summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint 4 — hardware benchmark harness (Blackhole).

REFUSES to run under the simulator (there is none here, but the guard is kept per the
GOAL rule). Emits a markdown + CSV table with provenance (git SHA, arch, date).

Two modes:
  * `--module`  (default, WORKS TODAY): a genuine on-device micro-benchmark of the
    Checkpoint-2-validated clamped-SwiGLU MLP with **resident weights** and device
    synchronization — a real `[hw: measured]` latency/throughput table. This is a
    module-level number, NOT full-model TTFT/tok-s.
  * `--model`   (NOT YET AVAILABLE): full-model TTFT + tok/s. Raises with a clear
    message because the end-to-end DeepSeek-V4 model (full MLA-v4 attention + decoder
    assembly) is not finished (see REPORT_2). Producing TTFT/tok-s without it would
    fabricate performance numbers, which the GOAL forbids.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

if os.environ.get("TT_METAL_SIMULATOR"):
    sys.exit("REFUSING: benchmark must run on hardware; unset TT_METAL_SIMULATOR.")

import torch

import ttnn
from models.demos.deepseek_v4.reference.reduced_config import build_reduced_model


def git_sha():
    try:
        return (
            subprocess.check_output(["git", "-C", os.environ.get("TT_METAL_HOME", "."), "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def module_benchmark(dev, token_counts, iters=30, warmup=5):
    """Resident-weight, device-synchronized micro-bench of the clamped SwiGLU MLP."""
    model, cfg = build_reduced_model(seed=0)
    se = model.model.layers[0].mlp.shared_experts
    H = cfg.hidden_size

    # Build resident device weights ONCE (transpose for xᵀ matmul convention).
    def dev_w(w, dt=ttnn.bfloat8_b):
        return ttnn.from_torch(
            w.t().contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    tg, tu, td = dev_w(se.gate_proj.weight.data), dev_w(se.up_proj.weight.data), dev_w(se.down_proj.weight.data)
    limit = cfg.swiglu_limit
    rows = []
    for n in token_counts:
        x = torch.randn(1, n, H)
        tx = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        def step():
            g = ttnn.clamp(ttnn.linear(tx, tg), max=limit)
            u = ttnn.clamp(ttnn.linear(tx, tu), min=-limit, max=limit)
            y = ttnn.linear(ttnn.multiply(ttnn.silu(g), u), td)
            return y

        for _ in range(warmup):
            step()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) / iters * 1e3
        tps = n / (ms / 1e3)
        rows.append((n, f"{ms:.4f}", f"{tps:,.0f}"))
        print(f"[hw] SwiGLU MLP  tokens={n:5d}  {ms:.4f} ms/iter  {tps:,.0f} tok/s [hw: measured, module-level]")
    return rows, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", action="store_true", help="module micro-benchmark (default)")
    ap.add_argument("--model", action="store_true", help="full-model TTFT/tok-s (not yet available)")
    ap.add_argument("--tokens", default="32,128,512,2048")
    ap.add_argument("--out", default="RESULTS_module")
    args = ap.parse_args()

    if args.model:
        sys.exit(
            "Full-model TTFT/tok-s is NOT available: the end-to-end DeepSeek-V4 TT-NN model "
            "(full MLA-v4 attention + decoder assembly) is not finished — see REPORT_2. "
            "Emitting TTFT/tok-s without it would fabricate performance numbers (GOAL forbids). "
            "Run with --module for the validated module-level [hw] benchmark."
        )

    token_counts = [int(t) for t in args.tokens.split(",")]
    dev = ttnn.CreateDevice(device_id=0)
    try:
        arch = str(dev.arch())
        rows, cfg = module_benchmark(dev, token_counts)
    finally:
        ttnn.CloseDevice(dev)

    hdr = ["tokens", "latency_ms [hw]", "tok/s [hw]"]
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    with open(f"{args.out}.md", "w") as f:
        f.write(f"# Module benchmark [hw]  (arch={arch}, sha={git_sha()}, date=2026-07-03)\n\n")
        f.write(
            "Clamped SwiGLU MLP (shared expert), bfloat8_b weights, reduced config "
            f"hidden={cfg.hidden_size}. **Module-level — NOT full-model TTFT/tok-s.**\n\n"
        )
        f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for r in rows:
            f.write("| " + " | ".join(map(str, r)) + " |\n")
    print("\n" + open(f"{args.out}.md").read())


if __name__ == "__main__":
    main()

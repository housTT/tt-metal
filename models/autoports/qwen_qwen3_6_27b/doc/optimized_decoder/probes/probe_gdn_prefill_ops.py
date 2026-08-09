# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Two `tt-perf-report` advice items on the gated-delta-net prefill path, tried rather than argued.

1. **The batched delta-rule matmuls carry "No program_config specified".**  96 rows, 2.5 ms of a
   30.5 ms `linear_attention` prefill.  This tries an explicit
   ``MatmulMultiCoreReuseProgramConfig`` on the two shapes that carry the advice.

2. **The causal conv costs ~1.9 ms of tilize/untilize.**  Its window concat prepends
   ``conv_kernel_size - 1 = 3`` rows and its taps then slice at rows 0, 1, 2 - none tile
   aligned, so TTNN untilizes and retilizes an ``[1, 1, L, 10240]`` tensor five times.  The
   dedicated op that would replace the whole thing is ``ttnn.conv1d``; this checks whether it can
   express a depthwise causal conv over 10240 channels at all, and times it if it can.

Emits ``GDNPREFILL `` JSON lines.
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")
sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes")

import torch  # noqa: E402
import ttnn  # noqa: E402
from matmul_sweep import time_op  # noqa: E402

L, CONV_DIM, K_CONV = 2048, 10240, 4
NV, CHUNK, DK, DV = 48, 64, 128, 128


def emit(**p):
    print("GDNPREFILL " + json.dumps(p, sort_keys=True, default=str), flush=True)


def pcc(a, b):
    a = a.to(torch.float64).flatten(); a = a - a.mean()
    b = b.to(torch.float64).flatten(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def batched_matmul_configs(device):
    """Advice item 1: explicit program config on the per-chunk delta-rule matmuls."""
    hifi4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                             math_approx_mode=False, fp32_dest_acc_en=True,
                                             packer_l1_acc=True)
    for label, (lead, m, k, n) in (("qk_T", (NV, CHUNK, DK, CHUNK)), ("inv_v", (NV, CHUNK, CHUNK, DV))):
        a = ttnn.from_torch(torch.randn(1, lead, m, k) * 0.1, dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        b = ttnn.from_torch(torch.randn(1, lead, k, n) * 0.1, dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        golden = ttnn.to_torch(ttnn.matmul(a, b, dtype=ttnn.float32,
                                           compute_kernel_config=hifi4,
                                           memory_config=ttnn.L1_MEMORY_CONFIG)).float()
        row = {"item": "batched_matmul", "shape": f"b={lead} x {m} x {k} x {n}", "config": "default"}
        row["us"] = time_op(lambda: ttnn.matmul(a, b, dtype=ttnn.float32,
                                                compute_kernel_config=hifi4,
                                                memory_config=ttnn.L1_MEMORY_CONFIG), device)
        emit(**row)
        for grid in ((8, 8), (8, 6), (4, 8), (8, 4)):
            for in0_block_w in (k // 32, max(1, k // 64), 1):
                row = {"item": "batched_matmul", "shape": f"b={lead} x {m} x {k} x {n}",
                       "config": f"reuse_{grid[0]}x{grid[1]}_ibw{in0_block_w}"}
                try:
                    pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                        compute_with_storage_grid_size=grid, in0_block_w=in0_block_w,
                        out_subblock_h=1, out_subblock_w=min(4, n // 32),
                        per_core_M=m // 32, per_core_N=n // 32)
                    got = ttnn.matmul(a, b, program_config=pc, dtype=ttnn.float32,
                                      compute_kernel_config=hifi4, memory_config=ttnn.L1_MEMORY_CONFIG)
                    row["pcc"] = pcc(golden, ttnn.to_torch(got).float())
                    ttnn.deallocate(got)
                    row["us"] = time_op(lambda: ttnn.matmul(a, b, program_config=pc,
                                                            dtype=ttnn.float32,
                                                            compute_kernel_config=hifi4,
                                                            memory_config=ttnn.L1_MEMORY_CONFIG), device)
                except Exception as exc:
                    lines = str(exc).splitlines()
                    row["error"] = lines[2] if len(lines) > 2 else str(exc)
                emit(**row)
        ttnn.deallocate(a)
        ttnn.deallocate(b)


def conv1d_candidate(device):
    """Advice item 2: can ttnn.conv1d express the depthwise causal conv?"""
    row = {"item": "conv1d", "channels": CONV_DIM, "kernel": K_CONV, "groups": CONV_DIM}
    if not hasattr(ttnn, "conv1d"):
        row["error"] = "ttnn.conv1d is not present in this build"
        emit(**row)
        return
    x = ttnn.from_torch(torch.randn(1, L, CONV_DIM) * 0.1, dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = ttnn.from_torch(torch.randn(CONV_DIM, 1, K_CONV) * 0.1, dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT)
    try:
        out = ttnn.conv1d(input_tensor=x, weight_tensor=w, in_channels=CONV_DIM,
                          out_channels=CONV_DIM, device=device, kernel_size=K_CONV, stride=1,
                          padding=K_CONV - 1, groups=CONV_DIM, batch_size=1, input_length=L)
        row["ok"] = True
        row["out_shape"] = str(out.shape if not isinstance(out, tuple) else out[0].shape)
    except Exception as exc:
        lines = str(exc).splitlines()
        row["error"] = lines[2] if len(lines) > 2 else str(exc)[:400]
    emit(**row)

    # The op's auto-slicer failed on the full width; adapt the shape and retry, which is the
    # bar $optimize sets before a layout cost may be called blocked.  Splitting the 10240
    # channels into narrower depthwise convs is mathematically exact - a depthwise conv does
    # not mix channels - so this is a legal decomposition, not an approximation.
    for split in (2, 4, 8, 16):
        chunk = CONV_DIM // split
        row = {"item": "conv1d_split", "splits": split, "channels_per_split": chunk,
               "kernel": K_CONV, "groups": chunk}
        xs = ttnn.from_torch(torch.randn(1, L, chunk) * 0.1, dtype=ttnn.bfloat16,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ws = ttnn.from_torch(torch.randn(chunk, 1, K_CONV) * 0.1, dtype=ttnn.bfloat16,
                             layout=ttnn.ROW_MAJOR_LAYOUT)
        try:
            out = ttnn.conv1d(input_tensor=xs, weight_tensor=ws, in_channels=chunk,
                              out_channels=chunk, device=device, kernel_size=K_CONV, stride=1,
                              padding=K_CONV - 1, groups=chunk, batch_size=1, input_length=L)
            row["ok"] = True
            row["us_per_split"] = time_op(
                lambda: ttnn.conv1d(input_tensor=xs, weight_tensor=ws, in_channels=chunk,
                                    out_channels=chunk, device=device, kernel_size=K_CONV,
                                    stride=1, padding=K_CONV - 1, groups=chunk, batch_size=1,
                                    input_length=L), device)
            row["us_total_estimate"] = round(row["us_per_split"] * split, 1)
        except Exception as exc:
            lines = str(exc).splitlines()
            row["error"] = lines[2] if len(lines) > 2 else str(exc)[:250]
        emit(**row)
        ttnn.deallocate(xs)


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        batched_matmul_configs(device)
        conv1d_candidate(device)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

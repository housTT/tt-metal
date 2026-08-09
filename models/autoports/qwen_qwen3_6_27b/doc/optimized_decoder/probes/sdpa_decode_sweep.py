# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode-SDPA program-config search (``$optimize`` OPT-002).

After O1/O2 the paged decode SDPA is the second largest row of a ``full_attention`` decode step
(226 us of 1082 us), and the fused stage ran it on the op's default configuration.  This sweeps
explicit ``ttnn.SDPAProgramConfig`` candidates on the real contract - 24 query heads, 4 KV
heads, head_dim 256, BFP8 paged cache, block size 64, position 2048 - and checks each against
the default's output so a faster-but-wrong config cannot win.

Emits ``SDPASWEEP `` JSON lines.
"""
from __future__ import annotations

import json
import math
import sys
import time

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")

import torch  # noqa: E402
import ttnn  # noqa: E402

N_HEADS, N_KV, HEAD_DIM = 24, 4, 256
BLOCK = 64
POSITIONS = (2048, 8192)
BATCH = 1
PADDED_HEADS = 32
ITERS = 20


def emit(**payload):
    print("SDPASWEEP " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def pcc(a, b):
    a = a.to(torch.float64).flatten(); a = a - a.mean()
    b = b.to(torch.float64).flatten(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    try:
        for position in POSITIONS:
            max_seq = ((position + 1 + 2047) // 2048) * 2048
            num_blocks = BATCH * (max_seq // BLOCK)
            torch.manual_seed(0)
            cache = [
                ttnn.from_torch(torch.randn(num_blocks, N_KV, BLOCK, HEAD_DIM) * 0.05,
                                dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for _ in range(2)
            ]
            table = torch.randperm(num_blocks).reshape(BATCH, -1).to(torch.int32)
            page_table = ttnn.from_torch(table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                         device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cur_pos = ttnn.from_torch(torch.tensor([position] * BATCH, dtype=torch.int32),
                                      dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q = ttnn.from_torch(torch.randn(1, BATCH, PADDED_HEADS, HEAD_DIM) * 0.05,
                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)

            def run(pc):
                return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q, cache[0], cache[1], page_table, cur_pos_tensor=cur_pos,
                    scale=HEAD_DIM ** -0.5, program_config=pc,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc)

            golden = ttnn.to_torch(run(None)).float()

            candidates = [("default", None)]
            for gx, gy in ((8, 8), (11, 10), (8, 10), (11, 8), (4, 8)):
                for k_chunk in (0, 128, 256, 512):
                    q_chunk = 0 if k_chunk == 0 else 32
                    candidates.append((
                        f"grid{gx}x{gy}_q{q_chunk}_k{k_chunk}",
                        ttnn.SDPAProgramConfig(
                            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                            q_chunk_size=q_chunk, k_chunk_size=k_chunk, exp_approx_mode=False),
                    ))
            for label, pc in candidates:
                row = {"position": position, "candidate": label}
                try:
                    got = run(pc)
                    row["pcc"] = pcc(golden, ttnn.to_torch(got).float())
                    ttnn.deallocate(got)
                    run(pc)
                    ttnn.synchronize_device(device)
                    start = time.perf_counter()
                    for _ in range(ITERS):
                        out = run(pc)
                        ttnn.deallocate(out)
                    ttnn.synchronize_device(device)
                    row["us"] = (time.perf_counter() - start) * 1e6 / ITERS
                except Exception as exc:
                    lines = str(exc).splitlines()
                    row["error"] = lines[2] if len(lines) > 2 else str(exc)
                emit(**row)
            for t in (*cache, page_table, cur_pos, q):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Op-level feasibility of a token-sorted MoE prefill on the existing sparse_matmul kernel.

Compares the current expert-union call (128-row groups, every routed expert of the
group) with one-hot 32-row groups of expert-sorted rows and a compact output.

Usage (one chip, from the tt-metal worktree):
    TT_VISIBLE_DEVICES=2 ... python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_sorted_moe
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import _optimized_sparse_prefill_config

HIDDEN = 2816
PACKED = 1408
INTER = 704
EXPERTS = 128
TOP_K = 8
TILE = 32
ROWS = 32


def _timed(device, call, repeats=3):
    best = None
    for _ in range(repeats):
        ttnn.synchronize_device(device)
        began = time.perf_counter()
        out = call()
        ttnn.synchronize_device(device)
        best = min(best or 1e9, time.perf_counter() - began)
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=str, default="128,2048")
    parser.add_argument("--groups-per-call", type=int, default=8)
    parser.add_argument("--in0-block-w", type=int, default=44)
    parser.add_argument("--rows-per-group", type=int, default=32)
    args = parser.parse_args()
    device = ttnn.open_device(device_id=0)
    report = []
    try:
        torch.manual_seed(0)
        w_gate_up = torch.randn(1, EXPERTS, HIDDEN, PACKED, dtype=torch.bfloat16) * 0.02
        tt_w = ttnn.from_torch(
            w_gate_up,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        output_tile = ttnn.Tile([TILE, TILE])
        for m in [int(v) for v in args.lengths.split(",")]:
            x = torch.randn(1, 1, m, HIDDEN, dtype=torch.bfloat16)
            scores = torch.rand(m, EXPERTS)
            topk = scores.topk(TOP_K, dim=-1).indices
            chunk = min(m, 1024)
            union_groups = max(1, chunk // 128)
            rows = chunk // union_groups
            union_inputs = []
            for start in range(0, m, chunk):
                union_inputs.append(
                    ttnn.from_torch(
                        x[:, :, start : start + chunk, :].reshape(1, union_groups, rows, HIDDEN),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
            sparsity_union = ttnn.from_torch(
                torch.ones(1, union_groups, 1, EXPERTS, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            union_config = _optimized_sparse_prefill_config(
                device, n=PACKED, groups=union_groups, requested_per_core_n=2, in0_block_w=44, per_core_m=rows // TILE
            )

            def union_call():
                for tt_x_union in union_inputs:
                    out = ttnn.sparse_matmul(
                        tt_x_union,
                        tt_w,
                        sparsity=sparsity_union,
                        nnz=None,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        output_tile=output_tile,
                        dtype=ttnn.bfloat16,
                        program_config=union_config,
                        compute_kernel_config=compute,
                    )
                    ttnn.deallocate(out)
                return None

            union_seconds = _timed(device, union_call)
            per_expert = [[] for _ in range(EXPERTS)]
            for token in range(m):
                for expert in topk[token].tolist():
                    per_expert[expert].append(token)
            sorted_tokens = []
            group_expert = []
            ROWS = args.rows_per_group
            for expert, tokens in enumerate(per_expert):
                for start in range(0, len(tokens), ROWS):
                    chunk = tokens[start : start + ROWS]
                    chunk = chunk + [chunk[-1]] * (ROWS - len(chunk))
                    sorted_tokens.extend(chunk)
                    group_expert.append(expert)
            total_groups = len(group_expert)
            pad_groups = (-total_groups) % args.groups_per_call
            group_expert = group_expert + [0] * pad_groups
            sorted_tokens = sorted_tokens + [0] * (pad_groups * ROWS)
            calls = []
            for start in range(0, len(group_expert), args.groups_per_call):
                experts = group_expert[start : start + args.groups_per_call]
                gc = len(experts)
                rows_idx = sorted_tokens[start * ROWS : (start + gc) * ROWS]
                x_sorted = x[0, 0, rows_idx, :].reshape(1, gc, ROWS, HIDDEN)
                one_hot = torch.zeros(1, gc, 1, EXPERTS, dtype=torch.bfloat16)
                for g, e in enumerate(experts):
                    one_hot[0, g, 0, e] = 1.0
                calls.append(
                    (
                        gc,
                        experts,
                        rows_idx,
                        ttnn.from_torch(
                            x_sorted,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        ),
                        ttnn.from_torch(
                            one_hot,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        ),
                    )
                )
            configs = {}
            outputs = {}
            for gc, *_ in calls:
                if gc not in configs:
                    configs[gc] = _optimized_sparse_prefill_config(
                        device,
                        n=PACKED,
                        groups=gc,
                        requested_per_core_n=2,
                        in0_block_w=args.in0_block_w,
                        per_core_m=ROWS // TILE,
                    )
                    outputs[gc] = ttnn.zeros(
                        (1, gc, ROWS, PACKED),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

            def sorted_call():
                last = None
                for gc, experts, rows_idx, tt_x, tt_s in calls:
                    last = ttnn.sparse_matmul(
                        tt_x,
                        tt_w,
                        sparsity=tt_s,
                        nnz=gc,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16,
                        program_config=configs[gc],
                        compute_kernel_config=compute,
                        optional_output_tensor=outputs[gc],
                    )
                return None

            sorted_seconds = _timed(device, sorted_call)
            gc, experts, rows_idx, tt_x, tt_s = calls[0]
            out = ttnn.sparse_matmul(
                tt_x,
                tt_w,
                sparsity=tt_s,
                nnz=gc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                program_config=configs[gc],
                compute_kernel_config=compute,
                optional_output_tensor=outputs[gc],
            )
            host = ttnn.to_torch(out).float()
            ref = torch.einsum(
                "gtk,gkn->gtn", x[0, 0, rows_idx, :].reshape(gc, ROWS, HIDDEN).float(), w_gate_up[0, experts].float()
            )
            host_rows = host.reshape(gc, ROWS, PACKED) if host.numel() == gc * ROWS * PACKED else host
            pcc = float(torch.corrcoef(torch.stack([host_rows.flatten(), ref.flatten()]))[0, 1])
            entry = {
                "m": m,
                "rows_per_group": ROWS,
                "groups_per_call": args.groups_per_call,
                "in0_block_w": args.in0_block_w,
                "per_core_n": configs[args.groups_per_call].per_core_N,
                "pairs": m * TOP_K,
                "sorted_groups": total_groups,
                "padded_groups": len(group_expert),
                "sorted_rows": total_groups * ROWS,
                "sorted_calls": len(calls),
                "union_groups": union_groups,
                "union_rows": rows,
                "union_ms": round(union_seconds * 1000, 2),
                "sorted_ms": round(sorted_seconds * 1000, 2),
                "speedup": round(union_seconds / sorted_seconds, 2),
                "output_shape": list(host.shape),
                "pcc_vs_torch_first_call": round(pcc, 5),
            }
            report.append(entry)
            print(json.dumps(entry), flush=True)
            for _, _, _, tt_x, tt_s in calls:
                ttnn.deallocate(tt_x)
                ttnn.deallocate(tt_s)
            for t in outputs.values():
                ttnn.deallocate(t)
            for tt_x_union in union_inputs:
                ttnn.deallocate(tt_x_union)
            ttnn.deallocate(sparsity_union)
    finally:
        ttnn.close_device(device)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

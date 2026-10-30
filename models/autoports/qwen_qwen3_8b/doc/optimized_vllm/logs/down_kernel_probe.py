# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Down-projection kernel bit-equality sweep: which kernel at ``dense`` rows reproduces, row for row, the solo
kernel of ``seq`` rows (the stage-7 table choice) on the same random input?

usage: down_kernel_probe.py <output.json> [--cases 4096:256,4096:128,2048:128,1024:128,512:128,2048:256,1024:256,1024:512,2048:512,4096:512]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator

MODEL_DIR = Path("models/autoports/qwen_qwen3_8b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--role", default="down_proj")
    ap.add_argument(
        "--cases",
        default="4096:256,4096:128,2048:128,1024:128,512:128,2048:256,1024:256,1024:512,2048:512,4096:512,256:32,2048:64",
    )
    args = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
    gen = build_generator(str(MODEL_DIR), mesh, num_layers=1, prepare=False, decode_buckets=(1,), num_blocks=8)
    layer = gen.model.layers[0]
    weight = getattr(layer.weights, args.role)
    copy = layer._interleaved_copy[id(weight)]
    k, n = weight.shape[-2], weight.shape[-1]
    dtype = layer.weight_dtypes[args.role]

    def run(x, choice):
        r = x.shape[-2]
        kernel, cfg = choice
        try:
            if kernel == "1d":
                return ttnn.linear(
                    x,
                    copy,
                    program_config=layer._prefill_1d_program_config(r, n, *cfg),
                    dtype=layer.activation_dtype,
                    compute_kernel_config=layer._prefill_compute_config(copy),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            if kernel == "2d":
                return ttnn.linear(
                    x,
                    copy,
                    program_config=layer._prefill_2d_program_config(r, k, n, cfg),
                    dtype=layer.activation_dtype,
                    compute_kernel_config=layer._prefill_compute_config(copy),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            if kernel == "auto":
                return ttnn.linear(
                    x,
                    copy,
                    dtype=layer.activation_dtype,
                    compute_kernel_config=layer._prefill_compute_config(copy),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            return ttnn.experimental.minimal_matmul(
                x,
                copy,
                dtype=layer.activation_dtype,
                compute_kernel_config=layer._prefill_compute_config(copy),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        except Exception as e:
            return str(e).splitlines()[0][:160]

    results = []
    for case in args.cases.split(","):
        dense, seq = (int(v) for v in case.split(":"))
        g = torch.Generator().manual_seed(dense + seq)
        host = (torch.randn(1, 1, dense, k, generator=g) * 0.5).to(torch.bfloat16)
        x_all = ttnn.from_torch(
            host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        solo_choice = layer.policy.prefill_kernel(args.role, seq, dtype)
        solo_rows = []
        for i in range(dense // seq):
            x_i = ttnn.from_torch(
                host[:, :, i * seq : (i + 1) * seq],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = run(x_i, solo_choice)
            solo_rows.append(ttnn.to_torch(out)[0, 0].clone())
            out.deallocate(True)
            x_i.deallocate(True)
        solo_cat = torch.cat(solo_rows, dim=0)
        rec = {
            "dense": dense,
            "seq": seq,
            "solo_kernel": solo_choice,
            "dense_table_kernel": layer.policy.prefill_kernel(args.role, dense, dtype),
            "candidates": {},
        }
        candidates = [
            ("minimal", ("minimal", 0)),
            ("2d_16", ("2d", 16)),
            ("2d_8", ("2d", 8)),
            ("2d_4", ("2d", 4)),
            ("2d_2", ("2d", 2)),
            ("auto", ("auto", 0)),
        ]
        if dense <= 512:
            candidates += [("1d_110_8", ("1d", (110, 8, 2, 2))), ("1d_110_4", ("1d", (110, 4, 2, 2)))]
        for name, choice in candidates:
            out = run(x_all, choice)
            if isinstance(out, str):
                rec["candidates"][name] = {"error": out}
                continue
            h = ttnn.to_torch(out)[0, 0]
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(3):
                o2 = run(x_all, choice)
                o2.deallocate(True)
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t0) / 3 * 1000
            out.deallocate(True)
            rows_eq = sum(torch.equal(h[i * seq : (i + 1) * seq], solo_rows[i]) for i in range(dense // seq))
            rec["candidates"][name] = {
                "rows_bit_equal": f"{rows_eq}/{dense // seq}",
                "max_abs_diff": float((h.float() - solo_cat.float()).abs().max()),
                "ms": ms,
            }
        x_all.deallocate(True)
        print(json.dumps(rec), flush=True)
        results.append(rec)
    Path(args.output).write_text(
        json.dumps(
            {"role": args.role, "results": results, "visible_devices": os.getenv("TT_VISIBLE_DEVICES")}, indent=2
        )
        + "\n"
    )
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

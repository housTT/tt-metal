# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""1D multicast (``mcast_in0``) kernel sweep for the wide-N roles (gate, up) over row pieces, for the batched
dense pass and for the 512-row solo table entry; bit-equality against the stage-7 solo kernel of ``seq`` rows.

usage: wide_kernel_probe.py <output.json> [--rows 512,1024,2048,4096] [--roles gate_proj,up_proj]
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
    ap.add_argument("--rows", default="512,1024,2048,4096")
    ap.add_argument("--roles", default="gate_proj,up_proj")
    ap.add_argument("--seq", type=int, default=128)
    args = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
    gen = build_generator(str(MODEL_DIR), mesh, num_layers=1, prepare=False, decode_buckets=(1,), num_blocks=8)
    layer = gen.model.layers[0]
    grid = layer.grid
    results = []
    for role in args.roles.split(","):
        weight = getattr(layer.weights, role)
        copy = layer._interleaved_copy[id(weight)]
        k, n = weight.shape[-2], weight.shape[-1]
        dtype = layer.weight_dtypes[role]
        ckc = layer._prefill_compute_config(copy)
        for total in (int(v) for v in args.rows.split(",")):
            g = torch.Generator().manual_seed(total + n)
            host = (torch.randn(1, 1, total, k, generator=g) * 0.7).to(torch.bfloat16)
            x_all = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            solo_choice = layer.policy.prefill_kernel(role, args.seq, dtype)
            solo_rows = []
            for i in range(total // args.seq):
                x_i = ttnn.from_torch(host[:, :, i * args.seq : (i + 1) * args.seq], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                kernel, cfg = solo_choice
                if kernel == "1d":
                    o = ttnn.linear(x_i, copy, program_config=layer._prefill_1d_program_config(args.seq, n, *cfg), dtype=layer.activation_dtype, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                else:
                    o = ttnn.linear(x_i, copy, dtype=layer.activation_dtype, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                solo_rows.append(ttnn.to_torch(o)[0, 0].clone())
                o.deallocate(True)
                x_i.deallocate(True)
            solo_cat = torch.cat(solo_rows, dim=0)
            rec = {"role": role, "dtype": dtype, "total_rows": total, "solo_seq": args.seq, "solo_kernel": solo_choice, "candidates": {}}

            def measure(name, fn):
                try:
                    out = fn()
                except Exception as e:
                    rec["candidates"][name] = {"error": str(e).splitlines()[0][:140]}
                    return
                h = ttnn.to_torch(out).reshape(total, n)
                out.deallocate(True)
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(3):
                    o2 = fn()
                    o2.deallocate(True)
                ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - t0) / 3 * 1000
                rec["candidates"][name] = {"ms": round(ms, 4), "bit_equal_solo": bool(torch.equal(h, solo_cat)), "max_abs_diff": float((h.float() - solo_cat.float()).abs().max())}

            measure("minimal", lambda: ttnn.experimental.minimal_matmul(x_all, copy, dtype=layer.activation_dtype, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            for piece in (total, 1024, 512, 256):
                if piece > total:
                    continue
                pieces = total // piece
                for cores in (110, 64):
                    for bw in (16, 8, 4):
                        for sub in ((2, 2), (1, 4), (4, 1)):
                            per_core_m = piece // 32
                            per_core_n = -(-(n // 32) // cores)
                            if per_core_m % sub[0] or per_core_n % sub[1]:
                                continue

                            def fn(piece=piece, pieces=pieces, cores=cores, bw=bw, sub=sub):
                                cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                                    compute_with_storage_grid_size=(grid.x, grid.y), in0_block_w=bw, out_subblock_h=sub[0], out_subblock_w=sub[1],
                                    per_core_M=piece // 32, per_core_N=-(-(n // 32) // cores), fuse_batch=pieces == 1, fused_activation=None, mcast_in0=True)
                                x = ttnn.reshape(x_all, (pieces, 1, piece, k)) if pieces > 1 else x_all
                                out = ttnn.linear(x, copy, program_config=cfg, dtype=layer.activation_dtype, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                                if pieces > 1:
                                    out = ttnn.reshape(out, (1, 1, total, n))
                                return out

                            measure(f"1d_c{cores}_bw{bw}_sub{sub[0]}x{sub[1]}_pieces{pieces}x{piece}", fn)
            x_all.deallocate(True)
            best = sorted(((v["ms"], kname) for kname, v in rec["candidates"].items() if "ms" in v))[:6]
            print(json.dumps({"role": role, "rows": total, "minimal": rec["candidates"].get("minimal"), "best": best, "errors": sum(1 for v in rec["candidates"].values() if "error" in v), "all_equal": all(v.get("bit_equal_solo", True) for v in rec["candidates"].values())}), flush=True)
            results.append(rec)
    Path(args.output).write_text(json.dumps({"results": results, "visible_devices": os.getenv("TT_VISIBLE_DEVICES")}, indent=2) + "\n")
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

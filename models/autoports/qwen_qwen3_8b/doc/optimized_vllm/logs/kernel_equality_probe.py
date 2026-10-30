# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Op-level bit-equality probe: does a dense prefill op at ``rows * seq`` rows reproduce the ``seq``-row
result of the same rows bit for bit? Checks the input RMSNorm, then the QKV projection under every kernel
candidate (the solo kernel of ``seq`` rows, and the 1D/2D/minimal kernels at ``rows * seq`` rows).

usage: kernel_equality_probe.py <output.json> [--layers 2] [--shapes 32x128,16x256,8x512]
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


def to_host(t):
    return ttnn.to_torch(t)[0, 0].clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--shapes", default="32x128,16x256,8x512,4x128")
    args = ap.parse_args()
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
    gen = build_generator(str(MODEL_DIR), mesh, num_layers=args.layers, prepare=False, decode_buckets=(1,))
    model = gen.model
    layer = model.layers[0]
    vocab = model.vocab_size
    results = []
    for rows, seq in shapes:
        g = torch.Generator().manual_seed(rows * 1000 + seq)
        ids = torch.randint(0, vocab, (1, rows * seq), generator=g, dtype=torch.int32)
        x_all = model._embed(ids)
        norm_all = layer._rms_norm(x_all, layer.weights.input_ln)
        h_norm_all = to_host(norm_all)
        norm_rows_equal = 0
        qkv_equal = {}
        candidates = {"table": None}
        role = "qkv"
        for name, choice in (
            ("1d_110_4", ("1d", (110, 4, 2, 2))),
            ("1d_110_8", ("1d", (110, 8, 2, 2))),
            ("2d_16", ("2d", 16)),
            ("2d_4", ("2d", 4)),
            ("minimal", ("minimal", 0)),
        ):
            candidates[name] = choice
        copy = layer._interleaved_copy[id(layer.weights.qkv)]
        weight = layer.weights.qkv

        def project(x, choice):
            r = x.shape[-2]
            if choice is None:
                return layer._linear(x, weight)
            kernel, cfg = choice
            try:
                if kernel == "1d":
                    return ttnn.linear(
                        x,
                        copy,
                        program_config=layer._prefill_1d_program_config(r, weight.shape[-1], *cfg),
                        dtype=layer.activation_dtype,
                        compute_kernel_config=layer._prefill_compute_config(copy),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                if kernel == "2d":
                    return ttnn.linear(
                        x,
                        copy,
                        program_config=layer._prefill_2d_program_config(r, weight.shape[-2], weight.shape[-1], cfg),
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
                return str(e).splitlines()[0][:200]

        solo_norm, solo_qkv = [], []
        for i in range(rows):
            x_i = model._embed(ids[:, i * seq : (i + 1) * seq])
            n_i = layer._rms_norm(x_i, layer.weights.input_ln)
            hn = to_host(n_i)
            solo_norm.append(hn)
            if torch.equal(hn, h_norm_all[i * seq : (i + 1) * seq]):
                norm_rows_equal += 1
            q = layer._linear(n_i, weight)
            solo_qkv.append(to_host(q))
            q.deallocate(True)
            n_i.deallocate(True)
            x_i.deallocate(True)
        timing = {}
        for name, choice in candidates.items():
            out = project(norm_all, choice)
            if isinstance(out, str):
                qkv_equal[name] = out
                continue
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(3):
                o2 = project(norm_all, choice)
                o2.deallocate(True)
            ttnn.synchronize_device(mesh)
            timing[name] = (time.perf_counter() - t0) / 3 * 1000
            h = to_host(out)
            out.deallocate(True)
            qkv_equal[name] = sum(torch.equal(h[i * seq : (i + 1) * seq], solo_qkv[i]) for i in range(rows))
        norm_all.deallocate(True)
        x_all.deallocate(True)
        rec = {
            "rows": rows,
            "seq": seq,
            "solo_kernel": layer.policy.prefill_kernel("qkv", seq, layer.weight_dtypes["qkv"]),
            "dense_kernel": layer.policy.prefill_kernel("qkv", rows * seq, layer.weight_dtypes["qkv"]),
            "norm_rows_bit_equal": f"{norm_rows_equal}/{rows}",
            "qkv_rows_bit_equal_by_candidate": qkv_equal,
            "candidate_ms": timing,
        }
        print(json.dumps(rec), flush=True)
        results.append(rec)
    Path(args.output).write_text(
        json.dumps(
            {"results": results, "layers": args.layers, "visible_devices": os.getenv("TT_VISIBLE_DEVICES")}, indent=2
        )
        + "\n"
    )
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

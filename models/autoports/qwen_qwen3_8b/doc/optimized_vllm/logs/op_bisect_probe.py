# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-op bisect of batched prefill against the solo path on one decoder layer.

usage: op_bisect_probe.py <output.json> [--shapes 16x256,32x128] [--layers 1]

Records every op output of the batched pass (dense ops as ``[rows * seq, W]``, head ops as ``[rows, ...]``)
and of each row's solo pass, then reports per op how many rows are bit-equal and the first op that differs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator
from models.autoports.qwen_qwen3_8b.tt import optimized_decoder as od

MODEL_DIR = Path("models/autoports/qwen_qwen3_8b")
RECORD: list[tuple[str, torch.Tensor]] = []


def rec(name, t):
    RECORD.append((name, ttnn.to_torch(t).clone()))
    return t


def install(layer):
    orig = {}
    for name in ("_rms_norm", "_prefill_matmul", "_prefill_matmul_residual", "_rope_rows", "_linear"):
        orig[name] = getattr(layer, name)
    layer._linear = lambda x, w: rec(f"linear_{layer._prefill_role[id(w)]}", orig["_linear"](x, w))
    layer._rms_norm = lambda x, w: rec("rms_norm", orig["_rms_norm"](x, w))
    layer._prefill_matmul = lambda x, w: rec(f"matmul_{layer._prefill_role[id(w)]}", orig["_prefill_matmul"](x, w))
    layer._prefill_matmul_residual = lambda x, w, r: rec(f"matmul_residual_{layer._prefill_role[id(w)]}", orig["_prefill_matmul_residual"](x, w, r))
    ops = {}
    for mod, name in ((ttnn.experimental, "nlp_create_qkv_heads"), (ttnn.experimental, "rotary_embedding_hf"), (ttnn.transformer, "scaled_dot_product_attention"), (ttnn.transformer, "chunked_scaled_dot_product_attention"), (ttnn.experimental, "nlp_concat_heads"), (ttnn, "mul")):
        ops[(mod, name)] = getattr(mod, name)

    def wrap(mod, name):
        fn = ops[(mod, name)]

        def inner(*a, **k):
            out = fn(*a, **k)
            if isinstance(out, tuple):
                for i, o in enumerate(out):
                    rec(f"{name}[{i}]", o)
            else:
                rec(name, out)
            return out

        setattr(mod, name, inner)

    for mod, name in ops:
        wrap(mod, name)
    return orig, ops


def row_view(name, t, rows, seq, i):
    if t.dim() == 4 and t.shape[0] == rows:
        return t[i]
    if t.dim() == 4 and t.shape[0] == 1 and t.shape[2] == rows * seq:
        return t[0, :, i * seq : (i + 1) * seq]
    if t.dim() == 4 and t.shape[0] == 1 and t.shape[2] == seq:
        return t[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--shapes", default="16x256,32x128")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
    gen = build_generator(str(MODEL_DIR), mesh, num_layers=args.layers, prepare=False, decode_buckets=(1,), num_blocks=512)
    model = gen.model
    layer = model.layers[0]
    orig_linear = layer._linear
    install(layer)
    results = []
    for shape in args.shapes.split(","):
        rows, seq = (int(v) for v in shape.split("x"))
        g = torch.Generator().manual_seed(rows * 7 + seq + 1000 * args.seed)
        prompts = [torch.randint(0, model.vocab_size, (seq,), generator=g).tolist() for _ in range(rows)]
        gen.reset()
        for i in range(rows):
            gen._ensure_blocks(i, seq)
        table = gen._host_page_table(list(range(rows)))
        tt_table = ttnn.from_torch(table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        RECORD.clear()
        hidden = model.prefill_hidden_batched(prompts, rows=rows, seq_len=seq, page_table=tt_table, kv_cache=gen.kv_cache)
        batched = list(RECORD)
        h_b = ttnn.to_torch(hidden)[0, 0]
        hidden.deallocate(True)
        tt_table.deallocate(True)
        per_op = {}
        final_equal = 0
        first_diff_rows = {}
        for i in range(rows):
            gen.reset()
            gen._ensure_blocks(0, seq)
            t1 = gen._host_page_table([0])
            tt1 = ttnn.from_torch(t1, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
            RECORD.clear()
            hs = model.prefill_hidden(prompts[i], start_pos=0, user_slot=0, page_table=tt1, kv_cache=gen.kv_cache, host_page_table=torch.cat([t1, torch.full((1, 1), gen.scratch_block, dtype=torch.int32)], dim=1))
            solo = list(RECORD)
            h_s = ttnn.to_torch(hs)[0, 0]
            hs.deallocate(True)
            tt1.deallocate(True)
            final_equal += int(torch.equal(h_s, h_b[i * seq : (i + 1) * seq]))
            bi = 0
            first = None
            for name_s, t_s in solo:
                while bi < len(batched) and batched[bi][0].replace("chunked_scaled_dot_product_attention", "scaled_dot_product_attention") != name_s.replace("chunked_scaled_dot_product_attention", "scaled_dot_product_attention"):
                    bi += 1
                if bi >= len(batched):
                    break
                name_b, t_b = batched[bi]
                bi += 1
                vb = row_view(name_b, t_b, rows, seq, i)
                vs = row_view(name_s, t_s, 1, seq, 0)
                key = f"{len(per_op) if name_s not in per_op else ''}{name_s}"
                if vb is None or vs is None or vb.shape != vs.shape:
                    per_op.setdefault(name_s, []).append(f"shape {None if vb is None else tuple(vb.shape)} vs {None if vs is None else tuple(vs.shape)}")
                    continue
                eq = bool(torch.equal(vb, vs))
                per_op.setdefault(name_s, []).append(eq)
                if not eq and first is None:
                    first = name_s
            first_diff_rows[i] = first
        summary = {name: (sum(1 for v in vals if v is True), len(vals)) for name, vals in per_op.items()}
        retry = {}
        if "linear_down_proj" in per_op and not all(v is True for v in per_op["linear_down_proj"]):
            mul_b = next(t for n, t in batched if n == "mul")
            down_b = next(t for n, t in batched if n == "linear_down_proj")
            weight = layer.weights.down_proj
            copy = layer._interleaved_copy[id(weight)]
            kk, nn = weight.shape[-2], weight.shape[-1]
            x_all = ttnn.from_torch(mul_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            solo_outs = {}
            for i in range(rows):
                if per_op["linear_down_proj"][i] is True:
                    continue
                x_i = ttnn.from_torch(mul_b[:, :, i * seq : (i + 1) * seq], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                o = orig_linear(x_i, weight)
                solo_outs[i] = ttnn.to_torch(o)[0, 0].clone()
                o.deallocate(True)
                x_i.deallocate(True)
                diff = (down_b[0, 0, i * seq : (i + 1) * seq].float() - solo_outs[i].float()).abs()
                retry[f"row{i}_recorded_batched_vs_solo"] = {"max_abs_diff": float(diff.max()), "n_diff": int((diff > 0).sum()), "argmax_pos": [int(v) for v in (diff == diff.max()).nonzero()[0]]}
            for name, choice in (("minimal", ("minimal", 0)), ("2d_4", ("2d", 4)), ("2d_2", ("2d", 2)), ("auto", ("auto", 0))):
                try:
                    kernel, cfg = choice
                    if kernel == "2d":
                        o = ttnn.linear(x_all, copy, program_config=layer._prefill_2d_program_config(rows * seq, kk, nn, cfg), dtype=layer.activation_dtype, compute_kernel_config=layer._prefill_compute_config(copy), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    elif kernel == "auto":
                        o = ttnn.linear(x_all, copy, dtype=layer.activation_dtype, compute_kernel_config=layer._prefill_compute_config(copy), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    else:
                        o = ttnn.experimental.minimal_matmul(x_all, copy, dtype=layer.activation_dtype, compute_kernel_config=layer._prefill_compute_config(copy), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                except Exception as e:
                    retry[name] = str(e).splitlines()[0][:120]
                    continue
                h = ttnn.to_torch(o)[0, 0]
                o.deallocate(True)
                retry[name] = {f"row{i}_bit_equal_solo": bool(torch.equal(h[i * seq : (i + 1) * seq], solo_outs[i])) for i in solo_outs}
                retry[name]["bit_equal_recorded_batched"] = bool(torch.equal(h, down_b[0, 0]))
            x_all.deallocate(True)
        rec_out = {"rows": rows, "seq": seq, "final_hidden_rows_bit_equal": f"{final_equal}/{rows}", "per_op_rows_bit_equal": summary, "first_differing_op_per_row": first_diff_rows, "down_retry": retry, "batched_op_sequence": [n for n, _ in batched]}
        print(json.dumps(rec_out), flush=True)
        results.append(rec_out)
    Path(args.output).write_text(json.dumps({"results": results, "visible_devices": os.getenv("TT_VISIBLE_DEVICES")}, indent=2) + "\n")
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

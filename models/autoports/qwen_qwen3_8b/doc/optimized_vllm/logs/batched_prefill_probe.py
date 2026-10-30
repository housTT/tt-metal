# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batched-prefill calibration and batched-equals-solo probe at the generator level.

usage: batched_prefill_probe.py <output.json> [--layers N] [--pairs 32x128,8x512,...] [--reps 3] [--kv-layers 0,-1]

For every (rows, seq_len) batched shape: the solo arm prefills each row alone, the serial arm sends every
row in one call with batched prefill off (the per-row loop the plugin sees without this stage), the batched
arm sends the same call with batched prefill on. It records wall time per arm (median of --reps), last-token
logits agreement per row (bit-equal, top-1, PCC), K/V cache pages at the real positions of the listed layers
(zero tolerance) and the program-cache entry count around the batched call.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator
from models.autoports.qwen_qwen3_8b.tt.model import batched_prefill_pairs

MODEL_DIR = Path("models/autoports/qwen_qwen3_8b")


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float() - a.float().mean()
    b = b.flatten().float() - b.float().mean()
    d = float(a.norm() * b.norm())
    return 1.0 if d == 0 else float(a @ b) / d


def lengths_for(rows: int, seq: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    lo = seq // 2 + 1 if seq > 32 else max(1, seq - 31)
    out = [seq, max(1, seq - 1)]
    while len(out) < rows:
        out.append(int(torch.randint(lo, seq + 1, (1,), generator=g)))
    return out[:rows]


def random_prompts(lengths: list[int], vocab: int, seed: int) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (n,), generator=g).tolist() for n in lengths]


def cache_rows(gen, layer: int, slot: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    key_cache, value_cache = gen.kv_cache[layer]
    blocks = gen.slot_blocks[slot]
    k = ttnn.to_torch(key_cache)
    v = ttnn.to_torch(value_cache)
    k_rows = torch.cat([k[b] for b in blocks], dim=1)[:, :length]
    v_rows = torch.cat([v[b] for b in blocks], dim=1)[:, :length]
    return k_rows.clone(), v_rows.clone()


def timed_call(gen, fn, reps: int) -> float:
    times = []
    for _ in range(reps):
        ttnn.synchronize_device(gen.mesh_device)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(gen.mesh_device)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--layers", type=int, default=int(os.getenv("QWEN3_FULL_MODEL_LAYERS", "36")))
    parser.add_argument("--pairs", default="")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--kv-layers", default="0,-1")
    args = parser.parse_args()
    pairs = (
        [tuple(int(v) for v in p.split("x")) for p in args.pairs.split(",")]
        if args.pairs
        else list(batched_prefill_pairs())
    )
    kv_layers = [int(v) for v in args.kv_layers.split(",")]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
    t0 = time.perf_counter()
    gen = build_generator(str(MODEL_DIR), mesh, num_layers=args.layers, decode_buckets=(1, 8, 16, 24, 32))
    build_s = time.perf_counter() - t0
    layers = [l % gen.model.num_layers for l in kv_layers]
    vocab = gen.model.vocab_size
    results = []
    program_cache_start = mesh.num_program_cache_entries()
    for pair_index, (rows, seq) in enumerate(pairs):
        lengths = lengths_for(rows, seq, 1000 + pair_index)
        prompts = random_prompts(lengths, vocab, 2000 + pair_index)
        solo_logits, solo_kv, solo_ms = [], [], 0.0
        for i, ids in enumerate(prompts):
            gen.reset()
            gen._ensure_blocks(0, len(ids))
            table = gen._host_page_table([0])
            call = lambda: gen.prefill_forward(
                torch.tensor([ids]), page_table=table, kv_cache=gen.kv_cache, prompt_lens=[len(ids)]
            )
            out = call()
            solo_logits.append(out[0, 0].clone())
            solo_kv.append({l: cache_rows(gen, l, 0, len(ids)) for l in layers})
            solo_ms += timed_call(gen, call, args.reps)
        width = max(lengths)
        tokens = torch.zeros(rows, width, dtype=torch.int64)
        for i, ids in enumerate(prompts):
            tokens[i, : len(ids)] = torch.tensor(ids)

        def multi_call():
            return gen.prefill_forward(tokens, page_table=table_multi, kv_cache=gen.kv_cache, prompt_lens=lengths)

        gen.reset()
        for i, n in enumerate(lengths):
            gen._ensure_blocks(i, n)
        table_multi = gen._host_page_table(list(range(rows)))
        gen.batched_prefill = False
        serial_out = multi_call()
        serial_ms = timed_call(gen, multi_call, args.reps)
        gen.batched_prefill = True
        before = gen.counter_snapshot()
        pc_before = mesh.num_program_cache_entries()
        batched_out = multi_call()
        pc_after = mesh.num_program_cache_entries()
        after = gen.counter_snapshot()
        batched_ms = timed_call(gen, multi_call, args.reps)
        batched_kv = [{l: cache_rows(gen, l, i, lengths[i]) for l in layers} for i in range(rows)]
        order = list(reversed(range(rows)))
        gen.reset()
        for i, n in enumerate(lengths):
            gen._ensure_blocks(i, n)
        table_rev = gen._host_page_table(list(range(rows)))
        reordered = gen.prefill_forward(
            tokens[order], page_table=table_rev, kv_cache=gen.kv_cache, prompt_lens=[lengths[i] for i in order]
        )
        per_row = []
        for i in range(rows):
            s, b, sr = solo_logits[i], batched_out[i, 0], serial_out[i, 0]
            per_row.append(
                {
                    "length": lengths[i],
                    "logits_bit_equal": bool(torch.equal(s, b)),
                    "serial_bit_equal_solo": bool(torch.equal(s, sr)),
                    "reordered_bit_equal": bool(torch.equal(b, reordered[order.index(i), 0])),
                    "top1_equal": int(s.argmax()) == int(b.argmax()),
                    "pcc": pcc(s, b),
                    "max_abs_diff": float((s - b).abs().max()),
                    "kv_bit_equal": {
                        str(l): bool(torch.equal(solo_kv[i][l][0], batched_kv[i][l][0]))
                        and bool(torch.equal(solo_kv[i][l][1], batched_kv[i][l][1]))
                        for l in layers
                    },
                    "kv_max_abs_diff": {
                        str(l): max(
                            float((solo_kv[i][l][0].float() - batched_kv[i][l][0].float()).abs().max()),
                            float((solo_kv[i][l][1].float() - batched_kv[i][l][1].float()).abs().max()),
                        )
                        for l in layers
                    },
                }
            )
        record = {
            "rows": rows,
            "seq_len": seq,
            "lengths": lengths,
            "solo_sum_ms": solo_ms,
            "serial_call_ms": serial_ms,
            "batched_call_ms": batched_ms,
            "speedup_vs_serial_call": serial_ms / batched_ms if batched_ms else None,
            "batched_wins": batched_ms < serial_ms,
            "program_cache_before": pc_before,
            "program_cache_after": pc_after,
            "counters_delta": {
                k: after[k] - before[k]
                for k in ("batched_prefill_groups", "batched_prefill_rows", "single_row_prefill_rows")
            },
            "rows_detail": per_row,
            "all_logits_bit_equal": all(r["logits_bit_equal"] for r in per_row),
            "all_top1_equal": all(r["top1_equal"] for r in per_row),
            "pcc_min": min(r["pcc"] for r in per_row),
            "all_kv_bit_equal": all(all(r["kv_bit_equal"].values()) for r in per_row),
            "all_reordered_bit_equal": all(r["reordered_bit_equal"] for r in per_row),
        }
        results.append(record)
        print(
            f"pair {rows}x{seq}: solo {solo_ms:.1f} ms, serial call {serial_ms:.1f} ms, batched {batched_ms:.1f} ms "
            f"(x{record['speedup_vs_serial_call']:.2f}); logits bit-equal {record['all_logits_bit_equal']}, top1 {record['all_top1_equal']}, "
            f"pcc_min {record['pcc_min']:.6f}, kv bit-equal {record['all_kv_bit_equal']}, program cache {pc_before}->{pc_after}",
            flush=True,
        )
    payload = {
        "pairs": results,
        "layers": gen.model.num_layers,
        "kv_layers_checked": layers,
        "build_s": build_s,
        "program_cache_start": program_cache_start,
        "program_cache_end": mesh.num_program_cache_entries(),
        "batched_prefill_shapes": list(gen.batched_prefill_shapes),
        "provenance": {
            "checkout_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "visible_devices": os.getenv("TT_VISIBLE_DEVICES"),
            "command": " ".join(os.sys.argv),
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "precision_config_id": gen.precision_config.get("config_id"),
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    gen.teardown()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
